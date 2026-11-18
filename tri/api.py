"""The public API of tri 4.0. Everything outside the library (the bot, scripts,
notebooks) imports from here and nowhere else in tri.

tri takes its inputs as arguments: a catalog, reviews, locators, authored
rules and patches. It never reads environment variables, never opens a
network connection and never talks to Supabase, Discord or dwt.sh; those are
bot-side adapters. It reads and writes local run files only.

Typed calls (results are JSON-serialisable: dicts, lists, str, float, int,
bool, None):

    tri = load_run("runs/live")                       # Tri, from <run>/model + <run>/catalog.json
    tri = with_memory(tri, reviews, locators)          # a new Tri; the old one is untouched
    review_build(tri, build, n=3)        -> [answer]   # smallest edits, one per archetype
    complete_build(tri, build)           -> answer     # fill to the talent cap and mantra slots
    dry_insert(tri, review, probes=())   -> dict       # what one review would change (non-mutating)
    agreement(tri)                       -> dict       # each review reproduced on its own build
    evaluate(tri, workers=2)             -> dict       # leave-one-build-out F1 with bootstrap SEs

Objects and helpers re-exported for callers that need them (types, codecs,
views, rule language, game data readers) are listed in __all__. Names are
resolved lazily, so `from tri.api import Build` loads the data model only.
"""
from __future__ import annotations

import dataclasses
import enum
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

# name -> (module, attribute); attribute None = the module itself
_EXPORTS: Dict[str, tuple] = {
    # model and data types
    "Tri": ("tri.engine.facade", "Tri"), "TriConfig": ("tri.engine.defs", "TriConfig"),
    "Catalog": ("tri.core.catalog", "Catalog"),
    **{n: ("tri.core.types", n) for n in (
        "ADD_REASONS", "Action", "Build", "IFFY_REASONS", "Item", "Kind", "Operation", "PRESENT_CONDITION",
        "RELATIONAL_REASONS", "REMOVE_ONLY_REASONS", "Reason", "Review", "ReviewError", "Strength", "Category")},
    "Locator": ("tri.reviews.defs", "Locator"), "Patch": ("tri.game.defs", "Patch"),
    "Alternative": ("tri.infer.defs", "Alternative"), "Change": ("tri.infer.defs", "Change"),
    "Answers": ("tri.infer.caution", "Answers"),
    # codecs
    **{n: ("tri.core.codec", n) for n in (
        "build_from_dict", "build_to_dict", "exclusion_message", "is_silentheart", "legacy_names", "prereq_text",
        "review_from_dict", "review_to_dict", "validate_review")},
    "wield_waived": ("tri.core.context", "wield_waived"),
    # game data (files only; downloads are bot-side)
    **{n: ("tri.game.gamedata", n) for n in (
        "GameData", "format_reqs", "name_key", "load_gamedata", "load_deepwokenco", "load_any", "merge_gamedata",
        "cached_gamedata", "cached_deepwokenco", "resolve_gamedata",
        "API_URL", "DWCO_URL", "REPO", "ZIP_SHA_URL", "ZIP_URL")},
    "equipped": ("tri.game.weapons", "equipped"), "is_weapon_token": ("tri.game.weapons", "is_weapon_token"),
    "weapon_label": ("tri.game.weapons", "label"), "resolve_weapon": ("tri.game.weapons", "resolve"),
    "weapon_token": ("tri.game.weapons", "weapon_token"),
    "sync": ("tri.game.sync", None),
    # archetypes, reviews, rules
    "archetype_key": ("tri.reviews.locators", "archetype_key"),
    "family_key": ("tri.reviews.variants", "family_key"), "split_archetype": ("tri.reviews.variants", "split_archetype"),
    "AuthoredRules": ("tri.rules.authored", "AuthoredRules"),
    "RuleSyntaxError": ("tri.rules.language", "RuleSyntaxError"),
    "statements": ("tri.rules.statements", None),
    "FACT_SYMBOLS": ("tri.rules.statements", "FACT_SYMBOLS"), "RULE_SYMBOLS": ("tri.rules.statements", "RULE_SYMBOLS"),
    "change_margin": ("tri.infer.draft", "change_margin"), "norm_class": ("tri.infer.focus", "norm_class"),
    "active": ("tri.infer.active", None),
    # records and builder formats (pure parsers; no HTTP)
    "dwt": ("tri.sources.dwt", None), "parse_drive_link": ("tri.sources.dwt", "parse_drive_link"),
    "PipelineConfig": ("tri.sources.defs", "PipelineConfig"), "parse_record": ("tri.sources.pipeline", "parse_record"),
    "freeze_record": ("tri.sources.record", "freeze"),
    # analysis and views
    "categorize": ("tri.analysis.categorize", None), "facts": ("tri.analysis.facts", None),
    "FactBook": ("tri.analysis.facts", "FactBook"), "stream": ("tri.analysis.stream", None),
    "figure": ("tri.views.figure", None), "plots": ("tri.views.plots", None), "insight": ("tri.views.insight", None),
    "export_depth": ("tri.views.depth", "export"), "interactive_page": ("tri.views.interactive", "page"),
    # process helpers, evaluation
    "accel": ("tri.model.accel", None), "release": ("tri.runtime", "release"), "cpu_quota": ("tri.runtime", "cpu_quota"),
    "evaluation": ("tri.evaluation", None),
    "VERSION": ("tri.version", "VERSION"),
}

_CALLS = ("load_run", "with_memory", "with_reviews", "review_build", "complete_build", "dry_insert", "agreement",
          "evaluate", "to_json")
__all__ = sorted(_EXPORTS) + list(_CALLS)

if TYPE_CHECKING:                                     # static view of the lazy names
    from tri.core.catalog import Catalog
    from tri.core.types import Build, Review
    from tri.engine.facade import Tri
    from tri.game.defs import Patch
    from tri.reviews.defs import Locator


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'tri.api' has no attribute {name!r}")
    mod = importlib.import_module(spec[0])
    value = mod if spec[1] is None else getattr(mod, spec[1])
    globals()[name] = value
    return value


def __dir__():
    return __all__


# ------------------------------------------------------------------ JSON
def to_json(x: Any) -> Any:
    """Dataclasses, enums, numpy scalars / arrays, sets and tuples -> JSON values."""
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    if isinstance(x, enum.Enum):
        return to_json(x.value)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_json(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        v = [to_json(y) for y in x]
        return sorted(v, key=str) if isinstance(x, (set, frozenset)) else v
    tolist = getattr(x, "tolist", None)
    if callable(tolist):                                # numpy
        return to_json(tolist())
    if hasattr(x, "__dict__"):
        return {k: to_json(v) for k, v in vars(x).items() if not k.startswith("_")}
    return str(x)


def _answer(a) -> Dict[str, Any]:
    d = to_json(a)
    d.pop("memory", None)                              # internal claim tuples
    d["items"] = list(a.talents) + list(a.mantras)
    return d


# ------------------------------------------------------------------ calls
def load_run(run_dir: str, catalog: Optional["Catalog"] = None) -> "Tri":
    """A fitted run: <run>/model with <run>/catalog.json (or the catalog given)."""
    import os
    from tri.core.catalog import Catalog as _C
    from tri.engine.facade import Tri as _T
    cat = catalog if catalog is not None else _C.from_json(os.path.join(run_dir, "catalog.json"))
    return _T.load(os.path.join(run_dir, "model"), cat)


def with_memory(tri: "Tri", reviews: Sequence["Review"], locators: Sequence["Locator"]) -> "Tri":
    """A copy of tri with this review log and these locators; tri is not changed."""
    return tri.with_memory(reviews, locators)


def with_reviews(tri: "Tri", reviews: Sequence["Review"]) -> "Tri":
    """A copy of tri with this review log; tri is not changed."""
    return tri.with_reviews(reviews)


def review_build(tri: "Tri", build: "Build", n: int = 3, archetype: Optional[str] = None,
                 locked: Sequence[str] = (), answers: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Smallest edits to a full build, best archetype first (JSON)."""
    return [_answer(a) for a in tri.review(build, n, locked=locked, archetype=archetype, answers=answers)]


def complete_build(tri: "Tri", build: "Build", archetype: Optional[str] = None,
                   answers: Optional[dict] = None) -> Dict[str, Any]:
    """Fill a partial build to the talent cap and mantra slots (JSON)."""
    return _answer(tri.complete(build, archetype=archetype, answers=answers))


def dry_insert(tri: "Tri", review: "Review", probes: Sequence["Build"] = ()) -> Dict[str, Any]:
    """What one review would change; tri is not changed (safe beside other calls)."""
    return to_json(tri.dry_insert(review, probes))


def agreement(tri: "Tri") -> Dict[str, Any]:
    a, rows = tri.review_agreement(detail=True)
    return dict(review_agreement=float(a), rows=to_json(rows))


def evaluate(tri: "Tri", workers: int = 2, B: int = 1000) -> Dict[str, Any]:
    """Leave-one-build-out operation F1 and review agreement on tri's own
    reviews, each {value, se} (tri.evaluation)."""
    from tri import evaluation as E
    rows = E.lobo(tri, workers=workers)
    ag = E.agreement(tri)
    return dict(lobo=E.bootstrap(rows, B=B), agreement=E.bootstrap(ag, E.agreement_metrics, B=B), units=len(rows))
