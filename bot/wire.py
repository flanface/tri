"""Result types shared by the model service and the bot, the .env loader, and
the catalog helpers that need only a catalog (name resolution, search,
requirement text, addable items). Standard library and tri.core only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from tri.core.codec import build_from_dict, build_to_dict
from tri.core.types import Build, Kind
from tri.game.gamedata import format_reqs, name_key

REASON_LABELS: Dict[str, str] = {
    "less_necessary_space": "less necessary (make space)",
    "necessary_for_archetype": "necessary for the archetype",
    "redundant_with": "redundant with",
    "works_against": "works against",
    "no_synergy": "no synergy with the build",
    "scenario_mismatch": "does not fit the scenario (strong: does not belong on this archetype)",
    "broken": "does not work",
    "generally_good": "generally good",
    "synergy_with": "synergises with",
    "needs": "does not work without",
    "luxury_less": "worth less as a luxury",
    "luxury_more": "worth more as a luxury",
    "sidegrade": "sidegrade to",
}


@dataclass
class ParsedBuild:
    build: Build                       # catalog items only; granted talents removed
    name: str
    author: Optional[str]
    granted: List[str]                 # granted talents removed while parsing
    unknown: List[str]                 # not in the model's catalog (kept in raw)
    unmet: List[str]                   # known requirements not met at the pre-shrine or final stats
    raw_talents: List[str] = field(default_factory=list)   # after granted removal, before catalog filter
    raw_mantras: List[str] = field(default_factory=list)

    unknown_why: Dict[str, str] = field(default_factory=dict)   # item -> short reason (why_missing)
    record: Dict = field(default_factory=dict)                  # the planner record (dwt.sh: raw build under "dwt")

    def notes(self) -> List[str]:
        out = []
        if self.granted:
            out.append(f"{len(self.granted)} granted talents ignored")
        if self.unknown:
            groups: Dict[str, List[str]] = {}
            for x in self.unknown:
                groups.setdefault(self.unknown_why.get(x, "not in the model's catalog"), []).append(x)
            for why, xs in groups.items():
                out.append(f"{why}: " + ", ".join(xs[:15]) + (" ..." if len(xs) > 15 else ""))
        if self.unmet:
            at = "any stage" if self.build.mid_stages else "the pre-shrine or final stats"
            out.append(f"requirements not met at {at}: " + ", ".join(self.unmet[:10]))
        return out

    def to_dict(self) -> dict:
        return dict(build=build_to_dict(self.build), name=self.name, author=self.author, granted=list(self.granted),
                    unknown=list(self.unknown), unmet=list(self.unmet), raw_talents=list(self.raw_talents),
                    raw_mantras=list(self.raw_mantras), unknown_why=dict(self.unknown_why))

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedBuild":
        return cls(build_from_dict(d["build"]), d.get("name") or "", d.get("author"), list(d.get("granted") or []),
                   list(d.get("unknown") or []), list(d.get("unmet") or []), list(d.get("raw_talents") or []),
                   list(d.get("raw_mantras") or []), dict(d.get("unknown_why") or {}))


@dataclass
class LocatorInfo:
    """A locator as the bot sees it: id, archetype, link and the parsed build."""
    locator_id: str
    archetype: str
    link: Optional[str] = None
    added_by: Optional[str] = None
    build: Optional[Build] = None

    def to_dict(self) -> dict:
        return dict(locator_id=self.locator_id, archetype=self.archetype, link=self.link, added_by=self.added_by,
                    build=build_to_dict(self.build) if self.build is not None else None)

    @classmethod
    def from_dict(cls, d: dict) -> "LocatorInfo":
        return cls(str(d["locator_id"]), d.get("archetype") or "", d.get("link"), d.get("added_by"),
                   build_from_dict(d["build"]) if d.get("build") else None)

    @classmethod
    def of(cls, loc, with_build: bool = False) -> "LocatorInfo":
        """From a tri.locators.Locator."""
        return cls(str(loc.locator_id), loc.archetype, getattr(loc, "link", None), getattr(loc, "added_by", None),
                   loc.build if with_build else None)


@dataclass
class PlanView:
    """A game data sync plan as the bot sees it (tri.sync.SyncPlan without the
    catalog). plan_id names the full plan held by the service until applied."""
    plan_id: Optional[str]
    sha: Optional[str]
    first: bool
    added: List[str]
    changed: List[str]
    removed: List[str]
    renames: Dict[str, str]
    relocked: List[str]

    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed or self.relocked)

    def summary(self) -> dict:
        return dict(sha=self.sha, first=self.first, added=self.added, changed=self.changed,
                    removed=self.removed, renames=self.renames, relocked=self.relocked)

    @classmethod
    def from_summary(cls, s: dict, plan_id: Optional[str] = None) -> "PlanView":
        return cls(plan_id, s.get("sha"), bool(s.get("first")), list(s.get("added") or []),
                   list(s.get("changed") or []), list(s.get("removed") or []), dict(s.get("renames") or {}),
                   list(s.get("relocked") or []))


def load_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE loader; existing environment variables win."""
    import os
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip().removeprefix("export ").strip(), v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            os.environ.setdefault(k, v)


# ---------------------------------------------------------------- catalog
class CatalogHelpers:
    """Methods that need only the catalog snapshot and the run's name tables.
    Mixed into ModelService; answered without a model call. Expects: catalog, canonical_names,
    _names, excluded, item_counts, min_item_count."""

    catalog: Any
    canonical_names: Dict[str, str]
    _names: List[str]
    excluded: Dict[str, Any]
    item_counts: Dict[str, dict]
    min_item_count: Optional[int]

    def _index_names(self) -> None:
        self._names = sorted(self.catalog.ids, key=str.casefold)
        self._count_keys = {name_key(n): n for n in self.item_counts}
        self._excl_keys = {name_key(n): n for n in self.excluded}

    def kind(self, item: str) -> Optional[str]:
        it = self.catalog.items.get(item)
        return it.kind.value if it else None

    def is_talent(self, item: str) -> bool:
        it = self.catalog.items.get(item)
        return bool(it and it.kind == Kind.TALENT)

    def requirement_text(self, item: str) -> str:
        it = self.catalog.items.get(item)
        if it is None:
            return "?"
        parts = [f"{v} {k}" for k, v in it.requires.items()]
        if it.requires_any:
            parts.append(" or ".join("+".join(f"{v} {k}" for k, v in r.items()) for r in it.requires_any))
        if it.clauses:
            parts.append(format_reqs(it.clauses))
        ctx = self.catalog.context_text(item)
        if ctx:
            parts.append(ctx)
        return ", ".join(p for p in parts if p) or "none"

    def resolve(self, name: str) -> Optional[str]:
        """Exact catalog id, or a case / spacing / punctuation-insensitive match."""
        if name in self.catalog.index:
            return name
        k = name_key(name)
        for kind in ("talent", "mantra"):
            hit = self.canonical_names.get(f"{kind}:{k}")
            if hit in self.catalog.index:
                return hit
        for n in self._names:
            if name_key(n) == k:
                return n
        return None

    @staticmethod
    def search(query: str, pool: Sequence[str], limit: int = 25) -> List[str]:
        """Names from `pool` containing `query`, ignoring case, spacing and
        punctuation ("knights" finds "Knight's Rally"); prefix matches first."""
        q = query.casefold().strip()
        if not q:
            return sorted(pool, key=str.casefold)[:limit]
        k = name_key(q)
        pre, sub = [], []
        for n in pool:
            c, nk = n.casefold(), name_key(n)
            if c.startswith(q) or (k and nk.startswith(k)):
                pre.append(n)
            elif q in c or (k and k in nk):
                sub.append(n)
        return (sorted(pre, key=str.casefold) + sorted(sub, key=str.casefold))[:limit]

    def addable(self, b: Build, exclude: Sequence[str] = ()) -> List[str]:
        """Catalog items not in the build, available in it (pre-shrine or final
        stats, and its oath / murmur / origin)."""
        have = set(b.items()) | set(exclude)
        mask = self.catalog.availability_in(b)
        return [n for n, ok in zip(self.catalog.ids, mask) if ok and n not in have]

    def why_missing(self, name: str) -> Optional[str]:
        """Why a name is not in the catalog (None if it is). Short reason."""
        if self.resolve(name):
            return None
        k = name_key(name)
        ex = self._excl_keys.get(k)
        if ex is not None:
            v = self.excluded[ex]
            grp = v.get("group") if isinstance(v, dict) else None
            return f"granted, not a slot pick ({grp})" if grp else "excluded as granted"
        cnt = self._count_keys.get(k)
        if cnt is not None:
            c = self.item_counts[cnt]
            if c.get("in_gamedata") is False:
                return "not in the game data"
            if self.min_item_count and c.get("count", 0) < self.min_item_count:
                return f"too rare in the training builds (< {self.min_item_count} picks)"
        if self.item_counts:
            return "never seen in the training builds"
        return "not in the model's catalog"
