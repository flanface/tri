"""tri: reconstructs Deepwoken's talent / mantra selection rulebase from builds and expert reviews.

Names are imported lazily (PEP 562): `import tri.core.types` or `from tri.game.gamedata
import name_key` stays light (standard library only), so the Discord bot can
use the data model without the training stack (scipy, scikit-learn). `from tri
import Tri` loads the engine on first use."""
from __future__ import annotations

import importlib

_EXPORTS = {
    "analysis.evaluate": ("certification", "completion_benchmark", "evaluate", "group_split", "mapping_benchmark",
                          "review_benchmark", "rule_benchmark", "strictness_benchmark"),
    "core.catalog": ("Catalog",),
    "core.codec": ("validate_review",),
    "core.types": ("Action", "Build", "Category", "Item", "Kind", "Operation", "Reason", "Review", "Strength"),
    "engine.defs": ("TriConfig",),
    "engine.facade": ("Tri",),
    "game.defs": ("Patch",),
    "game.patch": ("PatchLog", "diff_catalogs", "patch_from_diff"),
    "infer.active": ("ReviewQueue",),
    "infer.defs": ("Alternative", "Change", "InferConfig"),
    "infer.engine": ("PassEngine",),
    "model.defs": ("FieldPoint", "SpaceConfig", "WorthConfig"),
    "model.matrix": ("BuildMatrix", "context_tokens"),
    "model.space": ("ConceptSpace",),
    "model.worth": ("FieldWorth",),
    "reviews.authority": ("ReviewLedger",),
    "reviews.defs": ("LedgerConfig", "Locator", "LocatorConfig"),
    "reviews.locators": ("LocatorSet", "new_locator", "archetype_key"),
    "rules.defs": ("Condition", "PassConfig"),
    "rules.induction": ("Rule", "RuleInducer", "context_features"),
    "rules.passes": ("PassRule", "Rulebook"),
    "sources.defs": ("PipelineConfig",),
    "sources.pipeline": ("PretrainSet", "prepare_pretraining"),
    "version": ("VERSION",),
}
_WHERE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = sorted(_WHERE) + ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        return importlib.import_module(".version", __name__).VERSION
    mod = _WHERE.get(name)
    if mod is not None:
        value = getattr(importlib.import_module("." + mod, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tri' has no attribute {name!r}")


def __dir__():
    return __all__
