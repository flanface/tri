"""Definitions of the facade: VERSION, the category names and TriConfig."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


from tri.analysis.defs import SidegradeConfig
from tri.core.types import Build
from tri.infer.caution import CautionConfig
from tri.infer.defs import InferConfig
from tri.model.defs import SpaceConfig, WorthConfig
from tri.model.matrix import CONTEXT_FIELDS
from tri.reviews.defs import LedgerConfig, LocatorConfig
from tri.rules.defs import PassConfig
from tri.rules.passes import CENTRAL, FOUNDATIONAL, LUXURY
from tri.rules.selection import SelectionConfig

CATEGORIES = (FOUNDATIONAL, CENTRAL, LUXURY)


def _build_key(b: Build) -> tuple:
    """Exact, hashable identity of a build's content (memo key)."""
    d = lambda x: tuple(sorted(x.items())) if x else None
    return (d(b.highest_stats), d(b.final_stats), d(b.pre_stats), tuple(b.talents), tuple(b.mantras),
            b.build_id, b.ruleset, b.timestamp, repr(sorted((b.meta or {}).items(), key=lambda kv: str(kv[0]))),
            tuple(d(s) for s in (getattr(b, "mid_stages", None) or [])))


@dataclass
class TriConfig:
    """All knobs in one place (each field says what it does)."""
    space: SpaceConfig = field(default_factory=SpaceConfig)
    passes: PassConfig = field(default_factory=PassConfig)
    worth: WorthConfig = field(default_factory=WorthConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    locators: LocatorConfig = field(default_factory=LocatorConfig)
    sidegrades: SidegradeConfig = field(default_factory=SidegradeConfig)
    context_fields: Tuple[str, ...] = CONTEXT_FIELDS
    data_size: Optional[int] = 300_000      # corpus builds kept (uniform sample; None = all)
    rule_min_support: float = 0.5           # review rules: one weak op is enough
    refit_review_weight: float = 10.0       # refit: one corrected build counts as this many corpus builds
    refit_review_share: float = 0.25        # ...but all reviews together at most this share of the total weight
    calibrate_min: int = 20                 # refit() calibrates swap_margin once this many reviews exist
    swap_grid: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 1.01)
    no_fit_grid: Tuple[float, ...] = (0.02, 0.05, 0.08, 0.12, 0.18)
    seed: int = 0
    selection: SelectionConfig = field(default_factory=SelectionConfig)   # v3.0: review evidence on rules
    caution: CautionConfig = field(default_factory=CautionConfig)         # v3.1: hold back or ask

    def to_dict(self) -> dict:
        d = asdict(self)
        d["space"]["stat_grid"] = list(d["space"]["stat_grid"])
        d["context_fields"] = list(d["context_fields"])
        d["swap_grid"] = list(d["swap_grid"])
        d["no_fit_grid"] = list(d["no_fit_grid"])
        d["locators"]["metric_grid"] = list(d["locators"]["metric_grid"])
        d["sidegrades"]["contested"] = list(d["sidegrades"]["contested"])
        return d

    @staticmethod
    def from_dict(d: dict) -> "TriConfig":
        sc = dict(d["space"]); sc["stat_grid"] = tuple(sc["stat_grid"])
        sc = {k: v for k, v in sc.items() if k in SpaceConfig.__dataclass_fields__}
        lc = dict(d.get("locators", {}))
        if "metric_grid" in lc:
            lc["metric_grid"] = tuple(lc["metric_grid"])
        lc = {k: v for k, v in lc.items() if k in LocatorConfig.__dataclass_fields__}   # v2.15 knobs retired in v2.16
        gc = dict(d.get("sidegrades", {}))
        if "contested" in gc:
            gc["contested"] = tuple(gc["contested"])
        ic = {k: v for k, v in d["infer"].items() if k in InferConfig.__dataclass_fields__}
        return TriConfig(SpaceConfig(**sc), PassConfig(**{k: v for k, v in d["passes"].items() if k in PassConfig.__dataclass_fields__}), WorthConfig(**d["worth"]),
                         InferConfig(**ic), LedgerConfig(**{k: v for k, v in d["ledger"].items() if k in LedgerConfig.__dataclass_fields__}), LocatorConfig(**lc),
                         SidegradeConfig(**gc), tuple(d["context_fields"]),
                         d.get("data_size"), d.get("rule_min_support", 0.5), d.get("refit_review_weight", 10.0),
                         d.get("refit_review_share", 0.25), d.get("calibrate_min", 20),
                         tuple(d.get("swap_grid", (0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 1.01))),
                         tuple(d.get("no_fit_grid", (0.02, 0.05, 0.08, 0.12, 0.18))), d.get("seed", 0),
                         SelectionConfig(**{k: v for k, v in (d.get("selection") or {}).items()
                                            if k in SelectionConfig.__dataclass_fields__}),
                         CautionConfig(**{k: v for k, v in (d.get("caution") or {}).items()
                                          if k in CautionConfig.__dataclass_fields__}))
