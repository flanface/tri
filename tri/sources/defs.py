"""Definitions of data sources: pipeline configuration, parsed records, frozen
review records and marks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tri.core.types import DEFAULT_STATS, Build, Review

# ============================================================ configuration
#: planner stat name -> catalog stat name
STAT_ALIASES: Dict[str, str] = {
    "Light Wep.": "LightWeapon", "Medium Wep.": "MediumWeapon", "Heavy Wep.": "HeavyWeapon",
    "Strength": "Strength", "Fortitude": "Fortitude", "Agility": "Agility",
    "Intelligence": "Intelligence", "Willpower": "Willpower", "Charisma": "Charisma",
    "Flamecharm": "Flamecharm", "Frostdraw": "Frostdraw", "Thundercall": "Thundercall",
    "Galebreathe": "Galebreathe", "Shadowcast": "Shadowcast", "Ironsing": "Ironsing",
    "Bloodrend": "Bloodrend",
}

#: name -> source group, for every talent excluded by name
GRANTED_TALENTS: Dict[str, str] = {}

#: name prefixes / suffixes that mark granted talents
GRANTED_PREFIXES: Dict[str, str] = {"Oath: ": "oath", "Murmur: ": "murmur", "Resonance: ": "resonance",
                                    "Bell: ": "resonance"}

GRANTED_SUFFIXES: Dict[str, str] = {" Unbounded": "mastery"}


@dataclass
class PipelineConfig:
    """Every pipeline knob. Defaults are starting values."""
    # --- parsing
    stats: Tuple[str, ...] = DEFAULT_STATS
    stat_aliases: Dict[str, str] = field(default_factory=lambda: dict(STAT_ALIASES))
    item_aliases: Dict[str, str] = field(default_factory=dict)   # old name -> current name
    exclude_items: Tuple[str, ...] = ()                           # dropped from every build
    exclude_prefixes: Tuple[str, ...] = ()                        # e.g. ("Oath: ",)
    mantra_fallback: bool = True       # use content.mantraModifications keys if "mantras" missing
    accept_versions: Optional[Tuple[int, ...]] = None   # None = any version
    exclude_granted: bool = True       # drop slot-free talents (see GRANTED_TALENTS)
    granted_talents: Dict[str, str] = field(default_factory=lambda: dict(GRANTED_TALENTS))
    granted_prefixes: Dict[str, str] = field(default_factory=lambda: dict(GRANTED_PREFIXES))
    granted_suffixes: Dict[str, str] = field(default_factory=lambda: dict(GRANTED_SUFFIXES))
    keep_meta: bool = True             # store Build.meta codes (Origin, Oath, ...); False saves memory
    # authoritative per-name answers, checked before the rules above: group name
    # = granted, None = takes a slot. prepare_pretraining(gamedata=...) fills it.
    granted_known: Dict[str, Optional[str]] = field(default_factory=dict)
    # {loose key: canonical name} from game data (GameData.canonical_names); names
    # that are not exact matches are mapped through it after item_aliases
    canonical_names: Dict[str, str] = field(default_factory=dict)
    # --- filters
    min_power: int = 1
    min_talents: int = 1
    min_mantras: int = 0
    max_stat: int = 100
    include_private: bool = False
    max_unknown_frac: float = 0.25     # drop builds with more unknown items than this (given catalog)
    # --- dedup / lineage
    dedup: bool = True                 # exact duplicates (same items and stats) -> one build
    collapse_revisions: bool = True    # same author id + build name -> keep latest only
    # --- catalog derivation (only when no catalog is passed)
    min_item_count: int = 20           # items picked fewer times are left out of the catalog
    keep_items: Tuple[str, ...] = ()   # kept in the catalog however rarely picked (items released after
                                       # most of the corpus, e.g. from a game data sync; needs gamedata)
    # --- weights
    half_life_days: float = 120.0
    recency_floor: float = 0.05
    reference_time: Optional[str] = None     # ISO date; default = newest build in the file
    vote_strength: float = 0.25
    quality_clip: Tuple[float, float] = (0.25, 4.0)
    dup_alpha: float = 0.5             # weight multiplier = (copies) ** dup_alpha
    max_power: int = 20
    power_weight: bool = True          # weight *= power / max_power
    patches: Tuple[Tuple[str, str], ...] = ()      # ((iso_date, ruleset_name), ...) ascending
    ruleset_weights: Dict[str, float] = field(default_factory=dict)  # ruleset -> multiplier
    # --- statistics
    stats_use_weights: bool = True
    min_pair_count: float = 30.0       # lift trusted only above this weighted co-pick count
    min_joint_avail: float = 200.0
    auto_grant_rate: float = 0.98      # pick rate above which an item is reported as auto-granted
    meta_bound_purity: float = 0.98    # talent reported as meta-bound if this share of pickers share one value
    meta_bound_fields: Tuple[str, ...] = ("Oath", "Race", "Origin")


@dataclass
class ParsedRecord:
    """One record after parsing: the Build plus raw signals used for weighting."""
    build: Build
    updated: Optional[float]
    up: int
    down: int
    views: int
    saved: int
    power: int
    lineage: str
    granted: Tuple[str, ...] = ()      # talent names removed as granted


@dataclass
class Record:
    """A loaded frozen record: the full reviews (v3.5: audits are gone)."""
    reviews: List[Review]
    locators: List[dict]
    values: List[dict]
    meta: Dict[str, dict]                       # review_id -> pool_id, link, build_name, model_run
    manifest: dict = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def all(self) -> List[Review]:
        return sorted(self.reviews, key=lambda r: r.timestamp or 0.0)
# ====================================================================== sheets


@dataclass
class Mark:
    label: str                                   # keep | add | remove | iffy
    strength: Optional[str] = None
    reason: Optional[str] = None
    related: Tuple[str, ...] = ()
