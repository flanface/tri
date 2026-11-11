"""Definitions of rules: rule-mining configuration, conditions, the availability
context Ctx, and review cases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set

import numpy as np

from tri.core.types import Build


@dataclass
class PassConfig:
    """Rule-mining knobs."""
    theta_take: float = 0.80       # take: certified lower bound on the rate
    theta_never: float = 0.05      # never: certified upper bound on the rate
    theta_contrast: float = 0.10   # never: rate when one condition is flipped
    z: float = 1.645               # one-sided confidence (1.645 ~ 95%)
    min_support: float = 30.0      # weighted builds a rule must rest on
    max_depth: int = 3             # longest conjunction considered (v3.0: 3, was 5; interpretability)
    grounded: bool = True          # v3.0: stat literals only on stats of the item's own requirements,
                                   # its prerequisites and its groups (tri.rules.groups.grounded_stats)
    ground_fallback: bool = True   # v3.0: an item whose grounded take rules cover too few of its takers
    ground_cover: float = 0.5      # (share below ground_cover) is searched once more with every stat
    anchors: str = "related"       # v3.0: item literals: "related" (prerequisites, group mates; v3.0.1: no dependants),
                                   # "mi" (any item by mutual information, the pre-3.0 behaviour), "none"
    groups: bool = True            # v3.0: group literals |G(c)| >= n, |D(k)| >= n (known groups)
    archetypes: bool = True        # v3.0: archetype literals A = a (locator readings of the corpus)
    replicate: bool = True         # v3.0: the certified form must also pass on a third, untouched split
    necessity: bool = True         # v3.0: every literal must matter on the certification split
    stat_step: int = 5             # stat thresholds are multiples of this (1 = any integer)
    co_items: int = 32             # anchor candidates per item (by mutual information)
    min_breadth: float = 0.5       # foundational: overlap of covered builds' cluster mix with the corpus mix
    breadth_regions: int = 16      # pick-profile clusters used for breadth and uniformity
    uniform_support: float = 5.0   # uniformity: clusters need at least this weight inside a cell to veto ...
    uniform_share: float = 0.02    # ... and at least this share of the cell's weight (a handful cannot veto a big cell)
    repair_steps: int = 2          # a vetoed take rule gets up to this many literals splitting off the vetoing cluster
    leave_item_out: bool = True    # clusters for item t's veto are assigned without t's own column
    aggregates: bool = True        # stat totals as conditions (attunement points, stats at 75+, ...)
    max_overlap: float = 0.8       # drop a rule whose covered builds overlap a kept one's this much (weighted Jaccard)
    min_lift: float = 0.10         # conditional take rules of items below theta_take need rate >= (1 + min_lift) x base
    contrast_ratio: float = 3.0    # never: contrast >= max(theta_contrast, contrast_ratio x the rule's certified upper bound)
    multiple_testing: str = "bonferroni"   # certificate z per item: "bonferroni" (alpha / candidates) | "none"
    max_rules_per_item: int = 6    # take rules kept per item (most general first)
    max_never_per_item: int = 3    # never rules kept per item
    item_rows: Optional[int] = 40_000   # builds per item used to mine (None = all)
    n_jobs: int = 0                # worker processes for mining (0 = auto: all cores up to 8, only for big corpora; 1 = serial)
    sample: Optional[int] = 300_000
    seed: int = 0


@dataclass
class Condition:
    kind: str                      # stat | agg | context | item | group
    key: str                       # stat name / token / item / group name ("G(Bruiser)", "D(Flame Leap)")
    op: str                        # >= | < | has | not
    value: Optional[float] = None  # stat threshold / group count

    def id(self) -> str:
        return f"{self.kind}:{self.key}:{self.op}:{self.value}"

    def text(self) -> str:
        if self.kind in ("stat", "agg"):
            return f"{self.key} {self.op} {int(self.value)}"
        if self.kind == "group":
            return f"{int(self.value)}+ of {self.key}" if self.op == ">=" else f"fewer than {int(self.value)} of {self.key}"
        if self.kind == "context":
            f, v = self.key.split(":", 1)
            return f"{f} {'is' if self.op == 'has' else 'is not'} {v}"
        return f"with {self.key}" if self.op == "has" else f"without {self.key}"


@dataclass
class Ctx:
    """What a condition can look at: highest stats, picks, context tokens."""
    hi: Dict[str, int]
    items: Set[str]
    tokens: Set[str]
    region: int
    avail: np.ndarray
    groups: Optional[Dict[str, FrozenSet[str]]] = None   # known groups (Rulebook.groups), for group literals


@dataclass
class Case:
    """One piece of rule evidence: a review, its build and weight."""
    review_id: str
    reviewer_id: str
    build: Build                 # the original build (context)
    weight: float
    order: int = 0               # recency rank (higher = newer); authority mode only
