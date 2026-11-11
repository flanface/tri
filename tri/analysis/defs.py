"""Definitions of analysis results: atlases, facts and sidegrades."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ the atlas
@dataclass
class Atlas:
    """The corpus (or a sample of it) read as archetypes: everything Part A to C
    needs, computed once per field, locator set and review count."""
    names: List[str]                 # archetype display names
    source: str                      # "locators" | "regions"
    U: np.ndarray                    # [n, D'] concept coordinates
    w: np.ndarray                    # [n] build weights
    ids: List[Optional[str]]
    bm: object                       # BuildMatrix of the rows
    P: np.ndarray                    # [n, A] archetype probabilities
    dmin: np.ndarray                 # [n] distance to the nearest locator (region centre)
    top: np.ndarray                  # [n] top archetype index
    second: np.ndarray               # [n] second archetype index (-1: none)
    pi1: np.ndarray
    pi2: np.ndarray
    kind: np.ndarray                 # [n] reading (READINGS)
    centres: np.ndarray              # [A, D'] archetype centres
    share: np.ndarray                # [A] weighted share of placed builds
    count: np.ndarray                # [A] placed builds
    Q: np.ndarray                    # [n, N] field worth of every item
    avail: np.ndarray                # [n, N] the build can take the item
    X: np.ndarray                    # [n, N] picks (0/1)
    reach: float
    h: float
    ubar: np.ndarray = field(default=None)   # weighted mean coordinate

    @property
    def placed(self) -> np.ndarray:
        return self.kind != "unplaced"

    def members(self, a: int, placed: bool = True) -> np.ndarray:
        m = self.top == a
        return m & self.placed if placed else m

    def index(self, name: str) -> Optional[int]:
        key = " ".join(str(name).lower().split())
        for i, n in enumerate(self.names):
            if " ".join(n.lower().split()) == key:
                return i
        for i, n in enumerate(self.names):                # prefix / substring match
            if key and key in n.lower():
                return i
        return None


# ------------------------------------------------------------------ facts
@dataclass
class Fact:
    key: str
    family: str
    archetype: Optional[str]
    sentence: str
    numbers: dict
    p: Optional[float]              # held-out p-value (None: not testable yet)
    surprise: float
    support: int
    graphic: Optional[str] = None   # "card:<A>" | "item:<t>" | "stats:<t>" | "meta" | "tree"
    status: str = "candidate"       # certified | candidate
    score: float = 0.0
    items: List[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        """The fact in symbols (tri.symbolic.fact), with its p-value."""
        from tri.rules.symbolic import fact
        return fact(self)

    def to_dict(self) -> dict:
        return dict(asdict(self), symbol=self.symbol)


@dataclass
class SidegradeConfig:
    min_builds: int = 200            # regions smaller than this are not searched
    contested: Tuple[float, float] = (0.15, 0.85)   # pick-rate band of candidate items
    min_count: int = 20              # builds that must take a candidate item
    gap: float = 0.35                # pick-rate difference between the sides, per item
    min_share: float = 0.2           # each side chosen by at least this share of the region
    min_commit: float = 0.7          # share of builds committed to one side
    max_items: int = 8               # items per side
    max_per_region: int = 2
    min_cross: float = 0.2           # opposite-side picks: mean phi <= -min_cross
    min_within: float = 0.2          # same-side picks: mean phi >= min_within
    pair_overlap: float = 0.6        # core overlap for two regions to be searched together
    core_rate: float = 0.6           # an item is core to a region if this share of its builds take it
    core_hold: float = 0.8           # the pool: builds that take this share of the (shared) core
    restarts: int = 4
    iters: int = 25
    seed: int = 0


@dataclass
class Sidegrade:
    region: int                      # first region of the pool searched
    a: List[str]
    b: List[str]
    share_a: float                   # held-out share of the region's builds committed to a
    share_b: float
    commit: float                    # held-out share committed to either side
    builds: int                      # region builds in the held-out half
    gap_a: Dict[str, float] = field(default_factory=dict)   # held-out pick-rate difference per item
    gap_b: Dict[str, float] = field(default_factory=dict)
    regions: List[int] = field(default_factory=list)        # the pool: one region or two variants of one archetype
    cross: float = 0.0               # held-out mean phi between sides
    core: List[str] = field(default_factory=list)           # shared core picks that define where it applies
    core_hold: float = 0.8

    def applies(self, sel, available=None) -> bool:
        """The build holds core_hold of the core items it can take: it is a
        build of the archetype the sidegrade was found in."""
        if not self.core:
            return False
        can = [i for i in self.core if available is None or available(i) or i in sel]
        return bool(can) and sum(i in sel for i in can) >= self.core_hold * len(can)

    def counts(self, sel) -> Tuple[int, int]:
        sel = set(sel)
        return sum(i in sel for i in self.a), sum(i in sel for i in self.b)

    def side(self, sel) -> int:
        """+1 committed to a, -1 to b, 0 neither (side score threshold 0.5)."""
        ca, cb = self.counts(sel)
        s = ca / len(self.a) - cb / len(self.b)
        return 1 if s >= 0.5 else -1 if s <= -0.5 else 0

    def leaning(self, sel) -> int:
        """+1 if sel holds more of a than b, -1 if more of b, 0 if tied."""
        ca, cb = self.counts(sel)
        return int(np.sign(ca / len(self.a) - cb / len(self.b)))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Sidegrade":
        g = Sidegrade(**d)
        g.regions = g.regions or [g.region]
        return g
