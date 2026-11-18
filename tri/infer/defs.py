"""Definitions of inference: InferConfig, changes, alternatives, situations,
review-queue items and draft sheets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tri.core.types import Build, Review
from tri.rules.defs import Ctx
from tri.rules.passes import LUXURY

REVIEW = "review"
GLOBAL_RULE_KINDS = ("BROKEN", "REDUNDANT", "CONFLICT", "DEPENDS", "OFF_ARCHETYPE")  # + gate, + family-scoped
ARCHETYPE = "archetype"     # output pass of a mined take rule whose pick is not a confirmed central


@dataclass
class InferConfig:
    """Inference knobs. Worth values are local pick rates in [0, 1]."""
    swap_margin: float = 0.15      # luxury swap needs this much more worth
    max_swaps: int = 3
    # a worth swap never takes out a pick reviews of similar builds kept, nor
    # brings in one they removed or left out (any such claim, binding or not)
    swap_guard: bool = True
    rule_tiers: bool = True        # foundational / central talents come from rules only, never by worth
    tier_fallback: bool = True     # ... unless no luxury talent fits and the talent cap is unfilled
    tier_context: bool = True      # tiers are read per build: F / C with no rule firing here = luxury here
    no_fit_rate: float = 0.05      # a dropped pick taken this rarely nearby -> no_synergy
    iffy_margin: float = 0.08      # kept luxury pick within this of the best alternative -> iffy
    max_iffy_frac: float = 0.2
    min_plausibility: float = 0.2  # regions shown as alternative archetypes
    anchor_min_worth: float = 0.3  # a current pick anchors central rules only if required or at least this worth
    memory_hard: float = 0.5       # claim strength at or above this overrides; below it nudges worth
    review_rounds: int = 3         # completions re-asked of the reviews on their own result
    review_pass_rounds: int = 2    # completions reviewed as full builds before they are returned
    prereq_solo_min: float = 0.3   # a prerequisite the corpus takes without its dependants less often is
<<<<<<< HEAD
                                   #   taken only with one
=======
                                   #   taken only with one (math §2.8d)
>>>>>>> main
    package_min_q: float = 0.2     # ... and only with a dependant at this worth or more
    memory_nudge: float = 0.3      # worth nudge per unit of soft claim strength
    note_worth: float = 0.35       # worth added to indicator / core components near the noted build
    note_floor: float = 0.02       # ...only where kernel x archetype weight reaches this
    value_cap: float = 0.5         # largest admin flat value offset, either way
<<<<<<< HEAD
    emit_margin: float = 0.0       # Tri.as_review emits a change only at evidence margin >= this
    strong_margin: float = 0.5     # ... and marks it strong at margin >= this
    fill_capacity: bool = True     # analysis fills empty mantra slots, then talents to the cap, before swaps
    emit_iffy: bool = False        # ... and writes iffy marks (reviewers rarely do: 1 in 150 real marks)
=======
    emit_margin: float = 0.0       # Tri.as_review emits a change only at evidence margin >= this (math §11.4)
    strong_margin: float = 0.5     # ... and marks it strong at margin >= this
    fill_capacity: bool = True     # analysis fills empty mantra slots, then talents to the cap, before swaps
    emit_iffy: bool = False        # ... and writes iffy marks (reviewers rarely do: 1 in 150 real marks)
    consensus_central: float = 0.3  # a family tell confirms a central pick at c p(a | u) a(u) >= this
>>>>>>> main
    central_rounds: int = 4
    repair_rounds: int = 6         # rule re-checks after swaps
    cold_support: float = 1.0      # an unrated item with less local support than this has no worth evidence
    focus: bool = True             # commit to one weapon class before luxuries (v2.24, tri.focus)
    focus_k: int = 150             # class prior: nearest labelled corpus builds
    focus_min_labelled: int = 20   # ... fewer labelled builds than this: choose the class by worth
    focus_min_q: float = 0.3       # the class's talents taken first (before luxuries) at this worth or more
    # the 4.0 scorer (tri.infer.scorer): z = sc_bias + sc_field L + sc_nbr logit(q) + sc_direct D + sc_prop P
    sc_bias: float = 0.0
    sc_field: float = 0.0
    sc_nbr: float = 1.0
    sc_direct: float = 2.0          # fitted on the real LOBO F1 (docs/evaluation.md)
    sc_prop: float = 1.0
    sc_keep: float = 0.5           # keep (unmarked) claims count this much of an explicit operation
    sc_remove: float = -4.0        # a pick scoring below this is removed
    sc_fill: float = 1.0           # a review fills an empty slot with the best item scoring at least this (completions fill every slot)
    sc_swap: float = 6.0           # an outside item scoring this much more replaces the weakest pick
    sc_directed: bool = True       # explicit item-level claims reach along / across the item direction
    sc_h_par: float = 1.797        # along: the ledger's reach bandwidth, reach / sqrt(-2 ln sim_min)
    sc_h_perp: float = 4.49        # across: 2.5 x
    sc_propagate: bool = False     # archetype-fit claims nudge similar unclaimed items (off: no measured gain)
    sc_nbr_reviews: bool = True    # neighbour rates include reviewed builds (review_weight on touched items)
    sc_tau: float = 0.5            # ... with mean-centred worth cosine above this


@dataclass
class Change:
    """One proposed change, in the reviewers' vocabulary."""
    item: str
    action: str                     # add | remove
    reason: Optional[str] = None    # Reason value
    related: List[str] = field(default_factory=list)
    marginal: float = 0.0           # the item's worth here
    source: str = "worth"           # review | review_rule | rules | worth
    review_id: Optional[str] = None
    pass_: str = LUXURY             # foundational | central | luxury | review
    why: str = ""


@dataclass
class Alternative:
    """The answer under one archetype (region of the concept space)."""
    talents: List[str]
    mantras: List[str]
    plausibility: float
    gain: float                     # worth added minus worth removed
    changes: List[Change]
    iffy: List[str]
    mode: int                       # region id
    name: str = ""
    passes: Dict[str, str] = field(default_factory=dict)      # resulting item -> pass
    worth: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, tuple] = field(default_factory=dict)
    capacity: Tuple[int, int] = (0, 0)
    concept: List[float] = field(default_factory=list)
    violations: List[dict] = field(default_factory=list)   # rules the answer could not satisfy (and why)
    sidegrades: List[dict] = field(default_factory=list)   # the other side of each sidegrade the answer is committed to
    unrated: List[str] = field(default_factory=list)       # items in the answer with no worth evidence yet
    iffy_why: Dict[str, dict] = field(default_factory=dict)  # iffy item -> {reason, related, review_id}
    indicative: Dict[str, dict] = field(default_factory=dict)  # pick -> {archetype, kind, review_id, weight}
    focus: Dict = field(default_factory=dict)  # weapon class focus: {classes, source, share}

    def items(self) -> List[str]:
        return self.talents + self.mantras


@dataclass
class _Sit:
    """Everything inference knows about one build under one archetype."""
    build: Build
    ctx: Ctx
    feats: Set[str]
    q: np.ndarray
    support: np.ndarray
    memory: Dict[str, tuple]
    mode: int
    name: str
    plausibility: float
    z: np.ndarray
    oath: Optional[str] = None      # the build's oath (its mantra slots count)
    outside: Set[str] = field(default_factory=set)   # build picks outside the catalog (their slot grants count)
    req_why: Dict[str, tuple] = field(default_factory=dict)   # required item -> (reason, related)
    indicative: Dict[str, dict] = field(default_factory=dict)  # component -> note that rates it
    family: Optional[str] = None    # located family key of the reading (v2.19 chasm on mined rules)
    focus: Dict = field(default_factory=dict)   # weapon class focus (tri.focus)
    weapon: Dict = field(default_factory=dict)  # hardcoded weapon rules {forbid, require}: item -> why
<<<<<<< HEAD
    score: Dict = field(default_factory=dict)   # the scorer's parts {z, L, q, D, P, why} (tri.infer.scorer)
=======
>>>>>>> main


@dataclass
class QueueItem:
    build: Build
    priority: float
    novelty: float
    luxury: float
    ambiguity: float
    patch: float
    edits: int
    source: str              # active | random


@dataclass
class DraftSheet:
    """A drafted review: the Review, its reply lines and what was held back."""
    review: Review
    lines: List[str]
    archetype: Optional[str]
    margins: Dict[str, Optional[float]] = field(default_factory=dict)    # item -> evidence margin
    withheld: List[dict] = field(default_factory=list)                     # changes below emit_margin or over capacity
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)
