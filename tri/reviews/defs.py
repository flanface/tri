"""Definitions of review authority: claims, supersessions, weights and the
ledger, consensus and locator configurations; locators and archetype readings."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tri.core.codec import build_from_dict, build_to_dict, legacy_names
from tri.core.types import ABSENT_CONDITION, PRESENT_CONDITION, Build, Reason
from tri.reviews.variants import family_key, split_archetype

_CONDITIONAL = {r.value for r in PRESENT_CONDITION | ABSENT_CONDITION}


@dataclass
class Claim:
    """One item-level statement made by a review."""
    review_id: str
    order: int                    # position in recency order (higher = newer)
    item: str
    label: str                    # keep | add | remove | iffy
    reason: Optional[str] = None
    op_index: Optional[int] = None    # index in Review.operations (None for keep)
    weight: float = 1.0           # after supersession
    strength: Optional[str] = None    # weak | strong for add/remove; None otherwise
    related: Tuple[str, ...] = ()     # "redundant with y" / "works against y": the claim holds only where y is present
    origin: str = "review"            # review | overturn | uphold
    target: Optional[dict] = None     # overturn / uphold: the analysis change judged

    @property
    def broken(self) -> bool:
        """A "does not work" removal: context-free, it conflicts with every claim on the item."""
        return self.label == "remove" and self.reason == Reason.BROKEN.value

    @property
    def general(self) -> bool:
        """A "generally good" add: a foundational talent, offered to every build."""
        return self.label == "add" and self.reason == Reason.GENERALLY_GOOD.value

    @property
    def general_sidegrade(self) -> bool:
        """A strong "x is a sidegrade to y": holds generally, not only under this archetype."""
        return self.label == "iffy" and self.reason == Reason.SIDEGRADE.value and self.strength == "strong"

    @property
    def global_scope(self) -> bool:
        """Applies to builds of any archetype (broken, generally good, general sidegrade)."""
        return self.broken or self.general or self.general_sidegrade

    @property
    def weapons(self) -> Tuple[str, ...]:
        """v3.3: related weapons ("weapon:<name>"): the claim holds only on builds
        wielding one of them, whatever its reason."""
        return tuple(y for y in self.related if isinstance(y, str) and y.startswith("weapon:"))

    @property
    def item_related(self) -> Tuple[str, ...]:
        return tuple(y for y in self.related if not (isinstance(y, str) and y.startswith("weapon:")))

    @property
    def conditional(self) -> bool:
        """True when the claim is relative to other picks (its related items):
        redundant with / works against / synergy with y need y present;
        needs a, b (NW) holds only where one of a, b is missing; a related
        weapon (v3.3) needs that weapon equipped."""
        return bool(self.weapons) or (bool(self.item_related) and self.reason in _CONDITIONAL)

    @property
    def absent_condition(self) -> bool:
        return self.reason == Reason.NEEDS.value

    def holds_in(self, items) -> bool:
        """Does the claim transfer to a build with these picks? Present
        conditions need every related item present; the NW condition needs
        one of them missing."""
        if not self.conditional:
            return True
        ws = self.weapons
        if ws and not any(w in items for w in ws):     # items carry the build's "weapon:<name>" tokens
            return False
        rel = self.item_related
        if not rel or self.reason not in _CONDITIONAL:
            return True
        if self.absent_condition:
            return not all(y in items for y in rel)
        return all(y in items for y in rel)

    def condition_tokens(self) -> Tuple[str, ...]:
        """related items as worth conditions: "y" present, "!y" for NW (absent);
        related weapons are kept as "weapon:<name>" (worth cannot see equips,
        so it abstains on them)."""
        rel = self.item_related if self.reason in _CONDITIONAL else ()
        toks = tuple("!" + y for y in rel) if self.absent_condition else tuple(rel)
        return toks + self.weapons


@dataclass
class Supersession:
    item: str
    older: str                    # review id
    older_label: str
    newer: str
    newer_label: str
    similarity: float
    factor: float                 # weight multiplier applied to the older claim


@dataclass
class ReviewWeights:
    """Per-review weights after conflict resolution."""
    review_id: str
    ops: List[float]              # one per operation, same order as Review.operations
    keep: Dict[str, float]        # endorsed item -> weight
    review: float                 # mean claim weight (scales whole-review terms)

    @property
    def active(self) -> bool:
        return self.review > 0


@dataclass
class LedgerConfig:
    sim_min: float = 0.80         # below this, builds are different archetypes: no conflict
    sim_full: float = 0.95        # at or above this, the newer claim fully replaces the older
    sim_same: float = 0.99        # at or above this, a build counts as the reviewed build itself
    reach: float = 1.2            # concept-space distance (weighted RMS logit) at which sim = sim_min
    stat_scale: float = 100.0     # fallback similarity only (no concept space)
    patch_factor: float = 0.25    # weight of a pre-patch claim about a changed item
    bind_sim: float = 0.95        # a strong item claim from another build binds only at sim >= bind_sim
    # v2.25.1: unmarked items of a build tri made (a completion / proposal saved
    # to dwt.sh, then reviewed) are tri's own picks echoed back, not a reviewer's
    # choice: their endorsement counts this much, never binds, and never hides an
    # older explicit claim on the item. Explicit operations keep full weight.
    echo_keep: float = 0.25
    # v3.3 family reach (math §5.8): reviews of builds located in the query's own
    # family, at any distance inside it, pool their claims per item. A pooled
    # claim needs fam_min distinct reviewed builds behind one side and a weighted
    # share >= fam_share; it nudges, and binds only with fam_bind strong
    # item-level claims (not slot-relative) behind it and no strong one against.
    # Local claims (within reach) always take precedence; the chasm is never crossed.
    fam_reach: bool = True
    fam_min: int = 2
    fam_share: float = 0.75
    fam_bind: int = 3
    fam_kappa: float = 1.5
    fam_keep: float = 0.5         # an endorsement (unmarked pick) counts this much of an edit
    fam_overlap: float = 0.5      # family-posterior overlap 1 - TV a reviewed build needs
    # builds no locator places (neither of the two) share a family when they share
    # a concept region; their overlap counts this much
    fam_region: float = 0.6
    # a broad consensus binds whatever the strength: >= fam_consensus distinct
    # builds with share >= fam_consensus_share and no strong claim against
    fam_consensus: int = 4
    fam_consensus_share: float = 0.9
    # nearby reviewed builds (similarity >= pool_sim) join the pool with overlap =
    # similarity, so a consensus of neighbours counts even outside a located family
    pool_sim: float = 0.8
    # a binding pooled claim overrides a single local claim that would only nudge
    pool_override: bool = True
    # a full review that leaves an item out of the build it returns endorses its
    # absence: pooled, a consensus of such reviews ("absent") keeps worth from
    # swapping the item in (slot filling and requirements are unaffected)
    pool_absent: bool = True


@dataclass
class ConsensusConfig:
    kappa: float = 1.5               # pseudo-evidence: c = net / (net + kappa)
    min_reviewers: int = 2           # distinct units (see unit) before a tell is promoted
    unit: str = "build"              # "build": (reviewer, build) units; "reviewer": distinct reviewers
    op_weight: float = 0.5           # an archetype-reasoned edit counts this much of a note
    worth: float = 0.35              # eta_c: q_t += eta_c * c * p(a | u) * a(u)
    lift: float = 2.0                # lambda_c: nats per held tell on the reading (capped by indicator_lift)
    floor: float = 0.2               # the reading lift needs p(a | u) >= floor already
    enabled: bool = True


@dataclass
class Tell:
    key: str                         # family key
    item: str
    sign: int                        # +1 tell (take), -1 avoid
    confidence: float
    support: float                   # summed weight behind the sign
    against: float                   # summed weight against it
    reviewers: int                   # distinct reviewers behind the sign
    notes: int                       # how many of them were notes (not edits)
    reviews: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(key=self.key, item=self.item, sign=self.sign, confidence=round(self.confidence, 3),
                    support=round(self.support, 3), against=round(self.against, 3),
                    reviewers=self.reviewers, notes=self.notes, reviews=list(self.reviews))


@dataclass
class LocatorConfig:
    """Locator knobs."""
    bandwidth: Optional[float] = None       # h of the review barrier and reach decay (None = the space's bandwidth)
    read_bandwidth: Optional[float] = None  # v3.9: h of the archetype reading (None = chosen from read_grid)
    read_grid: tuple = (0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
    read_tie: float = 0.01                  # nats a narrower bandwidth must gain over a wider one
    reach: Optional[float] = None           # located only within this distance of a locator (None = ledger reach)
    min_confidence: float = 0.5             # ...and with top archetype probability at least this
    barrier: float = 1.0                    # 0 = reviews ignore archetypes; 1 = no review influence across archetypes
    select_metric: bool = True              # refit chooses metric_power by leave-one-out (needs enough locators)
    metric_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    select_z: float = 2.0                   # move only when the paired gain exceeds this many standard errors
    select_min_gain: float = 0.01           # ... and at least this many nats per locator (practical size)
    select_min_per_archetype: int = 3         # selection needs this many locators in each of
    select_min_archetypes: int = 2            # ...at least this many archetypes
    # twin split: archetypes the map confuses told apart by the few picks that differ (math §6.6)
    refine: bool = True                       # off = the kernel reading alone
    refine_prior: float = 4.0                 # kappa: pseudo-locators at the pair's pooled rate
    refine_min_logodds: float = 1.5           # an item is a tell at |log odds ratio| >= this between the twins (log 4.5)
    refine_min_locators: int = 2              # each twin needs this many locators
    twin_overlap: float = 0.2                 # a pair is twins if the kernel gives a locator of one this much of the other (leave-one-out mean)
    twin_mass: float = 0.5                    # split only where the pair holds this much of the kernel reading
    twin_clip: float = 4.0                    # largest log-likelihood ratio (nats) the tells may add
    twin_min_gain: int = 1                    # a pair is split only if held-out locators corrected - broken >= this
    # indicator notes ("# a: x, y, z" in a review, math §6.7): local, bounded lifts
    indicator_lift: float = 6.0               # nats added to the odds of the noted archetype (v2.18: own cap)
    indicator_floor: float = 0.05             # ...only where the kernel already gives it this much
    anchor_mass: float = 1.0                  # v3.9: a reviewed build noted "is a" joins a's locator mean with this weight
    tell_z: float = 1.64                      # v3.9: Wilson z of a tell's balanced precision among the locators
    anchor_veto: float = 0.9                  # "not a": k_a is scaled by 1 - veto * K at the noted build
    # v2.19 chasm (math §5.4): borders between archetype families are chasms
    chasm_power: float = 6.0                  # gamma: chi = 1 - barrier a(u) a(v) (1 - (1 - TV)^gamma)
    chasm_hard: bool = True                   # both builds located in different families -> chi = 1 - barrier
    chasm_corpus: bool = True                 # the chasm also gates corpus neighbours and the cross ring
    chasm_rules: bool = True                  # mined central take rules fire only in their home families
    chasm_rule_min: float = 5.0               # ...a family is home with this much covered weight and rate >= theta_take
    chasm_rule_sample: int = 30000            # reference builds used to find the homes
    # v2.19 reviewer consensus (math §8.5a): repeated notes / edits become family tells
    consensus: bool = True
    consensus_kappa: float = 1.5              # c = net / (net + kappa)
    consensus_min_reviewers: int = 2          # distinct (reviewer, build) units before a tell is promoted
    consensus_unit: str = "build"             # "reviewer": count distinct reviewers (pre-2.23)
    consensus_op_weight: float = 0.5          # an archetype-reasoned edit counts this much of a note
    consensus_worth: float = 0.35             # eta_c: q_t += eta_c c p(a | u) a(u)
    consensus_lift: float = 2.0               # lambda_c: nats per held tell (total capped by indicator_lift)
    consensus_floor: float = 0.2              # reading lift needs p(a | u) >= this already


@dataclass
class Locator:
    """One labelled build."""
    locator_id: str
    build: Build
    archetype: str
    link: Optional[str] = None
    added_by: Optional[str] = None
    timestamp: Optional[float] = None

    @property
    def key(self) -> str:
        """The family key: "Gank / Frost Mage" reads as Gank (tri.variants)."""
        return family_key(self.archetype)

    @property
    def family(self) -> str:
        return split_archetype(self.archetype)[0]

    @property
    def variant(self) -> Optional[str]:
        return split_archetype(self.archetype)[1]

    def to_dict(self) -> dict:
        return dict(locator_id=self.locator_id, archetype=self.archetype, link=self.link, added_by=self.added_by,
                    timestamp=self.timestamp, build=build_to_dict(self.build))

    @staticmethod
    def from_dict(d: dict) -> "Locator":
        d = legacy_names(d)                          # rows written before v2.5 say "purpose"
        return Locator(str(d.get("locator_id") or d.get("id") or uuid.uuid4().hex[:12]), build_from_dict(d["build"]),
                       str(d["archetype"]).strip(), d.get("link"), d.get("added_by"), d.get("timestamp"))


@dataclass
class ArchetypeReading:
    """The archetype a build reads as."""
    archetype: Optional[str]                  # top archetype if located, else None
    confidence: float                       # p(top archetype | u)
    posterior: Dict[str, float]             # archetype -> p
    nearest: Optional[str]                  # nearest locator id
    distance: float                         # its distance
    located: bool
    activity: float                         # a(u): 1 within reach, decaying beyond
    best: Optional[str] = None              # top archetype even when unlocated
    coarse: Optional[Dict[str, float]] = None   # the kernel reading before the fine reading
    forced: bool = False                    # the archetype was chosen by the caller

    def to_dict(self) -> dict:
        d = dict(archetype=self.archetype, located=self.located, confidence=round(self.confidence, 3),
                 best=self.best, nearest_locator=self.nearest, distance=round(self.distance, 3),
                 activity=round(self.activity, 3),
                 posterior={k: round(v, 3) for k, v in sorted(self.posterior.items(), key=lambda x: -x[1])})
        if self.coarse is not None:
            d["coarse"] = {k: round(v, 3) for k, v in sorted(self.coarse.items(), key=lambda x: -x[1])}
        if self.forced:
            d["forced"] = True
        return d
