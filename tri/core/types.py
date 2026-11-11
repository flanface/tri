"""Definitions of the data model: kinds, categories, review actions, strengths and
reasons, items, builds, operations and reviews.

Review semantics (the contract every other module relies on):

* The reviewer receives a full build (highest stats, final stats, talents, mantras).
* Stats are fixed; the reviewer never judges stat optimality.
* All talent slots are filled initially.
* Operations: ADD, REMOVE, IFFY. Each has a binary strength (WEAK | STRONG);
  ADD/REMOVE also carry a reason.
* Anything in the *resulting* build that is not marked IFFY is endorsed as
  "perfect for its place" in that resulting context.
* Items absent from both the original and the resulting build carry no label."""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from tri.core.catalog import Catalog


DEFAULT_STATS: Tuple[str, ...] = (
    "Strength", "Fortitude", "Agility", "Intelligence", "Willpower", "Charisma",
    "HeavyWeapon", "MediumWeapon", "LightWeapon",
    "Flamecharm", "Frostdraw", "Thundercall", "Galebreathe",
    "Shadowcast", "Ironsing", "Bloodrend",
)


class Kind(str, enum.Enum):
    """Item kind: talent or mantra."""
    TALENT = "talent"
    MANTRA = "mantra"


class Category(str, enum.Enum):
    """Prior structural hint supplied by builders; the model may override it."""
    FOUNDATIONAL = "foundational"
    CENTRAL = "central"
    LUXURY = "luxury"
    UNKNOWN = "unknown"


class Action(str, enum.Enum):
    """Review operation: add / remove / iffy edit the build. (v3.5: the audit
    verdicts overturn / uphold are gone; stored ones are dropped when read.)"""
    ADD = "add"
    REMOVE = "remove"
    IFFY = "iffy"


# v3.5: audit verdict actions. Not operations any more: codec drops them on read.
RETIRED_ACTIONS = ("overturn", "uphold")


class Strength(str, enum.Enum):
    """Binary importance of acting on a judgement. 'strong' = clear mistake /
    clear necessity; 'weak' = marginal. Mapped to loss weights by
    `strength_weight` (review-rule evidence; RuleInducer(strength_weights=...))."""
    WEAK = "weak"
    STRONG = "strong"


DEFAULT_STRENGTH_WEIGHTS: Dict["Strength", float] = {Strength.WEAK: 0.5, Strength.STRONG: 1.0}


def strength_weight(s: "Strength", table: Optional[Dict["Strength", float]] = None) -> float:
    """Numeric weight for a strength. Default: weak 0.5, strong 1.0."""
    t = table or DEFAULT_STRENGTH_WEIGHTS
    return float(t[Strength(s)])


class Reason(str, enum.Enum):
    """The reviewer reasons. REDUNDANT_WITH and WORKS_AGAINST need `related`.
    NECESSARY_FOR_ARCHETYPE and GENERALLY_GOOD are reasons for adds; the rest
    for removes. GENERALLY_GOOD (talents only) endorses a foundational talent:
    good for any build that can take it, not only this archetype."""
    LESS_NECESSARY_SPACE = "less_necessary_space"   # x removed to make space
    NECESSARY_FOR_ARCHETYPE = "necessary_for_archetype"  # x is needed for your archetype
    REDUNDANT_WITH = "redundant_with"                # x unnecessary because you have y
    WORKS_AGAINST = "works_against"                  # x works against y
    NO_SYNERGY = "no_synergy"                        # x does not synergise with build
    SCENARIO_MISMATCH = "scenario_mismatch"          # weak: x does not work in your scenario;
                                                     # strong: x does not belong on this archetype (off_archetype)
    BROKEN = "broken"                                # x does not work / garbage
    GENERALLY_GOOD = "generally_good"                # x is a foundational talent, good for any build
    # v2.17 shorthands
    SYNERGY_WITH = "synergy_with"                    # + x SY > y: add x as it synergises with y
    NEEDS = "needs"                                  # - x,y NW > a,b: x,y do not work without a,b
                                                     #   (equivalent to adding a,b should there be space)
    LUXURY_LESS = "luxury_less"                      # - x LL: worth less as a luxury
    LUXURY_MORE = "luxury_more"                      # + x LM: worth more as a luxury
    SIDEGRADE = "sidegrade"                          # = x SG > y: x is a sidegrade to y


RELATIONAL_REASONS = {Reason.REDUNDANT_WITH, Reason.WORKS_AGAINST, Reason.SYNERGY_WITH, Reason.NEEDS,
                      Reason.SIDEGRADE}
ADD_REASONS = {Reason.NECESSARY_FOR_ARCHETYPE, Reason.GENERALLY_GOOD, Reason.SYNERGY_WITH, Reason.LUXURY_MORE}
REMOVE_ONLY_REASONS = {Reason.NEEDS, Reason.LUXURY_LESS}
IFFY_REASONS = {Reason.SIDEGRADE}
#: claims that only hold where the related items are present (RW, WA, SY) or
#: where one of them is missing (NW)
PRESENT_CONDITION = {Reason.REDUNDANT_WITH, Reason.WORKS_AGAINST, Reason.SYNERGY_WITH}
ABSENT_CONDITION = {Reason.NEEDS}
#: relative judgements: bind the reviewed build only, nudge worth elsewhere
RELATIVE_REASONS = {Reason.LESS_NECESSARY_SPACE, Reason.LUXURY_LESS, Reason.LUXURY_MORE}


@dataclass
class Item:
    """One catalog entry.

    Requirements, all of which must hold:
      requires      {stat: min}, checked at one stat stage of the build: its
                    pre-shrine or its final stats (Catalog.available_in).
      requires_any  [{stat: min}, ...]: at least one dict holds at that stage.
      clauses       CNF from game data (tri.gamedata): a list of clauses, each a
                    list of atoms {"stats": [...], "value": v, "strict": bool};
                    a clause holds if any atom holds. An atom holds if the sum of
                    its stats at the stage is >= v (a talent taken at pre-shrine
                    stats is kept after the Shrine of Order lowers them). The
                    pseudo-stat "Total" is the sum of the same stage's stats. "strict" is kept
                    as data only: it marks requirements the Shrine of Mastery will
                    not let you reduce below; it does not affect availability.
    prerequisites      item ids that must all be selected.
    prerequisites_any  groups of item ids; each group needs one selected.
    exclusive_with     item ids that cannot be selected together with this one.
    tags               free-form; "type:<T>" (mantra type), "mantra_slots:<n>",
                       "retired" (removed from the game) and "voi_only" (only
                       obtainable in Vow of Iron): never available.
    context            CNF over context tokens ("oath:Arcwarder", "murmur:Tacet"):
                       each clause needs one of its tokens in the build. A clause
                       is only checked when the build records that field (an
                       Oath / Murmur key in Build.meta, or the field in the corpus).
                       From known requirements only (tri.gamedata).

    Availability (Catalog.available_in) also needs every prerequisite (and one
    of each prerequisites_any group) to be available itself, transitively, and
    mantras are unavailable under Catalog.mantra_locks (Silentheart).
    """
    id: str
    kind: Kind
    category: Category = Category.UNKNOWN
    requires: Dict[str, int] = field(default_factory=dict)
    requires_any: List[Dict[str, int]] = field(default_factory=list)
    clauses: List[List[dict]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # other item ids
    prerequisites_any: List[List[str]] = field(default_factory=list)
    exclusive_with: List[str] = field(default_factory=list)  # mechanical exclusion
    tags: List[str] = field(default_factory=list)            # optional typology
    context: List[List[str]] = field(default_factory=list)   # CNF over context tokens


TALENT_CAP_BASE = 76
TALENT_CAP_PER_MANTRA = 2       # each non-exempt mantra lowers the cap by two
# Mantra types (game data "type") that do not lower the talent cap. Oath
# mantras; "Origin" is the game data's type for the faction/origin mantras
# (Grand Warden's Axe, Glorious Charge, Alsin's Aid, Karita Swap, Hidden Blade);
# Monster mantras are exempt per the Deepwoken wiki.
CAP_EXEMPT_TYPES: Tuple[str, ...] = ("Oath", "Origin", "Monster")
#: Item tags that make an item never available: "retired" (removed from the
#: game) and "voi_only" (only obtainable in Vow of Iron; tri reads Pathfinder
#: builds).
NEVER_TAGS: Tuple[str, ...] = ("retired", "voi_only")


@dataclass
class Build:
    """A build as supplied to a reviewer: stats, talents, mantras (+ optional metadata)."""
    highest_stats: Dict[str, int]
    final_stats: Dict[str, int]
    talents: List[str]
    mantras: List[str]
    # Optional metadata (reviews do not need it; your data pipeline may use it).
    pre_stats: Optional[Dict[str, int]] = None   # pre-shrine stats: the other stage that can meet requirements
    build_id: Optional[str] = None
    ruleset: Optional[str] = None
    timestamp: Optional[float] = None   # unix seconds
    author: Optional[str] = None
    popularity: Optional[float] = None  # views/likes, if the site has them
    meta: Optional[Dict[str, object]] = None  # free-form context (oath, murmur, power, ...)
    # Stages between the initial and the final one (dwt.sh: after a Shrine of
    # Order, between Shrines of Mastery), in run order. Requirements may be met
    # at them; the model never learns from them (v2.22).
    mid_stages: Optional[List[Dict[str, int]]] = None

    def items(self) -> List[str]:
        """Talents followed by mantras."""
        return list(self.talents) + list(self.mantras)

    def item_set(self) -> frozenset:
        """frozenset of all selected ids."""
        return frozenset(self.items())

    def stages(self) -> List[Dict[str, int]]:
        """The stat stages at which a requirement can be met, in run order: the
        initial (pre-shrine) stage if known, every middle stage (dwt.sh), and
        the final one. Highest stats are an elementwise maximum of the initial
        and final stages, never held at one time, and are used for learning only."""
        out = []
        if self.pre_stats:
            out.append(dict(self.pre_stats))
        out += [dict(s) for s in (self.mid_stages or []) if s]
        out.append(dict(self.final_stats))
        return out

    def signature(self, stat_round: int = 5) -> str:
        """Exact-duplicate key: sorted selections plus coarsely rounded stats."""
        st = tuple(sorted((k, v // stat_round) for k, v in self.final_stats.items() if v))
        key = json.dumps([sorted(self.talents), sorted(self.mantras), st])
        return hashlib.sha1(key.encode()).hexdigest()


@dataclass
class Operation:
    """One reviewer operation. strength is binary (Strength.WEAK / STRONG)."""
    action: Action
    item_id: str
    strength: Strength = Strength.STRONG
    reason: Optional[Reason] = None
    related: List[str] = field(default_factory=list)  # the "y" in reasons
    group: Optional[str] = None          # edit-group id binding a trade together
    target: Optional[dict] = None        # v3.5: unused (was the audited change); kept so old rows load
    note: Optional[str] = None


@dataclass
class Review:
    """One expert review of one build. Unmarked items are endorsed."""
    review_id: str
    original: Build
    operations: List[Operation]
    reviewer_id: str
    ruleset: str = "unknown"
    timestamp: Optional[float] = None
    # full: slots were full, unmarked items are endorsed (v3.5: the only scope; audits are gone).
    scope: str = "full"
    # v2.17 notes (# or // lines): {"kind": "archetype", "archetype": a, "items": [..]}
    # (components indicative of archetype a), {"kind": "weapons", "classes": [..]} (the
    # weapon classes the build mainly equips) or {"kind": "comment", "text": ..}.
    # Each also keeps the reviewer's "text".
    # v2.18 adds {"kind": "is", "archetype": a} (the reviewed build is a: an anchor
    # in the archetype reading), {"kind": "not", "archetype": a} (it is not a),
    # {"kind": "core", "items": [..]} (components the build stands on, under its own
    # archetype) and {"kind": "avoid", "items": [..]} (items builds like it should not take).
    notes: List[dict] = field(default_factory=list)
    # v3.1.1: items the review says nothing about (an admin purge undid its marks on
    # them): no claim, no endorsement by silence, no operation counts on them.
    neutral: List[str] = field(default_factory=list)

    def note_items(self, kind: str) -> List[dict]:
        return [n for n in (self.notes or []) if n.get("kind") == kind]

    def weapons(self) -> List[str]:
        """Weapon classes the reviewed build equips (Build.meta). The weapons
        note is deprecated (v2.21): stored ones are kept but say nothing."""
        from tri.reviews.authority import build_weapons
        return sorted(build_weapons(self.original))

    # ---- derived views -------------------------------------------------
    def effective(self) -> List[Operation]:
        """Operations as edits (add / remove / iffy)."""
        return list(self.operations)


    def resulting(self) -> Build:
        """Original build with removals applied (additions need the catalog: use apply)."""
        t, m = list(self.original.talents), list(self.original.mantras)
        for op in self.effective():
            if op.action == Action.REMOVE:
                for lst in (t, m):
                    if op.item_id in lst:
                        lst.remove(op.item_id)
            elif op.action == Action.ADD:
                pass  # kind resolved by apply_review with catalog
        import dataclasses
        return dataclasses.replace(self.original, talents=t, mantras=m)

    def apply(self, catalog: Catalog) -> Build:
        """The build after all operations."""
        b = self.resulting()
        for op in self.effective():
            if op.action == Action.ADD:
                lst = b.talents if catalog.kind_of(op.item_id) == Kind.TALENT else b.mantras
                if op.item_id not in lst:
                    lst.append(op.item_id)
        return b

    def iffy(self) -> set:
        """Ids marked iffy."""
        return {o.item_id for o in self.operations if o.action == Action.IFFY}

    def added(self) -> List[Operation]:
        """ADD operations (including upheld adds)."""
        return [o for o in self.effective() if o.action == Action.ADD]

    def removed(self) -> List[Operation]:
        """REMOVE operations (including upheld removes)."""
        return [o for o in self.effective() if o.action == Action.REMOVE]

    def groups(self) -> Dict[str, List[Operation]]:
        """Edit groups. Ungrouped ops are auto-grouped: all 'make space' removals
        pair with all additions (the common one-trade review)."""
        g: Dict[str, List[Operation]] = {}
        auto = "__auto__"
        for o in self.operations:
            if o.action == Action.IFFY:
                continue
            g.setdefault(o.group or auto, []).append(o)
        return g


def off_archetype(op: "Operation") -> bool:
    """A strong scenario-mismatch removal (SM, strong): "this talent does not belong
    on this archetype". It is read as an avoid note for the reviewed build's
    archetype (worth, consensus) and as an OFF_ARCHETYPE review rule scoped to that
    family; a weak SM keeps the old meaning (not for this build's scenario)."""
    return (op.action == Action.REMOVE and op.reason == Reason.SCENARIO_MISMATCH
            and op.strength == Strength.STRONG)


class ReviewError(ValueError):
    """Raised when a review violates the protocol."""
