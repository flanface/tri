"""Pass-ordered inference: complete a build, or review one.

Order of authority, highest first:
  1. review memory    claims from reviews of this or an archetype-similar build
  2. review rules     BROKEN / NO_FIT / REDUNDANT / CONFLICT / PREFER / NECESSARY
                      induced from all reviews (tri.rules)
  3. categories       admin-set item categories (Tri.set_categories):
                      foundational = taken on every build (missing ones that
                      the stats cannot reach are reported), central = taken
                      whenever its requirements are met, luxury = never
                      required, taken when a slot is free, never dropped or
                      swapped out for worth. They replace the mined pass of
                      the item and outrank its never rules.
  4. pass rules       certified take rules (foundational, then central) and
                      never rules (exclusions) (tri.passes)
  5. worth            pass 3 luxury: the worth field plus nearby builds (tri.worth)

Unrated items (added to the game after the corpus, Tri.extend_catalog) have no
evidence until reviews or post-release builds give them some (support below
cold_support): worth never adds, drops, swaps or marks them iffy; reviews,
review rules and categories still can.

Sidegrades (tri.sidegrades): worth never adds an item of the side of a
sidegrade the answer holds fewer of, and an answer committed to one side
lists the other side in Alternative.sidegrades.

complete(build): keeps what is there, then fills each slot type in pass
order up to capacity (default: the median count among similar builds).

review(build): slots start full, so the answer is the smallest edit:
  - drop what memory or review rules reject;
  - add missing foundational / central picks (and memory adds), displacing
    the least worthy luxury picks;
  - swap a luxury pick for a clearly worthier one (gain >= swap_margin);
  - mark luxury picks iffy when the best alternative is nearly as worthy;
  - repair: re-check every rule against the result (a swap can add an anchor
    or an exclusion's trigger) until no take rule is missing and no never
    rule is broken, or report what could not be satisfied (Alternative.violations).
Each change says which pass or review made it, and why."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from tri.core.catalog import Catalog
from tri.core.types import Kind
from tri.infer.completion import _Completion
from tri.infer.constraints import _Constraints
from tri.infer.defs import GLOBAL_RULE_KINDS, InferConfig
from tri.infer.reviewing import _Reviewing
from tri.infer.situations import _Situations
from tri.rules.passes import NEVER, PassRule, Rulebook


class PassEngine(_Situations, _Constraints, _Completion, _Reviewing):
    def __init__(self, catalog: Catalog, space, rulebook: Rulebook, worth, review_rules: Sequence = (),
                 ledger=None, cfg: Optional[InferConfig] = None):
        self.c = catalog
        self.space, self.rb, self.worth = space, rulebook, worth
        # Only context-free review rules generalise globally (BROKEN, REDUNDANT
        # with a co-picked item, CONFLICT with a co-picked item). Archetype-relative
        # judgements (necessary / no synergy / prefer) spread through concept-space
        # proximity in the ledger and through refit, never as global rules.
        self.review_rules = [r for r in review_rules if r.status in ("supported", "refined")
                             and r.kind in GLOBAL_RULE_KINDS]
        self.ledger = ledger
        self.cfg = cfg or InferConfig()
        self.manual: Dict[str, str] = {}               # item -> foundational | central | luxury (admin)
        self.unrated_items: Set[str] = set()           # items added after the corpus (Tri.extend_catalog)
        self.sidegrades: list = []                     # [Sidegrade] (tri.sidegrades)
        self.note_worth: List[dict] = []               # indicator / core / avoid notes (Tri._sync_indicators)
        self.values: Dict[str, float] = {}             # admin flat value offsets (Tri.set_values)
        self.locset = None
        self.class_prior = None                        # tri.focus.ClassPrior from the corpus (Tri.engine)
        self.solo: Dict[str, float] = {}               # prerequisite -> corpus share without its dependants (v3.2)
        self.is_tal = np.array([catalog.items[i].kind == Kind.TALENT for i in catalog.ids])
        # talent cap: talents <= talent_cap_base - cap_per_mantra * (mantras not of an exempt type)
        self._talents = {i for i in catalog.ids if catalog.items[i].kind == Kind.TALENT}
        self._lowers = {i for i in catalog.ids if catalog.lowers_cap(i)}
        self._cap_base = int(getattr(catalog, "talent_cap_base", 76))
        self._cap_per = int(getattr(catalog, "cap_per_mantra", 2))
        self._never: Dict[str, List[PassRule]] = {}       # item -> never rules on it
        self._never_by_anchor: Dict[str, List[PassRule]] = {}   # item -> never rules it triggers
        for r in rulebook.rules:
            if r.effect == NEVER and r.status == "active":
                self._never.setdefault(r.item, []).append(r)
                for c in r.conds:
                    if c.kind == "item" and c.op == "has":
                        self._never_by_anchor.setdefault(c.key, []).append(r)
