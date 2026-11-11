"""Tri: the facade. Fit on builds, take reviews, answer.

    tri = Tri(catalog)
    tri.fit(builds, weights)            # concept space + pass rules + worth (seconds to minutes)
    tri.add_reviews(reviews)            # instant: memory, review rules, worth neighbours
    tri.refit()                         # re-mine passes and the space with reviews included
    tri.review(build)                   # [Alternative]: smallest edits, per plausible archetype
    tri.complete(build)                 # Alternative: fill empty slots in pass order
    tri.worth(build, "Item")            # worth of one item here, its pass, its rules
    tri.concept(build)                  # coordinates, regions, archetype, ambiguity, nearest reviews
    tri.add_locators([locator])         # instant: administrator-labelled builds name archetypes
    tri.archetype(build)                  # the archetype a build reads as (from the locators)
    tri.apply_patch(patch) / tri.migrate(new_catalog, renames)
    tri.extend_catalog(new_catalog)     # new game items without a refit (unrated until reviewed)
    tri.set_categories({"Item": "foundational"})   # admin categories: foundational | central | luxury
    tri.sidegrades                      # mined either/or clusters per archetype (tri.sidegrades)
    tri.save(path) / Tri.load(path, catalog)

Three parts (docs/Home.md): the worth field (tri.space, a logistic
factorisation trained by gradient descent; its coordinates measure worth
differences), certified rules (tri.passes, conjunctions mined by shallow trees
and certified on held-out halves), and pass-ordered inference with constraint
repair (tri.infer). Reviews outrank all three. Locators (tri.locators) name
archetypes and validate the geometry; they never train anything."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from tri.analysis.defs import Sidegrade
from tri.core.catalog import Catalog
from tri.core.types import Reason, Review
from tri.engine.answering import _Answering
from tri.engine.archetypes import _Archetypes
from tri.engine.defs import TriConfig
from tri.engine.fitting import _Fitting
from tri.engine.persist import _Persist
from tri.engine.reviewing import _Reviewing
from tri.game.patch import PatchLog
from tri.model.matrix import BuildMatrix
from tri.model.space import ConceptSpace
from tri.model.worth import FieldWorth
from tri.reviews.authority import ReviewLedger
from tri.reviews.defs import Locator
from tri.reviews.locators import LocatorSet
from tri.rules.authored import AuthoredRules
from tri.rules.passes import Rulebook


class Tri(_Fitting, _Archetypes, _Reviewing, _Answering, _Persist):
    def __init__(self, catalog: Catalog, cfg: Optional[TriConfig] = None):
        self.catalog = catalog
        self.cfg = cfg or TriConfig()
        self.data: Optional[BuildMatrix] = None       # corpus builds kept for refits and worth
        self.space: Optional[ConceptSpace] = None
        self.rulebook: Optional[Rulebook] = None       # mined rules
        self.selection: Optional[Rulebook] = None      # what decides F/C picks: mined + authored + reviews (v3.0)
        self.authored = AuthoredRules()                 # rules written in the rule language (v3.0)
        self.reviews: List[Review] = []
        self.patches = PatchLog()
        self.ledger: Optional[ReviewLedger] = None
        self.review_rules: list = []
        self.review_rules_all: list = []            # v3.4: before drop_rule's rv: ids
        self.sidegrades_dropped: set = set()
        self._worth: Optional[FieldWorth] = None
        self._Zdata: Optional[np.ndarray] = None
        self.fitted_at: Optional[float] = None
        self.locators: List[Locator] = []
        self.locset: Optional[LocatorSet] = None
        self.metric_selection: Optional[dict] = None   # the last leave-one-out choice of metric_power
        self.held_out_ids: List[str] = []               # locator builds kept out of the last field fit
        self.manual: Dict[str, str] = {}                # admin categories: item -> foundational | central | luxury
        self.values: Dict[str, float] = {}              # admin flat value offsets: item -> delta on worth (v2.18)
        self._note_worth: List[dict] = []               # indicator / core / avoid notes, for worth (v2.18)
        self.sidegrades: List[Sidegrade] = []
        self.load_info: Dict[str, str] = {}

    _CONSENSUS_REMOVE = (Reason.NO_SYNERGY, Reason.WORKS_AGAINST, Reason.BROKEN, Reason.SCENARIO_MISMATCH)

    # ------------------------------------------------------------ load cache
    LATENTS = "latents.npy"
