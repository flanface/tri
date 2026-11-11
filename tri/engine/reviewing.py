"""Tri methods: taking reviews: the ledger, induced rules, memory, categories and value overrides."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from tri.core.codec import validate_review
from tri.core.types import Build, Review
from tri.engine.defs import CATEGORIES
from tri.reviews.authority import ReviewLedger
from tri.reviews.defs import Locator
from tri.rules.induction import RuleInducer


class _Reviewing:
    """Tri methods (Tri methods: taking reviews: the ledger, induced rules, memory, categories and value overrides.)."""

    def _ledger_and_rules(self) -> None:
        from tri.game.weapons import with_weapon_notes
        for r in self.reviews:                     # v3.3: # weapon notes set the reviewed build's equips
            with_weapon_notes(r)
        self._note_stated()                  # v3.0.1: the families reviewers state
        emb = self._embed if self.space is not None else None
        if emb is not None:
            self._place_many([r.original for r in self.reviews])      # v3.1: one solve, not one per review
        kern = ReviewLedger.reach_kernel(self.cfg.ledger) if self.space is not None else None
        if kern is not None and self._barrier_on():
            kern = self.locset.kernel(kern)
        many = None
        if kern is not None and self._barrier_on():
            h2 = self.cfg.ledger.reach ** 2 / (-2.0 * np.log(self.cfg.ledger.sim_min))
            many = self.locset.kernel_many(h2)
        fam = self._family_many() if emb is not None else None
        self.ledger = ReviewLedger(self.reviews, self.catalog, self.cfg.ledger, embed=emb,
                                   kernel=kern, kernel_many=many, family_many=fam,
                                   change_times=self.patches.change_times()) if self.reviews else None
        ind = RuleInducer(self.catalog, authority_min_support=self.cfg.rule_min_support,
                          features=self._region_feats if self.space is not None else None)
        self.review_rules_all = ind.induce(self.reviews, ledger=self.ledger) if self.reviews else []
        self._apply_drops()
        self._sync_indicators()
        self._select()

    def _family_many(self):
        """(u, V) -> [m] family overlap for the ledger's family reach (math §5.8):
        the locators' family posteriors where both builds are located in one
        family; for two builds no locator places, cfg.ledger.fam_region when they
        share a concept region; else 0 (never across a located border)."""
        ls, sp_, w = self.locset, self.space, float(self.cfg.ledger.fam_region)
        has_loc = ls is not None and len(getattr(ls, "locators", []) or [])
        if sp_ is None:
            return None
        memo: dict = {}

        def fam(u, V):
            u = np.asarray(u, float); V = np.atleast_2d(V)
            O = ls.family_overlap(u, V) if has_loc else np.zeros(len(V))
            if w > 0:
                key = (V.shape, hash(V.tobytes()))
                if key not in memo:
                    memo.clear()
                    memo[key] = (np.asarray(sp_.region_of(V)), ls.located_many(V) if has_loc else np.zeros(len(V), bool))
                rv, lv = memo[key]
                lu = bool(ls.located_many(u[None])[0]) if has_loc else False
                if not lu:
                    ru = int(sp_.region_of(u[None])[0])
                    O = np.where((O <= 0) & ~lv & (rv == ru), w, O)
            return O
        return fam

    # ------------------------------------------------------------ selection rules (v3.0)
    @property
    def rules(self):
        """The selection rule set (mined + authored + review evidence), else the mined rulebook."""
        return self.selection if self.selection is not None else self.rulebook

    def rule_vocabulary(self):
        from tri.reviews.variants import family_key
        from tri.rules.language import Vocabulary
        rb = self.rulebook
        arch = sorted({family_key(n) for n in (self.locset.names.values() if self.locset is not None else ())})
        toks = set()
        if self.data is not None:
            toks |= set(self.data.tokens)
        for r in self.reviews[-500:]:
            from tri.rules.groups import rule_tokens
            toks |= rule_tokens(r.original)
        return Vocabulary(self.catalog, groups=(rb.groups if rb is not None else {}), archetypes=arch, tokens=toks)

    def _select(self) -> None:
        if self.rulebook is None:
            self.selection = None
            return
        from tri.rules.groups import derived_groups
        from tri.rules.selection import build_selection
        if not self.rulebook.groups and self.cfg.passes.groups:      # runs mined before v3.0
            self.rulebook.groups = derived_groups(self.catalog)
        if self.rulebook.reader is None and self.space is not None:
            self.rulebook.reader = self._archetype_reader
        fam = self._family_of if self.space is not None else None
        self.selection = build_selection(self.rulebook, self.authored, self.rule_vocabulary(), self.reviews,
                                         self.ledger, fam, self.data, self.cfg.selection)

    def selection_report(self) -> dict:
        from tri.rules.selection import selection_report
        return selection_report(self.rules) if self.rules is not None else {}

    def add_rule(self, text: str, author: str = "", note: str = "") -> dict:
        """Add a rule written in the rule language (tri.rules.language). Returns the
        entry, the rule in symbols and its corpus statistics (shown, never a gate)."""
        e = self.authored.add(text, self.rule_vocabulary(), author=author, note=note)
        self._select(); self._refresh_worth()
        return self._rule_info(e.rule_id)

    def edit_rule(self, rule_id: str, text: str, author: str = "", note: str = "") -> dict:
        """Replace an authored rule, or override a mined one (suppressed, the new text added)."""
        e = self.authored.edit(rule_id, text, self.rule_vocabulary(), author=author, note=note)
        self._select(); self._refresh_worth()
        return self._rule_info(e.rule_id)

    def drop_rule(self, rule_id: str, author: str = "", note: str = "") -> dict:
        """Remove an authored rule, suppress a mined one, or lift a suppression, by id.
        v3.4: also review rules (rv:...) and sidegrades (sg:...), switched off
        and on the same way."""
        from tri.rules.registry import PREFIXES
        e = self.authored.get(rule_id)
        if rule_id.startswith(PREFIXES):
            if e is not None and e.op == "suppress":
                self.authored.remove(rule_id)
                what = "lifted"
            elif not any(x["id"] == rule_id for x in self.all_rules()):
                raise KeyError(f"no rule {rule_id}")
            else:
                self.authored.suppress(rule_id, author=author, note=note)
                what = "suppressed"
            self._apply_drops(); self._refresh_worth()
            return dict(id=rule_id, done=what)
        if e is not None and e.op == "add":
            self.authored.remove(rule_id)
            what = "removed"
        elif e is not None and e.op == "suppress":
            self.authored.remove(rule_id)
            what = "lifted"
        else:
            if self.find_rule(rule_id) is None:
                raise KeyError(f"no rule {rule_id}")
            self.authored.suppress(rule_id, author=author, note=note)
            what = "suppressed"
        self._select(); self._refresh_worth()
        return dict(id=rule_id, done=what)

    def set_authored(self, entries) -> None:
        """Replace the authored rules (e.g. from Supabase)."""
        from tri.rules.authored import AuthoredRules
        self.authored = entries if isinstance(entries, AuthoredRules) else AuthoredRules.from_json(entries)
        self._apply_drops(); self._select(); self._refresh_worth()

    def find_rule(self, rule_id: str):
        from tri.rules.language import rule_id as rid
        for r in (self.rules.rules if self.rules is not None else []):
            if rid(r.key) == rule_id:
                return r
        return None

    def _rule_info(self, rule_id: str) -> dict:
        from tri.rules.language import rule_id as rid
        r = self.find_rule(rule_id)
        if r is None:
            return dict(id=rule_id)
        ev = (getattr(self.selection, "evidence", {}) or {}).get(r.key, {})
        return dict(id=rid(r.key), rule=r.text(self.rules.region_names), status=ev.get("status", r.status),
                    corpus=ev.get("corpus"), support=ev.get("support", []), counter=ev.get("counter", []))

    # ------------------------------------------------------------ every rule (v3.4)
    def dropped_ids(self) -> set:
        """Ids switched off with drop_rule (selection, rv:, sg:)."""
        return set(self.authored.suppressed) if getattr(self, "authored", None) is not None else set()

    def _apply_drops(self) -> None:
        """Review rules and sidegrades minus the dropped ids."""
        from tri.rules.registry import review_rule_id
        d = self.dropped_ids()
        allr = getattr(self, "review_rules_all", None)
        if allr is None:
            allr = self.review_rules_all = list(self.review_rules)
        self.review_rules = [r for r in allr if review_rule_id(r) not in d]
        self.sidegrades_dropped = {x for x in d if x.startswith("sg:")}

    def all_rules(self, item: Optional[str] = None, build: Optional[Build] = None) -> List[dict]:
        """Every rule that bears on a pick other than worth (tri.rules.registry):
        selection rules (also where item is only a condition), review rules,
        sidegrades, categories, weapon rules, game data,
        and with a build the class focus and the review claims that bind there,
        each marked droppable or not and, with a build, whether it fires."""
        from tri.rules.registry import collect
        return collect(self, item, build)

    def list_rules(self, item: Optional[str] = None, status: Optional[str] = None, source: Optional[str] = None,
                   build: Optional[Build] = None, every: bool = False) -> List[dict]:
        """Selection rules as {id, rule, status, source, support, counter}.
        every=True (v3.4): all_rules(item, build) filtered by status / source."""
        if every or build is not None:
            out = self.all_rules(item, build)
            return [e for e in out if (not status or str(e["status"]).startswith(status))
                    and (not source or e["source"] == source)]
        from tri.rules.language import rule_id as rid
        if self.rules is None:
            return []
        ev = getattr(self.selection, "evidence", {}) or {}
        out = []
        for r in self.rules.rules:
            if item and r.item != item:
                continue
            e = ev.get(r.key, {})
            st = e.get("status", r.status)
            if status and st != status:
                continue
            if source and r.source != source:
                continue
            out.append(dict(id=rid(r.key), rule=r.text(self.rules.region_names), status=st, source=r.source,
                            support=len(e.get("support", [])), counter=len(e.get("counter", []))))
        return out
    # ------------------------------------------------------------ reviews

    def validate(self, review: Review) -> List[str]:
        return validate_review(review, self.catalog)

    def set_memory(self, reviews: Sequence[Review], locators: Sequence[Locator]) -> dict:
        """set_reviews and set_locators in one step: the review ledger, rules and
        worth are rebuilt once instead of twice (what a bot reload needs)."""
        seen: Dict[str, Locator] = {}
        for l in locators:
            seen[l.locator_id] = l
        self.locators = list(seen.values())
        self.reviews = list(reviews)
        if self.space is not None:
            self._locset(); self._name_regions(); self._ledger_and_rules(); self._refresh_worth()
        return self.locator_report()

    def set_reviews(self, reviews: Sequence[Review]) -> None:
        """Replace the review log (e.g. everything in Supabase). Instant."""
        self.reviews = list(reviews)
        if self.space is not None:
            self._ledger_and_rules(); self._refresh_worth()

    def add_reviews(self, reviews: Sequence[Review], validate: bool = True) -> dict:
        """Add reviews. Takes effect at once (memory, review rules, worth
        neighbours); pass rules and the space change at the next refit()."""
        if validate:
            for r in reviews:
                self.validate(r)
        new = {r.review_id for r in reviews}           # same id again = the reviewer's corrected version
        self.set_reviews([r for r in self.reviews if r.review_id not in new] + list(reviews))
        return self.ledger_report()

    def ledger_report(self) -> dict:
        return self.ledger.report() if self.ledger is not None else dict(reviews=0)

    def rule_report(self, include_proposed: bool = False) -> List[str]:
        return [r.describe() for r in self.review_rules if include_proposed or r.status != "proposed"]
    # ------------------------------------------------------------ convergence

    def review_agreement(self, detail: bool = False):
        """How many claims of the active reviews tri reproduces on their own
        builds: removed items it removes, added items it adds, kept items it
        keeps. 1.0 = every review is reproduced."""
        if self.ledger is None:
            return (1.0, []) if detail else 1.0
        rows = []
        for r in self.ledger.reviews:
            if self.ledger.weights[r.review_id].review <= 0:
                continue
            alt = self.review(r.original, 1, keep_size=True)[0]
            res = set(alt.items())
            hits = n = 0
            for c in self.ledger.active_claims(r.review_id):
                if c.label == "iffy":
                    continue
                n += 1
                hits += (c.item not in res) if c.label == "remove" else (c.item in res)
            rows.append(dict(review_id=r.review_id, agreement=hits / n if n else 1.0, claims=n))
        a = float(np.mean([x["agreement"] for x in rows])) if rows else 1.0
        return (a, rows) if detail else a

    def dry_insert(self, review: Review, probe_builds: Sequence[Build] = ()) -> dict:
        """What one review would change, without keeping it: answers on its own
        build and on probe builds before and after, plus the overrides it causes.
        Not thread-safe (temporarily swaps the review log)."""
        self._need()
        keep = list(self.reviews)
        builds = [review.original] + list(probe_builds)

        def snap():
            out = []
            for b in builds:
                a = self.review(b, 1, keep_size=True)[0]
                out.append(dict(build=b.build_id, changes=[f"{c.action} {c.item}" for c in a.changes],
                                iffy=list(a.iffy)))
            return out
        before = snap()
        before_super = len(self.ledger.superseded) if self.ledger else 0
        try:
            self.set_reviews(keep + [review])
            after = snap()
            overrides = [s.__dict__ for s in (self.ledger.superseded[before_super:] if self.ledger else [])]
            rules = [r.describe() for r in self.review_rules if r.status != "proposed"
                     and any(x.review_id == review.review_id for x in r.support + r.counter)]
            agree = self.review(review.original, 1, keep_size=True)[0]
        finally:
            self.set_reviews(keep)
        return dict(before=before, after=after, overrides=overrides, rules_touched=rules,
                    reproduced=sorted(set(agree.items())), validation=self.validate(review))
    # ------------------------------------------------------------ review queue

    def next_reviews(self, pool: Sequence[Build], k: int = 10, random_share: float = 0.2, seed: int = 0):
        """Which pool builds experts should review next (active.ReviewQueue)."""
        from tri.infer.active import ReviewQueue
        return ReviewQueue(self, random_share, seed).next(pool, k)
    # ------------------------------------------------------------ categories

    def categories(self) -> Dict[str, str]:
        """item -> foundational | central | luxury: the admin category where one
        is set (set_categories), else the mined pass."""
        self._need()
        return {i: self.manual.get(i) or self.rules.pass_of(i) for i in self.catalog.ids}

    def set_categories(self, categories: Dict[str, Optional[str]], replace: bool = False) -> Dict[str, str]:
        """Admin categories, the expert's say on an item for every build:
          foundational  taken on every build; if its requirements are not met
                        at the pre-shrine or final stats, the answer reports it
                        as a missing pick instead of silently skipping it
          central       defines the build's playstyle (an attunement path, a
                        signature talent such as Blighted Song): taken whenever its
                        requirements are met and respected once picked, removed
                        only for a clash (math §7.10). Few talents are central.
          luxury        never required; taken when a slot is free; never
                        dropped or swapped out for worth
        A category replaces the mined pass of the item and outranks its never
        rules; binding review claims still outrank it (a review that removes
        the item on a similar build wins there). None clears an item.
        replace=True drops every category not in `categories`. Instant."""
        new = {} if replace else dict(self.manual)
        for it, c in categories.items():
            if c is None or c == "":
                new.pop(it, None)
                continue
            c = str(c).lower()
            if c not in CATEGORIES:
                raise ValueError(f"{it}: category is one of {', '.join(CATEGORIES)}, not {c}")
            if it not in self.catalog.index:
                raise ValueError(f"{it} is not in the catalog")
            new[it] = c
        self.manual = new
        return dict(self.manual)

    def set_values(self, values: Dict[str, Optional[float]], replace: bool = False) -> Dict[str, float]:
        """Admin flat value offsets: delta added to an item's worth on every build
        (math §5.6), within +-cfg.infer.value_cap. None or 0 clears an item.
        replace=True drops every value not in `values`. Instant."""
        cap = float(self.cfg.infer.value_cap)
        new = {} if replace else dict(self.values)
        for it, v in values.items():
            if v is None or float(v) == 0.0:
                new.pop(it, None)
                continue
            if it not in self.catalog.index:
                raise ValueError(f"{it} is not in the catalog")
            v = float(v)
            if not abs(v) <= cap:
                raise ValueError(f"{it}: a value is within +-{cap:g}, not {v:g}")
            new[it] = v
        self.values = new
        return dict(self.values)
