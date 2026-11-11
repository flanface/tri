"""Tri methods: answers: the pass engine, completion, review, worth and explanations."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.types import Build
from tri.version import VERSION
from tri.infer.caution import Answers
from tri.infer.defs import Alternative
from tri.infer.engine import PassEngine
from tri.rules.passes import CENTRAL, FOUNDATIONAL, LUXURY


class _Answering:
    """Tri methods (Tri methods: answers: the pass engine, completion, review, worth and explanations.)."""

    # ------------------------------------------------------------ answers
    def engine(self) -> PassEngine:
        self._need()
        e = PassEngine(self.catalog, self.space, self.rules, self._worth, self.review_rules,
                       self.ledger, self.cfg.infer)
        e.manual = {i: c for i, c in self.manual.items() if i in self.catalog.index}
        e.locset = self.locset
        e.stated = self.stated_by_review()           # v3.6: claims keep to their stated family
        e.unrated_items = {i for i in self.patches.release_times() if i in self.catalog.index}
        from tri.rules.registry import sidegrade_id
        gone = getattr(self, "sidegrades_dropped", set())
        e.sidegrades = [g for g in self.sidegrades if all(i in self.catalog.index for i in g.a + g.b)
                        and sidegrade_id(g) not in gone]
        e.note_worth = list(getattr(self, "_note_worth", []) or []) + list(getattr(self, "_answer_notes", []) or [])
        e.values = {i: float(v) for i, v in (getattr(self, "values", None) or {}).items() if i in self.catalog.index}
        e.rule_home = self._rule_homes()
        e.class_prior = self._class_prior()
        e.solo = self._prereq_solo()
        return e

    def _prereq_solo(self):
        """item -> share of the corpus builds with the prerequisite item that
        take none of the talents it unlocks (math §2.8d). Prerequisites seen in
        fewer than 5 builds are left out. Cached per corpus."""
        if self.data is None:
            return {}
        tag = (id(self.data), id(self.catalog))
        if getattr(self, "_solo_tag", None) == tag:
            return self._solo
        c = self.catalog
        deps = {}
        for x in c.ids:
            it = c.items[x]
            for p in set(it.prerequisites) | {y for g in it.prerequisites_any for y in g}:
                if p in c.index:
                    deps.setdefault(p, []).append(c.index[x])
        X = self.data.X.tocsc()
        w = np.asarray(self.data.w, dtype=float)
        out = {}
        for p, ds in deps.items():
            ds = [d for d in ds if d < X.shape[1]]
            if c.index[p] >= X.shape[1] or not ds:
                continue
            rows = X[:, c.index[p]].nonzero()[0]
            if len(rows) < 5:
                continue
            has = np.asarray(X[rows][:, ds].sum(axis=1)).ravel() > 0
            tot = w[rows].sum()
            if tot > 0:
                out[p] = float(w[rows][~has].sum() / tot)
        self._solo_tag, self._solo = tag, out
        return out

    def _class_prior(self):
        """Weapon-class prior from the corpus's weapon context (tri.focus,
        math §2.10). Built from the stored corpus on load or (re)fit; no
        refit. Cached per (corpus, coordinates)."""
        if not self.cfg.infer.focus or self.data is None or self._Zdata is None:
            return None
        tag = (id(self.data), id(self._Zdata), id(self.catalog))
        c = self.__dict__.get("_class_prior_cache")
        if c is not None and c[0] == tag:
            return c[1]
        from tri.infer.focus import ClassPrior
        try:
            cp = ClassPrior.from_matrix(self.data, self._Zdata, self.catalog, self.cfg.infer.focus_k)
        except Exception:
            cp = None
        self.__dict__["_class_prior_cache"] = (tag, cp)
        return cp

    def _rule_homes(self) -> Dict[str, set]:
        """Home families of the mined central take rules (math §5.4, v2.19): on a
        reference sample, the families whose located builds the rule covers with at
        least rule_home_min weight and among which the item is still taken at rate
        theta_take. A build located elsewhere is across the chasm: the rule does not
        fire there (local worth still decides the item). Cached per locator set."""
        ls, lc = self.locset, self.cfg.locators
        if ls is None or len(ls.keys) < 2 or not lc.chasm_rules or self.data is None or self._Zdata is None:
            return {}
        tag = (id(ls), id(self.rules), len(self.rules.rules), len(self.data))
        c = self.__dict__.get("_home_cache")
        if c is not None and c[0] == tag:
            return c[1]
        n = len(self.data)
        rows = np.arange(n)
        if n > lc.chasm_rule_sample:
            rows = np.sort(np.random.default_rng(self.cfg.seed).choice(n, lc.chasm_rule_sample, replace=False))
        bm, Z = self.data.subset(rows), self._Zdata[rows]
        P, d = [], []
        for i in range(0, len(Z), 4096):
            p_, d_, _ = ls.read_many(Z[i:i + 4096]); P.append(p_); d.append(d_)
        P, d = np.vstack(P), np.concatenate(d)
        fam = P.argmax(1)
        loc = (d <= ls.reach) & (P[np.arange(len(P)), fam] >= lc.min_confidence)
        w = bm.w
        th, mmin = float(self.cfg.passes.theta_take), float(lc.chasm_rule_min)
        from tri.reviews.variants import family_key
        out: Dict[str, set] = {}
        for r in self.rules.rules:
            if r.pass_ != CENTRAL or r.effect != "take" or r.status != "active" or r.item not in self.catalog.index:
                continue
            if r.source in ("authored", "review") or any(c.key.startswith("archetype:") for c in r.conds):
                continue                               # v3.0: an author, the reviews or an A literal already scope it
            m = self.rules.rule_mask(bm, r) & loc
            if w[m].sum() < mmin:
                continue
            y = np.asarray(bm.X[:, self.catalog.index[r.item]].todense()).ravel() > 0
            cov = np.bincount(fam[m], weights=w[m], minlength=len(ls.keys))
            tak = np.bincount(fam[m & y], weights=w[m & y], minlength=len(ls.keys))
            home = {ls.keys[k] for k in range(len(ls.keys)) if cov[k] >= mmin and tak[k] >= th * cov[k]}
            ev = (getattr(self.selection, "evidence", None) or {}).get(r.key) or {}
            fams = set(ev.get("families", ()))
            home |= {k for k in ls.keys if family_key(ls.names.get(k, k)) in fams}   # v3.0.1: reviewers kept it there
            out[r.key] = home
        self.__dict__["_home_cache"] = (tag, out)
        return out

    def review(self, build: Build, max_alternatives: int = 3, region: Optional[int] = None,
               locked: Sequence[str] = (), archetype: Optional[str] = None,
               keep_size: bool = False, answers=None, withheld: Sequence[str] = ()) -> List[Alternative]:
        """Smallest edits to a full build, one Alternative per plausible archetype
        (best first). locked: items never removed (pinned by the user).
        archetype: analyse the build as this archetype only (see force()).
        The answer fills the build's capacity (empty mantra slots, then talents
        up to the cap) before any swap; keep_size: keep the input's talent and
        mantra counts instead (an input above the cap or its slots is still
        trimmed: the cap is strict). answers: the asker's answers to caution
        questions (tri.infer.caution.Answers or its dict); they act on this call only.
        withheld: items not added in this call (held for thin evidence, v3.1.1)."""
        e, archetype, locked = self._answered(build, archetype, locked, answers)
        e.withheld = set(withheld) | Answers.of(answers).unwanted
        e.wanted = Answers.of(answers).wanted
        return e.review(build, max_alternatives, region, locked, force=self.force(archetype), keep_size=keep_size)

    def complete(self, build: Build, capacity: Optional[Tuple[int, int]] = None,
                 region: Optional[int] = None, archetype: Optional[str] = None, answers=None,
                 withheld: Sequence[str] = ()) -> Alternative:
        """Fill a partial build in pass order up to the talent cap and the
        mantra slots (math §2.6). capacity=(talents, mantras) only lowers
        them; either may be None. archetype: complete it as this archetype."""
        e, archetype, _ = self._answered(build, archetype, (), answers)
        e.withheld = set(withheld) | Answers.of(answers).unwanted
        e.wanted = Answers.of(answers).wanted
        return e.complete(build, capacity, region, force=self.force(archetype))

    def _answered(self, build: Build, archetype, locked, answers):
        """(engine, archetype, locked) with the caution answers applied locally (math §14.3):
        an archetype answer forces it; "central" locks the pick and reads it as central here."""
        from tri.infer.caution import Answers
        a = Answers.of(answers)
        e = self.engine()
        if archetype is None:
            archetype = a.archetype or (self._answer_book.archetype_for(build)
                                        if getattr(self, "_answer_book", None) is not None else None)
        yes = [i for i, v in a.central.items() if v and i in self.catalog.index]
        if yes:
            e.manual = dict(e.manual, **{i: CENTRAL for i in yes if i in build.items() or e.manual.get(i) != FOUNDATIONAL})
            locked = tuple(locked) + tuple(i for i in yes if i in build.items())
        if a.weapon:                                    # v3.2: a weapon name is read as equipped; a class sets the focus
            from tri.game.weapons import resolve
            from tri.infer.focus import norm_class
            names = resolve(a.weapon)
            if names:
                e.weapon_name = names[0]
            elif norm_class(a.weapon):
                e.weapon_answer = norm_class(a.weapon)
        return e, archetype, locked

    def caution(self, build: Build, result=None, answers=None, archetype: Optional[str] = None,
                mode: str = "review"):
        """Caution on an answer (math §14): which worth marks to hold back and which
        questions to ask. result: review alternatives or a completion (computed if None)."""
        from tri.infer.caution import assess
        if result is None:
            result = self.review(build, archetype=archetype, answers=answers) if mode == "review" \
                else self.complete(build, archetype=archetype, answers=answers)
        alts = result if isinstance(result, list) else [result]
        return assess(self, build, alts, answers, mode, archetype)

    def settle(self, build: Build, mode: str = "review", archetype: Optional[str] = None, answers=None,
               max_alternatives: int = 3, capacity: Optional[Tuple[int, int]] = None, rounds: int = 4,
               result=None):
        """(result, Caution) with held marks settled (v3.1.1, math §14.1). A held
        change is not just hidden: a held removal keeps the pick (locked) and a
        held add is not taken (withheld), and the answer is solved again, so the
        slot goes to the next pick and the answer still fills the cap and the
        mantra slots. Thin changes of the re-solved answer are held in turn, up
        to `rounds` times; after that the answer stands as solved. Only the
        first alternative is settled. Questions are those of every round."""
        from tri.infer.caution import Caution
        solve = (lambda keep, ban: self.review(build, max_alternatives, archetype=archetype, answers=answers,
                                               locked=tuple(keep), withheld=tuple(ban))) if mode == "review" else \
            (lambda keep, ban: self.complete(build, capacity, archetype=archetype, answers=answers,
                                             withheld=tuple(ban)))
        res = solve((), ()) if result is None else result
        c = self.caution(build, res, answers, archetype, mode)
        if not c.held:
            return res, c
        held, qs, keys = [], list(c.questions), {q.key for q in c.questions}
        keep, ban = set(), set()
        first = res
        for _ in range(max(int(rounds), 0)):
            if not c.held:
                break
            held += c.held
            keep |= {h["item"] for h in c.held if h["action"] == "remove"}
            ban |= {h["item"] for h in c.held if h["action"] == "add"}
            res = solve(keep, ban)
            if isinstance(first, list) and isinstance(res, list):
                res = res[:1] + list(first[1:])          # the other archetypes stay as they were
            c = self.caution(build, res, answers, archetype, mode)
            qs += [q for q in c.questions if q.key not in keys]
            keys |= {q.key for q in c.questions}
        info = dict(c.info, settled=dict(kept=sorted(keep), withheld=sorted(ban)))
        return res, Caution(not held and not qs, held, qs, info)

    def set_answers(self, rows: Sequence[dict]) -> None:
        """Answers from all users ({user, build, kind archetype|central, item, value, ts}).
        Only consensus reaches the model (AnswerBook)."""
        from tri.infer.caution import AnswerBook
        self.answers = list(rows)
        self._answer_book = AnswerBook(self.answers, self.cfg.caution)
        self._answer_notes = self._answer_book.central_notes(self) if self.space is not None else []

    def as_review(self, build: Build, archetype: Optional[str] = None, emit_margin: Optional[float] = None,
                  strong_margin: Optional[float] = None, review_id: str = "tri-draft", answers=None,
                  caution: Optional[bool] = None):
        """The analysis as a review sheet in the reply grammar (tri.draft.DraftSheet:
        .review validated by codec.validate_review, .lines, .withheld). caution: hold
        thin worth marks back into .withheld (default cfg.caution.drafts)."""
        from tri.infer.draft import as_review
        return as_review(self, build, archetype, emit_margin, strong_margin, review_id, answers=answers,
                         caution=self.cfg.caution.drafts if caution is None else caution)

    def worth(self, build: Build, item: str) -> dict:
        """Worth of one item in this build: local pick rate, rank among the
        same kind, best replacement, its pass and the rules naming it."""
        return self._worth_rows(build, [item])[0]

    def build_worth(self, build: Build, archetype: Optional[str] = None) -> List[dict]:
        """worth() for every item of the build, least worthy first (one
        situation for the whole build; archetype: as that forced archetype)."""
        out = self._worth_rows(build, [i for i in build.items() if i in self.catalog.index], archetype)
        return sorted(out, key=lambda d: d["worth"])

    def _worth_rows(self, build: Build, items: Sequence[str], archetype: Optional[str] = None) -> List[dict]:
        self._need()
        eng = self.engine()
        s = eng.situations(build, 1, force=self.force(archetype))[0]
        sel = set(build.items())
        ranked_by: Dict[str, List[str]] = {}
        out = []
        for item in items:
            t = self.catalog.index[item]
            kind = eng._kind(item)
            if kind not in ranked_by:
                same = [i for i in self.catalog.ids if eng._kind(i) == kind and s.ctx.avail[self.catalog.index[i]]]
                ranked_by[kind] = sorted(same, key=lambda i: -s.q[self.catalog.index[i]])
            ranked = ranked_by[kind]
            best = next((i for i in ranked if i not in sel and i != item and eng._feasible(i, sel - {item}, s)), None)
            q = float(s.q[t])
            out.append(dict(item=item, worth=round(q, 4), support=round(float(s.support[t]), 1),
                            unrated=eng._cold(item, s), category=self.manual.get(item),
                            available=bool(s.ctx.avail[t]), rank=ranked.index(item) + 1 if item in ranked else None,
                            of=len(ranked), pass_=eng._pass_here(item, s, sel),
                            best_swap=best,
                            swap_gain=round(float(s.q[self.catalog.index[best]]) - q, 4) if best else None,
                            rules=[r.text(self.rules.region_names) for r in self.rules.for_item(item) if r.status == "active"][:8],
                            review_rules=[r.describe() for r in self.review_rules if item in (r.subject, r.other)][:8],
                            region=s.mode, region_name=s.name))
        return out

    def explain(self, build: Build) -> dict:
        """Which pass each pick of the build belongs to, and which rule put it there."""
        eng = self.engine()
        s = eng.situations(build, 1)[0]
        s.ctx.items = set(build.items())
        fired = {}
        for r in eng._firing(s):
            fired.setdefault(r.item, r)
        out = {FOUNDATIONAL: [], CENTRAL: [], LUXURY: []}
        for i in build.items():
            if i not in self.catalog.index:
                continue
            r = fired.get(i)
            if i in self.manual:
                out[self.manual[i]].append(dict(item=i, rule=f"category: {self.manual[i]}",
                                                worth=round(float(s.q[self.catalog.index[i]]), 3)))
                continue
            out[r.pass_ if r else LUXURY].append(dict(item=i, rule=r.text(self.rules.region_names) if r else None,
                                                     worth=round(float(s.q[self.catalog.index[i]]), 3)))
        out["missing"] = [dict(item=r.item, rule=r.text(self.rules.region_names)) for r in fired.values()
                          if r.item not in set(build.items())]
        return out

    def summary(self) -> dict:
        self._need()
        return dict(version=VERSION, items=len(self.catalog), builds=len(self.data) if self.data is not None else 0,
                    reviews=len(self.reviews), space=self.space.report(), rules=self.rulebook.summary(), selection=self.selection_report(),
                    review_rules=sum(r.status != "proposed" for r in self.review_rules),
                    patches=[p.name for p in self.patches.patches], fitted_at=self.fitted_at)
