"""Explicit rule induction over the review store (ripple-down / ILP-lite).

Pass rules (tri.passes) are mined from builds; these rules are mined from
reviews. Each review operation proposes a typed rule; support and
counterexamples are counted, and a disputed rule gets the single condition
(another item present, a stat threshold, an oath or murmur, or the archetype
region) that best separates supporting cases from counterexamples, giving a
refined rule plus its exception.

Rule kinds
  REDUNDANT(x | y)       x unnecessary because y is present
  CONFLICT(x , y)        x works against y
  BROKEN(x)              x does not work
  PREFER(a > x)          a displaced x to make space
  NECESSARY(x)           x added as necessary for archetype
  GENERAL(x)             x added as generally good: a foundational talent for any
                         build (counterexamples: "does not work" / "does not fit"
                         removals of x anywhere). Inspectable only: tri applies
                         generally-good claims through the review ledger.
  NO_FIT(x)              x does not synergise / not for this scenario (weak SM)
  OFF_ARCHETYPE(x, A)    x does not belong on archetype family A (strong SM); applies
                         only to builds located in A (feature "family:<A>"), and is
                         countered only by endorsements of x in builds of A

Status  proposed  : too little evidence
        supported : Beta lower bound on precision above threshold
        disputed  : real support and real counterexamples
        refined   : disputed, but a condition explains the split
        overruled : (authority mode) disputed, no condition explains it, and the
                    newest evidence is a counterexample: the rule is not applied

Authority mode (induce(reviews, ledger=...)): reviews are ground truth, so a
single review is enough (min_support defaults to the weight of one weak op),
evidence superseded by newer reviews (authority.ReviewLedger) carries its
reduced weight or none, a rule with support and no active counterexample is
supported, and a dispute that no condition explains is settled by whichever
side has the newest evidence."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tri.core.catalog import Catalog
from tri.core.stats import wilson_lower
from tri.core.types import Action, Build, Reason, Review, Strength, off_archetype, strength_weight
from tri.rules.defs import Case
from tri.reviews.authority import made_by_tri

SPACE_REASONS = {Reason.LESS_NECESSARY_SPACE, Reason.LUXURY_LESS}
DEPENDS_JOIN = " + "                     # DEPENDS(x, "a + b"): the components x needs
NOFIT_REASONS = {Reason.NO_SYNERGY, Reason.SCENARIO_MISMATCH}


def context_features(b: Build, thresholds=(25, 40, 50, 60, 75, 90)) -> set:
    """Predicates a rule condition may test: item presence, stat thresholds,
    oath and murmur ("oath:<v>", "murmur:<v>"), weapon class ("weapon:<v>",
    from Build.meta or a review's weapons note). Tri adds "region:<k>" (the
    build's archetype region) through RuleInducer(features=...)."""
    from tri.model.matrix import context_tokens
    from tri.reviews.authority import build_weapons
    f = {f"has:{i}" for i in b.items()} | set(context_tokens(b)) | {f"weapon:{w}" for w in build_weapons(b)}
    for st, v in b.final_stats.items():
        for thr in thresholds:
            if v >= thr:
                f.add(f"{st}>={thr}")
    return f


def _family_token(feats: set) -> Optional[str]:
    """The family key in a feature set ("family:<key>"), or None."""
    for f in feats:
        if f.startswith("family:"):
            return f[7:]
    return None


@dataclass
class Rule:
    """A typed, inspectable rule with evidence and status (see module docstring)."""
    kind: str
    subject: str
    other: Optional[str]
    support: List[Case] = field(default_factory=list)
    counter: List[Case] = field(default_factory=list)
    status: str = "proposed"
    condition: Optional[str] = None      # refined applicability condition
    condition_gain: float = 0.0
    precision_lb: float = 0.0

    @property
    def key(self) -> Tuple[str, str, Optional[str]]:
        """(kind, subject, other): identity of the rule."""
        return (self.kind, self.subject, self.other)

    def applies(self, b: Build, feats: Optional[set] = None) -> bool:
        """True if the rule is active and its refined condition (if any) holds.
        feats: precomputed predicates of b (default context_features(b))."""
        if self.status not in ("supported", "refined"):   # proposed, disputed, overruled
            return False
        if self.kind == "OFF_ARCHETYPE" or (self.kind == "NECESSARY" and self.other):
            if feats is None or f"family:{self.other}" not in feats:
                return False                         # scoped to its archetype family
        if not self.condition:
            return True
        f = context_features(b) if feats is None else feats
        if self.condition.startswith("not "):
            return self.condition[4:] not in f
        return self.condition in f

    def describe(self) -> str:
        """The rule in symbols (tri.symbolic): [R supported] a ⊥ b | c   +3.0 −0.5 · lb 0.82."""
        from tri.rules.symbolic import review_rule
        return review_rule(self)

    def plain(self) -> str:
        """One-line description in words with evidence counts (the pre-2.14 wording)."""
        head = {"REDUNDANT": f"{self.subject} is redundant given {self.other}",
                "CONFLICT": f"{self.subject} works against {self.other}",
                "BROKEN": f"{self.subject} does not work",
                "PREFER": f"prefer {self.other} over {self.subject} when space is short",
                "NECESSARY": f"{self.subject} is necessary for {self.other or 'the archetype'}",
                "GENERAL": f"{self.subject} is generally good (foundational)",
                "NO_FIT": f"{self.subject} does not fit the build",
                "OFF_ARCHETYPE": f"{self.subject} does not belong on {self.other}",
                "SYNERGY": f"{self.subject} synergises with {self.other}",
                "DEPENDS": f"{self.subject} does not work without {self.other}",
                "SIDEGRADE": f"{self.subject} is a sidegrade to {self.other}"}[self.kind]
        cond = f"  IF {self.condition}" if self.condition else ""
        s, c = sum(x.weight for x in self.support), sum(x.weight for x in self.counter)
        return f"[{self.status}] {head}{cond}  (support {s:.1f}, counter {c:.1f}, lb {self.precision_lb:.2f})"

    def to_dict(self) -> dict:
        """JSON-serialisable dict (evidence as counts and review ids)."""
        return dict(kind=self.kind, subject=self.subject, other=self.other, status=self.status,
                    condition=self.condition, precision_lb=self.precision_lb,
                    support=[x.review_id for x in self.support],
                    counter=[x.review_id for x in self.counter])


def _beta_lb(s: float, c: float, z: float = 1.64) -> float:
    """Wilson lower bound on support / (support + counter)."""
    n = s + c
    return wilson_lower(s / n, n, z) if n > 0 else 0.0


class RuleInducer:
    """Induces rules from the raw review log. Stateless: call induce() on the
    whole log each time (it is cheap).
    """

    def __init__(self, catalog: Catalog, min_support: float = 2.0, lb_threshold: float = 0.6,
                 reviewer_weight: Optional[Dict[str, float]] = None,
                 strength_weights: Optional[Dict[Strength, float]] = None,
                 authority_min_support: float = 0.5, features=None):
        """min_support: weighted support needed before a rule can leave 'proposed'.
        lb_threshold: Wilson lower bound on precision needed for 'supported'.
        reviewer_weight: per-reviewer multiplier (default 1.0).
        strength_weights: numeric value of weak/strong (default 0.5 / 1.0).
        authority_min_support: min_support used when induce() gets a ledger.
        features: build -> predicate set for refinement (default context_features)."""
        self.features = features or context_features
        self.sw = strength_weights
        self.c = catalog
        self.min_support = min_support
        self.lb = lb_threshold
        self.rw = reviewer_weight or {}
        self.authority_min_support = authority_min_support
        self._ledger = None

    def induce(self, reviews: Sequence[Review], ledger=None) -> List[Rule]:
        """Propose, score and refine rules from reviews. Returns all rules,
        including 'proposed' ones (filter by status).
        ledger: authority.ReviewLedger over the same reviews; switches on
        authority mode (see module docstring).
        """
        self._ledger = ledger
        rules: Dict[tuple, Rule] = {}

        def get(kind, x, y=None):
            k = (kind, x, y)
            if k not in rules:
                rules[k] = Rule(kind, x, y)
            return rules[k]

        endorsed_pairs: Dict[frozenset, List[Case]] = defaultdict(list)
        endorsed_items: Dict[str, List[Case]] = defaultdict(list)
        removed_space: Dict[str, List[Case]] = defaultdict(list)
        removed_unfit: Dict[str, List[Case]] = defaultdict(list)   # does not work / does not fit, anywhere
        endorsed_ctx: Dict[str, List[tuple]] = defaultdict(list)   # item -> (case, items of the result)
        removed_any: Dict[str, List[tuple]] = defaultdict(list)    # item -> (case, items of the original)
        swapped: Dict[tuple, List[Case]] = defaultdict(list)       # (removed x, added y) for space / luxury

        for r in reviews:
            w = self.rw.get(r.reviewer_id, 1.0)
            lw = ledger.weights.get(r.review_id) if ledger is not None else None
            order = ledger.order.get(r.review_id, 0) if ledger is not None else 0
            orig = r.original
            case = Case(r.review_id, r.reviewer_id, orig, w, order)
            res = r.apply(self.c)
            iffy = r.iffy()
            neutral = set(getattr(r, "neutral", None) or ())   # v3.1.1: purged items say nothing
            added = {o.item_id for o in r.added()}
            # v2.25.1: tri's own picks echoed back by a review of a build tri made are no
            # reviewer's endorsement: they counter no rule and pair nothing (worth still
            # sees them, at LedgerConfig.echo_keep)
            endorsed = [] if made_by_tri(orig) else \
                [i for i in res.items() if i not in iffy and i not in added and i not in neutral
                 and (lw is None or lw.keep.get(i, 1.0) > 0)]
            for j, o in enumerate(r.operations):
                ow = 0.0 if o.item_id in neutral else (lw.ops[j] if lw is not None else 1.0)
                if ow <= 0:
                    continue
                cs = Case(r.review_id, r.reviewer_id, orig, w * ow * strength_weight(o.strength, self.sw), order)
                if o.action == Action.REMOVE:
                    removed_any[o.item_id].append((cs, o.reason))
                    if o.reason == Reason.NEEDS and o.related:
                        get("DEPENDS", o.item_id, DEPENDS_JOIN.join(sorted(o.related))).support.append(cs)
                    elif o.reason == Reason.REDUNDANT_WITH:
                        for y in o.related:
                            get("REDUNDANT", o.item_id, y).support.append(cs)
                    elif o.reason == Reason.WORKS_AGAINST:
                        for y in o.related:
                            a, b = sorted((o.item_id, y))
                            get("CONFLICT", a, b).support.append(cs)
                    elif o.reason == Reason.BROKEN:
                        get("BROKEN", o.item_id).support.append(cs)
                        removed_unfit[o.item_id].append(cs)
                    elif off_archetype(o):
                        # strong SM: "this talent does not belong on this archetype". Scoped to
                        # the reviewed build's family; with no located family it stays a ledger
                        # claim on similar builds and makes no global rule.
                        fam = _family_token(self.features(orig))
                        if fam:
                            get("OFF_ARCHETYPE", o.item_id, fam).support.append(cs)
                    elif o.reason in NOFIT_REASONS:
                        get("NO_FIT", o.item_id).support.append(cs)
                        removed_unfit[o.item_id].append(cs)
                    elif o.reason in SPACE_REASONS:
                        removed_space[o.item_id].append(cs)
                        grp = [p for p in r.operations if p.action == Action.ADD and
                               (p.group == o.group)]
                        for p in grp:
                            get("PREFER", o.item_id, p.item_id).support.append(cs)
                            swapped[(o.item_id, p.item_id)].append(cs)
                elif o.action == Action.ADD and o.reason == Reason.NECESSARY_FOR_ARCHETYPE:
                    # v3.6: necessary for *the reviewed build's* archetype (its "is" note,
                    # else its reading): scoped to that family. With no family it stays a
                    # ledger claim on similar builds and makes no rule (borders are chasms).
                    fam = _family_token(self.features(orig))
                    if fam:
                        get("NECESSARY", o.item_id, fam).support.append(cs)
                elif o.action == Action.ADD and o.reason == Reason.GENERALLY_GOOD:
                    get("GENERAL", o.item_id).support.append(cs)
                elif o.action == Action.ADD and o.reason == Reason.SYNERGY_WITH:
                    for y in o.related:
                        get("SYNERGY", o.item_id, y).support.append(cs)
                elif o.action == Action.IFFY and o.reason == Reason.SIDEGRADE:
                    for y in o.related:
                        get("SIDEGRADE", o.item_id, y).support.append(cs)
            res_items = set(res.items())
            for i in endorsed:
                endorsed_ctx[i].append((case, res_items))
                kw = lw.keep.get(i, 1.0) if lw is not None else 1.0
                endorsed_items[i].append(case if kw == 1.0 else Case(case.review_id, case.reviewer_id,
                                                                   case.build, w * kw, order))
            for a_i in range(len(endorsed)):
                for b_i in range(a_i + 1, len(endorsed)):
                    endorsed_pairs[frozenset((endorsed[a_i], endorsed[b_i]))].append(case)

        # ---- counterexamples
        for rule in rules.values():
            x, y = rule.subject, rule.other
            if rule.kind in ("REDUNDANT", "CONFLICT"):
                rule.counter = list(endorsed_pairs.get(frozenset((x, y)), []))
            elif rule.kind in ("BROKEN", "NO_FIT"):
                rule.counter = list(endorsed_items.get(x, []))
            elif rule.kind == "OFF_ARCHETYPE":
                # x endorsed in a build of the same family
                rule.counter = [cs for cs in endorsed_items.get(x, [])
                                if _family_token(self.features(cs.build)) == y]
            elif rule.kind == "GENERAL":
                rule.counter = list(removed_unfit.get(x, []))
            elif rule.kind == "SYNERGY":
                # x taken out of a build that had y (for a reason about x itself)
                rule.counter = [cs for cs, why in removed_any.get(x, [])
                                if y in cs.build.items() and why not in SPACE_REASONS]
            elif rule.kind == "DEPENDS":
                # x endorsed in a build that lacks one of its components
                need = y.split(DEPENDS_JOIN)
                rule.counter = [cs for cs, have in endorsed_ctx.get(x, []) if not all(n in have for n in need)]
            elif rule.kind == "SIDEGRADE":
                # one swapped for the other: a preference, not a sidegrade
                rule.counter = list(swapped.get((x, y), [])) + list(swapped.get((y, x), []))
            elif rule.kind == "PREFER":
                # y displaced for space while x was kept in the same build
                rule.counter = [cs for cs in removed_space.get(y, [])
                                if x in cs.build.items()]
        for rule in rules.values():
            self._score(rule)
        return sorted(rules.values(), key=lambda r: (-{"supported": 3, "refined": 2, "disputed": 1,
                                                        "overruled": 0, "proposed": 0}[r.status],
                                                       -r.precision_lb))

    # ------------------------------------------------------------ scoring
    def _score(self, rule: Rule) -> None:
        s = sum(x.weight for x in rule.support)
        c = sum(x.weight for x in rule.counter)
        rule.precision_lb = _beta_lb(s, c)
        if self._ledger is not None:
            self._score_authority(rule, s, c)
            return
        if s < self.min_support:
            rule.status = "proposed"
        elif rule.precision_lb >= self.lb:
            rule.status = "supported"
        elif c >= 1.0:
            rule.status = "disputed"
            self._refine(rule)
        else:
            rule.status = "proposed"

    def _score_authority(self, rule: Rule, s: float, c: float) -> None:
        """Reviews are accurate: no counterexample -> supported; a dispute is
        refined by a context condition if one explains it, else the side with
        the newest evidence wins."""
        if s < self.authority_min_support - 1e-9:
            rule.status = "proposed"
            return
        if c <= 1e-9:
            rule.status = "supported"
            return
        rule.status = "disputed"
        saved = self.min_support
        self.min_support = self.authority_min_support
        self._refine(rule)
        self.min_support = saved
        if rule.status == "refined":
            return
        ns = max(x.order for x in rule.support)
        nc = max(x.order for x in rule.counter)
        rule.status = "supported" if ns > nc else "overruled"

    def _features(self, b: Build) -> set:
        return self.features(b)

    def _refine(self, rule: Rule) -> None:
        """Find one condition that holds on support and fails on counter cases."""
        sup = [(self._features(x.build), x.weight) for x in rule.support]
        cnt = [(self._features(x.build), x.weight) for x in rule.counter]
        S = sum(w for _, w in sup); C = sum(w for _, w in cnt)

        def H(a, b):
            n = a + b
            if n <= 0 or a <= 0 or b <= 0:
                return 0.0
            p = a / n
            return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

        base = H(S, C)
        cands = set().union(*[f for f, _ in sup]) if sup else set()
        cands -= {f"has:{rule.subject}"} | ({f"has:{rule.other}"} if rule.other else set())
        best, best_gain = None, 0.0
        for f in cands:
            for neg in (False, True):
                hs = sum(w for fs, w in sup if (f in fs) != neg)
                hc = sum(w for fs, w in cnt if (f in fs) != neg)
                ns, nc = S - hs, C - hc
                n = S + C
                gain = base - ((hs + hc) / n * H(hs, hc) + (ns + nc) / n * H(ns, nc))
                if gain > best_gain and hs >= self.min_support * 0.5 and hs > hc:
                    best, best_gain = (f if not neg else f"not {f}"), gain
        if best is not None and best_gain > 0.2:
            rule.condition, rule.condition_gain = best, best_gain
            rule.status = "refined"
