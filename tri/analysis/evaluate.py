"""Measuring tri: held-out builds and held-out reviews, against simple baselines.

completion_benchmark   hide part of each held-out build's talents; refill.
                       Compares three fillers with the same slot count:
                         tri          passes + worth (the real thing)
                         worth_only   worth alone (pass rules switched off)
                         popularity   overall pick rate among builds that can take the item
                       Score = share of hidden talents recovered (recall; with
                       the slot count fixed, precision is the same number).
                       Also split by the pass the hidden talent belongs to.
rule_benchmark         on held-out full builds: per pass, how often a fired rule
                       is right (precision) and how many picks rules explain
                       (coverage). A deterministic rulebase shows high both.
review_benchmark       on held-out reviews: does tri make the reviewer's edits?
                       add / remove precision and recall, reason accuracy, and
                       removal AUC (low worth should mean removed).
group_split            reviews split by build signature (or your key).

v2 guarantees, measured (these need no ground truth):
strictness_benchmark   on held-out reviews: agreement with the reviewers' STRONG
                       operations, and the share of disagreements that tri
                       itself reports in Alternative.violations.
mapping_benchmark      on any builds: checks the concept-space isometry
                       |u_a - u_b| = omega-weighted RMS logit difference (exact up to float error)
                       and the worth bound weighted RMS |sigma_a - sigma_b| <= |u_a - u_b| / 4;
                       with labels, the share of same-label pairs closer than
                       different-label pairs (AUC).
certification          on held-out builds: share of mined rules whose rate on
                       those builds still meets its threshold (Rulebook.certification_check).

Run on your corpus with `python -m tri.cli.fit --builds builds.json` (it prints completion_benchmark and
rule_benchmark on held-out lineages, --test-frac); review_benchmark needs held-out reviews.
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from tri.core.types import Action, Build, Kind, Review
from tri.rules.passes import CENTRAL, FOUNDATIONAL, LUXURY


def group_split(reviews: Sequence[Review], test_frac: float = 0.2, seed: int = 0,
                key: Optional[Callable[[Review], str]] = None):
    """(train, test) reviews with whole groups on one side (default key: build signature)."""
    key = key or (lambda r: r.original.signature())
    groups = sorted({key(r) for r in reviews})
    random.Random(seed).shuffle(groups)
    test = set(groups[:max(1, int(len(groups) * test_frac))])
    return [r for r in reviews if key(r) not in test], [r for r in reviews if key(r) in test]


def _auc(pos: List[float], neg: List[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def _no_rules(tri):
    """An engine with pass rules switched off (worth only)."""
    eng = tri.engine()
    eng.rb = copy.copy(eng.rb)
    eng.rb.rules = []
    return eng


def _popularity_fill(tri, b: Build, n: int, kind: Kind) -> List[str]:
    c = tri.catalog
    p = tri._worth.prior
    avail = c.availability_in(b)
    sel = set(b.items())
    out = []
    for t in np.argsort(-p):
        it = c.ids[t]
        if len(out) >= n:
            break
        if c.items[it].kind != kind or it in sel or not avail[t]:
            continue
        x = c.items[it]
        if any(e in sel for e in x.exclusive_with) or any(it in c.items[s].exclusive_with for s in sel):
            continue
        if x.prerequisites and not all(q in sel for q in x.prerequisites):
            continue
        sel.add(it); out.append(it)
    return out


def completion_benchmark(tri, builds: Sequence[Build], hide_frac: float = 0.3, n: int = 1000,
                         seed: int = 0) -> dict:
    """Refill hidden talents of held-out builds (see module docstring)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(builds))[:n]
    eng, eng0 = tri.engine(), _no_rules(tri)
    score = {"tri": 0.0, "worth_only": 0.0, "popularity": 0.0}
    by_pass = {name: {p: [0, 0] for p in (FOUNDATIONAL, CENTRAL, LUXURY)} for name in score}
    total = hidden_n = 0
    cats = tri.categories()
    for i in order:
        b = builds[int(i)]
        tal = [t for t in b.talents if t in tri.catalog.index]
        k = int(len(tal) * hide_frac)
        if k == 0:
            continue
        hide = set(rng.choice(tal, k, replace=False).tolist())
        vis = copy.copy(b)
        vis.talents = [t for t in b.talents if t not in hide]
        vis.mantras = [m for m in b.mantras if m in tri.catalog.index]
        cap = (len(tal), len(vis.mantras))
        got_tri = {c.item for c in eng.complete(vis, cap).changes if c.action == "add"}
        got_w = {c.item for c in eng0.complete(vis, cap).changes if c.action == "add"}
        got_p = set(_popularity_fill(tri, vis, k, Kind.TALENT))
        score["tri"] += len(got_tri & hide) / k
        score["worth_only"] += len(got_w & hide) / k
        score["popularity"] += len(got_p & hide) / k
        for name, got in (("tri", got_tri), ("worth_only", got_w), ("popularity", got_p)):
            for h in hide:
                ps = cats.get(h, LUXURY)
                by_pass[name][ps][0] += h in got; by_pass[name][ps][1] += 1
        total += 1; hidden_n += k
    out = {name: round(v / max(total, 1), 4) for name, v in score.items()}
    out.update(n=total, hidden=hidden_n, hide_frac=hide_frac,
               by_pass={name: {p: round(a / b, 4) if b else None for p, (a, b) in d.items()}
                        for name, d in by_pass.items()},
               hidden_by_pass={p: b for p, (a, b) in by_pass["tri"].items()})
    return out


def rule_benchmark(tri, builds: Sequence[Build], n: int = 2000, seed: int = 0) -> dict:
    """Per pass: precision of fired rules and coverage of real picks, on held-out full builds."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(builds))[:n]
    eng = tri.engine()
    st = {p: dict(fired=0, right=0) for p in (FOUNDATIONAL, CENTRAL)}
    picks = covered = 0
    for i in order:
        b = builds[int(i)]
        S = {x for x in b.items() if x in tri.catalog.index}
        s = eng.situations(b, 1)[0]
        s.ctx.items = set(S)
        fired: Dict[str, str] = {}
        for r in tri.rulebook.firing(s.ctx):
            fired.setdefault(r.item, r.pass_)
        for it, ps in fired.items():
            st[ps]["fired"] += 1; st[ps]["right"] += it in S
        picks += len(S); covered += sum(1 for x in S if x in fired)
    return dict(n=len(order), coverage=round(covered / max(picks, 1), 4),
                **{p: dict(fired=v["fired"], precision=round(v["right"] / v["fired"], 4) if v["fired"] else None)
                   for p, v in st.items()})


def review_benchmark(tri, reviews: Sequence[Review]) -> dict:
    """Does tri reproduce held-out reviews it has not seen?

    strong_recall: share of the reviewers' strong operations tri also proposes
    (the non-negotiable part of a review). rule_precision: share of tri's
    changes that come from rules or review memory (not plain worth swaps) and
    that the reviewer also made. Weak luxury choices are partly taste, so plain
    add/remove precision is a pessimistic number."""
    tp = dict(add=0, remove=0); pred = dict(add=0, remove=0); true = dict(add=0, remove=0)
    reason_ok = reason_n = 0
    st_tp = st_n = rp_tp = rp_n = 0
    pos, neg = [], []
    for r in reviews:
        alt = tri.review(r.original, 1, keep_size=True)[0]
        P = {(c.action, c.item): c for c in alt.changes}
        T = {(o.action.value, o.item_id): o for o in r.operations if o.action != Action.IFFY}
        for a in ("add", "remove"):
            pa = {k for k in P if k[0] == a}; ta = {k for k in T if k[0] == a}
            tp[a] += len(pa & ta); pred[a] += len(pa); true[a] += len(ta)
        for k, o in T.items():
            if o.strength is not None and o.strength.value == "strong":
                st_n += 1; st_tp += k in P
        for k, c in P.items():
            if c.source != "worth":
                rp_n += 1; rp_tp += k in T
        for k, o in T.items():
            if k[0] == "remove" and k in P and o.reason is not None:
                reason_n += 1; reason_ok += P[k].reason == o.reason.value
        removed = {o.item_id for o in r.operations if o.action == Action.REMOVE}
        for it in r.original.items():
            if it in alt.worth:
                (pos if it in removed else neg).append(-alt.worth[it])
    f = lambda a, b: round(a / b, 4) if b else None
    return dict(n=len(reviews), add_precision=f(tp["add"], pred["add"]), add_recall=f(tp["add"], true["add"]),
                remove_precision=f(tp["remove"], pred["remove"]), remove_recall=f(tp["remove"], true["remove"]),
                reason_accuracy=f(reason_ok, reason_n), removal_auc=_auc(pos, neg),
                strong_recall=f(st_tp, st_n), rule_precision=f(rp_tp, rp_n))


def _ops_f1(engine, reviews: Sequence[Review]) -> float:
    """F1 of (action, item) operations, engine answers vs reviews."""
    tp = pp = tt = 0
    for r in reviews:
        alt = engine.review(r.original, 1, keep_size=True)[0]
        P = {(c.action, c.item) for c in alt.changes}
        T = {(o.action.value, o.item_id) for o in r.operations if o.action != Action.IFFY}
        tp += len(P & T); pp += len(P); tt += len(T)
    if pp + tt == 0:
        return 1.0
    return 2 * tp / (pp + tt)


def strictness_benchmark(tri, reviews: Sequence[Review]) -> dict:
    """Strong operations of held-out reviews: does tri's answer agree, and when
    it does not, does it say so (violations)? A strong add agrees if the item is
    in the answer, a strong remove if it is not."""
    agree = dis = flagged = 0
    for r in reviews:
        alt = tri.review(r.original, 1, keep_size=True)[0]
        out = set(alt.items())
        vio = {v.get("item") for v in (alt.violations or [])}
        for o in r.operations:
            if o.action == Action.IFFY or o.strength is None or o.strength.value != "strong":
                continue
            ok = (o.item_id in out) if o.action == Action.ADD else (o.item_id not in out)
            agree += ok
            if not ok:
                dis += 1; flagged += o.item_id in vio
    n = agree + dis
    return dict(strong_ops=n, agreement=round(agree / n, 4) if n else None, disagreements=dis,
                reported_as_violations=flagged)


def mapping_benchmark(tri, builds: Sequence[Build], labels: Optional[Sequence] = None,
                      pairs: int = 2000, seed: int = 0) -> dict:
    """Checks the concept-space guarantees on random pairs of builds.

    isometry_error   max | |u_a - u_b| - RMS_t^omega (l_at - l_bt) |   (should be ~1e-6)
    bound_holds      share of pairs with RMS^omega |sigma(l_a) - sigma(l_b)| <= |u_a - u_b| / 4 (should be 1)
                     (RMS^omega: mean over items weighted by the metric weights omega, mean 1)
    auc              with labels: share of (same-label, different-label) pair
                     combinations where the same-label pair is closer."""
    sp = tri.space
    rng = np.random.default_rng(seed)
    pts = [sp.point(b, loo=False) for b in builds]
    n = len(pts)
    if n < 2:
        return dict(pairs=0)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    om = sp.weights()
    err, ok, D, same = 0.0, 0, [], []
    for _ in range(pairs):
        a, b = rng.choice(n, 2, replace=False)
        d = float(np.linalg.norm(pts[a].u - pts[b].u))
        rms = float(np.sqrt(np.mean(om * (pts[a].logits - pts[b].logits) ** 2)))
        err = max(err, abs(d - rms))
        ok += float(np.sqrt(np.mean(om * (sig(pts[a].logits) - sig(pts[b].logits)) ** 2))) <= d / 4 + 1e-9
        D.append(d)
        if labels is not None:
            same.append(labels[a] == labels[b])
    out = dict(pairs=pairs, isometry_error=float(err), bound_holds=round(ok / pairs, 4),
               median_distance=round(float(np.median(D)), 4))
    if labels is not None:
        D, same = np.array(D), np.array(same, bool)
        if same.any() and (~same).any():
            out["auc"] = round(float(np.mean(D[same][:, None] < D[~same][None, :])), 4)
            out["same_label_median"] = round(float(np.median(D[same])), 4)
            out["different_label_median"] = round(float(np.median(D[~same])), 4)
    return out


def certification(tri, builds: Sequence[Build]) -> dict:
    """Held-out check of the mined rules (see Rulebook.certification_check)."""
    from tri.model.matrix import BuildMatrix
    tokens = tri.data.tokens if getattr(tri, "data", None) is not None else None
    bm = BuildMatrix.from_builds(list(builds), tri.catalog, fields=tri.cfg.context_fields, tokens=tokens)
    return tri.rulebook.certification_check(bm)


def evaluate(tri, builds: Sequence[Build] = (), reviews: Sequence[Review] = (), n: int = 1000, seed: int = 0) -> dict:
    """All benchmarks that have data."""
    out = {}
    if builds:
        out["completion"] = completion_benchmark(tri, builds, n=n, seed=seed)
        out["rules"] = rule_benchmark(tri, builds, n=n, seed=seed)
        out["mapping"] = mapping_benchmark(tri, list(builds)[:n], seed=seed)
        out["certification"] = certification(tri, builds)
    if reviews:
        out["reviews"] = review_benchmark(tri, reviews)
        out["strictness"] = strictness_benchmark(tri, reviews)
    return out
