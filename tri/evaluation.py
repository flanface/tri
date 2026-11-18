"""Evaluation on reviewed builds: leave-one-build-out operation F1, review
agreement and bootstrap standard errors.

Leave-one-build-out (LOBO). A unit is one reviewed build (its build id); its
truth is the newest review of that build. For each unit, the reviews of that
build and of every reviewed build at similarity >= `exclude_sim` (0.95) are
dropped, the build is answered, and the answer's add and remove operations are
scored against the reviewer's:

    add precision    = |A_tri & A_rev| / |A_tri|
    remove precision = |R_tri & R_rev| / |R_tri|
    recall           = (|A_tri & A_rev| + |R_tri & R_rev|) / (|A_rev| + |R_rev|)
    F1               = 2 P R / (P + R),   P = all hits / all tri operations

Counts are pooled over units (micro average). Iffy marks are not scored.
Standard errors come from a bootstrap over units (one unit = one review of a
distinct build)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

COUNTS = ("tp_add", "fp_add", "tp_rem", "fp_rem", "n_add", "n_rem")


@dataclass
class Unit:
    build_id: str
    truth: object                      # the newest Review of the build
    exclude: List[str] = field(default_factory=list)   # review ids dropped for this unit


def units(reviews: Sequence, sim: Callable[[str, str], float], exclude_sim: float = 0.95) -> List[Unit]:
    """One unit per reviewed build. sim(review_id, review_id) -> similarity."""
    by: Dict[str, list] = {}
    for r in reviews:
        by.setdefault(r.original.build_id or r.review_id, []).append(r)
    newest = {k: max(v, key=lambda r: (r.timestamp or 0.0)) for k, v in by.items()}
    keys = list(newest)
    out = []
    for k in keys:
        rid = newest[k].review_id
        drop = {r.review_id for r in by[k]}
        for k2 in keys:
            if k2 != k and sim(rid, newest[k2].review_id) >= exclude_sim:
                drop |= {r.review_id for r in by[k2]}
        out.append(Unit(k, newest[k], sorted(drop)))
    return out


def score(changes: Sequence, review) -> Dict[str, int]:
    """Counts of one answer against one review. changes: objects or dicts with
    item and action ("add" / "remove")."""
    get = lambda c, k: c[k] if isinstance(c, dict) else getattr(c, k)
    pa = {get(c, "item") for c in changes if str(get(c, "action")) in ("add", "Action.ADD")}
    pr = {get(c, "item") for c in changes if str(get(c, "action")) in ("remove", "Action.REMOVE")}
    ta = {o.item_id for o in review.operations if o.action.value == "add"}
    tr = {o.item_id for o in review.operations if o.action.value == "remove"}
    return dict(tp_add=len(pa & ta), fp_add=len(pa - ta), tp_rem=len(pr & tr), fp_rem=len(pr - tr),
                n_add=len(ta), n_rem=len(tr))


def metrics(rows: Sequence[Dict[str, int]]) -> Dict[str, float]:
    s = {k: float(sum(r[k] for r in rows)) for k in COUNTS}
    div = lambda a, b: a / b if b > 0 else 0.0
    tp = s["tp_add"] + s["tp_rem"]
    pred = tp + s["fp_add"] + s["fp_rem"]
    P, R = div(tp, pred), div(tp, s["n_add"] + s["n_rem"])
    return dict(f1=div(2 * P * R, P + R), precision=P, recall=R,
                add_precision=div(s["tp_add"], s["tp_add"] + s["fp_add"]),
                remove_precision=div(s["tp_rem"], s["tp_rem"] + s["fp_rem"]),
                add_recall=div(s["tp_add"], s["n_add"]), remove_recall=div(s["tp_rem"], s["n_rem"]),
                ops=pred, units=float(len(rows)))


def bootstrap(rows: Sequence, fn: Callable[[Sequence], Dict[str, float]] = metrics, B: int = 1000,
              seed: int = 0) -> Dict[str, Dict[str, float]]:
    """{metric: {value, se}}: fn on rows and the bootstrap standard error over rows."""
    rows = list(rows)
    base = fn(rows)
    if not rows:
        return {k: dict(value=v, se=0.0) for k, v in base.items()}
    rng = np.random.default_rng(seed)
    n = len(rows)
    draws = {k: [] for k in base}
    for _ in range(B):
        m = fn([rows[i] for i in rng.integers(0, n, n)])
        for k in base:
            draws[k].append(m[k])
    return {k: dict(value=base[k], se=float(np.std(draws[k], ddof=1))) for k in base}


def paired_gain(rows_a: Sequence, rows_b: Sequence, key: str = "f1", B: int = 1000, seed: int = 0) -> dict:
    """metric(b) - metric(a) on the same units, with a paired bootstrap SE."""
    assert len(rows_a) == len(rows_b)
    n = len(rows_a)
    rng = np.random.default_rng(seed)
    d0 = metrics(rows_b)[key] - metrics(rows_a)[key]
    ds = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        ds.append(metrics([rows_b[i] for i in ix])[key] - metrics([rows_a[i] for i in ix])[key])
    return dict(gain=d0, se=float(np.std(ds, ddof=1)))


def agreement_metrics(rows: Sequence[dict]) -> Dict[str, float]:
    return dict(review_agreement=float(np.mean([r["agreement"] for r in rows])) if rows else 1.0)


def _parallel(fn, items, workers: int):
    if workers <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(fn, items, chunksize=max(1, len(items) // (workers * 4)))


_STATE: dict = {}


def _lobo_one(i: int) -> dict:
    tri, us, answer, rebuild = _STATE["tri"], _STATE["units"], _STATE["answer"], _STATE["rebuild"]
    u = us[i]
    drop = set(u.exclude)
    rebuild(tri, [r for r in _STATE["reviews"] if r.review_id not in drop])
    changes = answer(tri, u.truth.original)
    return dict(build_id=u.build_id, review_id=u.truth.review_id, **score(changes, u.truth))


def default_answer(tri, build) -> list:
    alt = tri.review(build, 1)[0]
    return [c for c in alt.changes if c.action in ("add", "remove")]


def default_rebuild(tri, reviews) -> None:
    tri.set_reviews(reviews)


def lobo(tri, reviews: Optional[Sequence] = None, answer: Callable = default_answer,
         rebuild: Callable = default_rebuild, exclude_sim: float = 0.95, workers: Optional[int] = None,
         limit: Optional[int] = None) -> List[dict]:
    """Leave-one-build-out rows (one per reviewed build). tri keeps its reviews
    afterwards in the parent process (folds run in forked workers when workers > 1)."""
    reviews = list(tri.reviews if reviews is None else reviews)
    led = tri.ledger
    sim = (lambda a, b: led.similarity(a, b)) if led is not None else (lambda a, b: 0.0)
    us = units(reviews, sim, exclude_sim)
    if limit:
        us = us[:limit]
    _STATE.update(tri=tri, units=us, answer=answer, rebuild=rebuild, reviews=reviews)
    workers = workers if workers is not None else min(2, os.cpu_count() or 1)
    try:
        rows = _parallel(_lobo_one, list(range(len(us))), workers)
    finally:
        if workers <= 1:
            rebuild(tri, reviews)
        _STATE.clear()
    return rows


def agreement(tri) -> List[dict]:
    """Per-review agreement rows (Tri.review_agreement(detail=True))."""
    return tri.review_agreement(detail=True)[1]
