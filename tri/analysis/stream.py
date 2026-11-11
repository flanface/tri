from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from tri.analysis.categorize import sigmoid

POLARITY = {"keep": 1, "add": 1, "remove": 0}


def _stream(tri, max_reviews: Optional[int] = None):
    """(review ids in time order, U [m, D'], claims {rid: [(item, y)]})."""
    led = tri.ledger
    if led is None or not led.by_id:
        return [], np.zeros((0, 0)), {}
    order = {}
    for rid, cl in led.claims.items():
        order[rid] = min((c.order for c in cl), default=0)
    rid = sorted(led.by_id, key=lambda r: order.get(r, 0))
    if max_reviews is not None and len(rid) > max_reviews:
        rid = rid[-max_reviews:]
    U = np.array([led.vector(r) for r in rid])
    claims = {}
    for r in rid:
        claims[r] = [(c.item, POLARITY[c.label]) for c in led.claims.get(r, [])
                     if c.label in POLARITY and c.origin == "review" and c.item in tri.catalog.index]
    return rid, U, claims


def prequential(tri, max_reviews: Optional[int] = 400, window: int = 25, near: float = 0.95) -> dict:
    """Replay the reviews in time order; each claim predicted from earlier ones.
    Returns n (claims per review), accuracy (cumulative, memory then field),
    field (cumulative, field only), rolling (window), memory_share, final."""
    rid, U, claims = _stream(tri, max_reviews)
    if not rid:
        return dict(reviews=0, claims=0)
    led = tri.ledger
    kern = led.reach_kernel(led.cfg)
    smin = led.cfg.sim_min
    sp = tri.space
    hist: Dict[str, List[tuple]] = {}            # item -> [(review index, y)]
    hit, fhit, mem, per = [], [], [], []
    for k, r in enumerate(rid):
        q = sigmoid(sp.point(led.by_id[r].original, loo=True).logits)
        n0 = len(hit)
        for it, y in claims[r]:
            prev = hist.get(it, [])
            pred, used = None, False
            if prev:
                sims = np.array([kern(U[k], U[j]) for j, _ in prev])
                close = np.flatnonzero(sims >= near)
                if len(close):
                    pred, used = prev[int(close.max())][1], True       # newest close review wins
                else:
                    ok = sims >= smin
                    if ok.any():
                        ys = np.array([v for _, v in prev])[ok]
                        pred, used = int((sims[ok] @ ys) / sims[ok].sum() >= 0.5), True
            fpred = int(q[tri.catalog.index[it]] >= 0.5)
            if pred is None:
                pred = fpred
            hit.append(pred == y); fhit.append(fpred == y); mem.append(used)
        per.append(len(hit) - n0)
        for it, y in claims[r]:
            hist.setdefault(it, []).append((k, y))
    h = np.array(hit, float); f = np.array(fhit, float)
    ends = np.cumsum(per)
    acc = [float(h[:e].mean()) if e else np.nan for e in ends]
    fac = [float(f[:e].mean()) if e else np.nan for e in ends]
    roll = []
    for k in range(len(rid)):
        a = ends[max(0, k - window)] if k - window >= 0 else 0
        e = ends[k]
        roll.append(float(h[a:e].mean()) if e > a else np.nan)
    return dict(reviews=len(rid), claims=int(len(h)), n=per, accuracy=acc, field=fac, rolling=roll,
                memory_share=float(np.mean(mem)) if mem else 0.0,
                final=dict(accuracy=float(h.mean()) if len(h) else None, field=float(f.mean()) if len(f) else None,
                           last_window=roll[-1] if roll else None))


def variogram(tri, max_pairs: int = 40000, bins: Optional[np.ndarray] = None, seed: int = 0) -> dict:
    """Agreement of two reviews on an item against their concept distance.
    Returns edges, agree (all pairs), same / cross (same / different archetype), counts."""
    from tri.analysis.categorize import _top2, read_coords, soft_kind
    rid, U, claims = _stream(tri)
    if len(rid) < 2:
        return dict(pairs=0)
    P, dmin, *_ = read_coords(tri, U)
    top, _, p1, p2 = _top2(P)
    kind = soft_kind(p1, p2, dmin, tri.locset.reach if tri.locset is not None else float("inf"))
    arch = np.where(kind != "unplaced", top, -1)
    by: Dict[str, List[tuple]] = {}
    for k, r in enumerate(rid):
        for it, y in claims[r]:
            by.setdefault(it, []).append((k, y))
    A, B, Y = [], [], []
    for it, lst in by.items():
        for x in range(len(lst)):
            for z in range(x + 1, len(lst)):
                A.append(lst[x][0]); B.append(lst[z][0]); Y.append(lst[x][1] == lst[z][1])
    A, B, Y = np.array(A, int), np.array(B, int), np.array(Y, float)
    if len(A) > max_pairs:
        s = np.random.default_rng(seed).choice(len(A), max_pairs, replace=False)
        A, B, Y = A[s], B[s], Y[s]
    if not len(A):
        return dict(pairs=0)
    d = np.linalg.norm(U[A] - U[B], axis=1)
    edges = np.linspace(0, max(float(np.quantile(d, 0.98)), 1e-6), 13) if bins is None else np.asarray(bins)
    same = (arch[A] == arch[B]) & (arch[A] >= 0)

    def curve(m):
        out, cnt = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            k = m & (d >= a) & (d < b)
            out.append(float(Y[k].mean()) if k.sum() >= 5 else np.nan); cnt.append(int(k.sum()))
        return out, cnt

    al, ac = curve(np.ones(len(d), bool))
    sm, sc = curve(same)
    cr, cc = curve(~same)
    return dict(pairs=int(len(d)), edges=edges.tolist(), agree=al, counts=ac, same=sm, same_counts=sc,
                cross=cr, cross_counts=cc, reach=float(tri.ledger.cfg.reach))


def lowrank(tri, permutations: int = 200, seed: int = 0, top: int = 6) -> dict:
    """Review residuals E[r, t] = y - q_t(u_r) on claimed items. Is the top
    singular value larger than under permutations of each item's residuals
    across the reviews that claimed it? A small p is evidence that the field
    lacks a direction (a reason for a deliberate refit, never an automatic one)."""
    rid, U, claims = _stream(tri)
    if len(rid) < 5:
        return dict(reviews=len(rid))
    sp, ix = tri.space, tri.catalog.index
    N = len(tri.catalog.ids)
    E = np.zeros((len(rid), N))
    for k, r in enumerate(rid):
        q = sigmoid(sp.point(tri.ledger.by_id[r].original, loo=True).logits)
        for it, y in claims[r]:
            E[k, ix[it]] = y - q[ix[it]]
    cols = np.flatnonzero(np.abs(E).sum(0) > 0)
    E = E[:, cols]
    if E.shape[1] < 2:
        return dict(reviews=len(rid), items=int(E.shape[1]))
    u, s, vt = np.linalg.svd(E, full_matrices=False)
    s1 = float(s[0])
    rng = np.random.default_rng(seed)
    nz = [np.flatnonzero(E[:, j] != 0) for j in range(E.shape[1])]
    null = []
    for _ in range(permutations):
        Ep = np.zeros_like(E)
        for j, rows in enumerate(nz):
            Ep[rng.permutation(rows), j] = E[rows, j]
        null.append(float(np.linalg.svd(Ep, compute_uv=False)[0]))
    null = np.array(null)
    v = vt[0]
    o = np.argsort(-np.abs(v))[:top]
    return dict(reviews=len(rid), items=int(E.shape[1]), s1=s1, energy=float(s1 ** 2 / max((s ** 2).sum(), 1e-12)),
                null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)),
                p=float((1 + (null >= s1).sum()) / (1 + permutations)),
                direction=[dict(item=tri.catalog.ids[cols[j]], loading=round(float(v[j]), 3)) for j in o],
                singular=s[:10].tolist(), null=null.tolist())
