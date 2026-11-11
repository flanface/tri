"""Sidegrades: two clusters of picks that builds of one archetype take
instead of each other, each by a substantial share of them.

Within one region of the concept space (one archetype), most picks are either
nearly always taken or nearly never. The contested ones (taken by 15% to 85%
of the region's builds that can take them) are where builds differ. A
sidegrade is a split of contested picks into two sides A and B such that

  - commitment: most builds take mostly one side (side score
    mean(A picked) - mean(B picked) is at least 0.5 or at most -0.5);
  - balance: each side is chosen by at least min_share of the region's builds,
    so neither is a rare variant of the other: the corpus treats them as
    comparable choices (equal worth in the pick-rate sense of tri.worth);
  - separation: every item on a side is taken at least `gap` more often by
    builds committed to its side than by builds committed to the other;
  - substitution: picks on opposite sides are negatively correlated (mean
    phi at most -min_cross) and picks on one side positively (mean phi at
    least min_within, for sides of two or more items). Commitment alone is not
    enough: independent items also produce one-sided builds by chance.

A strong sidegrade is a large source of variation, so the concept space may
split one archetype into two regions along it. Mining therefore searches each
pair of regions that share most of their core picks (Jaccard overlap of the
items taken by at least 60% of each region's builds of at least
pair_overlap; such a pair is one archetype with two variants) and each region.
The builds searched are those holding 80% of the shared core, chosen by the
core alone and never by the items tested, so the choice of builds cannot
itself make two items look like substitutes.

Mining is held-out: the sides are found by a weighted 2-means split of one
random half of the region's builds and kept only if commitment, balance and
separation hold on the other half (items that fail separation there are
dropped). Nothing is fitted to reviews or locators.

Inference uses a sidegrade in two ways (tri.infer):
  - no mixing: worth-based adds never take an item of the side the answer
    holds fewer of (required picks from rules and reviews are not affected);
  - offer: an answer committed to one side lists the other as a sidegrade
    (the picks to remove, the feasible picks to add, and both shares)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from tri.analysis.defs import Sidegrade, SidegradeConfig
from tri.core.types import Kind


def _two_means(X: np.ndarray, w: np.ndarray, rng, restarts: int, iters: int) -> np.ndarray:
    """Weighted 2-means on binary rows; labels of the lowest-inertia restart."""
    best, best_in = None, np.inf
    n = len(X)
    for _ in range(restarts):
        i = rng.choice(n, p=w / w.sum())
        d = ((X - X[i]) ** 2).sum(1)
        j = rng.choice(n, p=(w * d) / max((w * d).sum(), 1e-12)) if d.sum() > 0 else rng.choice(n)
        C = np.stack([X[i], X[j]]).astype(np.float64)
        lab = np.zeros(n, int)
        for _ in range(iters):
            D = ((X[:, None, :] - C[None]) ** 2).sum(2)
            new = D.argmin(1)
            for k in (0, 1):
                m = new == k
                if m.any():
                    C[k] = (w[m] @ X[m]) / w[m].sum()
            if (new == lab).all():
                break
            lab = new
        inertia = float(w @ ((X - C[lab]) ** 2).sum(1))
        if inertia < best_in:
            best, best_in = lab.copy(), inertia
    return best


#: v3.7: rows per float block in weighted column sums (the bool tables are
#: never cast whole: a float copy of a 300k-row region was 1.7 GB per table).
ROW_CHUNK = 16384


def _wsum(w, M, rows) -> np.ndarray:
    """w[rows] @ M[rows] with M bool (or 0/1), cast to float one block at a time."""
    out = np.zeros(M.shape[1])
    for a in range(0, len(rows), ROW_CHUNK):
        r = rows[a:a + ROW_CHUNK]
        out += w[r] @ M[r].astype(np.float64)
    return out


def _rate(X, A, w, m) -> np.ndarray:
    rows = np.flatnonzero(m)
    num = _wsum(w, X, rows); den = _wsum(w, A, rows)
    return num / np.maximum(den, 1e-12)


def _avail_cols(data, rows, cols=None) -> np.ndarray:
    """[len(rows), N or len(cols)] bool availability of the rows, gathered in blocks
    (data.masks[data.gid] was a whole [B, N] copy before the column cut)."""
    rows = np.asarray(rows)
    n = data.masks.shape[1] if cols is None else len(cols)
    out = np.empty((len(rows), n), bool)
    for a in range(0, len(rows), ROW_CHUNK):
        m = np.asarray(data.masks[data.gid[rows[a:a + ROW_CHUNK]]], bool)
        out[a:a + ROW_CHUNK] = m if cols is None else m[:, cols]
    return out


def _score(X, A, ia, ib) -> np.ndarray:
    """Side score per row: share of a's available items picked minus b's."""
    fa = X[:, ia].sum(1) / np.maximum(A[:, ia].sum(1), 1)
    fb = X[:, ib].sum(1) / np.maximum(A[:, ib].sum(1), 1)
    return fa - fb


def _one(X, A, w, h1, h2, cand, rng, cfg):
    Xc = X[:, cand][h1].astype(np.float64)
    if len(Xc) < cfg.min_builds // 2:
        return None
    lab = _two_means(Xc, w[h1], rng, cfg.restarts, cfg.iters)
    if lab is None or lab.min() == lab.max():
        return None
    X1, A1 = X[h1], A[h1]
    d = _rate(X1, A1, w[h1], lab == 0)[cand] - _rate(X1, A1, w[h1], lab == 1)[cand]
    del X1, A1
    order = np.argsort(-np.abs(d))
    ia = [cand[k] for k in order if d[k] >= cfg.gap][: cfg.max_items]
    ib = [cand[k] for k in order if d[k] <= -cfg.gap][: cfg.max_items]
    if not ia or not ib:
        return None
    # held-out half: separation per item, then commitment and balance
    X2, A2, w2 = X[h2], A[h2], w[h2]
    for _ in range(2):
        sc = _score(X2, A2, ia, ib)
        ma, mb = sc >= 0.5, sc <= -0.5
        if not ma.any() or not mb.any():
            return None
        dd = _rate(X2, A2, w2, ma) - _rate(X2, A2, w2, mb)
        ia = [j for j in ia if dd[j] >= cfg.gap]
        ib = [j for j in ib if dd[j] <= -cfg.gap]
        if not ia or not ib:
            return None
    C = np.corrcoef(X2[:, list(ia) + list(ib)].astype(np.float64).T)
    C = np.nan_to_num(C)
    na = len(ia)
    cross = float(C[:na, na:].mean())

    def within(blk):
        k = len(blk)
        return 1.0 if k < 2 else float((blk.sum() - np.trace(blk)) / (k * (k - 1)))
    if cross > -cfg.min_cross or within(C[:na, :na]) < cfg.min_within or within(C[na:, na:]) < cfg.min_within:
        return None
    sc = _score(X2, A2, ia, ib)
    W = w2.sum()
    sa, sb = float(w2[sc >= 0.5].sum() / W), float(w2[sc <= -0.5].sum() / W)
    if sa + sb < cfg.min_commit or min(sa, sb) < cfg.min_share:
        return None
    ma, mb = sc >= 0.5, sc <= -0.5
    dd = _rate(X2, A2, w2, ma) - _rate(X2, A2, w2, mb)
    held = dict(share_a=round(sa, 4), share_b=round(sb, 4), commit=round(sa + sb, 4), builds=int(len(X2)),
                cross=round(cross, 3))
    return np.array(ia), np.array(ib), dict(held, gap_a={}, gap_b={}), dd


def _pools(data, regions: np.ndarray, cfg: SidegradeConfig) -> List[Tuple[List[int], List[int]]]:
    """(regions, core items) for each pair of regions sharing most of their core
    picks (first) and for each region."""
    ids = [int(r) for r in np.unique(regions)]
    core = {}
    for r in ids:
        rows = np.flatnonzero(regions == r)
        X = (data.X[rows] > 0).toarray()
        A = _avail_cols(data, rows) | X
        w = np.asarray(data.w[rows], float)
        rate = _rate(X, A, w, np.ones(len(rows), bool))
        core[r] = set(np.flatnonzero(rate >= cfg.core_rate).tolist())
    pools = []
    for i, r in enumerate(ids):
        for q in ids[i + 1:]:
            u = core[r] | core[q]
            if u and len(core[r] & core[q]) / len(u) >= cfg.pair_overlap:
                pools.append(([r, q], sorted(core[r] & core[q])))
    return pools + [([r], sorted(core[r])) for r in ids if core[r]]


def _holders(data, core: Sequence[int], hold: float) -> np.ndarray:
    """Rows that take at least `hold` of the core items they can take. The pool
    is chosen by the shared core only, never by the items being tested, so the
    selection cannot create a correlation between them (Berkson's paradox)."""
    X = (data.X[:, core] > 0).toarray()
    A = _avail_cols(data, np.arange(len(data)), list(core)) | X
    return np.flatnonzero(X.sum(1) >= hold * np.maximum(A.sum(1), 1))


def mine(data, regions: np.ndarray, catalog, cfg: Optional[SidegradeConfig] = None) -> List[Sidegrade]:
    """Sidegrades of every region (and every pair of variant regions) of the
    corpus (data: BuildMatrix; regions: [len(data)] region id per build), with
    held-out shares, per-item gaps and the cross-side correlation."""
    cfg = cfg or SidegradeConfig()
    rng = np.random.default_rng(cfg.seed)
    out: List[Sidegrade] = []
    seen = set()
    kinds = np.array([catalog.items[i].kind in (Kind.TALENT, Kind.MANTRA) for i in catalog.ids])
    regions = np.asarray(regions)
    for pool, core in _pools(data, regions, cfg):
        rows = _holders(data, core, cfg.core_hold)
        if len(rows) < cfg.min_builds:
            continue
        X = (data.X[rows] > 0).toarray()
        A = _avail_cols(data, rows) | X
        w = np.asarray(data.w[rows], float)
        if w.sum() <= 0:
            continue
        rate = _rate(X, A, w, np.ones(len(rows), bool))
        cnt = X.sum(0)
        cand = np.flatnonzero(kinds & (rate >= cfg.contested[0]) & (rate <= cfg.contested[1])
                              & (cnt >= cfg.min_count))
        cand = np.setdiff1d(cand, core)
        perm = rng.permutation(len(rows))
        h1 = np.zeros(len(rows), bool); h1[perm[: len(rows) // 2]] = True
        h2 = ~h1
        found = 0
        while found < cfg.max_per_region and len(cand) >= 2:
            res = _one(X, A, w, h1, h2, cand, rng, cfg)
            if res is None:
                break
            ia, ib, held, dd = res
            key = (frozenset(int(j) for j in ia), frozenset(int(j) for j in ib))
            cand = np.setdiff1d(cand, np.r_[ia, ib])
            found += 1
            if key in seen or key[::-1] in seen:
                continue
            seen.add(key)
            held["gap_a"] = {catalog.ids[j]: round(float(dd[j]), 3) for j in ia}
            held["gap_b"] = {catalog.ids[j]: round(float(-dd[j]), 3) for j in ib}
            out.append(Sidegrade(int(pool[0]), [catalog.ids[j] for j in ia], [catalog.ids[j] for j in ib],
                                 regions=[int(r) for r in pool], core=[catalog.ids[j] for j in core],
                                 core_hold=cfg.core_hold, **held))
    return out
