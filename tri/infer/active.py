"""Review queue: which builds experts should see next.

A review teaches the most where tri knows the least. For each pool build:

    novelty     1 - archetype similarity to the nearest reviewed build
                (nothing like it has been reviewed yet)
    luxury      share of its picks no pass rule covers (tri is relying on
                worth alone there)
    ambiguity   second region membership / first (archetype unclear)
    patch       share of its picks touched by a patch since the nearest
                review was written
    edits       how much tri would change (its own answer is at stake)

    priority = novelty * (0.5 + luxury) + 0.5 * ambiguity + patch + 0.1 * edits

Archetype similarity is the review ledger's kernel of concept distance
(ReviewLedger.reach_kernel: 0.95 within d ~ 0.58, 0.80 at d = reach). Builds
whose archetype nearly duplicates one already chosen (similarity >= 0.95) are
skipped, and random_share of the queue is uniform random so coverage
never depends only on tri's own beliefs."""
from __future__ import annotations

import random
from typing import List, Sequence

import numpy as np

from tri.core.types import Build
from tri.infer.defs import QueueItem
from tri.rules.passes import LUXURY


class ReviewQueue:
    def __init__(self, tri, random_share: float = 0.2, seed: int = 0):
        self.tri = tri
        self.random_share = random_share
        self.rng = random.Random(seed)

    def _sim(self):
        from tri.reviews.authority import ReviewLedger
        return ReviewLedger.reach_kernel(self.tri.cfg.ledger)

    def score(self, b: Build) -> QueueItem:
        tri = self.tri
        z = tri._embed(b)
        kern = self._sim()
        nov, newest = 1.0, None
        if tri.ledger is not None and tri.ledger.reviews:
            sims = [kern(tri.ledger.vector(r.review_id), z) for r in tri.ledger.reviews]
            j = int(np.argmax(sims))
            nov, newest = 1 - max(0.0, sims[j]), tri.ledger.reviews[j]
        ex = tri.explain(b)
        n = max(1, sum(len(ex[k]) for k in ("foundational", "central", "luxury")))
        lux = len(ex[LUXURY]) / n
        amb = tri.space.reading(z).ambiguity
        ct = tri.patches.change_times()
        t0 = (newest.timestamp or 0.0) if newest is not None else 0.0
        items = [i for i in b.items() if i in tri.catalog.index]
        pt = sum(1 for i in items if ct.get(i, -np.inf) > t0) / max(1, len(items)) if ct else 0.0
        edits = len(tri.review(b, 1, keep_size=True)[0].changes)
        pr = nov * (0.5 + lux) + 0.5 * amb + pt + 0.1 * edits
        return QueueItem(b, pr, nov, lux, amb, pt, edits, "active")

    def next(self, pool: Sequence[Build], k: int = 10) -> List[QueueItem]:
        pool = list(pool)
        self.rng.shuffle(pool)
        n_rand = int(round(k * self.random_share))
        rand = [QueueItem(b, 0.0, 0, 0, 0, 0, 0, "random") for b in pool[:n_rand]]
        scored = sorted((self.score(b) for b in pool[n_rand:]), key=lambda q: -q.priority)
        chosen: List[QueueItem] = []
        vecs: List[np.ndarray] = []
        kern = self._sim()
        for q in scored:
            v = self.tri._embed(q.build)
            if any(kern(v, u) >= 0.95 for u in vecs):
                continue
            chosen.append(q); vecs.append(v)
            if len(chosen) >= k - n_rand:
                break
        return chosen + rand
# ------------------------------------------------------------------ coverage (math §12.3)


def coverage(U_pool: np.ndarray, U_reviewed: np.ndarray) -> dict:
    """Distance from each pool build to its nearest reviewed build. The radius
    eps = max is a worth guarantee: every pool build is within RMS worth
    distance eps / 4 of a reviewed one (Theorem 4.2)."""
    U_pool = np.atleast_2d(np.asarray(U_pool, float))
    if len(U_reviewed) == 0 or len(U_pool) == 0:
        return dict(radius=float("inf"), distances=np.full(len(U_pool), np.inf), median=float("inf"), p90=float("inf"))
    R = np.atleast_2d(np.asarray(U_reviewed, float))
    d = np.full(len(U_pool), np.inf)
    for a in range(0, len(R), 256):
        d = np.minimum(d, np.sqrt(np.maximum(((U_pool[:, None] - R[None, a:a + 256]) ** 2).sum(-1), 0)).min(1))
    return dict(radius=float(d.max()), distances=d, median=float(np.median(d)), p90=float(np.quantile(d, 0.9)))


def kcenter_queue(U_pool: np.ndarray, U_reviewed: np.ndarray, k: int = 10) -> List[dict]:
    """Farthest-point order (Gonzalez): repeatedly the pool build farthest from
    everything reviewed or already queued. A 2-approximation to the k-centre
    cover, so it lowers the coverage radius about as fast as any queue can.
    Returns [dict(row, distance, radius_after)]."""
    U_pool = np.atleast_2d(np.asarray(U_pool, float))
    if len(U_pool) == 0:
        return []
    d = coverage(U_pool, U_reviewed)["distances"] if len(U_reviewed) else np.full(len(U_pool), np.inf)
    out = []
    for _ in range(min(k, len(U_pool))):
        j = int(np.argmax(d)) if np.isfinite(d).any() else 0
        dj = float(d[j])
        d = np.minimum(d, np.sqrt(((U_pool - U_pool[j]) ** 2).sum(1)))
        out.append(dict(row=j, distance=dj, radius_after=float(d.max())))
    return out
