"""Item similarity in worth space: the mean-centred cosine of two items'
worth over the corpus builds (worth 0 where a build cannot take the item),

    C_st = sum_b w_b (p_bs - pbar_s)(p_bt - pbar_t) / (sd_s sd_t),

so items valued by the same builds more than usual are similar, whatever
their overall pick rate. Depends on the space and the corpus only."""
from __future__ import annotations

import numpy as np


def worth_cosine(space, bm, sample: int = 3000, seed: int = 0) -> np.ndarray:
    """[N, N] mean-centred worth cosine over (a sample of) the builds of bm."""
    if sample is not None and len(bm) > sample:
        rows = np.sort(np.random.default_rng(seed).choice(len(bm), sample, replace=False))
        bm = bm.subset(rows)
    L = space.logits_rows(bm)
    P = (1.0 / (1.0 + np.exp(-L))) * np.asarray(bm.masks[bm.gid], float)
    w = np.asarray(bm.w, float)
    w = w / max(w.sum(), 1e-12)
    C = (P - w @ P) * np.sqrt(w)[:, None]
    sd = np.sqrt(np.maximum((C ** 2).sum(0), 1e-12))
    R = (C.T @ C) / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    return R
