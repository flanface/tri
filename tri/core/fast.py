"""Small numeric shortcuts shared by the hot paths (v3.1).

sqdist: pairwise squared distances. Above a small size it expands
|a - b|^2 = |a|^2 + |b|^2 - 2 a.b so the work is one matrix product (BLAS)
instead of a [q, m, D] cube; below it the direct form keeps exact zeros.
"""
from __future__ import annotations

import numpy as np


def sqdist(A: np.ndarray, B: np.ndarray, small: int = 64) -> np.ndarray:
    """[q, m] squared distances between the rows of A [q, D] and B [m, D], clipped at 0."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    if len(A) <= small:
        return np.maximum(((A[:, None, :] - B[None]) ** 2).sum(-1), 0.0)
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2.0 * (A @ B.T), 0.0)
