"""Worth: how strongly builds with this archetype take each item.

worth_t(b) combines the worth field with the builds around b (math §4.8):

    q_t = ( m sigmoid(logit_t(b)) + sum_n k_n rho_nt x_nt ) / ( m + sum_n k_n rho_nt a_nt )

    logit_t(b)  the field's leave-one-out logit (tri.space): what the rest of
                the build says about t
    n           the `k` nearest reference builds in the concept space
                (corpus builds and reviewed builds)
    k_n         w_n exp(-d(b, n)^2 / 2h^2), h = `bandwidth`
    x_nt, a_nt  n took t / n could take t
    m           prior_strength: pseudo-builds of the field's answer
    rho_nt      patch_factor for pre-patch builds on items the patch changed;
                review_weight for the items a review added or removed;
                0 for corpus builds older than the item itself (an item added
                to the game after the build could not have been taken)

Because concept distance bounds worth difference (|dq| <= c_t d, tri.space),
neighbours inside the bandwidth have nearly the same field worth: the average
is local evidence about the same quantity, not a vote of other archetypes.

Reviewed builds enter in their corrected form at the coordinates of the
original build (where unreviewed look-alikes sit). For the items the review
changed they count `review_weight` times, so one review moves worth for those
items on every build near it at once; for unmarked items they count once."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from tri.model.defs import FieldPoint, LocalRates, WorthConfig
from tri.model.matrix import BuildMatrix
from tri.model.space import sigmoid


class RowStack:
    """Reference coordinates without a copy (v3.7): rows `idx` of a shared base
    (the corpus coordinates Tri._Zdata, never copied) followed by a few extra
    rows (the reviewed builds). Indexing, row norms and products with a query
    point are what FieldWorth needs; a float32 base is cast in row chunks."""

    CHUNK = 16384

    def __init__(self, base: np.ndarray, idx: Optional[np.ndarray] = None, extra: Optional[np.ndarray] = None):
        self.base = base
        self.idx = None if idx is None else np.asarray(idx, np.int64)
        d = base.shape[1] if base.ndim == 2 else 0
        self.extra = np.zeros((0, d)) if extra is None or not len(extra) else np.asarray(extra, np.float64)
        self.nb = len(base) if self.idx is None else len(self.idx)

    def __len__(self) -> int:
        return self.nb + len(self.extra)

    @property
    def shape(self):
        return (len(self), self.base.shape[1])

    def _chunks(self, fn):
        out = [fn(self.base[a:a + self.CHUNK].astype(np.float64, copy=False))
               for a in range(0, len(self.base), self.CHUNK)]
        return np.concatenate(out) if out else np.zeros(0)

    def _of_base(self, v: np.ndarray) -> np.ndarray:
        return v if self.idx is None else v[self.idx]

    def sqnorm(self) -> np.ndarray:
        b = self._chunks(lambda U: (U * U).sum(1))
        return np.concatenate([self._of_base(b), (self.extra * self.extra).sum(1)])

    def dot(self, u: np.ndarray) -> np.ndarray:
        b = self.base @ u if self.base.dtype == np.float64 else self._chunks(lambda U: U @ u)
        return np.concatenate([self._of_base(b), self.extra @ u])

    def d2(self, u: np.ndarray, U2: np.ndarray, kmax: int, exclude=()) -> np.ndarray:
        """Squared distances to u, exact (float64) on every row that can be
        among the kmax nearest and +inf elsewhere (v3.8). A float32 base is
        screened in float32 with no cast or copy; rows whose screened distance
        exceeds the kmax-th smallest by more than twice the rounding bound
        (n+2) 2^-22 |b||u| per side cannot be among the kmax nearest, so only
        the survivors are recomputed in float64. A float64 base is exact
        throughout."""
        u = np.asarray(u, np.float64)
        uu = float(u @ u)
        n = len(self)
        if self.base.dtype == np.float64 or kmax >= n or self.nb == 0:
            d = np.maximum(U2 - 2 * self.dot(u) + uu, 0)
            if len(exclude):
                d[np.asarray(exclude)] = np.inf
            return d
        a = self.base @ u.astype(np.float32)                     # float32 screen, no copy
        a = np.concatenate([self._of_base(a).astype(np.float64), self.extra @ u])
        approx = U2 - 2 * a + uu
        if len(exclude):
            approx[np.asarray(exclude)] = np.inf
        fin = np.isfinite(approx)
        if fin.sum() <= kmax:
            d = np.maximum(U2 - 2 * self.dot(u) + uu, 0)
            d[~fin] = np.inf
            return d
        if getattr(self, "_bmax", None) is None:
            self._bmax = float(np.sqrt(U2.max())) if len(U2) else 0.0
        D = self.base.shape[1]
        err = 2 * 4 * (D + 2) * 2.0 ** -22 * self._bmax * np.sqrt(uu)      # 2 x |dot error|, 4x margin
        T = np.partition(approx, kmax - 1)[kmax - 1]
        cand = np.flatnonzero(approx <= T + 2 * err)
        d = np.full(n, np.inf)
        d[cand] = np.maximum(U2[cand] - 2 * (self[cand] @ u) + uu, 0)
        return d

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        out = np.empty((len(rows), self.base.shape[1]))
        inb = rows < self.nb
        r = rows[inb]
        out[inb] = self.base[r if self.idx is None else self.idx[r]]
        out[~inb] = self.extra[rows[~inb] - self.nb]
        return out


def smallest(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest entries of d in row order; rows at the same
    distance as the k-th are taken in row order too (v3.8), so the neighbour
    set and its order never depend on how the partition breaks ties."""
    nb = np.argpartition(d, k - 1)[:k]
    t = d[nb].max()
    eq = np.flatnonzero(d == t)
    if len(eq) == int((d[nb] == t).sum()):
        return np.sort(nb)                              # no tie straddles the boundary
    lo = np.flatnonzero(d < t)
    return np.sort(np.concatenate([lo, eq[: k - len(lo)]]))


class FieldWorth:
    """Reference builds with concept coordinates; answers worth around a FieldPoint."""

    def __init__(self, ref: BuildMatrix, U: np.ndarray, cfg: Optional[WorthConfig] = None,
                 is_review: Optional[np.ndarray] = None, touched=None, conditions=None, barrier=None,
                 chasm_corpus: bool = False):
        self.cfg = cfg or WorthConfig()
        self.ref = ref
        self.U = U if isinstance(U, RowStack) else RowStack(np.asarray(U, np.float64))
        self.U2 = self.U.sqnorm()
        self.is_review = np.zeros(len(ref), bool) if is_review is None else is_review
        self.touched = touched                        # csr [rows, N]: 1 where a review added/removed the item
        # {row: {col: (related cols)}}: a change made "because of y" (redundant
        # with y, works against y) is evidence only for builds that have y;
        # for other builds the row abstains on that item.
        self.conditions = conditions or {}
        self.change_time = np.full(ref.N, -np.inf)    # [N] latest patch time per item
        self.release_time = np.full(ref.N, -np.inf)   # [N] when the item entered the game
        # (u, U_rows) -> [rows] in [0, 1]: scales a reviewed build's boost when the
        # administrators' locators put it and the query under different archetypes
        self.barrier = barrier
        self.chasm_corpus = bool(chasm_corpus) and barrier is not None   # v2.19: the chasm gates corpus rows too

    @property
    def prior(self) -> np.ndarray:
        """[N] population pick rate among builds that can take each item (popularity baseline)."""
        return self.pick_rate()[0]

    def pick_rate(self):
        """(rate [N], eligible weight [N]) of the reference builds, computed once."""
        if getattr(self, "_rate", None) is None:
            self._rate = self.ref.pick_rate()
        return self._rate

    def set_change_times(self, times: dict) -> None:
        self.change_time[:] = -np.inf
        for t, v in times.items():
            self.change_time[t] = v

    def set_release_times(self, times: dict) -> None:
        self.release_time[:] = -np.inf
        for t, v in times.items():
            self.release_time[t] = v

    def rates(self, pt: FieldPoint, exclude: Sequence[int] = (), picks: Optional[Sequence[int]] = None) -> LocalRates:
        """Worth around the point. exclude: reference rows to skip. picks: the
        query build's item indices (for conditional review changes)."""
        f = self.cfg
        base = sigmoid(pt.logits)
        N = self.ref.N
        if len(self.U) == 0:
            return LocalRates(base.copy(), base, np.zeros(N), np.zeros(0, int), np.zeros(0), 0.0, 0.0)
        u = pt.u
        kmax = f.k + (int(f.k_cross) if f.cross_weight > 0 else 0)
        d2 = self.U.d2(u, self.U2, kmax, exclude)
        k = min(f.k, len(d2))
        nb = smallest(d2, k)
        kw = self.ref.w[nb] * np.exp(-d2[nb] / (2 * f.bandwidth ** 2))
        X = self.ref.X[nb].toarray()
        A = self.ref.masks[self.ref.gid[nb]].astype(np.float64)
        chi = None
        if self.chasm_corpus:                                   # borders are chasms (math §5.4)
            chi = np.asarray(self.barrier(u, self.U[nb]), float)
            kw = kw * np.where(self.is_review[nb], 1.0, chi)
        fac = None
        if np.isfinite(self.change_time).any():
            ts = np.nan_to_num(self.ref.ts[nb], nan=-np.inf)
            old = (ts[:, None] < self.change_time[None, :]) & ~self.is_review[nb, None]
            fac = np.where(old, f.patch_factor, 1.0)
        if np.isfinite(self.release_time).any():
            ts = np.nan_to_num(self.ref.ts[nb], nan=-np.inf)
            pre = (ts[:, None] < self.release_time[None, :]) & ~self.is_review[nb, None]
            fac = np.where(pre, 0.0, 1.0 if fac is None else fac)
        if self.touched is not None and self.is_review[nb].any():
            T = self.touched[nb].toarray()
            if T.any():
                if self.barrier is not None:
                    bf = np.ones(len(nb))
                    rv = self.is_review[nb]
                    bf[rv] = chi[rv] if chi is not None else self.barrier(u, self.U[nb[rv]])
                    T = T * bf[:, None]
                boost = 1.0 + (f.review_weight - 1.0) * T
                if self.conditions:
                    have = set(int(i) for i in picks) if picks is not None else set()
                    for j, row in enumerate(nb):
                        for col, rel in self.conditions.get(int(row), {}).items():
                            pos = [c for c in rel if c >= 0]
                            neg = [-c - 1 for c in rel if c < 0]          # NW: holds where one is missing
                            if not all(c in have for c in pos) or (neg and all(c in have for c in neg)):
                                boost[j, col] = 0.0        # abstain: the change was about items this build lacks
                fac = boost if fac is None else fac * boost
        if fac is not None:
            num = (kw[:, None] * fac * X).sum(0); den = (kw[:, None] * fac * A).sum(0)
        else:
            num = kw @ X; den = kw @ A
        if f.cross_weight > 0 and len(d2) > k:
            num, den = self._cross(d2, nb, num, den, u)
        m = f.prior_strength
        q = (num + m * base) / (den + m)
        return LocalRates(q, base, den, nb, kw, _wmedian(self.ref.n_tal[nb], kw), _wmedian(self.ref.n_man[nb], kw))

    def _cross(self, d2: np.ndarray, nb: np.ndarray, num: np.ndarray, den: np.ndarray, u=None):
        """Add the cross-archetype ring to the kernel sums (docs/math eq. 5.4)."""
        f = self.cfg
        k2 = min(int(f.k_cross) + len(nb), len(d2))
        ring = smallest(d2, k2)
        keep = ~np.isin(ring, nb) & ~self.is_review[ring] & np.isfinite(d2[ring])
        ring = ring[keep]
        if not len(ring):
            return num, den
        cw = f.cross_weight * self.ref.w[ring] * np.exp(-d2[ring] / (2 * f.cross_bandwidth ** 2))
        if self.chasm_corpus and u is not None:              # the ring crosses variants, not families (v2.19)
            cw = cw * np.asarray(self.barrier(u, self.U[ring]), float)
        Xr = self.ref.X[ring].toarray()
        Ar = self.ref.masks[self.ref.gid[ring]].astype(np.float64)
        if np.isfinite(self.release_time).any():
            ts = np.nan_to_num(self.ref.ts[ring], nan=-np.inf)
            pre = ts[:, None] < self.release_time[None, :]
            Xr = np.where(pre, 0.0, Xr); Ar = np.where(pre, 0.0, Ar)
        thin = f.prior_strength / (f.prior_strength + den)      # lambda_t: the ring acts where local support is thin
        return num + thin * (cw @ Xr), den + thin * (cw @ Ar)


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    if len(v) == 0 or w.sum() <= 0:
        return 0.0
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])
