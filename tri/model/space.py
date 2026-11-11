"""Concept space: a logistic worth field whose coordinates measure worth differences.

Model (math §3). For build b and item t:

    logit_bt = beta_t + Gamma_t . s_b + Omega_t . c_b + V_t . z_b
    P(b takes t | t available) = sigmoid(logit_bt)

    s_b   stat steps: 1[highest stat >= v] for every stat and grid value v
    c_b   context one-hots (oath, murmur)
    z_b   the build's archetype, a latent vector of `dim` numbers:
          z_b = E c_b + r_b  (the context's usual archetype plus the build's own part)

Training: weighted logistic loss over available entries only, L2 on every
parameter, minibatch Adam. Pre-patch builds count `patch_factor` times for the
items a patch changed. After training every build's r_b is re-solved exactly:
for fixed item parameters the loss in r_b is strictly convex, so each build has
one deterministic latent (solved by a majorize-minimize iteration). New builds
get theirs the same way.

Coordinates. Write y_b = [z_b, s_b, c_b] and A = [V, Gamma, Omega] (one row per
item), so logit_b = beta + A y_b. Each item has a fixed metric weight omega_t
(mean 1 over items): by default the between-build variance of its worth in the
corpus, so items whose worth separates archetypes (an Agile Dash taken by 90% of
one playstyle and 5% of another) dominate proximity, and items nearly every or
no build takes (whose logits drift without meaning) barely count. With the QR
factorisation diag(omega)^(1/2) A = Q R,

    u_b = R (y_b - y_mean) / sqrt(N)

and ||u_b - u_b'|| equals the omega-weighted root-mean-square difference of the
two builds' logits, exactly. Because |sigmoid(a) - sigmoid(a')| <= |a - a'| / 4,

    weighted RMS worth difference(b, b') <= ||u_b - u_b'|| / 4
    |worth_t(b) - worth_t(b')|  <= sqrt(N / omega_t) ||Q_t|| ||u_b - u_b'|| / 4

metric="uniform" (omega = 1) is the unweighted special case (runs before v2.3).

so builds close in the concept space are guaranteed to have close worth, and
builds whose worth differs must be far apart. Similarity is the Gaussian
kernel exp(-d^2 / 2h^2) of that distance (h = `bandwidth`, logit units).

Regions: k-means on u gives named archetypes (groups of builds with nearly the
same worth profile); a region's latent is the mean z of its members (used to
answer "as if this build had that archetype")."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from tri.core.catalog import Catalog
from tri.core.types import Build
from tri.model.defs import ConceptReading, FieldPoint, SpaceConfig
from tri.model.matrix import CONTEXT_FIELDS, BuildMatrix


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.lr = lr; self.t = 0

    def step(self, p, g, rows=None, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        if rows is None:
            self.m = b1 * self.m + (1 - b1) * g
            self.v = b2 * self.v + (1 - b2) * g * g
            p -= self.lr * (self.m / (1 - b1 ** self.t)) / (np.sqrt(self.v / (1 - b2 ** self.t)) + eps)
        else:
            m = b1 * self.m[rows] + (1 - b1) * g
            v = b2 * self.v[rows] + (1 - b2) * g * g
            self.m[rows], self.v[rows] = m, v
            p[rows] -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)


class ConceptSpace:
    """Fit with ConceptSpace.fit(bm); coordinates with .coords(bm) or .project(build)."""

    def __init__(self, catalog: Catalog, cfg: Optional[SpaceConfig] = None):
        self.c = catalog
        self.cfg = cfg or SpaceConfig()
        self.tokens: List[str] = []
        self.fields: Tuple[str, ...] = CONTEXT_FIELDS
        self.beta = self.G = self.O = self.V = self.E = None
        self.ymean = self.R = self.Q = None
        self.omega: Optional[np.ndarray] = None        # [N] metric weights (mean 1); None = uniform
        self.Hinv = None
        self.centroids: Optional[np.ndarray] = None     # [R, D'] region centres in u
        self.region_z: Optional[np.ndarray] = None      # [R, dim] region latents
        self.region_names: List[str] = []              # display names (archetype label, if located, + auto name)
        self.region_auto: List[str] = []               # names from each region's characteristic items
        self.region_size: List[float] = []
        self.region_mixed: List[bool] = []             # no characteristic item: a catch-all region
        self.region_stability: Dict[str, float] = {}   # k -> median ARI (region_select "stable")
        self.train_loss: float = float("nan")
        self.explained: float = float("nan")          # deviance explained vs the pick-rate-only model
        self.fitted = None                             # [N] bool: items the field was fitted with
        self.validation: Dict[str, object] = {}        # held-out deviance, best / stopped step, curve

    # ------------------------------------------------------------ inputs
    def stat_steps(self, hi: np.ndarray) -> np.ndarray:
        """[b, S*len(grid)] 0/1 stat steps."""
        g = np.asarray(self.cfg.stat_grid)
        return (hi[:, :, None] >= g[None, None, :]).reshape(len(hi), -1).astype(np.float64)

    def _tokens_of(self, bm: BuildMatrix, rows) -> np.ndarray:
        if list(bm.tokens) == self.tokens:
            return bm.T[rows].toarray().astype(np.float64)
        T = np.zeros((len(rows), len(self.tokens)))
        tix = {t: k for k, t in enumerate(self.tokens)}
        Tc = bm.T[rows].tocoo()
        for r, c in zip(Tc.row, Tc.col):
            k = tix.get(bm.tokens[c])
            if k is not None:
                T[r, k] = 1.0
        return T

    def _inputs(self, bm: BuildMatrix, rows):
        S = self.stat_steps(bm.hi[rows].astype(np.float64))
        C = self._tokens_of(bm, rows)
        X = bm.X[rows].toarray().astype(np.float64)
        A = bm.masks[bm.gid[rows]].astype(np.float64)
        return S, C, X, A

    # ------------------------------------------------------------ fit
    def fit(self, bm: BuildMatrix, change_time: Optional[np.ndarray] = None,
            is_review: Optional[np.ndarray] = None, verbose: bool = False,
            release_time: Optional[np.ndarray] = None) -> "ConceptSpace":
        """Train the field on bm. change_time: [N] latest patch time per item
        (-inf none); is_review: [len(bm)] rows that are reviewed builds (never discounted,
        never held out); release_time: [N] time an item entered the game (-inf: always
        there). A corpus build older than an item's release could not take it, so that
        entry is left out of the loss instead of counting as a non-pick.

        A share of the corpus builds (val_frac) is held out of the item parameters.
        Each one's archetype is solved from (1 - val_mask) of its available entries and
        the field is scored on predicting the rest, which is how any new build gets
        its coordinates; the same protocol on training builds gives the gap. The
        check is reported (space.validation), never used to choose parameters: on
        the playstyle benchmark, held-out deviance prefers fields that erase weak
        archetype signals (math §4.4)."""
        f = self.cfg
        rng = np.random.default_rng(f.seed)
        if f.sample is not None and len(bm) > f.sample:
            keep = np.sort(rng.choice(len(bm), f.sample, replace=False))
            bm = bm.subset(keep)
            is_review = None if is_review is None else is_review[keep]
        rev = np.zeros(len(bm), bool) if is_review is None else np.asarray(is_review, bool)
        self.fitted = np.ones(len(self.c), bool)
        self._rel = None
        if release_time is not None and np.isfinite(np.asarray(release_time, float)).any():
            self._rel = (np.asarray(release_time, float), np.nan_to_num(bm.ts, nan=-np.inf), rev)
        self._train(bm, change_time, rev, self._val_split(bm, rev, rng), verbose)
        self._rel = None
        self._finish(bm)
        # exact latents, then the loss and deviance explained
        Rl = self.solve_latent(bm)
        self.train_loss, self.explained = self._deviance(bm, Rl)
        self._fit_regions(bm, Rl)
        return self

    def _val_split(self, bm: BuildMatrix, rev: np.ndarray, rng) -> Optional[dict]:
        """Validation builds (corpus rows only) and the scored share of their entries."""
        f = self.cfg
        cand = np.flatnonzero(~rev)
        if f.val_frac <= 0 or len(cand) < 5 * f.val_min:
            return None
        nv = int(min(f.val_max, max(f.val_min, round(f.val_frac * len(cand)))))
        vrows = np.sort(rng.choice(cand, nv, replace=False))
        train = np.setdiff1d(np.arange(len(bm)), vrows)
        out = dict(rows=vrows, train=train, held=self._masked(bm, vrows, rng))
        # the same protocol on training builds: the gap between the two is the overfit
        out["probe"] = self._masked(bm, np.sort(rng.choice(train, min(nv, len(train)), replace=False)), rng)
        return out

    def _released(self, rows, A: np.ndarray) -> np.ndarray:
        """A with entries of items released after the (corpus) build set to 0."""
        rel = getattr(self, "_rel", None)
        if rel is None:
            return A
        rt, ts, rev = rel
        return A * ~((ts[rows, None] < rt[None, :]) & ~rev[rows, None])

    def _masked(self, bm: BuildMatrix, rows, rng) -> dict:
        S, C, X, A = self._inputs_raw(bm, rows)
        A = self._released(rows, A)
        M = (rng.random(A.shape) < self.cfg.val_mask) & (A > 0)
        for i in np.flatnonzero(~M.any(1)):       # every scored build scores at least one entry
            av = np.flatnonzero(A[i] > 0)
            if len(av):
                M[i, rng.choice(av)] = True
        return dict(S=S, C=C, X=X, A_fit=A * ~M, A_score=A * M, w=bm.w[rows].astype(np.float64))

    def _inputs_raw(self, bm: BuildMatrix, rows):
        """_inputs before the token list is fixed (training uses bm's own tokens)."""
        S = self.stat_steps(bm.hi[rows].astype(np.float64))
        C = bm.T[rows].toarray().astype(np.float64)
        X = bm.X[rows].toarray().astype(np.float64)
        A = bm.masks[bm.gid[rows]].astype(np.float64)
        return S, C, X, A

    def _val_score(self, val: dict, r0: Optional[np.ndarray] = None):
        """Held-out deviance: latents from the fitted entries, log loss on the scored ones."""
        f = self.cfg
        self.Hinv = np.linalg.inv(0.25 * self.V.T @ self.V + f.l2_latent * np.eye(f.dim))
        r = self._mm(val["S"], val["C"], val["X"], val["A_fit"], r0)
        L = self.beta + val["S"] @ self.G.T + val["C"] @ self.O.T + (val["C"] @ self.E + r) @ self.V.T
        p = np.clip(sigmoid(L), 1e-9, 1 - 1e-9)
        X, M, w = val["X"], val["A_score"], val["w"]
        ll = (-(X * np.log(p) + (1 - X) * np.log(1 - p)) * M).sum(1)
        cnt = M.sum(1)
        dev = float(w @ ll / max(w @ cnt, 1e-12))
        per = ll / np.maximum(cnt, 1)                      # per-build mean, for the standard error
        ww = w / max(w.sum(), 1e-12)
        se = float(np.sqrt(max(ww @ (per - ww @ per) ** 2, 0.0) * (ww @ ww) / max(1 - ww @ ww, 1e-12)))
        return dev, se, r

    def _device(self):
        from tri.model import accel
        return accel.resolve(getattr(self.cfg, "device", "auto"))

    def _train(self, bm: BuildMatrix, change_time, rev: np.ndarray, val: Optional[dict], verbose: bool) -> None:
        dev = self._device()
        if dev is not None:
            from tri.model import accel
            return accel.train(self, bm, change_time, rev, val, verbose, dev)
        f = self.cfg
        rng = np.random.default_rng(f.seed + 1)
        self.tokens, self.fields = list(bm.tokens), tuple(bm.fields)
        tr = np.arange(len(bm)) if val is None else val["train"]
        N, K = bm.N, len(self.tokens)
        n = len(tr)
        P = len(self.c.stats) * len(f.stat_grid)
        d = f.dim
        w = bm.w / max(bm.w[tr].mean(), 1e-12)
        pr = bm.subset(tr).pick_rate()[0] if val is not None else bm.pick_rate()[0]
        self.beta = np.log(np.clip(pr, 1e-4, 1 - 1e-4) / (1 - np.clip(pr, 1e-4, 1 - 1e-4)))
        self.G = np.zeros((N, P)); self.O = np.zeros((N, K))
        self.V = rng.normal(0, 0.1, (N, d)); self.E = np.zeros((K, d))
        r = rng.normal(0, 0.1, (len(bm), d))
        ct = np.full(N, -np.inf) if change_time is None else np.asarray(change_time, float)
        ts = np.nan_to_num(bm.ts, nan=-np.inf)
        opt = {k: _Adam(getattr(self, k).shape, f.lr) for k in ("beta", "G", "O", "V", "E")}
        opt_r = _Adam(r.shape, f.lr)
        steps_per = max(1, int(np.ceil(n / f.batch)))
        epochs = max(f.epochs, int(np.ceil(f.min_steps / steps_per)))
        total = epochs * steps_per
        marks = set(np.unique(np.linspace(0, total, max(1, f.val_points) + 1)[1:].round().astype(int)).tolist())
        lam = f.l2_items
        step, rv = 0, None
        curve: List[Tuple[int, float]] = []
        for ep in range(epochs):
            perm = tr[rng.permutation(n)]
            for a in range(0, n, f.batch):
                rows = perm[a:a + f.batch]
                S, C, X, A = self._inputs(bm, rows)
                A = self._released(rows, A)
                if np.isfinite(ct).any():
                    A = A * np.where((ts[rows, None] < ct[None, :]) & ~rev[rows, None], f.patch_factor, 1.0)
                Z = C @ self.E + r[rows]
                L = self.beta + S @ self.G.T + C @ self.O.T + Z @ self.V.T
                wb = w[rows] / w[rows].sum()
                Gm = (sigmoid(L) - X) * A * wb[:, None]            # dLoss/dlogit
                gZ = Gm @ self.V
                opt["beta"].step(self.beta, Gm.sum(0))
                opt["G"].step(self.G, Gm.T @ S + lam * self.G)
                opt["O"].step(self.O, Gm.T @ C + lam * self.O)
                opt["V"].step(self.V, Gm.T @ Z + lam * self.V)
                opt["E"].step(self.E, C.T @ gZ + lam * self.E)
                opt_r.step(r, gZ + wb[:, None] * f.l2_latent * r[rows], rows=rows)
                step += 1
                if val is not None and step in marks and step < total:
                    dev, _, rv = self._val_score(val["held"], rv)
                    curve.append((step, dev))
                    if verbose:
                        print(f"  step {step}/{total}: held-out deviance {dev:.5f}")
        self._validation_report(val, bm, tr, rv, curve, step)

    def _validation_report(self, val: Optional[dict], bm: BuildMatrix, tr, rv, curve, step: int) -> None:
        """space.validation after training (shared by the NumPy and torch paths)."""
        f = self.cfg
        if val is None:
            self.validation = {}
            return
        dev, se, _ = self._val_score(val["held"], rv)
        curve.append((step, dev))
        pdev, _, _ = self._val_score(val["probe"])
        prt = bm.subset(tr).pick_rate()[0]
        base = self._base_deviance(val["held"], prt)
        pbase = self._base_deviance(val["probe"], prt)
        # explained: held-out builds; probe_explained: training builds under the same masked protocol
        # (the like-for-like comparison; deviance_explained in report() is the full-entry training fit)
        self.validation = dict(builds=int(len(val["rows"])), scored_share=f.val_mask, steps=int(step),
                               deviance=round(dev, 6), deviance_se=round(se, 6), probe_deviance=round(pdev, 6),
                               gap=round(dev - pdev, 6), explained=round(1 - dev / max(base, 1e-12), 4),
                               probe_explained=round(1 - pdev / max(pbase, 1e-12), 4),
                               lowest=round(min(c[1] for c in curve), 6),
                               curve=[(int(a), round(b, 6)) for a, b in curve])

    @staticmethod
    def _base_deviance(held: dict, pr: np.ndarray) -> float:
        """Held-out deviance of the pick-rate-only model (the explained share's baseline)."""
        p = np.clip(pr, 1e-6, 1 - 1e-6)
        X, M, w = held["X"], held["A_score"], held["w"]
        ll = (-(X * np.log(p) + (1 - X) * np.log(1 - p)) * M).sum(1)
        return float(w @ ll / max(w @ M.sum(1), 1e-12))

    def _finish(self, bm: BuildMatrix) -> None:
        f = self.cfg
        d = f.dim
        self.Hinv = np.linalg.inv(0.25 * self.V.T @ self.V + f.l2_latent * np.eye(d))
        rows = np.arange(len(bm)) if len(bm) <= 50000 else np.sort(
            np.random.default_rng(f.seed).choice(len(bm), 50000, replace=False))
        Y = self._y_rows(bm, rows, self.solve_latent(bm, rows))
        self.ymean = (bm.w[rows] @ Y) / max(bm.w[rows].sum(), 1e-12)
        self.omega = self.metric_weights(bm, rows, Y) if f.metric != "uniform" else None
        self._factor()

    def _factor(self) -> None:
        """QR of the weighted item matrix: diag(omega)^(1/2) A = Q R."""
        A = self._A()
        if self.omega is not None:
            A = A * np.sqrt(self.omega)[:, None]
        self.Q, self.R = np.linalg.qr(A)                           # reduced: Q [N, D], R [D, D]

    def metric_weights(self, bm: BuildMatrix, rows=None, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """[N] omega_t: (between-build variance of item t's worth sigma(l_bt) over the builds
        it is available to, times its availability share (w-weighted))^metric_power, floored at
        metric_floor * mean, scaled to mean 1, then (metric_calibrate) multiplied by
        the scale constant c^2 of _calibration."""
        rows = np.arange(len(bm)) if rows is None else np.asarray(rows)
        if Y is None:
            Y = self._y_rows(bm, rows, self.solve_latent(bm, rows))
        om = self._variance_weights(bm, rows, Y)
        if self.cfg.metric_calibrate:
            om = om * self._calibration(Y, om)
        return om

    def _calibration(self, Y: np.ndarray, om: np.ndarray, n: int = 3000) -> float:
        """c^2 such that median distance to the k-th nearest neighbour under c^2 omega equals
        that under omega = 1 (on a fixed sample): the bandwidth keeps its local meaning,
        only which builds are neighbours changes."""
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(len(Y), min(n, len(Y)), replace=False)
        D = Y[idx] - self.ymean
        A = self._A()

        def kth(weights):
            _, R = np.linalg.qr(A * np.sqrt(weights)[:, None])
            U = D @ R.T
            sq = (U * U).sum(1)
            d2 = np.maximum(sq[:, None] - 2 * U @ U.T + sq[None], 0)
            np.fill_diagonal(d2, np.inf)
            k = min(self.cfg.metric_k, len(U) - 1)
            return float(np.median(np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])))
        if len(idx) < 3:
            return 1.0
        a, b = kth(np.ones(len(om))), kth(om)
        return (a / b) ** 2 if a > 0 and b > 0 else 1.0

    def _variance_weights(self, bm: BuildMatrix, rows, Y: np.ndarray) -> np.ndarray:
        w = bm.w[rows].astype(np.float64)
        Aw = self._A()
        s0 = np.zeros(len(self.c)); s1 = np.zeros(len(self.c)); s2 = np.zeros(len(self.c))
        for a in range(0, len(rows), 8192):
            sl = slice(a, a + 8192)
            _, _, _, Av = self._inputs(bm, rows[sl])
            P = sigmoid(self.beta + Y[sl] @ Aw.T)
            wa = w[sl][:, None] * Av
            s0 += wa.sum(0); s1 += (wa * P).sum(0); s2 += (wa * P * P).sum(0)
        W = max(float(w.sum()), 1e-12)
        mean = s1 / np.maximum(s0, 1e-12)
        var = np.maximum(s2 / np.maximum(s0, 1e-12) - mean ** 2, 0.0)
        om = (var * (s0 / W)) ** self.cfg.metric_power           # (variance where available x availability share)^power
        fit = self._fitted()
        if om[fit].mean() <= 0:
            return np.ones(len(self.c))
        if self.cfg.metric == "worth_variance_dedup":
            om = om / self.redundancy(bm, rows, w)
        om = om / om[fit].mean()
        om = np.maximum(om, self.cfg.metric_floor)
        om = om / om[fit].mean()                                 # mean 1 over the fitted items
        om[~fit] = 1.0                                           # unrated items have no direction: any weight
        return om

    def redundancy(self, bm: BuildMatrix, rows, w: np.ndarray) -> np.ndarray:
        """[N] m_t = sum_j phi_tj^2 over items j whose picks correlate with t's at
        phi_tj >= dedup_tau (w-weighted phi coefficient of the pick indicators over
        the builds), or that the catalog links to t (prerequisite, exclusion);
        m_t >= 1 (j = t). A bundle of k items that are always taken together then
        weighs about one item instead of k. Picks, not field logits: in a low-rank
        field every item that loads on one direction has correlated logits, which
        would merge whole archetype blocks; picks only merge real duplicates."""
        X = bm.X[rows].astype(np.float64)
        W = max(float(w.sum()), 1e-12)
        Xw = X.multiply(w[:, None]).tocsr() if sp.issparse(X) else X * w[:, None]
        p = np.asarray(Xw.sum(0)).ravel() / W
        E = np.asarray((X.T @ Xw).todense() if sp.issparse(X) else X.T @ Xw) / W
        C = E - np.outer(p, p)
        sd = np.sqrt(np.maximum(p * (1 - p), 1e-12))
        r = C / sd[:, None] / sd[None, :]
        r[(p <= 0) | (p >= 1)] = 0.0
        r[:, (p <= 0) | (p >= 1)] = 0.0
        np.fill_diagonal(r, 1.0)
        link = r >= self.cfg.dedup_tau
        ix = self.c.index
        for k, it in self.c.items.items():
            a = ix[k]
            for q in list(it.prerequisites) + list(it.exclusive_with) + [x for g in it.prerequisites_any for x in g]:
                if q in ix:
                    link[a, ix[q]] = link[ix[q], a] = True
        m = (r * r * link).sum(1)
        m[~self._fitted()] = 1.0
        return np.maximum(m, 1.0)

    def set_metric(self, bm: BuildMatrix, metric: str = "worth_variance_dedup") -> None:
        """Switch the distance's item weights without refitting the field (item
        parameters and latents are unchanged); coordinates and regions are recomputed."""
        if metric not in ("worth_variance_dedup", "worth_variance", "uniform"):
            raise ValueError("metric is 'worth_variance_dedup', 'worth_variance' or 'uniform'")
        self.cfg.metric = metric
        self.omega = self.metric_weights(bm) if metric != "uniform" else None
        self._factor()
        self._fit_regions(bm, self.solve_latent(bm))

    def _fitted(self) -> np.ndarray:
        """[N] items the field was fitted with (False: added later by extend())."""
        f = getattr(self, "fitted", None)
        return np.ones(len(self.c), bool) if f is None or len(f) != len(self.c) else f

    def n_fit(self) -> int:
        """Number of fitted items: the N of the distance identity."""
        return int(self._fitted().sum())

    def extend(self, catalog: Catalog, prior: Dict[str, float]) -> "ConceptSpace":
        """The same field over a larger catalog, without refitting. Every item of
        the old catalog keeps its parameters (rows are moved to the new order); an
        item that is new gets a zero direction (G, O, V rows 0) and the intercept
        logit(prior[id]). Its rows of the weighted item matrix are zero, so R,
        coordinates, distances, regions and the worth bound are unchanged
        exactly; only its own worth is the constant prior until reviews or a refit
        give it evidence. Items missing from `catalog` are not allowed (retire them
        in the catalog instead)."""
        old = list(self.c.ids)
        miss = [i for i in old if i not in catalog.index]
        if miss:
            raise ValueError(f"extend() keeps every item; retire instead of removing: {miss[:5]}")
        pos = np.array([catalog.index[i] for i in old])
        N = len(catalog)
        fit = np.zeros(N, bool); fit[pos] = self._fitted()

        def move(a, fill=0.0):
            out = np.full((N,) + a.shape[1:], fill, dtype=np.float64)
            out[pos] = a
            return out
        new = [i for i in catalog.ids if i not in self.c.index]
        self.beta = move(self.beta)
        for i in new:
            p = float(np.clip(prior.get(i, 0.05), 1e-4, 1 - 1e-4))
            self.beta[catalog.index[i]] = np.log(p / (1 - p))
        self.G, self.O, self.V, self.Q = move(self.G), move(self.O), move(self.V), move(self.Q)
        if self.omega is not None:
            self.omega = move(self.omega, 1.0)
        self.c = catalog
        self.fitted = fit
        return self

    def weights(self) -> np.ndarray:
        """[N] the metric weights omega (ones for the uniform metric)."""
        return np.ones(len(self.c)) if self.omega is None else self.omega

    # ------------------------------------------------------------ latents
    # ------------------------------------------------------------ latent cache
    def params_key(self) -> str:
        """Fingerprint of everything a latent depends on: item parameters, the
        solver's settings, the token list."""
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for a in (self.beta, self.G, self.O, self.V, self.E, self.Hinv):
            a = np.ascontiguousarray(np.asarray(a, np.float64))
            h.update(str(a.shape).encode()); h.update(a.tobytes())
        f = self.cfg
        h.update(json.dumps([f.dim, list(f.stat_grid), f.l2_latent, f.infer_iters, f.infer_tol,
                             list(self.tokens)]).encode())
        return h.hexdigest()

    @staticmethod
    def data_key(bm: BuildMatrix) -> str:
        """Fingerprint of a build matrix's latent inputs (picks, context, stats, availability)."""
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for M in (bm.X, bm.T):
            M = M.tocsr()
            for a in (M.indptr, M.indices):
                h.update(memoryview(np.ascontiguousarray(a)).cast("B"))   # v3.7: hashed in place, no copy
            h.update(str(M.shape).encode())
        for a in (bm.hi, bm.gid, bm.masks):
            a = np.ascontiguousarray(a)
            h.update(str(a.shape).encode()); h.update(memoryview(a).cast("B"))
        h.update(json.dumps(list(bm.tokens)).encode())
        return h.hexdigest()

    def latent_key(self, bm: BuildMatrix) -> str:
        return self.params_key() + ":" + self.data_key(bm)

    def remember(self, bm: BuildMatrix, r: np.ndarray, key: Optional[str] = None) -> bool:
        """Store the exact latents of every row of bm (memo for solve_latent).
        key: the latent_key they were solved under; refused (False) if it does
        not match the current one (stale cache)."""
        k = self.latent_key(bm)
        if key is not None and key != k:
            return False
        if r is None or len(r) != len(bm):
            return False
        memo = getattr(self, "_memo", None)
        if not isinstance(memo, dict):
            memo = self._memo = {}
        memo.pop(k, None)
        memo[k] = np.asarray(r, np.float64)
        while len(memo) > 3:                           # the corpus, corpus + reviews, the fit sample
            memo.pop(next(iter(memo)))
        return True

    def cached_latents(self, bm: BuildMatrix) -> Optional[np.ndarray]:
        memo = getattr(self, "_memo", None)
        if not memo:
            return None
        return memo.get(self.latent_key(bm))

    def solve_latent(self, bm: BuildMatrix, rows=None, partial: bool = False) -> np.ndarray:
        """[b, dim] the unique r minimising each build's own loss (exact, deterministic).
        partial=True: only the picks are evidence (completion of an unfinished build).
        Whole-matrix solves are memoised (remember / cached_latents): the
        latents depend only on the item parameters and the rows, so refits'
        repeated coordinate passes, set_metric and catalog extensions reuse them."""
        if rows is not None:
            rr = np.asarray(rows)
            if rr.ndim == 1 and len(rr) == len(bm) and (len(rr) == 0 or (rr[0] == 0 and np.all(np.diff(rr) == 1))):
                rows = None                            # every row in order: the whole matrix
        whole = rows is None and not partial and len(bm) >= getattr(self.cfg, "device_min_rows", 256)
        if whole:
            got = self.cached_latents(bm)
            if got is not None:
                return got.copy()
        rows = np.arange(len(bm)) if rows is None else np.asarray(rows)
        out = np.zeros((len(rows), self.cfg.dim))
        dev = self._device() if len(rows) >= getattr(self.cfg, "device_min_rows", 256) else None
        if dev is not None:
            from tri.model import accel
        for a in range(0, len(rows), 8192):
            rr = rows[a:a + 8192]
            S, C, X, A = self._inputs(bm, rr)
            if dev is not None:
                out[a:a + len(rr)] = accel.latents(self, S, C, X, X if partial else A, dev)
            else:
                out[a:a + len(rr)] = self._mm(S, C, X, X if partial else A)
        if whole:
            self.remember(bm, out)
        return out

    def _mm(self, S, C, X, A, r0: Optional[np.ndarray] = None) -> np.ndarray:
        """Majorize-minimize for r: the Hessian is bounded by V'V/4 + lambda I, so
        r <- r - grad H^-1 decreases the convex loss monotonically to its unique minimum
        (from any start r0; the minimum does not depend on it)."""
        f = self.cfg
        off = self.beta + S @ self.G.T + C @ self.O.T + (C @ self.E) @ self.V.T
        r = np.zeros((len(S), f.dim)) if r0 is None else np.array(r0, dtype=np.float64)
        for _ in range(f.infer_iters):
            g = ((sigmoid(off + r @ self.V.T) - X) * A) @ self.V + f.l2_latent * r
            step = g @ self.Hinv
            r -= step
            if np.abs(step).max() < f.infer_tol:
                break
        return r

    def _y_rows(self, bm: BuildMatrix, rows, r: np.ndarray) -> np.ndarray:
        S = self.stat_steps(bm.hi[rows].astype(np.float64))
        C = self._tokens_of(bm, rows)
        return np.concatenate([C @ self.E + r, S, C], 1)

    def _deviance(self, bm: BuildMatrix, Rl: np.ndarray) -> Tuple[float, float]:
        tot = tot0 = W = 0.0
        pr = np.clip(bm.pick_rate()[0], 1e-6, 1 - 1e-6)
        for a in range(0, len(bm), 8192):
            rr = np.arange(a, min(a + 8192, len(bm)))
            S, C, X, A = self._inputs(bm, rr)
            p = np.clip(sigmoid(self.beta + S @ self.G.T + C @ self.O.T + (C @ self.E + Rl[rr]) @ self.V.T), 1e-9, 1 - 1e-9)
            ll = -(X * np.log(p) + (1 - X) * np.log(1 - p)) * A
            ll0 = -(X * np.log(pr) + (1 - X) * np.log(1 - pr)) * A
            tot += bm.w[rr] @ ll.sum(1); tot0 += bm.w[rr] @ ll0.sum(1); W += bm.w[rr] @ A.sum(1)
        return float(tot / max(W, 1e-12)), float(1 - tot / max(tot0, 1e-12))

    # ------------------------------------------------------------ coordinates
    def u_of(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.ymean) @ self.R.T / np.sqrt(self.n_fit())

    def coords(self, bm: BuildMatrix, rows=None, partial: bool = False, dtype=np.float64,
               chunk: int = 8192) -> np.ndarray:
        """[b, D'] concept coordinates of the builds. v3.7: row chunks (the field
        vector y and its centred copy were [b, D'] float64 each, 2.5 GB at 300k
        builds); dtype: the stored corpus coordinates are float32 (Tri.coord_dtype)."""
        rows = np.arange(len(bm)) if rows is None else np.asarray(rows)
        r = self.solve_latent(bm, rows, partial)
        out = np.empty((len(rows), self.R.shape[0]), dtype=dtype)
        for a in range(0, len(rows), chunk):
            out[a:a + chunk] = self.u_of(self._y_rows(bm, rows[a:a + chunk], r[a:a + chunk]))
        return out

    def logits_rows(self, bm: BuildMatrix, rows=None, partial: bool = False, prior: bool = False) -> np.ndarray:
        """prior: add the fixed stat-benchmark prior (tri/priors.py; math §2.9)."""
        rows = np.arange(len(bm)) if rows is None else np.asarray(rows)
        Y = self._y_rows(bm, rows, self.solve_latent(bm, rows, partial))
        L = self.beta + Y @ self._A().T
        if prior:
            from tri.game import priors
            L = L + priors.stat_offsets(self.c, bm.pr[rows], bm.fi[rows])
        return L

    def _A(self) -> np.ndarray:
        return np.concatenate([self.V, self.G, self.O], 1)

    def _bm(self, b: Build) -> BuildMatrix:
        return BuildMatrix.from_builds([b], self.c, fields=self.fields, tokens=self.tokens)

    def ys(self, builds: Sequence[Build]) -> np.ndarray:
        """[n, D] field vectors y of many builds in one solve (v3.1): the same
        rows point(b).y gives one at a time (the latent solve is row-wise)."""
        if not len(builds):
            return np.zeros((0, self.cfg.dim))
        bm = BuildMatrix.from_builds(list(builds), self.c, fields=self.fields, tokens=self.tokens)
        rows = np.arange(len(bm))
        return self._y_rows(bm, rows, self.solve_latent(bm, rows, False))

    def point(self, b: Build, partial: bool = False, region: Optional[int] = None,
              loo: bool = True, latent: Optional[np.ndarray] = None) -> FieldPoint:
        """The build in the field. region: replace its archetype by that region's
        latent (stats and context kept): "this build, played as archetype k".
        latent: replace it by this archetype latent [dim] instead (a forced archetype).
        loo: each item's logit is judged by the rest of the build (its own pick or
        non-pick removed from the latent's evidence; second-order exact)."""
        bm = self._bm(b)
        r = self.solve_latent(bm, None, partial)
        y = self._y_rows(bm, [0], r)[0]
        d = self.cfg.dim
        L = self.beta + self._A() @ y
        if latent is not None:
            y = y.copy(); y[:d] = np.asarray(latent, float)[:d]
            L = self.beta + self._A() @ y
        elif region is not None and self.region_z is not None and region < len(self.region_z):
            y = y.copy(); y[:d] = self.region_z[region]
            L = self.beta + self._A() @ y
        elif loo:
            L = self._loo(bm, L, partial)
        if getattr(self, "priors", True):              # fixed priors the pretrain cannot learn (math §2.9)
            from tri.game import priors, weapons as W
            pre, fin = priors.stat_rows(self.c, [b])
            L = priors.apply(L, self.c, pre, fin, W.equipped(b))
        return FieldPoint(y[:d].copy(), y, self.u_of(y[None])[0], L)

    def _loo(self, bm: BuildMatrix, L: np.ndarray, partial: bool) -> np.ndarray:
        """Leave-one-out logits. With H the build's exact latent Hessian and
        h_t = V_t H^-1 V_t, removing item t's evidence moves its logit by
        (p_t - x_t) h_t / (1 - w_t h_t), w_t = p_t (1 - p_t) (Sherman-Morrison)."""
        x = bm.X[0].toarray()[0].astype(np.float64)
        a = x if partial else bm.masks[bm.gid[0]].astype(np.float64)
        p = sigmoid(L); wt = p * (1 - p) * a
        H = (self.V * wt[:, None]).T @ self.V + self.cfg.l2_latent * np.eye(self.cfg.dim)
        h = np.einsum("td,de,te->t", self.V, np.linalg.inv(H), self.V)
        return L + a * (p - x) * h / np.maximum(1 - wt * h, 1e-6)

    def project(self, b: Build, partial: bool = False) -> np.ndarray:
        """Concept coordinates u of one build."""
        return self.point(b, partial).u

    # ------------------------------------------------------------ distance
    @staticmethod
    def distance_u(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def kernel(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similarity of two coordinate vectors: exp(-d^2 / 2h^2)."""
        d = self.distance_u(a, b)
        return float(np.exp(-d * d / (2 * self.cfg.bandwidth ** 2)))

    def similarity(self, a: Build, b: Build) -> float:
        """Archetype similarity in [0, 1]; 1 for the same build id."""
        if a.build_id and b.build_id and a.build_id == b.build_id:
            return 1.0
        return self.kernel(self.project(a), self.project(b))

    def distance(self, a: Build, b: Build) -> float:
        """omega-weighted RMS logit difference of the two builds (concept-space distance)."""
        return self.distance_u(self.project(a), self.project(b))

    def bounds(self, d: float) -> dict:
        """What a concept-space distance d guarantees about worth."""
        qn = self.item_bound_factor() * 4
        return dict(rms_worth_diff_max=d / 4, item_worth_diff_max=float(qn.max() * d / 4),
                    median_item_worth_diff_max=float(np.median(qn) * d / 4))

    def item_bound_factor(self) -> np.ndarray:
        """[N] c_t with |worth_t(b) - worth_t(b')| <= c_t ||u_b - u_b'||."""
        return np.linalg.norm(self.Q, axis=1) * np.sqrt(self.n_fit() / self.weights()) / 4

    # ------------------------------------------------------------ regions
    def _kmeans(self, U: np.ndarray, w: np.ndarray, k: int, seed: int, iters: int = 40) -> np.ndarray:
        """Weighted k-means++ centres [k', D'] (k' <= k)."""
        rng = np.random.default_rng(seed)
        cent = [U[rng.choice(len(U), p=w / w.sum())]]
        near = ((U - cent[0]) ** 2).sum(1)                             # running min: O(n R D), same draws
        for _ in range(1, k):                                          # k-means++
            d2 = near * w
            if d2.sum() <= 0:
                break
            cent.append(U[rng.choice(len(U), p=d2 / d2.sum())])
            near = np.minimum(near, ((U - cent[-1]) ** 2).sum(1))
        C = np.array(cent)
        for _ in range(iters):
            lab = self._nearest(U, C)
            newC = C.copy()
            for j in range(len(C)):
                m = lab == j
                if w[m].sum() > 0:
                    newC[j] = w[m] @ U[m] / w[m].sum()
            if np.allclose(newC, C):
                break
            C = newC
        return C

    def choose_k(self, U: np.ndarray, w: np.ndarray) -> int:
        """Largest k of region_grid (<= n_regions) whose regions are stable: two
        fits on disjoint random halves of a sample, compared on the whole sample
        by the adjusted Rand index; the median over two splits must reach
        region_ari. Falls back to the smallest candidate (or n_regions when no
        grid value fits)."""
        from sklearn.metrics import adjusted_rand_score
        f = self.cfg
        rng = np.random.default_rng(f.seed)
        idx = rng.choice(len(U), min(f.region_sample, len(U)), replace=False)
        Us, ws = U[idx], w[idx]
        grid = [k for k in sorted(set(f.region_grid)) if 2 <= k <= f.n_regions and k <= len(Us) // 20]
        self.region_stability = {}
        if not grid:
            return min(f.n_regions, len(U))
        best = grid[0]
        for k in grid:
            aris = []
            for rep in range(2):
                perm = np.random.default_rng([f.seed, k, rep]).permutation(len(Us))
                h1, h2 = perm[: len(Us) // 2], perm[len(Us) // 2:]
                C1 = self._kmeans(Us[h1], ws[h1], k, f.seed + 1 + rep)
                C2 = self._kmeans(Us[h2], ws[h2], k, f.seed + 11 + rep)
                aris.append(adjusted_rand_score(self._nearest(Us, C1), self._nearest(Us, C2)))
            a = float(np.median(aris))
            self.region_stability[str(k)] = round(a, 4)
            if a >= f.region_ari:
                best = k
        return best

    def _fit_regions(self, bm: BuildMatrix, Rl: np.ndarray, iters: int = 40) -> None:
        """Regions: weighted k-means on the concept coordinates, k chosen for
        stability (choose_k) unless region_select is "fixed"; then regions
        closer than the bandwidth with overlapping signatures are merged, and
        regions with no characteristic item are marked mixed."""
        f = self.cfg
        rows = np.arange(len(bm))
        Y = self._y_rows(bm, rows, Rl)
        U = self.u_of(Y)
        w = bm.w
        if min(f.n_regions, len(U)) == 0:
            self.centroids = np.zeros((0, U.shape[1])); self.region_z = np.zeros((0, f.dim))
            self.region_mixed = []; return
        k = self.choose_k(U, w) if f.region_select == "stable" else min(f.n_regions, len(U))
        C = self._kmeans(U, w, min(k, len(U)), f.seed, iters)
        lab = self._nearest(U, C)
        if f.region_merge < 1.0:
            C, lab = self._merge_regions(bm, U, w, C, lab)
        self.centroids = C
        d = f.dim
        self.region_z = np.array([(w[lab == j] @ Y[lab == j, :d]) / max(w[lab == j].sum(), 1e-12)
                                  for j in range(len(C))])
        self.region_size = [float(w[lab == j].sum() / w.sum()) for j in range(len(C))]
        self.region_auto = self._name_regions(bm, lab)
        self.region_names = list(self.region_auto)

    def _signatures(self, bm: BuildMatrix, lab: np.ndarray, R: int, k: int = 3):
        """Per region: (top characteristic items, best lift, dominant context token or None)."""
        w = bm.w
        base_x = np.asarray(bm.X.T @ w).ravel() / max(w.sum(), 1e-12)
        out = []
        for r in range(R):
            wr = w * (lab == r)
            if wr.sum() <= 0:
                out.append(([], 0.0, None)); continue
            px = np.asarray(bm.X.T @ wr).ravel() / wr.sum()
            score = np.where(px >= 0.3, px * np.log((px + 1e-9) / (base_x + 1e-9)), -np.inf)
            top = [bm.catalog.ids[j] for j in np.argsort(-score)[:k] if np.isfinite(score[j]) and score[j] > 0]
            lift = np.where(px >= 0.3, (px + 1e-9) / (base_x + 1e-9), 0.0)
            tok = None
            if bm.T.shape[1]:
                pt = np.asarray(bm.T.T @ wr).ravel() / wr.sum()
                j = int(np.argmax(pt))
                if pt[j] >= 0.5:
                    tok = bm.tokens[j].split(":", 1)[1]
            out.append((top, float(lift.max()) if len(lift) else 0.0, tok))
        return out

    def _merge_regions(self, bm: BuildMatrix, U, w, C, lab):
        """Merge pairs of regions whose centres are closer than the bandwidth and
        whose signatures (top items and dominant context) overlap with Jaccard
        >= region_merge, closest pair first, until none is left."""
        f = self.cfg
        while len(C) > 1:
            sig = self._signatures(bm, lab, len(C))
            sets = [set(t) | ({"ctx:" + c} if c else set()) for t, _, c in sig]
            D = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
            best = None
            for a in range(len(C)):
                for b in range(a + 1, len(C)):
                    if D[a, b] >= f.bandwidth:
                        continue
                    u = sets[a] | sets[b]
                    if u and len(sets[a] & sets[b]) / len(u) >= f.region_merge and (best is None or D[a, b] < best[0]):
                        best = (D[a, b], a, b)
            if best is None:
                break
            _, a, b = best
            lab = np.where(lab == b, a, lab)
            lab = np.where(lab > b, lab - 1, lab)
            C = np.array([w[lab == j] @ U[lab == j] / max(w[lab == j].sum(), 1e-12) for j in range(len(C) - 1)])
        return C, lab

    @staticmethod
    def _nearest(U: np.ndarray, C: np.ndarray) -> np.ndarray:
        out = np.zeros(len(U), np.int64)
        for a in range(0, len(U), 20000):
            u = U[a:a + 20000]
            out[a:a + len(u)] = np.argmin((u * u).sum(1)[:, None] - 2 * u @ C.T + (C * C).sum(1)[None], 1)
        return out

    def _name_regions(self, bm: BuildMatrix, lab: np.ndarray, k: int = 3) -> List[str]:
        """Names from each region's dominant context and characteristic items; a
        region with no item of lift >= mixed_lift is named "mixed <r>" and
        marked in region_mixed."""
        names, mixed = [], []
        for r, (top, lift, tok) in enumerate(self._signatures(bm, lab, len(self.centroids), k)):
            mx = lift < self.cfg.mixed_lift
            mixed.append(bool(mx))
            if mx:
                names.append(f"mixed {r}" + (f" ({tok})" if tok else "")); continue
            names.append(" / ".join(([tok] if tok else []) + top) or f"region {r}")
        self.region_mixed = mixed
        return names

    def memberships(self, u: np.ndarray) -> np.ndarray:
        d2 = ((self.centroids - u[None]) ** 2).sum(1)
        s = -d2 / (2 * self.cfg.bandwidth ** 2)
        p = np.exp(s - s.max())
        return p / p.sum()

    def reading(self, u: np.ndarray, top: int = 3) -> ConceptReading:
        """Region memberships: softmax of -d^2 / 2h^2 to each region centre."""
        if self.centroids is None or not len(self.centroids):
            return ConceptReading(u, [], 0.0)
        p = self.memberships(u)
        order = np.argsort(-p)[:top]
        regs = [(int(k), self.region_names[k], float(p[k])) for k in order]
        amb = float(p[order[1]] / p[order[0]]) if len(order) > 1 else 0.0
        return ConceptReading(u, regs, amb)

    def region_of(self, U: np.ndarray) -> np.ndarray:
        """Hard region (nearest centre) per row of concept coordinates."""
        if self.centroids is None or not len(self.centroids):
            return np.zeros(len(U), np.int64)
        return self._nearest(np.asarray(U), self.centroids)

    # ------------------------------------------------------------ inspection
    def item_coords(self) -> np.ndarray:
        """[N, D'] item directions: logit_t = const + item_coords[t] . u."""
        return self.Q * np.sqrt(self.n_fit() / self.weights())[:, None]

    def axes(self, k: int = 6, U: Optional[np.ndarray] = None) -> List[dict]:
        """Principal directions of the region centres (or of coordinates U),
        with the items whose logits rise / fall most along each."""
        M = self.centroids if U is None else U
        M = M - M.mean(0)
        _, s, vt = np.linalg.svd(M, full_matrices=False)
        IC = self.item_coords()
        tot = (s ** 2).sum()
        out = []
        for j in range(min(len(s), 6)):
            v = IC @ vt[j]
            o = np.argsort(v)
            out.append(dict(axis=j, share=float(s[j] ** 2 / max(tot, 1e-12)),
                            positive=[self.c.ids[i] for i in o[::-1][:k]], negative=[self.c.ids[i] for i in o[:k]]))
        return out

    def report(self) -> dict:
        om = self.weights()
        top = np.argsort(-om)[:10]
        return dict(dim=int(self.cfg.dim), coordinates=int(self.R.shape[0]), bandwidth=self.cfg.bandwidth,
                    metric=self.cfg.metric if self.omega is not None else "uniform",
                    heaviest_items=[dict(item=self.c.ids[i], weight=round(float(om[i]), 3)) for i in top],
                    train_loss=round(self.train_loss, 4), deviance_explained=round(self.explained, 4),
                    validation={k: v for k, v in self.validation.items() if k != "curve"},
                    region_stability=dict(self.region_stability),
                    regions=[dict(id=k, name=n, share=round(s, 4),
                                  mixed=bool(self.region_mixed[k]) if k < len(self.region_mixed) else False)
                             for k, (n, s) in enumerate(zip(self.region_names, self.region_size))])

    # ------------------------------------------------------------ persistence
    _ARR = ("beta", "G", "O", "V", "E", "ymean", "R", "Q", "Hinv", "centroids", "region_z")

    def save(self, path: str) -> None:
        arr = {k: getattr(self, k) for k in self._ARR}
        if self.omega is not None:
            arr["omega"] = self.omega
        arr["fitted"] = self._fitted()
        np.savez_compressed(path + ".npz", **arr)
        cfg = asdict(self.cfg); cfg["stat_grid"] = list(cfg["stat_grid"])
        with open(path + ".json", "w") as f:
            json.dump(dict(cfg=cfg, tokens=self.tokens, region_names=self.region_names, region_size=self.region_size,
                           region_auto=self.region_auto, region_mixed=self.region_mixed,
                           region_stability=self.region_stability, validation=self.validation, train_loss=self.train_loss, explained=self.explained, fields=list(self.fields),
                           ids=list(self.c.ids)), f)

    @staticmethod
    def load(path: str, catalog: Catalog) -> "ConceptSpace":
        with open(path + ".json") as f:
            meta = json.load(f)
        if meta["ids"] != list(catalog.ids):
            raise ValueError("concept space was fitted on a different catalog; use Tri.migrate")
        cfg = dict(meta["cfg"]); cfg["stat_grid"] = tuple(cfg["stat_grid"])
        cfg = {k: v for k, v in cfg.items() if k in SpaceConfig.__dataclass_fields__}
        s = ConceptSpace(catalog, SpaceConfig(**cfg))
        z = np.load(path + ".npz")
        for k in ConceptSpace._ARR:
            setattr(s, k, z[k])
        s.omega = z["omega"] if "omega" in z.files else None    # runs before v2.3: uniform metric
        s.fitted = z["fitted"].astype(bool) if "fitted" in z.files else np.ones(len(catalog), bool)
        if s.omega is None:
            s.cfg.metric = "uniform"
        s.tokens, s.region_names, s.region_size = meta["tokens"], meta["region_names"], meta["region_size"]
        s.train_loss, s.explained = meta["train_loss"], meta["explained"]
        s.region_auto = meta.get("region_auto", list(s.region_names))
        s.region_mixed = list(meta.get("region_mixed", [False] * len(s.region_names)))
        s.region_stability = dict(meta.get("region_stability", {}))
        s.validation = meta.get("validation", {})
        s.fields = tuple(meta.get("fields", s.fields))
        return s
