"""Locators: administrator-labelled builds that name archetypes in the concept space.

A locator is a build (from a builder link) with a semantic archetype assigned by an
administrator ("ganking", "PvE tank", ...). Locators never enter the likelihood
of the field: their labels fit no parameter, and their builds are removed from
the corpus the field is trained on. They only

  read     the archetype of any build, from a kernel density of each archetype's
           locators (equal prior per archetype, so an archetype with more locators
           does not win by count):
               k_pi(u) = mean over locators j of archetype pi of exp(-||u - u_j||^2 / 2h^2)
               p(pi | u) = k_pi(u) / sum_rho k_rho(u)                 (h = bandwidth)
           A build is located when its nearest locator is within `reach` and
           its top archetype has p >= min_confidence; otherwise it is unlocated.
  name     regions: a region whose centre is located is named after the archetype.
  validate the geometry out of sample: leave-one-out reading of each locator
           from the others (accuracy, log-likelihood, same-vs-different AUC,
           conflicts), honest because no locator trained the field.
  select   the metric power gamma from a small grid by leave-one-out
           log-likelihood, moving from the default only when the paired gain
           exceeds select_z (2) standard errors and select_min_gain (0.01 nats)
           (math §6.5).
  separate review influence: the ledger kernel and the worth boost of reviewed
           builds are multiplied by
               1 - barrier * a(u) a(v) * TV(p(.|u), p(.|v))
           where TV is the total-variation distance of the two archetype readings
           (0 when they agree, 1 when they are confidently different) and a(u)
           is 1 within reach of a locator and decays beyond it, so a review does
           not carry across two archetypes the administrators told apart.
           Continuous; nothing is partitioned."""
from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.fast import sqdist

from tri.core.types import Build
from tri.reviews.defs import ArchetypeReading, Locator, LocatorConfig


def archetype_key(name: str) -> str:
    """Comparison key of an archetype name: case-folded, spaces collapsed."""
    return " ".join(str(name).split()).casefold()


class LocatorSet:
    """Locators with their coordinates in one concept space."""

    def __init__(self, locators: Sequence[Locator], space, cfg: Optional[LocatorConfig] = None,
                 reach: float = 1.2, vec=None, background: Optional[np.ndarray] = None):
        self.cfg = cfg or LocatorConfig()
        self.space = space
        self.locators: List[Locator] = list(locators)
        self.h = float(self.cfg.bandwidth if self.cfg.bandwidth is not None else space.cfg.bandwidth)
        self.reach = float(self.cfg.reach if self.cfg.reach is not None else reach)
        names: Dict[str, str] = {}
        for l in self.locators:                         # display name: the newest spelling of each family
            names[l.key] = l.family
        self.names = names
        self.keys: List[str] = sorted(names)
        self.kix = {k: i for i, k in enumerate(self.keys)}
        self.lab = np.array([self.kix[l.key] for l in self.locators], dtype=int)
        self.Y = space.ys([l.build for l in self.locators]) if self.locators else None
        self.U = space.u_of(self.Y) if self.locators else np.zeros((0, 0))
        # fine reading: each locator's picks x and availability a (vec(build) -> (x, a))
        self.vec = vec
        self._fine: Dict[bytes, np.ndarray] = {}
        self.Xl = self.Al = None
        if vec is not None and self.locators:
            xa = [vec(l.build) for l in self.locators]
            self.Xl = np.stack([np.asarray(x, float) for x, _ in xa])
            self.Al = np.stack([np.asarray(a, float) for _, a in xa])
            self.Xl = self.Xl * self.Al                  # a pick that is not available is no evidence
        self.indicators: List[dict] = []
        self.tells: List[dict] = []              # v2.19 reviewer consensus (math §8.5a)
        self.anchors: List[dict] = []            # v2.18 review anchors: {key, u, sign, review_id}
        self._UA = np.zeros((0, 0)); self._labA = np.zeros(0, int); self._sgnA = np.zeros(0)
        self.set_background(background)

    def __len__(self) -> int:
        return len(self.locators)

    def refresh(self) -> None:
        """Recompute coordinates after the metric changed (latents are unchanged)."""
        if self.locators:
            self.U = self.space.u_of(self.Y)
            if self.K is not None:
                self._fine.clear(); self.pairs = []; self._find_pairs()

    # ------------------------------------------------------------ twin split (math §6.6)
    def set_background(self, rate: Optional[np.ndarray]) -> None:
        """Background pick rate b_t (corpus rate given availability; default: the
        locators pooled, one pick in two pseudo-builds). Rebuilds the twin pairs."""
        self._fine.clear()
        self.pairs: List[dict] = []
        self.K = self.Nn = None
        if self.Xl is None:
            return
        if rate is None:
            rate = (self.Xl.sum(0) + 1.0) / (self.Al.sum(0) + 2.0)
        self.base = np.clip(np.asarray(rate, float), 0.02, 0.98)
        K = np.zeros((len(self.keys), self.Xl.shape[1])); Nn = np.zeros_like(K)
        np.add.at(K, self.lab, self.Xl); np.add.at(Nn, self.lab, self.Al)
        self.K, self.Nn = K, Nn
        self._find_pairs()

    def _loo_kernel(self) -> np.ndarray:
        """[m, A] kernel reading of each locator from the others."""
        D2 = np.maximum(((self.U[:, None] - self.U[None]) ** 2).sum(-1), 0.0)
        np.fill_diagonal(D2, np.inf)
        return self._posterior(self._archetype_logk(D2, self.lab))

    def _contrast(self, a: int, b: int, Ka, Na, Kb, Nb):
        """Shrunken rates of twins a, b toward their pooled rate and the weights of the
        tells: (Ra, Rb, T, w1, w0) with LLR = sum_T x w1 + (1 - x) w0 (positive: a)."""
        kap = self.cfg.refine_prior
        c = np.clip((Ka + Kb + 2.0 * self.base) / (Na + Nb + 2.0), 1e-3, 1 - 1e-3)
        Ra = np.clip((Ka + kap * c) / (Na + kap), 1e-4, 1 - 1e-4)
        Rb = np.clip((Kb + kap * c) / (Nb + kap), 1e-4, 1 - 1e-4)
        lo = np.log(Ra / (1 - Ra)) - np.log(Rb / (1 - Rb))
        T = (np.abs(lo) >= self.cfg.refine_min_logodds) & (Na > 0) & (Nb > 0)
        return Ra, Rb, T, np.where(T, np.log(Ra / Rb), 0.0), np.where(T, np.log((1 - Ra) / (1 - Rb)), 0.0)

    def _split(self, p: np.ndarray, a: int, b: int, llr: np.ndarray, near: Optional[np.ndarray]) -> np.ndarray:
        """Re-split the mass of the pair (a, b) by the clipped log-likelihood ratio, where
        the pair holds at least twin_mass of the reading (and the build is within reach)."""
        m = p[:, a] + p[:, b]
        on = m >= self.cfg.twin_mass
        if near is not None:
            on &= near
        if not on.any():
            return p
        s = np.clip(p[:, a] / np.maximum(m, 1e-300), 1e-9, 1 - 1e-9)
        z = np.log(s / (1 - s)) + np.clip(llr, -self.cfg.twin_clip, self.cfg.twin_clip)
        s2 = 1.0 / (1.0 + np.exp(-z))
        p = p.copy()
        p[on, a] = (m * s2)[on]
        p[on, b] = (m * (1 - s2))[on]
        return p

    def _find_pairs(self) -> None:
        """Twin pairs: archetypes the kernel confuses (leave-one-out overlap >= twin_overlap),
        each with >= refine_min_locators locators. A pair is active only if splitting it
        by its tells corrects at least twin_min_gain more held-out locators than it breaks."""
        A = len(self.keys)
        cnt = np.bincount(self.lab, minlength=A)
        if A < 2 or len(self.locators) < 3:
            return
        P = self._loo_kernel()
        for a in range(A):
            for b in range(a + 1, A):
                if min(cnt[a], cnt[b]) < self.cfg.refine_min_locators:
                    continue
                ia, ib = np.flatnonzero(self.lab == a), np.flatnonzero(self.lab == b)
                ov = max(float(P[ia, b].mean()), float(P[ib, a].mean()))
                if ov < self.cfg.twin_overlap:
                    continue
                Ra, Rb, T, w1, w0 = self._contrast(a, b, self.K[a], self.Nn[a], self.K[b], self.Nn[b])
                gain = loss = 0
                pred = {}
                for j in np.r_[ia, ib]:
                    Ka, Na, Kb, Nb = self.K[a].copy(), self.Nn[a].copy(), self.K[b].copy(), self.Nn[b].copy()
                    if self.lab[j] == a:
                        Ka -= self.Xl[j]; Na -= self.Al[j]
                    else:
                        Kb -= self.Xl[j]; Nb -= self.Al[j]
                    _, _, Tj, v1, v0 = self._contrast(a, b, Ka, Na, Kb, Nb)
                    x, av = self.Xl[j], self.Al[j]
                    llr = np.array([x @ v1 + (av - x) @ v0])
                    pf = self._split(P[j][None], a, b, llr, None)[0]
                    pred[int(j)] = pf
                    rk, rf = int(P[j].argmax()) == self.lab[j], int(pf.argmax()) == self.lab[j]
                    gain += int(rf and not rk); loss += int(rk and not rf)
                self.pairs.append(dict(a=a, b=b, overlap=round(ov, 3), tells=int(T.sum()), gain=gain, loss=loss,
                                       active=bool(T.any() and gain - loss >= self.cfg.twin_min_gain),
                                       Ra=Ra, Rb=Rb, T=T, w1=w1, w0=w0, loo=pred))

    @property
    def active_pairs(self) -> List[dict]:
        return [q for q in self.pairs if q["active"]] if self.cfg.refine else []

    @property
    def refining(self) -> bool:
        return bool(self.active_pairs) or bool(self.indicators) or bool(getattr(self, "tells", None))

    # ------------------------------------------------------------ reviewer consensus (math §8.5a)
    def set_consensus(self, tells: Sequence[dict]) -> None:
        """Promoted family tells: dicts key, t (column index), item, sign, confidence.
        Take tells (sign +1) lift the family's odds on builds holding them; every
        tell moves worth on builds read as the family (FieldWorth is not touched)."""
        self.tells = [dict(t) for t in tells if t.get("key") in self.kix and t.get("t") is not None]
        self._fine.clear()

    def consensus_of(self, key: str) -> List[dict]:
        return [t for t in getattr(self, "tells", None) or [] if t["key"] == key]

    # ------------------------------------------------------------ indicator notes (math §6.7)
    def set_indicators(self, indicators: Sequence[dict]) -> None:
        """Reviewers' notes "components C indicate archetype a": dicts with key (archetype
        key), items (column indices), tokens (oath:/murmur: context), u (where the
        reviewed build sits), review_id. Unknown archetypes are ignored."""
        self.indicators = [dict(i) for i in indicators if i.get("key") in self.kix
                           and (i.get("items") or i.get("tokens"))]
        self._fine.clear()

    def set_anchors(self, anchors: Sequence[dict]) -> None:
        """Reviewed builds whose reviewer said "this build is a" (sign +1) or "is not
        a" (sign -1): dicts with key, u, sign, review_id. A positive anchor adds
        anchor_weight / n_a kernel mass to archetype a around the noted build, a
        negative one scales k_a by 1 - anchor_veto * K there (math §6.7). Far from
        the noted build K -> 0 and the reading is unchanged."""
        rows = [dict(a) for a in anchors if a.get("key") in self.kix and a.get("u") is not None]
        self.anchors = rows
        if rows:
            self._UA = np.vstack([np.asarray(a["u"], float) for a in rows])
            self._labA = np.array([self.kix[a["key"]] for a in rows], int)
            self._sgnA = np.array([1.0 if float(a.get("sign", 1)) >= 0 else -1.0 for a in rows])
        else:
            self._UA = np.zeros((0, self.U.shape[1] if self.U.size else 0))
            self._labA, self._sgnA = np.zeros(0, int), np.zeros(0)
        self._fine.clear()

    def _anchor_logk(self, U: np.ndarray, lg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(log k with review anchors, distance to the nearest positive anchor)."""
        q = len(U)
        if not len(self._labA):
            return lg, np.full(q, np.inf)
        DA = sqdist(U, self._UA)
        K = np.exp(-DA / (2 * self.h ** 2))
        out = lg.copy()
        n = np.bincount(self.lab, minlength=len(self.keys)).astype(float)
        for p in np.unique(self._labA):
            if n[p] <= 0:
                continue
            pos = (self._labA == p) & (self._sgnA > 0)
            neg = (self._labA == p) & (self._sgnA < 0)
            if pos.any():
                add = (float(self.cfg.anchor_weight) / n[p]) * K[:, pos].sum(1)
                with np.errstate(divide="ignore"):
                    out[:, p] = np.logaddexp(out[:, p], np.log(np.maximum(add, 1e-300)))
            if neg.any():
                veto = np.prod(1.0 - float(self.cfg.anchor_veto) * K[:, neg], axis=1)
                with np.errstate(divide="ignore"):
                    out[:, p] = out[:, p] + np.log(np.maximum(veto, 1e-12))
        posm = self._sgnA > 0
        dA = np.sqrt(DA[:, posm].min(1)) if posm.any() else np.full(q, np.inf)
        return out, dA

    def _indicate(self, P: np.ndarray, X, U: Optional[np.ndarray], dmin: Optional[np.ndarray],
                  ctx: Optional[Sequence[set]] = None) -> np.ndarray:
        """Lift the odds of a by e^lift where a build holds every component of an
        indicator for a, sits within reach of the noted build and of a locator, and
        already gives a at least indicator_floor. One lift per archetype (no stacking)."""
        if (not self.indicators and not getattr(self, "tells", None)) or U is None:
            return P
        U = np.atleast_2d(np.asarray(U, float))
        n = len(P)
        near = np.ones(n, bool) if dmin is None else np.atleast_1d(np.asarray(dmin, float)) <= self.reach
        lift = np.zeros_like(P)
        L = float(self.cfg.indicator_lift)
        for ind in self.indicators:
            k = self.kix[ind["key"]]
            idx = list(ind.get("items") or [])
            if idx:
                sub = X[:, idx]
                sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
                have = (np.atleast_2d(sub) > 0).all(1)
            else:
                have = np.ones(n, bool)
            toks = list(ind.get("tokens") or [])
            if toks:
                have = have & np.array([ctx is not None and all(t in ctx[i] for t in toks) for i in range(n)])
            close = np.sqrt(((U - np.asarray(ind["u"], float)[None]) ** 2).sum(1)) <= self.reach
            ok = have & close & near & (P[:, k] >= self.cfg.indicator_floor)
            lift[ok, k] = L
        lift = self._consensus_lift(P, X, near, lift)
        if not lift.any():
            return P
        Q = P * np.exp(lift)
        return Q / Q.sum(1, keepdims=True)

    def _consensus_lift(self, P: np.ndarray, X, near: np.ndarray, lift: np.ndarray) -> np.ndarray:
        """Family tells on the reading: lambda_c * sum of the confidences of the held take
        tells of a, where the build is within reach of a locator and p(a | u) >= floor
        already; the total lift of a is capped at indicator_lift (no stacking past a note)."""
        tells = [t for t in getattr(self, "tells", None) or [] if t["sign"] > 0]
        if not tells:
            return lift
        L, lam, fl = float(self.cfg.indicator_lift), float(self.cfg.consensus_lift), float(self.cfg.consensus_floor)
        add = np.zeros_like(P)
        for t in tells:
            k = self.kix[t["key"]]
            col = X[:, t["t"]]
            col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
            ok = (col > 0) & near & (P[:, k] >= fl)
            add[ok, k] += lam * float(t["confidence"])
        return np.minimum(np.maximum(lift, add), L)

    def indicated(self, u: np.ndarray, b=None) -> List[dict]:
        """The indicator notes (and promoted family tells) that lift the reading of
        the build b at u."""
        if (not self.indicators and not self.tells) or b is None:
            return []
        x, _a = self.vec(b)
        from tri.core.context import build_context
        ctx = [set(build_context(b)[0])]
        P, dmin, _ = self.read_many(np.asarray(u, float)[None])
        out = []
        save_i, save_t = self.indicators, self.tells
        try:
            self.tells = []
            for ind in save_i:
                self.indicators = [ind]
                Q = self._indicate(P, np.asarray(x, float)[None], np.asarray(u, float)[None], dmin, ctx)
                if np.abs(Q - P).sum() > 1e-9:
                    out.append({"archetype": self.names[ind["key"]], "components": list(ind.get("names") or []),
                                "review_id": ind.get("review_id")})
            self.indicators = []
            for t in save_t:
                if t["sign"] <= 0:
                    continue
                self.tells = [t]
                Q = self._indicate(P, np.asarray(x, float)[None], np.asarray(u, float)[None], dmin, ctx)
                if np.abs(Q - P).sum() > 1e-9:
                    out.append({"archetype": self.names[t["key"]], "components": [t["item"]],
                                "consensus": round(float(t["confidence"]), 3), "reviewers": t.get("reviewers")})
        finally:
            self.indicators, self.tells = save_i, save_t
        return out

    def refine(self, P: np.ndarray, X, A, dmin: Optional[np.ndarray] = None, U: Optional[np.ndarray] = None,
               ctx: Optional[Sequence[set]] = None) -> np.ndarray:
        """The twin split: for each active twin pair, where the pair holds most of the
        kernel reading (and the build is within reach when dmin is given), re-split the
        pair's mass by the tells; every other archetype, and every other build, is unchanged."""
        P = np.atleast_2d(np.asarray(P, float))
        if not self.refining:
            return P
        X = X if hasattr(X, "multiply") else np.atleast_2d(np.asarray(X, float))
        A = np.atleast_2d(np.asarray(A, float))
        XA = X.multiply(A) if hasattr(X, "multiply") else X * A
        near = None if dmin is None else np.atleast_1d(np.asarray(dmin, float)) <= self.reach
        for q in self.active_pairs:
            llr = np.asarray(XA @ q["w1"]).ravel() + A @ q["w0"] - np.asarray(XA @ q["w0"]).ravel()
            P = self._split(P, q["a"], q["b"], llr, near)
        return self._indicate(P, X, U, dmin, ctx)

    def onehot(self, key: str) -> np.ndarray:
        p = np.zeros(len(self.keys)); p[self.kix[key]] = 1.0
        return p

    def note(self, u: np.ndarray, b=None, xa=None, force: Optional[str] = None,
             transient: bool = False) -> np.ndarray:
        """Fine reading of the build b at u, remembered for the review barrier (which
        only sees coordinates). force: an archetype key the caller chose.
        transient (v3.6): a query's reading, forgotten at the next query
        (forget_query), so an answer never depends on earlier questions."""
        u = np.asarray(u, float)
        if transient:
            q = self.__dict__.setdefault("_qnotes", {})
            k = u.tobytes()
            if k not in q:
                q[k] = self._fine.get(k)
        P, dmin, _ = self.read_many(u[None])
        if force is not None and force in self.kix:
            p = self.onehot(force)
        elif self.refining and (b is not None or xa is not None):
            x, a = xa if xa is not None else self.vec(b)
            ctx = None
            if b is not None:
                from tri.core.context import build_context
                ctx = [set(build_context(b)[0])]
            p = self.refine(P, np.asarray(x, float)[None], np.asarray(a, float)[None], dmin, u[None], ctx)[0]
        else:
            p = P[0]
        if len(self._fine) > 100000:
            self._fine.clear()
        self._fine[u.tobytes()] = p
        return p

    def forget_query(self) -> None:
        """Undo the transient notes of the last query (v3.6)."""
        q = self.__dict__.pop("_qnotes", None)
        if not q:
            return
        for k, old in q.items():
            if old is None:
                self._fine.pop(k, None)
            else:
                self._fine[k] = old
        self.__dict__.pop("_vmemo", None)

    def fine_rows(self, U: np.ndarray) -> np.ndarray:
        """Posterior of each row of U: the remembered fine reading, else the kernel reading."""
        U = np.atleast_2d(np.asarray(U, float))
        P = self.read_many(U)[0]
        if self._fine:
            for i in range(len(U)):
                f = self._fine.get(U[i].tobytes())
                if f is not None:
                    P[i] = f
        return P

    def contrast(self, Pc: np.ndarray, Pf: np.ndarray) -> np.ndarray:
        """[N] log-odds correction of worth when the split (or a forced archetype) moves
        mass inside a twin pair: on that pair's tells only, m [logit(s' Ra + (1 - s') Rb)
        - logit(s Ra + (1 - s) Rb)], with m the pair's mass; 0 elsewhere."""
        N = len(self.base) if self.K is not None else 0
        d = np.zeros(N)
        if not self.cfg.refine or not self.pairs:
            return d
        Pc, Pf = np.asarray(Pc, float), np.asarray(Pf, float)
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        for q in self.active_pairs:
            a, b = q["a"], q["b"]
            mc, mf = Pc[a] + Pc[b], Pf[a] + Pf[b]
            if max(mc, mf) < 1e-6:
                continue
            sc, sf = Pc[a] / max(mc, 1e-12), Pf[a] / max(mf, 1e-12)
            if abs(sf - sc) < 1e-6:
                continue
            d += np.where(q["T"], max(mc, mf) * (lg(sf * q["Ra"] + (1 - sf) * q["Rb"])
                                                - lg(sc * q["Ra"] + (1 - sc) * q["Rb"])), 0.0)
        return d

    def adjust(self, q: np.ndarray, u: np.ndarray, Pf: Optional[np.ndarray] = None) -> np.ndarray:
        """Worth q corrected to the fine (or forced) reading Pf at u."""
        if not self.refining:
            return q
        u = np.asarray(u, float)
        Pf = self._fine.get(u.tobytes()) if Pf is None else Pf
        if Pf is None:
            return q
        d = self.contrast(self.read_many(u[None])[0][0], Pf)
        if not np.any(d):
            return q
        qc = np.clip(q, 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-(np.log(qc / (1 - qc)) + d)))

    def pair_of(self, key: str) -> List[dict]:
        """Active twin pairs involving archetype key."""
        if key not in self.kix:
            return []
        i = self.kix[key]
        return [q for q in self.active_pairs if i in (q["a"], q["b"])]

    def distinct(self, key: str, other: str, k: int = 5) -> List[Tuple[int, float]]:
        """[(item index, log odds ratio)] tells of the active twin pair (key, other),
        strongest first (positive: key takes it more); [] if they are not active twins."""
        if key not in self.kix or other not in self.kix:
            return []
        a, b = self.kix[key], self.kix[other]
        for q in self.active_pairs:
            if {q["a"], q["b"]} == {a, b}:
                Ra, Rb = (q["Ra"], q["Rb"]) if q["a"] == a else (q["Rb"], q["Ra"])
                lo = np.where(q["T"], np.log(Ra / (1 - Ra)) - np.log(Rb / (1 - Rb)), 0.0)
                o = np.argsort(-np.abs(lo))[:k]
                return [(int(j), float(lo[j])) for j in o if abs(lo[j]) >= self.cfg.refine_min_logodds]
        return []

    # ------------------------------------------------------------ reading
    def _archetype_logk(self, D2: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """[q, P] log k_pi: log of the mean kernel over each archetype's locators. Infinite
        entries of D2 (a locator left out of its own reading) are not counted."""
        out = np.full((D2.shape[0], len(self.keys)), -np.inf)
        L = -D2 / (2 * self.h ** 2)
        for p in range(len(self.keys)):
            m = lab == p
            if not m.any():
                continue
            Lp = L[:, m]
            cnt = np.isfinite(Lp).sum(1)
            mx = np.max(np.where(np.isfinite(Lp), Lp, -np.inf), 1)
            ok = cnt > 0
            sm = np.exp(Lp[ok] - mx[ok, None]).sum(1)
            out[ok, p] = mx[ok] + np.log(sm) - np.log(cnt[ok])
        return out

    @staticmethod
    def _posterior(lg: np.ndarray) -> np.ndarray:
        """Normalise log k over archetypes, numerically stable; archetypes with no locator get 0."""
        mx = np.max(lg, 1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        e = np.where(np.isfinite(lg), np.exp(lg - mx), 0.0)
        s = e.sum(1, keepdims=True)
        return np.where(s > 0, e / np.maximum(s, 1e-300), 0.0)

    def activity(self, dmin: np.ndarray) -> np.ndarray:
        """a(u) = 1 within reach of a locator, exp(-(d - reach)^2 / 2h^2) beyond."""
        over = np.maximum(np.asarray(dmin, float) - self.reach, 0.0)
        return np.exp(-over ** 2 / (2 * self.h ** 2))

    def read_many(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(posterior [q, P], nearest distance [q], nearest locator index [q])."""
        U = np.atleast_2d(U)
        D2 = sqdist(U, self.U)                 # v3.1: BLAS, no [q, L, D] cube
        j = D2.argmin(1)
        lg, dA = self._anchor_logk(U, self._archetype_logk(D2, self.lab))
        d = np.minimum(np.sqrt(D2[np.arange(len(U)), j]), dA)    # a positive anchor locates like a locator
        return self._posterior(lg), d, j

    def read(self, u: np.ndarray, b=None, xa=None, force: Optional[str] = None) -> ArchetypeReading:
        """The archetype at u. With the build b (or its picks and availability xa), the
        fine reading; force: an archetype key the caller chose (always located)."""
        if not self.locators:
            return ArchetypeReading(None, 0.0, {}, None, float("inf"), False, 0.0)
        Pc, dmin, j = self.read_many(np.asarray(u)[None])
        pc = Pc[0]
        fine = b is not None or xa is not None or force is not None
        p = self.note(u, b, xa, force) if fine else pc
        k = int(p.argmax())
        forced = force is not None and force in self.kix
        located = forced or bool(dmin[0] <= self.reach and p[k] >= self.cfg.min_confidence)
        best = self.names[self.keys[k]]
        nm = lambda v: {self.names[self.keys[i]]: float(v[i]) for i in range(len(self.keys)) if v[i] > 1e-4}
        return ArchetypeReading(best if located else None, float(p[k]), nm(p),
                              self.locators[int(j[0])].locator_id, float(dmin[0]), located,
                              float(self.activity(dmin)[0]), best,
                              coarse=nm(pc) if fine and np.abs(p - pc).sum() > 1e-3 else None, forced=forced)

    # ------------------------------------------------------------ review barrier
    def barrier_factor(self, u: np.ndarray, V: np.ndarray) -> np.ndarray:
        """[m] the chasm: multiplier for the influence of builds at V on a build at u
        (math §5.4),
            chi = 1 - barrier a(u) a(v) (1 - (1 - TV(p(.|u), p(.|v)))^gamma),
        and chi = 1 - barrier when both are located in different families
        (chasm_hard). In [1 - barrier, 1]; 1 for builds of the same family."""
        V = np.atleast_2d(V)
        if not self.locators or self.cfg.barrier <= 0 or len(self.keys) < 2:
            return np.ones(len(V))
        u = np.asarray(u, float)
        Pq, dq, _ = self.read_many(u[None])
        Pv, dv = self._read_cached(V)
        if self._fine:
            f = self._fine.get(u.tobytes())
            Pq = Pq if f is None else f[None]
            for i in range(len(V)):
                f = self._fine.get(V[i].tobytes())
                if f is not None:
                    Pv[i] = f
        tv = np.clip(0.5 * np.abs(Pv - Pq[0][None]).sum(1), 0.0, 1.0)
        g = max(float(self.cfg.chasm_power), 1.0)
        b = float(self.cfg.barrier)
        chi = 1.0 - b * self.activity(dq)[0] * self.activity(dv) * (1.0 - (1.0 - tv) ** g)
        if self.cfg.chasm_hard:
            mc = float(self.cfg.min_confidence)
            ku = int(Pq[0].argmax())
            lu = dq[0] <= self.reach and Pq[0, ku] >= mc
            if lu:
                kv = Pv.argmax(1)
                lv = (dv <= self.reach) & (Pv[np.arange(len(V)), kv] >= mc)
                chi = np.where(lv & (kv != ku), 1.0 - b, chi)
        return chi

    def family_overlap(self, u: np.ndarray, V: np.ndarray) -> np.ndarray:
        """[m] v3.3 family reach (math §5.8): 1 - TV of the family posteriors of
        a build at u and builds at V when both are located in the same family
        (nearest locator within reach, top family probability >= min_confidence),
        else 0. Never crosses the chasm: different or unlocated families give 0."""
        V = np.atleast_2d(V)
        if not self.locators or not len(V):
            return np.zeros(len(V))
        from tri.reviews.variants import family_key
        fam = sorted({family_key(k) for k in self.keys})
        M = np.zeros((len(self.keys), len(fam)))
        for i, k in enumerate(self.keys):
            M[i, fam.index(family_key(k))] = 1.0
        Pq, dq, _ = self.read_many(np.asarray(u, float)[None])
        Pv, dv = self._read_cached(V)
        Fq, Fv = Pq @ M, Pv @ M
        mc = float(self.cfg.min_confidence)
        ku = int(Fq[0].argmax())
        if not (dq[0] <= self.reach and Fq[0, ku] >= mc):
            return np.zeros(len(V))
        kv = Fv.argmax(1)
        ok = (dv <= self.reach) & (Fv[np.arange(len(V)), kv] >= mc) & (kv == ku)
        return np.where(ok, 1.0 - np.clip(0.5 * np.abs(Fv - Fq[0][None]).sum(1), 0.0, 1.0), 0.0)

    def located_many(self, U: np.ndarray) -> np.ndarray:
        """[q] builds the locators place in one family (nearest locator within
        reach, top family probability >= min_confidence)."""
        U = np.atleast_2d(U)
        if not self.locators or not len(U):
            return np.zeros(len(U), bool)
        from tri.reviews.variants import family_key
        fam = sorted({family_key(k) for k in self.keys})
        M = np.zeros((len(self.keys), len(fam)))
        for i, k in enumerate(self.keys):
            M[i, fam.index(family_key(k))] = 1.0
        P, d = self._read_cached(U)
        return (d <= self.reach) & ((P @ M).max(1) >= float(self.cfg.min_confidence))

    def _read_cached(self, V: np.ndarray):
        """read_many(V) memoised on V's content (the reviewed builds repeat; v3.1)."""
        key = (V.shape, hash(V.tobytes()))
        memo = self.__dict__.setdefault("_vmemo", {})
        hit = memo.get(key)
        if hit is None:
            P, d, _ = self.read_many(V)
            if len(memo) > 64:
                memo.clear()
            hit = memo[key] = (P, d)
        return hit[0].copy(), hit[1]

    def kernel(self, base):
        """Wrap a similarity kernel (u, v) -> s with the barrier."""
        def k(a, b):
            s = base(a, b)
            return s * float(self.barrier_factor(np.asarray(a), np.asarray(b)[None])[0]) if s > 0 else s
        return k

    def kernel_many(self, h2: float):
        """(u, V) -> [m] the reach kernel times the barrier, in one call (v3.1)."""
        def k(u, V):
            s = np.exp(-((V - u[None]) ** 2).sum(1) / (2 * h2))
            on = s > 0
            if on.any():
                s = s.copy()
                s[on] = s[on] * self.barrier_factor(u, V[on])
            return s
        return k

    # ------------------------------------------------------------ region names
    def region_labels(self, centroids: np.ndarray) -> List[Optional[str]]:
        """Archetype of each region centre (None when the centre is unlocated)."""
        if not self.locators or centroids is None or len(centroids) == 0:
            return [None] * (0 if centroids is None else len(centroids))
        return [self.read(c).archetype for c in centroids]

    # ------------------------------------------------------------ validation
    def loo(self, U: Optional[np.ndarray] = None) -> dict:
        """Leave-one-out: each locator read from the others. Locators of an archetype
        with a single locator cannot be scored (no other locator names it)."""
        U = self.U if U is None else U
        m = len(self.locators)
        if m < 2 or len(self.keys) < 2:
            return dict(locators=m, archetypes=len(self.keys), scored=0)
        D2 = np.maximum(((U[:, None] - U[None]) ** 2).sum(-1), 0.0)
        np.fill_diagonal(D2, np.inf)
        P = self._posterior(self._archetype_logk(D2, self.lab))
        counts = np.bincount(self.lab, minlength=len(self.keys))
        ok = counts[self.lab] >= 2
        pt = np.clip(P[np.arange(m), self.lab], 1e-6, 1.0)
        pred = P.argmax(1)
        dmin = np.sqrt(D2.min(1))
        # same vs different archetype: AUC of -distance over all pairs
        iu = np.triu_indices(m, 1)
        d = np.sqrt(D2[iu]); same = self.lab[iu[0]] == self.lab[iu[1]]
        auc = _auc(-d[same], -d[~same]) if same.any() and (~same).any() else None
        per = {}
        for p, key in enumerate(self.keys):
            s = (self.lab == p) & ok
            if s.any():
                per[self.names[key]] = dict(locators=int(counts[p]), accuracy=round(float((pred[s] == p).mean()), 3),
                                            loglik=round(float(np.log(pt[s]).mean()), 3))
            else:
                per[self.names[key]] = dict(locators=int(counts[p]), accuracy=None, loglik=None)
        fine = None
        if self.refining and ok.any():
            Pf = self.loo_fine(P)
            predf = Pf.argmax(1)
            fine = round(float((predf[ok] == self.lab[ok]).mean()), 4)
            for p_, key in enumerate(self.keys):
                s_ = (self.lab == p_) & ok
                if s_.any():
                    per[self.names[key]]["fine_accuracy"] = round(float((predf[s_] == p_).mean()), 3)
        twins = [dict(a=self.names[self.keys[q["a"]]], b=self.names[self.keys[q["b"]]], overlap=q["overlap"],
                      tells=q["tells"], corrected=q["gain"], broken=q["loss"], active=q["active"] and self.cfg.refine)
                 for q in getattr(self, "pairs", [])]
        Pc, predc = (Pf, predf) if fine is not None else (P, pred)   # conflicts: the fine reading
        conflicts = []
        for j in np.flatnonzero(ok & (predc != self.lab) & (Pc.max(1) >= self.cfg.min_confidence)
                                & (dmin <= self.reach)):
            nn = int(np.argmin(D2[j]))
            conflicts.append(dict(locator=self.locators[j].locator_id, archetype=self.locators[j].archetype,
                                  reads_as=self.names[self.keys[int(predc[j])]], confidence=round(float(Pc[j].max()), 3),
                                  nearest=self.locators[nn].locator_id, distance=round(float(dmin[j]), 3)))
        return dict(locators=m, archetypes=len(self.keys), scored=int(ok.sum()),
                    fine_accuracy=fine, twins=twins,
                    accuracy=round(float((pred[ok] == self.lab[ok]).mean()), 4) if ok.any() else None,
                    loglik=round(float(np.log(pt[ok]).mean()), 4) if ok.any() else None,
                    chance_loglik=round(float(-np.log(len(self.keys))), 4),
                    auc=None if auc is None else round(auc, 4),
                    per_archetype=per, conflicts=conflicts)

    def loo_fine(self, P: np.ndarray) -> np.ndarray:
        """Leave-one-out twin split of each locator: the pair's tells rebuilt without it
        (the kernel part P is already leave-one-out)."""
        Pf = np.array(P, float, copy=True)
        for q in self.active_pairs:
            for j, pf in q["loo"].items():
                a, b = q["a"], q["b"]
                m = Pf[j, a] + Pf[j, b]
                if m >= self.cfg.twin_mass:
                    sh = pf[a] / max(pf[a] + pf[b], 1e-300)
                    Pf[j, a], Pf[j, b] = m * sh, m * (1 - sh)
        return Pf

    def loo_loglik(self, U: np.ndarray) -> np.ndarray:
        """[scored] per-locator leave-one-out log p(true archetype)."""
        D2 = np.maximum(((U[:, None] - U[None]) ** 2).sum(-1), 0.0)
        np.fill_diagonal(D2, np.inf)
        P = self._posterior(self._archetype_logk(D2, self.lab))
        ok = np.bincount(self.lab, minlength=len(self.keys))[self.lab] >= 2
        return np.log(np.clip(P[np.arange(len(U)), self.lab], 1e-6, 1.0))[ok]

    def eligible(self) -> bool:
        """Enough locators to select the metric: select_min_per_archetype in at least select_min_archetypes archetypes."""
        c = np.bincount(self.lab, minlength=len(self.keys)) if len(self.lab) else np.zeros(0, int)
        return int((c >= self.cfg.select_min_per_archetype).sum()) >= self.cfg.select_min_archetypes

    def select_power(self, bm, current: float) -> dict:
        """Leave-one-out log-likelihood of the locators under each metric power in
        metric_grid. The power moves from `current` to the grid value with the
        largest mean paired gain among those whose gain exceeds both select_z
        standard errors and select_min_gain nats; otherwise it stays."""
        sp = self.space
        grid = sorted(set(float(g) for g in self.cfg.metric_grid) | {float(current)})
        rows = np.arange(len(bm)) if len(bm) <= 20000 else np.sort(
            np.random.default_rng(sp.cfg.seed).choice(len(bm), 20000, replace=False))
        Yd = sp._y_rows(bm, rows, sp.solve_latent(bm, rows))
        old = sp.cfg.metric_power
        A = sp._A()
        ll = {}
        try:
            for g in grid:
                sp.cfg.metric_power = g
                om = sp._variance_weights(bm, rows, Yd)
                if sp.cfg.metric_calibrate:
                    om = om * sp._calibration(Yd, om)
                _, R = np.linalg.qr(A * np.sqrt(om)[:, None])
                U = (self.Y - sp.ymean) @ R.T / np.sqrt(len(sp.c))
                ll[g] = self.loo_loglik(U)
        finally:
            sp.cfg.metric_power = old
        base = ll[float(current)]
        rows_out, best, best_gain = [], float(current), 0.0
        for g in grid:
            diff = ll[g] - base
            mean = float(diff.mean()) if len(diff) else 0.0
            se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("inf")
            rows_out.append(dict(power=g, loglik=round(float(ll[g].mean()), 4), gain=round(mean, 4),
                                 se=round(se, 4) if np.isfinite(se) else None))
            if (g != current and mean > self.cfg.select_z * se and mean >= self.cfg.select_min_gain
                    and mean > best_gain):
                best, best_gain = g, mean
        return dict(current=float(current), chosen=best, moved=best != float(current), grid=rows_out,
                    scored=int(len(base)), z=self.cfg.select_z,
                    min_gain=self.cfg.select_min_gain)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties half (rank formula)."""
    s = np.r_[pos, neg]
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def new_locator(build: Build, archetype: str, link: Optional[str] = None, added_by: Optional[str] = None,
                locator_id: Optional[str] = None) -> Locator:
    """A locator with a fresh id and the current time."""
    if not archetype_key(archetype):
        raise ValueError("a locator needs an archetype")
    return Locator(locator_id or uuid.uuid4().hex[:12], build, " ".join(str(archetype).split()), link, added_by,
                   time.time())
