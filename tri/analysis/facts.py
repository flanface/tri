"""Nontrivial facts (math §12.2).

A fact is a short, local, certified sentence about the meta. The pipeline:

  1. propose candidates from the families below on the mining half of the
     builds (a split by build id, so a build and its copies stay together);
  2. drop candidates that do not beat their triviality baseline;
  3. certify each on the held-out half with a one-sided test against the
     baseline, Benjamini-Hochberg across all candidates at q = 0.05;
  4. rank by surprise x support (log(1 + n));
  5. render as a sentence with its numbers.

Families:
  threshold   C2: an item pays off in an archetype only well above its requirement;
  defining    C3: an item marks an archetype (lift >= 2 against all who could take it);
  separating  C3: the items that make up most of the distance of two neighbours;
  hole / gem  C4: reviewers almost always remove an item the pretrain takes (add one it skips);
  synergy     C5: removing s moves the worth of t, beyond rules and sidegrades;
  universal   C6: every archetype takes t except one;
  shift       C7: an archetype's picks moved after a patch (>= 30 post-patch builds).

Reviews are accurate, so C4 is tested on all reviews of the archetype against
the corpus rate (the reviews are the held-out evidence). Nothing here fits a
parameter or feeds back into worth."""
from __future__ import annotations

import json
import math
import os
import time as _time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tri.analysis.categorize import separating
from tri.analysis.defs import Atlas, Fact

Q_FDR = 0.05
MIN_POST_PATCH = 30
ALWAYS = 2 / 3                     # "always" for holes and gems: at least 2 in 3 reviews, and the binomial test
# ------------------------------------------------------------------ statistics


def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval of k successes in n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - r), min(1.0, c + r)


def _upper(z: float) -> float:
    """One-sided p-value P(Z >= z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def p_diff(k1, n1, k2, n2, delta0: float = 0.0) -> float:
    """H0: p1 - p2 <= delta0 (one-sided, unpooled, +0.5 smoothing)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)
    se = math.sqrt(p1 * (1 - p1) / (n1 + 1) + p2 * (1 - p2) / (n2 + 1))
    return _upper((p1 - p2 - delta0) / max(se, 1e-9))


def p_ratio(k1, n1, k2, n2, r0: float = 1.0) -> float:
    """H0: p1 / p2 <= r0 (one-sided, log scale, +0.5 smoothing)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = (k1 + 0.5) / (n1 + 1), (k2 + 0.5) / (n2 + 1)
    se = math.sqrt((1 - p1) / ((n1 + 1) * p1) + (1 - p2) / ((n2 + 1) * p2))
    return _upper((math.log(p1) - math.log(p2) - math.log(r0)) / max(se, 1e-9))


def p_binom(k: int, n: int, p0: float, lower: bool) -> float:
    """Exact binomial tail: P(X <= k) (lower) or P(X >= k) under p0."""
    from scipy.stats import binom
    if n <= 0:
        return 1.0
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    return float(binom.cdf(k, n, p0) if lower else binom.sf(k - 1, n, p0))


def benjamini_hochberg(p: Sequence[float], q: float = Q_FDR) -> np.ndarray:
    """Boolean mask of discoveries at FDR q."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    out = np.zeros(m, bool)
    if ok.any():
        out[o[: int(np.flatnonzero(ok).max()) + 1]] = True
    return out


def in10(p: float) -> str:
    """'9 in 10' for mid-range shares, a percentage near 0 or 1."""
    if 0.05 <= p <= 0.95:
        return f"{int(round(10 * p))} in 10"
    return f"{100 * p:.0f}%"


def _split(at: Atlas, seed: int = 0) -> np.ndarray:
    """True = held-out half (by build id; rows without one by position)."""
    out = np.zeros(len(at.ids), bool)
    for i, b in enumerate(at.ids):
        key = f"{seed}:{b if b is not None else i}"
        out[i] = zlib.crc32(key.encode()) % 2 == 1
    return out


def single_requirement(item) -> Optional[Tuple[str, int]]:
    """(stat, value) when the item has one stat requirement and nothing else."""
    if item.requires and len(item.requires) == 1 and not item.requires_any and not item.clauses:
        (s, v), = item.requires.items()
        return s, int(v)
    if not item.requires and len(item.clauses or []) == 1 and len(item.clauses[0]) == 1:
        lit = item.clauses[0][0]
        if isinstance(lit, dict) and len(lit.get("stats", [])) == 1:
            return lit["stats"][0], int(lit.get("value", 0))
    return None


class FactMiner:
    """Proposes, certifies and ranks facts on one atlas."""

    def __init__(self, tri, at: Atlas, seed: int = 0, min_builds: int = 30):
        self.tri, self.at = tri, at
        self.c = tri.catalog
        self.hold = _split(at, seed)
        self.min_builds = min_builds
        self.ids = tri.catalog.ids
        A = len(at.names)
        self.M = np.full((A, at.Q.shape[1]), np.nan)      # mean worth per archetype (builds that can take)
        self.R = np.full((A, at.Q.shape[1]), np.nan)      # pick rate per archetype
        self.Av = np.zeros((A, at.Q.shape[1]))            # availability rate per archetype
        for a in range(A):
            m = at.members(a)
            if m.sum() == 0:
                continue
            av = at.avail[m]
            den = av.sum(0)
            self.Av[a] = av.mean(0)
            self.M[a] = np.where(den > 0, (at.Q[m] * av).sum(0) / np.maximum(den, 1), np.nan)
            self.R[a] = np.where(den > 0, (at.X[m] & av).sum(0) / np.maximum(den, 1), np.nan)

    # ---------------------------------------------------------- helpers
    def _rate(self, rows: np.ndarray, t: int) -> Tuple[int, int]:
        av = self.at.avail[rows, t]
        return int((self.at.X[rows, t] & av).sum()), int(av.sum())

    def _arch(self, a: int, held: Optional[bool]) -> np.ndarray:
        m = self.at.members(a)
        if held is None:
            return np.flatnonzero(m)
        return np.flatnonzero(m & (self.hold == held))

    def _big(self) -> List[int]:
        return [a for a in range(len(self.at.names)) if self.at.count[a] >= self.min_builds]

    # ---------------------------------------------------------- C3 defining
    def defining(self, top: int = 4) -> List[Fact]:
        at, out = self.at, []
        IC = self.tri.space.item_coords()
        for a in self._big():
            mine = self._arch(a, False)
            oth = np.flatnonzero(~at.members(a) & ~self.hold)
            off = IC @ (at.centres[a] - at.ubar)
            cand = []
            for t in np.argsort(-off)[:40]:
                if self.Av[a, t] < 0.5 or off[t] <= 0:
                    continue
                k1, n1 = self._rate(mine, t); k2, n2 = self._rate(oth, t)
                if n1 < 10 or n2 < 10:
                    continue
                r1, r2 = k1 / n1, (k2 + 0.5) / (n2 + 1)
                if r1 >= 0.7 and r1 / r2 >= 2:
                    cand.append((r1 / r2, t))
            for _, t in sorted(cand, reverse=True)[:top]:
                hk = self._arch(a, True); ho = np.flatnonzero(~at.members(a) & self.hold)
                k1, n1 = self._rate(hk, t); k2, n2 = self._rate(ho, t)
                p = p_ratio(k1, n1, k2, n2, 1.2)
                r1, r2 = k1 / max(n1, 1), k2 / max(n2, 1)
                lift = (k1 + 0.5) / (n1 + 1) / ((k2 + 0.5) / (n2 + 1))
                nm, it = at.names[a], self.ids[t]
                out.append(Fact(f"defining:{nm}:{it}", "defining", nm,
                                f"{it} is the mark of {nm}: {in10(r1)} of its builds take it, against "
                                f"{in10(r2)} of others who could.",
                                dict(rate=round(r1, 3), others=round(r2, 3), lift=round(lift, 2)), p,
                                math.log(min(max(lift, 1e-9), 20.0) / 1.2), n1 + n2, f"card:{nm}", items=[it]))
        return out

    # ---------------------------------------------------------- C3 separating
    def separating(self) -> List[Fact]:
        at, out, seen = self.at, [], set()
        big = self._big()
        for a in big:
            others = [b for b in big if b != a]
            if not others:
                continue
            b = min(others, key=lambda x: float(np.linalg.norm(at.centres[a] - at.centres[x])))
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            unlocked = np.abs(self.Av[a] - self.Av[b]) <= 0.5      # baseline: locked by context
            sep = separating(self.tri, at.centres[a], at.centres[b], 2, mask=unlocked)
            if not sep:
                continue
            share = sum(s["share"] for s in sep)
            if share < 0.3:
                continue
            t = self.c.index[sep[0]["item"]]
            hi, lo = (a, b) if sep[0]["dlogit"] > 0 else (b, a)
            k1, n1 = self._rate(self._arch(hi, True), t); k2, n2 = self._rate(self._arch(lo, True), t)
            p = p_diff(k1, n1, k2, n2, 0.1)
            its = [s["item"] for s in sep]
            na, nb = at.names[a], at.names[b]
            out.append(Fact(f"separating:{'|'.join(sorted((na, nb)))}", "separating", na,
                            f"{na} and {nb} differ mostly in {' and '.join(its)} "
                            f"({100 * share:.0f}% of the distance between them).",
                            dict(share=round(share, 3), items=its, other=nb,
                                 shares=[round(float(x["share"]), 3) for x in sep],
                                 distance=round(float(np.linalg.norm(at.centres[a] - at.centres[b])), 3)),
                            p, share, min(n1, n2), "atlas", items=its))
        return out

    # ---------------------------------------------------------- C2 thresholds
    def thresholds(self, step: int = 5, min_gap: int = 10) -> List[Fact]:
        at, out = self.at, []
        stats = list(at.bm.catalog.stats)
        for t, it in enumerate(self.ids):
            req = single_requirement(self.c.items[it])
            if req is None or req[0] not in stats:
                continue
            s, r = req
            si = stats.index(s)
            for a in self._big():
                if self.Av[a, t] < 0.3 or not (0.15 <= np.nan_to_num(self.R[a, t]) <= 0.95):
                    continue
                mine = self._arch(a, False)
                mine = mine[at.avail[mine, t]]
                if len(mine) < 20:
                    continue
                v = at.bm.fi[mine, si]
                q = at.Q[mine, t]
                s_star = None
                for lv in range(r, 101, step):
                    m_lo, m_hi = (v >= lv) & (v < lv + step), v >= lv
                    if m_lo.sum() >= 5 and q[m_lo].mean() >= 0.5 and m_hi.sum() >= 10:
                        s_star = lv
                        break
                if s_star is None or s_star < r + min_gap:
                    continue
                below = q[(v >= r) & (v < s_star)]
                if len(below) < 5 or below.mean() >= 0.5:
                    continue
                ho = self._arch(a, True)
                ho = ho[at.avail[ho, t]]
                vh = at.bm.fi[ho, si]
                hi_, lo_ = ho[vh >= s_star], ho[(vh >= r) & (vh < s_star)]
                k1, n1 = int(at.X[hi_, t].sum()), len(hi_)
                k2, n2 = int(at.X[lo_, t].sum()), len(lo_)
                p = p_diff(k1, n1, k2, n2, 0.1)
                nm = at.names[a]
                out.append(Fact(f"threshold:{nm}:{it}", "threshold", nm,
                                f"In {nm}, {it} pays off only from {s_star} {s} (requirement {r}): "
                                f"{in10(k1 / max(n1, 1))} take it above, {in10(k2 / max(n2, 1))} below.",
                                dict(stat=s, requirement=r, threshold=s_star, above=round(k1 / max(n1, 1), 3),
                                     below=round(k2 / max(n2, 1), 3)),
                                p, (s_star - r) / 20.0, n1 + n2, f"stats:{it}", items=[it]))
        return out

    # ---------------------------------------------------------- C4 holes / gems
    def holes(self, min_reviews: int = 5) -> List[Fact]:
        """A hole: a take with a high pretrain pick rate (P >= 0.5) that reviewers
        almost always remove (share >= ALWAYS); a gem: low pick rate (P <= 0.3),
        almost always added (v2.21: "trap" renamed, "always" = ALWAYS)."""
        tri, at, out = self.tri, self.at, []
        led = tri.ledger
        if led is None or not led.by_id:
            return out
        from tri.analysis.categorize import _top2, read_coords, soft_kind
        from tri.model.matrix import BuildMatrix
        rid = list(led.by_id)
        Ur = np.array([led.vector(r) for r in rid])
        P, dmin, *_ = read_coords(tri, Ur)
        top, _, p1, p2 = _top2(P)
        kind = soft_kind(p1, p2, dmin, at.reach)
        origs = [led.by_id[r].original for r in rid]
        bm = BuildMatrix.from_builds(origs, tri.catalog, fields=tri.space.fields, tokens=tri.space.tokens)
        av = np.asarray(bm.masks[bm.gid], bool)
        for a in self._big():
            rows = [i for i in range(len(rid)) if top[i] == a and kind[i] != "unplaced"]
            if len(rows) < min_reviews:
                continue
            had: Dict[int, List[int]] = {}
            lack: Dict[int, List[int]] = {}
            for i in rows:
                cl = {c.item: c.label for c in led.active_claims(rid[i]) if c.origin == "review"}
                sel = set(origs[i].items())
                for it, lab in cl.items():
                    if it not in tri.catalog.index:
                        continue
                    t = tri.catalog.index[it]
                    if it in sel and lab in ("keep", "remove"):
                        had.setdefault(t, [0, 0])
                        had[t][0] += lab == "remove"; had[t][1] += 1
                for t in np.flatnonzero(av[i]):
                    it = self.ids[t]
                    if it in sel:
                        continue
                    lack.setdefault(int(t), [0, 0])
                    lack[int(t)][0] += cl.get(it) == "add"; lack[int(t)][1] += 1
            nm = at.names[a]
            for t, (k, n) in had.items():
                pc = float(np.nan_to_num(self.R[a, t]))
                if n < min_reviews or pc < 0.5 or k / n < ALWAYS:
                    continue
                p = p_binom(n - k, n, pc, lower=True)
                it = self.ids[t]
                out.append(Fact(f"hole:{nm}:{it}", "hole", nm,
                                f"A hole: {in10(pc)} {nm} builds take {it}, but reviewers remove it from "
                                f"{k} of the {n} they reviewed.",
                                dict(corpus=round(pc, 3), removed=k, reviewed=n), p, abs((1 - k / n) - pc), n,
                                f"item:{it}", items=[it]))
            for t, (k, n) in lack.items():
                pc = float(np.nan_to_num(self.R[a, t]))
                if n < min_reviews or pc > 0.3 or k / n < ALWAYS:
                    continue
                p = p_binom(k, n, pc, lower=False)
                it = self.ids[t]
                out.append(Fact(f"gem:{nm}:{it}", "gem", nm,
                                f"A gem: few {nm} builds take {it} ({100 * pc:.0f}%), but reviewers add it to "
                                f"{k} of the {n} they reviewed without it.",
                                dict(corpus=round(pc, 3), added=k, reviewed=n), p, abs(k / n - pc), n,
                                f"item:{it}", items=[it]))
        return out

    traps = holes                                          # the pre-2.21 name

    # ---------------------------------------------------------- C5 synergy
    def synergy(self, sources: int = 6, rows_per: int = 40, min_delta: float = 0.15) -> List[Fact]:
        tri, at, out = self.tri, self.at, []
        from tri.analysis.categorize import sigmoid
        c = self.c
        side_pairs = set()
        for g in tri.sidegrades or []:
            for x in g.a + g.b:
                for y in g.a + g.b:
                    side_pairs.add((x, y))
        for a in self._big():
            mine = self._arch(a, False)
            rate = np.nan_to_num(self.R[a])
            cand = [t for t in np.argsort(-(rate * (1 - rate))) if 0.3 <= rate[t] <= 0.9][:sources]
            for s in cand:
                rows = mine[at.X[mine, s]]
                if len(rows) < 15:
                    continue
                rows = rows[:rows_per]
                bm = at.bm.subset(rows)
                X = bm.X.tolil(); X[:, s] = 0; bm.X = X.tocsr(); bm.X.eliminate_zeros()
                q0 = at.Q[rows]
                q1 = sigmoid(tri.space.logits_rows(bm))
                av = at.avail[rows]
                d = np.where(av, q0 - q1, 0).sum(0) / np.maximum(av.sum(0), 1)
                d[s] = 0
                si = self.ids[s]
                for t in np.flatnonzero(np.abs(d) >= min_delta):
                    ti = self.ids[t]
                    it_t, it_s = c.items[ti], c.items[si]
                    if (si in it_t.prerequisites or ti in it_s.prerequisites or si in it_t.exclusive_with
                            or ti in it_s.exclusive_with or (si, ti) in side_pairs
                            or any(si in g for g in (it_t.prerequisites_any or []))):
                        continue
                    ho = self._arch(a, True)
                    ho = ho[at.avail[ho, t]]
                    w_, wo = ho[at.X[ho, s]], ho[~at.X[ho, s]]
                    k1, n1 = int(at.X[w_, t].sum()), len(w_)
                    k2, n2 = int(at.X[wo, t].sum()), len(wo)
                    pos = d[t] > 0
                    p = p_diff(k1, n1, k2, n2, 0.0) if pos else p_diff(k2, n2, k1, n1, 0.0)
                    nm = at.names[a]
                    word = "raises" if pos else "lowers"
                    out.append(Fact(f"synergy:{nm}:{si}:{ti}", "synergy" if pos else "anti-synergy", nm,
                                    f"In {nm}, {si} {word} the worth of {ti} "
                                    f"({d[t]:+.2f}): {in10(k1 / max(n1, 1))} with it take {ti}, "
                                    f"{in10(k2 / max(n2, 1))} without.",
                                    dict(delta=round(float(d[t]), 3), with_=round(k1 / max(n1, 1), 3),
                                         without=round(k2 / max(n2, 1), 3)),
                                    p, abs(float(d[t])), n1 + n2, f"card:{nm}", items=[si, ti]))
        return out

    # ---------------------------------------------------------- C6 universal
    def universal(self, high: float = 0.9, low: float = 0.5) -> List[Fact]:
        at, out = self.at, []
        big = self._big()
        if len(big) < 3:
            return out
        M = self.M[big]
        for t in range(M.shape[1]):
            col = M[:, t]
            if np.isnan(col).any():
                continue
            lowk = np.flatnonzero(col < low)
            if len(lowk) != 1 or (col >= high).sum() != len(big) - 1:
                continue
            cix = big[int(lowk[0])]
            if self.Av[cix, t] < 0.5:
                continue                                          # locked: trivial
            hc = self._arch(cix, True)
            ho = np.flatnonzero(at.placed & ~at.members(cix) & self.hold)
            k1, n1 = self._rate(ho, t); k2, n2 = self._rate(hc, t)
            p = p_diff(k1, n1, k2, n2, 0.2)
            takers = [b for b in big if b != cix]
            nb = min(takers, key=lambda x: float(np.linalg.norm(at.centres[cix] - at.centres[x])))
            sep = [s for s in separating(self.tri, at.centres[cix], at.centres[nb], 8) if s["dlogit"] > 0]
            it, nm = self.ids[t], at.names[cix]
            tail = f": {nm} builds take {sep[0]['item']} instead" if sep else ""
            out.append(Fact(f"universal:{it}:{nm}", "universal", nm,
                            f"Every archetype takes {it} (worth {high} or more), except {nm} "
                            f"({float(col[lowk[0]]):.1f}){tail}.",
                            dict(worth_except=round(float(col[lowk[0]]), 3), instead=sep[0]["item"] if sep else None),
                            p, float(high - col[lowk[0]]), n1 + n2, f"item:{it}", items=[it]))
        return out

    # ---------------------------------------------------------- C7 shifts
    def shifts(self, min_move: float = 0.25) -> List[Fact]:
        tri, at, out = self.tri, self.at, []
        pts = tri.patches.patches if tri.patches is not None else []
        ts = np.asarray(at.bm.ts, float)
        if not pts or not np.isfinite(ts).any():
            return out
        pt = pts[-1]
        post_all = np.isfinite(ts) & (ts >= pt.time)
        pre_all = np.isfinite(ts) & (ts < pt.time)
        for a in self._big():
            m = at.members(a)
            pre, post = np.flatnonzero(m & pre_all), np.flatnonzero(m & post_all)
            if len(pre) < 10 or len(post) < 5:
                continue
            ra = (at.X[pre] & at.avail[pre]).sum(0) / np.maximum(at.avail[pre].sum(0), 1)
            rb = (at.X[post] & at.avail[post]).sum(0) / np.maximum(at.avail[post].sum(0), 1)
            ok = (at.avail[pre].sum(0) >= 5) & (at.avail[post].sum(0) >= 5)
            dv = np.where(ok, rb - ra, 0)
            drop, gain = int(np.argmin(dv)), int(np.argmax(dv))
            moved = float(np.linalg.norm(at.U[post].mean(0) - at.U[pre].mean(0)))
            if max(-dv[drop], dv[gain]) < min_move and moved <= at.h:
                continue
            parts, its, ps, changes = [], [], [], []
            for t, verb in ((drop, "dropped"), (gain, "took")):
                if abs(dv[t]) < min_move:
                    continue
                kb, nb_ = self._rate(post, t); ka, na_ = self._rate(pre, t)
                ps.append(p_diff(ka, na_, kb, nb_, 0.1) if verb == "dropped" else p_diff(kb, nb_, ka, na_, 0.1))
                parts.append(f"{verb} {self.ids[t]} ({ra[t]:.2f} → {rb[t]:.2f})")
                changes.append(dict(item=self.ids[t], before=round(float(ra[t]), 3), after=round(float(rb[t]), 3)))
                its.append(self.ids[t])
            if moved > at.h:
                parts.append(f"moved {moved:.2f} in concept space")
            nm = at.names[a]
            p = min(1.0, min(ps) * len(ps)) if ps else None
            if len(post) < MIN_POST_PATCH:
                p = None                                          # stays a candidate
            out.append(Fact(f"shift:{pt.name}:{nm}", "shift", nm,
                            f"Since patch {pt.name}, {nm} builds " + " and ".join(parts) + f" ({len(post)} builds).",
                            dict(patch=pt.name, time=pt.time, post=len(post), pre=len(pre), moved=round(moved, 3),
                                 moved_far=bool(moved > at.h), changes=changes),
                            p, float(max(abs(dv[drop]), abs(dv[gain]), moved / max(at.h, 1e-9) / 4)), len(post),
                            f"item:{its[0]}" if its else "atlas", items=its))
        return out

    # ---------------------------------------------------------- pipeline
    def run(self, q: float = Q_FDR, families: Optional[Sequence[str]] = None) -> List[Fact]:
        fams = dict(defining=self.defining, separating=self.separating, threshold=self.thresholds,
                    hole=self.holes, synergy=self.synergy, universal=self.universal, shift=self.shifts)
        from tri.rules.triviality import trivial_fact
        facts: List[Fact] = []
        self.trivial = []
        for k, f in fams.items():
            if families is None or k in families or k == "hole" and "trap" in families:
                for x in f():
                    why = trivial_fact(x, self.c)
                    (self.trivial.append((x.key, why)) if why else facts.append(x))
        tested = [f for f in facts if f.p is not None]
        ok = benjamini_hochberg([f.p for f in tested], q)
        for f, o in zip(tested, ok):
            f.status = "certified" if o else "candidate"
        for f in facts:
            f.score = float(f.surprise * math.log1p(max(f.support, 0)))
        facts.sort(key=lambda f: (f.status != "certified", -f.score))
        return facts


def mine_facts(tri, at: Atlas, seed: int = 0, q: float = Q_FDR, families=None) -> List[Fact]:
    return FactMiner(tri, at, seed).run(q, families)
# ------------------------------------------------------------------ fact book (retirement, digest)


class FactBook:
    """Certified facts over time: when each was first and last certified, and
    when it was retired (failed recertification)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.rows: Dict[str, dict] = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.rows = json.load(f)
            except (OSError, ValueError):
                self.rows = {}

    def update(self, facts: Sequence[Fact], now: Optional[float] = None) -> dict:
        now = float(now if now is not None else _time.time())
        cert = {f.key: f for f in facts if f.status == "certified"}
        new, retired = [], []
        for k, f in cert.items():
            r = self.rows.get(k)
            if r is None or r.get("retired"):
                new.append(k)
                self.rows[k] = dict(sentence=f.sentence, symbol=f.symbol, family=f.family, archetype=f.archetype,
                                    first=now, last=now, retired=None)
            else:
                r.update(sentence=f.sentence, symbol=f.symbol, last=now)
        for k, r in self.rows.items():
            if k not in cert and not r.get("retired"):
                r["retired"] = now
                retired.append(k)
        self.save()
        return dict(new=new, retired=retired)

    def retired(self, since: float = 0.0) -> List[dict]:
        return [dict(key=k, **r) for k, r in self.rows.items() if r.get("retired") and r["retired"] >= since]

    def digest(self, since: float) -> dict:
        return dict(new=[dict(key=k, **r) for k, r in self.rows.items()
                         if r["first"] >= since and not r.get("retired")],
                    retired=self.retired(since))

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.rows, f)
        os.replace(tmp, self.path)
