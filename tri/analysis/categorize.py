"""Build categorization (math §12.1).

Every build gets three labels and a soft reading:

  - frame: oath, top two attunements, weapon class (game context, no model);
  - archetype: the located archetype of the build (math §4.7), or the name of
    its region when no locator exists;
  - variant: the side it takes of each sidegrade of its archetype (math §4.9);
  - reading: pure (pi_1 >= 0.8), leaning (0.5 <= pi_1 < 0.8), hybrid (pi_1 < 0.5
    and pi_1 + pi_2 >= 0.8) or unplaced (otherwise, or nearest locator beyond
    reach).

On top of these: the Ward hierarchy of archetype centres (families at reach),
discovery of unnamed archetypes among unplaced builds (stability-chosen
k-means, as for regions) and quality measures of the categorization.

Nothing here fits a parameter or feeds back into worth: all of it reads the
field, the locators and the ledger."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.fast import sqdist

from tri.analysis.defs import Atlas
from tri.core.context import build_context
from tri.core.types import Build

PURE, LEANING, HYBRID_SUM = 0.8, 0.5, 0.8
ATTUNEMENTS = ("Flamecharm", "Frostdraw", "Thundercall", "Galebreathe", "Shadowcast", "Ironsing", "Bloodrend")
WEAPONS = {"HeavyWeapon": "Heavy", "MediumWeapon": "Medium", "LightWeapon": "Light"}
READINGS = ("pure", "leaning", "hybrid", "unplaced")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def soft_kind(pi1, pi2, dmin, reach: float) -> np.ndarray:
    """A2 reading per build: pure | leaning | hybrid | unplaced."""
    pi1, pi2, dmin = (np.atleast_1d(np.asarray(x, float)) for x in (pi1, pi2, dmin))
    out = np.full(len(pi1), "unplaced", dtype=object)
    out[pi1 >= LEANING] = "leaning"
    out[pi1 >= PURE] = "pure"
    out[(pi1 < LEANING) & (pi1 + pi2 >= HYBRID_SUM)] = "hybrid"
    out[dmin > reach] = "unplaced"
    return out


def _rows(tri, bm):
    """(U, L) from one latent solve."""
    sp = tri.space
    rows = np.arange(len(bm))
    Y = sp._y_rows(bm, rows, sp.solve_latent(bm, rows, False))
    return sp.u_of(Y), sp.beta + Y @ sp._A().T


def read_coords(tri, U: np.ndarray, X=None, A=None, ctx=None) -> Tuple[np.ndarray, np.ndarray, List[str], str, float]:
    """(P [n, A], dmin [n], names, source, reach) for coordinates U. With the rows'
    picks X and availability A, the locator reading is the fine one (math §6.6)."""
    U = np.atleast_2d(U)
    ls = tri.locset
    if ls is not None and len(ls.keys) >= 1:
        P, dmin, _ = ls.read_many(U)
        if X is not None and A is not None and ls.refining:
            P = ls.refine(P, X, np.asarray(A, float), dmin, U, ctx)
        elif ls.refining:                            # reviewed builds: their remembered fine reading
            P = ls.fine_rows(U)
        return P, dmin, [ls.names[k] for k in ls.keys], "locators", ls.reach
    sp = tri.space
    C = sp.centroids
    d2 = sqdist(U, C)
    s = -d2 / (2 * sp.cfg.bandwidth ** 2)
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), np.sqrt(d2.min(1)), list(sp.region_names), "regions", float("inf")


def _ctx_rows(bm) -> List[set]:
    """Context tokens (oath:/murmur:) of each matrix row, for indicator notes."""
    T = bm.T.tocsr()
    return [{bm.tokens[j] for j in T.indices[T.indptr[i]:T.indptr[i + 1]]} for i in range(T.shape[0])]


def _top2(P: np.ndarray):
    o = np.argsort(-P, 1)
    top = o[:, 0]
    second = o[:, 1] if P.shape[1] > 1 else np.full(len(P), -1)
    pi1 = P[np.arange(len(P)), top]
    pi2 = P[np.arange(len(P)), second] if P.shape[1] > 1 else np.zeros(len(P))
    return top, second, pi1, pi2


def build_atlas(tri, builds=None, sample: Optional[int] = 3000, seed: int = 0) -> Atlas:
    from tri.views.plots import _ids, _matrix
    bm = _matrix(tri, builds, sample, seed)
    U, L = _rows(tri, bm)
    P, dmin, names, source, reach = read_coords(tri, U, bm.X, bm.masks[bm.gid], _ctx_rows(bm))
    top, second, pi1, pi2 = _top2(P)
    kind = soft_kind(pi1, pi2, dmin, reach)
    w = np.asarray(bm.w, float)
    A = len(names)
    placed = kind != "unplaced"
    cent = np.zeros((A, U.shape[1]))
    count = np.zeros(A, int)
    share = np.zeros(A)
    tot = max(w.sum(), 1e-12)
    for a in range(A):
        m = (top == a) & placed
        count[a] = int(m.sum())
        share[a] = float(w[m].sum() / tot)
        if m.any():
            cent[a] = w[m] @ U[m] / w[m].sum()
        elif source == "locators":
            cent[a] = tri.locset.U[tri.locset.lab == a].mean(0)
        else:
            cent[a] = tri.space.centroids[a]
    X = (bm.X.toarray() > 0)
    return Atlas(names, source, U, w, _ids(bm), bm, P, dmin, top, second, pi1, pi2, kind, cent, share, count,
                 sigmoid(L), np.asarray(bm.masks[bm.gid], bool), X, float(reach),
                 float(tri.locset.h if tri.locset is not None else tri.space.cfg.bandwidth), w @ U / tot)
# ------------------------------------------------------------------ one build


def frame_of(tri, b: Build) -> dict:
    """A1 frame: oath, top two attunements, weapon class (game context only)."""
    st = dict(b.final_stats or {})
    att = sorted(((s, v) for s, v in st.items() if s in ATTUNEMENTS and v > 0), key=lambda x: -x[1])[:2]
    wp = max(((WEAPONS[s], v) for s, v in st.items() if s in WEAPONS), key=lambda x: x[1], default=(None, 0))
    oath = tri.catalog.oath_of(b)
    parts = [oath or "no oath"] + [a for a, _ in att] + [wp[0] if wp[1] > 0 else "no weapon"]
    return dict(oath=oath, attunements=[a for a, _ in att], weapon=wp[0] if wp[1] > 0 else None,
                label=" · ".join(parts))


def variants_of(tri, b: Build, archetype_regions: Optional[Sequence[int]] = None) -> List[dict]:
    """A1 variant: side of each sidegrade that applies to the build."""
    sel = set(b.items())
    out = []
    for g in tri.sidegrades or []:
        if not g.applies(sel):
            continue
        ca, cb = g.counts(sel)
        s = ca / len(g.a) - cb / len(g.b)
        side = "A" if s >= 0.5 else "B" if s <= -0.5 else "mixed"
        out.append(dict(side=side, score=round(float(s), 3), a=list(g.a), b=list(g.b),
                        label=(f"{g.a[0]} side" if side == "A" else f"{g.b[0]} side" if side == "B"
                               else f"mixed {g.a[0]} / {g.b[0]}")))
    return out


def reading_text(kind: str, n1: str, n2: Optional[str]) -> str:
    if kind == "pure":
        return n1
    if kind == "leaning":
        return f"{n1}, leaning {n2}" if n2 else n1
    if kind == "hybrid":
        return f"{n1} / {n2} hybrid"
    return f"unplaced (nearest {n1})"


def categorize(tri, b: Build, top: int = 3) -> dict:
    """The three labels and the soft reading of one build (A1, A2)."""
    u = tri.space.point(b, loo=False).u
    xa = tri._vec(b) if tri.locset is not None else (None, None)
    P, dmin, names, source, reach = read_coords(tri, u[None], None if xa[0] is None else xa[0][None],
                                                None if xa[1] is None else xa[1][None],
                                                [set(build_context(b)[0])])
    p = P[0]
    o = np.argsort(-p)
    pi1 = float(p[o[0]]); pi2 = float(p[o[1]]) if len(o) > 1 else 0.0
    kind = str(soft_kind(pi1, pi2, dmin, reach)[0])
    n1, n2 = names[o[0]], (names[o[1]] if len(o) > 1 else None)
    arch = None if kind == "unplaced" else n1
    if arch is None and source == "regions":
        arch = n1
    return dict(frame=frame_of(tri, b), archetype=arch, best=n1, second=n2, reading=kind,
                text=reading_text(kind, n1, n2), source=source,
                pi=[(names[i], round(float(p[i]), 4)) for i in o[:top]],
                distance=round(float(dmin[0]), 4), reach=reach, variants=variants_of(tri, b), u=u)
# ------------------------------------------------------------------ geometry


def separating(tri, ua: np.ndarray, ub: np.ndarray, k: int = 3, mask: Optional[np.ndarray] = None) -> List[dict]:
    """C3: the items accounting for most of d(ua, ub)^2. Each item's share is
    omega_t (g_t . (ua - ub))^2 / sum_s omega_s (g_s . (ua - ub))^2 (exact by
    Theorem 4.1). dlogit > 0: the item is worth more at ua."""
    sp = tri.space
    dl = sp.item_coords() @ (np.asarray(ua) - np.asarray(ub))
    c = sp.weights() * dl ** 2
    tot = max(float(c.sum()), 1e-12)
    if mask is not None:
        c = np.where(mask, c, 0.0)
    o = np.argsort(-c)[:k]
    return [dict(item=tri.catalog.ids[j], share=float(c[j] / tot), dlogit=float(dl[j])) for j in o if c[j] > 0]


def signature(tri, uC: np.ndarray, ubar: np.ndarray, avail_rate: np.ndarray, k: int = 5) -> List[dict]:
    """A4: largest positive logit offsets g_t . (uC - ubar) among items the group can take."""
    off = tri.space.item_coords() @ (np.asarray(uC) - np.asarray(ubar))
    off = np.where(avail_rate >= 0.5, off, -np.inf)
    o = np.argsort(-off)[:k]
    return [dict(item=tri.catalog.ids[j], offset=float(off[j])) for j in o if np.isfinite(off[j]) and off[j] > 0]


def hierarchy(tri, at: Atlas, cut: Optional[float] = None, split_items: int = 3) -> dict:
    """A3: Ward dendrogram of the archetype centres (with builds), families cut
    at reach. Internal nodes carry the items separating their two children."""
    from scipy.cluster.hierarchy import fcluster, linkage
    keep = [a for a in range(len(at.names)) if at.count[a] > 0 or at.source == "locators"]
    cut = float(at.reach if np.isfinite(at.reach) else 2 * at.h) if cut is None else float(cut)
    out = dict(names=[at.names[a] for a in keep], index=keep, share=[float(at.share[a]) for a in keep],
               count=[int(at.count[a]) for a in keep], n=len(at.names), cut=cut, Z=None, families=[1] * len(keep), splits=[])
    if at.source == "locators":
        lab = tri.locset.lab
        out["locators"] = [int((lab == a).sum()) for a in keep]
    else:
        out["locators"] = [0] * len(keep)
    if len(keep) < 2:
        return out
    C = at.centres[keep]
    Z = linkage(C, "ward")
    out["Z"] = Z
    out["families"] = [int(f) for f in fcluster(Z, cut, "distance")]
    n = len(keep)
    members = {i: [i] for i in range(n)}
    wt = np.maximum(np.array(out["share"]), 1e-6)
    for j, (a, b, hgt, _) in enumerate(Z):
        a, b = int(a), int(b)
        ma, mb = members[a], members[b]
        ca = (wt[ma, None] * C[ma]).sum(0) / wt[ma].sum()
        cb = (wt[mb, None] * C[mb]).sum(0) / wt[mb].sum()
        out["splits"].append(dict(node=n + j, height=float(hgt), left=a, right=b,
                                  items=separating(tri, ca, cb, split_items)))
        members[n + j] = ma + mb
    return out


def families(tri, at: Atlas, cut: Optional[float] = None) -> List[List[str]]:
    h = hierarchy(tri, at, cut)
    fam: Dict[int, List[str]] = {}
    for n, f in zip(h["names"], h["families"]):
        fam.setdefault(f, []).append(n)
    return sorted(fam.values(), key=len, reverse=True)
# ------------------------------------------------------------------ discovery


def _km(sp, U: np.ndarray, w: np.ndarray, k: int, seed: int, restarts: int = 4) -> np.ndarray:
    """Weighted k-means, best of a few restarts (lowest weighted inertia), so the
    split-half comparison below measures the data, not the initialisation."""
    best, bi = None, np.inf
    for r in range(restarts):
        Ck = sp._kmeans(U, w, k, seed + 101 * r)
        lab = sp._nearest(U, Ck)
        inert = float(w @ ((U - Ck[lab]) ** 2).sum(1))
        if inert < bi:
            best, bi = Ck, inert
    return best


def _stable_k(sp, U: np.ndarray, w: np.ndarray, kmax: int = 8, ari: float = 0.8, seed: int = 0) -> int:
    """Largest k whose split-half clusterings agree (median ARI >= ari over 3 splits)."""
    from sklearn.metrics import adjusted_rand_score
    best = 1
    for k in range(2, min(kmax, len(U) // 20) + 1):
        aris = []
        for rep in range(3):
            perm = np.random.default_rng([seed, k, rep]).permutation(len(U))
            h1, h2 = perm[: len(U) // 2], perm[len(U) // 2:]
            C1 = _km(sp, U[h1], w[h1], k, seed + 1 + rep)
            C2 = _km(sp, U[h2], w[h2], k, seed + 11 + rep)
            aris.append(adjusted_rand_score(sp._nearest(U, C1), sp._nearest(U, C2)))
        if float(np.median(aris)) >= ari:
            best = k
    return best


def _density(U: np.ndarray, c: np.ndarray, h: float) -> float:
    return float(np.mean(np.exp(-((U - c) ** 2).sum(1) / (2 * h ** 2)))) if len(U) else 0.0


def discover(tri, at: Atlas, min_builds: int = 20, seed: int = 0, signature_items: int = 5) -> List[dict]:
    """A4: candidate archetypes among the unplaced builds. Each with its medoid,
    signature items, nearest archetypes, size and growth (from timestamps)."""
    sp = tri.space
    um = ~at.placed
    idx = np.flatnonzero(um)
    if len(idx) < max(min_builds, 40):
        return []
    U, w = at.U[idx], at.w[idx]
    k = _stable_k(sp, U, w, seed=seed)
    C = _km(sp, U, w, k, seed + 7) if k > 1 else (w @ U / w.sum())[None]
    lab = sp._nearest(U, C)
    dens_ref = [_density(at.U[at.members(a)], at.centres[a], at.h) for a in range(len(at.names)) if at.count[a] >= 5]
    ref = float(np.median(dens_ref)) if dens_ref else 0.0
    ts = np.asarray(at.bm.ts, float)
    known_ts = np.isfinite(ts)
    out = []
    for j in range(len(C)):
        m = lab == j
        if m.sum() < min_builds:
            continue
        rows = idx[m]
        dens = _density(at.U[rows], C[j], at.h)
        if dens < ref:
            continue
        sub = rows if len(rows) <= 300 else np.random.default_rng(seed).choice(rows, 300, replace=False)
        D = np.linalg.norm(at.U[sub][:, None] - at.U[sub][None], axis=2)
        med = int(sub[D.sum(1).argmin()])
        dc = np.linalg.norm(at.centres - C[j][None], axis=1)
        near = [dict(name=at.names[a], distance=float(dc[a])) for a in np.argsort(dc)[:2]]
        growth = None
        if known_ts.sum() >= 40:
            t_cut = np.quantile(ts[known_ts], 0.75)
            new_ = known_ts & (ts >= t_cut); old_ = known_ts & (ts < t_cut)
            inC = np.zeros(len(ts), bool); inC[rows] = True
            s_new = inC[new_].mean() if new_.any() else 0.0
            s_old = inC[old_].mean() if old_.any() else 0.0
            growth = dict(recent_share=float(s_new), earlier_share=float(s_old),
                          ratio=float(s_new / s_old) if s_old > 0 else None)
        out.append(dict(id=f"C{len(out) + 1}", builds=int(m.sum()), weight=float(w[m].sum()),
                        share=float(w[m].sum() / at.w.sum()), density=dens, reference_density=ref,
                        medoid=at.ids[med], medoid_row=med, centre=C[j],
                        signature=signature(tri, C[j], at.ubar, at.avail[rows].mean(0), signature_items),
                        nearest=near, growth=growth))
    out.sort(key=lambda d: -d["weight"])
    for i, d in enumerate(out):
        d["id"] = f"C{i + 1}"
    return out
# ------------------------------------------------------------------ quality


def quality(tri, at: Atlas) -> dict:
    """A5: locator leave-one-out confusion and margins, located share,
    stability under the removal of one locator, review consistency."""
    out: dict = dict(source=at.source, archetypes=len(at.names),
                     located_share=float(at.w[at.placed].sum() / max(at.w.sum(), 1e-12)),
                     readings={k: int((at.kind == k).sum()) for k in READINGS})
    ts = np.asarray(at.bm.ts, float)
    if np.isfinite(ts).sum() >= 20:
        t_cut = np.nanquantile(ts, 0.9)
        new_ = np.isfinite(ts) & (ts >= t_cut)
        out["located_share_new"] = float(at.placed[new_].mean()) if new_.any() else None
    led = tri.ledger
    rid = list(led.by_id) if led is not None else []
    if rid:
        Ur = np.array([led.vector(r) for r in rid])
        P, dmin, *_ = read_coords(tri, Ur)
        top, second, pi1, pi2 = _top2(P)
        kr = soft_kind(pi1, pi2, dmin, at.reach)
        out["located_share_reviewed"] = float((kr != "unplaced").mean())
        arch = {r: (int(top[i]) if kr[i] != "unplaced" else -1) for i, r in enumerate(rid)}
        conf = [s for s in led.superseded if s.similarity >= 0.95]
        same = [s for s in conf if arch.get(s.older, -2) == arch.get(s.newer, -3) and arch.get(s.older) != -1]
        out["review_conflicts"] = len(conf)
        out["review_consistency"] = float(len(same) / len(conf)) if conf else None
    ls = tri.locset
    if at.source != "locators" or ls is None:
        return out
    m = len(ls.locators)
    lab = ls.lab
    D2 = np.maximum(((ls.U[:, None] - ls.U[None]) ** 2).sum(-1), 0.0)
    np.fill_diagonal(D2, np.inf)
    Pl = ls._posterior(ls._archetype_logk(D2, lab))
    ok = np.bincount(lab, minlength=len(ls.keys))[lab] >= 2
    pred = Pl.argmax(1)
    A = len(ls.keys)
    conf = np.zeros((A, A), int)
    for i in np.flatnonzero(ok):
        conf[lab[i], pred[i]] += 1
    out["confusion"] = conf.tolist()
    out["loo_accuracy"] = float((pred[ok] == lab[ok]).mean()) if ok.any() else None
    D = np.sqrt(D2)
    margins = []
    for i in range(m):
        own = D[i][(lab == lab[i])]
        oth = D[i][(lab != lab[i])]
        own_d = float(own.min()) if np.isfinite(own).any() else None
        oth_d = float(oth.min()) if len(oth) else None
        mg = (oth_d - own_d) if (own_d is not None and oth_d is not None) else None
        margins.append(dict(locator=ls.locators[i].locator_id, archetype=ls.locators[i].archetype,
                            margin=None if mg is None else round(mg, 4)))
    out["margins"] = margins
    # stability: remove one locator at a time, re-read the atlas sample
    D2a = sqdist(at.U, ls.U)
    base = np.where(at.placed, at.top, -1)
    stab = []
    for j in range(m):
        Dj = D2a.copy(); Dj[:, j] = np.inf
        Pj = ls._posterior(ls._archetype_logk(Dj, lab))
        dj = np.sqrt(Dj.min(1))
        tj, _, p1, p2 = _top2(Pj)
        kj = soft_kind(p1, p2, dj, at.reach)
        lj = np.where(kj != "unplaced", tj, -1)
        stab.append(float((at.w * (lj == base)).sum() / at.w.sum()))
    out["stability"] = float(np.mean(stab)) if stab else None
    if stab:
        j = int(np.argmin(stab))
        out["weakest_locator"] = dict(locator=ls.locators[j].locator_id, archetype=ls.locators[j].archetype,
                                      stability=round(stab[j], 4))
    out["per_locator_stability"] = [round(s, 4) for s in stab]
    out["names"] = [ls.names[k] for k in ls.keys]
    return out


# ------------------------------------------------------------------ intersections
def intersection(at: Atlas, S: Sequence[int], k: int = 3, pool: int = 40, exclude=()) -> dict:
    """k representative builds where the archetypes S meet.

    Score of build i: the geometric mean g_i = (prod_{a in S} P_ia)^(1/|S|),
    largest when its archetype mass is split evenly inside S; strength
    s_i = |S| g_i lies in [0, 1] (1: all mass in S, evenly). Builds near a
    locator (dmin <= reach) are preferred. From the `pool` best, picks are
    greedy on g_i min(1, d_i / delta), d_i the concept distance to the builds
    already picked and delta the median pairwise distance of the pool, so
    the k builds differ. Rows whose id is in `exclude` never appear. Returns rows [dict(i, id, score, strength, shares)]
    and `near` (the share of rows near a locator)."""
    S = [int(a) for a in dict.fromkeys(S)]
    if len(S) < 2:
        raise ValueError("an intersection needs at least two archetypes")
    P = np.asarray(at.P, float)
    g = np.exp(np.log(np.maximum(P[:, S], 1e-12)).mean(1))
    near = np.asarray(at.dmin, float) <= float(at.reach)
    ok = np.array([i not in set(exclude) for i in at.ids], bool) if exclude else np.ones(len(P), bool)
    near &= ok
    cand = np.flatnonzero(near) if near.sum() >= k else np.flatnonzero(ok)
    cand = cand[np.argsort(-g[cand])][:max(pool, k)]
    U = np.asarray(at.U, float)[cand]
    D = np.sqrt(np.maximum(sqdist(U, U), 0.0)) if len(cand) > 1 else np.zeros((1, 1))
    delta = float(np.median(D[np.triu_indices(len(cand), 1)])) if len(cand) > 1 else 1.0
    delta = max(delta, 1e-9)
    chosen: List[int] = []
    dmin = np.full(len(cand), np.inf)
    for _ in range(min(k, len(cand))):
        val = g[cand] * np.minimum(1.0, dmin / delta)
        val[chosen] = -1.0
        j = int(np.argmax(val))
        chosen.append(j)
        dmin = np.minimum(dmin, D[j])
    rows = []
    for j in chosen:
        i = int(cand[j])
        rows.append(dict(i=i, id=at.ids[i], score=float(g[i]), strength=float(len(S) * g[i]),
                         near=bool(near[i]), shares={at.names[a]: float(P[i, a]) for a in S}))
    return dict(archetypes=[at.names[a] for a in S], rows=rows, near_locators=bool(near.sum() >= k))
