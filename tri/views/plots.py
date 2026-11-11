"""Research views of a fitted model: builds on the concept space, items on the
worth space, worth along paths, stats and distances.

Every view has two halves:

    data     a function returning plain numpy arrays and lists (no plotting);
             use it with your own tools
    plot_*   draws that data with matplotlib (optional dependency,
             `pip install matplotlib`) and returns the figure

Views
    concept_map      builds placed on a projection of the concept space (k axes),
                     coloured by archetype region; optional item arrows (biplot)
    item_map         items placed in the same plane: the direction in which each
                     item's logit rises fastest
    worth_space      items placed by their worth across builds: two items are
                     close when the same builds value them alike
    region_worth     archetype region x item table of mean worth (heatmap)
    worth_profile    one build's worth for every item, picked items marked
    worth_path       worth of items while one build's archetype moves to another's
    stat_response    worth of items as one stat of a build is varied
    distance_check   concept distance against worth difference, with the 1/4 bound
    review_reach     how strongly a review on one build reaches every other build

Projections (Projection, fit_projection)
    "pca"      weighted principal components of the builds' coordinates; the
               linear view that keeps the most of the true distances
    "regions"  principal components of the region centres only; spreads archetypes
    "items"    axis j = logit of item j (up to a constant): an exact, readable
               view of two chosen items' worth, not distance-preserving

Views and axes
    plot_concept_map and plot_worth_space draw one view: three axes (`dims`) as
    a 3-D volume with drop lines, or two as a plane. axis_groups cuts k axes
    into views (1-2-3, 4-5-6, ...); concept_axes / worth_axes say what each axis
    means (items at each end, regions at each end or pulling towards it), and
    `axes="explain"` writes that on the figure. plot_axis_summary draws the
    variance per axis and the distance each view keeps. concept_colour colours
    builds by an item's worth or a stat; worth_neighbours lists the items valued
    most like one item.

Every Projection reports `distance_kept`: the correlation between true concept
distances and distances in the picture on a sample of pairs; view_kept gives
the same for one view. Read each picture with that number in mind; a view of a
many-dimensional space always loses some distance.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tri.core.types import Build
from tri.model.matrix import BuildMatrix
from tri.model.space import sigmoid

Builds = Union[BuildMatrix, Sequence[Build], None]


# ------------------------------------------------------------------ helpers
def _plt():
    try:
        import matplotlib
        if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("tri.views plots need matplotlib: pip install matplotlib") from e
    return plt


def _matrix(tri, builds: Builds, sample: Optional[int], seed: int) -> BuildMatrix:
    """BuildMatrix of the builds (default: the model's corpus), aligned to the space."""
    sp = tri.space
    if builds is None:
        bm = tri.data
    elif isinstance(builds, BuildMatrix):
        bm = builds
    else:
        bm = BuildMatrix.from_builds(list(builds), tri.catalog, fields=sp.fields, tokens=sp.tokens)
    if sample is not None and len(bm) > sample:
        rows = np.sort(np.random.default_rng(seed).choice(len(bm), sample, replace=False))
        bm = bm.subset(rows)
    return bm


def _ids(bm: BuildMatrix) -> List[Optional[str]]:
    ids = getattr(bm, "ids", None)
    return list(ids) if ids is not None else [None] * len(bm)


def _items(tri, items) -> List[str]:
    if items is None:
        return list(tri.catalog.ids)
    return [i for i in items if i in tri.catalog.index]


def _pass_of(tri, i: str) -> str:
    try:
        return tri.rulebook.pass_of(i)
    except Exception:                                           # pragma: no cover
        return "luxury"


PASS_COLOURS = {"foundational": "#1f5f8b", "central": "#d1495b", "archetype": "#e07a3f", "luxury": "#8d99ae",
                "exclusion": "#2a9d8f", "review": "#e9c46a"}

# ------------------------------------------------------------------ themes
#: Colours per theme. "light" for papers and notebooks, "dark" for Discord (its
#: dark mode background), both drawn with seaborn's styles when seaborn is
#: installed and the same colours through plain matplotlib settings otherwise.
THEMES = {
    "light": dict(bg="#ffffff", panel="#f7f8fa", fg="#1f2328", muted="#6b7280", grid="#e3e6ea",
                  edge="#ffffff", star="#111111", style="whitegrid",
                  passes={"foundational": "#3b6fb6", "central": "#d1495b", "archetype": "#e07a3f", "luxury": "#9aa5b8",
                          "exclusion": "#2a9d8f", "review": "#e0a526"},
                  seq="mako_r", div="vlag", heat="rocket_r"),
    "dark": dict(bg="#2b2d31", panel="#313338", fg="#dbdee1", muted="#949ba4", grid="#3f4147",
                 edge="#2b2d31", star="#ffffff", style="darkgrid",
                 passes={"foundational": "#6ea8fe", "central": "#ff6b81", "archetype": "#f4a261", "luxury": "#a7b1c2",
                         "exclusion": "#3ddbbf", "review": "#ffd166"},
                 seq="mako", div="vlag", heat="rocket"),
}
_T = dict(THEMES["light"])


def _sns():
    try:
        import seaborn
        return seaborn
    except ImportError:
        return None


def theme_rc(name: str = "light") -> dict:
    """matplotlib settings of a theme (seaborn's style underneath when available)."""
    t = THEMES[name]
    rc = {}
    sns = _sns()
    if sns is not None:
        rc.update(sns.axes_style(t["style"]))
        rc.update(sns.plotting_context("notebook", font_scale=0.8))
    rc.update({"figure.facecolor": t["bg"], "savefig.facecolor": t["bg"], "axes.facecolor": t["panel"],
               "axes.edgecolor": t["grid"], "axes.labelcolor": t["fg"], "text.color": t["fg"],
               "xtick.color": t["muted"], "ytick.color": t["muted"], "grid.color": t["grid"],
               "axes.titlecolor": t["fg"], "axes.titleweight": "bold", "axes.titlesize": 10,
               "legend.facecolor": t["panel"], "legend.edgecolor": t["grid"], "legend.labelcolor": t["fg"],
               "legend.frameon": True, "font.family": ["DejaVu Sans"], "axes.spines.top": False,
               "axes.spines.right": False, "axes.grid": True, "grid.linewidth": 0.6})
    return rc


class theme:
    """Draw in a theme: `with render.theme("dark"): fig = render.plot_concept_map(...)`.
    Everything drawn inside uses the theme's colours; settings are restored after."""

    def __init__(self, name: str = "light"):
        if name not in THEMES:
            raise ValueError(f"theme is one of {', '.join(THEMES)}")
        self.name = name

    def __enter__(self):
        global _T
        plt = _plt()
        self._prev, self._rc = dict(_T), plt.rc_context(theme_rc(self.name))
        self._rc.__enter__()
        _T = dict(THEMES[self.name])
        return self

    def __exit__(self, *exc):
        global _T
        _T = self._prev
        return self._rc.__exit__(*exc)


def _pass_colour(p) -> str:
    return _T["passes"].get(p, _T["muted"])


def _cmap(kind: str = "seq"):
    """A theme colour map by role: seq (one value), div (signed), heat (tables)."""
    sns = _sns()
    name = _T[kind]
    if sns is not None:
        try:
            return sns.color_palette(name, as_cmap=True)
        except (ValueError, KeyError):
            pass
    return {"seq": "viridis", "div": "coolwarm", "heat": "magma"}[kind]


# ------------------------------------------------------------------ projections
@dataclass
class Projection:
    """x = (u - mean) @ W. W has one column per plotted axis."""
    method: str
    mean: np.ndarray
    W: np.ndarray
    explained: List[float] = field(default_factory=list)   # share of variance per axis ("pca", "regions")
    distance_kept: float = float("nan")                      # corr(true distance, plane distance)
    labels: List[str] = field(default_factory=list)          # axis names

    def transform(self, U: np.ndarray) -> np.ndarray:
        return (np.asarray(U) - self.mean) @ self.W

    def to_dict(self) -> dict:
        return dict(method=self.method, explained=[round(x, 4) for x in self.explained],
                    distance_kept=round(float(self.distance_kept), 4), labels=self.labels)


def _distance_kept(U: np.ndarray, XY: np.ndarray, pairs: int = 3000, seed: int = 0) -> float:
    n = len(U)
    if n < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    a, b = rng.integers(n, size=pairs), rng.integers(n, size=pairs)
    k = a != b
    d = np.linalg.norm(U[a[k]] - U[b[k]], axis=1)
    e = np.linalg.norm(XY[a[k]] - XY[b[k]], axis=1)
    if d.std() < 1e-12 or e.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d, e)[0, 1])


def fit_projection(tri, U: np.ndarray, method: str = "pca", k: int = 2, weights: Optional[np.ndarray] = None,
                   items: Optional[Sequence[str]] = None) -> Projection:
    """A k-axis linear view of concept coordinates U ([b, D']).
    method "pca" | "regions" | "items" (items: the k items whose logits become the axes)."""
    U = np.asarray(U, float)
    w = np.ones(len(U)) if weights is None else np.asarray(weights, float)
    sp = tri.space
    if method == "items":
        its = _items(tri, items)[:k]
        if len(its) < k:
            bad = [i for i in (items or []) if i not in tri.catalog.index]
            raise ValueError(f"method='items' needs {k} known items (unknown: {bad})")
        IC = sp.item_coords()
        W = np.stack([IC[tri.catalog.index[i]] for i in its], 1)
        mean = (w @ U) / w.sum()
        P = Projection("items", mean, W, [], labels=[f"logit {i} (centred)" for i in its])
    else:
        M = sp.centroids if method == "regions" else U
        mw = np.ones(len(M)) if method == "regions" else w
        if method not in ("pca", "regions"):
            raise ValueError("method is 'pca', 'regions' or 'items'")
        mean = (mw @ M) / mw.sum()
        C = (M - mean) * np.sqrt(mw / mw.sum())[:, None]
        _, s, vt = np.linalg.svd(C, full_matrices=False)
        W = vt[:k].T
        tot = float((s ** 2).sum()) or 1.0
        P = Projection(method, mean, W, [float(s[j] ** 2 / tot) for j in range(min(k, len(s)))],
                       labels=[f"{'PC' if method == 'pca' else 'region axis '}{j + 1}" for j in range(k)])
    P.distance_kept = _distance_kept(U, P.transform(U))
    return P


# ------------------------------------------------------------------ data
def concept_map(tri, builds: Builds = None, method: str = "pca", k: int = 2, sample: Optional[int] = 3000,
                seed: int = 0, items: Optional[Sequence[str]] = None, partial: bool = False,
                projection: Optional[Projection] = None) -> dict:
    """Builds on a k-D projection of the concept space.
    Returns xy [b, k], U [b, D'], region [b], region_names, centre_xy [R, k],
    region_share, ids, projection (reuse it to place more builds on the same plane)."""
    sp = tri.space
    bm = _matrix(tri, builds, sample, seed)
    U = sp.coords(bm, partial=partial)
    P = projection or fit_projection(tri, U, method, k, bm.w, items)
    reg = sp.region_of(U)
    return dict(xy=P.transform(U), U=U, region=reg, region_names=list(sp.region_names),
                region_share=list(sp.region_size), centre_xy=P.transform(sp.centroids), ids=_ids(bm),
                weights=np.asarray(bm.w), projection=P, bm=bm)


def place(tri, builds: Sequence[Build], projection: Projection, partial: bool = False) -> dict:
    """Extra builds (a reviewed build, your own draft) on an existing plane."""
    bm = _matrix(tri, builds, None, 0)
    U = tri.space.coords(bm, partial=partial)
    return dict(xy=projection.transform(U), U=U, region=tri.space.region_of(U),
                ids=[b.build_id for b in builds])


def item_map(tri, projection: Projection, items: Optional[Sequence[str]] = None) -> dict:
    """Items in the plane of `projection`: xy[t] is the direction in which item t's
    logit rises (logit_t = const + item_coords[t] . u), its length how fast."""
    its = _items(tri, items)
    IC = tri.space.item_coords()[[tri.catalog.index[i] for i in its]]
    xy = IC @ projection.W
    return dict(items=its, xy=xy, strength=np.linalg.norm(xy, axis=1),
                kind=[tri.catalog.items[i].kind.value for i in its], pass_=[_pass_of(tri, i) for i in its])


def worth_matrix(tri, builds: Builds = None, sample: Optional[int] = 2000, seed: int = 0,
                 items: Optional[Sequence[str]] = None, partial: bool = False) -> dict:
    """Field worth (probability) of every item for every build: P [b, n_items],
    available [b, n_items] (prerequisites and kind allow it), region [b]."""
    sp = tri.space
    bm = _matrix(tri, builds, sample, seed)
    its = _items(tri, items)
    cols = [tri.catalog.index[i] for i in its]
    L = sp.logits_rows(bm, partial=partial)
    U = sp.coords(bm, partial=partial)
    return dict(items=its, P=sigmoid(L[:, cols]), available=np.asarray(bm.masks[bm.gid][:, cols], bool),
                region=sp.region_of(U), weights=np.asarray(bm.w), ids=_ids(bm))


def worth_space(tri, builds: Builds = None, items: Optional[Sequence[str]] = None, k: int = 2,
                sample: Optional[int] = 2000, seed: int = 0, min_rate: float = 0.0, relative: bool = False) -> dict:
    """Items placed by their worth across builds. Each item is the vector of its
    worth over the builds; distance between two items = RMS difference of their
    worth over those builds (weighted; worth 0 where the build cannot take the
    item), so items valued by the same builds sit together. Returns items, xy [n, k], explained, distance_kept, distance [n, n],
    correlation [n, n], mean_worth, kind, pass_.
    min_rate: drop items whose mean worth is below this (removes the never-taken).
    relative: subtract each item's mean worth first, so items are compared by
    which builds value them more than usual, not by how often they are taken
    (the first axis of the plain view is mostly overall pick rate)."""
    wm = worth_matrix(tri, builds, sample, seed, items)
    P, w = wm["P"] * wm["available"], wm["weights"]          # an item a build cannot take is worth 0 to it
    keep = (w @ P) / w.sum() >= min_rate
    its = [i for i, k_ in zip(wm["items"], keep) if k_]
    P = P[:, keep]
    sw = np.sqrt(w / w.sum())[:, None]
    M = (P - (w @ P) / w.sum() if relative else P) * sw       # item t = column t, a point in build space
    G = M.T @ M                                               # squared distance via the Gram matrix
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0))
    Ci = (M - M.mean(1, keepdims=True)).T                     # items as rows, centred over items
    Ut, s2, Vt = np.linalg.svd(Ci, full_matrices=False)       # PCA of the item points = classical MDS of D
    xy = Ut[:, :k] * s2[:k]
    # which archetype regions push items along each axis: item t's coordinate j is
    # sum_b sw_b V_jb (worth_bt - mean), so region r pulls with sum_{b in r} sw_b V_jb
    reg = wm["region"]
    nreg = len(tri.space.region_names)
    pull = np.array([[float((sw[reg == r, 0] * Vt[j, reg == r]).sum()) for r in range(nreg)]
                     for j in range(min(k, len(Vt)))]) if nreg else np.zeros((min(k, len(Vt)), 0))
    sgn = np.sign(xy.sum(0) + 1e-300)                         # make each axis point to where most items are
    if not relative:
        xy, pull = xy * sgn, pull * sgn[: len(pull), None]
    tot = float((s2 ** 2).sum()) or 1.0
    C = (P - (w @ P) / w.sum()) * sw                          # per-item centred: correlation across builds
    sd = np.sqrt(np.maximum((C ** 2).sum(0), 1e-12))
    R = (C.T @ C) / np.outer(sd, sd)
    iu = np.triu_indices(len(its), 1)
    e = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    dk = float(np.corrcoef(D[iu], e[iu])[0, 1]) if len(its) > 2 else float("nan")
    return dict(items=its, xy=xy, relative=relative, explained=[float(s2[j] ** 2 / tot) for j in range(min(k, len(s2)))],
                distance_kept=dk, distance=D, correlation=R, mean_worth=(w @ P) / w.sum(),
                region_pull=pull, region_names=list(tri.space.region_names),
                kind=[tri.catalog.items[i].kind.value for i in its], pass_=[_pass_of(tri, i) for i in its])


def region_worth(tri, builds: Builds = None, items: Optional[Sequence[str]] = None, sample: Optional[int] = 4000,
                 seed: int = 0, top: Optional[int] = 40) -> dict:
    """Mean worth of each item in each archetype region (over builds that could take it).
    top: keep the items whose worth varies most across regions.
    Returns regions (names), items, M [R, n], share."""
    wm = worth_matrix(tri, builds, sample, seed, items)
    P, A, w, reg = wm["P"], wm["available"], wm["weights"], wm["region"]
    R = len(tri.space.region_names)
    M = np.full((R, P.shape[1]), np.nan)
    for r in range(R):
        m = reg == r
        ww = (w[m][:, None] * A[m])
        den = ww.sum(0)
        M[r] = np.where(den > 0, (ww * P[m]).sum(0) / np.maximum(den, 1e-12), np.nan)
    its = wm["items"]
    if top is not None and len(its) > top:
        order = np.argsort(-np.nan_to_num(np.nanstd(M, 0)))[:top]
        order = np.sort(order)
        M, its = M[:, order], [its[j] for j in order]
    return dict(regions=list(tri.space.region_names), items=its, M=M,
                share=[float((w[reg == r]).sum() / w.sum()) for r in range(R)])


def worth_profile(tri, build: Build, items: Optional[Sequence[str]] = None, partial: bool = False) -> dict:
    """One build's field worth for every item it could take (leave-one-out for its
    own picks: what the rest of the build says). Sorted by worth, highest first."""
    sp = tri.space
    pt = sp.point(build, partial=partial)
    bm = BuildMatrix.from_builds([build], tri.catalog, fields=sp.fields, tokens=sp.tokens)
    avail = np.asarray(bm.masks[bm.gid[0]], bool)
    its = [i for i in _items(tri, items) if avail[tri.catalog.index[i]]]
    p = sigmoid(pt.logits)
    sel = set(build.items())
    rows = sorted(((i, float(p[tri.catalog.index[i]])) for i in its), key=lambda x: -x[1])
    return dict(items=[i for i, _ in rows], worth=np.array([q for _, q in rows]),
                picked=[i in sel for i, _ in rows], kind=[tri.catalog.items[i].kind.value for i, _ in rows],
                pass_=[_pass_of(tri, i) for i, _ in rows], region=sp.reading(pt.u).to_dict())


def worth_path(tri, a: Build, b: Build, steps: int = 21, items: Optional[Sequence[str]] = None,
               top: int = 12) -> dict:
    """Worth of items while build a's archetype (latent, stats and context) moves in a
    straight line to build b's. Logits are linear along the path, so this shows
    exactly which items trade places between the two archetypes and where.
    items None: the `top` items whose worth changes most. Returns t, items, P [steps, n], distance."""
    sp = tri.space
    pa, pb = sp.point(a, loo=False), sp.point(b, loo=False)
    t = np.linspace(0, 1, steps)
    Y = (1 - t)[:, None] * pa.y[None] + t[:, None] * pb.y[None]
    L = sp.beta + Y @ sp._A().T
    P = sigmoid(L)
    if items is None:
        ch = np.abs(P[-1] - P[0])
        cols = list(np.argsort(-ch)[:top])
    else:
        cols = [tri.catalog.index[i] for i in _items(tri, items)]
    return dict(t=t, items=[tri.catalog.ids[c] for c in cols], P=P[:, cols],
                distance=sp.distance_u(pa.u, pb.u),
                a=a.build_id, b=b.build_id)


def stat_response(tri, build: Build, stat: str, items: Optional[Sequence[str]] = None,
                  values: Sequence[int] = tuple(range(0, 101, 5)), top: int = 10) -> dict:
    """Worth of items as `stat` (highest and final) of build is set to each value,
    picks kept. Shows the stat steps the field learnt (\"from 50 Agility\").
    items None: the `top` items that respond most. Returns values, items, P [v, n]."""
    import copy
    if stat not in tri.catalog.stats:
        raise ValueError(f"unknown stat {stat!r}; one of {tri.catalog.stats}")
    sp = tri.space
    rows = []
    for v in values:
        b = copy.deepcopy(build)
        b.highest_stats = dict(b.highest_stats); b.final_stats = dict(b.final_stats)
        b.highest_stats[stat] = int(v)
        b.final_stats[stat] = int(min(v, b.final_stats.get(stat, v))) if stat in b.final_stats else int(v)
        rows.append(sigmoid(sp.point(b, loo=False).logits))
    P = np.array(rows)
    if items is None:
        cols = list(np.argsort(-(P.max(0) - P.min(0)))[:top])
    else:
        cols = [tri.catalog.index[i] for i in _items(tri, items)]
    return dict(stat=stat, values=list(values), items=[tri.catalog.ids[c] for c in cols], P=P[:, cols])


def distance_check(tri, builds: Builds = None, pairs: int = 3000, sample: Optional[int] = 2000, seed: int = 0) -> dict:
    """Pairs of builds: concept distance d, RMS worth difference over items and the
    bound d/4 it can never exceed (docs/math, Theorem 4.2).
    Returns d, rms_worth_diff, bound, holds (share of pairs within the bound)."""
    sp = tri.space
    bm = _matrix(tri, builds, sample, seed)
    U = sp.coords(bm)
    P = sigmoid(sp.logits_rows(bm))
    rng = np.random.default_rng(seed)
    a, b = rng.integers(len(U), size=pairs), rng.integers(len(U), size=pairs)
    k = a != b
    a, b = a[k], b[k]
    d = np.linalg.norm(U[a] - U[b], axis=1)
    rms = np.sqrt((sp.weights() * (P[a] - P[b]) ** 2).mean(1))      # omega-weighted, as the metric
    return dict(d=d, rms_worth_diff=rms, bound=d / 4, holds=float(np.mean(rms <= d / 4 + 1e-9)),
                same_region=sp.region_of(U[a]) == sp.region_of(U[b]))


def review_reach(tri, build: Build, builds: Builds = None, sample: Optional[int] = 3000, seed: int = 0) -> dict:
    """Similarity of every build to `build` under the review kernel (LedgerConfig.reach):
    how much a review written on `build` counts on each of them. Same order as
    concept_map(tri, builds, sample=sample, seed=seed)."""
    from tri.reviews.authority import ReviewLedger
    sp = tri.space
    bm = _matrix(tri, builds, sample, seed)
    U = sp.coords(bm)
    u0 = sp.project(build)
    d = np.linalg.norm(U - u0[None], axis=1)
    cfg = tri.cfg.ledger
    h2 = cfg.reach ** 2 / (-2 * np.log(cfg.sim_min))
    sim = np.exp(-d * d / (2 * h2))
    return dict(similarity=sim, distance=d, sim_min=cfg.sim_min, sim_full=cfg.sim_full, reach=cfg.reach,
                kernel=ReviewLedger.reach_kernel(cfg))


# ------------------------------------------------------------------ plots
def _ax(ax, size=(8, 6)):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    return ax.figure, ax


def _region_colours(R: int):
    """R distinct colours; neighbouring region numbers get far-apart hues."""
    sns = _sns()
    if sns is None:
        cmap = _plt().get_cmap("tab20" if R > 10 else "tab10")
        return [cmap(i % cmap.N) for i in range(R)]
    if R <= 10:
        return list(sns.color_palette("bright" if _T is not None and _T["bg"] != "#ffffff" else "deep", R))
    pal = sns.color_palette("husl", R, desat=0.85 if _T["bg"] == "#ffffff" else 0.95)
    step = next(k for k in range(max(2, R // 3), R + 1) if np.gcd(k, R) == 1)   # hop around the wheel
    return [pal[(i * step) % R] for i in range(R)]


def _ax_for(ax, dims, size=(8, 6)):
    """A 2-D axes, or a 3-D one when three dims are drawn."""
    if ax is not None:
        return ax.figure, ax
    plt = _plt()
    if len(dims) == 3:
        fig = plt.figure(figsize=(size[0] + 2.6, size[1] + 0.6))
        ax = fig.add_axes([-0.03, -0.02, 0.74, 1.0], projection="3d")   # v2.21: less dead space
        ax.set_box_aspect(None, zoom=0.97)                    # room for the z label
        ax.set_facecolor(_T["bg"])
        return fig, ax
    return _ax(None, size)


def _is3d(ax) -> bool:
    return getattr(ax, "name", "") == "3d"


def _sc(ax, P, **kw):
    """scatter of the rows of P (2 or 3 columns)."""
    if _is3d(ax):
        kw.setdefault("depthshade", True)
    return ax.scatter(*np.asarray(P).T, **kw)


def _ann(ax, p, text, dx=4, dy=4, **kw):
    """Label a point: offset annotation in 2-D, text at the point in 3-D."""
    if _is3d(ax):
        return ax.text(*p, " " + str(text), **{k: v for k, v in kw.items() if k != "textcoords"})
    return ax.annotate(str(text), tuple(p), xytext=(dx, dy), textcoords="offset points", **kw)


def _stems(ax, P, color=None):
    """3-D: drop lines from points to the floor, so their height can be read."""
    if not _is3d(ax) or not len(P):
        return
    z0 = ax.get_zlim()[0]
    for x, y, z in P:
        ax.plot([x, x], [y, y], [z0, z], color=color or _T["muted"], lw=0.7, alpha=0.7, ls=":")


def plot_concept_map(cm: dict, ax=None, color: Optional[np.ndarray] = None, cmap: str = "viridis",
                     color_label: str = "", highlight: Optional[dict] = None, items: Optional[dict] = None,
                     n_arrows: int = 12, labels: bool = True, title: Optional[str] = None, size: float = 6.0,
                     dims: Sequence[int] = (0, 1), axes: str = "on", axis_info: Optional[List[dict]] = None,
                     locators: Optional[dict] = None, view: Tuple[float, float] = (22, -58),
                     fields: bool = False, core: float = 0.9):
    """Scatter of concept_map(). dims: the projection axes shown (0-based); two
    give a plane, three a 3-D volume seen from view = (elevation, azimuth), with
    drop lines under stars and diamonds. color: one value per build (e.g.
    review_reach similarity, concept_colour()) instead of region colours.
    highlight: place() output, drawn as stars with ids. items: item_map()
    output; the n_arrows strongest are drawn. axes: "off" (no axes), "on"
    (ticks, grid, origin lines) or "explain" ("on" plus, for each axis, the items
    whose logits fall and rise along it, from axis_info = concept_axes()).
    locators: place() output with "labels", drawn as diamonds.
    fields: draw each archetype region as a translucent convex polytope (the
    hull of its `core` share of builds nearest the region centre) with a rigid
    border, instead of one dot per build (v2.21; math §12.6)."""
    dims = tuple(dims)
    fig, ax = _ax_for(ax, dims)
    d3 = _is3d(ax)
    xy = cm["xy"][:, dims]
    ps = size * (2.2 if d3 else 1.0)
    if fields:
        region_fields(ax, cm, dims, core)
    elif color is None:
        cols = _region_colours(len(cm["region_names"]))
        _sc(ax, xy, s=ps, c=[cols[r] for r in cm["region"]], alpha=0.62 if d3 else 0.5, linewidths=0)
    else:
        sc = _sc(ax, xy, s=ps, c=color, cmap=_cmap("seq") if cmap in ("viridis", "magma", None) else cmap,
                 alpha=0.8 if d3 else 0.7, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, label=color_label, fraction=0.035 if d3 else 0.046, pad=0.08 if d3 else 0.02,
                          shrink=0.6 if d3 else 1.0)
        cb.outline.set_visible(False)
    if labels and len(cm["centre_xy"]):
        from matplotlib.lines import Line2D
        cols = _region_colours(len(cm["region_names"]))
        for r, p in enumerate(cm["centre_xy"][:, dims]):
            box = dict(boxstyle="circle,pad=0.2", fc=cols[r], ec=_T["bg"], lw=1.2, alpha=0.95)
            if d3:
                ax.text(*p, str(r), fontsize=7, ha="center", va="center", weight="bold", color=_T["bg"], bbox=box)
            else:
                ax.annotate(str(r), tuple(p), fontsize=7, ha="center", va="center", weight="bold", color=_T["bg"],
                            bbox=box)
        h = [Line2D([], [], marker="o", ls="", ms=6, color=cols[r],
                    label=f"{r}  {_short(n, 34)}  {100 * cm['region_share'][r]:.0f}%")
             for r, n in enumerate(cm["region_names"])]
        leg = ax.legend(handles=h, fontsize=6.8, loc="upper left", borderaxespad=0.2, handletextpad=0.3,
                        bbox_to_anchor=((1.07 if color is None else 1.25) if d3 else (1.01 if color is None else 1.3), 0.98),
                        framealpha=0.0, title="regions", title_fontsize=7.5, alignment="left")
        leg.get_title().set_color(_T["muted"])
    if items is not None:
        V = items["xy"][:, dims]
        st = np.linalg.norm(V, axis=1)
        o = np.argsort(-st)[:n_arrows]
        span = np.abs(xy).max() if len(xy) else 1.0
        sc_ = 0.8 * span / max(st[o].max(), 1e-12) if len(o) else 1.0
        for j in o:
            v = V[j] * sc_
            c = _pass_colour(items["pass_"][j])
            if d3:
                ax.quiver(0, 0, 0, *v, color=c, alpha=0.8, arrow_length_ratio=0.12, lw=1.2)
                ax.text(*(v * 1.05), items["items"][j], fontsize=7, color=c)
            else:
                ax.arrow(0, 0, *v, color=c, width=0.002 * span, head_width=0.03 * span, alpha=0.8,
                         length_includes_head=True)
                ax.text(*(v * 1.05), items["items"][j], fontsize=7, color=c)
    if locators is not None and len(locators["xy"]):
        lx = locators["xy"][:, dims]
        _sc(ax, lx, marker="D", s=46, c=_T["panel"], edgecolors=_T["fg"], linewidths=1.0, zorder=4)
        _stems(ax, lx)
        for p, lab in zip(lx, locators.get("labels", locators.get("ids", []))):
            _ann(ax, p, str(lab)[:30], 4, -9, fontsize=7, style="italic")
    if highlight is not None:
        hx = highlight["xy"][:, dims]
        _sc(ax, hx, marker="*", s=220, c=_T["star"], edgecolors=_T["bg"], linewidths=0.8, zorder=5)
        _stems(ax, hx)
        for p, i in zip(hx, highlight["ids"]):
            if i:
                _ann(ax, p, i, fontsize=8)
    P = cm["projection"]
    _style_axes(ax, axes, [_axis_label(P, d) for d in dims], axis_info, dims)
    if d3:
        ax.view_init(*view)
    kept = _distance_kept(cm["U"], xy)
    ax.set_title(title or f"concept space · axes {'-'.join(str(d + 1) for d in dims)} · distance kept {kept:.2f}",
                 fontsize=10, loc="left")
    return fig


def region_hulls(cm: dict, dims: Sequence[int], core: float = 0.9, crease: float = 0.82) -> List[Optional[dict]]:
    """Per region: the convex hull of its `core` share of builds nearest its
    centre in the drawn axes, {points, simplices, vertices, centre, edges}
    (None: too few). 3-D edges: creases, where neighbouring facets turn by more
    than arccos(crease) (about 35 degrees), so the border reads as rigid."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.asarray(cm["xy"])[:, list(dims)]
    reg = np.asarray(cm["region"])
    out: List[Optional[dict]] = []
    for r in range(len(cm["region_names"])):
        P = xy[reg == r]
        if len(P) < len(dims) + 2:
            out.append(None)
            continue
        c = np.median(P, 0)
        d = np.linalg.norm(P - c, axis=1)
        P = P[d <= np.quantile(d, core)]
        try:
            h = ConvexHull(P)
        except (QhullError, ValueError):
            out.append(None)
            continue
        if len(dims) == 3:                     # feature edges: between facets of different planes
            E = set()
            for f, (s_, nb) in enumerate(zip(h.simplices, h.neighbors)):
                for k in range(3):
                    g_ = nb[k]
                    if g_ > f and float(h.equations[f, :3] @ h.equations[g_, :3]) < crease:
                        a_, b_ = [s_[j] for j in range(3) if j != k]
                        E.add((int(min(a_, b_)), int(max(a_, b_))))
            edges = sorted(E)
        else:
            v = list(h.vertices)
            edges = [(int(v[i]), int(v[(i + 1) % len(v)])) for i in range(len(v))]
        out.append(dict(points=P, simplices=h.simplices, vertices=h.vertices, centre=c, edges=edges))
    return out


def region_fields(ax, cm: dict, dims: Sequence[int], core: float = 0.9, alpha: float = 0.16) -> None:
    """Translucent polytopes per region with rigid borders (2-D polygons, 3-D hulls)."""
    cols = _region_colours(len(cm["region_names"]))
    hulls = region_hulls(cm, dims, core)
    xy = np.asarray(cm["xy"])[:, list(dims)]
    if _is3d(ax):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for r, h in enumerate(hulls):
            if h is None:
                continue
            tri_ = [h["points"][s] for s in h["simplices"]]
            ax.add_collection3d(Poly3DCollection(tri_, facecolors=cols[r], alpha=alpha, edgecolors="none"))
            from mpl_toolkits.mplot3d.art3d import Line3DCollection  # rigid border: the hull's feature edges
            segs = [h["points"][[a, b]] for a, b in h["edges"]]
            ax.add_collection3d(Line3DCollection(segs, colors=[cols[r]], linewidths=0.9, alpha=0.85))
        lo, hi = xy.min(0), xy.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    else:
        from matplotlib.patches import Polygon
        for r, h in enumerate(hulls):
            if h is None:
                continue
            V = h["points"][h["vertices"]]
            ax.add_patch(Polygon(V, closed=True, fc=cols[r], alpha=alpha + 0.1, ec="none", zorder=1))
            ax.add_patch(Polygon(V, closed=True, fill=False, ec=cols[r], lw=1.6, zorder=2))
        pad = 0.04 * np.ptp(xy, 0)
        ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
        ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])


def _short(name: str, n: int) -> str:
    s = str(name)
    return s if len(s) <= n else s[: n - 1].rstrip(" /") + "…"


def _ends(info: dict, n: int = 3) -> Tuple[str, str]:
    """("low end", "high end") text of one axis from concept_axes()/worth_axes()."""
    def side(items, regs):
        t = ", ".join(x[:22] for x in items[:n])
        if regs:
            t += (" | " if t else "") + "regions " + ",".join(str(r) for r in regs[:2])
        return t
    return side(info.get("negative", []), info.get("regions_low", [])), \
        side(info.get("positive", []), info.get("regions_high", []))


def _end_items(info: dict) -> Tuple[str, str]:
    """The single item at each end of an axis, for 3-D axis labels."""
    lo, hi = info.get("negative", []), info.get("positive", [])
    return (_short(lo[0], 20) if lo else ""), (_short(hi[0], 20) if hi else "")


def _style_axes(ax, axes: str, labels: Sequence[str], info: Optional[List[dict]], dims: Sequence[int]) -> None:
    """Axis display: "off", "on" or "explain" (see plot_concept_map). In 2-D the
    explanation goes under each axis label; in 3-D (three labels) into a block
    of text under the volume, one line per axis end."""
    if axes not in ("off", "on", "explain"):
        raise ValueError("axes is 'off', 'on' or 'explain'")
    d3 = _is3d(ax)
    setters = [ax.set_xlabel, ax.set_ylabel] + ([ax.set_zlabel] if d3 else [])
    if axes == "off":
        if d3:
            ax.set_axis_off()
        else:
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            for f, lab in zip(setters, labels):
                f(lab, fontsize=8)
        return
    ax.tick_params(labelsize=6 if d3 else 7, colors=_T["muted"])
    explain = axes == "explain" and info is not None and max(dims) < len(info)
    if d3:
        labs = list(labels)
        if explain:                                  # one item at each end, under the axis name
            for k, d in enumerate(dims):
                lo, hi = _end_items(info[d])
                labs[k] = f"{labs[k]}\n{lo} ↔ {hi}" if lo or hi else labs[k]
        for f, lab in zip(setters, labs):
            f(lab, fontsize=7.5, labelpad=8, color=_T["fg"])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color(_T["panel"])
            a.pane.set_edgecolor(_T["grid"])
            a._axinfo["grid"].update(color=_T["grid"], linewidth=0.6)
            a.line.set_color(_T["grid"])
            a.set_major_locator(_plt().MaxNLocator(4))
        return
    ax.grid(True, lw=0.5, alpha=0.6)
    ax.axhline(0, color=_T["muted"], lw=0.6, alpha=0.5, zorder=0)
    ax.axvline(0, color=_T["muted"], lw=0.6, alpha=0.5, zorder=0)
    labs = list(labels)
    if explain:
        for k, d in enumerate(dims[:2]):
            lo, hi = _ends(info[d]); labs[k] = f"{labs[k]}\nlow: {lo}\nhigh: {hi}"
    for f, lab in zip(setters, labs):
        f(lab, fontsize=8)


def _axis_label(P: Projection, j: int) -> str:
    lab = P.labels[j] if j < len(P.labels) else f"axis {j + 1}"
    if j < len(P.explained):
        lab += f" ({100 * P.explained[j]:.0f}%)"
    return lab


def plot_item_map(im: dict, ax=None, top: Optional[int] = 40, title: Optional[str] = None):
    """Items as points at their direction in the plane (item_map()); colour = pass."""
    fig, ax = _ax(ax)
    o = np.argsort(-im["strength"])[: top or len(im["items"])]
    for j in o:
        c = _pass_colour(im["pass_"][j])
        m = "o" if im["kind"][j] == "talent" else "s"
        ax.scatter(im["xy"][j, 0], im["xy"][j, 1], c=[c], marker=m, s=24)
    _labels(ax, np.asarray(im["xy"])[:, :2], im["items"], o, len(o))
    ax.axhline(0, c=_T["grid"], lw=0.8); ax.axvline(0, c=_T["grid"], lw=0.8)
    _pass_legend(ax, {im["pass_"][j] for j in o})
    ax.set_title(title or "items: direction of rising worth in the concept plane")
    return fig


def _pass_legend(ax, passes):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=_pass_colour(p), label=p) for p in PASS_COLOURS if p in passes]
    h += [Line2D([], [], marker="o", ls="", color=_T["muted"], label="talent"),
          Line2D([], [], marker="s", ls="", color=_T["muted"], label="mantra")]
    kw = dict(loc="upper left", bbox_to_anchor=(1.0, 0.98)) if _is3d(ax) else dict(loc="best")
    ax.legend(handles=h, fontsize=7, framealpha=0.0 if _is3d(ax) else 0.85, **kw)


def _labels(ax, xy, names, order, limit, min_gap=0.035, taken=()):
    """Label points in `order`, skipping any too close to a label already placed
    (or to the points in `taken`, e.g. highlighted items labelled already)."""
    span = np.ptp(xy, 0) + 1e-12
    placed = [xy[j] / span for j in taken]
    limit += len(placed)
    for j in order:
        if len(placed) >= limit:
            break
        p = xy[j] / span
        if any(np.linalg.norm(p - q) < min_gap for q in placed):
            continue
        placed.append(p)
        _ann(ax, xy[j], _short(names[j], 24), 3, 3, fontsize=6.8, color=_T["fg"], alpha=0.9)


def plot_worth_space(ws: dict, ax=None, label_top: int = 40, title: Optional[str] = None,
                     highlight: Optional[Sequence[str]] = None, dims: Sequence[int] = (0, 1),
                     axes: str = "on", axis_info: Optional[List[dict]] = None,
                     view: Tuple[float, float] = (22, -58)):
    """Items on the worth space (worth_space()); colour = pass, size = mean worth,
    circle = talent, square = mantra. highlight: items drawn outlined and always
    labelled (the others fade), with drop lines in 3-D. dims, axes, axis_info,
    view: as plot_concept_map (axis_info = worth_axes())."""
    dims = tuple(dims)
    fig, ax = _ax_for(ax, dims, (8, 7))
    d3 = _is3d(ax)
    xy = ws["xy"][:, dims]
    hi = set(highlight or ())
    fade = 0.25 if hi else 0.75
    for j, i in enumerate(ws["items"]):
        c = _pass_colour(ws["pass_"][j])
        m = "o" if ws["kind"][j] == "talent" else "s"
        if i in hi:
            _sc(ax, xy[j:j + 1], c=[c], marker=m, s=40 + 160 * ws["mean_worth"][j], alpha=0.95,
                edgecolors=_T["fg"], linewidths=1.2, zorder=3)
        else:
            _sc(ax, xy[j:j + 1], c=[c], marker=m, s=10 + 120 * ws["mean_worth"][j], alpha=fade,
                edgecolors=_T["bg"], linewidths=0.4)
    spread = np.linalg.norm(xy - np.median(xy, 0), axis=1)
    if hi:
        idx = [j for j, i in enumerate(ws["items"]) if i in hi]
        import matplotlib.patheffects as pe
        _stems(ax, xy[idx])
        for j in idx:
            _ann(ax, xy[j], ws["items"][j], 5, 4, fontsize=8, fontweight="bold", zorder=5, color=_T["fg"],
                 path_effects=[pe.withStroke(linewidth=2.5, foreground=_T["bg"])])
        rest = [j for j in np.argsort(-spread) if ws["items"][j] not in hi]
        _labels(ax, xy, ws["items"], rest, max(0, label_top - len(idx)) // 2, min_gap=0.13, taken=idx)
    else:
        _labels(ax, xy, ws["items"], np.argsort(-spread), label_top)
    _pass_legend(ax, set(ws["pass_"]))
    ex = ws["explained"]
    lab = lambda j: f"worth axis {j + 1}" + (f" ({100 * ex[j]:.0f}%)" if j < len(ex) else "")
    _style_axes(ax, axes, [lab(d) for d in dims], axis_info, dims)
    if d3:
        ax.view_init(*view)
    kind = "relative worth" if ws.get("relative") else "worth"
    kept = view_kept(ws["distance"], ws["xy"], dims, distances=True)
    ax.set_title(title or f"{kind} space · axes {'-'.join(str(d + 1) for d in dims)} · distance kept {kept:.2f}",
                 fontsize=10, loc="left")
    return fig


# ------------------------------------------------------------------ motion and interactive export
def spin_gif(fig, frames: int = 24, dpi: int = 72, duration_ms: int = 110, elevation: Optional[float] = None,
             close: bool = True) -> bytes:
    """A 3-D figure turned once around (360 degrees over `frames` frames) as an
    animated GIF; Discord and most viewers play it inline. Raises ValueError for
    a figure without 3-D axes."""
    import io

    from PIL import Image
    axs = [a for a in fig.axes if _is3d(a)]
    if not axs:
        raise ValueError("spinning needs a 3-D view (per_view 3)")
    ax = axs[0]
    elev, az0 = (ax.elev if elevation is None else elevation), ax.azim
    imgs = []
    for f in range(max(2, int(frames))):
        ax.view_init(elev, az0 + 360.0 * f / max(2, int(frames)))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        if not imgs:                                   # one palette for every frame: no flicker
            pal = im.quantize(colors=255, method=Image.Quantize.FASTOCTREE)
        imgs.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0,
                 optimize=True, disposal=1)
    if close:
        _plt().close(fig)
    return out.getvalue()


def _hex(c) -> str:
    from matplotlib.colors import to_hex
    return to_hex(c)


def scatter_concept(cm: dict, dims: Sequence[int] = (0, 1, 2), highlight: Optional[dict] = None,
                    locators: Optional[dict] = None, axis_info: Optional[List[dict]] = None,
                    color: Optional[np.ndarray] = None, color_label: str = "", fields: bool = False) -> dict:
    """concept_map() as plain data for html_scatter(): points (x, y, z, colour,
    size, name, group, region) plus axis labels, a colour legend and the
    region polytopes (hulls; drawn when fields, toggled in the page)."""
    dims = tuple(dims)[:3]
    xy = cm["xy"][:, dims]
    names = cm["region_names"]
    cols = [_hex(c) for c in _region_colours(len(names))]
    if color is None:
        pc = [cols[r] for r in cm["region"]]
        legend = [(f"{r} {_short(n, 40)}", cols[r]) for r, n in enumerate(names)]
    else:
        import matplotlib as mpl
        v = np.asarray(color, float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        cmap = _cmap("seq")
        cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        pc = [_hex(cmap(0.0 if hi <= lo else (x - lo) / (hi - lo))) for x in v]
        legend = [(f"{color_label} low {lo:.2f}", _hex(cmap(0.0))), (f"{color_label} high {hi:.2f}", _hex(cmap(1.0)))]
    ids = _ids_of(cm)
    pts = [dict(p=[round(float(a), 4) for a in xy[i]], c=pc[i], s=2.2,
                n=(ids[i] or "build") + f" · region {cm['region'][i]}", g="build", r=int(cm["region"][i]))
           for i in range(len(xy))]
    for r, p in enumerate(cm["centre_xy"][:, dims]):
        pts.append(dict(p=[float(a) for a in p], c=cols[r], s=7, n=f"region {r}: {names[r]}", g="centre", t=str(r), r=r))
    if locators is not None and len(locators["xy"]):
        for p, lab in zip(locators["xy"][:, dims], locators.get("labels", locators.get("ids", []))):
            pts.append(dict(p=[float(a) for a in p], c=_T["fg"], s=5, n=f"locator: {lab}", g="locator", t=str(lab)[:30]))
    if highlight is not None:
        for p, i in zip(highlight["xy"][:, dims], highlight["ids"]):
            pts.append(dict(p=[float(a) for a in p], c=_T["star"], s=7, n=f"build: {i}", g="mark", t=str(i or "")))
    P = cm["projection"]
    return dict(title=f"concept space · axes {'-'.join(str(d + 1) for d in dims)}", points=pts,
                axes=[_html_axis(_axis_label(P, d), axis_info, d) for d in dims], legend=legend,
                hulls=html_hulls(cm, dims, cols), fields=bool(fields))


def html_hulls(cm: dict, dims: Sequence[int], cols: Sequence[str]) -> List[dict]:
    """region_hulls() as page data: vertices v, triangles f, border edges e."""
    out = []
    for r, h in enumerate(region_hulls(cm, dims)):
        if h is None:
            continue
        used = sorted({int(i) for s_ in h["simplices"] for i in s_} | {int(i) for e in h["edges"] for i in e})
        if len(dims) == 2:
            used = [int(i) for i in h["vertices"]]
        idx = {i: k for k, i in enumerate(used)}
        V = [[round(float(a), 4) for a in h["points"][i]] for i in used]
        if len(dims) == 2:
            F = [[0, k, k + 1] for k in range(1, len(used) - 1)]
        else:
            F = [[idx[int(i)] for i in s_] for s_ in h["simplices"]]
        out.append(dict(r=r, c=cols[r], v=V, f=F, e=[[idx[a], idx[b]] for a, b in h["edges"]]))
    return out


def _ids_of(cm: dict) -> List[Optional[str]]:
    ids = cm.get("ids")
    return list(ids) if ids is not None else [None] * len(cm["xy"])


def _html_axis(label: str, info: Optional[List[dict]], d: int) -> str:
    if info is not None and d < len(info):
        lo, hi = _end_items(info[d])
        if lo or hi:
            return f"{label}: {lo} ↔ {hi}"
    return label


def scatter_worth(ws: dict, dims: Sequence[int] = (0, 1, 2), highlight: Sequence[str] = (),
                  axis_info: Optional[List[dict]] = None) -> dict:
    """worth_space() as plain data for html_scatter()."""
    dims = tuple(dims)[:3]
    xy = ws["xy"][:, dims]
    hi = set(highlight or ())
    pts = []
    for j, i in enumerate(ws["items"]):
        w = float(ws["mean_worth"][j])
        pts.append(dict(p=[float(a) for a in xy[j]], c=_pass_colour(ws["pass_"][j]), s=2.5 + 9 * w,
                        n=f"{i} · {ws['kind'][j]} · {ws['pass_'][j]} · mean worth {w:.2f}",
                        g="mark" if i in hi else ("talent" if ws["kind"][j] == "talent" else "mantra"),
                        t=i if i in hi else ""))
    ex = ws["explained"]
    lab = lambda j: f"worth axis {j + 1}" + (f" ({100 * ex[j]:.0f}%)" if j < len(ex) else "")
    passes = [p for p in PASS_COLOURS if p in set(ws["pass_"])]
    return dict(title=f"worth space · axes {'-'.join(str(d + 1) for d in dims)}", points=pts,
                axes=[_html_axis(lab(d), axis_info, d) for d in dims], legend=[(p, _pass_colour(p)) for p in passes])


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;background:__BG__;color:__FG__;font:13px system-ui,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh;cursor:grab}#c.sel{cursor:crosshair}
.pn{position:fixed;background:__PANEL__e6;border:1px solid __GRID__;border-radius:6px;padding:7px 9px}
#hud{left:10px;top:10px;max-width:340px}#hud b{font-size:14px}#hud .m{color:__MUTED__;font-size:11.5px}
#hud button,#hud select{background:__BG__;color:__FG__;border:1px solid __GRID__;border-radius:4px;font:12px system-ui;padding:2px 6px;margin:2px 2px 0 0;cursor:pointer}
#hud button.on{border-color:__FG__}#hud input[type=range]{width:90px;vertical-align:middle}
#hud input[type=text]{background:__BG__;color:__FG__;border:1px solid __GRID__;border-radius:4px;width:150px;padding:2px 5px}
#leg{right:10px;top:10px;max-height:58vh;overflow:auto}#leg div{white-space:nowrap;font-size:12px;cursor:pointer;user-select:none}
#leg div.off{opacity:.35}#leg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#sel{right:10px;bottom:10px;max-width:360px;max-height:36vh;overflow:auto;display:none;font-size:12px}
#ann{left:10px;bottom:46px;max-width:340px;max-height:26vh;overflow:auto;display:none;font-size:12px}
#ann div{cursor:pointer}#ax{position:fixed;left:12px;bottom:10px;color:__MUTED__;font-size:11.5px;pointer-events:none}
#tip{position:fixed;pointer-events:none;display:none;max-width:380px;font-size:12px}
#box{position:fixed;border:1px dashed __FG__;background:__FG__11;display:none;pointer-events:none}
.bar{display:inline-block;height:8px;border-radius:2px;vertical-align:middle;margin-right:5px}</style></head><body>
<canvas id="c"></canvas>
<div id="hud" class="pn"><b>__TITLE__</b><br><span class="m">drag: turn · shift-drag or Select: box-select · click: annotate (Note mode) · wheel: zoom · double-click: reset</span><br>
<button id="bSel">Select</button><button id="bNote">Note</button><button id="bIso">Isolate</button><button id="bClr">Clear</button>
<button id="bSpin">Spin</button><button id="bLab" class="on">Labels</button><button id="bFld">Fields</button><button id="bPer" class="on">Perspective</button><br>
<button data-v="0,0">Front</button><button data-v="1.5707,0">Top</button><button data-v="0,1.5707">Side</button><button data-v="0.38,-0.9">Default</button>
<button id="bPng">PNG</button><button id="bJson">Export</button><button id="bImp">Import</button><br>
<span class="m">size</span> <input type="range" id="rS" min="0.3" max="3" step="0.1" value="1"> <span class="m">alpha</span> <input type="range" id="rA" min="0.05" max="1" step="0.05" value="0.55"><br>
<input type="text" id="q" placeholder="find (name, region, locator)"></div>
<div id="leg" class="pn"></div><div id="sel" class="pn"></div><div id="ann" class="pn"></div><div id="ax"></div><div id="tip" class="pn"></div><div id="box"></div>
<input type="file" id="fi" accept=".json" style="display:none">
<script>
const D=__DATA__;const $=i=>document.getElementById(i);const cv=$('c'),g=cv.getContext('2d'),tip=$('tip');
const AXC=['#ff6b6b','#51cf66','#4dabf7'],esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const RN=D.regions||[];const hidden=new Set();let sel=new Set(),iso=false,found=new Set();
let P=D.points.map(o=>Object.assign({},o));for(const o of P){while(o.p.length<3)o.p.push(0)}
let m=[0,0,0],r=1e-9;for(let k=0;k<3;k++){let a=Infinity,b=-Infinity;for(const o of P){a=Math.min(a,o.p[k]);b=Math.max(b,o.p[k])}if(a<b){m[k]=(a+b)/2;r=Math.max(r,(b-a)/2)}else m[k]=a}
const nq=p=>{const v=p.slice();while(v.length<3)v.push(0);return v.map((x,k)=>(x-m[k])/r)};for(const o of P)o.q=nq(o.p);
const H=(D.hulls||[]).map(h=>({r:h.r,c:h.c,v:h.v.map(nq),f:h.f,e:h.e||[]}));
let az=-0.9,el=0.38,zoom=1,drag=null,hover=-1,mode='turn',persp=true,spin=false,lab=true,fld=D.fields&&H.length>0,sz=1,al=0.55;
const KEY='tri-ann:'+D.title;let ANN=[];try{ANN=JSON.parse(localStorage.getItem(KEY)||'[]')}catch(e){}
function saveA(){try{localStorage.setItem(KEY,JSON.stringify(ANN))}catch(e){}renderA()}
function size(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio}addEventListener('resize',()=>{size();draw()});size();
function proj(q){const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
 const x=ca*q[0]-sa*q[1],y0=sa*q[0]+ca*q[1];const y=ce*q[2]-se*y0,z=se*q[2]+ce*y0;
 const s=Math.min(cv.width,cv.height)*0.40*zoom,f=persp?3.2/(3.2+z):1;return[cv.width/2+x*s*f,cv.height/2-y*s*f,z,f]}
const vis=o=>!(o.r!=null&&hidden.has(o.r))&&!(iso&&sel.size&&!sel.has(o.i));P.forEach((o,i)=>o.i=i);
function draw(){g.clearRect(0,0,cv.width,cv.height);const dp=devicePixelRatio;g.lineWidth=dp;
 for(let k=0;k<3;k++){const e=[0,0,0];e[k]=1.15;const a=proj(e.map(v=>-v)),b=proj(e);g.strokeStyle=AXC[k]+'88';g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();g.fillStyle=AXC[k];g.font=`${12*dp}px system-ui`;g.fillText('xyz'[k],b[0]+4*dp,b[1])}
 if(fld){const T=[];for(const h of H){if(hidden.has(h.r))continue;const V=h.v.map(proj);for(const f of h.f){T.push([(V[f[0]][2]+V[f[1]][2]+V[f[2]][2])/3,V,f,h.c])}}
  T.sort((a,b)=>b[0]-a[0]);for(const[,V,f,c]of T){g.globalAlpha=0.13;g.fillStyle=c;g.beginPath();g.moveTo(V[f[0]][0],V[f[0]][1]);g.lineTo(V[f[1]][0],V[f[1]][1]);g.lineTo(V[f[2]][0],V[f[2]][1]);g.closePath();g.fill()}
  g.globalAlpha=0.9;g.lineWidth=1.3*dp;for(const h of H){if(hidden.has(h.r))continue;const V=h.v.map(proj);g.strokeStyle=h.c;g.beginPath();for(const[a,b]of h.e){g.moveTo(V[a][0],V[a][1]);g.lineTo(V[b][0],V[b][1])}g.stroke()}}
 const S=[];P.forEach((o,i)=>{if(vis(o)&&!(fld&&o.g==='build'&&!sel.has(i)&&!found.has(i)))S.push([proj(o.q),i])});S.sort((a,b)=>b[0][2]-a[0][2]);
 for(const[[x,y,z,f],i]of S){const o=P[i];const big=o.g!=='build';const rad=o.s*dp*f*(big?1:sz);g.globalAlpha=big?0.95:(sel.size&&!sel.has(i)?al*0.25:al);g.fillStyle=o.c;
  g.beginPath();if(o.g==='locator'){g.moveTo(x,y-rad);g.lineTo(x+rad,y);g.lineTo(x,y+rad);g.lineTo(x-rad,y);g.closePath()}else if(o.g==='mantra'){g.rect(x-rad,y-rad,2*rad,2*rad)}else g.arc(x,y,rad,0,7);g.fill();
  if(big||i===hover||sel.has(i)||found.has(i)){g.globalAlpha=1;g.strokeStyle=found.has(i)?'#ffd43b':(sel.has(i)&&!big?'__FG__':'__BG__');g.lineWidth=1.4*dp;g.stroke()}
  if(lab&&o.t){g.globalAlpha=1;g.fillStyle='__FG__';g.font=`${11*dp}px system-ui`;g.fillText(o.t,x+rad+3*dp,y+4*dp)}}
 g.globalAlpha=1;ANN.forEach((a,k)=>{const[x,y]=proj(nq(a.p));g.fillStyle='#ffd43b';g.beginPath();g.arc(x,y,4*dp,0,7);g.fill();g.font=`bold ${11.5*dp}px system-ui`;
  const w=g.measureText(a.t).width+8*dp;g.fillStyle='__PANEL__';g.fillRect(x+7*dp,y-20*dp,w,16*dp);g.strokeStyle='#ffd43b';g.strokeRect(x+7*dp,y-20*dp,w,16*dp);g.fillStyle='#ffd43b';g.fillText(a.t,x+11*dp,y-8*dp)});
 $('ax').innerHTML=D.axes.map((a,i)=>`<span style="color:${AXC[i]}">${'xyz'[i]}</span> ${esc(a)}`).join('<br>')}
function pick(ex,ey){const dp=devicePixelRatio,X=ex*dp,Y=ey*dp;let best=-1,bd=1e9;P.forEach((o,i)=>{if(!vis(o))return;const[x,y]=proj(o.q);const d=(x-X)**2+(y-Y)**2;if(d<bd){bd=d;best=i}});return bd<(9*dp)**2?best:-1}
function legend(){$('leg').innerHTML=(D.legend||[]).map(([n,c],k)=>`<div data-r="${k}" class="${hidden.has(k)?'off':''}"><i style="background:${c}"></i>${esc(n)}</div>`).join('')+'<div class="m" style="color:__MUTED__;cursor:default">click: hide · alt-click: solo</div>';
 for(const el_ of $('leg').querySelectorAll('div[data-r]'))el_.onclick=e=>{const k=+el_.dataset.r;if(e.altKey){const all=(D.legend||[]).map((_,j)=>j);const solo=hidden.size===all.length-1&&!hidden.has(k);hidden.clear();if(!solo)all.forEach(j=>{if(j!==k)hidden.add(j)})}else hidden.has(k)?hidden.delete(k):hidden.add(k);legend();draw()}}
function summary(){const el_=$('sel');if(!sel.size){el_.style.display='none';return}const cnt={};let n=0,c=[0,0,0];const named=[];
 for(const i of sel){const o=P[i];if(o.g==='build'){n++;cnt[o.r]=(cnt[o.r]||0)+1;o.p.forEach((v,k)=>c[k]+=v)}else named.push(o.n)}
 const rows=Object.entries(cnt).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v])=>{const L=(D.legend||[])[k]||['region '+k,'#888'];return `<div><span class="bar" style="width:${Math.max(3,120*v/n)}px;background:${L[1]}"></span>${(100*v/n).toFixed(0)}% ${esc(L[0])}</div>`}).join('');
 el_.innerHTML=`<b>selection</b> · ${n} builds${named.length?' · '+named.length+' marked':''}<br>${rows}${named.slice(0,8).map(x=>'<div>'+esc(x)+'</div>').join('')}<div style="color:__MUTED__">centre (${c.map(v=>(v/Math.max(n,1)).toFixed(2)).join(', ')})</div>`;el_.style.display='block'}
function renderA(){const el_=$('ann');if(!ANN.length){el_.style.display='none';return}el_.innerHTML='<b>notes</b> <span style="color:__MUTED__">(click: remove)</span>'+ANN.map((a,k)=>`<div data-k="${k}">${k+1}. ${esc(a.t)}${a.on?' · '+esc(a.on):''}</div>`).join('');el_.style.display='block';
 for(const d of el_.querySelectorAll('div[data-k]'))d.onclick=()=>{ANN.splice(+d.dataset.k,1);saveA();draw()}}
function setMode(x){mode=mode===x?'turn':x;$('bSel').classList.toggle('on',mode==='sel');$('bNote').classList.toggle('on',mode==='note');cv.classList.toggle('sel',mode!=='turn')}
cv.onmousedown=e=>{if(mode==='sel'||e.shiftKey){drag={box:[e.clientX,e.clientY],add:e.ctrlKey||e.metaKey}}else drag={turn:[e.clientX,e.clientY,az,el],moved:false}};
addEventListener('mouseup',e=>{if(drag&&drag.box){const[x0,y0]=drag.box,x1=e.clientX,y1=e.clientY,dp=devicePixelRatio;$('box').style.display='none';
 if(Math.abs(x1-x0)>3&&Math.abs(y1-y0)>3){if(!drag.add)sel=new Set();const[a,b,c,d]=[Math.min(x0,x1)*dp,Math.max(x0,x1)*dp,Math.min(y0,y1)*dp,Math.max(y0,y1)*dp];
  P.forEach((o,i)=>{if(!vis(o))return;const[x,y]=proj(o.q);if(x>=a&&x<=b&&y>=c&&y<=d)sel.add(i)});summary();draw()}}
 else if(drag&&drag.turn&&!drag.moved&&mode==='note'){const h=pick(e.clientX,e.clientY);let p;if(h>=0)p=P[h].p.slice();else{p=[m[0],m[1],m[2]]}
  const t=prompt('note'+(h>=0?' on '+P[h].n:''));if(t){ANN.push({t,p,on:h>=0?P[h].n:''});saveA();draw()}}drag=null});
cv.onmousemove=e=>{if(drag&&drag.box){const[x0,y0]=drag.box,b=$('box');b.style.display='block';b.style.left=Math.min(x0,e.clientX)+'px';b.style.top=Math.min(y0,e.clientY)+'px';b.style.width=Math.abs(e.clientX-x0)+'px';b.style.height=Math.abs(e.clientY-y0)+'px';return}
 if(drag&&drag.turn){const t=drag.turn;if(Math.abs(e.clientX-t[0])+Math.abs(e.clientY-t[1])>3)drag.moved=true;if(drag.moved){az=t[2]+(e.clientX-t[0])*0.008;el=Math.max(-1.55,Math.min(1.55,t[3]+(e.clientY-t[1])*0.008));tip.style.display='none';draw()}return}
 const h=pick(e.clientX,e.clientY);if(h!==hover){hover=h;draw()}if(h>=0){tip.textContent=P[h].n;tip.style.display='block';tip.style.left=e.clientX+14+'px';tip.style.top=e.clientY+10+'px'}else tip.style.display='none'};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(0.3,Math.min(12,zoom*Math.exp(-e.deltaY*0.0015)));draw()};
cv.ondblclick=()=>{az=-0.9;el=0.38;zoom=1;draw()};
let t0=null;cv.addEventListener('touchstart',e=>{const t=e.touches[0];t0=[t.clientX,t.clientY,az,el]},{passive:true});
cv.addEventListener('touchmove',e=>{const t=e.touches[0];az=t0[2]+(t.clientX-t0[0])*0.008;el=Math.max(-1.55,Math.min(1.55,t0[3]+(t.clientY-t0[1])*0.008));draw()},{passive:true});
$('bSel').onclick=()=>setMode('sel');$('bNote').onclick=()=>setMode('note');
$('bIso').onclick=()=>{iso=!iso;$('bIso').classList.toggle('on',iso);draw()};$('bClr').onclick=()=>{sel=new Set();iso=false;$('bIso').classList.remove('on');summary();draw()};
$('bSpin').onclick=()=>{spin=!spin;$('bSpin').classList.toggle('on',spin)};$('bLab').onclick=()=>{lab=!lab;$('bLab').classList.toggle('on',lab);draw()};
$('bFld').onclick=()=>{fld=!fld&&H.length>0;$('bFld').classList.toggle('on',fld);draw()};$('bFld').classList.toggle('on',fld);if(!H.length)$('bFld').disabled=true;
$('bPer').onclick=()=>{persp=!persp;$('bPer').classList.toggle('on',persp);draw()};
for(const b of document.querySelectorAll('button[data-v]'))b.onclick=()=>{const[e_,a_]=b.dataset.v.split(',').map(Number);el=e_;az=a_;draw()};
$('rS').oninput=e=>{sz=+e.target.value;draw()};$('rA').oninput=e=>{al=+e.target.value;draw()};
$('q').oninput=e=>{const s=e.target.value.trim().toLowerCase();found=new Set();if(s.length>1)P.forEach((o,i)=>{if((o.n||'').toLowerCase().includes(s)&&o.g!=='build'||(o.g==='build'&&(o.n||'').toLowerCase()===s))found.add(i)});draw()};
$('bPng').onclick=()=>{const a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download=(D.title||'plot').replace(/\W+/g,'_')+'.png';a.click()};
$('bJson').onclick=()=>{const out={title:D.title,notes:ANN,selection:[...sel].map(i=>({name:P[i].n,p:P[i].p,region:P[i].r})),view:{az,el,zoom}};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download=(D.title||'plot').replace(/\W+/g,'_')+'_notes.json';a.click()};
$('bImp').onclick=()=>$('fi').click();$('fi').onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{try{const o=JSON.parse(t);ANN=o.notes||[];if(o.view){az=o.view.az;el=o.view.el;zoom=o.view.zoom}saveA();draw()}catch(_){alert('not a notes file')}})};
legend();renderA();(function tick(){if(spin&&!drag){az+=0.01;draw()}requestAnimationFrame(tick)})();draw();
</script></body></html>"""


def html_scatter(data: dict, theme_name: Optional[str] = None) -> str:
    """A self-contained HTML page (no network, no libraries) drawing
    scatter_concept() / scatter_worth() data as a 3-D scatter: drag to turn,
    wheel to zoom, hover for names, a spin toggle. Opens in any browser."""
    import html as _h
    import json
    t = THEMES[theme_name] if theme_name else _T
    js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    out = _HTML
    for k, v in (("__TITLE__", _h.escape(str(data.get("title", "plot")))), ("__BG__", t["bg"]), ("__FG__", t["fg"]),
                 ("__PANEL__", t["panel"]), ("__GRID__", t["grid"]), ("__MUTED__", t["muted"])):
        out = out.replace(k, v)
    return out.replace("__DATA__", js)


# ------------------------------------------------------------------ views on several axes
def axis_groups(k: int, spec: Optional[str] = None, size: int = 3) -> List[Tuple[int, ...]]:
    """0-based axis groups for views of k axes, `size` (2 or 3) axes per view.
    spec None: consecutive groups 1-2-3, 4-5-6, ... (a last partial group is
    filled with the axes before it, so every view has `size` axes); otherwise
    1-based groups such as "1-2-3 1-4-5" or "1-2 3-4" (2 or 3 axes each)."""
    import re
    if size not in (2, 3):
        raise ValueError("views have 2 or 3 axes")
    if k < size:
        raise ValueError(f"need at least {size} axes")
    if not spec or not spec.strip():
        out = [tuple(range(j, j + size)) for j in range(0, k - size + 1, size)]
        if k % size:
            out.append(tuple(range(k - size, k)))
        return out
    out = []
    for tok in re.split(r"[;\s]+", re.sub(r"\s*([-,/x:])\s*", r"\1", spec.strip())):
        if not tok:
            continue
        parts = re.split(r"[-,/x:]", tok)
        if len(parts) not in (2, 3) or not all(p_.isdigit() for p_ in parts):
            raise ValueError(f"axis group {tok!r}: write groups like 1-2-3 or 1-3")
        g = tuple(int(p_) - 1 for p_ in parts)
        if len(set(g)) != len(g) or min(g) < 0 or max(g) >= k:
            raise ValueError(f"axis group {tok!r}: different axes from 1 to {k}")
        if g not in out:
            out.append(g)
    return out


def axis_pairs(k: int, spec: Optional[str] = None) -> List[Tuple[int, ...]]:
    """axis_groups with two axes per view (planes)."""
    return axis_groups(k, spec, 2)


def concept_axes(tri, cm: dict, top: int = 4) -> List[dict]:
    """What each projection axis of concept_map() means: the items whose logits
    rise most (positive) and fall most (negative) along it, and the archetype
    regions whose centres sit at its high and low ends."""
    P = cm["projection"]
    L = tri.space.item_coords() @ P.W
    ids = tri.catalog.ids
    cx = cm["centre_xy"]
    out = []
    for j in range(P.W.shape[1]):
        o = np.argsort(L[:, j])
        ro = list(np.argsort(cx[:, j])) if len(cx) else []
        out.append(dict(axis=j, label=_axis_label(P, j),
                        explained=float(P.explained[j]) if j < len(P.explained) else float("nan"),
                        positive=[ids[i] for i in o[::-1][:top]], negative=[ids[i] for i in o[:top]],
                        regions_high=[int(r) for r in ro[::-1][:2]], regions_low=[int(r) for r in ro[:2]]))
    return out


def worth_axes(ws: dict, top: int = 4) -> List[dict]:
    """What each worth-space axis means: the items at its two ends, and the
    archetype regions whose builds push items towards each end (region_pull)."""
    xy, its = ws["xy"], ws["items"]
    pull = ws.get("region_pull")
    out = []
    for j in range(xy.shape[1]):
        o = np.argsort(xy[:, j])
        d = dict(axis=j, label=f"worth axis {j + 1}",
                 explained=float(ws["explained"][j]) if j < len(ws["explained"]) else float("nan"),
                 positive=[its[i] for i in o[::-1][:top]], negative=[its[i] for i in o[:top]],
                 regions_high=[], regions_low=[])
        if pull is not None and j < len(pull) and pull.shape[1]:
            ro = np.argsort(pull[j])
            d["regions_high"] = [int(r) for r in ro[::-1][:2] if pull[j, r] > 0]
            d["regions_low"] = [int(r) for r in ro[:2] if pull[j, r] < 0]
        out.append(d)
    return out


def view_kept(U_or_D: np.ndarray, xy: np.ndarray, dims: Sequence[int], distances: bool = False) -> float:
    """Distance kept (correlation of true and drawn distances) by one view on the
    axes `dims`. U_or_D: concept coordinates [b, D'] or, with distances=True, a
    distance matrix."""
    v = xy[:, list(dims)]
    if not distances:
        return _distance_kept(U_or_D, v)
    iu = np.triu_indices(len(v), 1)
    e = np.linalg.norm(v[:, None] - v[None], axis=2)
    return float(np.corrcoef(U_or_D[iu], e[iu])[0, 1]) if len(v) > 2 else float("nan")


def concept_colour(tri, cm: dict, item: Optional[str] = None, stat: Optional[str] = None) -> Tuple[np.ndarray, str]:
    """Per-build colour values for concept_map(): the field worth of `item` (0 where
    the build cannot take it) or the final value of `stat`."""
    bm = cm.get("bm")
    if bm is None:
        raise ValueError("concept_map() output without its build matrix")
    if item is not None:
        if item not in tri.catalog.index:
            raise ValueError(f"unknown item {item!r}")
        c = tri.catalog.index[item]
        p = sigmoid(tri.space.logits_rows(bm)[:, c]) * np.asarray(bm.masks[bm.gid][:, c], float)
        return p, f"worth of {item}"
    if stat is not None:
        if stat not in tri.catalog.stats:
            raise ValueError(f"unknown stat {stat!r}; one of {', '.join(tri.catalog.stats)}")
        return np.asarray(bm.fi[:, list(tri.catalog.stats).index(stat)], float), f"final {stat}"
    raise ValueError("give item or stat")


def worth_neighbours(ws: dict, item: str, n: int = 5, by: str = "correlation") -> List[Tuple[str, float, float]]:
    """[(item, worth distance, correlation)]: the n items valued most like `item`.
    by "correlation": valued by the same builds (highest correlation of worth
    across builds, whatever the pick rate); "distance": nearest in the worth
    space over all axes (similar worth, which also matches pick rate)."""
    if item not in ws["items"]:
        return []
    j = ws["items"].index(item)
    key = -np.nan_to_num(ws["correlation"][j], nan=-2.0) if by == "correlation" else ws["distance"][j]
    o = [k for k in np.argsort(key) if k != j][:n]
    return [(ws["items"][k], float(ws["distance"][j, k]), float(ws["correlation"][j, k])) for k in o]


def plot_axis_summary(info: List[dict], views: Sequence[Tuple[Tuple[int, ...], float]], total_kept: float,
                      title: str = "axes", region_names: Sequence[str] = ()):
    """Per axis: its share of variance (bar, cumulative line) and the items at
    its two ends, as a diverging strip; the title says how much of the true
    distances each drawn view keeps."""
    plt = _plt()
    k = len(info)
    fig = plt.figure(figsize=(9.5, 2.6 + 0.36 * k))
    g = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.04)
    ax = fig.add_subplot(g[0])
    ex = np.array([0.0 if np.isnan(d["explained"]) else d["explained"] for d in info])
    y = np.arange(k)
    ax.barh(y, 100 * ex, color=_pass_colour("foundational"), alpha=0.9, height=0.62)
    ax.plot(100 * np.cumsum(ex), y, "o-", color=_pass_colour("central"), ms=3.5, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([f"axis {d['axis'] + 1}" for d in info], fontsize=7.5)
    ax.set_ylim(k - 0.5, -1.1); ax.set_xlim(0, 100); ax.set_xlabel("% of variance (line: cumulative)", fontsize=7.5)
    ax.tick_params(labelsize=7); ax.grid(True, axis="x", alpha=0.6); ax.grid(False, axis="y")
    tx = fig.add_subplot(g[1], sharey=ax); tx.axis("off")
    for i, d in enumerate(info):
        lo, hi = _ends(d, 3)
        tx.text(0.48, i, lo, ha="right", va="center", fontsize=7, color=_T["fg"])
        tx.text(0.5, i, "↔", ha="center", va="center", fontsize=8, color=_T["muted"])
        tx.text(0.52, i, hi, ha="left", va="center", fontsize=7, color=_T["fg"])
    tx.set_xlim(0, 1)
    tx.text(0.48, -0.85, "low end", ha="right", fontsize=7, color=_T["muted"])
    tx.text(0.52, -0.85, "high end", ha="left", fontsize=7, color=_T["muted"])
    vtxt = " · ".join(f"{'-'.join(str(d + 1) for d in v)} {r:.2f}" for v, r in views)
    fig.suptitle(f"{title} · distance kept: {vtxt} · all {k} axes {total_kept:.2f}", fontsize=9.5,
                 x=0.02, ha="left", color=_T["fg"], weight="bold")
    return fig


def plot_region_worth(rw: dict, ax=None, title: Optional[str] = None):
    """Heatmap regions x items of mean worth (region_worth())."""
    n, R = len(rw["items"]), len(rw["regions"])
    fig, ax = _ax(ax, (max(8, 0.28 * n + 3), max(3, 0.38 * R + 2)))
    im = ax.imshow(rw["M"], aspect="auto", cmap=_cmap("heat"), vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(rw["items"], rotation=90, fontsize=7)
    ax.set_yticks(range(R))
    ax.set_yticklabels([f"{r}: {nm[:30]} ({100 * s:.0f}%)" for r, (nm, s) in enumerate(zip(rw["regions"], rw["share"]))],
                       fontsize=7)
    fig.colorbar(im, ax=ax, label="mean worth")
    ax.set_title(title or "mean worth by archetype region (blank = no build there can take it)")
    fig.tight_layout()
    return fig


def plot_worth_profile(wp: dict, ax=None, top: int = 40, title: Optional[str] = None):
    """Bars of one build's worth (worth_profile()); picked items solid, others hatched."""
    n = min(top, len(wp["items"]))
    fig, ax = _ax(ax, (8, max(4, 0.22 * n + 1)))
    y = np.arange(n)[::-1]
    for j in range(n):
        c = _pass_colour(wp["pass_"][j])
        ax.barh(y[j], wp["worth"][j], color=c if wp["picked"][j] else _T["panel"], edgecolor=c,
                hatch=None if wp["picked"][j] else "///")
    ax.set_yticks(y); ax.set_yticklabels(wp["items"][:n], fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel("worth (field probability; own picks leave-one-out)")
    reg = wp["region"].get("regions") or []
    name = reg[0]["name"] if reg else ""
    ax.set_title(title or f"worth profile ({name}); solid = picked")
    _pass_legend(ax, set(wp["pass_"][:n]))
    fig.tight_layout()
    return fig


def plot_worth_path(wpth: dict, ax=None, title: Optional[str] = None):
    """Lines of item worth along worth_path()."""
    fig, ax = _ax(ax)
    for j, i in enumerate(wpth["items"]):
        ax.plot(wpth["t"], wpth["P"][:, j], label=i)
    ax.set_xlabel(f"archetype of {wpth['a'] or 'a'}  ->  archetype of {wpth['b'] or 'b'}  (distance {wpth['distance']:.2f})")
    ax.set_ylabel("worth"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title or "worth along an archetype path")
    return fig


def plot_stat_response(sr: dict, ax=None, title: Optional[str] = None):
    """Lines of item worth against one stat (stat_response())."""
    fig, ax = _ax(ax)
    for j, i in enumerate(sr["items"]):
        ax.plot(sr["values"], sr["P"][:, j], marker=".", label=i)
    ax.set_xlabel(f"highest {sr['stat']}"); ax.set_ylabel("worth"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(title or f"worth against {sr['stat']}")
    return fig


def plot_distance_check(dc: dict, ax=None, title: Optional[str] = None):
    """Concept distance against RMS worth difference; the line is the d/4 bound."""
    fig, ax = _ax(ax)
    s = dc["same_region"]
    ax.scatter(dc["d"][~s], dc["rms_worth_diff"][~s], s=4, alpha=0.4, label="different region", c=_pass_colour("central"))
    ax.scatter(dc["d"][s], dc["rms_worth_diff"][s], s=4, alpha=0.4, label="same region", c=_pass_colour("foundational"))
    x = np.linspace(0, dc["d"].max() if len(dc["d"]) else 1, 50)
    ax.plot(x, x / 4, "--", color=_T["fg"], lw=1, label="bound d / 4")
    ax.set_xlabel("concept distance d"); ax.set_ylabel("RMS worth difference")
    ax.legend(fontsize=8)
    ax.set_title(title or f"distance bounds worth (holds on {100 * dc['holds']:.1f}% of pairs)")
    return fig


def plot_review_reach(cm: dict, rr: dict, build_xy: Optional[dict] = None, ax=None, title: Optional[str] = None):
    """concept_map() coloured by review_reach() similarity; build_xy: place() of the reviewed build."""
    return plot_concept_map(cm, ax=ax, color=rr["similarity"], cmap="viridis", color_label="review similarity",
                            highlight=build_xy, labels=True,
                            title=title or f"reach of a review (sim_min {rr['sim_min']} at distance {rr['reach']})")


# ------------------------------------------------------------------ gallery
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Projection):
        return x.to_dict()
    if isinstance(x, BuildMatrix):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not callable(v) and not isinstance(v, BuildMatrix)}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def save_data(data: dict, path: str) -> str:
    """Write any view's data as JSON (arrays become lists)."""
    with open(path, "w") as f:
        json.dump(_jsonable(data), f)
    return path


def gallery(tri, out_dir: str, builds: Builds = None, focus: Sequence[Build] = (), sample: int = 3000,
            seed: int = 0, dpi: int = 150, dims: int = 6, pairs: Optional[str] = None,
            axes: str = "explain", per_view: int = 3) -> Dict[str, str]:
    """The standard set of views as PNG + JSON in out_dir. focus: builds to mark
    on the maps and give profiles, reach maps and a path between the first two.
    dims / pairs: the concept and worth spaces are drawn on the first `dims`
    axes, per_view (3: a 3-D volume; 2: a plane) at a time (1-2-3, 4-5-6, ... or
    the groups given, e.g. "1-2-3 1-4-5"), each with an axis summary; axes:
    "off", "on" or "explain" (see plot_concept_map)."""
    plt = _plt()
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}

    def save(fig, data, name):
        p = os.path.join(out_dir, name + ".png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        save_data(data, os.path.join(out_dir, name + ".json"))
        out[name] = p

    kc = max(per_view, min(int(dims), int(tri.space.cfg.dim)))
    cm = concept_map(tri, builds, "pca", k=kc, sample=sample, seed=seed)
    im = item_map(tri, cm["projection"])
    hl = place(tri, list(focus), cm["projection"]) if focus else None
    ci = concept_axes(tri, cm)
    views = axis_groups(kc, pairs, per_view)
    save(plot_concept_map(cm, highlight=hl, items=im, dims=views[0], axes=axes, axis_info=ci), cm, "concept_map")
    for v in views[1:]:
        save(plot_concept_map(cm, highlight=hl, items=im, dims=v, axes=axes, axis_info=ci),
             dict(dims=list(v), axes=ci), "concept_map_" + "-".join(str(d + 1) for d in v))
    save(plot_axis_summary(ci, [(v, view_kept(cm["U"], cm["xy"], v)) for v in views], cm["projection"].distance_kept,
                           "concept space axes (pca)", cm["region_names"]), dict(axes=ci), "concept_axes")
    cr = concept_map(tri, builds, "regions", sample=sample, seed=seed)
    save(plot_concept_map(cr, highlight=place(tri, list(focus), cr["projection"]) if focus else None), cr,
         "concept_map_regions")
    save(plot_item_map(im), im, "item_map")
    ws = worth_space(tri, builds, sample=min(sample, 2000), seed=seed, min_rate=0.01, k=max(per_view, int(dims)))
    wi = worth_axes(ws)
    wv = axis_groups(ws["xy"].shape[1], pairs, per_view)
    save(plot_worth_space(ws, dims=wv[0], axes=axes, axis_info=wi), ws, "worth_space")
    for v in wv[1:]:
        save(plot_worth_space(ws, dims=v, axes=axes, axis_info=wi), dict(dims=list(v), axes=wi),
             "worth_space_" + "-".join(str(d + 1) for d in v))
    save(plot_axis_summary(wi, [(v, view_kept(ws["distance"], ws["xy"], v, distances=True)) for v in wv],
                           ws["distance_kept"], "worth space axes", ws["region_names"]), dict(axes=wi), "worth_axes")
    wr = worth_space(tri, builds, sample=min(sample, 2000), seed=seed, min_rate=0.01, relative=True)
    save(plot_worth_space(wr), wr, "worth_space_relative")
    rw = region_worth(tri, builds, sample=sample, seed=seed)
    save(plot_region_worth(rw), rw, "region_worth")
    dc = distance_check(tri, builds, sample=min(sample, 2000), seed=seed)
    save(plot_distance_check(dc), dc, "distance_check")
    for k, b in enumerate(focus):
        tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(b.build_id or k))
        wp = worth_profile(tri, b)
        save(plot_worth_profile(wp), wp, f"profile_{tag}")
        rr = review_reach(tri, b, builds, sample=sample, seed=seed)
        save(plot_review_reach(cm, rr, place(tri, [b], cm["projection"])), rr, f"reach_{tag}")
    if len(focus) >= 2:
        wpth = worth_path(tri, focus[0], focus[1])
        save(plot_worth_path(wpth), wpth, "path")
    return out
