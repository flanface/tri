"""Data views of a fitted model and the matplotlib themes of its graphics.

Data (plain numpy arrays and lists, no plotting)
    concept_map      builds placed on a projection of the concept space (k axes),
                     with their archetype regions and the region centres
    place            further builds placed on the same projection
    worth_matrix     field worth of every item for every build
    worth_space      items placed by their worth across builds: two items are
                     close when the same builds value them alike
    view_kept        the distance one view of a few axes keeps

Projections (Projection, fit_projection)
    "pca"      weighted principal components of the builds' coordinates; the
               linear view that keeps the most of the true distances
    "regions"  principal components of the region centres only; spreads archetypes
    "items"    axis j = logit of item j (up to a constant): an exact, readable
               view of two chosen items' worth, not distance-preserving

Every Projection reports `distance_kept`: the correlation between true concept
distances and distances in the picture on a sample of pairs. The /plot figures
are drawn by tri.views.figure; the archetype graphics by tri.views.insight,
inside `with theme(name):`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

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


# ------------------------------------------------------------------ plots
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


def _short(name: str, n: int) -> str:
    s = str(name)
    return s if len(s) <= n else s[: n - 1].rstrip(" /") + "…"


# ------------------------------------------------------------------ views on several axes
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


# ------------------------------------------------------------------ gallery