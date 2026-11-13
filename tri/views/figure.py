"""One transparent 3-D figure with no external annotations (the /plot style).

Everything sits inside the scene: three thin axis lines through the origin
(x red, y green, z blue) lettered at their ends, translucent group polytopes
with coloured edges (or one dot per point), locators as small white diamonds,
group centres as coloured circles with a faint name, marked builds as stars and
item vectors as arrows from an origin. No title, ticks, panes, grid, legend or
colourbar; the PNG background is transparent (assets/atlas_export1.png).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

AXIS_COLOURS = ("#ff5a6e", "#3ccf7a", "#4f9dff")
PALETTE = ("#ff4d6d", "#9b6bff", "#2fc4b2", "#ff9f1c", "#4d8dff", "#e9f542",
           "#ff70c8", "#7bd389", "#c08a5b", "#8fa3bf", "#ff6b3d", "#5ad1ff")
EDGE = "#202020"
FAINT = "#9a9a9a"
KIND_COLOURS = {"talent": "#8fa3bf", "mantra": "#c9a26b"}


def colours(n: int) -> List[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


@dataclass
class Scene:
    """What to draw. P: (n, 3) points; group[i] in 0..G-1 or -1 (ungrouped)."""
    P: np.ndarray
    group: np.ndarray
    names: Sequence[str]
    style: str = "fields"                       # "fields" (polytopes) or "dots"
    centres: Optional[np.ndarray] = None        # (G, 3)
    locators: Optional[np.ndarray] = None       # (m, 3) white diamonds
    stars: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    origin: Optional[np.ndarray] = None         # (3,) vectors start here
    vectors: List[Tuple[str, np.ndarray, str]] = field(default_factory=list)   # (label, tip, colour)
    point_colour: Optional[Sequence[str]] = None   # per point, overrides the group colour
    point_size: float = 7.0
    core: float = 0.9                           # share of a group's points (nearest its median) its field spans
    field_mask: Optional[np.ndarray] = None     # (n,) points that may shape a field (all when None)
    labels: Optional[Sequence[str]] = None      # per point hover name (interactive page)
    locator_labels: Optional[Sequence[str]] = None


def _hull_edges(h, crease: float = 0.82):
    """Feature edges of a 3-D hull: between facets turning by more than arccos(crease)."""
    E = set()
    for f, (s, nb) in enumerate(zip(h.simplices, h.neighbors)):
        for k in range(3):
            g = nb[k]
            if g > f and float(h.equations[f, :3] @ h.equations[g, :3]) < crease:
                a, b = [s[j] for j in range(3) if j != k]
                E.add((int(min(a, b)), int(max(a, b))))
    return sorted(E)


def field_hulls(sc: Scene) -> List[dict]:
    """Per group with at least 5 field points: the hull of the `core` share of
    them nearest the group median. dict(g, V vertices, F triangles, E crease edges)."""
    from scipy.spatial import ConvexHull, QhullError
    P0 = np.asarray(sc.P, float)
    ok = np.ones(len(P0), bool) if sc.field_mask is None else np.asarray(sc.field_mask, bool)
    out = []
    for g in range(len(sc.names)):
        P = P0[(np.asarray(sc.group) == g) & ok]
        if len(P) < 5:
            continue
        d = np.linalg.norm(P - np.median(P, 0), axis=1)
        P = P[d <= np.quantile(d, sc.core)]
        try:
            h = ConvexHull(P)
        except (QhullError, ValueError):
            continue
        out.append(dict(g=g, V=P, F=[list(map(int, s)) for s in h.simplices], E=_hull_edges(h)))
    return out


def _fields(ax, sc: Scene, cols: List[str], alpha: float = 0.2) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    for h in field_hulls(sc):
        P, c = h["V"], cols[h["g"]]
        ax.add_collection3d(Poly3DCollection([P[s] for s in h["F"]], facecolors=c, alpha=alpha, edgecolors="none"))
        ax.add_collection3d(Line3DCollection([P[[a, b]] for a, b in h["E"]], colors=[c], linewidths=0.9, alpha=0.9))


def _axes(ax, lo: np.ndarray, hi: np.ndarray, at: np.ndarray) -> None:
    """Three axis lines through `at`, spanning the scene, lettered at the + end."""
    span = hi - lo
    for j, (c, name) in enumerate(zip(AXIS_COLOURS, "xyz")):
        a, b = at.copy(), at.copy()
        a[j], b[j] = lo[j] - 0.08 * span[j], hi[j] + 0.08 * span[j]
        ax.plot(*np.c_[a, b], color=c, lw=0.7, alpha=0.55)
        ax.text(*b, name, color=c, fontsize=7)


def draw(sc: Scene, view: Tuple[float, float] = (22, -58), size: Tuple[float, float] = (10.0, 7.0)):
    """The figure for a Scene (matplotlib Figure; save with png())."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=size)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    cols = colours(len(sc.names))
    P = np.asarray(sc.P, float)
    allp = [P] + [x for x in (sc.locators, sc.centres) if x is not None and len(x)]
    allp += [np.atleast_2d(t) for _, t in sc.stars] + [np.atleast_2d(t) for _, t, _ in sc.vectors]
    Q = np.vstack(allp)
    lo, hi = Q.min(0), Q.max(0)
    at = np.asarray(sc.origin, float) if sc.origin is not None else np.clip(np.zeros(3), lo, hi)
    _axes(ax, lo, hi, at)
    if sc.style == "fields":
        _fields(ax, sc, cols)
    else:
        pc = sc.point_colour or [cols[g] if g >= 0 else FAINT for g in sc.group]
        ax.scatter(*P.T, s=sc.point_size, c=list(pc), alpha=0.6, linewidths=0, depthshade=False)
    if sc.locators is not None and len(sc.locators):
        ax.scatter(*np.asarray(sc.locators).T, marker="D", s=16, c="white", edgecolors=EDGE,
                   linewidths=0.8, depthshade=False, zorder=4)
    if sc.centres is not None:
        for g, p in enumerate(np.asarray(sc.centres)):
            if not np.all(np.isfinite(p)):
                continue
            ax.scatter(*p, s=60, c=cols[g], edgecolors=EDGE, linewidths=0.8, depthshade=False, zorder=5)
            ax.text(*p, "  " + str(sc.names[g])[:24], fontsize=5.5, color=FAINT, alpha=0.8)
    for lbl, p in sc.stars:
        ax.scatter(*p, marker="*", s=150, c="#ffd23f", edgecolors=EDGE, linewidths=0.6, depthshade=False, zorder=6)
        ax.text(*p, "  " + str(lbl)[:30], fontsize=6, color=FAINT)
    if sc.origin is not None:
        o = np.asarray(sc.origin, float)
        ax.scatter(*o, s=30, c="white", edgecolors=EDGE, linewidths=0.9, depthshade=False, zorder=6)
        for lbl, tip, c in sc.vectors:
            v = np.asarray(tip, float) - o
            ax.quiver(*o, *v, color=c, lw=1.4, arrow_length_ratio=0.08, alpha=0.95)
            ax.text(*(o + 1.04 * v), str(lbl)[:30], fontsize=6.5, color=c)
    pad = 0.02 * np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)), zoom=1.15)
    ax.view_init(*view)
    return fig


def png(fig, dpi: int = 150) -> bytes:
    """Transparent PNG bytes; closes the figure."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ worth origin and cosines
def dense_origin(X: np.ndarray, k: int = 12) -> np.ndarray:
    """The centre of the densest cluster of rows of X: the point whose k-th
    nearest neighbour is closest, averaged with those k neighbours."""
    X = np.asarray(X, float)
    if len(X) <= k:
        return X.mean(0)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    rk = np.sort(D, 1)[:, k]
    i = int(np.argmin(rk))
    nb = np.argsort(D[i])[: k + 1]
    return X[nb].mean(0)


def orthogonality(c) -> np.ndarray:
    """|sin(theta)| = sqrt(1 - cos^2): the share of one vector's length that is
    orthogonal to the other (0 parallel or antiparallel, 1 orthogonal)."""
    c = np.clip(np.asarray(c, float), -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - c * c))


def angle(c) -> np.ndarray:
    """The angle in degrees for cosine c."""
    return np.degrees(np.arccos(np.clip(np.asarray(c, float), -1.0, 1.0)))


def cosines(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of the rows of V (zero rows give 0)."""
    V = np.asarray(V, float)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / np.where(n > 1e-12, n, 1.0)
    return np.clip(U @ U.T, -1.0, 1.0)
