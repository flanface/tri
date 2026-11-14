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


# ------------------------------------------------------------------ vectors alone
VECTOR_ARCS_ALL = 4         # up to this many vectors: an arc for every pair; more: each to its nearest


def _arc(a: np.ndarray, b: np.ndarray, r: float, n: int = 32) -> np.ndarray:
    """Points of the arc of radius r from direction a to direction b (in their plane)."""
    ua = a / np.linalg.norm(a)
    ub = b / np.linalg.norm(b)
    c = float(np.clip(ua @ ub, -1.0, 1.0))
    th = np.arccos(c)
    w = ub - c * ua
    nw = np.linalg.norm(w)
    if nw < 1e-12:                                  # antiparallel: any perpendicular plane
        w = np.cross(ua, [1.0, 0, 0] if abs(ua[0]) < 0.9 else [0, 1.0, 0])
        nw = np.linalg.norm(w)
    w = w / nw
    t = np.linspace(0.0, th, n)[:, None]
    return r * (np.cos(t) * ua + np.sin(t) * w)


def vector_pairs(V: np.ndarray, every: int = VECTOR_ARCS_ALL) -> List[Tuple[int, int]]:
    """The pairs whose angle is drawn: all of them for up to `every` vectors,
    otherwise each vector with the one at the smallest angle to it."""
    n = len(V)
    ok = [i for i in range(n) if np.linalg.norm(V[i]) > 1e-12]
    if len(ok) <= every:
        return [(a, b) for k, a in enumerate(ok) for b in ok[k + 1:]]
    C = cosines(V)
    out = set()
    for a in ok:
        b = max((j for j in ok if j != a), key=lambda j: C[a, j])
        out.add((min(a, b), max(a, b)))
    return sorted(out)


def _spread(P: np.ndarray, W: np.ndarray, H: np.ndarray, fixed: np.ndarray = None,
            steps: int = 400) -> np.ndarray:
    """Move label boxes (centres P, widths W, heights H, all in pixels) apart
    until none overlap each other or any fixed point (a small keep-out box),
    each push along the axis of least overlap."""
    L = np.array(P, float)
    F = np.zeros((0, 2)) if fixed is None else np.asarray(fixed, float)
    pad = 4.0
    for _ in range(steps):
        moved = False
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                d = L[i] - L[j]
                ox = (W[i] + W[j]) / 2 + pad - abs(d[0])
                oy = (H[i] + H[j]) / 2 + pad - abs(d[1])
                if ox > 0 and oy > 0:
                    if ox < oy * 2.5:
                        s_ = np.sign(d[0]) or 1.0
                        L[i, 0] += s_ * ox / 2; L[j, 0] -= s_ * ox / 2
                    else:
                        s_ = np.sign(d[1]) or 1.0
                        L[i, 1] += s_ * oy / 2; L[j, 1] -= s_ * oy / 2
                    moved = True
            for f in F:
                d = L[i] - f
                ox = W[i] / 2 + 6 - abs(d[0])
                oy = H[i] / 2 + 6 - abs(d[1])
                if ox > 0 and oy > 0:
                    if oy <= ox:
                        L[i, 1] += (np.sign(d[1]) or 1.0) * oy
                    else:
                        L[i, 0] += (np.sign(d[0]) or 1.0) * ox
                    moved = True
        if not moved:
            break
    return L


def draw_vectors(labels: Sequence[str], V: np.ndarray, cols: Sequence[str],
                 view: Tuple[float, float] = (22, -58), size: Tuple[float, float] = (8.0, 7.0)):
    """Vectors from one origin (rows of V, already relative to it), nothing else:
    each tip labelled with its name and magnitude |v|, each drawn pair's angle
    as an arc with its degrees. Equal axis scale, so the drawing is undistorted.
    Arcs sit at radii scaled to the longest vector; a shorter vector is
    continued by a faint dotted line out to its arcs. Labels are spread apart
    on the page (thin leader lines back to what they label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    V = np.asarray(V, float)
    fig = plt.figure(figsize=size)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    norms = np.linalg.norm(V, axis=1)
    R = float(norms.max()) if len(norms) and norms.max() > 1e-12 else 1.0
    pairs = vector_pairs(V)
    radii = [R * (0.16 + 0.055 * (k % 5)) for k in range(len(pairs))]
    reach = np.zeros(len(V))
    for (a, b), r in zip(pairs, radii):
        reach[a], reach[b] = max(reach[a], r), max(reach[b], r)
    Q = np.vstack([V * 1.1, np.zeros((1, 3))])
    lo, hi = Q.min(0), Q.max(0)
    mid, half = (lo + hi) / 2, float(np.max(hi - lo)) / 2 or 1.0
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.1)
    ax.view_init(*view)
    ax.scatter(0, 0, 0, s=26, c="white", edgecolors=EDGE, linewidths=0.9, depthshade=False)
    for v, c, m, rr in zip(V, cols, norms, reach):
        ax.quiver(0, 0, 0, *v, color=c, lw=1.8, arrow_length_ratio=min(0.07, 0.07 * R / max(m, 1e-12) * 0.3))
        if m > 1e-12 and rr > m:
            e = v / m * rr * 1.04
            ax.plot(*np.vstack([v, e]).T, color=c, lw=0.8, ls=":", alpha=0.7)
    M = ax.get_proj()

    def flat(p):
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], M)
        return np.array([x, y])
    o2 = flat(np.zeros(3))
    anchors, texts, colours, sizes = [], [], [], []
    for lbl, v, c, m in zip(labels, V, cols, norms):
        anchors.append(flat(v))
        texts.append(f"{str(lbl)[:28]}\n|v| = {m:.3g}")
        colours.append(c)
        sizes.append(7.5)
    for (a, b), r in zip(pairs, radii):
        P = _arc(V[a], V[b], r)
        ax.plot(*P.T, color=FAINT, lw=0.9, alpha=0.9)
        anchors.append(flat(P[len(P) // 2]))
        texts.append(f"{float(angle(cosines(V[[a, b]])[0, 1])):.0f}°")
        colours.append(FAINT)
        sizes.append(7)
    A = np.array(anchors) if anchors else np.zeros((0, 2))
    if len(A):
        fig.canvas.draw()
        T = ax.transData
        Ap = T.transform(A)
        op = T.transform(o2[None])[0]
        k_ = fig.dpi / 72.0
        W = np.array([max(len(x) for x in t.split("\n")) * fs * 0.6 * k_ for t, fs in zip(texts, sizes)])
        H = np.array([(t.count("\n") + 1) * fs * 1.3 * k_ for t, fs in zip(texts, sizes)])
        start = []
        for p, w, h in zip(Ap, W, H):
            d = p - op
            n = float(np.hypot(*d))
            u = d / n if n > 1e-9 else np.array([0.0, 1.0])
            start.append(p + u * (0.5 * abs(u[0]) * w + 0.5 * abs(u[1]) * h + 6))
        seg = [T.transform(np.array([flat(v * t) for t in np.linspace(0.0, 1.0, 24)])) for v in V]
        Lp = _spread(np.array(start), W, H, fixed=np.vstack([Ap, op[None]] + seg))
        L = T.inverted().transform(Lp)
        for p, pp, l, lp, t, c, fs, w, h in zip(A, Ap, L, Lp, texts, colours, sizes, W, H):
            far = abs(lp[0] - pp[0]) > w / 2 + 8 or abs(lp[1] - pp[1]) > h / 2 + 8
            ax.annotate(t, xy=p, xytext=l, textcoords="data", xycoords="data", fontsize=fs, color=c,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6,
                                        shrinkA=0, shrinkB=2) if far else None)
    return fig
