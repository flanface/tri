"""Depth cues and transparent export for every figure.

2-D axes: markers and lines cast a soft drop shadow; bars carry a vertical
light gradient (lit from above). 3-D axes: see figure.py (lit polytope faces,
depth-faded points, vector shadows on the floor). export() saves any figure
as a transparent PNG with the cues applied.

  light(n, L) = |n · L|        (Lambert, two-sided)
  face colour = mix(c, 0.45 · light − 0.25)
"""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np

SHADOW = (1.2, -1.2)        # points, right and down
SHADOW_ALPHA = 0.28
GRADIENT = 0.32             # bar highlight / shade strength


def mix(c, t: float):
    """c lightened (t > 0, towards white) or darkened (t < 0); RGBA tuple."""
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(c)
    f = (lambda x: x + (1 - x) * t) if t >= 0 else (lambda x: x * (1 + t))
    return (f(r), f(g), f(b), a)


def lambert(normals: np.ndarray, L: Sequence[float]) -> np.ndarray:
    """|n · L| per row (unit normals; L normalised here)."""
    L = np.asarray(L, float)
    L = L / (np.linalg.norm(L) or 1.0)
    N = np.asarray(normals, float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return np.abs(N @ L)


def _bar_gradient(ax, p) -> None:
    x0, y0 = p.get_xy()
    w, h = p.get_width(), p.get_height()
    if not (np.isfinite([x0, y0, w, h]).all() and w and h):
        return
    t = np.linspace(1, -1, 48)[:, None]                 # top lit, bottom shaded
    rgba = np.zeros((48, 1, 4))
    rgba[..., :3] = (t > 0)[..., None] * 1.0
    rgba[..., 3] = np.abs(t) * GRADIENT
    lim = ax.get_xlim(), ax.get_ylim()
    im = ax.imshow(rgba, extent=(x0, x0 + w, y0, y0 + h) if h > 0 else (x0, x0 + w, y0 + h, y0),
                   aspect="auto", interpolation="bilinear", zorder=p.get_zorder() + 0.01, origin="upper")
    im.set_clip_path(p)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])


def shade2d(fig) -> None:
    """Drop shadows on markers and lines, light gradients on bars (2-D axes only)."""
    from matplotlib import patheffects as pe
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Rectangle
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d" or getattr(ax, "_tri_shaded", False):
            continue
        ax._tri_shaded = True
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()) <= 5000:
                c.set_path_effects([pe.SimplePatchShadow(offset=SHADOW, alpha=SHADOW_ALPHA, shadow_rgbFace="black"),
                                    pe.Normal()])
        for ln in ax.lines:
            if ln.get_linewidth() >= 1.0 and ln.get_linestyle() not in ("None", "", " "):
                ln.set_path_effects([pe.SimpleLineShadow(offset=SHADOW, alpha=SHADOW_ALPHA * 0.8), pe.Normal()])
        for p in list(ax.patches):
            if isinstance(p, Rectangle) and p.get_visible() and p.get_facecolor()[3] > 0 \
                    and p.get_transform() == ax.transData:
                p.set_path_effects([pe.SimplePatchShadow(offset=SHADOW, alpha=SHADOW_ALPHA * 0.8), pe.Normal()])
                _bar_gradient(ax, p)


def export(fig, dpi: int = 150, pad: float = 0.04) -> bytes:
    """Transparent PNG bytes with depth cues; closes the figure."""
    import matplotlib.pyplot as plt
    shade2d(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return buf.getvalue()
