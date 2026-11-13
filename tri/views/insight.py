"""Graphics of the categorization, the facts and the stream (math §12).
Same themes and conventions as tri.views.plots: draw inside
`with render.theme(name):`; every function returns a matplotlib figure.

  plot_atlas       B1  archetype atlas: planes with every build (used by /locator suggest)
  plot_card        B3  one archetype: core, contested, holes, stat envelope, neighbours
  plot_placement   B4  one build: inset atlas, nearest locators and reviews, pi bar
  plot_item        B5  worth landscape of one item, review +/- markers, g_t arrow
  plot_stats       B6  worth of an item against a stat, per archetype
  plot_replay      D   prequential curve, claim variogram, coverage
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from tri.analysis.categorize import separating, sigmoid, signature
from tri.analysis.defs import Atlas
from tri.views import plots as R

UNPLACED = "unplaced"


def _T():
    return R._T


_PALETTES = {
    True: ["#4ea1ff", "#ff9f1c", "#2ec4b6", "#ff4d6d", "#b388ff", "#e9ff70", "#f15bb5", "#7bd389", "#c9a227",
           "#9aa5b8"],                                                   # dark backgrounds
    False: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#bcbd22", "#8c564b",
            "#7f7f7f"],
}


def _colours(n: int):
    """Archetype colours: ten well-separated hues, the region palette beyond."""
    n = max(n, 1)
    if n <= 10:
        return _PALETTES[_T()["bg"] != "#ffffff"][:n]
    return R._region_colours(n)


def _title(fig, text: str):
    fig.suptitle(text, fontsize=10, x=0.01, ha="left", color=_T()["fg"], weight="bold")


# ------------------------------------------------------------------ the plane
def atlas_view(tri, at: Atlas, k: int = 9, candidates: Sequence[dict] = ()) -> dict:
    """Projection of the atlas sample onto its first k principal axes, with the
    locators, archetype centres (and their two signature items) and candidates."""
    k = max(2, min(k, at.U.shape[1]))
    P = R.fit_projection(tri, at.U, "pca", k, at.w)
    xy = P.transform(at.U)
    loc_xy, loc_lab = np.zeros((0, k)), np.zeros(0, int)
    if tri.locset is not None and at.source == "locators":
        loc_xy, loc_lab = P.transform(tri.locset.U), np.asarray(tri.locset.lab)
    av = at.avail.mean(0)
    sig = []
    for a in range(len(at.names)):
        m = at.members(a)
        rate = at.avail[m].mean(0) if m.any() else av
        sig.append([s["item"] for s in signature(tri, at.centres[a], at.ubar, rate, 2)])
    cand = [dict(id=c["id"], xy=P.transform(c["centre"][None])[0]) for c in candidates]
    kept = [R.view_kept(at.U, xy, (d, d + 1)) for d in range(0, k - 1, 2)]
    return dict(projection=P, xy=xy, centre_xy=P.transform(at.centres), loc_xy=loc_xy, loc_lab=loc_lab,
                signatures=sig, candidates=cand, kept=kept)


def _grid(xy: np.ndarray, n: int = 110, pad: float = 0.06):
    lo, hi = np.quantile(xy, 0.005, 0), np.quantile(xy, 0.995, 0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    gx, gy = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    return G, (lo[0], hi[0], lo[1], hi[1]), n


_MEMO: Dict[tuple, tuple] = {}


def _smooth_memo(key, ext, n, xy, V, w):
    """_smooth, remembered for the same atlas plane (the background is reused by
    every view drawn on that plane)."""
    if key in _MEMO:
        return _MEMO[key]
    if len(_MEMO) > 16:
        _MEMO.clear()
    _MEMO[key] = _smooth(ext, n, xy, V, w)
    return _MEMO[key]


def _smooth(ext, n: int, xy: np.ndarray, V: np.ndarray, w: np.ndarray, bw_frac: float = 0.05):
    """Nadaraya-Watson with a Gaussian kernel over the n x n grid of ext, by
    binning: weighted histograms of w and w * V, each blurred by the kernel,
    then divided. The grid is padded by three kernel widths (points outside
    the window still count near its edges) and cropped. Returns
    (values [n * n, ...], density [n * n])."""
    from scipy.ndimage import gaussian_filter
    bw = bw_frac * (np.ptp(xy, 0).max() + 1e-12)
    dx, dy = (ext[1] - ext[0]) / (n - 1), (ext[3] - ext[2]) / (n - 1)
    sx, sy = bw / dx, bw / dy
    mx, my = int(np.ceil(3 * sx)), int(np.ceil(3 * sy))
    nx, ny = n + 2 * mx, n + 2 * my
    ix = np.round((xy[:, 0] - ext[0]) / dx).astype(int) + mx
    iy = np.round((xy[:, 1] - ext[2]) / dy).astype(int) + my
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]

    def blur(weights):
        h = np.bincount(flat, weights=weights[ok], minlength=nx * ny).reshape(ny, nx)
        return gaussian_filter(h, (sy, sx), mode="constant")[my:my + n, mx:mx + n].reshape(-1)

    dens = blur(w)
    V2 = V.reshape(len(V), -1)
    out = np.stack([blur(w * V2[:, j]) for j in range(V2.shape[1])], -1) / np.maximum(dens, 1e-12)[:, None]
    return out.reshape((n * n,) + V.shape[1:]), dens


def _background(ax, at: Atlas, xy: np.ndarray, colours, n: int = 170, alpha: float = 0.55, ext=None):
    """argmax-archetype field with saturation pi_1; fades where no builds are.
    ext: the window to draw (default: the extent of the builds)."""
    import matplotlib.colors as mc
    if ext is None:
        G, ext, n = _grid(xy, n)
    Pg, dens = _smooth_memo((zlib.crc32(xy.tobytes()), zlib.crc32(at.P.tobytes()), n, tuple(map(float, ext))), ext, n, xy, at.P, at.w)
    top = Pg.argmax(1)
    pi1 = Pg.max(1)
    rgb = np.array([mc.to_rgb(colours[t]) for t in top])
    bg = np.array(mc.to_rgb(_T()["panel"]))
    dmask = np.clip(dens / (np.quantile(dens, 0.6) + 1e-12), 0, 1)
    a = alpha * np.clip((pi1 - 1.0 / max(len(at.names), 1)) / (1 - 1.0 / max(len(at.names), 1) + 1e-9), 0, 1) * dmask
    img = (1 - a[:, None]) * bg + a[:, None] * rgb
    ax.imshow(img.reshape(n, n, 3), origin="lower", extent=ext, aspect="auto", interpolation="bilinear", zorder=0)
    _boundaries(ax, top.reshape(n, n), dmask.reshape(n, n) > 0.15, ext)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    return ext


def _boundaries(ax, lab: np.ndarray, keep: np.ndarray, ext, colour=None, alpha: float = 0.75):
    """One-pixel lines where the argmax archetype changes (only where builds are)."""
    import matplotlib.colors as mc
    e = np.zeros(lab.shape, bool)
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    e &= keep
    rgba = np.zeros(lab.shape + (4,))
    rgba[..., :3] = mc.to_rgb(colour or _T()["muted"])
    rgba[..., 3] = e * alpha
    ax.imshow(rgba, origin="lower", extent=ext, aspect="auto", interpolation="nearest", zorder=1)


def _place_labels(ax, pts: np.ndarray, order: Sequence[int], gap=(0.13, 0.075), margin=(0.1, 0.07)):
    """Offsets (points) for labels at pts so their boxes do not overlap and stay
    inside the axes: each label moves up or down in steps until it clears the
    ones placed."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    f = (pts - [x0, y0]) / [x1 - x0, y1 - y0]
    pos = ax.get_position()
    W = ax.figure.get_size_inches()[0] * 72 * pos.width
    H = ax.figure.get_size_inches()[1] * 72 * pos.height
    placed, off = [], {}
    for j in order:
        base = np.clip(f[j], [margin[0], margin[1]], [1 - margin[0], 1 - margin[1]])
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            q = np.clip(base + [0, k * gap[1]], [margin[0], margin[1]], [1 - margin[0], 1 - margin[1]])
            if all(abs(q[0] - p[0]) >= gap[0] or abs(q[1] - p[1]) >= gap[1] for p in placed):
                break
        placed.append(q)
        off[j] = (float((q[0] - f[j][0]) * W), float((q[1] - f[j][1]) * H))
    return off


def plot_atlas(av: dict, at: Atlas, panels: Sequence[tuple] = ((0, 1), (2, 3)), points: int = 1500,
               title: Optional[str] = None):
    """B1: one panel per pair of axes."""
    plt = R._plt()
    panels = [p for p in panels if max(p) < av["xy"].shape[1]]
    fig, axs = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.6), squeeze=False)
    cols = _colours(len(at.names))
    rng = np.random.default_rng(0)
    show = rng.choice(len(at.U), min(points, len(at.U)), replace=False)
    wmax = max(at.w.max(), 1e-12)
    for ax, dims, kept in zip(axs[0], panels, av["kept"]):
        xy = av["xy"][:, dims]
        _background(ax, at, xy, cols)
        pl = show[at.placed[show]]
        up = show[~at.placed[show]]
        ax.scatter(xy[pl, 0], xy[pl, 1], s=4 + 10 * at.w[pl] / wmax, c=[cols[t] for t in at.top[pl]],
                   alpha=0.75, linewidths=0, zorder=2)
        ax.scatter(xy[up, 0], xy[up, 1], s=14, facecolors="none", edgecolors=_T()["muted"], linewidths=0.6,
                   alpha=0.8, zorder=2)
        if len(av["loc_xy"]):
            L = av["loc_xy"][:, dims]
            ax.scatter(L[:, 0], L[:, 1], marker="*", s=130, c=[cols[t] for t in av["loc_lab"]],
                       edgecolors=_T()["star"], linewidths=0.7, zorder=4)
        C = av["centre_xy"][:, dims]
        live = [a for a in np.argsort(-at.share) if at.count[a] > 0]
        off = _place_labels(ax, C, live)
        for a in live:
            nm = at.names[a]
            sig = " · ".join(R._short(s, 16) for s in av["signatures"][a])
            ax.annotate(f"{R._short(nm, 22)}\n{sig}" if sig else R._short(nm, 22), tuple(C[a]), xytext=off[a],
                        textcoords="offset points", ha="center", va="center", fontsize=7.2, color=_T()["fg"],
                        weight="bold", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc=_T()["bg"], ec=cols[a], lw=1.0, alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color=cols[a], lw=0.8) if np.hypot(*off[a]) > 4 else None)
        for c in av["candidates"]:
            p = c["xy"][list(dims)]
            ax.scatter([p[0]], [p[1]], s=260, facecolors="none", edgecolors=_T()["fg"], linewidths=1.4,
                       linestyle="--", zorder=5)
            ax.annotate(c["id"], tuple(p), xytext=(-22, 16), textcoords="offset points", fontsize=8.5,
                        color=_T()["bg"], weight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc=_T()["fg"], ec="none", alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=_T()["fg"], lw=0.8))
        ax.set_xlabel(f"axis {dims[0] + 1}", fontsize=8); ax.set_ylabel(f"axis {dims[1] + 1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        ax.text(0.99, 0.01, f"distance kept {kept:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=_T()["muted"])
    lab = f"{len(at.names)} archetypes ({at.source}) · ★ locators · ○ unplaced"
    _title(fig, title or f"Archetype atlas · {lab}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ------------------------------------------------------------------ tree

# ------------------------------------------------------------------ card
def archetype_card(tri, at: Atlas, a: int, traps: Sequence[dict] = (), core_min: float = 0.8,
                   contested: tuple = (0.15, 0.85), top: int = 14) -> dict:
    """B3 data of archetype a: core items by pass, contested items with their
    sidegrades, traps, stat envelope (pre-shrine and final, 10th to 90th
    percentile) and the two nearest archetypes with their separating items."""
    m = at.members(a)
    ids = tri.catalog.ids
    av = at.avail[m]
    den = av.sum(0)
    q = np.where(den > 0, (at.Q[m] * av).sum(0) / np.maximum(den, 1), np.nan)
    can = den >= max(3, 0.3 * m.sum())
    passes = {i: R._pass_of(tri, i) for i in ids}
    core = sorted([(ids[t], float(q[t])) for t in np.flatnonzero(can & (np.nan_to_num(q) >= core_min))],
                  key=lambda x: -x[1])
    by_pass: Dict[str, List[tuple]] = {}
    for it, v in core:
        by_pass.setdefault(passes[it], []).append((it, v))
    cont = [(ids[t], float(q[t])) for t in np.flatnonzero(can & (np.nan_to_num(q) >= contested[0])
                                                          & (np.nan_to_num(q) < contested[1]))]
    cont.sort(key=lambda x: -x[1])
    regions = set(int(r) for r in tri.space.region_of(at.U[m])) if m.any() else set()
    sides = []
    for g in tri.sidegrades or []:
        if regions & set(g.regions or [g.region]):
            sides.append(dict(a=list(g.a), b=list(g.b), share_a=g.share_a, share_b=g.share_b))
    stats = list(at.bm.catalog.stats)
    env = []
    if m.any():
        pr, fi = at.bm.pr[m], at.bm.fi[m]
        for s, st in enumerate(stats):
            lo_f, hi_f = np.quantile(fi[:, s], [0.1, 0.9])
            lo_p, hi_p = np.quantile(pr[:, s], [0.1, 0.9])
            if hi_f > 0 or hi_p > 0:
                env.append(dict(stat=st, final=(float(lo_f), float(hi_f)), pre=(float(lo_p), float(hi_p))))
    dc = np.linalg.norm(at.centres - at.centres[a][None], axis=1)
    dc[a] = np.inf
    dc[at.count == 0] = np.inf
    neigh = []
    for b in np.argsort(dc)[:2]:
        if not np.isfinite(dc[b]):
            continue
        neigh.append(dict(name=at.names[b], distance=float(dc[b]),
                          items=separating(tri, at.centres[a], at.centres[b], 3)))
    return dict(name=at.names[a], builds=int(m.sum()), share=float(at.share[a]), core=by_pass,
                contested=cont[:top], sidegrades=sides, traps=list(traps), envelope=env, neighbours=neigh,
                consensus=consensus_rows(tri, at.names[a]),
                readings={k: int(((at.top == a) & (at.kind == k)).sum()) for k in ("pure", "leaning", "hybrid")})


def consensus_rows(tri, name: str, top: int = 6) -> List[dict]:
    """Promoted reviewer tells of the family of archetype name (math §8.5a):
    [{item, sign, reviewers, notes, confidence}], take tells first."""
    rep = tri.consensus_report(name) if hasattr(tri, "consensus_report") else {}
    rows = list(rep.get(name, []))
    return [dict(item=r["item"], sign=r["sign"], reviewers=r["reviewers"], notes=r["notes"],
                 confidence=r["confidence"]) for r in rows[:top]]


def plot_card(card: dict, title: Optional[str] = None):
    plt = R._plt()
    fig = plt.figure(figsize=(12, 8.2))
    g = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.15, 1], wspace=0.42, hspace=0.42)
    T = _T()
    # core by pass
    ax = fig.add_subplot(g[0, 0])
    rows = []
    for p in ("foundational", "central", "luxury"):
        for it, v in card["core"].get(p, [])[:9]:
            rows.append((it, v, p))
    rows = rows[:18]
    y = np.arange(len(rows))[::-1]
    for j, (it, v, p) in enumerate(rows):
        ax.barh(y[j], v, color=R._pass_colour(p), height=0.7)
    ax.set_yticks(y); ax.set_yticklabels([R._short(r[0], 26) for r in rows], fontsize=7)
    ax.set_xlim(0.75, 1.0); ax.set_title("core (worth ≥ 0.8)", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    if rows:
        from matplotlib.patches import Patch
        ps = [p for p in ("foundational", "central", "luxury") if any(r[2] == p for r in rows)]
        ax.legend(handles=[Patch(color=R._pass_colour(p), label=p) for p in ps], fontsize=7, ncol=len(ps),
                  loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False, borderaxespad=0)
    # contested strip with sidegrades
    ax = fig.add_subplot(g[0, 1])
    cont = card["contested"]
    yy = np.arange(len(cont))[::-1]
    side_of = {}
    for k, sg in enumerate(card["sidegrades"]):
        for it in sg["a"]:
            side_of[it] = (k, "A")
        for it in sg["b"]:
            side_of[it] = (k, "B")
    sc = _colours(max(2, 2 * len(card["sidegrades"])))
    for j, (it, v) in enumerate(cont):
        s = side_of.get(it)
        c = sc[2 * s[0] + (s[1] == "B")] if s else T["muted"]
        ax.plot([0, v], [yy[j], yy[j]], color=T["grid"], lw=1, zorder=1)
        ax.scatter([v], [yy[j]], s=46, color=c, edgecolors=T["fg"] if s else "none", linewidths=0.6, zorder=3)
    pos = {it: yy[j] for j, (it, _) in enumerate(cont)}
    for k, sg in enumerate(card["sidegrades"]):
        for side, ks in (("a", 0), ("b", 1)):
            pts = [(dict(cont)[i], pos[i]) for i in sg[side] if i in pos]
            if len(pts) > 1:
                pts.sort(key=lambda x: x[1])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=sc[2 * k + ks], lw=1.2, alpha=0.8, zorder=2)
    ax.set_yticks(yy); ax.set_yticklabels([R._short(c[0], 24) for c in cont], fontsize=7)
    ax.set_xlim(0.1, 0.9); ax.axvline(0.5, color=T["muted"], lw=0.7, ls=":")
    ax.set_title("contested (lines: sidegrade sides)", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    # stat envelope
    ax = fig.add_subplot(g[1, 0])
    env = card["envelope"]
    x = np.arange(len(env))
    for j, e in enumerate(env):
        ax.plot([j - 0.12, j - 0.12], e["pre"], color=R._pass_colour("foundational"), lw=5, solid_capstyle="butt",
                label="pre-shrine" if j == 0 else None)
        ax.plot([j + 0.12, j + 0.12], e["final"], color=R._pass_colour("central"), lw=5, solid_capstyle="butt",
                label="final" if j == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([R._short(e["stat"], 12) for e in env], rotation=60, fontsize=6.8,
                                         ha="right")
    ax.set_ylim(0, 100); ax.set_title("stats (10th–90th pct)", fontsize=9, loc="left")
    ax.tick_params(axis="y", labelsize=7)
    if env:
        ax.legend(fontsize=7, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, borderaxespad=0.2)
    # neighbours + traps
    ax = fig.add_subplot(g[1, 1]); ax.axis("off")
    yv = 0.97
    ax.text(0, yv, "neighbours", fontsize=9, weight="bold", color=T["fg"], transform=ax.transAxes, va="top")
    yv -= 0.1
    for nb in card["neighbours"]:
        ax.text(0, yv, f"{R._short(nb['name'], 30)}  d {nb['distance']:.2f}", fontsize=8, color=T["fg"],
                transform=ax.transAxes, va="top")
        yv -= 0.075
        for s in nb["items"]:
            arrow = "▲" if s["dlogit"] > 0 else "▼"
            ax.text(0.04, yv, f"{arrow} {R._short(s['item'], 30)}  {100 * s['share']:.0f}%", fontsize=7.4,
                    color=R._pass_colour("central") if s["dlogit"] > 0 else R._pass_colour("foundational"),
                    transform=ax.transAxes, va="top")
            yv -= 0.065
        yv -= 0.03
    if card["traps"]:
        ax.text(0, yv, "holes", fontsize=9, weight="bold", color=T["fg"], transform=ax.transAxes, va="top")
        yv -= 0.09
        for t in card["traps"][:3]:
            ax.text(0.04, yv, f"✕ {R._short(t['item'], 28)}  taken {100 * t['corpus']:.0f}% · removed "
                              f"{t['removed']}/{t['reviewed']}", fontsize=7.4, color=T["fg"],
                    transform=ax.transAxes, va="top")
            yv -= 0.065
    if card.get("consensus"):
        yv -= 0.02
        ax.text(0, yv, "reviewer consensus", fontsize=9, weight="bold", color=T["fg"], transform=ax.transAxes, va="top")
        yv -= 0.09
        for c in card["consensus"][:4]:
            ax.text(0.04, yv, f"{'+' if c['sign'] > 0 else '−'} {R._short(c['item'], 28)}  {c['reviewers']} rev · "
                              f"c {c['confidence']:.2f}", fontsize=7.4,
                    color=R._pass_colour("central") if c["sign"] > 0 else R._pass_colour("foundational"),
                    transform=ax.transAxes, va="top")
            yv -= 0.065
    rd = card["readings"]
    _title(fig, title or f"{card['name']} · {card['builds']} builds · {100 * card['share']:.0f}% · "
                         f"pure {rd['pure']} · leaning {rd['leaning']} · hybrid {rd['hybrid']}")
    return fig


# ------------------------------------------------------------------ placement inset
def placement(tri, at: Atlas, av: dict, b, k: int = 3) -> dict:
    from tri.analysis.categorize import categorize
    cat = categorize(tri, b)
    u = cat["u"]
    P = av["projection"]
    out = dict(category=cat, xy=P.transform(u[None])[0], locators=[], reviews=[])
    if tri.locset is not None and len(tri.locset.locators):
        d = np.linalg.norm(tri.locset.U - u[None], axis=1)
        for j in np.argsort(d)[:k]:
            out["locators"].append(dict(id=tri.locset.locators[j].locator_id, distance=float(d[j]),
                                        archetype=tri.locset.locators[j].archetype,
                                        xy=P.transform(tri.locset.U[j][None])[0]))
    led = tri.ledger
    if led is not None and led.by_id:
        rid = list(led.by_id)
        Ur = np.array([led.vector(r) for r in rid])
        d = np.linalg.norm(Ur - u[None], axis=1)
        for j in np.argsort(d)[:k]:
            out["reviews"].append(dict(id=rid[j], distance=float(d[j]), xy=P.transform(Ur[j][None])[0]))
    return out


def _placement_pi(bx, pl: dict, at: Atlas, cols) -> None:
    pi = pl["category"]["pi"]
    names = [R._short(n, 22) for n, _ in pi][::-1]
    vals = [v for _, v in pi][::-1]
    idx = {n: i for i, n in enumerate(at.names)}
    bx.barh(range(len(vals)), vals, color=[cols[idx.get(n, 0)] for n, _ in pi][::-1], height=0.6)
    for y_, v in enumerate(vals):
        inside = v > 0.85
        bx.text(v - 0.02 if inside else v + 0.02, y_, f"{v:.2f}", va="center", ha="right" if inside else "left",
                fontsize=7.5, color=_T()["bg"] if inside else _T()["fg"], weight="bold")
    bx.set_yticks(range(len(vals))); bx.set_yticklabels(names, fontsize=7.5)
    bx.set_xlim(0, 1); bx.axvline(0.8, color=_T()["muted"], ls=":", lw=0.8); bx.axvline(0.5, color=_T()["muted"],
                                                                                       ls=":", lw=0.8)
    bx.tick_params(axis="x", labelsize=7)
    c = pl["category"]
    bx.set_title(f"{c['reading']} · d {c['distance']:.2f}", fontsize=8.5, loc="left")


def plot_placement(pl: dict, av: dict, at: Atlas, title: Optional[str] = None, part: Optional[str] = None):
    """B4: inset atlas (axes 1-2) with the build, lines to its nearest locators
    (solid) and reviewed builds (dotted) with distances; pi bar of the top three.
    part: "map" or "pi" draws that panel alone (v2.21: /analyze posts two images)."""
    plt = R._plt()
    cols = _colours(len(at.names))
    if part == "pi":
        fig, bx = plt.subplots(figsize=(4.6, 0.55 + 0.42 * len(pl["category"]["pi"])))
        _placement_pi(bx, pl, at, cols)
        fig.tight_layout(pad=0.4)
        return fig
    if part == "map":
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    else:
        fig = plt.figure(figsize=(9, 3.9))
        g = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.25)
        ax = fig.add_subplot(g[0])
    xy = av["xy"][:, :2]
    p = pl["xy"][:2]
    near = np.array([p] + [l["xy"][:2] for l in pl["locators"]] + [r["xy"][:2] for r in pl["reviews"]])
    half = max(float(np.abs(near - p).max()) * 1.35, 0.12 * float(np.ptp(xy, 0).max()))
    ext = (p[0] - half, p[0] + half, p[1] - half, p[1] + half)
    _background(ax, at, xy, cols, n=140, ext=ext)
    inwin = (np.abs(xy[:, 0] - p[0]) < half) & (np.abs(xy[:, 1] - p[1]) < half)
    s = np.flatnonzero(inwin)[:800]
    ax.scatter(xy[s, 0], xy[s, 1], s=5, c=[cols[t] for t in at.top[s]], alpha=0.6, linewidths=0)
    for l in pl["locators"]:
        q = l["xy"][:2]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=_T()["fg"], lw=0.9, zorder=3)
        ax.scatter([q[0]], [q[1]], marker="*", s=120, facecolors="none", edgecolors=_T()["star"], linewidths=1.1,
                   zorder=4)
        ax.annotate(f"{l['distance']:.2f}", tuple(q), xytext=(5, -9), textcoords="offset points", fontsize=6.5,
                    color=_T()["fg"], bbox=dict(boxstyle="round,pad=0.1", fc=_T()["bg"], ec="none", alpha=0.7))
    near_r = min(range(len(pl["reviews"])), key=lambda i: pl["reviews"][i]["distance"]) if pl["reviews"] else -1
    for i, r in enumerate(pl["reviews"]):
        q = r["xy"][:2]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=_T()["passes"]["review"], lw=0.9, ls=":", zorder=3)
        ax.scatter([q[0]], [q[1]], marker="D", s=24, color=_T()["passes"]["review"], zorder=4)
        if i == near_r:                                   # label only the nearest reviewed build
            ax.annotate(f"{r['distance']:.2f}", tuple(q), xytext=(5, 3), textcoords="offset points", fontsize=6.5,
                        color=_T()["passes"]["review"], weight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc=_T()["bg"], ec="none", alpha=0.8))
    top = at.index(pl["category"]["best"])
    ax.scatter([p[0]], [p[1]], marker="*", s=380, color=cols[top if top is not None else 0], edgecolors=_T()["fg"],
               linewidths=1.2, zorder=6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    C = av["centre_xy"][:, :2]
    vis = [a for a in np.argsort(-at.share) if at.count[a] and abs(C[a, 0] - p[0]) < half and abs(C[a, 1] - p[1]) < half]
    off = _place_labels(ax, C, vis, gap=(0.3, 0.07))
    for a in vis:
        ax.annotate(R._short(at.names[a], 18), tuple(C[a]), xytext=off[a], textcoords="offset points", ha="center",
                    fontsize=7, color=cols[a], weight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc=_T()["bg"], ec="none", alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("★ build · ☆ locators · ◆ reviewed builds", fontsize=8, loc="left")
    if part == "map":
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout(pad=0.3)
        return fig
    bx = fig.add_subplot(g[1])
    _placement_pi(bx, pl, at, cols)
    c = pl["category"]
    _title(fig, title or c["text"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)      # the equal-aspect map axes
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ------------------------------------------------------------------ item landscape
def item_landscape(tri, at: Atlas, av: dict, item: str) -> dict:
    t = tri.catalog.index[item]
    marks = []
    led = tri.ledger
    if led is not None:
        for rid in led.by_id:
            for c in led.active_claims(rid):
                if c.item == item and c.label in ("keep", "add", "remove") and c.origin == "review":
                    marks.append(dict(xy=av["projection"].transform(led.vector(rid)[None])[0],
                                      sign=-1 if c.label == "remove" else 1))
    g = tri.space.item_coords()[t]
    gp = g @ av["projection"].W
    return dict(item=item, t=t, marks=marks, g=gp, g_norm=float(np.linalg.norm(g)),
                in_plane=float(np.linalg.norm(gp[:2]) / max(np.linalg.norm(g), 1e-12)))


def plot_item(il: dict, av: dict, at: Atlas, title: Optional[str] = None):
    """B5: worth of the item over the plane (builds that can take it), archetype
    boundaries, review + (keep/add) and − (remove) markers, the g_t arrow."""
    plt = R._plt()
    fig, ax = plt.subplots(figsize=(8, 6.6))
    xy = av["xy"][:, :2]
    t = il["t"]
    G, ext, n = _grid(xy, 170)
    m = at.avail[:, t]
    if m.sum() < 5:
        raise ValueError(f"too few builds can take {il['item']} to draw its worth")
    V, dens = _smooth(ext, n, xy[m], at.Q[m, t][:, None], at.w[m])
    Pg, dall = _smooth_memo((zlib.crc32(xy.tobytes()), zlib.crc32(at.P.tobytes()), n, tuple(map(float, ext))), ext, n, xy, at.P, at.w)
    img = np.where(dens / (np.quantile(dall, 0.6) + 1e-12) > 0.05, V[:, 0], np.nan).reshape(n, n)
    im = ax.imshow(img, origin="lower", extent=ext, aspect="auto", cmap=R._cmap("heat"), vmin=0, vmax=1,
                   interpolation="bilinear", zorder=0)
    _boundaries(ax, Pg.argmax(1).reshape(n, n), (dall / (np.quantile(dall, 0.6) + 1e-12)).reshape(n, n) > 0.15,
                ext, _T()["fg"], 0.55)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    C = av["centre_xy"][:, :2]
    live = [a for a in np.argsort(-at.share) if at.count[a] > 0]
    off = _place_labels(ax, C, live, gap=(0.16, 0.05))
    for a in live:
        ax.annotate(R._short(at.names[a], 20), tuple(C[a]), xytext=off[a], textcoords="offset points", ha="center",
                    fontsize=7, color=_T()["fg"], weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=_T()["bg"], ec="none", alpha=0.7))
    for mk in il["marks"]:
        p = mk["xy"][:2]
        ax.scatter([p[0]], [p[1]], marker="+" if mk["sign"] > 0 else "_", s=48,
                   color=_T()["passes"]["exclusion"] if mk["sign"] > 0 else _T()["passes"]["foundational"],
                   linewidths=1.8, zorder=5)
    g = il["g"][:2]
    span = 0.25 * min(ext[1] - ext[0], ext[3] - ext[2])
    gn = np.linalg.norm(g)
    if gn > 1e-12:
        v = g / gn * span
        o = np.array([(ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2])
        ax.annotate("", o + v, o, arrowprops=dict(arrowstyle="-|>", color=_T()["fg"], lw=2.0), zorder=6)
        ax.annotate(f"g  ({100 * il['in_plane']:.0f}% in plane, |g|/4 = {il['g_norm'] / 4:.2f})", tuple(o + v),
                    xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=_T()["fg"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=_T()["bg"], ec="none", alpha=0.75))
    fig.colorbar(im, ax=ax, label="worth", shrink=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    _title(fig, title or f"{il['item']} · worth over the atlas · + kept/added · − removed")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ------------------------------------------------------------------ stat response per archetype
def stat_curves(tri, at: Atlas, item: str, stat: Optional[str] = None, rows: int = 30,
                values: Sequence[int] = tuple(range(0, 101, 5))) -> dict:
    """B6: mean worth of the item as one stat (pre-shrine, highest and final) is
    set to each value, over up to `rows` builds of each archetype."""
    from tri.analysis.facts import single_requirement
    t = tri.catalog.index[item]
    stats = list(at.bm.catalog.stats)
    req = single_requirement(tri.catalog.items[item])
    if stat is None:
        stat = req[0] if req and req[0] in stats else None
    if stat is None:                                  # the stat the item's worth follows most
        fi, q = at.bm.fi.astype(float), at.Q[:, t]
        sd = fi.std(0)
        cor = np.array([abs(np.corrcoef(fi[:, s], q)[0, 1]) if sd[s] > 0 else 0 for s in range(fi.shape[1])])
        stat = stats[int(np.nan_to_num(cor).argmax())]
    if stat not in stats:
        raise ValueError(f"unknown stat {stat!r}")
    si = stats.index(stat)
    from tri.game import priors
    bmk = priors.benchmark_of(item)
    bmk = bmk if bmk and bmk["stat"] == stat else None
    if bmk:                                            # resolve the steep step (math §2.9)
        k = int(bmk.get("at", 77))
        values = sorted(set(values) | set(range(k - 3, k + 5)))
    curves = []
    for a in range(len(at.names)):
        m = np.flatnonzero(at.members(a))
        if len(m) < 5:
            continue
        m = m[:rows]
        base = at.bm.subset(m)
        ys = []
        for v in values:
            bm = at.bm.subset(m)
            for arr in (bm.hi, bm.fi, bm.pr):
                arr[:, si] = v
            ys.append(float(sigmoid(tri.space.logits_rows(bm, prior=True)[:, t]).mean()))
        ys = np.array(ys)
        cross = next((v for v, y in zip(values, ys) if y >= 0.5), None)
        cross = cross if cross is not None and ys[0] < 0.5 else None      # a real crossing only
        curves.append(dict(archetype=at.names[a], index=a, worth=ys.tolist(), crosses=cross))
        del base
    return dict(item=item, stat=stat, values=list(values), requirement=req[1] if req and req[0] == stat else None,
                curves=curves, benchmark=dict(at=bmk.get("at", 77), why=bmk.get("why", "")) if bmk else None)


def plot_stats(sc: dict, title: Optional[str] = None, n_archetypes: Optional[int] = None):
    plt = R._plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cols = _colours(n_archetypes or (max((c["index"] for c in sc["curves"]), default=0) + 1))
    for c in sc["curves"]:
        ax.plot(sc["values"], c["worth"], color=cols[c["index"]], lw=1.8, label=R._short(c["archetype"], 24))
        if c["crosses"] is not None:
            ax.scatter([c["crosses"]], [0.5], color=cols[c["index"]], s=28, zorder=4, edgecolors=_T()["fg"],
                       linewidths=0.5)
    if sc["requirement"] is not None:
        ax.axvline(sc["requirement"], color=_T()["passes"]["review"], ls="--", lw=1.2)
        ax.text(sc["requirement"], 1.01, f" requirement {sc['requirement']}", fontsize=7.5,
                color=_T()["passes"]["review"], transform=ax.get_xaxis_transform())
    if sc.get("benchmark"):
        k = sc["benchmark"]["at"]
        ax.axvline(k, color=_T()["star"], ls="-.", lw=1.1)
        ax.text(k, 1.01, f" {sc['benchmark']['why'] or 'benchmark'} {k}", fontsize=7.5, color=_T()["star"],
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=_T()["muted"], ls=":", lw=0.8)
    ys = [y for c in sc["curves"] for y in c["worth"]]
    lo = max(0.0, np.floor((min(ys) - 0.05) * 10) / 10) if ys else 0.0     # v2.21: no dead band below the curves
    ax.set_ylim(lo, 1); ax.set_xlim(min(sc["values"]), max(sc["values"]))
    ax.set_xlabel(sc["stat"], fontsize=8); ax.set_ylabel("worth", fontsize=8); ax.tick_params(labelsize=7)
    if sc["curves"]:
        ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.6)
    else:
        ax.text(0.5, 0.5, "no archetype has enough builds", transform=ax.transAxes, ha="center",
                color=_T()["muted"], fontsize=9)
    _title(fig, title or f"{sc['item']} against {sc['stat']} · per archetype")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ------------------------------------------------------------------ meta over time


# ------------------------------------------------------------------ replay
def plot_replay(pq: dict, vg: dict, cov: dict, lr: dict, title: Optional[str] = None):
    """D: prequential curve (memory vs field), claim variogram (same vs other
    archetype, reach line), coverage of the pool by reviews (radius eps)."""
    plt = R._plt()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    T = _T()
    ax = axs[0]
    if pq.get("reviews"):
        k = np.arange(1, pq["reviews"] + 1)
        ax.plot(k, pq["accuracy"], color=R._pass_colour("central"), lw=1.8, label="memory + field")
        ax.plot(k, pq["field"], color=R._pass_colour("foundational"), lw=1.4, label="field only")
        ax.plot(k, pq["rolling"], color=R._pass_colour("review"), lw=0.9, alpha=0.8, label="rolling")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1); ax.set_xlabel("reviews (time order)", fontsize=8); ax.set_ylabel("claims predicted", fontsize=8)
    ax.set_title("prequential replay", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    ax = axs[1]
    if vg.get("pairs"):
        e = np.asarray(vg["edges"]); c = (e[:-1] + e[1:]) / 2
        ax.plot(c, vg["agree"], "o-", color=T["fg"], ms=3, lw=1.2, label="all")
        ax.plot(c, vg["same"], "o-", color=R._pass_colour("central"), ms=3, lw=1.2, label="same archetype")
        ax.plot(c, vg["cross"], "o-", color=R._pass_colour("foundational"), ms=3, lw=1.2, label="other archetype")
        ax.axvline(vg["reach"], color=R._pass_colour("review"), ls="--", lw=1)
        ax.legend(fontsize=7, loc="lower left")
    lo = np.nanmin([np.nanmin(np.asarray(vg[k], float)) for k in ("agree", "same", "cross")]) if vg.get("pairs") else 0
    ax.set_ylim(min(0.5, float(np.nan_to_num(lo, nan=0.5)) - 0.05), 1.02)
    ax.set_xlabel("concept distance", fontsize=8); ax.set_ylabel("claim agreement", fontsize=8)
    ax.set_title("claim variogram", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    ax = axs[2]
    d = np.asarray(cov.get("distances", []), float)
    d = d[np.isfinite(d)]
    if len(d):
        ax.hist(d, bins=30, color=R._pass_colour("luxury"), alpha=0.9)
        ax.axvline(cov["radius"], color=R._pass_colour("central"), lw=1.4)
        ax.text(cov["radius"], 1.0, f" ε {cov['radius']:.2f}", transform=ax.get_xaxis_transform(), fontsize=7.5,
                color=R._pass_colour("central"), ha="right", va="bottom")
    ax.set_xlabel("distance to the nearest review", fontsize=8); ax.set_ylabel("builds", fontsize=8)
    ax.set_title("coverage", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    fin = pq.get("final") or {}
    lt = f" · low-rank p {lr['p']:.2f}" if lr.get("p") is not None else ""
    acc = f"{fin.get('accuracy', 0):.2f} vs field {fin.get('field', 0):.2f}" if fin.get("accuracy") is not None else "-"
    _title(fig, title or f"Review stream · replay {acc}{lt}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ------------------------------------------------------------------ lookup (v2.21)
def plot_lookup(lk: dict, n_archetypes: Optional[int] = None, names: Sequence[str] = (), title: Optional[str] = None):
    """One item's worth per archetype (bars, sorted), its pick rate as a tick on
    each bar, hole / gem archetypes marked."""
    plt = R._plt()
    rows = lk.get("worth") or []
    hgt = max(1.6, 0.34 * len(rows) + 0.9)
    fig, ax = plt.subplots(figsize=(7.6, hgt))
    T = _T()
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "no archetype has builds that can take it", ha="center", va="center", color=T["muted"],
                fontsize=9, transform=ax.transAxes)
    else:
        cols = _colours(n_archetypes or len(rows))
        idx = {n: i for i, n in enumerate(names)} if names else {r["archetype"]: i for i, r in enumerate(rows)}
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [r["worth"] for r in rows], color=[cols[idx.get(r["archetype"], 0) % len(cols)] for r in rows],
                height=0.62, alpha=0.9)
        ax.scatter([r["pick"] for r in rows], y, marker="|", s=160, color=T["fg"], zorder=4, linewidths=1.4)
        for yi, r in zip(y, rows):
            tag = " hole" if r["archetype"] in lk.get("holes", []) else " gem" if r["archetype"] in lk.get("gems", []) else ""
            if tag:
                ax.text(max(r["worth"], r["pick"]) + 0.02, yi, tag, va="center", fontsize=7, color=T["star"])
        ax.set_yticks(y); ax.set_yticklabels([R._short(r["archetype"], 26) for r in rows], fontsize=7.5)
        ax.axvline(0.5, color=T["muted"], ls=":", lw=0.8)
        ax.set_xlim(0, 1.08); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("worth  (| pick rate)", fontsize=8); ax.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    _title(fig, title or f"{lk['item']} · worth per archetype")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / hgt))
    return fig
