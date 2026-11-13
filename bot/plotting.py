"""/plot concept, atlas and worth: one transparent 3-D PNG each, in the style of
assets/atlas_export1.png (tri.views.figure). Every option is a command
parameter; nothing is interactive.

    concept  PCA axes 1-2-3 of the concept space, groups = space regions
    atlas    PCA axes 1-2-3 of the archetype atlas, groups = archetypes
    worth    PCA axes 4-5-6 of the worth space (finer luxury worth); requested
             items are vectors from the dense origin, with their cosines

Each function returns ([(filename, png)], text).
"""
from __future__ import annotations

import threading
from dataclasses import replace
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:                                   # pragma: no cover
    from .service import ModelService

PLOT_LOCK = threading.Lock()            # matplotlib's state is not thread-safe
CONCEPT_DIMS = (0, 1, 2)
ATLAS_DIMS = (0, 1, 2)
WORTH_DIMS = (3, 4, 5)
WORTH_K = 9
STYLES = ("dots", "fields")


def png_bytes(fig, dpi: int = 110) -> bytes:
    """PNG of a matplotlib figure with trimmed margins (for the /archetype,
    /facts and graphic views); closes the figure."""
    import io

    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return buf.getvalue()


def _style(style: str) -> str:
    if style not in STYLES:
        raise ValueError("style is dots or fields")
    return style


def _concept_map(svc: "ModelService", sample: int) -> dict:
    from tri.views import plots as render
    key = ("pca", id(svc.tri.space))
    cache = getattr(svc, "_plot_cm", None)
    if not isinstance(cache, dict):
        cache = svc._plot_cm = {}
    if key not in cache:
        cache[key] = render.concept_map(svc.tri, sample=sample, method="pca", k=max(3, min(9, int(svc.tri.space.cfg.dim))))
    return cache[key]


def _mark_lines(svc: "ModelService", marks: Sequence[tuple], U: np.ndarray) -> List[str]:
    out = []
    for (lbl, b), u in zip(marks, U):
        rd = svc.tri.space.reading(u).to_dict()
        regs = ", ".join(f"{r['name'][:28]} {100 * r['membership']:.0f}%" for r in rd.get("regions", [])[:3]) or "none"
        arch = svc.tri.archetype(b) if svc.tri.locators else {}
        a = arch.get("archetype") or (f"nearest {arch.get('best')} (not located)" if arch.get("best") else "no locators")
        out.append(f"{str(lbl)[:40]}: regions {regs}; archetype {a}")
    return out


def plot_concept(svc: "ModelService", marks: Sequence[tuple] = (), locators: bool = True, sample: int = 3000,
                 style: str = "dots", view: Tuple[float, float] = (22, -58)) -> tuple:
    """The concept space on PCA axes 1-2-3: builds as dots (or region
    polytopes), region centres, locators as diamonds, marks [(label, Build)] as
    stars. text: each mark's regions and archetype."""
    from tri.views import figure as F
    from tri.views import plots as render
    style = _style(style)
    d = list(CONCEPT_DIMS)
    with PLOT_LOCK:
        with svc.lock:
            cm = _concept_map(svc, sample)
            text: List[str] = []
            stars = []
            if marks:
                bs = [replace(b, build_id=str(lbl)[:40]) for lbl, b in marks]
                hl = render.place(svc.tri, bs, cm["projection"], partial=True)
                stars = [(str(lbl), p[d]) for (lbl, _), p in zip(marks, hl["xy"])]
                text = _mark_lines(svc, marks, hl["U"])
            locs = list(svc.tri.locators) if locators else []
            lp = render.place(svc.tri, [l.build for l in locs], cm["projection"])["xy"][:, d] if locs else None
        sc = F.Scene(P=cm["xy"][:, d], group=np.asarray(cm["region"]), names=list(cm["region_names"]), style=style,
                     centres=np.asarray(cm["centre_xy"])[:, d] if len(cm["centre_xy"]) else None,
                     locators=lp, stars=stars)
        return [("concept_1-2-3.png", F.png(F.draw(sc, view)))], "\n".join(text)


def plot_atlas(svc: "ModelService", style: str = "fields", candidates: bool = True, locators: bool = True,
               marks: Sequence[tuple] = (), view: Tuple[float, float] = (22, -58)) -> tuple:
    """The archetype atlas on PCA axes 1-2-3: placed builds grouped by
    archetype (polytopes or dots), archetype centres, locators, candidate
    archetypes and marks as stars. text: archetypes, located share, candidates."""
    from tri.views import figure as F
    from tri.views import plots as render

    from .insight import _atlas
    style = _style(style)
    d = list(ATLAS_DIMS)
    with PLOT_LOCK:
        with svc.lock:
            c = _atlas(svc)
            at, av = c.atlas, c.view
            pl = np.asarray(at.placed, bool)
            stars = [(str(x["id"]), np.asarray(x["xy"])[d]) for x in (av.get("candidates") or [])] if candidates else []
            text: List[str] = []
            if marks:
                bs = [replace(b, build_id=str(lbl)[:40]) for lbl, b in marks]
                hl = render.place(svc.tri, bs, av["projection"], partial=True)
                stars += [(str(lbl), p[d]) for (lbl, _), p in zip(marks, hl["xy"])]
                text = _mark_lines(svc, marks, hl["U"])
        located = float(at.w[pl].sum() / max(at.w.sum(), 1e-12))
        share = ", ".join(f"{n} {100 * s:.0f}%" for n, s in sorted(zip(at.names, at.share), key=lambda t: -t[1]))
        head = f"{len(at.names)} archetypes · {100 * located:.0f}% located · {len(c.cands)} candidates\n{share}"
        sc = F.Scene(P=av["xy"][pl][:, d], group=np.asarray(at.top)[pl], names=list(at.names), style=style,
                     centres=np.asarray(av["centre_xy"])[:, d],
                     locators=np.asarray(av["loc_xy"])[:, d] if locators and len(av["loc_xy"]) else None,
                     stars=stars)
        return [("atlas_1-2-3.png", F.png(F.draw(sc, view)))], "\n".join([head] + text)


def worth_space(svc: "ModelService", relative: bool = False, sample: int = 2000) -> dict:
    """The worth space (k = WORTH_K axes), cached until reviews, locators or the space change."""
    from tri.views import plots as render
    key = (bool(relative), len(svc.tri.reviews), len(svc.tri.locators), id(svc.tri.space))
    cache = getattr(svc, "_plot_ws", None)
    if cache is None or cache[0] != key:
        ws = render.worth_space(svc.tri, sample=sample, relative=relative, min_rate=0.005, k=WORTH_K)
        svc._plot_ws = cache = (key, ws)
    return cache[1]


def worth_vectors(ws: dict, items: Sequence[str], dims: Sequence[int] = WORTH_DIMS) -> dict:
    """origin: centre of the dense cluster of item points on `dims`; per
    requested item its vector from the origin; cos: their pairwise cosines;
    nearest: per item the three items whose vectors point most alike."""
    from tri.views import figure as F
    X = np.asarray(ws["xy"])[:, list(dims)]
    o = F.dense_origin(X)
    V = X - o
    idx = [ws["items"].index(i) for i in items]
    C = F.cosines(V)
    nearest = {}
    for i, j in zip(items, idx):
        order = [t for t in np.argsort(-C[j]) if t != j][:3]
        nearest[i] = [(ws["items"][t], float(C[j, t])) for t in order]
    return dict(origin=o, X=X, idx=idx, cos=C[np.ix_(idx, idx)], nearest=nearest)


def plot_worth(svc: "ModelService", items: Sequence[str] = (), relative: bool = False, sample: int = 2000,
               view: Tuple[float, float] = (22, -58)) -> tuple:
    """Items on worth axes 4-5-6 (talents and mantras in two muted colours). Requested items are
    arrows from the origin (the dense cluster); text: cosine similarity of
    every pair of them and each one's closest items by cosine."""
    from tri.views import figure as F
    with PLOT_LOCK:
        with svc.lock:
            ws = worth_space(svc, relative, sample)
        if np.asarray(ws["xy"]).shape[1] <= max(WORTH_DIMS):
            raise ValueError("the worth space has fewer than 6 axes")
        text: List[str] = []
        known = [i for i in items if i in ws["items"]]
        text += [f"{i}: rarely taken (mean worth under 0.5%), not drawn" for i in items if i not in ws["items"]]
        wv = worth_vectors(ws, known)
        kinds = sorted(set(ws["kind"]))
        group = np.array([kinds.index(k) for k in ws["kind"]])
        cols = F.colours(max(len(known), 1))
        sc = F.Scene(P=wv["X"], group=group, names=kinds, style="dots",
                     point_colour=[F.KIND_COLOURS.get(k, F.FAINT) for k in ws["kind"]],
                     origin=wv["origin"], point_size=6.0,
                     vectors=[(i, wv["X"][j], cols[n]) for n, (i, j) in enumerate(zip(known, wv["idx"]))])
        if len(known) > 1:
            text.append("cosine similarity from the origin:")
            for a in range(len(known)):
                for b in range(a + 1, len(known)):
                    text.append(f"  {known[a]} · {known[b]}: {wv['cos'][a, b]:+.2f}")
        for i in known:
            text.append(f"{i} points like: " + ", ".join(f"{n} ({c:+.2f})" for n, c in wv["nearest"][i]))
        return [("worth_4-5-6.png", F.png(F.draw(sc, view)))], "\n".join(text)
