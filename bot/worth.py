"""Worth-space and placement numbers for inline calls (no graphics).

    worth_space   the worth space (k = WORTH_K axes), cached per review/locator state
    worth_vectors items as vectors from the dense origin on axes 4-5-6, their cosines and nearest items
    worth_pair    cosine, angle, orthogonality and projections of two items
    place_lines   a build's regions and archetype, as text

The interactive concept, atlas and worth graphs are on the repository's GitHub
Pages site (bot.site).
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:                                   # pragma: no cover
    from .service import ModelService

WORTH_DIMS = (3, 4, 5)
WORTH_K = 9


def pair_lines(names: Sequence[str], C: np.ndarray, indent: str = "  ",
               norms: Optional[Sequence[float]] = None) -> List[str]:
    """Per pair: cosine, angle, orthogonality |sin theta| and, given the norms,
    each vector's projection length on the other (|a| cos theta on b, |b| cos theta on a)."""
    from tri.api import figure as F
    C = np.asarray(C, float)
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            c = float(C[a, b])
            s = (f"{indent}{names[a]} · {names[b]}: cos {c:+.2f} · angle {float(F.angle(c)):.0f}° · "
                 f"orthogonality {float(F.orthogonality(c)):.2f}")
            if norms is not None:
                s += f" · proj on {names[b]} {float(norms[a]) * c:+.3g} · proj on {names[a]} {float(norms[b]) * c:+.3g}"
            out.append(s)
    return out


def _concept_map(svc: "ModelService", sample: int) -> dict:
    from tri.api import plots as render
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


def place_lines(svc: "ModelService", marks: Sequence[tuple], sample: int = 3000) -> List[str]:
    """Per mark (label, Build): its regions and archetype."""
    from tri.api import plots as render
    with svc.lock:
        cm = _concept_map(svc, sample)
        bs = [replace(b, build_id=str(lbl)[:40]) for lbl, b in marks]
        hl = render.place(svc.tri, bs, cm["projection"], partial=True)
        return _mark_lines(svc, marks, hl["U"])


def worth_space(svc: "ModelService", relative: bool = False, sample: int = 2000) -> dict:
    """The worth space (k = WORTH_K axes), cached until reviews, locators or the space change."""
    from tri.api import plots as render
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
    from tri.api import figure as F
    X = np.asarray(ws["xy"])[:, list(dims)]
    o = F.dense_origin(X)
    V = X - o
    idx = [ws["items"].index(i) for i in items]
    C = F.cosines(V)
    nearest = {}
    for i, j in zip(items, idx):
        order = [t for t in np.argsort(-C[j]) if t != j][:3]
        nearest[i] = [(ws["items"][t], float(C[j, t])) for t in order]
    return dict(origin=o, X=X, idx=idx, cos=C[np.ix_(idx, idx)], nearest=nearest,
                norms=[float(np.linalg.norm(V[j])) for j in idx])


def worth_pair(svc: "ModelService", a: str, b: str, relative: bool = False, sample: int = 2000) -> dict:
    """Cosine, angle and orthogonality of two items' worth vectors (axes 4-5-6, from the dense origin)."""
    from tri.api import figure as F
    with svc.lock:
        ws = worth_space(svc, relative, sample)
    miss = [i for i in (a, b) if i not in ws["items"]]
    if miss:
        raise ValueError(", ".join(miss) + ": rarely taken (mean worth under 0.5%), no worth vector")
    wv = worth_vectors(ws, [a, b])
    c, (na, nb) = float(wv["cos"][0, 1]), wv["norms"]
    return dict(a=a, b=b, cos=c, angle=float(F.angle(c)), orthogonality=float(F.orthogonality(c)),
                norm_a=na, norm_b=nb, a_on_b=na * c, b_on_a=nb * c)
