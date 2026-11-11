"""Fixed logit priors the pretrain cannot learn (v2.21, math §2.9).

Two known effects are added to the field's logits at inference; neither is
fitted, so they never move the space:

  - stat benchmarks (tri/data/benchmarks.json): past a benchmark k on stat s
    an item's worth rises steeply (guardbreaking mantras past 77):
        Delta_t(x) = kappa * sigma((x_s - k) / w),  x_s = max(pre-shrine, final)
  - weapon redundancy (tri/weapons.py scale): worth is multiplied by the share
    c of equipped weapons that leave the item useful,
        L' = logit(c * sigma(L))        (c = 0 is a lock, Catalog context)
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

from tri.data import path as _data_path


@lru_cache(maxsize=1)
def benchmarks() -> dict:
    p = _data_path("benchmarks.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(gain=0.0, width=1.0, benchmarks=[])


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def benchmark_of(item: str) -> Optional[dict]:
    return next((dict(b) for b in benchmarks().get("benchmarks", []) if b["item"] == item), None)


def stat_offsets(catalog, pre: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """[B, N] benchmark logit offsets for stat rows pre, fin [B, S]."""
    pre, fin = np.atleast_2d(pre).astype(float), np.atleast_2d(fin).astype(float)
    out = np.zeros((len(fin), len(catalog.ids)))
    B = benchmarks()
    k_, w = float(B.get("gain", 0.0)), max(float(B.get("width", 1.0)), 1e-6)
    if k_ == 0:
        return out
    for b in B.get("benchmarks", []):
        t, s = catalog.index.get(b["item"]), catalog.stat_index.get(b["stat"])
        if t is None or s is None:
            continue
        x = np.maximum(pre[:, s], fin[:, s])
        out[:, t] += k_ * _sig((x - float(b.get("at", 77))) / float(b.get("width", w)))
    return out


def weapon_scale(catalog, weapons: Sequence[str]) -> np.ndarray:
    """[N] worth multipliers c_t from the equipped weapons (1 where no rule)."""
    from tri.game import weapons as W
    c = np.ones(len(catalog.ids))
    if not weapons:
        return c
    for k in W.scaled_items():
        name = next((i for i in catalog.ids if W._loose(i) == k), None)
        if name is not None:
            c[catalog.index[name]] = W.scale(name, weapons)
    return c


def apply(L: np.ndarray, catalog, pre, fin, weapons: Sequence[str] = ()) -> np.ndarray:
    """Logits with both priors: L + Delta, then logit(c sigma(.))."""
    L = np.asarray(L, float) + (stat_offsets(catalog, pre, fin)[0] if np.ndim(L) == 1
                                else stat_offsets(catalog, pre, fin))
    c = weapon_scale(catalog, weapons)
    if (c < 1).any():
        p = np.clip(c * _sig(L), 1e-9, 1 - 1e-9)
        L = np.log(p) - np.log1p(-p)
    return L


def stat_rows(catalog, builds) -> tuple:
    S = len(catalog.stats)
    pre, fin = np.zeros((len(builds), S)), np.zeros((len(builds), S))
    for r, b in enumerate(builds):
        for s, v in (b.pre_stats or b.final_stats or {}).items():
            if s in catalog.stat_index:
                pre[r, catalog.stat_index[s]] = v
        for s, v in (b.final_stats or {}).items():
            if s in catalog.stat_index:
                fin[r, catalog.stat_index[s]] = v
    return pre, fin


def names() -> List[str]:
    return [b["item"] for b in benchmarks().get("benchmarks", [])]


def adjust_q(q: np.ndarray, catalog, build) -> np.ndarray:
    """Worth (local pick rates) of one build with both priors, on the logit."""
    from tri.game import weapons as W
    pre, fin = stat_rows(catalog, [build])
    d = stat_offsets(catalog, pre, fin)[0]
    c = weapon_scale(catalog, W.equipped(build))
    if not d.any() and (c >= 1).all():
        return q
    p = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    p = _sig(np.log(p) - np.log1p(-p) + d)
    return np.where(np.asarray(q) > 0, c * p, q)
