"""Weapon-class focus (math §2.10, docs/record/changelog.md 22.32).

A build usually commits to one weapon class: its weapon talents are those of
one weapon type (Rapier, Pistol, Fist, ...). Worth is a pick rate among the
builds that can take an item, and a weapon talent can only be taken by builds
that wield its weapon, so every class's talents look worthy to a build that
records no weapon. A completion that fills by worth then takes Fist, Rapier
and Pistol talents together.

Focus fixes the class first and treats the other classes' weapon talents as
unavailable for the rest of the completion or analysis:

    A(b) = classes of the weapons b equips                       if any, else
           classes of the weapon talents b already has            if any, else
           classes of weapon talents a binding review adds/keeps  if any, else
           argmax_c  P(c | u(b))                                  (class prior)

The class prior is the pretraining corpus's weapon context: for each corpus
build, the classes of the weapons it records (its lock signature) and of the
weapon talents it picked; P(c | u) is the weighted share of the k nearest
labelled corpus builds in the concept space. When too few neighbours are
labelled, the class whose available talents are worth most here (mean of the
top 3) is chosen instead. Evidence (weapons, picks, reviews) may name several
classes (multi-weapon builds); a prediction names one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from tri.game import weapons as W

# weapon types that are not a class of their own: a shield or an exclusive
# weapon goes with any class (Rally, Turtle Shell keep their wield locks)
NOT_CLASS = frozenset({"Shield", "Exclusive"})
_SPLIT = re.compile(r"\s*(?:,|/|;|\|)\s*")


_NAMES: Dict[int, Set[str]] = {}


def class_names() -> Set[str]:
    """Weapon types of the weapon table that are classes."""
    key = id(W.table())
    if key in _NAMES:
        return _NAMES[key]
    out: Set[str] = set()
    for v in W.table()["weapons"].values():
        for t in _SPLIT.split(str(v.get("type") or "")):
            if t and t not in NOT_CLASS:
                out.add(t)
    _NAMES.clear()
    _NAMES[key] = out
    return out


def token_classes(v: str) -> Set[str]:
    """Classes named by a weapon value or wield token ('wield:Spear / Rifle')."""
    s = str(v).split(":", 1)[1] if str(v).lower().startswith(W.FIELD + ":") else str(v)
    return {c for p in _SPLIT.split(s) if (c := norm_class(p))}


def norm_class(v: str) -> Optional[str]:
    """'Fists' -> 'Fist', 'wield:Rapier' -> 'Rapier'; None if not a class."""
    s = str(v).split(":", 1)[1] if str(v).lower().startswith(W.FIELD + ":") else str(v)
    s = s.strip()
    names = {n.lower(): n for n in class_names()}
    if s.lower() in names:
        return names[s.lower()]
    if s.lower().endswith("s") and s[:-1].lower() in names:
        return names[s[:-1].lower()]
    return None


def item_classes(item: str, catalog=None) -> FrozenSet[str]:
    """Weapon classes an item needs (any one of them), from the game data's
    weaponType condition (catalog tag condition:weaponType=...) and the
    weapon clauses of data/weapons.json. Empty for an item of no class."""
    out: Set[str] = set()
    it = catalog.items.get(item) if catalog is not None else None
    for t in (getattr(it, "tags", None) or []):
        if str(t).startswith("condition:weaponType="):
            out |= token_classes(str(t).split("=", 1)[1])
    for cl in W.item_clauses(item):
        for tok in cl:
            out |= token_classes(tok)
    return frozenset(out)


_TABLES: Dict[int, Dict[str, FrozenSet[str]]] = {}


def class_table(catalog) -> Dict[str, FrozenSet[str]]:
    """item -> classes, for the catalog's items that have a class (cached)."""
    key = id(catalog)
    if key not in _TABLES:
        if len(_TABLES) > 8:
            _TABLES.clear()
        _TABLES[key] = {i: c for i in catalog.ids if (c := item_classes(i, catalog))}
    return _TABLES[key]


def weapon_classes(weapons: Sequence[str]) -> Set[str]:
    """Classes of equipped weapons (table names)."""
    T = W.table()["weapons"]
    out: Set[str] = set()
    for w in weapons:
        out |= token_classes(str((T.get(w) or {}).get("type") or ""))
    return out


def evidence(b, catalog, memory: Optional[dict] = None, binds=None) -> Tuple[Set[str], str]:
    """(classes, source) the build itself commits to: its weapons, else its
    weapon talents, else weapon talents a binding review adds or keeps."""
    ws = weapon_classes(W.equipped(b))
    if ws:
        return ws, "weapon"
    tab = class_table(catalog)
    have = set().union(*[tab[i] for i in b.items() if i in tab]) if any(i in tab for i in b.items()) else set()
    if have:
        return set(have), "picks"
    rv: Set[str] = set()
    for it, v in (memory or {}).items():
        if it in tab and v and v[0] in ("add", "keep") and (binds is None or binds(v)):
            rv |= tab[it]
    if rv:
        return rv, "review"
    return set(), ""


@dataclass
class ClassPrior:
    """P(class | concept coordinates) from the pretraining corpus's weapon context."""
    Z: np.ndarray                 # [n, d] coordinates of labelled corpus builds
    L: np.ndarray                 # [n, K] class indicators
    w: np.ndarray                 # [n] corpus weights
    classes: List[str]
    rows: int                     # corpus builds seen
    k: int = 150

    @staticmethod
    def from_matrix(bm, Z: np.ndarray, catalog, k: int = 150) -> Optional["ClassPrior"]:
        """Labels each corpus build with the classes of the weapons its lock
        signature records and of the weapon talents it picked. None if no
        build has a label."""
        from tri.model.matrix import parse_signature
        tab = class_table(catalog)
        names = sorted(class_names() | {c for v in tab.values() for c in v})
        cx = {c: j for j, c in enumerate(names)}
        n = len(bm)
        L = np.zeros((n, len(names)), bool)
        if getattr(bm, "lk", None) is not None and bm.lk_vocab:
            sig_cls = []
            for sig in bm.lk_vocab:
                toks, _ = parse_signature(sig)
                cs = set().union(*[token_classes(t) for t in toks if t.startswith(W.FIELD + ":")] or [set()])
                sig_cls.append([cx[c] for c in cs if c in cx])
            for code, cols in enumerate(sig_cls):
                if cols:
                    rows = np.flatnonzero(np.asarray(bm.lk) == code)
                    if len(rows):
                        L[np.ix_(rows, cols)] = True
        if tab:
            X = bm.X.tocsc()
            for it, cs in tab.items():
                t = catalog.index[it]
                rows = X.indices[X.indptr[t]:X.indptr[t + 1]]
                if len(rows):
                    for c in cs:
                        L[rows, cx[c]] = True
        lab = L.any(1)
        if not lab.any():
            return None
        from tri.model.worth import RowStack
        Zs = RowStack(Z, np.flatnonzero(lab)) if isinstance(Z, np.ndarray) and Z.ndim == 2 \
            else RowStack(np.asarray(Z, np.float64)[lab])                 # v3.7: rows of Z, not a copy
        return ClassPrior(Zs, L[lab], np.asarray(bm.w, np.float64)[lab], names, n, k)

    def share(self, u: np.ndarray) -> Tuple[Dict[str, float], int]:
        """({class: weighted share among the k nearest labelled corpus builds}, k used)."""
        u = np.asarray(u, np.float64)
        if isinstance(self.Z, np.ndarray):
            d2 = ((self.Z - u[None]) ** 2).sum(1)
        else:                                          # RowStack: no [n, D'] temporary per query (v3.7)
            if getattr(self, "_Z2", None) is None:
                self._Z2 = self.Z.sqnorm()
            d2 = np.maximum(self._Z2 - 2 * self.Z.dot(u) + u @ u, 0)
        k = min(self.k, len(d2))
        nb = np.argpartition(d2, k - 1)[:k] if k < len(d2) else np.arange(len(d2))
        w = self.w[nb]
        tot = float(w.sum()) or 1.0
        s = (self.L[nb] * w[:, None]).sum(0) / tot
        return {c: float(v) for c, v in zip(self.classes, s) if v > 0}, int(k)
