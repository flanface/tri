"""Known talent groups and the context a selection rule can see (math §6.12).

Groups come from game data only. Nothing here is fitted or inferred.

    G(c)   the talents of game category c (pocamind "category", e.g.
           Flamecharmer, Bruiser). Generic categories (Equipment, Weapon,
           Miscellaneous, Innate, Outfit) are not families and are left out.
    D(k)   the derivation tree of a core talent k: k and every talent that
           needs it, directly or through another member (prerequisites and
           any-of prerequisites). k is core when at least `min_derived`
           (2) talents need it directly.

A group literal counts a build's picks in the group other than the rule's
own item:  |G(c)| >= n,  |D(k)| >= n.

Rule context tokens (kind "context"):
    oath:<v>  murmur:<v>  origin:<v>   the planner fields (Build.meta, or the
                                       corpus lock signature)
    weapon:<Heavy|Medium|Light>        the weapon class the build records
                                       (Build.meta weapon) or else the class
                                       of its highest weapon stat (> 0)
    archetype:<key>                    the archetype the locators read at the
                                       build's place in the concept space
                                       (tri.reviews.locators), family key
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from tri.core.types import Kind

GENERIC_CATEGORIES = frozenset({"equipment", "weapon", "miscellaneous", "innate", "outfit", "unknown"})
WEAPON_STATS = (("HeavyWeapon", "Heavy"), ("MediumWeapon", "Medium"), ("LightWeapon", "Light"))
PLANNER_PREFIXES = ("oath:", "murmur:", "origin:")


def group_name(kind: str, key: str) -> str:
    return f"{kind}({key})"


def parse_group(name: str) -> Tuple[str, str]:
    """"G(Flamecharmer)" -> ("G", "Flamecharmer")."""
    k, rest = name.split("(", 1)
    return k, rest[:-1]


def req_stats(item) -> Set[str]:
    """Stats named by an item's known requirements (all forms)."""
    out = set(item.requires or {})
    for d in item.requires_any or []:
        out |= set(d)
    for cl in item.clauses or []:
        for a in cl:
            out |= set(a.get("stats") or [])
    return out


def derived_groups(catalog, min_size: int = 3, min_derived: int = 2, categories: bool = True,
                   trees: bool = True) -> Dict[str, FrozenSet[str]]:
    """{group name: members} from the catalog (talents only)."""
    tal = [i for i in catalog.ids if catalog.items[i].kind == Kind.TALENT]
    out: Dict[str, FrozenSet[str]] = {}
    if categories:
        by: Dict[str, Set[str]] = {}
        for i in tal:
            for t in catalog.items[i].tags or []:
                if t.startswith("category:"):
                    c = t[9:]
                    if c.casefold() not in GENERIC_CATEGORIES:
                        by.setdefault(c, set()).add(i)
        for c, m in sorted(by.items()):
            if len(m) >= min_size:
                out[group_name("G", c)] = frozenset(m)
    if trees:
        need: Dict[str, Set[str]] = {}                 # k -> talents that need k directly
        for i in tal:
            it = catalog.items[i]
            for p in list(it.prerequisites or []) + [x for g in it.prerequisites_any or [] for x in g]:
                if p in catalog.index and catalog.items[p].kind == Kind.TALENT:
                    need.setdefault(p, set()).add(i)
        for k in sorted(need):
            if len(need[k]) < min_derived:
                continue
            seen, todo = {k}, list(need[k])
            while todo:
                x = todo.pop()
                if x not in seen:
                    seen.add(x); todo += list(need.get(x, ()))
            if len(seen) >= min_size:
                out[group_name("D", k)] = frozenset(seen)
    return out


def grounded_stats(catalog, item: str, groups: Dict[str, FrozenSet[str]]) -> Set[str]:
    """Stats a rule on item may test (PassConfig.grounded): the stats of its own
    requirements, of its prerequisites and of the members of its groups."""
    it = catalog.items[item]
    names = {item} | set(it.prerequisites or []) | {x for g in it.prerequisites_any or [] for x in g}
    for m in groups.values():
        if item in m:
            names |= m
    out: Set[str] = set()
    for n in names:
        if n in catalog.items:
            out |= req_stats(catalog.items[n])
    return out & set(catalog.stats)


def _closure(catalog) -> Dict[str, FrozenSet[str]]:
    """item -> its dependants, for every item, computed once per catalog state
    (reverse prerequisite graph, one DFS per item; v3.1 memo)."""
    key = (len(catalog.ids), id(catalog.items))
    hit = catalog.__dict__.get("_dependants")
    if hit is not None and hit[0] == key:
        return hit[1]
    need: Dict[str, Set[str]] = {}
    for j in catalog.ids:
        o = catalog.items[j]
        for p in list(o.prerequisites or []) + [x for g in o.prerequisites_any or [] for x in g]:
            need.setdefault(p, set()).add(j)
    out: Dict[str, FrozenSet[str]] = {}
    for item in need:
        seen: Set[str] = set()
        todo = list(need[item])
        while todo:
            x = todo.pop()
            if x not in seen and x != item:
                seen.add(x); todo += list(need.get(x, ()))
        out[item] = frozenset(seen)
    catalog.__dict__["_dependants"] = (key, out)
    return out


def dependants(catalog, item: str) -> Set[str]:
    """Items that need item, directly or through another dependant (all-of or
    any-of prerequisites). Picking one says the build has (or nearly always
    has) item: game data, not a selection rule (v3.0.1)."""
    return set(_closure(catalog).get(item, ()))


def tautological_groups(catalog, item: str, groups: Dict[str, FrozenSet[str]],
                        deps: Optional[Set[str]] = None) -> Set[str]:
    """Groups a literal on item may not count (v3.0.1): D(item) itself, and
    every group whose members other than item all need item."""
    deps = dependants(catalog, item) if deps is None else deps
    out = set()
    for g, m in groups.items():
        rest = set(m) - {item}
        if g == group_name("D", item) or (rest and rest <= deps):
            out.add(g)
    return out


def related_items(catalog, item: str, groups: Dict[str, FrozenSet[str]]) -> Set[str]:
    """Items a rule on item may name (PassConfig.anchors = "related"): its
    prerequisites and the members of its groups. Its dependants are left out
    (v3.0.1): "y ⇒ item" with y needing item is a prerequisite, not a rule."""
    it = catalog.items[item]
    out = set(it.prerequisites or []) | {x for g in it.prerequisites_any or [] for x in g}
    for m in groups.values():
        if item in m:
            out |= m
    out -= dependants(catalog, item)
    out.discard(item)
    return {x for x in out if x in catalog.index and x not in set(it.exclusive_with or [])}


# ---------------------------------------------------------------- context
def weapon_class(hi: Dict[str, int], recorded: Iterable[str] = ()) -> Optional[str]:
    """Heavy / Medium / Light: recorded class first, else the highest weapon stat."""
    rec = [w for w in recorded if w in ("Heavy", "Medium", "Light")]
    if rec:
        return rec[0]
    best, bv = None, 0
    for s, c in WEAPON_STATS:
        v = int(hi.get(s, 0) or 0)
        if v > bv:
            best, bv = c, v
    return best


def rule_tokens(b, archetype: Optional[str] = None) -> Set[str]:
    """Context tokens of a build for selection rules (see module docstring)."""
    from tri.core.context import build_context
    from tri.reviews.authority import build_weapons
    toks = {t for t in build_context(b)[0] if t.startswith(PLANNER_PREFIXES)}
    w = weapon_class(b.highest_stats, build_weapons(b))
    if w:
        toks.add(f"weapon:{w}")
    if archetype:
        toks.add(f"archetype:{archetype}")
    return toks


def corpus_tokens(bm, catalog, archetypes: Optional[Sequence[Optional[str]]] = None
                  ) -> Tuple[List[str], np.ndarray]:
    """(names, [n, K] float32) rule context tokens of every corpus row: the
    planner fields of its lock signature (or its context tokens), its weapon
    class from stats, and its archetype label (None = unlocated)."""
    n = len(bm)
    per: List[Set[str]] = [set() for _ in range(n)]
    lk, lkv = getattr(bm, "lk", None), list(getattr(bm, "lk_vocab", []) or [])
    if lk is not None and lkv:
        sig = [{t for t in v.split("||")[0].split("|") if t.startswith(PLANNER_PREFIXES)} for v in lkv]
        for r in range(n):
            per[r] |= sig[int(lk[r])]
    if bm.T is not None and bm.T.shape[1]:
        T = bm.T.tocsr()
        for r in range(n):
            for j in T.indices[T.indptr[r]:T.indptr[r + 1]]:
                t = bm.tokens[j]
                if t.startswith(PLANNER_PREFIXES):
                    per[r].add(t)
    six = {s: i for i, s in enumerate(catalog.stats)}
    cols = [(six[s], c) for s, c in WEAPON_STATS if s in six]
    if cols:
        W = np.stack([bm.hi[:, i] for i, _ in cols], 1)
        top = W.argmax(1); has = W.max(1) > 0
        for r in np.flatnonzero(has):
            per[r].add(f"weapon:{cols[int(top[r])][1]}")
    if archetypes is not None:
        for r, a in enumerate(archetypes):
            if a:
                per[r].add(f"archetype:{a}")
    names = sorted(set().union(*per)) if per else []
    ix = {t: i for i, t in enumerate(names)}
    M = np.zeros((n, len(names)), np.float32)
    for r, ts in enumerate(per):
        for t in ts:
            M[r, ix[t]] = 1.0
    return names, M


def group_counts(items: Iterable[str], groups: Dict[str, FrozenSet[str]]) -> Dict[str, int]:
    s = set(items)
    return {g: len(s & m) for g, m in groups.items()}
