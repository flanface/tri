<<<<<<< HEAD
"""Triviality baselines for statements.
=======
"""Triviality baselines for statements (math §12.6).
>>>>>>> main

A rule or fact is stated only when it says more than the game data already
says. Pom's five baselines; each statement on the left is trivial when the
condition on the right holds:

  x needs  s >= k                 k is x's requirement in s (within one grid step)
  x is common in A                x is common everywhere (P(x) >= 0.7 outside A)
  x and y are not taken together  x and y are mutually exclusive (or one locks the other)
  x follows y                     y is x's prerequisite (either way)
  A and B differ in x             x is locked by oath, murmur or origin and A, B
                                  have different ones (the lock explains it)

Group literals. A literal counts the picks of a known group other
than the rule's own item x: G' = G − {x}, |G'| = m, room ρ = the most members
of G' a build can hold together (exclusive pairs, the rule's own absences).

  |G| ≥ n, n ≤ 0                  vacuous (always holds)
  |G| ≥ n, n > ρ                  dead (never holds)
  |G| < n, n ≤ 0 or n ≤ forced    dead
  |G| ≥ m, all compatible         saturated: ≡ ∧ G' (every member), e.g. a
                                  category of x and two others: |G| ≥ 2 ≡ y ∧ z
  |G| < 1                         ≡ ∧ ¬G'
  |D(k)| ≥ 1, k ≠ x               ≡ k (every member is k or needs k)
  |G| ≥ n ≥ 1                     forces every item all members of G' are or need

A literal is read as the item literals it is equivalent to, and the item
baselines above then apply. cull() makes a rule inactive (status "trivial")
only when that changes nothing a build can see:

  dead                            it can never fire
  take x, conditions force x      x is the prerequisite of a forced item: x is
                                  in every build the rule covers (e.g. |D(x)| ≥ 1 ⇒ x)
  never x, conditions forbid x    a forced item excludes x
Rules that are only obvious (k = req, y ∈ pre(x), ...) stay active and are
hidden from listings, as before. Authored rules are never culled.

trivial_fact(fact, catalog) and trivial_rule(rule, catalog) return the
baseline that makes a statement trivial (a short string) or None.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

GRID = 5
COMMON = 0.7
LOCK_FIELDS = ("oath", "murmur", "origin")

BASELINES = (
    ("x ⇒ s ≥ k", "k = req_s(x)"),
    ("P(x | A) high", "P(x | ¬A) ≥ 0.7"),
    ("x ⊥ y", "x ⊕ y (mutually exclusive)"),
    ("y ⇒ x", "y ∈ pre(x)"),
    ("A ↔ B: x", "x ∈ lock(oath, murmur, origin), lock_A ≠ lock_B"),
)


def _req(catalog, item: str, stat: str) -> Optional[float]:
    it = catalog.items.get(item)
    if it is None:
        return None
    v = (it.requires or {}).get(stat)
    return float(v) if v is not None else None


def related(catalog, x: str, y: str) -> Optional[str]:
    """"exclusive" or "prerequisite" when the game data links x and y."""
    a, b = catalog.items.get(x), catalog.items.get(y)
    if a is None or b is None:
        return None
    if y in a.exclusive_with or x in b.exclusive_with:
        return "exclusive"
    if (y in a.prerequisites or x in b.prerequisites
            or any(y in g for g in (a.prerequisites_any or [])) or any(x in g for g in (b.prerequisites_any or []))):
        return "prerequisite"
    return None


def lock_locked(catalog, item: str) -> bool:
    """x is locked by oath, murmur or origin (a context clause on those fields)."""
    try:
        cls = catalog.enforced_clauses(item)
    except Exception:
        return False
    return any(any(str(t).split(":", 1)[0].lower() in LOCK_FIELDS for t in cl) for cl in cls)


def trivial_fact(f, catalog) -> Optional[str]:
    n, its, fam = f.numbers or {}, list(f.items or []), f.family
    if fam == "threshold" and its:
        r = n.get("requirement")
        if r is not None and abs(float(n.get("threshold", 0)) - float(r)) <= GRID:
            return "k = req"
    if fam == "defining" and n.get("others", 0) >= COMMON:
        return "common everywhere"
    if fam in ("synergy", "anti-synergy") and len(its) >= 2:
        k = related(catalog, its[0], its[1])
        if k == "exclusive" and fam == "anti-synergy" or k == "prerequisite":
            return k
    if fam == "separating" and its and all(lock_locked(catalog, x) for x in its):
        return "lock"
    return None


def trivial_rule(r, catalog, groups: Optional[Dict[str, FrozenSet[str]]] = None) -> Optional[str]:
    """A mined pass rule (tri.passes.PassRule). groups: the known groups, for
    the group baselines; group literals are read as the item literals
    they are equivalent to."""
    item = getattr(r, "item", None)
    conds = list(getattr(r, "conds", []) or [])
    if groups and any(c.kind == "group" for c in conds):
        why = cull_reason(r, catalog, groups)
        if why:
            return why
        conds = expand(conds, item, catalog, groups)
    for c in conds:
        if c.kind == "stat" and c.op == "<" and r.effect == "never" and len(conds) == 1:
            req = _req(catalog, item, c.key)           # "s < k ⇒ ¬x" is "x needs s ≥ k" with k = req
            if req is not None and abs(float(c.value) - req) <= GRID:
                return "k = req"
        if c.kind == "item":
            k = related(catalog, item, c.key)
            if k == "exclusive" and r.effect == "never" and c.op == "has":
                return k
            if k == "prerequisite" and c.op == "has" and r.effect != "never":
                return k
            if k == "prerequisite" and c.op == "not" and r.effect == "never":
                return k
        if c.kind == "context" and str(c.key).split(":", 1)[0].lower() in LOCK_FIELDS and r.effect == "never" \
                and lock_locked(catalog, item):
            return "lock"
    return None


# ---------------------------------------------------------------- groups
MAX_EXACT = 16          # exact room (largest compatible subset) up to this many members
MAX_EXPAND = 4          # a saturated literal is read as at most this many item literals


def _need(catalog, x: str) -> Set[str]:
    """x and every item x needs for certain (all-of prerequisites, transitively)."""
    out, todo = set(), [x]
    while todo:
        y = todo.pop()
        if y in out:
            continue
        out.add(y)
        it = catalog.items.get(y)
        if it is not None:
            todo += list(it.prerequisites or [])
    return out


def _excl(catalog, a: str, b: str) -> bool:
    x, y = catalog.items.get(a), catalog.items.get(b)
    return x is not None and y is not None and (b in (x.exclusive_with or []) or a in (y.exclusive_with or []))


def room(members: Sequence[str], catalog) -> int:
    """The most of members one build can hold (no two exclusive). Exact for
    small groups, else the member count (an upper bound)."""
    ms = list(members)
    if len(ms) > MAX_EXACT:
        return len(ms)
    bad = {(a, b) for a, b in combinations(ms, 2) if _excl(catalog, a, b)}
    if not bad:
        return len(ms)
    for k in range(len(ms), 0, -1):
        for sub in combinations(ms, k):
            if not any((a, b) in bad or (b, a) in bad for a, b in combinations(sub, 2)):
                return k
    return 0


def _state(conds, item: str, catalog, groups) -> Tuple[Set[str], Set[str], bool]:
    """(forced present, forced absent, dead) from a rule's conditions, read to a
    fixed point: item literals, then group literals."""
    pres = {c.key for c in conds if c.kind == "item" and c.op == "has"}
    absn = {c.key for c in conds if c.kind == "item" and c.op == "not"}
    for _ in range(4):
        before = (len(pres), len(absn))
        pres = set().union(*[_need(catalog, y) for y in pres]) if pres else set()
        absn |= {z for y in pres for z in (catalog.items[y].exclusive_with or []) if y in catalog.items}
        if pres & absn:
            return pres, absn, True
        for c in conds:
            if c.kind != "group":
                continue
            others = [y for y in groups.get(c.key, ()) if y != item]
            free = [y for y in others if y not in absn]
            have = [y for y in others if y in pres]
            n = float(c.value or 0)
            if c.op == ">=":
                if n <= len(have):
                    continue                               # vacuous here
                if n > room(free, catalog):
                    return pres, absn, True
                if n >= 1 and free:                        # what every possible member needs
                    pres |= set.intersection(*[_need(catalog, y) for y in free])
                if n == len(free) and room(free, catalog) == len(free):
                    pres |= set(free)
            else:
                if n <= len(have):
                    return pres, absn, True
                if n <= 1:
                    absn |= set(others)
        if (len(pres), len(absn)) == before:
            break
    return pres, absn, bool(pres & absn)


def cull_reason(r, catalog, groups) -> Optional[str]:
    """Why r may be culled without changing any answer (or None)."""
    if getattr(r, "source", "") == "authored":
        return None
    conds = list(getattr(r, "conds", []) or [])
    if not any(c.kind == "group" for c in conds):
        return None
    for c in conds:
        if c.kind == "group" and c.op == "<" and float(c.value or 0) <= 0:
            return f"dead: |{c.key}| < {int(c.value or 0)}"
    pres, absn, dead = _state(conds, r.item, catalog, groups)
    if dead:
        g = next((c for c in conds if c.kind == "group"), None)
        return f"dead: {g.key} cannot meet {g.op} {int(g.value)}" if g is not None else "dead"
    if r.effect == "take" and r.item in pres:
        return f"implied: {r.item} is needed by the group's picks"
    if r.effect == "never" and (r.item in absn or any(_excl(catalog, p, r.item) for p in pres)):
        return f"implied: {r.item} is excluded by the group's picks"
    return None


def expand(conds, item: str, catalog, groups) -> list:
    """Group literals as the item literals they are equivalent to, where they are
    (saturated, |G| < 1, |D(k)| ≥ 1); others stay."""
    from tri.rules.defs import Condition
    out = []
    for c in conds:
        if c.kind != "group":
            out.append(c); continue
        kind, key = c.key.split("(", 1)[0], c.key.split("(", 1)[1][:-1]
        others = [y for y in sorted(groups.get(c.key, ())) if y != item]
        n = float(c.value or 0)
        if c.op == ">=" and kind == "D" and n == 1 and key != item and key in others:
            out.append(Condition("item", key, "has"))
        elif c.op == ">=" and n == len(others) and len(others) <= MAX_EXPAND and room(others, catalog) == len(others):
            out += [Condition("item", y, "has") for y in others]
        elif c.op == "<" and n == 1 and len(others) <= MAX_EXPAND:
            out += [Condition("item", y, "not") for y in others]
        else:
            out.append(c)
    return out


def cull(rules, catalog, groups) -> List[Tuple[str, str]]:
    """Mark the active rules cull_reason finds (status "trivial", the reason in
    note). Returns [(rule key, reason)]."""
    out = []
    if not groups:
        return out
    for r in rules:
        if getattr(r, "status", "active") != "active":
            continue
        why = cull_reason(r, catalog, groups)
        if why:
            r.status = "trivial"
            r.note = (r.note + "; " if getattr(r, "note", "") else "") + why
            out.append((r.key, why))
    return out


# ---------------------------------------------------------------- pruning
MAX_POINTS = 337        # the most stat points a build has
SHRINE = 25             # what a shrine leaves of each requirement: req_s − 25


def demand(catalog, item: str, memo: Optional[dict] = None) -> Dict[str, float]:
    """D_s: the least each stat must reach for item to be taken — the
    item's all-of requirements and single-stat unit clauses, over item and its
    all-of prerequisites (transitively). A lower bound."""
    if memo is not None and item in memo:
        return memo[item]
    d: Dict[str, float] = {}
    for y in _need(catalog, item):
        it = catalog.items.get(y)
        if it is None:
            continue
        reqs = list((it.requires or {}).items())
        for cl in it.clauses or []:
            if len(cl) == 1 and len(cl[0].get("stats", ())) == 1 and cl[0]["stats"][0] != "Total":
                reqs.append((cl[0]["stats"][0], cl[0].get("value", 0)))
        for s, v in reqs:
            d[s] = max(d.get(s, 0.0), float(v or 0))
    if memo is not None:
        memo[item] = d
    return d


def shrined(catalog, item: str, outside: Sequence[str] = (), memo: Optional[dict] = None) -> float:
    """sh(x, S) = Σ_{s ∉ S} max(0, D_s − 25): x's shrined requirement
    outside the stats S of a points clause."""
    S = set(outside)
    return sum(max(0.0, v - SHRINE) for s, v in demand(catalog, item, memo).items() if s not in S)


def _members(c) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """(how, member stats) of a points literal, or None."""
    if c.kind == "stat":
        return "stat", (c.key,)
    if c.kind == "agg":
        from tri.rules.passes import AGGREGATES
        for name, how, mem, _ in AGGREGATES:
            if name == c.key and mem is not None and how in ("sum", "max"):
                return how, tuple(mem)
    return None


def point_literal(c, catalog, item: str, memo: Optional[dict] = None) -> Optional[str]:
    """"true" if the points literal holds in every build that can take
    item, "false" if in none, else None.

      top(x, S) = 337 − sh(x, S)          the most S can hold beside x
      v_S ≥ n:  n > top  false,   n ≤ Σ_{s∈S} D_s (sum) / max_{s∈S} D_s  true
      v_S < n:  n > top  true,    n ≤ that floor                        false"""
    m = _members(c)
    if m is None or c.op not in (">=", "<") or c.value is None:
        return None
    how, S = m
    D = demand(catalog, item, memo)
    top = MAX_POINTS - shrined(catalog, item, S, memo)
    vals = [D.get(s, 0.0) for s in S]
    floor = sum(vals) if how == "sum" else max(vals)
    n = float(c.value)
    if n > top:
        return "false" if c.op == ">=" else "true"
    if n <= floor:
        return "true" if c.op == ">=" else "false"
    return None


def _rank(r) -> tuple:
    return (getattr(r, "source", "") == "authored", r.pass_ == "foundational",
            getattr(r, "support", 0) or 0, getattr(r, "bound", 0) or 0)


def prune(rules, catalog) -> Tuple[list, List[Tuple[str, str]]]:
    """(kept rules, [(key, reason)]) — rules that say nothing are deleted:

      1. points literals: a false one deletes the rule; a true one is
         dropped (a rule left with none reads ⊤ ⇒ x);
      2. ⊤ ⇒ x present: every other take rule of x goes (never rules stay);
      3. equal keys collapse to one; ⊤ ⇒ x [F] absorbs ⊤ ⇒ x [C];
      4. a rule is deleted when a rule of the same item, effect and pass
         with weaker conditions exists (it fires wherever this one does).
    Authored rules are kept as written. Rules marked trivial by a group
    baseline (cull) are deleted."""
    from tri.rules.passes import TAKE, _implied
    gone: List[Tuple[str, str]] = []
    out, memo = [], {}
    for r in rules:
        if getattr(r, "source", "") == "authored":
            out.append(r); continue
        if getattr(r, "status", "active") == "trivial":
            gone.append((r.key, "trivial")); continue
        keep, dead = [], None
        for c in r.conds:
            v = point_literal(c, catalog, r.item, memo) if c.kind in ("stat", "agg") else None
            if v == "false":
                dead = c; break
            if v != "true":
                keep.append(c)
        if dead is not None:
            gone.append((r.key, f"never holds: {dead.text()}")); continue
        if len(keep) < len(r.conds):
            dropped = [c.text() for c in r.conds if c not in keep]
            r.note = (r.note + "; " if getattr(r, "note", "") else "") + "always holds: " + ", ".join(dropped)
            r.conds = keep
        out.append(r)
    active = [r for r in out if getattr(r, "status", "active") == "active"]
    top = {}
    for r in active:
        if r.effect == TAKE and not r.conds and (r.item not in top or _rank(r) > _rank(top[r.item])):
            top[r.item] = r
    best: Dict[str, object] = {}
    for r in out:
        if getattr(r, "status", "active") != "active" or getattr(r, "source", "") == "authored":
            continue
        if r.key not in best or _rank(r) > _rank(best[r.key]):
            best[r.key] = r
    stage = []
    for r in out:
        if getattr(r, "source", "") != "authored" and getattr(r, "status", "active") == "active":
            t = top.get(r.item)
            if r.effect == TAKE and t is not None and r is not t and (r.conds or getattr(t, "source", "") == "authored"
                                                                     or t.pass_ == "foundational"):
                gone.append((r.key, "⊤ ⇒ " + r.item)); continue
            if best.get(r.key) is not r:
                gone.append((r.key, "duplicate")); continue
        stage.append(r)
    live: Dict[tuple, list] = {}
    for r in stage:
        if getattr(r, "status", "active") == "active":
            live.setdefault((r.item, r.effect, r.pass_), []).append(r)
    kept = []
    for r in stage:
        if getattr(r, "source", "") != "authored" and getattr(r, "status", "active") == "active":
            g = next((q for q in live[(r.item, r.effect, r.pass_)] if q is not r and _implied(q.conds, r.conds)
                      and not _implied(r.conds, q.conds)), None)
            if g is not None:
                gone.append((r.key, "subsumed by " + (g.when() if g.conds else "⊤"))); continue
        kept.append(r)
    return kept, gone
