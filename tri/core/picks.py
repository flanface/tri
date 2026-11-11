"""Catalog methods: pick relations: exclusions, prerequisites and their order."""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple


class _Picks:
    """Catalog methods (Catalog methods: pick relations: exclusions, prerequisites and their order.)."""

    # ---- mutual exclusion and prerequisites ------------------------------
    def excludes(self, a: str, b: str) -> bool:
        """a and b are mutually exclusive (listed on either side in the game data)."""
        ia, ib = self.items.get(a), self.items.get(b)
        return a != b and ia is not None and ib is not None and (b in ia.exclusive_with or a in ib.exclusive_with)

    def exclusive_in(self, item: str, items) -> List[str]:
        """Items of `items` that cannot be taken together with `item`, sorted."""
        return sorted(x for x in set(items) if self.excludes(item, x))

    def prereqs_met(self, item: str, items) -> bool:
        """Every prerequisite of item (and one of each prerequisites_any group) is in items."""
        it = self.items.get(item)
        if it is None:
            return True
        S = set(items)
        return all(p in S for p in it.prerequisites) and all(any(p in S for p in g) for g in it.prerequisites_any)

    def prereq_closure(self, item: str) -> List[str]:
        """Every prerequisite of item, transitively (both kinds), sorted."""
        out: Set[str] = set()
        todo = [item]
        while todo:
            it = self.items.get(todo.pop())
            if it is None:
                continue
            for p in list(it.prerequisites) + [y for g in it.prerequisites_any for y in g]:
                if p not in out and p != item:
                    out.add(p); todo.append(p)
        return sorted(out)

    def prereq_order(self, names: Sequence[str], canon=None) -> List[str]:
        """names reordered so that every prerequisite in the list comes before
        the items needing it (stable otherwise). Planners and the dwt.sh API
        add picks in order and reject one whose prerequisite is not yet there.
        canon: name -> catalog id (for planner spellings)."""
        canon = canon or (lambda x: x)
        ids = [canon(n) for n in names]
        pos: Dict[str, List[int]] = {}
        for k, i in enumerate(ids):
            pos.setdefault(i, []).append(k)
        out: List[int] = []
        state: Dict[int, int] = {}

        def visit(k: int) -> None:
            if state.get(k):
                return
            state[k] = 1                                   # in progress (cycles are left in place)
            it = self.items.get(ids[k])
            if it is not None:
                for p in list(it.prerequisites) + [x for g in it.prerequisites_any for x in g]:
                    for j in pos.get(p, ()):
                        if not state.get(j):
                            visit(j)
            state[k] = 2
            out.append(k)

        for k in range(len(ids)):
            visit(k)
        return [names[k] for k in out]

    def dependents(self, removed, items) -> List[str]:
        """Items of `items` that lose a prerequisite once `removed` are gone,
        transitively (z requires y requires x: removing x removes y and z).
        Only items whose prerequisites hold in `items` count. Sorted."""
        S, gone = set(items), set(removed)
        out: Set[str] = set()
        while True:
            left = S - gone - out
            new = {z for z in left if self.prereqs_met(z, S) and not self.prereqs_met(z, left)}
            if not new:
                return sorted(out)
            out |= new

    def exclusion_removals(self, item: str, items) -> Tuple[List[str], List[str]]:
        """(blockers, dependents): what must leave `items` before `item` can be
        added. blockers are mutually exclusive with item; dependents need a
        blocker (directly or through another dependent)."""
        bl = self.exclusive_in(item, items)
        return bl, (self.dependents(bl, items) if bl else [])
