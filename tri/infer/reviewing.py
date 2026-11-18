"""PassEngine methods: review of full builds and assembling alternatives."""
from __future__ import annotations

from typing import List, Optional, Sequence, Set

from tri.core.types import Build
from tri.infer.defs import ARCHETYPE, Alternative, _Sit
from tri.rules.passes import CENTRAL, EXCLUSION, FOUNDATIONAL, LUXURY


class _Reviewing:
    """PassEngine methods: review of full builds and assembling alternatives."""

    # ------------------------------------------------------------ review
    def review(self, b: Build, max_alternatives: int = 3, region: Optional[int] = None,
               locked: Sequence[str] = (), force: Optional[dict] = None,
               keep_size: bool = False) -> List[Alternative]:
        """Smallest edits of a full build, one Alternative per plausible archetype
        (force: just the forced one). keep_size: keep the input's counts instead
        of filling its capacity (InferConfig.fill_capacity)."""
        return [self._review_scored(b, s, set(locked), keep_size)
                for s in self.situations(b, max_alternatives, region, force=force)]

    def _offers(self, s: _Sit, sel: Set[str], fixed: Set[str]) -> List[dict]:
        """The other side of every sidegrade the answer is committed to: picks
        to remove (not those required or pinned) and the feasible picks to add."""
        out = []
        for g in self._sides(sel, s):
            side = g.side(sel)
            if side == 0:
                continue
            have, other = (g.a, g.b) if side > 0 else (g.b, g.a)
            rem = [i for i in have if i in sel and i not in fixed]
            trial = set(sel) - set(rem)
            add = []
            for it in sorted(other, key=lambda i: -float(s.q[self.c.index[i]]) if i in self.c.index else 0):
                if it in trial or it not in self.c.index:
                    continue
                if self._feasible(it, trial, s):
                    trial.add(it); add.append(it)
            if g.leaning(trial) * side >= 0:            # the swap must actually change sides
                continue
            if not add:
                continue
            sh_have, sh_other = (g.share_a, g.share_b) if side > 0 else (g.share_b, g.share_a)
            out.append(dict(region=s.mode, name=s.name, remove=rem, add=add,
                            share_from=sh_have, share_to=sh_other,
                            why=f"{sh_other:.0%} of builds of this archetype take {', '.join(other)} "
                                f"instead of {', '.join(have)} ({sh_have:.0%})"))
        return out

    def _violations(self, s: _Sit, sel: Set[str], blocked: Set[str], pinned: Set[str]) -> List[dict]:
        """Rules the answer does not satisfy. Empty unless slots, exclusivity,
        prerequisites, pinned picks or binding reviews make them unsatisfiable."""
        out = []
        req = self._required(s, set(sel), set(blocked))
        for it, (ps, src, why, rid) in req.items():
            if it not in sel:
                v = dict(item=it, kind="missing", pass_=ps, rule=why)
                t = self.c.index.get(it)
                if t is not None and not s.ctx.avail[t]:
                    v["note"] = "requirements not met at the pre-shrine or final stats"
                out.append(v)
        # rules that apply to the answer only through a kept pick tri rates as
        # off-archetype (below anchor_min_worth, so it did not pull them in): the
        # answer breaks them, so say so rather than stay silent
        seen = {v["item"] for v in out}
        s.ctx.items = set(sel) - set(blocked)
        rev_block = set(self._hard(s, "remove"))
        for r in self._firing(s):
            if r.item in sel or r.item in seen:
                continue
            if r.item in blocked and (r.item in rev_block or                 # a never rule that held
                                      self._forbidden(r.item, set(sel) | {r.item}, s) is not None):
                continue                               # on the build but not on the answer is reported
            seen.add(r.item)
            out.append(dict(item=r.item, kind="unanchored", pass_=r.pass_, rule=r.text(self.rb.region_names),
                            note="applies through a pick rated off-archetype: add it, or drop that pick"))
        # a take rule and a never rule that both apply: the never rule won; say so
        for it, (ps, src, why, rid) in self._required(s, set(sel), set()).items():
            if it in sel or it in req or src != "rules":
                continue
            r = self._forbidden(it, set(sel) | {it}, s)
            if r is not None:
                out.append(dict(item=it, kind="conflict", pass_=ps, rule=why, against=r.text()))
        ck = self._cat_kept(s)
        for it in sel:
            r = self._forbidden(it, sel, s) if it not in ck else None
            if r is not None:
                out.append(dict(item=it, kind="forbidden", pass_=EXCLUSION, rule=r.text(),
                                pinned=it in pinned))
        if self.cap_excess(sel) > 0:                   # only if every talent is pinned or required
            out.append(dict(item="talent cap", kind="talent_cap", pass_=FOUNDATIONAL, rule=self._cap_note(sel),
                            talents=len(self._talents & set(sel)), cap=self.talent_cap(sel)))
        if not self.fits(sel, s):                      # only if every mantra is pinned or required
            out.append(dict(item="mantra slots", kind="mantra_slots", pass_=FOUNDATIONAL, rule=self._slot_note(sel, s)))
        return out
    # ------------------------------------------------------------ helpers

    def _pass_here(self, it: str, s: _Sit, sel: Set[str]) -> str:
        if it in self.manual:
            return self.manual[it]
        s.ctx.items = set(sel)
        for r in self._firing(s):
            if r.item == it:
                if r.pass_ == CENTRAL and it not in self._central({it}, s):
                    return ARCHETYPE                   # mined: necessary here, not confirmed central
                return r.pass_
        if it in self._central({it}, s):
            return CENTRAL
        return LUXURY

    @staticmethod
    def _net(changes, start, final, catalog=None) -> list:
        """The change list reduced to the final state: an item added and later
        dropped (or dropped and later taken back) is no change, so the answer
        never both adds and removes one item. The last entry of a kept change
        wins (its reason is the one that held)."""
        last = {}
        for k, c in enumerate(changes):
            last[c.item] = k
        out = []
        for k, c in enumerate(changes):
            if last[c.item] != k:
                continue
            was, now = c.item in start, c.item in final
            if (c.action == "add" and now and not was) or (c.action == "remove" and was and not now):
                out.append(c)
        adds = [c for c in out if c.action == "add"]          # a take comes after the takes it needs
        if len(adds) > 1 and catalog is not None:
            by = {c.item: c for c in adds}
            it = iter(by[i] for i in catalog.prereq_order([c.item for c in adds]))
            out = [next(it) if c.action == "add" else c for c in out]
        return out

    def _alt(self, b, s: _Sit, sel, changes, iffy, passes, cap) -> Alternative:
        changes[:] = self._net(changes, set(b.items()), set(sel), self.c)
        cen = self._central(set(sel) | set(b.items()), s) | {i for i, c in self.manual.items() if c == CENTRAL}
        for c in changes:
            if c.pass_ == CENTRAL and c.item not in cen:
                c.pass_ = ARCHETYPE                    # a mined take rule: necessary here, not playstyle-defining
        passes = {i: (ARCHETYPE if p == CENTRAL and i not in cen else p) for i, p in passes.items()}
        ind = {k: dict(v) for k, v in (getattr(s, "indicative", None) or {}).items() if k in set(sel)}
        order = {i: k for k, i in enumerate(b.items())}
        items = self.c.prereq_order(sorted(sel, key=lambda i: (order.get(i, 10 ** 6), i)))
        tal = [i for i in items if self._kind(i) == "t"]
        man = [i for i in items if self._kind(i) == "m"]
        gain = sum(c.marginal for c in changes if c.action == "add") - \
            sum(c.marginal for c in changes if c.action == "remove")
        w = {i: float(s.q[self.c.index[i]]) for i in set(items) | {c.item for c in changes}}
        return Alternative(tal, man, s.plausibility, gain, changes, iffy, s.mode, s.name, passes, w,
                           {k: v for k, v in s.memory.items() if k in w}, tuple(cap), [float(x) for x in s.z],
                           unrated=[i for i in items if self._cold(i, s)], indicative=ind)
