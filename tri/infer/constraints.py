"""PassEngine methods: constraints: caps, slots, feasibility, rules, reviews and sides."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from tri.core.types import Kind, Reason
from tri.infer.defs import _Sit
from tri.rules.passes import CENTRAL, FOUNDATIONAL, LUXURY, PassRule


class _Constraints:
    """PassEngine methods (PassEngine methods: constraints: caps, slots, feasibility, rules, reviews and sides.)."""

    def _kind(self, item: str) -> str:
        return "t" if self.c.items[item].kind == Kind.TALENT else "m"

    def _cold(self, item: str, s: _Sit) -> bool:
        """An unrated item without local evidence: worth says nothing about it."""
        return item in self.unrated_items and float(s.support[self.c.index[item]]) < self.cfg.cold_support

    def _lux(self, item: str) -> bool:
        return self.manual.get(item) == LUXURY

    def tier(self, item: str) -> str:
        """foundational | central | luxury: the admin category, else the selection
        rules (an item with an active F / C take rule anywhere), else luxury."""
        m = self.manual.get(item)
        if m:
            return m
        tr = self.__dict__.get("_rule_tier")
        if tr is None:
            from tri.rules.passes import CENTRAL as C_, FOUNDATIONAL as F_, TAKE
            tr = {}
            for r in self.rb.rules:
                if r.status == "active" and r.effect == TAKE and r.pass_ in (F_, C_):
                    if tr.get(r.item) != F_:
                        tr[r.item] = r.pass_
            self._rule_tier = tr
        return tr.get(item, LUXURY)

    def _worth_ok(self, item: str, fallback: bool = False, s: Optional[_Sit] = None) -> bool:
        """May worth take item? v3.0: only luxury talents are ranked by worth; a
        foundational / central talent is decided by the selection rules (mantras
        keep worth). v3.0.2 (InferConfig.tier_context): the tier is read on the
        build, so a talent whose F / C rules do not fire here is luxury here and
        worth may take it. Two further exceptions: (i) with s, a talent with a take rule whose
        context literals all hold here and that only lacks co-picks (anchors): worth
        may start such a cluster, since rules anchored on each other never fire on
        a build that has none of them; (ii) fallback, the last resort when no
        luxury talent fits and the talent cap is unfilled (InferConfig.tier_fallback)."""
        if not getattr(self.cfg, "rule_tiers", False) or item not in self._talents:
            return True
        if self.tier(item) == LUXURY:
            return True
        if s is not None and getattr(self.cfg, "tier_context", True) and not self._fires_here(item, s):
            return True                                  # v3.0.2: no rule takes it here, so it is luxury here
        if s is not None and self._anchor_only(item, s):
            return True
        return fallback and getattr(self.cfg, "tier_fallback", False)

    def tier_here(self, item: str, s: _Sit) -> str:
        """The tier of item on this build (v3.0.2, math §6.12): its category, else
        the pass of a take rule that fires here, else luxury. A talent that is
        foundational or central elsewhere but has no rule firing here is luxury
        here: worth ranks it like any other luxury talent."""
        m = self.manual.get(item)
        if m:
            return m
        if self.tier(item) == LUXURY or not self._fires_here(item, s):
            return LUXURY
        return self.tier(item)

    def _fires_here(self, item: str, s: _Sit) -> bool:
        """A foundational / central take rule for item fires on this build."""
        memo = getattr(s, "_fires", None)
        if memo is None:
            memo = {}
            for r in self._firing(s):
                if r.pass_ in (FOUNDATIONAL, CENTRAL):
                    memo[r.item] = True
            try:
                s._fires = memo
            except AttributeError:
                pass
        return item in memo

    def _anchor_only(self, item: str, s: _Sit) -> bool:
        from tri.rules.passes import TAKE, holds
        for r in self.rb.for_item(item):
            if r.status == "active" and r.effect == TAKE and any(c.kind == "item" for c in r.conds) and \
                    all(holds(c, s.ctx, item) for c in r.conds if c.kind != "item"):
                return True
        return False

    def _central(self, sel, s: _Sit) -> Set[str]:
        """Confirmed central picks of sel (v2.18): talents that define the build's
        playstyle. Only an admin "central" category, a reviewer's indicator / core
        note or a promoted reviewer-consensus tell of the build's family (v2.19) confirms one; a mined take rule does not (it reports its picks under
        the "archetype" pass). Central picks are respected: no removal for space,
        luxury, worth or the cap while anything else can go, and none unless a
        clash (a never rule naming a co-pick, mutual exclusion, a binding review,
        Vow of Iron) forces it."""
        out = {i for i in sel if self.manual.get(i) == CENTRAL and self._kind(i) == "t"}
        out |= {i for i, v in (getattr(s, "indicative", None) or {}).items()
                if i in sel and self.manual.get(i) != LUXURY
                and (v.get("kind") in ("indicator", "core")
                     or (v.get("kind") == "consensus" and v.get("weight", 0) >= self.cfg.consensus_central))}
        return out

    def _firing(self, s: _Sit, pass_: Optional[str] = None) -> List[PassRule]:
        """Take rules that fire for s, across the chasm (math §5.4, v2.19): a mined
        central rule whose covered builds lie in other families (its home) does not
        fire on a build located in a family outside its home. Foundational rules
        (spread over the corpus by construction) and category rules are unchanged."""
        rs = self.rb.firing(s.ctx, pass_)
        home = getattr(self, "rule_home", None)
        if not home or s.family is None:
            return rs
        return [r for r in rs if r.pass_ != CENTRAL or r.key not in home or s.family in home[r.key]]

    @staticmethod
    def _clash_rule(r: PassRule) -> bool:
        """A never rule that names a co-pick (x with y): a clash, which may remove a central pick."""
        return any(c.kind == "item" for c in r.conds)

    def _cat_kept(self, s: _Sit) -> Set[str]:
        """Items a category requires here (foundational; central when available)."""
        out = set()
        for it, c in self.manual.items():
            t = self.c.index.get(it)
            if t is not None and (c == FOUNDATIONAL or (c == CENTRAL and s.ctx.avail[t])):
                out.add(it)
        return out

    def _sides(self, sel, s: _Sit) -> list:
        """Sidegrades of the archetype sel belongs to (it holds their shared core)."""
        if not self.sidegrades:
            return []
        av = lambda i: i in self.c.index and bool(s.ctx.avail[self.c.index[i]])
        return [g for g in self.sidegrades if g.applies(sel, av)]

    def _side_block(self, item: str, sel, s: _Sit) -> bool:
        """Worth may not add item: it is on the side of a sidegrade sel holds fewer of,
        or a reviewer called it a sidegrade to a pick sel already holds."""
        if any(y in sel for y in self._review_sides(s).get(item, ())):
            return True
        for g in self._sides(sel, s):
            if item in g.a and g.leaning(sel) < 0:
                return True
            if item in g.b and g.leaning(sel) > 0:
                return True
        return False

    def _review_sides(self, s: _Sit) -> Dict[str, Set[str]]:
        """{x: {y, ...}} from binding "x is a sidegrade to y" claims, both ways
        (weak: under this archetype, through ledger reach; strong: anywhere)."""
        out: Dict[str, Set[str]] = {}
        for it, v in s.memory.items():
            if v[0] == "iffy" and v[4] == Reason.SIDEGRADE.value and len(v) > 7 and self._binds(v):
                for y in v[7]:
                    out.setdefault(it, set()).add(y)
                    out.setdefault(y, set()).add(it)
        return out

    def _note(self, rid) -> str:
        """The reviewer's first free-text note on review rid, as a short suffix."""
        r = self.ledger.by_id.get(rid) if (self.ledger is not None and rid) else None
        for n in (getattr(r, "notes", None) or []):
            if n.get("kind") == "comment" and n.get("text"):
                t = str(n["text"])
                return f'; note: "{t[:80] + ("..." if len(t) > 80 else "")}"'
        return ""

    def _depends_unmet(self, it: str, sel, s=None) -> Optional[List[str]]:
        """v3.6 (math §2.8e): the components a supported DEPENDS review rule says `it`
        does not work without (x NW > a, b) that sel lacks, or None. A worth pick
        is never taken without them; they are not pulled in for it (that is the
        dependant's job when they are worth it)."""
        b = getattr(s, "build", None)
        feats = getattr(s, "feats", None)
        for r in getattr(self, "review_rules", None) or ():
            if r.kind != "DEPENDS" or r.subject != it or not r.other:
                continue
            if b is not None and not r.applies(b, feats):
                continue
            miss = [y for y in r.other.split(" + ") if y not in sel]
            if miss:
                return miss
        return None

    def talent_cap(self, sel) -> int:
        """76 - 2m: m = selected mantras that are not Oath / Origin (faction) / Monster mantras."""
        return self._cap_base - self._cap_per * len(self._lowers & set(sel))

    def cap_excess(self, sel) -> int:
        """Talents above the cap (<= 0: the selection obeys it)."""
        sel = set(sel)
        return len(self._talents & sel) - self.talent_cap(sel)

    def _cap_note(self, sel) -> str:
        return f"over the talent cap ({len(self._talents & set(sel))} talents, cap {self.talent_cap(sel)} = " \
               f"{self._cap_base} - {self._cap_per} x {len(self._lowers & set(sel))} non-exempt mantras)"

    def fits(self, sel, s: _Sit) -> bool:
        """Every mantra of sel has a mantra slot of a type it fits (math §2.6)."""
        return self.c.mantras_fit(set(sel) | s.outside, s.oath)

    def slot_free(self, sel, s: _Sit) -> int:
        """Mantra slots of sel left empty."""
        return self.c.mantra_slot_usage(set(sel) | s.outside, s.oath)["free"]

    def _slot_needed(self, it: str, sel, s: _Sit) -> bool:
        """it grants mantra slots that sel's mantras need."""
        return bool(self.c.slot_grants(it)) and it in sel and not self.fits(set(sel) - {it}, s)

    def _feasible(self, item: str, sel: Set[str], s: _Sit, rules: bool = True, cap: bool = True) -> bool:
        """Can item join sel? rules=False skips the never rules (a binding review outranks them);
        cap=False skips the talent cap and the mantra slots (the caller frees room first)."""
        t = self.c.index.get(item)
        if t is None or item in sel or not s.ctx.avail[t]:
            return False
        it = self.c.items[item]
        if any(x in sel for x in it.exclusive_with):
            return False
        if any(item in self.c.items[x].exclusive_with for x in sel):
            return False
        if it.prerequisites and not all(p in sel for p in it.prerequisites):
            return False
        if it.prerequisites_any and not all(any(p in sel for p in g) for g in it.prerequisites_any):
            return False
        if cap and (item in self._talents or item in self._lowers) and self.cap_excess(set(sel) | {item}) > 0:
            return False
        if cap and it.kind == Kind.MANTRA and not self.fits(set(sel) | {item}, s):
            return False
        if not rules:
            return True
        if self._forbidden(item, sel, s) is not None:
            return False
        after = set(sel) | {item}                     # would it trigger an exclusion of a current pick?
        for r in self._never_by_anchor.get(item, ()):
            if r.item in sel and self._holds(r, after, s):
                return False
        return True

    def _room_for(self, items, sel: Set[str], s: _Sit, cap: Tuple[int, int], rejected: Set[str]) -> bool:
        """Can every item join sel without displacing a pick: available, not
        rejected, prerequisites in place (in any order), within the build's
        talent / mantra counts, the talent cap and the mantra slots?"""
        items = [y for y in items if y not in sel]
        if not items or any(y in rejected or y not in self.c.index for y in items):
            return bool(not items)
        after = set(sel) | set(items)
        for k, n in ((0, "t"), (1, "m")):
            if sum(self._kind(i) == n for i in after) > cap[k]:
                return False
        if self.cap_excess(after) > 0 or not self.fits(after, s):
            return False
        return all(self._feasible(y, after - {y}, s, cap=False) for y in items)

    def _holds(self, r: PassRule, items: Set[str], s: _Sit) -> bool:
        keep = s.ctx.items
        s.ctx.items = items
        try:
            return r.applies(s.ctx)
        finally:
            s.ctx.items = keep

    def _forbidden(self, item: str, sel: Set[str], s: _Sit) -> Optional[PassRule]:
        """The never rule that forbids item next to sel, if any."""
        t = self.c.index.get(item)
        if t is None or not s.ctx.avail[t]:
            return None
        if item in (s.weapon or {}).get("guard", ()):
            return None                                    # v3.4: weapon rules hold over mined never rules
        for r in self._never.get(item, ()):
            if self._holds(r, set(sel) - {item}, s):
                return r
        return None

    @staticmethod
    def _never_reason(r: PassRule) -> Tuple[str, List[str]]:
        rel = [c.key for c in r.conds if c.kind == "item" and c.op == "has"]
        if rel:
            return Reason.WORKS_AGAINST.value, rel
        if any(c.kind == "context" for c in r.conds):
            return Reason.SCENARIO_MISMATCH.value, []
        return Reason.NO_SYNERGY.value, []

    def _rule_block(self, item: str, sel: Set[str], s: _Sit):
        """(reason, related, why) if a review rule rejects item next to sel, else None."""
        if item in (s.weapon or {}).get("guard", ()):
            return None                                    # v3.4: weapon rules hold over review rules
        for r in self.review_rules:
            if r.subject != item or not r.applies(s.build, s.feats):
                continue
            if r.kind == "BROKEN":
                return Reason.BROKEN.value, [], r.describe()
            if r.kind == "NO_FIT":
                return Reason.NO_SYNERGY.value, [], r.describe()
            if r.kind == "OFF_ARCHETYPE":                   # strong SM: not on this archetype
                return Reason.SCENARIO_MISMATCH.value, [], r.describe()
            if r.kind == "REDUNDANT" and r.other in sel:
                return Reason.REDUNDANT_WITH.value, [r.other], r.describe()
            if r.kind == "CONFLICT" and r.other in sel:
                return Reason.WORKS_AGAINST.value, [r.other], r.describe()
            if r.kind == "PREFER" and r.other in sel:
                return Reason.LESS_NECESSARY_SPACE.value, [], r.describe()
        return None

    def _binds(self, v: tuple) -> bool:
        if self.ledger is None:
            return v[1] >= self.cfg.memory_hard
        return self.ledger.binding(v, self.cfg.memory_hard)

    @staticmethod
    def _memory_why(verb: str, v: tuple) -> str:
        """"added by review X (similarity s)", or for a pooled family claim (v3.3)
        "added by the reviews of n builds of this family (newest X)"."""
        if len(v) > 6 and v[6] == "family":
            n = v[8] if len(v) > 8 else "several"
            return f"{verb} by the reviews of {n} builds of this family (newest {v[2]})"
        return f"{verb} by review {v[2]} (similarity {v[3]:.2f})"

    @staticmethod
    def _general(v: tuple) -> bool:
        """A generally-good claim from a build of another archetype."""
        return len(v) > 6 and v[6] == "general"

    @staticmethod
    def _add_reason(ps: str, src: str) -> Optional[str]:
        if src == "general":
            return Reason.GENERALLY_GOOD.value
        return Reason.NECESSARY_FOR_ARCHETYPE.value if ps != LUXURY else None

    def _hard(self, s: _Sit, label: str) -> Dict[str, tuple]:
        return {i: v for i, v in s.memory.items() if v[0] == label and self._binds(v)}
