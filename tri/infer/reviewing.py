"""PassEngine methods: review of full builds and assembling alternatives."""
from __future__ import annotations

from collections import OrderedDict

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from tri.core.types import Build, Reason
from tri.infer.defs import ARCHETYPE, REVIEW, Alternative, Change, _Sit
from tri.rules.passes import CENTRAL, EXCLUSION, FOUNDATIONAL, LUXURY


class _Reviewing:
    """PassEngine methods (PassEngine methods: review of full builds and assembling alternatives.)."""

    # ------------------------------------------------------------ review
    def review(self, b: Build, max_alternatives: int = 3, region: Optional[int] = None,
               locked: Sequence[str] = (), force: Optional[dict] = None,
               keep_size: bool = False) -> List[Alternative]:
        """Smallest edits of a full build, one Alternative per plausible archetype
        (force: just the forced one). keep_size: keep the input's counts instead
        of filling its capacity (InferConfig.fill_capacity)."""
        return [self._review_one(b, s, set(locked), keep_size)
                for s in self.situations(b, max_alternatives, region, force=force)]

    def _review_one(self, b: Build, s: _Sit, locked: Set[str] = frozenset(), keep_size: bool = False) -> Alternative:
        import dataclasses
        cfg = dataclasses.replace(self.cfg, fill_capacity=False) if keep_size and self.cfg.fill_capacity else self.cfg
        S = [i for i in dict.fromkeys(b.items()) if i in self.c.index]
        cap = (sum(self._kind(i) == "t" for i in S), sum(self._kind(i) == "m" for i in S))
        q = lambda it: float(s.q[self.c.index[it]])
        # a generally-good claim from another archetype protects the talent from
        # displacement (it is required, below) but does not outrank exclusions
        keep_mem = {i: v for i, v in (self._hard(s, "keep") | self._hard(s, "add")).items() if not self._general(v)}
        keep_mem.update({i: ("keep", 1.0, None, 1.0, None, None) for i in locked if i in S})   # pinned by the user
        rem_mem = {i: v for i, v in self._hard(s, "remove").items() if i not in locked}
        left_out = set(self._hard(s, "absent"))           # v3.3: pooled "absent": no worth swap brings it in
        kept_by = set()
        if cfg.swap_guard:                                # v3.3: review evidence guards worth swaps
            left_out |= {i for i, v in s.memory.items() if v[0] in ("absent", "remove")}
            kept_by = {i for i, v in s.memory.items() if v[0] == "keep"}
        cat_keep = self._cat_kept(s)                   # categories outrank never rules
        cur: Set[str] = set(S)
        changes: List[Change] = []
        dropped: Set[str] = set()

        def drop(it, reason, related, source, why, pass_=LUXURY, rid=None):
            cur.discard(it); dropped.add(it)
            changes.append(Change(it, "remove", reason, list(related), q(it), source, rid, pass_, why))

        # 0. a Pathfinder build holds no Vow of Iron only pick (nor what requires one)
        for it, why in self._voi_drops(cur):
            drop(it, Reason.SCENARIO_MISMATCH.value, [], "rules", why, FOUNDATIONAL)
        for it, why in (s.weapon or {}).get("forbid", {}).items():   # v3.2 weapon rules hold over pins
            if it in cur:
                drop(it, Reason.SCENARIO_MISMATCH.value, [], "weapon", why, FOUNDATIONAL)
                keep_mem.pop(it, None)
        S = [i for i in S if i in cur]

        # 1. memory and review rules reject
        needs: Dict[tuple, list] = {}                  # NW groups: decided after the other drops
        for it in S:
            if it in rem_mem:
                v = rem_mem[it]
                lab, st, rid, sim = v[:4]
                reason, rel = (v[4], list(v[7])) if len(v) > 7 else (None, [])
                if reason is None:
                    rv = self.ledger.by_id[rid] if self.ledger is not None else None
                    for o in (rv.operations if rv is not None else []):
                        if o.item_id == it and o.reason is not None:
                            reason, rel = o.reason.value, list(o.related)
                if reason == Reason.NEEDS.value and rel:
                    needs.setdefault(tuple(sorted(rel)), []).append((it, rid, sim))
                    continue
                drop(it, reason or Reason.NO_SYNERGY.value, rel, "review",
                     self._memory_why("removed", v) + self._note(rid), REVIEW, rid)
        for it in list(cur):
            if it in keep_mem:
                continue
            blk = self._rule_block(it, cur - {it}, s)
            if blk:
                drop(it, blk[0], blk[1], "review_rule", blk[2])
        rejected: Set[str] = set(dropped)              # removed for cause: never re-added
        central = self._central(cur, s)                # respected unless a clash forces them out
        by_rule: Set[str] = set()                      # rejected only by a never rule on the context then
        for it in list(cur):                           # 1b. exclusions (never rules)
            if it in keep_mem or it in cat_keep or it not in cur:
                continue
            r = self._forbidden(it, cur, s)
            if r is not None and it in central and not self._clash_rule(r):
                continue                               # a context rule does not undo a central pick
            if r is not None:
                reason, rel = self._never_reason(r)
                drop(it, reason, rel, "rules", r.text(), EXCLUSION)
                rejected.add(it); by_rule.add(it)
        # 1c. NW "x does not work without a, b": add a, b when there is room
        # (equivalent to the removal), else remove x
        nw_kept: Set[str] = set()
        for rel, members in needs.items():
            xs = [m for m in members if m[0] in cur]
            if not xs:
                continue
            miss = [y for y in rel if y not in cur]
            if self._room_for(miss, cur, s, cap, rejected):
                for y in miss:
                    cur.add(y)
                    changes.append(Change(y, "add", Reason.SYNERGY_WITH.value, [x for x, _, _ in xs], q(y), "review",
                                          xs[0][1], REVIEW, f"{', '.join(x for x, _, _ in xs)} need {', '.join(rel)}: "
                                          f"room to add them (review {xs[0][1]})" + self._note(xs[0][1])))
                for x, rid, sim in xs:
                    nw_kept.add(x)
                    keep_mem[x] = ("keep", 1.0, rid, sim, None, None)
                    keep_mem.update({y: ("keep", 1.0, rid, sim, None, None) for y in miss})
            else:
                for x, rid, sim in xs:
                    drop(x, Reason.NEEDS.value, miss, "review", f"does not work without {', '.join(miss)}; no room "
                         f"to add them (review {rid}, similarity {sim:.2f})" + self._note(rid), REVIEW, rid)
                    rejected.add(x)
        # 2. required picks
        hard_block = rejected | (set(rem_mem) - nw_kept)
        req = self._required(s, cur, hard_block)
        guard = set((s.weapon or {}).get("guard", ()))     # v3.4: weapon-required picks and their prerequisites
        protected = set(keep_mem) | {i for i in cur if i in req} | (central & cur) | (guard & cur)
        free = {"t": cap[0] - sum(self._kind(i) == "t" for i in cur),
                "m": cap[1] - sum(self._kind(i) == "m" for i in cur)}

        def needed_by_others(it):
            rest = cur - {it}
            return any(it in self.c.items[x].prerequisites or any(it in g for g in self.c.items[x].prerequisites_any)
                       for x in rest) or self._slot_needed(it, cur, s)

        def anchors(it):
            """v3.2.1: it is the only pick a picked prerequisite unlocks (Unfazed
            for Grasp on Reality); a worth swap that takes it out strands the
            prerequisite, so the pair stays (math §2.8d)."""
            I = self.c.items.get(it)
            if I is None:
                return False
            pre = set(I.prerequisites) | {y for g in I.prerequisites_any for y in g}
            rest = cur - {it}
            return any(p in cur and not any(d in rest for d in self._dependants.get(p, ())) for p in pre)

        def displaceable(kind, worth: bool = False):
            """Picks that can make room, least worthy first; unrated picks last.
            worth=True: for a worth swap, which never takes out a luxury
            (category) or unrated pick."""
            c = [i for i in cur if self._kind(i) == kind and i not in protected and not needed_by_others(i)]
            if worth:
                c = [i for i in c if not self._lux(i) and not self._cold(i, s) and not anchors(i)]
            return sorted(c, key=lambda i: (self._cold(i, s), q(i)))

        def add_required(req) -> bool:
            moved = False
            for it, (ps, src, why, rid) in req.items():
                if it in cur:
                    protected.add(it); continue
                # v3.4 addition order: missing prerequisites come in first (Knight's Rally
                # before Turtle Shell), as in the completion (math §2.8d)
                chain = self._missing_chain(it, cur, s, hard_block - guard) if not self.c.prereqs_met(it, cur) else []
                if chain is None:
                    continue
                if chain:
                    # the chain and the pick go in together or not at all
                    snap = (set(cur), list(changes), set(dropped), dict(free), set(protected))
                    seq = [(y, (ps, src, f"prerequisite of {it}", rid)) for y in chain]
                    add_required(OrderedDict(seq))
                    if all(y in cur for y in chain):
                        add_required(OrderedDict([(it, (ps, src, why, rid))]))
                    if it not in cur:
                        cur.clear(); cur.update(snap[0]); changes[:] = snap[1]
                        dropped.clear(); dropped.update(snap[2]); free.update(snap[3])
                        protected.clear(); protected.update(snap[4])
                    else:
                        moved = True
                    continue
                k = self._kind(it)
                if not self._feasible(it, cur, s, rules=src not in ("review", "category"), cap=False):
                    if not exclusive_swap(it, ps, src):
                        continue
                if self._rule_block(it, cur, s):
                    continue
                if (it in self._lowers or k == "t") and free[k] > 0 and \
                        self.cap_excess(cur | {it}) > 0 and not cap_room(it):
                    continue                           # no talent can make room under the cap
                d = None
                if free[k] <= 0 or (k == "m" and not self.fits(cur | {it}, s)):
                    # make room; a mantra needs a slot of a type it fits
                    d = [v for v in displaceable(k) if k != "m" or self.fits((cur - {v}) | {it}, s)]
                    if not d and it in guard:
                        if self.cap_excess(cur | {it}) <= 0 and (k != "m" or self.fits(cur | {it}, s)):
                            free[k] += 1               # v3.4: a weapon-required pick grows a build held to its
                            d = None                   # size when the cap and slots allow (only game data blocks it)
                        else:                          # or displaces the least worthy pick that no rule, prerequisite
                            d = sorted((v for v in cur if self._kind(v) == k and v not in guard and v not in req
                                        and v not in central and not needed_by_others(v)
                                        and (k != "m" or self.fits((cur - {v}) | {it}, s))),
                                       key=lambda v: (self._cold(v, s), q(v)))   # or category holds (reviews and pins do not)
                            d = [v for v in d if v not in self.manual] or None
                            if d is None:
                                continue
                    elif not d:
                        continue
                if d:
                    v = d[0]
                    drop(v, Reason.LESS_NECESSARY_SPACE.value if q(v) >= cfg.no_fit_rate else Reason.NO_SYNERGY.value,
                         [], "rules" if src == "rules" else src, f"slot needed for {it} ({ps})", rid=rid)
                    free[k] += 1
                cur.add(it); free[k] -= 1; protected.add(it); moved = True
                rr, rl = (s.req_why[it] if src == "review" and it in s.req_why else (self._add_reason(CENTRAL, src), []))
                changes.append(Change(it, "add", rr, list(rl), q(it), src, rid, ps, why))
                trim_cap()
            return moved

        def exclusive_swap(it, ps, src) -> bool:
            """A required pick blocked only by mutually exclusive picks: remove the
            blockers and every pick that requires them, if none of them is
            protected, then re-check feasibility."""
            bl, dep = self.c.exclusion_removals(it, cur)
            out = bl + dep
            if not bl or any(x in protected for x in out):
                return False
            trial = cur - set(out)
            if not self._feasible(it, trial, s, rules=src not in ("review", "category"), cap=False):
                return False
            for x in bl:
                drop(x, Reason.WORKS_AGAINST.value, [it], "rules" if src == "rules" else src,
                     f"mutually exclusive with {it}, which is required ({ps})")
                free[self._kind(x)] += 1
            for x in dep:
                drop(x, Reason.LESS_NECESSARY_SPACE.value, [], "rules" if src == "rules" else src,
                     f"requires {', '.join(bl)}, removed for {it}")
                free[self._kind(x)] += 1
            return True

        def cap_room(it) -> bool:
            """Could the cap be met after adding it, by dropping displaceable picks?"""
            after = cur | {it}
            ex = self.cap_excess(after)
            # dropping a talent lowers the excess by 1, a non-exempt mantra by cap_per
            room_t = len(displaceable("t"))
            room_m = sum(1 for x in displaceable("m") if x in self._lowers)
            return room_t + self._cap_per * room_m >= ex

        def trim_cap() -> bool:
            """Strict talent cap: drop the lowest-worth displaceable talents (then
            non-exempt mantras, each of which raises the cap by cap_per)."""
            moved = False
            while self.cap_excess(cur) > 0:
                d = displaceable("t") or [x for x in displaceable("m") if x in self._lowers]
                if not d:                              # the cap is a hard clash: central picks last
                    d = sorted((x for x in cur & central if self._kind(x) == "t" and x not in keep_mem
                                and x not in req and not needed_by_others(x)), key=q)
                if not d:
                    break
                v = d[0]
                drop(v, Reason.LESS_NECESSARY_SPACE.value, [], "rules", self._cap_note(cur))
                if self._kind(v) == "m":
                    free["m"] += 1
                moved = True
            return moved

        def trim_slots() -> bool:
            """Strict mantra slots: drop the lowest-worth displaceable mantras
            until every mantra has a slot of a type it fits."""
            moved = False
            while not self.fits(cur, s):
                d = displaceable("m")
                if not d:
                    break
                v = next((x for x in d if self.fits(cur - {x}, s)), d[0])
                drop(v, Reason.LESS_NECESSARY_SPACE.value, [], "rules", self._slot_note(cur, s))
                moved = True
            return moved

        trim_cap()                                     # an input above the cap
        trim_slots()                                   # or above its mantra slots

        def capacity():
            """fill_capacity: the room is the build's capacity, not its input
            size: empty mantra slots, then talents up to 76 - 2m (math §2.7)."""
            if cfg.fill_capacity:
                free["m"] = max(0, self.slot_free(cur, s))
                free["t"] = max(0, self.talent_cap(cur) - len(self._talents & cur))
        capacity()
        add_required(req)
        # 2b. picks similar builds (almost) never take do not fit the archetype
        for it in sorted(cur, key=q):
            if q(it) >= cfg.no_fit_rate:
                break
            if it in protected or it not in S or needed_by_others(it) or self._lux(it) or self._cold(it, s):
                continue                               # (v3.0: an F / C talent no rule requires here may go)
            drop(it, Reason.NO_SYNERGY.value, [], "worth",
                 f"taken by only {q(it):.0%} of similar builds that can take it")
            free[self._kind(it)] += 1
        # 3. refill freed slots by worth

        def best_candidates(kind, cap=True, luxury_first=True, fallback=False):
            """Feasible additions by worth; luxury (category) items first when
            luxury_first (they are taken whenever a slot is free). v3.0: luxury
            talents only (fallback: the foundational / central ones, last resort)."""
            out = []
            for t in np.argsort(-s.q):
                it = self.c.ids[t]
                if self._kind(it) != kind or it in cur or it in dropped or it in rem_mem or it in rejected:
                    continue
                if not self._worth_ok(it, fallback, s) or (fallback and self._worth_ok(it, False, s)):
                    continue
                if self._cold(it, s) or self._side_block(it, cur, s):
                    continue
                if (self._feasible(it, cur, s, cap=cap) and not self._rule_block(it, cur, s)
                        and not self._depends_unmet(it, cur, s)):      # v3.6: x NW > y
                    out.append(it)
                if len(out) >= 8:
                    break
            if luxury_first:
                lux = [i for i in self.manual if self._lux(i) and i in self.c.index and self._kind(i) == kind
                       and i not in cur and i not in dropped and i not in rem_mem and i not in rejected
                       and not self._side_block(i, cur, s)
                       and self._feasible(i, cur, s, cap=cap) and not self._rule_block(i, cur, s)
                       and not self._depends_unmet(i, cur, s)]      # v3.6: x NW > y
                out = sorted(lux, key=lambda i: -q(i)) + [i for i in out if i not in lux]
            return out
        order = ("m", "t") if cfg.fill_capacity else ("t", "m")   # mantra first: each lowers the cap
        capacity()
        for k in order:
            if k == "t":
                capacity()
            while free[k] > 0:
                c = best_candidates(k)
                fb = False
                if not c and k == "t" and getattr(cfg, "rule_tiers", False) and getattr(cfg, "tier_fallback", False):
                    c, fb = best_candidates(k, luxury_first=False, fallback=True), True
                if not c:
                    break
                it = c[0]
                cur.add(it); free[k] -= 1
                changes.append(Change(it, "add", None, [], q(it), "worth", None, LUXURY,
                                      f"taken by {q(it):.0%} of similar builds that can take it"
                                      + (f"; no luxury talent fits ({self.tier(it)} talent, not required "
                                         "by a rule here)" if fb else "")))
        # 4. luxury swaps
        swaps = 0
        while swaps < cfg.max_swaps:
            best = None
            for k in ("t", "m"):
                d = displaceable(k, worth=True)
                c = best_candidates(k, cap=False, luxury_first=False)
                for cand in [x for x in c if x not in left_out][:3]:
                    # the least worthy pick whose slot the candidate can take
                    # (talent cap, mantra slot type)
                    out_ = next((v for v in d if v not in kept_by and self._feasible(cand, cur - {v}, s)), None)
                    if out_ is None:
                        continue
                    g = q(cand) - q(out_)
                    if g >= cfg.swap_margin and (best is None or g > best[0]):
                        best = (g, out_, cand)
                    break
            if best is None:
                break
            g, out_, in_ = best
            drop(out_, Reason.NO_SYNERGY.value if q(out_) < cfg.no_fit_rate else Reason.LUXURY_LESS.value,
                 [], "worth", f"{in_} is taken by {q(in_):.0%} of similar builds, this by {q(out_):.0%}")
            cur.add(in_)
            changes.append(Change(in_, "add", Reason.LUXURY_MORE.value, [], q(in_), "worth", None, LUXURY,
                                  f"taken by {q(in_):.0%} of similar builds that can take it"))
            swaps += 1
        # 4b. repair: every rule re-checked against the current answer
        for _ in range(cfg.repair_rounds):
            moved = False
            for it in list(cur):
                if it in keep_mem or it in cat_keep:
                    continue
                r = self._forbidden(it, cur, s)
                if r is not None and it in central and not self._clash_rule(r):
                    continue
                if r is not None:
                    reason, rel = self._never_reason(r)
                    drop(it, reason, rel, "rules", r.text(), EXCLUSION)
                    rejected.add(it); by_rule.add(it); protected.discard(it); free[self._kind(it)] += 1; moved = True
            for it in list(by_rule):                   # v3.0: picks (group counts, anchors) changed and the
                if self._forbidden(it, cur | {it}, s) is None:   # never rule no longer holds: a take rule may
                    rejected.discard(it); by_rule.discard(it)    # bring the item back
                    if it not in rem_mem:
                        hard_block.discard(it)
            req = self._required(s, cur, hard_block | rejected)
            moved |= add_required(req)
            moved |= trim_cap()
            moved |= trim_slots()
            capacity()
            for k in order:
                if k == "t":
                    capacity()
                while free[k] > 0:
                    c = best_candidates(k)
                    if not c:
                        break
                    cur.add(c[0]); free[k] -= 1; moved = True
                    changes.append(Change(c[0], "add", None, [], q(c[0]), "worth", None, LUXURY,
                                          f"taken by {q(c[0]):.0%} of similar builds that can take it"))
            if not moved:
                break
        # 5. iffy
        iffy: List[Tuple[float, str]] = []
        for k in ("t", "m"):
            c = best_candidates(k, cap=False, luxury_first=False)
            if not c:
                continue
            alt = q(c[0])
            for it in cur:
                if self._kind(it) == k and it not in protected and it in S and q(it) - alt < cfg.iffy_margin \
                        and not self._lux(it) and not self._cold(it, s):
                    iffy.append((q(it) - alt, it))
        lim = max(1, int(cfg.max_iffy_frac * len(S))) if S else 0
        iffy_items = [it for _, it in sorted(iffy)[:lim]]
        iffy_why: Dict[str, dict] = {}
        for it, v in s.memory.items():
            if v[0] == "iffy" and self._binds(v) and it in cur and it not in iffy_items:
                iffy_items.append(it)
            if v[0] == "iffy" and it in iffy_items and v[4] == Reason.SIDEGRADE.value and len(v) > 7:
                iffy_why[it] = dict(reason=v[4], related=list(v[7]), review_id=v[2])
        passes = {it: (req[it][0] if it in req else self._pass_here(it, s, cur)) for it in cur}
        if cfg.fill_capacity:
            cap = (len(self._talents & cur) + free["t"], sum(self._kind(i) == "m" for i in cur) + free["m"])
        alt = self._alt(b, s, cur, changes, iffy_items, passes, cap)
        alt.focus = dict(s.focus)
        alt.iffy_why = iffy_why
        alt.violations = self._violations(s, cur, hard_block | rejected, set(keep_mem))
        alt.sidegrades = self._offers(s, cur, set(locked) | set(keep_mem) | cat_keep |
                                      {i for i, v in req.items() if v[1] in ("review", "general", "category")})
        return alt

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
                if self._feasible(it, trial, s) and not self._rule_block(it, trial, s):
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
            if r.item in blocked and (r.item in rev_block or                 # v3.0: a never rule that held
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
