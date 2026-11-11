"""PassEngine methods: completion: required passes, the class focus, luxuries and mantra room."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tri.core.codec import is_silentheart
from tri.core.types import Build, Reason
from tri.infer.defs import REVIEW, Alternative, Change, _Sit
from tri.rules.passes import CENTRAL, FOUNDATIONAL, LUXURY


class _Completion:
    """PassEngine methods (PassEngine methods: completion: required passes, the class focus, luxuries and mantra room.)."""

    def _required(self, s: _Sit, sel: Set[str], blocked: Set[str]) -> "OrderedDict[str, tuple]":
        """Items the passes require for this build: item -> (pass, source, why, review_id).
        Central anchors are re-read as picks are added (up to central_rounds)."""
        req: "OrderedDict[str, tuple]" = OrderedDict()
        for it, why in (s.weapon or {}).get("require", {}).items():   # v3.2 weapon rules (math §2.8c)
            t = self.c.index.get(it)
            if t is not None and (it in sel or s.ctx.avail[t]):
                req[it] = (FOUNDATIONAL, "weapon", f"{it}: {why} (weapon rule)", None)
        for it, v in self._hard(s, "add").items():
            if it in req:
                continue                                  # v3.4: the weapon rule's entry stands
            if self._general(v):
                # v3.0.1: a generally-good claim made on another archetype yields to what
                # the reviews say about this one ([X] A = a ⇒ ¬t, strong SM here)
                if self._forbidden(it, sel, s) is None \
                        and self._rule_block(it, sel, s) is None:
                    req[it] = (FOUNDATIONAL, "general", f"generally good (foundational talent), review {v[2]}", v[2])
                continue
            req[it] = (REVIEW, "review", self._memory_why("added", v) + self._note(v[2]), v[2])
            if v[4] in (Reason.SYNERGY_WITH.value, Reason.LUXURY_MORE.value):
                s.req_why[it] = (v[4], list(v[7]) if len(v) > 7 else [])
        for it, v in self._hard(s, "remove").items():
            # NW: x does not work without a, b; keeping x (here: it is already picked
            # or pinned) is equivalent to adding a, b, so they are required
            if v[4] == Reason.NEEDS.value and it in sel and len(v) > 7:
                for y in v[7]:
                    if y not in sel and y not in req and y not in blocked and y in self.c.index:
                        req[y] = (REVIEW, "review", f"{it} needs {', '.join(v[7])} (review {v[2]}, "
                                                    f"similarity {v[3]:.2f})" + self._note(v[2]), v[2])
                        s.req_why[y] = (Reason.SYNERGY_WITH.value, [it])
        for it in sorted(getattr(self, "wanted", None) or ()):   # v3.6: "do you want X?" yes: take it
            t = self.c.index.get(it)                             # (still subject to the cap and slots)
            if t is not None and it not in req and it not in blocked and (it in sel or s.ctx.avail[t]):
                req[it] = (REVIEW, "answer", f"you said you want {it}", None)
        for r in self.review_rules:
            if r.kind == "NECESSARY" and r.applies(s.build, s.feats) and r.subject not in req:
                req[r.subject] = (CENTRAL, "review_rule", r.describe(), None)
        if self.manual:
            silent = is_silentheart(s.build, s.ctx.tokens)
            for it, c in self.manual.items():
                t = self.c.index.get(it)
                if t is None or it in req or (silent and self._kind(it) == "m"):
                    continue
                if c == FOUNDATIONAL:
                    req[it] = (FOUNDATIONAL, "category", f"{it} is foundational (category): taken on every build", None)
                elif c == CENTRAL and s.ctx.avail[t]:
                    req[it] = (CENTRAL, "category", f"{it} is central (category): taken whenever its "
                                                    f"requirements are met", None)
        for r in self.rb.firing(s.ctx, FOUNDATIONAL):
            if r.item not in self.manual:
                req.setdefault(r.item, (FOUNDATIONAL, "rules", r.text(self.rb.region_names), None))
        # existing picks anchor central rules only if they belong here: an
        # off-archetype pick must not pull in its own archetype's centrals (and so
        # protect itself); required picks always anchor
        cur = {x for x in sel if x in self.c.index and s.q[self.c.index[x]] >= self.cfg.anchor_min_worth} | set(req)
        for _ in range(self.cfg.central_rounds):
            s.ctx.items = cur - blocked
            new = [r for r in self._firing(s, CENTRAL) if r.item not in req
                   and r.item not in self.manual]
            if not new:
                break
            for r in new:
                req.setdefault(r.item, (CENTRAL, "rules", r.text(self.rb.region_names), None))
            cur |= {r.item for r in new}
        return OrderedDict((k, v) for k, v in req.items() if k not in blocked)
    # ------------------------------------------------------------ complete

    def complete(self, b: Build, capacity: Optional[Tuple[int, int]] = None,
                 region: Optional[int] = None, force: Optional[dict] = None) -> Alternative:
        """Complete b, then ask the reviews about the result (v3.0.4).

        A partial input sits far from the full builds the reviews were made on,
        so their claims do not reach it. The completed build does sit near them:
        a binding claim on the result (add an item it lacks, remove an item the
        completion added) is carried back and b is completed again, until no new
        claim binds or the result repeats (at most review_rounds rounds). The
        input's own picks stay."""
        own0 = set(b.items())
        self._ban_set = set()
        try:
            for _ in range(3):
                alt = self._complete_rounds(b, capacity, region, force)
                dang = self._dangling(set(alt.items()), own0)      # v3.2: a prerequisite left without its dependant
                if not dang:
                    return alt
                self._ban_set |= set(dang)
            return alt
        finally:
            self._ban_set = set()

    def _complete_rounds(self, b: Build, capacity=None, region=None, force=None) -> Alternative:
        alt = self._complete_once(b, capacity, region, force)
        if self.ledger is None:
            return self._through_review(b, alt, capacity, region, force)
        extra: Dict[str, tuple] = {}
        seen = {frozenset(alt.items())}
        own = set(b.items())
        for _ in range(int(getattr(self.cfg, "review_rounds", 3))):
            res = replace(b, talents=list(alt.talents), mantras=list(alt.mantras))
            mem = self.situations(res, 1, region, force=force)[0].memory
            its = set(alt.items())
            new = {k: v for k, v in mem.items() if k not in extra and k in self.c.index and self._binds(v) and
                   ((v[0] == "add" and k not in its) or (v[0] == "remove" and k in its and k not in own))}
            if not new:
                break
            extra.update(new)
            alt = self._complete_once(b, capacity, region, force, extra)
            key = frozenset(alt.items())
            if key in seen:
                break
            seen.add(key)
        return self._through_review(b, alt, capacity, region, force)

    def _through_review(self, b: Build, alt: Alternative, capacity=None, region=None, force=None) -> Alternative:
        """v3.2: the completed build is reviewed as a full build (the refine step of
        the original design: generate, check hard constraints, refine the complete
        set). The input's picks are pinned, except where a weapon rule forbids
        them; the review's edits (review claims on the result, weapon rules,
        required picks, swaps) are applied, up to review_pass_rounds times or until
        the review proposes nothing. Adds the review takes back never appear.
        capacity (an explicit lower limit) keeps the completion's counts."""
        rounds = int(getattr(self.cfg, "review_pass_rounds", 2))
        if rounds <= 0:
            return alt
        own = [i for i in b.items() if i in self.c.index]
        seen = {frozenset(alt.items())}
        for _ in range(rounds):
            res = replace(b, talents=list(alt.talents), mantras=list(alt.mantras))
            reg = region if region is not None or force is not None else alt.mode
            sits = self.situations(res, 1, reg, force=force)
            if not sits:
                break
            rv = self._review_one(res, sits[0], set(own) & set(res.items()), keep_size=capacity is not None)
            if not rv.changes:
                break
            added = {c.item for c in alt.changes if c.action == "add"}
            gone = {c.item for c in rv.changes if c.action == "remove"}
            merged = [c for c in alt.changes if not (c.action == "add" and c.item in gone)]
            back = {c.item for c in alt.changes if c.action == "remove"}
            for c in rv.changes:
                if c.action == "remove" and c.item in added:
                    continue                                 # the completion's own add, taken back
                if c.action == "add" and c.item in back:
                    merged = [m for m in merged if not (m.action == "remove" and m.item == c.item)]
                    continue                                 # a removal taken back
                merged.append(c)
            rv.changes = merged
            rv.capacity = alt.capacity if capacity is not None else rv.capacity
            rv.focus = dict(getattr(alt, "focus", None) or {})      # the completion's class focus decided
            alt = rv
            key = frozenset(alt.items())
            if key in seen:
                break
            seen.add(key)
        return alt

    def _merge_extra(self, memory: Dict[str, tuple], extra: Optional[Dict[str, tuple]]) -> None:
        """Carry the claims of the reviews of the result into the input's memory.
        v3.1.3: such a claim outranks a generally-good claim made on another
        build and any claim from a less similar build (the nearest review wins,
        math §5); before, a binding generally-good add kept a review's removal
        of the very result from taking effect."""
        for k, v in (extra or {}).items():
            old = memory.get(k)
            if old is None or not self._binds(old) or self._general(old) or float(v[3]) > float(old[3]):
                memory[k] = v

    # ------------------------------------------------------------ addition order (v3.2)
    @property
    def _dependants(self) -> Dict[str, List[str]]:
        d = getattr(self, "_deps_cache", None)
        if d is None:
            d = {}
            for x in self.c.ids:
                it = self.c.items[x]
                for p in set(it.prerequisites) | {y for g in it.prerequisites_any for y in g}:
                    d.setdefault(p, []).append(x)
            self._deps_cache = d
        return d

    @property
    def _ban(self) -> Set[str]:
        return getattr(self, "_ban_set", None) or set()

    def _prereq_only(self, it: str, sel: Set[str]) -> bool:
        """it is a prerequisite that the corpus takes almost only with one of the
        talents it unlocks (solo share below prereq_solo_min), and none is picked."""
        solo = (getattr(self, "solo", None) or {}).get(it)
        if solo is None or solo >= self.cfg.prereq_solo_min:
            return False
        return not any(d in sel for d in self._dependants.get(it, ()))

    def _dangling(self, sel: Set[str], own: Set[str]) -> List[str]:
        return sorted(x for x in sel if x not in own and x not in self.manual and self._prereq_only(x, sel))

    def _missing_chain(self, it: str, sel: Set[str], s: _Sit, blocked: Set[str]) -> Optional[List[str]]:
        """The prerequisites of it that sel lacks, transitively, in the order they
        are added (a prerequisite before its dependant); None when one cannot be
        taken (blocked, unavailable or ruled out)."""
        out: List[str] = []

        def ok(p):
            t = self.c.index.get(p)
            return t is not None and p not in blocked and p not in self._ban and bool(s.ctx.avail[t]) \
                and self._rule_block(p, sel, s) is None

        def need(x, depth=0):
            if depth > 8:
                return False
            I = self.c.items[x]
            for p in I.prerequisites:
                if p in sel or p in out:
                    continue
                if not ok(p) or not need(p, depth + 1):
                    return False
                out.append(p)
            for g in I.prerequisites_any:
                if any(p in sel or p in out for p in g):
                    continue
                for p in sorted((p for p in g if ok(p)), key=lambda p: -s.q[self.c.index[p]]):
                    keep = list(out)
                    if need(p, depth + 1):
                        out.append(p)
                        break
                    out[:] = keep
                else:
                    return False
            return True

        return out if need(it) else None

    def _complete_once(self, b: Build, capacity: Optional[Tuple[int, int]] = None,
                       region: Optional[int] = None, force: Optional[dict] = None,
                       extra: Optional[Dict[str, tuple]] = None) -> Alternative:
        """Fill b in pass order until no item fits; existing picks stay.

        The limits are the game's and move as picks are added: talents up to
        the talent cap 76 - 2m (m = mantras that are not Oath, Origin or
        Monster mantras), mantras up to the mantra slots of the result (base 3
        Combat, 1 Mobility, 1 Support, 1 Wildcard, plus what its oath and
        talents grant), each mantra in a slot of its type (math §2.6).
        capacity=(talents, mantras) only lowers them. Mantras are filled
        before talents, and an empty mantra slot left by a late slot talent is
        filled by giving up the least worthy talents this completion added.

        A build without mantras (and without the Silentheart oath) gets its
        mantras first: the talents that grant mantra slots (Neuroplasticity,
        Will o' Wisp, ...) it can take, then mantras by worth, then the usual
        passes. A Silentheart build gets no mantras: the oath makes them unusable."""
        s = self.situations(b, 1, region, partial=True, force=force)[0]
        self._merge_extra(s.memory, extra)              # claims the reviews make on the result
        have = [i for i in b.items() if i in self.c.index]
        n_m0 = sum(self._kind(i) == "m" for i in have)
        silent = is_silentheart(b, s.ctx.tokens)
        mantras_first = n_m0 == 0 and not silent
        limit = {"t": capacity[0], "m": capacity[1]} if capacity is not None else {"t": None, "m": None}
        limit = {k: (None if v is None else int(v)) for k, v in limit.items()}
        if silent:
            limit["m"] = n_m0
        sel: Set[str] = set(have)
        blocked = {i for i in self._hard(s, "remove")} - set((s.weapon or {}).get("guard", ()))
        changes: List[Change] = []
        for it, why in self._voi_drops(sel):            # never in a Pathfinder build
            sel.discard(it); blocked.add(it)
            changes.append(Change(it, "remove", Reason.SCENARIO_MISMATCH.value, [],
                                  float(s.q[self.c.index[it]]), "rules", None, FOUNDATIONAL, why))
        # v3.0.3: never ([X]) rules bind existing picks too, as in the analysis
        moved = True
        while moved:
            moved = False
            for it in sorted(sel, key=lambda x: s.q[self.c.index[x]] if x in self.c.index else 0.0):
                if it in self.manual:
                    continue
                r = self._forbidden(it, sel, s)
                if r is None:
                    continue
                rs, rel = self._never_reason(r)
                sel.discard(it); blocked.add(it); moved = True
                changes.append(Change(it, "remove", rs, rel, float(s.q[self.c.index[it]]), "rules", None,
                                      r.pass_, r.text(self.rb.region_names)))
                break
        have = [i for i in have if i in sel]
        passes: Dict[str, str] = {}
        count = lambda k: sum(self._kind(i) == k for i in sel)
        needed = lambda x: any(x in self.c.items[y].prerequisites or
                               any(x in g for g in self.c.items[y].prerequisites_any) for y in sel - {x}) \
            or self._slot_needed(x, sel, s)
        # the talent cap and the mantra slots are strict: an input above them
        # loses its lowest-worth picks that no other pick needs and no rule requires
        if self.cap_excess(sel) > 0 or not self.fits(sel, s):
            req0 = self._required(s, sel, blocked)
            for kind, over, note in (("t", lambda: self.cap_excess(sel) > 0, lambda: self._cap_note(sel)),
                                     ("m", lambda: not self.fits(sel, s), lambda: self._slot_note(sel, s))):
                cen = self._central(sel, s)
                for it in sorted((x for x in sel if self._kind(x) == kind),
                                 key=lambda x: (x in cen, s.q[self.c.index[x]])):
                    if not over():
                        break
                    if it in req0 or needed(it):
                        continue
                    why = note()
                    sel.discard(it)
                    changes.append(Change(it, "remove", Reason.LESS_NECESSARY_SPACE.value, [],
                                          float(s.q[self.c.index[it]]), "rules", None, LUXURY, why))
            have = [i for i in have if i in sel]

        worth_why = lambda it: f"taken by {s.q[self.c.index[it]]:.0%} of similar builds that can take it"
        fallback = [False]                                   # v3.0: F / C talents by worth only as a last resort

        def take(it, pass_, source, why, rid=None, reason=None, cold_ok=False, follow=True):
            k = self._kind(it)
            if limit[k] is not None and count(k) >= limit[k]:
                return False
            if source == "worth" and not self._worth_ok(it, fallback[0], s):
                return False
            rules = source not in ("review", "category")
            if it in blocked or it in self._ban or self._rule_block(it, sel, s):
                return False
            if source == "worth" and ((self._cold(it, s) and not cold_ok) or self._side_block(it, sel | intent, s)):
                return False
            if source == "worth" and self._depends_unmet(it, sel, s):
                return False                         # v3.6: x NW > y (review rule): never x without y
            if source == "worth" and follow and self._prereq_only(it, sel):
                # v3.2: a talent taken almost only to unlock another is taken with it
                # (the dependant pulls it in as a prerequisite), never alone
                for d in sorted((x for x in self._dependants.get(it, ()) if x not in sel),
                                key=lambda x: -s.q[self.c.index[x]]):
                    if s.q[self.c.index[d]] >= self.cfg.package_min_q and \
                            take(d, pass_, source, worth_why(d) + f" (with its prerequisite {it})",
                                 cold_ok=cold_ok, follow=False) and it in sel:
                        return True
                return False
            chain: List[str] = []
            if not self._feasible(it, sel, s, rules=rules):
                # v3.2 addition order: a pick whose prerequisites are missing brings
                # them in first, when the whole chain fits (math §2.8d)
                chain = self._missing_chain(it, sel, s, blocked) or []
                after = sel | set(chain) | {it}
                if not chain or any(limit[self._kind(y)] is not None and
                                    sum(self._kind(z) == self._kind(y) for z in after) > limit[self._kind(y)]
                                    for y in chain + [it]) or self.cap_excess(after) > 0 or not self.fits(after, s) \
                        or not all(self._feasible(y, after - {y}, s, cap=False, rules=rules) for y in chain + [it]):
                    return False
            for y in chain:
                sel.add(y); passes[y] = pass_
                changes.append(Change(y, "add", Reason.SYNERGY_WITH.value, [it], float(s.q[self.c.index[y]]), source,
                                      rid, pass_, f"prerequisite of {it}"))
            sel.add(it); passes[it] = pass_
            rel = []
            if source == "review" and it in s.req_why:
                reason, rel = s.req_why[it]
            changes.append(Change(it, "add", reason, list(rel), float(s.q[self.c.index[it]]), source, rid, pass_, why))
            return True

        intent: Set[str] = set(self._required(s, sel, blocked)) if self.sidegrades else set()   # sides follow the required picks
        if mantras_first:
            # mantra phase, in pass order: required mantras and slot talents,
            # then the slot talents the build can take, then mantras by worth
            is_mphase = lambda x: self._kind(x) == "m" or bool(self.c.slot_grants(x))
            for it, (ps, src, why, rid) in self._required(s, sel, blocked).items():
                if it not in sel and is_mphase(it):
                    take(it, ps, src, why, rid, self._add_reason(ps, src))
            for it in sorted((x for x in self._talents if self.c.slot_grants(x)), key=lambda x: -s.q[self.c.index[x]]):
                if it in sel or self._cold(it, s):
                    continue
                g = self.c.slot_grants(it)
                if set(g) <= {"Wisp"} and not any(self.c.mantra_slot(m) == "Wisp" and m not in blocked and
                                                  self._feasible(m, sel | {it}, s, cap=False)
                                                  for m in self.c.ids if self._kind(m) == "m"):
                    continue                                 # a Wisp slot holds only a Wisp mantra
                take(it, LUXURY, "rules", "grants a mantra slot; the build had no mantras")
            for t in np.argsort(-s.q):
                it = self.c.ids[t]
                if it not in sel and self._kind(it) == "m":
                    take(it, LUXURY, "worth", f"mantras first (the build had none); taken by {s.q[t]:.0%} "
                                              f"of similar builds that can take it")
        for _ in range(3):                                   # prerequisites may need a second sweep
            req = self._required(s, sel, blocked)
            moved = False
            for it, (ps, src, why, rid) in req.items():
                if it not in sel:
                    moved |= take(it, ps, src, why, rid, self._add_reason(ps, src))
            if not moved:
                break
        intent = set()
        if s.focus.get("classes"):                          # home in on the weapon class before luxuries
            from tri.infer.focus import class_table
            tab = class_table(self.c)
            cls = set(s.focus["classes"])
            for it in sorted((x for x, cs in tab.items() if cs & cls and self._kind(x) == "t"),
                             key=lambda x: -s.q[self.c.index[x]]):
                if it not in sel and s.q[self.c.index[it]] >= self.cfg.focus_min_q:
                    take(it, LUXURY, "worth", f"weapon class {'/'.join(sorted(tab[it] & cls))} ({self._focus_note(s)}); "
                                              f"taken by {s.q[self.c.index[it]]:.0%} of similar builds that can take it")
        for it in sorted((x for x, c in self.manual.items() if c == LUXURY and x in self.c.index),
                         key=lambda x: -s.q[self.c.index[x]]):      # luxury (category): taken when a slot is free
            if it not in sel and not self._side_block(it, sel, s):
                take(it, LUXURY, "category", f"{it} is luxury (category): taken when a slot is free")
        order = [self.c.ids[t] for t in np.argsort(-s.q)]
        added = lambda: [c.item for c in changes if c.action == "add" and c.source == "worth" and c.item in sel]
        for cold_ok in (False, True):                       # unrated items only when nothing rated fits
            for _ in range(4):
                n0 = len(sel)
                for k in ("m", "t"):                         # mantras first: each lowers the talent cap
                    for it in order:
                        if it not in sel and self._kind(it) == k:
                            take(it, LUXURY, "worth", worth_why(it), cold_ok=cold_ok)
                self._make_mantra_room(s, sel, blocked, changes, passes, order, added(), needed, limit,
                                       count, intent, cold_ok)
                if len(sel) == n0:
                    break
        if getattr(self.cfg, "rule_tiers", False) and getattr(self.cfg, "tier_fallback", False) and \
                (limit["t"] is None or count("t") < limit["t"]) and self.talent_cap(sel) > count("t"):
            fallback[0] = True                               # no luxury talent fits; the cap is unfilled
            for cold_ok in (False, True):
                for it in order:
                    if it not in sel and self._kind(it) == "t" and not self._worth_ok(it, False, s):
                        take(it, LUXURY, "worth", worth_why(it) + f"; no luxury talent fits ({self.tier(it)} "
                                                  "talent, not required by a rule here)", cold_ok=cold_ok)
            fallback[0] = False
        for it in have:
            passes.setdefault(it, self._pass_here(it, s, sel))
        cap_out = (self.talent_cap(sel) if limit["t"] is None else min(limit["t"], self.talent_cap(sel)),
                   (sum(self.c.mantra_slots_of(sel | s.outside, s.oath).values()) if not silent else 0)
                   if limit["m"] is None else limit["m"])
        alt = self._alt(b, s, sel, changes, [], passes, cap_out)
        alt.focus = dict(s.focus)
        alt.violations = self._violations(s, sel, blocked, set(have))
        alt.sidegrades = self._offers(s, sel, set(have) | self._cat_kept(s) |
                                      {i for i, v in self._required(s, set(sel), blocked).items()
                                       if v[1] in ("review", "general", "category")})
        return alt

    def _make_mantra_room(self, s, sel, blocked, changes, passes, order, added, needed, limit, count, intent,
                          cold_ok) -> None:
        """An empty mantra slot while the talents fill the cap: take the most
        worthy mantra that fits a free slot by giving up the least worthy
        talents this completion added (a mantra lowers the cap by cap_per)."""
        if limit["m"] is not None and count("m") >= limit["m"]:
            return
        while self.slot_free(sel, s) > 0:
            mine = sorted((x for x in added if x in sel and self._kind(x) == "t" and not needed(x)),
                          key=lambda x: s.q[self.c.index[x]])
            best = None
            for it in order:
                if it in sel or self._kind(it) != "m" or it in blocked or (self._cold(it, s) and not cold_ok):
                    continue
                if not self._feasible(it, sel, s, cap=False) or not self.fits(sel | {it}, s) or \
                        self._rule_block(it, sel, s) or self._side_block(it, sel | intent, s):
                    continue
                need = max(0, self.cap_excess(sel | {it}))
                out, after = [], sel | {it}
                for x in mine:                           # never a talent granting a slot the mantras use
                    if len(out) >= need:
                        break
                    if not self.c.slot_grants(x) or self.fits(after - {x}, s):
                        out.append(x); after = after - {x}
                if len(out) >= need and self.cap_excess(after) <= 0:
                    best = (it, out)
                    break
            if best is None:
                return
            it, out = best
            for x in out:
                sel.discard(x); passes.pop(x, None)
                for c in list(changes):
                    if c.item == x and c.action == "add":
                        changes.remove(c)
                added = [y for y in added if y != x]
            sel.add(it); passes[it] = LUXURY
            changes.append(Change(it, "add", None, [], float(s.q[self.c.index[it]]), "worth", None, LUXURY,
                                  f"fills an empty mantra slot; taken by {s.q[self.c.index[it]]:.0%} of similar "
                                  f"builds that can take it"
                                  + (f" (in place of {', '.join(out)})" if out else "")))

    @staticmethod
    def _focus_note(s: _Sit) -> str:
        f = s.focus or {}
        src = f.get("source")
        if src == "prior":
            c = (f.get("classes") or [""])[0]
            return f"{f.get('share', {}).get(c, 0):.0%} of similar builds use it"
        return {"weapon": "the build's weapon", "picks": "the build's weapon talents",
                "review": "a review's weapon talents", "worth": "its talents are worth most here"}.get(src, "")

    def _slot_note(self, sel, s: _Sit) -> str:
        u = self.c.mantra_slot_usage(sel | s.outside, s.oath)
        return "over the mantra slots (" + ", ".join(f"{k} {u['need'][k]}/{u['slots'][k]}"
                                                     for k in u["slots"] if u["slots"][k] or u["need"][k]) + \
            f"; {u['over']} mantra(s) fit no slot)"
