"""PassEngine methods: situations: reading the build, local worth, weapon-class focus and waivers."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import dataclasses

import numpy as np

from tri.core.types import Build, Kind
from tri.game import priors
from tri.game.weapon_rules import main_weapons, weapon_rules
from tri.infer.defs import _Sit
from tri.model.matrix import context_tokens
from tri.reviews.variants import family_key
from tri.rules.defs import Ctx
from tri.rules.groups import rule_tokens
from tri.rules.induction import context_features


class _Situations:
    """PassEngine methods (PassEngine methods: situations: reading the build, local worth, weapon-class focus and waivers.)."""

    # ------------------------------------------------------------ situation
    def situations(self, b: Build, max_alternatives: int = 3, region: Optional[int] = None,
                   partial: bool = False, force: Optional[dict] = None) -> List[_Sit]:
        """force: a forced archetype from Tri.force(): {name, key, latent} for a named
        archetype (its locators' mean latent), or {name, region} for a region."""
        b = self._answer_weapon(b)             # v3.2: the asker named the weapon
        pt = self.space.point(b, partial)
        ls = getattr(self, "locset", None)
        if ls is not None and hasattr(ls, "forget_query"):
            ls.forget_query()                  # v3.6: no reading carries over from an earlier query
        ls = ls if ls is not None and (ls.refining or (force or {}).get("key")) else None
        xa = ls.vec(b) if ls is not None else None
        rd = self.space.reading(pt.u, top=max(max_alternatives, 1))
        fpt = None
        if force is not None and force.get("latent") is not None:
            fpt = self.space.point(b, partial, latent=force["latent"])
        # v2.25.1: a build forced into an archetype asks the reviews of that archetype
        # (its forced coordinates), not of wherever its few picks happen to sit
        fv = fpt.u if fpt is not None else None
        if ls is not None and self.ledger is not None:   # v3.6: the query's own reading (forced archetype)
            emb = getattr(self.ledger, "embed", None)        # sets the review barrier of its claims
            uq = fv if fv is not None else (emb(b) if emb is not None else pt.u)
            ls.note(uq, xa=xa, force=(force or {}).get("key"), transient=True)
        memory = self.ledger.claims_for(b, vec=fv) if self.ledger is not None else {}
        b = self._waive(b, memory)
        b, wr0, memory = self._weapon_base(b, memory)   # v3.2 hardcoded weapon rules (math §2.8c)
        if fpt is not None:                    # placed again with the waived build, as before
            fpt = self.space.point(b, partial, latent=force["latent"])
        avail = self.c.availability_in(b)      # met at pre-shrine or final stats
        if wr0.forbid:
            avail = list(avail)
            for it in wr0.forbid:
                if it in self.c.index:
                    avail[self.c.index[it]] = False
        wh = [self.c.index[i] for i in (getattr(self, "withheld", None) or ()) if i in self.c.index and i not in b.items()]
        if wh:                                 # v3.1.1: adds held for thin evidence are not taken here
            avail = list(avail)
            for t in wh:
                avail[t] = False
        if memory:                                   # a claim made on a build on the other side of one of the
            for it in list(memory):                  # item's rule conditions (stat / context) does not transfer
                src = self.ledger.by_id.get(memory[it][2])
                if len(memory[it]) > 6 and memory[it][6] == "general":
                    continue                         # generally good: not tied to the source build's stats
                if src is not None and src.original.build_id != b.build_id and self.rb.separates(it, src.original, b):
                    del memory[it]
        if fpt is not None:
            regs = [(int(self.space.region_of(fpt.u[None])[0]), force["name"], 1.0)]
        elif force is not None and force.get("region") is not None:
            region = int(force["region"])
            regs = [(region, force["name"], 1.0)]
        elif region is not None:
            regs = [(region, self.space.region_names[region] if region < len(self.space.region_names) else "", 1.0)]
        else:
            regs = [r for i, r in enumerate(rd.regions) if i == 0 or r[2] >= self.cfg.min_plausibility]
            regs = regs[:max_alternatives] or [(0, "", 1.0)]
        out = []
        for i, (k, name, p) in enumerate(regs):
            ptq = fpt if fpt is not None else pt if i == 0 and region is None else self.space.point(b, partial, region=k)
            fine = ls.note(ptq.u, xa=xa, force=(force or {}).get("key"), transient=True) if ls is not None else None
            lr = self.worth.rates(ptq, picks=[self.c.index[i] for i in b.items() if i in self.c.index])
            q = lr.q.copy()
            if fine is not None:                                     # overlapping archetypes: the few picks
                q = ls.adjust(q, ptq.u, fine)                       # that tell them apart (math §6.6)
                if force is None and region is None and i == 0:
                    j = int(np.argmax(fine))                         # named by the fine reading, not the region
                    if fine[j] >= ls.cfg.min_confidence and ls.read_many(ptq.u[None])[1][0] <= ls.reach:
                        auto = (self.space.region_auto or [])
                        nm = ls.names[ls.keys[j]]
                        name = f"{nm} ({auto[k]})" if k < len(auto) and auto[k] else nm
            q = priors.adjust_q(q, self.c, b)                        # benchmarks, weapon redundancy (math §2.9)
            for it, v in memory.items():                             # soft claims nudge worth
                lab, st = v[0], v[1]
                t = self.c.index.get(it)
                if t is not None and not self._binds(v) and lab != "keep":
                    # (an unmarked pick on a similar build is already counted
                    # once through worth; it is not a reason to prefer it here)
                    q[t] = float(np.clip(q[t] + self.cfg.memory_nudge * st * (1 if lab in ("keep", "add") else -1 if lab == "remove" else 0), 0, 1))
            ind = self._notes_q(q, ptq.u, fine)
            fam = None
            if ls is not None and (force or {}).get("key") in getattr(ls, "keys", ()):
                fam = force["key"]                                   # v2.25.1: forced: its family, always
            elif ls is not None and fine is not None and len(ls.keys) >= 2:
                j = int(np.argmax(fine))
                if fine[j] >= ls.cfg.min_confidence and ls.read_many(ptq.u[None])[1][0] <= ls.reach:
                    fam = ls.keys[j]
            arch = family_key(ls.names.get(fam, fam)) if fam is not None else None
            wr = self._forbid_chain(weapon_rules(b, arch or (force or {}).get("name") or name))
            mem = self._same_family(memory, arch)
            guard = self._weapon_guard(wr, avail)
            if guard:                                                # v3.4: a weapon-required pick and its prerequisite
                mem = {k: v for k, v in memory.items()                # chain outrank every review claim against them
                       if not (k in guard and v[0] in ("remove", "absent"))}
            av, foc = self._focus(b, mem, avail, q, ptq.u)
            ctx = Ctx({s: int(v) for s, v in b.highest_stats.items()}, set(b.items()),
                      set(context_tokens(b)) | rule_tokens(b, archetype=arch), k, av,
                      groups=getattr(self.rb, "groups", None) or None)       # v3.0 literals
            feats = context_features(b) | {f"region:{k}"}
            if fam is not None:                                      # scope of OFF_ARCHETYPE rules (strong SM)
                feats.add(f"family:{family_key(ls.names.get(fam, fam))}")
            out.append(_Sit(b, ctx, feats, q, lr.support, mem, k, name, p, ptq.u,
                            self.c.oath_of(b), {i for i in b.items() if i not in self.c.index}, indicative=ind,
                            family=fam, focus=foc,
                            weapon=dict(forbid={i: w for i, w in wr.forbid.items() if i in self.c.index},
                                        require={i: w for i, w in wr.require.items() if i in self.c.index},
                                        guard=sorted(guard))))
        return out

    def _same_family(self, memory: dict, arch) -> dict:
        """v3.6 (math §5): a claim from a review whose reviewer states the build's
        archetype ("is" note) does not cross into another family: borders are
        chasms. Generally-good claims and claims on the same build are kept."""
        st = getattr(self, "stated", None) or {}
        if not arch or not st or not memory:
            return memory
        same = self.ledger.cfg.sim_same if self.ledger is not None else 1.0
        out = {}
        for it, v in memory.items():
            fam = st.get(str(v[2]).replace("pool-", "", 1)) if v[2] is not None else None
            if (fam and fam != arch and not (len(v) > 6 and v[6] == "general")
                    and not (len(v) > 3 and v[3] >= same)):
                continue
            out[it] = v
        return out

    def _notes_q(self, q: np.ndarray, u: np.ndarray, fine: Optional[np.ndarray]) -> Dict[str, dict]:
        """Review notes and admin values on worth, in place (math §5.6):
          indicator / core   q_t += note_worth * w,   w = K(u, u_r) * p(a | u)
          avoid              q_t -= note_worth * w
          value              q_t += clip(delta_t, -value_cap, value_cap)   (every build)
        K is the archetype kernel at the noted build u_r, zero beyond reach; p(a | u)
        the reading of the note's archetype (1 when it has none). Returns the
        components each indicator / core note rated, for the answer."""
        out: Dict[str, dict] = {}
        ls = getattr(self, "locset", None)
        h = float(ls.h) if ls is not None else float(self.space.cfg.bandwidth)
        reach = float(ls.reach) if ls is not None else np.inf
        P = None
        for n in self.note_worth:
            d = float(np.sqrt(np.maximum(((np.asarray(u, float) - np.asarray(n["u"], float)) ** 2).sum(), 0.0)))
            if d > reach:
                continue
            pa = 1.0
            if n.get("key") is not None and ls is not None and n["key"] in ls.kix:
                if fine is not None:
                    pa = float(fine[ls.kix[n["key"]]])
                else:
                    P = ls.read_many(np.asarray(u, float)[None])[0][0] if P is None else P
                    pa = float(P[ls.kix[n["key"]]])
            w = float(np.exp(-d * d / (2 * h * h))) * pa * float(n.get("scale", 1.0))   # v3.1: answer consensus is scaled down
            if ls is not None and w >= self.cfg.note_floor and len(ls.keys) >= 2:
                w *= float(ls.barrier_factor(np.asarray(u, float), np.asarray(n["u"], float)[None])[0])  # the chasm
            if w < self.cfg.note_floor:
                continue
            sign = -1.0 if n["kind"] == "avoid" else 1.0
            for t in n["items"]:
                q[t] = float(np.clip(q[t] + sign * self.cfg.note_worth * w, 0, 1))
                if sign > 0:
                    it = self.c.ids[t]
                    if it not in out or out[it]["weight"] < w:
                        out[it] = dict(archetype=n.get("archetype"), kind=n["kind"], review_id=n.get("review_id"),
                                       weight=round(w, 3))
        tells = getattr(ls, "tells", None) if ls is not None else None
        if tells and ls.cfg.consensus:                       # reviewer consensus (math §8.5a)
            Pc, dm, _ = ls.read_many(np.asarray(u, float)[None])
            Pu = fine if fine is not None else Pc[0]
            act = float(ls.activity(dm)[0])
            eta = float(ls.cfg.consensus_worth)
            for t in tells:
                w = float(Pu[ls.kix[t["key"]]]) * act * float(t["confidence"])
                if w < self.cfg.note_floor:
                    continue
                q[t["t"]] = float(np.clip(q[t["t"]] + t["sign"] * eta * w, 0, 1))
                if t["sign"] > 0:
                    it = t["item"]
                    if it not in out or out[it]["weight"] < w:
                        out[it] = dict(archetype=ls.names[t["key"]], kind="consensus", review_id=None,
                                       weight=round(w, 3), reviewers=t.get("reviewers"))
        cap = float(self.cfg.value_cap)
        for it, v in self.values.items():
            t = self.c.index.get(it)
            if t is not None:
                q[t] = float(np.clip(q[t] + np.clip(v, -cap, cap), 0, 1))
        return out
    # ------------------------------------------------------------ feasibility

    def _voi_drops(self, sel: Set[str]) -> List[Tuple[str, str]]:
        """(item, why) for the picks a Pathfinder build cannot hold: the Vow of
        Iron only picks, then every pick that requires one of them."""
        voi = sorted(i for i in sel if self.c.voi_only(i))
        if not voi:
            return []
        out = [(i, "only obtainable in Vow of Iron, not in a Pathfinder build") for i in voi]
        out += [(i, f"requires {', '.join(voi)}, which is only obtainable in Vow of Iron")
                for i in self.c.dependents(voi, sel)]
        return out

    def _focus(self, b: Build, memory: Dict[str, tuple], avail: np.ndarray, q: np.ndarray, u) -> Tuple[np.ndarray, dict]:
        """(availability with the other classes' weapon talents switched off,
        focus) for one situation (tri.focus, math §2.10). The build's own
        picks and items a binding review adds or keeps stay available."""
        if not self.cfg.focus:
            return avail, {}
        from tri.infer import focus as FO
        tab = FO.class_table(self.c)
        if not tab:
            return avail, {}
        allowed, src = FO.evidence(b, self.c, memory, self._binds)
        share: Dict[str, float] = {}
        cand: Dict[str, List[float]] = {}
        wa = getattr(self, "weapon_answer", None)
        if not allowed and wa:                          # v3.1.3: the asker named the class (math §14.3)
            allowed, src = {wa}, "answer"
        if not allowed:
            for it, cs in tab.items():
                t = self.c.index[it]
                if avail[t] and self.c.items[it].kind == Kind.TALENT:
                    for c in cs:
                        cand.setdefault(c, []).append(float(q[t]))
            if not cand:
                return avail, {}
            prior = self.class_prior
            if prior is not None:
                sh, k = prior.share(u)
                sh = {c: v for c, v in sh.items() if c in cand}
                if sh and k >= self.cfg.focus_min_labelled:
                    share, src = sh, "prior"
            if not share:
                share = {c: float(np.mean(sorted(v, reverse=True)[:3])) for c, v in cand.items()}
                src = "worth"
            allowed = {max(share, key=lambda c: (share[c], c))}
        keep = set(b.items()) | {it for it, v in memory.items() if v[0] in ("add", "keep") and self._binds(v)}
        av = avail.copy()
        off = []
        for it, cs in tab.items():
            if not (cs & allowed) and it not in keep:
                t = self.c.index[it]
                if av[t]:
                    av[t] = False
                    off.append(it)
        return av, dict(classes=sorted(allowed), source=src,
                        share={c: round(v, 3) for c, v in sorted(share.items(), key=lambda x: -x[1])[:4]},
                        off=len(off), candidates=sorted(cand, key=lambda c: -max(cand[c]))[:6])

    def _answer_weapon(self, b: Build) -> Build:
        """The weapon the asker named (Answers.weapon, a weapon name) when the build
        records none: it is read as equipped (v3.2, math §14.3)."""
        w = getattr(self, "weapon_name", None)
        if main_weapons(b):
            return b
        if w:
            return dataclasses.replace(b, meta={**(b.meta or {}), "weapon": w})
        wa = getattr(self, "weapon_answer", None)      # v3.6: a class answer feeds the weapon rules
        if wa:
            return dataclasses.replace(b, meta={**(b.meta or {}), "weapon_class": wa})
        return b

    def _weapon_base(self, b: Build, memory: Dict[str, tuple]):
        """(build with the rules' lock waivers, the archetype-free rules, memory
        without claims that keep or add a forbidden pick). math §2.8c."""
        wr = self._forbid_chain(weapon_rules(b))
        if wr.waive:                                  # the waiver covers the prerequisite chain (Knight's Rally > Turtle Shell)
            waive = set(wr.waive) | {x for w in wr.waive for x in self.c.prereq_closure(w)}
            wr = dataclasses.replace(wr, waive=waive)

            from tri.core.context import wield_waived
            b = dataclasses.replace(b, meta={**(b.meta or {}), "wield_waived": sorted(wield_waived(b) | wr.waive)})
        if wr.forbid and memory:
            memory = {k: v for k, v in memory.items() if not (k in wr.forbid and v[0] in ("add", "keep"))}
        return b, wr, memory

    def _forbid_chain(self, wr):
        """v3.8.1: a weapon rule's never(x) also forbids every pick whose hard
        prerequisite chain contains x (removed, unavailable, kept out of review)."""
        if not wr.forbid:
            return wr
        forbid = dict(wr.forbid)
        for it in self.c.ids:
            if it in forbid:
                continue
            todo, seen = list(self.c.items[it].prerequisites), set()
            while todo:
                x = todo.pop()
                if x in seen:
                    continue
                seen.add(x)
                if x in wr.forbid:
                    forbid[it] = f"needs {x}: {wr.forbid[x]}"
                    break
                if x in self.c.index:
                    todo.extend(self.c.items[x].prerequisites)
        return wr if len(forbid) == len(wr.forbid) else dataclasses.replace(wr, forbid=forbid)

    def _weapon_guard(self, wr, avail) -> set:
        """v3.4: the weapon-required picks whose stats are met, with their hard
        prerequisite chain (Knight's Rally for Turtle Shell). Only game data
        (stats, exclusive_with, the cap) can keep them out (math §2.8c)."""
        out: set = set()
        for it in wr.require:
            t = self.c.index.get(it)
            if t is None or not avail[t]:
                continue
            todo = [it]
            while todo:
                x = todo.pop()
                if x in out or x not in self.c.index:
                    continue
                out.add(x)
                todo.extend(self.c.items[x].prerequisites)
        return out

    def _waive(self, b: Build, memory: Dict[str, tuple]) -> Build:
        """A review of this very build that adds or keeps an item blocked only by
        a weapon lock waives that lock here (math §2.8, 2.9a): the build carries
        another weapon, wields one it did not select, or the talent works
        without it. Only the reviewed build itself (sim >= sim_same)."""
        if self.ledger is None or not memory:
            return b
        same = self.ledger.cfg.sim_same
        # v3.1.3: a talent a reviewer strongly marked generally good is used without
        # its weapon class (Turtle Shell, Unstoppable Force): the lock is waived anywhere
        cand = [it for it, v in memory.items() if isinstance(it, str) and it in self.c.index
                and v[0] in ("add", "keep") and (v[3] >= same or (
                    len(v) > 5 and v[4] == "generally_good" and v[5] == "strong"))]
        todo = list(cand)                             # v3.2: the prerequisite chain shares the waiver
        while todo:
            for x in self.c.prereq_closure(todo.pop()):
                if x not in cand:
                    cand.append(x); todo.append(x)

        from tri.core.context import wield_waived
        want: set = set()
        while True:                                   # v3.0.3: a waived lock can unlock a dependant (Knight's Rally > Turtle Shell)
            cur = b if not want else dataclasses.replace(
                b, meta={**(b.meta or {}), "wield_waived": sorted(wield_waived(b) | want)})
            new = {it for it in cand if it not in want and self.c.wield_only_block(it, cur)}
            if not new:
                return cur
            want |= new
