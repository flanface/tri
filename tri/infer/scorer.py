"""PassEngine methods: the 4.0 scorer.

One per-item logit on each situation (docs/scorer.md):

    z_t = b + a_f L_t + a_n logit(q_t) + a_d D_t + a_p P_t

    L_t   field logit at the build's point (tri.model.space)
    q_t   corpus-only neighbour pick rate (no reviewed rows), with the game
          priors, admin values and reviewer notes applied (tri.game.priors)
    D_t   direct review evidence: sum over active claims c on t within reach
          of sign(c) w_c K_c(b), keep claims scaled by keep_weight
    P_t   propagated review evidence on similar unclaimed items (tri.reviews.reach)

Hard constraints and authored rules filter or force items; the rest is
filled by score under the talent cap, the mantra slots and the exclusions.
Binding claims (the reviewed build itself) act as before: a review
reproduces on its own build.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tri.core.codec import is_silentheart
from tri.core.types import Build, Reason
from tri.game import priors
from tri.reviews.authority import build_weapons
from tri.infer.defs import REVIEW, Alternative, Change, _Sit
from tri.rules.passes import FOUNDATIONAL, LUXURY

_SIGN = {"add": 1.0, "keep": 1.0, "remove": -1.0}
#: reasons about the item itself: a directed reach (4.0, docs/scorer.md)
ITEM_REASONS = {Reason.NECESSARY_FOR_ARCHETYPE.value, Reason.GENERALLY_GOOD.value, Reason.NO_SYNERGY.value,
                Reason.SCENARIO_MISMATCH.value, Reason.BROKEN.value}
#: reasons about archetype fit: propagated to similar items
FIT_REASONS = {Reason.NECESSARY_FOR_ARCHETYPE.value, Reason.NO_SYNERGY.value, Reason.SCENARIO_MISMATCH.value}
_REMOVE_REASONS = {r.value for r in Reason} - {Reason.NECESSARY_FOR_ARCHETYPE.value, Reason.GENERALLY_GOOD.value,
                                               Reason.SYNERGY_WITH.value, Reason.LUXURY_MORE.value,
                                               Reason.SIDEGRADE.value}


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


class _Scoring:
    """PassEngine methods: the 4.0 scorer and its slot filler."""

    # ------------------------------------------------------------ the logit
    def _review_rows(self) -> np.ndarray:
        rows = self.__dict__.get("_rev_rows")
        if rows is None:
            isr = getattr(self.worth, "is_review", None)
            rows = self._rev_rows = np.flatnonzero(isr) if isr is not None else np.zeros(0, int)
        return rows

    def _score(self, b: Build, ptq, fine, arch: Optional[str], vec=None) -> dict:
        """The scorer's parts on one situation: {z, L, q, D, P, why}."""
        sc = self.cfg
        picks = [self.c.index[i] for i in b.items() if i in self.c.index]
        lr = self.worth.rates(ptq, exclude=() if sc.sc_nbr_reviews else self._review_rows(), picks=picks)
        q = lr.q.copy()
        ls = getattr(self, "locset", None)
        if fine is not None and ls is not None:                 # overlapping archetypes: the few picks
            q = ls.adjust(q, ptq.u, fine)                       # that tell them apart
        q = priors.adjust_q(q, self.c, b)                       # benchmarks, weapon redundancy
        ind = self._notes_q(q, ptq.u, fine)                     # reviewer notes, admin values
        L = np.asarray(ptq.logits, float)
        u = np.asarray(vec if vec is not None else ptq.u, float)
        D, why = self._direct(b, vec, arch, u)
        P, pwhy = self._propagated(b, vec, arch, u, why)
        z = sc.sc_bias + sc.sc_field * L + sc.sc_nbr * _logit(q) + sc.sc_direct * D + sc.sc_prop * P
        return dict(z=z, L=L, q=q, D=D, P=P, why=why, pwhy=pwhy, support=lr.support, indicative=ind,
                    vec=vec, arch=arch, u=np.asarray(ptq.u, float))

    def _claims_in_reach(self, b: Build, vec, arch: Optional[str], floor: Optional[float] = None):
        """(claim, review, similarity, global) for every active keep / add / remove
        claim that may reach b: reviews at similarity >= floor (default the ledger's
        sim_min; similarity includes the chasm), same weapon class unless the same
        build, not across a stated family, holding on b's items. Context-free
        claims (BROKEN, generally good) reach every build (global True)."""
        led = self.ledger
        if led is None or not led.reviews:
            return []
        lo = led.cfg.sim_min if floor is None else floor
        S = led.sims_to(b, vec)
        items = led._context(b)
        mine = build_weapons(b)
        st = getattr(self, "stated", None) or {}
        same = led.cfg.sim_same
        out, seen = [], set()
        for i in np.flatnonzero(S >= lo)[::-1]:                  # newest first
            r, sim = led.reviews[i], float(S[i])
            theirs = build_weapons(getattr(r, "original", None))
            if mine and theirs and sim < same and not (mine & theirs):
                continue
            fam = st.get(r.review_id)
            if arch and fam and fam != arch and sim < same:
                continue                                       # borders are chasms (stated family)
            for c in led.active_claims(r.review_id):
                if c.label not in _SIGN or c.item not in self.c.index or not c.holds_in(items):
                    continue
                if sim < same and not c.general and self.rb.separates(c.item, r.original, b):
                    continue
                out.append((c, r, sim, False))
                seen.add((r.review_id, c.item))
        for rid, c in led._global_claims():
            if (rid, c.item) in seen or c.label not in _SIGN or c.item not in self.c.index or not c.holds_in(items):
                continue
            out.append((c, led.by_id[rid], 1.0, True))
        return out

    def _kernel(self, c, r, sim: float, glob: bool, b: Build, u) -> float:
        """K_c(b). Context-free claims and the same build: 1. Explicit adds and
        removes with an item-level reason, when sc_directed: the directed kernel
            K = chi exp(-par^2 / 2 h_par^2) exp(-(|d|^2 - par^2) / 2 h_perp^2),
            d = u_b - u_r, par = d . g_t,
        g_t the unit direction of item t's field logit and chi the chasm.
        Other claims: the review's similarity. K >= sim_full counts as 1;
        K < sim_min as 0."""
        cfg = self.ledger.cfg
        if glob or sim >= cfg.sim_full:
            return 1.0
        k = sim
        if self.cfg.sc_directed and c.label in ("add", "remove") and c.reason in ITEM_REASONS and u is not None:
            k = self._directed(self.c.index[c.item], r.review_id, sim, u)
        if k >= cfg.sim_full:
            return 1.0
        return k if k >= cfg.sim_min else 0.0

    def _directed(self, t: int, rid: str, sim: float, u) -> float:
        led = self.ledger
        d = np.asarray(u, float) - led.vector(rid)
        d2 = float(d @ d)
        h2 = led.cfg.reach ** 2 / (-2.0 * np.log(led.cfg.sim_min))
        iso = float(np.exp(-d2 / (2 * h2)))
        chi = min(1.0, sim / iso) if iso > 1e-12 else 0.0       # the chasm, as sims_to applied it
        g = self._item_dirs()[t]
        par = float(d @ g)
        hp, hq = self.cfg.sc_h_par, self.cfg.sc_h_perp
        return chi * float(np.exp(-par * par / (2 * hp * hp)) * np.exp(-max(d2 - par * par, 0.0) / (2 * hq * hq)))

    def _item_dirs(self) -> np.ndarray:
        """[N, D] unit gradient of each item's field logit in concept
        coordinates: logit_t = const + sqrt(n) omega_t^(-1/2) Q_t . u."""
        G = self.__dict__.get("_gdir")
        if G is None:
            Q = np.asarray(self.space.Q, float)
            G = self._gdir = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        return G

    def _reach_floor(self) -> Optional[float]:
        """Reviews this similar or more are read; a directed kernel reaches
        further across the item direction (sc_h_perp > sc_h_par)."""
        if not self.cfg.sc_directed or self.ledger is None:
            return None
        cfg = self.ledger.cfg
        h2 = cfg.reach ** 2 / (-2.0 * np.log(cfg.sim_min))
        hq = max(self.cfg.sc_h_perp, self.cfg.sc_h_par)
        return float(cfg.sim_min ** (hq * hq / h2))       # iso similarity where the widest kernel reaches sim_min

    def _direct(self, b: Build, vec, arch: Optional[str], u=None) -> Tuple[np.ndarray, Dict[str, tuple]]:
        """D [N] and, per item, the claim with the largest |contribution|."""
        D = np.zeros(len(self.c.ids))
        why: Dict[str, tuple] = {}
        kw = float(self.cfg.sc_keep)
        for c, r, sim, glob in self._claims_in_reach(b, vec, arch, self._reach_floor()):
            k = self._kernel(c, r, sim, glob, b, u)
            if k <= 0:
                continue
            w = c.weight * k * (kw if (c.label == "keep" or c.origin == "echo") else 1.0)
            t = self.c.index[c.item]
            D[t] += _SIGN[c.label] * w
            if c.item not in why or abs(w) > why[c.item][0]:
                why[c.item] = (abs(w), c, r.review_id, sim)
        return D, why

    def _propagated(self, b: Build, vec, arch, u, why) -> Tuple[np.ndarray, Dict[str, tuple]]:
        """P [N] (sc_propagate): an explicit add / remove of item t with an
        archetype-fit reason nudges each unclaimed item s by
            sign w_c K_c max(0, cos(s, t) - tau) / (1 - tau),
        cos the mean-centred worth-space cosine (Tri.item_cosine). Never
        binds; skipped across the two sides of a sidegrade."""
        N = len(self.c.ids)
        P = np.zeros(N)
        C = getattr(self, "item_cos", None)
        if not self.cfg.sc_propagate or C is None or not why:
            return P, {}
        tau = float(self.cfg.sc_tau)
        claimed = np.zeros(N, bool)
        for it in why:
            claimed[self.c.index[it]] = True
        opp = self._opposite_sides()
        out: Dict[str, tuple] = {}
        for c, r, sim, glob in self._claims_in_reach(b, vec, arch, self._reach_floor()):
            if c.label not in ("add", "remove") or c.reason not in FIT_REASONS:
                continue
            k = self._kernel(c, r, sim, glob, b, u)
            if k <= 0:
                continue
            t = self.c.index[c.item]
            m = np.maximum(C[t] - tau, 0.0) / (1.0 - tau)
            m[claimed] = 0.0
            m[t] = 0.0
            for s_ in opp.get(t, ()):
                m[s_] = 0.0
            P += _SIGN[c.label] * c.weight * k * m
        return P, out

    def _opposite_sides(self) -> Dict[int, Set[int]]:
        """item -> items on the other side of a sidegrade."""
        o = self.__dict__.get("_opp")
        if o is None:
            o = {}
            for g in getattr(self, "sidegrades", None) or ():
                A = [self.c.index[i] for i in g.a if i in self.c.index]
                B = [self.c.index[i] for i in g.b if i in self.c.index]
                for x in A:
                    o.setdefault(x, set()).update(B)
                for y in B:
                    o.setdefault(y, set()).update(A)
            self._opp = o
        return o

    # ------------------------------------------------------------ selection
    def _review_scored(self, b: Build, s: _Sit, locked: Set[str] = frozenset(), keep_size: bool = False,
                       fill_min: Optional[float] = None) -> Alternative:
        """The answer by score: hard drops, binding claims, forced picks, then
        removals below remove_below, fills by score and swaps above swap_gain."""
        cfg = self.cfg
        z = s.score["z"]
        zz = lambda it: float(z[self.c.index[it]])
        qv = lambda it: float(s.q[self.c.index[it]])
        S0 = [i for i in dict.fromkeys(b.items()) if i in self.c.index]
        cur: Set[str] = set(S0)
        changes: List[Change] = []
        blocked: Set[str] = set()                     # removed for cause: never re-added

        def drop(it, reason, related, source, why_, pass_=LUXURY, rid=None):
            cur.discard(it); blocked.add(it)
            changes.append(Change(it, "remove", reason, list(related), qv(it), source, rid, pass_, why_))

        def add(it, reason, related, source, why_, pass_=LUXURY, rid=None):
            cur.add(it)
            changes.append(Change(it, "add", reason, list(related), qv(it), source, rid, pass_, why_))

        # 0. game constraints: Vow of Iron, weapon rules, never rules
        for it, w in self._voi_drops(cur):
            drop(it, Reason.SCENARIO_MISMATCH.value, [], "rules", w, FOUNDATIONAL)
        for it, w in (s.weapon or {}).get("forbid", {}).items():
            if it in cur:
                drop(it, Reason.SCENARIO_MISMATCH.value, [], "weapon", w, FOUNDATIONAL)
        keep_mem = {i: v for i, v in (self._hard(s, "keep") | self._hard(s, "add")).items()
                    if not self._general(v) and i not in blocked}
        keep_mem.update({i: ("keep", 1.0, None, 1.0, None, None) for i in locked if i in cur})
        guard = set((s.weapon or {}).get("guard", ()))
        for it in sorted(cur, key=zz):
            if it in keep_mem or it in guard or it not in cur:
                continue
            r = self._forbidden(it, cur, s)
            if r is not None:
                reason, rel = self._never_reason(r)
                drop(it, reason, rel, "rules", r.text())
        for it in sorted(cur, key=zz):                 # mutual exclusions: the lower score goes
            if it in cur and it not in keep_mem and any(
                    x in cur and (x in self.c.items[it].exclusive_with or it in self.c.items[x].exclusive_with)
                    and (x in keep_mem or zz(x) >= zz(it)) for x in cur - {it}):
                drop(it, Reason.WORKS_AGAINST.value, [x for x in cur if x in self.c.items[it].exclusive_with],
                     "rules", "mutually exclusive")

        # 1. binding removals (a review of this or a same build)
        needs: Dict[tuple, list] = {}
        for it, v in self._hard(s, "remove").items():
            if it not in cur or it in locked:
                continue
            reason, rel = (v[4], list(v[7])) if len(v) > 7 else (v[4] if len(v) > 4 else None, [])
            if reason == Reason.NEEDS.value and rel:
                needs.setdefault(tuple(sorted(rel)), []).append((it, v))
                continue
            drop(it, reason or Reason.NO_SYNERGY.value, rel, "review",
                 self._memory_why("removed", v) + self._note(v[2]), REVIEW, v[2])
        cap = (sum(self._kind(i) == "t" for i in S0), sum(self._kind(i) == "m" for i in S0))
        for rel, xs in needs.items():                  # NW: add the needed picks when there is room, else remove
            xs = [m for m in xs if m[0] in cur]
            miss = [y for y in rel if y not in cur]
            if xs and self._room_for(miss, cur, s, cap, blocked):
                for y in miss:
                    add(y, Reason.SYNERGY_WITH.value, [x for x, _ in xs], "review",
                        f"{', '.join(x for x, _ in xs)} need {', '.join(rel)} (review {xs[0][1][2]})", REVIEW, xs[0][1][2])
                    keep_mem[y] = ("keep", 1.0, xs[0][1][2], 1.0, None, None)
                for x, v in xs:
                    keep_mem[x] = ("keep", 1.0, v[2], v[3], None, None)
            else:
                for x, v in xs:
                    drop(x, Reason.NEEDS.value, miss, "review", f"does not work without {', '.join(miss)}; no room "
                         f"(review {v[2]})", REVIEW, v[2])

        # 2. forced picks: weapon rules, binding adds, categories, authored rules, answers
        req = self._required(s, cur, blocked | (set(self._hard(s, "remove")) - set(keep_mem)))
        protected = set(keep_mem) | (guard & cur) | {i for i in req if i in cur} | {i for i in cur if self._lux(i)}
        silent = is_silentheart(s.build, s.ctx.tokens)

        def requires(x, p):
            return p in self.c.items[x].prerequisites or any(p in g for g in self.c.items[x].prerequisites_any)

        def needed(it):
            """a prerequisite of another pick, a slot grant in use, or the only pick
            a picked prerequisite unlocks (removing it would leave that alone)."""
            rest = cur - {it}
            if any(requires(x, it) for x in rest) or self._slot_needed(it, cur, s):
                return True
            return any(p in rest and requires(it, p) and not any(requires(x, p) for x in rest) for p in cur)

        def room(items) -> bool:
            after = cur | set(items)
            return self.cap_excess(after) <= 0 and self.fits(after, s)

        def seq_ok(items) -> bool:
            tmp = set(cur)
            for y in items:
                if not self._feasible(y, tmp, s, cap=False) or self._side_block(y, tmp, s):
                    return False
                tmp.add(y)
            return True

        def make_room(items) -> bool:
            """drop the lowest-scoring unprotected picks until items fit."""
            while not room(items):
                kinds = {self._kind(y) for y in items}
                if self.cap_excess(cur | set(items)) > 0:
                    kinds.add("t")
                c = [i for i in cur if self._kind(i) in kinds and i not in protected and not needed(i)]
                if not c:
                    return False
                v = min(c, key=zz)
                drop(v, Reason.LESS_NECESSARY_SPACE.value, [], "worth", f"makes room for {', '.join(items)}")
            return True

        # 2a. game limits: an input over the talent cap or its mantra slots loses
        # its lowest-scoring picks (protected ones last, prerequisites never)
        def over() -> bool:
            return self.cap_excess(cur) > 0 or not self.fits(cur, s)

        for strict in (True, False):
            while over():
                kinds = ({"t"} if self.cap_excess(cur) > 0 else set()) | (set() if self.fits(cur, s) else {"m"})
                c = [i for i in cur if self._kind(i) in kinds and i not in locked and not needed(i)
                     and (not strict or i not in protected)]
                if "m" in kinds:                       # a mantra whose removal relieves the slots
                    c = [i for i in c if self._kind(i) != "m" or
                         self.c.mantra_slot_usage((cur - {i}) | s.outside, s.oath)["over"]
                         < self.c.mantra_slot_usage(cur | s.outside, s.oath)["over"]]
                if not c:
                    break
                v = min(c, key=zz)
                protected.discard(v)
                drop(v, Reason.LESS_NECESSARY_SPACE.value, [], "rules",
                     self._cap_note(cur) if self._kind(v) == "t" else "over its mantra slots")

        def force(kinds_):
            """forced picks of these kinds (with their prerequisite chains)."""
            for it, (ps, src, why_, rid) in req.items():
                if self._kind(it) not in kinds_:
                    continue
                if it in cur:
                    protected.add(it)
                    continue
                chain = [] if self.c.prereqs_met(it, cur) else self._missing_chain(it, cur, s, blocked)
                if chain is None:
                    continue
                pack = list(chain) + [it]
                if not seq_ok(pack):
                    continue
                snap = (set(cur), list(changes), set(blocked))
                if not make_room(pack):
                    cur.clear(); cur.update(snap[0]); changes[:] = snap[1]; blocked.clear(); blocked.update(snap[2])
                    continue
                for y in pack:
                    reason = self._add_reason(ps, src) if y == it else Reason.SYNERGY_WITH.value
                    add(y, reason, [] if y == it else [it], src if src != "category" else "rules",
                        why_ if y == it else f"prerequisite of {it}", ps, rid)
                    protected.add(y)

        protected |= {i for i in req if i in cur}
        force({"m"} if not silent else {"m", "t"})     # mantras first unless Silentheart

        # 3. removals by score
        th = float(cfg.sc_remove)
        for it in sorted(cur, key=zz):
            if it in protected or it in locked or zz(it) >= th or needed(it):
                continue
            e = s.score["why"].get(it)
            if e is not None and e[1].label == "remove":
                c = e[1]
                reason = c.reason if c.reason in _REMOVE_REASONS else Reason.NO_SYNERGY.value
                drop(it, reason, list(c.related), "review", f"score {zz(it):.2f}; removed by review {e[2]} "
                     f"(similarity {e[3]:.2f})" + self._note(e[2]), REVIEW, e[2])
            else:
                drop(it, Reason.NO_SYNERGY.value, [], "worth", f"score {zz(it):.2f} below {th:.2f}")

        # 4. fill by score: mantras first (unless Silentheart), then talents
        lo = float(cfg.sc_fill) if fill_min is None else float(fill_min)

        def candidates(kind):
            out = []
            for it in self.c.ids:
                if it in cur or it in blocked or self._kind(it) != kind or zz(it) < lo:
                    continue
                t = self.c.index[it]
                if not s.ctx.avail[t] or it in self._ban or self._cold(it, s):
                    continue
                out.append(it)
            return sorted(out, key=lambda i: -zz(i))

        def try_add(it, src="worth") -> bool:
            chain = [] if self.c.prereqs_met(it, cur) else self._missing_chain(it, cur, s, blocked)
            if chain is None:
                return False
            pack = list(chain) + [it]
            if not room(pack) or not seq_ok(pack):
                return False
            if any(self._kind(y) != self._kind(it) for y in pack) and not keep_size_ok(pack):
                return False
            for y in pack:
                e = s.score["why"].get(y)
                rr = e[1].reason if (e is not None and e[1].label == "add" and e[1].reason) else None
                add(y, rr if y == it else Reason.SYNERGY_WITH.value, [] if y == it else [it],
                    "review" if rr else src,
                    f"score {zz(y):.2f}" + (f"; added by review {e[2]}" if rr else "") if y == it
                    else f"prerequisite of {it}", REVIEW if rr else LUXURY, e[2] if rr else None)
            return True

        def keep_size_ok(pack) -> bool:
            if not keep_size:
                return True
            for k, n in ((0, "t"), (1, "m")):
                if sum(self._kind(i) == n for i in cur | set(pack)) > cap[k]:
                    return False
            return True

        kinds = ("t",) if silent else ("m", "t")
        if not keep_size:
            for k in kinds:
                if k == "t" and not silent:
                    force({"t"})
                for it in candidates(k):
                    if it not in cur:
                        try_add(it)
        else:                                          # refill what the answer removed, up to the input's counts
            for k in kinds:
                if k == "t" and not silent:
                    force({"t"})
                for it in candidates(k):
                    if it not in cur and keep_size_ok([it]):
                        try_add(it)

        # 5. swaps: an outside item worth swap_gain more than the weakest removable pick
        gain = float(cfg.sc_swap)
        for _ in range(int(cfg.max_swaps)):
            best = None
            for k in kinds:
                out = [i for i in cur if self._kind(i) == k and i not in protected and i not in locked and not needed(i)]
                if not out:
                    continue
                v = min(out, key=zz)
                cur.discard(v)
                for it in candidates(k):
                    if zz(it) - zz(v) <= gain:
                        break
                    if self.c.prereqs_met(it, cur) and room([it]) and seq_ok([it]):
                        if best is None or zz(it) - zz(v) > best[0]:
                            best = (zz(it) - zz(v), v, it)
                        break
                cur.add(v)
            if best is None:
                break
            _, v, it = best
            drop(v, Reason.LESS_NECESSARY_SPACE.value, [it], "worth", f"{it} scores {best[0]:.2f} higher")
            blocked.discard(it)
            try_add(it)

        # 6. iffy (binding sidegrade marks only), the answer
        iffy_items = [it for it, v in s.memory.items() if v[0] == "iffy" and self._binds(v) and it in cur]
        iffy_why = {it: dict(reason=v[4], related=list(v[7]), review_id=v[2]) for it, v in s.memory.items()
                    if it in iffy_items and v[4] == Reason.SIDEGRADE.value and len(v) > 7}
        passes = {it: (req[it][0] if it in req else self._pass_here(it, s, cur)) for it in cur}
        nt, nm = len(self._talents & cur), sum(self._kind(i) == "m" for i in cur)
        capn = cap if keep_size else (nt + max(0, self.talent_cap(cur) - nt), nm + max(0, self.slot_free(cur, s)))
        alt = self._alt(b, s, cur, changes, iffy_items, passes, capn)
        alt.focus = dict(s.focus)
        alt.iffy_why = iffy_why
        alt.violations = self._violations(s, cur, blocked, set(keep_mem))
        alt.sidegrades = self._offers(s, cur, set(locked) | set(keep_mem))
        return alt
