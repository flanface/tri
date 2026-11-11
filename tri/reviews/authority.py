"""Review authority: reviews are ground truth; conflicts resolve by recency.

Policy
  * Every review is accurate for the build it was written on. Nothing is
    averaged, voted or discounted as noise.
  * A review makes one claim per item it speaks about:
        keep    an item of the build left unmarked (endorsed)
        add     an added item
        remove  a removed item (with its reason)
        iffy    an item marked iffy
    An item that is neither in the build nor added is not a claim (slots are
    full, so not adding something says nothing against it).
  * Two claims conflict when they are about the same item, carry different
    labels, and come from similar builds. The newer claim wins; the older one
    is down-weighted by how similar the two builds are:
        sim >= sim_full                -> older claim weight 0 (superseded)
        sim_min <= sim < sim_full      -> linear between 1 and 0
        sim < sim_min                  -> both stand (different context)
    Similarity is archetype similarity: exp(-d^2 / 2h^2) of the two builds'
    concept-space distance d (the weighted RMS difference of their worth-field logits;
    Tri passes its space in), with h chosen so that sim = sim_min at d = reach
    (defaults: claims transfer in full within d ~ 0.58, fade to nothing at
    d = 1.2; same-archetype builds sit below ~1.05, different archetypes above ~1.6), so a review governs every build whose worth
    profile is nearly the same, not only builds that share items. The same build id counts as sim 1. A BROKEN removal is context-free: it
    conflicts with every other claim on that item, whatever the build (a
    newer "does not work" overrides older endorsements everywhere, and a newer
    endorsement overrides an older "does not work", e.g. after a fix).
  * Patches: a claim about an item that a later patch changed counts
    patch_factor times and is listed in ledger.stale (to re-review).
  * Order is by Review.timestamp; reviews without one are treated as oldest,
    and ties keep insertion order (later wins).

The resulting weights feed rule induction (rules.RuleInducer), the worth
field (reviewed builds become heavy neighbours) and the review memory used at
inference (ReviewLedger.claims_for), so a review changes answers immediately."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.catalog import Catalog
from tri.core.types import RELATIVE_REASONS, Action, Build, Reason, Review
from tri.reviews.defs import Claim, LedgerConfig, ReviewWeights, Supersession

LABELS = ("keep", "add", "remove", "iffy")
POLARITY = {"keep": 1, "add": 1, "remove": -1, "iffy": 0}


def build_similarity(a: Build, b: Build, stat_scale: float = 100.0) -> float:
    """1.0 for the same build id; otherwise Jaccard similarity of the item sets,
    scaled by how close the highest stats are (0.5 .. 1.0)."""
    if a.build_id and b.build_id and a.build_id == b.build_id:
        return 1.0
    A, B = set(a.items()), set(b.items())
    if not A and not B:
        return 0.0
    jac = len(A & B) / len(A | B)
    ks = set(a.highest_stats) | set(b.highest_stats)
    if ks:
        diff = sum(abs(a.highest_stats.get(k, 0) - b.highest_stats.get(k, 0)) for k in ks) / len(ks)
        st = max(0.0, 1.0 - diff / stat_scale)
    else:
        st = 1.0
    return jac * (0.5 + 0.5 * st)


def build_weapons(b: Build) -> set:
    """Weapon classes a build equips (Build.meta "weapon" / "weapons": str or
    list): the type and weight class of each weapon tri knows (tri/weapons.py),
    and any other value as written (title case), e.g. "Heavy"."""
    from tri.game import weapons as W
    m = {str(k).lower(): v for k, v in ((b.meta if b is not None else None) or {}).items()}
    out = set()
    T = W.table()["weapons"]
    for k in ("weapon", "weapons"):
        v = m.get(k)
        for w in ([v] if isinstance(v, str) else list(v or [])):
            known = W.resolve(w)
            for x in known:
                out.add(T[x]["type"].title())
                if T[x].get("weight"):
                    out.add(T[x]["weight"])
            if not known:
                for t in str(w).replace("/", ",").split(","):
                    if t.strip() and t.strip().lower() not in ("none", ""):
                        out.add(t.strip().title())
    return out


_RELATIVE = {r.value for r in RELATIVE_REASONS}


_MADE_BY_TRI = re.compile(r"\(tri (completion|proposal|analysis|review|plan|suggestion)\)\s*$", re.I)


def made_by_tri(b: Optional[Build]) -> bool:
    """A build tri made: meta made_by "tri" (dwt.sh description "tri <kind> of
    <url>", v2.25.1) or the name save_dwt gives it ("<name> (tri completion)")."""
    m = (getattr(b, "meta", None) or {}) if b is not None else {}
    if str(m.get("made_by") or "").lower() == "tri":
        return True
    return bool(_MADE_BY_TRI.search(str(m.get("name") or "")))


class ReviewLedger:
    """Claims of every review, with recency-based conflict resolution.

        led = ReviewLedger(reviews, catalog)
        led.weights["pool-12"].ops        # per-operation weights for training
        led.superseded                    # what was overridden, by what
        led.claims_for(build)             # active claims from similar reviewed builds
    """

    def __init__(self, reviews: Sequence[Review], catalog: Optional[Catalog] = None,
                 cfg: Optional[LedgerConfig] = None, embed: Optional[Callable[[Build], np.ndarray]] = None,
                 kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 change_times: Optional[Dict[str, float]] = None, kernel_many=None, family_many=None):
        """embed: build -> concept-space coordinates; kernel: (u, v) -> similarity
        (default: the concept-distance kernel, see reach_kernel); embed None falls back to build_similarity. change_times: item -> latest patch time."""
        self.cfg = cfg or LedgerConfig()
        self.embed = embed
        self.kernel = kernel
        self.kernel_many = kernel_many                 # (u, V) -> [m]; v3.1
        self.family_many = family_many                 # (u, V) -> [m] family overlap; v3.3
        self._vec: Dict[str, np.ndarray] = {}
        self.change_times = dict(change_times or {})
        self.stale: List[Tuple[str, str]] = []        # (review id, item) discounted by a patch
        self.catalog = catalog
        idx = list(range(len(reviews)))
        idx.sort(key=lambda i: (reviews[i].timestamp is not None, reviews[i].timestamp or 0.0, i))
        self.reviews: List[Review] = [reviews[i] for i in idx]          # oldest first
        self.order = {r.review_id: k for k, r in enumerate(self.reviews)}
        self.by_id = {r.review_id: r for r in self.reviews}
        self.claims: Dict[str, List[Claim]] = {}
        self._sim: Dict[Tuple[str, str], float] = {}
        self.superseded: List[Supersession] = []
        for k, r in enumerate(self.reviews):
            self.claims[r.review_id] = self._claims(r, k)
        self._resolve()
        self._patches()
        for cl in self.claims.values():                    # v2.25.1: echoed endorsements count less
            for c in cl:
                if c.origin == "echo":
                    c.weight *= self.cfg.echo_keep
        self.weights: Dict[str, ReviewWeights] = {r.review_id: self._weights(r) for r in self.reviews}

    # ------------------------------------------------------------ claims
    def _claims(self, r: Review, k: int) -> List[Claim]:
        out = []
        neutral = set(getattr(r, "neutral", None) or ())
        marked = set(neutral)                              # v3.1.1: a purged item says nothing
        for j, o in enumerate(r.operations):
            if o.item_id in neutral:
                continue
            st = o.strength.value if o.strength else None
            label = {Action.ADD: "add", Action.REMOVE: "remove", Action.IFFY: "iffy"}[o.action]
            out.append(Claim(r.review_id, k, o.item_id, label, o.reason.value if o.reason else None, j,
                             strength=st, related=tuple(o.related or ())))
            marked.add(o.item_id)
        echo = made_by_tri(r.original)                     # tri's own picks, echoed back (v2.25.1)
        for it in r.original.items():
            if it not in marked:
                out.append(Claim(r.review_id, k, it, "keep", origin="echo" if echo else "review"))
        return out

    def vector(self, review_id: str) -> np.ndarray:
        if review_id not in self._vec:
            self._vec[review_id] = self.embed(self.by_id[review_id].original)
        return self._vec[review_id]

    def build_sim(self, a: Build, b: Build, va=None, vb=None) -> float:
        """Archetype similarity of two builds in [0, 1]."""
        if a.build_id and b.build_id and a.build_id == b.build_id:
            return 1.0
        if self.embed is None:
            return build_similarity(a, b, self.cfg.stat_scale)
        va = self.embed(a) if va is None else va
        vb = self.embed(b) if vb is None else vb
        if self.kernel is None:
            self.kernel = ReviewLedger.reach_kernel(self.cfg)
        return float(self.kernel(va, vb))

    # v3.1: one kernel call per query build against every reviewed build
    def _matrix(self) -> Optional[np.ndarray]:
        """[R, D] coordinates of the reviewed builds (self.reviews order), or None."""
        if self.embed is None or not self.reviews:
            return None
        V = self.__dict__.get("_Vmat")
        if V is None or len(V) != len(self.reviews):
            V = self._Vmat = np.stack([self.vector(r.review_id) for r in self.reviews])
        return V

    def sims_to(self, build: Build, vec=None) -> np.ndarray:
        """[R] build_sim(build, r) for every reviewed build r (self.reviews order).
        Memoised for the last few queries: claims
        ask about the same build."""
        memo = self.__dict__.setdefault("_qmemo", {})
        v = vec if vec is not None else (self.embed(build) if self.embed is not None else None)
        key = (id(build), None if v is None else np.asarray(v, float).tobytes())
        hit = memo.get(key)
        if hit is not None and hit[0] is build:
            return hit[1]
        V = self._matrix()
        if V is None or v is None:
            S = np.array([self.build_sim(build, r.original) for r in self.reviews])
        else:
            many = getattr(self, "kernel_many", None)
            if many is None:
                h2 = self.cfg.reach ** 2 / (-2.0 * np.log(self.cfg.sim_min))
                base = np.exp(-((V - np.asarray(v, float)[None]) ** 2).sum(1) / (2 * h2))
                S = base if self.kernel is None or getattr(self.kernel, "plain", False) else \
                    np.array([float(self.kernel(v, V[i])) for i in range(len(V))])
            else:
                S = np.asarray(many(np.asarray(v, float), V), float)
        bid = build.build_id
        if bid:
            S = np.where([r.original.build_id == bid for r in self.reviews], 1.0, S)
        if len(memo) > 8:
            memo.clear()
        memo[key] = (build, S)
        return S

    def similarity(self, a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        if key not in self._sim and self.embed is not None and getattr(self, "kernel_many", None) is not None:
            ra = self.by_id[a].original
            S = self.sims_to(ra, self.vector(a))
            for r, x in zip(self.reviews, S):          # the whole row at once
                k2 = (a, r.review_id) if a < r.review_id else (r.review_id, a)
                self._sim.setdefault(k2, float(x) if not (ra.build_id and ra.build_id == r.original.build_id) else 1.0)
        if key not in self._sim:
            ra, rb = self.by_id[a].original, self.by_id[b].original
            if self.embed is None:
                self._sim[key] = self.build_sim(ra, rb)
            else:
                self._sim[key] = self.build_sim(ra, rb, self.vector(a), self.vector(b))
        return self._sim[key]

    def _patches(self) -> None:
        if not self.change_times:
            return
        for r in self.reviews:
            t = r.timestamp if r.timestamp is not None else float("-inf")
            for c in self.claims[r.review_id]:
                ct = self.change_times.get(c.item)
                if ct is not None and t < ct and c.weight > 0:
                    c.weight *= self.cfg.patch_factor
                    self.stale.append((r.review_id, c.item))

    def _factor(self, sim: float) -> float:
        lo, hi = self.cfg.sim_min, self.cfg.sim_full
        if sim >= hi:
            return 0.0
        if sim < lo:
            return 1.0
        return (hi - sim) / max(hi - lo, 1e-9)

    def _resolve(self):
        by_item: Dict[str, List[Claim]] = defaultdict(list)
        for cl in self.claims.values():
            for c in cl:
                by_item[c.item].append(c)
        for item, cs in by_item.items():
            if len({c.label for c in cs}) < 2:
                continue                                  # everyone agrees on this item
            cs.sort(key=lambda c: c.order)
            for i, old in enumerate(cs):
                f_min = 1.0
                hit = None
                for new in cs[i + 1:]:
                    if new.order == old.order or new.label == old.label:
                        continue
                    if new.origin == "echo" and old.origin != "echo":
                        continue                          # tri's own pick echoed back supersedes no reviewer
                    if not self._in_conflict(old, new):
                        continue
                    # a "does not work" claim overrides (or is overridden) everywhere; a
                    # "generally good" one competes with local claims by build similarity,
                    # so a removal on a different archetype stands
                    sim = 1.0 if (new.broken or old.broken) else self.similarity(old.review_id, new.review_id)
                    f = self._factor(sim)
                    if f < f_min:
                        f_min, hit = f, (new, sim)
                    if f_min == 0.0:
                        break
                if hit is not None:
                    old.weight *= f_min
                    self.superseded.append(Supersession(item, old.review_id, old.label, hit[0].review_id,
                                                        hit[0].label, round(hit[1], 4), round(f_min, 4)))

    def _in_conflict(self, a: Claim, b: Claim) -> bool:
        """Two claims on one item with different labels conflict unless one of
        them is conditional on items the other build does not have: "remove x,
        works against y" and "keep x" on a build without y both stand."""
        ia = self._context(self.by_id[a.review_id].original)
        ib = self._context(self.by_id[b.review_id].original)
        return a.holds_in(ib) and b.holds_in(ia)

    @staticmethod
    def _context(b: Build) -> set:
        """A build's picks and its weapons ("weapon:<name>", v3.3): what claim conditions read."""
        from tri.game.weapons import build_weapon_tokens
        return set(b.items()) | build_weapon_tokens(b)

    def _weights(self, r: Review) -> ReviewWeights:
        cl = self.claims[r.review_id]
        ops = [1.0] * len(r.operations)
        keep = {i: 0.0 for i in getattr(r, "neutral", None) or ()}           # v3.1.1: neutral items are not endorsed
        for j, o in enumerate(r.operations):
            if o.item_id in keep:
                ops[j] = 0.0
        for c in cl:
            if c.op_index is not None:
                ops[c.op_index] = c.weight
            else:
                keep[c.item] = c.weight
        ws = [c.weight for c in cl]
        return ReviewWeights(r.review_id, ops, keep, sum(ws) / len(ws) if ws else 1.0)

    # ------------------------------------------------------------ queries
    @staticmethod
    def reach_kernel(cfg: "LedgerConfig"):
        """(u, v) -> exp(-|u-v|^2 / 2h^2) with sim(reach) = sim_min."""
        h2 = cfg.reach ** 2 / (-2.0 * np.log(cfg.sim_min))
        k = lambda a, b: float(np.exp(-float(((np.asarray(a) - np.asarray(b)) ** 2).sum()) / (2 * h2)))
        k.plain = True                                  # sims_to may vectorise it
        return k

    def active_claims(self, review_id: str) -> List[Claim]:
        return [c for c in self.claims.get(review_id, []) if c.weight > 0]

    def neighbours(self, build: Build, min_sim: Optional[float] = None, vec=None) -> List[Tuple[Review, float]]:
        """Reviewed builds similar to `build` (sim >= min_sim, default cfg.sim_min), newest first.
        vec: the build's concept coordinates when they are not its own placement
        (a completion forced into an archetype, v2.25.1)."""
        lo = self.cfg.sim_min if min_sim is None else min_sim
        S = self.sims_to(build, vec)
        return [(self.reviews[i], float(S[i])) for i in range(len(self.reviews) - 1, -1, -1) if S[i] >= lo]

    def claims_for(self, build: Build, min_sim: Optional[float] = None, vec=None) -> Dict[str, tuple]:
        """{item: (label, weight, review_id, similarity, reason, strength)}: the
        newest active claim per item among reviews of similar builds.
        weight = claim weight x similarity factor (1.0 for the same build).
        BROKEN claims apply to any build. Whether a claim is binding or only
        nudges worth is decided by binding()."""
        out: Dict[str, tuple] = {}
        echo: Dict[str, tuple] = {}
        items = self._context(build)
        mine = build_weapons(build)
        for r, sim in self.neighbours(build, min_sim, vec):
            theirs = build_weapons(getattr(r, "original", None))
            if mine and theirs and sim < self.cfg.sim_same and not (mine & theirs):
                continue                                   # the reviewed build wields another weapon class
            f = 1.0 if sim >= self.cfg.sim_full else sim
            for c in self.active_claims(r.review_id):
                if c.item not in out and c.holds_in(items):
                    v = (c.label, c.weight * f, r.review_id, sim, c.reason, c.strength, c.origin, tuple(c.related))
                    if c.origin == "echo":                 # tri's own pick: only where nobody said more
                        echo.setdefault(c.item, v)
                    else:
                        out[c.item] = v
        for it, v in self.family_for(build, vec, items).items():     # v3.3: pooled claims fill gaps and
            cur = out.get(it)                                         # override a lone claim that only nudges
            if cur is None or (self.cfg.pool_override and self.binding(v) and not self.binding(cur)):
                out[it] = v
        for it, v in echo.items():
            out.setdefault(it, v)
        for r in reversed(self.reviews):                   # context-free claims
            for c in self.active_claims(r.review_id):
                if c.global_scope and (c.item not in out or out[c.item][6] == "family"):
                    # local claims (above) take precedence; a generally-good add from
                    # another archetype is marked origin "general" (binding(): strong only)
                    if not c.holds_in(items):
                        continue
                    out[c.item] = (c.label, c.weight, r.review_id, 1.0, c.reason, c.strength,
                                   "general" if c.general else c.origin, tuple(c.related))
        return out

    def _returned(self, r: Review) -> frozenset:
        """Items of the build a full review returns (original - removals + additions)."""
        memo = self.__dict__.setdefault("_ret_memo", {})
        if r.review_id not in memo:
            rem = {o.item_id for o in r.removed()}
            memo[r.review_id] = frozenset((set(r.original.items()) - rem) | {o.item_id for o in r.added()})
        return memo[r.review_id]

    def _candidates(self) -> frozenset:
        """Every item some reviewed build holds or a review adds (absence is only
        informative for items reviewers know to be in play)."""
        if self.__dict__.get("_cand_n") != len(self.reviews):
            self._cand = frozenset().union(*[self._returned(r) | set(r.original.items()) for r in self.reviews]) \
                if self.reviews else frozenset()
            self._cand_n = len(self.reviews)
        return self._cand

    def family_for(self, build: Build, vec=None, items=None) -> Dict[str, tuple]:
        """v3.3 family reach (math §5.8): {item: claim} pooled over the reviews of
        builds located in the same family as `build` (family_many overlap >=
        fam_overlap), newest claim per reviewed build. Echoed picks, iffy marks,
        rule verdicts and the build's own reviews are left out. A side needs
        fam_min distinct builds and a weighted share >= fam_share; weight
        n / (n + fam_kappa) x share; strong (binding) with >= fam_bind strong
        item-level claims behind it and no strong claim against."""
        cfg = self.cfg
        if not cfg.fam_reach or self.embed is None or not self.reviews:
            return {}
        V = self._matrix()
        v = np.asarray(vec if vec is not None else self.embed(build), float)
        O = np.asarray(self.family_many(v, V), float) if self.family_many is not None else np.zeros(len(V))
        S = self.sims_to(build, vec)
        if cfg.pool_sim < 1.0:                        # neighbours pool too (never below sim_min)
            near = S >= max(cfg.pool_sim, cfg.sim_min)
            O = np.where(near, np.maximum(O, S), O)
        if not (O >= cfg.fam_overlap).any():
            return {}
        items = self._context(build) if items is None else items
        have = set(build.items())
        mine = build_weapons(build)
        votes: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        absent: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        for i in range(len(self.reviews) - 1, -1, -1):  # newest first
            if O[i] < cfg.fam_overlap:
                continue
            r = self.reviews[i]
            bk = r.original.build_id or r.review_id
            if build.build_id and bk == build.build_id:
                continue                               # the build's own reviews act locally
            theirs = build_weapons(r.original)
            if mine and theirs and not (mine & theirs):
                continue
            claimed = set()
            for c in self.active_claims(r.review_id):
                claimed.add(c.item)
                if c.origin == "echo" or c.label not in ("keep", "add", "remove"):
                    continue
                if bk in votes[c.item] or not c.holds_in(items):
                    continue
                w = c.weight * float(O[i]) * (cfg.fam_keep if c.label == "keep" else 1.0)
                strong = c.label in ("add", "remove") and c.strength == "strong" and c.reason not in _RELATIVE
                votes[c.item][bk] = (POLARITY[c.label], w, strong, c, r.review_id, float(S[i]))
            if cfg.pool_absent:      # left out of the returned build: absence endorsed
                for it in self._candidates() - self._returned(r) - have - claimed:
                    if bk not in absent[it]:
                        absent[it][bk] = (float(O[i]) * cfg.fam_keep, r.review_id, float(S[i]))
        out: Dict[str, tuple] = {}
        for it, per in absent.items():
            if it in votes and any(x[0] > 0 for x in votes[it].values()):
                continue                                  # someone added it: no absence consensus
            n = len(per)
            if n >= cfg.fam_min:                          # weak below fam_consensus: it guards worth swaps only
                top = max(per.values(), key=lambda x: (self.order.get(x[1], -1)))
                w = n / (n + cfg.fam_kappa)
                out[it] = ("absent", w, top[1], max(x[2] for x in per.values()), None,
                           "strong" if n >= cfg.fam_consensus else "weak", "family", (), n)
        for it, per in votes.items():
            vs = list(per.values())
            if it not in have:                      # an add needs explicit adds: a neighbour
                vs = [x for x in vs if x[3].label != "keep"]   # keeping what it had is no add
            pos = [x for x in vs if x[0] > 0]
            neg = [x for x in vs if x[0] < 0]
            sp, sn = sum(x[1] for x in pos), sum(x[1] for x in neg)
            if sp + sn <= 0:
                continue
            if len(neg) >= cfg.fam_min and sn / (sp + sn) >= cfg.fam_share:
                lab, sup, opp, share = "remove", neg, pos, sn / (sp + sn)
            elif len(pos) >= cfg.fam_min and sp / (sp + sn) >= cfg.fam_share and \
                    (it in have or pos):
                lab, sup, opp, share = ("keep" if it in have else "add"), pos, neg, sp / (sp + sn)
            else:
                continue
            n = len(sup)
            agreed = n >= cfg.fam_consensus and share >= cfg.fam_consensus_share
            strong = not any(x[2] for x in opp) and (
                agreed or (lab != "keep" and sum(x[2] for x in sup) >= cfg.fam_bind))
            ex = [x for x in sup if x[3].label != "keep"] or sup
            top = max(ex, key=lambda x: (x[2], self.order.get(x[4], -1)))
            c = top[3]
            out[it] = (lab, n / (n + cfg.fam_kappa) * share, top[4], max(x[5] for x in sup),
                       c.reason if c.label != "keep" else None, "strong" if strong else "weak", "family",
                       tuple(c.related), n)
        return out

    def conditions(self) -> Dict[str, Dict[str, Tuple[str, ...]]]:
        """{review_id: {item: related items}} for conditional claims (redundant
        with y, works against y): worth counts such a change only for query
        builds that have y."""
        out: Dict[str, Dict[str, Tuple[str, ...]]] = {}
        for rid, cl in self.claims.items():
            d = {c.item: c.condition_tokens() for c in cl if c.conditional and c.op_index is not None}
            if d:
                out[rid] = d
        return out

    def binding(self, claim: tuple, hard: float = 0.5) -> bool:
        """A claim binds (overrides rules and worth) when its weight >= hard and
        - it is about this very build (similarity >= sim_same), or
        - it is a BROKEN removal (context-free), or
        - it is a strong "generally good" add (origin "general": made on a
          build of another archetype; it binds only where the talent is
          feasible and no exclusion rule forbids it), or
        - it is a strong "generally good" add from a similar build, or
        - it is a strong add/remove whose reason is about the item itself,
          not about slot space, made on a build at similarity >= bind_sim, or
        - it is a sidegrade mark (weak: within ledger reach, i.e. under this
          archetype; strong: global).
        Weak claims, keeps and relative judgements ("less necessary, need
        space", "worth less / more as a luxury") made on a merely similar build are relative to that build, so they only nudge worth."""
        label, w, _rid, sim, reason, strength, origin = (tuple(claim) + (None, None, None))[:7]
        if w < hard:
            return False
        if origin == "family":                             # v3.3 pooled family claim (math §5.8)
            return strength == "strong" and label in ("add", "remove", "keep", "absent")
        if origin == "general":                            # generally good, from another build:
            return strength == "strong"                    # strong binds (where feasible), weak nudges
        if sim >= self.cfg.sim_same:
            return True
        if label == "iffy" and reason == Reason.SIDEGRADE.value:
            return True                                    # weak: within reach (this archetype); strong: anywhere
        if label == "remove" and reason == Reason.BROKEN.value:
            return True
        if label == "add" and reason == Reason.GENERALLY_GOOD.value and strength == "strong":
            return True                                    # as from another archetype (origin "general")
        return label in ("add", "remove") and strength == "strong" and reason not in _RELATIVE \
            and sim >= self.cfg.bind_sim

    def report(self) -> dict:
        active = sum(w.active for w in self.weights.values())
        return dict(reviews=len(self.reviews), active=active, superseded_claims=len(self.superseded),
                    pre_patch_claims=len(self.stale), stale=[list(x) for x in self.stale[:200]],
                    fully_superseded_reviews=[w.review_id for w in self.weights.values() if not w.active],
                    overrides=[s.__dict__ for s in self.superseded[:200]])
