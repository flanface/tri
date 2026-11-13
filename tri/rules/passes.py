"""The rulebase: certified conjunctive rules, mined from builds.

A rule is a conjunction of conditions and an effect on one item:

    take  ITEM when C1 and C2 and ...     (foundational or central)
    never ITEM when C1 and C2 and ...     (exclusion)

Conditions look at what a build is, never at a fitted archetype label:
    stat     <stat> >= v  /  <stat> < v        (highest stats; v a multiple of `stat_step`, 5)
    context  oath / murmur is (not) v
    item     with ITEM2                        (anchors: picks of the build)

Mining (math §6.3), per item t, on the builds that can take t,
shuffled and split into halves A (search) and B (certification):
  1. Candidate conjunctions are the nodes of two shallow decision trees grown
     on A (entropy splits): one over stats + context + the `co_items` items
     that share most information with t, one over stats + context only.
  2. Each candidate is checked on A with a one-sided Wilson bound at
     confidence z on the weighted rate, with effective sample size
     n_eff = (sum w)^2 / sum w^2:
        take   lower bound >= theta_take  and support >= min_support
        never  upper bound <= theta_never, support >= min_support, and the
               rule has a contrast: flipping one of its conditions gives a
               rate >= theta_contrast (an exclusion explains a refusal, it is
               not merely "rarely taken")
     and for uniformity: no pick-profile cluster inside the rule's cell (with
     weight >= uniform_support) clearly contradicts it (a small archetype
     hidden inside a population-level cell).
  3. Conditions are dropped greedily while the rule stays certified (shortest
     certified form). The final form is certified again on B, which played
     no part in choosing it; only rules passing on B are kept. Duplicates and
     rules implied by a more general rule of the same item are removed.
  4. Conflicts: where a take rule and a never rule of one item overlap with
     enough weight, the observed rate there drops the one it contradicts.
  5. Pass: a take rule with stat conditions only (or none) whose covered builds
     are spread over pick-profile clusters like the corpus (breadth >=
     min_breadth) is foundational; every other take rule is central. Items
     with no take rule are luxury.

Guarantee: each rule states a rate bound that holds with confidence ~ Phi(z)
on the distribution its builds came from (per rule; see the docs for the
multiple-testing caveat). certification_check measures it on held-out builds."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from tri.core.stats import wilson
from tri.model.matrix import BuildMatrix, group_dot, mat_groups
from tri.rules.defs import Condition, Ctx, PassConfig

FOUNDATIONAL, CENTRAL, LUXURY, EXCLUSION = "foundational", "central", "luxury", "exclusion"
TAKE, NEVER = "take", "never"


def _dense_bool(X) -> np.ndarray:
    out = np.zeros(X.shape, bool)
    for a in range(0, X.shape[0], 32768):
        out[a:a + 32768] = X[a:a + 32768].toarray() > 0
    return out


def _n_jobs(f: "PassConfig", n_items: int, n_rows: int) -> int:
    """Processes for Rulebook.mine: n_jobs if set, else all cores (up to 8) once
    the corpus is big enough for start-up costs (a few seconds) to pay off.
    TRI_JOBS (environment) overrides."""
    import os
    env = os.environ.get("TRI_JOBS")
    j = int(env) if env not in (None, "") else int(getattr(f, "n_jobs", 0) or 0)
    if j == 0:
        if n_items < 64 or n_rows < 5000:
            return 1
        j = min(8, os.cpu_count() or 1)
    elif j < 0:
        j = os.cpu_count() or 1
    try:
        import joblib  # noqa: F401  (ships with scikit-learn)
    except Exception:
        return 1
    return max(1, min(j, n_items))


def _mine_chunk(book: "Rulebook", ctx: dict, items: List[str]) -> List[tuple]:
    """Worker: [(item, rules, veto info)] for a chunk of items."""
    out = []
    for i in items:
        info: dict = {}
        out.append((i, book._mine_one(i, ctx, info), info))
    return out


# ---------------------------------------------------------------- aggregates
ATTUNEMENTS = ("Flamecharm", "Frostdraw", "Thundercall", "Galebreathe", "Shadowcast", "Ironsing", "Bloodrend")
WEAPONS = ("HeavyWeapon", "MediumWeapon", "LightWeapon")
CORE = ("Strength", "Fortitude", "Agility", "Intelligence", "Willpower", "Charisma")
#: (name, how, member stats (None = all), thresholds on the stat grid)
AGGREGATES: Tuple[Tuple[str, str, Optional[Tuple[str, ...]], bool], ...] = (
    ("attunement points", "sum", ATTUNEMENTS, True),
    ("weapon points", "sum", WEAPONS, True),
    ("core points", "sum", CORE, True),
    ("top attunement", "max", ATTUNEMENTS, True),
    ("stats at 75+", "count75", None, False),
)


def aggregate_defs(stats: Sequence[str]) -> List[Tuple[str, str, List[int], bool]]:
    """Aggregates defined for a stat list: (name, how, member indices, on_grid).
    A sum / max needs at least two member stats in the list."""
    ix = {s: i for i, s in enumerate(stats)}
    out = []
    for name, how, mem, grid in AGGREGATES:
        cols = list(range(len(stats))) if mem is None else [ix[s] for s in mem if s in ix]
        if len(cols) >= 2:
            out.append((name, how, cols, grid))
    return out


def _agg(how: str, V: np.ndarray) -> np.ndarray:
    if how == "sum":
        return V.sum(1)
    if how == "max":
        return V.max(1)
    return (V >= 75).sum(1)


def aggregate_matrix(H: np.ndarray, stats: Sequence[str]) -> Tuple[List[str], List[bool], np.ndarray]:
    """(names, on_grid, [B, A] float32) aggregate values of highest stats H."""
    d = aggregate_defs(stats)
    H = np.asarray(H, np.float32)
    M = np.stack([_agg(how, H[:, cols]) for _, how, cols, _ in d], 1).astype(np.float32) if d \
        else np.zeros((len(H), 0), np.float32)
    return [x[0] for x in d], [x[3] for x in d], M


def aggregate_value(name: str, hi: Dict[str, int]) -> float:
    """One aggregate of a build's highest stats (dict)."""
    for n, how, mem, _ in AGGREGATES:
        if n == name:
            vals = [hi.get(s, 0) for s in mem] if mem is not None else list(hi.values())
            return float(_agg(how, np.array([vals], np.float32))[0]) if vals else 0.0
    raise KeyError(name)


def holds(c: Condition, x: Ctx, item: Optional[str] = None) -> bool:
    """c on the build x. A group literal counts the build's picks in the group
    other than `item` (the rule's own item)."""
    if c.kind in ("stat", "agg"):
        v = x.hi.get(c.key, 0) if c.kind == "stat" else aggregate_value(c.key, x.hi)
        return v >= c.value if c.op == ">=" else v < c.value
    if c.kind == "group":
        mem = (x.groups or {}).get(c.key, frozenset())
        v = len((set(x.items) - {item}) & mem)
        return v >= c.value if c.op == ">=" else v < c.value
    if c.kind == "context":
        return (c.key in x.tokens) == (c.op == "has")
    return (c.key in x.items) == (c.op == "has")


@dataclass
class PassRule:
    pass_: str                     # foundational | central | exclusion
    item: str
    conds: List[Condition]
    confidence: float              # weighted rate among covered builds
    bound: float                   # certified lower (take) / upper (never) bound
    support: float                 # weighted covered builds
    n_eff: float = 0.0
    lift: float = 1.0              # confidence / the item's overall rate
    contrast: Optional[float] = None   # never: rate with the pivotal condition flipped
    effect: str = TAKE
    status: str = "active"         # active | stale | trivial (v3.0.5, culled by a group baseline)
    source: str = "data"
    note: str = ""
    breadth: Optional[float] = None

    @property
    def key(self) -> str:
        return f"{self.effect}|{self.item}|" + "&".join(sorted(c.id() for c in self.conds))

    @property
    def cond(self) -> Optional[Condition]:
        """The first condition (None for "whenever available")."""
        return self.conds[0] if self.conds else None

    def applies(self, x: Ctx) -> bool:
        return all(holds(c, x, self.item) for c in self.conds)

    def when(self) -> str:
        return " and ".join(c.text() for c in self.conds) if self.conds else "whenever available"

    def text(self, region_names: Sequence[str] = ()) -> str:
        """The rule in symbols (tri.symbolic): [F] Strength ≥ 40 ⇒ item   P 0.92 ≥ 0.85 · n 340."""
        from tri.rules.symbolic import pass_rule
        return pass_rule(self, region_names)

    def plain(self, region_names: Sequence[str] = ()) -> str:
        """The rule in words (the pre-2.14 wording)."""
        st = f" [{self.status}]" if self.status != "active" else ""
        if self.effect == NEVER:
            return (f"{EXCLUSION}: never {self.item} when {self.when()} "
                    f"({self.confidence:.0%} of {self.support:.0f} builds, certified <= {self.bound:.1%}"
                    + (f"; {self.contrast:.0%} otherwise" if self.contrast is not None else "") + ")" + st)
        return (f"{self.pass_}: take {self.item} {'when ' if self.conds else ''}{self.when()} "
                f"({self.confidence:.0%} of {self.support:.0f} builds, certified >= {self.bound:.0%})" + st)

    def to_dict(self) -> dict:
        d = asdict(self); d["conds"] = [asdict(c) for c in self.conds]; return d

    @staticmethod
    def from_dict(d: dict) -> "PassRule":
        d = dict(d); d["conds"] = [Condition(**c) for c in d["conds"]]; return PassRule(**d)
# ---------------------------------------------------------------- statistics


def _implied(general: Sequence[Condition], specific: Sequence[Condition]) -> bool:
    """True if every build meeting `specific` also meets `general` (each general
    condition is matched by an equal or stricter specific one)."""
    def covers(g: Condition, c: Condition) -> bool:
        if (g.kind, g.key, g.op) != (c.kind, c.key, c.op):
            return False
        if g.op == ">=":
            return c.value >= g.value
        if g.op == "<":
            return c.value <= g.value
        return c.value == g.value
    return all(any(covers(g, c) for c in specific) for g in general)


def _picked_available(bm: BuildMatrix) -> sp.csr_matrix:
    X = bm.X.tocoo()
    keep = bm.masks[bm.gid[X.row], X.col]
    return sp.csr_matrix((X.data[keep], (X.row[keep], X.col[keep])), shape=bm.X.shape)


def _group_matrix(bm: BuildMatrix) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(bm)), (np.arange(len(bm)), bm.gid)), shape=(len(bm), len(bm.masks)))


def anchor_counts(bm: BuildMatrix, w: np.ndarray, XA=None, G=None):
    """(num [N,N], den [N,N]): num[a,t] = sum w x_a x_t, den[a,t] = sum w x_a a_t."""
    XA = _picked_available(bm) if XA is None else XA
    G = _group_matrix(bm) if G is None else G
    Xw = sp.diags(w) @ bm.X
    num = np.asarray((Xw.T @ XA).todense())
    H = (G.T @ Xw).T.tocsc()                    # [N,G] sparse: a dense [G,N] is 1.5 GB at 265k groups
    den = mat_groups(H, bm.masks)
    return num, den


def mutual_information(bm: BuildMatrix, w: np.ndarray, signed: bool = False) -> np.ndarray:
    """[N,N] MI[a,t] between picking a and picking t, over builds that can take t.
    signed: negative where a lowers the rate of t (an antagonist)."""
    XA, G = _picked_available(bm), _group_matrix(bm)
    n11, n1_ = anchor_counts(bm, w, XA, G)
    wg = np.asarray(G.T @ w).ravel()
    n_t = group_dot(wg, bm.masks)                 # builds that can take t
    pos = np.asarray(XA.T @ w).ravel()                        # ... and take it
    n = np.maximum(n_t, 1e-12)[None, :]
    p11 = n11 / n; p10 = (n1_ - n11) / n; p01 = (pos[None, :] - n11) / n
    p00 = np.clip(1 - p11 - p10 - p01, 0, 1)
    pa = p11 + p10; pt = p11 + p01

    def term(pj, px, py):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = pj * np.log(pj / (px * py))
        return np.nan_to_num(np.where(pj > 0, v, 0.0))
    mi = term(p11, pa, pt) + term(p10, pa, 1 - pt) + term(p01, 1 - pa, pt) + term(p00, 1 - pa, 1 - pt)
    np.fill_diagonal(mi, 0)
    if signed:
        mi = np.where(p11 < pa * pt, -mi, mi)
    return mi
# ---------------------------------------------------------------- rulebook


class Rulebook:
    """Certified rules. .firing(ctx) / .forbidding(ctx) give those that apply to a build."""

    def __init__(self, catalog, cfg: Optional[PassConfig] = None):
        self.c = catalog
        self.cfg = cfg or PassConfig()
        self.rules: List[PassRule] = []
        self.pick_rate: Optional[np.ndarray] = None
        self.region_names: List[str] = []
        self.vetoed: Dict[str, dict] = {}      # item -> why it has no take rule (the vetoing cluster)
        self._pick_ZC = None                    # (Z, C) of the last pick_regions call (leave-item-out)
        self.groups: Dict[str, frozenset] = {}  # known groups (tri.rules.groups), v3.0
        self.reader = None                      # bm -> archetype key per row (Tri sets it), v3.0
        self._ctx_cache: Dict[int, tuple] = {}

    def context_matrix(self, bm: BuildMatrix) -> Tuple[List[str], np.ndarray]:
        """(names, [n, K]) rule context tokens of bm's rows: planner fields, weapon
        class, archetype (when a reader is set and cfg.archetypes)."""
        from tri.rules.groups import corpus_tokens
        key = id(bm)
        hit = self._ctx_cache.get(key)
        if hit is not None and hit[0] is bm:
            return hit[1], hit[2]
        arch = self.reader(bm) if (self.reader is not None and self.cfg.archetypes) else None
        names, M = corpus_tokens(bm, self.c, arch)
        if len(self._ctx_cache) > 4:
            self._ctx_cache.clear()
        self._ctx_cache[key] = (bm, names, M)
        return names, M

    # ------------------------------------------------------------ masks
    def _mask(self, bm: BuildMatrix, c: Condition, Xd=None, item: Optional[str] = None) -> np.ndarray:
        if c.kind == "group":
            idx = [self.c.index[i] for i in self.groups.get(c.key, ()) if i in self.c.index and i != item]
            v = np.asarray(bm.X[:, idx].sum(1)).ravel() if idx else np.zeros(len(bm))
            return v >= c.value if c.op == ">=" else v < c.value
        if c.kind == "context" and c.key not in bm.tokens:
            names, M = self.context_matrix(bm)
            m = M[:, names.index(c.key)] > 0 if c.key in names else np.zeros(len(bm), bool)
            return m if c.op == "has" else ~m
        if c.kind in ("stat", "agg"):
            if c.kind == "stat":
                v = bm.hi[:, self.c.stat_index[c.key]]
            else:
                names, _, M = aggregate_matrix(bm.hi, self.c.stats)
                v = M[:, names.index(c.key)]
            return v >= c.value if c.op == ">=" else v < c.value
        if c.kind == "context":
            m = (np.asarray(bm.T[:, bm.tokens.index(c.key)].todense()).ravel() > 0) if c.key in bm.tokens \
                else np.zeros(len(bm), bool)
            return m if c.op == "has" else ~m
        j = self.c.index[c.key]
        m = (Xd[:, j] > 0) if Xd is not None else (np.asarray(bm.X[:, j].todense()).ravel() > 0)
        return m if c.op == "has" else ~m

    def rule_mask(self, bm: BuildMatrix, r: PassRule) -> np.ndarray:
        """Builds the rule covers: can take the item and meet every condition."""
        m = bm.masks[bm.gid, self.c.index[r.item]].copy()
        for c in r.conds:
            m &= self._mask(bm, c, item=r.item)
        return m

    def rule_stats(self, bm: BuildMatrix, r: PassRule, w: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """(rate, support, n_eff) of rule r on bm (weights w, default bm.w)."""
        w = bm.w if w is None else w
        m = self.rule_mask(bm, r)
        ww = w[m]
        S = float(ww.sum())
        if S <= 0:
            return 0.0, 0.0, 0.0
        y = np.asarray(bm.X[:, self.c.index[r.item]].todense()).ravel()[m] > 0
        return float(ww[y].sum() / S), S, float(S * S / max((ww * ww).sum(), 1e-12))

    # ------------------------------------------------------------ mining
    def mine(self, bm: BuildMatrix, regions: Optional[np.ndarray] = None, region_names: Sequence[str] = (),
             changed_since: Optional[Dict[str, float]] = None, items: Optional[Iterable[str]] = None) -> "Rulebook":
        """Mine rules from bm. regions: clusters per row used only for breadth
        (default: pick-profile clusters, Rulebook.pick_regions).
        items: re-mine only these items and keep the other items' rules.
        changed_since: item -> patch time; rules touching a changed item are
        re-scored on builds after that time and marked stale if too few remain."""
        f = self.cfg
        rng = np.random.default_rng(f.seed)
        if f.sample is not None and len(bm) > f.sample:
            rows = np.sort(rng.choice(len(bm), f.sample, replace=False))
            bm = bm.subset(rows)
            regions = None if regions is None else regions[rows]
        self.region_names = list(region_names)
        w = bm.w
        XA = _picked_available(bm)
        G = _group_matrix(bm)
        elig = group_dot(np.asarray(G.T @ w).ravel(), bm.masks)
        self.pick_rate = np.asarray(XA.T @ w).ravel() / np.maximum(elig, 1e-12)
        mi = mutual_information(bm, w, signed=True)
        rep = mi < 0                                # v3.0: antagonists may anchor ("works against")
        mi = np.abs(mi)
        Xd = _dense_bool(bm.X)                     # bool: an eighth of float64, a quarter of float32
        H = bm.hi.astype(np.float32)
        from tri.rules.groups import derived_groups
        self.groups = derived_groups(self.c) if f.groups else {}
        tokens, Tk = self.context_matrix(bm)       # v3.0: planner fields, weapon class, archetype
        gnames = sorted(self.groups)
        gidx = [np.array([self.c.index[i] for i in self.groups[g] if i in self.c.index], np.int64) for g in gnames]
        GC = np.stack([Xd[:, ix].sum(1) for ix in gidx], 1).astype(np.float32) if gnames \
            else np.zeros((len(bm), 0), np.float32)
        own = regions is None
        regions = self.pick_regions(bm, f.breadth_regions) if own else regions
        R = int(regions.max()) + 1 if len(regions) else 0
        gmix = np.bincount(regions, weights=w, minlength=R) if R else np.zeros(0)
        todo = list(self.c.ids) if items is None else [i for i in items if i in self.c.index]
        keep = [r for r in self.rules if r.item not in set(todo)] if items is not None else []
        an, ag_grid, AG = aggregate_matrix(H, self.c.stats) if f.aggregates else ([], [], np.zeros((len(bm), 0), np.float32))
        ctx = dict(w=w, Xd=Xd, H=H, Tk=Tk, mi=mi, rep=rep, regions=regions, gmix=gmix, masks=bm.masks, gid=bm.gid,
                   tokens=list(tokens), AG=AG, ag_names=an, ag_grid=ag_grid, GC=GC, gnames=gnames, gidx=gidx,
                   pick_ZC=self._pick_ZC if (own and f.leave_item_out) else None)
        new: List[PassRule] = []
        vet: Dict[str, dict] = {}
        jobs = _n_jobs(f, len(todo), len(bm))
        if jobs > 1:
            from joblib import Parallel, delayed
            lite = Rulebook(self.c, self.cfg)          # what a worker needs: catalog, cfg, pick rates
            lite.pick_rate, lite.region_names, lite.groups = self.pick_rate, self.region_names, self.groups
            chunks = [todo[i::jobs * 4] for i in range(jobs * 4)]
            parts = Parallel(n_jobs=jobs, backend="loky", max_nbytes="1M")(
                delayed(_mine_chunk)(lite, ctx, ch) for ch in chunks if ch)
            by = {}
            for part in parts:
                for item, rules, info in part:
                    by[item] = (rules, info)
            for item in todo:                          # catalog order, as the serial loop
                rules, info = by.get(item, ([], {}))
                new.extend(rules)
                if info:
                    vet[item] = info
        else:
            for item in todo:
                info: dict = {}
                new.extend(self._mine_one(item, ctx, info))
                if info:
                    vet[item] = info
        self.vetoed = ({k: v for k, v in self.vetoed.items() if k not in set(todo)} if items is not None else {})
        self.vetoed.update(vet)
        self.rules = keep + new
        if changed_since:
            self._rescore_changed(bm, changed_since)
        from tri.rules.triviality import cull           # v3.0.5: group baselines
        cull(self.rules, self.c, self.groups)
        return self

    def _mine_one(self, item: str, ctx: dict, info: Optional[dict] = None) -> List[PassRule]:
        """Rules of one item (Rulebook.mine). v3.0: grounded stats first; an item
        left without a take rule is searched once more with every stat
        (PassConfig.ground_fallback): a stat convention, marked "ungrounded"."""
        meta: dict = {}
        out = self._mine_item(item, ctx, info, meta=meta)
        f = self.cfg
        if f.grounded and f.ground_fallback and meta.get("cover", 0.0) < f.ground_cover:
            from tri.rules.groups import grounded_stats
            if len(grounded_stats(self.c, item, self.groups)) < len(self.c.stats):
                prior = [r for r in out if r.effect == TAKE]
                wide = self._mine_item(item, ctx, info, wide=True, prior=prior)
                takes = [r for r in wide if r.effect == TAKE]
                for r in takes:
                    if r not in prior and any(c.kind == "stat" and c.key not in meta.get("grounded", ())
                                              for c in r.conds):
                        r.note = (r.note + "; " if r.note else "") + "ungrounded stat convention"
                out = takes + [r for r in out if r.effect != TAKE]
        return out

    def _mine_item(self, item: str, ctx: dict, info: Optional[dict] = None, wide: bool = False,
                   prior: Sequence[PassRule] = (), meta: Optional[dict] = None) -> List[PassRule]:
        """One search of an item's rules (wide: every stat, not only grounded ones). Independent of every other item,
        with its own random stream (seed, item index), so the result does not
        depend on the order items are mined in or on n_jobs. info (if given)
        receives why the item has no take rule: the cluster that vetoed its
        unconditional rule."""
        from sklearn.tree import DecisionTreeClassifier
        f = self.cfg
        w, Xd, H, Tk, mi = ctx["w"], ctx["Xd"], ctx["H"], ctx["Tk"], ctx["mi"]
        regions, gmix, tokens = ctx["regions"], ctx["gmix"], ctx["tokens"]
        AG = ctx.get("AG"); an = list(ctx.get("ag_names") or []); ag_grid = list(ctx.get("ag_grid") or [])
        if AG is None:
            AG = np.zeros((len(H), 0), np.float32)
        GC = ctx.get("GC"); gn = list(ctx.get("gnames") or [])
        if GC is None:
            GC = np.zeros((len(H), 0), np.float32)
        sidx = list(range(H.shape[1]))
        if f.grounded and not wide:                 # v3.0: stats of the item's own requirements and groups
            from tri.rules.groups import grounded_stats
            gs = grounded_stats(self.c, item, self.groups)
            sidx = [i for i, s_ in enumerate(self.c.stats) if s_ in gs]
        S, K, A, NG = len(sidx), Tk.shape[1], AG.shape[1], GC.shape[1]
        R = len(gmix)
        out: List[PassRule] = []
        t = self.c.index[item]
        base = float(self.pick_rate[t]) if self.pick_rate is not None else 0.0
        rng = np.random.default_rng([f.seed, t])
        rows = np.nonzero(ctx["masks"][ctx["gid"], t])[0]
        if w[rows].sum() < f.min_support:
            return out
        if f.item_rows is not None and len(rows) > f.item_rows:
            rows = np.sort(rng.choice(rows, f.item_rows, replace=False))
        rows = rows[rng.permutation(len(rows))]
        half = len(rows) // 2
        y_all = Xd[rows, t] > 0
        w_all = w[rows]
        excl = set(self.c.items[item].exclusive_with)
        pool = None
        if f.anchors == "related":                  # v3.0.1: prerequisites, group mates (no dependants)
            from tri.rules.groups import related_items
            pool = {self.c.index[i] for i in related_items(self.c, item, self.groups)}
            if ctx.get("rep") is not None:          # antagonists: picking j lowers the rate of the item
                pool |= set(np.nonzero(ctx["rep"][:, t])[0].tolist())
        from tri.rules.groups import dependants, tautological_groups
        deps = dependants(self.c, item)              # v3.0.1: "a dependant ⇒ item" is game data
        co = [] if f.anchors == "none" else \
            [j for j in np.argsort(-mi[:, t]) if j != t and mi[j, t] > 1e-4 and self.c.ids[j] not in excl
             and self.c.ids[j] not in deps and (pool is None or j in pool)][: f.co_items]
        Grow = GC[rows].copy()                      # group counts without the item itself
        if NG:
            ing = np.array([item in self.groups.get(g, ()) for g in gn], bool)
            if ing.any():
                Grow[:, ing] -= y_all[:, None].astype(np.float32)
            taut = tautological_groups(self.c, item, self.groups, deps)
            tg = np.array([g in taut for g in gn], bool)
            if tg.any():
                Grow[:, tg] = 0.0                    # constant: never split on, never a literal
        names = ([("stat", self.c.stats[s]) for s in sidx] + [("agg", an[a]) for a in range(A)]
                 + [("context", tokens[k]) for k in range(K)] + [("group", gn[g]) for g in range(NG)]
                 + [("item", self.c.ids[j]) for j in co])
        grid = {an[a]: ag_grid[a] for a in range(A)}
        if f.replicate:                             # v3.0: discover on A, certify on B, replicate on C
            q2 = half + (len(rows) - half) // 2
            A_, B_, C_ = np.arange(half), np.arange(half, q2), np.arange(q2, len(rows))
        else:
            A_, B_, C_ = np.arange(half), np.arange(half, len(rows)), None
        HA = H[rows[A_]][:, sidx]
        GA = AG[rows[A_]]
        if f.stat_step > 1:                         # thresholds only at multiples of stat_step
            HA = np.floor(HA / f.stat_step) * f.stat_step
            if A:
                on = np.array(ag_grid, bool)
                GA = GA.copy(); GA[:, on] = np.floor(GA[:, on] / f.stat_step) * f.stat_step
        F = np.concatenate([HA, GA, Tk[rows[A_]], Grow[A_], Xd[rows[A_]][:, co]], 1)
        yA = y_all[A_]
        cand: Dict[str, List[Condition]] = {"": []}
        nb = S + A + K + NG
        for nf in ([F.shape[1], nb] if co else [nb]):
            if nf == 0 or yA.all() or not yA.any():
                continue
            tree = DecisionTreeClassifier(criterion="entropy", max_depth=f.max_depth,
                                          min_samples_leaf=max(5, int(f.min_support / 3)), random_state=f.seed)
            tree.fit(F[:, :nf], yA, sample_weight=w_all[A_])
            for conds in self._paths(tree.tree_, names, f.stat_step, grid):
                cand.setdefault("&".join(sorted(c.id() for c in conds)), conds)
        cols = {}                                   # condition masks over this item's rows

        def cm(c: Condition) -> np.ndarray:
            k = c.id()
            if k not in cols:
                if c.kind in ("stat", "agg"):
                    v = H[rows, self.c.stat_index[c.key]] if c.kind == "stat" else AG[rows, an.index(c.key)]
                    cols[k] = v >= c.value if c.op == ">=" else v < c.value
                elif c.kind == "context":
                    m = Tk[rows, tokens.index(c.key)] > 0 if c.key in tokens else np.zeros(len(rows), bool)
                    cols[k] = m if c.op == "has" else ~m
                elif c.kind == "group":
                    v = Grow[:, gn.index(c.key)] if c.key in gn else np.zeros(len(rows), np.float32)
                    cols[k] = v >= c.value if c.op == ">=" else v < c.value
                else:
                    m = Xd[rows, self.c.index[c.key]] > 0
                    cols[k] = m if c.op == "has" else ~m
            return cols[k]

        def cell(conds) -> np.ndarray:
            m = np.ones(len(rows), bool)
            for c in conds:
                m &= cm(c)
            return m

        def stats(conds, part):
            m = cell(conds)[part]
            ww = w_all[part][m]; Sw = float(ww.sum())
            if Sw <= 0:
                return 0.0, 0.0, 0.0
            return float(ww[y_all[part][m]].sum() / Sw), Sw, Sw * Sw / max(float((ww * ww).sum()), 1e-12)

        def contrast(conds, part):
            best = None
            for i, c in enumerate(conds):
                flip = Condition(c.kind, c.key, {">=": "<", "<": ">=", "has": "not", "not": "has"}[c.op], c.value)
                p, Sw, _ = stats(conds[:i] + [flip] + conds[i + 1:], part)
                if Sw >= f.min_support / 2 and (best is None or p > best):
                    best = p
            return best
        reg_rows = regions[rows]
        ZC = ctx.get("pick_ZC")
        if ZC is not None and R:                    # leave item t out of its own cluster assignment
            Z, C = ZC
            Zr = Z[rows]
            sc = np.asarray((Zr @ C.T)) if not sp.issparse(Zr @ C.T) else (Zr @ C.T).toarray()
            zt = np.asarray(Zr[:, t].todense()).ravel()
            reg_rows = np.asarray(np.argmax(sc - zt[:, None] * C[:, t][None, :], 1)).ravel()

        def veto(conds, part, eff):
            """The pick-profile cluster that contradicts the rule inside its cell,
            or None. A cluster vetoes if it carries at least
            max(uniform_support, uniform_share x cell weight), its own bound is on
            the wrong side of the threshold, and its rate differs from the rest of
            the cell by a two-proportion test at z (a real minority archetype
            hidden in a population-level cell, not a handful of builds)."""
            m = cell(conds)[part]
            rg, ww, yy = reg_rows[part][m], w_all[part][m], y_all[part][m]
            tot = np.bincount(rg, weights=ww, minlength=R)
            pos = np.bincount(rg, weights=ww * yy, minlength=R)
            Sw, P = float(tot.sum()), float(pos.sum())
            need = max(f.uniform_support, f.uniform_share * Sw)
            worst, wv = None, 0.0
            for g in np.nonzero(tot >= need)[0]:
                pg = pos[g] / tot[g]
                lo, hi = wilson(pg, tot[g], f.z)
                bad = (eff == TAKE and hi < f.theta_take) or (eff == NEVER and lo > f.theta_never)
                if not bad:
                    continue
                rest = Sw - tot[g]
                if rest >= 1.0:
                    pr = (P - pos[g]) / rest
                    pp = min(max(P / max(Sw, 1e-12), 1e-6), 1 - 1e-6)
                    zs = (pr - pg if eff == TAKE else pg - pr) / math.sqrt(pp * (1 - pp) * (1 / tot[g] + 1 / rest))
                    if zs < f.z:
                        continue
                if tot[g] > wv:
                    worst, wv = int(g), float(tot[g])
            return worst
        last_veto: Dict[str, Optional[int]] = {}

        def certify(conds, part, ms, z=None):
            z = f.z if z is None else z
            p, Sw, ne = stats(conds, part)
            if Sw < ms:
                return None
            lo, hi = wilson(p, ne, z)
            if lo >= f.theta_take:
                if conds and base < f.theta_take and p < (1 + f.min_lift) * base:
                    return None                      # the condition adds nothing over the item's base rate
                g = veto(conds, part, TAKE)
                last_veto["g"] = g
                return (TAKE, p, lo, Sw, ne, None) if g is None else None
            if hi <= f.theta_never and conds and veto(conds, part, NEVER) is None:
                ct = contrast(conds, part)
                if ct is not None and ct >= max(f.theta_contrast, f.contrast_ratio * hi):
                    return (NEVER, p, hi, Sw, ne, ct)
            return None

        def repair(conds, g):
            """Literal (from the same features) that best splits cluster g off the cell, on A."""
            m = cell(conds)[A_]
            if m.sum() < 10:
                return None
            tgt = reg_rows[A_][m] == g
            if tgt.all() or not tgt.any():
                return None
            st = DecisionTreeClassifier(criterion="entropy", max_depth=1, min_samples_leaf=5, random_state=f.seed)
            st.fit(F[m], tgt, sample_weight=w_all[A_][m])
            tr = st.tree_
            if tr.node_count < 3:
                return None
            l, r = tr.children_left[0], tr.children_right[0]
            share = lambda n: tr.value[n][0][1] / max(tr.value[n][0].sum(), 1e-12)
            side = "l" if share(l) < share(r) else "r"
            kind, key = names[tr.feature[0]]
            th = float(tr.threshold[0])
            if kind == "group":
                return Condition("group", key, "<" if side == "l" else ">=", float(math.floor(th) + 1))
            if kind in ("stat", "agg"):
                step = f.stat_step if (kind == "stat" or grid.get(key, True)) else 1
                v = float(step * (math.floor(th / step) + 1))
                return Condition(kind, key, "<" if side == "l" else ">=", v)
            if kind == "context":
                return Condition("context", key, "not" if side == "l" else "has")
            return Condition("item", key, "has") if side == "r" else None
        ms = f.min_support / 2                       # each half carries half the weight
        msB = f.min_support / (4 if C_ is not None else 2)

        def necessary(conds, part, eff) -> bool:
            """v3.0: every literal must matter on the certification split: the
            builds it alone excludes differ from the rule's builds by a
            one-sided two-proportion test at z (math §6.12)."""
            m = cell(conds)[part]
            ww = w_all[part][m]; S1 = float(ww.sum())
            if S1 <= 0:
                return False
            p1 = float(ww[y_all[part][m]].sum()) / S1
            n1 = S1 * S1 / max(float((ww * ww).sum()), 1e-12)
            for i in range(len(conds)):
                mo = cell(conds[:i] + conds[i + 1:])[part] & ~m
                wo = w_all[part][mo]; So = float(wo.sum())
                if So < 1.0:
                    return False                     # the literal excludes nothing: it says nothing
                po = float(wo[y_all[part][mo]].sum()) / So
                no = So * So / max(float((wo * wo).sum()), 1e-12)
                pp = min(max((p1 * n1 + po * no) / (n1 + no), 1e-6), 1 - 1e-6)
                zs = ((p1 - po) if eff == TAKE else (po - p1)) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / no))
                if zs < f.z:
                    return False
            return True
        found: Dict[str, Tuple[List[Condition], tuple]] = {}
        for ck, conds in cand.items():
            last_veto.clear()
            res = certify(conds, A_, ms)
            conds = list(conds)
            if res is None and last_veto.get("g") is not None:
                if ck == "" and info is not None:
                    g = last_veto["g"]
                    mA = reg_rows[A_] == g
                    tw = float(w_all[A_][mA].sum())
                    info.update(cluster=int(g), weight=round(2 * tw, 1),
                                rate=round(float(w_all[A_][mA & yA].sum() / max(tw, 1e-12)), 4))
                for _ in range(f.repair_steps):
                    lit = repair(conds, last_veto["g"])
                    if lit is None or any(c.id() == lit.id() for c in conds):
                        break
                    conds = conds + [lit]
                    last_veto.clear()
                    res = certify(conds, A_, ms)
                    if res is not None or last_veto.get("g") is None:
                        break
            if res is None:
                continue
            while True:                              # shortest form that still passes on A
                best = None
                for i in range(len(conds)):
                    trial = conds[:i] + conds[i + 1:]
                    r2 = certify(trial, A_, ms)
                    if r2 is not None and r2[0] == res[0] and (best is None or r2[3] > best[1][3]):
                        best = (trial, r2)
                if best is None:
                    break
                conds, res = best
            found.setdefault("&".join(sorted(c.id() for c in conds)), (conds, res))
        zB = f.z
        if f.multiple_testing == "bonferroni" and len(found) > 1:
            from statistics import NormalDist
            alpha = 1 - NormalDist().cdf(f.z)
            zB = NormalDist().inv_cdf(1 - alpha / len(found))
        got: Dict[str, PassRule] = {}
        scale = float(w_all.sum()) / max(float(w_all[B_].sum()), 1e-12)
        for conds, res in found.values():
            fin = certify(conds, B_, msB, zB)        # the certificate: independent split
            if fin is None or fin[0] != res[0]:
                continue
            if C_ is not None:                       # v3.0: replicated on a third split
                rp = certify(conds, C_, msB, f.z)    # a replication, at the unadjusted level
                if rp is None or rp[0] != res[0]:
                    continue
            if f.necessity and conds and not necessary(conds, B_, fin[0]):
                continue
            eff, p, b, Sw, ne, ct = fin
            rule = PassRule(CENTRAL if eff == TAKE else EXCLUSION, item, conds, p, b, scale * Sw, ne,
                            p / max(base, 1e-12), ct, eff)
            got.setdefault(rule.key, rule)
        wr = w_all
        if wide:                                     # the grounded take rules compete with the wide ones
            if prior:
                got = {k: r for k, r in got.items() if r.effect == TAKE}
            for r in prior:
                got.setdefault(r.key, r)
        rs = sorted(got.values(), key=lambda r: (len(r.conds), -r.support))
        kept: List[PassRule] = []
        for r in rs:                                 # drop rules implied by a more general one
            if any(k.effect == r.effect and _implied(k.conds, r.conds) for k in kept):
                continue
            kept.append(r)
        kept = self._resolve_conflicts(kept, cm, y_all, w_all, len(rows))
        if f.max_overlap < 1.0:                      # one take rule per set of covered builds (proxy families)
            dd: List[Tuple[PassRule, np.ndarray]] = []
            arch = lambda r: any(c.kind == "context" and c.key.startswith("archetype:") for c in r.conds)
            for r in sorted(kept, key=lambda r: not arch(r)):   # v3.0.1: the archetype reading over its proxies
                m = cell(r.conds)
                if r.effect == TAKE and any(k.effect == TAKE and
                       float(wr[m & mk].sum()) >= f.max_overlap * max(float(wr[m | mk].sum()), 1e-12)
                       for k, mk in dd):
                    continue
                dd.append((r, m))
            kept = [r for r, _ in dd]
        if wide and prior:                           # a grounded rule whose cell lies inside a wider
            pk = {r.key for r in prior}              # (ungrounded) take rule is its proxy: drop it
            ms_ = {r.key: cell(r.conds) for r in kept if r.effect == TAKE}
            kept = [r for r in kept if not (r.key in pk and any(
                k.effect == TAKE and k.key not in pk and k.support > r.support and
                float(wr[ms_[r.key] & ms_[k.key]].sum()) >= f.max_overlap * max(float(wr[ms_[r.key]].sum()), 1e-12)
                for k in kept))]
        kept = [r for r in kept if r.effect == TAKE][: f.max_rules_per_item] + \
               [r for r in kept if r.effect == NEVER][: f.max_never_per_item]
        for r in kept:
            if r.effect == TAKE and all(c.kind in ("stat", "agg", "group") for c in r.conds):
                # v3.0.1: group counts are build-wide too: a rule spread over the
                # corpus's pick profiles is foundational whichever literal carries it
                m = cell(r.conds)
                r.breadth = round(self._breadth(regions[rows][m], wr[m], gmix), 3)
                if r.breadth >= f.min_breadth:
                    r.pass_ = FOUNDATIONAL
            out.append(r)
        if meta is not None:                          # share of the item's takers its take rules cover
            yw = float(wr[y_all].sum())
            cov = np.zeros(len(rows), bool)
            for r in out:
                if r.effect == TAKE:
                    cov |= cell(r.conds)
            meta["cover"] = float(wr[cov & y_all].sum()) / yw if yw > 0 else 1.0
            meta["grounded"] = {self.c.stats[s_] for s_ in sidx}
        if info is not None and any(r.effect == TAKE for r in out):
            info.clear()
        return out

    def _resolve_conflicts(self, rules: List[PassRule], cm, y, w, n) -> List[PassRule]:
        """A take rule and a never rule of one item can both apply to a build.
        Where their cells overlap with enough weight (uniform_support), the
        observed rate there decides which one is wrong: a rate clearly above
        theta_never drops the never rule, clearly below theta_take drops the
        take rule. Smaller overlaps are left; the answer reports them."""
        f = self.cfg

        def mask(r):
            m = np.ones(n, bool)
            for c in r.conds:
                m &= cm(c)
            return m
        drop = set()
        takes = [r for r in rules if r.effect == TAKE]
        nevers = [r for r in rules if r.effect == NEVER]
        for a in takes:
            ma = mask(a)
            for b in nevers:
                m = ma & mask(b)
                Sw = float(w[m].sum())
                if Sw < f.uniform_support:
                    continue
                lo, hi = wilson(float(w[m][y[m]].sum()) / Sw, Sw, f.z)
                if lo > f.theta_never:
                    drop.add(id(b))
                if hi < f.theta_take:
                    drop.add(id(a))
        return [r for r in rules if id(r) not in drop]

    def pick_regions(self, bm: BuildMatrix, k: int, iters: int = 30) -> np.ndarray:
        """Clusters of builds by what they pick (spherical k-means on picks
        weighted by -log pick rate). Used only for breadth: it must not see stats,
        or a stat convention would look narrow because the clusters encode the stat."""
        rng = np.random.default_rng(self.cfg.seed)
        idf = np.minimum(-np.log(np.clip(self.pick_rate, 1e-6, 1)), 4.0)
        Z = sp.csr_matrix(bm.X.multiply(idf[None, :]), dtype=np.float64)      # sparse: corpus-sized
        nrm = np.sqrt(np.asarray(Z.multiply(Z).sum(1)).ravel())
        Z = sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ Z
        n = Z.shape[0]
        k = min(k, n)
        if k <= 1:
            return np.zeros(n, np.int64)
        C = Z[rng.choice(n, k, replace=False)].toarray()
        for _ in range(iters):
            lab = np.asarray((Z @ C.T).argmax(1)).ravel()
            M = sp.csr_matrix((bm.w, (lab, np.arange(n))), shape=(k, n))
            newC = np.asarray((M @ Z).todense())
            nn = np.linalg.norm(newC, axis=1, keepdims=True)
            newC = np.where(nn > 1e-9, newC / np.maximum(nn, 1e-12), C)
            if np.allclose(newC, C):
                break
            C = newC
        self._pick_ZC = (Z, C)
        return np.asarray((Z @ C.T).argmax(1)).ravel()

    @staticmethod
    def _breadth(reg: np.ndarray, w: np.ndarray, gmix: np.ndarray) -> float:
        if not len(gmix) or gmix.sum() <= 0:
            return 1.0
        f = np.bincount(reg, weights=w, minlength=len(gmix))
        if f.sum() <= 0:
            return 0.0
        return float(np.minimum(f / f.sum(), gmix / gmix.sum()).sum())

    @staticmethod
    def _paths(tr, names, step: int = 1, grid: Optional[Dict[str, bool]] = None) -> List[List[Condition]]:
        """Condition lists of every node of a fitted sklearn tree (root = []).
        step: the tree saw stats floored to multiples of step, so a split between
        two grid values becomes the exact raw threshold at the upper grid value.
        Negative item conditions are left out (absence of a pick is not evidence
        on an unfinished build); a stat or context bound is kept at its tightest."""
        out = []

        def norm(conds):
            best: Dict[Tuple[str, str, str], Condition] = {}
            for c in conds:
                k = (c.kind, c.key, c.op)
                if k in best and c.kind in ("stat", "agg", "group"):
                    o = best[k]
                    c = c if (c.value > o.value if c.op == ">=" else c.value < o.value) else o
                best[k] = c
            return list(best.values())

        def walk(node, conds):
            out.append(norm(conds))
            l, r = tr.children_left[node], tr.children_right[node]
            if l < 0:
                return
            kind, key = names[tr.feature[node]]
            th = float(tr.threshold[node])
            if kind in ("stat", "agg"):
                st = step if (kind == "stat" or (grid or {}).get(key, True)) else 1
                v = float(st * (math.floor(th / st) + 1))       # x <= th  <=>  x < st*(floor(th/st)+1)
                lc, rc = Condition(kind, key, "<", v), Condition(kind, key, ">=", v)
            elif kind == "group":
                v = float(math.floor(th) + 1)
                lc, rc = Condition("group", key, "<", v), Condition("group", key, ">=", v)
            elif kind == "context":
                lc, rc = Condition("context", key, "not"), Condition("context", key, "has")
            else:
                lc, rc = None, Condition("item", key, "has")
            walk(l, conds + ([lc] if lc else []))
            walk(r, conds + [rc])
        walk(0, [])
        return out

    def _rescore_changed(self, bm: BuildMatrix, changed_since: Dict[str, float]) -> None:
        f = self.cfg
        ts = np.nan_to_num(bm.ts, nan=-np.inf)
        for r in self.rules:
            names = {r.item} | {c.key for c in r.conds if c.kind == "item"}
            t0 = max([changed_since[i] for i in names if i in changed_since], default=None)
            if t0 is None:
                continue
            p, Sw, ne = self.rule_stats(bm, r, bm.w * (ts >= t0))
            if Sw >= f.min_support:
                lo, hi = wilson(p, ne, f.z)
                ok = lo >= f.theta_take if r.effect == TAKE else hi <= f.theta_never
                r.confidence, r.support, r.n_eff = p, Sw, ne
                r.bound = lo if r.effect == TAKE else hi
                r.status = "active" if ok else "stale"
                r.note = f"re-scored on builds after the patch ({Sw:.0f})"
            else:
                r.status, r.note = "stale", f"changed by a patch; only {Sw:.0f} builds since"

    # ------------------------------------------------------------ checks
    def drift(self, bm: BuildMatrix, regions: np.ndarray, since: float, min_drop: float = 0.15) -> List[dict]:
        """Rules whose rate on builds after `since` moved by >= min_drop against
        the rule (or whose support vanished): how a patch shows up in the data."""
        ts = np.nan_to_num(bm.ts, nan=-np.inf)
        post = bm.w * (ts >= since); pre = bm.w - post
        out = []
        if post.sum() <= 0 or pre.sum() <= 0:
            return out
        for r in self.rules:
            c0, s0, _ = self.rule_stats(bm, r, pre)
            c1, s1, _ = self.rule_stats(bm, r, post)
            moved = (c0 - c1) if r.effect == TAKE else (c1 - c0)
            if s0 >= self.cfg.min_support and (moved >= min_drop or s1 < 0.1 * s0 * post.sum() / pre.sum()):
                out.append(dict(rule=r.text(self.region_names), before=round(c0, 3), after=round(c1, 3),
                                support_before=round(s0, 1), support_after=round(s1, 1)))
        return sorted(out, key=lambda d: -abs(d["after"] - d["before"]))

    def certification_check(self, bm: BuildMatrix, min_support: Optional[float] = None) -> dict:
        """Re-measure every active rule on other builds (e.g. held out). A take
        rule holds if its rate there is >= theta_take, a never rule if <= theta_never."""
        ms = self.cfg.min_support if min_support is None else min_support
        rows, held = [], 0
        for r in self.active():
            p, Sw, _ = self.rule_stats(bm, r)
            if Sw < ms:
                continue
            ok = p >= self.cfg.theta_take if r.effect == TAKE else p <= self.cfg.theta_never
            held += ok
            rows.append(dict(rule=r.text(self.region_names), rate=round(p, 4), support=round(Sw, 1), holds=bool(ok)))
        return dict(checked=len(rows), holding=held, share=round(held / len(rows), 4) if rows else None,
                    failures=[x for x in rows if not x["holds"]])

    # ------------------------------------------------------------ use
    def active(self) -> List[PassRule]:
        return [r for r in self.rules if r.status == "active"]

    def _apply(self, x: Ctx, effect: str, pass_: Optional[str]) -> List[PassRule]:
        out = []
        for r in self.rules:
            if r.status != "active" or r.effect != effect or (pass_ and r.pass_ != pass_):
                continue
            t = self.c.index.get(r.item)
            if t is None or not x.avail[t] or not r.applies(x):
                continue
            out.append(r)
        return sorted(out, key=lambda r: (-r.bound if effect == TAKE else r.bound, -r.support))

    def firing(self, x: Ctx, pass_: Optional[str] = None) -> List[PassRule]:
        """Active take rules that apply to the build (item available), best first."""
        return self._apply(x, TAKE, pass_)

    def forbidding(self, x: Ctx) -> List[PassRule]:
        """Active never rules that apply to the build."""
        return self._apply(x, NEVER, None)

    def pass_of(self, item: str) -> str:
        ps = {r.pass_ for r in self.rules if r.item == item and r.status == "active" and r.effect == TAKE}
        return FOUNDATIONAL if FOUNDATIONAL in ps else CENTRAL if CENTRAL in ps else LUXURY

    def separates(self, item: str, a, b) -> bool:
        """True if some active rule of `item` judges builds a and b (Build) by a
        stat or context condition on which they differ: a review of one does
        not transfer to the other for that item."""
        from tri.rules.groups import group_counts, rule_tokens
        ix = self.__dict__.get("_by_item")
        if ix is None or ix[0] is not self.rules or ix[2] != len(self.rules):            # v3.8: rules per item, not a scan per call
            d: Dict[str, list] = {}
            for r in self.rules:
                d.setdefault(r.item, []).append(r)
            ix = self._by_item = (self.rules, d, len(self.rules))
        ta = tb = ga = gb = None
        for r in ix[1].get(item, ()):
            if r.status != "active":
                continue
            for c in r.conds:
                if c.kind in ("stat", "agg"):
                    if c.kind == "stat":
                        va, vb = a.highest_stats.get(c.key, 0), b.highest_stats.get(c.key, 0)
                    else:
                        va, vb = aggregate_value(c.key, a.highest_stats), aggregate_value(c.key, b.highest_stats)
                    if (va >= c.value) != (vb >= c.value):
                        return True
                elif c.kind == "context" and not c.key.startswith("archetype:"):
                    if ta is None:
                        ta, tb = rule_tokens(a), rule_tokens(b)
                    if (c.key in ta) != (c.key in tb):
                        return True
                elif c.kind == "group":
                    if ga is None:
                        ga, gb = group_counts(a.items(), self.groups), group_counts(b.items(), self.groups)
                    va, vb = ga.get(c.key, 0) - (item in a.items()), gb.get(c.key, 0) - (item in b.items())
                    if (va >= c.value) != (vb >= c.value):
                        return True
        return False

    def for_item(self, item: str) -> List[PassRule]:
        return [r for r in self.rules if r.item == item or any(c.kind == "item" and c.key == item for c in r.conds)]

    def summary(self) -> dict:
        from collections import Counter
        c = Counter((r.pass_, r.status) for r in self.rules)
        per = Counter(self.pass_of(i) for i in self.c.ids)
        return dict(rules=len(self.rules), by_kind={f"{a}/{s}": n for (a, s), n in sorted(c.items())},
                    items_by_pass=dict(per),
                    conditions=dict(Counter(len(r.conds) for r in self.rules)))

    def report(self, limit: int = 0) -> List[str]:
        order = {FOUNDATIONAL: 0, CENTRAL: 1, EXCLUSION: 2}
        rs = sorted(self.rules, key=lambda r: (order.get(r.pass_, 3), r.item, -r.support))
        lines = [r.text(self.region_names) for r in rs]
        return lines[:limit] if limit else lines

    def mark_stale(self, items: Iterable[str], note: str) -> int:
        items = set(items); n = 0
        for r in self.rules:
            if r.item in items or any(c.kind == "item" and c.key in items for c in r.conds):
                r.status, r.note = "stale", note; n += 1
        return n

    # ------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return dict(cfg=asdict(self.cfg), rules=[r.to_dict() for r in self.rules], region_names=self.region_names,
                    pick_rate=None if self.pick_rate is None else [float(x) for x in self.pick_rate],
                    vetoed=self.vetoed, groups={g: sorted(m) for g, m in self.groups.items()})

    @staticmethod
    def from_json(d: dict, catalog) -> "Rulebook":
        rb = Rulebook(catalog, PassConfig(**d["cfg"]))
        rb.rules = [PassRule.from_dict(r) for r in d["rules"]]
        rb.region_names = d.get("region_names", [])
        rb.pick_rate = None if d.get("pick_rate") is None else np.array(d["pick_rate"])
        rb.vetoed = dict(d.get("vetoed") or {})
        rb.groups = {g: frozenset(m) for g, m in (d.get("groups") or {}).items()}
        from tri.rules.triviality import cull           # v3.0.5: runs fitted before the group baselines
        cull(rb.rules, catalog, rb.groups)
        return rb
