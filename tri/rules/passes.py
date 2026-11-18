"""The rule container: conjunctive rules on one item, with an effect.

A rule is a conjunction of conditions and an effect on one item:

    take  ITEM when C1 and C2 and ...     (foundational or central)
    never ITEM when C1 and C2 and ...     (exclusion)

Conditions look at what a build is:
    stat     <stat> >= v  /  <stat> < v        (highest stats)
    context  oath / murmur / weapon class / archetype is (not) v
    item     with ITEM2
    group    count of a known group's members >= v / < v

4.0: rules are hand-authored (tri.rules.authored); nothing is mined. A run saved
by 3.x may carry mined rules; they load but no longer act.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.stats import wilson  # noqa: F401  (re-exported for the selection)
from tri.model.matrix import BuildMatrix
from tri.rules.defs import Condition, Ctx, PassConfig

FOUNDATIONAL, CENTRAL, LUXURY, EXCLUSION = "foundational", "central", "luxury", "exclusion"
TAKE, NEVER = "take", "never"


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
    status: str = "active"         # active | stale | trivial (culled by a group baseline)
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


class Rulebook:
    """Certified rules. .firing(ctx) / .forbidding(ctx) give those that apply to a build."""

    def __init__(self, catalog, cfg: Optional[PassConfig] = None):
        self.c = catalog
        self.cfg = cfg or PassConfig()
        self.rules: List[PassRule] = []
        self.pick_rate: Optional[np.ndarray] = None
        self.region_names: List[str] = []
        self.vetoed: Dict[str, dict] = {}      # item -> why it has no take rule (the vetoing cluster)
        self.pruned: List[Tuple[str, str]] = []  # (key, reason) of rules prune() deleted
        self._pick_ZC = None                    # (Z, C) of the last pick_regions call (leave-item-out)
        self.groups: Dict[str, frozenset] = {}  # known groups (tri.rules.groups)
        self.reader = None                      # bm -> archetype key per row (Tri sets it)
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

    # ------------------------------------------------------------ preparation
    def prepare(self, bm: BuildMatrix, region_names: Sequence[str] = ()) -> "Rulebook":
        """The container the authored rules live in (4.0: nothing is mined):
        region names, eligible pick rates and the known item groups."""
        self.region_names = list(region_names)
        self.pick_rate = bm.pick_rate()[0] if bm is not None and len(bm) else None
        from tri.rules.groups import derived_groups
        self.groups = derived_groups(self.c) if self.cfg.groups else {}
        self.rules, self.vetoed, self.pruned = [], {}, []
        return self

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
        if ix is None or ix[0] is not self.rules or ix[2] != len(self.rules):            # rules per item, not a scan per call
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
        from tri.rules.triviality import cull, prune    # runs fitted before the baselines
        cull(rb.rules, catalog, rb.groups)
        rb.rules, rb.pruned = prune(rb.rules, catalog)
        return rb
