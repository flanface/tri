"""Caution (v3.1, math §14): hold back a mark when the evidence for it is thin,
or ask the person asking a few questions first.

Only worth-sourced changes can be held. Rule changes are certified and review
changes bind; neither is held. A worth change on item t with local rate q and
support n is held when

    z = |q - 1/2| / sqrt(q (1 - q) / (n + n0)) < z_min

i.e. the local rate is not clearly away from a coin flip. Questions are of two
kinds: "what archetype is your build?" (when the reading is unlocated, unsure or
two alternatives are close), "is X central to your concept?" (for held
removals, and luxury picks a completion keeps on thin evidence), "do you want
X?" (for held adds, v3.1.3) and "what weapon class is this build?" (when the
build commits to no weapon class and the focus guessed one, v3.1.3). Only necessary
questions are asked, at most max_questions (6) per call, least sure first;
answering may make new ones necessary (asked in the next round). v3.6: where the
local support is thin (an archetype with few builds or reviews), the population
rate stands in when it clearly agrees, and kept luxury picks are asked about only
when their worth is low, so thin archetypes do not turn into questionnaires.

Answers come from non-experts. They act locally, on the one call they were
given for: an archetype answer forces that archetype; "yes, central" locks the
item and treats it as central for that call; "no" only releases a hold (it never
overrides a rule); "not sure" does nothing; "want X: yes" takes a held add, "no" never adds X
on that call; a weapon class answer sets the class focus for that call. Across users, only a large consensus
moves the model (AnswerBook), and then only by a small worth note.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class CautionConfig:
    enabled: bool = True
    z_min: float = 0.6              # hold a worth change below this z (v3.2: was 1.0)
    prior_strength: float = 2.0     # n0: pseudo-support added to the local support
    min_confidence: Optional[float] = None   # archetype reading below this asks (None: the locator setting)
    archetype_gap: float = 0.15     # two alternatives this close in plausibility ask
    max_questions: Optional[int] = 6      # v3.6: a question budget per call, most useful first (None: every necessary one)
    pool: bool = True               # v3.6: thin local support borrows the population rate before asking
    pool_thin: float = 4.0          # local support below this is thin enough to borrow
    pool_support_cap: float = 30.0  # the borrowed support counts at most this much
    pool_z: float = 1.0             # the population rate must clear this z (same side) to stand in
    central_max_worth: float = 0.5  # complete: ask "central?" only about kept luxury picks below this worth
    drafts: bool = False            # Tri.as_review holds marks by default
    consensus_min_users: int = 5    # distinct users before answers move the model
    consensus_share: float = 0.8    # ... agreeing at least this share
    consensus_weight: float = 0.25  # scale of a consensus worth note (a review note has 1)
    consensus_jaccard: float = 0.9  # archetype consensus: builds this alike count as the same
    max_want: int = 2               # "do you want X?" for at most this many held adds (least sure first; v3.2: was 3)
    ask_weapon: bool = True         # ask the weapon class when the focus had to guess it


@dataclass
class Question:
    key: str                        # "archetype", "central:<item>", "want:<item>" or "weapon"
    kind: str                       # archetype | central | want | weapon
    text: str
    options: List[str]
    item: Optional[str] = None
    why: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answers:
    """What the asker said. central: item -> True (central) / False (not); unsure is absent.
    want: item -> True (take the held add) / False (do not add it). weapon: a weapon class."""
    archetype: Optional[str] = None
    central: Dict[str, bool] = field(default_factory=dict)
    want: Dict[str, bool] = field(default_factory=dict)
    weapon: Optional[str] = None

    @staticmethod
    def of(x) -> "Answers":
        if x is None:
            return Answers()
        if isinstance(x, Answers):
            return x
        bools = lambda d: {str(k): bool(v) for k, v in (d or {}).items() if v is not None}
        return Answers(x.get("archetype") or None, bools(x.get("central")), bools(x.get("want")),
                       x.get("weapon") or None)

    @property
    def unwanted(self) -> set:
        return {i for i, v in self.want.items() if v is False}

    @property
    def wanted(self) -> set:
        return {i for i, v in self.want.items() if v is True}

    def to_dict(self) -> dict:
        return dict(archetype=self.archetype, central=dict(self.central), want=dict(self.want), weapon=self.weapon)

    def __bool__(self) -> bool:
        return bool(self.archetype or self.central or self.want or self.weapon)


@dataclass
class Caution:
    sufficient: bool
    held: List[dict]                # [{item, action, z, worth, support, why}]
    questions: List[Question]
    info: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(sufficient=self.sufficient, held=list(self.held),
                    questions=[q.to_dict() for q in self.questions], info=dict(self.info))


def z_score(q: float, n: float, n0: float) -> float:
    q = float(np.clip(q, 0.01, 0.99))
    return abs(q - 0.5) / float(np.sqrt(q * (1 - q) / max(n + n0, 1e-9)))


def assess(tri, build, alts: Sequence, answers=None, mode: str = "review",
           archetype: Optional[str] = None) -> Caution:
    """Which worth changes of alts[0] to hold, and which questions to ask.
    alts: Tri.review alternatives (mode "review") or [Tri.complete result] ("complete")."""
    cc = tri.cfg.caution
    ans = Answers.of(answers)
    if not cc.enabled or not alts:
        return Caution(True, [], [])
    top = alts[0]
    eng = tri.engine()
    sits = eng.situations(build, max(len(alts), 1), force=tri.force(ans.archetype or archetype))
    s = next((x for x in sits if x.mode == top.mode), sits[0])
    ix = tri.catalog.index

    def z_of(it):
        t = ix[it]
        return z_score(float(s.q[t]), float(s.support[t]), cc.prior_strength), float(s.q[t]), float(s.support[t])

    pop = None
    if cc.pool and getattr(eng, "worth", None) is not None:
        try:
            pop = eng.worth.pick_rate()                 # (population rate, eligible weight), memoised
        except Exception:
            pop = None

    def pooled(it, q, n) -> bool:
        """v3.6 (math §14.1): an archetype with few builds or reviews has thin local
        support everywhere. There the population rate stands in when it sits clearly
        on the same side of 1/2 as the local rate, instead of asking."""
        if pop is None or n >= cc.pool_thin:
            return False
        t = ix[it]
        p, m = float(pop[0][t]), min(float(pop[1][t]), cc.pool_support_cap)
        return (p - 0.5) * (q - 0.5) > 0 and z_score(p, m, cc.prior_strength) >= cc.pool_z

    held = []
    for c in top.changes:
        if c.source != "worth" or c.item not in ix:
            continue
        if c.action == "remove" and ans.central.get(c.item) is False:
            continue                                    # "not central": the removal stands
        if c.action == "add" and (ans.central.get(c.item) is False or ans.want.get(c.item) is True):
            continue                                    # "yes, I want it": the add stands
        z, q, n = z_of(c.item)
        if z < cc.z_min and not pooled(c.item, q, n):
            held.append(dict(item=c.item, action=c.action, z=round(z, 2), worth=round(q, 3), support=round(n, 1),
                             why="insufficient information"))
    qs: List[Question] = []
    # archetype
    if not ans.archetype and not archetype:
        opts, why, score = [], "", 0.0
        ls = tri.locset
        rd = None
        if ls is not None and len(ls.keys) >= 2:
            rd = ls.read(tri._embed(build), build)
            lo = cc.min_confidence if cc.min_confidence is not None else ls.cfg.min_confidence
            if not rd.located:
                why, score = "the build sits away from every known archetype", 1.0
            elif rd.confidence < lo:
                why, score = f"the archetype reading is unsure ({rd.confidence:.2f})", 1.0 - rd.confidence
            opts = [n for n, _ in sorted(rd.posterior.items(), key=lambda kv: -kv[1])][:4]
        if len(alts) >= 2 and mode == "review":
            gap = float(alts[0].plausibility) - float(alts[1].plausibility)
            if gap < cc.archetype_gap and not why:
                why, score = f"two archetypes fit about equally (gap {gap:.2f})", 1.0 - gap
        if why:
            for a in alts:
                nm = (a.name or "").split(" (")[0]
                if nm and nm not in opts:
                    opts.append(nm)
            if opts:
                qs.append(Question("archetype", "archetype", "What archetype is your build?", opts[:5], None, why,
                                   score + 1.0))
    # central
    sel = set(build.items())
    cand: Dict[str, tuple] = {}
    hz = {h["item"]: h for h in held}
    for c in top.changes:
        if c.action != "remove" or c.item in ans.central or c.item not in ix:
            continue
        if c.item in hz:
            cand[c.item] = (1.0 / (1.0 + hz[c.item]["z"]), "the removal rests on thin evidence")
    if mode == "complete":                              # only where the evidence is thin (necessary)
        for it in sel:
            if it in ans.central or it not in ix or top.passes.get(it) != "luxury":
                continue
            z, q, n = z_of(it)
            if z < cc.z_min and q < cc.central_max_worth and not pooled(it, q, n):
                cand[it] = (1.0 / (1.0 + z), "kept as a luxury pick on thin evidence; it may be the point of your build")
    for it, (sc, why) in sorted(cand.items(), key=lambda kv: -kv[1][0]):
        qs.append(Question(f"central:{it}", "central", f"Is {it} central to your concept?",
                           ["yes", "no", "not sure"], it, why, sc))
    # want: the held adds, least sure first (v3.1.3)
    wq = sorted((h for h in held if h["action"] == "add" and h["item"] not in ans.want), key=lambda h: h["z"])
    for h in wq[:max(int(cc.max_want), 0)]:
        qs.append(Question(f"want:{h['item']}", "want", f"Do you want {h['item']}?", ["yes", "no"], h["item"],
                           "about to be added on thin evidence", 1.0 / (1.0 + h["z"])))
    # weapon (v3.2): the build records no weapon, so the weapon rules (Speed Demon,
    # Turtle Shell, Flaming Flourish) cannot be read; a weapon name or a class answers
    foc = getattr(s, "focus", None) or {}
    from tri.game.weapon_rules import ITEMS, needs_weapon
    if cc.ask_weapon and not ans.weapon and needs_weapon(build) and any(i in ix for i in ITEMS):
        opts = list(dict.fromkeys(c.lower() for c in list(foc.get("classes") or []) + list(foc.get("share") or {})
                                  + list(foc.get("candidates") or [])))[:4]
        why = "no weapon recorded; Speed Demon and Turtle Shell depend on it"
        if foc.get("source") in ("prior", "worth"):
            why += f"; the class was guessed ({', '.join(foc.get('classes') or [])})"
        qs.append(Question("weapon", "weapon", "What weapon does this build use?", opts, None, why, 1.5))
    rank = {"archetype": 0, "weapon": 1}                # v3.2: the weapon right after the archetype
    qs.sort(key=lambda q: (rank.get(q.kind, 2), -float(q.score)))   # v3.6: then the least sure first
    if cc.max_questions is not None:
        qs = qs[:max(int(cc.max_questions), 0)]
    return Caution(not held and not qs, held, qs, dict(mode=mode, region=top.mode, name=top.name))


class AnswerBook:
    """Answers from many users. Only consensus counts:
    central:   per (family, item), at least consensus_min_users distinct users, of whom
               at least consensus_share said yes -> a small core note (scale consensus_weight)
               at the mean coordinates of their builds;
    archetype: per near-identical build (item Jaccard >= consensus_jaccard), the same
               majority -> the default archetype for that build."""

    def __init__(self, rows: Sequence[dict], cfg: CautionConfig):
        self.rows = [r for r in rows if r.get("kind") in ("archetype", "central")]
        self.cfg = cfg

    @staticmethod
    def _latest(rows):
        last: Dict[tuple, dict] = {}
        for r in sorted(rows, key=lambda r: r.get("ts") or 0.0):
            last[(r.get("user"), r.get("_k"))] = r
        return list(last.values())

    def central_notes(self, tri) -> List[dict]:
        cc = self.cfg
        groups: Dict[tuple, list] = {}
        for r in self.rows:
            if r["kind"] != "central" or r.get("item") not in tri.catalog.index or r.get("value") is None:
                continue
            b = r["build"]
            fam = tri._family_of(b) if tri.space is not None else None
            groups.setdefault((fam, r["item"]), []).append(dict(r, _k=(fam, r["item"])))
        out = []
        for (fam, it), rs in groups.items():
            rs = self._latest(rs)
            users = {r.get("user") for r in rs}
            yes = sum(1 for r in rs if r["value"] is True or r["value"] == "yes")
            if len(users) < cc.consensus_min_users or yes < cc.consensus_share * len(rs):
                continue
            U = np.array([tri._embed(r["build"]) for r in rs if r["value"] in (True, "yes")])
            key = None
            ls = tri.locset
            if fam is not None and ls is not None:
                from tri.reviews.variants import family_key
                key = next((k for k in ls.keys if family_key(ls.names.get(k, k)) == fam), None)
            out.append(dict(kind="core", items=[tri.catalog.index[it]], names=[it], u=U.mean(0), key=key,
                            archetype=fam, review_id=None, scale=cc.consensus_weight, source="answers",
                            users=len(users)))
        return out

    def archetype_for(self, build) -> Optional[str]:
        cc = self.cfg
        S = set(build.items())
        rs = []
        for r in self.rows:
            if r["kind"] != "archetype" or not r.get("value"):
                continue
            T = set(r["build"].items())
            if S and len(S & T) / max(len(S | T), 1) >= cc.consensus_jaccard:
                rs.append(dict(r, _k="a"))
        rs = self._latest(rs)
        if len({r.get("user") for r in rs}) < cc.consensus_min_users:
            return None
        votes: Dict[str, int] = {}
        for r in rs:
            votes[str(r["value"])] = votes.get(str(r["value"]), 0) + 1
        best, n = max(votes.items(), key=lambda kv: kv[1])
        return best if n >= cc.consensus_share * len(rs) else None
