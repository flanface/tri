<<<<<<< HEAD
"""The selection rule set: the hand-authored rules in one rulebook that forces
foundational and central picks and exclusions (4.0: mined rules and rules read
off the reviews are gone; reviews act through the scorer).
=======
"""The selection rule set: mined rules, authored rules and review
evidence in one rulebook that decides foundational and central picks and
exclusions (math §6.12).
>>>>>>> main

    sel = build_selection(rulebook, authored, vocab, reviews, ledger, family_of, bm)
    sel.rules            # what fires: the authored rules
    sel.evidence[key]    # {status, support, counter, by, corpus}

For each authored rule, every reviewed build whose original context the rule
applies to is a case, shown to its author and never acted on:

    take t:   reviewer adds t, or keeps t unmarked (not on a build tri made) -> support
              reviewer removes t (not for space)                              -> counter
    never t:  the mirror image.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from tri.core.types import Action, strength_weight
from tri.rules.defs import Ctx
from tri.rules.groups import rule_tokens
from tri.rules.features import SPACE_REASONS
from tri.rules.passes import TAKE, PassRule, Rulebook, wilson
from tri.reviews.authority import made_by_tri


@dataclass
class SelectionConfig:
    keep_weight: float = 1.0       # an unmarked keep, relative to an explicit operation
<<<<<<< HEAD
=======
    # selection rules read off the reviews themselves (math §6.12.5)
    review_rules: bool = True
    family_min: int = 3            # A = a ⇒ t / ¬t: reviews of family a that bear on t
    family_ops: int = 2            # ... of which explicit operations on t (adds / removals / verdicts)
    family_share: float = 0.75     # ... and the side's share of their weight (0.75, was 0.8)
    found_min: int = 8             # ⊤ ⇒ t: builds reviewed in full (not made by tri) that can take t ...
    found_share: float = 0.75      # ... that keep or get t, as a share of them (0.75, was 0.85)
    found_families: int = 2        # ... in at least this many stated / located families
    # A = a ⇒ t from endorsement alone (math §6.12.5): t kept or added on
    # >= keep_share of the full reviewed builds of family a that can take it
    keep_rules: bool = True
    keep_min: int = 4
    keep_share: float = 0.75
    avail_only: bool = True        # denominators count only builds that can take t
>>>>>>> main


@dataclass
class _Case:
    rid: str
    order: tuple
    sign: int                      # +1 support, -1 counter
    weight: float
    ctx: Ctx


def _ctx(b, family_of, catalog, groups) -> Ctx:
    fam = family_of(b) if family_of is not None else None
    return Ctx(dict(b.highest_stats), set(b.items()), rule_tokens(b, archetype=fam), 0,
               np.ones(len(catalog.ids), bool), groups=groups)


def _cases(reviews, ledger, catalog, groups, family_of, cfg) -> Dict[str, List[tuple]]:
    """item -> [(sign, review id, order, weight, ctx, how)]: per item, the review
    signals about taking it (sign +1 take / -1 don't; how "op" an explicit
    operation or verdict, "keep" an unmarked keep)."""
    out: Dict[str, List[tuple]] = {}
    for r in reviews:
        lw = ledger.weights.get(r.review_id) if ledger is not None else None
        order = (ledger.order.get(r.review_id, 0) if ledger is not None else 0, float(getattr(r, "timestamp", 0) or 0))
        ctx = None
        marked = set(getattr(r, "neutral", None) or ())   # neutral items say nothing
        sig = []
        for j, o in enumerate(r.operations):
            ow = 0.0 if o.item_id in marked else (lw.ops[j] if lw is not None else 1.0)
            marked.add(o.item_id)
            if ow <= 0:
                continue
            w = ow * strength_weight(o.strength)
            if o.action == Action.ADD:
                sig.append((o.item_id, 1, w, "op"))
            elif o.action == Action.REMOVE and o.reason not in SPACE_REASONS:
                sig.append((o.item_id, -1, w, "op"))
        if not made_by_tri(r.original):
            for it in r.original.items():
                if it not in marked:
                    kw = (lw.keep.get(it, 1.0) if lw is not None else 1.0) * cfg.keep_weight
                    if kw > 0:
                        sig.append((it, 1, kw, "keep"))
        if not sig:
            continue
        ctx = _ctx(r.original, family_of, catalog, groups)
        for it, s, w, how in sig:
            out.setdefault(it, []).append((s, r.review_id, order, w, ctx, how))
    return out


def _evidence(r: PassRule, sig) -> List[_Case]:
    cs = []
    for s, rid, order, w, ctx, _how in sig:
        x = Ctx(ctx.hi, ctx.items - {r.item}, ctx.tokens, 0, ctx.avail, ctx.groups)
        if r.applies(x):
            cs.append(_Case(rid, order, s if r.effect == TAKE else -s, w, x))
    return cs


<<<<<<< HEAD
=======
def _candidates(r: PassRule, rb: Rulebook, sup: List[_Case], cnt: List[_Case]) -> List[tuple]:
    """(rank, literal) that exclude every counter case."""
    c, groups, step = rb.c, rb.groups, max(rb.cfg.stat_step, 1)
    have = {x.id() for x in r.conds}
    out = []
    inter = set.intersection(*[x.ctx.tokens for x in cnt])
    for t in inter:                                    # every counter has t: rule for ¬t
        out.append((0 if t.startswith("archetype:") else 1, Condition("context", t, "not")))
    seen = set().union(*[x.ctx.tokens for x in sup]) if sup else set()
    for t in seen - set().union(*[x.ctx.tokens for x in cnt]):
        out.append((0 if t.startswith("archetype:") else 1, Condition("context", t, "has")))
    rel = related_items(c, r.item, groups)
    for g, mem in groups.items():
        if r.item not in mem and not (mem & rel):
            continue
        v = [len((x.ctx.items - {r.item}) & mem) for x in cnt]
        if min(v) > 0:
            out.append((2, Condition("group", g, "<", float(min(v)))))
        out.append((2, Condition("group", g, ">=", float(max(v) + 1))))
    for y in rel:
        hs = [y in x.ctx.items for x in cnt]
        if all(hs):
            out.append((3, Condition("item", y, "not")))
        elif not any(hs):
            out.append((3, Condition("item", y, "has")))
    for s in grounded_stats(c, r.item, groups):
        v = [x.ctx.hi.get(s, 0) for x in cnt]
        lo = (min(v) // step) * step
        if lo > 0:
            out.append((4, Condition("stat", s, "<", float(lo))))
        hi = (max(v) // step + 1) * step
        out.append((4, Condition("stat", s, ">=", float(hi))))
    return [(k, l) for k, l in out if l.id() not in have]


def _certify(rb: Rulebook, bm, r: PassRule) -> Optional[tuple]:
    p, S, n = rb.rule_stats(bm, r)
    if S < rb.cfg.min_support:
        return None
    lo, hi = wilson(p, n, rb.cfg.z)
    ok = lo >= rb.cfg.theta_take if r.effect == TAKE else hi <= rb.cfg.theta_never
    return (p, lo if r.effect == TAKE else hi, S, n) if ok else None


def _refine(r: PassRule, rb: Rulebook, bm, sup, cnt, cfg) -> Optional[PassRule]:
    if len(r.conds) >= rb.cfg.max_depth + 1:
        return None
    best = []
    for rank, lit in _candidates(r, rb, sup, cnt):
        q = PassRule(r.pass_, r.item, list(r.conds) + [lit], r.confidence, r.bound, r.support,
                     effect=r.effect, source="data+review")
        if any(q.applies(x.ctx) for x in cnt):
            continue
        kept = sum(x.weight for x in sup if q.applies(x.ctx))
        if kept <= 0:                                  # a refinement must keep some support
            continue
        best.append((-kept, rank, lit.op != "not", len(lit.text()), q))   # ties: exclude the counter context
    best.sort(key=lambda z: z[:4])
    for *_, q in best[:12]:
        if bm is not None and cfg.corpus_check:
            st = _certify(rb, bm, q)
            if st is None:
                continue
            q.confidence, q.bound, q.support, q.n_eff = st
        return q
    return None


def _fam(ctx: Ctx) -> Optional[str]:
    for t in ctx.tokens:
        if t.startswith("archetype:"):
            return t[10:]
    return None


def _views(reviews, family_of, catalog) -> List[tuple]:
    """(review id, build, family) of the builds reviewed in full and not made by tri:
    the denominators of ⊤ ⇒ t (every item they hold unmarked is endorsed)."""
    out = []
    for r in reviews:
        if made_by_tri(r.original):
            continue
        out.append((r.review_id, r.original, family_of(r.original) if family_of is not None else None))
    return out


def review_rules(sig: Dict[str, List[tuple]], views: List[tuple], catalog, cfg: SelectionConfig,
                 skip=frozenset()) -> List[PassRule]:
    """Selection rules the reviews state directly (math §6.12.5). Per item t
    and stated / located family a, over the reviews that bear on t:

      A = a ⇒ ¬t   [X]  >= family_ops explicit removals (not for space) of t on
                         family a, the counters' weight share >= family_share,
                         and the newest of these reviews a counter
      A = a ⇒ t    [C]  the mirror image (adds, upheld adds; unmarked keeps count
                         toward the share, never toward family_ops)
      A = a ⇒ t    [C]  (v3.0.2, keep_rules) t kept or added on >= keep_share of
                         the full, non-tri reviewed builds of family a that can
                         take t (>= keep_min of them), no explicit counter on a
                         newer than its newest support
      ⊤ ⇒ t        [F]  t kept or added on >= found_share of the full, non-tri
                         reviewed builds that can take t (>= found_min of them,
                         in >= found_families families), no explicit counter
                         newer than its newest support. A family with an [X]
                         rule on t is excepted: (A ≠ a ∧ ...) ⇒ t.

    "Can take t" (avail_only): t's requirements hold at the build's
    pre-shrine or final stats (Catalog.available_in), or the build holds t. A
    build that could never take t says nothing about it.

    Reviews are accurate: these rules are not certified on the corpus; they are
    what reviewers said, in the rule language. Their weight is the reviews'."""
    out: List[PassRule] = []
    avail: Dict[str, Dict[str, bool]] = {}             # review id -> {item: can take}

    def can(v, it, held_ids) -> bool:
        if not getattr(cfg, "avail_only", True) or v[0] in held_ids or v[1] is None:
            return True
        a = avail.get(v[0])
        if a is None:
            try:
                m = catalog.availability_in(v[1])
                a = {i: bool(x) for i, x in zip(catalog.ids, m)}
            except Exception:                          # noqa: BLE001 (a build without stats)
                a = {}
            avail[v[0]] = a
        return a.get(it, True)

    for it, cs in sig.items():
        if it in skip or it not in catalog.index or catalog.items[it].kind != Kind.TALENT:
            continue                                   # selection rules decide talents; mantras keep worth
        fams: Dict[str, list] = {}
        for c in cs:
            f = _fam(c[4])
            if f:
                fams.setdefault(f, []).append(c)
        never_fams = []
        for f, xs in sorted(fams.items()):
            rids = {x[1] for x in xs}
            if len(rids) < cfg.family_min:
                continue
            wp = sum(x[3] for x in xs if x[0] > 0); wn = sum(x[3] for x in xs if x[0] < 0)
            newest = max(xs, key=lambda x: (x[2], -x[0]))
            tok = Condition("context", f"archetype:{f}", "has")
            nops = lambda sg: len({x[1] for x in xs if x[0] == sg and x[5] == "op"})
            if nops(-1) >= cfg.family_ops and wn >= cfg.family_share * (wp + wn) and newest[0] < 0:
                out.append(PassRule(EXCLUSION, it, [tok], wp / max(wp + wn, 1e-12), wp / max(wp + wn, 1e-12),
                                    float(len(rids)), float(len(rids)), effect=NEVER, source="review",
                                    note=f"reviews −{nops(-1)} on {f}, newest {newest[1]}"))
                never_fams.append(f)
            elif nops(1) >= cfg.family_ops and wp >= cfg.family_share * (wp + wn) and newest[0] > 0:
                out.append(PassRule(CENTRAL, it, [tok], wp / max(wp + wn, 1e-12), wp / max(wp + wn, 1e-12),
                                    float(len(rids)), float(len(rids)), effect=TAKE, source="review",
                                    note=f"reviews +{nops(1)} on {f}, newest {newest[1]}"))
        pos = {x[1]: x for x in cs if x[0] > 0}
        stated = {r.key for r in out if r.item == it}
        kept: List[PassRule] = []                      # added unless ⊤ ⇒ t forms (it covers them)
        if getattr(cfg, "keep_rules", False):          # A = a ⇒ t from endorsement
            for f in sorted({v[2] for v in views if v[2]} - set(never_fams)):
                tok = Condition("context", f"archetype:{f}", "has")
                if PassRule(CENTRAL, it, [tok], 0, 0, 0, 0, effect=TAKE).key in stated:
                    continue
                vf = [v for v in views if v[2] == f and can(v, it, pos)]
                if len(vf) < cfg.keep_min:
                    continue
                held = [v for v in vf if v[0] in pos]
                negf = [x for x in cs if x[0] < 0 and _fam(x[4]) == f]
                if len(held) < cfg.keep_share * len(vf) or \
                        (negf and held and max(x[2] for x in negf) > max(pos[v[0]][2] for v in held)):
                    continue
                p = len(held) / len(vf)
                kept.append(PassRule(CENTRAL, it, [tok], p, p, float(len(vf)), float(len(vf)), effect=TAKE,
                                    source="review", note=f"kept or added on {len(held)} of {len(vf)} reviewed {f} "
                                                          "builds that can take it"))
        # ⊤ ⇒ t over the full, non-tri reviews (with the [X] families excepted)
        neg = [x for x in cs if x[0] < 0 and _fam(x[4]) not in never_fams]
        vs = [v for v in views if _fam_ok(v[2], never_fams) and can(v, it, pos)]
        held = [v for v in vs if v[0] in pos]          # every build that can take t counts
        fam_seen = {v[2] for v in held if v[2]}
        if len(vs) < cfg.found_min or len(held) < cfg.found_share * len(vs) or \
                len(fam_seen) < cfg.found_families or \
                (neg and max(x[2] for x in neg) > max(x[2] for x in pos.values())):
            out.extend(kept)
            continue
        conds = [Condition("context", f"archetype:{f}", "not") for f in never_fams]
        p = len(held) / len(vs)
        out.append(PassRule(FOUNDATIONAL, it, conds, p, p, float(len(vs)), float(len(vs)), effect=TAKE,
                            source="review", breadth=round(len(fam_seen) / max(len({v[2] for v in vs if v[2]}), 1), 3),
                            note=f"kept or added on {len(held)} of {len(vs)} reviewed builds that can take it"))
    return out


def _fam_ok(f: Optional[str], never_fams: Sequence[str]) -> bool:
    return f is None or f not in never_fams


>>>>>>> main
def authored_stats(rb: Rulebook, bm, r: PassRule) -> dict:
    """Corpus statistics of an authored rule, shown to its author (never a gate)."""
    if bm is None:
        return dict(rate=None, support=None, bound=None, certifies=None)
    p, S, n = rb.rule_stats(bm, r)
    lo, hi = wilson(p, n, rb.cfg.z)
    b = lo if r.effect == TAKE else hi
    ok = S >= rb.cfg.min_support and (lo >= rb.cfg.theta_take if r.effect == TAKE else hi <= rb.cfg.theta_never)
    return dict(rate=round(p, 4), support=round(S, 1), bound=round(b, 4), certifies=bool(ok))


def build_selection(rb: Rulebook, authored=None, vocab=None, reviews: Sequence = (), ledger=None,
                    family_of: Optional[Callable] = None, bm=None,
                    cfg: Optional[SelectionConfig] = None) -> Rulebook:
    """The effective rulebook (a copy of rb; rb is untouched)."""
    cfg = cfg or SelectionConfig()
    sel = copy.copy(rb)
    sel.mined = rb
    sel.evidence: Dict[str, dict] = {}
    sig = _cases(reviews, ledger, rb.c, rb.groups, family_of, cfg) if reviews else {}
    rules: List[PassRule] = []
    auth = list(authored.rules(vocab)) if (authored is not None and vocab is not None) else []
<<<<<<< HEAD
=======
    if cfg.review_rules and sig:                       # what the reviews say, as rules
        fo = family_of if family_of is not None else None
        for r in review_rules(sig, _views(reviews, fo, rb.c), rb.c, cfg):
            rid = rule_id(r.key)
            if rid in sup_ids:
                r.status = "suppressed"
            sel.evidence[r.key] = dict(status="review" if r.status == "active" else "suppressed", id=rid,
                                       support=[], counter=[])
            rules.append(r)
>>>>>>> main
    for e, r in auth:
        st = authored_stats(rb, bm, r)
        r.confidence = st["rate"] if st["rate"] is not None else (1.0 if r.effect == TAKE else 0.0)
        r.bound = st["bound"] if st["bound"] is not None else r.confidence
        r.support = st["support"] or 0.0
        ev = _evidence(r, sig.get(r.item, []))
        sel.evidence[r.key] = dict(status="authored", id=e.rule_id, by=e.author,
                                   support=[x.rid for x in ev if x.sign > 0],
                                   counter=[x.rid for x in ev if x.sign < 0], corpus=st)
        rules.append(r)
<<<<<<< HEAD
=======
    for r0 in rb.rules:
        r = copy.copy(r0)
        rid = rule_id(r.key)
        if rid in sup_ids:
            r.status = "suppressed"
            sel.evidence[r.key] = dict(status="suppressed", id=rid, by=authored.suppressed[rid].author,
                                       support=[], counter=[])
            rules.append(r); continue
        if r.status != "active":
            rules.append(r); continue
        ev = _evidence(r, sig.get(r.item, []))
        sup = [x for x in ev if x.sign > 0]
        cnt = [x for x in ev if x.sign < 0]
        info = dict(id=rid, support=[x.rid for x in sup], counter=[x.rid for x in cnt],
                    families=sorted({f for f in (_fam(x.ctx) for x in sup) if f}))
        if not cnt:
            info["status"] = "stands"
            if sup:
                r.note = (r.note + "; " if r.note else "") + f"reviews +{len(sup)}"
            sel.evidence[r.key] = info; rules.append(r); continue
        q = _refine(r, rb, bm, sup, cnt, cfg) if cfg.refine else None
        if q is not None:
            r.status = "refined"
            q.note = f"refined from {rid} by reviews {', '.join(sorted(set(info['counter'])))}"
            sel.evidence[r.key] = dict(info, status="refined", to=rule_id(q.key))
            sel.evidence[q.key] = dict(info, status="refined", id=rule_id(q.key), was=rid)
            rules += [r, q]; continue
        newest = max(ev, key=lambda x: (x.order, x.sign))
        if newest.sign < 0:
            r.status = "overruled"
            r.note = f"overruled by review {newest.rid}"
            sel.evidence[r.key] = dict(info, status="overruled", by=newest.rid)
        else:
            r.note = (r.note + "; " if r.note else "") + f"disputed +{len(sup)} −{len(cnt)}"
            sel.evidence[r.key] = dict(info, status="disputed")
        rules.append(r)
    from tri.rules.triviality import cull               # review and refined rules too
    for key, why in cull(rules, rb.c, rb.groups):
        sel.evidence.setdefault(key, {})["status"] = "trivial"
        sel.evidence[key]["why"] = why
    from tri.rules.triviality import prune
    was = {id(r): r.key for r in rules}
    rules, sel.pruned = prune(rules, rb.c)
    for key, _ in sel.pruned:
        sel.evidence.pop(key, None)
    for r in rules:                                     # a rewritten rule keeps its evidence
        if was[id(r)] != r.key and was[id(r)] in sel.evidence:
            sel.evidence[r.key] = dict(sel.evidence.pop(was[id(r)]), id=rule_id(r.key))
>>>>>>> main
    sel.rules = rules
    sel.pruned = []
    return sel


def selection_report(sel: Rulebook) -> dict:
    from collections import Counter
    ev = getattr(sel, "evidence", {}) or {}
    return dict(rules=len(sel.rules), active=len(sel.active()),
                by_status=dict(Counter(v.get("status") for v in ev.values())),
                authored=sum(1 for r in sel.rules if r.source == "authored"))

