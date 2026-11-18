"""The selection rule set: the hand-authored rules in one rulebook that forces
foundational and central picks and exclusions (4.0: mined rules and rules read
off the reviews are gone; reviews act through the scorer).

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
    sel.rules = rules
    sel.pruned = []
    return sel


def selection_report(sel: Rulebook) -> dict:
    from collections import Counter
    ev = getattr(sel, "evidence", {}) or {}
    return dict(rules=len(sel.rules), active=len(sel.active()),
                by_status=dict(Counter(v.get("status") for v in ev.values())),
                authored=sum(1 for r in sel.rules if r.source == "authored"))

