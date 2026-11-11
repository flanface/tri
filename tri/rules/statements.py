"""Rules and facts as two kinds of nontrivial statement (v2.21, math §12.6).

  rules   how the model generates completions and answers: the hard rules
          (cap, slots, capacity, requirements, locks, weapons, benchmarks,
          review authority) and the mined pass and review rules.
  facts   what the concept space and the meta look like (tri.facts).

Both are symbolic. Statements a baseline makes trivial (tri.triviality) are
dropped. lookup(...) gathers what tri knows about one talent or mantra: its
pocamind/data entry, its worth per archetype, the rules that touch it and
useful counts (pick rate, hole/gem status).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

import numpy as np

from tri.data import path as _data_path

RULE_SYMBOLS = ("T talents · M mantras · m cap-lowering mantras · cap = 76 − 2m · ⇒ then · ∧ and · ∨ or · "
                "¬ not · ⊥ works against · ⊕ exclusive · pre(x) prerequisites · req_s(x) requirement · "
                "σ logistic · κ gain · w worth · [F] foundational [C] central [X] exclusion [R] review")
FACT_SYMBOLS = ("A, B archetypes · P(x | A) share of A taking x · w(x) worth · Δw change in worth · "
                "↔ differ · d² squared distance · ⇒ raises/lowers · hole: high P, reviews remove · "
                "gem: low P, reviews add · p fluke chance")

HARD = (
    ("cap", "|T| ≤ 76 − 2m,  m = |{x ∈ M : type(x) ∉ {monster, oath, faction}}|"),
    ("slots", "M fits slots(B) = 3 combat + 1 mobility + 1 support + 1 wildcard (+1 wildcard Neuroplasticity, "
              "+1 wisp Will o' Wisp, + oath);  type(x) = combat ⇒ x ∉ support slot;  unknown type ⇒ wildcard"),
    ("capacity", "|T| < cap ⇒ add best until |T| = cap, before replacements;  |T| > cap ⇒ remove |T| − cap worst "
                 "first;  free slots ⇒ fill;  keep_size ⇒ ¬fill"),
    ("order", "complete M before T  (¬Silentheart)"),
    ("requirements", "x available ⇔ req(x) holds at pre-shrine ∨ final  (¬highest, ¬post-shrine)"),
    ("locks", "x locked by oath ∨ murmur ∨ origin ∨ wield ⇒ x ∉ B unless B has the token"),
    ("weapons", "clause(x) holds ⇔ ∃ weapon ∈ B satisfying it;  redundancy: w(x) × share of weapons without it"),
    ("benchmarks", "w(x): logit += κ σ((max(pre_s, fin_s) − k) / width)"),
    ("authority", "review ≻ analysis ≻ pretrain;  newer review ≻ older;  scope = the reviewed build's region"),
)


def model_rules(tri, item: Optional[str] = None, limit: int = 25, hard: bool = True) -> dict:
    """{rules: [{kind, symbol}], trivial: n}: the hard rules, then active mined
    pass rules and review rules (item filters to those naming it)."""
    from tri.game import priors, weapons as W
    from tri.rules.triviality import trivial_rule
    out, dropped = [], 0
    if hard and item is None:
        out += [dict(kind=k, symbol=s) for k, s in HARD]
    if item is not None:
        for cl in tri.catalog.enforced_clauses(item) if item in tri.catalog.items else []:
            out.append(dict(kind="lock", symbol=f"{item} ⇒ " + " ∨ ".join(W.describe(t) for t in cl)))
        bm = priors.benchmark_of(item)
        if bm:
            B = priors.benchmarks()
            out.append(dict(kind="benchmark", symbol=f"w({item}): logit += {B.get('gain')} σ((max(pre, fin)_"
                                                      f"{bm['stat']} − {bm.get('at', 77)}) / {B.get('width')})"))
        tok = W.table()["scale"].get(W._loose(item))
        if tok:
            out.append(dict(kind="weapons", symbol=f"w({item}) × share of equipped weapons with {tok}"))
    # v3.0.3: the selection (mined + authored + review rules, suppressions
    # applied), not the mined rulebook alone, so database rules show here
    rb = getattr(tri, "rules", None) if getattr(tri, "selection", None) is not None else None
    rb = rb if rb is not None else getattr(tri, "rulebook", None)
    names = list(getattr(rb, "region_names", []) or []) if rb is not None else []
    act = sorted(rb.active() if rb is not None else [], key=lambda r: r.source != "authored")
    for r in act:
        if item is not None and r.item != item and not any(c.key == item for c in r.conds):
            continue
        if r.source != "authored" and trivial_rule(r, tri.catalog, getattr(rb, "groups", None)):
            dropped += 1
            continue
        sym = r.text(names)
        if any(c.kind == "group" for c in r.conds) and getattr(rb, "groups", None):
            from tri.rules.symbolic import cond as _sym, conj
            from tri.rules.triviality import expand
            ex = expand(r.conds, r.item, tri.catalog, rb.groups)
            if [c.id() for c in ex] != [c.id() for c in r.conds]:       # v3.0.5: what the group literal means
                sym += "   ≡ " + conj(_sym(c) for c in ex)
        out.append(dict(kind="authored" if r.source == "authored" else r.pass_, symbol=sym))
    for r in getattr(tri, "review_rules", []) or []:
        if r.status == "proposed" or item is not None and item not in (r.subject, r.other):
            continue
        out.append(dict(kind="review", symbol=r.describe()))
    culled = sum(1 for r in (rb.rules if rb is not None else []) if r.status == "trivial" and
                 (item is None or r.item == item or any(c.key == item for c in r.conds)))
    keep = out[:limit] if limit else out
    return dict(rules=keep, total=len(out), trivial=dropped + culled, culled=culled, symbols=RULE_SYMBOLS)


@lru_cache(maxsize=1)
def entries() -> dict:
    p = _data_path("entries.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return dict(talents={}, mantras={})


def entry(item: str, kind: Optional[str] = None) -> Optional[dict]:
    """The pocamind/data entry of a talent or mantra (case-insensitive)."""
    E = entries()
    for k in ([kind + "s"] if kind else ["talents", "mantras"]):
        d = E.get(k, {})
        if item in d:
            return dict(d[item])
        low = {x.lower(): x for x in d}
        if item.lower() in low:
            return dict(d[low[item.lower()]])
    return None


def lookup(tri, item: str, at=None, miner=None, facts=None) -> dict:
    """What tri knows about one item (see the module docstring)."""
    from tri.core.types import Kind
    c = tri.catalog
    it = c.items.get(item)
    kind = None if it is None else ("mantra" if it.kind == Kind.MANTRA else "talent")
    out = dict(item=item, kind=kind, entry=entry(item, kind), requires=dict(it.requires) if it else {},
               prerequisites=list(it.prerequisites) if it else [], exclusive=list(it.exclusive_with) if it else [])
    if it is not None and it.kind == Kind.MANTRA:
        out.update(slot=c.mantra_slot(item) or "wildcard", lowers_cap=bool(c.lowers_cap(item)))
    rows: List[dict] = []
    if at is not None and miner is not None and item in c.index:
        t = c.index[item]
        for a in range(len(at.names)):
            if at.count[a] == 0 or np.isnan(miner.M[a, t]):
                continue
            rows.append(dict(archetype=at.names[a], worth=round(float(miner.M[a, t]), 3),
                             pick=round(float(np.nan_to_num(miner.R[a, t])), 3),
                             available=round(float(miner.Av[a, t]), 3), builds=int(at.count[a])))
        rows.sort(key=lambda r: -r["worth"])
        av = at.avail[:, t]
        out["pick_rate"] = round(float((at.X[:, t] & av).sum() / max(av.sum(), 1)), 3)
    out["worth"] = rows
    out["rules"] = model_rules(tri, item, limit=12, hard=False)["rules"]
    fs = [f for f in (facts or []) if item in (f.items or [])]
    out["holes"] = [f.archetype for f in fs if f.family in ("hole", "trap")]
    out["gems"] = [f.archetype for f in fs if f.family == "gem"]
    out["facts"] = [dict(family=f.family, symbol=f.symbol, status=f.status) for f in fs[:6]]
    return out
