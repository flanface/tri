"""Reviewer consensus per archetype family (math §8.5a, v2.19).

Single notes and review edits act locally (within reach of the reviewed build).
When several reviews say the same thing about an archetype ("Replenishing
Knockout and Agile Dash indicate Gank"), the agreement becomes a property of
the archetype: a tell. Tells act on every build read as that family, and only
there (the family reading p(a | u) is the gate, so other families are untouched).

Evidence e = (family a, item t, sign s, weight w, reviewer r, time tau):
  indicator / core note on t           s = +1, w = 1
  avoid note on t                      s = -1, w = 1
  add t, reason NA (build read as a)   s = +1, w = op_weight * strength
  remove t, reason NS / WA / BR / SM   s = -1, w = op_weight * strength
Per (a, t) the newest piece of evidence of each reviewer on each reviewed
build counts (a reviewer does not stack by repeating themselves on one build;
the same verdict on distinct builds is distinct evidence, v2.23: with one
reviewer the per-reviewer count promoted nothing). With n+ and n- the summed weights,
  net = n+ - n-,     c(a, t) = max(net, 0) / (max(net, 0) + kappa)
(an avoid tell uses max(-net, 0)). A tell is promoted once at least
min_reviewers distinct (reviewer, build) units back it and c > 0
(unit="reviewer" restores the pre-2.23 count of distinct reviewers)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from tri.reviews.defs import ConsensusConfig, Tell


def tally(evidence: Iterable[dict], cfg: Optional[ConsensusConfig] = None) -> Dict[str, List[Tell]]:
    """evidence: dicts key, item, sign, weight, reviewer, timestamp, review_id, note (bool).
    Returns family key -> promoted tells, strongest first."""
    cfg = cfg or ConsensusConfig()
    newest: Dict[Tuple[str, str, str], dict] = {}
    for e in sorted(evidence, key=lambda e: e.get("timestamp") or 0.0):
        if not e.get("key") or not e.get("item") or not e.get("sign"):
            continue
        who = str(e.get("reviewer") or e.get("review_id"))
        if cfg.unit == "build" and e.get("build"):
            who = f"{who}|{e['build']}"
        newest[(e["key"], e["item"], who)] = e
    acc: Dict[Tuple[str, str], List[dict]] = {}
    for (k, t, _r), e in newest.items():
        acc.setdefault((k, t), []).append(e)
    out: Dict[str, List[Tell]] = {}
    for (k, t), es in acc.items():
        pos = sum(float(e.get("weight", 1.0)) for e in es if e["sign"] > 0)
        neg = sum(float(e.get("weight", 1.0)) for e in es if e["sign"] < 0)
        net = pos - neg
        s = 1 if net > 0 else -1 if net < 0 else 0
        if s == 0:
            continue
        side = [e for e in es if (e["sign"] > 0) == (s > 0)]
        if len(side) < int(cfg.min_reviewers):
            continue
        c = abs(net) / (abs(net) + float(cfg.kappa))
        out.setdefault(k, []).append(Tell(k, t, s, c, pos if s > 0 else neg, neg if s > 0 else pos, len(side),
                                          sum(1 for e in side if e.get("note")),
                                          sorted({str(e.get("review_id")) for e in side})))
    for k in out:
        out[k].sort(key=lambda x: (-x.sign, -x.confidence, x.item))
    return out
