"""The real review record: frozen datasets and sheet scoring (v2.20).

Everything here reads reviews as they are stored (Supabase rows
written by the bot, see bot.ingest). Nothing here changes an answer. The model
enters only through two optional callables:

  reading(build) -> archetype name or None     (Tri.concept, or reading_by_overlap)
  rate(build, item) -> local pick rate in [0, 1] (Tri.worth)

Math sections: 11.6 (the reviewer-agreement scorecard), 11.4 (evidence
margins), 12.6 (insight reports). v3.5: audits and their labels are gone."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tri.core.codec import review_from_dict
from tri.core.types import Action, Build, Reason, Review
from tri.sources.defs import Mark, Record

KEEP, ADD, REMOVE, IFFY = "keep", "add", "remove", "iffy"
LABELS = (KEEP, ADD, REMOVE, IFFY)
PARTNER_REASONS = (Reason.REDUNDANT_WITH.value, Reason.WORKS_AGAINST.value, Reason.SYNERGY_WITH.value,
                   Reason.NEEDS.value)
CODE = {"less_necessary_space": "LN", "necessary_for_archetype": "NA", "redundant_with": "RW",
        "works_against": "WA", "no_synergy": "NS", "scenario_mismatch": "SM", "broken": "BR",
        "generally_good": "GG", "synergy_with": "SY", "needs": "NW", "luxury_less": "LL",
        "luxury_more": "LM", "sidegrade": "SG"}
# ====================================================================== the frozen dataset


def _body(row: dict) -> dict:
    b = row.get("operations")
    if isinstance(b, str):
        b = json.loads(b)
    return dict(b or {})


def freeze(export: dict, outdir: str, stamp: Optional[str] = None) -> str:
    """Write an export (table name -> rows, as read by bot.db) to
    outdir/record_<stamp>.json with a manifest (row counts, sha256). Returns the path.
    The file is never rewritten: a second freeze on the same stamp gets a suffix."""
    os.makedirs(outdir, exist_ok=True)
    stamp = stamp or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    data = json.dumps(export, sort_keys=True, default=str).encode()
    man = dict(stamp=stamp, sha256=hashlib.sha256(data).hexdigest(),
               counts={k: len(v) for k, v in export.items() if isinstance(v, list)})
    path = os.path.join(outdir, f"record_{stamp}.json")
    k = 1
    while os.path.exists(path):
        path = os.path.join(outdir, f"record_{stamp}_{k}.json"); k += 1
    with open(path, "w") as f:
        json.dump(dict(manifest=man, tables=export), f, default=str)
    return path


def load(path_or_export) -> Record:
    """A Record from a frozen file (freeze) or an in-memory export."""
    if isinstance(path_or_export, str):
        with open(path_or_export) as f:
            d = json.load(f)
    else:
        d = dict(path_or_export)
    tables = d.get("tables", d)
    man = d.get("manifest", {})
    revs, meta, bad = [], {}, []
    for row in tables.get("reviews") or []:          # v3.5: an "audits" table in old freezes is ignored
        try:
            body = _body(row)
            r = review_from_dict(body)
            meta[r.review_id] = dict(pool_id=body.get("pool_id") or row.get("pool_id"), link=body.get("link"),
                                     build_name=body.get("build_name"), model_run=body.get("model_run"),
                                     row_id=row.get("id"))
            revs.append(r)
        except Exception as e:                          # noqa: BLE001 (report, never drop silently)
            bad.append(f"full {row.get('id')}: {e}")
    return Record(revs, list(tables.get("locators") or []), list(tables.get("item_values") or []),
                  meta, man, bad)
# ====================================================================== readings without a model


def reading_by_overlap(locators: Sequence[dict], k: int = 3) -> Callable[[Build], Optional[str]]:
    """A model-free archetype reading: the majority archetype of the k locator builds
    with the highest pick overlap (Jaccard). Used for counting only, when no fitted
    run is at hand; with a run use Tri.concept(b)["archetype"]."""
    refs = []
    for l in locators:
        rec = l.get("record") or {}
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except ValueError:
                continue
        items = set(rec.get("talents") or []) | set(rec.get("mantras") or [])
        if not items and isinstance(rec.get("build"), dict):
            items = set(rec["build"].get("talents") or []) | set(rec["build"].get("mantras") or [])
        if items:
            refs.append((str(l.get("archetype") or "").strip(), items))

    def read(b: Build) -> Optional[str]:
        s = set(b.items())
        if not refs or not s:
            return None
        sc = sorted(((len(s & it) / len(s | it), a) for a, it in refs), reverse=True)[:k]
        if not sc or sc[0][0] <= 0:
            return None
        votes = Counter()
        for j, a in sc:
            votes[a] += j
        return votes.most_common(1)[0][0]
    return read
# ====================================================================== census


def census(rec: Record, reading: Optional[Callable[[Build], Optional[str]]] = None, thin: int = 10) -> dict:
    """Counts of the full reviews by archetype reading, reviewer, reason, strength and
    note kind; archetypes with fewer than `thin` reviews."""
    reading = reading or reading_by_overlap(rec.locators)
    arch = Counter(); who = Counter(); why = Counter(); st = Counter(); notes = Counter(); act = Counter()
    for r in rec.reviews:
        arch[reading(r.original) or "unread"] += 1
        who[r.reviewer_id or "?"] += 1
        for o in r.operations:
            act[o.action.value] += 1
            why[CODE.get(o.reason.value, o.reason.value) if o.reason else "-"] += 1
            st[o.strength.value if o.strength else "-"] += 1
        for n in r.notes or []:
            notes[n.get("kind") or "?"] += 1
    names = sorted({str(l.get("archetype")) for l in rec.locators})
    return dict(reviews=len(rec.reviews), endorsed=sum(1 for r in rec.reviews if not r.operations),
                by_archetype=dict(arch.most_common()), by_reviewer=dict(who.most_common()),
                by_action=dict(act), by_reason=dict(why.most_common()), by_strength=dict(st),
                by_note=dict(notes),
                thin=[a for a in names + [a for a in arch if a not in names] if arch.get(a, 0) < thin],
                reviewers=len(who), problems=list(rec.problems))


def sheet(r: Review) -> Tuple[Dict[str, Mark], List[dict]]:
    """A review as item -> Mark (every original item: keep unless marked; every add)
    and its notes. The newest line for an item wins (Review.operations order)."""
    m: Dict[str, Mark] = {i: Mark(KEEP) for i in r.original.items()}
    for o in r.operations:
        m[o.item_id] = Mark(o.action.value, o.strength.value if o.strength else None,
                            o.reason.value if o.reason else None, tuple(sorted(o.related or ())))
    return m, [dict(n) for n in r.notes or []]


def _note_items(notes: Sequence[dict], kinds=("archetype", "core")) -> set:
    out = set()
    for n in notes:
        if n.get("kind") in kinds:
            out |= set(n.get("items") or [])
    return out


@dataclass
class Score:
    """Accumulated item-by-item agreement between predicted and true sheets."""
    conf: Counter = field(default_factory=Counter)        # (true label, predicted label)
    reason: Counter = field(default_factory=Counter)      # (true reason, predicted reason) on label matches
    strength: Counter = field(default_factory=Counter)    # (true, predicted) on label matches
    partner_ok: int = 0
    partner_n: int = 0
    note_hit: int = 0
    note_n: int = 0
    silent_n: int = 0                                     # items the reviewer left unmarked
    silent_changed: int = 0                               # ...that the prediction changed
    sheets: int = 0
    exact: int = 0                                        # sheets matching on every label

    def add(self, pred: Review, true: Review) -> None:
        P, pn = sheet(pred)
        T, tn = sheet(true)
        self.sheets += 1
        same = True
        for it in set(P) | set(T):
            t = T.get(it, Mark(KEEP)); p = P.get(it, Mark(KEEP))
            if it not in true.original.item_set() and it not in T:
                t = Mark(KEEP)                               # an item neither had: not on either sheet
                if p.label == KEEP:
                    continue
                t = Mark("absent")
            self.conf[(t.label, p.label)] += 1
            same &= t.label == p.label or (t.label == "absent" and p.label == KEEP)
            if t.label == KEEP:
                self.silent_n += 1
                self.silent_changed += p.label != KEEP
            if t.label == p.label and t.label not in (KEEP, "absent"):
                self.reason[(t.reason, p.reason)] += 1
                self.strength[(t.strength, p.strength)] += 1
                if t.reason in PARTNER_REASONS:
                    self.partner_n += 1
                    self.partner_ok += t.reason == p.reason and set(t.related) == set(p.related)
        tni, pni = _note_items(tn), _note_items(pn)
        self.note_n += len(tni); self.note_hit += len(tni & pni)
        self.exact += same

    def report(self) -> dict:
        f1 = {}
        for a in LABELS:
            tp = self.conf[(a, a)]
            pp = sum(v for (t, p), v in self.conf.items() if p == a)
            tt = sum(v for (t, p), v in self.conf.items() if t == a)
            pr = tp / pp if pp else None; rc = tp / tt if tt else None
            f1[a] = dict(precision=_r(pr), recall=_r(rc), f1=_r(2 * pr * rc / (pr + rc)) if pr and rc else
                         (0.0 if (pp or tt) else None), true=tt, predicted=pp)
        ok = lambda c: sum(v for (t, p), v in c.items() if t == p)
        n = lambda c: sum(c.values())
        return dict(sheets=self.sheets, exact_sheets=self.exact, f1=f1,
                    reason_accuracy=_r(ok(self.reason) / n(self.reason)) if n(self.reason) else None,
                    strength_accuracy=_r(ok(self.strength) / n(self.strength)) if n(self.strength) else None,
                    partner_accuracy=_r(self.partner_ok / self.partner_n) if self.partner_n else None,
                    note_recall=_r(self.note_hit / self.note_n) if self.note_n else None,
                    over_editing=_r(self.silent_changed / self.silent_n) if self.silent_n else None,
                    reason_confusion={f"{CODE.get(t, t)}>{CODE.get(p, p)}": v for (t, p), v in self.reason.most_common()},
                    label_confusion={f"{t}>{p}": v for (t, p), v in self.conf.most_common()})


def _r(x):
    return None if x is None else round(float(x), 4)


def score(pairs: Iterable[Tuple[Review, Review]]) -> dict:
    """The reviewer-agreement scorecard over (predicted, true) sheet pairs (math §11.3)."""
    s = Score()
    for p, t in pairs:
        s.add(p, t)
    return s.report()


def jaccard(a: Build, b: Build) -> float:
    x, y = set(a.items()), set(b.items())
    return len(x & y) / len(x | y) if x | y else 1.0


def ceiling(reviews: Sequence[Review], sim: float = 0.99,
            similarity: Optional[Callable[[Build, Build], float]] = None) -> dict:
    """Agreement between two reviews of the same build (or within the same-build band,
    similarity >= sim): the ceiling of the scorecard. Each such pair is scored both
    ways. similarity: the model's build similarity; pick overlap without a model.
    A ceiling only: it never weights anything."""
    sim_f = similarity or jaccard
    rs = list(reviews)
    pairs = []
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if sim_f(rs[i].original, rs[j].original) >= sim:
                pairs += [(rs[i], rs[j]), (rs[j], rs[i])]
    out = score(pairs) if pairs else dict(sheets=0)
    out["pairs"] = len(pairs) // 2
    return out


def time_split(reviews: Sequence[Review], train: float = 0.7) -> Tuple[List[Review], List[Review]]:
    """(train, held-out) by time: the newest (1 - train) share is held out."""
    rs = sorted(reviews, key=lambda r: r.timestamp or 0.0)
    k = int(round(train * len(rs)))
    return rs[:k], rs[k:]


def reviewer_split(reviews: Sequence[Review], held: Sequence[str]) -> Tuple[List[Review], List[Review]]:
    """(train, held-out) by reviewer."""
    h = set(held)
    return [r for r in reviews if r.reviewer_id not in h], [r for r in reviews if r.reviewer_id in h]


# ====================================================================== evidence margins
_P = re.compile(r"P\s*([0-9.]+)\s*≥\s*([0-9.]+)")
_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
_SIM = re.compile(r"similarity\s*([0-9.]+)")


def margin_of(a: dict) -> Optional[float]:
    """Evidence margin of one analysed change in [0, 1] (math §11.4): a rule's
    certified bound; a review transfer's similarity; a worth change's distance of
    the local pick rate from one half (|2q - 1|); `marginal` when recorded."""
    w = str(a.get("why") or a.get("rule") or "")
    if a.get("margin") is not None:
        return float(a["margin"])
    m = _P.search(w)
    if m:
        return float(m.group(2))
    m = _SIM.search(w)
    if m:
        return float(m.group(1))
    m = _PCT.search(w)
    if m:
        return abs(2 * float(m.group(1)) / 100 - 1)
    if a.get("marginal") is not None:
        return abs(2 * float(a["marginal"]) - 1)
    return None


# ====================================================================== insights (section K)


def reason_mix(rec: Record, reading: Optional[Callable[[Build], Optional[str]]] = None) -> Dict[str, Dict[str, int]]:
    """Why picks get removed, per archetype: reason code -> count of removes."""
    reading = reading or reading_by_overlap(rec.locators)
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in rec.reviews:
        a = reading(r.original) or "unread"
        for o in r.operations:
            if o.action == Action.REMOVE:
                out[a][CODE.get(o.reason.value, o.reason.value) if o.reason else "-"] += 1
    return {a: dict(c.most_common()) for a, c in sorted(out.items())}


def reviewers_vs_meta(rec: Record, rate: Callable[[Build, str], Optional[float]],
                      reading: Optional[Callable[[Build], Optional[str]]] = None,
                      hi: float = 0.6, lo: float = 0.3) -> Dict[str, dict]:
    """Per archetype: items reviewers remove despite a high local pick rate (>= hi) and
    add despite a low one (<= lo), with counts and the reviews behind each line."""
    reading = reading or reading_by_overlap(rec.locators)
    out: Dict[str, dict] = defaultdict(lambda: dict(removed_despite_high={}, added_despite_low={}))
    for r in rec.reviews:
        a = reading(r.original) or "unread"
        for o in r.operations:
            if o.action not in (Action.ADD, Action.REMOVE):
                continue
            q = rate(r.original, o.item_id)
            if q is None:
                continue
            key = "removed_despite_high" if o.action == Action.REMOVE and q >= hi else \
                  "added_despite_low" if o.action == Action.ADD and q <= lo else None
            if key:
                row = out[a][key].setdefault(o.item_id, dict(count=0, rates=[], reviews=[]))
                row["count"] += 1; row["rates"].append(round(float(q), 3)); row["reviews"].append(r.review_id)
    return dict(out)


def hard_cases(rec: Record, rate: Callable[[Build, str], Optional[float]], hi: float = 0.6, lo: float = 0.3) -> List[str]:
    """Reviews where the reviewer contradicts the corpus (a trap removed despite a
    high pick rate, or a gem added despite a low one): their ids."""
    out = []
    for r in rec.reviews:
        for o in r.operations:
            q = rate(r.original, o.item_id) if o.action in (Action.ADD, Action.REMOVE) else None
            if q is not None and ((o.action == Action.REMOVE and q >= hi) or (o.action == Action.ADD and q <= lo)):
                out.append(r.review_id); break
    return out


def provenance(rec: Record, review_ids: Iterable[str]) -> List[dict]:
    """Links behind a report line: review id, pool id, builder link, build name."""
    out = []
    for rid in review_ids:
        m = rec.meta.get(rid) or {}
        out.append(dict(review_id=rid, pool_id=m.get("pool_id"), link=m.get("link"), build=m.get("build_name")))
    return out
