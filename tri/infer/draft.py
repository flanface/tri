"""tri's analysis as a review sheet in the reviewers' own grammar (v2.20).

Tri.as_review(build) turns the best Alternative of Tri.review into a Review
(scope full: every unmarked pick is endorsed) and the reply lines a reviewer
would type under the bot's sheet:

    - 4,11 NS            removes by sheet number (mantras, then talents, alphabetical)
    - 7 w RW > Item      a weak remove with its partner
    + Item, Item NA      adds
    = 3 > Item           iffy (sidegrade to Item)
    # Gank: A, B         components that indicate the archetype
    # core: A, B         confirmed central picks
    # is Gank            the archetype reading, when located

Strength and emission come from the evidence margin of each change (docs/math
11.7, record.margin_of): a change is emitted only when its margin reaches
InferConfig.emit_margin, and is strong when the margin reaches
InferConfig.strong_margin. A change without a margin is emitted, strong.
A change that is not emitted never appears, and the sheet keeps the build's
counts: its pick is kept (or not taken) and the review solved again (v3.1.2).
Withholding never breaks the protocol: the sheet is checked with
codec.validate_review; iffy marks are written only with
InferConfig.emit_iffy (reviewers stay silent on near-ties); adds that no longer fit (talent cap 76 - 2m and
mantra slots, which are interdependent) are withheld lowest margin first."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from tri.core.codec import validate_review
from tri.core.types import Action, Build, Operation, Reason, Review, ReviewError, Strength
from tri.infer.defs import DraftSheet

CODE = {"less_necessary_space": "LN", "necessary_for_archetype": "NA", "redundant_with": "RW",
        "works_against": "WA", "no_synergy": "NS", "scenario_mismatch": "SM", "broken": "BR",
        "generally_good": "GG", "synergy_with": "SY", "needs": "NW", "luxury_less": "LL",
        "luxury_more": "LM", "sidegrade": "SG"}
RELATIONAL = {"redundant_with", "works_against", "synergy_with", "needs", "sidegrade"}


def family(name: Optional[str]) -> Optional[str]:
    """'Gank (Ardour / ...)' or 'Gank / Frost Mage' -> 'Gank'."""
    if not name:
        return None
    return str(name).split(" (")[0].split(" / ")[0].strip() or None


def change_margin(c) -> Optional[float]:
    """Evidence margin of one Change in [0, 1] (record.margin_of on its fields)."""
    from tri.sources.record import margin_of
    return margin_of(dict(why=c.why, marginal=c.marginal if c.source == "worth" else None))


def numbered(b: Build) -> Dict[str, int]:
    """Sheet numbers: mantras, then talents, each alphabetical (bot Draft.numbered_items)."""
    order = sorted(b.mantras, key=str.casefold) + sorted(b.talents, key=str.casefold)
    return {it: i for i, it in enumerate(order, 1)}


def _ranges(ns: Sequence[int]) -> str:
    ns = sorted(set(ns))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if i == j else f"{ns[i]}-{ns[j]}")
        i = j + 1
    return ",".join(out)


def _op(c, strength: Strength) -> Operation:
    r = c.reason or ("necessary_for_archetype" if c.action == "add" else "no_synergy")
    if r not in Reason._value2member_map_:
        r = "necessary_for_archetype" if c.action == "add" else "no_synergy"
    rel = [x for x in (c.related or []) if x != c.item]
    if r in RELATIONAL and not rel:
        r = "necessary_for_archetype" if c.action == "add" else "no_synergy"
    return Operation(Action(c.action), c.item, strength, Reason(r), rel if r in RELATIONAL else [])


def render(b: Build, ops: Sequence[Operation], notes: Sequence[dict]) -> List[str]:
    """Reply lines for ops and notes (bot/replies.py grammar)."""
    num = numbered(b)
    lines: List[str] = []
    groups: Dict[Tuple, List[int]] = {}
    for o in ops:
        if o.action == Action.REMOVE:
            key = (o.strength.value, o.reason.value if o.reason else "", tuple(o.related))
            groups.setdefault(key, []).append(num[o.item_id])
    for (s, r, rel), ns in sorted(groups.items(), key=lambda kv: min(kv[1])):
        lines.append(" ".join(x for x in ("-", _ranges(ns), "w" if s == "weak" else "", CODE.get(r, r),
                                          ("> " + ", ".join(rel)) if rel else "") if x))
    adds: Dict[Tuple, List[str]] = {}
    for o in ops:
        if o.action == Action.ADD:
            key = (o.strength.value, o.reason.value if o.reason else "", tuple(o.related))
            adds.setdefault(key, []).append(o.item_id)
    for (s, r, rel), its in adds.items():
        lines.append(" ".join(x for x in ("+", ", ".join(its), "w" if s == "weak" else "", CODE.get(r, r),
                                          ("> " + ", ".join(rel)) if rel else "") if x))
    for o in ops:
        if o.action == Action.IFFY:
            rel = (" w > " + ", ".join(o.related)) if o.related else ""
            lines.append(f"= {num[o.item_id]}{rel}")
    for n in notes:
        k = n.get("kind")
        if k == "archetype":
            lines.append(f"# {n['archetype']}: " + ", ".join(n["items"]))
        elif k in ("core", "avoid"):
            lines.append(f"# {k}: " + ", ".join(n["items"]))
        elif k in ("is", "not"):
            lines.append(f"# {k} {n['archetype']}")
    return lines


def as_review(tri, build: Build, archetype: Optional[str] = None, emit_margin: Optional[float] = None,
              strong_margin: Optional[float] = None, review_id: str = "tri-draft", answers=None,
              caution: bool = False) -> DraftSheet:
    """tri's review of a full build as a reviewer's sheet (module docstring)."""
    icfg = tri.cfg.infer
    emit = float(icfg.emit_margin if emit_margin is None else emit_margin)
    strong = float(icfg.strong_margin if strong_margin is None else strong_margin)
    have = build.item_set()
    # v3.1.2: a mark that is not emitted (held for thin evidence, or below emit_margin)
    # never appears, and the sheet still keeps the build's counts: the pick it would
    # have removed is kept, the pick it would have added is not taken, and the review
    # is solved again (at most `rounds` times; after that the marks stand)
    keep, ban = set(), set()
    withheld: List[dict] = []
    quiet: set = set()
    for _ in range(int(getattr(icfg, "draft_rounds", 20))):
        alts = tri.review(build, max_alternatives=1, archetype=archetype, keep_size=True, answers=answers,
                          locked=tuple(keep), withheld=tuple(ban))
        if not alts:
            break
        held = {(h["item"], h["action"]) for h in tri.caution(build, alts, answers, archetype).held} if caution else set()
        new = [c for c in alts[0].changes if c.action in ("add", "remove") and (
            (c.item, c.action) in held or (change_margin(c) is not None and change_margin(c) < emit))
            and ((c.action == "remove") == (c.item in have))]
        quiet = {(c.item, c.action) for c in new}
        if not new:
            break
        withheld += [dict(item=c.item, action=c.action, margin=change_margin(c),
                          why="insufficient information" if (c.item, c.action) in held else "below emit_margin")
                     for c in new]
        keep |= {c.item for c in new if c.action == "remove"}
        ban |= {c.item for c in new if c.action == "add"}
    if not alts:
        return DraftSheet(Review(review_id, build, [], "tri"), [], None)
    al = alts[0]
    concept = tri.concept(build)
    reading = concept.get("archetype") or {}
    fam = family(reading.get("archetype") if reading.get("located") else None) or family(al.name)
    margins: Dict[str, Optional[float]] = {}
    ops: List[Operation] = []
    for c in al.changes:
        if c.action not in ("add", "remove"):
            continue
        if c.action == "remove" and c.item not in have or c.action == "add" and c.item in have:
            continue
        m = change_margin(c)
        margins[c.item] = m
        if (c.item, c.action) in quiet:                  # rounds ran out: the old behaviour, withheld
            continue
        ops.append(_op(c, Strength.STRONG if m is None or m >= strong else Strength.WEAK))
    removed = {o.item_id for o in ops if o.action == Action.REMOVE}
    for it in (al.iffy if icfg.emit_iffy else []):
        if it in have and it not in removed:
            w = (al.iffy_why or {}).get(it) or {}
            rel = [x for x in (w.get("related") or []) if x != it]
            sg = w.get("reason") == "sidegrade" and rel
            ops.append(Operation(Action.IFFY, it, Strength.WEAK, Reason.SIDEGRADE if sg else None, rel if sg else []))
    # notes: indicators of the reading's family, confirmed central picks, the reading itself
    notes: List[dict] = []
    final = set(al.items())
    ind = [k for k, v in (al.indicative or {}).items() if k in final and family(v.get("archetype")) == fam]
    if fam and ind:
        notes.append(dict(kind="archetype", archetype=fam, items=sorted(ind), text="tri"))
    core = sorted(k for k, p in (al.passes or {}).items() if p == "central" and k in have and k not in ind)
    if core:
        notes.append(dict(kind="core", items=core, text="tri"))
    if fam and reading.get("located"):
        notes.append(dict(kind="is", archetype=fam, text="tri"))
    warnings: List[str] = []
    r = Review(review_id, build, ops, "tri", scope="full", notes=notes)
    for _ in range(len(ops) + 1):
        try:
            warnings = validate_review(r, tri.catalog)
            break
        except ReviewError as e:
            adds = [o for o in r.operations if o.action == Action.ADD]
            if not adds:
                raise
            worst = min(adds, key=lambda o: (margins.get(o.item_id) is None, margins.get(o.item_id) or 0.0))
            r.operations.remove(worst)
            withheld.append(dict(item=worst.item_id, action="add", margin=margins.get(worst.item_id),
                                 why=f"does not fit: {e}"))
    return DraftSheet(r, render(build, r.operations, notes), fam, margins, withheld, list(warnings))
