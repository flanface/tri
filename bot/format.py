"""Plain-text review rendering used everywhere the bot shows operations:

    [+] Gale Trap (mantra)   weak    necessary for the archetype
    [-] Triathlete           strong  redundant with Moving Fortress
    [=] Last Laugh                   iffy

[+] add, [-] remove, [=] iffy (questionable, neither keep nor drop). Model
output also tags each add with the pass that asked for it: (foundational),
(central), (luxury) or (review). Lines are
grouped in that order and aligned inside a code block, so they read the same
on desktop and mobile. Pure functions, no Discord imports.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from .wire import REASON_LABELS

MARK = {"add": "[+]", "remove": "[-]", "iffy": "[=]", "overturn": "[x]", "uphold": "[v]"}
ORDER = {"add": 0, "remove": 1, "iffy": 2, "overturn": 3, "uphold": 3}
LEGEND = "[+] add   [-] remove   [=] iffy"
NAME_MAX = 28
PASS_TAGS = {"foundational": "(foundational)", "central": "(central)", "archetype": "(archetype)", "luxury": "(luxury)",
             "review": "(review)"}


def _name(item: str, kind: Optional[str]) -> str:
    return f"{item} (mantra)" if kind == "mantra" else item


def reason_text(o: dict) -> str:
    if o["action"] == "iffy":
        if o.get("reason"):                       # = x SG > y: a sidegrade (weak: this archetype, strong: generally)
            sc = " (generally)" if o.get("strength") == "strong" else " (this archetype)" if o.get("strength") else ""
            return f"iffy: {REASON_LABELS.get(o['reason'], o['reason'])} {', '.join(o.get('related') or [])}{sc}"
        return "iffy"
    if o["action"] in ("overturn", "uphold"):
        t = o.get("target") or {}
        n = f"#{t['n']} " if t.get("n") else ""
        inner = reason_text(dict(action=t.get("action", "?"), reason=t.get("reason"), related=t.get("related")))
        note = f' "{o["note"]}"' if o.get("note") else ""
        return f"{n}{MARK.get(t.get('action'), '[?]')} {inner}".rstrip() + note
    r = o.get("reason")
    lab = REASON_LABELS.get(r, r or "")
    if r == "needs" and o.get("related"):         # - x NW > a, b
        return f"does not work without {', '.join(o['related'])} (= adding them, given room)" + _pass_tag(o)
    if o.get("related"):
        lab += " " + ", ".join(o["related"])
    lab += _pass_tag(o, lab)
    if o.get("note") and o["action"] in ("add", "remove"):
        lab += f'  // {o["note"]}'
    return lab


def _pass_tag(o: dict, label: str = "") -> str:
    ps = o.get("pass_")
    if ps not in PASS_TAGS or ps in label:                   # "generally good (foundational talent)" says it already
        return ""
    return " " + PASS_TAGS[ps]                               # model output: which pass asked for it


def short_rid(rid) -> str:
    """A review id for display: 'pool-72' -> '#72', a direct review 'pool--3' -> 'direct #3'."""
    r = str(rid or "")
    if r.startswith("pool--"):
        return f"direct #{r[6:]}"
    if r.startswith("pool-"):
        return f"#{r[5:]}"
    return r


def source_label(c: dict) -> str:
    """Where a model change came from, for the change lists. No id is shown
    when the change has none (a slot freed for a review-driven pick)."""
    src, rid = c.get("source"), c.get("review_id")
    rv = f"review {short_rid(rid)}" if rid else "review"
    return {"rules": "rule", "review_rule": "review rule", "review": rv,
            "general": f"generally good, {rv}" if rid else "generally good",
            "worth": "worth", "category": "category", "indicator": f"indicative, {rv}" if rid else "indicative",
            "value": "value"}.get(src, src or "")


def op_lines(ops: Iterable[dict], kind_of: Callable[[str], Optional[str]] = lambda _: None,
             numbered: bool = False) -> List[str]:
    """One aligned line per operation, ordered + then - then =.
    numbered=True prefixes the draft position (as used by /undo order)."""
    rows = [(i, o) for i, o in enumerate(ops, 1)]
    rows.sort(key=lambda r: (ORDER.get(r[1]["action"], 9), r[0]))
    names = [_name(o["item_id"], kind_of(o["item_id"])) for _, o in rows]
    w = min(max((len(n) for n in names), default=0), NAME_MAX)
    has_strength = any(o["action"] != "iffy" and o.get("strength") for _, o in rows)
    out = []
    for (i, o), n in zip(rows, names):
        if len(n) > w:
            n = n[: w - 1] + "~"
        strength = "" if o["action"] == "iffy" and not o.get("reason") else (o.get("strength") or "")
        pre = f"{i:>2}. " if numbered else ""
        col = f"{strength:<6}  " if has_strength else ""
        src = f"  (review {short_rid(o['review_id'])})" if o.get("source") == "review" and o.get("review_id") else ""
        out.append(f"{pre}{MARK.get(o['action'], '[?]')} {n:<{w}}  {col}{reason_text(o)}{src}".rstrip())
    return out


def ops_block(ops: List[dict], kind_of: Callable[[str], Optional[str]] = lambda _: None,
              numbered: bool = False, legend: bool = True, limit: int = 1900) -> str:
    """A code block of op_lines, or the endorsement line when there are none."""
    if not ops:
        return "```\n[=] no changes: endorsed as is\n```"
    groups = grouped(list(enumerate(ops, 1)), lambda r: r[1]["action"], lambda r: r[1]["item_id"], kind_of)
    if len(groups) == 1:
        lines = op_lines(ops, kind_of, numbered)
    else:                                           # one heading per group: mantra additions, ...
        lines = []
        for title, part in groups:
            ls = op_lines([o for _, o in part], numbered=False)
            if numbered:
                ls = [f"{i:>2}. {ln}" for (i, _), ln in zip(sorted(part, key=lambda r: r[0]), ls)]
            lines += [f"-- {title.lower()}"] + ls
    if legend:
        lines = lines + ["", LEGEND]
    body = "\n".join(lines)
    if len(body) + 8 > limit:                       # keep whole lines
        keep, n = [], 0
        for ln in lines:
            if n + len(ln) + 1 > limit - 40:
                break
            keep.append(ln)
            n += len(ln) + 1
        body = "\n".join(keep) + f"\n... {len(lines) - len(keep)} more"
    return f"```\n{body}\n```"


def notes_block(notes: List[dict]) -> str:
    """Numbered note lines (# archetype / # weapons / // comment), or ''."""
    from .reviews import describe_note
    if not notes:
        return ""
    return "Notes:\n" + "\n".join(f"{i}. {describe_note(n)}" for i, n in enumerate(notes, 1))


def single(o: dict, kind_of: Callable[[str], Optional[str]] = lambda _: None) -> str:
    """One operation inline, e.g. for 'Added: ...' confirmations."""
    return f"`{op_lines([o], kind_of)[0]}`"


def alternative_ops(al: dict) -> List[dict]:
    """An analysis alternative (ModelService.analyze) as operation dicts, so the
    model's suggestions render exactly like a reviewer's."""
    ops = [dict(action="add", item_id=t["item"], strength="", reason=t.get("reason"),
                related=list(t.get("related") or []), source=t.get("source"), review_id=t.get("review_id"), pass_=t.get("pass_"))
           for t in al.get("takes", [])]
    ops += [dict(action="remove", item_id=d["item"], strength="", reason=d.get("reason"),
                 related=list(d.get("related") or []), source=d.get("source"), review_id=d.get("review_id"),
                 pass_=d.get("pass_") if d.get("pass_") == "review" else None)
            for d in al.get("drops", [])]
    why = al.get("iffy_why") or {}
    ops += [dict(action="iffy", item_id=x, strength="", reason=(why.get(x) or {}).get("reason"),
                 related=list((why.get(x) or {}).get("related") or [])) for x in al.get("iffy", [])]
    return ops


VIOLATION_TEXT = {"missing": "required, no room", "forbidden": "kept against an exclusion",
                  "conflict": "rules disagree", "unanchored": "applies through a pick rated off-archetype",
                  "talent_cap": "over the talent cap (all talents pinned or required)",
                  "mantra_slots": "over the mantra slots (all mantras pinned or required)"}


def violation_lines(vs: Iterable[dict]) -> List[str]:
    """[!] lines for rules an answer could not satisfy (Alternative.violations)."""
    vs = list(vs)
    w = min(max((len(v["item"]) for v in vs), default=0), NAME_MAX)
    return [f"[!] {v['item'][:w]:<{w}}  {VIOLATION_TEXT.get(v['kind'], v['kind'])}" for v in vs]


def kinds_from(svc) -> Callable[[str], Optional[str]]:
    return lambda item: svc.kind(item) if svc is not None else None


def review_summary(body: dict, kind_of: Callable[[str], Optional[str]] = lambda _: None) -> Dict[str, str]:
    """Title / link / block for a stored review object (reviews.operations)."""
    ops = [dict(action=o["action"], item_id=o["item_id"], strength=o.get("strength"),
                reason=o.get("reason"), related=o.get("related") or [], note=o.get("note"))
           for o in body.get("operations", []) if o.get("action") in ("add", "remove", "iffy")]
    block = ops_block(ops, kind_of, legend=bool(ops), limit=4000)
    nb = notes_block(body.get("notes") or [])
    if nb:
        block = block + "\n" + nb
    return dict(title=body.get("build_name") or body.get("original", {}).get("build_id") or "build",
                link=body.get("link") or "", reviewer=str(body.get("reviewer_id", "")),
                block=block)


# ------------------------------------------------------------------ grouped fields
#: Sheet groups, in display order: (action, kind, title). Kind None = any kind.
GROUPS = (("add", "mantra", "Mantra additions"), ("add", "talent", "Talent additions"),
          ("remove", "mantra", "Mantra removals"), ("remove", "talent", "Talent removals"),
          ("iffy", None, "Iffy picks"))
FIELD_MAX = 1024          # Discord: characters per embed field value
EMBED_MAX = 5800          # Discord: 6000 characters per embed, with room for title and footer


def group_title(action: str, kind: Optional[str]) -> Optional[str]:
    kind = "mantra" if kind == "mantra" else "talent"
    for a, k, t in GROUPS:
        if a == action and k in (None, kind):
            return t
    return None


def grouped(rows: Iterable, action_of: Callable, item_of: Callable,
            kind_of: Callable[[str], Optional[str]]) -> List[tuple]:
    """[(title, [row, ...])] in GROUPS order, empty groups left out."""
    by: Dict[str, list] = {}
    for r in rows:
        t = group_title(action_of(r), kind_of(item_of(r)))
        by.setdefault(t or "Other", []).append(r)
    order = [t for _, _, t in GROUPS] + ["Other"]
    return [(t, by[t]) for t in order if t in by]


def chunk_fields(title: str, lines: List[str], limit: int = FIELD_MAX) -> List[tuple]:
    """(title, code block) fields of whole lines, split across '(cont.)' fields."""
    out, cur, n = [], [], 0
    for ln in lines:
        ln = ln[: limit - 10]
        if cur and n + len(ln) + 1 > limit - 8:
            out.append(cur); cur, n = [], 0
        cur.append(ln); n += len(ln) + 1
    if cur:
        out.append(cur)
    return [(title if i == 0 else f"{title} (cont.)", "```\n" + "\n".join(c) + "\n```") for i, c in enumerate(out)]


def op_fields(ops: List[dict], kind_of: Callable[[str], Optional[str]] = lambda _: None,
              suffix: str = "") -> List[tuple]:
    """Operations as one embed field per group (mantra additions, talent additions,
    mantra removals, talent removals, iffy). The group says the kind, so names
    carry no '(mantra)' tag."""
    out = []
    for title, part in grouped(ops, lambda o: o["action"], lambda o: o["item_id"], kind_of):
        out += chunk_fields(title + suffix, op_lines(part))
    return out


def item_fields(items: List[tuple]) -> List[tuple]:
    """The build's own picks, numbered for fast reply reviews: items = [(n, item,
    kind)] -> 'Mantras' and 'Talents' fields."""
    out = []
    for kind, title in (("mantra", "Mantras"), ("talent", "Talents")):
        part = [(n, it) for n, it, k in items if (k == "mantra") == (kind == "mantra")]
        if part:
            out += chunk_fields(f"{title} ({len(part)})", [f"{n:>2}. {it}" for n, it in part])
    return out


def fit_fields(fields: List[tuple], used: int = 0, max_fields: int = 25, budget: int = EMBED_MAX) -> List[tuple]:
    """Keep whole fields within Discord's embed limits; the last kept field says
    how many were left out."""
    out, n = [], used
    for i, (name, value) in enumerate(fields):
        cost = len(name) + len(value)
        if len(out) >= max_fields - 1 and i < len(fields) - 1 or n + cost > budget - 60:
            left = len(fields) - i
            out.append(("More", f"{left} more section{'s' if left != 1 else ''} not shown (Discord's size limit)."))
            return out
        out.append((name, value)); n += cost
    return out
