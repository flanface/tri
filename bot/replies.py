"""Reviews written as replies to the bot's review sheet instead of commands.

After /review the bot posts a review sheet (a regular message). The reviewer
who claimed the build replies to it, or to any of the bot's answers to those
replies, with one operation per line:

    - 3,5-7 [s|w] code [> 2]     remove by number
    = 4 9  /  ~ 3                iffy / unmark
    + Item, Item [s|w] [code]    add
    - Item [s|w] code [> A, B]   remove by name
    draft  undo  clear  submit  skip

(v3.5: audits, x / v verdict lines, are gone.)

Numbers in a full review are the build's own picks as numbered on the sheet
(mantras first, then talents, each alphabetical; Draft.numbered_items).

Strength defaults to strong. Reason codes (case-insensitive; the full reason
name works too):

    LN  less necessary (make space)      NA  necessary for the archetype (adds; the default; NP still works)
    GG  generally good: a foundational talent, good for any build (adds of talents)
    RW  redundant with  (needs > item)   WA  works against   (needs > item)
    NS  no synergy with the build        SM  weak: does not fit the scenario;
                                             strong: does not belong on this archetype
    BR  does not work
    SY  synergises with (adds; > item; the default for an add with > item)
    NW  does not work without (removes; > the missing components; equivalent to
        adding them, given room)
    LL  worth less as a luxury (removes)    LM  worth more as a luxury (adds)
    SG  sidegrade to (iffy; > item; weak = under this archetype, strong = generally;
        the default for = with > item)

Notes start with # or // (also after an operation, as that operation's note):

    # Gank: Item, Item       the components indicate archetype Gank (also: # Item, Item > Gank)
    // free text             anything else useful to the model; kept with the review
    ~ #2  /  ~ #             drop note 2 / every note

A line for an item already marked replaces its operation (the latest line
wins). Lines are applied in order; a bad line is reported and skipped, and
`submit` is refused while any line in the message failed. Discord shows a
reply's text to a bot without the privileged Message Content intent only
when "ping on reply" is on.

Pure functions (no Discord), so the grammar is testable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .reviews import DraftError, ExclusionError, ReviewDesk, describe_op, parse_numbers

REASON_CODES = {
    "ln": "less_necessary_space",
    "na": "necessary_for_archetype",
    "rw": "redundant_with",
    "wa": "works_against",
    "ns": "no_synergy",
    "sm": "scenario_mismatch",
    "br": "broken",
    "gg": "generally_good",
    "sy": "synergy_with",
    "nw": "needs",
    "ll": "luxury_less",
    "lm": "luxury_more",
    "sg": "sidegrade",
}
CODE_OF = {v: k.upper() for k, v in REASON_CODES.items()}
ALIASES = {"np": "necessary_for_archetype", "syn": "synergy_with", "sw": "synergy_with",
           "side": "sidegrade", "sidegrade_to": "sidegrade", "without": "needs"}
WEAPON_HEADS = {"weapon", "weapons", "wpn", "wpns", "w", "equips", "uses"}      # NP: the code before v2.5 (necessary for purpose)
STRENGTHS = {"s": "strong", "strong": "strong", "w": "weak", "weak": "weak"}
_CLASS_WORDS = {"light", "medium", "heavy", "light weapon", "medium weapon", "heavy weapon",
                "dagger", "daggers", "sword", "swords", "spear", "spears", "fist", "fists", "gun", "guns",
                "rapier", "rapiers", "club", "clubs", "greataxe", "greatsword", "twinblade", "twinblades",
                "bow", "bows", "rifle", "rifles", "pistol", "pistols", "katana", "katanas"}
CONTROL = {"draft", "undo", "clear", "submit", "skip"}

_MENTION = re.compile(r"<@!?\d+>")
_NUMS = re.compile(r"^(\d+(-\d+)?)(,(\d+(-\d+)?))*,?$")

REPLY_HELP = (
    "Reply, one line per change. Numbers refer to the picks below.\n"
    "```\n"
    "- 3,5-7 [s|w] code [> 2]   remove (RW/WA need > item)\n"
    "= 4 9                      iffy\n"
    "~ 3                        unmark\n"
    "+ Item, Item [s|w] [code]  add (default s NA)\n"
    "+ Item > Other             add, synergy (SY)\n"
    "- 3 NW > Item              needs Item\n"
    "- 4 LL | + Item LM         luxury worth less | more\n"
    "= 4 [w|s] > Item           sidegrade\n"
    "# Gank: Item, Item         archetype indicators | // note\n"
    "# weapon: Name, Name       the build also uses these weapons\n"
    "- 3 NS > Weapon            only with that weapon (any reason but NW, SG)\n"
    "draft | undo | clear | submit | skip\n"
    "```\n"
    "LN less necessary, NA necessary, GG generally good, RW redundant, WA works against, NS no synergy, "
    "SM wrong scenario, BR broken, SY synergy, NW needs, LL/LM luxury worth, SG sidegrade. Unmarked = endorsed."
)
@dataclass
class Line:
    n: int                        # line number in the message (1-based)
    text: str
    kind: str                     # add | remove | iffy | unmark | control | note | unnote | error
    item: str = ""
    strength: Optional[str] = None
    reason: Optional[str] = None
    related: List[str] = field(default_factory=list)
    spec: str = ""                # item numbers on the sheet
    items: List[str] = field(default_factory=list)   # names (comma-separated on the line)
    note: Optional[str] = None
    control: str = ""
    error: str = ""
    note_index: Optional[int] = None   # unnote: the note number (None: every note)


_NOTE_RE = re.compile(r"\s(#|//)\s?(.*)$")


def split_note(text: str) -> Tuple[str, Optional[str]]:
    """"- 3 NS // why" -> ("- 3 NS", "why"). A note starts with # or // after a space."""
    m = _NOTE_RE.search(" " + text)
    if not m or m.start() == 0:
        return text, None
    return text[: m.start() - 1].rstrip(), (m.group(2).strip() or None)


IS_HEADS = {"is", "=", "archetype", "arch", "type"}
NOT_HEADS = {"not", "!=", "isnt", "isn't", "notarch"}
CORE_HEADS = {"core", "key", "keep"}
AVOID_HEADS = {"avoid", "never", "skip"}


def _split_items(x: str) -> List[str]:
    return [y.strip() for y in re.split(r"[,;]", x) if y.strip()]


def parse_note(body: str) -> dict:
    """The text after # or //: {"kind": "archetype",
    "archetype", "items"} (names unresolved), {"kind": "is" | "not", "archetype"},
    {"kind": "core" | "avoid", "items"} or {"kind": "comment"}; each keeps "text"."""
    t = body.strip()
    head, sep, rest = t.partition(":")
    words = head.strip().split()
    first = words[0].casefold() if words else ""
    tail = rest.strip() if sep else " ".join(head.strip().split()[1:]).strip()
    if first in IS_HEADS | NOT_HEADS and tail and (sep and len(words) == 1 or not sep):
        return dict(kind="is" if first in IS_HEADS else "not", archetype=tail, text=t)
    if first in CORE_HEADS | AVOID_HEADS and tail and (sep and len(words) == 1 or not sep):
        items = _split_items(tail)
        if items:
            return dict(kind="core" if first in CORE_HEADS else "avoid", items=items, text=t)
    if words and words[0].casefold() in WEAPON_HEADS and (sep and len(words) <= 2 or not sep and len(words) > 1
                                                          and words[0].casefold() != "w"):
        names = [x.strip() for x in re.split(r"[,;]", tail) if x.strip()] if tail else []
        if names and not all(x.casefold() in _CLASS_WORDS for x in names):   # v3.3: # weapon: A, B sets
            # equips by name; class names (# weapons: Heavy) stay the deprecated v2.21 comment
            return dict(kind="weapon", weapons=names, text=t)
        return dict(kind="comment", text=t)
    if sep and head.strip() and rest.strip():
        items = [x.strip() for x in re.split(r"[,;]", rest) if x.strip()]
        return dict(kind="archetype", archetype=head.strip(), items=items, text=t)
    if ">" in t:
        left, _, arc = t.rpartition(">")
        items = [x.strip() for x in re.split(r"[,;]", left) if x.strip()]
        if items and arc.strip():
            return dict(kind="archetype", archetype=arc.strip(), items=items, text=t)
    return dict(kind="comment", text=t)


def _reason(tok: str) -> Optional[str]:
    t = tok.casefold().strip(".,;:")
    if t in REASON_CODES:
        return REASON_CODES[t]
    if t in ALIASES:
        return ALIASES[t]
    t = t.replace("-", "_")
    return t if t in REASON_CODES.values() else None


def parse_line(n: int, raw: str) -> Optional[Line]:
    """One line of a reply, or None for blank lines."""
    text = _MENTION.sub("", raw).strip()
    if text.startswith("`") and text.endswith("`"):
        text = text.strip("`").strip()
    if not text:
        return None
    low = text.casefold()
    if low in CONTROL:
        return Line(n, raw, "control", control=low)
    if text.startswith("#") or text.startswith("//"):
        body = text[2:] if text.startswith("//") else text[1:]
        if not body.strip():
            return None
        return Line(n, raw, "note", note=body.strip())
    m = re.match(r"^~\s*#\s*(\d*)\s*$", text)
    if m:
        return Line(n, raw, "unnote", note_index=int(m.group(1)) if m.group(1) else None)
    text, op_note = split_note(text)
    if not text:
        return None
    # + - = ~ (a sign glued to the item also works: "+Fast Blade")
    sign = text[0]
    kind = {"+": "add", "-": "remove", "=": "iffy", "~": "unmark"}.get(sign)
    if kind is None:
        return Line(n, raw, "error", error="start the line with + (add), - (remove), = (iffy) or ~ (unmark), "
                                           "or use draft / undo / clear / submit / skip")
    body = text[1:].strip()
    related: List[str] = []
    if ">" in body:
        body, _, rel = body.partition(">")
        related = [x.strip() for x in re.split(r"[,;]", rel) if x.strip()]
    toks = body.split()
    strength = reason = None
    if kind in ("add", "remove", "iffy"):
        for _ in range(2):                         # strength and code, either order, at the end
            if not toks:
                break
            t = toks[-1].casefold()
            if strength is None and t in STRENGTHS:
                strength = STRENGTHS[t]; toks.pop()
            elif reason is None and _reason(t):
                reason = _reason(t); toks.pop()
    if toks and all(_NUMS.match(t) for t in toks):   # fast review: item numbers from the sheet
        if kind == "add":
            return Line(n, raw, "error", error="additions need item names (numbers are the build's own picks)")
        return Line(n, raw, kind, spec=" ".join(toks), strength=strength, reason=reason, related=related,
                    note=op_note)
    item = " ".join(toks).strip()
    names = [x.strip() for x in item.split(",") if x.strip()]
    if not names:
        return Line(n, raw, "error", error="no item named")
    return Line(n, raw, kind, item=names[0], items=names, strength=strength, reason=reason, related=related,
                note=op_note)


def parse_reply(text: str) -> List[Line]:
    out = []
    for i, raw in enumerate((text or "").splitlines(), 1):
        ln = parse_line(i, raw)
        if ln is not None:
            out.append(ln)
    return out


def _resolve(svc, name: str, pool: Sequence[str]) -> str:
    """Exact (or case / punctuation-insensitive) name, else a unique match in `pool`."""
    hit = svc.resolve(name)
    if hit is not None:
        return hit
    found = svc.search(name, list(pool), limit=5)
    pre = [x for x in found if x.casefold().startswith(name.casefold().strip())]
    if len(pre) == 1:
        return pre[0]
    if len(found) == 1:
        return found[0]
    if found:
        raise DraftError(f"'{name}' could mean {', '.join(found[:4])}; write more of the name.")
    raise DraftError(f"'{name}' is not a known talent or mantra.")


@dataclass
class ReplyResult:
    lines: List[Tuple[Line, str, bool]]        # (line, message, ok)
    control: List[str]

    @property
    def errors(self) -> int:
        return sum(1 for _, _, ok in self.lines if not ok)

    def text(self, labels=None) -> str:
        out = []
        for ln, msg, ok in self.lines:
            out.append(msg if ln.kind == "note" else f"{'ok ' if ok else 'err'} {ln.n}: {msg}")
        return "\n".join(out)


def apply_reply(desk: ReviewDesk, user_id: int, text: str, svc) -> ReplyResult:
    """Apply every line of a reply to the user's draft. Raises DraftError if the
    user has no draft. Control words (draft, undo, clear, submit, skip) other
    than undo / clear are returned for the caller to act on."""
    d = desk.require(user_id)
    lines = parse_reply(text)
    res = ReplyResult([], [])
    deferred: List[tuple] = []        # adds refused for mutual exclusion: retried after the other lines
    for ln in lines:
        try:
            if ln.kind == "error":
                raise DraftError(ln.error)
            if ln.kind == "control":
                if ln.control == "undo":
                    o = desk.undo(user_id)
                    res.lines.append((ln, f"undone {describe_op(o)}" if o else "draft already empty", True))
                elif ln.control == "clear":
                    desk.clear(user_id)
                    res.lines.append((ln, "draft cleared", True))
                else:
                    res.control.append(ln.control)
                continue
            if ln.kind == "note":
                res.lines.append((ln, _apply_note(desk, user_id, ln, svc), True))
                continue
            if ln.kind == "unnote":
                gone = desk.remove_note(user_id, ln.note_index)
                res.lines.append((ln, f"dropped {len(gone)} note(s)" if gone else "no notes to drop", True))
                continue
            b = d.build_obj()
            numbered = d.numbered_items()
            if ln.spec:                                 # item numbers from the review sheet
                items = [numbered[k - 1][1] for k in parse_numbers(ln.spec, len(numbered), what="item")]
            else:
                marked = desk.drafts[user_id].items_in_ops()
                if ln.kind == "add":
                    pool = svc.addable(b, exclude=[]) if hasattr(svc, "addable") else []
                elif ln.kind == "unmark":
                    pool = list(marked)
                else:
                    pool = list(b.items())
                items = list(ln.items or [ln.item])     # names, resolved one by one below
            anywhere = ln.reason in ("needs", "sidegrade") or (ln.kind == "iffy" and ln.related)
            rpool = (list(getattr(getattr(svc, "catalog", None), "items", {}) or [])
                     if anywhere else list(b.items()) + desk.drafts[user_id].adds())
            related = [_item_ref(svc, x, numbered, rpool, weapons=ln.reason not in ("needs", "sidegrade"))
                       for x in ln.related]
            for item in items:
                try:
                    if not ln.spec:
                        item = _resolve(svc, item, pool)
                    res.lines.append((ln, _apply_one(desk, user_id, ln, item, related, svc), True))
                except ExclusionError:
                    deferred.append((ln, item, related, len(items) > 1))
                except DraftError as e:
                    res.lines.append((ln, f"{item}: {e}" if len(items) > 1 else str(e), False))
        except DraftError as e:
            res.lines.append((ln, str(e), False))
    for ln, item, related, many in deferred:    # a later line may have removed the blocker
        try:
            res.lines.append((ln, _apply_one(desk, user_id, ln, item, related, svc), True))
        except DraftError as e:
            res.lines.append((ln, f"{item}: {e}" if many else str(e), False))
    res.lines.sort(key=lambda x: x[0].n)
    if getattr(svc, "catalog", None) is not None:
        for z, need in desk.drafts[user_id].stranded(svc.catalog):
            res.lines.append((Line(0, "", "note"), f"note: {z} requires {need}, which you removed; "
                                                    f"remove {z} too before submit", True))
    return res


def _archetype_name(svc, name: str) -> Optional[str]:
    """A known archetype (or variant, "Gank / Frost Mage") by name, any case."""
    a = getattr(svc, "archetypes", None)
    names = list((a() if callable(a) else a) or [])
    k = re.sub(r"\s*/\s*", " / ", name.strip()).casefold()
    return next((x for x in names if re.sub(r"\s*/\s*", " / ", x).casefold() == k), None)


def _apply_note(desk: ReviewDesk, user_id: int, ln: Line, svc) -> str:
    """A # / // line: archetype indicator, is / not, core / avoid, weapon
    classes or a comment."""
    from .reviews import describe_note
    n = parse_note(ln.note or "")
    if n["kind"] in ("is", "not"):
        hit = _archetype_name(svc, n["archetype"])
        n = dict(kind="comment", text=n["text"]) if hit is None else dict(n, archetype=hit)
    elif n["kind"] == "weapon":
        from tri.game.weapons import resolve
        got, bad = [], []
        for x in n["weapons"]:
            hit = resolve(x)
            (got.extend(w for w in hit if w not in got) if hit else bad.append(x))
        if bad:
            raise DraftError(f"unknown weapon {', '.join(bad)} (use the weapon's name, e.g. # weapon: Silver Dagger)")
        n = dict(n, weapons=got)
    elif n["kind"] in ("core", "avoid"):
        d = desk.require(user_id)
        pool = list(getattr(svc.catalog, "items", {}))
        n = dict(n, items=[_item_ref(svc, x, d.numbered_items(), pool) for x in n["items"]])
    if n["kind"] == "archetype":
        hit = _archetype_name(svc, n["archetype"])
        if hit is None:
            n = dict(kind="comment", text=n["text"])          # not a known archetype: keep it as text
        else:
            items = []
            for x in n["items"]:
                if ":" in x and x.split(":", 1)[0].strip().casefold() in ("oath", "murmur"):
                    f, v = x.split(":", 1)
                    items.append(f"{f.strip().casefold()}:{v.strip()}")
                else:
                    d = desk.require(user_id)
                    items.append(_item_ref(svc, x, d.numbered_items(), list(getattr(svc.catalog, "items", {}))))
            n = dict(kind="archetype", archetype=hit, items=items, text=n["text"])
    desk.add_note(user_id, n)
    return "noted " + describe_note(n)


def _item_ref(svc, x: str, numbered, pool, weapons: bool = False) -> str:
    """A related item: a number from the sheet or a name. weapons: a weapon
    name (not an item) gives "weapon:<name>" (v3.3)."""
    if weapons and not x.isdigit() and svc.resolve(x) is None:
        from tri.game.weapons import weapon_token
        w = weapon_token(x)
        if w is not None:
            return w
    if x.isdigit():
        k = int(x)
        if not 1 <= k <= len(numbered):
            raise DraftError(f"there is no item {k} on the sheet (1-{len(numbered)})")
        return numbered[k - 1][1]
    return _resolve(svc, x, pool)


def _apply_one(desk: ReviewDesk, user_id: int, ln: Line, item: str, related, svc) -> str:
    marked = desk.drafts[user_id].items_in_ops()
    if ln.kind == "unmark":
        o = desk.remove_item(user_id, item)
        return f"unmarked {describe_op(o)}" if o else f"{item} was not marked"
    old = marked.get(item)
    if old is not None:                                 # the latest line wins
        desk.remove_item(user_id, item)
    try:
        o = desk.add_op(user_id, ln.kind, item, svc, ln.strength, ln.reason, related, note=ln.note)
    except DraftError:
        if old is not None:                             # put the previous operation back
            desk.drafts[user_id].ops.append(old)
            desk.save()
        raise
    return ("replaced: " if old else "") + describe_op(o)
