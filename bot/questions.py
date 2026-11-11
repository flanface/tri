"""Caution questions as a sheet (v3.1.1, numbered answers v3.1.3).

Before /analyze or /complete posts, tri posts its questions as a numbered
sheet:

    1. Archetype? gank / chime
    2. Martyr central? yes / no / not sure
    3. Weapon class? fist / spear
    4. Want Exhaustion Strike? yes / no

The asker answers by number, on one line or several:

    1. gank 2. yes 3. fist 4. no     (also "1 gank", "3. 2" = the second option)
    gank, yes, fist, no              one answer per question, in order

The review-sheet grammar still works:

    + 2, 3          yes: central to the concept   (also "2 yes", "+ Martyr")
    - 4             no: not central               (also "4 no", "- Martyr")
    = 5             not sure                      (also "5 ?", "5 not sure")
    # Gank          the archetype                 (also "1 Gank", "# is Gank")
    # core: A, B    yes for named items (central)
    post            post the analysis now         (also "done", "submit", "skip")

Several answers may share a reply, one per line or split by ";". The round
closes when every question on the sheet has an answer, or on "post". Answers
can make new questions necessary; those come as a new sheet. Nothing here
touches Discord: render_sheet and parse_answers are pure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

YES = {"yes", "y", "+", "central", "core", "true"}
NO = {"no", "n", "-", "not central", "false"}
UNSURE = {"?", "=", "not sure", "unsure", "idk", "dunno", "maybe"}
DONE = {"post", "done", "submit", "ok", "skip", "finish", "go"}


@dataclass
class Parsed:
    archetype: Optional[str] = None
    central: Dict[str, bool] = field(default_factory=dict)
    want: Dict[str, bool] = field(default_factory=dict)
    weapon: Optional[str] = None
    unsure: Set[str] = field(default_factory=set)      # question keys answered "not sure"
    answered: Set[str] = field(default_factory=set)    # question keys with any answer
    done: bool = False
    errors: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)     # one confirmation per accepted answer


def render_sheet(questions: Sequence[dict], title: str = "", round_no: int = 0, timeout: float = 0) -> str:
    """The numbered question sheet, short (plain text for a channel message)."""
    out = [("More questions" if round_no else "Questions") + (f" on {title}" if title else "") + " (reply here):"]
    eg = []
    for i, q in enumerate(questions, 1):
        k, opts = q.get("kind"), [str(o).lower() for o in (q.get("options") or [])]
        if k == "archetype":
            line = "Archetype? " + " / ".join(opts)
        elif k == "weapon":
            line = "Weapon? " + (" / ".join(opts) + ", or the weapon's name" if opts else "the weapon's name or class")
        elif k == "want":
            line, opts = f"Want {q.get('item')}? yes / no", ["yes", "no"]
        else:
            line, opts = f"{q.get('item') or q['text']} central? yes / no / not sure", ["yes", "no"]
        out.append(f"{i}. {line}")
        eg.append(f"{i}. {opts[0] if opts else 'yes'}")
    out.append(f"Reply e.g. `{' '.join(eg[:4])}` · `post` posts now" + (f" · {int(timeout)} s" if timeout else ""))
    return "\n".join(out)


def _arch(name: str, options: Sequence[str], resolve: Optional[Callable[[str], Optional[str]]]) -> Optional[str]:
    k = re.sub(r"\s*/\s*", " / ", name.strip()).casefold()
    for o in options:
        if re.sub(r"\s*/\s*", " / ", o).casefold() == k:
            return o
    pre = [o for o in options if o.casefold().startswith(k)]
    if len(pre) == 1:
        return pre[0]
    return resolve(name) if resolve else None


def parse_answers(text: str, questions: Sequence[dict],
                  resolve_item: Optional[Callable[[str], Optional[str]]] = None,
                  resolve_archetype: Optional[Callable[[str], Optional[str]]] = None) -> Parsed:
    """A reply to a question sheet -> Parsed. resolve_item / resolve_archetype map a
    written name to a known one (None: unknown); without them only the names on
    the sheet are understood."""
    p = Parsed()
    qs = list(questions)
    arch_q = next((q for q in qs if q.get("kind") == "archetype"), None)
    by_item = {q["item"].casefold(): q for q in qs if q.get("kind") in ("central", "want") and q.get("item")}
    opts = list(arch_q.get("options") or []) if arch_q else []

    def item_of(tok: str):
        t = tok.strip()
        if t.isdigit():
            i = int(t)
            if not 1 <= i <= len(qs):
                p.errors.append(f"there is no question {i}")
                return None
            return qs[i - 1]
        q = by_item.get(t.casefold())
        if q is not None:
            return q
        hit = resolve_item(t) if resolve_item else None
        if hit is None:
            pre = [q for k, q in by_item.items() if k.startswith(t.casefold())]
            if len(pre) == 1:
                return pre[0]
            p.errors.append(f"'{t}' is not a question or a known talent or mantra")
            return None
        return by_item.get(hit.casefold()) or dict(key=f"central:{hit}", kind="central", item=hit)

    def central(q, val):
        if q.get("kind") in ("archetype", "weapon"):
            p.errors.append(f"the {q.get('kind')} question wants a name, e.g. `{qs.index(q) + 1 if q in qs else 1}. "
                            f"{(q.get('options') or ['name'])[0]}`")
            return
        it = q["item"]
        if q.get("kind") == "want":                  # v3.1.3: do you want the held add?
            if val is None:
                p.want.pop(it, None)
                p.unsure.add(q["key"])
            else:
                p.want[it] = val
                p.unsure.discard(q["key"])
            p.answered.add(q["key"])
            p.lines.append(f"want {it}: {'yes' if val else 'no' if val is False else 'not sure'}")
            return
        if val is None:
            p.central.pop(it, None)
            p.unsure.add(q["key"])
            p.lines.append(f"= {it}: not sure")
        else:
            p.central[it] = val
            p.unsure.discard(q["key"])
            p.lines.append(f"{'+' if val else '-'} {it}: {'central' if val else 'not central'}")
        p.answered.add(q["key"])

    def archetype(name: str):
        name = re.sub(r"^(is|as|archetype:?)\s+", "", name.strip(), flags=re.I).strip()
        if name.casefold() in UNSURE:
            if arch_q:
                p.answered.add(arch_q["key"])
                p.unsure.add(arch_q["key"])
                p.lines.append("= archetype: not sure")
            return
        a = _arch(name, opts, resolve_archetype)
        if a is None:
            p.errors.append(f"'{name}' is not a known archetype" + (f" (options: {', '.join(opts)})" if opts else ""))
            return
        p.archetype = a
        if arch_q:
            p.answered.add(arch_q["key"])
        p.lines.append(f"# {a}")

    def weapon(q, ans: str):
        a = ans.strip().casefold()
        if a in UNSURE:
            p.unsure.add(q["key"])
            p.answered.add(q["key"])
            return
        opts = [str(o) for o in (q.get("options") or [])]
        hit = next((o for o in opts if o.casefold() == a), None)
        if hit is None:
            try:
                from tri.infer.focus import norm_class
                hit = norm_class(ans.strip())
            except Exception:                       # noqa: BLE001
                hit = None
        if hit is None:                              # v3.2: a weapon name, kept as written in the game data
            try:                                     # (a class name wins: "sword" is the class)
                from tri.game.weapons import resolve
                names = resolve(ans.strip())
            except Exception:                       # noqa: BLE001
                names = []
            if names:
                p.weapon = names[0]
                p.answered.add(q["key"])
                p.lines.append(f"weapon: {p.weapon}")
                return
        if hit is None:
            pre = [o for o in opts if o.casefold().startswith(a)]
            hit = pre[0] if len(pre) == 1 else None
        if hit is None:
            p.errors.append(f"'{clip(ans)}' is not a weapon or a weapon class" + (f" (options: {', '.join(opts)})" if opts else ""))
            return
        p.weapon = hit.lower()
        p.answered.add(q["key"])
        p.lines.append(f"weapon: {p.weapon}")

    def answer(q, ans: str, n=None):
        """One answer to question q (by kind)."""
        a = ans.strip().strip("`").strip().rstrip(".").strip()
        low = a.casefold()
        opts = list(q.get("options") or [])
        if q.get("kind") in ("archetype", "weapon") and a.isdigit() and 1 <= int(a) <= len(opts):
            a = str(opts[int(a) - 1])                # "1. 2": the second option
        if q.get("kind") == "archetype":
            archetype(a)
        elif q.get("kind") == "weapon":
            weapon(q, a)
        elif low in YES:
            central(q, True)
        elif low in NO:
            central(q, False)
        elif low in UNSURE:
            central(q, None)
        else:
            p.errors.append(f"question {n or '?'}: answer " + ("yes or no" if q.get("kind") == "want"
                                                               else "yes, no or not sure"))

    def refs(s: str) -> List[str]:
        return [x for x in re.split(r"[,\s]+" if re.fullmatch(r"[\d,\s]+", s) else r"\s*,\s*", s.strip()) if x]

    text = text or ""
    # v3.1.3: "1. gank 2. yes 3. fist": numbered answers anywhere in the reply
    num = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[;,]))(\d{1,2})\s*[.):]\s*(?!\d)")
    hits = list(num.finditer(text))
    if not hits and not re.search(r"^\s*[+\-=?#]", text, re.M):   # "1 gank 2 yes 3 fist"
        bare = list(re.finditer(r"(?:(?<=\s)|(?<=^)|(?<=[;,]))(\d{1,2})\s+(?=[^\d\s,])", text))
        if len(bare) >= 2:
            hits = bare
    if hits:
        rest = text[:hits[0].start()]
        for j, m in enumerate(hits):
            seg = text[m.end(): hits[j + 1].start() if j + 1 < len(hits) else len(text)]
            seg = seg.strip().strip(",;").strip()
            i = int(m.group(1))
            if not 1 <= i <= len(qs):
                p.errors.append(f"there is no question {i}")
                continue
            low = seg.casefold().rstrip(".")
            tail = None
            for d in sorted(DONE, key=len, reverse=True):     # "4. no post"
                if low.endswith(" " + d):
                    seg, tail = seg[: -len(d)].strip(), d
                    break
            if seg:
                answer(qs[i - 1], seg, i)
            else:
                p.errors.append(f"question {i}: no answer")
            if tail or low in DONE:
                p.done = True
        text = rest
    elif qs and not re.search(r"^\s*[+\-=?#]", text, re.M):
        # "gank, yes, fist": one answer per question, in order
        parts = [x.strip() for x in re.split(r"[\n;,]+", text) if x.strip()]
        done = [x for x in parts if x.casefold().rstrip(".") in DONE]
        parts = [x for x in parts if x not in done]
        if len(parts) == len(qs) and len(qs) > 1:
            for i, (q, a) in enumerate(zip(qs, parts), 1):
                answer(q, a, i)
            p.done = p.done or bool(done)
            return p

    for raw in re.split(r"[\n;]+", text):
        ln = raw.strip().strip("`").strip()
        if not ln:
            continue
        low = ln.casefold().rstrip(".")
        if low in DONE:
            p.done = True
            continue
        m = re.match(r"^#\s*(core|avoid)\s*:\s*(.+)$", ln, re.I)
        if m:
            for t in refs(m.group(2)):
                q = item_of(t)
                if q is not None:
                    central(q, m.group(1).casefold() == "core")
            continue
        if ln.startswith("#"):
            archetype(ln[1:])
            continue
        m = re.match(r"^([+\-=?])\s*(.+)$", ln)
        if m:
            val = {"+": True, "-": False}.get(m.group(1))
            for t in refs(m.group(2)):
                q = item_of(t)
                if q is not None:
                    central(q, val)
            continue
        m = re.match(r"^([\d,\s]+?)\s*[:.)]?\s+(.+)$", ln)
        if m and re.search(r"\d", m.group(1)):
            for t in refs(m.group(1)):
                q = item_of(t)
                if q is not None:
                    answer(q, m.group(2), t)
            continue
        if low in YES | NO | UNSURE:
            open_c = [q for q in qs if q.get("kind") in ("central", "want")]
            if len(open_c) == 1:
                central(open_c[0], True if low in YES else False if low in NO else None)
            else:
                p.errors.append(f"'{ln}': say which question, e.g. `2 {low}`")
            continue
        if arch_q and _arch(ln, opts, resolve_archetype):
            archetype(ln)
            continue
        wq = [q for q in qs if q.get("kind") == "weapon"]
        if len(wq) == 1 and any(str(o).casefold() == low for o in wq[0].get("options") or []):
            weapon(wq[0], ln)
            continue
        p.errors.append(f"could not read '{clip(ln)}'")
    return p


def clip(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class Session:
    """One open question sheet: the asker, its questions and what they answered."""
    user_id: int
    questions: List[dict]
    future: object = None                       # asyncio.Future, set when the round closes
    message: object = None                      # the posted sheet
    archetype: Optional[str] = None
    central: Dict[str, bool] = field(default_factory=dict)
    want: Dict[str, bool] = field(default_factory=dict)
    weapon: Optional[str] = None
    answered: Set[str] = field(default_factory=set)
    done: bool = False

    def take(self, p: Parsed) -> None:
        if p.archetype:
            self.archetype = p.archetype
        if p.weapon:
            self.weapon = p.weapon
        for k in p.unsure:
            if k.startswith("central:"):
                self.central.pop(k.split(":", 1)[1], None)
            if k.startswith("want:"):
                self.want.pop(k.split(":", 1)[1], None)
        self.central.update(p.central)
        self.want.update(p.want)
        self.answered |= p.answered
        self.done = self.done or p.done

    def take_session(self, o: "Session") -> None:
        """Merge another round's answers into this one."""
        if o.archetype:
            self.archetype = o.archetype
        if o.weapon:
            self.weapon = o.weapon
        for q in o.questions:
            if q["key"] in o.answered and q.get("kind") == "central" and q.get("item") not in o.central:
                self.central.pop(q.get("item"), None)
            if q["key"] in o.answered and q.get("kind") == "want" and q.get("item") not in o.want:
                self.want.pop(q.get("item"), None)
        self.central.update(o.central)
        self.want.update(o.want)
        self.answered |= o.answered

    @property
    def closed(self) -> bool:
        return self.done or all(q["key"] in self.answered for q in self.questions)

    def open_numbers(self) -> List[int]:
        return [i for i, q in enumerate(self.questions, 1) if q["key"] not in self.answered]

    def answers(self) -> dict:
        return dict(archetype=self.archetype, central=dict(self.central), want=dict(self.want), weapon=self.weapon)
