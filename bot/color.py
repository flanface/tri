"""Colour for embed text (v3.0.4): every code block becomes an ``ansi`` block
and plain embed text is wrapped in one, with marks coloured by meaning.

    [+] add, [v] uphold, gem                 green
    [-] remove, [x] overturn, hole, !, ¬     red
    [=] iffy, review                         yellow
    foundational                             blue
    central, authored, numbers               cyan
    archetype, indicative of A               pink
    weak                                     grey (luxury, keys: plain)
    strong, headers, ⇒                       bold

TRI_COLOR=0 turns it off (plain code blocks, as before). Discord limits still hold: a part that would grow past its limit is left as
it was, and if the embed passes 6000 characters the parts that grew most go
back to plain text first. Text with links or mentions is never wrapped (a
code block would make it unclickable).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

ENABLED = os.environ.get("TRI_COLOR", "1").strip().lower() not in ("0", "false", "no", "off")

ESC = "\u001b["
RESET = ESC + "0m"

GREY, RED, GREEN, YELLOW, BLUE, PINK, CYAN, WHITE = 30, 31, 32, 33, 34, 35, 36, 37

PLAIN = 0
PASS_COLOR = {"foundational": BLUE, "central": CYAN, "archetype": PINK, "review": YELLOW,
              "luxury": PLAIN, "general": BLUE, "rules": CYAN, "worth": PLAIN, "authored": CYAN,
              "mined": PLAIN, "review_rule": YELLOW}


def c(text: str, color: int, bold: bool = False) -> str:
    return f"{ESC}{'1;' if bold else ''}{color}m{text}{RESET}"


_TOKENS: List[Tuple[str, object]] = [
    ("plus", r"\[\+\]|\[v\]"),
    ("minus", r"\[-\]|\[x\]"),
    ("iffy", r"\[=\]"),
    ("ruleX", r"\[X\]"),
    ("ruleC", r"\[[A-Z]\]"),
    ("passtag", r"\((?:" + "|".join(PASS_COLOR) + r")\)"),
    ("strong", r"\bstrong\b"),
    ("weak", r"\bweak\b"),
    ("over", r"\b\d+/\d+!|\b\d+ over\b"),
    ("indic", r"(?<=indicative of )[^,;()\n]+"),
    ("hole", r"^hole:"),
    ("gem", r"^gem:"),
    ("key", r"^[a-z][a-z /_-]{0,18}:(?=\s)"),
    ("arrow", r"⇒|=>"),
    ("neg", r"¬"),
    ("num", r"(?<![\w.'])[+-]?\d+(?:\.\d+)?%?(?:/\d+)?(?![\w'])"),
]
_RX = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKENS), re.M)


def _tok(m: re.Match) -> str:
    k, s = m.lastgroup, m.group(0)
    if k == "plus":
        return c(s, GREEN, True)
    if k == "minus":
        return c(s, RED, True)
    if k == "iffy":
        return c(s, YELLOW, True)
    if k == "ruleX":
        return c(s, RED, True)
    if k == "ruleC":
        return c(s, CYAN, True)
    if k == "passtag":
        col = PASS_COLOR.get(s[1:-1], PLAIN)
        return c(s, col) if col else s
    if k == "strong":
        return c(s, WHITE, True)
    if k == "key":
        return s
    if k == "weak":
        return c(s, GREY)
    if k in ("over", "hole", "neg"):
        return c(s, RED, k != "neg")
    if k == "gem":
        return c(s, GREEN)
    if k == "indic":
        return c(s, PINK)
    if k == "arrow":
        return c(s, WHITE, True)
    return c(s, CYAN)


def paint_line(line: str) -> str:
    if line.startswith("-- ") or (line.endswith(":") and len(line) < 40 and not line.startswith("[")):
        return c(line, BLUE, True)
    return _RX.sub(_tok, line)


def paint_lines(text: str) -> str:
    return "\n".join(paint_line(l) for l in text.split("\n"))


_BLOCK = re.compile(r"```([a-z]*)\n?(.*?)```", re.S)
_LINKY = re.compile(r"\]\(|https?://|<[@#&!:]|<t:")


def paint_text(text: Optional[str], wrap: bool = True) -> Optional[str]:
    """Every code block of text as an ansi block, coloured. With wrap, text
    with no code block, link or mention becomes one ansi block."""
    if not ENABLED or not text or not text.strip() or text.strip() == "\u200b":
        return text
    if "```" in text:
        return _BLOCK.sub(lambda m: m.group(0) if m.group(1) == "ansi" else
                          "```ansi\n" + paint_lines(m.group(2).rstrip("\n")) + "\n```", text)
    if not wrap or _LINKY.search(text):
        return text
    plain = text.replace("**", "").replace("`", "'")
    return "```ansi\n" + paint_lines(plain) + "\n```"


def _fit(old: Optional[str], new: Optional[str], limit: int) -> Optional[str]:
    if new is None or len(new) <= limit:
        return new
    retag = old.replace("```\n", "```ansi\n") if old and "```" in old else old
    return retag if retag is not None and len(retag) <= limit else old


def _chunks(value: str, limit: int = 1024) -> List[str]:
    """A painted value as code blocks of whole lines within limit."""
    m = re.fullmatch(r"```ansi\n(.*)\n```", value, re.S)
    if m is None:
        return [value] if len(value) <= limit else []
    out, cur = [], []
    for ln in m.group(1).split("\n"):
        if len(ln) + 13 > limit:
            return []
        if cur and len("\n".join(cur + [ln])) + 12 > limit:
            out.append(cur); cur = []
        cur.append(ln)
    out.append(cur)
    return ["```ansi\n" + "\n".join(x) + "\n```" for x in out]


def paint_embed(e, total: int = 6000, max_fields: int = 25):
    """The embed with its description and field values coloured (in place;
    returns it). A field that grows past 1024 characters continues in the
    next field ("(cont.)"). If the embed would pass Discord's limits, the
    fields that grew most go back to plain text. Painting twice is harmless."""
    if e is None or not ENABLED:
        return e
    d = e.description
    if d:
        e.description = _fit(d, paint_text(d), 4096)
    old = [(f.name, f.value, f.inline) for f in e.fields]
    new = []
    for name, value, inline in old:
        p = paint_text(value)
        parts = _chunks(p) if p != value else [value]
        if not parts:
            parts = [_fit(value, p, 1024)]
        more = name if name.endswith("(cont.)") else f"{name} (cont.)"
        new.append([(name if i == 0 else more, v, inline) for i, v in enumerate(parts)])

    def build(rows):
        e.clear_fields()
        for name, value, inline in rows:
            e.add_field(name=name[:256], value=value, inline=inline)

    grow = sorted(range(len(old)), key=lambda i: -(sum(len(v) for _, v, _ in new[i]) - len(old[i][1] or "")))
    plain = set()
    while True:
        rows = [r for i in range(len(old)) for r in ([old[i]] if i in plain else new[i])]
        build(rows)
        if (len(e) <= total and len(rows) <= max_fields) or len(plain) == len(old):
            break
        plain.add(next(i for i in grow if i not in plain))
    if len(e) > total and d:
        e.description = d
    return e


def paint_kw(kw: dict) -> dict:
    """Paint the embed / embeds of a send or edit call's keyword arguments."""
    if kw.get("embed") is not None:
        paint_embed(kw["embed"])
    for x in kw.get("embeds") or []:
        paint_embed(x)
    return kw


def paint_msg(text: Optional[str], limit: int = 1990) -> Optional[str]:
    """Message content: its code blocks coloured, if that still fits."""
    if not text or "```" not in text:
        return text
    p = paint_text(text, wrap=False)
    return p if len(p) <= limit else text
