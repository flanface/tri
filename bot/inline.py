"""Inline functions: square brackets in a server message call tri.

    [A ^ B ^ C]                 3 representative builds where the archetypes meet
    [worth of a, b, c > build]  the worth plot of the items (the build adds its own; optional)
    [a ⟂ b]  [a perp b]         cosine, angle and orthogonality of two worth vectors
    [a ~ b]                     the same, read as similarity
    [like a]                    the items whose worth vectors point most like a's
    [place > build]             a build's regions and archetype
    [help]  [?]                 this list

Any text outside the brackets is ignored, as is a bracket that reads as no
function (so ordinary brackets and Markdown links never trigger). At most
MAX_CALLS calls per message. Reading message text needs the Message Content
intent (TRI_MESSAGE_CONTENT=1 and the toggle in the developer portal);
without it, inline calls work only in messages that mention the bot.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:                                   # pragma: no cover
    from .discord_bot import TriBot

MAX_CALLS = 3
COOLDOWN = 4.0                  # seconds between one user's inline messages
INLINE_LIMIT = 1900             # longer answers go out as a .txt file

_CALL = re.compile(r"\[([^\[\]\n]{1,400})\](?!\()")
_PERP = re.compile(r"\s*(?:⟂|⊥|_\|_|\bperp\b|\borth\b)\s*", re.I)
_SIM = re.compile(r"\s*~\s*")

HELP = """inline functions (square brackets anywhere in a message):
[A ^ B ^ C]  three representative builds where the archetypes meet
[worth of a, b, c > build]  worth plot of the items; the build (optional) adds its own
[a ⟂ b]  or [a perp b]  cosine, angle and orthogonality of two worth vectors
[a ~ b]  the same, read as similarity
[like a]  items whose worth vectors point most like a's
[place > build]  a build's regions and archetype
[help]  this list"""


@dataclass
class Call:
    kind: str                                   # intersect | worth | perp | sim | like | place | help
    args: List[str] = field(default_factory=list)
    build: Optional[str] = None
    raw: str = ""


def _split_build(s: str) -> Tuple[str, Optional[str]]:
    head, sep, tail = s.rpartition(">")
    if sep and tail.strip():
        return head.strip(), tail.strip().split()[0]
    return s.strip(), None


def parse(s: str) -> Optional[Call]:
    """One bracket's text as a Call, or None if it reads as no function."""
    raw = s
    s = " ".join(s.split())
    low = s.casefold()
    if not s:
        return None
    if low in ("help", "?", "tri help", "tri ?"):
        return Call("help", raw=raw)
    if "^" in s:
        names = [x.strip() for x in s.split("^")]
        if len(names) >= 2 and all(names):
            return Call("intersect", names, raw=raw)
        return None
    m = re.match(r"worth(?:\s+of)?\s*(.*)$", s, re.I)
    if m:
        body, build = _split_build(m.group(1))
        items = [x.strip() for x in body.split(",") if x.strip()]
        if items or build:
            return Call("worth", items, build, raw=raw)
        return None
    m = re.match(r"place\s*>?\s*(\S+)$", s, re.I)
    if m:
        return Call("place", [], m.group(1), raw=raw)
    m = re.match(r"like\s+(.+)$", s, re.I)
    if m:
        return Call("like", [m.group(1).strip()], raw=raw)
    for kind, rx in (("perp", _PERP), ("sim", _SIM)):
        parts = [x.strip() for x in rx.split(s)]
        if len(parts) == 2 and all(parts):
            return Call(kind, parts, raw=raw)
    return None


def calls(text: str, limit: int = MAX_CALLS) -> List[Call]:
    """The function calls in a message, in order (at most `limit`)."""
    out = []
    for m in _CALL.finditer(text or ""):
        c = parse(m.group(1))
        if c is not None:
            out.append(c)
            if len(out) >= limit:
                break
    return out


# ------------------------------------------------------------------ running
@dataclass
class Answer:
    text: str
    files: List[Tuple[str, bytes]] = field(default_factory=list)


def _items(bot: "TriBot", names: List[str]) -> Tuple[List[str], List[str]]:
    hit, miss = [], []
    for x in names:
        r = bot.svc.resolve(x)
        if r is None:
            alt = bot.svc.search(x, bot.svc._names, limit=1)
            r = alt[0] if alt else None
        (hit.append(r) if r else miss.append(x))
    return list(dict.fromkeys(hit)), miss


async def _build(bot: "TriBot", hide: bool, ref: str):
    from .links import BuildLinkError
    ids = (await bot.fetcher.expand(ref, limit=1))[:1] or [ref]
    if hide and ids[0] in await bot.pending_ids():
        raise BuildLinkError(f"{ids[0]} is waiting for review; its place is held back until it is reviewed")
    return await bot.load_build(ids[0])


async def run(bot: "TriBot", c: Call, hide: bool) -> Answer:
    """One call's answer. Errors come back as text."""
    from . import insight as I
    from . import plotting as PL
    from .links import BuildLinkError
    try:
        if c.kind == "help":
            return Answer(HELP)
        if c.kind == "intersect":
            ex = await bot.pending_ids() if hide else set()
            r = await bot.model_call(I.intersect, bot.svc, c.args, 3, ex)
            return Answer(r["text"])
        if c.kind == "worth":
            hi, miss = _items(bot, c.args)
            if c.build:
                _, parsed = await _build(bot, hide, c.build)
                hi += [x for x in parsed.build.items() if x not in hi]
            if not hi:
                return Answer("Not found: " + ", ".join(miss[:10]) if miss else "name some talents or mantras")
            files, text = await asyncio.to_thread(bot.svc.plot_worth, hi, False, 2000)
            if miss:
                text = f"Not found: {', '.join(miss[:10])}\n" + text
            return Answer(text, list(files))
        if c.kind in ("perp", "sim"):
            hi, miss = _items(bot, c.args)
            if miss or len(hi) != 2:
                return Answer("Not found: " + ", ".join(miss) if miss else "name two different items")
            r = await asyncio.to_thread(PL.worth_pair, bot.svc, hi[0], hi[1])
            head = (f"{hi[0]} ⟂ {hi[1]}: orthogonality {r['orthogonality']:.2f}" if c.kind == "perp"
                    else f"{hi[0]} ~ {hi[1]}: cos {r['cos']:+.2f}")
            return Answer(f"{head}\n" + PL.pair_lines(hi, [[1.0, r["cos"]], [r["cos"], 1.0]], "")[0]
                          + "\n(worth axes 4-5-6 from the dense origin; orthogonality |sin θ|: 0 parallel, 1 orthogonal)")
        if c.kind == "like":
            hi, miss = _items(bot, c.args)
            if not hi:
                return Answer("Not found: " + ", ".join(miss))

            def near():
                with bot.svc.lock:
                    ws = PL.worth_space(bot.svc)
                if hi[0] not in ws["items"]:
                    raise ValueError(f"{hi[0]}: rarely taken (mean worth under 0.5%), no worth vector")
                from tri.views import figure as F
                X = PL.worth_vectors(ws, hi[:1])
                C = F.cosines(X["X"] - X["origin"])[X["idx"][0]]
                order = [t for t in C.argsort()[::-1] if t != X["idx"][0]][:8]
                return [(ws["items"][t], float(C[t])) for t in order]
            rows = await asyncio.to_thread(near)
            return Answer(f"{hi[0]} points like:\n" + "\n".join(f"  {n} · cos {c_:+.2f}" for n, c_ in rows))
        if c.kind == "place":
            bid, parsed = await _build(bot, hide, c.build)

            lines = await asyncio.to_thread(PL.place_lines, bot.svc, [(parsed.name or bid, parsed.build)])
            return Answer("\n".join(lines))
    except BuildLinkError as e:
        return Answer(f"Could not read that build: {e}")
    except ValueError as e:
        return Answer(str(e))
    return Answer("unknown function")


class Gate:
    """Per-user cooldown on inline messages."""

    def __init__(self, cooldown: float = COOLDOWN):
        self.cooldown = cooldown
        self.last: Dict[int, float] = {}

    def ok(self, user_id: int) -> bool:
        now = time.monotonic()
        if now - self.last.get(user_id, -1e9) < self.cooldown:
            return False
        self.last[user_id] = now
        return True


async def handle(bot: "TriBot", message) -> bool:
    """Answer the inline calls of a server message. False if it has none."""
    import discord

    from .embeds import clip
    cs = calls(message.content or "")
    if not cs:
        return False
    g = await bot.config(message)
    if not bot.is_user(message, g):
        return True                             # silent: no access, no noise
    gate = getattr(bot, "_inline_gate", None)
    if gate is None:
        gate = bot._inline_gate = Gate()
    if not gate.ok(message.author.id):
        return True
    hide = bool(g.hide_pending and not bot.is_admin(message, g))
    texts, files = [], []
    async with message.channel.typing():
        for c in cs:
            a = await run(bot, c, hide)
            texts.append(a.text if len(cs) == 1 else f"[{' '.join(c.raw.split())}]\n{a.text}")
            files += a.files
    text = "\n\n".join(t for t in texts if t)
    dfiles = [discord.File(io.BytesIO(b), n) for n, b in files[:9]]
    if len(text) > INLINE_LIMIT:
        dfiles.append(discord.File(io.BytesIO(text.encode("utf-8")), "tri.txt"))
        text = clip(text.split("\n", 1)[0], 300) + "\n(full answer attached)"
    no_pings = discord.AllowedMentions.none()
    try:
        await message.reply(clip(text, 1990) or None, files=dfiles, mention_author=False, allowed_mentions=no_pings)
    except discord.HTTPException:
        pass
    return True
