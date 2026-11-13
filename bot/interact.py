"""Answering interactions: replies, long replies as files, late delivery, edits."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import discord

from .color import paint_kw, paint_msg
from .embeds import clip
from .guilds import GuildConfig
from .texts import EXPIRED_CODES

log = logging.getLogger("tri.bot")


@dataclass
class Access:
    """Who may run an analysis command here. hide: hold back the model's view
    of builds waiting for review (always on outside tri servers)."""
    hide: bool
    external: bool = False
    g: Optional["GuildConfig"] = None


async def say(inter: discord.Interaction, msg: str = "", **kw):
    """Answer an interaction. If Discord no longer accepts answers to it (a
    command that ran past 15 minutes, such as a slow /model reload), the
    answer goes to the user by DM (ephemeral answers) or to the channel."""
    kw.setdefault("ephemeral", True)
    if kw.get("view", 0) is None:                    # discord.py wants MISSING, not None
        kw.pop("view")
    paint_kw(kw)
    msg = paint_msg(msg)
    try:
        if inter.response.is_done():
            await inter.followup.send(msg, **kw)
        else:
            await inter.response.send_message(msg, **kw)
    except discord.HTTPException as e:
        if getattr(e, "code", None) not in EXPIRED_CODES:
            raise
        await say_late(inter, msg, **kw)


async def say_long(inter: discord.Interaction, text: str, filename: str, head: Optional[str] = None,
                   limit: int = 1990, **kw):
    """Answer with `text`, or, when it does not fit one message, with `head`
    (default: its first line) and the whole text attached as `filename`."""
    if len(text) <= limit:
        return await say(inter, text, **kw)
    first = head if head is not None else text.split("\n", 1)[0]
    kw.pop("files", None)
    await say(inter, clip(first, limit - 60) + f"\n-# full text in {filename}",
              file=discord.File(io.BytesIO(plain_text(text).encode("utf-8")), filename=filename), **kw)


def plain_text(text: str) -> str:
    """Discord markdown (bold, inline code, -# small lines) stripped for a text file."""
    out = [ln[3:] if ln.startswith("-# ") else ln for ln in text.replace("**", "").replace("`", "").split("\n")]
    return "\n".join(out) + "\n"


async def say_late(inter: discord.Interaction, msg: str = "", **kw) -> bool:
    """Deliver an answer outside the interaction. True if delivered."""
    private = kw.pop("ephemeral", True)
    kw.pop("thinking", None)
    kw = {k: v for k, v in kw.items() if k in ("embed", "embeds", "file", "files", "view")}
    note = "(The command took longer than Discord allows for a reply, so here is the answer.)\n"
    if private:
        try:
            await inter.user.send(clip(note + msg, 1990), **kw)
            return True
        except discord.HTTPException:
            pass
    ch = inter.channel
    if ch is not None and hasattr(ch, "send"):
        try:
            await ch.send(clip(f"<@{inter.user.id}> " + note + msg, 1990), **kw,
                          allowed_mentions=discord.AllowedMentions(users=[inter.user], roles=False, everyone=False))
            return True
        except discord.HTTPException:
            pass
    log.warning("could not deliver a late answer to %s: %s", getattr(inter.user, "id", "?"), msg[:200])
    return False


async def edit_reply(inter: discord.Interaction, **kw):
    """Edit the message a component belongs to, before or after deferring."""
    paint_kw(kw)
    if kw.get("content"):
        kw["content"] = paint_msg(kw["content"])
    if inter.response.is_done():
        try:
            await inter.edit_original_response(**kw)
        except discord.NotFound:                       # the message is gone; say it as a follow-up
            await inter.followup.send(kw.get("content") or "Done.", ephemeral=True)
    else:
        await inter.response.edit_message(**kw)
