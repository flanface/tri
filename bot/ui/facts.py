"""Fact graphics under /facts."""
from __future__ import annotations

import io
import logging
from typing import Optional, Sequence

import discord

from ..embeds import clip, fact_line
from ..texts import NO_PINGS

log = logging.getLogger("tri.bot")


def fact_graphic(graphic: str) -> tuple:
    """A fact's graphic tag -> (insight kind, keyword arguments)."""
    head, _, arg = (graphic or "").partition(":")
    if head == "card":
        return "card", dict(archetype=arg)
    if head in ("item", "stats"):
        return head, dict(item=arg)
    if head in ("meta", "tree", "atlas"):          # tree and meta charts are deprecated (v2.21): the atlas
        return "atlas", {}
    raise ValueError(f"no graphic {graphic!r}")


class FactView(discord.ui.View):
    """Numbered buttons under /facts: each posts that fact's graphic.
    draw(graphic) -> [(filename, bytes)] is a coroutine."""

    def __init__(self, owner_id: int, rows: Sequence[dict], draw, timeout: float = 900):
        super().__init__(timeout=timeout)
        self.owner_id, self.rows, self.draw = owner_id, list(rows)[:10], draw
        self.message: Optional[discord.Message] = None
        for i, r in enumerate(self.rows):
            b = discord.ui.Button(label=str(i + 1), style=discord.ButtonStyle.secondary, row=i // 5)
            b.callback = self._cb(i)
            self.add_item(b)

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.owner_id:
            await inter.response.send_message("Only the person who asked can open these.", ephemeral=True)
            return False
        return True

    def _cb(self, i: int):
        async def cb(inter: discord.Interaction):
            await inter.response.defer(thinking=True)
            r = self.rows[i]
            try:
                files = await self.draw(r["graphic"])
                await inter.followup.send(clip(f"**{i + 1}.** {fact_line(r)}", 1990),
                                          files=[discord.File(io.BytesIO(b), n) for n, b in files[:1]],
                                          allowed_mentions=NO_PINGS)
            except Exception as e:
                log.warning("fact graphic %s failed", r.get("graphic"), exc_info=True)
                await inter.followup.send(f"Could not draw it ({type(e).__name__}).")
        return cb

    async def on_timeout(self) -> None:
        for c in self.children:
            c.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass
