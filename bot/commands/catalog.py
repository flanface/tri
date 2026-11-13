"""/category and /value: item categories (read only) and flat worth offsets."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..embeds import clip
from ..interact import say, say_long

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin
    item_name_ac = c.item_name_ac
    reviewer = c.reviewer

    category = app_commands.Group(name="category", description="Item categories (read only)",
                                  guild_only=True)

    @category.command(name="list", description="All categories (read only)")
    async def category_list(inter: discord.Interaction):
        g = await reviewer(inter)
        if not g:
            return
        cats = bot.svc.categories()
        if not cats:
            return await say(inter, "The loaded run stores no categories; tiers come from the selection rules.")
        lines = []
        for c in ("foundational", "central", "luxury"):
            xs = sorted((i for i, v in cats.items() if v == c), key=str.casefold)
            if xs:
                lines.append(f"{c} ({len(xs)}): " + ", ".join(xs))
        await say_long(inter, "\n".join(lines), "categories.txt")

    value = app_commands.Group(name="value", description="Flat worth offsets",
                               guild_only=True)

    @value.command(name="set", description="Add a flat worth offset (admin)")
    @app_commands.describe(item="talent or mantra", delta="offset in [-0.5, 0.5]",
                           note="optional reason")
    async def value_set(inter: discord.Interaction, item: str, delta: float, note: Optional[str] = None):
        if not await admin(inter):
            return
        name = bot.svc.resolve(item) or item
        if name not in bot.svc.catalog.index:
            return await say(inter, f"**{item}** is not in the model's catalog.")
        if delta == 0 or abs(delta) > 0.5:
            return await say(inter, "A value is a nonzero offset between -0.5 and 0.5. `/value clear` removes one.")
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            await bot.store.set_item_value(name, delta, set_by=f"discord:{inter.user.id}", note=note)
        except Exception as e:
            return await say(inter, f"Not saved, the database refused it (run supabase/schema.sql "
                                    f"if the table is missing): {e}")
        await asyncio.to_thread(bot.svc.set_value, name, delta)
        await say(inter, f"**{name}** worth {delta:+.2f} on every build.", ephemeral=False)

    @value.command(name="clear", description="Remove a flat worth offset (admin)")
    @app_commands.describe(item="talent or mantra")
    async def value_clear(inter: discord.Interaction, item: str):
        if not await admin(inter):
            return
        name = bot.svc.resolve(item) or item
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            await bot.store.clear_item_value(name)
        except Exception as e:
            return await say(inter, f"Not cleared, the database refused it: {e}")
        if name in bot.svc.catalog.index:
            await asyncio.to_thread(bot.svc.set_value, name, None)
        await say(inter, f"**{name}** has no flat value now.", ephemeral=False)

    @value.command(name="list", description="Every flat value")
    async def value_list(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        vals = bot.svc.values() if hasattr(bot.svc, "values") else {}
        if not vals:
            return await say(inter, "No flat values set. `/value set` adds one.")
        xs = sorted(vals.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
        await say(inter, clip("\n".join(f"{v:+.2f}  {i}" for i, v in xs), 1990))

    value_set.autocomplete("item")(item_name_ac)
    value_clear.autocomplete("item")(item_name_ac)

    tree.add_command(category)
    tree.add_command(value)
