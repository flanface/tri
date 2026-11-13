"""/talent and /mantra: catalog lookups."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, List

import discord
from discord import app_commands

from ..embeds import clip, lookup_embed, matches_embed
from ..interact import say
from ..texts import USER_APP_CONTEXTS, USER_APP_INSTALLS

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    insight = c.insight
    plot_fail = c.plot_fail
    plot_files = c.plot_files

    def _kind_ac(want: str):
        async def ac(inter: discord.Interaction, current: str):
            if not bot.svc:
                return []
            c = bot.svc.catalog
            names = [n for n in bot.svc.search(current.strip(), bot.svc._names)
                     if n in c.items and c.items[n].kind.value == want]
            return [app_commands.Choice(name=clip(n, 100), value=clip(n, 100)) for n in names[:25]]
        return ac

    def _kind_matches(name: str, want: str) -> List[str]:
        c = bot.svc.catalog
        return [n for n in bot.svc.search(name.strip(), bot.svc._names, limit=200)
                if n in c.items and c.items[n].kind.value == want]

    async def _lookup(inter: discord.Interaction, name: str, want: str):
        """v3.0.3: an exact (or loose) name, else a substring search: one match
        is looked up, several are listed."""
        if bot.svc and not bot.svc.resolve(name):
            hits = _kind_matches(name, want)
            if not hits:
                return await say(inter, f"no {want} matches `{clip(name, 80)}`", ephemeral=True)
            if len(hits) > 1:
                return await say(inter, embed=matches_embed(name, want, hits), ephemeral=True)
            name = hits[0]
        await inter.response.defer(thinking=True)
        try:
            r = await insight("lookup", item=name)
        except Exception as e:
            return await plot_fail(inter, e, "lookup")
        files = plot_files(r["files"])
        img = files[0].filename if files else None
        await say(inter, embed=lookup_embed(r["data"], img), files=files, ephemeral=False)

    @tree.command(name="talent", description="Talent lookup (substring search)")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(name="name or part of it")
    async def talent_cmd(inter: discord.Interaction, name: str):
        await _lookup(inter, name, "talent")

    @tree.command(name="mantra", description="Mantra lookup (substring search)")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(name="name or part of it")
    async def mantra_cmd(inter: discord.Interaction, name: str):
        await _lookup(inter, name, "mantra")

    talent_cmd.autocomplete("name")(_kind_ac("talent"))

    mantra_cmd.autocomplete("name")(_kind_ac("mantra"))
