"""/archetype, /facts and /rules: archetype cards, certified facts and model rules."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..embeds import clip, FACT_SYMBOLS, facts_text, RULE_SYMBOLS, rules_text, SEP
from ..interact import say, say_long
from ..texts import USER_APP_CONTEXTS, USER_APP_INSTALLS
from ..ui.facts import fact_graphic, FactView

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    analyst = c.analyst
    archetype_name_ac = c.archetype_name_ac
    insight = c.insight
    item_name_ac = c.item_name_ac
    plot_fail = c.plot_fail
    plot_files = c.plot_files

    @tree.command(name="archetype", description="Archetype card")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(name="archetype")
    async def archetype_cmd(inter: discord.Interaction, name: str):
        if not await analyst(inter, "reviewer"):
            return
        await inter.response.defer(thinking=True)
        try:
            r = await insight("card", archetype=name)
        except Exception as e:
            return await plot_fail(inter, e, "card")
        d = r["data"]
        head = f"**{d['name']}**{SEP}{d['builds']} builds ({100 * d['share']:.0f}%)"
        tells = "".join(f"\n`{t['symbol']}`" for t in d.get("tells") or [])
        cons = d.get("consensus") or []
        if cons:                                            # reviewer consensus (math §8.5a)
            tells += "\n`consensus: " + " ".join(f"{'+' if c['sign'] > 0 else '−'}{c['item']} ({c['reviewers']}, "
                                                  f"{c['confidence']:.2f})" for c in cons[:6]) + "`"
        msg = head + tells + ("\n" + facts_text(d["facts"]) if d.get("facts") else "")
        await say(inter, clip(msg, 1990), files=plot_files(r["files"]), ephemeral=False)

    archetype_cmd.autocomplete("name")(archetype_name_ac)

    @tree.command(name="facts", description="Certified facts")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(archetype="filter by archetype")
    async def facts_cmd(inter: discord.Interaction, archetype: Optional[str] = None):
        if not await analyst(inter, "reviewer"):
            return
        await inter.response.defer(thinking=True)
        try:
            r = await insight("facts", archetype=archetype, limit=10)
        except Exception as e:
            return await plot_fail(inter, e, "facts")
        rows = r["data"]["facts"]
        if not rows:
            return await say(inter, "No certified facts yet.", ephemeral=False)
        msg = r["text"] + "\n" + facts_text(rows) + "\n-# " + FACT_SYMBOLS

        async def draw(graphic: str):
            kind, kw = fact_graphic(graphic)
            return (await insight(kind, **kw))["files"]

        fv = FactView(inter.user.id, rows, draw)
        await say_long(inter, msg, "facts.txt", view=fv, ephemeral=False)
        try:
            fv.message = await inter.original_response()
        except discord.HTTPException:
            pass

    facts_cmd.autocomplete("archetype")(archetype_name_ac)

    @tree.command(name="rules", description="Model rules in symbols")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(item="filter by item")
    async def rules_cmd(inter: discord.Interaction, item: Optional[str] = None):
        await inter.response.defer(thinking=True)
        try:
            r = await insight("rules", item=item, limit=200)
        except Exception as e:
            return await plot_fail(inter, e, "rules")
        rows = r["data"]["rules"]
        if not rows:
            return await say(inter, "No rules touch that item.", ephemeral=False)
        await say_long(inter, r["text"] + "\n" + rules_text(rows) + "\n-# " + RULE_SYMBOLS, "rules.txt", ephemeral=False)

    rules_cmd.autocomplete("item")(item_name_ac)
