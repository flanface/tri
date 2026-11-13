"""/setup and /settings: server roles and analysis settings."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..db import StoreError
from ..embeds import config_embed
from ..guilds import GuildConfig
from ..interact import say
from ..ui.setup import open_setup

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin

    # ------------------------------------------------------------ setup
    @tree.command(name="setup", description="Server roles for tri")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    async def setup(inter: discord.Interaction):
        await open_setup(bot, inter)

    @tree.command(name="settings", description="Analysis settings for this server (admin)")
    @app_commands.guild_only()
    @app_commands.describe(caution="ask questions, only hold thin marks back, or show everything",
                           max_questions="most questions before an answer (0: every necessary one)",
                           question_timeout="seconds to answer a question sheet before tri posts anyway",
                           private_replies="analysis replies only the invoker sees",
                           verbose_default="/analyze opens the full page by default",
                           placement_inset="attach the concept-space inset",
                           record_answers="store answers (only a consensus of many users reaches the model)",
                           open_analysis="anyone may analyze, complete and plot")
    @app_commands.choices(caution=[app_commands.Choice(name="ask", value="ask"),
                                   app_commands.Choice(name="hold", value="hold"),
                                   app_commands.Choice(name="off", value="off")])
    async def settings_cmd(inter: discord.Interaction, caution: Optional[str] = None,
                           max_questions: Optional[app_commands.Range[int, 0, 25]] = None,
                           question_timeout: Optional[app_commands.Range[int, 30, 1800]] = None,
                           private_replies: Optional[bool] = None, verbose_default: Optional[bool] = None,
                           placement_inset: Optional[bool] = None, record_answers: Optional[bool] = None,
                           open_analysis: Optional[bool] = None):
        g = await admin(inter)
        if not g:
            return
        ch = dict(caution=caution, max_questions=max_questions, question_timeout=question_timeout,
                  private_replies=private_replies, verbose_default=verbose_default, placement_inset=placement_inset,
                  record_answers=record_answers, open_analysis=open_analysis)
        ch = {k: v for k, v in ch.items() if v is not None}
        if not ch:
            return await say(inter, embed=config_embed(g, inter.guild))
        g2 = GuildConfig.from_dict(g.guild_id, dict(g.to_dict(), **ch))
        await inter.response.defer(ephemeral=True)
        try:
            await bot.guild_cfg.put(g2, inter.user.id)
        except StoreError as e:
            return await say(inter, f"Could not save ({e.status}). Nothing changed.")
        await say(inter, "Saved: " + ", ".join(f"{k} = {v}" for k, v in ch.items()), embed=config_embed(g2, inter.guild))
