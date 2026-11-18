"""Shared helpers of the slash commands: permission checks, autocompletes, insight replies."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from tri.api import archetype_key

from ..embeds import clip
from ..guilds import GuildConfig
from ..interact import Access, say
from ..links import BuildLinkError
from ..texts import NOT_SET_UP

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def make_context(bot: TriBot) -> SimpleNamespace:
    """Permission checks, autocompletes and reply helpers the command modules share."""
    async def reviewer(inter) -> Optional[GuildConfig]:
        g = await bot.config(inter)
        if bot.is_reviewer(inter, g):
            return g
        await say(inter, NOT_SET_UP if not g.configured else "You need a reviewer role for this.")
        return None

    async def analyst(inter, level: str = "user") -> Optional[Access]:
        """Analysis commands. In a tri server: level "user" (analyze, complete, plot)
        passes reviewers, user roles or everyone when analysis is open; level
        "reviewer" (archetype, facts) needs a reviewer. Elsewhere (user-installed
        app): per USER_APP; pending builds are always held back."""
        if bot.in_guild(inter):
            if level == "reviewer":
                g = await reviewer(inter)
            else:
                g = await bot.config(inter)
                if not bot.is_user(inter, g):
                    await say(inter, NOT_SET_UP if not g.configured else "You need a tri user or reviewer role for this.")
                    g = None
            return Access(hide=g.hide_pending and not bot.is_admin(inter, g), g=g) if g else None
        mode = getattr(bot.settings, "user_app", "anyone")
        if mode == "off":
            await say(inter, "tri's analysis commands are only available in tri servers.")
            return None
        if mode == "reviewers" and not await bot.reviewer_anywhere(inter.user.id):
            await say(inter, "Outside tri servers, these commands are for tri reviewers.")
            return None
        return Access(hide=True, external=True)

    async def admin(inter) -> Optional[GuildConfig]:
        g = await bot.config(inter)
        if bot.is_admin(inter, g):
            return g
        await say(inter, "tri admins only (Manage Server or a tri admin role).")
        return None

    def choices(names):
        return [app_commands.Choice(name=n[:100], value=n[:100]) for n in names]

    # ------------------------------------------------------------ autocomplete
    async def ac_in_build(inter: discord.Interaction, current: str):
        d = bot.desk.draft(inter.user.id)
        if d is None:
            return []
        marked = set(d.items_in_ops())
        return choices(bot.svc.search(current, [x for x in d.build_obj().items() if x not in marked]))

    async def ac_related(inter: discord.Interaction, current: str):
        d = bot.desk.draft(inter.user.id)
        if d is None:
            return []
        # build and adds first; NW (missing components) and SG (sidegrade) take any item
        return choices(bot.svc.search(current, list(dict.fromkeys(d.build_obj().items() + d.adds()
                                                                  + list(bot.svc.catalog.items)))))

    async def ac_addable(inter: discord.Interaction, current: str):
        d = bot.desk.draft(inter.user.id)
        if d is None:
            return []
        return choices(bot.svc.search(current, bot.svc.addable(d.build_obj(), exclude=list(d.items_in_ops()))))

    async def ac_marked(inter: discord.Interaction, current: str):
        d = bot.desk.draft(inter.user.id)
        return choices(bot.svc.search(current, list(d.items_in_ops()))) if d else []

    async def insight_fail(inter, e: Exception, what: str):
        if isinstance(e, BuildLinkError):
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        if isinstance(e, ValueError):
            return await say(inter, clip(f"Could not answer: {e}", 1990), ephemeral=False)
        log.exception("insight %s", what)
        return await say(inter, f"Could not answer ({type(e).__name__}).", ephemeral=False)

    # ------------------------------------------------------------ archetypes and facts
    async def insight(kind: str, **kw) -> dict:
        r = await bot.model_call(bot.svc.insight, kind, **kw)
        if kind == "atlas" and (r.get("data") or {}).get("names"):
            bot.atlas_names = list(r["data"]["names"])      # autocomplete when no locator names exist
        return r

    async def send_insight(inter, kind: str, msg: Optional[str] = None, **kw):
        try:
            r = await insight(kind, **kw)
        except Exception as e:
            return await insight_fail(inter, e, kind)
        await say(inter, clip(msg if msg is not None else r.get("text", ""), 1990), ephemeral=False)

    async def archetype_name_ac(inter: discord.Interaction, current: str):
        cur = archetype_key(current)
        known = (bot.svc.archetypes if bot.svc else []) or getattr(bot, "atlas_names", [])
        names = [p for p in known if cur in archetype_key(p)]
        return [app_commands.Choice(name=n[:100], value=n[:100]) for n in names[:25]]

    async def item_name_ac(inter: discord.Interaction, current: str):
        return [app_commands.Choice(name=clip(n, 100), value=clip(n, 100))
                for n in bot.svc.search(current.strip(), bot.svc._names)[:25]] if bot.svc else []

    async def stat_name_ac(inter: discord.Interaction, current: str):
        cur = current.strip().lower()
        return [app_commands.Choice(name=s_, value=s_) for s_ in bot.svc.catalog.stats if cur in s_.lower()][:25] \
            if bot.svc else []

    return SimpleNamespace(
        bot=bot,
        ac_addable=ac_addable,
        ac_in_build=ac_in_build,
        ac_marked=ac_marked,
        ac_related=ac_related,
        admin=admin,
        analyst=analyst,
        archetype_name_ac=archetype_name_ac,
        insight=insight,
        insight_fail=insight_fail,
        item_name_ac=item_name_ac,
        reviewer=reviewer,
        send_insight=send_insight,
        stat_name_ac=stat_name_ac,
    )
