"""/locator: archetype locators (add, remove, list, check, suggest)."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict

import discord
from discord import app_commands

from tri.reviews.locators import archetype_key

from ..db import StoreError
from ..embeds import clip, locator_text, quality_lines, suggest_text
from ..interact import say, say_long
from ..links import BuildLinkError, canonical_link

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin
    insight = c.insight
    plot_fail = c.plot_fail
    plot_files = c.plot_files

    locator = app_commands.Group(name="locator", description="Archetype locators",
                                 guild_only=True)

    @locator.command(name="add", description="Label a build's archetype (admin)")
    @app_commands.describe(link="build link or id, or dwt.sh folder",
                           archetype="what the build is for, e.g. ganking, PvE tank (reuse names to group builds)")
    async def locator_add(inter: discord.Interaction, link: str, archetype: str):
        if not await admin(inter):
            return
        archetype = " ".join(archetype.split())
        if not archetype:
            return await say(inter, "Give the build an archetype.")
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            ids = await bot.fetcher.expand(link)
        except BuildLinkError as e:
            return await say(inter, f"Not added, the build could not be read: {e}")
        if not ids:
            return await say(inter, "No builds in that link.")
        same = {archetype_key(p): p for p in bot.svc.archetypes}.get(archetype_key(archetype))
        archetype = same or archetype                       # one spelling per archetype
        lines, done = [], 0
        for one in ids:                                     # one or many (dwt.sh drive folders, v2.23)
            try:
                bid, rec = await bot.fetcher.fetch_record(one)
                parsed = await bot.model_call(bot.svc.parse, bid, rec)
            except Exception as e:
                lines.append(f"not added, {one} could not be read: {e}")
                continue
            url = canonical_link(bid)
            if isinstance(rec, dict):                       # the raw dwt.sh document (cosmetics) is not stored
                rec = {k: v for k, v in rec.items() if k != "dwt"}
            try:
                row = await bot.store.insert_locator(url, archetype, rec, str(inter.user.id))
            except StoreError as e:
                if e.status == 404 or e.code in ("PGRST205", "42P01"):
                    return await say(inter, "The locators table does not exist yet: run supabase/schema.sql.")
                return await say(inter, clip("\n".join(lines + [f"Database error ({e.status}): {e}"]), 1900))
            row.setdefault("build_id", bid)
            try:
                loc = await asyncio.to_thread(bot.svc.add_locator, row)
            except Exception as e:
                lines.append(f"stored #{row.get('id')}, but this run cannot read it: {e}")
                continue
            done += 1
            lines.append(f"Locator #{loc.locator_id}: **{parsed.name}** {url}")
        n = sum(archetype_key(l.archetype) == archetype_key(archetype)
                for l in await bot.model_call(bot.svc.locator_list))
        head = (f"{done} of {len(ids)} builds labelled **{archetype}**. " if len(ids) > 1 else "")
        tail = (f" is **{archetype}**" if len(ids) == 1 and done else "")
        msg = head + "\n".join(lines) + tail + (f"\n{n} locator{'s' if n != 1 else ''} for this archetype. "
                                               "Effective now, no refit.")
        if n == 1:
            msg += "\nAdd a few more builds of this archetype so it can be checked."
        await say(inter, clip(msg, 1900))

    @locator_add.autocomplete("archetype")
    async def locator_archetype_ac(inter: discord.Interaction, current: str):
        cur = archetype_key(current)
        names = [p for p in bot.svc.archetypes if cur in archetype_key(p)] if bot.svc else []
        if current.strip() and archetype_key(current) not in {archetype_key(p) for p in names}:
            names = [" ".join(current.split())] + names
        return [app_commands.Choice(name=n[:100], value=n[:100]) for n in names[:25]]

    @locator.command(name="remove", description="Remove a locator (admin)")
    async def locator_remove(inter: discord.Interaction, id: int):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        try:
            ok = await bot.store.remove_locator(id)
        except StoreError as e:
            return await say(inter, f"Database error ({e.status}).")
        await asyncio.to_thread(bot.svc.remove_locator, id)
        await say(inter, f"Removed locator #{id}." if ok else f"No locator #{id}.")

    @locator.command(name="list", description="Locators by archetype")
    async def locator_list(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        by: Dict[str, list] = {}
        for l in await bot.model_call(bot.svc.locator_list):
            by.setdefault(l.archetype, []).append(l)
        if not by:
            return await say(inter, "No locators yet. `/locator add link archetype` labels a build.")
        lines = []
        for p in sorted(by, key=str.casefold):
            ls = by[p]
            lines.append(f"**{p}** ({len(ls)}): " + ", ".join(f"#{l.locator_id}" for l in ls[:20])
                         + (" ..." if len(ls) > 20 else ""))
        if bot.svc.locators_skipped:
            lines.append(f"{len(bot.svc.locators_skipped)} stored locators cannot be read by this run.")
        await say(inter, clip("\n".join(lines), 1990))

    @locator.command(name="check", description="Locator separation in the concept space")
    async def locator_check(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        rep = await bot.model_call(bot.svc.locator_report)
        text = locator_text(rep)
        try:
            q = (await bot.model_call(bot.svc.insight, "quality"))["data"]
            text += "\n" + "\n".join(quality_lines(q))
        except Exception:
            log.warning("categorization quality failed", exc_info=True)
        await say_long(inter, text, "locators.txt")

    @locator.command(name="suggest", description="Unnamed archetype candidates (admin)")
    async def locator_suggest(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            r = await insight("suggest")
        except Exception as e:
            return await plot_fail(inter, e, "suggest")
        rows = r["data"]["candidates"]
        if not rows:
            return await say(inter, f"No candidates among {r['data']['unplaced']} unplaced builds.")
        await say(inter, clip(suggest_text(rows), 1990), files=plot_files(r["files"]))

    tree.add_command(locator)
