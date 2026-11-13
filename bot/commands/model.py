"""/model: the loaded run (info, item, memory, reload, sync, replay)."""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..embeds import clip, item_text, SEP, sync_text
from ..interact import say
from ..links import canonical_link, parse_build_link

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin
    insight = c.insight
    item_name_ac = c.item_name_ac
    plot_fail = c.plot_fail
    plot_files = c.plot_files
    reviewer = c.reviewer

    model = app_commands.Group(name="model", description="The tri model run", guild_only=True)

    @model.command(name="info", description="Which model run is loaded")
    async def model_info(inter: discord.Interaction):
        s = bot.svc
        st = await asyncio.to_thread(s.state)
        sha = st.get("sync_sha")
        await say(inter, f"run {s.run_id}: {len(s.catalog)} items, ruleset {s.catalog.ruleset}\n"
                         f"review memory: {s.n_memory} reviews" + (f" ({len(s.memory_skipped)} skipped, "
                         "not valid for this run)" if s.memory_skipped else "") + "\n"
                         f"locators: {st.get('n_locators', 0)} in {len(s.archetypes)} archetypes\n"
                         f"categories: {st.get('n_categories', 0)}; sidegrades: {st.get('n_sidegrades', 0)}\n"
                         f"{s.load_text()}\n"
                         + (f"game data: pocamind/data {(sha or '')[:7]}, {st.get('n_unrated', 0)} unrated items"
                            if sha else "game data: as fitted (never synced)"))

    @model.command(name="item", description="An item in the model")
    @app_commands.describe(name="talent or mantra", link="optional build link or id")
    async def model_item(inter: discord.Interaction, name: str, link: Optional[str] = None):
        g = await reviewer(inter)
        if not g:
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        b = None
        if link:
            try:
                bid = parse_build_link(link)
                if g.hide_pending and not bot.is_admin(inter, g) and bid in await bot.pending_ids():
                    return await say(inter, "This build is waiting for review; the model's view is held back.")
                _, parsed = await bot.load_build(bid)
                b = parsed.build
            except Exception as e:
                return await say(inter, f"Could not read that build: {e}")
        x = await bot.model_call(bot.svc.explain_item, name, b)
        await say(inter, item_text(x))

    model_item.autocomplete("name")(item_name_ac)

    @model.command(name="reload", description="Reload the model (follows LATEST)")
    async def model_reload(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        t0 = time.monotonic()
        was = bot.svc.run_id if bot.svc else "none"
        if not getattr(bot.settings, "reload_swap", False) and bot.svc is not None:
            # v3.7: one model in memory: the old one is released before the new one
            # loads (commands answer "no model loaded" meanwhile); TRI_RELOAD_SWAP=1
            # keeps it serving, at twice the memory
            bot.svc = None
            from tri.runtime import release
            release()
        try:
            svc = await asyncio.to_thread(bot.open_service, 0.0, True)
        except Exception as e:
            if bot.svc is None:
                bot.load_error = f"{type(e).__name__}: {e}"
            keep = f"keeping {bot.svc.run_id}" if bot.svc else f"no model loaded (was {was}; /model reload again)"
            return await say(inter, f"Reload failed, {keep}: {e}")
        t_load = time.monotonic() - t0
        if t_load > 60:                                # say something before Discord's 15 minute limit
            try:
                await inter.followup.send(f"Run loaded in {t_load:.0f}s; loading reviews and locators...",
                                          ephemeral=True)
            except discord.HTTPException:
                pass
        # Game data first, so the database reload below validates reviews and
        # categories that name talents added since the run was fitted.
        synced = ""
        if getattr(bot.settings, "reload_sync", True):
            try:
                rep = await bot.sync_gamedata(apply=True, force=True, svc=svc, memory=False)
                if rep.get("applied"):
                    synced = f" Game data: +{len(rep.get('added', []))} items, {len(rep.get('changed', []))} changed."
                elif rep.get("error"):
                    synced = f" Game data sync failed: {clip(str(rep['error']), 200)}."
            except Exception as e:
                log.warning("game data sync on reload failed", exc_info=True)
                synced = f" Game data sync failed: {type(e).__name__}."
        mem = await bot.load_memory(svc)
        old = bot.svc.run_id if bot.svc else was
        bot.svc, bot.load_error = svc, None
        sec = mem.get("seconds", {})
        await say(inter, f"Loaded run {svc.run_id} (was {old}), {mem.get('loaded', 0)} reviews in memory"
                         + (f" ({mem['skipped']} skipped)" if mem.get("skipped") else "")
                         + f", {mem.get('locators', 0)} locators, {mem.get('authored_rules', 0)} authored rules.{synced} "
                         f"Open drafts keep their snapshot. "
                         f"Took {time.monotonic() - t0:.0f}s (run {t_load:.0f}s, Supabase {sec.get('fetch', 0):.0f}s, "
                         f"memory {sec.get('rebuild', 0):.0f}s).")

    @model.command(name="memory", description="Reload db state (no model reload)")
    async def model_memory(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        mem = await bot.load_memory(bot.svc)
        if mem.get("error"):
            return await say(inter, "Could not read the database; the memory is unchanged.")
        sec = mem.get("seconds", {})
        await say(inter, f"Reloaded from the database: {mem.get('loaded', 0)} reviews ({mem.get('skipped', 0)} skipped), "
                         f"{mem.get('locators', 0)} locators, {mem.get('authored_rules', 0)} authored rules"
                         + (f" ({mem['authored_broken']} broken)" if mem.get("authored_broken") else "")
                         + (" (rules unreadable, see log)" if mem.get("authored_error") else "")
                         + f". Took {sec.get('fetch', 0) + sec.get('rebuild', 0):.0f}s.")

    @model.command(name="sync", description="Check pocamind/data for changes (admin)")
    @app_commands.describe(apply="apply (default: dry run)",
                           force="re-read the game data even if the commit has not moved")
    async def model_sync(inter: discord.Interaction, apply: bool = False, force: bool = False):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=not apply, thinking=True)
        try:
            rep = await bot.sync_gamedata(apply=apply, force=force)
        except Exception as e:
            log.exception("model sync")
            return await say(inter, f"Sync failed, nothing changed: {type(e).__name__}: {e}", ephemeral=not apply)
        await say(inter, clip(sync_text(rep, dry_run=not apply), 1990), ephemeral=not apply)
        if apply and rep.get("applied"):
            await bot.announce_sync(rep)

    @model.command(name="replay", description="Review stream replay (admin)")
    async def model_replay(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            r = await insight("replay")
        except Exception as e:
            return await plot_fail(inter, e, "replay")
        d = r["data"]
        f = d.get("final") or {}
        bits = [f"{d['reviews']} reviews"]
        if f.get("accuracy") is not None:
            bits.append(f"accuracy {100 * f['accuracy']:.0f}% (field {100 * (f.get('field') or 0):.0f}%)")
        lr = d.get("lowrank") or {}
        if lr.get("p") is not None and lr["p"] < 0.05 and lr.get("direction"):
            bits.append("shared residual: " + ", ".join(x["item"] for x in lr["direction"][:3]))
        msg = SEP.join(bits)
        if d.get("queue"):
            msg += "\nreview next: " + ", ".join(f"<{canonical_link(str(q['build']))}>" for q in d["queue"][:5])
        await say(inter, clip(msg, 1990), files=plot_files(r["files"]))

    tree.add_command(model)
