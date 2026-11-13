"""/pool: the review pool (add, remove, list, status, retry, check, prune)."""
from __future__ import annotations

from typing import TYPE_CHECKING
from types import SimpleNamespace

import discord
from discord import app_commands

from ..db import RlsRefused, StoreError
from ..embeds import clip, pool_check_text
from ..interact import say
from ..links import BuildLinkError, canonical_link

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin

    # ------------------------------------------------------------ admin
    pool = app_commands.Group(name="pool", description="Manage the review pool", guild_only=True)

    @pool.command(name="add", description="Add a build link or dwt.sh folder to the pool")
    @app_commands.describe(link="build link or id, or dwt.sh folder")
    async def pool_add(inter: discord.Interaction, link: str):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            ids = await bot.fetcher.expand(link)
        except BuildLinkError as e:
            return await say(inter, f"Not added: {e}")
        if not ids:
            return await say(inter, "No builds in that link.")
        lines, added = [], 0
        for bid in ids:                                     # one or many (dwt.sh drive folders, v2.23)
            try:
                bid, parsed = await bot.load_build(bid)
            except Exception as e:
                lines.append(f"not added, {bid} could not be read: {e}")
                continue
            url = canonical_link(bid)
            try:
                if await bot.store.find_link(url):
                    lines.append(f"{url}: already in the pool")
                    continue
                rid = await bot.store.is_reviewed(url)
                if rid is not None:
                    lines.append(f"{url}: already reviewed (review pool-{rid})")
                    continue
                p = await bot.store.add_link(url)
            except StoreError as e:
                return await say(inter, clip("\n".join(lines + [f"Database error ({e.status}): {e}"]), 1900))
            added += 1
            notes = parsed.notes() if len(ids) == 1 else []
            lines.append(f"Added #{p.id}: **{parsed.name}** {url}" + ("\n" + "\n".join(notes) if notes else ""))
        head = f"{added} of {len(ids)} builds added.\n" if len(ids) > 1 else ""
        await say(inter, clip(head + "\n".join(lines), 1900))

    @pool.command(name="remove", description="Remove a pool entry by id")
    async def pool_remove(inter: discord.Interaction, id: int):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        try:
            if await bot.store.has_review(id):
                return await say(inter, "That entry has a review; delete the review row first.")
            ok = await bot.store.remove_link(id)
        except RlsRefused as e:
            return await say(inter, f"Not removed: {e}. `/pool check`.")
        except StoreError as e:
            return await say(inter, f"Database error ({e.status}).")
        await say(inter, f"Removed #{id}." if ok else f"No entry #{id}.")

    @pool.command(name="list", description="List pending pool entries")
    async def pool_list(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        pending = await bot.store.pending()
        claimed = bot.desk.claimed_ids()
        lines = [f"#{p.id} {p.link}" + (" (claimed)" if p.id in claimed else "")
                 + (f" (unreadable: {bot.desk.broken[p.id][:60]})" if p.id in bot.desk.broken else "")
                 for p in pending[:40]]
        more = f"\n... and {len(pending) - 40} more" if len(pending) > 40 else ""
        await say(inter, clip("\n".join(lines) + more, 1990) if lines else "Nothing pending.")

    @pool.command(name="status", description="Pool and review counts")
    async def pool_status(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        c = await bot.store.counts()
        await say(inter, f"{c['pending']} pending · {len(bot.desk.claimed_ids())} claimed · "
                         f"{len(bot.desk.broken)} unreadable · {c['reviews']} reviews"
                         + (f" · {c['done_in_pool']} done but still in the pool (`/pool prune`)" if c["done_in_pool"] else ""))

    @pool.command(name="check", description="Database key permissions")
    async def pool_check(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        try:
            c = await bot.store.check()
        except Exception as e:
            return await say(inter, f"Database unreachable: {type(e).__name__}: {clip(str(e), 300)}")
        await say(inter, pool_check_text(c))

    @pool.command(name="prune", description="Delete reviewed pool rows")
    async def pool_prune(inter: discord.Interaction):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True)
        try:
            r = await bot.store.prune()
        except StoreError as e:
            return await say(inter, f"Database error ({e.status}).")
        msg = f"Removed {len(r['popped'])} rows."
        if r.get("rls"):
            msg += f" Kept {len(r['kept'])}: row-level security blocks deletes with this key (`/pool check`)."
        elif r["kept"]:
            msg += f" Kept {len(r['kept'])}" + (" (reviews.id still references pool.id: run "
                                               "supabase/schema.sql)" if r["blocked"] else "") + \
                "; they are not served either way."
        await say(inter, msg)

    @pool.command(name="retry", description="Clear failed pool links")
    async def pool_retry(inter: discord.Interaction):
        if not await admin(inter):
            return
        n = len(bot.desk.broken)
        bot.desk.broken.clear()
        bot.desk.save()
        await say(inter, f"Cleared {n} unreadable marks.")

    tree.add_command(pool)
