"""The review flow: /review, /add, /remove, /iffy, /note, /draft, /unmark, /undo, /clear, /submit, /skip."""
from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..db import StoreError
from ..embeds import clip, draft_text
from ..format import fit_fields, item_fields, kinds_from, single
from ..interact import say
from ..links import full_link
from ..replies import REPLY_HELP
from ..reviews import DraftError
from ..texts import ADD_REASON_CHOICES, REMOVE_REASON_CHOICES, REVIEW_HELP, STRENGTH_CHOICES
from ..ui.submit import ConfirmSubmit

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    ac_addable = c.ac_addable
    ac_in_build = c.ac_in_build
    ac_marked = c.ac_marked
    ac_related = c.ac_related
    reviewer = c.reviewer

    # ------------------------------------------------------------ review flow
    async def open_sheet(inter, link: str, parsed) -> None:
        mins = int(bot.settings.claim_ttl_min)
        await say(inter, f"{link}\n{REVIEW_HELP}\nReserved for you for {mins} minutes.", ephemeral=False)
        d = bot.desk.require(inter.user.id)
        e = discord.Embed(title=clip(parsed.name, 256), url=link,
                          description="Numbered picks of the build")
        for name, value in fit_fields(item_fields(d.numbered_items())):
            e.add_field(name=name, value=value, inline=True)
        sheet = await bot.post_sheet(inter, inter.user.id,
                                     f"Review sheet for <@{inter.user.id}>: {link}\n{REPLY_HELP}", embed=e)
        if sheet is None:
            await say(inter, "I could not post a review sheet here (Send Messages permission), "
                             "so use the commands.")

    @tree.command(name="review", description="Review a build (link, id or random)")
    @app_commands.guild_only()
    @app_commands.describe(build="build link or id (skips the pool); "
                                 "leave empty for a random pool build")
    async def review(inter: discord.Interaction, build: Optional[str] = None):
        if not await reviewer(inter):
            return
        d = bot.desk.draft(inter.user.id)
        if d is not None:
            return await say(inter, f"You are reviewing {d.link}\n{REVIEW_HELP}", ephemeral=False)
        await inter.response.defer(thinking=True)            # public
        async with bot.claim_lock:
            if build:
                try:
                    bid, parsed = await bot.load_build(build)
                except Exception as e:
                    return await say(inter, f"Could not read that build: {clip(str(e), 300)}", ephemeral=False)
                link = full_link(build, bid)                 # v3.0.2: the link as given, not the bare id
                try:
                    bot.desk.claim(inter.user.id, None, link, parsed, direct=True)
                except DraftError as e:
                    return await say(inter, str(e), ephemeral=False)
                return await open_sheet(inter, link, parsed)
            try:
                pending = await bot.store.pending()
            except StoreError as e:
                log.exception("pending")
                return await say(inter, f"Database error ({e.status}).", ephemeral=False)
            cands = bot.desk.candidates(inter.user.id, pending)
            random.shuffle(cands)                            # a random pool build
            for p in cands:
                try:
                    _, parsed = await bot.load_build(p.link)
                except Exception as e:                       # broken / private / deleted link
                    log.warning("pool %s unreadable: %s", p.id, e)
                    bot.desk.mark_broken(p.id, str(e)[:200])
                    continue
                try:
                    bot.desk.claim(inter.user.id, p.id, p.link, parsed)
                except DraftError:                           # same build claimed directly by someone else
                    continue
                return await open_sheet(inter, p.link, parsed)
        await say(inter, "No builds are waiting for review. `/review build:` reviews any build by link or id.",
                  ephemeral=False)

    async def op(inter, action, item, strength=None, reason=None, related=()):
        if not await reviewer(inter):
            return
        try:
            o = bot.desk.add_op(inter.user.id, action, item, bot.svc, strength, reason, related)
        except DraftError as e:
            return await say(inter, str(e))
        d = bot.desk.draft(inter.user.id)
        await say(inter, f"{single(o, kinds_from(bot.svc))}  ({len(d.ops)} in draft, `/draft` to see all)")

    @tree.command(name="remove", description="[-] Mark an item for removal")
    @app_commands.guild_only()
    @app_commands.describe(item="talent or mantra in the build", strength="how sure you are",
                           reason="why it should go",
                           related="the other item (redundant with / works against; NW: a missing component)",
                           related2="a second related item")
    @app_commands.choices(strength=STRENGTH_CHOICES, reason=REMOVE_REASON_CHOICES)
    @app_commands.autocomplete(item=ac_in_build, related=ac_related, related2=ac_related)
    async def remove(inter: discord.Interaction, item: str, strength: app_commands.Choice[str],
                     reason: app_commands.Choice[str], related: Optional[str] = None, related2: Optional[str] = None):
        await op(inter, "remove", item, strength.value, reason.value, [x for x in (related, related2) if x])

    @tree.command(name="add", description="[+] Mark an item to add")
    @app_commands.guild_only()
    @app_commands.describe(item="talent or mantra available at these stats", strength="how sure you are",
                           reason="default: necessary for the archetype (synergises with, if related is set)",
                           related="synergises with: the pick it works with")
    @app_commands.choices(strength=STRENGTH_CHOICES, reason=ADD_REASON_CHOICES)
    @app_commands.autocomplete(item=ac_addable, related=ac_related)
    async def add(inter: discord.Interaction, item: str, strength: app_commands.Choice[str],
                  reason: Optional[app_commands.Choice[str]] = None, related: Optional[str] = None):
        await op(inter, "add", item, strength.value, reason.value if reason else None, [related] if related else [])

    @tree.command(name="iffy", description="[=] Mark an item iffy")
    @app_commands.guild_only()
    @app_commands.describe(sidegrade_to="it is a sidegrade to this item",
                           strength="with sidegrade_to: weak = under this archetype, strong = generally")
    @app_commands.choices(strength=STRENGTH_CHOICES)
    @app_commands.autocomplete(item=ac_in_build, sidegrade_to=ac_related)
    async def iffy(inter: discord.Interaction, item: str, sidegrade_to: Optional[str] = None,
                   strength: Optional[app_commands.Choice[str]] = None):
        if sidegrade_to:
            await op(inter, "iffy", item, strength.value if strength else "weak", "sidegrade", [sidegrade_to])
        else:
            await op(inter, "iffy", item)

    @tree.command(name="note", description="Add a note to your draft")
    @app_commands.guild_only()
    @app_commands.describe(text="Gank: Item, Item  |  any comment")
    async def note(inter: discord.Interaction, text: str):
        if not await reviewer(inter):
            return
        from .replies import Line, _apply_note
        try:
            msg = _apply_note(bot.desk, inter.user.id, Line(0, text, "note", note=text.lstrip("#/ ").strip()), bot.svc)
        except DraftError as e:
            return await say(inter, str(e))
        await say(inter, msg, ephemeral=False)

    @tree.command(name="draft", description="Show your current review draft")
    @app_commands.guild_only()
    async def draft(inter: discord.Interaction):
        try:
            d = bot.desk.require(inter.user.id)
        except DraftError as e:
            return await say(inter, str(e))
        await say(inter, draft_text(d, bot.svc))

    @tree.command(name="unmark", description="Drop one item's op from your draft")
    @app_commands.guild_only()
    @app_commands.autocomplete(item=ac_marked)
    async def unmark(inter: discord.Interaction, item: str):
        try:
            o = bot.desk.remove_item(inter.user.id, bot.svc.resolve(item) or item)
        except DraftError as e:
            return await say(inter, str(e))
        await say(inter, f"Unmarked {single(o, kinds_from(bot.svc))}" if o else f"{item} is not marked.")

    @tree.command(name="undo", description="Undo the last draft op")
    @app_commands.guild_only()
    async def undo(inter: discord.Interaction):
        try:
            o = bot.desk.undo(inter.user.id)
        except DraftError as e:
            return await say(inter, str(e))
        await say(inter, f"Undone {single(o, kinds_from(bot.svc))}" if o else "Your draft is already empty.")

    @tree.command(name="clear", description="Clear your draft")
    @app_commands.guild_only()
    async def clear(inter: discord.Interaction):
        try:
            bot.desk.clear(inter.user.id)
        except DraftError as e:
            return await say(inter, str(e))
        await say(inter, "Draft cleared.")

    @tree.command(name="submit", description="Submit your review")
    @app_commands.guild_only()
    async def submit(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        try:
            d = bot.desk.require(inter.user.id)
            _, warnings = bot.desk.review(d, bot.svc.catalog)
        except DraftError as e:
            return await say(inter, str(e))
        msg = draft_text(d, bot.svc, 1500)
        if warnings:
            msg += "\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings)
        await say(inter, clip(msg, 1990), view=ConfirmSubmit(bot, inter.user.id))

    @tree.command(name="skip", description="Release your claimed build")
    @app_commands.guild_only()
    async def skip(inter: discord.Interaction):
        d = bot.desk.release(inter.user.id, skip=True)
        await say(inter, "Released. `/review` gives you a different build." if d else "You have no build claimed.")
