"""/reviews: show, recent and purge stored reviews."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import discord
from discord import app_commands

from ..db import RlsRefused, StoreError
from ..embeds import clip, review_embed
from ..ingest import PURGE_MARKS, purge_plan, with_dependants
from ..interact import say

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


def register(bot: TriBot, c: SimpleNamespace) -> None:
    tree = bot.tree
    admin = c.admin
    archetype_name_ac = c.archetype_name_ac
    item_name_ac = c.item_name_ac
    reviewer = c.reviewer

    # ------------------------------------------------------------ stored reviews
    reviews = app_commands.Group(name="reviews", description="Stored reviews", guild_only=True)

    @reviews.command(name="show", description="Show a stored review by id")
    async def reviews_show(inter: discord.Interaction, id: int):
        if not await reviewer(inter):
            return
        await inter.response.defer(ephemeral=True)
        rows = await bot.store.reviews([id])
        if not rows:
            return await say(inter, f"No review #{id}.")
        body = rows[0]["operations"]
        body = json.loads(body) if isinstance(body, str) else body
        if not isinstance(body, dict):
            return await say(inter, f"Review #{id} was not written by the bot (no build snapshot).")
        await say(inter, embed=review_embed(body, bot.svc, id))

    @reviews.command(name="recent", description="The latest stored reviews")
    async def reviews_recent(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        await inter.response.defer(ephemeral=True)
        lines = []
        for row in await bot.store.recent_reviews(10):
            b = row["operations"]
            b = json.loads(b) if isinstance(b, str) else b
            if not isinstance(b, dict):
                lines.append(f"#{row['id']}  (not written by the bot)")
                continue
            ops = b.get("operations", [])
            n = {a: sum(o["action"] == a for o in ops) for a in ("add", "remove", "iffy")}
            tally = "endorsed" if not ops else f"[+]{n['add']} [-]{n['remove']} [=]{n['iffy']}"
            lines.append(f"#{row['id']}  {tally:<16}  {b.get('build_name') or ''}"[:90])
        await say(inter, ("```\n" + "\n".join(lines) + "\n```\n`/reviews show id` for one") if lines
                  else "No reviews stored yet.")

    @reviews.command(name="purge", description="Undo stored marks on an item across reviews (admin)")
    @app_commands.describe(item="the talent or mantra (* = every item, for the op marks such as strong_sm)",
                           mark="which marks to undo (all: every mark, including keeping it unmarked)",
                           archetype="only reviews that state this archetype (indicator: notes naming it)",
                           reviewer="only this reviewer's reviews",
                           whole="delete the matching reviews outright instead of stripping the marks",
                           having="only reviews of builds holding one of these (comma separated); "
                                  "talents that need them count too",
                           prereqs="also undo the marks on the item's prerequisites (Knight's Rally for Turtle Shell)",
                           confirm="True applies; False (default) only lists what would change")
    @app_commands.choices(mark=[app_commands.Choice(name=v[:100], value=k) for k, v in PURGE_MARKS.items()])
    async def reviews_purge(inter: discord.Interaction, item: str, mark: str = "all", archetype: Optional[str] = None,
                            reviewer: Optional[discord.User] = None, whole: bool = False,
                            having: Optional[str] = None, prereqs: bool = False, confirm: bool = False):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        rid = str(reviewer.id) if reviewer else None
        hv = with_dependants([x.strip() for x in (having or "").split(",") if x.strip()],
                             bot.svc.tri.catalog if bot.svc else None)
        names = [item]
        if prereqs and bot.svc and item.strip() not in ("*", "any"):      # v3.4: the prerequisite chain too
            cat = bot.svc.tri.catalog
            key = {k.casefold(): k for k in cat.items}
            names += cat.prereq_closure(key.get(item.strip().casefold(), item.strip()))
        if whole and item.strip() in ("*", "any"):
            return await say(inter, "`whole` with `*` would delete every matching review; strip the marks instead.")
        try:
            hits = purge_plan(await bot.store.reviews(), names, mark, archetype, rid, having=hv)
        except ValueError as e:
            return await say(inter, str(e))
        except StoreError as e:
            return await say(inter, f"Database error ({e.status}).")
        scope = f"{PURGE_MARKS[mark]} on {', '.join(names)}" + (f" ({archetype})" if archetype else "") + \
            (f" by {reviewer.display_name}" if reviewer else "") + (f" with {', '.join(hv)}" if hv else "")
        if not hits:
            return await say(inter, f"Nothing stored matches: {scope}.")
        lines = "\n".join(f"{h['table'][0]}#{h['id']}  {h['build'][:32]:<32}  " + "; ".join(h["removed"])
                          for h in hits[:20])
        more = f"\n... and {len(hits) - 20} more" if len(hits) > 20 else ""
        verb = "delete" if whole else "strip"
        if not confirm:
            return await say(inter, clip(f"{len(hits)} row(s) hold {scope}:\n```\n{lines}{more}\n```\n"
                                         f"Run again with `confirm: True` to {verb} them. Stripped items become "
                                         f"neutral: the review no longer says anything about {item}.", 1900))
        try:
            if whole:
                gone = await bot.store.delete_reviews([h["id"] for h in hits if h["table"] == "reviews"])
            else:
                gone = await bot.store.update_rows(bot.store.reviews_table,
                                                   {h["id"]: h["body"] for h in hits if h["table"] == "reviews"})
        except RlsRefused:
            return await say(inter, "Row-level security blocks changes with this key (`/pool check`).")
        except StoreError as e:
            return await say(inter, f"Database error ({e.status}).")
        log.info("reviews purge %s by %s (%s): %s", scope, inter.user.id, verb, gone)
        mem = await bot.load_memory(bot.svc)                 # the undone marks leave the model's memory
        tail = " The memory could not be reloaded: run `/model memory`." if mem.get("error") else \
            f" Memory reloaded ({mem.get('loaded', 0)} reviews)."
        await say(inter, clip(f"{'Deleted' if whole else 'Stripped'} {len(gone)} row(s): "
                              + ", ".join(f"#{i}" for i in gone) + "." + tail, 1900))

    reviews_purge.autocomplete("item")(item_name_ac)

    reviews_purge.autocomplete("archetype")(archetype_name_ac)

    tree.add_command(reviews)
