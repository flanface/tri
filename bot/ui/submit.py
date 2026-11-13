"""The confirm button of /submit."""
from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import json
import logging

import discord

from ..db import AlreadyReviewed, StoreError
from ..interact import edit_reply, say
from ..reviews import DraftError

if TYPE_CHECKING:                                   # pragma: no cover
    from ..discord_bot import TriBot

log = logging.getLogger("tri.bot")


class ConfirmSubmit(discord.ui.View):
    def __init__(self, bot: TriBot, user_id: int):
        super().__init__(timeout=600)
        self.bot, self.user_id = bot, user_id
        self._busy = False

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.user_id:
            await say(inter, "Only the reviewer of this build can submit it.")
            return False
        return True

    @discord.ui.button(label="Submit review", style=discord.ButtonStyle.success)
    async def confirm(self, inter: discord.Interaction, _button: discord.ui.Button):
        # Discord forgets an interaction that is not answered within 3 seconds;
        # storing the review can take longer, so acknowledge the click first.
        if self._busy:
            return await inter.response.defer()
        self._busy = True
        await inter.response.defer()
        bot = self.bot
        try:
            d = bot.desk.require(self.user_id)
            body, _ = bot.desk.payload(d, bot.svc.catalog, bot.svc.run_id)
            if d.direct:
                for attempt in range(3):                 # a fresh id; retried if someone takes it first
                    d.pool_id = await bot.store.direct_id()
                    body, _ = bot.desk.payload(d, bot.svc.catalog, bot.svc.run_id)
                    body["direct"] = True
                    try:
                        await bot.store.insert_review(d.pool_id, body)
                        break
                    except AlreadyReviewed:
                        if attempt == 2:
                            raise StoreError("could not allocate a review id", 409)
            else:
                await bot.store.insert_review(d.pool_id, body)
        except DraftError as e:
            self._busy = False
            return await edit_reply(inter, content=str(e), view=None)
        except AlreadyReviewed:
            bot.desk.finish(self.user_id)
            return await edit_reply(inter, content="This build was already reviewed by someone else; "
                                                   "your draft was discarded.", view=None)
        except StoreError as e:
            log.exception("insert_review")
            self._busy = False
            if d.direct:
                d.pool_id = None
            if getattr(e, "code", None) == "23503":     # reviews.id still references pool.id
                return await edit_reply(inter, content="The reviews table still requires a pool entry; run "
                                        "supabase/schema.sql to review builds "
                                        "outside the pool.", view=self)
            return await edit_reply(inter, content=f"Could not store the review ({e.status}); try again.", view=self)
        bot.desk.finish(self.user_id)
        self.stop()
        popped = await bot.pop_reviewed(d.pool_id, d.link)
        what = "endorsed as is" if body["endorsed"] else f"{len(body['operations'])} operations"
        msg = (f"Review stored as direct review #{d.pool_id} ({what})." if d.direct
               else f"Review stored for pool #{d.pool_id} ({what}).") + " `/review` for the next build."
        if popped and popped.get("kept") and not d.direct:
            msg += (" The pool row stays (row-level security: the bot's key cannot delete; `/pool check`)."
                    if popped.get("rls") else " The pool row stays (`/pool check`); it is not served again.")
        await edit_reply(inter, content=msg, view=None)
        try:                                           # after answering: this can take seconds
            from tri.core.codec import review_from_dict
            await asyncio.to_thread(bot.svc.add_review, review_from_dict(json.loads(json.dumps(body))))
        except Exception:
            log.exception("adding review to memory")
        await bot.post_log(inter, body, d.pool_id)

    @discord.ui.button(label="Keep editing", style=discord.ButtonStyle.secondary)
    async def cancel(self, inter: discord.Interaction, _button: discord.ui.Button):
        await edit_reply(inter, content="Not submitted. Your draft is unchanged.", view=None)
