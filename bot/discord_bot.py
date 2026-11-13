"""Discord front end for tri: collecting structured reviews and applying the model.

Each server is set up once in Discord: when the bot joins, it posts a "Set up tri"
button, and /setup opens the same panel. The panel sets reviewer roles (or
everyone), tri admin roles, a channel where submitted reviews are posted, and
whether the model's view of pending builds is hidden. No ids go in .env.

Reviewers
    /review                         claim the oldest pending build; you get its link only,
                                    and a review sheet you can reply to instead of using commands
    /remove item strength reason [related] [related2]
    /add item strength [reason]
    /iffy item
    /draft  /unmark item  /undo  /clear
    /submit                         store the review (confirm button); nothing marked = endorsed
    /skip                           release the build without reviewing it
    /analyze link                   apply the current model to any build
    /plot concept [builds]  /plot worth [items] [build]   draw the concept / worth space
    /reviews show id  /reviews recent
Admins (Manage Server, or a tri admin role from /setup)
    /setup (Manage Server only)
    /pool add link  /pool remove id  /pool list  /pool status  /pool retry
    /model info  /model item name [link]  /model reload
    /locator add link archetype       label a build with its archetype; effective at once, no refit
    /locator remove id  /locator list  /locator check

Operations are shown as
    [+] added item     strength  reason
    [-] removed item   strength  reason
    [=] iffy item
Reply reviews (bot/replies.py): reply to the review sheet with one line per change,
    + Item s NP   - Item w RW > Other   = Item   x 1-3   v rest   submit
Replies are ephemeral except the review log posts and the /analyze, /complete
and /review (claim prompt) results, which are public in the channel.
Locators are builds an administrator labelled with an archetype; they name archetypes
in the concept space and are never trained on (tri/locators.py).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Sequence

import discord
from discord import app_commands

from tri.reviews.locators import archetype_key

from .color import paint_embed, paint_msg
from .commands import register
from .config import Settings
from .db import StoreError, SupabaseStore
from .embeds import clip, digest_text, draft_text, review_embed, sync_text
from .guilds import GuildConfig, GuildConfigStore
from .links import BuildFetcher, BuildLinkError, parse_build_link
from .questions import parse_answers, render_sheet, Session
from .replies import apply_reply
from .reviews import DraftError, ReviewDesk
from .texts import LOADING_MSG, NO_PINGS, USER_APP_COMMANDS, user_install_url
from .ui.setup import WelcomeView
from .ui.submit import ConfirmSubmit


def ModelService(run: str):
    """The in-process engine, imported on first use."""
    from .service import ModelService as _MS
    return _MS(run)


log = logging.getLogger("tri.bot")


class ModelBusy(RuntimeError):
    """The model stayed busy (a reload, a review rebuild, a game data sync) for
    the whole command timeout."""

    def __init__(self):
        super().__init__("tri is busy updating its model (reviews, game data or a reload); try again in a minute.")


class TriTree(app_commands.CommandTree):
    """Command tree that answers 'still loading' until the model is in memory."""

    async def on_error(self, inter: discord.Interaction, error: app_commands.AppCommandError) -> None:
        orig = getattr(error, "original", error)
        # Log it and still answer, so the user never waits on
        # "thinking..." forever (a deferred interaction needs a followup).
        log.error("command %s failed", getattr(inter.command, "qualified_name", "?"), exc_info=orig)
        if isinstance(orig, ModelBusy):
            msg = str(orig)
        elif isinstance(orig, (asyncio.TimeoutError, TimeoutError)):
            msg = "tri took too long to answer; try again in a minute."
        else:
            msg = clip(f"Something went wrong: {type(orig).__name__}: {orig}", 1900)
        try:
            if inter.response.is_done():
                await inter.followup.send(msg, ephemeral=True)
            else:
                await inter.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        bot = self.client
        if getattr(bot, "svc", None) is not None:
            return True
        cmd = inter.command.qualified_name if inter.command else ""
        if cmd == "model reload" and getattr(bot, "load_error", None):
            return True                                          # lets an admin retry a failed load
        if inter.type == discord.InteractionType.autocomplete:
            return False
        err = getattr(bot, "load_error", None)
        msg = f"tri could not load its model: {err}" if err else LOADING_MSG
        try:
            await inter.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass
        return False


class TriBot(discord.Client):
    def __init__(self, settings: Settings, svc: Optional[ModelService] = None,
                 store: Optional[SupabaseStore] = None, fetcher: Optional[BuildFetcher] = None,
                 desk: Optional[ReviewDesk] = None, guilds: Optional[GuildConfigStore] = None):
        intents = discord.Intents.default()          # guild messages: replies to the review sheet
        intents.message_content = bool(getattr(settings, "message_content", False))   # privileged, opt-in
        super().__init__(intents=intents)
        self.settings = settings
        self.tree = TriTree(self)
        self.svc = svc
        self.load_error: Optional[str] = None
        self._load_task: Optional[asyncio.Task] = None
        self.store = store or SupabaseStore(settings.supabase_url, settings.supabase_key,
                                            settings.pool_table, settings.reviews_table,
                                            settings.locators_table,
                                            values_table=getattr(settings, "values_table", "item_values"),
                                            rules_table=getattr(settings, "rules_table", "tri_rules"))
        if fetcher is None:
            from .dwt import DwtClient
            fetcher = BuildFetcher(dwt=DwtClient(getattr(settings, "dwt_api_key", None),
                                                 getattr(settings, "dwt_folder", "tri"),
                                                 getattr(settings, "dwt_visibility", "unlisted"),
                                                 getattr(settings, "dwt_api", "https://api.dwt.sh")))
        self.fetcher = fetcher
        self.desk = desk or ReviewDesk(settings.state_path, settings.claim_ttl_min * 60)
        self.questions: Dict[int, Session] = {}          # open question sheets by message id (v3.1.1)
        self.guild_cfg = guilds or GuildConfigStore(self.store, settings.guild_config_path,
                                                 settings.guild_config_store, settings.guilds_table)
        self.claim_lock = asyncio.Lock()
        self._synced: set = set()
        self._sync_task: Optional[asyncio.Task] = None
        self.sync_lock = asyncio.Lock()
        self._reviewer_cache: Dict[int, tuple] = {}
        register(self)

    async def setup_hook(self):
        # Nothing slow here: discord.py connects to the gateway only after
        # setup_hook returns. The model loads in the background; commands answer
        # "still loading" until it is ready (TriTree.interaction_check).
        self.add_view(WelcomeView(self))                         # the join message's button survives restarts
        if getattr(self.settings, "health_port", None):
            await self.start_health(self.settings.health_port)
        base = getattr(self.store, "base", None)
        if base:
            log.info("supabase REST base %s (tables %s, %s)", base, getattr(self.store, "pool_table", "?"),
                     getattr(self.store, "reviews_table", "?"))
        if self.settings.sync_mode == "global":
            log.info("syncing global commands")
            await self.tree.sync()
        if self.svc is None:
            self._load_task = asyncio.create_task(self.load_model())
        else:
            await self.load_memory(self.svc)
            self.start_sync_loop()

    # ------------------------------------------------------------ hosting
    async def start_health(self, port: int) -> None:
        """GET / and /health on `port`: for hosts that run the bot as a web
        service (Render free plan). 200 once logged in, with the model status."""
        from aiohttp import web

        async def health(_req):
            svc = self.svc
            body = dict(ok=True, discord=self.is_ready(), model=svc.run_id if svc else None,
                        loading=svc is None and self.load_error is None, error=self.load_error)
            return web.json_response(body)
        app = web.Application()
        app.router.add_get("/", health)
        app.router.add_get("/health", health)
        self._health = web.AppRunner(app, access_log=None)
        await self._health.setup()
        await web.TCPSite(self._health, "0.0.0.0", int(port)).start()
        log.info("health endpoint on port %s", port)

    # ------------------------------------------------------------ model source
    def model_source(self) -> str:
        return f"run {self.settings.tri_run}"

    def open_service(self, wait: float = 0.0, reload: bool = False):
        """The model: an in-process ModelService for TRI_RUN (LATEST is followed on reload).
        TRI_RUN_REMOTE: the run is first brought up to date from Supabase Storage
        (bot.runstore; only changed files are downloaded)."""
        remote = getattr(self.settings, "run_remote", "")
        if remote:
            from .runstore import RunStore, RunStoreError, fetch
            try:
                fetch(self.settings.tri_run, RunStore(self.settings.supabase_url, self.settings.supabase_key),
                      remote, log=log.info)
            except (RunStoreError, OSError) as e:          # a stored copy on disk still loads
                log.warning("run download failed (%s); loading what is on disk", e)
        return ModelService(self.settings.tri_run)

    # ------------------------------------------------------------ game data sync
    def start_sync_loop(self) -> None:
        """Passive sync: every GAMEDATA_SYNC_HOURS, check pocamind/data and extend
        the model to new talents and mantras (unrated, no refit). 0 = off."""
        self.start_digest_loop()
        hours = float(getattr(self.settings, "gamedata_sync_hours", 0) or 0)
        if hours <= 0 or (self._sync_task and not self._sync_task.done()):
            return
        self._sync_task = asyncio.create_task(self._sync_loop(hours))

    async def _sync_loop(self, hours: float) -> None:
        await asyncio.sleep(60)                                   # let the bot settle after login
        while True:
            try:
                rep = await self.sync_gamedata(apply=True)
                if rep.get("applied"):
                    await self.announce_sync(rep)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("game data sync")
            await asyncio.sleep(hours * 3600)

    async def sync_gamedata(self, apply: bool = True, force: bool = False, svc=None, memory: bool = True) -> dict:
        """Check pocamind/data and (apply=True) extend the model to it.
        {head, unchanged, plan summary, applied, ...}. svc: a service that is not
        live yet (/model reload syncs the new run before it takes over);
        memory=False leaves the database reload to the caller."""
        own = svc is not None
        svc = svc or self.svc
        if svc is None:
            return dict(error="no model loaded")
        async with self.sync_lock:
            ref = getattr(self.settings, "gamedata_ref", "main") or "main"
            head, plan = await asyncio.to_thread(svc.sync_plan, ref, force)
            if plan is None:
                return dict(sha=head.get("sha"), date=head.get("date"), unchanged=True)
            rep = dict(plan.summary(), date=head.get("date"), applied=False)
            if apply and (own or svc is self.svc):
                out = await asyncio.to_thread(svc.sync_apply, plan)
                rep.update(out, applied=not plan.empty())
                if not plan.empty():
                    log.info("game data sync %s: +%d changed %d retired %d", (plan.sha or "")[:7],
                             len(plan.added), len(plan.changed), len(plan.removed))
                    if memory:
                        await self.load_memory(svc)  # reviews naming new items now validate
            return rep

    async def announce_sync(self, rep: dict) -> None:
        """Post an applied sync to every configured review log channel."""
        text = sync_text(rep)
        for guild in self.guilds_list():
            try:
                g = await self.guild_cfg.get(guild.id)
                ch = guild.get_channel(g.log_channel_id) if g.log_channel_id else None
                if ch is not None:
                    await ch.send(clip(text, 1990), allowed_mentions=NO_PINGS)
            except Exception:
                log.warning("could not post the sync summary in %s", getattr(guild, "id", "?"))

    # ------------------------------------------------------------ fact digest
    def start_digest_loop(self) -> None:
        """Every FACTS_DIGEST_DAYS, post the facts certified and retired since the
        last digest to every review log channel. 0 = off."""
        days = float(getattr(self.settings, "facts_digest_days", 0) or 0)
        if days <= 0 or (getattr(self, "_digest_task", None) and not self._digest_task.done()):
            return
        self._digest_task = asyncio.create_task(self._digest_loop(days))

    async def _digest_loop(self, days: float) -> None:
        import time as _t
        await asyncio.sleep(120)
        while True:
            since = _t.time() - days * 86400
            try:
                if self.svc is not None:
                    r = await self.model_call(self.svc.insight, "digest", since=since)
                    text = digest_text(r["data"])
                    if text:
                        await self.announce(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("fact digest")
            await asyncio.sleep(days * 86400)

    async def announce(self, text: str) -> None:
        """Post a message to every configured review log channel."""
        for guild in self.guilds_list():
            try:
                g = await self.guild_cfg.get(guild.id)
                ch = guild.get_channel(g.log_channel_id) if g.log_channel_id else None
                if ch is not None:
                    await ch.send(clip(text, 1990), allowed_mentions=NO_PINGS)
            except Exception:
                log.warning("could not post in %s", getattr(guild, "id", "?"))

    async def load_model(self) -> None:
        """Load the run and the stored reviews off the login path, logging each step."""
        import time
        try:
            t0 = time.time()
            log.info("loading model %s", self.model_source())
            svc = await asyncio.to_thread(self.open_service, getattr(self.settings, "api_wait", 0))
            log.info("model run %s loaded in %.1fs, %d items", svc.run_id, time.time() - t0, len(svc.catalog))
            t1 = time.time()
            await self.load_memory(svc)                          # reviews act at once; the bot never refits by itself
            log.info("reviews and locators applied in %.1fs", time.time() - t1)
            self.svc = svc
            log.info("tri ready (%.1fs after login)", time.time() - t0)
            self.start_sync_loop()
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            log.exception("loading the model failed; commands will report it. Fix TRI_RUN "
                          "and restart, or use /model reload")

    async def on_ready(self):
        perms = discord.Permissions(view_channel=True, send_messages=True, embed_links=True,
                                    read_message_history=True)     # replies to review sheets
        log.info("logged in as %s in %d servers. Invite: %s", self.user, len(self.guilds_list()),
                 discord.utils.oauth_url(self.application_id, permissions=perms,
                                         scopes=("bot", "applications.commands")))
        if getattr(self.settings, "pool_pop", True) and not getattr(self, "_pruned", False):
            self._pruned = True                                  # once: pop rows reviewed before v2.9
            asyncio.create_task(self._startup_prune())
        if self.user_app_on():
            log.info("user app (/%s anywhere): %s", ", /".join(USER_APP_COMMANDS), user_install_url(self.application_id))
        if self.settings.sync_mode == "guild":
            await self.sync_user_app()                           # user-app commands global, the rest per guild
            for g in self.guilds_list():
                await self.sync_guild(g)

    async def _startup_prune(self) -> None:
        try:
            r = await self.store.prune()
        except Exception:
            log.exception("pruning the pool")
            return
        if r["popped"] or r["kept"]:
            log.info("pool: removed %d reviewed rows, kept %d%s", len(r["popped"]), len(r["kept"]),
                     " (run supabase/schema.sql to allow deleting reviewed rows)"
                     if r["blocked"] else "")

    def user_app_on(self) -> bool:
        return getattr(self.settings, "user_app", "anyone") != "off"

    def user_app_payload(self) -> list:
        """Global command payload in guild sync mode: only the user-app commands
        (they must be global to be user-installable); the rest stay per guild."""
        if not self.user_app_on():
            return []
        return [c.to_dict(self.tree) for c in self.tree.get_commands() if c.name in USER_APP_COMMANDS]

    async def sync_user_app(self) -> None:
        """Guild sync mode: publish the user-app commands globally (or clear the
        global list when USER_APP=off). Stale global copies of other commands go."""
        try:
            await self.http.bulk_upsert_global_commands(self.application_id, self.user_app_payload())
        except discord.HTTPException:
            log.exception("syncing global user-app commands")

    def guilds_list(self):
        return list(self.guilds)

    async def sync_guild(self, guild: discord.abc.Snowflake):
        if self.settings.sync_mode != "guild" or guild.id in self._synced:
            return
        try:
            self.tree.copy_global_to(guild=guild)
            if self.user_app_on():                               # those are global already: no duplicates
                for name in USER_APP_COMMANDS:
                    self.tree.remove_command(name, guild=guild)
            await self.tree.sync(guild=guild)
            self._synced.add(guild.id)
        except discord.HTTPException:
            log.exception("sync to guild %s", guild.id)

    async def on_guild_join(self, guild: discord.Guild):
        await self.sync_guild(guild)
        g = await self.guild_cfg.get(guild.id)
        if g.configured:
            return
        ch = guild.system_channel
        if ch is None or not ch.permissions_for(guild.me).send_messages:
            ch = next((c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None)
        if ch is not None:
            try:
                await ch.send("tri collects structured build reviews and applies a trained model to builds. "
                              "Someone with Manage Server needs to set it up first.", view=WelcomeView(self),
                              allowed_mentions=NO_PINGS)
            except discord.HTTPException:
                log.warning("could not post the setup message in %s", guild.id)

    async def on_guild_remove(self, guild: discord.Guild):
        self.guild_cfg.forget(guild.id)
        self._synced.discard(guild.id)

    async def load_memory(self, svc: ModelService) -> dict:
        """Load every stored review into the model's review memory, so answers
        follow the reviews immediately (newest wins on similar builds).
        No refit: rules and the field change when you run tri.cli.fit and /model reload."""
        from .ingest import fetch_reviews
        log.info("fetching stored reviews and locators from Supabase")
        t0 = time.monotonic()
        try:
            reviews, problems = await asyncio.wait_for(fetch_reviews(self.store), 60)
        except Exception:
            log.exception("loading reviews into memory")
            r = dict(loaded=0, skipped=0, locators=0, error=True)
            r.update(await self.load_rules(svc))           # v3.0.3: rules do not depend on the reviews
            return r
        try:
            rows = await asyncio.wait_for(self.store.locators(), 60)
        except Exception:
            log.exception("loading locators")
            rows = []
        t1 = time.monotonic()
        try:
            r = await asyncio.to_thread(svc.set_memory, reviews, rows)  # one rebuild for both
        except Exception:
            log.exception("applying reviews and locators")
            r = dict(loaded=0, skipped=0, locators=0, locators_skipped=0, error=True)
        try:
            vals = await asyncio.wait_for(self.store.item_values(), 60) if hasattr(self.store, "item_values") else None
        except Exception:
            log.exception("loading flat values")
            vals = None
        if vals is not None and hasattr(svc, "set_values"):
            v = await asyncio.to_thread(svc.set_values, vals)
            r["values"], r["values_skipped"] = v["loaded"], v["skipped"]
        r.update(await self.load_rules(svc))
        r.update(await self.load_answers(svc))
        r["seconds"] = dict(fetch=round(t1 - t0, 1), rebuild=round(time.monotonic() - t1, 1))
        log.info("review memory: %d loaded, %d skipped, %d unreadable rows; locators: %d loaded, %d skipped "
                 "(fetch %.1fs, rebuild %.1fs)", r["loaded"], r["skipped"], len(problems), r["locators"],
                 r["locators_skipped"], t1 - t0, r["seconds"]["rebuild"])
        return r

    async def load_answers(self, svc: ModelService) -> dict:
        """Caution answers (tri_answers) into the model; {} without the table or on failure."""
        if not hasattr(self.store, "answers") or not hasattr(svc, "set_answers"):
            return {}
        try:
            rows = await asyncio.wait_for(self.store.answers(), 60)
            return await asyncio.to_thread(svc.set_answers, rows)
        except Exception:
            log.exception("loading caution answers")
            return dict(answers_error=True)

    async def load_rules(self, svc: ModelService) -> dict:
        """Authored rules (tri_rules) into the model. {authored_rules, authored_broken};
        {} when the store has no rules table or the read fails (logged)."""
        if not hasattr(self.store, "authored_rules") or not hasattr(svc, "set_rules"):
            return {}
        try:
            rows = await asyncio.wait_for(self.store.authored_rules(), 60)
        except Exception:
            log.exception("loading authored rules")
            return dict(authored_error=True)
        try:
            v = await asyncio.to_thread(svc.set_rules, rows)
        except Exception:
            log.exception("applying authored rules")
            return dict(authored_error=True)
        log.info("authored rules: %d rows, %d loaded, %d broken%s", len(rows), v["loaded"], v["broken"],
                 "".join(f"\n  {b}" for b in getattr(svc, "rules_broken", [])[:10]))
        return dict(authored_rules=v["loaded"], authored_broken=v["broken"])

    async def load_locators(self, svc: ModelService) -> int:
        """Load every locator (administrator-labelled build) into the model. Instant, no refit."""
        try:
            rows = await asyncio.wait_for(self.store.locators(), 60)
        except Exception:
            log.exception("loading locators")
            return 0
        r = await asyncio.to_thread(svc.set_locators, rows)
        if r["loaded"] or r["skipped"]:
            log.info("locators: %d loaded, %d skipped (unreadable for this run)", r["loaded"], r["skipped"])
        return r["loaded"]

    async def close(self):
        if self._sync_task:
            self._sync_task.cancel()
        if getattr(self, "_watch_task", None):
            self._watch_task.cancel()
        if getattr(self, "_health", None) is not None:
            await self._health.cleanup()
        await self.store.close()
        await self.fetcher.close()
        await super().close()

    # ------------------------------------------------------------ permissions
    @staticmethod
    def _user(x):
        """The member behind an Interaction (.user) or a Message (.author)."""
        for a in ("user", "author"):
            u = getattr(x, a, None)
            if u is not None:
                return u
        return x

    @classmethod
    def manages(cls, inter) -> bool:
        p = getattr(cls._user(inter), "guild_permissions", None)
        return bool(p and (p.manage_guild or p.administrator))

    def is_admin(self, inter, g: GuildConfig) -> bool:
        """inter: an Interaction or a Message."""
        if self.manages(inter):
            return True
        return bool({r.id for r in getattr(self._user(inter), "roles", [])} & set(g.admin_role_ids))

    def is_reviewer(self, inter, g: GuildConfig) -> bool:
        if self.is_admin(inter, g):
            return True
        if not g.configured:
            return False
        return g.open_reviewing or bool({r.id for r in getattr(self._user(inter), "roles", [])}
                                        & set(g.reviewer_role_ids))

    def is_user(self, inter, g: GuildConfig) -> bool:
        """analyze, complete and plot (v3.1): reviewers, user roles, or everyone when open."""
        if self.is_reviewer(inter, g):
            return True
        if not g.configured:
            return False
        return g.open_analysis or bool({r.id for r in getattr(self._user(inter), "roles", [])}
                                       & set(g.user_role_ids))

    async def reviewer_anywhere(self, user_id: int) -> bool:
        """True if the user holds a reviewer or admin role in a tri server the bot
        is in (cached 10 minutes). Used for USER_APP=reviewers."""
        now = time.monotonic()
        hit = self._reviewer_cache.get(user_id)
        if hit and now - hit[0] < 600:
            return hit[1]
        ok = False
        for guild in self.guilds_list():
            g = await self.guild_cfg.get(guild.id)
            if not g.configured:
                continue
            m = guild.get_member(user_id)
            if m is None:
                try:
                    m = await guild.fetch_member(user_id)
                except discord.HTTPException:
                    continue
            if self.is_reviewer(m, g):
                ok = True
                break
        self._reviewer_cache[user_id] = (now, ok)
        return ok

    def in_guild(self, inter) -> bool:
        """The interaction comes from a server the bot is in (a tri server)."""
        gid = getattr(inter, "guild_id", None)
        return bool(gid) and self.get_guild(gid) is not None

    async def config(self, inter) -> GuildConfig:
        gid = getattr(inter, "guild_id", None) or getattr(getattr(inter, "guild", None), "id", None)
        return await self.guild_cfg.get(gid or 0)

    # ------------------------------------------------------------ review sheets and replies
    async def post_sheet(self, inter: discord.Interaction, user_id: int, text: str,
                         embed: Optional[discord.Embed] = None) -> Optional[discord.Message]:
        """Post the review sheet as a regular channel message (a bot cannot read
        replies to an interaction response without the privileged intent) and
        register it on the user's draft. None if the bot cannot post here."""
        ch = inter.channel
        if ch is None or not hasattr(ch, "send"):
            return None
        try:
            m = await ch.send(paint_msg(clip(text, 1990)), embed=paint_embed(embed), allowed_mentions=NO_PINGS)
        except discord.HTTPException:
            log.warning("could not post a review sheet in %s", getattr(ch, "id", "?"))
            return None
        self.desk.attach_message(user_id, m.id, getattr(ch, "id", None))
        return m

    async def on_message(self, message: discord.Message):
        """A reply to a review sheet (or to the bot's answer to one) edits the draft."""
        if message.author.bot or message.guild is None:
            return
        ref = message.reference
        if ref is None or not ref.message_id:
            return
        qs = self.questions.get(ref.message_id)
        if qs is not None:
            return await self.handle_answers(message, qs)
        d = self.desk.draft_for_message(ref.message_id)
        if d is None:
            return
        await self.handle_reply(message, d)

    # ------------------------------------------------------------ question sheets (v3.1.1)
    async def post_questions(self, inter: discord.Interaction, questions: List[dict], title: str,
                             round_no: int, timeout: float) -> Optional[Session]:
        """Post a question sheet as a channel message and open its session. None if
        the bot cannot post here (private replies, DMs, a user install elsewhere)."""
        ch = inter.channel
        if ch is None or not hasattr(ch, "send") or inter.guild is None:
            return None
        text = render_sheet(questions, title, round_no, timeout)
        try:
            m = await ch.send(paint_msg(clip(f"<@{inter.user.id}> " + text, 1990)),
                              allowed_mentions=discord.AllowedMentions(users=[inter.user]))
        except discord.HTTPException:
            log.warning("could not post a question sheet in %s", getattr(ch, "id", "?"))
            return None
        s = Session(inter.user.id, list(questions), asyncio.get_running_loop().create_future(), message=m)
        self.questions[m.id] = s
        return s

    async def handle_answers(self, message: discord.Message, s: Session) -> None:
        """A reply to a question sheet: record the answers; close the round when
        every question has one, or on "post"."""
        async def answer(text: str):
            try:
                m = await message.reply(paint_msg(clip(text, 1990)), mention_author=False, allowed_mentions=NO_PINGS)
                self.questions[m.id] = s                 # replying to tri's answer works too
            except discord.HTTPException:
                log.warning("could not answer a question reply")
        if message.author.id != s.user_id:
            return await answer(f"These questions belong to <@{s.user_id}>.")
        if not (message.content or "").strip():
            return await answer("I could not read that reply. Turn on ping on reply (the @ ON toggle when "
                                "replying), or ask an admin to enable the Message Content intent.")
        svc = self.svc
        arch = getattr(svc, "archetypes", None) or []
        p = parse_answers(message.content, s.questions,
                          resolve_item=(svc.resolve if svc is not None else None),
                          resolve_archetype=lambda n: next((a for a in arch if archetype_key(a) == archetype_key(n)), None))
        s.take(p)
        # v3.1.3: no confirmation; tri replies only when something is unread or still open
        errs = "; ".join(p.errors)
        if errs or not s.closed:
            left = s.open_numbers()
            await answer(" ".join(x for x in ((f"Could not read: {errs}." if errs else ""),
                                              ("Open: " + ", ".join(map(str, left)) + "." if left and not s.closed
                                               else "")) if x))
        if s.closed and s.future is not None and not s.future.done():
            s.future.set_result("post" if s.done else "answered")

    async def ask_first(self, inter: discord.Interaction, gc: GuildConfig, res: dict, run, build, title: str):
        """v3.1.1: ask the caution questions before posting. run(answers, caution) ->
        result. Rounds go on while answers make new questions necessary (at most
        gc.max_questions in all when it is set). Returns (result to post, answers)."""
        qs0 = (res.get("caution") or {}).get("questions") or []
        if gc.caution != "ask" or not qs0:
            return res, None
        s_all = Session(inter.user.id, [])
        asked: set = set()
        cap = int(gc.max_questions or 0) or int(getattr(self.svc.tri.cfg.caution, "max_questions", 0) or 0)   # v3.6: a budget in all
        rounds = 0
        while rounds < 6:
            qs = [q for q in (res.get("caution") or {}).get("questions") or [] if q["key"] not in asked]
            if cap:
                qs = qs[:max(cap - len(asked), 0)]
            if not qs:
                break
            s = await self.post_questions(inter, qs, title, rounds, float(gc.question_timeout))
            if s is None:
                break
            try:
                how = await asyncio.wait_for(asyncio.shield(s.future), timeout=float(gc.question_timeout))
            except asyncio.TimeoutError:
                how = "timeout"
            finally:
                for k in [k for k, v in self.questions.items() if v is s]:
                    self.questions.pop(k, None)
            asked |= {q["key"] for q in qs}
            s_all.take_session(s)
            rounds += 1
            if how == "timeout":
                try:
                    await s.message.reply("No more answers in time; posting the analysis with what I have.",
                                          mention_author=False, allowed_mentions=NO_PINGS)
                except (discord.HTTPException, AttributeError):
                    pass
            res = await run(s_all.answers(), "ask")
            if how != "answered":
                break
        ans = s_all.answers()
        res = await run(ans, "hold")
        if rounds and (ans["archetype"] or ans["central"] or ans.get("want") or ans.get("weapon")) and gc.record_answers and self.in_guild(inter):
            st = getattr(self, "store", None)
            if st is not None and hasattr(st, "add_answers"):
                from tri.core.codec import build_to_dict
                try:
                    if await st.add_answers(str(inter.user.id), build_to_dict(build), ans):
                        await self.load_answers(self.svc)
                except Exception:
                    log.warning("storing caution answers failed", exc_info=True)
        return res, (ans if rounds else None)

    async def handle_reply(self, message: discord.Message, d) -> None:
        async def answer(text: str, view=None):
            try:
                kw = dict(mention_author=False, allowed_mentions=NO_PINGS)
                if view is not None:
                    kw["view"] = view
                m = await message.reply(paint_msg(clip(text, 1990)), **kw)
            except discord.HTTPException:
                log.warning("could not answer a review reply")
                return
            self.desk.attach_message(d.user_id, m.id)
        if message.author.id != d.user_id:
            return await answer(f"This review sheet belongs to <@{d.user_id}>. `/review` gets you a build of your own.")
        if self.svc is None:
            return await answer("tri is still loading; try again in a minute.")
        g = await self.config(message)
        if not self.is_reviewer(message, g):
            return await answer("You need a reviewer role for this.")
        if not (message.content or "").strip():
            return await answer("I could not read that reply. Turn on ping on reply (the @ ON toggle when "
                                "replying), or ask an admin to enable the Message Content intent "
                                "(TRI_MESSAGE_CONTENT=1).")
        try:
            res = apply_reply(self.desk, d.user_id, message.content, self.svc)
        except DraftError as e:
            return await answer(str(e))
        out = res.text()
        if "skip" in res.control:
            self.desk.release(d.user_id, skip=True)
            return await answer(("```\n" + out + "\n```\n" if out else "") +
                                "Released. `/review` gives you a different build.")
        d = self.desk.draft(d.user_id) or d
        view = None
        tail = draft_text(d, self.svc, 1200)
        if "submit" in res.control:
            if res.errors:
                tail += "\nNot submitted: fix the lines marked err, then reply `submit`."
            else:
                try:
                    _, warnings = self.desk.review(d, self.svc.catalog)
                    view = ConfirmSubmit(self, d.user_id)
                    tail += "\nPress Submit review to store it." + (
                        "\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings) if warnings else "")
                except DraftError as e:
                    tail += f"\nNot submitted: {e}"
        await answer(("```\n" + out + "\n```\n" if out else "") + tail, view=view)

    # ------------------------------------------------------------ builds
    async def model_call(self, fn, *a, **kw):
        """Run a model call off the event loop, giving up after
        TRI_COMMAND_TIMEOUT seconds (the command then says it timed out).
        v3.8: the call waits for the model lock at most that long and is dropped,
        not run late, once the command has given up; before, every timed-out
        command still ran afterwards, one after another under the lock, so one
        slow step (a reload, a review rebuild) made every later command time out
        too."""
        limit = float(getattr(self.settings, "command_timeout", 90.0))
        svc = self.svc
        lock = getattr(svc, "lock", None)
        gave_up = threading.Event()

        def run():
            if lock is None:
                return fn(*a, **kw)
            if not lock.acquire(timeout=max(1.0, limit - 5.0)):    # answers "busy" before the timeout
                raise ModelBusy()
            try:
                if gave_up.is_set():
                    return None
                return fn(*a, **kw)
            finally:
                lock.release()
        try:
            return await asyncio.wait_for(asyncio.to_thread(run), limit)
        except asyncio.TimeoutError:
            gave_up.set()
            raise

    async def load_build(self, link_or_id: str) -> tuple:
        """-> (build_id, ParsedBuild)."""
        bid, rec = await self.fetcher.fetch_record(link_or_id)
        parsed = await self.model_call(self.svc.parse, bid, rec)
        return bid, parsed

    async def pop_reviewed(self, pool_id: int, link: str) -> Optional[dict]:
        """Delete a reviewed build's pool row(s) (POOL_POP). Failures only log:
        the review is stored, and the row is never served again anyway."""
        if not getattr(self.settings, "pool_pop", True):
            return None
        try:
            r = await self.store.pop(pool_id, link)
        except StoreError:
            log.exception("popping pool #%s", pool_id)
            return None
        if r["blocked"] and not getattr(self, "_pop_warned", False):
            self._pop_warned = True
            log.warning("pool rows cannot be deleted after a review: reviews.id still references pool.id. "
                        "Run supabase/schema.sql to allow it (rows are skipped either way).")
        if r.get("rls") and not getattr(self, "_rls_warned", False):
            self._rls_warned = True
            log.warning("pool rows cannot be deleted: row-level security with a %s key. Set SUPABASE_SERVICE_KEY "
                        "to the secret key (sb_secret_...) or the service_role key.", self.store.kind)
        return r

    async def pending_ids(self) -> set:
        out = set()
        for p in await self.store.pending():
            try:
                out.add(parse_build_link(p.link))
            except BuildLinkError:
                pass
        return out

    async def post_log(self, inter: discord.Interaction, body: dict, pool_id: int):
        g = await self.config(inter)
        if not g.log_channel_id or inter.guild is None:
            return
        ch = inter.guild.get_channel(g.log_channel_id)
        if ch is None:
            return
        try:
            await ch.send(embed=paint_embed(review_embed(body, self.svc, pool_id)), allowed_mentions=NO_PINGS)
        except discord.HTTPException:
            log.warning("could not post to the review log channel in %s", inter.guild_id)


def main(argv: Optional[Sequence[str]] = None):
    import argparse
    ap = argparse.ArgumentParser(description="tri review bot")
    ap.add_argument("--env", default=".env", help="dotenv file (default .env)")
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env(a.env)
    TriBot(s).run(s.discord_token, log_handler=None)
