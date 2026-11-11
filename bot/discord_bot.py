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
import io
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import discord
from discord import app_commands

from tri.reviews.locators import archetype_key
from tri.sources.dwt import parse_drive_link as _dwt_drive

from .color import paint_embed, paint_kw, paint_msg  # noqa: F401
from .config import Settings
from .db import AlreadyReviewed, RlsRefused, StoreError, SupabaseStore
from .format import fit_fields, item_fields, kinds_from, single
from .guilds import GuildConfig, GuildConfigStore
from .ingest import PURGE_MARKS, purge_plan, with_dependants
from .links import BuildFetcher, BuildLinkError, canonical_link, full_link, parse_build_link
from .wire import REASON_LABELS


def ModelService(run: str):
    """The in-process engine, imported on first use."""
    from .service import ModelService as _MS
    return _MS(run)


from .questions import Session, parse_answers, render_sheet
from .replies import REPLY_HELP, apply_reply
from .reviews import REMOVE_REASONS, DraftError, ReviewDesk

log = logging.getLogger("tri.bot")

STRENGTH_CHOICES = [app_commands.Choice(name="strong", value="strong"),
                    app_commands.Choice(name="weak", value="weak")]
REMOVE_REASON_CHOICES = [app_commands.Choice(name=REASON_LABELS[r.value], value=r.value) for r in REMOVE_REASONS]
ADD_REASON_CHOICES = [app_commands.Choice(name=REASON_LABELS[k], value=k)
                      for k in ("necessary_for_archetype", "generally_good", "synergy_with", "luxury_more",
                                "scenario_mismatch", "no_synergy")]

REVIEW_HELP = ("Mark only what you would change; anything unmarked counts as endorsed. Talent slots are full. "
               "Ignore stat optimality.\n"
               "`/add` [+]  `/remove` [-]  `/iffy` [=]  then `/draft` to check, `/submit` when done, `/skip` to pass.")
NOT_SET_UP = "tri is not set up in this server yet. Someone with Manage Server can run `/setup`."
NO_PINGS = discord.AllowedMentions.none()
from .embeds import (COMPACT_GROUPS,  # noqa: F401  (re-exported: tests and callers import them from here)
                     FACT_SYMBOLS, RULE_SYMBOLS, SEP, SYMBOLS, analysis_embed, analysis_fields,
                     RULE_GRAMMAR, archetype_line, authored_rule_text, clip, rule_list_text, compact_fields, compact_footer, complete_embed, complete_fields,
                     completion_notes, concept_text, config_embed, digest_text, draft_text, fact_line,
                     facts_text, hide_placement, indicative_lines, item_text, locator_text, lookup_embed,
                     lookup_text, mantra_count, matches_embed, placement_note, pool_check_text, quality_lines, review_embed,
                     rules_text, sidegrade_lines, suggest_text, sync_text)

# ---------------------------------------------------------------- bot
LOADING_MSG = "tri is still loading its model. Try again in a minute."


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


#: Discord codes for an interaction that can no longer be answered: its token
#: lasts 15 minutes (50027 Invalid Webhook Token, 10015 Unknown Webhook) and the
#: first answer must come within 3 seconds (10062 Unknown interaction).
EXPIRED_CODES = (50027, 10015, 10062)


#: Commands that also work as a user-installed app (DMs, group DMs, servers without the bot).
USER_APP_COMMANDS = ("analyze", "complete", "plot")
#: Where they may be installed and used.
USER_APP_INSTALLS = dict(guilds=True, users=True)
USER_APP_CONTEXTS = dict(guilds=True, dms=True, private_channels=True)


def user_install_url(app_id) -> str:
    """Install link for the user app (Developer Portal: Installation, User Install on)."""
    return f"https://discord.com/oauth2/authorize?client_id={app_id}&integration_type=1&scope=applications.commands"


@dataclass
class Access:
    """Who may run an analysis command here. hide: hold back the model's view
    of builds waiting for review (always on outside tri servers)."""
    hide: bool
    external: bool = False
    g: Optional["GuildConfig"] = None


async def say(inter: discord.Interaction, msg: str = "", **kw):
    """Answer an interaction. If Discord no longer accepts answers to it (a
    command that ran past 15 minutes, such as a slow /model reload), the
    answer goes to the user by DM (ephemeral answers) or to the channel."""
    kw.setdefault("ephemeral", True)
    if kw.get("view", 0) is None:                    # discord.py wants MISSING, not None
        kw.pop("view")
    paint_kw(kw)
    msg = paint_msg(msg)
    try:
        if inter.response.is_done():
            await inter.followup.send(msg, **kw)
        else:
            await inter.response.send_message(msg, **kw)
    except discord.HTTPException as e:
        if getattr(e, "code", None) not in EXPIRED_CODES:
            raise
        await say_late(inter, msg, **kw)


async def say_late(inter: discord.Interaction, msg: str = "", **kw) -> bool:
    """Deliver an answer outside the interaction. True if delivered."""
    private = kw.pop("ephemeral", True)
    kw.pop("thinking", None)
    kw = {k: v for k, v in kw.items() if k in ("embed", "embeds", "file", "files", "view")}
    note = "(The command took longer than Discord allows for a reply, so here is the answer.)\n"
    if private:
        try:
            await inter.user.send(clip(note + msg, 1990), **kw)
            return True
        except discord.HTTPException:
            pass
    ch = inter.channel
    if ch is not None and hasattr(ch, "send"):
        try:
            await ch.send(clip(f"<@{inter.user.id}> " + note + msg, 1990), **kw,
                          allowed_mentions=discord.AllowedMentions(users=[inter.user], roles=False, everyone=False))
            return True
        except discord.HTTPException:
            pass
    log.warning("could not deliver a late answer to %s: %s", getattr(inter.user, "id", "?"), msg[:200])
    return False


# ---------------------------------------------------------------- setup panel
class SetupView(discord.ui.View):
    """Ephemeral panel: role / channel pickers and toggles, saved with one button."""

    def __init__(self, bot: TriBot, guild: discord.Guild, g: GuildConfig, user_id: int):
        super().__init__(timeout=900)
        self.bot, self.guild, self.user_id = bot, guild, user_id
        self.g = GuildConfig.from_dict(g.guild_id, g.to_dict())     # edit a copy until Save

        def roles(ids):
            return [r for r in (guild.get_role(i) for i in ids) if r is not None][:25]

        self.reviewers = discord.ui.RoleSelect(placeholder="Reviewer roles", min_values=0, max_values=10, row=0,
                                               default_values=roles(self.g.reviewer_role_ids))
        self.admins = discord.ui.RoleSelect(placeholder="tri admin roles (Manage Server always counts)",
                                            min_values=0, max_values=10, row=1,
                                            default_values=roles(self.g.admin_role_ids))
        self.users = discord.ui.RoleSelect(placeholder="User roles (analyze, complete, plot only)",
                                           min_values=0, max_values=10, row=2,
                                           default_values=roles(self.g.user_role_ids))
        ch = guild.get_channel(self.g.log_channel_id) if self.g.log_channel_id else None
        self.channel = discord.ui.ChannelSelect(placeholder="Channel for submitted reviews (optional)",
                                                channel_types=[discord.ChannelType.text, discord.ChannelType.news],
                                                min_values=0, max_values=1, row=3,
                                                default_values=[ch] if ch else [])
        for sel in (self.reviewers, self.admins, self.users, self.channel):
            sel.callback = self._picked
            self.add_item(sel)
        self._labels()

    def _labels(self):
        self.toggle_open.label = f"Everyone can review: {'on' if self.g.open_reviewing else 'off'}"
        self.toggle_open.style = discord.ButtonStyle.primary if self.g.open_reviewing else discord.ButtonStyle.secondary
        self.toggle_users.label = f"Everyone can analyze: {'on' if self.g.open_analysis else 'off'}"
        self.toggle_users.style = discord.ButtonStyle.primary if self.g.open_analysis else discord.ButtonStyle.secondary
        self.toggle_hide.label = f"Hide on pending: {'on' if self.g.hide_pending else 'off'}"
        self.toggle_hide.style = discord.ButtonStyle.primary if self.g.hide_pending else discord.ButtonStyle.secondary

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.user_id or not TriBot.manages(inter):
            await say(inter, "Only the person who opened this panel (with Manage Server) can use it.")
            return False
        return True

    async def _refresh(self, inter: discord.Interaction):
        self._labels()
        await inter.response.edit_message(embed=paint_embed(config_embed(self.g, self.guild)), view=self)

    async def _picked(self, inter: discord.Interaction):
        self.g.reviewer_role_ids = [r.id for r in self.reviewers.values]
        self.g.admin_role_ids = [r.id for r in self.admins.values]
        self.g.user_role_ids = [r.id for r in self.users.values]
        self.g.log_channel_id = self.channel.values[0].id if self.channel.values else None
        await self._refresh(inter)

    @discord.ui.button(label="Everyone can review", row=4)
    async def toggle_open(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.open_reviewing = not self.g.open_reviewing
        await self._refresh(inter)

    @discord.ui.button(label="Everyone can analyze", row=4)
    async def toggle_users(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.open_analysis = not self.g.open_analysis
        await self._refresh(inter)

    @discord.ui.button(label="Hide on pending", row=4)
    async def toggle_hide(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.g.hide_pending = not self.g.hide_pending
        await self._refresh(inter)

    @discord.ui.button(label="Save", style=discord.ButtonStyle.success, row=4)
    async def save(self, inter: discord.Interaction, _b: discord.ui.Button):
        notes = []
        if not self.g.open_reviewing and not self.g.reviewer_role_ids:
            notes.append("No reviewer roles and everyone-can-review is off: only admins can review.")
        if self.g.log_channel_id:
            ch = self.guild.get_channel(self.g.log_channel_id)
            me = self.guild.me
            if ch is None or not (ch.permissions_for(me).send_messages and ch.permissions_for(me).embed_links):
                notes.append("I cannot send embeds in the log channel; give me Send Messages and Embed Links there.")
        self.g.configured = True
        await inter.response.defer()                 # the database write can take over 3 seconds
        try:
            await self.bot.guild_cfg.put(self.g, inter.user.id)
        except StoreError as e:
            return await say(inter, f"Could not save ({e.status}). Nothing changed.")
        self.stop()
        await edit_reply(inter, content="Saved. Reviewers can now use `/review`."
                                          + ("\n" + "\n".join(f"- {n}" for n in notes) if notes else ""),
                                          embed=config_embed(self.g, self.guild), view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, row=4)
    async def cancel(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.stop()
        await inter.response.edit_message(content="Setup closed without changes.", embed=None, view=None)


async def open_setup(bot: TriBot, inter: discord.Interaction):
    if inter.guild is None:
        return await say(inter, "Run this in a server.")
    if not bot.manages(inter):
        return await say(inter, "You need Manage Server to set up tri.")
    g = await bot.guild_cfg.get(inter.guild.id)
    await say(inter, embed=config_embed(g, inter.guild), view=SetupView(bot, inter.guild, g, inter.user.id))


class WelcomeView(discord.ui.View):
    """Button on the message posted when the bot joins a server (persistent)."""

    def __init__(self, bot: TriBot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Set up tri", style=discord.ButtonStyle.success, custom_id="tri:setup")
    async def setup(self, inter: discord.Interaction, _b: discord.ui.Button):
        await open_setup(self.bot, inter)


async def edit_reply(inter: discord.Interaction, **kw):
    """Edit the message a component belongs to, before or after deferring."""
    paint_kw(kw)
    if kw.get("content"):
        kw["content"] = paint_msg(kw["content"])
    if inter.response.is_done():
        try:
            await inter.edit_original_response(**kw)
        except discord.NotFound:                       # the message is gone; say it as a follow-up
            await inter.followup.send(kw.get("content") or "Done.", ephemeral=True)
    else:
        await inter.response.edit_message(**kw)


class PlotView(discord.ui.View):
    """Buttons under a /plot answer (an experiment): turn and tilt the 3-D view,
    step through the axis groups, spin it as an animated GIF, or get an HTML
    page that can be turned freely in a browser. Discord cannot run scripts
    inside a message, so each press draws a new image and swaps it in.
    draw(pairs, view, out) -> [(filename, bytes)] runs off the event loop;
    only the person who ran the command can press the buttons."""

    TURN, TILT = 45, 15

    def __init__(self, owner_id: int, draw, groups: Sequence[str], view: tuple = (22, -58), d3: bool = True,
                 timeout: float = 900):
        super().__init__(timeout=timeout)
        self.owner_id, self.draw = owner_id, draw
        self.groups = list(groups) or ["1-2-3"]
        self.gi, self.elev, self.azim = 0, int(view[0]), int(view[1])
        self.message: Optional[discord.Message] = None
        self.busy = False
        if not d3:
            for b in (self.left, self.right, self.up, self.down, self.spin):
                b.disabled = True
        if len(self.groups) < 2:
            self.next_axes.disabled = True

    @property
    def pairs(self) -> str:
        return self.groups[self.gi % len(self.groups)]

    def turn(self, d_az: int = 0, d_el: int = 0) -> None:
        self.azim = (self.azim + d_az + 180) % 360 - 180
        self.elev = max(-90, min(90, self.elev + d_el))

    def caption(self) -> str:
        return f"axes {self.pairs}{SEP}elevation {self.elev}{SEP}azimuth {self.azim}"

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.owner_id:
            await inter.response.send_message("Only the person who drew this plot can turn it.", ephemeral=True)
            return False
        if self.busy:
            await inter.response.send_message("Still drawing the last view.", ephemeral=True)
            return False
        return True

    async def _show(self, inter: discord.Interaction, out: str = "png"):
        self.busy = True
        await inter.response.defer()
        try:
            files = await asyncio.to_thread(self.draw, self.pairs, (self.elev, self.azim), out)
            fs = [discord.File(io.BytesIO(b), n) for n, b in files[:1]]
            if out == "html":
                await inter.followup.send(f"Interactive {self.caption()}: open the file in a browser, drag to "
                                          "turn, scroll to zoom, hover for names.", files=fs)
            else:
                await inter.edit_original_response(content=self.caption(), attachments=fs, view=self)
        except Exception as e:                          # keep the buttons alive on a failed draw
            log.exception("plot view")
            await inter.followup.send(f"Could not draw that view ({type(e).__name__}: {clip(str(e), 200)}).",
                                      ephemeral=True)
        finally:
            self.busy = False

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, row=0)
    async def left(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.turn(d_az=-self.TURN)
        await self._show(inter)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, row=0)
    async def right(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.turn(d_az=self.TURN)
        await self._show(inter)

    @discord.ui.button(label="▲", style=discord.ButtonStyle.secondary, row=0)
    async def up(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.turn(d_el=self.TILT)
        await self._show(inter)

    @discord.ui.button(label="▼", style=discord.ButtonStyle.secondary, row=0)
    async def down(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.turn(d_el=-self.TILT)
        await self._show(inter)

    @discord.ui.button(label="Next axes", style=discord.ButtonStyle.secondary, row=1)
    async def next_axes(self, inter: discord.Interaction, _b: discord.ui.Button):
        self.gi = (self.gi + 1) % len(self.groups)
        await self._show(inter)

    @discord.ui.button(label="Spin", style=discord.ButtonStyle.primary, row=1)
    async def spin(self, inter: discord.Interaction, _b: discord.ui.Button):
        await self._show(inter, "gif")

    @discord.ui.button(label="Interactive", style=discord.ButtonStyle.primary, row=1)
    async def interactive(self, inter: discord.Interaction, _b: discord.ui.Button):
        await self._show(inter, "html")

    async def on_timeout(self):
        for c in self.children:
            c.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


def plot_groups(files) -> List[str]:
    """The axis groups drawn, from the image names (concept_1-2-3.png -> 1-2-3)."""
    out = []
    for n, _ in files:
        g = n.rsplit(".", 1)[0].split("_", 1)[-1]
        if g and g[0].isdigit():
            out.append(g)
    return out


def fact_graphic(graphic: str) -> tuple:
    """A fact's graphic tag -> (insight kind, keyword arguments)."""
    head, _, arg = (graphic or "").partition(":")
    if head == "card":
        return "card", dict(archetype=arg)
    if head in ("item", "stats"):
        return head, dict(item=arg)
    if head in ("meta", "tree", "atlas"):          # tree and meta charts are deprecated (v2.21): the atlas
        return "atlas", {}
    raise ValueError(f"no graphic {graphic!r}")


class FactView(discord.ui.View):
    """Numbered buttons under /facts: each posts that fact's graphic.
    draw(graphic) -> [(filename, bytes)] is a coroutine."""

    def __init__(self, owner_id: int, rows: Sequence[dict], draw, timeout: float = 900):
        super().__init__(timeout=timeout)
        self.owner_id, self.rows, self.draw = owner_id, list(rows)[:10], draw
        self.message: Optional[discord.Message] = None
        for i, r in enumerate(self.rows):
            b = discord.ui.Button(label=str(i + 1), style=discord.ButtonStyle.secondary, row=i // 5)
            b.callback = self._cb(i)
            self.add_item(b)

    async def interaction_check(self, inter: discord.Interaction) -> bool:
        if inter.user.id != self.owner_id:
            await inter.response.send_message("Only the person who asked can open these.", ephemeral=True)
            return False
        return True

    def _cb(self, i: int):
        async def cb(inter: discord.Interaction):
            await inter.response.defer(thinking=True)
            r = self.rows[i]
            try:
                files = await self.draw(r["graphic"])
                await inter.followup.send(clip(f"**{i + 1}.** {fact_line(r)}", 1990),
                                          files=[discord.File(io.BytesIO(b), n) for n, b in files[:1]],
                                          allowed_mentions=NO_PINGS)
            except Exception as e:
                log.warning("fact graphic %s failed", r.get("graphic"), exc_info=True)
                await inter.followup.send(f"Could not draw it ({type(e).__name__}).")
        return cb

    async def on_timeout(self) -> None:
        for c in self.children:
            c.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


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


# ---------------------------------------------------------------- commands
def register(bot: TriBot):
    tree = bot.tree

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

    @reviews_purge.autocomplete("item")
    async def reviews_purge_item_ac(inter: discord.Interaction, current: str):
        return choices(bot.svc.search(current, bot.svc._names)) if bot.svc else []

    @reviews_purge.autocomplete("archetype")
    async def reviews_purge_archetype_ac(inter: discord.Interaction, current: str):
        cur = archetype_key(current)
        names = [p for p in bot.svc.archetypes if cur in archetype_key(p)] if bot.svc else []
        return [app_commands.Choice(name=n[:100], value=n[:100]) for n in names[:25]]

    # ------------------------------------------------------------ analysis
    def few_talents(build) -> Optional[int]:
        """The talent count when it is below MIN_PLACE_TALENTS (no placement shown)."""
        need = int(getattr(bot.settings, "min_place_talents", 15) or 0)
        n = len(getattr(build, "talents", []) or [])
        return n if build is not None and n < need else None

    async def save_dwt(inter: discord.Interaction, parsed, alt: dict, kind: str, url: str):
        """Save tri's proposal (alt: drops/takes) as a dwt.sh build in DWT_FOLDER (v2.22)."""
        cl = getattr(bot.fetcher, "dwt", None)
        if cl is None or not cl.can_write:
            return await say(inter, "dwt.sh builds cannot be saved: DWT_API_KEY is not set.", ephemeral=False)
        drops = [c["item"] for c in alt.get("drops") or []]
        takes = [c["item"] for c in alt.get("takes") or []]
        arch = alt.get("name") or "its archetype"
        notes = ""
        if kind == "completion":                          # completion notes go to the summary notes (v2.23)
            desc = f"tri completion of {url} as {arch}."
            try:
                notes = completion_notes(parsed, alt, bot.svc, url)
            except Exception:
                log.warning("completion notes failed", exc_info=True)
        else:
            desc = "\n".join(x for x in (f"tri {kind} of {url} as {arch}.",
                                           ("removed: " + ", ".join(drops)) if drops else "",
                                           ("added: " + ", ".join(takes)) if takes else "") if x)
        try:
            link = await cl.save_proposal(parsed, drops, takes, bot.svc.catalog, f"{parsed.name} (tri {kind})", desc,
                                          notes=notes)
        except Exception as e:
            log.warning("dwt save failed: %s", e)
            return await say(inter, clip(f"Could not save the dwt.sh build: {e}", 1900), ephemeral=False)
        await say(inter, f"dwt.sh build ({arch}): {link}", ephemeral=False)

    def placed(res: dict, build) -> dict:
        n = few_talents(build)
        return res if n is None else hide_placement(res, n, int(bot.settings.min_place_talents))

    async def placement_inset(emb: discord.Embed, build) -> List[discord.File]:
        """Archetype line and placement image under /analyze and /complete
        (PLACEMENT_INSET). Never fails the answer. None for builds under
        MIN_PLACE_TALENTS talents."""
        if not getattr(bot.settings, "placement_inset", True) or build is None or few_talents(build) is not None:
            return []
        try:
            r = await bot.model_call(bot.svc.insight, "place", build=build, theme=bot.settings.plot_theme)
        except Exception:
            log.warning("placement inset failed", exc_info=True)
            return []
        if r.get("text"):
            emb.set_author(name=clip(r["text"], 256))
        if not r.get("files"):
            return []
        name, png = r["files"][0]                      # the map in the embed, the reading bar beside it (v2.21)
        emb.set_image(url=f"attachment://{name}")
        return [discord.File(io.BytesIO(p_), n_) for n_, p_ in r["files"][:2]]

    @tree.command(name="analyze", description="Analyze a build")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(link="build link or id",
                           verbose="full analysis page (other archetypes, iffy and weakest picks, rules, sidegrades)",
                           archetype="analyse it as this archetype instead of the one it reads as",
                           dwt="also save tri's proposal (first archetype) as a dwt.sh build")
    async def analyze(inter: discord.Interaction, link: str, verbose: Optional[bool] = None,
                      archetype: Optional[str] = None, dwt: bool = False):
        g = await analyst(inter)
        if not g:
            return
        gc = g.g or GuildConfig(0)
        verbose = gc.verbose_default if verbose is None else verbose
        await inter.response.defer(thinking=True, ephemeral=gc.private_replies)
        try:
            bid = parse_build_link(link)
            if g.hide and bid in await bot.pending_ids():
                return await say(inter, "This build is waiting for review; the model's view is held back "
                                        "until it is reviewed.", ephemeral=False)
            bid, parsed = await bot.load_build(bid)
            url = canonical_link(bid)
            res = await bot.model_call(bot.svc.analyze, parsed.build, archetype=archetype or None,
                                       caution=gc.caution)
        except (BuildLinkError, ValueError) as e:
            if "unknown archetype" in str(e):
                return await say(inter, clip(str(e), 1900), ephemeral=False)
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        except Exception as e:
            if "unknown archetype" in str(e):                # unknown archetype
                return await say(inter, clip(str(e).split("ValueError: ")[-1], 1900), ephemeral=False)
            log.exception("analyze")
            return await say(inter, f"Could not analyse that build ({type(e).__name__}).", ephemeral=False)

        async def rerun(ans, mode):
            return await bot.model_call(bot.svc.analyze, parsed.build, archetype=archetype or None, answers=ans,
                                        caution=mode)
        if not gc.private_replies:                       # questions first, then the analysis (v3.1.1)
            res, _ = await bot.ask_first(inter, gc, res, rerun, parsed.build, parsed.name or url)
        elif gc.caution == "ask":
            res = await rerun(None, "hold")
        emb = analysis_embed(parsed, placed(res, parsed.build), bot.svc, url, verbose=verbose)
        await say(inter, embed=emb, files=(await placement_inset(emb, parsed.build) if gc.placement_inset else []),
                  ephemeral=gc.private_replies)
        if dwt and res.get("alternatives"):
            await save_dwt(inter, parsed, res["alternatives"][0], "analysis", url)

    @tree.command(name="complete", description="Complete a partial build")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(link="build link or id",
                           talents="talent slots to fill to (default: typical for similar builds)",
                           mantras="mantra slots to fill to (default: typical for similar builds)",
                           archetype="complete it as this archetype instead of the one it reads as",
                           dwt="also save the completed build as a dwt.sh build")
    async def complete(inter: discord.Interaction, link: str, talents: Optional[int] = None,
                       mantras: Optional[int] = None, archetype: Optional[str] = None, dwt: bool = False):
        g = await analyst(inter)
        if not g:
            return
        gc = g.g or GuildConfig(0)
        await inter.response.defer(thinking=True, ephemeral=gc.private_replies)
        try:
            bid = parse_build_link(link)
            if g.hide and bid in await bot.pending_ids():
                return await say(inter, "This build is waiting for review; the model's view is held back.", ephemeral=False)
            bid, parsed = await bot.load_build(bid)
            res = await bot.model_call(bot.svc.complete, parsed.build, talents, mantras, archetype=archetype or None,
                                       caution=gc.caution)
        except (BuildLinkError, ValueError) as e:
            if "unknown archetype" in str(e):
                return await say(inter, clip(str(e), 1900), ephemeral=False)
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        except Exception as e:
            if "unknown archetype" in str(e):                # unknown archetype
                return await say(inter, clip(str(e).split("ValueError: ")[-1], 1900), ephemeral=False)
            log.exception("complete")
            return await say(inter, f"Could not complete that build ({type(e).__name__}).", ephemeral=False)

        async def rerun(ans, mode):
            return await bot.model_call(bot.svc.complete, parsed.build, talents, mantras, archetype=archetype or None,
                                        answers=ans, caution=mode)
        if not gc.private_replies:                       # questions first, then the completion (v3.1.1)
            res, _ = await bot.ask_first(inter, gc, res, rerun, parsed.build, parsed.name or canonical_link(bid))
        elif gc.caution == "ask":
            res = await rerun(None, "hold")
        emb = complete_embed(parsed, placed(res, parsed.build), bot.svc, canonical_link(bid))
        await say(inter, embed=emb, files=(await placement_inset(emb, parsed.build) if gc.placement_inset else []),
                  ephemeral=gc.private_replies)
        if dwt:
            await save_dwt(inter, parsed, res, "completion", canonical_link(bid))

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

    @model_item.autocomplete("name")
    async def model_item_ac(inter: discord.Interaction, current: str):
        return choices(bot.svc.search(current, bot.svc._names))

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

    category = app_commands.Group(name="category", description="Item categories (read only)",
                                  guild_only=True)

    @category.command(name="list", description="All categories (read only)")
    async def category_list(inter: discord.Interaction):
        g = await reviewer(inter)
        if not g:
            return
        cats = bot.svc.categories()
        if not cats:
            return await say(inter, "The loaded run stores no categories; tiers come from the selection rules.")
        lines = []
        for c in ("foundational", "central", "luxury"):
            xs = sorted((i for i, v in cats.items() if v == c), key=str.casefold)
            if xs:
                lines.append(f"{c} ({len(xs)}): " + ", ".join(xs))
        await say(inter, clip("\n".join(lines), 1990))

    value = app_commands.Group(name="value", description="Flat worth offsets",
                               guild_only=True)

    @value.command(name="set", description="Add a flat worth offset (admin)")
    @app_commands.describe(item="talent or mantra", delta="offset in [-0.5, 0.5]",
                           note="optional reason")
    async def value_set(inter: discord.Interaction, item: str, delta: float, note: Optional[str] = None):
        if not await admin(inter):
            return
        name = bot.svc.resolve(item) or item
        if name not in bot.svc.catalog.index:
            return await say(inter, f"**{item}** is not in the model's catalog.")
        if delta == 0 or abs(delta) > 0.5:
            return await say(inter, "A value is a nonzero offset between -0.5 and 0.5. `/value clear` removes one.")
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            await bot.store.set_item_value(name, delta, set_by=f"discord:{inter.user.id}", note=note)
        except Exception as e:
            return await say(inter, f"Not saved, the database refused it (run supabase/schema.sql "
                                    f"if the table is missing): {e}")
        await asyncio.to_thread(bot.svc.set_value, name, delta)
        await say(inter, f"**{name}** worth {delta:+.2f} on every build.", ephemeral=False)

    @value.command(name="clear", description="Remove a flat worth offset (admin)")
    @app_commands.describe(item="talent or mantra")
    async def value_clear(inter: discord.Interaction, item: str):
        if not await admin(inter):
            return
        name = bot.svc.resolve(item) or item
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            await bot.store.clear_item_value(name)
        except Exception as e:
            return await say(inter, f"Not cleared, the database refused it: {e}")
        if name in bot.svc.catalog.index:
            await asyncio.to_thread(bot.svc.set_value, name, None)
        await say(inter, f"**{name}** has no flat value now.", ephemeral=False)

    @value.command(name="list", description="Every flat value")
    async def value_list(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        vals = bot.svc.values() if hasattr(bot.svc, "values") else {}
        if not vals:
            return await say(inter, "No flat values set. `/value set` adds one.")
        xs = sorted(vals.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
        await say(inter, clip("\n".join(f"{v:+.2f}  {i}" for i, v in xs), 1990))

    @value_set.autocomplete("item")
    @value_clear.autocomplete("item")
    async def value_item_ac(inter: discord.Interaction, current: str):
        return choices(bot.svc.search(current, bot.svc._names))

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
        await say(inter, clip(text, 1990))

    # ------------------------------------------------------------ plots
    plot = app_commands.Group(name="plot", description="Concept or worth space",
                              allowed_installs=app_commands.AppInstallationType(guild=True, user=True),
                              allowed_contexts=app_commands.AppCommandContext(guild=True, dm_channel=True,
                                                                               private_channel=True))

    async def plot_builds(inter, g, links: str) -> List[tuple]:
        marks = []
        toks = (links or "").replace(",", " ").split()[:6]
        cap = 6 + 24 * any(_dwt_drive(t) for t in toks)    # a dwt.sh drive folder may mark up to 30 (v2.23)
        for bid in (await bot.fetcher.expand(" ".join(toks), limit=cap))[:cap]:
            if g.hide and bid in await bot.pending_ids():
                raise BuildLinkError(f"{bid} is waiting for review; its place is held back until it is reviewed")
            bid, parsed = await bot.load_build(bid)
            marks.append((parsed.name or bid, parsed.build))
        return marks

    AXES_CHOICES = [app_commands.Choice(name="explain: ticks, grid and what each axis end means", value="explain"),
                    app_commands.Choice(name="on: ticks and grid", value="on"),
                    app_commands.Choice(name="off: no axes", value="off")]
    COLOUR_CHOICES = [app_commands.Choice(name="archetype region", value="region"),
                      app_commands.Choice(name="worth of an item (set item)", value="item"),
                      app_commands.Choice(name="a final stat (set stat)", value="stat")]
    PROJ_CHOICES = [app_commands.Choice(name="pca: main directions of the builds", value="pca"),
                    app_commands.Choice(name="regions: main directions of the region centres", value="regions")]
    STYLE_CHOICES = [app_commands.Choice(name="dots: one per build", value="dots"),
                     app_commands.Choice(name="fields: translucent region polytopes, rigid borders", value="fields")]

    def plot_files(files) -> List[discord.File]:
        return [discord.File(io.BytesIO(png), name) for name, png in files[:10]]

    async def send_plot(inter, msg: str, files, draw, per_view: int, view: tuple):
        """Post the images with the turn/spin/interactive buttons (PLOT_BUTTONS)."""
        if not bot.settings.plot_buttons:
            return await say(inter, clip(msg, 1990), files=plot_files(files), ephemeral=False)
        pv = PlotView(inter.user.id, draw, plot_groups(files), view, d3=per_view == 3)
        await say(inter, clip(msg, 1990), files=plot_files(files), view=pv, ephemeral=False)
        try:
            pv.message = await inter.original_response()
        except discord.HTTPException:
            pass

    async def plot_fail(inter, e: Exception, what: str):
        if isinstance(e, BuildLinkError):
            return await say(inter, f"Could not read that build: {e}", ephemeral=False)
        if isinstance(e, ValueError):
            return await say(inter, clip(f"Could not draw the plot: {e}", 1990), ephemeral=False)
        if isinstance(e, ImportError):
            return await say(inter, "Plots need matplotlib on the bot's machine (pip install matplotlib).",
                             ephemeral=False)
        log.exception("plot %s", what)
        return await say(inter, f"Could not draw the plot ({type(e).__name__}).", ephemeral=False)

    @plot.command(name="concept", description="Concept space on several axes")
    @app_commands.describe(builds="up to 6 links/ids or dwt.sh folders, separated by spaces",
                           locators="draw the administrators' locator builds with their archetypes",
                           dims="axes to draw (default 6)",
                           pairs="axis groups to draw instead, e.g. 1-2-3 1-4-5 (or 1-3 for a plane; up to 6)",
                           axes="axis display", colour="what colours the builds",
                           per_view="axes per image: 3 (3-D volume, default) or 2 (plane)",
                           elevation="3-D camera height in degrees (default 22)",
                           azimuth="3-D camera turn in degrees (default -58)",
                           item="the item whose worth colours the builds (colour: item)",
                           stat="the final stat that colours the builds (colour: stat)",
                           projection="which directions become the axes",
                           summary="add an image with the variance and meaning of each axis",
                           style="dots (one per build) or fields (translucent region polytopes, rigid borders)")
    @app_commands.choices(axes=AXES_CHOICES, colour=COLOUR_CHOICES, projection=PROJ_CHOICES, style=STYLE_CHOICES)
    async def plot_concept_cmd(inter: discord.Interaction, builds: Optional[str] = None, locators: bool = True,
                               dims: app_commands.Range[int, 2, 9] = 6, pairs: Optional[str] = None,
                               axes: str = "explain", colour: str = "region", item: Optional[str] = None,
                               stat: Optional[str] = None, projection: str = "pca", summary: bool = True,
                               per_view: app_commands.Range[int, 2, 3] = 3,
                               elevation: app_commands.Range[int, -90, 90] = 22,
                               azimuth: app_commands.Range[int, -180, 180] = -58, style: str = "dots"):
        g = await analyst(inter)
        if not g:
            return
        await inter.response.defer(thinking=True)            # public
        try:
            it = None
            if item:
                it = bot.svc.resolve(item)
                if not it:
                    raise ValueError(f"no talent or mantra called {item!r}")
                if colour == "region":
                    colour = "item"
            if stat and colour == "region":
                colour = "stat"
            marks = await plot_builds(inter, g, builds)
            theme = bot.settings.plot_theme
            files, text = await asyncio.to_thread(bot.svc.plot_concept, marks, locators, 3000, dims, pairs,
                                                  projection, colour, it, stat, axes, summary, per_view,
                                                  (elevation, azimuth), theme, style=style)

            def draw(pr, vw, out, _a=(marks, locators, projection, colour, it, stat, axes, per_view, theme, style)):
                mk, lc, pj, co, it_, st, ax_, pv, th, sty = _a
                return bot.svc.plot_concept(mk, lc, 3000, 3, pr, pj, co, it_, st, ax_, False, pv, vw, th, out,
                                            style=sty)[0]
        except Exception as e:
            return await plot_fail(inter, e, "concept")
        names = ", ".join(n for n, _ in marks)
        how = {"region": "colour: region", "item": f"colour: worth of {it}", "stat": f"colour: final {stat}"}[colour]
        msg = SEP.join(["Concept space", f"{projection}", how] + (["fields"] if style == "fields" else []) + ["◆ locators"] + ([f"★ {names}"] if names else []))
        if text:
            msg += "\n" + text
        await send_plot(inter, msg, files, draw, per_view, (elevation, azimuth))

    @plot_concept_cmd.autocomplete("item")
    async def plot_concept_item_ac(inter: discord.Interaction, current: str):
        return [app_commands.Choice(name=clip(n, 100), value=clip(n, 100))
                for n in bot.svc.search(current.strip(), bot.svc._names)[:25]]

    @plot_concept_cmd.autocomplete("stat")
    async def plot_concept_stat_ac(inter: discord.Interaction, current: str):
        cur = current.strip().lower()
        return [app_commands.Choice(name=s_, value=s_) for s_ in bot.svc.catalog.stats if cur in s_.lower()][:25]

    @plot.command(name="worth", description="Worth space on several axes")
    @app_commands.describe(items="talents or mantras to mark, separated by commas",
                           build="builder link or id: mark that build's talents and mantras",
                           relative="compare items by which builds value them more than usual, not by pick rate",
                           dims="axes to draw (default 6)",
                           pairs="axis groups to draw instead, e.g. 1-2-3 1-4-5 (or 1-3 for a plane; up to 6)",
                           axes="axis display",
                           summary="add an image with the variance and meaning of each axis",
                           per_view="axes per image: 3 (3-D volume, default) or 2 (plane)",
                           elevation="3-D camera height in degrees (default 22)",
                           azimuth="3-D camera turn in degrees (default -58)")
    @app_commands.choices(axes=AXES_CHOICES)
    async def plot_worth_cmd(inter: discord.Interaction, items: Optional[str] = None, build: Optional[str] = None,
                             relative: bool = False, dims: app_commands.Range[int, 2, 9] = 6,
                             pairs: Optional[str] = None, axes: str = "explain", summary: bool = True,
                             per_view: app_commands.Range[int, 2, 3] = 3,
                             elevation: app_commands.Range[int, -90, 90] = 22,
                             azimuth: app_commands.Range[int, -180, 180] = -58):
        g = await analyst(inter)
        if not g:
            return
        await inter.response.defer(thinking=True)            # public
        hi, unknown = [], []
        for x in (items or "").split(","):
            x = x.strip()
            if x:
                r = bot.svc.resolve(x)
                (hi.append(r) if r else unknown.append(x))
        try:
            if build:
                marks = await plot_builds(inter, g, build.split()[0])
                hi += [x for x in marks[0][1].items() if x not in hi]
            theme = bot.settings.plot_theme
            files, text = await asyncio.to_thread(bot.svc.plot_worth, hi, relative, 2000, 30, dims, pairs,
                                                  axes, summary, 5, per_view, (elevation, azimuth), theme)

            def draw(pr, vw, out, _a=(list(hi), relative, axes, per_view, theme)):
                h, rel, ax_, pv, th = _a
                return bot.svc.plot_worth(h, rel, 2000, 30, 3, pr, ax_, False, 5, pv, vw, th, out)[0]
        except Exception as e:
            return await plot_fail(inter, e, "worth")
        msg = SEP.join(["Worth space", "● talent  ■ mantra", "colour: pass", "size: mean worth"])
        if unknown:
            msg += f"\nNot found: {', '.join(unknown[:10])}"
        if text:
            msg += "\n" + text
        await send_plot(inter, msg, files, draw, per_view, (elevation, azimuth))

    @plot_worth_cmd.autocomplete("items")
    async def plot_items_ac(inter: discord.Interaction, current: str):
        head, _, last = current.rpartition(",")
        pre = (head + ", ") if head else ""
        return [app_commands.Choice(name=clip(pre + n, 100), value=clip(pre + n, 100))
                for n in bot.svc.search(last.strip(), bot.svc._names)[:25]]

    # ------------------------------------------------------------ archetypes, facts, graphics
    async def insight(kind: str, **kw) -> dict:
        kw.setdefault("theme", bot.settings.plot_theme)
        r = await bot.model_call(bot.svc.insight, kind, **kw)
        if kind == "atlas" and (r.get("data") or {}).get("names"):
            bot.atlas_names = list(r["data"]["names"])      # autocomplete when no locator names exist
        return r

    async def send_insight(inter, kind: str, msg: Optional[str] = None, **kw):
        try:
            r = await insight(kind, **kw)
        except Exception as e:
            return await plot_fail(inter, e, kind)
        await say(inter, clip(msg if msg is not None else r.get("text", ""), 1990), files=plot_files(r["files"]),
                  ephemeral=False)

    async def archetype_name_ac(inter: discord.Interaction, current: str):
        cur = archetype_key(current)
        known = (bot.svc.archetypes if bot.svc else []) or getattr(bot, "atlas_names", [])
        names = [p for p in known if cur in archetype_key(p)]
        return [app_commands.Choice(name=n[:100], value=n[:100]) for n in names[:25]]

    analyze.autocomplete("archetype")(archetype_name_ac)
    complete.autocomplete("archetype")(archetype_name_ac)

    async def item_name_ac(inter: discord.Interaction, current: str):
        return [app_commands.Choice(name=clip(n, 100), value=clip(n, 100))
                for n in bot.svc.search(current.strip(), bot.svc._names)[:25]] if bot.svc else []

    async def stat_name_ac(inter: discord.Interaction, current: str):
        cur = current.strip().lower()
        return [app_commands.Choice(name=s_, value=s_) for s_ in bot.svc.catalog.stats if cur in s_.lower()][:25] \
            if bot.svc else []

    @plot.command(name="atlas", description="Archetype map in 3-D")
    @app_commands.describe(dims="axes to draw (default 6)",
                           pairs="axis groups to draw instead, e.g. 1-2-3 1-4-5 (or 1-3 for a plane; up to 6)",
                           axes="axis display",
                           per_view="axes per image: 3 (3-D volume, default) or 2 (plane)",
                           elevation="3-D camera height in degrees (default 22)",
                           azimuth="3-D camera turn in degrees (default -58)",
                           summary="add an image with the variance and meaning of each axis",
                           candidates="draw candidate archetypes as stars",
                           interactive="a page to open in a browser: turn, select archetypes, annotate")
    @app_commands.choices(axes=AXES_CHOICES)
    async def plot_atlas_cmd(inter: discord.Interaction, dims: app_commands.Range[int, 2, 9] = 6,
                             pairs: Optional[str] = None, axes: str = "explain",
                             per_view: app_commands.Range[int, 2, 3] = 3,
                             elevation: app_commands.Range[int, -90, 90] = 22,
                             azimuth: app_commands.Range[int, -180, 180] = -58, summary: bool = False,
                             candidates: bool = True, interactive: bool = False):
        if not await analyst(inter):
            return
        await inter.response.defer(thinking=True)            # public
        try:
            theme = bot.settings.plot_theme
            files, text = await asyncio.to_thread(bot.svc.plot_atlas, dims, pairs, axes, summary, per_view,
                                                  (elevation, azimuth), theme,
                                                  "html" if interactive else "png", 24, candidates)

            def draw(pr, vw, out, _a=(axes, per_view, theme, candidates)):
                ax_, pv, th, cd = _a
                return bot.svc.plot_atlas(3, pr, ax_, False, pv, vw, th, out, 24, cd)[0]
        except Exception as e:
            return await plot_fail(inter, e, "atlas")
        msg = SEP.join(["Archetype atlas", "fields", "◆ locators"] + (["★ candidates"] if candidates else []))
        msg += "\n" + text
        if interactive:
            return await say(inter, clip(msg + "\nOpen the file in a browser: drag to turn, scroll to zoom, "
                                         "hover for names.", 1990), files=plot_files(files), ephemeral=False)
        await send_plot(inter, msg, files, draw, per_view, (elevation, azimuth))

    @plot.command(name="item", description="Where an item is worth taking")
    @app_commands.describe(item="talent, mantra or weapon")
    async def plot_item_cmd(inter: discord.Interaction, item: str):
        if not await analyst(inter):
            return
        await inter.response.defer(thinking=True)
        await send_insight(inter, "item", item=item)

    @plot.command(name="stats", description="Item worth against a stat")
    @app_commands.describe(item="talent, mantra or weapon", stat="stat (default: most relevant)")
    async def plot_stats_cmd(inter: discord.Interaction, item: str, stat: Optional[str] = None):
        if not await analyst(inter):
            return
        await inter.response.defer(thinking=True)
        await send_insight(inter, "stats", item=item, stat=stat)

    plot_item_cmd.autocomplete("item")(item_name_ac)
    plot_stats_cmd.autocomplete("item")(item_name_ac)
    plot_stats_cmd.autocomplete("stat")(stat_name_ac)

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
        await say(inter, clip(msg, 1990), view=fv, ephemeral=False)
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
            r = await insight("rules", item=item, limit=20)
        except Exception as e:
            return await plot_fail(inter, e, "rules")
        rows = r["data"]["rules"]
        if not rows:
            return await say(inter, "No rules touch that item.", ephemeral=False)
        await say(inter, clip(r["text"] + "\n" + rules_text(rows) + "\n-# " + RULE_SYMBOLS, 1990), ephemeral=False)

    rules_cmd.autocomplete("item")(item_name_ac)

    # ------------------------------------------------------------ authored rules (v3.0)
    rule = app_commands.Group(name="rule", description="Selection rules",
                              guild_only=True)

    async def _store_rule_rows(rows=(), deleted=()):
        for r in rows:
            if r:
                await bot.store.put_rule(r)
        for rid in deleted:
            await bot.store.delete_rule(rid)

    @rule.command(name="add", description="Add a rule (admin)")
    @app_commands.describe(text="the rule in symbols or ASCII: [C] A = chime & Flamecharm >= 40 => Ignition",
                           note="optional reason")
    async def rule_add(inter: discord.Interaction, text: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        from tri.rules.language import RuleSyntaxError
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            info = await asyncio.to_thread(bot.svc.add_rule, text, f"discord:{inter.user.id}", note or "")
        except RuleSyntaxError as e:
            return await say(inter, f"Not added: {e}\n-# {RULE_GRAMMAR}")
        try:
            await _store_rule_rows([info.get("entry")])
        except Exception as e:
            await asyncio.to_thread(bot.svc.drop_rule, info["id"])
            return await say(inter, f"Not saved, the database refused it (run supabase/schema.sql "
                                    f"if the table is missing): {e}")
        await say(inter, authored_rule_text(info, "added"), ephemeral=False)

    @rule.command(name="edit", description="Replace a rule by id (admin)")
    @app_commands.describe(rule_id="rule id", text="the new rule", note="optional reason")
    async def rule_edit(inter: discord.Interaction, rule_id: str, text: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        from tri.rules.language import RuleSyntaxError
        await inter.response.defer(ephemeral=True, thinking=True)
        try:
            info = await asyncio.to_thread(bot.svc.edit_rule, rule_id.strip(), text, f"discord:{inter.user.id}",
                                           note or "")
        except RuleSyntaxError as e:
            return await say(inter, f"Not changed: {e}\n-# {RULE_GRAMMAR}")
        try:
            await _store_rule_rows(info.get("entries", []), info.get("deleted", []))
        except Exception as e:
            return await say(inter, f"Changed in the model but not saved (it is lost at the next reload): {e}")
        await say(inter, authored_rule_text(info, f"replaces {rule_id.strip()}"), ephemeral=False)

    @rule.command(name="drop", description="Remove or switch off a rule by id (admin)")
    @app_commands.describe(rule_id="rule id: a selection id, rv:... (review rule) or sg:... (sidegrade)", note="optional reason")
    async def rule_drop(inter: discord.Interaction, rule_id: str, note: Optional[str] = None):
        if not await admin(inter):
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        rid = rule_id.strip()
        try:
            out = await asyncio.to_thread(bot.svc.drop_rule, rid, f"discord:{inter.user.id}", note or "")
        except KeyError:
            return await say(inter, f"No rule {rid}. `/rule list` shows the ids.")
        try:
            if out["done"] in ("removed", "lifted"):
                await bot.store.delete_rule(rid)
            else:
                await bot.store.put_rule(out["entry"])
        except Exception as e:
            return await say(inter, f"Done in the model but not saved (it is lost at the next reload): {e}")
        await say(inter, f"Rule `{rid}` {out['done']}.", ephemeral=False)

    @rule.command(name="undo", description="Switch a mined rule back on (admin)")
    @app_commands.describe(rule_id="the id of a switched-off mined rule")
    async def rule_undo(inter: discord.Interaction, rule_id: str):
        if not await admin(inter):
            return
        rid = rule_id.strip()
        e = bot.svc.tri.authored.get(rid)
        if e is None or e.op != "suppress":
            return await say(inter, f"Rule `{rid}` is not switched off.")
        await inter.response.defer(ephemeral=True, thinking=True)
        await asyncio.to_thread(bot.svc.drop_rule, rid)
        try:
            await bot.store.delete_rule(rid)
        except Exception as e:
            return await say(inter, f"Lifted in the model but not saved: {e}")
        await say(inter, f"Rule `{rid}` is on again.", ephemeral=False)

    @rule.command(name="list", description="List rules")
    @app_commands.describe(item="only rules on this talent or mantra (also where it is a condition)",
                           status="stands, refined, overruled, disputed, suppressed or authored",
                           source="selection only (default without a build), or every source: review rules, "
                                  "sidegrades, weapon rules, game data, ...",
                           build="a build link: mark which rules act on it (implies every source)")
    @app_commands.choices(status=[app_commands.Choice(name=s, value=s) for s in
                                  ("stands", "refined", "overruled", "disputed", "suppressed", "authored", "review",
                                   "active", "dropped", "inactive")],
                          source=[app_commands.Choice(name=s, value=s) for s in
                                  ("every", "selection", "review_rule", "sidegrade", "category",
                                   "weapon", "game", "focus", "claim")])
    async def rule_list(inter: discord.Interaction, item: Optional[str] = None,
                        status: Optional[app_commands.Choice[str]] = None,
                        source: Optional[app_commands.Choice[str]] = None, build: Optional[str] = None):
        if not await reviewer(inter):
            return
        name = (bot.svc.resolve(item) or item) if item else None
        b = None
        if build:
            await inter.response.defer(thinking=True)
            try:
                b = (await bot.load_build(build))[1].build
            except Exception as e:
                return await say(inter, f"The build could not be read: {e}", ephemeral=False)
        src = source.value if source else None
        every = b is not None or src is not None
        rows = await asyncio.to_thread(bot.svc.list_rules, name, status.value if status else None,
                                       None if src in (None, "every") else src, b, every)
        if not rows:
            return await say(inter, "No rules match.")
        await say(inter, clip(rule_list_text(rows) + "\n-# " + RULE_GRAMMAR, 1990), ephemeral=False)

    @rule.command(name="groups", description="Groups rules can count")
    async def rule_groups(inter: discord.Interaction):
        if not await reviewer(inter):
            return
        g = bot.svc.rule_groups()
        if not g:
            return await say(inter, "No groups (the catalog has no categories or derivation trees).")
        await say(inter, clip("\n".join(f"**{k}** ({len(v)}): {', '.join(v[:12])}{' …' if len(v) > 12 else ''}"
                                         for k, v in sorted(g.items())), 1990))

    rule_list.autocomplete("item")(item_name_ac)

    def _kind_ac(want: str):
        async def ac(inter: discord.Interaction, current: str):
            if not bot.svc:
                return []
            c = bot.svc.catalog
            names = [n for n in bot.svc.search(current.strip(), bot.svc._names)
                     if n in c.items and c.items[n].kind.value == want]
            return [app_commands.Choice(name=clip(n, 100), value=clip(n, 100)) for n in names[:25]]
        return ac

    def _kind_matches(name: str, want: str) -> List[str]:
        c = bot.svc.catalog
        return [n for n in bot.svc.search(name.strip(), bot.svc._names, limit=200)
                if n in c.items and c.items[n].kind.value == want]

    async def _lookup(inter: discord.Interaction, name: str, want: str):
        """v3.0.3: an exact (or loose) name, else a substring search: one match
        is looked up, several are listed."""
        if bot.svc and not bot.svc.resolve(name):
            hits = _kind_matches(name, want)
            if not hits:
                return await say(inter, f"no {want} matches `{clip(name, 80)}`", ephemeral=True)
            if len(hits) > 1:
                return await say(inter, embed=matches_embed(name, want, hits), ephemeral=True)
            name = hits[0]
        await inter.response.defer(thinking=True)
        try:
            r = await insight("lookup", item=name)
        except Exception as e:
            return await plot_fail(inter, e, "lookup")
        files = plot_files(r["files"])
        img = files[0].filename if files else None
        await say(inter, embed=lookup_embed(r["data"], img), files=files, ephemeral=False)

    @tree.command(name="talent", description="Talent lookup (substring search)")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(name="name or part of it")
    async def talent_cmd(inter: discord.Interaction, name: str):
        await _lookup(inter, name, "talent")

    @tree.command(name="mantra", description="Mantra lookup (substring search)")
    @app_commands.allowed_installs(**USER_APP_INSTALLS)
    @app_commands.allowed_contexts(**USER_APP_CONTEXTS)
    @app_commands.describe(name="name or part of it")
    async def mantra_cmd(inter: discord.Interaction, name: str):
        await _lookup(inter, name, "mantra")

    talent_cmd.autocomplete("name")(_kind_ac("talent"))
    mantra_cmd.autocomplete("name")(_kind_ac("mantra"))

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

    for grp in (reviews, pool, model, category, value, locator, plot, rule):
        tree.add_command(grp)


def main(argv: Optional[Sequence[str]] = None):
    import argparse
    ap = argparse.ArgumentParser(description="tri review bot")
    ap.add_argument("--env", default=".env", help="dotenv file (default .env)")
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env(a.env)
    TriBot(s).run(s.discord_token, log_handler=None)
