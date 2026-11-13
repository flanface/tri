"""Bot-wide settings, read from the environment (and a .env file if present).
Server-specific settings (reviewer / admin roles, log channel, ...) are not here:
admins set them in Discord with /setup (see bot/guilds.py).

Required
    DISCORD_TOKEN           bot token (Discord developer portal > Bot)
    SUPABASE_URL            https://<project>.supabase.co
    SUPABASE_SERVICE_KEY    service-role key (Project settings > API). SUPABASE_KEY also accepted.
Model
    TRI_RUN                 run dir or runs folder with LATEST, loaded in-process (default runs)
    TRI_RUN_REMOTE          latest | <run name>: download the run from Supabase Storage into TRI_RUN
                            on start and on /model reload (python -m bot.runstore publish uploads one)
    TRI_RUN_BUCKET          Storage bucket of published runs (default tri-runs)
    TRI_COORDS              float32 (default) | float64: storage of the corpus coordinates
    TRI_RELOAD_SWAP         0 (default): /model reload frees the old model before loading the new
                            one (one model in memory); 1 keeps serving the old one until the new loads
Optional
    TRI_COMMAND_TIMEOUT     seconds a command waits for the model before saying it timed out (default 90)
    HEALTH_PORT / PORT      if set, answer GET / and /health on this port (Render web service
                            hosting; not needed for a Render background worker)
    CLAIM_TTL_MIN           minutes a claimed build stays reserved (default 120)
    STATE_PATH              drafts / claims file (default bot_state.json)
    GUILD_CONFIG_STORE      auto | supabase | local (default auto: Supabase table, file if missing)
    GUILD_CONFIG_PATH       local server-settings file (default guild_config.json)
    SYNC_MODE               guild (default: commands appear instantly in every server) | global
    POOL_TABLE, REVIEWS_TABLE, GUILDS_TABLE, LOCATORS_TABLE, VALUES_TABLE, RULES_TABLE
                            table names (default pool, reviews, guilds, locators,
                            item_values, tri_rules)
    GAMEDATA_SYNC_HOURS     hours between checks of pocamind/data for new or changed talents and
                            mantras (default 12; 0 turns the passive sync off; /model sync still works)
    GAMEDATA_REF            pocamind/data branch to follow (default main)
    USER_APP                who may use /analyze, /complete and /plot as a user-installed app (in DMs,
                            group DMs and servers without the bot): anyone (default) | reviewers (a
                            reviewer or admin role in at least one tri server) | off
    POOL_POP                1 (default): delete a pool row once its review is stored, with any
                            unreviewed duplicates of the same build; 0 keeps rows (they are
                            still never served again)
    PLOT_THEME              dark (default, matches Discord) | light
Builds are read from the public builder page; no game API key is needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .wire import load_dotenv  # noqa: E402,F401


def _user_app(v: str) -> str:
    v = (v or "anyone").strip().lower()
    if v not in ("anyone", "reviewers", "off"):
        raise SystemExit("USER_APP is anyone|reviewers|off")
    return v


@dataclass
class Settings:
    discord_token: str
    supabase_url: str
    supabase_key: str
    tri_run: str = "runs"
    claim_ttl_min: float = 120.0
    state_path: str = "bot_state.json"
    guild_config_store: str = "auto"
    guild_config_path: str = "guild_config.json"
    sync_mode: str = "guild"
    pool_table: str = "pool"
    reviews_table: str = "reviews"
    guilds_table: str = "guilds"
    locators_table: str = "locators"
    values_table: str = "item_values"
    rules_table: str = "tri_rules"        # RULES_TABLE: authored selection rules (v3.0)
    gamedata_sync_hours: float = 12.0
    gamedata_ref: str = "main"
    user_app: str = "anyone"
    pool_pop: bool = True
    plot_theme: str = "dark"
    placement_inset: bool = True         # PLACEMENT_INSET: archetype placement image under /analyze and /complete
    min_place_talents: int = 15          # MIN_PLACE_TALENTS: below this many talents no archetype placement is shown
    run_remote: str = ""                 # TRI_RUN_REMOTE: latest | run name (Supabase Storage, v3.7)
    reload_swap: bool = False            # TRI_RELOAD_SWAP: hold two models during /model reload (v3.7)
    reload_sync: bool = True             # RELOAD_SYNC: /model reload also syncs game data before the database
    facts_digest_days: float = 0.0       # FACTS_DIGEST_DAYS: post new and retired facts to the review log every N days (0 off)
    message_content: bool = False        # privileged intent: read replies even with ping on reply off
    command_timeout: float = 90.0
    health_port: Optional[int] = None
    dwt_api_key: Optional[str] = None    # DWT_API_KEY: dwt.sh key (reads need none; saving builds does)
    dwt_folder: str = "tri"              # DWT_FOLDER: folder id or name for saved proposals ("" = account root)
    dwt_visibility: str = "unlisted"     # DWT_VISIBILITY: public | unlisted | private
    dwt_api: str = "https://api.dwt.sh"  # DWT_API

    @classmethod
    def from_env(cls, dotenv: Optional[str] = ".env") -> "Settings":
        if dotenv:
            load_dotenv(dotenv)
        e = os.environ
        missing = [k for k in ("DISCORD_TOKEN", "SUPABASE_URL") if not e.get(k)]
        key = e.get("SUPABASE_SERVICE_KEY") or e.get("SUPABASE_KEY")
        if not key:
            missing.append("SUPABASE_SERVICE_KEY")
        if missing:
            raise SystemExit("missing settings: " + ", ".join(missing) + " (see .env.example)")
        store = e.get("GUILD_CONFIG_STORE", "auto")
        sync = e.get("SYNC_MODE", "guild")
        if store not in ("auto", "supabase", "local") or sync not in ("guild", "global"):
            raise SystemExit("GUILD_CONFIG_STORE is auto|supabase|local and SYNC_MODE is guild|global")
        return cls(discord_token=e["DISCORD_TOKEN"], supabase_url=e["SUPABASE_URL"], supabase_key=key,
                   tri_run=e.get("TRI_RUN", "runs"),
                   run_remote=(e.get("TRI_RUN_REMOTE") or "").strip(),
                   reload_swap=(e.get("TRI_RELOAD_SWAP") or "0").strip().lower() in ("1", "true", "yes", "on"),
                   claim_ttl_min=float(e.get("CLAIM_TTL_MIN", 120)), state_path=e.get("STATE_PATH", "bot_state.json"),
                   guild_config_store=store, guild_config_path=e.get("GUILD_CONFIG_PATH", "guild_config.json"),
                   sync_mode=sync, pool_table=e.get("POOL_TABLE", "pool"),
                   reviews_table=e.get("REVIEWS_TABLE", "reviews"), guilds_table=e.get("GUILDS_TABLE", "guilds"),
                   locators_table=e.get("LOCATORS_TABLE", "locators"),
                   values_table=e.get("VALUES_TABLE", "item_values"),
                   rules_table=e.get("RULES_TABLE", "tri_rules"),
                   gamedata_sync_hours=float(e.get("GAMEDATA_SYNC_HOURS", 12) or 0),
                   gamedata_ref=e.get("GAMEDATA_REF", "main") or "main",
                   user_app=_user_app(e.get("USER_APP", "anyone")),
                   pool_pop=(e.get("POOL_POP", "1") or "1").strip().lower() not in ("0", "false", "no", "off"),
                   plot_theme="light" if (e.get("PLOT_THEME") or "").strip().lower() == "light" else "dark",
                   placement_inset=(e.get("PLACEMENT_INSET") or "1").strip().lower() not in ("0", "false", "no", "off"),
                   facts_digest_days=float(e.get("FACTS_DIGEST_DAYS") or 0),
                   min_place_talents=int(e.get("MIN_PLACE_TALENTS") or 15),
                   reload_sync=(e.get("RELOAD_SYNC") or "1").strip().lower() not in ("0", "false", "no", "off"),
                   message_content=e.get("TRI_MESSAGE_CONTENT", "0").strip().lower() in ("1", "true", "yes", "on"),
                   command_timeout=float(e.get("TRI_COMMAND_TIMEOUT") or 90),
                   health_port=int(e.get("HEALTH_PORT") or e.get("PORT") or 0) or None,
                   dwt_api_key=(e.get("DWT_API_KEY") or "").strip() or None,
                   dwt_folder=e.get("DWT_FOLDER", "tri").strip(),
                   dwt_visibility=(e.get("DWT_VISIBILITY") or "unlisted").strip().lower(),
                   dwt_api=(e.get("DWT_API") or "https://api.dwt.sh").strip())
