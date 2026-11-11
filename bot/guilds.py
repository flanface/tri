"""Per-server settings, set in Discord with /setup instead of environment ids.

Stored in the Supabase table `guilds(id int8 primary key, config jsonb)` (see
supabase/schema.sql). If that table does not exist the bot falls back to a
local JSON file (GUILD_CONFIG_PATH) and logs a warning, so a missing table
never stops the bot; the file is lost if the host's disk is.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional

from .db import StoreError, SupabaseStore

log = logging.getLogger("tri.bot")

MISSING_TABLE = {"PGRST205", "42P01", "PGRST106"}


@dataclass
class GuildConfig:
    guild_id: int
    configured: bool = False             # reviewing is closed until an admin saves /setup once
    open_reviewing: bool = False         # anyone in the server may review
    reviewer_role_ids: List[int] = field(default_factory=list)
    admin_role_ids: List[int] = field(default_factory=list)
    log_channel_id: Optional[int] = None  # submitted reviews are posted here
    hide_pending: bool = True            # /analyze refuses builds still pending review (non-admins)
    user_role_ids: List[int] = field(default_factory=list)   # v3.1: analyze, complete and plot only
    open_analysis: bool = False          # anyone in the server may analyze, complete and plot
    caution: str = "ask"                 # ask | hold | off: thin marks are asked about, held, or shown
    max_questions: int = 0               # v3.1.1: 0 asks every necessary question; n caps them at n
    question_timeout: int = 300          # seconds to answer a question sheet before tri posts anyway
    private_replies: bool = False        # analysis replies only the invoker sees
    verbose_default: bool = False        # /analyze opens the full page by default
    placement_inset: bool = True         # attach the small concept-space inset
    record_answers: bool = True          # store answers (only a consensus of many users reaches the model)
    updated_by: Optional[int] = None
    updated_at: float = 0.0
    version: int = 2                     # config layout (2: v3.1.1, no default question cap)

    @classmethod
    def from_dict(cls, gid: int, d: dict) -> "GuildConfig":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in (d or {}).items() if k in known}
        if int(d.get("version", 1) or 1) < 2 and int(d.get("max_questions", 0) or 0) == 3:
            d["max_questions"] = 0               # the v3.1 default cap of 3 is gone (v3.1.1)
        d["guild_id"] = int(gid)
        g = cls(**d)
        g.reviewer_role_ids = [int(x) for x in g.reviewer_role_ids]
        g.admin_role_ids = [int(x) for x in g.admin_role_ids]
        g.user_role_ids = [int(x) for x in g.user_role_ids]
        g.caution = g.caution if g.caution in ("ask", "hold", "off") else "ask"
        g.max_questions = max(0, min(int(g.max_questions), 25))
        g.question_timeout = max(30, min(int(g.question_timeout), 1800))
        g.version = 2
        g.log_channel_id = int(g.log_channel_id) if g.log_channel_id else None
        return g

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("guild_id")
        return d


class GuildConfigStore:
    """get / put with an in-memory cache. backend: 'auto' | 'supabase' | 'local'."""

    def __init__(self, store: Optional[SupabaseStore], path: str = "guild_config.json",
                 backend: str = "auto", table: str = "guilds"):
        self.store = store
        self.path = path
        self.backend = "local" if store is None else backend
        self.table = table
        self._cache: Dict[int, GuildConfig] = {}

    # ------------------------------------------------------------ local file
    def _read_local(self) -> Dict[str, dict]:
        if not os.path.isfile(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)

    def _write_local(self, data: Dict[str, dict]):
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".guilds")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.path)

    def _fallback(self, e: StoreError) -> bool:
        if self.backend == "auto" and (e.code in MISSING_TABLE or e.status == 404):
            log.warning("Supabase table %r missing (%s); keeping server settings in %s. "
                        "Run supabase/schema.sql to store them in Supabase.", self.table, e.code, self.path)
            self.backend = "local"
            return True
        return False

    # ------------------------------------------------------------ api
    async def get(self, guild_id: int) -> GuildConfig:
        gid = int(guild_id)
        if gid in self._cache:
            return self._cache[gid]
        raw = None
        if self.backend != "local":
            try:
                rows = await self.store._request("GET", self.table, {"select": "id,config", "id": f"eq.{gid}"})
                raw = rows[0]["config"] if rows else None
                if isinstance(raw, str):
                    raw = json.loads(raw)
            except StoreError as e:
                if not self._fallback(e):
                    raise
        if self.backend == "local":
            raw = self._read_local().get(str(gid))
        g = GuildConfig.from_dict(gid, raw or {})
        self._cache[gid] = g
        return g

    async def put(self, g: GuildConfig, user_id: Optional[int] = None) -> GuildConfig:
        g.updated_by = user_id if user_id is not None else g.updated_by
        g.updated_at = time.time()
        if self.backend != "local":
            try:
                await self.store._request("POST", self.table, {"on_conflict": "id"},
                                          body={"id": g.guild_id, "config": g.to_dict()},
                                          headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
            except StoreError as e:
                if not self._fallback(e):
                    raise
        if self.backend == "local":
            data = self._read_local()
            data[str(g.guild_id)] = g.to_dict()
            self._write_local(data)
        self._cache[g.guild_id] = g
        return g

    def forget(self, guild_id: int):
        self._cache.pop(int(guild_id), None)
