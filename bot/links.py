"""Build links -> planner records.

A deepwoken.co builder link (https://deepwoken.co/builder?id=P8OTDgih) is a
server-rendered Nuxt page. The build is embedded in the page's __NUXT_DATA__
payload (devalue format: a flat JSON array where containers hold indices into
the array) at state.pinia.build.current, in the same record format as the
planner export (builds.json), so tri.pipeline.parse_record reads it directly.
The public page is the only source: no API key, no proxy.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from tri.sources import dwt as _dwt

BUILDER_HOSTS = ("deepwoken.co", "www.deepwoken.co")
PAGE_URL = "https://deepwoken.co/builder?id={id}"
_ID = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
_NUXT = re.compile(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', re.S)
_WRAP = {"Reactive", "ShallowReactive", "Ref", "ShallowRef"}
_EMPTY = {"EmptyRef", "EmptyShallowRef"}


class BuildLinkError(ValueError):
    """The link is not a builder link, or the build cannot be read."""


def parse_build_link(link: str) -> str:
    """Builder link (or a bare id) -> build id.

    Accepts https://deepwoken.co/builder?id=ID, the same without scheme or with
    www., and /builder/ID; and dwt.sh build links https://dwt.sh/builds/ID
    (-> "dwt.ID", also accepted bare). Raises BuildLinkError otherwise."""
    s = (link or "").strip().strip("<>")
    if _dwt.parse_drive_link(s):
        raise BuildLinkError("that is a dwt.sh drive (folder) link; /pool add, /locator add and "
                             "/plot concept builds accept folders")
    try:
        d = _dwt.parse_link(s)
    except ValueError as e:
        raise BuildLinkError(str(e))
    if d:
        return d
    if _ID.match(s):
        return s
    if "://" not in s:
        s = "https://" + s
    u = urlparse(s)
    if u.hostname not in BUILDER_HOSTS:
        raise BuildLinkError(f"not a deepwoken.co or dwt.sh build link: {link!r}")
    q = parse_qs(u.query).get("id")
    if q and _ID.match(q[0]):
        return q[0]
    parts = [p for p in u.path.split("/") if p]
    if len(parts) == 2 and parts[0] == "builder" and _ID.match(parts[1]):
        return parts[1]
    raise BuildLinkError(f"no build id in {link!r}")


def full_link(given: str, build_id: str) -> str:
    """The link a user gave (with a scheme), or the canonical link when they gave a
    bare id. Embeds and the stored review need a URL, never a bare id (v3.0.2)."""
    s = (given or "").strip().strip("<>")
    if s and not _ID.match(s) and not _dwt.is_dwt_id(s) and ("/" in s or "." in s):
        return s if "://" in s else "https://" + s
    return canonical_link(build_id)


def canonical_link(build_id: str) -> str:
    if _dwt.is_dwt_id(build_id):
        return _dwt.link(build_id)
    return PAGE_URL.format(id=build_id)


# ------------------------------------------------------------------ devalue
class _Devalue:
    """Lazy reader for a Nuxt devalue payload: only what is asked for is decoded."""

    def __init__(self, arr: list):
        self.a = arr

    def raw(self, i: int):
        x = self.a[i]
        while isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and x[0] in _WRAP:
            x = self.a[x[1]]
        return x

    def value(self, i: int, depth: int = 0):
        if depth > 200:
            raise BuildLinkError("payload nests too deeply")
        if not isinstance(i, int) or i < 0:       # -1 undefined, -2 hole, ...
            return None
        x = self.a[i]
        if isinstance(x, list):
            if x and isinstance(x[0], str):
                tag = x[0]
                if tag in _WRAP and len(x) == 2:
                    return self.value(x[1], depth + 1)
                if tag in _EMPTY:
                    return None
                if tag == "Set":
                    return [self.value(j, depth + 1) for j in x[1:]]
                if tag == "Map":
                    return {str(self.value(x[k], depth + 1)): self.value(x[k + 1], depth + 1)
                            for k in range(1, len(x) - 1, 2)}
                if tag == "Date":
                    return x[1]
                if tag in ("null", "undefined"):
                    return None
            return [self.value(j, depth + 1) for j in x]
        if isinstance(x, dict):
            return {k: self.value(v, depth + 1) for k, v in x.items()}
        return x

    def get(self, i: int, *path: str) -> Optional[int]:
        """Index reached by following object keys from index i (None if absent)."""
        for k in path:
            x = self.raw(i)
            if not isinstance(x, dict) or k not in x:
                return None
            i = x[k]
        return i


def is_blank(rec: Dict[str, Any]) -> bool:
    """True for the empty planner template the site serves for unknown ids."""
    st = rec.get("stats") or {}
    return (not rec.get("createdAt") and not rec.get("talents") and not rec.get("mantras")
            and not st.get("pointSpent") and not st.get("buildName"))


def record_from_page(html: str, build_id: Optional[str] = None) -> Dict[str, Any]:
    """Planner record embedded in a builder page. Raises BuildLinkError if the
    page has no build (private, deleted, or the site changed its layout)."""
    m = _NUXT.search(html)
    if not m:
        raise BuildLinkError("page has no __NUXT_DATA__ payload (layout changed?)")
    try:
        dv = _Devalue(json.loads(m.group(1)))
    except json.JSONDecodeError as e:
        raise BuildLinkError(f"unreadable page payload: {e}")
    root = None
    for i in range(min(8, len(dv.a))):          # the state object sits near the start
        x = dv.raw(i)
        if isinstance(x, dict) and "pinia" in x:
            root = i
            break
    if root is None:
        raise BuildLinkError("page payload has no store state")
    bi = dv.get(root, "pinia", "build")
    if bi is None:
        raise BuildLinkError("page payload has no build store")
    cur = dv.get(bi, "current")
    rec = dv.value(cur) if cur is not None else None
    if not isinstance(rec, dict) or not rec.get("attributes") or is_blank(rec):
        raise BuildLinkError("build not found (private, deleted, or wrong id)")
    if build_id is not None:
        got = dv.value(dv.get(bi, "currentId")) if dv.get(bi, "currentId") is not None else None
        if got not in (None, build_id):
            raise BuildLinkError(f"page returned build {got!r}, expected {build_id!r}")
    return rec


class BuildFetcher:
    """Fetches planner records from the public builder page, with a small
    in-memory cache. session: an aiohttp.ClientSession (created lazily if None)."""

    def __init__(self, ttl: float = 600.0, timeout: float = 30.0, session=None,
                 user_agent: str = "tri-review-bot (+https://deepwoken.co)", dwt=None):
        self.dwt = dwt                  # bot.dwt.DwtClient for dwt.sh links (created lazily if None)
        self.ttl = ttl
        self.timeout = timeout
        self._session = session
        self._own = session is None
        self.ua = user_agent
        self._cache: Dict[str, Tuple[float, dict]] = {}

    async def _sess(self):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self):
        if self.dwt is not None:
            await self.dwt.close()
        if self._own and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get(self, url: str, headers: Dict[str, str]) -> Tuple[int, str]:
        s = await self._sess()
        async with s.get(url, headers={"User-Agent": self.ua, **headers}) as r:
            return r.status, await r.text()

    async def expand(self, text: str, limit: int = 200) -> List[str]:
        """Links in `text` (spaces or commas) -> build ids, in order, without
        repeats. A dwt.sh drive link (https://dwt.sh/drive/ID) expands to every
        build in that folder and its subfolders (v2.23). BuildLinkError on a
        token that is neither."""
        out: List[str] = []
        for tok in (text or "").replace(",", " ").split():
            fid = _dwt.parse_drive_link(tok)
            if fid:
                from .dwt import DwtClient, DwtError
                if self.dwt is None:
                    self.dwt = DwtClient()
                try:
                    ids = await self.dwt.list_folder(fid, limit=limit)
                except DwtError as e:
                    raise BuildLinkError(str(e))
            else:
                ids = [parse_build_link(tok)]
            out += [i for i in ids if i not in out]
            if len(out) >= limit:
                break
        return out[:limit]

    async def fetch_record(self, link_or_id: str, use_cache: bool = True) -> Tuple[str, dict]:
        """(build_id, planner record). Raises BuildLinkError if the page cannot be read."""
        bid = parse_build_link(link_or_id)
        hit = self._cache.get(bid)
        if use_cache and hit and time.time() - hit[0] < self.ttl:
            return bid, hit[1]
        if _dwt.is_dwt_id(bid):
            from .dwt import DwtClient, DwtError
            if self.dwt is None:
                self.dwt = DwtClient()
            try:
                rec = _dwt.to_record(await self.dwt.get_build(bid), bid)
            except DwtError as e:
                raise BuildLinkError(str(e))
            self._cache[bid] = (time.time(), rec)
            return bid, rec
        try:
            status, text = await self._get(PAGE_URL.format(id=bid), {})
        except BuildLinkError:
            raise
        except Exception as e:
            raise BuildLinkError(f"could not load build {bid}: {type(e).__name__}: {e}")
        if status == 404:
            raise BuildLinkError(f"build {bid} not found (private, deleted, or wrong id)")
        if status != 200:
            raise BuildLinkError(f"could not load build {bid}: builder page HTTP {status}")
        rec = record_from_page(text, bid)
        self._cache[bid] = (time.time(), rec)
        return bid, rec
