"""dwt.sh API client (v2.22): read builds, save tri's proposals.

Reads (GET /builds/{id}) need no key for public and unlisted builds; the key
(DWT_API_KEY, sent as the `api-key` header) is needed to create builds and
folders. Proposals go to one folder of the key's account: DWT_FOLDER, a folder
id or name in the account's root (created, unlisted, if no such folder exists).
Pure conversion lives in tri/dwt.py.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple

from tri.sources import dwt as D


class DwtError(RuntimeError):
    """A dwt.sh request failed."""

    def __init__(self, msg: str, status: int = 0):
        super().__init__(msg)
        self.status = status


class DwtClient:
    def __init__(self, key: Optional[str] = None, folder: Optional[str] = "tri", visibility: str = "unlisted",
                 api: str = D.API, session=None, timeout: float = 30.0,
                 user_agent: str = "tri-review-bot (+https://dwt.sh)"):
        self.key = (key or "").strip() or None
        self.folder = (folder or "").strip() or None
        self.visibility = visibility if visibility in ("public", "private", "unlisted") else "unlisted"
        self.api = api.rstrip("/")
        self.timeout = timeout
        self.ua = user_agent
        self._session = session
        self._own = session is None
        self._folder_id: Optional[str] = None
        self._spell: Optional[Tuple[float, Dict[str, str]]] = None
        self._lock = asyncio.Lock()

    @property
    def can_write(self) -> bool:
        return self.key is not None

    async def _sess(self):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self):
        if self._own and self._session is not None:
            await self._session.close()
            self._session = None

    async def _req(self, method: str, path: str, body: Any = None, params: Optional[dict] = None,
                   auth: bool = True) -> Tuple[int, Any]:
        s = await self._sess()
        h = {"User-Agent": self.ua, "Accept": "application/json"}
        if auth and self.key:
            h["api-key"] = self.key
        data = None
        if body is not None:
            h["Content-Type"] = "application/json"
            data = json.dumps(body)
        async with s.request(method, self.api + path, headers=h, data=data, params=params) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt) if txt else None
            except ValueError:
                return r.status, txt

    # ------------------------------------------------------------ reading
    async def get_build(self, build_id: str, retries: Tuple[float, ...] = (0.8, 2.0)) -> dict:
        """GET /builds/{id} ("dwt.<id>", "<id>" or a dwt.sh link). A transient
        failure (network error, HTTP 429 or 5xx) is retried after each delay in
        `retries`; if the API still fails, the build is read from the site's
        page data (https://dwt.sh/builds/<id>/__data.json). A 404 is final."""
        rid = D.raw_id(D.parse_link(build_id) or build_id) if "/" in str(build_id) else D.raw_id(build_id)
        err: DwtError = DwtError(f"could not load dwt.sh build {rid}")
        for wait in (0.0,) + tuple(retries):
            if wait:
                await asyncio.sleep(wait)
            try:
                st, body = await self._req("GET", f"/builds/{rid}")
            except Exception as e:
                err = DwtError(f"could not load dwt.sh build {rid}: {type(e).__name__}: {e}")
                continue
            if st == 404:
                raise DwtError(f"dwt.sh build {rid} not found (private, deleted, or wrong id)", 404)
            if st == 200 and isinstance(body, dict):
                return body
            err = DwtError(f"could not load dwt.sh build {rid}: HTTP {st}", st)
            if st != 429 and st < 500:
                break
        body = await self._page_build(rid)
        if body is None:
            raise err
        return body

    async def _page_build(self, rid: str) -> Optional[dict]:
        """The build object from the site's SvelteKit page data, or None. Each
        data node is a devalue array; the build is the object holding both
        'id' and 'game' (the same shape as the API body)."""
        from .links import _Devalue
        try:
            s = await self._sess()
            async with s.get(f"{D.SITE}/builds/{rid}/__data.json",
                             headers={"User-Agent": self.ua, "Accept": "application/json"}) as r:
                if r.status != 200:
                    return None
                doc = json.loads(await r.text())
        except Exception:
            return None
        for node in (doc or {}).get("nodes") or []:
            arr = node.get("data") if isinstance(node, dict) else None
            if not isinstance(arr, list) or not arr:
                continue
            try:
                root = _Devalue(arr).value(0)
            except Exception:
                continue
            found = _find_build(root)
            if found is not None:
                return found
        return None

    async def list_folder(self, folder: str, depth: int = 4, limit: int = 500) -> list:
        """Every build in a dwt.sh drive folder (a drive link or id), subfolders
        included up to `depth` levels -> ["dwt.<id>", ...] in listing order."""
        fid = D.parse_drive_link(folder) or str(folder)
        out, seen, queue = [], set(), [(fid, 0)]
        while queue and len(out) < limit:
            f, d = queue.pop(0)
            if f in seen:
                continue
            seen.add(f)
            try:
                st, body = await self._req("GET", f"/resources/{f}/list")
            except Exception as e:
                raise DwtError(f"could not list dwt.sh folder {f}: {type(e).__name__}: {e}")
            if st == 404:
                raise DwtError(f"dwt.sh folder {f} not found (private, deleted, or not a folder)", 404)
            if st != 200 or not isinstance(body, dict):
                raise DwtError(f"could not list dwt.sh folder {f}: HTTP {st}", st)
            bs, fs = D.folder_builds(body)
            out += [b for b in bs if b not in out]
            if d + 1 <= depth:
                queue += [(x, d + 1) for x in fs]
        return out[:limit]

    async def spellings(self, ttl: float = 86400.0) -> Dict[str, str]:
        """name key -> dwt spelling of every talent and mantra (GET /data, cached a day)."""
        if self._spell and time.time() - self._spell[0] < ttl:
            return self._spell[1]
        try:
            st, body = await self._req("GET", "/data", auth=False)
            sp = D.spellings(body) if st == 200 and isinstance(body, dict) else {}
        except Exception:
            sp = {}
        if sp:
            self._spell = (time.time(), sp)
        return sp

    # ------------------------------------------------------------ writing
    async def folder_id(self) -> Optional[str]:
        """The proposals folder: DWT_FOLDER as an id or a name in the root,
        created (unlisted) when missing. None means the account's root."""
        if not self.folder:
            return None
        if self._folder_id:
            return self._folder_id
        async with self._lock:
            if self._folder_id:
                return self._folder_id
            st, body = await self._req("GET", "/resources/root")
            if st != 200 or not isinstance(body, dict):
                raise DwtError(f"could not list the dwt.sh account (HTTP {st}); check DWT_API_KEY", st)
            for c in body.get("children") or []:
                if c.get("kind") == "folder" and self.folder in (c.get("id"), c.get("name")):
                    self._folder_id = c["id"]
                    return self._folder_id
            st, body = await self._req("POST", "/resources/folders",
                                       dict(name=self.folder, visibility="unlisted"))
            if st not in (200, 201) or not isinstance(body, dict) or not body.get("id"):
                raise DwtError(f"could not create the dwt.sh folder {self.folder!r} (HTTP {st})", st)
            self._folder_id = body["id"]
            return self._folder_id

    async def create_build(self, game: dict, name: str, description: str = "", notes: str = "") -> str:
        """POST /builds into the proposals folder -> the new build's link."""
        if not self.can_write:
            raise DwtError("DWT_API_KEY is not set: dwt.sh builds cannot be created")
        parent = await self.folder_id()
        st, body = await self._req("POST", "/builds", D.payload(game, name, description, self.visibility, notes),
                                   params={"parent": parent} if parent else None)
        if st not in (200, 201) or not isinstance(body, dict) or not body.get("id"):
            msg = body if isinstance(body, str) else json.dumps(body)[:300] if body else ""
            raise DwtError(f"dwt.sh refused the build (HTTP {st}) {msg}".strip(), st)
        return D.link(D.PREFIX + body["id"])

    async def save_proposal(self, parsed, drops, takes, catalog, name: str, description: str = "",
                            notes: str = "") -> str:
        """Save tri's proposal for a parsed build (ParsedBuild with .record)
        as a new dwt.sh build -> its link."""
        sp = await self.spellings()
        _, _, game = D.proposal(getattr(parsed, "record", None) or {}, parsed.build, drops, takes, catalog, sp)
        return await self.create_build(game, name, description, notes)


def _find_build(x: Any, depth: int = 0) -> Optional[dict]:
    """First dict (breadth-first through 'build' keys and values) with 'id' and 'game'."""
    if depth > 4:
        return None
    if isinstance(x, dict):
        if "id" in x and isinstance(x.get("game"), dict):
            return x
        vals = ([x["build"]] if "build" in x else []) + [v for k, v in x.items() if k != "build"]
        for v in vals:
            got = _find_build(v, depth + 1)
            if got is not None:
                return got
    elif isinstance(x, list):
        for v in x[:20]:
            got = _find_build(v, depth + 1)
            if got is not None:
                return got
    return None
