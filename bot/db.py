"""Supabase storage for the review pool and received reviews (PostgREST over HTTPS).

Tables (see supabase/schema.sql):
    pool     id int8, link text            build links waiting for review
    reviews  id int8, operations json      one review per pool row; reviews.id = pool.id

A pool row is pending while it has no review, and no other pool
row of the same build (same build id, any link form) has a review. A review
pops its row: the bot deletes it, and any unreviewed duplicates of the same
build, once the review is stored (POOL_POP, default on). The review keeps the
link, so nothing is lost. Deleting a reviewed row needs reviews.id to not
reference pool.id (supabase/schema.sql); with the old
foreign key the row stays and is simply never served again. Audits were
removed in v3.5: the audits table is deprecated and never read
(supabase/drop_audits.sql drops it).

The bot authenticates with the project's secret key (sb_secret_..., or the
legacy service_role JWT) in SUPABASE_SERVICE_KEY, which bypasses row-level
security. Keep it in the bot's environment only. New-format keys (sb_...) are
not JWTs: they go on the apikey header only, never as a bearer token (the
platform rejects them there). A publishable / anon key reads nothing and
deletes nothing on tables with row-level security, silently; key_kind() and
check() say so.
"""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("tri.db")


class StoreError(RuntimeError):
    """A Supabase request failed."""

    def __init__(self, msg: str, status: int = 0, code: Optional[str] = None):
        super().__init__(msg)
        self.status = status
        self.code = code


class AlreadyReviewed(StoreError):
    """A reviews row with this id exists."""


def _content_total(hdr: dict) -> Optional[int]:
    """Total rows from a Content-Range header ('0-999/1234', '*/0'), or None."""
    v = next((val for k, val in (hdr or {}).items() if str(k).lower() == "content-range"), None)
    if not v or "/" not in str(v):
        return None
    t = str(v).rsplit("/", 1)[1].strip()
    return int(t) if t.isdigit() else None


@dataclass
class PoolEntry:
    id: int
    link: str


def build_key(link: str) -> str:
    """The build behind a pool link, so different link forms of one build match."""
    from .links import BuildLinkError, parse_build_link
    try:
        return parse_build_link(link)
    except BuildLinkError:
        return (link or "").strip()


def key_kind(key: str) -> str:
    """'secret' (sb_secret_), 'publishable' (sb_publishable_), 'service_role' or
    'anon' (legacy JWT, read from its role claim), or 'unknown'."""
    k = (key or "").strip()
    if k.startswith("sb_secret_"):
        return "secret"
    if k.startswith("sb_publishable_"):
        return "publishable"
    if k.count(".") == 2 and k.startswith("eyJ"):
        try:
            seg = k.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))
            role = str(claims.get("role") or "")
            return role if role in ("service_role", "anon", "authenticated") else "unknown"
        except Exception:
            return "unknown"
    return "unknown"


def elevated(kind: str) -> bool:
    """Keys that bypass row-level security."""
    return kind in ("secret", "service_role")


class RlsRefused(RuntimeError):
    """A write matched no row the key may touch (row-level security)."""


def _missing_table(e: "StoreError") -> bool:
    return e.status == 404 or e.code in ("PGRST205", "42P01")


def normalize_supabase_url(url: str) -> str:
    """The REST base for a project URL. Accepts the forms people paste:
    https://<p>.supabase.co, .../ , .../rest/v1, .../rest/v1/pool, quoted or
    with spaces. A doubled /rest/v1 is what makes PostgREST answer 404
    PGRST125 ("Invalid path specified in request URL")."""
    from urllib.parse import urlsplit
    u = (url or "").strip().strip("'\"").strip()
    if u and "://" not in u:
        u = "https://" + u
    sp = urlsplit(u)
    if not sp.netloc:
        raise ValueError(f"SUPABASE_URL is not a URL: {url!r} (expected https://<project>.supabase.co)")
    path = sp.path
    i = path.find("/rest/v1")
    if i >= 0:
        path = path[:i]
    if sp.netloc.endswith(".supabase.co"):
        path = ""                                     # hosted projects have no path prefix
    return f"{sp.scheme}://{sp.netloc}{path.rstrip('/')}/rest/v1"


def normalize_table(name: str) -> str:
    """Table name for the URL path: trimmed, "public." prefix and slashes removed."""
    t = (name or "").strip().strip("'\"").strip().strip("/")
    if t.lower().startswith("public."):
        t = t[7:]
    return t


class SupabaseStore:
    """Minimal async client for the bot's tables (pool, reviews; optional locators,
    item_values, tri_rules). Categories are not stored here (v3.0.2): they live in the run.

    url: https://<project>.supabase.co ; key: service-role key.
    page: rows per request (PostgREST caps responses, 1000 by default)."""

    def __init__(self, url: str, key: str, pool_table: str = "pool", reviews_table: str = "reviews",
                 locators_table: str = "locators",
                 session=None, page: int = 1000, timeout: float = 30.0,
                 values_table: str = "item_values", rules_table: str = "tri_rules",
                 answers_table: str = "tri_answers"):
        self.base = normalize_supabase_url(url)
        self.key = (key or "").strip().strip("'\"").strip()
        self.pool_table = normalize_table(pool_table)
        self.reviews_table = normalize_table(reviews_table)
        self.locators_table = normalize_table(locators_table)   # optional: administrator-labelled builds
        self.values_table = normalize_table(values_table)           # optional: admin flat values (v2.18)
        self.rules_table = normalize_table(rules_table)             # optional: authored selection rules (v3.0)
        self.answers_table = normalize_table(answers_table)         # optional: caution answers (v3.1)
        self.page = page
        self.timeout = timeout
        self._session = session
        self._own = session is None
        self.kind = key_kind(self.key)
        if self.kind in ("publishable", "anon", "authenticated"):
            log.warning("SUPABASE_SERVICE_KEY looks like a %s key: with row-level security on, reads come back "
                        "empty and deletes do nothing. Use the secret key (sb_secret_...) or the service_role key.",
                        self.kind)

    # ------------------------------------------------------------ http
    async def _sess(self):
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self):
        if self._own and self._session is not None:
            await self._session.close()
            self._session = None

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {"apikey": self.key, "Content-Type": "application/json"}
        if self.key.startswith("eyJ"):                 # legacy JWT keys may ride the bearer header too
            h["Authorization"] = f"Bearer {self.key}"
        h.update(extra or {})
        return h

    async def _request(self, method: str, path: str, params: Optional[Dict[str, str]] = None,
                       body: Any = None, headers: Optional[Dict[str, str]] = None):
        return (await self._request_full(method, path, params, body, headers))[0]

    async def _request_full(self, method: str, path: str, params: Optional[Dict[str, str]] = None,
                            body: Any = None, headers: Optional[Dict[str, str]] = None):
        """(body, response headers)."""
        s = await self._sess()
        async with s.request(method, f"{self.base}/{path}", params=params,
                             data=None if body is None else json.dumps(body),
                             headers=self._headers(headers)) as r:
            text = await r.text()
            if r.status >= 400:
                code = None
                try:
                    code = json.loads(text).get("code")
                except Exception:
                    pass
                raise StoreError(f"{method} {path}: HTTP {r.status} {text[:300]}", r.status, code)
            return (json.loads(text) if text.strip() else None), dict(getattr(r, "headers", {}) or {})

    async def _select_all(self, table: str, columns: str, order: Optional[str] = "id.asc",
                          filters: Optional[Dict[str, str]] = None) -> List[dict]:
        """Every row, page by page. The total comes from Content-Range (Prefer:
        count=exact), so a server row cap below `page` (PostgREST max-rows) does
        not cut the read short. order: a column of the table ("id.asc" only for
        tables with an id; tri_rules and item_values are keyed by rule_id / item).
        A missing order column (42703) is retried once unordered (v3.0.2)."""
        out, off, total = [], 0, None
        while True:
            params = {"select": columns, "limit": str(self.page), "offset": str(off)}
            if order:
                params["order"] = order
            params.update(filters or {})
            try:
                rows, hdr = await self._request_full("GET", table, params, headers={"Prefer": "count=exact"})
            except StoreError as e:
                if order and e.code == "42703" and order.split(".")[0] in str(e) and off == 0:
                    return await self._select_all(table, columns, order=None, filters=filters)
                raise
            rows = rows or []
            out.extend(rows)
            if total is None:
                total = _content_total(hdr)
            off += len(rows)
            if not rows or (total is not None and off >= total) or (total is None and len(rows) < self.page):
                return out

    # ------------------------------------------------------------ pool
    async def pool(self) -> List[PoolEntry]:
        return [PoolEntry(int(r["id"]), r["link"]) for r in await self._select_all(self.pool_table, "id,link")]

    async def reviewed_links(self) -> Dict[int, str]:
        """{review id: the reviewed link} (the link is read inside operations, so
        popped pool rows still count). Falls back to reading operations whole
        when the column is text rather than json(b)."""
        try:
            rows = await self._select_all(self.reviews_table, "id,link:operations->>link")
            return {int(r["id"]): r.get("link") or "" for r in rows}
        except StoreError as e:
            if e.status != 400 and e.code not in ("42883", "42804", "PGRST100"):
                raise
        out = {}
        for r in await self._select_all(self.reviews_table, "id,operations"):
            ops = r.get("operations")
            if isinstance(ops, str):
                try:
                    ops = json.loads(ops)
                except ValueError:
                    ops = {}
            out[int(r["id"])] = (ops.get("link") if isinstance(ops, dict) else None) or ""
        return out

    async def pool_state(self) -> dict:
        """One read of everything the pool's status depends on:
        pool, reviewed ids, reviewed build ids."""
        pool = await self.pool()
        rl = await self.reviewed_links()
        return dict(pool=pool, reviewed=set(rl), reviewed_builds={build_key(l) for l in rl.values() if l})

    @staticmethod
    def _split(st: dict) -> tuple:
        """(pending, done) pool rows: done = reviewed, or a build reviewed
        under another row."""
        pending, done = [], []
        for p in st["pool"]:
            if p.id in st["reviewed"] or build_key(p.link) in st["reviewed_builds"]:
                done.append(p)
            else:
                pending.append(p)
        return pending, done

    async def pending(self) -> List[PoolEntry]:
        """Pool rows still waiting for a review, oldest (lowest id) first, one
        per build (the oldest row of a build that was added twice)."""
        pending, _ = self._split(await self.pool_state())
        seen, out = set(), []
        for p in pending:
            k = build_key(p.link)
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out

    async def is_reviewed(self, link: str) -> Optional[int]:
        """The review id of a build (any link form), or None."""
        k = build_key(link)
        for rid, l in (await self.reviewed_links()).items():
            if l and build_key(l) == k:
                return rid
        return None

    async def pop(self, pool_id: int, link: Optional[str] = None) -> dict:
        """After a review of `pool_id`: delete its pool row and every other row of
        the same build that has no review. -> {popped: [ids], kept: [ids],
        blocked: bool}; blocked = the database refused to delete the reviewed row
        (reviews.id still references pool.id; run the v2.9 migration)."""
        st = await self.pool_state()
        me = next((p for p in st["pool"] if p.id == int(pool_id)), None)
        key = build_key(link or (me.link if me else ""))
        rows = [p for p in st["pool"] if p.id == int(pool_id) or (key and build_key(p.link) == key)]
        popped, kept, blocked, rls = [], [], False, False
        for p in rows:
            if p.id in st["reviewed"] and p.id != int(pool_id):
                kept.append(p.id)                       # other reviews keep their row
                continue
            try:
                if await self.remove_link(p.id):
                    popped.append(p.id)
            except RlsRefused:
                rls = True
                kept.append(p.id)
            except StoreError as e:
                if e.code == "23503" or e.status == 409:    # foreign_key_violation
                    blocked = True
                    kept.append(p.id)
                    continue
                raise
        return dict(popped=popped, kept=kept, blocked=blocked, rls=rls)

    async def prune(self) -> dict:
        """Pop every pool row that is already done (see _split). -> {popped,
        kept, blocked} as pop()."""
        st = await self.pool_state()
        _, done = self._split(st)
        popped, kept, blocked, rls = [], [], False, False
        for p in done:
            try:
                if await self.remove_link(p.id):
                    popped.append(p.id)
            except RlsRefused:
                rls = True
                kept.append(p.id)
            except StoreError as e:
                if e.code == "23503" or e.status == 409:
                    blocked = True
                    kept.append(p.id)
                    continue
                raise
        return dict(popped=popped, kept=kept, blocked=blocked, rls=rls)

    async def find_link(self, link: str) -> Optional[PoolEntry]:
        rows = await self._request("GET", self.pool_table, {"select": "id,link", "link": f"eq.{link}"})
        return PoolEntry(int(rows[0]["id"]), rows[0]["link"]) if rows else None

    async def add_link(self, link: str) -> PoolEntry:
        """Insert a link. Works whether pool.id is an identity column or a plain
        int8 (then max(id) + 1 is used, retried once on a race)."""
        try:
            rows = await self._request("POST", self.pool_table, body={"link": link},
                                       headers={"Prefer": "return=representation"})
            return PoolEntry(int(rows[0]["id"]), rows[0]["link"])
        except StoreError as e:
            if e.code != "23502":                       # not_null_violation: id has no default
                raise
        for _ in range(3):
            top = await self._request("GET", self.pool_table, {"select": "id", "order": "id.desc", "limit": "1"})
            rtop = await self._request("GET", self.reviews_table, {"select": "id", "order": "id.desc", "limit": "1"})
            nid = max([int(top[0]["id"]) if top else 0, int(rtop[0]["id"]) if rtop else 0]) + 1   # popped ids stay taken
            try:
                rows = await self._request("POST", self.pool_table, body={"id": nid, "link": link},
                                           headers={"Prefer": "return=representation"})
                return PoolEntry(int(rows[0]["id"]), rows[0]["link"])
            except StoreError as e:
                if e.code != "23505":                   # unique_violation: someone took nid
                    raise
        raise StoreError("could not allocate a pool id")

    async def remove_link(self, pool_id: int) -> bool:
        """Delete one pool row. False when there is no such row; RlsRefused when
        the row is there but the key may not delete it (row-level security)."""
        rows = await self._request("DELETE", self.pool_table, {"id": f"eq.{int(pool_id)}"},
                                   headers={"Prefer": "return=representation"})
        if rows:
            return True
        if await self.find_id(pool_id) is not None:
            raise RlsRefused(f"pool row {pool_id} exists but the delete matched nothing: row-level security "
                             f"with a {self.kind} key")
        return False

    async def find_id(self, pool_id: int) -> Optional[PoolEntry]:
        rows = await self._request("GET", self.pool_table, {"select": "id,link", "id": f"eq.{int(pool_id)}"})
        return PoolEntry(int(rows[0]["id"]), rows[0]["link"]) if rows else None

    async def check(self) -> dict:
        """What the key can do: {kind, elevated, pool, reviews, readable, problems}.
        Read-only (no row is written or deleted)."""
        out = dict(kind=self.kind, elevated=elevated(self.kind), problems=[])
        for name, table in (("pool", self.pool_table), ("reviews", self.reviews_table)):
            try:
                _, hdr = await self._request_full("GET", table, {"select": "id", "limit": "1"},
                                                  headers={"Prefer": "count=exact"})
                out[name] = _content_total(hdr)
            except StoreError as e:
                out[name] = None
                out["problems"].append(f"{name}: HTTP {e.status} {e.code or ''}".strip())
        try:
            await self.reviewed_links()
        except StoreError as e:
            out["problems"].append(f"reviews.operations->>link: HTTP {e.status} {e.code or ''}".strip())
        if not out["elevated"]:
            out["problems"].append(f"a {self.kind} key: row-level security hides rows and blocks deletes")
        out["readable"] = out.get("pool") is not None
        return out

    # ------------------------------------------------------------ reviews
    async def has_review(self, pool_id: int) -> bool:
        rows = await self._request("GET", self.reviews_table, {"select": "id", "id": f"eq.{int(pool_id)}"})
        return bool(rows)

    async def direct_id(self) -> int:
        """A fresh id for a review of a build that is not in the pool (/review build:).
        Direct reviews count down from -1, so they never meet a pool id (pool ids are
        positive, from an identity column or max + 1)."""
        low = await self._request("GET", self.reviews_table, {"select": "id", "order": "id.asc", "limit": "1"})
        return min(int(low[0]["id"]) if low else 0, 0) - 1

    async def insert_review(self, pool_id: int, operations: Any) -> None:
        """Store one review. Raises AlreadyReviewed if the id already has one."""
        try:
            await self._request("POST", self.reviews_table, body={"id": int(pool_id), "operations": operations},
                                headers={"Prefer": "return=minimal"})
        except StoreError as e:
            if e.code == "23505" or e.status == 409:
                raise AlreadyReviewed(f"pool entry {pool_id} already has a review", e.status, e.code)
            raise

    async def reviews(self, ids: Optional[Sequence[int]] = None) -> List[dict]:
        """[{id, operations}] (all, or the given ids)."""
        f = {"id": "in.(" + ",".join(str(int(i)) for i in ids) + ")"} if ids else None
        return await self._select_all(self.reviews_table, "id,operations", filters=f)

    async def delete_reviews(self, ids: Sequence[int]) -> List[int]:
        """Delete reviews by id; returns the ids deleted. RlsRefused when rows
        exist but the key may not delete them (row-level security)."""
        ids = sorted({int(i) for i in ids})
        if not ids:
            return []
        f = {"id": "in.(" + ",".join(map(str, ids)) + ")"}
        rows = await self._request("DELETE", self.reviews_table, f, headers={"Prefer": "return=representation"})
        gone = sorted(int(r["id"]) for r in rows or [])
        if not gone and await self.reviews(ids):
            raise RlsRefused(f"reviews {ids} exist but the delete matched nothing: row-level security "
                             f"with a {self.kind} key")
        return gone

    async def update_rows(self, table: str, bodies: Dict[int, Any]) -> List[int]:
        """Replace the operations JSON of rows by id (v3.1.1: purge strips marks in place).
        Returns the ids updated. RlsRefused when nothing matched but the rows exist."""
        done = []
        for i, body in sorted(bodies.items()):
            rows = await self._request("PATCH", table, {"id": f"eq.{int(i)}"}, body={"operations": body},
                                       headers={"Prefer": "return=representation"})
            if rows:
                done.append(int(i))
        if bodies and not done:
            raise RlsRefused(f"{table}: the update matched nothing: row-level security with a {self.kind} key")
        return done

    # ------------------------------------------------------------ locators
    async def insert_locator(self, link: str, archetype: str, record: Any, added_by: Optional[str] = None) -> dict:
        """Store a locator (one per link; adding the same link again changes its archetype
        and snapshot). Returns the row {id, link, archetype, record, added_by, created_at}."""
        for _ in range(2):
            col = self._locator_col
            body = {"link": link, col: archetype, "record": record, "added_by": added_by}
            try:
                rows = await self._request("POST", self.locators_table, {"on_conflict": "link"}, body=body,
                                           headers={"Prefer": "return=representation,resolution=merge-duplicates"})
            except StoreError as e:
                if self._old_column(e):
                    continue
                raise
            return self._archetype_row((rows or [body])[0])
        raise StoreError("locators table has neither an archetype nor a purpose column")

    # Tables made before v2.5 call the column "purpose"; supabase/schema.sql renames it.
    _locator_col = "archetype"

    def _old_column(self, e: StoreError) -> bool:
        """Switch to the pre-v2.5 column name once, on 'column does not exist'."""
        if self._locator_col == "archetype" and (e.code in ("42703", "PGRST204") or "archetype" in str(e)):
            log.warning("locators table still has the column 'purpose'; run supabase/schema.sql")
            self._locator_col = "purpose"
            return True
        return False

    @staticmethod
    def _archetype_row(r: dict) -> dict:
        if "archetype" not in r and "purpose" in r:
            r = dict(r); r["archetype"] = r.pop("purpose")
        return r

    async def locators(self) -> List[dict]:
        """Every locator row, or [] when the locators table does not exist."""
        for _ in range(2):
            try:
                rows = await self._select_all(self.locators_table,
                                              f"id,link,{self._locator_col},record,added_by,created_at")
                return [self._archetype_row(r) for r in rows]
            except StoreError as e:
                if e.status == 404 or e.code in ("PGRST205", "42P01"):
                    return []
                if self._old_column(e):
                    continue
                raise
        return []

    async def remove_locator(self, locator_id: int) -> bool:
        rows = await self._request("DELETE", self.locators_table, {"id": f"eq.{int(locator_id)}"},
                                   headers={"Prefer": "return=representation"})
        return bool(rows)

    # ------------------------------------------------------------ flat values (v2.18)
    async def item_values(self) -> List[dict]:
        """Every flat value row {item, delta, note, set_by, updated_at}, or []
        when the table does not exist (run supabase/schema.sql)."""
        try:
            return await self._select_all(self.values_table, "item,delta,note,set_by,updated_at", order="item.asc")
        except StoreError as e:
            if e.status == 404 or e.code in ("PGRST205", "42P01"):
                return []
            raise

    async def set_item_value(self, item: str, delta: float, set_by: Optional[str] = None,
                             note: Optional[str] = None) -> dict:
        import time as _t
        body = {"item": item, "delta": float(delta), "set_by": set_by, "note": note,
                "updated_at": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())}
        rows = await self._request("POST", self.values_table, {"on_conflict": "item"}, body=body,
                                   headers={"Prefer": "return=representation,resolution=merge-duplicates"})
        return (rows or [body])[0]

    async def clear_item_value(self, item: str) -> bool:
        rows = await self._request("DELETE", self.values_table, {"item": f"eq.{item}"},
                                   headers={"Prefer": "return=representation"})
        return bool(rows)

    # ------------------------------------------------------------ caution answers (v3.1)
    async def answers(self) -> List[dict]:
        """Every caution answer {id, user_id, build, kind, item, value, ts}, or [] when
        the table does not exist (run supabase/schema.sql)."""
        try:
            return await self._select_all(self.answers_table, "id,user_id,build,kind,item,value,ts")
        except StoreError as e:
            if _missing_table(e):
                self._no_answers_table = True
                return []
            raise

    async def add_answers(self, user_id: str, build: dict, answers: dict) -> int:
        """Store one asker's answers (tri.infer.caution.Answers.to_dict()) for a build."""
        import time as _t
        ts = _t.time()
        body = []
        if answers.get("archetype"):
            body.append(dict(user_id=str(user_id), build=build, kind="archetype", item=None,
                             value=str(answers["archetype"]), ts=ts))
        for it, v in (answers.get("central") or {}).items():
            if v is not None:
                body.append(dict(user_id=str(user_id), build=build, kind="central", item=it,
                                 value="yes" if v else "no", ts=ts))
        for it, v in (answers.get("want") or {}).items():      # v3.6: want and weapon answers are kept too
            if v is not None:                                   # (they act locally; the model reads only
                body.append(dict(user_id=str(user_id), build=build, kind="want", item=it,   # archetype / central)
                                 value="yes" if v else "no", ts=ts))
        if answers.get("weapon"):
            body.append(dict(user_id=str(user_id), build=build, kind="weapon", item=None,
                             value=str(answers["weapon"]), ts=ts))
        if self.__dict__.get("_old_answer_kinds"):
            body = [r for r in body if r["kind"] in ("archetype", "central")]
        if not body or self.__dict__.get("_no_answers_table"):
            return 0
        try:
            await self._request("POST", self.answers_table, {}, body=body, headers={"Prefer": "return=minimal"})
        except StoreError as e:
            if "23514" in str(e) and "kind" in str(e) and not self.__dict__.get("_old_answer_kinds"):
                # v3.6: the table predates want / weapon rows; keep the rest and warn once
                self._old_answer_kinds = True
                log.warning("table %s only allows archetype / central answers: want and weapon answers are not "
                            "stored. In the Supabase SQL editor: alter table public.%s drop constraint if exists "
                            "%s_kind_check; alter table public.%s add constraint %s_kind_check check (kind in "
                            "('archetype', 'central', 'want', 'weapon'));", self.answers_table,
                            *([self.answers_table] * 4))
                return await self.add_answers(user_id, build, answers)
            if not _missing_table(e):
                raise
            self._no_answers_table = True               # warn once; answers still act on the reply
            log.warning("table %s is missing: caution answers are not stored. Run supabase/schema.sql "
                        "in the Supabase SQL editor.", self.answers_table)
            return 0
        return len(body)

    # ------------------------------------------------------------ authored rules (v3.0)
    async def authored_rules(self) -> List[dict]:
        """Every authored rule row {rule_id, op, text, author, note, ts}, oldest
        first, or [] when the table does not exist (run supabase/schema.sql)."""
        try:
            rows = await self._select_all(self.rules_table, "rule_id,op,text,author,note,ts", order="rule_id.asc")
        except StoreError as e:
            if e.status == 404 or e.code in ("PGRST205", "42P01"):
                return []
            raise
        return sorted(rows, key=lambda r: str(r.get("ts") or ""))

    async def put_rule(self, entry: dict) -> dict:
        """Upsert one authored rule entry (tri.rules.authored.AuthoredEntry.to_dict())."""
        import time as _t
        ts = entry.get("ts")
        body = {"rule_id": entry["rule_id"], "op": entry["op"], "text": entry.get("text") or "",
                "author": entry.get("author"), "note": entry.get("note") or None,
                "ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(float(ts) if ts else None))}
        rows = await self._request("POST", self.rules_table, {"on_conflict": "rule_id"}, body=body,
                                   headers={"Prefer": "return=representation,resolution=merge-duplicates"})
        return (rows or [body])[0]

    async def delete_rule(self, rule_id: str) -> bool:
        rows = await self._request("DELETE", self.rules_table, {"rule_id": f"eq.{rule_id}"},
                                   headers={"Prefer": "return=representation"})
        return bool(rows)

    async def recent_reviews(self, n: int = 10) -> List[dict]:
        """Newest (highest id) reviews first."""
        return await self._request("GET", self.reviews_table,
                                   {"select": "id,operations", "order": "id.desc", "limit": str(int(n))}) or []

    async def counts(self) -> Dict[str, int]:
        """pool: rows; pending: builds waiting; done_in_pool: rows reviewed but
        not popped yet (/pool prune); reviews: stored reviews."""
        st = await self.pool_state()
        pending, done = self._split(st)
        return dict(pool=len(st["pool"]), pending=len({build_key(p.link) for p in pending}),
                    done_in_pool=len(done), reviews=len(st["reviewed"]),
                    reviewed=len(st["reviewed"] & {p.id for p in st["pool"]}))
