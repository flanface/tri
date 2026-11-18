"""Interactive plots on GitHub Pages: the bot uploads each HTML page to the
repository's Pages branch and links it, instead of attaching the file.

    PAGES_REPO      owner/name of the repository that serves the pages (e.g. flanface/tri)
    PAGES_TOKEN     token with Contents read/write on that repository (GITHUB_TOKEN also accepted)
    PAGES_BRANCH    branch GitHub Pages serves (default gh-pages)
    PAGES_URL       public base URL (default https://<owner>.github.io/<name>)
    PAGES_KEEP_DAYS plots older than this are pruned once a day (default 30; 0 keeps everything)

Pages are written to plots/<yyyy-mm-dd>/<name>-<hash>.html, so the same plot is
uploaded once. The branch is created (orphan, with .nojekyll and an index) when
missing, and Pages is switched on for it when the token may do so. Uploads go
to the Pages branch only: the default branch and its contributors are untouched.

    python -m bot.pages check            # settings, branch and Pages status
    python -m bot.pages prune --days 30  # delete old plots now
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger("tri.pages")

API = "https://api.github.com"
INDEX = """<!doctype html><meta charset="utf-8"><title>tri plots</title>
<style>body{background:#0e1015;color:#e6e6e6;font:15px system-ui;margin:3em}</style>
<h1>tri plots</h1><p>Interactive plots are posted here by the tri bot. Open a plot from its link in Discord.</p>
"""


@dataclass
class PagesConfig:
    repo: str = ""
    token: str = ""
    branch: str = "gh-pages"
    url: str = ""
    keep_days: float = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.repo and "/" in self.repo and self.token)

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url.rstrip("/")
        owner, name = self.repo.split("/", 1)
        if name.lower() == f"{owner.lower()}.github.io":
            return f"https://{name}"
        return f"https://{owner}.github.io/{name}"

    @classmethod
    def from_env(cls, env=None) -> "PagesConfig":
        e = os.environ if env is None else env
        return cls(repo=(e.get("PAGES_REPO") or "").strip(),
                   token=(e.get("PAGES_TOKEN") or e.get("GITHUB_TOKEN") or "").strip(),
                   branch=(e.get("PAGES_BRANCH") or "gh-pages").strip(),
                   url=(e.get("PAGES_URL") or "").strip(),
                   keep_days=float(e.get("PAGES_KEEP_DAYS") or 30))


def page_path(name: str, data: bytes, day: Optional[_dt.date] = None) -> str:
    """plots/<yyyy-mm-dd>/<name>-<hash>.html (name made URL-safe)."""
    day = day or _dt.datetime.now(_dt.timezone.utc).date()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.rsplit(".", 1)[0]).strip("-") or "plot"
    return f"plots/{day.isoformat()}/{stem}-{hashlib.sha256(data).hexdigest()[:12]}.html"


class PagesError(RuntimeError):
    pass


class Pages:
    """Uploads HTML pages to the Pages branch through the GitHub REST API."""

    def __init__(self, cfg: PagesConfig, session=None, timeout: float = 30.0):
        self.cfg = cfg
        self._session = session
        self.timeout = timeout
        self._ready = False
        self._lock = asyncio.Lock()
        self._pruned: Optional[_dt.date] = None

    # ------------------------------------------------------------ http
    async def _sess(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _call(self, method: str, path: str, body: Optional[dict] = None, ok=(200, 201)) -> Tuple[int, dict]:
        s = await self._sess()
        h = {"Authorization": f"Bearer {self.cfg.token}", "Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "tri-bot"}
        async with s.request(method, API + path, headers=h, data=None if body is None else json.dumps(body)) as r:
            txt = await r.text()
            try:
                data = json.loads(txt) if txt else {}
            except ValueError:
                data = {"message": txt[:200]}
            if r.status not in ok:
                raise PagesError(f"GitHub {method} {path}: {r.status} {data.get('message', '')}".strip())
            return r.status, data

    def _r(self, tail: str) -> str:
        return f"/repos/{self.cfg.repo}{tail}"

    # ------------------------------------------------------------ branch and Pages
    async def ensure(self) -> None:
        """The Pages branch exists (created orphan if missing) and Pages serves it."""
        if self._ready:
            return
        st, _ = await self._call("GET", self._r(f"/branches/{self.cfg.branch}"), ok=(200, 404))
        if st == 404:
            await self._create_branch()
        st, _ = await self._call("GET", self._r("/pages"), ok=(200, 404, 403))
        if st == 404:
            try:
                await self._call("POST", self._r("/pages"),
                                 {"source": {"branch": self.cfg.branch, "path": "/"}}, ok=(201, 409))
            except PagesError as e:                    # needs admin rights; the owner can switch it on
                log.warning("could not enable GitHub Pages (%s); enable it in Settings > Pages", e)
        self._ready = True

    async def _create_branch(self) -> None:
        blobs = []
        for path, text in ((".nojekyll", ""), ("index.html", INDEX)):
            _, b = await self._call("POST", self._r("/git/blobs"),
                                    {"content": base64.b64encode(text.encode()).decode(), "encoding": "base64"})
            blobs.append({"path": path, "mode": "100644", "type": "blob", "sha": b["sha"]})
        _, tree = await self._call("POST", self._r("/git/trees"), {"tree": blobs})
        _, commit = await self._call("POST", self._r("/git/commits"),
                                     {"message": "tri plots", "tree": tree["sha"], "parents": []})
        await self._call("POST", self._r("/git/refs"), {"ref": f"refs/heads/{self.cfg.branch}", "sha": commit["sha"]},
                         ok=(201, 422))

    # ------------------------------------------------------------ upload
    async def publish(self, name: str, data: bytes) -> str:
        """Upload one page; its public URL. The same content is uploaded once per day."""
        path = page_path(name, data)
        async with self._lock:
            await self.ensure()
            st, _ = await self._call("GET", self._r(f"/contents/{path}?ref={self.cfg.branch}"), ok=(200, 404))
            if st == 404:
                await self._call("PUT", self._r(f"/contents/{path}"),
                                 {"message": f"plot {path}", "branch": self.cfg.branch,
                                  "content": base64.b64encode(data).decode()})
        asyncio.ensure_future(self._maybe_prune())
        return f"{self.cfg.base_url}/{path}"

    # ------------------------------------------------------------ prune
    async def _maybe_prune(self) -> None:
        today = _dt.datetime.now(_dt.timezone.utc).date()
        if self.cfg.keep_days <= 0 or self._pruned == today:
            return
        self._pruned = today
        try:
            n = await self.prune(self.cfg.keep_days)
            if n:
                log.info("pruned %d old plots from %s", n, self.cfg.branch)
        except Exception as e:                         # pruning never breaks a plot
            log.warning("pruning plots failed: %s", e)

    async def prune(self, days: float) -> int:
        """Delete plots/<day>/ folders older than `days`; the number of files deleted."""
        cutoff = _dt.datetime.now(_dt.timezone.utc).date() - _dt.timedelta(days=days)
        async with self._lock:
            _, ref = await self._call("GET", self._r(f"/git/ref/heads/{self.cfg.branch}"))
            head = ref["object"]["sha"]
            _, commit = await self._call("GET", self._r(f"/git/commits/{head}"))
            _, tree = await self._call("GET", self._r(f"/git/trees/{commit['tree']['sha']}?recursive=1"))
            old = []
            for t in tree.get("tree", []):
                m = re.match(r"plots/(\d{4}-\d{2}-\d{2})/[^/]+$", t.get("path", ""))
                if t.get("type") == "blob" and m and _dt.date.fromisoformat(m.group(1)) < cutoff:
                    old.append(t["path"])
            if not old:
                return 0
            _, nt = await self._call("POST", self._r("/git/trees"), {
                "base_tree": commit["tree"]["sha"],
                "tree": [{"path": p, "mode": "100644", "type": "blob", "sha": None} for p in old]})
            _, nc = await self._call("POST", self._r("/git/commits"),
                                     {"message": f"prune {len(old)} plots before {cutoff}", "tree": nt["sha"],
                                      "parents": [head]})
            await self._call("PATCH", self._r(f"/git/refs/heads/{self.cfg.branch}"), {"sha": nc["sha"]})
            return len(old)


async def split_pages(pages: Optional[Pages], files: Sequence[tuple]) -> Tuple[List[tuple], List[str], List[str]]:
    """(files to attach, page URLs, errors): HTML files go to Pages when it is
    configured; on failure they stay attachments."""
    keep, urls, errs = [], [], []
    for name, data in files:
        if pages is not None and pages.cfg.enabled and str(name).lower().endswith(".html"):
            try:
                urls.append(await pages.publish(name, data))
                continue
            except Exception as e:
                log.warning("publishing %s failed: %s", name, e)
                errs.append(str(e))
        keep.append((name, data))
    return keep, urls, errs


def _main(argv=None) -> None:
    import argparse
    from .wire import load_dotenv
    ap = argparse.ArgumentParser(prog="python -m bot.pages", description="tri plots on GitHub Pages")
    ap.add_argument("--env", default=".env")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="settings, branch and Pages status")
    p = sub.add_parser("prune", help="delete old plots")
    p.add_argument("--days", type=float, default=None)
    a = ap.parse_args(argv)
    load_dotenv(a.env)
    cfg = PagesConfig.from_env()
    if not cfg.enabled:
        raise SystemExit("set PAGES_REPO (owner/name) and PAGES_TOKEN")

    async def run():
        pg = Pages(cfg)
        try:
            if a.cmd == "check":
                await pg.ensure()
                _, info = await pg._call("GET", pg._r("/pages"), ok=(200, 404, 403))
                print(f"repo {cfg.repo}, branch {cfg.branch}: ready; pages {info.get('html_url') or 'not enabled'}; "
                      f"links use {cfg.base_url}")
            else:
                print(f"deleted {await pg.prune(cfg.keep_days if a.days is None else a.days)} plots")
        finally:
            await pg.close()
    asyncio.run(run())


if __name__ == "__main__":
    _main()
