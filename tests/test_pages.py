"""Interactive plots on GitHub Pages (bot/pages.py), against a fake GitHub."""
import asyncio
import datetime as dt
import json
import re

from bot.pages import Pages, PagesConfig, page_path, split_pages


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeGitHub:
    """Branches, contents, trees and refs of one repository, in memory."""

    def __init__(self, pages=False):
        self.branches, self.files, self.calls, self.pages = {}, {}, [], pages
        self.closed = False

    def request(self, method, url, headers=None, data=None):
        path = url.split("/repos/o/r", 1)[1]
        body = json.loads(data) if data else {}
        self.calls.append((method, path))
        if method == "GET" and path.startswith("/branches/"):
            return _Resp(200 if path[10:] in self.branches else 404, {})
        if path == "/pages":
            if method == "POST":
                self.pages = True
                return _Resp(201, {})
            return _Resp(200 if self.pages else 404, {"html_url": "https://o.github.io/r/"})
        if method == "POST" and path == "/git/blobs":
            return _Resp(201, {"sha": "b" + str(len(self.calls))})
        if method == "POST" and path == "/git/trees":
            return _Resp(201, {"sha": "t" + str(len(self.calls)), "_tree": body})
        if method == "POST" and path == "/git/commits":
            return _Resp(201, {"sha": "c" + str(len(self.calls))})
        if method == "POST" and path == "/git/refs":
            self.branches[body["ref"].split("/")[-1]] = body["sha"]
            return _Resp(201, {})
        m = re.match(r"/contents/([^?]+)", path)
        if m and method == "GET":
            return _Resp(200 if m.group(1) in self.files else 404, {})
        if m and method == "PUT":
            assert body["branch"] == "gh-pages"
            self.files[m.group(1)] = body["content"]
            return _Resp(201, {})
        return _Resp(404, {"message": "not faked"})

    async def close(self):
        self.closed = True


def _run(coro):
    return asyncio.run(coro)


def test_config_and_url():
    c = PagesConfig.from_env({"PAGES_REPO": "flanface/tri", "GITHUB_TOKEN": "x"})
    assert c.enabled and c.branch == "gh-pages" and c.base_url == "https://flanface.github.io/tri"
    assert not PagesConfig.from_env({"PAGES_REPO": "flanface/tri"}).enabled
    assert PagesConfig(repo="a/a.github.io", token="x").base_url == "https://a.github.io"
    p = page_path("concept space.html", b"abc", dt.date(2026, 11, 18))
    assert re.fullmatch(r"plots/2026-11-18/concept-space-[0-9a-f]{12}\.html", p)


def test_publish_creates_branch_once_and_dedupes():
    gh = FakeGitHub()
    pg = Pages(PagesConfig(repo="o/r", token="x", keep_days=0), session=gh)

    async def go():
        a = await pg.publish("concept.html", b"<html>1</html>")
        b = await pg.publish("concept.html", b"<html>1</html>")
        c = await pg.publish("worth.html", b"<html>2</html>")
        return a, b, c
    a, b, c = _run(go())
    assert a == b != c and a.startswith("https://o.github.io/r/plots/")
    assert "gh-pages" in gh.branches and gh.pages
    assert len(gh.files) == 2
    assert sum(1 for m, p in gh.calls if p == "/git/refs") == 1


def test_split_pages_keeps_png_and_falls_back():
    gh = FakeGitHub(pages=True)
    pg = Pages(PagesConfig(repo="o/r", token="x", keep_days=0), session=gh)
    files = [("a.png", b"png"), ("a.html", b"<html/>")]
    keep, urls, errs = _run(split_pages(pg, files))
    assert keep == [("a.png", b"png")] and len(urls) == 1 and not errs
    keep, urls, errs = _run(split_pages(None, files))           # not configured: attach as before
    assert keep == files and not urls

    class Broken(FakeGitHub):
        def request(self, *a, **k):
            return _Resp(500, {"message": "down"})
    pg = Pages(PagesConfig(repo="o/r", token="x", keep_days=0), session=Broken())
    keep, urls, errs = _run(split_pages(pg, files))
    assert keep == files and not urls and errs
