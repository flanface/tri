"""Game data adapter (bot side): every download tri needs.

tri never touches the network; it reads game data from files
(tri.game.gamedata.load_gamedata / load_deepwokenco / cached_gamedata). This
module downloads them:

    python -m bot.gamesource fetch runs/gamedata                     # pocamind/data
    python -m bot.gamesource fetch runs/gamedata --source deepwoken.co
    python -m bot.gamesource head                                    # latest pocamind commit

GITHUB_TOKEN, if set, is sent to the GitHub API (60 unauthenticated requests
an hour per IP otherwise)."""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Optional, Sequence

from tri.api import (API_URL, DWCO_URL, REPO, ZIP_SHA_URL, ZIP_URL, GameData, load_deepwokenco, load_gamedata,
                     merge_gamedata, sync)


REFS_URL = "https://github.com/pocamind/data.git/info/refs?service=git-upload-pack"


def head_commit(ref: str = "main", timeout: float = 15.0) -> dict:
    """{"sha", "date"} of pocamind/data at `ref`. Uses the GitHub API (with
    GITHUB_TOKEN if set; unauthenticated it allows 60 requests an hour per IP).
    If the API refuses (rate limit, outage), falls back to the git protocol's
    ref list, which is not rate limited but carries no date (date is then
    the time of the check)."""
    import time as _time
    hdr = {"User-Agent": "tri", "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        hdr["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    try:
        req = urllib.request.Request(API_URL.format(ref=ref), headers=hdr)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            c = json.load(r)
        return dict(sha=c.get("sha"), date=((c.get("commit") or {}).get("committer") or {}).get("date"))
    except Exception as api_err:
        sha = _ref_sha(ref, timeout)
        if sha is None:
            raise api_err
        return dict(sha=sha, date=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()))


def _ref_sha(ref: str, timeout: float) -> Optional[str]:
    """The sha of a branch or tag from the git smart-HTTP ref advertisement."""
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        return ref.lower()
    req = urllib.request.Request(REFS_URL, headers={"User-Agent": "git/2.40 tri"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    want = {f"refs/heads/{ref}", f"refs/tags/{ref}", ref}
    for line in body.splitlines():
        parts = line[4:].split("\x00")[0].split(" ")       # pkt-line: 4 hex length, "<sha> <ref>"
        if len(parts) == 2 and len(parts[0]) == 40 and parts[1] in want:
            return parts[0]
    return None


def fetch_gamedata(dest: str, ref: str = "main", refresh: bool = False, timeout: float = 60.0) -> GameData:
    """Download pocamind/data (branch/tag/commit `ref`) to dest/pocamind-data.zip
    once, record the commit sha when the GitHub API answers, and load it.
    Reuses the download unless refresh=True."""
    os.makedirs(dest, exist_ok=True)
    zp = os.path.join(dest, "pocamind-data.zip")
    if refresh or not os.path.exists(zp):
        is_sha = len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref.lower())
        url = ZIP_SHA_URL.format(sha=ref) if is_sha else ZIP_URL.format(ref=ref)
        req = urllib.request.Request(url, headers={"User-Agent": "tri"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        zipfile.ZipFile(io.BytesIO(data))                      # validate before writing
        with open(zp + ".tmp", "wb") as f:
            f.write(data)
        os.replace(zp + ".tmp", zp)
        meta = dict(repo=REPO, ref=ref, fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            req = urllib.request.Request(API_URL.format(ref=ref),
                                         headers={"User-Agent": "tri", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                c = json.load(r)
            meta.update(sha=c.get("sha"), commit_date=(c.get("commit") or {}).get("committer", {}).get("date"))
        except Exception as e:                                  # rate limits etc.: sha is optional
            meta["sha_error"] = str(e)[:200]
        with open(os.path.join(dest, "pocamind-data.json"), "w") as f:
            json.dump(meta, f, indent=1)
    return load_gamedata(zp)


def fetch_deepwokenco(dest: str, refresh: bool = False, timeout: float = 60.0) -> GameData:
    """Download DWCO_URL once to dest/deepwokenco.json (reused unless refresh)."""
    os.makedirs(dest, exist_ok=True)
    fp = os.path.join(dest, "deepwokenco.json")
    if refresh or not os.path.exists(fp):
        req = urllib.request.Request(DWCO_URL, headers={"User-Agent": "Mozilla/5.0 (tri)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        d = json.loads(data)
        if not isinstance(d, dict) or not d.get("talents"):
            raise ValueError(f"{DWCO_URL}: unexpected response {str(d)[:200]}")
        with open(fp + ".tmp", "wb") as f:
            f.write(data)
        os.replace(fp + ".tmp", fp)
        with open(os.path.join(dest, "deepwokenco.meta.json"), "w") as f:
            json.dump(dict(url=DWCO_URL, fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           updated_max=max((x.get("_updatedAt") or "" for x in d["talents"]), default=None)),
                      f, indent=1)
    return load_deepwokenco(fp)


def fetch_all(dest: str, refresh: bool = False, ref: str = "main", log=None) -> GameData:
    """deepwoken.co (primary) merged with pocamind/data (fills gaps). Either
    may fail (network, rate limits); the other is then used alone."""
    got, errs = [], []
    for name, fn in (("deepwoken.co", lambda: fetch_deepwokenco(dest, refresh=refresh)),
                     ("pocamind/data", lambda: fetch_gamedata(dest, ref=ref, refresh=refresh))):
        try:
            got.append(fn())
        except Exception as e:
            errs.append(f"{name}: {e}")
            if log:
                log(f"game data: {name} unavailable ({str(e)[:160]})")
    if not got:
        raise RuntimeError("no game data source available: " + "; ".join(errs))
    gd = merge_gamedata(*got) if len(got) > 1 else got[0]
    if errs:
        gd.source["unavailable"] = errs
    return gd


def sync_now(tri, run_dir: Optional[str], cache_dir: str, ref: str = "main", exclude: Sequence[str] = (),
         keep_groups: Sequence[str] = (), dry_run: bool = False, force: bool = False) -> dict:
    """Check pocamind/data and, if its commit moved (or force), plan and apply a
    sync. dry_run: plan only. Returns the plan summary (+ what was applied)."""
    state = sync.load_state(run_dir) if run_dir else None
    head = head_commit(ref)
    if state and head.get("sha") and state.get("sha") == head["sha"] and not force:
        return dict(sha=head["sha"], unchanged=True)
    gd = fetch_gamedata(cache_dir, ref=head.get("sha") or ref, refresh=True)
    p = sync.plan(tri.catalog, gd, state, exclude, keep_groups, sha=head.get("sha"), commit_date=head.get("date"))
    if dry_run:
        return dict(p.summary(), dry_run=True)
    return sync.apply(tri, p, run_dir)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="bot.gamesource")
    ap.add_argument("cmd", choices=["fetch", "head"])
    ap.add_argument("dest", nargs="?", default="runs/gamedata")
    ap.add_argument("--source", default="pocamind", choices=["pocamind", "deepwoken.co", "all"])
    ap.add_argument("--ref", default="main")
    a = ap.parse_args(argv)
    if a.cmd == "head":
        print(json.dumps(head_commit(a.ref)))
        return
    if a.source == "pocamind":
        gd = fetch_gamedata(a.dest, ref=a.ref, refresh=True)
    elif a.source == "deepwoken.co":
        gd = fetch_deepwokenco(a.dest, refresh=True)
    else:
        gd = fetch_all(a.dest, refresh=True, ref=a.ref)
    print(f"{a.dest}: {len(gd.items)} items from {gd.source.get('name')}")


if __name__ == "__main__":
    main(sys.argv[1:])
