"""Fitted runs in Supabase Storage (v3.7): the database carries the model too.

    python -m bot.runstore publish runs/20261030-152416      # from the machine that fitted it
    python -m bot.runstore fetch /var/data/runs              # on the host (the bot does this itself)
    python -m bot.runstore list

publish uploads a run directory to the bucket TRI_RUN_BUCKET (default
tri-runs) under <name>/, in parts (TRI_RUN_PART_MB, default 6 MB; the free
plan caps a file at 50 MB), with a manifest (size and sha256 of
every file) and an object LATEST naming the run. The stored availability
table of data.npz is dropped on the way (it is recomputed on load, or read
from the avail-*.npz cache that is uploaded with the run), so the upload is
a fraction of the run's size. Every request is retried on network errors
and 5xx answers; part names carry the file's checksum, so a publish that
stopped part-way resumes where it stopped when run again (parts already in
the bucket are skipped). Uploads are written in 64 KB blocks with the
timeout (TRI_RUN_TIMEOUT, default 300 s) on each; a part that keeps stalling,
or that the bucket refuses as too large, is retried at half the size. Game data directories are not uploaded: the
host fetches pocamind/data itself (/model sync, GAMEDATA_SYNC_HOURS).

fetch downloads a run into <root>/<name>/ and writes <root>/LATEST, so
TRI_RUN=<root> follows it. Files already present with the same sha256 are
kept; the host's own game data sync (sync/) is never overwritten for the
same run. Downloads stream to disk (no file is held in memory).

The bot fetches on start and on /model reload when TRI_RUN_REMOTE is set
(latest, or a run name). Needs SUPABASE_URL and SUPABASE_SERVICE_KEY.
"""
from __future__ import annotations

import hashlib
import http.client
import socket
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

DEFAULT_BUCKET = "tri-runs"
PART_BYTES = int(float(os.environ.get("TRI_RUN_PART_MB", "6")) * 1024 * 1024)
MIN_PART = 512 * 1024              # a stalled part is retried at half the size, down to this
BLOCK = 64 * 1024                  # upload write size: the timeout applies per block
RETRIES = 6                        # attempts per request (network errors, 429, 5xx)
_NET_ERRORS = (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
               http.client.HTTPException, OSError)
SKIP_DIRS = {"gamedata", "gd", "analysis", "plots", "reports", "__pycache__", ".pytest_cache"}
SKIP_SUFFIX = (".tmp", ".tmp.npz", ".pyc", ".DS_Store")
KEEP_LOCAL = ("sync/",)            # the host's own game data sync


class RunStoreError(RuntimeError):
    pass


class _Stalled(RunStoreError):
    """An upload that kept timing out or dropping (not an HTTP error)."""


class _HTTPFail(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class RunStore:
    def __init__(self, url: str, key: str, bucket: Optional[str] = None, timeout: Optional[float] = None):
        from .db import normalize_supabase_url
        self.base = normalize_supabase_url(url)[: -len("/rest/v1")]
        self.key = (key or "").strip().strip("'\"").strip()
        self.bucket = bucket or os.environ.get("TRI_RUN_BUCKET") or DEFAULT_BUCKET
        self.timeout = timeout or float(os.environ.get("TRI_RUN_TIMEOUT", "300"))
        self._listed: Dict[str, Dict[str, int]] = {}

    @classmethod
    def from_env(cls) -> "RunStore":
        from .wire import load_dotenv
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RunStoreError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        return cls(url, key)

    # ------------------------------------------------------------ http
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h = {"apikey": self.key, "User-Agent": "tri-runstore"}
        if self.key.startswith("eyJ"):                 # legacy JWT keys; sb_secret_ keys go on apikey only
            h["Authorization"] = f"Bearer {self.key}"
        h.update(extra or {})
        return h

    def _url(self, path: str) -> str:
        return f"{self.base}/storage/v1/{path}"

    def _call(self, method: str, path: str, data: Optional[bytes] = None,
              headers: Optional[Dict[str, str]] = None, ok=(200,)):
        last = None
        for k in range(RETRIES):
            if k:
                time.sleep(min(2 ** k, 30))
            req = urllib.request.Request(self._url(path), data=data, method=method, headers=self._headers(headers))
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                body = e.read()[:300].decode("utf-8", "replace")
                last = RunStoreError(f"{method} {path}: HTTP {e.code} {body}")
                if e.code != 429 and e.code < 500:
                    raise last from None
            except _NET_ERRORS as e:
                last = RunStoreError(f"{method} {path}: {type(e).__name__}: {getattr(e, 'reason', e)}")
        raise last

    def _object(self, name: str) -> str:
        return f"object/{self.bucket}/" + urllib.parse.quote(name)

    def ensure_bucket(self) -> None:
        body = json.dumps(dict(id=self.bucket, name=self.bucket, public=False)).encode()
        try:
            self._call("POST", "bucket", body, {"Content-Type": "application/json"}).close()
        except RunStoreError as e:
            if "already exists" not in str(e) and "409" not in str(e) and "Duplicate" not in str(e):
                raise

    def _post_stream(self, path: str, data: bytes, headers: Dict[str, str]) -> None:
        """One upload, written in BLOCK-sized pieces with the timeout on each write
        (urllib's sendall of the whole body gave one opaque 'write operation timed
        out'). If the server stops reading, its own answer (413 too large, 4xx
        bucket or key errors) is read and raised instead of the timeout."""
        u = urllib.parse.urlsplit(self._url(path))
        Conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = Conn(u.netloc, timeout=self.timeout)
        h = self._headers(headers)
        h["Content-Length"] = str(len(data))
        try:
            conn.putrequest("POST", u.path + (f"?{u.query}" if u.query else ""), skip_accept_encoding=True)
            for k, v in h.items():
                conn.putheader(k, v)
            conn.endheaders()
            mv = memoryview(data)
            try:
                for a in range(0, len(data), BLOCK):
                    conn.send(mv[a:a + BLOCK])
            except _NET_ERRORS:
                early = None
                try:                                    # did the server answer and stop reading?
                    if conn.sock is not None:
                        conn.sock.settimeout(10)
                    r = conn.getresponse()
                    early = (r.status, r.read()[:300].decode("utf-8", "replace"))
                except Exception:
                    pass
                if early and early[0] >= 400:
                    raise _HTTPFail(early[0], f"POST {path}: HTTP {early[0]} {early[1]}") from None
                raise
            r = conn.getresponse()
            body = r.read()
            if r.status >= 300:
                raise _HTTPFail(r.status, f"POST {path}: HTTP {r.status} {body[:300].decode('utf-8', 'replace')}")
        finally:
            conn.close()

    def put(self, name: str, data: bytes, content_type: str = "application/octet-stream",
            attempts: int = RETRIES) -> None:
        """Upload with retries on network errors, 408, 429 and 5xx; other HTTP
        errors are raised at once (they do not go away on a retry)."""
        last = None
        for k in range(attempts):
            if k:
                time.sleep(min(2 ** k, 30))
            try:
                return self._post_stream(self._object(name), data,
                                         {"Content-Type": content_type, "x-upsert": "true"})
            except _HTTPFail as e:
                last = RunStoreError(str(e)) if e.code != 413 else _Stalled(str(e))
                if e.code == 413:                       # bucket size limit below the part size
                    raise last from None
                if e.code not in (408, 429) and e.code < 500:
                    raise last from None
            except _NET_ERRORS as e:
                last = _Stalled(f"POST {name}: {type(e).__name__}: {e}")
        raise last

    def get_bytes(self, name: str) -> bytes:
        with self._call("GET", self._object(name)) as r:
            return r.read()

    def download(self, name: str, fh) -> int:
        """Append object `name` to fh; a transfer cut mid-stream is retried from
        the part's start."""
        start = fh.tell()
        for k in range(RETRIES):
            try:
                n = 0
                with self._call("GET", self._object(name)) as r:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            return n
                        fh.write(b); n += len(b)
            except _NET_ERRORS as e:
                if k == RETRIES - 1:
                    raise RunStoreError(f"GET {name}: {type(e).__name__}: {e}") from None
                fh.seek(start); fh.truncate()
                time.sleep(min(2 ** (k + 1), 30))
        return 0

    def sizes(self, folder: str) -> Dict[str, int]:
        """{object name: size} of the objects directly in `folder` (cached)."""
        if folder not in self._listed:
            out: Dict[str, int] = {}
            off = 0
            while True:
                body = json.dumps(dict(prefix=folder, limit=1000, offset=off)).encode()
                with self._call("POST", f"object/list/{self.bucket}", body, {"Content-Type": "application/json"}) as r:
                    xs = json.loads(r.read() or b"[]")
                for o in xs:
                    if o.get("id") is not None:
                        out[f"{folder}/{o['name']}"] = int((o.get("metadata") or {}).get("size", -1))
                if len(xs) < 1000:
                    break
                off += 1000
            self._listed[folder] = out
        return self._listed[folder]

    def list(self, prefix: str = "") -> List[dict]:
        body = json.dumps(dict(prefix=prefix, limit=1000, offset=0)).encode()
        with self._call("POST", f"object/list/{self.bucket}", body, {"Content-Type": "application/json"}) as r:
            return json.loads(r.read() or b"[]")

    def latest(self) -> Optional[str]:
        try:
            return self.get_bytes("LATEST").decode().strip() or None
        except RunStoreError:
            return None

    def manifest(self, name: str) -> dict:
        return json.loads(self.get_bytes(f"{name}/manifest.json"))


# ------------------------------------------------------------------ files
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_files(run: str) -> List[str]:
    """Files of a run worth uploading (relative, '/'-separated)."""
    out = []
    for root, dirs, files in os.walk(run):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f.endswith(SKIP_SUFFIX) or f.startswith("."):
                continue
            out.append(os.path.relpath(os.path.join(root, f), run).replace(os.sep, "/"))
    return out


def slim_corpus(src: str, dst: str) -> bool:
    """Copy data.npz without its stored availability table (a placeholder is
    written; BuildMatrix.load recomputes availability for a non-empty corpus).
    False (nothing written) if there is nothing to drop."""
    import numpy as np
    with np.load(src, allow_pickle=False) as z:
        B = int(z["shape"][0])
        if not B or z["masks"].size <= z["shape"][1]:
            return False
        arrs = {k: z[k] for k in z.files if k not in ("masks", "gid")}
        arrs["masks"] = np.ones((1, int(z["shape"][1])), bool)
        arrs["gid"] = np.zeros(B, np.int64)
    np.savez_compressed(dst, **arrs)
    return True


def _upload_file(store: RunStore, path: str, base: str, size: int, part_bytes: int, rel: str,
                 log: Callable[[str], None]):
    """Upload `path` as parts named <base>.<offset>-<length>. Parts already in the
    bucket (an interrupted publish) are reused from any offset they start at; a
    part that keeps stalling is retried at half the size, and the smaller size is
    kept for the rest of the run. Returns (parts, reused, part_bytes)."""
    folder, stem = base.rsplit("/", 1)
    have: Dict[int, tuple] = {}
    for obj, sz in store.sizes(folder).items():
        tail = obj.rsplit("/", 1)[1]
        if tail.startswith(stem + "."):
            try:
                off, ln = (int(x) for x in tail[len(stem) + 1:].split("-"))
            except ValueError:
                continue
            if sz == ln:
                have.setdefault(off, (obj, ln))
    parts, reused, off = [], 0, 0
    with open(path, "rb") as f:
        while off < size or not parts:
            if off in have and (have[off][1] > 0 or size == 0):
                obj, ln = have[off]
                parts.append(obj); reused += 1; off += ln
                continue
            n = min(part_bytes, size - off)
            f.seek(off)
            b = f.read(n)
            obj = f"{base}.{off}-{n}"
            try:
                store.put(obj, b, attempts=3)
            except _Stalled as e:
                if part_bytes <= MIN_PART:
                    raise RunStoreError(f"{e}: the connection keeps stalling even at "
                                        f"{MIN_PART // 1024} KB parts") from None
                part_bytes = max(MIN_PART, part_bytes // 2)
                log(f"    {rel}: upload {'refused as too large' if 'HTTP 413' in str(e) else 'stalled'}, "
                    f"retrying in {part_bytes / 1048576:.1f} MB parts")
                continue
            parts.append(obj); off += n
            if size > part_bytes:
                log(f"    {rel}: {off / 1e6:.1f}/{size / 1e6:.1f} MB")
    return parts, reused, part_bytes


CACHE_FILES = ("avail-", "data.avail.npz", "latents.npy", "latents.json", "sidegrades.cache.json")


def warm(run: str, log: Callable[[str], None] = print) -> List[str]:
    """Load the run once so its derived caches exist before upload (v3.8): the
    availability table (model/avail-*.npz), the corpus latents and the mined
    sidegrades. A host that downloads them starts in seconds; without them it
    rebuilds them on its first load (minutes on one CPU), and commands answer
    "still loading" meanwhile. Returns the cache files present afterwards."""
    from tri.runtime import release
    t0 = time.time()
    log(f"warming caches of {os.path.basename(run.rstrip('/'))} (one model load) ...")
    from .service import ModelService
    svc = ModelService(run)
    del svc
    release()
    have = [r for r in run_files(run) if os.path.basename(r).startswith(CACHE_FILES)]
    log(f"  caches ready in {time.time() - t0:.0f}s: {', '.join(have) or 'none'}")
    return have


def publish(run: str, store: RunStore, name: Optional[str] = None, latest: bool = True,
            part_bytes: int = PART_BYTES, log: Callable[[str], None] = print, warm_first: bool = False) -> dict:
    run = os.path.abspath(os.path.expanduser(run))
    if not os.path.isfile(os.path.join(run, "catalog.json")):
        raise RunStoreError(f"{run} is not a tri run (no catalog.json)")
    if warm_first:
        warm(run, log)
    name = name or os.path.basename(run.rstrip("/"))
    store.ensure_bucket()
    tmp = tempfile.mkdtemp(prefix="tri-publish-")
    files: Dict[str, dict] = {}
    try:
        for rel in run_files(run):
            path = os.path.join(run, rel)
            if rel == "model/data.npz":
                slim = os.path.join(tmp, "data.npz")
                if slim_corpus(path, slim):
                    log(f"data.npz: stored availability table dropped "
                        f"({os.path.getsize(path) / 1e6:.0f} -> {os.path.getsize(slim) / 1e6:.0f} MB)")
                    path = slim
            size, sha = os.path.getsize(path), _sha256(path)
            parts, skipped, part_bytes = _upload_file(store, path, f"{name}/files/{rel}.{sha[:12]}",
                                                      size, part_bytes, rel, log)
            files[rel] = dict(size=size, sha256=sha, parts=parts)
            log(f"  {rel}  {size / 1e6:.1f} MB  ({len(parts)} part{'s' if len(parts) > 1 else ''}"
                + (f", {skipped} already uploaded" if skipped else "") + ")")
        man = dict(name=name, created=time.time(), files=files)
        store.put(f"{name}/manifest.json", json.dumps(man, indent=1).encode(), "application/json")
        if latest:
            store.put("LATEST", name.encode(), "text/plain")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = sum(f["size"] for f in files.values())
    log(f"published {name}: {len(files)} files, {total / 1e6:.0f} MB" + (" (LATEST)" if latest else ""))
    return man


def fetch(root: str, store: RunStore, name: str = "latest", log: Callable[[str], None] = print) -> str:
    """Download run `name` (or the LATEST one) into <root>/<name>; returns its path."""
    root = os.path.abspath(os.path.expanduser(root))
    if name in ("", "latest", "LATEST"):
        name = store.latest()
        if not name:
            raise RunStoreError(f"no LATEST in bucket {store.bucket}: publish a run first")
    man = store.manifest(name)
    dest = os.path.join(root, name)
    same_run = os.path.isfile(os.path.join(dest, "catalog.json"))
    got = kept = 0
    for rel, meta in man["files"].items():
        path = os.path.join(dest, *rel.split("/"))
        if os.path.isfile(path):
            if same_run and rel.startswith(KEEP_LOCAL):
                kept += 1; continue
            if os.path.getsize(path) == meta["size"] and _sha256(path) == meta["sha256"]:
                kept += 1; continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part = path + ".download.tmp"
        with open(part, "wb") as fh:
            for obj in meta["parts"]:
                store.download(obj, fh)
        if _sha256(part) != meta["sha256"]:
            os.remove(part)
            raise RunStoreError(f"{rel}: checksum mismatch after download")
        os.replace(part, path)
        got += 1
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "LATEST.tmp"), "w") as f:
        f.write(name)
    os.replace(os.path.join(root, "LATEST.tmp"), os.path.join(root, "LATEST"))
    log(f"run {name}: {got} downloaded, {kept} up to date -> {dest}")
    return dest


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m bot.runstore", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="upload a run")
    p.add_argument("run"); p.add_argument("--name"); p.add_argument("--no-latest", action="store_true")
    p.add_argument("--no-warm", action="store_true",
                   help="upload as is, without loading the run first to build its caches")
    f = sub.add_parser("fetch", help="download a run into ROOT/<name> and set ROOT/LATEST")
    f.add_argument("root"); f.add_argument("--name", default="latest")
    sub.add_parser("list", help="runs in the bucket")
    ap.add_argument("--bucket", default=None)
    a = ap.parse_args(argv)
    st = RunStore.from_env()
    if a.bucket:
        st.bucket = a.bucket
    try:
        if a.cmd == "publish":
            publish(a.run, st, a.name, latest=not a.no_latest, warm_first=not a.no_warm)
        elif a.cmd == "fetch":
            fetch(a.root, st, a.name)
        else:
            cur = st.latest()
            for o in st.list(""):
                if o.get("id") is None:                     # folders
                    print(o["name"] + ("  (LATEST)" if o["name"] == cur else ""))
    except RunStoreError as e:
        sys.exit(f"{e}\n(run the same command again: finished parts are kept and skipped)"
                 if a.cmd == "publish" else str(e))


if __name__ == "__main__":
    main()
