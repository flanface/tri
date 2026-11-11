"""Fitted runs in Supabase Storage (v3.7): the database carries the model too.

    python -m bot.runstore publish runs/20261030-152416      # from the machine that fitted it
    python -m bot.runstore fetch /var/data/runs              # on the host (the bot does this itself)
    python -m bot.runstore list

publish uploads a run directory to the bucket TRI_RUN_BUCKET (default
tri-runs) under <name>/, in parts below the Storage file-size limit (45 MB;
the free plan caps a file at 50 MB), with a manifest (size and sha256 of
every file) and an object LATEST naming the run. The stored availability
table of data.npz is dropped on the way (it is recomputed on load, or read
from the avail-*.npz cache that is uploaded with the run), so the upload is
a fraction of the run's size. Game data directories are not uploaded: the
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
PART_BYTES = 45 * 1024 * 1024
SKIP_DIRS = {"gamedata", "gd", "analysis", "plots", "reports", "__pycache__", ".pytest_cache"}
SKIP_SUFFIX = (".tmp", ".tmp.npz", ".pyc", ".DS_Store")
KEEP_LOCAL = ("sync/",)            # the host's own game data sync


class RunStoreError(RuntimeError):
    pass


class RunStore:
    def __init__(self, url: str, key: str, bucket: Optional[str] = None, timeout: float = 120.0):
        from .db import normalize_supabase_url
        self.base = normalize_supabase_url(url)[: -len("/rest/v1")]
        self.key = (key or "").strip().strip("'\"").strip()
        self.bucket = bucket or os.environ.get("TRI_RUN_BUCKET") or DEFAULT_BUCKET
        self.timeout = timeout

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
        req = urllib.request.Request(self._url(path), data=data, method=method, headers=self._headers(headers))
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            raise RunStoreError(f"{method} {path}: HTTP {e.code} {body}") from None

    def _object(self, name: str) -> str:
        return f"object/{self.bucket}/" + urllib.parse.quote(name)

    def ensure_bucket(self) -> None:
        body = json.dumps(dict(id=self.bucket, name=self.bucket, public=False)).encode()
        try:
            self._call("POST", "bucket", body, {"Content-Type": "application/json"}).close()
        except RunStoreError as e:
            if "already exists" not in str(e) and "409" not in str(e) and "Duplicate" not in str(e):
                raise

    def put(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._call("POST", self._object(name), data, {"Content-Type": content_type, "x-upsert": "true"}).close()

    def get_bytes(self, name: str) -> bytes:
        with self._call("GET", self._object(name)) as r:
            return r.read()

    def download(self, name: str, fh) -> int:
        n = 0
        with self._call("GET", self._object(name)) as r:
            while True:
                b = r.read(1 << 20)
                if not b:
                    return n
                fh.write(b); n += len(b)

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


def publish(run: str, store: RunStore, name: Optional[str] = None, latest: bool = True,
            part_bytes: int = PART_BYTES, log: Callable[[str], None] = print) -> dict:
    run = os.path.abspath(os.path.expanduser(run))
    if not os.path.isfile(os.path.join(run, "catalog.json")):
        raise RunStoreError(f"{run} is not a tri run (no catalog.json)")
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
            size = os.path.getsize(path)
            parts = []
            with open(path, "rb") as f:
                k = 0
                while True:
                    b = f.read(part_bytes)
                    if not b and k:
                        break
                    obj = f"{name}/files/{rel}.part{k:03d}"
                    store.put(obj, b)
                    parts.append(obj); k += 1
                    if len(b) < part_bytes:
                        break
            files[rel] = dict(size=size, sha256=_sha256(path), parts=parts)
            log(f"  {rel}  {size / 1e6:.1f} MB  ({len(parts)} part{'s' if len(parts) > 1 else ''})")
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
            publish(a.run, st, a.name, latest=not a.no_latest)
        elif a.cmd == "fetch":
            fetch(a.root, st, a.name)
        else:
            cur = st.latest()
            for o in st.list(""):
                if o.get("id") is None:                     # folders
                    print(o["name"] + ("  (LATEST)" if o["name"] == cur else ""))
    except RunStoreError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
