"""Follow pocamind/data without refitting.

    from tri import sync
    plan = sync.plan(tri.catalog, gamedata, state)          # what changed since the last sync
    rep = sync.apply(tri, plan, run_dir)                     # extend the model, write <run>/sync/
    sync.restore(tri, run_dir)                               # at startup: re-apply the saved sync

A sync compares the latest game data with the snapshot kept at the previous
sync (the first sync has none):

  added    slot-taking talents and mantras in the game data that the model's
           catalog lacks (new releases, and on the first sync also the items
           tri.cli.fit left out for having fewer than min_item_count picks) - except
           items the run excluded on purpose. They enter unrated
           (Tri.extend_catalog): the field gets a zero direction for them, so
           nothing else in the model moves, and corpus builds older than the
           sync never count against them. Reviews rate them at once.
  changed  catalog items whose requirements, prerequisites, exclusions, kind
           or slot flags differ from the previous snapshot. They take the new
           definitions and their certified rules go stale.
  removed  catalog items that were in the previous snapshot and are gone now.
           They stay in the model as retired (never available, never proposed).

Comparisons are snapshot to snapshot, never game data to the fitted catalog,
so differences between data sources (deepwoken.co and pocamind) are not
mistaken for patches.

The sync is saved in <run>/sync/: catalog.json (the extended catalog),
state.json (commit, snapshot, the patches applied) and patches.json (the same
patches in the --patches format of tri.cli.fit, so the next fit keeps the new items
and knows when they were released).
"""
from __future__ import annotations

import copy
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from tri.core.catalog import Catalog
from tri.core.types import Item
from tri.game.defs import GameItem, Patch
from tri.game.gamedata import API_URL, GameData, fetch_gamedata, name_key

SYNC_DIR = "sync"


REFS_URL = "https://github.com/pocamind/data.git/info/refs?service=git-upload-pack"


def head_commit(ref: str = "main", timeout: float = 15.0) -> dict:
    """{"sha", "date"} of pocamind/data at `ref`. Uses the GitHub API (with
    GITHUB_TOKEN if set; unauthenticated it allows 60 requests an hour per IP).
    If the API refuses (rate limit, outage), falls back to the git protocol's
    ref list, which is not rate limited but carries no date (date is then
    the time of the check)."""
    import os
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


def _fp(it: GameItem) -> str:
    """The v2.11 form (total mantra slots granted), plus the Vow of Iron flag
    only where it is set, so other items keep their fingerprint across
    versions (no spurious "changed")."""
    x = [it.kind.value, it.clauses, sorted(sorted(g) for g in it.prereqs), sorted(it.exclusive),
         it.slot, it.implicit, it.mtype, it.mantra_slots]
    if it.voi_only:
        x.append("voi_only")
    return json.dumps(x, sort_keys=True, default=str)


def snapshot(gd: GameData) -> Dict[str, str]:
    """{name: fingerprint} of everything a sync compares."""
    return {n: _fp(it) for n, it in gd.items.items()}


@dataclass
class SyncPlan:
    catalog: Catalog                  # the catalog to extend the model to
    added: List[str]
    changed: List[str]
    removed: List[str]
    renames: Dict[str, str]           # suggested (removed -> added); applied as removed + added
    snapshot: Dict[str, str]
    sha: Optional[str] = None
    time: float = 0.0                 # commit time (release time of the added items)
    first: bool = False
    relocked: List[str] = field(default_factory=list)   # items whose context locks were added

    def empty(self) -> bool:
        return not (self.added or self.changed or self.removed or self.relocked)

    def summary(self) -> dict:
        return dict(sha=self.sha, first=self.first, added=self.added, changed=self.changed,
                    removed=self.removed, renames=self.renames, relocked=self.relocked)


def _parse_time(s: Optional[str]) -> float:
    if not s:
        return time.time()
    import calendar
    return float(calendar.timegm(time.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")))


def merge_locks(items: Dict[str, Item], *sources: Catalog) -> List[str]:
    """Add the context clauses (oath, murmur, origin locks) that
    `sources` state for items in `items` and the items lack. Clauses are only
    added, never removed: a source that does not state a lock does not lift
    one. Returns the items that changed."""
    out = []
    for src in sources:
        for n, it in items.items():
            si = src.items.get(n)
            if si is None:
                continue
            new = [list(c) for c in si.context if sorted(c) not in [sorted(x) for x in it.context]]
            if new:
                it.context = [list(c) for c in it.context] + new
                out.append(n)
    return sorted(set(out))


def plan(catalog: Catalog, gd: GameData, state: Optional[dict] = None, exclude: Sequence[str] = (),
         keep_groups: Sequence[str] = (), sha: Optional[str] = None, commit_date: Optional[str] = None,
         locks_from: Sequence[GameData] = ()) -> SyncPlan:
    """What a sync to `gd` would change. state: the previous state.json (None on
    the first sync). exclude: names the run left out on purpose (excluded.json).
    Context locks of every kept item are refreshed from `gd` and `locks_from`
    (e.g. deepwoken.co) and listed in `relocked`;
    they make items unavailable and are not a change of the item (no stale
    rules), so a run fitted before the locks existed gets them on its next sync."""
    snap = snapshot(gd)
    prev = (state or {}).get("snapshot") or {}
    first = not prev
    full = gd.catalog(stats=catalog.stats, ruleset=catalog.ruleset, keep_groups=keep_groups)
    ex = set(exclude)
    # v2.25.1: names match by name_key (casefold, alphanumerics), as builds are
    # parsed. "To the Finish" in the game data IS the fitted "To The Finish";
    # an exact comparison added it again as an unrated twin, which completions
    # then picked beside the original (one talent short in game) and which split
    # the reviews' evidence.
    alias = catalog_aliases(catalog, full.ids)          # game name -> catalog id (spelling differs)
    gname = {v: k for k, v in alias.items()}            # catalog id -> game name
    g = lambda i: gname.get(i, i)                       # noqa: E731
    ex_keys = {name_key(x) for x in ex}
    added = sorted(n for n in full.ids if n not in catalog.index and n not in alias and name_key(n) not in ex_keys)
    changed = sorted(i for i in catalog.ids if g(i) in prev and g(i) in snap and prev[g(i)] != snap[g(i)]
                     and g(i) in full.index)
    removed = sorted(i for i in catalog.ids if g(i) in prev and g(i) not in snap)
    import difflib
    renames = {}
    for r in removed:
        m = difflib.get_close_matches(r, added, n=1, cutoff=0.75)
        if m:
            renames[r] = m[0]
    names = (set(catalog.ids) & set(full.ids)) | set(added) | set(alias)
    sub = _renamed(gd.catalog(names=names, stats=catalog.stats, ruleset=catalog.ruleset, keep_groups=keep_groups),
                   alias)
    items = {i: copy.deepcopy(catalog.items[i]) for i in catalog.ids if i not in removed}
    for n in added + changed:
        items[n] = copy.deepcopy(sub.items[n])
    ids = set(items)
    for it in items.values():                         # references stay inside the catalog
        it.exclusive_with = sorted(x for x in it.exclusive_with if x in ids)
        it.prerequisites = [x for x in it.prerequisites if x in ids]
        it.prerequisites_any = [g for g in ([x for x in g if x in ids] for g in it.prerequisites_any) if g]
    lock_src = [sub] + [_renamed(o.catalog(names=set(items) | set(alias), stats=catalog.stats, ruleset=catalog.ruleset,
                                           keep_groups=keep_groups), alias) for o in locks_from]
    relocked = [n for n in merge_locks(items, *lock_src) if n not in added]
    new = Catalog(items, stats=catalog.stats, ruleset=catalog.ruleset, talent_cap_base=catalog.talent_cap_base,
                  cap_exempt=catalog.cap_exempt, cap_per_mantra=catalog.cap_per_mantra,
                  mantra_locks=catalog.mantra_locks, lock_fields=catalog.lock_fields)
    return SyncPlan(new, added, changed, removed, renames, snap, sha, _parse_time(commit_date), first,
                    relocked=relocked)


def catalog_aliases(catalog: Catalog, names) -> Dict[str, str]:
    """{name: catalog id} for names that are a catalog item spelled differently
    (same kind is not required: name_key collisions across kinds do not occur
    in the game data, and a talent and a mantra never share a name)."""
    keys = {name_key(i): i for i in catalog.ids}
    out = {}
    for n in names:
        i = keys.get(name_key(n))
        if i is not None and i != n and n not in catalog.index:
            out[n] = i
    return out


def _renamed(cat: Catalog, alias: Dict[str, str]) -> Catalog:
    """cat with game spellings replaced by the catalog ids (items and references)."""
    if not alias:
        return cat
    r = lambda x: alias.get(x, x)                       # noqa: E731
    items = {}
    for n, it in cat.items.items():
        it = copy.deepcopy(it)
        it.id = r(it.id)
        it.exclusive_with = sorted({r(x) for x in it.exclusive_with})
        it.prerequisites = [r(x) for x in it.prerequisites]
        it.prerequisites_any = [[r(x) for x in grp] for grp in it.prerequisites_any]
        items[r(n)] = it
    return Catalog(items, stats=cat.stats, ruleset=cat.ruleset, talent_cap_base=cat.talent_cap_base,
                   cap_exempt=cat.cap_exempt, cap_per_mantra=cat.cap_per_mantra,
                   mantra_locks=cat.mantra_locks, lock_fields=cat.lock_fields)


def twins(fitted, synced) -> Dict[str, str]:
    """{twin: fitted id}: items a sync added that only respell a fitted item
    (the pre-2.25.1 exact-name bug). fitted, synced: catalogs or id lists."""
    f = list(getattr(fitted, "ids", fitted))
    keys = {name_key(i): i for i in f}
    fs = set(f)
    return {n: keys[name_key(n)] for n in getattr(synced, "ids", synced)
            if n not in fs and name_key(n) in keys}


def _drop_twins(cat: Catalog, tw: Dict[str, str]) -> Catalog:
    """cat without the twins; references to a twin point at the fitted item."""
    r = lambda x: tw.get(x, x)                          # noqa: E731
    items = {}
    for n, it in cat.items.items():
        if n in tw:
            continue
        it = copy.deepcopy(it)
        it.exclusive_with = sorted({r(x) for x in it.exclusive_with} - {n})
        it.prerequisites = [r(x) for x in it.prerequisites]
        it.prerequisites_any = [[r(x) for x in grp] for grp in it.prerequisites_any]
        items[n] = it
    return Catalog(items, stats=cat.stats, ruleset=cat.ruleset, talent_cap_base=cat.talent_cap_base,
                   cap_exempt=cat.cap_exempt, cap_per_mantra=cat.cap_per_mantra,
                   mantra_locks=cat.mantra_locks, lock_fields=cat.lock_fields)


def _paths(run_dir: str) -> dict:
    d = os.path.join(run_dir, SYNC_DIR)
    return dict(dir=d, catalog=os.path.join(d, "catalog.json"), state=os.path.join(d, "state.json"),
                patches=os.path.join(d, "patches.json"))


def load_state(run_dir: str) -> Optional[dict]:
    p = _paths(run_dir)["state"]
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def apply(tri, p: SyncPlan, run_dir: Optional[str] = None) -> dict:
    """Extend the model to the plan (no refit) and save the sync in <run>/sync/."""
    state = load_state(run_dir) if run_dir else None
    rep = dict(p.summary())
    if not p.empty():
        name = f"pocamind-{(p.sha or 'local')[:7]}"
        rep.update(tri.extend_catalog(p.catalog, time_=p.time, name=name,
                                      note=f"pocamind/data {p.sha or ''}: game data sync"))
    if run_dir:
        from tri.sources.pipeline import catalog_to_json
        paths = _paths(run_dir)
        os.makedirs(paths["dir"], exist_ok=True)
        catalog_to_json(tri.catalog, paths["catalog"] + ".tmp")
        os.replace(paths["catalog"] + ".tmp", paths["catalog"])
        mine = [x for x in tri.patches.to_json() if str(x.get("name", "")).startswith("pocamind-")]
        st = dict(sha=p.sha, time=p.time, synced_at=time.time(), snapshot=p.snapshot, patches=mine,
                  history=((state or {}).get("history") or []) + ([p.summary()] if not p.empty() else []))
        with open(paths["state"] + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(paths["state"] + ".tmp", paths["state"])
        with open(paths["patches"], "w") as f:
            json.dump(mine, f, indent=1)
    return rep


def restore(tri, run_dir: str) -> Optional[dict]:
    """Re-apply the saved sync to a freshly loaded run (startup, /model reload).
    None if the run has never been synced."""
    paths = _paths(run_dir)
    st = load_state(run_dir)
    if st is None or not os.path.exists(paths["catalog"]):
        return None
    cat = Catalog.from_json(paths["catalog"])
    tw = twins(tri.catalog, cat)
    if tw:                                             # v2.25.1: repair a sync saved by the exact-name bug
        cat = _drop_twins(cat, tw)
        st["patches"] = [dict(x, added=[a for a in x.get("added") or [] if a not in tw])
                         for x in st.get("patches") or []]
        from tri.sources.pipeline import catalog_to_json
        catalog_to_json(cat, paths["catalog"] + ".tmp")
        os.replace(paths["catalog"] + ".tmp", paths["catalog"])
        st.setdefault("repairs", []).append(dict(at=time.time(), twins_dropped=tw))
        with open(paths["state"] + ".tmp", "w") as f:
            json.dump(st, f)
        os.replace(paths["state"] + ".tmp", paths["state"])
        if os.path.exists(paths["patches"]):
            with open(paths["patches"], "w") as f:
                json.dump(st["patches"], f, indent=1)
    same_locks = all(
        sorted(map(sorted, cat.items[i].context)) == sorted(map(sorted, tri.catalog.items[i].context))
        for i in cat.ids if i in tri.catalog.index)
    if set(cat.ids) == set(tri.catalog.ids) and not st.get("patches") and same_locks:
        return dict(sha=st.get("sha"), added=[], changed=[], retired=[])
    patches = [Patch(**x) for x in st.get("patches") or []]
    return tri.extend_catalog(cat, time_=st.get("time"), name=None, patches=patches)


def sync(tri, run_dir: Optional[str], cache_dir: str, ref: str = "main", exclude: Sequence[str] = (),
         keep_groups: Sequence[str] = (), dry_run: bool = False, force: bool = False) -> dict:
    """Check pocamind/data and, if its commit moved (or force), plan and apply a
    sync. dry_run: plan only. Returns the plan summary (+ what was applied)."""
    state = load_state(run_dir) if run_dir else None
    head = head_commit(ref)
    if state and head.get("sha") and state.get("sha") == head["sha"] and not force:
        return dict(sha=head["sha"], unchanged=True)
    gd = fetch_gamedata(cache_dir, ref=head.get("sha") or ref, refresh=True)
    p = plan(tri.catalog, gd, state, exclude, keep_groups, sha=head.get("sha"), commit_date=head.get("date"))
    if dry_run:
        return dict(p.summary(), dry_run=True)
    return apply(tri, p, run_dir)
