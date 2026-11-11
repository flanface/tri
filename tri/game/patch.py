"""Patches: what changed, and how old evidence about it is discounted.

A Patch names the items a game update changed. Changes come from two places:
  - the game data itself: diff_catalogs(old, new) finds added, removed and
    renamed items and changed requirements / prerequisites / exclusions;
  - you: patch notes describe nerfs and buffs the game data cannot show, so
    list those items yourself (Patch(changed=[...])).

Effects once tri knows a patch (Tri.apply_patch):
  - pass rules touching a changed item are re-scored on post-patch builds only
    and marked stale (not applied) until enough new builds exist;
  - in worth, pre-patch builds count WorthConfig.patch_factor times for the
    changed items;
  - pre-patch review claims about the changed items count
    AuthorityConfig.patch_factor times and are queued for re-review."""
from __future__ import annotations

import dataclasses
import difflib
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from tri.core.catalog import Catalog
from tri.core.types import Build, Review
from tri.game.defs import Patch
from tri.model.matrix import BuildMatrix, avail_groups


class PatchLog:
    def __init__(self, patches: Sequence[Patch] = ()):
        self.patches: List[Patch] = sorted(patches, key=lambda p: p.time)

    def add(self, p: Patch) -> None:
        self.patches = sorted([x for x in self.patches if x.name != p.name] + [p], key=lambda p: p.time)

    def release_times(self) -> Dict[str, float]:
        """item -> time of the latest patch that added it to the game. Corpus
        builds older than this could not take it: they are no evidence about it."""
        out: Dict[str, float] = {}
        for p in self.patches:
            for i in p.added:
                out[i] = p.time
            for o, n in p.renamed.items():
                out.pop(o, None)
        return out

    def added_items(self) -> List[str]:
        return sorted(self.release_times())

    def change_times(self) -> Dict[str, float]:
        """item -> time of the latest patch that changed it."""
        out: Dict[str, float] = {}
        for p in self.patches:
            for i in p.touched():
                out[i] = p.time
        return out

    def to_json(self) -> list:
        return [asdict(p) for p in self.patches]

    @staticmethod
    def from_json(d: list) -> "PatchLog":
        return PatchLog([Patch(**x) for x in d])


def _fingerprint(it) -> str:
    return json.dumps(dict(kind=it.kind.value, req=it.requires, any=it.requires_any, cl=it.clauses,
                           pre=sorted(it.prerequisites), pre_any=it.prerequisites_any,
                           ex=sorted(it.exclusive_with)), sort_keys=True, default=str)


def diff_catalogs(old: Catalog, new: Catalog, rename_cutoff: float = 0.75) -> dict:
    """Structural differences between two catalogs.

    Returns dict(added, removed, changed, suggested_renames). A suggested rename
    pairs a removed and an added item of the same kind with identical
    requirements and similar names; confirm it before passing it to migrate."""
    a, b = set(old.ids), set(new.ids)
    added, removed = sorted(b - a), sorted(a - b)
    changed = sorted(i for i in a & b if _fingerprint(old.items[i]) != _fingerprint(new.items[i]))
    ren: Dict[str, str] = {}
    pool = list(added)
    for r in removed:
        fp = _fingerprint(old.items[r])
        same = [x for x in pool if _fingerprint(new.items[x]) == fp]
        m = difflib.get_close_matches(r, same, n=1, cutoff=rename_cutoff)
        if m:
            ren[r] = m[0]; pool.remove(m[0])
    return dict(added=added, removed=removed, changed=changed, suggested_renames=ren)


def patch_from_diff(name: str, time: float, diff: dict, renames: Optional[Dict[str, str]] = None,
                    extra_changed: Sequence[str] = (), note: str = "") -> Patch:
    """Patch from diff_catalogs output plus the nerfed/buffed items you list."""
    ren = dict(diff["suggested_renames"] if renames is None else renames)
    return Patch(name, float(time), changed=sorted(set(diff["changed"]) | set(extra_changed)),
                 added=[x for x in diff["added"] if x not in ren.values()],
                 removed=[x for x in diff["removed"] if x not in ren], renamed=ren, note=note)
# ---------------------------------------------------------------- migration


def remap_matrix(bm: BuildMatrix, new: Catalog, renames: Dict[str, str],
                 cache_dir: Optional[str] = None) -> BuildMatrix:
    """Same builds, new catalog: renamed items follow, removed items drop,
    availability recomputed with the new requirements (cached by content in
    cache_dir, v3.3). No data is lost."""
    old_ids = bm.catalog.ids
    col = np.array([new.index.get(renames.get(i, i), -1) for i in old_ids])
    X = bm.X.tocoo()
    keep = col[X.col] >= 0
    X2 = sp.csr_matrix((X.data[keep], (X.row[keep], col[X.col[keep]])), shape=(len(bm), len(new)))
    X2.sum_duplicates(); X2.data[:] = 1.0
    if list(new.stats) != list(bm.catalog.stats):
        raise ValueError("stat list changed; rebuild the matrix from builds instead")
    gid, masks = avail_groups(new, bm.pr, bm.fi, bm.T, bm.tokens, bm.lk, bm.lk_vocab, cache_dir=cache_dir)
    from tri.core.types import Kind
    tal = np.array([new.items[i].kind == Kind.TALENT for i in new.ids])
    n_tal = np.asarray(X2 @ tal.astype(np.float64)).ravel().astype(np.int64)
    n_man = np.asarray(X2.sum(1)).ravel().astype(np.int64) - n_tal
    return dataclasses.replace(bm, catalog=new, X=X2, gid=gid, masks=masks, n_tal=n_tal, n_man=n_man)


def remap_build(b: Build, renames: Dict[str, str], new: Catalog) -> Build:
    f = lambda xs: [renames.get(x, x) for x in xs if renames.get(x, x) in new.index]
    return dataclasses.replace(b, talents=f(b.talents), mantras=f(b.mantras))


def remap_review(r: Review, renames: Dict[str, str], new: Catalog) -> Optional[Review]:
    """Review with renamed items; operations on removed items are dropped
    (None if nothing is left)."""
    ops = []
    for o in r.operations:
        i = renames.get(o.item_id, o.item_id)
        if i not in new.index:
            continue
        ops.append(dataclasses.replace(o, item_id=i, related=[renames.get(x, x) for x in o.related
                                                              if renames.get(x, x) in new.index]))
    if r.operations and not ops:
        return None
    return dataclasses.replace(r, original=remap_build(r.original, renames, new), operations=ops)
