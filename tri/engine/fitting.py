"""Tri methods: fitting the model: fit, refit, worth refresh, calibration, catalog extension, patches and drift."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from tri.analysis.sidegrades import mine as mine_sidegrades
from tri.core.catalog import Catalog
from tri.core.types import Action, Build, Item, Kind
from tri.engine.defs import _build_key
from tri.game.defs import Patch
from tri.game.patch import remap_matrix, remap_review
from tri.infer.engine import PassEngine
from tri.model.matrix import BuildMatrix
from tri.model.space import ConceptSpace
from tri.model.worth import FieldWorth
from tri.rules.induction import context_features
from tri.rules.passes import Rulebook

COORDS_DEFAULT = "float32"     # TRI_COORDS (v3.7)


class _Fitting:
    """Tri methods (Tri methods: fitting the model: fit, refit, worth refresh, calibration, catalog extension, patches and drift.)."""

    # ------------------------------------------------------------ fitting
    def fit(self, builds, weights: Optional[Sequence[float]] = None, verbose: bool = False) -> dict:
        """Fit on corpus builds (list of Build, pipeline.CompactBuilds, or a BuildMatrix).
        Keeps a uniform sample of cfg.data_size builds for later refits."""
        t0 = time.time()
        bm = builds if isinstance(builds, BuildMatrix) else BuildMatrix.from_builds(
            builds, self.catalog, weights, self.cfg.context_fields)
        self.data = self._pathfinder_rows(bm).sample(self.cfg.data_size, self.cfg.seed)
        rep = self.refit(verbose=verbose)
        rep["seconds"] = round(time.time() - t0, 2)
        return rep

    def refit(self, verbose: bool = False) -> dict:
        """Refit the concept space and re-mine the pass rules on the kept corpus
        plus every active review (corrected builds, weighted). Seconds."""
        if self.data is None:
            raise RuntimeError("fit() first")
        t0 = time.time()
        self.data = self._pathfinder_rows(self.data)
        self._ledger_and_rules()
        rev = self._review_matrix()
        rev = self._pathfinder_rows(rev) if rev is not None else None
        rev = rev if rev is not None and len(rev) else None
        if rev is not None:          # corrected builds count refit_review_weight each, capped at a share of the corpus
            tot = float(self.data.w.sum())
            k = self.cfg.refit_review_weight
            if self.cfg.refit_review_share < 1:
                cap = self.cfg.refit_review_share / (1 - self.cfg.refit_review_share) * tot
                k = min(k, cap / max(float(rev.w.sum()), 1e-12))
            rev.w = rev.w * k
        full = self.data.append(rev) if rev is not None else self.data
        is_rev = np.r_[np.zeros(len(self.data), bool), np.ones(len(full) - len(self.data), bool)]
        # locator builds never train the field: their reading is out of sample
        lids = {l.build.build_id for l in self.locators if l.build.build_id is not None}
        keep = np.array([i not in lids for i in full.ids[: len(self.data)]] + [True] * (len(full) - len(self.data)))
        self.held_out_ids = sorted({i for i in self.data.ids if i in lids})
        fit_bm = full if keep.all() else full.subset(np.flatnonzero(keep))
        self.__dict__.pop("_place_cache", None)
        self.space = ConceptSpace(self.catalog, self.cfg.space).fit(fit_bm, self._change_array(), is_rev[keep],
                                                                    release_time=self._release_array())
        self.metric_selection = None
        self._locset()
        if self.locset is not None and self.cfg.locators.select_metric and self.space.omega is not None \
                and self.locset.eligible():
            sel = self.locset.select_power(fit_bm, self.space.cfg.metric_power)
            self.metric_selection = sel
            if sel["moved"]:
                self.space.cfg.metric_power = sel["chosen"]
                self.space.set_metric(fit_bm, self.space.cfg.metric)
                self.locset.refresh()
        self._name_regions()
        Z = self.space.coords(full)
        self._Zdata = np.ascontiguousarray(Z[: len(self.data)], dtype=self._coord_dtype())
        self.rulebook = Rulebook(self.catalog, self.cfg.passes)
        self.rulebook.reader = self._archetype_reader
        self.rulebook.mine(full, None, self.space.region_names, changed_since=self.patches.change_times())
        self._mine_sidegrades()
        self._ledger_and_rules()                       # similarity now uses the new space
        self._refresh_worth()
        self.fitted_at = time.time()
        cal = self.calibrate() if len(self.reviews) >= self.cfg.calibrate_min else None
        rep = dict(builds=len(self.data), reviews=len(self.reviews), space=self.space.report(),
                   rules=self.rulebook.summary(), calibration=cal, locators=self.locator_report(),
                   seconds=round(time.time() - t0, 2))
        if verbose:
            print(json.dumps(rep, indent=1))
        return rep

    def _pathfinder_rows(self, bm: BuildMatrix) -> BuildMatrix:
        """bm without the builds that hold a Vow of Iron only pick: those are
        Vow of Iron builds, and tri learns Pathfinder builds only."""
        voi = [self.catalog.index[i] for i in self.catalog.ids if self.catalog.voi_only(i)]
        if not voi or not len(bm):
            return bm
        bad = np.asarray(bm.X[:, voi].sum(1)).ravel() > 0
        return bm if not bad.any() else bm.subset(np.flatnonzero(~bad))

    def _release_array(self) -> np.ndarray:
        rt = np.full(len(self.catalog), -np.inf)
        for i, t in self.patches.release_times().items():
            if i in self.catalog.index:
                rt[self.catalog.index[i]] = t
        return rt

    def _mine_sidegrades(self) -> None:
        """Either/or clusters per region, from the corpus only (tri.sidegrades)."""
        if self.data is None or self.space is None or self._Zdata is None:
            self.sidegrades = []
            return
        regions = self.space.region_of(self._Zdata)
        d = getattr(self, "_run_path", None)
        key = f = None
        if d:                                          # v3.3: cached by content (corpus, regions, settings)
            import dataclasses as _dc
            import hashlib
            import json as _json
            import os
            h = hashlib.blake2b(digest_size=16)
            X = self.data.X
            for a in (X.indptr, X.indices, self.data.w, np.asarray(regions)):
                h.update(memoryview(np.ascontiguousarray(a)))
            h.update("\x1f".join(self.catalog.ids).encode())
            h.update(_json.dumps(_dc.asdict(self.cfg.sidegrades), sort_keys=True, default=str).encode())
            key, f = h.hexdigest(), os.path.join(d, "sidegrades.cache.json")
            try:
                with open(f) as fh:
                    c = _json.load(fh)
                if c.get("key") == key:
                    from tri.analysis.defs import Sidegrade
                    self.sidegrades = [Sidegrade(**g) for g in c["items"]]
                    return
            except (OSError, ValueError, TypeError, KeyError):
                pass
        self.sidegrades = mine_sidegrades(self.data, regions, self.catalog, self.cfg.sidegrades)
        if f:
            try:
                import dataclasses as _dc
                import json as _json
                with open(f, "w") as fh:
                    _json.dump(dict(key=key, items=[_dc.asdict(g) for g in self.sidegrades]), fh, default=float)
            except (OSError, TypeError):
                pass

    def _change_array(self) -> np.ndarray:
        ct = np.full(len(self.catalog), -np.inf)
        for i, t in self.patches.change_times().items():
            if i in self.catalog.index:
                ct[self.catalog.index[i]] = t
        return ct

    def _placed(self, b: Build) -> Tuple[np.ndarray, int]:
        """(concept coordinates u, region) of a build, memoised per field: the
        ledger and rule induction ask for the same review builds many times."""
        cache = self.__dict__.setdefault("_place_cache", {})
        sp_ = self.space
        key = (id(sp_), id(sp_.V), id(getattr(sp_, "centroids", None)), id(self.catalog), _build_key(b))
        hit = cache.get(key)
        if hit is None:
            if len(cache) > 50000:
                cache.clear(); self.__dict__.pop("_place_builds", None)
            u = self.space.point(b).u
            hit = cache[key] = (u, int(self.space.region_of(u[None])[0]))
            self.__dict__.setdefault("_place_builds", {})[key] = b
            if self.locset is not None and self.locset.refining:
                self.locset.note(u, b)               # fine reading, for the review barrier
        return hit

    def _renote(self) -> None:
        """The locator set changed: coordinates stay, fine readings are re-noted (v3.1)."""
        cache = self.__dict__.get("_place_cache") or {}
        pb = self.__dict__.get("_place_builds") or {}
        if self.locset is not None and self.locset.refining:
            for k, (u, _) in cache.items():
                if k in pb:
                    self.locset.note(u, pb[k])

    def _place_many(self, builds) -> None:
        """Fill the placement memo for many builds with one batched solve (v3.1)."""
        if self.space is None:
            return
        cache = self.__dict__.setdefault("_place_cache", {})
        sp_ = self.space
        head = (id(sp_), id(sp_.V), id(getattr(sp_, "centroids", None)), id(self.catalog))
        todo, seen = [], set()
        for b in builds:
            k = head + (_build_key(b),)
            if k not in cache and k not in seen:
                seen.add(k); todo.append((k, b))
        if len(todo) < 2:
            return
        U = sp_.u_of(sp_.ys([b for _, b in todo]))
        reg = sp_.region_of(U)
        refine = self.locset is not None and self.locset.refining
        pb = self.__dict__.setdefault("_place_builds", {})
        for (k, b), u, g in zip(todo, U, reg):
            cache[k] = (u, int(g))
            pb[k] = b
            if refine:
                self.locset.note(u, b)

    def _vec(self, b: Build) -> Tuple[np.ndarray, np.ndarray]:
        """(picks x [N], availability a [N]) of a build, as the corpus matrix sees it."""
        bm = self.space._bm(b)
        return bm.X[0].toarray()[0].astype(np.float64), bm.masks[bm.gid[0]].astype(np.float64)

    def _embed(self, b: Build) -> np.ndarray:
        return self._placed(b)[0].copy()

    def _region_feats(self, b: Build) -> set:
        f = context_features(b)
        if self.space is not None:
            f.add(f"region:{self._placed(b)[1]}")
            fam = self._family_of(b)
            if fam:
                f.add(f"family:{fam}")               # scope of OFF_ARCHETYPE rules (strong SM)
        return f

    def _family_of(self, b: Build) -> Optional[str]:
        """Family key of b's archetype, or None. A reviewed build's family is the
        one its reviewer states (# is a, else # a: ... notes naming a single
        family; v3.0.1), else the one it is located in, unless a # not note
        denies it. Reviews are accurate: a statement outranks the reading."""
        st = (getattr(self, "_stated", None) or {}).get(id(b))
        if st is not None and st[0] is b and st[1]:
            return st[1]
        ls = getattr(self, "locset", None)
        if ls is None or self.space is None or len(ls.keys) < 2:
            return None
        from tri.reviews.variants import family_key
        rd = ls.read(self._embed(b), b)
        fam = family_key(rd.archetype) if rd.archetype else None
        if fam is not None and st is not None and st[0] is b and fam in st[2]:
            return None
        return fam

    @staticmethod
    def stated_family(r) -> tuple:
        """(family key or None, families denied) a review's notes state about its build."""
        from tri.reviews.variants import family_key
        ns = list(getattr(r, "notes", None) or [])
        key = lambda n: family_key(str(n.get("archetype") or "")) if str(n.get("archetype") or "").strip() else None
        is_ = {key(n) for n in ns if n.get("kind") == "is"} - {None}
        not_ = {key(n) for n in ns if n.get("kind") == "not"} - {None}
        if not is_:
            is_ = {key(n) for n in ns if n.get("kind") == "archetype"} - {None}
        is_ -= not_
        return (next(iter(is_)) if len(is_) == 1 else None), frozenset(not_)

    def _note_stated(self) -> None:
        """Index the reviewed builds' stated families (by build identity)."""
        self._stated = {}
        for r in self.reviews or []:
            fam, nots = self.stated_family(r)
            if fam or nots:
                self._stated[id(r.original)] = (r.original, fam, nots)

    def stated_by_review(self) -> dict:
        """{review_id: family key} of reviews whose notes state their build's family."""
        out = {}
        for r in self.reviews or []:
            fam, _ = self.stated_family(r)
            if fam:
                out[str(r.review_id)] = fam
        return out

    def _archetype_reader(self, bm: BuildMatrix, chunk: int = 8192) -> Optional[List[Optional[str]]]:
        """Archetype family key per row of bm (None = unlocated), read coarsely by the
        locators at each row's place in the concept space (the A = a literal, v3.0)."""
        ls = getattr(self, "locset", None)
        if ls is None or self.space is None or len(ls.keys) < 2:
            return None
        from tri.reviews.variants import family_key
        Z = self._Zdata if (bm is self.data and self._Zdata is not None) else self.space.coords(bm)
        if len(Z) > len(bm):
            Z = Z[: len(bm)]
        if len(Z) < len(bm):                          # e.g. the corpus plus review rows
            Z = np.vstack([Z, self.space.coords(bm.subset(np.arange(len(Z), len(bm))))])
        fam = [family_key(ls.names[k]) for k in ls.keys]
        out: List[Optional[str]] = []
        for a in range(0, len(Z), chunk):
            P, d, _ = ls.read_many(Z[a:a + chunk])
            k = P.argmax(1)
            ok = (d <= ls.reach) & (P.max(1) >= ls.cfg.min_confidence)
            out += [fam[int(i)] if o else None for i, o in zip(k, ok)]
        return out

    def mine_rules(self, items=None, reset_config: bool = True, verbose: bool = False) -> dict:
        """Re-mine the rulebook on the stored corpus without touching the concept
        space (v3.0: picks up group, archetype and weapon literals and the
        stricter certification in a run saved before them). reset_config:
        mine with today's PassConfig defaults (keeping seed, sample, n_jobs and item_rows)."""
        self._need()
        t0 = time.time()
        if reset_config:
            from tri.rules.defs import PassConfig
            old = self.cfg.passes
            self.cfg.passes = PassConfig(seed=old.seed, sample=old.sample, n_jobs=old.n_jobs, item_rows=old.item_rows)
        if items is None:
            self.rulebook = Rulebook(self.catalog, self.cfg.passes)
        else:
            self.rulebook.cfg = self.cfg.passes
        self.rulebook.reader = self._archetype_reader
        self.rulebook.mine(self.data, None, self.space.region_names,
                           changed_since=self.patches.change_times(), items=items)
        self._ledger_and_rules(); self._refresh_worth()
        rep = dict(rules=self.rulebook.summary(), selection=self.selection_report(),
                   seconds=round(time.time() - t0, 2))
        if verbose:
            print(json.dumps(rep, indent=1))
        return rep

    def _review_matrix(self, touched: Optional[list] = None, rids: Optional[list] = None) -> Optional[BuildMatrix]:
        """Corrected builds of active reviews, weighted by their ledger weight.
        touched (if given) receives, per row, the catalog indices the review
        added or removed: worth counts the row review_weight times for those
        items only (an unmarked luxury pick is tolerated, not preferred)."""
        if not self.reviews or self.ledger is None:
            return None
        bs, ws, origs = [], [], []
        for r in self.ledger.reviews:
            w = self.ledger.weights[r.review_id].review
            if w <= 0:
                continue
            b = r.apply(self.catalog)
            if b.timestamp is None:
                b.timestamp = r.timestamp
            bs.append(b); ws.append(w); origs.append(r.original)
            if rids is not None:
                rids.append(r.review_id)
            if touched is not None:
                touched.append(sorted({self.catalog.index[o.item_id] for o in r.operations
                                       if o.action != Action.IFFY and o.item_id in self.catalog.index}))
        if not bs:
            return None
        tokens = self.data.tokens if self.data is not None else None
        bm = BuildMatrix.from_builds(bs, self.catalog, ws, self.cfg.context_fields, tokens=tokens)
        bm.originals = origs
        return bm

    def _refresh_worth(self) -> None:
        touched: list = []
        rids: list = []
        rev = self._review_matrix(touched, rids)
        tm = None
        conds = None
        self._worth = None                             # v3.7: the old reference is freed before the new one is built
        data = self.data
        sel = None
        n_ref = self.cfg.worth.reference_size
        if n_ref is not None and len(data) > n_ref:
            sel = np.sort(np.random.default_rng(self.cfg.seed).choice(len(data), n_ref, replace=False))
            data = data.subset(sel)
        from tri.model.worth import RowStack
        Zd = RowStack(self._Zdata, sel)                # rows of the corpus coordinates, not a copy
        if rev is not None:
            ref = data.append(rev)
            Uo = np.array([self._embed(b) for b in rev.originals])       # where unreviewed look-alikes sit
            Z = RowStack(self._Zdata, sel, Uo)
            is_rev = np.r_[np.zeros(len(data), bool), np.ones(len(rev), bool)]
            rows = np.repeat(np.arange(len(data), len(ref)), [len(t) for t in touched])
            cols = np.array([c for t in touched for c in t], dtype=np.int64)
            tm = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(ref), ref.N))
            lc = self.ledger.conditions()
            conds = {}
            for k, rid in enumerate(rids):
                d = lc.get(rid)
                if d:
                    ix = self.catalog.index
                    # present condition y -> index; NW absent condition "!y" -> -(index + 1)
                    # v3.3: a weapon condition cannot be read from picks: worth abstains
                    # on it (index N is never picked); the review memory applies it
                    nn = len(self.catalog)
                    conds[len(data) + k] = {ix[i]: ((nn,) if any(y.startswith("weapon:") for y in rel) else
                                                    tuple((-(ix[y[1:]] + 1) if y.startswith("!") else ix[y])
                                                          for y in rel if y.lstrip("!") in ix))
                                            for i, rel in d.items() if i in ix}
        else:
            ref, Z, is_rev = data, Zd, None
        self._worth = FieldWorth(ref, Z, self.cfg.worth, is_rev, touched=tm, conditions=conds,
                                 barrier=self._chasm if self._barrier_on() else None,
                                 chasm_corpus=self.cfg.locators.chasm_corpus)
        ct = self.patches.change_times()
        self._worth.set_change_times({self.catalog.index[i]: t for i, t in ct.items() if i in self.catalog.index})
        rt = self.patches.release_times()
        self._worth.set_release_times({self.catalog.index[i]: t for i, t in rt.items() if i in self.catalog.index})

    def _need(self):
        if self.space is None:
            raise RuntimeError("fit() or load() first")

    def calibrate(self, max_reviews: int = 300) -> dict:
        """Set InferConfig.swap_margin, then InferConfig.no_fit_rate, to the
        values that best reproduce the active reviews (F1 of add/remove operations). Each review is answered
        without review memory, so this measures what generalises, not recall
        of the review itself. Returns the score per grid value."""
        self._need()
        if self.ledger is None:
            return {}
        revs = [r for r in self.ledger.reviews if self.ledger.weights[r.review_id].review > 0][-max_reviews:]
        if not revs:
            return {}
        from tri.analysis.evaluate import _ops_f1
        scores = {}
        old = self.cfg.infer.swap_margin
        for m in self.cfg.swap_grid:
            self.cfg.infer.swap_margin = m
            eng = PassEngine(self.catalog, self.space, self.rules, self._worth, self.review_rules,
                             None, self.cfg.infer)
            scores[m] = _ops_f1(eng, revs)
        best = max(scores, key=lambda m: (scores[m], m))
        self.cfg.infer.swap_margin = best
        nf_scores = {}
        old_nf = self.cfg.infer.no_fit_rate
        for v in self.cfg.no_fit_grid:                 # then the "does not fit" threshold
            self.cfg.infer.no_fit_rate = v
            eng = PassEngine(self.catalog, self.space, self.rules, self._worth, self.review_rules,
                             None, self.cfg.infer)
            nf_scores[v] = _ops_f1(eng, revs)
        nf = max(nf_scores, key=lambda v: (nf_scores[v], -abs(v - old_nf)))
        self.cfg.infer.no_fit_rate = nf
        return dict(swap_margin=best, previous=old, f1={str(k): round(v, 4) for k, v in scores.items()},
                    no_fit_rate=nf, previous_no_fit_rate=old_nf,
                    f1_no_fit={str(k): round(v, 4) for k, v in nf_scores.items()}, reviews=len(revs))
    # ------------------------------------------------------------ patches

    def apply_patch(self, patch: Patch) -> dict:
        """Register a patch: discount pre-patch evidence on the items it touched.
        Renames/removals need migrate(); new builds after the patch need refit()."""
        self._need()
        self.patches.add(patch)
        touched = patch.touched()
        n = self.rulebook.mark_stale(touched, f"changed by patch {patch.name}; refit after new builds arrive")
        self._ledger_and_rules(); self._refresh_worth()
        return dict(patch=patch.name, items=touched, rules_stale=n,
                    stale_claims=len(self.ledger.stale) if self.ledger else 0)

    def migrate(self, new_catalog: Catalog, renames: Optional[Dict[str, str]] = None, refit: bool = True) -> dict:
        """Move to a new catalog (new game data). Corpus builds and reviews are
        kept and remapped (renames followed, removed items dropped), then refit."""
        ren = dict(renames or {})
        self.data = remap_matrix(self.data, new_catalog, ren, cache_dir=getattr(self, "_run_path", None)) if self.data is not None else None
        revs = [remap_review(r, ren, new_catalog) for r in self.reviews]
        dropped = sum(r is None for r in revs)
        self.catalog = new_catalog
        self.reviews = [r for r in revs if r is not None]
        for p in self.patches.patches:
            p.changed = [ren.get(i, i) for i in p.changed]
        rep = dict(items=len(new_catalog), reviews_kept=len(self.reviews), reviews_dropped=dropped)
        if refit and self.data is not None:
            rep["refit"] = self.refit()
        return rep

    def drift(self, since: float, min_drop: float = 0.15) -> List[dict]:
        """Pass rules whose confidence fell on builds after `since` (patch detection)."""
        self._need()
        return self.rulebook.drift(self.data, self.space.region_of(self._Zdata), since, min_drop)
    # ------------------------------------------------------------ catalog sync

    def extend_catalog(self, new_catalog: Catalog, time_: Optional[float] = None, name: Optional[str] = None,
                       note: str = "", patches: Optional[Sequence[Patch]] = None) -> dict:
        """Follow new game data without a refit.

        Items new in `new_catalog` are added unrated: the field gets a zero
        direction for them (ConceptSpace.extend), so coordinates, distances,
        regions, rules and every other item's worth are exactly unchanged; their
        own worth has no evidence until reviews (or, at the next fit, builds
        made after `time_`) give it some. Corpus builds older than `time_`
        never count against them. Items whose requirements, prerequisites or
        exclusions changed take the new definitions; their certified rules go
        stale (re-scored at the next refit). Items missing from `new_catalog`
        are kept but retired (tag "retired": never available, never proposed).
        The whole step is recorded as a Patch (added / changed / removed).
        Renamed items count as removed + added; migrate() at the next refit
        carries their evidence over. patches: record these instead of one new
        patch (tri.sync.restore re-applies saved syncs this way). Seconds."""
        from tri.game.patch import diff_catalogs
        self._need()
        t = float(time_ if time_ is not None else time.time())
        d = diff_catalogs(self.catalog, new_catalog)
        import copy as _copy
        items = {i: _copy.deepcopy(it) for i, it in new_catalog.items.items()}
        retired = []
        for i in self.catalog.ids:
            if i not in items:
                old = self.catalog.items[i]
                tags = list(old.tags) + ([] if "retired" in old.tags else ["retired"])
                items[i] = Item(old.id, old.kind, old.category, dict(old.requires), list(old.requires_any),
                                [list(c) for c in old.clauses], list(old.prerequisites),
                                [list(g) for g in old.prerequisites_any], list(old.exclusive_with), tags,
                                context=[list(c) for c in old.context])
                retired.append(i)
        for i, it in items.items():                    # exclusions both ways
            for x in it.exclusive_with:
                if x in items and i not in items[x].exclusive_with:
                    items[x].exclusive_with = sorted(set(items[x].exclusive_with) | {i})
        ext = Catalog(items, stats=self.catalog.stats, ruleset=self.catalog.ruleset,
                      talent_cap_base=self.catalog.talent_cap_base, cap_exempt=self.catalog.cap_exempt,
                      cap_per_mantra=self.catalog.cap_per_mantra, mantra_locks=new_catalog.mantra_locks,
                      lock_fields=new_catalog.lock_fields)
        added = [i for i in ext.ids if i not in self.catalog.index]
        import copy as _copy

        from tri.game.patch import _fingerprint

        def core(it):                                   # an exclusion with a new item is not a change of this one
            c = _copy.copy(it); c.exclusive_with = [x for x in it.exclusive_with if x not in set(added)]
            return _fingerprint(c)
        changed = sorted(i for i in self.catalog.ids if i in ext.index and i not in retired
                         and core(self.catalog.items[i]) != core(ext.items[i]))
        prior = self._kind_prior()
        self._worth = None                             # v3.7: freed before the corpus is remapped
        r_old = self.space.solve_latent(self.data)     # memoised; unchanged by the extension (zero rows)
        self.space.extend(ext, {i: prior.get(ext.items[i].kind, 0.05) for i in added})
        self.data = remap_matrix(self.data, ext, {}, cache_dir=getattr(self, "_run_path", None))
        self.space.remember(self.data, r_old)
        revs = [remap_review(r, {}, ext) for r in self.reviews]
        self.reviews = [r for r in revs if r is not None]
        self.catalog = ext
        self.rulebook.c = ext
        p = Patch(name or time.strftime("sync-%Y%m%d-%H%M%S", time.gmtime(t)), t, changed=changed, added=added,
                  removed=retired, note=note or "game data sync")
        rep = dict(patch=p.name, added=added, changed=changed, retired=retired,
                   renames_suggested=d.get("suggested_renames", {}))
        if patches is not None:
            for x in patches:
                self.patches.add(x)
            ch = sorted({i for x in patches for i in x.changed} & set(ext.ids))
            rep["rules_stale"] = self.rulebook.mark_stale(ch, "changed by a game data sync; refit after new builds arrive")
        elif added or changed or retired:
            self.patches.add(p)
            rep["rules_stale"] = self.rulebook.mark_stale(changed, f"changed by {p.name}; refit after new builds arrive")
        self._locset(); self._name_regions()
        # v3.7: the corpus coordinates are kept: new items have zero rows, so
        # coordinates are unchanged exactly (ConceptSpace.extend) and the rows of
        # the corpus keep their order (remap_matrix); recomputing them doubled
        # the startup peak
        self._ledger_and_rules(); self._refresh_worth()
        return rep

    def _coord_dtype(self):
        """Storage type of the corpus coordinates _Zdata (v3.7): TRI_COORDS=float32
        halves the largest array the engine holds (B x D' x 8 bytes, 830 MB at
        300k builds); every distance is still computed in float64."""
        import os
        return np.float32 if os.environ.get("TRI_COORDS", COORDS_DEFAULT).strip().lower() in ("float32", "f4", "32") \
            else np.float64

    def _kind_prior(self) -> Dict[Kind, float]:
        """Median corpus pick rate per item kind: the constant worth of an unrated item."""
        pr = self.data.pick_rate()[0]
        out = {}
        rel = self.patches.release_times()             # once, not once per item
        for k in (Kind.TALENT, Kind.MANTRA):
            v = [pr[j] for j, i in enumerate(self.catalog.ids) if self.catalog.items[i].kind == k
                 and i not in rel]
            out[k] = float(np.median(v)) if v else 0.05
        return out
