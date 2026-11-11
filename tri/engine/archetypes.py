"""Tri methods: archetypes: locators, regions, consensus, metrics and concept readings."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from tri.core.types import Action, Build, Reason, Strength, off_archetype
from tri.reviews.consensus import tally
from tri.reviews.defs import ConsensusConfig, Locator
from tri.reviews.locators import LocatorSet, archetype_key
from tri.reviews.variants import family_key, join_archetype, read_variant, split_archetype, style_of


class _Archetypes:
    """Tri methods (Tri methods: archetypes: locators, regions, consensus, metrics and concept readings.)."""

    def _sync_indicators(self) -> None:
        """Indicator notes ("# a: x, y, z") of the reviews -> the locator set (math §6.7).
        The newest note per component set wins; components are catalog items or
        oath:/murmur: tokens."""
        ls = getattr(self, "locset", None)
        self._note_worth = []
        if self.space is None:
            return
        out: Dict[frozenset, dict] = {}
        anchors: Dict[tuple, dict] = {}
        worth: Dict[tuple, dict] = {}
        for r in sorted(self.reviews, key=lambda r: r.timestamp or 0.0):
            notes = list(getattr(r, "notes", None) or [])
            off = [self.catalog.index[o.item_id] for o in r.operations
                   if off_archetype(o) and o.item_id in self.catalog.index]
            if not notes and not off:
                continue
            u = self._embed(r.apply(self.catalog))         # where the corrected build sits
            if off:                                        # strong SM = an avoid note for this archetype
                worth[("avoid", frozenset(off), r.review_id)] = dict(
                    kind="avoid", key=None, items=off, u=u, review_id=r.review_id,
                    names=[self.catalog.ids[i] for i in off], build=r.original)
            for n in notes:
                kind = n.get("kind")
                if kind == "archetype":
                    comps = [str(c) for c in n.get("items") or []]
                    idx = [self.catalog.index[c] for c in comps if c in self.catalog.index]
                    toks = [c for c in comps if ":" in c and c not in self.catalog.index]
                    key = family_key(str(n.get("archetype", "")))
                    anchors[(r.review_id, key)] = dict(key=key, u=u, sign=1, review_id=r.review_id)
                    if not comps or len(idx) + len(toks) < len(comps):
                        continue
                    out[frozenset(comps)] = dict(key=key, items=idx, tokens=toks, names=comps, u=u,
                                                 review_id=r.review_id)
                    if idx:
                        worth[("indicator", frozenset(idx))] = dict(kind="indicator", key=key, items=idx,
                                                                   names=[c for c in comps if c in self.catalog.index],
                                                                   u=u, review_id=r.review_id,
                                                                   archetype=str(n.get("archetype", "")))
                elif kind in ("is", "not"):
                    key = family_key(str(n.get("archetype", "")))
                    anchors[(r.review_id, key)] = dict(key=key, u=u, sign=1 if kind == "is" else -1,
                                                       review_id=r.review_id, variant=str(n.get("archetype", "")))
                elif kind in ("core", "avoid"):
                    idx = [self.catalog.index[c] for c in n.get("items") or [] if c in self.catalog.index]
                    if idx:
                        worth[(kind, frozenset(idx))] = dict(kind=kind, key=None, items=idx, u=u, review_id=r.review_id,
                                                             names=[self.catalog.ids[i] for i in idx], build=r.original)
        if ls is not None:
            ls.set_anchors(list(anchors.values()))
            ls.set_indicators(list(out.values()))
            for w in worth.values():                     # core / avoid: under the reviewed build's own reading
                if w["key"] is None:
                    rd = ls.read(w["u"], w.pop("build", None))
                    w["key"] = archetype_key(rd.best) if rd.best else None
                    w["archetype"] = rd.best
                w.pop("build", None)
        else:
            for w in worth.values():
                w.pop("build", None)
        self._note_worth = list(worth.values())
        self._sync_consensus()
        self._renote()                               # placements re-note their fine reading

    def _sync_consensus(self) -> None:
        """Reviewer consensus (math §8.5a): notes and archetype-reasoned edits of
        every review, per (family, item), tallied into promoted tells."""
        ls = getattr(self, "locset", None)
        self.consensus: Dict[str, list] = {}
        lc = self.cfg.locators
        if ls is None or self.space is None or not lc.consensus or not self.reviews:
            if ls is not None:
                ls.set_consensus([])
            return
        ix = self.catalog.index
        ev: List[dict] = []
        self._place_many([r.apply(self.catalog) for r in self.reviews])     # v3.1: one batched solve
        for r in self.reviews:
            fam = None

            def family():
                nonlocal fam
                if fam is None:
                    b = r.apply(self.catalog)
                    rd = ls.read(self._embed(b), b)
                    fam = family_key(rd.archetype) if rd.archetype else ""
                return fam or None
            base = dict(reviewer=r.reviewer_id or r.review_id, timestamp=r.timestamp or 0.0, review_id=r.review_id,
                        build=r.original.build_id or r.review_id)
            for n in getattr(r, "notes", None) or []:
                kind = n.get("kind")
                if kind == "archetype":
                    key = family_key(str(n.get("archetype", "")))
                    for c in n.get("items") or []:
                        if c in ix:
                            ev.append(dict(base, key=key, item=c, sign=1, weight=1.0, note=True))
                elif kind in ("core", "avoid"):
                    key = family()
                    for c in n.get("items") or []:
                        if c in ix and key:
                            ev.append(dict(base, key=key, item=c, sign=1 if kind == "core" else -1, weight=1.0, note=True))
            for o in r.operations:
                if o.item_id not in ix or o.reason is None:
                    continue
                if off_archetype(o):                       # strong SM: counts like an avoid note
                    key = family()
                    if key:
                        ev.append(dict(base, key=key, item=o.item_id, sign=-1, weight=1.0, note=True))
                    continue
                st = 1.0 if o.strength == Strength.STRONG else 0.5
                if o.action == Action.ADD and o.reason == Reason.NECESSARY_FOR_ARCHETYPE:
                    sign = 1
                elif o.action == Action.REMOVE and o.reason in self._CONSENSUS_REMOVE:
                    sign = -1
                else:
                    continue
                key = family()
                if key:
                    ev.append(dict(base, key=key, item=o.item_id, sign=sign, weight=lc.consensus_op_weight * st, note=False))
        cc = ConsensusConfig(kappa=lc.consensus_kappa, min_reviewers=lc.consensus_min_reviewers,
                             unit=getattr(lc, "consensus_unit", "build"))
        tells = tally(ev, cc)
        self.consensus = {k: [t.as_dict() for t in v] for k, v in tells.items()}
        ls.set_consensus([dict(t, t=ix[t["item"]]) for v in self.consensus.values() for t in v])

    def consensus_report(self, archetype: Optional[str] = None) -> Dict[str, list]:
        """Promoted reviewer tells per family (math §8.5a), optionally for one archetype."""
        c = getattr(self, "consensus", None) or {}
        if archetype is None:
            return {(self.locset.names.get(k, k) if self.locset is not None else k): v for k, v in c.items()}
        k = family_key(archetype)
        return {archetype: list(c.get(k, []))}
    # ------------------------------------------------------------ locators

    def _locset(self) -> None:
        bg = self.data.pick_rate()[0] if self.data is not None and len(self.data) else None
        self.locset = (LocatorSet(self.locators, self.space, self.cfg.locators, self.cfg.ledger.reach,
                                  vec=self._vec, background=bg)
                       if self.space is not None and self.locators else None)
        self._renote()                               # placements re-note their fine reading
        if self.locset is not None and getattr(self, "reviews", None):
            self._sync_indicators()

    def archetypes(self) -> List[str]:
        """Names a caller may force: locator archetypes, else the region names."""
        if self.locset is not None:
            fams = [self.locset.names[k] for k in self.locset.keys]
            var = sorted({join_archetype(self.locset.names[l.key], l.variant) for l in self.locset.locators
                          if l.variant}, key=str.casefold)
            return fams + var
        return list(self.space.region_names) if self.space is not None else []

    def force(self, name: Optional[str]) -> Optional[dict]:
        """A forced archetype for review / complete: a locator archetype (its locators'
        mean latent, and its profile for the overlapping picks) or a region by name.
        Raises ValueError for an unknown name."""
        if name is None or not str(name).strip():
            return None
        self._need()
        key = family_key(name)
        ls = self.locset
        if ls is not None and key in ls.kix:
            d = self.space.cfg.dim
            fam, var = split_archetype(name)
            m = ls.lab == ls.kix[key]
            if var:                                      # a variant: its own locators' mean, if it has any
                mv = m & np.array([(l.variant or "").casefold() == var.casefold() for l in ls.locators])
                if mv.any():
                    m = mv
                else:
                    var = None
            z = ls.Y[m, :d].mean(0)
            return dict(name=join_archetype(ls.names[key], var), key=key, latent=z, variant=var)
        names = list(self.space.region_names)
        full = archetype_key(name)                       # region names may hold " / " (no variants there)
        for k, n in enumerate(names):
            short = n.split(" (")[0]
            if archetype_key(n) in (key, full) or archetype_key(short) in (key, full):
                return dict(name=n, region=k)
        known = ", ".join(self.archetypes()[:12])
        raise ValueError(f"unknown archetype {name!r}; known: {known}")

    def _chasm(self, u, V):
        """The chasm of the current locator set (math §5.4); 1 without one."""
        ls = self.locset
        if ls is None or len(ls.keys) < 2:
            return np.ones(len(np.atleast_2d(V)))
        return ls.barrier_factor(u, V)

    def _barrier_on(self) -> bool:
        return self.locset is not None and self.cfg.locators.barrier > 0 and len(self.locset.keys) >= 2

    def _name_regions(self) -> None:
        """Region display names: the located archetype of the centre, then the item-based name."""
        sp = self.space
        auto = sp.region_auto or list(sp.region_names)
        labels = self.locset.region_labels(sp.centroids) if self.locset is not None else [None] * len(auto)
        mixed = list(getattr(sp, "region_mixed", []) or [])
        labels = [None if (k < len(mixed) and mixed[k]) else lab for k, lab in enumerate(labels)]  # catch-all: no archetype name
        sp.region_names = [f"{lab} ({a})" if lab else a for lab, a in zip(labels, auto)]
        if self.rulebook is not None:
            self.rulebook.region_names = list(sp.region_names)

    def set_locators(self, locators: Sequence[Locator]) -> dict:
        """Replace the locators (e.g. every row of the Supabase table). Instant: archetype
        readings, region names and the review barrier change; the field does not."""
        seen: Dict[str, Locator] = {}
        for l in locators:                            # same id again = the corrected version
            seen[l.locator_id] = l
        self.locators = list(seen.values())
        if self.space is not None:
            self._locset(); self._name_regions(); self._ledger_and_rules(); self._refresh_worth()
        return self.locator_report()

    def add_locators(self, locators: Sequence[Locator]) -> dict:
        new = {l.locator_id for l in locators}
        return self.set_locators([l for l in self.locators if l.locator_id not in new] + list(locators))

    def remove_locator(self, locator_id: str) -> bool:
        n = len(self.locators)
        self.set_locators([l for l in self.locators if l.locator_id != str(locator_id)])
        return len(self.locators) < n

    def archetype(self, build: Build) -> dict:
        """The archetype the build reads as: top archetype, its probability, the nearest
        locator, and whether it is located (within reach and confident)."""
        self._need()
        if self.locset is None:
            return dict(archetype=None, located=False, locators=0)
        u = self.space.point(build, loo=False).u
        out = self.locset.read(u, b=build).to_dict()
        ind = self.locset.indicated(u, build)
        if ind:
            out["indicated"] = ind                     # reviewers' indicator notes that lift this reading
        self._add_variant(out, u, build)
        return out

    def _add_variant(self, out: dict, u, build: Build) -> dict:
        """The finer reading (tri.variants): the variant of the family the build
        reads as, its style from the stats, and the full name."""
        fam = out.get("archetype") or out.get("best")
        v = read_variant(self.locset, u, fam) if fam else None
        if v is not None:
            out["variant"] = v
        out["style"] = style_of(build, list((build.meta or {}).get("weapons") or []) if build.meta else [])
        if out.get("archetype"):
            out["full_name"] = join_archetype(out["archetype"], v["variant"] if v and v.get("located") else None)
        return out

    def locator_report(self) -> dict:
        """Leave-one-out validation of the locators, the metric selection, and how
        many locator builds trained the field (0 after a refit: they are held out)."""
        if self.locset is None:
            return dict(locators=len(self.locators))
        rep = self.locset.loo()
        ids = set(self.data.ids) if self.data is not None else set()
        held = set(self.held_out_ids)
        rep["in_training"] = sum(1 for l in self.locators if l.build.build_id in ids and l.build.build_id not in held)
        rep["metric_power"] = self.space.cfg.metric_power
        rep["metric_selection"] = self.metric_selection
        rep["regions_named"] = sum(1 for a, n in zip(self.space.region_auto, self.space.region_names) if a != n)
        return rep

    def select_metric(self, apply: bool = False) -> dict:
        """Leave-one-out choice of metric_power on the current field (no refit). apply:
        switch to it (coordinates, regions, review similarity and worth recomputed)."""
        self._need()
        if self.locset is None or not self.locset.eligible():
            c = self.cfg.locators
            return dict(eligible=False, needs=f"{c.select_min_per_archetype} locators in each of "
                                              f"{c.select_min_archetypes} archetypes")
        sel = self.locset.select_power(self.data, self.space.cfg.metric_power)
        sel["eligible"] = True
        if apply and sel["moved"]:
            self.metric_selection = sel
            self.set_metric("worth_variance", power=sel["chosen"])
        return sel

    def set_metric(self, metric: str = "worth_variance_dedup", power: Optional[float] = None) -> dict:
        """Switch the concept-space distance's item weights on a fitted or loaded run
        without refitting the field: coordinates, regions, review similarity and
        worth neighbourhoods are recomputed (seconds). power: metric_power to use.
        Runs saved before v2.3 use "uniform"; call set_metric() then save() to upgrade one in place."""
        if self.space is None or self.data is None:
            raise RuntimeError("fit() or load() first")
        if power is not None:
            self.space.cfg.metric_power = float(power)
        self.space.set_metric(self.data, metric)
        self.cfg.space.metric = metric
        self.cfg.space.metric_power = self.space.cfg.metric_power
        if self.locset is not None:
            self.locset.refresh()
        self._name_regions()
        self._Zdata = None
        self._Zdata = self.space.coords(self.data, dtype=self._coord_dtype())
        self._mine_sidegrades()                        # regions moved
        self._ledger_and_rules(); self._refresh_worth()
        return self.space.report()

    def concept(self, build: Build, k_reviews: int = 3) -> dict:
        """Coordinates, archetype regions, ambiguity and nearest reviewed builds."""
        self._need()
        pt = self.space.point(build)
        d = self.space.reading(pt.u).to_dict()
        if self.locset is not None:
            u = self.space.point(build, loo=False).u
            d["archetype"] = self._add_variant(self.locset.read(u, b=build).to_dict(), u, build)
            ind = self.locset.indicated(u, build)
            if ind:
                d["archetype"]["indicated"] = ind
        d["style"] = style_of(build, list((build.meta or {}).get("weapons") or []) if build.meta else [])
        if self.ledger is not None:
            d["nearest_reviews"] = [dict(review_id=r.review_id, similarity=round(s, 3))
                                    for r, s in sorted(self.ledger.neighbours(build, 0.0), key=lambda x: -x[1])[:k_reviews]]
        return d
