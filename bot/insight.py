"""Categorization, facts and stream views for the bot
(math §12).

One entry point, ModelService.insight(kind, **kw) -> dict(files, text, data):

  atlas                 B1 archetype atlas (two planes)
  tree                  B2 archetype dendrogram
  card      archetype   B3 archetype card (+ its top facts)
  item      item        B5 worth landscape of an item
  stats     item, stat  B6 worth against a stat, per archetype
  meta                  B7 archetype shares over time
  facts     archetype   C  certified facts (and the fact book: new / retired)
  suggest               A4 candidate archetypes among unplaced builds
  quality               A5 categorization quality
  replay                D  prequential replay, variogram, coverage, low-rank test
  place     build       B4 placement inset of one build
  categorize build      A1/A2 labels of one build (no image)
  digest    since       C8 facts certified / retired since a time
  worth     build       worth bars of a build's picks (central / indicative marks, flat values)
  variants              archetype families and variants among the locators
  reviews               review reasons and note kinds
  metrics               leave-one-out locator accuracy per archetype

The atlas (a corpus sample read as archetypes), its projection, the discovered
candidates and the mined facts are cached per field, review count and locator
set. Nothing is refitted.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

KINDS = ("atlas", "card", "item", "stats", "facts", "rules", "suggest", "quality", "replay", "place",
         "categorize", "digest", "lookup")
#: v2.21 deprecated the tree, meta, worth, variants, reviews and metrics charts.
DEPRECATED = ("tree", "meta", "worth", "variants", "reviews", "metrics")
ATLAS_SAMPLE = int(os.environ.get("TRI_ATLAS_SAMPLE", "3000"))
MIN_PLACE_TALENTS = int(os.environ.get("TRI_MIN_PLACE_TALENTS", "15"))   # fewer talents: no placement graphic


def _key(svc) -> tuple:
    t = svc.tri
    rid = t.reviews[-1].review_id if t.reviews else None
    locs = tuple(sorted((l.locator_id, l.archetype) for l in (t.locators or [])))
    return (id(t.space), len(t.reviews), rid, locs, len(t.patches.patches) if t.patches is not None else 0)


class _Cache:
    def __init__(self):
        self.key = None
        self.atlas = self.view = self.cands = self.facts = self.tree = self.miner = None


def _cache(svc) -> _Cache:
    c = getattr(svc, "_insight_cache", None)
    if c is None:
        c = svc._insight_cache = _Cache()
    k = _key(svc)
    if c.key != k:
        c.__init__()
        c.key = k
    return c


def _atlas(svc):
    from tri.analysis import categorize as C
    from tri.views import insight as V
    c = _cache(svc)
    if c.atlas is None:
        c.atlas = C.build_atlas(svc.tri, sample=ATLAS_SAMPLE)
    if c.cands is None:
        c.cands = C.discover(svc.tri, c.atlas)
    if c.view is None:
        c.view = V.atlas_view(svc.tri, c.atlas, candidates=c.cands)
    return c


def _facts(svc):
    from tri.analysis import facts as F
    c = _atlas(svc)
    if c.facts is None:
        c.facts = F.mine_facts(svc.tri, c.atlas)
        book = fact_book(svc)
        c.book_update = book.update(c.facts)
    return c


def fact_book(svc):
    from tri.analysis.facts import FactBook
    path = os.environ.get("TRI_FACTS_PATH") or os.path.join(os.path.dirname(os.path.abspath(svc.run)), "facts.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        path = None
    return FactBook(path)


def _arch_index(at, name: Optional[str]) -> int:
    i = at.index(name or "")
    if i is None:
        raise ValueError(f"unknown archetype {name!r}; one of: " + ", ".join(at.names[:8])
                         + (f" (+{len(at.names) - 8} more)" if len(at.names) > 8 else ""))
    return i


def _item(svc, name: str) -> str:
    it = svc.resolve(name) if name else None
    if it is None or it not in svc.tri.catalog.index:
        raise ValueError(f"unknown item {name!r}")
    return it


def _draw(theme: str, fn, *a, **kw) -> bytes:
    """Draw under the theme (the caller holds the plot lock)."""
    from tri.views import plots as render

    from .service import _png
    with render.theme(theme):
        return _png(fn(*a, **kw))


def _diverse(facts):
    """Ranked facts, each family's best first (round robin), so a short list
    is not all one kind of fact."""
    by = {}
    for f in facts:
        by.setdefault(f.family, []).append(f)
    out, k = [], 0
    while any(k < len(v) for v in by.values()):
        out.extend(sorted((v[k] for v in by.values() if k < len(v)), key=lambda f: -f.score))
        k += 1
    return out


def _tells(tri, name: str, k: int = 2) -> List[dict]:
    """The picks that tell a locator archetype from its twins: the archetypes the map
    confuses it with whose split passed the held-out check (math §6.6).
    [{name, items: [{item, sign}], symbol}]."""
    ls = tri.locset
    if ls is None or not ls.refining:
        return []
    from tri.reviews.locators import archetype_key
    key = archetype_key(name)
    if key not in ls.kix:
        return []
    a = ls.kix[key]
    out = []
    for q in sorted(ls.pair_of(key), key=lambda q: -q["overlap"])[:k]:
        other = ls.keys[q["b"] if q["a"] == a else q["a"]]
        its = [dict(item=tri.catalog.ids[i], sign=1 if lo > 0 else -1) for i, lo in ls.distinct(key, other, 5)]
        if its:
            sym = f"{ls.names[key]} ↔ {ls.names[other]}: " + " ".join(
                ("+" if t["sign"] > 0 else "−") + t["item"] for t in its)
            out.append(dict(name=ls.names[other], overlap=q["overlap"], items=its, symbol=sym))
    return out


def _fact_rows(facts, archetype: Optional[str] = None, status: str = "certified", limit: int = 10):
    out = []
    for f in _diverse(facts):
        if status and f.status != status:
            continue
        if archetype is not None and f.archetype != archetype and archetype not in f.sentence:
            continue
        out.append(dict(key=f.key, family=f.family, archetype=f.archetype, symbol=f.symbol, sentence=f.sentence,
                        p=None if f.p is None else float(f.p), score=round(f.score, 3), graphic=f.graphic,
                        items=list(f.items), status=f.status))
        if len(out) >= limit:
            break
    return out


def insight(svc, kind: str, theme: str = "dark", archetype: Optional[str] = None, item: Optional[str] = None,
            stat: Optional[str] = None, build=None, limit: int = 10, since: Optional[float] = None,
            status: str = "certified", interactive: bool = False) -> dict:
    """See the module docstring. Returns dict(files=[(name, png)], text, data)."""
    from tri.analysis import categorize as C, stream as S
    from tri.infer import active
    from tri.views import insight as V
    if kind in DEPRECATED:
        raise ValueError(f"the {kind} chart is deprecated (v2.21)")
    if kind not in KINDS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}")
    tri = svc.tri
    files, data, text = [], {}, ""
    if kind == "place" and build is not None and len(build.talents) < MIN_PLACE_TALENTS:
        return dict(files=[], text=f"placement not shown: {len(build.talents)} talents, fewer than "
                                   f"{MIN_PLACE_TALENTS}", data=dict(hidden=True, talents=len(build.talents)))
    if kind == "categorize":
        with svc.lock:
            cat = C.categorize(tri, build)
        cat.pop("u", None)
        return dict(files=[], text=cat["text"], data=cat)
    if kind == "digest":
        with svc.lock:
            c = _facts(svc)
        book = fact_book(svc)
        d = book.digest(float(since if since is not None else time.time() - 7 * 86400))
        return dict(files=[], text=f"{len(d['new'])} new · {len(d['retired'])} retired", data=d)
    from .service import _PLOT_LOCK
    with _PLOT_LOCK, svc.lock:                  # same order as the /plot views
        c = _atlas(svc)
        at, av = c.atlas, c.view
        if kind == "atlas":
            if interactive:                     # v2.21: the interactive atlas page
                from tri.views import plots as render
                with render.theme(theme):
                    files.append(("atlas.html", render.html_scatter(V.atlas_scatter(at, av, fields=True)).encode()))
            else:                               # v2.25: 3-D, always fields
                files.append(("atlas.png", _draw(theme, V.plot_atlas3d, tri, av, at)))
            data = dict(names=at.names, shares=[round(float(s), 4) for s in at.share],
                        located=round(float(at.w[at.placed].sum() / max(at.w.sum(), 1e-12)), 4),
                        readings={k: int((at.kind == k).sum()) for k in C.READINGS}, candidates=len(c.cands),
                        source=at.source)
            text = f"{len(at.names)} archetypes · {100 * data['located']:.0f}% located"
        elif kind == "card":
            a = _arch_index(at, archetype)
            cf = _facts(svc)
            nm = at.names[a]
            traps = [dict(item=f.items[0], **f.numbers) for f in cf.facts if f.family in ("hole", "trap") and f.archetype == nm]
            card = V.archetype_card(tri, at, a, traps)
            files.append(("archetype.png", _draw(theme, V.plot_card, card)))
            data = dict(name=nm, builds=card["builds"], share=card["share"], readings=card["readings"],
                        facts=_fact_rows(cf.facts, nm, limit=3),
                        neighbours=[dict(name=n["name"], distance=round(n["distance"], 3),
                                         items=[s["item"] for s in n["items"]]) for n in card["neighbours"]],
                        tells=_tells(tri, nm), consensus=card.get("consensus", []))
            text = nm
        elif kind == "item":
            it = _item(svc, item)
            il = V.item_landscape(tri, at, av, it)
            files.append(("item.png", _draw(theme, V.plot_item, il, av, at)))
            data = dict(item=it, kept=sum(m["sign"] > 0 for m in il["marks"]),
                        removed=sum(m["sign"] < 0 for m in il["marks"]), in_plane=round(il["in_plane"], 3))
            text = it
        elif kind == "stats":
            it = _item(svc, item)
            if stat:
                st = {s.lower(): s for s in at.bm.catalog.stats}.get(stat.lower().replace(" ", ""), stat)
            else:
                st = None
            sc = V.stat_curves(tri, at, it, st)
            files.append(("stats.png", _draw(theme, V.plot_stats, sc, None, len(at.names))))
            data = dict(item=it, stat=sc["stat"], requirement=sc["requirement"],
                        crosses={cv["archetype"]: cv["crosses"] for cv in sc["curves"]})
            text = f"{it} · {sc['stat']}"
        elif kind == "facts":
            cf = _facts(svc)
            nm = at.names[_arch_index(at, archetype)] if archetype else None
            rows = _fact_rows(cf.facts, nm, status=status, limit=limit)
            n_c = sum(f.status == "certified" for f in cf.facts)
            data = dict(facts=rows, certified=n_c, candidates=len(cf.facts) - n_c,
                        new=len(cf.book_update.get("new", [])), retired=len(cf.book_update.get("retired", [])))
            text = f"{n_c} certified · {len(cf.facts) - n_c} candidates"
        elif kind == "rules":                   # v2.21: how the model answers
            from tri.rules import statements as ST
            it = _item(svc, item) if item else None
            data = ST.model_rules(tri, it, limit=limit if limit else 25)
            text = f"{data['total']} rules" + (f" · {data['trivial']} trivial dropped" if data["trivial"] else "")
        elif kind == "lookup":                  # v2.21: /talent, /mantra
            from tri.analysis import facts as F
            from tri.rules import statements as ST
            it = _item(svc, item)
            cf = _facts(svc)
            if getattr(c, "miner", None) is None:
                c.miner = F.FactMiner(tri, at)
            data = ST.lookup(tri, it, at, c.miner, cf.facts)
            files.append(("lookup.png", _draw(theme, V.plot_lookup, data, len(at.names), at.names)))
            if data.get("requires") or any(r["kind"] == "benchmark" for r in data["rules"]):
                sc = V.stat_curves(tri, at, it, None)
                if sc["curves"]:
                    files.append(("stats.png", _draw(theme, V.plot_stats, sc, None, len(at.names))))
            text = it
        elif kind == "suggest":
            rows = []
            for cd in c.cands:
                rows.append(dict(id=cd["id"], builds=cd["builds"], share=round(cd["share"], 4), medoid=cd["medoid"],
                                 signature=[s["item"] for s in cd["signature"]],
                                 nearest=[dict(name=n["name"], distance=round(n["distance"], 3)) for n in cd["nearest"]],
                                 growth=cd["growth"]))
            if rows:
                files.append(("suggest.png", _draw(theme, V.plot_atlas, av, at, ((0, 1),), 1500,
                                                   f"Candidate archetypes · {len(rows)} among unplaced builds")))
            data = dict(candidates=rows, unplaced=int((~at.placed).sum()))
            text = f"{len(rows)} candidates"
        elif kind == "quality":
            data = C.quality(tri, at)
            text = f"located {100 * data['located_share']:.0f}%"
        elif kind == "replay":
            pq, vg, lr = S.prequential(tri), S.variogram(tri), S.lowrank(tri, 100)
            led = tri.ledger
            Ur = np.array([led.vector(r) for r in led.by_id]) if led is not None and led.by_id else np.zeros((0, 1))
            cov = active.coverage(at.U, Ur)
            q = active.kcenter_queue(at.U, Ur, 5)
            files.append(("replay.png", _draw(theme, V.plot_replay, pq, vg, cov, lr)))
            data = dict(final=pq.get("final"), reviews=pq.get("reviews", 0), memory_share=pq.get("memory_share"),
                        lowrank=dict(p=lr.get("p"), direction=lr.get("direction")),
                        coverage=dict(radius=cov["radius"], median=cov["median"], p90=cov["p90"]),
                        queue=[dict(build=at.ids[x["row"]], distance=round(x["distance"], 3)) for x in q])
            text = "replay"
        elif kind == "place":
            pl = V.placement(tri, at, av, build)
            files.append(("placement.png", _draw(theme, V.plot_placement, pl, av, at, None, "map")))
            files.append(("reading.png", _draw(theme, V.plot_placement, pl, av, at, None, "pi")))
            cat = dict(pl["category"]); cat.pop("u", None)
            data = cat
            text = cat["text"]
    return dict(files=files, text=text, data=data)
