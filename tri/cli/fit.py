"""nonsupervised refit

    python -m tri.cli.fit                                   # reads ./builds.json, writes ./runs/<timestamp>/
    python -m tri.cli.fit --builds data/builds.json --name s12
    python -m tri.cli.fit --limit 50000                     # quick trial on the first 50k records
    python -m tri.cli.fit --config tri.json                 # TriConfig overrides
    python -m tri.cli.fit --patches patches.json            # game patches: discount pre-patch evidence
    python -m tri.cli.fit --patches runs/<old>/sync/patches.json   # keep items a game data sync added
    python -m tri.cli.fit --device cuda                     # GPU training: auto (default) | cuda | cuda:N | mps | cpu | numpy
    python -m tri.cli.fit --jobs 8                          # rule mining processes (0 auto, -1 all cores)

builds.json comes from poioi

Output (runs/<name>/; runs/LATEST holds its absolute path):
  model/            Tri.save: tri.json, space/, data.npz, rulebook.json, ...
  catalog.json      catalog the run uses (needed to load it)
  report.json       pipeline report (drops, unknown names, game data, ...)
  excluded_talents.json
  item_counts.json  {name: {count, in_gamedata}} for every name seen
  granted.json      {talent: group or null} used when parsing
  config.json       arguments, PipelineConfig, TriConfig, source file info
  metrics.json      held-out benchmarks (tri vs worth only vs popularity), timings
  regions.txt       archetype regions and the space's main axes

Load the result:
    from tri import Tri, Catalog
    cat = Catalog.from_json("runs/<name>/catalog.json")
    tri = Tri.load("runs/<name>/model", cat)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np

from tri import Catalog, PipelineConfig, PretrainSet, Tri, TriConfig  # noqa: E402
from tri.analysis.evaluate import (certification, completion_benchmark, mapping_benchmark,  # noqa: E402
                                   rule_benchmark)
from tri.sources.pipeline import (catalog_to_json, exclude_items, granted_group,  # noqa: E402
                                  prepare_pretraining)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path):
    cfg = PipelineConfig()
    if path:
        with open(path) as f:
            for k, v in json.load(f).items():
                if not hasattr(cfg, k):
                    raise SystemExit(f"unknown PipelineConfig field in {path}: {k}")
                cur = getattr(cfg, k)
                if k == "patches":
                    v = tuple(map(tuple, v))
                elif isinstance(cur, tuple):
                    v = tuple(v)
                elif isinstance(cur, dict) and k.startswith("granted_"):
                    v = {**cur, **v}                # extend the defaults; map a name to null to remove it
                    v = {n: g for n, g in v.items() if g}
                setattr(cfg, k, v)
    return cfg


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: getattr(o, "value", str(o))))


def cache_key(src, a, cfg, gd=None) -> str:
    st = os.stat(src)
    h = hashlib.sha256()
    h.update(json.dumps(dict(src=os.path.abspath(src), size=st.st_size, mtime=st.st_mtime_ns,
                             limit=a.limit, ruleset=a.ruleset, cfg=jsonable(asdict(cfg))),
                        sort_keys=True).encode())
    if a.catalog:
        with open(a.catalog, "rb") as f:
            h.update(f.read())
    if gd is not None:
        h.update(json.dumps([gd.fingerprint(), sorted(a.keep_group)]).encode())
    return h.hexdigest()[:16]


def get_gamedata(a):
    from tri.game.gamedata import resolve_gamedata
    dest = os.path.join(a.out, "gamedata")
    try:
        return resolve_gamedata(a.gamedata or ["auto"], dest, a.refresh_gamedata, a.gamedata_ref, log=log)
    except Exception as e:
        raise SystemExit(f"game data: {e}; pass --gamedata PATH to a local copy or --gamedata none")


def load_tri_cfg(path) -> TriConfig:
    cfg = TriConfig()
    if not path:
        return cfg
    with open(path) as f:
        d = json.load(f)
    base = cfg.to_dict()
    for k, v in d.items():
        if k not in base:
            raise SystemExit(f"unknown TriConfig field in {path}: {k}")
        if isinstance(base[k], dict):
            unknown = set(v) - set(base[k])
            if unknown:
                raise SystemExit(f"unknown {k} fields in {path}: {sorted(unknown)}")
            base[k].update(v)
        else:
            base[k] = v
    return TriConfig.from_dict(base)


def remine(run: str, jobs=None) -> dict:
    """Re-mine a saved run's rulebook (Tri.mine_rules) and save it back."""
    t0 = time.time()
    model = os.path.join(run, "model") if os.path.isdir(os.path.join(run, "model")) else run
    catp = os.path.join(run, "catalog.json")
    if not os.path.exists(catp):
        raise SystemExit(f"{catp} not found: --rules-only needs a run directory from this script")
    cat = Catalog.from_json(catp)
    tri = Tri.load(model, cat)
    if jobs is not None:
        tri.cfg.passes.n_jobs = jobs
    old = tri.rulebook.summary()
    rep = tri.mine_rules()
    tri.save(model)
    log(f"re-mined {model}: {old['rules']} -> {rep['rules']['rules']} rules "
        f"({time.time() - t0:.0f} s); selection {rep['selection']}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit tri on a planner builds.json.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--builds", default="builds.json", help="planner export (object, array or JSONL)")
    ap.add_argument("--out", default="runs", help="directory for runs and the parse cache")
    ap.add_argument("--name", help="run directory name (default: timestamp)")
    ap.add_argument("--catalog", help="your catalog.json; default derives one from the builds")
    ap.add_argument("--config", help="JSON file of TriConfig overrides, e.g. {\"passes\": {\"theta_c\": 0.8}}")
    ap.add_argument("--pipeline-config", help="JSON file of PipelineConfig overrides")
    ap.add_argument("--patches", help="JSON list of patches (Patch fields: name, time, changed, added, "
                    "removed, renamed, note); applied before fitting")
    ap.add_argument("--ruleset", default="unknown")
    ap.add_argument("--limit", type=int, help="read only the first N records (trials)")
    ap.add_argument("--reprepare", action="store_true", help="ignore the parse cache")
    # game data
    ap.add_argument("--gamedata", action="append", default=[], metavar="auto|PATH|none",
                    help="auto (default) = pocamind/data, downloaded once to <out>/gamedata/; "
                         "all = deepwoken.co + pocamind/data; or local paths (repeat to merge, first wins)")
    ap.add_argument("--gamedata-ref", default="main", help="branch, tag or commit for --gamedata auto")
    ap.add_argument("--refresh-gamedata", action="store_true", help="download the game data again")
    ap.add_argument("--keep-group", action="append", default=[], metavar="GROUP",
                    help="granted group to treat as choices, e.g. faction (repeatable)")
    # exclusion
    ap.add_argument("--keep-meta-bound", action="store_true", help="do not exclude meta-bound talents")
    ap.add_argument("--exclude-meta-bound", action="store_true",
                    help="exclude meta-bound talents even with game data")
    ap.add_argument("--exclude-auto-grant", action="store_true",
                    help="also exclude items with pick rate >= auto_grant_rate")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME", help="extra item to exclude")
    ap.add_argument("--keep", action="append", default=[], metavar="NAME",
                    help="never exclude this item in passes 2-3")
    # fitting and measurement
    ap.add_argument("--test-frac", type=float, default=0.02, help="held-out share of lineages")
    ap.add_argument("--eval-n", type=int, default=2000, help="held-out builds used for the benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="compute device for training and latent solves: auto (cuda > mps > "
                    "torch cpu > numpy), cuda, cuda:N, mps (Apple Metal), cpu (torch), numpy. Needs torch "
                    "except numpy. Default: the config's (auto), or TRI_DEVICE")
    ap.add_argument("--jobs", type=int, help="processes for rule mining (0 auto, -1 all cores, 1 off); "
                    "default: the config's (auto), or TRI_JOBS")
    ap.add_argument("--rules-only", metavar="RUN",
                    help="re-mine the rules of a saved run (RUN/model + RUN/catalog.json) with today's "
                         "rule language and certification; the concept space is not touched (v3.0)")
    a = ap.parse_args(argv)

    np.random.seed(a.seed)
    if a.rules_only:
        return remine(a.rules_only, a.jobs)
    if not os.path.exists(a.builds):
        raise SystemExit(f"{a.builds} not found (use --builds PATH)")
    t0 = time.time()
    cfg = load_cfg(a.pipeline_config)
    tcfg = load_tri_cfg(a.config)
    tcfg.seed = a.seed
    if a.device:
        tcfg.space.device = a.device
    if a.jobs is not None:
        tcfg.passes.n_jobs = a.jobs
    cfg.keep_meta = True                      # oath / murmur feed the concept space and rules

    # ------------------------------------------------------------ game data
    os.makedirs(a.out, exist_ok=True)
    gd = get_gamedata(a)
    if gd is not None:
        src = gd.source
        parts = [src] if src.get("name") != "merged" else [src["primary"]] + src["others"]
        for p in parts:
            log(f"game data {p.get('name')}: {p.get('path')}  {p.get('n_talents')} talents, "
                f"{p.get('n_mantras')} mantras" + (f"  commit {p['sha'][:10]}" if p.get("sha") else ""))
        if src.get("added"):
            log(f"game data merged: {src['n_talents']} talents, {src['n_mantras']} mantras "
                f"(added from later sources: {src['added']})")
        if gd.errors:
            log(f"game data: {len(gd.errors)} entries with unreadable requirements (see report.json)")
        cfg.granted_known = {**gd.granted_known(a.keep_group), **cfg.granted_known}
        cfg.canonical_names = {**gd.canonical_names(), **cfg.canonical_names}

    plog = None
    if a.patches:                                  # added items stay in the catalog however rarely picked
        from tri.game.patch import PatchLog
        with open(a.patches) as f:
            plog = PatchLog.from_json(json.load(f))
        extra = [i for i in plog.added_items() if i not in set(cfg.keep_items)]
        if extra and gd is not None:
            cfg.keep_items = tuple(cfg.keep_items) + tuple(extra)
            log(f"patches add {len(extra)} items; they are kept in the catalog and pre-release builds "
                "do not count against them")

    # ------------------------------------------------------------ prepare (cached)
    cdir = os.path.join(a.out, "cache", cache_key(a.builds, a, cfg, gd))
    if not a.reprepare and os.path.exists(os.path.join(cdir, "builds.npz")):
        log(f"loading cached parse {cdir}")
        ps = PretrainSet.load(cdir)
    else:
        log(f"parsing {a.builds}")
        cat = Catalog.from_json(a.catalog) if a.catalog else None
        ps = prepare_pretraining(a.builds, cat, cfg, a.ruleset, a.limit, progress=100_000,
                                 gamedata=gd, keep_groups=a.keep_group)
        ps.save(cdir)
        log(f"parse cached at {cdir}")
    t_prep = time.time() - t0
    r = ps.report
    log(f"records {r['records']:,}  kept {r['kept']:,}  items {r['n_items']}  "
        f"effective n {r['weight_summary']['effective_n']:,.0f}")
    if not r["kept"]:
        raise SystemExit("no builds survived filtering; see report drops: " + json.dumps(r["drops"]))

    # ------------------------------------------------------------ granted exclusion (passes 2-3)
    excluded = {n: dict(pass_="gamedata" if gd is not None and n in gd.items else "name",
                        group=v["group"], count=v["count"])
                for n, v in r.get("granted_excluded", {}).items()}
    keep = set(a.keep)
    drop = {}
    if a.exclude_meta_bound or (gd is None and not a.keep_meta_bound):
        for c in r.get("meta_bound_candidates", []):
            if c["item"] not in keep:
                drop[c["item"]] = dict(pass_="meta_bound", group=c["field"].lower(), value=c["value"],
                                       purity=c["purity"], count=c["support"])
    if a.exclude_auto_grant:
        for n, p in r.get("auto_granted_candidates", []):
            if n not in keep and n not in drop and ps.catalog.items[n].kind.value == "talent":
                drop[n] = dict(pass_="auto_grant", group="auto_grant", pick_rate=p)
    for n in a.exclude:
        drop.setdefault(n, dict(pass_="manual", group=granted_group(n, cfg) or "manual"))
    drop = {n: v for n, v in drop.items() if n in ps.catalog.index}
    if drop:
        ps = exclude_items(ps, list(drop), cfg)
        excluded.update(drop)
    by_pass = {}
    for v in excluded.values():
        by_pass[v["pass_"]] = by_pass.get(v["pass_"], 0) + 1
    log(f"excluded granted talents: {by_pass or 'none'}  -> {len(ps.catalog)} items")

    # ------------------------------------------------------------ run directory
    run = os.path.join(a.out, a.name or time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(run, "model"), exist_ok=True)
    catalog_to_json(ps.catalog, os.path.join(run, "catalog.json"))
    item_counts = ps.report.pop("item_counts", {})
    with open(os.path.join(run, "item_counts.json"), "w") as f:
        json.dump(dict(min_item_count=ps.report.get("min_item_count"), items=item_counts), f, indent=0, sort_keys=True)
    with open(os.path.join(run, "report.json"), "w") as f:
        json.dump(ps.report, f, indent=2, default=str)
    with open(os.path.join(run, "granted.json"), "w") as f:
        json.dump(dict(gamedata=gd.source if gd is not None else None, keep_groups=a.keep_group,
                       granted_known=cfg.granted_known, canonical_names=cfg.canonical_names),
                  f, indent=1, sort_keys=True, default=str)
    with open(os.path.join(run, "excluded_talents.json"), "w") as f:
        json.dump({n: {("pass" if k == "pass_" else k): x for k, x in v.items()}
                   for n, v in sorted(excluded.items())}, f, indent=1)

    # ------------------------------------------------------------ fit
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump(jsonable(dict(args=vars(a), source=dict(path=os.path.abspath(a.builds),
                                                          size=os.path.getsize(a.builds)),
                                cache=cdir, gamedata=gd.source if gd is not None else None,
                                gamedata_fingerprint=gd.fingerprint() if gd is not None else None,
                                pipeline=asdict(cfg), tri=tcfg.to_dict())), f, indent=1)
    tr_idx, te_idx = ps.split(test_frac=a.test_frac, seed=a.seed)
    log(f"fit on {len(tr_idx):,} builds, held out {len(te_idx):,} (whole lineages)")
    tri = Tri(ps.catalog, tcfg)
    from tri.model import accel
    d = accel.describe(tcfg.space.device)
    log(f"device: {d['device']} ({d['dtype']}" + (f", {d['gpu']}" if d.get("gpu") else "") + ")"
        + (f"; torch {d['torch']}" if d.get("torch") else "; torch not installed (pip install torch)"))
    if plog is not None:
        for pt in plog.patches:
            pt.changed = [x for x in pt.changed if x in ps.catalog.index]
            pt.added = [x for x in pt.added if x in ps.catalog.index]
            pt.removed = [x for x in pt.removed if x in ps.catalog.index]
            tri.patches.add(pt)
        log(f"patches: {', '.join(pt.name for pt in plog.patches)}")
    t1 = time.time()
    rep = tri.fit(ps.builds.subset(tr_idx), weights=ps.weights[tr_idx])
    t_fit = time.time() - t1
    s = rep["rules"]
    log(f"fit {t_fit:.1f}s: {len(rep['space'].get('regions', []))} archetype regions; rules {s}")
    v = rep["space"].get("validation") or {}
    if v:
        log(f"held-out field check on {v['builds']} builds: deviance {v['deviance']:.4f} +- {v['deviance_se']:.4f}, "
            f"training builds {v['probe_deviance']:.4f} (gap {v['gap']:+.4f}), {v['explained']:.1%} explained")
    lr = rep.get("locators") or {}
    if lr.get("scored"):
        sel = lr.get("metric_selection") or {}
        log(f"locators read from each other: {lr['accuracy']:.0%} correct, log-likelihood {lr['loglik']:.2f} "
            f"(chance {lr['chance_loglik']:.2f}), {len(lr['conflicts'])} conflicts"
            + (f"; metric power {sel['current']:g} -> {sel['chosen']:g}" if sel.get("moved") else ""))

    # ------------------------------------------------------------ measure (held-out lineages)
    test = ps.builds.subset(te_idx)
    comp = rules = cert = mapping = None
    if len(test):
        comp = completion_benchmark(tri, test, n=a.eval_n, seed=a.seed)
        rules = rule_benchmark(tri, test, n=a.eval_n, seed=a.seed)
        cert = certification(tri, test)
        sub = [test[i] for i in range(min(len(test), 500))]
        mapping = mapping_benchmark(tri, sub, seed=a.seed)
        log(f"held-out rule certification: {cert.get('share')} of {cert.get('checked')} rules hold; "
            f"concept-space isometry error {mapping.get('isometry_error', 0):.1e}, "
            f"worth bound holds on {mapping.get('bound_holds')} of pairs")
        log("held-out completion recall: " + ", ".join(f"{k} {comp[k]:.3f}" for k in ("tri", "worth_only", "popularity")))
        log("by pass (tri): " + ", ".join(f"{p} {v:.2f}" for p, v in comp["by_pass"]["tri"].items()
                                          if v is not None))

    # ------------------------------------------------------------ export
    tri.save(os.path.join(run, "model"))
    with open(os.path.join(a.out, "LATEST"), "w") as f:
        f.write(os.path.abspath(run))
    with open(os.path.join(run, "regions.txt"), "w") as f:
        f.write(json.dumps(tri.space.report(), indent=1, default=str) + "\n")
    metrics = dict(completion=comp, rules=rules, certification=cert, mapping=mapping, fit=jsonable(rep),
                   n_train=int(len(tr_idx)), n_test=int(len(te_idx)), n_items=len(ps.catalog),
                   seconds=dict(prepare=round(t_prep, 1), fit=round(t_fit, 1), total=round(time.time() - t0, 1)),
                   note="agreement with planner conventions, not build quality; reviews measure quality")
    with open(os.path.join(run, "metrics.json"), "w") as f:
        json.dump(jsonable(metrics), f, indent=1)
    log(f"saved {run}  (load: Tri.load('{run}/model', Catalog.from_json('{run}/catalog.json')))")
    return run


if __name__ == "__main__":
    main()
