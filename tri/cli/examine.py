"""metrics from fitted runs
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from tri import Catalog, Tri  # noqa: E402
from tri.game.gamedata import format_reqs  # noqa: E402


# ------------------------------------------------------------------ loading
def resolve_run(run, out):
    if run:
        return run
    p = os.path.join(out, "LATEST")
    if not os.path.exists(p):
        raise SystemExit(f"no --run given and no {p}; run python -m tri.cli.fit first")
    with open(p) as f:
        return f.read().strip()


def load_run(run):
    cat = Catalog.from_json(os.path.join(run, "catalog.json"))
    tri = Tri.load(os.path.join(run, "model"), cat)
    return tri, cat


# ------------------------------------------------------------------ part 1
def rules_json(tri) -> list:
    """Every rule as a dict (PassRule.to_dict plus its text): conditions are
    lists, so item names containing " and " stay unambiguous."""
    out = []
    for r in tri.rulebook.rules:
        d = r.to_dict()
        d["text"] = r.text(tri.rulebook.region_names)
        out.append(d)
    return out


def audit_requirements(tri, cat, max_rate: float = 0.01, min_support: float = 1000.0,
                       min_contrast: float = 0.1) -> list:
    """Active exclusions with one condition "Stat < v", rate <= max_rate on
    >= min_support builds and contrast >= min_contrast. Conditions read highest
    stats, so below v at the highest value means below v at both stages: on
    builds the catalog calls able to take the item, a near-zero rate suggests
    the real requirement is higher than the catalog's (or one the catalog does
    not hold). For checking against the game data; never enforced."""
    out = []
    for r in tri.rulebook.active():
        if r.effect != "never" or len(r.conds) != 1:
            continue
        c = r.conds[0]
        if c.kind != "stat" or c.op != "<":
            continue
        if r.confidence > max_rate or r.support < min_support or (r.contrast or 0) < min_contrast:
            continue
        it = cat.items[r.item]
        out.append(dict(item=r.item, condition=c.text(), rate=round(r.confidence, 4), support=round(r.support, 1),
                        contrast=round(r.contrast or 0, 4),
                        catalog_requirement=format_reqs(it.clauses) if it.clauses else str(it.requires or "()"),
                        prerequisites=", ".join(it.prerequisites + [" or ".join(g) for g in it.prerequisites_any])))
    return sorted(out, key=lambda x: -x["support"])


def learned(tri, cat, axes_k=3):
    rules = tri.rulebook.report()
    review_rules = tri.rule_report()
    sp = tri.space.report()
    sp["axes"] = tri.space.axes(6)[:axes_k]
    rate = tri.data.pick_rate()[0]
    rows = []
    for i in cat.ids:
        rs = tri.rulebook.for_item(i)
        ps = tri.rulebook.pass_of(i)
        v = getattr(tri.rulebook, "vetoed", {}).get(i) if ps == "luxury" else None
        vb = (f"pick cluster {v['cluster']} ({v['weight']:.0f} builds, {v['rate']:.0%} take it)" if v else "")
        rows.append(dict(item=i, kind=cat.items[i].kind.value, pass_=ps,
                         pick_rate=round(float(rate[cat.index[i]]), 4), vetoed_by=vb,
                         rules=" | ".join(r.text(tri.rulebook.region_names) for r in rs)))
    order = {"foundational": 0, "central": 1, "luxury": 2}
    rows.sort(key=lambda r: (order.get(r["pass_"], 3), -r["pick_rate"]))
    return rules, review_rules, sp, rows


def print_part1(rules, review_rules, sp, rows):
    s = {}
    for r in rows:
        s[r["pass_"]] = s.get(r["pass_"], 0) + 1
    print(f"\n== items by pass: {s}")
    print(f"\n== rulebook ({len(rules)} rules)")
    for line in rules[:60]:
        print("  " + line)
    if len(rules) > 60:
        print(f"  ... {len(rules) - 60} more in analysis/rules.txt")
    if review_rules:
        print(f"\n== review rules in force ({len(review_rules)})")
        for line in review_rules[:20]:
            print("  " + line)
    print(f"\n== concept space: {sp['dim']} latent dims, {sp['coordinates']} coordinates, "
          f"{sp['deviance_explained']:.0%} of deviance explained (vs pick rates alone)")
    for r in sp["regions"]:
        print(f"  region {r['id']:>2}  {r['share']:6.1%}  {r['name']}")
    for ax in sp["axes"]:
        print(f"  axis {ax['axis']} ({ax['share']:.0%}): + {', '.join(ax['positive'][:4])}  |  - "
              f"{', '.join(ax['negative'][:4])}")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect a tri run: rulebook, regions, axes, item table.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", help="run directory (default: contents of <out>/LATEST)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--axes", type=int, default=3, help="concept-space axes to print")
    ap.add_argument("--audit-requirements", action="store_true",
                    help="list stat exclusions that look like missing requirements (requirement_audit.csv)")
    a = ap.parse_args(argv)

    run = resolve_run(a.run, a.out)
    tri, cat = load_run(run)
    adir = os.path.join(run, "analysis")
    os.makedirs(adir, exist_ok=True)
    print(f"[examine] run {run}: {len(cat)} items, {len(tri.data)} builds kept")
    result = dict(run=run)

    if True:
        rules, review_rules, sp, rows = learned(tri, cat, a.axes)
        print_part1(rules, review_rules, sp, rows)
        with open(os.path.join(adir, "rules.txt"), "w") as f:
            f.write("\n".join(rules + ([""] + review_rules if review_rules else [])) + "\n")
        with open(os.path.join(adir, "regions.json"), "w") as f:
            json.dump(sp, f, indent=1, default=str)
        with open(os.path.join(adir, "rules.json"), "w") as f:
            json.dump(rules_json(tri), f, indent=1)
        with open(os.path.join(adir, "items.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        result.update(items=rows, rules=rules, space=sp)
    if a.audit_requirements:
        aud = audit_requirements(tri, cat)
        with open(os.path.join(adir, "requirement_audit.csv"), "w", newline="") as f:
            fields = ["item", "condition", "rate", "support", "contrast", "catalog_requirement", "prerequisites"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(aud)
        print(f"\n== requirement audit: {len(aud)} stat exclusions that look like requirements")
        for x in aud[:20]:
            print(f"  {x['item']}: never when {x['condition']} ({x['rate']:.0%} of {x['support']:.0f}); "
                  f"catalog: {x['catalog_requirement']}")
        result["requirement_audit"] = aud

    print(f"\n[examine] outputs in {adir}")
    return result


if __name__ == "__main__":
    main()
