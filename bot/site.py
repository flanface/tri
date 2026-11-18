"""The interactive graphs as a static site with tabs, for GitHub Pages.

    python -m bot.site build --run runs/live                    # reads the tables (.env)
    python -m bot.site build --run runs/live --snapshot snap.json
    python -m bot.site build --run runs/live --no-tables        # the run alone: no reviews or locators

writes site/ (index.html with one tab per graph, and one self-contained page
per graph). Commit site/ to main: .github/workflows/pages.yml publishes it to
https://<owner>.github.io/<repo>/ with the workflow's own token, so no key is
needed. Rebuild after a refit or when the locators change.

    concept    the concept space on PCA axes 1-2-3: builds by region, region centres, locators
    atlas      the archetype atlas on PCA axes 1-2-3: archetype fields, centres, locators, candidates
    worth      items on worth axes 4-5-6 from the dense origin
    relative   the same, by which builds value an item more than usual instead of by pick rate

Every page turns, zooms and pans; its command box finds a build, item,
locator or centre by name, and on the worth pages adds that item's vector
from the origin.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import shutil
from typing import List, Optional

import numpy as np

CONCEPT_SAMPLE = 3000
WORTH_SAMPLE = 2000
FIELD_MARGIN = 0.25         # atlas: least pi1 - pi2 for a build to shape or show in its archetype's field
FIELD_CORE = 0.8            # atlas: share of those builds (nearest the median) a field spans
VIEW = (22, -58)

TABS = (("concept", "Concept space", "PC1-2-3: builds coloured by region, region centres and locators"),
        ("atlas", "Archetype atlas", "PC1-2-3: archetype fields from decisively placed builds, centres, "
                                     "locators and candidate archetypes"),
        ("worth", "Worth space", "PC4-5-6: talents and mantras; type an item to draw its vector from the origin"),
        ("relative", "Relative worth", "PC4-5-6 by which builds value an item more than usual, not by pick rate"))


# ------------------------------------------------------------------ scenes
def concept_scene(svc, sample: int = CONCEPT_SAMPLE):
    from tri.api import figure as F
    from tri.api import plots as render
    d = [0, 1, 2]
    with svc.lock:
        cm = render.concept_map(svc.tri, sample=sample, method="pca", k=max(3, min(9, int(svc.tri.space.cfg.dim))))
        locs = list(svc.tri.locators)
        lp = render.place(svc.tri, [l.build for l in locs], cm["projection"])["xy"][:, d] if locs else None
    return F.Scene(P=cm["xy"][:, d], group=np.asarray(cm["region"]), names=list(cm["region_names"]), style="dots",
                   centres=np.asarray(cm["centre_xy"])[:, d] if len(cm["centre_xy"]) else None,
                   locators=lp, labels=[str(i) for i in cm.get("ids", [])] or None,
                   locator_labels=[f"locator {l.locator_id} · {l.archetype}"[:60] for l in locs] or None)


def atlas_scene(svc):
    from tri.api import figure as F

    from .insight import _atlas
    d = [0, 1, 2]
    with svc.lock:
        c = _atlas(svc)
        at, av = c.atlas, c.view
        pl = np.asarray(at.placed, bool)
        dec = pl & ((np.asarray(at.pi1) - np.asarray(at.pi2)) >= FIELD_MARGIN)
        stars = [(str(x["id"]), np.asarray(x["xy"])[d]) for x in (av.get("candidates") or [])]
        ids = np.asarray(at.ids, object)[dec]
        top = np.asarray(at.top)[dec]
        return F.Scene(P=av["xy"][dec][:, d], group=top, names=list(at.names), style="fields", core=FIELD_CORE,
                       centres=np.asarray(av["centre_xy"])[:, d],
                       locators=np.asarray(av["loc_xy"])[:, d] if len(av["loc_xy"]) else None,
                       stars=stars, labels=[f"{i} · {at.names[g]}" for i, g in zip(ids, top)])


def worth_scene(svc, relative: bool = False, sample: int = WORTH_SAMPLE):
    from tri.api import figure as F

    from .worth import WORTH_DIMS, worth_space
    with svc.lock:
        ws = worth_space(svc, relative, sample)
    X = np.asarray(ws["xy"])
    if X.shape[1] <= max(WORTH_DIMS):
        raise ValueError("the worth space has fewer than 6 axes")
    X = X[:, list(WORTH_DIMS)]
    kinds = sorted(set(ws["kind"]))
    return F.Scene(P=X, group=np.array([kinds.index(k) for k in ws["kind"]]), names=kinds, style="dots",
                   point_colour=[F.KIND_COLOURS.get(k, F.FAINT) for k in ws["kind"]],
                   origin=F.dense_origin(X), point_size=6.0, labels=list(ws["items"]))


# ------------------------------------------------------------------ pages
def _index(meta: dict) -> str:
    buttons = "".join(f'<button data-tab="{k}" title="{html.escape(d)}">{html.escape(t)}</button>' for k, t, d in TABS)
    frames = "".join(f'<iframe id="f-{k}" data-src="{k}.html" title="{html.escape(t)}"></iframe>' for k, t, _ in TABS)
    descs = json.dumps({k: d for k, _, d in TABS})
    foot = html.escape(" · ".join(f"{k}: {v}" for k, v in meta.items()))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tri · interactive graphs</title>
<style>
:root{{--bg:#0e1015;--fg:#e6e6e6;--mut:#8b93a1;--line:#232733;--acc:#5aa9ff}}
*{{box-sizing:border-box}}html,body{{height:100%;margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
body{{display:flex;flex-direction:column}}
header{{display:flex;align-items:center;gap:16px;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}}
h1{{font-size:16px;margin:0;font-weight:600;letter-spacing:.02em}}
nav{{display:flex;gap:4px;flex-wrap:wrap}}
nav button{{background:none;border:1px solid transparent;color:var(--mut);padding:6px 12px;border-radius:6px;
cursor:pointer;font:inherit}}
nav button:hover{{color:var(--fg)}}nav button[aria-selected=true]{{color:var(--fg);border-color:var(--line);
background:#161a22}}
#desc{{color:var(--mut);font-size:13px;flex:1;min-width:200px}}
main{{flex:1;position:relative;min-height:0}}
iframe{{position:absolute;inset:0;width:100%;height:100%;border:0;display:none}}
iframe.on{{display:block}}
footer{{color:var(--mut);font-size:12px;padding:4px 14px;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1>tri</h1><nav role="tablist">{buttons}</nav><div id="desc"></div></header>
<main>{frames}</main>
<footer>drag to turn · right or shift-drag to pan · wheel to zoom · type a name in a graph's command box · {foot}</footer>
<script>
const D={descs};
const tabs=[...document.querySelectorAll('nav button')];
function show(k){{
  if(!D[k])k=tabs[0].dataset.tab;
  for(const b of tabs)b.setAttribute('aria-selected',b.dataset.tab===k);
  for(const f of document.querySelectorAll('iframe')){{
    const on=f.id==='f-'+k;
    if(on&&!f.src)f.src=f.dataset.src;      // each graph loads the first time its tab opens
    f.classList.toggle('on',on);
  }}
  document.getElementById('desc').textContent=D[k];
  if(location.hash.slice(1)!==k)history.replaceState(null,'','#'+k);
}}
for(const b of tabs)b.onclick=()=>show(b.dataset.tab);
addEventListener('hashchange',()=>show(location.hash.slice(1)));
show(location.hash.slice(1));
</script></body></html>
"""


def build(svc, out: str, run_name: str = "", note: str = "") -> List[str]:
    """Write out/index.html and one page per tab; the files written."""
    from tri.api import interactive_page as page
    os.makedirs(out, exist_ok=True)
    pages = {
        "concept": (lambda: concept_scene(svc), "concept space", ["PC1", "PC2", "PC3"]),
        "atlas": (lambda: atlas_scene(svc), "archetype atlas", ["PC1", "PC2", "PC3"]),
        "worth": (lambda: worth_scene(svc, False), "worth space", ["PC4", "PC5", "PC6"]),
        "relative": (lambda: worth_scene(svc, True), "relative worth space", ["PC4", "PC5", "PC6"]),
    }
    written = []
    for k, (scene, title, axes) in pages.items():
        p = os.path.join(out, f"{k}.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(page(scene(), title, axes, VIEW))
        written.append(p)
    t = svc.tri
    meta = {"run": run_name or "?", "reviews": len(t.reviews), "locators": len(t.locators or []),
            "built": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if note:
        meta["note"] = note
    for name, text in (("index.html", _index(meta)), (".nojekyll", "")):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(os.path.join(out, name))
    return written


def _main(argv: Optional[list] = None) -> None:
    from .evaluate import _snapshot, load_service
    ap = argparse.ArgumentParser(prog="python -m bot.site", description="tri's interactive graphs for GitHub Pages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the site")
    b.add_argument("--run", default="runs/live", help="fitted run dir, or a runs folder with LATEST")
    b.add_argument("--out", default="site")
    b.add_argument("--snapshot", help="table snapshot (python -m bot.evaluate snapshot); default: read the tables")
    b.add_argument("--no-tables", action="store_true", help="the run alone: no reviews or locators")
    b.add_argument("--env", default=".env")
    a = ap.parse_args(argv)
    if a.no_tables:
        data = {"reviews": []}
    elif a.snapshot:
        data = json.load(open(a.snapshot))
    else:
        import asyncio
        import tempfile
        tmp = os.path.join(tempfile.mkdtemp(), "snap.json")
        asyncio.run(_snapshot(tmp, a.env))
        data = json.load(open(tmp))
        shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
    svc = load_service(data, a.run)
    run_name = os.path.basename(os.path.abspath(getattr(svc, "run", a.run)))
    for p in build(svc, a.out, run_name):
        print(f"{p}  {os.path.getsize(p) / 1e6:.2f} MB")


if __name__ == "__main__":
    _main()
