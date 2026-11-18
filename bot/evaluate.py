"""Evaluate tri on the live reviews database (bot side: it reads Supabase).

    python -m bot.evaluate snapshot  out.json      # save reviews, locators, rules, values, answers
    python -m bot.evaluate run       out.json [--workers 2] [--save rows.json]

`snapshot` reads the tables once; `run` replays them into a fresh service and
prints leave-one-build-out operation F1, add / remove precision and recall, and
review agreement, each with a bootstrap standard error (tri.evaluation)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

TABLES = ("tri_answers", "tri_rules", "item_values", "locators")


async def _snapshot(path: str, env: str) -> None:
    import os
    from bot.config import load_dotenv
    from bot.db import SupabaseStore
    load_dotenv(env)
    s = SupabaseStore(os.environ["SUPABASE_URL"], os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"])
    out = {}
    try:
        out["reviews"] = await s._select_all("reviews", "*", order="id.asc")
        for t in TABLES:
            out[t] = await s._select_all(t, "*")
    finally:
        await s.close()
    json.dump(out, open(path, "w"))
    print({k: len(v) for k, v in out.items()})


def load_service(data: dict, run_dir: str):
    from bot.ingest import is_audit_row, review_from_row
    from bot.service import ModelService
    svc = ModelService(run_dir)
    revs = []
    for row in data["reviews"]:
        if is_audit_row(row):
            continue
        try:
            revs.append(review_from_row(row))
        except Exception:
            pass
    svc.set_memory(revs, data.get("locators") or [])
    if isinstance(data.get("item_values"), list):
        svc.set_values(data["item_values"])
    if isinstance(data.get("tri_rules"), list):
        svc.set_rules(data["tri_rules"])
    if isinstance(data.get("tri_answers"), list):
        svc.set_answers(data["tri_answers"])
    return svc


def _run(path: str, run_dir: str, workers: int, save) -> None:
    from tri.api import evaluation as E
    svc = load_service(json.load(open(path)), run_dir)
    t0 = time.time()
    rows = E.lobo(svc.tri, workers=workers)
    ag = E.agreement(svc.tri)
    m, a = E.bootstrap(rows), E.bootstrap(ag, E.agreement_metrics)
    out = {k: (round(v["value"], 4), round(v["se"], 4)) for k, v in {**m, **a}.items()}
    print(json.dumps(out), "secs", round(time.time() - t0))
    if save:
        json.dump(dict(lobo=rows, agreement=ag), open(save, "w"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="bot.evaluate")
    p.add_argument("cmd", choices=["snapshot", "run"])
    p.add_argument("path")
    p.add_argument("--run", default="runs/live")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save")
    p.add_argument("--env", default=".env")
    a = p.parse_args(argv)
    if a.cmd == "snapshot":
        asyncio.run(_snapshot(a.path, a.env))
    else:
        _run(a.path, a.run, a.workers, a.save)


if __name__ == "__main__":
    main(sys.argv[1:])
