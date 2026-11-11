"""Read stored reviews and locators back out of Supabase.

Training stays in your own pipeline; this only exports and checks.

    python -m bot.ingest --out reviews.json              # all reviews, review_to_dict format
    python -m bot.ingest --out reviews.json --check runs # also validate against a run's catalog
                                                         # and print a dry-insert summary
    python -m bot.ingest --locators locators.json        # locator rows (an export; the bot loads them itself)

In code:
    reviews, problems = asyncio.run(fetch_reviews(store))
    tri.add_reviews(reviews)        # or tri.dry_insert(reviews[0]) first
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import List, Sequence, Tuple

from tri.core.codec import review_from_dict, review_to_dict
from tri.core.types import Review

from .db import SupabaseStore


def review_from_row(row: dict) -> Review:
    """One reviews row -> Review. operations must be the bot's stored object
    (schema tri.review/1); a bare list of operations has no build snapshot and
    is refused."""
    body = row["operations"]
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict) or "original" not in body:
        raise ValueError(f"review {row.get('id')}: operations has no build snapshot (not written by the bot)")
    return review_from_dict(body)


def indicator_matches(rows: Sequence[dict], archetype: str, item: str) -> List[dict]:
    """Reviews whose archetype note ("# A: x, y") names item as an indicator of
    archetype. archetype matches by family, case-insensitively ("chime" matches
    "Chime" and "Gank / Chime"); item matches case-insensitively.
    Returns [{id, build, link, note}] in id order."""
    from tri.reviews.variants import family_key, split_archetype
    want_fam, want_item = family_key(archetype), " ".join(str(item).split()).casefold()
    want_var = (split_archetype(archetype)[1] or "").casefold()
    out = []
    for row in rows:
        body = row.get("operations")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                continue
        if not isinstance(body, dict):
            continue
        for n in body.get("notes") or []:
            if not isinstance(n, dict) or n.get("kind") != "archetype":
                continue
            name = str(n.get("archetype") or "")
            fam, var = family_key(name), (split_archetype(name)[1] or "").casefold()
            if (fam, var) != (want_fam, want_var) if want_var else want_fam not in (fam, var):
                continue
            if any(" ".join(str(c).split()).casefold() == want_item for c in n.get("items") or []):
                out.append(dict(id=int(row["id"]), build=body.get("build_name") or "", link=body.get("link") or "",
                                note=n.get("text") or f"{name}: " + ", ".join(map(str, n.get("items") or []))))
                break
    return sorted(out, key=lambda d: d["id"])


PURGE_MARKS = {
    "generally_good": "generally good adds (GG)",
    "add": "every add",
    "remove": "every removal",
    "iffy": "iffy marks",
    "indicator": "archetype indicator notes (# A: x)",
    "core": "core and avoid notes (# core: x)",
    "endorsement": "keeping it unmarked (endorsement by silence)",
    "strong_sm": "strong scenario mismatch removals (SM strong)",
    "sm": "scenario mismatch removals, weak or strong (SM)",
    "all": "every mark above",
}


def _norm(x) -> str:
    return " ".join(str(x).split()).casefold()


def _body(row: dict):
    b = row.get("operations")
    if isinstance(b, str):
        try:
            b = json.loads(b)
        except ValueError:
            return None
    return b if isinstance(b, dict) else None


def _families(body: dict) -> set:
    from tri.reviews.variants import family_key
    return {family_key(str(n.get("archetype") or "")) for n in body.get("notes") or []
            if isinstance(n, dict) and n.get("kind") in ("archetype", "is") and n.get("archetype")}


def with_dependants(names: Sequence[str], catalog=None) -> List[str]:
    """names plus every talent or mantra that needs one of them, transitively
    (Knife's Journey -> Decisive Blow), in catalog spelling when known."""
    out = list(dict.fromkeys(names))
    if catalog is None:
        return out
    key = {_norm(k): k for k in catalog.items}
    out = [key.get(_norm(n), n) for n in out]
    grew = True
    while grew:
        grew = False
        have = {_norm(x) for x in out}
        for k, it in catalog.items.items():
            pre = set(it.prerequisites) | {y for g in it.prerequisites_any for y in g}
            if _norm(k) not in have and any(_norm(p) in have for p in pre):
                out.append(k); grew = True
    return out


def purge_plan(rows: Sequence[dict], item: str, mark: str, archetype: str = None, reviewer: str = None,
               table: str = "reviews", having: Sequence[str] = ()) -> List[dict]:
    """What /reviews purge undoes (v3.1.1): for every stored review row,
    the marks on item of the given kind (PURGE_MARKS), stripped from a copy of
    its body. The item also goes on the body's "neutral" list, so the review says
    nothing about it any more: no claim, no endorsement by silence. archetype:
    only reviews that state that family (indicator: notes naming it); reviewer:
    only that reviewer's rows; having (v3.2.2): only rows whose reviewed build
    holds one of these items. Returns [{id, table, build, removed: [text], body}]
    in id order; rows with nothing to undo are left out."""
    import copy
    from tri.reviews.variants import family_key, split_archetype
    if mark not in PURGE_MARKS:
        raise ValueError(f"unknown mark {mark!r}: one of {', '.join(PURGE_MARKS)}")
    names = [item] if isinstance(item, str) else list(item)
    anyitem = any(str(x).strip() in ("*", "any", "") for x in names)        # v3.4: "*" = every item
    wants = {_norm(x) for x in names}
    match = (lambda x: True) if anyitem else (lambda x: _norm(x) in wants)
    item = ", ".join(names)
    every = mark == "all"
    fam = family_key(archetype) if archetype else None
    var = (split_archetype(archetype)[1] or "").casefold() if archetype else ""
    out = []
    for row in rows:
        body = _body(row)
        if body is None or "operations" not in body:
            continue
        if reviewer and str(body.get("reviewer_id")) != str(reviewer):
            continue
        if fam and mark != "indicator" and fam not in _families(body):
            continue
        if having:
            o0 = body.get("original") or {}
            if not {_norm(x) for x in having} & {_norm(x) for x in list(o0.get("talents") or []) + list(o0.get("mantras") or [])}:
                continue
        b = copy.deepcopy(body)
        gone, stripped = [], []
        keep_ops = []
        for o in b.get("operations") or []:
            if not match(o.get("item_id", "")):
                keep_ops.append(o)
                continue
            act, rsn = o.get("action"), o.get("reason")
            sm = act == "remove" and rsn == "scenario_mismatch"
            hit = (every and not anyitem) or (mark == "generally_good" and act == "add" and rsn == "generally_good") \
                or (mark == act and act in ("add", "remove", "iffy")) \
                or (mark == "sm" and sm) or (mark == "strong_sm" and sm and o.get("strength") == "strong")
            if hit:
                gone.append(f"{act} {o.get('item_id')}" + (f" ({rsn})" if rsn else ""))
                stripped.append(o.get("item_id"))
            else:
                keep_ops.append(o)
        b["operations"] = keep_ops
        notes = []
        for n in b.get("notes") or []:
            if not isinstance(n, dict) or n.get("kind") not in ("archetype", "core", "avoid"):
                notes.append(n)
                continue
            kind = n.get("kind")
            if kind == "archetype" and fam and mark == "indicator":
                nf, nv = family_key(str(n.get("archetype") or "")), (split_archetype(str(n.get("archetype") or ""))[1] or "").casefold()
                if ((nf, nv) != (fam, var)) if var else fam not in (nf, nv):
                    notes.append(n)
                    continue
            if not (every or (mark == "indicator" and kind == "archetype") or (mark == "core" and kind in ("core", "avoid"))):
                notes.append(n)
                continue
            items = [x for x in n.get("items") or [] if anyitem or not match(x)]
            if len(items) == len(n.get("items") or []):
                notes.append(n)
                continue
            gone.append(f"note # {n.get('archetype') or kind}: {item}")
            stripped += [x for x in n.get("items") or [] if match(x)]
            if items:
                n = dict(n, items=items)
                n.pop("text", None)
                notes.append(n)
        b["notes"] = notes
        orig = b.get("original") or {}
        held = [] if anyitem else [x for x in list(orig.get("talents") or []) + list(orig.get("mantras") or [])
                                   if match(x)]
        for h in held:
            endorsed = not any(
                _norm(o.get("item_id", "")) == _norm(h) for o in body.get("operations") or [])
            if (every or mark == "endorsement") and endorsed and h not in (b.get("neutral") or []):
                gone.append(f"endorsement of {h}")
                stripped.append(h)
        if not gone:
            continue
        for name in dict.fromkeys(x for x in stripped if x):
            if name not in (b.get("neutral") or []):
                b["neutral"] = list(b.get("neutral") or []) + [name]
        b["endorsed"] = bool(b.get("endorsed")) and not b["operations"]
        out.append(dict(id=int(row["id"]), table=table, build=body.get("build_name") or "", link=body.get("link") or "",
                        removed=gone, body=b))
    return sorted(out, key=lambda d: d["id"])


def is_audit_row(row: dict) -> bool:
    """A leftover audit (removed in v3.5) stored in the reviews table: scope
    "audit", or only overturn / uphold verdicts. Such rows are skipped."""
    body = row.get("operations")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return False
    if not isinstance(body, dict):
        return False
    ops = [o for o in body.get("operations") or [] if isinstance(o, dict)]
    return body.get("scope") == "audit" or (bool(ops) and all(o.get("action") in ("overturn", "uphold") for o in ops))


async def fetch_reviews(store: SupabaseStore) -> Tuple[List[Review], List[str]]:
    """(reviews, problems) for every row in the reviews table. Audits were
    removed in v3.5: the audits table is never read, and leftover audit rows
    in the reviews table are skipped."""
    out, problems = [], []
    for row in await store.reviews():
        if is_audit_row(row):
            continue
        try:
            out.append(review_from_row(row))
        except Exception as e:
            problems.append(str(e))
    return out, problems


FREEZE_TABLES = ("reviews", "locators", "item_values", "tri_rules", "pool")
_TABLE_ATTR = {"item_values": "values_table", "tri_rules": "rules_table"}
_ORDER = {"item_values": "item.asc", "tri_rules": "rule_id.asc"}       # tables without an id (v3.0.2)


async def freeze_store(store: SupabaseStore, outdir: str) -> str:
    """Every review-record table as stored, frozen with a manifest (tri.record.freeze).
    A missing table is recorded as empty."""
    from tri.sources.record import freeze
    export = {}
    try:
        for t in FREEZE_TABLES:
            name = getattr(store, _TABLE_ATTR.get(t, f"{t}_table"), None) or t
            try:
                export[t] = await store._select_all(name, "*", order=_ORDER.get(t, "id.asc"))
            except Exception as e:                       # noqa: BLE001 (optional tables)
                export[t] = []
                export.setdefault("_missing", []).append(f"{t}: {e}")
    finally:
        await store.close()
    return freeze(export, outdir)


def main(argv=None):
    import os

    from .config import load_dotenv
    ap = argparse.ArgumentParser(description="export tri reviews from Supabase")
    ap.add_argument("--out", help="write reviews here")
    ap.add_argument("--locators", help="write locator rows {id, link, archetype, record, added_by} here")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--check", metavar="RUN", help="validate against this run (dir or runs folder)")
    ap.add_argument("--freeze", metavar="DIR", help="write a frozen, dated record of every table to DIR "
                    "(tri.record.freeze; read with tri.record.load)")
    a = ap.parse_args(argv)
    load_dotenv(a.env)
    store = SupabaseStore(os.environ["SUPABASE_URL"],
                          os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"],
                          os.environ.get("POOL_TABLE", "pool"), os.environ.get("REVIEWS_TABLE", "reviews"),
                          os.environ.get("LOCATORS_TABLE", "locators"))
    if a.freeze:
        path = asyncio.run(freeze_store(store, a.freeze))
        print(f"record frozen to {path}")
        if not a.out and not a.locators:
            return
    if not a.out and not a.locators:
        ap.error("give --out, --locators and/or --freeze")

    async def go():
        try:
            return (await fetch_reviews(store) if a.out else ([], [])), (await store.locators() if a.locators else [])
        finally:
            await store.close()

    (reviews, problems), locs = asyncio.run(go())
    if a.locators:
        with open(a.locators, "w") as f:
            json.dump(locs, f, indent=1)
        print(f"{len(locs)} locators written to {a.locators}")
    if not a.out:
        return
    with open(a.out, "w") as f:
        json.dump([review_to_dict(r) for r in reviews], f, indent=1)
    print(f"{len(reviews)} reviews written to {a.out}; {len(problems)} rows skipped")
    for p in problems:
        print("skipped:", p)
    if a.check:
        from tri.core.codec import validate_review
        from tri.core.types import ReviewError

        from .service import ModelService
        svc = ModelService(a.check)
        ok = 0
        for r in reviews:
            try:
                for w in validate_review(r, svc.catalog):
                    print(f"{r.review_id}: warning: {w}")
                rep = svc.tri.dry_insert(r)
                ok += 1
                b0, a0 = rep["before"][0]["changes"], rep["after"][0]["changes"]
                print(f"{r.review_id}: {len(r.operations)} ops; model on its build before: {b0 or 'no changes'}; "
                      f"after: {a0 or 'no changes'}; {len(rep['overrides'])} older claims overridden")
            except (ReviewError, KeyError) as e:
                print(f"{r.review_id}: invalid for run {svc.run_id}: {e}")
        print(f"{ok}/{len(reviews)} valid for run {svc.run_id}")


if __name__ == "__main__":
    main()
