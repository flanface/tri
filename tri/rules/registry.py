"""Every rule that bears on a pick, other than worth (v3.4, math §2.11).

tri.all_rules(item=None, build=None) lists them in one shape:

    id         stable id (drop ids below)
    source     selection | review_rule | sidegrade | category |
               weapon | game | focus | claim
    rule       the rule in symbols
    effect     take | never | yields | blocks | shapes
    items      the items it names (subject first)
    status     active | dropped | inactive (and the selection / review status)
    droppable  True when tri.drop_rule(id) switches it off
    change     how to change it when it is not droppable
    fires      with a build: does it act on that build (None: not checked)

Drop ids:
    <6 hex>                 a selection rule (mined, authored or review evidence)
    rv:<KIND>:<a>[:<b>]     a review rule (induced from the reviews)
    sg:<6 hex>              a mined sidegrade

Hardcoded and not droppable: the weapon rules (tri/game/weapon_rules.py), game
data (stat requirements, prerequisites, exclusive_with, Vow of Iron, the talent
cap 76 - 2m, mantra slots) and the class focus. Review claims (memory) are
changed by editing or purging the reviews, not by dropping.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

PREFIXES = ("rv:", "sg:")      # v3.5: sc: (standing corrections from audits) is gone


def review_rule_id(r) -> str:
    return f"rv:{r.kind}:{r.subject}" + (f":{r.other}" if r.other else "")


def sidegrade_id(g) -> str:
    k = "|".join(sorted(g.a)) + "~" + "|".join(sorted(g.b))
    return "sg:" + hashlib.sha1(k.encode()).hexdigest()[:6]


def _e(id_, source, rule, effect, items, status="active", droppable=False, change="", fires=None) -> dict:
    return dict(id=id_, source=source, rule=rule, effect=effect, items=list(items), status=status,
                droppable=droppable, change=change, fires=fires)


def _names(r) -> List[str]:
    """Items a selection rule names: its item, then the items of its conditions."""
    out = [r.item]
    for c in r.conds:
        if c.kind == "item" and c.key not in out:
            out.append(c.key)
    return out


def collect(tri, item: Optional[str] = None, build=None) -> List[dict]:
    from tri.infer.defs import GLOBAL_RULE_KINDS
    from tri.rules.language import rule_id as rid
    from tri.rules.passes import NEVER
    dropped = tri.dropped_ids()
    eng = s = None
    sel: set = set()
    if build is not None and tri.space is not None:
        eng = tri.engine()
        sits = eng.situations(build, 1)
        s = sits[0] if sits else None
        sel = {x for x in build.items() if x in tri.catalog.index}
    on = (lambda names: True) if not item else (lambda names: item in names)
    out: List[dict] = []

    # selection rules: mined, authored, review evidence (as anchor too)
    rb = tri.rules
    if rb is not None:
        ev = getattr(tri.selection, "evidence", {}) or {}
        for r in rb.rules:
            names = _names(r)
            if not on(names):
                continue
            st = (ev.get(r.key) or {}).get("status", r.status)
            fires = None
            if s is not None and st in ("active", "review"):
                fires = eng._holds(r, sel - {r.item}, s) if r.effect == NEVER else eng._holds(r, sel, s)
            out.append(_e(rid(r.key), "selection", r.text(rb.region_names), "never" if r.effect == NEVER else "take",
                          names, st, True, "", fires))
    # review rules (induced)
    for r in getattr(tri, "review_rules_all", None) or tri.review_rules:
        names = [r.subject] + ([r.other] if r.other else [])
        if not on(names):
            continue
        i = review_rule_id(r)
        live = r.status in ("supported", "refined") and r.kind in GLOBAL_RULE_KINDS
        st = "dropped" if i in dropped else ("active" if live else f"inactive ({r.status}"
                                             + ("" if r.kind in GLOBAL_RULE_KINDS else ", local kind: acts through review claims") + ")")
        fires = None
        if s is not None and st == "active":
            fires = r.applies(s.build, s.feats) and (r.other is None or r.kind not in ("REDUNDANT", "CONFLICT", "PREFER")
                                                     or r.other in sel)
        out.append(_e(i, "review_rule", r.describe(), "never", names, st, True, "", fires))
    # sidegrades
    for g in getattr(tri, "sidegrades", None) or []:
        names = list(g.a) + list(g.b)
        if not on(names):
            continue
        i = sidegrade_id(g)
        out.append(_e(i, "sidegrade", "{" + ", ".join(g.a) + "} ⇄ {" + ", ".join(g.b) + "}", "blocks", names,
                      "dropped" if i in dropped else "active", True, "", None))
    # categories (admin)
    for it, c in sorted((getattr(tri, "manual", None) or {}).items()):
        if on([it]):
            out.append(_e(f"cat:{it}", "category", f"[{c[0].upper()}] {it}", "take" if c != "luxury" else "shapes", [it],
                          "active", False, "set its category to none (categories)",
                          None if build is None else (it in sel)))
    # weapon rules (hardcoded)
    from tri.game.weapon_rules import ITEMS, weapon_rules
    if build is not None:
        wr = weapon_rules(build, s.name if s is not None else None)
        for it, why in wr.forbid.items():
            if on([it]):
                out.append(_e(f"wp:{it}:never", "weapon", f"w ⇒ ¬{it}  ({why})", "never", [it], "active", False,
                              "hardcoded (tri/game/weapon_rules.py)", True))
        for it, why in wr.require.items():
            if on([it]):
                pre = tri.catalog.prereq_closure(it) if it in tri.catalog.index else []
                out.append(_e(f"wp:{it}:take", "weapon", f"w ⇒ {it}" + (f" ∧ {' ∧ '.join(pre)}" if pre else "")
                              + f"  ({why}); outranks reviews and rules", "take", [it] + pre, "active", False,
                              "hardcoded (tri/game/weapon_rules.py)", True))
    else:
        for it in ITEMS:
            if on([it]):
                out.append(_e(f"wp:{it}", "weapon", f"weapon rules on {it} (math §2.8c)", "take", [it], "active", False,
                              "hardcoded (tri/game/weapon_rules.py)", None))
    # game data
    c = tri.catalog
    its = [item] if item else []
    for it in its:
        I = c.items.get(it)
        if I is None:
            continue
        if I.prerequisites or I.prerequisites_any:
            txt = " ∧ ".join(I.prerequisites) + "".join(f" ∧ ({' ∨ '.join(g)})" for g in I.prerequisites_any)
            out.append(_e(f"gd:pre:{it}", "game", f"{it} ⇒ {txt.strip(' ∧')}", "blocks", [it] + list(I.prerequisites),
                          "active", False, "game data",
                          None if build is None else (it not in sel and not c.prereqs_met(it, sel))))
        for x in I.exclusive_with:
            out.append(_e(f"gd:excl:{it}:{x}", "game", f"{it} ⊥ {x}", "never", [it, x], "active", False, "game data",
                          None if build is None else x in sel))
        if c.voi_only(it):
            out.append(_e(f"gd:voi:{it}", "game", f"Pathfinder ⇒ ¬{it}", "never", [it], "active", False, "game data", None))
        if s is not None:
            t = c.index[it]
            out.append(_e(f"gd:stat:{it}", "game", f"stats(b) ⊨ req({it})", "blocks", [it], "active", False,
                          "game data", not bool(s.ctx.avail[t]) and it not in sel))
    if build is not None and eng is not None and s is not None:
        out.append(_e("gd:cap", "game", f"talents ≤ 76 − 2m = {eng.talent_cap(sel)}", "blocks", [], "active", False,
                      "game data", eng.cap_excess(sel) >= 0))
        if s.focus.get("classes"):
            out.append(_e("fx:focus", "focus", f"class focus {'/'.join(s.focus['classes'])}: other classes' weapon talents off",
                          "blocks", [], "active", False, "equip a weapon or answer the weapon question", True))
        # review claims that bind here (changed by editing or purging reviews)
        for it, v in s.memory.items():
            if not isinstance(it, str) or (item and it != item) or not eng._binds(v):
                continue
            rsn = f" ({v[4]}" + (f", {v[5]}" if len(v) > 5 and v[5] else "") + ")" if len(v) > 4 and v[4] else ""
            out.append(_e(f"cl:{it}", "claim", f"{v[0]} {it}{rsn}: " + eng._memory_why("stated", v),
                          "take" if v[0] in ("add", "keep") else "never", [it], "active", False,
                          "edit or purge the review", True))
    return out
