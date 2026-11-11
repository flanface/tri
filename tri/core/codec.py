"""Validation and JSON forms of builds and reviews, plus small text helpers."""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Sequence

from tri.core.catalog import Catalog
from tri.core.tables import MANTRA_SLOT_TYPES
from tri.core.types import (ADD_REASONS, IFFY_REASONS, RELATIONAL_REASONS, REMOVE_ONLY_REASONS, RETIRED_ACTIONS,
                            Action, Build, Kind, Operation, Reason, Review, ReviewError, Strength)


def validate_review(r: Review, catalog: Catalog, strict_capacity: bool = True) -> List[str]:
    """Return a list of warnings; raise ReviewError for contract violations."""
    warn: List[str] = []
    orig = r.original.item_set()
    for o in r.operations:
        if o.item_id not in catalog.items:
            raise ReviewError(f"unknown item {o.item_id}")
        if o.action == Action.REMOVE and o.item_id not in orig:
            raise ReviewError(f"remove of absent item {o.item_id}")
        if o.action == Action.ADD and o.item_id in orig:
            raise ReviewError(f"add of present item {o.item_id}")
        if o.action != Action.IFFY and o.reason is None:
            raise ReviewError(f"{o.action.value} {o.item_id} missing reason")
        if o.reason in RELATIONAL_REASONS and not o.related:
            raise ReviewError(f"{o.reason.value} on {o.item_id} needs related item(s)")
        if o.reason in IFFY_REASONS and o.action != Action.IFFY:
            raise ReviewError(f"{o.reason.value} is a reason for iffy marks ({o.item_id})")
        if o.action == Action.IFFY and o.reason is not None and o.reason not in IFFY_REASONS:
            raise ReviewError(f"iffy {o.item_id}: the only iffy reason is sidegrade")
        if o.reason in REMOVE_ONLY_REASONS and o.action != Action.REMOVE:
            raise ReviewError(f"{o.reason.value} is a reason for removes ({o.item_id})")
        if o.action == Action.REMOVE and o.reason in (Reason.SYNERGY_WITH, Reason.LUXURY_MORE):
            raise ReviewError(f"{o.reason.value} is a reason for adds, not removes ({o.item_id})")
        for y in o.related:
            if isinstance(y, str) and y.startswith("weapon:"):       # v3.3: a weapon scopes the claim
                from tri.game.weapons import weapon_token
                if weapon_token(y) != y:
                    raise ReviewError(f"unknown weapon {y[7:]} on {o.item_id}")
                if o.reason in (Reason.NEEDS, Reason.SIDEGRADE):
                    raise ReviewError(f"{o.reason.value} on {o.item_id} names items, not a weapon ({y[7:]})")
                continue
            if o.reason in (Reason.NEEDS, Reason.SIDEGRADE) and y not in catalog.items:
                raise ReviewError(f"unknown related item {y}")
            if y == o.item_id:
                raise ReviewError(f"{o.item_id} cannot be related to itself")
        try:
            o.strength = Strength(o.strength)
        except ValueError:
            raise ReviewError(f"strength must be 'weak' or 'strong' on {o.item_id}")
        if o.reason == Reason.GENERALLY_GOOD:
            if o.action != Action.ADD:
                raise ReviewError(f"generally good is a reason for adds, not removes ({o.item_id})")
            if catalog.items[o.item_id].kind != Kind.TALENT:
                raise ReviewError(f"generally good applies to talents; {o.item_id} is a mantra")
        if o.action == Action.ADD and o.reason not in (None,) + tuple(ADD_REASONS):
            warn.append(f"unusual reason {o.reason.value} on an ADD of {o.item_id}")
    res = r.apply(catalog)
    final = set(res.items())
    for o in r.operations:                               # NW: the missing components stay missing
        if o.reason == Reason.NEEDS and o.action == Action.REMOVE:
            have = [y for y in o.related if y in final]
            if len(have) == len(o.related):
                raise ReviewError(f"{o.item_id} is removed for lacking {', '.join(o.related)}, "
                                  "which the reviewed build has")
    for n in r.notes or []:
        if n.get("kind") in ("archetype", "core", "avoid"):
            bad = [y for y in n.get("items") or [] if y not in catalog.items and ":" not in str(y)]
            if bad:
                raise ReviewError(f"note: unknown component {', '.join(map(str, bad))}")
            if n.get("kind") != "archetype" and not n.get("items"):
                raise ReviewError(f"note: {n['kind']} needs items")
        if n.get("kind") == "avoid":
            held = [y for y in n.get("items") or [] if y in final]
            if held:
                raise ReviewError(f"note: avoid {', '.join(held)}, which the reviewed build keeps")
        if n.get("kind") == "weapon":                    # v3.3: # weapon: A, B (equips by name)
            from tri.game.weapons import resolve
            bad = [w for w in n.get("weapons") or [] if not resolve(w)]
            if bad or not n.get("weapons"):
                raise ReviewError(f"note: unknown weapon {', '.join(map(str, bad))}" if bad else "note: weapon needs names")
        if n.get("kind") in ("is", "not") and not str(n.get("archetype") or "").strip():
            raise ReviewError(f"note: {n['kind']} needs an archetype")
    for o in r.added():                                  # mutual exclusion
        bl = catalog.exclusive_in(o.item_id, final)
        if bl:
            dep = catalog.dependents(bl, set(r.original.items()) | final)
            raise ReviewError(exclusion_message(o.item_id, bl, [d for d in dep if d in final]))
    kept = final & orig                                  # a removal strands what requires it
    for z in sorted(kept):
        if catalog.prereqs_met(z, orig | final) and not catalog.prereqs_met(z, final):
            raise ReviewError(f"{z} requires {prereq_text(catalog, z)}, which the review removes: remove {z} too")
    for o in r.added():
        if not catalog.prereqs_met(o.item_id, final):
            warn.append(f"added {o.item_id} requires {prereq_text(catalog, o.item_id)}, not in the reviewed build")
    for o in r.added():
        if catalog.voi_only(o.item_id):
            raise ReviewError(f"{o.item_id} is only obtainable in Vow of Iron, not in a Pathfinder build")
    for o in r.added():
        if not catalog.available_in(o.item_id, r.original):
            warn.append(f"added {o.item_id} appears unavailable at this build's pre-shrine and final stats")
    seen = set()
    for o in r.operations:
        if o.item_id in seen:
            raise ReviewError(f"two operations on {o.item_id}")
        seen.add(o.item_id)
    ex = catalog.cap_excess(res.talents + res.mantras)
    if ex > 0:
        raise ReviewError(f"reviewed build has {catalog.n_talents(res.talents)} talents, over the talent cap "
                          f"of {catalog.talent_cap(res.talents + res.mantras)} ({catalog.talent_cap_base} - "
                          f"{catalog.cap_per_mantra} x non-exempt mantras)")
    oath = catalog.oath_of(r.original)
    u, u0 = catalog.mantra_slot_usage(res.items(), oath), catalog.mantra_slot_usage(r.original.items(), oath)
    lost = sorted(x for x in orig - set(res.items()) if catalog.slot_grants(x))
    if u["over"] > 0 and (u["over"] > u0["over"] or any(catalog.kind_of(o.item_id) == Kind.MANTRA
                                                        for o in r.added() if o.item_id in catalog.items)):
        raise ReviewError("reviewed build is over its mantra slots (" + ", ".join(
            f"{k} {u['need'][k]}/{u['slots'][k]}" for k in MANTRA_SLOT_TYPES if u["slots"][k] or u["need"][k]) +
            f"; {u['over']} mantra(s) fit no slot)" + (
                f": removing {', '.join(x for x in lost)} takes away mantra slots" if lost else "") +
            ". A mantra takes a slot of its type, a Wisp mantra a Wisp or Support slot, and any mantra a Wildcard slot")
    if strict_capacity:
        dt = len(res.talents) - len(r.original.talents)
        if dt > 0:
            warn.append(f"talent count grew by {dt}; original was declared full")
    return warn


def build_to_dict(b: Build) -> dict:
    """Build -> plain dict (JSON-serialisable)."""
    return asdict(b)


def build_from_dict(d: dict) -> Build:
    """Plain dict -> Build (unknown keys are ignored)."""
    import dataclasses
    keys = {f.name for f in dataclasses.fields(Build)}
    return Build(**{k: v for k, v in d.items() if k in keys})


def review_to_dict(r: Review) -> dict:
    """Review -> plain dict; enums are stored as their string values."""
    d = asdict(r)
    for o in d["operations"]:
        o["action"] = o["action"].value if hasattr(o["action"], "value") else o["action"]
        o["strength"] = o["strength"].value if hasattr(o["strength"], "value") else o["strength"]
        if o["reason"] is not None and hasattr(o["reason"], "value"):
            o["reason"] = o["reason"].value
    return d


def prereq_text(catalog: Catalog, item: str) -> str:
    """Prerequisites of an item as text: "A and B", groups as "C or D"."""
    it = catalog.items[item]
    parts = list(it.prerequisites) + [" or ".join(g) for g in it.prerequisites_any]
    return " and ".join(parts) or "nothing"


def exclusion_message(item: str, blockers: Sequence[str], dependents: Sequence[str]) -> str:
    """"y and x are mutually exclusive: remove x (and z, w, which require it) to add y"."""
    b = ", ".join(blockers)
    msg = f"{item} and {b} are mutually exclusive: remove {b}"
    if dependents:
        msg += f" and {', '.join(dependents)} ({'requires' if len(dependents) == 1 else 'require'} it)"
    return msg + f" to add {item}"


def is_silentheart(b: "Build", tokens: Iterable[str] = ()) -> bool:
    """True for builds under the Silentheart oath (their mantras are unusable)."""
    meta = {str(k).lower(): v for k, v in (b.meta or {}).items()}
    vals = [meta.get("oath")] + list(tokens) + list(b.talents)
    return any("silentheart" in str(v).lower() for v in vals if v)


def legacy_names(x):
    """Read data written before v2.5, when archetypes were called purposes:
    dict keys and string values containing 'purpose' get 'archetype'
    (e.g. reason necessary_for_purpose, locator field purpose,
    select_min_per_purpose). Free text such as notes is left alone."""
    if isinstance(x, dict):
        return {(k.replace("purpose", "archetype") if isinstance(k, str) else k):
                (v if k in ("note", "notes", "text", "link", "build_name", "name", "purpose", "archetype") else legacy_names(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [legacy_names(v) for v in x]
    if isinstance(x, str) and "purpose" in x and x.replace("_", "").isalpha():
        return x.replace("purpose", "archetype")
    return x


def review_from_dict(d: dict) -> Review:
    """Plain dict -> Review. Missing strength defaults to 'strong'. Reads
    reviews written before v2.5 (reason necessary_for_purpose). v3.5: an audit
    (scope "audit", or only overturn / uphold verdicts) raises ReviewError; a
    stray verdict inside a full review is dropped."""
    d = legacy_names(d)
    raw = list(d["operations"])
    if d.get("scope") == "audit" or (raw and all(o.get("action") in RETIRED_ACTIONS for o in raw)):
        raise ReviewError(f"{d.get('review_id')}: audits were removed in v3.5")
    ops = [Operation(action=Action(o["action"]), item_id=o["item_id"],
                     strength=Strength(o.get("strength", "strong")),
                     reason=Reason(o["reason"]) if o.get("reason") else None,
                     related=o.get("related", []), group=o.get("group"), note=o.get("note"))
           for o in raw if o.get("action") not in RETIRED_ACTIONS]
    return Review(review_id=d["review_id"], original=build_from_dict(d["original"]),
                  operations=ops, reviewer_id=d["reviewer_id"],
                  ruleset=d.get("ruleset", "unknown"), timestamp=d.get("timestamp"),
                  scope="full",
                  notes=[dict(n) for n in d.get("notes") or [] if isinstance(n, dict)],
                  neutral=[str(x) for x in d.get("neutral") or []])
