"""Bundled game tables: mantra types and slots, Voi-only items, context locks."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tri.data import path as _data_path

_TYPE_TABLE: Optional[Dict[str, str]] = None


def _loose(name: str) -> str:
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKC", str(name)).casefold() if ch.isalnum())


def known_mantra_type(name: str) -> Optional[str]:
    """Type of a mantra from the bundled table tri/data/mantra_types.json (Oath,
    Monster, Origin, Event; None = Normal or unknown). Fallback for catalogs
    written before mantra types were stored as "type:<T>" tags."""
    global _TYPE_TABLE
    if _TYPE_TABLE is None:
        p = _data_path("mantra_types.json")
        try:
            with open(p) as f:
                _TYPE_TABLE = {_loose(k): v for k, v in json.load(f)["types"].items()}
        except (OSError, ValueError, KeyError):
            _TYPE_TABLE = {}
    return _TYPE_TABLE.get(_loose(name))


#: Mantra slot types. A mantra takes a slot of its own type; the Wildcard slot
#: takes any mantra; a Wisp mantra also fits a Support slot.
MANTRA_SLOT_TYPES: Tuple[str, ...] = ("Combat", "Mobility", "Support", "Wisp", "Wildcard")
#: Slots every build starts with (Deepwoken wiki, Mantras).
BASE_MANTRA_SLOTS: Dict[str, int] = {"Combat": 3, "Mobility": 1, "Support": 1, "Wisp": 0, "Wildcard": 1}
_SLOT_TABLE: Optional[dict] = None


def _slot_table() -> dict:
    global _SLOT_TABLE
    if _SLOT_TABLE is None:
        p = _data_path("mantra_slots.json")
        try:
            with open(p) as f:
                raw = json.load(f)
            _SLOT_TABLE = dict(grants={_loose(k): v for k, v in raw.get("grants", {}).items()},
                               mantra_slot={_loose(k): v for k, v in raw.get("mantra_slot", {}).items()})
        except (OSError, ValueError):
            _SLOT_TABLE = dict(grants={}, mantra_slot={})
    return _SLOT_TABLE


def known_slot_grants(name: str) -> Dict[str, int]:
    """Mantra slots a talent grants by type, from the bundled table
    tri/data/mantra_slots.json ({} if none). Covers the oath talents
    ("Oath: Arcwarder": Combat 2, Wildcard 1), which builds record as metadata."""
    return dict(_slot_table()["grants"].get(_loose(name), {}))


def known_mantra_slot(name: str) -> Optional[str]:
    """Slot type of a mantra (Combat, Mobility, Support, Wisp) from the bundled table."""
    return _slot_table()["mantra_slot"].get(_loose(name))


def fit_mantra_slots(types: Sequence[Optional[str]], slots: Dict[str, int],
                     typed: Optional[bool] = None) -> Tuple[Dict[str, int], int]:
    """Place mantras of these slot types into slots: each takes a slot of its
    type, a Wisp mantra a Wisp slot and then a Support slot, the rest take
    Wildcard slots. A mantra of unknown type (None) takes only a Wildcard slot
    when the catalog knows slot types (typed, default: some type is known), and
    any slot left in a catalog without slot types. So a Support slot never
    holds a Combat mantra. This order is optimal (math §2.6).
    Returns (used slots by type, mantras that do not fit)."""
    if typed is None:
        typed = any(t is not None for t in types)
    used = {k: 0 for k in MANTRA_SLOT_TYPES}
    rest = wisps = unknown = 0
    for t in types:
        if t == "Wisp":
            wisps += 1
        elif t is None:
            if typed:
                rest += 1
            else:
                unknown += 1
        elif t in ("Combat", "Mobility", "Support") and used[t] < slots.get(t, 0):
            used[t] += 1
        else:
            rest += 1
    for k in ("Wisp", "Support"):
        n = min(wisps, slots.get(k, 0) - used[k])
        used[k] += n; wisps -= n
    rest += wisps
    n = min(rest, slots.get("Wildcard", 0))
    used["Wildcard"] = n
    over = rest - n
    for k in MANTRA_SLOT_TYPES:
        n = min(unknown, slots.get(k, 0) - used[k])
        used[k] += n; unknown -= n
    return used, over + unknown


_LOCK_TABLE: Optional[Dict[str, List[List[str]]]] = None
_VOI_TABLE: Optional[Set[str]] = None


def known_voi_only(name: str) -> bool:
    """True for talents and mantras only obtainable in Vow of Iron, from the
    bundled table tri/data/voi_only.json (pocamind/data "voi_only"). Tags a
    saved catalog written without the "voi_only" tag (Catalog.from_dict)."""
    global _VOI_TABLE
    if _VOI_TABLE is None:
        p = _data_path("voi_only.json")
        try:
            with open(p) as f:
                d = json.load(f)
            _VOI_TABLE = {_loose(x) for k in ("talents", "mantras") for x in d.get(k, [])}
        except (OSError, ValueError):
            _VOI_TABLE = set()
    return _loose(name) in _VOI_TABLE


def known_context(name: str) -> List[List[str]]:
    """Context clauses of an item from the bundled table tri/data/context_locks.json
    (oath / murmur / origin locks from the game data). Filled in when a saved
    catalog states none for the item, so a catalog written without locks still
    keeps Jet Kick to Jetstriker builds. [] if unknown."""
    global _LOCK_TABLE
    if _LOCK_TABLE is None:
        p = _data_path("context_locks.json")
        try:
            with open(p) as f:
                _LOCK_TABLE = {_loose(k): v for k, v in json.load(f)["locks"].items()}
        except (OSError, ValueError, KeyError):
            _LOCK_TABLE = {}
    return [list(c) for c in _LOCK_TABLE.get(_loose(name), [])]
