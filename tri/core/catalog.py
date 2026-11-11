"""The catalog: every talent and mantra of one ruleset, with requirements and exclusions.

Methods are split by concern into availability.py, picks.py and slots.py."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tri.core.availability import _Availability
from tri.core.context import CONTEXT_LOCK_FIELDS, MANTRA_LOCKS
from tri.core.picks import _Picks
from tri.core.slots import _Slots
from tri.core.tables import known_context, known_voi_only
from tri.core.types import (CAP_EXEMPT_TYPES, DEFAULT_STATS, TALENT_CAP_BASE, TALENT_CAP_PER_MANTRA, Category,
                            Item, Kind)


@dataclass
class Catalog(_Availability, _Picks, _Slots):
    """All items for one ruleset. `ids` is sorted and defines tensor order.

    Talent cap: talents in a build may not exceed
    talent_cap_base - cap_per_mantra * m (76 - 2m), m = number of mantras whose
    type (tag "type:<T>") is not in cap_exempt (Oath, Origin = faction, Monster).
    Mantras with no type tag count towards m."""
    items: Dict[str, Item]
    stats: Tuple[str, ...] = DEFAULT_STATS
    ruleset: str = "unknown"
    talent_cap_base: int = TALENT_CAP_BASE
    cap_exempt: Tuple[str, ...] = CAP_EXEMPT_TYPES
    cap_per_mantra: int = TALENT_CAP_PER_MANTRA
    mantra_locks: Tuple[str, ...] = MANTRA_LOCKS   # context tokens under which no mantra is available
    lock_fields: Tuple[str, ...] = CONTEXT_LOCK_FIELDS   # context fields whose clauses are enforced

    def __post_init__(self) -> None:
        self.ids: List[str] = sorted(self.items)
        self.index: Dict[str, int] = {k: i for i, k in enumerate(self.ids)}
        self.stat_index: Dict[str, int] = {s: i for i, s in enumerate(self.stats)}

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def from_json(path: str) -> "Catalog":
        """Load a catalog file: {stats?, ruleset?, items: [{id, kind, category?,
        requires?, requires_any?, clauses?, prerequisites?, prerequisites_any?,
        exclusive_with?, tags?, context?}], mantra_locks?, lock_fields?}.
        """
        with open(path) as f:
            return Catalog.from_dict(json.load(f))

    def to_dict(self) -> dict:
        """The JSON form from_json / from_dict read (pipeline.catalog_to_json writes it)."""
        items = []
        for iid in self.ids:
            it = self.items[iid]
            d = dict(id=it.id, kind=it.kind.value, category=it.category.value)
            for k in ("requires", "requires_any", "clauses", "prerequisites", "prerequisites_any",
                      "exclusive_with", "tags", "context"):
                v = getattr(it, k)
                if v:
                    d[k] = v
            items.append(d)
        return dict(ruleset=self.ruleset, stats=list(self.stats),
                    talent_cap=dict(base=self.talent_cap_base, exempt=list(self.cap_exempt),
                                    per_mantra=self.cap_per_mantra),
                    mantra_locks=list(self.mantra_locks),
                    lock_fields=list(self.lock_fields), items=items)

    @staticmethod
    def from_dict(raw: dict) -> "Catalog":
        """Catalog.from_json of an already-parsed dict."""
        items = {}
        for r in raw["items"]:
            items[r["id"]] = Item(
                id=r["id"], kind=Kind(r["kind"]),
                category=Category(r.get("category", "unknown")),
                requires=r.get("requires", {}), requires_any=r.get("requires_any", []),
                clauses=r.get("clauses", []),
                prerequisites=r.get("prerequisites", []),
                prerequisites_any=r.get("prerequisites_any", []),
                exclusive_with=r.get("exclusive_with", []), tags=list(r.get("tags", [])),
                context=r.get("context", []) or known_context(r["id"]),
            )
            if "voi_only" not in items[r["id"]].tags and known_voi_only(r["id"]):
                items[r["id"]].tags.append("voi_only")
        tc = raw.get("talent_cap") or {}
        return Catalog(items=items, stats=tuple(raw.get("stats", DEFAULT_STATS)),
                       ruleset=raw.get("ruleset", "unknown"),
                       talent_cap_base=int(tc.get("base", TALENT_CAP_BASE)),
                       cap_exempt=tuple(tc.get("exempt", CAP_EXEMPT_TYPES)),
                       cap_per_mantra=int(tc.get("per_mantra", TALENT_CAP_PER_MANTRA)),
                       mantra_locks=tuple(raw.get("mantra_locks", MANTRA_LOCKS)),
                       lock_fields=tuple(raw.get("lock_fields", CONTEXT_LOCK_FIELDS)))
