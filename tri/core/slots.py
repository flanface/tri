"""Catalog methods: mantra slots and the talent cap."""
from __future__ import annotations

from typing import Dict, Optional

from tri.core.context import _empty
from tri.core.tables import (BASE_MANTRA_SLOTS, MANTRA_SLOT_TYPES, fit_mantra_slots, known_mantra_slot,
                             known_mantra_type, known_slot_grants)
from tri.core.types import Build, Kind


class _Slots:
    """Catalog methods (Catalog methods: mantra slots and the talent cap.)."""

    def mantra_type(self, item_id: str) -> Optional[str]:
        """Mantra type from the "type:<T>" tag (None if absent or not a mantra)."""
        it = self.items.get(item_id)
        if it is None or it.kind != Kind.MANTRA:
            return None
        for t in it.tags:
            if t.startswith("type:"):
                return t[5:]
        return known_mantra_type(item_id)

    def _slots(self):
        """(item -> mantra slot type, item -> slot grants), built once."""
        c = getattr(self, "_slot_cache", None)
        if c is None:
            mtype, grants = {}, {}
            for k, it in self.items.items():
                if it.kind == Kind.MANTRA:
                    t = next((x[9:] for x in it.tags if x.startswith("category:") and x[9:] in MANTRA_SLOT_TYPES), None)
                    mtype[k] = t or known_mantra_slot(k)
                else:
                    g = {}
                    for x in it.tags:
                        if x.startswith("mantra_slot:") and "=" in x:
                            a, _, b = x[12:].partition("=")
                            try:
                                g[a] = g.get(a, 0) + int(b)
                            except ValueError:
                                pass
                    g = g or known_slot_grants(k)
                    if g:
                        grants[k] = g
            c = self._slot_cache = (mtype, grants)
        return c

    def slots_typed(self) -> bool:
        """True when some mantra's slot type is known: then a mantra of unknown
        type fits only a Wildcard slot (fit_mantra_slots)."""
        return any(v is not None for v in self._slots()[0].values())

    def mantra_slot(self, item_id: str) -> Optional[str]:
        """Slot type of a mantra: Combat, Mobility, Support or Wisp (None if
        unknown; such a mantra only fits a Wildcard slot)."""
        return self._slots()[0].get(item_id)

    def slot_grants(self, item_id: str) -> Dict[str, int]:
        """Mantra slots a talent grants by type ({} if none)."""
        g = self._slots()[1].get(item_id)
        return dict(g) if g is not None else known_slot_grants(item_id)

    def mantra_slots_of(self, items, oath: Optional[str] = None) -> Dict[str, int]:
        """Mantra slots of a selection by type: the base slots, plus what its
        talents grant, plus the oath's ("Oath: <oath>"), counted once. Silentheart has none: the oath makes mantras unusable."""
        if oath and str(oath).strip().lower() == "silentheart":
            return {k: 0 for k in BASE_MANTRA_SLOTS}
        out = dict(BASE_MANTRA_SLOTS)
        items = set(items)
        names = {i for i in items if i in self._slots()[1]}
        if oath and not _empty(oath):
            names.add(f"Oath: {str(oath).strip()}")
        names |= {i for i in items if str(i).startswith("Oath:")}
        for n in names:
            for k, v in self.slot_grants(n).items():
                if k in out:
                    out[k] += int(v)
        return out

    def mantra_slot_usage(self, items, oath: Optional[str] = None) -> dict:
        """{slots: {type: n}, used: {type: n}, over: mantras that fit no slot,
        free: slots left}."""
        slots = self.mantra_slots_of(items, oath)
        ms = [i for i in set(items) if i in self.items and self.items[i].kind == Kind.MANTRA]
        used, over = fit_mantra_slots([self.mantra_slot(i) for i in ms], slots, self.slots_typed())
        need = dict(used)
        need["Wildcard"] = used.get("Wildcard", 0) + over      # v3.0.3: overflow is wildcard demand
        return dict(slots=slots, used=used, need=need, over=over,
                    free=sum(slots.values()) - sum(used.values()))

    def oath_of(self, b: "Build") -> Optional[str]:
        """The build's oath (metadata, or shown by its picks), None if none."""
        for t in self.build_context(b)[0]:
            f, _, v = str(t).partition(":")
            if f.strip().lower() == "oath" and v.strip():
                return v.strip()
        return None

    def mantras_fit(self, items, oath: Optional[str] = None) -> bool:
        """True if every mantra of the selection has a slot."""
        return self.mantra_slot_usage(items, oath)["over"] == 0

    def lowers_cap(self, item_id: str) -> bool:
        """True for mantras that lower the talent cap (by cap_per_mantra)."""
        it = self.items.get(item_id)
        return it is not None and it.kind == Kind.MANTRA and self.mantra_type(item_id) not in self.cap_exempt

    def talent_cap(self, items) -> int:
        """76 - 2m for a selection (m = non-exempt mantras in it)."""
        return self.talent_cap_base - self.cap_per_mantra * sum(1 for i in set(items) if self.lowers_cap(i))

    def n_talents(self, items) -> int:
        """Talents in a selection (ids not in the catalog are ignored)."""
        return sum(1 for i in set(items) if i in self.items and self.items[i].kind == Kind.TALENT)

    def cap_excess(self, items) -> int:
        """Talents above the cap (<= 0 means the selection obeys the cap)."""
        return self.n_talents(items) - self.talent_cap(items)
