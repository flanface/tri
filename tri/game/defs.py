"""Definitions of game data: game items, requirement parse errors and patches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tri.core.types import Kind


class ReqParseError(ValueError):
    """A requirement string that does not follow the grammar."""


@dataclass
class GameItem:
    """One talent or mantra from the game data."""
    name: str
    key: str                      # file stem, used by prerequisite references
    kind: Kind
    rarity: Optional[str] = None
    category: Optional[str] = None
    slot: bool = True             # takes a talent slot (talents); mantras: True
    implicit: bool = False
    vaulted: bool = False
    reqs: str = "()"
    clauses: List[List[dict]] = field(default_factory=list)
    prereqs: List[List[str]] = field(default_factory=list)   # groups of "ns:key"; one per group needed
    exclusive: List[str] = field(default_factory=list)       # names
    conditions: Dict[str, object] = field(default_factory=dict)  # requirements a build cannot show
    source: str = "pocamind"                                  # which data set it came from
    mtype: Optional[str] = None   # mantras: Normal | Oath | Monster | Origin | Event (game data "type")
    slot_grants: Dict[str, int] = field(default_factory=dict)   # talents: mantra slots granted by type
    voi_only: bool = False        # only obtainable in Vow of Iron (pocamind "voi_only"): never available

    @property
    def mantra_slots(self) -> int:
        """Total mantra slots granted."""
        return sum(self.slot_grants.values())

    @property
    def granted(self) -> bool:
        """True for talents that take no slot (never True for mantras)."""
        return self.kind == Kind.TALENT and (not self.slot or self.implicit)


@dataclass
class Patch:
    name: str
    time: float                                   # unix seconds the patch went live
    changed: List[str] = field(default_factory=list)   # items whose value changed
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    renamed: Dict[str, str] = field(default_factory=dict)   # old id -> new id
    note: str = ""

    def touched(self) -> List[str]:
        return sorted(set(self.changed) | set(self.added) | set(self.renamed.values()))
