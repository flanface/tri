"""Hardcoded weapon rules (v3.2, math §2.8c).

The corpus records weapons too rarely to learn these, so they are stated from
the game data (tri/data/weapons.json: deepwoken.co bleed tags, pocamind weight
classes). w ranges over the build's main weapons (equipped, not offhand-only);
w0 is the first.

    SD   some w bleeds                          => never Speed Demon
         no w bleeds, archetype not Trackstar
         or Support                             => Speed Demon is required
         (Trackstar, Support: worth decides)
    FF   w in {Cursed Flames, Pleeksty's Inferno} => never Flaming Flourish
    TS   w0 light, w0 not in TS_LIGHT_OK        => never Turtle Shell
         w0 light, w0 in TS_LIGHT_OK            => Turtle Shell allowed (lock waived; worth decides)
         w0 medium or heavy, not in TS_HEAVY_NO => Turtle Shell is required (lock waived)

"never" removes the pick and makes it unavailable; "required" takes it at the
foundational pass whenever its stat requirements are met. Both hold over
reviews: a review claim that contradicts one is dropped. A build that records
no weapon gets none of these (the weapon question asks for it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from tri.game import weapons as W

SPEED_DEMON = "Speed Demon"
FLAMING_FLOURISH = "Flaming Flourish"
TURTLE_SHELL = "Turtle Shell"

FF_NO = frozenset({"Cursed Flames", "Pleeksty's Inferno"})
TS_LIGHT_OK = frozenset({"The Flippers of Fate", "Flareblood Kamas"})
TS_HEAVY_NO = frozenset({"Purple Cloud", "Wyrmtooth", "Railblade", "Duskpole", "Sancticar", "Imperial Staff"})
SD_WORTH = ("trackstar", "support")          # archetypes where worth decides Speed Demon

ITEMS = (SPEED_DEMON, FLAMING_FLOURISH, TURTLE_SHELL)


@dataclass
class WeaponRules:
    forbid: Dict[str, str] = field(default_factory=dict)      # item -> why
    require: Dict[str, str] = field(default_factory=dict)     # item -> why
    waive: Set[str] = field(default_factory=set)              # weapon locks waived

    def __bool__(self) -> bool:
        return bool(self.forbid or self.require or self.waive)


def main_weapons(b) -> List[str]:
    T = W.table()["weapons"]
    return [w for w in W.equipped(b) if not (T.get(w) or {}).get("only_offhand")]


def _sd_by_worth(archetype: Optional[str]) -> bool:
    a = str(archetype or "").strip().casefold()
    return any(a.startswith(k) for k in SD_WORTH)


def answered_class(b) -> Optional[str]:
    """The weapon class the asker named for a build that records no weapon
    (Build.meta "weapon_class", set by the answer; v3.6)."""
    return (getattr(b, "meta", None) or {}).get("weapon_class") or None


def class_rules(cls: str, archetype: Optional[str] = None) -> WeaponRules:
    """v3.6: the rules read from a weapon class alone (no named weapon). A rule
    holds when it holds for every weapon of the class: Fist, Dagger, Rapier and
    Pistol are light, so never Turtle Shell; a class of medium or heavy weapons
    requires it; a class with no bleeding weapon requires Speed Demon (worth
    decides for Trackstar and Support), a class of bleeding weapons forbids it."""
    T = W.table()["weapons"]
    ws = [v for v in T.values() if str(v.get("type") or "").casefold() == str(cls).casefold() and v.get("weight")]
    out = WeaponRules()
    if not ws:
        return out
    wts = {v["weight"] for v in ws}
    bleed = {bool(v.get("bleed")) for v in ws}
    if bleed == {True}:
        out.forbid[SPEED_DEMON] = f"{cls} weapons already bleed"
    elif bleed == {False} and not _sd_by_worth(archetype):
        out.require[SPEED_DEMON] = f"{cls} weapons have no innate bleed"
    if wts == {"Light"}:
        out.forbid[TURTLE_SHELL] = f"{cls} weapons are light"
    elif wts <= {"Medium", "Heavy"}:
        out.require[TURTLE_SHELL] = f"{cls} weapons are {'/'.join(sorted(w.lower() for w in wts))}"
        out.waive.add(TURTLE_SHELL)
    return out


def weapon_rules(b, archetype: Optional[str] = None) -> WeaponRules:
    """The rules for build b read as `archetype` (a family key or name). A build
    that records no weapon but whose class was answered gets class_rules (v3.6)."""
    ws = main_weapons(b)
    out = WeaponRules()
    if not ws:
        cls = answered_class(b)
        return class_rules(cls, archetype) if cls else out
    T = W.table()["weapons"]
    bleeding = [w for w in ws if (T.get(w) or {}).get("bleed")]
    if bleeding:
        out.forbid[SPEED_DEMON] = f"{bleeding[0]} already bleeds"
    elif not _sd_by_worth(archetype):
        out.require[SPEED_DEMON] = f"{ws[0]} has no innate bleed"
    ff = [w for w in ws if w in FF_NO]
    if ff:
        out.forbid[FLAMING_FLOURISH] = f"not with {ff[0]}"
    w0 = ws[0]
    wt = (T.get(w0) or {}).get("weight")
    if wt == "Light":
        if w0 in TS_LIGHT_OK:
            out.waive.add(TURTLE_SHELL)
        else:
            out.forbid[TURTLE_SHELL] = f"{w0} is a light weapon"
    elif wt in ("Medium", "Heavy") and w0 not in TS_HEAVY_NO:
        out.require[TURTLE_SHELL] = f"{w0} is a {wt.lower()} weapon"
        out.waive.add(TURTLE_SHELL)
    return out


def needs_weapon(b) -> bool:
    """True when the rules above cannot be read: the build records no main weapon
    and no weapon class was answered."""
    return not main_weapons(b) and not answered_class(b)
