"""Finer archetypes: families, variants and build styles (v2.18, math §6.11).

An archetype name may carry a variant after a slash: "Gank / Frost Mage" is the
variant "Frost Mage" of the family "Gank". Every reading that existed before
(the kernel posterior, the twin split, indicators, the review barrier) runs on
families, so splitting a family into variants never changes how a build reads
at the family level. Within the family the variant is read by the same kernel,
restricted to the family's variant locators (and "is" anchors naming a
variant), with an equal prior per variant.

Independently of any label, every build gets a style from its stats: the
attunements it invests in, its weapon class and whether it leans on magic,
weapons or both ("Frost mage", "Flame / Thunder hybrid, heavy weapon"). The
style needs no locator and no review, so it is always there; it describes, it
never changes a rating.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEP = "/"

ATTUNEMENTS = {"Flamecharm": "Flame", "Frostdraw": "Frost", "Thundercall": "Thunder",
               "Galebreathe": "Wind", "Shadowcast": "Shadow", "Ironsing": "Iron", "Bloodrend": "Blood"}
WEAPONS = {"HeavyWeapon": "heavy", "MediumWeapon": "medium", "LightWeapon": "light"}
BASES = {"Strength": "strength", "Fortitude": "fortitude", "Agility": "agility",
         "Intelligence": "intelligence", "Willpower": "willpower", "Charisma": "charisma"}

STYLE_MIN = 25          # a stat counts toward the style from this many points
STYLE_SHARE = 0.5       # ...and only if it is at least half the largest of its group
MAGE_RATIO = 1.5        # attunement total / weapon total for "mage" (and the inverse for "warrior")


def _norm(name: str) -> str:
    return " ".join(str(name).split())


def split_archetype(name: str) -> Tuple[str, Optional[str]]:
    """'Gank / Frost Mage' -> ('Gank', 'Frost Mage'); 'Gank' -> ('Gank', None)."""
    s = _norm(name)
    if SEP not in s:
        return s, None
    fam, var = s.split(SEP, 1)
    fam, var = _norm(fam), _norm(var.replace(SEP, " "))
    return (fam, var or None) if fam else (var, None)


def join_archetype(family: str, variant: Optional[str]) -> str:
    return f"{family} {SEP} {variant}" if variant else family


def family_key(name: str) -> str:
    """Comparison key of the family of an archetype name."""
    return split_archetype(name)[0].casefold()


def variant_key(name: str) -> Optional[str]:
    v = split_archetype(name)[1]
    return v.casefold() if v else None


# ---------------------------------------------------------------- styles
def style_of(build, weapons: Sequence[str] = ()) -> dict:
    """The style of a build from its final stats (highest if final is empty):
    {elements, weapon, role, focus, label}. weapons: classes a review note gave
    (they override the stat reading of the weapon class)."""
    st = dict(getattr(build, "final_stats", None) or {}) or dict(getattr(build, "highest_stats", None) or {})
    att = {ATTUNEMENTS[k]: int(st.get(k, 0) or 0) for k in ATTUNEMENTS}
    wep = {WEAPONS[k]: int(st.get(k, 0) or 0) for k in WEAPONS}
    base = {BASES[k]: int(st.get(k, 0) or 0) for k in BASES}

    def top(group: Dict[str, int]) -> List[str]:
        mx = max(group.values()) if group else 0
        if mx < STYLE_MIN:
            return []
        return [k for k, v in sorted(group.items(), key=lambda x: -x[1]) if v >= max(STYLE_MIN, STYLE_SHARE * mx)]

    elements = top(att)[:2]
    wcls = [w.lower() for w in weapons] if weapons else top(wep)[:1]
    A, W = sum(att.values()), sum(wep.values())
    if A >= STYLE_MIN and A >= MAGE_RATIO * max(W, 1):
        role = "mage"
    elif W >= STYLE_MIN and W >= MAGE_RATIO * max(A, 1):
        role = "warrior"
    elif A >= STYLE_MIN and W >= STYLE_MIN:
        role = "hybrid"
    else:
        role = "generalist"
    focus = top(base)[:1]
    head = " / ".join(elements)
    if role == "warrior":
        label = f"{wcls[0].title()} warrior" if wcls else "Warrior"
        if head:
            label += f", {head.lower()}"
    else:
        label = f"{head} {role}".strip() if head else role.title()
        if wcls:
            label += f", {wcls[0]} weapon"
    label = label[0].upper() + label[1:] if label else label
    return dict(elements=elements, weapon=wcls[0] if wcls else None, role=role,
                focus=focus[0] if focus else None, label=label)


# ---------------------------------------------------------------- variant reading
def read_variant(ls, u: np.ndarray, family: str) -> Optional[dict]:
    """The variant of `family` at u: kernel posterior over the family's variant
    labels (locators named "family / variant" and "is" anchors naming one), equal
    prior per variant. None when the family has no variants. located: the nearest
    variant point is within reach and the top variant has min_confidence."""
    if ls is None or not getattr(ls, "locators", None):
        return None
    fk = family_key(family)
    pts: List[np.ndarray] = []
    labs: List[str] = []
    names: Dict[str, str] = {}
    for l, uu in zip(ls.locators, ls.U):
        if family_key(l.archetype) != fk:
            continue
        v = split_archetype(l.archetype)[1]
        if v:
            pts.append(np.asarray(uu, float)); labs.append(v.casefold()); names[v.casefold()] = v
    for a in getattr(ls, "anchors", []) or []:
        v = split_archetype(str(a.get("variant") or ""))[1]
        if v and a.get("key") == fk and float(a.get("sign", 1)) > 0:
            pts.append(np.asarray(a["u"], float)); labs.append(v.casefold()); names.setdefault(v.casefold(), v)
    if not pts:
        return None
    P = np.vstack(pts)
    u = np.asarray(u, float).ravel()
    d2 = np.maximum(((P - u[None]) ** 2).sum(1), 0.0)
    L = -d2 / (2 * ls.h ** 2)
    keys = sorted(set(labs))
    lab = np.array([keys.index(x) for x in labs])
    lg = np.full(len(keys), -np.inf)
    for i in range(len(keys)):
        m = lab == i
        mx = L[m].max()
        lg[i] = mx + np.log(np.exp(L[m] - mx).sum()) - np.log(m.sum())
    p = np.exp(lg - lg.max()); p /= p.sum()
    k = int(p.argmax())
    dmin = float(np.sqrt(d2.min()))
    located = dmin <= ls.reach and p[k] >= ls.cfg.min_confidence
    return dict(variant=names[keys[k]] if located else None, best=names[keys[k]], confidence=round(float(p[k]), 3),
                posterior={names[keys[i]]: round(float(p[i]), 3) for i in np.argsort(-p) if p[i] > 1e-4},
                distance=round(dmin, 3), located=bool(located))


def full_name(reading: dict) -> Optional[str]:
    """'Gank / Frost Mage' when a variant is read, the family otherwise."""
    fam = reading.get("archetype") or reading.get("best")
    if not fam:
        return None
    v = (reading.get("variant") or {}).get("variant")
    return join_archetype(fam, v)


_SPACES = re.compile(r"\s*/\s*")


def same_name(a: str, b: str) -> bool:
    return _SPACES.sub(" / ", _norm(a)).casefold() == _SPACES.sub(" / ", _norm(b)).casefold()
