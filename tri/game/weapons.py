"""Forced weapon context (v2.21, math §2.8).

The planner records what a build equips (Build.meta "weapon", the deepwoken.co
"weapons" field). Corpus builds rarely act on it, so the pretrain has no sense
that Turtle Shell needs a two-handed main weapon (its shield sits on the back;
the shield itself is left to the Shieldmaster prerequisites, since planners
seldom list it), or that Speed
Demon's bleed is redundant on a weapon that already bleeds. tri states these
as context clauses in the field "wield", read from the bundled table
tri/data/weapons.json (game data, never statistics):

  - weapon_tokens(w): what one equipped weapon shows: "wield:<name>",
    "wield:<type>", "wield:<weight> Weapon", "wield:one-handed" or
    "wield:two-handed", "wield:bleed" or "wield:no bleed".
  - item_clauses(item): [[token, ...], ...]; the item is available only when
    every clause has a token some equipped weapon shows (any weapon, so a build
    that carries a second weapon is never locked out of what that weapon uses).
  - scale(item, weapons): the share of equipped weapons that leave the item
    useful (Speed Demon: weapons without bleed); worth is multiplied by it.

A build that records no weapon, or none tri knows, records no "wield" field:
no weapon clause is checked and nothing is scaled.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional, Sequence, Set

from tri.data import path as _data_path

FIELD = "wield"
_SPLIT = re.compile(r"\s*(?:,|/|;|\+|\||&|\band\b)\s*")


def _loose(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


@lru_cache(maxsize=1)
def table() -> dict:
    p = _data_path("weapons.json")
    try:
        with open(p) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        raw = {}
    ws = raw.get("weapons", {})
    return dict(weapons=ws, by_loose={_loose(k): k for k in ws},
                types={_loose(v["type"]): v["type"] for v in ws.values()},
                clauses={_loose(k): v for k, v in raw.get("clauses", {}).items()},
                scale={_loose(k): v for k, v in raw.get("scale", {}).items()})


def resolve(value) -> List[str]:
    """Weapon names from a Build.meta value (str or list; "A, B" or "A / B"
    lists several), each matched to the table. Unknown names are dropped."""
    vals = [value] if isinstance(value, str) else list(value or [])
    out: List[str] = []
    T = table()
    for v in vals:
        s = str(v).strip()
        k = T["by_loose"].get(_loose(s))
        parts = [s] if k else [p for p in _SPLIT.split(s) if p]
        for p in parts:
            k = T["by_loose"].get(_loose(p))
            if k and k not in out:
                out.append(k)
    return out


def equipped(b) -> List[str]:
    """Weapons a build equips, from Build.meta "weapon" / "weapons"."""
    m = {str(k).lower(): v for k, v in (getattr(b, "meta", None) or {}).items()}
    out: List[str] = []
    for k in ("weapon", "weapons"):
        for w in resolve(m.get(k)):
            if w not in out:
                out.append(w)
    return out


def weapon_tokens(w: str) -> Set[str]:
    rec = table()["weapons"].get(w)
    if rec is None:
        return set()
    t = rec["type"]
    out = {f"{FIELD}:{w}", f"{FIELD}:{t}", f"{FIELD}:{t}s" if not t.endswith("s") else f"{FIELD}:{t[:-1]}",
           f"{FIELD}:two-handed" if rec.get("hands") == 2 else f"{FIELD}:one-handed",
           f"{FIELD}:bleed" if rec.get("bleed") else f"{FIELD}:no bleed"}
    if rec.get("weight"):
        out.add(f"{FIELD}:{rec['weight']} Weapon")
    return out


def tokens_of(weapons: Sequence[str]) -> List[str]:
    out: Set[str] = set()
    for w in weapons:
        out |= weapon_tokens(w)
    return sorted(out)


def item_clauses(item: str) -> List[List[str]]:
    return [list(c) for c in table()["clauses"].get(_loose(item), [])]


def scale(item: str, weapons: Sequence[str]) -> float:
    """Share of the equipped weapons that leave the item useful (1 when the
    item has no scale rule or the build records no weapon)."""
    tok = table()["scale"].get(_loose(item))
    if tok is None or not weapons:
        return 1.0
    want = f"{FIELD}:{tok}".lower()
    ok = sum(1 for w in weapons if want in {x.lower() for x in weapon_tokens(w)})
    return ok / len(weapons)


def scaled_items() -> List[str]:
    return list(table()["scale"])


def describe(token: str) -> str:
    v = str(token).split(":", 1)[1] if ":" in str(token) else str(token)
    return f"wielding {v}"


def summary(weapons: Sequence[str]) -> str:
    T = table()["weapons"]
    return ", ".join(f"{w} ({T[w]['type']}, {'2H' if T[w].get('hands') == 2 else '1H'}"
                     f"{', bleed' if T[w].get('bleed') else ''})" for w in weapons if w in T)


# ---------------------------------------------------------------- v3.3: weapons in reviews
#: a review's related item "weapon:<name>" scopes its claim to builds wielding <name>
WEAPON_PREFIX = "weapon:"


def is_weapon_token(x) -> bool:
    return isinstance(x, str) and x.startswith(WEAPON_PREFIX)


def weapon_token(name) -> Optional[str]:
    """"weapon:<canonical name>" for a weapon name (or token), else None."""
    s = str(name or "").strip()
    if is_weapon_token(s):
        s = s[len(WEAPON_PREFIX):]
    hit = resolve(s)
    return WEAPON_PREFIX + hit[0] if len(hit) == 1 and _loose(hit[0]) == _loose(s) else None


def build_weapon_tokens(b) -> Set[str]:
    """{"weapon:<name>"} for every weapon the build equips (Build.meta)."""
    return {WEAPON_PREFIX + w for w in equipped(b)}


def with_weapon_notes(r) -> List[str]:
    """Weapons a review's "# weapon: A, B" notes name, added to the reviewed
    build's equips (Build.meta "weapon"); idempotent. Returns the equips."""
    names: List[str] = []
    for n in (getattr(r, "notes", None) or []):
        if isinstance(n, dict) and n.get("kind") == "weapon":
            for w in n.get("weapons") or []:
                for k in resolve(w):
                    if k not in names:
                        names.append(k)
    b = getattr(r, "original", None)
    if not names or b is None:
        return equipped(b) if b is not None else []
    have = equipped(b)
    new = [w for w in names if w not in have]
    if new:
        meta = dict(b.meta or {})
        key = next((k for k in meta if str(k).lower() in ("weapon", "weapons")), "weapon")
        meta[key] = ", ".join(have + new)
        b.meta = meta
    return have + new
