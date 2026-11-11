"""dwt.sh builds <-> tri (v2.22).

dwt.sh (https://dwt.sh, API https://api.dwt.sh, docs https://dwt.sh/docs/api)
stores a build as `meta` (name, description, visibility, stars, ...) and `game`
(stages, talents, mantras, oath, origin, murmur, weapons, gear, appearance).

Stages. `game.stages` is the run in order, at most 32 stages. A stage's `stats`
are the points spent by the end of that stage and its `name` is the shrine used
to leave it ("Shrine of Order", "Shrine of Mastery"); the last stage (name "")
is the finished build. dwt's own "PreShrine" is the first stage named Shrine of
Order. tri reads:

    initial  = stages[0]        -> Build.pre_stats   (learned from)
    final    = stages[-1]       -> Build.final_stats (learned from)
    middle   = stages[1:-1]     -> Build.mid_stages  (requirements only)

Every stage is a state the character really stood in and picked talents at,
so requirements may be met at any of them (Build.stages()). The model learns
only from the initial and final stages (highest_stats = max(initial, final),
batch matrices carry those two), so a long Shrine of Mastery chain never
changes what a build teaches.

This module is pure: no network. bot/dwt.py is the HTTP client.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

API = "https://api.dwt.sh"
SITE = "https://dwt.sh"
HOSTS = ("dwt.sh", "www.dwt.sh")
PREFIX = "dwt."                           # tri build ids of dwt.sh builds: "dwt.<id>"
_RAW = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

#: dwt stat names -> planner stat names (tri.pipeline.STAT_ALIASES keys)
STAT_TO_PLANNER: Dict[str, str] = {"Light": "Light Wep.", "Medium": "Medium Wep.", "Heavy": "Heavy Wep."}
#: dwt stat names -> tri stat names
STAT_FROM_DWT: Dict[str, str] = {
    "Strength": "Strength", "Fortitude": "Fortitude", "Agility": "Agility", "Intelligence": "Intelligence",
    "Willpower": "Willpower", "Charisma": "Charisma",
    "Light": "LightWeapon", "Medium": "MediumWeapon", "Heavy": "HeavyWeapon",
    "Flamecharm": "Flamecharm", "Frostdraw": "Frostdraw", "Thundercall": "Thundercall",
    "Galebreathe": "Galebreathe", "Shadowcast": "Shadowcast", "Ironsing": "Ironsing", "Bloodrend": "Bloodrend",
}
STAT_TO_DWT: Dict[str, str] = {v: k for k, v in STAT_FROM_DWT.items()}
ANY_RACE = ("", "any", "none")


def key(s: str) -> str:
    """Name key shared with tri.pipeline._name_key: casefold, alphanumerics only."""
    return "".join(ch for ch in str(s).casefold() if ch.isalnum())


# ------------------------------------------------------------------ ids and links
def is_dwt_id(build_id: str) -> bool:
    return str(build_id or "").startswith(PREFIX)


def raw_id(build_id: str) -> str:
    """"dwt.<id>" -> "<id>"."""
    return str(build_id)[len(PREFIX):] if is_dwt_id(build_id) else str(build_id)


def link(build_id: str) -> str:
    return f"{SITE}/builds/{raw_id(build_id)}"


def parse_link(text: str) -> Optional[str]:
    """dwt.sh build link (or "dwt.<id>") -> "dwt.<id>"; None if it is not a
    dwt.sh link. ValueError if it is one but names no build."""
    s = (text or "").strip().strip("<>")
    if is_dwt_id(s) and _RAW.match(raw_id(s)):
        return s
    if "://" not in s:
        if not s.split("/")[0].lower() in HOSTS:
            return None
        s = "https://" + s
    u = urlparse(s)
    if (u.hostname or "").lower() not in HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "builds" and _RAW.match(parts[1]):
        return PREFIX + parts[1]
    if parts and parts[0] == "share":
        raise ValueError("dwt.sh share links cannot be read; use the build's /builds/<id> link")
    raise ValueError(f"no dwt.sh build id in {text!r}")


def parse_drive_link(text: str) -> Optional[str]:
    """dwt.sh drive (folder) link -> folder id; None if it is not one (v2.23).
    https://dwt.sh/drive/<id> ; "drive:<id>" also accepted."""
    s = (text or "").strip().strip("<>")
    if s.lower().startswith("drive:") and _RAW.match(s[6:]):
        return s[6:]
    if "://" not in s:
        if not s.split("/")[0].lower() in HOSTS:
            return None
        s = "https://" + s
    u = urlparse(s)
    if (u.hostname or "").lower() not in HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "drive" and _RAW.match(parts[1]):
        return parts[1]
    return None


def folder_builds(listing: dict) -> Tuple[List[str], List[str]]:
    """(build ids "dwt.<id>", subfolder ids) of a FolderListing
    (GET /resources/{id}/list). Kits, drips and documents are skipped."""
    b, f = [], []
    for c in (listing or {}).get("children") or []:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        if c.get("kind") == "build":
            b.append(PREFIX + str(c["id"]))
        elif c.get("kind") == "folder":
            f.append(str(c["id"]))
    return b, f


# ------------------------------------------------------------------ reading
def stage_stats(stage: dict) -> Dict[str, int]:
    """One dwt stage -> tri stat names (zero stats dropped)."""
    out: Dict[str, int] = {}
    for k, v in ((stage or {}).get("stats") or {}).items():
        n = STAT_FROM_DWT.get(k)
        if n is not None and isinstance(v, (int, float)) and v:
            out[n] = int(v)
    return out


def _planner_block(stage: dict) -> dict:
    """One dwt stage -> a planner stat block ({"base": {planner name: v}})."""
    return {"base": {STAT_TO_PLANNER.get(k, k): int(v) for k, v in ((stage or {}).get("stats") or {}).items()
                     if k in STAT_FROM_DWT and isinstance(v, (int, float))}}


def weapons_of(game: dict) -> List[dict]:
    """The build's weapons ordered by priority (even = main hand, odd = its offhand)."""
    ws = [dict(w) for w in ((game or {}).get("weapons") or {}).values() if isinstance(w, dict) and w.get("name")]
    return sorted(ws, key=lambda w: int(w.get("priority") or 0))


def to_record(obj: dict, build_id: Optional[str] = None) -> Dict[str, Any]:
    """A dwt.sh build (GET /builds/{id}) -> a planner record that
    tri.pipeline.parse_record reads. Keeps the dwt build under "dwt" so a
    proposal can be written back over it (to_game)."""
    g = dict((obj or {}).get("game") or {})
    m = dict((obj or {}).get("meta") or {})
    stages = [s for s in (g.get("stages") or []) if isinstance(s, dict)]
    blocks = [_planner_block(s) for s in stages]
    final = blocks[-1] if blocks else {}
    pre = blocks[0] if len(blocks) > 1 else {}
    mid = blocks[1:-1]
    race = str(g.get("race") or "")
    meta = dict(Oath=g.get("oath") or "", Origin=g.get("origin") or "", Murmur=g.get("murmur") or "",
                Race="" if race.strip().lower() in ANY_RACE else race,
                Bell=", ".join(str(x) for x in (g.get("bell") or [])))
    ws = weapons_of(g)
    main = [w for w in ws if int(w.get("priority") or 0) % 2 == 0]
    owner = (obj or {}).get("owner") or {}
    bid = build_id or (PREFIX + str(obj.get("id")) if obj.get("id") else None)
    return dict(
        source="dwt", id=bid,
        attributes=final, preShrine=pre, midStages=mid,
        stageNames=[str(s.get("name") or "") for s in stages],
        talents=[t for t in (g.get("talents") or []) if isinstance(t, str)],
        mantras=[k for k in (g.get("mantras") or {}).keys() if isinstance(k, str)],
        stats=dict(buildName=m.get("name") or bid, buildDescription=m.get("description") or "",
                   meta=meta, buildAuthor=owner.get("username") if owner else None),
        weapons=", ".join(w["name"] for w in ws) or None,
        enchant=next((w.get("enchant") for w in main if w.get("enchant")), None),
        meta=dict(visibility=m.get("visibility"), votes={"up": int(m.get("stars") or 0)}),
        author=dict(id=owner.get("id"), name=owner.get("username")) if owner else None,
        createdAt=m.get("createdAt"), updatedAt=m.get("modifiedAt"),
        dwt=copy.deepcopy(obj))


# ------------------------------------------------------------------ writing
def default_game() -> Dict[str, Any]:
    """Every field GameData requires, empty (POST /builds rejects a game that lacks one)."""
    return dict(stages=[], talents=[], race="Any", oath="", bell=[], mantras={}, weapons={},
                startsWithMultifaceted=False, origin="", boons=[], flaws=[], traits={}, equipment={},
                cosmetic={}, dyes={}, skinTone=0, variant="")


def stage(name: str, stats: Dict[str, int]) -> dict:
    """A dwt stage from tri stats."""
    return dict(name=str(name or ""), stats={STAT_TO_DWT[s]: int(v) for s, v in (stats or {}).items()
                                             if s in STAT_TO_DWT}, obtained=[])


def stages_of(build) -> List[dict]:
    """dwt stages from a Build: initial (named Shrine of Order), middle
    stages (named by Build.meta "stages" when known), final."""
    names = list((getattr(build, "meta", None) or {}).get("stages") or [])
    pre, mid = getattr(build, "pre_stats", None) or {}, list(getattr(build, "mid_stages", None) or [])
    seq = ([pre] if pre else []) + mid + [build.final_stats]
    if len(names) != len(seq):
        names = (["Shrine of Order"] if pre else []) + ["Shrine of Mastery"] * len(mid) + [""]
    return [stage(n, s) for n, s in zip(names, seq)]


def wildcard_flags(mantras: Sequence[str], catalog=None, items: Iterable[str] = (),
                   oath: Optional[str] = None, prefer: Iterable[str] = ()) -> Dict[str, bool]:
    """isWildcard per mantra: a mantra past its own type's slots takes a
    wildcard slot. prefer: mantras the source build already had in a wildcard
    slot (they are the first to stay there). Without a catalog every flag is False."""
    out = {m: False for m in mantras}
    if catalog is None:
        return out
    slots = catalog.mantra_slots_of(list(items) + list(mantras), oath)
    used: Dict[str, int] = {}
    pk = {key(x) for x in prefer}
    order = sorted(mantras, key=lambda x: key(x) in pk)       # stable: own-slot mantras first
    typ = {m: (catalog.mantra_slot(m) if m in catalog.items else None) for m in order}

    def place(m, k):
        if used.get(k, 0) < int(slots.get(k, 0)):
            used[k] = used.get(k, 0) + 1
            return True
        return False
    # same placement as fit_mantra_slots: own type, then wisps Wisp -> Support, rest wildcard
    for m in order:
        t = typ[m]
        out[m] = not (t in ("Combat", "Mobility", "Support") and place(m, t))
    for m in order:
        if typ[m] == "Wisp":
            out[m] = not (place(m, "Wisp") or place(m, "Support"))
    return out


def apply_changes(names: Sequence[str], drops: Iterable[str], takes: Iterable[str],
                  spell: Optional[Dict[str, str]] = None) -> List[str]:
    """A raw name list with drops removed and takes appended (matched by
    name key, so "To the Finish" and "To The Finish" are one talent). spell:
    name key -> dwt spelling for added names."""
    gone = {key(x) for x in drops}
    out = [n for n in names if key(n) not in gone]
    have = {key(n) for n in out}
    for t in takes:
        if key(t) not in have:
            out.append((spell or {}).get(key(t), t))
            have.add(key(t))
    return out


def to_game(talents: Sequence[str], mantras: Sequence[str], *, base: Optional[dict] = None,
            build=None, record: Optional[dict] = None, catalog=None,
            oath: Optional[str] = None) -> Dict[str, Any]:
    """GameData for a proposal. base: the dwt game it came from (stages,
    gear, appearance and mantra fittings are kept; only talents and mantras
    change). Without one the game is made from the Build (stages) and the
    planner record (oath, origin, murmur, race, weapons)."""
    g = copy.deepcopy(base) if base else default_game()
    for k, v in default_game().items():
        g.setdefault(k, v)
    if not base:
        if build is not None:
            g["stages"] = stages_of(build)
        sm = ((record or {}).get("stats") or {}).get("meta") or {}
        g["oath"] = str(sm.get("Oath") or oath or "")
        g["origin"] = str(sm.get("Origin") or "")
        if sm.get("Murmur"):
            g["murmur"] = str(sm["Murmur"])
        if sm.get("Race"):
            g["race"] = str(sm["Race"])
    if catalog is not None:                       # the API adds in order: prerequisites first
        talents = catalog.prereq_order(list(talents), lambda x: _canon(x, catalog))
        mantras = catalog.prereq_order(list(mantras), lambda x: _canon(x, catalog))
    g["talents"] = list(talents)
    old = {key(k): v for k, v in (g.get("mantras") or {}).items()}
    flags = wildcard_flags([_canon(m, catalog) for m in mantras], catalog, items=[_canon(t, catalog) for t in talents],
                           oath=oath or g.get("oath") or None,
                           prefer=[_canon(k, catalog) for k, v in (g.get("mantras") or {}).items()
                                   if isinstance(v, dict) and v.get("isWildcard")])
    out = {}
    for m in mantras:
        info = dict(old.get(key(m)) or dict(modifiers={}, sparks=[], level=1))
        info["isWildcard"] = bool(flags.get(_canon(m, catalog), False))
        out[m] = info
    g["mantras"] = out
    return g


def _canon(name: str, catalog) -> str:
    if catalog is None:
        return name
    if name in catalog.items:
        return name
    idx = getattr(catalog, "_dwt_keys", None)
    if idx is None:
        idx = {key(i): i for i in catalog.items}
        try:
            catalog._dwt_keys = idx
        except Exception:
            pass
    return idx.get(key(name), name)


def payload(game: dict, name: str, description: str = "", visibility: str = "unlisted",
            notes: str = "") -> Dict[str, Any]:
    """Body of POST /builds. `notes` is the build's summary notes (meta.notes),
    where tri puts its completion notes (v2.23)."""
    meta = dict(name=str(name)[:100], description=str(description)[:4000],
                visibility=visibility if visibility in ("public", "private", "unlisted") else "unlisted")
    if notes:
        meta["notes"] = str(notes)[:8000]
    return dict(meta=meta, game=game)


def spellings(data: dict) -> Dict[str, str]:
    """name key -> dwt spelling, from the dwt data bundle (GET /data)."""
    out: Dict[str, str] = {}
    for sect in ("talents", "mantras"):
        for v in ((data or {}).get(sect) or {}).values():
            if isinstance(v, dict) and v.get("name"):
                out.setdefault(key(v["name"]), str(v["name"]))
    return out


def proposal(record: dict, build, drops: Iterable[str], takes: Iterable[str], catalog=None,
             spell: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """(talents, mantras, game) of tri's proposal for a build read from
    `record`: its raw picks (granted talents and "Oath: X" included) with the
    drops removed and the takes added. Works for dwt.sh and deepwoken.co records."""
    drops, takes = list(drops), list(takes)
    is_m = (lambda x: catalog is not None and _canon(x, catalog) in catalog.items
            and catalog.items[_canon(x, catalog)].kind.value == "mantra")
    base = (record or {}).get("dwt") or {}
    bg = base.get("game") if base else None
    raw_t = list((bg or {}).get("talents") or (record or {}).get("talents") or [])
    raw_m = list(((bg or {}).get("mantras") or {}).keys()) if bg else list((record or {}).get("mantras") or [])
    t = apply_changes(raw_t, [x for x in drops if not is_m(x)], [x for x in takes if not is_m(x)], spell)
    m = apply_changes(raw_m, [x for x in drops if is_m(x)], [x for x in takes if is_m(x)], spell)
    oath = None
    if catalog is not None and build is not None:
        try:
            oath = catalog.oath_of(build)
        except Exception:
            oath = None
    return t, m, to_game(t, m, base=bg, build=build, record=record, catalog=catalog, oath=oath)
