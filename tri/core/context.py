"""Context tokens: the forced context of a build (oath, murmur, race, weapon ...)
and how a lock atom holds in it."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from tri.core.tables import _loose
from tri.core.types import Build


def atom_holds(atom: dict, highest: Dict[str, int], final: Optional[Dict[str, int]] = None) -> bool:
    """One requirement atom at one stat stage: stats are read from `highest`
    (callers pass the stage, pre-shrine or final), "Total" from the sum of
    `final` (defaults to the same stage)."""
    final = highest if final is None else final
    if "Total" in atom["stats"]:
        return sum(final.values()) >= atom["value"]
    return sum(highest.get(s, 0) for s in atom["stats"]) >= atom["value"]


CONTEXT_LOCK_FIELDS: Tuple[str, ...] = ("oath", "murmur", "origin", "wield")   # wield: tri/weapons.py (v2.21)
MANTRA_LOCKS: Tuple[str, ...] = ("oath:Silentheart",)   # Silentheart builds cannot use mantras
#: Build.meta key -> lock field. The planner records Oath, Murmur and Origin on
#: every build (an empty value means none).
META_LOCK_KEYS: Dict[str, str] = {"oath": "oath", "murmur": "murmur", "origin": "origin"}
#: Fields every build with planner metadata records (empty = none, so locks apply).
PLANNER_FIELDS: Tuple[str, ...] = ("oath", "murmur", "origin")
#: Single-valued fields: a pick locked to one value of the field shows the value.
SINGLE_VALUED: Tuple[str, ...] = ("oath", "murmur", "origin")


def _field(token: str) -> str:
    return str(token).split(":", 1)[0].lower()


@lru_cache(maxsize=4096)
def token_key(t: str) -> str:
    """Comparable form of a context token: field lower case, value loosened."""
    f, _, v = str(t).partition(":")
    return f"{f.strip().lower()}:{_loose(v)}"


def describe_token(t: str) -> str:
    """Readable form of a context token: "Oath: X", "Murmur: X", "origin X"."""
    f, _, v = str(t).partition(":")
    f = f.strip().lower(); v = v.strip()
    return {"oath": f"Oath: {v}", "murmur": f"Murmur: {v}", "origin": f"origin {v}",
            "wield": f"wielding {v}"}.get(f, str(t))


def _empty(v) -> bool:
    return v is None or str(v).strip() in ("", "None", "none")


def wield_waived(b: "Build") -> Set[str]:
    """Items whose weapon locks a review waived for this build (Build.meta
    "wield_waived": the build equips another weapon, wields one it did not
    select, or the talent works without it). v2.23."""
    m = {str(k).lower(): v for k, v in ((getattr(b, "meta", None) or {})).items()}
    v = m.get("wield_waived") or ()
    return {str(x) for x in ([v] if isinstance(v, str) else v)}


def build_context(b: "Build") -> Tuple[List[str], Set[str]]:
    """(tokens, fields) of a build from Build.meta: tokens "oath:<v>",
    "murmur:<v>", "origin:<v>", and the lock fields the build records. A build
    with planner metadata records oath, murmur and origin (an empty value means
    none, so their locks apply); a build without metadata records nothing (no
    lock is checked). See Catalog.build_context for the version that also reads
    the build's picks."""
    meta = {str(k).lower(): v for k, v in (b.meta or {}).items()}
    tokens: List[str] = []
    fields: Set[str] = set(PLANNER_FIELDS) if meta else set()
    for k, f in META_LOCK_KEYS.items():
        if k in meta:
            fields.add(f)
            if not _empty(meta[k]):
                tokens.append(f"{f}:{meta[k]}")
    from tri.game import weapons as W  # forced weapon context (v2.21)
    ws = [] if meta.get("wield_free") else W.equipped(b)   # wield_free: a review waived every weapon lock (v2.23)
    if ws:
        fields.add(W.FIELD)
        tokens += W.tokens_of(ws)
    return tokens, fields
