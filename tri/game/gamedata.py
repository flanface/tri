"""Known game data: talents and mantras from public sources.

Two sources are supported and can be merged (merge_gamedata):

  * deepwoken.co  - the build planner's own database, served publicly at
                    https://api.deepwoken.co/get?type=all (load_deepwokenco /
                    fetch_deepwokenco). Its names are exactly the planner's, it
                    carries structured requirements, prerequisites, exclusions
                    and countTowardsTalentTotal, and it is the default primary.
  * pocamind/data - https://github.com/pocamind/data, a flat JSON database (one
                    file per talent / mantra) with requirement strings
                    (load_gamedata / fetch_gamedata). Used to fill names the
                    primary does not know.

Both give, per entry, the requirements, prerequisites, mutual exclusions and
whether the talent takes a talent slot. tri uses them for

  * requirements   - exact, instead of inferring them from builds
  * granted talents - count_towards_talent_total == false, or implicit == true
                      (mantra-level, Unbounded, oath, murmur, race, origin, quest,
                      equipment, outfit, weapon, memento, ... talents)
  * prerequisites and exclusions, enforced by Tri.review

    gd = fetch_all("runs/gamedata")              # deepwoken.co + pocamind, merged (cached; --gamedata all)
    gd = load_any("deepwokenco.json")            # either source from a local file / dir / zip
    gd = load_gamedata("path/to/data")          # pocamind: a clone, an extracted zip, or the .zip itself
    gd = fetch_gamedata("runs/gamedata")         # pocamind main, downloaded once (cached)
    cat = gd.catalog()                           # every slot-taking talent + every mantra
    gd.granted()                                 # {talent name: group} that take no slot

Requirement strings (the grammar of the pocamind `deepwoken` crate):
    "()"                                   nothing
    "30r STR"                              30 Strength, reducible
    "25s AGL"                              25 Agility, strict
    "40r AGL, 60r STR"                     comma = AND of clauses
    "20r MED OR 20r LHT OR 20r FTD"        OR inside a clause
    "LHT + MED + HVY = 90r"                sum of stats
    "135s TTL"                             total points invested
  Unmarked atoms: strict if a single stat in an AND clause, reducible otherwise.
  Every requirement must be met when the talent is taken, i.e. at the highest
  stats the build reached; talents are kept when the Shrine of Order lowers
  stats afterwards. Strict vs reducible only matters to the Shrine of Mastery
  (it will not lower a stat below a held talent's strict requirement), so tri
  stores the flag but does not use it for availability. "TTL" is checked
  against total points invested (the sum of final stats)."""
from __future__ import annotations

import functools
import io
import json
import os
import re
import time
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tri.core.catalog import Catalog
from tri.core.types import DEFAULT_STATS, Category, Item, Kind
from tri.game.defs import GameItem, ReqParseError

REPO = "pocamind/data"
ZIP_URL = "https://codeload.github.com/pocamind/data/zip/refs/heads/{ref}"
ZIP_SHA_URL = "https://codeload.github.com/pocamind/data/zip/{sha}"
API_URL = "https://api.github.com/repos/pocamind/data/commits/{ref}"
#: requirement abbreviations and names -> tri stat names ("Total" = all points)
STAT_CODES: Dict[str, str] = {
    "STR": "Strength", "FTD": "Fortitude", "AGL": "Agility", "INT": "Intelligence",
    "WLL": "Willpower", "WIL": "Willpower", "CHA": "Charisma",
    "HVY": "HeavyWeapon", "MED": "MediumWeapon", "LHT": "LightWeapon",
    "ICE": "Frostdraw", "FLM": "Flamecharm", "FIR": "Flamecharm", "LTN": "Thundercall",
    "WND": "Galebreathe", "SDW": "Shadowcast", "MTL": "Ironsing", "BLD": "Bloodrend",
    "TTL": "Total", "TOT": "Total",
    "STRENGTH": "Strength", "FORTITUDE": "Fortitude", "AGILITY": "Agility",
    "INTELLIGENCE": "Intelligence", "WILLPOWER": "Willpower", "CHARISMA": "Charisma",
    "HEAVY": "HeavyWeapon", "MEDIUM": "MediumWeapon", "LIGHT": "LightWeapon",
    "FROSTDRAW": "Frostdraw", "FLAMECHARM": "Flamecharm", "THUNDERCALL": "Thundercall",
    "GALEBREATHE": "Galebreathe", "SHADOWCAST": "Shadowcast", "IRONSING": "Ironsing",
    "BLOODREND": "Bloodrend", "TOTAL": "Total",
}


def name_key(name: str) -> str:
    """Loose matching key: case-folded letters and digits only."""
    return _name_key(name if isinstance(name, str) else str(name))


@functools.lru_cache(maxsize=65536)
def _name_key(name: str) -> str:
    """name_key of a str, memoised (v3.8: hundreds of thousands of calls per review rebuild)."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", name).casefold() if ch.isalnum())


# ------------------------------------------------------------------ requirement parser
_SUM = re.compile(r"^([A-Za-z]+(?:\s*\+\s*[A-Za-z]+)*)\s*=\s*(\d+)\s*([rRsS]?)$")
_VAL = re.compile(r"^(\d+)\s*([rRsS]?)\s*([A-Za-z]+)$")
_OR = re.compile(r"\s+or\s+|\s*\|\|\s*", re.IGNORECASE)


def _strip_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def _stat(code: str) -> str:
    st = STAT_CODES.get(code.strip().upper())
    if st is None:
        raise ReqParseError(f"unknown stat {code!r}")
    return st


def _atom(text: str, in_or: bool) -> dict:
    t = _strip_parens(text)
    m = _SUM.match(t)
    if m:
        stats = [_stat(x) for x in m.group(1).split("+")]
        flag = m.group(3).lower()
        strict = flag == "s" if flag else (len(stats) == 1 and not in_or)
        return dict(stats=sorted(set(stats)), value=int(m.group(2)), strict=strict)
    m = _VAL.match(t)
    if m:
        val, flag, code = m.groups()
        if flag and code.upper() not in STAT_CODES and (flag + code).upper() in STAT_CODES:
            flag, code = "", flag + code             # "30SDW": S belongs to the stat
        flag = flag.lower()
        strict = flag == "s" if flag else not in_or
        return dict(stats=[_stat(code)], value=int(val), strict=strict)
    raise ReqParseError(f"cannot parse requirement atom {text!r}")


def parse_reqs(text: Optional[str]) -> List[List[dict]]:
    """Requirement string -> CNF clauses [[{"stats", "value", "strict"}, ...], ...].
    Raises ReqParseError on malformed input."""
    if text is None:
        return []
    t = _strip_parens(str(text))
    if not t:
        return []
    clauses = []
    for part in t.split(","):
        part = _strip_parens(part)
        if not part:
            continue
        alts = [a for a in _OR.split(part) if a.strip()]
        clauses.append([_atom(a, len(alts) > 1) for a in alts])
    return clauses


def format_reqs(clauses: List[List[dict]]) -> str:
    """Inverse of parse_reqs (canonical form, tri stat names)."""
    inv = {"Strength": "STR", "Fortitude": "FTD", "Agility": "AGL", "Intelligence": "INT",
           "Willpower": "WLL", "Charisma": "CHA", "HeavyWeapon": "HVY", "MediumWeapon": "MED",
           "LightWeapon": "LHT", "Frostdraw": "ICE", "Flamecharm": "FLM", "Thundercall": "LTN",
           "Galebreathe": "WND", "Shadowcast": "SDW", "Ironsing": "MTL", "Bloodrend": "BLD", "Total": "TTL"}

    def one(a):
        f = "s" if a.get("strict") else "r"
        if len(a["stats"]) == 1:
            return f"{a['value']}{f} {inv.get(a['stats'][0], a['stats'][0])}"
        return " + ".join(inv.get(x, x) for x in a["stats"]) + f" = {a['value']}{f}"
    return ", ".join(" OR ".join(one(a) for a in cl) for cl in clauses) or "()"
# ------------------------------------------------------------------ data model


def mantra_type_of(d: dict) -> Optional[str]:
    """Mantra type of a game-data entry. An "Oath: X" attribute makes it Oath
    whatever "type" says (deepwoken.co lists Soul Beam as Normal)."""
    if any(str(a).startswith("Oath:") for a in d.get("attributes") or []):
        return "Oath"
    return str(d["type"]) if d.get("type") else None


def _slot_grants(d: dict) -> Dict[str, int]:
    """Mantra slots a talent grants by type: the "<Type> Mantra Slot" entries
    of its stats ({"Combat": 2, "Wildcard": 1})."""
    out: Dict[str, int] = {}
    for k, v in (d.get("stats") or {}).items():
        k = str(k).strip()
        if k.lower().endswith("mantra slot"):
            try:
                t = k[: -len("mantra slot")].strip().title() or "Wildcard"
                out[t] = out.get(t, 0) + int(v)
            except (TypeError, ValueError):
                pass
    return out


@dataclass
class GameData:
    items: Dict[str, GameItem]
    source: dict = field(default_factory=dict)
    errors: List[Tuple[str, str]] = field(default_factory=list)   # (name, message)

    def __post_init__(self):
        self.by_ref: Dict[str, str] = {}
        self.by_key: Dict[str, str] = {}                     # "talent:<loose key>" -> name
        for n, it in self.items.items():
            self.by_ref[f"{it.kind.value}:{it.key}"] = n
            self.by_key.setdefault(f"{it.kind.value}:{name_key(n)}", n)

    def lookup(self, name: str, kind: Optional[str] = None) -> Optional[str]:
        """Canonical name for `name`, ignoring case, spacing and punctuation
        ("To the Finish" -> "To The Finish"); None if unknown. Loose matching
        is per kind ("talent" / "mantra"; both tried if kind is None), since
        e.g. the talent "Gale Trap" and the mantra "Galetrap" share a key."""
        if name in self.items and (kind is None or self.items[name].kind.value == kind):
            return name
        for k in ([kind] if kind else ["talent", "mantra"]):
            hit = self.by_key.get(f"{k}:{name_key(name)}")
            if hit:
                return hit
        return None

    def canonical_names(self) -> Dict[str, str]:
        """{"talent:<loose key>" / "mantra:<loose key>": canonical name};
        PipelineConfig.canonical_names."""
        return dict(self.by_key)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def suggest(self, name: str, n: int = 3) -> List[str]:
        """Closest known names (difflib on loose keys), for unknown names."""
        import difflib
        keys = {k.split(":", 1)[1]: v for k, v in self.by_key.items()}
        return [keys[k] for k in difflib.get_close_matches(name_key(name), list(keys), n=n, cutoff=0.75)]

    def fingerprint(self) -> str:
        """Short hash of everything tri reads from the data (names, kinds,
        requirements, slot/implicit flags, prerequisites, exclusions)."""
        import hashlib
        h = hashlib.sha256()
        for n in sorted(self.items):
            it = self.items[n]
            h.update(json.dumps([n, it.kind.value, it.reqs, it.clauses, it.slot, it.implicit, it.vaulted,
                                 it.rarity, it.prereqs, it.exclusive] + ([True] if it.voi_only else []), sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    def granted(self) -> Dict[str, str]:
        """{talent name: group} for talents that take no slot. Group is the
        rarity in lower case (oath, equipment, outfit, weapon, quest, origin,
        murmur, innate, memento, faction, spec), or mantra_level / mastery for
        the Common-rarity attunement tiers and Unbounded talents."""
        out = {}
        for n, it in self.items.items():
            if it.granted:
                g = (it.rarity or "unknown").lower()
                if g == "common":
                    g = "mastery" if n.endswith(" Unbounded") else "mantra_level"
                out[n] = g
        return out

    def granted_known(self, keep_groups: Sequence[str] = ()) -> Dict[str, Optional[str]]:
        """{talent name: group or None}: an authoritative answer for every known
        talent (None = takes a slot / is treated as a choice). Groups listed in
        keep_groups (e.g. "faction": no slot, but chosen from a separate pool)
        are treated as choices. Used as PipelineConfig.granted_known."""
        g = {n: x for n, x in self.granted().items() if x not in set(keep_groups)}
        return {n: g.get(n) for n, it in self.items.items() if it.kind == Kind.TALENT}

    def resolve(self, ref: str) -> Optional[str]:
        """'talent:speed_demon' -> 'Speed Demon' (None for other namespaces / unknown keys)."""
        return self.by_ref.get(ref.strip())

    def context_token(self, name: Optional[str]) -> Optional[str]:
        """Context token a granted oath / murmur talent stands for:
        "Oath: Arcwarder" -> "oath:Arcwarder", "Murmur: Tacet" -> "murmur:Tacet";
        other talents of rarity Oath / Murmur -> "<rarity>:<category>". None otherwise."""
        if not name:
            return None
        for pre, f in (("Oath:", "oath"), ("Murmur:", "murmur")):
            if name.startswith(pre):
                v = name[len(pre):].strip()
                return f"{f}:{v}" if v else None
        g = self.items.get(name)
        r = (getattr(g, "rarity", "") or "").lower() if g is not None else ""
        if r in ("oath", "murmur") and getattr(g, "category", None):
            return f"{r}:{g.category}"
        return None

    def ref_token(self, ref: str) -> Optional[str]:
        """Context token of a prerequisite reference that is not an item:
        "origin:authority_ensign" -> "origin:Authority Ensign" (pocamind/data
        writes origin requirements this way). None for other references."""
        kind, _, key = str(ref).partition(":")
        if kind.strip().lower() == "origin" and key.strip():
            return "origin:" + " ".join(w.capitalize() for w in key.replace("_", " ").split())
        return None

    def condition_tokens(self, k: str, v) -> Optional[List[str]]:
        """Alternatives (one clause) for a lock condition, or None if k is not
        one: oath / murmur / origin -> "<field>:<v>"; "or" -> the union when
        every alternative is a single lock condition. Weapon conditions are not
        locks (a build's weapon is not checked)."""
        k = str(k)
        kl = k.lower()
        if kl == "or":
            if not isinstance(v, (list, tuple)) or not v:
                return None
            out = []
            for a in v:
                if not isinstance(a, dict) or len(a) != 1:
                    return None
                (kk, vv), = a.items()
                t = self.condition_tokens(kk, vv)
                if not t:
                    return None
                out += t
            return out
        vals = [x for x in (v if isinstance(v, (list, tuple)) else [v]) if str(x).strip()]
        if not vals:
            return None
        if kl in ("oath", "murmur", "origin"):
            return [f"{kl}:{str(x).strip()}" for x in vals]
        return None

    def catalog(self, names: Optional[Iterable[str]] = None, stats: Sequence[str] = DEFAULT_STATS,
                ruleset: str = "unknown", include_granted: bool = False, keep_groups: Sequence[str] = (),
                include_vaulted: bool = True, report: Optional[dict] = None) -> Catalog:
        """Catalog of slot-taking talents and all mantras (restricted to `names`
        if given). Requirements -> Item.clauses; prerequisite groups whose every
        alternative is in the catalog -> prerequisites (one alternative) or
        prerequisites_any (several). keep_groups: granted groups kept as
        choices (see granted_known). Groups whose every alternative is a granted
        oath or murmur talent ("Oath: Arcwarder") become context requirements
        (Item.context, a clause of tokens "oath:Arcwarder"), as do "oath" /
        "murmur" conditions (counted in report["context_clauses"]). Other groups
        pointing outside the catalog (objectives, weapons, equipment, ...)
        cannot be checked from a build and are dropped (counted in
        report["prereq_groups_dropped"]).
        Exclusions are kept where both items are in the catalog. Rarity,
        category and vaulted status go into Item.tags."""
        pick = set(names) if names is not None else None
        gr = {n for n, x in self.granted().items() if x not in set(keep_groups)}
        chosen = {n: it for n, it in self.items.items()
                  if (pick is None or n in pick) and (include_granted or n not in gr)
                  and (include_vaulted or not it.vaulted)}
        dropped = nctx = 0
        items = {}
        for n, it in chosen.items():
            pre, pre_any, ctx = [], [], []
            for grp in it.prereqs:
                alts = [self.resolve(r) for r in grp]
                if alts and all(a in chosen for a in alts):
                    alts = sorted(set(alts))
                    (pre.append(alts[0]) if len(alts) == 1 else pre_any.append(alts))
                    continue
                toks = [self.context_token(a) if a is not None else self.ref_token(r)
                        for r, a in zip(grp, alts)]
                if alts and all(toks):
                    ctx.append(sorted(set(toks))); nctx += 1
                else:
                    dropped += 1
            for k, v in sorted(it.conditions.items()):
                toks = self.condition_tokens(k, v)
                if toks:
                    cl = sorted(set(toks))
                    if cl not in ctx:
                        ctx.append(cl); nctx += 1
            tags = [f"rarity:{it.rarity}"] if it.rarity else []
            if it.category:
                tags.append(f"category:{it.category}")
            if it.vaulted:
                tags.append("vaulted")
            if it.voi_only:
                tags.append("voi_only")
            if it.mtype:
                tags.append(f"type:{it.mtype}")
            if it.slot_grants:
                tags.append(f"mantra_slots:{it.mantra_slots}")
                tags += [f"mantra_slot:{k}={v}" for k, v in sorted(it.slot_grants.items())]
            tags += [f"condition:{k}={v}" for k, v in sorted(it.conditions.items())]
            items[n] = Item(n, it.kind, Category.UNKNOWN, clauses=[list(c) for c in it.clauses],
                            prerequisites=sorted(set(pre)), prerequisites_any=pre_any,
                            exclusive_with=sorted(x for x in it.exclusive if x in chosen and x != n),
                            tags=tags, context=ctx)
        if report is not None:
            report["prereq_groups_dropped"] = dropped
            report["context_clauses"] = nctx
        return Catalog(items, stats=tuple(stats), ruleset=ruleset)
# ------------------------------------------------------------------ loading


def _find_root(names: List[str]) -> Optional[str]:
    """Prefix under which talents/ and mantras/ live, in a list of paths."""
    for n in names:
        parts = n.replace("\\", "/").split("/")
        if "talents" in parts:
            i = parts.index("talents")
            root = "/".join(parts[:i])
            if any(x.replace("\\", "/").startswith((root + "/" if root else "") + "mantras/") for x in names):
                return root
    return None


def _read_entries(path: str):
    """Yield (table, stem, json) for talents/ and mantras/ under a dir or zip."""
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        names = z.namelist()
        root = _find_root(names)
        if root is None:
            raise FileNotFoundError(f"{path}: no talents/ and mantras/ folders")
        pre = root + "/" if root else ""
        for n in names:
            for table in ("talents", "mantras"):
                if n.startswith(pre + table + "/") and n.endswith(".json"):
                    yield table, os.path.splitext(os.path.basename(n))[0], json.loads(z.read(n))
        return
    root = None
    for cand in (path, *[os.path.join(path, d) for d in sorted(os.listdir(path))]):
        if os.path.isdir(os.path.join(cand, "talents")) and os.path.isdir(os.path.join(cand, "mantras")):
            root = cand
            break
    if root is None:
        raise FileNotFoundError(f"{path}: no talents/ and mantras/ folders")
    for table in ("talents", "mantras"):
        d = os.path.join(root, table)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    yield table, fn[:-5], json.load(f)


def load_gamedata(path: str) -> GameData:
    """Load pocamind/data from a clone, an extracted archive, or the .zip.
    Entries whose requirement string does not parse keep no clauses and are
    listed in GameData.errors."""
    from tri.sources.pipeline import normalize_name
    items: Dict[str, GameItem] = {}
    errors = []
    for table, stem, d in _read_entries(path):
        name = normalize_name(d.get("name") or stem)
        kind = Kind.TALENT if table == "talents" else Kind.MANTRA
        try:
            clauses = parse_reqs(d.get("reqs"))
        except ReqParseError as e:
            clauses = []
            errors.append((name, str(e)))
        prereqs = [[x.strip() for x in str(p).split("|") if x.strip()] for p in d.get("prereqs") or []]
        it = GameItem(name=name, key=stem, kind=kind, rarity=d.get("rarity"), category=d.get("category"),
                      slot=bool(d.get("count_towards_talent_total", True)) if kind == Kind.TALENT else True,
                      implicit=bool(d.get("implicit", False)), vaulted=bool(d.get("vaulted", False)),
                      reqs=str(d.get("reqs") or "()"), clauses=clauses, prereqs=[g for g in prereqs if g],
                      exclusive=[normalize_name(x) for x in d.get("exclusive") or []],
                      mtype=(mantra_type_of(d) if kind == Kind.MANTRA else None),
                      slot_grants=_slot_grants(d) if kind == Kind.TALENT else {},
                      voi_only=d.get("voi_only") is True)
        if name in items and items[name].kind != kind:
            errors.append((name, "name used by both a talent and a mantra; keeping the talent"))
            if kind == Kind.MANTRA:
                continue
        items[name] = it
    meta = {}
    for cand in (path + ".json", os.path.join(os.path.dirname(path), "pocamind-data.json")):
        if os.path.isfile(cand):
            with open(cand) as f:
                meta = json.load(f)
            break
    src = dict(name="pocamind/data", path=os.path.abspath(path),
               n_talents=sum(i.kind == Kind.TALENT for i in items.values()),
               n_mantras=sum(i.kind == Kind.MANTRA for i in items.values()), **meta)
    return GameData(items, src, errors)


def fetch_gamedata(dest: str, ref: str = "main", refresh: bool = False, timeout: float = 60.0) -> GameData:
    """Download pocamind/data (branch/tag/commit `ref`) to dest/pocamind-data.zip
    once, record the commit sha when the GitHub API answers, and load it.
    Reuses the download unless refresh=True."""
    os.makedirs(dest, exist_ok=True)
    zp = os.path.join(dest, "pocamind-data.zip")
    if refresh or not os.path.exists(zp):
        is_sha = len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref.lower())
        url = ZIP_SHA_URL.format(sha=ref) if is_sha else ZIP_URL.format(ref=ref)
        req = urllib.request.Request(url, headers={"User-Agent": "tri"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        zipfile.ZipFile(io.BytesIO(data))                      # validate before writing
        with open(zp + ".tmp", "wb") as f:
            f.write(data)
        os.replace(zp + ".tmp", zp)
        meta = dict(repo=REPO, ref=ref, fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            req = urllib.request.Request(API_URL.format(ref=ref),
                                         headers={"User-Agent": "tri", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                c = json.load(r)
            meta.update(sha=c.get("sha"), commit_date=(c.get("commit") or {}).get("committer", {}).get("date"))
        except Exception as e:                                  # rate limits etc.: sha is optional
            meta["sha_error"] = str(e)[:200]
        with open(os.path.join(dest, "pocamind-data.json"), "w") as f:
            json.dump(meta, f, indent=1)
    return load_gamedata(zp)


# ------------------------------------------------------------------ deepwoken.co
DWCO_URL = "https://api.deepwoken.co/get?type=all"
#: deepwoken.co stat names -> tri stat names
DWCO_STATS: Dict[str, str] = {
    "Strength": "Strength", "Fortitude": "Fortitude", "Agility": "Agility", "Intelligence": "Intelligence",
    "Willpower": "Willpower", "Charisma": "Charisma",
    "Light Weapon": "LightWeapon", "Medium Weapon": "MediumWeapon", "Heavy Weapon": "HeavyWeapon",
    "Light Wep.": "LightWeapon", "Medium Wep.": "MediumWeapon", "Heavy Wep.": "HeavyWeapon",
    "Flamecharm": "Flamecharm", "Frostdraw": "Frostdraw", "Thundercall": "Thundercall",
    "Galebreathe": "Galebreathe", "Shadowcast": "Shadowcast", "Ironsing": "Ironsing", "Bloodrend": "Bloodrend",
}
#: category pseudo-stats: any one member meets the value (reducible), as in pocamind
DWCO_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Weapon": ("LightWeapon", "MediumWeapon", "HeavyWeapon"),
    "Mind": ("Intelligence", "Willpower", "Charisma"),
    "Body": ("Strength", "Fortitude", "Agility"),
    "Attunement": ("Flamecharm", "Frostdraw", "Thundercall", "Galebreathe", "Shadowcast", "Ironsing", "Bloodrend"),
    "Element": ("Flamecharm", "Frostdraw", "Thundercall", "Galebreathe", "Shadowcast", "Ironsing", "Bloodrend"),
}


def power_total(power: int) -> int:
    """Total points invested at power level p (15 p + 15); pocamind writes
    deepwoken.co's "Power 8" as "135s TTL"."""
    return 15 * int(power) + 15


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")


def _dw_stat_clauses(stats: dict) -> List[List[dict]]:
    """{"Galebreathe": 40, "Weapon": 90, "Power": 8} -> CNF clauses."""
    out = []
    for k, v in (stats or {}).items():
        if not isinstance(v, (int, float)) or v <= 0:
            continue
        v = int(v)
        if k == "Power":
            out.append([dict(stats=["Total"], value=power_total(v), strict=True)])
        elif k in ("Total", "TTL"):
            out.append([dict(stats=["Total"], value=v, strict=True)])
        elif k in DWCO_GROUPS:
            out.append([dict(stats=[s], value=v, strict=False) for s in DWCO_GROUPS[k]])
        elif k in DWCO_STATS:
            out.append([dict(stats=[DWCO_STATS[k]], value=v, strict=True)])
        else:
            raise ReqParseError(f"unknown stat {k!r}")
    return out


def _dw_requirements(r: dict):
    """deepwoken.co requirements -> (clauses, prereq groups, conditions).

    stats      AND of single-stat atoms (category / Power pseudo-stats expanded)
    add        [{"stats": [...], "value": v}] sums
    or         alternatives; expressed exactly when every alternative is pure
               stats (CNF by distribution) or a single talent / mantra
               (prerequisite group); otherwise kept as a condition
    talents, mantras   prerequisites (each needed)
    anything else (weaponType, origin, aspect, objectives, quests, slay,
    equipment, outfit, memento, set, murmur, weapon, ...) -> conditions."""
    r = r or {}
    clauses = _dw_stat_clauses(r.get("stats"))
    for a in r.get("add") or []:
        st = [DWCO_STATS.get(x) or x for x in a.get("stats") or []]
        if a.get("value"):
            clauses.append([dict(stats=sorted(set(st)), value=int(a["value"]), strict=False)])
    prereqs = [[f"talent:{_slug(t)}"] for t in r.get("talents") or []]
    prereqs += [[f"mantra:{_slug(m)}"] for m in r.get("mantras") or []]
    cond = {k: v for k, v in r.items() if k not in ("stats", "add", "or", "talents", "mantras") and v}
    alts = r.get("or") or []
    if alts:
        if all(set(a) <= {"stats"} and a.get("stats") for a in alts):
            dnf = [_dw_stat_clauses(a["stats"]) for a in alts]      # each: AND of OR-clauses
            import itertools
            combos = 1
            for d in dnf:
                combos *= max(1, len(d))
            if combos <= 64:
                for pick in itertools.product(*dnf):
                    atoms = []
                    for cl in pick:
                        atoms += cl
                    seen, uniq = set(), []
                    for at in atoms:
                        k = (tuple(at["stats"]), at["value"])
                        if k not in seen:
                            seen.add(k); uniq.append(dict(at, strict=False))
                    clauses.append(uniq)
            else:
                cond["or"] = alts
        elif all(len(a) == 1 and len(a.get("talents") or a.get("mantras") or []) == 1 for a in alts):
            prereqs.append([f"talent:{_slug(a['talents'][0])}" if a.get("talents") else
                            f"mantra:{_slug(a['mantras'][0])}" for a in alts])
        else:
            cond["or"] = alts
    return clauses, prereqs, cond


def _read_dwco(path: str) -> dict:
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        n = next(x for x in z.namelist() if x.endswith(".json"))
        return json.loads(z.read(n))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_deepwokenco(path: str) -> GameData:
    """Load a deepwoken.co dump ({"talents": [...], "mantras": [...], ...}, as
    served by DWCO_URL). Several entries can share a name (e.g. a memento /
    VOI variant of a slot talent): the entry whose _id equals the name wins,
    else the slot-taking one. Unknown stat names -> GameData.errors."""
    from tri.sources.pipeline import normalize_name
    d = _read_dwco(path)
    if not isinstance(d, dict) or "talents" not in d:
        raise ValueError(f"{path}: not a deepwoken.co dump (no 'talents' list)")
    items: Dict[str, GameItem] = {}
    rank: Dict[str, tuple] = {}
    errors = []
    for table, kind in (("talents", Kind.TALENT), ("mantras", Kind.MANTRA)):
        for x in d.get(table) or []:
            if not isinstance(x, dict) or not x.get("name"):
                continue
            name = normalize_name(x["name"])
            try:
                clauses, prereqs, cond = _dw_requirements(x.get("requirements"))
            except ReqParseError as e:
                clauses, prereqs, cond = [], [], {}
                errors.append((name, str(e)))
            slot = bool(x.get("countTowardsTalentTotal", True)) if kind == Kind.TALENT else True
            it = GameItem(name=name, key=_slug(name), kind=kind, rarity=x.get("rarity"),
                          category=x.get("category"), slot=slot, implicit=False,
                          vaulted=bool(x.get("vaulted", False)), reqs=format_reqs(clauses),
                          clauses=clauses, prereqs=prereqs,
                          exclusive=[normalize_name(e) for e in x.get("mutualExclusives") or [] if e],
                          conditions=cond, source="deepwoken.co",
                          mtype=(mantra_type_of(x) if kind == Kind.MANTRA else None))
            r = (kind == Kind.TALENT, str(x.get("_id", "")) == x["name"], slot)
            if name in items and items[name].kind != kind:
                errors.append((name, "name used by both a talent and a mantra; keeping the talent"))
            if name not in items or r > rank[name]:
                items[name], rank[name] = it, r
    meta = {}
    for cand in (os.path.splitext(path)[0] + ".meta.json",):
        if os.path.isfile(cand):
            with open(cand) as f:
                meta = json.load(f)
    src = dict(name="deepwoken.co", path=os.path.abspath(path),
               n_talents=sum(i.kind == Kind.TALENT for i in items.values()),
               n_mantras=sum(i.kind == Kind.MANTRA for i in items.values()), **meta)
    return GameData(items, src, errors)


def fetch_deepwokenco(dest: str, refresh: bool = False, timeout: float = 60.0) -> GameData:
    """Download DWCO_URL once to dest/deepwokenco.json (reused unless refresh)."""
    os.makedirs(dest, exist_ok=True)
    fp = os.path.join(dest, "deepwokenco.json")
    if refresh or not os.path.exists(fp):
        req = urllib.request.Request(DWCO_URL, headers={"User-Agent": "Mozilla/5.0 (tri)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        d = json.loads(data)
        if not isinstance(d, dict) or not d.get("talents"):
            raise ValueError(f"{DWCO_URL}: unexpected response {str(d)[:200]}")
        with open(fp + ".tmp", "wb") as f:
            f.write(data)
        os.replace(fp + ".tmp", fp)
        with open(os.path.join(dest, "deepwokenco.meta.json"), "w") as f:
            json.dump(dict(url=DWCO_URL, fetched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           updated_max=max((x.get("_updatedAt") or "" for x in d["talents"]), default=None)),
                      f, indent=1)
    return load_deepwokenco(fp)


def load_any(path: str) -> GameData:
    """deepwoken.co dump (a .json with a 'talents' list) or pocamind/data (dir / zip)."""
    if os.path.isfile(path) and not zipfile.is_zipfile(path):
        return load_deepwokenco(path)
    try:
        return load_gamedata(path)
    except FileNotFoundError:
        return load_deepwokenco(path)


def merge_gamedata(primary: GameData, *others: GameData) -> GameData:
    """Union by loose name: entries of `primary` win; names only the others know
    are added (with their prerequisite references re-pointed to primary keys
    where the target is known there). Exclusions are united both ways."""
    items = dict(primary.items)
    keys = {f"{it.kind.value}:{name_key(n)}": n for n, it in items.items()}
    added = {}
    for o in others:
        for n, it in o.items.items():
            k = f"{it.kind.value}:{name_key(n)}"
            if k in keys:
                cur = items[keys[k]]
                extra = [e for e in it.exclusive if e not in cur.exclusive]
                if extra:
                    cur = items[keys[k]] = replace_item(cur, exclusive=cur.exclusive + extra)
                if cur.mtype in (None, "Normal") and it.mtype not in (None, "Normal"):   # a specific type beats Normal
                    cur = items[keys[k]] = replace_item(cur, mtype=it.mtype)
                if it.voi_only and not cur.voi_only:     # deepwoken.co's "VOI" means obtainable there, not only there
                    cur = items[keys[k]] = replace_item(cur, voi_only=True)
                locks = {kk: vv for kk, vv in it.conditions.items()
                         if str(kk).lower() in LOCK_CONDITIONS and kk not in cur.conditions}
                if locks:                               # a lock the primary does not state
                    items[keys[k]] = replace_item(cur, conditions={**cur.conditions, **locks})
                continue
            groups = []
            for g in it.prereqs:
                gg = []
                for ref in g:
                    tgt = o.resolve(ref)
                    tk = keys.get(f"{o.items[tgt].kind.value}:{name_key(tgt)}") if tgt else None
                    gg.append(f"{items[tk].kind.value}:{items[tk].key}" if tk else ref)
                groups.append(gg)
            items[n] = replace_item(it, prereqs=groups)
            keys[k] = n
            added[o.source.get("name", o.source.get("repo", "other"))] = \
                added.get(o.source.get("name", o.source.get("repo", "other")), 0) + 1
    src = dict(name="merged", primary=primary.source, others=[o.source for o in others], added=added,
               n_talents=sum(i.kind == Kind.TALENT for i in items.values()),
               n_mantras=sum(i.kind == Kind.MANTRA for i in items.values()))
    return GameData(items, src, list(primary.errors) + [e for o in others for e in o.errors])


LOCK_CONDITIONS = ("oath", "murmur", "origin")


def replace_item(it: GameItem, **kw) -> GameItem:
    from dataclasses import replace as _r
    return _r(it, **kw)


def fetch_all(dest: str, refresh: bool = False, ref: str = "main", log=None) -> GameData:
    """deepwoken.co (primary) merged with pocamind/data (fills gaps). Either
    may fail (network, rate limits); the other is then used alone."""
    got, errs = [], []
    for name, fn in (("deepwoken.co", lambda: fetch_deepwokenco(dest, refresh=refresh)),
                     ("pocamind/data", lambda: fetch_gamedata(dest, ref=ref, refresh=refresh))):
        try:
            got.append(fn())
        except Exception as e:
            errs.append(f"{name}: {e}")
            if log:
                log(f"game data: {name} unavailable ({str(e)[:160]})")
    if not got:
        raise RuntimeError("no game data source available: " + "; ".join(errs))
    gd = merge_gamedata(*got) if len(got) > 1 else got[0]
    if errs:
        gd.source["unavailable"] = errs
    return gd


def coverage(source, gd: GameData, limit: Optional[int] = None, top: int = 50) -> dict:
    """How much of a planner export the game data knows, read straight from the
    raw talents / mantras lists (fast; no stats parsing).

    Returns {records, picks, known_share, loose_matches (names known only by
    loose matching, with counts), unknown: [(name, count, suggestions)]}."""
    from collections import Counter

    from tri.sources.pipeline import iter_records, normalize_name
    cnt: Counter = Counter()                                  # (kind, name)
    n = 0
    it = iter_records(source) if isinstance(source, str) else source
    for _, rec in it:
        n += 1
        for fld, kind in (("talents", "talent"), ("mantras", "mantra")):
            for x in rec.get(fld) or []:
                if isinstance(x, str) and x.strip():
                    cnt[(kind, normalize_name(x))] += 1
        if limit and n >= limit:
            break
    total = sum(cnt.values())
    exact = {k: c for k, c in cnt.items() if k[1] in gd.items and gd.items[k[1]].kind.value == k[0]}
    loose = {k: c for k, c in cnt.items() if k not in exact and gd.lookup(k[1], k[0])}
    unk = Counter({k: c for k, c in cnt.items() if k not in exact and k not in loose})
    return dict(records=n, picks=total, distinct_names=len(cnt),
                known_share=(sum(exact.values()) + sum(loose.values())) / total if total else 1.0,
                loose_matches=sorted(((k[1], gd.lookup(k[1], k[0]), c) for k, c in loose.items()),
                                     key=lambda x: -x[2])[:top],
                unknown_distinct=len(unk), unknown_picks=sum(unk.values()),
                unknown=[(k[1], c, gd.suggest(k[1])) for k, c in unk.most_common(top)],
                unknown_kinds=[k[0] for k, _ in unk.most_common(top)])


def resolve_gamedata(specs: Sequence[str], dest: str, refresh: bool = False, ref: str = "main", log=None):
    """specs: ["auto"] -> pocamind/data (v3.7: every requirement, prerequisite,
    exclusion and lock tri reads is in pocamind/data); "all" -> fetch_all
    (deepwoken.co merged with pocamind/data, the pre-v3.7 auto); ["none"] ->
    None; paths -> load_any, merged in order (first wins); "auto" may be mixed
    with paths."""
    specs = [s for s in specs if s]
    if not specs or specs == ["none"]:
        return None
    parts = []
    for sp in specs:
        if sp in ("auto", "pocamind"):
            parts.append(fetch_gamedata(dest, ref=ref, refresh=refresh))
        elif sp == "all":
            parts.append(fetch_all(dest, refresh=refresh, ref=ref, log=log))
        elif sp in ("deepwoken.co", "dwco"):
            parts.append(fetch_deepwokenco(dest, refresh=refresh))
        else:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"game data {sp} not found")
            parts.append(load_any(sp))
    return parts[0] if len(parts) == 1 else merge_gamedata(*parts)
