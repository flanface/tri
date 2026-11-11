"""Pretraining pipeline for the build-planner export format.

Input: ONE JSON file mapping build id -> build record, in the planner's format
(version 3):

    {"Prjazh0D": {"version": 3, "createdAt": ..., "updatedAt": ...,
                  "author": {"id", "verifiedId", "name"},
                  "stats": {"buildName", "traits", "meta": {Origin, Oath, ...}, "power", ...},
                  "attributes": {"weapon": {...}, "attunement": {...}, "base": {...}},
                  "preShrine":  {"weapon": {...}, "attunement": {...}, "base": {...}},
                  "talents": [...], "mantras": [...],
                  "content": {"mantraModifications": {...}},
                  "meta": {"votes": {"up", "down"}, "views", "saved", "visibility", ...},
                  "weapons", "enchant", ...}, ...}

A top-level array of records, or JSON Lines (one record or {id: record} per
line), are accepted too. The file is streamed, so its size is not limited by RAM.

Stat convention (as specified):
    final   = attributes
    highest = per-stat max(attributes, preShrine)

Output: a `PretrainSet` holding
    builds      CompactBuilds, a lazy Sequence[Build] (about 200 bytes per build)
    weights     per-build sampling weights (dedup x recency x quality x power)
    lineage     group id per build (for held-out splits)
    catalog     the catalog the builds are aligned to (given, or derived)
    pick_rate, lift, lift_mask   availability-conditioned pick and pair statistics (diagnostics)
    report      counts of every drop, unknown names, game-data coverage, ...

Fit with tri.fit(ps.builds, weights=ps.weights), or from the command line with tri.cli.fit."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from array import array
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from tri.core.catalog import Catalog
from tri.core.types import Build, Category, Item, Kind
from tri.sources.defs import GRANTED_TALENTS, ParsedRecord, PipelineConfig

STAT_GROUPS = ("weapon", "attunement", "base")
#: build-level context kept in Build.meta (not used by the model today)
META_FIELDS = ("Origin", "Oath", "Outfit", "Race", "Murmur", "Bell")
# ------------------------------------------------------------ granted talents
# Talents that do not take a talent slot (Deepwoken Wiki, "Talents"): mantra
# level talents, mastery ("Unbounded") talents, oath, race, origin, echo/quest
# and weapon-equipped talents. They are removed from builds' talent lists
# because they are consequences of stats / oath / race / weapon, not choices.
_ATTUNEMENT_HOLDERS = ("Flamecharmer", "Frostdrawer", "Thundercaller", "Galebreather",
                       "Shadowcaster", "Ironsinger", "Bloodrender")
_UNBOUNDED = ("Agility", "Bloodrend", "Flamecharm", "Fortitude", "Frostdraw", "Galebreathe",
              "Heavy Weapons", "Intelligence", "Ironsing", "Light Weapons", "Medium Weapons",
              "Shadowcast", "Strength", "Thundercall", "Willpower", "Charisma")
_RACE = ("Chitin", "Deepfolk", "Feathered Glider", "Echolocator", "Loyalty", "Mark of Jurik",
         "Mark of Ku", "Mark of Nemit", "Maudet", "Molt", "Mothwing Dust", "Navae's Guidance",
         "Nightchild", "Seaborne", "Teachings of the Edenkite", "Versatile")
_ORIGIN = ("Authority Ensign", "Horn of the Authority", "Call of the Deep", "Deepbound Contract",
           "Ignition Card", "Union Pager", "Union Hook", "Mark of the Lone Warrior",
           "Grasp of Eylis", "Voideye", "Voidwalker Contract")
_ECHO_QUEST = ("Thresher Scales", "Chain of Perfection", "Critical Attack", "Engage",
               "Beiruul's Vengence", "Beiruul's Vengeance", "Careful Handling",
               "Drowned Secret: Ethiron's Gaze", "Excavator", "Fisher's Lure", "Harvester",
               "Hook, Line, and Sinker", "Imperium Kata", "Landshark", "Legion Kata",
               "Nanji's Training", "Pleeksty's Will", "Proficient Baiting", "Spell Shout",
               "Supernatural Sense", "Treefelling Blow", "Jus Karita", "Emergency Repairs",
               "Spare Nails")
_WEAPON = ("Assassin's Strike", "Battle Frenzy", "Belief", "Blitz", "Blood Drinker",
           "Bulwark Breaker", "Clutch Recovery", "Cruel Counter", "Curved Edge", "Defensive Stance",
           "Expeditionary Evasiveness", "Expeditious Combo", "Extended Trauma", "Finishing Slay",
           "Forced Fulfilment", "Pierce Through", "Pugnacious", "Punishing Riposte", "Rat's Spirit",
           "Rosen's Technique", "Sacrificial Boon", "Sovereign's Counter", "Spinal Splinter",
           "Static Beatdown", "Swift Strike", "Vital Wound", "Blade Threading", "Fortify",
           "Hex Bullet", "Monster Slayer", "Night Night", "Reforged Alloy", "Reverse Necrosis",
           "Sadistic Split", "Shatter", "Subzero")
for _h in _ATTUNEMENT_HOLDERS:
    for _p in ("", "Adept ", "Expert ", "Master "):
        GRANTED_TALENTS[_p + _h] = "mantra_level"
for _g, _names in (("race", _RACE), ("origin", _ORIGIN), ("echo_quest", _ECHO_QUEST), ("weapon", _WEAPON)):
    for _n in _names:
        GRANTED_TALENTS[_n] = _g
for _u in _UNBOUNDED:
    GRANTED_TALENTS[_u + " Unbounded"] = "mastery"


def granted_group(name: str, cfg: "PipelineConfig") -> Optional[str]:
    """Source group if a talent name is granted (no slot) under cfg, else None.
    cfg.granted_known (game data) wins over the name lists and patterns."""
    if name in cfg.granted_known:
        return cfg.granted_known[name]
    g = cfg.granted_talents.get(name)
    if g:
        return g
    for p, g in cfg.granted_prefixes.items():
        if name.startswith(p):
            return g
    for sfx, g in cfg.granted_suffixes.items():
        if name.endswith(sfx):
            return g
    return None
# ============================================================ streaming JSON


def iter_records(path: str, chunk_size: int = 1 << 22) -> Iterator[Tuple[str, dict]]:
    """Yield (build_id, record) from a top-level object, a top-level array, or
    JSON Lines. Memory use is bounded by the largest single record."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        first = head.lstrip()[:1]
        lines = head.lstrip().split("\n", 1)
        if path.endswith((".jsonl", ".ndjson")) or (first == "{" and _looks_jsonl(lines)):
            for n, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield from _unwrap(obj, f"line{n}")
            return
        yield from _stream_top(f, chunk_size)


def _looks_jsonl(lines: List[str]) -> bool:
    try:
        json.loads(lines[0])
        return len(lines) > 1
    except Exception:
        return False


def _unwrap(obj, fallback_id):
    if isinstance(obj, dict) and "talents" not in obj and len(obj) == 1:
        (k, v), = obj.items()
        yield str(k), v
    elif isinstance(obj, dict):
        yield str(obj.get("id") or obj.get("_id") or fallback_id), obj


def _stream_top(f, chunk_size: int):
    dec = json.JSONDecoder()
    buf, pos, eof = "", 0, False

    def fill():
        nonlocal buf, pos, eof
        more = f.read(chunk_size)
        if not more:
            eof = True
        buf = buf[pos:] + more
        pos = 0

    def skip_ws():
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf) or eof:
                return
            fill()

    def value():
        nonlocal pos
        while True:
            try:
                v, end = dec.raw_decode(buf, pos)
                if end == len(buf) and not eof:      # could be truncated: read more first
                    raise json.JSONDecodeError("boundary", buf, end)
                pos = end
                return v
            except json.JSONDecodeError:
                if eof:
                    raise
                fill()

    fill()
    skip_ws()
    opener = buf[pos]
    pos += 1
    if opener not in "{[":
        raise ValueError("expected a JSON object or array at top level")
    closer = "}" if opener == "{" else "]"
    n = 0
    while True:
        skip_ws()
        if pos >= len(buf) or buf[pos] == closer:
            return
        if opener == "{":
            key = value()
            skip_ws()
            while pos >= len(buf) and not eof:
                fill()
            if buf[pos] != ":":
                raise ValueError(f"malformed JSON near key {key!r}")
            pos += 1
            skip_ws()
            yield str(key), value()
        else:
            yield from _unwrap(value(), f"item{n}")
        n += 1
        if pos > chunk_size:
            buf, pos = buf[pos:], 0
# ============================================================ parsing one record


def normalize_name(s: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """NFKC, straight quotes, single spaces, then aliases."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2019", "'").replace("\u2018", "'")
    s = " ".join(s.split())
    return (aliases or {}).get(s, s)


def _name_key(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum())


def _ts(x) -> Optional[float]:
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _stat_block(block: dict, cfg: PipelineConfig) -> Optional[Dict[str, int]]:
    out: Dict[str, int] = {}
    for g in STAT_GROUPS:
        for k, v in (block.get(g) or {}).items():
            name = cfg.stat_aliases.get(k)
            if name is None:
                continue
            if not isinstance(v, (int, float)) or v < 0 or v > cfg.max_stat:
                return None
            out[name] = int(v)
    return out


def parse_record(build_id: str, rec: dict, cfg: PipelineConfig) -> Tuple[Optional[ParsedRecord], str]:
    """Parse one planner record. Returns (ParsedRecord, "ok") or (None, drop_reason)."""
    if not isinstance(rec, dict):
        return None, "not_an_object"
    if cfg.accept_versions is not None and rec.get("version") not in cfg.accept_versions:
        return None, "version"
    m = rec.get("meta") or {}
    if not cfg.include_private and (m.get("private") or m.get("visibility") not in (None, "public")):
        return None, "private"
    st = rec.get("stats") or {}
    power = int(st.get("power") or 0)
    if power < cfg.min_power and not (rec.get("source") == "dwt" and not st.get("power")):  # dwt.sh records no power
        return None, "power"
    attrs = rec.get("attributes")
    if not isinstance(attrs, dict):
        return None, "no_attributes"
    final = _stat_block(attrs, cfg)
    pre = _stat_block(rec.get("preShrine") or {}, cfg)
    mid = [_stat_block(s or {}, cfg) for s in (rec.get("midStages") or [])]   # dwt.sh middle stages
    if final is None or pre is None or any(s is None for s in mid):
        return None, "bad_stats"
    highest = {s: max(final.get(s, 0), pre.get(s, 0)) for s in set(final) | set(pre)}

    def clean(names, kind):
        seen, out = set(), []
        for n in names or []:
            if not isinstance(n, str) or not n.strip():
                continue
            n = normalize_name(n, cfg.item_aliases)
            if cfg.canonical_names:
                n = cfg.canonical_names.get(f"{kind}:{_name_key(n)}", n)
            if n in seen or n in cfg.exclude_items or n.startswith(tuple(cfg.exclude_prefixes) or ("\0",)):
                continue
            seen.add(n); out.append(n)
        return out

    talents = clean(rec.get("talents"), "talent")
    granted: Tuple[str, ...] = ()
    if cfg.exclude_granted:
        granted = tuple(t for t in talents if granted_group(t, cfg))
        if granted:
            gs = set(granted)
            talents = [t for t in talents if t not in gs]
    raw_m = rec.get("mantras")
    if not raw_m and cfg.mantra_fallback:
        raw_m = list(((rec.get("content") or {}).get("mantraModifications") or {}).keys())
    mantras = clean(raw_m, "mantra")
    if len(talents) < cfg.min_talents:
        return None, "few_talents"
    if len(mantras) < cfg.min_mantras:
        return None, "few_mantras"
    author = rec.get("author") or {}
    aid = author.get("verifiedId") or author.get("id")
    bname = normalize_name(st.get("buildName") or "").lower()
    lineage = f"a:{aid}|{bname}" if aid else f"b:{build_id}"
    votes = m.get("votes") or {}
    meta = {k: v for k, v in (st.get("meta") or {}).items() if k in META_FIELDS and v not in (None, "", "None")}
    for t in list(rec.get("talents") or []):          # an "Oath: X" / "Murmur: X" talent states the oath
        for prefix, k in (("Oath:", "Oath"), ("Murmur:", "Murmur")):
            if isinstance(t, str) and t.startswith(prefix) and t[len(prefix):].strip() and k not in meta:
                meta[k] = t[len(prefix):].strip()
    meta.update(power=power, weapon=rec.get("weapons") or None, enchant=rec.get("enchant") or None,
                name=st.get("buildName") or None, version=rec.get("version"),
                source=rec.get("source") or None, stages=list(rec.get("stageNames") or []) or None)
    if re.match(r"\s*tri \w+ of \S+", str(st.get("buildDescription") or "")):
        meta["made_by"] = "tri"                         # saved by the bot (save_dwt), v2.25.1
    updated = _ts(rec.get("updatedAt")) or _ts(rec.get("createdAt"))
    b = Build(highest, final, talents, mantras, pre_stats=pre, build_id=build_id, timestamp=updated,
              author=str(aid) if aid else None, popularity=float(_count(m.get("views"))),
              meta={k: v for k, v in meta.items() if v is not None},
              mid_stages=[s for s in mid if s] or None)
    if not isinstance(votes, dict):
        votes = {"up": votes}
    return ParsedRecord(b, updated, _count(votes.get("up")), _count(votes.get("down")),
                        _count(m.get("views")), _count(m.get("saved")), power, lineage,
                        granted), "ok"


def _count(x) -> int:
    """Non-negative int from a count field (int, float, numeric string, or a
    list / dict of voters); anything else -> 0."""
    if isinstance(x, (list, dict)):
        return len(x)
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return 0
    return min(max(v, 0), 2_000_000_000)


# ============================================================ compact storage
#: Build.meta keys stored per build (as integer codes)
META_KEYS = META_FIELDS + ("weapon", "enchant", "power", "version")


class _Vocab:
    def __init__(self, names: Sequence[str] = ()):
        self.names: List[str] = list(names)
        self.index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

    def get(self, n) -> int:
        i = self.index.get(n)
        if i is None:
            i = self.index[n] = len(self.names)
            self.names.append(n)
        return i


def _h64(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") >> 1


class CompactBuilds(Sequence):
    """Lazy Sequence[Build] backed by flat arrays (about 200 bytes per build).

    builds[i] constructs a Build on demand, so Tri.fit can sample
    from millions of builds without holding Build objects in memory.
    Arrays: hi, fi [B,S] uint8 (catalog.stats order); items (catalog indices,
    talents first per build); offsets [B+1]; n_talents [B]; timestamps [B]
    (unix s, NaN if unknown); meta_codes [B,K] int32 into meta_vocab (-1 = absent),
    or None when PipelineConfig.keep_meta is False."""

    def __init__(self, stats: Sequence[str], item_names: Sequence[str],
                 hi: np.ndarray, fi: np.ndarray, items: np.ndarray, offsets: np.ndarray,
                 n_talents: np.ndarray, ids: Sequence[str], timestamps: np.ndarray,
                 meta_codes: Optional[np.ndarray] = None, meta_vocab: Optional[List[List[str]]] = None,
                 rulesets: Optional[Sequence[str]] = None, pr: Optional[np.ndarray] = None):
        self.stats = tuple(stats)
        self.item_names = list(item_names)
        self.hi, self.fi, self.items, self.offsets = hi, fi, items, offsets
        # pre-shrine stats; sets saved before v2.3 have none, and highest stats stand in
        self.pr = pr if pr is not None else hi
        self.n_talents, self.ids, self.timestamps = n_talents, list(ids), timestamps
        self.meta_codes, self.meta_vocab = meta_codes, meta_vocab
        self.rulesets = list(rulesets) if rulesets is not None else None

    def __len__(self) -> int:
        return len(self.ids)

    def meta(self, i: int) -> Optional[dict]:
        """Build.meta of build i (oath, murmur, bell, power, weapon, ...)."""
        if self.meta_codes is None:
            return None
        out = {}
        for k, (key, c) in enumerate(zip(META_KEYS, self.meta_codes[i])):
            if c >= 0:
                v = self.meta_vocab[k][c]
                out[key] = int(v) if key in ("power", "version") and v.lstrip("-").isdigit() else v
        return out

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        it = self.items[self.offsets[i]:self.offsets[i + 1]]
        nt = int(self.n_talents[i])
        names = [self.item_names[k] for k in it]
        st = self.stats
        ts = float(self.timestamps[i])
        return Build({s: int(v) for s, v in zip(st, self.hi[i]) if v},
                     {s: int(v) for s, v in zip(st, self.fi[i]) if v},
                     names[:nt], names[nt:], pre_stats={s: int(v) for s, v in zip(st, self.pr[i]) if v},
                     build_id=self.ids[i],
                     ruleset=self.rulesets[i] if self.rulesets else None,
                     timestamp=None if math.isnan(ts) else ts, meta=self.meta(i))

    def item_indices(self, i: int) -> np.ndarray:
        """Catalog indices of build i (talents first)."""
        return self.items[self.offsets[i]:self.offsets[i + 1]]

    def gather(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(items, local_row) for the builds idx, concatenated in that order."""
        idx = np.asarray(idx, dtype=np.int64)
        lens = (self.offsets[idx + 1] - self.offsets[idx])
        tot = int(lens.sum())
        local = np.repeat(np.arange(len(idx), dtype=np.int64), lens)
        start = np.cumsum(lens) - lens
        pos = self.offsets[idx][local] + (np.arange(tot, dtype=np.int64) - start[local])
        return self.items[pos].astype(np.int64), local

    def chunks(self, order: Optional[np.ndarray] = None, max_entries: int = 8_000_000):
        """Yield index arrays of consecutive builds (in `order`) with at most
        max_entries item entries each; bounds memory for whole-set passes."""
        order = np.arange(len(self)) if order is None else np.asarray(order)
        lens = np.diff(self.offsets)[order]
        cum = np.cumsum(lens)
        a, base = 0, 0
        while a < len(order):
            b = int(np.searchsorted(cum, base + max_entries, side="right"))
            b = max(b, a + 1)
            yield order[a:b]
            base = int(cum[b - 1]); a = b

    def subset(self, idx: Sequence[int]) -> "CompactBuilds":
        """New CompactBuilds with only these builds, in this order."""
        idx = np.asarray(idx, dtype=np.int64)
        items, local = self.gather(idx)
        lens = np.bincount(local, minlength=len(idx)) if len(idx) else np.zeros(0, np.int64)
        off = np.zeros(len(idx) + 1, dtype=np.int64); np.cumsum(lens, out=off[1:])
        return CompactBuilds(self.stats, self.item_names, self.hi[idx], self.fi[idx],
                             items.astype(self.items.dtype), off, self.n_talents[idx],
                             [self.ids[i] for i in idx], self.timestamps[idx],
                             self.meta_codes[idx] if self.meta_codes is not None else None,
                             self.meta_vocab, [self.rulesets[i] for i in idx] if self.rulesets else None,
                             self.pr[idx])
# ============================================================ result


@dataclass
class PretrainSet:
    """Everything the pipeline produces. See module docstring."""
    builds: CompactBuilds
    weights: np.ndarray
    lineage: np.ndarray            # int group id per build
    catalog: Catalog
    pick_rate: np.ndarray          # [N], aligned to catalog.ids
    lift: np.ndarray               # [N,N]
    lift_mask: np.ndarray          # [N,N] bool
    report: dict

    def split(self, test_frac: float = 0.05, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """(train_idx, test_idx) with whole lineages on one side."""
        rng = np.random.default_rng(seed)
        groups = np.unique(self.lineage)
        test_g = rng.choice(groups, max(1, int(len(groups) * test_frac)), replace=False)
        mask = np.isin(self.lineage, test_g)
        return np.nonzero(~mask)[0], np.nonzero(mask)[0]

    def save(self, path: str) -> None:
        """Write builds.npz, index.json, catalog.json and report.json to a directory."""
        os.makedirs(path, exist_ok=True)
        b = self.builds
        np.savez_compressed(os.path.join(path, "builds.npz"), hi=b.hi, fi=b.fi, pr=b.pr, items=b.items,
                            offsets=b.offsets, n_talents=b.n_talents, timestamps=b.timestamps,
                            meta_codes=b.meta_codes if b.meta_codes is not None else np.zeros((0, 0)),
                            weights=self.weights, lineage=self.lineage, pick_rate=self.pick_rate,
                            lift=self.lift.astype(np.float32), lift_mask=self.lift_mask)
        with open(os.path.join(path, "index.json"), "w") as f:
            json.dump(dict(stats=list(b.stats), item_names=b.item_names, ids=b.ids,
                           rulesets=b.rulesets, meta_vocab=b.meta_vocab), f)
        catalog_to_json(self.catalog, os.path.join(path, "catalog.json"))
        with open(os.path.join(path, "report.json"), "w") as f:
            json.dump(self.report, f, indent=2, default=str)

    @staticmethod
    def load(path: str, catalog: Optional[Catalog] = None) -> "PretrainSet":
        """Reload a saved PretrainSet (optionally with your own Catalog object)."""
        z = np.load(os.path.join(path, "builds.npz"))
        with open(os.path.join(path, "index.json")) as f:
            ix = json.load(f)
        mc = z["meta_codes"]
        b = CompactBuilds(ix["stats"], ix["item_names"], z["hi"], z["fi"], z["items"], z["offsets"],
                          z["n_talents"], ix["ids"], z["timestamps"],
                          mc if mc.size or ix.get("meta_vocab") else None, ix.get("meta_vocab"),
                          ix.get("rulesets"), z["pr"] if "pr" in z.files else None)
        cat = catalog or Catalog.from_json(os.path.join(path, "catalog.json"))
        with open(os.path.join(path, "report.json")) as f:
            rep = json.load(f)
        return PretrainSet(b, z["weights"], z["lineage"], cat, z["pick_rate"],
                           z["lift"].astype(np.float64), z["lift_mask"], rep)


def catalog_to_json(cat: Catalog, path: str) -> None:
    """Write a catalog in the format Catalog.from_json reads (edit it by hand freely)."""
    with open(path, "w") as f:
        json.dump(cat.to_dict(), f, indent=1)
# ============================================================ main entry point


def _group_first(keys: np.ndarray, ts: np.ndarray, keep_mask: np.ndarray):
    """Among rows with keep_mask, group by key; return (is_newest_in_group, group_id)
    where group_id indexes unique keys (rows outside keep_mask get -1)."""
    idx = np.nonzero(keep_mask)[0]
    t = np.nan_to_num(ts[idx], nan=-np.inf)
    order = idx[np.lexsort((-t, keys[idx]))]
    k = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    gid_sorted = np.cumsum(first) - 1
    newest = np.zeros(len(keys), dtype=bool); newest[order[first]] = True
    gid = np.full(len(keys), -1, dtype=np.int64); gid[order] = gid_sorted
    return newest, gid


def prepare_pretraining(source, catalog: Optional[Catalog] = None,
                        cfg: Optional[PipelineConfig] = None, ruleset: str = "unknown",
                        limit: Optional[int] = None, progress: int = 0,
                        gamedata=None, keep_groups: Sequence[str] = ()) -> PretrainSet:
    """Parse, filter, deduplicate, weight and summarise builds.

    source:  path to the JSON file, or an iterable of (build_id, record).
    catalog: your catalog. If given, builds are aligned to it and unknown names
             are dropped and counted.
    gamedata: a tri.gamedata.GameData (pocamind/data). Granted talents are taken
             from it (count_towards_talent_total / implicit; the name rules only
             cover names it does not know), and when catalog is None the
             catalog is built from it: known, slot-taking items picked >=
             min_item_count times, with their real requirements,
             prerequisites and exclusions. Names it does not know are dropped
             as unknown (report["not_in_gamedata"]).
    keep_groups: granted groups to treat as choices, e.g. ("faction",).
             With neither catalog nor gamedata, a catalog without any
             requirements is derived from the builds (kind by majority).
    limit:   stop after this many records (for trials).
    progress: print a line to stderr every `progress` records (0 = silent).

    Steps: parse -> filter -> exact dedup (newest kept, copies counted, votes
    summed) -> revision collapse (same author id + build name, newest kept) ->
    catalog alignment -> weights -> statistics."""
    cfg = cfg or PipelineConfig()
    if gamedata is not None:
        cfg = replace(cfg, granted_known={**gamedata.granted_known(keep_groups), **cfg.granted_known},
                      canonical_names={**gamedata.canonical_names(), **cfg.canonical_names})
    stats = tuple(catalog.stats) if catalog is not None else tuple(cfg.stats)
    S = len(stats)
    sidx = {s: i for i, s in enumerate(stats)}
    recs = iter_records(source) if isinstance(source, (str, os.PathLike)) else iter(source)

    vocab = _Vocab()
    kind_votes: List[List[int]] = []                 # per vocab id: [as talent, as mantra]
    meta_vocab = [_Vocab() for _ in META_KEYS]
    hi_a, fi_a, pr_a = array("B"), array("B"), array("B")
    items_a, offs = array("I"), array("q", [0])
    nt_a, ts_a = array("H"), array("d")
    up_a, down_a, saved_a, power_a = array("i"), array("i"), array("i"), array("H")
    sig_a, lin_a, meta_a = array("q"), array("q"), array("i")
    ids: List[str] = []
    drops: Counter = Counter()
    granted_seen: Counter = Counter()
    n_in = 0
    for bid, rec in recs:
        if limit and n_in >= limit:
            break
        n_in += 1
        if progress and n_in % progress == 0:
            print(f"[pipeline] {n_in:,} records, {len(ids):,} parsed", file=sys.stderr)
        pr, why = parse_record(bid, rec, cfg)
        if pr is None:
            drops[why] += 1
            continue
        b = pr.build
        granted_seen.update(pr.granted)
        hv = [0] * S; fv = [0] * S; pv = [0] * S
        for st, v in b.highest_stats.items():
            if st in sidx: hv[sidx[st]] = v
        for st, v in b.final_stats.items():
            if st in sidx: fv[sidx[st]] = v
        for st, v in (b.pre_stats or b.final_stats).items():
            if st in sidx: pv[sidx[st]] = v
        t_idx = [vocab.get(n) for n in b.talents]
        m_idx = [vocab.get(n) for n in b.mantras]
        while len(kind_votes) < len(vocab.names):
            kind_votes.append([0, 0])
        for t in t_idx: kind_votes[t][0] += 1
        for t in m_idx: kind_votes[t][1] += 1
        hi_a.extend(hv); fi_a.extend(fv); pr_a.extend(pv)
        items_a.extend(t_idx); items_a.extend(m_idx)
        offs.append(offs[-1] + len(t_idx) + len(m_idx))
        nt_a.append(len(t_idx)); ts_a.append(pr.updated if pr.updated is not None else float("nan"))
        up_a.append(pr.up); down_a.append(pr.down); saved_a.append(pr.saved); power_a.append(min(pr.power, 65535))
        sig_a.append(_h64(json.dumps([sorted(t_idx), sorted(m_idx), hv, fv])))
        lin_a.append(_h64(pr.lineage) if pr.lineage.startswith("a:") else -1)
        if cfg.keep_meta:
            mt = b.meta or {}
            for k, key in enumerate(META_KEYS):
                v = mt.get(key)
                meta_a.append(-1 if v is None else meta_vocab[k].get(str(v)))
        ids.append(sys.intern(b.build_id) if len(b.build_id) < 64 else b.build_id)

    B0 = len(ids)
    hi = np.frombuffer(hi_a, dtype=np.uint8).reshape(B0, S).copy(); del hi_a
    fi = np.frombuffer(fi_a, dtype=np.uint8).reshape(B0, S).copy(); del fi_a
    pr_ = np.frombuffer(pr_a, dtype=np.uint8).reshape(B0, S).copy(); del pr_a
    items_raw = np.frombuffer(items_a, dtype=np.uint32).copy(); del items_a
    offsets = np.frombuffer(offs, dtype=np.int64).copy(); del offs
    n_t = np.frombuffer(nt_a, dtype=np.uint16).astype(np.int64)
    ts = np.frombuffer(ts_a, dtype=np.float64).copy()
    up = np.frombuffer(up_a, dtype=np.int32).astype(np.float64)
    down = np.frombuffer(down_a, dtype=np.int32).astype(np.float64)
    saved = np.frombuffer(saved_a, dtype=np.int32).astype(np.float64)
    power = np.frombuffer(power_a, dtype=np.uint16).astype(np.float64)
    sig = np.frombuffer(sig_a, dtype=np.int64).copy()
    lin = np.frombuffer(lin_a, dtype=np.int64).copy()
    meta_codes = (np.frombuffer(meta_a, dtype=np.int32).reshape(B0, len(META_KEYS)).copy()
                  if cfg.keep_meta else None)
    rep: dict = dict(records=n_in, parsed=B0, drops=dict(drops),
                     granted_excluded={n: dict(count=c, group=granted_group(n, cfg))
                                       for n, c in granted_seen.most_common()})

    # ---------------------------------------------------- dedup and revisions
    alive = np.ones(B0, dtype=bool)
    dup = np.ones(B0)
    if cfg.dedup and B0:
        newest, gid = _group_first(sig, ts, alive)
        size = np.bincount(gid, minlength=gid.max() + 1).astype(np.float64)
        sums = [np.bincount(gid, weights=x, minlength=len(size)) for x in (up, down, saved)]
        keep = np.nonzero(newest)[0]
        dup[keep] = size[gid[keep]]
        up[keep], down[keep], saved[keep] = (x[gid[keep]] for x in sums)
        rep["drops"]["duplicate"] = int(B0 - newest.sum())
        alive &= newest
    if cfg.collapse_revisions and B0:
        has = alive & (lin >= 0)
        newest, _ = _group_first(lin, ts, has)
        rep["drops"]["older_revision"] = int(has.sum() - newest.sum())
        alive &= ~has | newest
    # lineage for held-out splits: author+name if known, else the dedup signature
    lineage_key = np.where(lin >= 0, lin, sig)

    # ---------------------------------------------------- catalog alignment
    V = len(vocab.names)
    builds_raw = CompactBuilds(stats, vocab.names, hi, fi, items_raw, offsets, n_t, ids, ts, pr=pr_)
    counts = np.zeros(V)
    for ch in builds_raw.chunks(np.nonzero(alive)[0]):
        it, _ = builds_raw.gather(ch)
        counts += np.bincount(it, minlength=V)
    kind_of = [Kind.TALENT if kv[0] >= kv[1] else Kind.MANTRA for kv in kind_votes]
    rep["kind_collisions"] = [vocab.names[v] for v, kv in enumerate(kind_votes) if kv[0] and kv[1]][:50]
    if catalog is None and gamedata is not None:
        unk = Counter({vocab.names[v]: int(counts[v]) for v in range(V)
                       if counts[v] > 0 and vocab.names[v] not in gamedata})
        rep["not_in_gamedata"] = [(n, c, gamedata.suggest(n)) for n, c in unk.most_common(100)]
        tot = float(counts.sum())
        rep["gamedata_coverage"] = dict(known_pick_share=(1 - sum(unk.values()) / tot) if tot else 1.0,
                                        unknown_distinct=len(unk), unknown_picks=int(sum(unk.values())))
        keep = [vocab.names[v] for v in range(V) if counts[v] >= cfg.min_item_count and vocab.names[v] in gamedata]
        keep += [n for n in (gamedata.lookup(x) or x for x in cfg.keep_items) if n in gamedata and n not in set(keep)]
        catalog = gamedata.catalog(keep, stats=stats, ruleset=ruleset, keep_groups=keep_groups, report=rep)
        rep["rare_items_dropped"] = int(sum(1 for v in range(V) if 0 < counts[v] < cfg.min_item_count))
        rep["kind_mismatch"] = [n for v, n in enumerate(vocab.names) if n in catalog.index
                                and kind_votes[v][catalog.items[n].kind != Kind.TALENT] == 0][:50]
        rep["gamedata"] = dict(gamedata.source, parse_errors=len(gamedata.errors))
    elif catalog is None:
        keep_v = [v for v in range(V) if counts[v] >= cfg.min_item_count]
        catalog = Catalog({vocab.names[v]: Item(vocab.names[v], kind_of[v], Category.UNKNOWN) for v in keep_v},
                          stats=stats, ruleset=ruleset)
        rep["rare_items_dropped"] = int(sum(1 for v in range(V) if 0 < counts[v] < cfg.min_item_count))
        rep["warning"] = "catalog derived without requirements; pass gamedata= for known requirements"
    # every name seen, for item diagnostics (tri.cli.fit writes it to item_counts.json)
    rep["item_counts"] = {vocab.names[v]: dict(count=int(counts[v]),
                                               in_gamedata=(vocab.names[v] in gamedata) if gamedata is not None else None)
                          for v in range(V) if counts[v] > 0}
    rep["min_item_count"] = cfg.min_item_count
    v2c = np.full(V + 1, -1, dtype=np.int64)
    for v, n in enumerate(vocab.names):
        if n in catalog.index:
            v2c[v] = catalog.index[n]
    rep["unknown_items_top"] = Counter({vocab.names[v]: int(counts[v]) for v in range(V)
                                        if v2c[v] < 0 and counts[v] > 0}).most_common(50)

    # remap chunk by chunk; drop unknown names; drop builds with too many unknowns
    out_items, new_len, new_nt = [], np.zeros(B0, dtype=np.int64), np.zeros(B0, dtype=np.int64)
    n_unknown_drop = n_few = 0
    for ch in builds_raw.chunks(np.nonzero(alive)[0]):
        it, local = builds_raw.gather(ch)
        mapped = v2c[it]
        known = mapped >= 0
        lens = np.bincount(local, minlength=len(ch))
        start = np.cumsum(lens) - lens
        is_tal = (np.arange(len(it)) - start[local]) < n_t[ch][local]
        unk = np.bincount(local, weights=~known, minlength=len(ch)) / np.maximum(lens, 1)
        kl = np.bincount(local, weights=known, minlength=len(ch)).astype(np.int64)
        kt = np.bincount(local, weights=known & is_tal, minlength=len(ch)).astype(np.int64)
        bad_u = unk > cfg.max_unknown_frac
        bad_f = ~bad_u & ((kt < cfg.min_talents) | (kl - kt < cfg.min_mantras))
        n_unknown_drop += int(bad_u.sum()); n_few += int(bad_f.sum())
        ok_b = ~(bad_u | bad_f)
        alive[ch[~ok_b]] = False
        sel = known & ok_b[local]
        out_items.append(mapped[sel].astype(np.uint16 if len(catalog) < 65535 else np.uint32))
        new_len[ch] = np.where(ok_b, kl, 0); new_nt[ch] = np.where(ok_b, kt, 0)
    rep["drops"]["unknown_items"] = n_unknown_drop
    rep["drops"]["few_known_items"] = n_few
    idx = np.nonzero(alive)[0]
    items_c = np.concatenate(out_items) if out_items else np.zeros(0, dtype=np.uint16)
    del builds_raw, items_raw, out_items
    off_c = np.zeros(len(idx) + 1, dtype=np.int64); np.cumsum(new_len[idx], out=off_c[1:])

    rulesets = None
    if cfg.patches:
        bounds = np.array([_ts(d) for d, _ in cfg.patches])
        names = [n for _, n in cfg.patches]
        pos = np.searchsorted(bounds, np.nan_to_num(ts[idx], nan=-np.inf), side="right") - 1
        rulesets = [names[p] if p >= 0 else "pre-" + names[0] for p in pos]
    builds = CompactBuilds(stats, catalog.ids, hi[idx], fi[idx], items_c, off_c, new_nt[idx],
                           [ids[i] for i in idx], ts[idx],
                           meta_codes[idx] if meta_codes is not None else None,
                           [v.names for v in meta_vocab] if meta_codes is not None else None, rulesets,
                           pr_[idx])
    del hi, fi, pr_, ids

    # ---------------------------------------------------- weights
    tsa = ts[idx]
    if cfg.reference_time:
        ref = _ts(cfg.reference_time)
    else:
        ref = float(np.nanmax(tsa)) if np.isfinite(tsa).any() else 0.0
    age = np.nan_to_num((ref - tsa) / 86400.0, nan=cfg.half_life_days * 4).clip(min=0)
    w_rec = np.maximum(cfg.recency_floor, 0.5 ** (age / cfg.half_life_days))
    u, d_, sv = up[idx], down[idx], saved[idx]
    u, d_, sv = np.maximum(u, 0), np.maximum(d_, 0), np.maximum(sv, 0)
    q = (1 + cfg.vote_strength * np.log1p(u + sv)) * ((u + 1) / (u + d_ + 2)) / 0.5
    w = w_rec * np.clip(q, *cfg.quality_clip) * dup[idx] ** cfg.dup_alpha
    if cfg.power_weight:
        w = w * np.clip(power[idx] / cfg.max_power, 0.05, 1.0)
    if rulesets and cfg.ruleset_weights:
        w = w * np.array([cfg.ruleset_weights.get(r, 1.0) for r in rulesets])
    w = w / max(w.mean(), 1e-12) if len(w) else w
    _, lineage = np.unique(lineage_key[idx], return_inverse=True)
    lineage = lineage.reshape(-1).astype(np.int64)

    # ---------------------------------------------------- statistics
    pick, lift, lmask, viol = pair_statistics(builds, catalog, w if cfg.stats_use_weights else None,
                                              cfg.min_pair_count, cfg.min_joint_avail)
    rep.update(
        kept=int(len(idx)), n_items=len(catalog), lineages=int(lineage.max() + 1) if len(idx) else 0,
        weight_summary=dict(min=float(w.min()) if len(w) else 0.0, max=float(w.max()) if len(w) else 0.0,
                            effective_n=float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)) if len(w) else 0.0),
        auto_granted_candidates=[(catalog.ids[i], round(float(pick[i]), 4)) for i in np.argsort(-pick)
                                 if pick[i] >= cfg.auto_grant_rate][:100],
        requirement_violations=viol, trusted_pairs=int(lmask.sum() // 2),
    )
    ps = PretrainSet(builds, w, lineage, catalog, pick, lift, lmask, rep)
    rep["meta_bound_candidates"] = meta_bound_candidates(ps, cfg)
    return ps


def meta_bound_candidates(ps: "PretrainSet", cfg: Optional[PipelineConfig] = None) -> List[dict]:
    """Talents whose pickers almost all share one value of a meta field
    (cfg.meta_bound_fields, default Oath / Race / Origin). These are usually
    granted talents not in GRANTED_TALENTS (e.g. oath progression talents),
    but can be slot talents that require an oath: audit before excluding.
    Needs Build.meta codes (cfg.keep_meta). Returns
    [{"item", "field", "value", "purity", "support", "value_share"}], sorted by support."""
    cfg = cfg or PipelineConfig()
    b = ps.builds
    if b.meta_codes is None or not len(b) or not b.meta_vocab:
        return []
    N = len(ps.catalog)
    tal = np.zeros(N, dtype=bool)
    for i, iid in enumerate(ps.catalog.ids):
        tal[i] = ps.catalog.items[iid].kind == Kind.TALENT
    out = []
    for f in cfg.meta_bound_fields:
        if f not in META_KEYS:
            continue
        k = META_KEYS.index(f)
        codes = b.meta_codes[:, k]
        V = len(b.meta_vocab[k])
        if V == 0:
            continue
        share = np.bincount(codes[codes >= 0], minlength=V) / max(len(b), 1)
        M = np.zeros((N, V + 1))
        for ch in b.chunks():
            it, local = b.gather(ch)
            c = codes[ch][local]
            np.add.at(M, (it, np.where(c >= 0, c, V)), 1.0)
        sup = M.sum(1)
        top = M[:, :V].argmax(1)
        pur = M[np.arange(N), top] / np.maximum(sup, 1)
        for i in np.nonzero(tal & (sup >= cfg.min_item_count) & (pur >= cfg.meta_bound_purity))[0]:
            if share[top[i]] < 0.5:                      # value is not simply the majority
                out.append(dict(item=ps.catalog.ids[i], field=f, value=b.meta_vocab[k][top[i]],
                                purity=round(float(pur[i]), 4), support=int(sup[i]),
                                value_share=round(float(share[top[i]]), 4)))
    out.sort(key=lambda d: -d["support"])
    return out


def exclude_items(ps: "PretrainSet", names: Sequence[str], cfg: Optional[PipelineConfig] = None) -> "PretrainSet":
    """New PretrainSet without these items: removed from the catalog and from
    every build, then pick rates and lift recomputed. Builds and weights are
    kept (no re-parse). Use for granted talents found after preparation."""
    cfg = cfg or PipelineConfig()
    drop = set(names) & set(ps.catalog.ids)
    if not drop:
        return ps
    cat = Catalog({k: v for k, v in ps.catalog.items.items() if k not in drop},
                  stats=ps.catalog.stats, ruleset=ps.catalog.ruleset, mantra_locks=ps.catalog.mantra_locks,
                  lock_fields=ps.catalog.lock_fields)
    old = ps.catalog.ids
    remap = np.array([cat.index.get(n, -1) for n in old] + [-1], dtype=np.int64)
    b = ps.builds
    lens = np.diff(b.offsets)
    owner = np.repeat(np.arange(len(b)), lens)
    pos = np.arange(len(b.items)) - b.offsets[:-1][owner]
    is_tal = pos < b.n_talents[owner]
    new = remap[b.items.astype(np.int64)]
    keep = new >= 0
    nl = np.bincount(owner[keep], minlength=len(b))
    nt = np.bincount(owner[keep & is_tal], minlength=len(b))
    off = np.zeros(len(b) + 1, dtype=np.int64); np.cumsum(nl, out=off[1:])
    nb = CompactBuilds(b.stats, cat.ids, b.hi, b.fi, new[keep].astype(b.items.dtype), off, nt,
                       b.ids, b.timestamps, b.meta_codes, b.meta_vocab, b.rulesets, b.pr)
    pick, lift, lmask, viol = pair_statistics(nb, cat, ps.weights if cfg.stats_use_weights else None,
                                              cfg.min_pair_count, cfg.min_joint_avail)
    rep = dict(ps.report)
    rep.update(n_items=len(cat), requirement_violations=viol, trusted_pairs=int(lmask.sum() // 2),
               excluded_after_prepare=sorted(drop) + list(rep.get("excluded_after_prepare", [])),
               auto_granted_candidates=[(cat.ids[i], round(float(pick[i]), 4)) for i in np.argsort(-pick)
                                        if pick[i] >= cfg.auto_grant_rate][:100])
    return PretrainSet(nb, ps.weights, ps.lineage, cat, pick, lift, lmask, rep)


def pair_statistics(builds: CompactBuilds, catalog: Catalog, weights: Optional[np.ndarray] = None,
                    min_pair_count: float = 30.0, min_joint_avail: float = 200.0,
                    pairs_per_chunk: int = 8_000_000):
    """Pick rates and pairwise lift, conditioned on availability.

    pick_rate[i] = sum_b w x_bi / sum_b w a_bi
    lift[i,j]    = P(i,j | both available) / (P(i | both) P(j | both))
                 = co[i,j] ja[i,j] / (xa[i,j] xa[j,i])
    with co = sum w x_i x_j, ja = sum w a_i a_j, xa = sum w x_i a_j.
    Availability depends only on (highest, final) stats, so builds are grouped by
    availability mask and ja / xa are computed per group (G x N^2, not B x N^2).
    lift_mask is True where co >= min_pair_count and ja >= min_joint_avail, or
    where a pair is never picked together despite an expected count >=
    min_pair_count (a trusted "never together").
    Returns (pick_rate, lift, lift_mask, requirement_violations)."""
    N, B = len(catalog), len(builds)
    w = np.ones(B) if weights is None else np.asarray(weights, dtype=np.float64)
    has_req = any(catalog.items[i].requires or catalog.items[i].requires_any or catalog.items[i].clauses
                  for i in catalog.ids)
    if has_req and B:
        S_ = builds.fi.shape[1]
        key = np.concatenate([builds.pr, builds.fi], axis=1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        packed = np.concatenate([np.packbits(catalog.availability_stages(
            uniq[a:a + 8192, :S_], uniq[a:a + 8192, S_:]), axis=1)
            for a in range(0, len(uniq), 8192)])
        masks_p, inv2 = np.unique(packed, axis=0, return_inverse=True)
        gid = inv2.reshape(-1)[inv.reshape(-1)]
    else:
        masks_p = np.packbits(np.ones((1, N), dtype=bool), axis=1)
        gid = np.zeros(B, dtype=np.int64)
    G = len(masks_p)

    def mask_rows(a, b):
        return np.unpackbits(masks_p[a:b], axis=1, count=N).astype(np.float64)

    wg = np.bincount(gid, weights=w, minlength=G)
    ja = np.zeros((N, N))
    for a in range(0, G, 4096):
        Mg = mask_rows(a, min(G, a + 4096))
        ja += (Mg * wg[a:a + len(Mg), None]).T @ Mg
    xa = np.zeros((N, N))
    co = np.zeros(N * N)
    order = np.argsort(gid, kind="stable")
    for ch in builds.chunks(order, max_entries=pairs_per_chunk // 80 + 1):
        it, local = builds.gather(ch)
        wl = w[ch][local]
        g_u, g_loc = np.unique(gid[ch], return_inverse=True)
        Xs = np.bincount(g_loc.reshape(-1)[local] * N + it, weights=wl,
                         minlength=len(g_u) * N).reshape(len(g_u), N)
        Mg = np.unpackbits(masks_p[g_u], axis=1, count=N).astype(np.float64)
        xa += Xs.T @ Mg
        lens = np.bincount(local, minlength=len(ch))
        pl = lens * lens
        if pl.sum():
            start = np.cumsum(lens) - lens
            bp = np.repeat(np.arange(len(ch)), pl)
            pos = np.arange(pl.sum()) - (np.cumsum(pl) - pl)[bp]
            i = start[bp] + pos // lens[bp]
            j = start[bp] + pos % lens[bp]
            co += np.bincount(it[i] * N + it[j], weights=w[ch][bp], minlength=N * N)
    co = co.reshape(N, N)
    diag_x, avail_w = np.diag(co).copy(), np.diag(ja).copy()
    viol = int((diag_x > avail_w + 1e-6).sum())
    pick = np.clip(diag_x / np.maximum(avail_w, 1e-12), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = co * ja / (xa * xa.T)
        expected = xa * xa.T / ja
    lift = np.nan_to_num(lift, nan=1.0, posinf=1.0, neginf=1.0)
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0)
    np.fill_diagonal(lift, 1.0)
    lmask = ((co >= min_pair_count) | ((co == 0) & (expected >= min_pair_count))) & (ja >= min_joint_avail)
    np.fill_diagonal(lmask, False)
    return pick, lift, lmask, viol
