"""ModelService: a fitted tri run loaded in-process by the Discord bot.

A run directory is what tri.cli.fit writes (runs/<name>/ with model/, catalog.json,
granted.json, report.json). Point TRI_RUN at it, or at the runs folder (then
runs/LATEST is followed).

Reviews take effect the moment they are submitted (review memory and worth).
Locators (administrator-labelled builds) take effect the moment they are added.
Categories are read only: they come from the run (tri.json), never the
database (v3.0.2). A game data sync adds
new talents and mantras unrated, without a refit (tri.sync).
The service never refits: rules and the field change only when you run tri.cli.fit
and reload. The only thing written into the run is <run>/sync/ (the saved
game data sync, re-applied at every load).
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional, Sequence


from tri import Catalog, Tri
from tri.core.codec import build_from_dict, validate_review
from tri.core.types import Build, Review, ReviewError
from tri.game.gamedata import name_key
from tri.reviews.defs import Locator
from tri.reviews.locators import archetype_key
from tri.sources.defs import PipelineConfig
from tri.sources.pipeline import parse_record

from .wire import REASON_LABELS, CatalogHelpers, LocatorInfo, ParsedBuild  # noqa: F401 (re-exported)


def resolve_run(path: str) -> str:
    """A run dir, or a runs folder containing LATEST (absolute, or relative to the folder)."""
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "catalog.json")):
        return os.path.abspath(path)
    latest = os.path.join(path, "LATEST")
    if os.path.isfile(latest):
        with open(latest) as f:
            run = f.read().strip()
        for cand in (run, os.path.join(path, run), os.path.join(path, os.path.basename(run.rstrip("/")))):
            if os.path.isfile(os.path.join(cand, "catalog.json")):
                return os.path.abspath(cand)
    raise FileNotFoundError(f"{path} is neither a tri run (catalog.json) nor a runs folder with LATEST")


_BUILD_ID = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


def build_id_of(link: str) -> str:
    """Builder link (deepwoken.co or dwt.sh) or bare id -> build id (the bot
    sends build_id with each locator row; this is the fallback for rows
    without it)."""
    from .links import BuildLinkError, parse_build_link
    try:
        return parse_build_link(link)
    except BuildLinkError:
        raise ValueError(f"not a builder link: {link!r}")


class ModelService(CatalogHelpers):
    """Loads a run once; all model calls go through one lock (the bot calls
    these from a worker thread so its event loop stays responsive)."""

    def __init__(self, run: str):
        t0 = time.monotonic()
        self.run = resolve_run(run)
        self.catalog = Catalog.from_json(os.path.join(self.run, "catalog.json"))
        self.tri = Tri.load(os.path.join(self.run, "model"), self.catalog)
        self.sync_info: Optional[dict] = None
        self.sync_error: Optional[str] = None
        try:
            from tri.game import sync as _sync
            self.sync_info = _sync.restore(self.tri, self.run)
            self.catalog = self.tri.catalog
        except Exception as e:                             # a broken sync never stops the bot
            self.sync_error = f"saved game data sync not applied: {e}"[:300]
        g = {}
        p = os.path.join(self.run, "granted.json")
        if os.path.isfile(p):
            with open(p) as f:
                g = json.load(f)
        self.granted_known = g.get("granted_known", {})
        self.canonical_names = g.get("canonical_names", {})
        self.run_id = os.path.basename(self.run.rstrip("/"))
        self.excluded = self._read_json("excluded_talents.json", {})
        ic = self._read_json("item_counts.json", {})
        self.item_counts: Dict[str, dict] = ic.get("items", {})
        self.min_item_count = ic.get("min_item_count")
        self.lock = threading.RLock()            # v3.8: re-entrant, so the bot can hold it around a call
        self._index_names()
        self.memory_skipped: List[str] = []
        self.locators_skipped: List[str] = []
        self.categories_skipped: List[str] = []
        self.values_skipped: List[str] = []
        self.load_seconds = round(time.monotonic() - t0, 2)
        from tri.runtime import release
        release()                                          # v3.7: the load's temporaries back to the OS

    def load_text(self) -> str:
        """Where the latents came from (the run's cache or a fresh solve), the load
        time and the compute device the next fit would use."""
        if getattr(self, "_load_text", None):
            return self._load_text
        from tri.model import accel
        src = (getattr(self.tri, "load_info", None) or {}).get("latents", "?")
        d = accel.describe(self.tri.cfg.space.device)
        self._load_text = f"loaded in {self.load_seconds}s (latents: {src}); device: {d.get('device')} ({d.get('dtype')})"
        return self._load_text

    # ------------------------------------------------------------ game data sync
    def _keep_groups(self) -> List[str]:
        cfg = self._read_json("config.json", {})
        return list(((cfg.get("args") or {}).get("keep_group")) or [])

    def sync_state(self) -> Optional[dict]:
        from tri.game import sync as _sync
        return _sync.load_state(self.run)

    def sync_plan(self, ref: str = "main", force: bool = False):
        """(head, plan or None): the latest pocamind/data commit and, if it moved
        since the last sync (or force), the plan. Network only, no lock."""
        from tri.game import sync as _sync
        from tri.game.gamedata import fetch_gamedata
        state = _sync.load_state(self.run)
        head = _sync.head_commit(ref)
        if state and head.get("sha") and state.get("sha") == head["sha"] and not force:
            return head, None
        gd = fetch_gamedata(os.path.join(self.run, _sync.SYNC_DIR, "gamedata"), ref=head.get("sha") or ref,
                            refresh=True)
        # Items the corpus had too few picks of were dropped at fit time; they are
        # usually new talents, so the sync adds them as unrated like any new item.
        # TRI_SYNC_SKIP_RARE=1 keeps them out (pre-v2.18 behaviour).
        skip_rare = os.environ.get("TRI_SYNC_SKIP_RARE", "0").strip().lower() in ("1", "true", "yes", "on")
        seen = [n for n in self.item_counts if n not in self.catalog.index] if skip_rare else []
        p = _sync.plan(self.catalog, gd, state, exclude=list(self.excluded) + seen, keep_groups=self._keep_groups(),
                       sha=head.get("sha"), commit_date=head.get("date"), locks_from=self._lock_sources())
        return head, p

    def _lock_sources(self) -> list:
        """deepwoken.co game data as a second source of context locks (oath,
        origin) for the sync, best effort. v3.7: off unless TRI_SYNC_DWCO=1;
        pocamind/data holds every lock the sync reads."""
        if os.environ.get("TRI_SYNC_DWCO", "0").strip().lower() in ("0", "false", "no", "off"):
            return []
        from tri.game import sync as _sync
        from tri.game.gamedata import fetch_deepwokenco
        try:
            return [fetch_deepwokenco(os.path.join(self.run, _sync.SYNC_DIR, "gamedata"), refresh=True, timeout=30)]
        except Exception as e:
            self.sync_error = f"deepwoken.co locks skipped: {type(e).__name__}: {e}"[:300]
            return []

    def sync_apply(self, plan) -> dict:
        """Apply a plan from sync_plan (no refit) and save it in <run>/sync/."""
        from tri.game import sync as _sync
        with self.lock:
            rep = _sync.apply(self.tri, plan, self.run)
            self.catalog = self.tri.catalog
            self._index_names()
        self.sync_info = rep
        self.__dict__.pop("_sha_cache", None)
        return rep

    # ------------------------------------------------------------ categories (read only, v3.0.2)
    def categories(self) -> Dict[str, str]:
        return dict(self.tri.manual)

    def set_values(self, rows: Sequence[dict]) -> dict:
        """Replace the admin flat values with the stored rows ({item, delta}).
        Rows outside this run's catalog, or past the cap, are skipped."""
        cap = float(self.tri.cfg.infer.value_cap)
        ok, skipped = {}, []
        for r in rows:
            it = r.get("item")
            try:
                v = float(r.get("delta"))
            except (TypeError, ValueError):
                v = None
            if it in self.catalog.index and v is not None and abs(v) <= cap and v != 0:
                ok[it] = v
            else:
                skipped.append(f"{it}: {r.get('delta')}")
        with self.lock:
            self.tri.set_values(ok, replace=True)
        self.values_skipped = skipped
        return dict(loaded=len(ok), skipped=len(skipped))

    def set_value(self, item: str, delta: Optional[float]) -> None:
        with self.lock:
            self.tri.set_values({item: delta})

    def values(self) -> Dict[str, float]:
        return dict(self.tri.values)

    # ------------------------------------------------------------ authored rules (v3.0)
    def set_rules(self, rows: Sequence[dict]) -> dict:
        """Replace the authored rules with the stored rows; rows that no longer
        parse (an item renamed or removed) are kept but listed as broken."""
        from tri.rules.authored import AuthoredRules
        ar = AuthoredRules.from_json(rows)
        with self.lock:
            self.tri.set_authored(ar)
            broken = list(getattr(self.tri.authored, "broken", []))
        self.rules_broken = [f"{e.rule_id}: {msg}" for e, msg in broken]
        return dict(loaded=len(ar) - len(broken), broken=len(broken))

    def add_rule(self, text: str, author: str = "", note: str = "") -> dict:
        """Parse and add a rule (RuleSyntaxError says what is wrong). Returns the
        rule info and the entry to store."""
        with self.lock:
            info = self.tri.add_rule(text, author=author, note=note)
            info["entry"] = self.tri.authored.get(info["id"]).to_dict()
        return info

    def edit_rule(self, rule_id: str, text: str, author: str = "", note: str = "") -> dict:
        with self.lock:
            before = {e.rule_id for e in self.tri.authored.entries}
            info = self.tri.edit_rule(rule_id, text, author=author, note=note)
            after = {e.rule_id: e for e in self.tri.authored.entries}
            info["entries"] = [after[r].to_dict() for r in after if r not in before or r == info["id"]
                               or (r == rule_id and after[r].op == "suppress")]
            info["deleted"] = [r for r in before if r not in after]
        return info

    def drop_rule(self, rule_id: str, author: str = "", note: str = "") -> dict:
        with self.lock:
            out = self.tri.drop_rule(rule_id, author=author, note=note)
            e = self.tri.authored.get(rule_id)
            out["entry"] = e.to_dict() if e is not None else None
        return out

    def list_rules(self, item: Optional[str] = None, status: Optional[str] = None,
                   source: Optional[str] = None, build=None, every: bool = False) -> List[dict]:
        """v3.4: every=True (or a build) lists every non-worth rule (Tri.all_rules)."""
        with self.lock:
            return self.tri.list_rules(item=item, status=status, source=source, build=build, every=every)

    def rule_groups(self) -> Dict[str, List[str]]:
        rb = self.tri.rulebook
        return {g: sorted(m) for g, m in (rb.groups if rb is not None else {}).items()}

    def consensus(self, archetype: Optional[str] = None) -> Dict[str, list]:
        """Promoted reviewer tells per family (math §8.5a)."""
        return self.tri.consensus_report(archetype)

    def _read_json(self, name: str, default):
        p = os.path.join(self.run, name)
        if not os.path.isfile(p):
            return default
        with open(p) as f:
            return json.load(f)

    # ------------------------------------------------------------ review memory
    @property
    def n_memory(self) -> int:
        """Reviews currently in the review memory."""
        return len(self.tri.reviews)

    def _valid(self, r: Review) -> Optional[str]:
        try:
            validate_review(copy.deepcopy(r), self.catalog)
            return None
        except (ReviewError, KeyError, ValueError) as e:
            return str(e)

    def set_reviews(self, reviews: Sequence[Review]) -> dict:
        """Replace the review memory (e.g. every row in Supabase at startup).
        Reviews that do not validate against this run's catalog are skipped."""
        ok, skipped = [], []
        for r in reviews:
            e = self._valid(r)
            (skipped.append(f"{r.review_id}: {e}") if e else ok.append(r))
        with self.lock:
            self.tri.set_reviews(ok)
        self.memory_skipped = skipped
        return dict(loaded=len(ok), skipped=len(skipped))

    def set_answers(self, rows: Sequence[dict]) -> dict:
        """Caution answers from every user (db tri_answers); only consensus reaches the model."""
        out = []
        for r in rows:
            try:
                bd = r["build"] if isinstance(r["build"], dict) else json.loads(r["build"])
                out.append(dict(user=str(r.get("user_id") or r.get("user") or ""), build=build_from_dict(bd),
                                kind=r["kind"], item=r.get("item"), value=r.get("value"), ts=float(r.get("ts") or 0.0)))
            except Exception:
                continue
        with self.lock:
            self.tri.set_answers(out)
        return dict(answers=len(out), notes=len(getattr(self.tri, "_answer_notes", []) or []))

    def set_memory(self, reviews: Sequence[Review], rows: Sequence[dict]) -> dict:
        """set_reviews and set_locators with one rebuild (startup and /model reload)."""
        ok, skipped = [], []
        for r in reviews:
            e = self._valid(r)
            (skipped.append(f"{r.review_id}: {e}") if e else ok.append(r))
        locs, lskip = [], []
        for r in rows:
            try:
                locs.append(self.locator_from_row(r))
            except Exception as e:
                lskip.append(f"{r.get('id')}: {e}")
        with self.lock:
            self.tri.set_memory(ok, locs)
        from tri.runtime import release
        release()
        self.memory_skipped, self.locators_skipped = skipped, lskip
        return dict(loaded=len(ok), skipped=len(skipped), locators=len(locs), locators_skipped=len(lskip))

    def add_review(self, review: Review) -> bool:
        """Add (or replace, same id) one review; effective immediately. False
        if it does not validate."""
        if self._valid(review):
            return False
        with self.lock:
            self.tri.add_reviews([review], validate=False)
        return True

    # ------------------------------------------------------------ locators
    def locator_from_row(self, row: dict) -> Locator:
        """One locators row {id, link, archetype, record, added_by, created_at} -> Locator.
        The stored record is parsed with this run's catalog. Raises ValueError."""
        rec = row.get("record")
        if isinstance(rec, str):
            rec = json.loads(rec)
        if not isinstance(rec, dict):
            raise ValueError(f"locator {row.get('id')}: no build record")
        bid = row.get("build_id") or build_id_of(row.get("link") or "")
        b = self.parse(bid, rec).build
        b.build_id = bid
        return Locator(str(row["id"]), b, " ".join(str(row.get("archetype", row.get("purpose", ""))).split()), row.get("link"),
                       row.get("added_by"), None)

    def set_locators(self, rows: Sequence[dict]) -> dict:
        """Replace the locators (every row of the locators table). Instant, no refit."""
        ok, skipped = [], []
        for r in rows:
            try:
                ok.append(self.locator_from_row(r))
            except Exception as e:
                skipped.append(f"{r.get('id')}: {e}")
        with self.lock:
            self.tri.set_locators(ok)
        self.locators_skipped = skipped
        return dict(loaded=len(ok), skipped=len(skipped))

    def add_locator(self, row: dict) -> Locator:
        loc = self.locator_from_row(row)
        with self.lock:
            self.tri.add_locators([loc])
        return loc

    def remove_locator(self, locator_id) -> bool:
        with self.lock:
            return self.tri.remove_locator(str(locator_id))

    @property
    def archetypes(self) -> List[str]:
        """Archetype names in use, for autocomplete (one spelling per archetype)."""
        seen: Dict[str, str] = {}
        for l in self.tri.locators:
            seen[archetype_key(l.archetype)] = l.archetype
        return sorted(seen.values(), key=str.casefold)

    def locator_report(self) -> dict:
        with self.lock:
            rep = self.tri.locator_report()
            rep["recommended"] = self.tri.select_metric(apply=False) if self.tri.locset is not None else None
        return rep

    def archetype(self, b: Build) -> dict:
        with self.lock:
            return self.tri.archetype(b)

    # ------------------------------------------------------------ item diagnostics
    def explain_item(self, name: str, b: Optional[Build] = None) -> dict:
        """Everything about one item: catalog status (or why it is missing),
        requirement, training-pick count, review memory claims, and with a
        build its status there: endorsed / iffy / suggested drop / suggested
        take / not suggested, with worth and rank."""
        item = self.resolve(name)
        k = name_key(name)
        cnt = self.item_counts.get(item or "") or self.item_counts.get(self._count_keys.get(k, ""), None)
        out = dict(query=name, item=item, in_catalog=item is not None, why_missing=self.why_missing(name),
                   picks=cnt.get("count") if cnt else None, min_item_count=self.min_item_count)
        if item is None:
            return out
        out.update(kind=self.kind(item), requirement=self.requirement_text(item))
        if b is None:
            return out
        with self.lock:
            al = self.tri.review(b, max_alternatives=1)[0]
            w = self.tri.worth(b, item)
            claim = self.tri.ledger.claims_for(b).get(item) if self.tri.ledger else None
        drops = {c.item: c for c in al.changes if c.action == "remove"}
        takes = {c.item: c for c in al.changes if c.action == "add"}
        if item in drops:
            st = "suggested drop"
        elif item in takes:
            st = "suggested take"
        elif item in b.items():
            st = "iffy" if item in al.iffy else "endorsed"
        else:
            why_not = self.catalog.unavailable_reason(item, b)
            st = "not suggested" if why_not is None else (
                "requirements not met at any of the build's stages" if why_not == "stats" else f"not available: {why_not}")
        src = (drops.get(item) or takes.get(item))
        out.update(in_build=item in b.items(), status=st, worth=float(w["worth"]), rank=w["rank"], of=w["of"],
                   pass_=w["pass_"], rules=list(w["rules"][:3]), archetype=w["region_name"],
                   best_swap=w["best_swap"], swap_gain=w["swap_gain"],
                   source=src.source if src else ("review" if claim else "model"),
                   why=src.why if src else None,
                   review_claim=dict(label=claim[0], strength=float(claim[1]), review_id=claim[2],
                                     similarity=float(claim[3])) if claim else None)
        return out

    # ------------------------------------------------------------ parsing
    def parse(self, build_id: str, record: dict) -> ParsedBuild:
        """Planner record -> ParsedBuild. Raises ValueError if unreadable."""
        cfg = PipelineConfig(include_private=True, min_power=0, min_talents=0,
                             granted_known=dict(self.granted_known), canonical_names=dict(self.canonical_names))
        pr, why = parse_record(build_id, record, cfg)
        if pr is None:
            raise ValueError(f"could not read build {build_id}: {why}")
        b = pr.build
        raw_t, raw_m = list(b.talents), list(b.mantras)
        unknown = [x for x in b.items() if x not in self.catalog.index]
        b.talents = [t for t in b.talents if t in self.catalog.index]
        b.mantras = [m for m in b.mantras if m in self.catalog.index]
        if unknown:                                     # v3.6: they stay in the saved build, so they take room
            b.meta = dict(b.meta or {}, unknown_items=dict(
                talents=[x for x in unknown if x in raw_t], mantras=[x for x in unknown if x in raw_m and x not in raw_t]))
        unmet = [x for x in b.items() if not self.catalog.available_in(x, b)]
        why = {x: self.why_missing(x) or "not in the model's catalog" for x in unknown}
        st = record.get("stats") or {}
        name = str(st.get("buildName") or build_id)
        au = record.get("author")
        author = st.get("buildAuthor") or (au.get("name") if isinstance(au, dict) else au) or None
        return ParsedBuild(b, name, author, list(pr.granted), unknown, unmet, raw_t, raw_m, why, record=record)

    # ------------------------------------------------------------ applying the model
    def _ops(self, al, b: Optional[Build] = None, held=()) -> dict:
        from tri.infer.draft import change_margin
        cat = self.catalog
        held = set(held)

        def row(c):
            return dict(item=c.item, kind=cat.items[c.item].kind.value, reason=c.reason, related=list(c.related),
                        source=c.source, review_id=c.review_id, pass_=c.pass_, why=c.why,
                        marginal=float(c.marginal), margin=change_margin(c), held=(c.item, c.action) in held)
        iffy = [x for x in al.iffy if b is None or x in b.items()]
        return dict(mode=al.mode, name=al.name, plausibility=float(al.plausibility), gain=float(al.gain),
                    drops=[row(c) for c in al.changes if c.action == "remove" and (c.item, c.action) not in held],
                    takes=[row(c) for c in al.changes if c.action == "add" and (c.item, c.action) not in held],
                    held=[row(c) for c in al.changes if (c.item, c.action) in held],
                    iffy=iffy, capacity=list(al.capacity),
                    violations=[dict(item=v["item"], kind=v["kind"], rule=v.get("rule"), against=v.get("against"))
                                for v in (al.violations or [])],
                    sidegrades=[dict(g) for g in (getattr(al, "sidegrades", None) or [])],
                    unrated=list(getattr(al, "unrated", None) or []),
                    iffy_why={k: dict(v) for k, v in (getattr(al, "iffy_why", None) or {}).items() if k in iffy},
                    indicative={k: dict(v) for k, v in (getattr(al, "indicative", None) or {}).items()},
                    focus=dict(getattr(al, "focus", None) or {}))

    def _caution(self, b: Build, result, answers, archetype, caution: str, mode: str, **kw):
        """(result, caution dict). caution: "ask" (hold thin marks and ask), "hold"
        (hold only) or "off". Held marks are settled (Tri.settle): the answer is
        solved again without them, so every slot is still filled."""
        if caution == "off" or not self.tri.cfg.caution.enabled:
            return result, dict(mode="off", held=[], questions=[], sufficient=True)
        result, c = self.tri.settle(b, mode, archetype, answers, result=result, **kw)
        d = c.to_dict()
        if caution != "ask":
            d["questions"] = []
        d["mode"] = caution
        return result, d

    def analyze(self, b: Build, max_alternatives: int = 3, worst: int = 10,
                archetype: Optional[str] = None, answers=None, caution: str = "ask") -> dict:
        """Smallest edits per plausible archetype, and the current picks ranked by worth.
        archetype: analyse as that archetype only (ValueError if unknown). answers:
        the asker's answers (local to this call); caution: ask | hold | off."""
        from tri.infer.caution import Answers
        a = Answers.of(answers)
        archetype = archetype or a.archetype
        with self.lock:
            alts = self.tri.review(b, max_alternatives=max_alternatives, archetype=archetype, answers=a)
            ws = self.tri.build_worth(b, archetype)
            concept = self.tri.concept(b)
            alts, ca = self._caution(b, alts, a, archetype, caution, "review", max_alternatives=max_alternatives)
        held = ()                                        # settled: held marks are no longer in the changes
        out = dict(n_talents=len(b.talents), n_mantras=len(b.mantras), ambiguity=float(concept["ambiguity"]),
                   archetype=concept.get("archetype"), regions=list(concept.get("regions") or []),
                   nearest_reviews=list(concept.get("nearest_reviews") or []),
                   talent_cap=self.catalog.talent_cap(b.items()),
                   mantra_slots=self.catalog.mantra_slot_usage(b.items(), self.catalog.oath_of(b)),
                   alternatives=[self._ops(al, b, held if i == 0 else ()) for i, al in enumerate(alts)],
                   worst=[], memory=self.n_memory, caution=ca,
                   review_claims=len(alts[0].memory) if alts else 0,
                   forced=alts[0].name if archetype and alts else None)
        for w in [w for w in ws if w["pass_"] not in ("foundational",)][:worst]:
            out["worst"].append(dict(item=w["item"], worth=float(w["worth"]), rank=w["rank"], of=w["of"],
                                     best_swap=w["best_swap"], swap_gain=float(w["swap_gain"] or 0.0),
                                     pass_=w["pass_"]))
        return out

    def _unknown_room(self, b: Build):
        """(unknown talents, unknown mantras) the build's record keeps (v3.6)."""
        u = (b.meta or {}).get("unknown_items") or {}
        return len(u.get("talents") or ()), len(u.get("mantras") or ())

    def complete(self, b: Build, talents: Optional[int] = None, mantras: Optional[int] = None,
                 archetype: Optional[str] = None, answers=None, caution: str = "ask") -> dict:
        """Fill a partial build in pass order (foundational, central, luxury).
        talents / mantras: optional upper bounds; the game's limits (the talent
        cap 76 - 2m and the build's mantra slots) always apply. archetype:
        complete it as that archetype (ValueError if unknown)."""
        with self.lock:
            cap = (talents, mantras) if talents is not None or mantras is not None else None
            from tri.infer.caution import Answers
            ans = Answers.of(answers)
            archetype = archetype or ans.archetype
            ut, um = self._unknown_room(b)
            if um:                                      # v3.6: unknown mantras fill slots (read as wildcards)
                tot = sum(self.catalog.mantra_slot_usage(b.items(), self.catalog.oath_of(b))["slots"].values())
                cap = (talents, max(0, min(tot - um, mantras if mantras is not None else tot)))
            for _ in range(3):
                al = self.tri.complete(b, capacity=cap, archetype=archetype, answers=ans)
                al, ca = self._caution(b, al, ans, archetype, caution, "complete", capacity=cap)
                if not (ut or um):
                    break
                room = self.catalog.talent_cap(al.items()) - self.catalog.cap_per_mantra * um - ut
                if len(al.talents) <= room:
                    break                               # unknown talents count, unknown mantras lower the cap
                cap = (min(room, talents) if talents is not None else room, cap[1] if cap else None)
            concept = self.tri.concept(b)
        hset = set()                                     # settled: held adds were replaced by the next pick
        d = self._ops(al, held=hset)
        d["caution"] = ca
        have = set(b.items())
        by_pass: Dict[str, List[str]] = {}
        for c in al.changes:
            if c.action == "add" and (c.item, "add") not in hset:
                by_pass.setdefault(c.pass_, []).append(c.item)
        d.update(forced=al.name if archetype else None, by_pass=by_pass, kept=[x for x in al.items() if x in have], ambiguity=float(concept["ambiguity"]),
                 n_talents=len([x for x in al.talents if (x, "add") not in hset]) + ut,
                 n_mantras=len([x for x in al.mantras if (x, "add") not in hset]) + um,
                 removed=[c.item for c in al.changes if c.action == "remove" and (c.item, "remove") not in hset],
                 talent_cap=self.catalog.talent_cap(al.items()) - self.catalog.cap_per_mantra * um,
                 unknown=dict((b.meta or {}).get("unknown_items") or {}),
                 mantra_slots=self.catalog.mantra_slot_usage(al.items() + [x for x in b.items() if x not in self.catalog.items],
                                                             self.catalog.oath_of(b)))
        from tri.core.codec import is_silentheart
        silent = is_silentheart(b)
        d.update(silentheart=silent, mantras_first=not silent and not any(self.kind(x) == "mantra" for x in have)
                 and any(self.kind(c.item) == "mantra" for c in al.changes if c.action == "add"))
        return d

    # ------------------------------------------------------------ for the bot
    def locator_list(self, with_builds: bool = False) -> List[LocatorInfo]:
        return [LocatorInfo.of(l, with_builds) for l in self.tri.locators]

    def state(self) -> dict:
        """What a client caches between calls (cheap; no model work)."""
        return dict(run_id=self.run_id, n_memory=self.n_memory, n_locators=len(self.tri.locators),
                    archetypes=self.archetypes, locators_skipped=list(self.locators_skipped),
                    memory_skipped=list(self.memory_skipped), categories_skipped=list(self.categories_skipped),
                    sync_info=self.sync_info, sync_error=self.sync_error, load_text=self.load_text(),
                    n_items=len(self.catalog), catalog_rev=self.catalog_rev(), ruleset=self.catalog.ruleset,
                    n_categories=len(self.tri.manual), n_sidegrades=len(getattr(self.tri, "sidegrades", None) or []),
                    sync_sha=self._sync_sha(),
                    n_unrated=len([i for i in self.tri.patches.release_times() if i in self.catalog.index]))

    def _sync_sha(self) -> Optional[str]:
        if not hasattr(self, "_sha_cache"):
            self._sha_cache = (self.sync_state() or {}).get("sha")
        return self._sha_cache

    def catalog_rev(self) -> str:
        """Changes whenever the catalog does (reload or applied sync)."""
        s = self.sync_info or {}
        return f"{self.run_id}:{len(self.catalog)}:{s.get('sha') or ''}:{id(self.catalog)}"

    def catalog_bundle(self) -> dict:
        """The catalog snapshot a client needs for autocomplete and draft checks."""
        return dict(catalog=self.catalog.to_dict(), canonical_names=self.canonical_names,
                    excluded=self.excluded, item_counts=self.item_counts, min_item_count=self.min_item_count,
                    run_id=self.run_id, catalog_rev=self.catalog_rev())

    def plot_concept(self, *a, **kw) -> tuple:
        from .plotting import plot_concept
        return plot_concept(self, *a, **kw)

    def plot_worth(self, *a, **kw) -> tuple:
        from .plotting import plot_worth
        return plot_worth(self, *a, **kw)

    def plot_atlas(self, *a, **kw) -> tuple:
        from .plotting import plot_atlas
        return plot_atlas(self, *a, **kw)

    def insight(self, kind: str, **kw) -> dict:
        """Archetypes, facts and stream views (bot.insight)."""
        from .insight import insight
        return insight(self, kind, **kw)


