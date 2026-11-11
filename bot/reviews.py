"""Review sessions: claiming pool entries, building a draft of operations,
validating it against the tri review contract and producing the JSON stored in
reviews.operations. No Discord code here, so it can be tested and reused.

Protocol (same as tri): the build's talent slots are full; anything the
reviewer does not mark is endorsed. Each item gets at most one operation:
    remove  item  weak|strong  reason [related]   reason is one of the 6 remove reasons
    add     item  weak|strong  [reason]           reason defaults to necessary_for_archetype;
                                                  generally_good (talents) endorses a foundational talent
    iffy    item                                  no strength, no reason
redundant_with / works_against need related item(s) from the build or the draft's adds.
An add of a talent mutually exclusive with a pick is refused until that pick
and every pick that requires it are removed (ExclusionError says which).

Claims and drafts live in a small JSON file (STATE_PATH) so a restart does not
lose work. One bot process per state file.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tri.core.codec import build_from_dict, build_to_dict, exclusion_message, review_to_dict, validate_review
from tri.core.types import (ADD_REASONS, IFFY_REASONS, RELATIONAL_REASONS, REMOVE_ONLY_REASONS, Action, Build,
                            Kind, Operation, Reason, Review, ReviewError, Strength)

SCHEMA = "tri.review/1"
REMOVE_REASONS = [r for r in Reason if r not in ADD_REASONS and r not in IFFY_REASONS]


_REL_WORD = {Reason.REDUNDANT_WITH: "redundant with", Reason.WORKS_AGAINST: "working against",
             Reason.SYNERGY_WITH: "synergising with", Reason.NEEDS: "not working without",
             Reason.SIDEGRADE: "a sidegrade to"}


def _with_draft_waivers(b: Build, d) -> Build:
    """The build with the weapon locks the draft's adds already waived, so an
    add whose prerequisite was waived in the same review is available."""
    w = [o["item_id"] for o in d.ops if o.get("action") == "add"
         and (o.get("waived") or "weapon requirement waived" in str(o.get("note") or ""))]
    if not w:
        return b
    import dataclasses
    from tri.core.context import wield_waived
    meta = dict(b.meta or {})
    meta["wield_waived"] = sorted(wield_waived(b) | set(w))
    return dataclasses.replace(b, meta=meta)


class DraftError(ValueError):
    """An operation or submission was refused; the message is user-facing."""


class ExclusionError(DraftError):
    """An add refused because the item is mutually exclusive with current picks.
    blockers / dependents: what has to be removed first."""

    def __init__(self, msg: str, item: str, blockers, dependents):
        super().__init__(msg)
        self.item, self.blockers, self.dependents = item, list(blockers), list(dependents)


@dataclass
class Draft:
    user_id: int
    pool_id: Optional[int]               # None: a direct review (/review build:) until /submit gives it a negative id
    link: str
    build_id: str
    name: str
    claimed_at: float
    expires_at: float
    build: dict                          # build_to_dict of the parsed (catalog-only) build
    raw: dict = field(default_factory=dict)   # raw_talents, raw_mantras, granted, unknown
    ops: List[dict] = field(default_factory=list)
    messages: List[int] = field(default_factory=list)    # bot messages a reply to which edits this draft
    channel_id: Optional[int] = None
    direct: bool = False                 # reviewed by link, never added to the pool; id given at /submit
    notes: List[dict] = field(default_factory=list)   # # or // lines (Review.notes): archetype / weapons / comment

    def build_obj(self) -> Build:
        return build_from_dict(dict(self.build))

    def numbered_items(self) -> List[tuple]:
        """The build's own picks as numbered on the review sheet, for fast
        replies: [(n, item, kind)], mantras then talents, each alphabetical."""
        m = sorted(self.build.get("mantras", []), key=str.casefold)
        t = sorted(self.build.get("talents", []), key=str.casefold)
        return [(i, it, k) for i, (it, k) in enumerate([(x, "mantra") for x in m] + [(x, "talent") for x in t], 1)]

    def items_in_ops(self) -> Dict[str, dict]:
        return {o["item_id"]: o for o in self.ops}

    def adds(self) -> List[str]:
        return [o["item_id"] for o in self.ops if o["action"] == "add"]

    def removes(self) -> List[str]:
        return [o["item_id"] for o in self.ops if o["action"] == "remove"]

    def current(self) -> set:
        """The picks as the draft stands: the build minus removals plus additions."""
        picks = set(self.build.get("talents", [])) | set(self.build.get("mantras", []))
        return (picks - set(self.removes())) | set(self.adds())

    def number_of(self, item: str) -> Optional[int]:
        for n, it, _k in self.numbered_items():
            if it == item:
                return n
        return None

    def stranded(self, catalog) -> List[tuple]:
        """[(pick, missing prerequisites text)]: picks still in the draft whose
        prerequisite a removal took away (submission is refused until they go)."""
        from tri.core.codec import prereq_text
        cur = self.current()
        full = cur | set(self.build.get("talents", [])) | set(self.build.get("mantras", []))
        return [(z, prereq_text(catalog, z)) for z in sorted(cur) if z in catalog.items
                and catalog.prereqs_met(z, full) and not catalog.prereqs_met(z, cur)]


class ReviewDesk:
    """claims: one active draft per user, one user per pool entry, with expiry."""

    def __init__(self, state_path: str, claim_ttl_s: float = 7200.0, clock=time.time):
        self.path = state_path
        self.ttl = claim_ttl_s
        self.clock = clock
        self.drafts: Dict[int, Draft] = {}         # user_id -> draft
        self.skipped: Dict[int, List[int]] = {}    # user_id -> pool ids they skipped
        self.broken: Dict[int, str] = {}           # pool_id -> why the link could not be read
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self):
        if not os.path.isfile(self.path):
            return
        with open(self.path) as f:
            d = json.load(f)
        from tri.core.codec import legacy_names
        fields = set(Draft.__dataclass_fields__)
        self.drafts = {int(k): Draft(**{f: x for f, x in legacy_names(v).items() if f in fields})
                       for k, v in d.get("drafts", {}).items()       # drafts saved before v2.5 say "purpose"
                       if v.get("scope", "full") != "audit"}         # v3.5: open audits are dropped
        self.skipped = {int(k): list(v) for k, v in d.get("skipped", {}).items()}
        self.broken = {int(k): v for k, v in d.get("broken", {}).items()}

    def save(self):
        d = dict(drafts={str(k): v.__dict__ for k, v in self.drafts.items()},
                 skipped={str(k): v for k, v in self.skipped.items()},
                 broken={str(k): v for k, v in self.broken.items()})
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".state")
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ claims
    def expire(self) -> List[Draft]:
        now = self.clock()
        gone = [d for d in self.drafts.values() if d.expires_at <= now]
        for d in gone:
            del self.drafts[d.user_id]
        if gone:
            self.save()
        return gone

    def draft(self, user_id: int) -> Optional[Draft]:
        self.expire()
        return self.drafts.get(user_id)

    def require(self, user_id: int) -> Draft:
        d = self.draft(user_id)
        if d is None:
            raise DraftError("You have no build claimed. Use /review to get one.")
        return d

    # ------------------------------------------------------------ reply sheets
    def attach_message(self, user_id: int, message_id: int, channel_id: Optional[int] = None, keep: int = 50):
        """Register a bot message as a review sheet: replies to it edit this draft."""
        d = self.drafts.get(user_id)
        if d is None:
            return
        if message_id not in d.messages:
            d.messages = (d.messages + [int(message_id)])[-keep:]
        if channel_id is not None:
            d.channel_id = int(channel_id)
        self.save()

    def draft_for_message(self, message_id: int) -> Optional[Draft]:
        self.expire()
        for d in self.drafts.values():
            if int(message_id) in d.messages:
                return d
        return None

    def claimed_ids(self) -> set:
        self.expire()
        return {d.pool_id for d in self.drafts.values() if d.pool_id is not None}

    def candidates(self, user_id: int, pending: Sequence) -> List:
        """Pending pool entries this user may claim, oldest first."""
        taken = self.claimed_ids()
        skip = set(self.skipped.get(user_id, []))
        return [p for p in pending if p.id not in taken and p.id not in skip and p.id not in self.broken]

    def claim(self, user_id: int, pool_id: Optional[int], link: str, parsed, direct: bool = False) -> Draft:
        """parsed: bot.wire.ParsedBuild of the entry's current build. direct: a
        build given by link that is not in the pool (pool_id None until /submit)."""
        if user_id in self.drafts:
            raise DraftError("You already have a build claimed; /submit or /skip it first.")
        if pool_id is not None and pool_id in self.claimed_ids():
            raise DraftError("That build is being reviewed by someone else.")
        b = parsed.build
        if b.build_id and any(o.build_id == b.build_id and (direct or o.direct)
                              for o in self.drafts.values()):
            raise DraftError("That build is being reviewed by someone else.")
        now = self.clock()
        d = Draft(user_id=user_id, pool_id=pool_id, link=link, build_id=b.build_id or "",
                  name=parsed.name, claimed_at=now, expires_at=now + self.ttl, build=build_to_dict(b),
                  raw=dict(talents=parsed.raw_talents, mantras=parsed.raw_mantras,
                           granted=parsed.granted, unknown=parsed.unknown), direct=direct)
        self.drafts[user_id] = d
        self.save()
        return d

    def release(self, user_id: int, skip: bool = False) -> Optional[Draft]:
        d = self.drafts.pop(user_id, None)
        if d is not None and skip and not d.direct and d.pool_id is not None:
            self.skipped.setdefault(user_id, []).append(d.pool_id)
        self.save()
        return d

    def mark_broken(self, pool_id: int, why: str):
        self.broken[pool_id] = why
        self.save()

    def touch(self, d: Draft):
        d.expires_at = self.clock() + self.ttl

    # ------------------------------------------------------------ operations
    def add_op(self, user_id: int, action: str, item: str, svc, strength: Optional[str] = None,
               reason: Optional[str] = None, related: Sequence[str] = (), note: Optional[str] = None) -> dict:
        """Validate and append one operation. svc: bot.service.ModelService.
        `item` and `related` must already be resolved to catalog ids (or are resolved here)."""
        d = self.require(user_id)
        b = d.build_obj()
        name = svc.resolve(item)
        if name is None:
            raise DraftError(f"'{item}' is not a known talent or mantra.")
        if name in d.items_in_ops():
            raise DraftError(f"{name} already has an operation; /undo it first.")
        act = Action(action)
        in_build = name in b.item_set()
        if act in (Action.REMOVE, Action.IFFY) and not in_build:
            raise DraftError(f"{name} is not in this build.")
        if act == Action.ADD:
            if in_build:
                raise DraftError(f"{name} is already in this build.")
            b = _with_draft_waivers(b, d)                # v3.0.3: a lock waived on an earlier add counts here
            why_not = svc.catalog.unavailable_reason(name, b)
            if why_not == "stats":
                raise DraftError(f"{name} is not available at this build's stats "
                                 f"(needs {svc.requirement_text(name)}).")
            waived = why_not is not None and why_not != "stats" and svc.catalog.wield_only_block(name, b)
            if why_not is not None and not waived:
                raise DraftError(f"{name} is not available to this build ({why_not}).")
            cur = d.current()
            bl, dep = svc.catalog.exclusion_removals(name, cur)
            if bl:
                raise ExclusionError(self.exclusion_hint(d, name, bl, dep), name, bl, dep)
        op = dict(action=act.value, item_id=name, strength=Strength.STRONG.value, reason=None, related=[], group=None)
        if act == Action.ADD and waived:                 # a review may waive a weapon lock (v2.23)
            op["waived"] = True
            op["note"] = (note + " | " if note else "") + f"weapon requirement waived by review ({why_not})"
            note = None
        if act == Action.IFFY and not reason and not related:
            if note:
                op["note"] = note
            d.ops.append(op)
            self.touch(d)
            self.save()
            return op
        try:
            op["strength"] = Strength(strength or "strong").value
        except ValueError:
            raise DraftError("Strength is weak or strong.")
        if act == Action.IFFY:
            r = Reason(reason) if reason else Reason.SIDEGRADE
            if r != Reason.SIDEGRADE:
                raise DraftError("The iffy reason is SG (sidegrade to > item).")
        elif act == Action.ADD:
            r = Reason(reason) if reason else (Reason.SYNERGY_WITH if related else Reason.NECESSARY_FOR_ARCHETYPE)
            if r == Reason.GENERALLY_GOOD and svc.catalog.items[name].kind != Kind.TALENT:
                raise DraftError(f"'Generally good' marks foundational talents; {name} is a mantra.")
            if r in REMOVE_ONLY_REASONS or r in IFFY_REASONS:
                raise DraftError(f"'{r.value}' is not a reason for adds.")
        else:
            if not reason:
                raise DraftError("A remove needs a reason.")
            r = Reason(reason)
            if r in NON_CLASH_REASONS and _is_central(svc, name):
                raise DraftError(f"{name} is central: it defines the build's playstyle and is respected. Remove it "
                                 f"only for a clash (RW, WA, BR, SM, NW), not {r.value}.")
            if r in ADD_REASONS:
                raise DraftError(f"'{r.value}' is a reason for adds, not removes.")
            if r in IFFY_REASONS:
                raise DraftError(f"'{r.value}' is a reason for iffy marks (= item SG > other).")
        op["reason"] = r.value
        rel = []
        here = set(b.item_set()) | set(d.adds())
        for x in related:
            if not x:
                continue
            if str(x).startswith("weapon:"):            # v3.3: the claim holds where this weapon is wielded
                from tri.game.weapons import weapon_token
                wt = weapon_token(x)
                if wt is None:
                    raise DraftError(f"'{x[7:]}' is not a known weapon.")
                if r in (Reason.NEEDS, Reason.SIDEGRADE):
                    raise DraftError(f"'{r.value}' names talents or mantras, not a weapon.")
                rel.append(wt)
                continue
            rx = svc.resolve(x)
            if rx is None:
                raise DraftError(f"'{x}' is not a known talent or mantra.")
            if rx == name:
                raise DraftError("An item cannot be related to itself.")
            if r not in (Reason.NEEDS, Reason.SIDEGRADE) and rx not in here:
                raise DraftError(f"Related item '{x}' must be in the build or one of your adds.")
            rel.append(rx)
        if r in RELATIONAL_REASONS and not rel:
            raise DraftError(f"'{r.value}' needs a related item (> item: what is it "
                             f"{_REL_WORD.get(r, 'related to')}?).")
        if r == Reason.NEEDS:
            gone = set(d.removes()) | {name}
            if all(y in here and y not in gone for y in rel):
                raise DraftError(f"{name}: the build already has {', '.join(rel)}; NW names the components it lacks.")
            if any(y in d.adds() for y in rel):
                raise DraftError("NW names components the build lacks; you are adding "
                                 f"{', '.join(y for y in rel if y in d.adds())} (use + ... SY > {name} instead).")
        op["related"] = rel
        if note:
            op["note"] = note
        d.ops.append(op)
        self.touch(d)
        self.save()
        return op

    @staticmethod
    def exclusion_hint(d: Draft, item: str, blockers, dependents) -> str:
        """User-facing: what to remove before item can be added, with sheet numbers
        and a reply line to paste."""
        msg = exclusion_message(item, blockers, dependents) + "."
        out = list(blockers) + list(dependents)
        added = [x for x in out if x in d.adds()]
        nums = [d.number_of(x) for x in out if x not in added]
        if added:
            msg += f" Unmark the add of {', '.join(added)} first (~ {', '.join(added)})."
        if nums and all(n is not None for n in nums):
            msg += (f" Reply with both lines, e.g. `- {','.join(str(n) for n in sorted(nums))} s LN` and "
                    f"`+ {item}` (a removal for space pairs with the add).")
        return msg

    def undo(self, user_id: int) -> Optional[dict]:
        d = self.require(user_id)
        if not d.ops:
            return None
        op = d.ops.pop()
        # drop dangling relations to an add that was just undone
        for o in d.ops:
            if op["action"] == "add" and op["item_id"] in o["related"]:
                o["related"] = [x for x in o["related"] if x != op["item_id"]]
        self.touch(d)
        self.save()
        return op

    def remove_item(self, user_id: int, item: str) -> Optional[dict]:
        d = self.require(user_id)
        for i, o in enumerate(d.ops):
            if o["item_id"] == item:
                d.ops.pop(i)
                self.save()
                return o
        return None

    def add_note(self, user_id: int, note: dict) -> dict:
        d = self.require(user_id)
        d.notes.append(dict(note))
        self.touch(d)
        self.save()
        return note

    def remove_note(self, user_id: int, k: Optional[int] = None) -> List[dict]:
        """Drop note k (1-based), or every note."""
        d = self.require(user_id)
        if k is None:
            out, d.notes = d.notes, []
        else:
            if not 1 <= k <= len(d.notes):
                raise DraftError(f"There is no note {k} (the draft has {len(d.notes)}).")
            out = [d.notes.pop(k - 1)]
        self.save()
        return out

    def clear(self, user_id: int):
        d = self.require(user_id)
        d.notes = []
        d.ops = []
        self.touch(d)
        self.save()

    # ------------------------------------------------------------ submission
    def review(self, d: Draft, catalog) -> Tuple[Review, List[str]]:
        ops = [Operation(action=Action(o["action"]), item_id=o["item_id"], strength=Strength(o["strength"]),
                         reason=Reason(o["reason"]) if o.get("reason") else None,
                         related=list(o.get("related") or []), group=o.get("group"))
               for o in d.ops if o.get("action") in ("add", "remove", "iffy")]
        now = time.time()
        rid = f"pool-{d.pool_id}"
        r = Review(review_id=rid, original=d.build_obj(), operations=ops,
                   reviewer_id=f"discord:{d.user_id}", ruleset=getattr(catalog, "ruleset", None) or d.build.get("ruleset"),
                   timestamp=now, notes=[dict(n) for n in d.notes])
        for o in ops:
            bad = [x for x in o.related if x not in r.original.item_set() and x not in {a.item_id for a in r.added()}]
            if o.reason in (Reason.NEEDS, Reason.SIDEGRADE):
                bad = [x for x in o.related if x not in catalog.items]
            if o.reason in RELATIONAL_REASONS and (not o.related or bad):
                raise DraftError(f"{o.item_id}: related items must be in the build or your adds ({', '.join(bad)}).")
        try:
            warnings = validate_review(r, catalog)
        except ReviewError as e:
            raise DraftError(f"Review rejected: {e}")
        return r, warnings

    def payload(self, d: Draft, catalog, run_id: Optional[str] = None) -> Tuple[dict, List[str]]:
        """(JSON for reviews.operations, warnings)."""
        r, warnings = self.review(d, catalog)
        body = review_to_dict(r)
        body.update(schema=SCHEMA, pool_id=d.pool_id, link=d.link, build_name=d.name,
                    raw=d.raw, model_run=run_id, endorsed=not d.ops)
        return body, warnings

    def finish(self, user_id: int):
        self.drafts.pop(user_id, None)
        self.save()


def slot_status(b: Build, catalog) -> dict:
    """Talents against the cap and mantras against their slots for one build.
    The two are interdependent: every non-exempt mantra lowers the talent cap by 2
    (76 - 2m), and talents such as Neuroplasticity and Will o' Wisp (and the oath)
    grant mantra slots."""
    items = [i for i in list(b.talents) + list(b.mantras) if i in catalog.items]
    oath = catalog.oath_of(b) if hasattr(catalog, "oath_of") else None
    u = catalog.mantra_slot_usage(items, oath)
    m = sum(1 for i in items if catalog.lowers_cap(i))
    return dict(talents=catalog.n_talents(items), cap=catalog.talent_cap(items), lowering=m,
                excess=catalog.cap_excess(items), slots=u["slots"], used=u.get("need", u["used"]), over=u["over"],
                oath=oath)


def slot_line(st: dict) -> str:
    """'talents 55/58 (76 - 2x9) · mantras combat 3/3 mobility 1/1 ... wildcard 4/3!':
    mantras past their own slots count against wildcard; overflows marked with !"""
    t = f"talents {st['talents']}/{st['cap']} (76 - 2x{st['lowering']})" + (" !" if st["excess"] > 0 else "")
    ms = " ".join(f"{k.lower()} {st['used'].get(k, 0)}/{n}" + ("!" if st["used"].get(k, 0) > n else "")
                  for k, n in st["slots"].items() if n or st["used"].get(k, 0))
    return f"{t} · mantras {ms}"


def parse_numbers(spec: str, total: int, what: str = "item") -> List[int]:
    """Sheet numbers from "3", "1-4", "2,5 7" or "all", in order, without repeats,
    each checked against 1..total."""
    s = (spec or "").strip().casefold()
    if not s:
        raise DraftError(f"Give {what} numbers: 3, 1-4, 2,5,7.")
    if s == "all":
        return list(range(1, total + 1))
    out: List[int] = []
    for part in s.replace(",", " ").split():
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                nums = range(lo, hi + 1)
            else:
                nums = [int(part)]
        except ValueError:
            raise DraftError(f"'{part}' is not a {what} number or range (use 3, 1-4).")
        for n in nums:
            if n < 1 or n > total:
                raise DraftError(f"No {what} number {n}; the sheet lists 1-{total}.")
            if n not in out:
                out.append(n)
    return out


def describe_op(o: dict, labels: Optional[Dict[str, str]] = None) -> str:
    """One line for a draft listing."""
    from .wire import REASON_LABELS
    labels = labels or REASON_LABELS
    tail = f"  // {o['note']}" if o.get("note") else ""
    if o["action"] == "iffy":
        if o.get("reason"):
            return (f"iffy {o['item_id']} ({o['strength']}): {labels.get(o['reason'], o['reason'])} "
                    f"{', '.join(o.get('related') or [])}" + tail)
        return f"iffy {o['item_id']}" + tail
    rel = f" {', '.join(o['related'])}" if o.get("related") else ""
    return f"{o['action']} {o['item_id']} ({o['strength']}): {labels.get(o['reason'], o['reason'])}{rel}" + tail


NON_CLASH_REASONS = {Reason.LESS_NECESSARY_SPACE, Reason.LUXURY_LESS, Reason.NO_SYNERGY}


def _is_central(svc, name: str) -> bool:
    """A confirmed central talent (admin category), read from the service."""
    try:
        cats = svc.categories()
    except Exception:
        return False
    return (cats or {}).get(name) == "central"


def describe_note(n: dict) -> str:
    """One line for a note: # archetype a: x, y / # weapons: Heavy / // text."""
    k = n.get("kind")
    if k == "archetype":
        return f"# {n.get('archetype')}: {', '.join(n.get('items') or [])} (indicative components)"
    if k == "weapons":
        return f"# weapons: {', '.join(n.get('classes') or [])}"
    if k == "weapon":
        return f"# weapon: {', '.join(n.get('weapons') or [])} (the build equips these)"
    if k == "is":
        return f"# is: {n.get('archetype')} (the build is this archetype)"
    if k == "not":
        return f"# not: {n.get('archetype')} (the build is not this archetype)"
    if k == "core":
        return f"# core: {', '.join(n.get('items') or [])} (the build stands on these)"
    if k == "avoid":
        return f"# avoid: {', '.join(n.get('items') or [])} (builds like this should not take these)"
    return f"// {n.get('text', '')}"
