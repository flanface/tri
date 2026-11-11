"""Authored selection rules (v3.0): rules people write in the rule language
(tri.rules.language), and suppressions of mined rules.

    ar = AuthoredRules()
    e = ar.add("[C] A = chime ∧ Flamecharm ≥ 40 ⇒ Ignition", vocab, author="pom", note="core")
    ar.edit(e.rule_id, "[C] A = chime ∧ Flamecharm ≥ 50 ⇒ Ignition", vocab)
    ar.suppress("3fa2c1", author="pom", note="nonsense")   # a mined rule's id
    ar.remove(e.rule_id)

An entry is {rule_id, op, text, author, ts, note}: op "add" carries a rule,
op "suppress" names a mined rule's id (tri.rules.language.rule_id of its key).
Entries live in the run (authored_rules.json) and, for the bot, in the Supabase
table tri_rules. The selection (tri.rules.selection) reads them; nothing here
touches the corpus. Adding never rejects a rule for its corpus statistics; the
caller shows them (Rulebook.rule_stats) so the author can judge.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from tri.rules.language import Parsed, Vocabulary, parse, rule_id

FILE = "authored_rules.json"


@dataclass
class AuthoredEntry:
    rule_id: str
    op: str                        # add | suppress
    text: str = ""                 # add: the rule as written (normalised)
    author: str = ""
    ts: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AuthoredEntry":
        """Also reads database rows (v3.0.3): ts as an ISO timestamp, null author / note."""
        kw = {k: d[k] for k in ("rule_id", "op", "text", "author", "ts", "note") if k in d}
        for k in ("text", "author", "note"):
            if kw.get(k) is None and k in kw:
                kw[k] = ""
        if "ts" in kw:
            kw["ts"] = _ts(kw["ts"])
        return AuthoredEntry(**kw)


def _ts(v) -> float:
    """Seconds since the epoch from a float, a numeric string or an ISO timestamp."""
    if v is None or v == "":
        return time.time()
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def to_pass_rule(p: Parsed, author: str = "", note: str = ""):
    """A parsed rule as a PassRule (source "authored"; statistics filled by the selection)."""
    from tri.rules.passes import PassRule
    return PassRule(p.pass_, p.item, list(p.conds), confidence=float("nan"), bound=float("nan"),
                    support=0.0, effect=p.effect, source="authored",
                    note=(f"by {author}" if author else "by hand") + (f": {note}" if note else ""))


def key_of(p: Parsed) -> str:
    return f"{p.effect}|{p.item}|" + "&".join(sorted(c.id() for c in p.conds))


class AuthoredRules:
    """The authored rule set: ordered entries, newest last."""

    def __init__(self, entries: Optional[List[AuthoredEntry]] = None):
        self.entries: List[AuthoredEntry] = list(entries or [])

    # ------------------------------------------------------------ edits
    def add(self, text: str, vocab: Vocabulary, author: str = "", note: str = "") -> AuthoredEntry:
        """Parse and add a rule (RuleSyntaxError if it does not parse). The same
        rule again replaces the old entry."""
        from tri.rules.language import format_rule
        p = parse(text, vocab)
        rid = rule_id(key_of(p))
        self.entries = [e for e in self.entries if e.rule_id != rid]
        e = AuthoredEntry(rid, "add", format_rule(p.pass_, p.item, p.effect, p.conds), author, time.time(), note)
        self.entries.append(e)
        return e

    def edit(self, rid: str, text: str, vocab: Vocabulary, author: str = "", note: str = "") -> AuthoredEntry:
        """Replace an authored rule, or override a mined rule: the mined one is
        suppressed and the new text added."""
        old = self.get(rid)
        if old is None or old.op != "add":
            self.suppress(rid, author=author, note=note or "edited")
        else:
            self.remove(rid)
        return self.add(text, vocab, author=author, note=note or (old.note if old else ""))

    def suppress(self, rid: str, author: str = "", note: str = "") -> AuthoredEntry:
        """Switch off a mined rule by its id."""
        self.entries = [e for e in self.entries if not (e.rule_id == rid and e.op == "suppress")]
        e = AuthoredEntry(rid, "suppress", "", author, time.time(), note)
        self.entries.append(e)
        return e

    def remove(self, rid: str) -> bool:
        """Drop an authored rule or lift a suppression."""
        n = len(self.entries)
        self.entries = [e for e in self.entries if e.rule_id != rid]
        return len(self.entries) < n

    def get(self, rid: str) -> Optional[AuthoredEntry]:
        for e in reversed(self.entries):
            if e.rule_id == rid:
                return e
        return None

    # ------------------------------------------------------------ reading
    @property
    def suppressed(self) -> Dict[str, AuthoredEntry]:
        return {e.rule_id: e for e in self.entries if e.op == "suppress"}

    def rules(self, vocab: Vocabulary) -> List[tuple]:
        """[(entry, PassRule)] of the add entries that still parse (a renamed
        item drops its rule, listed in .broken)."""
        out, self.broken = [], []
        for e in self.entries:
            if e.op != "add":
                continue
            try:
                out.append((e, to_pass_rule(parse(e.text, vocab), e.author, e.note)))
            except ValueError as err:
                self.broken.append((e, str(err)))
        return out

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------ persistence
    def to_json(self) -> List[dict]:
        return [e.to_dict() for e in self.entries]

    @staticmethod
    def from_json(d) -> "AuthoredRules":
        return AuthoredRules([AuthoredEntry.from_dict(x) for x in (d or [])])

    def save(self, run_dir: str) -> str:
        p = os.path.join(run_dir, FILE)
        with open(p, "w") as f:
            json.dump(self.to_json(), f, indent=1, ensure_ascii=False)
        return p

    @staticmethod
    def load(run_dir: str) -> "AuthoredRules":
        p = os.path.join(run_dir, FILE)
        if not os.path.exists(p):
            return AuthoredRules()
        with open(p) as f:
            return AuthoredRules.from_json(json.load(f))
