"""The rule language: read a selection rule written in the symbols tri prints.

    [C] A = chime ∧ Flamecharm ≥ 40 ∧ |D(Flame Leap)| ≥ 1 ⇒ Ignition
    [X] oath = Silentheart ⇒ ¬Prediction
    [F] ⊤ ⇒ Speed Demon

Grammar (math §6.13; ASCII forms in brackets):

    rule  ::= [tag] lhs ⇒ rhs [stats]          ⇒ [=>, ->]
    tag   ::= [F] | [C] | [X]                  foundational, central, exclusion;
                                               default [C], or [X] when rhs is ¬t
    lhs   ::= ⊤ | lit (∧ lit)*                 ⊤ [true, always], ∧ [&, and, ,]
    lit   ::= t | ¬t                           a pick, its absence  ¬ [!, not]
            | s ≥ n | s < n                    a highest stat or a stat total
                                               ≥ [>=], ≤ n = < n+1, > n = ≥ n+1
            | |G(c)| ≥ n | |D(k)| ≥ n          picks in a known group (bars optional)
            | f = v | f ≠ v                    f in oath, murmur, origin, weapon,
                                               A [archetype];  ≠ [!=]
    rhs   ::= t | ¬t

Anything after two spaces following rhs (the "P 0.92 ≥ 0.85 · n 340" tail tri
prints) is ignored, so a printed rule can be pasted back. Names match
case-insensitively (talents, mantras, stats, groups, archetypes).
`format_rule` writes a rule back; `parse(format_rule(r))` gives r's
conditions again.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from tri.rules.defs import Condition

FIELDS = {"oath": "oath", "murmur": "murmur", "origin": "origin", "weapon": "weapon",
          "a": "archetype", "archetype": "archetype"}
TAGS = {"F": "foundational", "C": "central", "X": "exclusion"}
WEAPONS = ("Heavy", "Medium", "Light")


class RuleSyntaxError(ValueError):
    """A rule that does not parse, with the reason."""


@dataclass
class Parsed:
    pass_: str                     # foundational | central | exclusion
    item: str
    effect: str                    # take | never
    conds: List[Condition]


def _key(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(s).casefold())


def rule_id(key: str) -> str:
    """Short stable id of a rule key (PassRule.key)."""
    return hashlib.sha1(key.encode()).hexdigest()[:6]


class Vocabulary:
    """Names the parser resolves: items, stats, aggregates, groups, archetypes,
    and the context values seen in the corpus (for their spelling)."""

    def __init__(self, catalog, groups: Iterable[str] = (), archetypes: Iterable[str] = (),
                 tokens: Iterable[str] = (), aliases: Optional[Dict[str, str]] = None):
        from tri.rules.passes import AGGREGATES
        self.c = catalog
        self.items = {_key(i): i for i in catalog.ids}
        self.stats = {_key(s): ("stat", s) for s in catalog.stats}
        self.stats.update({_key(n): ("agg", n) for n, *_ in AGGREGATES})
        self.groups = {_key(g): g for g in groups}
        self.arch = {_key(a): a for a in archetypes}
        for a, b in (aliases if aliases is not None else {"pve": "boss pve", "bosspve": "boss pve"}).items():
            if _key(b) in self.arch:
                self.arch.setdefault(_key(a), self.arch[_key(b)])
        self.values: Dict[str, Dict[str, str]] = {}
        for t in tokens:
            if ":" in t:
                f, v = t.split(":", 1)
                self.values.setdefault(f, {})[_key(v)] = v

    def item(self, name: str) -> str:
        k = _key(name)
        if k not in self.items:
            raise RuleSyntaxError(f"unknown talent or mantra: {name.strip()!r}")
        return self.items[k]


_SPLIT_AND = re.compile(r"\s*(?:∧|&&|&|\band\b|,)\s*", re.I)
_ARROW = re.compile(r"\s*(?:⇒|=>|->|→)\s*")
_NEG = re.compile(r"^\s*(?:¬|!|not\s+)\s*", re.I)
_CMP = re.compile(r"^(.*?)\s*(≥|>=|≤|<=|>|<)\s*(-?\d+)\s*$")
_EQ = re.compile(r"^(.*?)\s*(≠|!=|=)\s*(.+?)\s*$")


def _lit(s: str, V: Vocabulary) -> Condition:
    s = s.strip()
    m = _CMP.match(s)
    if m:
        name, op, n = m.group(1).strip().strip("|").strip(), m.group(2), int(m.group(3))
        if op in ("≤", "<="):
            op, n = "<", n + 1
        elif op == ">":
            op, n = ">=", n + 1
        op = ">=" if op in ("≥", ">=") else "<"
        g = re.match(r"^([GD])\s*\((.+)\)$", name)
        if g:
            gk = _key(f"{g.group(1)}({g.group(2)})")
            if gk not in V.groups:
                raise RuleSyntaxError(f"unknown group: {name!r} (groups are G(<game category>) and "
                                      f"D(<core talent>); see /rules groups)")
            return Condition("group", V.groups[gk], op, float(n))
        k = _key(name)
        if k not in V.stats:
            raise RuleSyntaxError(f"unknown stat: {name!r}")
        kind, key = V.stats[k]
        return Condition(kind, key, op, float(n))
    m = _EQ.match(s)
    if m and _key(m.group(1)) in FIELDS:
        f, neg, v = FIELDS[_key(m.group(1))], m.group(2) != "=", m.group(3).strip()
        if f == "archetype":
            if _key(v) not in V.arch:
                known = ", ".join(sorted(V.arch.values())) or "none yet (add locators)"
                raise RuleSyntaxError(f"unknown archetype: {v!r} (known: {known})")
            v = V.arch[_key(v)]
        elif f == "weapon":
            hit = [w for w in WEAPONS if _key(w) == _key(v)]
            if not hit:
                raise RuleSyntaxError(f"weapon is Heavy, Medium or Light, not {v!r}")
            v = hit[0]
        else:
            v = V.values.get(f, {}).get(_key(v), v)
        return Condition("context", f"{f}:{v}", "not" if neg else "has")
    neg = bool(_NEG.match(s))
    name = _NEG.sub("", s, count=1) if neg else s
    return Condition("item", V.item(name), "not" if neg else "has")


def parse(text: str, V: Vocabulary) -> Parsed:
    """One rule from its text; RuleSyntaxError says what is wrong."""
    s = " ".join(str(text).replace("\u00a0", " ").split("\n")).strip()
    tag = None
    m = re.match(r"^\[\s*([FCX])\s*\]\s*", s, re.I)
    if m:
        tag = m.group(1).upper(); s = s[m.end():]
    parts = _ARROW.split(s, maxsplit=1)
    if len(parts) != 2:
        raise RuleSyntaxError("a rule needs an arrow: conditions ⇒ item (or =>)")
    lhs, rhs = parts[0].strip(), parts[1]
    rhs = re.split(r"\s{2,}|\s+P\s+\d", rhs.strip(), maxsplit=1)[0].strip()   # drop the printed tail
    neg = bool(_NEG.match(rhs))
    item = V.item(_NEG.sub("", rhs, count=1) if neg else rhs)
    conds: List[Condition] = []
    if lhs and _key(lhs) not in ("", "true", "always") and lhs != "⊤":
        for p in _SPLIT_AND.split(lhs):
            if p.strip() and p.strip() != "⊤":
                conds.append(_lit(p, V))
    if any(c.kind == "item" and c.key == item for c in conds):
        raise RuleSyntaxError(f"{item} cannot be a condition of its own rule")
    effect = "never" if neg else "take"
    if tag == "X" and not neg:
        raise RuleSyntaxError("[X] rules say what a build never takes: write ⇒ ¬item")
    if neg and tag in ("F", "C"):
        raise RuleSyntaxError("⇒ ¬item is an exclusion: tag it [X]")
    pass_ = TAGS[tag] if tag else ("exclusion" if neg else "central")
    seen, uniq = set(), []
    for c in conds:
        if c.id() not in seen:
            seen.add(c.id()); uniq.append(c)
    return Parsed(pass_, item, effect, uniq)


def format_rule(pass_: str, item: str, effect: str, conds: Sequence[Condition]) -> str:
    """The rule in symbols, without statistics (what parse reads)."""
    from tri.rules.symbolic import PASS_TAG, cond, conj
    tag = "X" if effect == "never" else PASS_TAG.get(pass_, "C")
    return f"[{tag}] {conj(cond(c) for c in conds)} ⇒ {'¬' if effect == 'never' else ''}{item}"
