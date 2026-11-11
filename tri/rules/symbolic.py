"""Symbolic notation for rules and facts.

One notation for every rule and fact the model states, so the bot
and the docs show the same thing. Read it with docs/plain ("Reading the
symbols") or docs/math (Appendix C).

    lhs ⇒ rhs        if the left holds, then the right
    ∧  ¬  ⊤          and, not / without, always (no condition)
    a ⊥ b            a works against b
    a ≻ b            a is preferred over b (when space is short)
    A                the build's archetype
    P(t | A)         share of A builds (that could take t) that take t
    w(t)             worth of t: how likely a build like this one is to take it
    Δ, →             change; before → after
    ∀                every
    x | c            x, given the condition c
    n, p             count; chance the statement is a fluke

Tags in front of a rule: [F] foundational, [C] central, [X] exclusion
(corpus rules, certified by a lower or upper bound); [R] a rule induced from
reviews, with its status.

Functions return plain strings (Unicode, no markup), short enough for a
Discord line. Nothing here changes a rule or a fact; it only writes them.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

LEGEND = (("⇒", "then"), ("∧", "and"), ("¬", "not / without"), ("⊤", "always"), ("⊥", "works against"),
          ("≻", "preferred over"), ("A", "the build's archetype"), ("P(t | A)", "share of A builds taking t"),
          ("w(t)", "worth of t"), ("Δ", "change"), ("→", "before → after"), ("∀", "every"),
          ("| c", "given c"), ("n", "count"), ("p", "chance of a fluke"))

PASS_TAG = {"foundational": "F", "central": "C", "exclusion": "X", "luxury": "L"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def legend(compact: bool = True) -> str:
    """The symbols in one line (compact) or one per line."""
    sep = " · " if compact else "\n"
    return sep.join(f"{s} {m}" for s, m in LEGEND)


def num(x: Optional[float], d: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pval(p: Optional[float]) -> str:
    """p 0.012, p < 10⁻⁶; "p –" when not testable yet."""
    if p is None:
        return "p –"
    if p < 1e-15:
        return "p < 10⁻¹⁵"
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"p < 10{str(e + 1).translate(_SUP)}"
    return f"p {p:.3f}"


# ------------------------------------------------------------------ conditions
def cond(c) -> str:
    """A rulebook Condition (tri.passes) as a symbol: Strength ≥ 40, oath = X,
    oath ≠ X, item, ¬item."""
    if c.kind in ("stat", "agg"):
        return f"{c.key} {'≥' if c.op == '>=' else '<'} {int(c.value)}"
    if c.kind == "group":
        return f"|{c.key}| {'≥' if c.op == '>=' else '<'} {int(c.value)}"
    if c.kind == "context":
        f, v = c.key.split(":", 1)
        if f == "archetype":
            f = "A"
        return f"{f} {'=' if c.op == 'has' else '≠'} {v}"
    return c.key if c.op == "has" else f"¬{c.key}"


def token(s: Optional[str], region_names: Sequence[str] = ()) -> str:
    """A review-rule condition string (tri.rules.context_features) as a symbol:
    "has:X" → X, "Strength>=40" → Strength ≥ 40, "oath:v" → oath = v,
    "not ..." → negation."""
    if not s:
        return "⊤"
    if s.startswith("not "):
        inner = s[4:]
        if inner.startswith("has:"):
            return f"¬{inner[4:]}"
        t = token(inner, region_names)
        return t.replace(" = ", " ≠ ").replace(" ≥ ", " < ") if (" = " in t or " ≥ " in t) else f"¬({t})"
    if s.startswith("has:"):
        return s[4:]
    if ">=" in s:
        a, b = s.split(">=", 1)
        return f"{a} ≥ {b}"
    if ":" in s:
        f, v = s.split(":", 1)
        if f == "region" and v.isdigit() and int(v) < len(region_names):
            v = region_names[int(v)]
        return f"{f} = {v}"
    return s


def conj(parts: Iterable[str]) -> str:
    parts = [p for p in parts if p]
    return " ∧ ".join(parts) if parts else "⊤"


# ------------------------------------------------------------------ rules
def pass_rule(r, region_names: Sequence[str] = ()) -> str:
    """A rulebook PassRule:
    [F] Strength ≥ 40 ∧ Flame Leap ⇒ Ignition   P 0.92 ≥ 0.85 · n 340
    [X] oath = Silentheart ⇒ ¬Mantra X   P 0.01 ≤ 0.02 · n 120 · else 0.40"""
    lhs = conj(cond(c) for c in r.conds)
    st = f" · {r.status}" if getattr(r, "status", "active") != "active" else ""
    if r.effect == "never":
        tail = f" · else {num(r.contrast)}" if r.contrast is not None else ""
        return f"[X] {lhs} ⇒ ¬{r.item}   P {num(r.confidence)} ≤ {num(r.bound, 3)} · n {r.support:.0f}{tail}{st}"
    tag = PASS_TAG.get(r.pass_, "?")
    return f"[{tag}] {lhs} ⇒ {r.item}   P {num(r.confidence)} ≥ {num(r.bound)} · n {r.support:.0f}{st}"


_REVIEW_HEAD = {
    "REDUNDANT": lambda t, o: f"{o} ⇒ ¬{t}",
    "CONFLICT": lambda t, o: f"{t} ⊥ {o}",
    "BROKEN": lambda t, o: f"⊤ ⇒ ¬{t}",
    "PREFER": lambda t, o: f"{o} ≻ {t} | space short",
    "NECESSARY": lambda t, o: f"A = {o} ⇒ {t}" if o else f"A ⇒ {t}",
    "GENERAL": lambda t, o: f"⊤ ⇒ {t}",
    "NO_FIT": lambda t, o: f"A ⇒ ¬{t}",
    "OFF_ARCHETYPE": lambda t, o: f"{o} ⇒ ¬{t}",
    "SYNERGY": lambda t, o: f"A ∧ {o} ⇒ {t}",
    "DEPENDS": lambda t, o: f"A ∧ ¬({' ∧ '.join(o.split(' + '))}) ⇒ ¬{t} ≡ {t} ⇒ {' ∧ '.join(o.split(' + '))} | space",
    "SIDEGRADE": lambda t, o: f"{t} ∼ {o}",
}


def review_rule(r, region_names: Sequence[str] = ()) -> str:
    """A rule induced from reviews (tri.rules.Rule):
    [R supported] Agile Dash ⊥ Shoulder Bash | Strength ≥ 40   +3.0 −0.5 · lb 0.82"""
    head = _REVIEW_HEAD[r.kind](r.subject, r.other)
    if r.condition:
        c = token(r.condition, region_names)
        head = f"{head} ∧ {c}" if " | " in head else f"{head} | {c}"
    s, c = sum(x.weight for x in r.support), sum(x.weight for x in r.counter)
    return f"[R {r.status}] {head}   +{s:.1f} −{c:.1f} · lb {r.precision_lb:.2f}"


# ------------------------------------------------------------------ facts
def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fact(f, with_p: bool = True) -> str:
    """A certified fact (tri.facts.Fact) as one symbolic line."""
    n, A, its = f.numbers or {}, f.archetype or "A", list(f.items or [])
    t = its[0] if its else "t"
    fam = f.family
    if fam == "defining":
        s = f"P({t} | {A}) {num(n.get('rate'))}, P({t} | ¬{A}) {num(n.get('others'))}   ×{n.get('lift', 0):.1f}"
    elif fam == "separating":
        B = n.get("other", "B")
        sh = n.get("shares") or []
        terms = " + ".join(f"{x}" for x in its) or t
        s = f"{A} ↔ {B}: {terms} = {_pct(n.get('share', 0))} of d²" + \
            (f" ({', '.join(_pct(v) for v in sh)})" if len(sh) > 1 else "")
    elif fam == "threshold":
        s = (f"{A}: w({t}) ≥ ½ ⇔ {n.get('stat')} ≥ {n.get('threshold')} (req {n.get('requirement')})   "
             f"P {num(n.get('above'))} vs {num(n.get('below'))}")
    elif fam in ("hole", "trap"):
        s = f"{A}: hole {t}  P({t}) {num(n.get('corpus'))}, reviews −{t} {n.get('removed')}/{n.get('reviewed')}"
    elif fam == "gem":
        s = f"{A}: gem {t}  P({t}) {num(n.get('corpus'))}, reviews +{t} {n.get('added')}/{n.get('reviewed')}"
    elif fam in ("synergy", "anti-synergy"):
        a, b = (its + ["s", "t"])[:2]
        s = (f"{A}: {a} ⇒ Δw({b}) {n.get('delta', 0):+.2f}   "
             f"P({b} | {a}) {num(n.get('with_'))}, P({b} | ¬{a}) {num(n.get('without'))}")
    elif fam == "universal":
        s = f"∀A ≠ {A}: w({t}) ≥ 0.9; w({t} | {A}) {num(n.get('worth_except'))}"
        if n.get("instead"):
            s += f"; {A} ⇒ {n['instead']}"
    elif fam == "shift":
        parts = [f"P({c['item']}) {num(c['before'])} → {num(c['after'])}" for c in n.get("changes") or []]
        if n.get("moved_far"):
            parts.append(f"Δu {num(n.get('moved'))}")
        s = f"{A} @{n.get('patch')}: " + conj(parts) + f"   n {n.get('post')}"
    else:
        s = f.sentence
    return s + (f" · {pval(f.p)}" if with_p else "")
