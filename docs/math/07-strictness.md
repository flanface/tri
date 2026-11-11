# 7. Strictness

## 7.1 Definition

> **Strictness.** Every answer tri gives (an analysis, a completion, or a review it accepts) obeys the game's hard constraints: staged availability (each pick taken at one of the build's stages, its prerequisites by the same stage), prerequisites, mutual exclusions, and the talent cap of $76-2m$. Every rule that applies to the answer is either satisfied or listed as a violation with its reason. Nothing is broken silently.

## 7.2 Interpretation

Strictness is the difference between a recommender and a checker. A recommender can be "mostly right". A build tool cannot propose a build the game would refuse, and an analysis that quietly drops a rule gives the user no way to know it was dropped. Strictness makes every gap visible.

In data science terms, strictness is **constrained optimisation with an audit trail**: worth is the objective, the hard constraints are the feasible set, the rules are soft constraints with a fixed priority, and every unmet constraint is reported.

## 7.3 Authority

Sources are ordered, highest first:

$$
\text{hard constraints}\ \succ\ \text{binding reviews}\ \succ\ \text{global review rules}\ \succ\ \text{categories}\ \succ\ \text{certified rules}\ \succ\ \text{worth } q. \tag{7.1}
$$

No source can make an answer infeasible. A binding review overrides a rule; a rule overrides worth. Vetoes remove a rule for the builds they reach; they add nothing of their own.

**Categories** are an administrator's say on an item for every build. They are stored in the run (`tri.json`, set in code with `Tri.set_categories`):

| category | meaning |
|---|---|
| foundational | taken on every build (mantras not on Silentheart); if its requirements are not met at any stage, the answer lists it as a missing pick |
| central | taken whenever its requirements are met |
| luxury | never required; taken when a slot is free, before other fill; never dropped as no fit, swapped out or marked iffy |

A category replaces the mined pass of its item and silences the item's mined never rules, so the expert's word outranks the corpus. It does not outrank reviews: a binding review that removes the item on a similar build wins there, because reviews are fully accurate for what they cover. Categories change no coordinate and no worth, so they cannot move other builds in the field.

**Sidegrades** ([Section 4.9](04-concept.md)) sit with worth: they only restrict which worth adds are made, and offer the other side.

**Global review rules** are rules induced from many reviews (`RuleInducer`): broken, redundant, conflict, necessary, no fit, prefer, synergy, depends, sidegrade. Only broken, redundant and conflict apply to every build; the rest are archetype-relative and act through the ledger's reach, and are shown for inspection. Disputed ones are refined by the single best separating condition. Overturns count as counterexamples; so do removals of $t$ on builds with its partner (synergy), endorsements of $t$ on builds lacking a component (depends), and one item swapped for the other (sidegrade).

## 7.4 Analysing a full build

The build's slot counts are its capacity: slots start full. tri produces one answer per plausible region (membership at least 0.2), in these steps (`PassEngine.review`):

0. **Cap.** If the build is over the talent cap, run CapRepair (Section 7.5).
1. **Reject** picks that a binding review or a global review rule rejects.
2. **Exclude** picks that a never rule forbids in this context. Then **needs**: for each binding NW claim $(X,C)$, add $C\setminus b$ if it fits in the room the earlier steps freed, else remove $X$ (5.3).
3. **Require.** Add missing foundational and central items, binding review adds and binding GG talents, re-reading anchors up to 4 times (a new pick can switch on another central rule). Each add displaces the least worthy unprotected luxury pick of its kind, then CapRepair runs. A rule-driven or GG add that a never rule forbids is not made; a binding review add is. If a required item is blocked only by an exclusion, the **exclusive swap** removes its blockers (reason *works against*) and their dependents (reason *less necessary, make space*), provided none is protected; otherwise the item is reported missing.
4. **No fit.** Drop unprotected picks whose worth is below 0.05.
5. **Refill** free slots with the worthiest feasible items; no talent or counted mantra is added if it would exceed the cap.
6. **Swap** a luxury pick for one at least 0.15 worthier, up to 3 times; the pair is reported as *worth less / more as a luxury* (LL, LM). Worth never adds an item next to its sidegrade (SG).
7. **Iffy.** Mark kept luxury picks within 0.08 of the best alternative, and kept picks with a binding sidegrade claim (shown with the other item).
8. **Repair.** Re-check every rule and the cap against the result and fix, up to 6 rounds. A swap can add an anchor that switches on a take rule or a never rule.
9. **Violations.** List every constraint still unmet (Section 7.7).

Each change records its pass, its source (review, review rule, rules, worth), the review id, and the rule in symbols (Appendix C), for example `[C] HeavyWeapon ≥ 25 ∧ oath ≠ Silentheart ⇒ V1_0_1   P 0.96 ≥ 0.93 · n 370`. Reviewers can overturn any numbered change ([Section 2.5](02-constraints.md)).

## 7.5 CapRepair

While the selection is over the cap: among the **unprotected** picks (not pinned by the user, not required by a rule or binding review, not a prerequisite of another pick), remove the talent of least worth; if there are no unprotected talents, remove the counted mantra of least worth; if there is neither, stop. Each removal is reported as "less necessary (make space): over the talent cap ($n$ talents, cap $76-2m$)".

**Proposition 7.1 (CapRepair ends).** CapRepair makes at most as many removals as the starting excess, and ends either within the cap or with every remaining talent and counted mantra protected.

*Reason.* Each removal lowers the excess by at least 1 ([Section 2.3](02-constraints.md)), and the excess is a whole number. $\square$

## 7.6 Completing a partial build

Keep the picks (after CapRepair if over the cap), then fill in pass order (foundational, central with anchors re-read, then by worth) up to capacity, adding an item only if the selection stays feasible and no never rule forbids it. The archetype uses the partial evidence ([Section 3.3](03-worth.md)). Capacity defaults to the neighbours' median.

**Mantras first.** If the partial build has no mantras and is not Silentheart, completion fills mantras before talents:

1. add every required item that is a mantra or a mantra-slot talent (Neuroplasticity, Will o' Wisp, Chorus of Souls, read from the game data's "Mantra Slot" stats), in pass order;
2. add the remaining mantra-slot talents by worth, if feasible and within the cap;
3. add mantras by worth up to the mantra capacity, which is the larger of the neighbours' median and the median mantra count of corpus builds that have mantras.

Then the talents are filled as usual. A Silentheart build gets no mantras. Each step keeps the selection feasible and within the cap, so the guarantees below are unchanged.

**Never rules on kept picks.** A kept pick $t$ with a firing never rule $[X]\ \phi\Rightarrow\neg t$ is removed first, least worth first, until no kept pick is forbidden. Manual categories are exempt.

**Reviews of the result.** Claims bind by similarity to the reviewed build ([Section 5.5](05-locality.md)), and a sparse input $b$ is far from the full builds the reviews were made on, so a review of an earlier completion does not reach $b$. Let $\mathcal C(b)$ be completion given the claims $M$, and $M(b')$ the binding claims on $b'$ (scoped claims, [Section 5.5](05-locality.md); a transfer binds at similarity $\ge 0.95$). Then

$$a_0=\mathcal C(b;\,M(b)),\qquad a_{k+1}=\mathcal C\big(b;\,M(b)\cup E_k\big),\qquad E_k=\bigcup_{j\le k}\{(x,v)\in M(a_j):\ v=\text{add},\,x\notin a_j\ \ \lor\ \ v=\text{remove},\,x\in a_j\setminus b\}$$

stopping when $E_k$ gains nothing, $a_{k+1}\in\{a_0,\dots,a_k\}$, or $k=$ `review_rounds` (3). Claims on $b$ itself win over carried ones, and $b$'s own picks stay. Each $a_k$ is a completion, so the guarantees below hold for it.

## 7.7 The strictness theorem

The **violation list** of an answer contains:

| kind | when |
|---|---|
| missing | a take rule applies but no slot can be freed (slots full of protected picks) or prerequisites block it |
| forbidden | a never rule applies to a pick kept by a binding review or a user pin |
| conflict | a take and a never rule both apply; the never rule won |
| unanchored | a take rule applies only through a kept pick of low worth (below `anchor_min_worth`), which by design does not pull in its own archetype's central items |
| talent_cap | the answer is over the cap and every remaining talent and counted mantra is protected |

**Theorem 7.2 (strictness).** For every answer of analysis or completion:

1. every certified or global review rule that applies to the answer is either satisfied or in its violation list;
2. every pick is available, has its prerequisites, and excludes no other pick;
3. the answer is within the talent cap, or its violation list has a talent_cap entry.

*Reason.* (1) The last step evaluates every rule against the final answer itself, with every kept pick as an anchor, and lists each unsatisfied one. (2) Every add passes the feasibility test. Removals cannot break availability or exclusions, and they cannot break prerequisites because prerequisites of kept picks are protected, except in the exclusive swap, which removes dependents with blockers (Proposition 2.1). (3) Every ordinary add checks the cap; binding adds and the input are followed by CapRepair, which ends within the cap or with everything protected (Proposition 7.1), which is exactly the talent_cap entry. $\square$

Theorem 7.2 was checked on every build of the synthetic worlds, including inputs set two talents over their cap.

## 7.8 Strict reviews

The same constraints bind reviewers. A review is accepted only if its corrected build obeys the talent cap, adds nothing that excludes a remaining pick, and strands no prerequisite. A reviewer who adds a talent to a build at its cap must also remove a talent or a counted mantra. A reviewer who adds an item exclusive with a pick must remove that pick and everything that depends on it. Competitions are held to the same cap.

## 7.9 Worked example

A build has 5 counted mantras (cap 66) and 67 talents. A strong binding review on a similar build adds talent X.

1. Step 0: excess 1. CapRepair removes the least worthy unprotected talent L1. Now 66 talents.
2. Step 3: X is required, but it excludes pick Y, and pick Z requires Y. The exclusive swap removes Y (reason *works against X*) and its dependent Z (reason *less necessary, make space*). Neither is protected, so the swap goes ahead and X is added. The build now has 65 talents, within the cap, so CapRepair has nothing to do.
3. Step 9: if a central rule needed Z, the final check lists it as missing with the reason, rather than silently losing it.

## 7.10 Central picks

**Definition.** A pick is **central** if it defines the build's playstyle: an attunement path or a signature talent such as Blighted Song. Few talents are central. tri confirms a central pick only from experts: a `central` category stored in the run, or the components of a reviewer's indicator or core note, or a take tell of the build's family on which reviewers agree ([Section 8](08-convergence.md).5a, $c\,p\,a\ge0.3$) (luxury-category items excluded). A mined take rule does not confirm one; such picks are reported under the pass **archetype** (necessary for the build's archetype, not playstyle-defining).

**Rule (respect).** Let $Z$ be the confirmed central picks of a build. Then

1. no never rule without a co-pick condition, no space trim, no luxury swap and no no-synergy removal removes $t\in Z$;
2. a clash, a never rule naming a co-pick, or a review reason RW, WA, BR, SM or NW, may remove $t\in Z$;
3. under the talent cap, central picks are trimmed last, and only when nothing else can go.

Reviewers are held to the same rule: a review removal of a confirmed central pick must give a clash reason.

**Proposition 7.3.** If a build's picks contain no clash for $t\in Z$, every answer keeps $t$ whenever the talent cap can be met without it being removed.

*Reason.* Items 1 and 3 above remove every non-clash path to dropping $t$; the only remaining one is the cap trim, which reaches $Z$ only after all unprotected picks. $\square$

Code: `PassEngine._central`, `_clash_rule`, `_review_one`, `complete`, `_complete_once`.
