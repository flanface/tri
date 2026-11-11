# 2. Builds and the game's constraints

## 2.1 Builds, stats and stages

The **catalog** is the list of all items, read from the game data ([api.deepwoken.co](https://api.deepwoken.co/get?type=all) and [pocamind/data](https://github.com/pocamind/data)). Each item is a talent or a mantra and carries:

- a **requirement**: thresholds on stats, possibly sums of stats, combined with "and" and "or";
- **prerequisites**: groups of items, at least one of each group must also be picked;
- **exclusions**: items it cannot be taken with;
- tags such as rarity, category and, for mantras, a type (Normal, Oath, Monster, Origin, Event).

A build records three stat vectors:

- **pre-shrine stats**: stats at the [Shrine of Order](https://deepwoken.fandom.com/wiki/Deep_Shrines/Shrine_of_Mastery), before it redistributes them;
- **final stats**: the stats at the end;
- **highest stats**: the largest value each stat reached at any time.

Items are only granted at two moments, pre-shrine and final. These are the build's **stages**. A build without pre-shrine stats has one stage. Highest stats describe how much the player invested in each stat. They are used as model features and in rule conditions, but never to decide whether an item can be taken.

**Finer stages (dwt.sh, v2.22).** A [dwt.sh](https://dwt.sh/docs/api) build records its run as an ordered list of stages $\sigma_0, \dots, \sigma_{S-1}$ ($S \le 32$), each the stats at the end of that stage and named by the shrine used to leave it (Shrine of Order, Shrine of Mastery; the last is the finished build). tri reads

$$\text{pre} = \sigma_0, \qquad \text{mid} = (\sigma_1, \dots, \sigma_{S-2}), \qquad \text{final} = \sigma_{S-1}.$$

Every $\sigma_s$ is a state the character really held and picked talents at, so all $S$ stages are stages of the availability rule below. Learning uses only the two ends: highest stats are $h = \max(\sigma_0, \sigma_{S-1})$ elementwise, and the batch matrices carry $\sigma_0$ and $\sigma_{S-1}$ only. A long chain of Shrines of Mastery therefore changes what a build may hold, never what it teaches.

A build also has its **picks** (talents and mantras), its **context** (oath, murmur and origin, written as tokens like `oath:Silentheart` or `origin:Authority Ensign`), and a **weight** (1 unless the pipeline supplies another). The worth field ([Section 3](03-worth.md)) sees only the oath and murmur tokens; the other fields are used only for availability.

## 2.2 Availability

An item is **available** to a build if it can be taken at some stage: that stage meets its whole requirement, and each of its prerequisites can be taken at that stage or an earlier one. Stats are never mixed across stages, and highest stats never count. Index the stages $s=0$ (pre-shrine) to $s=S-1$ (final), with $S=2$ for a planner build and up to 32 for a dwt.sh build; a talent taken before the shrine is kept after it, but a talent whose requirement is met only after the shrine is not there before it. With $o^{s}_{bt}$ equal to 1 when item $t$ can be taken by build $b$ at stage $s$ (taken there or earlier),

$$
o^{s}_{bt}=1 \quad\text{exactly when}\quad o^{s-1}_{bt}\ \text{or}\ \Big[\text{req}_t(\text{stage}_s(b))\ \text{and}\ \text{ctx}_t(b)\ \text{and}\ \text{pre}_t(o^{s}_{b\cdot})\Big],\qquad o^{-1}\equiv0,
$$

$$
a_{bt}=o^{1}_{bt}. \tag{2.1}
$$

**Example (staging).** Magical Resolve needs 40 Willpower and Battle Tendency, which needs 15 Fortitude and 15 Willpower. A build with 40 Willpower and no Fortitude before the shrine, and 15 Fortitude but 20 Willpower after it, meets each requirement at some stage, yet cannot take Magical Resolve: before the shrine Battle Tendency is not obtainable, and after it 40 Willpower is gone. (2.1) gives $o^0_{\text{BT}}=0$, $o^1_{\text{BT}}=1$, and $o^0_{\text{MR}}=o^1_{\text{MR}}=0$. The reverse order (Battle Tendency before the shrine, 40 Willpower after) is allowed. Merging the stages first, as tri did before v2.15.1, wrongly allowed both.

**Context.** Some requirements are not stats. Oath kit needs its oath (Jet Kick needs Jetstriker), some talents need a murmur or an origin, and a Silentheart build cannot use any mantra. Weapon requirements are not modelled: a build's weapon is never checked. tri writes these as a conjunction of clauses over **context tokens** such as `oath:Arcwarder`:

$$
\text{ctx}_t(b)=\bigwedge_{C\in\mathcal C_t,\ \text{field}(C)\in F_b}\ \big(C\cap K_b\neq\varnothing\big)\ \wedge\ \big(t\ \text{is a mantra}\Rightarrow K_b\cap L=\varnothing\big).
$$

- $K_b$ is the build's set of tokens.
- $F_b$ is the set of context fields the build records.
- $\mathcal C_t$ is the item's clause list (`Item.context`).
- $L$ is the set of mantra locks (`Catalog.mantra_locks`, Silentheart).

A clause on a field the build does not record is skipped. A build of unknown oath is therefore not locked out of oath kit, while a build whose oath is recorded as "none" is. The fields are `oath`, `murmur` and `origin` (`Catalog.lock_fields`). A build's weapon is not a lock.

- A build with planner metadata records oath, murmur and origin (an empty value means none). A build without metadata records nothing, so no lock is checked.
- **Formal statement.** item $i$ is context-available to a build with tokens $T$ and recorded fields $R$ when every clause $C$ of $i$ with $\mathrm{fields}(C) \subseteq \mathcal{L}$ and $\mathrm{fields}(C) \cap R \neq \varnothing$ meets $C \cap T \neq \varnothing$. Here $\mathcal{L} = \{\text{oath}, \text{murmur}, \text{origin}\}$. A clause naming any other field (a weapon clause in a catalog saved before v2.11.2) is skipped. The clauses are compiled once per catalog into field and token sets, so checking all $N$ items costs one set intersection per clause.
- **Tokens implied by picks.** Oath, murmur and origin each take one value. If a build leaves one of them unrecorded or empty, a pick that needs exactly one value of it supplies that value. So does an `Oath: X` or `Murmur: X` talent. An Arcwarder build with an empty oath field but Arc Suit picked is then read as Arcwarder, and Jet Kick (Jetstriker only) stays unavailable to it. Only a pick the build already holds can imply a token. Statistics never do.

The clauses come from the game data. A prerequisite group whose alternatives are all granted oath or murmur talents ("requires Oath: Arcwarder") becomes a clause. So do an oath, murmur or origin condition, and an origin reference (`origin:authority_ensign`). A game data sync only ever adds clauses ([Section 8.10](08-convergence.md)), so a source that does not state a lock never lifts one. Nothing is inferred from builds. A saved catalog whose items carry no clauses (one written without game data) gets them on load from a table bundled with the package (`known_context`), so an older run still keeps Jet Kick to Jetstriker builds.

**Never-available items.** An item tagged `retired` (gone from the game data) or `voi_only` (only obtainable in Vow of Iron, the pocamind/data flag `voi_only`) has $\text{req}_t \equiv 0$: tri reads Pathfinder builds, where neither can be picked. The Vow of Iron list is bundled (`tri/data/voi_only.json`, 165 talents and 1 mantra), so a saved catalog without the tag is tagged on load. A build that holds such a pick is a Vow of Iron build, or a planner mistake. Analysis and completion remove the pick and everything that requires it (foundational pass, reason `scenario_mismatch`), review validation refuses adding it, and `Tri.fit` and `refit` leave out every build that holds one, so the field and the rules learn from Pathfinder builds only. deepwoken.co's `VOI` flag means "obtainable in Vow of Iron" (Fire Blade has it) and is not used.

**Prerequisites.** $\text{pre}_t(o^{s}_{b\cdot})$ holds when every required item is obtainable by stage $s$ and each "one of" group has a member obtainable by stage $s$. Because (2.1) refers to itself within a stage, each stage is computed as a fixed point, stage 0 first, then stage 1 with stage 0 held.

**Proposition 2.1.** At stage $s$, start from $a^{(0)}_{bt}=o^{s-1}_{bt}\vee(\text{req}_t(\text{stage}_s)\wedge\text{ctx})$ and repeat $a^{(k+1)}_{bt}=o^{s-1}_{bt}\vee\big(a^{(k)}_{bt}\wedge\text{pre}_t(a^{(k)}_{b\cdot})\big)$. The sequence reaches a fixed point within $N$ steps, and that fixed point is the largest $o^{s}$ consistent with (2.1).

*Proof.* Entries of $o^{s-1}$ never change (they were consistent at stage $s-1$ and $\text{pre}$ is monotone). Each step can only turn other entries off, and there are $N$ entries per build. So the sequence is monotone and finite, and it stops after at most $N$ changes. Suppose $a^\star$ satisfies (2.1). Then $a^\star\le a^{(0)}$. If $a^\star\le a^{(k)}$, monotonicity of $\text{pre}_t$ gives $a^\star=a^\star\wedge\text{pre}(a^\star)\le a^{(k)}\wedge\text{pre}(a^{(k)})=a^{(k+1)}$. By induction $a^\star$ lies below the limit, so the limit is the largest solution. $\square$

**Interpretation.** Without context, an Arcwarder item counts as "available but not taken" for every other oath. The field then spends its largest direction on structural zeros, and the rule miner relearns the oath requirement as an exclusion. With (2.1), those builds drop out of the item's statistics, exactly as builds without the stats do.

**Interpretation.** Availability is the line between "did not take" and "could not take". A build that could not take an item tells us nothing about that item, so every statistic in tri is computed only over builds that could take it. This single choice keeps unavailable items from looking unpopular.

**Why not highest stats?** Highest stats are always at least as large as either stage, and requirements only ask for "at least". So judging by highest stats would mark more items available than the game allows. The stage rule is the conservative and correct one. Corpora recorded before v2.3 have no pre-shrine stats and use highest stats as a stand-in.

All standard requirement thresholds are multiples of 5. [Section 6.3](06-accuracy.md) uses this fact.

## 2.3 The talent cap

A build can hold at most

$$
\text{cap} = 76 - 2m
$$

talents, where $m$ is the number of its mantras that are **not** Oath, Origin (the faction and origin mantras such as Grand Warden's Axe or Hidden Blade) or Monster mantras, following the [Deepwoken Wiki](https://deepwoken.fandom.com/wiki/Talents). A mantra of unknown type counts against the cap. Each counted mantra takes the place of two talents.

The **excess** of a selection is its talent count minus its cap. A selection obeys the cap when the excess is 0 or less.

**Worked example.** A build with 5 Normal mantras, 1 Oath mantra and 1 Monster mantra has $m=5$ and cap $76-10=66$. If it has 67 talents, its excess is 1. Removing one talent fixes it. Removing one Normal mantra raises the cap to 68 and also fixes it, with one slot to spare.

**The one fact the algorithms use.** Adding a talent raises the excess by 1; adding a counted mantra raises it by 2; adding an exempt mantra leaves it unchanged. Removals are the reverse. So every removal of a talent or a counted mantra lowers the excess by at least 1.

Code: `Catalog.talent_cap`, `Catalog.cap_excess`. The numbers 76, 2 and the exempt types are stored in `catalog.json`.

## 2.4 Feasible selections, exclusions and dependents

A selection of picks is **feasible** for a build when:

1. every pick is available (Section 2.2);
2. every pick's prerequisites are met inside the selection;
3. no two picks exclude each other;
4. the talent cap holds;
5. the mantras fit the build's mantra slots (Section 2.6).

Exclusion is read in both directions: if the game data lists "A excludes B" only on A, B still excludes A (`Catalog.excludes`).

**Dependents.** Removing an item can strand others. If Z requires Y and Y requires X, removing X must also remove Y and Z. The **dependents** of a removed set are found by repetition: start with nothing; add every remaining pick whose prerequisites held before and fail now; repeat until nothing changes. This stops after at most as many rounds as there are picks. A pick whose prerequisite group still has another member in the selection is not a dependent.

**Exclusion removals.** To add an item that excludes some current picks (its **blockers**), tri removes the blockers and their dependents (`Catalog.exclusion_removals`).

**Proposition 2.1.** If a selection meets conditions 2 and 3, it still meets them after removing the blockers of $t$ and their dependents, and adding $t$ then creates no exclusion.

*Proof.* Removing items cannot create an exclusion, and no blocker of $t$ remains. Every remaining pick kept its prerequisites, because any pick that lost them would have been a dependent. $\square$

## 2.5 Reviews and claims

A **review** of a build is a list of operations, made by a reviewer who sees the build's highest stats, final stats, talents and mantras:

- **add** an item, weak or strong, with a reason: *necessary for archetype* (NA), *generally good* (GG, talents only), *synergises with* (SY, names the partner items $Y$) or *worth more as a luxury* (LM);
- **remove** an item, weak or strong, with a reason: *less necessary, make space* (LN), *redundant with* (RW), *works against* (WA), *no synergy* (NS), *scenario mismatch* (SM), *broken* (BR), *does not work without* (NW, names the missing components $C$) or *worth less as a luxury* (LL). RW and WA name the related items;
- **iffy**: the item is borderline; optionally a *sidegrade to* $y$ (SG), weak meaning "under this archetype" and strong meaning "generally".

A review may also carry **notes**: an **indicator** "components $C$ indicate archetype $\pi$" ([Section 6.7](06-accuracy.md)), the **weapon classes** $W_r$ the build mainly equips ([Section 5.5](05-locality.md)), and free text, which is stored and shown with every change the review causes but never learned from.

An admin can **neutralise** item $t$ in review $r$: every operation and note entry of $r$ on $t$ is dropped and $t \in N_r$, the neutral set. A neutral item carries no claim, and keeping it is no endorsement:

$$t \in N_r \;\Rightarrow\; \text{claims}_r(t) = \varnothing \,\wedge\, \text{keep}_r(t) = 0 \,\wedge\, w_r(o) = 0 \ \ \forall o \text{ on } t$$

The rest of $r$ is unchanged.

The new reasons, in symbols, for a review of build $b$ ($X$ the removed items, $t$ the item):

| shorthand | claim | holds on build $b'$ when |
|---|---|---|
| `+ t SY > Y` | add $t$ | $Y\subseteq b'$ |
| `- X NW > C` | remove each $x\in X$; equivalently add $C\setminus b$ given room | $C\not\subseteq b'$ |
| `- t LL`, `+ t LM` | remove / add $t$, relative to the luxury fill | everywhere in reach, binding only on $b$ |
| `= t SG > y` (weak) | $t\sim y$ | within reach of $b$ |
| `= t SG > y` (strong) | $t\sim y$ | every build |

Slots start full, and every pick the reviewer leaves unmarked is **endorsed**. Items that are neither picked nor added say nothing: the slots were full, so not adding something is not a judgement against it. A review must leave a valid build: it obeys the talent cap and the mantra slots (Section 2.6), adds nothing that excludes a remaining pick, and leaves no kept pick without its prerequisites. Invalid reviews are refused (`codec.validate_review`).

Each item a review speaks about becomes a **claim**: add, remove, iffy, or keep (for unmarked picks). Three kinds of claim need care:

- **Conditional claims.** "Remove X, redundant with Y" and "remove X, works against Y" are statements about builds that have Y. They only apply to builds that contain all of Y.
- **Generally good.** "Add X, generally good" says X is foundational for every build, not just this archetype. It has global scope, with limits described in [Section 5.5](05-locality.md).

Audits (verdicts on tri's own analysis) were removed in v3.5: they had no removal reasons and poisoned the reviews. Stored audits are ignored.

Code: `types.Review`, `authority.Claim`.

## 2.6 Mantra slots

A mantra occupies a **slot**, and a slot accepts only some mantra types. There are five slot types,

$$
\mathcal{S} = \{\mathrm{C}, \mathrm{Mo}, \mathrm{Su}, \mathrm{Wi}, \mathrm{X}\}
\quad (\text{Combat, Mobility, Support, Wisp, Wildcard}),
$$

and every mantra $x$ has a type $\tau(x) \in \{\mathrm{C}, \mathrm{Mo}, \mathrm{Su}, \mathrm{Wi}\}$, the `category` field of [pocamind/data](https://github.com/pocamind/data). This holds for Oath, Origin and Monster mantras too: all of them take slots. A mantra $x$ may sit in the slots

$$
A(\mathrm{C}) = \{\mathrm{C}, \mathrm{X}\},\quad
A(\mathrm{Mo}) = \{\mathrm{Mo}, \mathrm{X}\},\quad
A(\mathrm{Su}) = \{\mathrm{Su}, \mathrm{X}\},\quad
A(\mathrm{Wi}) = \{\mathrm{Wi}, \mathrm{Su}, \mathrm{X}\}.
$$

A catalog that does not state a mantra's type (and whose name is not in the bundled table) gives it type $?$ with $A(?) = \mathcal{S}$, so that missing data never forbids a pick.

**The slot formula.** The slots of a selection $P$ of a build with oath $o$ form a vector $\sigma(P) \in \mathbb{N}^{\mathcal{S}}$:

$$
\sigma(P) = \beta + g(\text{Oath: } o) + \sum_{t \in P} g(t),
\qquad
\beta = (\mathrm{C}{:}3,\ \mathrm{Mo}{:}1,\ \mathrm{Su}{:}1,\ \mathrm{Wi}{:}0,\ \mathrm{X}{:}1),
$$

where $g(t)$ is the grant of talent $t$ (zero for most talents). A Silentheart build has $\sigma = 0$: the oath makes mantras unusable. The grants are the "... Mantra Slot" stats of pocamind/data:

| source | grant |
|---|---|
| Neuroplasticity | X 1 |
| Will o' Wisp, Chorus of Souls | Wi 1 |
| Oath: Bladeharper, Dawnwalker | C 3 |
| Oath: Arcwarder, Blightsurger, Chainwarden, Contractor, Fadetrimmer, Soulbreaker, Starkindred | C 2, X 1 |
| Oath: Saintsworn | C 2, Su 2, X 1 |
| Oath: Blindseer | C 1, Su 1, X 1 |
| Oath: Linkstrider | Su 2, X 1 |
| Oath: Visionshaper | Su 1, X 2 |
| Oath: Saltchemist | Su 1, X 1 |
| Oath: Jetstriker | Mo 1, X 1 |
| Oath: Oathless | X 3 |

**Fitting.** The mantras $M \subseteq P$ **fit** when there is an injective map $\pi$ from $M$ to the slot multiset of $\sigma(P)$ with $\pi(x) \in A(\tau(x))$ for every $x$. This is a bipartite matching problem; the number of mantras that do not fit is

$$
\mathrm{over}(P) = |M| - \nu(P),
$$

where $\nu(P)$ is the size of a maximum matching. The selection fits when $\mathrm{over}(P) = 0$. The **free** slots are $\sum_k \sigma_k(P) - \nu(P)$.

**The greedy fit** (`tables.fit_mantra_slots`). Let $n_k$ be the number of mantras of type $k$ and $\sigma_k$ the slots of type $k$.

1. Each C, Mo and Su mantra takes a slot of its own type while one is left.
2. Wisp mantras take Wisp slots, then the Support slots left.
3. All mantras still unplaced (C, Mo, Su overflow and Wisp overflow) take Wildcard slots.
4. Mantras of type $?$ take any slot still empty.

**Proposition 2.2.** The greedy fit places $\nu(P)$ mantras: it is a maximum matching.

*Proof.* Take a maximum matching $\pi^*$ and transform it into the greedy one without lowering its size. (i) If a C mantra $x$ sits in X while a C slot holds a $?$ mantra $y$ or is empty, give the C slot to $x$ and the X slot to $y$ (X accepts everything); the size is unchanged. Repeating this for C, Mo, Su gives step 1. (ii) A Wisp slot accepts only Wisp and $?$ mantras, so moving a Wisp mantra from Su or X into a Wisp slot held by a $?$ mantra (which then takes the freed slot, since $A(?) = \mathcal{S}$) or an empty one keeps the size. (iii) A Support slot left after step 1 accepts Wisp and $?$ mantras; an X slot accepts a superset. So a Wisp mantra in X while a Su slot is empty or holds a $?$ can swap with it, keeping the size: step 2. (iv) After steps 1 and 2, every remaining C, Mo, Su or Wisp mantra can only go to X, and X slots taken by $?$ mantras can be swapped with them, which is step 3. (v) $?$ mantras then fill what is left, which is step 4, and any empty slot left at that point could hold no unplaced mantra. Each exchange keeps the size, so the greedy count equals $\nu(P)$. $\square$

It agrees with an augmenting-path matching on 2000 random slot vectors and type lists (a brute-force check).

**Worked example.** Pleek Arc v2 ([deepwoken.co](https://deepwoken.co/builder?id=d8zr7ica)) is an Arcwarder build with Neuroplasticity and Will o' Wisp:

$$
\sigma = \beta + (\mathrm{C}{:}2, \mathrm{X}{:}1) + (\mathrm{X}{:}1) + (\mathrm{Wi}{:}1)
= (\mathrm{C}{:}5,\ \mathrm{Mo}{:}1,\ \mathrm{Su}{:}1,\ \mathrm{Wi}{:}1,\ \mathrm{X}{:}3),
$$

11 slots, as the builder shows. A selection under these slots with 5 Combat, 2 Mobility, 1 Support and 1 Wisp mantra, the greedy fit uses C 5/5, Mo 1/1, Su 1/1, Wi 1/1 and puts the second Mobility mantra in X; 2 X slots stay free.

**Monotonicity facts the algorithms use.**

1. Adding a mantra raises $\mathrm{over}$ by 0 or 1; removing one lowers it by 0 or 1.
2. Adding a talent never raises $\mathrm{over}$; removing a talent $t$ with $g(t) \neq 0$ may raise it.
3. The talent cap and the slots interact: a counted mantra lowers the cap by 2 (Section 2.3), and a slot talent costs a talent. Both limits are recomputed after every pick; neither is a fixed count.

**How the limits are used.**

- **Completion** (`PassEngine.complete`) treats both as strict. An input over either limit first loses its least worthy picks that no rule requires and no other pick needs; a slot talent whose slots the mantras use counts as needed. It then fills mantras before talents in every round, a mantra only when it fits. If a mantra slot is still empty when the talents have reached the cap, it takes the most worthy mantra that fits and gives up the least worthy talents the completion itself added, never one whose slots the result uses. A Wisp-only slot talent is taken in the mantras-first phase only when a Wisp mantra can fill it. A slot stays empty only when no available mantra fits it under the rules. An explicit `capacity=(talents, mantras)` can only lower the limits.
- **Review** (`PassEngine.review`) trims an over-slot input the same way, displaces a mantra that leaves room of the right type when a binding add needs one, and never removes a slot talent whose slots the build's mantras need. Each alternative fits.
- **Validation** (`codec.validate_review`) refuses a review that leaves the build over its slots when the review adds a mantra or makes the excess worse, for example by removing Will o' Wisp from a build that uses its Wisp slot. A build that was already over (old data) can still be reviewed in other ways.
- **Analysis output** reports the slots, the used slots by type and the free slots (`Catalog.mantra_slot_usage`).

Code: `Catalog.mantra_slots_of`, `Catalog.mantra_slot_usage`, `Catalog.mantras_fit`, `tables.fit_mantra_slots`, `tri/data/mantra_slots.json` (the grants and the types of 250 mantras, used when a catalog's tags do not state them).

## 2.7 Capacity

The cap (Section 2.3) and the slots (Section 2.6) are not only upper limits: an analysis also uses them as the target size. Write $\mathrm{cap}(P) = 76 - 2m(P)$ and $\mathrm{free}_M(P)$ for the free slots. Before any replacement, the review step (`PassEngine._review_one`, `InferConfig.fill_capacity`) brings the build to capacity:

$$
|T(P)| > \mathrm{cap}(P) \;\Rightarrow\; \text{remove } |T(P)| - \mathrm{cap}(P) \text{ least worthy unprotected talents},
$$

$$
|T(P)| < \mathrm{cap}(P) \;\Rightarrow\; \text{add the best available talents until } |T(P)| = \mathrm{cap}(P),
\qquad
\mathrm{free}_M(P) > 0 \;\Rightarrow\; \text{fill with mantras that fit}.
$$

Mantras are filled first (unless Silentheart), since a counted mantra lowers the cap by 2. Each added mantra must fit a slot of its own type: a Combat mantra never takes the Support slot, so a build cannot swap its Support mantra for a Combat one unless a Wildcard slot is free. In a catalog that states types, a mantra of unknown type takes only a Wildcard slot. The fill reruns after every repair round, and the final cap is reported with each alternative.

Trimming is always on, because the cap is strict. Filling can be switched off with `keep_size=True` (`Tri.review`, `Engine.review`); the review comparisons in `evaluate.py`, `draft.py`, `active.py` and the scoring functions of `record.py` use it, so a reviewer's verdicts are compared with an answer of the same size.

**Worked example.** A build with 54 talents and 3 counted mantras has $\mathrm{cap} = 70$. Analysis first adds the best 16 available talents (and any mantra a free slot of the right type can hold, which lowers the cap by 2 each), then proposes replacements. A build that comes in with 76 talents under the same cap of 60 first loses 16 talents.

## 2.8 Weapon context

Some talents depend on what a build wields, which the pretrain cannot see. tri reads the build's weapons from the planner (`Build.meta["weapon"]`; a list or `"A, B"` / `"A / B"` for builds that equip several) and the bundled table `tri/data/weapons.json` (281 weapons from pocamind/data and deepwoken.co: type, weight class, hands, bleed; bleed is the deepwoken.co bleed tag). Each weapon $w$ gives tokens in the field `wield`:

$$
\omega(w) = \{\text{name},\ \text{type},\ \text{weight class},\ \text{one- or two-handed},\ \text{bleed or no bleed}\}.
$$

An item $x$ with weapon clauses $\kappa_1, \dots, \kappa_r$ (sets of tokens) is available only when

$$
\forall i\ \exists\, w \in W(B):\ \kappa_i \cap \omega(w) \neq \varnothing,
$$

where $W(B)$ is the set of equipped weapons. The quantifier over weapons is an "or": a second weapon is never ignored, and a build is locked out of an item only when none of its weapons allows it. A build that records no known weapon has no `wield` field, and no weapon clause is checked. The clauses are game data, not statistics: the 57 `weaponType` requirements of deepwoken.co, plus Turtle Shell (needs a two-handed weapon) and Speed Demon (needs a weapon without bleed).

**Redundancy.** Speed Demon on a build that wields one bleeding weapon and one that does not stays available, but its worth is scaled by the share of equipped weapons that leave it useful:

$$
q'(x) = c_x\, q(x), \qquad c_x = \frac{|\{w \in W(B): \text{no bleed} \in \omega(w)\}|}{|W(B)|}.
$$

The `# weapons:` review note is deprecated, since the build's own weapons now decide; review claims are scoped by the weapons the reviewed build records.

**Waivers.** Weapon clauses describe the planner record, and the record can be wrong about the build: a build may carry a weapon it did not select (a shield it is implied to wield), equip several, or use a talent that works without its weapon. Let $V(B)\subseteq\mathcal I$ be the items whose weapon locks are waived on $B$. Availability then reads

$$
x\ \text{available on}\ B \iff \text{(2.2) holds with the field}\ \texttt{wield}\ \text{dropped from}\ F(B)\ \text{when}\ x\in V(B),
$$

and `meta.wield_free` drops the field for every item. A review waives a lock on its own build: when a claim on $x$ is an add or a keep at similarity $\ge s_{\text{same}}$ and $x$ is blocked only by a weapon clause (`Catalog.wield_only_block`),

$$
x\in V(B). \tag{2.9a}
$$

A strong generally-good claim on $x$ waives the lock at any similarity (v3.1.3): such talents (Turtle Shell, Unstoppable Force) are used without their weapon class,

$$
\text{claim}(x) = (\text{add}\mid\text{keep},\ \text{GG},\ \text{strong}) \;\Rightarrow\; x\in V(B). \tag{2.9b}
$$

Otherwise the waiver never leaves the reviewed build, so other builds that wield the wrong weapon stay locked out. A review may add such an item; it is noted "weapon requirement waived by review".

A waiver covers the prerequisite chain: with $P^*(x)$ the prerequisites of $x$, transitively,

$$
x\in V(B) \;\Rightarrow\; P^*(x)\subseteq V(B). \tag{2.9c}
$$

## 2.8c Hardcoded weapon rules

The corpus records weapons too rarely to learn three talents that depend on them, so they are stated from the game data (v3.2, `tri/game/weapon_rules.py`). Let $W_0(B)$ be the main weapons (equipped, not offhand-only), $w_0$ the first, $\beta(w)$ the bleed tag, $\lambda(w)\in\{\text{light},\text{medium},\text{heavy}\}$ the weight class, $A$ the archetype read, and

$$
\begin{aligned}
F_{\text{no}} &= \{\text{Cursed Flames},\ \text{Pleeksty's Inferno}\},\\
L_{\text{ok}} &= \{\text{The Flippers of Fate},\ \text{Flareblood Kamas}\},\\
H_{\text{no}} &= \{\text{Purple Cloud},\ \text{Wyrmtooth},\ \text{Railblade},\ \text{Duskpole},\ \text{Sancticar},\ \text{Imperial Staff}\}.
\end{aligned}
$$

With $\text{never}(x)$: $x$ is removed and $\text{avail}(x)=0$; $\text{req}(x)$: $x$ is taken at the foundational pass whenever (2.2) holds:

$$
\begin{aligned}
&\exists w\in W_0:\ \beta(w) &&\Rightarrow\ \text{never(Speed Demon)}\\
&\forall w\in W_0:\ \neg\beta(w),\ A\notin\{\text{Trackstar},\text{Support}\} &&\Rightarrow\ \text{req(Speed Demon)}\\
&W_0\cap F_{\text{no}}\neq\varnothing &&\Rightarrow\ \text{never(Flaming Flourish)}\\
&\lambda(w_0)=\text{light},\ w_0\notin L_{\text{ok}} &&\Rightarrow\ \text{never(Turtle Shell)}\\
&\lambda(w_0)=\text{light},\ w_0\in L_{\text{ok}} &&\Rightarrow\ \text{Turtle Shell}\in V(B)\\
&\lambda(w_0)\in\{\text{medium},\text{heavy}\},\ w_0\notin H_{\text{no}} &&\Rightarrow\ \text{req(Turtle Shell)},\ \text{Turtle Shell}\in V(B)
\end{aligned}
$$

On Trackstar and Support, worth decides Speed Demon. Names match exactly (Prototype Railblade is not Railblade). $W_0=\varnothing$ gives no rule; the weapon question asks for one (Section 14.2).

**Class answers (v3.6).** When $W_0=\varnothing$ but the weapon answer is a class $k$ with weapons $W_k$, a rule holds when it holds for every $w\in W_k$:

$$
\begin{aligned}
&\forall w\in W_k:\ \lambda(w)=\text{light} &&\Rightarrow\ \text{never(Turtle Shell)}\\
&\forall w\in W_k:\ \lambda(w)\in\{\text{medium},\text{heavy}\} &&\Rightarrow\ \text{req(Turtle Shell)},\ \text{Turtle Shell}\in V(B)\\
&\forall w\in W_k:\ \beta(w) &&\Rightarrow\ \text{never(Speed Demon)}\\
&\forall w\in W_k:\ \neg\beta(w),\ A\notin\{\text{Trackstar},\text{Support}\} &&\Rightarrow\ \text{req(Speed Demon)}
\end{aligned}
$$

So Fist forbids Turtle Shell and Greatsword requires it. Before v3.6 a class answer set only the focus (2.10), and a strong generally-good Turtle Shell, whose weapon requirement is waived, went onto fist builds.

**Precedence (v3.4).** Let $Q(B)$ be the required items whose stats are met, closed under hard prerequisites:

$$
G(B) = \{\,y : y\in P^*_{\text{hard}}(x)\cup\{x\},\ \text{req}(x),\ \text{avail}_B(x)=1\,\}.
$$

For $y\in G(B)$: review removals and absences of $y$ are dropped on $B$; no never rule (2.4), or review rule (7) blocks $y$; $y$ is protected from worth swaps and space removals. $G(B)$ is added in prerequisite order (2.8d), as one unit: Knight's Rally enters only if Turtle Shell enters with it. When a build is held to its size and nothing is displaceable, $y$ is added anyway if the cap and slots allow; otherwise it displaces the least worthy pick that is not in $G(B)$, required, central, a prerequisite of another pick or set by a category. Pins and reviews do not protect against it. Only game data keeps $y$ out:

$$
y\notin \text{result} \;\Rightarrow\; \text{avail}_B(y)=0\ \lor\ \exists z\in S:\ y\perp z\ \lor\ \text{no pick can make room under } 76-2m.
$$

A never item is removed even when pinned, and a claim that adds or keeps it is dropped on $B$.

## 2.8d Addition order

A pick $x$ enters a selection $S$ only with its missing prerequisites, which enter first, in prerequisite order:

$$
\text{add}(x, S) = S \cup M(x, S)\cup\{x\},\qquad M(x,S) = P^*(x)\setminus S,
$$

and only when every $y\in M(x,S)\cup\{x\}$ is available, unblocked, and the result meets the counts, the cap and the slots; for a group in `prerequisites_any`, the member with the highest $q$ that can be taken. For a prerequisite $p$ with dependants $D(p)$, let the solo share be

$$
\sigma(p) = \frac{\sum_{i:\,p\in B_i,\ D(p)\cap B_i=\varnothing} w_i}{\sum_{i:\,p\in B_i} w_i}
$$

over the corpus. When $\sigma(p) < \sigma_{\min} = 0.3$ (`prereq_solo_min`), $p$ is taken by worth only as $M(d, S)$ for some $d\in D(p)$ with $q(d)\ge 0.2$ (`package_min_q`). A completion that still ends with such a $p$ and no $d\in D(p)$ (a dependant later removed) is computed again with $p$ unavailable, up to 3 times.

A worth swap never removes the last dependant of a picked prerequisite:

$$
x \in D(p),\quad p \in S,\quad D(p)\cap (S\setminus\{x\}) = \varnothing \;\Rightarrow\; x \text{ is not displaceable by worth.}
$$

So Unfazed stays with Grasp on Reality when both are met at the same stage (for example the pre-shrine stage).

## 2.8e Completion as a reviewed build

A completion $C(b)$ is checked as a full build before it is returned (the refine step: generate, check hard constraints, refine the complete set). With $R$ the review step (Section 7) on a full build, input picks pinned,

$$
C_0 = C(b),\qquad C_{k+1} = R(C_k),\qquad k < 2\ (\texttt{review\_pass\_rounds}),
$$

stopping when $R$ changes nothing or a selection repeats. An add of $C_k$ that $R$ removes never appears, and a removal $R$ takes back never appears. The weapon rules (2.8c), required picks and the addition order hold at every step.

**Dependencies (v3.6).** A supported DEPENDS review rule $x \Rightarrow C$ (from NW removals, 5.5 item 7) gates every worth pick, in completion and in $R$:

$$\text{worth-take}(x \mid S) \;\Rightarrow\; C \subseteq S.$$

$C$ is not pulled in for $x$; it enters on its own worth, and $x$ may follow. So Grasp on Reality never comes without Unfazed, nor Laced Traps without Caltrops. Before v3.6 the engine kept only BROKEN, REDUNDANT and CONFLICT review rules, so DEPENDS and OFF_ARCHETYPE rules were induced but never applied.

## 2.8f What is hardcoded

Most of what tri does is learned from the corpus and the reviews. The rest is fixed in code or game data, and no review, rule or drop changes it:

| What | Where | Effect |
|---|---|---|
| stat requirements, pre-shrine and final stage only | game data, 2.2 | $\text{avail}_B(x)$ |
| prerequisites, `exclusive_with` | game data, 2.4 | $x\Rightarrow P(x)$, $x\perp y$ |
| Vow of Iron only picks | game data | Pathfinder $\Rightarrow\neg x$ |
| talent cap $76-2m$, mantra slots | game data, 2.3, 2.6 | counts |
| Speed Demon, Flaming Flourish, Turtle Shell | `tri/game/weapon_rules.py`, 2.8c | never / req, over reviews |
| class focus | 2.10 | other classes' talents off |
| addition order | 2.8d | prerequisites first |

Everything else that bears on a pick, apart from worth, is listed by `Tri.all_rules` (2.11) and can be switched off by id.

## 2.9 Stat benchmarks

The worth of some items rises sharply once a stat passes a benchmark, which a smooth field flattens. Guardbreaking mantras past 77 in their attunement are the case in point. For a benchmark $(x, s, k)$ with gain $\kappa$ and width $\omega$ (`tri/data/benchmarks.json`: $\kappa = 2.5$, $\omega = 2$, $k = 77$ for the ten guardbreakers of deepwoken.co), the logit of $x$ is shifted by

$$
\Delta_x(B) = \kappa\, \sigma\!\left(\frac{\max(\mathrm{pre}_s(B), \mathrm{fin}_s(B)) - k}{\omega}\right),
$$

with $\sigma$ the logistic function. Only the pre-shrine and final stages count, as for requirements (Section 2.1). The shift is applied wherever worth is read: the analysis (`priors.adjust_q`), the concept point (`space.point`) and the stat curves of `/plot stats`, which mark the benchmark. Between 75 and 79 the shift moves from $0.27\kappa$ to $0.73\kappa$, so the curve is steep at the benchmark and flat elsewhere.

## 2.10 Weapon-class focus

A build usually commits to one weapon class. Let $\mathcal C$ be the weapon types of the table that are classes (Shield and Exclusive are not), and let $C(x)\subseteq\mathcal C$ be the classes item $x$ needs: the game data's `weaponType` condition and the weapon clauses of Section 2.8 ("Fists" is read as Fist, "Pistol, Rifle, Greatcannon" as three classes). $C(x)=\varnothing$ for an item of no class. Worth (3.1) is a pick rate among the builds that can take an item, and only builds that wield a weapon can take its talents, so for a build that records no weapon every class's talents look worthy. Filling by worth then takes Fist, Rapier and Pistol talents together.

**The class.** For a build $B$ at concept point $u$, the allowed classes are

$$
A(B) = \begin{cases}
\bigcup_{w\in W(B)} C(w) & \text{if } W(B)\text{ names a class (equipped weapons)},\\
\bigcup_{x\in S(B)} C(x) & \text{else, if nonempty (the build's own weapon talents)},\\
\bigcup_{x\in R^+(B)} C(x) & \text{else, if nonempty (binding review adds and keeps)},\\
\{\arg\max_{c\in\mathcal C_a} \hat P(c\mid u)\} & \text{otherwise},
\end{cases} \tag{2.10}
$$

where $\mathcal C_a$ are the classes with an available talent. Evidence may name several classes (a build with two weapons); a prediction names one.

**The class prior.** Each corpus build $n$ is labelled with the classes of the weapons its lock signature records and of the weapon talents it picked, $L_n\subseteq\mathcal C$. With $N_k(u)$ the $k$ nearest labelled corpus builds to $u$ in the concept map and $w_n$ their corpus weights,

$$
\hat P(c\mid u) = \frac{\sum_{n\in N_k(u)} w_n\,[c\in L_n]}{\sum_{n\in N_k(u)} w_n}, \qquad k = 150. \tag{2.10a}
$$

When fewer than 20 labelled builds exist, $\hat P(c\mid u)$ is replaced by the mean of the top three $q(x)$ over the available talents of class $c$. The prior is computed from the stored corpus and its coordinates on load or fit; nothing is refitted. A corpus whose builds record weapons (Section 2.8) labels more builds, so a refit on such a corpus sharpens it.

**Enforcement.** Talents of other classes are unavailable for the rest of the answer:

$$
x \text{ blocked} \iff C(x)\neq\varnothing,\ C(x)\cap A(B)=\varnothing,\ x\notin S(B)\cup R^+(B).
$$

The build's own picks and binding review items are never blocked, so a review can still add another class. In a completion the class is filled before the luxuries: after the required passes, the talents of $A(B)$ with $q(x)\ge q_{\text{focus}} = 0.3$ are taken by worth, then the luxury categories and the rest by worth.

## 2.11 The rule registry

Every influence on a pick other than worth $q$ is one entry $(\iota, \text{source}, \rho, \text{effect}, \text{status}, \delta, \phi)$: an id, its source, the rule in symbols, take / never / blocks / yields / shapes, its status, whether it is droppable ($\delta$) and, for a build $B$, whether it fires ($\phi$). Sources:

| Source | Id | $\delta$ |
|---|---|---|
| selection rules: mined, authored, review evidence (2.4) | 6 hex | yes |
| review rules (7.3) | `rv:KIND:a[:b]` | yes |
| sidegrades (4.9) | `sg:` 6 hex | yes |
| categories | `cat:item` | no: set the category |
| weapon rules (2.8c) | `wp:item:take\|never` | no |
| game data (2.8f) | `gd:pre:`, `gd:excl:`, `gd:voi:`, `gd:stat:`, `gd:cap` | no |
| class focus (2.10) | `fx:focus` | no |
| binding review claims here | `cl:item` | no: edit or purge the review |

A selection rule is listed for $x$ when $x$ is its item or an item in its conditions, so a rule $y\in S\Rightarrow\neg z$ appears under $y$ too. Review rules of kinds other than BROKEN, REDUNDANT and CONFLICT are listed as inactive: they act only through the review claims they come from. Dropping writes a suppression with the same mechanism as for mined rules; dropping again lifts it.
