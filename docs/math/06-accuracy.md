# 6. Accuracy

## 6.1 Definition

> **Accuracy.** Every statement tri makes, whether a fitted worth, a certified rule, an archetype reading or a chosen setting, has been checked on data that played no part in producing it, and is kept only if it passes that check by a stated margin. Reviews are treated as fully accurate, so on reviewed builds accuracy means agreeing with the reviewers.

## 6.2 Interpretation

The danger in any pattern-mining system is **overfitting**: finding rules in noise, or tuning a setting until it fits the data you tuned it on. tri's defence is a single principle, familiar from any data science course: **the data that proposes must not be the data that judges.** Every component applies it in its own form:

| component | proposes on | judges on |
|---|---|---|
| worth field | 95% of corpus builds | 5% held out, scored on hidden entries |
| a build's own picks | the rest of the build | leave-one-out worth ([Section 3.4](03-worth.md)) |
| certified rules | search half of the builds | certification half |
| archetype geometry | the corpus (no labels) | locators, which never train it |
| metric power | locators | leave-one-out readings, with a two-standard-error margin |

The second idea is **fitting as little as possible to labels**. Locators choose one scalar, nothing else. Reviews do not tune parameters unless you explicitly refit.

## 6.3 Certified rules

A **rule** says "builds of this kind take item X" (a take rule) or "builds of this kind never take X" (a never rule). "Of this kind" is a condition made of at most three **atoms** (v3.0; five before):

| atom | form | example |
|---|---|---|
| stat | a highest stat at least, or below, a multiple of 5, of a **grounded** stat (Section 6.12) | Agility at least 50 |
| context | an oath, murmur, origin or weapon-class token present or absent | oath is Silentheart; weapon = Heavy |
| archetype | the build's archetype family as the locators read it, $A=a$ (Section 6.12) | $A=\text{chime}$ |
| group | at least, or fewer than, $n$ picks of a known group (Section 6.12) | $\lvert G(\text{Flamecharm})\rvert\ge 2$ |
| anchor | a **related** item is picked: a prerequisite, a dependant or a group mate | with Flame Leap |
| aggregate | a stat total at least, or below, a value | attunement points below 30; stats at 75+ at least 2 |

Atoms never mention a fitted region or coordinate. The archetype atom names an administrator's archetype, read by the locators ([Section 4](04-concept.md)), so its meaning is the administrators' label, not a fitted cluster; a refit can move which builds it reads, which `Tri.mine_rules` re-checks. Mined rules never use "without item Y": an unfinished build lacks most picks, so absence is not evidence. Authored rules may (Section 6.13). A rule applies to a build if the item is available to it and the condition holds.

For a rule, the **cell** is the builds it applies to, its **support** is their total weight, and its **rate** is the weighted share that picked the item.

**Wilson bounds.** A rate measured on $n$ builds is uncertain. tri uses the Wilson score interval with $z=1.645$ (one-sided 95%):

$$
\text{LB},\ \text{UB}=\frac{\hat p+\frac{z^2}{2n}\mp z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}. \tag{6.1}
$$

Here $n$ is the effective sample size of the cell, $(\sum w)^2/\sum w^2$, which equals the count when all weights are 1.

**Certification.** For each item, shuffle the builds that can take it and split them into a **search half** and a **certification half** (v3.0: the certification half is split again into a certification quarter $B$ and a replication quarter $C$; Section 6.12). Rules are proposed and screened on the search half only. A rule is then kept only if, on the certification half:

- take rule: lower bound at least 0.80 and support at least 30;
- conditional take rule of an item whose base rate is below 0.80: also a rate at least $1.1$ times the base rate (`min_lift`), so the condition adds something;
- never rule: upper bound at most 0.05, support at least 30, and **contrast**. Changing one atom of the condition must give a cell where the item is taken at a rate of at least $\max(0.10,\ 3\,\text{UB})$ (`contrast_ratio`). This keeps "never" for real refusals (wanted next door, not here), not for items that are rare everywhere.

**Many candidates.** One item can send $m_t$ candidates to the certification half. Each is tested at the same level, so the chance that at least one false rule passes grows with $m_t$. tri applies the Bonferroni correction within each item:

$$
z_t=\Phi^{-1}\!\big(1-\alpha/m_t\big),\qquad \alpha=1-\Phi(z). \tag{6.1a}
$$

By the union bound, the chance that any false rule of item $t$ is certified is then at most $\alpha$. For $m_t=10$, $z_t\approx 2.58$ instead of 1.645.

**Worked example.**

| observed | Wilson bound | verdict |
|---|---|---|
| 27 of 30 take it (90%) | LB 0.774 | not certified: 30 builds cannot rule out 80% |
| 90 of 100 take it (90%) | LB 0.840 | certified take rule |
| 57 of 60 (95%) | LB 0.881 | certified |
| 0 of 30 take it | UB 0.083 | not certified never rule: 30 builds cannot rule out 5% |
| 0 of 60 take it | UB 0.043 | certified, if the contrast holds |

**Why the split matters.** Because the condition is chosen on the search half, the certification half is a fresh sample for a fixed rule, and the bound is an honest one-sided 95% bound for that rule. Without the split, 87% of rules held on fresh builds in the rich synthetic world; with it, 99.7%. The statement is per rule: among hundreds of rules a few percent can fail by chance (multiple testing). `Rulebook.certification_check` and `evaluate.certification` measure the realised share; raising $z$ lowers it.

**The mining procedure**, per item, on the builds that can take it:

1. **Propose.** Grow two shallow decision trees (depth at most 3) on the search half: one over grounded stats, aggregates, context, archetype, group counts and the related items most informative about the item (at most 32), one without items. Aggregates are fixed functions of stats: attunement points, weapon points, core points, the top attunement and the number of stats at 75 or more. They express "the budget is spent elsewhere" in one atom, where single stats need a family of proxies. Each path is a candidate condition. Stats are floored to multiples of 5 first, which loses nothing because game thresholds are multiples of 5 ([Section 2.2](02-constraints.md)) and cuts each stat to about 20 cut points. Trees only propose.
2. **Screen** candidates against the certification thresholds on the search half.
3. **Uniformity.** Group the cell's builds by pick-profile cluster, assigned without the item's own column (otherwise the item would help define the clusters that judge it). A cluster $g$ **vetoes** the rule if all three hold:
   - its weight is at least $\max(5,\ 0.02\,S_w)$, where $S_w$ is the cell weight (`uniform_support`, `uniform_share`);
   - its own Wilson bound is on the wrong side of the threshold;
   - its rate differs from the rest of the cell by a two-proportion test,
   $$
   z_g=\frac{\hat p_{\text{rest}}-\hat p_g}{\sqrt{\bar p(1-\bar p)\,(1/n_g+1/n_{\text{rest}})}}\ \ge z .
   $$
   A real minority archetype hidden in a large cell still vetoes, but a handful of builds cannot.

   **Repair.** A vetoed take rule gets up to two more atoms (`repair_steps`). Each is the depth-1 split of the cell's search-half rows that best separates $g$, and the rule is then screened again. A rejection becomes "take X except where Y". An item left with no take rule because of a veto is recorded in `Rulebook.vetoed`.
4. **Shorten.** Drop atoms while the rule still passes on the search half.
5. **Certify** on the certification half.
6. **Tidy.** Remove duplicates and rules implied by a more general rule of the same item. Drop a take rule whose covered builds overlap an already kept take rule's with weighted Jaccard of at least 0.8 (`max_overlap`): proxies of one concept cover the same builds. Keep at most 6 take and 3 never rules per item.
7. **Conflicts.** Where a take and a never rule of one item overlap heavily, the observed rate in the overlap removes the rule it contradicts.

**Passes.** Each item is assigned the strongest pass its rules justify:

| pass | condition | meaning |
|---|---|---|
| foundational | a take rule with stat or group atoms only (or none; group atoms from v3.0.1), and breadth at least 0.5 | everyone takes it, at most gated by stats |
| central | any other take rule | builders of this kind (oath, anchors) take it |
| exclusion | a never rule | builders of this kind refuse it |
| luxury | no take rule | taste: decided by worth $q$ |

**Breadth** of a rule is the overlap between the cluster mix of its cell and the cluster mix of the corpus, from 0 (one cluster) to 1 (a cross-section). Clusters are formed from picks only, so a stat convention does not look narrow. A foundational rule must be broad.

Code: `tri/rules/passes.py` (`Rulebook`), `tri/rules/defs.py` (`PassConfig`).

## 6.4 The held-out check of the field

5% of corpus builds (at least 50, at most 2000) are removed from training. For each, a random 20% of its available entries are hidden; its archetype is solved from the other 80% exactly as a new build's would be, and the hidden entries are scored by cross-entropy. This **held-out deviance** is an unbiased estimate of how well the field predicts a new build, because the held-out builds and the hidden entries are independent of the fitted parameters. The same score on training builds gives the **gap**. Both are also reported as deviance explained (`explained` for held-out builds, `probe_explained` for training builds under the same masked protocol), so the two numbers that should be compared appear side by side. The training fit's own deviance explained scores a different task (every entry, nothing hidden) and is not comparable with either.

**A check, not a selector.** The held-out deviance chooses nothing. This is deliberate: in the playstyle world it was lowest after 50 training steps (0.1544) or with `l2_items` 0.1 (0.1526), but those fields erased the playstyle signal (10-nearest-neighbour playstyle purity 0.80 and 0.49, against 0.82 at the defaults). Deviance is dominated by entries that separate no one, so minimising it smooths away weak archetype signals. What an archetype is, is what the administrators say it is, so geometry is judged by locators (Section 6.5). The held-out check stays as a guard: a growing gap flags a field that fits its training builds rather than builds in general.

Code: `ConceptSpace._val_split`, `_val_score`, `space.validation`.

## 6.5 Locator leave-one-out and the metric power

Each locator is read from all the others (leave-one-out), counting only locators whose archetype has at least one other locator. tri reports the accuracy, the mean log-probability of the correct archetype against chance, the AUC of distance for same- against different-archetype pairs, and **conflicts**: locators within reach of another that read confidently as a different archetype. Since the coordinates do not depend on the labels and locator builds were never trained on, every one of these is an out-of-sample measure of how well the space separates the administrators' archetypes.

**Choosing the metric power.** The only setting locators choose is $\gamma$ in (4.4), from $\{0,\tfrac14,\tfrac12,\tfrac34,1\}$. For each candidate, compute every locator's leave-one-out log-probability and its paired difference from the current power. A candidate replaces the current power only if its mean gain is

- more than **two standard errors** above zero (the gain is reliable), and
- at least **0.01 nats per locator**, about one percentage point more probability on the right archetype (the gain is worth having).

The largest such gain wins; otherwise the power stays. Selection runs at refit only, with at least 3 locators for each of at least 2 archetypes, and needs no retraining because the archetypes do not depend on the weights.

**How often can this go wrong?** If no candidate is truly better, each has about a 2.3% chance of passing the two-standard-error test by luck. With four alternatives, the chance of a false move is at most about $4\times2.3\%\approx9\%$. With the usual one-standard-error rule it would be 63%, which is why the stricter rule is used. The margin of 0.01 nats matters when locators are already read almost perfectly: then the standard error is tiny and any sharpening looks significant, but sharpening stretches each archetype and shrinks the share a review reaches. In the simple world, the locators favoured $\gamma=1$ by 0.0024 nats, many standard errors, and that move would have cut a review's reach within its archetype from 93% to 73%.

Whatever is chosen, Theorems 4.1 and 4.2 still hold, because they hold for any positive weights.

Code: `LocatorSet.loo`, `LocatorSet.select_power`, `Tri.select_metric`.

## 6.6 The twin split of overlapping archetypes

**The problem.** The kernel reading of [Section 4.7](04-concept.md) sees only coordinates. Two archetypes whose builds differ by a few picks, for example gank and chime builds (a few talents and mantras apart), sit almost on top of each other in the concept space: a handful of items moves a coordinate by much less than the bandwidth $h$. Their kernel densities are then nearly equal, the reading splits between them, and the archetype with fewer typical builds is almost never named. The fix must not touch the field (locator labels never enter a loss), must not change any reading that does not involve such a pair, and must not act unless it is shown to help.

**Why v2.15 was too loose.** v2.15 compared every archetype with at least $2\%$ kernel probability against the corpus background, with a likelihood clip of $8$ nats. With five archetypes almost every build has $2\%$ on several of them, and each archetype's common items (Carnivore and Grand Feast for boss PVE builds) differ from the background far more than twins differ from each other. So the picks re-split the reading of most builds, not only twins: on synthetic worlds $36\%$ to $67\%$ of builds outside any twin pair moved, leaning builds became pure, and with two locators per twin the twins themselves were read worse. The map's colours stopped following its geometry.

**Twins.** Let $P^{\text{loo}}_{j\cdot}$ be the kernel reading of locator $j$ from the others. Archetypes $\pi,\rho$, each with at least $2$ locators (`refine_min_locators`), are **twins** when the map confuses them:

$$
o_{\pi\rho}=\max\Bigl(\overline{P^{\text{loo}}_{j\rho}}_{\,j\in L_\pi},\ \overline{P^{\text{loo}}_{j\pi}}_{\,j\in L_\rho}\Bigr)\ \ge\ \tau,\qquad \tau=0.2\ (\texttt{twin\_overlap}). \tag{6.2}
$$

Pairs the map already separates are never split.

**Tells.** With $k_{\pi t}$ the number of $\pi$'s locators that took item $t$ and $n_{\pi t}$ the number that could, let $c_t=(k_{\pi t}+k_{\rho t}+2b_t)/(n_{\pi t}+n_{\rho t}+2)$ be the pooled rate of the pair. The shrunken rates and the **tells** are

$$
r_{\pi t}=\frac{k_{\pi t}+\kappa\,c_t}{n_{\pi t}+\kappa},\qquad T_{\pi\rho}=\Bigl\{t:\ \bigl|\operatorname{logit}r_{\pi t}-\operatorname{logit}r_{\rho t}\bigr|\ge\theta\Bigr\},\qquad\kappa=4,\ \theta=\log4.5. \tag{6.3}
$$

Shrinking toward the pair, not the corpus, means the shared core of the twins and everything common to the corpus cancel; only the few picks that separate the twins remain. For a build with picks $x_t$ and availability $a_t$,

$$
\Lambda_{\pi\rho}(x)=\sum_{t\in T_{\pi\rho}}a_t\Bigl[x_t\log\frac{r_{\pi t}}{r_{\rho t}}+(1-x_t)\log\frac{1-r_{\pi t}}{1-r_{\rho t}}\Bigr]. \tag{6.4}
$$

**The split.** Let $m=p_\pi+p_\rho$ be the pair's kernel mass and $s=p_\pi/m$. Where $m\ge\mu$ ($\mu=0.5$, `twin_mass`) and the build is within reach,

$$
\tilde p_\pi=m\,\sigma\bigl(\operatorname{logit}s+\operatorname{clip}(\Lambda_{\pi\rho},\pm c)\bigr),\qquad \tilde p_\rho=m-\tilde p_\pi,\qquad c=4\ (\texttt{twin\_clip}), \tag{6.5}
$$

and $\tilde p=p$ everywhere else. This is Bayes' rule inside the pair, with the kernel share as the prior and the tells as a naive likelihood.

**The held-out gate.** A pair is split only if, reading each of its locators from the others (kernel reading leave-one-out, tells rebuilt without it), the split corrects at least $g=1$ (`twin_min_gain`) more locators than it breaks. Otherwise it is off and the kernel reading stands. This is the strictness principle of [Section 7](07-strictness.md) applied to the reading: no correction without held-out evidence.

**Properties.**

- **Locality.** Only $\tilde p_\pi$ and $\tilde p_\rho$ of a build in the pair's shared region (mass at least $\mu$, within reach) can change. Every other archetype, every build elsewhere, and every build when no pair passes the gate keeps the kernel reading exactly, so the atlas colours follow its geometry.
- **Accuracy.** The likelihood uses only the pair's tells, shrunk by $\kappa$, and is clipped at $e^{\pm4}$; the gate requires a held-out improvement.
- **No refit.** (6.2) to (6.5) use locator picks and coordinates only, and are rebuilt when a locator is added or removed or the metric changes.

**Where it is used.** The split names the build, the first alternative of an analysis, the atlas, and replaces the kernel reading in the barrier (5.2). Worth is corrected on the tells only: with $s,\tilde s$ the pair shares before and after,

$$
\operatorname{logit}\tilde q_t=\operatorname{logit}q_t+m\Bigl[\operatorname{logit}\bigl(\tilde s\,r_{\pi t}+(1-\tilde s)\,r_{\rho t}\bigr)-\operatorname{logit}\bigl(s\,r_{\pi t}+(1-s)\,r_{\rho t}\bigr)\Bigr],\qquad t\in T_{\pi\rho}, \tag{6.6}
$$

and $\tilde q_t=q_t$ elsewhere.

**Forcing an archetype.** A caller may ask for the analysis or completion of a build as archetype $\pi^\ast$. The build's latent is replaced by the mean latent of the locators of $\pi^\ast$ (a region named without locators uses that region's latent), and the reading is the one-hot $\tilde p=e_{\pi^\ast}$ in (6.6) and in the barrier. Every hard constraint of [Section 7](07-strictness.md) is unchanged.

**Tells on the card.** For an active pair the tells, strongest first, are shown as `Gank ↔ Chime: +L0_6 −C0_5`; archetypes without an active twin show none.

**Synthetic check.** Twin world (`make_twin_world`: half the builds of one archetype become a twin that adds three talents and drops two), five archetypes, held-out builds, three seeds:

| locators per archetype | reading | twin accuracy | other accuracy | builds outside the pair whose reading moved |
|---|---|---|---|---|
| 4 | kernel | 0.64 to 0.76 | 1.0 | |
| 4 | v2.15 fine reading | 0.98 to 1.0 | 1.0 | 49% to 60% |
| 4 | twin split | 0.98 to 1.0 | 1.0 | 0% |
| 2 | kernel | 0.51 to 0.68 | 1.0 | |
| 2 | v2.15 fine reading | 0.50 to 0.68 (0.59 → 0.50 in one seed) | 1.0 | 53% to 67% |
| 2 | twin split | 0.51 to 0.68 (gate off) | 1.0 | 0% |

On worlds without twins (`make_rich_world`) the twin split finds no pair and is the kernel reading exactly, where v2.15 moved $36\%$ to $63\%$ of readings and lowered accuracy from 1.0 to 0.99 in one seed.

Code: `LocatorSet.set_background`, `_find_pairs`, `_contrast`, `_split`, `refine`, `note`, `fine_rows`, `contrast`, `adjust`, `distinct`, `pair_of`, `loo_fine`; `Tri.force`.

## 6.7 Indicator notes

**The problem.** A reviewer often knows that a few picks give a build away ("Chronostasis and Knight's Rally make this a gank build") where the kernel reading sees only coordinates. Such a note must sharpen the reading near the reviewed build and nowhere else, however strongly reviews are weighted.

**The lift.** An indicator $\iota=(\pi,C,u_r)$ names an archetype $\pi$, components $C$ (catalog items, or `oath:` and `murmur:` tokens) and the reviewed build's coordinates $u_r$. It applies to a build with picks and context $x$ at $u$ when

$$
C\subseteq x,\qquad \lVert u-u_r\rVert\le R,\qquad d_{\min}(u)\le R,\qquad p_\pi(u)\ge\varphi, \tag{6.7}
$$

with $R$ the locator reach and $\varphi=0.05$ (`indicator_floor`). Then the odds of $\pi$ are multiplied by $e^{\lambda}$:

$$
\tilde p_\pi=\frac{e^{\lambda}p_\pi}{e^{\lambda}p_\pi+1-p_\pi},\qquad \tilde p_\rho=\frac{p_\rho}{e^{\lambda}p_\pi+1-p_\pi}\ (\rho\neq\pi),\qquad \lambda=\texttt{indicator\_lift}=6. \tag{6.8}
$$

Several indicators for the same $\pi$ lift once; the newest note per component set wins, and a note leaves with its review. The lift is applied after the twin split, so it reaches the same places: the build's name, the first alternative and the barrier.

**Proposition 6.1 (indicators are local and bounded).** If any condition of (6.7) fails, $\tilde p=p$ exactly. Otherwise $\tilde p$ preserves the ratios among $\rho\neq\pi$ and the log-odds of $\pi$ rise by exactly $\lambda$ per archetype.

*Reason.* (6.8) multiplies $p_\pi$ by $e^\lambda$ and renormalises; renormalising scales every other entry by the same factor, and $\operatorname{logit}\tilde p_\pi-\operatorname{logit}p_\pi=\log e^\lambda$. $\square$

So a note cannot name an archetype the kernel gives less than $5\%$, cannot act outside the reviewed build's neighbourhood, and cannot move builds that lack a component, however many reviews repeat it. It changes no coordinate and no worth row, and needs no refit.

Since v2.18 the lift has its own cap, 6 nats (odds $\times e^6\approx400$), no longer the twin clip: a reviewer's note is the most direct evidence the model has, and Proposition 6.1 keeps it local however large $\lambda$ is.

**Anchors.** An `is` note ("this build is $\pi$") makes the reviewed build an anchor $(\pi,u_r,\pm)$; a `not` note a negative one. With $K$ as in (4.6) and $n_a$ the anchors of $\pi$, the kernel mass of $\pi$ at $u$ becomes

$$
k_\pi(u)\ \leftarrow\ k_\pi(u)+\frac{w_a}{n_a}\sum_{a^+}K(u,u_a),\qquad k_\pi(u)\ \leftarrow\ k_\pi(u)\prod_{a^-}\big(1-v\,K(u,u_a)\big), \tag{6.8a}
$$

with $w_a=4$ (`anchor_weight`) and $v=0.9$ (`anchor_veto`). A positive anchor also counts in $d_{\min}$, so an `is` note can place a build near it. Both terms vanish beyond $R$, so anchors are local in the sense of Proposition 6.1.

Code: `LocatorSet.set_indicators`, `_indicate`, `indicated`, `set_anchors`, `_anchor_logk`; `Tri._sync_indicators`.

## 6.8 Accuracy against reviews

Reviews are ground truth, so on reviewed builds tri's analysis is scored against the reviewer's operations: precision and recall of adds and removes, accuracy of reasons, agreement with strong operations, and the share of disagreements tri itself reported as violations. These are `evaluate.review_benchmark` and `evaluate.strictness_benchmark`. Audits give a direct per-change accuracy: the share of tri's changes that reviewers upheld.

## 6.9 What is fitted and what is chosen

| fitted from data | chosen by hand (Appendix A) |
|---|---|
| item coefficients, archetypes, weights, regions | dimension 16, stat grid, L2 strengths |
| certified rules | thresholds 0.80, 0.05, 0.10, $z$, supports, breadth |
| metric power (by locators only) | bandwidths, reach, similarity bands, binding threshold |
| | review weights, refit share, patch factor, held-out share |

Hand choices are defaults with stated reasons. None is tuned on the data it is scored on.

## 6.10 The held-out sidegrade test

A split found on one half of a region is kept only if Definition 4.3 holds on the other half: commitment, balance and the substitution correlations are recomputed there, and items that fail separation there are dropped (if a side becomes empty, the split is rejected). Since the half used to propose the split is independent of the half used to test it, a split produced by noise in the first half passes in the second only by chance. With four restarts and at most two sidegrades per region, the number of candidate tests stays small.

**Synthetic check.** In a world where one archetype takes $A=\{SA_1,SA_2,MSA\}$ or $B=\{SB_1,SB_2,MSB\}$ with commitment 0.92, mining found exactly that sidegrade (shares 0.48 and 0.41, cross correlation $-0.84$), offered it on every committed held-out build of that archetype, and found no sidegrade in the standard synthetic world, which has none.

Code: `sidegrades._one`.

## 6.11 Families, variants and styles

**The problem.** An archetype such as Gank holds builds that play very differently (frost mage, heavy bruiser). One label per archetype hides this; one archetype per playstyle would starve each of locators.

**Names.** A locator is named "family / variant" (e.g. `Gank / Frost Mage`) or by family alone. The located posterior ([Section 4.7](04-concept.md)) is taken over families, so variants never split a family's evidence. Within the read family $F$, the variant posterior is the kernel posterior over $F$'s variant labels $v$, with an equal prior per variant:

$$
p(v\mid u,F)=\frac{\frac1{n_v}\sum_{\ell\in v}K(u,u_\ell)}{\sum_{v'}\frac1{n_{v'}}\sum_{\ell\in v'}K(u,u_\ell)}. \tag{6.9}
$$

A variant is reported when the nearest point of $F$ is within $R$ and $\max_v p\ge0.5$; `is` anchors that name a variant count as its points.

**Style.** Independently, a deterministic **style** is read from the final stats: the dominant attunements, the weapon class (or the classes a weapons note names), and a role, giving a label such as "Frost mage". The style is descriptive only; it changes no reading and no worth.

**Locality.** (6.9) uses only $F$'s locators within reach, so naming a variant cannot move any build of another family.

Code: `tri/reviews/variants.py` (`split_archetype`, `family_key`, `read_variant`, `style_of`), `Tri._add_variant`.

## 6.12 Selection rules

From v3.0 tri separates two kinds of decision. Foundational and central talents, and every exclusion, are decided by **selection rules**: certified statements about the context of a build. Only luxury talents, which are volatile, are ranked by worth (Section 3). Mantras keep worth, since a mantra's value depends on the whole kit.

**Context.** The context of a build $b$ is
$$
\kappa(b)=\big(\mathbf h_b,\;P_b,\;T_b,\;A_b\big),
$$
with $\mathbf h_b$ the highest stats, $P_b$ the picks, $T_b$ the tokens (oath, murmur, origin, weapon class $W_b\in\{\text{Heavy},\text{Medium},\text{Light}\}$) and $A_b$ the archetype family the locators read at $b$'s place in the concept space ([Section 4](04-concept.md)): the nearest locator family $a$ with distance at most the reach and confidence at least `min_confidence`, else none. The six families are gank, chime, trackstar, support, boss PvE and wave PvE.

**Groups.** Groups come from game data only. For a category $c$ (pocamind `category`, generic ones left out) and a core talent $k$ (one that at least two talents need directly),
$$
G(c)=\{t:\operatorname{cat}(t)=c\},\qquad D(k)=\{k\}\cup\{t:\ t \text{ needs } k \text{ directly or through } D(k)\},
$$
each kept if it has at least 3 members. A group literal counts the build's picks in the group other than the rule's own item $x$:
$$
\lvert G(c)\rvert_b=\lvert (P_b\setminus\{x\})\cap G(c)\rvert .
$$
Group literals describe a build by what it is made of (its derivation trees and families) instead of by stat correlations.

**Grounding.** The stats a rule on $x$ may use are
$$
\Sigma(x)=\operatorname{req}(x)\;\cup\bigcup_{y\in\operatorname{pre}(x)}\operatorname{req}(y)\;\cup\bigcup_{g\ni x}\ \bigcup_{y\in g}\operatorname{req}(y),
$$
the stats named by the requirements of $x$, its prerequisites and its group mates. Anchors are **related** items: prerequisites and group mates. Dependants are excluded: with $\operatorname{dep}(x)$ the items that need $x$ directly or transitively, a literal $y\in\operatorname{dep}(x)$, $\lvert D(x)\rvert\ge n$, or $\lvert g\rvert\ge n$ with $g\setminus\{x\}\subseteq\operatorname{dep}(x)$ implies $x$ by the game's prerequisites, so a rule built on it restates game data. Such columns are constant in the search and are never literals. If no grounded take rule certifies for $x$, the search is run once more with every stat (`ground_fallback`); such a rule is marked "ungrounded stat convention".

**Stricter certification.** Builds that can take $x$ are split $A{:}B{:}C=50{:}25{:}25$. Rules are proposed on $A$ (depth at most 3), certified on $B$ as in Section 6.3 with support $m/4$, and must **replicate** on $C$: the same test at the unadjusted level $z$ (a second, independent confirmation; Bonferroni on a quarter split lost the small archetypes' rules). Every literal $\ell$ of a certified condition $\phi$ must be **necessary** on $B$: with $p_1$ the rate on the cell of $\phi$, $p_0$ the rate on the builds that $\phi\setminus\{\ell\}$ admits but $\phi$ excludes, $n_1,n_0$ their effective sizes and $\bar p$ the pooled rate,
$$
z_\ell=\frac{\pm(p_1-p_0)}{\sqrt{\bar p(1-\bar p)\,(1/n_1+1/n_0)}}\ \ge\ z, \tag{6.12}
$$
with sign $+$ for take and $-$ for never rules (the certification itself is at the Bonferroni level $z_t$ of (6.1a)). A literal that excludes nothing fails. This removes "nonsense that works": a literal that rides along with a real one is dropped.

**Review evidence.** Each reviewed build whose original context a rule $\rho$ applies to is a case. For a take rule on $x$: an add of $x$, or an unmarked $x$ on a build tri did not make is support; a remove of $x$ not for space is counter. Never rules are the mirror image. Removals for space and iffy marks say nothing. Case weights are the ledger weights ([Section 5](05-locality.md)) times the strength weight. Then:

1. no counter: $\rho$ **stands**;
2. counters: $\rho$ is **refined** by one literal $\ell$ (archetype, other token, group count, related pick or grounded stat) that holds on no counter case and on the most support, accepted only if $\rho\wedge\ell$ still certifies on the corpus (Wilson bound at $z$, support $\ge m$);
3. otherwise the newest case decides: a counter **overrules** $\rho$, a support keeps it (**disputed**).

Reviews are taken as accurate, so (3) follows the newest review. A refinement keeps the rule where reviews agree and silences it exactly where they do not.

**Stated families.** A reviewed build's family is the one its reviewer states, not the one it is read in:
$$
A_b=\begin{cases}a & \text{the review notes "\# is }a\text{" (else "\# }a\text{: ..." notes naming only }a)\\ \text{none} & \text{the reading lies in a family a "\# not" note denies}\\ \text{the reading} & \text{otherwise.}\end{cases}
$$
Reviews are accurate, so a statement outranks the locators. This scopes strong SM removals (OFF_ARCHETYPE, Section 5) and every case above. Without it a near-empty build reviewed as chime is unlocated, and its removals scope to nothing.

**Rules the reviews state.** Mined rules are certified on a corpus that knows nothing of archetypes beyond the locators' reading. Reviews do, so the selection set also holds rules read directly off them. For a talent $t$ (mantras keep worth) and a family $a$, let $V_a(t)$ be the reviews of family $a$ that bear on $t$ (an operation on $t$ or an unmarked keep), $w^+,w^-$ their support and counter weights, $o^\pm$ the number of explicit operations (adds and removals not for space, verdicts) on each side, and $\nu$ the newest of them. Then
$$
\begin{aligned}
[X]\ \ A=a\Rightarrow\neg t\quad&\text{if } \lvert V_a\rvert\ge 3,\ o^-\ge 2,\ w^-\ge 0.8\,(w^++w^-),\ \nu\text{ a counter},\\
[C]\ \ A=a\Rightarrow t\quad&\text{if } \lvert V_a\rvert\ge 3,\ o^+\ge 2,\ w^+\ge 0.8\,(w^++w^-),\ \nu\text{ a support}.
\end{aligned}
$$
Keeps count toward the share but never toward $o^\pm$: a family rule needs reviewers to have acted on $t$. (v3.0.2: the share is $0.75$, was $0.8$.)

**Availability.** A build that can never take $t$ says nothing about $t$. Write $\alpha_b(t)$ for "$t$'s requirements hold at $b$'s pre-shrine or final stats, or $b$ holds $t$" (`Catalog.available_in`). Every denominator below counts only builds with $\alpha_b(t)$. Before v3.0.2 every build counted, so a talent with a stat requirement, such as Everchanging Aegis, was held to its share of all builds and fell short of the foundational form.

For the foundational form, let $U(t)$ be the builds reviewed in full and not made by tri, outside the families with an $[X]$ rule on $t$, with $\alpha_b(t)$. Let $H(t)\subseteq U(t)$ be those that keep or get $t$. Then
$$
[F]\ \ \bigwedge_{a:\,[X]\,A=a\Rightarrow\neg t}A\ne a\ \Rightarrow\ t\qquad\text{if } \lvert U\rvert\ge 8,\ \lvert H\rvert\ge 0.75\,\lvert U\rvert,\ H \text{ spans}\ge 2 \text{ families},
$$
and no explicit counter is newer than the newest support (v3.0.2: $8$ and $0.75$, were $10$ and $0.85$). When the foundational form fails, endorsement alone may still state a family rule. With $U_a(t)$ the builds of $U(t)$ in family $a$ and $H_a(t)$ those that keep or get $t$,
$$
[C]\ \ A=a\Rightarrow t\qquad\text{if } \lvert U_a\rvert\ge 4,\ \lvert H_a\rvert\ge 0.75\,\lvert U_a\rvert,
$$
and no explicit counter on $a$ is newer than the newest support (`SelectionConfig.keep_rules`, v3.0.2). An unmarked keep is an endorsement (the review protocol), so a talent every chime reviewer left in place is central for chime even if none had to add it.

These rules are not certified on the corpus. They are what the reviewers said, written in the rule language, and they carry source `review`. On the 97 reviews of November 2026 (82 of them full reviews of builds tri did not make, 57 with a stated family), Exoskeleton is kept on 78 of the 82 and To The Finish on 76, and neither is ever removed, so both become $[F]\ \top\Rightarrow t$. On builds stated gank, Martyr is added 7 times; on builds stated chime, it is removed with strong SM 7 times. It becomes $[C]\ A=\text{gank}\Rightarrow$ Martyr and $[X]\ A=\text{chime}\Rightarrow\neg$Martyr, and no $\top$ rule. With availability, Everchanging Aegis is kept on 69 of the 70 builds that can take it and Emperor Flame on 21 of 23, and both become $[F]\ \top\Rightarrow t$; before, they were only $[C]$ on gank and chime. The review rules grow from 46 to 145, about 90 of them foundational.

**Generally good yields to the family.** A strong "generally good" add has global scope (Section 5). It no longer makes $t$ foundational on a build where an $[X]$ rule or an OFF_ARCHETYPE rule rejects $t$. One generally-good add of Martyr on a gank build does not outweigh ten removals on chime.

**Group rules may be foundational.** The breadth test (Section 6.3) now also applies to take rules whose literals are group counts. A rule such as $\lvert G(c)\rvert\ge 1\Rightarrow t$ that covers the corpus's pick profiles evenly is foundational, like a stat-only rule. Foundational rules are not subject to the chasm (Section 5.4). A mined central rule's home also includes every family whose reviewed builds kept $t$ under it. When two take rules on $t$ cover the same builds, the one with an archetype literal is kept over its proxies.

**The selection set.** With $\mathcal R_{\text{mined}}$ the certified rules after review evidence, $\mathcal R_{\text{auth}}$ the authored rules and $\mathcal S$ the suppressed rule ids (Section 6.13),
$$
\mathcal R=\big((\mathcal R_{\text{mined}}\cup\mathcal R_{\text{rev}})\setminus\mathcal S\big)\cup\mathcal R_{\text{auth}} ,
$$
with $\mathcal R_{\text{rev}}$ the rules the reviews state.

**Tiers.** A talent's tier is its manual category if set, else
$$
\tau(t)=\begin{cases}\text{foundational}&\exists\,\rho\in\mathcal R \text{ taking } t \text{ at the foundational pass}\\ \text{central}&\exists\,\rho\in\mathcal R \text{ taking } t \text{ at the central pass}\\ \text{luxury}&\text{otherwise.}\end{cases}
$$
**Tiers on the build.** $\tau(t)$ is global: one firing rule anywhere makes $t$ foundational or central everywhere. Read globally, a talent with only $[C]\ A=\text{chime}\Rightarrow t$ could enter a gank completion only through the fallback, after every luxury talent. That loses information the field holds, so the tier is read on the build $b$:
$$
\tau_b(t)=\begin{cases}\tau(t)&\text{a manual category, or } \exists\,\rho\in\mathcal R \text{ firing at } b \text{ and taking } t \text{ at the foundational or central pass}\\ \text{luxury}&\text{otherwise.}\end{cases}
$$
A talent with $\tau_b(t)\neq$ luxury enters only through its firing rule, a review claim or a necessary review rule. A talent with $\tau_b(t)=$ luxury is ranked by worth with the other luxury talents, unless a never rule, an $[X]$ rule, a hard review constraint or a sidegrade blocks it (`InferConfig.tier_context`, default on; off restores the global reading). The transformation is information-preserving in this sense: a rule that fires still decides, and where no rule fires the field's worth is used rather than discarded. The two older exceptions remain for the global reading. (i) If a take rule of $t$ has every non-item literal true at $b$ and lacks only co-picks, worth may add $t$: rules anchored on each other ($y\Rightarrow t$, $t\Rightarrow y$) never fire on a build holding neither, so worth starts the cluster and the rules then complete it. (ii) When no luxury talent fits and the cap is unfilled (`InferConfig.tier_fallback`); the completion notes it. Worth may remove a foundational or central talent that no rule requires at $b$, and a pick excluded by a never rule on the original build returns if the rule no longer holds on the answer and a take rule requires it. The talent cap $76-2m$ binds every case.

**Experiment.** On the rich synthetic world (6,000 builds mined, 4,000 fresh builds to check; `exp_v30_rules.py`), with "nonsense" the share of literals naming nothing the item's generating rule depends on:

| configuration | rules | hold on fresh builds | literals per rule | nonsense share | F/C items with a take rule |
|---|---|---|---|---|---|
| v2.25 (depth 5, any stat, anchors by mutual information) | 298 | 1.00 | 1.09 | 0.253 | 1.00 |
| v3.0 without the archetype reader | 199 | 1.00 | 0.95 | 0.189 | 0.90 |
| v3.0 with the archetype reader | 206 | 1.00 | 0.96 | 0.107 | 0.91 |

Both rule sets hold on fresh builds; v3.0 says it with fewer, shorter rules, and the share of proxy literals falls from 25% to 11%. The cost is recall on the smallest groups (hybrid and oath-and-murmur items with about 100 to 200 builds), which a quarter split cannot certify. Those items become luxury and are ranked by worth, which is what the tier rule intends for items no certified rule explains.

Code: `tri/rules/groups.py`, `tri/rules/passes.py` (`necessary`, replication, `ground_fallback`), `tri/rules/selection.py` (`build_selection`, `review_rules`), `tri/engine/fitting.py` (`stated_family`), `tri/infer/constraints.py` (`tier`, `tier_here`, `_fires_here`).

## 6.13 The rule language

Rules are written and read in the symbols of [Appendix C](appendix-c-statements.md), so a mined rule can be copied, edited and added back. A rule is
$$
[\pi]\quad \ell_1\wedge\dots\wedge\ell_k\ \Rightarrow\ x \qquad\text{or}\qquad [\pi]\quad \ell_1\wedge\dots\wedge\ell_k\ \Rightarrow\ \neg x,
$$
with pass $\pi\in\{F,C,X\}$ (foundational, central, exclusion) and literals
$$
\ell::=\ s\ge v\ \mid\ s<v\ \mid\ \tau\ \mid\ \neg\tau\ \mid\ A=a\ \mid\ \lvert G(c)\rvert\ge n\ \mid\ \lvert D(k)\rvert<n\ \mid\ y\ \mid\ \neg y\ \mid\ \top .
$$
ASCII forms are accepted (`>=`, `<`, `!`, `=>`, `&`, `T`). $s\le v$ is read as $s<v+1$ and $s>v$ as $s\ge v+1$. Text after the item (the printed rates) is ignored. Names are matched against the catalog, the context tokens, the archetype families and the groups; unknown names are an error, never a guess.

Each rule has an id, the first six hex digits of $\operatorname{sha1}$ of its key, so the same rule has the same id in every run. An authored entry either **adds** a rule (it always fires; its corpus rates and review evidence are shown) or **suppresses** a mined rule by id. Unlike mined rules, an authored rule may use $\neg y$, since an administrator can state an absence on purpose. Edits keep the id history, and authored rules are stored with the run (`authored_rules.json`).

Code: `tri/rules/language.py` (`parse`, `format_rule`, `rule_id`), `tri/rules/authored.py`, `Tri.add_rule`, `Tri.edit_rule`, `Tri.drop_rule`.
