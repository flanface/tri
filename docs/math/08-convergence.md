# 8. Rapid convergence

## 8.1 Definition

> **Rapid convergence.** A correction takes effect immediately and completely on the builds it concerns: one review fixes its own build at once, and a handful of reviews fixes an archetype, without retraining the model. When reviews disagree, the most recent one decides at once, rather than being outvoted by older ones. Every numerical procedure that places or reads a build reaches its unique answer in a bounded, small number of steps.

## 8.2 Interpretation

In ordinary machine learning, a correction becomes part of the model only when it is retrained, and then it is one observation among thousands. That is slow (a refit), and weak (a single correction barely moves a fitted model). It is the wrong behaviour for expert reviews, which are accurate, rare and often about a change in the game's meta.

tri separates **learning** from **remembering**. The field and rules are learned slowly from the corpus. Reviews are remembered, and applied at query time through the concept space. So the speed of a correction does not depend on how much corpus data disagrees with it.

Rapid convergence must not come at the expense of locality. A correction converges quickly **inside its reach** and does not leak outside it ([Section 5](05-locality.md)).

## 8.3 Mechanism: reviews act at once

**The ledger.** Every accepted review is stored in the review ledger (`authority.ReviewLedger`). For any build, `claims_for` returns, per item, the newest active claim among reviews within reach ([Section 5.3](05-locality.md)) that holds for the build. Nothing is fitted.

**Binding or nudging.** A claim **binds**, overriding rules and worth, if its strength is at least 0.5 (`memory_hard`) and any of these holds:

- it is about this build itself (similarity at least 0.99);
- it is a broken removal;
- it is a strong GG add from another archetype ([Section 5.5](05-locality.md));
- it is a strong add or remove whose reason is about the item, not slot space, made on a build at similarity at least 0.95 (`bind_sim`, v2.23).

Otherwise it **nudges** worth by $\pm0.3\times\text{strength}$ (`memory_nudge`): a soft push rather than an order.

**Consequence.** One strong review on a build changes the analysis of that build, and of every build within its full-strength band, on the next query.

## 8.4 Mechanism: newest wins

Two claims **conflict** if they concern the same item, carry different labels, come from builds within reach of each other, and each holds in the other's build. When they conflict, the older claim's weight is multiplied by

$$
\phi(\text{sim})=\begin{cases}0, & \text{sim}\ge0.95,\\[4pt]\dfrac{0.95-\text{sim}}{0.95-0.80}, & 0.80\le\text{sim}<0.95,\\[8pt]1, & \text{sim}<0.80.\end{cases} \tag{8.1}
$$

**Interpretation.** A newer review on the same build or a very similar one replaces the older one completely. On a looser neighbour, both survive in proportion. Outside reach, they do not interact at all. A changed meta therefore takes one review per archetype to correct, not enough reviews to outvote the old ones.

**Worked example.** An old review removed X from build A. A new review keeps X on build B at similarity 0.875. The old claim's weight becomes $(0.95-0.875)/0.15=0.5$ of what it was. If B were at similarity 0.97, the old claim would vanish.

**Proposition 8.1 (one outcome).** For any set of reviews with ordered timestamps, the final claim weights are uniquely determined.

*Reason.* Each final weight is its base weight times the factor (8.1) for every newer conflicting claim, which depends only on the set and the order. $\square$

Missing timestamps count as oldest. A broken removal conflicts with every claim on its item, so a newer endorsement overrides it.

## 8.5 Mechanism: reviewed neighbours count 40 times

In the blended worth (4.5), an item that a review of a neighbour changed carries evidence weight 40 (`review_weight`). One reviewed neighbour at kernel weight 0.5 contributes like 20 ordinary corpus builds at full weight. Since the field's prior counts as 15, a single close reviewed neighbour moves worth most of the way to the reviewer's answer, even before any binding rule applies.

The corrected build enters at the coordinates of the original build, and conditional claims enter only for builds where they hold ([Section 5.5](05-locality.md)), so the boost stays local.

## 8.5a Mechanism: reviewer consensus

A single note or edit acts locally (within reach of the reviewed build). When several reviewers say the same thing about one family, the agreement is a property of the family, and tri treats it as one (`tri.consensus`).

**Evidence.** Each review contributes signed evidence $e=(\pi,t,s,w)$ on family $\pi$ and item $t$: an indicator or core note on $t$ gives $s=+1$, $w=1$; an avoid note $s=-1$, $w=1$; an add with reason NA on a build located in $\pi$ gives $s=+1$, $w=\omega\cdot\text{strength}$; a remove with reason NS, WA, BR or SM gives $s=-1$, $w=\omega\cdot\text{strength}$ ($\omega=0.5$, `consensus_op_weight`; strength 1 strong, 0.5 weak). Per reviewer and reviewed build, only the newest evidence on $(\pi,t)$ counts (v2.23; before, per reviewer only).

**Tally.** With $n^\pm(\pi,t)$ the summed weights,

$$
\nu=n^+-n^-,\qquad c(\pi,t)=\frac{|\nu|}{|\nu|+\kappa},\qquad\kappa=1.5. \tag{8.1a}
$$

The pair is a **tell** (take if $\nu>0$, avoid if $\nu<0$) once at least two distinct (reviewer, build) units back its sign (`consensus_min_reviewers`, `consensus_unit="build"`). Counting reviewers instead (`consensus_unit="reviewer"`) promoted nothing on the real record, which has one reviewer. The same verdict on two different builds is two observations, while repeating it on one build is still one.

**Effect.** For a build $u$ with family reading $p(\pi\mid u)$ and activity $a(u)$,

$$
q_t\leftarrow\operatorname{clip}\bigl(q_t+\operatorname{sgn}(\nu)\,\eta_c\,c\,p(\pi\mid u)\,a(u)\bigr),\qquad\eta_c=0.35, \tag{8.1b}
$$

and a take tell held by $u$ lifts the log odds of $\pi$ by $\lambda_c\sum c$ ($\lambda_c=2$ nats, total capped at the indicator lift 6) where $p(\pi\mid u)\ge0.2$. A take tell with $c\,p\,a\ge0.3$ also confirms the pick as central ([Section 7.10](07-strictness.md)). The tells are listed on the archetype card.

**Proposition 8.2 (monotone convergence).** With $k$ agreeing reviewers and no dissent, $c=k/(k+\kappa)$ is increasing in $k$, so $c\to1$: two reviewers give $0.57$, four $0.73$, ten $0.87$. A reviewer repeating a note leaves $c$ unchanged, and one dissenting reviewer subtracts at most $1$ from $\nu$.

*Proof.* $k\mapsto k/(k+\kappa)$ is increasing for $\kappa>0$, and deduplication by reviewer keeps one piece of evidence per reviewer. $\square$

**Locality.** The gate $p(\pi\mid u)\,a(u)$ is near zero on builds of other families, and Proposition 5.1 holds: tells move a family everywhere and nowhere else.

**Worked example.** Three reviewers write "# Gank: Replenishing Knockout, Agile Dash". Then $c=3/4.5=0.67$ for both talents. A Gank build far from every reviewed build ($p=0.9$, $a=1$) gains $0.35\times0.67\times0.9=0.21$ worth on each, and both are confirmed central; a Chime build gains nothing.

## 8.6 Mechanism: locators and settings without retraining

Adding or removing a locator changes readings, region names, the chasm and rule homes instantly. Changing the metric (`Tri.set_metric`) recomputes weights, coordinates and regions in seconds without retraining the field, because the archetypes do not depend on the weights.

## 8.7 Mechanism: fast, exact numerics

- Placing a build is the majorise-minimise solve of [Section 3.3](03-worth.md): monotone, unique, with one fixed $d\times d$ matrix and at most 200 steps (it stops when no coordinate moves by more than $10^{-5}$; `infer_iters`, `infer_tol`).
- Leave-one-out worth is closed-form (3.3): one formula per item, no refits.
- The distance between builds is a short vector difference (Theorem 4.1), not a sum over every item.
- Training and the archetype solves can run on a GPU (CUDA or Apple Metal) or on PyTorch's CPU kernels (Section 8.12). The results agree with the float64 NumPy path to float32 precision, which leaves locality, accuracy and strictness unchanged.

## 8.8 Optional: folding reviews into the field

Reviews never need a refit to act. If you choose to call `Tri.refit()` in your own code, corrected builds join the corpus with weight

$$
\kappa=\min\Big(10,\ \frac{0.25}{0.75}\cdot\frac{\text{corpus weight}}{\text{review weight}}\Big), \tag{8.2}
$$

so reviews are at most 25% of the total weight (`refit_review_weight`, `refit_review_share`). The field, space, regions and rulebase are rebuilt, and a review can retire a rule outright. With at least 20 reviews, refit also tunes the swap margin and the no-fit rate against reviewers' operations.

**Worked example.** With 10,000 corpus builds and 100 reviews, the cap is $\tfrac13\cdot10{,}000/100\approx33$, so $\kappa=10$. With 5,000 reviews it is $\tfrac13\cdot2=0.67$: reviews can never outweigh the corpus three to one.

Nothing refits on its own, and `tri.cli.fit` fits on the corpus only. Refitting is a deliberate decision made in Python.

## 8.9 How rapid convergence is measured

The **meta shift** experiment: reviewers say one central talent no longer belongs in archetype 0. Measure the share of 188 unreviewed archetype-0 builds where tri now removes it, after 1 review, after 5, and after a refit. In the v2.4 rerun with the default metric, this is 0.93 after 1 review, 1.00 after 5 reviews, and 1.00 after a refit (v1.0: 0.67 after 1 review).

Review agreement after 100 reviews and a refit is the other measure: add precision rose from 0.70 to 0.80 and reason accuracy from 0.87 to 0.98 in the simple world ([Section 11](11-experiments.md)).

## 8.10 Mechanism: new items without refit

New talents and mantras appear in the game after the corpus was gathered. tri adds them without a refit (`Tri.extend_catalog`). A new item $t$ gets a zero direction, $G_t=O_t=V_t=Q_t=0$, and the intercept $\operatorname{logit}(\bar p_k)$ where $\bar p_k$ is the median pick rate of its kind. Then for every build the logit of every existing item is unchanged, so the archetype solve, the coordinates (4.1) and every distance are exactly unchanged.

**Proposition 8.2 (extension is invisible).** After extension, every coordinate, region, rule and answer for builds without the new item is the same as before.

*Reason.* Coordinates and the metric depend only on fitted items: the metric weights are normalised over the fitted set of size $N_{\text{fit}}$, and new items are not fitted. Rules mention only old items. Worth of old items is unchanged because neighbours and kernels are unchanged. $\square$

The new item is **unrated**: corpus builds before its release time (the commit date of the game data) could not take it, so their evidence weight $\rho_{nt}$ in (4.5) is 0, and an item with local support below `cold_support` (1.0) has no worth evidence. Worth never adds an unrated item and never drops one. Evidence arrives first from reviews, which carry weight 40 (Section 8.5): one strong generally good review adds the item on every build it reaches. At the next refit, builds released after the item rate it like any other, and pre-release builds are masked from its column.

**Game data sync.** `tri.sync` compares a fingerprint of every item in the pocamind/data game data (requirements, prerequisites, exclusions, slot flags, mantra type) with the last sync: items not in the catalog are added, changed items take the new requirements and exclusions, and items gone from the data are **retired** (never available, never suggested; old reviews that name them still load). The bot checks every 12 hours and applies the result; `/model sync` does the same on demand.

**Locks on sync.** A sync also refreshes the context clauses ([Section 2.2](02-constraints.md)) of every kept item, from pocamind/data and, when it is reachable, from deepwoken.co. Clauses are only added. A lock makes an item unavailable but does not change what the item does, so it is not in the fingerprint and does not mark rules stale. A run fitted before a lock existed gets it at its next sync: its answers stop offering the locked item at once. The field and the rules keep the statistics they were fitted with until the next refit.

## 8.11 Limits

- Memory acts on neighbours, so before a refit, many new reviews can lower add precision temporarily: tri adds what reviewers added on similar builds, including adds that were about slot space on those builds. The binding rules (item-specific reasons only) limit this.
- Convergence is local by design. A new archetype that no review or locator has touched converges only as fast as the corpus reflects it.
- Until a refit, an unrated item is suggested only where reviews or a category ask for it. At the refit, pre-release builds still enter the archetype solves; they are masked only from the new item's column.

## 8.12 Mechanism: compute that does not change the answer

Fitting and loading are made faster in three ways. Each one comes with an argument that the speed-up does not change what the model says.

**Device numerics.** On a torch device (CUDA, Metal `mps`, or CPU) the field is trained by the same minibatch Adam as the NumPy path, with the same NumPy random draws for initial values and batch order, but in float32 (Metal has no float64). Let $\theta$ be the float64 iterate and $\tilde\theta$ the float32 iterate. Each step is a Lipschitz map of the parameters and of the gradient, and rounding adds a relative error of about $\epsilon_{32}\approx 6\times10^{-8}$ per operation. So $\lVert\tilde\theta-\theta\rVert$ grows at most geometrically in the number of steps, and in practice it stays near $10^{-5}$ because the loss is strongly convex in each build's archetype and the L2 penalty contracts the item parameters. On 20,000 real-catalogue builds the two paths give the same explained deviance (0.35778) and validation score (0.110256), parameters within $1.3\times10^{-5}$, and pairwise concept distances with correlation $1-10^{-13}$. Since every guarantee in Parts I and II is stated in terms of distances, logits and certified bounds, and certification uses held-out data with a margin far larger than $10^{-5}$, the guarantees carry over. A GPU fit is not bit-identical to a NumPy fit, and two runs on different devices can differ in the last digits.

**Latent memo and load cache.** The archetype of build $n$ is the unique minimiser $r_n^\star(\theta, x_n)$ of a strictly convex loss ([Section 3.3](03-worth.md)). It depends only on the item parameters $\theta=(\beta,G,O,V,E)$, the solver settings and that build's row $x_n$ (picks, stats, oath, murmur and availability). Write $\kappa(\theta)$ and $\kappa(X)$ for 128-bit BLAKE2b fingerprints of those inputs.

**Proposition 8.3 (the cache is exact).** If the stored key equals $\kappa(\theta)\Vert\kappa(X)$, the stored latents equal a fresh solve, up to hash collisions (probability about $2^{-128}$).

*Proof.* Equal keys mean equal $\theta$, solver settings and rows. The minimiser is unique and the solver is deterministic, so a fresh solve returns the stored array. $\square$

`Tri.save` writes `latents.npy` with its key and `Tri.load` reads it. A run whose parameters or data changed has a different key, so the cache is ignored, solved again and rewritten. The same memo (the last three matrices) serves repeated whole-corpus solves inside a fit and across a catalog extension. An extension leaves every latent unchanged by Proposition 8.2, so the old latents are stored again under the new key.

**Order-free rule mining.** Each item's rules are mined with their own random stream, seeded by (seed, item index). The mined rulebook is then a function of the item and the data alone, not of the order in which items are processed. So mining the items in parallel processes and putting them back in order gives exactly the same rulebook as one process (`PassConfig.n_jobs`). This changed the random streams compared with v2.7, so a refit in v2.8 can mine slightly different rules from the same data. The certification guarantee ([Section 6](06-accuracy.md)) holds for any stream.
