# 3. The worth field

## 3.1 The model

The worth field answers one question for every build $b$ and item $t$: **if this build can take this item, how likely is it to?** It is logistic regression with one extra ingredient, a learned archetype per build.

For each build, tri computes three descriptions:

- **stat steps** $s_b$: for each stat and each threshold 5, 10, ..., 95, a 1 if the highest stat reaches it. Steps, not raw values, so a convention such as "only from 50 Agility" is a single jump the model can learn;
- **context** $c_b$: one-hot oath and murmur tokens;
- **archetype** $z_b$: a vector of $d=16$ numbers, the part of the build's choices that stats and context do not explain. It is the sum of a learned vector for its context tokens and a residual of its own.

The logit of item $t$ for build $b$ is a sum of four effects:

$$
\ell_{bt}=\beta_t+\Gamma_t\cdot s_b+\Omega_t\cdot c_b+V_t\cdot z_b,\qquad p_{bt}=\sigma(\ell_{bt}). \tag{3.1}
$$

| term | meaning |
|---|---|
| $\beta_t$ | how popular $t$ is in general |
| $\Gamma_t\cdot s_b$ | how much the build's stat investment raises or lowers it |
| $\Omega_t\cdot c_b$ | the effect of the oath and murmur |
| $V_t\cdot z_b$ | the effect of the build's archetype; $V_t$ is the item's archetype profile |

$p_{bt}$ is the **field worth**: the probability that a build like $b$ takes $t$, given that it can.

It is convenient to stack the three descriptions into one **description vector** $y_b$ and the item coefficients into one row $A_t$, so that $\ell_{bt}=\beta_t+A_t\cdot y_b$. The whole field is then one matrix $A$ and one vector $\beta$.

**Interpretation.** This is a factor model. $V_t\cdot z_b$ says "items with a profile pointing the same way as this build's archetype are taken more". Two builds with similar archetypes, stats and context get similar logits for every item. That is what [Section 4](04-concept.md) turns into geometry.

## 3.2 Training

The parameters are fitted by minimising the usual logistic loss (cross-entropy), summed over every build and every item **the build could take**, plus small L2 penalties:

$$
\text{loss}=\sum_{\text{builds}} w_b\sum_{\text{available items}}\rho_{bt}\,\big[\text{cross-entropy of } p_{bt} \text{ against picked or not}\big]+\lambda_r\sum_b\lVert r_b\rVert^2+\lambda_\theta\lVert\theta\rVert^2. \tag{3.2}
$$

- Unavailable items contribute nothing.
- $\rho_{bt}$ is 0.25 for builds made before a patch that changed item $t$ ([Section 10](10-patches.md)), else 1.
- $\lambda_r=5$ (`l2_latent`) keeps the per-build residuals small; $\lambda_\theta=0.01$ (`l2_items`) keeps item coefficients finite.
- The optimiser is minibatch Adam (1000 steps by default).

Builds named by locators ([Section 4.8](04-concept.md)) and a random 5% of the corpus ([Section 6.2](06-accuracy.md)) are left out of training.

## 3.3 The exact archetype of any build

After training, the item parameters are frozen and every build's residual archetype is solved again, exactly. For a given build, the residual is the one that best explains its picks: it minimises that build's own cross-entropy plus the L2 penalty. For a finished build the evidence is every available item (picked or refused). For a partial build only the picks count, because an unfinished build has refused nothing.

**Proposition 3.1 (one answer).** This minimisation has exactly one solution.

*Reason.* Each cross-entropy term is convex in the logit, the logit is linear in the residual, and the penalty $\lambda_r\lVert r\rVert^2$ is strictly convex and grows without bound. A strictly convex function that grows without bound has exactly one minimum. $\square$

**Proposition 3.2 (the solver always improves).** tri solves it with a majorise-minimise step: Newton's method with the curvature replaced by its upper bound $H=\tfrac14V^\top V+\lambda_rI$. Each step lowers the objective, and it stops lowering it only at the solution.

*Reason.* The logistic curvature $p(1-p)$ is at most $\tfrac14$, so $H$ is never smaller than the true curvature. A quadratic with that curvature sits above the objective and touches it at the current point; moving to its minimum can only go down. $\square$

**Interpretation.** Every build, new, reviewed or partial, gets a deterministic point: the same input always gives the same archetype, and placing a build takes milliseconds. This is the foundation of rapid convergence ([Section 8](08-convergence.md)): a review can be applied to a new build without retraining anything.

Code: `ConceptSpace.solve_latent`, `_mm`.

## 3.4 Leave-one-out worth

To judge whether a pick suits a build, the pick itself must not be the evidence. Otherwise every pick would justify itself. tri therefore computes the logit item $t$ would have if the build's decision on $t$ were left out of its archetype. One Newton step and the Sherman-Morrison identity give it in closed form:

$$
\ell^{\text{loo}}_{bt}=\ell_{bt}+\frac{(p_{bt}-x_{bt})\,h_t}{1-p_{bt}(1-p_{bt})\,h_t}, \tag{3.3}
$$

where $x_{bt}$ is 1 if $b$ picked $t$, and $h_t$ measures how much $t$ alone moves the archetype (its leverage). The correction applies only to items that were evidence for the build.

**Interpretation.** A pick that the rest of the build does not support drops; a non-pick that the rest of the build calls for rises. This is how tri can say "this pick does not fit" about a pick the build actually made.

Code: `ConceptSpace._loo`.
