# Evaluation

The code is `tri/evaluation.py`, and `tri.api.evaluate` wraps it. `python -m bot.evaluate` runs it on a snapshot of the live tables. Every number below was measured on the real reviews database and the fitted live run. The synthetic test worlds are used for tests only.

## Data

The data are 198 reviews of 190 distinct reviewed builds, with 114 locators. Reviewers marked 512 adds, 620 removes and 6 iffy items.

## Leave-one-build-out operation F1

A unit is one reviewed build, and its truth is the newest review of that build. For each unit $u$:

1. drop the reviews of $u$ and of every reviewed build $r$ with $\sigma(u,r) \ge 0.95$;
2. answer $u$ with the remaining reviews (`Tri.review`, best archetype);
3. compare tri's adds $A$ and removes $R$ with the reviewer's $A^{\star}$ and $R^{\star}$.

Counts are pooled over units:

$$P_+ = \frac{\sum|A\cap A^{\star}|}{\sum|A|}, \qquad P_- = \frac{\sum|R\cap R^{\star}|}{\sum|R|}, \qquad \mathrm{Rec} = \frac{\sum\left(|A\cap A^{\star}|+|R\cap R^{\star}|\right)}{\sum\left(|A^{\star}|+|R^{\star}|\right)},$$

$$P = \frac{\sum\left(|A\cap A^{\star}|+|R\cap R^{\star}|\right)}{\sum\left(|A|+|R|\right)}, \qquad F_1 = \frac{2P\,\mathrm{Rec}}{P+\mathrm{Rec}}.$$

Iffy marks are not scored.

## Review agreement

For every active review, tri answers the reviewed build itself with the full review log, keeping the input's counts. Agreement is the share of the review's claims reproduced: removed items removed, added items added and kept items kept. The reported value is the mean over reviews. It measures convergence: a review acts at once and reproduces on its own build.

## Standard errors

Every number has a bootstrap standard error over units ($B = 1000$; for agreement, over reviews). A change is compared with its predecessor by a paired bootstrap on the same units:

$$\Delta = m(\text{after}) - m(\text{before}), \qquad \mathrm{se}(\Delta) = \operatorname{sd}_{\text{boot}}\left(m^{\star}(\text{after}) - m^{\star}(\text{before})\right).$$

A change to the method or to a constant is accepted only if $\Delta F_1 > 2\,\mathrm{se}(\Delta)$, or, for a change the release required, if it does not lower $F_1$.

## Measured numbers

The 4.0 release history (mean, with standard error in brackets):

| state | $F_1$ | $P_+$ | $P_-$ | Rec | tri ops | agreement |
|---|---|---|---|---|---|---|
| v3.11.1 | 0.324 (0.016) | 0.185 (0.012) | 0.243 (0.018) | 0.683 (0.022) | 3522 | 0.955 (0.003) |
| library detached | 0.323 (0.016) | 0.184 (0.012) | 0.242 (0.018) | 0.683 (0.022) | 3530 | 0.956 (0.003) |
| one scorer | 0.375 (0.019) | 0.348 (0.023) | 0.356 (0.032) | 0.402 (0.023) | 1246 | 0.973 (0.002) |
| claims by weight × reach | 0.375 (0.019) | 0.348 (0.023) | 0.356 (0.032) | 0.402 (0.023) | 1246 | 0.976 (0.001) |
| keep supersedes on the same build only | 0.407 (0.019) | 0.350 (0.023) | 0.393 (0.030) | 0.447 (0.023) | 1311 | 0.977 (0.001) |
| directed reach | 0.408 (0.019) | 0.350 (0.023) | 0.392 (0.030) | 0.450 (0.023) | 1321 | 0.977 (0.001) |
| propagation measured, off | 0.408 (0.019) | 0.350 (0.023) | 0.392 (0.030) | 0.450 (0.023) | 1321 | 0.977 (0.001) |
| fitted constants ($\alpha_D = 2$) | 0.412 (0.019) | 0.359 (0.023) | 0.402 (0.031) | 0.447 (0.023) | 1278 | 0.978 (0.001) |

Reviewers make 5.9 operations per build. v3.11.1 made 18.5.

Paired gains, each against the row above it:

| change | $\Delta F_1$ | $\Delta P_+$ | $\Delta P_-$ | $\Delta$Rec |
|---|---|---|---|---|
| one scorer | +0.053 (0.015) | +0.164 (0.017) | +0.114 (0.020) | −0.281 (0.022) |
| claims by weight × reach | 0 | 0 | 0 | 0 |
| keep supersedes on the same build only | +0.031 (0.008) | | +0.037 (0.012) | +0.046 (0.010) |
| directed reach ($h_\perp = 2.5h$) | +0.0008 (0.0023) | | | |
| fitted constants | +0.0040 (0.0020) | +0.0090 (0.0037) | +0.0097 (0.0030) | −0.0037 (0.0023) |

Claims by weight × reach raised agreement by +0.0032 (0.0008). Directed reach with $h_\perp = 2h$ gained +0.0005 (0.0023), and with $3h$ +0.0006 (0.0024).

## Ablation of the old rule layers

Each layer was switched off on the detached library. The gain is the metric with the layer off minus the metric with it on.

| layer off | $\Delta F_1$ | $\Delta P_+$ | $\Delta P_-$ | $\Delta$Rec |
|---|---|---|---|---|
| mined rulebook | +0.0004 (0.0007) | | | |
| induced review rules | +0.023 (0.006) | +0.018 (0.005) | +0.028 (0.005) | −0.021 (0.012) |
| family pooling | −0.005 (0.003) | −0.009 (0.003) | −0.003 (0.004) | +0.020 (0.008) |
| consensus tells | −0.002 (0.003) | −0.001 (0.002) | −0.001 (0.003) | −0.011 (0.007) |
| all four | +0.008 (0.008) | +0.008 (0.007) | +0.014 (0.007) | −0.034 (0.016) |

No layer gained more than two standard errors, so all four were removed. On the live run, the mined rulebook had no rule that acted.

## Propagation to similar cards

| $\tau$ | $F_1$ | $\Delta F_1$ against off |
|---|---|---|
| 0.3 | 0.405 (0.019) | −0.0030 (0.0017) |
| 0.5 | 0.408 (0.019) | +0.0003 (0.0016) |
| 0.7 | 0.407 (0.019) | −0.0005 (0.0005) |

No setting gained, so `sc_propagate` is off.

## Fitting the constants

Coordinate descent over the scorer constants used the LOBO $F_1$. Each step changes one constant from the current best. A step is accepted when $\Delta F_1 > 2\,\mathrm{se}$, and later steps start from the accepted point. There are no per-item parameters.

| step | constant | from | to | $\Delta F_1$ | accepted |
|---|---|---|---|---|---|
| 1 | $\theta_-$ | -4 | -3.5 | -0.0086 (0.0034) | no |
| 2 | $\theta_-$ | -4 | -4.5 | +0.0036 (0.0033) | no |
| 3 | $\alpha_D$ | 3 | 2 | +0.0040 (0.0020) | yes |
| 4 | $\alpha_q$ | 1 | 0.5 | -0.0007 (0.0077) | no |
| 5 | $\alpha_q$ | 1 | 2 | -0.0667 (0.0104) | no |
| 6 | $\alpha_L$ | 0 | 0.5 | -0.0879 (0.0100) | no |
| 7 | $\alpha_L$ | 0 | 1 | -0.1955 (0.0121) | no |
| 8 | $\alpha_0$ | 0 | -0.5 | -0.0108 (0.0039) | no |
| 9 | $\alpha_0$ | 0 | 0.5 | +0.0030 (0.0036) | no |
| 10 | $\alpha_K$ | 0.5 | 0.25 | +0.0006 (0.0016) | no |
| 11 | $\alpha_K$ | 0.5 | 1 | -0.0018 (0.0018) | no |
| 12 | $\theta_+$ | 1 | 0.5 | +0.0008 (0.0020) | no |
| 13 | $\theta_+$ | 1 | 2 | -0.0039 (0.0029) | no |
| 14 | $\delta$ | 6 | 4 | -0.0089 (0.0031) | no |
| 15 | $\delta$ | 6 | 9 | +0.0030 (0.0021) | no |
| 16 | $h_\parallel$ | 1.797 | 1.4 | +0.0000 (0.0000) | no |
| 17 | $h_\parallel$ | 1.797 | 2.3 | +0.0000 (0.0000) | no |
| 18 | $\alpha_D$ | 2 | 1.5 | +0.0016 (0.0018) | no |
| 19 | $\theta_-$ | -4 | -4.5 | +0.0018 (0.0030) | no |
| 20 | $\delta$ | 6 | 9 | +0.0030 (0.0021) | no |

The first pass (steps 1–17) visited each constant in turn. The second pass (steps 18–20) revisited the three closest candidates from the accepted point $\alpha_D = 2$.

$h_\parallel$ had no effect: no answer changed between 1.4 and 2.3. The directed kernel's width across the item ($h_\perp$) dominates, because the item direction is one of many concept dimensions. $\tau$ and $\alpha_P$ act only with propagation on (see above).

## Speed

Adding one review to the 198-review log (`add_reviews`) takes 1.0 s on the live run (median of 3; 2 CPUs). It was 4.28 s on v3.11.1 and 3.5 s after the step 1 memo on build contexts. Rule induction, removed in 4.0, took 1.7 s of that. The time now goes to supersession (0.61 s, 54,323 pair similarities) and locator reads (0.55 s). `with_reviews` takes 0.94 s and `dry_insert` 1.36 s. The rebuild path was not optimised further.
