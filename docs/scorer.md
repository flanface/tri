# The scorer and how reviews act

tri answers a build in one pass. Hard constraints and authored rules filter or force items. Every other item gets one logit, and slots are filled by that logit under the talent cap, the mantra slots and the exclusions. The code is `tri/infer/scorer.py`. Claims and their supersession are in `tri/reviews/authority.py`.

## Symbols

| symbol | meaning |
|---|---|
| $b$ | the build being answered; $u_b$ its point in concept space |
| $r$ | a reviewed build; $u_r$ its point |
| $t, s$ | items (talents and mantras) |
| $L_t$ | field logit of $t$ at $u_b$ |
| $q_t$ | neighbour pick rate of $t$ around $u_b$ |
| $c$ | a claim: one reviewer label (add, remove, keep, iffy) on one item |
| $w_c$ | claim weight after supersession |
| $\sigma(b,r)$ | build similarity |
| $\chi(b,r)$ | the chasm factor |
| $K_c(b)$ | reach of claim $c$ at $b$ |

## Similarity and the chasm

$$\sigma(b,r) = \chi(b,r)\,\exp\!\left(-\frac{\lVert u_b-u_r\rVert^2}{2h^2}\right), \qquad h^2 = \frac{\rho^2}{-2\ln\sigma_{\min}}$$

with $\rho = 1.2$ (reach) and $\sigma_{\min} = 0.8$, so $h = 1.797$. A build with the same build id has $\sigma = 1$. The chasm factor is

$$\chi = 1 - \beta\, a(u_b)\, a(u_r)\left(1-(1-\mathrm{TV})^{\gamma}\right),$$

where $\mathrm{TV}$ is the total variation distance between the archetype posteriors of the two points, $a$ is the locator activity, $\gamma = 6$ and $\beta = 1$. If both builds are located in different families, $\chi = 1-\beta = 0$. Reviews never cross archetype families.

Three similarity levels matter:

| level | value | meaning |
|---|---|---|
| $\sigma_{\min}$ | 0.80 | below: no conflict and no reach |
| $\sigma_{\text{full}}$ | 0.95 | at or above: the newer claim replaces the older outright, and reach counts as 1 |
| $\sigma_{\text{same}}$ | 0.99 | at or above: the reviewed build itself |

## Claims and supersession

A review of build $r$ gives one claim per item. Explicit operations become add or remove claims, sidegrade marks become iffy claims, and unmarked items become keep claims. Each claim starts at $w_c = 1$. A claim made before a patch on an item that the patch changed starts at $0.25$, and an unmarked item of a build that tri itself made starts at $0.25$.

When two claims on one item disagree, the older one is scaled by

$$f(\sigma) = \begin{cases} 0 & \sigma \ge \sigma_{\text{full}} \\ \dfrac{\sigma_{\text{full}}-\sigma}{\sigma_{\text{full}}-\sigma_{\min}} & \sigma_{\min} \le \sigma < \sigma_{\text{full}} \\ 1 & \sigma < \sigma_{\min} \end{cases}$$

using the smallest factor over all newer conflicting claims. Two exceptions apply:

- A keep claim supersedes an older add or remove only when $\sigma \ge \sigma_{\text{same}}$.
- A claim conditional on items the other build lacks does not conflict.

A broken claim conflicts at $\sigma = 1$ everywhere. Supersession is computed once when the review log is set. Nothing is refitted.

## Reach of a claim

Claims are read from reviews at $\sigma \ge \sigma_{\min}$. Reviews of another weapon class are skipped unless $\sigma \ge \sigma_{\text{same}}$. Broken and generally-good claims are context-free and reach every build with $K = 1$.

For an explicit add or remove whose reason is about the item itself (necessary for archetype, generally good, no synergy, scenario mismatch, broken), the reach is directed. Let $d = u_b - u_r$ and let $g_t$ be the unit gradient of $L_t$ in concept coordinates. Then

$$\pi = d\cdot g_t, \qquad K_c(b) = \chi(b,r)\,\exp\!\left(-\frac{\pi^2}{2h_\parallel^2}\right)\exp\!\left(-\frac{\lVert d\rVert^2-\pi^2}{2h_\perp^2}\right).$$

The kernel is narrow along the item's direction and wide across it. Every other claim uses $K_c(b) = \sigma(b,r)$. In both cases $K \ge \sigma_{\text{full}}$ counts as 1 and $K < \sigma_{\min}$ as 0.

## The logit

$$z_t = \alpha_0 + \alpha_L L_t + \alpha_q \operatorname{logit} q_t + \alpha_D D_t + \alpha_P P_t$$

Direct evidence sums every claim on $t$ in reach:

$$D_t = \sum_{c \text{ on } t} \operatorname{sign}(c)\, w_c\, K_c(b)\, \kappa_c, \qquad \operatorname{sign}(\text{add}) = \operatorname{sign}(\text{keep}) = 1,\ \operatorname{sign}(\text{remove}) = -1,$$

with $\kappa_c = \alpha_K$ for keep claims and echoed picks, and $\kappa_c = 1$ otherwise. Claims are combined by weight times reach. The newest claim does not hide a closer active one; superseded claims already have $w_c = 0$.

Propagated evidence nudges unclaimed similar items:

$$P_s = \sum_{c \text{ on } t} \operatorname{sign}(c)\, w_c\, K_c(b)\, \frac{\max(0, \cos(s,t)-\tau)}{1-\tau},$$

where $\cos$ is the cosine of the mean-centred worth vectors. It applies to add and remove claims with archetype-fit reasons only (necessary for archetype, no synergy, scenario mismatch), never between the two sides of a sidegrade, and never binds. It is off by default (`sc_propagate`), because it gained nothing measurable ([evaluation](evaluation.md)).

The neighbour rate $q_t$ is the pick rate among the nearest corpus builds. With `sc_nbr_reviews` it includes the reviewed builds, whose reviewed items count with the review weight. $q_t$ is then adjusted by the game priors (benchmarks, weapon redundancy), admin item values and reviewer notes.

## Binding claims

A binding claim overrides the scorer. A claim with $w_c \ge 0.5$ binds when any of the following holds:

- it is on the build itself ($\sigma \ge \sigma_{\text{same}}$);
- it is a broken removal;
- it is a strong generally-good add (where feasible and not excluded);
- it is a strong add or remove with a non-relative reason from a build at $\sigma \ge 0.95$;
- it is a sidegrade mark.

Relative reasons are less necessary for space, worth less as a luxury and worth more as a luxury. Because the build itself always binds, every review reproduces on its own build at once.

## Selection

1. Game constraints: Vow of Iron, weapon rules, never rules and mutual exclusions. Of two exclusive picks, the lower $z$ goes.
2. Binding removals. A "needs" removal adds the needed picks if there is room and removes otherwise.
3. Game limits, then forced picks. An input over the talent cap $76-2m$ or over its mantra slots first loses its lowest-$z$ unprotected picks; prerequisites in use are never removed. Then forced mantras go in: weapon rules, binding adds, admin categories, authored take rules and the caller's answers, each with its prerequisite chain placed first.
4. Removals: an unprotected pick with $z_t < \theta_-$ is removed unless it is a prerequisite in use, a slot grant in use or the only dependant of a picked prerequisite.
5. Fill by $z$: mantras first (talents only on Silentheart), then the forced talents, then talents. A review fills only with items scoring at least $\theta_+$ and keeps the input's counts. A completion fills every slot. Wildcard slots take any mantra type.
6. Swaps: up to 3 times, an outside item replaces the weakest removable pick when it scores more than $\delta$ higher.

A sidegrade side is committed as soon as one of its items is picked, so the two sides of a sidegrade are never blended. Removed items are never re-added.

## Constants

| name | symbol | value |
|---|---|---|
| `sc_bias` | $\alpha_0$ | 0 |
| `sc_field` | $\alpha_L$ | 0 |
| `sc_nbr` | $\alpha_q$ | 1 |
| `sc_direct` | $\alpha_D$ | 2 (fitted; was 3) |
| `sc_prop` | $\alpha_P$ | 1 (acts only with `sc_propagate`) |
| `sc_keep` | $\alpha_K$ | 0.5 |
| `sc_remove` | $\theta_-$ | −4 |
| `sc_fill` | $\theta_+$ | 1 |
| `sc_swap` | $\delta$ | 6 |
| `sc_h_par` | $h_\parallel$ | 1.797 ($= h$) |
| `sc_h_perp` | $h_\perp$ | 4.49 ($= 2.5h$) |
| `sc_tau` | $\tau$ | 0.5 |

There are no per-item parameters. How these values were fitted is in [evaluation](evaluation.md).

## Categories

`Tri.categories()` reports foundational, central or luxury for each item, in this order of authority:

1. the admin category;
2. the pass of an authored take rule;
3. foundational for a talent taken by at least 90% of the corpus builds that can take it;
4. luxury.

Only admin categories and authored rules force picks. The pick-rate tier is reported but not enforced.
