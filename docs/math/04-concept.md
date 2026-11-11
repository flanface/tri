# 4. The concept field

## 4.1 The concept map

The concept field gives each build a coordinate $u_b$. Write the field's logits for build $b$ as a list over all $N$ items. The **concept distance** between two builds is the weighted root mean square difference of their logits:

$$
d(b,b')=\sqrt{\frac1N\sum_{t}\omega_t\,(\ell_{bt}-\ell_{b't})^2}. \tag{4.1}
$$

The weights $\omega_t$ (Section 4.4) say how much each item counts. Computing (4.1) for every pair would require all $N$ logits per build. Instead tri builds coordinates with the same distances. Take the QR factorisation of the weighted coefficient matrix, $\Omega^{1/2}A=QR$, where $Q$ has orthonormal columns and $R$ is a small square matrix. The concept map is

$$
u_b=\frac{R\,(y_b-\bar y)}{\sqrt N}, \tag{4.2}
$$

with $\bar y$ the average description vector.

**Theorem 4.1 (isometry).** For any two builds, the straight-line distance between their coordinates equals their concept distance (4.1).

*Reason.* The weighted logit difference is $\Omega^{1/2}A(y_b-y_{b'})=QR(y_b-y_{b'})$. Multiplying by $Q$ does not change length, because its columns are orthonormal. So the length is $\lVert R(y_b-y_{b'})\rVert$, and dividing by $\sqrt N$ gives $\lVert u_b-u_{b'}\rVert$. $\square$

**Interpretation.** Distance in the concept space **is** a difference in worth, item by item, not a proxy for it. Two builds at distance 0 receive identical logits for every item. The check in `plots.distance_check` and `evaluate.mapping_benchmark` finds errors of about $10^{-15}$, which is rounding.

**Fitted items.** Here $N$ is the number of fitted items $N_{\text{fit}}$. Items added after the fit ([Section 8.10](08-convergence.md)) have zero rows in $A$, so they add nothing to any distance, and the theorem holds unchanged after any number of extensions.

## 4.2 Items as directions

Each item is a direction in the concept space. With $g_t$ a fixed vector computed from row $t$ of $Q$,

$$
\ell_{bt}=\text{constant}_t+g_t\cdot u_b. \tag{4.3}
$$

Moving a build along $g_t$ raises the logit of $t$ at a constant rate; moving at right angles to it leaves $t$ unchanged. Items pointing the same way are valued by the same builds. Code: `ConceptSpace.item_coords`.

## 4.3 Worth space and the worth bound

The **worth vector** of a build lists its field worth $p_{bt}$ for every item. The **worth distance** between two builds is the weighted root mean square difference of their worth vectors, with the same weights as (4.1).

**Theorem 4.2 (worth bound).** For any two builds at concept distance $d$:

- their worth distance is at most $d/4$;
- for each item $t$, their worths differ by at most $\tfrac14\lVert g_t\rVert\,d$.

*Reason.* The logistic function never has slope above $\tfrac14$, so a logit difference of $\Delta$ is a worth difference of at most $\Delta/4$. Average this over items with the weights and apply Theorem 4.1 for the first statement. For the second, the logit difference of item $t$ is $g_t\cdot(u_b-u_{b'})$ by (4.3), which is at most $\lVert g_t\rVert\,d$. $\square$

**Interpretation.** The bound reads both ways. Builds close in the concept space value every item alike. Builds whose worth vectors differ by $\delta$ are at least $4\delta$ apart. This is what makes locality safe ([Section 5](05-locality.md)): evidence from a near neighbour is evidence about the same worth.

**Worked example.** Two builds at concept distance 0.4 have a weighted RMS worth difference of at most 0.1. If an item's direction has length $\lVert g_t\rVert=2$, its worth differs between them by at most $\tfrac14\cdot2\cdot0.4=0.2$.

Code: `ConceptSpace.bounds`, `item_bound_factor`.

## 4.4 The metric weights

Under equal weights, every item counts the same in (4.1). Most items do not tell archetypes apart: rare items sit near 0 everywhere and differ only by noise; universal items sit near 1 everywhere. Together they can make up most of the distance and drown the few items that separate two playstyles sharing an oath and stats. In the synthetic playstyle world, 300 rare items carried about 23% of the squared distance under equal weights, all of it noise.

tri weights each item by how much its worth varies between builds that can take it:

$$
\omega_t \propto \big(\text{availability share}_t\times\text{worth variance}_t\big)^{\gamma},\qquad\gamma=0.5\ (\texttt{metric\_power}). \tag{4.4}
$$

- **Availability share**: the fraction of builds that can take $t$.
- **Worth variance**: the variance of $p_{bt}$ over those builds. An item every build values the same has variance 0.
- **Tempering** $\gamma=\tfrac12$: weight by the standard deviation rather than the variance, so items that vary within one archetype (with stats, say) do not stretch it too much.
- **Floor**: after normalising to mean 1, weights are floored at 0.02 (`metric_floor`) and normalised again, so every item still counts a little.
- **Calibration**: a final constant rescales the weights so that the median distance to a build's 10th nearest neighbour equals its value under equal weights. Bandwidths and reach ([Section 5](05-locality.md)) therefore keep their meaning; only who is near whom changes.

**Why not Fisher information?** Weighting by $\bar p(1-\bar p)$ favours items taken by half of all builds, whoever takes them. A coin-flip item every archetype takes at 50% would dominate. Its worth variance is zero, which is correct: it tells no one apart.

**Interpretation.** The weights measure exactly what proximity is meant to resolve. An item at 0.9 in one playstyle and 0.05 in the other has a large weight; a two-group item with worth $q$ on a share $\pi$ of builds and $q'$ on the rest has variance $\pi(1-\pi)(q-q')^2$, so its weight grows with the gap.

**Redundancy.** Weights (4.4) treat items as independent. A bundle of $k$ items that are always taken together then counts $k$ times. The default metric, `worth_variance_dedup`, divides each weight by the item's redundancy:

$$
m_t=\sum_j \phi_{tj}^2\ \mathbb 1\big[\phi_{tj}\ge\tau\ \text{or}\ (t,j)\ \text{linked}\big],\qquad \omega_t'=\omega_t/m_t,\qquad \tau=0.9\ (\texttt{dedup\_tau}). \tag{4.4a}
$$

- $\phi_{tj}$ is the weighted phi coefficient of the two pick indicators.
- "Linked" means a catalog prerequisite or exclusion between the two items.
- $m_t\ge 1$, because $j=t$ is always included.

For $k$ identical items, $m=k$, so the bundle's total weight is that of one item. $\omega$ multiplies squared differences, which is why the divisor is $m_t$ and not $\sqrt{m_t}$.

**Why picks and not the field?** In a rank-$d$ field, items that load on the same direction have strongly correlated logits even when they are separate choices. Deduplicating on logit correlation merged whole archetype blocks and weakened the separation of archetypes. Pick correlation at $\tau=0.9$ merges only items that really are taken together. The redundancy does not use regions, so it is not circular.

The weights depend only on the field's probabilities and the picks, so `ConceptSpace.set_metric` can change them without retraining. Theorems 4.1 and 4.2 hold for every choice of positive weights. Runs saved before v2.3.1 load with equal weights.

## 4.5 The item worth space and pictures

The worth matrix can also be read by columns. Each item becomes a list of worths over a sample of builds (unavailable entries set to 0). Two items are close when the same builds value them. A **relative** version subtracts each item's mean first, so that items group by which builds value them more than usual rather than by overall popularity. Principal components of these columns are an exact classical multidimensional scaling of the item distances.

**Views.** A picture of either space projects it onto $k$ orthonormal axes (6 by default) and shows three at a time as a volume: axes 1 to 3, then 4 to 6. Each view reports how much of the true distance it keeps. An axis of the concept space is read through the item loadings $g_t$ projected on it: moving one unit along the axis changes item $t$'s logit by that loading. The items at the two ends, and the regions whose centres sit there, label the axis. These readings describe the linear map exactly; they say what an axis measures, not that it is an archetype.

Code: `plots.worth_space`, `plots.concept_axes`, `plots.worth_axes`, `plots.view_kept`.

## 4.6 Regions

Weighted $k$-means (with $k$-means++ starts) partitions the concept space.

**Choosing $k$.** The number of regions is chosen by stability, not fixed. For each $k$ in `region_grid` up to `n_regions` (24), two fits on disjoint random halves of a sample label the whole sample. The **adjusted Rand index** (ARI) of the two labellings is 1 when they agree and about 0 when they agree only by chance. tri takes the largest $k$ whose median ARI is at least 0.8 (`region_ari`). A larger $k$ is kept only if the data really supports it.

**Merging and mixed regions.**

- Two regions merge if their centres are closer than the bandwidth and their signatures overlap with Jaccard of at least 0.5. A signature is the region's top items plus its dominant context token. The closest pair is merged first, until no pair qualifies.
- A region with no signature item of lift at least 1.2 is a catch-all. It is named "mixed" and gets no locator label.

The remaining description holds for any $k$. Each build has a hard region (the nearest centre) and soft memberships from a Gaussian kernel of width 0.6 around each centre. Its **ambiguity** is the ratio of its second-largest to largest membership. Each region carries an average archetype, so tri can answer "this build, played as region $k$" by swapping in that archetype while keeping the build's stats and context.

Regions name and enumerate archetypes for display. They are a convenient partition of a continuous space, not a discovered truth, and rules never condition on them.

## 4.7 Locators

An administrator can name archetypes by pointing at builds. A **locator** is a build and an archetype name ("PvE Tank", "Silentheart ganker"). Names are compared case-insensitively with spaces collapsed.

**Design rule.** Locator labels never enter any loss, and locator builds are removed from the field's training rows (`Tri.held_out_ids`). Two consequences follow:

- relabelling the locators changes nothing about the field, the weights, the regions or any coordinate (checked by refitting with shuffled labels and comparing exactly);
- each locator's coordinate is the one the field would give a build it has never seen.

**Reading a build.** For a point $u$, each archetype gets a density: the average of $\exp(-\lVert u-u_j\rVert^2/2h^2)$ over its locators, with $h=0.6$. Dividing by the archetype's count means no archetype wins by having more locators. Normalising the densities gives the probability of each archetype at $u$. A build is **located** when its nearest locator is within reach 1.2 and the best archetype has probability at least 0.5. Otherwise it is reported with its nearest locator and best guess, never forced.

Its **activity** is 1 within reach and fades as a Gaussian of width $h$ beyond it.

Two archetypes a few picks apart overlap under this kernel; the **twin split** of [Section 6.6](06-accuracy.md) splits them by those picks where the map confuses them, and is the reading used everywhere below.

Regions whose centre is located as an archetype are displayed under that name. Readings, names and the barrier ([Section 5.4](05-locality.md)) change the moment a locator is added or removed; the field does not.

Code: `tri/reviews/locators.py`, `Tri.set_locators`, `Tri.archetype`.

## 4.8 Blended worth

The worth that inference ranks by, $q_t(b)$, combines the field with what nearby builds actually did. Take the 150 nearest reference builds (corpus and corrected reviewed builds). Give each neighbour $n$ a kernel weight $\kappa_n=w_n\exp(-d(b,n)^2/2\cdot0.6^2)$ and an evidence weight $\rho_{nt}$: 40 for an item a review of $n$ changed, 0.25 for a pre-patch build on a changed item, 0 for a conditional claim that does not hold in $b$, otherwise 1. Then

$$
q_t(b)=\frac{15\,\sigma(\ell^{\text{loo}}_{bt})+\sum_n\kappa_n\rho_{nt}\,x_{nt}}{15+\sum_n\kappa_n\rho_{nt}\,a_{nt}}. \tag{4.5}
$$

**Shrinkage form.** With $W_t$ the total neighbour weight that could take $t$, (4.5) is

$$
q_t(b)=\lambda_t\cdot\text{field worth}+(1-\lambda_t)\cdot\text{neighbour pick rate},\qquad\lambda_t=\frac{15}{15+W_t}.
$$

So $q$ is always between 0 and 1, and it equals the field's leave-one-out worth when no neighbour could take the item. The prior strength 15 (`prior_strength`) is how many neighbour-equivalents the field counts as.

**Worked example.** The field says 0.40. Neighbours that could take the item carry weight 45 and picked it at a rate of 0.80. Then $\lambda=15/60=0.25$ and $q=0.25\cdot0.40+0.75\cdot0.80=0.70$.

**Cross-archetype ring.** Where an archetype is thinly sampled, (4.5) leans on the field alone. A second, wider ring lets worth extrapolate a little across archetype borders. Let $\mathcal R(b)$ be the next $k_\times=600$ nearest corpus builds beyond the 150 (reviewed builds excluded), with weights $\kappa^\times_{nt}=\lambda_t\,\lambda_\times w_n\exp(-d(b,n)^2/2h_\times^2)$, $\lambda_\times=0.15$, $h_\times=1.5=2.5h$, where $\lambda_t=15/(15+W_t)$ is the shrinkage weight of the near neighbours: the ring speaks only where local support $W_t$ is thin. Then

$$
q_t(b)=\frac{15\,\sigma(\ell^{\text{loo}}_{bt})+\sum_n\kappa_n\rho_{nt}\,x_{nt}+\sum_{n\in\mathcal R}\kappa^\times_{nt} x_{nt}}{15+\sum_n\kappa_n\rho_{nt}\,a_{nt}+\sum_{n\in\mathcal R}\kappa^\times_{nt} a_{nt}}. \tag{4.5b}
$$

The ring is still a convex combination, so $q\in[0,1]$. On the rich synthetic world, expert agreement ([Section 11](11-experiments.md)) is 0.9842 with the ring and 0.9835 without; an unscaled ring ($\lambda_t=1$) gave 0.9789, which is why the ring is gated by $\lambda_t$. Each ring build counts at most $\lambda_\times=0.15$ of a near neighbour, and a reviewed build's 40-fold weight never enters it, so review weight stays local ([Section 5](05-locality.md)).

**Ties and screening (v3.8).** Neighbours at equal distance are taken in corpus row order, so the neighbour sets are functions of $b$ alone. With float32 coordinates the distances are screened in single precision: for $\tilde d_n^2$ the screened value and $\tilde d^2_{(k)}$ its $k$-th smallest ($k=150+k_\times$),
$$
|\tilde d_n^2-d_n^2|\le\varepsilon=8\,(D'+2)\,2^{-22}\max_m\lVert u_m\rVert\,\lVert u\rVert,
$$
so a row with $\tilde d_n^2>\tilde d^2_{(k)}+2\varepsilon$ is never among the $k$ nearest, and $d_n^2$ is recomputed in double precision on the remaining rows only. The rings and their weights are those of the exact distances.

Code: `worth.FieldWorth`, `FieldWorth._cross`, `WorthConfig.k_cross`, `cross_bandwidth`, `cross_weight`, `worth.smallest`, `RowStack.d2`. The default completion capacity is the weighted median slot count of the neighbours.

## 4.9 Sidegrades

Within one archetype, most picks are nearly always taken or nearly never. The **contested** picks, taken by between 15% and 85% of the region's builds that can take them, are where builds of one purpose differ. Sometimes they differ in a structured way: two clusters of picks of about equal worth, each taken by a substantial share of the archetype, and almost never together. Such a pair is a **sidegrade**.

**Definition 4.3 (sidegrade).** Let $R$ be a set of builds of one archetype, with weights $w_n$. A sidegrade is a pair of disjoint item sets $A,B$ of contested items such that, with the side score

$$
s_n=\frac{1}{|A|}\sum_{t\in A}x_{nt}-\frac{1}{|B|}\sum_{t\in B}x_{nt},
$$

and $R_A=\{n:s_n\ge\tfrac12\}$, $R_B=\{n:s_n\le-\tfrac12\}$ the builds committed to each side:

1. **commitment**: $w(R_A\cup R_B)\ge0.7\,w(R)$;
2. **balance**: $w(R_A)\ge0.2\,w(R)$ and $w(R_B)\ge0.2\,w(R)$;
3. **separation**: every $t\in A$ has pick rate on $R_A$ at least $0.35$ above its rate on $R_B$, and symmetrically for $B$;
4. **substitution**: the mean phi coefficient between items on opposite sides is at most $-0.2$, and within a side (two or more items) at least $0.2$.

Balance is what makes the two sides of equal worth in the sense of [Section 3](03-worth.md): worth is a local pick rate, and both sides are picked by a comparable share of the same archetype, so neither is a rare variant of the other. Substitution is needed because commitment alone is weak: independent items with rates near one half also produce one-sided builds by chance, but their phi coefficients are near 0.

**Where to look.** A strong sidegrade is a large source of variation, so the field may split one archetype into two neighbouring regions along it. tri therefore searches pairs of regions whose cores (items taken by at least 60% of a region's builds) overlap with Jaccard index at least 0.6, as one archetype with two variants, and then each region alone. The builds searched are those holding at least 80% of the shared core. They are chosen by the core and never by the items tested; selecting on the tested items would induce a spurious negative correlation between them (Berkson's paradox).

**Finding the sides.** On one random half of $R$, a weighted 2-means on the contested columns proposes two clusters; an item joins side $A$ or $B$ if its rate differs between the clusters by at least the gap. The held-out test of [Section 6.10](06-accuracy.md) then decides.

**Use.** A sidegrade changes inference in two places ([Section 7.3](07-strictness.md)): worth never adds an item of the side the answer holds fewer of, so an answer is never a blend of two complete strategies; and an answer committed to one side lists the other as an **offer**: the picks to remove, the feasible picks to add, and both shares. Rules and reviews still require what they require.

**Locality.** Sidegrades are mined from the corpus inside one archetype and apply only to builds that can take items of both sides. They do not move the field, the worth or any rule, so no other part of the concept space changes. Reviews do not create or remove sidegrades; a review that adds a side item acts as usual ([Section 7.3](07-strictness.md)).

Code: `sidegrades.mine`, `SidegradeConfig`, `PassEngine._sides`, `PassEngine._offers`.

## 4.10 Note worth and flat values

Two inputs adjust the blended worth $q$ after (4.5b), both instant and both reversible.

**Note worth.** A review note that names components, an indicator note ($\pi$ with items $C$) or a core note, raises the worth of those components near the reviewed build $u_r$; an avoid note lowers it. With $K(u,u_r)=\exp(-\lVert u-u_r\rVert^2/2h^2)$ for $\lVert u-u_r\rVert\le R$ (else 0) and $p_\pi(u)$ the reading of the noted archetype (1 for core and avoid notes, which use the reviewed build's own reading),

$$
\tilde q_t=\operatorname{clip}_{[0,1]}\big(q_t\pm\eta\,K(u,u_r)\,p_\pi(u)\big),\qquad \eta=0.35, \tag{4.6}
$$

applied only when the weight is at least $0.02$ (`note_floor`). Components so raised are reported as **indicative** picks.

**Flat values.** An admin may set a flat offset $\delta_t\in[-0.5,0.5]$ for an item, added on every build:

$$
\hat q_t=\operatorname{clip}_{[0,1]}(\tilde q_t+\delta_t). \tag{4.7}
$$

**Proposition 4.3 (note worth is local).** If $\lVert u-u_r\rVert>R$ for every note, $\tilde q=q$ exactly; a build that never holds a component cannot be moved by its note's indicator lift ([Section 6.7](06-accuracy.md)). Flat values are global by design and bounded by $0.5$.

*Reason.* $K=0$ outside the reach, and (6.8) requires $C\subseteq x$. $\square$

Code: `PassEngine._notes_q`, `InferConfig.note_worth`, `note_floor`, `value_cap`; `Tri.set_values`.
