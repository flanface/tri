# 5. Locality

## 5.1 Definition

> **Locality.** A piece of evidence about one build (a review, a neighbour's pick, a veto, a locator) influences another build only through their similarity in the concept space. Its influence is largest on the build it was made on, decreases smoothly with concept distance, is negligible beyond a fixed reach, and does not cross the border between two archetype families: borders are chasms, not slopes. Evidence with a stated condition applies only where the condition holds.

## 5.2 Interpretation

Locality is what lets tri weigh reviews heavily without the weight **bleeding** into unrelated builds. A review is ground truth for the build it was written on and a strong hint for builds of the same archetype; for a build of another archetype it is no evidence at all. Without locality, a review of a tank saying "remove this luxury talent" would push every build to remove it.

Locality is only meaningful if "near" means "values items alike". That is what Theorem 4.2 guarantees, and what the weights of [Section 4.4](04-concept.md) sharpen: near builds have nearly the same worth for every item, so their evidence is evidence about the same quantity, not a vote from another archetype.

In data science terms, tri is a **kernel smoother** over the concept space with a strong local prior (the field), and reviews are high-weight observations whose kernel is wide enough to cover an archetype and no wider.

## 5.3 Mechanisms: kernels and reach

There are two kernels, both Gaussian in concept distance.

**Worth kernel** ([Section 4.8](04-concept.md)), width $h=0.6$, over the 150 nearest neighbours. It is narrow: it averages what very similar builds did.

| distance | 0.26 | 0.58 | 1.0 | 1.2 | 1.6 |
|---|---|---|---|---|---|
| worth kernel | 0.91 | 0.63 | 0.25 | 0.14 | 0.03 |

**Review kernel**, width $h_L$ set so that similarity is $s_{\min}=0.80$ at the reach distance 1.2:

$$
\text{sim}(b,b')=\exp\!\Big(-\frac{d(b,b')^2}{2h_L^2}\Big),\qquad h_L^2=\frac{1.2^2}{-2\ln 0.80},\qquad h_L\approx1.80. \tag{5.1}
$$

| distance | 0.26 | 0.58 | 1.0 | 1.2 | 2.0 | 2.7 |
|---|---|---|---|---|---|---|
| review similarity | 0.99 | 0.95 | 0.86 | 0.80 | 0.54 | 0.32 |

Three bands follow from it (`LedgerConfig`):

- **same build**, similarity at least 0.99 (distance up to about 0.26): the review is about this build;
- **full strength**, at least 0.95 (distance up to about 0.58): the review applies at full weight;
- **within reach**, at least 0.80 (distance up to 1.2): the review applies, scaled by its similarity. Beyond reach it does not apply at all.

A claim's **strength** for a build is its weight times a factor: 1 in the full-strength band, the similarity itself between 0.80 and 0.95. [Section 8.3](08-convergence.md) says when a strength is enough to bind.

## 5.4 Mechanism: the chasm

The review kernel cannot know where one archetype family ends and the next begins. Locators can. For two builds with family readings $p_u$ and $p_v$ ([Section 4.7](04-concept.md)), the **total variation** $\mathrm{TV}(p_u,p_v)=\tfrac12\sum_\pi|p_u(\pi)-p_v(\pi)|$ is between 0 (same reading) and 1 (opposite readings). Write $a(\cdot)$ for activity (1 within reach of a locator, decaying beyond), and say a build is **located in** $\pi$ when it is within reach and $p(\pi)\ge0.5$. The **chasm** is

$$
\chi(u,v)=\begin{cases}1-\beta, & u\ \text{located in}\ \pi,\ v\ \text{located in}\ \pi'\neq\pi,\\[4pt]
1-\beta\,a(u)\,a(v)\,\bigl(1-(1-\mathrm{TV}(p_u,p_v))^{\gamma}\bigr), & \text{otherwise,}\end{cases}
\qquad \beta=1,\ \gamma=6. \tag{5.2}
$$

($\beta$ = `barrier`, $\gamma$ = `chasm_power`, the first case = `chasm_hard`.) With $\beta=1$, two builds located in different families do not influence each other at all. The power makes the soft case steep: a reading difference of $\mathrm{TV}=0.2$ keeps $(0.8)^6=26\%$ of the influence, where the linear barrier of v2.18, $1-\mathrm{TV}$, kept 80%.

$\chi$ multiplies every channel through which one build's evidence reaches another:

1. review similarity in the ledger, and the review boost of a reviewed neighbour in (4.5);
2. the kernel weight $k_j$ of every corpus neighbour in (4.5) (`chasm_corpus`);
3. the cross ring (4.5b), which therefore extrapolates across variants of one family, never across families;
4. note worth ([Section 4.10](04-concept.md)): $w=K(u,u_r)\,p(a\mid u)\,\chi(u,u_r)$;
5. mined central take rules. On a reference sample, the **home** of a rule $r$ is
$$
H(r)=\bigl\{\pi:\ W_\pi(r)\ge m_H\ \text{and}\ \textstyle\sum_{j\in\pi,\,r\ \text{covers}\ j} w_j\,x_{jt}\ \ge\ \theta_{\text{take}}\,W_\pi(r)\bigr\}, \tag{5.2a}
$$
with $W_\pi(r)$ the weight of located builds of $\pi$ the rule covers and $m_H=5$ (`chasm_rule_min`). A rule does not fire on a build located in $\pi\notin H(r)$; local worth still decides the item there. Foundational rules (spread across the corpus by construction) and category rules are not gated.

**Properties.**

- $\chi\in[1-\beta,1]$. It can only weaken evidence, never strengthen it.
- $\chi(u,u)=1$, and $\chi\to1$ far from every locator: without locators nothing changes.
- Inside one family $\chi=1$ whenever readings agree; variants of a family ("Gank / Frost Mage") share one family reading, so they do not cut each other off.
- Unlocated builds keep a smooth (but steep) transition, so nothing is partitioned where the locators have nothing to say.

**Proposition 5.1 (no bleed across families).** Let $u$ be located in $\pi$ and let $E$ be any evidence (a review, a note, a corpus neighbour, a ring row) attached to a build $v$ located in $\pi'\neq\pi$. With $\beta=1$, $E$ contributes nothing to the worth, claims or rules at $u$.

*Proof.* Each channel above multiplies $E$'s weight at $u$ by $\chi(u,v)=1-\beta=0$; a rule whose home excludes $\pi$ does not fire at $u$. Family tells ([Section 8](08-convergence.md).5a) are gated by $p(\pi'\mid u)$, which is below $0.5$ and, by the floor, contributes only where it reaches `note_floor`. $\square$

**Why total variation, and why a power?** A first version used the probability that two builds have different archetypes, $1-\sum_\pi p_u(\pi)p_v(\pi)$; it is large whenever readings are uncertain, and it cut same-archetype pairs. Total variation is zero when readings agree. The linear form $1-\mathrm{TV}$ let a fifth of the reading's difference through as four fifths of the influence; the power $\gamma$ turns the slope into a chasm while staying continuous at $\mathrm{TV}=0$.

## 5.5 Mechanism: scoped claims

Some evidence carries its own scope. Each rule below narrows where a claim applies.

1. **Conditional claims.** "Remove X, redundant with Y" or "works against Y" only applies to builds that contain all of Y. For other builds it is not returned, does not conflict with their reviews, and adds no evidence to their worth (its weight in (4.5) is 0). So "remove X: works against Y", written on a build with both, says nothing about a build with X alone, however similar.
2. **Rule boundaries.** A claim about item X does not transfer to a build on the other side of one of X's certified rule conditions (a stat threshold or an oath, for example) from the reviewed build. A review of a 50-Agility build does not decide X for a 20-Agility build if a certified rule for X splits them (`Rulebook.separates`).
3. **Generally good (GG).** A GG add has global scope, because the reviewer says it is foundational for everyone. It is still kept local where it matters:
   - a local claim on the same talent (from a review within reach) takes precedence;
   - only a strong GG binds outside its own neighbourhood; a weak one only nudges worth;
   - in newest-wins conflicts it uses its real similarity, so a newer removal on another archetype stands alongside it while a newer removal on the same archetype replaces it;
   GG speaks about one talent, never about the build's stats or other picks.
4. **Vetoes.** Removed in v3.5 together with audits.
5. **Broken.** A removal with reason *broken* is the only removal that applies to every build regardless of distance, because the reviewer says the item itself is broken. A newer endorsement of the item overrides it (for example after a fix).
6. **Synergy (SY).** "Add $t$: synergises with $Y$" is a present condition like RW and WA: it holds on $b'$ only if $Y\subseteq b'$. A similar build without the partner learns nothing from it.
7. **Needs (NW).** "Remove $X$: does not work without $C$" is an absent condition: it holds on $b'$ only if $C\not\subseteq b'$, and worth abstains (weight 0 in (4.5)) on builds that have all of $C$. The claim is a disjunction, and tri reads it as one:
$$
\text{NW}(X,C)\ \text{on}\ b'\ \Longleftrightarrow\ \bigl(X\cap b''=\varnothing\bigr)\ \lor\ \bigl(C\subseteq b''\bigr)\quad\text{for the answer }b''. \tag{5.3}
$$
The second branch is taken when $C\setminus b'$ fits without displacing a pick (slot counts, the talent cap, mantra slots, availability and prerequisites); otherwise the first. The reviewer chose removal because the build was full; the equivalent add is what the same reviewer would accept given room.
8. **Relative judgements (LN, LL, LM).** "Less necessary", "worth less as a luxury" and "worth more as a luxury" compare an item with the rest of one build. They bind that build only; on a similar build they only nudge worth.
9. **Transfers.** A strong item-specific add or remove from another build binds only at similarity $\ge s_{\text{bind}}=0.95$ (`bind_sim`); below that it nudges. A strong GG add binds at any similarity. The old audits measured why: binding transfers were upheld 5 times in 47 ([Section 11.6](11-experiments.md)).
10. **Standing corrections.** Removed in v3.5 together with audits. Audits held verdicts without removal reasons, so the corrections built from them carried almost no information and overrode real reviews. Equation (5.4) is retired.
11. **Echoes.** Let $T$ be the set of reviewed builds that tri made, identified by `meta.made_by = "tri"` or a name ending "(tri completion)". On $b\in T$, an unmarked item $x$ means only that the reviewer let tri's pick stand. That is weaker than an endorsement, and counting it in full feeds tri's output back to tri as if a reviewer had said it. An echo keep therefore has weight
$$
w_{\text{echo}}(x\mid b)=\varepsilon\cdot w_{\text{keep}}(x\mid b),\qquad \varepsilon=0.25\ (\texttt{echo\_keep}), \tag{5.5}
$$
and three rules apply to it. (i) An echo is used only when no explicit claim exists: $\text{claim}(x\mid b')=\text{explicit}(x\mid b')$ if defined, else $\text{echo}(x\mid b')$. (ii) An echo never supersedes an explicit claim under newest-wins. (iii) An echo is never counter-evidence to a rule, so a single review of a completion cannot dispute OFF_ARCHETYPE. Since $\varepsilon<\tfrac12$, an echo never binds. Explicit adds, removes and iffies on $b\in T$ count in full.
12. **Sidegrades (SG).** A weak sidegrade has the usual reach (similar builds of this archetype); a strong one has global scope like BR and GG, but only says something where $t$ or $y$ is picked. Either way it is symmetric for worth: neither item is added next to the other, and neither replaces the other.
13. **Weapon classes.** If review $r$ notes classes $W_r$ and a query build $b'$ records classes $W_{b'}$ (`meta.weapon`), claims of $r$ reach $b'$ only if $W_r\cap W_{b'}\neq\varnothing$ or one of them is empty, or $b'$ is the reviewed build. Weapon classes also enter rule features as `weapon:Heavy`, so a disputed review rule can be refined by class.

14. **Named weapons.** A review may note weapons by name, $N_r$ (`# weapon: Silver Dagger, Railblade`); the names join the build's weapons $W(b)$ and their classes. A claim may relate to a weapon token $\omega=\texttt{weapon:}n$ (`> Silver Dagger`). It is then conditional on that weapon:
$$
\text{claim}(x,\omega)\ \text{holds on}\ b'\ \Longleftrightarrow\ n\in W(b'), \tag{5.5'}
$$
and worth abstains on it (weight 0 in (4.5)), as for RW and WA. NW and SG cannot relate to a weapon.

15. **Stated families (v3.6).** A review whose notes state its build's archetype (`# is gank`) speaks for that family. Its claims do not reach a query read in another family, except generally-good claims and claims on the same build:
$$
\text{fam}(r)\neq\varnothing,\ \text{fam}(r)\neq A(b')\ \Rightarrow\ \text{claims}(r)\ \text{do not reach}\ b'. \tag{5.5''}
$$
In the same way, a NECESSARY review rule is scoped to the reviewed build's family, $A=\text{fam}(r)\Rightarrow t$, and is not induced when the family is unknown. Before, four reviews of builds noted as gank made Jolting Current necessary everywhere, and their claims reached chime builds through proximity.

Rules themselves are local in a different sense: their conditions are written only in terms of observable stats, context tokens, other picks, group counts and the archetype the locators read ([Section 6.3](06-accuracy.md)), never fitted coordinates. A rule applies to exactly the builds that meet its condition. Review evidence on a rule ([Section 6.12](06-accuracy.md)) is local the same way: a review counts for a rule only if the rule applies to the reviewed build, and a refinement adds the literal that separates it.

## 5.6 How locality is measured

- **Reach within and across archetypes.** The share of same-archetype pairs and of other-archetype pairs at full review strength. Good locality is high within and low across.
- **Different-label pairs inside the full-strength band.** In the playstyle world this fell from 2.0% with equal weights to 0.9% with the weighted metric.
- **Bound check.** `plots.distance_check` plots worth distance against concept distance with the line $d/4$; every point lies under it.
- **Meta shift.** After one review on one archetype, how many unreviewed builds of that archetype change, and how many of other archetypes do ([Section 11](11-experiments.md)).

## 5.7 Worked example

A reviewer strongly removes talent X from build A, reason *no synergy*. Three other builds:

| build | distance from A | similarity | chasm $\chi$ | effective | effect on X |
|---|---|---|---|---|---|
| B, same archetype | 0.4 | 0.98 | 1.0 | 0.98 | full strength, binds |
| C, same archetype, looser | 1.0 | 0.86 | 1.0 | 0.86 | applies at strength 0.86 and nudges (binding needs 0.95, [Section 8.3](08-convergence.md)) |
| D, other family | 1.1 | 0.83 | 0 | 0 | across the chasm: no effect |
| E, far | 2.0 | 0.54 | 1.0 | 0.54 | out of reach; only the corpus and rules decide |

Build D shows why the chasm exists: by distance alone it is within reach, but the locators read it as another family, so the review, and also A's corpus neighbours, say nothing about it.

## 5.8 Mechanism: pooled reach

Kernel reach (5.3) and the bind threshold $s_{\text{bind}}$ make most review evidence act on few builds: a single claim from a build at similarity below 0.95 only nudges. v3.3 pools the reviews of many builds, as the v1.0 ledger did within reach, so that agreement between reviewers counts where one review alone would not. The chasm still holds.

**Pool.** For a query $b$ and a reviewed build $b'\neq b$ let
$$
\varpi(b,b')=\max\Bigl(\ \varpi_{\text{fam}}(b,b'),\ \operatorname{sim}(b,b')\cdot[\operatorname{sim}(b,b')\ge s_{\text{pool}}]\ \Bigr), \tag{5.6}
$$
$$
\varpi_{\text{fam}}(b,b')=\begin{cases}1-\operatorname{TV}(\pi_b,\pi_{b'}), & b,b'\ \text{located in one family},\\ \rho, & \text{neither located},\ \operatorname{reg}(b)=\operatorname{reg}(b'),\\ 0, & \text{otherwise},\end{cases}
$$
where $\pi$ is the locators' family posterior, $\operatorname{reg}$ the concept region, $s_{\text{pool}}=0.8$ (`pool_sim`) and $\rho=0.6$ (`fam_region`). The pool is $P(b)=\{b' : \varpi(b,b')\ge\varpi_{\min}=0.5\}$ (`fam_overlap`), less builds whose weapon classes are disjoint from those of $b$. A located build and an unlocated one never pool through $\varpi_{\text{fam}}$, so the pool never crosses a located border.

**Votes.** Each $b'\in P(b)$ gives at most one vote per item $x$: its newest active claim (echoes, iffies, rule verdicts and claims whose condition fails on $b$ left out), with polarity $\pm1$ and weight $w\cdot\varpi\cdot(\kappa_{\text{keep}}\ \text{for keeps})$, $\kappa_{\text{keep}}=0.5$. For $x\notin b$ keeps are dropped: a neighbour keeping what it had is no add. A full review whose returned build lacks $x$ and says nothing about $x$ casts an **absence** vote, for every $x$ some reviewed build holds or a review adds (`pool_absent`).

**Label.** With $n_\pm$ voters and weighted shares $\sigma_\pm$,
$$
\ell(x)=\begin{cases}\text{remove}, & n_-\ge n_{\min},\ \sigma_-\ge\sigma,\\ \text{keep}\ (x\in b)\ /\ \text{add}\ (x\notin b), & n_+\ge n_{\min},\ \sigma_+\ge\sigma,\\ \text{absent}, & x\notin b,\ n_{\text{abs}}\ge n_{\min},\ n_+=0,\end{cases}\qquad w(x)=\frac{n}{n+\kappa}\,\sigma, \tag{5.7}
$$
with $n_{\min}=2$, $\sigma=0.75$, $\kappa=1.5$ (`fam_min`, `fam_share`, `fam_kappa`).

**Binding.** A pooled claim binds iff $w\ge\tfrac12$, no strong claim opposes it, and
$$
\#\{\text{strong, item-level supporting claims}\}\ \ge\ n_{\text{bind}}=3\qquad\text{or}\qquad n\ge n_{\text{cons}}=4,\ \sigma\ge\sigma_{\text{cons}}=0.9 \tag{5.8}
$$
(`fam_bind`, `fam_consensus`, `fam_consensus_share`; a keep binds only through the second branch, an absence through $n_{\text{abs}}\ge n_{\text{cons}}$). Pooled claims fill items no local claim covers; a binding pooled claim also replaces a local claim that would only nudge (`pool_override`). A local claim that binds, a global claim and a standing correction take precedence.

**Effect on worth swaps.** A binding absence keeps $x$ out of worth swaps; slot filling and requirements are unaffected. With `swap_guard`, any claim near $b$, binding or not, guards a swap:
$$
\text{swap}(y\to x)\ \text{allowed}\ \Longrightarrow\ \ell(y)\neq\text{keep}\ \land\ \ell(x)\notin\{\text{remove},\text{absent}\}. \tag{5.9}
$$

**Measured.** On a time split of the debug reviews (oldest 70% plus audits as memory, newest 30% held out, [Section 11](11-experiments.md)), macro agreement rose from 0.190 to 0.208 and over-editing fell from 0.127 to 0.102; with pooling off, the swap guard alone gives 0.205. Before (5.9), worth swaps were right 4 times in 65.

## 5.9 Limits

- Locality is only as good as the geometry. Two archetypes that the field places together cannot be separated by kernels, and the chasm is only as sharp as the locators' readings of them. Unlocated builds keep the steep but continuous transition of (5.2).
- Builds of another playstyle that share an archetype's oath and stats can sit inside the reach. The weighting reorders neighbours and empties the full-strength band of most of them, but does not move the reach boundary.
- Global claims (broken, strong GG) deliberately ignore distance. Their scope is the reviewer's responsibility.
- Review claims are not rules. A strong scenario mismatch removal acts through the claims it makes on nearby builds, not through a droppable rule; it is undone by purging the marks (`strong_sm` on any item), not by `drop_rule`.
- Hardcoded rules (2.8f) are not learned and do not move with reviews. A wrong weapon rule is fixed in code.
- Purging an item's marks leaves the marks on its prerequisites. A removal of Knight's Rally strands Turtle Shell unless the weapon rule protects it (2.8c) or the purge covers prerequisites.
