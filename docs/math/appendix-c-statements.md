# Appendix C. Symbolic statements

Every rule and fact the model states is written in one notation (`tri/rules/symbolic.py`). Each written form is shorthand for a precise statement below. $t,s$ are items, $A,B$ archetypes, $C$ a conjunction of conditions, $\hat p$ a held-out rate, $w$ worth ([Section 3](03-worth.md)).

**Conditions.** `Strength ≥ 40` is $\text{hi}_{\text{Strength}}\ge40$; `oath = X` and `oath ≠ X` test a context token (also `weapon = Heavy`, from a weapons note); `s` and `¬s` test whether $s$ is picked; `⊤` is the empty conjunction (always true); `∧` is conjunction.

v3.0 adds `origin = X`; `A = chime` ($A_b=\text{chime}$, the locators' family at $b$; [Section 6.12](06-accuracy.md)); `|G(c)| ≥ n` and `|D(k)| < n` (picks of category $c$, or of the derivation tree of core talent $k$, other than the rule's item). The same text is the input language for authored rules ([Section 6.13](06-accuracy.md)): `[C] A = chime ∧ |D(Flame Leap)| ≥ 2 ⇒ Flame Wisp`, or in ASCII `[C] A = chime & |D(Flame Leap)| >= 2 => Flame Wisp`. Each rule prints its id `#3fa2c1` and its selection status (stands, refined, overruled, disputed, suppressed, authored).

**Corpus rules** ([Section 6](06-accuracy.md)). With $\mathcal C$ the builds where $C$ holds and $t$ is available,

$$
\texttt{[F|C] } C \Rightarrow t \quad\text{means}\quad \Pr(t\mid\mathcal C)\ \ge\ \ell \ \text{(certified lower bound, written after } \ge\text{)},
$$

$$
\texttt{[X] } C \Rightarrow \neg t \quad\text{means}\quad \Pr(t\mid\mathcal C)\ \le\ u \ \text{(certified upper bound)},
$$

with `P` the observed rate, `n` the weighted support and `else` the rate when the pivotal condition is flipped. The tag is the pass: F foundational, C central, X exclusion.

**Review rules** ([Section 5.6](05-locality.md)). `[R status]` then one of

| written | statement |
|---|---|
| `o ⇒ ¬t` | $t$ is redundant given $o$ |
| `t ⊥ o` | $t$ works against $o$ |
| `⊤ ⇒ ¬t` | $t$ does not work |
| `o ≻ t \| space short` | prefer $o$ over $t$ when slots or the cap are short |
| `A ⇒ t`, `A ⇒ ¬t` | $t$ is necessary for, or does not fit, the build's archetype |
| `gank ⇒ ¬t` | $t$ does not belong on the named family (a strong SM removal); applies only to builds located there |
| `⊤ ⇒ t` | $t$ is generally good (foundational) |
| `A ∧ o ⇒ t` | $t$ synergises with $o$: necessary for $A$ where $o$ is picked |
| `A ∧ ¬(c₁ ∧ c₂) ⇒ ¬t ≡ t ⇒ c₁ ∧ c₂ \| space` | $t$ does not work without $c_1,c_2$; with room, keeping $t$ means adding them (5.3) |
| `t ∼ o` | $t$ is a sidegrade to $o$ |

**Notes.** `ι: C ⊢ A` is an indicator (6.7): the components $C$ lift the odds of $A$ by $e^{\lambda}$ within reach. `W = {Heavy}` is a weapons note ([Section 5.5](05-locality.md), item 10).

`| c` restricts the rule to builds where $c$ holds; `+s −c` are support and counter weight; `lb` the lower bound on precision.

**Facts** ([Section 12.2](12-reading-the-field.md)). Rates are held-out; `p` is the held-out p-value (shown as an upper power of ten when tiny).

| written | statement |
|---|---|
| `P(t \| A) a, P(t \| ¬A) b ×L` | $\hat p(t\mid A)=a$, $\hat p(t\mid\neg A)=b$ among builds that could take $t$; lift $L$ |
| `A ↔ B: t + s = σ of d²` | $t,s$ carry share $\sigma$ of $d(\bar u_A,\bar u_B)^2$ in (12.2) |
| `A: w(t) ≥ ½ ⇔ S ≥ k (req r)` | in $A$, worth of $t$ reaches $\tfrac12$ only from stat $S=k>r$ |
| `A: hole t  P(t) a, reviews −t k/n` | hole: pretrain rate $a$, reviewers removed $t$ in $k$ of $n$ reviews (gem: `+t`, added) |
| `A: P(t) a, reviews +t k/n` | gem: corpus rate $a$, reviewers added $t$ in $k$ of $n$ |
| `A: s ⇒ Δw(t) δ` | in $A$, removing $s$ changes the worth of $t$ by $-\delta$ |
| `∀A ≠ X: w(t) ≥ 0.9; w(t \| X) v` | every archetype but $X$ values $t$ at 0.9 or more; $X$ at $v$ |
| `A @P: P(t) a → b` | after patch $P$, the rate of $t$ in $A$ moved from $a$ to $b$ |
