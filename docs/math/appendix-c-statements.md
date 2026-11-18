# Appendix C. Symbolic statements

Every rule and fact the model states is written in one notation (`tri/rules/symbolic.py`). Each written form is shorthand for a precise statement below. $t,s$ are items, $A,B$ archetypes, $C$ a conjunction of conditions, $\hat p$ a held-out rate, $w$ worth ([Section 3](03-worth.md)).

**Conditions.** `Strength ≥ 40` is $\text{hi}_{\text{Strength}}\ge40$; `oath = X` and `oath ≠ X` test a context token (also `weapon = Heavy`, the class, and `weapon = Railblade`, true iff $\text{Railblade}\in W(b)$, the equipped weapons and weapon notes); `s` and `¬s` test whether $s$ is picked; `⊤` is the empty conjunction (always true); `∧` is conjunction.

v3.0 adds `origin = X`; `A = chime` ($A_b=\text{chime}$, the locators' family at $b$; [Section 6.12](06-accuracy.md)); `|G(c)| ≥ n` and `|D(k)| < n` (picks of category $c$, or of the derivation tree of core talent $k$, other than the rule's item). The same text is the input language for authored rules ([Section 6.13](06-accuracy.md)): `[C] A = chime ∧ |D(Flame Leap)| ≥ 2 ⇒ Flame Wisp`, or in ASCII `[C] A = chime & |D(Flame Leap)| >= 2 => Flame Wisp`. Each rule prints its id `#3fa2c1` and its selection status (stands, refined, overruled, disputed, suppressed, authored).

**Corpus rules** ([Section 6](06-accuracy.md)). With $\mathcal C$ the builds where $C$ holds and $t$ is available,

$$
\texttt{[F|C] } C \Rightarrow t \quad\text{means}\quad \Pr(t\mid\mathcal C)\ \ge\ \ell \ \text{(certified lower bound, written after } \ge\text{)},
$$

$$
\texttt{[X] } C \Rightarrow \neg t \quad\text{means}\quad \Pr(t\mid\mathcal C)\ \le\ u \ \text{(certified upper bound)},
$$

with `P` the observed rate, `n` the weighted support and `else` the rate when the pivotal condition is flipped. The tag is the pass: F foundational, C central, X exclusion.

**Pruning** (`tri/rules/triviality.py`, `prune`). Let $M=337$ be the most points a build has. For an item $t$ let $D_s(t)$ be its least demand on stat $s$: the maximum of $t$'s single-stat requirements over $t$ and its all-of prerequisites, transitively. A points literal $v_S\ \mathrm{op}\ n$ tests a stat ($S=\{s\}$) or an aggregate over member stats $S$ (sum or max). Its shrined requirement outside $S$ and the most $S$ can hold beside $t$ are

$$
\mathrm{sh}(t,S)=\sum_{s\notin S}\max\bigl(0,\,D_s(t)-25\bigr),\qquad \mathrm{top}(t,S)=M-\mathrm{sh}(t,S),
$$

so $\mathrm{sh}(\text{Ghost},\varnothing)=40-25=15$ and $\mathrm{sh}(\text{All Above, Gods Below},\varnothing)=(100-25)+(100-25)=150$. With $f(t,S)=\sum_{s\in S}D_s(t)$ for a sum, $\max_{s\in S}D_s(t)$ for a max or a stat,

$$
\begin{aligned}
n>\mathrm{top}(t,S)&:\quad v_S\ge n\equiv\bot,\qquad v_S<n\equiv\top,\\
n\le f(t,S)&:\quad v_S\ge n\equiv\top,\qquad v_S<n\equiv\bot,
\end{aligned}
$$

on every build that can take $t$. Mined and review rules (authored rules are kept as written) are then rewritten to a fixed set $R'$:

$$
\begin{aligned}
&\text{(1)}\quad \bot\in C\ \Rightarrow\ \text{delete } (C\Rightarrow\cdot);\qquad C\setminus\{\top\}\ \text{replaces}\ C\ \ (\varnothing\ \text{reads}\ \top),\\
&\text{(2)}\quad (\top\Rightarrow t)\in R'\ \Rightarrow\ \text{delete every other } (C\Rightarrow t);\ \ (C\Rightarrow\neg t)\ \text{stays},\\
&\text{(3)}\quad \text{equal } (\text{pass},C,t,\text{effect})\ \text{collapse to one};\ \ [F]\,\top\Rightarrow t\ \text{absorbs}\ [C]\,\top\Rightarrow t,\\
&\text{(4)}\quad C'\subsetneq_{\Rightarrow} C,\ (C'\Rightarrow e)\in R'\ \text{same pass}\ \Rightarrow\ \text{delete } (C\Rightarrow e),
\end{aligned}
$$

where $C'\subsetneq_{\Rightarrow}C$ means every build meeting $C$ meets $C'$ and not conversely. Rules a group baseline makes dead or implied are deleted too. Since $270+150>337$: `[F] attunement points < 270 ⇒ All Above, Gods Below` becomes `[F] ⊤ ⇒ All Above, Gods Below` and `[X] attunement points ≥ 270 ⇒ ¬All Above, Gods Below` is deleted. Pruning runs after mining, on load (no refit) and on the selection rule set.

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
