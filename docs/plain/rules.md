# Rules and facts

tri states two kinds of short, checkable claims.

- **Rules** say how tri builds its answers. Some are game limits it never breaks: the talent cap, the mantra slots of each type, filling a build up to its cap before suggesting swaps, requirements met before the shrine or at the end, oath, murmur and origin locks, weapon needs, and stat benchmarks. Others are mined from the builds, for example "with these stats, always take this talent", and only kept if they still hold on builds they were not learned from. The rest come from patterns in the reviews.

Since v3.0, foundational and central talents are chosen only by these rules, never by worth. A rule looks at the build's context: its stats, picks, oath, murmur, origin, weapon class, how many talents it has from a known family (a game category or a talent tree, like "two or more Flamecharm talents"), and its archetype (gank, chime, trackstar, support, boss PvE, wave PvE), read from where the build sits in the concept space. Rules are short (at most three conditions), every condition has to matter on builds the rule was not found on, and the rule has to hold a third time on yet another set of builds. Stats are only used if the talent's requirements, or its family's, mention them. Reviews then check each rule: it stands, gets one extra condition that separates the builds reviewers disagreed on, or is overruled by the newest review. Administrators can write rules themselves with `/rule add` in the same symbols. Worth only ranks luxury talents.

  Since v3.0.1, some rules also come straight from the reviews. If reviewers keep a talent on almost every build they review, across several archetypes, it becomes `⊤ ⇒ t` (always take it). If they keep adding it on one archetype, it becomes `A = gank ⇒ t`. If they keep removing it there, it becomes `A = chime ⇒ ¬t`. The newest reviews decide. A review's `# is chime` note sets the archetype of the build it reviews, even where the map cannot place the build. A rule like "a talent that needs x ⇒ x" only restates the game's prerequisites, so tri no longer makes it.

  Since v3.0.2 these review rules are easier to form. A build that could never take a talent (its stats are too low) no longer counts against it, so a talent like Everchanging Aegis, kept on nearly every build that can reach it, is "always take it". Three in four reviewed builds is enough (it was more than four in five). And if reviewers simply leave a talent in place on most builds of one archetype, without having to add it, it becomes `A = chime ⇒ t` for that archetype. A rule also decides only where it applies: a talent that is central for chime is not shut out of a gank build; there the model ranks it by worth like any other optional talent.
- **Facts** describe the map and the meta, for example "this talent is the mark of this archetype", or "reviewers keep removing this popular pick". A fact is found on one half of the builds and checked on the other half. Across all facts, at most about 5% are allowed to be flukes. Facts that stop being true are retired and listed in the digest.

Both are written in symbols, so they are short and you can compare them at a glance.

**Trivial statements are left out.** A statement the game data already explains says nothing new, so tri drops it:

| statement | left out when |
|---|---|
| x needs stat ≥ k | k is just x's requirement |
| x is common in archetype A | x is common everywhere |
| x and y are not taken together | x and y are mutually exclusive |
| x follows y | y is x's prerequisite |
| y ⇒ x, or \|D(x)\| ≥ n ⇒ x | y needs x |
| archetypes A and B differ in x | x is locked by oath, murmur or origin, and A and B have different ones |

**Holes and gems.** A hole is a pick almost every build of an archetype takes but reviewers almost always remove (Endurance Runner on a low-Agility build). A gem is the reverse: rarely taken, almost always added by reviewers.

**Weapons.** Some talents depend on the weapon: Speed Demon is pointless when the weapon already bleeds, and Turtle Shell needs a two-handed weapon. tri reads the build's weapons. A build that carries several is only locked out when none of them allows the talent; Speed Demon with one bleeding and one plain weapon is kept but counts half as much.

  Since v3.2, three weapon rules are fixed from the game data, because the builds rarely record a weapon. They hold even over reviews.

  | rule | read it as |
  |---|---|
  | `bleed(w) ⇒ ¬Speed Demon` | a weapon that already bleeds never gets Speed Demon |
  | `¬bleed(w) ∧ A ∉ {trackstar, support} ⇒ Speed Demon` | otherwise always take it; on trackstar and support, worth decides |
  | `w ∈ {Cursed Flames, Pleeksty's Inferno} ⇒ ¬Flaming Flourish` | never Flaming Flourish with these |
  | `w ∈ {Purple Cloud, Wyrmtooth, Railblade, Duskpole, Sancticar, Imperial Staff} ⇒ ¬Turtle Shell` | any of these equipped: never Turtle Shell, nor anything that needs it; reviews cannot add it back |
  | `light(w) ⇒ ¬Turtle Shell` | no Turtle Shell on a light weapon, except The Flippers of Fate, Flareblood Kamas and Frostthorn |
  | `medium(w) ∨ heavy(w) ⇒ Turtle Shell` | always Turtle Shell on a medium or heavy weapon, unless one of the weapons above is equipped |

  Names must match exactly, so Prototype Railblade is not Railblade. When no weapon is recorded, tri asks for it.

  If the build records no weapon but you answered with a class (fist, greatsword), a rule applies when it holds for every weapon of that class: fist never takes Turtle Shell, greatsword always does.

  Since v3.4, a required weapon talent beats everything except the game itself. Review removals, mined rules and review rules cannot take it out, and its prerequisites (Knight's Rally for Turtle Shell) are protected the same way. It is left out only when the build's stats do not meet it or its prerequisite, a mutually exclusive pick is held, or the talent cap leaves no room.

**Order of picks.** A talent comes in together with its missing prerequisites, which go in first (Knight's Rally, then Turtle Shell). A prerequisite that builds take almost only to unlock another talent is never left on its own. A swap made on worth never takes out the only talent a picked prerequisite unlocks, so Unfazed stays with Grasp on Reality. Every completion is then checked like a full build under review before it is shown.

**Benchmarks.** Some picks become much better past a stat, for example guardbreaking mantras past 77 in their attunement. Their worth jumps steeply around that point instead of creeping up.

## What is fixed

Some things are written into tri and no review or rule can change them:

- stat requirements, read at the pre-shrine and final stages only
- prerequisites and mutually exclusive picks
- Vow of Iron only picks, never in a Pathfinder build
- the talent cap (76 − 2m) and the mantra slots
- the three weapon rules above
- the weapon-class focus: once a class is chosen, other classes' weapon talents are off
- prerequisites go in before what needs them

A wrong fixed rule has to be fixed in the code or the game data.

## Every rule on a pick

Apart from worth, everything that pushes a pick in or out is listed in one place: mined and written rules (also the ones where the item is only a condition), rules drawn from the reviews, sidegrades, categories, the weapon rules, game data and, for a given build, the class focus and the review claims that bind there. Each entry says whether it can be switched off and, given a build, whether it acts on it.

Mined rules, review rules (ids `rv:...`) and sidegrades (`sg:...`) are switched off by id and back on the same way. The fixed rules above are listed but cannot be switched off. Review claims are not rules: they change when the review is edited or its marks undone.

Limits: rules drawn from the reviews other than broken, redundant and conflicting act only through the review claims they came from, so they are listed as inactive. Undoing an item's marks does not undo the marks on its prerequisites unless asked to.

## Reading the symbols

| symbol | read it as |
|---|---|
| `⇒` | then (if the left side holds, the right side follows) |
| `∧` | and |
| `¬` | not, or without |
| `⊤` | always (no condition) |
| `⊥` | works against |
| `≻` | is preferred over |
| `A` | the build's archetype |
| `P(t \| A)` | share of A builds (that could take t) that do take t |
| `w(t)` | worth of t |
| `Δ` | change |
| `a → b` | was a, became b |
| `∀` | every |
| `\| c` | given c (only when c holds) |
| `n` | how many builds or reviews were counted |
| `p` | chance the fact is a fluke; smaller is stronger. `p < 10⁻⁶` means less than one in a million |

The item names in these examples are illustrations.

**Rules.** The tag in front says where the rule comes from: `[F]` foundational, `[C]` central, `[X]` exclusion (mined from builds), `[R …]` learned from reviews.

| written | in words |
|---|---|
| `[C] Flamecharm ≥ 50 ∧ Flame Leap ⇒ Ignition   P 0.92 ≥ 0.85 · n 340` | Builds with at least 50 Flamecharm that have Flame Leap take Ignition. 92% of 340 such builds do, and it is certified that at least 85% would on new builds. |
| `[X] Agility < 30 ⇒ ¬Rapid Slashes   P 0.01 ≤ 0.02 · n 120 · else 0.40` | Builds under 30 Agility never take Rapid Slashes: 1% do, certified under 2%; 40% of builds with more Agility do. |
| `[F] ⊤ ⇒ Vitality` | Always take Vitality when you can. |
| `[R supported] Agile Dash ⊥ Shoulder Bash   +3.0 −0.5 · lb 0.82` | Reviewers say Agile Dash works against Shoulder Bash. Support weighs 3.0 and counter-evidence 0.5; the rule is right at least 82% of the time. |
| `[R supported] Item B ⇒ ¬Item A` | Item A is redundant if you have Item B. |
| `[R supported] B ≻ A \| space short` | When slots are short, prefer B over A. |
| `[R supported] A ⇒ ¬t` | t does not fit this archetype. |
| `[R supported] chime ⇒ ¬t` | t does not belong on Chime builds (a strong SM). It says nothing about other archetypes. |
| `capacity  \|T\| < cap ⇒ add best until \|T\| = cap` | A build under its talent cap is filled before swaps are suggested; one over it is trimmed first. |
| `slots  type(x) = combat ⇒ x ∉ support slot` | A combat mantra never takes the support slot. |
| `benchmarks  logit += κ σ((max(pre, fin) − k) / width)` | Worth jumps around stat k (κ is how far, width how sharp). |
| `Speed Demon ⇒ wielding no bleed` | Speed Demon needs a weapon without bleed. |

**Facts.**

| written | in words |
|---|---|
| `P(F6 \| Frost mage) 0.92, P(F6 \| ¬Frost mage) 0.30   ×3.1` | 92% of Frost mages take F6, against 30% of other builds that could; that is 3.1 times as often. F6 marks a Frost mage. |
| `Frost mage ↔ Iron tank: C1_4 + C1_5 = 64% of d²` | Most of what separates these two archetypes (64%) comes down to C1_4 and C1_5. |
| `Frost mage: w(t) ≥ ½ ⇔ Strength ≥ 55 (req 40)` | t is unlocked at 40 Strength, but only becomes a common pick from 55. |
| `Frost mage: hole F6  P(F6) 0.99, reviews −F6 9/13` | A hole: almost every Frost mage takes F6, yet reviewers removed it in 9 of 13 reviews. A gem is the reverse: few take it, reviewers add it. |
| `Frost mage: gem G1  P(G1) 0.12, reviews +G1 7/10` | A gem: few take G1, but reviewers added it 7 times in 10. |
| `Frost mage: s ⇒ Δw(t) +0.21` | Having s makes t more worth taking (by 0.21). A minus sign means s makes t worse. |
| `∀A ≠ Frost mage: w(t) ≥ 0.9; w(t \| Frost mage) 0.31` | Every archetype takes t except Frost mage. |
| `Frost mage @P1: P(t) 0.62 → 0.21` | Since patch P1, Frost mages stopped taking t (62% then, 21% now). |

A one-line key: `T` is the talents, `M` the mantras, `m` the mantras that lower the cap, `⊕` mutually exclusive, `σ` the S-shaped curve.
