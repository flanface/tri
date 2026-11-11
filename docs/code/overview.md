# 2. The big picture

tri has three learned parts, one memory and one decision procedure.

```
   corpus builds (from your pipeline)
        │
        ▼
  matrix.BuildMatrix ──► space.ConceptSpace ──► coordinates u, regions
        │                (worth field +              │
        │                 concept field)             ▼
        ├──────────────► passes.Rulebook        worth.FieldWorth
        │                (certified rules)      (field + neighbours)
        │                        │                   │
 reviews ──► authority.ReviewLedger (memory)         │
        └─► rules.RuleInducer (review rules)         │
                    │            │                   │
 locators ─► locators.LocatorSet (readings, chasm)  │
                    │            ▼                   ▼
                    └────────► infer.PassEngine ◄────┘
                                     │
                                     ▼
                Alternative (changes, passes, worth, violations)
```

| Part | Question it answers | How it is made | Design concept it serves most |
|---|---|---|---|
| Worth field (`space`) | how likely is each pick, given stats, oath, murmur and archetype? | logistic factor model, Adam; exact per-build archetypes | accuracy (held-out check) |
| Concept field (`space`) | where is this build, and who is like it? | QR of the weighted field; distance = weighted RMS logit difference | locality |
| Rulebook (`passes`) | what do builders always or never do here? | tree-proposed conjunctions certified with Wilson bounds on a held-out half | accuracy, strictness |
| Worth (`worth`) | among picks no rule decides, what do builds like this take? | field shrunk towards kernel-weighted neighbours | locality |
| Memory (`authority`) | what have reviewers said about builds like this? | not learned: claims with reach, newest wins, binding | rapid convergence, locality |
| Locators (`locators`) | which named archetype is this? where do archetypes end? | not learned: kernel readings over labelled builds | accuracy, locality |
| Inference (`infer`) | what is the smallest edit that meets every rule? | not learned: authority order, passes, repair, violations | strictness |

Authority, highest first: hard constraints, binding review memory, global review rules, certified rules, worth.
