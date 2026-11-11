# Results so far

## Real reviews

These numbers come from the real review record (32 reviews and 3 audits by one reviewer, taken on 2 November 2026). Audits were removed in v3.5, so the audit figures here are history. With one reviewer they show where tri goes wrong, not how well it agrees with reviewers in general.

- **Central picks are trusted.** In audits, the reviewer upheld all 13 of tri's central changes (real).
- **Archetype and copied-review changes are not.** All 18 archetype-pass changes and all 10 changes copied from other reviews were overturned (real). These are the first things to test by switching them off.
- **Mined patterns are a coin flip.** 34 of 73 changes from mined patterns were upheld (real, 47%), against 53 of 79 changes from plain worth (real, 67%). Wrong patterns are the most common reason a change was overturned (39 times, real).
- **Confidence helps only at the top.** Changes tri was at least 95% sure of were upheld 9 times in 11 (real). Below that, confidence barely predicts a verdict yet, so tri cannot yet hold back its weaker changes and still be right 9 times in 10.
- **Reviewers stay quiet on close calls.** The reviewer marked a pick as iffy once in 150 marks (real), so tri's draft reviews no longer write iffy marks by default.

## Real reviews, 5 November 2026

The record has 60 reviews and 63 audits, all from one reviewer (real).

- **The reviewer is consistent about items.** Take two reviewed builds that share only a fifth of their picks. When both have a mark on the same item, the marks agree 88% of the time (real).
- **Copying a review to another build still fails.** When tri carried a review's mark onto a different build, the reviewer overturned it 42 times in 47 (real). Now a copied mark only counts as a hint, unless the two builds are near twins.
- **tri kept repeating rejected changes.** 305 of the 535 overturned changes repeated a change already overturned on another build (real), for example adding Fishman (overturned 15 times, never upheld). From v2.23 to v3.4, once the reviewer had rejected the same change on two builds, tri stopped suggesting it. Replayed over the record, this would have prevented 85 overturns and blocked 2 good changes (real). It went away with audits in v3.5.
- **One reviewer now counts.** Agreement between reviews used to need two different people, so with one reviewer it never happened. It now needs two different builds.
- **Weapons can be waived.** Some builds use a talent without its weapon, carry two weapons, or clearly hold a shield they never selected. If a review adds or keeps such a talent, tri lifts that weapon lock, but only for the reviewed build.

How closely tri's own draft review matches the reviewer's is measured by the scoring functions in `tri.sources.record`, which need the fitted model and run on your machine.

## Test worlds

Every number in this part comes from made-up worlds where the right answer is known. They show the method works; they do not show it fits the real game.

- **The map works.** Builds of the same archetype end up close together. A random pair from the same archetype is closer than a pair from different archetypes 99.8% of the time (synthetic).
- **Rules hold up.** 231 of 233 mined rules (synthetic, 99%) still held on builds they had never seen. In a simple world, all 47 true rules were found with no false ones (synthetic).
- **Completion is better than "just take what's popular".** It recovered 71% of the right picks, against 56% for popularity alone (synthetic).
- **Reviews take effect fast.** After a meta change, one review fixed 93% of the affected builds it had not seen, and five reviews fixed all of them (synthetic). Adding 100 reviews takes about a second.
- **Near twins are told apart.** In a test world where one archetype was split into twins three talents and two mantras apart, with four pinned builds each, the map alone named held-out twin builds right 64% to 76% of the time; with the telling picks, 98% to 100% (synthetic), and no other build's reading changed. With only two pinned builds each, tri kept the split off rather than guess. A review of one twin reaches the other at 4% strength, against 99% inside its own twin (synthetic).
- **Reviews stay local.** Pinning 10 example builds per archetype halved how far a review leaks into other archetypes (0.33 to 0.15, synthetic), while keeping 93% of its effect inside its own archetype (synthetic).
- **Chasms help.** Cutting influence between families raised agreement with expert reviews from 98.35% to 98.48% (synthetic), and repeated notes on one family changed its far builds by more than 0.1 worth while another family's builds moved by less than 0.002 (synthetic).
- **Hard limits are never silently broken.** Every answer keeps the talent cap and mantra slots, or says exactly which rule it could not meet. This is proved, not measured.
- **Categories and facts.** On a test world, the map recovered the named archetypes, found the unnamed ones on its own, and caught a planted hole and a planted post-patch shift. Replaying 148 reviews in order, tri predicted 94% of reviewers' calls, against 73% without review memory (synthetic).

## Draft reviews

tri can now write its analysis as a review in the reviewers' own shorthand (`- 4,11 NS`, `+ Item NA`, `# core: A, B`). Every draft obeys the review rules, including the talent cap and mantra slots. It holds back changes it is unsure of, and keeps changes that would no longer fit out of the draft.
