# tri in plain words

The idea behind tri and how well it works so far without the maths.

## What tri is

tri looks at thousands of Deepwoken builds and learns what builds like yours usually take. Expert reviewers correct it, and corrections take effect at once. Given a build it tells you:

- what kind of build it is (its archetype);
- which talents and mantras to add or drop, and why;
- how to fill in an unfinished build.

It always keeps the game's hard limits: the talent cap, mantra slots, stat requirements, oath locks and exclusive talents.

## The big idea: a map of builds

Picture every build as a dot on a map. Two builds are close when they would value the same talents. Thunder bruisers bunch up in one place and Frost mages in another; hybrids sit in between. Nobody labels the dots. The map comes only from which talents builds take together.

Two numbers come off the map:

- **Worth**, written `w(t)`: how likely a build like this one is to take talent or mantra `t`. A worth of 0.9 means 9 in 10 similar builds take it.
- **Distance**: how far apart two builds are. Close builds have close worth for every item, and this is guaranteed by how the map is built, not just usually true. That guarantee is what lets one review safely guide nearby builds.

**Locators** are builds that admins pin with a name ("this is a Frost mage"). They turn regions of the map into named archetypes. They do not move the map; they only name parts of it.

**Near-twin archetypes.** Gank builds and chime builds share almost everything and differ by a few talents and mantras, so on the map they sit on top of each other and gank builds used to be read as chime. tri now checks whether the pinned builds of two archetypes get mixed up on the map. If they do, it learns the few picks where they differ (for example "gank takes these three talents, chime takes these two mantras") and uses those picks to decide between just those two, only for builds that sit in their shared spot. It switches this on only if it gets more pinned builds right than wrong when each one is hidden and guessed from the rest. Every other build, and every other archetype, is read from the map alone, exactly as before. `/archetype` shows the telling picks as `Gank ↔ Chime: +X +Y −Z`, and `/locator check` says whether the split is on.

## Caution

When tri does not have enough evidence for a mark, it holds the mark back instead of guessing: it keeps the pick it would have dropped, or skips the pick it would have added, and fills the slot with its next choice, so no slot is left empty. Held marks are never shown. It can also ask questions, such as "what archetype is your build?", "is X central to your concept?", "do you want X?" or "what weapon does this build use?" (a weapon's name or a class). The questions come first, as a numbered sheet; reply by number, e.g. `1. gank 2. yes 3. fist`, or `post` to post now (the review-sheet grammar `+ 2,3`, `- 4`, `# Gank` also works); tri asks only what it needs, at most 6 questions per build, the weapon right after the archetype, and an answer can lead to a follow-up sheet. Where an archetype has few builds or reviews, tri leans on what builds take overall instead of asking, but only when that agrees with what it would do anyway. Answering "yes" to "do you want X?" makes tri take X if it fits. Answers change only the answer for your build. Since anyone can answer and answers can be wrong, they only shift other builds a little, and only once many different people agree.
