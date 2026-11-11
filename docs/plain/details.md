# Shrine timing

A talent that needs another talent can only be taken once you have that other talent. tri checks the order: if you meet Magical Resolve's 40 Willpower only before the Shrine of Order, but Battle Tendency's 15 Fortitude only after it, you can never hold both requirements at once, so tri will not suggest Magical Resolve. Taking Battle Tendency before the shrine and Magical Resolve after it is fine.

## dwt.sh builds

tri reads [dwt.sh](https://dwt.sh) build links anywhere it reads a builder link. dwt.sh splits a run into more stages (before the Shrine of Order, between Shrines of Mastery, the end). A talent counts as reachable if you could take it at any of those stages. tri only learns from the first and the last stage, so a long shrine chain does not change what a build teaches it.

## Weapon class

Most builds use one weapon class. Before tri fills a build, it picks the class: the weapon the build equips, or the class of the weapon talents it already has, or one a reviewer added. If there is none, it looks at the most similar builds in its data and picks the class they use most. Talents of other classes are left out, and the chosen class's good talents are taken before the extras. A reviewer can still add a talent of another class.

## What it does not do

- It does not judge stat spreads, only talent and mantra picks.
- It does not retrain on its own. Reviews act at once, but the map only changes when someone refits it.
- It does not name archetypes. Admins do, with locators.
- Facts are statistics. With about 5% allowed to be flukes, a single fact is strong evidence, not proof.
- Balance changes that are not in the game data have to be entered by hand.
