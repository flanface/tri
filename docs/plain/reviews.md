# How reviews change answers

A reviewer looks at a build and marks what is wrong ("remove X, works against Y"; "add Z"). Then:

- the correction applies to that build and to builds close to it on the map, right away, with no retraining;
- it does not cross into other archetype families at all: the border between two families is a chasm, not a slope, so fixing a PvE build does not change gank builds;
- when two reviews disagree on similar builds, the newer one wins;
- "works against Y" only affects builds that actually have Y;
- a review of a completed build takes effect in the next completion at once, even over a "generally good" mark made on another build;
- a talent marked strongly generally good ignores its weapon class requirement: some talents, such as Turtle Shell and Unstoppable Force, are used without their weapon.

tri never retrains by itself. The map stays as it was fitted; reviews act on top of it.

How much a pick is worth comes mostly from the closest builds on the map, and only from builds of the same family. Where a family has few builds, tri also glances at builds a little further away, at a small weight, but only at other variants of the same family. Patterns mined from the whole corpus ("builds with Y take X") apply only in the families where they actually hold.

## When reviewers agree

One note changes builds near the reviewed one. When several reviewers say the same thing about a family, for example three of them write that Replenishing Knockout and Agile Dash point to Gank, tri treats it as a fact about Gank: every Gank build values those talents more, holding them makes a build read as Gank more readily, and they count as central. The more reviewers agree, the stronger it gets (two reviewers about 57%, four about 73%, ten about 87%); one reviewer repeating themselves counts once, and someone disagreeing pulls it back. Other families are not affected. The archetype card lists these agreed picks under "reviewer consensus".

## Many reviews at once

A single review of a build that is merely similar only nudges. When several reviews of builds in the same family, or of close neighbours, agree on an item, tri treats the agreement as binding: three strong marks, or four builds that agree almost without exception. A full review that leaves an item out of the build it returns counts as a vote against adding it. tri never swaps a pick out that nearby reviewers kept, or swaps one in that they removed or left out. Empty slots are still filled, and requirements still apply.

## Reasons and notes

A reviewer can say why, in a few characters:

| Write | Means | What tri does with it |
|---|---|---|
| `+ X > Y` | add X, it synergises with Y | adds X on nearby builds that have Y; ignores builds without Y |
| `- X NW > A, B` | X does not work without A and B | on nearby builds missing A or B: adds them if there is room, otherwise drops X; once supported, tri never adds X without A and B anywhere |
| `- X LL` | X is worth less as a luxury | drops X here; nearby builds only see X as a little less useful |
| `+ X LM` | X is worth more as a luxury | the opposite |
| `= X w > Y` | X is a sidegrade to Y in this archetype | marks X as iffy and never stacks X with Y nearby |
| `= X s > Y` | the same, for every build | the same, everywhere |
| `# Gank: A, B` | A and B point to Gank | builds near this one that hold A and B lean towards Gank, if Gank was already plausible |
| `# is Gank` (or `# is Gank / Frost Mage`) | this build is a Gank build | builds very close to this one are read as Gank more readily |
| `# not: Chime` | this build is not Chime | builds very close to this one are read as Chime less readily |
| `# core: A, B` | A and B define this build | nearby builds value A and B more, tri calls them "indicative", and it never drops them just to make space |
| `# avoid: X` | nearby builds should not take X | nearby builds value X less |
| `# weapon: Silver Dagger, Railblade` | the build also uses these weapons | they are added to the build's weapons, with their classes |
| `- X NS > Silver Dagger` | X is wrong with this weapon | applies only to nearby builds that use Silver Dagger; others are unaffected |
| `// text` | any comment | stored and shown next to every change the review causes |

A comment after a line (`- 3 NS // too slow`) belongs to that line.

## No audits

Audits (judging tri's own changes instead of the build) were removed in v3.5. They had no removal reasons, so they said almost nothing and pulled real reviews off course. Stored audits are ignored and can be deleted.

## Undoing marks

An admin can take any stored mark on an item back out of the reviews: generally good, adds, removals, iffy, indicator and core notes, or simply having kept it. The item then becomes neutral in those reviews: they say nothing about it, for or against. The rest of each review stays. This can be narrowed to one archetype or one reviewer, or whole reviews can be deleted instead. Since v3.4 the marks can also be chosen by reason across every item, for example every strong scenario mismatch removal at once, and undoing an item's marks can cover its prerequisites too (Knight's Rally with Turtle Shell). A strong scenario mismatch is a review claim, not a rule, so it goes this way and not by switching a rule off.

## Central picks

A central pick is one that defines how the build plays: an attunement path, or a very noticeable talent such as Blighted Song. Only a few talents are central. tri treats a pick as central only when an admin marked it central, a reviewer named it in a `#` note, or reviewers agree on it for the build's family. Central picks are respected: tri never drops one to make space or for being a luxury, only when it clashes with another pick. Reviewers follow the same rule: to remove a central pick, give a clash reason (redundant with, works against, broken, scenario mismatch, does not work without). Picks that the data shows almost every build of an archetype takes are labelled "archetype" instead.

## Archetype families and styles

An archetype is a family, like Gank. Inside a family, builds can play very differently, so locators can name a variant, like "Gank / Frost Mage". tri first decides the family, then which variant the build is closest to. It also describes the build's style from its stats ("Frost mage"), which is only a label and changes nothing.

## Small builds

A build with fewer than 15 talents has too few picks to tell its archetype, so tri shows only its changes, with no archetype name or map.

A `# is gank` note says which archetype the reviewed build is. tri trusts it over its own reading, and the review's claims then stay with that archetype: a necessary pick on a gank build is not pushed onto chime builds nearby.
