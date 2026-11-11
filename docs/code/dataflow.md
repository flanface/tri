# 4. Data flow

## 4.1 Fit

```
builds (+ weights)
  └─ BuildMatrix.from_builds ─► data = sample(data_size)
       └─ refit():
            1. ledger + review rules from current reviews (none on a first fit)
            2. corrected reviewed builds appended, weight kappa each (at most 25% of weight)
            3. ConceptSpace.fit(full)       field, held-out check, exact archetypes, Q R, regions
                                            (locator builds and the held-out 5% left out)
            3a. locators: select the metric power   only with >= 3 locators in >= 2 archetypes
            3b. locators placed, regions named
            4. space.coords(full)
            5. Rulebook.mine(full)          certified rules, passes
            6. ledger + review rules again  (similarity now uses the new space)
            7. worth rebuilt on a reference sample
            8. calibrate() if enough reviews
```

Step 3 takes about 28 s for 8,000 builds and 217 items on two cores; step 5 about 5 s. `tri.cli.fit` fits on the corpus only. Reviews enter the field only if you call `Tri.refit()` yourself after adding them; nothing refits on its own.

## 4.2 Add a review

`add_reviews` validates, replaces any review with the same id, rebuilds the ledger and review rules, and rebuilds worth with the corrected build at the original's coordinates. It takes effect at once.

## 4.3 Add a locator

`add_locators`, `set_locators` and `remove_locator` place locator builds (the field is unchanged), rename regions, and rebuild the ledger kernel and the worth boost with the barrier. Immediate. A locator whose build was a training row is counted in `locator_report()["in_training"]` until a refit with `Tri.set_locators` first.

## 4.4 Answer

```
build ─► situations(force=Tri.force(archetype))
           point = space.point(build)          archetype, u, leave-one-out logits
                   (forced: the archetype's mean locator latent; region: that region's latent)
           fine  = locset.note(u, picks)       twin split (or the forced one), kept for the barrier
           q     = worth.rates(point)          blended worth
           q     = locset.adjust(q, u, fine)   twins: worth moved on their tells only
           mem   = ledger.claims_for(build)    newest relevant claims (similarity x barrier)
           veto  = ledger.vetoes_for(build)
           ctx   = Ctx(stats, picks, tokens)
           weapon = weapon_rules(build, archetype)  never / required items, lock waivers (math §2.8c)
       ─► review / complete
           weapon rules > hard constraints > binding memory > review rules > certified rules > worth
           a pick enters with its missing prerequisites, prerequisites first (math §2.8d)
           repair until every rule is met, else violations
           complete only: the result is reviewed as a full build, up to 2 rounds (math §2.8e)
       ─► [Alternative]                         best region first (named with the twin split); forced: one
```

The twin split ([math §6.6](../math/06-accuracy.md)) changes nothing unless the locators show two archetypes the map confuses (leave-one-out overlap at least 0.2) and splitting them by their tells corrects more held-out locators than it breaks. Then it re-splits only that pair's probability, only for builds where the pair holds at least half the reading and that are within reach; every other archetype and every other build keeps the kernel reading exactly. The same reading names the first alternative, feeds `archetype()`, the atlas and the review barrier, and corrects worth on the tells only.

`dry_insert(review)` runs this with one extra review and returns the difference without keeping it.

## 4.5 Insight (categories, facts, stream)

`ModelService.insight(kind)` (HTTP `POST /v1/insight/{kind}`) builds an `Atlas` once per memory state (space, review count, last review, locators, patches) and caches it with the mined facts. Kinds: `atlas`, `tree`, `card`, `item`, `stats`, `meta`, `facts`, `suggest`, `quality`, `replay`, `place`, `categorize`, `digest`. Each returns `files` (PNG), a one-line `text` and `data`. Mining updates the fact book (`TRI_FACTS_PATH`, else `facts.json` next to the run). None of it changes the model.

## 4.6 Patches

`apply_patch` logs a patch, marks touched rules stale and discounts pre-patch evidence. `refit()` re-scores stale rules. `migrate(new_catalog, renames)` remaps data and reviews, then refits.

`extend_catalog(new_catalog, time_)` needs no refit: new items get a zero field direction (`ConceptSpace.extend`, `fitted` mask), missing items are retired, changed items are patched, and data and reviews are remapped. `sync.apply` calls it from a game data diff and saves `<run>/sync/{catalog.json, state.json, patches.json}`; `sync.restore` replays it after every load.

## 4.7 Speed

Measured on a synthetic 40,000-build world on the real catalog, two CPUs.

| step | v3.0.5 | v3.1.0 | change |
|---|---|---|---|
| load | 2.7 s | 1.6 s | availability masks cached beside the data |
| memory (200 reviews, 80 locators) | 20–31 s | 3.6 s | batched placement, vectorised review similarity, `sqdist` |
| review (3 alternatives) | 3.1 s | 0.27 s | memoised dependants and kernel rows |
| complete | 0.7–1.7 s | 0.18 s | same |
| review sheet | 0.5–1.0 s | 0.08 s | same |

Numba was tried on the distance kernel: 20 ms against 33 ms for the BLAS expansion and 301 ms for the plain numpy cube, so it is not a dependency.
