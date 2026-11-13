# 3. Modules

One section per main module, then a table of the supporting ones. Paths are from the repository root; the full tree is in [Section 1](layout.md). Definitions (configurations and value types) live in each subpackage's `defs.py`.

## 3.1 `tri/core`: the data model

Definitions are in `types.py` (kinds, reasons, `Item`, `Build`, `Operation`, `Review`) and `context.py` (tokens). `Catalog` is in `catalog.py`; its methods are split into `availability.py`, `picks.py` (exclusions, prerequisites, `prereq_order`) and `slots.py` (cap and mantra slots). `codec.py` holds `validate_review` and the JSON forms; `tables.py` the bundled tables.

| Object | Role |
|---|---|
| `Kind` | talent or mantra |
| `Category` | foundational / central / luxury hint |
| `Action`, `Strength`, `Reason` | review vocabulary: add / remove / iffy (`RETIRED_ACTIONS`: the old audit verdicts overturn / uphold, refused since v3.5); weak / strong; eight reasons (NA, GG for adds; LN, RW, WA, NS, SM, BR for removes) |
| `Item` | one catalog entry: requirements (clauses of stat-sum thresholds), prerequisites, exclusions, tags |
| `Catalog` | all items of one ruleset; `ids` is sorted and fixes every array's column order. Availability: `available(item, stats)` for one stage, `available_in`, `availability_in`, `availability_stages(pr, fi)` over pre-shrine or final stats, never highest; context clauses (`Item.context`, `context_ok`, `context_mask`, `mantra_locks`, `lock_fields`, compiled once per catalog; `build_context` adds tokens implied by picks; `unavailable_reason`, `context_text`; `known_context` fills locks from the bundled `tri/data/context_locks.json` when a saved catalog has none) and prerequisite propagation (`propagate_prereqs`), [math §2.2](../math/02-constraints.md). JSON: `from_json`, `from_dict`, `to_dict`. Cap: `talent_cap`, `cap_excess`, `talent_cap_base` (76), `cap_per_mantra` (2), `cap_exempt` (Oath, Origin, Monster). Mantra slots ([math §2.6](../math/02-constraints.md)): `mantra_slot` (a mantra's slot type), `slot_grants` (a talent's or oath's grants), `mantra_slots_of(items, oath)`, `mantra_slot_usage` (slots, used, over, free), `mantras_fit`, `oath_of`. Exclusion: `excludes` (symmetric), `dependents`, `exclusion_removals`. Order: `prereq_order(names)` puts every prerequisite before the items that need it (a stable topological sort), so a list of adds can be applied one by one |
| `Build` | highest, final and optional pre-shrine stats, `mid_stages` (dwt.sh middle stages: availability only, never learned from), talents, mantras, `meta` (Oath, Murmur, ..., `stages` names), `build_id`, timestamp; `stages()` (initial, middle, final) |
| `build_context`, `token_key`, `describe_token`, `CONTEXT_LOCK_FIELDS`, `PLANNER_FIELDS`, `MANTRA_LOCKS` | a build's context tokens (oath, murmur, origin) and recorded fields, used by availability |
| `Operation` | action, `item_id`, strength, reason, `related`, `group`, `note` |
| `Review` | an edit of a full build (`scope="full"`: unmarked items endorsed; `apply` gives the corrected build); audits are refused since v3.5 |
| `validate_review` | protocol checks: cap, exclusions, stranded prerequisites, reason rules; raises `ReviewError` or returns warnings |
| `fit_mantra_slots`, `BASE_MANTRA_SLOTS`, `known_slot_grants`, `known_mantra_slot` | the typed slot fit and the bundled slot table ([math §2.6](../math/02-constraints.md)) |
| `is_silentheart`, `legacy_names` | Silentheart; reading old `purpose` keys as `archetype` |
| `*_to_dict` / `*_from_dict` | JSON forms used by stores and saved runs |

## 3.2 `tri/model/matrix.py`: builds as matrices

`BuildMatrix` holds `X` (sparse picks, builds by items), `hi` / `pr` / `fi` (stats), `w` (weights), oath and murmur tokens, `gid` + `masks` (availability, one row per distinct pattern), `ids`, `times`, and for reviewed rows `originals` (the builds as submitted). Every statistic in tri is computed from this form. `from_builds` accepts a list of `Build` or `pipeline.CompactBuilds`; `pick_rate`, `sample`, `subset`, `append`, `save` / `load`. Each row also carries a lock signature (`lk` into `lk_vocab`: oath, murmur, origin, weapon and weapon-type tokens with the recorded fields; `lock_signature`, `parse_signature`), so availability groups apply every lock. Matrices saved before v2.11 have none and check only oath and murmur.

## 3.3 `tri/model/space.py`: the two fields

`ConceptSpace` owns the worth field $\ell_{bt}=\beta_t+\Gamma_t\cdot s_b+\Omega_t\cdot c_b+V_t\cdot z_b$ and its concept coordinates ([math §3](../math/03-worth.md), [§4](../math/04-concept.md)).

| Method | Does |
|---|---|
| `fit(bm, change_time, is_review)` | trains the field by minibatch Adam on weighted logistic loss over available entries; discounts pre-patch rows; holds out 5% for the check; solves every archetype exactly; builds weights, $Q$, $R$, regions |
| `solve_latent(bm, rows, partial)`, `_mm` | the unique archetype of each build (majorise-minimise) |
| `point(b, partial, region, loo)` | `FieldPoint(z, y, u, logits)`; `loo=True` gives leave-one-out logits (`_loo`); `region=k` plays the build as region $k$ |
| `coords(bm)`, `project(b)`, `u_of` | concept coordinates |
| `distance`, `distance_u`, `kernel`, `similarity` | weighted RMS logit distance and its Gaussian kernel |
| `metric_weights`, `_variance_weights`, `_calibration`, `_factor`, `set_metric`, `weights` | item weights ([math §4.4](../math/04-concept.md)); `set_metric` switches without retraining |
| `bounds(d)`, `item_bound_factor()` | what a distance guarantees about worth (Theorem 4.2) |
| `regions`, `region_of`, `reading`, `memberships`, `_fit_regions` | $k$-means regions, names, latents, soft memberships |
| `item_coords`, `axes`, `report` | item directions, principal axes, deviance explained |
| `_val_split`, `_val_score`, `validation` | the held-out check ([math §6.4](../math/06-accuracy.md)); diagnostic only |
| `save` / `load` | arrays and JSON meta; runs without weights load with the uniform metric |
| `params_key`, `data_key`, `latent_key`, `remember`, `cached_latents` | the latent memo: whole-matrix solves are stored under a fingerprint of the parameters and rows ([math §8.12](../math/08-convergence.md)) |
| `_device`, `_validation_report` | the compute device (`accel.resolve`); the held-out report shared by both training paths |

`accel.py` mirrors the NumPy training loop on a torch device in float32: `resolve(name)` and `describe(name)` choose and report the device, `train(space, ...)` runs Adam with the same random draws, and `latents(space, S, C, X, A, dev)` runs the archetype solve. Without torch everything runs on NumPy.

## 3.4 `tri/rules/passes.py`: the rulebase

| Object | Role |
|---|---|
| `Condition(kind, key, op, value)` | stat (`>=`, `<`, multiples of 5, on highest stats), context (`has`, `not`), item (`has`) |
| `Ctx`, `holds` | what conditions look at; evaluate one |
| `PassRule` | pass, item, conditions, effect (take / never), rate, certified bound, support, effective size, contrast, breadth, status (active / stale); `applies`, `text` |
| `wilson(p, n, z)` | one-sided Wilson bounds |
| `Rulebook.mine(bm, ...)` | the mining procedure of [math §6.3](../math/06-accuracy.md): split, trees, screen, uniformity, shorten, certify, tidy, conflicts, passes |
| `firing`, `forbidding`, `pass_of`, `for_item`, `separates(item, a, b)` | applying rules; whether a rule condition separates two builds (used to stop claims crossing rule boundaries) |
| `certification_check`, `drift`, `mark_stale`, `_rescore_changed` | re-measurement, meta shifts, patches |
| `summary`, `report`, `to_json` / `from_json` | inspection and persistence |

### v3.0 selection rules

| Module / object | Role |
|---|---|
| `tri/rules/groups.py`: `derived_groups`, `grounded_stats`, `related_items`, `weapon_class`, `rule_tokens`, `group_counts` | groups $G(c)$ and $D(k)$ from game data; the stats and anchors a rule on an item may use; context tokens of a build (`oath:`, `murmur:`, `origin:`, `weapon:`, `archetype:`) ([math §6.12](../math/06-accuracy.md)) |
| `Condition(kind="group")`, `Ctx.groups` | group literals `\|G(c)\| ≥ n` |
| `Rulebook.mine` additions | grounding with fallback, related anchors, A50/B25/C25 replication, literal necessity (`necessary`), `reader` (the archetype reader set by `Tri`) |
| `tri/rules/language.py`: `Vocabulary`, `parse`, `format_rule`, `rule_id`, `RuleSyntaxError` | the rule language ([math §6.13](../math/06-accuracy.md)) |
| `tri/rules/authored.py`: `AuthoredEntry`, `AuthoredRules` | added and suppressed rules; `authored_rules.json` in the run |
| `tri/rules/selection.py`: `SelectionConfig`, `build_selection`, `selection_report`, `authored_stats` | the selection set: each mined rule stands, is refined, overruled or disputed by review evidence; authored rules always fire |
| `Tri.rules`, `Tri.selection`, `Tri.authored`, `add_rule`, `edit_rule`, `drop_rule`, `list_rules`, `find_rule`, `selection_report`, `mine_rules` | engine access; `mine_rules` re-mines without a refit |
| `tri/rules/registry.py`: `collect`, `review_rule_id`, `sidegrade_id`, `PREFIXES`; `Tri.all_rules(item, build)`, `dropped_ids`, `_apply_drops`, `review_rules_all` | v3.4: every non-worth rule in one shape (id, source, rule, effect, items, status, droppable, change, fires); `drop_rule` takes `rv:`, `sc:` and `sg:` ids through `AuthoredRules.suppress` ([math §2.11](../math/02-constraints.md)) |
| `_Situations._weapon_guard`, `_Sit.weapon["guard"]` | v3.4: weapon-required picks with met stats and their hard prerequisites; exempt from review removals, never rules, review rules and displacement ([math §2.8c](../math/02-constraints.md)) |
| `Constraints.tier`, `tier_here`, `InferConfig.rule_tiers`, `tier_fallback`, `tier_context` | worth ranks only luxury talents; v3.0.2 reads the tier on the build |

## 3.5 `tri/model/worth.py`: blended worth

`FieldWorth(ref, U, cfg, is_review, touched, conditions)` holds reference builds (corpus sample and corrected reviewed builds at their originals' coordinates). `rates(point, picks)` returns `LocalRates`: `q` per item, the field's own answer, support, neighbours and weights, and the capacity estimate. Items a review changed count 40 times; conditional claims count only where they hold; the chasm ([math §5.4](../math/05-locality.md)) scales the review boost and every corpus and ring neighbour. `prior` is the popularity baseline.

## 3.6 `tri/reviews/authority.py`: the review ledger

`ReviewLedger(reviews, catalog, cfg, embed, kernel, change_times)` turns reviews into `Claim`s and resolves conflicts newest-first by similarity (`Supersession`). It answers:

| Method | Returns |
|---|---|
| `neighbours(build)` | reviewed builds within reach, newest first |
| `claims_for(build, vec)` | per item, the newest active claim that holds in the build; global claims (broken, GG) after local ones; echo keeps only where no explicit claim exists. `vec` overrides the build's own coordinates (the forced latent of a completion) |
| `binding(claim)` | whether a claim overrides rules and worth ([math §8.3](../math/08-convergence.md)) |
| `conditions()` | related items of conditional claims, for worth |
| `report()`, `stale` | inspection, patch-stale claims |

`made_by_tri(build)` marks builds tri made (`meta.made_by == "tri"`, set by the source pipeline for "tri completion of …" descriptions, or a name ending "(tri completion)"); unmarked items on them are echo keeps at `echo_keep` ([math §5.5](../math/05-locality.md) item 11).

`reach_kernel(cfg)` is the review kernel of the concept distance; with locators it is wrapped by the chasm (`LocatorSet.kernel`). `build_similarity` (Jaccard with a stat factor) is the fallback when no space is fitted.

## 3.7 `tri/rules/induction.py`: rules from reviews

`RuleInducer.induce(reviews, ledger)` proposes typed rules (BROKEN, REDUNDANT, CONFLICT, NECESSARY, NO_FIT, OFF_ARCHETYPE (strong SM, scoped to one family through the `family:<key>` feature), PREFER, and GENERAL for inspection) from operations and upheld changes, scores support against counterexamples (including overturns) with ledger weights, and refines disputed ones with the best separating condition from `context_features`. Only BROKEN, REDUNDANT and CONFLICT act on every build.

## 3.8 `tri/infer`: answers

`PassEngine(catalog, space, rulebook, worth, review_rules, ledger, cfg)` is constructed in `engine.py`; its methods are in `situations.py`, `constraints.py`, `completion.py` and `reviewing.py`. `InferConfig`, `Change`, `Alternative` and the review-queue and draft types are in `defs.py`. Every list of adds is put in prerequisite-first order (`Catalog.prereq_order`) before it is applied or returned, so a derivative never reaches the game API ahead of its prerequisite.

| Method | Does |
|---|---|
| `situations(b, ...)` | one situation per plausible region: field point, worth, context, memory (claims that do not cross a rule boundary) |
| `review(b, ...)` | steps 0 to 9 of [math §7.4](../math/07-strictness.md): cap, reject, exclude, require (with exclusive swap), no fit, refill, swap, iffy, repair, violations |
| `complete(b, capacity, region)` | fill in pass order up to the talent cap and the typed mantra slots, both recomputed after every pick, leaving no slot empty that an available pick fits; mantras first for a build without mantras unless Silentheart ([math §2.6](../math/02-constraints.md), 7.6); `capacity` only lowers the limits |
| `_feasible`, `_forbidden`, `_required`, `_general`, `_violations`, `cap_excess` | the checks behind the steps |

`PassEngine` also holds `manual` (admin categories), `unrated_items` (synced items with no evidence yet) and `sidegrades`; `_required` applies categories before mined rules, `_lux` and `_cat_kept` protect luxury and categorized items, `_sides` and `_offers` keep answers on one side of a sidegrade and list the other.

`Alternative`: talents, mantras, plausibility, `changes` (each a `Change` with action, item, pass, source, reason, rule text, review id), `iffy`, region name, `passes`, `worth`, `memory`, capacity, `violations` (missing, forbidden, conflict, unanchored, talent_cap, mantra_slots), `sidegrades` (offers), `unrated`.

## 3.9 `tri/reviews/locators.py`: named archetypes

| Object | Role |
|---|---|
| `Locator`, `new_locator`, `archetype_key` | a labelled build; name normalisation |
| `LocatorSet(locators, coords, cfg, reach, vec, background)` | `read` / `read_many` (equal-prior kernel posterior), `activity`, `barrier_factor`, `kernel`, `region_labels`, `loo`, `select_power` |
| twin split ([math §6.6](../math/06-accuracy.md)) | `set_background`, `_find_pairs` (archetype pairs the kernel confuses leave-one-out, with their tells and a held-out gate), `_contrast` (rates shrunk toward the pair, tells), `_split`, `refine` (re-splits only an active pair's mass, only where the pair holds most of the reading and within reach), `active_pairs`, `pair_of`, `note` / `fine_rows` (remembered readings, so the review barrier sees them), `contrast` / `adjust` (worth corrected on the tells only), `distinct` (the tells of an active pair), `loo_fine` |
| `ArchetypeReading` | archetype or None, confidence, posterior, nearest locator, distance, `located`; `coarse` (the kernel reading, when the fine one moved it) and `forced` |
| `LocatorConfig` | `bandwidth`, `reach`, `min_confidence`, `barrier`, `chasm_power`, `chasm_hard`, `chasm_corpus`, `chasm_rules`, `chasm_rule_min`, `chasm_rule_sample`, `consensus`, `consensus_kappa`, `consensus_min_reviewers`, `consensus_op_weight`, `consensus_worth`, `consensus_lift`, `consensus_floor`, selection settings, `refine`, `refine_prior`, `refine_min_logodds`, `refine_min_locators`, `twin_overlap`, `twin_mass`, `twin_clip`, `twin_min_gain` (the v2.15 `refine_floor` and `refine_clip` are ignored when a saved run is loaded) |

## 3.10 `tri/engine`: the facade

`Tri(catalog, cfg)` is constructed in `facade.py`; the method groups below live in `fitting.py`, `reviewing.py`, `answering.py`, `archetypes.py` and `persist.py`, and `TriConfig` in `defs.py`. It owns `data`, `space`, `rulebook`, worth, `reviews`, `ledger`, `review_rules`, locators and `patches`.

| Group | Methods |
|---|---|
| fitting | `fit(builds, weights)`, `refit()`, `calibrate()` |
| reviews | `add_reviews`, `set_reviews`, `set_memory`, `validate`, `dry_insert`, `review_agreement`, `ledger_report`, `rule_report` |
| answers | `review` (alias `recommend`), `complete`, `worth`, `build_worth`, `explain`, `concept`, `categories`, `summary`; `review`, `complete` and `build_worth` take `archetype=` to force one |
| locators | `set_locators`, `add_locators`, `remove_locator`, `archetype` (with the twin split), `locator_report`, `select_metric`, `set_metric`, `held_out_ids`; `archetypes()` (names that can be forced), `force(name)` (a forced archetype: its locators' mean latent and its profile, or a region by name; `ValueError` if unknown), `_vec` (a build's picks and availability) |
| review queue | `next_reviews(pool)` |
| game changes | `apply_patch`, `migrate`, `extend_catalog` (no refit), `drift` |
| categories | `set_categories`, `manual` |
| sidegrades | `sidegrades` (mined at refit and on load if missing) |
| persistence | `save(path)`, `Tri.load(path, catalog)` |

`TriConfig` groups `space`, `passes`, `worth`, `infer`, `ledger`, `locators`, plus `context_fields`, `data_size`, refit weights and calibration grids.

## 3.11 Supporting modules

| Module | Role |
|---|---|
| `tri/game/patch.py` | `Patch`, `PatchLog`, `diff_catalogs`, `patch_from_diff`, remapping for `migrate` |
| `tri/infer/active.py` | `ReviewQueue`: scores pool builds by ambiguity, disputed review rules and novelty |
| `tri/analysis/evaluate.py` | `certification`, `strictness_benchmark`, `mapping_benchmark`, `completion_benchmark`, `rule_benchmark`, `review_benchmark`, `group_split`, `evaluate` |
| `tri/game/gamedata.py` | parse requirements from pocamind/data (`auto`, `pocamind`; `all` adds api.deepwoken.co) into a `Catalog`; granted talents; `coverage` |
| `tri/sources/pipeline.py` | optional: planner export to `PretrainSet` (parsing, granted-talent exclusion, dedupe, lineage weighting). Your own pipeline can skip it and call `Tri.fit` |
| `tri/views/plots.py` | data functions (`concept_map`, `place`, `worth_matrix`, `worth_space`, `fit_projection`, `view_kept`) and the themes; reads the model, never changes it |
| `tri/views/figure.py` | `Scene`, `draw` (one transparent 3-D figure), `png`, `field_hulls`, `dense_origin`, `cosines`, `angle`, `orthogonality` ([math §4.5](../math/04-concept.md)) |
| `tri/views/interactive.py` | `scene_data`, `page`: a `Scene` as one self-contained HTML page (no network) |
| `tri/core/stats.py` | `wilson`, `wilson_lower` (score interval of a proportion; tell weight [math §6.7](../math/06-accuracy.md)) |
| `tri/analysis/categorize.py` | `build_atlas` (one latent solve per build, readings from the locators or regions), `categorize(tri, b)` (frame, archetype, reading, variants), `soft_kind`, `separating` ([math §12.2](../math/12-reading-the-field.md)), `hierarchy` (Ward on centres, families at reach), `discover` (stable k-means of unplaced builds), `quality` (margins, stability, review consistency), `intersection` (representative builds where archetypes meet, [math §4.5](../math/04-concept.md)) |
| `tri/rules/symbolic.py` | `LEGEND`, `legend`, `num`, `pval`, `cond`, `token`, `conj`, `pass_rule`, `review_rule`, `fact`: one symbolic line per condition, mined rule, review rule and fact family ([math Appendix C](../math/appendix-c-statements.md)). `PassRule.text()`, `Rule.describe()` and `Fact.symbol` call it; `PassRule.plain()` and `Rule.plain()` keep the sentences |
| `tri/analysis/facts.py` | `FactMiner` families (defining, separating, threshold, hole / gem, synergy, universal, shift), held-out tests, `benjamini_hochberg`, `mine_facts`, `FactBook` (JSON; first certified, retired, digest) |
| `tri/analysis/stream.py` | `prequential`, `variogram`, `lowrank` (re-exported by `evaluate.py`) |
| `tri/views/insight.py` | `atlas_view`, `plot_atlas`, `archetype_card` / `plot_card`, `placement` / `plot_placement` (`part="map"` or `"pi"`), `item_landscape` / `plot_item`, `stat_curves` / `plot_stats` (benchmark line), `plot_lookup`, `plot_replay`; same themes as `plots.py` |
| `tri/rules/statements.py` | `HARD` (the hard rules), `model_rules`, `entries` / `entry` (pocamind/data), `lookup`, `RULE_SYMBOLS`, `FACT_SYMBOLS` ([math §12.6](../math/12-reading-the-field.md)) |
| `tri/rules/triviality.py` | `BASELINES`, `trivial_fact`, `trivial_rule`, `related`, `lock_locked` ([math §12.6](../math/12-reading-the-field.md)) |
| `tri/game/weapons.py` | `table`, `resolve`, `equipped`, `weapon_tokens`, `tokens_of`, `item_clauses`, `scale`, `summary` ([math §2.8](../math/02-constraints.md)) |
| `tri/game/weapon_rules.py` | `WeaponRules`, `weapon_rules`, `main_weapons`, `needs_weapon` ([math §2.8c](../math/02-constraints.md)) |
| `tri/infer/focus.py` | `class_names`, `norm_class`, `token_classes`, `item_classes`, `class_table`, `weapon_classes`, `evidence`, `ClassPrior` ([math §2.10](../math/02-constraints.md)) |
| `tri/game/priors.py` | `benchmarks`, `benchmark_of`, `stat_offsets`, `weapon_scale`, `apply`, `adjust_q` ([math §2.9](../math/02-constraints.md)) |
| `tri/analysis/sidegrades.py` | `SidegradeConfig`, `Sidegrade` (`side`, `leaning`, `applies`), `mine(data, regions, catalog, cfg)`: proposal on one half, test on the other ([math §4.9](../math/04-concept.md), 6.10) |
| `tri/game/sync.py` | `head_commit` (GitHub API; falls back to the git ref list when rate limited), `snapshot`, `plan` (diff of game data snapshots), `SyncPlan`, `apply` (extend and save to `<run>/sync/`), `restore` (also repairs case twins saved by earlier versions), `twins`, `sync`; names are matched by `name_key`, so a respelling such as "To the Finish" maps onto the fitted "To The Finish" |
