# Appendix B. Sections and code

| section | code |
|---|---|
| 2 builds, availability, cap, exclusions, reviews | `tri/core/catalog.py` (`Catalog`), `tri/core/types.py` (`Review`, `Operation`), `tri/core/codec.py` (`validate_review`), `tri/model/matrix.py`, `tri/game/gamedata.py` |
| 3 worth field | `tri/model/space.py` (`ConceptSpace.fit`, `solve_latent`, `_mm`, `_loo`) |
| 4.1 to 4.4 concept map, weights, bound | `tri/model/space.py` (`u_of`, `coords`, `distance`, `item_coords`, `metric_weights`, `set_metric`, `bounds`) |
| 4.5 item worth space and views | `tri/views/plots.py` (`worth_space`, `concept_axes`, `worth_axes`, `view_kept`, `distance_check`) |
| 4.6 regions | `tri/model/space.py` (`_fit_regions`, `reading`) |
| 6.6 twin split, forced archetype, tells | `tri/reviews/locators.py` (`_find_pairs`, `_split`, `refine`, `note`, `adjust`, `distinct`), `tri/engine/archetypes.py` (`force`, `archetypes`), `tri/infer/situations.py` (`situations`) |
| 4.7, 5.4, 6.5 locators, chasm, power selection | `tri/reviews/locators.py` (`barrier_factor`), `tri/engine/archetypes.py` (`set_locators`, `archetype`, `select_metric`, `_chasm`), `tri/engine/answering.py` (`_rule_homes`), `tri/infer/constraints.py` (`_firing`) |
| 8.5a reviewer consensus | `tri/reviews/consensus.py` (`tally`), `tri/engine/archetypes.py` (`_sync_consensus`, `consensus_report`), `tri/reviews/locators.py` (`set_consensus`, `_consensus_lift`), `tri/infer/situations.py` (`_notes_q`), `tri/views/insight.py` (`consensus_rows`) |
| 4.8, 8.5 blended worth, cross ring | `tri/model/worth.py` (`FieldWorth`, `_cross`) |
| 4.9, 6.10 sidegrades | `tri/analysis/sidegrades.py` (`mine`), `tri/analysis/defs.py` (`SidegradeConfig`), `tri/infer/constraints.py` (`_sides`), `tri/infer/reviewing.py` (`_offers`) |
| 7.3 categories | `tri/engine/reviewing.py` (`set_categories`, stored in the run; no database table since v3.0.2), `tri/infer/completion.py` (`_required`), `tri/infer/constraints.py` (`_lux`, `_cat_kept`) |
| 8.12 devices, latent cache, parallel mining | `tri/model/accel.py` (`resolve`, `train`, `latents`), `tri/model/space.py` (`params_key`, `remember`, `solve_latent`), `tri/engine/persist.py` (`_save_latents`, `_load_latents`), `tri/rules/passes.py` (`_mine_one`, `_n_jobs`) |
| 8.10 new items, sync | `tri/engine/fitting.py` (`extend_catalog`), `tri/model/space.py` (`extend`, `n_fit`), `tri/game/sync.py` |
| 5.3, 5.5, 8.3, 8.4 review ledger | `tri/reviews/authority.py` (`ReviewLedger`, `claims_for`, `vetoes_for`, `binding`), `tri/reviews/defs.py` (`Claim.holds_in`) |
| 6.3 certified rules | `tri/rules/passes.py` (`Rulebook`), `tri/rules/defs.py` (`PassConfig`) |
| 2.5, 5.5 reasons and notes | `tri/core/types.py` (`Reason`, `Review.notes`), `tri/reviews/authority.py` (`build_weapons`), `tri/reviews/defs.py` (`Claim.condition_tokens`) |
| 6.7 indicator notes, anchors | `tri/reviews/locators.py` (`set_indicators`, `_indicate`, `indicated`, `set_anchors`, `_anchor_logk`), `tri/engine/archetypes.py` (`_sync_indicators`) |
| 4.10 note worth, flat values | `tri/infer/situations.py` (`_notes_q`), `tri/engine/reviewing.py` (`set_values`) |
| 6.11 families, variants, styles | `tri/reviews/variants.py`, `tri/engine/archetypes.py` (`_add_variant`) |
| 7.10 central picks | `tri/infer/constraints.py` (`_central`, `_clash_rule`) |
| 2.8 weapon context | `tri/game/weapons.py`, `tri/data/weapons.json`, `tri/core/availability.py` (`build_context`, `enforced_clauses`) |
| 2.9 stat benchmarks | `tri/game/priors.py`, `tri/data/benchmarks.json`, `tri/infer/situations.py`, `tri/model/space.py` |
| 2.6, 2.7 capacity | `tri/infer/defs.py` (`InferConfig.fill_capacity`), `tri/infer/reviewing.py` (`_review_one`), `tri/core/tables.py` (`fit_mantra_slots`), `tri/core/slots.py` (`slots_typed`) |
| 12.5 fields | `tri/views/plots.py` (`region_hulls`, `region_fields`, `html_hulls`) |
| 12.6 rules, facts, triviality | `tri/rules/statements.py`, `tri/rules/triviality.py`, `tri/analysis/facts.py` (`holes`) |
| 6.4 held-out check | `tri/model/space.py` (`_val_split`, `_val_score`, `validation`) |
| 7.3 global review rules | `tri/rules/induction.py` (`RuleInducer`) |
| 7.4 to 7.7 inference | `tri/infer/reviewing.py` (`PassEngine.review`, `_violations`), `tri/infer/completion.py` (`PassEngine.complete`), `tri/core/picks.py` (`Catalog.prereq_order`) |
| 2.10 weapon-class focus | `tri/infer/focus.py`, `tri/infer/completion.py` |
| 8.8 refit | `tri/engine/fitting.py` (`Tri.refit`) |
| 10 patches | `tri/game/patch.py`, `Tri.apply_patch`, `Tri.migrate`, `Tri.drift` |
| 12.1 categories, tree, discovery, quality | `tri/analysis/categorize.py` (`build_atlas`, `categorize`, `soft_kind`, `separating`, `hierarchy`, `discover`, `quality`) |
| 12.2 facts | `tri/analysis/facts.py` (`FactMiner`, `mine_facts`, `benjamini_hochberg`, `FactBook`) |
| 12.3 stream | `tri/analysis/stream.py` (`prequential`, `variogram`, `lowrank`), `tri/infer/active.py` (`coverage`, `kcenter_queue`) |
| 12 graphics | `tri/views/insight.py` |
| Appendix C notation | `tri/rules/symbolic.py` (`pass_rule`, `review_rule`, `fact`, `legend`) |
| 11 measurement | `tri/analysis/evaluate.py` |
| 11.3 to 11.5 scorecard, drafts | `tri/sources/record.py` (`freeze`, `load`, `census`, `score`, `ceiling`, `time_split`), `tri/infer/draft.py` (`as_review`, `render`) |
