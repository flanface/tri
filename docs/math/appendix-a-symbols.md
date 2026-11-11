# Appendix A. Symbols and configuration

| symbol | meaning | config field (default) |
|---|---|---|
| $N$ | items in the catalog | |
| $d$ | archetype dimensions | `SpaceConfig.dim` (16) |
| | stat-step grid | `SpaceConfig.stat_grid` (5, ..., 95) |
| $\lambda_r,\lambda_\theta$ | L2 strengths | `l2_latent` (5.0), `l2_items` (0.01) |
| | pre-patch weight | `patch_factor` (0.25) |
| $h$ | worth, region and locator bandwidth | `bandwidth` (0.6) |
| | regions: most, selection, grid, stability, merge, mixed | `n_regions` (24), `region_select` (`"stable"`), `region_grid` (8, ..., 24), `region_ari` (0.8), `region_merge` (0.5), `mixed_lift` (1.2) |
| | neighbours, prior strength | `WorthConfig.k` (150), `prior_strength` (15) |
| | weight of reviewed items | `review_weight` (40) |
| | rule thresholds take, never, contrast | `PassConfig` (0.80, 0.05, 0.10) |
| $z$ | Wilson confidence | `z` (1.645) |
| | support, uniformity support, share | `min_support` (30), `uniform_support` (5), `uniform_share` (0.02) |
| | veto repair, leave item out, aggregates | `repair_steps` (2), `leave_item_out` (True), `aggregates` (True) |
| | take overlap, lift, contrast ratio, multiple testing | `max_overlap` (0.8), `min_lift` (0.1), `contrast_ratio` (3), `multiple_testing` (`"bonferroni"`) |
| | stat step of thresholds | `stat_step` (5) |
| | breadth threshold, clusters | `min_breadth` (0.5), `breadth_regions` (16) |
| $s_{\min}, s_{\text{full}}, s_{\text{same}}$, reach | review similarity bands | `LedgerConfig` (0.80, 0.95, 0.99, 1.2) |
| $s_{\text{bind}}$ | similarity at which a transferred item claim binds (5.5) | `bind_sim` (0.95) |
| $\varepsilon$, $T$ | echo weight, tri-made builds (5.5) | `echo_keep` (0.25), `made_by_tri` |
| | refit weight, share | `refit_review_weight` (10), `refit_review_share` (0.25) |
| | swap margin, no-fit, iffy margin | `InferConfig` (0.15, 0.05, 0.08) |
| | binding, nudge | `memory_hard` (0.5), `memory_nudge` (0.3) |
| | compute device, device memory, smallest device solve | `SpaceConfig.device` (`"auto"`), `device_budget_gb` (2.0), `device_min_rows` (256) |
| | rule mining processes | `PassConfig.n_jobs` (0 = auto) |
| | talent cap base, talents per counted mantra, exempt types | `Catalog.talent_cap_base` (76), `cap_per_mantra` (2), `cap_exempt` (Oath, Origin, Monster) |
| $\omega$ | metric weights | `SpaceConfig.metric` (`"worth_variance_dedup"`, `"worth_variance"` or `"uniform"`) |
| $\tau$ | redundancy threshold | `dedup_tau` (0.9) |
| $\gamma$ | tempering exponent | `metric_power` (0.5) |
| | weight floor | `metric_floor` (0.02) |
| | calibration, neighbour rank | `metric_calibrate` (True), `metric_k` (10) |
| | held-out share, size bounds, mask rate | `val_frac` (0.05), `val_min` (50), `val_max` (2000), `val_mask` (0.2) |
| | locator reach, confidence | `LocatorConfig.reach` (1.2), `min_confidence` (0.5) |
| $\beta$ | chasm strength | `LocatorConfig.barrier` (1.0) |
| $A_b$ | archetype family read at $b$ (6.12) | `PassConfig.archetypes` (True) |
| $G(c), D(k)$ | category and derivation-tree groups (6.12) | `PassConfig.groups` (True), size 3, core needs 2 |
| $\Sigma(x)$ | grounded stats, related anchors, fallback (6.12) | `grounded` (True), `anchors` (`"related"`), `ground_fallback` (True) |
| | rule depth, replication split, literal necessity (6.12) | `max_depth` (3), `replicate` (True), `necessity` (True) |
| | refinement, corpus check, keep weight (6.12) | `SelectionConfig` (True, True, 1.0) |
| $\tau(t), \tau_b(t)$ | tier gating, fallback, tier read on the build (6.12) | `InferConfig.rule_tiers` (True), `tier_fallback` (True), `tier_context` (True) |
| $\alpha_b(t)$ | review-rule denominators: builds that can take $t$ (6.12) | `SelectionConfig.avail_only` (True), `found_min` 8, `found_share` 0.75, `family_share` 0.75, `keep_rules` (True), `keep_min` 4, `keep_share` 0.75 |
| $\gamma$ | chasm power, hard chasm, corpus and ring gate (5.2) | `chasm_power` (6.0), `chasm_hard` (True), `chasm_corpus` (True) |
| $m_H$ | rule home weight, reference sample (5.2a) | `chasm_rules` (True), `chasm_rule_min` (5.0), `chasm_rule_sample` (30000) |
| $\kappa$, $\omega$ | consensus shrinkage, edit weight (8.1a) | `consensus_kappa` (1.5), `consensus_op_weight` (0.5), `consensus_min_reviewers` (2), `consensus_unit` ("build") |
| $\eta_c$, $\lambda_c$ | consensus worth, reading lift, floor (8.1b) | `consensus_worth` (0.35), `consensus_lift` (2.0), `consensus_floor` (0.2); central at `InferConfig.consensus_central` (0.3) |
| $\kappa$, $\theta$ | tell shrinkage, tell threshold | `refine_prior` (4.0), `refine_min_logodds` (1.5, about $\log4.5$) |
| $\tau$, $\mu$, $c$, $g$ | twin overlap, pair mass, likelihood clip, held-out gain | `twin_overlap` (0.2), `twin_mass` (0.5), `twin_clip` (4.0), `twin_min_gain` (1) |
| $\lambda$, $\varphi$ | indicator lift, indicator floor (6.7) | `indicator_lift` (6.0), `indicator_floor` (0.05) |
| $w_a$, $v$ | anchor weight, anchor veto (6.8a) | `anchor_weight` (4.0), `anchor_veto` (0.9) |
| $k_\times$, $h_\times$, $\lambda_\times$ | cross-archetype ring (4.5b) | `WorthConfig.k_cross` (600), `cross_bandwidth` (1.5), `cross_weight` (0.15) |
| $\eta$ | note worth, floor, flat value cap (4.6), (4.7) | `InferConfig.note_worth` (0.35), `note_floor` (0.02), `value_cap` (0.5) |
| | split on, locators per twin | `refine` (True), `refine_min_locators` (2) |
| | unrated support | `InferConfig.cold_support` (1.0) |
| | weapon-class focus (2.10) | `InferConfig.focus` (on), `focus_k` (150), `focus_min_labelled` (20), `focus_min_q` (0.3) |
| | sidegrade band, gap, share, commitment | `SidegradeConfig.contested` (0.15, 0.85), `gap` (0.35), `min_share` (0.2), `min_commit` (0.7) |
| | sidegrade correlations, pools | `min_cross` (0.2), `min_within` (0.2), `pair_overlap` (0.6), `core_rate` (0.6), `core_hold` (0.8) |
| | power grid and selection | `metric_grid` (0, ..., 1), `select_z` (2.0), `select_min_gain` (0.01), `select_min_per_archetype` (3), `select_min_archetypes` (2) |
