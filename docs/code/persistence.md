# 5. Persistence

`Tri.save(path)` writes:

| File | Content |
|---|---|
| `tri.json` | version, `TriConfig`, catalog ids, tokens, data ids, patch log, `metric_selection`, `held_out_ids` |
| `space/` | field arrays (`.npz`) and JSON meta (regions, names, report, `validation`) |
| `data.npz` | the kept corpus `BuildMatrix` |
| `rulebook.json` | every rule with its certificate and status |
| `authored_rules.json` | v3.0: added and suppressed rules in the rule language ([math §6.13](../math/06-accuracy.md)); the selection set is rebuilt from it, the rulebook and the reviews on load |
| `latents.npy`, `latents.json` | the load cache: every corpus build's archetype and the key it was solved under. Optional; a missing or stale cache is solved again and rewritten |

`tri.json` also holds `categories` (the only store of categories since v3.0.2), and `space/` the `fitted` mask.

A run holds no reviews, locators or sidegrades. The caller hands them in after load (`Tri.add_reviews`, `Tri.set_locators`); sidegrades are mined again from the corpus. Old `reviews.jsonl`, `locators.jsonl` and `sidegrades.json` files are ignored. The catalog's availability masks are cached beside the data as `data.avail.npz`, keyed by a catalog digest. v3.3 adds two caches in the model folder, both keyed by content and safe to delete: `avail-<hash>.npz` (the availability groups after a catalog extension, the newest 4 kept) and `sidegrades.cache.json` (mined sidegrades). The first start after a game-data change computes them once.

`Tri.load` reads the latent cache before computing coordinates, which is most of the load time. `tri.load_info["latents"]` says `cache` or `solved`. `Tri.load(path, catalog)` refuses a catalog with different ids; load with the old one, then `migrate`. Runs from `tri.cli.fit` live in `runs/<name>/` with `model/`, `catalog.json`, `report.json`, `metrics.json`, `regions.txt` (and `tri.cli.examine` writes the rulebook to `analysis/`: `rules.txt`, `rules.json`); `runs/LATEST` points to the newest. v1 runs cannot be loaded.

## 5.1 Memory and the run store (v3.7)

Loading keeps the corpus availability as groups: `BuildMatrix.masks` is one row per distinct pattern and every per-build product goes through `group_dot` / `mat_groups` in chunks of `MASK_CHUNK` groups, so no builds-by-items dense table is formed. `MaskStack` reads two mask tables as one after `append`. `FieldWorth` and `ClassPrior` index the corpus coordinates through `RowStack` instead of copying rows. Coordinates are stored as `float32` unless `TRI_COORDS=float64`. `tri/runtime.py` returns freed memory to the system (`release`) after load and after each memory rebuild. `save` writes a placeholder availability table for a non-empty corpus; `load` recomputes it (cached in `data.avail.npz`).
