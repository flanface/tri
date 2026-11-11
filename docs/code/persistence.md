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

## 5.2 Speed (v3.8)

A warm answer is one placement, one neighbour search and the rule passes; everything else is a cache.

- Caches shipped with the run. `python -m bot.runstore publish` loads the run once first (`warm`; `--no-warm` skips it), so `model/avail-*.npz`, `data.avail.npz`, `latents.npy`/`latents.json` and `sidegrades.cache.json` exist and are uploaded. A host that has them loads in seconds; one that lacks them solves the latents, recomputes the availability groups and mines the sidegrades on its first start (minutes on one CPU).
- Neighbour search. `RowStack.d2` screens a float32 base without casting it and recomputes exact distances on the rows that can be among the nearest ([Section 4.8 of the math](../math/04-concept.md)); `worth.smallest` breaks ties by row order.
- Memos, each tied to the state it reads: the ledger is fixed once built, so active claims, the context-free claims and the items a review leaves out are computed once per ledger; locator readings of a point are kept while the locator and anchor arrays are the same objects; the review sidegrade pairs of a situation are kept until `_merge_extra` changes its memory; the rules of an item are indexed once per rule list; the picks and availability of a reviewed build (`Tri._vec`) are kept per field and catalog, and `name_key` per string, so a review rebuild (`set_memory`, `add_review`) no longer solves availability for the same builds thousands of times.
- Sidegrade mining takes every rate over the candidate columns only and splits without a $[n,2,k]$ array.
- Threads. `bot/__main__.py` calls `tri.runtime.cap_threads()` before numpy loads: BLAS and OpenMP pools get as many threads as the cgroup CPU quota allows (`cpu.max`), not one per host core.
- Commands. `model_call` waits for the model lock at most `TRI_COMMAND_TIMEOUT` $-\,5$ s and then answers that the model is busy; a command that timed out is dropped instead of running late under the lock.

