# Public API

`tri.api` is the one public module. Every name outside it is private and may change between releases. The library takes its inputs as arguments; it never reads environment variables, never opens a network connection and reads and writes local run files only.

## Calls

Every call returns JSON values (dicts, lists, `str`, `float`, `int`, `bool`, `None`).

| call | returns |
|---|---|
| `load_run(run_dir, catalog=None)` | a `Tri` from `<run>/model` and `<run>/catalog.json` (or the catalog given) |
| `with_reviews(tri, reviews)` | a new `Tri` with this review log; `tri` is unchanged |
| `with_memory(tri, reviews, locators)` | a new `Tri` with these reviews and locators; `tri` is unchanged |
| `review_build(tri, build, n=3, archetype=None, locked=(), answers=None)` | up to `n` answers, best archetype first: the smallest edits to a full build |
| `complete_build(tri, build, archetype=None, answers=None)` | one answer: the build filled to its talent cap and mantra slots |
| `dry_insert(tri, review, probes=())` | answers on the review's build and on `probes` before and after the review, the supersessions it causes, the claims it reproduces on its own build and validation messages; `tri` is unchanged |
| `agreement(tri)` | `{review_agreement, rows}`: each active review answered on its own build |
| `evaluate(tri, workers=2, B=1000)` | leave-one-build-out metrics and review agreement, each `{value, se}` ([evaluation](evaluation.md)) |
| `to_json(x)` | dataclasses, enums, numpy values, sets and tuples as JSON values |

An answer holds `talents`, `mantras`, `items`, `changes` (each `{item, action, reason, related, worth, source, review_id, pass_, why}`), `iffy`, `iffy_why`, `capacity` (talent cap, mantra slots), `concept`, `name` (archetype), `plausibility`, `violations`, `sidegrades`, `unrated`, `indicative` and `focus`.

## Inputs

| input | type | where it comes from |
|---|---|---|
| catalog | `Catalog` | `<run>/catalog.json`, or `Catalog.from_json` |
| builds | `Build` | `build_from_dict`, or the builder parsers in `dwt` |
| reviews | `Review` | `review_from_dict`; validated by `validate_review` |
| locators | `Locator` | archetype anchors kept by the caller |
| authored rules | text in the rule language | `Tri.add_rule`, `Tri.set_authored` |
| patches | `Patch` | `Tri.apply_patch` before a refit |
| game data | files | `load_gamedata`, `load_deepwokenco`, `merge_gamedata`; downloading them is the caller's job |

A fitted run stores the model only (`model/`, `catalog.json`, `report.json`, `metrics.json`). Reviews, locators and authored rules are handed in after loading.

## Shared state

`Tri.set_reviews`, `set_memory`, `add_reviews` and `apply_patch` change the object they are called on. `Tri.fork()` returns a copy that shares the fitted model and owns its review state, and `with_reviews`, `with_memory` and `dry_insert` work on a fork. An async caller can therefore build the next state on a fork and swap it in while answers run on the current one.

## Re-exports

`tri.api.__all__` also re-exports the data types (`Build`, `Review`, `Item`, `Kind`, `Reason`, `Strength`, `Category`, `Operation`, `Locator`, `Patch`, `Alternative`, `Change`, `TriConfig`), codecs, game data readers, the rule language (`AuthoredRules`, `RuleSyntaxError`, `statements`), views (`figure`, `plots`, `insight`, `export_depth`, `interactive_page`) and `evaluation`. These names are resolved lazily, so `from tri.api import Build` loads only the data model.

## Fitting

```
python -m tri.cli.fit --builds builds.json --out runs --name live --gamedata PATH
python -m tri.cli.examine runs/live
```

`--gamedata auto` reads the game data already downloaded into `<out>/gamedata/`. The library never fetches it.

## Boundary test

`tests/test_boundary.py` fails if any `tri` module imports `bot`, a network module (`urllib.request`, `http.client`, `socket`, `ssl`, `requests`, `aiohttp`, `httpx`, `discord`) or reads `os.environ`, `os.getenv` or `os.putenv`, or if any `bot` module imports from `tri` other than `tri.api`.
