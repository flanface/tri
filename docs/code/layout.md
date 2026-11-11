# 1. Project layout

tri is the library package `tri` plus this wiki. (The repository also carries a front end package, `bot`, which these docs do not cover.) Each library area is a subpackage. Inside a subpackage, `defs.py` holds the **definitions** (configurations, value types and constants with no behaviour of their own) and the other modules hold the **methods** (the algorithms). The three largest classes are split by concern into mixin modules, so every file stays readable on its own:

| Class | Lives in | Method modules (mixins) |
|---|---|---|
| `Tri` | `tri/engine/facade.py` | `fitting.py`, `archetypes.py`, `reviewing.py`, `answering.py`, `persist.py` |
| `PassEngine` | `tri/infer/engine.py` | `situations.py`, `constraints.py`, `completion.py`, `reviewing.py` |
| `Catalog` | `tri/core/catalog.py` | `availability.py`, `picks.py`, `slots.py` |

A mixin holds only methods; the class keeps its constructor and state, and inherits the mixins (`class Tri(_Fitting, _Archetypes, _Reviewing, _Answering, _Persist)`). Use the public class; the mixins are internal.

## 1.1 Tree

```
tri/                              repository root
├── README.md                     overview, quick start, wiki entry
├── pyproject.toml                packages "tri", "bot" 3.8.0; numpy, scipy, scikit-learn; extras: render, bot, accel
│
├── tri/                          the library (no I/O beyond explicit save / load)
│   ├── __init__.py               lazy public API (PEP 562); __version__
│   ├── version.py                VERSION
│   ├── runtime.py                v3.7: release() returns freed memory to the system; rss_mb, peak_mb; v3.8: cap_threads()
│   ├── data/                     bundled JSON tables; data.path(name)
│   │   ├── voi_only.json         talents and mantras only obtainable in Vow of Iron (math §2.2)
│   │   ├── mantra_slots.json     mantra slot types and slot grants (math §2.6)
│   │   ├── mantra_types.json     fallback mantra types for old catalogs
│   │   ├── context_locks.json    oath / murmur / race / weapon locks of items
│   │   ├── weapons.json          weapons and weapon clauses (math §2.8)
│   │   ├── benchmarks.json       stat benchmarks (math §2.9)
│   │   └── entries.json          talent and mantra entries for /talent and /mantra
│   │
│   ├── core/                     the data model
│   │   ├── types.py              defs: Kind, Category, Action, Strength, Reason, Item, Build, Operation, Review
│   │   ├── context.py            context tokens of a build; lock atoms
│   │   ├── tables.py             bundled tables: mantra types and slots, Voi-only, context locks
│   │   ├── catalog.py            Catalog (fields, JSON) + mixins:
│   │   ├── availability.py         stats, stages, context locks, prerequisite propagation, masks
│   │   ├── picks.py                exclusions, prerequisites, prerequisite-first order
│   │   ├── slots.py                mantra slots, talent cap
│   │   ├── codec.py              validate_review, JSON forms, text helpers
│   │   └── fast.py               sqdist: squared distances by BLAS expansion
│   │
│   ├── game/                     game knowledge
│   │   ├── defs.py               defs: GameItem, ReqParseError, Patch
│   │   ├── gamedata.py           pocamind/data (default) or + api.deepwoken.co -> Catalog; granted talents; coverage
│   │   ├── fix_types.py          v3.0.3: `python -m tri.game.fix_types runs/<name>` patches catalog.json (Soul Beam type:Oath)
│   │   ├── weapons.py            forced weapon context, weapon clauses, weapon scale (math §2.8)
│   │   ├── weapon_rules.py       hardcoded Speed Demon / Flaming Flourish / Turtle Shell rules (math §2.8c)
│   │   ├── priors.py             stat benchmarks and weapon scale as fixed logit priors (math §2.9)
│   │   ├── patch.py              PatchLog, catalog diffs, remapping
│   │   └── sync.py               follow pocamind/data without refitting
│   │
│   ├── sources/                  data in and out
│   │   ├── defs.py               defs: PipelineConfig, ParsedRecord, Record, Mark, granted-name tables
│   │   ├── pipeline.py           planner export -> PretrainSet (optional; your pipeline may call Tri.fit directly)
│   │   ├── record.py             the frozen review record; sheet scoring
│   │   └── dwt.py                dwt.sh builds <-> planner records (links, stages, proposals; no network)
│   │
│   ├── model/                    the learned model
│   │   ├── defs.py               defs: SpaceConfig, FieldPoint, ConceptReading, WorthConfig, LocalRates
│   │   ├── matrix.py             BuildMatrix: builds as sparse matrices + availability masks
│   │   ├── space.py              ConceptSpace: the worth field and the concept field (math §3, §4)
│   │   ├── worth.py              FieldWorth: field blended with kernel-weighted neighbours (math §4.8)
│   │   └── accel.py              optional PyTorch path (CUDA, Metal, CPU)
│   │
│   ├── rules/                    rules and statements
│   │   ├── defs.py               defs: PassConfig, Condition, Ctx, Case
│   │   ├── passes.py             Rulebook: certified take / never rules (math §6.3, §6.12)
│   │   ├── groups.py             v3.0 groups G(c), D(k); grounded stats; related items; v3.0.1 dependants, tautological groups; rule tokens
│   │   ├── language.py           v3.0 rule language: parse, format_rule, rule_id (math §6.13)
│   │   ├── authored.py           v3.0 AuthoredRules: added and suppressed rules (authored_rules.json)
│   │   ├── selection.py          v3.0 build_selection: mined + authored + review evidence; v3.0.1 review_rules; v3.0.2 availability, endorsement rules (math §6.12)
│   │   ├── induction.py          RuleInducer: typed rules from reviews
│   │   ├── statements.py         hard + mined + review rules; item lookup; symbol keys
│   │   ├── symbolic.py           symbolic notation (math Appendix C)
│   │   ├── registry.py           v3.4 every non-worth rule: ids, sources, drops (math §2.11)
│   │   └── triviality.py         triviality baselines, group baselines and cull (math §12.6)
│   │
│   ├── infer/                    answers
│   │   ├── defs.py               defs: InferConfig, Change, Alternative, situations, QueueItem, DraftSheet
│   │   ├── engine.py             PassEngine (constructor) + mixins:
│   │   ├── situations.py           reading a build: regions, local worth, class focus, waivers
│   │   ├── constraints.py          cap, slots, feasibility, rules, review memory, sidegrades
│   │   ├── completion.py           completion (math §7.6)
│   │   ├── reviewing.py            review of full builds, change lists, alternatives (math §7.4)
│   │   ├── focus.py              weapon-class focus (math §2.10)
│   │   ├── draft.py              an analysis as a review sheet
│   │   ├── caution.py            held marks, questions, answer book (math §14)
│   │   └── active.py             ReviewQueue: which builds experts should see next
│   │
│   ├── reviews/                  expert reviews
│   │   ├── defs.py               defs: Claim, Supersession, ReviewWeights, LedgerConfig, ConsensusConfig, Tell,
│   │   │                         LocatorConfig, Locator, ArchetypeReading
│   │   ├── authority.py          ReviewLedger: claims, reach, newest wins, binding
│   │   ├── consensus.py          reviewer consensus per archetype family (math §8.5a)
│   │   ├── locators.py           LocatorSet: named archetypes, readings, the chasm, twin split
│   │   └── variants.py           families and variants ("Gank / Frost Mage")
│   │
│   ├── analysis/                 reading the field
│   │   ├── defs.py               defs: Atlas, Fact, SidegradeConfig, Sidegrade
│   │   ├── categorize.py         frame / archetype / variant per build (math §12.1)
│   │   ├── facts.py              certified facts (math §12.2)
│   │   ├── sidegrades.py         equal-worth alternative pick clusters
│   │   ├── stream.py             prequential replay, variogram, low-rank residual test (math §12.3)
│   │   └── evaluate.py           benchmarks
│   │
│   ├── views/                    graphics (matplotlib; optional extra)
│   │   ├── plots.py              builds on the concept space, items on the worth space, themes, GIFs, HTML
│   │   └── insight.py            atlas, archetype card, placement, item landscape, stat curves, replay
│   │
│   ├── engine/                   the facade
│   │   ├── defs.py               defs: CATEGORIES, TriConfig
│   │   ├── facade.py             Tri (constructor, state) + mixins:
│   │   ├── fitting.py              fit, refit, worth refresh, calibration, catalog extension, patches, drift
│   │   ├── archetypes.py           locators, regions, consensus, metrics, concept readings
│   │   ├── reviewing.py            ledger, induced rules, memory, categories, values
│   │   ├── answering.py            the pass engine, complete, review, worth, explain, class prior
│   │   └── persist.py              save, load
│   │
│   └── cli/                      command line
│       ├── fit.py                python -m tri.cli.fit: fit on a planner export, save a run under runs/
│       └── examine.py            python -m tri.cli.examine: inspect a run; apply it to a build
│
├── bot/                          front end (not covered here)
│
└── docs/                         this wiki (start at Home.md)
```

## 1.2 Import rules

- `import tri` is light. Names such as `tri.Tri` or `tri.Catalog` load their module on first use, so callers can read the data model without scipy or scikit-learn.
- Inside the library every import is absolute (`from tri.core.types import Build`).
- `defs.py` modules import only from `core`, other `defs.py` modules and the standard library, so definitions never pull in algorithms.
- Mixins import their helpers from `defs.py`, never from the class module, so there are no import cycles.

## 1.3 Where to find a name

| Looking for | Module |
|---|---|
| a configuration (`XConfig`) | the area's `defs.py` |
| a result type (`Alternative`, `Change`, `FieldPoint`, `Claim`, `Fact`, ...) | the area's `defs.py` |
| a `Tri` method | `tri/engine/<concern>.py` (table above) |
| a `PassEngine` method | `tri/infer/<concern>.py` |
| a `Catalog` method | `tri/core/availability.py`, `picks.py` or `slots.py` |
| a formula's code | [math Appendix B](../math/appendix-b-code.md) |
