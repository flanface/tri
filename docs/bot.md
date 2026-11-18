# Bot setup

The Discord bot and the Supabase database are the front end. The bot is a separate distribution, `tri-bot` (`bot/pyproject.toml`), which depends on `tri[render]==4.0.0` and imports only `tri.api`. All network access lives in the bot: Discord, Supabase, dwt.sh, deepwoken.co and the pocamind/data downloads.

## Install

```
pip install ".[render]" ./bot
cp .env.example .env        # fill in; never commit .env
python -m bot
```

## Environment

Required:

| variable | value |
|---|---|
| `DISCORD_TOKEN` | bot token (Discord developer portal, Bot) |
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | service-role key (`SUPABASE_KEY` also accepted) |

Model:

| variable | default | value |
|---|---|---|
| `TRI_RUN` | `runs` | run dir, or a runs folder with `LATEST` |
| `TRI_RUN_REMOTE` | empty | `latest` or a run name: download the run from Supabase Storage into `TRI_RUN` at start and on `/model reload` |
| `TRI_RUN_BUCKET` | `tri-runs` | Storage bucket of published runs |
| `TRI_COORDS` | `float32` | `float64` stores the corpus coordinates exactly |
| `TRI_RELOAD_SWAP` | `0` | `1` keeps serving the old model until the new one loads |
| `TRI_DEVICE` | unset | compute device, passed to the library as a load override |

Optional:

| variable | default | value |
|---|---|---|
| `TRI_COMMAND_TIMEOUT` | 90 | seconds a command waits for the model |
| `HEALTH_PORT` / `PORT` | unset | answer `GET /` and `/health` (web service hosting) |
| `CLAIM_TTL_MIN` | 120 | minutes a claimed pool build stays reserved |
| `STATE_PATH` | `bot_state.json` | drafts and claims |
| `GUILD_CONFIG_STORE` | `auto` | `supabase`, `local` or `auto` |
| `GUILD_CONFIG_PATH` | `guild_config.json` | local server settings |
| `SYNC_MODE` | `guild` | `guild` or `global` command sync |
| `POOL_TABLE`, `REVIEWS_TABLE`, `GUILDS_TABLE`, `LOCATORS_TABLE`, `VALUES_TABLE`, `RULES_TABLE` | `pool`, `reviews`, `guilds`, `locators`, `item_values`, `tri_rules` | table names |
| `GAMEDATA_SYNC_HOURS` | 12 | hours between pocamind/data checks (0 = off; `/model sync` still works) |
| `GAMEDATA_REF` | `main` | pocamind/data branch |
| `GITHUB_TOKEN` | unset | raises the GitHub API limit for game data checks |
| `USER_APP` | `anyone` | `anyone`, `reviewers` or `off`: user-installed `/analyze`, `/complete`, `/plot` |
| `POOL_POP` | 1 | delete a pool row once its review is stored |
| `PLOT_THEME` | `dark` | `dark` or `light` |
| `TRI_MESSAGE_CONTENT` | 0 | 1 reads inline calls in any message (needs the Message Content intent) |
| `DWT_API_KEY` | unset | needed only to save proposals as dwt.sh builds |
| `DWT_FOLDER` | `tri` | folder for saved proposals |
| `DWT_VISIBILITY` | `unlisted` | `public`, `unlisted` or `private` |

Server roles and channels are set in Discord with `/setup`, not in the environment.

## Runs

```
python -m bot.gamesource fetch runs/gamedata        # download pocamind/data (--source all adds deepwoken.co)
python -m tri.cli.fit --builds builds.json --out runs --name live
python -m bot.runstore publish runs/live            # upload to Supabase Storage (warms caches first)
python -m bot.runstore list
python -m bot.runstore fetch runs --name live
```

On start the bot loads the run, reads reviews, locators, authored rules, admin values and categories from the tables, and passes them to the library. A new review takes effect at once: the bot builds the next state on a fork (`with_memory`) and swaps it in under its lock. Answers keep running on the current state meanwhile.

## Hosting

`render.yaml` defines one Render background worker (`plan: 1c-2g`) with a 5 GB disk at `/var/data`. It sets `TRI_RUN=/var/data/runs` and `TRI_RUN_REMOTE=latest`, builds with `pip install ".[render]" ./bot` and starts with `python -m bot`. Render asks for `DISCORD_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` and `GITHUB_TOKEN` once.

## Evaluation on the live tables

```
python -m bot.evaluate snapshot snap.json
python -m bot.evaluate run snap.json --workers 2 --save rows.json
```

`snapshot` reads the tables once. `run` replays the snapshot into a fresh model and prints the metrics in [evaluation](evaluation.md).
