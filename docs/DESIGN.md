# Courtside — design

**Product:** *the NBA Morning Brief*. It should be good enough that people would pay for a subscription. A box score is only a snapshot of one game; the Brief tells you what it means: who's rising, who's sliding, who quietly won the game, and what to expect tonight. Hosting is local for now, but everything else (data quality, model rigor, writing, design) is built to product standard.

## What the Brief contains (the product spec)

1. **Last night**: one card per game.
   - Final score, and our pregame pick versus the result.
   - **The story**: 2–4 sentences generated from structured facts only, so nothing is hallucinated.
   - **Turning point**: the biggest win-probability swing or decisive run, with time and score.
   - **Unsung hero**: the role player whose impact outstripped their points (on/off, plus-minus versus minutes, efficiency, defense and hustle where available).
   - Standout lines, and in-game injuries.
2. **Team pulse**: teams trending up or down. Measures are net rating over the last 10 games versus the season, rating change, streaks, and strength of schedule faced. Also includes a model power ranking.
3. **Player pulse**: who is hot or cold over the last 5/10 games versus their own baseline. It flags role changes (a bench player's minutes or usage jumping, a new starter) and lists the leaders over the last 10 games by an impact metric, not just points.
4. **Injury watch**: new absences, returns, and how each one moves tonight's odds.
5. **Tonight**: every game with win probability, projected margin and total, our number versus the market line, key factors (rest, travel, injuries, form), and confidence. Includes an "upset watch".
6. **Scorecard**: the model's season record and calibration, and how it compares to the market. Showing this openly earns trust.

Quality bar: correct numbers every time, explained in plain English, fast to scan on a phone, and with a consistent editorial voice.

**Data licensing:** the free endpoints (NBA.com, ESPN) are fine for personal and local use. A paid public product would need licensed feeds (e.g. Sportradar or SportsDataIO). Keep every source behind the `sources/` interface so a feed can be swapped without touching analytics.

## Prediction model: full latitude, judged only by honest out-of-sample results

- **Data:** 10+ seasons of team and player game logs (stats.nba.com `leaguegamelog`, team and player, about two calls per season). Historical official injury reports from 2021-22 onward via `nbainjuries`. Schedule-derived rest and travel. Play-by-play for the win-probability model. Historical closing lines from any obtainable archive, used as the benchmark.
- **Ladder:**
  1. Elo carried across seasons.
  2. Player-based team strength: player ratings × expected minutes given availability, so injuries flow straight into the prediction.
  3. Gradient-boosted model on pregame features.
  4. Neural nets or sequence models, only if they beat (3) out of sample.
- **Targets:** predict the margin and total as distributions; derive win probability from the margin distribution. Calibrate it.
- **Evaluation:** walk-forward by season, never a random split. Metrics are log-loss, Brier, margin MAE, calibration, and against-the-spread results versus the closing line. Every model change must beat the current one on the same backtest.
- Heavy training runs on the Mac mini (M4, 32 GB).

The old 2023-24 notebooks live in `legacy/`. Don't import from them. Port the good ideas instead: lagged rolling stats, Elo, and the "missing rotation player" idea.

## Principles

- **Archive first.** Free sources don't keep history for odds or injury reports, so save every raw response, gzipped, before parsing it. Parsers can be rewritten later. Missed snapshots are gone for good.
- **Portable.** Runs on the Mac mini (main, always-on) and on Linux (geekom1). Never hard-code paths. All state lives under `COURTSIDE_DATA`, which defaults to `<repo>/var/` and is gitignored.
- **Polite and robust.** Use one HTTP client with a disk archive, per-host rate limiting, retries with backoff, and explicit detection of HTML/blocked responses. Assume an endpoint can break for days at a time, and keep a fallback source for each need.
- **No leakage.** Predictive features use only information available before tip-off. Evaluation is always chronological.

## Stack

- Python ≥ 3.12, managed with `uv`. Run with `uv run courtside ...`.
- HTTP: `curl_cffi` with `impersonate="chrome"`. This is required: cdn.nba.com and stats.nba.com 403 or hang for plain clients. Send `Referer: https://www.nba.com/` and `Origin: https://www.nba.com` to nba.com hosts.
- Storage: gzipped raw JSON plus SQLite (`$COURTSIDE_DATA/courtside.db`). Use stdlib `sqlite3`, no ORM.
- CLI: stdlib `argparse`, entry point `courtside`.
- Tests: `pytest`. They must never touch the network; use saved fixtures and monkeypatching.
- Later: pandas, scikit-learn, or LightGBM for the model; FastAPI and Jinja for the local web page. For storyline prose, a headless `claude -p` / `codex exec` call through the user's subscription, never a paid API key.

## Layout

```
src/courtside/
  config.py        DATA_DIR, DB_PATH, NBA_TZ ("America/New_York"), nba_day() helper
  http.py          fetch_json(), BlockedError, FetchError
  archive.py       save_raw() / load_raw() / raw_path()
  db.py            connect(), migrate(), upsert helpers
  teams.py         30 teams: nba tricode, espn abbr, team_id, name, arena lat/lon, tz
  sources/
    nba_cdn.py     schedule, scoreboard, boxscore, playbyplay, odds (cdn.nba.com)
    espn.py        scoreboard (incl. odds), summary, injuries (site.api.espn.com)
    injury_report.py  official NBA injury report PDF via `nbainjuries` (optional, needs Java)
  collect.py       pregame / postgame / backfill jobs
  cli.py           `courtside collect pregame|postgame [--date YYYY-MM-DD]`, `courtside db migrate`
deploy/            launchd plists (macOS) + systemd user timers (Linux) + install scripts
tests/             fixtures/ with small trimmed real payloads
```

## Key definitions

- **NBA day**: the US/Eastern calendar date a game is scheduled on. `config.nba_day(dt_utc)` converts. "Last night" means the previous NBA day.
- **game_id**: the NBA 10-digit id (e.g. `0022600001`) is canonical. The ESPN event id is stored alongside and matched by (NBA day, home tricode, away tricode) through `teams.py`. ESPN abbreviations differ: GS/GSW, NY/NYK, SA/SAS, NO/NOP, UTAH/UTA, WSH/WAS.

## Raw archive layout

`$COURTSIDE_DATA/raw/{source}/{kind}/{nba_day}/{key}.json.gz`

- Final per-game data (`boxscore`, `playbyplay`, `summary`) is keyed by game id. Re-fetching overwrites it.
- Snapshot kinds (`odds`, `injuries`, `injury_report`, `scoreboard`) append a UTC timestamp, `{key}__{YYYYMMDDTHHMMSSZ}.json.gz`, and never overwrite.

## SQLite (v1)

- `games(game_id PK, espn_id, season, nba_day, tipoff_utc, home, away, status, home_score, away_score, updated_at)`. Status is `scheduled|live|final|postponed`.
- `odds_snapshots(id PK, game_id, captured_at, source, provider, home_spread, total, home_ml, away_ml, raw_path)`
- `injury_snapshots(id PK, captured_at, source, team, player_name, player_id NULL, status, reason, raw_path)`
- `collect_runs(id PK, job, started_at, finished_at, ok, detail)`

Later migrations will add normalized box score, play-by-play, and prediction tables. Migrations are append-only and tracked in a `schema_version` table.

## Collection schedule (US/Eastern)

- `pregame`: 10:00, 13:30 (after the 11am–1pm game-day injury resubmission), 17:00, 18:30, 20:00, and 22:00 (the last two catch late-tip lines and injury statuses). Each run refreshes the schedule (cached daily), upserts today's games, and snapshots ESPN odds, cdn.nba.com odds, ESPN injuries, and the official injury report.
- `postgame`: 03:00 and 09:00. For each game on the previous NBA day that isn't fully archived yet, fetch the cdn.nba.com boxscore and playbyplay plus the ESPN summary, and mark the game final. The job is idempotent.

## Roadmap

1. **Collector** (before 2026-10-20): the pieces above, deployed on the Mac mini.
2. **History backfill and backtest harness**: multi-season game logs, historical injury reports, the walk-forward evaluation framework, and baselines (home team, Elo, market).
3. **Model ladder** (see above): each rung is kept only if it beats the last.
4. **Analytics**: normalized box scores and play-by-play, the win-probability model, runs, clutch play, the impact metric, and team and player trend detection.
5. **Brief**: the LLM writes from a structured facts JSON, with a fixed editorial style guide and a fact-check pass.
6. **Web app**: a polished, mobile-first Brief with Today, Last Night, Teams, and Players, served locally and reachable over Tailscale.
