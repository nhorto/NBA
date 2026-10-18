# Courtside — design

A local NBA app with two jobs:

1. **Today**: win probabilities (later spread/total) for each of today's games.
2. **Last night**: a morning recap that goes beyond the box score. It covers turning points, runs, clutch play, high-impact role players, team trends, and a short storyline write-up.

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

- `pregame`: 10:00, 13:30 (after the 11am–1pm game-day injury resubmission), 17:00, and 18:30. Each run refreshes the schedule (cached daily), upserts today's games, and snapshots ESPN odds, cdn.nba.com odds, ESPN injuries, and the official injury report.
- `postgame`: 03:00 and 09:00. For each game on the previous NBA day that isn't fully archived yet, fetch the cdn.nba.com boxscore and playbyplay plus the ESPN summary, and mark the game final. The job is idempotent.

## Roadmap

1. **Collector** (before 2026-10-20): the pieces above, deployed on the Mac mini.
2. **Honest model**: multi-season history from stats.nba.com `leaguegamelog` (one call per season) and box scores. Elo carried across seasons. Rest, back-to-back, and travel features. Injury-adjusted availability built from the archived snapshots. Chronological backtest scored on log-loss and Brier, against "home team", "Elo", and the closing line.
3. **Recap**: normalize the box scores and play-by-play, then compute win-probability swings, runs, clutch stats, an impact score for role players, and team trend flags. An LLM writes the storyline from structured facts only.
4. **Web page**: Today and Last Night tabs, served locally and reachable over Tailscale.
