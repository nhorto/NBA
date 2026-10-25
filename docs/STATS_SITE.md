# Stats site: the Courtside pro dashboard (2026-10-25)

The owner wants Courtside to grow into a professional stats dashboard: teams, players, a player's
history against a team, next-day box scores and league leaders. It's for his own knowledge and
for DFS. Everything is served locally from `courtside.db`, which the nightly jobs keep fresh.
There are no on-click API calls.

Visual reference: `docs/mockups/stats-mockup-2026-10-25.html` is self-contained; open it in a
browser and use `#games`, `#teams`, `#player` or `#leaders`. Its source is
`docs/mockups/stats-mockup-template.html`. The pages copy its look, density and components.

## Owner decisions (interview 2026-10-25)

- **Look:** a dark pro dashboard with dense, sortable tables, league-rank chips (blue best, gray
  middle, red worst, always with the rank number printed) and card panels. Desktop first, but
  it must work on a phone: tables scroll sideways and grids collapse to one column.
- **First version:** team pages + rankings, player pages + versus a team, next-day box scores,
  and league leaders.
- **History:** this season plus the last two (2024-25, 2025-26, 2026-27). A season switch on
  the pages.
- **DFS layer:**
  - DK salary and DK points history in the game logs;
  - defense vs position on team pages;
  - tonight's projection on player pages;
  - real ownership history from uploaded standings (`dfs_contest_ownership`).
- **Versus a team:** the games against that team plus **who guarded him** (`player_matchups`).
- **Extras:** on/off and 5-man lineups (`stints`), play types (`synergy_playtypes`), and shot
  charts (play-by-play x/y). No tracking or hustle pages.
- **One site.** Every page moves onto the new look.
- **Nothing is lost:**
  - The Brief narrative stays on the home page **Today**: headline, "60-second version",
    recaps, player stories, tonight's games and picks.
  - "Tonight's players" (projections + prop cards) becomes **Tonight**.
  - The game deep-dive (`/game/<id>`) keeps all its story sections and gains the box score.
- **Props stay information only:** no over/under/edge/lean/bet wording anywhere.

## Navigation and URLs

Top menu: **Today · Tonight · Games · Teams · Players · Leaders · DFS · Health**, plus a global
search box for players and teams.

| URL | Page |
|---|---|
| `/` , `/brief/<day>` , `/archive` | Today (the Brief), unchanged content |
| `/tonight` , `/tonight/<day>` | Tonight's players (was `/players`, `/players/<day>`). The old `/players/<day>` URLs redirect (301) to `/tonight/<day>` |
| `/games` , `/games/<day>` | scoreboard for a night (default: the latest night with finals; date prev/next) |
| `/game/<game_id>` | existing preview / deep-dive, plus the box score: Traditional / Advanced / DFS views |
| `/teams` | league table: all 30 teams, sortable, rank chips, season switch |
| `/team/<ABBR>` | team page |
| `/players` | player directory: search + sortable list of the season's players |
| `/player/<player_id>` | player page |
| `/leaders` | league leaders with stat / season / per-game / last-N filters, plus team leaders |
| `/api/search?q=` | JSON for the search box: players and teams |

## Data (tables in `courtside.db`)

| Data | Table(s) |
|---|---|
| Box scores | `player_games`, `team_games` (2000-01 →), `player_games_adv`, `team_games_adv` |
| Who started | `player_starts` |
| Bio | `player_index`, `team_rosters`, `team_coaches` |
| Schedule / scores | `games` (2026-27), with `game_facts` for the story pieces |
| Who guarded whom | `player_matchups` (per game), `season_matchups` |
| Tracking / hustle | `player_tracking`, `player_hustle` (per game, 2019-20 →), `player_tracking_ext` (potential assists, drives, touches) |
| Shot quality | `shot_xfg` (per-season expected FG%), via `stats/opportunity.py` |
| On/off and lineups | `stints` (`home5` / `away5` JSON id lists, points and possessions per stint) |
| Play types | `synergy_playtypes` |
| Shot charts | raw play-by-play `var/raw/nba_cdn/playbyplay/<day>/<game_id>.json.gz`; actions with `isFieldGoal == 1` carry `xLegacy` / `yLegacy` (tenths of feet from the rim) and `shotResult` |
| DFS: current-season salary, positions, FPPG | `dfs_salaries` + `dfs_slates` (classic `game_type_id = 70`) |
| DFS: historical salary | `dfs_hist_salaries` (RotoGuru, ends 2021) |
| DFS: real ownership | `dfs_contest_ownership` + `dfs_contests` |
| DFS: our projections | `player_projections` + `player_projection_stats` |
| DK scoring | `courtside.proj.dfs.rules.dk_points` (do not reimplement) |

Known gaps, handled by the data workstream:
- 2026-27 rows are missing from `stints`, `player_matchups` and `synergy_playtypes`;
- play-by-play is archived for only part of 2024-25 and 2025-26;
- there is no DK salary for 2021-22 to 2025-26.

Pages must show a clear empty state ("not collected yet") rather than break when a season has no
rows.

### Data: what the data workstream added (2026-10-25)

**New table `shots` (migration 17, `analytics/shots_store.py`).** One row per field-goal attempt
from the cdn play-by-play. Read shot charts from this table, not from the raw files.
- Columns: `game_id, action_number` (primary key), `player_id, team, season, season_type,
  game_date, period, clock, x, y, distance, made, shot_type, action_type, sub_type, descriptor,
  zone`. Indexes on `(player_id, season)` and `(team, season)`.
- `x` / `y` are `xLegacy` / `yLegacy`: tenths of a foot from the rim. `y` grows towards half
  court (baseline at −52.5, half-court line at 417.5). `x < 0` is the NBA's "left" side.
- `distance` is in feet. `made` is 1 or 0. `shot_type` is `2pt` or `3pt` (the official call).
  `clock` is the time left in the period as `M:SS` (`0:05.4` under a minute).
- `zone` is computed from x / y, one of: `Restricted Area`, `Paint (Non-RA)`, `Mid-Range`,
  `Left Corner 3`, `Right Corner 3`, `Above the Break 3`, `Backcourt`.
  - The restricted area uses the NBA's measured ~4.4 ft boundary, so zone splits match nba.com.
  - 98–99% of shots agree with the feed's own `area` label. The rest are the feed's errors:
    putbacks at (0, 0) labelled Mid-Range, and heaves.
  - Coverage: every archived game from 2019-20 on (2024-25 and 2025-26 are complete).

**New nightly postgame steps (`collect.py`):**
- `stints` and `shots` parse each game archived that night. No network.
- `matchups` fills `player_matchups` from stats.nba.com `boxscorematchupsv3`.
  - One request per finished game, plus a week of catch-up in the current season.
  - If stats.nba.com has not processed a game yet, the next run retries it.
- `season_tables` refreshes `synergy_playtypes` and `season_matchups`.
  - Current season only, at most once per NBA day, and only after a day with games.
  - 41 requests: Regular Season, plus Playoffs from April to June.
  - Each refresh replaces that season's slice, so the table mirrors nba.com.
  - Early in the season nba.com returns few synergy rows (5 isolation players on 10-25,
    against 284 for 2025-26). Show the empty state when rows are few.

**Commands:**
- `courtside backfill stats --season 2026-27` backfills one season. In order: shots, stints,
  matchups per game, then the season tables. Resumable. `--offline` runs only the
  archive-based parts.
- `courtside backfill shots [--season S]` fills `shots` from the whole archive. No network,
  resumable.

**Gaps:**
- **Play-by-play:** 2024-25 and 2025-26 are fully archived: all 1,230 regular-season games,
  84/85 playoff games and 6 play-in games each. The "≈200" figure was a count of day folders,
  not games. `courtside backfill pbp` still fetches any game that is missing.
- **DK salary:** none for 2021-22 → 2025-26. RotoGuru stopped after 2020-21.

### Data: opportunity vs conversion (2026-10-25, second round)

"LeBron got 8 rebounds on 10 chances": chances, conversions, and whether a hot run will hold.

**Tables (migration 18):**
- `player_tracking` / `player_hustle` (2019-20 →) now fill nightly for the current season. The
  rows are identical to `backfill players --per-game`, which `proj/features/tracking.py` (F3)
  reads.
- **`player_tracking_ext`** (new, `analytics/tracking_ext_store.py`): one row per player-game.
  - Columns: `pot_ast` (potential assists), `ast_pts_created`, `passes_made/received`,
    `secondary_ast`, `ft_ast`, `drives` + `drive_fgm/fga/ftm/fta/pts/passes/ast/tov/pf`,
    `touches`, `front_ct_touches`, `elbow/post/paint_touches`, `time_of_poss`,
    `avg_sec_per_touch`, `avg_drib_per_touch`.
  - Source: stats.nba.com `leaguedashptstats` (the nba.com Tracking pages) queried with
    DateFrom = DateTo = one day. A player plays at most one game a day, so each row is his
    game. It is matched to the game by (day, team).
  - Requests: one per day × measure (Passing / Drives / Possessions), so 3 a day.
  - Why a new source: `boxscoreplayertrackv3` has no potential assists or drives (checked on
    raw files 2019-20 → 2025-26), and `boxscorematchupsv3` leaves `matchupPotentialAssists` at
    0 in all 1.9 M `player_matchups` rows.
  - Check: on 2026-03-10, AST matches the box score on 241/241 rows. League potential assists
    are 1.70 × AST that day and 2.07 × in 2026-27 to date.
- **`shot_xfg`** (new, `analytics/xfg_store.py`): `(season, zone, dist_band, action_type, fga,
  fgm, xfg)`. Use it through `stats.opportunity.shot_quality`, not directly.
  - The cell of a shot: `shots.zone` × distance band × shot family (jumpshot / layup / dunk /
    hook / other).
  - Distance bands are 1-ft to 29 ft, then 30-35 / 36-46 / 47+, split 2pt / 3pt. These are the
    projection model's F4 bins.
  - Smoothing: cell → zone × band → zone → league. Then a per-zone calibration, so the league's
    and each zone's xFG% equal the actual FG% by construction.
  - `*` rows are the pooled fallbacks.
  - It is not the projection model's LightGBM xFG. That one is walk-forward, so league xFG
    drifts from the season's FG%, and 2019-20 has no model. It also needs the score margin and
    play qualifiers, which `shots` does not keep, and its fit is heavy.
  - Validation, 2025-26: league xFG% = FG% = 0.470, and xPTS = PTS = 255,405. Rim-heavy bigs
    sit at xFG 0.58–0.71 (Gobert 0.714, Duren 0.614). The best shot-makers are Durant
    (+268 pts over expected), SGA (+234) and Jokić (+219).

**Nightly postgame steps (`collect.py`):**
- `shot_xfg`: after `shots`, refits the day's season (~1 s, no network).
- `tracking` / `hustle`: after `matchups`, the same pattern: one request per finished game,
  archived.
  - Catch-up covers 14 days of the current season, because nba.com processes tracking days
    late: on 10-25 the 10-23 games were still empty.
  - An empty answer is not archived and is retried. Three empty answers in a row stop fetching.
- `tracking_ext`: 3 requests per finished day, plus 14 days of catch-up, at most 30 requests.
  - A day's response is archived only when it covers every team that played. Otherwise it is
    stored and retried.
- A stats.nba.com block fails the step that hit it; the later stats.nba.com per-game steps of
  that run skip without making a request.

**Commands:**
- `courtside backfill xfg [--season S]` fits `shot_xfg` from `shots`: every season in ~6 s, no
  network.
- `courtside backfill stats --season S` now also covers `shot_xfg`, `player_tracking`,
  `player_hustle` (per game) and `player_tracking_ext` (per day).
  - `--offline` parses only the archive.
  - In the current season, an empty per-game payload is never archived, so reruns retry it.

**On the Mac after deploy** (migration 18 applies on the first command):
1. `courtside backfill xfg`
2. `courtside backfill stats --season 2026-27`: ~110 per-game requests plus ~15 per-day
   requests. Games nba.com has not processed yet stay pending; the nightly run picks them up.
3. Optional, for potential-assist history (network only; nothing is archived yet):
   - `courtside backfill stats --season 2025-26`, then `2024-25`.
   - Each is ~600 `leaguedashptstats` requests (~10–15 min at the 0.7 s throttle). Everything
     else is cached.

**Query helpers: `src/courtside/stats/opportunity.py`.** Every formula is in its docstring.
- `rebound_opportunity`, `assist_opportunity`, `shot_quality`, `defense_activity`,
  `regression_flags(conn, season, last_n)` and `league_conversions(conn, season)`.
- **`player_opportunity(conn, pid, season, season_type)`** feeds the player page's "Chances"
  panel. It is cached; the dict shape is in its docstring. It returns None without data.
- Timing on devdata 2025-26: a cold panel takes ~0.15 s per player, plus ~1.5 s once per season
  for the league aggregates. Warm renders are ~0 ms.

**Gaps:**
- Devdata has no `player_tracking_ext` rows yet: the history needs the network backfill above.
- Devdata also has no 2026-27 tracking or hustle rows: they need step 2 on the Mac.
- Pages show the empty state when `has.pot_ast` / `has.tracking` is false.

## Rules for the build

- **Web process safety:**
  - The web process opens the DB **read-only** (see `web/store.py`) and never writes or
    migrates.
  - Heavy aggregates (league tables, ranks, leaders, on/off, lineups, shot bins) are cached in
    memory, keyed on the DB file's mtime and size, so a page render stays fast.
  - Target a warm render under 50 ms and a cold one under 1 s.
- **Code layout:**
  - Query code lives in `src/courtside/stats/`: pure functions that take a `sqlite3.Connection`
    and return plain dicts/lists, with no FastAPI.
  - Web routes and view models live in `src/courtside/web/`.
  - Templates live in `src/courtside/web/templates/`.
- **No build step:** no Node, no CDN. Vanilla JS in `web/static/`.
- **Tests:**
  - Every query function gets a test on a tiny fixture DB built in the test.
  - Every route gets a status and content smoke test.
  - Run only the tests you touch: `TMPDIR=~/Documents/NBA/var/tmp uv run pytest -q tests/<file>`.
    The full suite is run by the orchestrator.
  - `tests/conftest.py` pins model switches; leave it alone.
- **Machine limits:** geekom1 is the owner's desktop with 11 GB. At most one heavy job
  (full test suite or headless browser) at a time, so screenshot checks run one browser at once.
  Use your own headless chromium with `--user-data-dir=/tmp/<yours>`; never attach to the
  owner's browser.
- **Git:** agents never commit. The orchestrator stages explicit paths.
