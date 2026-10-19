# Research: player predictions and the data landscape (2026-10-19)

This file condenses three research passes: the industry and methodology, probes of player and matchup data, and a wider scan of data sources. Scratch probes are in `/tmp/player_probe/` and `/tmp/data_scan/` on geekom1 (ephemeral).

## Product decisions (from the owner)

- Start archiving player props, referee assignments, line movement and DFS salaries from opening night (2026-10-20).
- v1 projects **points, rebounds, assists and minutes, plus DK/FD fantasy points**. Fantasy scoring also needs 3PM, steals, blocks, turnovers and double-double bonuses, so those are projected internally.
- **A DFS lineup optimizer is in v1.**
- Props are shown as "our number vs the market": information, not leans or betting advice.

## Competitors

| Who | Output | The "why" |
|---|---|---|
| DFS sites (Stokastic, SaberSim, FantasyLabs, RotoGrinders) | Minutes × fantasy points per minute; floor/ceiling (25th/75th percentile); SaberSim simulates each slate 10,000 times; ownership | Defense-vs-position ranks, pace, usage notes |
| DARKO (free) | Daily box-score projections; Kalman filter + GBM; DPM rating | None |
| EPM, LEBRON, BBall Index | Impact metrics; "Matchup Difficulty" | Descriptive |

**Gap:** nobody pairs calibrated distributions with an auditable plain-English why, and **nobody automates the post-game "why he did well"**.

## Methodology best practice

- **Structure:** a minutes model × per-minute or per-possession rate models with empirical-Bayes shrinkage and recency weighting. Simulate jointly from the game-level outcome (pace, blowouts) so the stats correlate.
- **Minutes is the biggest error source.** Its drivers are the injury report, the spread (blowout risk), rest, and coach rotation habits.
- **Usage redistribution:** with/without splits shrunk toward priors. Usage goes mostly to direct replacements.
- **Distributions:** roughly normal for points; negative binomial for rebounds and assists; Poisson-like for 3PM, steals and blocks.
- **Evaluation:** calibration and log-loss of over/under probabilities against no-vig prop lines, not MAE. Rough points MAE is 5–6 for starters (unverified). Target: interval coverage of 78–82% for an 80% interval, and minutes MAE of about 4.
- **Prop markets** are widely described as softer than sides and totals (lower limits, later posting). This is industry consensus, not a peer-reviewed finding.

## Defender effects (temper expectations)

- Defenders mostly change *where and how often* opponents shoot, not their make rate (Franks et al.; Sloan's "Dwight Effect").
- Opponent 3P% on open looks is largely luck.
- Matchup data is noisy per pair. Use it to identify the **primary defender**, then apply a heavily shrunk effect.
- Shot quality: build our own xFG from play-by-play shot location, type and qualifiers. The commercial ones (qSQ, xeFG%) are proprietary.
- Coach style is persistent (e.g. how many players a coach runs 30+ minutes). Estimate coach effects on minutes ourselves.

## Data availability (probed live through curl_cffi)

| Need | Endpoint | Granularity | History | Requests |
|---|---|---|---|---|
| Who guarded whom | `boxscorematchupsv3?GameID` | per game: defender × offensive player (matchup minutes, partial possessions, points, FGM/FGA, 3PM/3PA, switches, help blocks, shooting fouls) | 2017-18+ | 1 per game (~190 KB) |
| | `leagueseasonmatchups` (no player filter) | season, all pairs | 2017-18+ | 1 per season (25 MB; use a 90 s timeout) |
| | `matchupsrollup` | season: defender × opponent position (G/F/C) with % of time | 2016-17+ | 1 per season |
| Defended FG% vs expected | `leaguedashptdefend` (DefenseCategory Overall / 3 Pointers / 2 Pointers / Less Than 6Ft; supports date ranges) | season or day | 2015-16+ | 1 per category |
| Tracking | `boxscoreplayertrackv3` (touches, distance, contested/uncontested FG, defended at rim) | per game | 2013-14+ | 1 per game (~22 KB) |
| Hustle | `boxscorehustlev2` (contests, deflections, charges, screen assists, box-outs) | per game | 2016-17+ | 1 per game |
| Shots | `shotchartdetail`; **our archived cdn play-by-play already has x/y, shotDistance, subType, descriptor, qualifiers, assist/block ids** | per game | 2019-20+ in our archive | 0 extra |
| Defender-distance buckets | `leaguedashplayerptshot?CloseDefDistRange=…` (0-2, 2-4, 4-6, 6+ ft) | season or day, player-level | 2013-14+ | 1 per bucket |
| Play types | `synergyplaytypes` (param `SeasonYear`, not `Season`) | season | 2014-15+ | per play type × off/def |
| Lineups / on-off | `teamdashlineups`, `teamplayeronoffsummary`. **`leaguedashlineups` is capped at 2,000 rows.** Prefer our own stints. | team-season | | 30 per season |
| Bio | `playerindex` (position, height, weight, draft, country; no birthdate); `commonteamroster` (birthdate, age, experience, **coaches**) | | 2015-16+ | 1 and 30 per season |
| Referees | cdn boxscore `officials` (already archived); `boxscoresummaryv3` | per game | | |
| **Props** | ESPN core `…/events/{id}/competitions/{id}/odds/{provider}/propBets?page=N` (25 per page, ~43 pages per game; DraftKings = provider 100, ESPN BET = 58). Over/under semantics unverified. **Live only: DK props 404 after the game.** | per game | live | ~43 pages per game |
| Starters pre-tip | not available (the cdn boxscore 403s before tip) | | | Use injury reports plus a minutes model |

**Broken:** `playerdashptshotdefend` and `playerdashptshots` (HTTP 500). Use the league-wide equivalents instead.

**Backfill estimate for 2019-20→2025-26** (~9,000 games, ~1.2 s per call): matchups + playertrack + hustle ≈ 27,000 calls ≈ 8–9 h, ~320 MB gzipped. The season tables take minutes.

## Wider data scan: ranked additions

1. **sportsdataverse releases** (`github.com/sportsdataverse/sportsdataverse-data/releases`, public, current through 2026-10-18):
   - `nba_stats_game_lineups`, 1997–2026, 94 MB: on-court 5v5 per play-by-play action
   - `nba_stats_possessions`, 1997–2026, 326 MB
   - `espn_nba_pbp`, 2002+, with shot x/y
   - `nba_stats_officials`, 1997+

   These extend stints, on/off, RAPM and win probability back 23 seasons. Validate them against our own 2019+ stints. The `nba_player_impact` RAPM data looks mis-scaled, so don't trust it.
2. **Transactions:** `stats.nba.com/js/data/playermovement/NBA_Player_Movement.json` and ESPN `/transactions`. Use these for a "moves" section in the Brief.
3. **National TV:** `nationalTvBroadcasters`, already in `scheduleLeagueV2`.
4. **Referee assignments:** `official.nba.com/referee-assignments/`, posted about 9:00 ET on game days; plus hoopR officials history. The home-foul bias is documented, but the effect on totals beyond what the market already prices is unproven. Test it on residuals.
5. **Action Network scoreboard API** (undocumented): per-book lines with timestamps. Poll it to build line-movement history.
6. **RAPTOR** (fivethirtyeight/data, CC BY 4.0, through 2022): useful as a benchmark or prior.
7. ESPN news JSON and Wikidata, for storytelling.
8. Kaggle `eoinamoore` (CC0, box scores since 1947, updated nightly).
9. pbpstats API, to cross-check possessions.

**Skip:**
- Kaggle `wyattowalsh` (CC BY-SA, older play-by-play)
- DARKO, EPM, LEBRON and Cleaning the Glass (permission or paywall)
- Basketball-Reference (its ToS forbids scraping)
- public betting percentages (null without login; a weak signal)
- Reddit and X (cost and terms)
- HoopsHype (blocks AI crawlers)

## Fantasy and DFS

- Historical DK salaries and points: RotoGuru archives (depth unclear). Contest and ownership history is mostly paid.
- **Log DK/FD salaries daily from now.** Fantasy points are a deterministic function of the box score.
- **Optimizer:** integer programming (`pydfs-lineup-optimizer` supports NBA DK/FD). Research: Hunter, Vielma & Zaman, "Picking Winners Using Integer Programming" (GPP portfolios that maximize expected score under variance and correlation constraints). Simulation-based builders use correlated outcomes and ownership leverage.
- Season-long: Sleeper has a free read-only API; ESPN has the unofficial `espn-api` (cookies for private leagues); Yahoo uses OAuth; ADP is on FantasyPros.
- No public archive of historical projections exists, so benchmark against prop lines.

## Licensing (for a paid product)

- **stats.nba.com and cdn.nba.com** (and the hoopR data derived from them) are undocumented endpoints whose terms restrict commercial reuse. That's fine for personal use; they need a license before any paid product.
- **ESPN and Action Network APIs** are unofficial, personal use only.
- **Licensing options:** balldontlie GOAT (~$40/mo per sport, includes props and lineups), Sportradar via its Acceleradar startup program, MySportsFeeds.
- Clean licenses: RAPTOR (CC BY), Kaggle eoinamoore (CC0), and the Kaggle odds data (CC0).
