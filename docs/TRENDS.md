# Courtside trends engine: team pulse and player pulse

This is the engine behind the Brief's sections 2 (**Team pulse**) and 3 (**Player pulse**).
The code lives in `src/courtside/trends/`. Every entry point takes `as_of: date` and reads only
games with `game_date < as_of`. That filter is applied in SQL, so the output is reproducible
and leak-free.

```
team.team_pulse(conn, as_of, season=None)     -> list[TeamTrend]    (sorted by power rank)
player.player_pulse(conn, as_of, season=None) -> dict
summarize.league_facts(conn, as_of)           -> dict  (JSON facts document for the writer)
calibrate                                     -> historical calibration/validation (offline)
synthetic.make_league(conn, ...)              -> tiny schema-true leagues for tests
```

- **Data.** The engine reads `team_games` and `team_games_adv` (possessions), and
  `player_games` and `player_games_adv` (usage). Only regular-season games are used.
- **Window.** Each call loads the current and prior season and nothing else. The one exception
  is the career-high lookup (see below), which is indexed and cached.
- **Seasons.** `season_of(d)` uses July as the cut, so `2026-02-15` falls in `2025-26`.
- **Read-only access.** `common.connect_ro()` opens the DB read-only. It never runs migrations,
  so it's safe while collectors write.
- **Demo.** `uv run python -m courtside.trends.summarize 2026-02-15` prints the document.

## Team pulse (`team.py`)

1. **Ratings: ridge SRS on offense and defense.**
   - Each team-game is one equation: `ORtg(t vs o) − L = off_t + dal_o + h·hca/2`. `dal` is
     points allowed above average, so positive means bad defense. `h` is +1 home, −1 road and
     0 at neutral sites. Net = off − dal.
   - Each team is pulled toward its **prior-season** fit × `RHO = 0.6`, with weight
     `K_PRIOR = 16` pseudo-games. Home court has a 2.5 prior.
   - Possessions are the mean of the two adv rows. If adv is missing, they are estimated as
     FGA − OREB + TOV + 0.44·FTA.
2. **Per-game adjustment.** Every game's ORtg and DRtg is corrected for the opponent's fitted
   offense and defense and for venue. These adjusted values feed the windowed numbers: the
   last-10 and last-5 adjusted ratings, and the strength of schedule (the mean opponent
   adjusted net over the last 10, ranked 1 = hardest).
3. **Trend.**
   - The **baseline** is the team's games *before* the last-10 window blended with `K_PRIOR`
     pseudo-games of prior.
   - `delta` = adjusted last-10 net − baseline.
   - `z = delta / (SIGMA_NET·√(1/10 + 1/(n_before + K)))`, with `SIGMA_NET = 12.2`.
   - `score = delta·τ²/(τ² + noise var)`, with `τ = 1.5`. This is the part of the gap expected
     to persist.
4. **Flag.**
   - `rising` / `sliding` needs all three: `|z| ≥ 2.25`, `|delta| ≥ 4.0`, and at least 15 games
     played.
   - Before 15 games, `trend.suppressed` explains why there is no flag.
5. **Reasons** are machine-readable `{kind, value, detail}` entries:
   - `net_trend`
   - `defense_collapse` / `defense_surge` / `offense_surge` / `offense_slump`. These need the
     adjusted side z ≥ 1.5 and ≥ 3 pts. The detail quotes raw DRtg/ORtg, last 10 vs season.
   - `streak` (≥ 4 games)
   - `last10_record` (8+ wins or losses)
   - `schedule_tough` / `schedule_soft` (opponent adjusted net beyond ±2)
   - `home_road_split` (≥ 15 pts/100)
   - In `league_facts`, flagged teams also carry `key_absences`: recently missing players who
     averaged ≥ 24 mpg.
6. **Power ranking.**
   - `power = rating + 0.10·(last-10 adj net − rating)`, where `rating` is the shrunk season
     fit.
   - `rank_7d_ago` and `change_7d` are recomputed for `as_of − 7 days`.

## Player pulse (`player.py`)

**Courtside Impact (CI)** is computed per game, in Game Score units:

```
CI = GmSc + 0.5·(PTS − 2·TSA·par_TS) + 0.15·REL
par_TS = league_TS(season) − 0.25·(USG − 0.20)
REL    = +/- − MIN·(team margin / game minutes)
```

- **Game Score** is the base.
- **Efficiency term.** Game Score charges a missed shot only 0.7, so it pays for volume. The
  efficiency term adds or removes half a point per point above or below a usage-adjusted par
  true shooting. High-usage creators face a slightly lower bar.
- **REL** is the on-court half of on/off: +/- relative to the team's own margin per minute in
  that game. It's single-game noise, so its weight is small, but it accumulates over 10 games.
  This is the "box score doesn't tell everything" term.
- CI is linear, so a window CI is just the mean of the game CIs.
- Every number the writer sees is also shipped raw: pts/reb/ast/stocks/3PM/TS%/USG%/+/-.

**Hot and cold.**
- **Eligibility:** ≥ 8 season games, ≥ 20 mpg in the season and ≥ 15 over the last 5, and
  played within the last 10 days.
- **z** is computed for each of the last 5 and last 10 games, against a baseline of the
  player's earlier games this season plus 10 pseudo-games of his prior-season CI. The larger
  `|z|` is kept.
- **Spread.** The game-to-game sd uses the player's own residuals from this season and the
  last, shrunk toward 6.5 over 15 pseudo-games, with a floor of 4.
- The **top 5** with `z ≥ 2.0` (or `≤ −2.0`) are listed.

**Role changes.**
- A player belongs to the team of his latest appearance.
- There are no DNP rows. A team game, seen through teammates' rows, that has no row for the
  player counts as a missed game.
- Games before the player joined the team are excluded from his baseline.
- Kinds:
  - `minutes_up` / `minutes_down`: ±8 mpg, last 5 vs earlier, on games played.
  - `usage_up` / `usage_down`: ±5 points of usage, with ≥ 20 mpg.
  - `new_in_rotation`: ≥ 15 mpg over the last 5 after < 8 mpg per team game, counting DNPs
    as 0.
  - `starter_minutes` / `lost_starter_minutes`: entered or left the team's top 5 by minutes.
    The earlier rank is based on minutes per game *played*, so injury DNPs don't demote anyone.
    **This is a proxy.** The live cdn boxscore has a `starter` flag that should replace it.
  - `new_team` / `new_arrival`: trades and signings.
  - `out_of_rotation`: a ≥ 20 mpg player has missed ≥ 3 straight team games. That could be
    injury, rest or a coach's decision, so cross-check Injury watch. The facts document keeps
    only absences of ≤ 10 games.

**Leaders, last 10.**
- The window is the player's current team's last 10 games, with ≥ 6 appearances.
- Categories: impact (with ≥ 20 mpg), pts, reb, ast, stocks, 3PM, and TS% (with ≥ 12 TSA
  per game).

**Streaks.**
- Active streaks over consecutive regular-season appearances, **carried across seasons**
  (Gilgeous-Alexander's 20-point streak began in 2024-25). A streak that covers every loaded
  game is reported with `at_least`.
- Minimum lengths: 20-pt games 5, 30-pt games 3, double-doubles 5, triple-doubles 2.

**Career highs.**
- The test is a value greater than every earlier regular-season game since 2000-01. The
  current-season part of that is vectorized.
- The pre-season maxima come from one indexed query over this season's players (~0.25 s),
  cached per (db, season).
- Reporting floors: pts 20, reb 12, ast 10, 3PM 5. Players with < 82 prior games need 1.5× the
  floor.
- Careers that began before 2000 would be incomplete; none are active.

## Facts document (`summarize.league_facts`)

**Top-level keys:** `schema, as_of, season, data_through, games_per_team_median, early_season,
notes, method, teams, players`.

**`teams`:** `rising`, `sliding`, `power_top10`, `power_movers_7d`, `streaks` (5 or more).

**`players`:**
- `hot`, `cold`
- `role_changes`: grouped per player, at most 6 per headline kind and 20 in total
- `leaders_last10`
- `streaks`
- `career_highs_last7d`

**`method`** carries the rules and the measured historical validity, so the writer can hedge
correctly ("trending up, and teams like this keep it up about 60% of the time").

- The document is about 30 KB mid-season.
- Generation takes **1.5–2.0 s** on geekom1, including the DB read.
- Early in the season the document is suppressed (see below).

## Calibration and validation (`python -m courtside.trends.calibrate`, about 2 minutes)

Constants were **fit on 2001-02..2014-15** and **validated out of sample on
2015-16..2024-25**. The checkpoints are weekly, from the point where the median team has 10
games until 10 games remain. The "next 10" outcome is opponent-adjusted using end-of-season
ratings; that is evaluation only.

**Fitted constants:**
- **`SIGMA_NET` 12.2.** This is the sd of a game's adjusted net around the team's season
  rating. `SIGMA_SIDE` is 10.1.
- **`RHO`, `K_PRIOR`.** A grid search minimized the RMSE of the rest-of-season adjusted net
  given the first n ∈ {0, 5, 10, 20, 30} games plus the prior. The best was ρ = 0.6, k = 16,
  with RMSE 3.28.
- **`τ` (real 10-game form shifts).** Computed as √(var(delta) − noise var):

  | sample | τ (pts/100) |
  |---|---|
  | fit | 1.5 |
  | validation | 2.1 |

  Recent form is mostly noise.
- **`W_RECENT`.** The slope of (next 10 − rating) on (last 10 − rating) is 0.10 (0.12 out of
  sample).

### Team flags, by z threshold

**Null model.** Each team's game order is shuffled within the season, 4× per checkpoint. This
keeps the fat tails and removes real form.

**FDR.** The column is null rate / real rate. It is the share of flags that chance alone would
produce, which makes it an upper bound.

| z ≥ | flag rate | null rate | FDR | rising: next 10 > rating | rising Δ | sliding: next 10 < rating | sliding Δ |
|---|---|---|---|---|---|---|---|
| 2.00 | 6.9% | 5.5% | 0.79 | 60.6% (n=213) | +1.1 | 62.1% (n=182) | −1.6 |
| **2.25** | **4.3%** | **3.1%** | **0.72** | **61.7% (n=133)** | **+1.5** | **64.5% (n=110)** | **−2.1** |
| 2.50 | 2.7% | 1.7% | 0.63 | 60.5% (n=86) | +1.3 | 67.7% (n=65) | −2.7 |
| 3.00 | 0.9% | 0.4% | 0.41 | 69.0% (n=29) | +2.2 | 73.9% (n=23) | −4.5 |

All rows are validation, 2015-16..2024-25 (5,708 team-checkpoints).

- **Base rate.** Across all team-checkpoints, next 10 > rating 49.7% of the time.
- **Threshold choice.** `Z_FLAG = 2.25` was chosen on the fit set, where it gave the best
  persistence (64.3% / 66.4%). That works out to about 1.3 flags per date league-wide.
- **Against the pre-window baseline.** Flagged teams beat it 80% of the time (rising) and fall
  below it 76% of the time (sliding).
- **Honest reading.** A flag marks real but modest carry-over, about 1.5–2 pts/100 of a
  ~10 pt gap. It shouldn't be sold as a regime change.

### Hot/cold players

Validation sample: 2015-16..2024-25, 11,750 player-checkpoints. Next-10 CI is compared with the
same baseline.

| z | hot: next 10 > baseline | cold: next 10 < baseline | FDR |
|---|---|---|---|
| ≥ 2.0 | 76.0% (+2.5 CI, n=1,029) | 68.3% (−1.5 CI, n=398) | 0.61 |
| ≥ 2.5 | 78.7% | 69.3% | 0.48 |

- Base rate: 54.5%.
- Player form carries over much more than team form: the slope is 0.23 against the team's 0.10.
  Minutes and role changes are a big part of that.

## Early season

At `as_of = 2025-11-01` (about 5 games per team):
- There are no rising or sliding flags, no hot/cold entries and no role changes. `notes` say why.
- The power ranking leans on the prior. A 5-0 SAS ranks 10th, and a 5-0 CHI is outside the
  top 10.
- Leaders are labelled with their game count.
- Factual streaks and career highs (Reaves 51, Markkanen 51) still appear.

Before a season's first game, the power ranking is the regressed prior-season rating.

## Tests (`tests/test_trends_*.py`, synthetic leagues, no network)

- No lookahead: games on or after `as_of` are vandalized and the output must not change.
- Early-season shrinkage and suppression.
- Team and player streaks, including a streak carried across seasons.
- Role changes: new in rotation, out of rotation, minutes up.
- Rising and sliding flags.
- Hot detection.
- Career highs.
- The impact metric checked by hand.
- `json.dumps(..., allow_nan=False)` at every stage of a season.

## Limitations and next steps

- **Starter proxy.** Replace minutes rank with the cdn boxscore `starter` flag for live seasons.
  Closing-lineup share from `analytics.stints` would also help.
- **Absences.** Join `out_of_rotation` and `key_absences` with the injury snapshots, to separate
  injury from benching.
- **Season-level on/off.** Use non-garbage on/off from play-by-play (ANALYTICS "Next") in place
  of the box-score REL term.
- **Playoffs** are ignored (regular season only). Adding a playoff mode would mean a
  series-aware window.
- **Pre-2024 neutral sites** are treated as home games (26 games, negligible). `model/sites.py`
  has the curated list.
