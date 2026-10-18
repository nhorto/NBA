# Courtside — prediction model

Status: **rung 1 (Elo) plus schedule features.** This is the foundation every later rung must
beat on the identical backtest. Reproduce with:

```
uv run courtside model backtest            # all baselines, test seasons 2012-13..latest
uv run courtside model backtest --models elo,elo_sched --from 2020-21
uv run courtside model tune-elo            # re-run the (frozen) Elo hyperparameter search
```

Each run prints the tables below and writes `var/reports/backtest_<UTC timestamp>.json`. The
report holds per-season, aggregate and last-3-season metrics by segment, 10-bin calibration
tables, the fitted coefficients for every fold, and the data anomaly report.

## Code map

| module | role |
|---|---|
| `teams.franchise(tricode, season)` | historical tricode → current franchise (roster continuity) |
| `model/sites.py` | where each game was played: era-aware home arenas, the bubble, neutral venues |
| `model/dataset.py` | `load_games(conn)`: one validated row per game |
| `model/schedule.py` | rest, back-to-backs, 3-in-4, 4-in-6, games in the last 7 days, travel km, time-zone change |
| `model/elo.py` | `run_elo(games, EloParams)`, `TUNED`, `TUNED_REST`, `tune_all` |
| `model/baselines.py` | `HomeModel`, `EloModel`, `EloScheduleModel`, registry `MODELS` |
| `model/backtest.py` | `walk_forward_splits`, `run_backtest`, `evaluate`, `check_no_lookahead`, report |
| `model/metrics.py` | log-loss, Brier, accuracy, calibration table, ECE, MAE, RMSE |

## Data

The source is `team_games`: stats.nba.com `leaguegamelog`, 2000-01..2025-26, covering Regular
Season, PlayIn and Playoffs. That gives **33,447 games** after cleaning. `build_games` pairs the
two team rows of each game and **raises** on structural problems:

- not exactly two rows
- two home teams
- the two rows disagree on season, type or date
- team and opponent fields that don't cross-match
- a tied final score
- a game where one row has 0 minutes and the other doesn't

### Anomalies found (full data)

- **One junk game dropped:** `0021201214`, BOS–IND on 2013-04-16. It was cancelled after the
  Boston Marathon bombing and appears in the data as two 0-minute, 0-point rows.
- **No team_id in `team_games`, and team_id would be wrong anyway.** stats.nba.com gives the
  2000-02 Charlotte Hornets (`CHH`) TEAM_ID 1610612766, which is today's Charlotte, because in
  2014 the NBA reassigned the 1988-2002 history to Charlotte. That roster actually moved to New
  Orleans (NOH → NOK → NOP); Charlotte restarted as the 2004 expansion Bobcats.
  - `franchise()` therefore maps by tricode and season (`CHH` → NOP; `CHA` before 2004-05 → NOP),
    not by id. There's no need to add team_id to the backfill.
  - All 36 distinct tricodes in the data map to the 30 franchises: NJN → BKN, SEA → OKC,
    VAN → MEM, and NOH/NOK/CHH → NOP.
- **Neutral sites, 36 regular-season games:**
  - **From 2024-25 on (10 games, detected automatically).** stats.nba.com marks these with "@"
    on both sides, so both rows have `is_home = 0`. They are the Mexico City, Paris, Berlin and
    London games and the NBA Cup semifinals in Las Vegas. The nominal home team is the
    alphabetically first one; this is harmless because the home advantage is 0.
  - **Before 2024-25 (26 games, curated).** These are listed in `model/sites.py`: London
    (2011-19), Paris (2020, 2023, 2024), Mexico City (2014-23), and the 2023 In-Season
    Tournament semis in Las Vegas. Every entry was verified to match a real game.
  - **One candidate rejected.** The planned SAS–MIN Mexico City game in 2013-14 (id
    0021300275) was postponed and played in Minneapolis on 2014-04-08.
  - NBA Cup finals don't count as regular-season games, so they are not in the data.
- **Bubble, 172 games:** every 2019-20 game on or after 2020-07-30. That is 88 seeding games,
  1 play-in and 83 playoff games. They are flagged `is_neutral` and located in Orlando.
- **Unusual seasons:**
  - 2011-12 lockout: 990 regular-season games.
  - 2012-13: 1,229 games (because of the cancelled game).
  - 2019-20: 1,059 games.
  - 2020-21: 1,080 games, with empty or limited arenas.
  - The Raptors played 2020-21 home games in Tampa (36 games). These are treated as home games
    located in Tampa.
- **Other checks:** 1,985 overtime games (team minutes > 240). No team plays twice on one day.
  No game_id prefix disagrees with its season type. No W/L flag disagrees with the scores.

### Location approximations (travel / time zone)

- **Home games use the franchise's current arena,** except for explicit era overrides:
  - Seattle (SEA), Vancouver (VAN), Charlotte Coliseum (CHH), New Jersey (NJN)
  - Oklahoma City for NOK in 2005-07
  - Tampa for TOR in 2020-21
- **Moves within a metro area are ignored** (all under 20 km), e.g. Oakland → San Francisco or
  Staples → Intuit Dome.
- **Season openers** measure travel from the team's home arena.
- **Time zones** use the IANA zone of the venue, with DST taken into account.

## Methodology: why this is leakage-free

1. **Walk-forward by season.** For each test season S in 2012-13..2025-26, every learned
   parameter is fit on seasons `< S` only, then season S is predicted. Every split comes from
   `walk_forward_splits`; a test asserts that train dates are strictly before test dates. There
   is never a random split.
2. **Elo hyperparameters were frozen before the first test season.** They were tuned only on
   2000-01..2010-11, scored by log-loss on 2001-02..2010-11 (2000-01 is cold-start burn-in).
   - The search used 75 configurations: a 36-config grid, a 27-config refinement around the
     winner, then 12 rest/travel configs.
   - A test checks that scrambling post-2010-11 results leaves the tuning scores unchanged.
   - No hyperparameter was chosen by looking at 2012-13+ results.
3. **Elo is strictly sequential.** Pre-game ratings for game i are recorded before game i's
   result is applied. A team plays at most once per day (checked), so the order of games within
   a day cannot leak.
   - Tests check that changing a game's result leaves every prediction up to and including that
     game unchanged, while still propagating forward.
   - Tests also check that the walk-forward online Elo equals one chronological pass.
4. **Online models and the test contract.** `predict_proba(test_df)` receives the test season
   with results, so online models can update after each game, exactly as the morning job will
   in production.
   - The rule is that a prediction for day d may use only training data and test games before
     day d.
   - `check_no_lookahead` enforces this: it scrambles every result on or after a cut date and
     requires predictions on or before that date to be bit-identical.
   - Tests run it on every registered model, plus a deliberately cheating model that it must
     catch.
5. **Schedule features use no results.** Each one is a function of the team's earlier game
   dates and venues plus the current game's own date and venue, all of which are published in
   advance. A test flips every result and asserts the features are identical.
6. **Margin scale and the logistic layer** (points per Elo, elo_sched coefficients) are refit on
   each training window. The first season is excluded as burn-in.

## Models

- **home.** Always picks the home team. P(home win) is the training-set home win rate; the
  margin is the training-set mean margin.
- **elo.** FiveThirtyEight-style Elo:
  - Margin-of-victory multiplier `(|MOV|+3)^0.8 / (7.5 + 0.006·Elo_diff_winner)`, where the
    denominator is the autocorrelation correction.
  - Ratings regress between seasons toward 1505.
  - Expansion teams start at 1300 (the 2004 Bobcats).
  - No home advantage at neutral sites or in the bubble.
  - Outputs P(home win), an expected margin, and a spread (`elo_spread = −margin`, negative
    means the home team is favored).
  - **Frozen parameters: K = 20, HCA = 90 Elo, carry = 0.65** (35% of each rating's deviation
    is regressed over the summer). Tuning log-loss was 0.60224. For comparison, 538's published
    K=20, HCA=100, carry=0.75 scored 0.60310 on the same window. The surface is flat: the top
    10 configs fall within 0.001.
  - The implied margin uses the points-per-Elo slope fit walk-forward (≈27.7, close to 538's
    28).
- **elo_rest.** The same Elo with a rest adjustment inside the rating update: **−40 Elo for a
  team on the second night of a back-to-back** (tuning log-loss 0.60083). A travel-distance
  penalty made tuning log-loss worse at every level, so it is 0.
- **elo_sched.** Logistic regression (L2, C=1, no free intercept) refit each fold. Inputs:
  - the Elo rating difference without home advantage, in log-odds units
  - a home-court indicator, which is 0 at neutral sites, and its interaction with the
    postseason
  - home and away b2b, 3-in-4 and 4-in-6 indicators
  - differences in rest, games in the last 7 days, travel (1000 km) and |tz change|

  The structure is antisymmetric, so equal teams at a neutral site come out exactly 50/50. The
  margin comes from a linear regression on the same design.

  Coefficients in the final 2025-26 fold:
  - Elo log-odds: 1.00. Elo's scale is right; its bias is the home advantage.
  - Home court: +0.33 logit (≈ 58 Elo). Postseason home-court extra: +0.21.
  - Home b2b: −0.24. Away b2b: +0.32.
  - Everything else is near zero.

## Results (walk-forward, test seasons 2012-13 → 2025-26)

Regular Season, all test seasons (16,898 games):

| model | log-loss | Brier | accuracy | ECE | margin MAE | margin RMSE |
|---|---|---|---|---|---|---|
| home | 0.6848 | 0.2458 | 57.0% | 0.028 | 11.43 | 14.43 |
| elo | 0.6181 | 0.2148 | 65.5% | 0.040 | 10.30 | 13.20 |
| elo_rest | 0.6170 | 0.2143 | 65.8% | 0.044 | 10.28 | 13.17 |
| **elo_sched** | **0.6131** | **0.2126** | **66.1%** | **0.021** | **10.24** | **13.12** |

Regular Season, last 3 seasons (2023-24, 2024-25, 2025-26; 3,690 games):

| model | log-loss | Brier | accuracy | ECE | margin MAE | margin RMSE |
|---|---|---|---|---|---|---|
| home | 0.6926 | 0.2497 | 54.7% | 0.043 | 12.72 | 16.01 |
| elo | 0.6141 | 0.2130 | 66.0% | 0.062 | 11.25 | 14.39 |
| elo_rest | 0.6116 | 0.2119 | 66.3% | 0.062 | 11.21 | 14.35 |
| **elo_sched** | **0.6044** | **0.2088** | **67.4%** | **0.035** | **11.13** | **14.25** |

Playoffs, all test seasons (1,174 games):

| model | log-loss | Brier | accuracy |
|---|---|---|---|
| home | 0.6770 | 0.2420 | 58.9% |
| elo | 0.6413 | 0.2251 | 63.1% |
| elo_rest | 0.6412 | 0.2250 | 63.2% |
| elo_sched | 0.6439 | 0.2261 | 63.0% |

Regular Season log-loss for elo_sched, season by season:

| season | log-loss | season | log-loss |
|---|---|---|---|
| 2012-13 | 0.5903 | 2019-20 | 0.6266 |
| 2013-14 | 0.6001 | 2020-21 | 0.6475 |
| 2014-15 | 0.6003 | 2021-22 | 0.6346 |
| 2015-16 | 0.5820 | 2022-23 | 0.6445 |
| 2016-17 | 0.6249 | 2023-24 | 0.6098 |
| 2017-18 | 0.6133 | 2024-25 | 0.6063 |
| 2018-19 | 0.6121 | 2025-26 | 0.5972 |

Seasonal spread like this is normal: parity and tanking vary from year to year. The run prints
the full table for every model.

## Where we stand, honestly

- **Benchmarks.** Closing-line favorites win about 67-68% of games, and the closing line's Brier
  score is about 0.20. Good public models land at 65-68% accuracy.
  - Our best rung (elo_sched) reaches 66.1% / Brier 0.2126 over 14 seasons, and 67.4% / 0.2088
    over the last three.
  - That is respectable for a results-only model, but **clearly behind the market** on Brier,
    which is the metric that matters for "our number vs. the line".
- **The legacy model's "69%" was a random-split artifact** and is not comparable. Nothing here
  uses a random split.
- **Elo's calibration error is mostly a home-court bias, not overconfidence.**
  - HCA was tuned on 2000-11, when home teams won about 60%. In 2019-20 onward they won about
    54-55%.
  - Over the test seasons, Elo's mean P(home) is 0.61 against an observed 0.57. elo_sched
    halves the ECE (0.040 → 0.021) by re-learning home-court each fold, but it weights all
    history equally, so it still lags the decline.
- **Schedule effects are real but small.**
  - Back-to-backs are worth about 0.25-0.3 logit.
  - Travel distance and time zones add little once b2b is known.
  - elo_rest's built-in b2b term is worth 0.001 log-loss. elo_sched's whole gain over elo,
    0.005 log-loss, also includes re-learning home court each fold.
- **In the playoffs, the logistic layer doesn't help.** Its playoff log-loss is 0.0026 worse.
  The postseason is small-sample (≈ 84 games a season), and Elo alone is the better playoff
  model for now.
- **We do not yet benchmark against closing lines in the backtest.** The collector only started
  archiving odds in 2026. A historical closing-line archive is the most important missing
  benchmark (DESIGN.md).

## Next steps (ladder)

1. **Time-varying home advantage.** Two options, both leakage-free:
   - Estimate HCA online, e.g. an EWMA of home residuals in Elo.
   - Recency-weight the logistic layer, e.g. with a season half-life. To avoid selecting on test
     results, choose the half-life with an inner walk-forward fold inside each training window.
   - Expected gain: most of the remaining ECE.
2. **Player-availability-adjusted team strength (rung 2), the big one.**
   - Build per-player ratings from `player_games` and `player_games_adv`. Start with a
     box-score/plus-minus prior plus a ridge-regularized on/off or RAPM-lite rating, updated
     online.
   - Team strength for a game = Σ rating × expected minutes over available players. Get
     availability from the official injury reports (2021-22+) and, historically, from "played
     or not" in the box score, then backtest.
   - The idea of a missing rotation player falls out of this directly. It is also what the
     market prices and Elo can't see: rest days for stars, trades, and injuries.
3. **GBM on pregame features (rung 3).** Use LightGBM with monotone constraints where sensible,
   and fit walk-forward. Features:
   - Elo and player-strength differences, with home-court and rest terms
   - Rolling 5/10/20-game offensive, defensive and net rating, pace, four factors, and opponent
     strength-adjusted net rating, all from `team_games_adv` and lagged
   - Roster continuity (minutes returning)
   - Altitude (DEN/UTA)
   - Season phase and tanking proxies (late season, out of contention)
4. **Predict the margin and total as distributions.** For example, a normal or t distribution on
   the margin with σ ≈ 13, possibly depending on pace. Derive P(win) from it, and calibrate
   (isotonic or Platt) on the training window.
5. **Market benchmark.** Get historical closing spreads and moneylines from any obtainable
   archive, then report ATS results and Brier against the closing line.
6. **Separate the playoff treatment.** Possible levers are a higher K, series state and rotation
   tightening. Keep each only if it beats Elo on playoff log-loss.
