# Courtside — prediction model

Status: **rung 2 (player-based team strength) beats rung 1 (Elo + schedule) in every test
season.** The current best is `ps_elo_sched_mh`. Every later rung must beat it on the identical
backtest. Reproduce with:

```
uv run courtside model backtest            # rung-1 baselines, test seasons 2012-13..latest
uv run courtside model backtest --models elo_sched,ps_elo_sched_mh,ps_elo_sched_roster \
    --dump-preds var/reports/preds.csv.gz  # rung 2 (full history: run on the Mac mini)
uv run courtside model tune-elo            # re-run the (frozen) Elo hyperparameter search
uv run courtside model tune-players        # re-run the (frozen) player-rating search
uv run courtside model player-ratings      # top/bottom ratings after the last game
uv run courtside model market var/reports/preds.csv.gz   # dumped predictions vs closing lines
```

When `closing_lines` exists, `backtest` also prints the model-vs-market table and registers the
market-feature variants (`mkt`, `elo_sched_mkt`, `ps_elo_sched_mkt`).

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
| `model/players.py` | rung-2 engine `run_player_engine`, `PlayerParams`/`TUNED`, `PlayerState.expected_lineup`, `availability_from_injuries` |
| `model/player_model.py` | rung-2 heads, shared `EngineCache`, registry `make_registry` (`SPECS`), `tune_all` |
| `model/market.py` | closing lines (ESPN, then Kaggle), de-vig, model vs market, ATS |

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

## Rung 1 results (walk-forward, test seasons 2012-13 → 2025-26)

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

## Rung 2: player-based team strength

### Design (`model/players.py`)

One causal pass over all games, day by day: a morning solve of the player ratings, then
pre-game strengths for each of the day's games, then ingestion of that day's box scores. The
pass over 33k games takes about 80 s on the M4 (one Cholesky solve per game day over the
players active in the last 4 years, about 1,000-1,400 of them).

- **Ratings: ridge APM with a box-score prior.** One observation per player-game, the player's
  on-court plus-minus. Its expectation, assuming random lineup mixing within the game, is
  `(m_i/48)·(r_i + Σ teammates r_k·4m_k/(5G−m_i) − Σ opponents r_j·m_j/G + h)`, with weight
  `48/m_i` (variance grows with minutes). Ratings are in points per 48 minutes, so
  `Σ r·minutes/48` is the team's expected margin against an average team.
  - The posterior mean solves `(A + λI) r = b + λμ`. A = X'WX and b = X'Wy are accumulated
    incrementally with exponential time decay (half-life 4 years; time-scaled storage, so the
    decay is free).
  - Why this design: player-game plus-minus is the richest outcome data we have without stints.
    The ridge/Bayesian form is standard RAPM, and the sufficient statistics make an exact daily
    refit cheap. A Kalman/Elo-per-player update would have to drop the teammate covariance,
    which is the point of APM.
- **Box prior μ = β·(z − z̄).** z holds the player's decayed per-100-possession rates of pts,
  FGA, FTA, 3PA, ast, tov, oreb, dreb, stl, blk and pf (shrunk toward league rates with 600
  pseudo-possessions), plus minutes per game, log evidence minutes, and rookie/sophomore flags.
  - β is refit at every season start by a **predictive** regression: game margins in the
    previous 6 seasons on Σ(minutes/48·z_pre), home minus away, where z_pre is the pre-game
    feature vector. Final-fold β: stl +1.8, tov −0.7, fga −0.7, pts +0.6, blk +0.6, ...
  - Nothing from the current season enters β.
- **Rookies and low-minute players.** A never-seen player gets μ at the default features:
  league rates, 12 minutes, zero evidence and the rookie flag. That is a replacement-level
  prior learned from real debuts (−6.5 pts/48 in 2025-26). Shrinkage of rates and evidence
  handles low-minute players.
- **Between seasons.** APM evidence is multiplied by 0.8 and box evidence by 0.6 each summer, on
  top of the time decay. Ratings regress toward the prior.
- **Expected minutes.** Each player's projection is a 50/50 blend of a fast (α=0.2) and a slow
  (α=0.02) EWMA of his minutes when he plays, starting at 12. It follows the player across
  trades. MAE is 5.3 min per player-game.
  - `expected_lineup(team, game, availability)` weights each projection by P(plays), drops deep
    bench (projection < 6 min, keeping at least 8 players), and renormalises to 240 with a
    42-minute cap (water-filling). OT is ignored.
  - The engine and production call the same `select_lineup`/`allocate_minutes`.
- **Availability.**
  - In the backtest it is the oracle: the players who logged minutes in the game.
  - In production, `availability_from_injuries` maps the latest `injury_snapshots` row per
    player to a probability. The official report wins over ESPN. Out 0, Doubtful 0.15,
    Questionable 0.5, Day-To-Day 0.6, Probable 0.9.
- **Heads (`model/player_model.py`), refit each fold, seasons 2000-03 dropped as burn-in.**
  - Design: `ps_diff` (home − away strength), optionally the Elo difference, and the elo_sched
    home-court and schedule terms.
  - `logit` is an L2 logistic with no intercept.
  - `margin` (`_mh`) is OLS on the margin, with P = Φ(μ/σ) and σ = training RMSE (12.3).
  - **The margin head wins on regular-season log-loss** (−0.0015 ± 0.0006) and is kept. The
    logit head is slightly better in the playoffs.
  - Final-fold margin coefficients: ps_diff 0.77, Elo 1.42 pts per Elo log-odds unit (about 6.3 when Elo is used alone),
    home court 2.2.
- **Hyperparameters were frozen on 2000-01..2010-11.** `tune-players` ran 102 engine passes in 5
  stages, scored by in-sample 3-parameter logistic log-loss on 2003-04..2010-11.
  - Best: λ=300, APM half-life 1460 d, summer carry 0.8, box half-life 365 d, box summer carry
    0.6, box_n0=600, minutes α 0.2/0.02. The surface is flat: the top 10 fall within 0.0003.
  - floor=6 and active_days=1460 were set a priori.
  - A test checks that the tuning score ignores post-2010-11 data.
- **Leakage tests** (`tests/test_model_players.py`, synthetic):
  - Scrambling every box score on or after a cut date leaves all strengths on or before it
    bit-identical.
  - `check_no_lookahead` passes for every player model, plus the same check with scrambled box
    scores.
  - A prefix run equals the full run, which justifies the cache slicing one primed pass.
  - Other tests cover minutes renormalisation and caps, the unseen-player prior, and
    `expected_lineup` with an injured player.

### Results (walk-forward, Regular Season)

| model | 2012-13→2025-26 LL | Brier | acc | MAE | last 3 LL | Brier | acc | MAE |
|---|---|---|---|---|---|---|---|---|
| elo_sched (rung 1) | 0.6131 | 0.2126 | 66.1% | 10.24 | 0.6044 | 0.2088 | 67.4% | 11.13 |
| ps (player strength + HCA only) | 0.5997 | 0.2061 | 67.9% | 10.09 | 0.5844 | 0.1997 | 69.5% | 10.87 |
| ps_elo | 0.5993 | 0.2061 | 67.9% | 10.07 | 0.5846 | 0.1999 | 69.4% | 10.86 |
| ps_elo_sched | 0.5981 | 0.2056 | 67.9% | 10.04 | 0.5821 | 0.1989 | 69.6% | 10.81 |
| **ps_elo_sched_mh** | **0.5966** | **0.2053** | 67.8% | **10.04** | **0.5819** | 0.1990 | 69.5% | 10.81 |

- ps_elo_sched_mh minus elo_sched is −0.0165 ± 0.0012 log-loss (paired SE over 16,898 games).
  It is better in **all 14** test seasons. ECE is 0.013, against 0.021 for elo_sched.
- Margin RMSE is 12.86 against 13.12.
- Playoffs (1,174 games): ps_elo_sched scores 0.6270 against elo_sched's 0.6439. That is mostly
  availability, since stars' playoff minutes are known.

**Ablations** (Regular Season 2012-13→2025-26 log-loss; logit head unless noted):

| variant | LL | Δ vs ps_elo_sched |
|---|---|---|
| ps_elo_sched (oracle availability) | 0.5981 | — |
| ps_elo_sched_nofloor (garbage-time players in the lineup) | 0.5981 | 0.0000 |
| ps_elo_sched_box (box prior only, no plus-minus) | 0.6019 | +0.0038 ± 0.0010 |
| ps_elo_sched_soft (status-change players Questionable w.p. 0.5 → p=0.5) | 0.5994 | +0.0014 ± 0.0005 |
| ps_elo_sched_flip (status-change players' availability flipped w.p. 0.25) | 0.6011 | +0.0030 ± 0.0006 |
| ps_elo_sched_roster (ignore absences: last-10-games rotation) | 0.6100 | +0.0119 |
| ps_roster (player strength alone, ignoring absences) | 0.6150 | +0.0153 vs ps |

- **Availability is the whole story.** Ignoring absences, player strength alone is *worse* than
  Elo (0.6150 vs 0.6131). With Elo and schedule it is only 0.0031 ± 0.0008 better than
  elo_sched. Knowing who plays is worth about 0.012-0.015 log-loss.
- **The oracle's optimism is modest.** "Status change" means a player who missed the team's
  last game and plays today, or played last game and sits today. That is about 1.9 per
  team-game, deep bench included.
  - Marking half of them Questionable (p=0.5, i.e. what production would do) costs 0.0014.
  - Getting a quarter of them outright wrong costs 0.0030.
  - Both noisy variants still beat elo_sched by 0.012-0.014.
  - Garbage-time appearances, the other way the oracle could leak, are worth 0.0000.
- **Plus-minus adds real value over the box prior** (0.0038).

**Market** (`closing_lines`; de-vigged closing moneyline on 16,321 Regular Season test games
with a line, ESPN first, then Kaggle):

| | LL | Brier | acc | margin MAE (vs closing spread) | ATS |
|---|---|---|---|---|---|
| closing line | **0.5918** | **0.2035** | 68.1% | **9.89** | — |
| ps_elo_sched_mh | 0.5960 | 0.2050 | 67.9% | 10.04 | 50.7% |
| elo_sched | 0.6121 | 0.2122 | 66.2% | 10.24 | 50.1% |
| mkt (walk-forward logistic on the spread) | 0.5935 | 0.2042 | 68.2% | 9.89 | — |
| ps_elo_sched_mkt (+ spread as a feature) | 0.5922 | 0.2034 | 68.3% | 9.90 | 50.9% |

- Rung 2 closes about 80% of elo_sched's gap to the closing line, **but it is still behind the
  market**. Against the spread it is 50.7% (break-even is 52.4%), so there is no betting edge.
- Adding the spread to the head helps a little: −0.0012 ± 0.0008 against the spread alone on
  all games; last 3 seasons 0.5773 vs 0.5805. Elo adds nothing on top of the market.

**Top 15 at the end of 2025-26** (≥ 1,000 minutes; pts/48 vs average):

1. Jokić 8.5
2. Wembanyama 7.7
3. Kawhi Leonard 6.8
4. Giannis 5.7
5. Gilgeous-Alexander 5.6
6. Clingan 5.4
7. Kornet 5.3
8. Hartenstein 5.1
9. J. Allen 4.9
10. Holmgren 4.9
11. Curry 4.8
12. Gobert 4.7
13. Queta 4.6
14. Dončić 4.5
15. B. Lopez 4.5

The bottom of the list is young players on bad teams (Traoré −5.4, Powell −4.7, Cody Williams
−4.1). The ranking is plausible. Rim-protecting bigs are over-represented, a known APM and
box (blk/dreb) tilt.

## Where we stand, honestly (rung 1, kept for the record)

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
- **Closing lines are now benchmarked.** See the rung-2 market table above.

## Next steps (ladder)

Rung 2 is done (above). The next steps are in priority order.

1. **Rung 3: GBM stacking with the market line.** Use LightGBM (or a regularised stack) on:
   - `ps_diff`, the Elo difference, the closing or opening spread and total, and schedule terms
   - availability deltas, i.e. the strength lost to absences (played − roster) per team
   - rolling four factors and net rating from `team_games_adv`
   - roster continuity and season phase or tanking proxies

   Fit it walk-forward on 2007-08+ (where lines exist), with an inner fold for
   hyperparameters. The target is closing-line log-loss/Brier. The bar is
   `ps_elo_sched_mkt`'s 0.5922 / 0.2034, and ATS with an edge threshold must clear 52.4%.
2. **Recency and home-advantage drift.** Home court fell from about 60% to 54-55%, and the heads
   weight all history equally.
   - Add a recency-weighted head (season half-life chosen by an inner walk-forward fold), or an
     online HCA (EWMA of home residuals).
   - The APM's own `h` already decays with a 4-year half-life.
   - Also try per-season pace and scoring scaling of the margin σ (σ grows with pace: last-3
     MAE is 10.8 against 10.0 overall).
3. **Playoff handling.** Use rotation tightening (starters' minutes up, deep bench out) in
   `expected_lineup`, and a separate playoff head or σ. The margin head is worse than the logit
   head in the playoffs (0.6307 vs 0.6270). Keep each change only if it beats the logit head's
   0.6270 playoff log-loss.
4. **Production wiring.** In the morning job, run the engine to yesterday, then call
   `availability_from_injuries` and `expected_lineup` per team. Validate the name matching on
   the archived 2026 snapshots, and log unmatched names. Calibrate the status → P(play) map
   once a season of snapshots is joined to box scores.
5. **Ratings quality.**
   - Big men look over-rated (Kornet, Queta, Clingan in the top 15). Try separate offence and
     defence ratings, a position or role-aware prior, and an age curve if birthdates are added.
   - The box prior is linear: try interactions such as usage × TS.
