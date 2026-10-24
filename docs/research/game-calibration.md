# Game model calibration: is `gbm_market` overconfident in season?

*2026-10-21. HANDOFF §6 follow-up: "the 2025-26 day-by-day replay showed a calibration slope of
about 0.8". Scripts: `research/game_calibration.py` (analysis, tables) and
`research/game_calibration_replay.py` (day-by-day production replay on a scratch database copy).
Research only: no product code changed, no switch built.*

## 0. Plain-language summary

- **The 0.8 slope is real arithmetic but it is not a property of the model.** It reproduces
  exactly (0.80, z = +2.47 on 838 games), but those games are only **21 Oct 2025 – 20 Feb 2026**:
  the replay stopped in February. Over the **whole 2025-26 regular season** the same day-by-day
  replay has slope **0.97 (SE 0.07), z = +0.52**, and the walk-forward backtest 0.98 (0.07).
  After 20 Feb the model was *under*confident (slope 1.32, z = −2.45), which cancelled out the
  first two-thirds of the season.
- **The market went through the same stretch.** In December 2025 the closing line's slope was
  0.51 against the model's 0.50: favourites lost more than anyone priced, then won more than
  anyone priced from late February on. Across 14 walk-forward seasons the model's slope is
  **0.98 (0.02)** and the closing line's is **1.00 (0.02)**; neither is significantly
  miscalibrated (z = +1.42 and +0.56). December and January are the months where both run
  overconfident (January: model 0.88, market 0.90), every year.
- **What is specific to the model:** its probabilities sit slightly further from 50% than the
  close's, and in 2025-26 noticeably so (model slope minus market slope **−0.10 (0.014)**
  against **−0.011 (0.004)** over all seasons). The model's departures from the closing line
  carry little information: only about **30%** of each departure is borne out by results
  (coefficient 0.29, SE 0.12). In 2025-26 that figure was about zero and the departures were
  a third to 40% larger than in the two seasons before, mostly as team-level tilts (it rated the
  strongest teams higher and the weakest lower than the market did).
- **Not early-season priors:** in team games 0–10 (all seasons) the model's slope is 0.93 (0.05)
  against the market's 0.96 (0.06), and its log-loss gap to the market there (+0.0005, SE
  0.0013) is no worse than later in the season. In 2025-26 games 0–10 had slope 1.05.
- **Not missing injury data:** the replay never "lacked" injuries in a harmful way; for past days
  it uses the rotation regulars who actually played. Pricing the archived official injury
  report instead (available from 22 Dec 2025) moved probabilities by 0.7 points on average and
  changed nothing: log-loss −0.0003 (SE 0.0007), slope 1.05 vs 1.05 on the 808 games where both
  teams had a report. Capping each game's injury news at its own tip-off (IMPROVEMENTS B3; the
  old replay read reports up to 1 AM ET) also changes nothing: ΔLL +0.00000 (SE 0.00003), §6.1.
- **No fix clears the ship bar, so no switch was built.** Temperature/Platt layers (global, by
  season phase, early season only) make log-loss worse. The best candidate, pulling the model
  toward the closing line by a weight fit on earlier seasons (w ≈ 0.2–0.3), improves log-loss by
  **−0.00089 (SE 0.00039), 2.3 clustered SE**, but is worse in 2024-25 by **1.06 SE**, just
  over the "no season worse by more than 1 SE" rule. It is recorded here as the near miss.
- **Monitoring is already the right tool, read next to the market.** Replaying the MLOps
  calibration test over 2025-26: the last-300 z would have warned around 1 January (+3.31) and
  gone critical by mid-January (+4.04), but the closing line's z on the same games was +2.42 and
  +3.29. The alert cleared by early March. Only the season-window gap (model +2.85 vs market
  +1.51 in mid-January) was the model's own.

## 1. Where the 0.8 came from

- **Source.** The local database (`var/courtside.db`, geekom1) holds a production backfill
  (`courtside predict --backfill-from`, `predict.predict_days`, refit every day) for
  2025-10-21 → 2026-02-20: 838 regular-season games, `gbm_market` and `gbm_opinion`, created
  2026-10-18 before the official injury archive was loaded (1,638 of 1,676 team-sides have
  availability basis `regulars_played`). `mlops monitor --replay-from` turned it into
  `metric_snapshots`; the last season-window row (as of 2026-02-21) reads slope 0.8018, SE
  0.0802, z 2.4725. That is the number HANDOFF quotes. The Mac prod database was rebuilt on
  2026-10-19 and holds no 2025-26 prediction rows.
- **Reproduced three ways** on the same 838 games:

  | arm | LL | slope (date-clustered SE) | z |
  |---|---|---|---|
  | stored replay (2026-10-18 code) | 0.6216 | 0.80 (0.08) | +2.47 |
  | new replay, current code, no injury rows | 0.6211 | 0.80 (0.08) | +2.44 |
  | new replay, current code, injury report priced | 0.6216 | 0.80 (0.08) | +2.46 |
  | walk-forward backtest (season fit), same games | 0.6195 | 0.82 (0.09) | +2.21 |
  | closing line (de-vigged), same games | 0.6152 | 0.92 (0.09) | +0.97 |

- **The rest of the season** (392 games, 21 Feb → 12 Apr): replay slope 1.32 (0.12), z −2.45;
  closing line 1.40 (0.14), z −2.95. Full season replay: 0.97 (0.07).
- **What the live monitor would have said** (Spiegelhalter z as in `mlops.monitor`, "keep"
  replay; alert at |z| > 2.5, critical > 3.5):

  | as of | games | season z, model | season z, close | last-300 z, model | last-300 z, close |
  |---|---|---|---|---|---|
  | 2025-12-15 | 382 | +1.21 | +0.07 | +1.12 | +0.12 |
  | 2026-01-01 | 496 | +2.31 | +1.12 | +3.31 | +2.42 |
  | 2026-01-15 | 602 | +2.85 | +1.51 | +4.04 | +3.29 |
  | 2026-02-01 | 729 | +2.79 | +1.33 | +2.50 | +1.64 |
  | 2026-02-15 | 819 | +2.61 | +1.14 | +1.32 | +0.44 |
  | 2026-03-01 | 895 | +2.15 | +0.65 | −0.27 | −0.97 |
  | 2026-04-01 | 1,134 | +1.03 | −0.49 | −1.98 | −2.38 |
  | 2026-04-13 | 1,230 | +0.56 | −1.02 | −1.84 | −2.37 |

## 2. Method

- **Predictions.** Walk-forward `gbm_market` and `gbm_opinion`, test seasons 2012-13 → 2025-26,
  regenerated with current code (`courtside model gbm-backtest --models gbm_market,gbm_opinion
  --no-ablations`, Mac scratch, 2026-10-22 03:09Z). Each season is predicted by models fit on
  earlier seasons only, with leak-safe availability (`avail_tight`: only regulars' absences are
  known). Regular season only: 16,898 games, 16,321 with a closing moneyline.
- **Market.** De-vigged closing moneyline (`model.market.load_closing`: ESPN, else Kaggle;
  Kaggle through 2022-23, ESPN from 2023-24). 2022-23 has lines for only 664 games.
- **Day-by-day replays of 2025-26** (`research/game_calibration_replay.py`, scratch copies of
  the prod database): `predict_days` with daily refits and closing lines as the market input,
  once with `injury_snapshots` emptied ("drop") and once with the 177,721 archived official
  report rows ("keep"; 22 Dec 2025 → 13 Jun 2026, no 2025-26 ESPN rows exist).
- **Statistics.** Logistic recalibration y ~ a + b·logit(p), slope b < 1 = overconfident, with
  a date-clustered sandwich SE (all games of an NBA day form one cluster). The MLOps z exactly
  as `mlops.monitor.spiegelhalter_z` (z > 0 = overconfident). Model-minus-market slope
  differences use a date-cluster bootstrap on the same games. Paired log-loss/Brier differences
  use date-clustered SEs. "b_disagree" is the coefficient on (logit p_model − logit p_market)
  in y ~ a + b1·logit(p_market) + b2·(logit p_model − logit p_market): 1 would mean the model's
  departures from the close are fully borne out, 0 that they are noise.
- **Fixes** are walk-forward layers on top of the walk-forward predictions: the layer for test
  season S is fit on the predictions for seasons before S (first fixed season 2014-15, so every
  layer has at least two seasons). Ship bar (task spec): pooled log-loss better by ≥ 2 clustered
  SE and no season worse by more than 1 SE.

## 3. Is it real? Calibration by season

### Calibration by season (Regular Season, walk-forward)

| season | n | model LL | model slope (SE) | model z | market LL | market slope (SE) | market z | model − market slope (SE) |
|---|---|---|---|---|---|---|---|---|
| 2012-13 | 1229 | 0.5811 | 1.00 (0.07) | -0.04 | 0.5793 | 1.06 (0.08) | -0.81 | -0.061 (0.020) |
| 2013-14 | 1230 | 0.5856 | 0.98 (0.08) | +1.29 | 0.5853 | 1.02 (0.08) | +0.13 | -0.050 (0.014) |
| 2014-15 | 1230 | 0.5774 | 1.02 (0.07) | +0.10 | 0.5733 | 1.04 (0.07) | -0.14 | -0.014 (0.013) |
| 2015-16 | 1230 | 0.5706 | 1.06 (0.07) | -0.90 | 0.5712 | 1.01 (0.07) | -0.29 | +0.047 (0.011) |
| 2016-17 | 1230 | 0.6108 | 0.85 (0.07) | +2.38 | 0.6115 | 0.85 (0.07) | +2.11 | -0.002 (0.012) |
| 2017-18 | 1230 | 0.5963 | 0.97 (0.08) | +0.61 | 0.5923 | 1.01 (0.08) | -0.12 | -0.041 (0.015) |
| 2018-19 | 1230 | 0.5940 | 0.96 (0.08) | +0.43 | 0.5930 | 0.94 (0.08) | +0.42 | +0.013 (0.012) |
| 2019-20 | 1059 | 0.6117 | 0.95 (0.08) | +0.89 | 0.6115 | 0.92 (0.08) | +1.20 | +0.031 (0.011) |
| 2020-21 | 1080 | 0.6164 | 0.97 (0.08) | +0.96 | 0.6165 | 0.92 (0.07) | +1.07 | +0.052 (0.012) |
| 2021-22 | 1230 | 0.6042 | 1.05 (0.08) | +0.03 | 0.6047 | 1.00 (0.08) | +0.45 | +0.043 (0.015) |
| 2022-23 | 1230 | 0.6253 | 0.88 (0.08) | +0.82 | 0.6390 | 0.78 (0.12) | +1.14 | -0.009 (0.015) |
| 2023-24 | 1230 | 0.5814 | 1.15 (0.08) | -1.56 | 0.5812 | 1.12 (0.08) | -1.22 | +0.024 (0.012) |
| 2024-25 | 1230 | 0.5815 | 1.02 (0.07) | +0.03 | 0.5828 | 1.04 (0.08) | -0.27 | -0.017 (0.009) |
| 2025-26 | 1230 | 0.5738 | 0.98 (0.07) | +0.35 | 0.5713 | 1.08 (0.07) | -1.02 | -0.096 (0.014) |
| **all** | 16898 | 0.5932 | 0.98 (0.02) | +1.42 | 0.5918 | 1.00 (0.02) | +0.56 | -0.011 (0.004) |

Market columns are on games with a moneyline; the model-minus-market slope is on those same
games. Market LL for 2022-23 is on 664 games only.

- **No season is significantly miscalibrated for the model at the MLOps threshold (|z| > 2.5).**
  2016-17 is the closest (z +2.38), and the market is at +2.11 in the same season.
- **The model and the close move together** season to season (2016-17 both 0.85, 2023-24 both
  ≈ 1.13). Season-level swings of ±0.15 are mostly a property of the season, not the model.
- **2025-26 is the season where the model was most spread out relative to the close** (−0.096,
  about 7 SE, the largest gap in 14 seasons). The close was underconfident that season (1.08), so
  the model's full-season slope still came out at 0.98.
- `gbm_opinion` (no market input), for reference: all seasons slope 0.99 (0.02), z +0.84;
  2025-26 0.95 (0.07), z +0.67. Its departures from the close carry even less information
  (b_disagree 0.11, SE 0.05).
- The pre-2026-10-18 walk-forward (`/tmp/cs-mlops/rung3_merged_preds.csv.gz`, before the
  season-opening rolling-prior fix 2aa9a89) gives the same picture: all seasons 0.98 (0.02),
  games 0–10 0.92 vs 0.93 now.

## 4. Where is it?

### By team games into the season (all seasons)

`gis` = the smaller of the two teams' regular-season games already played.

| group | n | model slope (SE) | model z | market slope (SE) | market z | model − market slope (SE) | LL model − market (SE) | b_disagree (SE) |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 2401 | 0.93 (0.05) | +1.31 | 0.96 (0.06) | +0.61 | -0.032 (0.010) | +0.0005 (0.0013) | 0.40 (0.29) |
| 11-25 | 3195 | 1.02 (0.05) | -0.21 | 1.04 (0.05) | -0.55 | -0.012 (0.010) | +0.0004 (0.0012) | 0.40 (0.30) |
| 26+ | 11302 | 0.99 (0.02) | +1.24 | 0.99 (0.03) | +0.69 | -0.005 (0.005) | +0.0012 (0.0006) | 0.26 (0.14) |

### By month (all seasons)

| month | n | model slope (SE) | model z | market slope (SE) | market z | model − market slope (SE) | LL model − market (SE) | b_disagree (SE) |
|---|---|---|---|---|---|---|---|---|
| Oct | 822 | 0.91 (0.10) | +0.88 | 0.94 (0.10) | +0.51 | -0.037 (0.017) | +0.0022 (0.0023) | 0.01 (0.56) |
| Nov | 2876 | 1.03 (0.05) | -0.85 | 1.07 (0.05) | -1.50 | -0.031 (0.010) | -0.0002 (0.0012) | 0.53 (0.29) |
| Dec | 2895 | 0.92 (0.05) | +2.33 | 0.94 (0.05) | +1.85 | -0.022 (0.010) | +0.0020 (0.0013) | 0.00 (0.31) |
| Jan | 3164 | 0.88 (0.05) | +2.96 | 0.90 (0.05) | +2.51 | -0.011 (0.009) | +0.0018 (0.0012) | 0.09 (0.27) |
| Feb | 2395 | 1.02 (0.05) | +0.02 | 1.01 (0.06) | +0.19 | +0.004 (0.010) | -0.0001 (0.0013) | 0.61 (0.33) |
| Mar | 3080 | 1.07 (0.05) | -1.62 | 1.07 (0.05) | -1.65 | +0.014 (0.008) | -0.0011 (0.0011) | 0.79 (0.29) |
| Apr | 1443 | 1.00 (0.06) | +0.57 | 1.03 (0.06) | -0.14 | -0.017 (0.013) | +0.0042 (0.0020) | -0.09 (0.34) |

(May = the end of the 2020-21 season, n 135; Aug = 2020 bubble seeding games, n 80; both
too small to read.)

### By favourite size (the market favourite's probability; model's own where no line)

| group | n | model slope (SE) | model z | market slope (SE) | market z | model − market slope (SE) | LL model − market (SE) | b_disagree (SE) |
|---|---|---|---|---|---|---|---|---|
| 50-60% | 4830 | 0.66 (0.10) | +3.67 | 0.90 (0.12) | +0.84 | -0.249 (0.052) | +0.0022 (0.0010) | 0.07 (0.23) |
| 60-70% | 4941 | 0.90 (0.05) | +2.08 | 0.95 (0.05) | +0.76 | -0.061 (0.011) | -0.0004 (0.0009) | 0.68 (0.23) |
| 70-80% | 3865 | 1.02 (0.04) | -0.11 | 1.03 (0.04) | -0.44 | -0.010 (0.006) | +0.0020 (0.0010) | 0.02 (0.26) |
| 80%+ | 3262 | 1.04 (0.04) | -0.56 | 1.01 (0.04) | +0.51 | +0.041 (0.005) | -0.0003 (0.0012) | 0.64 (0.28) |

Caveat: bins are cut on the market's number, so within a bin most of the model's spread is its
disagreement with the market. A low slope there means "when the close calls it near even and
the model does not, the close is usually right", not that the model's 55% games come in at 50%.

### Market odds present or not

| | n | model slope (SE) | z |
|---|---|---|---|
| closing moneyline present | 16,321 | 0.99 (0.02) | +1.47 |
| no closing moneyline (model runs without the market columns) | 577 | 0.98 (0.11) | −0.15 |

### 2025-26 by window, phase and month (walk-forward)

| group | n | model slope (SE) | model z | market slope (SE) | market z | model − market slope (SE) |
|---|---|---|---|---|---|---|
| to 20 Feb | 838 | 0.82 (0.09) | +2.21 | 0.92 (0.09) | +0.97 | -0.090 (0.016) |
| 21 Feb on | 392 | 1.31 (0.12) | -2.39 | 1.40 (0.14) | -2.95 | -0.089 (0.027) |
| games 0-10 | 169 | 1.05 (0.18) | -0.23 | 1.16 (0.19) | -0.71 | -0.109 (0.035) |
| games 11-25 | 229 | 0.77 (0.17) | +1.92 | 0.85 (0.17) | +1.34 | -0.051 (0.022) |
| games 26+ | 832 | 1.04 (0.08) | -0.50 | 1.15 (0.09) | -1.64 | -0.111 (0.017) |
| Oct | 80 | 0.96 (0.30) | +0.08 | 1.05 (0.30) | -0.20 | -0.088 (0.054) |
| Nov | 219 | 1.11 (0.16) | -0.57 | 1.27 (0.17) | -1.29 | -0.110 (0.032) |
| Dec | 197 | 0.50 (0.16) | +3.51 | 0.51 (0.17) | +3.24 | -0.012 (0.023) |
| Jan | 233 | 0.72 (0.17) | +1.64 | 0.84 (0.19) | +0.73 | -0.125 (0.031) |
| Feb | 166 | 1.26 (0.16) | -0.75 | 1.45 (0.18) | -1.27 | -0.183 (0.048) |
| Mar | 239 | 1.27 (0.16) | -1.80 | 1.33 (0.17) | -2.10 | -0.058 (0.033) |
| Apr | 96 | 1.32 (0.29) | -1.27 | 1.48 (0.36) | -1.78 | -0.166 (0.165) |

### Reading the splits against the three candidates

1. **Early-season priors (Elo/APM carryover): a small, shared effect, not the cause.**
   Games 0–10 run slightly overconfident for both (0.93 vs 0.96), the model a little more
   (−0.03, 3 SE), but its log-loss against the market in those games is no worse than later
   (+0.0005 vs +0.0012). In 2025-26 games 0–10 were fine (1.05). A layer fit only on games
   0–10 made them worse (§5, `temp_early`).
2. **Missing injury information in the replay: no effect.** The replay was not blind: on past
   days the engine knows which rotation regulars (projected ≥ 20 min) played. Pricing the
   official report instead, as production does, barely moves `gbm_market` because the closing
   line already carries the news:

   | 2025-26, day-by-day replay | n | LL keep − drop (SE) | slope keep − drop (SE) | mean \|Δp\| |
   |---|---|---|---|---|
   | all games | 1,230 | −0.00018 (0.00043) | −0.003 (0.002) | 0.0045 |
   | both teams have an official report (22 Dec on) | 808 | −0.00028 (0.00066) | −0.005 (0.004) | 0.0068 |
   | no report archived (before 22 Dec) | 422 | 0 | 0 | 0 |

   On the 808 report games: replay slope 1.05 (0.09), close 1.16 (0.09). The 422 games before
   the archive starts have slope 0.82, and they include the December stretch that the close
   also got wrong (0.51).
3. **The model departing from a correct market: yes, this is the model-specific part.** Pooled,
   only 29% (SE 12%) of the model's logit departure from the close is borne out by results;
   in 2025-26 about none of it. The 2025-26 departures were larger (SD 0.17 logit vs 0.12–0.13
   in 2023-25) and mostly team-level: per-team mean departures had SD 0.074 logit vs 0.037 in
   each of the two prior seasons, with the model higher than the close on OKC (+0.17), HOU
   (+0.14), CLE (+0.10) and lower on WAS (−0.19), BKN (−0.12), SAC (−0.09), and those tilts did
   not line up with results (correlation 0.07 with the teams' results relative to the close).
   This report does not trace why the player-strength terms spread the league out more in
   2025-26.

## 5. Candidate fixes (walk-forward only)

All layers are fit on the walk-forward predictions of seasons before the test season. Test
seasons 2014-15 → 2025-26, 14,439 games. Changes are new minus current (negative is better).

| fix | what it does | LL change (SE) | z | Brier change (SE) | accuracy new / old | slope new / old | worst season (z) | clears bar |
|---|---|---|---|---|---|---|---|---|
| temp | one Platt layer a + b·logit p | +0.00015 (0.00014) | +1.12 | +0.00006 (0.00006) | 0.6793 / 0.6801 | 1.001 / 0.984 | 2022-23 (+2.98) | no |
| temp_phase | a Platt layer per phase (games 0–10 / 11–25 / 26+) | +0.00068 (0.00023) | +2.97 | +0.00027 (0.00010) | 0.6798 / 0.6801 | 0.994 / 0.984 | 2015-16 (+3.29) | no |
| temp_early | a Platt layer for games 0–10 only | +0.00019 (0.00009) | +2.17 | +0.00007 (0.00004) | 0.6804 / 0.6801 | 0.989 / 0.984 | 2023-24 (+2.62) | no |
| **shrink** | logit p' = logit p_mkt + w·(logit p − logit p_mkt) where a moneyline exists | **−0.00089 (0.00039)** | **−2.29** | −0.00038 (0.00016) | 0.6805 / 0.6801 | 0.988 / 0.984 | **2024-25 (+1.06)** | **no (by 0.06 SE)** |
| shrink_phase | the same with one w per phase (Platt elsewhere) | −0.00074 (0.00040) | −1.85 | −0.00032 (0.00016) | 0.6807 / 0.6801 | 0.986 / 0.984 | 2024-25 (+1.03) | no |
| market_only | reference, not a candidate: the de-vigged close wherever it exists | −0.00081 (0.00051) | −1.60 | −0.00035 (0.00021) | 0.6807 / 0.6801 | 0.987 / 0.984 | 2024-25 (+1.29) | no |

### Per season, the shrink layer

| season | w (fit on earlier seasons) | LL change (SE) | z | slope old → new |
|---|---|---|---|---|
| 2014-15 | 0.29 | -0.00336 (0.00130) | -2.59 | 1.023 → 1.038 |
| 2015-16 | 0.12 | +0.00028 (0.00147) | +0.19 | 1.059 → 1.020 |
| 2016-17 | 0.22 | +0.00026 (0.00143) | +0.18 | 0.849 → 0.854 |
| 2017-18 | 0.28 | -0.00329 (0.00136) | -2.43 | 0.971 → 1.004 |
| 2018-19 | 0.18 | -0.00106 (0.00141) | -0.75 | 0.955 → 0.947 |
| 2019-20 | 0.18 | -0.00038 (0.00128) | -0.30 | 0.949 → 0.925 |
| 2020-21 | 0.20 | -0.00044 (0.00174) | -0.25 | 0.970 → 0.930 |
| 2021-22 | 0.24 | +0.00018 (0.00120) | +0.15 | 1.046 → 1.016 |
| 2022-23 | 0.27 | -0.00212 (0.00130) | -1.63 | 0.883 → 0.894 |
| 2023-24 | 0.23 | -0.00040 (0.00116) | -0.34 | 1.146 → 1.130 |
| 2024-25 | 0.24 | +0.00119 (0.00113) | +1.06 | 1.017 → 1.032 |
| 2025-26 | 0.29 | -0.00144 (0.00139) | -1.04 | 0.980 → 1.049 |

- **Temperature layers don't help** because there is almost nothing to correct on average: the
  fitted slope on earlier seasons is 0.97–1.01 every year. The layer mostly learns a small home
  intercept (−0.01 to −0.05), which hurt in 2022-23 when the home advantage moved the other way.
- **Phase-specific layers hurt** (+0.0007, 3 SE worse): the phase effects are too small and too
  unstable from year to year to fit.
- **The shrink layer is the only candidate with a real gain**, and it is not a calibration fix
  in the slope sense (the pooled slope moves 0.984 → 0.988). It works because the model's
  departures from the close are mostly noise: the fitted weight keeps a quarter of each
  departure. It fails the per-season rule by a hair (2024-25 at +1.06 SE), and the same check
  started from 2013-14 gives the same verdict (−0.00090 (0.00039), 2024-25 +1.06 SE). It also
  carries a production caveat the backtest cannot measure: in production the market input is
  the newest pre-tip line (often a morning line), not the close the layer was fit against.
- **Stronger early-season prior regression** (changing Elo carry or the APM summer carry) was
  not run. Both are frozen hyperparameters tuned on 2000-01 → 2010-11; retuning them on later
  seasons would break that rule, and the early-season splits above show no gap to fix.

## 6. Caveats

- **One market source per era.** Kaggle closing lines through 2022-23, ESPN from 2023-24. The
  2022-23 Kaggle coverage is 664 of 1,230 games, so that season's market rows are a subset.
- **The walk-forward availability is leak-safe but not production-identical:** regulars' absences
  come from who played. The "keep" replay prices the official report as production does, but
  only from 22 Dec 2025 (the archive start). The replays above used the end-of-day report
  (`injury_cutoff` for past days was 05:00 UTC the next day, after every tip): 679 of the 808
  report games read a batch published after their tip (median 2.5 h after). Since B3
  (2026-10-23) each past game is capped at its own tip; re-running the replay that way changed
  nothing measurable (§6.1).
- **The replays use closing lines as the market input.** Production prices with the newest
  pre-tip snapshot, so live calibration can differ; the MLOps monitor measures that directly.
  IMPROVEMENTS G1 measured the gap (`line-timing.md`): fed the ESPN open instead of the close,
  `gbm_market` loses 0.008 LL on 879 2025-26 games. A "news since the line" shift
  (`COURTSIDE_NEWS_SINCE_LINE`, default OFF) recovers little of that, and live lines are a
  median of 1 min old at capture.
- **Slopes over ~200-game windows are noisy** (SE 0.15–0.20); monthly 2025-26 rows are
  descriptive only.
- **Clustering by date** treats games on different days as independent. Season-level shocks
  (the December 2025 run of upsets) are shared by all games in a stretch, so even the clustered
  SEs on within-season windows understate the real uncertainty about "how the model will do
  next December".

### 6.1 Re-run with pre-tip injury cutoffs (IMPROVEMENTS B3, 2026-10-23)

`research/game_calibration_replay.py --arms legacy:0,tip:0,tip:1,legacy:1,virtual:0`: the same
daily fits price every arm, 2025-26 Regular Season (21 Oct → 12 Apr), Mac scratch copy with
schema 16 + the `game_date` backfill. `legacy` = the old end-of-day cutoff, `tip` = each game's
own tip-off (the code as shipped; tips from the archived ESPN scoreboard, all 1,230 known),
`virtual` = 12:00Z for every game (an 8 AM ET morning run, sensitivity only), `:1` = the opt-in
game-day rows (`COURTSIDE_INJ_GAMEDAY_GAME=1`: on a back-to-back only the row for the game's own
date). `gbm_market`; differences are arm minus `legacy`, date-clustered SEs
(`research/b3_injury_cutoff_eval.py`).

| arm | games | LL | Brier | slope (SE) | ΔLL vs legacy (SE) | ΔBrier (SE) | Δslope (SE) | games with p changed |
|---|---|---|---|---|---|---|---|---|
| legacy (old cutoff) | 1,230 | 0.5750 | 0.1968 | 0.965 (0.069) | | | | |
| **tip (new)** | 1,230 | 0.5750 | 0.1968 | 0.965 (0.069) | +0.00000 (0.00003) | +0.00000 (0.00002) | +0.000 (0.000) | 11 (max 1.3 pts) |
| tip + game-day rows | 1,230 | 0.5750 | 0.1968 | 0.965 (0.069) | +0.00004 (0.00005) | +0.00002 (0.00002) | −0.000 (0.000) | 1 vs tip (2.0 pts) |
| virtual 12:00Z | 1,230 | 0.5750 | 0.1968 | 0.970 (0.069) | +0.00003 (0.00068) | +0.00002 (0.00028) | +0.005 (0.004) | 766 vs tip (mean 0.5 pts) |
| legacy, report games | 808 | 0.5560 | 0.1883 | 1.048 (0.088) | | | | |
| **tip, report games** | 808 | 0.5560 | 0.1883 | 1.048 (0.088) | +0.00000 (0.00005) | +0.00000 (0.00002) | +0.000 (0.000) | |
| tip + game-day, report games | 808 | 0.5560 | 0.1884 | 1.048 (0.088) | +0.00006 (0.00007) | +0.00003 (0.00004) | −0.000 (0.000) | |

Close on the same games: 0.5713 / slope 1.083 (0.075) on 1,228 with a line; 0.5541 / 1.156
(0.094) on the 808 report games. So the old replay was not flattered in any measurable way: the
post-tip batches almost never changed a priced status for a game already under way. The 1.05
(report games) and 0.97 (full season) above stand. The game-day rows move one game: a single
official batch lists a player for two dates in only 148 of 171,832 player-batch rows
(22 Dec → 12 Apr, 24 with different statuses), so the back-to-back merge almost never bites at
pre-tip cutoffs. The 12:00Z arm shows what the late-afternoon reports are worth to the model
(0.5 pts of p on average) and that the log-loss is indifferent to them, consistent with §4 (the
close already carries the news).

## 7. Recommendations

1. **Close the HANDOFF item:** the 0.8 was a partial-season window; full-season calibration is
   fine, and the closing line was overconfident in the same weeks.
2. **When the calibration alert fires, read it next to the market's z on the same games**
   (`metric_snapshots` already stores the market baseline per window). Only a model-minus-market
   gap is ours to fix.
3. **Revisit the shrink layer after 2026-27**, refit on 2012-13 → 2025-26 (w ≈ 0.29). If it
   still improves log-loss and no season is worse by more than 1 SE, it can go behind
   `COURTSIDE_GAMECAL=1` (default OFF) for the owner to decide. It would change the Brief's
   headline probability toward the market's, so it is a product decision as much as a model
   one.
4. Worth a look separately: why the player-strength terms spread the league further than the
   close in 2025-26 (per-team tilts twice the usual size).

## Reproduce

```
# Mac scratch (heavy): walk-forward predictions with current code
COURTSIDE_DATA=<scratch> COURTSIDE_CACHE=<scratch>/cache uv run courtside model gbm-backtest \
    --models gbm_market,gbm_opinion --no-ablations --threads 1
# day-by-day replays on a scratch copy (never the prod file)
uv run python research/game_calibration_replay.py --src <prod.db> --db <scratch>/replay_drop.db \
    --injuries drop --start 2025-10-21 --end 2026-04-12 --out replay_drop.csv.gz
uv run python research/game_calibration_replay.py --src <prod.db> --db <scratch>/replay_keep.db \
    --injuries keep --start 2025-10-21 --end 2026-04-12 --out replay_keep.csv.gz
# §6.1 (B3): every cutoff arm from the same fits (scratch DB needs schema 16 + game_date backfill)
uv run python research/game_calibration_replay.py --src <scratch-gd.db> --db <scratch>/b3.db \
    --arms legacy:0,tip:0,tip:1,legacy:1,virtual:0 --fast --start 2025-10-21 --end 2026-04-12 \
    --out 'b3_{arm}.csv.gz'
uv run python research/b3_injury_cutoff_eval.py --db <scratch-gd.db> --arm legacy_off=b3_legacy_off.csv.gz \
    --arm tip_off=b3_tip_off.csv.gz --arm tip_on=b3_tip_on.csv.gz --out b3_eval.json
# analysis (light, geekom1)
uv run python research/game_calibration.py --preds rung3_preds_<ts>.csv.gz --db var/courtside.db \
    --replay orig=replay_orig.csv.gz --replay drop=replay_drop.csv.gz \
    --replay keep=replay_keep.csv.gz --out var/tmp/gcal/gbm_market.json --md var/tmp/gcal/tables.md
```

The §6.1 arms (`all_{arm}.csv.gz`) and `b3_eval.json` are in `var/tmp/b3/` on geekom1 (Mac scratch
`~/Developer/courtside-compute/b3/`). Inputs and outputs from this run are in `var/tmp/gcal/` on geekom1 (`rung3_preds_20261022T030937Z.csv.gz`,
`replay_{orig,drop,keep}.csv.gz`, `gbm_market.json`, `gbm_opinion.json`, `tables.md`). The keep
replay was run in three date chunks (21 Oct–31 Jan, 1 Feb–5 Mar, 6 Mar–12 Apr) for speed; each
day is an independent refit, so chunking does not change any prediction.
