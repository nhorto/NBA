# Courtside model card

_Generated 2026-10-26 09:00 UTC by `courtside mlops card` from the model registry (`model_versions`). Do not edit by hand; see docs/MODEL.md for the methodology and docs/MLOPS.md for how versions change._

## Intended use

Pre-game win probabilities, expected margins and totals for NBA games, shown in the Courtside Morning Brief as **information, not betting advice**. The model has no demonstrated edge against the betting market (see limitations). It is not meant for wagering, player evaluation or any decision with money at stake.

## Champion versions

| model | role in the product | version | champion since | code | spec hash |
|---|---|---|---|---|---|
| gbm_market | headline win probability, margin and total shown in the Brief | `gbm_market@v1-1d8aa7da` | 2026-10-19 | `99d093c3c599` | `1d8aa7da90b8` |
| gbm_opinion | "our number": independent of the market, used for the edge vs the line | `gbm_opinion@v1-8b7ad87b` | 2026-10-19 | `99d093c3c599` | `8b7ad87ba35b` |

## Training data

- 2000-01 → 2026-27: 33,491 games (2000-10-31 to 2026-10-25) in `team_games`; the first 3 seasons are burn-in for the player engine and the GBM heads.
- Production refits every model every morning on **all games strictly before the day** (continuous training; a refit is not a new version). Hyperparameters are fixed per version; recency weighting comes from the version's half-life.
- Inputs: player-strength engine (box scores), Elo, schedule/travel, rolling team form and four factors, league environment, home-court drift terms; `gbm_market` also the closing (backtest) or latest pre-tip (live) spread and total; injury reports priced into expected lineups when available.

## gbm_market (`gbm_market@v1-1d8aa7da`)

Market input: yes. Search grid the fixed values come from: 4 configs (inner time-ordered validation over the last 3 training seasons).

| head | leaves | min leaf | lr | L2 | feature frac | bagging | half-life | trees |
|---|---|---|---|---|---|---|---|---|
| margin | 7 | 300 | 0.05 | 10.0 | 0.7 | 0.8 | none | 22 |
| total | 7 | 300 | 0.05 | 10.0 | 0.7 | 0.8 | 4 seasons | 12 |

**Benchmark** (walk-forward, test seasons 2012-13 → 2025-26; rung-3 walk-forward 2012-13..2025-26 (inner search over gbm.GRID per fold), cache/full2/rung3_merged_preds.csv.gz, 2026-10-18; docs/MODEL.md):

| scope | n | log-loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| all test games | 18,109 | 0.5953 | 0.2049 | 0.680 | 0.014 |
| regular season | 16,898 | 0.5933 | 0.2041 | 0.681 | 0.013 |
| playoffs + play-in | 1,211 | 0.6233 | 0.2167 | 0.659 | 0.042 |
| last season (2025-26) | 1,321 | 0.5806 | 0.1991 | 0.683 | 0.026 |

**Versus the closing line** (same games):

| games with a closing moneyline | n | model LL | market LL | model - market (paired SE) |
|---|---|---|---|---|
| regular season | 16,321 | 0.5929 | 0.5918 | +0.0010 ± 0.0005 |
| regular season, last 3 seasons | 3,687 | 0.5785 | 0.5785 | +0.0000 ± 0.0009 |
| playoffs | 1,090 | 0.6215 | 0.6159 | +0.0056 ± 0.0032 |

**Calibration** (all test games, 10 bins):

| P(home) bin | n | mean predicted | observed |
|---|---|---|---|
| 0.0-0.1 | 59 | 0.081 | 0.034 |
| 0.1-0.2 | 556 | 0.160 | 0.158 |
| 0.2-0.3 | 1,403 | 0.256 | 0.230 |
| 0.3-0.4 | 2,156 | 0.351 | 0.373 |
| 0.4-0.5 | 2,244 | 0.444 | 0.450 |
| 0.5-0.6 | 2,342 | 0.558 | 0.544 |
| 0.6-0.7 | 3,380 | 0.651 | 0.631 |
| 0.7-0.8 | 3,306 | 0.748 | 0.738 |
| 0.8-0.9 | 2,208 | 0.845 | 0.835 |
| 0.9-1.0 | 455 | 0.925 | 0.941 |

## gbm_opinion (`gbm_opinion@v1-8b7ad87b`)

Market input: no. Search grid the fixed values come from: 4 configs (inner time-ordered validation over the last 3 training seasons).

| head | leaves | min leaf | lr | L2 | feature frac | bagging | half-life | trees |
|---|---|---|---|---|---|---|---|---|
| margin | 15 | 300 | 0.05 | 10.0 | 0.7 | 0.8 | none | 89 |
| total | 7 | 300 | 0.05 | 10.0 | 0.7 | 0.8 | 4 seasons | 31 |

**Benchmark** (walk-forward, test seasons 2012-13 → 2025-26; rung-3 walk-forward 2012-13..2025-26 (inner search over gbm.GRID per fold), cache/full2/rung3_merged_preds.csv.gz, 2026-10-18; docs/MODEL.md):

| scope | n | log-loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| all test games | 18,109 | 0.6049 | 0.2091 | 0.669 | 0.006 |
| regular season | 16,898 | 0.6022 | 0.2079 | 0.672 | 0.005 |
| playoffs + play-in | 1,211 | 0.6435 | 0.2257 | 0.633 | 0.050 |
| last season (2025-26) | 1,321 | 0.5904 | 0.2029 | 0.686 | 0.029 |

**Versus the closing line** (same games):

| games with a closing moneyline | n | model LL | market LL | model - market (paired SE) |
|---|---|---|---|---|
| regular season | 16,321 | 0.6017 | 0.5918 | +0.0099 ± 0.0013 |
| regular season, last 3 seasons | 3,687 | 0.5870 | 0.5785 | +0.0086 ± 0.0025 |
| playoffs | 1,090 | 0.6430 | 0.6159 | +0.0271 ± 0.0069 |

**Calibration** (all test games, 10 bins):

| P(home) bin | n | mean predicted | observed |
|---|---|---|---|
| 0.0-0.1 | 68 | 0.074 | 0.073 |
| 0.1-0.2 | 495 | 0.156 | 0.137 |
| 0.2-0.3 | 1,095 | 0.253 | 0.255 |
| 0.3-0.4 | 1,825 | 0.354 | 0.357 |
| 0.4-0.5 | 2,654 | 0.450 | 0.445 |
| 0.5-0.6 | 3,323 | 0.551 | 0.550 |
| 0.6-0.7 | 3,283 | 0.646 | 0.644 |
| 0.7-0.8 | 3,014 | 0.750 | 0.738 |
| 0.8-0.9 | 1,931 | 0.846 | 0.833 |
| 0.9-1.0 | 421 | 0.929 | 0.919 |

## Live performance (latest monitoring snapshot)

| model | window | n | log-loss | vs market (SE) | ECE |
|---|---|---|---|---|---|
| gbm_market | last300 | 43 | 0.5552 | -0.0021 ± 0.0120 | 0.138 |
| gbm_market | season | 43 | 0.5552 | -0.0021 ± 0.0120 | 0.138 |
| gbm_opinion | last300 | 43 | 0.6045 | +0.0472 ± 0.0328 | 0.186 |
| gbm_opinion | season | 43 | 0.6045 | +0.0472 ± 0.0328 | 0.186 |

## Versions in evaluation

None.

## Recent registry events

| time | version | action | reason |
|---|---|---|---|
| 2026-10-26 07:02 | `dfs_engine@v8-e0f16e1a` | promote | newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `dfs_engine@v7-adb358e9` | retire | replaced by dfs_engine@v8-e0f16e1a: newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `dfs_engine@v8-e0f16e1a` | register | first dfs_engine run with pipeline pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_dist@v8-faba2471` | promote | newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_dist@v7-1fb30f93` | retire | replaced by player_dist@v8-faba2471: newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_dist@v8-faba2471` | register | first player_dist run with pipeline pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_rates@v8-656bfefd` | promote | newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_rates@v7-b9f26a89` | retire | replaced by player_rates@v8-656bfefd: newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_rates@v8-656bfefd` | register | first player_rates run with pipeline pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_minutes@v8-70069014` | promote | newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_minutes@v7-1549e62e` | retire | replaced by player_minutes@v8-70069014: newest DFS pipeline version pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |
| 2026-10-26 07:02 | `player_minutes@v8-70069014` | register | first player_minutes run with pipeline pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr |

## Known limitations

- **Injury data only from 2026.** Official/ESPN injury reports are archived from the 2026-27 season on. The 2012-26 benchmark never saw a real injury report: availability there is a leak-safe proxy (below), so live accuracy on days with real reports is not yet measured.
- **Availability proxy.** Backtest player strength uses the engine's `avail_tight` mode: the rotation regulars who actually played (projection >= players.REGULAR_MIN) plus fringe players on the recent roster. It ignores real absences of players under ~20 minutes and still knows which regulars played, so true production performance sits between this and the (leaky) full oracle. Live days use the injury report or the full recent roster.
- **No edge against the spread.** Every ATS and over/under record in the benchmark is inside noise and below the 52.4% break-even at -110. `gbm_market` is about as good as the closing line (it uses the line); `gbm_opinion` is ~0.01 log-loss worse than the line.
- **Early-season priors.** In the first weeks of a season rolling form, pace and rating windows are padded with a regressed prior-season prior and player ratings carry over the summer; new rosters, rookies and trades are only learned as games are played.
- **Playoffs.** Fewer games, tighter rotations; the market is clearly better there (see vs-market rows).
- **Market input timing.** `gbm_market` is trained on closing lines but priced on the latest pre-tip line available in the morning, a slightly noisier input.

## Promotion policy

A new version must beat the champion on the frozen walk-forward benchmark by more than 2 paired standard errors of per-game log-loss, not be worse on the most recent season beyond noise, and not worsen ECE by more than 0.005; then predict live in shadow (never shown) for at least 100 games with paired log-loss ≤ the champion's and no open critical alert. Rollback restores the previous champion in one command (`courtside mlops rollback <model> --reason ...`).
