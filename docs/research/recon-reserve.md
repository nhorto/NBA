# Reconciliation reserve (`+rr`, IMPROVEMENTS.md B2)

2026-10-23. Switch `COURTSIDE_RECON_RESERVE` (default OFF, `pipeline.set_recon_reserve`,
model version `<arm>+rr` after `+pq`, same bundle, same seeds). **Verdict: don't ship.**
Per-player CRPS doesn't move. The team-sum bias in points moves toward 0 on the historical
harness, but moves away from 0 on the live path that production uses. The live reserve is
minutes held back for players who are already projected, so subtracting their points
double-counts. The fix belongs in the minutes engine (§4).

## 1. Design

- `reconcile.reserve_target(implied, E[reserve min], S) = implied - E[res] x bench_ppm(S)`.
  It applies at both reconcile call sites (`copula.simulate_game` and `simulate_game_v1`) through
  `copula._reconcile`.
- E[res] is the sims' mean of the reserve that `_team_minutes` actually used: each
  scenario's `_reserve_of(proj)`, plus `bo_reserve` under `+bo`. It is collected through a new
  `res_out` dict, which changes no draw.
- With ON, the team info also gets `implied`, `reserve_min` and `reserve_pts`. With OFF none of
  this code runs.
- **Rate: league, not team.** It is fit on box rows outside a team's last-10-game players (the
  `pregame_panel` candidate rule) and pooled over the 3 seasons < S (walk-forward):
  `recon_reserve_params.json`, produced by `research/recon_reserve.py fit`. Values:
  0.401 (2019-20), 0.416 (2020-21), 0.408 (2023-24), 0.420 (2024-25), 0.418 (2025-26),
  0.429 (2026-27).
- Why league: a team gets only ~250 reserve minutes a season, and its reserve players change
  every year.
  - Out of sample, the team-season MSE of the league rate equals or beats a team rate with no
    shrinkage in every season (e.g. 2023-24: 1.12 vs 1.93), and is within 0.1 of the best
    shrunk team rate.
  - The reserve rate at openers (0.385) equals the season rate (0.386).
  - Debutants score 0.35 per minute, returning players 0.44.

## 2. OFF is bit-identical

Scratch Mac box, HEAD tree (`git archive`) vs this tree, same DB copy and bundles, live arm.
- `id_driver` hist 2024-01-10 and 2020-01-15, plus slate 154575 at 2026-10-20T22:15Z:
  players, stats, every sim array (fp / minutes / ids) and meta are identical (3/3).
- `research/p2_identity.py` live / live10 / avail 2026-01-15, opener 2026-10-20 and avail
  2024-01-10: sha1 identical (5/5).
- `tests/test_recon_reserve.py` (7 tests):
  - OFF adds no team-info keys;
  - ON leaves the minutes, margins and totals unchanged and moves only the target, at both
    copula paths;
  - ON with a zero rate draws exactly OFF;
  - the rate table is walk-forward;
  - the versions compose.

## 3. Results (4,000 sims, live arm cx + p2 + copula v0; ON - OFF, SE clustered by date)

`n` is the number of rows with E[min] >= 10. "Worst tier" is the largest CRPS z among
star / starter / rotation / bench.

| set | arm | n | DK CRPS off | ΔCRPS (SE) | DK MAE off | ΔMAE (z) | worst tier CRPS z |
|---|---|---|---|---|---|---|---|
| 2023-24 harness | base | 21746 | 5.653 | -0.0004 (0.0007) | 8.020 | -0.0035 (-3.0) | bench +1.4 |
| 2024-25 harness | base | 21865 | 5.646 | -0.0012 (0.0010) | 7.995 | -0.0030 (-1.6) | star +0.6 |
| 2025-26 live | base | 14939 | 5.739 | -0.0029 (0.0016) | 8.151 | -0.0071 (-2.7) | **bench +2.4** |
| 2025-26 live10 | base | 14824 | 5.621 | -0.0018 (0.0016) | 7.967 | -0.0069 (-3.0) | bench +2.0 |
| 2019-20 RotoGuru | base | 18681 | 5.555 | -0.0018 (0.0012) | 7.865 | -0.0073 (-2.9) | bench +1.4 |
| 2020-21 RotoGuru | base | 19079 | 5.514 | -0.0004 (0.0012) | 7.798 | -0.0050 (-2.6) | bench +0.8 |
| 2023-24 harness | +mf | 21886 | 5.634 | +0.0008 (0.0007) | 7.988 | -0.0025 (-2.4) | **rotation +2.7** |
| 2024-25 harness | +mf | 22045 | 5.630 | +0.0016 (0.0011) | 7.971 | -0.0003 (-0.1) | star +1.3 |
| 2025-26 live | +mf | 15039 | 5.731 | -0.0014 (0.0018) | 8.141 | -0.0052 (-1.8) | **rotation +2.1** |
| 2025-26 live10 | +mf | 14928 | 5.608 | -0.0003 (0.0018) | 7.951 | -0.0049 (-2.1) | **rotation +2.2** |
| 2019-20 RotoGuru | +mf | 18858 | 5.537 | +0.0011 (0.0010) | 7.842 | -0.0035 (-2.2) | rotation +1.3 |
| 2020-21 RotoGuru | +mf | 19266 | 5.499 | +0.0024 (0.0014) | 7.777 | -0.0025 (-1.2) | rotation +1.8 |

**CRPS is neutral:** every |Δ| is ≤ 1.8 SE.

**The MAE gain is not evidence of better calibration.** DK distributions are right-skewed, so
MAE rewards any downward shift of the mean. The per-player DK bias moves by -0.12 to -0.16.

**Tier check fails** in 4 of 12 set-arms (more than 2 SE worse). The failures are the bench
and rotation tiers, mostly with `+mf`.

### Team sums: projected players' projection - their actual (base arm; `+mf` is within 0.2)

The points projection is the reconcile target. "Others" are the points of players who were
not projected.

| set | team-games | E[res] min | others' min / pts | points: off -> on | DK: off -> on |
|---|---|---|---|---|---|
| 2023-24 all | 2460 | 1.86 | 1.90 / 0.72 | +0.61 -> -0.15 | -1.16 -> -2.30 |
| openers (team game < 5) | 150 | 3.05 | 9.43 / 3.57 | +3.86 -> +2.62 | +2.43 -> +0.59 |
| high reserve (≥ p90) | 246 | 7.48 | 8.63 / 3.31 | +3.35 -> +0.30 | +1.90 -> -2.68 |
| 2024-25 all | 2456 | 2.20 | 1.77 / 0.57 | +0.18 -> -0.74 | -0.52 -> -1.91 |
| openers | 150 | 3.30 | 8.49 / 2.89 | +2.07 -> +0.69 | +1.18 -> -0.88 |
| high reserve | 246 | 10.02 | 8.43 / 2.86 | +2.04 -> -2.16 | +2.72 -> -3.47 |
| 2019-20 all / openers | 2045 / 148 | 1.94 / 5.38 | 3.02 / 17.95 min | +0.59 -> -0.19 / +7.71 -> +5.56 | -0.84 -> -1.95 / -5.49 -> -8.27 |
| 2020-21 all / openers | 2095 / 146 | 2.18 / 4.49 | 2.50 / 13.19 min | +0.80 -> -0.11 / +5.84 -> +3.97 | +0.48 -> -0.85 / -1.63 -> -4.21 |
| **2025-26 live all** | 1616 | 2.44 | **0.27 / 0.07** | -0.08 -> -1.09 | -1.18 -> -2.72 |
| **live high reserve** | 162 | 10.84 | **0.84 / 0.12** | -1.19 -> -5.67 | -3.47 -> -10.17 |

(live10: points -0.08 -> -1.06 overall, -1.52 -> -5.93 on high-reserve team-games.)

**Historical harness.** The points bias moves toward 0 overall in 3 of 4 seasons (2024-25
overshoots to -0.74). At openers it moves toward 0 but stays positive, because E[res] is
3-5 min while the actual figure is 8.5-18 min (the debut reserve only runs under `+pre`).

**Live path.** The DK pool forces almost every player into the candidate set, so only 0.27
min per team-game go to players who were not projected. The reserve (2.4 min, 10.8 on
high-reserve team-games) is held back for nobody, and ON makes the bias 5-6 points worse
exactly where B2 expected the most gain.

**DK sums.** OFF is already slightly low overall, because the non-points stats are not
reconciled, and ON makes it lower.

## 4. Correction to B2

The over-projection B2 describes is real on the historical harness, at about +0.2 to +0.8
points per team overall and +2 to +8 at openers. It is not real in live: there, the minutes
reserve itself is the bug.
- `_fit_reserve` learns the minutes outside the *last-10* candidates, with no extras.
- At run time, the report and DK-pool extras are projected, but the reserve is not shrunk to
  account for them.

The fix that follows is in `minutes.py`: refit the reserve on panels with the run-time extras
(or subtract the extras' expected minutes), then re-test `+rr`. Once the reserve matches the
players who really aren't projected, `+rr` is the right reconcile and should be small in live.

## 5. Reproduce

```
python research/recon_reserve.py rates      # rates, team vs league, subgroups
python research/recon_reserve.py fit        # recon_reserve_params.json
# Mac ~/Developer/courtside-compute/rr: run.sh (live arm, MF= / RR= env), worker.sh over jobs/*.job,
# each = recon_reserve.py harness S {avail|live|live10} (rows_* + teams_* parquet in out/),
# then lane_dk.sh = recon_reserve.py dk OFF ON LABEL (logs/dk_*.log, out/dk_*.json, teamsum_*.json)
# identity: lane_id.sh (id_driver hist/slate + research/p2_identity.py, head vs code)
```
