# Role-aware blend (`+rb`, IMPROVEMENTS.md P1 + D1)

2026-10-23. Switch `COURTSIDE_ROLE_BLEND` (default OFF, `pipeline.set_role_blend`, model
version `<arm>+rb` after `+rr`, same bundle, same seeds). Design from B10
(`docs/research/truth-beta-roles.md`). **Verdict: ship.** DK CRPS and MAE improve on all six sets
(CRPS z −1.7 to −4.3, two-way date × player). No tier is worse by 2 SE on the E[min] ≥ 10
population. Promoted rows are no worse. The RotoGuru lineup backtest is neutral.

## 1. Design

    mu* = mu_played + beta(r) · (L10 − mu_played),   r = planned minutes / last-10 minutes

- **Inputs** (`proj/role_blend.py`, `RoleHistory`). All are strictly before the day:
  - `mu_played` / planned = the sims' mean DK points / minutes over the sims where the player
    plays.
  - L10 = mean DK of the last up-to-10 games played (seasons S−1..S, the `BoxBaselines`
    convention).
  - last-10 minutes = mean minutes of the last up-to-10 appearances (seasons S−3..S).
  - returning = ≥ 3 team games missed right before the game, after an appearance this season.
- **beta** by bucket r < 0.8 / 0.8–0.95 / 0.95–1.05 / 1.05–1.25 / > 1.25.
  - It is 0 for returning players, players with no history and players with no L10.
  - Fit per season S by grid argmin (0..0.30, step 0.02, ≥ 150 rows) of the B10 CRPS proxy, on
    the OFF harness rows of all datasets of seasons < S. The rows are 2018-19 (new, so 2019-20
    has a true walk-forward fold), RG 2019-20/2020-21, harness 2023-24/2024-25 and live 2025-26.
  - Output: `research/role_blend_eval.py fit` → `src/courtside/proj/role_blend_params.json`.
    The file stores the betas, the training datasets and each bucket's summed loss curve.
  - Fitted values, which match B10:

| S | <0.8 | 0.8–0.95 | 0.95–1.05 | 1.05–1.25 | >1.25 |
|---|---|---|---|---|---|
| 2019-20 (on 2018-19) | 0 | .12 | .18 | .06 | .04 |
| 2020-21 | 0 | .12 | .16 | .06 | .06 |
| 2023-24 | 0 | .14 | .16 | .10 | .06 |
| 2024-25 | 0 | .16 | .14 | .08 | .08 |
| 2025-26 | 0 | .16 | .14 | .08 | .06 |
| 2026-27 (live) | 0 | .16 | .14 | .10 | .06 |

- **Where the shift lives: the projection pipeline.**
  - `build_games` attaches the inputs as `rb_*` columns of each TeamSpec frame (ON only).
    `run_games` applies `RB.apply` right after the sims are drawn and before `summarize_games`:
    every played sim moves by delta, and DNP sims stay 0.
  - So the stored `player_projections` / stats, the sims npz, `inputs_json.mu_played` (plus a new
    `rb_shift`), the display, cash, Top N, builder candidates, the field proxy's "ours" share and
    the contest-builder sims all carry mu*. That is D1 with no optimizer changes (`runner.py`
    unchanged), and the display and DFS cannot disagree.
  - The DFS runner would have been the wrong place. It would leave the stored projection, the
    grades and the web on raw mu, and it would need a second copy of the box history.
  - `copula.py` is untouched; the shift is applied after the draws, as `truth_sims` does it.
  - The P10 scenario arrays are saved from the shifted sims.
  - **Caveat:** under copula v1, the per-stat count sims are not shifted. The live arm is v0.
- **`truth_sims`** (`portfolio.py`): `blended=None` means the process switch. When it is true,
  beta = 0 and TRUTH_TAU stays.
  - `build_for_slate` and the portfolio CLI resolve `blended` from the run's model version
    (`RB.run_blended`), so the web contest builder is right whatever env it runs with.
  - `field.score_run` (the runner hook, same process as the projection) uses the switch.
  - This composes with the `gs` split: `blended` is passed to the medium truth and to
    `strength_range`.
- **Weekly refit:** `RB.refit_weekly(conn, season, until=..., weight=1)`. It is not scheduled.
  - It adds the season's graded live rows to the stored training curves of S and re-minimises
    each bucket. Live rows from ON runs are un-shifted first, via `rb_shift`.
  - At about 150 graded rows a night, one week moves a bucket only a little against ~100k
    training rows. Raise `weight` to make the refit track this season faster.

## 2. OFF is bit-identical

Scratch Mac `courtside-compute/rb`. Base = snapshot of the tree before `+rb`; code = base + this
change. Same DB copy and bundles, live arm (cx + p2 + copula v0 + mf).
- `id_driver` hist 2024-01-10 and 2020-01-15, and slate 154575 at 2026-10-20T22:15Z: players,
  stats, every sim array and meta are identical (3/3).
- `research/p2_identity.py` live / live10 / avail 2026-01-15, opener 2026-10-20 and avail
  2024-01-10: the sha1 values are identical (5/5).
- **ON shift check** (slate 154575, hist 2024-01-10):
  - Minutes are identical and DNP sims are 0.
  - Each player's played sims move by one constant, equal to `rb_shift` (max error 4e-6).
  - 214 and 80 players are shifted; the mean |Δ mean| is 0.19 and 0.23 DK.
- `tests/test_role_blend.py` (12 tests):
  - the switch, and the version composing with `+rr`;
  - buckets / delta;
  - run_games OFF never calls the blend, even with `rb_*` columns present;
  - ON moves only the played sims, by exactly beta·(L10 − mu_played), and dk_mean by
    delta·P(play);
  - ON with zero betas draws exactly OFF;
  - RoleHistory: strictly-before rule, returning, season window;
  - the params are walk-forward, with the fallback;
  - the fit recovers a known shift;
  - the weekly refit;
  - truth_sims skips beta only when blended;
  - run_blended.

## 3. Results (4,000 sims; ON − OFF; SE two-way date × player, `paired_cluster2`)

Population = E[min] ≥ 10 in either arm, DNP = 0 (as `switch-ledger.md`). Tiers by
E[min | plays]. Promoted = r > 1.05.

| set | n | CRPS off | ΔCRPS (SE) | z | ΔMAE | z | worst tier CRPS z | promoted ΔCRPS (z) | promoted ΔMAE z |
|---|---|---|---|---|---|---|---|---|---|
| 2023-24 harness | 21886 | 5.634 | −0.0070 (0.0021) | −3.30 | −0.0097 | −2.84 | bench −0.59 | −0.0045 (−1.77) | −3.57 |
| 2024-25 harness | 22045 | 5.630 | −0.0044 (0.0019) | −2.29 | −0.0056 | −2.03 | bench +0.92 | +0.0013 (+0.31) | −1.38 |
| 2025-26 live | 15039 | 5.731 | −0.0074 (0.0027) | −2.74 | −0.0124 | −2.77 | starter −1.30 | −0.0089 (−1.66) | −3.33 |
| 2025-26 live10 | 14928 | 5.608 | −0.0046 (0.0028) | −1.66 | −0.0085 | −1.99 | star −0.73 | −0.0038 (−0.72) | −3.27 |
| 2019-20 RotoGuru | 18858 | 5.537 | −0.0052 (0.0017) | −2.99 | −0.0057 | −1.76 | bench +1.79 | −0.0040 (−1.82) | −4.47 |
| 2020-21 RotoGuru | 19266 | 5.499 | −0.0072 (0.0017) | −4.32 | −0.0102 | −3.39 | bench −0.51 | −0.0076 (−2.50) | −5.17 |
| **live, +rx arm** (shipped) | 15108 | 5.725 | −0.0081 (0.0025) | −3.23 | −0.0132 | −3.16 | starter −1.04 | −0.0107 (−2.26) | −3.90 |

- **Arm.** The six main sets are `pathA-v0.3+mf+p2`: they ran before `+rx` became the default
  (commit 7b7b693). The live slice was re-run on `+mf+p2+rx` and the result is slightly stronger.
- **All projected rows** (incl. E[min] < 10): CRPS z −3.35 / −2.30 / −2.51 / −1.44 / −2.72 /
  −4.42.
  - One tier fails on this secondary population: 2019-20 bench, +0.0015 CRPS (0.08 %), z +2.57.
    Its E[min] ≥ 10 counterpart is +1.79.
- **Where the gain is.**
  - 0.8–0.95 (players planned slightly below their recent minutes): CRPS −0.011 to −0.023,
    z −1.4 to −4.9. The bias moves from about −0.4/−0.7 to about 0.
  - On promoted rows MAE improves strongly in every set (bias +0.1/+0.4 → about 0) and CRPS is
    flat to better. The worst is 2024-25 > 1.25 at +0.008 (z +0.73).
  - The r < 0.8 and returning rows are unchanged (beta = 0) by design.
- **The gain is larger than B10's proxy estimate** (−0.004 to −0.007 vs −0.002 to −0.006). The
  live set is ≥ 2 SE (−2.7; −3.2 on +rx). live10 is −1.7. Every set's sign is negative.

### RotoGuru lineup backtest (actual RotoGuru DK points; ON − OFF, SE by date)

| season | method | OFF | ON | Δ (SE) | / hindsight |
|---|---|---|---|---|---|
| 2019-20 (149 dates) | cash | 275.91 | 276.27 | +0.36 (1.23) | 0.6921 → 0.6930 |
| | mean | 276.04 | 277.69 | +1.65 (1.41) | 0.6925 → 0.6966 |
| | Top-20 mean / best | 276.18 / 308.70 | 275.45 / 308.18 | −0.73 (0.53) / −0.52 (0.88) | 0.6928 → 0.6910 |
| 2020-21 (139 dates) | cash | 273.79 | 273.38 | −0.41 (1.30) | 0.6927 → 0.6917 |
| | mean | 273.81 | 272.20 | −1.62 (1.52) | 0.6928 → 0.6887 |
| | Top-20 mean / best | 270.75 / 301.81 | 271.03 / 302.42 | +0.28 (0.43) / +0.61 (0.92) | 0.6850 → 0.6857 |

Neutral: every |Δ| ≤ 1.4 SE, with signs split across seasons. The projection gain is per-player
calibration, and it is too small to move single-lineup scores measurably over ~140 dates.

## 4. Notes

- Salary is not part of this step (B10 item 3).
- Live-season betas (2026-27) are fit on everything through the 2025-26 live slice. Run
  `refit_weekly` once there are about 10 graded nights.
- The harness cache hash (`backtest.CODE_FILES`) does not cover `role_blend.py` or its params.
  Clear `reports/backtest/cache/*/<S>/*rb*` after changing either.

## 5. Reproduce

```
# Mac ~/Developer/courtside-compute/rb: run.sh TREE SCRIPT (env RB=0/1, RX=0/1), worker.sh over jobs/*.job
python research/role_blend_eval.py harness S {avail|live|live10} [--lineups]   # rows_* / lineups_*
python research/role_blend_eval.py fit out                                      # role_blend_params.json
python research/role_blend_eval.py dk OFF ON LABEL; ... lineups OFF ON LABEL    # lane_dk.sh
# identity: lane_id.sh (id_driver hist/slate/same/shiftcheck + p2_identity, base vs code); lane_rx.sh
```
