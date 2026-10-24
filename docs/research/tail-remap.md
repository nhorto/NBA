# Upper-tail quantile remap (`+tr`, IMPROVEMENTS.md D6)

2026-10-23. Switch `COURTSIDE_TAIL_REMAP` (default OFF, `pipeline.set_tail_remap`, model version
`<arm>+tr` after `+tm`, same bundle, same seeds). Evidence: `xfactor.md` §2 (upper tails too wide,
worst for bench players). **Verdict: recommend ship** (owner decision), with the caveats in §6.
DK CRPS improves on all six sets (z −1.2 to −6.2, two-way), no tier is worse by 2 SE, and on the
pooled held-out avail seasons every tier's upper-tail PIT share is within ±0.3 pt of nominal at
q90 / q95 / q99. The live slice stays about 0.8 pt too wide for role and star players.

## 1. Design (`proj/tail_remap.py`, params `tail_remap_params.json`)

    U      = F_played(y)                  the PIT of the outcome in the player's played sims
    q_c(v) = v-quantile of U              v in {0.75, 0.90, 0.95, 0.98, 0.99}, per cell c
    x'     = Q_played(g_c(F_played(x)))   for each played sim x of a player in cell c

- **g_c** is piecewise linear through (0, 0), (0.5, 0.5), (v, q_c(v)) and (1, 1).
  - It is monotone and the identity at and below the median: g(0.5) = 0.5 is imposed.
  - The remapped v-quantile is the old q_c(v)-quantile, which the outcome exceeds with share
    1 − v. So the map is exact at the grid levels.
  - It only re-labels the upper half of the player's own sims (mid-rank F, tied values stay
    tied). Every sim keeps its rank, so the copula's rank structure is unchanged. The minutes,
    the DNP sims (0) and every value at or below the player's median are bit-identical.
- **Cells:** model tier (bench / role / star, the copula `tier`) × E[min | plays] band (< 8,
  8–16, 16–24, 24–32, 32+).
  - Each level is shrunk toward the tier's with 2,000 prior rows, and the tier's toward the
    pooled value.
  - A player with no cell uses his tier. A player with < 20 played sims is left as is.
- **Fit** (`research/tail_remap_eval.py fit`): U = (pit − (1 − p)) / p on harness rows where the
  player played and sim P(play) ≥ 0.85. This is the population `xfactor.md` §2 grades.
  - With P(play) ≥ 0.2 or ≥ 0.5, the held-out bench shares stayed ~0.3 pt short.
  - Rows: the OFF rows of `courtside-compute/rb/out` (avail 2018-19, 2019-20, 2020-21, 2023-24,
    2024-25 and live 2025-26 `+rx`).
  - Walk-forward: season S uses the datasets of seasons < S. 2019-20 uses 2018-19 only. 2026-27
    (live) uses all six.
- **Fitted shape** (2026-27): q(0.99) is 0.981 (bench 16–24) to 0.991 (role 32+). q(0.95) is
  0.927 to 0.947, and q(0.75) is 0.73 to 0.76. Bench 16–24 is the widest, as in §2 of the X
  factor doc.
- **Where it runs:** `pipeline.run_games`, after the draws and after the role blend (if on), and
  before `summarize_games`.
  - So the stored projections and stats, the sims npz, the cards, the X factor, the field
    proxy's "ours" and the contest builder all see it.
  - `info` gets `tr_shift` (the mean change of the played sims). `mu_played` moves by it, and
    `inputs_json.tr_shift` is stored (ON only). `meta.tail_remap` holds {remapped, mean_shift}.
  - The remap commutes with the role blend's constant shift. With `+rb`, the remap changes the
    mean by exactly as much as without it.
- **Caveats:**
  - Under copula v1 the per-stat count sims are not remapped (the live arm is v0).
  - `role_blend.live_rows` un-shifts `rb_shift` only. A weekly rb refit on `+tr` runs would see
    remapped q10–q90 (only q75 / q90 move).

## 2. OFF is bit-identical

- **Unit tests** (`tests/test_tail_remap.py`, 13 tests):
  - the switch and env flag, and the version composing as `+rb+tm+tr`;
  - the knots are monotone with the median fixed;
  - remap_column keeps the lower half, the order and ties, and is exact under identity knots;
  - a fit on a 1.4× too-wide normal model restores 10 / 5 / 1 % shares;
  - small-cell shrinkage, walk-forward params with fallback, and the shipped params file;
  - run_games OFF never calls the remap;
  - **OFF digest == the pipeline of HEAD 102ed53** on the copula fixture (Linux-only golden);
  - ON relabels the upper half only (ranks, minutes and DNP kept, dk_mean moves by
    tr_shift × P(play), q95 down);
  - identity knots give exactly OFF;
  - the remap runs after the role blend.
- **Frame level, Mac** `courtside-compute/tr` (same DB copy and bundles, live arm):
  - `head` = `git archive` of the HEAD before the coordinator's commit (3c40fbf).
  - `base` = the tree before `+tr`. `code` = the tree with `+tr`.
  - `head2` = `git archive 102ed53`. `mine` = head2 + this change's files.
  - Cases: `id_driver` hist 2024-01-10, hist 2020-01-15, and slate 154575 at
    2026-10-20T22:15Z (the live path).
  - code vs base and code vs head: players, stats, every sim array and meta are identical (6/6).
    `p2_identity` (live / live10 / avail 2026-01-15, opener, avail 2024-01-10): 5/5 identical
    hashes.
  - mine vs head2, with RB = 0 and RB = 1: all 3 cases identical in both (6/6), including the
    `+rb` frames (rb_shift column).
- **ON check** (`id_driver trcheck`, slate 154575 and hist 2024-01-10, both also with RB = 1):
  - Minutes are identical and DNP sims are 0.
  - All 106 / 286 players moved. Every value at or below the player's OFF median is unchanged
    (max |Δ| 0). Ranks are kept and the max is never exceeded.
  - Played mean −0.20 / −0.18 DK, played q95 −1.8 / −1.4, q99 −3.0 / −2.6. dk_mean (all sims)
    −0.12.

## 3. Upper-tail PIT (held-out, P(play) ≥ 0.85, unconditional randomised PIT as `xfactor.md` §2)

Live arm `+mf+p2+rx+rb` (the shipped arm) OFF → ON. Shares (%) above the level, ± SE clustered
by date. Nominal is 10 / 5 / 1. The pooled avail row covers 2019-20, 2020-21, 2023-24 and 2024-25
(76,017 rows), each fit on seasons < S only.

| set · tier | n | > .90 OFF → ON | > .95 OFF → ON | > .99 OFF → ON |
|---|---|---|---|---|
| **avail pooled · all** | 76017 | 9.22 → **10.08** ± .11 | 4.28 → **4.99** ± .08 | 0.79 → **1.02** ± .04 |
| avail pooled · bench | 16072 | 8.46 → **9.96** ± .23 | 3.27 → **4.79** ± .17 | 0.49 → **0.98** ± .08 |
| avail pooled · role | 42236 | 9.55 → **10.24** ± .15 | 4.51 → **5.09** ± .11 | 0.87 → **1.02** ± .05 |
| avail pooled · star | 17709 | 9.11 → **9.80** ± .23 | 4.63 → **4.92** ± .17 | 0.89 → **1.06** ± .08 |
| live 2025-26 · all | 13383 | 8.32 → 9.09 ± .27 | 3.67 → 4.24 ± .19 | 0.68 → 0.89 ± .08 |
| live · bench | 2483 | 8.50 → 10.07 ± .60 | 3.34 → 4.99 ± .44 | 0.32 → 1.09 ± .19 |
| live · role | 8594 | 8.52 → 9.16 ± .30 | 3.93 → 4.28 ± .25 | 0.78 → 0.87 ± .10 |
| live · star | 2306 | 7.42 → 7.81 ± .62 | 3.04 → 3.30 ± .36 | 0.69 → 0.74 ± .19 |

- **Avail pooled: every tier is within ±0.3 pt at all three levels** (largest miss: bench q95
  −0.21, star q90 −0.20). OFF missed by up to 1.7 pt (bench q95).
- **Single seasons** scatter about ±0.5 pt around nominal. That is the season-to-season spread
  OFF already has: 2020-21 is narrow (ON q95 4.62) and 2023-24 wide (5.42). One season's tier
  cell has an SE of 0.2–0.5 pt, so ±0.3 per season × tier is below the noise floor.
  Per set: `var/tmp/tr/out/pit_rb1.txt`.
- **Live is still too wide** for role and star (q95 4.3 / 3.3). That is the live path's extra
  availability and minutes uncertainty, which an avail fit does not see.
  - A live-only fit is not stable. Fit on the first half of 2025-26 live, it over-corrects the
    second half (role q95 6.5, star 6.3), because early-season tails are wider. So the pooled
    avail fit stays.
  - Closing the live gap belongs to the minutes and availability work, not this remap.
- **Without the role blend** (`+mf+p2+rx`) the table is the same to ±0.05: avail pooled ON
  10.08 / 4.97 / 1.00; bench 10.01 / 4.77 / 0.95, role 10.22 / 5.06 / 1.00, star 9.82 / 4.94 /
  1.06.

## 4. DK CRPS, MAE and mean (ON − OFF, 4,000 sims, same seeds; two-way date × player SE)

Population: E[min] ≥ 10 in either arm, DNP = 0 (as `switch-ledger.md`). Arm `+mf+p2+rx+rb`.

| set | n | CRPS off | ΔCRPS | SE 1-way | SE 2-way | z | ΔMAE | MAE z | Δmean | worst tier z |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-24 | 21886 | 5.6269 | −0.0008 | 0.0006 | 0.0007 | −1.18 | −0.0031 | −2.49 | −0.160 | starter +0.81 |
| 2024-25 | 22045 | 5.6258 | −0.0024 | 0.0005 | 0.0005 | −4.43 | −0.0049 | −3.89 | −0.134 | starter −1.01 |
| live | 15108 | 5.7169 | −0.0042 | 0.0006 | 0.0007 | −6.18 | −0.0045 | −2.97 | −0.142 | bench −1.82 |
| live10 | 14988 | 5.6008 | −0.0038 | 0.0006 | 0.0006 | −6.11 | −0.0037 | −2.73 | −0.141 | bench −1.68 |
| RG 2019-20 | 18858 | 5.5318 | −0.0018 | 0.0005 | 0.0006 | −3.13 | −0.0057 | −5.20 | −0.141 | bench +0.36 |
| RG 2020-21 | 19266 | 5.4922 | −0.0028 | 0.0005 | 0.0006 | −5.09 | −0.0062 | −5.98 | −0.130 | bench −0.37 |

- Worst tier is over the four E[min] tiers and the three model tiers.
- **Every set improves.** 5 of 6 are ≥ 3 SE. 2023-24 is −1.2 SE: 2023-24 ran wide OFF (§3), so
  less was there to gain.
- **No tier is worse by 2 SE.** The worst is 2023-24 starter at +0.81 (model-tier role +0.56).
- **The mean drops 0.13–0.16 DK per player** (−0.6 to −0.8 DK over played sims, i.e. q90 −0.6).
  - It helps where the model ran high (starter / role bias +0.1/+0.6 → about 0).
  - It hurts slightly where it ran low: model-tier star bias −0.6 → −0.7; star MAE z +1.9 / +2.5
    in 2023-24 / 2020-21, while CRPS is still flat to better.
  - The CRPS gain is the tail shape, net of this bias cost.
- **Without the role blend** the numbers match to ±0.0001 (z −1.20 / −4.58 / −6.45 / −6.20 /
  −3.06 / −5.14). The two switches compose additively.

## 5. X factor (big night = DK ≥ 6 × salary/1000; RotoGuru 2019-20 / 2020-21, P(play) ≥ 0.85)

`research/tail_remap_eval.py bignight`. The rows are the same ones as §3, `+rb` arm.

| | OFF | ON |
|---|---|---|
| observed rate | 24.24 % | 24.24 % |
| raw sim chance (mean) | 24.89 % | 24.50 % |
| live `xfactor.STRETCH` applied (mean) | 23.56 % | 23.25 % |
| tagged rows (bench/role, live-map chance ≥ 1/3), 2019-20 / 2020-21 | 1621 / 1408 | 1543 / 1376 (−5 % / −2 %) |
| tagged: predicted → observed, 2019-20 | 38.8 → 41.5 ± 1.2 % | 39.0 → 41.9 ± 1.2 % |
| tagged: predicted → observed, 2020-21 | 40.0 → 44.2 ± 1.7 % | 40.1 → 44.2 ± 1.7 % |
| refit stretch (other season), tagged pred → obs | 39.5 → 40.0 / 40.1 → 40.8 | 39.5 → 40.8 / 40.1 → 40.7 |

- **The effect on the calibrated big-night chance is small.** 6 × salary sits around a bench
  player's q75–q90, mostly below the levels the remap moves much. The raw chance falls 0.4 pt,
  toward the observed rate. Through the live STRETCH it falls 0.3 pt, and the tag set shrinks
  2–5 %.
- **The live STRETCH double-counts a little with `+tr`.** It was fit on OFF sims. On ON sims the
  calibrated mean is 1.0 pt under the observed rate (OFF: 0.7 pt). The tagged sets are still
  under-predicted, in both arms.
- **If `+tr` ships, refit `STRETCH` on `+tr` sims.** The refit params move the bench intercept
  from about −0.19 to −0.28 (2019-20 fold). That is `xfactor.py`, which is not this change's
  file. The rows for the refit are in `var/tmp/tr/out/rows_avail_20{19-20,20-21}_*+rb+tr.parquet`
  (`xfactor_calibration.py bignight` on them gives the per-tier fit).

## 6. TRUTH_TAU and the contest builder

`portfolio.truth_sims` scores each played sim on fp + β(FPPG − ours) + τ·sd·z, with
`TRUTH_TAU` = 0.18. τ came from "the residual sd (9.6) exceeds the sims' sd (9.45) by about 0.18
of a player's sd" on the T2.1 rows (2019-21, an older arm). It is symmetric noise, so it widens
both tails.

**TRUTH_TAU check** (`tail_remap_eval.py tau`).
- 30 evenly spaced dates each of 2019-20 and 2023-24, avail sims, `+rb` arm, OFF and ON with
  the same seeds.
- PIT of the actual in the truth distribution with β = 0 and τ = 0 / 0.18. P(play) ≥ 0.85.
- Residual criterion on the played rows of the same players.

| arm | τ | > .90 all / bench / role / star | > .95 all / bench / role / star | > .99 all | resid sd vs sims sd | implied τ |
|---|---|---|---|---|---|---|
| OFF | 0 | 9.57 / 8.16 / 9.73 / 10.51 | 4.77 / 3.26 / 5.06 / 5.50 | 0.95 | 9.97 vs 10.27 | 0 |
| OFF | 0.18 | 9.24 / 8.09 / 9.39 / 9.96 | 4.54 / 3.13 / 4.77 / 5.32 | 0.91 | | |
| ON | 0 | 10.42 / 9.14 / 10.66 / 11.06 | 5.58 / 4.77 / 5.75 / 5.93 | 1.24 | 9.97 vs 10.02 | 0 |
| ON | 0.18 | 10.08 / 8.81 / 10.32 / 10.70 | 5.37 / 4.57 / 5.47 / 5.87 | 1.15 | | |

(60 dates, so SEs are about 0.3–0.4 pt per cell. These dates run wider than the full-season
§3 table.)

- **The criterion TRUTH_TAU came from no longer holds on the current arm.**
  - The played residual sd (9.97) is below the sims' played sd: 10.27 OFF, 10.02 ON. The implied
    τ is 0 in both arms, and per season it is at most 0.12 (2023-24 ON).
  - The sims are already at least as wide as the errors on this population.
- **τ = 0.18 re-widens the upper tail.** It moves the shares by −0.2 to −0.35 pt at q90 / q95
  and about −0.05 to −0.1 pt at q99, in both arms. So with `+tr` it undoes part of the remap in
  the builder's scoring truth (about a third of the q95 correction for bench).
- **Conclusion:** with `+tr` ON, TRUTH_TAU is not needed for calibration. τ = 0 is the
  consistent value. Its default is unchanged here (instruction); the builder evidence below
  cannot tell τ = 0 from 0.18.

**Builder P(top 1 %) vs realised** (`tail_remap_eval.py builder`).
- RotoGuru 2019-20 / 2020-21, 57 slates (30 evenly spaced dates per season; 3 were skipped as
  too thin). The `research/portfolio_backtest.py` setup:
  - avail projections, `+rb` arm, 4,000 sims;
  - a live-like public field (½ ours + ½ season average), preset 5,000-entry GPP, N = 20;
  - greedy portfolio, no entries curve.
- Realised = ranks at the ACTUAL DK points against the same field proxy. There are no RotoGuru
  contest standings, so this is the honest-but-modelled version of "vs realised".
- SE over slates.

| arm | τ | per-lineup P(top 1 %) pred → real | P(any top 1 %) pred → real | E[payout] pred → real (fees) |
|---|---|---|---|---|
| OFF | 0.18 | 5.01 → 5.35 ± 0.97 % | 47.4 → 52.6 ± 6.7 % | 78.7 → 80.8 ± 15.3 |
| OFF | 0 | 5.06 → 4.65 ± 0.95 % | 47.8 → 45.6 ± 6.7 % | 79.1 → 70.3 ± 17.2 |
| ON | 0.18 | 5.19 → 5.18 ± 0.92 % | 48.4 → 49.1 ± 6.7 % | 81.6 → 99.2 ± 20.2 |
| ON | 0 | 5.29 → 5.35 ± 1.05 % | 48.9 → 47.4 ± 6.7 % | 82.8 → 85.7 ± 17.8 |

- **All four are calibrated within 1 SE on P(top 1 %).** One realised night per slate gives
  ±1 pt on a 5 % rate over 57 slates, so this cannot separate the arms or the τ values.
- **Paired ON − OFF** (same slates, field and seeds):
  - Predicted P(top 1 %) +0.19 / +0.23 pt (±0.03) at τ 0.18 / 0.
  - Realised share −0.18 ± 0.56 / +0.70 ± 0.55 pt.
  - Realised payout +18.4 ± 12.9 / +15.4 ± 10.6 fees: 1.4 SE, neutral to positive.
- The remap does not make the builder over-confident about the top 1 %. Its reported chance
  moves up slightly, because narrower player tails make our lineups' scores less noisy relative
  to the field.

**Not changed (other agents' files):** `portfolio.TRUTH_TAU` (`portfolio.py`) and
`xfactor.STRETCH` (`xfactor.py`).

## 7. Recommendation

- **Ship `+tr`** (owner decision): set `tail_remap.TAIL_REMAP_DEFAULT = True`; roll back with
  `COURTSIDE_TAIL_REMAP=0`. The evidence:
  - The avail held-out tails are calibrated by tier (±0.3 pt pooled).
  - CRPS is better on all six sets (5 of them ≥ 3 SE). No tier is worse.
  - The mean cost is small (−0.14 DK).
  - The X factor and the builder are unchanged within noise.
- **With it:**
  - Refit `xfactor.STRETCH` on `+tr` sims (`xfactor.py` owner). The live map now
    under-predicts by about 1 pt.
  - Consider `TRUTH_TAU` = 0 for `+tr` runs (`portfolio.py` owner), since §6's criterion gives 0.
    Wait for ≥ 10 uploaded GPP standings before changing it, because the RotoGuru builder test
    cannot tell.
- **Open:**
  - The live slice's role / star tails stay ~0.7 pt too wide at q95 (minutes / availability
    work).
  - Refit the params once 2026-27 live rows are graded (`fit` with the live rows added). Watch
    whether early-season live tails need their own cell.
  - Coordinate with `+c1r`, which also changes tails. Refit on its rows if it ships.

## 8. Reproduce

```
# geekom1: fit (rows from courtside-compute/rb/out), unit tests
uv run python research/tail_remap_eval.py fit var/tmp/tr/rows        # -> proj/tail_remap_params.json
TMPDIR=var/tmp uv run pytest -q tests/test_tail_remap.py
# Mac ~/Developer/courtside-compute/tr: run.sh TREE SCRIPT (env TR=0/1, RB=0/1, RX=1),
#   worker.sh over jobs/*.job (h TR RB SEASON PROTO), worker2.sh over jobs2 (i / b / t, RB=1)
python research/tail_remap_eval.py harness S {avail|live|live10}    # rows_<P>_<S>_<MV>.parquet (--tails)
python research/tail_remap_eval.py dk OFF ON LABEL                  # §4
python research/tail_remap_eval.py pit off=F1,F2 on=F3,F4 ...       # §3
python research/tail_remap_eval.py bignight OFF19 ON19 OFF20 ON20   # §5
python research/tail_remap_eval.py tau S --dates 30                 # §6 truth PIT / residual sd
python research/tail_remap_eval.py builder S --dates 30             # §6 builder P(top 1 %)
# identity: lane_id.sh (head/base/code), lane_id2.sh (head2 = 102ed53 vs mine, RB 0/1); id_driver trcheck
```
- Harness caches: `backtest.CODE_FILES` does not cover `tail_remap.py` or its params. Clear
  `reports/backtest/cache/*/<S>/*tr*` after changing either.

## Addendum (2026-10-23): `STRETCH_TR`, the X factor map refit on `+tr` sims

`xfactor.STRETCH_TR` is a second table, used only while `tail_remap.TAIL_REMAP` is on
(`xfactor.stretch_table`). With `+tr` OFF the chance still uses `STRETCH`, bit-identical:
- a Linux golden of `calibrate` and `run_table` against HEAD 102ed53's module,
  `tests/test_xf_ship_prep.py`;
- the existing xf goldens.

**Method.** The same as `xfactor.md` §3:
- per-tier logit p′ = a + b·logit p, k = 6, P(play) ≥ 0.85;
- RotoGuru 2015-16 … 2020-21 avail harness rows with `--tails`;
- `research/xf_ship_prep.py harness` / `stretch`.

The rows come from the shipped arm `+mf+p2+rx+rb`, ON = `+tr`:
- 2019-21 are the §5 rows (walk-forward remap params).
- 2015-19 were run on Mac scratch `courtside-compute/xfprep`. Those seasons have no remap params
  (the fit starts with 2018-19 data), so they use 2019-20's, which are fit on 2018-19 only.
  Neither held-out fold therefore sees its own season through the remap.

| tier | a | b | (STRETCH: a, b) |
|---|---|---|---|
| bench | −0.2715 | 0.8181 | −0.2902, 0.8275 |
| role | −0.1860 | 0.8142 | −0.2890, 0.7789 |
| star | −0.2532 | 0.8272 | −0.3186, 0.7724 |

**Held-out 2019-20 + 2020-21, `+tr` ON.**
- Walk-forward: each season's map is fit on the seasons before it.
- n = 35,433. Cells are predicted → observed %, ± SE clustered by date.

| | all | bench | role | star | tag set (bench/role, chance ≥ 1 in 3) |
|---|---|---|---|---|---|
| observed | 24.24 | 19.50 ± .50 | 25.90 ± .37 | 24.86 ± .52 | |
| STRETCH | 23.25 | 18.58 | 24.79 | 24.08 | 39.5 → 43.0 ± 1.0 (n 2,919) |
| refit (STRETCH_TR method) | 24.07 | 18.98 | 26.12 | 24.12 | 39.8 → 40.7 ± 0.9 (n 3,843) |

By predicted bin (refit):

| tier | 10–15 % | 15–20 % | 20–30 % | 30–40 % | 40 %+ |
|---|---|---|---|---|---|
| bench | 12.7 → 12.4 ± .8 | 17.5 → 16.5 ± .8 | 24.0 → 25.5 ± .8 | 33.2 → 37.5 ± 2.5 | 44.1 → 42.9 ± 8.3 (n 28) |
| role | 13.0 → 12.9 ± .9 | 17.8 → 16.5 ± .7 | 24.9 → 24.6 ± .4 | 33.9 → 34.3 ± .7 | 47.2 → 48.0 ± 1.6 |
| star | 13.1 → 15.0 ± 1.4 | 17.7 → 18.4 ± 1.0 | 24.5 → 24.8 ± .7 | 33.6 → 34.9 ± 1.2 | 47.0 → 50.2 ± 2.9 |

With STRETCH, role 30–40 % is 33.8 → 36.6 ± .8 and 40 %+ is 47.1 → 49.3 ± 1.9; bench 30–40 % is
33.0 → 37.8 ± 2.7. Below 10 % the bins are tiny (n < 1,000), as in §3 of the X factor doc.

- **The refit closes most of the gap:** the mean is −0.2 pt (STRETCH: −1.0), Brier .17544 vs
  .17556, and the tag set is within 1 SE (STRETCH under-predicts it by 3.5 pt).
- **Cost: the tag set grows about 30 %** before the popularity filter and the cap of 5.
- **Bench stays 0.5 pt low.**
- **Part of the gap is the arm, not `+tr`.** On the same arm OFF (2015-16, 2018-21 rows),
  STRETCH is 0.7 pt low on held-out 2019-21 (23.56 vs 24.24; tag set 39.3 → 42.7). `+tr` adds
  0.3 pt. STRETCH_TR is fit on the live arm with `+tr`, so it absorbs both. STRETCH itself is
  unchanged (the OFF rule).
- **Caveats:**
  - Like STRETCH, the walk-forward fits for 2016-21 move (role a −0.06 … −0.19). The table is
    the fit on all six seasons.
  - The switch is the process flag, not the run's model version. An old (non-`+tr`) run viewed
    with `+tr` on gets STRETCH_TR.
- **Reproduce:**
  - Mac `courtside-compute/xfprep`: `jobs{A,B}` + `worker.sh` run `harness S --tr-key 2019-20`
    with TR 0/1. `out/` holds the rows plus `stretch.json` / `stretch.txt`.
  - geekom1: `uv run python research/xf_ship_prep.py stretch OUT.json <+rb+tr rows> --off <+rb rows>`
    (rows in `var/tmp/xfprep/rows`).
