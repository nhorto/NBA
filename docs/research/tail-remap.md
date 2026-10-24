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

## Addendum (2026-10-24): the remap and `STRETCH_TR` refit on copula v1r (`+c1r`) sims

`+c1r` (copula v1 retune, `COURTSIDE_COPULA=v1r`) was approved to ship (`p8-e3.md` §4). Its
per-stat draws change the DK tails, and the remap params and `STRETCH_TR` were fit on copula v0
sims. This addendum refits both on c1r sims and re-checks the stack. Script
`research/c1r_ship_prep.py`; Mac scratch `courtside-compute/c1rship/` (rows in `out/`, logs in
`logs/`, code tree = the working tree at ce02661 + this change).

**Selection (no default flipped).**
- `tail_remap.params_path()` reads `tail_remap_params_c1r.json` only while
  `pipeline.COPULA == "v1r"`; copula v0 / v1 read `tail_remap_params.json` as before. A missing
  c1r file falls back to the v0 file (`cells` is cached per file).
- `xfactor.stretch_table()`: `STRETCH_TR_C1R` only while the remap is on AND reads its c1r
  params; otherwise `STRETCH_TR` (remap on) / `STRETCH` (off). The run-table cache key gets
  `+c1r` in that state only.
- So deleting `tail_remap_params_c1r.json` restores exactly the state E3 tested `+c1r` in (v0
  remap params + `STRETCH_TR`), with no code change.

**Copula v0 is bit-identical** (live until the flip):
- `tests/test_c1r_ship_prep.py::test_v0_on_matches_head_golden`: `run_games` with `+tr` ON and
  the shipped params equals the digest computed with HEAD ce02661's `tail_remap.py` loaded in
  place (Linux-only golden). The existing `+tr` OFF golden and the `STRETCH` / `STRETCH_TR`
  goldens (`test_tail_remap.py`, `test_xf_ship_prep.py`) pass unchanged.
- Frame level, Mac (`c1rship/p3lane.sh`, `p8e3/id_driver.py`): the live stack (copula v0, `+tr`)
  with this tree vs `p8e3`'s outputs of the pre-change tree (ce02661-equivalent):
  `project_historical` 2024-01-10 and 2020-01-15 and `project_slate` 154575 at
  2026-10-20T22:15Z: players, stats, sims (5.05M fp values) and meta identical (3/3).

### a. The remap refit (`tail_remap_params_c1r.json`)

Same method, population, shrinkage, datasets and walk-forward as `tail_remap_params.json` (§1):
`c1r_ship_prep.py fit` = `tail_remap_eval.py fit` on harness rows (4,000 sims, `--tails`) of
the live arm with copula v1r and the remap OFF, `pathA-v0.3+mf+ep+ms+c1r+p2+rx+rb`: avail
2018-19, 2019-20, 2020-21, 2023-24, 2024-25 and live 2025-26 (real tip times). Season S uses the
datasets of seasons < S; 2026-27 uses all six.
- **c1r tails are already a little narrower than v0's**, so the c1r map moves less. 2026-27
  tier q(0.90 / 0.95 / 0.99), v0 → c1r: bench 0.884 / 0.931 / 0.982 → 0.892 / 0.938 / 0.983,
  role 0.892 / 0.943 / 0.988 → 0.894 / 0.944 / 0.988, star 0.890 / 0.945 / 0.989 → 0.897 / 0.947
  / 0.989. (The v0 file was fit on an older arm, `+mf+p2`, with the 19:30-tip live rows.)

**Held-out upper-tail PIT** (P(play) ≥ 0.85, as §3; avail pooled = 2019-20, 2020-21, 2023-24,
2024-25, each on its walk-forward params; % above the level, ± SE by date):

| set · tier | n | > .90 c1r OFF → ON | > .95 OFF → ON | > .99 OFF → ON |
|---|---|---|---|---|
| **avail pooled · all** | 76017 | 9.56 → **9.92** ± .11 | 4.40 → **4.81** ± .08 | 0.78 → **0.93** ± .04 |
| avail pooled · bench | 15608 | 9.19 → **9.85** ± .23 | 3.67 → **4.54** ± .16 | 0.53 → **0.89** ± .08 |
| avail pooled · role | 42707 | 9.59 → **9.95** ± .14 | 4.48 → **4.82** ± .10 | 0.84 → **0.91** ± .05 |
| avail pooled · star | 17702 | 9.80 → **9.91** ± .23 | 4.85 → **5.02** ± .17 | 0.86 → **0.99** ± .07 |
| live 2025-26 · all | 13462 | 8.58 → 9.03 ± .23 | 3.66 → 4.12 ± .16 | 0.65 → 0.84 ± .09 |
| live · bench | 2423 | 9.62 → 10.52 ± .63 | 4.09 → 5.20 ± .39 | 0.45 → 1.11 ± .21 |
| live · role | 8737 | 8.42 → 8.86 ± .26 | 3.64 → 3.99 ± .20 | 0.66 → 0.76 ± .10 |
| live · star | 2302 | 8.08 → 8.08 ± .62 | 3.30 → 3.43 ± .36 | 0.83 → 0.87 ± .20 |

- Avail pooled is within ±0.2 pt of nominal everywhere except bench q95 (−0.46, 2.8 SE) and q99
  (−0.11). Single seasons scatter ±0.5 pt as before (all-tier q95: 4.82 / 4.41 / 5.25 / 4.69).
- Live role / star stay ~1 pt too wide at q95, as with v0 (§3: 4.28 / 3.30): the live path's
  availability and minutes uncertainty, which an avail fit does not see.

### b. DK CRPS: `+c1r +tr` (c1r params) vs the live stack

`p8_e3_eval.py dk` (the E3 metric: E[min] ≥ 10 in either arm, DNP = 0, two-way date × player
SEs, 4 E[min] tiers, E4 value metric on the DFS population; FPPG anchor on the harness / live
sets, the real salary on RotoGuru). OFF = the live stack `pathA-v0.3+mf+ep+ms+p2+rx+rb+tr` rows
of `p8e3/out` (same tree apart from this change; real tip times). ON = the same arm with
copula v1r and the c1r remap params, same seeds.

| set | n | CRPS off | diff | SE 1-way | SE 2-way | z | MAE diff (z) | mean diff | worst tier (z) | value diff (z) | E3 `+c1r` (v0 params) z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-24 | 21914 | 5.6060 | -0.0018 | 0.0005 | 0.0005 | -3.28 | +0.0009 (+1.44) | +0.074 | starter +0.64 | +0.0002 (+3.25) | -1.43 |
| 2024-25 | 22068 | 5.6037 | -0.0019 | 0.0006 | 0.0006 | -3.22 | +0.0009 (+1.24) | +0.074 | starter +0.52 | +0.0002 (+2.29) | -3.30 |
| 2023-25 pooled | 43982 | 5.6049 | -0.0018 | 0.0004 | 0.0004 | -4.26 | +0.0009 (+1.81) | +0.074 | starter +0.81 | +0.0002 (+3.80) | -3.08 |
| live | 15061 | 5.6807 | -0.0036 | 0.0006 | 0.0007 | -4.80 | +0.0010 (+1.43) | +0.064 | starter -1.01 | +0.0002 (+3.07) | -5.63 |
| live10 | 14934 | 5.5705 | -0.0022 | 0.0006 | 0.0007 | -3.12 | +0.0011 (+1.61) | +0.065 | starter -0.50 | +0.0002 (+2.54) | -4.03 |
| RG 2019-20 | 18841 | 5.5346 | -0.0021 | 0.0006 | 0.0006 | -3.79 | +0.0039 (+4.88) | +0.109 | starter +0.49 | +0.0000 (+1.18) | -3.06 |
| RG 2020-21 | 19243 | 5.4955 | -0.0013 | 0.0006 | 0.0006 | -2.00 | +0.0038 (+4.18) | +0.098 | starter +1.81 | +0.0001 (+1.82) | -2.96 |
| RG pooled | 38084 | 5.5149 | -0.0017 | 0.0004 | 0.0004 | -3.90 | +0.0038 (+6.08) | +0.104 | starter +1.54 | +0.0000 (+2.12) | -- |
| live10 with `+pq` (both arms; offsets leave-one-month-out) | 14912 | 5.5600 | -0.0024 | 0.0006 | 0.0007 | -3.37 | +0.0009 (+1.33) | +0.064 | starter -0.75 | +0.0002 (+3.10) | -- |

- **The combination passes:** negative on all six sets (z −2.0 to −4.8), no tier worse by 2 SE
  (worst RG 2020-21 starter +1.81), value metric better on every set (z +1.2 to +3.8).
- **Means move up** (+0.06 to +0.11 DK; the c1r map pulls the upper tail in less than the v0 map
  did). The live stack runs low (bias −0.23 harness, −0.15 RotoGuru), so the bias shrinks; MAE
  of the mean gets slightly worse (+0.001 harness / live, +0.004 RotoGuru, z +6.1 pooled), as
  the mean moves away from the median of a right-skewed outcome. CRPS is the judge, as for `+tr`.

**The refit's own increment** (ON vs E3's `+c1r` rows, i.e. c1r with the v0 remap params; same
seeds):

| set | diff | z | star | starter | rotation | bench |
|---|---|---|---|---|---|---|
| 2023-25 pooled | -0.0004 | -1.54 | -1.39 | -1.36 | -0.06 | +2.59 |
| RG pooled | +0.0001 | +0.53 | -0.91 | +1.30 | +0.63 | -0.14 |
| live | +0.0009 | +2.37 | +3.98 | +2.12 | -2.57 | +1.05 |
| live10 | +0.0010 | +2.82 | +2.67 | +2.33 | -0.78 | +0.45 |

- The refit is better on 2023-24, neutral on 2024-25 and RotoGuru, and **worse on the 2025-26
  slice** (z +2.4 / +2.8, mostly stars). The v0 params narrow the star / starter tails more than
  c1r sims need on the avail sets, which happens to offset part of the live path's too-wide tails
  (table a). Both versions beat the live stack on every set.
- The refit is the calibrated one (table a) and the one this addendum stages. Deleting
  `tail_remap_params_c1r.json` gives the E3 configuration instead (better on the slice by
  0.0008-0.0010 CRPS, z 2.4-2.8; worse on 2023-24 by 0.0009).

### c. X factor: `STRETCH_TR_C1R`

Same method as the `STRETCH_TR` addendum above (`xf_ship_prep.py stretch`, per-tier logit map,
k = 6, P(play) ≥ 0.85, RotoGuru avail rows with `--tails`), on `+c1r +tr` sims (c1r remap
params; seasons with no c1r params use 2019-20's, `--tr-key 2019-20`).
- **Seasons 2016-17 … 2020-21** (n = 98,475). 2015-16 drops out: `+ep` cannot be fit for it (no
  bio season before it), so the live arm has no bundle there. 2016-17's c1r copula uses v0's tier
  pairs (no T2.7 extracts before 2016-17). The held-out folds (2019-20, 2020-21) are unaffected.
- **Fit on all five seasons:** bench (−0.2248, 0.8571), role (−0.2063, 0.8054), star (−0.2678,
  0.7865) → `xfactor.STRETCH_TR_C1R`.

Held-out 2019-20 + 2020-21 on `+c1r +tr` sims (walk-forward refit; predicted → observed %):

| | all | bench | role | star | tag set (bench/role, ≥ 1 in 3) | Brier |
|---|---|---|---|---|---|---|
| observed | 24.24 | 19.25 ± .50 | 25.96 ± .37 | 24.77 ± .53 | | |
| STRETCH_TR (live map) | 24.20 | 19.22 | 26.25 | 23.90 | 39.6 → 39.9 ± 0.9 (n 4,089) | .17526 |
| refit (STRETCH_TR_C1R method) | 24.22 | 19.15 | 26.15 | 24.33 | 39.4 → 40.1 ± 0.9 (n 3,942) | .17529 |

- **`STRETCH_TR` is already calibrated on c1r sims** (mean −0.04 pt, tag set within 1 SE):
  c1r's remapped tails at 6 × salary are close to v0's. The refit is equivalent (Brier +.00003),
  a little better for stars (−0.4 vs −0.9 pt) and leaves the tag set ~4 % smaller. Shipped as
  `STRETCH_TR_C1R` per the selection rule; using `STRETCH_TR` instead would be just as good.

### d. Props check (report only)

`c1r_ship_prep.py propsrun` / `props`, on a copy of the 2026-10-24 04:30 prod snapshot: every
final 2026-27 game (10-20 … 10-23) with props, at the G3 lock-45 state (`props_blend.lock45_run`
per game). Each of the 21 lock-45 runs was re-projected on its slate at its created_at with the
current live stack, copula v0 and v1r (25,000 sims, the slate × kind seed, pre-fit 2026-27
bundles), and the props scored at the same market snapshot (`props.compare_game`) and graded as
`props_blend.day_rows`. v0 prices props from the NB1 approximation (`dist_source` approx); v1r
from its per-stat sims (sims), which the remap does not touch (§1), so this check does not depend
on the remap params.

| | n (graded) | games / players | LL v0 | LL v1r | v1r − v0 (SE 2-way game × player) | z |
|---|---|---|---|---|---|---|
| all | 1,671 | 28 / 226 | 0.6950 | 0.6960 | +0.0009 (0.0008) | +1.17 |
| pts | 395 | | 0.6913 | 0.6910 | −0.0003 (0.0019) | −0.15 |
| reb | 365 | | 0.7269 | 0.7289 | +0.0020 (0.0009) | +2.22 |
| ast | 285 | | 0.6934 | 0.6964 | +0.0029 (0.0012) | +2.49 |
| fg3m | 323 | | 0.6614 | 0.6623 | +0.0010 (0.0008) | +1.23 |
| pra | 303 | | 0.6989 | 0.6983 | −0.0007 (0.0015) | −0.45 |

- Neutral overall on four nights (z +1.2); rebounds and assists slightly worse with the sims
  (z +2.2 / +2.5), points / PRA flat. The market's no-vig chance is better than both (LL 0.6882;
  v0 − market +0.0069, z +1.5).
- **G3 caveat:** the props blend test (`g3-preregistration.md`) scores slates from 2026-10-24 on;
  the flip moves our props chance from the approximation to the per-stat sims part-way through
  it (one or two slates in). Its ledger records `dist_source` per row, so the test can be split
  or restarted; that is the G3 owner's call.

### e. Ship

- Prod files (staged on the Mac in `courtside-compute/c1rship/prod_stage/`):
  `bundle_2026-27_v1_cx_mf_ep_ms_c1r.json` (md5 8aa3f4f8…) and `corr_2026-27.json` (512b5eb5…).
  The bundle's v0 part equals prod's live `bundle_2026-27_v1_cx_mf_ep_ms.json` (a3998048…) key
  for key, and refitting `copula_v1` from that prod file (`p8_e3_eval.py fitbundle 2026-27`,
  `c1rship/shipchk`) reproduces the staged file byte for byte.
- Then `pipeline.COPULA_DEFAULT = "v1r"`; rollback `COURTSIDE_COPULA=v0`. The code reads
  `tail_remap_params_c1r.json` and `STRETCH_TR_C1R` automatically under v1r.
- Reproduce: Mac `courtside-compute/c1rship/`: `lane{1,2}.sh` (TR = 0 rows + bundles),
  `p2lane{1,2}.sh` (fit, TR = 1 rows, props), `p3lane.sh` (v0 identity), `p4lane.sh` (bundle
  check); geekom1: `var/tmp/c1rship/dk.sh {new|inc|pq} LABEL SET...`,
  `tail_remap_eval.py pit`, `c1r_ship_prep.py stretch` / `props` on the fetched rows.
