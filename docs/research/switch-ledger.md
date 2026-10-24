# Switch ledger

Each model switch decision is recorded here with its metric, the SEs (one-way by date and
two-way date x player, Cameron-Gelbach-Miller, `metrics.paired_cluster2`), the evaluation sets
and the decision. Two-way SEs matter because a mis-modelled player stays wrong for weeks, so
one-way-by-date SEs are too small (IMPROVEMENTS E2).

## +mf (minutes fix) -- shipped 2026-10-23, commit 5598456

- **Metric:** DK CRPS, paired difference `+mf+p2` minus OFF (`+p2`); negative = better. Rows with
  E[min] >= 10 in either arm, DNP = 0, same population as `research/minutes_fix_eval.py dk`.
- **Sets:** harness `avail` 2023-24 and 2024-25, `live` and `live10` (2025-26), RotoGuru 2019-20
  and 2020-21. Per-row outputs: Mac `~/Developer/courtside-compute/mf/out_dk/rows_*` (no re-run).
- **Re-computed 2026-10-23** by `research/e2_twoway_mf.py` (one-way matches the stored
  `dk_mf_*.json`, e.g. 2023-24 diff -0.01911, SE 0.00312).

| set | n | dates | players | diff | SE 1-way | SE 2-way | ratio | z 1-way | z 2-way |
|---|---|---|---|---|---|---|---|---|---|
| 2023-24 | 22012 | 160 | 520 | -0.0191 | 0.00312 | 0.00368 | 1.18 | -6.13 | -5.19 |
| 2024-25 | 22193 | 162 | 534 | -0.0236 | 0.00457 | 0.00513 | 1.12 | -5.17 | -4.61 |
| live | 15065 | 104 | 517 | -0.0088 | 0.00343 | 0.00439 | 1.28 | -2.55 | -1.99 |
| live10 | 14951 | 104 | 519 | -0.0138 | 0.00377 | 0.00471 | 1.25 | -3.66 | -2.93 |
| RG 2019-20 | 18941 | 149 | 477 | -0.0179 | 0.00483 | 0.00514 | 1.06 | -3.70 | -3.48 |
| RG 2020-21 | 19350 | 139 | 505 | -0.0117 | 0.00390 | 0.00405 | 1.04 | -3.01 | -2.90 |
| 2023-25 pooled | 44205 | 322 | 634 | -0.0214 | 0.00277 | 0.00330 | 1.19 | -7.71 | -6.47 |

- **Per-season sign:** negative (better) on every set.
- **Effect of two-way:** SEs widen 4-28 % (median ~18 %), most on the 2025-26 live sets (players
  repeat across fewer dates). Backtest sets stay at |z| > 4.6; live drops to z = -1.99 (borderline),
  live10 to -2.93.
- **Decision:** ship stands. Sign is consistent across all six sets and the pooled backtest is
  far from noise; the live-set evidence alone is no longer individually decisive at 2 sigma.

## +p2 (P(play) v2) -- shipped 2026-10-21, commit a334ff4; restated out of sample 2026-10-23

- **Metric:** DK CRPS, paired difference `pathA-v0.3+p2` minus `pathA-v0.3` (v1); negative = better.
  Rows present in both arms with a CRPS (`pq/cmp.py` rule), 26,626 rows. SE 2-way is
  `backtest._two_way_var` (date x player), not the CGM `paired_cluster2` of the +mf entry.
- **Sets:** T2.4 live-like slice 2025-12-22..2026-04-12 (104 dates, real tip times), lock-45 (`live`)
  and lock-10 (`live10`), 4,000 sims. Per-row outputs: Mac `~/Developer/courtside-compute/pq/out_v1`
  and `out_off` (stored), `b7/out_s_old_*` (p2, shipped fringe) and `b7/out_s_oldoos_*` (p2, OOS fringe).
- **Restated 2026-10-23 (B7):** the fringe "mid" P(play) model had been fit on the slice's own fringe
  rows. The OOS rows below use params fit leave-one-month-out (`research/b7_fringe_oos.py`, fringe log
  loss 0.5279 in sample vs 0.5452 OOS). Nothing else differs from the stored p2 frames (reproduced bit
  for bit). The packaged `fringe_params.json` is unchanged.

| set | n | dates | players | diff | SE 1-way | SE 2-way | ratio | z 1-way | z 2-way |
|---|---|---|---|---|---|---|---|---|---|
| lock-45, as shipped | 26626 | 104 | 571 | -0.0747 | 0.0109 | 0.0156 | 1.44 | -6.88 | -4.79 |
| lock-45, OOS fringe | 26626 | 104 | 571 | -0.0730 | 0.0111 | 0.0156 | 1.41 | -6.58 | -4.68 |
| lock-10, as shipped | 26626 | 104 | 572 | -0.0765 | 0.0103 | 0.0152 | 1.47 | -7.40 | -5.03 |
| lock-10, OOS fringe | 26626 | 104 | 572 | -0.0756 | 0.0106 | 0.0152 | 1.44 | -7.16 | -4.96 |

- **Per-tier sign (OOS fringe, lock-45 / lock-10):** bench -0.101 / -0.098, role -0.058 / -0.069, star
  -0.016 / -0.008 (-0.9 / -0.5 SE): negative on every tier; stars are not individually significant.
- **Effect of OOS fringe:** CRPS +0.0017 (3.2 SE) / +0.0008 (1.8 SE) vs the in-sample fringe, 2% / 1% of
  the gain. On the fixed E[min] >= 10 population: -0.0968 (z -5.46) / -0.1041 (z -6.28).
- **Caveat:** the stored p2 and v1 frames came from different minutes bundles. With one shared bundle:
  as shipped -6.70 / -7.19, OOS fringe -6.40 / -6.95 SE (two-way -4.59 / -4.82).
- **Decision:** ship stands. The restated headline is -6.6 / -7.2 SE (was -6.9 / -7.4); sign is
  consistent on every tier and both locks. Fringe params are not refit by this entry.


## +rb (role-aware blend, P1 + D1) -- shipped 2026-10-23 (owner OK)

- **Metric:** DK CRPS (and MAE), paired `+rb` minus OFF on identical rows. Negative = better.
  Rows with E[min] >= 10 in either arm, DNP = 0. `research/role_blend_eval.py dk`; write-up
  `docs/research/role-blend.md`.
- **Arm:** `pathA-v0.3+mf+p2`. The runs predate `+rx` becoming the default; the live slice was
  re-run on `+mf+p2+rx` (last row). 4,000 sims, same seeds.
- **Per-row outputs:** Mac `~/Developer/courtside-compute/rb/out/rows_*`.

| set | n | dates | diff | SE 1-way | SE 2-way | z 2-way | MAE diff | MAE z 2-way |
|---|---|---|---|---|---|---|---|---|
| 2023-24 | 21886 | 160 | -0.0070 | 0.0019 | 0.0021 | -3.30 | -0.0097 | -2.84 |
| 2024-25 | 22045 | 162 | -0.0044 | 0.0017 | 0.0019 | -2.29 | -0.0056 | -2.03 |
| live | 15039 | 104 | -0.0074 | 0.0026 | 0.0027 | -2.74 | -0.0124 | -2.77 |
| live10 | 14928 | 104 | -0.0046 | 0.0025 | 0.0028 | -1.66 | -0.0085 | -1.99 |
| RG 2019-20 | 18858 | 149 | -0.0052 | 0.0016 | 0.0017 | -2.99 | -0.0057 | -1.76 |
| RG 2020-21 | 19266 | 139 | -0.0072 | 0.0014 | 0.0017 | -4.32 | -0.0102 | -3.39 |
| live (+rx arm) | 15108 | 104 | -0.0081 | -- | 0.0025 | -3.23 | -0.0132 | -3.16 |

- **Per-set sign:** negative (better) on every set.
- **Tiers:** no tier is worse by more than 2 SE. The worst is RG 2019-20 bench at z +1.79.
- **Promoted rows (r > 1.05):** CRPS z between -2.50 and +0.31. MAE is better on every set
  (z -1.4 to -5.2).
- **RotoGuru lineup backtest:** neutral. Cash +0.36 / -0.41, mean +1.65 / -1.62, Top-20 mean
  -0.73 / +0.28 DK, all within 1.4 SE.
- **OFF:** bit-identical (3 id_driver cases + 5 p2_identity cases).
- **Decision:** recommend ship (owner decision). The gain is consistent and >= 2 SE on 5 of 6
  sets plus the +rx live slice, with no tier or promoted-row harm. To ship, set
  `role_blend.ROLE_BLEND_DEFAULT = True`; roll back with `COURTSIDE_ROLE_BLEND=0`.


## +tm (tier match, B11) -- leave OFF, 2026-10-23; committed OFF in 102ed53

- **Change:** the copula draw assigns rho / CV tiers from the fit's inputs (`r_dk x mpg`, `mpg`)
  instead of `mu_played` / `min_played`. Draw side only, no refit, no bundle suffix; version
  `<model version>+tm` (after `+rb`). Write-up `docs/research/tier-match.md`,
  `research/tier_match.py`.
- **Mismatch:** 10-23 % of player-games change tier (15-23 % of E[min] >= 10 rows; 2024-25 18.3 %
  of P(play) >= .5 rows; real days 13-14 %). ON: 0 mismatches on all sets.
- **Metric:** DK CRPS, paired `+tm` minus OFF on the live arm (`+mf+p2+rx`), same population as
  `research/tier_match.py dk`, SE 2-way = `metrics.paired_cluster2` (date x player).

| set | n | dates | players | diff | SE 1-way | SE 2-way | z 2-way |
|---|---|---|---|---|---|---|---|
| 2023-24 | 21886 | 160 | 516 | +0.00002 | 0.00006 | 0.00006 | +0.34 |
| 2024-25 | 22045 | 162 | 528 | +0.00004 | 0.00007 | 0.00006 | +0.65 |
| live | 15108 | 104 | 521 | -0.00007 | 0.00014 | 0.00013 | -0.54 |
| live10 | 14988 | 104 | 519 | +0.00007 | 0.00014 | 0.00012 | +0.57 |

- **Tiers:** no tier moves by more than z +1.8 (2023-24 bench +1.77); rows with a changed tier
  are no different from the rest.
- **OFF:** bit-identical (2 real days vs a `git show HEAD` copula tree, 3.9M fp values; 4 harness
  frames equal the earlier `rx` frames exactly; `tests/test_tier_match.py`).
- **Decision:** neutral by construction (v0 marginals do not depend on rho, so CRPS cannot see a
  rho change). Leave OFF. Ship only if a lineup backtest (RG 2019-20 / 2020-21) is at least
  neutral; `COURTSIDE_TIER_MATCH=1` to enable.


## +k (field ownership uncertainty, D5) -- held OFF 2026-10-23 until 3+ more uploaded standings (owner left it to the orchestrator)

- **Switch:** `COURTSIDE_FIELD_K` (default 0 = OFF; `1` = 4 fields, or an integer). The field
  version is `field-proxy-1.2+k`. Write-up: `docs/research/field-k.md`; script
  `research/field_k.py`; outputs geekom1 `var/reports/field_k_20261023/`.
- **Noise:** log-odds, centred (`mean_shift`), with Sinkhorn back to the slot totals and the cap.
  `OWN_SD` = 0.86 GPP (fitted on the 4PP MAE, 4.57 pp) and 2.65 cash (fitted on the double-ups'
  top-10 MAE, 27.4 pp), by simulation on the 10-20 uploads. Late-news beneficiaries get ×2.
- **Not a CRPS switch.** There is no paired player metric. The evidence is the one uploaded slate
  (154575, 4 contests), so no SE is meaningful.

| check (10-20 unless noted) | OFF | ON |
|---|---|---|
| field Σ own² at actuals, DU / GPP (real 4.2-5.1 / 2.43) | 3.20 / 2.06 | 4.19 / 2.56 |
| top-1 % line at actuals, DU / GPP (real 283.5 / 290.5) | 285.7 / 293.8 | 284.0 / 291.1 |
| cash-line spread between fields at actuals, DU (real same-night contests sd 5.7) | 0.8 | 7.9 |
| mean P(cash) of our 50 lineups, DU / GPP (realised 0.48 / 0.30) | 0.541 / 0.498 | 0.538 / 0.501 |
| PIT median, 4 contests | 0.52 / 0.49 / 0.35 / 0.33 | 0.52 / 0.49 / 0.36 / 0.32 |
| builder 155078 $100K Fadeaway N = 20 / 150: seconds (Mac) | 39.8 / 84.9 | 44.7 / 86.1 |
| same: E[payout] | 3,328 / 13,366 | 4,360 / 15,840 |

- **OFF:** bit-identical. There is a golden of the pre-change field.py
  (`tests/fixtures/field_k_off_golden.json`, Linux only). The builder's OFF golden in
  `tests/test_portfolio_split.py` still passes.
- **Decision:** OK to turn ON (owner decision). Set `field.FIELD_K_DEFAULT = 4`; roll back with
  `COURTSIDE_FIELD_K=0`.
  - For: the cost is small; P(cash) is unchanged; the field's structure moves toward the real
    contests.
  - Against: GPP top-1 % chances and E[payout] rise 7-31 % (a lumpier field, `field-k.md` §5.3);
    dollar figures stay uncalibrated until 10 slates.
  - PIT: the same ON and OFF on one night.
  - Refit `OWN_SD` per uploaded slate.

## cash-pcash-1 (D2: cash lineup by P(cash), Top N by mean then P(cash)) -- keep OFF, 2026-10-23; committed OFF in 102ed53

- **Metric:** realised double-up cash rate, paired by slate, ON (`COURTSIDE_CASH_PCASH=1`: max
  P(cash) on the select half of the sims over Top 10 + λ in {0, 0.15, 0.3, 0.5}) minus the current
  rule (μ − 0.3(μ − p25)). Positive = better. Cash line: preset 500-entry double-up, 44.4 % paid,
  cash-archetype proxy field (public = ½ ours + ½ season average), at the actual DK points with
  200 opponent redraws (`research/cash_calibration.py`'s T2.13 line). One row per slate, so the
  SE is one-way by date (no player dimension). `research/cash_pcash_eval.py rg`; write-up
  `docs/research/cash-pcash.md`.
- **Arm:** `pathA-v0.3+mf+p2+rx`, `protocol='avail'`, 10,000 sims.
- **Per-slate outputs:** Mac `~/Developer/courtside-compute/d2/out/rg_*.csv` (candidates with
  simulated and realised numbers in the `cands` column).

| set | slates | pick changed | rule | ON | diff | SE | z | points diff | points z |
|---|---|---|---|---|---|---|---|---|---|
| RG 2019-20 | 143 | 73 % | 0.676 | 0.704 | +0.0274 | 0.0370 | +0.74 | -0.15 | -0.09 |
| RG 2020-21 | 138 | 75 % | 0.699 | 0.608 | -0.0913 | 0.0372 | -2.45 | -6.44 | -4.02 |
| pooled | 281 | 74 % | 0.687 | 0.657 | -0.0309 | 0.0264 | -1.17 | -3.24 | -2.69 |

- **Per-set sign:** mixed (+0.7 / −2.5 z). Held-out simulated gain on changed slates +2.5 pp (z
  +15), realised −4.2 ± 3.6 pp: the gain is the model's, not the games'.
- **Top N re-rank:** new #1 − old #1 −0.7 ± 1.8 pp (neutral).
- **10-20 opener (real lines 237.25 / 247.75 / 238.75, n = 1 slate):** ON cashes 3 / 3 in both
  lock runs (258.0 pts) vs the rule 0 / 3 (lock-10, 226.75) and 1 / 3 (lock-45, 238.25).
- **Runtime:** +0.3-0.8 s per build (Mac, idle); the cash field is reused by `score_run`.
- **OFF:** bit-identical (`tests/test_cash_pcash.py`; `runner._build` on 11 stored runs, 121
  lineups, HEAD tree vs working tree byte-identical).
- **Decision:** keep OFF (default `cash_pcash.CASH_PCASH_DEFAULT = "off"`). Re-test with ≥ 10
  slates of uploaded double-up standings.

## +nsl (news since the line, G1) -- keep OFF, 2026-10-23; committed OFF 2026-10-24

- **What:** `COURTSIDE_NEWS_SINCE_LINE` (`model/line_news.py`, `predict.price_day`). It moves
  `gbm_market`'s snapshot line by 0.45 × (`gbm_opinion` margin with the reports now − with the
  reports at the line's capture time), guarded per team. Totals are not shifted (β 0). Market
  rows are tagged `+nsl`. `docs/research/line-timing.md`.
- **Phase 1 slope:** ESPN open→close move on the opinion news value is **0.45 (date-clustered SE
  0.05)**, n 879, 2025-26 with the official archive (open proxy 03:00Z the evening before,
  guarded). It is 0.41-0.50 at every guarded proxy up to noon on game day, and not ≈ 1. The
  total slope is −0.8 (0.9), i.e. none.
- **Metric:** walk-forward (β fit on earlier days) paired ON − OFF. `gbm_market` is fed the
  ESPN **open** line (no 2025-26 snapshots) with lineups at the run time. SEs are date /
  two-way date × home team (`paired_cluster2`).

| scenario | n | dates | ΔLL | SE date | SE 2-way | ΔCRPS margin (2-way SE) | ΔCRPS total (2-way SE) |
|---|---|---|---|---|---|---|---|
| open 03:00Z → tip (pre-registered) | 879 | 146 | −0.00033 | 0.00124 | 0.00126 | −0.013 (0.012) | +0.0003 (0.0019) |
| same, Regular Season | 791 | 102 | −0.00072 | 0.00133 | 0.00135 | −0.019 (0.014) | +0.0002 (0.0021) |
| open 17:00Z → tip | 899 | 151 | −0.00097 | 0.00082 | 0.00082 | −0.012 (0.009) | +0.0005 (0.0005) |
| open 17:00Z → 21:00Z run | 899 | 151 | −0.00088 | 0.00048 | 0.00056 | −0.011 (0.005) | −0.0006 (0.0005) |
| open 17:00Z the day before → tip | 881 | 148 | +0.00120 | 0.00085 | 0.00092 | +0.004 (0.008) | −0.0013 (0.0011) |

- **Context:** fed the close instead of the open, the same model scores −0.0083 LL. The shift
  recovers ≤ 12% of that. Live line age at capture is a median of 1 min (2026-27, 88 rows), so
  live news since the line is near zero.
- **OFF:** byte-identical to HEAD `predict.py` on 3 synthetic worlds and one real past day
  (`tests/test_news_since_line.py`; JSON dump + `cmp`).
- **Decision:** keep OFF (`line_news.DEFAULT = False`). The pre-registered cell is at 0.3 SE,
  and the best post-hoc cell at 1.6 SE on one partial season. Re-test on 2026-27 with real
  intraday capture times (about 60 days).

## +tr (upper-tail quantile remap, D6) -- SHIPPED 2026-10-24 (owner OK)

- **Metric:** DK CRPS (and MAE), paired `+tr` minus OFF on identical rows. Negative = better.
  Rows with E[min] >= 10 in either arm, DNP = 0. `research/tail_remap_eval.py dk`; write-up
  `docs/research/tail-remap.md`.
- **Arm:** `pathA-v0.3+mf+p2+rx+rb` (shipped), 4,000 sims, same seeds. The params are fit
  walk-forward on seasons < S. Per-row outputs: Mac `~/Developer/courtside-compute/tr/out/rows_*`.

| set | n | diff | SE 1-way | SE 2-way | z 2-way | MAE diff | MAE z 2-way | mean diff |
|---|---|---|---|---|---|---|---|---|
| 2023-24 | 21886 | -0.0008 | 0.0006 | 0.0007 | -1.18 | -0.0031 | -2.49 | -0.160 |
| 2024-25 | 22045 | -0.0024 | 0.0005 | 0.0005 | -4.43 | -0.0049 | -3.89 | -0.134 |
| live | 15108 | -0.0042 | 0.0006 | 0.0007 | -6.18 | -0.0045 | -2.97 | -0.142 |
| live10 | 14988 | -0.0038 | 0.0006 | 0.0006 | -6.11 | -0.0037 | -2.73 | -0.141 |
| RG 2019-20 | 18858 | -0.0018 | 0.0005 | 0.0006 | -3.13 | -0.0057 | -5.20 | -0.141 |
| RG 2020-21 | 19266 | -0.0028 | 0.0005 | 0.0006 | -5.09 | -0.0062 | -5.98 | -0.130 |

- **Per-set sign:** negative (better) on every set. Without `+rb` it is the same to ±0.0001.
- **Tiers:** no tier is worse by more than 2 SE. The worst is 2023-24 starter at z +0.81. Model-
  tier star MAE is z +1.9 / +2.5 in 2023-24 / 2020-21, because its negative bias grows by 0.1;
  its CRPS is flat to better.
- **Tail calibration (the D6 acceptance):** pooled held-out avail, P(play) >= 0.85. Shares above
  q90 / q95 / q99 by tier:

  | tier | OFF | ON |
  |---|---|---|
  | bench | 8.46 / 3.27 / 0.49 | 9.96 / 4.79 / 0.98 |
  | role | 9.55 / 4.51 / 0.87 | 10.24 / 5.09 / 1.02 |
  | star | 9.11 / 4.63 / 0.89 | 9.80 / 4.92 / 1.06 |

  - ON is within ±0.3 pt everywhere.
  - Live 2025-26 is still too wide for role and star (q95 4.28 / 3.30).
- **X factor:** the calibrated big-night chance is -0.3 pt and the tag set 2-5 % smaller. Refit
  `xfactor.STRETCH` on `+tr` sims if it ships.
- **Builder** (RotoGuru, 57 slates, field proxy at actual points): P(top 1 %) is calibrated
  within 1 SE in both arms and at both TRUTH_TAU values. ON - OFF realised payout is +18 ± 13
  fees (1.4 SE).
- **TRUTH_TAU:** the residual-sd criterion now implies tau = 0 in both arms, and tau = 0.18
  re-widens the q95 share by about 0.2-0.35 pt. The default is unchanged.
- **OFF:** bit-identical.
  - 6/6 `id_driver` frames against the pre-change tree and against old HEAD 3c40fbf.
  - 6/6 against HEAD 102ed53, with RB 0 and RB 1.
  - 5/5 `p2_identity` hashes.
  - The unit golden is the HEAD pipeline (Linux-only).
- **Decision:** recommend ship (owner decision). Set `tail_remap.TAIL_REMAP_DEFAULT = True`;
  roll back with `COURTSIDE_TAIL_REMAP=0`.

## xf-v2 (X factor as top-1 % leverage + leverage-tilted builder solves, D3) -- SHIPPED 2026-10-24 (owner OK; replaces xf-v1)

- **Switch:** `COURTSIDE_XF_V2` (`xfactor.XF_V2_DEFAULT = False`). ON changes the DFS pool's
  X factor tag (`xf-v2`) and adds round(100 x scale) leverage-tilted scenario solves to the
  contest builder (`info.version` +xf2). Write-up: `docs/research/xfactor-v2.md`; eval
  `research/xfactor_v2_eval.py`; outputs on the Mac `~/Developer/courtside-compute/d3/out/`
  (`rg_*.csv/jsonl`, `builder_n20.json`, `builder_n150.json`).
- **Leverage:** share of the modelled GPP field's top-1 % entries / the field's ownership,
  computed over K = 4 noisy fields (D5 machinery, explicitly). Shrunk: (share + 0.25 %) / (own + 0.25 %).
- **Metric 1 (tag):** realised leverage of the tagged set = Σ actual top-1 % share / Σ field
  ownership. Hindsight top 1 % = the 4 modelled fields scored on the actual DK points. SE is
  clustered by date (delta method); the paired v2 - v1 SE comes from a bootstrap over dates.
- **Sets:** RotoGuru 2019-20 (143 dates) and 2020-21 (138), walk-forward bundles, `avail`, 4,000
  sims, proxy ownership (`field.hist_inputs`, preset 10,000-entry large GPP).

| set | v1 tags / date | v1 leverage | v2 tags / date | v2 leverage (predicted) | v2 - v1 | z |
|---|---|---|---|---|---|---|
| RG 2019-20 | 1.97 | 1.14 ± 0.09 | 2.33 | 1.99 ± 0.15 (1.73) | +0.85 ± 0.18 | +4.8 |
| RG 2020-21 | 1.62 | 1.31 ± 0.12 | 2.59 | 1.79 ± 0.16 (1.72) | +0.48 ± 0.19 | +2.6 |
| pooled | 1.80 | 1.20 ± 0.07 | 2.46 | 1.89 ± 0.11 (1.72) | +0.69 ± 0.13 | +5.4 |

- **Real standings (10-20 4PP, 29,726 entries):** v1 tagged 0 players in both runs, v2 0 and 1
  (Mikal Bridges, not in any real top-1 % entry). Predicted vs real leverage rank corr. is about
  0 (41 players). Not informative.
- **Metric 2 (builder):** E[payout], paired per row ON - OFF and ON - ctl (ctl = the same extra
  solves with no tilt), on the `gs` R rows and on 5,000 held-out stored rows. 3 seeds, lock-45
  runs; 3 GPPs at N = 150 and 4 at N = 20.
  - N = 150 holdout: ON - OFF +4.4 to +6.5 %, ON - ctl +2.4 ± 1.1 / +3.1 ± 0.5 / +3.1 ± 1.0 %.
  - N = 20: same sign on the 150-max GPPs. On the 20-max 4PP: -4.2 ± 3.2 % vs OFF.
  - All of this is against the modelled field.
- **OFF:** bit-identical (`tests/test_xfactor_v2.py` goldens from HEAD 102ed53's portfolio.py /
  xfactor.py loaded in place; builder split off / gs / gs + K = 2; the full run_table).
- **Decision:** keep OFF. Recommend ON for the tag after the owner OKs the meaning change (it
  tags pricier and star players, and often nobody on small slates). For the builder, confirm
  on 3+ more slates first.

## +ep (early-season prior: preseason games in the rate state, P3) -- SHIPPED 2026-10-24 (owner OK)

- **What:** `COURTSIDE_EARLY_PRIOR` (`proj/early_prior.py`). The EB rate adds the season's
  preseason games (no scrimmages, no games vs non-NBA teams, before the first regular-season
  day) with a weight w and a per-minute level v per stat, fit on seasons < S. Bundle suffix `_ep`
  (state["early_prior"]), version `<arm>+ep`; seeds unchanged. Write-up
  `docs/research/early-prior.md`.
- **Rejected at the state level:** the age delta on last season's rates (hurt 2025-26, z +2.07;
  centred +1.95), the rookie draft-slot prior and role-change shrinkage (|z| ≤ 1.2, no consistent
  sign).
- **Metric:** DK CRPS, paired `+ep` minus OFF on identical rows, E[min] >= 10 in either arm,
  DNP = 0. SE two-way date x player (`paired_cluster2`). First 30 days = dates < first
  regular-season day + 30.
- **Arm:** `pathA-v0.3+mf+p2+rx+rb`, 4,000 sims. Per-row outputs: Mac
  `~/Developer/courtside-compute/ep/out/rows_*`.

| set | first-30 n | first-30 diff (SE) | z 2-way | full n | full diff (SE) | z 2-way |
|---|---|---|---|---|---|---|
| 2023-24 | 3895 | -0.0166 (0.0049) | -3.39 | 21886 | -0.0028 (0.0012) | -2.36 |
| 2024-25 | 3941 | -0.0145 (0.0051) | -2.84 | 22045 | -0.0023 (0.0012) | -1.92 |
| RG 2019-20 | 3748 | -0.0119 (0.0064) | -1.87 | 18858 | -0.0013 (0.0015) | -0.84 |
| RG 2020-21 | 3665 | -0.0156 (0.0053) | -2.94 | 19266 | -0.0020 (0.0012) | -1.61 |
| live 2025-26 (no October) | -- | -- | -- | 15108 | -0.0015 (0.0007) | -2.11 |

- **Per-set sign:** negative on every set, in both windows.
- **Tiers:** no tier is worse by 2 SE. Bench leans worse on three sets (CRPS z up to +1.66
  first-30 and +1.50 full; 2024-25 bench full-season MAE z +2.03).
- **OFF:** bit-identical. HEAD state.py vs the working tree: walk_forward on 3 seasons and
  build_state on 3 days, exact. HEAD state.py + pipeline.py tree vs code: 4 id_driver cases and 5
  p2_identity hashes.
- **Decision:** recommend ship (owner decision). Copy the pre-fit
  `ep/data_ship/proj/bundle_2026-27_v1_cx_mf_ep.json` to prod `var/proj/` and set
  `early_prior.EARLY_PRIOR_DEFAULT = True`. Roll back with `COURTSIDE_EARLY_PRIOR=0`. The value is
  front-loaded (October-November).

## +rs (minutes restriction from ESPN comments, P7) -- keep OFF, 2026-10-23; committed OFF 2026-10-24

- **Metric:** minutes MAE (E[min | plays] vs actual, played rows) and DK CRPS, paired ON - OFF,
  two-way date x player SEs (`metrics.paired_cluster2`); negative = better.
- **Set:** the 2026-27 live slice, lock-45, 2026-10-20 .. 10-22, P(play) shipped, 4,000 sims. The
  arm is `pathA-v0.3+mf+p2+rx+rb`. 2025-26 has no ESPN comment archive, so ON = OFF there by
  construction. Rows: Mac `~/Developer/courtside-compute/rs/out/`.

| cell | n | min MAE diff (SE) | DK CRPS diff (SE) |
|---|---|---|---|
| flagged, played | 3 | -0.23 (1.77) | +0.69 (0.44) |
| unflagged, played | 328 | -0.004 (0.005) | +0.001 (0.002) |
| all rows | 528 | -- | +0.006 (0.007) |

- **Fit:** none. There are 0 flagged rows before 2026-27, so the rule is the parsed cap, else
  0.75 x normal minutes.
- **OFF:** bit-identical. The HEAD tree and the working tree give equal live-slice rows (528, all
  columns). The engine tests compare against a HEAD copy of `minutes.py`.
- **Decision:** keep OFF. n = 3, and the only "soft" flag (Allen, "keep a close eye on his minutes",
  played 28) was wrong. Revisit with >= 100 flagged rows and hard / soft phrases split.
  `docs/research/min-restrict.md`.

## +as (absence scale, G4) -- keep OFF, do not shrink, 2026-10-23; committed OFF 2026-10-24

- **What:** `COURTSIDE_ABSENCE_SCALE` (`model/absence_scale.py`, 3-line hook in
  `predict.price_day`). ON pulls the injury-report lineup strengths toward the full roster:
  `roster − s·(roster − lineup)`, for both models, the per-player injury effects and the G1
  capture-time lineups. Value `on` = `SCALE` (1.0), or a number. Both models' rows get `+as`.
  `docs/research/absence-scale.md`.
- **Question:** G1's open→close slope 0.45 (SE 0.05) said our absence values are ~2× the
  market's. They are not: against outcomes the absence component scales by **1.29 (SE 0.17)**,
  the close prices it at **0.92 (0.04)**, and the open prices a third of the later news already
  (0.36, SE 0.09). Placebos (within-date permutation, same team other date, pre-open news) are
  null. P(play) cells (B4) change nothing (best −0.00015 LL, 0.4 SE).
- **Metric:** walk-forward on 2025-26 (s fit on earlier days, opinion margin MSE, 150-game
  minimum; the archive has one season, so not "seasons < S"), paired ON − OFF, SEs date /
  two-way date × home team (`paired_cluster2`), n 899, 151 dates, run time = tip lineups.

| arm | model | ΔLL | SE date | SE 2-way | ΔCRPS margin (2-way SE) |
|---|---|---|---|---|---|
| s walk-forward (ends 1.5) | gbm_opinion | +0.00050 | 0.00192 | 0.00216 | −0.018 (0.022) |
| s walk-forward | gbm_market, fed close | −0.00020 | 0.00055 | 0.00059 | −0.008 (0.006) |
| s walk-forward | gbm_market, fed open | −0.00084 | 0.00054 | 0.00058 | −0.016 (0.006) |
| s = 0.5 (halve) | gbm_opinion | +0.00131 | 0.00242 | 0.00285 | +0.056 (0.027) |
| s = 0.5 | gbm_market, fed close | +0.00004 | 0.00068 | 0.00080 | +0.010 (0.007) |
| s = 0.5 | gbm_market, fed open | +0.00091 | 0.00067 | 0.00077 | +0.021 (0.007) |
| s = 0 (ignore the report) | gbm_opinion | +0.00963 | 0.00489 | 0.00548 | +0.209 (0.054) |

- **OFF:** byte-identical to the tree before the hook on two synthetic worlds and a real past day
  (2026-01-15, 9 games; JSON dump + `cmp`); `tests/test_absence_scale.py` (8 tests).
- **Decision:** keep OFF and do not shrink (`DEFAULT = None`, `SCALE = 1.0`). Larger s helps only
  `gbm_market` fed a stale open (CRPS 2.6 SE), which is G1's stale-line case; live lines are
  fresh. Re-test with 2026-27 intraday snapshots alongside +nsl.
- **Player side (measure only):** the DFS star-out bump realises 1.01× (0.04) for top-6
  teammates in the 2023-25 harness and 0.84× (0.07) on the 2025-26 live slice
  (`research/as_star_out_bump.py`): not 0.5×; at most a 15–20 % trim of top-6 minutes later.
