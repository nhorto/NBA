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


## +tm (tier match, B11) -- leave OFF, 2026-10-23, not committed

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

## cash-pcash-1 (D2: cash lineup by P(cash), Top N by mean then P(cash)) -- keep OFF, 2026-10-23, not committed

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
