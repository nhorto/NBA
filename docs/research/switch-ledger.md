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
