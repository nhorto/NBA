# D2: pick the cash lineup by P(beat the cash line)

*2026-10-23. IMPROVEMENTS.md D2. Code: `src/courtside/proj/dfs/cash_pcash.py` (switch
`COURTSIDE_CASH_PCASH`, default OFF, version `cash-pcash-1`), `topn.rank_by_mean`, two hooks in
`runner._execute`. Tests: `tests/test_cash_pcash.py`. Evaluation: `research/cash_pcash_eval.py`
(Mac `~/Developer/courtside-compute/d2/`, outputs `out/`; copies in geekom1
`var/reports/cash_pcash_20261023/`). Live model `pathA-v0.3+mf+p2+rx`.*

## Plain-language summary

- **What it does (ON).** Instead of "mean minus 0.3 x downside", the cash lineup is whichever of
  ~10 candidates (the Top 10 plus solves at λ = 0.15 / 0.5) has the highest simulated chance to
  beat the double-up cash line against our modelled double-up field. The pick is made on one half
  of the sims and its chance is reported on the other half. Top N is re-ranked by mean, with
  lineups within 0.5 DK of each other ordered by that chance (PLAYERS.md §5.4).
- **Result: it does not beat the current rule.** Over 281 RotoGuru slates the P(cash) pick cashed
  3.1 points less often than the current rule (65.7 % vs 68.7 %, ±2.6, z −1.2) and scored 3.2
  DK points less (z −2.7). One season was slightly up (+2.7 ± 3.7 pp), the other clearly down
  (−9.1 ± 3.7 pp). The simulation promised +2.5 pp on the slates where the pick changed, and that
  promise held on the held-out sims, so it is not selection noise: the model's chance differences
  between near-identical lineups simply do not show up in real results (within a slate, the rank
  correlation between simulated and realised cash is 0.03).
- **Top N re-ranking is neutral** (new #1 vs old #1: −0.7 ± 1.8 pp).
- **Opening night looked great, but it is one slate.** The P(cash) pick on 10-20 was the same
  lineup in both lock runs (lock-10 #10 / lock-45 #4), 258.0 points: it would have cashed all
  three real double-ups; the rule's lineup (226.75 at lock-10) cashed none.
- **Cost:** ~0.3-0.8 s per build on an idle Mac, because the double-up field is built once and
  reused by the run's field scoring.
- **Recommendation: keep OFF.** Re-test when uploaded standings give real cash lines on ≥ 10
  slates.

## Design

1. Candidates: the OFF cash lineup (λ = 0.3, first), the Top N (mean objective, no-good cuts), and
   solves of μ − λ(μ − p25) for λ in {0, 0.15, 0.3, 0.5}; λ = 0 is the Top N #1, λ = 0.3 the
   OFF lineup, so two solves are new. Duplicates are dropped (≈ 10 candidates per slate: the λ
   lineups usually coincide with Top N entries). PLAYERS.md §5.3's "k-best, 100 each" was not
   built: it would cost ~400 solves, and with this result there is no reason to.
2. Field: the slate's double-up reference contest and its field exactly as `field.score_run`
   builds it (`default_contests`, `field_for(n=10_000, seed=0)`, `portfolio.truth_sims` with the
   same seed). `cash_pcash.reuse_fields` hands that field to the run's `score_run`, so it is built
   once (`field.py` is untouched; the hook swaps `field.field_for` only while the run's
   `score_run` runs, only for a matching call, and only with the switch ON). It passes
   `own_unc` like `score_run` when D5's `COURTSIDE_FIELD_K` is on.
3. Split: `portfolio.split_rows(S, 0, "gs")`: select on 55 % of the truth rows, report on the
   other 45 % (13,750 / 11,250 of 25,000).
4. Pick: max P(score ≥ cash line) on S; ties go to the OFF lineup.
5. Top N: `topn.rank_by_mean`. Walking down the mean order, a group is a leader plus every next
   lineup within 0.5 DK of it; each group is sorted by P(cash) on S. The set never changes.
6. Recorded in `dfs_runs.detail["cash_pcash"]`: version, contest, candidates (source, mean,
   P(cash) on S), pick, whether it changed, P(cash) on R for the pick and for the OFF lineup, the
   new Top N order, timings. `dfs_lineups.p_cash` is still `score_run`'s (all rows).
7. Never fails a run: errors or no sims (FPPG fallback) keep the OFF lineups with a status.
   Late swap and the swap candidates still re-solve the cash lineup under λ = 0.3.

## OFF is bit-identical

- `tests/test_cash_pcash.py` (11 tests): with the switch OFF the pick is never called, no
  `cash_pcash` detail is written and the saved rows equal the plain `_build` solves;
  `reuse_fields` is a no-op without a pending field; ON without sims saves the same rows as OFF.
- Stored slates: `runner._build` on 11 stored lock-45 / lock-10 runs (10-20 to 10-23, 7 slates,
  121 lineups, their stored projections and exclusions) from a `git archive HEAD` tree and from
  the working tree: the two JSON dumps are byte-identical (`cash_pcash_eval.py offcheck`).

## Runtime (Mac, idle, stored lock-10 runs)

| slate | run | pool | choose (field / solves / score) | score_run plain → reused | added |
|---|---|---|---|---|---|
| 155482 | c31d865cb8bb | 68 | 2.04 s (1.60 / 0.02 / 0.42) | 5.20 → 3.89 | +0.72 s |
| 155311 | 7583893037c9 | 102 | 1.87 (1.39 / 0.02 / 0.46) | 6.36 → 5.15 | +0.66 |
| 155316 | 9418d9ee3778 | 104 | 2.21 (1.63 / 0.02 / 0.56) | 6.64 → 5.19 | +0.75 |
| 154575 | 42da71bcf37a | 125 | 2.14 (1.50 / 0.02 / 0.62) | 6.25 → 4.88 | +0.78 |
| 155313 | 3f204b906fc3 | 71 | 1.44 | 5.28 → 4.32 | +0.48 |
| 155486 / 155487 | | 34 | 0.58-0.64 | | +0.31 / +0.40 |

The added cost is one more scoring pass of the field (the S and R evaluations) plus two CP-SAT
solves. Production builds run under contention (on 10-22 22:45 the cash field took 104 s and its
scoring 17.6 s in `c31d865cb8bb`); scaled the same way the added cost would be ~20 s, within the
30 s budget. Without the reuse it would have been ~+120 s on that run.

## RotoGuru walk-forward (ON vs current rule, paired by slate)

Harness: `cash_calibration.py`'s T2.13 setup. `project_historical(protocol='avail')`, bundles fit
on seasons < S, 10,000 sims. **Cash line:** a preset 500-entry double-up paying 44.4 %
(`field.preset_contest("double_up", 500)`), its cash-archetype field proxy from public
information (public projection = ½ ours + ½ season average, `field-proxy-1.2`), scored at the
slate's ACTUAL DK points with opponents redrawn 200 times; realised = the share of redraws in
which the lineup's actual score beats the line (median line 266.0; the simulated line's median
264.5). Pick on S, report on R of the calibrated truth. Every slate with ≥ 40 FPPG players.

| set | slates | pick changed | realised: rule | ON | Top-N #1 | ON − rule (SE, z) | points ON − rule (z) |
|---|---|---|---|---|---|---|---|
| 2019-20 | 143 | 73 % | 0.676 | 0.704 | 0.657 | +2.7 ± 3.7 pp (+0.74) | −0.15 ± 1.77 (−0.09) |
| 2020-21 | 138 | 75 % | 0.699 | 0.608 | 0.622 | −9.1 ± 3.7 pp (−2.45) | −6.44 ± 1.60 (−4.02) |
| pooled | 281 | 74 % | 0.687 | 0.657 | 0.640 | **−3.1 ± 2.6 pp (−1.17)** | −3.24 ± 1.21 (−2.69) |

- **Held-out sims vs reality.** On the 208 changed slates the pick's simulated P(cash) on R beat
  the rule's by +2.5 ± 0.2 pp (z +15); realised −4.2 ± 3.6 pp. The pick is right about the model
  and wrong about the games. Within a slate the candidates' simulated P(cash) and realised cash
  are uncorrelated (mean Spearman 0.03 over 218 slates; mean projection vs realised −0.01).
- **Variants (offline, same candidates):** a minimum gain of 0.5 / 1 / 2 pp: −5.1 / −5.2 / −2.5
  pp; only candidates within 0.5 / 1 / 2 DK of the rule's mean: −4.4 / −6.3 / −3.1 pp. None is
  better than the rule.
- **Top N re-ranking:** the order changed on 99.6 % of slates (Top 10 lineups are usually within
  0.5 DK of a neighbour) and the #1 on 38 %; new #1 − old #1 realised −0.7 ± 1.8 pp.
- **The rule itself:** the λ = 0.3 lineup beat Top-N #1 by +4.8 ± 1.6 pp (z +3.0) and +1.5 DK.
- **Calibration:** realised − simulated (R) for the pick: −1.4 ± 2.8 pp pooled.
- **Fixed 300 line** (`backtest.py`'s `cash_rate_proxy_ge_300`): rule 0.238, ON 0.210 (−2.8 ±
  2.1 pp).
- Picks: the rule's lineup 73 times, Top N #2-#10 the rest (#1 only 10 times).

## 10-20 opener (real standings; anecdotal, n = 1 slate)

The stored runs (pathA-v0.2, the projections and sims of the night), each real double-up's field
built as production builds it, the pick on S, scored with the night's actual DK points.

| run | contest (entries, real line) | rule lineup | cashed | ON pick | cashed |
|---|---|---|---|---|---|
| lock-10 42da71bcf37a | $2 DU 196970732 (287, 237.25) | 226.75 | no | Top N #10, 258.0 | yes |
| | $2 DU 196456023 (1,149, 247.75) | 226.75 | no | Top N #10, 258.0 | yes |
| | SE $5 DU 196456027 (459, 238.75) | 226.75 | no | Top N #10, 258.0 | yes |
| lock-45 c15398ddd87f | 237.25 / 247.75 / 238.75 | 238.25 | 1 of 3 | Top N #4, 258.0 | 3 of 3 |

Both runs pick the same lineup (the "highest P(cash)" lineup already noted in
`real-standings-2026-10-20.md` §5). Simulated P(cash) on R: rule 0.63 / ON 0.68 (lock-10), 0.61
/ 0.73 (lock-45). The re-ranked Top N #1 would have been lock-10 #2 (253.75). One night, one
draw: this says nothing against the 281-slate result.

## Decision

Keep `COURTSIDE_CASH_PCASH` OFF. The P(cash) pick is −3.1 ± 2.6 pp against the rule on the
walk-forward (−9.1 pp in 2020-21) and costs 3.2 DK points a slate; the opener is one favourable
draw. The likely reason is the field model, not the search: among ten lineups within a few DK
of each other the modelled field decides which one "beats the line", and those differences do not
transfer to real results (proxy ownership, preset contest). Re-test after ≥ 10 slates of uploaded
double-up standings (real cash lines, real duplication); the tooling (`cash_pcash_eval.py rg /
opener`) is in place.
