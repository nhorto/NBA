# Contest builder: generate / select / report on disjoint sims (B1)

2026-10-23. IMPROVEMENTS.md B1. Code: `proj/dfs/portfolio.py` (`COURTSIDE_PORTFOLIO_SPLIT`,
`split_rows`, `_cycler`; modes `1` = three-way split, `gs` = report-only split, section 3b), tests `tests/test_portfolio_split.py` (+ golden
`tests/fixtures/portfolio_split_off_golden.json`), eval `research/portfolio_split_eval.py`.
Data: the 2026-10-23 nightly copy of prod (geekom1 `var/tmp/b1/`, Mac scratch
`~/Developer/courtside-compute/b1/`; outputs `out/*.json` there). Everything is simulated
against the modelled field except section 4's realised results.

## 1. Design

Switch `COURTSIDE_PORTFOLIO_SPLIT` (default OFF; `set_split`, or `build_portfolio(split=...)`).
ON, the scoring truth (`truth_sims`, computed exactly as before on all rows) is shuffled with
`default_rng(seed + 11)` and cut into three disjoint row sets:

| set | share | rows at 25,000 | used for |
|---|---|---|---|
| G | `SPLIT_G` 10 % | 2,500 | scenario, min-locked and refill objectives, drawn WITHOUT replacement (reshuffled when used up) |
| S | rest | 11,250 | candidate scoring for the greedy, the curve's picks, the baselines' picks |
| R | `SPLIT_R` 45 % | 11,250 | everything reported: per-lineup mean / p10-p90 / P(cash) / P(top 1 %) / E[payout] / single-entry numbers / dupes, portfolio summary, relative score, cash line, entries curve, baselines' numbers; soft / sharp strengths draw from R's stored rows |

Row counts. G only seeds objectives (<= ~600 draws at N = 150 with refills); with replacement
2,000-2,500 rows repeated ~40 objectives at N = 150 (6 % fewer unique candidates), hence the
no-replacement draw. R = 11,250 gives a per-lineup P(top 1 %) of 2 % an SE of 0.0013 (7 %
relative; 4.7 % on 25,000) and a portfolio P(any top 1 %) of 0.4 an SE of 0.005. S = R is B1's
50/50; a 70 / 20 S / R variant (section 3) did not pick better and halves R.

Candidates are scored on S and on R (G is not scored): no extra cost, the greedy runs on 45 % of
the rows, so ON is faster. `info.version` = `gpp-portfolio-1.0+split`, `info.split` = {G, S, R,
by_source_selected}; `Portfolio.sources` (not stored) = each pick's source. OFF runs none of it.

**OFF bit-identical:** the golden was written by the pre-change portfolio.py (HEAD 4d7df36) for
two synthetic builds covering min-locked jobs, build refills, curve extension + curve refills,
strengths and baselines; the test compares every lineup / summary / exposure / info field
(floats by repr, timings dropped). ON tests: G / S / R disjoint and complete; generation gets
exactly the G rows; every per-lineup mean / p90 / P(cash) / P(top 1 %) and P(any cash) equals an
independent re-score of the chosen lineups on R (exact), E[payout] within the dupe noise.

## 2. ON vs OFF on stored slates

Four GPPs: 154575 (10-20 opener, 3 games) $60K Zone and 155482 (10-22) $40K Zone (both $5,
$10K to 1st), 155078 (10-21) $100K Fadeaway ($15, $25K to 1st), 155487 showdown $120K Fadeaway
($15). Each build sees the first 20,000 stored rows (G / S / R = 2,000 / 9,000 / 9,000); both
arms' chosen sets are also scored on the last 5,000 rows with fresh truth noise and the same
dupe model ("holdout", which neither arm saw). opt = reported / holdout - 1. Paired = ON - OFF
on the holdout (same rows), mean over seeds, SE = max(within, between seeds). Mac mini,
threads 2; N = 150: 6 seeds, N = 20: 4 seeds.

| contest | N | E[payout] reported OFF -> ON | opt OFF / ON | ON - OFF holdout | P(any top 1 %) rep OFF / ON | scenario share OFF / ON | s OFF / ON |
|---|---|---|---|---|---|---|---|
| $60K Zone | 20 | 407 -> 358 (-12 %) | +11.8 / -4.3 % | +2.9 +/- 3.6 % | 0.398 / 0.395 | 0.53 / 0.50 | 9 / 9 |
| $40K Zone | 20 | 505 -> 464 (-8 %) | +11.8 / -4.8 % | +8.0 +/- 6.4 % | 0.349 / 0.332 | 0.58 / 0.60 | 7 / 7 |
| $100K Fadeaway | 20 | 3,704 -> 3,542 (-4 %) | -0.6 / -2.2 % | -2.8 +/- 1.4 % | 0.647 / 0.634 | 0.17 / 0.17 | 18 / 17 |
| Showdown | 20 | 2,431 -> 2,380 (-2 %) | -0.4 / +2.5 % | -4.8 +/- 2.7 % | 0.342 / 0.340 | 0.76 / 0.70 | 6 / 5 |
| $60K Zone | 150 | 2,091 -> 1,894 (-9 %) | +6.5 / -0.9 % | -2.6 +/- 2.0 % | 0.890 / 0.893 | 0.69 / 0.70 | 54 / 43 |
| $40K Zone | 150 | 2,467 -> 2,280 (-8 %) | +6.2 / 0.0 % | -1.8 +/- 1.4 % | 0.805 / 0.807 | 0.69 / 0.68 | 41 / 31 |
| $100K Fadeaway | 150 | 14,038 -> 13,671 (-3 %) | +0.8 / -0.3 % | -1.6 +/- 0.6 % | 0.956 / 0.953 | 0.48 / 0.44 | 93 / 68 |
| Showdown | 150 | 9,549 -> 9,276 (-3 %) | -0.9 / -2.6 % | -1.2 +/- 0.8 % | 0.730 / 0.733 | 0.81 / 0.80 | 21 / 12 |

P(any cash) is 0.95-1.00 in both arms and reported = holdout within 0.01 everywhere; mean
per-lineup P(cash) / P(top 1 %) drop by <= 0.01 / 0.002. Relative score drops with E[payout]
(Zones N = 20: 1.32 -> 1.15, 1.78 -> 1.63; N = 150: 1.08 -> 0.98, 1.46 -> 1.35).

- **The bias is in E[payout] (so ROI and relative score) on top-heavy GPPs**: OFF reports
  6-12 % above the holdout on the two $10K-to-1st Zones, ~0 on the Fadeaway and the showdown.
  ON's report is unbiased within noise (mean opt -1.6 % over the 8 rows; the fixed holdout block
  may itself read ~1 % high, which would make OFF's true optimism a little larger).
- **The chances were not inflated by this mechanism.** P(any cash) / P(top 1 %) reported = holdout
  in both arms. The "our chances read high" of the 10-20 standings is not B1's doing.
- **Scenario lineups do not lose share** (0.69 -> 0.70 at N = 150 on the Zones): their selection
  was not driven by the own-row bonus. B1's expectation that the share drops is not met.
- **Pick quality: ON costs ~1-3 % of honest E[payout] at N = 150** (all four negative, pooled
  about -1.8 % +/- 0.6 %); at N = 20 it is noise (+8 % to -5 %).
- Runtime: ON 20-45 % faster at N = 150 (greedy on 9,000 rows instead of 20,000), same at N = 20.

## 3. Variants (N = 150, 2-6 seeds, paired holdout ON - OFF in % of OFF)

| variant | $60K Zone | $40K Zone | Fadeaway | Showdown | ON opt (mean) |
|---|---|---|---|---|---|
| G with replacement, S / R 45 / 45 | -3.2 % | -5.4 % | -0.4 % | -0.4 % | -0.5 % |
| G with replacement, S / R 70 / 20 | -2.7 % | -2.5 % | -1.3 % | +0.2 % | -2.0 % |
| mode `1`: G without replacement, 45 / 45 (6 seeds) | -2.6 % | -1.8 % | -1.6 % | -1.2 % | -1.0 % |
| **mode `gs`**: G + S together 55 / R 45 (6 seeds) | -2.1 % | -2.2 % | -0.4 % | +0.8 % | -0.8 % |

A larger S alone (70 / 20) does not recover the loss and R = 20 % makes the report noisier.

### 3b. Mode `gs` (report-only split)

`COURTSIDE_PORTFOLIO_SPLIT=gs` (version `gpp-portfolio-1.0+split-gs`, `info.split.mode = "gs"`):
generation draws uniformly with replacement from G + S (as OFF does from all rows), the greedy
selects on G + S, and R reports. Ratio 55 / 45 with the SAME R rows as mode `1` (same
permutation): report noise is unchanged and the two modes compare on identical report rows; R =
20 % was already shown to be noisier without helping the picks. Mode `1` is pinned by a second
golden (`portfolio_split_on_golden.json`, written by the pre-`gs` code) and unchanged.

Same contests, rows and seeds as section 2 (N = 150: 6 seeds, N = 20: 4):

| contest | N | reported OFF -> gs | opt OFF / gs | gs - OFF holdout | s OFF / gs |
|---|---|---|---|---|---|
| $60K Zone | 20 | 407 -> 364 | +11.8 / -0.3 % | +0.3 +/- 4.1 % | 10 / 9 |
| $40K Zone | 20 | 505 -> 480 | +11.8 / +1.0 % | +5.1 +/- 3.3 % | 8 / 7 |
| $100K Fadeaway | 20 | 3,704 -> 3,600 | -0.6 / -0.4 % | -3.0 +/- 1.0 % | 18 / 18 |
| Showdown | 20 | 2,431 -> 2,317 | -0.4 / -1.8 % | -3.3 +/- 4.1 % | 6 / 6 |
| $60K Zone | 150 | 2,091 -> 1,903 | +6.5 / -1.0 % | -2.1 +/- 1.0 % | 50 / 41 |
| $40K Zone | 150 | 2,467 -> 2,285 | +6.2 / +0.6 % | -2.2 +/- 1.3 % | 34 / 26 |
| $100K Fadeaway | 150 | 14,038 -> 13,727 | +0.8 / -1.0 % | -0.4 +/- 0.4 % | 68 / 61 |
| Showdown | 150 | 9,549 -> 9,520 | -0.9 / -2.0 % | +0.8 +/- 0.7 % | 21 / 14 |

- Report still honest: gs opt -2.0 to +1.0 % (mean -0.6 %), like mode `1`.
- Pick cost at N = 150 pooled -1.0 % +/- 0.5 % (mode `1`: -1.8 % +/- 0.7 %): about half of it is
  gone, but the two top-heavy Zones still lose ~2 % (consistent with the greedy's late picks
  needing more than 11,000 rows there). N = 20: noise (mean -0.2 %).
- Runtime: 10-35 % faster than OFF at N = 150, same at N = 20.

## 4. Re-scoring the 10-20 opener (real standings)

Slate 154575, run c15398ddd87f (lock-45), the four uploaded contests (captured payout tables),
N = 20 (single entry N = 1), 3 seeds per arm, all 25,000 rows (ON: R = 11,250), final code.
Realised = the chosen lineups' ACTUAL DK points ranked in the real standings. pay pct = where the
realised total payout falls in the arm's simulated total; PIT = median over lineups of where
(actual - real cash line) falls in the simulated (lineup - cash line).

| contest | arm | mean P(cash) pred / real | P(top 1 %) pred / real | E[payout] pred / real ($) | pay pct | PIT |
|---|---|---|---|---|---|---|
| $2 DU 287 (N 20) | OFF | 0.52 / 0.67 | 0.031 / 0 | 41.1 / 52.0 | 0.77 | 0.56 |
| | ON | 0.52 / 0.68 | 0.032 / 0.017 | 41.2 / 54.7 | 0.84 | 0.63 |
| SE $5 DU 459 (N 1) | OFF | 0.75 / 1 | 0.029 / 0 | 6.9 / 10 | 0.68 | 0.71 |
| | ON | 0.75 / 1 | 0.029 / 0 | 6.9 / 10 | 0.68 | 0.71 |
| $2 DU 1,149 (N 20) | OFF | 0.52 / 0.47 | 0.037 / 0 | 41.4 / 37.3 | 0.38 | 0.49 |
| | ON | 0.52 / 0.52 | 0.038 / 0.017 | 41.6 / 41.3 | 0.49 | 0.51 |
| 4PP GPP 29,726 (N 20) | OFF | 0.45 / 0.38 | 0.037 / 0 | 299 / 63 | 0.38 | 0.39 |
| | ON | 0.46 / 0.45 | 0.038 / 0 | 277 / 75 | 0.44 | 0.47 |

The night sits inside both arms' ranges (PIT 0.39-0.71, pay percentile 0.38-0.84). Double-ups
are flat (nothing to over-fit), so ON = OFF there. On the 4PP GPP ON predicts 7 % less E[payout]
($277 vs $299 for $80 of entries); both realised $63-75 because no lineup reached the real top-1 %
line (290.5). One night cannot separate the arms; that needs many uploaded GPP standings.

## 5. Conclusions

- ON removes a 6-12 % over-statement of E[payout] / ROI / relative score on top-heavy GPPs
  (~0 elsewhere) and leaves the chances as they were; it is faster.
- Mode `1` costs ~1-3 % of (simulated, honest) E[payout] in the N = 150 picks; mode `gs` keeps
  the honest report and halves that cost (~1 %, ~2 % on the top-heavy Zones), and is simpler.
- Recommendation: ship `gs` (`SPLIT_DEFAULT = "gs"`). Honest headline E[payout] / ROI / relative
  score is worth ~1 % of simulated E[payout]; OFF keeps over-stating top-heavy GPPs by 6-12 %.
  If the Zone cost matters, more stored sims (e.g. 40,000 rows) would give gs more selection rows.
