# B7: props PHI and fringe P(play) restated out of sample

*2026-10-23. Items B7 (2) and B7 (3) of `docs/IMPROVEMENTS.md`. The minutes-eval sd (B7 (1)) is
another agent's. This is a restatement, not a ship: the live model, the packaged
`fringe_params.json` and the live `props.PHI` are unchanged.*

Code: `src/courtside/proj/props.py` (`phi_seasons`, `phi_for`, `fit_phi(before=)`),
`research/b7_fringe_oos.py`, `research/b7_p2_compare.py`, `research/b7_props_phi_effect.py`,
`research/copula_v1_eval.py` (now scores the props approximation with `phi_for`). Tests:
`tests/test_props_phi_oos.py`. Outputs: `var/tmp/b7/` (copies), Mac `~/Developer/courtside-compute/b7/`.

## 1. Props PHI fit on seasons < S

**Change.** `fit_phi(conn, seasons=None, window=30, *, before=None)`: with no `seasons` it fits the
3 seasons before `before` (default `LIVE_PHI_SEASON = "2026-27"` = 2023-24..2025-26, the old
default). `phi_for(conn, S)` returns the packaged `PHI` for S >= 2026-27 and a fit on the 3 seasons
before S otherwise (memoised per DB file). `PlayerDist.phi` / `player_dist(stats, phi=)` carry an
override; `None` means the live `PHI`. `fit_phi` raises if a given season is not < `before`.
CLI: `python -m courtside.proj.props fit-phi --before 2024-25`.

**Live value unchanged, proved three ways.**
- `PHI` is the same literal dict (`{pts 2.40, reb 1.11, ast 1.16, fg3m 1.09, pra 2.05}`, pinned by a test);
  `phi_for(None, "2026-27") is PHI`; `PlayerDist.pmf` computes `(self.phi or PHI)[stat]`, and a test
  asserts `np.array_equal` of every stat's pmf with `phi=None` vs `phi=PHI`.
- On the dev DB `fit_phi(conn, 2023-24..2025-26) == fit_phi(conn) == fit_phi(conn, before="2026-27")`
  (all `True`, `{pts 2.372, reb 1.107, ast 1.15, pra 2.019, fg3m 1.079}`).
- The packaged constants were rounded by hand from an earlier fit (today's refit on the same three
  seasons is up to 0.03 lower), and they are kept as is so live output stays byte-identical.
- `tests/test_proj_props.py` and `tests/test_market_props.py` pass unchanged (40 tests with the new file).

**Leak test.** `test_fit_never_reads_season_S_or_later`: inflating the noise of S and later seasons
does not move the fit for S; inflating an earlier season does.

**Out-of-sample PHI by evaluation season** (trailing 3 seasons before S):

| S | window | pts | reb | ast | 3PM | PRA |
|---|---|---|---|---|---|---|
| live const (2023-26) | | 2.40 | 1.11 | 1.16 | 1.09 | 2.05 |
| 2023-24 | 2020-21..2022-23 | 2.355 | 1.097 | 1.149 | 1.082 | 1.981 |
| 2024-25 | 2021-22..2023-24 | 2.339 | 1.096 | 1.145 | 1.078 | 1.989 |
| 2025-26 | 2022-23..2024-25 | 2.354 | 1.106 | 1.140 | 1.081 | 2.002 |

**Who scored props on past seasons.** Only `research/copula_v1_eval.py` (`PR.player_dist(...)`, the
"props approximation" columns of the T2.10 per-stat calibration in `docs/PLAYERS.md`, 2023-24 /
2024-25). It now passes `phi=PR.phi_for(conn, season)`. Nothing else: `proj/features/market_props.py`
(`PHI_MKT`) and `props.compare_day` run on live 2026-27 props only (no props exist before 2026-10-19),
`copula_v1r_eval.py` does not use `PHI`. The live-season path is untouched.

**How the stored calibration numbers change.** The eval's minutes draws are not stored, so the
full table cannot be recomputed without re-running two seasons of sims. `research/b7_props_phi_effect.py`
re-scores the stored rows (mean_approx, y, std line) with NB1 at the stored mean under live vs OOS PHI,
same PIT draw (no minutes mixture, so absolute levels are sharper than the stored ones; the delta
is the quantity). A smaller PHI is slightly sharper, so the approximation gets slightly worse:

| stat, season | PIT max dev live PHI -> OOS | Spiegelhalter z | log loss at the std line |
|---|---|---|---|
| pts 2023-24 | 0.0719 -> 0.0743 (+0.0024) | 14.9 -> 15.4 | 0.6555 -> 0.6558 |
| pts 2024-25 | 0.0702 -> 0.0732 (+0.0030) | 17.8 -> 18.5 | 0.6586 -> 0.6592 |
| reb 2023-24 / 2024-25 | +0.0012 / +0.0016 | 9.2 -> 9.5 / 10.3 -> 10.6 | +0.0002 / +0.0003 |
| ast 2023-24 / 2024-25 | +0.0009 / +0.0015 | 2.0 -> 2.1 / 2.3 -> 2.5 | 0 / +0.0001 |
| 3PM 2023-24 / 2024-25 | +0.0006 / +0.0005 | 0.17 -> 0.17 / 3.04 -> 3.04 | +0.0001 / 0 |

So the stored "approx" columns (pts PIT max dev 0.033 / 0.031, z 8.1 / 10.4) would move by about
+0.002..0.003 and +0.5..0.7 z, i.e. a little worse; every other stat is within 0.002. No conclusion
changes: the points left-tail failure and the |z| < 2.5 failures are already there at the live PHI.
No stored report is restated in place (they came from sims that are not re-run); this table is the
delta to apply.

## 2. Fringe P(play) "mid" model out of sample

**What was in sample.** The "mid" model (`FRINGE_FEATURES["mid"] = one, avg10, l_app_season0`) was
fit by `research/p2_fit.py` on all the T2.4 slice's fringe rows, then the slice scored the same
2025-12-22..2026-04-12 games with it. (The "early" / "early_nosal" models are fit on RotoGuru
2017-21 and are not used mid-season, so they are already out of sample for the slice.)
`fringe_params.json` meta had a leave-month-out log loss (0.5501, n 868) but the slice never used it.

**Method: leave-one-month-out (lomo).** Seasons < S is not possible: fringe rows need the report
status ("on no report"), and official reports exist only from 2025-12-22, so there are no rows
before the slice. The slice has five calendar months (Dec 22-31, Jan, Feb, Mar, Apr); each day is
scored with params fit on the other four months (same scheme as the report cells and the `+pq`
offsets). Forward-only (months before) is also reported. Rows were rebuilt with the real tip times
(`p2_rows.py slice --offset 45` on the Mac with the ESPN scoreboard archive in `raw/`), which gives
n = 886 mid fringe rows (the shipped fit had 868 from the artifact frames). Played share 0.245.
Params per held-out month: `var/tmp/b7/fringe_lomo.json`. The packaged file is only read.

**Fringe log loss (n 886, played 0.245; SE clustered by date):**

| fit | log loss | Brier |
|---|---|---|
| shipped params (fit on artifact-frame rows, scored here) | 0.5282 | 0.1728 |
| refit on all rows (in sample) | **0.5279** | 0.1728 |
| **leave-one-month-out (OOS)** | **0.5452** | 0.1790 |
| forward only (n 800, Jan-Apr; lomo on the same rows 0.5505) | 0.5542 | 0.1821 |
| constant: in-sample base rate / training-months base rate | 0.5567 / 0.5726 | 0.1849 / 0.1911 |
| minutes engine v1 (`pp_engine`, ~0.61 mean) | 0.9662 | 0.3383 |

- OOS minus in-sample: **+0.0172 (SE 0.0031)**. The in-sample number overstated the fit by 0.017 nats.
- OOS still beats the OOS constant by 0.027 (SE 0.008) and the v1 engine by 0.42. The model is real;
  it is 3 params on 886 rows and months disagree (the January intercept is -1.39 when January is
  held out, vs -1.93 when April is). By month (shipped / lomo): Jan 0.405 / 0.436 (n 299),
  Feb 0.570 / 0.578, Mar 0.594 / 0.599, Apr 0.829 / 0.871 (n 58).
- Mean absolute shift of P(play) for a fringe row, lomo vs shipped: 0.029 (max 0.073).

## 3. The p2-vs-v1 headline with OOS fringe params

Same T2.4 slice protocol (104 dates, 4,000 sims, 26,626 rows, real tip times, `pathA-v0.3`
vs `pathA-v0.3+p2`, paired DK CRPS, p2 minus v1). The OOS arm differs from p2 only by the fringe
params: each day uses the params fit without its month (`availability.load_fringe` patched in the
research driver). Rule is `pq/cmp.py`'s (rows in both arms with a CRPS, one-way SE by date) plus
the two-way date x player SE of `backtest._two_way_var`.

**Restatement of the stored frames** (v1 = `pq/out_v1`; p2 arm run with the same bundle the stored
p2 frames used, which reproduces them bit for bit):

| | n | diff | SE 1-way | SE 2-way | z 1-way | z 2-way |
|---|---|---|---|---|---|---|
| lock-45, as shipped (stored) | 26,626 | -0.0747 | 0.0109 | 0.0156 | **-6.88** | -4.79 |
| lock-45, **OOS fringe** | 26,626 | -0.0730 | 0.0111 | 0.0156 | **-6.58** | -4.68 |
| lock-10, as shipped (stored) | 26,626 | -0.0765 | 0.0103 | 0.0152 | **-7.40** | -5.03 |
| lock-10, **OOS fringe** | 26,626 | -0.0756 | 0.0106 | 0.0152 | **-7.16** | -4.96 |

- OOS minus in-sample fringe: +0.0017 (3.2 SE) at lock-45, +0.0008 (1.8 SE) at lock-10, i.e. 2% /
  1% of the gain. On the fixed population (E[min] >= 10 in either arm) it is +0.0005 (1.1 SE) /
  -0.0005 (-1.3 SE): so most of the difference sits in rows under 10 expected minutes.
- No tier worse (OOS vs v1, lock-45 / lock-10): bench -0.1008 (-7.8 SE) / -0.0982 (-8.0), role -0.0575
  (-3.8) / -0.0688 (-4.6), star -0.0159 (-0.9) / -0.0076 (-0.5). DK MAE gain is unchanged within 0.003.
- Fixed population (E[min] >= 10 either arm), OOS vs v1: -0.0968 (-5.46 SE 1-way, -3.84 2-way) /
  -0.1041 (-6.28, -4.29).
- **p2 stands**: -6.9 / -7.4 SE becomes **-6.6 / -7.2 SE** (two-way -4.7 / -5.0).

**Side finding: the stored comparison was not bundle-paired.** The stored p2 frames came from a
different minutes bundle (`p2/data/proj/bundle_2025-26_v1_cx.json`, fit 2026-10-21 08:41) than the
stored v1 frames (`pq/data_v1/proj`, fit 23:56). They differ in `minutes.base_coef` etc.; the
harness itself is deterministic (v1 reproduces bit for bit on rerun). With ONE shared bundle for
all arms (a fresh fit, identical to the v1 one), the numbers are: as shipped -6.70 / -7.19 SE,
**OOS -6.40 / -6.95 SE** (two-way -4.59 / -4.82; OOS minus in-sample +0.0016 (2.9 SE) / +0.0009 (1.9 SE)).
The bundle choice moves the headline by 0.2 SE; the OOS fringe moves it by 0.2-0.3 SE either way.

## 4. Reproduce

```
# local
python -m courtside.proj.props fit-phi --before 2024-25
python research/b7_props_phi_effect.py STATS_DIR/2023-24 '{"pts": 2.355, ...}'   # Mac, c1/out/ev_cx2_v1
# Mac scratch ~/Developer/courtside-compute/b7 (db = copy of p2/data, raw = pq/raw, code = pq/code + research)
./run.sh rows code/research/p2_rows.py slice --offset 45 --out rows            # real-tip fringe rows
python code/research/b7_fringe_oos.py fit rows/rows_slice_live45.parquet oos    # fringe_lomo.json + b7_fringe.json
B7_LOMO=oos/fringe_lomo.json ./run.sh DIR code/research/b7_fringe_oos.py slice live|live10 --sims 4000
python code/research/b7_p2_compare.py live V1_DIR P2_DIR P2_OOS_DIR
```
`run.sh` = single-threaded, `nice -n 10`, no work 17:45-19:30 ET (the driver sleeps through the
blackout windows). Two lanes at most.
