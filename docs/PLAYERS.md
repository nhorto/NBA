# Courtside players v2: DFS-first projections and optimizer

Status: **design v2, 2026-10-19.** Nothing is built yet, and this version goes to a reviewer before any build starts.

v2 replaces v1, which was projections-first with DFS as one consumer. In v2 the **DraftKings optimizer is the primary deliverable**, and the projections exist to serve it. Props, player cards and the "why" are kept but trimmed.

Inputs: `docs/research/dfs.md` (decisions, rules, sources, correlation study, review), `docs/research/player-predictions.md`, `research/dfs_correlations_output.txt`, `docs/MODEL.md`, `docs/MLOPS.md`, `docs/ANALYTICS.md`.

## 0. Changelog v1 → v2

### 0.1 Owner decisions (2026-10-19)

| Decision | Where it lands |
|---|---|
| The DFS optimizer is the top priority: pro-grade accuracy, honest backtests, simple use, a "why" for every lineup | Whole structure; §2, §5, §7.3 |
| DraftKings only in v1; FanDuel later via CSV | All FD content removed; timing is §9 Q7 |
| Two build modes: **Top N** (optional minimum number of different players) and **Contest builder** (an N-entry portfolio) | §2.2, §2.3, §5.4, §5.5 |
| Light guidance: simulated cash and top-finish odds plus a "sensible stakes" note. No dollar or Kelly sizing | §2.3 |
| Post-game why v0 ships in Phase 1; the full why engine moves to December | §2.8, T1.8, Phase 4 |
| Data policy: RotoGuru history, light DK capture, no crawl or login, never automate entry | §3, §9 |
| 150-max portfolios and automated late swap are lower priority | N defaults to 20 (max 150, slower). Late swap is a manual "re-optimize" action plus a CSV, never automated. |

### 0.2 Review items

| Item | Verdict | Where / why |
|---|---|---|
| **M1** iid sim overstates variance | **Accepted, generalized** | §4.5, T2.5/T2.6: step-0 measurement, game latent η_g, multinomial possessions, Q4 slope in the SMM. Feedback acts on *every* 6-min segment, because Q1–Q3 alone carry ≈ 200 > 154 of margin variance, so Q4-only can't close the gap. Gate G1 falls back to allocation-only. The **copula is the GPP default** until the sim beats it (G2). |
| **M2** sim tuned on test data | **Accepted** | §7.1 L2: everything is fit on seasons < S. Step 0 uses 2019-24, so sim claims are quoted only on 2024-25/2025-26. The "freeze on 2019-21" option is not used for the box path, because those are RotoGuru test seasons. |
| **M3** FanDuel has no data | Accepted (decided) | §1 |
| **M4** DK rule verification | **Accepted** | §5.1, T0.6 (all four acceptance points, plus a RotoGuru cross-check) |
| **M5** Phase 1 cites a Phase 2 harness | **Accepted** | T1.0 eval-lite; RotoGuru backtest T2.1 |
| **M6** minutes targets inverted; P(play) to Phase 1 | **Accepted** | §4.2 targets, T1.3 (uses the 151 rescued report days) |
| **M7** reconciliation unmeetable with OT/m₃ | **Accepted, option (b)** | §4.4: pilot → Newton step → full run. With m₃ feedback, even the regulation mean isn't analytic. |
| **M8** late news | **Accepted** | §2.5, §5.6, T1.6, T2.14: lock−45 and lock−10 runs, `late_news`, a late-swap re-solver |
| S1 prior double-counts last season | Accepted | §4.3: the prior uses seasons ≤ S−2 |
| S2 scenarios by mass | Accepted | §4.1: ≥ 95% mass, cap 16 |
| S3 randomized PIT | Accepted | §7.2 |
| S4 two-way clustering | Accepted | §7.2: game × player-season |
| S5 xFG acceptance | **Deferred** with xFG to Phase 4 (T4.1) | Not on the DFS path (resequence) |
| S6 per-coalition reconciliation in Shapley | **Deferred** with Shapley (T4.3) | Same |
| S7 team and missed-FT rebounds | Accepted | §4.5 Path B |
| S8 market % beside ours | Accepted | §2.8 cards |
| S9 templated one-liners; LLM writes ≤ 30 | Accepted | §2.8, T1.8 |
| S10 T0.1 is done | Accepted | T0.1 becomes verify + rolling capture |
| S11 payout ladder, 25–50k sims, 300 single-worker solves | Accepted | §5.5 |
| S12 40/60 field-proxy mixture, labelled | Accepted | §5.5; the weights are unverifiable until standings are uploaded (T3.2) |
| S13 DFS backtest metrics | Accepted | §7.3 |
| S14 value-play promotion (+8 min test) | Accepted | §4.2, T1.2 acceptance |
| S15 stint failure → box minutes | Accepted, made central | §4.2: the box path is primary (it alone covers RotoGuru seasons) |
| N1 books' team totals as environment | Accepted | §4.4 (`line_snapshots` team totals, from 2026-10-19) |
| N2 preseason role evidence | Accepted (light) | T1.9 (preseason boxes aren't in the DB yet) |
| N3 | Deferred → Phase 4 | A post-game "teammate left the game" term (from `possible_injuries`) joins the full why engine (T4.x); why v0 notes it in text only |
| N4 coach by date | Accepted | T2.2: a `coach_changes` override on top of per-season `team_coaches` |
| N5 PRA props | Accepted | §2.8, §7.5 |
| N6 Showdown captain binary | Accepted | §5.7 |
| Answer: parity bar → calibrated / beats naive / adds information | Accepted | §7.5 |
| Answer: DFS bar (FP MAE vs salary-implied and `fppg`; percentile vs field proxy) | Accepted | §7.3 |
| Answer: boom% is slate-relative | Accepted | §2.1 |
| Resequence (minutes ≫ usage > correlation > environment > defender/refs) | Accepted | §8; xFG, Shapley, full decomposition, defender, refs, matchups, season views → Phase 4 |

No review item is rejected outright. Three are accepted with a different option than suggested (M1 segment-wide feedback, M2 walk-forward over freeze, M7 pilot). Two should-fixes (S5, S6) are deferred because their components are deferred.

### 0.3 Facts found while revising

| Finding | Consequence |
|---|---|
| Migrations 7 and 8 exist (8 has `dfs_salaries`, `line_snapshots`, `prop_snapshots`) | New tables go in **migration 9**; `dfs_salaries` is altered |
| `sources/dfs.py` drops `playerGameAttributes` (id 100 = starter) and `isSwappable` | T0.7 adds them (needed at lock−10) |
| Official reports are rescued: 151 game days, 2025-12-22 → 2026-06-13 | P(play) calibration now (T1.3), plus a live-like slice (T2.4) |
| RotoGuru: one primary position and one salary per player per day, 2014-15 → 2020-21. PBP/stints start 2019-20 | **The DFS backtest runs the box-only engine**. The stint features are tested on projections for 2021-22+. |
| The correlation study's residuals are vs in-season means (hindsight) | Targets are recomputed causally vs the pregame projection (T2.7) |
| The Vegas total explains 4.9% of team DK variance | Environment ranks below minutes and usage |
| The scheduler uses fixed ET times | Lock-relative runs need a watcher (T1.6) |

### 0.4 Build decisions (2026-10-19)

| Decision | Evidence |
|---|---|
| **Next man up is probabilistic** (softmax over candidates by group, starts, EWMA and pair share), not the literal "top remaining player" (T1.2) | The literal rule is the real replacement starter only ~49% of the time. When it's wrong it projects +6.3 min against +0.6 realised. Lock−45 MAE is 0.06 better with softmax. |
| **The 240-minute reconciliation spreads the gap by each player's variance**, not proportional `allocate_minutes` | Proportional pushed the errors onto starters: MAE +0.13 |
| The S14 "two starters Out" test is measured on the **value play** (largest projected rise, P(play) ≥ 0.8), not the top healthy bench player | Value play: +9.2/+9.6 projected vs +8.9/+8.8 realised (passes). Top bench player: +5.0 vs +3.7/+4.0 (the ≥ 8 bar doesn't describe him). |
| Star absence: the minutes path gives +1.4 Δmin for the remaining top 6. The per-minute bump (T1.1: other star +5.6%, rotation +3.3%) supplies the rest of the DK effect | T1.2/T1.1 walk-forward 2023-24/2024-25 |
| **Margin noise varies by season, ~12.4–14.0.** The closing line's margin residual sd was 12.2–12.8 in 2016-20, 13.5–13.6 in 2020-22, **12.5 in 2022-23**, then 13.5–13.9 in 2023-26. Total sd is 17.6–18.3. Residual corr(margin, total) ≈ 0 (\|r\| ≤ 0.08), and corr(\|margin\|, total residual) ≈ −0.03 (T0.8, `research/game_env_residuals.py`). | Sim and expected-minutes margin sd is fit per season on seasons < S, not hard-coded |
| **Causal correlations (T2.7) flip v1's sign:** teammate DK residual vs our pregame projection is **−0.034** (v1 +0.034 was hindsight), opponents 0.000, no drift across seasons, team effects negligible. T1.4's tier-pair copula inputs already match v2 (r = 0.99). | Copula keeps T1.4's construction; v2 U table is the validation target |
| **Projection biases found by T2.7:** with a star known out, the remaining top 6 land **0.83 ± 0.13 DK below** projection (the bump is over-credited: minutes promotion + usage bump double-count); tier bias stars +0.8, role −0.7 DK | Fix in T2.3 (usage v1) / T2.2 (minutes v1), re-check with T2.1 |
| **T2.3 diagnosis: the −0.83 DK star-absence gap is composition, not usage.** Top-6 players on star-out nights run −0.85 DK in their own healthy games (the role-tier bias); per player the residual is +0.05 ± 0.06 (minutes −0.22, per-minute +0.27). Usage v1 (log-additive with/without factors, capped ±0.4) passes §4.3 (+4.9 / +7.1 SE over the tier bump) but moves DK MAE only −0.006. | Committed behind a switch, **not live**; ships with minutes v1 as one model bump after the T2.1 harness. Real fixes: role-tier bias and minutes over-promotion (T2.2) |
| **Minutes v1 (T2.2) held back from live** after one deploy (2026-10-19): walk-forward it wins (MAE 4.49 / 4.40 vs 4.63 / 4.58; DK gate 8.049 vs 8.181), but on the live opener slate projected team minutes summed to ~205–213 vs ~224–236 for v0.5, and points reconciliation then inflated stars by up to +7 DK. The eval used the oracle panel, not the live candidate path. | Live = v0.5 (`pathA-v0.1`) until v1 is fixed and passes a live-path team-minutes test |
| **Phase 2F, first ablation (F0–F2):** a per-player descriptor model **P** fixes the role-tier bias (end-to-end DK MAE −0.022 to −0.025, 6–8 SE, every season). **F2 opponent defense** (league-relative allowed-vs-expected index by position + 15 team box rates) clears the bar in 2023-24 / 2024-25 (per-minute NB LL +12 SE, CRPS 3–5 SE; harness CRPS 3.9 / 1.45 SE); it helps rebounds, blocks, assists, steals, turnovers and 3PM, not points (the market total covers points). **F1 opponent availability does not ship** (CRPS < 1 SE in 2024-25, negative in 2019-20). | P+F2 committed, wired into the live pipeline only after the opener with live-path checks |
| **P + F2 wired into the pipeline behind a switch (2026-10-20), default OFF.** `pipeline.CONTEXT_ON` (`CONTEXT_DEFAULT = False`, env `COURTSIDE_CONTEXT=1/0` overrides). ON = `pathA-v0.3`, bundle `bundle_<S>_v1_cx.json` (the OFF bundle + the layer fit on seasons < S, 1 LightGBM thread so refits are byte-identical); seeds stay on `pathA-v0.2` so ON/OFF share draws. The F2 pace term uses the environment's market total (live books total on the live path; there is no closing line before the tip). With the switch OFF, projections are bit-identical to HEAD (bdb8a56) on a historical day and on the live opener slate. | Walk-forward ON vs OFF, every season better at 5.5–9.5 SE (DK MAE −0.026…−0.038, CRPS −0.011…−0.018); star bias −1.1…−1.4 → −0.6…−0.9; coverage unchanged; gate lines all PASS in both arms; lineups +1.3 / +1.9 pts (< 1.2 SE). Opener: team minutes/points unchanged, team DK −3%…+1%, max player move 2.1 DK. Table under Phase 2F |
| **Phase 2F, second ablation (F3 / F5 / F6 on top of P+F2, 2026-10-20):** **F3 tracking** (rebound chances, passes, touches, deflections, contested shots, box-outs: shrunk per-minute rates, opportunity conversions, tracking-implied rate vs the EB rate) clears the per-minute bar by a wide margin in every season 2022-23…2025-26 (NB LL +7.7…+10.3 SE, CRPS +2.7…+4.2 SE). **F6 schedule** (home, rest, density, travel, time zones, altitude, season stage) adds on top of F3 (LL +1.0…+3.1 SE, CRPS +1.2…+2.8 SE). End to end (T2.1 harness) **F3 + F6** beats P+F2 by CRPS +0.0017 (1.3 SE) / +0.0041 (3.0 SE) and MAE +0.0042 (1.9 SE) / +0.0052 (2.3 SE) in 2023-24 / 2024-25; F3 alone is only 0.6 SE in 2023-24. **F5 matchups** (likely defenders from 2.2M matchup rows x shrunk defender quality) is real but tiny per minute (LL +1.4…+5.6 SE on top of F3+F6) and **adds nothing end to end** (CRPS +0.0000 / −0.0002): dropped. | **P+F2+F3+F6 kept, F5 dropped.** Wired opt-in as `pathA-v0.4` (`COURTSIDE_CONTEXT=1 COURTSIDE_CONTEXT_EXT=1`, bundle `bundle_<S>_v1_cx2.json`); default OFF; with the extension off ON stays bit-identical to `pathA-v0.3`. Star bias moves −0.06…−0.10 DK further negative (star-tier MAE within 1 SE). Table under Phase 2F |
| **Phase 2F, third ablation (F4 shot quality / luck on top of P+F2+F3+F6, 2026-10-20):** an xFG model from 1.92M play-by-play shots (log-loss −0.016 vs distance-only, all of it on 2s; 3s are near-unmodellable without defender distance) and per-player shot quality / diet / long-memory skill / **the share of the EB rate that is shooting luck**. The luck premise holds: 3PT over-expected keeps only 0.39 of itself in the EB window, and actual / expected fg3m runs 1.16 → 0.93 across luck deciles. Per minute F4 adds LL +2.5…+4.7 SE and CRPS +1.9…+4.0 SE on top of F3+F6 (mostly 3PM). **End to end it adds nothing**: CRPS −0.0004 (−0.5 SE) / −0.0003 (−0.4 SE), MAE −0.0008 / +0.0001 in 2023-24 / 2024-25. Market reconciliation cancels the points part; the 3PM part is too small in DK. | **F4 dropped**, not wired (`pipeline.py` untouched; the code stays as group F4 for research). Table under Phase 2F |
| P(play) uses hierarchical Beta cells (status × report age × reason × role) fit on the 151 rescued days (T1.3). Report age barely matters within a status; re-reading the latest report at lock−45/−10 is what matters. | CV log-loss on D/Q/P .463 vs .503 for the old map |
| **Late swap (T2.14, `proj/dfs/swap.py`)**: a slot is fixed when its game has tipped or DK's file says so (`(LOCKED)` cell, "In Progress"/"Final" in the file's player list); fixed players are pinned and every result re-checked, failures are left unchanged. Open players may move between open slots; a second solve keeps everyone else in his old seat. An entry changes only when a ruled-out player must go or the objective gains ≥ 0.5 pts. Objective: cash for our cash lineup and double-up / 50-50 / H2H contest names, else mean. Same-contest entries stay ≥ 1 player apart (identical entries solve once); optional overlap / exposure caps count the locked parts and are relaxed, with a note, only when locks force it. Unchanged and locked cells are written back exactly as DK wrote them; new cells use the ID from DK's own player list in the upload. Read-only: stored projections, not a fresh pipeline run. The runner's swap loop moved to `swap.resolve_targets` unchanged. | 150 real-slate entries in 0.9 s (budget 30 s); 20 tests, including 40 random lock/shock trials and a byte-for-byte equivalence test of the runner's loop |
| **Showdown (T2.15, `proj/dfs/showdown.py`)**: verified against pydfs Captain Mode. Slates are 30 RotoGuru single games (CPT = 1.5 × DK salary; actual points plus a noisy quarter-point variant) and the 3 captured DK Showdown slates (FPPG). pydfs hard-codes `max_from_one_team = 5`, which in a 2-team game is the same rule as our `teamCount.min = 2`. DK's CPT salary is exactly 1.5 × UTIL on all 250 captured rows: UTIL prices are in $200 steps, so no rounding rule is ever needed. | `courtside dfs showdown-check`: 33/33 slates, 63/63 comparisons with an equal objective, 0 ties |
| **GPP portfolio + contest builder (T2.12 / T2.16, `proj/dfs/portfolio.py`, `web/dfs_contest.py`)**: candidates per §5.5.1 (150/100/50 + Top-max(50, N) by mean, scaled by N/50 above 50); field + scoring reuse `field.py` (`evaluate(per_sim=True, own_entries=N)`); the greedy maximises the portfolio's marginal E[payout] with **our entries occupying ranks jointly**. Caps exact: γ = 5 (≥ #locks), max 60 % **from N ≥ 5**, a player's own minimum raises his default cap, locks in every lineup; refills under no-good cuts. Stored in `dfs_portfolios` (migration 13) + `dfs_lineups` mode `gpp` (run_id = portfolio id), **kept out of `dfs_runs`**. **Scoring truth is calibrated**: the sims assumed our projection was exact and beat the field by construction (live page showed "ROI +634 % simulated"). On the T2.1 harness rows (~46k player-games) (actual − ours) regresses on (FPPG − ours) with β = 0.16 ± 0.02 / 0.20 ± 0.01, and residual sd exceeds the sims' sd by ~0.18 sd, so each sim is scored on fp + β(FPPG − ours) + 0.18·sd·z (β = 0.18, only when the player plays). **Strength = information, not mix**: §5.5.3's mix strengths did not order (70 % optimizer field = 4.4k unique lineups from 200 resampled solves, a lower top tail: soft 748 / medium 563 / sharp 658 for a fixed set); now the field projection is 25 / 50 / 75 % ours at mix 0.4 → soft $845 > medium $594 > sharp $393. **Display:** no dollar E[payout] / ROI headlines until ≥ 10 slates of uploaded standings; headline = relative score vs the Top-N-by-projection set (+28 % ± 3 % on the opener $60K Zone) and P(cash) / P(top 1 %) bands; raw numbers in a "details (simulated, uncalibrated)" toggle and `/contest/<id>.json`; the slate page's chance to cash is banded too. | **T2.12 acceptance NOT met.** RotoGuru 2019-20/2020-21, N = 20, preset 5k GPP, realised = actual points vs the field proxy (`research/portfolio_backtest.py`). Original (season-average field, uncalibrated): **108 slates, greedy − top-N-by-individual-E[payout] = +2.9 ± 18.8 fees (0.15 SE)**. Calibrated, live-like field (public = ½ ours + ½ season average), same 108 slates: greedy − top-individual +13.6 ± 13.4 (1.0 SE), − capped top-individual +6.3 ± 4.2 (1.5 SE), − Top-N-by-mean +15.0 ± 13.2 (1.1 SE); the uncapped joint-marginal greedy − top-individual +10.9 ± 4.8 (2.25 SE). Greedy stays the default (it enforces the caps). Calibration, simulated / realised E[payout], before → after (same slates, field and sets): greedy 1.07 → 0.79 [0.63, 1.05], top-individual 1.55 → 1.08, Top-N-by-mean 1.46 → 0.99, all five sets pooled 0.93; τ barely matters (greedy 0.79 at τ = 0). Timing on geekom1 (2 threads, 25k sims, 14k-entry GPP): N = 20 in 22 s (37 s with strengths), N = 150 in 146 s. |
| **Slate-page P(cash) / P(top 1 %) use the calibrated truth** (`field.score_run(truth=True)`, the runner's lock45/lock10 hook; `score_run_safe` still never raises): same β-shift + model-error truth as the contest builder, live-like field (½ ours + ½ DK FPPG, cash archetype for double-ups). **P(cash) is off the slate-page headline** (owner decision 2026-10-20): it sits in a collapsed "Details (simulated vs a modelled field, uncalibrated)" toggle on the cash and Top N cards as the estimate ± 3 pp (field sampling); the headline returns once ≥ 10 slates of standings are uploaded, through the same gate as the contest page's dollar numbers (`portfolio.payout_calibrated`). Opener slate 154575, $5 Double Up (459): cash lineup 0.80 → 0.77 (seed spread ± 0.6 pp), GPP P(top 1 %) 0.066 → 0.055. | **T2.13 bar met against the proxy line** (`research/cash_calibration.py`, 108 RotoGuru slates, 500-entry double-up, realised = actual points vs the proxy field): cash lineup simulated − realised P(cash), medium field +4.4 ± 4.1 pp before → **−2.7 ± 4.1 after**; sharp field +6.5 ± 4.4 → **+1.0 ± 4.4**. Our cash lineup really beats the proxy cash line 73 % (medium) / 67 % (sharp) of the time, so the remaining gap to real double-up rates (~50–65 %) is the proxy field being weaker than real cash fields, not the sims: it needs real cash lines from uploaded standings (§9: bands until ≥ 10 slates). |
| **Path A copula v1 (T2.10) behind a switch, default OFF, not shipped as the DFS default (2026-10-20).** `copula.simulate_game_v1` draws the box components (pts, fg3m, reb, ast, stl, blk, tov) per player and sim: NB2 count marginals given the sim's minutes and environment, with the dispersion fit by level per stat on seasons < S (the v1 CV-by-level fit); a 7 x 7 within-player latent correlation per role tier, fit by simulated moments (pts–fg3m 0.57–0.70); across players the T2.7 causal PM tier x position matrices (`corr_<S>.json`) on each player's latent DK direction; a winner/loser per-minute shift by signed margin and tier (losers' tov up to +17 %, big winners' fg3m / ast +8…20 %; points only move between tiers, so the market total still holds); DK scored exactly from the counts (real DD/TD jumps). Same environment, reconciliation, availability and minutes draws as v0 (common random numbers). Per-stat sims stored as `<run>.stats.npz` (uint8, first 5,000 sims, 7.2 MB per 15-game run); `props.py` reads them, with the NB-given-minutes approximation as the fallback. Switch: `COURTSIDE_COPULA=v1` (or `pipeline.COPULA_DEFAULT`), versions compose as suffixes: `pathA-v0.2+c1` / `v0.3+c1` / `v0.4+c1`, bundle `<arm>_c1.json`. | On top of `pathA-v0.4` (T2.1 harness rows, 2023-24 / 2024-25, paired, date-clustered SE): DK CRPS (exp min ≥ 10) −0.0003 (−0.3 SE) / −0.0011 (−0.9 SE), all rows +1.0 / +0.1 SE, MAE +0.4 / +0.1 SE. 80 % coverage 0.820 → 0.825 / 0.824 → 0.829 (+0.4…0.5 pp, ~7 SE: slightly wider), PIT max deviation 0.007 → 0.008 / 0.011 → 0.012. Per-stat sims calibrated: points PIT max deviation 0.020 vs 0.031–0.033 for the props approximation; reb / ast / 3PM ≤ 0.006 both ways. Blowout curve: stars miss ±1.0 DK in both arms (minutes-driven; v1 fixes the losing side of the signed curve, overshoots big winners). Runtime 15 games x 25k sims on geekom1: 11.7 s warm / 19.0 s cold (v0 5.7 / 11.5). Every arm with the switch off is bit-identical to HEAD (c402c4a). **Ship bar not met** (2024-25 CRPS −0.9 SE, coverage and PIT a little worse); results block under the Phase 2 table |
| **T2.4 live-like slice (2026-10-21):** each game is run at its own tip − 45 on the live candidate path, with the report of the time, P(play) cells fit leaving the month out, and closing lines (no 2025-26 snapshots). The monitor's thresholds come from the slice: `mlops/live_like_baseline.json` (live-like minutes MAE 4.457, nightly p10–p90 ranges, new info rule `dfs:live_range:<metric>`). Found: listed players' DNP is counted twice (report cell × minutes-engine P(plays∣available)) | live − avail on 104 dates: minutes MAE +0.054 (0.016), DK MAE +0.216 (0.048), CRPS +0.147 (0.032); lock−10 recovers nearly all (+0.04 DK MAE). Listed Q/P/Available: P(play) 0.69 vs 0.79 played, DK bias −1.33 (0.39). Opener inside every band. Results block under Phase 2 |
| **P(play) v2 (p2, 2026-10-21) behind `COURTSIDE_PPLAY=v2`, default OFF.** Fixes the two T2.4 findings on the live candidate paths only: (1) a listed, not-Out player's P(plays ∣ available) is 1, so his total P(play) is the report cell (the cell is fit on *played*, so it already holds the coach's-decision DNPs); (2) a forced DK-pool player outside the team's last 10 games and not on the report gets an honest fringe P(play) (logistic: mid-season fit on the slice's fringe rows; the season's first 5 team games fit on the RotoGuru DK pools 2017-21 with the DK Classic salary, reweighted to the not-listed population, and players priced ≥ $4.5k keep the engine there; DK starter flag 0.99) instead of being scored as an active roster player. Version suffix `+p2` on any arm (`pathA-v0.3+p2`); bundle and seeds unchanged. With the switch off every arm is bit-identical, and the avail oracle is identical with it on | Slice, v0.3+p2 − v0.3, paired on a fixed population: lock−45 DK MAE −0.069 (0.020), CRPS −0.061 (0.017), unconditional minutes MAE −0.071 (0.019); lock−10 −0.080 (0.020) / −0.067 (0.017); bench tier −0.19…−0.22, stars ±0.01. Listed P(play) 0.62 → 0.71 vs 0.71 played (DK bias −1.46 → −0.31); fringe 0.66 → 0.25 vs 0.25 played (Brier 0.36 → 0.17). Opener: Gray 0.83 → 0.12 (DNP), priced rotation movers unchanged (≥ $4.5k keep the engine at season start), Top-10 actual +5.2 / +4.0 on the two Classic slates. Results block under Phase 2 |
| **Minutes ratio (mr, 2026-10-21) behind `COURTSIDE_MINRATIO=1`, default OFF.** The per-minute minutes table is re-keyed from realised m / mpg to the unplanned deviation m / P (P = the minutes engine's E[min ∣ plays] of each sim's availability scenario; in training the same engine walk-forward), so planned changes (injury ramps, promotions when a star sits) no longer get the per-minute penalty or boost of unplanned ones (early exits, blowouts). The ON bundle `<arm>_mr.json` refits the table, the usage bump, copula v0 and the context layer (offset E_s re-keyed) on seasons < S. Version `<arm>+mr`; seeds unchanged; with the switch off everything is bit-identical (2024-01-10, opener slate 154575, p2 identity hashes). A second factor g_p(P / mpg) was tested and dropped (weaker; star tier +1.3 SE) | Design B (the switch's), T2.1 avail, paired, date-clustered: DK CRPS −0.0037 (−2.8 SE) / −0.0031 (−2.0 SE) in 2023-24 / 2024-25, DK MAE −0.004; live / live10 + p2 CRPS −0.6 / +0.6 SE; RotoGuru 2019-20 / 2020-21 CRPS −3.3 / −4.9 SE. Per-minute error at > 1.4× recent: −4.3…−4.7 % → −2.8…−3.1 %; promoted bias −0.75 → −0.5; game-0 returners +1.6 → +1.1; star-out gap unchanged. Cost: the bench tier (E[min ∣ plays] < 15) is worse in all six sets (DK MAE +0.010…+0.035; CRPS +2.9 SE in 2019-20). **Bench follow-up:** not small-P instability or the context refit. The pooled f_u is too steep for bench plans (refit by P they are flatter at short minutes, lower at m / P 0.8–1.0), and the re-key unmasks the minutes engine's over-projection of planned-demoted bench players (P(play) +0.04…+0.06, E[min] +0.4…+0.8). Design C (f_u by P tier, `PL_TIERS_C`) is the better table at realised minutes and in aggregate: CRPS −3.7 / −2.4 SE, RotoGuru −4.7 / −5.9, live −0.4 / +0.7, bench DK MAE 10–30 % smaller. But bench is still +2.7 SE (2019-20) / +1.03 SE (2024-25), and its steeper mid-P curve widens rotation intervals: rotation +1.6 / +2.3 SE on the live slice. Cash lineups −2.2 / −2.5 (± 1.5 / 1.3). **Bar not met by either design; not shipped.** The switch keeps B, and C is a research option. Next: the minutes side for short-plan players. Results block under Phase 2 |
| **Preseason minutes + season-start debut reserve (pre, 2026-10-21) behind `COURTSIDE_PRESEASON=1`, default OFF.** The minutes LightGBM gets 9 preseason columns from the migration 15 logs (the row team's preseason only, strictly before the date: games, mpg, last-2-games mpg, share of team minutes, rank, vs prior-season mpg and vs teammates, did-not-play flag; NaN from team game 20 on). Scrimmages are dropped; games against international clubs are kept. Walk-forward finding: the minutes of players outside the candidate set are ~29 per team at game 0 (13 debutants + 16 non-regular arrivals in the oracle) against a season-average fitted reserve of ~3, and ~1–3 later. After game 0, prior-season minutes are not mis-weighted (roster rows' bias is +0.04…+0.10). So the reserve is split: debutants by team games into the season × the camp's new players' preseason minutes (winsorised at the training p95), and the rest. Version `<arm>+pre`, bundle `<arm>_pre.json`; OFF is bit-identical | T2.1 avail ON − OFF, first 15 team games: minutes MAE −0.027 (−1.4 SE) / −0.032 (−2.8 SE) / −0.020 (−0.9 SE) and DK CRPS −1.4 / −3.5 / −0.9 SE in 2023-24 / 2024-25 / 2025-26. Full season ±0. Star tier full-season CRPS +1.4 SE (2023-24) and +2.6 SE (2025-26). Opener 2026-27: the debut reserve takes 18–30 team minutes from BOS / OKC / PHI / SAS, but debutants played 0–13. **Bar not met; not shipped.** Results block under Phase 2 |
| **Minutes fix (mf, 2026-10-22) behind `COURTSIDE_MINFIX=1`, default OFF.** Diagnosis of the +mr / +pre minutes-side failures: the planned-demotion bias is the v1 P(plays ∣ available) logit, which never sees the GBM's plan (demoted bench +0.05…+0.06, promoted −0.065; three quarters of demoted bench rows are the player's own return / DNP streak, 7–9 % a starter back today); E[min ∣ plays] is close; the 240 rebalance only carries P(play) changes into the starters. The sims' noise was fit on log(actual / plan), which already holds the margin effect, and its tier lognormal is not mean-preserving, so the sims moved ~0.6 min a game from stars to the bench and made stars / starters too wide (bench draws given play were too narrow). Fix, refit on seasons < S: P(plays) logit + the plan (log plan, plan / EWMAs / median 5, depth rank by plan), fit on out-of-fold plans; noise fit net of the margin curve, then location / scale per tier calibrated by simulation on the training projections. A LightGBM P(plays) was dropped (it learns the oracle rule that available regulars always play). Version `<arm>+mf`, bundle `<arm>_mf.json` (after `_pre`); composes with +mr / +pre. Debut reserve (under +pre): driver = the camp's two largest new players' preseason minutes, capped at the training max (the camp sum ran out of range: OKC 126) | ON − OFF, date-clustered: DK CRPS −6.1 / −5.2 SE (2023-24 / 2024-25 avail), live / live10 + p2 −2.6 / −3.7, RotoGuru −3.7 / −3.0; every tier better in every set (worst −0.7 SE); DK MAE −0.013…−0.029, minutes MAE −0.022…−0.043; star DK bias −0.40 / −0.25 → −0.19 / −0.09. **+mf+mr (B)** −7.2 / −6.4, live −2.9 / −3.0, RotoGuru −4.5 / −5.2, bench no longer worse (C passes too, weaker). **+mf+pre** −4.4 / −4.5, live −1.7 / −2.7, first 15 games CRPS −2.1 / −3.2 SE. Demoted-bench rows with E[min] ≥ 10 keep most of their P(play) bias (+0.067 / +0.080 → +0.063 / +0.075). 28 openers of 2026-27: the debut reserve is unbiased (the 6-team opener was the sample) but OKC is still over-reserved. OFF bit-identical (2024-01-10, slate 154575, 5 / 5 p2 hashes). **Bar met by +mf, +mf+mr, +mf+pre; not shipped (owner's call).** Results block under Phase 2 |

## 1. Scope and design in one paragraph

**Product scope:**
- the DK **Classic** optimizer (cash, Top N, contest portfolio), with **Showdown** in v1.x;
- DK-point and component projections (pts, reb, ast, 3PM, stl, blk, tov, DD/TD) and minutes;
- props (pts/reb/ast/PRA/3PM) as information only.

**Design:**
- **P(play)** is calibrated on official reports.
- A **minutes engine** redistributes vacated minutes, promotes value plays, and is conditional on the game margin, so blowouts come from the margin draw.
- **Per-minute EB rates** get a usage bump under absences.
- A **joint distribution** per slate: a Gaussian copula with a shared margin draw (default), upgraded to a possession sim only if it wins.
- **CP-SAT** solves cash (P(score ≥ cash line)), Top N (k-best) and contest portfolios (payout ladder vs a labelled field proxy).
- Runs happen at lock−45 and lock−10, plus late-swap re-solves. Every lineup carries templated reasons.

## 2. Product surface (phone-first, DFS page first)

### 2.1 `/dfs/<day>`: slate and player pool
- **Slate picker:** main Classic by default; Showdown in v1.x. A run badge reads "Built 18:20 (lock−40) · 2 late-news players", plus stale-data warnings.
- **Player pool**, sortable:
  - salary and DK positions;
  - P(play) and status, or a starter tick;
  - minutes and mean DK;
  - range (p25–p75 bar; p90 is the "ceiling");
  - value (mean per $1k);
  - **boom%** = P(FP ≥ salary/1000 × v*), where v* is the points per $1k of the slate's simulated top-1% lineups. This makes it **slate-relative**;
  - **popularity (proxy)** as Low/Med/High;
  - a `late_news` dot.
- **Phones** get a card list. **Controls:** lock/ban, min/max exposure, exclude Q/GTD.

### 2.2 Top N lineups
- **Ranked list**, default N = 10, up to 50. Each lineup shows the 8 slotted players, salary used, mean, p10–p90 range, and P(cash) once T2.13 ships.
- **Diff vs #1:** players in and out, Δmean, Δsalary.
- **Setting:** "min different players" d ∈ {1…4}, default 1 (§5.4).

### 2.3 Contest builder
1. **Pick a contest** from the archived lobby (fee, field, max entries, prizes, payout table), or **describe one**: fee, field size, max entries, and a payout curve from presets or pasted rows.
2. **Choose N entries**, up to the per-user maximum (cap 150).
3. **Output:**
   - the portfolio (§5.5) with an exposure table;
   - **simulated cash rate**, **top-1% rate** and **expected payout** for the set;
   - **expected ROI** labelled "*simulated against a modelled field, not real results*", shown as a soft–sharp field range;
   - **dupe risk** (Low/Med/High) per lineup.
4. **Light guidance** (static text):
   - cash contests suit projection-driven builds;
   - single-entry and 3-max GPPs avoid 150-entry pros;
   - large fields carry the most variance;
   - a **"sensible stakes" note**: a strong player still loses over about half of 100-contest stretches, so stake only entertainment money.

   **No dollar or Kelly sizing.**

### 2.4 "Why this lineup" (templated, no LLM)
Up to 3 reasons per lineup, picked from facts by rule. Each number comes straight from the run:
- **Minutes opened:** "+7 min for Hart with Brunson out". Fires when Δ projected minutes ≥ 4 from an absence.
- **Value:** "$4,100 for 24 pts (5.9 per $1k, slate top-5)".
- **Ceiling:** "28% to reach the slate's boom mark".
- **Environment:** "3 players from the slate's highest team total (CHI 121.5)".
- **Lineup level:** "uses $49,800; lowest blowout risk of the top 10".

Reasons are ranked by |contribution to the lineup's objective − slate median|.

### 2.5 Late news and late swap
- **Lock-relative runs:** the build runs automatically at **lock−45** and **lock−10** for each slate (§5.6).
- **`late_news` flag.** A player is flagged when, since the previous run or within 90 minutes of lock:
  - his status changed;
  - he was named a starter or a non-starter against expectation;
  - his projected minutes moved ≥ 3.
- **Banner:** "2 lineups contain late-news players".
- **"Re-optimize unlocked slots"** re-solves with started games' players fixed and downloads a DK edit CSV. The user uploads it by hand.

### 2.6 Showdown (v1.x)
- The same page with a CPT column (1.5× salary and points).
- Top N plus a contest builder for single-game contests (§5.7).

### 2.7 Exports
- **DK CSV** in the upload template (slot columns, `Name (draftableId)`), for new entries and for edits.
- **Never** automated entry, login or edits.

### 2.8 Kept, trimmed
- **Tonight player cards** (Phase 2). Top 4 per team:
  - `27.4 pts · 4.1 reb · 7.2 ast · 35 min` with an 80% bar;
  - one factor chip;
  - status and P(play);
  - the market line as **"Line 26.5 · our chance above 58% · market 52%"**.

  Both percentages are shown (S8). "Over", "edge" and "value" stay banned (STYLE.md).
- **Post-game why v0** (Phase 1), for |actual − projection| ≥ 6 DK or ≥ 5 pts:
  - a log-additive split into minutes, volume (FGA + FTA per minute) and making (pts per shot vs his EB norm), so it sums exactly;
  - a **templated one-liner**: "Hart 31 (our 17): +9 min, 7-of-10 from three";
  - the LLM writes at most **30** short stories a night (S9), under v1 §3.4's fact-check and sign rules.

## 3. Data

### 3.1 RotoGuru history (T0.2)
- **Fetch:** `hyday.pl?game=dk&mon=M&day=D&year=Y` in the HTML view (scsv truncates after 2017), 2014-10-29 → 2021-05. About 1,200 requests at 1/s, archived to `var/raw/rotoguru/dk/<date>.html.gz` and never re-fetched.
- **Rows:** id, primary position, name, starter `^`, DK pts, salary, team, opp, venue, minutes, stats. They are joined to `player_games` on (date, team, normalised name).
- **Slate:** the games where ≥ 80% of rotation players have a salary. A secondary main-slate proxy keeps games tipping ≥ 19:00 ET.
- **Eligibility:** derived from the single position (PG → PG/G/UTIL). It is applied identically to our lineups, the baselines and hindsight-optimal.

### 3.2 Light daily DK capture (T0.7)
- **Already running:** lobby, Classic/Showdown draftables, and rules at each pregame time.
- **Added: payout tables** for **30–40 main-slate contests a day**: all double-ups and 50/50s, the top 10 GPPs by prizes, single-entry and 3-max GPPs, and up to 3 Showdowns.
  - Fetched once before lock from `contests/v1/contests/{id}`, because `entries` reads 0 after completion.
  - Stored fields: fee, `maximumEntries`, `maximumEntriesPerUser`, `isGuaranteed`, `payoutSummary`, attributes.
- **Lock-relative draftables** at lock−50 and lock−12: status, disabled, swappable, news, starter.
- **Not collected:** historical draft groups, leaderboards, ownership, anything behind a login.

### 3.3 Other inputs
- **Official reports:** the 151 rescued days (verified by T0.1), the rolling capture, and a fetch right before each lock run.
- **ESPN injuries.**
- **Book lines and team totals** from `line_snapshots` (N1). **Props** from `prop_snapshots`.
- **Box:** `player_games` back to 2000 (the box path).
- **Stints** for 2019-20+, via T0.3/T0.4.

### 3.4 ID matching (T0.5)
- **Keys:** DK `playerId` (stable across slates; `draftableId` is per slate) and the RotoGuru id → NBA `player_id`, persisted in `player_ids(source, source_id)`.
- **Method:** name + team + date against `team_rosters` and the recent game rosters, reusing `injuries.match_player` (accents, suffixes, nicknames, FUZZY_MIN 0.86 with a 0.05 margin).
- **Same-team near-duplicates** (Jalen/Jaylin Williams) need an exact first-token match.
- **Unmatched players** raise a data check and go to `player_aliases`.
- **Targets:**
  - DK ≥ 98% of salaried players automatically, and 100% of players with projected minutes ≥ 10 after aliases;
  - RotoGuru ≥ 99% of salaried player-games.

## 4. Modeling

### 4.1 Availability: P(play) (T1.3, Phase 1)
- **Inputs:** the latest official status before the run; ESPN status; DK `status` and starter flag; hours to tip; and a reason class (rest / injury / illness / personal / G-League), parsed from the report text.
- **Model:** a Beta posterior per (status × hours-to-tip bucket {> 6, 2–6, 0.75–2, < 0.75} × reason class).
  - Prior: the current `injuries.PLAY_PROB` value with 20 pseudo-observations.
  - Updated with snapshot status → played or not, using the 151 rescued days plus live data, refit weekly.
  - A cell with n < 30 falls back to its status-level posterior.
- **Overrides:**
  - DK starter flag = 1 → P = 0.99.
  - Unlisted regulars: P = 0.99.
- **Scenarios (S2):** per team, enumerate joint availability outcomes of the uncertain players in descending probability until ≥ 95% of the mass is covered, capped at 16. The remaining mass is folded into the nearest enumerated scenario.
- **Acceptance (T1.3):**
  - reliability within ±0.08 per status with n ≥ 50;
  - Brier ≤ the static table, reported with game-clustered SE.

### 4.2 Minutes engine (the centerpiece)
**Minutes are modelled conditional on the margin.**
- **minutes = f(pregame features) + g_tier(|final margin|, won/lost)**, where g is a blowout curve fit on box data for seasons < S. Shape to reproduce: starters −3.5 min at 26+.
- At prediction time, g is evaluated at each sim's *drawn* margin.
- This box path runs back to 2000, covers every RotoGuru season, and is the S15 fallback.
- From 2019-20, Q1–Q3 stint shares plus Q4-by-|m₃| shares can replace g on the sim path.

**Base minutes model** (Phase 1 v0: EWMA plus rules; Phase 2 v1: LightGBM refit per season walk-forward, pre-*d* features only):

| Group | Features |
|---|---|
| role | fast/slow minutes EWMA (α 0.2/0.02), last game, median of last 5, started last game, starts in last 10, team minutes rank |
| return | games since returning from ≥ 3 missed games, games missed |
| absences (per scenario) | vacated minutes: total, and in his position group G/F/C; his shrunk historical share of each absent teammate's minutes; **depth rank within his position group after removing absent players** |
| coach | rotation concentration (top-7 share, count of 30+ minute players), coach by date (N4), 40 pseudo-games of shrinkage |
| schedule | b2b, rest, 3-in-4, age × b2b, his rate of sitting second legs |
| news | DK starter flag (lock−10 only), status, reason class |
| early season | preseason starts and minutes in the last 2 preseason games, for the first 10 games (N2, T1.9) |

**Value-play promotion (S14).** Proportional renormalisation spreads vacated minutes too thinly and misses the biggest DFS edge. Instead:
1. **Next man up.** For each absent rotation player j, the top remaining player in j's position group, ranked by (starts in the last 10, minutes EWMA, historical share of j's minutes), is lifted toward the coach's median starter minutes for that group, shrunk 50% to his own EWMA. The remaining vacated minutes go to the group, then to adjacent groups (G↔F↔C), by historical share when j sat.
2. **Team total** of 240 minutes, via `players.allocate_minutes` water-filling with a 40-minute cap.
3. **Starter flag.** At lock−10, an unexpected confirmed starter is re-run through step 1.

**Tests (T1.2):**
- **Two starters Out:** the top bench player's projected minutes rise **≥ 8** vs healthy (mean over ≥ 200 team-games), and the realised rise is within ±2.5 of the projected one.
- **Star absence:** the remaining top-6 projected ΔDK is **+3.1 ± 0.6**.

**Noise.** The share is multiplied by exp(ξ), with ξ ~ N(0, σ²_tier), plus an early-exit mixture with probability q₀ + q₁·PF-rate, scaled by U(0.35, 0.85). σ_tier, q₀ and q₁ are fit on seasons < S (M2).

**Targets (M6):**
- minutes MAE for projected ≥ 15 is **≤ 4.5 in the avail backtest** (v1, Phase 2), with v0 ≤ 4.8;
- ≥ 8% better than the rung-2 EWMA on the same rows;
- the **live-like slice** (2025-12-22 → 2026-04-12, real official reports instead of the oracle) is reported separately and is expected to be worse;
- **live MAE within +0.5 of the live-like slice.**

### 4.3 Rates with usage redistribution
- **Quantities:** per-minute DK components (pts via 2PA/3PA/FTA and makes, reb, ast, stl, blk, tov). The sim path uses per-possession rates.
- **Estimate (EB):** `r̂ = (Σ_g w_g x_g + κ·μ_prior)/(Σ_g w_g n_g + κ)`.
  - Weights have a 120-day half-life in season, and last season is multiplied by 0.6.
  - **S1:** μ_prior is a ridge on position group, height, weight, age, draft slot, and the rate over **seasons ≤ S−2** (never last season, which is already in the likelihood).
  - κ is set per stat by beta-binomial method of moments on seasons < S.
  - The age curve is the delta method, as in v1.
- **Usage under absences:**
  - **Phase 1:** a role-tier per-minute bump fit on star-absence natural experiments in seasons < S. The study's values to reproduce: the other star +7.6%/min, non-star rotation +4.2%/min.
  - **Phase 2 (T2.3):** individual with/without factors φ_ij from stints (2019+, `φ = φ_emp·n_off/(n_off+K_φ)`, K_φ chosen walk-forward) and box-level with/without for earlier seasons. Acceptance: on "teammate sat" games, the usage log-likelihood beats the tier bump by > 2 SE.
- **Opponent and pace (Phase 2, T2.8, low priority):** opponent-by-position indices shrunk as in v1 (3P% allowed K ≈ 5,000). They ship only if DK CRPS improves by > 1 SE.

### 4.4 Environment and reconciliation
- **Environment (N1):** team implied points come from the book team totals in `line_snapshots`, using the median across books at the run time. Without them, they are (μ_T ± μ_M)/2 from `gbm_market`.
- **Scaling (copula and Phase 1 path):** each team's projected points are scaled to the implied total, with α split between pace (all counting stats) and efficiency (points only). The prior guess is α = 0.5, fit on seasons < S.
- **Sim path (M7):**
  - Run a **2,000-sim pilot**.
  - Measure the gap between Σ E_sim[pts_i] (including OT and the m₃ feedback) and the target.
  - Take one Newton step on the efficiency shift δ_t, using the pilot's numerical derivative. Then run the full set.
- **Acceptance:**
  - |Σ_i E[pts_i] − target| ≤ 0.5 for every team-game.
  - A pre-reconciliation gap > 6 raises a data check (stale report or mis-matched name).
  - |δ_t| is capped at 0.15 logit, with any remainder reported.

### 4.5 Joint distribution

**The facts that set the bar** (study, 2015-16 → 2025-26):
- teammate DK residual r = 0.034 (high-usage pairs 0.000, C–C −0.09, star–bench −0.03, bench–bench +0.09);
- opponents 0.011;
- drift by season, 0.017 → 0.059;
- blowouts mostly cost minutes (stars −5.5 DK at |m| ≥ 26; losing starters −4.7 at |m| ≥ 16);
- normal marginals fit stars and role players, gamma fits bench, and CV falls 0.79 → 0.24 with level.

Correlation is a weak lever, so **the marginals and the shared margin draw matter most.**

**Path A: copula (T1.4 v0 → T2.10 v1; the GPP default).** Per sim and game:
1. Draw (H, A) errors from a bivariate normal with the game model's OOF sd(margin), sd(total) and correlation (T0.8, seasons < S). This gives the margin m_s.
2. Draw the availability scenario.
3. Draw minutes: base + g_tier(|m_s|, sign) + noise, then water-fill.
4. Draw outcomes given minutes (means = minutes × rates, with the usage bump and a winner/loser per-minute shift):
   - **v0** draws DK FP directly: normal for projected ≥ 25, gamma below, with CV-by-level fit on seasons < S;
   - **v1** draws the **components** (pts, reb, ast, stl, blk, tov, 3PM) with a within-player 7×7 Gaussian copula per role tier, then scores them, so DD/TD are exact and props share the draws.
5. Couple players with a Gaussian copula: the role-pair × position-pair matrix plus opponent entries, from T2.7 (last 1–3 seasons < S, **residuals vs our pregame projection**).

Path A takes team variance top-down from the market-calibrated game model, so M1's over-dispersion can't occur. It costs 15–30 s per 15-game slate. What it gives up: team totals match only in expectation.

**Path B: possession sim (T2.5 → T2.6), measurement first.**

**Step 0** measures on 2019-20 → 2023-24 PBP:
- Var(team | line) and Cov(H, A | line);
- the per-possession points variance;
- the **lead restoring force**, E[Δm over the next 6 minutes | m];
- the slope b₄ of final margin on m₃ (expect < 1);
- E[Q4 pace | |m₃|];
- leader/trailer Q4 efficiency;
- team-rebound and missed-FT-rebound rates (S7).

The arithmetic behind it:
- Measured (T2.5, 2019-24 vs the closing line): Var(team | line) 125.6 and Cov(H, A | line) 38.7. Given possessions and the line, total-score variance equals iid possession noise, so there is no room for a shared game latent; the restoring force (ρ ≈ −0.07 per 6-minute segment, −0.11 in the last) is what keeps sd(margin) near 13 instead of the iid 16.
- iid possession noise alone is ≈ 130 per team, before pace (≈ 25) and rate uncertainty.
- So an iid sim gives sd(margin) ≈ 16.4. Real games need a restoring force and a shared latent.

**Per sim:**
- **Latents:** a game latent η_g ~ N(0, τ_g²) on both teams' efficiency; team latents η_t; player shooting latents ζ_i; pace P_s ~ N(P̄, σ_P²).
- **Regulation** runs as eight 6-minute segments; in segments 7–8, pace shifts by E[pace | |m₃|].
- **Each possession** draws a **multinomial terminal outcome** over {2PA, 3PA, FT trip, TOV} from the on-court rates. Independent Poissons are not used, because they double-count pace variance.
  - Makes are binomial on logit p̂ + δ_t + η_g + η_t + ζ − ρ·m_{k−1}·s_t (s_t = +1 for the leader), so the **restoring force acts in every segment**. Q4 adds a leader-coasting term β_q(|m₃|).
  - Misses continue as OREB with Binomial(reboundable misses, oreb%), for 3 rounds (residual mass < 0.3%).
  - Reboundable misses include the last missed FT, and a team-rebound share is removed before player allocation (S7).
- **Allocation** is Dirichlet-multinomial by on-court weight. Minutes come from §4.2: Q1–Q3 and Q4-by-|m₃| shares on the stint path, else box minutes pro rata.
- **OT** periods are 5/48 of P_s each, repeated while tied.

**SMM (M1e, M2):**
- σ_P, τ_g, τ_e, τ_p, ρ and β_q are fit **per test season on seasons < S**.
- **Fit targets:** Var(H | line), Cov(H, A | line), sd(margin) and sd(total) within ±3%; b₄ within ±0.05; Q4 pace within ±1 per bin; P(OT) within ±1 pp of the per-season rate (4.4–6.1% in 2019-26; the old 5.5–6.5% band fails 2023-26); FG% dispersion. Measured values: `research/sim_step0_targets.json`, write-up `docs/research/sim-step0.md` (T2.5).
- **Validation targets (not fitted)**, from T2.7 on seasons < S:
  - role-pair and opponent correlations within ±0.02;
  - the blowout curve within ±1.0 DK per bin;
  - the star-absence bump within ±0.6;
  - CV by level within ±0.03.
- All of these are quoted only on 2024-25 and 2025-26.

**Gates:**
- **G1:** if the τ's reach 0 and the variance targets still fail, Path B runs in **allocation mode**. Team scores come top-down as in Path A, and the possessions only split them among players.
- **G2:** Path B replaces Path A for DFS only if, on 2024-25/2025-26, player DK CRPS is not worse by more than 1 SE, lineup-score calibration is at least as good, and the realised proxy payout is not worse. Otherwise it serves props and the why.

**Monte Carlo and seeds.**
- DFS uses **S = 25,000** sims, with 50,000 for portfolios of N > 20. At p = 0.01 the SE is 0.0006.
- Seeds are `hash(slate_id, run_kind, model_version)`. These are common random numbers, so projections move between runs only when the inputs move.
- Per-sim player FP is stored as float16 `npz`: about 8 MB per slate, pruned after 8 days.

## 5. DFS engine (`proj/dfs/`)

### 5.1 Rules and scoring (T0.6, M4)
- **Scoring constants** live in `sources/dfs.CLASSIC_SCORING`, because DK's point values aren't in any JSON. They are verified two ways:
  1. **FPPG reconciliation:** for **≥ 50 draftables** per check, recompute DK's `fppg` (attribute 219) from `player_games` over the matching window (season to date, or last season before ~5 games). Accept if |Δ| ≤ 0.1 for ≥ 95% of them. Players with ≥ 10 triple-doubles in the window discriminate between a 4.5 and a 3.0 TD bonus (about 0.5 FPPG apart).
  2. **RotoGuru DK pts:** an exact match on ≥ 99.5% of joined player-games, with ≥ 20 triple-doubles among them.

  The check reruns weekly as a data check.
- **Rules** come from the archived `rules_json` at build time and are **never hard-coded**: slots, salary cap, `gameCount.min`, and **`teamCount.min` (= 2)**.
- **Eligibility** is per slot from the draftable rows: G = {PG, SG}, F = {SF, PF}, UTIL = any. **`isDisabled` rows are excluded**, and players whose game has started are locked.

### 5.2 Solver
- **OR-Tools CP-SAT** (Apache-2.0, arm64 wheels). The variables are binary x_{i,slot} over eligible slots. Constraints:
  - each slot filled once, each player at most once;
  - salary ≤ cap;
  - ≥ `gameCount.min` games and ≥ `teamCount.min` teams (indicator variables);
  - locks and bans.

  FP is scaled ×100 to integers. The roster model is about 200 lines.
- **`pydfs-lineup-optimizer` is the test oracle:** equal optimal objectives on 50 slates (RotoGuru and live). It isn't used in production: its rules are hard-coded, the CBC backend is about 10× slower, and it has no simulation objectives.
- **Concurrency:** each solve uses `num_workers=1`, run in a thread pool of 2 on the Mac.

### 5.3 Cash
- **Primary objective:** maximise **P(score ≥ c_s)** over the shared sims.
  - c_s is the cash line in sim s: the field-proxy (§5.5) score at the contest's paid fraction from its payout table (e.g. 44.4% for a double-up).
  - Candidates: k-best (§5.4, 100 each) under μ_i − λ·(μ_i − p25_i) for λ ∈ {0, 0.15, 0.3, 0.5}, each scored on the sims. The best is returned, with its P(cash).
- **Fallback** (Phase 1, before the field proxy exists): maximise Σ(μ_i − λ·(μ_i − p25_i))·x_i, with λ chosen on the RotoGuru backtest (default 0.3).
- **Availability:**
  - At lock−45, players with P(play) < 0.85 are excluded *only if their game locks before the next run*.
  - Later-game Q players are allowed, with an automatic swap candidate recorded for §5.6.

### 5.4 Top N
- **k-best by no-good cuts:** after solution L_k, add Σ_{i∈L_k} x_i ≤ 8 − d (d = "min different players").
- N solves are sequential. Lineups are ranked by mean, then by P(cash) if available.
- **Display:** diff vs #1 and the p10–p90 range from the sims.

### 5.5 GPP portfolio (contest builder)
**1. Candidates (S11).** 300 CP-SAT solves on the 2-thread pool:
- 150 **scenario objectives** 0.5·μ_i + 0.5·FP_{i,s}, for random s;
- 100 **ownership-penalised** objectives μ_i − λ·own_i, with λ ∈ U(0, 0.5·σ_μ);
- 50 **noise** objectives μ_i·exp(N(0, 0.12²));
- plus the Top-50 by mean.

**2. Ownership proxy** (no ownership data exists, so it is labelled "proxy" everywhere).
- **Model:** logit own_i = a + b₁·value_z + b₂·position projection rank + b₃·salary_z + b₄·**injury_replacement** (promoted with Δmin ≥ 6) + b₅·`fppg`_z + b₆·team-total_z, normalised to Σ = 800%.
- The b's start hand-set and documented. They are refit by beta regression on uploaded `%Drafted` once ≥ 10 slates exist (T3.2).

**3. Field proxy (S12).** F = min(field size, 20,000) lineups:
- **40% noisy-optimal:** 200 CP-SAT solves on μ_i·exp(N(0, 0.15²)), resampled with replacement. This reproduces optimizer duplication.
- **60% value-greedy with chalk tilt:** sequential sampling ∝ own_i^1.5 × value_i, subject to the slot and cap constraints, rejecting lineups below 97% of the cap.
- The weights 40/60 and the tilt are **assumptions**.
- **ROI is reported under three field strengths:** soft (20/80), medium (40/60), sharp (70/30). The page shows medium, with the soft–sharp range.

**4. Payout ladder.**
- Use the contest's real `payoutSummary`. A described contest uses its preset curve; with none, the fallback is the top-20% / 1% / 0.1% ladder.
- Per sim, store the field-proxy score at each payout-tier boundary rank (F scaled to the field size): a tiers × S matrix built in chunks.
- A lineup's payout in sim s is a tier lookup. Ties and our own entries split or occupy positions jointly.

**5. Selection.**
- Greedily add the candidate with the largest **marginal expected payout** of the portfolio, averaged over S = 25–50k.
- Constraints: overlap ≤ γ shared players with every chosen lineup (default 5 of 8) and per-player exposure caps (default max 60%, plus user min/max).

**6. Duplication.**
- E[dupes_L] = field size × q(L), where q(L) ≈ c·Π_{i∈L} own_i × h(salary left).
- c and h are calibrated on the field-proxy sample: exact matches for common lineups, and the product form for rare ones.
- Expected dupes split the payout in step 5. The indicator reads Low < 0.3, Med 0.3–2, High > 2.

### 5.6 Lock-relative runs and late swap (T1.6, T2.14)
- **Watcher.** A launchd/systemd job on the Mac fires every 5 minutes from 16:00 to 23:59 ET. Each tick:
  - for each archived slate, triggers `courtside dfs run --slate X --kind lock45|lock10` when now ∈ [lock−47, lock−40] or [lock−12, lock−7];
  - first refreshes the official report, ESPN injuries and draftables (§3.2);
  - is idempotent on (slate, kind).
- **Late swap.** For each later game tip in a slate, a `swap` run fires at tip−10, but only if `late_news` touches a player in a saved lineup.
- **Re-solver:** players in started games are fixed. The unlocked slots are re-solved under the same objective with the remaining cap. Portfolio overlap and exposure are re-checked against the fixed parts. The output is a DK edit CSV.
- No automation of DK beyond the CSV.

### 5.7 Showdown (T2.15, N6)
- Binary c_i (captain) and u_i (UTIL) per player, with c_i + u_i ≤ 1, Σc = 1 and Σu = 5. Salary is Σ(1.5·s_i·c_i + s_i·u_i) ≤ cap. Score is 1.5·FP for the captain. ≥ 2 teams comes from `rules_json`.
- One game, so the shared margin draw and the opposing-star correlation matter more. Path B is preferred here once G2 passes.
- The field proxy gets a captain ownership term, so captain chalk is modelled separately.
- No late swap (DK rule).

### 5.8 Runtime budgets (Mac mini M4, loaded, 2 threads)

| Step | Budget |
|---|---|
| nightly state (rates, minutes features, ownership refit) | ≤ 3 min |
| projection, 15-game slate, Path A, 25k sims | ≤ 30 s |
| projection, Path B (pilot + 10k per game) | ≤ 60 s (limit 120) |
| cash + Top-20 | ≤ 10 s |
| GPP N = 20: 300 candidates + 200 field solves (~0.1–0.2 s each, 2 workers) ≈ 50 s; field 20k sampled ≈ 10 s; tier matrix + greedy over 25k sims ≈ 30 s | **≤ 120 s** |
| GPP N = 150 at 50k sims | ≤ 300 s |
| late-swap re-solve | ≤ 30 s |

## 6. Data model: migration 9 (owner T0.0)

| Table | Key columns |
|---|---|
| `dfs_salaries` (alter) | + `starter` INTEGER, + `swappable` INTEGER |
| `player_ids` | (source, source_id) PK, player_id, name, team, first_seen, last_seen, how, confidence |
| `player_aliases` | (source, name_norm, team) PK, player_id, note |
| `dfs_hist_salaries` | (game_date, rg_id) PK, player_id, name, pos, team, opp, home, starter, salary, dk_pts, minutes, raw_stats, raw_path |
| `dfs_slates` | slate_id PK (draft group), nba_day, game_type_id, tag (main / early / late / showdown), game_ids_json, lock_utc, rules_json |
| `dfs_contests` | contest_id PK, slate_id, name, entry_fee, field_size, max_per_user, guaranteed, payout_json, attrs_json, captured_at, raw_path |
| `player_projections` | id PK, game_id, slate_id, player_id, run_id, model_version, role, p_play, minutes_json, inputs_json |
| `player_projection_stats` | (projection_id, stat, cond) PK; stat ∈ min, pts, reb, ast, fg3m, stl, blk, tov, pra, dk; cond ∈ played / all; mean, sd, q05…q95 |
| `dfs_runs` | run_id PK, slate_id, kind (lock45 / lock10 / swap / manual), created_at, model_version, sim_path, path (A/B), late_news_json, seconds |
| `dfs_ownership_proxy` | (run_id, player_id) PK, own, band, version |
| `dfs_lineups` | id PK, run_id, mode (cash / topn / gpp / showdown / swap), contest_id, rank, slots_json, salary, mean, p10, p50, p90, p_cash, p_top1, e_payout, dupes, why_json, actual, pct_field_proxy, ratio_hindsight |
| `dfs_contest_results` | (contest_id, rank, entry_hash) PK, points, lineup_text, payout, uploaded_at (manual uploads) |
| `dfs_contest_ownership` | (contest_id, player_id) PK, name, pct_drafted, fpts |
| `prop_compare` | (projection_id, market, line, captured_at) PK, mkt_p_above, our_p_above |
| `postgame_why_v0` | (game_id, player_id) PK, expected, actual, terms_json, text |
| `stints`, `player_game_oncourt` | as in v1 (Phase 2) |
| `player_metric_snapshots`, `dfs_metric_snapshots` | (as_of, model, metric, window) PK, n, value, se, details_json |

## 7. Evaluation

### 7.1 Leakage rules (each one enforced by a test)
- **L1. No lookahead.** Scrambling all box, PBP, salary and injury rows on or after a cut date leaves projections and lineups up to that date bit-identical.
- **L2. Fit < S.**
  - Every model coefficient and every sim or copula parameter (σ_tier, q's, κ, CV-by-level, correlation matrix, SMM τ's, ρ, ownership b's) is fit on seasons < S inside the walk-forward.
  - Box-path hyperparameters (κ, half-lives) are frozen on **2010-11 → 2013-14**, before every RotoGuru season.
  - Stint-path hyperparameters (K_φ) are frozen on 2019-20/2020-21 and are tested only on 2021-22+.
- **L3. Season aggregates are lagged one season or rebuilt causally.** This covers `matchup_rollup`, `season_matchups`, `pt_*`, `synergy_playtypes`, `hustle_season`, role tiers, and the correlation study. `player_index` positions are a current snapshot (a documented minor leak; the DK/RotoGuru position is preferred).
- **L4. Availability.**
  - The backtest uses the `avail` oracle for regulars with expected minutes ≥ 20. It is labelled "known availability", which equals a perfect lock−10.
  - The **live-like slice** uses real reports.
  - The leak probe from v1 is kept: < 0.5% MAE gain from availability-only features.
- **L5. Starters.** The lock−45 mode never sees starters. Lock−10 mode uses the actual starters as a proxy for announced ones (≈ 99% identical), labelled as such.
- **L6. Never features:** salaries (they are only benchmarks and ownership inputs), props, same-game facts, and closing lines beyond what `gbm_market` and book team totals give at run time.

### 7.2 Projection metrics
- **Measures:** MAE, RMSE, CRPS from samples, bias, 50%/80% coverage, PIT.
- **Randomised PIT for counts (S3):** u = F(y−1) + V·(F(y) − F(y−1)), V ~ U(0,1).
- **SEs** are **two-way clustered by game and player-season** (Cameron–Gelbach–Miller, S4).
- **Population:** projected minutes ≥ 15 and played, for the conditional metrics. For DK FP, the **unconditional** population is every salaried player with projected minutes ≥ 10, with a DNP scored 0. Never filter on actual minutes.
- **Benchmarks:**
  - season-to-date average (this equals DK `fppg`; historically reconstructed);
  - last-10;
  - **salary-implied** (a per-season linear fit of DK pts on salary, on seasons < S);
  - rung-2 EWMA × per-minute rates;
  - live: DK `fppg` itself.

### 7.3 DFS backtest (T2.1, T2.17)
- **Design:** walk-forward over the RotoGuru seasons, test **2015-16 → 2020-21** (about 1,000 slates), with the box path, Path A, and causal state day by day. Each slate runs cash, Top N and GPP (N = 20, medium field).
- **Metrics:**
  - FP MAE/CRPS vs salary-implied, the `fppg` equivalent and last-10;
  - lineup score / hindsight-optimal (same eligibility);
  - percentile vs the field proxy at all three strengths;
  - a paired comparison vs the same optimizer fed salary-implied or `fppg`;
  - realised cash rate vs the proxy cash line (the 44.4% field score);
  - calibration of the simulated lineup-score distribution.
- **Acceptance** (significant at 2 SE, two-way or slate clustered):

| Metric | Target |
|---|---|
| DK FP MAE (unconditional) | ≥ 5% better than the best of salary-implied / `fppg`-equivalent / last-10 |
| DK FP CRPS | ≥ 5% better than a naive-mean + tier-CV normal |
| 80% / 50% coverage (DK FP) | 77–83% / 47–53%; each role bucket 74–86% |
| PIT deciles (randomised) | each 8–12% |
| cash lineup vs `fppg`-optimizer lineup | mean paired Δ > 0 at 2 SE; wins ≥ 55% of slates |
| hindsight ratio (cash) | reported; must beat both baseline optimizers |
| lineup-score calibration | PIT deciles 8–12%; 80% coverage 76–84% |
| GPP simulated vs realised top-1% (vs field proxy) | ratio within 0.7–1.3 |
| minutes MAE (≥ 15) | ≤ 4.5 avail; ≥ 8% better than EWMA |

- **Projection-only backtest 2021-22 → 2025-26** (no salaries): the stint path (T2.2–T2.6) must not be worse on CRPS than the box path before it is used for DFS.

### 7.4 Live tracking (from 2026-10-20)
- **Every lock−10 build is stored:** cash, Top-N and default GPP. The nightly scoring gives realised points, hindsight ratio, field-proxy percentile, the comparison against the `fppg` optimizer, and FP MAE vs DK `fppg` and salary-implied.
- These feed `dfs_metric_snapshots` (windows 7d, 30d, season) and a `/health` DFS tab.
- **Real ground truth:** the owner may upload DK contest standings CSVs from contests he entered. They contain ranks, points, lineups, `%Drafted` and FPTS. From them we get:
  - the **real cash line**, compared with our predicted c_s distribution (PIT);
  - **real top-1% scores**;
  - **real ownership**, which calibrates the proxy (Phase 3: MAE and rank correlation of the proxy are reported once ≥ 10 slates exist).

### 7.5 Props bar (Fable answer, N5)
- **Markets:** pts, reb, ast, PRA, 3PM, using the distribution **conditional on playing**.
- **Three criteria:**
  1. **Calibrated:** Spiegelhalter |z| < 2.5 on P(above line), n ≥ 500.
  2. **Beats naive:** log-loss better than season-average and last-10 distributions.
  3. **Adds information:** the logit-average of ours and the no-vig market beats the market alone by > 2 clustered SE, with both taken at the **same snapshot time**.
- Parity with the market is *not* the v1 bar.

### 7.6 What we can and can't claim
- Simulated ROI isn't real ROI, and the field is a proxy.
- A 19%-ROI player still loses money over 48% of 100-GPP stretches, so the owner's occasional entries **cannot** measure ROI. They measure ownership and cash lines.
- The RotoGuru backtest is optimistic on minutes (known availability) and pessimistic on eligibility (single positions). Honest live numbers start 2026-10-20, and `/health` says so.
- With ~1,000 slates, a top-1% rate has an SE of ≈ 0.3 pp: enough to tell 1% from 2%, but only against the proxy field.

## 8. Build plan

- **Code:** `src/courtside/proj/` and `proj/dfs/`. Tests: `tests/test_proj_*.py`, `tests/test_dfs_*.py`.
- **Ownership of shared files:** `db.py` and the `cli.py` hooks belong to T0.0. `collect.py` and `sources/dfs.py` belong to T0.7, then are released.
- **Legend:** ★ critical path; ∥ parallel; O / S / H = Opus / Sonnet / Haiku; **Mac** = heavy, run on the Mac mini (`~/Developer/NBA`, push from geekom1).
- **Agent limit:** at most **3 agents at once on geekom1**, and one heavy job at a time.
- **Priority order:** P(play) + minutes ≫ rates with usage > joint correlation (GPP only) > pace/opponent > defender/refs.

### Phase 0: foundations (Oct 19–22)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T0.0 ★ S | migration 9, `courtside proj` / `courtside dfs` CLI | `db.py`, `cli.py`, `proj/__init__.py`, `proj/cli.py` | §6 → schema | idempotent migrate; `test_db` green |
| T0.1 ∥ H | verify rescued reports + rolling capture (S10) | `backfill_injuries.py` (checks only) | `injury_snapshots` → data-check report | 151 game days present; parse ≥ 99% of rows; name match ≥ 98%; a report capture is scheduled before every lock run |
| T0.2 ★ H Mac | RotoGuru backfill + parse | `sources/rotoguru.py`, `backfill_rotoguru.py` | HTML → raw + `dfs_hist_salaries` | ≥ 98% of dates 2014-10-29 → 2021-05; ≥ 99% joined to `player_games`; slate table per §3.1 |
| T0.3 ∥ H | cdn box backfill 2019+, PBP rsync to the Mac | `backfill_pbp.py` (box kind) | cdn → raw | ≥ 99% of PBP games have a box |
| T0.4 ∥ S Mac | stints store (Phase 2 input) | `analytics/stints_store.py` | → `stints`, `player_game_oncourt` | ≥ 99% of games validate |
| T0.5 ★ S | ID mapping DK/RotoGuru → NBA | `proj/ids.py` | §3.4 | §3.4 rates; Jalen/Jaylin test |
| T0.6 ★ H | rules + scoring verification (M4) | `proj/dfs/rules.py` | `rules_json`, draftables, RotoGuru → rule objects + check | §5.1: ≥ 50 FPPG within 0.1; RotoGuru exact ≥ 99.5% incl. ≥ 20 TDs; `teamCount`, disabled, eligibility tests |
| T0.7 ★ S | DK capture: starter/swappable, contest payouts, lock-relative refresh | `sources/dfs.py`, `collect.py` (dfs step) | lobby → `dfs_contests`, `dfs_slates` | 30–40 contests/day captured before lock; ≤ 60 DK requests/day added |
| T0.8 ∥ H | GBM total OOF sd + margin–total residual corr per season | `model/gbm.py` (2 attrs) | — | values logged per fold; rung-3 tests green |

### Phase 1: cash + Top N live (Oct 20 – 27)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T1.0 ★ S Mac | **eval-lite** (M5): walk-forward 2024-25 (fit < 2024-25) and RotoGuru 2019-20/2020-21 | `proj/evallite.py` | projections → report | runs in ≤ 20 min; Phase 1 gate below |
| T1.1 ★ S | EB per-minute rate state, nightly | `proj/state.py` | box → `rate_state` | no-lookahead test; ≤ 60 s |
| T1.2 ★ O | minutes v0.5: EWMA + vacated minutes + promotion + margin-conditional g + starter flag | `proj/minutes.py` | state, scenarios → minute distributions | S14 ≥ 8-min test; star-absence +3.1 ± 0.6 DK; avail MAE ≤ 4.8 |
| T1.3 ∥ S | P(play) recalibration (M6, was T2.9) | `proj/availability.py` | reports → Beta cells | §4.1 |
| T1.4 ★ S | Path A v0 copula (tier correlations), reconciliation to team totals, store | `proj/copula.py`, `proj/reconcile.py`, `proj/store.py`, `proj/pipeline.py` | → sims, `player_projection_*` | DK FP 80% coverage 76–84% on 2024-25; ≤ 30 s per slate |
| T1.5 ★ S | CP-SAT engine: cash fallback, Top N, min-different, lock/ban, exposure; pydfs oracle | `proj/dfs/optimizer.py`, `proj/dfs/topn.py` | rules, projections → lineups | pydfs equal on 50 slates; Top-20 ≤ 5 s |
| T1.6 ★ S | lock-relative watcher, `late_news`, `dfs_runs` | `proj/dfs/runner.py`, `deploy/` (watcher units) | schedule, draftables → runs | replay of 3 captured slates fires inside the windows; idempotent; late_news unit tests |
| T1.7 ∥ S | `/dfs` page v1: pool, Top N, cash, why-this-lineup, DK CSV | `web/dfs_view.py`, `templates/dfs.html`, `proj/dfs/why.py`, `proj/dfs/export.py` | runs → page/CSV | 390 px light/dark; CSV accepted by the DK template (owner manual check) |
| T1.8 ∥ S | post-game why v0 + ≤ 30 LLM stories | `proj/why_v0.py`, `brief/prompts/player_why.md` | box vs projection → `postgame_why_v0` | exact sums; 50 samples: 0 sign violations, 0 unsupported numbers |
| T1.9 ∥ H | preseason role evidence (N2) | `proj/preseason.py` | cdn preseason boxes → features | flags for players with ≥ 1 preseason start |

**Phase 1 gate** (on eval-lite):
- DK FP MAE ≥ 3% better than the season-to-date average on 2024-25 (avail);
- not worse than salary-implied on 2019-21;
- 80% coverage 76–84%;
- minutes MAE ≤ 4.8.

If the gate fails, the `/dfs` page ships with a "beta" label.

### Phase 2: model v1 + backtest harness (Oct 28 – Nov 14)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T2.1 ★ S (O review) Mac | RotoGuru DFS backtest harness: walk-forward, baselines, hindsight, random-lineup percentile, leak tests | `proj/backtest.py`, `proj/metrics.py` | history → report | reproduces T1.0; L1–L6 tests; leak probe < 0.5% |
| T2.2 ★ O Mac | minutes v1 GBM (+ coach by date N4, learned promotion, S15 fallback) | `proj/minutes.py` (takes over) | → | §4.2 targets; S14 still passes |
| T2.3 ∥ O | usage v1: with/without φ | `proj/usage.py` | stints, box → φ | > 2 SE over the tier bump |
| T2.4 ∥ S | live-like slice eval (2025-12-22 → 2026-04-12, real reports, lock−45 mode) | `proj/backtest.py` (slice mode, after T2.1) | → report | minutes MAE and DK CRPS reported vs the avail backtest; sets the live alert baseline (M6) |
| T2.5 ★ O Mac | **sim step 0 measurement** (M1a) | `research/sim_step0.py`, `docs/research/sim-step0.md` | PBP 2019-24 → targets | all §4.5 step-0 quantities with CIs |
| T2.6 ★ O Mac | Path B sim v1 + SMM per fold + pilot reconciliation (M7) | `proj/sim.py`, `proj/smm.py` | → | §4.5 fit/validation targets; G1/G2 verdict written |
| T2.7 ∥ S Mac | causal correlation study v2 (residual vs projection, tiers to date, per season < S) | `research/dfs_correlations_v2.py`, `proj/corr.py` | → matrices | matrices per fold; drift reported |
| T2.8 ∥ S | environment v1 (team totals, pace α) + opponent-by-position | `proj/environment.py`, `proj/defense.py` | → | ships only with CRPS gain > 1 SE |
| T2.9 ∥ S | Tonight cards + `prop_compare` (pts/reb/ast/PRA/3PM) | `web/players_view.py`, `templates/_players.html`, `proj/props.py` | → | ≥ 90% main props joined; both % shown |
| T2.10 ★ S | Path A copula v1 (T2.7 matrices, CV-by-level fit, winner/loser shift) | `proj/copula.py` | → | §7.3 coverage/PIT; blowout curve ±1.0 DK |

**T2.10 results: Path A copula v1 vs v0 (2026-10-20). Verdict: not shipped as the DFS default; kept behind `COURTSIDE_COPULA=v1`.**

What v1 is (`proj/copula.py`, section v1; `docs/PLAYERS.md` §4.5 step 4 v1):
- **Components, not DK points.** Per player and sim, v1 draws pts, fg3m, reb, ast, stl, blk and tov. Each is an NB2 count, var = m + c²(m)·m², at the sim's mean. The means are the same as v0's: rate x minutes x f(min/mpg) x bump, with points scaled by g·c and the other stats by (g·c)^α. Draws are inverse-CDF table lookups (4,096 quantile cells x 321 mean rows per stat), with stochastic rounding between grid means, so E[count] equals the mean exactly.
- **CV-by-level fit (per stat).** c²(m) comes from 16 level bins per stat, on residuals given actual minutes and actual team points over the 2 seasons < S. Points: 1.9 (m ≈ 0.7) → 0.034 (m ≈ 28). reb / 3PM / tov are near Poisson. blk and stl are the most over-dispersed. The implied DK spread matches the observed DK residual within 1–5 % at every level (`copula_v1.diagnostics.dk_by_level`), e.g. 2024-25 sd 7.26 vs 7.30 at 20–30 DK and 10.99 vs 10.53 at 50+.
- **Within-player correlation.** A 7 x 7 latent Gaussian correlation per role tier, fit by simulated moments: start from cov / E[h_s h_t] (h = Hermite-1 coefficient of the marginal), then four Newton steps on the simulated count covariances. The simulated count correlations match the observed ones to ±0.03. Latent pts–fg3m is 0.57 (stars) / 0.70 (role) / 0.69 (bench); star pts–ast −0.17; reb–tov +0.08; everything else |r| ≤ 0.09.
- **Across players.** The T2.7 causal PM tables (`corr_<S>.json`: seasons S−3…S−1, k = 2,000; teammates by tier x position cell, opponents by tier pair) act on each player's latent DK direction. Z_i = b_i·w_i + E_i, with w ~ N(0, R_players), b_i = R_w a_i / √(a_i′R_w a_i), a_i = DK weight x h at the player's level, and E_i ~ N(0, R_w − b_i b_i′). So Var(Z_i) = R_w exactly, and the player-level correlation is the table's (unit test: DK corr 0.29 for ρ = 0.30).
- **Winner/loser shift.** Per stat and tier, sum(y) / sum(m) by signed team margin (9 bins), relative to the tier's overall ratio, shrunk n / (n + 300) and clipped to [0.8, 1.25]. Fit on seasons < S given actual minutes and team points. In the sim it is applied at the sim's margin; points are re-normalised per team and sim, so the market total still holds. 2024-25 fit: losers' tov ×1.07–1.17; big winners' tov ×0.85–0.92, fg3m ×1.15–1.20, ast ×1.08–1.11; big losers' blk ×0.85–0.88.
- **DK points** are scored exactly from the counts with the `dfs.rules` Classic table: DD +1.5 and TD +3 more are discrete jumps (test: every played sim's DK minus the linear part is in {0, 1.5, 4.5}).
- **Common random numbers.** The first four RNG streams are v0's, so v1 has the same margins, totals, availability scenarios, minutes and tiers as v0 for the same seed. Two new streams draw the component noise and the grid rounding.
- **Per-stat sims stored.** `<run_id>.stats.npz` sits beside the DK sims: uint8 counts for the 7 stats + minutes rounded up, first 5,000 sims, 7.2 MB for a 15-game / 468-player run (the DK npz: 12.3 MB). `props.compare_game` uses the empirical played-sims pmfs (PRA per sim) when the file exists and the player has ≥ 200 played sims; otherwise the NB-given-minutes approximation. `player_projection_stats` gets real sd / quantiles for pts / reb / ast / fg3m on v1 runs (count quantiles).

**Switch and versioning.** `pipeline.COPULA` ("v0" default; env `COURTSIDE_COPULA=v1|v0`; `set_copula()` at run time). It composes with the context switches as a suffix:

| Context arm | v0 (default) | v1 |
|---|---|---|
| OFF | `pathA-v0.2`, `bundle_<S>_v1.json` | `pathA-v0.2+c1`, `bundle_<S>_v1_c1.json` |
| P+F2 (`COURTSIDE_CONTEXT=1`) | `pathA-v0.3`, `_v1_cx.json` | `pathA-v0.3+c1`, `_v1_cx_c1.json` |
| P+F2+F3+F6 (`+COURTSIDE_CONTEXT_EXT=1`) | `pathA-v0.4`, `_v1_cx2.json` | `pathA-v0.4+c1`, `_v1_cx2_c1.json` |

A `_c1` bundle is the arm's v0 bundle (loaded from disk, or fitted the v0 way) plus `copula_v1`. The params fit takes ~6 s. The bundle carries the T2.7 table, falling back to v0's tier pairs when no `corr_<S>.json` / extracts exist; the source is recorded in `copula_v1.corr.source`. Seeds stay on `pathA-v0.2`, so v1 shares draws with its arm.

**Every existing arm is unchanged.** HEAD c402c4a (`git archive`) was compared with HEAD + only the T2.10 files, with the switch unset and with `COURTSIDE_COPULA=v0`. The comparisons ran on 2025-01-15 (11 games, 4,000 sims, 1.31M sim values) and on the live opener slate 154575 at 2026-10-20 12:00Z (2.65M sim values). OFF `pathA-v0.2`, P+F2 `pathA-v0.3` and P+F2+F3+F6 `pathA-v0.4` are each **bit-identical**: players, stats, sims and meta (9 / 9 comparisons). A `Bundle` without `copula_v1` serialises byte for byte as before.

**Evaluation on top of `pathA-v0.4`.** T2.1 harness per-date code (`backtest.SeasonRun.run_date`, avail protocol, closing lines, 4,000 sims, every date) via `research/copula_v1_eval.py`, which also keeps the per-stat, blowout and lineup metrics. Both arms share bundles and seeds, so the rows and expected minutes are identical and the differences are paired. SE clustered by date. Gain > 0 means v1 is better. Mac scratch `courtside-compute/c1`; reports in `var/reports/copula_v1/`.

| Season | DK CRPS, exp min ≥ 10 (gain, SE, z) | DK CRPS, all rows | DK MAE, exp min ≥ 10 | 80 % cov v0 → v1 (Δ, SE) | 50 % cov | PIT deciles max dev | Ship bar |
|---|---|---|---|---|---|---|---|
| 2023-24 (21,767 / 38,125 rows, 160 dates) | 5.6503 → 5.6507 (−0.0003, 0.0010, −0.3) | +0.0006 (+1.0 SE) | 8.0207 → 8.0202 (+0.4 SE) | 0.820 → 0.825 (+0.0044, 0.0007) | 0.505 → 0.513 | 0.007 → 0.008 | CRPS ok; cov / PIT slightly worse |
| 2024-25 (21,865 / 38,409 rows, 162 dates) | 5.6423 → 5.6434 (−0.0011, 0.0013, −0.9) | +0.0001 (+0.1 SE) | 7.9897 → 7.9895 (+0.1 SE) | 0.824 → 0.829 (+0.0051, 0.0007) | 0.513 → 0.522 | 0.011 → 0.012 | **CRPS −0.9 SE: fails** |
| 2019-20, P+F2 arm (RotoGuru) | 5.5548 → 5.5547 (+0.1 SE) | +0.8 SE | −1.8 SE | 0.821 → 0.826 | | 0.008 → 0.008 | |
| 2020-21, P+F2 arm (RotoGuru) | 5.5145 → 5.5134 (+1.1 SE) | +1.9 SE | +2.5 SE | 0.826 → 0.830 | | 0.012 → 0.011 | |

By tier, v1 widens bench (0.854 → 0.861 / 0.849 → 0.857) and role (0.797 → 0.803 / 0.805 → 0.811). Stars are unchanged (0.837 / 0.845). v0 was already a little wide (0.82), so v1 moves coverage away from 0.80. Likely cause: both arms' dispersion is fit on plain EB-rate residuals, while the v0.4 rates are sharper. v1's per-stat Poisson floor (3PM and tov bins are under-dispersed vs Poisson) adds a little more.

*Lineups* (sum of per-sim DK over the lineup; 10–90 coverage and PIT of the actual score):

| Lineups | v0 cov80 (SE) | v1 cov80 | PIT max dev v0 → v1 | Notes |
|---|---|---|---|---|
| Cash lineup (`ours_cash`), 2019-20 / 2020-21, P+F2 arm, real RotoGuru scores | 0.832 (0.031) / 0.856 (0.030) | 0.839 / 0.863 | 0.06 → 0.07 / 0.07 → 0.08 | Simulated mean − actual is +8…+11 in both arms: selection bias of an optimised lineup, not the copula. 11–13 % land below q10 vs 2–4 % above q90 |
| 30 random 8-player sets per date (≥ 15 projected minutes), 2023-24 / 2024-25 | 0.826 / 0.831 (0.007 / 0.006) | 0.829 / 0.830 | 0.016 → 0.017 / 0.023 → 0.024 | the joint (correlation) calibration; unchanged |
| Projected top-8 per date, 2023-24 / 2024-25 (n 160 / 158) | 0.844 / 0.816 | 0.894 / 0.835 | | noisy (SE 0.03) |

*Per-stat calibration* (played, ≈ 25.7k player-games per stat and season). Rows compare v1's component sims with the props.py approximation on the same v1 run's means. The v0 run's approximation is within ±0.002 of the v1 one. "Line" = floor(season-to-date average) + 0.5, a market-like line; "median line" = our own median + 0.5.

| Stat | PIT max dev, sims vs approx (2023-24 / 2024-25) | P(above line) vs realised, sims (Spiegelhalter z) | approx z | Log-loss sims vs approx |
|---|---|---|---|---|
| pts | 0.020 / 0.020 vs 0.033 / 0.031 | 0.462 vs 0.469 (7.7) / 0.474 vs 0.480 (10.3) | 8.1 / 10.4 | equal (±0.0001) |
| reb | 0.004 / 0.005 vs 0.004 / 0.005 | 0.454 vs 0.455 (3.7) / 0.459 vs 0.459 (4.1) | 3.1 / 3.6 | equal |
| ast | 0.004 / 0.006 vs 0.006 / 0.009 | 0.445 vs 0.446 (1.5) / 0.455 vs 0.454 (1.5) | 0.9 / 0.9 | equal |
| 3PM | 0.004 / 0.003 vs 0.003 / 0.004 | 0.366 vs 0.363 (0.6) / 0.373 vs 0.378 (3.4) | 0.2 / 3.1 | +0.0002…0.0003 worse (Monte Carlo noise of 4k sims) |

The per-stat sims are sane and better shaped for points: the first PIT decile is 12.0 % vs 13.1–13.3 %. Both methods miss the points left tail: early exits (injury, fouls) are not in the minutes engine. At the season-average line the mean chance is right (±0.007), but points and rebounds fail §7.5's |z| < 2.5 in both methods: they are too sharp around the line, so they need a wider minutes left tail, not a copula fix.

*Blowout curve* (per tier: mean DK by margin bin minus the tier's overall, simulated at the sim's margin vs actual at the actual margin; T2.10 target ±1.0 DK):
- **|margin|.** Stars miss in both arms. 2024-25: 0–5 −0.7 (v0 −0.9), 6–10 −1.8, 16–20 +1.7, 21–25 +2.0, 26+ +1.8. 2023-24: up to +3.3 at 26+. The sim's star curve spans 4.5 DK; the actual spans 7.0. The curve is minutes-driven: in the T2.7 study stars lose 4.7 min at 26+ and gain 2.5 in close games, and v1 shares v0's minutes by construction. Role and bench are within ±1.0 in both arms, except role 26+ in 2023-24 at +0.93.
- **Signed margin.** The winner/loser shift fixes the losing side: stars ≤ −21 +3.0 → +1.5 (2024-25) and +4.9 → +3.7 (2023-24); −20…−11 +2.0 → +0.7 and +1.9 → +0.8; role ≤ −21 +1.1 → +0.4 and +1.7 → +1.1. It overshoots big winners: stars ≥ 21 +1.2 → +2.4 and +1.6 → +2.7. The per-minute boost is right given actual minutes, but the sim's winning stars keep too many minutes.
- **Fix belongs in the minutes engine** (T2.2): the `g_table` blowout response of stars is too flat in the sim.

*Runtime.* Projection of a 15-game day (2025-02-12, 468 players) at 25k sims, P+F2+F3+F6, geekom1 (Ryzen 7 8745HS, 2 threads):

| | v0 | v1 |
|---|---|---|
| Warm total | 5.7 s | 11.7 s |
| Cold total | 11.5 s | 19.0 s |
| Sims | 2.7 s | 7.2 s |
| Summaries | 1.95 s | 3.2 s (vectorised count summaries) |
| Peak RSS | 1.0 GB | 1.2 GB |

On the Mac harness at 4,000 sims, sim time per date is 0.72 → 1.05–1.28 s. Extrapolated to the loaded Mac (similar single-core speed, ±30 %): v1 ≈ 12–25 s, **inside the §5.8 30 s budget**.

**Verdict and next step.** v1 is DK-neutral within ~1 SE: CRPS −0.3 / −0.9 SE on the harness population, +0.1…+1.9 SE on all rows, and +1.1 SE in 2020-21. But it is a little wider: +0.4…0.5 pp of 80 % coverage at ~7 SE, PIT slightly worse. 2024-25 misses the "no CRPS loss beyond −0.5 SE" bar, so **it is not the DFS default**. It is the better source of per-stat distributions for props and exact DD/TD. Before re-testing:
1. Shrink the dispersion to the arm's sharper rates: fit c²(m) on the context-arm residuals, or one variance factor fit on season S−1 end to end.
2. Fix the stars' blowout minutes in the minutes engine.

Then re-run `research/copula_v1_eval.py`. Turning it on (owner decision) is described in the T2.10 deliverable notes: `COURTSIDE_COPULA=v1` in the launchd plists, or `COPULA_DEFAULT = "v1"`; prefit `bundle_2026-27_v1_cx2_c1.json` with `corr_2026-27.json` present; then `courtside proj live-path-eval`.

**T2.4 results: live-like slice (2026-10-21).** `courtside proj backtest-dfs --seasons 2025-26 --protocol live|live10 --slice` (`proj/backtest.py` `live_like_day` / `LiveInputs`; driver `research/t24_slice.py`, Mac scratch, 4,000 sims). 104 regular-season dates 2025-12-22 → 2026-04-12, 808 games.

How it was run:
- **Run time:** each game was run the way the live lock runs see it, at its **own tip − 45** (DK has a Showdown slate per game, so a game's graded run is its own lock run). The official report was the one published by then.
- **Candidate path:** live (`merge_extras` with a DK-pool stand-in first, report-listed players forced, `candidate_panel` + `drop_moved_players`).
- **DK-pool stand-in:** each player's team of his next game; players with no earlier game in the season only within 2 days of their debut.
- **P(play) cells:** fit **leaving the date's month out**. The shipped cells were fit on exactly these days. In-sample cells change nothing material: P(play) gaps within 0.01, DK MAE −0.009.
- **Lines:** no 2025-26 `line_snapshots` exist, so the closing line was used for all 808 games (recorded per game).
- **`live10`:** the same at tip − 10, with the actual starters as the announced-starter proxy (§7.1 L5).
- **SEs:** two-way clustered by day and player.

Results by arm and protocol: value (SE).

| arm / protocol | minutes MAE (played, E[min∣plays] ≥ 15) | DK MAE (E[min] ≥ 10, DNP = 0) | DK CRPS | 80% cov | DK bias | minutes MAE (unconditional) |
|---|---|---|---|---|---|---|
| v0.2 avail | 4.403 (0.064) | 7.998 (0.094) | 5.648 (0.064) | 0.833 | +0.14 | 4.998 |
| v0.2 **live (lock−45)** | **4.457 (0.065)** | **8.214 (0.101)** | **5.796 (0.069)** | 0.836 | +0.08 | 5.248 |
| v0.2 live10 | 4.387 (0.063) | 8.039 (0.094) | 5.685 (0.065) | 0.835 | +0.08 | 4.993 |
| v0.3 avail / live / live10 | 4.403 / 4.457 / 4.387 | 7.978 / 8.193 / 8.015 | 5.638 / 5.785 / 5.673 | 0.833 / 0.836 / 0.834 | | |
| v0.4 avail / live / live10 | 4.403 / 4.457 / 4.387 | 7.980 / 8.198 / 8.019 | 5.637 / 5.786 / 5.674 | 0.835 / 0.835 / 0.835 | | |

What not knowing who plays costs (live − avail, same arm and dates; every arm within 0.004):
- **Lock−45:**
  - minutes MAE **+0.054 (0.016)**;
  - DK MAE **+0.216 (0.048)**, +2.7%;
  - DK CRPS **+0.147 (0.032)**, +2.6%;
  - unconditional minutes MAE +0.25 (0.06);
  - coverage and bias unchanged.
- **Lock−10:** nearly all of it is recovered. Minutes MAE −0.016 (0.016), DK MAE +0.04 (0.04), CRPS +0.04 (0.03).
- **Context arms** keep their gains on the live path: v0.3 − v0.2 DK MAE −0.021 (0.004), CRPS −0.011 (0.003); v0.4 −0.016 (0.006) / −0.010 (0.003).
- **Tiers (v0.2, lock−45):**
  - DK bias: bench −0.67 (0.23), role +0.52 (0.13), star −0.52 (0.27).
  - Under avail it is −0.27 / +0.45 / −0.19: the live path moves stars and bench down.
  - Minutes bias, played ≥ 15: −0.20 / +0.38 / −0.10.

P(play) by report status at lock−45 (v0.2; the arms are identical). Columns: projection P(play) / report cell P / share that played.
- Out 0.003 / 0.005 / 0.002 (n 4,980).
- Doubtful 0.004 / 0.005 / 0.000 (111).
- **Questionable** 0.474 / 0.541 / 0.539 (514), gap −0.065 (0.026).
- **Probable** 0.845 / 0.939 / 0.947 (207), gap −0.102 (0.019).
- **Available** 0.753 / 0.871 / 0.865 (1,315), gap −0.111 (0.012).
- Not listed (team filed) 0.811 / – / 0.835 (18,590), gap −0.025 (0.005).
- At lock−10: Q 0.621 / 0.702 / 0.597 (gap +0.024 (0.033)), P −0.054, Available −0.099.
- GTD: no rows. The official reports have no GTD status, and no ESPN rows are archived for the slice.

Two findings for the pipeline owner (not changed here: `pipeline.py` / `copula.py`):
1. **DNP is counted twice for listed players.**
   - The report cells are fit on *played* (min > 0), so they already contain coach's-decision DNPs. They are calibrated: Q/P/Available pooled 0.795 vs 0.791 played.
   - `build_games` multiplies them by the minutes engine's P(plays | available) (0.52 bench / 0.90 rotation / 0.99 starter by E[min∣plays]). Listed bench players come out at 0.29 vs 0.45 played, rotation 0.72 vs 0.85; starters are fine (0.82 vs 0.85).
   - DK bias of listed Q/P/Available players: **−1.33 (0.39)** at lock−45, −0.61 (0.33) at lock−10.
   - Unlisted bench players (the engine alone) are only mildly low: 0.50 vs 0.55, rotation 0.93 vs 0.95.
   - Hook: for a listed, not-Out player, use the report cell as P(play) and the engine's played-minutes distribution, i.e. P(plays | available) = 1, or refit the cells on "active".
2. **Forced pool players outside the rotation are over-played.** These are players forced in from the pool who are neither on the report nor in the team's last 10 games: about 8 a night, P(play) 0.69 vs 0.24 played. Live forces the whole DK pool the same way.

**Live alert baselines (M6)**, written to `src/courtside/mlops/live_like_baseline.json`. It ships with the code; `<data>/proj/live_like_baseline.json` overrides it, and `$COURTSIDE_LIVE_LIKE_MIN_MAE` overrides `min_mae`.
- `min_mae` = **4.457**, the lock−45 v0.2 value. It wakes `dfs:live_min_mae`, which fires at > 4.96 over the last 300 rows, n ≥ 150.
- Nightly ranges are percentiles of the slice nights (median 144 rows), as p10 / p50 / p90:
  - DK MAE 7.37 / 8.10 / 9.15;
  - coverage 0.800 / 0.835 / 0.880;
  - minutes MAE, played ≥ 15: 3.93 / 4.42 / 5.26;
  - unconditional minutes MAE 4.43 / 5.10 / 6.18;
  - CRPS 5.23 / 5.69 / 6.39;
  - P(play) calibration error on report-uncertain rows (0.015 < report P < 0.985, about 16 a night): −0.24 / −0.12 / +0.05. It is centred below 0 because of finding 1.
- Last-300-row windows, p5–p95: coverage 0.797–0.876, minutes MAE 3.96–5.11, starter minutes bias −0.55…+0.86 (inside the ±1.0 `dfs:min_bias` limit).
- The file also has per-model blocks for v0.3 / v0.4 and a lock−10 reference (minutes MAE 4.39, DK MAE 7.23 / 7.94 / 8.88).
- New rule `dfs:live_range:<metric>`, **info** severity: last night outside p10–p90, widened by √(144 / rows tonight). It is high-side only for the MAEs. Existing alerts are unchanged.
- These are Dec–Apr nights. October nights (prior-heavy rates) may run worse.

**Opener check (2026-10-20, graded 03:00).** Graded on a clone of the prod DB.
- Night as the monitor sees it: newest lock10 per game, 60 rows. DK MAE 8.71, coverage 0.817, minutes MAE (played ≥ 15) 3.16–3.27, unconditional minutes MAE 6.16. The main-slate lock10 alone (n = 49): DK MAE 7.33, minutes 4.63, coverage 0.857.
- Every value is inside its band, widened ×1.55 for 60 rows, or below it on the good side. No `live_range` alert fires, and `live_min_mae` waits for 150 rows.

**P(play) v2 results (p2, 2026-10-21).** Switch `COURTSIDE_PPLAY=v2` (`pipeline.PPLAY`, default "v1"; `set_pplay()`), version suffix `+p2` on whatever arm is active (`pathA-v0.3+p2`, `pathA-v0.3+c1+p2`); the bundle file and the seeds do not change. Code: `pipeline.pplay_v2_fix` (called from `project_games` and `backtest.live_like_day`), `build_games(pp_fix=...)` → `copula.ScenarioMinutes` (a `pp_fix` column replaces the minutes engine's P(plays ∣ available) where set), `availability.fringe_features` / `FringeParams` / `fringe_params.json`, `pipeline.classic_salaries`. Research: `research/p2_rows.py` (candidate rows), `p2_fit.py` (fringe fit + CV), `p2_eval.py` (slice tables), `p2_opener.py` (live slates off vs p2 + lineups), `p2_identity.py` (bit-identity); Mac scratch `courtside-compute/p2` on a copy of the prod DB of 2026-10-21 08:41.

Fix 1, listed players: **P(available) = the report cell, P(plays ∣ available) = 1** for every listed, not-Out player (Q / P / Available / D / removed / day-to-day).
- Why this one: the cells are fit on *played*, so they are already the calibrated total (slice: Q/P/Available 0.795 vs 0.791). Setting the engine's term to 1 makes the total equal the cell with nothing to divide or clip. Dividing by the engine's P(plays ∣ available) fails exactly where the bug is: for bench players the engine says ~0.52 and the cell ~0.85, so the ratio is clipped at 1 and the total stays low; the engine's value also moves with every scenario (depth rank). Refitting the cells on "active" would need an active/inactive label we don't store, and it would leave the total resting on the engine's DNP model for an atypical population (players coming back, restrictions): two calibrations instead of one.
- A listed player's non-play is now drawn as a scenario absence (teammates are re-projected, as for any absence), not as an in-sim DNP.

Fix 2, fringe pool players (forced in, `roster == 0` = not in the team's last 10 games, no listed report status): a logistic P(plays ∣ available), features strictly before the day.
- **Mid season** (the team has played ≥ 5 games, or the player has appeared this season): `one, avg10 (mean minutes of the last 10 appearances), no appearance this season`, fit on the 868 slice fringe rows at lock−45. Leave-one-month-out: P 0.261 vs 0.252 played, log-loss 0.550 vs 0.565 for a constant. Most features carry little here (days since last game, games missed, app365 all flat at ~0.25); only avg10 > 20 (0.40–0.46) and a first appearance (0.57, mostly 10-day signings) stand out.
- **Season start** (team games < 5, no appearance this season; every opener player): `one, avg10, log(1 + appearances in 365 days), log(DK Classic salary / 3000), salary at the minimum`, fit on the RotoGuru DK pools 2017-18 … 2020-21 (two-way-contract era; 1,375 rows). No reports exist for those seasons, so pool players a report would list are in. Each row is weighted by P(not listed ∣ played / DNP returning within 3 games / DNP streak, role) measured on the slice's forced rows (0.89 / 0.56 / 0.41 bench, 0.55 / 0.24 / 0.16 rotation); rows at the $3,000 minimum stay unweighted. Leave-one-season-out on the weighted population: P 0.360 vs 0.356, log-loss 0.498 vs 0.651. By salary: $3,000 0.20 / 0.19 played, $3.1–3.9k 0.70–0.75, $4.5k+ 0.82–0.89 vs 0.86–0.91. Without the weights the $4.5k+ movers came out at 0.65–0.75 (7 of 8 played at the opener); weighting the minimum-salary rows too gave them 0.33 vs 2 of 9 (both 1-minute cameos). Showdown slates read the day's Classic salaries; no salary → a salary-free variant (`early_nosal`).
- **Priced players keep the engine at season start:** a fringe player DK prices at ≥ $4,500 (Classic) in the early regime keeps the minutes engine's P(plays ∣ available) (`availability.EARLY_ENGINE_SALARY`). The weights cannot remove every would-be-listed player from the RotoGuru pools, and for priced players those leftovers are nearly all of the DNPs: the model reads 0.82–0.89 where the live, not-listed population plays almost always (opener: 4 of 4 at ≥ $4.5k played). On the 2026-10-21 preview the model alone cut the 47 priced fringe players (starters and movers) by up to 0.28 (Irving, VanVleet, Lillard, Haliburton back from season-long injuries: 0.71–0.83), which is the "don't penalise real rotation players" case. The rule only exists in the first 5 team games; mid season there is no salary input.
- DK starter flag on a fringe player → 0.99 (§4.1). Clamp [0.02, 0.97].

Bit-identical with the switch off (`research/p2_identity.py`: sha1 over every sim array and the players frame; HEAD 32a6a02 via `git archive` vs the working tree, same DB copy and bundles, 2,000 sims):
- v0.2, v0.3, v0.4: live (lock−45) and live10 on 2026-01-15, avail 2026-01-15, reports 2026-01-15, the live opener path (all five DK slates of 2026-10-20 at lock−45 with their pools); v0.3 / v0.4 also avail 2024-01-10. v0.4+c1: avail 2024-01-10; v0.3+c1: avail 2020-01-15. **All 19 hashes identical.** (v0.2+c1 has no prefit bundle; its code path is the same `pp_fix=None` one.)
- With p2 **on**, the avail protocol (the T2.1 harness's oracle) is identical to off on 20 / 20 dates of 2023-24 and 20 / 20 of 2024-25 plus 2024-01-10 and 2026-01-15: it has no report and no forced pool players, so nothing changes there and the harness cannot regress. The live, live10, reports and opener hashes change, as they should.
- `tests/test_proj_pplay_v2.py` (10 tests: versions, off never calls the v2 path, all-NaN override identical, listed total = cell, rule table, project_slate +p2, features without lookahead, opener edge cases, shipped calibration, weighted logit). `tests/test_proj*` + `tests/test_dfs*`: 587 passed.

Slice (T2.4 setup, 104 dates, 808 games, 4,000 sims, P(play) cells leave-one-month-out, closing lines), v0.3 vs v0.3+p2. Paired p2 − off on a **fixed** population (E[min] ≥ 10 in either arm, DNP = 0; each arm's own population would drop the fringe rows p2 pushes below 10), SE two-way clustered by day and player:

| | lock−45 off → p2 | Δ (SE) | lock−10 off → p2 | Δ (SE) |
|---|---|---|---|---|
| DK MAE (15,140 / 14,992 rows) | 8.218 → 8.148 | **−0.069 (0.020)** | 8.035 → 7.955 | **−0.080 (0.020)** |
| DK CRPS | 5.793 → 5.732 | **−0.061 (0.017)** | 5.675 → 5.608 | **−0.067 (0.017)** |
| DK bias | +0.01 → +0.04 | +0.030 (0.020) | +0.04 → +0.08 | +0.038 (0.019) |
| 80% coverage | 0.836 → 0.840 | +0.003 (0.002) | 0.835 → 0.838 | +0.002 (0.001) |
| minutes MAE, unconditional | 5.322 → 5.251 | −0.071 (0.019) | 5.081 → 4.995 | −0.086 (0.019) |
| DK MAE bench / role / star | | −0.191 (0.053) / −0.040 (0.023) / +0.001 (0.020) | | −0.216 (0.051) / −0.052 (0.021) / +0.011 (0.014) |
| DK CRPS bench / role / star | | −0.155 (0.039) / −0.037 (0.020) / −0.013 (0.019) | | −0.168 (0.037) / −0.048 (0.020) / +0.007 (0.009) |
| DK bias bench / role / star | | +0.05 (0.06) / +0.01 (0.02) / +0.07 (0.03) | | +0.08 (0.06) / +0.02 (0.02) / +0.07 (0.03) |
| listed, not Out (1,634 rows): DK MAE / bias | 10.09 → 9.85 / −1.46 → −0.31 | −0.240 (0.059) / +1.15 (0.08) | | −0.259 (0.069) / +1.29 (0.09) |
| fringe (176 rows): DK MAE / bias | 11.90 → 8.33 / +8.46 → −2.22 | −3.56 (1.10) / −10.7 (0.6) | | −4.06 (1.00) / −9.2 (0.7) |
| everyone else (13,330): DK MAE | 7.940 → 7.937 | −0.002 (0.006) | | −0.007 (0.005) |

The slice's own cells (each arm's population, `slice_report`): lock−45 DK MAE 8.204 → 8.153 (−0.051, 0.016), CRPS 5.777 → 5.738 (−0.039, 0.013), minutes MAE (played, ≥ 15) 4.461 → 4.466 (+0.005, 0.008); lock−10 −0.066 (0.016) / −0.049 (0.012) / −0.008 (0.008). Stars and starters' minutes are unchanged within noise; the gain is bench and role players.

P(play) calibration, lock−45 (projection P off → p2 vs played; gap SE two-way clustered):

| group (role by avg10) | n | off | p2 | played |
|---|---|---|---|---|
| listed, not Out: all | 2,275 | 0.622 (−0.086, 0.011) | **0.714 (+0.007, 0.010)** | 0.707 |
| … bench / rotation / starter | 471 / 867 / 937 | 0.251 / 0.630 / 0.800 | 0.507 / 0.716 / 0.816 | 0.423 / 0.723 / 0.836 |
| Questionable / Probable / Available | 682 / 248 / 1,178 | 0.455 / 0.834 / 0.760 | 0.530 / 0.926 / 0.876 | 0.522 / 0.923 / 0.868 |
| fringe: all | 868 | 0.658 (+0.405, 0.028) | **0.246 (−0.006, 0.022)** | 0.252 |
| … bench / rotation / starter-level | 616 / 187 / 65 | 0.613 / 0.773 / 0.750 | 0.220 / 0.299 / 0.343 | 0.221 / 0.289 / 0.446 |
| roster, unlisted (unchanged) | 18,611 | 0.811 | 0.811 | 0.836 |
| Out | 4,876 | 0.003 | 0.003 | 0.001 |

Brier: listed 0.131 → 0.123, fringe 0.360 → 0.174. At lock−10 the fringe is 0.257 vs 0.253 and listed 0.798 vs 0.758: the **lock−10 Questionable cell runs high** (0.663 vs 0.531 played, +0.13 (0.024); off was +0.06 because the double count pulled it down). That is the cells' own lt45m-bucket problem (T2.4 saw it too), now visible because p2 uses the cell as is: refit / pool that bucket next. Lock−10 DK metrics still improve.

Opener 2026-10-20 re-projected on the live path (`research/p2_opener.py`: every Showdown slate's game at its lock−45, DK pool of the capture before then, report of the time, production 2026-27 bundles, v0.3 arm, 5,000 sims, Classic salaries):
- Team minute sums (Σ P × E[min ∣ plays]) unchanged: BOS 236.8 → 236.7, DET 228.5 → 228.4, NYK 235.5, PHI 231.9, OKC 232.4, SAS 233.3 (actual 228–241). The minutes p2 takes from fringe players go to the team's real rotation.
- **Hayden Gray** (BOS, $3,000, 1 game in 365 days): P 0.83 → **0.12**, 16.7 → 2.7 min, 13.5 → 2.2 DK (DNP). **Bruce Brown** (NYK, $4,100, 88 games): 0.93 → 0.81, 18.2 → 16.3 min, 14.5 → 12.9 DK (DNP; his salary and history say rotation, so the model keeps him fairly high). Other min-salary fringe (Agbaji, Williams, Peter, Eubanks, Bradley, Z. Smith, D. Jones): 0.36–0.53 → 0.11–0.24, none played.
- Rotation players forced in from the pool: priced ≥ $4.5k keep the engine (George, LeBron, J. Brown, Collins 0.97–0.99, all played 25–35; Drummond 0.41 and Robinson 0.74 stay, both played). Below $4.5k the model: KCP ($4.2k) 0.95 → 0.80 (played 26), Joe ($3.5k) 0.93 → 0.76 (19), Conley 0.71 → 0.79, Rupert 0.80 → 0.71 (both DNP). Without the priced rule the model had cut George / LeBron / J. Brown / Collins to 0.83–0.92 (−1.9…−2.7 DK each).
- Listed: Simons 0.62 → 0.65 (cell 0.65), Wade 0.54 → 0.60 (0.60), Bona 0.23 → 0.47 (0.46).
- Top 15 by DK mean: same players, moves −0.02 … +1.5 DK (minutes freed by the fringe players go to the rotation).
- Fringe rows (25, 10 played): mean P 0.64 → 0.55, log-loss 0.68 → 0.56.
- Lineups (cash + Top-10, min-diff 1, scored with actual DK points): slate 154575 (3 games) cash 238.2 both, Top-10 mean 240.8 → 246.0; slate 154827 (2 games) cash 214.5 both, Top-10 mean 226.8 → 230.8, DNP slots in the Top-10 1 (Bruce Brown) → 0. Two slates: anecdotal, no lineup claim. The slice has no 2025-26 salaries, so there is no lineup backtest for p2.

Tonight 2026-10-21 (preview at each slate's lock−45 with the 10:00 capture and report, DK ids resolved on a DB copy; 11 games, no outcomes yet). Every team has played 0–1 games, so the pool carries 113 fringe players (offseason movers, players back from long injuries, two-ways):
- Priced ≥ $4.5k (47, e.g. Doncic, Leonard, Irving, VanVleet, Lillard, Haliburton): unchanged, P 0.93 on average.
- $3,000 minimum (29): P 0.51 → 0.20, DK 4.7 → 1.8. Julian Reese (NOP, $3,000, avg10 30.3) was the #8 value play by DK/$ at 0.96 → 0.17; Trey Lyles 0.80 → 0.10, Buddy Hield 0.90 → 0.26, Ben Simmons 0.64 → 0.12.
- $3.1–3.9k (27): 0.72 → 0.73 on average (both directions). $3.9–4.5k (10): 0.84 → 0.76 (Beal 0.98 → 0.70, Allen 0.96 → 0.79, Smart 0.98 → 0.79): the honest early-season rate for that price (RotoGuru weighted 0.80; opener 1 of 2).
- Listed, not Out (23): P 0.35 → 0.56 = the cells (0.56).
- Team minute sums unchanged (220–240 off and p2, max move 0.2); top-20 projections move −0.9 … +2.9 DK.

**Minutes-ratio results (mr, 2026-10-21; switch default OFF, not shipped; bench follow-up the same evening: design C tested, not adopted).** Switch `COURTSIDE_MINRATIO=1` (`pipeline.MINRATIO`, `MINRATIO_DEFAULT = False`; `set_minratio()`). It is a suffix on any arm: bundle `<arm's bundle>_mr.json` (before a `_c1`), version `<arm>+mr` (`pathA-v0.3+mr+p2`), seeds unchanged (common random numbers with the arm's OFF runs). Code: `proj/minutes_ratio.py`, `state.minutes_multiplier(..., planned=)` / `minutes_multiplier_mr`, `copula._team_minutes(plan_out=)` / `_PlanGrids` (v0 and v1 sims), `copula.fit_copula(planned=)`, `state.fit_usage_bump` (reads a `planned` column), `pipeline._load_bundle_mr`, `features.store._key` (adds the table's hash only when present). Research: `research/minutes_ratio_eval.py` (`run`: paired tables with bench cuts; `proxy`: table designs at realised minutes); Mac scratch `courtside-compute/mr` on a copy of the prod DB (ON rows: design B `out_b`, design C `out_c`).

The finding (docs/research/return-minutes.md): per-minute DK error moves with E[min ∣ plays] / recent minutes for everyone (+2…9 % under-projected below, −2…−6 % over-projected above 1.2×). The cause: f_stat(m / mpg) (`state.fit_params`, `minutes_adj`) is fit on *realised* minutes, where a low ratio is mostly an unplanned early exit (fouls, injury, blowout) with lower production per minute, and the sims apply it at the drawn minutes over mpg, so a *planned* change (an injury ramp, a promotion when a star sits) gets the same per-minute penalty or boost.

Design:
- **f is re-keyed on the plan:** E[stat ∣ m] = rate × m × f_u(m / P). P = the minutes engine's E[min ∣ plays] before the margin. In the sims it is the `mean` of each sim's own availability scenario (a promoted player's P is higher in the scenarios where the starter sits), so the draw's deviation from its plan is what gets the per-minute change. In training, P is the same engine walk-forward under the avail oracle (season X's minutes params fit on the 5 seasons < X; 98.3 % of rows have a plan, the rest fall back to P = mpg). f_u is the binned ratio of sums over the 5 seasons < S, on the same 9 bins as f. This pooled table is **design B, the switch's design** (`minutes_ratio.PL_EDGES = ()`); the P-tiered design C was tested in the bench follow-up below and not adopted.
- Fitted pts f_u (2023-24): 0.73 / 0.77 / 0.85 / 0.93 / 0.99 / 1.02 / 1.05 / 1.08 / 1.11 at m / P ≈ 0.26 … 1.9, vs f(m / mpg) 0.77 … 1.09. It is a bit steeper: the unplanned deviation is a cleaner signal.
- **Refit for the ON arm, all on seasons < S:** the usage bump (a ratio of sums given the table), copula v0 (alpha, CV-by-level and correlations are residuals given the table), and the **context layer, with its Poisson offset E_s re-keyed** (E0_s × f_u / f per row, then the market reconciliation recomputed; `minutes_ratio.adjust_offsets`). The layer's F2 inputs keep the old f expectations: they are built the same way in training and live, and live has no plan for other teams' past games. Copula v1 params are kept (the combination is wired, not evaluated). Minutes params and P(play) are untouched, so minutes are bit-identical ON vs OFF.
- **Variant tested and dropped: a second factor g_p(P / mpg)** (backfitted with f_u, normalised to 1 at P = mpg; `PLAN_FACTOR = True`). The training fit gives g_p ≈ 1.02–1.07 at P = 1.3–1.5 × mpg (per-minute *gain* for planned promotions). Walk-forward it pushes promoted players back up, against the over-projection the finding is about. It is weaker everywhere (CRPS −0.0022 (0.0009) / −0.0020 (0.0009), live −0.0006 (0.0017), live10 +0.0014 (0.0018)) and the star tier is worse in 2023-24 (+1.3 SE), so it fails the bar. The switch runs the plain re-key (g_p ≡ 1). The table records `plan_factor`, and `_load_bundle_mr` refits a cached bundle of the other design.

Bit-identical with the switch off, final tree. HEAD c8ded64 via `git archive` vs HEAD + this change, same DB copy and bundles, `pathA-v0.3+p2`:
- Historical 2024-01-10: players 308 × 36, stats 3,696 × 12, 30 sim arrays (1.23M values) and meta all identical. This also matches the 8a05151 output the OFF evaluation rows came from.
- Opener slate 154575 at 2026-10-20 22:15Z: players 106 × 38, stats 1,272 × 12, 9 sim arrays (2.65M values) and meta all identical.
- p2_identity: 5 / 5 hashes identical.
- With the switch ON, design B's outputs on both are unchanged bit for bit (historical 2024-01-10 and slate 154575 vs design B's earlier ON outputs: all identical).
- `tests/test_proj_minutes_ratio.py`: 15 tests. Two are new: the tiered table recovers a flatter bench curve, is continuous in P and equals the pooled table when its tiers agree; and the shrinkage / version guard. The default design is asserted to be B. `tests/test_proj_{copula,state,pipeline,features,context,pplay_v2,backtest}*` pass (157).

Evaluation, ON − OFF paired on identical rows (E[min] ≥ 10, DNP = 0; avail = T2.1 harness, 4,000 sims, closing lines, every date, each arm's bundles fit < S; live-like = T2.4 slice 2025-26 with p2, 104 dates). SE clustered by date. Minutes and P(play) are identical in every pair.

| | 2023-24 avail | 2024-25 avail | 2025-26 live+p2 | live10+p2 |
|---|---|---|---|---|
| rows | 21,767 | 21,865 | 14,937 | 14,814 |
| DK MAE (OFF) | 8.025 | 7.995 | 8.153 | 7.962 |
| DK MAE Δ | −0.0040 (0.0021) | −0.0041 (0.0026) | −0.0065 (0.0044) | −0.0017 (0.0044) |
| **DK CRPS Δ** | **−0.0037 (0.0013), −2.8 SE** | **−0.0031 (0.0015), −2.0 SE** | −0.0018 (0.0028) | +0.0017 (0.0029) |
| DK bias (mean − actual) | −0.08 → −0.11 | +0.05 → +0.01 | +0.08 → +0.03 | +0.11 → +0.06 |
| 80 % coverage | 0.820 → 0.824 | 0.824 → 0.829 | 0.840 → 0.843 | 0.838 → 0.840 |
| PIT max deviation | 0.0072 → 0.0085 | 0.0108 → 0.0129 | 0.0175 → 0.0177 | 0.0172 → 0.0174 |
| CRPS z star / starter / rotation / bench | −0.3 / −3.1 / −1.8 / +0.2 | −0.8 / −1.3 / −2.0 / +0.9 | −0.3 / −0.3 / −0.8 / −0.4 | +0.4 / +0.7 / +0.1 / −0.1 |
| bench (E[min ∣ plays] < 15) DK MAE Δ | +0.013 (0.007) | **+0.020 (0.008)** | +0.010 (0.009) | +0.012 (0.009) |
| promoted (E[min ∣ plays] > 1.2 × last-10), DK MAE Δ / bias∣played | −0.026 (0.005) / −0.79 → −0.55 | −0.013 (0.007) / −0.74 → −0.48 | −0.012 (0.011) / −1.02 → −0.76 | −0.007 (0.012) / −0.88 → −0.60 |
| ratio < 0.8, bias∣played | −0.21 → −0.49 | +0.51 → +0.27 | +0.66 → +0.39 | +0.66 → +0.37 |
| return game 0, CRPS Δ / bias∣played | −0.046 (0.020) / +1.55 → +1.08 | −0.027 (0.014) / +1.62 → +1.18 | −0.035 (0.017) / +0.71 → +0.24 | −0.023 (0.019) / +0.78 → +0.35 |
| long absence 16+, games 0–2: bias∣played | +1.23 → +0.99 | −0.22 → −0.41 | −0.95 → −1.24 | −0.99 → −1.23 |
| soft tissue, games 0–2: bias∣played | | | −0.71 → −0.98 | −0.73 → −0.99 |
| per-minute error, ratio 0.7–0.8 / > 1.4 | +2.9 → +1.2 % / −4.3 → −2.8 % | +3.2 → +1.9 % / −4.7 → −3.1 % | +5.7 → +4.1 % / −3.6 → −2.1 % | +4.8 → +3.1 % / −3.3 → −1.7 % |
| star out, remaining top 6: bias∣played; gap vs other nights | −0.11 → −0.08; −0.04 → −0.10 | −0.57 → −0.52; −0.40 → −0.46 | −1.01 → −0.94; −0.48 → −0.51 | −0.92 → −0.85; −0.60 → −0.64 |

RotoGuru seasons (same harness with DK salaries, so it also has lineup lines; minutes are box minutes):

| | 2019-20 avail | 2020-21 avail |
|---|---|---|
| rows | 18,681 | 19,079 |
| DK MAE Δ | −0.0062 (0.0021) | −0.0109 (0.0025) |
| DK CRPS Δ | −0.0047 (0.0014), −3.3 SE | −0.0073 (0.0015), −4.9 SE |
| 80 % coverage / PIT max deviation | 0.821 → 0.825 / 0.0082 → 0.0091 | 0.827 → 0.831 / 0.0117 → 0.0136 |
| CRPS z star / starter / rotation / bench | −0.6 / −3.9 / −2.4 / **+2.9** | −1.4 / −4.6 / −3.0 / +0.3 |
| bench DK MAE Δ | **+0.035 (0.007)** | **+0.018 (0.007)** |
| promoted, DK MAE Δ / bias∣played | −0.028 (0.008) / −0.51 → −0.22 | −0.037 (0.008) / −1.00 → −0.73 |
| return game 0, CRPS Δ / bias∣played | −0.073 (0.022) / +1.84 → +1.20 | −0.090 (0.024) / +1.96 → +1.34 |
| per-minute error > 1.4× | −2.7 → −0.8 % | −4.5 → −2.8 % |
| star out top 6: gap vs other nights | −0.57 → −0.58 | −0.66 → −0.69 |
| cash lineup, actual DK (paired by date) | 277.5 → 276.5 (−1.0 ± 1.6) | 273.5 → 272.1 (−1.4 ± 1.6) |
| Top-20 mean, actual DK | 275.6 → 276.8 (+1.2 ± 0.4) | 271.5 → 271.6 (+0.1 ± 0.5) |
| cash hindsight ratio | 0.698 → 0.695 | 0.693 → 0.690 |

Reading:
- **The mechanism is real and the fix removes about 35–50 % of it.** Per-minute error at > 1.4× drops from −4.3/−4.7/−3.6/−3.3 % to −2.8/−3.1/−2.1/−1.7 %. Promoted players' DK over-projection shrinks by about 0.25 DK, and game-0 returners' under-projection by about 0.45. The rest of the gradient is not the minutes table: the usage bump and the minutes engine's own promotion (T2.7/T2.3) remain.
- **The overall gain is small.** DK MAE −0.004 (≈ 1.6–1.9 SE). That is a third of Fix B's proxy (−0.009…−0.014). Fix B was a free post-hoc factor keyed on the same box ratio it was scored on; the structural fix can only move what the table explains. CRPS clears 2 SE in 2023-24 (−2.8) and only just in 2024-25 (−2.0).
- **The star-out gap does not move** (T2.7's −0.83 is now +0.11 / −0.40 avail and −0.48 / −0.60 live after P + p2; ON moves star-out and normal nights alike, +0.02…+0.07). It is not a minutes-table artefact. The S14 value play is inside the promoted rows (less over-projected).
- **Long-absence and soft-tissue returners get slightly worse on the live path** (−0.95 → −1.24 DK bias, n ≈ 135, SE ≈ 0.85, so noise-level). This is the study's predicted unmasking: their minutes are over-projected by 0.5–1.5, and the old per-minute penalty had been hiding it.
- **The older seasons agree and are stronger** (CRPS −3.3 / −4.9 SE, DK MAE −3.0 / −4.4 SE). Lineups do not move: cash −1.0 / −1.4 (± 1.6), Top-20 +1.2 / +0.1.
- **Costs:**
  - The bench tier (E[min ∣ plays] < 15) is worse in all six sets: DK MAE +0.010…+0.035 (2.4–4.8 SE in 2019-20, 2020-21 and 2024-25), and CRPS is worse too in 2019-20 (+2.9 SE). The first hypothesis (f_u too steep below m / P ≈ 0.6 for bench draws) is only part of it: see the bench follow-up below.
  - Coverage is +0.3…0.5 pp wider and PIT max deviation is slightly worse. The ON mean is about 0.03 DK lower overall.

**Bench follow-up (2026-10-21 evening): diagnosis, design C tested, not adopted.** Design B's bench tier (E[min ∣ plays] < 15) was worse in all six sets.
- **Small-P instability: no.** The scored bench rows (E[min] ≥ 10) all have P ≈ 11–15 min, so m / P never explodes.
- **The pooled table's shape: partly.** Refit by P on the training rows (2019-20 fit), the bench curve (P < 15) is 0.77 / 0.81 / 0.86 / 0.88 / 0.91 / 0.98 / 1.01 / 1.07 / 1.11. For P ≥ 15 it is 0.67 / 0.76 / 0.84 / 0.93 / 0.99 / 1.03 / 1.05 / 1.09 / 1.10.
  - The bench curve is flatter below m / P ≈ 0.6: a bench player's short nights are spot minutes, a starter's are early exits.
  - It is 4–8 % lower at 0.8–1.0. So the pooled f_u over-projects bench plans at their usual minutes, even in sample (P 10–15: −0.7 / −1.2 %).
  - Out of sample at realised minutes (2019-20) B is −4.1 / −1.1 %, where OFF is −1.3 / +1.2 %.
  - The steep short tail alone is not the story.
- **The context offset refit: no.** The ON mean shift on bench rows (+0.14 DK) sits on the re-key's own axis: +0.17…+0.38 DK when P < 0.9 × last-10. The copula's alpha and CV-by-level move by ≤ 0.02.
- **The bench P distribution: yes, the rest.** About 40 % of bench rows are planned demotions (P < 0.9 × last-10).
  - The re-key is right for them per minute. At realised minutes OFF under-projects them by 6–12 % per minute, B by 0–4 %.
  - Their *minutes* are over-projected on the avail harness: P(play) +0.04…+0.06 and E[min] +0.4…+0.8 above actual in every season. The old per-minute penalty was hiding this, as it was for the long-absence returners.
  - The harness loss sits in the bench rows that played little or not at all: rows with 0 minutes have CRPS +0.08…+0.12 each under B.
  - Bench intervals were already too wide (80 % coverage 0.85–0.87 OFF).

**Design C** (`minutes_ratio.PL_TIERS_C`; set `PL_EDGES = PL_TIERS_C` to run it; table version 2):
- f_u is fit per tier of the plan P: 12 / 18 / 24 / 30 min, mean plans ≈ 8, 15, 21, 27, 33. Each cell is shrunk to the pooled f_u with 300 pseudo-rows and read linearly in P between the tiers.
- Fitted pts f_u, 2024-25: P ≈ 8: 0.83 … 0.95 / 0.98 … 1.09; P ≈ 15: 0.69 … 0.93 / 0.99 … 1.12; P ≈ 33: 0.78 … 1.01 / 1.03 … 1.09.
- At realised minutes (`minutes_ratio_eval.py proxy`: fit < S, scored on S without sims), the calibration RMS over P tier × m / P cells (2019-20 / 2020-21 / 2023-24 / 2024-25) is:
  - OFF: 0.041 / 0.045 / 0.041 / 0.039
  - B: 0.039 / 0.038 / 0.035 / 0.028
  - one split at 15 min: 0.034 / 0.037 / 0.031 / 0.029
  - C: 0.028 / 0.031 / 0.022 / 0.021
- C also has the lowest DK MAE in 3 of 4 seasons (tied with finer 10…30 tiers), and bench-tier MAE improves on B in all four.

Design C, ON − OFF, same harness and arms. Design B's value is in brackets:

| | 2023-24 avail | 2024-25 avail | 2025-26 live+p2 | live10+p2 | 2019-20 avail | 2020-21 avail |
|---|---|---|---|---|---|---|
| DK MAE Δ | −0.0071 (0.0019) | −0.0051 (0.0023) | −0.0072 (0.0040) | −0.0033 (0.0040) | −0.0081 (0.0019) | −0.0112 (0.0023) |
| **DK CRPS Δ (SE), z** | −0.0049 (0.0013), **−3.7** [−2.8] | −0.0035 (0.0014), **−2.4** [−2.0] | −0.0009 (0.0026), −0.4 [−0.6] | +0.0017 (0.0027), +0.7 [+0.6] | −0.0062 (0.0013), −4.7 [−3.3] | −0.0081 (0.0014), −5.9 [−4.9] |
| CRPS z star / starter / rotation / bench | −3.1 / −3.6 / −0.7 / +0.7 | −3.8 / −1.0 / −0.3 / **+1.0** | −2.9 / −0.1 / **+1.6** / +0.2 | −2.2 / +0.8 / **+2.3** / +0.3 | −3.5 / −4.4 / −1.3 / **+2.7** | −4.7 / −4.7 / −1.4 / +0.1 |
| 80 % coverage / PIT max dev. | 0.823 / 0.0080 | 0.829 / 0.0131 | 0.844 / 0.0187 | 0.842 / 0.0183 | 0.825 / 0.0094 | 0.830 / 0.0129 |
| bench DK MAE Δ | +0.009 (0.007) [+0.013] | +0.014 (0.007) [+0.020] | +0.007 (0.009) [+0.010] | +0.010 (0.008) [+0.012] | +0.031 (0.007) [+0.035] | +0.016 (0.007) [+0.018] |
| bench bias (mean − actual), OFF → ON | −0.32 → −0.23 [−0.18] | +0.14 → +0.26 [+0.30] | −0.70 → −0.62 [−0.59] | −0.53 → −0.42 [−0.39] | +0.12 → +0.25 [+0.27] | −0.00 → +0.10 [+0.11] |
| bench per-minute error (played) | +4.6 → +3.8 % | +3.2 → +2.1 % | +3.6 → +2.9 % | +2.3 → +1.3 % | +1.4 → +0.2 % | +3.7 → +2.7 % |
| bench P < 0.9 × last-10: P(play) / E[min] bias | +0.045 / +0.81 | +0.038 / +0.54 | +0.011 / +0.04 | +0.023 / +0.20 | +0.061 / +0.40 | +0.038 / +0.71 |
| 80 % width bench / rotation / star (DK) | +0.4 / +0.6 / −0.4 | +0.4 / +0.6 / −0.3 | +0.3 / +0.5 / −0.3 | +0.4 / +0.5 / −0.4 | +0.4 / +0.6 / −0.4 | +0.4 / +0.6 / −0.4 |
| promoted: bias∣played | −0.79 → −0.52 | −0.74 → −0.48 | −1.02 → −0.76 | −0.88 → −0.60 | −0.51 → −0.23 | −1.00 → −0.73 |
| return game 0: bias∣played | +1.55 → +1.17 | +1.62 → +1.27 | +0.71 → +0.33 | +0.78 → +0.43 | +1.84 → +1.28 | +1.96 → +1.43 |
| long absence 16+ / soft tissue, games 0–2: bias∣played | +1.23 → +1.06 | −0.22 → −0.33 | −0.95 → −1.14 / −0.71 → −0.92 | −0.99 → −1.14 / −0.73 → −0.94 | +2.01 → +1.81 | −0.32 → −0.52 |
| per-minute error > 1.4× | −4.3 → −2.6 % | −4.7 → −3.1 % | −3.6 → −2.1 % | −3.3 → −1.7 % | −2.7 → −0.8 % | −4.5 → −2.8 % |
| cash lineup / Top-20, actual DK Δ | | | | | −2.2 ± 1.5 [−1.0] / +1.2 ± 0.4 | −2.5 ± 1.3 [−1.4] / −0.2 ± 0.5 |

Reading:
- **C is the better per-minute table and the better aggregate.** CRPS gains are larger in every avail season, and DK MAE is lower everywhere, live included.
- **The gain moves from rotation to stars.** The stars' f_u is flatter at both ends, so their draws are no longer charged the starters' early-exit curve.
- **The bench cost shrinks but does not go.** On DK MAE it is 10–30 % smaller, and bench per-minute error improves in all six sets.
- **C costs the rotation tier on the live slice:** +1.6 / +2.3 SE. Its steeper curve at P ≈ 15–21 widens rotation and bench intervals by about 0.5 DK, where the sims' minutes already make them too wide (rotation coverage 0.835 → 0.843 live). Bench 2019-20 stays +2.7 SE, and 2024-25 reaches +1.03.
- **Lineups:** cash is −2.2 / −2.5 (± 1.5 / 1.3). The swaps are starters (E[min ∣ plays] ≈ 30) on mean differences of 0.1–0.3 DK.
- **What is left is not the per-minute table.** It is how the sims' minutes treat short-plan players: demoted bench players' P(play) and E[min] are over-projected on the harness, and rotation / bench minutes draws are too dispersed.
- **Not done:** a per-minute patch for it, such as blending back to m / mpg for short plans or flattening the tail further, would be wrong at realised minutes.

**Ship bar** (CRPS ≥ 2 SE better in 2023-24 and 2024-25, no tier worse by > 1 SE in any set, live slice not worse):
- **Design B (the switch's design): not met.** CRPS −2.8 / −2.0 SE (2024-25 exactly at the bar); live −0.6 / +0.6 SE; tiers within 1 SE in five sets but bench +2.9 SE in 2019-20 (bench DK MAE worse in all six, 2.4–4.8 SE in three).
- **Design C: not met.** CRPS −3.7 / −2.4 SE; live −0.4 / +0.7 SE; tier failures in four sets: bench +2.7 (2019-20) and +1.03 (2024-25), rotation +1.6 / +2.3 on the live slice.
- C has the better aggregate, B the fewer tier failures and the safer live path, so the switch keeps B and C stays a research option (`PL_EDGES = PL_TIERS_C`).

The switch stays OFF. Shipping is the owner's call.
- Next step: the minutes side, not the per-minute table. Fix P(play) and E[min] for players planned below their recent minutes (P < 0.9 × last-10), and check the dispersion of rotation / bench minutes draws (their 80 % intervals cover 0.84–0.87). Then re-run the same arms, B and C. C should then be the better design: it is the better table at realised minutes.
- If it ships, the long-absence returners' minutes over-projection, which the switch unmasks, should be fixed on the minutes side as well.

**Preseason results (pre, item #2, 2026-10-21; switch default OFF, not shipped).** Switch `COURTSIDE_PRESEASON=1` (`pipeline.PRESEASON`, `PRESEASON_DEFAULT = False`; `set_preseason()`). It is a suffix on any arm: bundle `<arm's bundle>_pre.json` (before `_mr` / `_c1`), version `<arm>+pre` (`pathA-v0.3+pre+p2`), seeds unchanged. The ON bundle is the arm's bundle with the minutes params refit on seasons < S (`minutes_params_<S>_v1_pre.json`). Code: `minutes.Preseason` / `load_preseason` / `attach_preseason` / `preseason_columns` (`V1_PRESEASON`, `PRE_TEAM_COLS`, `PRE_MAX_TGS = 20`), `debut_reserve` / `debut_minutes` / `fit(debut_reserve=True)` (`MinutesParams.reserve_debut`, a JSON key only when set), `project_arrays(reserve_add=)`, `pipeline._load_bundle_pre`, and the Context attaches the logs when ON. Tests: `tests/test_proj_preseason_features.py`. Research: `research/preseason_eval.py`. Mac scratch `courtside-compute/pre`.

Data and features:
- **Source:** `preseason_player_games` / `preseason_team_games` 2010-11+, NBA teams only.
  - **Scrimmages are dropped.** These are the July 2020 bubble games (40 minutes long, ~65 games into 2019-20); they are not a camp before an opener.
  - **Games against international clubs are kept** (~6 % of rows). They are part of the camp rotation, and the per-team-game share normalises them.
- **Every feature uses the row team's preseason games dated strictly before the date.** A player who changed teams reads as "did not play" rather than carrying his new team's role, so last season's departed players in historical panels do not leak roster moves.
- **Columns:** games, mpg, mean minutes in the team's last 2 games (DNP = 0), games played among those 2, share of team preseason minutes × 5, rank on the team, mpg − the team's median, mpg / prior-season mpg (regular seasons < S, floor 5), and a did-not-play flag (NaN when the team has no logs).
- **Decay:** the columns are NaN from team game 20 on. Ungated, the minutes-level walk-forward lost +0.005…+0.011 MAE in games 15–29; LightGBM barely used the columns that late (gain share ≤ 0.3 %).
- **Live:** the Context loads the current season's logs (2026-27 has 1,991 rows). Checked on 2026-10-21 (11 games, 261 candidates): all 261 have the did-not-play flag and the debut input, and the 186 who played in camp have the full set.

Early-season finding (minutes v1 OFF, walk-forward on the avail oracle, 2023-24 / 2024-25):
- **Over-projection is concentrated at game 0.** Played rows' E[min ∣ plays] residual is −0.35 / −0.80 in team games 0–4, −0.27…−0.57 in games 5–14, and ≈ 0 after.
- **The cause is the team total, not the weighting of prior-season minutes.**
  - At game 0, minutes outside the candidate set are 29 per team-game: 13 from debutants (no box row before) and 16 from non-regular arrivals that the oracle's last-season roster misses.
  - The fitted reserve is a season-average spline and gives ~3.
  - From game 1 on, outside minutes are 1–3 against a reserve of 1.4–2.3. Roster rows' unconditional bias is +0.04…+0.10 from game 1, so prior-season minutes are not over-weighted after the opener.
- **The rate state needs no change.** It already fits its season carry-over (calendar half-life × season-break discount, chosen by NB log-likelihood on seasons < S, early-season cell prior blended with last season); not touched.
- **The fix (ON only): debutants get their own reserve.**
  - OLS on seasons < S of debut minutes on the stage bin ({0}, {1–4}, {5–14}, {15+}) × the camp's new players' preseason minutes per team game, winsorised at the training p95 (~70).
  - Historically that input correlates 0.44–0.71 with opener debut minutes in every season 2018-19 … 2025-26.
  - The usual spline is refit without the debutants. Fitted (2024-25): game 0 = −6.5 + 0.58 × D; games 1–4 ≈ 0.09 × D.
  - The non-regular arrivals are an oracle artefact: live, the DK pool forces them in. They are left alone, so part of the game-0 harness gain comes from minutes taken from oracle-only DNP candidates.

Bit-identical with the switch off (HEAD c8ded64 via `git archive` vs HEAD + the change, same DB copy and bundles, `pathA-v0.3+p2`):
- Historical 2024-01-10 (avail, 4,000 sims): players 308 × 36, stats 3,696 × 12, 30 sim arrays (1.23M values) and meta all identical.
- Live opener slate 154575 at 2026-10-20 22:15Z: players 106 × 38, stats 1,272 × 12, 9 sim arrays (2.65M values) and meta all identical.
- `research/p2_identity.py` hashes (live and live10 2026-01-15, avail 2026-01-15 and 2024-01-10, opener 2026-10-20): 5 / 5 identical.
- OFF reads no preseason table, adds no panel column and writes no new JSON key.

Evaluation setup: ON − OFF paired on identical rows, T2.1 harness, avail, 4,000 sims, closing lines, every date. Both arms' minutes params are refit with the same code (1 LightGBM thread); the cached 2023-24 OFF bundle came from an older DB and differs. The population is E[min] (OFF) ≥ 10 with DNP = 0. SE is clustered by date: first 15 games ≈ 33 dates, game 0 = 3 dates, so the game-0 z values are unreliable. The 2025-26 opener weeks run on avail only: the live-like slice needs official reports, which start 2025-12-22.

| | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| rows first 15 / full | 3,940 / 21,746 | 3,899 / 21,865 | 3,964 / 22,308 |
| **first 15: minutes MAE** (OFF 4.91 / 4.70 / 4.81) | −0.027 (−1.4 SE) | **−0.032 (−2.8 SE)** | −0.020 (−0.9 SE) |
| **first 15: DK CRPS** | −0.0137 (−1.4 SE) | **−0.0253 (−3.5 SE)** | −0.0108 (−0.9 SE) |
| first 15: DK MAE | −0.024 (−1.3 SE) | −0.033 (−3.6 SE) | −0.022 (−1.0 SE) |
| game 0: minutes MAE / CRPS | −0.21 / −0.16 | −0.23 / −0.15 | −0.33 / −0.19 |
| games 1–4 / 5–14: CRPS z | −2.3 / +0.6 | −4.2 / −2.6 | −0.2 / +0.5 |
| full season: minutes MAE / CRPS / DK MAE z | −0.6 / −0.4 / +0.3 | −1.4 / 0.0 / −0.3 | +0.1 / +0.3 / +0.1 |
| 80 % coverage, full | 0.820 → 0.821 | 0.824 → 0.824 | 0.834 → 0.834 |
| full-season CRPS z star / starter / rotation / bench | **+1.4** / +0.6 / −1.2 / −0.9 | −0.5 / −0.1 / +0.2 / +0.5 | **+2.6** / −1.7 / +0.1 / +0.5 |
| first 15 minutes MAE: rookies / new team / returning (n) | −0.039 (−1.3) / −0.061 (−1.8) / +0.03 (30) | −0.019 (−0.7) / −0.025 (−1.2) / −0.38 (23) | −0.022 (−0.8) / 0.000 / +0.15 (26) |

Opener 2026-27 (live path, slate 154575 at lock −45, ON vs OFF):
- **The debut reserve over-reserves on this sample.** It takes 18–30 team minutes from BOS / OKC / PHI / SAS (OKC 232.5 → 202.6; 175.0 before the winsorisation: its camp's new players averaged 126 minutes per preseason game against a training maximum of ~95). Their actual debut minutes were 0 / 13 / 0 / 3. DET and NYK are unchanged (±1).
- Player moves: max |Δ DK| 2.8, mean −0.22.
- That is one night and 6 contenders, against a historical opener mean of 11–19 debut minutes per team. Tonight's 24 openers (reserves 0–32 per team, e.g. LAC 31.7, TOR 30.8) are the real check.

Reading:
- **Preseason information is real but small.** It is worth ~0.03 minutes MAE over the first 15 team games, concentrated in games 0–4, and nothing after. LightGBM gives the columns ≤ 0.3 % of the gain; recent box minutes dominate as soon as they exist.
- **About half of the first-15 gain is the game-0 reserve, and part of that is an oracle artefact.**
  - The oracle's game-0 candidates include last season's departed non-regulars as available DNPs (~110 rows a season with ~6.5 expected minutes).
  - A larger reserve takes minutes from them. Live, `drop_moved_players` already removes them.
  - On played rows at game 0 the reserve alone was +0.16…+0.33 MAE (worse); with the features, −0.07…+0.03.

**Ship bar** (first-15 minutes MAE and DK CRPS ≥ 2 SE better in 2023-24 and 2024-25, full season not worse, no tier worse by > 1 SE): **not met.**
- 2024-25 passes (−2.8 / −3.5 SE). 2023-24 does not (−1.4 / −1.4 SE), and its star tier is +1.4 SE.
- 2025-26 (outside the bar) is −0.9 SE, with the star tier +2.6 SE over the full season.
- The full season is flat.

The switch stays OFF. Next steps:
- Validate the debut reserve on tonight's openers, and refit it as a saturating function before any use.
- Live, project rookies from their preseason role instead of leaving them to the reserve. `candidate_panel` drops players without box history, so lottery picks get no projection before their debut.
- If the owner wants the feature part alone, it can be tested without the reserve.

**Minutes-fix results (mf, 2026-10-22; switch default OFF, not shipped).** Switch `COURTSIDE_MINFIX=1` (`pipeline.MINFIX`, `MINFIX_DEFAULT = False`; `set_minfix()`). It is a suffix on any arm: bundle `<arm's bundle>_mf.json` (after `_pre`, before `_mr` / `_c1`) = the arm's bundle with the minutes params refit on seasons < S (`minutes_params_<S>_v1[_pre]_mf.json`), version `<arm>+mf` (`pathA-v0.3+mf+p2`, `pathA-v0.3+mf+mr+p2`, `pathA-v0.3+pre+mf+p2`), seeds unchanged (common random numbers with OFF). Code: `minutes.MF_*`, `mf_plan_columns` / `mf_design` / `mf_pplay`, `fit_v1(minfix=...)`, `_mf_calibrate_noise`, `_g_rows`, `MinutesParams.minfix` (a JSON key only when set), `pipeline._load_bundle_mf`. Tests: `tests/test_proj_minutes_fix.py`. Research: `research/minutes_fix_eval.py` (`diag` / `fit` / `report`: the minutes engine alone on the avail oracle panel with minute draws as the sims make them; `harness` / `dk`: the T2.1 harness and the T2.4 slice with minutes PIT / coverage / CRPS columns added; `debut`). Mac scratch `courtside-compute/mf` on a backup of the prod DB of 2026-10-21 23:06.

Diagnosis (minutes engine alone, avail oracle, season S's params fit < S, every available candidate, 2023-24 / 2024-25; "demoted" = E[min ∣ plays] < 0.9 × the mean of the last 10 appearances, "promoted" > 1.1 ×):
- **Which players.** Demoted plans are 53 % of bench candidates (E[min ∣ plays] < 15) and 43 % of the harness's bench rows (E[min] ≥ 10). Why the plan is lower (bench demoted, 7.9k / 7.8k rows):
  - the player's own return or DNP streak (missed ≥ 1 team game, or ≤ 2 games back from ≥ 3 missed): 75 % / 73 %. P(play) +0.054 / +0.074, E[min] +0.65 / +0.66 over actual;
  - a starter back today: 7 % / 9 %, +0.065 / +0.055, E[min] +0.95 / +0.53;
  - a teammate in his groups back today: 11 % / 10 %, −0.005 / −0.032 (fine);
  - a teammate back within 2 games: 2 %, +0.07 / +0.09;
  - other (coach shortening, noise): 5 %, +0.08 / +0.03.
- **Which component.** It is P(plays ∣ available), the v1 logit. It reads only history (EWMAs, appearances, missed games, the v0.5 depth rank), not the GBM's plan for the night: demoted rows +0.050 / +0.060, promoted rows −0.067 / −0.063, steady −0.01. The minutes when they play are close: sims 9.5 vs 9.2 actual for demoted bench, the GBM base within 0.1–0.3 on played rows. The 240 rebalancing is not the source, but it carries any P(play) change into the starters. Fixing P(play) alone took 0.17 minutes per game from starters in 2024-25, through the variance weights.
- **Dispersion.** The noise was fit on log(actual / plan), which already holds the margin effect, and the sims add g(margin) on top. Its lognormal location per tier is the MLE of the log ratio, not mean-preserving: before the per-sim 240 it inflates the bench by ~9 % and stars by ~1 %. The sims therefore moved ~0.6 min a game from stars to the lower tiers: stars' E[min] −0.45 / −0.50 vs actual, while the engine's own expectation was +0.28. Minutes PIT, harness population: stars and starters too wide (80 % coverage 0.86 / 0.83–0.85, PIT humped in the middle). Bench intervals are wide only through the DNP mass; *given that he plays* a bench player's draws were too narrow (training coverage 0.72). Rotation is about right (0.84).

The fix (`fit_v1(minfix=True)`, everything on the 5 seasons < S):
1. **P(plays ∣ available) reads the plan.** The v1 logit's columns, plus log plan, log plan / fast EWMA, / slow EWMA, / median of 5, plus dummies of the depth rank by the plan among the available (≤ 5, 7 … 12). It is fit on the out-of-fold GBM plans of the training panel (departed players removed, forced extras as roster, as before) and scored with each scenario's plan, so it moves with every availability scenario. Minutes level, 2024-25, all candidates: demoted bench +0.060 → +0.039, promoted −0.063 → 0.000, Brier −0.0021 (−5.5 SE).
2. **Noise net of the margin curve, then calibrated by simulation.** MLE on log(actual / (plan + g(actual margin))). Then 4 rounds of simulation on the training projections at each game's actual margin and overtime, per tier: the location to match actual minutes given play on average, the scale to give 80 % coverage given play. Fitted σ, 2024-25, tiers < 12 / 12–20 / 20–26 / 26–32 / 32+: 0.78 / 0.40 / 0.24 / 0.17 / 0.13 → 0.91 / 0.44 / 0.25 / 0.15 / 0.07. Stars get sharper; bench given play gets wider.
- **Dropped:** a LightGBM classifier for P(plays). It looked much better on the harness (minutes CRPS −12 SE), but the gain is the avail oracle's own rule: available regulars (base_ew ≥ 20) always play, because those who did not are marked absent. It learns the jump at 20 (played 0.80 → 0.90 → 1.00); live has no such rule.
- Minutes params are refit with the same recipe as OFF. The GBM base, the v0.5 parts and the reserve are byte-identical. The OFF arm of the evaluation is the same code's refit, which matches the canonical bundles to ≤ 0.004 minutes MAE.

Evaluation, ON − OFF paired on identical rows (E[min] ≥ 10 in either arm, DNP = 0): T2.1 harness (avail, 4,000 sims, closing lines, every date), the T2.4 slice 2025-26 with p2 (live / live10, P(play) cells leave-one-month-out, **real tip times**: the scratch data dir has the ESPN scoreboard archive), and the RotoGuru seasons. Each arm's bundles are fit < S. SE clustered by date, z in brackets. Tiers by OFF E[min ∣ plays]: star ≥ 30, starter 24–30, rotation 15–24, bench < 15. The minutes columns (sims' minutes PIT / 80 % interval / CRPS of the box minutes) come from the driver, `proj/backtest.py` unchanged.

**+mf** (`pathA-v0.3+mf+p2` vs `pathA-v0.3+p2`):

| | 2023-24 | 2024-25 | live | live10 | 2019-20 | 2020-21 |
|---|---|---|---|---|---|---|
| rows | 22012 | 22193 | 15065 | 14951 | 18941 | 19350 |
| DK MAE OFF / Δ (z) | 8.010 / -0.0290 (-6.0) | 7.996 / -0.0291 (-4.6) | 8.096 / -0.0128 (-2.7) | 7.922 / -0.0191 (-3.6) | 7.858 / -0.0245 (-3.4) | 7.793 / -0.0179 (-3.2) |
| DK CRPS OFF / Δ (z) | 5.645 / -0.0191 (-6.1) | 5.645 / -0.0236 (-5.2) | 5.711 / -0.0088 (-2.6) | 5.606 / -0.0138 (-3.7) | 5.548 / -0.0179 (-3.7) | 5.508 / -0.0117 (-3.0) |
| 80% cov OFF → ON | 0.820 → 0.816 | 0.825 → 0.821 | 0.838 → 0.836 | 0.837 → 0.836 | 0.822 → 0.822 | 0.827 → 0.823 |
| DK PIT max dev | 0.0070 → 0.0088 | 0.0098 → 0.0100 | 0.0162 → 0.0152 | 0.0158 → 0.0147 | 0.0074 → 0.0070 | 0.0110 → 0.0089 |
| CRPS z star/starter/rotation/bench | -3.8 / -2.5 / -3.2 / -3.6 | -3.0 / -2.4 / -3.2 / -2.9 | -3.2 / -1.4 / -0.7 / -0.8 | -4.0 / -2.1 / -1.6 / -1.2 | -4.0 / -2.7 / -1.6 / -3.0 | -4.0 / -2.3 / -0.8 / -0.7 |
| minutes MAE OFF / Δ (z) | 5.009 / -0.043 (-9.0) | 4.986 / -0.033 (-6.1) | 5.182 / -0.023 (-4.4) | 4.931 / -0.032 (-5.5) | 4.893 / -0.028 (-4.0) | 4.988 / -0.022 (-3.7) |
| minutes CRPS Δ (z) | -0.022 (-5.7) | -0.022 (-5.0) | -0.009 (-2.0) | -0.014 (-3.2) | -0.015 (-3.4) | -0.007 (-1.5) |
| minutes 80% cov OFF → ON | 0.846 → 0.830 | 0.851 → 0.837 | 0.869 → 0.855 | 0.870 → 0.856 | 0.849 → 0.845 | 0.845 → 0.836 |
| minutes PIT max dev | 0.0211 → 0.0181 | 0.0290 → 0.0262 | 0.0377 → 0.0318 | 0.0380 → 0.0330 | 0.0305 → 0.0316 | 0.0293 → 0.0287 |
| minutes cov80 star/starter/rotation/bench OFF→ON | 0.86→0.81 / 0.83→0.80 / 0.84→0.86 / 0.87→0.90 | 0.86→0.82 / 0.85→0.82 / 0.84→0.85 / 0.88→0.91 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.86→0.89 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.85→0.89 | 0.87→0.85 / 0.86→0.84 / 0.83→0.85 / 0.80→0.84 | 0.87→0.84 / 0.85→0.83 / 0.82→0.83 / 0.83→0.86 |
| DK bias star/starter/rotation/bench OFF→ON | -0.40→-0.19 / +0.17→-0.03 / +0.10→-0.07 / -0.50→-0.40 | -0.25→-0.09 / +0.33→+0.13 / +0.03→-0.15 / -0.13→-0.06 | +0.34→+0.50 / +0.50→+0.35 / -0.47→-0.61 / -0.69→-0.55 | +0.10→+0.26 / +0.62→+0.47 / -0.46→-0.58 / -0.70→-0.52 | -0.33→-0.24 / +0.38→+0.16 / -0.03→-0.14 / -0.18→+0.01 | -0.57→-0.42 / +0.60→+0.38 / +0.16→+0.04 / -0.12→+0.08 |
| bench demoted (n): P(play) bias, E[min] bias OFF→ON | (595) +0.067→+0.063, +0.90→+0.78 | (662) +0.080→+0.075, +0.87→+0.71 | (394) +0.015→+0.017, -0.39→-0.37 | (474) -0.003→+0.001, -0.68→-0.61 | (458) +0.095→+0.092, +1.54→+1.43 | (437) +0.049→+0.041, +1.07→+0.92 |
| bench demoted DK CRPS Δ (z) | -0.086 (-4.3) | -0.099 (-3.5) | +0.024 (+1.1) | +0.017 (+0.9) | -0.032 (-1.4) | +0.016 (+0.6) |

**+mf+mr, design B** (the minutes-ratio switch's design; its planned minutes in training are the OFF engine's walk-forward plans, as in the +mr evaluation):

| | 2023-24 | 2024-25 | live | live10 | 2019-20 | 2020-21 |
|---|---|---|---|---|---|---|
| rows | 22012 | 22193 | 15065 | 14951 | 18941 | 19350 |
| DK MAE OFF / Δ (z) | 8.010 / -0.0337 (-6.6) | 7.996 / -0.0341 (-5.7) | 8.096 / -0.0196 (-3.3) | 7.922 / -0.0217 (-3.4) | 7.858 / -0.0313 (-4.2) | 7.793 / -0.0304 (-5.4) |
| DK CRPS OFF / Δ (z) | 5.645 / -0.0240 (-7.2) | 5.645 / -0.0278 (-6.4) | 5.711 / -0.0114 (-2.9) | 5.606 / -0.0129 (-3.0) | 5.548 / -0.0225 (-4.5) | 5.508 / -0.0200 (-5.2) |
| 80% cov OFF → ON | 0.820 → 0.820 | 0.825 → 0.825 | 0.838 → 0.838 | 0.837 → 0.838 | 0.822 → 0.826 | 0.827 → 0.829 |
| DK PIT max dev | 0.0070 → 0.0077 | 0.0098 → 0.0099 | 0.0162 → 0.0150 | 0.0158 → 0.0149 | 0.0074 → 0.0082 | 0.0110 → 0.0112 |
| CRPS z star/starter/rotation/bench | -4.1 / -3.3 / -3.9 / -3.9 | -3.5 / -3.0 / -3.6 / -3.1 | -3.0 / -1.4 / -0.9 / -0.8 | -3.0 / -1.4 / -1.3 / -1.1 | -3.9 / -3.6 / -2.4 / -2.6 | -5.0 / -4.0 / -1.8 / -0.8 |
| minutes MAE OFF / Δ (z) | 5.009 / -0.043 (-9.0) | 4.986 / -0.033 (-6.1) | 5.182 / -0.023 (-4.4) | 4.931 / -0.032 (-5.5) | 4.893 / -0.028 (-4.0) | 4.988 / -0.022 (-3.7) |
| minutes CRPS Δ (z) | -0.022 (-5.7) | -0.022 (-5.0) | -0.009 (-2.0) | -0.014 (-3.2) | -0.015 (-3.4) | -0.007 (-1.5) |
| minutes 80% cov OFF → ON | 0.846 → 0.830 | 0.851 → 0.837 | 0.869 → 0.855 | 0.870 → 0.856 | 0.849 → 0.845 | 0.845 → 0.836 |
| minutes PIT max dev | 0.0211 → 0.0181 | 0.0290 → 0.0262 | 0.0377 → 0.0318 | 0.0380 → 0.0330 | 0.0305 → 0.0316 | 0.0293 → 0.0287 |
| minutes cov80 star/starter/rotation/bench OFF→ON | 0.86→0.81 / 0.83→0.80 / 0.84→0.86 / 0.87→0.90 | 0.86→0.82 / 0.85→0.82 / 0.84→0.85 / 0.88→0.91 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.86→0.89 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.85→0.89 | 0.87→0.85 / 0.86→0.84 / 0.83→0.85 / 0.80→0.84 | 0.87→0.84 / 0.85→0.83 / 0.82→0.83 / 0.83→0.86 |
| DK bias star/starter/rotation/bench OFF→ON | -0.40→-0.27 / +0.17→-0.10 / +0.10→-0.03 / -0.50→-0.27 | -0.25→-0.16 / +0.33→+0.04 / +0.03→-0.13 / -0.13→+0.07 | +0.34→+0.42 / +0.50→+0.28 / -0.47→-0.59 / -0.69→-0.44 | +0.10→+0.19 / +0.62→+0.37 / -0.46→-0.58 / -0.70→-0.39 | -0.33→-0.33 / +0.38→+0.07 / -0.03→-0.10 / -0.18→+0.13 | -0.57→-0.45 / +0.60→+0.28 / +0.16→+0.06 / -0.12→+0.16 |
| bench demoted (n): P(play) bias, E[min] bias OFF→ON | (595) +0.067→+0.063, +0.90→+0.78 | (662) +0.080→+0.075, +0.87→+0.71 | (394) +0.015→+0.017, -0.39→-0.37 | (474) -0.003→+0.001, -0.68→-0.61 | (458) +0.095→+0.092, +1.54→+1.43 | (437) +0.049→+0.041, +1.07→+0.92 |
| bench demoted DK CRPS Δ (z) | -0.092 (-4.8) | -0.102 (-3.7) | +0.018 (+0.8) | +0.008 (+0.4) | -0.003 (-0.2) | +0.016 (+0.6) |

**+mf+mr, design C** (`PL_EDGES = PL_TIERS_C`):

| | 2023-24 | 2024-25 | live | live10 |
|---|---|---|---|---|
| rows | 22012 | 22193 | 15065 | 14951 |
| DK MAE OFF / Δ (z) | 8.010 / -0.0354 (-6.8) | 7.996 / -0.0342 (-5.7) | 8.096 / -0.0189 (-3.4) | 7.922 / -0.0224 (-3.6) |
| DK CRPS OFF / Δ (z) | 5.645 / -0.0224 (-6.4) | 5.645 / -0.0259 (-5.8) | 5.711 / -0.0081 (-2.1) | 5.606 / -0.0102 (-2.3) |
| 80% cov OFF → ON | 0.820 → 0.820 | 0.825 → 0.826 | 0.838 → 0.840 | 0.837 → 0.840 |
| DK PIT max dev | 0.0070 → 0.0077 | 0.0098 → 0.0105 | 0.0162 → 0.0162 | 0.0158 → 0.0159 |
| CRPS z star/starter/rotation/bench | -4.1 / -3.3 / -3.0 / -3.7 | -4.2 / -2.6 / -2.9 / -3.0 | -3.8 / -1.1 / +0.2 / -0.6 | -3.9 / -1.2 / -0.0 / -1.0 |
| minutes MAE OFF / Δ (z) | 5.009 / -0.043 (-9.0) | 4.986 / -0.033 (-6.1) | 5.182 / -0.023 (-4.4) | 4.931 / -0.032 (-5.5) |
| minutes CRPS Δ (z) | -0.022 (-5.7) | -0.022 (-5.0) | -0.009 (-2.0) | -0.014 (-3.2) |
| minutes 80% cov OFF → ON | 0.846 → 0.830 | 0.851 → 0.837 | 0.869 → 0.855 | 0.870 → 0.856 |
| minutes PIT max dev | 0.0211 → 0.0181 | 0.0290 → 0.0262 | 0.0377 → 0.0318 | 0.0380 → 0.0330 |
| minutes cov80 star/starter/rotation/bench OFF→ON | 0.86→0.81 / 0.83→0.80 / 0.84→0.86 / 0.87→0.90 | 0.86→0.82 / 0.85→0.82 / 0.84→0.85 / 0.88→0.91 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.86→0.89 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.85→0.89 |
| DK bias star/starter/rotation/bench OFF→ON | -0.40→-0.22 / +0.17→-0.13 / +0.10→-0.09 / -0.50→-0.31 | -0.25→-0.12 / +0.33→+0.02 / +0.03→-0.17 / -0.13→+0.04 | +0.34→+0.46 / +0.50→+0.28 / -0.47→-0.65 / -0.69→-0.46 | +0.10→+0.24 / +0.62→+0.37 / -0.46→-0.63 / -0.70→-0.42 |
| bench demoted (n): P(play) bias, E[min] bias OFF→ON | (595) +0.067→+0.063, +0.90→+0.78 | (662) +0.080→+0.075, +0.87→+0.71 | (394) +0.015→+0.017, -0.39→-0.37 | (474) -0.003→+0.001, -0.68→-0.61 |
| bench demoted DK CRPS Δ (z) | -0.090 (-4.7) | -0.100 (-3.6) | +0.022 (+0.9) | +0.011 (+0.5) |

Reading:
- **The minutes side was the problem, and the fix is broad.** +mf alone is better in every set: CRPS −6.1 / −5.2 SE on the harness, −2.6 / −3.7 SE live / live10, −3.7 / −3.0 SE on RotoGuru. Every tier is better in every set: worst tier z −0.7. DK MAE −0.013…−0.029, minutes MAE −0.022…−0.043.
- **Where the gain comes from.** Both parts contribute (minutes level, 2024-25, each part alone with the plain-plan logit: P(play) minutes CRPS −0.012 (−4.1 SE), noise −0.004 (−6.0 SE); the split was not run at the DK level). The noise part moves the tiers: stars' and starters' minutes get sharper and are no longer shifted to the bench by the lognormal's tier-dependent inflation.
  - Stars' DK bias moves −0.40 / −0.25 → −0.19 / −0.09 on the harness.
  - Starters' over-projection shrinks (+0.17 / +0.33 → −0.03 / +0.13).
  - Minutes PIT max deviation drops in five of six sets. Stars' and starters' minutes 80 % coverage falls from 0.83–0.88 to 0.80–0.85.
  - DK coverage moves by −0.4…0.0 pp.
- **The planned demotions themselves move less than the aggregate.**
  - All candidates (minutes level): their P(play) bias falls from +0.05 / +0.06 to +0.02 / +0.04, and promoted bench players' from −0.065 to ~0.
  - In the harness population (E[min] ≥ 10, i.e. P(play) ≳ 0.7 for a bench plan) the bias stays: +0.067 / +0.080 → +0.063 / +0.075. E[min] bias +0.90 / +0.87 → +0.78 / +0.71. DK CRPS −0.086 / −0.099 (−4.3 / −3.5 SE) on the harness, +0.02 (+1.1 / +0.9 SE) live, which is not a tier.
  - The high-P demoted bench is still over-projected. A linear logit with the plan cannot fully separate it.
- **+mr now composes.** On top of +mf, design B adds CRPS on every set (2023-24 −7.2, 2024-25 −6.4, live −2.9, live10 −3.0, RotoGuru −4.5 / −5.2 SE). Every tier is better (the +mr-alone bench costs are gone: bench −3.9 / −3.1 / −0.8 / −1.1 / −2.6 / −0.8). Coverage is unchanged (0.820 → 0.820).
  - Design C passes too (−6.4 / −5.8 / −2.1 / −2.3; rotation live +0.2 / −0.0) but is weaker than B everywhere, so B stays the switch's design.
- **Opener 2026-10-20, live path** (slate 154575 at 22:15Z, production 2026-27 bundle with the fix refit < 2026-27; one night):
  - Team minute sums are unchanged (BOS 236.8, DET 228.5, NYK 235.5, OKC 232.5, PHI 231.8, SAS 233.4).
  - Starters lose up to 1.5 min and bench P(play) rises (e.g. 0.58 → 0.84, played 25 min).
  - Max player move 2.7 DK. Minutes MAE vs the box 3.58 → 3.46, P(play) Brier 0.103 → 0.088 (106 players).

Bit-identical with the switch off: HEAD c45cb29 via `git archive` vs HEAD + this change, same DB copy and canonical bundles, `pathA-v0.3+p2`, switch unset / 0:
- Historical 2024-01-10 (avail, 4,000 sims): players 308 × 36, stats 3,696 × 12, 30 sim arrays (1.23M values) and meta all identical.
- Opener slate 154575 at 2026-10-20 22:15Z: players 106 × 38, stats 1,272 × 12, 9 sim arrays (2.65M values) and meta all identical.
- `research/p2_identity.py` (live / live10 / avail 2026-01-15, opener 2026-10-20, avail 2024-01-10): 5 / 5 hashes identical.
- OFF writes no `minfix` key, and no OFF code path reads it.
- Tests: `tests/test_proj_minutes_fix.py`, 13 tests: switch / versions compose, JSON identity, OFF P(plays) = the v1 logit exactly, ON P(plays) moves with the plan, both logit variants fit, unknown variant rejected, noise calibration stored, `_g_rows` = `g_value`, `_load_bundle_mf` refits the minutes only and caches, the debut driver code (old params read the camp sum) and the top-2 column. `tests/test_proj_*.py` + `test_pplay_q_refit.py`: 396 passed.

**Ship bar** (CRPS ≥ 2 SE better in 2023-24 and 2024-25, no tier worse by > 1 SE in any set, live slice not worse):
- **+mf: met.** −6.1 / −5.2 SE; worst tier z −0.7 in any of six sets; live −2.6 / −3.7 SE.
- **+mf+mr (B): met.** −7.2 / −6.4 SE; worst tier −0.8; live −2.9 / −3.0. (C: met, weaker.)

**+mf+pre** (`pathA-v0.3+pre+mf+p2`: the preseason columns, the debut reserve with the new driver below and the minutes fix, one refit < S):

| | 2023-24 | 2024-25 | live | live10 |
|---|---|---|---|---|
| rows | 22034 | 22207 | 15077 | 14959 |
| DK MAE OFF / Δ (z) | 8.010 / -0.0276 (-3.7) | 7.996 / -0.0328 (-4.0) | 8.097 / -0.0054 (-1.0) | 7.917 / -0.0123 (-2.1) |
| DK CRPS OFF / Δ (z) | 5.646 / -0.0188 (-4.4) | 5.646 / -0.0266 (-4.5) | 5.713 / -0.0064 (-1.7) | 5.603 / -0.0113 (-2.7) |
| 80% cov OFF → ON | 0.820 → 0.817 | 0.824 → 0.822 | 0.838 → 0.835 | 0.837 → 0.835 |
| DK PIT max dev | 0.0067 → 0.0090 | 0.0096 → 0.0082 | 0.0159 → 0.0144 | 0.0158 → 0.0148 |
| CRPS z star/starter/rotation/bench | -2.6 / -2.1 / -2.8 / -3.5 | -2.7 / -2.4 / -3.2 / -2.4 | -1.6 / -0.8 / -0.3 / -1.4 | -1.8 / -1.7 / -1.1 / -1.2 |
| minutes MAE OFF / Δ (z) | 5.014 / -0.045 (-6.6) | 4.986 / -0.045 (-6.4) | 5.185 / -0.020 (-3.0) | 4.930 / -0.029 (-4.2) |
| minutes CRPS Δ (z) | -0.022 (-4.4) | -0.031 (-5.1) | -0.008 (-1.6) | -0.015 (-2.8) |
| minutes 80% cov OFF → ON | 0.846 → 0.831 | 0.851 → 0.838 | 0.868 → 0.855 | 0.870 → 0.855 |
| minutes PIT max dev | 0.0209 → 0.0172 | 0.0294 → 0.0270 | 0.0374 → 0.0322 | 0.0380 → 0.0336 |
| minutes cov80 star/starter/rotation/bench OFF→ON | 0.86→0.81 / 0.83→0.80 / 0.84→0.86 / 0.87→0.91 | 0.86→0.82 / 0.85→0.82 / 0.84→0.85 / 0.88→0.91 | 0.88→0.85 / 0.86→0.83 / 0.87→0.88 / 0.85→0.89 | 0.88→0.84 / 0.86→0.83 / 0.87→0.88 / 0.85→0.89 |
| DK bias star/starter/rotation/bench OFF→ON | -0.40→-0.21 / +0.17→-0.04 / +0.11→-0.04 / -0.59→-0.40 | -0.25→-0.08 / +0.33→+0.13 / +0.02→-0.15 / -0.16→-0.01 | +0.34→+0.51 / +0.50→+0.36 / -0.47→-0.59 / -0.80→-0.63 | +0.10→+0.26 / +0.61→+0.47 / -0.46→-0.56 / -0.71→-0.50 |
| bench demoted (n): P(play) bias, E[min] bias OFF→ON | (614) +0.066→+0.065, +0.77→+0.78 | (671) +0.066→+0.063, +0.77→+0.71 | (398) +0.012→+0.015, -0.47→-0.38 | (478) -0.007→-0.001, -0.69→-0.57 |
| bench demoted DK CRPS Δ (z) | -0.097 (-3.6) | -0.086 (-2.8) | +0.002 (+0.1) | +0.021 (+0.8) |

First 15 team games (the preseason arm's own window), ON − OFF: minutes MAE −0.056 (−2.3 SE) / −0.073 (−3.8 SE), DK CRPS −0.033 (−2.1 SE) / −0.058 (−3.2 SE) in 2023-24 / 2024-25. +mf alone gives −0.041 / −0.033 and −0.021 / −0.027. With the old debut driver the numbers were within 0.005 of these (2026-27 openers are not in the harness).
- +mf+pre is weaker than +mf over the full season (CRPS −4.4 / −4.5 vs −6.1 / −5.2 SE; live −1.7 vs −2.6).
- It is stronger in the first 15 games.
- The +pre-alone star-tier cost (+1.4 / +2.6 SE) is gone: stars −2.6 / −2.7 SE.

**Debut reserve re-check (item 3; under +pre).** `research/minutes_fix_eval.py debut`: every team-game of a season's first 15, debut minutes = minutes of players with no box row before the date. 2026-27 uses the Mac prod DB read-only (backup after the 03:00 postgame: the 28 openers of 10-20 and 10-21). Leave-one-season-out, each fit on the 5 seasons before; MAE of team debut minutes at game 0 (bias in brackets):

| design | 2019-20 … 2025-26 (mean of 7) | 2026-27 openers (28) |
|---|---|---|
| constant by stage | 15.2 | 14.2 (−0.6) |
| first design: stage × camp new players' minutes (sum), cap p95 | 11.2 | 12.2 (+0.1) |
| stage × sum, cap p50 | 12.6 | 12.0 (−0.8) |
| **stage × the two largest new players' minutes, cap = training max (shipped in +pre)** | **11.0** | **11.2 (−0.7)** |
| + stage × roster slack (240 − returning EWMAs) | 11.0 | 11.2 (−0.7) |
| stage × number of new players ≥ 15 mpg | 11.5 | 10.0 (−1.5) |

- **The opener's "over-reservation" was mostly the sample.** The 6 teams of 10-20 had 2.7 debut minutes on average; the 28 openers have 14.4, in line with 11–17 in 2019-26. The first design is unbiased on them (+0.1).
- **The input range was the real fragility.** The sum grows with camp invitees who never play: OKC 126 vs a training p95 of 70 and a max of ~95 (123 in 2016-17). Two players cannot grow that way: the 2026-27 maximum is 51.4 vs a training maximum of 52.5. The top-2 driver also has a slightly lower mean error over 2019-26 (better in 4 of 7 seasons).
- `debut_reserve` now reads a driver code stored after the cap (2 = top-2), so params fit with the first design read unchanged. The reserve is still clipped to [0, 40].
- **Still a miss: deep contenders.** OKC (two lottery picks, 13 debut minutes) is predicted 34. On the opener's live path, +pre+mf takes 14 / 32 / 19 / 14 team minutes from BOS / OKC / PHI / SAS (debutants played 0 / 13 / 0 / 3). DET and NYK are unchanged.
  - A roster-depth term (240 − returning EWMAs) did not help leave-one-season-out.
  - The fix for the case is to project rookies from their preseason role. `candidate_panel` drops players without box history, so they live in the reserve: the open item from the preseason block.

**Ship bar** for **+mf+pre: met** (−4.4 / −4.5 SE; every tier ≤ −0.3 SE; live −1.7 / −2.7 SE). Its debut reserve is calibrated on average but still misses deep rosters at game 0.

The switch stays OFF in every combination; shipping is the owner's call. If it ships:
- +mf+mr (B) is the strongest arm.
- The minutes-ratio table's training plans should then be refit from the +mf engine (planned rows here are the OFF engine's).
- The +p2 / +pq live paths' listed and fringe P(play) are unchanged by +mf (it replaces only the engine's P(plays ∣ available) of unlisted roster players).

### Phase 2F: feature program (owner request 2026-10-19: use the depth of the data)

v0 projects per-minute rates from the player's own history plus a position × tier prior; opponent, matchup, tracking, shot-quality and referee data are not used yet (the environment enters only through the market total). Phase 2F adds a **learned context layer**: per stat, rate = EB rate × exp(f(features)), with f a LightGBM fit on seasons < S (Poisson/NB objective, minutes-weighted, pregame features only), on top of minutes v1. **Every group ships only if it improves NB log-lik / DK CRPS by > 1 SE on 2023-24 and 2024-25 in the T2.1 harness**; results go in an ablation table (kept or dropped, with numbers).

| ID | Group | Data | Notes |
|---|---|---|---|
| F0 ★ | feature store + context-layer harness + ablation protocol | `proj/features/`, T2.1 | strictly pre-tip, walk-forward; leak tests |
| F1 ★ | **opponent availability**: absent opposing rim protector / rebounder / primary perimeter defender, opponent depth | availability scenarios, box | evaluated per sim scenario |
| F2 ★ | **opponent defense per stat × position group** (shrunk, recency-weighted): reb / 3PA / ast / tov / FTA / blk allowed, pace beyond the total | box, PBP, team adv | DvP is noisy: heavy shrinkage |
| F3 | tracking role signals: rebound chances, potential assists, touches, contested share | `player_tracking` (2019+) | faster-stabilising rates |
| F4 | shot quality / shooting luck: xFG from PBP shot location, type, distance | cdn PBP 2019+ | regress 3P% luck |
| F5 | matchups: likely primary defender (historical assignment shares × tonight's roster), defender quality (own shrunk matchup / defended-FG% vs expected) | `player_matchups` 2.17M rows (2017+), `pt_defend` | expect small; heavy shrinkage |
| F6 | context: home/away, travel and time zones, altitude, season stage | schedule | |
| F7 | referees: crew foul / FTA tendencies | referee assignments (live), sportsdataverse officials (history) | |
| F8 | player vs this opponent / coach history | box | expected noise; tested and reported |
| F9 | player prop lines (market) as features / blend | `prop_snapshots` (live since 2026-10-19) | after ≥ 3 weeks of data |

**P + F2 live wiring, ON (`pathA-v0.3`) vs OFF (`pathA-v0.2`), 2026-10-20.** Walk-forward (every season's bundle and layer are fit on seasons < S). Avail protocol, closing lines, 4,000 sims, every date. ON and OFF use the same base bundle and common random numbers, so the rows are identical and the differences are paired. SEs are clustered by date. Gain > 0 means ON is better. Source: Mac scratch `courtside-compute/cx`; reports copied to `var/reports/context_live/`.

| Run | n rows / dates | DK MAE OFF → ON (gain, SE) | DK CRPS OFF → ON (gain, SE) | 80% cov OFF → ON | star bias OFF → ON |
|---|---|---|---|---|---|
| T2.1 harness 2023-24 | 21,767 / 160 | 8.058 → 8.025 (+0.033, 0.0038) | 5.670 → 5.652 (+0.0175, 0.0021) | 0.819 → 0.820 | −1.38 → −0.90 |
| T2.1 harness 2024-25 | 21,865 / 162 | 8.023 → 7.995 (+0.028, 0.0043) | 5.659 → 5.646 (+0.0130, 0.0023) | 0.823 → 0.824 | −1.13 → −0.60 |
| T2.1 harness 2019-20 (RotoGuru) | 18,681 / 149 | 7.891 → 7.865 (+0.026, 0.0040) | 5.567 → 5.555 (+0.0125, 0.0023) | 0.820 → 0.821 | −1.27 → −0.81 |
| T2.1 harness 2020-21 (RotoGuru) | 19,079 / 139 | 7.836 → 7.798 (+0.038, 0.0040) | 5.533 → 5.515 (+0.0181, 0.0024) | 0.825 → 0.827 | −1.35 → −0.85 |
| eval-lite 2024-25 (T1.0 part 1) | 21,882 / 162 | 8.021 → 7.995 (+0.026, 0.0044) | 5.661 → 5.650 (+0.0110, 0.0026) | 0.823 → 0.824 | −1.13 → −0.60 |

| Gate line | OFF | ON |
|---|---|---|
| G1 2024-25 DK MAE gain vs season-to-date (≥ 3%) | 7.36% (SE 0.52) PASS | 7.68% (SE 0.52) PASS |
| G3 2024-25 80% coverage (0.76–0.84) | 0.823 PASS | 0.824 PASS |
| G4 2024-25 minutes MAE (≤ 4.8) | 4.422 PASS | 4.422 PASS (minutes untouched) |
| G2 2019-20 / 2020-21 player MAE gain vs best salary-implied | 3.36% / 3.58% PASS | 3.68% / 4.05% PASS |
| G2b cash lineup minus salary-implied lineup, 2019-20 / 2020-21 | +67.2 / +90.0 pts | +68.5 / +91.9 pts |
| T2.1 cash lineup ON − OFF (paired), 2019-20 / 2020-21 | | +1.32 (SE 1.61) / +1.85 (SE 1.62); wins 32% / 33%, ties 36% / 35% |
| T2.1 cash lineup − FPPG-optimizer lineup, 2019-20 / 2020-21 | +18.9 (SE 3.1) / +27.9 (SE 3.5) | +20.2 (SE 3.2) / +29.8 (SE 3.2) |
| T2.1 Top-N mean ON − OFF, 2019-20 / 2020-21 | | +0.50 (SE 0.61) / +0.61 (SE 0.68) |
| Hindsight ratio (cash), 2019-20 / 2020-21 | 0.693 / 0.687 | 0.696 / 0.692 |

**Live path (opener, slate 154575 at 2026-10-20 12:00Z).**
- Team minutes and team points are identical ON and OFF, because the layer touches only per-minute rates and points are still reconciled to the implied total.
- Team DK changes: BOS −2.0, DET −3.4, NYK +1.4, PHI −6.9, OKC −0.8, SAS +0.2 (−3%…+1%).
- Top-12 moves stay within ±2.1 DK (SGA +2.1, Wembanyama +1.2, Maxey −1.0, LeBron −1.1).
- Multipliers stay within [0.8, 1.25]. Only fg3m (5.7%) and tov (4.7%) touch the clip. The DK-rate multiplier ranges 0.93–1.04.
- `live-path-eval` with the switch ON passes (minutes only, identical by construction).
- Verdict: ship after the opener is graded.

**Phase 2F, second ablation: F3 tracking, F5 matchups, F6 schedule on top of P+F2 (2026-10-20).** Code: `proj/features/{tracking,matchups,schedule}.py`, `features.extend_frame` / `features.LiveExtras`, `context.run_ablation` (`research/context_ablation2.py`; `--staged` = F5 / F6 on top of P+F2+F3). Each layer is fit walk-forward on the 6 regular seasons < S, with one LightGBM thread. The per-minute metrics are NB log-lik per row (sum of the 7 DK stats) and DK CRPS / MAE given actual minutes, on all regular-season rows of S (n ≈ 26k). The SE is the larger of the two-way (game × player-season) and date-clustered SEs. Gain > 0 means the first arm is better. Reports: `var/reports/context_ablation2/` (Mac scratch `courtside-compute/f356`).

| Comparison | Season | NB LL/row gain (SE, z) | DK CRPS gain (SE, z) | DK MAE gain (SE, z) |
|---|---|---|---|---|
| **+F3 vs P+F2** | 2022-23 | +0.00550 (0.00071, 7.7) | +0.0031 (0.0011, 2.7) | +0.0059 (0.0019, 3.0) |
| | 2023-24 | +0.00645 (0.00074, 8.7) | +0.0046 (0.0012, 3.9) | +0.0057 (0.0020, 2.9) |
| | 2024-25 | +0.00785 (0.00082, 9.6) | +0.0059 (0.0014, 4.2) | +0.0087 (0.0025, 3.6) |
| | 2025-26 | +0.00885 (0.00086, 10.3) | +0.0047 (0.0014, 3.4) | +0.0076 (0.0024, 3.1) |
| +F5 vs P+F2 (alone) | 2023-24 / 2024-25 | +0.00073 (1.7) / +0.00078 (1.9) | +0.0010 (1.2) / +0.0011 (1.4) | +0.0017 (1.1) / +0.0014 (1.0) |
| +F6 vs P+F2 (alone) | 2023-24 / 2024-25 | +0.00037 (1.3) / +0.00066 (2.3) | +0.0015 (2.6) / +0.0009 (1.5) | +0.0028 (2.9) / +0.0020 (2.0) |
| **+F6 vs P+F2+F3** | 2022-23 | +0.00032 (0.00034, 1.0) | +0.0009 (0.0006, 1.7) | +0.0021 (0.0010, 2.1) |
| | 2023-24 | +0.00083 (0.00030, 2.7) | +0.0018 (0.0006, 2.8) | +0.0030 (0.0011, 2.8) |
| | 2024-25 | +0.00093 (0.00030, 3.1) | +0.0007 (0.0006, 1.2) | +0.0019 (0.0011, 1.7) |
| | 2025-26 | +0.00086 (0.00036, 2.4) | +0.0007 (0.0006, 1.3) | +0.0015 (0.0010, 1.5) |
| +F5 vs P+F2+F3+F6 | 2022-23 | +0.00209 (0.00037, 5.6) | +0.0026 (0.0007, 3.8) | +0.0035 (0.0012, 3.0) |
| | 2023-24 | +0.00056 (0.00039, 1.4) | +0.0009 (0.0008, 1.1) | +0.0018 (0.0013, 1.3) |
| | 2024-25 | +0.00079 (0.00038, 2.0) | +0.0009 (0.0008, 1.2) | +0.0015 (0.0014, 1.1) |
| | 2025-26 | +0.00102 (0.00039, 2.6) | +0.0017 (0.0008, 2.2) | +0.0013 (0.0013, 1.0) |
| all (F3+F5+F6) vs P+F2 | 2022-23 … 2025-26 | +0.0079 / +0.0078 / +0.0096 / +0.0107 (9.1…11.7) | +0.0066 / +0.0073 / +0.0076 / +0.0071 (4.7…5.0) | +0.0114 / +0.0106 / +0.0122 / +0.0104 (4.0…4.8) |

End to end (T2.1 harness, avail protocol, closing lines, 4,000 sims, every date; each arm's bundle = the OFF bundle + its layer fit on seasons < S, via the pipeline switch; paired against the P+F2 ON runs with common random numbers; rows identical, same expected minutes; SE clustered by date):

| Arm vs P+F2 ON (`pathA-v0.3`) | Season | DK MAE (gain, SE, z) | DK CRPS (gain, SE, z) | 80% cov | bias (all / star tier) | Verdict |
|---|---|---|---|---|---|---|
| +F3 | 2023-24 | 8.0249 → 8.0225 (+0.0025, 0.0021, 1.2) | 5.6520 → 5.6514 (+0.0007, 0.0012, 0.6) | 0.820 → 0.820 | −0.08 → −0.12 / −0.92 → −1.01 | below bar alone |
| | 2024-25 | 7.9950 → 7.9902 (+0.0048, 0.0022, 2.1) | 5.6464 → 5.6425 (+0.0039, 0.0013, 2.9) | 0.824 → 0.824 | +0.05 → +0.03 / −0.61 → −0.71 | |
| **+F3+F6** | 2023-24 | 8.0249 → 8.0207 (+0.0042, 0.0023, 1.9) | 5.6520 → 5.6503 (+0.0017, 0.0013, 1.3) | 0.820 → 0.820 | −0.08 → −0.11 / −0.92 → −0.98 | **kept** |
| | 2024-25 | 7.9950 → 7.9897 (+0.0052, 0.0022, 2.3) | 5.6464 → 5.6423 (+0.0041, 0.0014, 3.0) | 0.824 → 0.824 | +0.05 → +0.03 / −0.61 → −0.71 | |
| +F3+F5+F6 vs +F3+F6 | 2023-24 | −0.0006 (0.0012, −0.5) | +0.0000 (0.0007, 0.0) | 0.820 → 0.821 | | **F5 dropped** |
| | 2024-25 | −0.0002 (0.0012, −0.2) | −0.0002 (0.0007, −0.3) | 0.824 → 0.825 | | |

F3+F6 vs OFF (`pathA-v0.2`): CRPS +0.0192 (7.8 SE) / +0.0171 (6.9 SE), MAE +0.0372 / +0.0335. F6 on top of F3 end to end: CRPS +0.0010 (1.9 SE) / +0.0002 (0.4 SE).

**Per stat** (NB LL/row ×10⁻⁴ of the full F3+F5+F6 vs P+F2, z in brackets): assists +20…+26 (5.0–6.4) and steals +7…+26 (3.0–7.2) gain most, then 3PM +11…+21 (3.5–5.7), rebounds +6…+17 (1.9–4.8), points +6…+12 (2.0–4.3) and turnovers +4…+9 (1.9–4.5). Blocks are small (+4…+5, 2.7–3.1) and slightly negative in 2023-24 (−1.7, −0.7). Where F3 acts (split gain): the opportunity conversions do most of the work. These are AST per pass, DREB / OREB per chance, uncontested FG% (shooting-luck regression, used by both points and 3PM), and STL per deflection. Next come the tracking-implied-vs-EB gaps for steals, rebounds and assists, then touches for turnovers and contested 2PT shots for blocks. F6 acts almost entirely through season stage (rebounds, assists, blocks, turnovers); rest / travel / altitude barely split. F5's signal is the matchup-weighted 3PA / AST / TOV quality of the defenders faced and the attacking tendencies of the players a defender guards (steals, blocks).

**Leak tests** (`tests/test_proj_features2.py`, `tests/test_proj_context_ext.py`):
- L1 scramble per group (F3, F5): every tracking / hustle / matchup / box stat on or after the cut is re-randomised; features up to the cut are identical and later ones change.
- L5 truncation (F3, F5): every row after the cut is deleted, and features up to it are identical.
- F6: only earlier games and the game's own date / site are used. Dropping later games and zeroing results changes nothing, and the per-day hook equals the table with the day's games missing from `team_games` (the live case).
- Live hooks: `LiveExtras.day` reproduces the frame's F3 / F5 columns exactly. In the pipeline with ON + ext, deleting the day's and later box / tracking / hustle / matchup rows leaves the sims identical.
- Walk-forward: the extended bundle is fit on seasons < S only.

**Live path (opener, slate 154575 at 2026-10-20 12:00Z, ON + ext vs ON).**
- No NaN in the 46 extension columns for any of the 106 slate players, including 29 traded players.
- Every 2026-27 roster player of the six opener teams was checked through the same hook, including 16 rookies / camp players with no NBA minutes. They get the group prior (`t3_ln_min` 0) and finite multipliers inside the clip.
- Team DK moves −1.4…+1.5 and the largest player move is 1.0 DK (SGA −1.0, Pritchard −0.8, George +0.8).
- ON P+F2 with this code is bit-identical to the committed P+F2 wiring on 2025-01-15 (11 games, 1.31M sim values, same seed).

Not done / caveats:
- Potential assists are not in the source (stats.nba.com leaves `matchupPotentialAssists` at 0). Drives and contested rebounds are not in the box-score tracking feed.
- The F3 / F5 / F6 shrinkage constants were set by hand, not tuned.
- F3 adds 0.06–0.10 DK to the star tier's under-projection.
- The RotoGuru lineup seasons (2019-20 / 2020-21) were not rerun: they have no tracking history to train on.

**Phase 2F, third ablation: F4 shot quality and shooting luck (xFG) on top of P+F2+F3+F6 (2026-10-20). Verdict: dropped.** Real per minute, nothing end to end.

Code:
- `proj/features/shots.py` (shot store, xFG model, F4 features), registered as group F4 (`features.EXTENSION_GROUPS`, `extend_frame`, `LiveExtras(..., xfg=)`).
- `context.VARIANTS_4`; `research/xfg_quality.py`, `research/f4_luck_diag.py`.
- Not wired into the pipeline: `pipeline.py` is unchanged.

Reports are in `var/reports/f4/` (Mac scratch `courtside-compute/f4`).

*Shot store.* Every FGA and FT of the archived cdn play-by-play, 2019-20 … 2025-26: 8,647 games and 1.92M shots. 2019-20 has only 874 games archived. The store records:
- shooter, made, legacy x / y, distance, shot type, 14 descriptor flags (driving, pull-up, step-back, …), second-chance / fast-break / off-turnover, period, seconds left, score before the shot;
- assisted / blocked, which are outcome-dependent and kept out of the xFG model.

It is cached per season as parquet under `<data>/proj/features/shots/`. Updates are incremental: only new archive files are parsed.

*xFG model.* LightGBM binary on the pre-shot features, fit on the shots of seasons < S, with per-kind Platt scaling on the held-out last training season. It is scored on every FGA of S against a distance-only baseline (make rate per 2 / 3 × 1-ft bin, same training seasons):

| Test season (fit on) | FGA | Log-loss xFG vs distance (gain) | 2PT gain | 3PT gain | Brier | ECE | Calibration slope all / 3PT |
|---|---|---|---|---|---|---|---|
| 2022-23 (2019-20…2021-22) | 232,918 | 0.6345 vs 0.6521 (+0.0175) | +0.0282 | +0.0007 | 0.2232 vs 0.2301 | 0.009 vs 0.013 | 1.00 / 0.88 |
| 2023-24 (…2022-23) | 233,597 | 0.6357 vs 0.6527 (+0.0171) | +0.0275 | +0.0010 | 0.2237 vs 0.2305 | 0.008 vs 0.009 | 0.98 / 1.16 |
| 2024-25 (…2023-24) | 234,975 | 0.6361 vs 0.6523 (+0.0162) | +0.0276 | +0.0005 | 0.2238 vs 0.2303 | 0.005 vs 0.008 | 0.99 / 0.95 |
| 2025-26 (…2024-25) | 234,671 | 0.6361 vs 0.6520 (+0.0159) | +0.0263 | +0.0011 | 0.2237 vs 0.2299 | 0.005 vs 0.009 | 1.02 / 0.95 |

Distance and shot type carry 81% of the split gain; seconds left, location and the cutting / tip / finger-roll flags carry the rest. **3s are almost unmodellable** without defender distance: the gain over distance-only is 0.001. Before Platt scaling, the 3PT calibration slope was 0.79–0.90.

*Features* (pts and fg3m only; strictly pre-tip, `common.ew_asof`). Unless noted, they are shrunk to the position-group prior with a 180-day half-life.
- Shot quality: xFG% on 2s / 3s and expected points per FGA vs the league.
- Shot diet: rim / short / mid / corner-3 / above-the-break-3 shares, 3PA rate, FT rate, FT%, and the assisted share of made 2s / 3s.
- Long-memory shooting over expected (365-day half-life, net of the league drift).
- Opponent xFG and luck allowed.
- **The share of the EB rate that is luck**, `f4_lk_pts` / `f4_lk_fg3m`: made-minus-xFG points / 3PM accumulated with the EB state's own decay, divided by (n_s + κ_s) and r_s. These are monotone −1.

*The luck premise holds.* Two checks support it:
- **Per player-game.** In 2024-25, 3PT shooting-over-expected in a 120-day window (the fg3m EB window) carries only **0.39** of itself into the next game. At a 365-day memory it carries 0.56; for 2s, 0.67 and 0.76. So about 60% of a hot 3PT stretch is luck.
- **On the EB base.** Actual / expected fg3m falls monotonically across deciles of `f4_lk_fg3m`: from 1.16 (cold) to 0.93 (hot) in 2024-25, and from 1.13 to 0.95 in 2023-24. For points the range is 1.02 → 0.98 and 1.04 → 0.97.
  - The P+F2+F3+F6 layer already removes about half of this. F3's uncontested / contested FG% conversions are a luck regression too: fg3m runs 1.08 → 0.96.
  - Adding F4 removes most of the rest: 1.02 → 0.98 in 2024-25, but 1.02 → 1.02 in 2023-24, where the top decile is slightly over-corrected.

*Per minute* (frame protocol as in the second ablation; SE = the larger of the two-way and date-clustered SEs):

| Comparison | Season | NB LL/row gain (SE, z) | DK CRPS gain (SE, z) | DK MAE gain (SE, z) | LL by stat ×10⁻⁴: pts / fg3m (z) |
|---|---|---|---|---|---|
| **+F4 vs P+F2+F3+F6** | 2022-23 | +0.00126 (0.00034, 3.7) | +0.0017 (0.0007, 2.4) | +0.0031 (0.0012, 2.6) | +6 (3.0) / +7 (3.5) |
| | 2023-24 | +0.00093 (0.00037, 2.5) | +0.0019 (0.0008, 2.5) | +0.0039 (0.0014, 2.8) | −1 (−0.3) / +11 (5.5) |
| | 2024-25 | +0.00121 (0.00041, 3.0) | +0.0015 (0.0008, 1.9) | +0.0025 (0.0013, 1.9) | +2 (0.7) / +10 (3.3) |
| | 2025-26 | +0.00178 (0.00038, 4.7) | +0.0030 (0.0008, 4.0) | +0.0047 (0.0012, 3.9) | +10 (5.0) / +8 (2.7) |
| +F4 vs P+F2 (alone) | 2022-23 … 2025-26 | +0.0029 / +0.0025 / +0.0034 / +0.0038 (4.7…5.8) | +0.0020 / +0.0041 / +0.0030 / +0.0046 (2.1…4.2) | +0.0037 / +0.0054 / +0.0045 / +0.0075 (2.2…3.8) | pts +4…+16, fg3m +21…+24 |
| +F4 vs P+F2+F3+F6, multipliers **reconciled per team-game** (points rescaled to the team's expected points, others by c^0.5, as the pipeline does) | 2023-24 / 2024-25 | +0.00089 (2.4) / +0.00108 (2.6) | +0.0016 (2.1) / +0.0005 (0.7) | +0.0030 (2.3) / +0.0010 (0.8) | pts −1 / +2, fg3m +10 / +9 |

Alone on P+F2, F4 is about 2.5× its gain on top of F3. The overlap is F3's FG% conversions.

*End to end* (T2.1 harness: avail protocol, closing lines, 4,000 sims, every date). The bundle is the OFF bundle + the P+F2+F3+F6+F4 layer fit on seasons < S, with the xFG model fit on seasons < S. Runs are paired with common random numbers against the P+F2+F3+F6 runs: the rows are identical and so are the expected minutes. SE is clustered by date.

| Arm | Season | DK MAE (gain, SE, z) | DK CRPS (gain, SE, z) | 80% cov | bias (all / star tier) | Verdict |
|---|---|---|---|---|---|---|
| +F4 vs P+F2+F3+F6 (`pathA-v0.4`) | 2023-24 | 8.0207 → 8.0215 (−0.0008, 0.0013, −0.6) | 5.6503 → 5.6507 (−0.0004, 0.0008, −0.5) | 0.820 → 0.820 | −0.110 → −0.107 / −0.96 → −0.94 | **dropped** |
| | 2024-25 | 7.9897 → 7.9896 (+0.0001, 0.0013, +0.1) | 5.6423 → 5.6426 (−0.0003, 0.0008, −0.4) | 0.824 → 0.824 | +0.030 → +0.036 / −0.72 → −0.72 | |
| (P+F2+F3+F6+F4 vs P+F2) | 2023-24 / 2024-25 | +0.0034 (1.3) / +0.0054 (2.1) | +0.0013 (0.9) / +0.0038 (2.6) | | | below +F3+F6 alone (1.9 / 2.3, 1.3 / 3.0) |

Tiers vs P+F2+F3+F6: the bench MAE is −0.0032 (SE 0.0017) in 2023-24. Role and star are within 1 SE both seasons.

**Wiring check.** On 2025-01-15 the pipeline's F4 columns (bundle layer + `LiveExtras` with the bundle's xFG model) equal the frame's for every played player-game (max |diff| 2.6e-13, no NaN mismatch). So the null end-to-end result is not a wiring artefact.

**Existing arms unchanged.** OFF `pathA-v0.2`, P+F2 `pathA-v0.3` and P+F2+F3+F6 `pathA-v0.4` were run with HEAD (93f7a5b+) and with this code on 2025-01-15: 11 games, 4,000 sims, same seed. Each arm is bit-identical: players, stats and 1.31M sim values. ON and ext still differ from each other.

**Why it doesn't carry.**
- **Points.** The pipeline rescales team points to the market's implied total, so a within-team luck regression mostly cancels. Reconciled, the points LL gain is about 0, and in 2024-25 the CRPS gain falls from 1.9 to 0.7 SE.
- **3PM.** The 3PM gain survives reconciliation, but 3PM carries only +0.5 DK per make. Per player the multiplier moves are about 1–3% of one stat, below what the projected-minutes and simulation noise of the harness can resolve.

**Leak tests** (`tests/test_proj_shots.py`):
- L1 scramble: every shot (location, type, result) and box stat on or after the cut is re-randomised; F4 up to the cut is identical and later rows change.
- L5 truncation: games after the cut are deleted, so their shots drop out through the team_games join; F4 up to the cut is identical.
- The xFG model is walk-forward (seasons < S).
- `LiveExtras.day` reproduces the frame's F4 columns and the row-level luck shares exactly.
- Before the shot era the features are NaN; players without shots sit at the prior; a planted hot shooter carries positive luck.
- Parser dialects: "Jump Shot" / "jumpshot", "step back" / "stepback", ": Made", legacy coordinates, score before the shot.
- The store is incremental.

**Not done / caveats:**
- No defender distance (stats.nba.com shot dashboards are season-level only), so 3PT xFG is close to league average for everyone. Tracking's contested / uncontested split (F3) is the better 3PT-quality signal we have.
- The shrinkage constants and half-lives were set by hand (the skill K was checked against next-game persistence), not tuned.
- 2019-20 is missing about 210 archived games.
- If F4 is revisited, the open lever is **3PM-only** (its gain survives reconciliation), judged end to end in a season where DK 3PM exposure matters (Showdown / GPP), not more per-minute work on points.

### Phase 2.5: contest builder, GPP, late swap, Showdown (Nov 15 – 21)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T2.11 ★ O | ownership proxy + field proxy (3 strengths) | `proj/dfs/field.py` | → field samples | Σ own = 800%; field lineups valid; ≤ 20 s |
| T2.12 ★ O | GPP portfolio: candidates, payout ladder, greedy, caps, dupes | `proj/dfs/portfolio.py` | → | N = 20 ≤ 120 s; caps exact; greedy portfolio beats "top N by individual E[payout]" on realised proxy payout over 100 backtest slates (paired) |
| T2.13 ★ S | cash P(score ≥ c_s) | `proj/dfs/cash.py` | → | P(cash) calibration on the backtest proxy line within ±5 pp |
| T2.14 ∥ S | late-swap re-solver + edit CSV + UI action | `proj/dfs/swap.py` | → | locked players untouched; ≤ 30 s |
| T2.15 ∥ S | Showdown (CPT binary) | `proj/dfs/showdown.py` | → | pydfs equal on 30 Showdown slates |
| T2.16 ∥ S | contest builder UI + guidance text | `templates/dfs_contest.html`, `web/dfs_view.py` | → | labels present ("simulated, not real", "proxy"); screenshots |
| T2.17 S Mac | backtest extension: field-proxy percentiles, `fppg` optimizer, cash proxy, lineup calibration | `proj/backtest.py` | → | §7.3 table filled |

### Phase 3: DFS MLOps and calibration (pulled forward to Oct 20 – 24 by the owner, 2026-10-19)

**Owner request (2026-10-19):** track every DFS build and result from opening night, keep all data locally, and use real contest results to learn ownership. Order:
1. **T3.0 nightly DFS grading** (new, before the first games are final on 2026-10-21): `proj/dfs/grade.py`, a postgame step. It writes `player_projection_grades` and `dfs_lineup_grades` (migration 12): actual vs projected, hindsight-optimal ratio, random-lineup percentile, and baselines (DK FPPG optimizer, salary-implied). It also estimates the cash line from uploaded standings when present, otherwise from the payout table plus a labelled proxy.
2. **T3.1** registry specs, `dfs_metric_snapshots` (7d / 30d / season), alerts, `/health` DFS tab.
3. **T3.2** upload page for DK standings CSVs ("Export lineups to CSV" after a contest: every entry's rank, points and lineup, plus %Drafted and FPTS per player) and the owner's own entries (`dfs_uploads`, `dfs_entries`, `dfs_contest_results`, `dfs_contest_ownership`). Real cash lines, top scores and ownership feed the ownership model (T2.11). Simulated and real results are always shown separately.
4. **Retention:** simulation files are kept indefinitely (no 8-day prune); raw DK, report and line archives were already permanent.

| ID | Task | Owned files | Acceptance |
|---|---|---|---|
| T3.1 S | registry specs (`player_minutes`, `player_rates`, `player_dist`, `dfs_engine`), gates, `dfs_metric_snapshots`, alerts, `/health` DFS tab | `mlops/dfs_monitor.py`, `web/health.py` (tab) | replay of the season to date produces snapshots; alerts dedupe |
| T3.2 S | standings-CSV importer + ownership-proxy calibration | `proj/dfs/standings.py` | importer exact on 3 sample CSVs; calibration report once ≥ 10 slates |
| T3.3 H | guidance polish + weekly DFS report | `proj/reports.py` | clustered SEs; proxy labels |

**Alerts:**
- minutes bias on projected starters > ±1.0 min (last 300);
- DK FP 80% coverage outside 0.80 ± 3 SE;
- FPPG reconciliation failure;
- a missed lock run;
- ID match < 98%;
- reconciliation gap > 0.5;
- live minutes MAE > live-like + 0.5.

### Phase 4: why engine and deferred items (Dec 8 – 23)

| ID | Task | Acceptance |
|---|---|---|
| T4.1 O | xFG (with S5's revised criteria) | per v1 §3.2, criteria rewritten at start |
| T4.2 O | full post-game decomposition | exact sums |
| T4.3 O | pre-game Shapley (S6 per-coalition reconciliation) | exact sums; ≤ 10 s per slate |
| T4.4 O | primary defender, matchup ingestion | ships only with a CRPS gain |
| T4.5 S | referee crew test | written verdict |
| T4.6 S | player page season views | screenshots |

**Critical path:** T0.0 → T0.5/T0.6 (and T0.2 → T1.0) → T1.1 → T1.2 → T1.4 → T1.5 → T1.6 (live by Oct 27) → T2.1 → T2.2 → T2.10 → T2.11 → T2.12. T2.5 → T2.6 is a parallel upgrade track that is off the GPP critical path, because of the copula.

**Machine plan:**
- On geekom1, run at most 3 agents: the critical-path agent, one ∥ coding agent, and one H/data agent.
- On the Mac, run the backtests, SMM, RotoGuru parse, stints and T2.7, one heavy job at a time.

## 9. Risks and open questions

**Risks.**
- **ToS.**
  - DK forbids automated collection "for any purpose". Our capture (~100 requests a day, no login) is still automated, and the owner accepted the risk. If DK blocks or objects, fall back to the lobby's manual salary CSV.
  - RotoGuru: polite, one-time, personal use, credited.
  - NBA, ESPN and Action Network data is personal-use only.
  - Entry is never automated.
- **Single machine.** Lock runs need the Mac awake. A missed run alerts. Fallback: `courtside dfs run --kind manual` on geekom1, which fetches its own draftables and stores locally.
- **Ownership proxy limits.** Without uploads there is no truth, so leverage and dupe numbers can be badly wrong. They are shown as bands, labelled, and ROI is shown as a range.
- **RotoGuru main-slate ambiguity.** The game set is inferred, positions are single, and 2019-20 includes the bubble. Report both slate definitions, and don't carry historical cash rates to live.
- **Backtest optimism** (known availability), and **early season** (prior-heavy rates, noisy preseason roles). Show "small sample" and widen the intervals.
- **Over-engineering.** Correlations are about 0.03. Path B ships for DFS only through G2.
- **DK rule changes.** Rules are read every run; scoring is checked weekly.

**Open questions for the owner (with proposals).**
1. **Top N defaults:** N = 10, min-different d = 1?
2. **Contest selection rule** for the payout archive (§3.2), with a cap of 40 a day. OK?
3. **Will you upload standings CSVs** from contests you enter? They are the only real ownership and cash-line truth (§7.4).
4. **Ownership display:** bands only, or percentages after calibration?
5. **Late-news alert channel:** page banner only (proposed), or a push notification?
6. **Tonight cards in Phase 2** (proposed) rather than Phase 1?
7. **FanDuel CSV import:** when?
8. ~~N3~~ Resolved: N3 = a post-game "teammate left the game" term (from `possible_injuries`), deferred to the Phase 4 why engine.
9. **Stint-path features in production DFS:** allow them once the 2021-26 projection backtest shows no CRPS loss, even though the DFS backtest only validates the box path? Proposed: yes.
