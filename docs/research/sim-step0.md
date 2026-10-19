# Research: sim step 0, the targets a possession sim must hit (T2.5, M1a, 2026-10-19)

This is measurement only: what Path B (docs/PLAYERS.md §4.5) has to reproduce before anyone builds it. Script: `research/sim_step0.py`. Full table (every quantity, CI and season): `research/sim_step0_output.txt`. Machine-readable targets for the T2.6 SMM: `research/sim_step0_targets.json`. Tests: `tests/test_sim_step0.py`.

## Data and method

- **Fit window:** 2019-20 → 2023-24 regular season, 5,554 games. **Held out:** 2024-25 (1,224 games) and 2025-26 (1,224 games). Nothing is fitted on the hold-out; it is reported as an out-of-sample (OOS) check.
- **Inputs:** the cdn.nba.com PBP archive (`var/raw/nba_cdn/playbyplay`), parsed with `pbp.normalize` after `stints_store.repair_legacy_pbp`. Possessions use the counting method of `pbp._mark_possessions`. Lines are the `closing_lines` close (kaggle source preferred, espn as fallback).
- **Game filter:** a game needs PBP and a closing line, and its PBP final score must equal `team_games`. 10 of 8,012 games were dropped for score mismatches. 2019-20 has only 786 games because the archive has gaps that season.
- **CIs:** 95% percentile intervals from B = 1,000 bootstraps that resample whole games within each season (clustered by game, stratified by season). Every quantity is a smooth function of per-game sums, so one weight matrix serves all of them.
- **Segments:** regulation is split into eight 6-minute segments. m_k is the home margin at the end of segment k (m₃ ≡ m after Q3 = m₆ in segment units). e = −closing spread is the expected home margin.
- **Runtime:** 56 s on geekom1 with 3 workers. 52 s of that is parsing the 8,012 PBP files; with a cached parse, the bootstrap takes 4 s. The Mac was not needed.

## Targets

Fit = 2019-24 pooled. OOS = 2024-26 pooled. Season ranges are the five fit seasons.

### Line-conditional team scoring

| Quantity | Fit [95% CI] | OOS [95% CI] | Fit seasons |
|---|---|---|---|
| Var(H \| line) | 126.1 [121.6, 130.9] | 127.8 [121.2, 135.0] | 124–130 |
| Var(A \| line) | 125.1 [120.2, 130.4] | 127.0 [119.8, 134.3] | 120–129 |
| **Var(team \| line)** | **125.6 [121.7, 129.4]** | 127.4 [122.3, 132.8] | 122–129 |
| **Cov(H, A \| line)** | **38.7 [34.9, 42.4]** | 33.7 [28.7, 39.0] | 32–49 |
| corr(H, A \| line) | 0.31 [0.28, 0.33] | 0.26 [0.23, 0.30] | 0.26–0.39 |
| sd(margin \| line) | 13.19 [12.88, 13.45] | 13.69 [13.26, 14.11] | 12.43–13.61 |
| sd(total \| line) | 18.13 [17.76, 18.49] | 17.95 [17.46, 18.48] | 17.72–18.77 |

**The 13.5–13.9 margin sd claimed in docs/PLAYERS.md §0.4 does not hold for every season since 2020-21.** On the DB's closing lines, 2022-23 is 12.4–12.5: kaggle espnbet gives 12.39 and sbr gives 12.54, on all games, not just the PBP subset. 2019-20 is also 12.4–12.8. The other seasons are 13.5–13.9. Because the pooled margin sd is lower, the pooled Cov comes out at 38.7 rather than the ≈ 31 in §4.5. The OOS value, 33.7, is closer.

### Pace and points per possession

| Quantity | Fit | OOS |
|---|---|---|
| Possessions per team per game | 99.1 [99.0, 99.2] | 99.4 |
| sd(possessions \| total line) | 4.86 [4.74, 4.98] | 4.72 |
| Points per possession (PPP): mean | 1.137 [1.135, 1.140] | 1.154 |
| **Per-possession points variance σ²** | **1.4075 [1.405, 1.410]** | 1.429 |
| PPP distribution: 0 / 1 / 2 / 3 / 4+ | .491 / .032 / .329 / .146 / .002 | .487 / .031 / .325 / .155 / .002 |
| **Points per extra possession**, given the line (team on its own N) | **0.665 [0.616, 0.709]** | 0.684 |

PPP and possession length depend on how the possession started (fit window):

| Start | Share | PPP | Var | Seconds |
|---|---|---|---|---|
| After a make | .482 | 1.109 | 1.414 | 17.6 |
| After a defensive rebound | .359 | 1.150 | 1.416 | 11.3 |
| After a steal | .075 | 1.308 | 1.291 | 8.6 |
| After a dead-ball turnover | .064 | 1.104 | 1.413 | 15.6 |
| Period start | .021 | 1.054 | 1.389 | 18.8 |

**Drift:** PPP rose from 1.105 (2019-20) to 1.158 (2023-24), and is 1.150 and 1.158 in the OOS seasons. σ² is stable at 1.40–1.43. Only 23 of 1.81M points could not be attributed to a possession.

### The §4.5 arithmetic, redone

| Quantity | Fit | OOS |
|---|---|---|
| iid possession noise N̄σ² (per team) | 139.5 | 142.0 |
| plus exogenous pace μ²·Var(N \| line) | 30.6 | 29.7 |
| = iid-sim Var(team) | 170.1 | 171.7 |
| **Real minus iid** | **−44.4 [−48.4, −40.7]** | −44.3 |
| iid-sim sd(margin) / sd(total) | 16.70 / 20.03 | 16.85 / 20.07 |
| **Var(total \| N, line)** vs its iid value 2N̄σ² | **279.2 [269.3, 289.4] vs 279.0** | 267.3 vs 284.1 |
| Var(margin \| N, line) | 173.9 [166.0, 180.9] | 187.4 |
| Cov(H, A \| N, line) = (VT − VM)/4 | 26.3 [23.2, 29.4] | 20.0 |
| Implied shared-efficiency latent τ_g (PPP) | **0.002 [0, 0.016]** | 0 (excess < 0) |

### Segments: variance growth and the restoring force

Var(lead vs line) at each segment boundary, with lead L_k = m_k − e·k/8:

| After segment | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Fit | 35.5 | 67.4 | 94.3 | 118.1 | 141.8 | 160.3 | 175.8 | **172.6** |
| OOS | 35.8 | 68.0 | 95.3 | 120.1 | 147.7 | 171.0 | 190.7 | 187.1 |
| AR(1) using the measured ρ_k and a constant innovation of 35.1 | 35.5 | 65.9 | 93.8 | 117.4 | 140.1 | 158.2 | 174.9 | 173.8 |

Var(total vs line) grows linearly instead: about 37 per segment, reaching 300.9 after segment 8 (OOS 296.3).

**Restoring force ρ_k:** the OLS slope of Δm over segment k on m at its start, controlling for e. ρ < 0 means the lead shrinks.

| Segment | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Pooled |
|---|---|---|---|---|---|---|---|---|
| Fit | −.068 | −.057 | −.063 | −.054 | −.063 | −.060 | **−.109** | **−.071 [−.077, −.065]** |
| 95% CI | ±.026 | ±.018 | ±.016 | ±.014 | ±.014 | ±.013 | ±.012 | |
| OOS | −.086 | −.062 | −.070 | −.049 | −.051 | −.048 | −.097 | −.066 [−.074, −.057] |

Other segment results:
- **The pull is toward a margin of 0, not toward the line's path.** In the pooled fit the coefficient on e·k/8 is −0.122 [−0.155, −0.090] (OOS −0.073 [−0.115, −0.032]). It would be 0 if leads decayed toward the spread-implied path.
- Per-segment line drift: 0.184·e (OOS 0.166), not e/8 = 0.125.
- The residual sd of Δm per segment is 5.93, i.e. variance 35.1. The iid value is 2 × 12.2 × 1.41 ≈ 34.4.
- **There is no restoring force on the total.** The pooled ρ_T is +0.011 [0.005, 0.016], slight persistence; per segment it runs from +0.04 to 0.00.
- Leader's Δm minus line drift over the next 6 minutes, by |m| at the segment start:

| \|m\| | 0–5 | 6–10 | 11–15 | 16–20 | 21+ |
|---|---|---|---|---|---|
| Q1–Q3 starts | −0.19 | −0.51 | −0.68 | −1.07 | −0.96 |
| Q4 starts | −0.36 | −0.94 | −1.25 | −1.37 | −2.12 |

  OOS is the same shape: Q4 −0.33 / −0.85 / −0.90 / −1.51 / −1.63.

### b₄, final margin on m₃

| Quantity | Fit | OOS |
|---|---|---|
| **b₄** (controls for e) | **0.843 [0.826, 0.861]** | 0.861 [0.836, 0.887] |
| Coefficient on e | 0.276 [0.244, 0.308] | 0.297 |
| Residual sd | 7.76 | 7.83 |
| b₄ without the line | 0.899 [0.882, 0.916] | 0.938 |
| b₄ on the regulation margin | 0.845 | 0.864 |

b₄ by fit season: 0.81, 0.84, 0.85, 0.85, 0.86 (OOS 0.86, 0.87). It drifts up slightly.

### Q4 by |m₃|

| \|m₃\| | 0–5 | 6–10 | 11–15 | 16–20 | 21+ |
|---|---|---|---|---|---|
| Share of games | .325 | .234 | .180 | .119 | .142 |
| Q4 possessions per team | 23.60 | 23.73 | 23.84 | 24.11 | 24.52 |
| Q4 ÷ mean Q1–Q3 quarter pace | .952 | .956 | .959 | .966 | .976 |
| Leader Q4 PPP | 1.137 | 1.114 | 1.110 | 1.100 | 1.081 |
| Trailer Q4 PPP | 1.144 | 1.165 | 1.172 | 1.164 | 1.164 |
| **Leader Q4 net/100** [CI] | −0.7 [−2.3, 0.9] | **−5.1 [−6.7, −3.3]** | **−6.2 [−8.3, −4.0]** | **−6.4 [−8.9, −3.9]** | **−8.3 [−10.7, −6.0]** |
| Line-implied net/100 for the leader | +1.0 | +1.7 | +3.2 | +3.6 | +5.9 |
| OOS leader Q4 net/100 | −0.6 | −3.4 | −2.8 | −6.0 | −4.7 |

Q4 is 2–5% slower than Q1–Q3 in every bin. It speeds up only mildly with |m₃|: 23.6 → 24.5 possessions, a range under 1 possession. OOS bins are within ±0.25 of fit.

### Rebounds (S7)

Outcome of each miss: player or team rebound, offensive or defensive. Every miss in the feed has a rebound row: "no rebound" = 0.000.

| Quantity | Missed FG: fit | Missed FG: OOS | Missed final FT: fit | Missed final FT: OOS |
|---|---|---|---|---|
| Misses per game | 94.2 | 94.7 | 4.91 | 4.85 |
| OREB rate (player + team) | **.283 [.282, .284]** | .305 | **.112 [.108, .116]** | .129 |
| Team-rebound share of all rebounds | **.114** | .120 | **.049** | .054 |
| Team share of defensive rebounds | .061 | .069 | .032 | .034 |
| Team share of offensive rebounds | .246 | .238 | .188 | .190 |

Dead-ball team rebounds come after non-final FTs, which are excluded. Their share is 0.000 for FGs and 0.002 for final FTs. **OREB is drifting:** the FG OREB rate went .275 → .292 across the fit seasons, then .301 and .308 OOS. Fit it per season.

### Overtime

| Quantity | Fit | OOS |
|---|---|---|
| **P(OT)** | **.055 [.049, .061]** | **.047 [.039, .055]** |
| By fit season | .069, .052, .047, .064, .048 | .049, .044 |
| By \|spread\|: 0–3 / 3.5–6 / 6.5–10 / 10.5+ | .062 / .061 / .052 / .038 | .064 / .041 / .050 / .034 |

## What it means for Path B

- **The restoring force is material, and on its own it closes the margin gap.** An iid sim overshoots Var(margin | line) by 60%: 279 against 174. The first 6 minutes are exactly iid (35.5 against 34.4), and the shrinkage starts after that. An AR(1) using the measured ρ_k (−0.06 per segment, −0.11 in the last) and the iid innovation reproduces the whole variance curve to within about 2. The pull is toward 0, not toward the line, so the §4.5 spec "−ρ·m_{k−1}·s_t on every segment" is the right shape. The team-strength drift must then be larger than the line's e/8 per segment (0.18·e) to make up for being pulled back.
- **There is no room for a shared game latent (τ_g ≈ 0).** Given the game's possessions and the line, Var(total) equals the iid possession noise to within 0.2 (279.2 against 279.0). The 95% upper bound on τ_g is 0.016 PPP, and OOS the excess is negative. **Cov(H, A | line) = 38.7 is not evidence for η_g.** About 12 of it is shared pace (0.72² × 23.6). The other 26 is a mechanical by-product of the margin shrinking while the total doesn't: (279 − 174)/4.
- **Every other latent has to fit in a budget of about 10 points of total variance.** Independent team latents η_t, player shooting latents ζ_i and rate (δ_t) uncertainty all add total variance, and there is no restoring force on totals to absorb it (ρ_T ≈ +0.01). As a rough ceiling, τ_t ≲ 0.02 PPP. Anything the sim adds to margin variance must be offset by a stronger ρ. The measured ρ is reduced-form, so it already includes any latent persistence.
  - **This makes G1 a live risk:** the SMM will probably push the τ's to 0.
  - G1 only triggers if the variance targets still fail at that point. With pace and ρ they need not fail: total = iid + pace, margin = iid + ρ.
- **Pace is endogenous; don't draw it exogenously.** An extra possession is worth 0.67–0.72 points, not μ = 1.14, because empty trips start fast possessions:
  - after a steal: 8.6 s;
  - after a defensive rebound: 11.3 s;
  - after a make: 17.6 s.

  An exogenous P_s ~ N(P̄, σ_P²) filled at μ PPP adds about 31 to Var(team) (and 122 to Var(total)) where reality adds about 12 (≈ 49). Path B should run on a clock, with possession duration drawn by start type (the table above). Then pace and efficiency co-move by construction, and σ_P becomes a small residual.
- **Q4 coasting exists on top of the game-wide force.**
  - The segment-8 ρ is about 1.8× the others (−0.11 against −0.06).
  - With |m₃| ≥ 6, the Q4 leader is outscored per possession (net/100 −5 to −8) when the line expects +2 to +6. That gap is the β_q(|m₃|) target.
  - b₄ = 0.843 (OOS 0.861): use 0.84–0.86 with the ±0.05 band.
  - The Q4 pace shift by |m₃| is small (under 1 possession across bins), so the "±1 per bin" target is loose; match the ratio column instead.
- **Retune two SMM bands.**
  - P(OT) is 4.4–4.9% in 2023-26. §4.5's 5.5–6.5% band would fail on the hold-out seasons. Use the per-fold fit value ± 1 pp.
  - sd(margin) varies by season: 12.4 in 2019-20 and 2022-23, 13.5–13.9 otherwise. The ±3% band must use each fold's own seasons, as §0.4 already says.
- **S7 inputs:** team rebounds take 11.4% of FG-miss rebounds (24.6% of offensive ones) and 4.9% of final-FT-miss rebounds. OREB is .283 after a missed FG and .112 after a missed final FT. Both drift upward, so fit them per season on seasons < S.

## Caveats

- **The possession-level variance is pooled** across teams, quarters and game states. Its between-team part is about (0.05)² ≈ 0.003 of 1.41, which is negligible.
- **Conditioning on N is not neutral.** N is endogenous: hot shooting means more makes, slower possessions and a lower N. So part of any efficiency latent can hide in the N slope. The unconditional check agrees anyway: 279 + 4 × 0.72² × 23.6 ≈ 328, against Var(total | line) = 328.6.
- **Segment boundaries** use the score after the last event at or before the boundary, in feed order. An FT at exactly 6:00 counts in the earlier segment.
