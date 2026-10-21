# Return from absence: minutes and DK error (task #3, 2026-10-21)

*Research only: we measured before building anything. Driver: `research/return_minutes.py`. Tables: `var/tmp/ret3/out/tables.md`, `results.json`. Live model: `pathA-v0.3+p2`.*

## Answer in short

- **(a) We do not systematically over-project players in games 0–2 back.** Minutes v1 already cuts their minutes:
  - E[min ∣ plays] is 20.0 / 21.5 / 22.6 in games 0 / 1 / 2 after the return, against 26.9 in normal games.
  - On average the actual minutes match that cut. The first game back is in fact slightly **under**-projected: +0.6 min and **+1.5 DK** (avail, 5 seasons, n 1,802). This is +0.63 (0.16) min and **+1.67 (0.22) DK** more than the same players' error in normal games.
  - Games 1 and 2 are unbiased.
  - About 60% of the game-0 DK gap is per minute (+4.9% (1.0) DK/min), not minutes.
  - Over-projection does show up in two narrower places, both small-n:
    - **long absences (16+ team games):** −0.55 (0.26) min in games 1–2 (avail); −1.1 to −1.3 min in games 0–2 on the live path (n 54–120, ≈ 1.8 SE);
    - **soft-tissue returns** (hamstring/calf/groin/Achilles, 2025-26 only): −1.50 (0.45) min in games 0–2 (n 159, 50 players).
  - In both cases DK is *not* over-projected, because the per-minute under-projection offsets the minutes.
- **(b) The pattern is the same on the live path.**
  - live+p2 first game back: −0.55 (0.40) min, +0.60 (0.59) DK. Avail 2025-26 on the same dates: −0.27 / +0.98.
  - Lock−10 is the same as lock−45.
  - The live path adds one P(play) issue. Returners listed **Questionable** at lock−45 play 68% of the time against P 0.55: −0.125 (0.038), DK bias +2.8 (1.2), n 96. At lock−10 this is fine (−0.01).
- **(c) A return-specific fix is not worth building.**
  - We fit a post-fit multiplier by games-since-return × absence length (± tier) walk-forward. It gains nothing in any test season: DK MAE overall −0.0006 … +0.0039, rows it touches −0.007 … +0.035.
  - Even fit in-sample on the live slice (the ceiling), it gives DK MAE **−0.0025 overall (−0.03%)** and −0.023 (−0.3%) on the 11% of rows it touches.
- **The lead worth testing is general, not about returns.**
  - The per-minute error moves with E[min ∣ plays] relative to the player's recent minutes, for **everyone**:
    - projected well below his recent minutes: per-minute rate +2…+9% under-projected;
    - projected > 1.2× his recent minutes: −2…−6% over-projected, DK bias −0.8 … −2.8.
  - This fits the state's `minutes_adj` table f(m / mpg), which was fit on *realised* minutes ratios (blowouts, foul trouble, injury exits) but is applied to *planned* changes: ramps, promotions when a star sits.
  - Our measure of its size: a walk-forward DK factor by that ratio ("Fix B") improves **DK MAE by −0.009 … −0.014 in every season (3–4 SE)** and quantile-score CRPS by −0.003 … −0.005 (≈ 2 SE), live+p2 included. Most of the gain comes from promoted players, which is T2.7's "star-out bump over-credited" finding.
- **"Minutes restriction" news: we have no usable signal yet.**
  - Official reports never mention minutes. Their only relevant label is "Return to Competition Reconditioning": 9 players in 2025-26, 4 returns in the slice.
  - ESPN comments exist only since 2026-10-18 (4 days, 33 captures). They contain 2 explicit restriction mentions (Sexton, Z. Collins), 1 "limited minutes" (Bona) and 6 "ramping up".
  - There is no history, so no backtest is possible.

## Data and definitions

Projection rows come from the existing walk-forward caches on the Mac scratch box. Nothing was re-simulated. Arms were identified from the drivers' logs (`courtside-compute/{p2,t24,cx}/logs`, `cx/out/harness_*.json`).

| arm | source | protocol | rows |
|---|---|---|---|
| live+p2 | `p2/out/players_pathA-v0.3+p2_live.parquet` (cache `p2/.../live/2025-26/v1-1888940222`) | T2.4 slice, lock−45, 104 dates 2025-12-22 → 2026-04-12 | 26,630 |
| live10+p2 | `p2/out/players_pathA-v0.3+p2_live10.parquet` (`v1-bab3326db9`) | lock−10 | 26,624 |
| live (p2 off) | `p2/out/players_pathA-v0.3_live.parquet` (`v1-3cd421b06b`) | lock−45, reference | 26,630 |
| avail 2025-26 | `t24/out/players_pathA-v0.3_avail.parquet` (`v1-47d4067b9f`) | oracle availability, same dates | 24,778 |
| avail 2019-20 / 2020-21 | `cx/.../avail/2019-20/v1-16c46b8db0`, `2020-21/v1-c19b1a7e03` | context ON (= v0.3), RotoGuru DK pool rows | 33,813 / 34,387 |
| avail 2023-24 / 2024-25 | `cx/.../avail/2023-24/v1-df4e9671e4`, `2024-25/v1-b2065d96b8` | context ON, every simulated player | 38,125 / 38,409 |

In each cx directory the other hash is the context-OFF twin. `var-t21` is the T2.1 run on pathA-v0.1 (minutes v0) and was not used.

**Return state.** Computed from regular-season box appearances strictly before the game, using the local DB read-only.
- `missed_now`: team games of his current team since his last appearance. The season start counts if he played the season before, and a whole lost season counts as 82+. Rookies and first seasons are excluded.
- `gsr` (games since return): appearances since the end of the latest run of ≥ 3 missed team games. 0 = first game back.
- **Absence length:** that run of missed games, bucketed 3–5 / 6–15 / 16+.
- **Tier:** mean minutes of his last 10 appearances before the absence. star ≥ 30, starter 24–30, rotation 15–24, bench < 15.
- **Normal games:** no ≥ 3-game absence earlier in the season, or ≥ 16 games since it.
- **Excluded from the main cut:** rows with a team change (trades and new teams).
- **Main cut:** pre-absence ≥ 15 min.

Box scores have no DNP rows, so a bench player's "absence" is often a DNP-CD. That is why bench is reported separately.

**Errors.**
- Minutes: actual − E[min ∣ plays], where E[min ∣ plays] = `exp_min / p_play`, exactly the sim's `min_played`.
- DK: actual − E[DK ∣ plays] = `mean / p_play`.
- Both on played rows with P(play) ≥ 0.05.
- SEs are clustered by date.
- "Excess" = this bucket's mean error − the same players' mean error in their normal games, from a stacked difference with date clustering.

## Main table: games since return (pre-absence ≥ 15 min, played rows)

| | avail, 5 seasons: n / min bias (SE) / DK bias (SE) / excess min, DK | live+p2 (lock−45, 2025-26): n / min / DK / excess | live10+p2 |
|---|---|---|---|
| all rows | 84,035 / −0.04 (0.03) / −0.12 (0.03) | 12,282 / −0.27 (0.07) / −0.34 (0.08) | −0.20 / −0.23 |
| **gsr 0** (proj 20.0 min) | 1,802 / **+0.57 (0.16)** / **+1.52 (0.22)** / +0.63, **+1.67 (0.22)** | 332 / −0.55 (0.40) / +0.60 (0.59) / −0.06, +1.28 (0.63) | −0.53 / +0.67 / −0.13, +1.22 (0.62) |
| gsr 1 | 1,943 / −0.06 (0.14) / +0.18 (0.21) / +0.01, +0.33 | 321 / +0.05 (0.36) / +0.09 (0.52) / +0.48, +0.75 | +0.02 / +0.06 |
| gsr 2 | 1,801 / −0.06 (0.14) / −0.06 (0.21) / +0.01, +0.09 | 300 / −0.42 (0.27) / −0.39 (0.49) / +0.02, +0.24 | −0.48 / −0.43 |
| gsr 3–5 | 4,712 / +0.12 (0.08) / +0.03 (0.14) | 820 / −0.19 (0.22) / −0.25 (0.36) | −0.22 / −0.27 |
| gsr 6–15 | 11,148 / +0.05 (0.06) / −0.19 (0.09) | 2,086 / −0.00 (0.13) / +0.02 (0.21) | +0.01 / +0.06 |
| 1–2 games missed (back now) | 1,909 / +0.45 (0.15) / +0.59 (0.23) | 300 / +0.31 (0.35) / +0.24 (0.55) | +0.30 / +0.25 |
| normal | 60,720 / −0.10 (0.03) / −0.19 (0.04) | 8,123 / −0.37 (0.08) / −0.52 (0.10) | −0.25 / −0.35 |
| **gsr 0–2** | 5,546 / +0.14 (0.09) / **+0.54 (0.13)** / +0.22, **+0.70 (0.13)** | 953 / −0.31 (0.22) / +0.11 (0.32) / +0.18, +0.79 (0.35) | −0.33 / +0.12 / +0.08, +0.67 (0.34) |

MAE in the same cells:
- minutes MAE: gsr 0 is 5.11 (avail) / 4.99 (live) vs 4.30 / 4.29 normal; gsr 1–2 is 4.46–4.74;
- DK MAE: gsr 0 is 7.45 / 7.64 vs 7.77 / 7.81 normal. The players project lower, so absolute errors are not larger.

Per-minute DK rate (actual vs projected, played rows): gsr 0 **+4.9% (1.0)** avail and **+5.8% (2.6)** live+p2. gsr 1 is +1.1% / +0.2%, gsr 2 is +0.0% / +0.1%, normal is −0.4% / −0.5%.

By season (avail, gsr 0, minutes / DK bias):

| season | minutes bias | DK bias |
|---|---|---|
| 2019-20 | +0.91 (0.42) | +1.89 (0.55) |
| 2020-21 | +1.18 (0.32) | +1.82 (0.48) |
| 2023-24 | −0.08 (0.35) | +1.06 (0.45) |
| 2024-25 | +0.96 (0.31) | +1.80 (0.44) |
| 2025-26 | −0.27 (0.42) | +0.98 (0.58) |

The DK under-projection holds in every season. The minutes part does not.

## By absence length (excess vs own normal games, min / DK)

| | avail, 5 seasons | live+p2 | live10+p2 |
|---|---|---|---|
| 3–5, gsr 0 | n 1,078: min +0.62 (0.21), DK +1.32 (0.28); excess +0.71 / +1.50 | n 161: −0.13 (0.54), +0.52 (0.80) | −0.03, +0.66 |
| 3–5, gsr 1–2 | 1,952: −0.03, −0.01 | 294: +0.02, −0.28 | +0.01, −0.28 |
| 6–15, gsr 0 | 567: +0.68 (0.27), +1.54 (0.39) | 117: −0.78 (0.57), +0.72 (0.87) | −0.67, +0.94 |
| 6–15, gsr 1–2 | 1,236: +0.10, +0.31 | 207: +0.07, +0.39 | −0.04, +0.30 |
| **16+, gsr 0** | 157: −0.26 (0.46), **+2.85 (0.81)** | 54: **−1.27 (0.71)**, +0.57 (1.31) | **−1.70 (0.67)**, +0.09 |
| **16+, gsr 1–2** | 556: **−0.55 (0.26)**, −0.21 (0.37) | 120: **−1.10 (0.60)**, −0.73 (0.83) | −1.09 (0.61), −0.75 |
| 16+, gsr 3–5 | 679: −0.20, −0.14 | 150: −0.10, +0.99 | −0.23, +0.90 |

Long absences are the only place where minutes are over-projected, by about 0.5–1.7 min. On the live slice n is small (54–150), at about 1.5–2.5 SE.

## By pre-absence tier (gsr 0–2, excess vs own normal, min / DK)

| tier | avail, 5 seasons | live+p2 |
|---|---|---|
| star | n 1,480: −0.10 (0.15) / +0.18 (0.28) | n 235: −0.74 (0.41) / −0.12 (0.73) |
| starter | 1,490: +0.32 (0.17) / +0.80 (0.27) | 280: +0.57 (0.36) / +0.72 (0.73) |
| rotation | 2,576: +0.47 (0.14) / +0.99 (0.18) | 438: +0.75 (0.38) / +1.49 (0.52) |
| bench (often DNP-CD streaks) | 6,353: +0.50 (0.12) / +0.86 (0.13) | 1,087: +0.28 (0.29) / +0.68 (0.37) |

Stars come back at about the projected level; the live slice hints at −0.7 min for them. Starters and rotation players beat their projection.

## Report reason before the return (2025-26 slice only, live+p2, gsr 0–2)

The reason comes from the last official-report row for the player before the return game, captured after the absence began. Report names were matched to player ids at 98.3%. The local DB has no 2025-26 tip times, so the cutoff is 22:00Z on game day.

| reason | n (players) | proj → act min | min bias (SE) | DK bias (SE) | excess min / DK |
|---|---|---|---|---|---|
| knee/ankle/foot | 257 (78) | 24.1 → 23.9 | −0.20 (0.31) | −0.46 (0.56) | +0.21 / +0.17 |
| **soft tissue** (hamstring, calf, groin, Achilles, adductor, soleus, quad) | 159 (50) | 23.0 → 21.5 | **−1.50 (0.45)** | −0.57 (0.79) | **−0.98 (0.47)** / +0.51 |
| other (back, shoulder, hand, concussion, …) | 235 (75) | 22.5 → 23.2 | +0.70 (0.40) | +1.60 (0.63) | +1.25 / +2.29 |
| none (listed, empty reason) | 151 (60) | 17.7 → 16.9 | −0.73 (0.57) | +0.47 (0.76) | +0.11 / +1.42 |
| G League / two-way | 68 (19) | 16.8 → 15.9 | −0.86 (0.83) | −1.72 (1.01) | −1.58 / −3.21 |
| rest / management | 49 (16) | 21.4 → 21.3 | −0.09 (0.83) | −0.11 (1.18) | +0.02 / −0.83 |
| illness | 22 (8) | 21.0 → 20.9 | −0.12 (0.96) | +0.09 (1.25) | |
| reconditioning ("Return to Competition Reconditioning") | 12 (4) | 24.5 → 25.2 | +0.79 (1.74) | −0.83 (3.26) | |

Avail 2025-26 gives the same picture (soft tissue −1.21 (0.45) min). Soft tissue is the one reason with a minutes ramp the model misses: about −1.2 to −1.5 min over games 0–2. That is one season and 50 players, and it cannot be tested walk-forward until 2026-27 reports accumulate.

Note: `availability.reason_class` files "Return to Competition Reconditioning" under `gleague`. It is a long-injury return label (9 players in 2025-26: 445 Out, 65 Q, 42 Available rows), not a G League one. This is harmless for P(play) today, but worth knowing if reason ever feeds minutes.

Game-day status at the first game back (live+p2):
- listed **Questionable** (n 91): −1.31 (0.62) min, −1.0 (1.0) DK;
- not listed (91): −0.16 min, +1.0 DK;
- Available (60): +0.17 min, +1.7 DK;
- Probable (42): −0.74 min.

## The live path: P(play) for a return still pending

These are rows whose ≥ 3-game absence is still running and who are not listed Out (pre ≥ 15 min). Columns: P(play) / share played / gap (SE) / DK bias with DNP = 0.

| status | lock−45 live+p2 | lock−10 live10+p2 | lock−45 p2 off |
|---|---|---|---|
| all (608 / 588) | 0.549 / 0.548 / +0.001 (0.017) / +0.80 | 0.598 / 0.566 / +0.032 / +0.03 | 0.559 / +0.011 / +0.99 |
| **Questionable (96 / 69)** | 0.552 / **0.677** / **−0.125 (0.038)** / +2.80 (1.23) | 0.686 / 0.696 / −0.010 / −0.30 | 0.441 / −0.236 / +5.04 |
| Available (112) | 0.940 / 0.938 / +0.002 / +0.81 | 0.960 / 0.943 | 0.766 / −0.172 / +4.34 |
| Probable (20) | 0.957 / 1.000 / −0.043 | | 0.738 / −0.262 |
| unlisted (267) | 0.478 / 0.382 / +0.096 (0.030) / −0.79 | same | same |
| fringe pool, no status (84) | 0.339 / 0.488 / −0.149 (0.056) / +4.01 (1.02) | 0.394 / −0.094 | 0.825 / +0.337 |

- p2 already fixed most of this: the Available/Probable returners were at −0.17 / −0.26.
- What is left: a returning Questionable player plays 68% vs the cell's 55% at lock−45. The slice-wide Q rate is 52–54%, so "returning" is informative inside the Q cell.
- The unlisted rows (absence running, not on the report) are mostly DNP-CD/two-way and are slightly high.

## Fixes tested (walk-forward: fit on seasons < S, applied to S)

**Fix A: return multiplier**
- The cells are games-since-return {0, 1, 2, 3–5} × absence length, optionally × tier. Each is a ratio shrunk with 30 pseudo-rows at 1.
- E[min ∣ plays] is scaled by the minutes ratio; the DK mean and quantiles are scaled by the DK ratio.
- Scored on the standard DK population (E[min] ≥ 10, DNP = 0). The quantile score (QS, 5 stored quantiles) runs at ≈ 0.98 × CRPS.

Prior-season cells (gsr×abs):
- gsr 0: minutes ×1.01–1.04, DK ×1.07–1.13;
- gsr ≥ 1: within ±4% of 1.

| test | DK MAE Δ overall (SE) | QS Δ overall | touched rows: DK MAE base → Δ | touched: minutes MAE Δ |
|---|---|---|---|---|
| 2020-21 | +0.0035 (0.0024) | +0.0018 | 7.95 → +0.035 (0.024) | −0.001 |
| 2023-24 | −0.0006 (0.0019) | −0.0000 | 8.11 → −0.007 | +0.005 |
| 2024-25 | +0.0024 (0.0021) | +0.0011 | 8.29 → +0.022 | −0.016 |
| 2025-26 avail | +0.0017 (0.0025) | +0.0019 | 8.22 → +0.016 | +0.016 |
| 2025-26 live+p2 | −0.0004 (0.0023) | +0.0022 | 8.49 → −0.004 | +0.019 |
| 2025-26 live10+p2 | +0.0039 (0.0026) | +0.0034 | 8.16 → +0.035 | +0.019 |
| live+p2, in-sample cells (ceiling) | **−0.0025 (0.0015)** | −0.0012 | 8.49 → −0.023 (0.014) | −0.020 |

With × tier, the walk-forward results are the same (−0.0002 … +0.0034) and the ceiling is −0.0051 (0.0023).

The verdict: no gain. The minutes are already right on average. The game-0 DK gap is real (+1.5) but too small and too noisy per row to move MAE or CRPS. Shifting the mean of a skewed DK distribution doesn't help its median or its quantiles.

**Fix B: DK factor by E[min ∣ plays] / last-10 average minutes** (9 bins, every row, level-neutral)
- This is a stand-in for the structural fix, not the fix itself.
- Prior-season factors run from ×1.034 (< 0.7) and ×1.01–1.02 (0.7–0.97), through 1.00 near 1, down to ×0.976 (1.2–1.4) and ×0.965 (> 1.4).

| test | DK MAE Δ (SE) | QS Δ (SE) | gsr 0–2 rows | ratio < 0.8 | ratio > 1.2 |
|---|---|---|---|---|---|
| 2020-21 | **−0.0092 (0.0024)** | −0.0031 (0.0013) | +0.002 | +0.004 | −0.035 (0.005) |
| 2023-24 | **−0.0136 (0.0031)** | −0.0039 (0.0017) | −0.027 (0.012) | +0.020 | −0.061 (0.010) |
| 2024-25 | **−0.0096 (0.0029)** | −0.0025 (0.0014) | +0.004 | +0.013 | −0.051 (0.012) |
| 2025-26 avail | **−0.0144 (0.0036)** | −0.0045 (0.0022) | −0.024 (0.012) | −0.010 | −0.051 (0.014) |
| 2025-26 live+p2 | **−0.0141 (0.0039)** | −0.0052 (0.0022) | −0.019 (0.013) | −0.014 | −0.045 (0.015) |
| 2025-26 live10+p2 | **−0.0115 (0.0036)** | −0.0036 (0.0023) | −0.008 | −0.010 | −0.040 (0.014) |

Per-minute error by the projected / recent minutes ratio (avail, 5 seasons, rows other than returns):

| ratio | < 0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.1 | 1.1–1.2 | 1.2–1.4 | > 1.4 |
|---|---|---|---|---|---|---|---|
| per-minute error | +3.4% (0.9) | +2.3% (0.7) | +0.6% | −0.4…−0.6% | −0.8% | −2.0% (0.5) | **−4.8% (0.8)** |
| DK bias, > 1.4 bin | | | | | | | −1.61 |

Returns at < 0.7 run +8.8% (1.5) per minute. On live+p2 the > 1.4 bin is −5.8% (1.8) with DK bias −2.80.

The mechanism, which we have not yet proven:
- In `copula._grids` and `state.minutes_multiplier`, each sim's DK uses rate × m × f(m / mpg).
- f was fit on realised m / mpg. When a player is pulled early or blown out, he produces less per minute (pts ×0.85–0.92 at 0.6–0.8 of mpg), and the f table carries that over.
- A *planned* change in E[min] is a different case: a ramp after injury, a promotion when a star sits.

What f does to planned changes:
- A ramped-down returner is drawn at about 0.8 × mpg and loses about 5–6% DK per minute he doesn't lose in reality.
- A promoted player at 1.3–1.5 × gets +3–5% he doesn't get. This adds to the usage bump that T2.7/T2.3 already flagged as over-credited.

The fix that addresses the cause is to key f on unplanned deviation, m / E[m ∣ plays], refit on that basis, rather than on m / mpg.

## Recommendation

1. **Don't build a return-from-injury multiplier or a return GBM feature now.** The V1_RETURN features (missed_now, gsr_now, abs_len_now) already capture the ramp on average. A return-only correction gains nothing walk-forward. Its in-sample ceiling is −0.0025 DK MAE overall (−0.03%) and −0.023 on the rows it touches.
2. **Do build and test (≈ 1–1.5 days): re-key `minutes_adj` to planned minutes.**
   - In `state.fit_params`, fit f on m / E_pre[m]. E_pre[m] is the walk-forward minutes-v1 prediction for that game; m / mpg can stay as the fallback where no prediction exists.
   - Apply f to the sim's m / E[m ∣ plays] in `copula._grids` / `simulate_game*` and in `usage.py`.
   - Expected gain, from Fix B as a proxy: DK MAE −0.009 … −0.014 overall (3–4 SE every season, the live path included) and CRPS ≈ −0.003 … −0.005. That is about half of what the context layer gave. Most of it lands on promoted players (−0.04 … −0.06 DK MAE per row at ratio > 1.2); returners get −0.02 (game-0 DK bias +1.5 → ≈ 0).
   - **Test** with the T2.1 harness on the Mac scratch: an avail arm for 2023-24 and 2024-25, with the f refit on seasons < S, paired vs v0.3 with clustered SE. Then the T2.4 slice live / live10 with p2. Bar: CRPS better by ≥ 2 SE in both seasons and no tier worse by > 1 SE. Also check the S14 value-play test and T2.7's star-out gap, which this should shrink.
   - **Re-measure afterwards:** with the per-minute offset gone, the long-absence / soft-tissue minutes over-projection (−0.5 … −1.5 min) will show in DK.
3. **Small, cheap P(play) item (≈ 2 h, when the lock−10 Questionable bucket is refit, HANDOFF next #2).**
   - Add a "returning from ≥ 3 missed games" split to the Questionable cell at lock−45: 0.68 played vs 0.55 projected, −0.125 (0.038), n 96.
   - Fit it leave-one-month-out on the 151 rescued days and check Brier on the slice. The expected effect is small (about 100–150 rows a season, +2.8 DK bias on them).
4. **Start collecting the restriction signal now, and build no feature yet.**
   - Store ESPN `longComment` as well as `shortComment`; the DB `reason` is cut at about 200 characters.
   - Add a regex flag (`minutes restriction|minutes limit|limited minutes|ramping up|reconditioning`) in the injuries collector, which is a `src/` change for the owner.
   - Revisit around January with about 2.5 months of 2026-27 flags and reports: fit soft-tissue / long-absence / restriction-flag minutes multipliers on 2025-26 and test on 2026-27 (the walk-forward the report data allows).

## Caveats

- **2025-26 has only 104 dates.** On the live slice the gsr 0 cells are n ≈ 330, the long-absence cells 54–150, and the reason cells ≤ 260 (reconditioning: 4 players). Read anything under about 2.5 SE as a hint.
- **Absences come from box appearances.** For bench players they are often DNP-CD streaks, not injuries. That is why the main cut uses pre-absence ≥ 15 min.
- **Avail protocol:** the oracle knows the player is available. Live rows use the real report.
- **"Excess vs own normal"** uses each player's normal games over the whole season, for diagnosis only. The walk-forward fixes use only seasons < S.
- **Fix B is a proxy, not the proposed fix.**
  - It is a post-hoc DK factor by a box-derived ratio (last-10 average minutes, not the state's decayed mpg).
  - QS is a 5-quantile CRPS proxy.
  - The structural version can do better or worse, and the harness has to judge it.

## Reproduce

```
python research/return_minutes.py pull    # scp/rsync the caches above into var/tmp/ret3 (~45 MB)
python research/return_minutes.py run     # local, about 1 min, light: state, reasons, tables, fixes
python research/return_minutes.py news    # stored-text grep (local DB + var/raw)
```

The Mac-side ESPN raw grep (`var/raw/espn/injuries`, 33 captures 2026-10-18 → 2026-10-21) was run read-only with a one-off script. Its results are quoted above.
