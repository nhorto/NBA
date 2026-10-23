# B10: the truth-shift β by role change and anchor (2026-10-23)

Measurement only, no source changes. This is the B10 precondition for P1/D1
(`docs/IMPROVEMENTS.md`). Script: `research/truth_beta_roles.py`. It uses no sims. It reads the
existing per-row harness outputs of the live arm `pathA-v0.3+mf+p2` (Mac scratch
`courtside-compute/mf/out_dk/rows_*`), and the 2026-10-23 nightly DB copy for box scores,
RotoGuru salaries and the live nights.

## Method

- **β**: through-origin slope of (actual − μ_played) on (anchor − μ_played), on played rows.
  This is the quantity `truth_sims` uses. μ_played = dk_mean / p_play. SE is two-way clustered
  by date × player: the `paired_cluster2` math applied to the slope's influence function, or
  the CGM sandwich for the mix. The live sample covers only 3 nights, so there it is clustered
  by game × player.
- **Anchors**:
  - **fppg**: DK FPPG. No DK FPPG is stored before 2026-27, so historical rows use our
    box-score proxy (`BoxBaselines`: season-to-date once ≥ 5 games, else last season). Live
    rows use DK's own FPPG.
  - **l10**: last-10 DK average.
  - **sal**: salary-implied, `sal_played`, an OLS fit on RotoGuru played rows of seasons < S.
    It exists only for RotoGuru seasons and live.
  - **mix**: a joint regression on the anchors above.
- **Buckets**: r = E[min | plays] / last-10 minutes average. The "returning" group (it
  overlaps the r buckets) is players who missed ≥ 3 team games right before this one, after at
  least one appearance this season.
- **Gains**: walk-forward. Each design is fit on the datasets of seasons < S and applied as
  `truth_sims` applies it (played mass only).
  - **MAE** of μ_played + δ, on played rows.
  - **CRPS**: a proxy on all projected rows, using the quantile function rebuilt from q10..q90.
    The proxy tracks the sim CRPS: means within 0.02, r = 0.99.
  - Both are paired vs OFF (δ = 0). Units below are DK points per row.

## β table (SE). Datasets: RG = RotoGuru 2019-21 pooled, HZ = harness 2023-25 pooled, LP = 2025-26 live-path backtest, LIVE = 2026-10-20..22

| bucket (played n RG/HZ/LP) | fppg RG | fppg HZ | fppg LP | l10 RG | l10 HZ | l10 LP | sal RG | fppg LIVE | sal LIVE |
|---|---|---|---|---|---|---|---|---|---|
| all (43k/51k/17k; live 323) | .11 (.01) | .08 (.01) | .11 (.03) | .10 (.02) | .09 (.01) | .12 (.03) | .12 (.02) | .33 (.15) | .46 (.11) |
| <0.8 | .12 (.03) | .12 (.03) | .14 (.05) | .07 (.02) | .09 (.02) | .15 (.04) | .12 (.03) | .40 (.28) | .54 (.28) |
| 0.8–0.95 | .15 (.03) | .11 (.03) | .09 (.05) | .15 (.03) | .18 (.03) | .09 (.05) | .18 (.03) | .13 (.27) | .48 (.20) |
| 0.95–1.05 | .08 (.03) | .09 (.03) | .06 (.05) | .14 (.03) | .09 (.03) | .10 (.05) | .13 (.03) | .52 (.38) | .35 (.28) |
| 1.05–1.25 | .13 (.03) | .05 (.02) | .17 (.04) | .10 (.03) | .05 (.03) | .23 (.04) | .12 (.03) | .13 (.47) | .48 (.26) |
| >1.25 | .08 (.03) | .06 (.02) | .08 (.05) | .06 (.03) | .05 (.03) | .07 (.05) | .05 (.03) | .68 (.24) | .46 (.15) |
| returning | .26 (.05) | .18 (.05) | .14 (.06) | .24 (.05) | .17 (.05) | .12 (.06) | .14 (.04) | – | – |

The per-season fppg β values are 0.09 / 0.12 (RG), 0.08 / 0.08 (HZ) and 0.11 (LP). The joint
mix splits the weight roughly evenly: RG fppg .06 / l10 .02 / sal .07 (SE .02 each), HZ fppg
.03 / l10 .07.

**β on the current arm is about half of `TRUTH_BETA`.** It is 0.08–0.12, not 0.18; the minutes
fix and p2 have absorbed part of it.

**β does not reverse for promoted rows.** It is still positive (0.05–0.08 at r > 1.25), so
FPPG / L10 do carry signal there. But the gap is about twice as wide (sd 7 vs 3 DK), so the
flat 0.18 shift moves those rows about −1.0 DK on average.

**The least-squares β is misleading where the anchor sits above μ** (demoted and returning
rows). Their gap distribution is skewed: a few big outcomes drive the LS slope, while CRPS and
MAE get worse when the shift is applied.

The **live** slopes look high, but they come from 3 nights at season open, mostly v0.2 and
v0.3+p2 (pre-mf). FPPG there is last season's average, and the slate scored below projection,
which loads a level bias onto the through-origin slope. Do not use them to set β. The
salary-implied live slope (0.46) is the one live signal worth re-testing after about 10 nights.

## Gains vs OFF, walk-forward (×1000 DK; SE)

CRPS on all rows (MAE on played rows in brackets). Base CRPS ≈ 4.3; at r > 1.25 it is ≈ 3.7.

| design | RG 2020-21 | HZ 2023-24 | HZ 2024-25 | LP 2025-26 |
|---|---|---|---|---|
| flat 0.18·fppg (today) — all | −2 (3) [−14] | **+8 (3)** [+7] | **+12 (3)** [+6] | +10 (6) [−10] |
| ↳ promoted r>1.05 | −6 (7) [−58] | +9 (5) [−18] | +13 (7) [−16] | +13 (14) [−59] |
| ↳ r>1.25 | +8 (15) | +7 (9) | **+31 (13)** | +51 (29) |
| ↳ r<0.8 (demoted) | **+21 (8)** [+70] | **+35 (6)** [+121] | **+40 (7)** [+65] | **+31 (10)** [+92] |
| ↳ bench / returning | +11 / +3 | +17 / +5 | +26 / +18 | +28 / +26 |
| flat refit fppg (≈0.09) — all | −5 (2) | +1 (2) | +2 (1) | +0 (3) |
| **bucket β(r) · l10, CRPS-fit — all** | **−6 (1)** [−11] | **−4 (1)** [−9] | **−2 (1)** [−6] | **−5 (2)** [−12] |
| ↳ star / role / bench | −15 / −6 / −2 | −1 / −13 / −0 | −7 / −8 / +2 | −7 / −7 / −2 |
| ↳ promoted r>1.05 | −7 (3) [−32] | −3 (2) [−25] | +0 (3) [−23] | −5 (4) [−37] |
| ↳ r>1.25 | −9 (6) | −8 (4) | +3 (5) | +4 (8) |
| bucket β(r) · fppg, CRPS-fit — all | −6 (1) | −1 (2) | −1 (1) | −3 (2) |
| flat sal (RG only) — all / bench | +4 (2) / **+23 (3)** | – | – | – |

The **flat 0.18 β hurts**:
- **Promoted rows, by +0.01 CRPS (0.2–0.3 %).** At r > 1.25 the harm is +0.03–0.05 (≈1 %;
  significant in 2024-25). The promoted rows' MAE still improves, so the shift points the right
  way for their mean but is too large, and it costs the upside tail.
- **Demoted (r < 0.8) rows by much more:** +0.02–0.04 CRPS (≈1.5 %) and +0.07–0.12 MAE. Here
  FPPG/L10 push the shift up for players whose minutes really are falling, and onto rows that
  often DNP.
- **Bench and returning rows.**
- **Overall:** +0.008 to +0.012 CRPS on the harness (≥ 2.5 SE).

The salary-implied anchor helps star/role (−0.017 / −0.013) but hurts bench (+0.079 MAE). The
linear line overstates minimum-salary players.

The fitted β(r) for the l10 design is stable across all five folds:
- r < 0.8: **0**
- 0.8–0.95: 0.14–0.18
- 0.95–1.05: 0.12–0.14
- 1.05–1.25: 0.06–0.12
- > 1.25: 0.06–0.08
- returning: **0**
- no history: 0

The fppg variant has the same shape, about 0.02 lower.

## Recommendation for P1 / D1

1. **Replace the flat `TRUTH_BETA·(FPPG − μ)` with a role-aware shift:**

   μ* = μ_played + β(r)·(L10 − μ_played)

   - Use the CRPS-fit bucket table above, walk-forward on seasons < S, refit weekly.
   - Starting values: 0, .16, .12, .09, .07, with returning and no-history = 0.
   - **Anchor: last-10, not FPPG.** It is better on CRPS in 3 of 4 folds and equal in the
     fourth. It also stays sensible at season open, when FPPG is last season's average.
   - Expected gain vs OFF, both ≥ 2 SE in every fold:
     - CRPS: −0.002 to −0.006 (0.05–0.15 %).
     - MAE: −0.006 to −0.012.
   - Vs today's flat 0.18: about −0.010 to −0.015 CRPS on the harness. On promoted rows,
     −0.01 CRPS vs today and not worse than OFF, which meets the B10 bar. No tier is worse than
     OFF by more than about 1 SE (bench +0.002 in 2024-25).
2. **Use μ* everywhere** (D1): display, cash, Top N, candidates and sims. Then drop the
   separate `truth_sims` β step, or set it to 0. Keep `TRUTH_TAU` as is.
3. **Salary** (now approved): add it in P1 as a **star/role-only** feature, or use a
   non-linear salary map with a min-salary floor. Do not use it as a linear anchor. The live
   season-open slope (0.46 ± 0.11) suggests it matters most early in the season. Re-test on
   ≥ 10 graded nights of the +mf arm. **Props:** no history yet, so they are not testable.
4. **Not recommended**:
   - LS-fit bucket β. Its "returning" β ≈ 0.2 hurts out of sample.
   - A flat β tuned on |FPPG − μ|. β is flat in |gap| (0.03–0.14 with no trend).
   - The fppg+l10 joint mix. It is no better than l10 alone.

Caveats:
- The CRPS figures are a quantile-proxy, validated (r = 0.99), not sims.
- Historical "FPPG" is our proxy, not DK's.
- The live gains are MAE-only, on 323 played rows.
