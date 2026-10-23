# Ownership uncertainty in the field: K fields (D5)

2026-10-23. IMPROVEMENTS.md D5. Code: `proj/dfs/field.py` (switch `COURTSIDE_FIELD_K`,
`perturb_ownership`, `mean_shift`, `FieldSet`, `build_fields`, `_evaluate_k`,
`run_own_uncertainty`), call sites in `proj/dfs/portfolio.py` (`strength_field`, `score` →
`_score_k`, `_k_bands`, `strength_range`, `realised`, `build_for_slate`, CLI) and
`field.score_run`. Tests `tests/test_field_k.py` (+ golden `tests/fixtures/field_k_off_golden.json`).
Research `research/field_k.py` (`fit`, `pit`, `lines`, `payout`, `bench`). Data: the 2026-10-23
nightly prod copy (geekom1 scratch, deleted after; Mac scratch `~/Developer/courtside-compute/d5/`).
Outputs: geekom1 `var/reports/field_k_20261023/*.json`. Everything is simulated against a
modelled field except the real standings of 2026-10-20 (slate 154575, one 3-game night).

## Plain-language summary

- Today the field is built from one best guess of who the public will play. On 10-20 that guess
  was far off for some players: Harper was 10-15 % in our proxy and 66-93 % in reality. With the
  switch ON, the builder and the slate page score against 4 fields instead of 1. Each field uses
  a different random version of the ownership guess. The noise is sized to how wrong the guess
  really was, and each player keeps the same expected ownership.
- On 10-20 the ON fields look more like the real contests than the single field does. They are
  as bunched up as the real fields, and their top-1 % scores sit closer to the real ones. Real double-ups on the same night had cash lines 10 points apart. ON fields differ from
  each other by about 8 points; OFF fields differ by under 1.
- Effect on the numbers: P(cash) barely moves (within 2 points). P(top 1 %) rises (+7 to +50 %)
  and big-GPP E[payout] rises (+19 to +31 %). The fields' best lineups score lower when ownership
  is lumpier, so first place is easier to reach. Bands now show the spread across the 4 fields:
  about ±5 points of P(cash) in a double-up.
- Where our lineups' real results fell in the simulated range (the PIT) is the same ON and OFF.
  One night cannot show more.
- Cost: +5 s at N = 20 (+12 %), +1 s at N = 150 (+1 %), on the Mac. No cheaper scheme needed.
- Recommendation: OK to turn ON (owner decision). Refit the noise as uploads come in.

## 1. Design

Switch `COURTSIDE_FIELD_K`, read once at import (`set_field_k` changes it at run time):
- Default `0` = OFF.
- `1` / `on` = `FIELD_K_ON` = 4 fields.
- An integer >= 2 = that many fields.

The version tag is `field-proxy-1.2+k`, on `score_run`'s `field_version`, the builder's
`info.field.version` and the per-field ownership versions. OFF runs none of the new code.

ON, per contest (`build_fields`):

1. **Ownership draw.** The usual proxy `own` (at the contest's archetype) is drawn K times:

       logit(own'_i) = logit(own_i) + c_i + sd_a * u_i * z_i,   z_i ~ N(0, 1)

   - Ownership is clipped to [1e-4, 1 - 1e-4]; players with 0 stay at 0.
   - `sd_a` = `OWN_SD[archetype]` (section 2).
   - `u_i` = `late_news_tag.own_uncertainty`: 2.0 for late-news beneficiaries, else 1.0.
   - `c_i` = `mean_shift`, solved by Gauss-Hermite quadrature and bisection so that
     E[own'_i] = own_i.

   Each player's per-slot row is scaled by own'/own. `sinkhorn` then re-balances the draw to the
   slot totals (800 %) and the row cap. `perturb_ownership` does this.
2. **Fields.** Each field is `build_field` on its own draw, with seed + j.
   - The ownership-sampled part follows the draw through the existing IPF.
   - The optimizer-like solves (200 CP-SAT solves on the public projection) are made once and
     shared (`build_field(opt_pairs=)`, `Field.opt_solves`). They depend on the projection, not
     on ownership.
3. **Scoring** (`evaluate` on a `FieldSet` → `_evaluate_k`).
   - Sim row s is scored against field s mod K. The pieces are put back in row order, so the
     cash and top-1 % lines and the per-sim score / above / tied keep their shapes.
   - Every chance or payout averaged over rows is then the average over the K fields. Scoring
     costs the same as one field.
   - Per lineup it also adds `*_k` (each field's value) and `*_lo` / `*_hi` (min / max) for
     P(cash), P(top 1 %) and E[payout], plus `dupes_k`.
   - With fewer rows than fields (one row of actual points), every field scores every row and
     the results are averaged.
4. **Builder.** `strength_field` returns a `FieldSet`. `score()` adds the per-field dupe Poisson
   to each field's own rows (`_score_k`). The greedy, curve, baselines and strengths need no
   change, because they work on the per-sim rows. Display fields (`_k_bands`):
   - each lineup's `p_cash_lo/hi` and `p_top1_lo/hi`, from the joint ranks on each field's
     report rows;
   - `summary.field_k`: P(any cash), P(any top 1 %) and E[payout], per field and lo / hi.

   `op` (exposure table, ownership penalty) stays the un-noised proxy. `realised` averages over
   the fields.
5. **Slate page** (`score_run`). Both reference contests use K fields. The stored columns
   (`p_cash`, `p_top1`, `e_payout`, `dupes`) hold the means. `detail.field.per_lineup` adds the
   lo / hi bands, and `detail.field.field_k` holds K and the flagged players.
6. **Late news.** `run_own_uncertainty(conn, run_id, slate_id=)` returns the stored run's tag.
   The runner scores the field before it saves the run row, so for an unsaved run the tag is
   computed as of now. This needs no `runner.py` edit, and the tag still runs after
   `score_run` as `test_late_news` requires. `build_for_slate` and the CLI pass the run's tag
   to `build_portfolio(own_unc=)`.

**Display.** The bands are in the stored JSON only. No template renders them yet.

## 2. Noise fit (`research/field_k.py fit`)

**Method.** The lock-45 run c15398ddd87f's inputs give the proxy at each archetype. For each sd
on a 0.05 grid from 0.10 to 3.00:
1. Make 60 draws (`perturb_ownership`, centred, Sinkhorn).
2. Score proxy vs draw with exactly the metric used for proxy vs real
   (`real_standings_calib.own_metrics`): players above 0.1 % in either; top 10 = by the "real"
   value, here the draw.
3. Take the mean metric per sd. The sd is then interpolated where the mean equals the real
   value.

Targets, as in D5:

| | real (proxy vs %Drafted) | fitted sd | implied at the fitted sd |
|---|---|---|---|
| large GPP (4PP, 29.7k) | MAE 4.57 pp | **0.86** | top-10 MAE 13.9 (real 12.6), top-10 bias -11.2 (-12.3), rank corr 0.92 (0.87) |
| cash (3 double-ups, mean) | top-10 MAE 27.4 pp | **2.65** | MAE 10.8 (real 10.0), top-10 bias -23.7 (-24.9), rank corr 0.79 (0.77) |

- small GPP has no data, so it uses the large-GPP value.
- **Cross-checks.**
  - The direct log-odds residual sd (logit real - logit proxy, players > 0.1 %) is 1.04 for the
    GPP and 1.5 / 1.7 / 2.0 for the double-ups.
  - The cash sd sits about where the slate's measured concentration exponent predicts:
    double-up ≈ GPP^k with k = 2.05-3.0, so k × 0.86 = 1.8-2.6.
  - Fitting cash on its overall MAE instead would give 2.29.
- **Centring matters.** With uncentred noise (an earlier run), mean ownership moved from the
  chalk to the tail, and the double-up field got weaker on average: P(cash) 0.54 → 0.60 and the
  simulated cash line p50 269.6 → 264.8. That is a weaker field, not an uncertain one. Centred:
  P(cash) 0.54 → 0.54 and p50 269.6 → 269.5.
- **What the sampled field realises.** Realised-ownership MAE vs the proxy, for 4 fields at the
  fitted sd:
  - large GPP: 3.6-4.6 pp, for draws of 3.7-4.9 (about 95 % realised);
  - cash: 5.7-8.5, for draws of 9.6-11.9 (about 70 %). The 40 % optimizer part is shared and
    does not follow the draw.

  The cash sd was not inflated to make up for this.
- **The draws vs reality, per player** (PIT of real %Drafted inside 2,000 draws; in-sample,
  because the sd was fitted on the same data). 90 % intervals cover 66 / 70 / 85 % of players in
  the double-ups and 78 % in the GPP. The tails are heavier than log-normal.
  - Harper (proxy 0.10 cash / 0.15 GPP) lands outside the draws' 95th percentile everywhere:
    0.55 vs 0.90-0.93 real, and 0.37 vs 0.66.
  - Tatum is at the 97-99.5th percentile.

  He had no late-news flag (that run's tag is empty), so the ×2 multiplier did not apply.

## 3. OFF is bit-identical

- **Golden** (`test_off_bit_identical_to_pre_change_golden`, Linux only). The golden file was
  written by `python tests/test_field_k.py golden <field.py at 3c40fbf>`, on the pre-change
  module loaded in place. It covers:
  - ownership (cash, large GPP) incl. per-slot sums and λ;
  - two fields' lineups, counts, optimizer counts and dupe constant, and their summary;
  - `evaluate` per-lineup metrics, the cash / top-1 % / tier lines and per-sim score / above.

  The current module reproduces it exactly (floats by repr).
- **Builder.** `tests/test_portfolio_split.py`'s OFF golden (the pre-split builder) still passes.
- **Platform-free tests:**
  - an OFF `field_for` returns a plain `Field` with no `+k` and no `*_k` keys;
  - an OFF build ignores `own_unc` and has no `field_k` / band keys;
  - `build_field(opt_pairs=)` reuses the solves.

## 4. Runtime (Mac mini, 2 threads, `nice 10`, web settings: strengths on, curve, `gs` split, 25k sims)

Slate 155078 (11 games), lock-45 run d514cb25b711, $100K Fadeaway (7,843 entries, max 150),
one build each (`research/field_k.py bench`):

| N | OFF s | ON s | ON - OFF | field OFF / ON | strengths OFF / ON |
|---|---|---|---|---|---|
| 20 | 39.8 | 44.7 | +4.9 (+12 %) | 6.2 / 7.8 | 14.2 / 17.4 |
| 150 | 84.9 | 86.1 | +1.2 (+1 %) | 6.3 / 8.6 | 16.1 / 17.5 |

- Building the K fields adds about 2 s per strength: 4 samplings, one set of solves. Scoring is
  unchanged (row-wise).
- The Mac was lightly loaded. The "~200 s" for N = 150 in IMPROVEMENTS came from a loaded box.
  Both are well inside §5.8 (N = 20 ≤ 120 s, N = 150 ≤ 300 s).
- A cheaper scheme is not needed. If it ever were, the soft / sharp strength fields could stay
  single, which saves about 1-3 s.

## 5. 10-20 against reality

### 5.1 Field at the actual box scores (`lines`, 3 seeds; ON = 12 fields, OFF = 3)

Σ own² = concentration of real %Drafted / realised field ownership. "1st" = mean max over 20
draws of M entries.

| contest | real cash / top 1 % / 1st / Σ own² | OFF | ON (between-field sd) |
|---|---|---|---|
| $2 DU 287 | 237.3 / 283.5 / 290.3 / 5.09 | 248.1 / 285.7 / 293.1 / 3.20 | 245.8 (7.9) / 284.0 (2.8) / 287.4 / 4.19 (0.46) |
| SE $5 DU 459 | 238.8 / 283.5 / 293.0 / 4.21 | 248.0 / 285.7 / 296.4 / 3.20 | 245.8 / 284.0 / 288.5 / 4.19 |
| $2 DU 1,149 | 247.8 / 283.5 / 292.3 / 4.73 | 248.1 / 285.7 / 301.6 / 3.20 | 245.9 / 284.0 / 290.2 / 4.19 |
| 4PP 29.7k | 253.3 / 290.5 / 320.3 / 2.43 | 254.4 / 293.8 / 318.0 / 2.06 | 251.7 (5.8) / 291.1 (4.5) / 314.1 / 2.56 (0.16) |

- ON matches the real concentration: double-ups 4.2 vs 4.2-5.1 real (OFF 3.2); GPP 2.56 vs
  2.43 (OFF 2.06).
- ON's top-1 % lines move toward the real ones by 1.7-2.7 points.
- Winners are mixed. ON is closer on 1 of 3 double-ups, but its mean absolute error is lower
  (3.2 vs 5.2 points). The GPP winner is 6 points low (OFF 2).
- The real double-ups' cash lines that night spread 237-248 (sd 5.7). ON fields differ from
  each other by 7.9; OFF seeds differ by 0.8.

### 5.2 Chances and PIT of our 50 lineups (`pit`, 3 seeds, 10k sims, calibrated truth)

PIT = where the night's (lineup - real cash line) fell in the simulated (lineup - cash line)
distribution.

| contest | realised cashed | P(cash) OFF → ON (ON band) | P(top 1 %) OFF → ON | PIT median OFF → ON | PIT mean OFF → ON | share in [0.25, 0.75] OFF → ON |
|---|---|---|---|---|---|---|
| $2 DU 287 | 0.60 | 0.541 → 0.537 (0.48-0.59) | 0.032 → 0.049 | 0.516 → 0.516 | 0.543 → 0.544 | 0.74 → 0.76 |
| SE $5 DU 459 | 0.54 | 0.541 → 0.538 (0.48-0.59) | 0.034 → 0.050 | 0.493 → 0.494 | 0.523 → 0.525 | 0.72 → 0.78 |
| $2 DU 1,149 | 0.30 | 0.541 → 0.538 (0.48-0.59) | 0.033 → 0.050 | 0.346 → 0.359 | 0.407 → 0.415 | 0.68 → 0.74 |
| 4PP (top 25 %) | 0.30 | 0.498 → 0.501 (0.47-0.52) | 0.034 → 0.042 | 0.325 → 0.324 | 0.381 → 0.381 | 0.68 → 0.70 |

- Pooled over the 200 lineup-contest pairs: PIT mean 0.464 → 0.466, sd 0.219 → 0.210 (uniform:
  0.289).
- The simulated (lineup - cash line) sd widens by 4-8 % (26.5 → 28.6 in the double-ups).
- Brier score for P(cash) vs cashed is unchanged (0.224-0.254 both).
- **What one night says.** ON is not more uniform. It is the same, slightly pulled toward the
  middle, because the simulated spread is a little wider. All 50 lineups share one night, so
  their PITs are one correlated draw. Uniformity can only be judged across slates (repeat `pit`
  per uploaded slate). Nothing here is significant.
- The across-field P(cash) band (about ±5 points in double-ups, ±3 in the GPP) is mostly field
  spread. At 2,500 rows per field, sim noise alone gives about ±1.

### 5.3 Why E[payout] rises in a big GPP (`payout`; 155078, $100K Fadeaway)

The OFF N = 20 set was scored on the same truth rows against four arms:

| | E[payout] ± SE | P(any 1st) | P(any top 1 %) | mean P(cash) | top-1 % line p50 |
|---|---|---|---|---|---|
| one field (OFF) | 3,301 ± 51 | 0.075 | 0.636 | 0.622 | 342.2 |
| K = 4, sd 0 (seeds only) | 3,259 ± 51 | 0.075 | 0.636 | 0.622 | 342.2 |
| K = 4, fitted sd | 4,320 ± 61 | 0.107 | 0.648 | 0.611 | 340.9 |
| K = 4, 2 × sd | 5,890 ± 74 | 0.157 | 0.678 | 0.614 | 338.5 |

- The rise comes from the noise, not from the extra fields or seeds.
- Every one of the 4 fields individually beats the OFF value. So this is not averaging over
  fields smoothing a convex payout. A drawn ownership vector is lumpier than the smooth proxy:
  Σ own² rises from 0.82 to 0.95-1.47, and the same players appear together more often. With
  fewer distinct lineups near the top, the field's best score drops by 3-5.5 points (mean max
  388.1 → 382.6-385.2 over 4,000 sims), and first place is easier.
- Section 5.1 shows real fields are lumpier than the proxy in the same way. So the OFF field
  likely overstates the top of a real GPP, though one slate cannot pin by how much.
- The builder headline is the relative score against the Top-N-by-mean set on the same field,
  and dollar E[payout] stays in the "uncalibrated" details until 10 uploaded slates. The level
  shift therefore does not reach a headline.
- Builds: N = 20 E[payout] 3,328 → 4,360, P(any cash) 0.999 → 0.998, mean P(cash) 0.623 →
  0.603. N = 150: 13,366 → 15,840, mean P(cash) 0.538 → 0.525.

## 6. Recommendation and follow-ups

- **OK to turn ON** (owner decision): set `field.FIELD_K_DEFAULT = 4`, or
  `COURTSIDE_FIELD_K=1` on the Mac jobs and web. Roll back with `COURTSIDE_FIELD_K=0`.
  - For: the cost is small; on the one uploaded night the field's structure (concentration, top
    lines, spread between same-night contests) moves toward reality; P(cash) is unchanged.
  - Against: the sd values come from one 3-game slate. The cash sd (2.65) is large because 10-20
    had extreme double-up misses (Harper, Tatum). And GPP top-1 % / E[payout] move up by 7-31 %.
- Refit `OWN_SD` per uploaded slate (`research/field_k.py fit --slate S --run R`), alongside
  `courtside dfs field-calib`, and re-run `pit` and `lines`. Uniformity of the PIT needs about 10
  slates.
- **Not tested:** the late-news ×2 multiplier. 10-20's lock-45 run has no flags. It is unit
  tested only.
- **D2 (`cash_pcash`) interplay.** `cash_pcash.field_kw` calls `run_own_uncertainty(conn,
  run_id)` without `slate_id`, so for the runner's not-yet-saved run it gets {}. `score_run`
  passes `slate_id` and gets the computed tag. When a run has late-news flags and both switches
  are ON, the reuse key then differs, and `score_run` builds its own cash field. That is
  correct, just one more build. The fix is to pass `slate_id=` in `field_kw` (D2's file).
- **UI:** render `p_cash_lo/hi` / `summary.field_k` as "P(cash) range across ownership
  scenarios" (display only).
