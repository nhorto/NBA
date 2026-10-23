# Tier match (B11, switch `COURTSIDE_TIER_MATCH`, tag `+tm`)

*2026-10-23. Code: `copula.TIER_MATCH`, `copula.draw_tiers`, `copula.fit_style_tiers`,
`pipeline.set_tier_match`. Driver: `research/tier_match.py`. Tests: `tests/test_tier_match.py`.
Default OFF; nothing is committed or shipped.*

## The mismatch

`copula.role_tiers(team, dk, minutes)`: stars = a team's top 2 by `dk` among players with
`minutes >= 28`; role = `minutes >= 20`; the rest bench. The tier picks the correlation row
(`rho_team`) and the CV/phi cells.

| where | `dk` | `minutes` |
|---|---|---|
| fit (`fit_copula`, `v1_frame`, `v1r_frame`) | `r_dk x mpg` (rate state) | `mpg` (season-style state minutes) |
| draw (`simulate_game`, `simulate_game_v1`) | `mu_played` (E[DK \| played]) | `min_played` (E[min \| played]) |

So a player whose planned minutes tonight differ from his `mpg` (promotion, injury absorbers,
a questionable star) sat in another tier at the draw than in the fit that produced his rho.

## Design

Draw side only (the draw-time inputs exist at fit time too: `r_dk` and `mpg` are columns of
`TeamSpec.players`), so no refit and no bundle suffix. ON: `draw_tiers` calls
`role_tiers(team, r_dk x mpg, mpg)` with the fit's inputs. Players who play in no sim (certain
outs, `p_play == 0`) get `mpg = 0` (bench): the fit only sees players who played, and the old draw
did the same (an out player has `min_played = 0`). A questionable player keeps his tier given he
plays. Version `<model version>+tm` (after `+rb`), same bundle and seeds; minutes, means and
reconciliation are untouched (same random draws), only the teammate-correlation matrix and the
CV/phi tier cells of the draw move. The copula v1 draw uses the same helper.

## Mismatch counts (OFF arm, `research/tier_match.py harness`, live arm `+mf+p2+rx`)

Rows = player-games the draw saw, one per (date, player). "Changed" = tier under the old draw
definition differs from the fit definition. `min10` = rows with E[min | plays] >= 10 (the DK-metric
population).

| set | rows | changed, all | changed, P(play) >= .5 | changed, min10 | team-games with a change |
|---|---|---|---|---|---|
| harness 2023-24 | 38,125 | 4,004 (10.5%) | 3,968 / 25,753 (15.4%) | 3,997 / 25,269 (15.8%) | 1,819 / 2,460 |
| harness 2024-25 | 38,409 | 4,831 (12.6%) | 4,783 / 26,144 (18.3%) | 4,828 / 25,198 (19.2%) | 1,935 / 2,456 |
| live 2025-26 (lock-45) | 26,630 | 4,583 (17.2%) | 3,507 / 17,691 (19.8%) | 4,539 / 20,441 (22.2%) | 1,350 / 1,616 |
| live10 2025-26 | 26,624 | 4,750 (17.8%) | 3,627 / 17,650 (20.5%) | 4,689 / 20,165 (23.3%) | 1,376 / 1,616 |
| real day, hist 2024-01-10 | 308 | 40 (13.0%) | 40 / 201 (19.9%) | 40 / 199 (20.1%) | 16 / 20 |
| real slate 154575 (2026-10-20, lock) | 106 | 15 (14.2%) | 15 / 69 (21.7%) | 15 / 70 (21.4%) | 6 / 6 |

Moves among P(play) >= .5 rows (old -> new), 2024-25: star->role 951, star->bench 87,
role->star 526, role->bench 2,418, bench->role 801. The big flow is role->bench: players with
`mpg < 20` that tonight's plan gives >= 20 minutes. Live 2025-26: 504 / 14 / 386 / 1,927 / 676.

**ON: the mismatch count is 0** on every set (`vs_used` = 0 of 26,630 / 26,624 / 38,125 /
38,409 rows, and 0 of 308 and 0 of 106 on the real days). OFF: the draw's tier equals the old
definition on every row (`old_vs_used` = 0).

## OFF identity

- Frame level, 2 real days (`id_driver.py`, live arm, `COURTSIDE_TIER_MATCH` unset): historical
  day 2024-01-10 (10 games, 1,232,000 fp values) and slate 154575 at 2026-10-20T22:15Z
  (2,650,000 fp values): players, stats, all sims, minutes, player ids and meta **identical**
  against a tree whose `copula.py` is `git show HEAD:src/courtside/proj/copula.py` (plus two
  inert constants so `pipeline.py` imports). The capture-hook run is identical to HEAD too.
- Harness level: the four OFF harness frames (live, live10, 2023-24, 2024-25; 130k rows) are
  `assert_frame_equal(check_exact=True)` identical to the `rx` runs of this morning.
- Tests: `tests/test_tier_match.py` (default OFF and env flag, version composes and is model
  version only, the fit definition, OFF = old definition, ON = fit definition with the same
  minutes), plus the existing copula / reserve tests.

## DK CRPS, ON minus OFF (negative = better)

Live arm `COURTSIDE_CONTEXT=1 PPLAY=v2 COPULA=v0 MINFIX=1 RESERVE_EXTRAS=1` (`pathA-v0.3+mf+p2+rx`
vs `+tm`), 4,000 sims, same seeds. Rows with E[min] >= 10 in either arm, DNP = 0. SE 2-way =
`metrics.paired_cluster2` (date x player). Per-row outputs: Mac
`~/Developer/courtside-compute/tm/out/`.

| set | n | dates | players | CRPS OFF | diff | SE 1-way | SE 2-way | z 2-way |
|---|---|---|---|---|---|---|---|---|
| 2023-24 | 21,886 | 160 | 516 | 5.6339 | +0.00002 | 0.00006 | 0.00006 | +0.34 |
| 2024-25 | 22,045 | 162 | 528 | 5.6302 | +0.00004 | 0.00007 | 0.00006 | +0.65 |
| live (2025-26, lock-45) | 15,108 | 104 | 521 | 5.7250 | -0.00007 | 0.00014 | 0.00013 | -0.54 |
| live10 (2025-26) | 14,988 | 104 | 519 | 5.6064 | +0.00007 | 0.00014 | 0.00012 | +0.57 |

By tier (E[min | plays] OFF, z 2-way; star / starter / rotation / bench): 2023-24 +0.90 / +0.37 /
-0.91 / +1.77; 2024-25 +0.66 / +0.57 / -0.26 / +0.60; live -0.44 / -0.95 / +0.72 / +0.33;
live10 -0.44 / +0.39 / +0.84 / +0.74. Rows whose copula tier changed vs not: diff +0.00011 /
-0.00012 (live), +0.00012 / +0.00006 (live10), -0.00001 / +0.00002 (2023-24), +0.00007 /
+0.00003 (2024-25), all |z| < 1.2. The largest single cells are noise-sized (2023-24 changed
bench->role -2.2, role->star +2.0, among the changed-tier cells; 2024-25 role->star -1.7).

**Reading.** Neutral, as it must be: the v0 draw's per-player marginal is a function of
(`mu`, CV) and the copula only couples them, so rho changes move a player's CRPS only through
sampling noise (the "tier unchanged" rows move as much as the changed ones). DK CRPS cannot
see this change. What the tier changes is the *joint* law: teammate correlations (`rho_team` is
-0.02 to -0.12 in the 2025-26 bundle) and so lineup and team-sum variance.

## Sensitivity: first design

A first version also benched players with P(play) < .5. That put a questionable star into the
bench row when he plays (live: 6,295 changed rows, 23.6%); CRPS was equally neutral (live
-0.00013, z -0.77). Dropped for the `p_play == 0` rule above, which matches the old draw.

## Recommendation

Leave OFF (do not ship on this evidence): no measurable gain, and the case rests on
consistency. The fix is free and safe to ship if a lineup backtest (RotoGuru 2019-20 / 2020-21,
`backtest-dfs` cash / Top-20) is at least neutral; that is the only metric that can see it.
