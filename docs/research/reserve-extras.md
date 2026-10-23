# Reserve extras (`+rx`, IMPROVEMENTS.md B2b)

2026-10-23. Switch `COURTSIDE_RESERVE_EXTRAS` (default OFF, `pipeline.set_reserve_extras`,
`minutes.RESERVE_EXTRAS`). Model version `<arm>+rx`, placed before `+rr`. Same bundle, same
seeds. Built on the live arm (`pathA-v0.3+mf+p2`).

**Verdict:** worth shipping. On the live path the reserve is now close to the minutes that
really go to unprojected players: 2.43 -> 0.14 per team-game, actual 0.27. At openers it is
13.1, actual 14.4. DK CRPS improves by 2.4-3.2 SE on live and live10, and no tier is worse by
more than 1 SE. The historical harness is bit-identical, by construction.

With `+rx` in place, `+rr` is almost a no-op mid-season and helps at openers. It still fails the
harness tier check, so it stays off (§4).

## 1. Why the reserve was wrong live

`_fit_reserve` is fit on `oracle_panel`: the last-10 roster plus regulars who played. It learns
the minutes of everybody else.

Box scores 2014-15 to 2025-26 (`research/reserve_extras.py decomp`) split those minutes:
- **Players with box history** take 85-90 % of them: 2.0-3.3 min per team-game, 12-20 at
  openers. Live, the DK pool forces them in as candidates.
- **Career debutants** take the rest: 0.3-0.5 min per team-game, 3-5 at the team-game < 5
  openers, 12-17 at a team's first game. Nobody can project them.

On the 2025-26 live slice the debutants' minutes are 0.271 per team-game. That is exactly the
0.27 that went to unprojected players. So with the pool in the candidates, the right reserve is
the debutants' minutes.

**Design chosen with evidence.** Subtracting the forced extras' expected minutes (option 2 of
the brief) was built first and tested on the live slice:
- The extras' planned minutes are calibrated: 1.12 planned vs 1.25 actual per team-game.
- Taking them out of the reserve moved E[res] only from 2.43 to 1.91. Also counting
  listed-out non-regulars as available, the way the oracle sees them, got it to 1.39.
- The fitted reserve is an average over all team-games, but the extras' minutes are
  concentrated. 39 % of team-games have no extra at all and kept the full reserve. Elsewhere
  the floor at 0 cut off the rest.

So the reserve is **refit with the run-time candidate definition** (option 1 of the brief):
- **Candidates hold the DK pool.** `pipeline.candidate_panel` marks `forced`, and `rx_pool` is
  true when the forced rows cover at least half of the recent roster. The reserve is then
  `pool_reserve(S, team games into the season)`: the mean career-debut minutes per team-game
  by `DEBUT_BINS` {0}, {1-4}, {5-14}, {15+}.
  - It is pooled over the 3 regular seasons before S (`reserve_extras_params.json`, from
    `research/reserve_extras.py fit`).
  - 2026-27 values: 14.47 / 0.81 / 0.20 / 0.17. 2025-26: 12.44 / 0.91 / 0.25 / 0.13.
- **No pool** (report-only `proj day`). The fitted reserve is computed on the fit's own
  candidate set: forced extras are out of its input and absent non-regulars count as
  available. Then the forced extras' planned minutes (p x plan) are subtracted, with a floor
  of 0.
- **Panels without forced rows** are unchanged. That covers the avail oracle and every fit.

Production always projects through `project_slate`, so the DK pool is always in its
candidates.

## 2. OFF is bit-identical

Scratch Mac box (`~/Developer/courtside-compute/rx`). "base" = this tree minus the `+rx`
edits, compared with this tree. Same DB copy and bundles, live arm, switch unset.
- `id_driver`: hist 2024-01-10 and 2020-01-15, and slate 154575 at 2026-10-20T22:15Z. All 3 are
  identical (players, stats, every sim array, meta). Setting `RX=0` explicitly is also
  identical.
- `research/p2_identity.py`: live, live10 and avail 2026-01-15, opener 2026-10-20, and avail
  2024-01-10. All 5 sha1 match.
- With `RX=1`, avail hist 2024-01-10 is identical (no forced rows) and slate 154575 differs.
- Harness rows for `+rx` equal OFF in all 4 avail seasons.
- `tests/test_reserve_extras.py` (10 tests):
  - flag and env parsing, and the version composes as `+rx+rr`;
  - which rows `rx_rows` flags;
  - OFF flags nothing, and an oracle-like panel is unchanged ON;
  - subtraction with a floor of 0;
  - the oracle-consistent input;
  - pool detection and its reserve;
  - the table is walk-forward;
  - the simulation reads the new reserve.

## 3. Results (4,000 sims; ON - OFF, SE clustered by date; OFF = live arm cx + p2 + c0 + mf)

DK CRPS and MAE are on rows with E[min] >= 10. The tier column is the worst DK CRPS z among
star / starter / rotation / bench.

| set | arm | DK CRPS off | ΔCRPS (z) | DK MAE off | ΔMAE (z) | min MAE Δ (z) | worst tier |
|---|---|---|---|---|---|---|---|
| 2025-26 live | +rx | 5.727 | -0.0020 (-2.4) | 8.136 | +0.0023 (+2.4) | -0.0052 (-1.5) | starter +1.0 |
| 2025-26 live | +rx+rr | 5.727 | -0.0020 (-2.6) | 8.136 | +0.0020 (+2.2) | -0.0052 (-1.5) | starter +0.9 |
| 2025-26 live10 | +rx | 5.609 | -0.0024 (-3.2) | 7.953 | +0.0006 (+0.6) | -0.0095 (-3.0) | starter +0.8 |
| 2025-26 live10 | +rx+rr | 5.609 | -0.0024 (-3.4) | 7.953 | +0.0003 (+0.3) | -0.0095 (-3.0) | starter +0.5 |
| 2023-24 harness | +rx | 5.634 | 0 (identical) | 7.988 | 0 | 0 | - |
| 2023-24 harness | +rx+rr | 5.634 | +0.0008 (+1.2) | 7.988 | -0.0025 (-2.4) | 0 | **rotation +2.7** |
| 2024-25 harness | +rx / +rx+rr | 5.630 | 0 / +0.0016 (+1.4) | 7.971 | 0 / -0.0003 (-0.1) | 0 | - / star +1.3 |
| 2019-20 RotoGuru | +rx / +rx+rr | 5.537 | 0 / +0.0011 (+1.1) | 7.842 | 0 / -0.0035 (-2.2) | 0 | - / rotation +1.3 |
| 2020-21 RotoGuru | +rx / +rx+rr | 5.499 | 0 / +0.0024 (+1.7) | 7.777 | 0 / -0.0025 (-1.2) | 0 | - / rotation +1.8 |

On the harness, `+rx+rr` equals `+rr` alone (`docs/research/recon-reserve.md`), because `+rx`
does nothing there.

**Minutes, live.**
- Mean minutes bias goes from -0.13 to +0.08. Rotation goes -0.61 -> -0.43 and bench
  -0.59 -> -0.45.
- Stars (+0.20 -> +0.43) and starters (+0.24 -> +0.49) were already high and get part of the
  freed minutes. Their minutes CRPS worsens by about 3.8 z. Their DK CRPS does not
  (-0.9 / +1.0).
- The cause is how the reconcile allocates the extra minutes, not the reserve. Follow-up below.
- DK MAE rises slightly on live, because every mean moves up and DK is right-skewed (the mirror
  image of `+rr`'s MAE "gain").

### Team sums: projected players' projection - their actual

| set | E[res] off -> on | unprojected min (actual) | DK sum off -> rx / rx+rr | points off -> rx / rx+rr |
|---|---|---|---|---|
| live all (1616) | 2.43 -> 0.14 | 0.27 | -1.23 -> -0.62 / -0.70 | -0.08 -> -0.08 / -0.13 |
| live high reserve (162) | 10.80 -> 0.27 | 0.84 | -3.62 -> -0.61 / -0.72 | -1.19 -> -1.19 / -1.26 |
| live10 all | 2.36 -> 0.14 | 0.27 | -1.15 -> -0.56 / -0.64 | -0.08 -> -0.08 / -0.13 |
| live10 high reserve | 10.63 -> 0.26 | 1.00 | -3.98 -> -1.03 / -1.14 | -1.52 -> -1.52 / -1.58 |
| live openers 2026-27 (28 team-games) | 8.52 -> 13.07 | 14.36 (all debutants) | - | +7.05 -> +7.05 / **+1.07** (rr alone +3.02) |

How to read the table:
- **Points.** They are reconciled to the implied total, so `+rx` alone cannot move them; only
  `+rr` can.
- **Minutes at openers.** The projected players' minutes bias goes from +5.8 to +1.3.
- **Opener spread.** The mean |team error| goes from 13.2 to 14.5. Debutants' minutes range
  from 0 to 60 per team, and a table by team game cannot tell rebuilding teams from contenders.
- **Opener sample.** 28 team-games on two slates (154575 and 155078 at lock - 45, data
  copy + prod box scores read-only), so the opener rows are noisy (SE ~3 points).

## 4. `+rr` on top

- **Live.** It now passes: no tier is worse by more than 1 SE, against +2.1 to +2.2 (rotation)
  with `+rr` alone. The bias damage is gone (points -0.08 -> -0.13 overall, against -1.09
  before).
- **Openers.** The points bias falls from +7.1 to +1.1.
- **Historical harness.** It still fails the tier check (2023-24 rotation +2.7). That path has
  no DK pool, so `+rx` does not change it, and the failure is the one in recon-reserve.md.
  Production never runs that path.
- **Recommendation.** Ship `+rx`. Keep `+rr` off for now. On the production path `+rr` is a
  near no-op mid-season (E[res] x 0.42 ≈ 0.06 points). Its only real effect is at openers,
  which this season's 28 team-games support but cannot prove. Revisit after another opener
  window, or once the opener reserve below exists.

## 5. Follow-ups

- **Rookie-aware opener reserve.** The DK pool lists each team's no-history players, with
  salary. The minutes they are expected to play would replace the table's 14.5 at game 0, and
  would cut the 13-15 mean |team error|.
- **Allocation of freed minutes.** The reconcile CV weighting gives stars and starters part of
  them, though they are already slightly over-projected. That is where the minutes CRPS rise
  comes from (§3).

## 6. Reproduce

```
python research/reserve_extras.py decomp     # debutants vs players with history, by season
python research/reserve_extras.py fit        # src/courtside/proj/reserve_extras_params.json
# Mac ~/Developer/courtside-compute/rx: run.sh (live arm; MF= RX= RR= env), worker.sh over
# jobs/*.job "MF RX RR SEASON PROTO" = reserve_extras.py harness (rows_/teams_/rxf_ parquet in out/),
# lane_dk.sh PROTO:SEASON:LABEL ... = reserve_extras.py dk (logs/dk_*.log, teamsum_*.json, rxflags_*.json)
# identity: lane_id.sh (id_driver hist/slate + research/p2_identity.py, base vs code)
# interim designs (subtract only / + oracle input): out/rxa, out/rxb
```
