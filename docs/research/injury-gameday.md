# Official injury rows keyed by game date (B8, `+gd`)

*2026-10-23. Driver: `research/injury_gameday.py`. Code: `availability.INJ_GAMEDAY` (switch
`COURTSIDE_INJ_GAMEDAY=1`, module default `INJ_GAMEDAY_DEFAULT = False`), its own fit
`src/courtside/proj/availability_params_gd.json`; schema 16 (`injury_snapshots.game_date`);
backfill `courtside backfill injuries --game-date`. Tests: `tests/test_inj_gameday.py`.
Live model unchanged; the switch is OFF and was not turned on. IMPROVEMENTS.md item B8.*

## 0. Answer in short

- **The data fix is in and always on.** Migration 16 adds a nullable `game_date` to
  `injury_snapshots` (+ index `(source, game_date, captured_at)`); `collect` and
  `backfill_injuries` write the parser's `game_date` from now on. The backfill re-parses each
  archived PDF and lines its rows up with the DB rows of that `raw_path` (both were written in
  parse order). Dev DB: **2,089 reports, 177,721 / 177,721 rows matched**, all by order, 0
  PDFs missing. Prod snapshot (scratch copy on the Mac): **2,130 reports, 180,429 / 180,429**,
  0 missing (~6 min). 26 % of official rows (46,282) are *next-day* rows; 1,203 of 2,089 batches
  span two game dates; 148 (batch, team, player) triples are a back-to-back listed twice.
- **The keying fix is behind `+gd`.** ON, `_slate` and `build_samples` take batches from
  `load_reports_gd` (keyed by `game_date`; rows never backfilled keep the capture-day key) and
  `load_params()` serves the gd refit. OFF is bit-identical: **62 / 62 real frames identical**
  to the pre-change `availability.py` with `game_date` filled (2026-01-14, 01-15, 03-20, 04-10
  at 10:00 / 13:30 ET and lock−45 / lock−10 of every tip; the two newest live DK slates with the
  DK starter flags of the time), plus the unit tests (1,769 pass).
- **What B8 got right and wrong (measured, §2).** The day-before rows *were* filed under the
  wrong day, but at the 10:00 ET run the OFF path's fallback (`injuries.resolve` → the latest
  official batch ≤ now, ≤ 48 h old) already serves the same 10 PM day-before report for every
  team, with the right `hours_to_tip` (hours are always `captured_at → tip`). So OFF and ON
  price the same statuses for 99.9 % of rows at 10:00. What OFF loses is batch *order* (no
  "removed" from a day-before listing) and back-to-back *separation* (one row per game date
  instead of "most severe wins"): statuses differ on 10 rows at 10:00, 43 at 13:30, 20 / 24 at
  lock−45 / lock−10, over 95–104 days. The 9 % "no report" rows at 13:30 (both arms) are the
  *second-night* teams of back-to-backs: their report is due 1 PM game day and often lands after
  13:30, so no keying can help them.
- **Refit deltas (§3):** every status × bucket cell moves ≤ 0.007 except **`removed`** (n 66 →
  125; 45m_2h 0.069 → 0.153, 2h_6h 0.081 → 0.170) and `unlisted_noreport` (n 152 → 62, 0.693 →
  0.714). The gt6h cells gain the day-before rows (out 5,154 → 5,607, Q 1,155 → 1,313) and
  barely move; day-before Questionable plays 0.557 (n 158) vs 0.497 same-day — no new bucket.
- **P(play) log loss ON vs OFF (§4), out of sample, SE clustered by day:** better everywhere,
  nowhere near a ship-bar z. All candidates, walk-forward: 10:00 **−0.0005 (0.0006)**, 13:30
  **−0.0012 (0.0011)**, lock−45 −0.0006 (0.0007), lock−10 −0.0004 (0.0005). Uncertain
  (D/Q/P): 13:30 −0.0020 (0.0010, −1.9 SE); lock−45 / lock−10 ±0.0001. Leave-one-month-out
  agrees (13:30 −0.0015 (0.0009), −1.7 SE).
- **DFS CRPS slice: not run.** At lock−45 / lock−10 the uncertain-cell P moves < 0.001 and
  statuses differ on ~0.1 % of rows; the slice could not resolve that.
- **Recommendation.** Ship the data part now (done: schema 16 + writers; run the backfill on
  prod). Turning `+gd` ON is safe (OFF-identical, log loss never worse, 'removed' cell fit on
  twice the data, back-to-backs right) but buys correctness rather than a measurable gain; turn
  it on with the next model-version bump rather than as its own release. B8's "row 1" priority
  was overstated: the training population barely changes.

## 1. What was wrong, exactly

The official PDF published at time *t* lists every game of the ET day of *t* **and** of the
next day (from ~5 PM ET). `collect` / `backfill_injuries` stored one row per (team, player,
status) with `captured_at = t` and dropped the parser's `game_date`; `load_reports` keyed
batches by `nba_day(captured_at)`. Consequences, as found in the data:

| | capture-day keying (OFF) | game-date keying (ON) |
|---|---|---|
| next-day rows (26 % of rows) | filed under the capture day | filed under their game |
| 10:00 ET run | no batch for the day → `_fallback_batches`: the one latest official batch (10 PM day-before) via `injuries.resolve` | the day-before 5 PM … 10 PM batches, in order |
| back-to-back player (148 / season in one batch) | one row, most severe status wins, for *both* days | his own row per game date |
| "removed" (listed earlier, dropped later) | only from game-day batches | from day-before batches too (n 66 → 125) |
| `hours_to_tip` | right (captured_at → tip) in both paths | right |
| training rows (`build_samples`) | a b2b team's day-before batch for *tomorrow* attached to *today's* game when published before today's tip | correct |

The 2026-01-14 22:00Z–03:00Z batches B8 cites (BOS/CHI/DEN/… none of which played on 01-14)
are 01-15 rows; they now carry `game_date = 2026-01-15`.

## 2. Frames: what each arm sees (104 RS days 2025-12-22 … 2026-04-12, serving path `_slate`)

Share of candidate rows by source, and the median report age of the batch used (listed rows):

| run | rows | official OFF / ON | prior OFF / ON | removed OFF / ON | median hours OFF / ON |
|---|---|---|---|---|---|
| 10:00 ET | 27,616 | 0.976 / 0.976 | 0.024 / 0.024 | 0.000 / 0.000 | 22 / 22 |
| 13:30 ET | 26,844 | 0.910 / 0.911 | 0.090 / 0.089 | 0.000 / 0.001 | 6 / 6 |
| lock−45 | 27,693 | 0.999 / 0.999 | 0.001 / 0.001 | 0.001 / 0.001 | 1 / 1 |
| lock−10 | 27,705 | 0.999 / 0.999 | 0.001 / 0.001 | 0.001 / 0.002 | 0.5 / 0.5 |

Status transitions OFF → ON at 10:00 (fwd rows): 10 of 25,156 differ (out→available 1,
out→questionable 1, out→unlisted 2, available→unlisted 2, probable→unlisted 1,
unlisted→probable 1, unlisted→removed 2). Rows in one arm only: 27 (26 ON-only, all `removed`
or `out`, none played). The 13:30 "prior" rows (2,380 rows, 156 team-days) were checked by
day: every team is a second-night back-to-back whose first row for the game date is captured
14:00–16:00 ET.

## 3. Refit: `availability_params_gd.json` vs `availability_params.json`

Same window (2025-12-22 … 2026-04-12 RS), same `m1 = m = 20`, same monotone chain; ON bucket
rows 96,208 (OFF 93,851), 808 team-games. Level-2 P(play) (reasons / roles pooled), OFF → ON:

| status | n OFF → ON | lt45m | 45m_2h | 2h_6h | gt6h (n OFF → ON) |
|---|---|---|---|---|---|
| out | 21,707 → 22,151 | 0.005 | 0.005 | 0.005 | 0.005 (5,154 → 5,607) |
| doubtful | 697 → 734 | 0.011 → 0.012 | 0.005 | 0.005 | 0.017 → 0.018 (296 → 333) |
| questionable | 2,879 → 3,037 | 0.390 → 0.391 | 0.539 | 0.521 | 0.497 → 0.504 (1,155 → 1,313) |
| probable | 1,116 → 1,164 | 0.863 | 0.876 | 0.867 → 0.868 | 0.865 → 0.864 (429 → 477) |
| available | 3,465 → 3,486 | 0.876 → 0.877 | 0.876 | 0.867 → 0.868 | 0.865 → 0.864 (458 → 491) |
| removed | 66 → 125 | 0.079 → 0.106 | 0.069 → 0.153 | 0.081 → 0.170 | 0.083 → 0.130 |
| unlisted | 63,769 → 65,449 | 0.830 | 0.827 | 0.824 | 0.821 → 0.820 (16,304 → 17,991) |
| unlisted_noreport | 152 → 62 | 0.693 → 0.714 | | | |

Day-before share of the ON gt6h rows (hours ≥ 12): out 456 / 5,607 (0.002 played), Q 158
(0.557 vs 0.497 same-day), probable 48 (0.854 vs 0.923), available 37, doubtful 37. The
differences are inside noise at these n, so there is no separate day-before bucket.

## 4. P(play) log loss, ON vs OFF

Rows from §2 priced out of sample from each arm's *own* bucket rows: `fwd` = cells fit on the
months before the row's month (Dec rows dropped), `lomo` = every other month. Lock−10 box
starters get 0.99 in both arms (the live DK-flag override). Paired on (day, game, player);
d = ON − OFF, SE clustered by day.

| run | subset | n | played | P OFF / ON | ll OFF | ll ON | d (SE) fwd | d (SE) lomo |
|---|---|---|---|---|---|---|---|---|
| 10:00 | all | 25,156 | 0.619 | 0.637 / 0.637 | 0.4205 | 0.4200 | −0.0005 (0.0006) | −0.0003 (0.0005) |
| 10:00 | uncertain | 1,837 | 0.484 | 0.496 / 0.500 | 0.5357 | 0.5337 | −0.0021 (0.0015) | −0.0009 (0.0010) |
| 13:30 | all | 24,463 | 0.624 | 0.637 / 0.640 | 0.3383 | 0.3370 | −0.0012 (0.0011) | −0.0015 (0.0009) |
| 13:30 | uncertain | 1,853 | 0.507 | 0.527 / 0.530 | 0.4938 | 0.4918 | −0.0020 (0.0010) | −0.0009 (0.0007) |
| lock−45 | all | 25,206 | 0.617 | 0.627 / 0.627 | 0.2758 | 0.2751 | −0.0006 (0.0007) | −0.0002 (0.0007) |
| lock−45 | uncertain | 789 | 0.564 | 0.581 / 0.581 | 0.4581 | 0.4580 | −0.0001 (0.0001) | +0.0001 (0.0000) |
| lock−10 | all | 25,962 | 0.627 | 0.648 / 0.648 | 0.2480 | 0.2476 | −0.0004 (0.0005) | −0.0001 (0.0005) |
| lock−10 | uncertain | 299 | 0.592 | 0.686 / 0.686 | 0.3452 | 0.3449 | −0.0003 (0.0003) | +0.0003 (0.0001) |

(fwd columns n / played / P / ll; lomo n is ~10 % larger since December is kept.) The 13:30
gain is spread over the `out`, `doubtful` and `questionable` cells (−0.002 … −0.004 each, each
≤ 1.5 SE); the 10:00 `available` cell is −0.022 (0.007) on 559 rows, offset by `out` +0.003
(0.002) on 5,195 and `unlisted_noreport` +0.023 (0.012) on 442. The handful of rows whose status
changes (10–43 per run) are too few to score.

## 5. Other readers of `injury_snapshots` (not changed; owners noted)

| reader | what it does | same bug? |
|---|---|---|
| `injuries.resolve` / `current` / `availability` (:78–160, :448) → `predict.py` (`injury_cutoff`, :294, :587) and the Brief's injury lists | per team the newest batch ≤ at (≤ 48 h), rows indexed "most severe wins" | **partly**: no day keying at all, so a back-to-back player's two rows merge (tomorrow's status leaks into tonight's lineup and vice versa); for a single-game team the latest batch is right. `predict.py` is another agent's. |
| `model/players.py:641–651` (`_injury_overrides`-style per-team latest batch) | `MAX(captured_at) ≤ as_of` per team, then `av[pid] = …` row by row | **yes, b2b**: the later (tomorrow's) row overwrites tonight's |
| `injuries.match_report` (:507) | rosters by `nba_day_of(captured_at)` | cosmetic (match rate only) |
| `brief/assemble.py:908`, `web/players_view.py:62` | freshness stamp / cache key | no |
| `mlops/checks.py:195` | "an official report was captured today" | no (it checks capture, which is what it means) |

The fix for the first two is the same `game_date` filter (`WHERE game_date = ? OR game_date IS
NULL`); it is a one-line change in each once `+gd` is accepted.

## 6. Operating it

```
courtside db migrate                                   # schema 16 (db.connect does it too)
courtside backfill injuries --game-date [--dry-run] [--limit N]   # no network; idempotent
COURTSIDE_INJ_GAMEDAY=1 ...                            # serve with game-date keying (+gd)
python -m courtside.proj.availability fit              # OFF: availability_params.json; ON: _gd.json
```

The backfill reads only official rows with `game_date IS NULL`, re-parses their PDF (its own
`raw_path`, else the same day/key under this machine's `COURTSIDE_DATA/raw`), commits per report,
and prints rows matched / unmatched / PDFs missing. Re-running is a no-op. Prod: run it once
after deploying (≈ 6 min for 2,130 reports); the live pregame rows written afterwards already
carry `game_date`.

Research (Mac scratch `~/Developer/courtside-compute/gd/`, prod snapshot via `sqlite3 .backup`,
migrated + backfilled there): `run.sh code/research/injury_gameday.py samples | frames | fit |
evaluate | identity --ref availability_ref.py --day D --slate ID`. Outputs in `gd/out/`
(`frames.parquet`, `T_off/T_on.parquet`, `evaluate.txt`, `identity.txt`).

## 7. Corrections to the B8 text

- Effect 1 ("day-before reports are invisible to the game day's official path; runs before ~1 PM
  fall back to a different code path") is true as a code path but not as an outcome: the fallback
  serves the same latest day-before batch with the right report age. The real losses are batch
  order ("removed") and back-to-back separation.
- Effect 3 ("`hours_to_tip` is wrong for those rows") is wrong for serving; hours come from
  `captured_at → tip` in both paths. It was wrong only for training rows of back-to-back teams.
- Effect 2 (training cells) is right in kind, small in size: ≤ 0.007 on every status × bucket
  cell except `removed`.
- Priority "row 1" should read "data fix now; switch is a correctness cleanup".
