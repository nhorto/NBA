# Lock−10 Questionable P(play) refit (+ a "returning" split), HANDOFF next #2

*2026-10-21/22. Driver: `research/pplay_q_refit.py`. Code: `availability.PPLAYQ` (switch
`COURTSIDE_PPLAYQ=1`, module default `PPLAYQ_DEFAULT = False`), fitted file
`src/courtside/proj/pplay_q_params.json`. Tests: `tests/test_pplay_q_refit.py`. Live model
unchanged (`pathA-v0.3+p2`); the switch is OFF and was not turned on.*

## 0. Answer in short

- **What ran high, and why.** At lock−10 the live path knows the starting lineups (DK starter
  flags), and every flagged player gets P(play) = 0.99. The report cells are fit on *everybody*
  with a status, starters included. So once the starters are lifted to 0.99, the cell is too high
  for the players who are **not** in the lineup. On 2025-26 (correct tip times):
  - Questionable at lock−10: 193 rows, P 0.703 vs 0.601 played (+0.10, SE 0.03).
  - Questionable starters (72): 0.99 vs 1.00. Fine.
  - **Questionable non-starters (121): P 0.532 vs 0.364 played (+0.17, SE 0.04).**
  - The worst cell is a starter-role player (last-10 average ≥ 26 min) who is Questionable and
    not in the lineup: **P 0.63 vs 0.14 played** (n 37). A usual starter missing from the
    lineup almost never plays.
  - The same thing, smaller, for Available non-starters: +0.04 (0.015). For Available bench
    non-starters it is +0.10 (0.03).
- **It is not the lt45m bucket.** With real tip times only 14 Questionable rows a season sit in
  the `lt45m` bucket at lock−10 (21 % played). Lock−10 Questionable rows are mostly in `45m_2h`.
  Teams resolve most Questionable tags in their last report before the tip. Pooling, the prior,
  the report-vs-lock time and "removed" handling are all fine (§2).
- **Slice artifact found.** The p2 slice outputs (`courtside-compute/p2/out`, the "0.66 vs 0.53"
  in HANDOFF) were run with an empty `data/raw`, so there was no ESPN scoreboard archive and no
  tip times. Every game was run at the 19:30 ET default (lock−10 = 19:20 ET for every game). That
  is why that slice had 429 lock−10 Questionable rows instead of 196: for late games it used
  reports 2–3 h before the real tip. The T2.4 slice (`t24/out`) had the archive and is fine.
  `docs/research/return-minutes.md` live+p2 numbers come from the same p2 frames.
- **The fix (A, lineup known).** A logit offset on the cell for listed, not-Out players who are
  not flagged when the team's lineup is out (≥ 3 of its candidates flagged). Groups are status
  (Questionable / Probable+Available / removed) × role. Each offset is shrunk to 0 (= the cell)
  by a ridge prior worth about 16 rows. Shipped offsets: Q starter-role −1.48, Q rotation 0.00,
  Q bench −0.12, P/A bench −0.34, P/A starter −0.46, P/A rotation −0.02, removed −0.25.
- **Ship bar: met** on held-out data, with one small cell to watch (§4):
  - lock−10 listed log loss **−0.0135 (SE 0.0042), −3.2 SE** strict walk-forward;
    **−0.0132 (0.0032), −4.1 SE** leave-one-month-out;
  - DFS CRPS on the T2.4 lock−10 slice **−0.0084 (0.0031), −2.7 SE**, every tier better or
    flat; lock−45 is identical.
- **The returning split (B): not shipped.** On correct tip times a returning Questionable player
  at lock−45 plays 0.547 vs P 0.479 (−0.07, SE 0.04; n 106). The return-minutes 0.68 vs 0.55
  came from the no-tip frames. The split is strong in-sample only for starter-role returners
  (0.91 vs 0.65, n 33). Held out it gains −0.3 to −0.8 SE at lock−45 whichever way it is cut, and
  the rotation cell gets worse. The code path stays, but the shipped file has no (B) offsets.

## 1. Data and protocol

- **Rows.** `availability.build_samples` on the rescued official reports, 2025-12-22 →
  2026-04-12 (regular season; 108 days, 808 games), read-only on the local DB. Training uses
  `bucket` rows, 93,851 (exactly the shipped cells' rows). Evaluation uses `scenario` rows at
  each game's own tip − 10 / − 45 / − 300 min (27.7k each), with tips from the ESPN scoreboard
  archive.
- **Lineups.** At lock−10 the box starters (`player_starts`) stand in for the DK flags, as in
  the T2.4 live10 protocol. Live lock−10 runs do have the flags: on 2026-10-21 the 23:20Z capture
  flagged the 23:00Z and 23:30Z games. At lock−45 there are no flags (the live lock−45 captures
  have none).
- **Added per row.** `started`, `start10` (start share of the last 10 appearances), `missed`
  (`availability.games_missed`: team games since his last appearance, strictly before the day)
  and `role` (last-10 average minutes, the cells' own role).
- **Cells always out of sample.** `fwd` (strict walk-forward) fits on the months before the test
  month and tests on Jan–Apr. `lomo` fits on every other month, which is what the T2.4 slice
  does. The offsets are fit inside the same training months, against those months' own cells.
  The ridge prior (4) was set before testing (sensitivity in §4). `ret_n` for (B) is chosen by
  leave-one-month-out inside the training months.
- **SEs.** Clustered by date throughout.

## 2. Step 1: where the cell runs high (LOMO cells, lock−10 starters at 0.99)

| scenario / status | n | P | played | gap (SE) |
|---|---|---|---|---|
| lock−10 Questionable | 193 | 0.703 | 0.601 | +0.102 (0.031) |
| … started | 72 | 0.990 | 1.000 | −0.010 |
| … **not started** | 121 | 0.532 | 0.364 | **+0.168 (0.044)** |
| … not started, role starter / rotation / bench | 37 / 58 / 26 | 0.63 / 0.55 / 0.36 | 0.14 / 0.55 / 0.27 | +0.50 (0.05) / −0.01 / +0.09 (0.10) |
| lock−10 Available, not started | 887 | 0.804 | 0.761 | +0.043 (0.015) |
| … bench role | 357 | 0.610 | 0.507 | +0.103 (0.034) |
| lock−10 Probable, not started | 26 | 0.911 | 0.808 | +0.10 (0.07) |
| lock−10 removed, not started | 26 | 0.097 | 0.000 | +0.097 |
| lock−45 Questionable | 507 | 0.540 | 0.540 | 0.000 (0.025) |
| lock−5h Questionable | 1,207 | 0.524 | 0.500 | +0.024 (0.017) |

What was ruled out:
- **Bucket.** Lock−10 Questionable rows by bucket: `45m_2h` 179 (P 0.72 vs 0.63), `lt45m` 14
  (0.52 vs 0.21). The `lt45m` Questionable cell has 14 training rows. It is nearly empty because
  the last report before a tip seldom keeps a Questionable tag; it is not the bulk of the problem.
- **Prior / pooling / PAVA.** The same cells are calibrated at lock−45 (0.540 vs 0.540) and
  lock−5h. The excess only appears once the starters are taken out at 0.99.
- **Report used at lock−10.** `team_game_rows` uses the latest report at or before the lock, as
  it should. The excess is the same in every bucket.
- **"removed".** Its cell is ~0.10, and 0 of 26 played. That is small, and it is in the fix.
- **Starter role vs start share.** "Usual starter" by the last-10 start share gives the same
  picture (0.61 vs 0.17, n 46). Role is used because it is already in the frame live;
  `player_starts` is not filled for 2026-27 games.

## 3. Step 2: the fix and the returning split

- **(A) Lineup known.** `q_group` → `ns|q|<role>`, `ns|pa|<role>`, `ns|removed`. Flagged
  starters keep 0.99, unlisted players keep the engine, Out and Doubtful are unchanged.
  - `P = sigmoid(logit(cell) + offset)`. Rows without an offset keep the cell bit for bit.
- **(B) Returning.** `ret|q|<role>` when the lineup is not known and `missed ≥ ret_n`.
- **(B) variants tested (fwd, lock−45 Q log loss Δ):**
  - by role, nested ret_n: −0.0010 (0.0035);
  - pooled over roles: +0.0039 (0.0026) nested, −0.0008 (0.0012) at N = 3;
  - pre-absence ≥ 15 min (return-minutes' own cut): +0.0030 (0.0043) nested, −0.0019 (0.0027)
    at N = 3.
  - The nested N choice flips between 1, 2 and 3 across folds.
  - Lock−5h shows a hint, −0.0042 (0.0020) by role, but lock−45 never clears 1 SE, and the
    rotation cell gets worse (+0.013 (0.011)). **Not shipped.**

## 4. Results (held out)

P(play) on listed players (Q / P / Available / Doubtful / removed / day-to-day). New minus old,
date-clustered SE:

| | n | log loss old → new | Δ (SE) | Brier Δ (SE) |
|---|---|---|---|---|
| **fwd, lock−10, all listed** | 1,795 | 0.2771 → 0.2636 | **−0.0135 (0.0042)** | −0.0044 (0.0019) |
| fwd, lock−10, Questionable | 179 | 0.4835 → 0.4120 | −0.0715 (0.0232) | −0.0327 (0.0105) |
| fwd, all three scenarios | 6,076 | 0.3670 → 0.3630 | −0.0040 (0.0012) | −0.0013 (0.0005) |
| **lomo, lock−10, all listed** | 1,971 | 0.2667 → 0.2535 | **−0.0132 (0.0032)** | −0.0053 (0.0015) |
| lomo, lock−10, Questionable | 193 | 0.4547 → 0.3725 | −0.0823 (0.0234) | −0.0381 (0.0107) |
| lock−45 / lock−5h | | | 0 (no flags: unchanged) | |

Calibration by refit cell, lock−10 (lomo; fwd in brackets): gap old → new, log loss Δ (SE).

| cell | n | played | gap old → new | Δ log loss (SE) |
|---|---|---|---|---|
| Q, starter role | 37 | 0.135 | +0.50 → +0.19 [+0.53 → +0.32] | −0.457 (0.072) [−0.377 (0.072)] |
| Q, rotation | 58 | 0.552 | −0.01 → 0.00 [+0.08 → +0.08] | +0.005 (0.005) [+0.012 (0.010)] |
| Q, bench | 26 | 0.269 | +0.09 → +0.09 [+0.12 → +0.10] | **+0.028 (0.014)** [+0.007 (0.018)] |
| P/A, bench | 363 | 0.507 | +0.11 → +0.02 [+0.12 → 0.00] | −0.026 (0.013) [−0.035 (0.020)] |
| P/A, rotation | 438 | 0.929 | 0.00 → 0.00 [−0.03 → −0.03] | 0.000 [0.000] |
| P/A, starter role | 112 | 0.938 | +0.03 → +0.02 [+0.01 → +0.01] | −0.004 (0.008) [−0.001 (0.005)] |
| removed | 26 | 0.000 | +0.10 → +0.09 [+0.17 → +0.17] | −0.010 (0.002) [−0.010 (0.001)] |
| Questionable non-starters, all | 121 | 0.364 | +0.17 → +0.08 [+0.23 → +0.16] | −0.131 (0.035) [−0.114 (0.034)] |

- **Watch:** the Q-bench cell is +2.0 SE worse under lomo on 26 rows (7 played). That is +0.7
  nats in total, against −17 nats on the Q starter-role cell. Under fwd it is +0.4 SE. Its offset
  is small (−0.12) and changes sign between folds. Read it as noise, but it is the one cell to
  re-check once 2026-27 adds rows.
- **Still high:** the shrinkage leaves Questionable non-starters about +0.08 high (lomo). A
  weaker ridge closes more of that gap but does no better overall.
- **Ridge sensitivity** (fwd, all scenarios, both families): 1 → −0.0045 (0.0018), 2 → −0.0047
  (0.0016), **4 → −0.0048 (0.0014)**, 8 → −0.0038 (0.0012), 16 → −0.0030 (0.0009).

**DFS, T2.4 slice** (`pathA-v0.3+p2`, 104 dates, 4,000 sims, cells and offsets leave-one-month-out,
**correct tip times**; ON vs OFF, paired, common random numbers; Mac scratch
`courtside-compute/pq`, ~5 min per arm):

| lock−10 | n | CRPS OFF → ON | Δ CRPS (SE) | Δ DK MAE (SE) |
|---|---|---|---|---|
| all rows | 26,626 | 3.7295 → 3.7211 | **−0.0084 (0.0031)** | −0.0116 (0.0031) |
| E[min] ≥ 10 in either arm | 14,794 | 5.6037 → 5.5900 | −0.0137 (0.0052) | −0.0179 (0.0053) |
| … bench / role / star | 3,338 / 8,935 / 2,521 | | −0.004 (0.004) / −0.009 (0.005) / **−0.045 (0.014)** | −0.001 / −0.012 / −0.061 |
| teammates in affected games | 5,455 | | +0.0002 (0.0010) | −0.0020 (0.0013) |

- **Slice P(play) at lock−10, non-starters (OFF → ON vs played):** Questionable 0.530 → 0.442
  vs 0.358 (n 123); Available 0.798 → 0.759 vs 0.754 (862); Probable 0.911 → 0.898 vs 0.821.
- **Lock−45:** the ON run is identical to OFF (26,626 rows, every numeric column equal).
- **T2.1 harness (avail):** unaffected. That protocol has no report and no flags, so
  `p_play_slate` is never called.
- **Star tier gain:** it is the Questionable starter-role players who are not in the lineup
  (P 0.63 → ~0.3 vs 0.14 played).

## 5. 2026-27 so far (descriptive, tiny n)

- Only the opener has box scores in the prod DB; 2026-10-21 games are not graded yet.
- Opener main slate 154575, listed not-Out players: 3 at lock−45 (Bona, Wade, Simons; 2 played)
  and 2 at lock−10 (Bona, Simons, both Available, both played).
- PHI's lineup was not flagged in any DK capture before PHI–NYK's lock−10. So under the switch
  nothing would have changed in the real lock−10 run.
- In a whole-day frame where the flags were present, the switch would have cut Bona 0.56 → 0.47
  and Simons 0.87 → 0.80. Both played (10 and 25 min).
- Slate 155316 (2026-10-21 late games): no listed non-starter changes at either lock.

## 6. OFF is bit-identical

`research/pplay_q_refit.py identity`: `p_play_slate` frames (sha1 over ids, P and every column,
plus `np.array_equal` on P) from this tree with the switch OFF vs `git show
HEAD:src/courtside/proj/availability.py`, same DB and params files.
- **2024-01-10, local DB**: every tip group at lock−45 and lock−10 with box starters. All 12
  frames identical. No official reports exist then, so ON changes nothing either.
- **2026-01-15, local DB**: 7 tip groups × 2 locks. All 14 identical. ON changes 0 rows at
  lock−45 and 10–14 rows at lock−10.
- **2026-27 slates 154575 and 155316**, on a backup copy of the Mac prod DB, at each tip's
  lock−45 / lock−10 with the DK starter flags of the time. All 10 frames identical.
  (`var/tmp/pq/identity_local.txt`, `identity_opener.txt`.)
- **Tests** (`tests/test_pplay_q_refit.py`):
  - OFF never calls the refit;
  - ON with an empty params file equals OFF;
  - no flags means no change.

## 7. Reproduce

```
python research/pplay_q_refit.py samples && python research/pplay_q_refit.py diagnose
python research/pplay_q_refit.py evaluate     # var/tmp/pq/evaluate.txt
python research/pplay_q_refit.py fit          # pplay_q_params.json + var/tmp/pq/pq_lomo.json
# slice (Mac scratch, one data dir per arm with the ESPN scoreboard archive in raw/):
COURTSIDE_PPLAYQ=1 PQ_LOMO=pq_lomo.json python research/pplay_q_refit.py slice live10
```

Refit the offsets whenever the report cells are refit (weekly refit item in HANDOFF): they are
offsets *on* the cells. Re-check the Q-bench cell and (B) with 2026-27 report history.

## 8. OOS restatement (B7)

*2026-10-23, `docs/research/b7-oos.md`.* The headline p2-vs-v1 numbers quoted in HANDOFF came from
a slice scored with a fringe "mid" model fit on the slice's own rows. Refit leave-one-month-out
(`research/b7_fringe_oos.py`; seasons < S is impossible, official reports start 2025-12-22):
fringe log loss 0.5279 in sample vs **0.5452 out of sample** (+0.0172, SE 0.0031; n 886). The slice
with OOS fringe params: DK CRPS p2 vs v1 **-6.88 -> -6.58 SE (lock-45), -7.40 -> -7.16 SE (lock-10)**
(two-way -4.79 -> -4.68, -5.03 -> -4.96), no tier worse; p2 stands. The packaged `fringe_params.json`
is unchanged. Side finding: the stored p2 and v1 frames used different minutes bundles; with one
shared bundle the numbers are -6.70/-7.19 (shipped) and -6.40/-6.95 (OOS).

