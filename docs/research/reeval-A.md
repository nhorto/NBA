# Re-evaluation A: availability and minutes (H6, H7)

*2026-10-25, workstream A of `docs/REEVAL_2026-10.md`. Nothing here is shipped: three new switches
are built OFF (bit-identical), the backtests are written and validated on small samples, and
the full runs wait for the Mac tonight.*

## 0. Headline

- **The "71 surprise DNPs vs ~50 expected" gap is not a calibration problem.** The 21-DNP
  excess comes from **ghost candidates**. These are players from last April's end-of-season
  games (two-way, 10-day and tank-minutes players, e.g. Okani, Williamson and Bal at MEM, Mbeng
  at UTA). They are on no 2026-27 roster and not in the DK pool, yet the engine still treats
  them as unlisted regulars. On the scheduled runs, the rest of the P ≥ 0.7 population has
  **39 DNPs against 46.0 expected**, which is slightly under-confident.
- **The ghosts also distort minutes.** Over 191 rows they carry P(play) 0.33 and
  **11.8 expected minutes per team-game**, and none of them played. Those minutes are taken
  from the real players. In-pool players' E[min] sums to 229.7 per team-game against 238.7
  actual (2-night replay).
- **Two smaller availability holes:**
  - The report matcher only knows players with a box row in the last 300 days. A long-absent
    player listed Out (Lively, Beal) is silently dropped from the report. He then comes back
    through the DK pool as a "fringe" player and gets P 0.5–0.9.
  - DK's 'O' status is not used. 149 DK-'O' rows, none played. 19 of them were fringe rows
    projected at P 0.48.
- **Minutes MAE matches the backtest.** Live played E[min | plays] ≥ 15: **4.56 (game-clustered
  SE 0.18)**, bias −0.03, n 595. The T2.4 live-like slice is 4.457 and its daily p50 is 4.42.
  The largest structured error is at **lock-10 with lineups known**:
  - unexpected starters (DK-flagged, projected < 26 min): **−3.0 min (SE 0.9, n 66)**;
  - expected starters left out of the lineup: **+3.4 (SE 1.0, n 61)**;
  - flagged regular starters: −1.0 (SE 0.4), most of it ghost-induced.
- **About 60% of the DK error is the per-minute rate, not minutes or availability.** DK MAE
  ladder (grader population, n 664):
  - as projected: 8.71;
  - with P(play) known: 7.41;
  - with P(play) and minutes known: 5.55.

  On played rows, the minutes part (minutes error × projected DK/min) is 35% of the error
  variance and the rate part is 60%.
- **Fixes built (all OFF):**
  - `+gh` (`COURTSIDE_GHOST_DROP`): drop ghost candidates;
  - `+rm` (`COURTSIDE_REPORT_MATCH`): match report names on the current roster, and treat
    DK 'O' as out when the report is silent;
  - `+lu` (`COURTSIDE_LINEUP`): lineup-known demotion and lift weights, fit on seasons < S,
    plus P 0.99 for confirmed starters off the report.

## 1. Data and population

- **Source:** geekom1 `var/courtside.db`, a read-only snapshot of prod from 2026-10-25 05:42 ET
  (`mode=ro`).
- **Population:** NBA days 2026-10-20 → 10-24: 36 games, 5 nights. For each (game, player), the
  last pre-tip projection is the newest `player_projections` row whose `dfs_runs` run is ok and
  was created before `games.tipoff_utc`. That gives 1,343 graded rows:
  - by run kind: 1,064 lock10, 194 swap, 70 lock45, 15 manual;
  - by model version:
    - `+p2` 464;
    - `+mf+p2+rx+rb` 450;
    - `+mf+ep+ms+p2+pq+rx+rb+tr` 199;
    - `+mf+ep+ms+p2+rx+rb+tr` 109;
    - v0.2 106;
    - v0.1 15.

  The live model changed under the week, so per-version cells are tiny. The paired replay in
  §4 re-runs every night on today's stack.
- **Status at run time:** rebuilt with the live code itself (`availability.p_play_slate` at the
  run's `created_at`, with that slate's DK starter flags at or before the run).
  - "fringe" means projected but in neither the report frame nor the team's last 10 games.
  - "Ghost" means not in the slate's DK pool and on no `team_rosters` 2026-27 team.
- **Script:** `research/reeval_a_live_diag.py frame|report [--kinds lock45|lock10]`. It is pure
  SQL / pandas and takes about 15 s on geekom1. Output goes to `var/tmp/reeval-a/live_diag*.md`.

## 2. P(play) calibration (H6)

### 2.1 By bin (all 1,343 rows)

| P bin | n | mean P | played | exp DNP | act DNP | excess |
|---|---|---|---|---|---|---|
| < 0.02 | 153 | 0.00 | 0.00 | 152.5 | 153 | +0.5 |
| 0.02–0.1 | 72 | 0.06 | 0.15 | 68.0 | 61 | −7.0 |
| 0.1–0.3 | 171 | 0.20 | 0.22 | 136.6 | 134 | −2.6 |
| 0.3–0.5 | 128 | 0.40 | 0.33 | 77.0 | 86 | +9.0 |
| 0.5–0.7 | 138 | 0.60 | 0.51 | 55.8 | 67 | +11.2 |
| 0.7–0.85 | 143 | 0.77 | 0.73 | 32.2 | 39 | +6.8 |
| 0.85–0.95 | 118 | 0.91 | 0.88 | 11.0 | 14 | +3.0 |
| 0.95–0.99 | 224 | 0.98 | 0.94 | 5.3 | 14 | +8.7 |
| ≥ 0.99 | 196 | 0.99 | 0.99 | 1.2 | 1 | −0.2 |

Without the 191 ghost rows, the 0.3–0.7 bins flip to slightly under-confident: 0.40 vs 0.49
and 0.60 vs 0.61 played.

### 2.2 Where the P ≥ 0.7 excess comes from

| cell | n | mean P | played | exp DNP | act DNP | excess |
|---|---|---|---|---|---|---|
| **all P ≥ 0.7** | 681 | 0.93 | 0.90 | 49.6 | 68 | **+18.4** |
| ghost candidates (no DK pool entry, no 2026-27 roster) | 24 | 0.88 | 0.00 | 2.8 | 24 | **+21.2** |
| stale manual runs (10-20 opener, pathA-v0.1, old rosters) | 8 | 0.90 | 0.38 | 0.8 | 5 | +4.2 |
| **non-ghost, scheduled runs** | 649 | 0.93 | 0.94 | 46.0 | 39 | **−7.0** |
| · DK starter flag | 274 | 0.98 | 1.00 | 6.4 | 0 | −6.4 |
| · fringe pool player | 40 | 0.86 | 0.72 | 5.4 | 11 | +5.6 |
| · · of which DK status OUT | 5 | 0.79 | 0.00 | 1.0 | 5 | +4.0 |
| · listed on the report (not Out) | 21 | 0.89 | 0.76 | 2.4 | 5 | +2.6 |
| · unlisted regular, appeared this season | 177 | 0.90 | 0.98 | 16.8 | 4 | −12.8 |
| · unlisted regular, team opener | 127 | 0.90 | 0.91 | 13.0 | 11 | −2.0 |
| · unlisted regular, sat all his team's games so far | 10 | 0.81 | 0.20 | 1.9 | 8 | +6.1 |

**Causes of the 68 P ≥ 0.7 DNPs:**
- 57 are unlisted players who never appeared on any report, i.e. DNP-coach's-decision in the
  data. 24 of those are ghosts.
- 11 are fringe pool players who were not used.
- **Late scratches: 0.** No P ≥ 0.7 player was Out on the final pre-tip report after not being
  Out at the run.
- **G League / two-way:** no report row carried a G-League reason for these players. In
  practice the two-way problem is the ghost problem (last season's two-way players), plus a
  few fringe rows.

### 2.3 By listed status and hours to tip

By lock-45 and lock-10 on game day the report has nearly resolved.

| run | status | n | mean P | played |
|---|---|---|---|---|
| lock45 | Available | 32 | 0.83 | 0.78 |
| lock45 | Probable | 4 | 0.95 | 0.50 |
| lock45 | Questionable | 5 | 0.45 | 0.60 |
| lock45 | Doubtful | 1 | 0.01 | 0.00 |
| lock45 | Out | 135 | 0.00 | 0.00 |
| lock10 | Available | 39 | 0.83 | 0.77 |
| lock10 | Out | 130 | 0.00 | 0.00 |
| both | ESPN Day-to-Day (GTD) | 0 at run time | | |

- Every listed row at lock-45 is in the 0.75–2 h bucket, and every row at lock-10 in the
  < 0.75 h bucket.
- By hours from run to tip:

  | run to tip | n | mean P | played |
  |---|---|---|---|
  | ≤ 15 min | 1,178 | 0.60 | 0.59 |
  | 15–60 min | 150 | 0.61 | 0.52 |
  | > 8 h (manual runs) | 15 | 0.62 | 0.27 |

- **Verdict on the report cells:** listed rows sum to 145.9 expected DNPs against 148 actual.
  - With ~170 listed rows, five nights add almost nothing to the 93,851 training rows.
  - **A scheduled refit is not where the live gap is.** It is still worth scheduling as a
    monthly no-op guard. The refit already exists: `availability.fit(conn, lo, hi)` plus the
    `+gd` refit.
  - The hierarchy-vs-simpler-cells question cannot be told apart on this sample.

### 2.4 The engine's P(plays | available) for unlisted regulars

Excluding ghosts, the engine is under-confident for players who have appeared this season (0.90
vs 0.98, −12.8 DNPs) and over-confident for the few who sat all their team's games (0.81 vs
0.20, n 10). Both are early-season effects: the "last 10 team games" window still holds last
April. This is **not built**:
- the 10-row cell is too small;
- the historical analogue (the first games of past seasons) needs the minutes harness to
  separate it from the ghost fix.

It is listed as the next candidate in §5.

## 3. Minutes and DK error (H7)

### 3.1 Minutes MAE

Played rows, E[min | plays] ≥ 15:

| cut | n | MAE | bias (proj − actual) |
|---|---|---|---|
| **all** | 595 | **4.56** (SE 0.18) | −0.03 |
| T2.4 live-like slice (lock-45, 2025-26) | | 4.457 (daily p50 4.42, p90 5.26) | |
| all played rows, any E[min] | 774 | 4.80 | |
| overtime (1 game) | 17 | 5.80 | −3.78 |
| blowout, final margin ≥ 20 (8 games) | 130 | 4.99 | **+2.93** |
| not a blowout | 465 | 4.44 | −0.85 |
| new team (traded or signed) | 136 | 4.74 | +0.07 |
| rookie | 18 | 6.38 | −1.51 |
| team opener (no games yet) | 239 | 4.48 | +1.04 |
| has appeared this season | 348 | 4.59 | −0.82 |
| first game back after sitting all team games | 8 | 5.44 | +2.55 |
| team had 0 / 1 / 2 / 3+ players Out | 105 / 134 / 138 / 218 | 4.09 / 3.80 / 4.88 / 5.04 | −0.36 / −0.18 / −0.85 / +0.75 |
| teammates had 0 / 1 / 2 / 3+ surprise DNPs (P ≥ 0.7) | 369 / 134 / 64 / 28 | 4.47 / 4.72 / 4.76 / 4.5 | +0.49 / −0.15 / −0.86 / −4.4 |

By tier (E[min | plays]; played rows):

| tier | n | MAE | bias |
|---|---|---|---|
| bench < 12 | 112 | 5.37 | +0.28 |
| 12–20 | 194 | 5.25 | +0.06 |
| 20–26 | 174 | 5.04 | +0.14 |
| 26–32 | 217 | 4.41 | −0.14 |
| star ≥ 32 | 77 | 3.42 | +0.51 |

By margin, the bias runs −1.5 at 0–5 points, +2.1 at 20–30 and +4.2 at 30+. That is the expected
shape for an E[min] averaged over margins: the sims draw margin-conditional minutes, and `+bo`
already tested the curve (fails the bar). Stars and starters in blowouts are +3.3 to +6.2 (n 65),
the same "minutes do not fall enough in blowouts" gap as the `+bo` research. It is not
re-litigated here.

**Lineup known (lock-10 / swap; played):**

| cell | n | bias | SE | MAE |
|---|---|---|---|---|
| DK-flagged, projected < 26 min (unexpected starter) | 66 | **−3.00** | 0.90 | 5.95 |
| DK-flagged, projected ≥ 26 | 215 | −0.98 | 0.43 | 3.85 |
| not flagged, projected ≥ 26 (left out of lineup) | 61 | **+3.42** | 0.98 | 5.11 |
| not flagged, < 26 | 390 | +0.78 | 0.35 | 4.97 |

Flagged-starter bias by team ghost minutes:
- 0 ghost minutes: +1.4 (n 30);
- 0.5–8: −1.5 (n 140);
- > 8: −2.1 (n 111).

So most of the −1.0 is the ghost artefact. The unexpected-starter and left-out cells are not.

Signed-error OLS (played ≥ 15, game-clustered SEs):

| term | minutes | SE |
|---|---|---|
| OT × starter-level | −3.8 | 0.3 |
| OT × other | −1.4 | 0.2 |
| blowout × starter-level | +3.8 | 0.9 |
| blowout × other | +2.0 | 0.9 |
| first game of the season | +1.4 | 0.5 |
| new team | −0.1 | 0.5 |
| teammate surprise DNP | −1.4 | 0.5 |

**Biggest misses (E[min | plays] → actual):**

| player (team, day) | projected → actual | likely cause |
|---|---|---|
| Cody Williams (MIN, 10-21) | 26.2 → 3 | new team, opener, rotation churn; 25-pt blowout |
| Keon Ellis (BKN, 10-21) | 11.2 → 34 | fringe on a new team, became a starter-level player |
| Will Riley (WAS, 10-23) | 20.7 → 1 | early-season DNP-CD |
| Nique Clifford (SAC, 10-23) | 20.2 → 2 | early-season DNP-CD |
| Cameron Johnson (DEN, 10-22) | 28.7 → 11 | flagged starter; in-game exit |
| Ty Jerome (MEM, 10-21) | 14.6 → 32 | unexpected starter, lift too small |
| Tari Eason (HOU, 10-21) | 20.3 → 3 | early-season DNP-CD |
| Vučević (ORL, 10-21) | 21.0 → 5 | fringe, new team |
| Jalen Duren (DET, 10-23) | 25.9 → 10 | flagged starter; in-game exit / fouls |

The pattern is **early-season role churn and lock-10 lineup changes**, not blowouts or OT.

### 3.2 DK points: minutes vs rate

| prediction (grader population: proj_min ≥ 10, DNP = 0, n 664) | DK MAE | SE |
|---|---|---|
| live, as projected | 8.71 | 0.25 |
| with P(play) known (played → E[DK \| plays], DNP → 0) | 7.41 | 0.22 |
| with P(play) and minutes known (actual min × projected DK/min) | 5.55 | 0.19 |

- **Played rows (n 602):** error = minutes part (E[min | plays] − actual) × projected DK/min,
  plus the rate part.

  | quantity | total | minutes part | rate part |
  |---|---|---|---|
  | variance | 104.7 | 36.8 (35%) | 62.3 (60%) |
  | mean abs | 8.17 | 4.69 | 6.12 |
  | bias | +0.48 | −0.25 | +0.73 |

  The rate part is biased high in the rotation (+1.8) and star (+1.0) tiers.
- **Live vs DK FPPG:** on the same rows, ours 8.35 vs FPPG 9.36; 80% coverage 0.825.

## 4. Candidate fixes (all `*_DEFAULT = False`, env override, model-version tag only, same bundle and seeds)

| switch | tag | where | what ON does | data it reads |
|---|---|---|---|---|
| `COURTSIDE_GHOST_DROP` | `+gh` | `pipeline.drop_moved_players(strict=True)`, live path of `project_games` only | drops a candidate who is not forced in (DK pool or report) and is on no team of the season's `team_rosters`; no snapshot → no-op | current roster snapshot (pre-game) |
| `COURTSIDE_REPORT_MATCH` | `+rm` | `availability.current_roster_extra` → `team_game_rows(pool_extra=…)`; `pipeline.project_day(dk_out=…)` | (1) a report name the box roster cannot resolve gets a second-stage match on the current roster (names that matched before match the same player); (2) a DK-pool player DK marks 'O' whom the report does not list gets P(available) 0 | roster snapshot; DK 'O' status from the slate capture at or before the run |
| `COURTSIDE_LINEUP` | `+lu` | `minutes.project_arrays` (lineup known), `pipeline.pplay_v2_fix` | demotion `min(adj, (1−w_d) adj + w_d bench_target)` and unexpected-starter target `w_t·t_g + (1−w_t)·EWMA` with the season's fitted weights (OFF 0.5 / 0.5); DK-flagged starters off the report get P(plays \| available) 0.99 | `lineup_params.json` per season S, fit on seasons < S (`research/reeval_a_lineup.py`) |

**OFF proof.**
- Each OFF path is the untouched original line, behind `if FLAG`.
- `research/reeval_a_live_replay.py identity` compares current code OFF against a reference
  tree in which availability / minutes / pipeline are the HEAD versions, with everything else
  identical.
- Result: sha1 over every game's fp / minutes / margin / total sims arrays, plus the players
  frame, **identical on 2026-10-22 and 10-24, lock10 and lock45: 20/20 game projections**
  (1,000 sims).
- Unit tests (`tests/test_reeval_a.py`):
  - `+rm` OFF never reads the roster table;
  - `+rm` ON without a snapshot gives a frame identical to OFF;
  - `+rm` ON leaves every previously matched row unchanged;
  - `+lu` at (0.5, 0.5) is `array_equal` to OFF, and `+lu` changes nothing without a lineup.

**Leak tests.**
- `+rm` reads only reports published by `at` (test `test_rm_reads_only_reports_published_by_at`).
- `+lu` weights for S never read an entry fit on S or later (`lineup_weights`: entry ≤ S, and
  the entry for S is fit on seasons < S; test `test_lu_weights_table_and_fallbacks`).
- `+gh` and `+rm` read the roster snapshot, which is pre-game information live. **In the T2.4
  slice:**
  - the 2025-26 snapshot was fetched after the season, so `+rm` there is a mild name-only
    leak;
  - `+gh` is not evaluated there: the slice's DK-pool stand-in uses future appearances, so it
    would be an oracle. The slice path (`live_like_day`) never calls the strict drop.

**Expected impact (my estimate before the full runs):**
- **`+gh`:**
  - fixes the P ≥ 0.7 excess (−21 DNPs), the 191 P-0.33 rows and the ~9–12 min per team-game
    deficit of real players;
  - large CRPS gain on the "all candidates" population;
  - on the DK-pool population, small, and could be slightly negative for bench players. The
    freed minutes go mostly to uncertain bench players through the variance-weighted 240
    reconciliation.
  - Matters most in October and after waivers / trades.
- **`+rm`:**
  - removes ~10 of the 19 DK-OUT fringe rows (report matching);
  - DK 'O' removes the rest;
  - small but clean, mostly a DFS-pool hygiene gain: no rostering a player who is Out.
- **`+lu`:** the targeted cells are 127 lock-10 rows (±3 min each). If the historical fit agrees
  with live, expect a lock-10 minutes MAE gain of ~0.1–0.2 overall and much more on those
  cells. The P part is worth 6.4 expected DNPs on 274 flagged starters.

**Not built, with reasons:**
- Overtime-aware ranges: 1 OT game; 4% of the absolute error; the sims already add OT when
  |margin| < 1.
- A minutes-by-margin curve fix: `+bo` exists and missed the bar.
- A separate early-season rotation weighting: the "sat all games so far" cell is n 10. The
  "appeared this season" under-confidence is the next candidate once `+gh` is in, to separate
  the two.

## 5. Quick checks (SMALL SAMPLE, not evidence)

Live replay on 2026-10-22 and 10-24 only: 10 games, lock10, **1,000 sims**, today's stack.
Each arm is compared with OFF, paired, SE clustered by game.

| arm | population | n | DK CRPS diff (SE) | DK MAE diff (SE) | min MAE diff, played E ≥ 15 (SE) | P(play) Brier diff (SE) |
|---|---|---|---|---|---|---|
| `+gh` | all candidates, E[min] ≥ 10 | 176 | −0.220 (0.118) | −0.435 (0.206) | −0.134 (0.086) | −0.0251 (0.0085) |
| `+gh` | DK pool, E[min] ≥ 10 | 169 | +0.010 (0.067) | −0.012 (0.106) | −0.134 (0.086) | +0.0008 (0.0014) |
| `+rm` | DK pool, all | 316 | −0.014 (0.018) | −0.037 (0.038) | 0.000 | −0.0022 (0.0021) |
| `+lu` (P part only; no weights file yet) | DK pool, all | 316 | −0.073 (0.048) | −0.083 (0.053) | +0.027 (0.053) | −0.0041 (0.0038) |
| all three | DK pool, all | 316 | −0.061 (0.069) | −0.005 (0.084) | −0.083 (0.067) | −0.0058 (0.0042) |

- Minutes sum per team-game, in-pool players: OFF 229.7 vs `+gh` 238.5 vs actual 238.7.
- `+lu` minutes-level smoke test (in-sample 2026-27 params, 3–4 dates, lean panels): the grid
  picks w_demote 1.0 and w_target 0.5–0.7; MAE moves by under 0.01. This only validates the
  code, not the effect.

## 6. Pending backtests (Mac tonight, `research/reeval_a_mac.sh`)

**Sets and arms.** The T2.1 avail harness cannot see any of the three switches: it has no
reports, no DK pool and no lineup. So the sets are:

- **(a) Minutes-level walk-forward for `+lu`** (`reeval_a_lineup.py all`):
  - weights fit for S ∈ {2023-24, 2024-25, 2025-26, 2026-27}, each on the ≤ 3 most recent
    seasons < S that have a live-stack bundle (c1rship: 2016-17…2020-21 and 2023-24…2025-26;
    there are no 2021-22 or 2022-23 bundles);
  - OFF vs ON minutes MAE and bias on 2023-24, 2024-25 and 2025-26, by cell (expected starter /
    demoted / unexpected starter / bench), game-clustered.
- **(b) T2.4 live-like slice, 2025-26** (`c1r_ship_prep.py harness`, live stack with `+pq`
  leave-month-out):
  - arms: off and `+rm` at live (lock-45) and live10; `+lu` and `+rm+lu` at live10;
  - paired by `p8_e3_eval.py dk`: DK CRPS / MAE, minutes CRPS, two-way date × player SEs, by
    tier.
- **(c) 2026-27 live replay** (`reeval_a_live_replay.py`):
  - 5 nights × lock10 / lock45 × {off, gh, rm, lu, all} at 4,000 sims;
  - every game re-projected the way `project_slate` did it at the run time: the DK pool capture
    at or before the run, the reports and lines published by then, and the live seed;
  - `report` writes the paired table.

  This is the deciding set for `+gh` (the only honest one).

**Bar (REEVAL protocol).** A switch passes only if it clears all of:
- **> 1 SE better on DK CRPS end to end:**
  - the slice for `+rm` / `+lu`;
  - the replay for `+gh`;
  - with ~34 games the replay SEs are wide, so "not worse" plus a real minutes or P(play) gain
    is the realistic outcome there;
- **no tier worse by > 2 SE**;
- **not worse on the live slice;**
- for `+lu`: **minutes MAE better on all three seasons**.

**Commands** (geekom1 first, then the Mac; one heavy process):

```bash
# geekom1: ship the working tree (B / C edits in it are OFF by default)
rsync -a --delete --exclude var --exclude .venv --exclude cache --exclude var-mlops \
  --exclude __pycache__ ~/Documents/NBA/ nicholashorton@mac-mini:Developer/courtside-compute/reeval-a/code/

# Mac (outside 12:00-12:50 and 15:45-19:45 ET; the script also guards each job start)
mkdir -p ~/Developer/courtside-compute/reeval-a && cd ~/Developer/courtside-compute/reeval-a
zsh code/research/reeval_a_mac.sh setup          # ~5-10 min: DB online backup (ro source), bundles, tips, features
nohup zsh code/research/reeval_a_mac.sh all > logs/all.out 2>&1 &   # lineup -> replay -> slice -> compare
tail -f logs/done.txt
```

**Runtime estimates (single thread, nice 10):**

| step | runtime | basis |
|---|---|---|
| `lineup` | ~1–1.5 h | 6 season frames at ~5–10 min each, plus 64 grid points × 6 seasons at ~5 s, cached per season |
| `replay` | ~20–30 min | 10 runs; geekom1 took 17 s for 2 nights at 1,000 sims |
| `slice` | ~1.5 h | 6 runs at ~14 min each (c1rship live10: 825 s), plus a one-off 2025-26 feature build if the cache copy misses |
| `compare` | ~5 min | |
| **total** | **~3–3.5 h** | |

**Outputs:** `~/Developer/courtside-compute/reeval-a/out/`:
- `lineup_fit_*.json`, `lineup_eval_*.md`;
- `replay/report_lock10.md`, `replay/report_lock45.md`;
- `dk_slice_*.json`.

The fitted `+lu` weights are written to `code/src/courtside/proj/lineup_params.json` in the
scratch tree. Copy that file back to geekom1 before any `+lu` decision. Without it, `+lu` is the
OFF weights plus the P part.

## 7. Files and coordination

**New:**
- `research/reeval_a_live_diag.py`, `research/reeval_a_live_replay.py`,
  `research/reeval_a_lineup.py`, `research/reeval_a_mac.sh`;
- `tests/test_reeval_a.py` (11 tests).

**Edited (switch code only, OFF-guarded):**
- `src/courtside/proj/availability.py` (`+rm`);
- `src/courtside/proj/minutes.py` (`+lu`);
- `src/courtside/proj/pipeline.py`:
  - `+gh` constant, setter, tag and `drop_moved_players(strict)`;
  - `set_report_match` / `set_lineup`;
  - the `dk_out` kwarg of `project_day` and the collection in `project_slate` (`+rm` only);
  - the LU P part in `pplay_v2_fix`.

**For B / C:** I did not touch `features/`, `context.py` or `dfs/`. `pipeline.py` has edits from
A in `_versions`, the setters, `drop_moved_players`, `project_games`, `project_day`,
`project_slate` and `pplay_v2_fix`; merge with care if B also edits `pipeline.py`.

**No conftest pin:** every new switch defaults OFF.

**Tests run:**
- `tests/test_reeval_a.py`;
- `test_proj_availability`, `test_pplay_v2_paths`, `test_proj_pplay_v2`, `test_pplay_q_refit`,
  `test_proj_minutes`, `test_proj_minutes_fix`, `test_reserve_extras`, `test_proj_pipeline`,
  `test_proj_backtest_live`.

All pass.
