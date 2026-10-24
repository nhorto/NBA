# Minutes restrictions and news latency (P7, 2026-10-23)

*Switch `COURTSIDE_MIN_RESTRICT` (`+rs`), default OFF, not committed. Driver:
`research/min_restrict_eval.py`. Mac scratch: `~/Developer/courtside-compute/rs/` (out/, logs/).
Live arm: `pathA-v0.3+mf+p2+rx+rb` (`COURTSIDE_CONTEXT=1 PPLAY=v2 COPULA=v0 MINFIX=1
RESERVE_EXTRAS=1 ROLE_BLEND=1`).*

## Answer in short

- **Parser works and is precise. There is almost nothing to learn from yet.** The ESPN archive starts
  2026-10-18: 59 snapshots, 247 unique comments. 5 comments (4 players) flag a restriction, and 1
  has a number (Z. Collins, "a minutes cap of 16 minutes").
- **No history.** Seasons before 2026-27 have no flagged rows. ESPN comments are not archived before
  2026-10-18, and official reports never mention minutes (`return-minutes.md`). So nothing is fit:
  the prior is a documented rule.
- **Evaluation: 3 flagged player-games** on the 2026-27 live slice (10-20 to 10-22, lock−45).
  - Minutes MAE: −0.23 min (SE 1.77).
  - DK CRPS: **+0.69 (SE 0.44), worse**.
  - This is noise either way.
- **OFF is bit-identical** to HEAD on the live slice (528 rows, every column).
- **Latency.**
  - Official-report news reaches a finished re-run in a median **35 min** (p75 84 min, p90 5.1 h).
    That counts the report from 5 min after its nominal time. 4 of 46 events had no re-run before
    tip. All 4 came from the 10-21 6:15 PM ET report for the 7 PM games, whose main slate never got
    its lock-10 run (the open "no lock-10 run" alert).
  - ESPN status changes arrive about a day early (median 26 h before tip) and wait for the next
    lock run (median 16 h).
  - 1 ESPN miss: Dean Wade (Out), seen 9.6 min before tip.
- **Recommendation: keep `+rs` OFF.** Keep the parser and the `/health/dfs` row running. Re-fit
  when there are ≥ 100 flagged rows (likely December–January), with "hard" restrictions (stated
  restriction or number) split from "soft" ones (monitor, keep an eye on).

## Part A: restriction flags

### Parser (`src/courtside/sources/espn_restrict.py`)

Input: `shortComment` + `longComment` of each player in an ESPN injuries snapshot
(`raw/espn/injuries/<day>/injuries__<ts>.json.gz`). It works per sentence and is precision-first.

**Positive phrases:**
- "minutes restriction / limit / cap"
- "N-minute limit"
- "restricted / limited / reduced minutes"
- "a limit on his minutes"
- "minutes will be monitored / limited"
- "eased (back) in(to)"
- "keep a (close) eye on his minutes", "monitor his minutes"
- "face (minutes) restrictions"
- "on a minutes limit"
- "~N minutes", "will play around N(–M) minutes", "capped at N minutes"

**Vetoes, per sentence:**
- negations: no / not / without / n't / lifted / no longer
- past seasons: "last season", "2025-26"
- contract talk: "restricted free agent", extension
- teammate sentences: another player tagged "Name (injury)", or a sentence that opens with "With
  &lt;someone else&gt;"

**Cap:** a number tied to the restriction wording, kept in 6–36. A range gives its midpoint.

**Hand check, the whole archive** (`out/flags.json`). 133 sentences match a minutes / limit / ease /
restrict / ramp keyword. Of those, 5 are restriction statements about the player himself, and the
parser flags exactly those 5. That is 5/5 precision and 5/5 recall on this sample.

| player | news | phrase | cap |
|---|---|---|---|
| Zach Collins | 10-19 17:58Z | "could potentially face restrictions early on" | – |
| Adem Bona | 10-19 21:13Z | "he'll play limited minutes behind Joel Embiid" | – |
| Collin Sexton | 10-21 00:06Z | "He'll likely have some minutes restrictions if he's cleared" | – |
| Jarrett Allen | 10-22 15:08Z | "may keep a close eye on his minutes if he's cleared" | – |
| Zach Collins | 10-22 20:07Z | "had a minutes cap of 16 minutes ... carry a similar restriction" | 16 |

The traps it rejects, from the same archive:
- "restricted / unrestricted free agent"
- "limited participant", "limited to individual work"
- "limited McConnell to 56 games last season"
- "X could see an uptick in minutes"
- stat lines ("in 18 minutes during ...")
- "may not play meaningful minutes"

`tests/test_min_restrict.py` pins 15 real comments, 16 brief phrasings and their negations.

### Storage: no migration

The flags are a derived table built on the fly. `flags_at(conn, at, since)` reads the ESPN snapshots
captured in [at − 72 h, at] through `injury_snapshots.raw_path`, re-rooted under this machine's
`var/raw/`. Each file is parsed once (in-memory LRU). No schema change.

**Sticky flags.** ESPN replaces a comment with the next news. Collins's 10-19 remark was replaced on
10-20 by an unrelated note, so his 10-21 game would have been missed. Within the 72 h window, the
newest flagged comment of any snapshot therefore counts. It is dropped when:
- the newest snapshot lists him Out, or
- he has played a game since the news (`player_games` × `games.tipoff_utc`).

This was a recall fix made from the comment text, before looking at outcomes.

### Prior (`src/courtside/proj/min_restrict.py`) and where it acts

- `lim = parsed cap`, else `RS_FRAC × 0.5 (ew_f + ew_s)` with **RS_FRAC = 0.75**, floored at 6.
  Normal minutes, not the plan, is the base. A returner the engine already cuts is left alone
  (min(plan, lim)).
- **Why 0.75 is not fit:** there are 0 flagged rows before 2026-27, so it is a rule. A typical
  announced restriction is about 20–24 minutes for a ~30-minute starter.
- **Plan input:**
  - `pipeline.project_games` computes `{(key, player): lim}` at the run time (`at`, or `rs_at` on
    the report path).
  - `build_games(rs_caps=)` adds an `rs_cap` column to the flagged team's panel only.
  - `minutes.base_and_pplay` attaches it to the base (`V1Base.rs_cap`).
  - `project_arrays` caps the plan after the lifts and passes the per-row ceiling to the reconcile.
    The freed minutes go to teammates and the team total is kept.
  - P(plays) and the per-sim noise are unchanged.
- **Version:** `<arm>+rs` (after `+tr`), same bundle and seeds.

### Evaluation (live slice 2026-27, lock−45, P(play) shipped, 4,000 sims)

`research/min_restrict_eval.py harness 2026-10-20 2026-10-22` (OFF / ON), then `compare`.

- `backtest.live_like_day` calls `build_games` directly, so the driver adds the same caps at the
  group's `at`.
- 2025-26 has no ESPN snapshot, so ON = OFF there by construction (`restrict_caps` returns {}).
- SEs: two-way date × player (`metrics.paired_cluster2`).

| cell | n | min MAE OFF | ON | ON−OFF (SE) | DK CRPS OFF | ON | ON−OFF (SE) |
|---|---|---|---|---|---|---|---|
| flagged, played | 3 | 3.73 | 3.50 | −0.23 (1.77) | 3.86 | 4.54 | +0.69 (0.44) |
| unflagged, played | 328 | 4.973 | 4.968 | −0.004 (0.005) | 6.809 | 6.810 | +0.001 (0.002) |
| all rows (DK) | 528 | | | | 5.409 | 5.416 | +0.006 (0.007) |

The "unflagged" cell moves only through the flagged players' teammates. Rows outside the flagged
games are identical ON vs OFF (n 435).

| player-game | lim | plan OFF → ON | actual min | DK mean OFF → ON | actual DK |
|---|---|---|---|---|---|
| Bona PHI 10-20 | 9.9 (0.75 × 13.2) | 14.2 → 10.8 | 10 | 4.5 → 3.4 | 7.25 |
| Collins CHI 10-21 (sticky) | 13.5 | 13.3 → 13.2 | 9 | 10.3 → 10.2 | 6.25 |
| Allen CLE 10-22 ("keep an eye") | 21.6 | 30.7 → 22.5 | 28 | 21.3 → 15.2 | 26.25 |

What the three rows show:
- **Bona:** the minutes were right, but he beat his per-minute rate.
- **Collins:** the cap barely bound.
- **Allen:** the "soft" phrase was wrong. He was not restricted.

With n = 3 this says nothing about the rule. It is a reason to separate soft phrases before any
ship.

### OFF identity

- **Frame level, live slice:** the HEAD tree (`git archive HEAD`, `rs/base`) and the working tree
  (`rs/code`), both OFF, give equal rows (528 rows, every column bit for bit; `... same`). The
  working tree also holds the other agents' uncommitted OFF switches.
- **Engine:**
  - `test_off_equals_head_copy_engine` and `test_off_simulation_frame_equals_head_copy` compare a
    `git show HEAD:` copy of `minutes.py` (var/tmp/p7) for base, P(plays), the scenario projection
    and a whole simulated game.
  - An all-NaN `rs_cap` column changes nothing.
  - The golden float test is Linux-only.

## Part B: news latency (display only)

`src/courtside/proj/news_latency.py` is DB only. `/health/dfs` shows a "News to re-run" card in
the slates section: per-source quantiles plus the last-hour news list.

**Definitions:**
- **Event:**
  - an official-report status change between a team's consecutive reports (news time = the
    report's nominal time), or
  - an ESPN status change between captures (news time = first capture).
  - The first batch of the season is the baseline.
- **Game:** the team's next tip within 30 h.
- **Re-run:** the first DFS build (ok, re-projected) of a slate with that game that started after
  the news was out and finished by tip.
  - Official reports count from 5 min after their nominal time, when the PDF is posted.
  - ESPN captures count at once (5 s slack for a capture made by the same tick).
- **Lock:** tip.

Data: the 2026-10-23 backup (games through 10-22). There were 57 more events for games not yet
played.

| news | events | to re-run p25 / p50 / p75 / p90 | within 1 h | to tip p50 | missed |
|---|---|---|---|---|---|
| official report | 46 | 17 / 35 / 84 / 309 min | 61 % | 52 min | 4 |
| ESPN status | 53 | 75 min / 16 h / 17 h / 21 h | 25 % | 26 h | 1 |
| all | 99 | 21 / 84 min / 16 h / 20 h | 41 % | 5.0 h | 5 |

Reading the table:
- **Official reports** cluster in the last hour (median 52 min before tip), so the lock-45 / lock-10
  runs pick them up. The 4 misses are a missed lock-10 run, not slow news: the 6:15 PM report was
  out before the 7 PM lock, and only the lock-10 run could have used it.
- **ESPN status changes** mostly arrive the evening before. Their latency is the wait for the next
  day's lock-45. ESPN status drives P(play) only when a team has no official report
  (`availability`), so this is not a model gap.
- **ESPN publication → our capture:** median 47 min (p25 19 min, p75 106 min, p90 9 h, mostly
  overnight gaps). Source: the comment `date` against the first capture showing it.

**Caveat.** The 5-minute posting lag is an assumption. The DB has no fetch time per report; the PDF
archive's file times on the Mac would give it.

## Files

- New:
  - `src/courtside/sources/espn_restrict.py`
  - `src/courtside/proj/min_restrict.py`
  - `src/courtside/proj/news_latency.py`
  - `research/min_restrict_eval.py`
  - `tests/test_min_restrict.py`
  - `tests/fixtures/espn_injuries_restrict.json.gz`
- Edited:
  - `proj/minutes.py`: `V1Base.rs_cap`, the attach in `base_and_pplay`, the cap in
    `project_arrays`.
  - `proj/pipeline.py`: `+rs` tag, `set_min_restrict`, `build_games(rs_caps=)`,
    `project_games(rs_at=)`, `project_historical` passes `rs_at`.
  - `web/health_dfs.py` + `templates/health_dfs.html`: the latency card.
