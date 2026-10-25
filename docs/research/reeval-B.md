# Re-evaluation B: context features (H1–H5), 2026-10-25

Workstream B of `docs/REEVAL_2026-10.md`. The per-minute ablation runner is written and checked on a small sample on geekom1. The full runs are queued for the Mac tonight (§6). The H1 and H3 variance-component diagnostics need no model fit, so they were run on the full frame on geekom1. They are real numbers, not a sample.

Nothing is wired into the pipeline. Nothing is committed.

## 1. What was built

All new groups sit behind one switch: `features.CONTEXT_B_DEFAULT = False`, with the env override `COURTSIDE_CONTEXT_B=1` or `features.set_context_b(on)`.
- **OFF:** `features.GROUPS` / `REEVAL_GROUPS` are exactly the pre-reeval registry, and `extend_frame` ignores the new names.
- **ON:** the groups below are registered.

They are in no live group set, and B does not touch `pipeline.py`. Its current diff is workstream A's.

| Group | H | Module | Key | Columns (per DK stat) |
|---|---|---|---|---|
| **F8** | H1 | `proj/features/opp_history.py` | (player, opponent franchise) | `f8_<s>`, `f8_dk`, `f8_n` (+1 monotone on own stat) |
| **F8r** | H1 | same | same | `f8r_<s>`, `f8r_dk`, `f8r_n`: fixed light shrinkage (3 games) |
| **F8c** | H1 | same | (player, opponent head coach) | `f8c_<s>`, `f8c_dk`, `f8c_n` |
| **F7** | H3 | `proj/features/referees.py` | game (both teams) | `f7_fta`, `f7_pf`, `f7_pace`, `f7_n` |
| **F3x** | H4 | `proj/features/tracking_ext.py` | (player, day) | potential assists, AST per potential assist, drives (+ pts / FGA / passes / TOV per drive), touches by area, time of possession; `t3x_gap_ast` |
| F1 (existing) | H2 | `opp_avail.py` | team-game | retested as +F1 on both stacks |

Plumbing:
- `store.extend_frame` builds and caches the B tables. Their cache files are `ext_<S>_<g>b1_*`.
- `context.run_ablation` and `extract_proxy` take `pairs=`.
- The extension detection in `context.fit_season` / `run_ablation` also checks `FX.REEVAL_GROUPS`, which is empty when OFF.

### F8 / F8r / F8c definition (H1)

The feature is a residual, so it can only add the player × opponent *interaction*.

Base for each past game: Eb = E_s × exp(`f2_dvp_<s>_grp`). That is the EB rate × actual minutes × market reconciliation × the opponent's as-of allowed-vs-expected index for the player's position group. This is F2, which is live, so opponent defence by position is not counted twice.

Player level from his *other* opponents (leave-pair-out):
R = (A_y − P_y + 10 ē) / (A_E − P_E + 10 ē)

Pair:
- X = R × P_E, the expected count of the pair at his current level.
- r = P_y / X.
- Sampling variance: c × Σw²·(Eb + φEb²) / X², using the state's NB φ.

Weights: half-life 540 d and 0.8 per season boundary, so 3+ seasons back carry under 6%. Everything is strictly before the game day (`ew_asof`).

EB shrinkage toward 0, the BLUP of a random pair effect:
θ = 1 + τ²/(τ² + var r)·(r − 1), and `f8_<s>` = log θ.

- **τ², c (walk-forward fit on seasons < S):** a placebo-corrected method of moments over (player, opponent, season) blocks.
  - c = Σ(Y−X)² / ΣV on blocks whose opponent labels are permuted within player-season. It calibrates the NB variance and comes out at 0.99–1.04 for the box stats.
  - τ² = (Σ(Y−X)² − c ΣV) / ΣX² on the real blocks, with the SE clustered by player.
  - The implied pseudo-count is k = c/τ².
- **F8r** uses the same ratio with a fixed 3-game pseudo-count, so the GBM can judge it itself. It checks that the EB fit did not throw a usable signal away.
- **F8c** keys the pair on the opponent's head coach. `team_coaches` is a season snapshot, so a mid-season hire is known for the whole season. That is an identity lookahead, not an outcome one, so F8c is exploratory.

### F7 definition (H3)

Each target is per game, "actual vs expected without the referees":
- **fta:** Σ FTA vs Σ E_fta over both teams, with E market-reconciled and the NB variance.
- **pf:** both teams' fouls vs possessions × the league's as-of fouls per possession.
- **pace:** game possessions vs (min/240) × pace_T × pace_O / pace_L. The team paces are as-of values shrunk to the league with 10 games.

Per official: an EW index (half-life 365 d, 0.9 per season), EB-shrunk like F8, with τ² and c from (official, season) blocks of seasons < S. The placebo permutes officials within a season. The crew feature is the mean of the officials' log indices.
- Games with no league history yet carry no weight; there are no global fallbacks.
- Live hook: `referees.History.day_features` + `crews_from_assignments`. The latter maps `referee_assignments` names to official ids, by full name and then by jersey number.

### F3x definition (H4)

F3's form, applied to `player_tracking_ext`:
- Per-minute rates shrunk to the group's league rate as of the day (240 prior minutes).
- Conversions shrunk in denominator units.
- `t3x_gap_ast` = log(potential-assist rate × league AST per potential assist / the EB AST rate).
- NaN before the table's first row, or when the table is missing.

## 2. Leak tests and OFF identity

File: `tests/test_proj_reeval_b.py`, 19 tests. Together with `test_proj_features2.py` and `test_proj_context_ext.py`, all 36 pass.

- **L1 scramble (F8, F8r, F8c, F7, F3x, each separately).** Every box, team-box and `player_tracking_ext` stat on or after a cut in season S is re-randomised. Features up to the cut day are identical, and later ones change.
  - The first F7 version failed this test. Two global fallbacks (the median of all possessions and the all-data PF rate, used on days with no league history) leaked into the training-season targets and so into τ². Those games now carry no weight.
- **L5 truncation (all B groups).** Every row after the cut is deleted, and features up to the cut are identical.
- **Walk-forward.** τ² / c are fit only on the rows of seasons < S (`build(..., season=S)`).
- **Planted signals.**
  - A 1.6× points pair gets `f8_pts` +0.24, shrunk below log 1.6 = 0.47. Other players stay near 0.
  - With a planted per-pair sd of 0.25 on rebounds, τ² is recovered in [0.02, 0.15] at z > 3. With nothing planted, |z| < 3.5.
  - The split-half on the planted data: real > placebo + 0.15.
  - A whistle-heavy official gets a positive crew `f7_fta`.
  - F8c follows the coach and not the franchise: a new coach means no evidence.
- **No history gives exactly 0.** A player's first game against an opponent has every F8 / F8r column at 0.
- **Live hooks.** F7 `day_features` reproduces the frame, also from history without the day's results. The `referee_assignments` → official-id mapping is tested, including the jersey-number fallback and replay-center rows dropped.
- **OFF identity.**
  - With the switch OFF, `GROUPS` is exactly the old set, and the switch round-trips.
  - The columns and monotone constraints of the existing groups are unchanged with it ON.
  - `extend_frame` ignores the B names when OFF.
  - A P+F2 layer fit on a frame that carries the B columns is **byte-identical** (`to_json`) to the same layer fit without them.
  - B does not touch `pipeline.py`, and `test_proj_context_ext.py` (the pipeline ON/ext paths) passes.

## 3. H1: how much signal survives shrinkage (real data, full frame)

Data: regular-season rows of 2019-20 … 2024-25 (149,544 player-games), from the 2025-26 frame. Blocks are (player, opponent, season). The split-half pairs are (player, opponent) over all 6 seasons with ≥ 6 games (about 11k pairs), using a precision-weighted correlation.

| Stat | τ² (SE, z) | sd of the true pair effect | k (games) | Split-half r real / placebo | EB weight at 4 / 8 / 12 games |
|---|---|---|---|---|---|
| pts | −0.00075 (0.00078, −1.0) | 0 (95% ≤ 2.9%) | ∞ | 0.029 / 0.016 | 0 / 0 / 0 |
| fg3m | −0.00283 (0.00186, −1.5) | 0 | ∞ | 0.011 / 0.021 | 0 |
| reb | +0.00082 (0.00069, +1.2) | 2.9% | 305 | 0.038 / 0.021 | 0.01 / 0.03 / 0.04 |
| ast | +0.00128 (0.00103, +1.2) | 3.6% | 333 | 0.045 / 0.034 | 0.01 / 0.02 / 0.03 |
| stl | +0.00135 (0.00302, +0.4) | 3.7% | 1,065 | 0.018 / 0.030 | ≤ 0.01 |
| blk | +0.00027 (0.00427, +0.1) | 1.7% | 8,142 | 0.026 / 0.021 | 0 |
| tov | −0.00085 (0.00183, −0.5) | 0 | ∞ | 0.035 / 0.021 | 0 |
| **DK** | −0.00015 (0.00034, −0.4) | 0 (95% ≤ 2.3%) | ∞ | 0.027 / 0.033 | 0 |

The coach-keyed version gives the same picture: pts −0.00077, ast +0.00156 (z 1.5), DK −0.00009.

**Verdict.** After the EB level, the market and F2, no stat has a player × opponent effect distinguishable from noise.
- Real minus placebo split-half is +0.01 ± 0.01 throughout. The ~0.02–0.03 shared by real and placebo pairs is a shared-level artefact: both halves divide by the same estimate of the player's level.
- At the 95% upper bound, the DK pair sd is 2.3%. With the EB weight that implies at ~12 games (~0.1), the feature could move a 40-DK player by about ±0.3 DK.
- Most "he kills this team" stories are a few hot games, and they are fully explained by noise.

## 4. Data coverage and verdicts

- **H3 referees: usable history.**
  - sportsdataverse `nba_stats_officials` covers 1996-97 → 2025-26: every regular-season game, 3 officials (a 4th in about 5% of games), plus playoffs.
  - 8,903 of 8,906 frame games 2019-20 … 2024-25 have a crew.
  - Live: `referee_assignments` has 36 games (10-20 → 10-24), posted around 9 am on game day, with names and numbers but no ids. `crews_from_assignments` maps them to ids.
  - Official effects (2019-20 … 2024-25, 479 official-seasons):

    | Target | τ² | sd per official | z |
    |---|---|---|---|
    | FTA | 0.00067 | 2.6% | 4.6 |
    | Fouls | 0.00051 | 2.3% | 5.1 |
    | Pace | 0.000004 | 0.2% | 1.9 |

  - So the effects are real for whistles and nil for pace. These are beyond the closing total, because E is market-reconciled.
  - If the effects add up across the crew, a crew is about ±4.5% of game FTA, roughly ±2 FTA. For an 8-FTA player that is about ±0.3 DK, and points are rescaled to the implied total anyway.
- **H4 `player_tracking_ext`: no data at all.**
  - The table arrives with migration 18. The geekom1 snapshot and Mac prod are both at schema 16, so the table does not exist anywhere. Zero rows, and no archived `leaguedashptstats` files.
  - History needs the network backfill: `courtside backfill stats --season S`, about 600 requests and 10–15 min per season.
  - The protocol needs 3 test seasons each fit on prior seasons, so it needs a backfill of 2019-20 … 2025-26: about 4,200 requests, roughly 1 h. The code and its tests are ready. Without that backfill, H4 cannot be evaluated this round.
- **H2 F1: why it was dropped and whether that still holds.**
  - The first ablation (2026-10-19, commit d894531) dropped it because per-minute CRPS was < 1 SE in 2024-25 and negative in 2019-20.
  - That test used the **actual** absences (the oracle), so a better P(play) cannot rescue it. P(play) v2 only matters for an end-to-end run, and F1 needs the per-scenario sim wiring that was never built (`context.py` INTEGRATION note).
  - Historical P(play) cannot be backtested over full seasons: injury snapshots only start in 2025-12.
  - The reason still holds structurally:
    - The closing line already prices the opponent's absences, and E carries it.
    - F2's recency-weighted DvP sees the opponent's recent lineups.
    - The usage bump handles the player's own team.
  - Raw check on the real frame (actual / E × F2 index, mpg ≥ 15, regular season 2019-26, game-clustered SE):

    | Cell | Ratio | Healthy-opponent ratio |
    |---|---|---|
    | C rebounds, opponent's top rebounder out | 1.042 (0.008) | 1.001 |
    | C points, rim protector out | 1.030 (0.009) | |
    | All points, rim protector out | 1.013 (0.002) | 1.003 |
    | G steals, opponent's ball-handler out | 1.032 (0.011) | |
    | G assists, perimeter defender out | 1.014 (0.006) | |

    Real, but small and confined to rare cells. That is consistent with the original per-minute null.

## 5. Small-sample validation (geekom1, a pipeline check, NOT evidence)

Setup:
- Test 2025-26. Fit on 2023-24 … 2024-25 only. 1/4 of players. lr 0.15, min leaf 300.
- n = 5,176 test rows, 1 LightGBM thread, capped at 2 GB.
- Each full stage ran in 6–10 s.

| Arm vs its base | NB LL/row (z) | DK CRPS (z) |
|---|---|---|
| P+F2 +F8 | −0.00085 (−0.8) | −0.0022 (−1.0) |
| P+F2 +F8r | +0.00015 (+0.1) | −0.0016 (−0.8) |
| P+F2 +F8c | +0.00100 (+0.8) | +0.0003 (+0.1) |
| P+F2 +F7 | +0.00041 (+0.4) | −0.0010 (−0.4) |
| P+F2 +F1 | −0.00010 (−0.1) | +0.0005 (+0.2) |
| P+F2+F3+F6 +F8 / +F8r / +F7 / +F1 | −0.6 / −0.2 / −0.7 / −1.1 | −0.7 / +0.2 / −1.8 / −0.4 |

Every arm fits, predicts inside the clip, and produces the paired SEs and the T2.7 extract end-to-end proxy. The numbers are for checking the pipeline only.

## 6. Run plan (Mac, tonight)

Script: `research/reeval_b_mac.sh`. The full commands are in its header.
- Scratch: `~/Developer/courtside-compute/reeval-b/`.
- One process, `nice -n 10`, 1 thread.
- Never between 12:00–12:50 or 15:45–19:45 ET.

Steps:
1. **From geekom1:** rsync the working tree to `reeval-b/code/`, excluding `.git`, `.venv` and `var`. The tree also carries A's and C's uncommitted edits. The frame-based ablation reads cached frames and `features/`, so they do not enter these numbers. Then scp `var/tmp/reeval-b/data/proj/features/state_params_*.json` and `var/proj/corr_extract_{2023-24,2024-25,2025-26}.pkl` into `reeval-b/data/proj/…`.
2. **On the Mac:** `cd ~/Developer/courtside-compute/reeval-b && zsh code/research/reeval_b_mac.sh setup`. This makes APFS clones of:
   - the three cached frames (`xf/…/frame_2023-24_fe24174165`, `xf/…/frame_2024-25_2e386b7e88`, `pq/data_v1/…/frame_2025-26_8f88c15a91`; the keys were verified against these state params);
   - the officials parquet;
   - `xf/data/courtside.db`, which has every historical table the runner reads.
3. **On the Mac:** `nohup zsh code/research/reeval_b_mac.sh run > logs/run.log 2>&1 &`. This runs `reeval_b_ablation.py <S> --stage all` for 2023-24, 2024-25 and 2025-26.
   - Per season: vc ~5 s; live (6 arms × ~60 s) ~7 min; ext (5 arms, plus the F3/F6 tables built once) ~7–8 min.
   - **About 45 min on an idle Mac, 1.5–2.5 h at today's load.** Peak memory is about 2 GB.
   - Outputs: `out/reeval_b_{vc,live,ext}_<S>.{txt,json}`.
4. **Gate.**
   - A group goes to the T2.1 end-to-end step only if per-minute DK CRPS is > 1 SE in all three seasons and it is not worse on NB LL.
   - That step needs a wiring change first: an F7 / F8 `LiveExtras` hook behind the switch. The F7 hook already exists.
   - It then needs the T2.4 live slice.
   - H4 needs the 2019-26 `player_tracking_ext` backfill first; then run `--f3x`.

## 7. Expectations

- **H1 (F8 / F8r / F8c): drop.** The variance component is zero within noise for every stat, and EB shrinks the feature to about 0. F8r lets the GBM find any remaining signal; I expect |z| < 1 per minute. The owner's intuition is not supported once the player's level, the market and the opponent's defence by position are accounted for.
- **H2 (F1 retest): drop again (< 1 SE).** The real effects (big-man rebounding when the opponent's top rebounder is out) are small and rare. P(play) v2 cannot change a per-minute result measured with oracle absences.
- **H3 (F7): small per-minute gain at most; drop end to end.** It is the only real new signal (whistle tendencies at z ≈ 5). It may reach about 1 SE in points / FTA log-lik, but the reconciliation of points to the implied total cancels most of it end to end, and pace is nil.
- **H4 (F3x): not testable this round (no data).** After a backfill, expect a modest assists gain on top of F3 (potential assists vs passes). It is unlikely to change the end-to-end verdict on F3+F6.
- **H5 (attempts × efficiency): not built.** F4 already carried xFG quality, shot diet and luck as features. It was real per minute and null end to end, because the points reconciliation cancels within-team efficiency shifts. A structural attempts × efficiency rate would face the same cancellation. The remaining lever is 3PM-only (PLAYERS.md F4 note), so I recommend not spending Mac time on H5 now.
