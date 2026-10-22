# HANDOFF: start here after a context reset

*Last updated 2026-10-21 ~23:10 ET (day 2; live model pathA-v0.3+p2).*

## 0. Where we are, in one paragraph

Courtside, the "NBA Morning Brief", is **built and live** on the Mac mini. It covers:
- collectors and the game model, which ties the closing line
- play-by-play analytics, win probability and trends
- the Brief with its LLM writer
- the web app
- MLOps: registry, monitoring, `/health`, and a weekly retrain

The **next project is the DFS optimizer plus player projections**. Its plan is `docs/PLAYERS.md` **v2 (DFS-first)**.
- **Review:** Fable reviewed v1, and v2 applies all review items.
- **Reviewer doc:** https://claude.ai/code/artifact/672b2260-140e-49b3-9992-7e644f44a0be (Claude Doc id `672b2260-140e-49b3-9992-7e644f44a0be`, tab file `54bc59b6-a58d`, body `30cf8362-11fc`).
- **Approval:** after v2 the owner said "please continue". Treat the plan as **approved, including the proposed defaults** in PLAYERS.md §9 (listed below). If the owner or the reviewer comments on the doc, apply those changes.
- **What's next:** start the **Phase 0 + Phase 1 build** (§3 below).

## 1. Machines, paths, services

| | geekom1 (Linux, dev, 11 GB RAM) | Mac mini (prod, M4 32 GB, Eastern TZ) |
|---|---|---|
| Repo | `~/Documents/NBA` (remote `origin` = github nhorto/NBA, `mac` = `nicholashorton@mac-mini:Developer/NBA`) | `~/Developer/NBA`. **Must not move into ~/Documents**: macOS TCC blocks launchd there. |
| Data | `var/` (COURTSIDE_DATA) | `~/Developer/NBA/var/` |
| Scheduled | nothing | launchd: `com.courtside.pregame` (10:00, 13:30, 17:00, 18:30, 20:00, 22:00 ET), `.postgame` (03:00, 09:00), `.weekly` (Mon 05:00, mlops retrain/shadows/card), `.web` (KeepAlive) |
| Web | none permanent | **http://mac-mini:8765** (tailnet IP 100.81.92.75), `/health` = Model Health |
| Heavy compute | ≤ 3 agents, one heavy job | scratch: `~/Developer/courtside-compute/` (rsync with `--exclude var --exclude .venv --exclude cache --exclude var-mlops`); never write the prod DB from scratch jobs |

Operational notes:
- **Deploying to the Mac:** commit on geekom1, `git push origin main`, then on the Mac run `git pull --ff-only && uv sync --frozen` and the tests. The Mac has no GitHub credentials, so always push from geekom1. After web changes, run `launchctl kickstart -k gui/501/com.courtside.web`.
- **uv** on geekom1 needs `eval "$(mise env -s bash)"`. On the Mac use `export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH`.
- **LLM prose** comes from the owner's subscription via CLI, never an API key. On geekom1 `claude -p` works. On the Mac, `claude` is "Not logged in" over SSH, so it falls back to `codex exec`, which works.
- **The owner's client can't show host-path images.** Show UI through a live URL.

## 2. State of data and jobs (updated 2026-10-19 ~11:50 ET)

- **Still running on geekom1:** per-game player backfill PID 2962632 (`var/logs/backfill_players.log`; matchups ~2022-23, then tracking/hustle). When it finishes, rsync `var/raw/nba_stats/{boxscorematchupsv3,boxscoreplayertrackv3,boxscorehustlev2}` to the Mac and run `courtside backfill players --per-game all --reparse` there.
- **Mac prod DB was rebuilt from archives today** (schema 11): rosters + player index 2026-27, RotoGuru history, 2025-26 official injury rows (177k), DK/RotoGuru ids, starters (334k), stints (8,604 games), rate_state. Raw archives copied: rotoguru, nba_cdn boxscore + playbyplay, sportsdataverse, injury PDFs, season tables. Model bundle cached at `var/proj/bundle_2026-27.json`.
- **Mac launchd:** pregame, postgame, weekly, web, and **`com.courtside.dfs-tick`** (every 5 min 11:00–23:55 ET → `courtside dfs tick`: lock45/lock10 builds, late swap).
- **Fixed today:** DK lists WNBA draft groups under Sport NBA (filtered; one bogus slate removed from the Mac DB). Pipeline slate-day bug (UTC vs ET).

## 3. Build status (approved plan: `docs/PLAYERS.md` v2 §8; decisions made while building: §0.4)

**Phase 0: complete.** T0.0 migrations 9–11 + CLI; T0.1 reports verified; T0.2 RotoGuru 1,110 days / 249k rows; T0.3 cdn boxscores; T0.4 stints (99.55%); T0.5 ids; T0.6 rules/scoring (TD = +4.5 confirmed); T0.7 DK capture (slates, payouts ≤ 60 added DK requests/day); T0.8 game-env residuals (margin sd ~13.5–14, not 12.4).

**Phase 1: complete (2026-10-19, a week early).** T1.0 gate PASSES on every line (full runs): 2024-25 DK MAE +5.54% vs season-to-date (SE 0.47), 80% coverage 0.825, minutes MAE 4.57, RotoGuru 2019-20/2020-21 player MAE +2.3% vs best salary-implied; lineups 0.68 of hindsight vs 0.47–0.52 for salary-implied (`var/reports/evallite_20261019.txt`). Live model `pathA-v0.1`. Postgame job now also writes `rate_state` and runs the post-game why.

**Status at 2026-10-21 10:30 ET (day 2 of the season):**
- **Live model `pathA-v0.3+p2`** (owner approved 2026-10-21 after opener grading): P+F2 context layer (`CONTEXT_DEFAULT = True`) + P(play) v2 (`PPLAY_DEFAULT = "v2"`: listed players use the calibrated report cell; fringe pool players get a fitted P(play)). Rollback: `COURTSIDE_CONTEXT=0` / `COURTSIDE_PPLAY=v1` or flip the constants in `proj/pipeline.py`. Tests pin `COURTSIDE_PPLAY=v1` in `tests/conftest.py`.
- Opt-in, not shipped: F3+F6 (`COURTSIDE_CONTEXT_EXT=1`, `v0.4`: no gain over v0.3 on the T2.4 live slice); copula v1 (`COURTSIDE_COPULA=v1`, `+c1`: CRPS −0.9 SE in 2024-25, ranges too wide; needs residual-fit variance + blowout-minutes fix). Dropped: F1, F4 (xFG), F5 (matchups).
- **Opener graded** (main-slate lock10, n=49): DK MAE 7.33 vs DK FPPG 7.25, minutes MAE 4.63, cov80 0.857 — within the T2.4 live baselines (`mlops/live_like_baseline.json`). Lineups lost to the FPPG/salary baselines on night 1 (noise: one 3-game night).
- **Fixed 2026-10-21:** first run of a NEW slate projected before its DK ids were mapped (optimizer: no legal lineup) — `project_slate` now maps first. Brief "teammates not playing" used last season's teams (traded players) — now current roster only. Adaptive lock10 window (Mac CI load made builds take 10 min).
- Shipped 2026-10-20: late swap, Showdown check, contest builder (banded chances, no ROI headlines until ≥ 10 uploaded standings), Tonight player cards + prop_compare + grading, Brief Player stories, T2.4 live slice + baselines.

**Added 2026-10-21 afternoon:**
- **X factor (`xf-v1`, live, commit 099e370/8a05151).** DFS pool tag + "Big-night chance" column: chance of DK ≥ 6 × salary/$1k from the run's sims, recalibrated per tier (`proj/dfs/xfactor.py`, `docs/research/xfactor.md`). Tag = bench/role, P(play) ≥ 0.85, proxy band Low, chance ≥ 1 in 3, not among the slate's 10 best projections per $1k, max 5. Contest builder shows how many lineups carry one. UI: the pool defaults to the top 40 by projection, so some tagged players need "Show all" or the Big-night sort; on phones the column needs a sideways scroll.
- **#3 minutes after injury (research, a73d9f9; `docs/research/return-minutes.md`).** No return multiplier: the real issue is a general per-minute bias tied to projected vs recent minutes (the f(min/mpg) adjustment applied to planned changes). The re-key fix (`COURTSIDE_MINRATIO`, `+mr`) is being finished behind a switch, default OFF. Smaller follow-ups: a "returning" split in the lock−45 Questionable cell (with the lock−10 Q refit); reason parser files "Return to Competition Reconditioning" as G League; parse ESPN full comments for minutes-restriction flags (~January; raw snapshots archived).
- **Owner's build order after X factor:** #3 fix → #2 preseason minutes + early-season weighting (stats.nba.com `SeasonType='Pre Season'` works; minutes but no starters; 2019-20 includes bubble scrimmages) → #1 prop lines as features → #5 contest builder "how many lineups" curve (chances vs entries from the same sims; bands until calibrated; no spend advice). #5 is in progress alongside the #3 fix, since it doesn't touch the minutes model.

**Added 2026-10-21 evening:**
- **Preseason history (fe2ad0f, deployed):** migration 15, own tables `preseason_team_games` / `preseason_player_games` (+ `scrimmage`, `vs_non_nba`); `courtside backfill preseason`; 38,101 player rows 2010-11 → 2026-27. Whole minutes, no starters (optional boxscore starters crawl ~850 requests, not run).
- **#5 curve (76bdbd5, deployed):** contest builder shows chances by number of entries (nested greedy portfolio on the same sims). Chances are against our modelled field, which is far weaker than a real DK field, so they read high (1 lineup ~65–80% in the money); the note says so, and bands stay until ≥ 10 uploaded standings. N=150 takes ~200 s.
- **#3 minutes-ratio re-key (`COURTSIDE_MINRATIO=1`, `+mr`, c8ded64 / 7b9836d): OFF, not shipped.** Design B (pooled) CRPS −2.8 / −2.0 SE (2023-24 / 2024-25), −3.3 / −4.9 SE RotoGuru, live flat, but the bench tier is worse in all six sets. Design C (P-tiered, `PL_TIERS_C`) has bigger CRPS gains but fails four tier checks. Root cause: planned demotions (P < 0.9 × last-10 minutes) get P(play) +0.04–0.06 and E[min] +0.4–0.8 too high, which the old per-minute penalty hid.
- **#2 preseason minutes (`COURTSIDE_PRESEASON=1`, `+pre`, 7b9836d): OFF, not shipped.** First 15 games: minutes MAE −1.4 / −2.8 / −0.9 SE (2023-24 / 2024-25 / 2025-26); flat after; star tier worse in full-season terms (+1.4 / +2.6 SE). The season-start debut reserve over-reserves (opener: 18–30 min held back for BOS/OKC/PHI/SAS, debutants played 0–13).
- **In progress (agents):** `COURTSIDE_MINFIX` (`+mf`): fix demotion over-projection + minutes dispersion + a robust debut reserve, then re-evaluate +mf, +mf+mr, +mf+pre. #1 prop lines as an F0 feature group (as-of join, de-vig, readiness report), off by default; only 2 nights of props so far.

**Next (in order):**
1. Watch tonight's lock runs (first lock 19:00 ET) and tomorrow's 03:00 grading under `pathA-v0.3+p2`; add a `+p2` block to `live_like_baseline.json` once it has a few nights.
2. Lock−10 Questionable cell runs high (0.66 vs 0.53 played on the slice): refit that bucket.
3. Blowout minutes (stars don't lose enough minutes in simulated blowouts), then retune copula v1 variance on the context arm's residuals → re-evaluate `+c1` (per-stat sims for props).
4. After ~3 weeks of props: F9 market prop lines as features; prop calibration report. After ≥ 10 uploaded standings: ownership refit + calibrated payout display.
5. Later: Path B sim (T2.6), T2.17 backtest extension, T3.3 weekly DFS report, Phase 4 (why engine, referees, player pages), FanDuel.

**Phase 2 / 2F / 3 summary (2026-10-19):** done — T2.1 harness, T2.2 minutes v1 (+ live-path fix), T2.3 usage v1 (not live), T2.5 sim step-0, T2.7 causal correlations, T2.11 ownership proxy + simulated field (P(cash)/P(top1)/E[payout]/dupes per lineup, labelled proxy/optimistic), T3.0 grading, T3.1 `/health/dfs`, T3.2 `/dfs/results`, lineup sim ranges, `courtside proj live-path-eval`, F0–F2 feature store (P + F2 pass, F1 fails; not live). Use `TMPDIR=~/Documents/NBA/var/tmp` for tests; never `git commit -a` with agents running.

**Open items:** owner to test one DK CSV upload (numeric draftable ids vs "Name (ID)"); weekly availability refit not yet scheduled (owner: it changes P(play) weekly). Closed 2026-10-21: `why_v0.top_stories` is wired into the Brief (Player stories); the 22 non-final cdn boxscores (17 in-progress, 5 stubs; 2019-20 except one 2024-04-03 game, of 8,713) are what the cdn serves, and a re-fetch is refused (403); left as is. Then Phase 2 (§8).

**Agent rules** (unchanged): ≤ 3 agents on geekom1, one heavy job; agents don't commit; give each its task row, owned files, acceptance, and the no-browser/no-DK-crawl rules.

## 4. Owner decisions (binding)

- **Product:** a paid-subscription quality bar, hosted locally for now. Full latitude on modeling, judged only by honest walk-forward results.
- **DFS priority:** the DFS optimizer is the top priority, "as accurate and professional as possible". The owner may play occasionally but doesn't usually play DFS.
- **DraftKings only for v1.** FanDuel comes later via a user-downloaded CSV, after Phase 3.
- **Two build modes:** **Top N lineups** (default N = 10, min-different d = 1) and a **Contest builder** (a diversified N-entry portfolio for a chosen contest).
- **Guidance:** light only (simulated cash and top-1% odds, a dupe-risk band, a "sensible stakes" note). No dollar or Kelly sizing. Simulated ROI is always labelled simulated.
- **DK data policy:**
  - DK's terms forbid automated collection and entry.
  - RotoGuru for history, plus a **light** daily DK capture (≤ ~100 requests/day, payout tables for ≤ 40 main contests/day).
  - **No bulk DK crawl, no login, never automate entry or edits.** The user uploads CSVs by hand.
- **Post-game why v0 ships in Phase 1;** the full why engine comes in December.
- **Props** are shown as "our number vs the market", with both percentages, as information. Never "over", "edge" or "lean" (STYLE.md).
- **Prop success bar:** calibrated, beats the naive baselines, and *adds information* (the blend beats the market by > 2 clustered SE). Not parity.
- **Defaults adopted from PLAYERS.md §9:**
  - **Contest archive:** payout tables for main-slate Classic contests with fee ≥ $3 and field ≥ 100, at most 40 a day.
  - **Standings uploads:** optional, the owner's choice.
  - **Ownership display:** bands until ≥ 10 calibrated slates.
  - **Late-news alerts:** page banner only.
  - **Tonight player cards:** Phase 2.
  - **Play-by-play features in live DFS:** allowed once the 2021-26 projection backtest shows no CRPS loss.
  - **N3** (a "teammate left the game" post-game term): deferred to Phase 4.

## 5. Key facts that steer the build (`docs/research/dfs.md`)

- **DK rules:**
  - Classic: PG/SG/SF/PF/C/G/F/UTIL, $50k, **≥ 2 games and ≥ 2 teams**, late swap allowed.
  - Scoring: pts 1, 3PM +0.5, reb 1.25, ast 1.5, stl 2, blk 2, TO −0.5, DD +1.5, TD +3. **A triple-double gets both bonuses (+4.5)**; verify this.
  - Showdown: CPT is 1.5× salary and points, no late swap.
  - Read rules from `rules_json` at build time.
- **From our own data (2015-26):**
  - Teammate DK correlation is about **0.03** (high-usage pairs 0.00). Stacking is weak.
  - **Blowouts:** stars lose −5.5 DK at margins of 26+, almost all of it lost minutes.
  - **Star out:** each remaining top-6 player gains **+3.1 DK**.
  - **Distributions:** normal for stars and role players, gamma for bench.
  - Correlations drift by season.
  - Script: `research/dfs_correlations.py`.
- **Fable review must-fixes (all in v2):**
  - M1: the bottom-up sim overstates variance, so the **copula is the default** and the possession sim must beat it.
  - M2: fit only on seasons before the test season.
  - M8: lock-relative runs and late swap.
  - S14: value-play promotion tests.
- **Game model honest numbers** (`docs/MODEL.md`): gbm_market 68.1% (LL .5933) ties the close (.5918), with no ATS edge. **Never quote the old leaky "69.5%".** The 2025-26 day-by-day replay showed mild **overconfidence (calibration slope ≈ 0.8)**, an open item to investigate.

## 6. Open follow-ups (not in PLAYERS.md)

- **Sync the player data, the PBP and the 2025-26 injury rows to the Mac** after the backfill (§2).
- **Game model in-season overconfidence: investigated 2026-10-21 (`docs/research/game-calibration.md`), not real at season level.** The 0.8 slope came from a replay that stopped 2026-02-20; the full 2025-26 replay gives 0.97 (0.07), all 14 seasons 0.98 (0.02), and the closing line was just as overconfident in Dec/Jan. Missing injury rows: no effect. The model-only part is its departures from the market (2025-26 team-level spread). Pull-toward-close (w ≈ 0.2–0.3) is −2.3 SE log loss but 2024-25 +1.06 SE, so it misses the bar; re-check after 2026-27. Expect the live monitor to warn around January (the market would too).
- **The `predictions` view `latest_pretip_predictions` only shows champion/NULL rows,** and shadow rows have model = version id (MLOPS.md).
- **Opening night is 2026-10-20 (first tip 19:00Z).** On 10-21, check the first real Brief, the props, the referee capture, DFS salaries and `/health`.
- **(Optional) Log in `claude` on the Mac's desktop,** so the writer uses Claude rather than Codex there.

## 7. Docs map

| Doc | What |
|---|---|
| `README.md` | Overview and quick start |
| `docs/DESIGN.md` | Product spec and architecture |
| `docs/MODEL.md` | Game model ladder and honest results |
| `docs/MLOPS.md` | Registry, monitoring, gates |
| `docs/ANALYTICS.md`, `docs/TRENDS.md`, `docs/BRIEF.md`, `docs/STYLE.md`, `docs/WEB.md` | Subsystems |
| **`docs/PLAYERS.md`** | **The approved DFS-first plan v2 (build from §8)** |
| `docs/research/dfs.md`, `docs/research/player-predictions.md` | Research behind the plan |
| `deploy/README.md` | Schedulers and services |
