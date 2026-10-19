# HANDOFF: start here after a context reset

*Last updated 2026-10-19 ~09:50 ET (day before the 2026-27 opener).* Read this whole file first, then `docs/PLAYERS.md` §8 (the task tables). Everything important is in the repo or in memory, not only in old chat.

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

## 2. State of data and jobs

- **Running now on geekom1** (detached, survives compaction): the per-game player backfill, PID 2962632.
  - Command: `courtside backfill players --per-game all --from 2017-18`
  - Log: `var/logs/backfill_players.log`
  - Progress: matchups were at 2,600/11,537 at 09:47 (about 3 h left), then tracking and hustle for 2019-20+, a few more hours.
  - It's resumable: rerun the same command if it dies.
- **When it finishes,** sync to the Mac:
  1. rsync `var/raw/nba_stats/{boxscorematchupsv3,boxscoreplayertrackv3,boxscorehustlev2,...season tables...}` and `var/raw/nba_official/injury_report` and `var/raw/sportsdataverse` to the Mac's `var/raw/`.
  2. On the Mac, rebuild the DB tables from the archives: `courtside backfill players --season-tables` and `--per-game all --reparse` (no network for archived items); `courtside backfill injuries` (skips archived PDFs; **check that it inserts rows for archived PDFs**, and if not, add a reparse path).
  3. Today **the Mac DB lacks**:
     - the 2025-26 official-report rows (177k, rescued on geekom1)
     - all player tables (migration 7)
     - the PBP archive (8,647 games, on geekom1 only)
- **Have (geekom1):**
  - game and player logs 2000-01 → 2025-26
  - closing lines 2007 → 2026
  - cdn PBP 2019-20 → 2025-26
  - official injury reports 2025-12-22 → 2026-06-13 (151/151 game days)
  - season tables, about 1.3M rows
  - sportsdataverse lineups and possessions 1997 → 2026 (parquet, `var/raw/sportsdataverse/`)
- **Live capture since 2026-10-19:**
  - ESPN DK props (`prop_snapshots`, a change-log)
  - Action Network lines (`line_snapshots`)
  - DK salaries plus rules (`dfs_salaries`, raw under `var/raw/draftkings/`)
  - referee assignments
- **Not yet:**
  - RotoGuru history (T0.2)
  - contest payout tables (T0.7)
  - cdn box scores 2019+ (T0.3: only about 13 are archived)

## 3. What to build next (approved plan: `docs/PLAYERS.md` v2 §8)

**Phase 0 (Oct 19–22):** T0.0 (migration **9**, `courtside proj`/`courtside dfs` CLI), T0.1, T0.2 (RotoGuru), T0.3, T0.4, T0.5 (IDs), T0.6 (rules/scoring check), T0.7 (DK starter, swappable and contest payouts), T0.8.

**Phase 1 (to about Oct 27):** T1.0 eval-lite, T1.1 rate state, T1.2 minutes v0.5 (value-play promotion), T1.3 P(play), T1.4 copula v0, T1.5 CP-SAT cash + Top N, T1.6 lock-relative watcher plus `late_news`, T1.7 `/dfs` page plus CSV, T1.8 why v0, T1.9 preseason.

**Phase 1 gate** (from eval-lite):
- DK FP MAE ≥ 3% better than the season average (2024-25)
- not worse than salary-implied on 2019-21
- 80% coverage 76–84%
- minutes MAE ≤ 4.8

If it fails, label the page "beta".

Then Phase 2 (models + backtest, to Nov 14), Phase 2.5 (contest builder, GPP, late swap, Showdown, to Nov 21), Phase 3 (DFS monitoring, standings import), Phase 4 (the full why engine, Dec).

**Suggested first wave**, at most 3 agents on geekom1:
- T0.0 (Sonnet), which unblocks T0.5 to T0.7 and everything in `proj/`
- T0.2 RotoGuru (Haiku/Sonnet; ~1,200 polite requests at 1/s; run from geekom1's home IP)
- T0.6 rules check (Sonnet)

Then T0.5 and T0.7, then Phase 1. Give each agent its exact task row, owned files, acceptance criteria, and these rules:
- no git commits (the orchestrator commits)
- run only their own tests
- heavy work goes to the Mac scratch dir
- never write the prod DB from scratch jobs
- never attach to the user's browser (headless Playwright only)

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
- **Investigate the game model's in-season overconfidence** (slope ≈ 0.8 on the 2025-26 replay). Candidates are early-season priors and missing injury data in the replay. The monitoring will show live evidence.
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
