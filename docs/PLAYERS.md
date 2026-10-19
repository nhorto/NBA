# Courtside players v2: DFS-first projections and optimizer

Status: **design v2, 2026-10-19.** Nothing is built yet, and this version goes to a reviewer before any build starts.

v2 replaces v1, which was projections-first with DFS as one consumer. In v2 the **DraftKings optimizer is the primary deliverable**, and the projections exist to serve it. Props, player cards and the "why" are kept but trimmed.

Inputs: `docs/research/dfs.md` (decisions, rules, sources, correlation study, review), `docs/research/player-predictions.md`, `research/dfs_correlations_output.txt`, `docs/MODEL.md`, `docs/MLOPS.md`, `docs/ANALYTICS.md`.

## 0. Changelog v1 → v2

### 0.1 Owner decisions (2026-10-19)

| Decision | Where it lands |
|---|---|
| The DFS optimizer is the top priority: pro-grade accuracy, honest backtests, simple use, a "why" for every lineup | Whole structure; §2, §5, §7.3 |
| DraftKings only in v1; FanDuel later via CSV | All FD content removed; timing is §9 Q7 |
| Two build modes: **Top N** (optional minimum number of different players) and **Contest builder** (an N-entry portfolio) | §2.2, §2.3, §5.4, §5.5 |
| Light guidance: simulated cash and top-finish odds plus a "sensible stakes" note. No dollar or Kelly sizing | §2.3 |
| Post-game why v0 ships in Phase 1; the full why engine moves to December | §2.8, T1.8, Phase 4 |
| Data policy: RotoGuru history, light DK capture, no crawl or login, never automate entry | §3, §9 |
| 150-max portfolios and automated late swap are lower priority | N defaults to 20 (max 150, slower). Late swap is a manual "re-optimize" action plus a CSV, never automated. |

### 0.2 Review items

| Item | Verdict | Where / why |
|---|---|---|
| **M1** iid sim overstates variance | **Accepted, generalized** | §4.5, T2.5/T2.6: step-0 measurement, game latent η_g, multinomial possessions, Q4 slope in the SMM. Feedback acts on *every* 6-min segment, because Q1–Q3 alone carry ≈ 200 > 154 of margin variance, so Q4-only can't close the gap. Gate G1 falls back to allocation-only. The **copula is the GPP default** until the sim beats it (G2). |
| **M2** sim tuned on test data | **Accepted** | §7.1 L2: everything is fit on seasons < S. Step 0 uses 2019-24, so sim claims are quoted only on 2024-25/2025-26. The "freeze on 2019-21" option is not used for the box path, because those are RotoGuru test seasons. |
| **M3** FanDuel has no data | Accepted (decided) | §1 |
| **M4** DK rule verification | **Accepted** | §5.1, T0.6 (all four acceptance points, plus a RotoGuru cross-check) |
| **M5** Phase 1 cites a Phase 2 harness | **Accepted** | T1.0 eval-lite; RotoGuru backtest T2.1 |
| **M6** minutes targets inverted; P(play) to Phase 1 | **Accepted** | §4.2 targets, T1.3 (uses the 151 rescued report days) |
| **M7** reconciliation unmeetable with OT/m₃ | **Accepted, option (b)** | §4.4: pilot → Newton step → full run. With m₃ feedback, even the regulation mean isn't analytic. |
| **M8** late news | **Accepted** | §2.5, §5.6, T1.6, T2.14: lock−45 and lock−10 runs, `late_news`, a late-swap re-solver |
| S1 prior double-counts last season | Accepted | §4.3: the prior uses seasons ≤ S−2 |
| S2 scenarios by mass | Accepted | §4.1: ≥ 95% mass, cap 16 |
| S3 randomized PIT | Accepted | §7.2 |
| S4 two-way clustering | Accepted | §7.2: game × player-season |
| S5 xFG acceptance | **Deferred** with xFG to Phase 4 (T4.1) | Not on the DFS path (resequence) |
| S6 per-coalition reconciliation in Shapley | **Deferred** with Shapley (T4.3) | Same |
| S7 team and missed-FT rebounds | Accepted | §4.5 Path B |
| S8 market % beside ours | Accepted | §2.8 cards |
| S9 templated one-liners; LLM writes ≤ 30 | Accepted | §2.8, T1.8 |
| S10 T0.1 is done | Accepted | T0.1 becomes verify + rolling capture |
| S11 payout ladder, 25–50k sims, 300 single-worker solves | Accepted | §5.5 |
| S12 40/60 field-proxy mixture, labelled | Accepted | §5.5; the weights are unverifiable until standings are uploaded (T3.2) |
| S13 DFS backtest metrics | Accepted | §7.3 |
| S14 value-play promotion (+8 min test) | Accepted | §4.2, T1.2 acceptance |
| S15 stint failure → box minutes | Accepted, made central | §4.2: the box path is primary (it alone covers RotoGuru seasons) |
| N1 books' team totals as environment | Accepted | §4.4 (`line_snapshots` team totals, from 2026-10-19) |
| N2 preseason role evidence | Accepted (light) | T1.9 (preseason boxes aren't in the DB yet) |
| N3 | Deferred → Phase 4 | A post-game "teammate left the game" term (from `possible_injuries`) joins the full why engine (T4.x); why v0 notes it in text only |
| N4 coach by date | Accepted | T2.2: a `coach_changes` override on top of per-season `team_coaches` |
| N5 PRA props | Accepted | §2.8, §7.5 |
| N6 Showdown captain binary | Accepted | §5.7 |
| Answer: parity bar → calibrated / beats naive / adds information | Accepted | §7.5 |
| Answer: DFS bar (FP MAE vs salary-implied and `fppg`; percentile vs field proxy) | Accepted | §7.3 |
| Answer: boom% is slate-relative | Accepted | §2.1 |
| Resequence (minutes ≫ usage > correlation > environment > defender/refs) | Accepted | §8; xFG, Shapley, full decomposition, defender, refs, matchups, season views → Phase 4 |

No review item is rejected outright. Three are accepted with a different option than suggested (M1 segment-wide feedback, M2 walk-forward over freeze, M7 pilot). Two should-fixes (S5, S6) are deferred because their components are deferred.

### 0.3 Facts found while revising

| Finding | Consequence |
|---|---|
| Migrations 7 and 8 exist (8 has `dfs_salaries`, `line_snapshots`, `prop_snapshots`) | New tables go in **migration 9**; `dfs_salaries` is altered |
| `sources/dfs.py` drops `playerGameAttributes` (id 100 = starter) and `isSwappable` | T0.7 adds them (needed at lock−10) |
| Official reports are rescued: 151 game days, 2025-12-22 → 2026-06-13 | P(play) calibration now (T1.3), plus a live-like slice (T2.4) |
| RotoGuru: one primary position and one salary per player per day, 2014-15 → 2020-21. PBP/stints start 2019-20 | **The DFS backtest runs the box-only engine**. The stint features are tested on projections for 2021-22+. |
| The correlation study's residuals are vs in-season means (hindsight) | Targets are recomputed causally vs the pregame projection (T2.7) |
| The Vegas total explains 4.9% of team DK variance | Environment ranks below minutes and usage |
| The scheduler uses fixed ET times | Lock-relative runs need a watcher (T1.6) |

## 1. Scope and design in one paragraph

**Product scope:**
- the DK **Classic** optimizer (cash, Top N, contest portfolio), with **Showdown** in v1.x;
- DK-point and component projections (pts, reb, ast, 3PM, stl, blk, tov, DD/TD) and minutes;
- props (pts/reb/ast/PRA/3PM) as information only.

**Design:**
- **P(play)** is calibrated on official reports.
- A **minutes engine** redistributes vacated minutes, promotes value plays, and is conditional on the game margin, so blowouts come from the margin draw.
- **Per-minute EB rates** get a usage bump under absences.
- A **joint distribution** per slate: a Gaussian copula with a shared margin draw (default), upgraded to a possession sim only if it wins.
- **CP-SAT** solves cash (P(score ≥ cash line)), Top N (k-best) and contest portfolios (payout ladder vs a labelled field proxy).
- Runs happen at lock−45 and lock−10, plus late-swap re-solves. Every lineup carries templated reasons.

## 2. Product surface (phone-first, DFS page first)

### 2.1 `/dfs/<day>`: slate and player pool
- **Slate picker:** main Classic by default; Showdown in v1.x. A run badge reads "Built 18:20 (lock−40) · 2 late-news players", plus stale-data warnings.
- **Player pool**, sortable:
  - salary and DK positions;
  - P(play) and status, or a starter tick;
  - minutes and mean DK;
  - range (p25–p75 bar; p90 is the "ceiling");
  - value (mean per $1k);
  - **boom%** = P(FP ≥ salary/1000 × v*), where v* is the points per $1k of the slate's simulated top-1% lineups. This makes it **slate-relative**;
  - **popularity (proxy)** as Low/Med/High;
  - a `late_news` dot.
- **Phones** get a card list. **Controls:** lock/ban, min/max exposure, exclude Q/GTD.

### 2.2 Top N lineups
- **Ranked list**, default N = 10, up to 50. Each lineup shows the 8 slotted players, salary used, mean, p10–p90 range, and P(cash) once T2.13 ships.
- **Diff vs #1:** players in and out, Δmean, Δsalary.
- **Setting:** "min different players" d ∈ {1…4}, default 1 (§5.4).

### 2.3 Contest builder
1. **Pick a contest** from the archived lobby (fee, field, max entries, prizes, payout table), or **describe one**: fee, field size, max entries, and a payout curve from presets or pasted rows.
2. **Choose N entries**, up to the per-user maximum (cap 150).
3. **Output:**
   - the portfolio (§5.5) with an exposure table;
   - **simulated cash rate**, **top-1% rate** and **expected payout** for the set;
   - **expected ROI** labelled "*simulated against a modelled field, not real results*", shown as a soft–sharp field range;
   - **dupe risk** (Low/Med/High) per lineup.
4. **Light guidance** (static text):
   - cash contests suit projection-driven builds;
   - single-entry and 3-max GPPs avoid 150-entry pros;
   - large fields carry the most variance;
   - a **"sensible stakes" note**: a strong player still loses over about half of 100-contest stretches, so stake only entertainment money.

   **No dollar or Kelly sizing.**

### 2.4 "Why this lineup" (templated, no LLM)
Up to 3 reasons per lineup, picked from facts by rule. Each number comes straight from the run:
- **Minutes opened:** "+7 min for Hart with Brunson out". Fires when Δ projected minutes ≥ 4 from an absence.
- **Value:** "$4,100 for 24 pts (5.9 per $1k, slate top-5)".
- **Ceiling:** "28% to reach the slate's boom mark".
- **Environment:** "3 players from the slate's highest team total (CHI 121.5)".
- **Lineup level:** "uses $49,800; lowest blowout risk of the top 10".

Reasons are ranked by |contribution to the lineup's objective − slate median|.

### 2.5 Late news and late swap
- **Lock-relative runs:** the build runs automatically at **lock−45** and **lock−10** for each slate (§5.6).
- **`late_news` flag.** A player is flagged when, since the previous run or within 90 minutes of lock:
  - his status changed;
  - he was named a starter or a non-starter against expectation;
  - his projected minutes moved ≥ 3.
- **Banner:** "2 lineups contain late-news players".
- **"Re-optimize unlocked slots"** re-solves with started games' players fixed and downloads a DK edit CSV. The user uploads it by hand.

### 2.6 Showdown (v1.x)
- The same page with a CPT column (1.5× salary and points).
- Top N plus a contest builder for single-game contests (§5.7).

### 2.7 Exports
- **DK CSV** in the upload template (slot columns, `Name (draftableId)`), for new entries and for edits.
- **Never** automated entry, login or edits.

### 2.8 Kept, trimmed
- **Tonight player cards** (Phase 2). Top 4 per team:
  - `27.4 pts · 4.1 reb · 7.2 ast · 35 min` with an 80% bar;
  - one factor chip;
  - status and P(play);
  - the market line as **"Line 26.5 · our chance above 58% · market 52%"**.

  Both percentages are shown (S8). "Over", "edge" and "value" stay banned (STYLE.md).
- **Post-game why v0** (Phase 1), for |actual − projection| ≥ 6 DK or ≥ 5 pts:
  - a log-additive split into minutes, volume (FGA + FTA per minute) and making (pts per shot vs his EB norm), so it sums exactly;
  - a **templated one-liner**: "Hart 31 (our 17): +9 min, 7-of-10 from three";
  - the LLM writes at most **30** short stories a night (S9), under v1 §3.4's fact-check and sign rules.

## 3. Data

### 3.1 RotoGuru history (T0.2)
- **Fetch:** `hyday.pl?game=dk&mon=M&day=D&year=Y` in the HTML view (scsv truncates after 2017), 2014-10-29 → 2021-05. About 1,200 requests at 1/s, archived to `var/raw/rotoguru/dk/<date>.html.gz` and never re-fetched.
- **Rows:** id, primary position, name, starter `^`, DK pts, salary, team, opp, venue, minutes, stats. They are joined to `player_games` on (date, team, normalised name).
- **Slate:** the games where ≥ 80% of rotation players have a salary. A secondary main-slate proxy keeps games tipping ≥ 19:00 ET.
- **Eligibility:** derived from the single position (PG → PG/G/UTIL). It is applied identically to our lineups, the baselines and hindsight-optimal.

### 3.2 Light daily DK capture (T0.7)
- **Already running:** lobby, Classic/Showdown draftables, and rules at each pregame time.
- **Added: payout tables** for **30–40 main-slate contests a day**: all double-ups and 50/50s, the top 10 GPPs by prizes, single-entry and 3-max GPPs, and up to 3 Showdowns.
  - Fetched once before lock from `contests/v1/contests/{id}`, because `entries` reads 0 after completion.
  - Stored fields: fee, `maximumEntries`, `maximumEntriesPerUser`, `isGuaranteed`, `payoutSummary`, attributes.
- **Lock-relative draftables** at lock−50 and lock−12: status, disabled, swappable, news, starter.
- **Not collected:** historical draft groups, leaderboards, ownership, anything behind a login.

### 3.3 Other inputs
- **Official reports:** the 151 rescued days (verified by T0.1), the rolling capture, and a fetch right before each lock run.
- **ESPN injuries.**
- **Book lines and team totals** from `line_snapshots` (N1). **Props** from `prop_snapshots`.
- **Box:** `player_games` back to 2000 (the box path).
- **Stints** for 2019-20+, via T0.3/T0.4.

### 3.4 ID matching (T0.5)
- **Keys:** DK `playerId` (stable across slates; `draftableId` is per slate) and the RotoGuru id → NBA `player_id`, persisted in `player_ids(source, source_id)`.
- **Method:** name + team + date against `team_rosters` and the recent game rosters, reusing `injuries.match_player` (accents, suffixes, nicknames, FUZZY_MIN 0.86 with a 0.05 margin).
- **Same-team near-duplicates** (Jalen/Jaylin Williams) need an exact first-token match.
- **Unmatched players** raise a data check and go to `player_aliases`.
- **Targets:**
  - DK ≥ 98% of salaried players automatically, and 100% of players with projected minutes ≥ 10 after aliases;
  - RotoGuru ≥ 99% of salaried player-games.

## 4. Modeling

### 4.1 Availability: P(play) (T1.3, Phase 1)
- **Inputs:** the latest official status before the run; ESPN status; DK `status` and starter flag; hours to tip; and a reason class (rest / injury / illness / personal / G-League), parsed from the report text.
- **Model:** a Beta posterior per (status × hours-to-tip bucket {> 6, 2–6, 0.75–2, < 0.75} × reason class).
  - Prior: the current `injuries.PLAY_PROB` value with 20 pseudo-observations.
  - Updated with snapshot status → played or not, using the 151 rescued days plus live data, refit weekly.
  - A cell with n < 30 falls back to its status-level posterior.
- **Overrides:**
  - DK starter flag = 1 → P = 0.99.
  - Unlisted regulars: P = 0.99.
- **Scenarios (S2):** per team, enumerate joint availability outcomes of the uncertain players in descending probability until ≥ 95% of the mass is covered, capped at 16. The remaining mass is folded into the nearest enumerated scenario.
- **Acceptance (T1.3):**
  - reliability within ±0.08 per status with n ≥ 50;
  - Brier ≤ the static table, reported with game-clustered SE.

### 4.2 Minutes engine (the centerpiece)
**Minutes are modelled conditional on the margin.**
- **minutes = f(pregame features) + g_tier(|final margin|, won/lost)**, where g is a blowout curve fit on box data for seasons < S. Shape to reproduce: starters −3.5 min at 26+.
- At prediction time, g is evaluated at each sim's *drawn* margin.
- This box path runs back to 2000, covers every RotoGuru season, and is the S15 fallback.
- From 2019-20, Q1–Q3 stint shares plus Q4-by-|m₃| shares can replace g on the sim path.

**Base minutes model** (Phase 1 v0: EWMA plus rules; Phase 2 v1: LightGBM refit per season walk-forward, pre-*d* features only):

| Group | Features |
|---|---|
| role | fast/slow minutes EWMA (α 0.2/0.02), last game, median of last 5, started last game, starts in last 10, team minutes rank |
| return | games since returning from ≥ 3 missed games, games missed |
| absences (per scenario) | vacated minutes: total, and in his position group G/F/C; his shrunk historical share of each absent teammate's minutes; **depth rank within his position group after removing absent players** |
| coach | rotation concentration (top-7 share, count of 30+ minute players), coach by date (N4), 40 pseudo-games of shrinkage |
| schedule | b2b, rest, 3-in-4, age × b2b, his rate of sitting second legs |
| news | DK starter flag (lock−10 only), status, reason class |
| early season | preseason starts and minutes in the last 2 preseason games, for the first 10 games (N2, T1.9) |

**Value-play promotion (S14).** Proportional renormalisation spreads vacated minutes too thinly and misses the biggest DFS edge. Instead:
1. **Next man up.** For each absent rotation player j, the top remaining player in j's position group, ranked by (starts in the last 10, minutes EWMA, historical share of j's minutes), is lifted toward the coach's median starter minutes for that group, shrunk 50% to his own EWMA. The remaining vacated minutes go to the group, then to adjacent groups (G↔F↔C), by historical share when j sat.
2. **Team total** of 240 minutes, via `players.allocate_minutes` water-filling with a 40-minute cap.
3. **Starter flag.** At lock−10, an unexpected confirmed starter is re-run through step 1.

**Tests (T1.2):**
- **Two starters Out:** the top bench player's projected minutes rise **≥ 8** vs healthy (mean over ≥ 200 team-games), and the realised rise is within ±2.5 of the projected one.
- **Star absence:** the remaining top-6 projected ΔDK is **+3.1 ± 0.6**.

**Noise.** The share is multiplied by exp(ξ), with ξ ~ N(0, σ²_tier), plus an early-exit mixture with probability q₀ + q₁·PF-rate, scaled by U(0.35, 0.85). σ_tier, q₀ and q₁ are fit on seasons < S (M2).

**Targets (M6):**
- minutes MAE for projected ≥ 15 is **≤ 4.5 in the avail backtest** (v1, Phase 2), with v0 ≤ 4.8;
- ≥ 8% better than the rung-2 EWMA on the same rows;
- the **live-like slice** (2025-12-22 → 2026-04-12, real official reports instead of the oracle) is reported separately and is expected to be worse;
- **live MAE within +0.5 of the live-like slice.**

### 4.3 Rates with usage redistribution
- **Quantities:** per-minute DK components (pts via 2PA/3PA/FTA and makes, reb, ast, stl, blk, tov). The sim path uses per-possession rates.
- **Estimate (EB):** `r̂ = (Σ_g w_g x_g + κ·μ_prior)/(Σ_g w_g n_g + κ)`.
  - Weights have a 120-day half-life in season, and last season is multiplied by 0.6.
  - **S1:** μ_prior is a ridge on position group, height, weight, age, draft slot, and the rate over **seasons ≤ S−2** (never last season, which is already in the likelihood).
  - κ is set per stat by beta-binomial method of moments on seasons < S.
  - The age curve is the delta method, as in v1.
- **Usage under absences:**
  - **Phase 1:** a role-tier per-minute bump fit on star-absence natural experiments in seasons < S. The study's values to reproduce: the other star +7.6%/min, non-star rotation +4.2%/min.
  - **Phase 2 (T2.3):** individual with/without factors φ_ij from stints (2019+, `φ = φ_emp·n_off/(n_off+K_φ)`, K_φ chosen walk-forward) and box-level with/without for earlier seasons. Acceptance: on "teammate sat" games, the usage log-likelihood beats the tier bump by > 2 SE.
- **Opponent and pace (Phase 2, T2.8, low priority):** opponent-by-position indices shrunk as in v1 (3P% allowed K ≈ 5,000). They ship only if DK CRPS improves by > 1 SE.

### 4.4 Environment and reconciliation
- **Environment (N1):** team implied points come from the book team totals in `line_snapshots`, using the median across books at the run time. Without them, they are (μ_T ± μ_M)/2 from `gbm_market`.
- **Scaling (copula and Phase 1 path):** each team's projected points are scaled to the implied total, with α split between pace (all counting stats) and efficiency (points only). The prior guess is α = 0.5, fit on seasons < S.
- **Sim path (M7):**
  - Run a **2,000-sim pilot**.
  - Measure the gap between Σ E_sim[pts_i] (including OT and the m₃ feedback) and the target.
  - Take one Newton step on the efficiency shift δ_t, using the pilot's numerical derivative. Then run the full set.
- **Acceptance:**
  - |Σ_i E[pts_i] − target| ≤ 0.5 for every team-game.
  - A pre-reconciliation gap > 6 raises a data check (stale report or mis-matched name).
  - |δ_t| is capped at 0.15 logit, with any remainder reported.

### 4.5 Joint distribution

**The facts that set the bar** (study, 2015-16 → 2025-26):
- teammate DK residual r = 0.034 (high-usage pairs 0.000, C–C −0.09, star–bench −0.03, bench–bench +0.09);
- opponents 0.011;
- drift by season, 0.017 → 0.059;
- blowouts mostly cost minutes (stars −5.5 DK at |m| ≥ 26; losing starters −4.7 at |m| ≥ 16);
- normal marginals fit stars and role players, gamma fits bench, and CV falls 0.79 → 0.24 with level.

Correlation is a weak lever, so **the marginals and the shared margin draw matter most.**

**Path A: copula (T1.4 v0 → T2.10 v1; the GPP default).** Per sim and game:
1. Draw (H, A) errors from a bivariate normal with the game model's OOF sd(margin), sd(total) and correlation (T0.8, seasons < S). This gives the margin m_s.
2. Draw the availability scenario.
3. Draw minutes: base + g_tier(|m_s|, sign) + noise, then water-fill.
4. Draw outcomes given minutes (means = minutes × rates, with the usage bump and a winner/loser per-minute shift):
   - **v0** draws DK FP directly: normal for projected ≥ 25, gamma below, with CV-by-level fit on seasons < S;
   - **v1** draws the **components** (pts, reb, ast, stl, blk, tov, 3PM) with a within-player 7×7 Gaussian copula per role tier, then scores them, so DD/TD are exact and props share the draws.
5. Couple players with a Gaussian copula: the role-pair × position-pair matrix plus opponent entries, from T2.7 (last 1–3 seasons < S, **residuals vs our pregame projection**).

Path A takes team variance top-down from the market-calibrated game model, so M1's over-dispersion can't occur. It costs 15–30 s per 15-game slate. What it gives up: team totals match only in expectation.

**Path B: possession sim (T2.5 → T2.6), measurement first.**

**Step 0** measures on 2019-20 → 2023-24 PBP:
- Var(team | line) and Cov(H, A | line);
- the per-possession points variance;
- the **lead restoring force**, E[Δm over the next 6 minutes | m];
- the slope b₄ of final margin on m₃ (expect < 1);
- E[Q4 pace | |m₃|];
- leader/trailer Q4 efficiency;
- team-rebound and missed-FT-rebound rates (S7).

The arithmetic behind it:
- The market errors (sd 17.7 total, 12.4 margin) imply Var(team) ≈ 117 and Cov ≈ 40.
- iid possession noise alone is ≈ 130 per team, before pace (≈ 25) and rate uncertainty.
- So an iid sim gives sd(margin) ≈ 16.4. Real games need a restoring force and a shared latent.

**Per sim:**
- **Latents:** a game latent η_g ~ N(0, τ_g²) on both teams' efficiency; team latents η_t; player shooting latents ζ_i; pace P_s ~ N(P̄, σ_P²).
- **Regulation** runs as eight 6-minute segments; in segments 7–8, pace shifts by E[pace | |m₃|].
- **Each possession** draws a **multinomial terminal outcome** over {2PA, 3PA, FT trip, TOV} from the on-court rates. Independent Poissons are not used, because they double-count pace variance.
  - Makes are binomial on logit p̂ + δ_t + η_g + η_t + ζ − ρ·m_{k−1}·s_t (s_t = +1 for the leader), so the **restoring force acts in every segment**. Q4 adds a leader-coasting term β_q(|m₃|).
  - Misses continue as OREB with Binomial(reboundable misses, oreb%), for 3 rounds (residual mass < 0.3%).
  - Reboundable misses include the last missed FT, and a team-rebound share is removed before player allocation (S7).
- **Allocation** is Dirichlet-multinomial by on-court weight. Minutes come from §4.2: Q1–Q3 and Q4-by-|m₃| shares on the stint path, else box minutes pro rata.
- **OT** periods are 5/48 of P_s each, repeated while tied.

**SMM (M1e, M2):**
- σ_P, τ_g, τ_e, τ_p, ρ and β_q are fit **per test season on seasons < S**.
- **Fit targets:** Var(H | line), Cov(H, A | line), sd(margin) and sd(total) within ±3%; b₄ within ±0.05; Q4 pace within ±1 per bin; P(OT) 5.5–6.5%; FG% dispersion.
- **Validation targets (not fitted)**, from T2.7 on seasons < S:
  - role-pair and opponent correlations within ±0.02;
  - the blowout curve within ±1.0 DK per bin;
  - the star-absence bump within ±0.6;
  - CV by level within ±0.03.
- All of these are quoted only on 2024-25 and 2025-26.

**Gates:**
- **G1:** if the τ's reach 0 and the variance targets still fail, Path B runs in **allocation mode**. Team scores come top-down as in Path A, and the possessions only split them among players.
- **G2:** Path B replaces Path A for DFS only if, on 2024-25/2025-26, player DK CRPS is not worse by more than 1 SE, lineup-score calibration is at least as good, and the realised proxy payout is not worse. Otherwise it serves props and the why.

**Monte Carlo and seeds.**
- DFS uses **S = 25,000** sims, with 50,000 for portfolios of N > 20. At p = 0.01 the SE is 0.0006.
- Seeds are `hash(slate_id, run_kind, model_version)`. These are common random numbers, so projections move between runs only when the inputs move.
- Per-sim player FP is stored as float16 `npz`: about 8 MB per slate, pruned after 8 days.

## 5. DFS engine (`proj/dfs/`)

### 5.1 Rules and scoring (T0.6, M4)
- **Scoring constants** live in `sources/dfs.CLASSIC_SCORING`, because DK's point values aren't in any JSON. They are verified two ways:
  1. **FPPG reconciliation:** for **≥ 50 draftables** per check, recompute DK's `fppg` (attribute 219) from `player_games` over the matching window (season to date, or last season before ~5 games). Accept if |Δ| ≤ 0.1 for ≥ 95% of them. Players with ≥ 10 triple-doubles in the window discriminate between a 4.5 and a 3.0 TD bonus (about 0.5 FPPG apart).
  2. **RotoGuru DK pts:** an exact match on ≥ 99.5% of joined player-games, with ≥ 20 triple-doubles among them.

  The check reruns weekly as a data check.
- **Rules** come from the archived `rules_json` at build time and are **never hard-coded**: slots, salary cap, `gameCount.min`, and **`teamCount.min` (= 2)**.
- **Eligibility** is per slot from the draftable rows: G = {PG, SG}, F = {SF, PF}, UTIL = any. **`isDisabled` rows are excluded**, and players whose game has started are locked.

### 5.2 Solver
- **OR-Tools CP-SAT** (Apache-2.0, arm64 wheels). The variables are binary x_{i,slot} over eligible slots. Constraints:
  - each slot filled once, each player at most once;
  - salary ≤ cap;
  - ≥ `gameCount.min` games and ≥ `teamCount.min` teams (indicator variables);
  - locks and bans.

  FP is scaled ×100 to integers. The roster model is about 200 lines.
- **`pydfs-lineup-optimizer` is the test oracle:** equal optimal objectives on 50 slates (RotoGuru and live). It isn't used in production: its rules are hard-coded, the CBC backend is about 10× slower, and it has no simulation objectives.
- **Concurrency:** each solve uses `num_workers=1`, run in a thread pool of 2 on the Mac.

### 5.3 Cash
- **Primary objective:** maximise **P(score ≥ c_s)** over the shared sims.
  - c_s is the cash line in sim s: the field-proxy (§5.5) score at the contest's paid fraction from its payout table (e.g. 44.4% for a double-up).
  - Candidates: k-best (§5.4, 100 each) under μ_i − λ·(μ_i − p25_i) for λ ∈ {0, 0.15, 0.3, 0.5}, each scored on the sims. The best is returned, with its P(cash).
- **Fallback** (Phase 1, before the field proxy exists): maximise Σ(μ_i − λ·(μ_i − p25_i))·x_i, with λ chosen on the RotoGuru backtest (default 0.3).
- **Availability:**
  - At lock−45, players with P(play) < 0.85 are excluded *only if their game locks before the next run*.
  - Later-game Q players are allowed, with an automatic swap candidate recorded for §5.6.

### 5.4 Top N
- **k-best by no-good cuts:** after solution L_k, add Σ_{i∈L_k} x_i ≤ 8 − d (d = "min different players").
- N solves are sequential. Lineups are ranked by mean, then by P(cash) if available.
- **Display:** diff vs #1 and the p10–p90 range from the sims.

### 5.5 GPP portfolio (contest builder)
**1. Candidates (S11).** 300 CP-SAT solves on the 2-thread pool:
- 150 **scenario objectives** 0.5·μ_i + 0.5·FP_{i,s}, for random s;
- 100 **ownership-penalised** objectives μ_i − λ·own_i, with λ ∈ U(0, 0.5·σ_μ);
- 50 **noise** objectives μ_i·exp(N(0, 0.12²));
- plus the Top-50 by mean.

**2. Ownership proxy** (no ownership data exists, so it is labelled "proxy" everywhere).
- **Model:** logit own_i = a + b₁·value_z + b₂·position projection rank + b₃·salary_z + b₄·**injury_replacement** (promoted with Δmin ≥ 6) + b₅·`fppg`_z + b₆·team-total_z, normalised to Σ = 800%.
- The b's start hand-set and documented. They are refit by beta regression on uploaded `%Drafted` once ≥ 10 slates exist (T3.2).

**3. Field proxy (S12).** F = min(field size, 20,000) lineups:
- **40% noisy-optimal:** 200 CP-SAT solves on μ_i·exp(N(0, 0.15²)), resampled with replacement. This reproduces optimizer duplication.
- **60% value-greedy with chalk tilt:** sequential sampling ∝ own_i^1.5 × value_i, subject to the slot and cap constraints, rejecting lineups below 97% of the cap.
- The weights 40/60 and the tilt are **assumptions**.
- **ROI is reported under three field strengths:** soft (20/80), medium (40/60), sharp (70/30). The page shows medium, with the soft–sharp range.

**4. Payout ladder.**
- Use the contest's real `payoutSummary`. A described contest uses its preset curve; with none, the fallback is the top-20% / 1% / 0.1% ladder.
- Per sim, store the field-proxy score at each payout-tier boundary rank (F scaled to the field size): a tiers × S matrix built in chunks.
- A lineup's payout in sim s is a tier lookup. Ties and our own entries split or occupy positions jointly.

**5. Selection.**
- Greedily add the candidate with the largest **marginal expected payout** of the portfolio, averaged over S = 25–50k.
- Constraints: overlap ≤ γ shared players with every chosen lineup (default 5 of 8) and per-player exposure caps (default max 60%, plus user min/max).

**6. Duplication.**
- E[dupes_L] = field size × q(L), where q(L) ≈ c·Π_{i∈L} own_i × h(salary left).
- c and h are calibrated on the field-proxy sample: exact matches for common lineups, and the product form for rare ones.
- Expected dupes split the payout in step 5. The indicator reads Low < 0.3, Med 0.3–2, High > 2.

### 5.6 Lock-relative runs and late swap (T1.6, T2.14)
- **Watcher.** A launchd/systemd job on the Mac fires every 5 minutes from 16:00 to 23:59 ET. Each tick:
  - for each archived slate, triggers `courtside dfs run --slate X --kind lock45|lock10` when now ∈ [lock−47, lock−40] or [lock−12, lock−7];
  - first refreshes the official report, ESPN injuries and draftables (§3.2);
  - is idempotent on (slate, kind).
- **Late swap.** For each later game tip in a slate, a `swap` run fires at tip−10, but only if `late_news` touches a player in a saved lineup.
- **Re-solver:** players in started games are fixed. The unlocked slots are re-solved under the same objective with the remaining cap. Portfolio overlap and exposure are re-checked against the fixed parts. The output is a DK edit CSV.
- No automation of DK beyond the CSV.

### 5.7 Showdown (T2.15, N6)
- Binary c_i (captain) and u_i (UTIL) per player, with c_i + u_i ≤ 1, Σc = 1 and Σu = 5. Salary is Σ(1.5·s_i·c_i + s_i·u_i) ≤ cap. Score is 1.5·FP for the captain. ≥ 2 teams comes from `rules_json`.
- One game, so the shared margin draw and the opposing-star correlation matter more. Path B is preferred here once G2 passes.
- The field proxy gets a captain ownership term, so captain chalk is modelled separately.
- No late swap (DK rule).

### 5.8 Runtime budgets (Mac mini M4, loaded, 2 threads)

| Step | Budget |
|---|---|
| nightly state (rates, minutes features, ownership refit) | ≤ 3 min |
| projection, 15-game slate, Path A, 25k sims | ≤ 30 s |
| projection, Path B (pilot + 10k per game) | ≤ 60 s (limit 120) |
| cash + Top-20 | ≤ 10 s |
| GPP N = 20: 300 candidates + 200 field solves (~0.1–0.2 s each, 2 workers) ≈ 50 s; field 20k sampled ≈ 10 s; tier matrix + greedy over 25k sims ≈ 30 s | **≤ 120 s** |
| GPP N = 150 at 50k sims | ≤ 300 s |
| late-swap re-solve | ≤ 30 s |

## 6. Data model: migration 9 (owner T0.0)

| Table | Key columns |
|---|---|
| `dfs_salaries` (alter) | + `starter` INTEGER, + `swappable` INTEGER |
| `player_ids` | (source, source_id) PK, player_id, name, team, first_seen, last_seen, how, confidence |
| `player_aliases` | (source, name_norm, team) PK, player_id, note |
| `dfs_hist_salaries` | (game_date, rg_id) PK, player_id, name, pos, team, opp, home, starter, salary, dk_pts, minutes, raw_stats, raw_path |
| `dfs_slates` | slate_id PK (draft group), nba_day, game_type_id, tag (main / early / late / showdown), game_ids_json, lock_utc, rules_json |
| `dfs_contests` | contest_id PK, slate_id, name, entry_fee, field_size, max_per_user, guaranteed, payout_json, attrs_json, captured_at, raw_path |
| `player_projections` | id PK, game_id, slate_id, player_id, run_id, model_version, role, p_play, minutes_json, inputs_json |
| `player_projection_stats` | (projection_id, stat, cond) PK; stat ∈ min, pts, reb, ast, fg3m, stl, blk, tov, pra, dk; cond ∈ played / all; mean, sd, q05…q95 |
| `dfs_runs` | run_id PK, slate_id, kind (lock45 / lock10 / swap / manual), created_at, model_version, sim_path, path (A/B), late_news_json, seconds |
| `dfs_ownership_proxy` | (run_id, player_id) PK, own, band, version |
| `dfs_lineups` | id PK, run_id, mode (cash / topn / gpp / showdown / swap), contest_id, rank, slots_json, salary, mean, p10, p50, p90, p_cash, p_top1, e_payout, dupes, why_json, actual, pct_field_proxy, ratio_hindsight |
| `dfs_contest_results` | (contest_id, rank, entry_hash) PK, points, lineup_text, payout, uploaded_at (manual uploads) |
| `dfs_contest_ownership` | (contest_id, player_id) PK, name, pct_drafted, fpts |
| `prop_compare` | (projection_id, market, line, captured_at) PK, mkt_p_above, our_p_above |
| `postgame_why_v0` | (game_id, player_id) PK, expected, actual, terms_json, text |
| `stints`, `player_game_oncourt` | as in v1 (Phase 2) |
| `player_metric_snapshots`, `dfs_metric_snapshots` | (as_of, model, metric, window) PK, n, value, se, details_json |

## 7. Evaluation

### 7.1 Leakage rules (each one enforced by a test)
- **L1. No lookahead.** Scrambling all box, PBP, salary and injury rows on or after a cut date leaves projections and lineups up to that date bit-identical.
- **L2. Fit < S.**
  - Every model coefficient and every sim or copula parameter (σ_tier, q's, κ, CV-by-level, correlation matrix, SMM τ's, ρ, ownership b's) is fit on seasons < S inside the walk-forward.
  - Box-path hyperparameters (κ, half-lives) are frozen on **2010-11 → 2013-14**, before every RotoGuru season.
  - Stint-path hyperparameters (K_φ) are frozen on 2019-20/2020-21 and are tested only on 2021-22+.
- **L3. Season aggregates are lagged one season or rebuilt causally.** This covers `matchup_rollup`, `season_matchups`, `pt_*`, `synergy_playtypes`, `hustle_season`, role tiers, and the correlation study. `player_index` positions are a current snapshot (a documented minor leak; the DK/RotoGuru position is preferred).
- **L4. Availability.**
  - The backtest uses the `avail` oracle for regulars with expected minutes ≥ 20. It is labelled "known availability", which equals a perfect lock−10.
  - The **live-like slice** uses real reports.
  - The leak probe from v1 is kept: < 0.5% MAE gain from availability-only features.
- **L5. Starters.** The lock−45 mode never sees starters. Lock−10 mode uses the actual starters as a proxy for announced ones (≈ 99% identical), labelled as such.
- **L6. Never features:** salaries (they are only benchmarks and ownership inputs), props, same-game facts, and closing lines beyond what `gbm_market` and book team totals give at run time.

### 7.2 Projection metrics
- **Measures:** MAE, RMSE, CRPS from samples, bias, 50%/80% coverage, PIT.
- **Randomised PIT for counts (S3):** u = F(y−1) + V·(F(y) − F(y−1)), V ~ U(0,1).
- **SEs** are **two-way clustered by game and player-season** (Cameron–Gelbach–Miller, S4).
- **Population:** projected minutes ≥ 15 and played, for the conditional metrics. For DK FP, the **unconditional** population is every salaried player with projected minutes ≥ 10, with a DNP scored 0. Never filter on actual minutes.
- **Benchmarks:**
  - season-to-date average (this equals DK `fppg`; historically reconstructed);
  - last-10;
  - **salary-implied** (a per-season linear fit of DK pts on salary, on seasons < S);
  - rung-2 EWMA × per-minute rates;
  - live: DK `fppg` itself.

### 7.3 DFS backtest (T2.1, T2.17)
- **Design:** walk-forward over the RotoGuru seasons, test **2015-16 → 2020-21** (about 1,000 slates), with the box path, Path A, and causal state day by day. Each slate runs cash, Top N and GPP (N = 20, medium field).
- **Metrics:**
  - FP MAE/CRPS vs salary-implied, the `fppg` equivalent and last-10;
  - lineup score / hindsight-optimal (same eligibility);
  - percentile vs the field proxy at all three strengths;
  - a paired comparison vs the same optimizer fed salary-implied or `fppg`;
  - realised cash rate vs the proxy cash line (the 44.4% field score);
  - calibration of the simulated lineup-score distribution.
- **Acceptance** (significant at 2 SE, two-way or slate clustered):

| Metric | Target |
|---|---|
| DK FP MAE (unconditional) | ≥ 5% better than the best of salary-implied / `fppg`-equivalent / last-10 |
| DK FP CRPS | ≥ 5% better than a naive-mean + tier-CV normal |
| 80% / 50% coverage (DK FP) | 77–83% / 47–53%; each role bucket 74–86% |
| PIT deciles (randomised) | each 8–12% |
| cash lineup vs `fppg`-optimizer lineup | mean paired Δ > 0 at 2 SE; wins ≥ 55% of slates |
| hindsight ratio (cash) | reported; must beat both baseline optimizers |
| lineup-score calibration | PIT deciles 8–12%; 80% coverage 76–84% |
| GPP simulated vs realised top-1% (vs field proxy) | ratio within 0.7–1.3 |
| minutes MAE (≥ 15) | ≤ 4.5 avail; ≥ 8% better than EWMA |

- **Projection-only backtest 2021-22 → 2025-26** (no salaries): the stint path (T2.2–T2.6) must not be worse on CRPS than the box path before it is used for DFS.

### 7.4 Live tracking (from 2026-10-20)
- **Every lock−10 build is stored:** cash, Top-N and default GPP. The nightly scoring gives realised points, hindsight ratio, field-proxy percentile, the comparison against the `fppg` optimizer, and FP MAE vs DK `fppg` and salary-implied.
- These feed `dfs_metric_snapshots` (windows 7d, 30d, season) and a `/health` DFS tab.
- **Real ground truth:** the owner may upload DK contest standings CSVs from contests he entered. They contain ranks, points, lineups, `%Drafted` and FPTS. From them we get:
  - the **real cash line**, compared with our predicted c_s distribution (PIT);
  - **real top-1% scores**;
  - **real ownership**, which calibrates the proxy (Phase 3: MAE and rank correlation of the proxy are reported once ≥ 10 slates exist).

### 7.5 Props bar (Fable answer, N5)
- **Markets:** pts, reb, ast, PRA, 3PM, using the distribution **conditional on playing**.
- **Three criteria:**
  1. **Calibrated:** Spiegelhalter |z| < 2.5 on P(above line), n ≥ 500.
  2. **Beats naive:** log-loss better than season-average and last-10 distributions.
  3. **Adds information:** the logit-average of ours and the no-vig market beats the market alone by > 2 clustered SE, with both taken at the **same snapshot time**.
- Parity with the market is *not* the v1 bar.

### 7.6 What we can and can't claim
- Simulated ROI isn't real ROI, and the field is a proxy.
- A 19%-ROI player still loses money over 48% of 100-GPP stretches, so the owner's occasional entries **cannot** measure ROI. They measure ownership and cash lines.
- The RotoGuru backtest is optimistic on minutes (known availability) and pessimistic on eligibility (single positions). Honest live numbers start 2026-10-20, and `/health` says so.
- With ~1,000 slates, a top-1% rate has an SE of ≈ 0.3 pp: enough to tell 1% from 2%, but only against the proxy field.

## 8. Build plan

- **Code:** `src/courtside/proj/` and `proj/dfs/`. Tests: `tests/test_proj_*.py`, `tests/test_dfs_*.py`.
- **Ownership of shared files:** `db.py` and the `cli.py` hooks belong to T0.0. `collect.py` and `sources/dfs.py` belong to T0.7, then are released.
- **Legend:** ★ critical path; ∥ parallel; O / S / H = Opus / Sonnet / Haiku; **Mac** = heavy, run on the Mac mini (`~/Developer/NBA`, push from geekom1).
- **Agent limit:** at most **3 agents at once on geekom1**, and one heavy job at a time.
- **Priority order:** P(play) + minutes ≫ rates with usage > joint correlation (GPP only) > pace/opponent > defender/refs.

### Phase 0: foundations (Oct 19–22)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T0.0 ★ S | migration 9, `courtside proj` / `courtside dfs` CLI | `db.py`, `cli.py`, `proj/__init__.py`, `proj/cli.py` | §6 → schema | idempotent migrate; `test_db` green |
| T0.1 ∥ H | verify rescued reports + rolling capture (S10) | `backfill_injuries.py` (checks only) | `injury_snapshots` → data-check report | 151 game days present; parse ≥ 99% of rows; name match ≥ 98%; a report capture is scheduled before every lock run |
| T0.2 ★ H Mac | RotoGuru backfill + parse | `sources/rotoguru.py`, `backfill_rotoguru.py` | HTML → raw + `dfs_hist_salaries` | ≥ 98% of dates 2014-10-29 → 2021-05; ≥ 99% joined to `player_games`; slate table per §3.1 |
| T0.3 ∥ H | cdn box backfill 2019+, PBP rsync to the Mac | `backfill_pbp.py` (box kind) | cdn → raw | ≥ 99% of PBP games have a box |
| T0.4 ∥ S Mac | stints store (Phase 2 input) | `analytics/stints_store.py` | → `stints`, `player_game_oncourt` | ≥ 99% of games validate |
| T0.5 ★ S | ID mapping DK/RotoGuru → NBA | `proj/ids.py` | §3.4 | §3.4 rates; Jalen/Jaylin test |
| T0.6 ★ H | rules + scoring verification (M4) | `proj/dfs/rules.py` | `rules_json`, draftables, RotoGuru → rule objects + check | §5.1: ≥ 50 FPPG within 0.1; RotoGuru exact ≥ 99.5% incl. ≥ 20 TDs; `teamCount`, disabled, eligibility tests |
| T0.7 ★ S | DK capture: starter/swappable, contest payouts, lock-relative refresh | `sources/dfs.py`, `collect.py` (dfs step) | lobby → `dfs_contests`, `dfs_slates` | 30–40 contests/day captured before lock; ≤ 60 DK requests/day added |
| T0.8 ∥ H | GBM total OOF sd + margin–total residual corr per season | `model/gbm.py` (2 attrs) | — | values logged per fold; rung-3 tests green |

### Phase 1: cash + Top N live (Oct 20 – 27)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T1.0 ★ S Mac | **eval-lite** (M5): walk-forward 2024-25 (fit < 2024-25) and RotoGuru 2019-20/2020-21 | `proj/evallite.py` | projections → report | runs in ≤ 20 min; Phase 1 gate below |
| T1.1 ★ S | EB per-minute rate state, nightly | `proj/state.py` | box → `rate_state` | no-lookahead test; ≤ 60 s |
| T1.2 ★ O | minutes v0.5: EWMA + vacated minutes + promotion + margin-conditional g + starter flag | `proj/minutes.py` | state, scenarios → minute distributions | S14 ≥ 8-min test; star-absence +3.1 ± 0.6 DK; avail MAE ≤ 4.8 |
| T1.3 ∥ S | P(play) recalibration (M6, was T2.9) | `proj/availability.py` | reports → Beta cells | §4.1 |
| T1.4 ★ S | Path A v0 copula (tier correlations), reconciliation to team totals, store | `proj/copula.py`, `proj/reconcile.py`, `proj/store.py`, `proj/pipeline.py` | → sims, `player_projection_*` | DK FP 80% coverage 76–84% on 2024-25; ≤ 30 s per slate |
| T1.5 ★ S | CP-SAT engine: cash fallback, Top N, min-different, lock/ban, exposure; pydfs oracle | `proj/dfs/optimizer.py`, `proj/dfs/topn.py` | rules, projections → lineups | pydfs equal on 50 slates; Top-20 ≤ 5 s |
| T1.6 ★ S | lock-relative watcher, `late_news`, `dfs_runs` | `proj/dfs/runner.py`, `deploy/` (watcher units) | schedule, draftables → runs | replay of 3 captured slates fires inside the windows; idempotent; late_news unit tests |
| T1.7 ∥ S | `/dfs` page v1: pool, Top N, cash, why-this-lineup, DK CSV | `web/dfs_view.py`, `templates/dfs.html`, `proj/dfs/why.py`, `proj/dfs/export.py` | runs → page/CSV | 390 px light/dark; CSV accepted by the DK template (owner manual check) |
| T1.8 ∥ S | post-game why v0 + ≤ 30 LLM stories | `proj/why_v0.py`, `brief/prompts/player_why.md` | box vs projection → `postgame_why_v0` | exact sums; 50 samples: 0 sign violations, 0 unsupported numbers |
| T1.9 ∥ H | preseason role evidence (N2) | `proj/preseason.py` | cdn preseason boxes → features | flags for players with ≥ 1 preseason start |

**Phase 1 gate** (on eval-lite):
- DK FP MAE ≥ 3% better than the season-to-date average on 2024-25 (avail);
- not worse than salary-implied on 2019-21;
- 80% coverage 76–84%;
- minutes MAE ≤ 4.8.

If the gate fails, the `/dfs` page ships with a "beta" label.

### Phase 2: model v1 + backtest harness (Oct 28 – Nov 14)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T2.1 ★ S (O review) Mac | RotoGuru DFS backtest harness: walk-forward, baselines, hindsight, random-lineup percentile, leak tests | `proj/backtest.py`, `proj/metrics.py` | history → report | reproduces T1.0; L1–L6 tests; leak probe < 0.5% |
| T2.2 ★ O Mac | minutes v1 GBM (+ coach by date N4, learned promotion, S15 fallback) | `proj/minutes.py` (takes over) | → | §4.2 targets; S14 still passes |
| T2.3 ∥ O | usage v1: with/without φ | `proj/usage.py` | stints, box → φ | > 2 SE over the tier bump |
| T2.4 ∥ S | live-like slice eval (2025-12-22 → 2026-04-12, real reports, lock−45 mode) | `proj/backtest.py` (slice mode, after T2.1) | → report | minutes MAE and DK CRPS reported vs the avail backtest; sets the live alert baseline (M6) |
| T2.5 ★ O Mac | **sim step 0 measurement** (M1a) | `research/sim_step0.py`, `docs/research/sim-step0.md` | PBP 2019-24 → targets | all §4.5 step-0 quantities with CIs |
| T2.6 ★ O Mac | Path B sim v1 + SMM per fold + pilot reconciliation (M7) | `proj/sim.py`, `proj/smm.py` | → | §4.5 fit/validation targets; G1/G2 verdict written |
| T2.7 ∥ S Mac | causal correlation study v2 (residual vs projection, tiers to date, per season < S) | `research/dfs_correlations_v2.py`, `proj/corr.py` | → matrices | matrices per fold; drift reported |
| T2.8 ∥ S | environment v1 (team totals, pace α) + opponent-by-position | `proj/environment.py`, `proj/defense.py` | → | ships only with CRPS gain > 1 SE |
| T2.9 ∥ S | Tonight cards + `prop_compare` (pts/reb/ast/PRA/3PM) | `web/players_view.py`, `templates/_players.html`, `proj/props.py` | → | ≥ 90% main props joined; both % shown |
| T2.10 ★ S | Path A copula v1 (T2.7 matrices, CV-by-level fit, winner/loser shift) | `proj/copula.py` | → | §7.3 coverage/PIT; blowout curve ±1.0 DK |

### Phase 2.5: contest builder, GPP, late swap, Showdown (Nov 15 – 21)

| ID | Task | Owned files | In → out | Acceptance |
|---|---|---|---|---|
| T2.11 ★ O | ownership proxy + field proxy (3 strengths) | `proj/dfs/field.py` | → field samples | Σ own = 800%; field lineups valid; ≤ 20 s |
| T2.12 ★ O | GPP portfolio: candidates, payout ladder, greedy, caps, dupes | `proj/dfs/portfolio.py` | → | N = 20 ≤ 120 s; caps exact; greedy portfolio beats "top N by individual E[payout]" on realised proxy payout over 100 backtest slates (paired) |
| T2.13 ★ S | cash P(score ≥ c_s) | `proj/dfs/cash.py` | → | P(cash) calibration on the backtest proxy line within ±5 pp |
| T2.14 ∥ S | late-swap re-solver + edit CSV + UI action | `proj/dfs/swap.py` | → | locked players untouched; ≤ 30 s |
| T2.15 ∥ S | Showdown (CPT binary) | `proj/dfs/showdown.py` | → | pydfs equal on 30 Showdown slates |
| T2.16 ∥ S | contest builder UI + guidance text | `templates/dfs_contest.html`, `web/dfs_view.py` | → | labels present ("simulated, not real", "proxy"); screenshots |
| T2.17 S Mac | backtest extension: field-proxy percentiles, `fppg` optimizer, cash proxy, lineup calibration | `proj/backtest.py` | → | §7.3 table filled |

### Phase 3: DFS MLOps and calibration (Nov 24 – Dec 5)

| ID | Task | Owned files | Acceptance |
|---|---|---|---|
| T3.1 S | registry specs (`player_minutes`, `player_rates`, `player_dist`, `dfs_engine`), gates, `dfs_metric_snapshots`, alerts, `/health` DFS tab | `mlops/dfs_monitor.py`, `web/health.py` (tab) | replay of the season to date produces snapshots; alerts dedupe |
| T3.2 S | standings-CSV importer + ownership-proxy calibration | `proj/dfs/standings.py` | importer exact on 3 sample CSVs; calibration report once ≥ 10 slates |
| T3.3 H | guidance polish + weekly DFS report | `proj/reports.py` | clustered SEs; proxy labels |

**Alerts:**
- minutes bias on projected starters > ±1.0 min (last 300);
- DK FP 80% coverage outside 0.80 ± 3 SE;
- FPPG reconciliation failure;
- a missed lock run;
- ID match < 98%;
- reconciliation gap > 0.5;
- live minutes MAE > live-like + 0.5.

### Phase 4: why engine and deferred items (Dec 8 – 23)

| ID | Task | Acceptance |
|---|---|---|
| T4.1 O | xFG (with S5's revised criteria) | per v1 §3.2, criteria rewritten at start |
| T4.2 O | full post-game decomposition | exact sums |
| T4.3 O | pre-game Shapley (S6 per-coalition reconciliation) | exact sums; ≤ 10 s per slate |
| T4.4 O | primary defender, matchup ingestion | ships only with a CRPS gain |
| T4.5 S | referee crew test | written verdict |
| T4.6 S | player page season views | screenshots |

**Critical path:** T0.0 → T0.5/T0.6 (and T0.2 → T1.0) → T1.1 → T1.2 → T1.4 → T1.5 → T1.6 (live by Oct 27) → T2.1 → T2.2 → T2.10 → T2.11 → T2.12. T2.5 → T2.6 is a parallel upgrade track that is off the GPP critical path, because of the copula.

**Machine plan:**
- On geekom1, run at most 3 agents: the critical-path agent, one ∥ coding agent, and one H/data agent.
- On the Mac, run the backtests, SMM, RotoGuru parse, stints and T2.7, one heavy job at a time.

## 9. Risks and open questions

**Risks.**
- **ToS.**
  - DK forbids automated collection "for any purpose". Our capture (~100 requests a day, no login) is still automated, and the owner accepted the risk. If DK blocks or objects, fall back to the lobby's manual salary CSV.
  - RotoGuru: polite, one-time, personal use, credited.
  - NBA, ESPN and Action Network data is personal-use only.
  - Entry is never automated.
- **Single machine.** Lock runs need the Mac awake. A missed run alerts. Fallback: `courtside dfs run --kind manual` on geekom1, which fetches its own draftables and stores locally.
- **Ownership proxy limits.** Without uploads there is no truth, so leverage and dupe numbers can be badly wrong. They are shown as bands, labelled, and ROI is shown as a range.
- **RotoGuru main-slate ambiguity.** The game set is inferred, positions are single, and 2019-20 includes the bubble. Report both slate definitions, and don't carry historical cash rates to live.
- **Backtest optimism** (known availability), and **early season** (prior-heavy rates, noisy preseason roles). Show "small sample" and widen the intervals.
- **Over-engineering.** Correlations are about 0.03. Path B ships for DFS only through G2.
- **DK rule changes.** Rules are read every run; scoring is checked weekly.

**Open questions for the owner (with proposals).**
1. **Top N defaults:** N = 10, min-different d = 1?
2. **Contest selection rule** for the payout archive (§3.2), with a cap of 40 a day. OK?
3. **Will you upload standings CSVs** from contests you enter? They are the only real ownership and cash-line truth (§7.4).
4. **Ownership display:** bands only, or percentages after calibration?
5. **Late-news alert channel:** page banner only (proposed), or a push notification?
6. **Tonight cards in Phase 2** (proposed) rather than Phase 1?
7. **FanDuel CSV import:** when?
8. ~~N3~~ Resolved: N3 = a post-game "teammate left the game" term (from `possible_injuries`), deferred to the Phase 4 why engine.
9. **Stint-path features in production DFS:** allow them once the 2021-26 projection backtest shows no CRPS loss, even though the DFS backtest only validates the box path? Proposed: yes.
