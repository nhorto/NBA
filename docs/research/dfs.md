# Research: DFS optimizer (2026-10-19)

This file condenses three DFS research passes (historical data, pro methodology, rules plus our own correlation study) and the Fable plan review. The correlation script and its output are at `research/dfs_correlations.py` and `research/dfs_correlations_output.txt`. Data samples are in `research/samples/`.

## Owner decisions

- **The DFS optimizer is the top priority.** It should be pro-grade in accuracy. The owner may play occasionally but doesn't usually play DFS, so it needs pro-grade accuracy and honest backtests, simple use, and a "why this lineup" for each build. Contest and bankroll guidance stays light.
- **DraftKings only for v1.** FanDuel comes later, via a CSV the user downloads (FD's API needs a login).
- **Two build modes:**
  - **Top N lineups:** ranked, with an optional minimum number of different players between lineups.
  - **Contest builder:** a diversified N-entry portfolio for a chosen contest.
- **Guidance level:** lineups plus light guidance (simulated cash and top-finish odds, a "sensible stakes" note). No dollar or Kelly sizing.
- **The post-game "why" v0 ships in Phase 1.** The full why engine is deferred to December.
- **Data policy (DK terms forbid automated collection "for any purpose"):**
  - RotoGuru for history.
  - A light daily DK capture: salaries, plus payout tables for a few dozen main contests.
  - No bulk DK crawl, no login.
  - Never automate entry or lineup edits; lineups are uploaded by hand as CSV.

## Rules (DK)

| Item | Value | Source |
|---|---|---|
| Classic scoring | Pts 1, 3PM +0.5, Reb 1.25, Ast 1.5, Stl 2, Blk 2, TO −0.5, DD +1.5, TD +3 | Rotowire DFS 101; Rotogrinders. DK's page is JS-rendered and Akamai-blocked. |
| Triple-double | **Earns both bonuses: +4.5 total** (medium-high confidence; verify against a contest results CSV) | Rotowire blog |
| Classic roster | 8 slots: PG, SG, SF, PF, C, G (PG/SG), F (SF/PF), UTIL. $50,000 cap. **≥ 2 games and ≥ 2 teams.** No per-team cap. Late swap allowed. | archived `gametype_70` payload |
| Showdown ("Captain Mode", id 81) | CPT + 5 UTIL, $50,000. CPT scores 1.5× points at 1.5× salary. One game, ≥ 2 teams. No late swap. | `gametype_81` payload + salaries |
| Tiers (id 73) | No-salary format, one player per tier. Out of scope. | lobby |
| FanDuel Classic | 2 PG/2 SG/2 SF/2 PF/1 C, $60,000, max 4 per team; Pts 1, Reb 1.2, Ast 1.5, Stl 3, Blk 3, TO −1 (sources conflict on Stl/Blk; newer sources say 3) | secondary |
| Positions | Multi-eligibility from the salary file rows. Read roster rules from `rules_json` at build time; never hard-code them. | |
| Status | Draftables `status` is None, Q or OUT. `isDisabled`/`isSwappable` per row. Each player locks at his game's `competition.startTime`. `playerGameAttributes` id 100 = "In Starting Lineup". `draftStatAttributes` id 219 = DK's FPPG (a free baseline to beat). `playerId` is stable across slates; `draftableId` is per slate. | |

## Data sources

| Need | Source | Depth | Notes |
|---|---|---|---|
| **Historical salaries + DK points** | **RotoGuru** `rotoguru1.com/cgi-bin/hyday.pl?game=dk` | 2014-10-29 → 2021-05 | One salary per player per day (main slate, likely). Use the HTML view (the `&scsv=1` export is truncated after 2017). Columns: Pos, Name, DK Pts, Salary, Team, Opp, Score, Min, Stats; `^` = starter. robots.txt is 404. ~1,200 requests at 1/s. Spot check: salaries matched DK exactly for 62 of 62 players. |
| Live salaries | DK draftables (already captured daily) | from 2026-10-19 | |
| Payout curves | DK `api.draftkings.com/contests/v1/contests/{id}?format=json` → `contestDetail.payoutSummary[]` (`minPosition`, `maxPosition`, `payoutDescriptions[0].value`), `entryFee`, `maximumEntries`, `maximumEntriesPerUser`, `isGuaranteed` | works for past contests too | Archive it before lock, because `entries` reads 0 after completion. Lobby `getcontests` fields: `id`, `n`, `a` (fee), `m` (field), `mec` (per-user max), `po` (prizes), `dg`, `gameTypeId`, `attr` (`IsDoubleUp`/`IsFiftyfifty`/`IsGuaranteed`…). |
| Winning scores, cash lines, ownership | Login only (DK leaderboard "Invalid userKey"; the standings CSV needs an entry) or paid tools (RotoGrinders ResultsDB ~$100–130/mo; SaberSim $97–297; Stokastic ~$150–230) | — | **No free source.** Contest-level evaluation uses payout shapes plus simulated fields. Real results only come from the owner's own manual standings exports. |

## Pro methodology (what pro tools do)

- **Methodology stack, in priority order:**
  1. Minutes and injury scenarios, with late-news re-simulation. **Late swap is "the single highest-value action" in NBA DFS** (Stokastic).
  2. Ownership model: a GBM point forecast plus calibrated uncertainty (RotoGrinders "Blitz" uses XGBoost on hundreds of features). Haugh & Singal model it as Dirichlet regression on salary, projection and expert ownership. A point estimate captured most of the gain.
  3. Field simulator from sampled ownership, with archetypes by buy-in and size, against the real payout curve.
  4. Portfolio: ILP candidates (randomized or penalized objectives) scored across shared sim scenarios, then greedy marginal-expected-payout selection with overlap and exposure caps (Hunter–Vielma–Zaman; Haugh–Singal). Add duplication risk.
  5. Cash: maximize P(score ≥ cash line), not the mean.
  6. Late-swap re-solver with locked players fixed.
  7. Backtest harness.
- **Evaluation:**
  - Simulated ROI ≠ real ROI.
  - Tournament ROI needs thousands of entries to stabilize (19%-ROI player: loses money 48% of the time over 100 GPPs, 17% over 1,000).
  - Use faster diagnostics: cash rate, top-1% rate, mean percentile finish, calibration of simulated rank vs realized.
  - Baselines: a plain projection-maximizing ILP, DK FPPG, salary-implied.
- **Feature checklist:**
  - **MVP:** Classic optimizer with sim ranking, exposure caps, lock/ban, DK CSV export, cash build.
  - **v2:** contest sim (cash %, top-1 %, ROI), contest-specific builds, late swap, dupe-risk indicator, Showdown, flashback/backtest.
  - **Later:** entry manager, contest advisor, post-contest grading.
- **Ownership accuracy:** no public numbers exist. Without paid data our ownership model can't be trained on real ownership. Use a proxy, and label it as one.

## Our own data: DFS correlations (2015-16 → 2025-26 regular season, 244k player-games)

- **Teammate correlation is weak.**
  - Overall residual r = **0.034**; star–star 0.048; high-usage pairs **0.000**; C–C −0.088; star–bench −0.034; bench–bench +0.089.
  - **Stacking is a weak lever.** Don't double-stack high-usage teammates.
  - Opponents: r = 0.011 overall; opposing stars +0.053.
  - Correlations drift by season (0.02 → 0.06). Estimate them from the last 1–3 seasons, and allow team-specific variation.
- **Blowouts:** the effect is almost entirely lost minutes.

  | Final margin | Star DK points | Starters DK points | Bench DK points |
  |---|---|---|---|
  | 0–5 | +2.3 | | |
  | 16–20 | −2.1 | | |
  | 26+ | **−5.5** | −2.9 | +1.1 |

  At |margin| ≥ 16, starters on the losing team average −4.7.
- **Game environment:** the actual game total explains 17.5% of team DK variance, but the **Vegas total explains only 4.9%**. Each point of actual total adds 0.53 team DK; each point of |margin| removes 0.45.
- **Star absence:** each remaining top-6 player gets **+3.1 DK** (+1.7 minutes, +5% DK per minute); the other star gets +3.3 (mostly per-minute). Treat it as a deterministic shift when the absence is known.
- **Distributions:**

  | Tier | Mean | SD | Skew | p10 / p50 / p90 / p99 |
  |---|---|---|---|---|
  | Stars | 41.0 | 13.8 | 0.33 | 24 / 40 / 59 / 77 |
  | Role | 24.7 | 11.3 | 0.44 | 11 / 24 / 40 / 54 |
  | Bench | 14.6 | 9.7 | 0.80 | 3 / 13 / 28 / 42 |

  - Normal fits stars and role players; **gamma fits bench**; lognormal is worst everywhere.
  - CV falls from ~0.8 (mean ≤ 10) to ~0.24 (mean ≥ 50).
  - Stars: 30% double-double rate, 2.8% triple-double rate.

## Fable review (plan v1): key points for the DFS-first revision

Verdict: **approve with changes.**

- **M1:** the bottom-up iid simulation overstates team and margin variance (sd(margin) ≈ 16 vs the real ~12.4), which makes the optimizer over-stack. Fixes:
  - Measure the target variances first.
  - Make Q4 pace and efficiency depend on the margin.
  - Add a shared game latent factor.
  - Model team possessions with a multinomial.

  Our correlation study confirms that weak teammate correlation is the truth to match.
- **M2:** fit every simulation parameter only on seasons before the test season.
- **M3:** DraftKings only (decided).
- **M4:** verify DK scoring by reconciling against DK's FPPG. Read slots, cap, ≥ 2 games and ≥ 2 teams from `rules_json`. Exclude disabled players.
- **M5:** add eval-lite to Phase 1. Historical salaries are now available (RotoGuru 2014–21).
- **M6:**
  - Live minutes MAE will be worse than the backtest's, not better.
  - Move P(play) recalibration into Phase 1, using the 151 rescued injury-report days.
- **M7:** reconcile simulated OT.
- **M8:** **lock-relative DFS runs** (lock −45 and −10 min), a `late_news` flag, and **late swap**.
- **Should-fix:**
  - S11: payout-ladder GPP objective; 25–50k sims for portfolio selection.
  - S12: a chalk-tilted field proxy.
  - S13: baselines include DK FPPG.
  - S14: **the minutes model must promote bench players into big roles when starters sit** (the value-play engine).
  - S2, S3, S4: scenario cap, randomized PIT, two-way clustering.
- **Resequence:** minutes and P(play) ≫ usage under absences > correlation (GPP only) > environment > defenders and referees. Defer xFG, Shapley, the full post-game decomposition, the defender effect, refs and matchup ingestion.
