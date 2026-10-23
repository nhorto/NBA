# IMPROVEMENTS: methodology review and improvement spec (2026-10-23)

*Status: spec, nothing implemented yet. Source: a read-only review on 2026-10-23 of the player
projections, the DFS layer, the game model / props and the evaluation process (three review
agents plus spot checks by the orchestrator), then a second, independent review by Fable that verified every item and added B8–B11, P10,
E7, E8 (see §9).
Live model at the time: `pathA-v0.3+p2`, field `own-proxy-1.1` / `field-proxy-1.1`, X factor
`xf-v1`. Read `docs/HANDOFF.md` first for state; this doc is the work list that follows it.*

Rules that still apply to every item here (HANDOFF §3, §4):
- Model changes ship **behind an env switch, default OFF**, with a bit-identical-OFF proof; the
  owner approves switch-ons.
- ≤ 3 agents on geekom1, one heavy job; heavy evals go to the Mac scratch dir
  (`~/Developer/courtside-compute/`), never the prod DB; check the clock against the lock
  blackout before each stage.
- Agents don't commit; stage explicit paths only (never `git commit -a`).
- No DK crawling/login/automated entry; no Kelly or dollar sizing; props shown as information.

Item IDs: **B** = bug, **P** = projection, **D** = DFS, **G** = game model / props,
**E** = evaluation / MLOps. Each item lists impact (H/M/L) and effort (S/M/L).

---

## 0. Verdict

The engine is careful. It fits walk-forward only, has scramble/lookahead probes, shares random
numbers between arms, and gates changes by tier. No as-of leak was found in the core
projection joins. The upside is in four places:

1. **Market and public information is unused in the projection.** That covers DK FPPG, prop lines
   and salary. Our own code measures that DK FPPG predicts part of our error (D1/P1).
2. **The DFS objective doesn't target "low-owned players with a big night" directly.** The X
   factor is a side display, ownership is treated as known, and the GPP tails are too wide
   (D3–D7).
3. **The contest builder's numbers are optimistic.** Lineups are generated, selected and
   reported on the same sims (B1).
4. **No untouched evaluation data remains.** About 20 decisions were tuned on the same seasons
   (E1).

---

## 1. Bugs (fix first)

### B1. Contest builder generates, selects and reports on the same sims (H / S)
- **Status 2026-10-23: built behind `COURTSIDE_PORTFOLIO_SPLIT` (OFF)** (`docs/research/portfolio-split.md`).
  - **Design:** G/S/R = 10/45/45 % of the truth rows. Version `gpp-portfolio-1.0+split`.
    OFF is bit-identical to a golden file.
  - **Payout overstatement:** OFF overstates E[payout] by 6–12 % on top-heavy GPPs, and ON
    removes it. P(cash) and P(top 1 %) were never inflated (both match the holdout within 0.01).
  - **Cost:** ON's own picks are 1–3 % worse at N=150.
  - **Corrections:** B1 does not explain "our chances read high", and the scenario share does
    not fall.
  - **`gs` mode** (generate + select on G+S = 55 %, report on R = 45 %; `+split-gs`):
    - reported bias −2 % to +1 % (mean −0.6 %);
    - pick cost −1.0 ± 0.5 % at N=150 (mode `1`: −1.8 %);
    - 10–35 % faster.
  - **Recommended default:** `gs` (owner to approve, since it changes what the builder
    reports). More stored sim rows would recover the remaining ~2 % on top-heavy Zones.
- **Where:**
  - `proj/dfs/portfolio.py:262-265` (scenario jobs): a lineup is solved on
    `scen_w·base + (1−scen_w)·fp[r]`.
  - `:274-278` (min-locked jobs) and `:741-743` (greedy refill) do the same.
  - `fp` comes from `truth_sims` at `:655`.
  - The same `fp` scores every candidate (`score(cands)`, ~`:700`).
  - The greedy pick and the reported E[payout], P(top 1 %) and entries curve use the same `fp`
    again (`:763-790`).
- **Effect:**
  - A scenario lineup is near-optimal in "its" sim row. That adds about top-prize / S to its
    expected payout: ~$0.40 on a $10k first prize, tens of dollars on a $1M one.
  - This favours scenario lineups over Top-N / own-penalty lineups and inflates the ROI and
    relative score.
  - Selecting the max of ~350 candidates on the same sims it then reports is a winner's curse.
  - Both effects are consistent with "our chances read high" (`real-standings-2026-10-20.md`).
- **Change:** split the sim rows into three disjoint sets:
  - G: scenario draws for candidate generation;
  - S: selection (greedy);
  - R: reporting.

  For example, use a separate sim draw for G, and a 50/50 split of the rest for S/R. Or draw
  scenarios from an independent copula run with another seed. Put it behind a switch (e.g.
  `COURTSIDE_PORTFOLIO_SPLIT=1`).
- **Accept:**
  - With the switch OFF, output is bit-identical.
  - With it ON, on a few stored slates: reported E[payout] / P(cash) / P(top 1 %) for the
    chosen portfolio, ON vs OFF. Expect ON lower; record by how much.
  - The share of scenario lineups among those selected also drops.
  - Re-score the 10-20 opener contests and compare the percentile of realised results.

### B2. Reconciliation gives the reserve's points to projected players (M / S)
- **Status 2026-10-23: built behind `COURTSIDE_RECON_RESERVE` (`+rr`, OFF); do not ship**
  (`docs/research/recon-reserve.md`). OFF is bit-identical.
  - **Results with it ON:**
    - CRPS is neutral in every set (|Δ| ≤ 1.8 SE).
    - MAE improves, but only because every mean drops.
    - The tier check fails in 4 of 12 set/arm runs (bench, and rotation under +mf).
    - Live team-sum bias gets worse: −0.08 → −1.09, and on high-reserve teams −1.19 → −5.67.
  - **Correction:** the over-projection is real in the harness (+0.2–0.8 per team, +2–8 at
    openers) but not live. Live, the DK pool puts almost every player in the projected set, so
    only 0.27 min per team-game go to unprojected players. The minutes engine still holds back
    2.4 min (10.8 on high-reserve teams), because the reserve was fit on last-10 candidates
    without the run-time extras.
  - **B2b status 2026-10-23: built behind `COURTSIDE_RESERVE_EXTRAS` (`+rx`, OFF); recommend ship
    (owner OK needed)** (`docs/research/reserve-extras.md`).
    - **What it does:** when a team's candidates hold the DK pool, the reserve becomes the
      debutants' expected minutes (fit walk-forward). On the live path the held-back minutes go
      2.43 → 0.14 per team-game (actual 0.27).
    - **Live:** DK CRPS z −2.4 (live) / −3.2 (live10). DK MAE +2.4 z (means move up). Worst
      tier: starters +1.0.
    - **Harness and RotoGuru:** identical by construction.
    - **With `+rr`:** the live tier check now passes, but the harness tier check still fails.
      Keep `+rr` off.
  - **Real fix (B2b):** in `minutes.py`, refit the reserve with the run-time extras, or subtract
    their expected minutes. Then re-test `+rr`. The "~13 min at openers" only holds under `+pre`.
- **Where:**
  - The minutes total removes a fitted reserve for players outside the candidate set
    (`proj/minutes.py:~2176`, `proj/copula.py:621` `_reserve_of`).
  - Rookies and two-way players with no box history are not projected
    (`proj/pipeline.py:~1218-1236`).
  - But `reconcile_team` scales the candidates' points to the **full** implied team total
    (`copula.py:26` docstring, `:692`, `:1178`; `proj/reconcile.py:44-66`).
- **Effect:**
  - Normally ~3 reserve minutes, so ~1 % too many points for every projected player.
  - At openers or with heavy two-way usage it is ~13 minutes. This is part of the
    7–15-point pre-reconciliation gap and of the over-projection on debut-heavy teams.
- **Change:** target = implied − E[reserve minutes] × (bench points per minute; league or team
  bench rate). Apply the same in both reconcile call sites. Switch-gated.
- **Accept:**
  - Bit-identical OFF.
  - Harness/live CRPS and DK MAE at least neutral.
  - Team-sum bias (projected vs actual team DK from projected players) moves toward 0, most at
    openers and on high-reserve teams.

### B3. Past-day replays can see post-tip injury rows (M for honesty / S)
- **Status 2026-10-23: built, uncommitted; no measurable effect** (`game-calibration.md` §6.1).
  - **Cutoff:** past-day replays now cap each game at its own tip, using the ESPN scoreboard
    archive since past seasons have no tips in `games`. Live days are byte-identical. 10 tests.
  - **Replay (2025-26):** LL 0.5750, slope 0.965 under both the old and the new cutoff
    (ΔLL +0.00000, SE 0.00003). 679 report games read a post-tip report under the old cutoff,
    but only 11 games' p moved. The headline numbers stand; MODEL.md is unchanged.
  - **Back-to-back merge (game side):** `COURTSIDE_INJ_GAMEDAY_GAME` (OFF) changes 1 game.
    Turn it on together with `+gd` after the prod `game_date` backfill.
- **Where:** `predict.py:111-117` `injury_cutoff`. For past days the cutoff is 05:00 UTC the
  next day (~1 AM ET, after tip). Live days are unaffected.
- **Effect:** game-model replays and backfills (e.g. the 2025-26 replay in
  `game-calibration.md`) can use news after the tip, which flatters them.
- **Change:** cap at the game's tip time (per game), or at a fixed pre-tip virtual time like
  `_virtual_now` (12:00Z). Re-run the 2025-26 replay and update `game-calibration.md` /
  `MODEL.md` if numbers move.
- **Accept:** a test that a past-day replay never reads an injury row with
  `created_at > tip`.

### B4. Game model uses hard-coded P(play) instead of the fitted model (M / S)
- **Where:** `injuries.py:272-276` `PLAY_PROB` (Doubtful 0.10, Questionable 0.5, DTD 0.6).
  `model/players.py:~619` uses Doubtful 0.15, so the two paths are inconsistent.
- **Evidence:** `pplay-q-refit.md` §0. Questionable non-starters play ~36 %; Q starter-role
  players missing from DK starters ~14 %.
- **Change:**
  - Use the player side's calibrated P(play) (report cell + DK starter flags at lock−10) for the
    game model's absence value.
  - At minimum, make the two constant tables agree.
  - Switch-gated for the game model.
- **Accept:** game log loss vs close at least neutral on the replay. Note that `PLAY_PROB` barely
  mattered in the rung-2 ablation (0.0014 LL), so the expected gain is small. The main value is
  consistency.

### B5. Ownership row cap too low for double-ups (L / S)
- **Status 2026-10-23: built (no switch; versions `own-proxy-1.2` / `field-proxy-1.2`).** Cash cap
  0.97, GPP bit-identical.
  - On the 10-20 DUs the proxy's top player still saturates (now at 0.97), and the modelled
    cash line barely moves (248.5 → 248.4 vs real 237–248).
  - The real miss is *who* is concentrated (proxy top Anunoby, real top Harper 0.93). That is
    E8/D7 work, not the cap. Low value; fine to ship with the next field version.
- **Where:** `proj/dfs/field.py:106` `ROW_CAP = 0.92`. Real DU ownership reached 93 % (10-20).
- **Change:** cap by contest type (e.g. 0.97 for cash/DU, 0.92 for GPP).
- **Accept:** bit-identical for GPP. Re-run `courtside dfs field-calib` on the 10-20 uploads:
  the DU top-player ownership no longer saturates at the cap.

### B6. Showdown captain ownership term missing (M for Showdown / S–M)
- **Where:** `PLAYERS.md` §5.7 specifies a CPT term, but `utility()` (`field.py:438-462`) has
  none, so CPT uses the UTIL utility.
- *Correction (§9):* the first review said the salary shadow price pushes CPT picks to cheaper
  players. That is unverified and probably backwards. The value-heavy utility underspends, so
  λ < 0 and `exp(−λ(s_k − mean))` favours expensive players, more so at CPT (1.5× salary).
  Measure it before changing anything.
- **Change:** a separate CPT utility, star-weighted (public projection rank, not per-$1k value).
  Gate the weights on uploaded Showdown standings, starting with a sensible prior.
- **Accept:** CPT ownership on an uploaded Showdown contest: MAE and rank correlation vs the
  current proxy.
- **Priority:** low unless the owner plays Showdown (they mostly play Classic).

### B7. Minor inconsistencies (L / S)
- The minutes eval integrates margins at a hard-coded sd 12.4 (`minutes.py:~136`,
  `minutes_eval.py:~273`). The sims use the per-season fit, 13.5–14 (`copula.py:~142`). Use the
  fitted value.
- The props φ is fit on 2023-26 (`proj/props.py:93`, `fit_phi` ~`:688`), which overlaps the
  evaluation seasons. Fit on seasons < S.
- The fringe P(play) "mid" model is fit on the T2.4 slice's own rows
  (`availability.py:~790-791`, `fringe_params.json` meta). That feeds the headline p2-vs-v1
  number (−6.9 SE). Refit out-of-sample (leave-month-out or seasons < S) and restate the number.

### B8. Official injury rows are keyed to the capture day, not the game day (M–H / S–M)
- **Status 2026-10-23: built, uncommitted** (`docs/research/injury-gameday.md`).
  - **Data fix (always on):** migration 16 `injury_snapshots.game_date`, written by collect and
    backfill. `courtside backfill injuries --game-date` matched 177,721/177,721 dev rows and
    180,429/180,429 on a prod snapshot. Run it on prod after deploy.
  - **Switch:** `COURTSIDE_INJ_GAMEDAY` (`+gd`, default OFF) with its own
    `availability_params_gd.json`. OFF is identical on 62/62 frames.
  - **P(play) log loss ON−OFF:** −0.0005 to −0.0012 overall. Never worse, never past ~2 SE.
    Only the `removed` cell moves much (n 66 → 125).
  - **Corrections:**
    - Effect 1 is overstated: at 10:00 the OFF fallback already serves the latest day-before
      batch. What's lost is batch order and back-to-back separation.
    - Effect 3 is wrong for serving: `hours_to_tip` is always `captured_at` → tip.
    - Revised priority: the data fix now; the switch is a correctness cleanup to fold into the
      next version bump.
  - **Spin-off:** the same back-to-back merge bug is in the game model (`injuries._index`,
    `model/players.py:641-651`), now handled with B3.
- **Where:**
  - `proj/availability.py:457-480`: `load_reports` groups batches by
    `config.nba_day(captured_at)`.
  - `:697`: `_slate` loads `[_day_start_utc(day), at]`.
  - `collect.py:355-359` and `backfill_injuries.py:241-247` drop the parser's `game_date`
    (`sources/injury_report.py:204` yields it). `injury_snapshots` has no game-date column
    (confirmed in the schema).
- **Evidence (dev DB, Fable):**
  - The 2026-01-14 22:00Z–03:00Z batches list BOS/CHI/DEN/DET/MIA/MIL/NOP/PHX, none of which
    played on 01-14. Those are 01-15 rows filed under 01-14.
  - On the NYK back-to-back, the 01-15T03:00Z batch shows a player's 01-15 status under the
    01-14 key.
- **Effects:**
  1. Day-before reports (5 PM / 10 PM ET) are invisible to the game day's official path. Runs
     before ~1 PM ET and the Brief fall back to `_fallback_batches`, a different code path
     with one unordered batch.
  2. The `gt6h` and "removed" P(play) training cells (`build_samples`, `:574-602`) never see
     day-before batches, so early-run cells are fit on the wrong population.
  3. `hours_to_tip` is wrong for those rows.
  - Lock−45 and lock−10 for evening games look unaffected (no back-to-back status merge was
    found).
- **Change:**
  - Add `game_date` to `injury_snapshots` (migration), write it in collect/backfill, and
    backfill it from the archived PDFs.
  - Key batches by game date.
  - Refit `availability_params.json`. Switch-gated, since it changes P(play).
- **Accept:**
  - A test that a 5 PM ET day-before row is the latest batch for the next day's game at
    10:00 ET.
  - Refit cell deltas reported.
  - P(play) log loss at the 10:00/13:30 runs, before and after.
- **Priority:** row 1. It changes the training set behind every later availability number.

### B9. `environment()` has no staleness bound on book lines (M / S)
- **Status 2026-10-23: built behind `COURTSIDE_LINE_FRESH` (`+lf`, OFF).** `env` now records
  `line_ages_min`; 10 tests.
  - First 3 nights (49 game-runs): books were dropped in 11 runs. In those runs the gap to the
    close fell from 0.364 to 0.307 points (0.255 → 0.242 over all runs).
  - Descriptive only. Re-evaluate after a few weeks of 2026-27 snapshots.
- **Where:** `proj/pipeline.py:864-882` takes the newest row per book with `captured_at <= at`
  and takes the median across books, with no maximum age. A book that stopped posting (common
  after a star is ruled out) keeps pulling the median.
- **Change:** drop books older than ~3 h, or more than 1 h older than the newest book. Record
  the line ages in `env`.
- **Accept:** team-point target vs the later closing line, before/after (needs 2026-27 line
  snapshots from the Mac).

### B10. `truth_sims` shrinks toward a stale season average, hurting X-factor targets (M / S)
- **Status 2026-10-23: measured** (`docs/research/truth-beta-roles.md`). On the current arm β is
  ~0.08–0.11, not 0.18.
  - **Harm from the flat 0.18:** promoted rows are hurt slightly (CRPS +0.01, and +0.03–0.05 for
    ratios > 1.25). Demoted, bench and returning rows are hurt more.
  - **Recommended P1:** μ* = μ + β(ratio)·(last-10 − μ), with β by planned/last-10 minutes ratio
    bucket (0, .16, .12, .09, .07; 0 for returning / no history). Fit by CRPS walk-forward and
    refit weekly. Expected gain: −0.002 to −0.006 CRPS vs no shift (2–6 SE), and about −0.013
    vs today. Promoted rows are no worse.
  - Salary as a feature for stars/role players only; re-test after ~10 graded nights.
- **Where:** `proj/dfs/portfolio.py:466-489`. Every playing player's truth is shifted by
  β·(FPPG − μ), β = 0.18.
- **Problem:** for promoted or returning players FPPG is far below μ, so the scoring truth
  undervalues them. Greedy selection then discounts exactly the low-owned, rising-role players
  the owner wants.
- **Change:** this is a precondition for P1/D1. The blend must be role-aware: condition β on
  |planned − recent minutes|, or use last-10 / salary-implied as the public anchor. Measure β
  by role-change bucket before choosing.
- **Accept:** β by bucket (planned/recent minutes ratio) with clustered SE. The role-aware
  blend's CRPS on promoted rows is not worse than OFF.

### B11. Copula tier assignment differs between fit and use (L / S)
- **Where:** `copula.py:300-303` fits ρ/CV tiers from `r_dk × mpg`, while `:707-708` applies
  them from `mu_played`. A player can sit in a different tier at fit and at draw.
- **Change:** use identical `role_tiers` inputs.
- **Accept:** bit-identical OFF; the tier-mismatch count is 0 ON.

---

## 2. Projections

### P1. Learn a blend of our projection with DK FPPG and simple baselines (H / S)
- **Evidence:**
  - `proj/dfs/portfolio.py:92-99`: (actual − ours) regresses on (FPPG − ours) with
    β = 0.16–0.20 at 8–20 SE historically, and 0.45 ± 0.18 on two live nights.
  - The residual sd (9.6) exceeds the sims' sd (9.45) by ~0.18 sd.
  - β is used only in `truth_sims`. The projection, cash, Top N and candidate means use raw
    `mu` (`proj/dfs/runner.py:624-627`, `portfolio.py:636`).
- **Change:**
  - Fit a small walk-forward blend by tier, on seasons < S: ours, DK FPPG, last-10, season
    average, and possibly salary-implied.
  - Test whether shrinkage should grow with |FPPG − ours|. Large gaps are often our
    minutes errors.
  - Re-measure β on v0.3+p2 first, since it may be smaller now.
  - Use the blended mean everywhere: display, cash, Top N, candidates and sims. Then drop the
    separate `TRUTH_BETA` step or set it to the residual β.
  - Refit β weekly from nightly grading (~100–150 player-games a night; 10 nights ≈ SE 0.08).
- **Accept:**
  - DK MAE / CRPS on harness, RotoGuru and live slice, ≥ 2 SE better, no tier worse.
  - Lineup backtest at least neutral.
- **Precondition:** B10. The blend must be role-aware, or it will drag promoted and
  returning players toward a stale average.
- **Note:** FPPG is a season average, not a projection. Most of its value is likely as a
  recency regulariser, so compare against our own season average as a baseline too.

### P2. Market prop lines (and DK salary) as projection inputs (H / M; needs owner decision)
- **Blocked by** owner rule L6 (`PLAYERS.md` §7.1: props and salary are never features).
  **Proposal for the owner:** allow props and salary as inputs for **DFS projections only**.
  Keep the "our number vs the market" prop display computed from a market-free model so it
  isn't circular.
- **State:** F9 is built, OFF (f3c86c1). There is no free prop history: lines start
  2026-10-19, from one book (DK via ESPN).
- **Change:**
  - Not a GBM feature group yet, since there's no training history.
  - Use a few-parameter Bayesian blend of our mean with the de-vigged prop median (and minutes
    implied by props), with weights by stat and tier.
  - Update the weights online through 2026-27.
- **Accept:** a pre-registered live test (see E2) after ~3 weeks of props: DK CRPS and MAE on
  players with props, game×player-clustered SE.

### P3. Early-season and role-change priors (M–H now / M)
- **Evidence:**
  - The rate prior is league means by position × own minutes tier (`proj/state.py:356-387`).
  - The planned S1 prior (height, weight, age, draft slot) and an age curve aren't
    implemented.
  - October rates are mostly last season's, with no development or decline. *Correction
    (§9):* the 0.6 is the fixed mpg-state ρ (`state.py:74`); per-stat ρ is fit per stat
    (`RHO_GRID`, `:588`). It is not a flat ×0.6 on rates.
  - HANDOFF: promoted and new players get low per-minute rates, and the market scaling smears
    the gap across the whole team.
- **Change:**
  - Add an age/experience delta on last season's rate, fit on seasons < S.
  - Add a rookie prior from draft slot and college/G-League where available.
  - Use preseason per-minute rates as weak evidence (the preseason tables exist since
    migration 15).
  - Shrink faster toward current-season evidence for players with a role change (planned
    minutes ≫ last season's).
- **Accept:** first-30-days DK CRPS ≥ 2 SE better on ≥ 2 seasons, no tier worse full-season.

### P4. Non-uniform reconciliation (M / S)
- **Where:** `proj/reconcile.py:44-66` scales every player by one factor c, clipped to
  [0.75, 1.33].
- **Change:**
  - Distribute the gap in proportion to each player's rate uncertainty (posterior variance or
    games of evidence), so new and promoted players absorb most of it. This mirrors the
    variance-weighted 240-minute reconciliation, which beat proportional allocation.
  - Treat |gap| > 6 (`GAP_CHECK`) as a projection error to log and investigate, not just a
    flag.
- **Depends on:** B2 (fix the target first).

### P5. Ship the switches that clear the bar (M / S; owner decision)
- **Status 2026-10-23: `+mf` shipped** (owner OK; commit 5598456; live `pathA-v0.3+mf+p2`; pre-fit
  bundle copied to prod; rollback `COURTSIDE_MINFIX=0`). Next: `+pq`.
- **`+mf` (MINFIX):**
  - CRPS −6.1/−5.2 SE harness, −2.6/−3.7 live, no tier worse. **Recommend ship first.**
  - Pre-fit bundle in `~/Developer/courtside-compute/ship/data/proj/`.
  - Copy `bundle_2026-27_v1_cx_mf.json` + `minutes_params_2026-27_v1_mf.json` into prod
    `var/proj/` before flipping `MINFIX_DEFAULT`.
- **`+pq`:** lock−10 Q refit, −3.2 SE log loss, DFS CRPS −2.7 SE. Ship second.
- **`+c1r`:** judge on top of `+mf` on the confirmation set (E1), not the harness it was shaped
  on. Copy `corr_2026-27.json` with the bundle (HANDOFF warning).
- **Leave off:** `+mr` (adds little over `+mf`, costs the bench) and `+bo` (~0 over `+mf`).

### P6. Player-specific minutes spread (M / M)
- **Where:** minutes noise σ is per tier only (`minutes.py` step 6, `_fit_noise`). `+mf`'s
  gains came mostly from fixing noise, so the spread clearly matters.
- **Change:** a distributional or quantile model for minutes (NGBoost-style or LightGBM
  quantiles) on the GBM's own features. Add per-player volatility, foul rate,
  return-from-injury and new-team flags.
- **Accept:** minutes CRPS and PIT by tier, then DK CRPS end to end.

### P7. News latency and minutes restrictions (M–H / M)
- **Change:**
  - Bring forward the ESPN full-comment parser for minutes-restriction flags from January.
    Raw snapshots are already archived.
  - Add a "restricted" minutes cap/prior.
  - Measure latency: for each report or status change, minutes from the news to our next
    re-run and to lock. Add a dashboard row on `/health/dfs`.
- **Accept:** restricted-player minutes MAE vs unflagged; latency distribution reported.

### P8. Usage redistribution beyond the top-2 star bump (M / M)
- **Where:** live redistribution is a tier bump only when one of a team's top-2 stars is out
  (`pipeline.py:~919-930`, `state.py:~702`). A third starter out gives no per-minute shift.
  Usage v1 barely moved MAE (−0.006).
- **Change:** re-evaluate usage v1 on the value-ranking metric (E4), since DFS cares about who
  absorbs usage, not the team total. Or let props (P2) carry it.

### P9. Lower priority
- Split points into shot attempts × efficiency with separate shrinkage (F4 suggests the market
  total cancels most of it).
- Referee pace; per-minute blowout effects (stars in big wins run 2.0–2.6 DK low at the right
  minutes).
- A two-component talent + form rate state (Kalman / DARKO-like) instead of one exponential
  decay.

### P10. Keep scenario-conditional outputs instead of throwing them away (H for the owner's goal / S–M)
- **Where:**
  - `copula.py:592-635`: `_team_minutes` knows each sim's availability scenario
    (`sc["assign"]`, `kept`).
  - But `GameSims` (`:448-460`) and `store.save_sims` (`store.py:70-85`) keep only
    `fp`/`minutes`.
- **Effect:**
  - Nothing can show "X if Y sits: 28 min, P(big night) 55 %; if Y plays: 14 %".
  - Player cards can't condition on that.
  - The D3 leverage can't be scenario-aware.
  - Late swap (D8) can't condition on a now-known Out.
- **Change:**
  - Save a per-team `(S,)` uint8 scenario index plus the kept-absent masks in the sims npz.
  - Add conditional means and chances to the player card and to `xfactor.run_table`.
- **Accept:**
  - Conditional means equal `ScenarioMinutes.project(kept)` × rates within MC error.
  - The card shows the split for players whose teammates have 0.05 < P(avail) < 0.95.
- **Priority:** before D3. It is the cheap enabler for D4, D3 and D8, with no model risk.

---

## 3. DFS

### D1. Use the blended mean in every optimizer (H / S)
This is the DFS side of P1. Cash, Top N and builder candidates should all use the same
best-estimate mean the sims use. **Do not** apply the flat
`mu + TRUTH_BETA·(FPPG − mu)` stopgap: per B10 it would push the optimizer away from
promoted and low-owned players. Wait for the role-aware blend (P1 + B10).
Accept: RotoGuru lineup backtest (`proj/backtest.py`) at least neutral, plus live paired vs
OFF.

### D2. Pick the cash lineup by P(beat the cash line) (M / S)
- **Where:** cash = mean − 0.3·downside (`runner.py:624`, `CASH_LAMBDA`). `PLAYERS.md` §5.3
  already specifies P(cash).
- **Why:** real DUs put 30–35 % of entries on one chalk lineup, so the cash line moves with it.
  Only P(score ≥ line) over the shared sims captures that covariance.
- **Change:**
  - Score Top N plus a small λ grid with `field.evaluate` against the DU field.
  - Choose on held-out sims (B1's split).
  - Run the planned walk-forward test (`research/cash_calibration.py`).
- **Also:** `PLAYERS.md` §5.4 says Top N is "ranked by mean, then by P(cash) if available".
  Today the ranking is solve order only (`topn.py:118-131`), and P(cash) is only annotated
  afterwards. Fix that here.
- **Accept:** RotoGuru cash rate vs current rule, paired by slate.

### D3. Redefine the X factor as leverage (H for the owner's goal / M)
- **Current:** `xf-v1` = bench/role, P(play) ≥ 0.85, proxy band Low, calibrated
  P(DK ≥ 6×) ≥ 1/3, not in our top-10 value, max 5 (`proj/dfs/xfactor.py:120-242`). That
  mostly selects cheap value plays, which the public also finds. Value per $1k is the
  ownership proxy's biggest weight, and the proxy runs low on rising-role cheap players.
- **Change (`xf-v2`, behind a switch):**
  - For each sim, take the modelled field's top-1 % lineups (already sorted in
    `field.evaluate`).
  - Player leverage = share of top-1 % lineups containing him ÷ his ownership, shrunk toward 1
    and computed with ownership uncertainty (D5).
  - Tag = high leverage, P(play) ≥ 0.85, and a minimum absolute top-1 % share, so we don't tag
    0.1 %-owned scrubs.
  - Interim cheap fix: exclude by **public** value rank (the proxy's public projection per
    $1k) instead of ours.
  - Feed it into the builder as leverage-tilted candidate solves next to the own-penalised
    ones.
- **Order (§9):** D4 → P10 conditional chances → D3, paired with D5. Top-1 %-share leverage
  is only honest with ownership noise.
- **Accept:**
  - `xf-v1` stays bit-identical with the switch OFF.
  - Walk-forward on RotoGuru seasons with the proxy ownership: tagged players' share of
    actual top-1 % lineups vs their ownership.
  - Once standings arrive: realised leverage on real top-1 % lineups.

### D4. Late-news X-factor flag (M–H / S–M)
- **Why:** the opener's real X-factor nights (Drummond 43 DK at 0.3 %, Green 26.5 at 0.4 %)
  were minutes and P(play) misses, not tail misses (`xfactor.md` §0).
- **Change:**
  - Flag the projected beneficiaries of a starter or rotation player ruled out within ~30–60
    minutes of lock. That is when ownership lags the news.
  - Show it as its own tag ("late news") and raise these players' ownership uncertainty in
    D5.
  - Measure later: does ownership of such players under-react? Upload checks will tell.
- **Accept:** count of flags per slate, plus each flagged player's realised DK vs projection
  and, on uploaded slates, realised ownership vs proxy.

### D5. Model ownership uncertainty in the field (H / M)
- **Where:** the field is one fixed sample from point ownership (`field.py:~1133-1245`). The
  proxy had Harper at 13 % vs 66–93 % real.
- **Change:**
  - Draw K fields (4–8), each with log-odds ownership noise sized to observed errors: GPP MAE
    ~4.6 points, DU top-10 MAE ~27; refit as uploads accrue.
  - Average payouts across fields.
  - Leverage bets on "Low" players then get discounted honestly.
- **Accept:** on uploaded slates, the percentile of realised results within simulated
  distributions moves toward uniform. Runtime stays within the §5.8 budgets; N=150 is ~200 s
  today.

### D6. Fix GPP tail shape before payout scoring (H / M)
- **Evidence:** `xfactor.md` §2. Upper tails are too wide: bench players at 16–24 minutes
  reach their q99 0.35 % of the time instead of 1 %, and their q95 3.1 % instead of 5 %. Yet
  `truth_sims` adds symmetric noise (`TRUTH_TAU`, `portfolio.py:99`).
- **Change:**
  - Per tier × minutes-bucket quantile remap of each player's sims, fit on seasons < S. This
    generalises the X factor's per-tier logistic map.
  - Then re-check whether `TRUTH_TAU` is still needed.
  - Coordinate with `+c1r`, which also changes tails.
- **Accept:** upper-tail PIT shares by tier within ±0.3 pt of nominal on held-out seasons;
  builder P(top 1 %) on RotoGuru seasons vs realised.

### D7. Ownership proxy features (M / M; gated on ≥ 10 uploaded slates)
- **Where:** current inputs are value, projection, team total, replacement, Q and bench
  (`field.py:~94-104`, `:~438-462`).
- **Add:**
  - last 1–3 games' DK points and minutes (the public chases recent form);
  - salary change since the last slate;
  - news recency;
  - a contest-type term (DU vs GPP);
  - slate size;
  - a "consensus lineup" share. The single most-entered lineup was the same in all four
    10-20 contests (30–35 % of DU entries); model it as a few public-optimal lineups built on
    the public projection;
  - a CPT term (B6).
- **Why:** the misses were rising-role young players and stars back from long layoffs.
- **Precondition:** E8 (fit on log-odds).

### D8. Contest-aware late swap (M / M)
- **Where:** builder entries are re-solved by mean (`proj/dfs/swap.py:~116, ~166`), which
  strips the leverage and diversity they were built with.
- **Change:**
  - GPP: re-score swap options with the portfolio objective, conditioned on locked players'
    realised points.
  - Cash: maximise P(final ≥ line | locked points).
  - Condition on now-known Outs via P10's scenario ids.
- **Accept:** replay stored late-swap nights: portfolio E[payout] given the locked points,
  contest-aware vs mean re-solve, on held-out sims.

### D9. Tail-estimation noise (M / S)
- In a 30k GPP, P(1st) per lineup is ~1e-4, so 25k sims give a handful of events.
- Report SEs on the entries-curve points.
- Consider choosing on E[payout] with the top prizes flattened, or blended with P(top 1 %).
- **Accept:** the curve shows SEs. Selection stability: the overlap of chosen portfolios across
  two sim seeds goes up.

### D10. Showdown
Point Showdown users at the contest builder (joint sims carry single-game correlation) rather
than Top N by mean. Depends on B6.

### Not recommended
NBA stacking or bring-back rules: causal teammate correlation is −0.03, opponents ~0
(`docs/research/dfs.md`), and the joint sims already carry it.

---

## 4. Game model and props

### G1. Line-timing mismatch: train on close, run on a stale line (H / M)
- **Where:**
  - `gbm_market` learns absence effects (~0.2 pts) from closing lines that already price the
    news (`model/features.py:113-127`, `model/gbm.py:246-262`).
  - Live, its market input is the last captured line (`predict.py:221-257`), possibly hours
    old.
  - Backfills and replays use closes (`predict.py:~421`), flattering live.
- **Evidence (read-only check):** on 3,586 ESPN games, open→close moves 1.66 pts on average.
  Margin RMSE is 14.01 vs the open and 13.73 vs the close.
- **Change:**
  - Store the capture time of each line.
  - Add a "news since the line" term: the change in the opinion model's absence value between
    the injury state at line capture and now.
  - Validate on ESPN open→close (3,865 games from 2022-23) plus the official injury archive.
    Does the model's injury value explain line moves with slope ≈ 1?
- **New objective:** beat the line available at prediction time, and feed the player sims a
  good margin/total. Score by CRPS of margin and total vs that line, not only log loss vs the
  close.

### G2. Stop trying to beat the close (decision, no code)
- Beating the close isn't realistic with free data. `gbm_market`'s departures from the close
  are ~30 % borne out, `gbm_opinion`'s ~11 % (`game-calibration.md` §0, §3).
- Stint-based player ratings and rest/travel may help the opinion model by ~0.001 LL, and
  nothing against the close: low priority.
- Margin sd varying by game is mild (13.97 pick'em vs 12.9–13.3 otherwise, confounded by
  season). Revisit only if blowout-minute errors trace to it.

### G3. Props: pre-register the blend test now (M now, H later / S)
- **Form:** logit p = a·logit(market) + b·logit(ours), fit on earlier dates only.
- **Inputs:** the same lock−45 snapshot state.
- **SEs:** clustered by game and player.
- **Minimum:** ~60 slates.
- **De-vig:** proportional/multiplicative is fine for now. A read-only check on 16,311 game
  closes found power de-vig worse (+0.0007 ± 0.0003 LL). Once ~10k graded props exist, compare
  multiplicative / power / additive plus an over-side shade term.
- **Fix B7 (φ fit window) first.**

---

## 5. Evaluation and MLOps

### E1. Locked confirmation set and pre-registration (H / S)
- **Problem:**
  - Harness 2023-24/2024-25, RotoGuru 2019-21 and the 2025-26 live slice have each been
    reused for ~15–20 arms and redesigns.
  - 2022-23 was used in the F3/F6 feature tests (`PLAYERS.md` Phase 2F table).
  - 2021-22 is the least-used past season.
  - The "≥ 2 SE in both seasons" bar guards against noise on a single test, not against
    designs shaped by those seasons over many iterations.
- **Change:**
  - Declare **2026-27 live** the confirmatory set. Run candidate switches as **live shadows**
    (same seeds as the champion; store their projections, don't serve them).
  - Flip a switch only after a pre-registered primary metric passes on N slates.
  - Use 2021-22 once per release candidate as a secondary check, and record that it has been
    touched.
  - Keep a **switch ledger** (`docs/research/switch-ledger.md`): hypothesis, primary metric,
    bar, seasons used, number of attempts and redesigns, result.

### E2. Two-way clustered SEs everywhere (M / S)
- **Status 2026-10-23: `metrics.paired_cluster2` added** (7 tests). Ledger started at
  `docs/research/switch-ledger.md`.
  - Re-computing `+mf` two-way widens the SEs ×1.04–1.28: z −5.2 / −4.6 harness, live −2.0,
    live10 −2.9, RotoGuru −3.5 / −2.9. The sign is negative on every set, so the ship stands.
  - Not yet wired into the eval scripts' decision outputs.
- **Where:** `proj/metrics.py:28-42` `paired_cluster` is one-way by date. The harness uses
  two-way date × player (`proj/backtest.py:~1289`). A mis-modelled player stays wrong for
  weeks.
- **Change:** use two-way (date × player, or game × player) in all decision metrics, and report
  the per-season sign.
- **Accept:** re-compute one past decision (e.g. `+mf`) to see how much the SEs widen; record
  it in the ledger.

### E3. Test the stack, not single toggles (M / M)
Switches interact: `+bo` ≈ 0 over `+mf`, and `+mr` costs the bench without `+mf`. Use forward
selection over the candidate stack on the harness, then confirm once on E1.

### E4. DFS-relevant evaluation populations and a value-ranking metric (M / S)
- Add a salary ≥ $4k / projected ≥ 15 DK population. All-row CRPS is diluted by deep-bench
  zeros.
- Add a value-ranking metric: corr(ours − salary-implied, actual − salary-implied). This is
  what drives lineups.
- Make live10 the primary gate for minutes and availability changes. The oracle hides
  scenario redistribution.

### E5. Real-standings calibration process (M / S)
- **Unit = slate.** Every lineup on a night shares one draw.
- **DU P(cash):** use the percentile of (lineup score − real cash line) within its simulated
  distribution. That detects a 0.10 shift in ~33 slates and 0.15 in ~15. A plain cash rate
  (0.70 vs 0.55) needs ~45 slates.
- **Top-1 % hit rate:** needs 100+ slates. Track mean finishing percentile and its calibration
  instead.
- **Ownership:** per slate, MAE, rank correlation and top-10 bias, split by contest type and
  slate size (3 vs 11 games).
- **Nightly, no standings needed:** β drift (P1), tail calibration by tier (D6), ours vs an
  FPPG-optimizer baseline (paired), and our score as a share of the hindsight-optimal lineup.
- **Ask the owner:** upload one big-GPP standings file per slate where possible. That is worth
  more than several small DUs.
- **Live projection tracking:** a weekly paired test ours vs DK FPPG with game-clustered SE and
  pre-set alarm levels. The opener's 7.33 vs 7.25 at n = 2 means nothing.

### E6. MLOps gaps (M / S–M)
- **No gate on the player side:** `mlops/dfs_monitor.py:6-10` makes the newest version
  champion, and switches are flipped by hand. E1's shadows fix this.
- **Game online gate:** "n ≥ 100 and mean diff ≤ 0" (`mlops/retrain.py:~418`) is a coin flip.
  At n = 100 the SE is ~0.003, against effects of ~0.0005. Restate it as "not clearly worse"
  (reject if worse by > 2 SE).
- **Offline gate SE:** it uses plain per-game SE (`retrain.py:~97-101`) and pools playoffs.
  Use clustered SE and the regular season only (low impact).
- **Weekly retrain:** re-running the same grid can't pass its own gate, so it does nothing.
  Make it monthly or alert-triggered.
- **Calibration alert:** alert on model **minus market**, since the market also goes
  overconfident in Dec/Jan. Daily rolling 2-SE checks will false-alarm several times a season,
  so require persistence or use CUSUM.
- **Data fingerprint:** the version hash covers model, params, features and code SHA, not the
  data. Backfills (e.g. the injury archive) silently change a version. Add a data fingerprint
  (row counts plus max dates per source table, or a hash).
- **PSI drift on seasonal features** (e.g. `elo_diff` early season) is mostly noise. Suppress
  it or set season-aware baselines.

### E7. Shipped P(play) v2 has no default test coverage (L / S)
- **Status 2026-10-23: done.** `tests/test_pplay_v2_paths.py` adds 20 cases on top of the existing
  `test_proj_pplay_v2.py`: fringe model, paths, oracle identity, and version composition. No bugs found.
- **Where:** `tests/conftest.py:8` pins `COURTSIDE_PPLAY=v1`, while prod runs v2.
- **Change:** add a v2 test module that exercises `pplay_v2_fix` and the fringe model.

### E8. Ownership refit objective is dominated by chalk (M / S)
- **Where:** `field.py:655-660` minimises MAE in percentage points, so one 10-point miss on a
  60 %-owned player outweighs twenty 1 %-vs-5 % misses. Leverage (D3) and ownership noise (D5)
  live in the low tail.
- **Change:** fit on log-odds (or weight by 1/own). Report tail MAE (own < 10 %) separately,
  and size D5's noise in log-odds.
- **Accept:** on uploaded slates, tail log-odds error goes down without a large loss on chalk.

---

## 6. Owner decisions needed

1. **Rule L6:** allow prop lines and DK salary as inputs to DFS projections only (P2). The
   props display stays market-free.
2. **Switch-ons:** `+mf` (recommended first), then `+pq` (P5).
3. **Live shadows + switch ledger** as the ship process from now on (E1).
4. **X factor v2 as leverage** (D3), with the late-news flag (D4). This changes what the tag
   means on the page.
5. Upload a big-GPP standings file per slate where possible (E5).
6. *For awareness:* DK FPPG (public DK data) is already used inside `truth_sims` and the
   ownership proxy. That is within the letter of L6, but the builder's numbers already lean on
   DK data.

## 7. Suggested order (revised after the second review)

| # | Item | Why now |
|---|---|---|
| 1 | B1, B2, B3, **B8** | Small, and every later number depends on them; B8 changes the P(play) training set |
| 2 | P5 (`+mf`, then `+pq`) | Already cleared the bar; owner flip (re-check `+pq` after B8's refit) |
| 3 | E1 shadows + ledger, E2 two-way SE | Needed to judge everything after |
| 4 | **P10** scenario-conditional outputs | Small, no model risk; enables D4, D3, D8 |
| 5 | **B10** then P1 / D1 role-aware FPPG blend | Largest cheap gain, without hurting promoted players |
| 6 | Owner decision on L6 → P2 | Props accumulate; start the blend weights early |
| 7 | D4 late-news flag → conditional chances → D3 X factor v2, with **E8** + D5 | Owner's stated goal; leverage needs honest ownership noise |
| 8 | D6 tail remap | GPP honesty |
| 9 | D2 cash by P(cash) + the §5.4 Top-N tie-break | Uses B1's split |
| 10 | P3 early-season priors, P4 reconciliation | Matters most in Oct–Nov |
| 11 | B4, B5, B7, B9, B11, E7, G1, G3, E3–E6, D7–D9, P6–P9 | As capacity allows; D7 gated on 10 slates |
| 12 | B6, D10 (Showdown) | Only if the owner plays Showdown |

## 8. Read-only checks done during the review

- **De-vig:** power de-vig is worse than multiplicative on 16,311 game closes (+0.0007 ± 0.0003
  LL).
- **Open vs close:** on 3,586 ESPN games, mean |open→close| is 1.66 pts; margin RMSE 14.01 vs
  open, 13.73 vs close.
- **Local DB state:** no stored portfolios (prod builds live on the Mac); 2,873 prop snapshot
  rows; `prop_compare` empty. Use the Mac prod DB for anything about live DFS or props.
- **Spot-checked in code by the orchestrator:**
  - B1 (`portfolio.py:262-265`, `:655`, `:741`);
  - B2 (`copula.py:26`, `:621`, `:692`; `reconcile.py`);
  - B3 (`predict.py:111-117`);
  - B4 (`injuries.py:272`);
  - E2 (`metrics.py:28-42`);
  - D1 (`runner.py:624`, `portfolio.py:98`);
  - B5 (`field.py:106`).

## 9. Second review (Fable, 2026-10-23)

An independent read-only review in three parts:
1. A code review done before opening this file.
2. `PLAYERS.md` plan vs code.
3. A check of every item here.

The orchestrator spot-checked B8 in code and schema.

**Item verdicts:**
- **Confirmed:** B1–B5, B7, P1, P2, P4–P6, P8, D1–D3, D5–D9, E1–E6. G1 is confirmed in part:
  `predict.py:221-257` was checked, but `features.py:113-127` was not.
- **Overstated:** B6 (the shadow-price direction is probably reversed) and P3 (the "× 0.6"
  detail). Both are corrected in place.
- **Unverifiable (plans or ideas):** P7, P9, D4, D10.

**New items folded in:** B8–B11, P10, E7, E8, the D1 stopgap warning (from B10), the §5.4
Top-N tie-break (in D2), missing acceptance criteria (B5, B6, D3, D4, D8, D9) and the revised
§7 order.

**`PLAYERS.md` plan vs code (drift):**
- §5.3 cash by P(score ≥ cash line): not built. Only `μ − 0.3(μ − p25)` exists (`runner.py:77`,
  `:624-627`; `swap.py:116`) → D2.
- §5.4 Top N "then by P(cash)": ranking is solve order only (`topn.py:118-131`) → D2.
- §4.3 S1 prior (height, weight, age, draft slot; age curve): absent (`state.py:356-387`) → P3.
  Usage v1 (T2.3) is built but not live.
- §4.1 weekly availability refit: not scheduled (a HANDOFF open item). Buckets are
  report-age-to-tip rather than run-time-to-tip; fine, but B8 corrupts the long buckets. The
  0.75-h bucket edge is finer than the hourly report cadence.
- §5.5 ownership proxy: the planned logit with position-rank/salary_z became a hand-set
  utility + Sinkhorn with a salary shadow price (documented). "Three field strengths" became
  information shares (`STRENGTH_W`), not optimizer mixes (documented).
- §5.7 captain ownership term: missing → B6.
- §4.2 nmu "top" rule → fitted softmax; water-filling → variance-weighted reconciliation. Both
  documented and justified.
- §4.5 "50,000 sims for N > 20": not verified.
- §7.1 L6: DK FPPG is used inside `truth_sims` and the field proxy (see §6 item 6).
