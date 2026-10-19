# Courtside players: projections, the "why", fantasy and DFS

Status: **design, 2026-10-19.** Nothing here is built yet. This doc guides a multi-agent build. It sits on top of `docs/MODEL.md` (game model), `docs/ANALYTICS.md` (PBP, stints, garbage time), `docs/TRENDS.md`, `docs/MLOPS.md` and `docs/research/player-predictions.md` (owner decisions).

**v1 scope (owner).**
- Projections for pts, reb, ast and minutes, plus DK/FD fantasy points. 3PM, stl, blk, tov and double-doubles are projected internally.
- A DFS optimizer (cash and GPP).
- Props shown as "our number vs the market", as information only.
- A plain-English why, both pre-game and post-game.

**Design in one paragraph.** P(play) comes from the injury report. A minutes model sets each player's share of Q1–Q3 minutes. Q4 and OT minutes are then driven by the *simulated* score. Per-possession rates come from empirical-Bayes (EB) shrinkage, then a log-linear adjustment for opponent, teammate absences, venue and the primary defender. Each game is simulated 10,000 times *top-down by team*. Pace and team shooting are shared within a sim, and team play counts are allocated to players multinomially. As a result:
- team totals always add up;
- teammates compete for shots;
- rebounds track misses and assists track teammates' makes;
- blowouts cut starters' minutes.

Pre-game and post-game whys are both exact Shapley decompositions, so the terms always sum to the number being explained.

## 0. Facts found while designing (they change the plan)

| finding | consequence |
|---|---|
| The PBP archive (8,647 games, 2019-20+) is on **geekom1 only**. The Mac's `var/raw/nba_cdn` has no `playbyplay`. | Sync it to the Mac (T0.2). |
| Only **10 cdn boxscores** are archived. `stints.build_rotation` needs the box for period-1 starters, silent-player recovery and the validation gate. | Back-fill the cdn boxscores for 2019-20+ (≈8,900 calls, ≈1.5 h). This also gives us `starter` flags and `officials`. |
| Official injury PDFs are only served back to **about late December 2025** (`sources/injury_report.py`), and the window keeps rolling. | **Urgent:** archive the 2025-26 game-day reports now. They are the only real pre-game availability history we can ever have (T0.1). |
| `player_games` has no DNP rows and no starter flag. | Absences are inferred from team games, as in `trends/player.py`. Starts come from the cdn box (2019+). |
| `GBMModel` stores `oof_resid_sd_` for the margin only. | Add the total's OOF sd and the margin–total residual correlation (T0.8). |

## 1. Product surface (phone-first, same components as the Brief)

**Tonight: player cards.** Tapping a game row (WEB.md) shows a **Players** segment with the top 4 per team by projected minutes, plus any listed player.
- **Row:** name, status badge ("Q · 55% to play"), then `27.4 pts · 4.1 reb · 7.2 ast · 35 min`, with the 80% range as a thin bar (`19–36`).
- **Why chips:** up to two, in the existing `factor_chip` style, e.g. `+3.1 Brunson out (usage)` and `−1.4 rim protection`.
- **Market line:** small and grey, e.g. `Line 26.5 · our chance above: 58%`. The words "over", "edge" and "value" are banned (STYLE.md).
- **Doubtful/Questionable players:** numbers are shown "if he plays", with the play probability beside them.
- Tapping the player opens his page.

**Last Night: "why" beat.** Each card gets one beat line for the game's biggest surprise, e.g. `Why Hart's 31 (our number 17): +9 min, 7-of-10 from three`. "Read more" adds a 2-sentence story. The game page adds an *Expected vs actual* table for everyone who played, each row with a server-SVG waterfall of the terms. Player Pulse gains a short list of "biggest surprises".

**DFS page `/dfs/<day>?site=dk|fd`.**
- Slate picker (main slate first).
- Sortable player list: salary, position, mean, floor (p25), ceiling (p90), value (mean per $1k), boom% (P(FP ≥ 6× salary/1000)), status. On phones it is a card list: name, salary, mean, then a p25–p90 strip.
- Cash lineup (1) and a GPP set (default 20, up to 150), with an exposure table and CSV export in each site's upload format.
- Footer: "Projections are simulations, not advice."

**Player page `/player/<id>`.**
- Tonight: a quantile strip per stat, the full why waterfall, and the market line.
- Last 10 games: expected vs actual, each with a one-line why.
- Season views: minutes trend, cumulative shot-making vs xFG ("running hot / cold"), and usage with and without his key teammates.

**Brief JSON 1.2 (additive).**
- `tonight.games[i].players.{home,away}[]`
- `last_night.games[i].player_why[]`, each with a `text` slot
- `player_pulse.surprises[]`
- `meta.dfs_available`

## 2. Modeling architecture

Notation: player *i* on team *t*; *n* = team possessions with *i* on court; rates are per on-court team possession.

### 2.1 Availability: P(play)
- **Starting table:** `injuries.PLAY_PROB` (Out 0, Doubtful 0.10, Questionable 0.5, Probable 0.9, unlisted 1.0).
- **Recalibration:** each status *s* gets a Beta posterior, starting from the current value with 20 pseudo-observations, and updated by matching snapshot statuses to who actually played. The snapshot used is the last one before tip. This runs weekly once n ≥ 30 per status.
- **Unlisted regulars:** P = 0.99, which covers late rest and DNP-CD.
- **Fringe players** (expected Q1–Q3 minutes < 4): they appear only through Q4 blowout minutes (§2.2). This is how game script, not the injury report, decides whether they play.
- **In the sim:** each listed player is a Bernoulli draw per sim. Usage and minutes are recomputed for each availability *scenario*. A team with q uncertain players has 2^q scenarios, capped at 16 by merging p < 0.05 / p > 0.95 into certainty.

### 2.2 Minutes model
**Target.** Minutes in Q1–Q3 when the player plays, from stints, 2019-20+ (≈190k player-games). Q1–Q3 is used because Q4 is where garbage time lives (ANALYTICS §3). Training on full-game minutes would teach the model the blowouts it is meant to simulate.

**Model.** LightGBM (L2), refit each season walk-forward, with features from before day *d* only:

| group | features |
|---|---|
| role | fast/slow minutes EWMA (α 0.2/0.02, as rung 2), last game, median of last 5, started last game, starts in last 10, team minutes rank |
| return | games since returning from ≥ 3 missed games (ramp-up), and games missed |
| absences (per scenario) | vacated minutes: Σ EWMA minutes of absent teammates, in total and in his position group (G/F/C); his shrunk historical share of that teammate's minutes when the teammate sat |
| coach | coach's shrunk rotation concentration (top-7 share of minutes, count of 30+ min players; coach id from `commonteamroster`, follows the coach across teams; 40 pseudo-games toward the league) |
| schedule | b2b, days rest, 3-in-4, age × b2b, and last season's rate of sitting second legs of back-to-backs |
| game | playoff flag, pregame \|spread\| (only for coaches who shorten early; the main blowout effect is in the sim), season phase |

The prediction is mapped to a team total of 180 minutes with `players.allocate_minutes` (water-filling, cap 36 per player over the 3 quarters).

**Q4 and OT (in the sim).**
- Q4 shares depend on the simulated margin after Q3: bins \|m₃\| < 10, 10–15, 15–20, 20–25, ≥ 25. They are estimated from stints for each role tier: top-5 by minutes, 6–8, 9+.
- Example of the expected shape (to be measured): starters' Q4 share falls from about 1.0× their Q1–Q3 rate to about 0.35× at 25+, and that time goes to the 9+ tier.
- If the sim is tied after Q4, OT minutes (25 per team) go out by closing-lineup share (last 5:00 of close games).
- **Calibration targets:** P(OT) 5.5–6.5% (the historical rate is 1,985/33,447 = 5.9%), and starters' minutes error by \|final margin\| bin within ±0.7 min.

**Noise.** Each player's Q1–Q3 share is multiplied by exp(ξ), with ξ ~ N(0, σ²_tier), and then renormalised.
- An "early exit" component replaces this with U(0.35, 0.85) with probability q_i = q₀ + q₁·PF/36. That covers foul trouble, in-game injury and ejection.
- σ_tier, q₀ and q₁ are fit so that the simulated minutes residual quantiles (p5…p95) match the backtest's.

### 2.3 Per-possession rates (empirical Bayes)
**Quantities.**
- **Volume** per on-court possession: 2PA, 3PA, FT trips, TOV.
- **Shares** while on court: assist share (AST ÷ teammates' FGM), OREB and DREB share (÷ available rebounds), STL share (÷ opponent TOV), BLK share (÷ opponent 2PA).
- **Makes:** 2P%, 3P%, FT%, and and-ones per FGM.

**Estimate.**
`r̂ = (Σ_g w_g x_g + κ·μ_prior) / (Σ_g w_g n_g + κ)`
- w_g has a 120-day half-life within a season, and earlier seasons are multiplied by 0.6 each summer (the rung-2 box carry).
- κ is fit per stat by beta-binomial method of moments on 2015-20, with the walk-forward folds as a check. Starting guesses: 3P% about 600 attempts, FT% 200, 2P% 300; volume rates and shares about 400 possessions.
- μ_prior comes from a ridge regression on position group, height, weight, age, draft pick (rookies) and last season's rate.

**Age curve.** The delta method on consecutive seasons, weighted by the harmonic mean of possessions, smoothed with a spline per stat. It is applied as `exp(a(age_now) − a(age_at_data))`. Birthdates come from `commonteamroster` (2015-16+); with no birthdate, the adjustment is 0.

**Adjustments.** Each player-game rate is multiplied by a log-linear adjustment:
`log r = log r̂ + β_opp·D_opp,k,pos + β_home·home + β_rest·b2b + Σ_{j absent} φ_ij,k + β_def·Θ_i + [β_ref·R_crew]`
- β is fit as a Poisson/binomial GLM with offset log(n·r̂), per stat, refit each season (2019-20+). It is kept log-linear on purpose, so the attribution in §3.1 is exact and cheap.
- **Opponent defense by stat and position.**
  - `D = log(Σ actual / Σ expected)` over opposing players in group *pos* against that defense. Expected uses each player's own pre-game r̂, which removes the "who they faced" bias of raw defense-vs-position.
  - Recency-weighted and shrunk by N/(N+K_D), with N = expected count.
  - Separate K per stat: volume and shares K ≈ 300 expected events; **3P% allowed gets K ≈ 5,000**, which is almost no effect, because opponent 3P% is mostly luck. 3PA *rate* allowed is real scheme and gets the normal K.
  - Position groups come from `playerindex`. `matchupsrollup` gives the defender's G/F/C time mix, which is used for the primary defender below.
- **Pace** does not enter the rates. It is the possession count (§2.4).
- **Teammate absences.**
  - Multinomial allocation already renormalises usage proportionally. φ_ij captures the *deviation* from proportional, from our stints over the last 2 seasons (one-season half-life):
    `φ^emp_ij = log(usg_i|j off / usg_i|j on) − log(proportional factor)`, with
    `φ_ij = φ^emp · n_off/(n_off + K_φ)`.
  - n_off is the possessions with *i* on and *j* off. K_φ ∈ {250, 500, 1000, 2000, 4000} is chosen walk-forward on "natural experiment" games where *j* sat.
  - The same machinery is used for assist share. Rebound share uses proportional renormalisation only.
- **Primary defender.**
  - Expected matchup shares s_id come from prior meetings (`boxscorematchupsv3`, partial possessions, causal), else the opponent's G/F/C assignment mix from `matchupsrollup`, lagged one season.
  - Defender effect: `θ_d = Σ(pts − ê_o)/Σ pp`, summed over the matchups defender *d* has guarded, with ê_o the offensive player's own expected points per partial possession and pp the partial possessions.
  - Shrunk as `θ̃_d = θ_d · N_d/(N_d + K_θ)`, with **K_θ = N̄·(1 − ρ)/ρ**. ρ is the odd/even-game split-half reliability of θ_d (expected 0.1–0.3, so K_θ is several thousand partial possessions, i.e. more than a starter's season).
  - `Θ_i = Σ_d s_id·θ̃_d / ppp_i`, capped at ±4% of points. This is applied to volume and makes jointly, because defenders move shot frequency more than make rate.
  - It is shown in the why only when \|Δpts\| ≥ 0.5.
- **Referee crew** (FT trips, fouls): included only if crew effects have split-half reliability > 0.2 and walk-forward FT-trip log-likelihood improves by more than 2 SE. Otherwise it is dropped and documented as tested.

### 2.4 Joint simulation (per game, S = 10,000)
The game model is the **environment**: `gbm_market` mean margin μ_M and total μ_T (it sees the game line, not props, so "our prop number" stays independent of prop markets). `gbm_opinion` is the fallback when there is no line, and runs as a logged shadow.

A useful fact for calibration: the variance of a team's score around a good line, about (17.7² + 12.4²)/4 ≈ 117 (sd ≈ 10.8, from the market's total and margin errors), is close to pure possession-outcome noise (≈ 1.25 pts² per possession × 100). So the simulator must be **bottom-up**: almost all the variance comes from shot outcomes, and only small shared latent terms are added on top.

**Per sim s:**
1. **Pace.** `P_s ~ N(P̄, σ_P²)` possessions, shared by both teams. P̄ is a log5 pace forecast, scaled so that the expected total matches μ_T (§2.5).
2. **Team shooting latents.** `η_t,s ~ N(0, τ_e²)` for each team, on the logit of make probabilities. There is also a per-player latent `ζ_i,s ~ N(0, τ_p²)` for night-to-night shooting variance beyond binomial.
3. **Availability scenario** draw (§2.1), which fixes the weights.
4. **Phase A (Q1–Q3)**, with 75% of P_s possessions:
   - Minutes: shares from §2.2 with noise.
   - Team play counts: N_t,k ~ Poisson(0.75·P_s · Σ_i (M_i/36)·r_ik). The Q1–Q3 minute shares (M_i/36) average to the 5 on-court slots.
   - Allocation to players: **Dirichlet-multinomial** with weights ∝ M_i·r_ik. Concentration is fit from player-game overdispersion; this is what plays the role of a negative binomial for volume.
   - Makes: Binomial(attempts, logistic(logit p̂ + δ_t + η + ζ)).
   - Free throws: FTA = 2·trips + and-ones.
   - Rebounds: team OREB ~ Binomial(misses, oreb%_{t vs opp}), and DREB is the rest of the opponent's misses. Both are allocated by share.
   - Assists: team AST ~ Binomial(FGM, ast%), allocated by share.
   - STL ~ Binomial(opp TOV, share); BLK ~ Binomial(opp 2PA, rate). Both allocated.
5. **Q4** (25% of P_s), with Q4 minute shares set by \|m₃\|. Then OT phases at 5/48 of P_s each, repeated while tied.
6. **Derived:** pts, 3PM, DD/TD, DK/FD points, and garbage-time splits (expected garbage share of Q4 as a function of \|m₃\|, measured from PBP).

**Distribution choices.**

| quantity | choice | why |
|---|---|---|
| pace | normal | possessions per game are close to Gaussian |
| play volume | Poisson team totals, Dirichlet-multinomial allocation | team consistency, overdispersed per player (NB-like) |
| makes | binomial with logit-normal latents | shared team nights, 3PM falls out of the same draw |
| reb, ast, stl, blk | binomial on simulated opportunities | correlation with misses, makes and turnovers |
| minutes | lognormal share noise, early-exit mixture, water-filling | foul trouble gives a heavy left tail |

Points are therefore emergent: about normal for starters, right-skewed for bench players.

**Variance calibration.** σ_P, τ_e and τ_p are fit by simulated method of moments on 3,000 backtest games. The targets are:
- sd(margin) and sd(total) equal to the game model's OOF residual sd, within ±3%;
- the corr(home, away points) implied by those two;
- P(OT) in 5.5–6.5%;
- player FG% dispersion.

**Correlation is structural, not a fitted matrix:**
- shared pace (everyone, positively);
- η (teammates' points, positively);
- shot competition (teammates' FGA, negatively);
- AST↔teammates' FGM and REB↔misses;
- margin→Q4 minutes (starters negatively with blowouts);
- availability (a teammate's absence raises usage).

**Validation (T2.7).** Empirical within-team residual correlations on 2024-25 must be matched within ±0.05:
- pts of the top two scorers;
- ast of the guard and pts of the teammates;
- reb vs opponent FG%.

**Monte Carlo error at S = 10k.** About 0.08 pts on a mean (sd 8), and at most 0.005 on P(above a line). The same sim index is used across games, so slate-level FP matrices are valid for DFS (games are independent). Seeds are deterministic: `hash(game_id, nba_day, model_version)`. These are common random numbers, so a projection moves between pregame runs only when inputs move.

### 2.5 Reconciliation (team totals)
After rate adjustments, solve two numbers per game:
1. **Pace.** Set P̄ so that the log-gap g = log(μ_T / T_players) is split α to pace and 1 − α to efficiency. α is fit from history by regressing log(actual possessions / log5 pace) on g; the prior guess is 0.5.
2. **Efficiency.** Per team, solve for the logit shift δ_t (1-D root, monotone) so that the mean-field E[team pts] = (μ_T ± μ_M)/2.

**Rules.**
- **Acceptance:** \|Σ_i E_sim[pts_i] − E_model[team pts]\| ≤ 0.5 for every team-game.
- The pre-reconciliation gap is logged as a diagnostic. \|gap\| > 6 pts raises a data-check warning, which usually means a stale injury report or a mis-matched name.
- \|δ_t\| > 0.15 logit is capped, and the remaining gap is reported rather than hidden.

### 2.6 Runtime budget (Mac mini M4, loaded, 2 threads)

| step | where | budget |
|---|---|---|
| nightly state: EB rates, stints, with/without, defense indices, xFG for last night's shots, post-game whys | postgame 03:00/09:00 | ≤ 5 min |
| pregame for a 15-game slate: load state 2 s, minutes GBM over ≤ 16 scenarios per team 3 s, reconciliation 2 s, sims 10k × 15 (vectorised conditional-binomial multinomials, ≈ 1–2 s per game) ≤ 30 s, attribution 10 s, store 5 s | each pregame run | **≤ 60 s** (limit 120 s) |
| DFS build: cash plus 20 GPP lineups | 17:00 and 18:30 only | ≤ 90 s |

Storage: per-sim player FP and stat arrays as int16/float32 `npz` at `$COURTSIDE_DATA/sims/<day>/<run>.npz` (about 30 MB per slate, pruned after 8 days like `production.KEEP_DAYS`).

## 3. The "why" engine

### 3.1 Pre-game attribution
**Baseline B** is his season average per game played, the number readers know.
- With fewer than 10 games: B = (n·avg_cur + (10 − n)·avg_prev)/10.
- Rookies with no history: B is the model's neutral projection, and the form term is 0.

**Factor groups g:**
1. form/talent (EB rate vs his raw season rate);
2. role (rest, ramp-up, coach, other minutes changes);
3. teammate absences (minutes and usage);
4. opponent defense;
5. primary defender;
6. pace and game environment;
7. venue and referees.

**Mean-field function.** `f(S)` is the analytic expectation (minutes × possessions × rates × make probabilities) with the groups in S at tonight's values and the rest at neutral values: his season-average minutes, pace and rates, an average opponent, a healthy team, and neutral venue.

**Exact decomposition.**
`Projection − B = Σ_g φ_g + script + (f(∅) − B)`, where:
- φ_g is the **exact Shapley value** over the 7 groups (2⁷ = 128 vectorised evaluations per player-stat);
- `script = sim mean − f(all)` (blowout risk, OT, caps; the reader label is "game script");
- `f(∅) − B` is folded into the form term.

Inside the absences group, each absent teammate gets an Owen-value split (Shapley within the group), giving chips like "+3.1 Brunson out (usage to Hart)".

**Display.** Show factors with \|φ\| ≥ 0.5 pts (≥ 0.3 reb/ast), at most 3. Fantasy points are attributed as the linear combination of the stat attributions, plus a DD/TD bonus term.

### 3.2 xFG (our shot-quality model)
**Data.** Every FGA in the cdn PBP from 2019-20 on, about 1.4M shots, excluding `heave` actions.

**Features:**
- distance;
- angle from (x, y);
- `area`, corner-3 flag, is_three;
- `sub_type` (layup, dunk, hook, jump shot);
- `descriptor` (pull-up, step-back, driving, floating, turnaround, fadeaway, alley-oop, putback, tip, cutting);
- qualifiers (fast break, 2nd chance, from turnover);
- seconds since the possession started and since an OREB (from events);
- period, clock and last-2-minutes flag;
- score margin and garbage flag;
- home;
- **context prior:** the shooter's *previous-season* defender-distance mix (shares 0–2 / 2–4 / 4–6 / 6+ ft) and the shot's zone-by-distance bucket.

**Never used:**
- `assist_id` (only exists on makes) and `block_id` (only on misses);
- any qualifier with less than 1% of its occurrences on misses (an automated audit test). `pointsinthepaint` is the prime suspect.
- shooter identity: xFG is shot quality, not skill.

**Model.** LightGBM binary classifier, then isotonic calibration on the last training season.
- **Evaluation:** walk-forward by season (train < S, test S), 2021-22..2025-26.
- **Baselines:** zone × shot-type FG%, and distance-only logistic.
- **Acceptance:**
  - log-loss at least 1.5% better than zone × type in every test season;
  - ECE ≤ 0.01;
  - per-zone calibration within ±1.5 pp for zones with n ≥ 5,000;
  - team-game xPTS vs actual FG points r ≥ 0.6.

**Skill.** The player's shot-making ratio m̂_i = EB(FG pts / xPTS), with κ ≈ 400 xPTS-weighted shots. It is reported, not baked into xFG. The production model is frozen per season and refit in October.

### 3.3 Post-game decomposition (actual − expected sums exactly)
Expected is the stored last pre-tip projection, conditional on playing. DNPs get no why; DFS scores them 0.

**Points.** Split actual and expected into garbage and non-garbage parts (garbage time as in ANALYTICS; expected garbage points come from the sim).
`F = M · π · (v·q·m + f·c)` on non-garbage play, where:
- M = minutes;
- π = team possessions per minute while on court;
- v = FGA per on-court possession;
- q = xPTS per FGA from his actual shots;
- m = FG pts / xPTS;
- f = FTA per on-court possession;
- c = FT%.

Expected values (M̂, π̂, v̂, q̂ = his EB xPTS/FGA × the opponent's quality-allowed index, m̂, f̂, ĉ) come from the pre-game state.

**Terms.** Exact Shapley over six groups: **minutes, pace, shot volume, shot quality, shot-making, free throws**, plus **garbage time** = G − Ĝ.
- **Jensen gap:** E[product] ≠ product of expectations, so M̂ is replaced by an effective M̃ = M̂·(Ŷ_ng/F(x̂)). The test is \|M̃ − M̂\| ≤ 1.5 min for 95% of player-games.
- **Defender carve-out:** `D = Σ_d (s_d^act − s_d^exp)·θ̃_d·n`, moved from shot-making to a **defender** term. It is shown only when \|D\| ≥ 0.5 and his matchup partial possessions are ≥ 20.
- Defenders also appear as descriptive facts: who guarded him (≥ 25% share), partial possessions, his points and FGM/FGA against them, and the defender's points allowed per 100 partial possessions tonight vs that defender's norm.
- **Minutes sub-labels:** the minutes term is split by phase (Q1–Q3, Q4, OT), with labels foul trouble (PF ≥ 4 by half-time), blowout (\|m₃\| ≥ 15), overtime, early exit (the ANALYTICS heuristic), else rotation. If the phase differences offset each other (\|ΔM\| < 1), it stays one term.

**Rebounds and assists:**
- REB = M·π·o·s, with o = rebound chances per on-court possession and s = share.
- AST = M·π·t·a, with t = teammates' FGM per on-court possession and a = assist share.
- Terms: minutes, pace, opportunities, share, garbage.

**Fantasy points:** the scoring-weighted sum of the stat terms plus a bonus term, regrouped by label.

**Tests:** Σ terms = actual − expected to 1e-9 on all 2024-25 player-games; terms are zero-safe (Shapley handles FGA = 0).

### 3.4 Output to the writer, and fact-check rules
`why_facts` per player-game:
```
{player, team, stat, expected, actual, diff, range80,
 terms: [{term, value, detail}],          # sorted by |value|
 shooting: {fgm, fga, fg3m, fg3a, xfg_pts, rim_att, typical_rim_att},
 minutes: {expected, actual, label, pf},
 defenders: [{name, partial_poss, pts, fgm, fga, allowed_vs_norm}],
 flags}
```

**Selection.** At most 2 players per game: the largest \|PIT − 0.5\| among players with expected minutes ≥ 20 or actual pts ≥ 25. The winning side breaks ties.

**Fact-check** (extends `brief/factcheck.py` with kind `player_why`, in a new module):
1. Existing number, name and pair rules apply.
2. A term phrase (lexicon: minutes, shots/volume, shot quality/looks, shooting/made, free throws, defense/guarded, garbage/blowout) may only be used if \|value\| ≥ 1.0 pt (0.7 reb/ast) **and its direction matches the sign**.
3. "Luck" and "unsustainable" are banned. Use "shot-making above his norm".
4. A causal defender claim ("held by", "slowed by") needs a defender term ≥ 1.5. Otherwise only the descriptive "mostly guarded by X", and only when share ≥ 40%.
5. "Injury" words follow the existing rule (`possible_injuries` gives only "left and did not return").
6. Our number is "our number" or "projection", never a tip.
7. If no term passes rule 2, fall back to the template: "{n} more minutes than expected and {k}-of-{m} from three".

## 4. Fantasy and DFS

**Scoring** (DraftKings NBA Classic and FanDuel NBA rules pages; **confirm every coefficient against the `rules_json` captured with each slate before T1.4 ships**):
- **DK:** `PTS + 0.5·3PM + 1.25·REB + 1.5·AST + 2·STL + 2·BLK − 0.5·TOV + 1.5·DD + 3·TD`. DD and TD count across PTS/REB/AST/STL/BLK. **To confirm:** whether a triple-double also earns the DD bonus (believed yes, 4.5 total).
- **FD:** `PTS + 1.2·REB + 1.5·AST + 3·STL + 3·BLK − TOV`.
- **Rosters:**
  - DK: 8 slots (PG, SG, SF, PF, C, G, F, UTIL), $50,000 cap, players from ≥ 2 games.
  - FD: 9 slots (2 PG, 2 SG, 2 SF, 2 PF, C), $60,000 cap, ≤ 4 players per team, ≥ 3 teams.
  - Positional eligibility comes from each salary file. All rules are to be confirmed.

**Distributions** are computed per sim from the simulated box line. Shown: mean, p25 (floor), p50, p75, p90 (ceiling), boom%, and P(DD).
- DFS uses the **unconditional** distribution (0 when the player doesn't play).
- Props use the distribution **conditional on playing**, because props are void on a DNP.
- With late swap out of scope, cash lineups exclude players with P(play) < 0.85 by default.

**Optimizer: OR-Tools CP-SAT** (Apache-2.0, arm64 wheels, fast repeated solves), with our own roster model of about 200 lines. `pydfs-lineup-optimizer` is the test oracle: it must agree on the optimal objective for 50 random slates.
- **Why not pydfs in production:**
  - its rules are hard-coded, so a site rule change means waiting for a release, and its maintenance is uncertain;
  - its PuLP/CBC backend is about 10× slower per solve;
  - GPP needs custom simulation objectives.
- **Model.** Binary x_{i,slot} over eligible slots:
  - each slot filled once;
  - each player used at most once;
  - salary ≤ cap;
  - team and game constraints;
  - FP scaled ×100 to integers;
  - locks and excludes.
- **Cash:** maximise Σ μ_i·x_i.
- **GPP (Hunter–Vielma–Zaman style, but with a simulated objective):**
  1. **Candidates:** for 600 random sims, solve the ILP on `0.5·μ_i + 0.5·FP_i,s` (the sim brings correlated outcomes: OT games, pace-ups, teammate absences). Add the top 50 mean lineups via no-good cuts.
  2. **Field proxy (no ownership data):** 5,000 lineups from a randomised greedy-plus-swap builder on `μ_i·exp(N(0, 0.15²))`, standing in for a field of optimizer users. The per-sim target t_s is the 99.8th percentile of field scores in sim s, so the bar rises in sims where everyone scores.
  3. **Portfolio:** greedily add the lineup that most increases `(1/S)·Σ_s 1[max_k score_k,s ≥ t_s]`, subject to overlap ≤ γ shared players (DK 5 of 8, FD 6 of 9, the HVZ overlap constraint) and exposure caps (default max 60%, user min/max per player).
- **Without ownership data:** we show no ownership or leverage numbers. The field proxy is used internally only, and the GPP output is labelled "high-ceiling, diversified". If ownership is ever bought, it replaces the field-sampling weights.

## 5. Evaluation and MLOps

### 5.1 Backtest
**Design.**
- **Walk-forward by season:** test 2021-22..2025-26, with rates and minutes state run causally day by day. Model coefficients (GLM β, minutes GBM, xFG) are fit on seasons < S. Hyperparameters (κ, K_φ, K_D, K_θ, half-lives) are frozen on 2019-20/2020-21.
- **Environment:** the dumped walk-forward `gbm_market` predictions (rung 3).
- **Population:** players with pregame expected minutes ≥ 15 who played. Fringe players are reported separately. **Never filter on actual minutes.**

**Benchmarks:**
1. season-to-date average (prior-season blend under 10 games);
2. last-10 average;
3. rung-2 EWMA minutes × season per-minute rates;
4. **salary-implied** DK FP: a linear fit on salary, *if* RotoGuru history is reachable (T0.9);
5. DARKO, **only with permission** (the research says don't scrape; ask Kostya Medvedovsky);
6. from 2026-10-20, no-vig props: our P(above line) vs the market's.

No-vig method: multiplicative and power both get computed, and the one with better outcome log-loss after 4 weeks is kept as the benchmark.

**Metrics:**
- per stat: MAE, RMSE, CRPS (from samples), bias;
- coverage of the 50% and 80% intervals, and the PIT histogram;
- minutes MAE;
- calibration of P(above line): Spiegelhalter z on the over/under outcome. Before props exist, the "lines" are proxies (season average rounded to .5); live, they are real lines;
- log-loss vs no-vig on the same props, with SE clustered by game;
- DFS: the actual score's percentile among 100k random valid lineups (salary ≥ 98% of cap), the ratio to hindsight-optimal, and a paired comparison against lineups the same optimizer builds from naive projections.

**Acceptance targets for v1 champion** (walk-forward, all significant at 2 SE clustered by game):

| metric | target |
|---|---|
| minutes MAE (expected ≥ 15) | ≥ 8% better than rung-2 EWMA on the same rows; ≤ 4.5 in backtest; ≤ 4.0 live with real reports |
| pts / reb / ast MAE and CRPS | ≥ 4% / 3% / 3% better than the best naive benchmark |
| 80% / 50% coverage | 78–82% / 47–53% overall; 75–85% in each of 6 role buckets |
| PIT | every decile bin within 8–12% |
| P(above line) calibration | \|z\| < 2.5 on proxy lines (n ≥ 10k); live paired log-loss vs no-vig reported weekly (we expect the market to be sharper) |
| DK FP MAE | ≥ 10% better than salary-implied (if available) |
| cash lineup | median percentile ≥ 85th; beats the naive-projection optimizer in ≥ 55% of slates |

The research's "starters' points MAE 5–6" is unverified. We publish our own number instead.

### 5.2 Leakage rules (tests enforce each one)
1. **Availability (the rung-2 trap).**
   - Teammates use the `avail` rule: oracle only for regulars with expected ≥ 20 min; fringe teammates are available.
   - The player himself is evaluated conditional on playing.
   - Starting on 2025-12-xx, the backfilled official reports (T0.1) give a production-faithful slice. Report both, and quote only the faithful one as "live-like".
   - **Leak probe:** a walk-forward LightGBM predicting projection residuals from availability-derived features alone must improve MAE by < 0.5%, the analogue of the 58.3% → 50.8% ATS probe.
2. **No lookahead.** Scrambling every box, PBP, matchup and tracking row on or after a cut date must leave projections up to that date bit-identical (`check_no_lookahead` pattern).
3. **Same-game matchup and tracking data are post-game only.** Pre-game defender shares come from games before *d*. **Season tables** (matchupsrollup, leagueseasonmatchups, defended FG%, defender distance, synergy) are end-of-season aggregates and are only used **lagged one season**. Causal to-date versions are rebuilt from per-game data.
4. **No starters pre-tip.** Only "started last game" or starts in the last 10.
5. **Never features:** props, salaries, actual game script (foul trouble, garbage time), and closing lines beyond what `gbm_market` already uses. The xFG feature audit (§3.2) runs in CI-less local tests.
6. **Rotation features** come from games before *d* only, and DNP-CDs are never inferred from same-day box scores.

### 5.3 Registry, monitoring, alerts (reuse `mlops/`)
**Model names:**
- `player_minutes`
- `player_rates`
- `player_sim`: distribution and variance parameters; its spec hash includes the hashes of the other two, so it is the champion that labels rows.
- `xfg`

Each row in `player_projections` carries `model_version` and `role`. One shadow is allowed, because it doubles the sim time.

**Gates.**
- **Offline:** composite CRPS (mean of per-stat CRPS ÷ baseline CRPS) below −2 SE; no stat worse beyond 2 SE; coverage still in band.
- **Online:** ≥ 3,000 resolved player-games and ≥ 1,000 props, with paired CRPS ≤ 0 and no critical alert.

**Monitoring.** A new table `player_metric_snapshots`, per model × stat, over windows `7d`, `30d`, `last2000` (player-games), `season` and `props_last1000`.

**Alerts:**

| alert | fires when |
|---|---|
| performance | props log-loss worse than no-vig by > 2 clustered SE (last 1,000) |
| calibration | Spiegelhalter \|z\| > 2.5 on P(above line), n ≥ 500 |
| coverage | 80% coverage outside 0.80 ± 3 clustered SE (last 2,000) |
| minutes bias | projected starters off by more than ±1.0 min (last 300) |
| reconciliation | any team gap > 0.5 after reconciliation, or > 6 before |
| xFG drift | league xFG% − FG% beyond ±1.5 pp over 7 days |
| drift | PSI on projected minutes, usage weights, pace |

**Data checks:**
- props captured pre-tip for ≥ 90% of games (last snapshot ≤ 90 min before tip);
- DK/FD main-slate salaries by 12:00 ET;
- ID match rate ≥ 98%;
- matchups fetched for all finals;
- stint validation pass rate ≥ 99%;
- all shots scored by xFG.

**Surfaces.** A "Players" tab on `/health`.

## 6. Data model (migration 7: one owner, task T0.0)

| table | key columns |
|---|---|
| `player_bio` | player_id PK, name, birthdate, height_in, weight, position, pos_group (G/F/C), draft_year, draft_pick, source, updated_at |
| `player_ids` | (source, source_id) PK, player_id, name, team, first_seen, last_seen, how (game_join / roster_name / fuzzy / manual), confidence |
| `player_aliases` | (source, name_norm, team) PK, player_id, note: manual overrides |
| `stints` | game_id, period, start, end, home5, away5, home_pts, away_pts, home_poss, away_poss |
| `player_game_oncourt` | (game_id, player_id) PK, sec_q13, sec_q4, sec_ot, sec_garbage, on_poss, opp_on_poss, team_fga/fgm/misses_on, opp_misses_on, tm_fgm_on (teammates), opp_tov_on, opp_2pa_on, started |
| `shots` | (game_id, action_number) PK, player_id, team, period, clock, x, y, distance, area, is_three, sub_type, descriptor, quals, made, garbage, xfg, xfg_version |
| `rate_state` | (as_of, player_id, stat) PK, rate, n_eff, prior, kappa: nightly EB state |
| `player_projections` | id PK, game_id, nba_day, player_id, team, created_at, model, model_version, role, p_play, scenario_json, inputs_json; UNIQUE(game_id, player_id, created_at, model_version) |
| `player_projection_stats` | (projection_id, stat, cond) PK; stat ∈ min, pts, reb, ast, fg3m, stl, blk, tov, dd, td, dk, fd; cond ∈ played / all; mean, sd, q05, q10, q25, q50, q75, q90, q95 |
| `projection_factors` | (projection_id, stat, factor, sub) PK, value, detail_json |
| `sim_runs` | run_id PK, nba_day, created_at, n_sims, seed, model_version, path, seconds, recon_json |
| `prop_compare` | (projection_id, market, line, captured_at) PK, over_price, under_price, mkt_p_above (no-vig), our_p_above, our_p_push |
| `postgame_why` | (game_id, player_id, stat) PK, model_version, expected, actual, terms_json, facts_json, computed_at |
| `postgame_terms` | (game_id, player_id, stat, term) PK, value: long form for season "running hot" views |
| `dfs_slates` | slate_id PK, site, nba_day, slate_type, game_ids_json, lock_utc, rules_json |
| `dfs_salaries` | (slate_id, site_player_id) PK, player_id, name, team, positions, salary, status, captured_at |
| `dfs_lineups` | id PK, slate_id, created_at, model_version, mode (cash/gpp), rank, players_json, salary, mean, p90, p_target, actual, pct_random |
| `player_metric_snapshots` | (as_of, model, stat, window) PK, model_version, n, mae, rmse, crps, cov50, cov80, bias, mkt_n, ll, ll_mkt, ll_diff, ll_diff_se, details_json |

The prop, salary and referee *snapshot* tables belong to the opening-night collector. If that agent has already defined them, adopt its tables. This design only requires these columns:
- props: `game_id, captured_at, provider, espn_athlete_id, market, line, over_price, under_price, raw_path`;
- salaries: `site_player_id, name, team, positions, salary, game`.

**ID matching** (`proj/ids.py`, reusing `injuries.match_player`: accents, suffixes, nickname groups, FUZZY_MIN 0.86 with a 0.05 margin):
1. **ESPN athlete id:** join each final game's ESPN summary box to the cdn box by (game, team, normalised name). That is exact for anyone who has played. Players who haven't played yet are matched by name against the ESPN team roster plus `commonteamroster`.
2. **DK/FD:** match name + team + slate date against `recent_rosters`, extended with `commonteamroster` for new signings. Persist (site, site_player_id), with DK's stable player id, not the per-slate draftable id (to confirm).
3. **Same-team near-duplicates** (Jalen and Jaylin Williams, OKC) need an exact first-name token match whenever a second candidate on the team clears 0.80.
4. **Unmatched** players raise a data check and go to the aliases table.

**Expected match rates:** ESPN → NBA ≥ 99.5% after week 1; DK/FD ≥ 98% of salaried players automatically and 100% of players with expected ≥ 10 min after aliases.

## 7. Phased build plan

Code lives in a new package `src/courtside/proj/` (`dfs/` inside it), with tests `tests/test_proj_*.py`. `db.py` migrations and the `cli.py` hook are owned **only** by T0.0; later schema changes go through one agent at a time.

Legend: ★ = critical path; ∥ = can run in parallel within its phase; O / S / H = Opus / Sonnet / Haiku.

### Phase 0: foundations (Oct 19–23, all ∥ after T0.0)

| ID | task | files owned | in → out | acceptance |
|---|---|---|---|---|
| T0.0 ★ S | migration 7 (all §6 tables), `courtside proj` CLI hook | `db.py`, `cli.py`, `proj/__init__.py`, `proj/cli.py` | §6 → schema | migrate is idempotent; `tests/test_db.py` green |
| T0.1 ★ S | **URGENT** backfill of official injury PDFs, late Dec 2025 → June 2026: game days at 13:30, 17:30 and the last before first tip | `backfill_injuries.py` | server → raw + `injury_snapshots` | earliest available date recorded; ≥ 95% of game days captured; parse rate ≥ 99% of rows |
| T0.2 ★ H | cdn boxscore backfill 2019-20+; rsync the PBP archive to the Mac | `backfill_pbp.py` (box kind) | cdn → raw | ≥ 99% of games that have PBP also have a box |
| T0.3 ★ S | historical stints and on-court tallies | `analytics/stints_store.py` | box + PBP → `stints`, `player_game_oncourt` | validation pass (±1 s, +/- exact) ≥ 99% of games; ≤ 10 min run |
| T0.4 H | bio, positions, coaches | `proj/bio.py` | playerindex, commonteamroster → `player_bio` | position on ≥ 99% of 2019+ player-games; birthdate ≥ 95% for 2015+ |
| T0.5 S | ingest matchups, tracking, hustle | `proj/matchups.py` | tonight's raw → tables | Σ partial possessions per offensive player within 10% of on-court possessions in ≥ 95% of player-games |
| T0.6 S | ID mapping (ESPN, DK, FD) + data check | `proj/ids.py` | §6 | rates as in §6 on archived and opening-night data |
| T0.7 H | fantasy scoring and roster rules from `rules_json` | `proj/fantasy.py` | rules → functions | 30 hand-scored lines exact; DD/TD edge cases |
| T0.8 H | GBM total OOF sd and margin–total residual correlation | `model/gbm.py` (2 attributes) | — | values logged; rung-3 tests green |
| T0.9 H | probes: RotoGuru salary history depth; any CC-licensed historical props data | `docs/research/` note | — | yes/no with seasons and licence |

### Phase 1: MVP, live by about Oct 27 (ships early)
- **Minutes v0:** rung-2 EWMA + P(play) + `allocate_minutes`. The blowout effect comes from a box-score regression of the minutes deviation on final \|margin\| by tier (2001+ data is available now).
- **Rates v0:** EB per-minute rates.
- **Sim v0:** per-stat negative binomial, sharing the minutes draw, pace scaled by the total.
- Proportional reconciliation, cash optimizer, Tonight cards, and DFS page v0.

| ID | task | files | acceptance |
|---|---|---|---|
| T1.1 ★ S | rate state v0 and nightly job | `proj/state.py` | causal (no-lookahead test); ≤ 60 s nightly |
| T1.2 ★ S | minutes v0 + sim v0 + store + pregame hook | `proj/minutes.py`, `proj/sim.py`, `proj/store.py`, `proj/pipeline.py` | on 2024-25 (avail mode): pts MAE ≤ season-average benchmark; 80% coverage 75–85%; ≤ 60 s per slate |
| T1.3 ∥ S | prop parse, no-vig, `prop_compare` | `proj/props.py` | over/under semantics verified (P(over) falls as the alt line rises); ≥ 90% of main props joined |
| T1.4 ∥ S | cash optimizer (CP-SAT) + pydfs oracle test | `proj/dfs/optimizer.py` | matches pydfs on 50 slates; < 0.5 s |
| T1.5 ∥ S | web: Tonight player panel, `/dfs` v0, Brief 1.2 keys | `web/players_view.py`, `templates/dfs.html`, `_players.html`, `brief/assemble.py` (players keys only) | 390 px screenshots light/dark; null-safe |

### Phase 2: model v1 (Oct 27 – Nov 21)

| ID | task | files | acceptance |
|---|---|---|---|
| T2.1 ★ S (O review) | player backtest harness: walk-forward, benchmarks, metrics, leak tests and probe, report | `proj/backtest.py`, `proj/metrics.py` | reproduces the T1.2 numbers; probe < 0.5% |
| T2.2 ∥ O | with/without engine (φ, K_φ fit) | `proj/usage.py` | on "teammate sat" games: usage log-likelihood beats proportional-only by > 2 SE |
| T2.3 ∥ S | opponent defense indices by stat and position | `proj/defense.py` | 3P% allowed effect shrunk to near 0; β_opp in [0.5, 1.5] |
| T2.4 ∥ O | primary-defender θ with reliability shrinkage | `proj/defender.py` | ρ and K_θ reported; adds ≥ 0 CRPS (else ships switched off) |
| T2.5 ★ O | rate model v1: EB + age curve + GLM | `proj/rates.py` | §5.1 pts/reb/ast targets reached with T2.6–T2.7 |
| T2.6 ★ O | minutes v1: Q1–Q3 GBM, Q4 tables, OT, noise mixture, coach | `proj/minutes.py` (takes it over) | minutes target in §5.1; P(OT) 5.5–6.5% |
| T2.7 ★ O | joint team-allocation sim, variance SMM, reconciliation | `proj/sim.py`, `proj/reconcile.py` | §2.4–2.5 targets; ≤ 30 s per 15 games |
| T2.8 ∥ S | referee crew test | `proj/refs.py` | written verdict with reliability and log-likelihood gain |
| T2.9 ∥ S | P(play) recalibration (weekly) | `proj/availability.py` | Beta posteriors stored; Brier vs the static table reported |

### Phase 3: the why engine (Nov 3 – Dec 5)

| ID | task | files | acceptance |
|---|---|---|---|
| T3.1 ★ O | xFG model + per-shot scoring + m̂ | `proj/xfg.py` | §3.2 targets; leakage audit test |
| T3.2 ★ O | post-game decomposition (all stats, FP) | `proj/postgame.py` | exact sum to 1e-9; M̃ test; zero-safe |
| T3.3 ∥ O | pre-game Shapley attribution | `proj/attribution.py` | exact sum; ≤ 10 s per slate |
| T3.4 S | why facts + writer prompt + fact-check extension | `proj/facts.py`, `brief/prompts/player_why.md`, `brief/factcheck_why.py` | 50 sampled games: 0 unsupported numbers, 0 sign violations after retry |
| T3.5 ∥ S | web: Last Night beats, game-page waterfall, `/player/<id>` | `templates/player.html`, `web/charts.py` (waterfall fn only) | screenshots; fallbacks quote only facts |

A post-game why v0 (minutes / volume / making against his FG%) can ship as soon as T0.3 and T1.2 exist, before xFG.

### Phase 4: DFS GPP + MLOps (Nov 17 – Dec 12)

| ID | task | files | acceptance |
|---|---|---|---|
| T4.1 O | GPP portfolio (candidates, field proxy, greedy P(≥ t), overlap/exposure) | `proj/dfs/portfolio.py` | ≤ 90 s for 20 lineups; exposure caps exact; simulated P(≥ t) of the portfolio > best single lineup |
| T4.2 ∥ S | DFS page full + CSV export | `templates/dfs.html`, `proj/dfs/export.py` | CSVs import cleanly into DK/FD templates (manual check) |
| T4.3 ∥ S | MLOps: registry specs, `player_metric_snapshots`, alerts, `/health` Players tab | `mlops/player_monitor.py`, `web/health.py` (tab only) | replay over 2025-26 produces snapshots; alerts dedupe |
| T4.4 ∥ S | weekly props benchmark + DFS backtest report | `proj/reports.py` | clustered SE; random-lineup percentiles |

**Critical path:** T0.0 → T0.2 → T0.3 → T2.1 → (T2.5, T2.6) → T2.7 → T3.2 → T3.4. xFG (T3.1) can start once T0.2 is done, in parallel with phase 2. **Dependencies:** T1.x needs T0.0, T0.6 and T0.7. T2.2 and T3.2 need T0.3. T2.4 needs T0.5. Keep at most 3 agents at once on geekom1, and run heavy fits on the Mac.

## 8. Risks and open questions

**Risks.**
- **Licensing.** NBA.com, ESPN, Action Network and DK/FD data is personal-use only. A paid launch needs licensed feeds (balldontlie, Sportradar), and the DFS optimizer may raise gambling-adjacent legal questions by state.
- **Minutes and late scratches.** The last pregame run is at 22:00, but scratches happen right before tip. Proposal: add game-relative runs at T−60 and T−20 min for injuries and props.
- **No historical availability before December 2025.** Backtests use the `avail` proxy and overstate accuracy. The honest numbers come from the T0.1 slice and the live season. Say so on `/health`.
- **Props semantics are unverified,** and DK props 404 after the game: a missed pre-tip snapshot is gone for good.
- **Sim mis-specification.** Fat tails in margins, and the OREB↔FGA feedback is ignored in v1. Guarded by the coverage and PIT alerts.
- **Small effects adding noise** (defender, referees, position defense). Each must show a measured gain, or it ships switched off.
- **Early season** (5–10 games): rates lean on priors. Show "small sample" and widen intervals; don't hide it.
- **The LLM over-reading the why.** Thresholds and sign checks in the fact-checker, with a template fallback.

**Open questions (with a proposed answer).**
1. Should the environment be the market game model or our opinion model? Proposal: market, with opinion as a shadow; compare props log-loss after 4 weeks.
2. Should the ceiling be p75 (industry habit) or p90? Proposal: show p25–p75 as the range and label p90 as the "ceiling".
3. Which props to show? Proposal: pts/reb/ast main lines only in v1; 3PM and PRA later.
4. DK rules: does a triple-double also earn the DD bonus? Is the DK player id stable across slates? Answer from captured `rules_json` and salary files (T0.7).
5. Is there a historical props or salary source we're allowed to use? T0.9 decides; otherwise all benchmarks are live from 2026-10-20 (≈350 props per night gives an SE of about 0.001 log-loss within a month, before clustering).
6. Contest results and cash lines are paid data. Until then, DFS quality is judged against random and naive-optimizer lineups only.
