# X factor: are our upper tails honest enough to flag big nights?

*2026-10-21. Owner request: flag low-popularity bench and role players with a real chance of a big
night. Opening-night examples: Andre Drummond 43 DK at 0.3 % estimated popularity, Javonte
Green 26.5 at 0.4 %, Jared McCain 29.5 at about 2 %. Duncan Robinson scored 29.5 but had 19 %
estimated popularity (Medium), so he would not qualify. Script: `research/xfactor_calibration.py`.
Product code: `src/courtside/proj/dfs/xfactor.py`.*

## 0. Verdict

- **The upper tails are good enough after a small correction, so the tag is built.**
- **Direct check.** We measured P(DK ≥ 6 × salary/1000) on 117k RotoGuru player-games
  (2015-16 to 2020-21). In the range that matters (10–30 %), the raw chance is close to
  calibrated. It beats a salary-band baseline in every season: Brier skill 3.0–5.1 %, with each
  season's difference at least 7 standard errors.
- **What needed fixing.** The raw sims overstate the top of the range. A predicted 30–40 % came in
  at 31 %, and a predicted 40 %+ (mean 48 %) came in at 42 %.
- **The fix.** A per-tier logistic map, fit only on earlier seasons, corrects this. On later
  seasons it gives 33.8 % predicted vs 33.7 % observed, and 46.8 % vs 46.0 %.
- **What the tag picks.** We applied the tag's chance rule walk-forward, without the popularity
  filter, which has no history. It picks players who reach 6× in **40–47 %** of their games
  (predicted 41–47 %). The salary-band rate for those same players is **24–26 %**.
- **Definition (`xf-v1`).** An X factor is a player who meets all of these:
  - tier bench or role;
  - P(play) ≥ 0.85;
  - popularity proxy band **Low**;
  - **calibrated chance of DK ≥ 6 × salary/1000 of at least 1 in 3**;
  - among the **top 5 per slate** by that chance.
- **Tag counts.**
  - 2026-10-20 (the opener): 0 tags. No candidate reached 1 in 3.
  - 2026-10-21 main slate (11 games, a manual v0.3+p2 run): 18 players pass the filters and the
    5 highest are tagged.
- **Honest limits.**
  - The opener's three examples would **not** have been tagged. Our sims gave Drummond 3 %,
    Green 5–6 % and McCain 15 %. Drummond and Green were minutes and P(play) misses (Drummond
    projected at 8.7 minutes if he played, actual 24), not tail misses.
  - The popularity proxy is uncalibrated. On big slates most players are "Low", and players we
    project as strong for their salary are probably more popular in reality than the proxy says.

## 1. Data: which caches are which arm

The walk-forward backtest caches on the Mac scratch box (`~/Developer/courtside-compute/`) are
identified by their driver logs.

| cache | arm | protocol | seasons |
|---|---|---|---|
| `cx/data/reports/backtest/cache/avail/<S>/v1-16c46b8db0, c19b1a7e03, df4e9671e4, b2065d96b8` | `cx/run.sh on` = context ON = **v0.3** | avail | 2019-20, 2020-21, 2023-24, 2024-25 |
| `cx/.../v1-281195b8b4, 01dd6ad338, 8c64bec897, b3440c614a` | context OFF = v0.2 | avail | same |
| `p2/.../live/2025-26/v1-1888940222`, `live10/.../v1-bab3326db9` | **v0.3+p2 (live)** | live (lock−45), live10 (lock−10) | 2025-26 slice (Dec 22 – Apr 12) |
| `p2/.../v1-3cd421b06b`, `v1-96e9869e88` | v0.3 | live, live10 | same |
| `t24/...` | v0.2/v0.3/v0.4 T2.4 lanes, superseded by p2 | | |
| **new** `xf/data/reports/backtest/cache/avail/<S>/` | **v0.3+p2 code** (current repo), `--tails` | avail, player level | 2015-16 … 2020-21 |

On the dates where they overlap, the new runs give means identical to the cx ON caches. The
avail protocol does not read P(play) v2, so v0.3 and v0.3+p2 coincide there.

**Harness option (new).** `courtside proj backtest-dfs --tails` (or `backtest.run(tails=True)`)
adds these columns per player: `q95`, `q99`, and on RotoGuru rows `p_ge_5x/6x/7x` =
P(DK ≥ k·salary/1000) from the sims.
- `Settings.tails` is dropped from the config hash when it is off, so existing caches and keys
  are unchanged. `tests/test_proj_backtest_live.py` pins this.
- Runtime on the Mac: 3 minutes per season with existing bundles, about 7 minutes with a bundle
  fit. All six seasons took about 25 minutes on 2 single-threaded processes.

## 2. Upper-tail PIT reliability

The randomised PIT is stored by the harness. Each cell is the share (%) of player-games with
PIT above the threshold, ± SE clustered by date. Rows are players with P(play) ≥ 0.85 (expected
to play).

| source (arm) | n | > .80 (20) | > .90 (10) | > .95 (5) | > .98 (2) | > .99 (1) | < .10 (10) |
|---|---|---|---|---|---|---|---|
| avail 2015-16 | 20,841 | 18.9 ± .3 | 8.6 ± .2 | 3.9 ± .2 | 1.6 ± .1 | 0.85 ± .10 | 10.1 |
| avail 2016-17 | 20,796 | 18.3 ± .3 | 8.0 ± .2 | 3.4 ± .1 | 1.3 ± .1 | 0.66 ± .09 | 9.7 |
| avail 2017-18 | 20,588 | 18.7 ± .3 | 8.7 ± .2 | 4.0 ± .1 | 1.5 ± .1 | 0.80 ± .08 | 9.8 |
| avail 2018-19 | 20,304 | 19.4 ± .2 | 9.1 ± .2 | 4.3 ± .1 | 1.7 ± .1 | 0.73 ± .06 | 9.7 |
| avail 2019-20 | 17,206 | 19.2 ± .3 | 8.9 ± .2 | 4.3 ± .2 | 1.5 ± .1 | 0.78 ± .09 | 10.2 |
| avail 2020-21 | 17,499 | 18.6 ± .3 | 8.6 ± .2 | 3.7 ± .2 | 1.4 ± .1 | 0.66 ± .09 | 10.0 |
| avail 2023-24 | 19,925 | 19.6 ± .3 | 9.2 ± .2 | 4.4 ± .2 | 1.6 ± .1 | 0.85 ± .06 | 10.3 |
| avail 2024-25 | 19,915 | 18.9 ± .3 | 8.7 ± .2 | 4.0 ± .2 | 1.4 ± .1 | 0.67 ± .06 | 10.2 |
| **live 2025-26 (v0.3+p2, lock−45)** | 13,182 | 17.9 ± .4 | 7.8 ± .2 | 3.5 ± .2 | 1.3 ± .1 | 0.58 ± .07 | 9.4 |
| live10 2025-26 | 13,427 | 18.1 ± .4 | 7.9 ± .2 | 3.5 ± .1 | 1.3 ± .1 | 0.62 ± .07 | 9.4 |

The same thresholds by tier and expected minutes if he plays, pooled over all the caches above:

| tier · minutes | n | > .80 | > .90 | > .95 | > .98 | > .99 |
|---|---|---|---|---|---|---|
| bench 8–16 | 10,022 | 21.7 ± .4 | 9.9 ± .3 | 4.4 ± .2 | 1.4 ± .1 | 0.54 ± .08 |
| bench 16–24 | 26,241 | 19.1 ± .3 | 7.9 ± .2 | 3.1 ± .1 | 0.91 ± .06 | 0.35 ± .04 |
| role 16–24 | 30,863 | 19.8 ± .2 | 9.1 ± .2 | 4.1 ± .1 | 1.5 ± .1 | 0.76 ± .06 |
| role 24–32 | 63,436 | 18.1 ± .2 | 8.2 ± .1 | 3.8 ± .1 | 1.5 ± .1 | 0.72 ± .04 |
| role 32+ | 11,375 | 18.4 ± .4 | 9.3 ± .3 | 4.5 ± .2 | 2.0 ± .1 | 1.14 ± .11 |
| star 24–32 | 16,615 | 17.7 ± .3 | 8.0 ± .2 | 3.8 ± .2 | 1.4 ± .1 | 0.69 ± .07 |
| star 32+ | 25,114 | 19.0 ± .3 | 9.0 ± .2 | 4.5 ± .1 | 1.9 ± .1 | 1.04 ± .07 |

By salary band (RotoGuru seasons only), the share above .95 is 3.6–4.4 % in every band. The
.99 share is 0.61 % below $4k and 0.77–0.96 % above.

What the PIT says:

- **The upper tails are too wide, not too thin.** For players expected to play, the top 5 % of
  our distribution is reached 3.4–4.4 % of the time, and the top 1 % 0.6–0.85 % of the time.
- **It is worst for bench players.** With 16–24 minutes, a bench player reaches his q99 0.35 %
  of the time against an expected 1 %. A raw "q99 ceiling" for a bench player is about 3×
  too generous.
- **It holds in every season** (2015-16 to 2025-26).
- **The live arm is a little wider than avail** (3.5 % vs about 4 % above .95). That is the
  live path's extra availability and minutes uncertainty.
- **The bottom tail is fine.** The share below .10 sits near 10 %. Only bench players with 8–16
  minutes are off, at 13.8 %, because DNPs are more common than the model allows.
- **The whole population looks calibrated** (≈ 5.0 / 1.0 % with every row in). That is only
  because players with P(play) < 0.85 have tails that are too light, which offsets the rest.
  Judge by subgroup, not the total.

## 3. Direct big-night check: P(DK ≥ k × salary/1000)

The population is the RotoGuru seasons with P(play) ≥ 0.85 (salaries exist only for RotoGuru
seasons; 2021-26 have none). The salary baseline is the historical big-night frequency by $500
salary bin, fit on the seasons before S. In 2015-16 nothing earlier exists, so the baseline
there uses the other seasons and that row is shape only. Brier difference = salary − ours, SE
clustered by date.

| k | season | base rate | mean pred | Brier ours | Brier salary | skill | diff ± SE | Brier, stretched (fit < S) |
|---|---|---|---|---|---|---|---|---|
| 6 | 2015-16* | 23.3 | 24.5 | .1692 | .1783 | 5.1 % | .0091 ± .0011 | – |
| 6 | 2016-17 | 25.1 | 27.1 | .1828 | .1885 | 3.0 % | .0057 ± .0007 | .1819 |
| 6 | 2017-18 | 21.5 | 23.2 | .1637 | .1690 | 3.2 % | .0053 ± .0006 | .1629 |
| 6 | 2018-19 | 25.7 | 26.7 | .1835 | .1909 | 3.8 % | .0073 ± .0008 | .1832 |
| 6 | 2019-20 | 25.7 | 26.6 | .1843 | .1914 | 3.7 % | .0071 ± .0007 | .1838 |
| 6 | 2020-21 | 22.7 | 24.6 | .1673 | .1757 | 4.8 % | .0085 ± .0010 | .1666 |
| 5 | 2016-21 | 39–44 | 39–43 | | | 3.7–5.3 % | 9–13 SE | |
| 7 | 2016-21 | 10.8–13.6 | 12.6–15.7 | | | 2.1–4.8 % | 4–7 SE | |

For bench and role players with 8–24 minutes, skill vs salary is 2.9–6.1 % at 6× and 2.4–4.6 %
at 7×.

**Reliability at k = 6**, pooled over 2016-17 to 2020-21 (each season's stretch is fit on the
seasons before it). Cells are predicted → observed, in %.

| predicted bin | n raw | raw | n stretched | stretched |
|---|---|---|---|---|
| 0–2 % | 180 | 0.4 → 13.9 | 142 | 0.3 → 16.2 |
| 2–5 % | 418 | 3.9 → 4.3 | 171 | 3.9 → 4.1 |
| 5–10 % | 3,849 | 8.1 → 7.7 | 2,574 | 8.3 → 6.7 |
| 10–15 % | 10,033 | 12.8 → 13.9 | 9,846 | 12.9 → 13.0 |
| 15–20 % | 16,680 | 17.6 → 18.0 | 19,669 | 17.7 → 17.8 |
| 20–30 % | 36,524 | 24.8 → 23.7 | 43,103 | 24.7 → 24.7 |
| 30–40 % | 19,650 | 34.2 → 30.9 | 16,379 | 33.8 → 33.7 |
| 40 %+ | 9,059 | 48.3 → 41.7 | 4,509 | 46.8 → 46.0 |

What the reliability table says:

- **The overstatement is at the top end, as the PIT predicted.** It is also stronger at 7×,
  where a predicted 51 % came in at 44 %.
- **The 0–2 % bin is the "unknown unknowns" bin**, and it is the opener's Drummond case. It is
  tiny (about 30 player-games a season), but when our sims say a playing player has almost no
  chance, about 15 % of them get there anyway. The usual cause is minutes we did not see coming.
  The tag can't help here, because it only fires on high chances.

**The recalibration (live parameters).** The map is logit p′ = a + b·logit p, fit per model tier
on every RotoGuru season (2015-16 to 2020-21, k = 6). It is stored as `xfactor.STRETCH`:

| tier | a | b |
|---|---|---|
| bench | −0.290 | 0.827 |
| role | −0.289 | 0.779 |
| star | −0.319 | 0.772 |

Applied to raw chances, it maps 0.30 → 0.28, 0.40 → 0.35 and 0.60 → 0.51 for role players. The
walk-forward fits for 2017-21 are close to these values (role a −0.29…−0.28, b 0.77–0.78), so
the map is stable.

## 4. The tag rule, walk-forward (popularity filter not testable)

There is no ownership history, so this applies everything except the popularity band. The
population is bench and role players with P(play) ≥ 0.85, at k = 6, with the stretch fit on
the seasons before S. Each "day" is every game of the day, which is more than a DK main slate.

| season | chance ≥ 1 in 3: per day | predicted | observed ± SE | salary-band rate, same players | top 5/day: predicted | observed ± SE |
|---|---|---|---|---|---|---|
| 2016-17 | 18.8 | 40.0 | 37.9 ± 0.9 | 24.5 | 47.3 | 44.0 ± 1.9 |
| 2017-18 | 8.9 | 38.6 | 38.0 ± 1.1 | 25.7 | 41.2 | 40.4 ± 2.0 |
| 2018-19 | 15.1 | 38.9 | 40.2 ± 1.0 | 24.7 | 43.8 | 46.9 ± 2.0 |
| 2019-20 | 13.5 | 39.3 | 39.4 ± 1.0 | 24.9 | 43.7 | 41.9 ± 1.9 |
| 2020-21 | 12.8 | 40.7 | 40.6 ± 1.5 | 24.8 | 45.2 | 44.1 ± 2.1 |

How to read this table:

- **The predicted chance is honest for the tagged set.** The worst gap is −3.3 points
  (2016-17, about 1.7 SE).
- **The tagged players reach a big night 1.6–1.9× as often as their salaries imply.**
- **At 7× with chance ≥ 0.25** (tried as an alternative), the stretched predictions run 0–6
  points high in 2016-18 and 0–3 points low in 2018-21. That still means 2–3× the salary rate,
  but with fewer than 5 players a day and noisier calibration.

## 5. Opening night (2026-10-20) and 10-21, from the live runs

These checks are read-only, using the prod DB and the run npz files. The opener ran on
`pathA-v0.2`. The P(play) v2 re-projection on the scratch box (`p2/out/opener_2026-10-20_lock45.parquet`)
gives the same P(play) and minutes for these players.

| player | salary | actual DK | run | P(play) | mean | raw P(≥6×) | calibrated | popularity (proxy) |
|---|---|---|---|---|---|---|---|---|
| Andre Drummond | 4,500 | 43.0 | lock10 | 0.42 | 4.0 | 2.4 % | 3.4 % ("1 in 30") | 0.35 % Low |
| Javonte Green | 3,300 | 26.5 | lock10 | 0.31 | 3.6 | 4.2 % | 5.3 % ("1 in 20") | 0.40 % Low |
| Jared McCain | 4,300 | 29.5 | lock10 | 0.85 | 14.8 | 13.0 % | 14.6 % ("1 in 6") | 1.65 % Low |
| Duncan Robinson | 4,600 | 29.5 | lock10 | 0.98 | 21.7 | 25.1 % | 24.2 % ("1 in 4") | 19.4 % Med |
| Paul George | 6,300 | 60.75 | lock10 | 0.98 | 32.3 | 32.7 % | 29.4 % ("1 in 3") | 32.6 % High |

Raw P(≥7×) for the same players: Drummond 1.6 %, Green 2.5 %, McCain 6.9 %, Robinson 13.6 %. In
lock45 the numbers are the same within a point.

Tags per slate under the shipped rule:

| slate | games / players | pass the filters | tagged |
|---|---|---|---|
| 10-20 main 154575, lock45 and lock10 | 96 players | 0 | 0 |
| 10-20 main 154827, lock45 and lock10 | 64 players | 0 | 0 |
| 10-21 main 155078, manual run `0a3afd3ff557` (v0.3+p2, 11 games) | 329 players | 18 | 5 |

The five tagged on 10-21 (calibrated chance; popularity proxy):
- Kris Dunn: 1 in 2 (0.52); 6.1 %
- Moussa Diabate: 0.51; 6.3 %
- Brook Lopez: 0.50; 3.7 %
- Sion James: 0.47; 5.2 %
- Gui Santos: 0.44; 3.3 %

**Caveat.** Dunn and Diabate are also in tonight's cash lineup. Early in the season our
projections and DK's salaries disagree, and players we project well above their salary are
exactly the ones optimizers (and the public) will pick. The proxy's 3–6 % is probably low for
them. The tag says "low *estimated* popularity", and the band stays a proxy until ≥ 10 uploaded
standings allow a refit (`field.fit_from_uploads`).

## 6. Design decisions

- **Fixed multiple (6×), not the slate-relative boom mark of §2.1.**
  - The fixed multiple is what the history could calibrate.
  - The slate-relative mark v* would need a simulated top-1 % per slate, and the harness has no
    field proxy.
  - The realised hindsight optimum per $1k (RotoGuru, all games of a day) has a median of 8.0
    and an IQR of 7.6–8.3, and grows with slate size (6.5–7.2 for ≤ 3 games, 8.2–8.5 for 10+; 2019-21). So a
    slate-relative mark would move the threshold by only about ±0.5×.
  - 6× matches the owner's examples: Robinson's 6.4× and McCain's 6.9× count as big nights.
- **Chance ≥ 1 in 3 plus a cap of 5 per slate.** A bare threshold gives 9–19 players a day
  historically, and 18 on tonight's 11-game slate. The cap keeps it to a handful, and the
  capped sets are still calibrated (§4).
- **P(play) ≥ 0.85** is the population the calibration was measured on.
- **Bench or role only.** That is the owner's intent, and stars are rarely Low.
- **Computed at read time** from the run's npz file (`xfactor.run_table`), cached per run id and
  sim-file mtime.
  - No migration and no runner change.
  - Works for every stored run, including the opener.
  - Costs about 0.1 s cold and 0.02 s warm on the Mac (checked on a copy of the prod DB).
- **Chance text is banded** ("1 in N", N from 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50). The chance
  is unconditional: it already includes the chance he does not play.
- **Contest builder.** Default behaviour is unchanged. The build summary gets
  `summary["xfactor"] = {n_players, lineups_with, entries, players_used}`, and the CLI prints
  "X factor players (estimate; reporting only): k of N lineups include at least one".

## 7. Open items

- **No salaries for 2021-26,** so the direct 6× check stops at 2020-21. The live slice's PIT (§2)
  is a little wider than avail's, so live chances may still run slightly high after the stretch.
  Grade it: from `player_projection_grades` (actual_dk, salary) plus the run npz, compare
  predicted vs observed 6× for the tagged and all players after about 3 weeks.
- **The popularity proxy is uncalibrated.** Revisit "Low" (< 10 %) once uploaded standings
  exist. A stricter cut (< 5 %) may be the honest one.
- **Bench players' q99 is about 3× too generous** (§2). Any "ceiling" number shown elsewhere
  should use the same stretch, or a per-tier tail shrink in the sim (an item for copula v1
  variance work).
- **The 0–2 % bin misses about 15 % of the time.** These are minutes surprises like Drummond.
  The return-minutes and replacement-minutes work is what helps here, not the tag.

## 8. Reproduce

```bash
# scratch box (Mac): code rsynced to ~/Developer/courtside-compute/xf/code, DB cloned from cx/data
~/Developer/courtside-compute/xf/run.sh harness 2019-20          # avail, --tails, no lineups
# geekom1 (caches rsynced to var/tmp/xf/)
uv run python research/xfactor_calibration.py pit var/tmp/xf/bt/avail_2024-25 ...   # §2
uv run python research/xfactor_calibration.py bignight var/tmp/xf/tails/<S>/* ...   # §3-4
# Mac prod, read-only
COURTSIDE_DATA=~/Developer/NBA/var python research/xfactor_calibration.py opener --day 2026-10-20
```

## Addendum (2026-10-21, after review): top value plays are left out

On the 10-21 main slate (run `0a3afd3ff557`) the first rule tagged Dunn, Diabate, Lopez, Sion James
and Gui Santos: slate value ranks (mean DK per $1k, P(play) ≥ 0.5) 1, 5, 3, 7 and 11, all with a
popularity proxy of 3–6 %. Those are the players every projection source finds, so real
ownership is likely far above the proxy (which is hand-set and uncalibrated until ≥ 10 uploaded
standings). The tag now also requires value rank > 10 (`xfactor.VALUE_TOP_EXCLUDE`); the table
keeps `value_rank` and `mean` per player. Revisit the cut with real `%Drafted` from uploads.
