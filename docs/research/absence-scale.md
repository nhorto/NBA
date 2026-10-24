# Are our injury/absence values 2× the market's? (IMPROVEMENTS G4)

*2026-10-23. Switch `COURTSIDE_ABSENCE_SCALE` (default OFF, version tag `+as`), code in
`src/courtside/model/absence_scale.py` and a 3-line hook in `predict.price_day`. Scripts:
`research/as_eval.py` (checks + walk-forward, light), `research/as_replay.py` (Mac scratch
`~/Developer/courtside-compute/as/`: per-player report state + compact engine state at the G1
cutoffs), `research/as_off_identity.py` (OFF byte check), `research/as_star_out_bump.py`
(player side). Outputs: `var/tmp/as/` on geekom1 (`eval_base.json`, `eval_maps.json`,
`days/`, `player/`). Honest numbers only; see MODEL.md for the model.*

## 0. Summary

- **No. The 0.45 slope is a timing effect, not an overstatement.** Against outcomes our
  absence values are, if anything, slightly too *small*, and the closing line prices them at
  about 0.9× ours:
  - margin − our no-absence margin, on our absence component A at the tip: **k = 1.29
    (SE 0.17)**, n 899 (2025-26, official report archive);
  - close − our no-absence margin on A: **m = 0.92 (0.05)**;
  - the open prices the absences already on the report at 0.71 (0.05) **and 0.36 (0.09) of the
    news that comes later** (it anticipates it); the close prices the news at 0.88 (0.11). The
    open→close move per point of our news is therefore ≈ 0.88 − 0.36 ≈ 0.5: G1's 0.45.
- **The two extra checks agree.** (a) Outcomes vs the open on our news: slope 0.69 (0.27), between
  "market right" (0.45) and "we are right" (1); given the line move our news adds nothing
  (0.06, SE 0.25). (b) Placebos are null: news permuted within date 0.06 (0.07); the same home
  team's news from another date 0.00 (0.04); pre-open news (the day before → the evening before)
  −0.20 (0.23).
- **P(play) is not the cause (B4).** A Questionable player on the evening report plays about
  50 % of the time, which is the hand map's 0.5. Rebuilding the lineups with the DFS side's
  calibrated report cells (walk-forward or shipped) or with `players.py`'s Doubtful 0.15 leaves
  every slope where it was, and the best score change is −0.00015 LL (0.4 SE) at a noon run
  (§4). The low slope sits in news that is not a Questionable call, much of it on the second
  night of back-to-backs, where the evening-before state is the previous game's report: noise
  in the measured news, not in the priced absence.
- **Walk-forward scale: nothing to ship.** The fitted s runs to the grid cap (1.5) and is
  flat: opinion LL +0.0005 (2-way SE 0.0022), `gbm_market` fed the close −0.0002 (0.0006).
  Halving (s = 0.5, what 0.45 suggests) is worse: opinion margin CRPS +0.056 (0.027), opinion
  LL +0.0013 (0.0029), `gbm_market` fed the open +0.0009 (0.0008). **Recommendation: keep the
  switch OFF and do not shrink.** It stays as a re-test hook (`COURTSIDE_ABSENCE_SCALE=<s>`).
- **Player side: the star-out bump is about right.** Top-6 teammates of a star ruled out
  realise 1.01× (SE 0.04) the projected DK gain in the 2023-25 harness and 0.84× (0.07) on the
  2025-26 live slice (§6); not 0.5×.

## 1. Data and method

- **Games.** 2025-26 games covered by the official report archive (22 Dec 2025 → 13 Jun 2026,
  regular season, play-in, playoffs): 899 with a usable report at the tip, 879 also at the open
  proxy and with an ESPN open and close. These are the G1 per-day dumps
  (`var/tmp/g1/days`: the production pair fit on games before each day, the day's feature rows,
  production injury lineups at 8 cutoffs). Results from `dataset.load_games`.
- **Absence component A** (home-margin points): `gbm_opinion` margin with the injury-report
  lineups minus with both teams at full recent-roster strength (ps + abs_loss of the feature
  row), everything else fixed. This is the number the Brief's injury effects are built from.
- **News Δ** (G1's): opinion margin with the tip lineups minus with the guarded lineups at the
  pre-registered open proxy d0_03 (03:00Z, the evening before).
- **Lineups under other P(play) maps** come from `research/as_replay.py`: per cutoff, the
  matched report records and a compact `PlayerState`, so `expected_lineup` is rebuilt offline.
  The base map rebuilt this way equals the G1 lineups to 1e-15.
- **SEs**: slopes have date-clustered sandwich SEs; paired score differences have date and
  two-way (date × home team, `paired_cluster2`) SEs.
- **"Seasons < S".** The report archive has one partial season, so the walk-forward scale is
  fit on *earlier days* (expanding window, 150-game minimum, s = 1 before), not earlier seasons.
  The per-season "no season worse" rule cannot be checked.

## 2. The three slope checks

`b` is the slope through the origin, `b (int)` with an intercept; `move` = close − open home
margin line; `r_open` = margin − open; `r_close` = margin − close.

| # | check | n | b (SE) | b with intercept (SE) | R² | reads as |
|---|---|---|---|---|---|---|
| 1 | move ~ Δ (G1, reproduced) | 879 | **0.45 (0.05)** | 0.45 (0.05) | 0.125 | |
| 2 | (a) r_open ~ Δ | 879 | 0.69 (0.27) | 0.68 (0.27) | 0.007 | we right: 1; market right: 0.45 |
| 3 | (a) r_close ~ Δ | 879 | 0.24 (0.27) | 0.24 (0.27) | 0.001 | we right: 0.55; market right: 0 |
| 4 | (a) r_open ~ move + Δ: Δ coef (move coef) | 879 | 0.05 (0.25) | (1.41 (0.26)) | | the move carries the news |
| 5 | (a) level: margin − op(full) ~ A (k) | 899 | **1.29 (0.17)** | 1.29 (0.18) | 0.050 | we right: 1 |
| 6 | (a) level: close − op(full) ~ A (m) | 899 | **0.92 (0.04)** | 0.93 (0.05) | 0.425 | the close's scale of our A |
| 7 | (a) level: margin − close ~ A (k − m) | 899 | 0.37 (0.17) | 0.37 (0.18) | 0.004 | market right: 0 |
| 8 | level: open − op(full) ~ A | 899 | 0.60 (0.04) | 0.59 (0.04) | 0.219 | the open's scale |
| 9 | (b) placebo: move ~ Δ permuted within date (2,000 draws) | 879 | 0.06 (0.07) | | | p(\|b\| ≥ 0.45) < 0.0005 |
| 10 | (b) placebo: r_open ~ Δ permuted within date | 879 | −0.00 (0.26) | | | |
| 11 | (b) placebo: move ~ same home team's Δ from another date (200 draws) | 879 | −0.00 (0.04) | | | randomised timing |
| 12 | (b) placebo: r_open ~ same home team's Δ, another date | 879 | −0.03 (0.28) | | | |
| 13 | (b) pre-open news: move ~ Δ(d1_17 → d0_03) | 879 | −0.21 (0.23) | −0.20 (0.23) | 0.002 | the open has it |
| 14 | (b) pre-open news: r_open ~ Δ(d1_17 → d0_03) | 879 | 0.22 (1.02) | 0.21 (1.02) | 0.000 | |

Reading:
- Row 1 reproduces G1 (0.447).
- Rows 2–4 (outcomes): if our news values were 2× too big, `r_open` on Δ would be ~0.45 and
  `r_close` ~0. They are 0.69 (0.27) and 0.24 (0.27): too noisy alone (news Δ has sd 1.8 pts
  against a 13-pt margin sd) to pick a side. Given the move, Δ adds nothing (row 4), and the
  move itself is under-sized against outcomes (1.41, SE 0.26).
- Rows 5–8 (levels) have far more power, since standing absences vary much more than news.
  Outcomes scale A by 1.29 (0.17); the close by 0.92 (0.05); the open by 0.60 (0.04).
  Regular season only: k 1.32 (0.17), m 0.92 (0.05). With the no-absence margin as a free
  regressor: k 1.27 (0.18), m 0.93 (0.05).
- Rows 9–14 (placebos): all null, so the 0.45 is not an artefact of team-level or date-level
  structure, and news from before the open is already priced.

Decomposition (same 879 games, both regressors at once):

| y − our no-absence margin | absences on the report at the open proxy | news after it |
|---|---|---|
| open | 0.71 (0.05) | 0.36 (0.09) |
| close | 0.95 (0.04) | 0.88 (0.11) |
| actual margin | 1.29 (0.21) | 1.22 (0.25) |

So the open already prices a third of the news we call "news" (e.g. a Questionable player the
market expects to sit), and by the close the market prices our values at about 0.9. The market
is not right *instead of* us: at the close it agrees with us, and the outcomes, if anything,
want a bit more.

## 3. Where the absence value comes from

- `injuries.availability` → `{pid: P(play)}` from `injuries.PLAY_PROB` (Out 0, Doubtful 0.10,
  Questionable 0.5, Day-To-Day 0.6, Probable 0.9); `production.lineup_strengths` →
  `PlayerState.expected_lineup` (P-weighted minutes, renormalised to 240) → lineup strength;
  `production.apply_overrides` replaces ps_home/ps_away and recomputes ps_diff and the
  abs_loss features (`model/features.py` defines them; training uses the leak-safe
  `avail_tight` oracle, so the heads learn the absence value from realised absences).
- `players.availability_from_injuries` (`STATUS_P_PLAY`, Doubtful 0.15) is only used by
  `gbm.py:726`'s injury-override path, not by `price_day`.

## 4. Is it P(play)? (B4)

Where P(play) enters: at the open proxy most news is a status resolving. Transitions between the
evening report and the tip, per player (guarded sides, 879 games): Questionable → Available
783 (87 % played), unlisted → Out 700, Questionable → Out 577, Doubtful → Out 326, Probable →
Available 310, Out → unlisted 237 (66 % played), Out → Available 199. A Questionable player on
the evening report ends up playing (783 × 0.87) / 1,360 ≈ **50 %**: the hand map's 0.5 is right
at that time. Doubtful players play ~4 % (vs 0.10), worth little.

By news type (move ~ Δ; r_open ~ Δ):

| news between d0_03 and the tip | games | move ~ Δ | r_open ~ Δ |
|---|---|---|---|
| only Questionable/Doubtful/Probable resolving | 359 | 0.63 (0.11) | 0.13 (0.77) |
| only new Out/Doubtful (unlisted or Available before) | 44 | 0.83 (0.09) | 0.48 (0.99) |
| mixed | 422 | 0.38 (0.05) | 0.79 (0.28) |
| any game with Questionable news | 672 | 0.47 (0.06) | 1.07 (0.40) |

The low slopes sit in games whose news is *not* a Questionable call. About 40 % of those
transitions are on teams that played the night before: with `INJ_GAMEDAY_GAME` off, the
evening-before state can be the previous game's report (Out → unlisted at the tip, 34 % of whom
then did not play). That is noise in Δ, which attenuates the slope; it is not an error in the
absence value priced at the tip.

The lineups rebuilt under other maps (scores paired against the base map rebuilt from the same
dumps; see the note below):

| P(play) map | run time | games changed | move ~ Δ | r_open ~ Δ | k | m | Δ opinion LL (date / 2-way SE) | Δ opinion CRPS margin (2-way SE) | Δ market (close) LL (2-way SE) |
|---|---|---|---|---|---|---|---|---|---|
| base (rebuilt) | tip | 0 | 0.45 (0.05) | 0.69 (0.27) | 1.31 (0.17) | 0.92 (0.04) | 0 | 0 | 0 |
| `players.py` (Doubtful 0.15) | tip | 0 | 0.45 (0.05) | 0.68 (0.27) | 1.31 (0.17) | 0.92 (0.04) | 0 | 0 | 0 |
| `players.py` (Doubtful 0.15) | noon (d0_17) | 198 | | | | | +0.00016 (0.00019 / 0.00019) | +0.0002 (0.0014) | −0.00000 (0.00003) |
| cells, walk-forward | tip | 9 | 0.44 (0.05) | 0.64 (0.27) | 1.31 (0.17) | 0.92 (0.04) | +0.00001 (0.00001 / 0.00001) | +0.0002 (0.0002) | +0.00001 (0.00000) |
| cells, walk-forward | noon (d0_17) | 635 | | | | | −0.00015 (0.00045 / 0.00041) | −0.0034 (0.0043) | −0.00005 (0.00008) |
| cells, shipped (in-sample) | tip | 9 | 0.44 (0.05) | 0.62 (0.27) | 1.31 (0.17) | 0.92 (0.04) | +0.00000 (0.00002 / 0.00003) | +0.0002 (0.0002) | +0.00001 (0.00001) |
| cells, shipped (in-sample) | noon (d0_17) | 694 | | | | | −0.00002 (0.00044 / 0.00047) | −0.0018 (0.0054) | +0.00003 (0.00012) |

- "cells" = `proj.availability.p_play_slate` at the same cutoff (status × hours-to-tip × reason
  × role), applied to listed, not-Out players whose slate status matches the priced one.
  Walk-forward = cells fit on regular-season days before the test month (hand map before
  January); shipped = the in-sample JSON. Lock−10 starter flags do not exist at these times.
- At the tip almost every status is resolved, so the map cannot matter; at noon the cells move
  635 games and gain −0.00015 LL (0.4 SE). **B4's fix is not what the 0.45 needs**, and its gain
  is noise, as the rung-2 ablation predicted (0.0014 LL for all of P(play)). B4 can still make
  the two tables agree for consistency; it is not shipped here.
- **Fragility note.** The base lineups rebuilt offline equal the G1 dumps to 5e-15, yet 45 of 899
  opinion margins differ, by up to 0.98 pts, because a LightGBM split sits on the last bit of a
  lineup strength (a full-roster strength equals a training value). Map comparisons are
  therefore paired with the rebuilt base, not the dumped one. Production is unaffected (one code
  path), but it shows how close to split thresholds the strength features can sit.

## 5. Walk-forward ON vs OFF

ON = the lineups scaled toward the roster by s (`lineup = roster − s·(roster − lineup)`);
OFF = s 1. `gbm_market` is fed the close (the replay convention) or the ESPN open (a stale
morning line). Run time T = the tip lineups, or 17:00Z (about noon ET). Paired ON − OFF; SEs
date / two-way. Close ML is the de-vigged closing moneyline on the same games.

| arm (T, s) | model | n | s last | LL OFF | LL ON | ON − OFF LL (date / 2-way) | ON − OFF Brier (2-way) | ON − OFF CRPS margin (date / 2-way) | close ML LL |
|---|---|---|---|---|---|---|---|---|---|
| tip, s walk-forward | gbm_opinion | 899 | 1.5 | 0.5789 | 0.5794 | +0.00050 (0.00192 / 0.00216) | +0.00048 (0.00095) | -0.0184 (0.0209 / 0.0220) | 0.5646 |
| tip, s walk-forward | gbm_market, close | 899 | 1.5 | 0.5673 | 0.5671 | -0.00020 (0.00055 / 0.00059) | +0.00000 (0.00025) | -0.0081 (0.0060 / 0.0060) | 0.5646 |
| tip, s walk-forward | gbm_market, open | 899 | 1.5 | 0.5757 | 0.5748 | -0.00084 (0.00054 / 0.00058) | -0.00025 (0.00025) | -0.0164 (0.0062 / 0.0062) | 0.5646 |
| tip, s = 0.5 | gbm_opinion | 899 | 0.5 | 0.5789 | 0.5802 | +0.00131 (0.00242 / 0.00285) | +0.00019 (0.00125) | +0.0565 (0.0267 / 0.0271) | 0.5646 |
| tip, s = 0.5 | gbm_market, close | 899 | 0.5 | 0.5673 | 0.5673 | +0.00004 (0.00068 / 0.00080) | -0.00003 (0.00032) | +0.0103 (0.0072 / 0.0073) | 0.5646 |
| tip, s = 0.5 | gbm_market, open | 899 | 0.5 | 0.5757 | 0.5766 | +0.00091 (0.00067 / 0.00077) | +0.00032 (0.00033) | +0.0209 (0.0072 / 0.0071) | 0.5646 |
| tip, s = 0.0 | gbm_opinion | 899 | 0.0 | 0.5789 | 0.5885 | +0.00963 (0.00489 / 0.00548) | +0.00360 (0.00245) | +0.2093 (0.0537 / 0.0537) | 0.5646 |
| tip, s = 0.0 | gbm_market, close | 899 | 0.0 | 0.5673 | 0.5677 | +0.00049 (0.00126 / 0.00147) | +0.00006 (0.00060) | +0.0251 (0.0129 / 0.0135) | 0.5646 |
| tip, s = 0.0 | gbm_market, open | 899 | 0.0 | 0.5757 | 0.5779 | +0.00219 (0.00124 / 0.00145) | +0.00076 (0.00063) | +0.0459 (0.0136 / 0.0138) | 0.5646 |
| d0_17, s walk-forward | gbm_opinion | 899 | 1.5 | 0.5801 | 0.5802 | +0.00010 (0.00163 / 0.00163) | +0.00033 (0.00069) | -0.0215 (0.0196 / 0.0181) | 0.5646 |
| d0_17, s walk-forward | gbm_market, close | 899 | 1.5 | 0.5668 | 0.5665 | -0.00022 (0.00049 / 0.00044) | +0.00001 (0.00019) | -0.0066 (0.0055 / 0.0046) | 0.5646 |
| d0_17, s walk-forward | gbm_market, open | 899 | 1.5 | 0.5761 | 0.5755 | -0.00054 (0.00049 / 0.00045) | -0.00012 (0.00020) | -0.0109 (0.0059 / 0.0050) | 0.5646 |
| d0_17, s = 0.5 | gbm_opinion | 899 | 0.5 | 0.5801 | 0.5818 | +0.00168 (0.00198 / 0.00193) | +0.00046 (0.00084) | +0.0514 (0.0221 / 0.0190) | 0.5646 |
| d0_17, s = 0.5 | gbm_market, close | 899 | 0.5 | 0.5668 | 0.5670 | +0.00027 (0.00054 / 0.00057) | +0.00004 (0.00024) | +0.0112 (0.0062 / 0.0058) | 0.5646 |
| d0_17, s = 0.5 | gbm_market, open | 899 | 0.5 | 0.5761 | 0.5767 | +0.00069 (0.00055 / 0.00056) | +0.00022 (0.00024) | +0.0168 (0.0066 / 0.0060) | 0.5646 |
| d0_17, s = 0.0 | gbm_opinion | 899 | 0.0 | 0.5801 | 0.5885 | +0.00840 (0.00428 / 0.00439) | +0.00323 (0.00195) | +0.1761 (0.0508 / 0.0492) | 0.5646 |
| d0_17, s = 0.0 | gbm_market, close | 899 | 0.0 | 0.5668 | 0.5677 | +0.00100 (0.00103 / 0.00104) | +0.00029 (0.00044) | +0.0273 (0.0117 / 0.0108) | 0.5646 |
| d0_17, s = 0.0 | gbm_market, open | 899 | 0.0 | 0.5761 | 0.5779 | +0.00181 (0.00107 / 0.00106) | +0.00064 (0.00046) | +0.0383 (0.0130 / 0.0117) | 0.5646 |

- The walk-forward s goes to the cap: outcomes want more absence value, not less. The gain is
  not significant for the opinion model; for `gbm_market` fed the **open** the margin CRPS gain
  is 2.6 SE, because the open under-prices absences (0.6) and the larger absence features
  compensate. That is G1's stale-line problem, and live lines are fresh (median 1 minute old).
- Halving is worse everywhere it is measurable, and s = 0 (ignore the report) costs the opinion
  model +0.0096 LL and +0.21 margin CRPS (3.9 SE).
- Brief "injury moves the odds": unchanged (OFF). ON scales each player's effect with s.

## 6. Player side: the DFS star-out bump (measure only)

`research/as_star_out_bump.py` on the existing `pathA-v0.3+mf+p2+rb` row frames (Mac
`courtside-compute/rb/out/rows_*`). Star = T2.7's rule (top 2 of the team by prior rolling-20
DK, ≥ 28 mpg); a star-out team-game counts only if the projection knew (harness: not
available; live: listed Out/Doubtful). Gain = the player's value in star-out games minus his
own star-in mean for the team-season; ratio = realised / projected; bootstrap SEs over
team-seasons.

| slice | tier | n | projected DK / min | realised DK / min | ratio DK (SE) | ratio min (SE) |
|---|---|---|---|---|---|---|
| 2023-25 | top-6 | 7187 | +4.38 / +3.62 | +4.41 / +3.49 | 1.01 (0.04) | 0.96 (0.03) |
| 2023-25 | star | 1110 | +1.70 / +0.05 | +2.30 / +0.34 | 1.35 (0.22) | n/a |
| 2023-25 | role | 4675 | +2.66 / +1.89 | +3.43 / +2.17 | 1.29 (0.09) | 1.15 (0.08) |
| 2023-25 | bench | 10073 | +3.55 / +3.70 | +3.51 / +3.81 | 0.99 (0.03) | 1.03 (0.02) |
| live 2025-26 | top-6 | 2272 | +4.38 / +3.34 | +3.68 / +2.57 | 0.84 (0.07) | 0.77 (0.07) |
| live 2025-26 | star | 299 | +1.41 / −0.09 | −0.39 / −1.20 | ≈ −0.3 (gap −1.8 ± 1.0 DK) | n/a |
| live 2025-26 | role | 1826 | +2.48 / +1.58 | +2.34 / +1.24 | 0.95 (0.12) | 0.78 (0.13) |
| live 2025-26 | bench | 2562 | +3.94 / +4.00 | +4.76 / +4.86 | 1.21 (0.05) | 1.21 (0.04) |
| live10 2025-26 | top-6 | 2364 | +4.47 / +3.36 | +3.85 / +2.67 | 0.86 (0.07) | 0.79 (0.06) |
| live10 2025-26 | star | 331 | +1.58 / −0.05 | −0.51 / −1.29 | ≈ −0.3 (gap −2.1 ± 0.9 DK) | n/a |

- Where the bump comes from: the minutes engine (`proj/minutes.py` `scenario_features`
  vacated minutes as GBM inputs; `project_arrays` next-man-up promotion, leftover spread and
  the pair-share term) plus a per-minute multiplier when a top-2 star is out
  (`proj/state.py` `fit_usage_bump` / `bump_for`: ×1.035 rotation, ×1.057 the other star;
  applied in `proj/copula.py`).
- The harness says the bump is right for top-6 (1.01) and *under*-credits role players (1.29).
  The 2025-26 live slice realises 0.84–0.86 for top-6 (about −0.65 DK, ~2 SE), and the
  shortfall is minutes going to the bench (bench 1.21), not per-minute scoring. The other star
  loses about 2 DK against the projection, matching T2.7's −1.9.
- So the game side's "half" does not carry over. At most, a 15–20 % trim of the top-6 minutes
  bump in favour of the bench, if 2026-27 confirms the live slice. No change made.
- Caveat: the frames are the `+mf+rb` arm, not the live `pathA-v0.3+p2`.

## 7. Implementation

- `model/absence_scale.py`: `ENV = "COURTSIDE_ABSENCE_SCALE"`, `DEFAULT = None` (OFF),
  `TAG = "+as"`, `SCALE = 1.0` (the value of `on`; any number sets s, clipped to [0, 2]).
- ON, `price_day` wraps the engine state in `ScaledState`, whose `expected_lineup` strength is
  `roster − s·(roster − strength)` (s = 1 returns the strength exactly). The priced lineups,
  `production.injury_effects` (Brief "injury moves the odds") and the G1 capture-time lineups
  all go through it, so they stay consistent. Both models' rows get `+as` and
  `inputs.absence_scale`. Days without a usable report are unchanged.
- **OFF is byte-identical.** `price_day` never calls the module OFF (a test monkeypatches every
  entry point to raise). Stripped rows of two synthetic worlds (no report; Out + Questionable +
  Doubtful) and of a real past day (2026-01-15, 9 games, official report priced, on the Mac scratch box) were dumped as JSON with the tree before the hook
  and with it, and matched with `cmp`. `tests/test_absence_scale.py` (8 tests): env parsing,
  OFF never enters, s = 1 equals OFF, lineup/margin/effect monotone in s with the exact
  interpolation, no-report days unchanged, the proxy.

## Reproduce

```
# Mac scratch (2 lanes, nice 10; read-only DB with schema 16 + game_date backfill)
run.sh code/research/as_replay.py --db <scratch-gd.db> --start 2025-12-22 --end 2026-03-15 --out out/days
run.sh code/research/as_replay.py --db <scratch-gd.db> --start 2026-03-16 --end 2026-06-30 --out out/days
# geekom1 (light; OMP_NUM_THREADS=2)
uv run python research/as_eval.py --g1 var/tmp/g1/days --db var/courtside.db --maps base \
    --cache var/tmp/as/cache --out var/tmp/as/eval_base.json
uv run python research/as_eval.py --g1 var/tmp/g1/days --db var/courtside.db \
    --as-days var/tmp/as/days --maps base,pp15,cells,cells_ship --cache var/tmp/as/cache \
    --out var/tmp/as/eval_maps.json
uv run python research/as_star_out_bump.py
```
