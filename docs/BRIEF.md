# Courtside Brief: JSON schema (version 1.1)

One document per NBA day, built by `courtside.brief.assemble.build_brief(conn, day)`.

- Stored in SQLite table `briefs(nba_day, brief_json, built_at, version)` and written to
  `$COURTSIDE_DATA/briefs/<day>.json`.
- Real examples (day 2026-02-20, built from real data): `tests/fixtures/brief/sample_brief_gbm.json`
  (schema 1.1, the production models with injuries priced: **build against this one**) and
  `tests/fixtures/brief/sample_brief.json` (schema 1.0, results-only `elo_sched`, kept as the
  "model does not price injuries" caveat case).
- **1.1** (additive over 1.0): production predictions are now the stacked models with injuries priced
  in. New keys: `meta.opinion_model`, `meta.model_uses_market`, `meta.availability_basis`;
  `tonight.games[i].opinion`, `.availability`, `edge.model`; factors of kind `model_factor`;
  `model_effect_margin` is filled; `scorecard.opinion`. Nothing was removed or renamed.
- CLI: `courtside brief build [--date D] [--no-writer]`, `courtside brief show [--date D] [--json]`.
- Built automatically at the end of `collect postgame` (for day + 1) and `collect pregame`
  (for today, after refreshing predictions, lines and injuries).

## The day model (read this first)

`build_brief(day)` is the **morning-of** Brief for `day` (US/Eastern NBA day):

| section | covers |
|---|---|
| `last_night` | games of `day - 1` (`meta.last_night_day`) |
| `team_pulse`, `player_pulse` | trends computed from games strictly before `day` |
| `injury_watch` | latest injury snapshots at or before the end of `day`, diffed against ~24 h earlier |
| `tonight` | games of `day`; win probability, margin and total come from the production models (see "Production models") |
| `scorecard` | stored pre-tip predictions with `nba_day < day`, season to date |

A Brief for `day` is rebuilt several times: after postgame (03:00/09:00), and after every pregame run
(lines and injuries move). The latest build wins; `meta.built_at` tells you which one you have.

## Stability rules

- `meta.schema_version` is `"MAJOR.MINOR"`. **Minor** bumps only add keys. **Major** bumps may rename or
  remove keys. A client must ignore unknown keys.
- Every key listed below is **always present**. Missing data is `null` (scalars/objects) or `[]` (lists);
  it is never an absent key, never `NaN`. Numbers are plain JSON numbers.
- Every `text` slot is `null` until the writer fills it. A client must render a good page with all of
  them `null` (use the structured fields) and show text where present. Text is plain, no markdown.
- Probabilities are in `[0, 1]`. Spreads follow the betting convention: **negative = favourite**
  (`home_spread: -4.5` means the home team is favoured by 4.5). `home_margin` is the opposite sign:
  positive = home wins by that much. Team codes are NBA tricodes (`BOS`, `NYK`, ...).
- Times are ISO-8601 UTC strings (`2026-02-20T22:00:00Z`); `day` fields are `YYYY-MM-DD` Eastern dates.
- Games with `tipoff_utc: null` come from the historical game log (past days); live days carry a tip-off.

## Top level

```
{ "meta": {...}, "headline": {"text": null}, "last_night": {...}, "team_pulse": {...},
  "player_pulse": {...}, "injury_watch": {...}, "tonight": {...}, "scorecard": {...} }
```

## `meta`

| key | type | notes |
|---|---|---|
| `schema_version` | string | `"1.1"` |
| `day` | string | the Brief's day |
| `last_night_day` | string | `day - 1` |
| `season` | string | e.g. `"2025-26"` |
| `built_at` | string | UTC build time |
| `model` | string | headline prediction model (`gbm_market`) |
| `model_uses_injuries` | bool | `true` for the production models: the injury report is priced into probability, margin and total (via expected lineups). `false` for results-only models (`elo_sched`): injuries are shown as factors but are **not** priced. Show a caveat when `false`. |
| `opinion_model` | string\|null | the independent model behind "our number" and `edge` (`gbm_opinion`: no market input); null for results-only models |
| `model_uses_market` | bool | the headline model sees the market line, so it will sit close to it by design; use `opinion` for "our number vs the market" |
| `availability_basis` | object | team-sides of tonight's slate by how availability was determined: `injury_report` (a usable official/ESPN report), `regulars_played` (backfilled/demo days only: the rotation regulars who actually played, which is leak-safe but is not an injury report), `full_roster` (no report: everyone on the recent roster assumed available) |
| `data_freshness` | object | `team_games_through`, `player_games_through` (latest game date used by trends), `odds_latest_captured_at`, `injuries_latest_captured_at`, `predictions_latest_created_at`, `last_night_games`, `last_night_games_with_facts` |
| `notes` | string[] | human-readable data caveats (missing facts/predictions/injuries, stale game logs...). Surface them in a small "data notes" footer. |
| `text_filled` | bool | `true` when the writer produced the headline |
| `path` | string | file path relative to the data dir |

## `headline`

`{"text": null}`. One or two sentences for the top of the page.

## `last_night`

```
{ "day", "games": [Card], "summary": {...}, "text": null }
```

`summary`: `games`, `final`, `picks_scored`, `picks_correct`, `market_favorites_scored`,
`market_favorites_correct`. Cards are ordered by tip-off (then game id).

**Card**

| key | type | notes |
|---|---|---|
| `game_id` | string | NBA 10-digit id |
| `status` | string | `final`, `live`, `scheduled`, ... |
| `final` | bool | |
| `season_type` | string | `Regular Season`, `PlayIn`, `Playoffs` |
| `home`, `away` | `{team, name, score}` | `score` null until final |
| `result` | object\|null | `{home_score, away_score, winner, margin (>=0), total, overtimes}` |
| `prediction` | object\|null | our latest pre-tip pick: `{model, created_at, home_win_prob, away_win_prob, home_margin, total, pick, pick_prob, confidence}`. `confidence` is `toss-up` (<56%), `lean` (<65%), `solid` (<75%), `strong` |
| `correct` | bool\|null | pick == winner; null if no prediction or not final |
| `market` | object\|null | line we had: `{source, provider, captured_at, kind ("snapshot"\|"closing"), home_spread, total, home_ml, away_ml, home_prob (de-vigged)}` plus result flags `home_covered`, `went_over` (null on a push), `favorite`, `favorite_won` |
| `facts` | object\|null | the full analytics `game_facts` document (see `docs/ANALYTICS.md` section 4): top performers, unsung hero candidates, runs, turning point, clutch, notable lines, incidents, validation. Null with `facts_available: false` if the box/play-by-play was not archived or analytics failed |
| `facts_available` | bool | |
| `text` | `{headline, story}` | both null until written. `story` is 2-4 sentences |

## `team_pulse`

`{"available": false, "reason", "text": null}` when the trends engine failed; otherwise:

| key | notes |
|---|---|
| `available`, `as_of`, `season`, `data_through`, `early_season` (flags suppressed before ~15 games), `games_per_team_median`, `notes` | |
| `method` | `net_ratings`, `flag_rule`, `flag_validity`: the measured reliability of "rising/sliding" flags; use it to hedge wording |
| `rising`, `sliding` | team entries, each with `text: null` (blurb slot) |
| `power_top10` | `{rank, team, record, power, adj_net, last10_adj_net, change_7d}` |
| `power_movers_7d` | `{team, rank, rank_7d_ago, change_7d}` |
| `streaks` | `{team, streak ("W6"/"L15"), record, detail}` (5+ games) |
| `text` | section blurb slot |

**Team entry** (`rising`/`sliding`): `team, name, record, last10, streak, power_rank`,
`net {season_raw, last10_raw, season_adj, last10_adj}` (points per 100 possessions),
`off {season,last10}`, `def {season,last10}`, `trend {delta, z, baseline, score}`,
`sos_last10 {opp_adj_net, home_games, rank, of}`, `reasons [{kind, value, detail}]` (machine-readable
kinds in `docs/TRENDS.md`), `key_absences [{player, team_games_missed, mpg_before, last_played}]`,
`text`.

## `player_pulse`

Same availability/header keys as `team_pulse` (`available, as_of, season, data_through, early_season,
games_per_team_median, notes, method {impact_metric, hot_cold_rule, hot_cold_validity}, text`).

| key | notes |
|---|---|
| `hot`, `cold` | `{player_id, player, team, z, window ("last 5"\|"last 10"), impact_recent, impact_baseline, impact_season, impact_prior_season, recent{...}, season{...}, detail, text}`. Impact is Courtside Impact (Game Score units) |
| `role_changes` | per player: `{player_id, player, team, kinds[], details[], numbers{}}`; kinds such as `minutes_up`, `starter_minutes`, `new_in_rotation`, `out_of_rotation`, `usage_up`, `new_team` |
| `leaders_last10` | `{impact\|pts\|reb\|ast\|stocks\|fg3m\|ts_pct: [{player, team, value, games, line "pts/reb/ast", ts_pct, min, impact}]}` |
| `streaks` | `{player, team, kind, games, games_this_season, at_least, since, avg_pts, avg_reb, avg_ast, detail}` |
| `career_highs_last7d` | `{player, team, stat, value, previous_high, date, game_id, detail}` |

## `injury_watch`

| key | notes |
|---|---|
| `available` | any injury data at all |
| `as_of`, `latest_snapshot`, `baseline_snapshot` | the two snapshot times compared (baseline ~24 h before latest, at least 12 h) |
| `diff_available` | false when fewer than two usable snapshots exist; then `changes` lists are empty |
| `sources` | `nba_official` and/or `espn` |
| `changes.new_absences`, `.returns`, `.status_changes` | Change entries, teams playing tonight first, then by impact (max 40) |
| `out_tonight` | highest-impact Out/Doubtful/Questionable players of teams playing today, Out/Doubtful first then by impact (max 15) |
| `text` | |

**Change entry:** `kind` (`new_absence`/`return`/`status_change`), `team`, `player`, `player_id` (null for the
official report), `from_status` (null = not listed), `to_status` (`Out`, `Doubtful`, `Questionable`, `Healthy`),
`direction` (`worse`/`better`), `reason`, `source`, `mpg`, `impact`, `impact_games`, `last_played` (player's
recent minutes/Courtside Impact; null when the name is unknown), `model_effect_margin` (**filled since 1.1**:
points of *his own team's* expected margin that the opinion model attributes to his status, negative = hurts his
team, e.g. `-2.2` for a Questionable LeBron; `0` for a listed player who is not in the rotation; null when
the player's team is not playing tonight or the name could not be matched to a roster player),
`playing_tonight`, `game_id`.

**Out-tonight entry:** `team, player, status, severity (3 Out, 2 Doubtful, 1 Questionable), reason, source,
captured_at, mpg, impact, impact_games, last_played, model_effect_margin, game_id, opponent`.

## `tonight`

`{ "day", "games": [Game], "upset_watch": [game_id], "summary", "text": null }`;
`summary`: `games, with_prediction, with_market, upset_watch`.

**Game**

| key | notes |
|---|---|
| `game_id`, `tipoff_utc`, `status`, `season_type` | |
| `home`, `away` | `{team, name, record, last10, streak, power_rank}` (record fields null if unknown) |
| `prediction` | headline model (`gbm_market`), as in the card: `model, home_win_prob, away_win_prob, home_margin, total, pick, pick_prob, confidence`, `created_at`; null if no model output |
| `opinion` | "our independent number" (`gbm_opinion`, no market input), same shape as `prediction`; null when only a results-only model ran |
| `market` | `{source, provider, captured_at, kind, home_spread, total, home_ml, away_ml, home_prob}`; null if no line |
| `edge` | our number minus the market's: `model` (which model: `gbm_opinion` when `opinion` exists, else the headline), `win_prob` (home, in probability points), `margin` (home margin vs market-implied margin, points), `total`. Positive `win_prob`/`margin` = we like the home team more than the market does |
| `availability` | `{home, away}`, each `{basis, source, captured_at}`: how the model got that team's availability (`basis` as in `meta.availability_basis`; `source` is `nba_official`/`espn` for `injury_report`); null for older rows |
| `factors` | machine-readable reasons, see below |
| `injuries` | `{home: [...], away: [...]}` Out/Doubtful/Questionable players with `mpg`, `impact` (same shape as an out-tonight entry), most severe first |
| `upset_watch` | bool |
| `upset` | `{flag, favorite, underdog, model_prob, market_prob, edge, model_picks_underdog}` or null when there is no market probability. The flag is set when the market favourite is >= 60% to win, our model gives the underdog >= 40% and at least 5 points more than the market does |
| `text` | `{preview: null}` |

**Factor** `{kind, team, side ("home"\|"away"\|null), value, favors ("home"\|"away"\|null), detail}`.
`detail` is a plain-English phrase; `value` and `kind` are the machine fields.

| kind | value | meaning |
|---|---|---|
| `back_to_back` / `four_in_six` / `three_in_four` | `true` | schedule fatigue for `team`; `favors` the other side |
| `long_travel` | km | >= 2000 km since last game |
| `time_zone` | hours | >= 2 hour shift |
| `rest_edge` | days | one team has >= 2 more days of rest (`team` is the rested one) |
| `neutral_site` | `true` | no home court |
| `injury` | `{player, status, mpg, impact}` | up to 3 highest-impact absences per team with >= 18 mpg |
| `form` | `{record, last10, streak, power_rank}` | one per team |
| `model_vs_market` | gap in win probability | present when |our number (opinion model) - market| >= 6 points |
| `model_factor` | points (`impact_pts`) | **1.1.** The model's own top 3-4 reasons for the margin, strongest first, listed *before* the schedule factors. Extra keys: `factor` (`absences`, `player_strength`, `team_rating`, `rest`, `travel`, `recent_form`, `continuity`), `impact_pts` (= `value`, points of expected margin **from `team`'s point of view**; negative = it hurts `team`). `favors` is the side it helps. `detail` is short plain English, e.g. `LAL absences: LeBron James (Out), Austin Reaves (Questionable)` or `LAC rest and schedule -1.3 pts`. For `absences`, `impact_pts` is the cost against a fully healthy roster (an exact re-score of the model), not the mean-centred `explain` share. Home court and market are deliberately not listed. Clients that do not know the kind should show `detail` or skip it |

## `scorecard`

Season-to-date accuracy of the **stored latest pre-tip prediction** per game against the final result.
Predictions are append-only; backfilled walk-forward ones are marked by `created_at` at 12:00 UTC of the day.

| key | notes |
|---|---|
| `model`, `season`, `through` (last day included), `scope` | |
| `n`, `correct`, `incorrect`, `accuracy`, `brier`, `log_loss`, `ece`, `mean_pick_prob` | all null/0 when `n == 0` |
| `calibration` | `[{lo, hi, n, mean_p, frac_home_won}]` (10 bins when n >= 200, else 5; empty bins have null means) |
| `by_confidence` | `[{tier, n, correct, accuracy}]` |
| `vs_market` | on exactly the games that have a market probability: `{n, model {accuracy, brier, log_loss}, market {...}, model_beats_market_brier, brier_diff, margin_mae {n, model, market}, total_mae {n, model, market}}`; null if no market games |
| `recent` | `{last_night {n, correct}, last_7_days {n, correct}}` |
| `text` | |

| `opinion` | **1.1.** The independent model (`gbm_opinion`) on the same footing: `{model, season, n, correct, incorrect, accuracy, brier, log_loss, ece, vs_market}`; null while it has no resolved predictions. This is the honest "us vs the market" line |

Be honest in the UI: if `vs_market.model_beats_market_brier` is false, say the market is currently
sharper. The headline `gbm_market` model sees the line, so it ties the market by construction; the
independent `opinion.vs_market` is where "are we better than the market" is actually answered (so far:
no, the market is sharper).

## Text slots (what the writer fills)

`headline.text`; each `last_night.games[i].text.{headline, story}`; `team_pulse.text` and
`rising[i].text` / `sliding[i].text`; `player_pulse.text` and `hot[i].text` / `cold[i].text`;
`injury_watch.text`; `tonight.text` and `tonight.games[i].text.preview`; `last_night.text`; `scorecard.text`.
The writer receives the whole Brief and returns it with these filled (`courtside.brief.writer.fill`).
If it is absent or raises, the slots stay `null`.

## Production models (what the numbers are)

- `predict.predict_day` stores **two** `predictions` rows per game: `gbm_market` (headline win
  probability / margin / total; takes the live market line as a feature, `mkt_margin = -home_spread`) and
  `gbm_opinion` (no market input at all). Both are the stacked LightGBM models of `docs/MODEL.md` rung 3.
- **Lines.** The newest pre-tip `odds_snapshots` batch; if there is none (past/demo days), the
  `closing_lines` row. Whichever line was used is stored in `inputs.market` and in the `market_*` columns of
  both rows.
- **Injuries are priced.** The newest usable report (the official report when it is the newer one, otherwise
  ESPN; `injuries.resolve`) is matched by name to each team's recent roster (accents, Jr./III, nicknames,
  "Last, First"; `injuries.match_player`), mapped to P(plays) (Out 0, Doubtful 0.1, Questionable 0.5,
  Day-To-Day 0.6, Probable 0.9, Available or not listed 1) and turned into an expected lineup whose strength
  replaces the engine's availability. A team the report does not list is healthy.
- **Without a report** the engine keeps its own availability. On a live day that is the full recent roster
  (`full_roster`). On a *backfilled/demo* day it is the rotation regulars (projected >= 20 min) who actually
  played, with fringe players assumed available: this is leak-safe (see MODEL.md, "The availability leak") but
  it is **not** an injury report. Only 2025-26 days 02-19 and 02-20 have official reports, so earlier
  backfilled days use `regulars_played`; `meta.availability_basis` and `meta.notes` say so.
- **Cache.** Fitted models are cached per NBA day in `$COURTSIDE_DATA/models/<day>/` and invalidated when
  `team_games` gains results before the day; later pregame runs only re-price with fresh lines and injuries.
- **Stored per row** (`inputs_json`): `schedule`, `elo`, `market`, `injuries` (Out/Doubtful/Questionable as the
  Brief lists them), `availability.{home,away}` (`basis, source, captured_at, lineup_strength, players[]` with
  `status, p_play, player_id, how` the match rule), `factors` (the Brief's `model_factor` items), `explain`
  (raw top-8 from `GBMModel.explain`), `model_effects` (opinion row: per listed player the effect on his
  team's margin) and `fit` (`fitted_at, cache hit/miss, seconds`).

## Writer contract (unchanged)

`brief/writer.py` reads `tonight.games[i]` through `normalize_prediction`: `prediction.*`, `edge.margin`,
`market.*`, `upset`, `injuries`, and each `factors[]` item only through its `detail` string (`reasons`, first 8,
`form` dropped, `injury` dropped when an injuries list exists). `model_factor` items are shaped like the old
factors (`kind, team, side, value, favors, detail`), so nothing changes there; they simply come first. The
writer copies `meta.model_uses_injuries` onto each game. `web/view.py`'s `factor_chip` renders an unknown
kind with `detail` as the label, so `model_factor` items appear as chips until the web gives them their own
treatment (filter on `kind == "model_factor"` and use `factor`/`impact_pts`).

## Predictions and data tables (for reference)

- `predictions(game_id, model, created_at, home_win_prob, home_margin, total, market_home_spread,
  market_total, market_home_prob, inputs_json, nba_day, home, away)`: append-only. View
  `latest_pretip_predictions` = newest row per (game, model) made at or before tip-off. Models in use:
  `gbm_market`, `gbm_opinion` (production); `elo_sched` rows from before the switch stay for comparison.
- `game_facts(game_id, nba_day, facts_json, computed_at)`: analytics facts per final game.
- `briefs(nba_day, brief_json, built_at, version)`.
