# MLOps

Goal: always know how the model is doing, catch problems before readers do, and change models only when a change is proven better.

Everything runs locally and free. SQLite is the source of truth for production lineage and monitoring. MLflow, with a local SQLite backend, tracks experiments (backtests, tuning and challenger runs) and gives them a browsable UI. Schema: `db.py` migrations 5 and 6.

## 1. Versioning and lineage

- **A model version is a spec:** `model_name` plus code SHA, hyperparameters, and feature set. The `spec_hash` is a hash of those parts. The production models already refit every day on new data ("continuous training"); a refit is *not* a new version. A new version means new code, new hyperparameters, or new features.
- **`model_versions`** lists each version with a `status`: `candidate` → `shadow` → `champion` → `retired`. There is at most one champion per model name (`gbm_market`, `gbm_opinion`).
- **`registry_events`** is the append-only audit trail of who changed what, and why.
- **Every prediction row is tagged** with `model_version` and `role`. A `champion` prediction is shown to readers; a `shadow` prediction is logged for evaluation only. Any number in the Brief can therefore be traced to the exact code and parameters that produced it.

### Implementation (`mlops/registry.py`, `production.py`, `predict.py`)

- **Ids** are human-readable: `gbm_opinion@v2-1a2b3c4d` (per-model sequence number, then the first 8 hex digits of the spec hash). Registering an identical spec again returns the existing id.
- **Data fingerprint.** The version hash does not cover data, so every `register` event also records `details.data_fingerprint`: `{hash, tables: {name: [row_count, max_date]}}` over games, odds_snapshots, closing_lines, injury_snapshots, team_games, player_games, player_games_adv, dfs_hist_salaries, dfs_salaries, dfs_slates and dfs_contests (`registry.data_fingerprint`). It is never part of `spec_hash` or the version id. `registry.fingerprint_diff` compares two fingerprints; a row-count change with the max date unchanged reads as a backfill. `retrain` logs the changes since the champion was registered.
- **`spec_hash`** is the sha256 of canonical JSON of `{model_name, params, features, code_sha}`.
  - `params` is `{"market", "fixed": {"margin": [GBMConfig, n_trees], "total": [...]}, "search_grid": [...]}`. `fixed` is the `gbm.FROZEN` format, which the daily fit uses.
  - `features` is the LightGBM feature list.
  - `code_sha` is the last git commit that touched `src/courtside/model/` or `production.py`. If those files have uncommitted changes, it becomes `<sha>-dirty-<content hash>`. Commits to unrelated files don't change it.
- **Production reads the registry.** `predict_day` fits each production model with its champion's `fixed` params. With an empty or unreadable registry, it falls back to `gbm.FROZEN` and leaves `model_version` NULL. A promotion therefore changes what readers see starting with the next fit. The day's model cache key includes the version ids and params.
- **Shadows** are fitted and priced alongside the champions every time the pair is predicted (`predict_day` only, not backfills).
  - Shadow rows are stored with `model` = the version id, e.g. `gbm_opinion@v2-…`, plus `model_version` = the same id and `role='shadow'`.
  - Champion rows keep `model` = `gbm_market`/`gbm_opinion`, with `role='champion'`.
  - Rows from before the registry have `role` NULL and are treated as champion.
  - A shadow whose fit fails is dropped with a warning. It never blocks the champions.
- **Migration 6** rebuilds `predictions` with a surrogate `id` and a `UNIQUE (game_id, model, created_at, COALESCE(model_version,''))` key. Rows of different versions can then never replace each other. Reruns of the same version at the same virtual time still replace, as before.
  - **`latest_pretip_predictions`** is what readers see: champion/NULL rows only, the newest pre-tip row per (game, model), ties broken by `id`.
  - **`latest_pretip_predictions_all`** is the same, per (game, model, role, version), for evaluation.
  - The Brief, the scorecard and `predict.latest_pretip` read only champion/NULL rows (`predict.shown_roles_sql`).
- **`inputs_json.features`** holds the full priced input vector of every production row. Drift monitoring uses it. Keys:
  - strength: `strength_diff`, `lineup_strength_home/away/diff`
  - absences: `abs_loss_home/away`; `absences` (away loss minus home loss, so it has the sign of the home margin); `absences_total`
  - injury-report counts: `out_*`, `doubtful_*`, `questionable_*`
  - schedule and rating: `elo_diff`, `rest_diff`, `b2b_home/away/total`
  - market: `market_spread`, `market_total`, `market_home_prob`
  - outputs: `pred_home_prob`, `pred_margin`, `pred_total`
- **Bootstrap** (`courtside mlops bootstrap`, idempotent) registers today's `gbm.FROZEN`/`gbm.GRID` specs as champion v1. Their benchmark is the 2026-10-18 rung-3 walk-forward (= docs/MODEL.md). That run predates the season-opening rolling-prior fix (2aa9a89), so the challenger gate always recomputes the champion on the current code rather than reusing those numbers.

## 2. Monitoring (daily, after postgame)

For each model (champion, opinion, any shadows) plus the market baseline, `metric_snapshots` holds resolved-prediction metrics over the windows `7d`, `30d`, `last100`, `last300` and `season`:

- Log-loss, Brier, accuracy, calibration (ECE)
- **Bias** (are we over- or under-predicting?): home win rate predicted vs actual, margin bias, total bias
- **Versus the market on the same games:** the log-loss difference with a paired standard error

**Data checks** (`data_checks`) run after every collect job:

- Did the scheduled jobs run and succeed?
- Odds coverage of today's games
- Official injury report captured on game days
- Injury name-match rate
- Box scores and play-by-play archived for all finals
- Game logs refreshed
- Prediction coverage: every game predicted before tip-off
- Brief built

**Drift** (`drift_snapshots`) is the PSI of key live features (strength diff, absences, rest, market spread/total, predicted probability) against the champion's training reference.

## 3. Alerts

`alerts` holds one open row per key: re-firing updates the row and clearing it resolves it. Severities are `info`, `warn` and `critical`. Alerts appear as a banner on the site and on the Model Health page, and are written to the log.

Initial rules (to be tuned once the season has data):

| Kind | Fires when |
|---|---|
| performance | Last-300 log-loss is worse than the market by more than 2 SE, or worse than the backtest expectation band |
| calibration | **Model minus market** on the same games (n ≥ 200, last 300 or season): \|z\| > 2 in the same direction on 3 consecutive daily snapshots (within 7 days); critical if \|z\| > 3.5 on all 3. Falls back to the model-only Spiegelhalter test (\|z\| > 2.5 warn, > 3.5 critical, no persistence) when fewer than 200 games have a market line. |
| bias | Last-300 home, margin or total bias is beyond 2 SE ("over-predicting totals by 2.1 pts") |
| data | Any check fails; repeated warnings escalate |
| drift | Any key feature has PSI > 0.25 (warn) or > 0.5 (critical). Seasonal features (`elo_diff`, `strength_diff`, `lineup_strength_diff`) are skipped for 6 weeks from Oct 20, because early-season ratings are compressed. |

**Calibration vs the market.** The market also goes overconfident in Dec/Jan, so a model-only test false-alarms with it. Per game, `t = (y−p)(1−2p)` is the Spiegelhalter score term (its sum is the numerator of the z below; t > 0 is overconfident). The alert uses `d = t_model − t_market` on the same games and `z = mean(d) / date-clustered SE` (`monitor.calibration_vs_market`, stored as `details.cal_vs_mkt` with the slope difference). z > 0 means more overconfident than the market. Three consecutive daily snapshots are required instead of a CUSUM. The rolling windows overlap, so persistence only suppresses one-day blips; the market subtraction is what removes league-wide noise. A CUSUM would need a reference value tuned to ~0.1 sd per game and could not be recomputed from stored snapshots.

**Model-only calibration test (fallback).** `z = Σ(y−p)(1−2p) / √(Σ(1−2p)² p(1−p))` is approximately N(0,1) when the probabilities are calibrated, so the threshold has a known false-alarm rate (about 1% at 2.5). z > 0 means overconfident: favorites win less often than predicted. `details_json` also stores the logistic recalibration of y on logit(p): slope < 1 is overconfident and slope > 1 is underconfident (a calibrated model has intercept 0, slope 1). The alert message states the direction and the slope. ECE (10 bins) is still stored and shown, but it is not an alert: at n=300 even the market scores about 0.065, so any fixed ECE threshold fires on noise.

Early in the season, windows hold few games. Rules require a minimum *n* and use standard errors, so a bad first week doesn't page anyone.

## 4. Retraining and promotion

- **Daily:** the champion refits on new data. This already happens via the production cache.
- **Monthly, or on an alert:** the weekly job calls `courtside mlops retrain --if-due`. The retrain step runs only on the first weekly run of a month (day 1-7) or while a calibration or drift alert is open (`retrain.retrain_due`). Re-running the same grid on the same procedure gives diff = 0 and cannot pass its own gate, so a weekly retrain is wasted compute. `shadows` and `card` still run every week. The challenger job (`courtside mlops retrain`) runs these steps:
  1. Re-tunes hyperparameters on the latest data and/or builds any new spec, then registers the result as a `candidate` with an MLflow run.
  2. **Offline gate:** the candidate must beat the champion on the frozen walk-forward benchmark (2012-13 → last season, regular season only) by a significant paired log-loss margin (date-clustered SE). It must also not be worse on the most recent season or on calibration.
  3. **Shadow:** passing candidates predict live, logged but not shown, for at least 100 games.
  4. **Online gate:** the shadow must be "not clearly worse" than the champion on the same live games (paired log-loss diff ≤ 2 date-clustered SE) with no open critical alert.
  5. **Promote:** the old champion is retired, the event is logged, and `courtside mlops rollback` restores it in one command.
- **Triggered:** an open calibration or drift alert makes the next weekly run retrain (see above).

### Implementation (`mlops/retrain.py`)

**What the benchmark measures.** The fixed values in `gbm.FROZEN` were chosen on all data, test seasons included, so the walk-forward benchmark never uses them. It evaluates the *procedure* instead. For each test season S (2012-13 → last completed season), the inner time-ordered search runs on seasons before S only, picks a config and tree count, and predicts S. This is the inner validation that never touches the test season. It is how docs/MODEL.md benchmarks the champion (`gbm.GRID`, 4 configs).

- **`courtside mlops retrain [--search small|medium]`** builds the challenger from a wider grid, which always contains `gbm.GRID`.
  - `small` has 18 configs: leaves {7, 15, 31} × half-life {none, 4, 2} × min leaf {300, 100}.
  - `medium` adds L2 {10, 50} and two low-learning-rate configs.
  - Tree count is chosen by early stopping on the inner folds.
  - The champion and the challenger share one fit memo, so the champion's own search costs almost nothing extra. The champion is recomputed in the same run on the current code, never read from an old file.
  - After the gate, the challenger's production `fixed` params come from the same search on every completed season. This is exactly how `gbm.FROZEN` was produced.
- **`--params JSON`** builds a hand-made challenger.
  - The FROZEN format (`{"margin": [{"num_leaves": 31}, 60]}`; cfg keys override the champion's) is benchmarked with those values fixed in every fold, against the champion with its own fixed values in every fold.
  - `{"search_grid": [...]}` is benchmarked as a search procedure.
- **Scope.** Only the margin head is benchmarked, because the win probability (and so log-loss) is the calibrated margin. The total head is carried over from the champion unless it is given explicitly.
- **Limitation.** Re-running the *same* grid on newer data gives new `fixed` values but an identical benchmark procedure (diff = 0), so it cannot pass the gate. That is why the weekly job only retrains monthly or on an alert. A once-a-season FROZEN refresh is a manual `register` + `promote` (or a `--params` challenger).

**Offline gate.** Regular-season test games only (no playoffs), matched on identical game ids. For each game, `d = LL(challenger) − LL(champion)`, and SE is the date-clustered SE of mean(d) (`proj.metrics.paired_cluster`; per-game SE if the frame has no `game_date`). The challenger must meet all three rules:
1. **Overall:** mean(d) < −2·SE.
2. **Most recent test season:** mean(d) ≤ 2·SE of that season (also clustered). It may not be worse beyond noise.
3. **Calibration:** ECE(challenger) − ECE(champion) ≤ 0.005, using 10 equal-width bins over the same games.

A pass is registered as `candidate`, then moves to `shadow` (event `shadow`). A fail is registered as `candidate`, then moves to `retired` (event `reject`, with the reasons). If the spec is identical to the champion's, nothing is registered. Each run logs params, gate and summaries to MLflow, with per-game predictions as an artifact.

**Cost.** A `retrain --model gbm_opinion --search small` run on the M4 mini (2026-10-19) took 383 s wall time with peak RSS 1.35 GB, even though the machine was heavily loaded by other jobs:
- engine and features: 28 s
- 14-fold benchmark of champion and challenger: 335 s
- final search fit: 15 s

That is cheap enough to run monthly for both models. The first result was a rejection: +0.00026 ± 0.00019, 18,109 games.

**Online gate** (`courtside mlops shadows [--dry-run]`). For each shadow, take the resolved live games where both the shadow (`latest_pretip_predictions_all`) and its model's reader-visible champion row have a pre-tip prediction, and compute `d = LL(shadow) − LL(champion)` per game:
The offline gate already demanded a proven improvement, so this gate only guards against a live regression ("not clearly worse"). SE is the SE of mean(d) clustered by `nba_day`. At n = 100 the SE is about 0.003, against true effects of about 0.0005, so a plain "mean(d) ≤ 0" was a coin flip.
- n < 100: wait.
- n ≥ 100 and mean(d) > 2·SE: **reject** (event `reject`).
- n ≥ 100, otherwise, and no open `critical` alert: **promote**. The old champion is retired; `rollback` undoes it.
- n ≥ 100, not clearly worse, but a critical alert is open: wait.
- If there is no usable SE (for example, one night of games), the old rule applies: mean(d) ≤ 0 promotes, and n ≥ 300 with mean(d) > 0 rejects.

**Registry operations** (`mlops/registry.py`).
- `promote` and `rollback` run in one write transaction (`BEGIN IMMEDIATE`, or a SAVEPOINT inside a caller's transaction). They retire the old champion, set the new one, log both events, and re-check the one-champion invariant before committing.
- `rollback` restores the most recently retired *former champion*. Rejected candidates are never restored.

**Tracking** (`mlops/tracking.py`).
- MLflow stores runs at `sqlite:///<data_dir>/mlflow.db` and artifacts in `<data_dir>/mlartifacts`.
- It is imported lazily and never by `predict.py`. If MLflow is missing or broken, or `COURTSIDE_MLFLOW=0` is set, every helper returns `None` and logs a warning.
- Browse runs with `mlflow ui --backend-store-uri sqlite:///<data_dir>/mlflow.db`.

**Commands:**
- `courtside mlops bootstrap | versions [--model] [--events N] | promote <version> --reason | rollback <model> --reason | retrain [--model gbm_market|gbm_opinion|all] [--params JSON|@file] [--search] [--if-due] | shadows [--dry-run] | card`
- `courtside mlops card` regenerates `docs/MODEL_CARD.md`.

## 5. Surfaces

- **Website `/health`:** status tiles, rolling log-loss/Brier vs the market over the season, a calibration plot, bias trends, the model version timeline, open alerts, and data-check results.
- **MLflow UI:** experiment history and comparisons, served locally on the Mac.
- **`docs/MODEL_CARD.md`:** generated from the registry. It covers the champion's intended use, training data, metrics, known limitations and caveats.

## 6. Player-side shadows (IMPROVEMENTS E1)

The 2026-27 live season is the confirmatory set for player-side switches. Candidate switches run as live shadows next to the champion; a switch is flipped only after its pre-registered primary metric passes. Code: `proj/shadows.py`; registry: `mlops/shadow_registry.json`; tests: `tests/test_player_shadows.py`.

- **Where they run:** `deploy/run-job.sh` runs `courtside proj shadows run` after every pregame job (10:00, 13:30, 17:00, 18:30, 20:00, 22:00 ET), in its own process at `nice -n 10`. They never run in the DFS tick. A failure is logged and never fails the pregame job.
- **What runs:** the day's Classic slate with the most games still to tip. The champion runs once, then each active arm: the champion with one switch on, set through its `pipeline.set_*` setter. All of them use the same seed (`stable_seed(slate, "pregame", SEED_VERSION)`) and the same sim count (10,000; `COURTSIDE_SHADOW_SIMS`), so the pairs share random numbers. The arm order rotates by run. After each arm the switches, the bundle suffix and MODEL_VERSION are restored and checked.
- **Never served:** the database connection is read-only. Output goes to `<data>/proj/shadows/<day>/<HHMM>_<arm>.npz` plus a `<HHMM>.json` manifest, and never to `player_projections` or `dfs_runs`. Those tables feed the site, the props and the DFS monitor's champion registry. Each npz keeps the games not yet tipped: p_play, E[min], E[min | plays], E[DK] and 100 mid-point quantiles of the DK distribution (DNP = 0), about 120 KB for 11 games.
- **Guards:**
  - **Lock windows:** the run skips itself when any lock-45, lock-10 or swap window (the tick's own `runner.due_runs`) falls within [now − 10 min, now + budget + 2 min].
  - **Time budget:** 90 s for the whole run (`COURTSIDE_SHADOW_BUDGET_S`). An arm that would not fit is skipped, and a running arm is stopped by SIGALRM at the deadline.
  - **Bundles:** an arm whose bundle file is missing is skipped. A shadow never fits a bundle.
  - **Off switch:** `COURTSIDE_SHADOWS=0`, or `shadows.SHADOWS_DEFAULT = False`.
  - **Arm list:** `COURTSIDE_SHADOW_ARMS=+tr,+ep` overrides the registry's active list.
- **Grading:** the postgame step `shadow_grade` runs after `dfs_grade` and never fails the job.
  - **Pairs:** each arm is paired with the champion of the same run, on identical (game, player) rows. Actuals come from `player_games` (DNP = 0). Complete nights are cached as `<day>/graded.csv.gz`.
  - **Primary metric:** the arm's registered metric at its registered run, accumulated over the season. Metrics: DK CRPS (from the quantiles), DK MAE, minutes MAE on played rows, or P(play) log loss. The run is a slot ("13:30") or "last" (the latest run before each game).
  - **SE:** mean(arm − champion) with the two-way night × player clustered SE (`metrics.paired_cluster2`). The tier breakdown uses the champion's tiers.
  - **Report:** `<data>/reports/shadows/shadow_grades.json`, plus a dated copy. It is shown on `/health/dfs` under **Shadows**. `courtside proj shadows grade --date D` re-grades.
- **Decision rule (`shadows.decide`):** nothing is decided before `min_nights` (default 14) or `min_rows`.
  - **`lower` arms:** pass when z ≤ the bar (−2), no tier z is above +2 and every guard metric holds. Fail when z ≥ +2, or when `max_nights` (60) pass without a pass. Otherwise pending.
  - **`not_worse` arms:** pass when z ≤ +2 with the same tier rule; otherwise fail.
  - **Caveat:** the rule is checked every night from `min_nights` on. Those repeated looks make a −2 SE pass somewhat easier than a single test would. Treat a pass as the condition for asking the owner to flip the switch, not as the flip itself.
- **Registered arms (2026-10-23):** `+tr`, `+ep` (first 30 days only, needs `bundle_2026-27_v1_cx_mf_ep.json` in prod `var/proj/`), `+gd` (P(play) log loss at 10:00, not worse), `+lf` (latest run), `+tm` (not worse), `+rs` (minutes MAE on changed rows, ≥ 100 rows), `+pq` (latest run). Hypotheses and bars come from each arm's ledger or IMPROVEMENTS entry. Change a bar only before an arm's first graded night; after that, register a new arm.
- **Cost (11-game slate, geekom1):** about 48–57 s per run. Context and the champion take 12–14 s; most arms take 4–7 s, and `+ep` 8–11 s with its bundle load. A 2-game slate takes about 25 s.
