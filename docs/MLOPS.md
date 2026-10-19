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
| calibration | Spiegelhalter z-test on the last 300 games (n ≥ 200): \|z\| > 2.5 warns, > 3.5 is critical. The season window is tested too, and the alert fires if either is significant, reporting both. |
| bias | Last-300 home, margin or total bias is beyond 2 SE ("over-predicting totals by 2.1 pts") |
| data | Any check fails; repeated warnings escalate |
| drift | Any key feature has PSI > 0.25 (warn) or > 0.5 (critical) |

**Calibration test.** `z = Σ(y−p)(1−2p) / √(Σ(1−2p)² p(1−p))` is approximately N(0,1) when the probabilities are calibrated, so the threshold has a known false-alarm rate (about 1% at 2.5). z > 0 means overconfident: favorites win less often than predicted. `details_json` also stores the logistic recalibration of y on logit(p): slope < 1 is overconfident and slope > 1 is underconfident (a calibrated model has intercept 0, slope 1). The alert message states the direction and the slope. ECE (10 bins) is still stored and shown, but it is not an alert: at n=300 even the market scores about 0.065, so any fixed ECE threshold fires on noise.

Early in the season, windows hold few games. Rules require a minimum *n* and use standard errors, so a bad first week doesn't page anyone.

## 4. Retraining and promotion

- **Daily:** the champion refits on new data. This already happens via the production cache.
- **Weekly:** a challenger job (`courtside mlops retrain`) runs these steps:
  1. Re-tunes hyperparameters on the latest data and/or builds any new spec, then registers the result as a `candidate` with an MLflow run.
  2. **Offline gate:** the candidate must beat the champion on the frozen walk-forward benchmark (2012-13 → last season) by a significant paired log-loss margin. It must also not be worse on the most recent season or on calibration.
  3. **Shadow:** passing candidates predict live, logged but not shown, for at least 100 games.
  4. **Online gate:** the shadow's paired log-loss vs the champion on the same live games must be ≤ 0 with no new alerts.
  5. **Promote:** the old champion is retired, the event is logged, and `courtside mlops rollback` restores it in one command.
- **Triggered:** a performance or drift alert queues an early challenger run.

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
- **Limitation.** Re-running the *same* grid on newer data gives new `fixed` values but an identical benchmark procedure (diff = 0), so it cannot pass the gate. A once-a-season FROZEN refresh is a manual `register` + `promote` (or a `--params` challenger).

**Offline gate.** All test games count, regular season and playoffs, matched on identical game ids. For each game, `d = LL(challenger) − LL(champion)`, and SE = sd(d)/√n. The challenger must meet all three rules:
1. **Overall:** mean(d) < −2·SE.
2. **Most recent test season:** mean(d) ≤ 2·SE of that season. It may not be worse beyond noise.
3. **Calibration:** ECE(challenger) − ECE(champion) ≤ 0.005, using 10 equal-width bins over all test games.

A pass is registered as `candidate`, then moves to `shadow` (event `shadow`). A fail is registered as `candidate`, then moves to `retired` (event `reject`, with the reasons). If the spec is identical to the champion's, nothing is registered. Each run logs params, gate and summaries to MLflow, with per-game predictions as an artifact.

**Cost.** A `retrain --model gbm_opinion --search small` run on the M4 mini (2026-10-19) took 383 s wall time with peak RSS 1.35 GB, even though the machine was heavily loaded by other jobs:
- engine and features: 28 s
- 14-fold benchmark of champion and challenger: 335 s
- final search fit: 15 s

That is cheap enough to run weekly for both models. The first result was a rejection: +0.00026 ± 0.00019, 18,109 games.

**Online gate** (`courtside mlops shadows [--dry-run]`). For each shadow, take the resolved live games where both the shadow (`latest_pretip_predictions_all`) and its model's reader-visible champion row have a pre-tip prediction, and compute `d = LL(shadow) − LL(champion)` per game:
- n < 100: wait.
- n ≥ 100, mean(d) ≤ 0 and no open `critical` alert: **promote**. The old champion is retired; `rollback` undoes it.
- n ≥ 300 and mean(d) > 0: **reject** (event `reject`).
- Otherwise (better but a critical alert is open, or worse with n < 300): wait.

**Registry operations** (`mlops/registry.py`).
- `promote` and `rollback` run in one write transaction (`BEGIN IMMEDIATE`, or a SAVEPOINT inside a caller's transaction). They retire the old champion, set the new one, log both events, and re-check the one-champion invariant before committing.
- `rollback` restores the most recently retired *former champion*. Rejected candidates are never restored.

**Tracking** (`mlops/tracking.py`).
- MLflow stores runs at `sqlite:///<data_dir>/mlflow.db` and artifacts in `<data_dir>/mlartifacts`.
- It is imported lazily and never by `predict.py`. If MLflow is missing or broken, or `COURTSIDE_MLFLOW=0` is set, every helper returns `None` and logs a warning.
- Browse runs with `mlflow ui --backend-store-uri sqlite:///<data_dir>/mlflow.db`.

**Commands:**
- `courtside mlops bootstrap | versions [--model] [--events N] | promote <version> --reason | rollback <model> --reason | retrain [--model gbm_market|gbm_opinion|all] [--params JSON|@file] [--search] | shadows [--dry-run] | card`
- `courtside mlops card` regenerates `docs/MODEL_CARD.md`.

## 5. Surfaces

- **Website `/health`:** status tiles, rolling log-loss/Brier vs the market over the season, a calibration plot, bias trends, the model version timeline, open alerts, and data-check results.
- **MLflow UI:** experiment history and comparisons, served locally on the Mac.
- **`docs/MODEL_CARD.md`:** generated from the registry. It covers the champion's intended use, training data, metrics, known limitations and caveats.
