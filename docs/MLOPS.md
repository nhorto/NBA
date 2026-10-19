# MLOps

Goal: always know how the model is doing, catch problems before readers do, and change models only when a change is proven better.

Everything runs locally and free. SQLite is the source of truth for production lineage and monitoring. MLflow, with a local SQLite backend, tracks experiments (backtests, tuning and challenger runs) and gives them a browsable UI. Schema: `db.py` migration 5.

## 1. Versioning and lineage

- **A model version is a spec:** `model_name` plus code SHA, hyperparameters, and feature set. The `spec_hash` is a hash of those parts. The production models already refit every day on new data ("continuous training"); a refit is *not* a new version. A new version means new code, new hyperparameters, or new features.
- **`model_versions`** lists each version with a `status`: `candidate` → `shadow` → `champion` → `retired`. There is at most one champion per model name (`gbm_market`, `gbm_opinion`).
- **`registry_events`** is the append-only audit trail of who changed what, and why.
- **Every prediction row is tagged** with `model_version` and `role`. A `champion` prediction is shown to readers; a `shadow` prediction is logged for evaluation only. Any number in the Brief can therefore be traced to the exact code and parameters that produced it.

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
| calibration | Last-300 ECE > 0.05 |
| bias | Last-300 home, margin or total bias is beyond 2 SE ("over-predicting totals by 2.1 pts") |
| data | Any check fails; repeated warnings escalate |
| drift | Any key feature has PSI > 0.25 (warn) or > 0.5 (critical) |

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

## 5. Surfaces

- **Website `/health`:** status tiles, rolling log-loss/Brier vs the market over the season, a calibration plot, bias trends, the model version timeline, open alerts, and data-check results.
- **MLflow UI:** experiment history and comparisons, served locally on the Mac.
- **`docs/MODEL_CARD.md`:** generated from the registry. It covers the champion's intended use, training data, metrics, known limitations and caveats.
