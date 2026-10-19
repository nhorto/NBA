"""MLflow experiment tracking: benchmarks, tuning and challenger runs (docs/MLOPS.md).

Local only: tracking store `sqlite:///<data_dir>/mlflow.db`, artifacts under
`<data_dir>/mlartifacts`. Browse with

    mlflow ui --backend-store-uri sqlite:///<data_dir>/mlflow.db

MLflow is optional. It is imported lazily, never by the production path (`predict.py`), and
every helper here degrades to a logged warning and a `None` run id when mlflow is missing,
disabled (`COURTSIDE_MLFLOW=0`) or misconfigured. SQLite (`model_versions.mlflow_run_id`)
stays the source of truth; MLflow is the browsable history.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

from .. import config

log = logging.getLogger("courtside.mlops.tracking")

EXPERIMENT = "courtside-models"
_MAX_PARAM = 500                 # mlflow rejects longer param values


def tracking_uri() -> str:
    return f"sqlite:///{config.data_dir() / 'mlflow.db'}"


def artifact_root() -> Path:
    return config.data_dir() / "mlartifacts"


def _import_mlflow():
    """The mlflow module, or None (missing / disabled). Separate so tests can break it."""
    if os.environ.get("COURTSIDE_MLFLOW", "1").lower() in ("0", "false", "off", "no"):
        return None
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        return mlflow
    except Exception as e:  # noqa: BLE001 - ImportError or a broken install
        log.info("mlflow unavailable (%s: %s); tracking disabled", type(e).__name__, e)
        return None


def available() -> bool:
    return _import_mlflow() is not None


def _flatten(d: dict, prefix: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


def _params(params: dict) -> dict:
    out = {}
    for k, v in _flatten(params).items():
        s = v if isinstance(v, str) else json.dumps(v, default=str)
        out[k[:250]] = s[:_MAX_PARAM]
    return out


def _metrics(metrics: dict) -> dict:
    out = {}
    for k, v in _flatten(metrics).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isfinite(float(v)):
            out[k.replace("|", "_").replace(" ", "_")[:250]] = float(v)
    return out


def _experiment_id(mlflow, name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    root = artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(name, artifact_location=root.resolve().as_uri())


def log_benchmark(run_name: str, params: dict, metrics: dict, artifacts: dict | None = None,
                  *, tags: dict | None = None, experiment: str = EXPERIMENT) -> str | None:
    """One MLflow run: flattened params, the finite numeric leaves of `metrics` (nested dicts
    become dotted keys), and `artifacts` ({name: dict | list | str | Path | DataFrame}; dicts
    and lists as JSON, DataFrames as csv.gz, Paths copied). Returns the run id, or None when
    tracking is unavailable or fails (never raises)."""
    mlflow = _import_mlflow()
    if mlflow is None:
        return None
    try:
        config.data_dir().mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri())
        exp_id = _experiment_id(mlflow, experiment)
        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            p = _params(params)
            if p:
                mlflow.log_params(p)
            m = _metrics(metrics)
            if m:
                mlflow.log_metrics(m)
            if tags:
                mlflow.set_tags({k: str(v)[:5000] for k, v in tags.items()})
            for name, obj in (artifacts or {}).items():
                _log_artifact(mlflow, name, obj)
            return run.info.run_id
    except Exception as e:  # noqa: BLE001 - tracking must never break a job
        log.warning("mlflow logging failed for %s (%s: %s)", run_name, type(e).__name__, e)
        return None


def _log_artifact(mlflow, name: str, obj) -> None:
    if isinstance(obj, (dict, list)):
        mlflow.log_dict(obj, name if name.endswith(".json") else f"{name}.json")
        return
    if isinstance(obj, Path):
        mlflow.log_artifact(str(obj))
        return
    if hasattr(obj, "to_csv"):
        with tempfile.TemporaryDirectory() as td:
            fn = name if name.endswith(".csv.gz") else f"{name}.csv.gz"
            path = Path(td) / fn
            obj.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        return
    mlflow.log_text(str(obj), name if "." in name else f"{name}.txt")
