"""Model registry: versions, status transitions, promotion and rollback (docs/MLOPS.md, 1 + 4).

A *version* is a spec: model name + hyperparameters + feature set + code SHA, identified by
`spec_hash`. Daily refits of a spec on new data are not new versions. Ids are human-readable:
`gbm_market@v1-1a2b3c4d` (per-model sequence number + first 8 hex digits of the spec hash).

Status lifecycle: candidate -> shadow -> champion -> retired. At most one champion per model
name; `promote` and `rollback` swap champions inside one write transaction (BEGIN IMMEDIATE,
or a SAVEPOINT inside a caller's transaction) and re-check the invariant before committing.
Every transition appends to `registry_events` (register | shadow | promote | rollback |
retire | reject).

Params format (`model_versions.params_json`) for the GBM models:

    {"market": bool,                                  # gbm.SPECS
     "fixed": {"margin": [GBMConfig dict, n_trees],   # what the daily fit uses (gbm.FROZEN)
               "total":  [GBMConfig dict, n_trees]},
     "search_grid": [GBMConfig dict, ...]}            # the search the fixed values came from

Production (`predict.predict_day`) reads the champion's `fixed` params from here (falling back
to `gbm.FROZEN` when the registry is empty), so a promotion changes what readers see.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .. import config

log = logging.getLogger("courtside.mlops.registry")

STATUSES = ("candidate", "shadow", "champion", "retired")
ACTIONS = ("register", "shadow", "promote", "rollback", "retire", "reject")
# The files that define a model; their git state is the version's code identity.
MODEL_PATHS = ("src/courtside/model", "src/courtside/production.py")


class RegistryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------ identity

def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True, timeout=20).stdout


def _content_hash(root: Path, paths=MODEL_PATHS) -> str:
    h = hashlib.sha256()
    for rel in paths:
        p = root / rel
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            if f.exists():
                h.update(str(f.relative_to(root)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def code_sha(root: Path | None = None) -> str:
    """Code identity of the model-defining files (`MODEL_PATHS`): the last commit touching
    them (12 hex), plus `-dirty-<8 hex>` (hash of their working-tree content) when they have
    uncommitted changes. Without git: `nogit-<12 hex content hash>`. Unrelated commits (docs,
    web, ...) do not change it."""
    root = Path(root or config.REPO_ROOT)
    try:
        commit = _git(["log", "-1", "--format=%H", "--", *MODEL_PATHS], root).strip()
        if not commit:
            raise RuntimeError("no commit touches the model files")
        dirty = _git(["status", "--porcelain", "--untracked-files=all", "--", *MODEL_PATHS],
                     root)
        dirty = [ln for ln in dirty.splitlines() if ln.strip().endswith(".py")]
        if dirty:
            return f"{commit[:12]}-dirty-{_content_hash(root)[:8]}"
        return commit[:12]
    except Exception:  # noqa: BLE001 - no git binary / not a repo / shallow copy
        return f"nogit-{_content_hash(root)[:12]}"


def spec_hash(model_name: str, params: dict, feature_list, code_sha: str | None) -> str:
    """Deterministic sha256 (hex) of the spec: key order and list/tuple types don't matter."""
    doc = {"model_name": model_name, "params": params, "features": list(feature_list or []),
           "code_sha": code_sha}
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ------------------------------------------------------------------ data fingerprint

# (table, date column or None). The version hash covers model, params, features and code, not the
# data; this is the data side. A backfill (rows added in the past) changes a count with the max
# date unchanged; a normal daily collect moves both. Recorded on the `register` event
# (`details.data_fingerprint`), never part of `spec_hash` or the version id.
FINGERPRINT_TABLES = (
    ("games", "nba_day"), ("odds_snapshots", "captured_at"), ("closing_lines", None),
    ("injury_snapshots", "captured_at"), ("team_games", "game_date"),
    ("player_games", "game_date"), ("player_games_adv", None),
    ("dfs_hist_salaries", "game_date"), ("dfs_salaries", "captured_at"),
    ("dfs_slates", "nba_day"), ("dfs_contests", "captured_at"))


def data_fingerprint(conn) -> dict:
    """{"hash": sha256 hex, "tables": {table: [row_count, max_date]}} over FINGERPRINT_TABLES.
    A missing table is [0, None]. Read-only; cheap (COUNT + MAX on indexed columns)."""
    tables: dict[str, list] = {}
    for t, col in FINGERPRINT_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            mx = conn.execute(f"SELECT MAX({col}) FROM {t}").fetchone()[0] if col else None
        except sqlite3.Error:
            n, mx = 0, None
        tables[t] = [int(n), mx]
    blob = json.dumps(tables, sort_keys=True, separators=(",", ":"), default=str)
    return {"hash": hashlib.sha256(blob.encode()).hexdigest(), "tables": tables}


def fingerprint_diff(old: dict | None, new: dict | None) -> list[str]:
    """Human-readable table changes between two fingerprints ([] = identical data). Flags rows
    added or removed with the max date unchanged as a backfill."""
    if not old or not new:
        return []
    out = []
    for t, (n1, d1) in (new.get("tables") or {}).items():
        n0, d0 = (old.get("tables") or {}).get(t, [0, None])
        if (n0, d0) == (n1, d1):
            continue
        tag = " (backfill-like: max date unchanged)" if d0 == d1 and n0 != n1 else ""
        out.append(f"{t}: rows {n0}->{n1}, max {d0}->{d1}{tag}")
    return out


def version_fingerprint(conn, version_id: str) -> dict | None:
    """The data fingerprint recorded when `version_id` was registered (None for versions
    registered before fingerprints existed)."""
    for e in events(conn, version_id=version_id):
        if e["action"] == "register" and e["details"] and e["details"].get("data_fingerprint"):
            return e["details"]["data_fingerprint"]
    return None


# ------------------------------------------------------------------ production specs

def production_names() -> tuple[str, ...]:
    from ..model import gbm
    return (gbm.PRODUCTION_WINPROB, gbm.PRODUCTION_OPINION)


def feature_list(model_name: str) -> list[str]:
    """LightGBM feature columns of a production GBM (the linear base is fixed in code)."""
    from ..model import gbm
    return list(gbm._gbm_cols(bool(gbm.SPECS[model_name]["market"])))


def production_params(model_name: str) -> dict:
    """The params of today's hard-coded production spec (`gbm.FROZEN`, `gbm.GRID`)."""
    from ..model import gbm
    fx = gbm.FROZEN[model_name]
    return {"market": bool(gbm.SPECS[model_name]["market"]),
            "fixed": {t: [cfg.to_dict(), int(n)] for t, (cfg, n) in fx.items()},
            "search_grid": [c.to_dict() for c in gbm.GRID]}


def fixed_from_params(params: dict) -> dict:
    """{target: (GBMConfig, n_trees)} (the gbm.FROZEN format) from a params dict."""
    from ..model import gbm
    return {t: (gbm.GBMConfig(**cfg), int(n)) for t, (cfg, n) in params["fixed"].items()}


# ------------------------------------------------------------------ transactions

@contextmanager
def _tx(conn: sqlite3.Connection):
    """One write transaction: BEGIN IMMEDIATE (takes the write lock before we read the
    current champion), or a SAVEPOINT inside a caller's open transaction."""
    if conn.in_transaction:
        conn.execute("SAVEPOINT registry_tx")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK TO registry_tx")
            conn.execute("RELEASE registry_tx")
            raise
        conn.execute("RELEASE registry_tx")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _event(conn, version_id: str, action: str, reason: str | None, details: dict | None = None,
           ts: str | None = None) -> None:
    if action not in ACTIONS:
        raise RegistryError(f"unknown action {action!r}")
    conn.execute("INSERT INTO registry_events (ts, version_id, action, reason, details_json) "
                 "VALUES (?,?,?,?,?)", (ts or _now(), version_id, action, reason,
                                        json.dumps(details, default=str) if details else None))


def _check_one_champion(conn, model_name: str) -> None:
    n = conn.execute("SELECT COUNT(*) FROM model_versions WHERE model_name=? AND "
                     "status='champion'", (model_name,)).fetchone()[0]
    if n > 1:
        raise RegistryError(f"{model_name}: {n} champions (invariant: at most one)")


# ------------------------------------------------------------------ reads

def _row(r) -> dict:
    d = dict(r)
    for k in ("params", "benchmark"):
        raw = d.pop(f"{k}_json", None)
        d[k] = json.loads(raw) if raw else None
    return d


def _select(conn, where: str = "", args=()) -> list[dict]:
    cur = conn.execute(f"SELECT * FROM model_versions {where}", args)
    cols = [c[0] for c in cur.description]
    return [_row(dict(zip(cols, r))) for r in cur.fetchall()]


def get_version(conn, version_id: str) -> dict | None:
    rows = _select(conn, "WHERE version_id=?", (version_id,))
    return rows[0] if rows else None


def get_champion(conn, model_name: str) -> dict | None:
    rows = _select(conn, "WHERE model_name=? AND status='champion'", (model_name,))
    return rows[0] if rows else None


def list_versions(conn, model_name: str | None = None, status: str | None = None) -> list[dict]:
    where, args = [], []
    if model_name:
        where.append("model_name=?")
        args.append(model_name)
    if status:
        where.append("status=?")
        args.append(status)
    sql = ("WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY model_name, created_at, rowid"
    return _select(conn, sql, args)


def events(conn, version_id: str | None = None, model_name: str | None = None,
           limit: int | None = None) -> list[dict]:
    sql = ("SELECT e.* FROM registry_events e JOIN model_versions v ON v.version_id = e.version_id")
    where, args = [], []
    if version_id:
        where.append("e.version_id=?")
        args.append(version_id)
    if model_name:
        where.append("v.model_name=?")
        args.append(model_name)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY e.id DESC" + (f" LIMIT {int(limit)}" if limit else "")
    cur = conn.execute(sql, args)
    cols = [c[0] for c in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        d["details"] = json.loads(d.pop("details_json")) if d.get("details_json") else None
        out.append(d)
    return out


def production_versions(conn) -> dict:
    """{"champions": {model_name: row}, "shadows": [row, ...]} for the production models.
    Empty when the registry tables don't exist (an unmigrated DB)."""
    names = production_names()
    try:
        champs = {n: c for n in names if (c := get_champion(conn, n)) is not None}
        shadows = [v for v in list_versions(conn, status="shadow") if v["model_name"] in names]
    except sqlite3.OperationalError:
        return {"champions": {}, "shadows": []}
    return {"champions": champs, "shadows": shadows}


# ------------------------------------------------------------------ writes

def register(conn, model_name: str, params: dict, feature_list=None, *, code: str | None = None,
             status: str = "candidate", benchmark: dict | None = None, notes: str | None = None,
             mlflow_run_id: str | None = None, reason: str | None = None,
             data_fp: dict | None = None) -> str:
    """Register a spec and return its version id. An identical spec (same spec_hash for the
    model) is not registered twice: its existing id is returned unchanged. `status`:
    candidate (default), shadow, retired, or champion (which promotes it atomically).
    `data_fp`: data fingerprint to record on the register event (default: computed now)."""
    if status not in STATUSES:
        raise RegistryError(f"unknown status {status!r}")
    code = code if code is not None else code_sha()
    data_fp = data_fp if data_fp is not None else data_fingerprint(conn)
    sh = spec_hash(model_name, params, feature_list, code)
    with _tx(conn):
        hit = conn.execute("SELECT version_id FROM model_versions WHERE model_name=? AND "
                           "spec_hash=?", (model_name, sh)).fetchone()
        if hit:
            return hit[0]
        n = conn.execute("SELECT COUNT(*) FROM model_versions WHERE model_name=?",
                         (model_name,)).fetchone()[0]
        vid = f"{model_name}@v{n + 1}-{sh[:8]}"
        ts = _now()
        conn.execute(
            "INSERT INTO model_versions (version_id, model_name, spec_hash, code_sha, "
            "params_json, status, created_at, benchmark_json, mlflow_run_id, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (vid, model_name, sh, code, json.dumps(params, sort_keys=True), "candidate", ts,
             json.dumps(benchmark, default=str) if benchmark is not None else None,
             mlflow_run_id, notes))
        _event(conn, vid, "register", reason or notes,
               {"spec_hash": sh, "code_sha": code, "features": len(feature_list or []),
                "data_fingerprint": data_fp}, ts)
        if status == "champion":
            _promote(conn, vid, reason or "registered as champion", None)
        elif status != "candidate":
            _set_status(conn, vid, status, reason, None, None)
    return vid


def _set_status(conn, version_id, status, reason, action, details) -> None:
    row = get_version(conn, version_id)
    if row is None:
        raise RegistryError(f"unknown version {version_id!r}")
    if status == "champion":
        raise RegistryError("use promote() to make a champion")
    if status not in STATUSES:
        raise RegistryError(f"unknown status {status!r}")
    if row["status"] == status:
        return
    ts = _now()
    retired_at = ts if status == "retired" else None
    conn.execute("UPDATE model_versions SET status=?, retired_at=? WHERE version_id=?",
                 (status, retired_at, version_id))
    action = action or {"shadow": "shadow", "retired": "retire", "candidate": "register"}[status]
    _event(conn, version_id, action, reason, {"from": row["status"], "to": status,
                                              **(details or {})}, ts)


def set_status(conn, version_id: str, status: str, reason: str | None = None, *,
               action: str | None = None, details: dict | None = None) -> None:
    """candidate | shadow | retired (not champion: see promote). `action` overrides the logged
    event (e.g. "reject" for a retirement by a gate). Retiring the champion leaves the model
    without one (production then falls back to gbm.FROZEN); prefer promote/rollback."""
    with _tx(conn):
        _set_status(conn, version_id, status, reason, action, details)


def _promote(conn, version_id, reason, details, action: str = "promote") -> str | None:
    row = get_version(conn, version_id)
    if row is None:
        raise RegistryError(f"unknown version {version_id!r}")
    if row["status"] == "champion":
        return None
    name = row["model_name"]
    old = get_champion(conn, name)
    ts = _now()
    if old is not None:
        conn.execute("UPDATE model_versions SET status='retired', retired_at=? "
                     "WHERE version_id=?", (ts, old["version_id"]))
        _event(conn, old["version_id"], "retire", f"replaced by {version_id}: {reason}",
               {"from": "champion", "to": "retired", "replaced_by": version_id}, ts)
    conn.execute("UPDATE model_versions SET status='champion', promoted_at=?, retired_at=NULL "
                 "WHERE version_id=?", (ts, version_id))
    _event(conn, version_id, action, reason,
           {"from": row["status"], "previous_champion": old["version_id"] if old else None,
            **(details or {})}, ts)
    _check_one_champion(conn, name)
    return old["version_id"] if old else None


def promote(conn, version_id: str, reason: str, *, details: dict | None = None) -> str | None:
    """Make `version_id` the champion of its model: atomically retire the current champion,
    set the new one and log both events. Returns the previous champion's id (None if none).
    Promoting the current champion is a no-op."""
    with _tx(conn):
        return _promote(conn, version_id, reason, details)


def rollback(conn, model_name: str, reason: str) -> str:
    """Restore the most recently retired *former champion* of `model_name` (rejected
    candidates/shadows never were champions and are skipped). Returns its id."""
    with _tx(conn):
        cur = get_champion(conn, model_name)
        args: list = [model_name]
        sql = ("SELECT version_id FROM model_versions WHERE model_name=? AND status='retired' "
               "AND promoted_at IS NOT NULL")
        if cur is not None:
            sql += " AND version_id != ?"
            args.append(cur["version_id"])
        r = conn.execute(sql + " ORDER BY retired_at DESC, rowid DESC LIMIT 1", args).fetchone()
        if r is None:
            raise RegistryError(f"{model_name}: no retired champion to roll back to")
        _promote(conn, r[0], reason,
                 {"rolled_back_from": cur["version_id"] if cur else None}, action="rollback")
        return r[0]


# ------------------------------------------------------------------ bootstrap

# Champion v1 benchmarks: the prior full rung-3 walk-forward (2012-13 -> 2025-26, inner search
# over gbm.GRID per fold; run 2026-10-18 on the Mac, cache/full2/rung3_merged*), summarised by
# retrain.summarize over all 18,109 test games; vs-market rows from the same run's report
# (= docs/MODEL.md, rung 3). That run predates the season-opening rolling-prior fix
# (commit 2aa9a89), so a challenger is always compared with the champion recomputed on the
# current code, not with these numbers.
BOOTSTRAP_SOURCE = ("rung-3 walk-forward 2012-13..2025-26 (inner search over gbm.GRID per "
                    "fold), cache/full2/rung3_merged_preds.csv.gz, 2026-10-18; docs/MODEL.md")
BOOTSTRAP_BENCHMARKS: dict[str, dict] = {
    'gbm_market': {
        'seasons': ['2012-13', '2025-26'],
        'all': {'n': 18109, 'log_loss': 0.5953, 'brier': 0.20492, 'accuracy': 0.67972, 'ece': 0.01441, 'margin_mae': 9.98344},
        'regular_season': {'n': 16898, 'log_loss': 0.59329, 'brier': 0.20408, 'accuracy': 0.6812, 'ece': 0.01298, 'margin_mae': 9.90737},
        'playoffs': {'n': 1211, 'log_loss': 0.62332, 'brier': 0.21666, 'accuracy': 0.65896, 'ece': 0.04223},
        'last_season': {'season': '2025-26', 'n': 1321, 'log_loss': 0.5806, 'brier': 0.19911, 'accuracy': 0.68282, 'ece': 0.02588},
        'by_season_log_loss': {
            '2012-13': 0.58221,
            '2013-14': 0.59228,
            '2014-15': 0.57915,
            '2015-16': 0.5689,
            '2016-17': 0.60752,
            '2017-18': 0.59458,
            '2018-19': 0.59689,
            '2019-20': 0.61322,
            '2020-21': 0.61988,
            '2021-22': 0.60588,
            '2022-23': 0.62793,
            '2023-24': 0.58427,
            '2024-25': 0.58587,
            '2025-26': 0.5806,
        },
        'calibration': [
            {'lo': 0.0, 'hi': 0.1, 'n': 59, 'mean_p': 0.0813, 'frac_pos': 0.0339},
            {'lo': 0.1, 'hi': 0.2, 'n': 556, 'mean_p': 0.1596, 'frac_pos': 0.1583},
            {'lo': 0.2, 'hi': 0.3, 'n': 1403, 'mean_p': 0.2556, 'frac_pos': 0.2302},
            {'lo': 0.3, 'hi': 0.4, 'n': 2156, 'mean_p': 0.3515, 'frac_pos': 0.3734},
            {'lo': 0.4, 'hi': 0.5, 'n': 2244, 'mean_p': 0.4441, 'frac_pos': 0.4496},
            {'lo': 0.5, 'hi': 0.6, 'n': 2342, 'mean_p': 0.5576, 'frac_pos': 0.5444},
            {'lo': 0.6, 'hi': 0.7, 'n': 3380, 'mean_p': 0.6506, 'frac_pos': 0.6311},
            {'lo': 0.7, 'hi': 0.8, 'n': 3306, 'mean_p': 0.7483, 'frac_pos': 0.7377},
            {'lo': 0.8, 'hi': 0.9, 'n': 2208, 'mean_p': 0.8454, 'frac_pos': 0.8347},
            {'lo': 0.9, 'hi': 1.0, 'n': 455, 'mean_p': 0.9252, 'frac_pos': 0.9407},
        ],
        'vs_market': {
            'regular_season': {'n': 16321, 'log_loss': 0.59286, 'market_log_loss': 0.59183, 'll_minus_market': {'n': 16321, 'diff': 0.00102, 'se': 0.00051}},
            'regular_season_last3': {'n': 3687, 'log_loss': 0.57849, 'market_log_loss': 0.57845, 'll_minus_market': {'n': 3687, 'diff': 3e-05, 'se': 0.00093}},
            'playoffs': {'n': 1090, 'log_loss': 0.62154, 'market_log_loss': 0.61593, 'll_minus_market': {'n': 1090, 'diff': 0.00561, 'se': 0.00315}},
        },
    },
    'gbm_opinion': {
        'seasons': ['2012-13', '2025-26'],
        'all': {'n': 18109, 'log_loss': 0.60495, 'brier': 0.2091, 'accuracy': 0.66906, 'ece': 0.00583, 'margin_mae': 10.14804},
        'regular_season': {'n': 16898, 'log_loss': 0.60219, 'brier': 0.2079, 'accuracy': 0.67162, 'ece': 0.0054, 'margin_mae': 10.06153},
        'playoffs': {'n': 1211, 'log_loss': 0.64347, 'brier': 0.22574, 'accuracy': 0.63336, 'ece': 0.04972},
        'last_season': {'season': '2025-26', 'n': 1321, 'log_loss': 0.59039, 'brier': 0.20288, 'accuracy': 0.68584, 'ece': 0.02948},
        'by_season_log_loss': {
            '2012-13': 0.59228,
            '2013-14': 0.59827,
            '2014-15': 0.59381,
            '2015-16': 0.57611,
            '2016-17': 0.615,
            '2017-18': 0.60963,
            '2018-19': 0.60745,
            '2019-20': 0.62233,
            '2020-21': 0.63008,
            '2021-22': 0.61711,
            '2022-23': 0.63549,
            '2023-24': 0.59334,
            '2024-25': 0.59303,
            '2025-26': 0.59039,
        },
        'calibration': [
            {'lo': 0.0, 'hi': 0.1, 'n': 68, 'mean_p': 0.0745, 'frac_pos': 0.0735},
            {'lo': 0.1, 'hi': 0.2, 'n': 495, 'mean_p': 0.1558, 'frac_pos': 0.1374},
            {'lo': 0.2, 'hi': 0.3, 'n': 1095, 'mean_p': 0.2532, 'frac_pos': 0.2548},
            {'lo': 0.3, 'hi': 0.4, 'n': 1825, 'mean_p': 0.354, 'frac_pos': 0.3567},
            {'lo': 0.4, 'hi': 0.5, 'n': 2654, 'mean_p': 0.4501, 'frac_pos': 0.4446},
            {'lo': 0.5, 'hi': 0.6, 'n': 3323, 'mean_p': 0.5515, 'frac_pos': 0.5504},
            {'lo': 0.6, 'hi': 0.7, 'n': 3283, 'mean_p': 0.6457, 'frac_pos': 0.6442},
            {'lo': 0.7, 'hi': 0.8, 'n': 3014, 'mean_p': 0.75, 'frac_pos': 0.7376},
            {'lo': 0.8, 'hi': 0.9, 'n': 1931, 'mean_p': 0.8457, 'frac_pos': 0.8327},
            {'lo': 0.9, 'hi': 1.0, 'n': 421, 'mean_p': 0.9293, 'frac_pos': 0.9192},
        ],
        'vs_market': {
            'regular_season': {'n': 16321, 'log_loss': 0.6017, 'market_log_loss': 0.59183, 'll_minus_market': {'n': 16321, 'diff': 0.00986, 'se': 0.00126}},
            'regular_season_last3': {'n': 3687, 'log_loss': 0.587, 'market_log_loss': 0.57845, 'll_minus_market': {'n': 3687, 'diff': 0.00855, 'se': 0.00254}},
            'playoffs': {'n': 1090, 'log_loss': 0.64301, 'market_log_loss': 0.61593, 'll_minus_market': {'n': 1090, 'diff': 0.02708, 'se': 0.00693}},
        },
    },
}


def bootstrap(conn, benchmarks: dict | None = None, *, log_mlflow: bool = True,
              log_fn: Callable[[str], None] = print) -> list[dict]:
    """Register today's production specs (gbm.FROZEN params, gbm.GRID search, feature lists)
    as champion v1 with their benchmark metrics. Idempotent: a model that already has a
    champion is left alone. Returns [{model_name, version_id, created}]."""
    from . import tracking
    benchmarks = benchmarks if benchmarks is not None else BOOTSTRAP_BENCHMARKS
    out = []
    for name in production_names():
        cur = get_champion(conn, name)
        if cur is not None:
            out.append({"model_name": name, "version_id": cur["version_id"], "created": False})
            log_fn(f"{name}: champion {cur['version_id']} (unchanged)")
            continue
        params, feats = production_params(name), feature_list(name)
        bench = benchmarks.get(name)
        bench = {"source": BOOTSTRAP_SOURCE, **bench} if bench else None
        run_id = None
        if log_mlflow and bench:
            run_id = tracking.log_benchmark(
                f"bootstrap/{name}", {"model_name": name, "params": params,
                                      "features": ",".join(feats)},
                {k: v for k, v in bench.items() if k not in ("calibration", "source")},
                {"benchmark": bench}, tags={"kind": "bootstrap", "model_name": name})
        vid = register(conn, name, params, feats, status="champion", benchmark=bench,
                       mlflow_run_id=run_id, notes="bootstrap: production spec (gbm.FROZEN)",
                       reason="bootstrap: current production spec")
        out.append({"model_name": name, "version_id": vid, "created": True,
                    "mlflow_run_id": run_id})
        log_fn(f"{name}: registered champion {vid}" + (f" (mlflow {run_id})" if run_id else ""))
    return out
