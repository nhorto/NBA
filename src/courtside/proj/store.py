"""Projection store (docs/PLAYERS.md §6, T1.4): `player_projections`, `player_projection_stats`,
per-sim DK FP files, and the `dfs_runs.sim_path` pointer.

* `player_projections`: one row per player per run (game_id, nba_day, slate_id, player_id, team,
  run_id, model_version, role = copula tier, p_play = simulated P(plays)), `minutes_json`
  (minutes summary) and `inputs_json` (rates, environment, scenario facts).
* `player_projection_stats`: (projection_id, stat, cond) with stat in {min, dk} from the sims
  (mean, sd, q05..q95) and {pts, reb, ast, fg3m} means (rates x minutes, environment-scaled;
  sd/quantiles NULL until v1 draws components), cond in {played, all}.
* Sims: float16 `npz` at `<data_dir>/sims/<slate>/<run_id>.npz` with `fp` (S, n), `player_id`
  (NBA ids) and optional `dk_id`; files older than 8 days are pruned on every save.

Writing a run replaces that (run_id, model_version)'s earlier rows, so re-running is idempotent.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

KEEP_DAYS = 8
STAT_COLS = ("mean", "sd", "q05", "q10", "q25", "q50", "q75", "q90", "q95")
PROJ_COLS = ("game_id", "nba_day", "slate_id", "player_id", "team", "run_id", "model_version",
             "role", "p_play", "minutes_json", "inputs_json")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def sims_dir() -> Path:
    return config.data_dir() / "sims"


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ============================================================================== sims

def save_sims(slate: str, run_id: str, player_id, fp: np.ndarray, *, dk_id=None,
              game_id=None, keep_days: int = KEEP_DAYS, root: Path | None = None) -> Path:
    """Per-sim DK FP (S, n) as float16 npz; returns the path. Prunes old files."""
    root = root or sims_dir()
    d = root / _safe(slate)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_safe(run_id)}.npz"
    arrs = {"fp": np.asarray(fp, np.float16),
            "player_id": np.asarray(player_id, np.int64)}
    if dk_id is not None:
        arrs["dk_id"] = np.asarray([str(x) if x is not None else "" for x in dk_id])
    if game_id is not None:
        arrs["game_id"] = np.asarray([str(x) for x in game_id])
    np.savez_compressed(path, **arrs)
    prune_sims(keep_days, root=root)
    return path


def load_sims(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prune_sims(keep_days: int = KEEP_DAYS, *, now: float | None = None,
               root: Path | None = None) -> int:
    """Delete sim files older than `keep_days` (mtime) and empty slate dirs; returns count."""
    root = root or sims_dir()
    if not root.exists():
        return 0
    cut = (now if now is not None else time.time()) - keep_days * 86400
    n = 0
    for f in root.glob("*/*.npz"):
        if f.stat().st_mtime < cut:
            f.unlink()
            n += 1
    for d in root.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return n


# ============================================================================== tables

def write_projections(conn: sqlite3.Connection, proj: pd.DataFrame, stats: pd.DataFrame,
                      *, created_at: str | None = None) -> int:
    """Insert one `player_projections` row per `proj` row (PROJ_COLS; minutes_json /
    inputs_json may be dicts) and its `player_projection_stats` rows. `stats` has a `key`
    column = the index label of its `proj` row, plus stat, cond and STAT_COLS. Earlier rows of
    the same (run_id, model_version) are replaced. Returns the number of projections."""
    if proj.empty:
        return 0
    created_at = created_at or _now()
    runs = proj[["run_id", "model_version"]].drop_duplicates().itertuples(index=False)
    by_key = {k: g for k, g in stats.groupby("key", sort=False)} if len(stats) else {}
    with conn:
        for run_id, mv in runs:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM player_projections WHERE run_id IS ? AND model_version IS ?",
                (run_id, mv))]
            if ids:
                q = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM player_projection_stats WHERE projection_id IN ({q})",
                             ids)
                conn.execute(f"DELETE FROM player_projections WHERE id IN ({q})", ids)
        for key, r in proj.iterrows():
            vals = []
            for c in PROJ_COLS:
                v = r.get(c)
                if c in ("minutes_json", "inputs_json") and not isinstance(v, (str, type(None))):
                    v = json.dumps(v, default=_json_default)
                elif c == "player_id":
                    v = int(v)
                elif c == "p_play":
                    v = _num(v)
                elif v is not None and not isinstance(v, str):
                    v = None if (isinstance(v, float) and not np.isfinite(v)) else str(v)
                vals.append(v)
            cur = conn.execute(
                f"INSERT INTO player_projections ({', '.join(PROJ_COLS)}, created_at) "
                f"VALUES ({', '.join('?' * (len(PROJ_COLS) + 1))})", (*vals, created_at))
            pid = cur.lastrowid
            g = by_key.get(key)
            if g is None:
                continue
            conn.executemany(
                f"INSERT INTO player_projection_stats (projection_id, stat, cond, "
                f"{', '.join(STAT_COLS)}) VALUES (?, ?, ?, {', '.join('?' * len(STAT_COLS))})",
                [(pid, s.stat, s.cond, *(_num(getattr(s, c, None)) for c in STAT_COLS))
                 for s in g.itertuples(index=False)])
    return len(proj)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def read_projections(conn: sqlite3.Connection, run_id: str,
                     model_version: str | None = None) -> pd.DataFrame:
    """Projections of a run with their stats, wide: <stat>_<cond>_<col> columns."""
    sql = "SELECT * FROM player_projections WHERE run_id = ?"
    args: list = [run_id]
    if model_version is not None:
        sql += " AND model_version = ?"
        args.append(model_version)
    p = pd.read_sql_query(sql, conn, params=args)
    if p.empty:
        return p
    q = ",".join("?" * len(p))
    st = pd.read_sql_query(f"SELECT * FROM player_projection_stats WHERE projection_id IN ({q})",
                           conn, params=[int(x) for x in p["id"]])
    if len(st):
        st["k"] = st["stat"] + "_" + st["cond"]
        w = st.pivot_table(index="projection_id", columns="k", values=list(STAT_COLS),
                           aggfunc="first", dropna=False)
        w.columns = [f"{k}_{c}" for c, k in w.columns]
        p = p.merge(w, left_on="id", right_index=True, how="left")
    return p


def record_run(conn: sqlite3.Connection, run_id: str, *, slate_id: str | None, kind: str,
               model_version: str, sim_path: str | None, seconds: float | None = None,
               detail: dict | None = None, path: str = "A") -> None:
    """Upsert `dfs_runs(run_id)`: creates the row (kind, created_at) if missing, and sets
    model_version, sim_path, path and seconds (detail only when the row is new)."""
    with conn:
        conn.execute(
            "INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, sim_path, "
            "path, seconds, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET model_version=excluded.model_version, "
            "sim_path=excluded.sim_path, path=excluded.path, "
            "seconds=COALESCE(dfs_runs.seconds, excluded.seconds)",
            (run_id, slate_id, kind, _now(), model_version, sim_path, path, seconds,
             json.dumps(detail, default=_json_default) if detail else None))
