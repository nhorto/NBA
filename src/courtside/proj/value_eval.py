"""E4: DFS-relevant evaluation populations and the value-ranking metric on the live grades
(docs/IMPROVEMENTS.md E4). Report-only: reads ``player_projection_grades`` and never writes.

Population (`live_frame`): the graded projections of DK Classic slates (game type 70), one row per
(game, player): the main slate first, then the run kinds lock10 > lock45 > swap (newest first),
manual runs only on nights with no scheduled run (the weekly report's rule, `reports.dedupe`).
DK-FPPG fallback runs and RotoGuru ``rg:`` slates are left out. DNP scores 0.

Cuts (`live_report`):

* ``grader``: projected minutes >= 10 (the nightly grader's population);
* ``dfs``: salary >= $4k or projected >= 15 DK by us or by DK FPPG (`metrics.dfs_population`),
  players DK listed OUT at the run's salary capture removed (nobody rosters them; the grader's
  baselines drop them too).

Per cut: DK MAE ours vs DK FPPG (paired on the same rows, SE clustered by date and by game),
our CRPS, and the value-ranking metric corr(forecast - salary-implied, actual - salary-implied)
with the salary-implied line an OLS of actual DK on salary per slate (`metrics.value_ranking`),
ours vs DK FPPG paired, SE clustered by date (and by game as a second reading).

CLI: ``courtside proj value-eval [--from 2026-10-20] [--to DAY] [--json]``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from .. import config
from . import metrics as MT

VERSION = "value-eval-v1"
CLASSIC = 70
SEASON_START = "2026-10-20"
KIND_PRIORITY = {"lock10": 0, "lock45": 1, "swap": 2, "manual": 3}
OUT_STATUSES = frozenset({"OUT", "O"})


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _dk_id(s) -> str | None:
    try:
        v = json.loads(s) if s else {}
    except (TypeError, ValueError):
        return None
    d = v.get("dk_id") if isinstance(v, dict) else None
    return None if d is None else str(d)


def _status_lookup(conn, frame: pd.DataFrame) -> np.ndarray:
    """DK status of each row's player at the salary capture at or before its run (first capture
    if none is earlier)."""
    out = np.full(len(frame), None, dtype=object)
    cache: dict = {}
    caps: dict = {}
    for i, (sid, ts, dk) in enumerate(zip(frame["slate_id"], frame["created_at"], frame["dk_id"])):
        if sid is None or dk is None:
            continue
        sid = str(sid)
        if sid not in caps:
            caps[sid] = [r[0] for r in conn.execute(
                "SELECT DISTINCT captured_at FROM dfs_salaries WHERE slate_id=? ORDER BY "
                "captured_at", (sid,))]
        c = caps[sid]
        if not c:
            continue
        pick = max((x for x in c if isinstance(ts, str) and x <= ts), default=c[0])
        key = (sid, pick)
        if key not in cache:
            cache[key] = {str(r[0]): r[1] for r in conn.execute(
                "SELECT site_player_id, status FROM dfs_salaries WHERE slate_id=? AND "
                "captured_at=?", (sid, pick)) if r[0] is not None}
        out[i] = cache[key].get(dk)
    return out


def live_frame(conn: sqlite3.Connection, lo: str, hi: str) -> pd.DataFrame:
    """Graded Classic-slate projections of [lo, hi], one row per (game, player) (see module doc),
    with ``dk_status`` at the run's capture."""
    try:
        df = pd.read_sql_query(
            "SELECT g.projection_id, g.game_id, g.player_id, g.run_id, g.played, g.actual_dk, "
            "g.proj_min, g.proj_dk, g.crps, g.dk_fppg, g.salary, pp.nba_day AS day, pp.slate_id, "
            "pp.model_version, pp.inputs_json, r.kind, r.created_at, s.tag "
            "FROM player_projection_grades g JOIN player_projections pp ON pp.id=g.projection_id "
            "LEFT JOIN dfs_runs r ON r.run_id=g.run_id "
            "LEFT JOIN dfs_slates s ON s.slate_id=pp.slate_id "
            "WHERE pp.nba_day BETWEEN ? AND ? AND s.game_type_id=? "
            "AND COALESCE(pp.slate_id, '') NOT LIKE 'rg:%'", conn, params=(lo, hi, CLASSIC))
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    if df.empty:
        return df
    mv = df["model_version"].fillna("").str.lower()
    df = df[~(mv.str.startswith("fppg") | mv.str.contains("fallback"))].copy()
    df["kind"] = df["kind"].fillna("manual")
    sched = set(df.loc[df["kind"].isin(("lock10", "lock45", "swap")), "day"])
    df = df[(df["kind"] != "manual") | ~df["day"].isin(sched)].copy()
    df["prio"] = df["kind"].map(KIND_PRIORITY).fillna(2)
    df["not_main"] = (df["tag"].fillna("") != "main").astype(int)
    df = df.sort_values(["not_main", "prio", "created_at"], ascending=[True, True, False],
                        na_position="last")
    df = df.drop_duplicates(["game_id", "player_id"]).reset_index(drop=True)
    df["dk_id"] = [_dk_id(s) for s in df["inputs_json"]]
    df["dk_status"] = _status_lookup(conn, df)
    return df.drop(columns=["inputs_json"])


def populations(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """{cut name: boolean mask} (see the module doc)."""
    pm = df["proj_min"].to_numpy(float)
    out = np.array([str(s or "").upper() in OUT_STATUSES for s in df["dk_status"]])
    dfs = MT.dfs_population(df["salary"].to_numpy(float), df["proj_dk"].to_numpy(float),
                            df["dk_fppg"].to_numpy(float)) & ~out
    return {"grader": np.isfinite(pm) & (pm >= 10), "dfs": dfs}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def cut_report(d: pd.DataFrame) -> dict:
    """Numbers of one cut (rows with our projection and the actual)."""
    d = d[d["proj_dk"].notna() & d["actual_dk"].notna()]
    out: dict = {"n": int(len(d)), "nights": int(d["day"].nunique()),
                 "slates": int(d["slate_id"].nunique()), "games": int(d["game_id"].nunique())}
    if d.empty:
        return out
    y = d["actual_dk"].to_numpy(float)
    ours = d["proj_dk"].to_numpy(float)
    out["dk_mae"] = _f(np.abs(ours - y).mean())
    out["dk_bias"] = _f((ours - y).mean())
    c = d["crps"].to_numpy(float)
    if np.isfinite(c).any():
        pc = MT.paired_cluster(c[np.isfinite(c)], d["day"].to_numpy()[np.isfinite(c)])
        out["dk_crps"] = {"mean": pc["mean"], "se_date": pc["se"], "n": pc["n"]}
    p = d[d["dk_fppg"].notna() & d["salary"].notna()]
    if len(p):
        yy = p["actual_dk"].to_numpy(float)
        eo = np.abs(p["proj_dk"].to_numpy(float) - yy)
        ef = np.abs(p["dk_fppg"].to_numpy(float) - yy)
        dd = MT.paired_cluster(eo - ef, p["day"].to_numpy())
        dg = MT.paired_cluster(eo - ef, p["game_id"].to_numpy())
        out["mae_vs_fppg"] = {"n": int(len(p)), "mae_ours": _f(eo.mean()), "mae_fppg": _f(ef.mean()),
                              "diff": dd["mean"], "se_date": dd["se"], "se_game": dg["se"],
                              "n_games": dg["n_clusters"]}
        v = MT.value_ranking({"ours": p["proj_dk"].to_numpy(float),
                              "fppg": p["dk_fppg"].to_numpy(float)}, yy,
                             p["salary"].to_numpy(float), p["slate_id"].astype(str).to_numpy(),
                             p["day"].to_numpy(), base="fppg")
        # the same paired difference clustered by game (a second reading with more clusters)
        s = MT.salary_implied(yy, p["salary"].to_numpy(float), p["slate_id"].astype(str).to_numpy())
        ok = np.isfinite(s)
        g = MT.corr_paired_cluster(p["proj_dk"].to_numpy(float)[ok] - s[ok],
                                   p["dk_fppg"].to_numpy(float)[ok] - s[ok], yy[ok] - s[ok],
                                   p["game_id"].to_numpy()[ok])
        if "ours" in v["vs_base"]:
            v["vs_base"]["ours"]["se_game"] = g["se"]
            v["vs_base"]["ours"]["n_games"] = g["n_clusters"]
        out["value"] = v
    return out


def live_report(conn: sqlite3.Connection, lo: str = SEASON_START, hi: str | None = None) -> dict:
    hi = hi or config.yesterday_nba_day().isoformat()
    df = live_frame(conn, lo, hi)
    rep: dict = {"version": VERSION, "from": lo, "to": hi, "rows": int(len(df)), "cuts": {},
                 "by_night": {}}
    if df.empty:
        return rep
    pops = populations(df)
    rep["nights"] = sorted(df["day"].unique().tolist())
    for name, m in pops.items():
        rep["cuts"][name] = cut_report(df[m])
    m = pops["dfs"]
    for day, x in df[m].groupby("day"):
        r = cut_report(x)
        vb = (r.get("value") or {}).get("vs_base", {}).get("ours", {})
        rep["by_night"][day] = {"n": r["n"], "r_ours": vb.get("r_a"), "r_fppg": vb.get("r_b"),
                                "mae_ours": (r.get("mae_vs_fppg") or {}).get("mae_ours"),
                                "mae_fppg": (r.get("mae_vs_fppg") or {}).get("mae_fppg")}
    return _clean(rep)


def _n(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def format_report(rep: dict) -> str:
    L = [f"E4 value ranking on live grades ({rep['version']}): {rep['from']} .. {rep['to']}, "
         f"{len(rep.get('nights') or [])} nights"]
    for name, c in rep.get("cuts", {}).items():
        L.append(f"  [{name}] n {c['n']}, {c['nights']} nights, {c['slates']} slates, "
                 f"{c['games']} games")
        mv = c.get("mae_vs_fppg")
        if mv:
            L.append(f"    DK MAE ours {_n(mv['mae_ours'], 2)} vs FPPG {_n(mv['mae_fppg'], 2)}: "
                     f"diff {_n(mv['diff'], 2)} (SE date {_n(mv['se_date'], 2)}, "
                     f"game {_n(mv['se_game'], 2)}; n {mv['n']})")
        if c.get("dk_crps"):
            L.append(f"    DK CRPS ours {_n(c['dk_crps']['mean'], 2)} "
                     f"(SE {_n(c['dk_crps']['se_date'], 2)})")
        v = (c.get("value") or {}).get("vs_base", {}).get("ours")
        if v:
            L.append(f"    value corr ours {_n(v['r_a'])} (SE {_n(v['se_a'])}) vs FPPG "
                     f"{_n(v['r_b'])} (SE {_n(v['se_b'])}): diff {_n(v['diff'])} "
                     f"(SE date {_n(v['se'])}, game {_n(v.get('se_game'))}), n {v['n']}, "
                     f"{v['n_clusters']} nights")
            f = c["value"]["forecasts"]
            L.append(f"    per-slate mean r: ours {_n(f['ours']['slate_mean_r'])}, FPPG "
                     f"{_n(f['fppg']['slate_mean_r'])} over {f['ours']['n_slates_r']} slates")
    for day, r in (rep.get("by_night") or {}).items():
        L.append(f"  {day}: n {r['n']}, value r ours {_n(r['r_ours'])} / FPPG {_n(r['r_fppg'])}, "
                 f"MAE {_n(r['mae_ours'], 2)} / {_n(r['mae_fppg'], 2)}")
    return "\n".join(L)


def add_commands(sub) -> dict:
    p = sub.add_parser("value-eval", help="E4: DFS-relevant population and the value-ranking "
                                          "metric (ours vs DK FPPG) on the live grades")
    p.add_argument("--from", dest="lo", default=SEASON_START)
    p.add_argument("--to", dest="hi", default=None)
    p.add_argument("--db", default=None, help="database path (opened read-only)")
    p.add_argument("--json", action="store_true")
    return {"value-eval": _cmd}


def _cmd(args) -> int:
    from pathlib import Path
    path = Path(args.db) if args.db else config.db_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        rep = live_report(conn, args.lo, args.hi)
    finally:
        conn.close()
    print(json.dumps(rep, indent=1) if args.json else format_report(rep))
    return 0
