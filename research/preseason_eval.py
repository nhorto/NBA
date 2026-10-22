"""Preseason arm (COURTSIDE_PRESEASON=1: `minutes.V1_PRESEASON` in the minutes LightGBM + the
season-start debut reserve) vs OFF: paired evaluation (item #2).

Reads per-player projection rows of both arms of the T2.1 harness (`proj/backtest.py`, avail
protocol, 4,000 sims, closing lines, bundles fit on seasons < S; same seeds, so every difference
is the minutes engine):

    RAW/rows_avail_<S>_off.parquet / rows_avail_<S>_on.parquet

and cuts them by team games into the season (from `team_games`), tier, and player type:
rookie (no regular-season box row before the season), new team (his latest earlier
regular-season team differs), returning (missed >= 3 team games right before this game).

Population (as `backtest.player_report`): projected rows with OFF E[min] >= 10, DNP scored 0;
"all" = every projected row. Every difference is ON - OFF on identical rows, SE clustered by date.
Metrics: minutes MAE (E[min] incl. P(play) vs actual), DK MAE, CRPS, bias, 80 % coverage.

    python research/preseason_eval.py RAW_DIR [--db var/courtside.db] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["date", "player_id", "game_id"]
P_MIN = 0.05
TIERS = (("star", 30, 99), ("starter", 24, 30), ("rotation", 15, 24), ("bench", 0, 15))
EARLY = 15


def cl_mean(z, cl) -> tuple[float, float]:
    z = np.asarray(z, float)
    n = len(z)
    if n < 2:
        return float("nan"), float("nan")
    m = float(z.mean())
    s = pd.Series(z - m).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    return m, float(np.sqrt(g / max(g - 1, 1) * (s ** 2).sum()) / n)


def context_rows(conn: sqlite3.Connection, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team game index into the season per (game_id, team), player flags per (game_id,
    player_id): rookie, new_team, returning)."""
    tg = pd.read_sql_query(
        "SELECT game_id, team, game_date FROM team_games WHERE season=? AND "
        "season_type='Regular Season' ORDER BY team, game_date, game_id", conn, params=(season,))
    tg["tgs"] = tg.groupby("team").cumcount()
    pg = pd.read_sql_query(
        "SELECT game_id, player_id, team, season, game_date FROM player_games WHERE "
        "season_type='Regular Season' AND min > 0 AND season <= ?", conn, params=(season,))
    prev = pg[pg["season"] < season].sort_values("game_date")
    last_team = prev.groupby("player_id")["team"].last()
    cur = pg[pg["season"] == season].merge(tg[["game_id", "team", "tgs"]], on=["game_id", "team"])
    cur = cur.sort_values(["player_id", "game_date"])
    cur["rookie"] = ~cur["player_id"].isin(last_team.index)
    lt = last_team.reindex(cur["player_id"]).to_numpy()
    cur["new_team"] = (~cur["rookie"]) & (lt != cur["team"].to_numpy())
    # returning: team games missed right before this appearance (same team, same season)
    cur["prev_tgs"] = cur.groupby(["player_id", "team"])["tgs"].shift(1)
    gap = cur["tgs"] - cur["prev_tgs"] - 1
    cur["returning"] = (gap >= 3).fillna(False)
    return tg[["game_id", "team", "tgs"]], cur[["game_id", "player_id", "rookie", "new_team",
                                               "returning"]]


def paired(off: pd.DataFrame, on: pd.DataFrame, tg, flags) -> pd.DataFrame:
    cols = ["y", "minutes", "has_proj", "mean", "q10", "q90", "exp_min", "p_play", "crps",
            "team"]
    a = off[KEY + cols].drop_duplicates(KEY)
    b = on[KEY + ["mean", "q10", "q90", "crps", "exp_min", "p_play"]].drop_duplicates(KEY)
    d = a.merge(b, on=KEY, suffixes=("_off", "_on"), how="inner")
    d = d[d["has_proj"].astype(bool)].reset_index(drop=True)
    d = d.merge(tg, on=["game_id", "team"], how="left")
    d = d.merge(flags, on=["game_id", "player_id"], how="left")
    for c in ("rookie", "new_team", "returning"):
        d[c] = d[c].fillna(False).astype(bool)
    d["min_c"] = d["exp_min_off"] / d["p_play_off"].clip(lower=P_MIN)
    d["tier4"] = "bench"
    for name, lo, hi in TIERS:
        d.loc[(d["min_c"] >= lo) & (d["min_c"] < hi), "tier4"] = name
    d["pop"] = d["exp_min_off"] >= 10
    return d


def metrics(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(float)
    mins = d["minutes"].to_numpy(float)
    cl = d["date"].to_numpy()
    out = {"n": int(len(d)), "dates": int(d["date"].nunique())}
    fns = {"min_mae": lambda a: np.abs(mins - d[f"exp_min_{a}"].to_numpy(float)),
           "min_bias": lambda a: d[f"exp_min_{a}"].to_numpy(float) - mins,
           "dk_mae": lambda a: np.abs(y - d[f"mean_{a}"].to_numpy(float)),
           "crps": lambda a: d[f"crps_{a}"].to_numpy(float),
           "dk_bias": lambda a: d[f"mean_{a}"].to_numpy(float) - y,
           "cov80": lambda a: ((d[f"q10_{a}"].to_numpy(float) <= y)
                               & (y <= d[f"q90_{a}"].to_numpy(float))).astype(float)}
    for k, f in fns.items():
        z0, z1 = f("off"), f("on")
        ok = np.isfinite(z0) & np.isfinite(z1)
        m0, _ = cl_mean(z0[ok], cl[ok])
        dd, se = cl_mean(z1[ok] - z0[ok], cl[ok])
        out[k] = {"off": m0, "on": float(z1[ok].mean()) if ok.any() else float("nan"),
                  "diff": dd, "se": se, "z": dd / se if se and se > 0 else float("nan")}
    return out


def cuts(d: pd.DataFrame) -> dict:
    early = d["tgs"] < EARLY
    out = {}
    for pop_name, pm in (("pop", d["pop"]), ("all", pd.Series(True, index=d.index))):
        for span, sm in (("first15", early), ("full", pd.Series(True, index=d.index)),
                         ("game0", d["tgs"] == 0), ("games1-4", d["tgs"].between(1, 4)),
                         ("games5-14", d["tgs"].between(5, 14))):
            out[f"{pop_name}|{span}"] = metrics(d[pm & sm])
    for t, _, _ in TIERS:
        for span, sm in (("first15", early), ("full", pd.Series(True, index=d.index))):
            out[f"tier {t}|{span}"] = metrics(d[d["pop"] & sm & (d["tier4"] == t)])
    for flag in ("rookie", "new_team", "returning"):
        for span, sm in (("first15", early), ("full", pd.Series(True, index=d.index))):
            out[f"{flag}|{span}"] = metrics(d[d["pop"] & sm & d[flag]])
    return out


def fmt(name: str, rep: dict) -> list[str]:
    lines = [f"== {name}", f"{'cut':<22} {'n':>6} {'dates':>5} | "
             + " | ".join(f"{k:>24}" for k in ("min_mae", "dk_mae", "crps", "cov80"))]
    for cut, m in rep.items():
        if m["n"] < 2:
            continue
        cells = []
        for k in ("min_mae", "dk_mae", "crps", "cov80"):
            v = m[k]
            cells.append(f"{v['off']:7.3f} {v['diff']:+7.4f} ({v['z']:+5.1f})")
        lines.append(f"{cut:<22} {m['n']:>6} {m['dates']:>5} | " + " | ".join(
            f"{c:>24}" for c in cells))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--db", default="var/courtside.db")
    ap.add_argument("--json", default="")
    ap.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25", "2025-26"])
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    raw = Path(a.raw)
    reps = {}
    for s in a.seasons:
        fo, fn = raw / f"rows_avail_{s}_off.parquet", raw / f"rows_avail_{s}_on.parquet"
        if not (fo.exists() and fn.exists()):
            continue
        tg, flags = context_rows(conn, s)
        d = paired(pd.read_parquet(fo), pd.read_parquet(fn), tg, flags)
        reps[s] = cuts(d)
        print("\n".join(fmt(f"avail {s} (ON - OFF, diff (z), SE clustered by date)", reps[s])))
        print()
    if a.json:
        Path(a.json).write_text(json.dumps(reps, indent=1, default=float))


if __name__ == "__main__":
    main()
