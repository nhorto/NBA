"""eval-lite: the Phase 1 gate (docs/PLAYERS.md §7, §8 row T1.0, review item M5).

    courtside proj evallite [--dates N] [--seasons 2024-25 2019-20 2020-21] [--sims 4000]
                            [--no-lineups] [--out DIR]

Register by adding ``"courtside.proj.evallite"`` to ``PROJ_MODULES`` in ``proj/cli.py``.

Honest walk-forward only: every season S is projected with a bundle fit on seasons < S
(`pipeline.load_bundle`, cached under ``var/proj``), protocol ``avail`` (the oracle of §7.1 L4:
regulars who did not play are out, so the minutes numbers are optimistic), closing lines as the
market. Nothing here is tuned on the test seasons.

Part 1 - projection gate (post-RotoGuru seasons, default 2024-25), all via `pipeline.backtest`:
  * unconditional DK FP (projected minutes >= 10, DNP scored 0): MAE vs the season-to-date DK
    average on identical rows, paired difference with SE clustered by date, 80/50% coverage,
    CRPS, calibration by tier and by level;
  * minutes MAE on rows with played and expected minutes >= 15 (SE clustered by date); the
    full-season ``minutes_eval`` number is quoted next to it when its report is on disk.

Part 2 - RotoGuru seasons (default 2019-20 and 2020-21), salaried on-slate players:
  * player level: MAE of the projection vs the salary-implied line (OLS of DK points on salary
    fit on RotoGuru rows of seasons < S only; three variants, the strongest is the comparator);
  * lineup level: per date the optimal Classic lineup built from our means, from our cash
    objective (mu - 0.3 (mu - p25)), from the salary-implied line and from the hindsight actuals,
    all scored with the ACTUAL RotoGuru DK points. Extra rows: salary-implied and season-to-date
    average (reconstructed from box scores) restricted to the players our availability oracle
    keeps (the like-for-like comparators, because "avail" hands our model the DNPs).
  The 2020-21 season (shortened, bubble-adjacent, odd schedule) and the 2019-20 bubble dates
  (Aug 2020) are kept.

Gate lines (`GATES`) get PASS/FAIL from the point estimate; `sig` records whether the estimate
also clears 2 SE. A line whose season was not run is SKIP.

Public API
  paired_cluster(d, clusters) -> {mean, se, n, n_clusters}
  fit_salary_line(rows, season, mode) -> (a, b);  season_start(season) -> 'YYYY-MM-DD'
  lineup_actual(ids, actual) -> float
  gate_table(results) -> list[dict];  format_report(results) -> str
  run_projection_gate(conn, season, ...), run_rotoguru(conn, season, ...), run(conn, ...)
"""

from __future__ import annotations

import bisect
import json
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .. import config

# ------------------------------------------------------------------------------ gate thresholds
GATE_FP_GAIN_PCT = 3.0            # DK FP MAE >= 3% better than season-to-date (2024-25, avail)
GATE_COV80 = (0.76, 0.84)         # 80% interval coverage
GATE_MINUTES_MAE = 4.8            # minutes MAE (projected >= 15)
GATE_VS_SALARY_PCT = 0.0          # "not worse than salary-implied"
CASH_LAMBDA = 0.3
PROJECTION_DEFAULT = ("2024-25",)
ROTOGURU_DEFAULT = ("2019-20", "2020-21")
ROTOGURU_FIRST = 2014             # first RotoGuru season (2014-15)


# ============================================================================== statistics
# moved to proj/metrics.py (T2.1); re-exported here so the T1.0 API and tests are unchanged
from .metrics import gain_pct, paired_cluster, rate_se  # noqa: E402,F401


# ============================================================================== salary-implied

def season_start(season: str) -> str:
    """First day that belongs to `season` for RotoGuru bookkeeping. 2020-21 starts in Nov 2020
    because the 2019-20 bubble (Aug-Oct 2020) belongs to 2019-20."""
    if season == "2020-21":
        return "2020-11-01"
    return f"{int(season[:4])}-09-01"


def next_season(season: str) -> str:
    y = int(season[:4]) + 1
    return f"{y}-{(y + 1) % 100:02d}"


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def season_label(day: str) -> str:
    """RotoGuru-era season of an ISO date (Aug-Oct 2020 bubble -> 2019-20)."""
    d = str(day)[:10]
    y = int(d[:4])
    s = f"{y}-{(y + 1) % 100:02d}"
    if d >= season_start(s):
        return s
    return prev_season(s)


SALARY_MODES = ("all_seasons", "prev_season", "played_only")


def fit_salary_line(rows: pd.DataFrame, season: str, mode: str = "all_seasons"
                    ) -> tuple[float, float]:
    """OLS dk_pts = a + b * salary on RotoGuru rows (columns game_date, salary, pts, minutes) of
    seasons strictly before `season`. Modes: all_seasons (all earlier rows, DNP = 0),
    prev_season (only the previous season), played_only (minutes > 0, all earlier seasons).
    Rows on or after `season_start(season)` are never used."""
    cut = season_start(season)
    r = rows[rows["game_date"] < cut]
    if mode == "prev_season":
        r = r[r["game_date"] >= season_start(prev_season(season))]
    elif mode == "played_only":
        r = r[r["minutes"] > 0]
    elif mode != "all_seasons":
        raise ValueError(f"unknown salary mode {mode!r}")
    if len(r) < 50:
        raise ValueError(f"only {len(r)} RotoGuru rows before {cut} for mode {mode}")
    x = r["salary"].to_numpy(float)
    y = r["pts"].to_numpy(float)
    xm, ym = x.mean(), y.mean()
    b = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
    return float(ym - b * xm), b


def lineup_actual(ids: Iterable[str], actual: dict) -> float:
    """Score of a lineup with the ACTUAL DK points (missing id = 0, a DNP)."""
    return float(sum(float(actual.get(i, 0.0) or 0.0) for i in ids))


# ============================================================================== helpers

def _pick(items: list, n: int) -> list:
    """n evenly spaced items (n <= 0 or n >= len: all)."""
    if n <= 0 or n >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def _f(x):
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _f(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def is_rotoguru_season(season: str) -> bool:
    y = int(season[:4])
    return ROTOGURU_FIRST <= y <= 2020


# ============================================================================== part 1

def usable_dates(conn: sqlite3.Connection, season: str) -> tuple[list[date], list[dict]]:
    """Regular-season dates of `season` that `pipeline.project_historical` can project, and the
    excluded ones. `pipeline._home_away` needs one home and one away row per game; the NBA Cup
    semifinals (2024-12-14) are neutral-site games with is_home = 0 for both teams in
    team_games, so on such a date the frame is empty and project_historical raises KeyError
    'game_id'. A date with only some unpairable games is kept (those games are not projected)."""
    rows = conn.execute(
        "SELECT game_date, game_id, SUM(is_home=1), SUM(is_home=0) FROM team_games "
        "WHERE season=? AND season_type='Regular Season' GROUP BY game_date, game_id", (season,))
    ok: dict[str, int] = {}
    bad: dict[str, int] = {}
    for d, _, h, a in rows:
        if h == 1 and a == 1:
            ok[d] = ok.get(d, 0) + 1
        else:
            bad[d] = bad.get(d, 0) + 1
    days = sorted(ok)
    excluded = [{"date": d, "games_without_home_away_pair": n, "reason": "neutral site: no home "
                 "team in team_games (pipeline._home_away drops the game)"}
                for d, n in sorted(bad.items())]
    return [date.fromisoformat(d) for d in days], excluded


def run_projection_gate(conn: sqlite3.Connection, season: str = "2024-25", *, n_dates: int = 0,
                        n_sims: int = 4_000, log: Callable[[str], None] = print) -> dict:
    """Walk-forward (fit < S), avail protocol, via `pipeline.backtest` on `n_dates` evenly spaced
    regular-season dates (0 = every projectable date). Adds date-clustered SEs and minutes MAE."""
    from . import pipeline as P
    t0 = time.time()
    days, excluded = usable_dates(conn, season)
    res = P.backtest(conn, season, n_sims=n_sims, protocol="avail", log=log,
                     dates=_pick(days, n_dates))
    df = res.pop("frame")
    out = evaluate_frame(df)
    out.update({"season": season, "protocol": "avail", "n_dates_run": int(df["game_date"].nunique()),
                "n_dates_available": len(days), "excluded_dates": excluded, "n_sims": n_sims,
                "seconds": round(time.time() - t0, 1),
                "dates_first_last": [str(df["game_date"].min()), str(df["game_date"].max())],
                "sim_corr": res.get("sim_corr"), "actual_corr": res.get("actual_corr")})
    out["minutes_eval_reference"] = minutes_eval_reference(season)
    return out


def evaluate_frame(df: pd.DataFrame) -> dict:
    """All part-1 metrics from `pipeline.backtest` rows (pure; tested on synthetic frames)."""
    from . import pipeline as P
    u = df[(df["exp_min"] >= 10) & df["base_std"].notna()]
    y = u["y"].to_numpy(float)
    day = u["game_date"].to_numpy()
    e_m = np.abs(u["mean"].to_numpy(float) - y)
    e_s = np.abs(u["base_std"].to_numpy(float) - y)
    g = gain_pct(e_s, e_m, day)
    fp = {"n": g["n"], "n_dates": g["n_clusters"], "mae_model": g["mae_model"],
          "mae_season_to_date": g["mae_base"], "gain_vs_std_pct": g["gain_pct"],
          "gain_vs_std_se_pct": g["se_pct"], "dnp_rate": float((~u["played"]).mean()),
          "bias": float((u["mean"] - u["y"]).mean()), "crps": float(u["crps"].mean())}
    ew = u["base_ewma"].fillna(u["base_std"]).to_numpy(float)
    ge = gain_pct(np.abs(ew - y), e_m, day)
    fp["mae_ewma_min_x_rate"] = ge["mae_base"]
    fp["gain_vs_ewma_pct"] = ge["gain_pct"]
    fp["gain_vs_ewma_se_pct"] = ge["se_pct"]
    c80 = rate_se(((u["q10"] <= y) & (y <= u["q90"])).to_numpy(), day)
    c50 = rate_se(((u["q25"] <= y) & (y <= u["q75"])).to_numpy(), day)
    fp["cov80"], fp["cov80_se"] = c80["rate"], c80["se"]
    fp["cov50"], fp["cov50_se"] = c50["rate"], c50["se"]
    # game-clustered SE as in pipeline.evaluate_rows (cross-check)
    fp["gain_vs_std_se_pct_game_cluster"] = float(
        P._cl_se(e_s - e_m, u["game_id"].to_numpy()) / e_s.mean() * 100)
    # CRPS of the season-to-date point forecast equals its MAE
    fp["crps_ratio_vs_std_mae"] = fp["crps"] / fp["mae_season_to_date"]
    cal = P.evaluate_rows(df)["calibration"]
    m = df[df["played"] & (df["exp_min"] >= 15)]
    em = (m["exp_min"] - m["act_min"]).to_numpy(float)
    pc = paired_cluster(np.abs(em), m["game_date"].to_numpy())
    minutes = {"n": pc["n"], "mae": pc["mean"], "mae_se": pc["se"],
               "bias": float(em.mean()) if len(em) else None,
               "rmse": float(np.sqrt((em ** 2).mean())) if len(em) else None}
    return {"fp": fp, "calibration": cal, "minutes": minutes}


def minutes_eval_reference(season: str) -> dict | None:
    """The T1.2 full-season lock-45 minutes number (reports/minutes_v05_eval.json), if any."""
    p = config.data_dir() / "reports" / "minutes_v05_eval.json"
    try:
        s = json.loads(p.read_text())["seasons"][season]["lock45"]
        return {"n": s["n"], "mae": s["mae"], "bias": s["bias"],
                "baseline_mae_same_rows": s["baseline_mae_same_rows"]}
    except (OSError, KeyError, ValueError):
        return None


# ============================================================================== part 2

def load_rotoguru(conn: sqlite3.Connection) -> pd.DataFrame:
    """All on-slate salaried RotoGuru rows (salary-line fits use the earlier seasons of this)."""
    return pd.read_sql_query(
        "SELECT game_date, rg_id, player_id, salary, dk_pts AS pts, minutes, pos "
        "FROM dfs_hist_salaries WHERE on_slate=1 AND salary>0 AND dk_pts IS NOT NULL", conn)


def std_baseline(conn: sqlite3.Connection, season: str, rows: pd.DataFrame) -> np.ndarray:
    """Season-to-date DK average (the DK FPPG convention: this season's mean of games played
    before the date once >= 5 games, else last season's mean) rebuilt from box scores for
    rows (player_id, game_date). Season boundaries follow `season_label` (bubble = 2019-20)."""
    from . import pipeline as P
    a = P._actuals(conn, season)
    a = a[a["min"] > 0].sort_values(["player_id", "game_date"])
    cs0, cs1 = season_start(season), season_start(prev_season(season))
    by = {}
    for pid, g in a.groupby("player_id"):
        dts = g["game_date"].to_numpy()
        by[int(pid)] = (list(dts), np.concatenate([[0.0], np.cumsum(g["dk"].to_numpy(float))]))
    out = np.full(len(rows), np.nan)
    for k, (pid, day) in enumerate(zip(rows["player_id"].to_numpy(), rows["game_date"].to_numpy())):
        if pd.isna(pid) or int(pid) not in by:
            continue
        dts, cs = by[int(pid)]
        i = bisect.bisect_left(dts, day)            # games strictly before the date
        j = bisect.bisect_left(dts, cs0)
        j = min(j, i)
        n_cur = i - j
        if n_cur >= 5:
            out[k] = (cs[i] - cs[j]) / n_cur
            continue
        h = min(bisect.bisect_left(dts, cs1), j)
        n_last = j - h
        if n_last > 0:
            out[k] = (cs[j] - cs[h]) / n_last
        elif n_cur > 0:
            out[k] = (cs[i] - cs[j]) / n_cur
    return out


def run_rotoguru(conn: sqlite3.Connection, season: str, *, n_dates: int = 0,
                 n_sims: int = 4_000, lineups: bool = True,
                 log: Callable[[str], None] = print) -> dict:
    """Player- and lineup-level evaluation on the RotoGuru dates of `season` (fit < S)."""
    from . import pipeline as P
    from .dfs import optimizer as O
    t0 = time.time()
    rg = load_rotoguru(conn)
    lo, hi = season_start(season), season_start(next_season(season))
    days = [d for d in O.hist_dates(conn) if lo <= d < hi]
    n_all = len(days)
    days = _pick(days, n_dates)
    lines = {m: fit_salary_line(rg, season, m) for m in SALARY_MODES}
    log(f"{season}: {len(days)}/{n_all} RotoGuru dates; salary lines "
        + ", ".join(f"{m}: {a:.2f}+{b * 1000:.2f}/$1k" for m, (a, b) in lines.items()))
    ctx = P.Context(conn, season, log=log)
    bundle = P.load_bundle(conn, season, ctx, log=log)
    rules = O.classic_rules() if lineups else None
    if lineups and rules is None:
        raise LookupError("no archived Classic rules payload (needed for the lineup level)")
    season_rows = rg[(rg["game_date"] >= lo) & (rg["game_date"] < hi)]
    std_all = pd.DataFrame({"player_id": season_rows["player_id"], "game_date": season_rows["game_date"]})
    std_all["std"] = std_baseline(conn, season, season_rows)
    std_map = {(d, int(p)): v for d, p, v in zip(std_all["game_date"], std_all["player_id"], std_all["std"])
               if not pd.isna(p) and np.isfinite(v)}

    rows, lineup_rows, skipped = [], [], []
    for k, day in enumerate(days):
        t1 = time.time()
        try:
            res = P.project_historical(conn, day, protocol="avail", n_sims=n_sims, ctx=ctx,
                                       bundle=bundle)
        except Exception as e:                      # keep the evaluation running, say why
            skipped.append({"date": day, "error": f"{type(e).__name__}: {e}"})
            log(f"  {day}: SKIPPED {type(e).__name__}: {e}")
            continue
        pl = res.players.drop_duplicates("player_id").set_index("player_id")
        pool_rows = rg[rg["game_date"] == day]
        proj = {}
        for r in pool_rows.itertuples(index=False):
            if pd.isna(r.player_id) or int(r.player_id) not in pl.index:
                continue
            q = pl.loc[int(r.player_id)]
            proj[str(r.rg_id)] = q
        for r in pool_rows.itertuples(index=False):
            q = proj.get(str(r.rg_id))
            row = {"date": day, "rg_id": str(r.rg_id), "salary": float(r.salary),
                   "y": float(r.pts), "minutes": float(r.minutes or 0.0),
                   "std": std_map.get((day, int(r.player_id)), np.nan)
                   if not pd.isna(r.player_id) else np.nan}
            if q is not None:
                row.update({"mean": float(q["dk_mean"]), "q10": float(q["dk_q10"]),
                            "q25": float(q["dk_q25"]), "q75": float(q["dk_q75"]),
                            "q90": float(q["dk_q90"]), "exp_min": float(q["min_mean"])})
            rows.append(row)
        if lineups:
            try:
                lineup_rows.append(_lineups_for_day(conn, day, pool_rows, proj, lines, std_map,
                                                    rules))
            except Exception as e:
                skipped.append({"date": day, "error": f"lineups {type(e).__name__}: {e}"})
                log(f"  {day}: lineups SKIPPED {type(e).__name__}: {e}")
        if k % 10 == 0 or k == len(days) - 1:
            log(f"  {day} ({k + 1}/{len(days)}): {len(res.sims)} games, "
                f"{time.time() - t1:.1f}s, elapsed {time.time() - t0:.0f}s, "
                f"rss {P.peak_rss_mb():.0f} MB")
    df = pd.DataFrame(rows)
    out = {"season": season, "protocol": "avail", "n_sims": n_sims, "n_dates_all": n_all,
           "n_dates_run": len(days) - len({s["date"] for s in skipped}),
           "dates_first_last": [days[0], days[-1]] if days else None,
           "skipped": skipped, "salary_lines": {m: {"a": a, "b_per_1k": b * 1000}
                                                for m, (a, b) in lines.items()}}
    out["player"] = evaluate_rotoguru_players(df, lines)
    if lineups:
        out["lineup"] = evaluate_lineups(pd.DataFrame(lineup_rows))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def evaluate_rotoguru_players(df: pd.DataFrame, lines: dict) -> dict:
    """Player-level MAE vs salary-implied on identical rows. Population: salaried on-slate
    players with a projection and expected minutes >= 10, DNP scored 0 (selection uses only our
    projection). Also reported: every projected player, and coverage of the pool."""
    out: dict = {"pool_rows": int(len(df)),
                 "share_with_projection": float(df["mean"].notna().mean()) if len(df) else None}
    if "mean" not in df or not df["mean"].notna().any():
        return out
    pr = df[df["mean"].notna()]
    pops = {"exp_min_ge_10": pr[pr["exp_min"] >= 10], "all_projected": pr}
    for name, u in pops.items():
        y = u["y"].to_numpy(float)
        day = u["date"].to_numpy()
        e_m = np.abs(u["mean"].to_numpy(float) - y)
        sal = {m: np.abs(a + b * u["salary"].to_numpy(float) - y) for m, (a, b) in lines.items()}
        best = min(sal, key=lambda m: sal[m].mean())
        r = {"n": int(len(u)), "n_dates": int(u["date"].nunique()),
             "dnp_rate": float((u["minutes"] <= 0).mean()),
             "mae_model": float(e_m.mean()), "best_salary_mode": best,
             "vs_salary_best": gain_pct(sal[best], e_m, day),
             "vs_salary_modes": {m: gain_pct(sal[m], e_m, day) for m in sal},
             "bias_model": float((u["mean"] - u["y"]).mean())}
        fixed = np.abs(5.0 * u["salary"].to_numpy(float) / 1000 - y)
        r["vs_fixed_5_per_1k"] = gain_pct(fixed, e_m, day)
        s = u[u["std"].notna()]
        if len(s):
            ys = s["y"].to_numpy(float)
            r["vs_season_to_date_reconstructed"] = gain_pct(
                np.abs(s["std"].to_numpy(float) - ys), np.abs(s["mean"].to_numpy(float) - ys),
                s["date"].to_numpy())
        c80 = rate_se(((u["q10"] <= y) & (y <= u["q90"])).to_numpy(), day)
        c50 = rate_se(((u["q25"] <= y) & (y <= u["q75"])).to_numpy(), day)
        r.update({"cov80": c80["rate"], "cov80_se": c80["se"], "cov50": c50["rate"],
                  "cov50_se": c50["se"]})
        out[name] = r
    return out


LINEUP_METHODS = ("ours_mean", "ours_cash", "salary", "salary_avail", "std_avail", "hindsight")


def _lineups_for_day(conn, day: str, pool_rows: pd.DataFrame, proj: dict, lines: dict,
                     std_map: dict, rules) -> dict:
    """Actual points of each method's optimal Classic lineup on one RotoGuru date."""
    from .dfs import optimizer as O
    pool = O.hist_pool(conn, day, rules)
    actual = {str(r.rg_id): float(r.pts) for r in pool_rows.itertuples(index=False)}
    in_pool = {p.dk_player_id: p for p in pool.players}
    a, b = lines["all_seasons"]
    ours = {k: O.Projection(float(q["dk_mean"]), p25=float(q["dk_q25"])) for k, q in proj.items()
            if k in in_pool}
    sal = {k: O.Projection(a + b * p.salary) for k, p in in_pool.items()}
    avail = {k for k, q in proj.items() if k in in_pool and float(q["min_mean"]) >= 10}
    sal_av = {k: v for k, v in sal.items() if k in avail}
    std_av = {}
    for r in pool_rows.itertuples(index=False):
        k = str(r.rg_id)
        v = std_map.get((day, int(r.player_id))) if not pd.isna(r.player_id) else None
        if k in avail and v is not None:
            std_av[k] = O.Projection(float(v))
    out: dict = {"date": day, "n_pool": len(in_pool), "n_ours": len(ours)}
    jobs = {"ours_mean": (ours, "mean"), "ours_cash": (ours, "cash"), "salary": (sal, "mean"),
            "salary_avail": (sal_av, "mean"), "std_avail": (std_av, "mean"),
            "hindsight": (O.hist_projections(conn, pool, "actual"), "mean")}
    for name, (pj, mode) in jobs.items():
        lu = O.optimize(pool, pj, rules, mode=mode, lam=CASH_LAMBDA, time_limit_s=20.0)
        out[name] = lineup_actual(lu.player_ids, actual) if lu is not None else np.nan
        out[f"{name}_proj"] = float(lu.projected) if lu is not None else np.nan
    return out


def evaluate_lineups(ld: pd.DataFrame) -> dict:
    """Per-method mean actual points, paired differences vs ours (SE over dates) and the ratio
    to the hindsight-optimal lineup. Dates where any method failed are dropped for all."""
    if ld.empty:
        return {}
    ok = ld[list(LINEUP_METHODS)].notna().all(axis=1)
    d = ld[ok]
    day = d["date"].to_numpy()
    hs = d["hindsight"].to_numpy(float)
    out: dict = {"n_dates": int(len(d)), "n_dates_dropped": int((~ok).sum()),
                 "mean_pool": float(d["n_pool"].mean()), "methods": {}, "paired": {}}
    for m in LINEUP_METHODS:
        v = d[m].to_numpy(float)
        pc = paired_cluster(v, day)
        out["methods"][m] = {"mean_actual": pc["mean"], "se": pc["se"],
                             "ratio_to_hindsight": float(v.sum() / hs.sum()),
                             "mean_per_date_ratio": float((v / hs).mean()),
                             "cash_rate_proxy_ge_300": float((v >= 300).mean())}
    for ours in ("ours_mean", "ours_cash"):
        for other in ("salary", "salary_avail", "std_avail"):
            diff = d[ours].to_numpy(float) - d[other].to_numpy(float)
            pc = paired_cluster(diff, day)
            out["paired"][f"{ours}_minus_{other}"] = {
                "mean_diff": pc["mean"], "se": pc["se"],
                "z": pc["mean"] / pc["se"] if pc["se"] and np.isfinite(pc["se"]) else None,
                "win_rate": float((diff > 0).mean()), "tie_rate": float((diff == 0).mean()),
                "n": pc["n"]}
    return out


# ============================================================================== gate table

def judge(value: float | None, op: str, threshold) -> bool | None:
    """True / False / None (no value). ops: ge, le, between (threshold = (lo, hi))."""
    if value is None or not np.isfinite(value):
        return None
    if op == "ge":
        return bool(value >= threshold)
    if op == "le":
        return bool(value <= threshold)
    if op == "between":
        return bool(threshold[0] <= value <= threshold[1])
    raise ValueError(op)


def _line(gid: str, desc: str, value, se, n, op: str, thr, *, core=True, sig=None,
          unit: str = "") -> dict:
    ok = judge(value, op, thr)
    return {"id": gid, "desc": desc, "value": _f(value), "se": _f(se), "n": n, "op": op,
            "threshold": thr, "unit": unit, "core": core, "sig": sig,
            "verdict": "SKIP" if ok is None else ("PASS" if ok else "FAIL")}


def gate_table(results: dict) -> list[dict]:
    """PASS/FAIL per Phase 1 gate line from the `run` results (see `run`). The core lines decide
    the overall verdict; lineup-level lines are supplementary (same threshold logic)."""
    rows: list[dict] = []
    pg = results.get("projection", {})
    for season, r in pg.items():
        fp, mi = r["fp"], r["minutes"]
        z = fp["gain_vs_std_pct"] / fp["gain_vs_std_se_pct"] if fp["gain_vs_std_se_pct"] else None
        rows.append(_line("G1", f"{season} DK FP MAE gain vs season-to-date (avail, uncond.)",
                          fp["gain_vs_std_pct"], fp["gain_vs_std_se_pct"], fp["n"], "ge",
                          GATE_FP_GAIN_PCT, sig=bool(z is not None and z >= 2), unit="%"))
        rows.append(_line("G3", f"{season} DK FP 80% interval coverage", fp["cov80"],
                          fp["cov80_se"], fp["n"], "between", GATE_COV80))
        rows.append(_line("G4", f"{season} minutes MAE (played, projected >= 15)", mi["mae"],
                          mi["mae_se"], mi["n"], "le", GATE_MINUTES_MAE))
    for season, r in results.get("rotoguru", {}).items():
        p = r.get("player", {}).get("exp_min_ge_10")
        if p:
            v = p["vs_salary_best"]
            z = v["gain_pct"] / v["se_pct"] if v["se_pct"] else None
            rows.append(_line("G2", f"{season} player MAE gain vs best salary-implied (RotoGuru)",
                              v["gain_pct"], v["se_pct"], v["n"], "ge", GATE_VS_SALARY_PCT,
                              sig=bool(z is not None and z >= 2), unit="%"))
        lu = r.get("lineup", {}).get("paired", {})
        for key, label in (("ours_mean_minus_salary", "mean lineup"),
                           ("ours_cash_minus_salary", "cash lineup")):
            q = lu.get(key)
            if q:
                rows.append(_line("G2b", f"{season} {label} actual pts minus salary-implied lineup",
                                  q["mean_diff"], q["se"], q["n"], "ge", 0.0, core=False,
                                  sig=bool(q["z"] is not None and q["z"] >= 2), unit=" pts"))
    return rows


def overall(rows: list[dict]) -> str:
    core = [r for r in rows if r["core"]]
    if not core:
        return "NO RESULTS"
    if any(r["verdict"] == "FAIL" for r in core):
        return "FAIL"
    if any(r["verdict"] == "SKIP" for r in core):
        return "INCOMPLETE"
    return "PASS"


def _thr(r: dict) -> str:
    t = r["threshold"]
    if r["op"] == "between":
        return f"{t[0]:.2f}-{t[1]:.2f}"
    return (">= " if r["op"] == "ge" else "<= ") + f"{t:g}{r['unit']}"


def format_gate(rows: list[dict]) -> str:
    lines = [f"{'id':<4} {'verdict':<7} {'value':>10} {'SE':>7} {'n':>8}  {'threshold':<12} line"]
    for r in rows:
        v = "-" if r["value"] is None else (f"{r['value']:.3f}" if r["op"] == "between"
                                           else f"{r['value']:+.2f}{r['unit']}"
                                           if r["op"] == "ge" else f"{r['value']:.3f}")
        se = "-" if r["se"] is None else f"{r['se']:.3f}"
        tag = "" if r["core"] else "  [supplementary]"
        sg = "" if r["sig"] is None else ("  (>= 2 SE)" if r["sig"] else "  (< 2 SE)")
        lines.append(f"{r['id']:<4} {r['verdict']:<7} {v:>10} {se:>7} {r['n']:>8}  "
                     f"{_thr(r):<12} {r['desc']}{sg}{tag}")
    lines.append(f"OVERALL (core lines): {overall(rows)}")
    return "\n".join(lines)


# ============================================================================== report

def format_report(results: dict) -> str:
    rows = results["gate"]
    L = [f"eval-lite {results['generated']}  ({results['seconds']:.0f}s total, "
         f"peak RSS {results.get('peak_rss_mb', 0):.0f} MB)", "", "PHASE 1 GATE", format_gate(rows), ""]
    for season, r in results.get("projection", {}).items():
        fp, mi = r["fp"], r["minutes"]
        L += [f"== {season} projection gate (fit < {season}, avail, {r['n_dates_run']}/{r['n_dates_available']} "
              f"dates {r['dates_first_last'][0]}..{r['dates_first_last'][1]}, {r['n_sims']} sims, "
              f"{r['seconds']:.0f}s)",
              *([f"excluded dates (pipeline cannot project): {r['excluded_dates']}"]
                if r["excluded_dates"] else []),
              f"unconditional DK FP (exp min >= 10, DNP = 0): n={fp['n']} ({fp['n_dates']} dates) "
              f"dnp {fp['dnp_rate']:.3f}",
              f"  MAE model {fp['mae_model']:.3f} | season-to-date {fp['mae_season_to_date']:.3f} "
              f"| EWMA min x rate {fp['mae_ewma_min_x_rate']:.3f}",
              f"  gain vs season-to-date {fp['gain_vs_std_pct']:+.2f}% (SE by date "
              f"{fp['gain_vs_std_se_pct']:.2f}; by game {fp['gain_vs_std_se_pct_game_cluster']:.2f}), "
              f"vs EWMA x rate {fp['gain_vs_ewma_pct']:+.2f}% (SE {fp['gain_vs_ewma_se_pct']:.2f})",
              f"  CRPS {fp['crps']:.3f} (season-to-date point forecast CRPS = MAE "
              f"{fp['mae_season_to_date']:.3f}; ratio {fp['crps_ratio_vs_std_mae']:.3f})",
              f"  cov80 {fp['cov80']:.3f} (SE {fp['cov80_se']:.3f})  cov50 {fp['cov50']:.3f} "
              f"(SE {fp['cov50_se']:.3f})  bias {fp['bias']:+.3f}",
              "  calibration:"]
        for k, v in r["calibration"].items():
            L.append(f"    {k:<6} n={v['n']:<5} pred {v['pred']:6.2f} actual {v['actual']:6.2f} "
                     f"MAE {v['mae']:.2f} (std {v['mae_std']:.2f}) cov80 {v['cov80']:.3f} "
                     f"cov50 {v['cov50']:.3f}")
        ref = r.get("minutes_eval_reference")
        L.append(f"minutes (played, exp >= 15): n={mi['n']} MAE {mi['mae']:.3f} (SE {mi['mae_se']:.3f}) "
                 f"bias {mi['bias']:+.3f} RMSE {mi['rmse']:.3f}"
                 + (f" | minutes_eval lock-45 full season: n={ref['n']} MAE {ref['mae']:.3f} "
                    f"bias {ref['bias']:+.3f}" if ref else ""))
        L.append("")
    for season, r in results.get("rotoguru", {}).items():
        L += [f"== RotoGuru {season} (fit < {season}, avail; {r['n_dates_run']}/{r['n_dates_all']} "
              f"dates {r['dates_first_last'][0] if r['dates_first_last'] else '-'}.."
              f"{r['dates_first_last'][1] if r['dates_first_last'] else '-'}; {r['seconds']:.0f}s)"]
        sl = r["salary_lines"]
        L.append("salary lines (seasons < S): " + "; ".join(
            f"{m} {v['a']:+.2f} + {v['b_per_1k']:.2f}/$1k" for m, v in sl.items()))
        if r["skipped"]:
            L.append(f"skipped dates: {len(r['skipped'])} e.g. {r['skipped'][:2]}")
        for name, p in r.get("player", {}).items():
            if not isinstance(p, dict):
                continue
            v = p["vs_salary_best"]
            tag = " (includes players the avail oracle drops: inflated)" if name == "all_projected" else ""
            L.append(f"player [{name}]{tag}: n={p['n']} ({p['n_dates']} dates) dnp {p['dnp_rate']:.3f} "
                     f"MAE ours {p['mae_model']:.3f} vs salary[{p['best_salary_mode']}] "
                     f"{v['mae_base']:.3f} -> {v['gain_pct']:+.2f}% (SE {v['se_pct']:.2f}); "
                     f"fixed 5/$1k {p['vs_fixed_5_per_1k']['mae_base']:.3f}; "
                     f"cov80 {p['cov80']:.3f} cov50 {p['cov50']:.3f} bias {p['bias_model']:+.2f}")
            for m, g in p["vs_salary_modes"].items():
                L.append(f"    salary[{m}] MAE {g['mae_base']:.3f} -> ours {g['gain_pct']:+.2f}% "
                         f"(SE {g['se_pct']:.2f})")
            s = p.get("vs_season_to_date_reconstructed")
            if s:
                L.append(f"    reconstructed season-to-date avg (box scores; n={s['n']}): MAE "
                         f"{s['mae_base']:.3f} -> ours {s['gain_pct']:+.2f}% (SE {s['se_pct']:.2f})")
        lu = r.get("lineup")
        if lu:
            L.append(f"lineups: {lu['n_dates']} dates (dropped {lu['n_dates_dropped']}), mean pool "
                     f"{lu['mean_pool']:.0f}; mean ACTUAL DK points")
            for m, v in lu["methods"].items():
                L.append(f"    {m:<13} {v['mean_actual']:7.2f} (SE {v['se']:.2f})  /hindsight "
                         f"{v['ratio_to_hindsight']:.3f}  mean per-date ratio "
                         f"{v['mean_per_date_ratio']:.3f}  >=300 {v['cash_rate_proxy_ge_300']:.3f}")
            for k, v in lu["paired"].items():
                z = "-" if v["z"] is None else f"{v['z']:+.1f}"
                L.append(f"    {k:<30} {v['mean_diff']:+7.2f} (SE {v['se']:.2f}, z {z}) "
                         f"wins {v['win_rate']:.3f} ties {v['tie_rate']:.3f}")
        L.append("")
    L += ["NOTES",
          "- avail protocol = known availability (a perfect lock-10): minutes and DK MAE are "
          "optimistic (PLAYERS.md 7.6); eligibility is single-position pools (pessimistic).",
          "- salary-implied: OLS of DK pts on salary using RotoGuru rows of seasons < S only; "
          "three variants, the best on the test rows is the comparator (conservative for us).",
          "- DK FPPG is not archived for 2019-21: the season-to-date average is rebuilt from "
          "box scores (FPPG convention) and shown as an extra, not a gate line.",
          "- 'salary' lineups can pick players who then DNP (no availability information); "
          "'salary_avail'/'std_avail' are restricted to players our availability oracle keeps, "
          "the like-for-like comparison for our lineups.",
          "- 2020-21 is shortened with a compressed schedule; 2019-20 includes the Aug 2020 "
          "bubble. Both are kept; season labels follow the bubble = 2019-20."]
    return "\n".join(L)


# ============================================================================== run + CLI

def run(conn: sqlite3.Connection, *, seasons: Sequence[str] | None = None, n_dates: int = 0,
        n_sims: int = 4_000, lineups: bool = True, log: Callable[[str], None] = print) -> dict:
    from . import pipeline as P
    t0 = time.time()
    seasons = list(seasons) if seasons else list(PROJECTION_DEFAULT + ROTOGURU_DEFAULT)
    res: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "seasons": seasons,
                 "n_dates_per_season": n_dates or "all", "projection": {}, "rotoguru": {}}
    for s in seasons:
        if is_rotoguru_season(s):
            res["rotoguru"][s] = run_rotoguru(conn, s, n_dates=n_dates, n_sims=n_sims,
                                              lineups=lineups, log=log)
        else:
            res["projection"][s] = run_projection_gate(conn, s, n_dates=n_dates, n_sims=n_sims,
                                                       log=log)
    res["gate"] = gate_table(res)
    res["overall"] = overall(res["gate"])
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_mb"] = P.peak_rss_mb()
    return res


def write_reports(results: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (config.data_dir() / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    jp, tp = out_dir / f"evallite_{stamp}.json", out_dir / f"evallite_{stamp}.txt"
    jp.write_text(json.dumps(_clean(results), indent=1))
    tp.write_text(format_report(results) + "\n")
    return jp, tp


def add_commands(sub) -> dict:
    p = sub.add_parser("evallite", help="Phase 1 gate: walk-forward 2024-25 + RotoGuru 2019-21")
    p.add_argument("--dates", type=int, default=0,
                   help="evenly spaced dates per season (0 = every date)")
    p.add_argument("--seasons", nargs="+", default=None,
                   help="default: 2024-25 2019-20 2020-21 (<= 2020-21 runs the RotoGuru part)")
    p.add_argument("--sims", type=int, default=4_000)
    p.add_argument("--no-lineups", action="store_true", help="skip the lineup level")
    p.add_argument("--out", default=None, help="report directory (default var/reports)")
    return {"evallite": _cmd}


def _cmd(args) -> int:
    from .. import db
    conn = db.connect()
    res = run(conn, seasons=args.seasons, n_dates=args.dates, n_sims=args.sims,
              lineups=not args.no_lineups, log=lambda m: print(m, flush=True))
    jp, tp = write_reports(res, Path(args.out) if args.out else None)
    print(format_report(res))
    print(f"\nwrote {jp}\nwrote {tp}")
    return 0
