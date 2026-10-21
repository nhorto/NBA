"""P(play) v2 (p2) research rows: who the live candidate path forces in, and whether they played.

Builds one row per live-path candidate (the panel `pipeline.build_games` would see) with the
availability facts (report status / P, roster flag = in the team's last 10 games) and the
fringe-model features (`availability.fringe_features`), plus the outcome (played, minutes).

    python research/p2_rows.py slice [--offset 45] [--out DIR]   # T2.4 slice, live path
    python research/p2_rows.py rg 2017-18 2018-19 ... [--out DIR] # RotoGuru pools (salaries)

Slice rows: each game at its own tip - offset, official report of the time, the DK-pool
stand-in of T2.4 (`backtest.LiveInputs`), `merge_extras` + `candidate_panel` +
`drop_moved_players` exactly as `backtest.live_like_day`. Also the minutes engine's
P(plays | available) of every candidate in the no-absence scenario (`pp_engine`).

RotoGuru rows (2015-16 .. 2020-21): the real DK pool of the day (salary) forced in as extras;
no injury reports exist for those seasons, so pool players who would have been listed Out are
NOT removed (the DNP share of the fringe population is overstated there).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import availability as AV
from courtside.proj import backtest as B
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

M.GBM_PARAMS["num_threads"] = 1


def say(m: str) -> None:
    print(f"[p2rows] {time.strftime('%H:%M:%S')} {m}", flush=True)


def _played(conn, game_ids) -> dict:
    ids = list(game_ids)
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    return {(g, int(p)): float(m) for g, p, m in conn.execute(
        f"SELECT game_id, player_id, min FROM player_games WHERE game_id IN ({ph}) AND min>0",
        ids)}


def panel_rows(conn, ctx, bundle, d: date, gg: pd.DataFrame, ex: dict, current: dict | None,
               status: dict, p_rep: dict, starters: set | None, hist: AV.History,
               salary: dict | None = None) -> pd.DataFrame:
    panel = P.candidate_panel(ctx, d, gg, ex)
    if current is not None:
        panel = P.drop_moved_players(panel, current, ex)
    if not len(panel):
        return pd.DataFrame()
    feats = AV.fringe_features(panel, ctx.h, salary)
    pp = np.full(len(panel), np.nan)
    base = np.full(len(panel), np.nan)
    if bundle is not None:
        for _, idx in panel.groupby("key", sort=False).indices.items():
            sub = panel.iloc[idx].reset_index(drop=True)
            b, q = M.base_and_pplay(sub, bundle.minutes, np.zeros(len(sub), bool))
            pp[idx] = q
            base[idx] = np.asarray(b, float)
    pl = _played(conn, gg["game_id"])
    out = pd.DataFrame({
        "date": d.isoformat(), "game_id": panel["game_id"].to_numpy(),
        "team": panel["team"].to_numpy(), "player_id": panel["player_id"].astype(np.int64),
        "roster": panel["roster"].to_numpy(int), "app10": panel["app10"].to_numpy(float),
        "status": [status.get(int(p), "") for p in panel["player_id"]],
        "p_report": [p_rep.get(int(p), np.nan) for p in panel["player_id"]],
        "dk_starter": [int(starters is not None and int(p) in starters)
                       for p in panel["player_id"]],
        "pp_engine": pp, "base": base,
        "salary": [np.nan if salary is None else salary.get(int(p), np.nan)
                   for p in panel["player_id"]],
    })
    for c in feats.columns:
        out[c] = feats[c].to_numpy()
    out["minutes"] = [pl.get((g, int(p)), 0.0) for g, p in zip(out["game_id"], out["player_id"])]
    out["played"] = (out["minutes"] > 0).astype(int)
    return out


def cmd_slice(a) -> None:
    conn = db.connect()
    season = "2025-26"
    st = B.Settings(protocol="live", n_sims=10, modes=(), pplay="loo_month")
    sr = B.SeasonRun(conn, season, st, log=say, need_context=True)
    live = sr.live
    days = [d for d in sr.dates(0) if B.SLICE[0] <= d <= B.SLICE[1]]
    rows = []
    t0 = time.time()
    for i, ds in enumerate(days):
        d = date.fromisoformat(ds)
        games = P._home_away(conn, d)
        tips = AV.day_games(conn, d)
        tip_of = dict(zip(tips["game_id"], tips["tip"]))
        games = games.assign(tip=[tip_of.get(g, AV._default_tip(d)) for g in games["game_id"]])
        current = live.team_at(d)
        params = live.params(d)
        for tip, gg in games.groupby("tip", sort=True):
            gg = gg[["game_id", "home", "away"]].reset_index(drop=True)
            at = tip - timedelta(minutes=a.offset)
            starters = live.starters(gg["game_id"]) if a.offset <= 10 else None
            av = AV.p_play_slate(conn, d, at, params, starters=starters, hist=live.hist)
            av = av[av["game_id"].isin(set(gg["game_id"]))]
            p_avail, rep_extras, status = B.report_inputs(av)
            pool = live.pool_extras(current, gg)
            ex = P.merge_extras(pool, rep_extras)
            r = panel_rows(conn, sr.ctx, sr.bundle, d, gg, ex, current, status, p_avail,
                           starters, live.hist)
            if len(r):
                r["at"] = P._iso(at)
                rows.append(r)
        if i % 10 == 0:
            say(f"{ds} ({i + 1}/{len(days)}) {time.time() - t0:.0f}s")
    df = pd.concat(rows, ignore_index=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    f = od / f"rows_slice_live{a.offset}.parquet"
    df.to_parquet(f)
    say(f"wrote {f} ({len(df)} rows)")


def cmd_rg(a) -> None:
    conn = db.connect()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    h = M.load_history(conn, since="2007-08")
    hist = AV.load_history(conn, date(2014, 10, 1))
    rg = pd.read_sql_query(
        "SELECT game_date, player_id, team, salary, game_id FROM dfs_hist_salaries "
        "WHERE salary>0 AND player_id IS NOT NULL AND game_id IS NOT NULL", conn)

    class _Ctx:                       # candidate_panel only needs .h
        pass
    ctx = _Ctx()
    ctx.h = h
    for season in a.seasons:
        lo, hi = B.E.season_start(season), B.E.season_start(B.E.next_season(season))
        days = sorted(d for d in rg["game_date"].unique() if lo <= d < hi)
        if a.first:
            days = days[:a.first]
        rows = []
        t0 = time.time()
        for i, ds in enumerate(days):
            d = date.fromisoformat(ds)
            games = P._home_away(conn, d)
            r0 = rg[rg["game_date"] == ds]
            if not len(games) or not len(r0):
                continue
            team_game = {}
            for g in games.itertuples(index=False):
                team_game[g.home] = g.game_id
                team_game[g.away] = g.game_id
            ex: dict = {}
            for r in r0.itertuples(index=False):
                gid = team_game.get(r.team)
                if gid is None:
                    continue
                ex.setdefault((gid, r.team), []).append(int(r.player_id))
            sal = {int(p): float(s) for p, s in zip(r0["player_id"], r0["salary"])}
            x = panel_rows(conn, ctx, None, d, games, ex, None, {}, {}, None, hist, sal)
            if len(x):
                x["in_pool"] = [int(p) in sal for p in x["player_id"]]
                rows.append(x)
            if i % 25 == 0:
                say(f"{season} {ds} ({i + 1}/{len(days)}) {time.time() - t0:.0f}s")
        df = pd.concat(rows, ignore_index=True)
        df["season"] = season
        f = od / f"rows_rg_{season}.parquet"
        df.to_parquet(f)
        say(f"wrote {f} ({len(df)} rows)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice")
    s.add_argument("--offset", type=int, default=45)
    s.add_argument("--out", default=os.environ.get("P2_OUT", "out"))
    r = sub.add_parser("rg")
    r.add_argument("seasons", nargs="+")
    r.add_argument("--first", type=int, default=0, help="only the first N days of each season")
    r.add_argument("--out", default=os.environ.get("P2_OUT", "out"))
    a = ap.parse_args(argv)
    {"slice": cmd_slice, "rg": cmd_rg}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
