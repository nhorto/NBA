"""G4 (IMPROVEMENTS, absence scale) replay: injury state at the G1 cutoffs, per player.

Same walk-forward frame as `research/g1_line_news_replay.py`, but the production pair is NOT
refit every day (the per-day fitted scorers are already in the G1 dumps, `var/tmp/g1/days`):
`refit_days` is huge, so the only per-day cost is the engine state. `predict.price_day` is
replaced by a dumper that writes, per NBA day, `--out/day_<d>.pkl` with

* `state`: a compact copy of the day's `PlayerState` (every per-player array, `team_history`
  for the day's teams only), enough to rebuild any expected lineup offline with
  `PlayerState.expected_lineup`;
* `cover[name]`: per raw team, `injuries.availability` at the cutoff `name` (source,
  captured_at, avail {pid: P}, the matched report records with status), per game;
* `lineups[name]`: the production lineup strengths (identity check against the G1 dumps);
* `cells[name]` / `cells_ship[name]`: {pid: P(play)} from the DFS side's calibrated report
  cells (`proj.availability.p_play_slate`) at the same cutoff, with params fit on regular-season
  days strictly before the test month (`cells`, walk-forward; none before January: those keep
  the hand map) and the shipped in-sample params (`cells_ship`).

    run.sh code/research/as_replay.py --db <scratch-gd.db> --start 2025-12-22 --end 2026-02-28 \\
        --out out/days
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

CUTOFFS = ("d1_17", "d1_20", "d1_23", "d0_03", "d0_17", "d0_21", "tip90", "tip")


def compact_state(st, teams: set[str]):
    s = copy.copy(st)
    s.team_history = {t: st.team_history[t] for t in teams if t in st.team_history}
    s._pos = {}
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="database (opened read-only)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "g1r", Path(__file__).with_name("g1_line_news_replay.py"))
    g1r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1r)
    from courtside import injuries, predict, production
    from courtside.proj import availability as pav

    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    ship = pav.load_params()
    month_params: dict[date, object] = {}

    def params_for(day: date):
        m0 = date(day.year, day.month, 1)
        if m0 <= pav.TRAIN_LO:
            return None
        hi = min(m0 - timedelta(days=1), pav.RS_END)
        if hi not in month_params:
            t = time.time()
            month_params[hi] = pav.fit(conn, pav.TRAIN_LO, hi)
            print(f"  cells fit through {hi}: {month_params[hi].meta.get('rows')} rows, "
                  f"{time.time() - t:.0f}s", flush=True)
        return month_params[hi]

    def slate_p(day, cut, params, teams):
        if params is None:
            return {}
        df = pav.p_play_slate(conn, day, cut, params)
        df = df[df["team"].isin(teams)]
        return {(r.team, int(r.player_id)): (float(r.p_play), r.status)
                for r in df.itertuples() if r.player_id == r.player_id}

    def dump_day(conn_, day, train, test, bundle, created_at, elo_pre):
        t0 = time.time()
        test = test.reset_index(drop=True)
        tips = predict._past_tips(conn, day, {})
        state = production.ensure_state(conn, bundle, train)
        params = params_for(day)
        cover, lineups, cells, cells_ship = {}, {}, {}, {}
        for name in CUTOFFS:
            groups: dict[str, list[int]] = {}
            for i, gid in enumerate(test["game_id"]):
                groups.setdefault(g1r.cutoff_of(name, day, tips[gid]), []).append(i)
            lh = np.full(len(test), np.nan)
            la = np.full(len(test), np.nan)
            cv: dict[str, dict] = {}
            cw: dict[str, dict] = {}
            cs: dict[str, dict] = {}
            for cut, idx in groups.items():
                sub = test.iloc[idx]
                teams = set(sub["home_raw"]) | set(sub["away_raw"])
                c = injuries.availability(conn, cut, day, teams)
                w = slate_p(day, cut, params, teams)
                s = slate_p(day, cut, ship, teams)
                for k in idx:
                    gid = test["game_id"].iat[k]
                    cv[gid] = {"cut": cut, "teams": {t: {kk: c[t][kk] for kk in
                               ("source", "captured_at", "avail", "players")}
                               for t in (test["home_raw"].iat[k], test["away_raw"].iat[k])
                               if t in c}}
                    tt = (test["home_raw"].iat[k], test["away_raw"].iat[k])
                    cw[gid] = {key: v for key, v in w.items() if key[0] in tt}
                    cs[gid] = {key: v for key, v in s.items() if key[0] in tt}
                if not c:
                    continue
                h, aw = production.lineup_strengths(state, sub, {t: x["avail"]
                                                                 for t, x in c.items()})
                lh[idx], la[idx] = h, aw
            cover[name], lineups[name], cells[name], cells_ship[name] = cv, (lh, la), cw, cs
        meta = test[["game_id", "home", "away", "home_raw", "away_raw", "season",
                     "season_type"]].copy()
        meta["tip"] = [tips[g] for g in meta["game_id"]]
        rec = {"day": day.isoformat(), "test": meta,
               "state": compact_state(state, set(test["home"]) | set(test["away"])),
               "cover": cover, "lineups": lineups, "cells": cells, "cells_ship": cells_ship,
               "cells_fit_hi": (params.meta.get("fit_hi") if params is not None else None),
               "seconds": time.time() - t0}
        (out / f"day_{day.isoformat()}.pkl").write_bytes(pickle.dumps(rec))
        return []

    predict.price_day = dump_day
    predict.store_predictions = lambda conn, rows: len(rows)
    t0 = time.time()
    res = predict.predict_days(conn, a.start, a.end, log_fn=print, refit_days=100000)
    print(json.dumps(res, default=str)[:400], f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
