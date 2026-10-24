"""G1 (IMPROVEMENTS §4) replay: per-day fitted production pair + injury lineups at many times.

Runs `predict.predict_days` (the production pair, refit every day on games strictly before the
day) on a database opened **read-only**, with `predict.price_day` replaced by a dumper: nothing
is priced or stored. Per NBA day it pickles to `--out/day_<d>.pkl`:

* `scorers`: the day's fitted `gbm_market` / `gbm_opinion` (`production.Scorer`, small),
* `F`: the day's feature rows (no line, no lineup override; the engine's own availability),
* `test`: game_id, home/away (franchise + raw), season, season_type, tip (ISO UTC),
* `lineups[name]`: expected-lineup strengths (home, away) from `injuries.availability` at the
  cutoff `name` (below), `listed[name]` whether the newest usable batch listed each team, and
  `fresh[name]` whether any usable batch existed (if not the lineup is NaN: the analysis then
  uses the full recent roster, which is what a live run without a report does).

Cutoffs (UTC, each capped at the game's tip): d1_17 / d1_20 / d1_23 = 17:00, 20:00, 23:00 the day
before; d0_03 = 03:00 of the NBA day (10-11 PM ET the evening before); d0_17 / d0_21 = 17:00 and
21:00 of the day (about noon and 4-5 PM ET); tip90 = 90 min before tip; tip = the tip itself.
The official archive has no batches between ~03:00Z and 16:00Z, so d0_03 also stands for a morning
(10 AM ET) run.

    run.sh code/research/g1_line_news_replay.py --db <scratch-gd.db> --start 2025-12-22 \\
        --end 2026-02-28 --out out/days
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

CUTOFFS = ("d1_17", "d1_20", "d1_23", "d0_03", "d0_17", "d0_21", "tip90", "tip")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cutoff_of(name: str, day, tip: str) -> str:
    t = datetime.fromisoformat(tip.replace("Z", "+00:00"))
    d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    fixed = {"d1_17": d0 - timedelta(hours=7), "d1_20": d0 - timedelta(hours=4),
             "d1_23": d0 - timedelta(hours=1), "d0_03": d0 + timedelta(hours=3),
             "d0_17": d0 + timedelta(hours=17), "d0_21": d0 + timedelta(hours=21)}
    if name == "tip":
        c = t
    elif name == "tip90":
        c = t - timedelta(minutes=90)
    else:
        c = fixed[name]
    return _iso(min(c, t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="database (opened read-only)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True, help="directory for day_<d>.pkl")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from courtside import injuries, predict, production

    def dump_day(conn, day, train, test, bundle, created_at, elo_pre):
        t0 = time.time()
        pos = {g: i for i, g in enumerate(bundle.game_ids)}
        F = bundle.F.iloc[[pos[g] for g in test["game_id"]]].reset_index(drop=True)
        test = test.reset_index(drop=True)
        tips = predict._past_tips(conn, day, {})
        state = production.ensure_state(conn, bundle, train)
        lineups, listed, fresh, caps = {}, {}, {}, {}
        for name in CUTOFFS:
            lh = np.full(len(test), np.nan)
            la = np.full(len(test), np.nan)
            lst = np.zeros((len(test), 2), bool)
            fr = np.zeros(len(test), bool)
            cp = [[None, None] for _ in range(len(test))]
            groups: dict[str, list[int]] = {}
            for i, gid in enumerate(test["game_id"]):
                groups.setdefault(cutoff_of(name, day, tips[gid]), []).append(i)
            for cut, idx in groups.items():
                sub = test.iloc[idx]
                cover = injuries.availability(conn, cut, day,
                                              set(sub["home_raw"]) | set(sub["away_raw"]))
                if not cover:
                    continue
                h, w = production.lineup_strengths(state, sub, {t: c["avail"]
                                                                for t, c in cover.items()})
                lh[idx], la[idx] = h, w
                for k, (_, r) in zip(idx, sub.iterrows()):
                    fr[k] = True
                    for j, s in enumerate(("home_raw", "away_raw")):
                        c = cover.get(r[s]) or {}
                        lst[k, j] = c.get("source") is not None
                        cp[k][j] = c.get("captured_at")
            lineups[name], listed[name], fresh[name], caps[name] = (lh, la), lst, fr, cp
        meta = test[["game_id", "home", "away", "home_raw", "away_raw", "season",
                     "season_type"]].copy()
        meta["tip"] = [tips[g] for g in meta["game_id"]]
        rec = {"day": day.isoformat(), "scorers": bundle.scorers, "F": F, "test": meta,
               "lineups": lineups, "listed": listed, "fresh": fresh, "caps": caps,
               "train_games": bundle.train_games, "seconds": time.time() - t0}
        (out / f"day_{day.isoformat()}.pkl").write_bytes(pickle.dumps(rec))
        return []

    predict.price_day = dump_day
    predict.store_predictions = lambda conn, rows: len(rows)
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    t0 = time.time()
    res = predict.predict_days(conn, a.start, a.end, log_fn=print, refit_days=1)
    print(json.dumps(res, default=str)[:400], f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
