"""B9: line freshness in `pipeline.environment()` (`COURTSIDE_LINE_FRESH=1`, `+lf`), descriptive.

For each finished game since the season start, at the pregame run times (10:00, 13:30, 17:00,
18:30 ET on the game day, only runs before tip) compare the implied team points from
`environment()` OFF vs ON against the closing line (`environment()` OFF at tip: all books'
latest pre-tip snapshot; also the ON close). Reports the mean absolute gap of each team's implied
points to the close, and how many game-runs / books the bound dropped.

    COURTSIDE_DATA=var/tmp/b9 uv run python research/line_fresh_eval.py [--since 2026-10-20]
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from courtside import config, db
from courtside.proj import pipeline as P

RUNS = [time(10, 0), time(13, 30), time(17, 0), time(18, 30)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-20")
    a = ap.parse_args()
    conn = db.connect(config.db_path())
    games = conn.execute("SELECT game_id, nba_day, tipoff_utc FROM games WHERE nba_day>=? AND "
                         "status='final' ORDER BY tipoff_utc", (a.since,)).fetchall()
    rows = []
    for gid, day, tip in games:
        tip_dt = datetime.fromisoformat(tip.replace("Z", "+00:00"))
        d = datetime.fromisoformat(day).date()
        P.set_line_fresh(False)
        close = P.environment(conn, gid, tip_dt, mode="live")
        P.set_line_fresh(True)
        close_on = P.environment(conn, gid, tip_dt, mode="live")
        for rt in RUNS:
            at = datetime.combine(d, rt, tzinfo=config.NBA_TZ).astimezone(timezone.utc)
            if at >= tip_dt:
                continue
            P.set_line_fresh(False)
            off = P.environment(conn, gid, at, mode="live")
            P.set_line_fresh(True)
            on = P.environment(conn, gid, at, mode="live")
            if not off["source"].startswith("books:") or not close["source"].startswith("books:"):
                continue
            n_off, n_on = int(off["source"][6:]), int(on["source"][6:])
            rows.append({"game_id": gid, "run": rt.strftime("%H:%M"), "n_off": n_off,
                         "n_on": n_on, "dropped": n_off - n_on,
                         "age_max_off": off["line_age_max_min"],
                         "age_max_on": on["line_age_max_min"],
                         "gap_off": (abs(off["home_pts"] - close["home_pts"])
                                     + abs(off["away_pts"] - close["away_pts"])) / 2,
                         "gap_on": (abs(on["home_pts"] - close["home_pts"])
                                    + abs(on["away_pts"] - close["away_pts"])) / 2,
                         "gap_off_c2": (abs(off["home_pts"] - close_on["home_pts"])
                                        + abs(off["away_pts"] - close_on["away_pts"])) / 2,
                         "gap_on_c2": (abs(on["home_pts"] - close_on["home_pts"])
                                       + abs(on["away_pts"] - close_on["away_pts"])) / 2,
                         "tot_gap_off": abs(off["mu_total"] - close["mu_total"]),
                         "tot_gap_on": abs(on["mu_total"] - close["mu_total"])})
    P.set_line_fresh(False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"games {len(games)}  game-runs {len(df)}  (games with books: {df['game_id'].nunique()})")
    g = df.groupby("run").agg(n=("gap_off", "size"), dropped_runs=("dropped", lambda x: (x > 0).sum()),
                              books_dropped=("dropped", "sum"), gap_off=("gap_off", "mean"),
                              gap_on=("gap_on", "mean"), tot_off=("tot_gap_off", "mean"),
                              tot_on=("tot_gap_on", "mean"))
    print(g.round(3))
    print("ALL: n=%d  runs with a drop=%d  books dropped=%d  games with a drop=%d" % (
        len(df), (df["dropped"] > 0).sum(), df["dropped"].sum(),
        df.loc[df["dropped"] > 0, "game_id"].nunique()))
    for c in ("gap_off", "gap_on", "gap_off_c2", "gap_on_c2", "tot_gap_off", "tot_gap_on",
              "age_max_off", "age_max_on"):
        print(f"  mean {c:12s} {df[c].mean():.3f}")
    d = df[df["dropped"] > 0]
    if len(d):
        print("on runs where books were dropped: n=%d  gap_off=%.3f  gap_on=%.3f" % (
            len(d), d["gap_off"].mean(), d["gap_on"].mean()))
        print(d[["game_id", "run", "n_off", "n_on", "age_max_off", "age_max_on", "gap_off",
                 "gap_on"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
