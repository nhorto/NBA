"""G4 OFF identity: dump the stripped `predict_day` rows (no wall-clock fields) as JSON, to `cmp`
the output of the tree with the absence-scale hook (switch OFF) against the tree before it.

    PYTHONPATH=<tree>/src:tests python research/as_off_identity.py --out a.json            # synth
    PYTHONPATH=<tree>/src python research/as_off_identity.py --db <ro db> --day 2026-01-15 \\
        --out b.json                                                                     # real day
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date

INS = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
       "VALUES (?,?,?,?,?,?)")


def strip(rows):
    def one(r):
        out = {k: v for k, v in r.items() if k not in ("created_at", "opinion")}
        out["inputs"] = {k: v for k, v in r["inputs"].items() if k != "fit"}
        if r.get("opinion"):
            out["opinion"] = one(r["opinion"])
        return out
    return [one(r) for r in rows]


def synth_worlds():
    from synth import day_date, synth_conn

    from courtside import predict
    predict.MIN_TRAIN_GAMES = 20
    day = day_date("2024-25", 22)
    out = {}
    conn = synth_conn()
    out["no_report"] = strip(predict.predict_day(conn, day, store=False))
    g = out["no_report"][0]
    at = f"{day}T15:00:00Z"
    for k, (team, status) in enumerate(((g["home"], "Out"), (g["away"], "Questionable"),
                                        (g["away"], "Doubtful"))):
        conn.execute(INS, (at, "nba_official", team, f"{team} Player {k}", status, "knee"))
    conn.commit()
    out["report"] = strip(predict.predict_day(conn, day, store=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--day")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.db:
        from courtside import predict
        conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        res = {a.day: strip(predict.predict_day(conn, date.fromisoformat(a.day), store=False))}
    else:
        res = synth_worlds()
    with open(a.out, "w") as fh:
        json.dump(res, fh, sort_keys=True, default=str, indent=0)
    print(a.out, {k: len(v) for k, v in res.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
