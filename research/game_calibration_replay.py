"""Day-by-day production replay of one season for the game-calibration study.

Runs `predict.predict_days` (the production pair, refit every day on games strictly before the
day, closing lines as the market input) on a *scratch copy* of the database and exports the
stored `gbm_market` / `gbm_opinion` rows to CSV. Two variants:

* `--injuries keep`  : archived injury snapshots are priced exactly as production would
  (official report / ESPN -> expected lineups). For 2025-26 the official report archive starts
  2025-12-22, so earlier days fall back to the engine's leak-safe "regulars who played".
* `--injuries drop`  : the copy's `injury_snapshots` are emptied first, so every day uses the
  "regulars who played" availability (what the original 838-game replay effectively had).

The scratch copy is written (predictions, and `injury_snapshots` for `drop`); never point
`--db` at a production database.

    uv run python research/game_calibration_replay.py --src /path/to/prod.db \\
        --db scratch/replay_keep.db --injuries keep --start 2025-10-21 --end 2026-04-12 \\
        --out scratch/replay_keep.csv.gz
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd


def copy_db(src: str, dst: str) -> None:
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    s.backup(d)
    d.close()
    s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="database to copy from (opened read-only)")
    ap.add_argument("--db", required=True, help="scratch database (created from --src)")
    ap.add_argument("--injuries", choices=("keep", "drop"), default="keep")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--refit-days", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.src:
        if Path(a.db).exists():
            raise SystemExit(f"{a.db} exists; refusing to overwrite")
        copy_db(a.src, a.db)
    from courtside import predict

    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("DELETE FROM predictions WHERE nba_day BETWEEN ? AND ?", (a.start, a.end))
        if a.injuries == "drop":
            conn.execute("DELETE FROM injury_snapshots")
    t0 = time.time()
    res = predict.predict_days(conn, a.start, a.end, log_fn=print, refit_days=a.refit_days)
    print(json.dumps(res, default=str)[:400], f"{time.time() - t0:.0f}s")
    rows = []
    for r in conn.execute(
            "SELECT game_id, nba_day, model, home_win_prob, home_margin, market_home_prob, "
            "market_home_spread, inputs_json FROM predictions WHERE nba_day BETWEEN ? AND ? "
            "AND model IN ('gbm_market','gbm_opinion')", (a.start, a.end)):
        j = json.loads(r["inputs_json"])
        av = j.get("availability") or {}
        rows.append({"game_id": r["game_id"], "nba_day": r["nba_day"], "model": r["model"],
                     "p": r["home_win_prob"], "margin_pred": r["home_margin"],
                     "market_home_prob": r["market_home_prob"],
                     "market_home_spread": r["market_home_spread"],
                     "basis_home": (av.get("home") or {}).get("basis"),
                     "basis_away": (av.get("away") or {}).get("basis")})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
