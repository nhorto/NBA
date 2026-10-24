"""G1 (IMPROVEMENTS §4): how stale is the line the live game model prices with?

Read-only on a database copy (e.g. the nightly prod snapshot gunzipped to var/tmp/g1/). For every
champion `gbm_market` row created before its (final) game's tip-off:

* age     = created_at - inputs.market.captured_at (minutes): how old the line was when priced,
* to_tip  = tip - created_at (hours): which pregame run it was,
* move    = |line used - last pre-tip line| (spread and total), where the last pre-tip line is
            exactly what `predict.market_line(conn, gid, tip)` returns (the newest snapshot batch
            at or before the tip): the closest thing to a close the live archive has,
* last_age = tip - captured_at of that last pre-tip line (how far before the tip the archive's
            "close" was captured).

    uv run python research/g1_staleness.py --db var/tmp/g1/prod.db --from 2026-10-20 \\
        --out var/tmp/g1/staleness.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd


def _ts(s: str | None):
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--from", dest="start", default="2026-10-20")
    ap.add_argument("--out")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    from courtside import predict

    tips = {r["game_id"]: r["tipoff_utc"] for r in conn.execute(
        "SELECT game_id, tipoff_utc FROM games WHERE nba_day >= ? AND status = 'final'",
        (a.start,))}
    last = {g: predict.market_line(conn, g, t) for g, t in tips.items() if t}
    rows = []
    for r in conn.execute(
            "SELECT game_id, created_at, inputs_json, market_home_spread, market_total "
            "FROM predictions WHERE model='gbm_market' AND (role IS NULL OR role='champion') "
            "AND nba_day >= ?", (a.start,)):
        tip = tips.get(r["game_id"])
        if not tip or r["created_at"] >= tip:
            continue
        mk = (json.loads(r["inputs_json"]) or {}).get("market") or {}
        lc = last.get(r["game_id"]) or {}
        cr, tp = _ts(r["created_at"]), _ts(tip)
        cap = _ts(mk.get("captured_at"))
        rows.append({
            "game_id": r["game_id"], "created_at": r["created_at"], "tip": tip,
            "kind": mk.get("kind"), "source": mk.get("source"),
            "to_tip_h": (tp - cr).total_seconds() / 3600,
            "age_min": (cr - cap).total_seconds() / 60 if cap else np.nan,
            "last_age_min": ((tp - _ts(lc["captured_at"])).total_seconds() / 60
                             if lc.get("captured_at") else np.nan),
            "d_spread": (r["market_home_spread"] - lc["home_spread"]
                         if r["market_home_spread"] is not None
                         and lc.get("home_spread") is not None else np.nan),
            "d_total": (r["market_total"] - lc["total"]
                        if r["market_total"] is not None and lc.get("total") is not None
                        else np.nan)})
    df = pd.DataFrame(rows)
    bins = [0, 2, 4, 8, 13, 48]
    df["bucket"] = pd.cut(df["to_tip_h"], bins, right=False,
                          labels=["0-2h", "2-4h", "4-8h", "8-13h", "13h+"])

    def summ(g):
        return pd.Series({
            "rows": len(g), "games": g["game_id"].nunique(),
            "age_med_min": g["age_min"].median(), "age_p90_min": g["age_min"].quantile(0.9),
            "spread_moved_share": (g["d_spread"].abs() > 0).mean(),
            "abs_d_spread": g["d_spread"].abs().mean(),
            "abs_d_spread_p90": g["d_spread"].abs().quantile(0.9),
            "abs_d_total": g["d_total"].abs().mean(),
            "n_total": g["d_total"].notna().sum()})
    tab = df.groupby("bucket", observed=True).apply(summ, include_groups=False)
    tab.loc["all"] = summ(df)
    print(f"{len(df)} pre-tip rows, {df['game_id'].nunique()} games, kinds "
          f"{df['kind'].value_counts().to_dict()}")
    print(f"last pre-tip line captured median {df.drop_duplicates('game_id')['last_age_min'].median():.0f}"
          f" min before tip (p90 {df.drop_duplicates('game_id')['last_age_min'].quantile(0.9):.0f})")
    print(tab.round(3).to_string())
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"table": tab.round(4).reset_index().to_dict("records"),
                       "n_rows": len(df), "n_games": int(df["game_id"].nunique()),
                       "last_age_med_min": float(df.drop_duplicates("game_id")["last_age_min"]
                                                 .median())}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
