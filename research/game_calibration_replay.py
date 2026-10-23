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

IMPROVEMENTS B3 / game-day rows (docs/research/game-calibration.md §6):

* `--arms legacy:0,tip:0,tip:1` prices every day once per arm with the same daily fit (rows of
  the first arm are also stored in the scratch DB; one CSV per arm, `--out` with `{arm}`).
  `tip` (the code as shipped): each past game prices the newest report at or before its own
  tip. `legacy`: `predict.injury_cutoff` monkeypatched back to the pre-B3 end of the NBA day
  (05:00 UTC next day, after every tip). `virtual` (sensitivity): the 12:00Z virtual time for
  every game, ignoring tips. `:1` = `injuries.set_inj_gameday_game(True)`
  (COURTSIDE_INJ_GAMEDAY_GAME): only the official rows filed for the game's own date (needs
  schema 16 + `courtside backfill injuries --game-date`).
* `--fast`: skip `production.injury_effects` (the per-player "what is his absence worth"
  explanation re-scores). It feeds only `inputs.model_effects` and the absentee sort order, not
  the probabilities (checked: identical p on a 2026-01 sample with and without it).

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


def legacy_cutoff(day, created_at, tip=None):
    """The pre-B3 `predict.injury_cutoff`: now for live days, else 05:00 UTC the next day."""
    from datetime import datetime, timedelta, timezone

    from courtside import config
    if day >= config.today_nba_day():
        return created_at
    return (datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            + timedelta(days=1, hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(r: dict, tips: dict) -> dict:
    j = r["inputs"]
    av = j.get("availability") or {}
    caps = [c for c in ((av.get(s) or {}).get("captured_at") for s in ("home", "away")) if c] \
        + list((j.get("injuries_batches") or {}).values())
    return {"game_id": r["game_id"], "nba_day": r["nba_day"], "model": r["model"],
            "p": r["home_win_prob"], "margin_pred": r["home_margin"],
            "market_home_prob": r["market_home_prob"],
            "market_home_spread": r["market_home_spread"],
            "basis_home": (av.get("home") or {}).get("basis"),
            "basis_away": (av.get("away") or {}).get("basis"),
            "src_home": (av.get("home") or {}).get("source"),
            "src_away": (av.get("away") or {}).get("source"),
            "inj_at": j.get("injuries_at"), "max_captured_at": max(caps, default=None),
            "tip": tips.get(r["game_id"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="database to copy from (opened read-only)")
    ap.add_argument("--db", required=True, help="scratch database (created from --src)")
    ap.add_argument("--injuries", choices=("keep", "drop"), default="keep")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--refit-days", type=int, default=1)
    ap.add_argument("--out", required=True, help="csv path; with several --arms it must "
                    "contain {arm}")
    ap.add_argument("--arms", default="tip:0",
                    help="comma list of cutoff:gameday_game, cutoff in {tip, legacy}, e.g. "
                         "legacy:0,tip:0,tip:1 (the same daily fits price every arm)")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    arms = [(c, int(g)) for c, g in (x.split(":") for x in a.arms.split(","))]
    assert all(c in ("tip", "legacy", "virtual") for c, _ in arms)
    if len(arms) > 1 and "{arm}" not in a.out:
        raise SystemExit("--out needs {arm} with several arms")

    if a.src:
        if Path(a.db).exists():
            raise SystemExit(f"{a.db} exists; refusing to overwrite")
        copy_db(a.src, a.db)
    from courtside import injuries, predict, production

    if a.fast:
        production.injury_effects = lambda *args, **kw: []
    tip_cutoff, real_price = predict.injury_cutoff, predict.price_day

    def virtual_cutoff(day, created_at, tip=None):
        """Sensitivity arm: ignore the tip, i.e. the 12:00Z virtual time of
        `default_created_at` (an 8 AM ET morning run) for every past game."""
        return tip_cutoff(day, created_at)
    recs: dict[str, list[dict]] = {f"{c}_{'on' if g else 'off'}": [] for c, g in arms}

    def price_all(conn, day, *args, **kw):
        tips = predict._past_tips(conn, predict._as_date(day), {})
        out = None
        for c, g in reversed(arms):                 # the first arm last: its rows are stored
            predict.injury_cutoff = {"legacy": legacy_cutoff, "tip": tip_cutoff,
                                     "virtual": virtual_cutoff}[c]
            injuries.set_inj_gameday_game(bool(g))
            out = real_price(conn, day, *args, **kw)
            lab = f"{c}_{'on' if g else 'off'}"
            for r in out:
                recs[lab] += [_record(r, tips), _record(r["opinion"], tips)]
        return out
    predict.price_day = price_all
    print(f"arms={arms} fast={a.fast}")

    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("DELETE FROM predictions WHERE nba_day BETWEEN ? AND ?", (a.start, a.end))
        if a.injuries == "drop":
            conn.execute("DELETE FROM injury_snapshots")
    t0 = time.time()
    res = predict.predict_days(conn, a.start, a.end, log_fn=print, refit_days=a.refit_days)
    print(json.dumps(res, default=str)[:400], f"{time.time() - t0:.0f}s")
    for lab, rows in recs.items():
        out = a.out.replace("{arm}", lab)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
