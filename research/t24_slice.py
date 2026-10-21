"""T2.4 live-like slice driver (docs/PLAYERS.md §8 T2.4, review item M6).

Runs the projection backtest (`proj/backtest.py`, player level, 2025-26) over the slice
2025-12-22 .. 2026-04-12 under the protocols avail / live (lock-45) / live10 (lock-10), one arm
per process (the arm is the context switch read at import: COURTSIDE_CONTEXT /
COURTSIDE_CONTEXT_EXT), then pools every arm x protocol into the slice report and the monitor's
baseline file.

    python research/t24_slice.py fit                         # bundle of 2025-26 for this arm
    python research/t24_slice.py run PROTOCOL [--sims 4000] [--dates N] [--pplay loo_month]
    python research/t24_slice.py report OUT_DIR [--arm pathA-v0.2] [--baseline PATH]
    python research/t24_slice.py pplay OUT_DIR               # availability-only calibration

`run` works through the dates in chunks and sleeps through the Mac's live lock windows
(14:05-14:55, 18:05-18:55, 20:35-21:25 ET): the per-date cache makes it resumable. It writes
OUT_DIR/players_<arm>_<protocol>.parquet and .meta.json (OUT_DIR = $T24_OUT or ./out).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config, db
from courtside.proj import backtest as B
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

M.GBM_PARAMS["num_threads"] = 1            # shared Mac: <= 3 threads over all jobs
SEASON = "2025-26"
BLACKOUTS_ET = (((14, 5), (14, 55)), ((18, 5), (18, 55)), ((20, 35), (21, 25)))
ARM = P.MODEL_VERSION


def say(m: str) -> None:
    print(f"[t24 {ARM}] {datetime.now():%H:%M:%S} {m}", flush=True)


def wait_for_window() -> None:
    """Sleep while the ET clock is inside a live-lock blackout window."""
    while True:
        now = datetime.now(config.NBA_TZ)
        hm = (now.hour, now.minute)
        hit = [b for b in BLACKOUTS_ET if b[0] <= hm < b[1]]
        if not hit:
            return
        end = now.replace(hour=hit[0][1][0], minute=hit[0][1][1], second=0, microsecond=0)
        secs = max(30.0, (end - now).total_seconds())
        say(f"blackout {hit[0]}: sleeping {secs / 60:.0f} min")
        time.sleep(secs)


def out_dir(arg: str | None = None) -> Path:
    p = Path(arg or os.environ.get("T24_OUT", "out"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_fit(a) -> None:
    conn = db.connect()
    t0 = time.time()
    b = P.load_bundle(conn, SEASON, None, log=say)
    say(f"bundle {SEASON} ready in {time.time() - t0:.0f}s; context "
        f"{None if b.context is None else b.context.train_seasons}")


def cmd_run(a) -> None:
    conn = db.connect()
    st = B.Settings(protocol=a.protocol, n_sims=a.sims, modes=(), baselines=B.BASELINES,
                    pplay=(a.pplay or "loo_month") if a.protocol in B.LIVE_OFFSETS else "")
    sr = B.SeasonRun(conn, SEASON, st, log=say, need_context=False)
    days = [d for d in sr.dates(0) if B.SLICE[0] <= d <= B.SLICE[1]]
    days = B.E._pick(days, a.dates)
    cache = sr.cache()
    say(f"{a.protocol}: {len(days)} dates, cache {cache.root}")
    pending = [d for d in days if not cache.has(d)]
    t0 = time.time()
    for i in range(0, len(pending), a.chunk):
        wait_for_window()
        if sr.ctx is None:
            sr.load()
        B._run_days(sr, pending[i:i + a.chunk], cache, say)
        say(f"  {min(i + a.chunk, len(pending))}/{len(pending)} done, {time.time() - t0:.0f}s")
    players, _, meta = B.run_season(conn, SEASON, st, log=say, date_lo=B.SLICE[0],
                                    date_hi=B.SLICE[1], n_dates=a.dates)
    od = out_dir(a.out)
    tag = f"{ARM}_{a.protocol}" + (f"_{a.pplay}" if a.pplay and a.pplay != "loo_month" else "")
    players.to_parquet(od / f"players_{tag}.parquet")
    meta.update({"arm": ARM, "protocol": a.protocol, "pplay": st.pplay, "n_sims": a.sims})
    (od / f"players_{tag}.meta.json").write_text(json.dumps(B.E._clean(meta), indent=1))
    say(f"wrote {od / f'players_{tag}.parquet'} ({len(players)} rows, skipped "
        f"{len(meta['skipped'])}, env {meta.get('env_sources')})")


def load_frames(od: Path) -> dict:
    frames = {}
    for f in sorted(od.glob("players_*.parquet")):
        m = json.loads(f.with_suffix(".meta.json").read_text()) if \
            f.with_suffix(".meta.json").exists() else {}
        stem = f.stem[len("players_"):]
        if "_shipped" in stem:
            continue                       # in-sample P(play) variant: reported separately
        arm, proto = m.get("arm"), m.get("protocol")
        if not arm or not proto:
            arm, proto = stem.rsplit("_", 1)
        frames[(arm, proto)] = pd.read_parquet(f)
    return frames


def cmd_report(a) -> None:
    od = out_dir(a.out)
    frames = load_frames(od)
    say(f"frames: {sorted(frames)}")
    rep = B.slice_report(frames, reference="avail", primary="live", base_arm=a.arm)
    metas = {f.stem: json.loads(f.read_text()) for f in sorted(od.glob("players_*.meta.json"))}
    rep["run_meta"] = {k: {x: v.get(x) for x in ("arm", "protocol", "pplay", "n_sims", "n_dates",
                                                 "n_run", "skipped", "env_sources")}
                       for k, v in metas.items()}
    (od / "slice_report.json").write_text(json.dumps(B.E._clean(rep), indent=1))
    txt = B.format_slice_report(rep)
    (od / "slice_report.txt").write_text(txt + "\n")
    print(txt)
    base = B.live_baseline_json(rep, arm=a.arm, protocol="live",
                                note="lock-45 per game, report P(play) leave-one-month-out, "
                                     "closing lines (no 2025-26 line snapshots); Dec-Apr, so "
                                     "October nights (prior-heavy rates) may run worse")
    if "live10" in rep["protocols"]:
        b10 = B.live_baseline_json(rep, arm=a.arm, protocol="live10")
        base["lock10_reference"] = {k: b10[k] for k in ("min_mae", "daily", "window300", "means")}
    p = Path(a.baseline) if a.baseline else od / "live_like_baseline.json"
    p.write_text(json.dumps(B.E._clean(base), indent=1))
    say(f"wrote {p}")


def cmd_pplay(a) -> None:
    """Availability-only P(play) calibration at each game's tip - 45 by status: shipped params
    (in-sample) vs leave-one-month-out (what the slice uses). No simulation."""
    from courtside.proj import availability as AV
    conn = db.connect()
    rows = []
    li = {m: B.LiveInputs(conn, SEASON, m, log=say) for m in B.PPLAY_MODES}
    day = date.fromisoformat(B.SLICE[0])
    hi = date.fromisoformat(B.SLICE[1])
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM team_games WHERE game_date BETWEEN ? AND ? AND "
        "season_type='Regular Season' ORDER BY 1", (B.SLICE[0], B.SLICE[1]))]
    for ds in days:
        d = date.fromisoformat(ds)
        tips = AV.day_games(conn, d)
        for tip, g in tips.groupby("tip"):
            at = tip - timedelta(minutes=45)
            for mode, L in li.items():
                av = AV.p_play_slate(conn, d, at, L.params(d), hist=L.hist)
                av = av[av["game_id"].isin(set(g["game_id"]))]
                played = {(r[0], int(r[1])) for r in conn.execute(
                    f"SELECT game_id, player_id FROM player_games WHERE min>0 AND game_id IN "
                    f"({','.join('?' * len(g))})", list(g["game_id"]))}
                for r in av.itertuples(index=False):
                    rows.append({"date": ds, "mode": mode, "game_id": r.game_id,
                                 "player_id": int(r.player_id), "status": r.status,
                                 "p": float(r.p_play),
                                 "played": int((r.game_id, int(r.player_id)) in played)})
    df = pd.DataFrame(rows)
    od = out_dir(a.out)
    df.to_parquet(od / "pplay_rows.parquet")
    out = {}
    for mode, x in df.groupby("mode"):
        x = x.assign(played=x["played"].astype(float), p_play=x["p"], p_avail=x["p"])
        out[mode] = B.pplay_by_status(x)
    (od / "pplay_status.json").write_text(json.dumps(B.E._clean(out), indent=1))
    for mode, t in out.items():
        print(f"\n{mode}")
        for r in t:
            print(f"  {r['label']:<30} n={r['n']:<6} P {r['p_play']:.3f} played {r['played']:.3f}"
                  f" gap {r['gap']:+.3f} ({r['gap_se'] or float('nan'):.3f}) Brier {r['brier']:.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    r = sub.add_parser("run")
    r.add_argument("protocol", choices=("avail", *B.LIVE_OFFSETS))
    r.add_argument("--sims", type=int, default=4_000)
    r.add_argument("--dates", type=int, default=0)
    r.add_argument("--chunk", type=int, default=6)
    r.add_argument("--pplay", choices=B.PPLAY_MODES, default="")
    r.add_argument("--out", default=None)
    rp = sub.add_parser("report")
    rp.add_argument("out")
    rp.add_argument("--arm", default="pathA-v0.2")
    rp.add_argument("--baseline", default=None)
    pp = sub.add_parser("pplay")
    pp.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        a.out = a.out
    {"fit": cmd_fit, "run": cmd_run, "report": cmd_report, "pplay": cmd_pplay}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
