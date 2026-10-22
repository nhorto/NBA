"""`courtside proj slate|day|backtest`: Path A v0 projections (T1.4).

Register by adding ``"courtside.proj.cli_proj"`` to ``PROJ_MODULES`` in ``proj/cli.py``.

    courtside proj slate --slate 154575 [--at 2026-10-20T18:15:00Z] [--sims 25000] [--no-store]
    courtside proj day --date 2026-10-20 [--at ISO] [--store]
    courtside proj day --date 2025-01-15 --historical [--protocol avail|reports]
    courtside proj backtest --season 2024-25 --dates 60
    courtside proj weekly-report [--week-end 2026-10-26] [--days 7] [--json] [--no-write]
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd


def add_commands(sub) -> dict:
    p = sub.add_parser("slate", help="project a DK slate (Path A v0) and store the run")
    p.add_argument("--slate", required=True, help="draft group id")
    p.add_argument("--at", default=None, help="ISO time (default now, UTC)")
    p.add_argument("--sims", type=int, default=25_000)
    p.add_argument("--run-id", default=None)
    p.add_argument("--kind", default=None, help="lock45 / lock10 / swap / manual")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--top", type=int, default=40, help="rows to print (0 = all)")
    d = sub.add_parser("day", help="project every game of an NBA day (NBA player ids)")
    d.add_argument("--date", required=True)
    d.add_argument("--at", default=None, help="ISO time (default now; live mode)")
    d.add_argument("--sims", type=int, default=25_000)
    d.add_argument("--historical", action="store_true",
                   help="backtest mode: closing lines, data before the date only")
    d.add_argument("--protocol", choices=("avail", "reports"), default="avail")
    d.add_argument("--store", action="store_true", help="write player_projections (live)")
    d.add_argument("--top", type=int, default=40)
    b = sub.add_parser("backtest", help="quick walk-forward DK FP evaluation (fit < season)")
    b.add_argument("--season", default="2024-25")
    b.add_argument("--dates", type=int, default=60)
    b.add_argument("--sims", type=int, default=4_000)
    b.add_argument("--protocol", choices=("avail", "reports"), default="avail")
    b.add_argument("--json", default=None, help="write the summary JSON here")
    w = sub.add_parser("weekly-report", help="weekly DFS report (T3.3): projections and lineups "
                       "vs the baselines, clustered SEs, real vs simulated kept apart")
    w.add_argument("--week-end", default=None,
                   help="last NBA day of the window (default: the newest graded day)")
    w.add_argument("--days", type=int, default=7)
    w.add_argument("--json", action="store_true", help="print the JSON instead of the text")
    w.add_argument("--no-write", action="store_true",
                   help="do not write var/reports/weekly_dfs_<date>.json")
    w.add_argument("--out-dir", default=None, help="report directory (default var/reports)")
    return {"slate": _slate_cmd, "day": _day_cmd, "backtest": _backtest_cmd,
            "weekly-report": _weekly_report_cmd}


def _conn():
    from .. import db
    return db.connect()


def _at(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _names(conn, pids) -> dict:
    pids = [int(p) for p in pids]
    if not pids:
        return {}
    q = ",".join("?" * len(pids))
    rows = conn.execute(f"SELECT player_id, first_name || ' ' || last_name FROM player_index "
                        f"WHERE player_id IN ({q})", pids).fetchall()
    out = {int(r[0]): r[1] for r in rows}
    missing = [p for p in pids if p not in out]
    if missing:
        q = ",".join("?" * len(missing))
        for r in conn.execute(f"SELECT player_id, MAX(player_name) FROM player_games WHERE "
                              f"player_id IN ({q}) GROUP BY player_id", missing):
            out[int(r[0])] = r[1]
    return out


def print_table(conn, players: pd.DataFrame, top: int = 40) -> None:
    pl = players.sort_values("dk_mean", ascending=False)
    if top:
        pl = pl.head(top)
    names = _names(conn, pl["player_id"])
    print(f"{'player':<24} {'team':<4} {'tier':<5} {'P(av)':>5} {'P(pl)':>5} {'min':>5} "
          f"{'mean':>6} {'p10':>6} {'p25':>6} {'p50':>6} {'p90':>6} {'pts':>5}")
    for r in pl.itertuples(index=False):
        nm = (names.get(int(r.player_id)) or str(r.player_id))[:24]
        print(f"{nm:<24} {r.team:<4} {r.tier:<5} {r.p_avail:5.2f} {r.p_play:5.2f} "
              f"{r.min_played:5.1f} {r.dk_mean:6.2f} {r.dk_q10:6.2f} {r.dk_q25:6.2f} "
              f"{r.dk_q50:6.2f} {r.dk_q90:6.2f} {r.pts_all:5.1f}")


def _print_meta(meta: dict) -> None:
    rec = meta.get("recon", {})
    print(f"{meta.get('games')} games, {meta.get('n_sims')} sims, seed {meta.get('seed')}; "
          f"sim {meta.get('seconds_sim')}s, total {meta.get('seconds_total', meta.get('seconds'))}s; "
          f"reconciliation max |resid| {rec.get('max_abs_resid', 0):.3f}, max |gap| "
          f"{rec.get('max_abs_gap', 0):.1f}, data checks {rec.get('data_checks')}")
    for gid, e in (meta.get("env") or {}).items():
        print(f"  {gid}: home margin {e['mu_margin']:+.1f}, total {e['mu_total']:.1f} "
              f"({e['source']})")


def _slate_cmd(args) -> int:
    from . import pipeline as P
    conn = _conn()
    run_id = args.run_id or f"manual-{uuid.uuid4().hex[:10]}"
    try:
        out = P.project_slate(conn, args.slate, _at(args.at), run_id=run_id, n_sims=args.sims,
                              kind=args.kind, store=not args.no_store, log=print)
    except (LookupError, ValueError) as e:
        print(f"error: {e}")
        return 2
    print_table(conn, out.result.players, args.top)
    _print_meta(out.meta)
    print(f"run {run_id}: {len(out)} DK players projected of {out.meta['dk_pool']} "
          f"({len(out.meta['dk_unprojected'])} without a projection)"
          + (f"; sims {out.meta.get('sim_path')}" if out.meta.get("sim_path") else ""))
    return 0


def _day_cmd(args) -> int:
    from . import pipeline as P
    conn = _conn()
    if args.historical:
        res = P.project_historical(conn, args.date, protocol=args.protocol, n_sims=args.sims,
                                   log=print)
    else:
        res = P.project_day(conn, args.date, _at(args.at), n_sims=args.sims, log=print)
        if args.store:
            run_id = f"day-{args.date}-{uuid.uuid4().hex[:8]}"
            info = P.save_run(conn, res, run_id=run_id, slate_id=None, kind="manual")
            print(f"stored run {run_id} -> {info['sim_path']}")
    print_table(conn, res.players, args.top)
    _print_meta(res.meta)
    return 0


def _backtest_cmd(args) -> int:
    from . import pipeline as P
    conn = _conn()
    res = P.backtest(conn, args.season, n_dates=args.dates, n_sims=args.sims,
                     protocol=args.protocol, log=print)
    res.pop("frame", None)
    print(P.format_backtest(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


def _weekly_report_cmd(args) -> int:
    from datetime import date
    from pathlib import Path

    from . import reports as RP
    conn = _conn()
    try:
        if args.week_end:
            end = date.fromisoformat(args.week_end)
        else:
            end = RP.latest_graded_day(conn)
            if end is None:
                print("nothing has been graded yet; pass --week-end to report an empty window")
                return 0
        rep = RP.weekly_dfs_report(conn, end, days=args.days)
    finally:
        conn.close()
    path = None if args.no_write else RP.write_report(
        rep, Path(args.out_dir) if args.out_dir else None)
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(RP.render_text(rep))
    if path:
        print(f"wrote {path}")
    return 0
