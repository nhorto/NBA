"""`courtside dfs refresh` and `courtside dfs capture-status`: the light DraftKings capture.

Add ``"courtside.proj.dfs.capture_cli"`` to ``DFS_MODULES`` in ``proj/cli.py``. The capture itself
(slates, payout tables, request budget) is ``courtside.sources.dfs_capture``; the daily
collector runs it through ``courtside collect pregame --steps dfs_salaries``.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from ... import config
from ...sources import dfs_capture


def add_commands(sub) -> dict:
    p = sub.add_parser("refresh", help="re-fetch ONE slate's draftables (status, swappable, "
                                       "starter, news); counts against the daily DK budget")
    p.add_argument("--slate", required=True, help="draft group id (see capture-status)")
    p.add_argument("--reason", default="manual",
                   help="label, e.g. lock45 / lock10 (the same reason within 2 min is a no-op)")
    p.add_argument("--force", action="store_true", help="ignore the 2-minute repeat guard")
    p = sub.add_parser("capture-status", help="today's slates, captured payout tables and the "
                                              "DK request budget")
    p.add_argument("--date", default=None, help="NBA day YYYY-MM-DD (default today)")
    return {"refresh": _refresh_cmd, "capture-status": _status_cmd}


def _conn():
    from ... import db
    return db.connect()


def _refresh_cmd(args) -> int:
    conn = _conn()
    res = dfs_capture.refresh_slate(conn, args.slate, reason=args.reason, force=args.force)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] in ("ok", "recent") else 1


def _status_cmd(args) -> int:
    conn = _conn()
    day = date.fromisoformat(args.date) if args.date else config.today_nba_day()
    now = datetime.now(timezone.utc)
    b = dfs_capture.requests_today(conn, now)
    print(f"DK requests {b['day']} (ET): {b['total']} total, {b['added']}/{b['budget']} added "
          f"({b['remaining']} left); by kind {b['by_kind']}")
    slates = conn.execute("SELECT * FROM dfs_slates WHERE nba_day=? ORDER BY lock_utc, slate_id",
                          (day.isoformat(),)).fetchall()
    if not slates:
        print(f"no dfs_slates for {day}")
        return 1
    for s in slates:
        n = conn.execute("SELECT COUNT(*) FROM dfs_contests WHERE slate_id=?",
                         (s["slate_id"],)).fetchone()[0]
        games = len(json.loads(s["game_ids_json"] or "[]"))
        state = "locked" if s["lock_utc"] and s["lock_utc"] <= dfs_capture.iso(now) else "open"
        print(f"  {s['slate_id']}  type {s['game_type_id']:<3} {s['tag']:<8} {games} games  "
              f"lock {s['lock_utc']} ({state})  contests {n}")
    total = conn.execute("SELECT COUNT(*) FROM dfs_contests WHERE slate_id IN "
                         "(SELECT slate_id FROM dfs_slates WHERE nba_day=?)",
                         (day.isoformat(),)).fetchone()[0]
    print(f"contest payout tables stored for {day}: {total}")
    return 0
