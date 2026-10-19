"""`courtside dfs run` and `courtside dfs tick`: the lock-relative DFS runs (T1.6).

Add ``"courtside.proj.dfs.runner_cli"`` to ``DFS_MODULES`` in ``proj/cli.py``.

    courtside dfs run --slate 154575 --kind lock45          # idempotent on (slate, kind)
    courtside dfs run --slate 154575 --kind lock10 --force  # run again
    courtside dfs run --slate 154575 --kind swap            # next later game; only if late news
    courtside dfs tick                                      # what the scheduler runs every 5 min
    courtside dfs tick --now 2026-10-20T18:50:00Z --dry-run

``tick`` prints one JSON line per slate whose window it hit (nothing at all otherwise, so the
scheduler's log stays empty on the 150 idle ticks a day). Exit code 1 when a run failed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def add_commands(sub) -> dict:
    p = sub.add_parser("run", help="one lock-relative run of a slate: refresh, project, cash + "
                                   "Top N, late news (idempotent on slate + kind)")
    p.add_argument("--slate", required=True, help="draft group id (see capture-status)")
    p.add_argument("--kind", required=True, choices=("lock45", "lock10", "swap", "manual"))
    p.add_argument("--force", action="store_true", help="run again although an ok run exists")
    p.add_argument("--n", type=int, default=10, help="Top N size (default 10)")
    p.add_argument("--min-diff", type=int, default=1, help="min different players d (default 1)")
    p.add_argument("--now", default=None, help="ISO time to run as of (default: now)")
    p.add_argument("--tip", default=None, help="swap: the later game's start (default: the next)")
    p.add_argument("--all", action="store_true", dest="resolve_all",
                   help="swap: re-solve every saved lineup, not only those with late news")
    p.add_argument("--no-refresh", action="store_true", help="do not fetch anything first")
    p = sub.add_parser("tick", help="fire the lock45 / lock10 / swap runs whose window contains "
                                    "now (no request unless a window is hit)")
    p.add_argument("--now", default=None, help="ISO time (default: now)")
    p.add_argument("--dry-run", action="store_true", help="show what would run; touch nothing")
    p.add_argument("--no-refresh", action="store_true", help="do not fetch anything first")
    return {"run": _run_cmd, "tick": _tick_cmd}


def _conn():
    from ... import db
    return db.connect()


def _when(text: str | None) -> datetime | None:
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _run_cmd(args) -> int:
    from . import runner
    conn = _conn()
    res = runner.run_slate(conn, args.slate, args.kind, now=_when(args.now), n=args.n,
                           min_diff=args.min_diff, force=args.force,
                           refresh=not args.no_refresh, tip=args.tip,
                           resolve_all=args.resolve_all)
    print(json.dumps(res, indent=2, default=str))
    return 1 if res["status"] in ("error", "unknown_slate") else 0


def _tick_cmd(args) -> int:
    from . import runner
    conn = _conn()
    results = runner.tick(conn, _when(args.now), dry_run=args.dry_run,
                          refresh=not args.no_refresh)
    bad = False
    for r in results:
        if r["action"] == "already_done":
            continue
        print(json.dumps(r, default=str))
        bad = bad or (r["action"] == "ran" and r["result"]["status"] == "error")
    return 1 if bad else 0
