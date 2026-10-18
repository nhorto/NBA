"""courtside command line."""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _parse_day(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {s!r}, expected YYYY-MM-DD")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="courtside")
    sub = p.add_subparsers(dest="cmd", required=True)

    db = sub.add_parser("db", help="database commands").add_subparsers(dest="db_cmd", required=True)
    db.add_parser("migrate", help="create/upgrade the SQLite database")

    col = sub.add_parser("collect", help="run collection jobs").add_subparsers(
        dest="job", required=True)
    for job in ("pregame", "postgame"):
        sp = col.add_parser(job)
        sp.add_argument("--date", type=_parse_day, default=None,
                        help="NBA day (YYYY-MM-DD); default today (pregame) / yesterday (postgame)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "db":
        from . import config, db
        conn = db.connect()
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        print(f"{config.db_path()} at schema version {version}")
        return 0
    if args.cmd == "collect":
        from . import collect, config
        if args.job == "pregame":
            collect.run_pregame(args.date or config.today_nba_day())
        else:
            collect.run_postgame(args.date or config.yesterday_nba_day())
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
