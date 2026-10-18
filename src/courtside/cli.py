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

    bf = sub.add_parser("backfill", help="historical backfills").add_subparsers(
        dest="what", required=True)
    gl = bf.add_parser("gamelogs", help="team/player game logs from stats.nba.com")
    gl.add_argument("--from", dest="start", default="2010-11", help="first season, e.g. 2010-11")
    gl.add_argument("--to", dest="end", default="2025-26", help="last season, e.g. 2025-26")
    gl.add_argument("--no-advanced", action="store_true", help="skip advanced-stat endpoints")
    gl.add_argument("--force", action="store_true", help="re-fetch even if already archived")
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
    if args.cmd == "backfill":
        from . import backfill
        try:
            seasons = backfill.seasons_range(args.start, args.end)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        summary = backfill.backfill_gamelogs(
            seasons, include_advanced=not args.no_advanced, force=args.force)
        print(f"fetched={summary['fetched']} cached={summary['cached']} "
              f"failed={summary['failed']} rows={summary['rows']}")
        for err in summary["errors"]:
            print(f"  error: {err}", file=sys.stderr)
        return 1 if summary["failed"] else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
