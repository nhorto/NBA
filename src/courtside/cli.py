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

    md = sub.add_parser("model", help="prediction model").add_subparsers(
        dest="model_cmd", required=True)
    bt = md.add_parser("backtest", help="walk-forward-by-season backtest")
    bt.add_argument("--models", default="home,elo,elo_rest,elo_sched",
                    help="comma-separated: home, elo, elo_rest, elo_sched")
    bt.add_argument("--from", dest="start", default="2012-13", help="first test season")
    bt.add_argument("--to", dest="end", default=None, help="last test season (default: latest)")
    bt.add_argument("--no-report", action="store_true", help="don't write the JSON report")
    md.add_parser("tune-elo", help="re-run the Elo hyperparameter search (2000-01..2010-11)")
    return p


def _model_cmd(args) -> int:
    from . import config, db
    from .model import backtest, baselines, elo

    frame = backtest.build_frame(db.connect())
    if args.model_cmd == "tune-elo":
        r = elo.tune_all(frame)
        print(f"tuning window {elo.TUNE_SEASONS[0]}..{elo.TUNE_SEASONS[1]}, "
              f"scored from {elo.TUNE_EVAL_FROM}")
        for label, rows in (("stage 1", r["stage1"][:5]), ("stage 2", r["stage2"][:5]),
                            ("rest/travel", r["rest"][:5])):
            print(label)
            for ll, prm in rows:
                print(f"  {ll:.5f}  k={prm.k:g} hca={prm.hca:g} carry={prm.carry:g} "
                      f"b2b={prm.b2b_elo:g} travel={prm.travel_elo_per_1000km:g}")
        print(f"frozen TUNED: {elo.TUNED}\nfrozen TUNED_REST: {elo.TUNED_REST}")
        return 0
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in baselines.MODELS]
    if unknown:
        print(f"error: unknown model(s) {unknown}; choose from {list(baselines.MODELS)}",
              file=sys.stderr)
        return 2
    seasons = [s for s in backtest.seasons_in(frame)
               if s >= args.start and (args.end is None or s <= args.end)]
    if not seasons or seasons[0] <= frame["season"].min():
        print("error: need at least one training season before the first test season",
              file=sys.stderr)
        return 2
    print(f"{len(frame)} games {frame['season'].min()}..{frame['season'].max()}; "
          f"testing {seasons[0]}..{seasons[-1]} with {names}")
    preds, desc = backtest.run_backtest(frame, {n: baselines.MODELS[n] for n in names},
                                        seasons, log=print)
    ev = backtest.evaluate(preds)
    print(backtest.format_tables(ev))
    if not args.no_report:
        path = backtest.write_report(ev, desc, frame, config.data_dir() / "reports",
                                     {"tuned_elo": elo.TUNED.to_dict(),
                                      "tuned_elo_rest": elo.TUNED_REST.to_dict()})
        print(f"\nreport: {path}")
    return 0


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
            summary = collect.run_pregame(args.date or config.today_nba_day())
        else:
            summary = collect.run_postgame(args.date or config.yesterday_nba_day())
        # The job logs its own summary; a failed step must still surface to the scheduler.
        return 0 if summary.get("ok", True) else 1
    if args.cmd == "model":
        return _model_cmd(args)
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
