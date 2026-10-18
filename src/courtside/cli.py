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
                    help="comma-separated: home, elo, elo_rest, elo_sched, the player models "
                         "(ps, ps_elo_sched, ...; see model/player_model.SPECS) and, when "
                         "closing_lines exists, mkt, elo_sched_mkt, ps_elo_sched_mkt")
    bt.add_argument("--from", dest="start", default="2012-13", help="first test season")
    bt.add_argument("--to", dest="end", default=None, help="last test season (default: latest)")
    bt.add_argument("--no-report", action="store_true", help="don't write the JSON report")
    bt.add_argument("--dump-preds", default=None, help="write per-game predictions (csv.gz)")
    md.add_parser("tune-elo", help="re-run the Elo hyperparameter search (2000-01..2010-11)")
    tp = md.add_parser("tune-players",
                       help="player-rating hyperparameter search (2000-01..2010-11 only)")
    tp.add_argument("--workers", type=int, default=4)
    pr = md.add_parser("player-ratings", help="player ratings after the last game in the data")
    pr.add_argument("--top", type=int, default=15)
    pr.add_argument("--min-minutes", type=float, default=1000.0,
                    help="minimum minutes in the last season to be listed")
    mk = md.add_parser("market", help="compare dumped predictions with closing lines")
    mk.add_argument("preds", help="csv.gz written by backtest --dump-preds")
    mk.add_argument("--models", default=None)

    pd_ = sub.add_parser("predict", help="predictions for a day (fit on games before it)")
    pd_.add_argument("--date", type=_parse_day, default=None, help="NBA day; default today")
    pd_.add_argument("--model", default=None, help="model name (default: PRODUCTION_MODEL)")
    pd_.add_argument("--no-store", action="store_true", help="print only; don't write predictions")
    pd_.add_argument("--backfill-from", type=_parse_day, default=None, metavar="YYYY-MM-DD",
                     help="walk-forward predictions for every game day in [from, --to]")
    pd_.add_argument("--to", type=_parse_day, default=None,
                     help="last day of --backfill-from (default: --date or today)")

    br = sub.add_parser("brief", help="the daily Brief").add_subparsers(
        dest="brief_cmd", required=True)
    bb = br.add_parser("build", help="build and store the Brief for a day")
    bb.add_argument("--date", type=_parse_day, default=None, help="NBA day; default today")
    bb.add_argument("--no-writer", action="store_true", help="leave the text slots null")
    bs = br.add_parser("show", help="print a stored Brief as text")
    bs.add_argument("--date", type=_parse_day, default=None, help="NBA day; default today")
    bs.add_argument("--json", action="store_true", help="print the raw JSON instead")
    return p


def _model_cmd(args) -> int:
    from . import config, db
    from .model import backtest, baselines, elo

    if args.model_cmd in ("tune-players", "player-ratings", "market"):
        return _player_cmd(args)
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
    from .model import market, player_model, players
    conn = db.connect()
    has_mkt = market.has_closing_lines(conn)
    if has_mkt:
        frame = market.add_market_columns(frame, conn)
    preg, pcache = player_model.make_registry(lambda: players.load_player_games(conn),
                                              market=has_mkt)
    registry = {**baselines.MODELS, **preg}
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        print(f"error: unknown model(s) {unknown}; choose from {list(registry)}",
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
    import time
    t0 = time.time()
    if any(n in preg for n in names):
        # one causal pass over all games; every fold reads an exact prefix of it
        pcache.prime(frame[frame["season"] <= seasons[-1]])
        print(f"player engine pass: {time.time() - t0:.1f}s")
    preds, desc = backtest.run_backtest(frame, {n: registry[n] for n in names},
                                        seasons, log=print)
    print(f"backtest wall time: {time.time() - t0:.1f}s")
    ev = backtest.evaluate(preds)
    print(backtest.format_tables(ev))
    extra = {"tuned_elo": elo.TUNED.to_dict(), "tuned_elo_rest": elo.TUNED_REST.to_dict(),
             "player_params": players.TUNED.to_dict(), "wall_time_s": time.time() - t0}
    if has_mkt:
        cmp_ = market.compare(preds, market.load_closing(conn))
        print("\nRegular Season vs closing line (games with a de-vigged moneyline)")
        print(market.format_compare(cmp_))
        extra["market"] = cmp_
    if args.dump_preds:
        preds.to_csv(args.dump_preds, index=False)
        print(f"predictions: {args.dump_preds}")
    if not args.no_report:
        path = backtest.write_report(ev, desc, frame, config.data_dir() / "reports", extra)
        print(f"\nreport: {path}")
    return 0


def _player_cmd(args) -> int:
    from . import config, db
    from .model import backtest, market, player_model, players

    if args.model_cmd == "tune-players":
        import json
        from datetime import datetime, timezone
        res = player_model.tune_all(str(config.db_path()), workers=args.workers)
        print(f"best: {res['best']}")
        out = config.data_dir() / "reports" / (
            "tune_players_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        print(f"report: {out}")
        return 0
    conn = db.connect()
    if args.model_cmd == "market":
        import pandas as pd
        preds = pd.read_csv(args.preds, dtype={"game_id": str})
        models = args.models.split(",") if args.models else None
        print(market.format_compare(market.compare(preds, market.load_closing(conn), models)))
        return 0
    import time
    t0 = time.time()
    frame = backtest.build_frame(conn)
    _, st = players.run_player_engine(frame, players.load_player_games(conn), players.TUNED,
                                      modes=("played",), return_state=True)
    tab = st.ratings_table(args.min_minutes)
    print(f"ratings as of {st.as_of:%Y-%m-%d} (pts/48 vs average; >= {args.min_minutes:g} "
          f"min in {frame['season'].max()}), pass {time.time() - t0:.0f}s")
    cols = ["name", "team", "rating", "prior", "season_min"]
    print("top\n" + tab.head(args.top)[cols].to_string(float_format="%.2f"))
    print("bottom\n" + tab.tail(args.top)[cols].to_string(float_format="%.2f"))
    print(f"new-player prior {st.mu_new:.2f}, APM home court {st.hca:.2f}")
    print("box prior beta: " + ", ".join(f"{k}={v:.3f}" for k, v in st.beta.items()))
    print("minutes projection MAE by season: " + ", ".join(
        f"{s}={v['mae']:.2f}" for s, v in st.minutes_error.items()))
    return 0


def _predict_cmd(args) -> int:
    from . import config, db, predict
    conn = db.connect()
    if args.backfill_from:
        end = args.to or args.date or config.today_nba_day()
        res = predict.predict_days(conn, args.backfill_from, end, model=args.model,
                                   log_fn=lambda m: print(m, flush=True))
        print(f"{res['model']}: {res['games']} predictions over {res['days']} days "
              f"({args.backfill_from}..{end}); skipped days: {res['skipped_days']}")
        return 0
    day = args.date or config.today_nba_day()
    rows = predict.predict_day(conn, day, model=args.model, store=not args.no_store)
    if not rows:
        print(f"no predictions for {day} (no scheduled games, or not enough history)")
        return 0
    for r in rows:
        mk = (f"  mkt {r['home']} {100 * r['market_home_prob']:.0f}% "
              f"spread {r['market_home_spread']:+.1f}" if r["market_home_prob"] is not None
              and r["market_home_spread"] is not None else "")
        out = sum(len(v) for v in r["injuries"].values())
        print(f"{r['away']} @ {r['home']}  {r['home']} {100 * r['home_win_prob']:.0f}%  "
              f"margin {r['home_margin']:+.1f}  total {r['total']:.1f}{mk}"
              f"{f'  [{out} injured/questionable]' if out else ''}")
    return 0


def _brief_cmd(args) -> int:
    import json

    from . import config, db
    from .brief import assemble, show
    conn = db.connect()
    day = args.date or config.today_nba_day()
    if args.brief_cmd == "build":
        b = assemble.build_brief(conn, day, run_writer=not args.no_writer)
        m = b["meta"]
        print(f"brief {m['day']} built -> {config.data_dir() / m['path']}  (schema {m['schema_version']}, "
              f"{b['last_night']['summary']['games']} last-night games, "
              f"{b['tonight']['summary']['games']} tonight)")
        for n in m["notes"]:
            print(f"  note: {n}")
        return 0
    b = assemble.load_brief(conn, day)
    if b is None:
        print(f"no stored Brief for {day}; building (without writer)", file=sys.stderr)
        b = assemble.build_brief(conn, day, run_writer=False)
    print(json.dumps(b, indent=1, ensure_ascii=False) if args.json else show.format_brief(b))
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
    if args.cmd == "predict":
        return _predict_cmd(args)
    if args.cmd == "brief":
        return _brief_cmd(args)
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
