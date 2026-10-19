"""`courtside mlops monitor | checks | drift | alerts | status` (monitoring half of MLOps)."""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta


def _date(s: str) -> date:
    return date.fromisoformat(s)


def add_commands(sub) -> dict:
    p = sub.add_parser("monitor", help="compute metric snapshots (model + market, 5 windows)")
    p.add_argument("--as-of", type=_date, help="score games before this day (default: today)")
    p.add_argument("--replay-from", type=_date, help="backfill every day from here ...")
    p.add_argument("--to", type=_date, help="... to here (inclusive; default: today)")

    p = sub.add_parser("checks", help="run the data-quality / pipeline-health checks")
    p.add_argument("--job", choices=("pregame", "postgame"),
                   help="job to check (default: both)")
    p.add_argument("--date", type=_date, help="NBA day (default: today / yesterday per job)")

    p = sub.add_parser("drift", help="PSI of live features vs the reference")
    p.add_argument("--as-of", type=_date)

    p = sub.add_parser("alerts", help="list alerts (open by default)")
    p.add_argument("--all", action="store_true", help="include resolved alerts")
    p.add_argument("--evaluate", action="store_true", help="re-evaluate the rules first")
    p.add_argument("--as-of", type=_date)

    p = sub.add_parser("status", help="compact model-health dashboard")
    p.add_argument("--as-of", type=_date)
    return {"monitor": _monitor, "checks": _checks, "drift": _drift, "alerts": _alerts,
            "status": _status}


def _today() -> date:
    from .. import config
    return config.today_nba_day()


def _conn():
    from .. import db
    return db.connect()


def _monitor(args) -> int:
    from . import monitor
    conn = _conn()
    if args.replay_from:
        t0 = time.time()
        r = monitor.replay(conn, args.replay_from, args.to or _today())
        print(f"replayed {r['days']} days ({r['start']} -> {r['end']}): {r['rows']} snapshot "
              f"rows in {time.time() - t0:.1f}s; games scored {r['games']}")
        return 0
    as_of = args.as_of or _today()
    rows = monitor.compute_metrics(conn, as_of)
    print(f"{len(rows)} snapshot rows for {as_of}")
    for r in rows:
        if r["window"] in ("season", "last100"):
            print(_metric_line(r))
    return 0


def _checks(args) -> int:
    from . import checks
    conn = _conn()
    jobs = [args.job] if args.job else ["pregame", "postgame"]
    bad = 0
    for job in jobs:
        day = args.date or (_today() if job == "pregame" else _today() - timedelta(days=1))
        print(f"== {job} checks for {day}")
        for r in checks.run_checks(conn, job, day):
            print(_check_line(r))
            bad += r["status"] == "fail"
    return 1 if bad else 0


def _drift(args) -> int:
    from . import drift
    conn = _conn()
    as_of = args.as_of or _today()
    rows = drift.compute_drift(conn, as_of)
    if not rows:
        print(f"no drift computed for {as_of}: fewer than {drift.MIN_N} live predictions in "
              f"the last {drift.WINDOW_DAYS} days, or no reference")
        return 0
    print(f"PSI as of {as_of}  (reference: {rows[0]['ref_label']}; warn >0.25, critical >0.5)")
    for r in sorted(rows, key=lambda r: -r["psi"]):
        print(f"  {r['feature']:<22} {r['psi']:>7.3f}  n={r['cur_n']}")
    return 0


def _alerts(args) -> int:
    from . import alerts
    conn = _conn()
    if args.evaluate:
        alerts.evaluate(conn, args.as_of or _today())
    rows = alerts.list_alerts(conn, include_resolved=args.all)
    if not rows:
        print("no open alerts" if not args.all else "no alerts")
        return 0
    for a in rows:
        tail = f" (resolved {a['resolved_at']})" if a["status"] == "resolved" else ""
        print(f"[{a['severity']:<8}] {a['key']}: {a['message']}  since {a['created_at']}{tail}")
    return 0


# ------------------------------------------------------------------ formatting

def _f(v, fmt="{:.4f}", dash="   -  "):
    return dash if v is None else fmt.format(v)


def _metric_line(r: dict) -> str:
    vs = "" if r["ll_diff_vs_mkt"] is None else \
        f"  vs mkt {r['ll_diff_vs_mkt']:+.4f}±{r['ll_diff_se'] or 0:.4f}"
    return (f"  {r['model']:<22}{r['window']:<8} n={r['n']:<4} ll={_f(r['log_loss'])}"
            f" brier={_f(r['brier'])} acc={_f(r['accuracy'], '{:.1%}')} ece={_f(r['ece'], '{:.3f}')}"
            f"{vs}")


def _check_line(r: dict) -> str:
    return f"  [{r['status']:<4}] {r['check_name']:<28} {r['detail']}"


def _status(args) -> int:
    from . import alerts, checks, drift, monitor
    conn = _conn()
    as_of = args.as_of or _today()
    snaps = monitor.snapshots(conn, as_of)
    day = snaps[0]["as_of"] if snaps else None
    print(f"Courtside model health - as of {as_of}"
          + (f" (metrics snapshot {day})" if day and day != as_of.isoformat() else ""))
    by = {(r["model"], r["window"]): r for r in snaps}
    models = sorted({r["model"] for r in snaps}, key=lambda m: (m == monitor.MARKET, m))
    if not models:
        print("\nno metric snapshots yet: run `courtside mlops monitor --as-of D` "
              "(or --replay-from D --to D)")
    else:
        print("\nPerformance (log-loss lower is better; vs mkt = paired log-loss minus market "
              "on the same games, negative beats it;\nslope <1 = overconfident, >1 = "
              "underconfident; calz = Spiegelhalter z, |z|>2.5 is significant, >0 overconfident)")
        hdr = (f"{'model':<24}{'window':<8}{'n':>4}  {'logloss':>7} {'brier':>6} {'acc':>6} "
               f"{'ece':>5} {'slope':>5} {'calz':>5}  {'vs mkt (+-se)':>16}  {'home p/act':>11} {'mgn bias':>8} "
               f"{'tot bias':>8}")
        print(hdr)
        for m in models:
            for w in ("season", "last100", "last300"):
                r = by.get((m, w))
                if r is None:
                    continue
                vs = "-" if r["ll_diff_vs_mkt"] is None else \
                    f"{r['ll_diff_vs_mkt']:+.4f}+-{r['ll_diff_se'] or 0:.4f}"
                hm = "-" if r["home_pred"] is None else \
                    f"{r['home_pred']:.1%}/{r['home_actual']:.1%}"
                d = r["details"]
                print(f"{m:<24}{w:<8}{r['n']:>4}  {_f(r['log_loss'], '{:.4f}'):>7} "
                      f"{_f(r['brier'], '{:.4f}'):>6} {_f(r['accuracy'], '{:.1%}'):>6} "
                      f"{_f(r['ece'], '{:.3f}'):>5} {_f(d.get('cal_slope'), '{:.2f}'):>5} "
                      f"{_f(d.get('spiegelhalter_z'), '{:+.1f}'):>5}  {vs:>16}  {hm:>11} "
                      f"{_f(r['margin_bias'], '{:+.2f}'):>8} {_f(r['total_bias'], '{:+.2f}'):>8}")
    print("\nOpen alerts")
    al = alerts.list_alerts(conn)
    if not al:
        print("  none")
    for a in al:
        print(f"  [{a['severity']}] {a['message']}")
    dr = drift.latest(conn, as_of)
    if dr:
        top = ", ".join(f"{r['feature']} {r['psi']:.2f}" for r in dr[:4])
        print(f"\nDrift (PSI vs {dr[0]['ref_label']}, as of {dr[0]['as_of']}): {top}")
    ck = checks.latest(conn)
    print("\nLatest data checks")
    if not ck:
        print("  none recorded")
    for r in ck:
        print(f"  [{r['status']:<4}] {r['check_name']:<28} {r['detail']}  ({r['run_at']})")
    return 0
