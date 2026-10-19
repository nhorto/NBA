"""`courtside mlops` registry / retraining commands (routed by mlops/cli.py).

    bootstrap                      register the production specs as champion v1 (idempotent)
    versions [--model M]           list registered versions
    promote VERSION --reason R     make VERSION its model's champion (old one retired)
    rollback MODEL --reason R      restore MODEL's most recently retired champion
    retrain [--model M|all] [--params JSON] [--search small|medium]
                                   challenger + offline gate; pass -> shadow, fail -> rejected
    shadows [--dry-run]            online gate for shadow versions (promote / reject / wait)
    card [--out PATH]              regenerate docs/MODEL_CARD.md from the registry
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

PRODUCTION = ("gbm_market", "gbm_opinion")


def _conn():
    from .. import db
    return db.connect()


def _bootstrap(args) -> int:
    from . import registry
    conn = _conn()
    out = registry.bootstrap(conn, log_mlflow=not args.no_mlflow)
    print(json.dumps(out, indent=1))
    return 0


def _fmt_ll(v: dict) -> str:
    b = v.get("benchmark") or {}
    s = b.get("summary", b)
    ll = (s.get("all") or {}).get("log_loss")
    return f"{ll:.4f}" if isinstance(ll, (int, float)) else "-"


def _versions(args) -> int:
    from . import registry
    conn = _conn()
    rows = registry.list_versions(conn, args.model)
    if not rows:
        print("no registered versions (run `courtside mlops bootstrap`)")
        return 0
    print(f"{'version':<34}{'status':<11}{'created':<12}{'promoted':<12}{'retired':<12}"
          f"{'bench LL':>9}  code / notes")
    for v in rows:
        print(f"{v['version_id']:<34}{v['status']:<11}{(v['created_at'] or '')[:10]:<12}"
              f"{(v['promoted_at'] or '')[:10]:<12}{(v['retired_at'] or '')[:10]:<12}"
              f"{_fmt_ll(v):>9}  {v['code_sha']}  {v['notes'] or ''}")
    if args.events:
        for e in registry.events(conn, model_name=args.model, limit=args.events):
            print(f"  {e['ts'][:19]}  {e['action']:<9}{e['version_id']:<34}{e['reason'] or ''}")
    return 0


def _promote(args) -> int:
    from . import registry
    conn = _conn()
    prev = registry.promote(conn, args.version, args.reason)
    print(f"{args.version} is champion" + (f" (retired {prev})" if prev else ""))
    return 0


def _rollback(args) -> int:
    from . import registry
    conn = _conn()
    vid = registry.rollback(conn, args.model, args.reason)
    print(f"{args.model}: rolled back to {vid}")
    return 0


def _params_arg(raw: str | None) -> dict | None:
    if not raw:
        return None
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    return json.loads(raw)


def _retrain(args) -> int:
    from . import retrain
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = _conn()
    names = PRODUCTION if args.model == "all" else (args.model,)
    params = _params_arg(args.params)
    for n in names:
        res = retrain.run_challenger(conn, n, search=args.search, params=params,
                                     seasons_from=args.seasons_from, seasons_to=args.seasons_to,
                                     threads=args.threads, register_result=not args.no_register)
        g = res["gate"]
        print(json.dumps({k: res.get(k) for k in ("model_name", "version_id", "status", "mode",
                                                  "champion", "timing", "mlflow_run_id")},
                         indent=1))
        print(json.dumps({k: g.get(k) for k in ("passed", "reasons", "n", "overall", "recent",
                                                "ece", "log_loss")}, indent=1, default=str))
        print("challenger params:", json.dumps(res["params"]["fixed"]))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(res, indent=1, default=str))
    return 0


def _shadows(args) -> int:
    from . import retrain
    conn = _conn()
    out = retrain.evaluate_shadows(conn, dry_run=args.dry_run, min_games=args.min_games,
                                   max_games=args.max_games)
    if not out:
        print("no shadow versions")
    return 0


def _card(args) -> int:
    from . import modelcard
    conn = _conn()
    path = modelcard.write(conn, args.out)
    print(f"wrote {path}")
    return 0


def add_commands(sub) -> dict:
    p = sub.add_parser("bootstrap", help="register the production specs as champion v1")
    p.add_argument("--no-mlflow", action="store_true", help="don't log benchmarks to MLflow")

    p = sub.add_parser("versions", help="list registered model versions")
    p.add_argument("--model", default=None, help="gbm_market | gbm_opinion")
    p.add_argument("--events", type=int, default=0, metavar="N",
                   help="also show the last N registry events")

    p = sub.add_parser("promote", help="make a version its model's champion")
    p.add_argument("version")
    p.add_argument("--reason", required=True)

    p = sub.add_parser("rollback", help="restore a model's previous champion")
    p.add_argument("model")
    p.add_argument("--reason", required=True)

    p = sub.add_parser("retrain", help="challenger run + offline gate (heavy: run on the Mac)")
    p.add_argument("--model", default="all", choices=(*PRODUCTION, "all"))
    p.add_argument("--params", default=None,
                   help="explicit params JSON (or @file), e.g. "
                        "'{\"margin\": [{\"num_leaves\": 31}, 60]}' or '{\"search_grid\": [...]}'")
    p.add_argument("--search", default="small", choices=("small", "medium"))
    p.add_argument("--seasons-from", default="2012-13")
    p.add_argument("--seasons-to", default=None, help="default: last completed season")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-register", action="store_true", help="evaluate only")
    p.add_argument("--json-out", default=None, help="write the full result as JSON")

    p = sub.add_parser("shadows", help="online gate: promote / reject / keep shadow versions")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-games", type=int, default=100)
    p.add_argument("--max-games", type=int, default=300)

    p = sub.add_parser("card", help="regenerate docs/MODEL_CARD.md")
    p.add_argument("--out", default=None)

    return {"bootstrap": _bootstrap, "versions": _versions, "promote": _promote,
            "rollback": _rollback, "retrain": _retrain, "shadows": _shadows, "card": _card}
