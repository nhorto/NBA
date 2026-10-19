"""`courtside dfs build`: run the lineup optimizer on a captured slate or a RotoGuru date.

Add ``"courtside.proj.dfs.optimizer_cli"`` to ``DFS_MODULES`` in ``proj/cli.py``.

    courtside dfs build --slate 154575 --mode topn --n 5 --proj fppg
    courtside dfs build --slate 154575 --mode cash --lock "Wembanyama" --ban "Name Here"
    courtside dfs build --date 2019-03-22 --proj actual --mode topn --n 3   # hindsight-optimal
    courtside dfs build --date 2019-03-22 --proj salary                      # 5 FP per $1k

Until the projection pipeline exists, ``--proj fppg`` (DraftKings' own FPPG as the mean, the S13
baseline) is the live source. ``--lock``/``--ban`` take player names (accent- and case-insensitive,
unique substring or all-words match; ``NAME@SLOT`` pins the slot). ``--now`` (default: now) bans
players whose game has started, as late swap does; ``--now none`` turns that off.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from . import optimizer as O
from . import rules as R
from . import topn as T


def add_commands(sub) -> dict:
    p = sub.add_parser("build", help="optimal lineup (cash) or Top N for a slate / RotoGuru date")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--slate", help="draft group id (see `courtside dfs rules`)")
    g.add_argument("--date", help="RotoGuru date YYYY-MM-DD (dfs_hist_salaries)")
    p.add_argument("--mode", choices=("cash", "topn"), default="topn")
    p.add_argument("--n", type=int, default=10, help="lineups for --mode topn (default 10)")
    p.add_argument("--min-diff", type=int, default=1, help="min different players d (default 1)")
    p.add_argument("--proj", default="fppg",
                   choices=("fppg", "model", "actual", "rotoguru-actual", "salary"),
                   help="fppg: DK's FPPG as mu; model: Path A projections (mean + p25, "
                        "courtside proj); actual: real DK points (hindsight, --date); "
                        "salary: 5 points per $1k")
    p.add_argument("--sims", type=int, default=25_000, help="sims for --proj model")
    p.add_argument("--lam", type=float, default=O.DEFAULT_LAMBDA,
                   help="cash fallback lambda (needs p25; ignored while only mu exists)")
    p.add_argument("--lock", nargs="+", action="append", metavar="NAME", default=[])
    p.add_argument("--ban", nargs="+", action="append", metavar="NAME", default=[])
    p.add_argument("--max-exposure", type=float, default=None,
                   help="max fraction of the N lineups any player may be in")
    p.add_argument("--min-salary", type=int, default=None)
    p.add_argument("--time-limit", type=float, default=None, help="seconds per solve")
    p.add_argument("--now", default="now", help="ISO time for late swap, or 'none'")
    p.add_argument("--include-out", action="store_true",
                   help="keep players DK marks Out (excluded by default)")
    p.add_argument("--save", action="store_true", help="write dfs_runs / dfs_lineups")
    return {"build": _build_cmd}


def _conn():
    from ... import db
    return db.connect()


# --------------------------------------------------------------------------- names

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(".", "")).strip()


def resolve_name(pool: R.SlatePool, text: str):
    """The one pool player a name refers to; ValueError when none or several match."""
    q = _norm(text)
    norm = [(_norm(p.name), p) for p in pool.players]
    exact = [p for n, p in norm if n == q]
    cands = exact or [p for n, p in norm if q and q in n] \
        or [p for n, p in norm if q and all(w in n.split() for w in q.split())]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ValueError(f"no player matches {text!r}")
    raise ValueError(f"{text!r} is ambiguous: " + ", ".join(f"{p.name} ({p.team})" for p in cands))


def _resolve_locks(pool, names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for raw in names:
        name, _, slot = raw.partition("@")
        out[resolve_name(pool, name).dk_player_id] = slot.strip().upper() or None
    return out


# --------------------------------------------------------------------------- output

def _money(v: int) -> str:
    return f"${v:,}"


def _print_lineup(rank: int | None, lu: O.Lineup, rules: R.Rules, diff=None, names=None) -> None:
    head = f"#{rank} " if rank else ""
    print(f"{head}proj {lu.projected:.2f}  salary {_money(lu.salary)} "
          f"(left {_money(rules.salary_cap - lu.salary)})"
          + ("" if lu.optimal else "  [time limit: not proven optimal]"))
    for e in lu.entries:
        p = e.player
        print(f"   {e.slot:<4} {p.name:<26} {p.team or '?':<4} {_money(e.salary):>8} "
              f"{e.mean:>7.2f}")
    if diff is not None and diff.n_different and names:
        ins = ", ".join(names.get(i, i) for i in diff.players_in)
        outs = ", ".join(names.get(i, i) for i in diff.players_out)
        print(f"   vs #1: in {ins} | out {outs} | proj {diff.d_projected:+.2f} "
              f"salary {diff.d_salary:+,}")


# --------------------------------------------------------------------------- command

def _build_cmd(args) -> int:
    conn = _conn()
    try:
        return _build(conn, args)
    except (ValueError, LookupError) as e:
        print(f"error: {e}")
        return 2


def _build(conn, args) -> int:
    if args.slate:
        pool = R.build_pool(conn, args.slate)
        if pool.rules is None:
            raise LookupError(f"no rules for slate {args.slate} (no rules_json, no archive)")
        if args.proj in ("actual", "rotoguru-actual"):
            raise ValueError("--proj actual needs a RotoGuru --date (hindsight)")
        if args.proj == "model":
            from ..pipeline import project_slate
            at = (datetime.now(timezone.utc) if args.now in ("now", "none")
                  else datetime.fromisoformat(args.now.replace("Z", "+00:00")))
            projs = project_slate(conn, pool.slate_id, at, run_id=f"build-{uuid.uuid4().hex[:10]}",
                                  n_sims=args.sims, kind="manual", store=args.save)
        elif args.proj == "fppg":
            projs = {p.dk_player_id: O.Projection(p.fppg) for p in pool.players
                     if p.fppg is not None}
        else:
            projs = {p.dk_player_id: O.Projection(round(O.SALARY_FP_PER_K * p.salary / 1000, 2))
                     for p in pool.players}
        label = f"slate {pool.slate_id} ({pool.game_type}) captured {pool.captured_at}"
    else:
        pool = O.hist_pool(conn, args.date, O.classic_rules())
        if args.proj in ("fppg", "model"):
            raise ValueError(f"--proj {args.proj} needs a captured --slate; use actual or "
                             "salary with --date")
        projs = O.hist_projections(conn, pool, args.proj)
        label = f"RotoGuru {args.date} (classic){'  HINDSIGHT-OPTIMAL' if 'actual' in args.proj else ''}"
    rules = pool.rules
    bans = _resolve_locks(pool, [n for grp in args.ban for n in grp])
    locks = _resolve_locks(pool, [n for grp in args.lock for n in grp])
    bans = set(bans)
    if not args.include_out:
        bans |= {p.dk_player_id for p in pool.players
                 if p.status == "O" and p.dk_player_id not in locks}
    now = None
    if args.now != "none":
        now = (datetime.now(timezone.utc) if args.now == "now"
               else datetime.fromisoformat(args.now.replace("Z", "+00:00")))
    missing = sum(1 for p in pool.players if p.dk_player_id not in projs)
    opts = dict(locks=locks, bans=bans, now=now, min_salary=args.min_salary,
                time_limit_s=args.time_limit)
    names = {p.dk_player_id: p.name for p in pool.players}
    print(f"{label}; proj {args.proj}; {len(pool.players)} players "
          f"({missing} without a projection), {len(pool.games)} games; "
          f"cap {_money(rules.salary_cap)}, min games {rules.min_games}, "
          f"min teams {rules.min_teams}")
    t0 = datetime.now()
    if args.mode == "cash":
        lu = O.optimize(pool, projs, rules, mode="cash", lam=args.lam, **opts)
        ranked = [] if lu is None else [T.RankedLineup(1, lu, T.diff_vs(lu, lu))]
        stopped = None if lu else "infeasible"
    else:
        res = T.top_n(pool, projs, rules, args.n, min_diff=args.min_diff, mode="mean",
                      max_exposure=args.max_exposure, **opts)
        ranked, stopped = list(res.lineups), res.stopped
    secs = (datetime.now() - t0).total_seconds()
    if not ranked:
        print("no legal lineup (locks/bans/rules leave none)")
        return 1
    for r in ranked:
        _print_lineup(r.rank if args.mode == "topn" else None, r.lineup, rules, r.diff, names)
    if args.mode == "topn" and len(ranked) < args.n:
        print(f"only {len(ranked)} of {args.n} lineups exist ({stopped})")
    print(f"solved in {secs:.2f}s")
    if args.save:
        run = O.save_run(conn, slate_id=pool.slate_id, kind=args.mode, seconds=secs,
                         detail={"proj": args.proj, "n": args.n, "min_diff": args.min_diff,
                                 "lam": args.lam, "locks": locks, "bans": sorted(bans),
                                 "max_exposure": args.max_exposure})
        O.save_lineups(conn, run, args.mode, ranked)
        print(f"saved run {run}")
    return 0
