"""Bit-identity check for switches that must not change anything when off (P(play) v2).

Runs a fixed set of projection cases for the arm given by the environment (COURTSIDE_CONTEXT /
_CONTEXT_EXT / _COPULA / _PPLAY) and prints one sha1 per case over every sim array
(player ids, DK points, minutes) and the players summary frame. Run it under the old code
tree and the new one (same DB copy, same bundles) and diff the outputs.

    python research/p2_identity.py CASE [CASE ...]
    cases: live:<day> live10:<day> avail:<day> reports:<day> opener:<day>
           availpair:<season>:<n>   (avail protocol, P(play) v2 off vs on, n dates)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta

import numpy as np

from courtside import db
from courtside.proj import backtest as B
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

M.GBM_PARAMS["num_threads"] = 1


def digest(res) -> str:
    h = hashlib.sha1()
    for g in res.sims:
        for a in (g.player_id, g.fp, g.minutes):
            h.update(np.ascontiguousarray(a).tobytes())
        if g.stats is not None:
            for s in sorted(g.stats):
                h.update(np.ascontiguousarray(g.stats[s]).tobytes())
    pl = res.players
    num = pl.select_dtypes(include=[np.number]).sort_index(axis=1)
    h.update(num.to_numpy(float).tobytes())
    return h.hexdigest()[:16]


def opener(conn, ctx, b, day: date):
    """The live path of every DK slate's games at lock - 45 (as research/p2_opener.py)."""
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import p2_opener as O
    out = []
    for sid, gids, lock in O.slates(conn, day.isoformat()):
        at = lock - timedelta(minutes=45)
        extras, starters, salaries, _, _ = O.pool_inputs(conn, sid, at, gids)
        kw = {"salaries": salaries} if getattr(P, "PPLAY", "v1") == "v2" else {}
        r = P.project_day(conn, day, at, n_sims=2000, kind="lock45", game_ids=gids, ctx=ctx,
                          bundle=b, extras=extras, starters=starters or None,
                          seed=P.stable_seed(sid, "lock45", P.SEED_VERSION), **kw)
        out.append(digest(r))
    return hashlib.sha1("".join(out).encode()).hexdigest()[:16]


def avail_pair(conn, ctxs, season: str, n: str) -> dict:
    """The avail protocol (the T2.1 harness's oracle) with P(play) v2 off and on, on `n`
    evenly spaced regular-season dates of `season`: number of dates with identical sims."""
    from courtside.proj import evallite as E
    days = [d.isoformat() for d in E.usable_dates(conn, season)[0]]
    days = E._pick(days, int(n))
    if season not in ctxs:
        ctx = P.Context(conn, season)
        ctxs[season] = (ctx, P.load_bundle(conn, season, ctx))
    ctx, b = ctxs[season]
    was = P.PPLAY
    same = 0
    for ds in days:
        dg = {}
        for v in ("v1", "v2"):
            P.set_pplay(v)
            seed = P.stable_seed(ds, "hist-avail", P.SEED_VERSION)
            dg[v] = digest(P.project_historical(conn, ds, protocol="avail", n_sims=2000,
                                                ctx=ctx, bundle=b, seed=seed))
        same += dg["v1"] == dg["v2"]
    P.set_pplay(was)
    return {"dates": len(days), "identical": same}


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    conn = db.connect()
    ctxs: dict = {}
    res = {"model_version": P.MODEL_VERSION}
    for case in argv:
        if case.startswith("availpair:"):
            res[case] = avail_pair(conn, ctxs, *case.split(":")[1:])
            print(case, res[case], flush=True)
            continue
        kind, ds = case.split(":")
        day = date.fromisoformat(ds)
        season = P.season_of(day)
        if season not in ctxs:
            ctx = P.Context(conn, season)
            ctxs[season] = (ctx, P.load_bundle(conn, season, ctx))
        ctx, b = ctxs[season]
        if kind in ("live", "live10"):
            live = B.LiveInputs(conn, season, "loo_month", log=lambda m: None)
            off = B.LIVE_OFFSETS[kind]
            r = B.live_like_day(conn, ctx, b, live, ds, offset_min=off, n_sims=2000,
                                use_starters=off <= 10)
            res[case] = digest(r)
        elif kind in ("avail", "reports"):
            r = P.project_historical(conn, ds, protocol=kind, n_sims=2000, ctx=ctx, bundle=b)
            res[case] = digest(r)
        elif kind == "availpair":                 # availpair:<first day of season>:<n dates>
            pass
        elif kind == "opener":
            res[case] = opener(conn, ctx, b, day)
        else:
            raise ValueError(case)
        print(case, res[case], flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
