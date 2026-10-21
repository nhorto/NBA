"""P(play) v2 on real live slates: re-project a day's DK slates on the live path, off vs p2.

For every single-game (Showdown) slate of the day, else the Classic slates, at the slate's
lock - OFFSET: the DK pool of the latest capture at or before that time (forced extras, DK
starter flags, salaries), the official report of the time, `pipeline.project_day` with the
production bundle of the season, once with COURTSIDE_PPLAY off (v1) and once with p2 (same
seeds). Writes per-player rows (projection both ways, fringe features, salary, actual minutes)
and prints team minute sums, the fringe players and the top projections.

    python research/p2_opener.py 2026-10-20 [--offset 45] [--sims 5000] [--out DIR]
    python research/p2_opener.py lineups 2026-10-20     # cash + Top-10 off vs p2, actual pts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import availability as AV
from courtside.proj import ids as IDS
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj.dfs import rules as R

M.GBM_PARAMS["num_threads"] = 1


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def slates(conn, day: str) -> list[tuple[str, list[str], datetime]]:
    rows = conn.execute("SELECT slate_id, game_type_id, game_ids_json, lock_utc FROM dfs_slates "
                        "WHERE nba_day=? ORDER BY lock_utc", (day,)).fetchall()
    sd = [(str(r[0]), json.loads(r[2]), _utc(r[3])) for r in rows if r[1] == 81]
    covered = {g for _, gs, _ in sd for g in gs}
    cl = [(str(r[0]), json.loads(r[2]), _utc(r[3])) for r in rows if r[1] != 81]
    for sid, gs, lock in cl:                       # games without a Showdown slate
        rest = [g for g in gs if g not in covered]
        if rest:
            sd.append((sid, rest, lock))
            covered |= set(rest)
    return sd


def pool_inputs(conn, sid: str, at: datetime, game_ids: list[str]):
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (sid, P._iso(at))).fetchone()[0]
    if cap is None:
        cap = conn.execute("SELECT MIN(captured_at) FROM dfs_salaries WHERE slate_id=?",
                           (sid,)).fetchone()[0]
    pool = R.build_pool(conn, sid, captured_at=cap)
    dk2nba = IDS.nba_ids(conn, "dk", [p.dk_player_id for p in pool.players])
    extras, starters, salaries, names = {}, set(), {}, {}
    for p in pool.players:
        nid = dk2nba.get(str(p.dk_player_id))
        if nid is None or not p.game_id or not p.team or p.game_id not in game_ids:
            continue
        extras.setdefault((p.game_id, p.team), []).append(nid)
        if p.starter:
            starters.add(nid)
        salaries.setdefault(nid, int(p.salary))
        names[nid] = p.name
    return extras, starters, salaries, names, cap


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day")
    ap.add_argument("--offset", type=int, default=45)
    ap.add_argument("--sims", type=int, default=5_000)
    ap.add_argument("--out", default=os.environ.get("P2_OUT", "out"))
    ap.add_argument("--resolve-ids", action="store_true",
                    help="map the day's DK ids first (as the runner does at build time; "
                         "writes player_ids: use on a DB copy)")
    a = ap.parse_args(argv)
    conn = db.connect()
    if a.resolve_ids:
        print(IDS.resolve_dk(conn, since=f"{a.day}T00:00:00Z", write_check=False)["rate"])
    day = date.fromisoformat(a.day)
    season = P.season_of(day)
    ctx = P.Context(conn, season)
    b = P.load_bundle(conn, season, ctx)
    act = {(g, int(p)): float(m) for g, p, m in conn.execute(
        "SELECT game_id, player_id, min FROM player_games WHERE game_date=?", (a.day,))}
    frames = []
    for sid, gids, lock in slates(conn, a.day):
        at = lock - timedelta(minutes=a.offset)
        extras, starters, salaries, names, cap = pool_inputs(conn, sid, at, gids)
        gt = conn.execute("SELECT game_type_id FROM dfs_slates WHERE slate_id=?", (sid,))
        if gt.fetchone()[0] != P.CLASSIC_GAME_TYPE:     # as project_slate: Classic prices
            salaries = P.classic_salaries(conn, day)
        # the panel the live path builds (for the roster flag / fringe features)
        games = P._day_games(conn, day, gids)
        p_avail, rep_extras, _ = P.report_avail(conn, day, at, games, starters or None)
        ex = P.merge_extras(extras, rep_extras)
        panel = P.drop_moved_players(P.candidate_panel(ctx, day, games, ex),
                                     P.current_teams(conn, season), ex)
        ff = AV.fringe_features(panel, ctx.h, salaries)
        info = pd.DataFrame({"player_id": panel["player_id"].astype(np.int64),
                             "roster": panel["roster"].to_numpy(int)})
        for c in ("avg10", "app365", "team_gp", "salary", "regime"):
            info[c] = ff[c].to_numpy()
        res = {}
        for ver in ("v1", "v2"):
            P.set_pplay(ver)
            r = P.project_day(conn, day, at, n_sims=a.sims, kind="lock45", game_ids=gids,
                              ctx=ctx, bundle=b, extras=extras, starters=starters or None,
                              salaries=salaries if ver == "v2" else None,
                              seed=P.stable_seed(sid, "lock45", P.SEED_VERSION))
            res[ver] = r.players.set_index("player_id")
        P.set_pplay("v1")
        x = res["v1"][["game_id", "team", "p_avail", "p_play", "min_mean", "min_played",
                       "dk_mean"]].join(
            res["v2"][["p_play", "min_mean", "min_played", "dk_mean"]], rsuffix="_p2")
        x = x.reset_index().merge(info, on="player_id", how="left")
        x["slate_id"] = sid
        x["at"] = P._iso(at)
        x["capture"] = cap
        x["name"] = [names.get(int(p), "") for p in x["player_id"]]
        x["listed"] = [int(p) in P.listed_of(rep_extras) for p in x["player_id"]]
        x["on_report"] = [int(p) in p_avail for p in x["player_id"]]
        x["dk_starter"] = [int(p) in starters for p in x["player_id"]]
        x["minutes"] = [act.get((g, int(p)), 0.0) for g, p in zip(x["game_id"], x["player_id"])]
        x["played"] = (x["minutes"] > 0).astype(int)
        frames.append(x)
    df = pd.concat(frames, ignore_index=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    f = od / f"opener_{a.day}_lock{a.offset}.parquet"
    df.to_parquet(f)
    pd.set_option("display.width", 220)
    print(f"wrote {f} ({len(df)} rows)")
    print("\nteam minutes (sum of P x E[min|plays] = min_mean), off vs p2, actual")
    t = df.groupby(["game_id", "team"]).agg(off=("min_mean", "sum"), p2=("min_mean_p2", "sum"),
                                            actual=("minutes", "sum"))
    print(t.round(1).to_string())
    fr = df[(df["roster"] == 0) & ~df["on_report"]]
    print("\nfringe (forced, not in last 10, not on the report)")
    print(fr[["team", "name", "salary", "avg10", "app365", "regime", "p_play", "min_mean",
              "p_play_p2", "min_mean_p2", "dk_mean", "dk_mean_p2", "minutes"]]
          .sort_values("min_mean", ascending=False).round(2).to_string(index=False))
    li = df[df["listed"]]
    print("\nlisted, not Out")
    print(li[["team", "name", "p_avail", "p_play", "p_play_p2", "min_mean", "min_mean_p2",
              "dk_mean", "dk_mean_p2", "minutes"]].round(2).to_string(index=False))
    print("\ntop 15 by DK mean (p2)")
    print(df.sort_values("dk_mean_p2", ascending=False).head(15)[
        ["team", "name", "salary", "p_play", "p_play_p2", "min_mean", "min_mean_p2", "dk_mean",
         "dk_mean_p2", "minutes"]].round(2).to_string(index=False))



def lineups(day: str, offset: int = 45, sims: int = 5_000, n_top: int = 10) -> None:
    """Lineup check on the day's Classic slates: cash lineup + Top-N (min-diff 1) from the off
    and the p2 projections at the slate's lock - offset, scored with actual DK points (DNP 0)."""
    from courtside.proj.dfs import optimizer as O
    from courtside.proj.dfs import topn as T
    conn = db.connect()
    d = date.fromisoformat(day)
    season = P.season_of(d)
    ctx = P.Context(conn, season)
    b = P.load_bundle(conn, season, ctx)
    act = {int(r[0]): R.dk_points(dict(zip(("pts", "fg3m", "reb", "ast", "stl", "blk", "tov"),
                                           r[1:])))
           for r in conn.execute("SELECT player_id, pts, fg3m, reb, ast, stl, blk, tov FROM "
                                 "player_games WHERE game_date=?", (day,))}
    rows = conn.execute("SELECT slate_id, game_ids_json, lock_utc FROM dfs_slates WHERE "
                        "nba_day=? AND game_type_id=?", (day, P.CLASSIC_GAME_TYPE)).fetchall()
    for sid, gj, lock in rows:
        gids = json.loads(gj)
        at = _utc(lock) - timedelta(minutes=offset)
        extras, starters, salaries, names, cap = pool_inputs(conn, str(sid), at, gids)
        pool = R.build_pool(conn, str(sid), captured_at=cap)
        dk2nba = IDS.nba_ids(conn, "dk", [p.dk_player_id for p in pool.players])
        print(f"\n=== slate {sid} ({len(gids)} games) at {P._iso(at)}, pool {len(pool.players)}")
        for ver in ("v1", "v2"):
            P.set_pplay(ver)
            r = P.project_day(conn, d, at, n_sims=sims, kind="lock45", game_ids=gids, ctx=ctx,
                              bundle=b, extras=extras, starters=starters or None,
                              salaries=salaries if ver == "v2" else None,
                              seed=P.stable_seed(str(sid), "lock45", P.SEED_VERSION))
            nba2dk = {}
            for dk, nid in dk2nba.items():
                nba2dk.setdefault(nid, dk)
            r.players["dk_id"] = [nba2dk.get(int(p)) for p in r.players["player_id"]]
            proj = r.projections(key="dk_id")
            score = lambda lu: sum(act.get(dk2nba.get(e.player.dk_player_id), 0.0)  # noqa: E731
                                   for e in lu.entries)
            dnp = lambda lu: [e.player.name for e in lu.entries  # noqa: E731
                              if dk2nba.get(e.player.dk_player_id) not in act]
            cash = O.optimize(pool, proj, mode="cash", time_limit_s=20)
            top = T.top_n(pool, proj, n=n_top, min_diff=1, time_limit_s=20)
            tl = [x.lineup for x in top.lineups]
            print(f"  {ver}: cash projected {cash.projected:.1f} actual {score(cash):.1f} "
                  f"DNP {dnp(cash)}")
            print(f"       top-{len(tl)} actual mean {np.mean([score(x) for x in tl]):.1f} "
                  f"best {max(score(x) for x in tl):.1f}; DNP slots "
                  f"{sum(len(dnp(x)) for x in tl)} ({sorted({n for x in tl for n in dnp(x)})})")
            hind = O.optimize(pool, {k: v for k, v in
                                     ((dk, O.Projection(mean=act.get(n, 0.0), p25=act.get(n, 0.0),
                                                        p10=act.get(n, 0.0),
                                                        p50=act.get(n, 0.0),
                                                        p90=act.get(n, 0.0)))
                                      for dk, n in dk2nba.items())}, time_limit_s=20)
            if ver == "v1":
                print(f"  hindsight optimum {hind.projected:.1f}")
        P.set_pplay("v1")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "lineups":
        lineups(sys.argv[2])
    else:
        sys.exit(main())
