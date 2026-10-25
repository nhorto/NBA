"""Re-evaluation workstream A: live replay of the 2026-27 lock runs (OFF vs +gh / +rm / +lu).

Docs: docs/REEVAL_2026-10.md, docs/research/reeval-A.md.  Switches under test:
`pipeline.GHOST_DROP` (+gh), `availability.REPORT_MATCH` (+rm), `minutes.LINEUP` (+lu).

    python research/reeval_a_live_replay.py plan [--protocol lock10]
    python research/reeval_a_live_replay.py run ARM [--protocol lock10] [--sims 4000] [--day D]
    python research/reeval_a_live_replay.py report [--protocol lock10] [--arms gh,rm,lu,all]
    python research/reeval_a_live_replay.py identity A.parquet B.parquet

Units.  For every game of NBA days 2026-10-20 .. 24, the newest ok `dfs_runs` run of the
protocol's kind (lock10 / lock45) created before the tip that covers it (Classic or Showdown
slate).  Each unit is re-projected exactly the way `pipeline.project_slate` did it at the run
time: the slate's DK pool from its newest `dfs_salaries` capture AT OR BEFORE the run (forced
extras, DK starter flags, DK 'O' statuses, Classic salaries), the official / ESPN reports and
book lines published by then (the live as-of joins), the live seed `stable_seed(slate, kind,
SEED_VERSION)` (common random numbers across arms; every game draws from its own
SeedSequence, so projecting a subset of the slate's games does not change its draws).  All arms
use today's live stack and the prod bundle (`bundle_2026-27_v1_cx_mf_ep_ms_c1r.json` +
`corr_2026-27.json` in $COURTSIDE_DATA/proj; geekom1 copy: ~/Backups/courtside/var/proj), so
the arm - off difference is the switch alone.

ARM: off (no setter is called: works on a reference code tree too, for `identity`), gh, rm,
lu, all (= gh + rm + lu).  Output: $REEVAL_A_OUT/replay/<protocol>_<arm>.parquet, one row per
projected (game, player): p_play, E[min], E[min | plays], DK mean / q10 / q90, DK and minutes
CRPS and DK PIT from the sims vs the box score (DNP = 0), the in-pool flag, plus a sha1 of each
game's sim arrays (identity).  The DB is opened read-only; nothing is written to it.

Light enough for geekom1 on one day at --sims 1000; the full replay (5 nights x 2 protocols x
5 arms at 4,000 sims) belongs on the Mac scratch (research/reeval_a_mac.sh).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config

OUT = Path(os.environ.get("REEVAL_A_OUT", config.REPO_ROOT / "var" / "tmp" / "reeval-a"))
LO, HI = "2026-10-20", "2026-10-24"
SEASON = "2026-27"
CLASSIC = 70
ARMS = ("off", "gh", "rm", "lu", "all")


def say(m: str) -> None:
    print(f"[replay] {time.strftime('%H:%M:%S')} {m}", flush=True)


def ro_conn(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.db_path()
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


# ----------------------------------------------------------------------------- units

def units(conn, protocol: str) -> pd.DataFrame:
    """(run_id, slate_id, kind, created_at, nba_day, game_type_id, game_id) per replayed game."""
    runs = pd.read_sql_query(
        "SELECT r.run_id, r.slate_id, r.kind, r.created_at, s.nba_day, s.game_type_id, "
        "s.game_ids_json FROM dfs_runs r JOIN dfs_slates s ON s.slate_id = r.slate_id "
        "WHERE r.ok = 1 AND r.kind = ? AND s.nba_day BETWEEN ? AND ?", conn,
        params=(protocol, LO, HI))
    tips = dict(conn.execute("SELECT game_id, tipoff_utc FROM games WHERE nba_day BETWEEN ? "
                             "AND ?", (LO, HI)).fetchall())
    rows = []
    for r in runs.itertuples(index=False):
        for g in json.loads(r.game_ids_json or "[]"):
            if tips.get(g) and r.created_at < tips[g]:
                rows.append({**r._asdict(), "game_id": g})
    df = pd.DataFrame(rows).drop(columns=["game_ids_json"])
    df = df.sort_values(["game_id", "created_at", "game_type_id"])
    return df.drop_duplicates("game_id", keep="last").reset_index(drop=True)


def cmd_plan(a) -> None:
    conn = ro_conn(a.db)
    u = units(conn, a.protocol)
    print(u.groupby(["nba_day", "run_id", "slate_id", "game_type_id", "created_at"])
          .size().rename("games").reset_index().to_string())
    print(f"{len(u)} games in {u['run_id'].nunique()} runs")


# ----------------------------------------------------------------------------- run

def set_arm(arm: str) -> None:
    from courtside.proj import pipeline as P
    if arm == "off":
        return
    on = {"gh": ("gh",), "rm": ("rm",), "lu": ("lu",), "all": ("gh", "rm", "lu")}[arm]
    P.set_ghost_drop("gh" in on)
    P.set_report_match("rm" in on)
    P.set_lineup("lu" in on)


def slate_inputs(conn, slate_id: str, at: str, day: date):
    """project_slate's DK inputs from the slate's newest capture at or before `at`."""
    from courtside.proj import ids as IDS
    from courtside.proj import pipeline as P
    from courtside.proj.dfs import rules as R
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (str(slate_id), at)).fetchone()[0]
    pool = R.build_pool(conn, slate_id, captured_at=cap)
    dk2nba = IDS.nba_ids(conn, "dk", [p.dk_player_id for p in pool.players])
    extras, starters, salaries, dk_out = {}, set(), {}, set()
    for p in pool.players:
        nid = dk2nba.get(str(p.dk_player_id))
        if nid is None or not p.game_id or not p.team:
            continue
        extras.setdefault((p.game_id, p.team), []).append(nid)
        if p.starter:
            starters.add(nid)
        if getattr(p, "status", None) == "O":
            dk_out.add(nid)
        if p.salary and pool.game_type_id == CLASSIC:
            salaries.setdefault(nid, int(p.salary))
    if P.PPLAY == "v2" and pool.game_type_id != CLASSIC:
        salaries = P.classic_salaries(conn, day)
    pool_ids = {n for v in extras.values() for n in v}
    return extras, starters, salaries, dk_out, pool_ids, cap


def _sha(*arrs) -> str:
    h = hashlib.sha1()
    for x in arrs:
        x = np.ascontiguousarray(x)
        h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def box(conn, game_ids) -> dict:
    from courtside.proj.dfs import grade as G
    return G.load_box(conn, set(game_ids))


def cmd_run(a) -> None:
    from courtside.proj import pipeline as P
    from courtside.proj.metrics import crps_samples_matrix, pit_randomized
    set_arm(a.arm)
    say(f"arm {a.arm}: {P.MODEL_VERSION}")
    conn = ro_conn(a.db)
    u = units(conn, a.protocol)
    if a.day:
        u = u[u["nba_day"].isin(a.day)]
    ctx = P.Context(conn, SEASON, log=say)
    b = P.load_bundle(conn, SEASON, ctx, log=say)
    rows, t0 = [], time.time()
    kw_ok = "dk_out" in P.project_day.__code__.co_varnames
    for (day_s, run_id, sid, kind, at), g in u.groupby(
            ["nba_day", "run_id", "slate_id", "kind", "created_at"], sort=True):
        day = date.fromisoformat(day_s)
        extras, starters, salaries, dk_out, pool_ids, cap = slate_inputs(conn, sid, at, day)
        seed = P.stable_seed(str(sid), kind, P.SEED_VERSION)
        kw = {"dk_out": dk_out} if kw_ok else {}
        t1 = time.time()
        res = P.project_day(conn, day, at, n_sims=a.sims, seed=seed, kind=kind,
                            game_ids=list(g["game_id"]), ctx=ctx, bundle=b, extras=extras,
                            starters=starters or None,
                            salaries=salaries if P.PPLAY == "v2" else None, **kw)
        bx = box(conn, g["game_id"])
        pl = res.players.drop_duplicates("player_id").set_index("player_id")
        rng = np.random.default_rng(P.stable_seed(str(sid), kind, "reeval-a-pit"))
        for gs in res.sims:
            pids = np.asarray(gs.player_id)
            y = np.array([bx[int(p)]["dk"] if int(p) in bx else 0.0 for p in pids], float)
            m = np.array([bx[int(p)]["min"] if int(p) in bx else 0.0 for p in pids], float)
            X = gs.fp.astype(float)
            Xm = gs.minutes.astype(float)
            crps = crps_samples_matrix(X, y)
            crps_m = crps_samples_matrix(Xm, m)
            pit = pit_randomized(X, y, rng.random(len(y)))
            sha = _sha(gs.fp, gs.minutes, gs.margin, gs.total)
            for j, pid in enumerate(pids):
                q = pl.loc[int(pid)]
                rows.append({
                    "nba_day": day_s, "run_id": run_id, "slate_id": sid, "kind": kind,
                    "at": at, "pool_capture": cap, "game_id": gs.game_id, "player_id": int(pid),
                    "team": str(gs.team[j]), "in_pool": int(pid) in pool_ids,
                    "dk_out": int(pid) in dk_out, "starter_flag": int(pid) in starters,
                    "p_play": float(q["p_play"]), "min_mean": float(q["min_mean"]),
                    "min_played": float(q.get("min_played", np.nan)),
                    "dk_mean": float(q["dk_mean"]), "dk_q10": float(q["dk_q10"]),
                    "dk_q90": float(q["dk_q90"]), "tier": q.get("tier"),
                    "y_dk": y[j], "y_min": m[j], "played": bool(m[j] > 0),
                    "crps": float(crps[j]), "crps_min": float(crps_m[j]), "pit": float(pit[j]),
                    "game_sha": sha})
        say(f"{day_s} {sid} {kind} {len(g)} games: {time.time() - t1:.0f}s")
    df = pd.DataFrame(rows)
    df["arm"] = a.arm
    df["model_version"] = P.MODEL_VERSION
    od = OUT / "replay"
    od.mkdir(parents=True, exist_ok=True)
    f = Path(a.out) if a.out else od / f"{a.protocol}_{a.arm}.parquet"
    df.to_parquet(f)
    say(f"wrote {f} ({len(df)} rows, {time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- report

def _pc(v, cl):
    from courtside.proj.metrics import paired_cluster
    r = paired_cluster(np.asarray(v, float), np.asarray(cl))
    return r["mean"], r["se"]


def paired(off: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """Rows of `off` joined to `arm` on (game, player); a player the arm did not project counts
    as a certain DNP there (P 0, E[min] 0, DK 0: CRPS = actual DK)."""
    k = ["game_id", "player_id"]
    cols = ["p_play", "min_mean", "min_played", "dk_mean", "dk_q10", "dk_q90", "crps",
            "crps_min"]
    m = off.merge(arm[k + cols], on=k, how="outer", suffixes=("_a", "_b"), indicator=True)
    for c in ("y_dk", "y_min", "played", "in_pool", "tier", "nba_day", "team"):
        if c in m:
            alt = arm.set_index(k)[c] if c in arm else None
            if alt is not None:
                fill = pd.Series([alt.get((g, p)) for g, p in zip(m["game_id"], m["player_id"])],
                                 index=m.index)
                m[c] = m[c].where(m[c].notna(), fill)
    for side in ("a", "b"):
        miss = m[f"dk_mean_{side}"].isna()
        for c in ("p_play", "min_mean", "dk_mean", "dk_q10", "dk_q90"):
            m.loc[miss, f"{c}_{side}"] = 0.0
        m.loc[miss, f"crps_{side}"] = m.loc[miss, "y_dk"].astype(float)
        m.loc[miss, f"crps_min_{side}"] = m.loc[miss, "y_min"].astype(float)
    m["played"] = m["played"].astype(bool)
    return m


def cmd_report(a) -> None:
    od = OUT / "replay"
    off = pd.read_parquet(od / f"{a.protocol}_off.parquet")
    lines = [f"## Live replay, protocol {a.protocol} ({off['nba_day'].nunique()} nights, "
             f"{off['game_id'].nunique()} games); arm - off, SE clustered by game\n"]
    hdr = ("| arm | population | n | DK CRPS off | diff | se | z | DK MAE diff | se | "
           "min MAE (played, E>=15) diff | se | P(play) Brier diff | se |")
    lines += [hdr, "|" + "---|" * 13]
    for arm in a.arms.split(","):
        f = od / f"{a.protocol}_{arm}.parquet"
        if not f.exists():
            continue
        m = paired(off, pd.read_parquet(f))
        for pop, sel in (("all, E[min]off>=10", m["min_mean_a"] >= 10),
                         ("DK pool, E[min]off>=10", (m["min_mean_a"] >= 10) & (m["in_pool"] == True)),  # noqa: E712
                         ("DK pool, all", m["in_pool"] == True)):  # noqa: E712
            s = m[sel]
            cl = s["game_id"].to_numpy()
            d_crps = s["crps_b"] - s["crps_a"]
            d_mae = (s["dk_mean_b"] - s["y_dk"]).abs() - (s["dk_mean_a"] - s["y_dk"]).abs()
            pm = s[s["played"] & (s["min_played_a"] >= 15)]
            d_min = ((pm["min_played_b"] - pm["y_min"]).abs()
                     - (pm["min_played_a"] - pm["y_min"]).abs())
            yp = s["played"].astype(float)
            d_br = (s["p_play_b"] - yp) ** 2 - (s["p_play_a"] - yp) ** 2
            c1, e1 = _pc(d_crps, cl)
            c2, e2 = _pc(d_mae, cl)
            c3, e3 = _pc(d_min, pm["game_id"])
            c4, e4 = _pc(d_br, cl)
            lines.append(f"| {arm} | {pop} | {len(s)} | {s['crps_a'].mean():.3f} | {c1:+.4f} | "
                         f"{e1:.4f} | {c1 / e1 if e1 else np.nan:+.1f} | {c2:+.4f} | {e2:.4f} | "
                         f"{c3:+.3f} | {e3:.3f} | {c4:+.5f} | {e4:.5f} |")
        tm = m[(m["min_mean_a"] >= 10) & (m["in_pool"] == True)]  # noqa: E712
        tm = tm.assign(tier=tm["tier"].fillna("none"))
        for t, s in tm.groupby("tier"):
            c, e = _pc(s["crps_b"] - s["crps_a"], s["game_id"])
            lines.append(f"| {arm} | tier {t} (DK pool) | {len(s)} | {s['crps_a'].mean():.3f} | "
                         f"{c:+.4f} | {e:.4f} | {c / e if e else np.nan:+.1f} | | | | | | |")
    text = "\n".join(lines)
    print(text)
    (od / f"report_{a.protocol}.md").write_text(text + "\n")


def cmd_identity(a) -> None:
    x = pd.read_parquet(a.a)
    y = pd.read_parquet(a.b)
    gx = x.drop_duplicates("game_id").set_index("game_id")["game_sha"]
    gy = y.drop_duplicates("game_id").set_index("game_id")["game_sha"]
    same = (gx == gy.reindex(gx.index)).sum()
    cols = ["game_id", "player_id", "p_play", "min_mean", "min_played", "dk_mean", "dk_q10",
            "dk_q90", "crps"]
    fx = x[cols].sort_values(cols[:2]).reset_index(drop=True)
    fy = y[cols].sort_values(cols[:2]).reset_index(drop=True)
    frame_same = fx.shape == fy.shape and bool((fx.fillna(-9) == fy.fillna(-9)).all().all())
    print(f"games with identical sim arrays: {same} / {len(gx)}; players frame identical: "
          f"{frame_same}")
    sys.exit(0 if (same == len(gx) and frame_same) else 1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--protocol", default="lock10", choices=("lock10", "lock45"))
        s.add_argument("--db", default=None)
        if name == "run":
            s.add_argument("arm", choices=ARMS)
            s.add_argument("--sims", type=int, default=4000)
            s.add_argument("--day", action="append")
            s.add_argument("--out", default=None)
        if name == "report":
            s.add_argument("--arms", default="gh,rm,lu,all")
    s = sub.add_parser("identity")
    s.add_argument("a")
    s.add_argument("b")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "identity": cmd_identity}[a.cmd](a)


if __name__ == "__main__":
    main()
