"""B8: official injury rows keyed by game date (availability.INJ_GAMEDAY, +gd).

Driver for docs/research/injury-gameday.md. Runs against a migrated (schema 16) and
`courtside backfill injuries --game-date`-filled DB copy; never writes the DB.

    python research/injury_gameday.py samples  [--db PATH]        # T_off / T_on bucket rows
    python research/injury_gameday.py frames   [--db PATH]        # serving-path frames, both arms
    python research/injury_gameday.py evaluate                    # walk-forward log loss ON vs OFF
    python research/injury_gameday.py fit [--db PATH] [--no-write]  # availability_params_gd.json
    python research/injury_gameday.py identity --ref availability_ref.py --day D [--slate ID] [--save F]

`frames` calls the real serving path (`availability._slate`, with its ESPN / older-report
fallback) for every regular-season day in the fit window at the 10:00 and 13:30 ET pregame runs
and at lock-45 / lock-10 of every game, once per arm (OFF = capture-day keying, ON = game-date
keying), and labels the rows with who played. `evaluate` prices every row out of sample from the
arm's own bucket rows: 'fwd' = cells fit on the months before the row's month, 'lomo' = every
other month; lock-10 box starters get 0.99 in both arms (the live DK-flag override).
Outputs under $GD_OUT (default var/tmp/gd).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config
from courtside.proj import availability as AV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pplay_q_refit import cells_p, clustered, dk_starters, frame_hash, load_ref, month  # noqa: E402

OUT = Path(os.environ.get("GD_OUT", config.REPO_ROOT / "var" / "tmp" / "gd"))
LO, HI = AV.TRAIN_LO, AV.RS_END
CLOCKS = ((10, 0), (13, 30))
LOCKS = (45, 10)
SCENARIOS = ("ET10:00", "ET13:30", "T-45m", "T-10m")
ET = config.NBA_TZ


def say(m: str) -> None:
    print(f"[gd] {time.strftime('%H:%M:%S')} {m}", flush=True)


def ro_conn(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.db_path()
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    for c in ("at", "tip"):
        if c in df:
            df[c] = df[c].astype(str)
    df.to_parquet(path)


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{name}.parquet")
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


# ----------------------------------------------------------------------------- samples

def cmd_samples(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = ro_conn(a.db)
    t0 = time.time()
    hist = AV.load_history(conn, LO)
    for arm in ("off", "on"):
        AV.set_inj_gameday(arm == "on")
        T = AV.build_samples(conn, LO, HI, mode="bucket", hist=hist)
        say(f"{arm}: bucket rows {len(T)} ({time.time() - t0:.0f}s)")
        _to_parquet(T, OUT / f"T_{arm}.parquet")
    AV.set_inj_gameday(False)
    say(f"wrote {OUT}")


# ----------------------------------------------------------------------------- frames

def report_days(conn, lo: date, hi: date) -> list[date]:
    days = [date.fromisoformat(r[0]) for r in conn.execute(
        "SELECT DISTINCT game_date FROM team_games WHERE game_date>=? AND game_date<=? AND "
        "season_type='Regular Season' ORDER BY game_date", (lo.isoformat(), hi.isoformat()))]
    have = {r[0] for r in conn.execute(
        "SELECT DISTINCT COALESCE(game_date, substr(captured_at, 1, 10)) FROM injury_snapshots "
        "WHERE source='nba_official'")}
    return [d for d in days if d.isoformat() in have]


def box_starters(conn, lo: date, hi: date) -> set[tuple[str, int]]:
    return {(r[0], int(r[1])) for r in conn.execute(
        "SELECT ps.game_id, ps.player_id FROM player_starts ps JOIN team_games tg ON "
        "tg.game_id = ps.game_id AND tg.team = ps.team WHERE tg.game_date>=? AND tg.game_date<=?",
        (lo.isoformat(), hi.isoformat()))}


def cmd_frames(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = ro_conn(a.db)
    hist = AV.load_history(conn, LO)
    prm = AV.AvailabilityParams()                      # P is re-priced out of sample later
    days = report_days(conn, LO, HI)
    if a.dates:
        days = days[:a.dates]
    starters = box_starters(conn, LO, HI)
    say(f"{len(days)} days {days[0]}..{days[-1]}, {len(starters)} box starters")
    rows = []
    t0 = time.time()
    for i, day in enumerate(days, 1):
        games = AV.day_games(conn, day)
        if games.empty:
            continue
        tips = dict(zip(games["game_id"], games["tip"]))
        decisions = [(f"ET{h:02d}:{m:02d}",
                      datetime(day.year, day.month, day.day, h, m, tzinfo=ET).astimezone(timezone.utc),
                      None) for h, m in CLOCKS]
        for g in games.itertuples():
            for off in LOCKS:
                decisions.append((f"T-{off}m", g.tip - timedelta(minutes=off), g.game_id))
        for scen, at, gid in decisions:
            for arm in ("off", "on"):
                AV.set_inj_gameday(arm == "on")
                df = AV._slate(conn, day, at, prm, game_ids={gid} if gid else None, hist=hist)
                if df.empty:
                    continue
                df = df[[tips[g] > at for g in df["game_id"]]]
                df = df.assign(day=day, scenario=scen, arm=arm, at=_iso(at),
                               tip=[_iso(tips[g]) for g in df["game_id"]],
                               played=[int((g, int(p)) in hist.played)
                                       for g, p in zip(df["game_id"], df["player_id"])],
                               started=[int((g, int(p)) in starters)
                                        for g, p in zip(df["game_id"], df["player_id"])])
                rows.append(df)
        AV.set_inj_gameday(False)
        if i % 10 == 0:
            say(f"{i}/{len(days)} days, {sum(len(r) for r in rows)} rows ({time.time() - t0:.0f}s)")
    F = pd.concat(rows, ignore_index=True).rename(columns={"reason_class": "reason",
                                                           "hours_to_tip": "hours"})
    F.to_parquet(OUT / "frames.parquet")
    say(f"wrote {len(F)} rows -> {OUT / 'frames.parquet'}")


# ----------------------------------------------------------------------------- evaluation

def oos_p(T: pd.DataFrame, F: pd.DataFrame, scheme: str) -> np.ndarray:
    """Cell P for F rows, never fit on the row's own month ('lomo') / only earlier months ('fwd')."""
    out = np.full(len(F), np.nan)
    tm = np.array([month(d) for d in T["day"]])
    fm = np.array([month(d) for d in F["day"]])
    for m in sorted(set(fm)):
        tr = T[tm != m] if scheme == "lomo" else T[tm < m]
        if tr.empty:
            continue
        idx = np.where(fm == m)[0]
        out[idx] = cells_p(tr, F.iloc[idx])
    return out


def ll(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), AV.CLAMP[0], AV.CLAMP[1])
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


KEY = ["day", "scenario", "game_id", "player_id"]
SUBSETS = {"all": lambda d: np.ones(len(d), bool),
           "listed_either": lambda d: d["status_off"].isin(AV.LISTED) | d["status_on"].isin(AV.LISTED),
           "uncertain_either": lambda d: d["status_off"].isin(AV.UNCERTAIN) | d["status_on"].isin(AV.UNCERTAIN),
           "status_differs": lambda d: d["status_off"] != d["status_on"],
           "p_differs": lambda d: d["p_off"] != d["p_on"]}


def paired(J: pd.DataFrame) -> dict:
    y = J["played"].to_numpy()
    lo, ln = ll(y, J["p_off"]), ll(y, J["p_on"])
    d, se = clustered(ln - lo, J["day"])
    db_, seb = clustered((J["p_on"] - y) ** 2 - (J["p_off"] - y) ** 2, J["day"])
    return {"n": len(J), "days": J["day"].nunique(), "played": float(y.mean()),
            "p_off": float(J["p_off"].mean()), "p_on": float(J["p_on"].mean()),
            "ll_off": float(lo.mean()), "ll_on": float(ln.mean()), "d_ll": d, "se": se,
            "z": d / se if se > 0 else np.nan, "d_brier": db_, "se_brier": seb}


def price(F: pd.DataFrame, T: dict[str, pd.DataFrame], scheme: str) -> pd.DataFrame:
    """Out-of-sample P per arm with the lock-10 starter override, then OFF/ON paired by row."""
    parts = []
    for arm in ("off", "on"):
        A = F[F["arm"] == arm].copy()
        p = oos_p(T[arm], A, scheme)
        m = (A["scenario"] == "T-10m").to_numpy() & (A["started"] == 1).to_numpy()
        p[m] = 0.99
        A["p"] = p
        A = A[np.isfinite(p)]
        parts.append(A.set_index(KEY)[["status", "source", "bucket", "reason", "role", "p", "played",
                                       "started", "report_time"]].add_suffix(f"_{arm}"))
    J = parts[0].join(parts[1], how="outer").reset_index()
    J["played"] = J["played_off"].fillna(J["played_on"]).astype(int)
    J["role"] = J["role_off"].fillna(J["role_on"])
    return J


def cmd_evaluate(a) -> None:
    F = load("frames")
    T = {arm: load(f"T_{arm}") for arm in ("off", "on")}
    for arm in T:
        T[arm] = T[arm][T[arm]["season_type"] == "Regular Season"].reset_index(drop=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    txt: list[str] = []

    def p(s=""):
        txt.append(s)
        print(s, flush=True)

    p(f"frames {len(F)} rows, {F['day'].nunique()} days; T_off {len(T['off'])} T_on {len(T['on'])} bucket rows")
    # --- what the two keyings see (no pricing) ---
    p("\n== source share by scenario x arm (share of candidate rows priced from an official batch)")
    src = F.assign(official=(F["source"] == "nba_official")).groupby(["scenario", "arm"]).agg(
        rows=("player_id", "size"), official=("official", "mean"), prior=("source", lambda s: (s == "prior").mean()),
        espn=("source", lambda s: (s == "espn").mean()),
        removed=("status", lambda s: (s == "removed").mean()),
        noreport=("status", lambda s: (s == "unlisted_noreport").mean()))
    p(src.round(3).to_string())
    p("\n== hours report->tip of the batch used, listed rows, median by scenario x arm")
    L = F[F["status"].isin(AV.LISTED)]
    p(L.groupby(["scenario", "arm"])["hours"].describe()[["count", "25%", "50%", "75%"]].round(1).to_string())

    # --- cells: OFF vs ON refit on the whole window ---
    prm = {arm: AV.fit_from_samples(T[arm], m1=20.0, m=AV.DEFAULT_M) for arm in T}
    p("\n== status x bucket cells (level 2 P(play), n): OFF, ON, delta")
    to = prm["off"].table().set_index("status")
    tn = prm["on"].table().set_index("status")
    cells = []
    for s in AV.STATUSES:
        r = {"status": s, "n_off": int(to.loc[s, "n"]), "n_on": int(tn.loc[s, "n"])}
        for b in AV.BUCKETS:
            r[f"{b}"] = f"{to.loc[s, b]:.3f}->{tn.loc[s, b]:.3f} ({tn.loc[s, b] - to.loc[s, b]:+.3f})"
            r[f"n_{b}"] = f"{int(to.loc[s, f'n_{b}'])}->{int(tn.loc[s, f'n_{b}'])}"
        cells.append(r)
    p(pd.DataFrame(cells).to_string(index=False))
    p("\n== ON gt6h bucket rows: the day-before share and play rate by status (hours >= 12 = report from the day before)")
    g6 = T["on"][(T["on"]["bucket"] == "gt6h") & T["on"]["status"].isin(AV.LISTED + ("removed",))].copy()
    g6["day_before"] = g6["hours"] >= 12
    p(g6.groupby(["status", "day_before"])["played"].agg(["size", "mean"]).round(3).to_string())

    # --- out-of-sample pricing ---
    res = {}
    for scheme in ("fwd", "lomo"):
        J = price(F, T, scheme)
        both = J["p_off"].notna() & J["p_on"].notna()
        p(f"\n#### scheme {scheme}: {int(both.sum())} paired rows, {int((~both).sum())} in one arm only "
          f"(ON only {int((J['p_off'].isna() & J['p_on'].notna()).sum())}, OFF only {int((J['p_on'].isna() & J['p_off'].notna()).sum())})")
        JJ = J[both].reset_index(drop=True)
        rows = []
        for scen in SCENARIOS:
            S = JJ[JJ["scenario"] == scen]
            for name, f in SUBSETS.items():
                sub = S[f(S)]
                if len(sub):
                    rows.append({"scenario": scen, "subset": name, **paired(sub)})
        R = pd.DataFrame(rows)
        res[scheme] = R
        p("== P(play) log loss, ON vs OFF, paired rows, SE clustered by day (d = ON - OFF; negative = ON better)")
        p(R.round(4).to_string(index=False))
        p("\n== by status (OFF status) at the 10:00 / 13:30 runs")
        for scen in ("ET10:00", "ET13:30"):
            S = JJ[JJ["scenario"] == scen]
            t = S.groupby("status_off").apply(lambda g: pd.Series(paired(g)), include_groups=False)
            p(f"-- {scen}\n" + t[["n", "played", "p_off", "p_on", "ll_off", "ll_on", "d_ll", "se"]].round(4).to_string())
        p("\n== status transitions OFF -> ON at the 10:00 run (rows)")
        S = JJ[JJ["scenario"] == "ET10:00"]
        p(pd.crosstab(S["status_off"], S["status_on"]).to_string())
        one = J[~both]
        if len(one):
            p("\n== rows in one arm only, by scenario x arm present")
            one = one.assign(arm=np.where(one["p_on"].notna(), "on", "off"),
                             status=one["status_on"].fillna(one["status_off"]))
            p(one.groupby(["scenario", "arm", "status"])["played"].agg(["size", "mean"]).round(3).to_string())
        R.to_parquet(OUT / f"eval_{scheme}.parquet")
    (OUT / "evaluate.txt").write_text("\n".join(txt) + "\n")
    say(f"wrote {OUT / 'evaluate.txt'}")


# ----------------------------------------------------------------------------- fit

def cmd_fit(a) -> None:
    conn = ro_conn(a.db)
    T = load("T_on") if (OUT / "T_on.parquet").exists() else None
    AV.set_inj_gameday(True)
    prm = AV.fit(conn, samples=T, seed_note="game-date keyed batches (B8, +gd); "
                 "research/injury_gameday.py fit; docs/research/injury-gameday.md")
    AV.set_inj_gameday(False)
    print(f"fitted on {prm.meta['rows']} rows / {prm.meta['games']} team-games, keying {prm.meta['keying']}")
    print(prm.table().round(3).to_string(index=False))
    out = prm.save(OUT / "availability_params_gd.json")
    print("wrote", out)
    if not a.no_write:
        print("wrote", prm.save(AV.GD_PARAMS_PATH))
        AV._PARAMS_CACHE.clear()


# ----------------------------------------------------------------------------- identity

def cmd_identity(a) -> None:
    """OFF (this tree, switch off, game_date filled) vs a reference availability.py: identical
    p_play_slate frames at the 10:00 / 13:30 runs and lock-45 / lock-10 of every tip (box
    starters at lock-10; --slate: the DK starter flags of the time).  Also counts the rows ON
    changes."""
    conn = ro_conn(a.db)
    ref = load_ref(a.ref)
    cases = []
    for d in a.day or []:
        day = date.fromisoformat(d)
        tips = AV.day_games(conn, day)
        st = {int(r[0]) for r in conn.execute(
            "SELECT player_id FROM player_starts WHERE game_id IN (SELECT game_id FROM team_games "
            "WHERE game_date=?)", (d,))}
        for h, m in CLOCKS:
            cases.append((f"{d} ET{h:02d}:{m:02d}", day,
                          datetime(day.year, day.month, day.day, h, m, tzinfo=ET).astimezone(timezone.utc), None))
        for tip in sorted(tips["tip"].unique()):
            for off in LOCKS:
                cases.append((f"{d} tip {pd.Timestamp(tip):%H:%M}Z lock-{off}", day,
                              pd.Timestamp(tip).to_pydatetime() - timedelta(minutes=off),
                              st if off == 10 and st else None))
    for sid in a.slate or []:
        r = conn.execute("SELECT nba_day, game_ids_json FROM dfs_slates WHERE slate_id=?", (sid,)).fetchone()
        day = date.fromisoformat(str(r[0])[:10])
        gids = set(json.loads(r[1])) if r[1] else None
        tips = AV.day_games(conn, day)
        if gids:
            tips = tips[tips["game_id"].isin(gids)]
        for h, m in CLOCKS:
            cases.append((f"slate {sid} ET{h:02d}:{m:02d}", day,
                          datetime(day.year, day.month, day.day, h, m, tzinfo=ET).astimezone(timezone.utc), None))
        for tip in sorted(tips["tip"].unique()):
            for off in LOCKS:
                at = pd.Timestamp(tip).to_pydatetime() - timedelta(minutes=off)
                st = dk_starters(conn, sid, _iso(at))
                cases.append((f"slate {sid} tip {pd.Timestamp(tip):%H:%M}Z lock-{off} ({len(st)} DK starters)",
                              day, at, st or None))
    out, same_n = [], 0
    for name, day, at, st in cases:
        AV.set_inj_gameday(False)
        f_off = AV.p_play_slate(conn, day, at, starters=st)
        f_ref = ref.p_play_slate(conn, day, at, starters=st)
        AV.set_inj_gameday(True)
        AV._PARAMS_CACHE.clear()
        f_on = AV.p_play_slate(conn, day, at, starters=st)
        AV.set_inj_gameday(False)
        same = frame_hash(f_off) == frame_hash(f_ref) and \
            np.array_equal(f_off["p_play"].to_numpy(), f_ref["p_play"].to_numpy())
        same_n += same
        j = f_off.merge(f_on[["game_id", "team", "player_id", "status", "p_play", "source", "report_time"]],
                        on=["game_id", "team", "player_id"], how="outer", suffixes=("", "_on"))
        ch = j[(j["p_play"] != j["p_play_on"]) | (j["status"] != j["status_on"])]
        out.append(f"{name}: rows {len(f_off)}, OFF {frame_hash(f_off)} ref {frame_hash(f_ref)} "
                   f"{'IDENTICAL' if same else 'DIFFERENT'}; ON changes {len(ch)} rows "
                   f"(status {int((j['status'] != j['status_on']).sum())})")
        for r in ch.head(a.show).itertuples(index=False):
            out.append(f"    {r.team} {r.player or r.player_id} {r.status}->{r.status_on} "
                       f"{r.p_play:.3f}->{r.p_play_on:.3f} src {r.source}->{r.source_on} "
                       f"rpt {r.report_time}->{r.report_time_on}")
    out.append(f"{same_n}/{len(cases)} frames identical OFF vs reference")
    s = "\n".join(out)
    print(s)
    if a.save:
        Path(a.save).write_text(s + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["samples", "frames", "evaluate", "fit", "identity"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--dates", type=int, default=0, help="frames: first N days only")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--ref", default=None, help="identity: reference availability.py")
    ap.add_argument("--day", action="append", help="identity: NBA day (repeatable)")
    ap.add_argument("--slate", action="append", help="identity: DK slate id (repeatable)")
    ap.add_argument("--show", type=int, default=8, help="identity: changed rows to list per frame")
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    {"samples": cmd_samples, "frames": cmd_frames, "evaluate": cmd_evaluate, "fit": cmd_fit,
     "identity": cmd_identity}[a.cmd](a)


if __name__ == "__main__":
    main()
