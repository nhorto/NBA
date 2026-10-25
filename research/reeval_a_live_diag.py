"""Re-evaluation workstream A, task 1: live diagnosis of P(play) and minutes (H6 / H7).

Docs: docs/REEVAL_2026-10.md (protocol), docs/research/reeval-A.md (results).

    python research/reeval_a_live_diag.py frame      # build + cache the live frame (read-only DB)
    python research/reeval_a_live_diag.py report     # all diagnosis tables (markdown on stdout)
    (--kinds lock45 / --kinds lock10: the last pre-tip run of that kind only)

Population: for every (game, player) of NBA days 2026-10-20 .. 2026-10-24 the LAST pre-tip
projection: the newest `player_projections` row whose `dfs_runs` run is ok and was created
before the game's `games.tipoff_utc` (any run kind: lock45 / lock10 / swap / manual).  Its
grade (`player_projection_grades`) gives played / actual minutes / actual DK.

Status at run time is rebuilt with the live code itself (`availability.p_play_slate` at the
run's `created_at`, with the DK starter flags of that slate's latest capture at or before the
run): listed status, hours-to-tip bucket, reason, role, source.  A projected player the slate
frame does not contain is "fringe" (not on the report, not in his team's last 10 games).

Pure SQL / pandas, no simulation.  The DB is opened read-only (`mode=ro`); output goes to
$REEVAL_A_OUT (default var/tmp/reeval-a).
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

OUT = Path(os.environ.get("REEVAL_A_OUT", config.REPO_ROOT / "var" / "tmp" / "reeval-a"))
LO, HI = "2026-10-20", "2026-10-24"
TIER_EDGES = (12.0, 20.0, 26.0, 32.0)
TIERS = ("bench", "fringe_rot", "rotation", "starter", "star")   # by E[min | plays]
P_BINS = (0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1.0001)


def say(m: str) -> None:
    print(f"[reeval-a] {time.strftime('%H:%M:%S')} {m}", file=sys.stderr, flush=True)


def ro_conn(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.db_path()
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


# ----------------------------------------------------------------------------- frame

def _q(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


KINDS: tuple[str, ...] = ()           # --kinds: restrict the runs (default: every kind)


def last_pretip(conn) -> pd.DataFrame:
    df = _q(conn, """
        SELECT pp.id AS projection_id, pp.game_id, pp.nba_day AS day, pp.player_id, pp.team,
               pp.run_id, pp.slate_id, pp.created_at, pp.model_version, pp.role AS pp_role,
               pp.p_play, pp.minutes_json, pp.inputs_json, r.kind, r.created_at AS run_at,
               g.tipoff_utc, g.home, g.away, g.home_score, g.away_score
        FROM player_projections pp JOIN dfs_runs r ON r.run_id = pp.run_id
        JOIN games g ON g.game_id = pp.game_id
        WHERE pp.nba_day BETWEEN ? AND ? AND r.ok = 1 AND r.created_at < g.tipoff_utc
          AND COALESCE(pp.slate_id, '') NOT LIKE 'rg:%'""", (LO, HI))
    if KINDS:
        df = df[df["kind"].isin(KINDS)]
    df = df.sort_values(["game_id", "player_id", "run_at", "projection_id"])
    df = df.drop_duplicates(["game_id", "player_id"], keep="last").reset_index(drop=True)
    gr = _q(conn, """SELECT projection_id, played, actual_min, actual_dk, proj_min, proj_dk, pit,
                     in80, crps, dk_fppg, salary FROM player_projection_grades""")
    df = df.merge(gr, on="projection_id", how="left")
    st = _q(conn, f"""SELECT projection_id, stat, cond, mean, q10, q50, q90
                      FROM player_projection_stats WHERE projection_id IN
                      ({','.join(str(int(x)) for x in df['projection_id'])})
                      AND stat IN ('min', 'dk')""")
    for (s, c), g in st.groupby(["stat", "cond"]):
        g = g.set_index("projection_id")
        for col in ("mean", "q10", "q50", "q90"):
            df[f"{s}_{c}_{col}"] = df["projection_id"].map(g[col])
    inp = df["inputs_json"].map(lambda s: json.loads(s or "{}"))
    df["p_avail_in"] = inp.map(lambda d: d.get("p_avail"))
    df["mu_margin"] = inp.map(lambda d: (d.get("env") or {}).get("mu_margin"))
    df["dk_id"] = inp.map(lambda d: d.get("dk_id"))
    df["hours_run_to_tip"] = ((pd.to_datetime(df["tipoff_utc"]) - pd.to_datetime(df["run_at"]))
                              .dt.total_seconds() / 3600)
    return df.drop(columns=["inputs_json", "minutes_json"])


def _starters_at(conn, slate_id: str, at: str) -> tuple[set, dict]:
    """(DK starter-flag NBA ids, {nba id: DK status}) from the slate's latest capture <= at."""
    cap = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=? AND "
                       "captured_at<=?", (str(slate_id), at)).fetchone()[0]
    if cap is None:
        return set(), {}
    rows = conn.execute("SELECT s.site_player_id, s.starter, s.status, i.player_id FROM "
                        "dfs_salaries s LEFT JOIN player_ids i ON i.source='dk' AND "
                        "i.source_id=s.site_player_id WHERE s.slate_id=? AND s.captured_at=?",
                        (str(slate_id), cap)).fetchall()
    st = {int(r[3]) for r in rows if r[3] is not None and r[1] == 1}
    dks = {int(r[3]): (r[2] or "") for r in rows if r[3] is not None}
    return st, dks


def status_frames(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the availability frame of every (day, run time, slate) used."""
    out = []
    hist_by_day: dict = {}
    keys = df[["day", "run_at", "slate_id"]].drop_duplicates().itertuples(index=False)
    for day_s, at, sid in keys:
        day = date.fromisoformat(day_s)
        if day not in hist_by_day:
            hist_by_day[day] = AV.load_history(conn, day)
        starters, dks = _starters_at(conn, sid, at)
        av = AV.p_play_slate(conn, day, at, starters=starters or None, hist=hist_by_day[day])
        av = av.assign(day=day_s, run_at=at, slate_id=sid)
        av["dk_status"] = av["player_id"].map(lambda p: dks.get(int(p), None))
        av["starter_flag"] = av["player_id"].isin(starters)
        out.append(av)
        # players of this run that are not in the frame: DK status only
        sub = df[(df["day"] == day_s) & (df["run_at"] == at) & (df["slate_id"] == sid)]
        miss = sub[~sub["player_id"].isin(av["player_id"])]
        if len(miss):
            out.append(pd.DataFrame({"player_id": miss["player_id"], "game_id": miss["game_id"],
                                     "day": day_s, "run_at": at, "slate_id": sid,
                                     "status": "fringe",
                                     "starter_flag": miss["player_id"].isin(starters),
                                     "dk_status": miss["player_id"].map(
                                         lambda p: dks.get(int(p), None))}))
    return pd.concat(out, ignore_index=True)


def final_report_status(conn, df: pd.DataFrame) -> pd.Series:
    """Status in the newest official report published before the tip (late news check)."""
    res = []
    hist_by_day: dict = {}
    for (day_s, gid, tip), g in df.groupby(["day", "game_id", "tipoff_utc"]):
        day = date.fromisoformat(day_s)
        if day not in hist_by_day:
            hist_by_day[day] = AV.load_history(conn, day)
        at = (pd.Timestamp(tip) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        av = AV.p_play_slate(conn, day, at, hist=hist_by_day[day])
        av = av[av["game_id"] == gid].drop_duplicates("player_id").set_index("player_id")
        for idx, pid in zip(g.index, g["player_id"]):
            res.append((idx, av["status"].get(int(pid), "fringe"),
                        av["reason_class"].get(int(pid), None)))
    s = pd.DataFrame(res, columns=["idx", "status_tip", "reason_tip"]).set_index("idx")
    return s


def context_cols(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Game and player context: OT, margin, actual starter, history-based tags."""
    gids = ",".join(f"'{g}'" for g in df["game_id"].unique())
    tm = _q(conn, f"SELECT game_id, team, SUM(min) AS team_min FROM player_games WHERE "
                  f"game_id IN ({gids}) GROUP BY 1, 2")
    ot = tm.groupby("game_id")["team_min"].max()
    df["n_ot"] = df["game_id"].map(((ot - 240) / 25).round().clip(lower=0)).astype(int)
    df["margin"] = (df["home_score"] - df["away_score"]).abs()
    df["team_won"] = np.where(df["team"] == df["home"], df["home_score"] > df["away_score"],
                              df["away_score"] > df["home_score"])
    stt = _q(conn, f"SELECT game_id, player_id FROM player_starts WHERE game_id IN ({gids})")
    sset = set(zip(stt["game_id"], stt["player_id"].astype(int)))
    df["started"] = [(g, int(p)) in sset for g, p in zip(df["game_id"], df["player_id"])]
    df["starts_known"] = df["game_id"].isin(set(stt["game_id"]))
    # history before 2026-27
    pids = ",".join(str(int(p)) for p in df["player_id"].unique())
    pg = _q(conn, f"SELECT player_id, team, season, game_date, game_id, min FROM player_games "
                  f"WHERE player_id IN ({pids}) AND game_date >= '2024-10-01' AND min > 0")
    prev = pg[pg["season"] < "2026-27"].sort_values("game_date")
    last_team = prev.groupby("player_id")["team"].last()
    l10 = prev.groupby("player_id").tail(10).groupby("player_id")["min"].mean()
    df["last_season_team"] = df["player_id"].map(last_team)
    df["prev_l10_min"] = df["player_id"].map(l10)
    df["no_nba_history"] = df["last_season_team"].isna()
    df["new_team"] = df["last_season_team"].notna() & (df["last_season_team"] != df["team"])
    # appearances / team games this season before the day
    cur = pg[pg["season"] == "2026-27"]
    tg = _q(conn, "SELECT game_id, team, game_date FROM team_games WHERE season='2026-27'")
    app_before, team_gp, l_min = [], [], []
    for pid, team, day in zip(df["player_id"], df["team"], df["day"]):
        a = cur[(cur["player_id"] == pid) & (cur["game_date"] < day)]
        app_before.append(len(a))
        team_gp.append(int(((tg["team"] == team) & (tg["game_date"] < day)).sum()))
        l_min.append(float(a.sort_values("game_date")["min"].iloc[-1]) if len(a) else np.nan)
    df["app_season"] = app_before
    df["team_gp"] = team_gp
    df["last_game_min"] = l_min
    rost = _q(conn, "SELECT player_id, exp FROM team_rosters WHERE season='2026-27'")
    df["rookie"] = df["player_id"].map(rost.drop_duplicates("player_id").set_index(
        "player_id")["exp"]).fillna(-1).astype(int) == 0
    return df


def cmd_frame(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = ro_conn(a.db)
    t0 = time.time()
    df = last_pretip(conn)
    say(f"last pre-tip rows {len(df)} ({time.time() - t0:.0f}s)")
    av = status_frames(conn, df)
    av = av.drop_duplicates(["day", "run_at", "slate_id", "player_id"])
    keep = ["day", "run_at", "slate_id", "player_id", "status", "p_play", "source", "reason_class",
            "role", "hours_to_tip", "bucket", "override", "starter_flag", "dk_status"]
    av = av[[c for c in keep if c in av]].rename(columns={"p_play": "p_cell", "role": "av_role"})
    df = df.merge(av, on=["day", "run_at", "slate_id", "player_id"], how="left")
    df["status"] = df["status"].fillna("fringe")
    say(f"status frames ({time.time() - t0:.0f}s)")
    fs = final_report_status(conn, df)
    df = df.join(fs)
    say(f"tip-time status ({time.time() - t0:.0f}s)")
    df = context_cols(conn, df)
    say(f"context ({time.time() - t0:.0f}s)")
    f = OUT / f"live_frame{'_' + '_'.join(KINDS) if KINDS else ''}.parquet"
    df.to_parquet(f)
    say(f"wrote {f} ({len(df)} rows)")


# ----------------------------------------------------------------------------- report

def tier_of(m) -> np.ndarray:
    idx = np.searchsorted(np.asarray(TIER_EDGES), np.nan_to_num(np.asarray(m, float), nan=-1),
                          side="right")
    return np.array([TIERS[i] for i in idx], object)


def clustered_mean(v, cl) -> tuple[float, float]:
    v = np.asarray(v, float)
    cl = np.asarray(cl)
    ok = np.isfinite(v)
    v, cl = v[ok], cl[ok]
    if len(v) < 2:
        return float(v.mean()) if len(v) else np.nan, np.nan
    m = v.mean()
    s = pd.Series(v - m).groupby(cl).sum()
    g = len(s)
    se = np.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / len(v)
    return float(m), float(se)


def md(df: pd.DataFrame, floatfmt: int = 2) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def path_of(r) -> str:
    if r["starter_flag"] is True or r.get("override") is True:
        return "dk_starter_flag"
    s = r["status"]
    if s == "fringe":
        return "fringe"
    if s in ("unlisted", "unlisted_noreport"):
        return "unlisted_regular"
    return "listed"


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"live_frame{'_' + '_'.join(KINDS) if KINDS else ''}.parquet")
    df = df[df["played"].notna()].copy()
    df["played"] = df["played"].astype(int)
    df["path"] = df.apply(path_of, axis=1)
    df["e_min_p"] = df["min_played_mean"]
    df["tier"] = tier_of(df["e_min_p"])
    df["dnp"] = 1 - df["played"]
    df["cl"] = df["game_id"]
    return df


def calib_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    t = pd.DataFrame({"n": g.size(), "mean_p": g["p_play"].mean(),
                      "played_rate": g["played"].mean(),
                      "exp_dnp": g["p_play"].apply(lambda p: (1 - p).sum()),
                      "act_dnp": g["dnp"].sum()})
    t["excess_dnp"] = t["act_dnp"] - t["exp_dnp"]
    return t.reset_index()


def dnp_cause(r) -> str:
    """Why a P >= 0.7 player did not play (one label)."""
    st_run, st_tip = r["status"], r["status_tip"]
    if st_tip == "out" and st_run != "out":
        return "late_scratch_out_on_final_report"
    if r["reason_tip"] == "gleague" or r["reason_class"] == "gleague":
        return "gleague_two_way"
    if st_tip in ("doubtful", "questionable", "daytoday") or st_run in ("doubtful", "questionable",
                                                                         "daytoday"):
        return "listed_uncertain_sat"
    if r["path"] == "fringe":
        return "fringe_not_used"
    if r["dk_status"] in ("OUT", "O"):
        return "dk_out_flag"
    return "coach_decision_unlisted"


def cmd_report(a) -> None:
    df = load_frame()
    P = []
    pr = P.append
    pr(f"Population: {len(df)} graded last-pre-tip rows, {df['game_id'].nunique()} games, "
       f"{df['day'].nunique()} nights; kinds {df['kind'].value_counts().to_dict()}; "
       f"model versions {df['model_version'].value_counts().to_dict()}")
    # ---- P(play) calibration
    df["p_bin"] = pd.cut(df["p_play"], P_BINS, right=False).astype(str)
    pr("\n### P(play) calibration by bin\n")
    pr(md(calib_table(df, "p_bin")))
    hi = df[df["p_play"] >= 0.7].copy()
    pr(f"\nP >= 0.7: n {len(hi)}, expected DNP {(1 - hi['p_play']).sum():.1f}, "
       f"actual {hi['dnp'].sum()}")
    pr("\n### P >= 0.7 by path\n")
    pr(md(calib_table(hi, "path")))
    pr("\n### P >= 0.7 by status at run time\n")
    pr(md(calib_table(hi, "status")))
    pr("\n### All rows by status at run time\n")
    pr(md(calib_table(df, "status")))
    pr("\n### Listed rows by status x hours bucket\n")
    pr(md(calib_table(df[df["path"] == "listed"], ["status", "bucket"])))
    df["h_run"] = pd.cut(df["hours_run_to_tip"], [0, 0.25, 1, 3, 8, 48]).astype(str)
    pr("\n### All rows by hours from run to tip\n")
    pr(md(calib_table(df, ["h_run"])))
    pr("\n### P >= 0.7 by path x tier (E[min|plays])\n")
    pr(md(calib_table(hi, ["path", "tier"])))
    hi["cause"] = hi.apply(dnp_cause, axis=1)
    d = hi[hi["dnp"] == 1]
    pr("\n### P >= 0.7 DNPs by cause\n")
    t = d.groupby("cause").agg(n=("dnp", "size"), mean_p=("p_play", "mean"),
                               mean_emin=("e_min_p", "mean")).reset_index()
    pr(md(t))
    pr("\n### P >= 0.7 by run kind\n")
    pr(md(calib_table(hi, "kind")))
    pr("\n### P >= 0.7 by app_season == 0 (no appearance yet this season) x path\n")
    hi["no_app"] = hi["app_season"] == 0
    pr(md(calib_table(hi, ["no_app", "path"])))
    pr("\n### P >= 0.7 by DK salary band\n")
    hi["sal_band"] = pd.cut(hi["salary"], [0, 3100, 4000, 5500, 20000]).astype(str)
    pr(md(calib_table(hi, ["sal_band"])))
    pr("\n### P >= 0.7 by E[min|plays] band\n")
    hi["emin_band"] = pd.cut(hi["e_min_p"], [0, 5, 10, 15, 20, 60]).astype(str)
    pr(md(calib_table(hi, ["emin_band"])))
    pr("\n### Unlisted regulars, P(play) by last-10 role (avail role) \n")
    ur = df[df["path"] == "unlisted_regular"]
    pr(md(calib_table(ur, ["av_role"])))
    # the projection-relevant population: E[min|plays] >= 10 (DFS-relevant)
    rel = hi[hi["e_min_p"] >= 10]
    pr(f"\nP >= 0.7 and E[min|plays] >= 10: n {len(rel)}, exp DNP {(1 - rel['p_play']).sum():.1f}, "
       f"act {rel['dnp'].sum()}")
    pr("\n### Top P >= 0.7 DNPs (E[min|plays] >= 10)\n")
    cols = ["day", "team", "player_id", "path", "status", "status_tip", "reason_tip", "p_play",
            "e_min_p", "salary", "app_season", "cause"]
    dd = d[d["e_min_p"] >= 10].sort_values("e_min_p", ascending=False)[cols]
    pr(md(dd.head(40)))
    # ---- minutes
    pl = df[(df["played"] == 1) & df["e_min_p"].notna()].copy()
    pl["err"] = pl["e_min_p"] - pl["actual_min"]
    pl["ae"] = pl["err"].abs()
    p15 = pl[pl["e_min_p"] >= 15]
    m, se = clustered_mean(p15["ae"], p15["cl"])
    mb, seb = clustered_mean(p15["err"], p15["cl"])
    pr(f"\n### Minutes (played, E[min|plays] >= 15): MAE {m:.2f} (game-clustered SE {se:.2f}), "
       f"bias {mb:+.2f} ({seb:.2f}), n {len(p15)}; T2.4 live-like slice 4.457 (lock-45)")
    m2, se2 = clustered_mean(pl["ae"], pl["cl"])
    pr(f"All played rows: MAE {m2:.2f} ({se2:.2f}), n {len(pl)}")

    def mtab(sub, by):
        g = sub.groupby(by, dropna=False)
        t = pd.DataFrame({"n": g.size(), "mae": g["ae"].mean(), "bias": g["err"].mean(),
                          "share_abs_err": g["ae"].sum() / sub["ae"].sum()})
        return t.reset_index()

    pr("\n#### by tier (E[min|plays]); played rows\n")
    pr(md(mtab(pl, "tier")))
    pl["ot"] = pl["n_ot"] > 0
    pl["blowout20"] = pl["margin"] >= 20
    pl["margin_band"] = pd.cut(pl["margin"], [0, 5, 10, 15, 20, 30, 80], right=False).astype(str)
    pl["returning"] = False
    pl["early_gp"] = pd.cut(pl["team_gp"], [-1, 0, 1, 2, 10]).astype(str)
    for by in ("ot", "blowout20", "margin_band", "started", "new_team", "no_nba_history",
               "rookie", "early_gp", "kind"):
        pr(f"\n#### played E[min|plays] >= 15 by {by}\n")
        pr(md(mtab(p15, by) if by in p15 else mtab(pl[pl['e_min_p'] >= 15], by)))
    p15 = pl[pl["e_min_p"] >= 15].copy()
    pr("\n#### played >= 15 by blowout x tier x won\n")
    pr(md(mtab(p15, ["blowout20", "team_won", "tier"])))
    pr("\n#### played >= 15 by started x tier\n")
    pr(md(mtab(p15, ["started", "tier"])))
    # role change: started vs projected role
    p15["proj_starter"] = p15["e_min_p"] >= 26
    pr("\n#### played >= 15: projected starter-level (>= 26) vs actually started\n")
    pr(md(mtab(p15, ["proj_starter", "started"])))
    # returning: no appearance this season but history (season-opening absences)
    p15["ret_cat"] = np.where(p15["app_season"] == 0, np.where(p15["no_nba_history"], "debut",
                              "first_game_of_season"), "has_app")
    pr("\n#### played >= 15 by first appearance of the season\n")
    pr(md(mtab(p15, ["ret_cat"])))
    # team had absences: teammates listed out at run time (count of out regulars)
    outs = df[(df["status"] == "out")].groupby(["game_id", "team"]).size()
    p15["team_outs"] = [outs.get((g, t), 0) for g, t in zip(p15["game_id"], p15["team"])]
    p15["team_outs_b"] = pd.cut(p15["team_outs"], [-1, 0, 1, 2, 20]).astype(str)
    pr("\n#### played >= 15 by team players Out on the report (projected rows)\n")
    pr(md(mtab(p15, ["team_outs_b"])))
    # surprise absences: teammates with P >= 0.7 who did not play
    surp = df[(df["p_play"] >= 0.7) & (df["dnp"] == 1) & (df["e_min_p"] >= 15)] \
        .groupby(["game_id", "team"]).size()
    p15["team_surprise"] = [surp.get((g, t), 0) for g, t in zip(p15["game_id"], p15["team"])]
    pr("\n#### played >= 15 by surprise teammate DNPs (P >= 0.7, E[min|plays] >= 15)\n")
    pr(md(mtab(p15, ["team_surprise"])))
    # big misses
    pr("\n#### 25 biggest minutes misses (played, E[min|plays] >= 10)\n")
    bm = pl[pl["e_min_p"] >= 10].sort_values("ae", ascending=False).head(25)
    pr(md(bm[["day", "team", "player_id", "tier", "e_min_p", "actual_min", "err", "started",
              "margin", "n_ot", "app_season", "new_team", "status"]]))
    # a simple attribution: MAE reduction if each factor's rows were perfect, and the
    # regression of signed error on indicators
    pr("\n#### Linear decomposition of the signed error (played >= 15), OLS with game-clustered SE\n")
    X = pd.DataFrame({
        "one": 1.0,
        "ot_x_starter": (p15["n_ot"] > 0) * (p15["e_min_p"] >= 26),
        "ot_x_other": (p15["n_ot"] > 0) * (p15["e_min_p"] < 26),
        "blowout_x_starter": p15["blowout20"] * (p15["e_min_p"] >= 26),
        "blowout_x_other": p15["blowout20"] * (p15["e_min_p"] < 26),
        "bench_started": (~p15["proj_starter"]) & p15["started"],
        "starter_benched": p15["proj_starter"] & (~p15["started"]) & p15["starts_known"],
        "first_game_season": p15["app_season"] == 0,
        "new_team": p15["new_team"],
        "team_surprise_dnp": p15["team_surprise"] > 0,
    }).astype(float)
    y = p15["err"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
    res = y - X.to_numpy() @ beta
    XtX_inv = np.linalg.pinv(X.to_numpy().T @ X.to_numpy())
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in p15.reset_index(drop=True).groupby("cl").indices.items():
        s = X.to_numpy()[idx].T @ res[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    tab = pd.DataFrame({"term": X.columns, "coef_min": beta, "se": np.sqrt(np.diag(V)),
                        "n_rows": X.sum().to_numpy()})
    pr(md(tab))
    # ---- DK decomposition
    u = df[(df["proj_min"] >= 10) & df["actual_dk"].notna() & df["dk_all_mean"].notna()].copy()
    rate = (u["dk_played_mean"] / u["min_played_mean"]).replace([np.inf, -np.inf], np.nan)
    u["pred_all"] = u["dk_all_mean"]
    u["pred_pknown"] = np.where(u["played"] == 1, u["dk_played_mean"], 0.0)
    u["pred_pmknown"] = np.where(u["played"] == 1, rate * u["actual_min"], 0.0)
    y = u["actual_dk"].to_numpy(float)
    rows = []
    for name, col in (("live (as projected)", "pred_all"), ("+ P(play) known", "pred_pknown"),
                      ("+ minutes known (x projected DK/min)", "pred_pmknown")):
        e = np.abs(u[col].to_numpy(float) - y)
        mm, ss = clustered_mean(e, u["cl"])
        rows.append((name, len(u), mm, ss))
    pr("\n### DK MAE ladder (proj_min >= 10, DNP = 0; the grader population)\n")
    pr(md(pd.DataFrame(rows, columns=["prediction", "n", "dk_mae", "se"])))
    plu = u[u["played"] == 1].copy()
    r_ = rate[u["played"] == 1]
    tot = plu["dk_played_mean"] - plu["actual_dk"]
    minpart = r_ * (plu["min_played_mean"] - plu["actual_min"])
    ratepart = tot - minpart
    pr(f"\nPlayed rows (n {len(plu)}): Var(total err) {tot.var():.1f}, Var(minutes part) "
       f"{minpart.var():.1f}, Var(rate part) {ratepart.var():.1f}, 2Cov {2 * np.cov(minpart, ratepart)[0, 1]:.1f}; "
       f"minutes share of variance {minpart.var() / tot.var():.2f}")
    pr(f"Mean |err| {tot.abs().mean():.2f}; mean |rate part| {ratepart.abs().mean():.2f}; "
       f"mean |minutes part| {minpart.abs().mean():.2f}; bias total {tot.mean():+.2f}, "
       f"minutes {minpart.mean():+.2f}, rate {ratepart.mean():+.2f}")
    plu["tier"] = tier_of(plu["min_played_mean"])
    plu["tot"], plu["minpart"], plu["ratepart"] = tot, minpart, ratepart
    t = plu.groupby("tier").agg(n=("tot", "size"), bias_total=("tot", "mean"),
                                bias_min=("minpart", "mean"), bias_rate=("ratepart", "mean"),
                                mae_total=("tot", lambda v: v.abs().mean()),
                                mae_rate_only=("ratepart", lambda v: v.abs().mean()))
    pr("\n#### by tier (played)\n")
    pr(md(t.reset_index()))
    text = "\n".join(P)
    print(text)
    (OUT / f"live_diag{'_' + '_'.join(KINDS) if KINDS else ''}.md").write_text(text + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("frame", "report"))
    ap.add_argument("--db", default=None, help="DB path (opened read-only)")
    ap.add_argument("--kinds", default="", help="comma-separated run kinds (default: all)")
    a = ap.parse_args(argv)
    global KINDS
    KINDS = tuple(k for k in a.kinds.split(",") if k)
    {"frame": cmd_frame, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
