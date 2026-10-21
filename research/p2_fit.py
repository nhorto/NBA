"""Fit and check the P(play) v2 fringe model (`availability.FringeParams`).

Fringe = a live-path candidate forced in from the DK pool (roster == 0: not in the team's last
10 games) with no listed report status (`pipeline.pplay_v2_fix`; rows from
`research/p2_rows.py`).

* "mid" on the T2.4 slice rows at lock-45 (rows_slice_live45.parquet); honest numbers are
  leave-one-month-out.
* "early" / "early_nosal" on the RotoGuru DK pools 2017-18 .. 2020-21 (rows_rg_*.parquet; the
  two-way-contract era, EARLY_SEASONS): the
  season's first EARLY_GP team games, players without an appearance yet; honest numbers are
  leave-one-season-out. There are no injury reports for those seasons, so a fringe row there
  may be a player the live report would list (and the live path would then not treat as
  fringe). Each RotoGuru row is weighted by P(not listed | outcome, pattern, role) measured on
  the slice's forced (roster 0) rows, with pattern = DNP now and in none of the team's next 3
  games (the future is used only for this population weight, never as a feature) and role =
  avg10 >= 15: played 0.55-0.89 / DNP returning soon 0.24-0.56 / DNP streak 0.16-0.41; rows at
  the DK minimum ($3,000) stay unweighted. Unweighted, the $4.5k+ players come out at 0.67-0.78
  vs 7 of 8 played at the 2026-10-20 opener; weighted min-salary players at 0.33 vs 2 of 9.

    python research/p2_fit.py ROWS_DIR [--save]     # prints the tables; --save writes the JSON
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import availability as AV

LISTED = {"out", "doubtful", "questionable", "probable", "available", "removed", "daytoday"}
SAL_BINS = [0, 3000, 3400, 3900, 4500, 5500, 6500, 13000]
# two-way contracts (2017-18+) put 2-3 near-minimum players per team in the DK pool who rarely
# play early: min-salary fringe played 0.24-0.38 in 2015-17 vs 0.16-0.22 after (opener 2026: 2/9)
EARLY_SEASONS = ("2017-18", "2018-19", "2019-20", "2020-21")


def ll(y, p, w=None) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = np.ones(len(y)) if w is None else w
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum())


def fringe(df: pd.DataFrame) -> pd.DataFrame:
    # = pipeline.pplay_v2_fix: forced (roster 0) and no listed report status
    return df[(df["roster"] == 0) & ~df["status"].isin(LISTED)].copy()


def next_played(conn, rows: pd.DataFrame, seasons: list[str], k: int = 3) -> np.ndarray:
    """Did the player appear in any of his team's next k games (population weight only)."""
    ph = ",".join("?" * len(seasons))
    tg = pd.read_sql_query(f"SELECT game_id, team FROM team_games WHERE season IN ({ph}) "
                           "ORDER BY game_date, game_id", conn, params=seasons)
    pg = conn.execute(f"SELECT game_id, player_id FROM player_games WHERE season IN ({ph}) "
                      "AND min>0", seasons).fetchall()
    played = {(g, int(p)) for g, p in pg}
    sched = {t: list(g["game_id"]) for t, g in tg.groupby("team")}
    pos = {t: {gid: i for i, gid in enumerate(v)} for t, v in sched.items()}
    out = np.zeros(len(rows), bool)
    for j, (t, g, p) in enumerate(zip(rows["team"], rows["game_id"], rows["player_id"])):
        i = pos.get(t, {}).get(g)
        if i is not None:
            out[j] = any((x, int(p)) in played for x in sched[t][i + 1:i + 1 + k])
    return out


def _cell(played, nxt, avg10) -> list[tuple]:
    return [("played" if pl else ("dnp_soon" if nx else "dnp_streak"),
             "rot" if a >= 15 else "bench") for pl, nx, a in zip(played, nxt, avg10)]


def listing_weights(S: pd.DataFrame) -> dict:
    """P(not listed | cell) over the slice's forced (roster 0) rows."""
    x = S[S["roster"] == 0]
    cells = _cell(x["played"], x["nxt"], x["avg10"])
    nl = ~x["status"].isin(LISTED).to_numpy()
    out = {}
    for c in sorted(set(cells)):
        m = np.array([cc == c for cc in cells])
        out["|".join(c)] = {"n": int(m.sum()), "not_listed": float(nl[m].mean())}
    return out


def cv(F: pd.DataFrame, regime: str, fold: str, weight: str | None = None) -> np.ndarray:
    out = np.zeros(len(F))
    for k in F[fold].unique():
        te = (F[fold] == k).to_numpy()
        m = AV.fit_fringe_model(F[~te], regime, weight=weight)
        out[te] = AV.FringeParams(models={regime: m}).p(F[te].assign(regime=regime))
    return out


def table(F: pd.DataFrame, p: np.ndarray, col: str, bins, w: str | None = None) -> str:
    x = F.assign(p=p, w=F[w] if w else 1.0)
    x["pw"], x["yw"] = x["p"] * x["w"], x["played"] * x["w"]
    g = x.groupby(pd.cut(x[col], bins), observed=True)
    t = pd.DataFrame({"n": g.size(), "p": g["pw"].sum() / g["w"].sum(),
                      "played": g["yw"].sum() / g["w"].sum()})
    return t.round(3).T.to_string()


def report(name: str, F: pd.DataFrame, p: np.ndarray, w: str | None = None, **extra) -> dict:
    y = F["played"].to_numpy(float)
    ww = F[w].to_numpy(float) if w else np.ones(len(F))
    ybar = float(np.sum(ww * y) / ww.sum())
    r = {"n": int(len(F)), "played": ybar, "p": float(np.sum(ww * p) / ww.sum()),
         "ll": ll(y, p, ww), "ll_const": ll(y, np.full(len(y), ybar), ww),
         "brier": float(np.sum(ww * (p - y) ** 2) / ww.sum()),
         "brier_const": float(np.sum(ww * (ybar - y) ** 2) / ww.sum()), **extra}
    print(f"\n== {name}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                        for k, v in r.items()))
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.rows)
    conn = db.connect()
    out = {}
    # ---- mid: slice lock-45
    S = pd.read_parquet(d / "rows_slice_live45.parquet")
    S["nxt"] = next_played(conn, S, ["2025-26"])
    wts = listing_weights(S)
    print("P(not listed | cell), slice forced rows:", json.dumps(wts))
    F = fringe(S)
    F = F[F["regime"] == "mid"].copy()
    F["month"] = F["date"].str[:7]
    p_cv = cv(F, "mid", "month")
    out["mid"] = report("mid (slice lock-45, leave-one-month-out)", F, p_cv,
                        engine_pp=float(F["pp_engine"].mean()))
    print(table(F, p_cv, "avg10", [-1, 5, 10, 15, 20, 25, 45]))
    print(table(F, p_cv, "app_season", [-1, 0, 3, 10, 25, 90]))
    models = {"mid": AV.fit_fringe_model(F, "mid")}
    # ---- early: RotoGuru pools, weighted to the not-listed population
    R = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(d / "rows_rg_*.parquet")))],
                  ignore_index=True)
    E = R[(R["roster"] == 0) & R["in_pool"].astype(bool) & (R["regime"] == "early")
          & R["season"].isin(EARLY_SEASONS)].copy()
    E["nxt"] = next_played(conn, E, sorted(E["season"].unique()))
    E["w"] = [wts["|".join(c)]["not_listed"] for c in _cell(E["played"], E["nxt"], E["avg10"])]
    # min-salary rows stay unweighted: weighted they come out at 0.33 vs 2 of 9 (both 1-minute
    # cameos) at the 2026-10-20 opener; above the minimum the weights move $4.5k+ players from
    # 0.67-0.78 to 0.86-0.91 (opener: 7 of 8 played)
    E.loc[E["salary"] <= 3000, "w"] = 1.0
    for reg in ("early", "early_nosal"):
        p_cv = cv(E, reg, "season", weight="w")
        out[reg] = report(f"{reg} (RotoGuru 2017-21, first {AV.EARLY_GP} team games, weighted, "
                          "leave-one-season-out)", E, p_cv, w="w")
        print(table(E, p_cv, "salary", SAL_BINS, w="w"))
        print(table(E, p_cv, "avg10", [-1, 5, 10, 15, 20, 25, 45], w="w"))
        print(table(E, p_cv, "app365", [-1, 5, 20, 40, 60, 110], w="w"))
        models[reg] = AV.fit_fringe_model(E, reg, weight="w")
    p_unw = cv(E, "early", "season")
    out["early_unweighted"] = report("early, unweighted (for reference)", E, p_unw)
    print(table(E, p_unw, "salary", SAL_BINS))
    for k, m in models.items():
        print(k, dict(zip(m["features"], np.round(m["coef"], 3))))
    fp = AV.FringeParams(models=models, meta={
        "mid": "T2.4 slice 2025-12-22..2026-04-12, lock-45 live path, fringe rows "
               f"n={len(F)}",
        "early": f"RotoGuru DK pools {EARLY_SEASONS[0]}..{EARLY_SEASONS[-1]}, team games < {AV.EARLY_GP}, no "
                 f"appearance this season, n={len(E)}, weighted by P(not listed | cell) "
                 "from the slice's forced rows",
        "listing_weights": wts, "cv": out, "script": "research/p2_fit.py"})
    if a.save:
        print("wrote", fp.save())
    print(json.dumps(fp.to_json()["models"]))


if __name__ == "__main__":
    sys.exit(main())
