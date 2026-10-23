"""Reconciliation reserve (IMPROVEMENTS.md B2, switch COURTSIDE_RECON_RESERVE, tag `+rr`).

Docs: docs/research/recon-reserve.md.  Code under test: `reconcile.RECON_RESERVE`,
`reconcile.reserve_target`, the two reconcile call sites in `copula.py`.

    python research/recon_reserve.py rates            # bench points / minute by season, league vs
                                                      # team, debutants vs others (read-only DB)
    python research/recon_reserve.py fit              # write src/courtside/proj/recon_reserve_params.json
    python research/recon_reserve.py harness S PROTO  # one harness arm (minutes_fix_eval harness +
                                                      # per-team-game reserve / team sums)
    python research/recon_reserve.py dk OFF ON LABEL  # paired ON - OFF DK metrics + team-sum bias

OFF identity: research/p2_identity.py under the HEAD tree and this tree (see the write-up).

"Outside the candidate set" is approximated from the box scores the way `pregame_panel` builds
candidates: a team-game row (min > 0) is a reserve row when the player has no min > 0 row for
that team in the team's previous 10 games (any season type, across seasons).  The extras the live
path adds (report / DK pool players with box history) are not modelled, so this slightly
over-counts reserve rows; the rate (points per minute) is what is used, not the count.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config

ROOT = config.REPO_ROOT
OUT = Path(os.environ.get("RR_OUT", ROOT / "var" / "tmp" / "rr" / "out"))
WINDOW = 10
FIRST_FIT = "2012-13"          # earliest season pooled into a rate
POOL = 3                       # seasons < S pooled (minutes-weighted)
TARGETS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
           "2026-27")


def say(m: str) -> None:
    print(f"[rr] {time.strftime('%H:%M:%S')} {m}", flush=True)


def ro_conn(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.db_path()
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


# ============================================================================== rates

def reserve_rows(conn) -> pd.DataFrame:
    """Every box row (min > 0) with `res` = outside the team's last-10-game candidates and `deb` =
    no earlier box row at all (the debut reserve's population)."""
    g = pd.read_sql_query(
        "SELECT game_id, player_id, team, season, season_type, game_date, min, pts "
        "FROM player_games WHERE min > 0", conn)
    tg = pd.read_sql_query("SELECT DISTINCT game_id, team, game_date FROM team_games", conn)
    tg = tg.sort_values(["team", "game_date", "game_id"], kind="mergesort")
    tg["tgi"] = tg.groupby("team").cumcount()
    g = g.merge(tg[["game_id", "team", "tgi"]], on=["game_id", "team"], how="inner")
    g = g.sort_values(["player_id", "team", "tgi"], kind="mergesort").reset_index(drop=True)
    prev = g.groupby(["player_id", "team"])["tgi"].shift(1)
    g["res"] = ~((g["tgi"] - prev) <= WINDOW)
    g = g.sort_values(["player_id", "game_date", "game_id"], kind="mergesort")
    g["deb"] = (g.groupby("player_id").cumcount() == 0) & g["res"]
    return g.reset_index(drop=True)


def season_rates(g: pd.DataFrame) -> pd.DataFrame:
    rs = g[g["season_type"] == "Regular Season"]
    rows = []
    for s, x in rs.groupby("season"):
        r = x[x["res"]]
        d = r[r["deb"]]
        o = r[~r["deb"]]
        ntg = x.groupby(["game_id", "team"]).ngroups
        rows.append({"season": s, "team_games": ntg, "res_min_tg": r["min"].sum() / ntg,
                     "ppm_res": r["pts"].sum() / r["min"].sum(),
                     "ppm_deb": d["pts"].sum() / max(d["min"].sum(), 1e-9),
                     "ppm_oth": o["pts"].sum() / max(o["min"].sum(), 1e-9),
                     "ppm_all": x["pts"].sum() / x["min"].sum(),
                     "deb_share": d["min"].sum() / max(r["min"].sum(), 1e-9)})
    return pd.DataFrame(rows)


def pooled_rate(g: pd.DataFrame, season: str, pool: int = POOL) -> dict:
    """League reserve points / minute pooled over the `pool` seasons < `season`."""
    rs = g[(g["season_type"] == "Regular Season") & g["res"] & (g["season"] < season)
           & (g["season"] >= FIRST_FIT)]
    ss = sorted(rs["season"].unique())[-pool:]
    x = rs[rs["season"].isin(ss)]
    return {"ppm": float(x["pts"].sum() / x["min"].sum()), "train_seasons": ss,
            "minutes": float(x["min"].sum())}


def team_vs_league(g: pd.DataFrame, season: str) -> dict:
    """Out-of-sample check of a team bench rate (team's reserve ppm over the pooled seasons,
    shrunk) vs the league rate: minutes-weighted squared error on season `season`'s team
    reserve ppm (each team-season's reserve points vs the rate x its reserve minutes)."""
    rs = g[(g["season_type"] == "Regular Season") & g["res"]]
    tr = rs[(rs["season"] < season) & (rs["season"] >= FIRST_FIT)]
    ss = sorted(tr["season"].unique())[-POOL:]
    tr = tr[tr["season"].isin(ss)]
    te = rs[rs["season"] == season]
    if not len(te):
        return {}
    lg = tr["pts"].sum() / tr["min"].sum()
    t = tr.groupby("team").agg(p=("pts", "sum"), m=("min", "sum"))
    e = te.groupby("team").agg(p=("pts", "sum"), m=("min", "sum"))
    out = {}
    for k in (0.0, 300.0, 1000.0, 3000.0, np.inf):
        rate = (t["p"] + k * lg) / (t["m"] + k) if np.isfinite(k) else pd.Series(lg, t.index)
        r = rate.reindex(e.index).fillna(lg)
        out[f"k={k:g}"] = float(((e["p"] - r * e["m"]) ** 2).sum() / e["m"].sum())
    return out


def cmd_rates() -> None:
    g = reserve_rows(ro_conn())
    sr = season_rates(g)
    print(sr[sr["season"] >= "2010-11"].round(4).to_string(index=False))
    print("\nteam vs league (MSE of team-season reserve points per reserve minute; k = pseudo "
          "minutes of league prior, inf = league rate)")
    for s in TARGETS[:-1]:
        print(s, {k: round(v, 3) for k, v in team_vs_league(g, s).items()})
    rs = g[(g["season_type"] == "Regular Season") & g["res"]]
    rs = rs.assign(tgn=rs.merge(_team_game_no(g), on=["game_id", "team"], how="left")["tgn"]
                   .to_numpy())
    op = rs[rs["tgn"] < 5]
    print("\nopeners (first 5 team games): ppm", round(op["pts"].sum() / op["min"].sum(), 4),
          "all:", round(rs["pts"].sum() / rs["min"].sum(), 4))


def _team_game_no(g: pd.DataFrame) -> pd.DataFrame:
    rs = g[g["season_type"] == "Regular Season"][["game_id", "team", "season", "game_date"]]
    tg = rs.drop_duplicates(["game_id", "team"]).sort_values("game_date", kind="mergesort")
    tg["tgn"] = tg.groupby(["team", "season"]).cumcount()
    return tg[["game_id", "team", "tgn"]]


def cmd_fit() -> None:
    from courtside.proj import reconcile as RC
    g = reserve_rows(ro_conn())
    out = {"pool": POOL, "window": WINDOW, "rates": {}}
    for s in TARGETS:
        r = pooled_rate(g, s)
        out["rates"][s] = {"ppm": round(r["ppm"], 5), "train_seasons": r["train_seasons"],
                           "minutes": round(r["minutes"], 1)}
        say(f"{s}: {r['ppm']:.4f} pts/min on {r['train_seasons']} ({r['minutes']:.0f} min)")
    RC.RR_PATH.write_text(json.dumps(out, indent=1) + "\n")
    say(f"wrote {RC.RR_PATH}")


# ============================================================================== harness

def _patch_teams() -> None:
    """Capture each harness date's per-team-game reserve (240 + 25 n_OT - sum of the
    candidates' sim minutes, mean over sims), target and recon scale -> OUT/teams_*.parquet."""
    from courtside.proj import backtest as B
    from courtside.proj import minutes as M
    from courtside.proj import pipeline as P
    rows: list = []
    ph, ll = P.project_historical, B.live_like_day

    from courtside.proj.copula import OT_BAND

    def grab(res, day):
        for g in res.sims:
            n_ot = (np.abs(g.margin) < OT_BAND).astype(float)
            for t in np.unique(g.team):
                sel = g.team == t
                x = g.minutes[:, sel].astype(float).sum(1)
                info = g.teams.get(t, {})
                rows.append({"date": str(day), "game_id": g.game_id, "team": t,
                             "res_e": float((M.TEAM_MIN + M.OT_TEAM_MIN * n_ot - x).mean()),
                             "target": info.get("target"), "scale": info.get("scale"),
                             "baseline": info.get("baseline"),
                             "res_pts": info.get("reserve_pts", 0.0)})

    def ph2(*a, **k):
        r = ph(*a, **k)
        grab(r, getattr(r, "day", None))
        return r

    def ll2(*a, **k):
        r = ll(*a, **k)
        grab(r, getattr(r, "day", None))
        return r

    P.project_historical, B.live_like_day = ph2, ll2
    return rows


def cmd_harness(season: str, protocol: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import minutes_fix_eval as MF
    from courtside.proj import pipeline as P
    MF.OUT = OUT
    rows = _patch_teams()
    MF.cmd_harness(season, protocol)
    f = OUT / f"teams_{protocol}_{season}_{P.MODEL_VERSION}.parquet"
    pd.DataFrame(rows).to_parquet(f)
    say(f"{len(rows)} team-games -> {f}")


# ============================================================================== DK + team sums

def team_sums(rows: pd.DataFrame, teams: pd.DataFrame | None, pg: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: projected players' DK mean sum and actual DK sum (DNP = 0), their actual
    points, the actual points of everybody else, and (from the teams capture) the reconcile
    target (= the projected players' E[points]) and E[reserve minutes]."""
    r = rows.dropna(subset=["player_id"]).drop_duplicates(["date", "player_id"])
    r = r[r["has_proj"].astype(bool)].copy()
    r["player_id"] = r["player_id"].astype(np.int64)
    r = r.merge(pg[["game_id", "player_id", "pts", "min"]], on=["game_id", "player_id"],
                how="left")
    s = r.groupby(["date", "game_id", "team"]).agg(proj=("mean", "sum"), act=("y", "sum"),
                                                   pts_proj_act=("pts", "sum"),
                                                   min_proj_act=("min", "sum")).reset_index()
    tp = pg.groupby(["game_id", "team"])[["pts", "min"]].sum().add_suffix("_team").reset_index()
    s = s.merge(tp, on=["game_id", "team"], how="left")
    s["pts_other"] = s["pts_team"] - s["pts_proj_act"]
    s["min_other"] = s["min_team"] - s["min_proj_act"]
    if teams is not None and len(teams):
        t = teams.drop_duplicates(["game_id", "team"])[["game_id", "team", "res_e", "target"]]
        s = s.merge(t, on=["game_id", "team"], how="left")
    return s


def _load_teams(rows_path: str) -> pd.DataFrame | None:
    p = Path(rows_path.replace("rows_", "teams_"))
    return pd.read_parquet(p) if p.exists() else None


def cmd_dk(off: str, on: str, label: str) -> None:
    """minutes_fix_eval's paired DK table + team sums: projected-player DK sum - actual and
    projected-player E[points] (= the reconcile target) - their actual points, overall, at
    openers (team game < 5) and on high-reserve team-games (E[reserve] >= its 90th pct)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import minutes_fix_eval as MF
    MF.OUT = OUT
    MF.cmd_dk(off, on, label)
    conn = MF._conn()
    a, b = pd.read_parquet(off), pd.read_parquet(on)
    gids = set(a["game_id"].dropna())
    pg = pd.read_sql_query("SELECT game_id, player_id, team, pts, min FROM player_games", conn)
    pg = pg[pg["game_id"].isin(gids)].astype({"player_id": np.int64})
    sa, sb = team_sums(a, _load_teams(off), pg), team_sums(b, _load_teams(on), pg)
    keep = ["date", "game_id", "team", "proj"] + (["target"] if "target" in sb else [])
    d = sa.merge(sb[keep], on=["date", "game_id", "team"], suffixes=("_off", "_on"))
    tg = pd.read_sql_query("SELECT game_id, team, game_date FROM team_games WHERE "
                           "season_type = 'Regular Season'", conn).sort_values("game_date")
    tg["tgn"] = tg.groupby([tg["team"], tg["game_id"].str[3:5]]).cumcount()
    d = d.merge(tg[["game_id", "team", "tgn"]], on=["game_id", "team"], how="left")
    sets = {"all": d, "openers (team game < 5)": d[d["tgn"] < 5]}
    if "res_e" in d and d["res_e"].notna().any():
        q = d["res_e"].quantile(0.9)
        sets[f"high reserve (E[res] >= {q:.1f})"] = d[d["res_e"] >= q]
    pairs = {"dk": ("proj_off", "proj_on", "act")}
    if "target_off" in d and "target_on" in d:
        pairs["pts"] = ("target_off", "target_on", "pts_proj_act")
    out = {}
    print(f"\n## team sums {label} (projected players: projection - actual; SE by date)")
    for k, x in sets.items():
        if len(x) < 5:
            continue
        cl = x["date"].to_numpy()
        row = {"n": int(len(x)), "res_e": float(x["res_e"].mean()) if "res_e" in x else None,
               "pts_other": float(x["pts_other"].mean()),
               "min_other": float(x["min_other"].mean())}
        for nm, (c0, c1, ca) in pairs.items():
            b0, se0 = MF.cl_mean((x[c0] - x[ca]).to_numpy(float), cl)
            b1, se1 = MF.cl_mean((x[c1] - x[ca]).to_numpy(float), cl)
            row[nm] = {"off": b0, "se_off": se0, "on": b1, "se_on": se1,
                       "abs_off": float((x[c0] - x[ca]).abs().mean()),
                       "abs_on": float((x[c1] - x[ca]).abs().mean())}
        out[k] = row
        txt = "  ".join(f"{nm} {v['off']:+.2f}({v['se_off']:.2f}) -> {v['on']:+.2f}({v['se_on']:.2f})"
                        for nm, v in row.items() if isinstance(v, dict))
        print(f"  {k:28s} n={len(x):5d}  E[res] {row['res_e'] or float('nan'):5.2f}  "
              f"others' min {row['min_other']:5.2f} pts {row['pts_other']:5.2f}  {txt}")
    (OUT / f"teamsum_{label}.json").write_text(json.dumps(out, indent=1, default=float))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rates")
    sub.add_parser("fit")
    h = sub.add_parser("harness")
    h.add_argument("season")
    h.add_argument("protocol", choices=("avail", "live", "live10"))
    d = sub.add_parser("dk")
    d.add_argument("off")
    d.add_argument("on")
    d.add_argument("label")
    x = ap.parse_args(argv)
    if x.cmd == "rates":
        cmd_rates()
    elif x.cmd == "fit":
        cmd_fit()
    elif x.cmd == "harness":
        cmd_harness(x.season, x.protocol)
    else:
        cmd_dk(x.off, x.on, x.label)


if __name__ == "__main__":
    main()
