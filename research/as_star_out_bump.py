"""G4 (player side): is the projection's teammate "star out" bump too big?  Measure only.

Uses EXISTING per-player projection frames (rb/out rows_* from the Mac, arm pathA-v0.3+mf+p2+rb)
copied to var/tmp/as/player/, plus box scores from the local DB (read-only) for the star
definition (T2.7's `dfs_correlations_v2.star_out_frame` rule, re-implemented to return the
star ids): stars = a team's top 2 by prior rolling-20 DK mean among regulars (>= 7 of the last
10 team games or one of the last 3) with prior rolling-20 minutes >= 28 (corr.STAR_MIN).

Star-out team-game: >= 1 star did not play AND the projection knew it (harness `avail`:
p_avail == 0 or no row; live/live10: p_avail <= 0.25, i.e. listed out/doubtful, or no row).
Unlisted surprise star DNPs are dropped from both groups. Star-in: both stars played.

Rows: teammates the projection has available (p_avail >= 0.5), absent stars excluded; DNPs
kept as 0 DK / 0 min (the projection mean includes P(play)). For each star-out row the baseline
is the same player-team-season's mean over star-in games (all, or within +-30 days):
  projected gain = mean - base(mean),   realised gain = y - base(y)   (and exp_min / minutes)
ratio = realised / projected = 1 + (residual diff) / projected.  SEs: cluster bootstrap over
team-seasons (resamples star-out rows and their baselines together), 400 reps.
Tiers: the player's modal projection tier (frame `tier`, copula role tiers) in star-in games;
top-6 = rank <= 6 by prior rolling-20 mpg among the team-game's available rows.

Run:  eval "$(mise env -s bash)"; TMPDIR=~/Documents/NBA/var/tmp uv run python research/as_star_out_bump.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from courtside.proj import corr as K  # noqa: E402

DIR = ROOT / "var" / "tmp" / "as" / "player"
ARM = "pathA-v0.3+mf+p2+rb"
SLICES = {"2023-24": f"rows_avail_2023-24_{ARM}.parquet",
          "2024-25": f"rows_avail_2024-25_{ARM}.parquet",
          "live": f"rows_live_2025-26_{ARM}.parquet",
          "live10": f"rows_live10_2025-26_{ARM}.parquet"}
NB = 400
WINDOW = 30


def stars_frame(conn, seasons):
    """(game_id, team, star_id, played) for each team-game's 2 stars, and per-row prior mpg."""
    s0 = int(seasons[0][:4])
    sl = [K.season_label(s0 - 1)] + list(seasons)
    q = ",".join("?" * len(sl))
    pg = pd.read_sql_query(
        f"""SELECT game_id, player_id, team, season, game_date AS date, min, pts, fg3m, reb, ast,
                   stl, blk, tov FROM player_games WHERE season_type='Regular Season'
            AND min>0 AND season IN ({q})""", conn, params=sl)
    pg["dk"] = K.dk_points(pg)
    pg = pg.sort_values(["player_id", "date", "game_id"])
    g = pg.groupby("player_id", sort=False)
    pg["rm"] = g["min"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    pg["rd"] = g["dk"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    stars, mpg = [], []
    for team, tp in pg.groupby("team"):
        games = tp[["game_id", "date"]].drop_duplicates().sort_values(["date", "game_id"])
        gid = games["game_id"].to_numpy()
        gidx = {g_: i for i, g_ in enumerate(gid)}
        pids = np.sort(tp["player_id"].unique())
        pidx = {p: i for i, p in enumerate(pids)}
        G, P_ = len(gid), len(pids)
        played = np.zeros((G, P_), bool)
        rm = np.full((G, P_), np.nan)
        rd = np.full((G, P_), np.nan)
        gi = tp["game_id"].map(gidx).to_numpy()
        pi = tp["player_id"].map(pidx).to_numpy()
        played[gi, pi] = True
        rm[gi, pi] = tp["rm"].to_numpy()
        rd[gi, pi] = tp["rd"].to_numpy()
        rm = pd.DataFrame(rm).ffill().shift(1).to_numpy()
        rd = pd.DataFrame(rd).ffill().shift(1).to_numpy()
        pl = pd.DataFrame(played.astype(float))
        l10 = pl.rolling(10, min_periods=1).sum().shift(1).to_numpy()
        l3 = pl.rolling(3, min_periods=1).sum().shift(1).to_numpy()
        elig = ((l10 >= 7) | (l3 >= 1)) & (rm >= K.STAR_MIN)
        score = np.where(elig, rd, -np.inf)
        score = np.where(np.isnan(score), -np.inf, score)
        top = np.argsort(-score, axis=1)[:, :2]
        ar = np.arange(G)
        for j in range(2):
            ok = np.isfinite(score[ar, top[:, j]])
            stars.append(pd.DataFrame({"game_id": gid[ok], "team": team,
                                       "star_id": pids[top[ok, j]],
                                       "star_played": played[ar[ok], top[ok, j]]}))
        m = pd.DataFrame(rm, columns=pids)
        m["game_id"] = gid
        m = m.melt(id_vars="game_id", var_name="player_id", value_name="mpg_prior").dropna()
        m["team"] = team
        mpg.append(m)
    mp = pd.concat(mpg, ignore_index=True)
    mp["player_id"] = mp["player_id"].astype(np.int64)
    return pd.concat(stars, ignore_index=True), mp


def build(d: pd.DataFrame, st: pd.DataFrame, mp: pd.DataFrame, live: bool) -> pd.DataFrame:
    d = d[d["has_proj"]].copy()
    tg = d[["game_id", "team"]].drop_duplicates()
    s = st.merge(tg, on=["game_id", "team"])
    s = s.merge(d[["game_id", "team", "player_id", "p_avail"]].rename(
        columns={"player_id": "star_id", "p_avail": "star_pa"}), on=["game_id", "team", "star_id"],
        how="left")
    known = s["star_pa"].isna() | (s["star_pa"] <= (0.25 if live else 0.0))
    s["out_known"] = ~s["star_played"] & known
    s["out_surprise"] = ~s["star_played"] & ~known
    g = s.groupby(["game_id", "team"]).agg(n_out=("out_known", "sum"),
                                           n_sur=("out_surprise", "sum")).reset_index()
    d = d.merge(g, on=["game_id", "team"], how="inner")
    d = d[d["n_sur"] == 0]
    absent = s.loc[~s["star_played"], ["game_id", "team", "star_id"]].rename(
        columns={"star_id": "player_id"})
    absent["_abs"] = True
    d = d.merge(absent, on=["game_id", "team", "player_id"], how="left")
    d = d[d["_abs"].isna() & (d["p_avail"] >= 0.5)].drop(columns="_abs")
    d = d.merge(mp, on=["game_id", "team", "player_id"], how="left")
    d["mpg_prior"] = d["mpg_prior"].fillna(0.0)
    d["top6"] = d.groupby(["game_id", "team"])["mpg_prior"].rank(ascending=False,
                                                                  method="first") <= 6
    d["so"] = d["n_out"] >= 1
    d["minutes"] = d["minutes"].fillna(0.0)
    d["pts_key"] = d["player_id"].astype(str) + "|" + d["team"] + "|" + d["season"]
    d["ts"] = d["team"] + "|" + d["season"]
    d["dt"] = pd.to_datetime(d["date"])
    base_tier = (d[~d["so"]].groupby("pts_key")["tier"]
                 .agg(lambda x: x.value_counts().index[0]).rename("tier0"))
    return d.join(base_tier, on="pts_key")


COLS = ("mean", "y", "exp_min", "minutes")


def gains(d: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """One row per star-out player-game: gain of each COL over the same player-team-season's
    star-in mean (all star-in games, or those within +-window days)."""
    so = d[d["so"]]
    si = d[~d["so"]]
    if window is None:
        b = si.groupby("pts_key")[list(COLS)].mean().add_suffix("_b")
        x = so.join(b, on="pts_key", how="inner")
    else:
        m = so[["pts_key", "game_id", "dt"]].merge(
            si[["pts_key", "dt", *COLS]], on="pts_key", suffixes=("", "_i"))
        m = m[(m["dt_i"] - m["dt"]).abs() <= pd.Timedelta(days=window)]
        b = m.groupby(["pts_key", "game_id"])[list(COLS)].mean().add_suffix("_b").reset_index()
        x = so.merge(b, on=["pts_key", "game_id"], how="inner")
    for c in COLS:
        x[f"g_{c}"] = x[c] - x[f"{c}_b"]
    return x


def summarise(x: pd.DataFrame, rng) -> dict:
    gc = ["g_mean", "g_y", "g_exp_min", "g_minutes"]
    ts = x["ts"].to_numpy()
    uts, inv = np.unique(ts, return_inverse=True)
    S = np.zeros((len(uts), len(gc)))
    np.add.at(S, inv, x[gc].to_numpy(float))
    N = np.bincount(inv, minlength=len(uts)).astype(float)
    est = S.sum(0) / N.sum()
    bs = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, len(uts), len(uts)), minlength=len(uts)).astype(float)
        bs.append((w @ S) / max(w @ N, 1.0))
    bs = np.array(bs)
    r_dk = bs[:, 1] / bs[:, 0]
    r_mn = bs[:, 3] / bs[:, 2]
    return {"n": len(x), "tg": x[["game_id", "team"]].drop_duplicates().shape[0],
            "p_dk": est[0], "p_dk_se": bs[:, 0].std(), "r_dk": est[1], "r_dk_se": bs[:, 1].std(),
            "p_min": est[2], "r_min": est[3], "r_min_se": bs[:, 3].std(),
            "ratio_dk": est[1] / est[0], "ratio_dk_se": r_dk.std(),
            "ratio_min": est[3] / est[2] if est[2] else np.nan, "ratio_min_se": r_mn.std(),
            "resid": est[1] - est[0], "resid_se": (bs[:, 1] - bs[:, 0]).std()}


def main() -> int:
    conn = sqlite3.connect(f"file:{ROOT / 'var' / 'courtside.db'}?mode=ro", uri=True)
    st, mp = stars_frame(conn, ["2023-24", "2024-25", "2025-26"])
    rng = np.random.default_rng(7)
    frames = {}
    for lab, fn in SLICES.items():
        frames[lab] = build(pd.read_parquet(DIR / fn), st, mp, live=lab.startswith("live"))
    frames["2023-25"] = pd.concat([frames["2023-24"], frames["2024-25"]], ignore_index=True)
    hdr = ("| slice | base | group | n | star-out TG | proj DK | real DK | ratio DK (SE) | "
           "resid DK (SE) | proj min | real min | ratio min (SE) |")
    print(hdr)
    print("|" + "---|" * 12)
    for lab in ("2023-24", "2024-25", "2023-25", "live", "live10"):
        d = frames[lab]
        for bl, win in (("all", None), (f"+-{WINDOW}d", WINDOW)):
            x = gains(d, win)
            groups = {"top6": x["top6"], "star": x["tier0"] == "star",
                      "role": x["tier0"] == "role", "bench": x["tier0"] == "bench",
                      "top6 non-star": x["top6"] & (x["tier0"] != "star"), "all": x["top6"] | True}
            for gname, m in groups.items():
                if m.sum() < 30:
                    continue
                r = summarise(x[m.to_numpy()], rng)
                print(f"| {lab} | {bl} | {gname} | {r['n']} | {r['tg']} | {r['p_dk']:+.2f} | "
                      f"{r['r_dk']:+.2f} | {r['ratio_dk']:.2f} ({r['ratio_dk_se']:.2f}) | "
                      f"{r['resid']:+.2f} ({r['resid_se']:.2f}) | {r['p_min']:+.2f} | "
                      f"{r['r_min']:+.2f} | {r['ratio_min']:.2f} ({r['ratio_min_se']:.2f}) |")
        so = d[d["so"]]
        print(f"<!-- {lab}: rows {len(d)}, star-out rows {len(so)}, star-out team-games "
              f"{so[['game_id', 'team']].drop_duplicates().shape[0]}, team-games "
              f"{d[['game_id', 'team']].drop_duplicates().shape[0]} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
