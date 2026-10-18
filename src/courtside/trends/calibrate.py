"""Historical calibration + validation of the trend constants. Not used at runtime.

    uv run python -m courtside.trends.calibrate            # team + player, prints tables
    uv run python -m courtside.trends.calibrate --team     # team only

Split: constants are *fit* on 2001-02..2014-15 and the resulting flags are *validated* out of
sample on 2015-16..2024-25 (the numbers quoted in docs/TRENDS.md).
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from courtside.trends import player as P
from courtside.trends import team as T
from courtside.trends.common import (
    connect_ro,
    load_player_games,
    load_team_games,
    prior_season,
)

FIT = [f"{y}-{(y + 1) % 100:02d}" for y in range(2001, 2015)]
VAL = [f"{y}-{(y + 1) % 100:02d}" for y in range(2015, 2025)]


def _checkpoints(cur: pd.DataFrame, min_games: int, step: int = 7, tail: int = 10) -> list[date]:
    """Weekly dates from when the median team has ``min_games`` until ``tail`` games remain."""
    g = cur.sort_values("game_date")
    g = g.assign(k=g.groupby("fr").cumcount() + 1)
    total = g.groupby("fr")["k"].max().median()
    start = g[g.k >= min_games].groupby("fr")["game_date"].min().median()
    end = g[g.k >= total - tail].groupby("fr")["game_date"].min().median()
    out, d = [], start + pd.Timedelta(days=1)
    while d <= end:
        out.append(d.date())
        d += pd.Timedelta(days=step)
    return out


def team_rows(all_games: pd.DataFrame, seasons: list[str], z_shuffles: int = 4,
              seed: int = 0) -> pd.DataFrame:
    """One row per (season, checkpoint, team): trend stats as of the date + what happened next."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        cur_all = all_games[all_games.season == s]
        prev = all_games[all_games.season == prior_season(s)]
        if cur_all.empty or prev.empty:
            continue
        prior = T.prior_from_season(prev)
        final = T.fit_ratings(cur_all, None, k=1.0)          # ex-post, evaluation only
        post = T.adjust_games(cur_all, final)
        for d in _checkpoints(cur_all, 10):
            ts = pd.Timestamp(d)
            cur = cur_all[cur_all.game_date < ts]
            r = T.fit_ratings(cur, prior)
            adj = T.adjust_games(cur, r).sort_values(["game_date", "game_id"])
            st = T.trend_stats(adj, prior).set_index("fr")
            fut = post[post.game_date >= ts].sort_values("game_date").groupby("fr").head(10)
            fsum = fut.groupby("fr").agg(next_adj=("adj_net", "mean"), next_n=("adj_net", "size"),
                                         next_win=("win", "mean"))
            past_win = cur.groupby("fr")["win"].mean()
            # null: shuffle each team's game order (kills real form, keeps fat tails)
            null_z = {}
            for fr, grp in adj.groupby("fr"):
                v = grp["adj_net"].to_numpy()
                po, pd_ = prior.get(fr, (0.0, 0.0))
                zs = []
                for _ in range(z_shuffles):
                    vv = rng.permutation(v)
                    _, _, dl, z = T._window_stats(vv, po - pd_, T.WINDOW, T.K_PRIOR, T.SIGMA_NET)
                    zs.append((z, dl))
                null_z[fr] = zs
            for fr, x in st.iterrows():
                if fr not in fsum.index or fsum.loc[fr, "next_n"] < 10:
                    continue
                rows.append({"season": s, "date": d, "fr": fr, "n": x.n, "z": x.z,
                             "delta": x.delta, "recent": x.recent, "base": x.base,
                             "rating": r.net(fr), "next_adj": fsum.loc[fr, "next_adj"],
                             "next_win": fsum.loc[fr, "next_win"], "past_win": past_win[fr],
                             "null": null_z.get(fr, [])})
    return pd.DataFrame(rows)


def sigma_and_prior(all_games: pd.DataFrame, seasons: list[str]) -> None:
    """SIGMA_NET / SIGMA_SIDE from end-of-season residuals; RHO, K_PRIOR by prediction error."""
    res_net, res_off = [], []
    for s in seasons:
        g = all_games[all_games.season == s]
        r = T.fit_ratings(g, None, k=1.0)
        a = T.adjust_games(g, r)
        res_net.append(a.adj_net - (a.fr.map(r.off) - a.fr.map(r.dal)))
        res_off.append(a.adj_off - a.fr.map(r.off))
    print(f"SIGMA_NET  sd(game adj net - season rating) = {pd.concat(res_net).std():.2f}")
    print(f"SIGMA_SIDE sd(game adj off - season off)     = {pd.concat(res_off).std():.2f}")
    # prior: predict rest-of-season mean adj net from first n games + rho * prior rating
    recs = []
    for s in seasons:
        g = all_games[all_games.season == s]
        prev = all_games[all_games.season == prior_season(s)]
        if prev.empty:
            continue
        pr = T.fit_ratings(prev, None, k=1.0)
        r = T.fit_ratings(g, None, k=1.0)
        a = T.adjust_games(g, r).sort_values("game_date")
        for fr, grp in a.groupby("fr"):
            v = grp.adj_net.to_numpy()
            recs.append((pr.net(fr) if fr in pr.off else 0.0, v))
    print("\nRHO/K grid: RMSE of rest-of-season mean adj net, averaged over n in {0,5,10,20,30}")
    best = None
    for rho in (0.4, 0.5, 0.6, 0.7, 0.8):
        line = []
        for k in (4, 8, 12, 16, 24, 32):
            err = []
            for n in (0, 5, 10, 20, 30):
                for p, v in recs:
                    if len(v) <= n + 10:
                        continue
                    est = (v[:n].sum() + k * rho * p) / (n + k)
                    err.append((est - v[n:].mean()) ** 2)
            e = float(np.sqrt(np.mean(err)))
            line.append(f"k={k}:{e:.3f}")
            if best is None or e < best[0]:
                best = (e, rho, k)
        print(f"  rho={rho}: " + "  ".join(line))
    print(f"  best: rho={best[1]} k={best[2]} rmse={best[0]:.3f}")


def team_report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== team flags, {label}: {len(df)} team-checkpoints, "
          f"{df.season.nunique()} seasons ===")
    noise = T.SIGMA_NET ** 2 * (1 / 10 + 1 / (df.n - 10 + T.K_PRIOR))
    tau2 = df.delta.var() - noise.mean()
    print(f"TAU: var(delta)={df.delta.var():.2f} noise={noise.mean():.2f} "
          f"-> tau={np.sqrt(max(tau2, 0)):.2f}")
    resid = df.next_adj - df.rating
    slope = np.polyfit(df.recent - df.rating, resid, 1)[0]
    print(f"W_RECENT: slope of (next10 - rating) on (last10 - rating) = {slope:.3f}")
    print(" z_thr  real%  null%   FDR  | rising: n  next10>rating  mean(next-rating) win%chg"
          " | sliding: n  next10<rating  mean  win%chg")
    nulls = [(z, dl) for zs in df["null"] for z, dl in zs]
    nz = np.array([z for z, _ in nulls])
    nd = np.array([dl for _, dl in nulls])
    for zt in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
        up = (df.z >= zt) & (df.delta >= T.MIN_DELTA) & (df.n >= T.MIN_GAMES_FLAG)
        dn = (df.z <= -zt) & (df.delta <= -T.MIN_DELTA) & (df.n >= T.MIN_GAMES_FLAG)
        real = (up | dn).mean()
        null = (((nz >= zt) & (nd >= T.MIN_DELTA)) | ((nz <= -zt) & (nd <= -T.MIN_DELTA))).mean()
        u, d_ = df[up], df[dn]
        print(f" {zt:5.2f} {100 * real:6.2f} {100 * null:6.2f} {null / real:6.2f} |"
              f" {len(u):6d}  {100 * (u.next_adj > u.rating).mean():6.1f}%  "
              f"{(u.next_adj - u.rating).mean():+6.2f}  {100 * (u.next_win - u.past_win).mean():+5.1f}"
              f" | {len(d_):6d}  {100 * (d_.next_adj < d_.rating).mean():6.1f}%  "
              f"{(d_.next_adj - d_.rating).mean():+6.2f}  "
              f"{100 * (d_.next_win - d_.past_win).mean():+5.1f}")
    base_up = (df.next_adj > df.rating).mean()
    print(f" all team-checkpoints: next10 > rating {100 * base_up:.1f}% "
          f"(mean resid {resid.mean():+.2f})")
    # does the flag add information beyond the (shrunk) season rating? vs raw season mean too
    for zt in (T.Z_FLAG,):
        up = (df.z >= zt) & (df.delta >= T.MIN_DELTA) & (df.n >= T.MIN_GAMES_FLAG)
        dn = (df.z <= -zt) & (df.delta <= -T.MIN_DELTA) & (df.n >= T.MIN_GAMES_FLAG)
        print(f" at Z_FLAG={zt}: rising next10 beats pre-window baseline "
              f"{100 * (df[up].next_adj > df[up].base).mean():.1f}%, sliding below baseline "
              f"{100 * (df[dn].next_adj < df[dn].base).mean():.1f}%; flags per checkpoint "
              f"{(up | dn).sum() / df.groupby(['season', 'date']).ngroups:.2f}")


def player_report(conn: sqlite3.Connection, seasons: list[str], label: str, seed: int = 0) -> None:
    """Hot/cold: null rate via within-player shuffles, and next-10 persistence of the flag."""
    rng = np.random.default_rng(seed)
    real_rows, null_z = [], []
    for s in seasons:
        y = int(s[:4])
        pg = load_player_games(conn, [prior_season(s), s], date(y + 1, 7, 1))
        cur_all = pg[pg.season == s]
        tg = load_team_games(conn, [s], date(y + 1, 7, 1))
        dates = _checkpoints(tg, 20, step=21, tail=10)
        for d in dates:
            ts = pd.Timestamp(d)
            base = pg[pg.game_date < ts]
            st = P.hot_cold_table(base, s, ts)
            if st.empty:
                continue
            fut = cur_all[cur_all.game_date >= ts].copy()
            fut["imp"] = P.impact_per_game(fut)
            nxt = fut.sort_values("game_date").groupby("player_id").head(10)
            nx = nxt.groupby("player_id").agg(next_imp=("imp", "mean"), next_n=("imp", "size"))
            st = st.join(nx, on="player_id")
            st["season"], st["date"] = s, d
            real_rows.append(st)
            # null
            cur = base[base.season == s].copy()
            cur["imp"] = P.impact_per_game(cur)
            ids = set(st.player_id)
            for pid, grp in cur[cur.player_id.isin(ids)].groupby("player_id"):
                row = st[st.player_id == pid].iloc[0]
                v = grp.sort_values("game_date")["imp"].to_numpy()
                for _ in range(3):
                    vv = rng.permutation(v)
                    null_z.append(P.hot_z(vv, row.prior_mean, row.prior_n, row.sd))
    df = pd.concat(real_rows, ignore_index=True)
    nz = np.array(null_z)
    print(f"\n=== player hot/cold, {label}: {len(df)} player-checkpoints ===")
    print(" z_thr  real_hot%  null_hot%  FDR | hot: n  next10>baseline  mean(next-base)"
          " | cold: n  next10<baseline  mean")
    df = df[df.next_n >= 5]
    for zt in (1.5, 2.0, 2.5, 3.0):
        h, c = df[df.z >= zt], df[df.z <= -zt]
        real = (df.z >= zt).mean()
        null = (nz >= zt).mean()
        print(f" {zt:5.2f} {100 * real:9.2f} {100 * null:9.2f} {null / max(real, 1e-9):5.2f} |"
              f" {len(h):6d}  {100 * (h.next_imp > h.baseline).mean():6.1f}%  "
              f"{(h.next_imp - h.baseline).mean():+5.2f} | {len(c):6d}  "
              f"{100 * (c.next_imp < c.baseline).mean():6.1f}%  {(c.next_imp - c.baseline).mean():+5.2f}")
    print(f" all: next10 > baseline {100 * (df.next_imp > df.baseline).mean():.1f}%")
    resid = df.next_imp - df.baseline
    print(f" slope of (next10 - baseline) on (recent - baseline): "
          f"{np.polyfit(df.recent - df.baseline, resid, 1)[0]:.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--team", action="store_true")
    ap.add_argument("--player", action="store_true")
    a = ap.parse_args(argv)
    conn = connect_ro(a.db)
    do_team = a.team or not a.player
    do_player = a.player or not a.team
    if do_team:
        g = load_team_games(conn, [f"{y}-{(y + 1) % 100:02d}" for y in range(2000, 2026)],
                            date(2026, 7, 1))
        sigma_and_prior(g, FIT)
        team_report(team_rows(g, FIT), "FIT 2001-02..2014-15")
        team_report(team_rows(g, VAL), "VALIDATION 2015-16..2024-25 (out of sample)")
    if do_player:
        player_report(conn, FIT[-6:], "FIT 2009-10..2014-15")
        player_report(conn, VAL, "VALIDATION 2015-16..2024-25 (out of sample)")


if __name__ == "__main__":
    main()
