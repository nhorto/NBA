"""Player pulse: impact metric, hot/cold, role changes, last-10 leaders, streaks, career highs.

Leak-free: only player games strictly before ``as_of`` are loaded (SQL filter). ``player_games``
has no DNP rows, so a missed game is a *team* game (seen via teammates' rows) with no row for
the player.

**Courtside Impact (CI)**, per game, in Game Score units::

    CI = GmSc + 0.5 * (PTS - 2 * TSA * par_TS) + 0.15 * REL

* ``GmSc`` = Hollinger Game Score: PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA - FTM) + 0.7 OREB
  + 0.3 DREB + STL + 0.7 AST + 0.7 BLK - 0.4 PF - TOV.
* ``TSA = FGA + 0.44 FTA``; ``par_TS = league_TS(season) - 0.25 * (USG - 0.20)``. This is the
  efficiency correction: Game Score charges a missed shot only 0.7, so it pays volume. The term
  adds or removes half a point per point scored above or below a usage-adjusted par. High-usage
  creators get a slightly lower bar, because efficiency falls as usage rises.
* ``REL = +/- - MIN * team_margin / game_minutes``: on-court +/- relative to what the team did
  per minute in that game, i.e. the on-court part of on/off. Single-game +/- is mostly noise, so
  its weight (0.15) is deliberately small; over 10 games it adds up.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from courtside.trends.common import (
    as_date,
    jsonable,
    load_player_games,
    prior_season,
    r1,
    season_of,
)

# --- impact metric ---------------------------------------------------------------------------
EFF_WEIGHT = 0.5
USG_SLOPE = 0.25
USG_PIVOT = 0.20
REL_WEIGHT = 0.15

# --- hot / cold (calibrated: docs/TRENDS.md) ---------------------------------------------------
HOT_MIN_GAMES = 8          # season games before hot/cold claims (last 5 + at least 3 before)
HOT_MIN_MPG = 20.0         # rotation player: season mpg
HOT_RECENT_MPG = 15.0      # and still playing real minutes over the last 5
ACTIVE_DAYS = 10           # must have played within this many days of as_of
K_PLAYER = 10.0            # prior-season weight (pseudo-games) in the baseline
SD_POP = 6.5               # population game-to-game sd of CI (shrink target)
SD_NU = 15.0               # pseudo-games behind SD_POP
SD_FLOOR = 4.0
HOT_Z = 2.0                # list threshold
HOT_TOP = 5
# Out-of-sample (2015-16..2024-25): next-10 CI vs the same baseline (calibrate.py)
HOT_VALIDITY = {
    "hot": {"n": 1029, "next10_beat_baseline_pct": 76.0, "next10_vs_baseline": 2.45},
    "cold": {"n": 398, "next10_below_baseline_pct": 68.3, "next10_vs_baseline": -1.47},
    "base_rate_pct": 54.5,
    "share_expected_by_chance": 0.61,
}

# --- role changes ------------------------------------------------------------------------------
ROLE_WINDOW = 5
ROLE_BASE_MIN = 5          # baseline team games needed
MIN_JUMP = 8.0             # mpg
USG_JUMP = 0.05            # usage share (5 percentage points)
NEW_ROT_MPG = 15.0
OLD_ROT_MPG = 8.0
OUT_MIN_TEAM_GAMES = 3
OUT_ROT_MPG = 20.0
STARTER_FROM_RANK = 7      # minutes rank before (>=) for the new-starter proxy

# --- leaders / streaks / highs -------------------------------------------------------------------
LEAD_WINDOW = 10
LEAD_MIN_APPS = 6
LEAD_MIN_MPG = 20.0
TS_MIN_TSA = 12.0
LEAD_TOP = 5
STREAKS = {  # name: (predicate column, min length to report)
    "20_point_games": ("pts20", 5),
    "30_point_games": ("pts30", 3),
    "double_doubles": ("dd", 5),
    "triple_doubles": ("td", 2),
}
HIGH_STATS = {"pts": 20, "reb": 12, "ast": 10, "fg3m": 5}   # minimum value to report
HIGH_MIN_PRIOR_GAMES = 82


def impact_per_game(df: pd.DataFrame) -> pd.Series:
    """Courtside Impact for every row of a player-games frame (see module docstring)."""
    if df.empty:
        return pd.Series(dtype=float)
    gmsc = (df.pts + 0.4 * df.fgm - 0.7 * df.fga - 0.4 * (df.fta - df.ftm) + 0.7 * df.oreb
            + 0.3 * df.dreb + df.stl + 0.7 * df.ast + 0.7 * df.blk - 0.4 * df.pf - df.tov)
    tsa = df.fga + 0.44 * df.fta
    by_season = df.groupby("season")
    lg_ts = (by_season.pts.transform("sum") / (2.0 * (by_season.fga.transform("sum")
                                                      + 0.44 * by_season.fta.transform("sum"))))
    usg = df["usg_pct"].fillna(USG_PIVOT) if "usg_pct" in df else USG_PIVOT
    par = lg_ts - USG_SLOPE * (usg - USG_PIVOT)
    pop = df.pts - 2.0 * tsa * par
    g = df.groupby(["game_id", "team"])
    tot_pm, tot_min = g.plus_minus.transform("sum"), g["min"].transform("sum")
    rel = df.plus_minus - df["min"] * (tot_pm / tot_min.where(tot_min > 0))
    return gmsc + EFF_WEIGHT * pop + REL_WEIGHT * rel.fillna(0.0)


def _prep(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg[pg["min"] > 0].copy()
    pg["imp"] = impact_per_game(pg)
    pg["tsa"] = pg.fga + 0.44 * pg.fta
    pg["stocks"] = pg.stl + pg.blk
    tens = (pg[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    pg["dd"] = tens >= 2
    pg["td"] = tens >= 3
    pg["pts20"] = pg.pts >= 20
    pg["pts30"] = pg.pts >= 30
    return pg.sort_values(["game_date", "game_id"])


# --- hot / cold --------------------------------------------------------------------------------
def _window_z(v: np.ndarray, prior_mean: float, prior_n: int, sd: float, w: int):
    if len(v) < w:
        return np.nan, np.nan, np.nan
    prev = v[:-w]
    k = min(K_PLAYER, prior_n) if np.isfinite(prior_mean) else 0.0
    if len(prev) + k < 3:
        return np.nan, np.nan, np.nan
    pm = prior_mean if k else 0.0
    base = (prev.sum() + k * pm) / (len(prev) + k)
    recent = v[-w:].mean()
    z = (recent - base) / (sd * np.sqrt(1.0 / w + 1.0 / (len(prev) + k)))
    return z, recent, base


def hot_z(v: np.ndarray, prior_mean: float, prior_n: int, sd: float) -> float:
    """Combined hot/cold z: whichever of the last-5 / last-10 z is larger in magnitude."""
    z5, _, _ = _window_z(v, prior_mean, prior_n, sd, 5)
    z10, _, _ = _window_z(v, prior_mean, prior_n, sd, 10)
    if not np.isfinite(z10):
        return z5
    if not np.isfinite(z5):
        return z10
    return z5 if abs(z5) > abs(z10) else z10


def _player_sd(cur: np.ndarray, prior_resid: np.ndarray) -> float:
    resid = np.concatenate([cur - cur.mean(), prior_resid]) if len(cur) else prior_resid
    dof = max(len(resid) - 2, 0)
    s2 = float(np.sum(resid ** 2) / dof) if dof else SD_POP ** 2
    sd = np.sqrt((SD_NU * SD_POP ** 2 + dof * s2) / (SD_NU + dof))
    return max(sd, SD_FLOOR)


def hot_cold_table(pg: pd.DataFrame, season: str, as_of: pd.Timestamp | date) -> pd.DataFrame:
    """All eligible rotation players with z, windows and baselines (unfiltered by threshold)."""
    as_of = pd.Timestamp(as_of)
    pg = pg[pg.game_date < as_of]
    pg = _prep(pg)
    cur = pg[pg.season == season]
    prev = pg[pg.season == prior_season(season)]
    pstats = prev.groupby("player_id").imp.agg(["mean", "size"])
    prev_resid = {pid: (g.imp - g.imp.mean()).to_numpy() for pid, g in prev.groupby("player_id")}
    rows = []
    for pid, g in cur.groupby("player_id"):
        n = len(g)
        if n < HOT_MIN_GAMES or g["min"].mean() < HOT_MIN_MPG:
            continue
        if g["min"].tail(5).mean() < HOT_RECENT_MPG:
            continue
        if (as_of - g.game_date.iloc[-1]).days > ACTIVE_DAYS:
            continue
        v = g.imp.to_numpy()
        pm, pn = (pstats.loc[pid, "mean"], int(pstats.loc[pid, "size"])) if pid in pstats.index \
            else (np.nan, 0)
        sd = _player_sd(v, prev_resid.get(pid, np.array([])))
        z5, r5, b5 = _window_z(v, pm, pn, sd, 5)
        z10, r10, b10 = _window_z(v, pm, pn, sd, 10)
        use10 = np.isfinite(z10) and not (np.isfinite(z5) and abs(z5) > abs(z10))
        z, recent, base, w = (z10, r10, b10, 10) if use10 else (z5, r5, b5, 5)
        if not np.isfinite(z):
            continue
        last = g.tail(w)
        rows.append({
            "player_id": int(pid), "player": g.player_name.iloc[-1], "team": g.team.iloc[-1],
            "games": n, "z": z, "z5": z5, "z10": z10, "window": w, "recent": recent,
            "baseline": base, "season_imp": v.mean(), "prior_mean": pm, "prior_n": pn, "sd": sd,
            "season_mpg": g["min"].mean(),
            "recent_line": _line(last), "season_line": _line(g),
        })
    return pd.DataFrame(rows)


def _line(g: pd.DataFrame) -> dict:
    tsa = g.tsa.sum()
    return {"games": len(g), "min": r1(g["min"].mean()), "pts": r1(g.pts.mean()),
            "reb": r1(g.reb.mean()), "ast": r1(g.ast.mean()), "stocks": r1(g.stocks.mean()),
            "fg3m": r1(g.fg3m.mean()),
            "ts_pct": r1(g.pts.sum() / (2 * tsa), 3) if tsa > 0 else None,
            "usg_pct": r1(np.average(g.usg_pct.fillna(0), weights=g["min"]), 3)
            if g["min"].sum() > 0 and g.usg_pct.notna().any() else None,
            "plus_minus": r1(g.plus_minus.mean()), "impact": r1(g.imp.mean())}


def _hot_entry(row: pd.Series) -> dict:
    return {"player_id": row.player_id, "player": row.player, "team": row.team,
            "z": r1(row.z, 2), "window": f"last {row.window}",
            "impact_recent": r1(row.recent), "impact_baseline": r1(row.baseline),
            "impact_season": r1(row.season_imp),
            "impact_prior_season": r1(row.prior_mean) if row.prior_n else None,
            "recent": row.recent_line, "season": row.season_line,
            "detail": f"CI {row.recent:.1f} over last {row.window} vs {row.baseline:.1f} "
                      f"baseline (z {row.z:+.1f}); {row.recent_line['pts']} pts, "
                      f"{row.recent_line['reb']} reb, {row.recent_line['ast']} ast, TS "
                      f"{_pct(row.recent_line['ts_pct'])} vs season "
                      f"{row.season_line['pts']}/{row.season_line['reb']}/"
                      f"{row.season_line['ast']}, TS {_pct(row.season_line['ts_pct'])}"}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


# --- role changes ------------------------------------------------------------------------------
def role_changes(cur: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
    """Minutes/usage jumps, new rotation players, starter-minutes proxy, disappearances.

    A player belongs to the team of his latest appearance. Team games come from teammates'
    rows, so a team game without a row for him is a DNP (or he was not yet on the team: games
    before his first appearance for the team are excluded from his baseline).
    """
    out: list[dict] = []
    if cur.empty:
        return out
    last_team = cur.groupby("player_id").team.last()
    for team, tgrp in cur.groupby("team"):
        tg = tgrp[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
        gids = tg.game_id.to_numpy()
        G = len(gids)
        if G < ROLE_WINDOW + ROLE_BASE_MIN:
            continue
        roster = last_team.index[last_team == team]
        sub = tgrp[tgrp.player_id.isin(roster)]
        if sub.empty:
            continue
        gpos = {g: i for i, g in enumerate(gids)}
        M = sub.pivot_table(index="game_id", columns="player_id", values="min", aggfunc="sum")
        M = M.reindex(index=gids).fillna(0.0)
        pids = M.columns.to_numpy()
        mins = M.to_numpy()                                   # G x P, DNP = 0
        first = np.array([gpos[g] for g in sub.sort_values("game_date")
                          .groupby("player_id").game_id.first().reindex(pids)])
        lastp = np.array([gpos[g] for g in sub.sort_values("game_date")
                          .groupby("player_id").game_id.last().reindex(pids)])
        rec_lo = G - ROLE_WINDOW
        rows = np.arange(G)[:, None]
        on = rows >= first[None, :]
        base_mask = on & (rows < rec_lo)
        base_n = base_mask.sum(axis=0)
        base_team_mpg = np.where(base_n > 0, (mins * base_mask).sum(0) / np.maximum(base_n, 1),
                                 np.nan)
        rec_team_mpg = mins[rec_lo:].mean(axis=0)
        rank_now = pd.Series(rec_team_mpg).rank(ascending=False, method="min").to_numpy()
        recent_ids = set(gids[rec_lo:])
        # baseline minutes rank on minutes per game *played* (so injury DNPs don't demote a
        # starter), among players with >= ROLE_BASE_MIN baseline appearances
        bsub = sub[~sub.game_id.isin(recent_ids)].groupby("player_id")["min"].agg(["mean", "size"])
        bsub = bsub[bsub["size"] >= ROLE_BASE_MIN]["mean"].rank(ascending=False, method="min")
        rank_base = bsub.reindex(pids).to_numpy()
        by_p = dict(tuple(sub.groupby("player_id")))
        for j, p in enumerate(pids):
            g = by_p[p]
            ent = {"player_id": int(p), "player": g.player_name.iloc[-1], "team": team}
            since = G - 1 - lastp[j]
            last10 = g.tail(10)
            if since >= OUT_MIN_TEAM_GAMES and len(g) >= 5 and last10["min"].mean() >= OUT_ROT_MPG:
                out.append({**ent, "kind": "out_of_rotation", "team_games_missed": int(since),
                            "last_played": g.game_date.iloc[-1].strftime("%Y-%m-%d"),
                            "mpg_before": r1(last10["min"].mean()),
                            "detail": f"missed the last {since} team games (last played "
                                      f"{g.game_date.iloc[-1]:%Y-%m-%d}, "
                                      f"{last10['min'].mean():.1f} mpg before): injury, rest or "
                                      f"coach's decision; cross-check injury reports"})
                continue
            rec = g[g.game_id.isin(recent_ids)]
            if len(rec) < ROLE_WINDOW - 1:
                continue
            rm = rec["min"].mean()
            if first[j] >= rec_lo:                             # joined during the window
                elsewhere = cur[(cur.player_id == p) & (cur.team != team)]
                if rm >= NEW_ROT_MPG:
                    prev_t = elsewhere.team.iloc[-1] if len(elsewhere) else None
                    out.append({**ent, "kind": "new_team" if prev_t else "new_arrival",
                                "mpg_recent": r1(rm), "previous_team": prev_t,
                                "detail": f"{rm:.1f} mpg in {len(rec)} games since joining {team}"
                                          + (f" from {prev_t}" if prev_t else "")})
                continue
            bn, bm = int(base_n[j]), base_team_mpg[j]
            if bn < ROLE_BASE_MIN:
                continue
            base = g[~g.game_id.isin(recent_ids)]
            if rec_team_mpg[j] >= NEW_ROT_MPG and bm < OLD_ROT_MPG:
                out.append({**ent, "kind": "new_in_rotation", "mpg_recent": r1(rm),
                            "mpg_before": r1(bm), "games_before": len(base),
                            "detail": f"{rm:.1f} mpg over the last {ROLE_WINDOW} after {bm:.1f} "
                                      f"mpg (DNPs as 0) in {bn} earlier team games"})
                continue
            if len(base) < 3:
                continue
            bpm = base["min"].mean()
            d = rm - bpm
            if d >= MIN_JUMP or (d <= -MIN_JUMP and bpm >= 20):
                out.append({**ent, "kind": "minutes_up" if d > 0 else "minutes_down",
                            "mpg_recent": r1(rm), "mpg_before": r1(bpm), "change": r1(d),
                            "detail": f"{rm:.1f} mpg last {len(rec)} vs {bpm:.1f} before "
                                      f"({d:+.1f})"})
            ur, ub = _usg(rec), _usg(base)
            if ur is not None and ub is not None and rm >= 20 and abs(ur - ub) >= USG_JUMP:
                out.append({**ent, "kind": "usage_up" if ur > ub else "usage_down",
                            "usg_recent": r1(ur, 3), "usg_before": r1(ub, 3),
                            "change": r1(ur - ub, 3),
                            "detail": f"usage {100 * ur:.1f}% last {len(rec)} vs {100 * ub:.1f}% "
                                      f"before ({100 * (ur - ub):+.1f} pts)"})
            rn, rb = rank_now[j], rank_base[j]
            if np.isfinite(rb):
                if rn <= 5 and rb >= STARTER_FROM_RANK and rec_team_mpg[j] >= 24 and d >= 5:
                    out.append({**ent, "kind": "starter_minutes", "minutes_rank_recent": int(rn),
                                "minutes_rank_before": int(rb), "mpg_recent": r1(rm),
                                "mpg_before": r1(bpm),
                                "detail": f"top-5 minutes on {team} over the last {ROLE_WINDOW} "
                                          f"(rank {int(rn)}, was {int(rb)}; {bpm:.1f} -> {rm:.1f} "
                                          f"mpg): likely a new starter (proxy, game logs have no "
                                          f"starter flag)"})
                elif rb <= 5 and rn >= STARTER_FROM_RANK and bpm >= 26 and d <= -5:
                    out.append({**ent, "kind": "lost_starter_minutes",
                                "minutes_rank_recent": int(rn), "minutes_rank_before": int(rb),
                                "mpg_recent": r1(rm), "mpg_before": r1(bpm),
                                "detail": f"minutes rank on {team} fell from {int(rb)} to "
                                          f"{int(rn)} ({bpm:.1f} -> {rm:.1f} mpg)"})
    return out


def _usg(g: pd.DataFrame) -> float | None:
    g = g[g.usg_pct.notna()]
    if g.empty or g["min"].sum() <= 0:
        return None
    return float(np.average(g.usg_pct, weights=g["min"]))


# --- leaders, streaks, highs -----------------------------------------------------------------------
def leaders(cur: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, list[dict]]:
    """Top 5 over each player's current team's last 10 games (min 6 appearances)."""
    if cur.empty:
        return {}
    tg = cur[["team", "game_id", "game_date"]].drop_duplicates().sort_values("game_date")
    last10 = {t: set(g.game_id.tail(LEAD_WINDOW)) for t, g in tg.groupby("team")}
    team_now = cur.groupby("player_id").team.last()
    rows = []
    for pid, g in cur.groupby("player_id"):
        t = team_now[pid]
        w = g[(g.team == t) & g.game_id.isin(last10[t])]
        if len(w) < LEAD_MIN_APPS or (as_of - g.game_date.iloc[-1]).days > ACTIVE_DAYS:
            continue
        line = _line(w)
        rows.append({"player_id": int(pid), "player": g.player_name.iloc[-1], "team": t,
                     **line, "tsa_pg": w.tsa.mean()})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    cats = {"impact": ("impact", df["min"] >= LEAD_MIN_MPG), "pts": ("pts", None),
            "reb": ("reb", None), "ast": ("ast", None), "stocks": ("stocks", None),
            "fg3m": ("fg3m", None), "ts_pct": ("ts_pct", df.tsa_pg >= TS_MIN_TSA)}
    keep = ["player_id", "player", "team", "games", "min", "pts", "reb", "ast", "stocks",
            "fg3m", "ts_pct", "impact"]
    for name, (col, mask) in cats.items():
        d = df if mask is None else df[mask]
        d = d[d[col].notna()].sort_values(col, ascending=False).head(LEAD_TOP)
        out[name] = [{**{k: r[k] for k in keep}, "value": r[col]} for _, r in d.iterrows()]
    return out


def _run_len(flags: np.ndarray) -> int:
    n = 0
    for f in flags[::-1]:
        if not f:
            break
        n += 1
    return n


def streaks(cur: pd.DataFrame, prev: pd.DataFrame | None = None) -> list[dict]:
    """Active streaks over consecutive regular-season appearances, carried across seasons.

    Only players who appeared this season are reported. ``prev`` (prior season) extends the
    count; a streak that covers every loaded game is marked ``at_least``.
    """
    out = []
    allg = cur if prev is None or prev.empty else pd.concat(
        [prev[prev.player_id.isin(cur.player_id.unique())], cur])
    allg = allg.sort_values(["game_date", "game_id"])
    season = cur.season.iloc[0] if len(cur) else None
    for pid, g in allg.groupby("player_id"):
        if season is None or g.season.iloc[-1] != season:
            continue
        for name, (col, min_len) in STREAKS.items():
            n = _run_len(g[col].to_numpy())
            if n >= min_len:
                tail = g.tail(n)
                this = int((tail.season == season).sum())
                at_least = n == len(g)
                label = name.replace("_", " ").replace("20 point", "20-point")
                label = label.replace("30 point", "30-point")
                out.append({"player_id": int(pid), "player": g.player_name.iloc[-1],
                            "team": g.team.iloc[-1], "kind": name, "games": n,
                            "games_this_season": this, "at_least": bool(at_least),
                            "since": tail.game_date.iloc[0].strftime("%Y-%m-%d"),
                            "avg_pts": r1(tail.pts.mean()), "avg_reb": r1(tail.reb.mean()),
                            "avg_ast": r1(tail.ast.mean()),
                            "detail": f"{'at least ' if at_least else ''}{n} straight {label}"
                                      f"{f' ({this} this season)' if this < n else ''}, "
                                      f"avg {tail.pts.mean():.1f} pts/{tail.reb.mean():.1f} reb/"
                                      f"{tail.ast.mean():.1f} ast"})
    out.sort(key=lambda x: -x["games"])
    return out


_CAREER_CACHE: dict[tuple, pd.DataFrame] = {}


def career_max_before(conn: sqlite3.Connection, player_ids: list[int], before: date) -> pd.DataFrame:
    """Regular-season career maxima before ``before`` (history since 2000-01). Cached."""
    try:
        path = conn.execute("PRAGMA database_list").fetchone()[2] or id(conn)
    except sqlite3.Error:  # pragma: no cover
        path = id(conn)
    key = (path, before.isoformat(), tuple(sorted(player_ids)))
    if key in _CAREER_CACHE:
        return _CAREER_CACHE[key]
    if not player_ids:
        return pd.DataFrame(columns=["n", *HIGH_STATS]).rename_axis("player_id")
    cols = ", ".join(f"MAX({c}) AS {c}" for c in HIGH_STATS)
    q = (f"SELECT player_id, COUNT(*) AS n, {cols} FROM player_games "
         f"WHERE player_id IN ({','.join('?' * len(player_ids))}) "
         f"AND season_type = 'Regular Season' AND game_date < ? GROUP BY player_id")
    df = pd.read_sql_query(q, conn, params=[*map(int, player_ids), before.isoformat()])
    df = df.set_index("player_id")
    if len(_CAREER_CACHE) > 16:
        _CAREER_CACHE.clear()
    _CAREER_CACHE[key] = df
    return df


def career_highs(conn: sqlite3.Connection, cur: pd.DataFrame, season: str) -> list[dict]:
    """Games this season (before as_of) that set a regular-season career high."""
    if cur.empty:
        return []
    start = date(int(season[:4]), 7, 1)
    prior = career_max_before(conn, sorted(cur.player_id.unique().tolist()), start)
    c = cur.sort_values(["game_date", "game_id"])
    pn = c.player_id.map(prior["n"]).fillna(0).astype(int)
    out = []
    for stat, floor in HIGH_STATS.items():
        before_season = c.player_id.map(prior[stat]).fillna(-1.0)
        running = c.groupby("player_id")[stat].shift().fillna(-1.0)
        running = running.groupby(c.player_id).cummax()
        best = np.maximum(before_season, running)
        hit = (c[stat] > best) & (c[stat] >= floor) & ((pn >= HIGH_MIN_PRIOR_GAMES)
                                                       | (c[stat] >= 1.5 * floor))
        for i in c.index[hit]:
            row, b_ = c.loc[i], best[i]
            out.append({"player_id": int(row.player_id), "player": row.player_name,
                        "team": row.team, "stat": stat, "value": int(row[stat]),
                        "previous_high": int(b_) if b_ >= 0 else None,
                        "date": row.game_date.strftime("%Y-%m-%d"), "game_id": row.game_id,
                        "prior_career_games": int(pn[i]),
                        "detail": f"career-high {int(row[stat])} {stat} (previous "
                                  f"{int(b_) if b_ >= 0 else 'n/a'})"})
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


# --- entry point -------------------------------------------------------------------------------
def compute(conn: sqlite3.Connection | None, pg: pd.DataFrame, as_of: date, season: str) -> dict:
    ts = pd.Timestamp(as_of)
    pg = pg[pg.game_date < ts]
    prepped = _prep(pg)
    cur = prepped[prepped.season == season]
    team_games = cur.groupby("team").game_id.nunique()
    notes = []
    hc = hot_cold_table(pg, season, ts)
    hot = cold = []
    if not hc.empty:
        hot = [_hot_entry(r) for _, r in hc[hc.z >= HOT_Z].sort_values("z", ascending=False)
               .head(HOT_TOP).iterrows()]
        cold = [_hot_entry(r) for _, r in hc[hc.z <= -HOT_Z].sort_values("z")
                .head(HOT_TOP).iterrows()]
    if team_games.empty or team_games.median() < HOT_MIN_GAMES:
        notes.append(f"early season: hot/cold needs {HOT_MIN_GAMES} games; teams have played "
                     f"{int(team_games.median()) if len(team_games) else 0}")
    out = {
        "season": season, "as_of": as_of.isoformat(),
        "impact_metric": "CI = GmSc + 0.5*(PTS - 2*TSA*par_TS) + 0.15*on-court rel +/-",
        "eligible_hot_cold": len(hc),
        "hot": hot, "cold": cold,
        "role_changes": role_changes(cur, ts),
        "leaders_last10": leaders(cur, ts),
        "streaks": streaks(cur, prepped[prepped.season == prior_season(season)]),
        "career_highs": career_highs(conn, cur, season) if conn is not None else [],
        "notes": notes,
    }
    return jsonable(out)


def player_pulse(conn: sqlite3.Connection, as_of: date, season: str | None = None) -> dict:
    """Player pulse as of the morning of ``as_of`` (games before that date only)."""
    as_of = as_date(as_of)
    season = season or season_of(as_of)
    pg = load_player_games(conn, [prior_season(season), season], as_of)
    return compute(conn, pg, as_of, season)


__all__ = [
    "career_highs",
    "hot_cold_table",
    "hot_z",
    "impact_per_game",
    "leaders",
    "player_pulse",
    "role_changes",
    "streaks",
]
