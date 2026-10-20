"""F5: matchups (docs/PLAYERS.md Phase 2F): likely defenders x defender quality.

Source: `player_matchups` (stats.nba.com boxscorematchupsv3, 2017-18 on, ~2.2M rows): per game
and (offensive player, defender) the matchup minutes, partial possessions and the offensive
player's output against that defender. Everything is strictly before the game day
(`common.ew_asof`, recency half-life + season carry-over) and heavily shrunk.

1. **Attacker baseline.** Each offensive player's own per-matchup-possession rates (points,
   FGA, 3PA, AST, TOV, FTA) and FG% / 3P% as of each day, shrunk to the league
   (K_O possessions / K_FG attempts).
2. **Defender quality** (allowed vs expected). For every matchup row the expected output is the
   attacker's baseline as of that day x the row's possessions (FGM: his FG% x the FGA; 3PM: his
   3P% x the 3PA). Per defender j: q_j,x = (Y + K_D,x l_x) / (E + K_D,x), relative to the
   league ratio l_x at the date (log). x in pts, fga, fg3a, fgm (FG% allowed), fg3m (3P%
   allowed), ast, tov (forced), fta.
3. **Exposure** of attacker a to each defender b on tonight's opponent roster R (players who
   play, weight = their pre-game minutes state; live: x P(available)):
     prior   p_ab ∝ w_b A_b(grp_a), A_b(g) = b's share of defensive matchup time against
             attackers of position group g (shrunk to his group's league profile, K_A minutes)
     direct  s_ab = (D_ab + LAMBDA p_ab) / (T_ab + LAMBDA), D_ab = weighted minutes b guarded a in
             their earlier games against each other, T_ab = a's total matchup minutes in those
             games; then renormalised over R.
   The reverse view (a defending R's attackers k) uses the same pair history with
   T = a's total defensive matchup minutes and B_k(g) = k's share of time guarded by group g.
4. **Features** (per player-game):
     f5_q_<x>      sum_b s_ab log q_b,x       matchup-weighted defender quality he faces
     f5_prim_share max_b s_ab                 how concentrated the primary assignment is
     f5_prim_pts   log q_b*,pts of the likely primary defender
     f5_r_<x>      sum_k s'_ak log(o_k,x / league)   the attacking tendencies of whom HE guards
                   (x in pts, fga, fg3a, ast, tov: drives his steals / blocks / rebounds)
     f5_ln_dir     log(1 + sum_b T_ab)       direct matchup evidence
   Rows before the matchup era (no league matchup sums before the day) are NaN.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .. import state as ST
from .common import ew_asof

MATCHUP_START = "2017-18"
H_O, RHO_O, K_O, K_FG = 120.0, 0.7, 300.0, 150.0     # attacker baselines
H_D, RHO_D = 240.0, 0.8                               # defender quality / profiles / pairs
K_D = {"pts": 400.0, "fga": 300.0, "fg3a": 100.0, "fgm": 150.0, "fg3m": 50.0, "ast": 80.0,
       "tov": 50.0, "fta": 80.0}
K_A, LAMBDA = 120.0, 40.0
O_RATES = ("pts", "fga", "fg3a", "ast", "tov", "fta")            # per possession
DQ = ("pts", "fga", "fg3a", "fgm", "fg3m", "ast", "tov", "fta")
R_X = ("pts", "fga", "fg3a", "ast", "tov")

Q_COLS = tuple(f"f5_q_{x}" for x in DQ)
R_COLS = tuple(f"f5_r_{x}" for x in R_X)
TEAM_COLS = Q_COLS + R_COLS + ("f5_prim_share", "f5_prim_pts", "f5_ln_dir")   # per player-game
ROW_COLS: tuple = ()
NEUTRAL = {c: 0.0 for c in Q_COLS + R_COLS + ("f5_prim_pts",)}

PER_STAT = {
    "pts": ("q_pts", "q_fga", "q_fgm", "q_fg3m", "q_fta", "prim_share", "prim_pts"),
    "fg3m": ("q_fg3a", "q_fg3m", "q_pts", "prim_share"),
    "reb": ("q_fgm", "q_fga", "r_fga", "r_fg3a"),
    "ast": ("q_ast", "q_tov", "q_pts", "prim_share"),
    "stl": ("r_tov", "r_pts", "r_fga", "r_ast"),
    "blk": ("r_fga", "r_fg3a", "r_pts"),
    "tov": ("q_tov", "q_ast", "prim_share", "prim_pts"),
}
OWN_MONO = {"pts": "f5_q_pts", "fg3m": "f5_q_fg3m", "ast": "f5_q_ast", "tov": "f5_q_tov",
            "stl": "f5_r_tov"}


# ============================================================================== loading

class History:
    """Matchup rows and who-played rosters of seasons [first, last] (games before `before`
    only, when given), with day / sidx and position groups."""

    def __init__(self, conn: sqlite3.Connection, first: str | None = None,
                 last: str | None = None, before=None, pos=None):
        first = max(first or MATCHUP_START, MATCHUP_START)
        gd = ("SELECT game_id, MIN(game_date) AS game_date, MIN(season) AS season FROM "
              "team_games GROUP BY game_id")
        cond, args = " WHERE g.season >= ?", [first]
        if last:
            cond += " AND g.season <= ?"
            args.append(last)
        if before is not None:
            cond += " AND g.game_date < ?"
            args.append(pd.Timestamp(before).date().isoformat())
        m = pd.read_sql_query(
            "SELECT m.game_id, m.off_player_id AS off, m.def_player_id AS def, m.matchup_min AS "
            "mmin, m.partial_poss AS poss, m.pts, m.fgm, m.fga, m.fg3m, m.fg3a, m.ast, m.tov, "
            f"m.fta, g.game_date, g.season FROM player_matchups m JOIN ({gd}) g "
            f"ON g.game_id = m.game_id" + cond, conn, params=args)
        r = pd.read_sql_query(
            "SELECT p.game_id, p.team, p.player_id, g.game_date, g.season FROM player_games p "
            f"JOIN ({gd}) g ON g.game_id = p.game_id" + cond + " AND p.min > 0", conn,
            params=args)
        pos = pos if pos is not None else ST.load_positions(conn)
        for d in (m, r):
            d["day"] = ST._to_day(d["game_date"]) if len(d) else np.zeros(0, np.int64)
            d["sidx"] = d["season"].str[:4].astype(int) if len(d) else np.zeros(0, int)
        num = ["mmin", "poss", "pts", "fgm", "fga", "fg3m", "fg3a", "ast", "tov", "fta"]
        m[num] = m[num].astype(float).fillna(0.0)
        m["off_grp"] = ST._attach_grp(m["off"].to_numpy(), m["sidx"].to_numpy(), pos) \
            if len(m) else np.zeros(0, int)
        m["def_grp"] = ST._attach_grp(m["def"].to_numpy(), m["sidx"].to_numpy(), pos) \
            if len(m) else np.zeros(0, int)
        self.m = m
        self.rosters = r[["game_id", "team", "player_id", "day", "sidx"]]
        self.pos = pos


# ============================================================================== 1-2. rates, quality

def _attacker_sums(m: pd.DataFrame):
    """Per (off, game) totals: the attacker history rows."""
    cols = ["poss"] + list(O_RATES) + ["fgm", "fg3m"]
    return m.groupby(["off", "game_id", "day", "sidx"], sort=False)[cols].sum().reset_index()


def attacker_rates(m: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Shrunk per-possession rates (O_RATES), FG% and 3P% of each query (player_id, day, sidx)
    from matchup rows before the day, plus the league values (columns lg_<x>)."""
    a = _attacker_sums(m)
    cols = ["poss"] + list(O_RATES) + ["fgm", "fg3m"]
    qd, qs = q["day"].to_numpy(), q["sidx"].to_numpy()
    S = ew_asof(a["off"].to_numpy(), a["day"].to_numpy(), a["sidx"].to_numpy(),
                a[cols].to_numpy(float), q["player_id"].to_numpy(), qd, qs,
                half_life=H_O, rho=RHO_O)
    lg = a.groupby(["day", "sidx"], sort=False)[cols].sum().reset_index()
    L = ew_asof(np.zeros(len(lg), int), lg["day"].to_numpy(), lg["sidx"].to_numpy(),
                lg[cols].to_numpy(float), np.zeros(len(q), int), qd, qs, half_life=H_O, rho=RHO_O)
    ix = {c: i for i, c in enumerate(cols)}
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for x in O_RATES:
            l = L[:, ix[x]] / L[:, ix["poss"]]
            out[f"lg_{x}"] = l
            out[x] = (S[:, ix[x]] + K_O * l) / (S[:, ix["poss"]] + K_O)
        for made, att in (("fgm", "fga"), ("fg3m", "fg3a")):
            l = L[:, ix[made]] / L[:, ix[att]]
            out[f"lg_{made}"] = l
            out[made] = (S[:, ix[made]] + K_FG * l) / (S[:, ix[att]] + K_FG)
    return pd.DataFrame(out, index=q.index)


def _expected(m: pd.DataFrame) -> pd.DataFrame:
    """Per (def, game): Y and E (expected given the attackers' baselines as of that day)."""
    a = m[["off", "game_id", "day", "sidx"]].drop_duplicates(["off", "game_id"]).reset_index(
        drop=True)
    r = attacker_rates(m, a.rename(columns={"off": "player_id"})).add_prefix("rt_")
    r["off"], r["game_id"] = a["off"].to_numpy(), a["game_id"].to_numpy()
    x = m.merge(r, on=["off", "game_id"], how="left")
    poss = x["poss"].to_numpy(float)
    E = {f"E_{k}": x[f"rt_{k}"].to_numpy(float) * poss
         for k in ("pts", "fga", "fg3a", "ast", "tov", "fta")}
    E["E_fgm"] = x["rt_fgm"].to_numpy(float) * x["fga"].to_numpy(float)
    E["E_fg3m"] = x["rt_fg3m"].to_numpy(float) * x["fg3a"].to_numpy(float)
    d = pd.DataFrame({"def": x["def"].to_numpy(), "game_id": x["game_id"].to_numpy(),
                      "day": x["day"].to_numpy(), "sidx": x["sidx"].to_numpy()})
    ok = np.ones(len(x), bool)
    for k in DQ:
        ok &= np.isfinite(E[f"E_{k}"])
    for k in DQ:
        d[f"Y_{k}"] = np.where(ok, x[k].to_numpy(float), 0.0)
        d[f"E_{k}"] = np.where(ok, np.nan_to_num(E[f"E_{k}"]), 0.0)
    cols = [f"Y_{k}" for k in DQ] + [f"E_{k}" for k in DQ]
    return d.groupby(["def", "game_id", "day", "sidx"], sort=False)[cols].sum().reset_index()


def defender_quality(dq_hist: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """log(q_j,x / l_x) per query (player_id = defender, day, sidx), x in DQ."""
    cols = [f"Y_{k}" for k in DQ] + [f"E_{k}" for k in DQ]
    qd, qs = q["day"].to_numpy(), q["sidx"].to_numpy()
    S = ew_asof(dq_hist["def"].to_numpy(), dq_hist["day"].to_numpy(),
                dq_hist["sidx"].to_numpy(), dq_hist[cols].to_numpy(float),
                q["player_id"].to_numpy(), qd, qs, half_life=H_D, rho=RHO_D)
    lg = dq_hist.groupby(["day", "sidx"], sort=False)[cols].sum().reset_index()
    L = ew_asof(np.zeros(len(lg), int), lg["day"].to_numpy(), lg["sidx"].to_numpy(),
                lg[cols].to_numpy(float), np.zeros(len(q), int), qd, qs,
                half_life=H_D, rho=RHO_D)
    n = len(DQ)
    out = {}
    for j, k in enumerate(DQ):
        l = (L[:, j] + 1.0) / (L[:, n + j] + 1.0)
        qq = (S[:, j] + K_D[k] * l) / (S[:, n + j] + K_D[k])
        out[k] = np.log(qq / l)
    return pd.DataFrame(out, index=q.index)


# ============================================================================== 3. exposure

def _profiles(m: pd.DataFrame, q: pd.DataFrame, side: str) -> np.ndarray:
    """(nq, 3) shares of a player's matchup time by the OTHER side's position group, as of the
    day, shrunk to the league profile of his own group. side 'def': player = defender, shares
    over attacker groups (A); side 'off': player = attacker, shares over defender groups (B)."""
    me, other = ("def", "off") if side == "def" else ("off", "def")
    g = m[f"{other}_grp"].to_numpy()
    V = np.column_stack([m["mmin"].to_numpy(float) * (g == k) for k in range(3)])
    h = pd.DataFrame(V, columns=["g0", "g1", "g2"])
    h["p"], h["day"], h["sidx"] = m[me].to_numpy(), m["day"].to_numpy(), m["sidx"].to_numpy()
    h["grp"] = m[f"{me}_grp"].to_numpy()
    per = h.groupby(["p", "day", "sidx"], sort=False)[["g0", "g1", "g2"]].sum().reset_index()
    qd, qs = q["day"].to_numpy(), q["sidx"].to_numpy()
    S = ew_asof(per["p"].to_numpy(), per["day"].to_numpy(), per["sidx"].to_numpy(),
                per[["g0", "g1", "g2"]].to_numpy(float), q["player_id"].to_numpy(), qd, qs,
                half_life=H_D, rho=RHO_D)
    lgg = h.groupby(["grp", "day", "sidx"], sort=False)[["g0", "g1", "g2"]].sum().reset_index()
    qg = np.clip(q["grp"].to_numpy(int), 0, 3)
    L = ew_asof(lgg["grp"].to_numpy(), lgg["day"].to_numpy(), lgg["sidx"].to_numpy(),
                lgg[["g0", "g1", "g2"]].to_numpy(float), qg, qd, qs, half_life=H_D, rho=RHO_D)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = L / L.sum(1, keepdims=True)
    lp = np.where(np.isfinite(lp), lp, 1.0 / 3.0)
    return (S + K_A * lp) / (S.sum(1, keepdims=True) + K_A)


def pair_history(h: History) -> pd.DataFrame:
    """Ordered pairs (a attacker, b defender) of every history game where both played on
    opposite teams: D = minutes b guarded a, Toff = a's total offensive matchup minutes,
    Tdef = b's total defensive matchup minutes (in that game)."""
    r = h.rosters
    x = r.merge(r[["game_id", "team", "player_id"]].rename(
        columns={"team": "team_b", "player_id": "b"}), on="game_id")
    x = x[x["team"] != x["team_b"]].rename(columns={"player_id": "a"})
    m = h.m
    D = m.groupby(["game_id", "off", "def"], sort=False)["mmin"].sum().rename("D")
    toff = m.groupby(["game_id", "off"], sort=False)["mmin"].sum().rename("Toff")
    tdef = m.groupby(["game_id", "def"], sort=False)["mmin"].sum().rename("Tdef")
    x = x.merge(D.reset_index().rename(columns={"off": "a", "def": "b"}),
                on=["game_id", "a", "b"], how="left")
    x = x.merge(toff.reset_index().rename(columns={"off": "a"}), on=["game_id", "a"], how="left")
    x = x.merge(tdef.reset_index().rename(columns={"def": "b"}), on=["game_id", "b"], how="left")
    x[["D", "Toff", "Tdef"]] = x[["D", "Toff", "Tdef"]].fillna(0.0)
    # games without any matchup data contribute nothing
    has = x["game_id"].isin(set(m["game_id"]))
    return x[has][["game_id", "a", "b", "day", "sidx", "D", "Toff", "Tdef"]].reset_index(
        drop=True)


def _pair_key(a, b) -> np.ndarray:
    return np.asarray(a, np.int64) * 100_000_000 + np.asarray(b, np.int64)


# ============================================================================== 4. features

def features(h: History, rows: pd.DataFrame, *, ph: pd.DataFrame | None = None,
             dq: pd.DataFrame | None = None) -> pd.DataFrame:
    """F5 per (game_id, player_id) of `rows`: game_id, team, player_id, day, sidx, grp, w (the
    pre-game minutes weight; 0 = not playing). Every player of both teams of each game must be
    in `rows` (they form each other's rosters). History from `h` strictly before each day."""
    rows = rows.reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=range(len(rows)), columns=list(TEAM_COLS))
    out.insert(0, "game_id", rows["game_id"].to_numpy())
    out.insert(1, "player_id", rows["player_id"].to_numpy())
    if not len(rows) or not len(h.m):
        return out
    ph = ph if ph is not None else pair_history(h)
    dq = dq if dq is not None else _expected(h.m)
    q = rows[["player_id", "day", "sidx", "grp"]]
    # per player-day: defender quality, attacker tendencies, profiles
    qual = defender_quality(dq, q)
    att = attacker_rates(h.m, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        o_rel = {x: np.log(att[x].to_numpy() / att[f"lg_{x}"].to_numpy()) for x in R_X}
    A = _profiles(h.m, q, "def")          # as a defender: time vs attacker groups
    B = _profiles(h.m, q, "off")          # as an attacker: time vs defender groups
    # era: any league matchup data before the day
    era = np.isfinite(att["lg_pts"].to_numpy()) & (att["lg_pts"].to_numpy() > 0)
    # ordered pairs of the query games (a attacker, b defender), opposite teams
    r = rows[["game_id", "team", "player_id", "day", "sidx", "grp", "w"]].copy()
    r["_i"] = np.arange(len(r))
    x = r.merge(r[["game_id", "team", "_i"]].rename(columns={"team": "team_b", "_i": "_j"}),
                on="game_id")
    x = x[x["team"] != x["team_b"]].reset_index(drop=True)
    ia, jb = x["_i"].to_numpy(), x["_j"].to_numpy()
    pa, pb = rows["player_id"].to_numpy()[ia], rows["player_id"].to_numpy()[jb]
    w = rows["w"].to_numpy(float)
    ga = np.clip(rows["grp"].to_numpy(int)[ia], 0, 3)
    hk = _pair_key(ph["a"], ph["b"])
    Sd = ew_asof(hk, ph["day"].to_numpy(), ph["sidx"].to_numpy(),
                 ph[["D", "Toff", "Tdef"]].to_numpy(float), _pair_key(pa, pb),
                 x["day"].to_numpy(), x["sidx"].to_numpy(), half_life=H_D, rho=RHO_D)
    # reverse direction: b attacks, a defends -> history pair (attacker b, defender a); its
    # Tdef is a's total defensive matchup time
    Sr =ew_asof(hk, ph["day"].to_numpy(), ph["sidx"].to_numpy(),
                 ph[["D", "Tdef"]].to_numpy(float), _pair_key(pb, pa),
                 x["day"].to_numpy(), x["sidx"].to_numpy(), half_life=H_D, rho=RHO_D)
    key = x["game_id"].astype(str).to_numpy() + "|" + pa.astype(str)
    # attacker view: a attacked by b
    Ab = A[jb, np.minimum(ga, 2)]
    Ab = np.where(ga == 3, 1.0 / 3.0, Ab)
    pr = w[jb] * Ab
    pr = pr / np.maximum(pd.Series(pr).groupby(key).transform("sum").to_numpy(), 1e-12)
    s = (Sd[:, 0] + LAMBDA * pr) / (Sd[:, 1] + LAMBDA)
    s = np.where(w[jb] > 0, s, 0.0)
    s = s / np.maximum(pd.Series(s).groupby(key).transform("sum").to_numpy(), 1e-12)
    # defender view: a guards b (b attacks); prior ∝ w_b x B_b(grp_a)
    Bb = B[jb, np.minimum(ga, 2)]
    Bb = np.where(ga == 3, 1.0 / 3.0, Bb)
    pr2 = w[jb] * Bb
    pr2 = pr2 / np.maximum(pd.Series(pr2).groupby(key).transform("sum").to_numpy(), 1e-12)
    s2 = (Sr[:, 0] + LAMBDA * pr2) / (Sr[:, 1] + LAMBDA)
    s2 = np.where(w[jb] > 0, s2, 0.0)
    s2 = s2 / np.maximum(pd.Series(s2).groupby(key).transform("sum").to_numpy(), 1e-12)
    agg = pd.DataFrame({"_i": ia})
    for k in DQ:
        agg[f"f5_q_{k}"] = s * np.nan_to_num(qual[k].to_numpy()[jb])
    for k in R_X:
        agg[f"f5_r_{k}"] = s2 * np.nan_to_num(o_rel[k][jb])
    agg["_T"] = Sd[:, 1]
    g = agg.groupby("_i", sort=False)
    res = g.sum()
    # the primary defender: highest exposure
    xs = pd.DataFrame({"_i": ia, "s": s, "qp": np.nan_to_num(qual["pts"].to_numpy()[jb])})
    top = xs.sort_values(["_i", "s"], ascending=[True, False], kind="stable").drop_duplicates("_i")
    top = top.set_index("_i")
    idx = res.index.to_numpy()
    for c in Q_COLS + R_COLS:
        out.loc[idx, c] = res[c].to_numpy()
    out.loc[idx, "f5_ln_dir"] = np.log1p(np.maximum(res["_T"].to_numpy(), 0.0))
    out.loc[top.index.to_numpy(), "f5_prim_share"] = top["s"].to_numpy()
    out.loc[top.index.to_numpy(), "f5_prim_pts"] = top["qp"].to_numpy()
    out.loc[~era, list(TEAM_COLS)] = np.nan
    return out


def build(conn: sqlite3.Connection, rows: pd.DataFrame, *, h: History | None = None,
          pos=None, log=None) -> pd.DataFrame:
    """F5 for frame rows (game_id, team, player_id, day, sidx, grp, mpg): who played in each
    game is the opponent roster (the avail oracle), weights = pre-game mpg."""
    say = log or (lambda m: None)
    if h is None:
        last = ST.season_label(int(rows["sidx"].max()))
        h = History(conn, last=last, pos=pos)
        say(f"F5 history: {len(h.m)} matchup rows, {len(h.rosters)} roster rows")
    r = rows[["game_id", "team", "player_id", "day", "sidx", "grp", "mpg"]].drop_duplicates(
        ["game_id", "player_id"]).reset_index(drop=True)
    r["w"] = np.maximum(r["mpg"].to_numpy(float), 1.0)
    return features(h, r.drop(columns="mpg"))


def columns(stat: str) -> list[str]:
    return [f"f5_{c}" for c in PER_STAT[stat]] + ["f5_ln_dir"]


def monotone(stat: str) -> dict:
    """Facing weaker defenders (allowed > expected) can only raise the stat."""
    return {OWN_MONO[stat]: 1} if stat in OWN_MONO else {}
