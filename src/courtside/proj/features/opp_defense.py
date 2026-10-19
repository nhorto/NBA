"""F2: opponent defense per stat x position group (docs/PLAYERS.md Phase 2F, §4.3 "opponent").

Three blocks, all strictly before the game date (`common.ew_asof`), recency-weighted with a
per-season carry-over, and shrunk:

1. **Residual DvP index** per (opponent O, position group g, stat s):
       l        = (Y_L + 1) / (E_L + 1)                  the whole league (drift vs the EB state)
       idx_team = (Y_O + K_t l) / (E_O + K_t)
       idx_g    = (Y_Og + K_g idx_team) / (E_Og + K_g)
   and the features are log(idx / l): only between-opponent differences (league drift is an
   EB-state matter, not opponent information).
   Y = weighted sum of what O's opponents actually produced, E = what our pre-game EB model
   expected of those same players given their actual minutes (`common.base_rows` E_s). So the
   index is "allowed vs expected", which controls for who O happened to face and for the league
   era. Shrinkage: g -> team-level -> league. K_t and K_g (in team-games of expected count)
   and the half-life are frozen on 2010-11 -> 2013-14 (§7.1 L2, box-path hyperparameters) by the
   NB log-likelihood of y given E x idx_g (`fit_dvp_params`; `DVP_PARAMS`).
   Columns: f2_dvp_<s>_<G|F|C|all> (log idx); row level f2_dvp_<s>_grp (the player's group).
2. **Team box rates** (log of O's shrunk rate / the league's season-to-date rate):
   allowed: offensive-rebound % allowed, 3PA rate, FTA rate, assisted-FG rate, turnovers forced
   per possession, blocks per opponent 2PA, steals per possession, 3P% / 2P% allowed (3P% with
   K = 5,000 attempts), points per possession allowed; O's own offense: offensive-rebound %,
   turnovers per possession, 2PA and misses per possession; O's pace.
3. **Pace beyond the market total:** f2_pace_beyond = mean(log pace_T, log pace_O) (vs league)
   - log(closing total / league mean total to date). The total itself is not a feature (the
   pipeline already scales to it).
"""

from __future__ import annotations

import sqlite3
from typing import Sequence

import numpy as np
import pandas as pd

from .. import state as ST
from .common import ew_asof, league_asof, load_team_games

DVP_STATS = ("pts", "fg3m", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "fta")
GRPS = ("G", "F", "C")

# frozen on 2010-11 .. 2013-14 by `fit_dvp_params` (python -m courtside.proj.context dvp-params)
DVP_PARAMS: dict = {
    "fit_seasons": ["2010-11", "2013-14"], "H": 120.0, "rho": 1.0, "base": "E (market-reconciled)",
    "shrink_to": "league allowed/expected ratio at the date",
    "k_team": {"pts": 1e9, "fg3m": 20.0, "reb": 20.0, "oreb": 40.0, "dreb": 20.0, "ast": 10.0,
               "stl": 20.0, "blk": 5.0, "tov": 20.0, "fta": 10.0},
    "k_grp": {"pts": 40.0, "fg3m": 160.0, "reb": 80.0, "oreb": 1e9, "dreb": 80.0, "ast": 160.0,
              "stl": 160.0, "blk": 160.0, "tov": 80.0, "fta": 80.0},
    # mean expected count per team-game / per team-game-group on the fit seasons (K units)
    "scale": {"pts": [98.851, 33.667], "fg3m": [6.8571, 2.3354], "reb": [42.135, 14.351],
              "oreb": [11.061, 3.7672], "dreb": [31.076, 10.584], "ast": [21.622, 7.3641],
              "stl": [7.4335, 2.5318], "blk": [4.9249, 1.6774], "tov": [13.838, 4.7129],
              "fta": [23.765, 8.094]},
    # NB log-lik gain per row of y | E x idx_grp on the fit seasons (the standalone signal)
    "ll_gain_per_row": {"pts": 0.00051, "fg3m": 0.00176, "reb": 0.00156, "oreb": 0.0005,
                        "dreb": 0.00174, "ast": 0.00256, "stl": 0.00089, "blk": 0.00438,
                        "tov": 0.00221, "fta": 0.00266},
}

BOX_H, BOX_RHO, BOX_K_GAMES, LEAGUE_M0 = 90.0, 0.7, 10.0, 200.0
# (name, numerator, denominator, K override in denominator units or None)
BOX_RATES = (
    ("def_oreb", lambda t: t.o_oreb, lambda t: t.o_oreb + t.dreb, None),
    ("def_fg3a", lambda t: t.o_fg3a, lambda t: t.o_fga, None),
    ("def_fta", lambda t: t.o_fta, lambda t: t.o_fga, None),
    ("def_ast", lambda t: t.o_ast, lambda t: t.o_fgm, None),
    ("def_tov", lambda t: t.o_tov, lambda t: t.o_poss, None),
    ("def_blk", lambda t: t.blk, lambda t: t.o_fga - t.o_fg3a, None),
    ("def_stl", lambda t: t.stl, lambda t: t.o_poss, None),
    ("def_fg3p", lambda t: t.o_fg3m, lambda t: t.o_fg3a, 5000.0),
    ("def_fg2p", lambda t: t.o_fgm - t.o_fg3m, lambda t: t.o_fga - t.o_fg3a, 3000.0),
    ("def_ppp", lambda t: t.o_pts, lambda t: t.o_poss, None),
    ("off_oreb", lambda t: t.oreb, lambda t: t.oreb + t.o_dreb, None),
    ("off_tov", lambda t: t.tov, lambda t: t.poss, None),
    ("off_fg2a", lambda t: t.fga - t.fg3a, lambda t: t.poss, None),
    ("off_miss", lambda t: t.fga - t.fgm, lambda t: t.poss, None),
    ("pace", lambda t: t.poss, lambda t: t["min"] / 240.0, None),
)
BOX_COLS = tuple(f"f2_o_{n}" for n, *_ in BOX_RATES)
TEAM_COLS = BOX_COLS + ("f2_pace_beyond",) + tuple(
    f"f2_dvp_{s}_{g}" for s in DVP_STATS for g in GRPS + ("all",))
ROW_COLS = tuple(f"f2_dvp_{s}_grp" for s in DVP_STATS)

# per model stat: which DvP indices it sees (its own, plus the components that drive it)
DVP_FOR = {"pts": ("pts", "fta", "fg3m"), "fg3m": ("fg3m",), "reb": ("reb", "oreb", "dreb"),
           "ast": ("ast",), "stl": ("stl",), "blk": ("blk",), "tov": ("tov",)}


# ============================================================================== DvP

def _dvp_history(wf: pd.DataFrame, stats: Sequence[str]):
    """History rows aggregated per (defending team, group, day): Y and E per stat."""
    h = wf[["opp", "grp", "day", "sidx"] + [s for s in stats] + [f"E_{s}" for s in stats]]
    h = h.rename(columns={"opp": "def"})
    cols = list(stats) + [f"E_{s}" for s in stats]
    grp = h.groupby(["def", "grp", "day", "sidx"], sort=False)[cols].sum().reset_index()
    team = h.groupby(["def", "day", "sidx"], sort=False)[cols].sum().reset_index()
    return grp, team


def dvp_sums(wf: pd.DataFrame, queries: pd.DataFrame, H: float, rho: float,
             stats: Sequence[str] = DVP_STATS) -> dict:
    """Weighted Y / E sums before each query (opp, day, sidx) for the league, the opponent and
    the opponent per group: {'league' | 'team' | 'G' | 'F' | 'C': (nq, 2 * ns)} (Y columns
    first, then E)."""
    hg, ht = _dvp_history(wf, stats)
    cols = list(stats) + [f"E_{s}" for s in stats]
    qk, qd, qs = (queries["opp"].to_numpy(), queries["day"].to_numpy(),
                  queries["sidx"].to_numpy())
    hl = ht.groupby(["day", "sidx"], sort=False)[cols].sum().reset_index()
    out = {"league": ew_asof(np.zeros(len(hl), int), hl["day"].to_numpy(), hl["sidx"].to_numpy(),
                             hl[cols].to_numpy(float), np.zeros(len(qk), int), qd, qs,
                             half_life=H, rho=rho),
           "team": ew_asof(ht["def"].to_numpy(), ht["day"].to_numpy(), ht["sidx"].to_numpy(),
                           ht[cols].to_numpy(float), qk, qd, qs, half_life=H, rho=rho)}
    for g, name in enumerate(GRPS):
        d = hg[hg["grp"] == g]
        out[name] = ew_asof(d["def"].to_numpy(), d["day"].to_numpy(), d["sidx"].to_numpy(),
                            d[cols].to_numpy(float), qk, qd, qs, half_life=H, rho=rho)
    return out


def _per_game_scale(wf: pd.DataFrame, stats) -> dict:
    """Mean expected count per team-game (team level) and per team-game-group, per stat: the
    unit of the K's ('K team-games of evidence')."""
    tg = wf.groupby(["game_id", "team"])[[f"E_{s}" for s in stats]].sum()
    tgg = wf[wf["grp"] < 3].groupby(["game_id", "team", "grp"])[[f"E_{s}" for s in stats]].sum()
    return {s: (float(tg[f"E_{s}"].mean()), float(tgg[f"E_{s}"].mean())) for s in stats}


def _levels(sums: dict, j: int, ns: int, kt: float, kg: float, grp_idx=None):
    """(log league ratio, log team index, {group: log group index}) in absolute terms:
    league l = (Y_L + 1) / (E_L + 1); team = (Y_O + kt l) / (E_O + kt); group shrinks to team."""
    YL, EL = sums["league"][:, j], sums["league"][:, ns + j]
    lg = (YL + 1.0) / (EL + 1.0)
    Y, E = sums["team"][:, j], sums["team"][:, ns + j]
    it = (Y + kt * lg) / (E + kt)
    gi = {}
    for g in GRPS:
        Yg, Eg = sums[g][:, j], sums[g][:, ns + j]
        gi[g] = np.log((Yg + kg * it) / (Eg + kg))
    return np.log(lg), np.log(it), gi


def dvp_indices(sums: dict, scale: dict, k_team: dict, k_grp: dict,
                stats: Sequence[str] = DVP_STATS) -> pd.DataFrame:
    """Opponent indices RELATIVE to the league's own allowed-vs-expected ratio at that date, so
    they carry only between-opponent differences (a league-wide drift of a stat vs the EB state
    is not opponent information and is left out of F2)."""
    ns = len(stats)
    out = {}
    for j, s in enumerate(stats):
        ll, lt, gi = _levels(sums, j, ns, k_team[s] * scale[s][0], k_grp[s] * scale[s][1])
        out[f"f2_dvp_{s}_all"] = lt - ll
        for g in GRPS:
            out[f"f2_dvp_{s}_{g}"] = gi[g] - ll
    return pd.DataFrame(out)


K_GRID = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 1e9)
H_GRID = ((60.0, 0.5), (120.0, 0.5), (240.0, 0.5), (120.0, 1.0), (365.0, 0.3))


def fit_dvp_params(wf: pd.DataFrame, fit_seasons: Sequence[str], *, log=None) -> dict:
    """Choose (H, rho) and per stat (K_team, K_grp) by the NB log-likelihood of y given
    E x idx_grp on regular-season rows of `fit_seasons` (each row's index uses only earlier
    games). `wf` must hold the fit seasons plus at least one earlier season of history."""
    say = log or (lambda m: None)
    fs = sorted(fit_seasons)
    rows = wf[(wf["season"] >= fs[0]) & (wf["season"] <= fs[-1])
              & (wf["season_type"] == "Regular Season") & (wf["grp"] < 3)].reset_index(drop=True)
    q = rows[["opp", "day", "sidx"]]
    scale = _per_game_scale(wf, DVP_STATS)
    best: dict = {s: (-np.inf, None) for s in DVP_STATS}
    base_ll = {}
    g = rows["grp"].to_numpy(int)
    for H, rho in H_GRID:
        sums = dvp_sums(wf, q, H, rho)
        ns = len(DVP_STATS)
        for j, s in enumerate(DVP_STATS):
            y = rows[s].to_numpy(float)
            E = np.maximum(rows[f"E_{s}"].to_numpy(float), 1e-9)
            d = rows[f"d_{s}"].to_numpy(float)
            if s not in base_ll:
                base_ll[s] = float(ST.nb_logpmf(y, E, d).sum())
            for kt in K_GRID:
                for kg in K_GRID:
                    _, _, gi = _levels(sums, j, ns, kt * scale[s][0], kg * scale[s][1])
                    idx = np.exp(np.column_stack([gi[n] for n in GRPS])[np.arange(len(g)), g])
                    ll = float(ST.nb_logpmf(y, E * idx, d).sum())
                    if ll > best[s][0]:
                        best[s] = (ll, (H, rho, kt, kg))
        say(f"dvp grid H={H} rho={rho} done")
    # one (H, rho) for all stats: the most common winner, then re-pick K's at that (H, rho)
    hr = pd.Series([best[s][1][:2] for s in DVP_STATS]).value_counts().index[0]
    out = {"fit_seasons": [fs[0], fs[-1]], "H": float(hr[0]), "rho": float(hr[1]),
           "k_team": {}, "k_grp": {}, "ll_gain_per_row": {}}
    sums = dvp_sums(wf, q, hr[0], hr[1])
    ns = len(DVP_STATS)
    for j, s in enumerate(DVP_STATS):
        y = rows[s].to_numpy(float)
        E = np.maximum(rows[f"E_{s}"].to_numpy(float), 1e-9)
        d = rows[f"d_{s}"].to_numpy(float)
        bb = (-np.inf, None)
        ll0 = None
        for kt in K_GRID:
            for kg in K_GRID:
                lgl, _, gi = _levels(sums, j, ns, kt * scale[s][0], kg * scale[s][1])
                idx = np.exp(np.column_stack([gi[n] for n in GRPS])[np.arange(len(g)), g])
                ll = float(ST.nb_logpmf(y, E * idx, d).sum())
                if ll0 is None:
                    ll0 = float(ST.nb_logpmf(y, E * np.exp(lgl), d).sum())
                if ll > bb[0]:
                    bb = (ll, (kt, kg))
        out["k_team"][s], out["k_grp"][s] = float(bb[1][0]), float(bb[1][1])
        out["ll_gain_per_row"][s] = (bb[0] - base_ll[s]) / len(rows)
        out.setdefault("ll_gain_league_only", {})[s] = (ll0 - base_ll[s]) / len(rows)
    out["n_rows"] = int(len(rows))
    out["scale"] = {s: [float(v[0]), float(v[1])] for s, v in scale.items()}
    return out


# ============================================================================== box rates

def box_rates(t: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Each query team's shrunk pre-game box rates relative to the league (log ratio).
    `t` = `common.load_team_games` history; queries: team, day, sidx. Columns = rate names."""
    num = np.column_stack([np.asarray(f(t), float) for _, f, _, _ in BOX_RATES])
    den = np.column_stack([np.asarray(g(t), float) for _, _, g, _ in BOX_RATES])
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = np.where(ok, num, 0.0), np.where(ok, den, 0.0)
    pair = np.empty((len(t), 2 * len(BOX_RATES)))
    pair[:, 0::2], pair[:, 1::2] = num, den
    S = ew_asof(t["team"].to_numpy(), t["day"].to_numpy(), t["sidx"].to_numpy(),
                pair, queries["team"].to_numpy(), queries["day"].to_numpy(),
                queries["sidx"].to_numpy(), half_life=BOX_H, rho=BOX_RHO)
    # league rate and league denominator per team-game, season to date (causal)
    pair2 = np.empty((len(t), 4 * len(BOX_RATES)))
    pair2[:, 0::4], pair2[:, 1::4] = num, den
    pair2[:, 2::4], pair2[:, 3::4] = den, 1.0
    L2 = league_asof(t["day"].to_numpy(), t["sidx"].to_numpy(), pair2,
                     queries["day"].to_numpy(), queries["sidx"].to_numpy(), LEAGUE_M0)
    out = {}
    for j, (name, _, _, kover) in enumerate(BOX_RATES):
        lg, den_pg = L2[:, 2 * j], L2[:, 2 * j + 1]
        K = kover if kover is not None else BOX_K_GAMES * den_pg
        r = (S[:, 2 * j] + K * lg) / (S[:, 2 * j + 1] + K)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.log(r / lg)
    return pd.DataFrame(out)


def closing_totals(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT game_id, AVG(total_close) AS mu_total FROM closing_lines WHERE total_close "
        "IS NOT NULL GROUP BY game_id", conn)


# ============================================================================== build

def build(conn: sqlite3.Connection, wf: pd.DataFrame, *, params: dict | None = None,
          t: pd.DataFrame | None = None, queries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Team-game F2 table: one row per (game_id, team) of `queries` (default: the team-games of
    `wf`) = the team the features are FOR; every value describes its opponent, except the pace
    term. `wf` = `common.base_rows` history (only rows dated before each query day are used),
    `t` = `common.load_team_games` history likewise; queries: game_id, team, opp, day, sidx."""
    p = params or DVP_PARAMS
    first, last = wf["season"].min(), wf["season"].max()
    if t is None:
        t = load_team_games(conn, first, last)
    if queries is None:
        queries = wf[["game_id", "team", "opp", "day", "sidx"]]
    q = queries[["game_id", "team", "opp", "day", "sidx"]].drop_duplicates(["game_id", "team"])
    q = q.reset_index(drop=True)
    sums = dvp_sums(wf, q, p["H"], p["rho"])
    scale = ({s: tuple(v) for s, v in p["scale"].items()} if "scale" in p
             else _per_game_scale(wf, DVP_STATS))
    dvp = dvp_indices(sums, scale, p["k_team"], p["k_grp"])
    # box rates of every team before each queried game (both sides), then attach the opponent's
    tq = pd.concat([q[["game_id", "team", "day", "sidx"]],
                    q[["game_id", "opp", "day", "sidx"]].rename(columns={"opp": "team"})],
                   ignore_index=True).drop_duplicates(["game_id", "team"]).reset_index(drop=True)
    br = box_rates(t, tq)
    br.insert(0, "game_id", tq["game_id"].to_numpy())
    br.insert(1, "team", tq["team"].to_numpy())
    ob = br.rename(columns={"team": "opp", **{n: f"f2_o_{n}" for n, *_ in BOX_RATES}})
    out = pd.concat([q[["game_id", "team", "opp", "day", "sidx"]], dvp], axis=1)
    out = out.merge(ob, on=["game_id", "opp"], how="left")
    own = br[["game_id", "team", "pace"]].rename(columns={"pace": "_own_pace"})
    out = out.merge(own, on=["game_id", "team"], how="left")
    # pace beyond the closing total
    tot = t[["game_id", "team", "day", "sidx"]].copy()
    tot["pts2"] = (t["pts"] + t["o_pts"]).to_numpy(float)
    tot = tot.drop_duplicates("game_id")
    lt = league_asof(tot["day"].to_numpy(), tot["sidx"].to_numpy(),
                     np.column_stack([tot["pts2"], np.ones(len(tot))]),
                     out["day"].to_numpy(), out["sidx"].to_numpy(), LEAGUE_M0)[:, 0]
    out = out.merge(closing_totals(conn), on="game_id", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f2_pace_beyond"] = (0.5 * (out["_own_pace"] + out["f2_o_pace"])
                                 - np.log(out["mu_total"] / lt))
    out = out.drop(columns=["opp", "day", "sidx", "_own_pace", "mu_total"])
    return out[["game_id", "team", *TEAM_COLS]].reset_index(drop=True)


def row_features(rows: pd.DataFrame) -> pd.DataFrame:
    """f2_dvp_<s>_grp: the index of the player's own group (team level when unknown)."""
    g = np.clip(rows["grp"].to_numpy(int), 0, 3)
    out = {}
    for s in DVP_STATS:
        m = np.column_stack([rows[f"f2_dvp_{s}_{n}"] for n in GRPS + ("all",)])
        out[f"f2_dvp_{s}_grp"] = m[np.arange(len(rows)), g]
    return pd.DataFrame(out, index=rows.index)


def columns(stat: str) -> list[str]:
    dv = []
    for s in DVP_FOR[stat]:
        dv += [f"f2_dvp_{s}_grp", f"f2_dvp_{s}_all"]
    return dv + list(BOX_COLS) + ["f2_pace_beyond"]


def monotone(stat: str) -> dict:
    """The stat's own DvP index can only raise its rate (allowed vs expected)."""
    return {f"f2_dvp_{stat}_grp": 1, f"f2_dvp_{stat}_all": 1}
