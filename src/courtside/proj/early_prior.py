"""Early-season and role-change priors for the per-minute rate state (`+ep`, IMPROVEMENTS P3).

Switch `COURTSIDE_EARLY_PRIOR` (default OFF; `pipeline.set_early_prior`). ON = bundle suffix
`_ep` (the arm's bundle with `state["early_prior"]` fit on seasons < S, `fit`) and model
version `<arm>+ep`; seeds unchanged. OFF: nothing of it runs and `state.rate_frame` is the
base code path (bit-identical).

The EB rate (`state._rates`) is r = (Sx + kappa mu) / (Sn + kappa) over decayed past games.
`+ep` splits the sums into the current season's games (cur) and every earlier season's (prev)
and adds the current season's preseason games (pre)::

    r = (Xcur + w Xpre + g d Xprev + kappa m mu) / (Ncur + w Npre + g Nprev + kappa)

* **d = age delta** per (stat, season-age bucket): last season's per-minute rate carried into
  this season with the development / decline of a player of that age (age on Feb 1 of the
  season, from `team_rosters.birth_date`; unknown age: 1).
* **w = preseason weight** per stat: the season's preseason games (`preseason_player_games`,
  NBA teams, no scrimmages, no games against non-NBA teams, before the first regular-season
  day) count as w of a regular-season game, decayed like one (half-life H of the stat).
* **m = rookie prior** per (stat, draft bucket): a player in his first NBA season (no
  `player_games` row in an earlier season) has his cell prior mu scaled by m (draft slot from
  `player_index`; undrafted is its own bucket).
* **g = role-change weight** per (stat, ratio bucket): once a player has >= ROLE_MIN_GP games
  this season, his current-season mpg over last season's (>= ROLE_MIN_PREV_GP games) picks a
  bucket; outside the neutral bucket the earlier seasons' evidence counts g (< 1 = faster
  shrinkage toward this season's games).

All four are fit by `fit` on seasons < S (maximum NB log-likelihood of each training game's
count given its actual minutes and the minutes table, as `state.fit_params`, with the base
H / rho / kappa / phi of S held fixed), coordinate-wise on grids. Neutral values (d = g = m =
1, w = 0) give the base rates up to float rounding (the cur + prev split). `n_<stat>` and
`p_<stat>` of the output stay the base values (inputs of the context layer); only `r_<stat>`
(and `r_dk`) change.

Public API
  EARLY_PRIOR, set_early_prior(on), EARLY_PRIOR_SUFFIX / EARLY_PRIOR_TAG
  load_data(conn) -> EPData            bio (birth day, first NBA season, draft pick), preseason
  EPInput(params, data)                what `state.rate_frame(..., ep=EPInput)` takes
  rates(fr, params, ep) -> {stat: (r, sn, prior)}   (state._rates contract)
  fit(conn, season, sp, data=None, games=None, pos=None, log=None) -> early_prior params
  neutral_params() -> params that reproduce the base rates
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import state as ST

EARLY_PRIOR_DEFAULT = True             # shipped 2026-10-24 (owner OK; env still overrides)
EARLY_PRIOR_ENV = "COURTSIDE_EARLY_PRIOR"
EARLY_PRIOR_SUFFIX = "_ep"
EARLY_PRIOR_TAG = "+ep"
VERSION = 1

# season age (on Feb 1 of the season): <21, 21, 22, 23, 24, 25, 26-27, 28-29, 30-31, 32-33, 34+
AGE_EDGES = (21, 22, 23, 24, 25, 26, 28, 30, 32, 34)
N_AGE = len(AGE_EDGES) + 1             # + 1 more index (N_AGE) = unknown age (d = 1)
# rookie draft buckets: picks 1-3, 4-10, 11-20, 21-30, 31+ and undrafted / unknown
DRAFT_EDGES = (4, 11, 21, 31)
N_DRAFT = len(DRAFT_EDGES) + 2
# role ratio = current-season mpg / last season's mpg: < 0.7, neutral, 1.25-1.6, >= 1.6
ROLE_EDGES = (0.7, 1.25, 1.6)
ROLE_NEUTRAL = 1
N_ROLE = len(ROLE_EDGES) + 1
ROLE_MIN_GP = 2
ROLE_MIN_PREV_GP = 10
FIRST_BIO_SEASON = "2015-16"           # team_rosters (birth dates) start here

D_GRID = tuple(np.round(np.arange(0.80, 1.2501, 0.01), 2))
W_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0)
V_GRID = tuple(np.round(np.arange(0.70, 1.1501, 0.025), 3))
M_GRID = tuple(np.round(np.arange(0.50, 1.5001, 0.02), 2))
G_GRID = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1)
MIN_ROWS = 1500                        # a cell with fewer training rows keeps its neutral value
FIT_SEASONS = 5                        # seasons < S used (>= FIRST_BIO_SEASON)
# pieces `fit` turns on by default (the evidence: docs/research/early-prior.md); "d" (age delta,
# "c" = centred), "m" (rookie prior) and "g" (role change) stay neutral unless asked for
PARTS = ("w", "v")


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


EARLY_PRIOR = _flag(EARLY_PRIOR_ENV, EARLY_PRIOR_DEFAULT)


def set_early_prior(on: bool) -> None:
    global EARLY_PRIOR
    EARLY_PRIOR = bool(on)


# ============================================================================== data

@dataclass
class EPData:
    bio: pd.DataFrame          # player_id, birth_day (float, NaN unknown), first_sidx, pick
    pre: pd.DataFrame          # pid, day, sidx, min, STATS (preseason games kept)


@dataclass
class EPInput:
    params: dict               # state["early_prior"]
    data: EPData


def load_data(conn: sqlite3.Connection) -> EPData:
    """Player bio (birth day from any `team_rosters` row, first season with a `player_games`
    row, draft pick from `player_index`) and the preseason games `+ep` counts."""
    br = pd.read_sql_query("SELECT player_id, birth_date FROM team_rosters "
                           "WHERE birth_date IS NOT NULL AND birth_date != ''", conn)
    br = br.drop_duplicates("player_id")
    br["birth_day"] = ST._to_day(br["birth_date"]).astype(float) if len(br) else []
    fs = pd.read_sql_query("SELECT player_id, MIN(season) AS s FROM player_games WHERE min > 0 "
                           "GROUP BY player_id", conn)
    fs["first_sidx"] = fs["s"].str[:4].astype(int)
    pi = pd.read_sql_query("SELECT player_id, draft_number FROM player_index", conn)
    pi = pi.drop_duplicates("player_id").rename(columns={"draft_number": "pick"})
    bio = (fs[["player_id", "first_sidx"]]
           .merge(br[["player_id", "birth_day"]], on="player_id", how="outer")
           .merge(pi, on="player_id", how="outer"))
    bio = bio[bio["player_id"].notna()].astype({"player_id": np.int64}).reset_index(drop=True)
    return EPData(bio=bio, pre=load_preseason(conn))


def load_preseason(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = ["pid", "day", "sidx", "min", *ST.STATS]
    try:
        pl = pd.read_sql_query(
            "SELECT player_id, season, game_date, min, " + ", ".join(ST.STATS)
            + " FROM preseason_player_games WHERE scrimmage = 0 AND COALESCE(vs_non_nba, 0) = 0"
            " AND min > 0 AND team IN (SELECT DISTINCT team FROM team_games)", conn)
        rs = pd.read_sql_query("SELECT season, MIN(game_date) AS d0 FROM team_games WHERE "
                               "season_type = 'Regular Season' GROUP BY season", conn)
    except Exception:  # pragma: no cover - DB before migration 15
        return pd.DataFrame(columns=cols)
    if not len(pl):
        return pd.DataFrame(columns=cols)
    pl = pl.merge(rs, on="season", how="left")
    pl = pl[pl["d0"].isna() | (pl["game_date"] < pl["d0"])]
    out = pd.DataFrame({"pid": pl["player_id"].astype(np.int64).to_numpy(),
                        "day": ST._to_day(pl["game_date"]),
                        "sidx": pl["season"].str[:4].astype(int).to_numpy(),
                        "min": pd.to_numeric(pl["min"], errors="coerce").fillna(0).to_numpy(float)})
    for s in ST.STATS:
        out[s] = pd.to_numeric(pl[s], errors="coerce").fillna(0).to_numpy(float)
    return out.reset_index(drop=True)


# ============================================================================== params

def neutral_params() -> dict:
    return {"version": VERSION, "age_edges": list(AGE_EDGES), "draft_edges": list(DRAFT_EDGES),
            "role_edges": list(ROLE_EDGES), "role_min_gp": ROLE_MIN_GP,
            "role_min_prev_gp": ROLE_MIN_PREV_GP,
            "d": {s: [1.0] * (N_AGE + 1) for s in ST.STATS},
            "w": {s: 0.0 for s in ST.STATS},
            "v": {s: 1.0 for s in ST.STATS},
            "m": {s: [1.0] * N_DRAFT for s in ST.STATS},
            "g": {s: [1.0] * N_ROLE for s in ST.STATS},
            "parts": ["d", "w", "m", "g"], "fit_seasons": [], "diagnostics": {}}


def has_ep(state: dict | None) -> bool:
    return bool(state and state.get("early_prior"))


# ============================================================================== per-row inputs

@dataclass
class _Rows:
    """Per reported row of a state frame: the cur/prev split, buckets and preseason sums."""
    walk_c: ST._Walk           # walk over (player, season) blocks: current-season sums
    age_b: np.ndarray          # (n_rep,) age bucket (N_AGE = unknown)
    rookie: np.ndarray         # (n_rep,) bool
    draft_b: np.ndarray        # (n_rep,) draft bucket
    role_b: np.ndarray         # (n_rep,) role bucket (ROLE_NEUTRAL when not applicable)
    pre_key: pd.DataFrame      # (rep row, day lag, vals...) preseason games before each row


def _rows(fr: ST._Frame, params: dict, data: EPData) -> _Rows:
    ri = fr.rep_idx
    pid, day, sidx = fr.pid, fr.day, fr.sidx
    key = pid.astype(np.int64) * 10_000 + sidx.astype(np.int64)
    walk_c = ST._Walk(key, day, sidx)
    real = (~fr.isq).astype(float)
    # current-season games and minutes before each row (no decay), last season's mpg
    cnt = walk_c.sums(np.column_stack([fr.vals[:, 0], real]), np.zeros(len(pid)))[ri]
    d = pd.DataFrame({"pid": pid[~fr.isq], "sidx": sidx[~fr.isq], "min": fr.vals[~fr.isq, 0]})
    tot = d.groupby(["pid", "sidx"])["min"].agg(["sum", "size"]).reset_index()
    tot["sidx"] = tot["sidx"] + 1
    q = pd.DataFrame({"pid": pid[ri], "sidx": sidx[ri]})
    prev = q.merge(tot, on=["pid", "sidx"], how="left")
    prev_gp = prev["size"].fillna(0).to_numpy(float)
    prev_mpg = prev["sum"].fillna(0).to_numpy(float) / np.maximum(prev_gp, 1.0)
    cur_gp, cur_min = cnt[:, 1], cnt[:, 0]
    ok = ((cur_gp >= params.get("role_min_gp", ROLE_MIN_GP))
          & (prev_gp >= params.get("role_min_prev_gp", ROLE_MIN_PREV_GP)) & (prev_mpg > 0))
    ratio = np.where(ok, cur_min / np.maximum(cur_gp, 1.0) / np.maximum(prev_mpg, 1e-9), 1.0)
    role_b = np.where(ok, np.searchsorted(params.get("role_edges", ROLE_EDGES), ratio,
                                          side="right"), ROLE_NEUTRAL)
    # bio
    b = q[["pid"]].merge(data.bio.rename(columns={"player_id": "pid"}), on="pid", how="left")
    feb1 = np.array([ST._day_of(f"{s + 1}-02-01") for s in range(1990, 2040)], float)
    sx = sidx[ri].astype(int)
    age = (feb1[np.clip(sx - 1990, 0, len(feb1) - 1)] - b["birth_day"].to_numpy(float)) / 365.25
    age_b = np.where(np.isfinite(age),
                     np.searchsorted(params.get("age_edges", AGE_EDGES), np.floor(age),
                                     side="right"), N_AGE)
    fsx = b["first_sidx"].to_numpy(float)
    rookie = ~np.isfinite(fsx) | (fsx >= sx)
    pick = b["pick"].to_numpy(float)
    draft_b = np.where(np.isfinite(pick) & (pick > 0),
                       np.searchsorted(params.get("draft_edges", DRAFT_EDGES), pick,
                                       side="right"), N_DRAFT - 1)
    # preseason games of the row's season strictly before its day
    pk = pd.DataFrame(columns=["qi", "lag", "min", *ST.STATS])
    if len(data.pre):
        qq = pd.DataFrame({"qi": np.arange(len(ri)), "pid": pid[ri], "sidx": sidx[ri],
                           "qday": day[ri]})
        qq = qq[qq["pid"].isin(np.unique(data.pre["pid"]))]
        m = qq.merge(data.pre, on=["pid", "sidx"], how="inner")
        m = m[m["day"] < m["qday"]]
        m["lag"] = (m["qday"] - m["day"]).astype(float)
        pk = m[["qi", "lag", "min", *ST.STATS]].reset_index(drop=True)
    return _Rows(walk_c=walk_c, age_b=age_b.astype(int), rookie=rookie,
                 draft_b=draft_b.astype(int), role_b=role_b.astype(int), pre_key=pk)


def _pre_sums(rows: _Rows, n: int, h: float) -> np.ndarray:
    """(n_rep, 1 + len(STATS)) decayed preseason sums [min, stats] with half-life h."""
    out = np.zeros((n, 1 + len(ST.STATS)))
    pk = rows.pre_key
    if not len(pk):
        return out
    w = np.exp(-pk["lag"].to_numpy(float) * (ST.LN2 / h))
    v = pk[["min", *ST.STATS]].to_numpy(float) * w[:, None]
    qi = pk["qi"].to_numpy(int)
    for j in range(v.shape[1]):
        out[:, j] = np.bincount(qi, weights=v[:, j], minlength=n)
    return out


@dataclass
class _Parts:
    """Per stat, per reported row: the pieces of the `+ep` rate."""
    xc: np.ndarray
    nc: np.ndarray
    xp: np.ndarray
    np_: np.ndarray
    xs: np.ndarray             # preseason
    ns: np.ndarray
    mu: np.ndarray
    sn: np.ndarray             # base effective minutes (reported n_)


def _parts(fr: ST._Frame, params: dict, rows: _Rows) -> dict[str, _Parts]:
    ri = fr.rep_idx
    groups: dict[tuple[float, float], list[str]] = {}
    for s in ST.STATS:
        p = params["stats"][s]
        groups.setdefault((p["H"], p["rho"]), []).append(s)
    v = fr.vals
    out: dict[str, _Parts] = {}
    pre_cache: dict[float, np.ndarray] = {}
    for (h, rho), ss in groups.items():
        idx = [0] + [1 + ST.STATS.index(s) for s in ss]
        e = fr.walk.sums(v[:, idx], fr.walk.clock(h, rho))[ri]
        ec = rows.walk_c.sums(v[:, idx], rows.walk_c.clock(h, rho))[ri]
        ep = np.maximum(e - ec, 0.0)
        if h not in pre_cache:
            pre_cache[h] = _pre_sums(rows, len(ri), h)
        ps = pre_cache[h]
        for j, s in enumerate(ss):
            k = ST.STATS.index(s)
            out[s] = _Parts(xc=ec[:, 1 + j], nc=ec[:, 0], xp=ep[:, 1 + j], np_=ep[:, 0],
                            xs=ps[:, 1 + k], ns=ps[:, 0], mu=fr.prior[ri, k], sn=e[:, 0])
    return out


def _rate(pt: _Parts, kappa: float, d, w, g, m, v=1.0) -> np.ndarray:
    gd = g * d
    return ((pt.xc + w * v * pt.xs + gd * pt.xp + kappa * m * pt.mu)
            / (pt.nc + w * pt.ns + g * pt.np_ + kappa))


def _cell_values(ep: dict, s: str, rows: _Rows) -> tuple:
    d = np.asarray(ep["d"][s], float)[rows.age_b]
    g = np.asarray(ep["g"][s], float)[rows.role_b]
    m = np.where(rows.rookie, np.asarray(ep["m"][s], float)[rows.draft_b], 1.0)
    return d, float(ep["w"][s]), g, m, float(ep.get("v", {}).get(s, 1.0))


def rates(fr: ST._Frame, params: dict, ep: EPInput) -> dict:
    """`state._rates` contract ({stat: (rate, base sn, base prior)}) with the `+ep` pieces."""
    if params.get("relative", False):        # the relative state has no `+ep` form
        return ST._rates(fr, params)
    p = ep.params
    rows = _rows(fr, p, ep.data)
    parts = _parts(fr, params, rows)
    out = {}
    for s in ST.STATS:
        pt = parts[s]
        d, w, g, m, v = _cell_values(p, s, rows)
        out[s] = (_rate(pt, params["stats"][s]["kappa"], d, w, g, m, v), pt.sn, pt.mu)
    return out


# ============================================================================== fitting

def _train_frame(conn, season: str, sp: dict, data: EPData, games, pos, seasons):
    first = ST.season_label(ST.sidx_of(seasons[0]) - sp.get("history_seasons", 6))
    if games is None:
        games = ST.load_games(conn, since_season=first, until_season=seasons[-1])
    games = games[(games["season"] >= first) & (games["season"] <= seasons[-1])]
    games = games.reset_index(drop=True)
    pos = pos if pos is not None else ST.load_positions(conn)
    fr = ST._prepare(games, pos, sp, None)
    return fr


def components(fr: ST._Frame, sp: dict, ep: dict, data: EPData, rows_mask=None) -> dict:
    """Everything the fit / the state-level evaluation needs on the frame's reported rows:
    y, minutes x the minutes table, phi row, buckets and the per-stat parts."""
    rows = _rows(fr, ep, data)
    parts = _parts(fr, sp, rows)
    ri = fr.rep_idx
    minutes = fr.vals[ri, 0]
    mpg = fr.mpg[ri]
    tier = fr.tier[ri]
    mm = {s: minutes * ST.minutes_multiplier(sp, s, minutes, mpg) for s in ST.STATS}
    phi = {s: ST.phi_for(sp, s, tier) for s in ST.STATS}
    y = {s: fr.vals[ri, 1 + ST.STATS.index(s)] for s in ST.STATS}
    return {"rows": rows, "parts": parts, "mm": mm, "phi": phi, "y": y, "sidx": fr.sidx[ri],
            "day": fr.day[ri], "pid": fr.pid[ri], "tier": tier}


def _ll(y, mu, phi) -> np.ndarray:
    mu = np.maximum(mu, 1e-9)
    return y * np.log(mu) - (y + 1.0 / phi) * np.log1p(phi * mu)


def _sub(pt: _Parts, m: np.ndarray) -> _Parts:
    return _Parts(*(getattr(pt, f)[m] for f in ("xc", "nc", "xp", "np_", "xs", "ns", "mu", "sn")))


def _grid_cell(pt: _Parts, kappa, y, mm, phi, vals: dict, name: str, grid) -> tuple[float, float]:
    """Best value of one piece (`name`) on the rows given; the others fixed in `vals`."""
    best, best_ll = None, -np.inf
    for x in grid:
        v = dict(vals)
        v[name] = x
        r = _rate(pt, kappa, v["d"], v["w"], v["g"], v["m"], v.get("v", 1.0))
        ll = float(np.sum(_ll(y, r * mm, phi)))
        if ll > best_ll:
            best, best_ll = float(x), ll
    return best, best_ll


def fit(conn: sqlite3.Connection, season: str, sp: dict, *, data: EPData | None = None,
        games: pd.DataFrame | None = None, pos=None, parts=PARTS,
        n_seasons: int = FIT_SEASONS, passes: int = 2,
        log: Callable[[str], None] | None = None) -> dict:
    """`+ep` params for season S on the up-to-`n_seasons` seasons < S from FIRST_BIO_SEASON on,
    the base state params `sp` (of S) fixed. Pieces not in `parts` stay neutral."""
    say = log or (lambda m: None)
    data = data or load_data(conn)
    s0 = ST.sidx_of(season)
    lo = max(s0 - n_seasons, ST.sidx_of(FIRST_BIO_SEASON))
    seasons = [ST.season_label(i) for i in range(lo, s0)]
    ep = neutral_params()
    ep["parts"] = list(parts)
    ep["fit_seasons"] = [seasons[0], seasons[-1]]
    fr = _train_frame(conn, season, sp, data, games, pos, seasons)
    c = components(fr, sp, ep, data)
    tr = c["sidx"] >= lo
    rows = c["rows"]
    say(f"ep fit {season}: {int(tr.sum())} training rows on {seasons[0]}..{seasons[-1]}")
    diag: dict = {"n_train": int(tr.sum()), "ll_gain": {}}
    for s in ST.STATS:
        pt, y, mm, phi = c["parts"][s], c["y"][s], c["mm"][s], c["phi"][s]
        kappa = sp["stats"][s]["kappa"]
        d = np.asarray(ep["d"][s], float)
        g = np.asarray(ep["g"][s], float)
        m = np.asarray(ep["m"][s], float)
        w, vv = 0.0, 1.0

        def cur(mask=None):
            sel = tr if mask is None else (tr & mask)
            return sel, {"d": d[rows.age_b][sel], "w": w, "v": vv, "g": g[rows.role_b][sel],
                         "m": np.where(rows.rookie, m[rows.draft_b], 1.0)[sel]}

        base_r = _rate(pt, kappa, 1.0, 0.0, 1.0, 1.0)
        ll0 = float(np.sum(_ll(y[tr], (base_r * mm)[tr], phi[tr])))
        for _ in range(passes):
            if "w" in parts:
                sel, v = cur(pt.ns > 0)
                if sel.sum() >= MIN_ROWS:
                    sub, best = _sub(pt, sel), -np.inf
                    for vx in (V_GRID if "v" in parts else (1.0,)):
                        v["v"] = vx
                        wx, ll = _grid_cell(sub, kappa, y[sel], mm[sel], phi[sel], v, "w", W_GRID)
                        if ll > best:
                            best, w, vv = ll, wx, float(vx)
            if "d" in parts:
                n_a = np.zeros(N_AGE)
                for a in range(N_AGE):
                    sel, v = cur((rows.age_b == a) & (pt.np_ > 0))
                    n_a[a] = sel.sum()
                    if sel.sum() >= MIN_ROWS:
                        d[a], _ = _grid_cell(_sub(pt, sel), kappa, y[sel], mm[sel], phi[sel], v,
                                             "d", D_GRID)
                if "c" in parts and n_a.sum() > 0:      # remove the league-wide drift
                    d[:N_AGE] = d[:N_AGE] / float(np.sum(n_a * d[:N_AGE]) / n_a.sum())
            if "m" in parts:
                for k in range(N_DRAFT):
                    sel, v = cur(rows.rookie & (rows.draft_b == k))
                    if sel.sum() >= MIN_ROWS:
                        m[k], _ = _grid_cell(_sub(pt, sel), kappa, y[sel], mm[sel], phi[sel], v,
                                             "m", M_GRID)
            if "g" in parts:
                for k in range(N_ROLE):
                    if k == ROLE_NEUTRAL:
                        continue
                    sel, v = cur((rows.role_b == k) & (pt.np_ > 0))
                    if sel.sum() >= MIN_ROWS:
                        g[k], _ = _grid_cell(_sub(pt, sel), kappa, y[sel], mm[sel], phi[sel], v,
                                             "g", G_GRID)
        ep["d"][s], ep["g"][s], ep["m"][s], ep["w"][s], ep["v"][s] = (
            [float(x) for x in d], [float(x) for x in g], [float(x) for x in m], float(w),
            float(vv))
        sel, v = cur()
        r1 = _rate(_sub(pt, sel), kappa, v["d"], v["w"], v["g"], v["m"], v["v"])
        ll1 = float(np.sum(_ll(y[sel], r1 * mm[sel], phi[sel])))
        diag["ll_gain"][s] = (ll1 - ll0) / max(int(tr.sum()), 1)
        say(f"  {s}: w {w:.2f} v {vv:.3f} d {np.round(d[:N_AGE], 2).tolist()} "
            f"m {np.round(m, 2).tolist()} g {np.round(g, 2).tolist()} "
            f"ll/row {diag['ll_gain'][s]:+.5f}")
    diag["n_by"] = {"age": np.bincount(rows.age_b[tr], minlength=N_AGE + 1).tolist(),
                    "rookie_draft": np.bincount(rows.draft_b[tr & rows.rookie],
                                                minlength=N_DRAFT).tolist(),
                    "role": np.bincount(rows.role_b[tr], minlength=N_ROLE).tolist(),
                    "pre": int(np.sum(tr & (c["parts"]["pts"].ns > 0)))}
    ep["diagnostics"] = diag
    return ep
