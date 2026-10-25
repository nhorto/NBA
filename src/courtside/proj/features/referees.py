"""F7: referee crew tendencies (re-evaluation H3, docs/research/reeval-B.md).

History: sportsdataverse `nba_stats_officials` (officials per game, 1996-97 -> 2025-26, every
regular-season game: 3 officials, a 4th in ~5% of games; `sources.sportsdataverse`). Live: the
nba.com referee assignments posted on game day (`referee_assignments`: names, numbers and roles,
mapped to official ids by `crews_from_assignments`).

Per game three targets, each "actual vs what the model expected without the referees":
  fta    sum over both teams of FTA vs sum of E_fta (`common.base_rows`: EB x actual minutes x
         the market reconciliation), NB variance sum (E + phi E^2)
  pf     both teams' personal fouls vs possessions x the league's fouls per possession as of the
         day (`common.league_asof`)
  pace   game possessions vs the expected possessions of the two teams: minutes/240 x
         pace_T x pace_O / pace_L, with pace_T a team's as-of possessions per 240 minutes
         (`common.ew_asof`, half-life 90 d, 0.7 per season, shrunk to the league with 10 games)
Per official j the recency-weighted (half-life H days, RHO per season) sums over his earlier
games give r_j = sum Y / sum X, shrunk by empirical Bayes toward 1 like F8:
      theta_j = 1 + tau2 / (tau2 + c sum(w^2 V) / X^2) x (r_j - 1)
with tau2 (official effect variance) and c (variance calibration) by the placebo-corrected
method of moments over (official, season) blocks of the regular-season games of seasons < S
(`fit_params`; labels permuted within season for the placebo). The crew feature is the mean
over the game's officials of log theta_j. Heavy shrinkage by construction: an official works
~60-70 games a season and the effect variance is small.

Columns (per (game_id, team), the same for both teams; NaN when the crew is unknown)
  f7_fta, f7_pf, f7_pace   crew mean log index
  f7_n                     log(1 + crew mean weighted games)
Every DK stat sees all four (the layer can interact them with the P descriptors, e.g. a
whistle-heavy crew x a high FTA-rate player). Neutral: 0 (f7_n: its median).
"""

from __future__ import annotations

import math
import sqlite3
from typing import Sequence

import numpy as np
import pandas as pd

from .. import state as ST
from .common import ew_asof, league_asof, load_team_games, season_label

H, RHO = 365.0, 0.9
PACE_H, PACE_RHO, PACE_K = 90.0, 0.7, 10.0
TARGETS = ("fta", "pf", "pace")
TEAM_COLS = ("f7_fta", "f7_pf", "f7_pace", "f7_n")
NEUTRAL = {"f7_fta": 0.0, "f7_pf": 0.0, "f7_pace": 0.0}
TAU2_FLOOR = 1e-7
LIVE_ROLES = ("crew_chief", "referee", "umpire")


# ============================================================================== sources

def load_officials(seasons: Sequence[str]) -> pd.DataFrame:
    """game_id, official_id, first_name, last_name, jersey_num for `seasons` (missing season
    files are skipped: the feature is then NaN for those games)."""
    from ...sources import sportsdataverse as SDV
    parts = []
    for s in seasons:
        try:
            parts.append(SDV.read_officials(s, columns=["game_id", "official_id", "first_name",
                                                        "last_name", "jersey_num"]))
        except FileNotFoundError:
            continue
    if not parts:
        return pd.DataFrame(columns=["game_id", "official_id", "first_name", "last_name",
                                     "jersey_num"])
    o = pd.concat(parts, ignore_index=True)
    o["game_id"] = o["game_id"].astype(str)
    o["official_id"] = o["official_id"].astype(np.int64)
    return o.drop_duplicates(["game_id", "official_id"]).reset_index(drop=True)


def crews_from_assignments(conn: sqlite3.Connection, day, officials: pd.DataFrame
                           ) -> pd.DataFrame:
    """game_id, official_id of the day's posted crews (crew chief / referee / umpire), names
    mapped to ids by full name, then by jersey number among officials of the latest season.
    Unmapped officials are dropped (the crew mean uses the ones found)."""
    d = pd.Timestamp(day).date().isoformat()
    a = pd.read_sql_query(
        "SELECT game_id, official_name, official_number, role FROM referee_assignments "
        "WHERE nba_day = ? AND game_id IS NOT NULL", conn, params=(d,))
    a = a[a["role"].isin(LIVE_ROLES)].drop_duplicates(["game_id", "official_name"])
    if not len(a) or not len(officials):
        return pd.DataFrame(columns=["game_id", "official_id"])
    o = officials.copy()
    o["name"] = (o["first_name"].astype(str).str.strip() + " "
                 + o["last_name"].astype(str).str.strip()).str.lower()
    o = o.sort_values("game_id")
    by_name = dict(zip(o["name"], o["official_id"]))
    last = o[o["game_id"].str[3:5] == o["game_id"].str[3:5].max()]
    by_num = {}
    for num, oid in zip(pd.to_numeric(last["jersey_num"], errors="coerce"), last["official_id"]):
        if np.isfinite(num):
            by_num[int(num)] = oid
    ids = []
    for nm, num in zip(a["official_name"].astype(str).str.strip().str.lower(),
                       a["official_number"]):
        oid = by_name.get(nm)
        if oid is None and pd.notna(num):
            oid = by_num.get(int(num))
        ids.append(oid)
    a["official_id"] = ids
    a = a[a["official_id"].notna()]
    return pd.DataFrame({"game_id": a["game_id"].astype(str).to_numpy(),
                         "official_id": a["official_id"].astype(np.int64).to_numpy()})


# ============================================================================== targets

def game_targets(conn: sqlite3.Connection, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per completed game of `frame`: game_id, day, sidx, season_type and Y / X / V per
    target (fta from the frame's player rows; pf and pace from team_games)."""
    fr = frame
    phi = fr["d_fta"].to_numpy(float) if "d_fta" in fr else 0.0
    E = np.maximum(fr["E_fta"].to_numpy(float), 0.0)
    g = pd.DataFrame({"game_id": fr["game_id"].to_numpy(), "y": fr["fta"].to_numpy(float),
                      "x": E, "v": E + phi * E ** 2})
    g = g.groupby("game_id", sort=False).sum()
    first = season_label(int(fr["sidx"].min()) - 1)
    last = season_label(int(fr["sidx"].max()))
    t = load_team_games(conn, first, last)
    t = t[t["game_id"].isin(set(fr["game_id"]))] if len(t) else t
    out = _box_targets(t)
    out = out.merge(g.rename(columns={"y": "y_fta", "x": "x_fta", "v": "v_fta"}),
                    left_on="game_id", right_index=True, how="inner")
    for q in TARGETS:                 # a target without a finite expectation carries no weight
        bad = ~(np.isfinite(out[f"x_{q}"].to_numpy(float))
                & np.isfinite(out[f"y_{q}"].to_numpy(float)))
        out.loc[bad, [f"y_{q}", f"x_{q}", f"v_{q}"]] = 0.0
    return out.reset_index(drop=True)


def _team_pace(t: pd.DataFrame) -> np.ndarray:
    """As-of (strictly before the day) possessions per 240 minutes of each team-game's team and
    the league, shrunk to the league with PACE_K games."""
    v = np.column_stack([t["poss"].to_numpy(float), t["min"].to_numpy(float) / 240.0])
    S = ew_asof(t["team"].to_numpy(), t["day"].to_numpy(), t["sidx"].to_numpy(), v,
                t["team"].to_numpy(), t["day"].to_numpy(), t["sidx"].to_numpy(),
                half_life=PACE_H, rho=PACE_RHO)
    L = ew_asof(np.zeros(len(t), int), t["day"].to_numpy(), t["sidx"].to_numpy(), v,
                np.zeros(len(t), int), t["day"].to_numpy(), t["sidx"].to_numpy(),
                half_life=PACE_H, rho=PACE_RHO)
    with np.errstate(invalid="ignore", divide="ignore"):
        # no league history yet (the first day of the data): NaN, so the game carries no
        # weight (`game_targets`); never a fallback computed from all the data
        lg = np.where(L[:, 1] > 0, L[:, 0] / np.maximum(L[:, 1], 1e-9), np.nan)
        pace = (S[:, 0] + PACE_K * lg) / (S[:, 1] + PACE_K)
    return np.column_stack([pace, lg])


def _box_targets(t: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "day", "sidx", "season_type", "y_pf", "x_pf", "v_pf", "y_pace", "x_pace",
            "v_pace"]
    if not len(t):
        return pd.DataFrame(columns=cols)
    t = t.sort_values(["day", "game_id", "team"]).reset_index(drop=True)
    pl = _team_pace(t)
    t["_pace"], t["_lg"] = pl[:, 0], pl[:, 1]
    # league fouls per possession as of the day (season-to-date blended with last season)
    pf_rate = league_asof(t["day"].to_numpy(), t["sidx"].to_numpy(),
                          np.column_stack([t["pf"].to_numpy(float), t["poss"].to_numpy(float)]),
                          t["day"].to_numpy(), t["sidx"].to_numpy(), 200.0)[:, 0]
    t["_pfr"] = pf_rate                    # NaN before any league history: no weight
    h = t[t["is_home"].astype(int) == 1].copy()
    o = t[["game_id", "team", "_pace"]].rename(columns={"team": "opp", "_pace": "_opace"})
    h = h.merge(o, on=["game_id", "opp"], how="left")
    poss = (h["poss"].to_numpy(float) + h["o_poss"].to_numpy(float)) / 2.0
    frac = h["min"].to_numpy(float) / 240.0
    xp = frac * h["_pace"].to_numpy(float) * h["_opace"].to_numpy(float) / h["_lg"].to_numpy(float)
    out = pd.DataFrame({"game_id": h["game_id"].to_numpy(), "day": h["day"].to_numpy(),
                        "sidx": h["sidx"].to_numpy(), "season_type": h["season_type"].to_numpy(),
                        "y_pf": (h["pf"] + h["o_pf"]).to_numpy(float),
                        "x_pf": poss * 2.0 * h["_pfr"].to_numpy(float),
                        "y_pace": poss, "x_pace": xp})
    out["v_pf"] = out["x_pf"]
    out["v_pace"] = out["x_pace"]
    return out[cols]


# ============================================================================== shrinkage

def _official_rows(gt: pd.DataFrame, officials: pd.DataFrame) -> pd.DataFrame:
    """(game, official) rows with the game's targets."""
    return gt.merge(officials[["game_id", "official_id"]], on="game_id", how="inner")


def fit_params(gt: pd.DataFrame, officials: pd.DataFrame, train_mask: np.ndarray, *,
               seed: int = 13) -> dict:
    """tau2 / c per target on the regular-season games of `train_mask` (seasons < S), by the
    placebo-corrected method of moments over (official, season) blocks."""
    m = np.asarray(train_mask, bool) & (gt["season_type"] == "Regular Season").to_numpy()
    r = _official_rows(gt[m], officials)
    out = {}
    if not len(r):
        return {q: {"tau2": TAU2_FLOOR, "c": 1.0, "raw_tau2": 0.0, "se": float("nan")}
                for q in TARGETS}
    rng = np.random.default_rng(seed)
    plac = r.copy()
    lab = plac["official_id"].to_numpy().copy()
    for ix in plac.groupby("sidx", sort=False).indices.values():
        lab[ix] = lab[rng.permutation(ix)]
    plac["official_id"] = lab
    for q in TARGETS:
        def blocks(d):
            b = d.groupby(["official_id", "sidx"])[[f"y_{q}", f"x_{q}", f"v_{q}"]].sum()
            return b[f"y_{q}"].to_numpy(), b[f"x_{q}"].to_numpy(), b[f"v_{q}"].to_numpy(), \
                b.index.get_level_values(0).to_numpy()
        # level per season: the game-level ratio's season mean (blocks compare to it)
        lv = r.groupby("sidx")[[f"y_{q}", f"x_{q}"]].sum()
        lvl = (lv[f"y_{q}"] / lv[f"x_{q}"]).to_dict()
        rr = r.assign(**{f"x_{q}": r[f"x_{q}"] * r["sidx"].map(lvl)})
        pp = plac.assign(**{f"x_{q}": plac[f"x_{q}"] * plac["sidx"].map(lvl)})
        yp, xp, vp, _ = blocks(pp)
        c = float(((yp - xp) ** 2).sum() / max(vp.sum(), 1e-12))
        y, x, v, who = blocks(rr)
        num = (y - x) ** 2 - c * v
        tau2 = float(num.sum() / max((x ** 2).sum(), 1e-12))
        per = pd.Series(num).groupby(who).sum().to_numpy()
        G = len(per)
        se = float(math.sqrt(G / max(G - 1, 1) * np.sum((per - per.mean()) ** 2))
                   / max((x ** 2).sum(), 1e-12))
        out[q] = {"tau2": max(tau2, TAU2_FLOOR), "c": c, "raw_tau2": tau2, "se": se,
                  "sd_effect": math.sqrt(max(tau2, 0.0)), "n_blocks": int(len(y))}
    return out


def official_index(hist: pd.DataFrame, q: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per query (official_id, day, sidx): log theta per target and the weighted games, from the
    history (game, official) rows strictly before the day."""
    nq = len(q)
    out = {}
    if nq == 0:
        return pd.DataFrame(columns=[f"lt_{t}" for t in TARGETS] + ["n"])
    hv = np.column_stack([hist[f"y_{t}"] for t in TARGETS] + [hist[f"x_{t}"] for t in TARGETS]
                         + [np.ones(len(hist))])
    hk, hd, hs = (hist["official_id"].to_numpy(), hist["day"].to_numpy(),
                  hist["sidx"].to_numpy())
    qk, qd, qs = q["official_id"].to_numpy(), q["day"].to_numpy(), q["sidx"].to_numpy()
    S = ew_asof(hk, hd, hs, hv, qk, qd, qs, half_life=H, rho=RHO)
    V2 = ew_asof(hk, hd, hs, np.column_stack([hist[f"v_{t}"] for t in TARGETS]), qk, qd, qs,
                 half_life=H / 2.0, rho=RHO ** 2)
    # the league level of each target over the same window (the official is compared to it)
    Lg = ew_asof(np.zeros(len(hist), int), hd, hs, hv, np.zeros(nq, int), qd, qs,
                 half_life=H, rho=RHO)
    k = len(TARGETS)
    for j, t in enumerate(TARGETS):
        p = params.get(t, {})
        tau2, c = float(p.get("tau2", TAU2_FLOOR)), float(p.get("c", 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            lvl = np.where(Lg[:, k + j] > 0, Lg[:, j] / np.maximum(Lg[:, k + j], 1e-9), 1.0)
            X = S[:, k + j] * lvl
            r = np.where(X > 1e-9, S[:, j] / np.maximum(X, 1e-9), 1.0)
            noise = np.where(X > 1e-9, c * V2[:, j] / np.maximum(X, 1e-9) ** 2, np.inf)
        w = np.where(np.isfinite(noise), tau2 / (tau2 + noise), 0.0)
        out[f"lt_{t}"] = np.log(np.maximum(1.0 + w * (r - 1.0), 0.5))
    out["n"] = S[:, -1]
    return pd.DataFrame(out)


def crew_features(hist: pd.DataFrame, crews: pd.DataFrame, params: dict) -> pd.DataFrame:
    """game_id + TEAM_COLS for the crews (game_id, official_id, day, sidx); `hist` = the
    (game, official) target rows of completed games (anything on or after a query's day is
    ignored by `ew_asof`)."""
    if not len(crews):
        return pd.DataFrame(columns=["game_id", *TEAM_COLS])
    q = crews.reset_index(drop=True)
    ix = official_index(hist, q, params)
    d = pd.concat([q[["game_id"]], ix], axis=1)
    g = d.groupby("game_id", sort=False).mean()
    return pd.DataFrame({"game_id": g.index.to_numpy(), "f7_fta": g["lt_fta"].to_numpy(),
                         "f7_pf": g["lt_pf"].to_numpy(), "f7_pace": g["lt_pace"].to_numpy(),
                         "f7_n": np.log1p(np.maximum(g["n"].to_numpy(), 0.0))})


def build(conn: sqlite3.Connection, frame: pd.DataFrame, *, season: str = "",
          officials: pd.DataFrame | None = None) -> pd.DataFrame:
    """F7 per (game_id, team) for the rows of `frame` (params fit on games of seasons < S)."""
    season = season or str(frame["season"].max())
    if officials is None:
        seasons = sorted(frame["season"].unique())
        officials = load_officials([season_label(int(seasons[0][:4]) - 1)] + list(seasons))
    gt = game_targets(conn, frame)
    gt["season"] = [season_label(int(s)) for s in gt["sidx"]]
    prm = fit_params(gt, officials, (gt["season"] < season).to_numpy())
    hist = _official_rows(gt, officials)
    crews = hist[["game_id", "official_id", "day", "sidx"]]
    f = crew_features(hist, crews, prm)
    pairs = frame[["game_id", "team"]].drop_duplicates()
    out = pairs.merge(f, on="game_id", how="left")
    out.attrs["params"] = prm
    return out.reset_index(drop=True)


class History:
    """The live / per-day hook: completed-game targets and params, then `day_features` for a
    day's posted crews (strictly earlier games only)."""

    def __init__(self, conn: sqlite3.Connection, frame: pd.DataFrame, season: str,
                 officials: pd.DataFrame | None = None):
        seasons = sorted(frame["season"].unique())
        self.officials = officials if officials is not None else load_officials(
            [season_label(int(seasons[0][:4]) - 1)] + list(seasons))
        gt = game_targets(conn, frame)
        gt["season"] = [season_label(int(s)) for s in gt["sidx"]]
        self.params = fit_params(gt, self.officials, (gt["season"] < season).to_numpy())
        self.hist = _official_rows(gt, self.officials)

    def day_features(self, day, games: pd.DataFrame, crews: pd.DataFrame) -> pd.DataFrame:
        """(game_id, team) F7 rows for `games` (game_id, home, away) on `day` given `crews`
        (game_id, official_id; e.g. `crews_from_assignments`)."""
        d = pd.Timestamp(day).date()
        q = crews.copy()
        q["day"] = int(ST._day_of(d))
        q["sidx"] = ST.sidx_of(ST.season_of(d))
        f = crew_features(self.hist, q, self.params)
        keys = pd.concat([pd.DataFrame({"game_id": games["game_id"], "team": games["home"]}),
                          pd.DataFrame({"game_id": games["game_id"], "team": games["away"]})],
                         ignore_index=True)
        keys["game_id"] = keys["game_id"].astype(str)
        return keys.merge(f, on="game_id", how="left")


def columns(stat: str) -> list[str]:
    return list(TEAM_COLS)


def monotone(stat: str) -> dict:
    return {}
