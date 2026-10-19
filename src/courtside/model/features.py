"""Rung-3 pre-game feature matrix (one row per game), strictly causal.

`build_features(frame, eng, elo)` turns the game frame (dataset + schedule columns, optional
per-game box columns `h_*`/`a_*` from `attach_box` and market columns from `attach_market`)
plus the player-engine and Elo outputs into the feature matrix the GBM uses. Every feature of a
game on day d is a function of

* the schedule (dates, sites) and the game's own pre-announced facts (home court, season type),
* results and box scores of games on days < d (rolling form, league environment, continuity),
* the player engine's pre-game strengths (availability oracle in the backtest, the expected
  lineup in production) and Elo's pre-game ratings,
* the market line (pre-game information; NaN when there is no line).

Rolling statistics are per team and per season, shifted by one game (a team plays at most once
per day, checked in dataset); league statistics use only days strictly before d. Box columns
are outcomes of their own game and are listed in BOX_OUTCOME_COLUMNS so leakage tests can
scramble them together with the score.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import market as market_mod

ELO_PTS = 28.0          # Elo points per point of margin (538; walk-forward fit gives ~27.7)
ELO_MEAN = 1505.0
HCA_PER100 = 3.0        # a-priori home edge removed from per-game net ratings before rolling
WINDOWS = (5, 10, 20)
LEAGUE_GAMES = 300      # league scoring/pace environment: last N games before the day
LEAGUE_HCA_GAMES = 1230  # league home-margin drift: last N non-neutral games before the day
PRIOR_STATS = ("pace", "ortg", "drtg", "tot")   # rolling stats padded with a last-season prior
PRIOR_WINDOW = 10
# Year-to-year persistence of a team's regular-season pace / ORtg / DRtg / game total, measured
# on 2003-2025 (regression slope of demeaned team means on last season's: 0.63/0.54/0.56/0.59).
PRIOR_CARRY = 0.6

BOX_STATS = ("poss", "ortg", "drtg", "efg", "tov", "orb", "ftr")
BOX_OUTCOME_COLUMNS = tuple(f"{s}_{c}" for s in ("h", "a") for c in BOX_STATS)

SCHED = ("home_b2b", "away_b2b", "home_three_in_four", "away_three_in_four",
         "home_four_in_six", "away_four_in_six", "rest_diff", "games_last7_diff",
         "travel_diff_1000km", "tz_abs_diff")

# Feature columns fed to the GBM (market columns are added for the market variant).
CORE_FEATURES = (
    "ps_diff", "ps_home", "ps_away", "abs_loss_home", "abs_loss_away", "elo_diff",
    "home_court", "is_playoff", "is_playin", *SCHED,
    "net_diff_r5", "net_diff_r10", "net_diff_r20",
    "efg_diff_r10", "tov_diff_r10", "orb_diff_r10", "ftr_diff_r10",
    "cont_home", "cont_away", "gp_home", "gp_away", "season_day",
    "lg_hca", "apm_hca",
    "pace_r10_sum", "ortg_r10_home", "drtg_r10_home", "ortg_r10_away", "drtg_r10_away",
    "tot_r10_home", "tot_r10_away", "lg_total", "lg_pace",
)
MARKET_FEATURES = ("mkt_margin", "mkt_total")


# --- loaders ---------------------------------------------------------------------------------

def load_team_box(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per (game_id, raw team) box: possessions, off/def rating, eFG, TOV%, OREB%, FT rate."""
    df = pd.read_sql_query(
        "SELECT t.game_id, t.team, t.fta, t.fga, a.poss, a.off_rating, a.def_rating, "
        "a.efg_pct, a.tov_pct, a.oreb_pct FROM team_games t "
        "LEFT JOIN team_games_adv a USING (game_id, team)", conn)
    out = pd.DataFrame({"game_id": df["game_id"], "team": df["team"]})
    out["poss"] = df["poss"].astype(float)
    out["ortg"] = df["off_rating"].astype(float)
    out["drtg"] = df["def_rating"].astype(float)
    out["efg"] = df["efg_pct"].astype(float)
    out["tov"] = df["tov_pct"].astype(float)
    out["orb"] = df["oreb_pct"].astype(float)
    fga = df["fga"].astype(float)
    out["ftr"] = np.where(fga > 0, df["fta"].astype(float) / fga.where(fga > 0), np.nan)
    return out


def attach_box(frame: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """frame + h_<stat>, a_<stat> (NaN where missing, e.g. games not played yet)."""
    out = frame.copy()
    b = box.set_index(["game_id", "team"])[list(BOX_STATS)]
    for side, col in (("h", "home_raw"), ("a", "away_raw")):
        idx = pd.MultiIndex.from_arrays([frame["game_id"].to_numpy(), frame[col].to_numpy()])
        vals = b.reindex(idx).to_numpy()
        for j, c in enumerate(BOX_STATS):
            out[f"{side}_{c}"] = vals[:, j]
    return out


def load_closing_full(conn: sqlite3.Connection) -> pd.DataFrame:
    """game_id, mkt_margin (= -closing home spread), mkt_total, mkt_p (de-vigged moneyline);
    one row per game, market.SOURCE_PRIORITY order."""
    df = pd.read_sql_query(
        "SELECT game_id, source, home_spread_close, total_close, home_ml_close, away_ml_close "
        "FROM closing_lines", conn)
    rank = {s: i for i, s in enumerate(market_mod.SOURCE_PRIORITY)}
    df["rank"] = df["source"].map(rank).fillna(len(rank))
    ok = (df["home_ml_close"].abs() >= 100) & (df["away_ml_close"].abs() >= 100)
    ph = market_mod.american_to_prob(df["home_ml_close"].where(ok))
    pa = market_mod.american_to_prob(df["away_ml_close"].where(ok))
    df["mkt_p"] = np.where(ok, ph / (ph + pa), np.nan)
    df = df.sort_values(["game_id", "rank"]).drop_duplicates("game_id")
    return pd.DataFrame({"game_id": df["game_id"].to_numpy(),
                         "mkt_margin": -df["home_spread_close"].to_numpy(float),
                         "mkt_total": df["total_close"].to_numpy(float),
                         "mkt_p": df["mkt_p"].to_numpy(float)})


def attach_market(frame: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    """frame + mkt_margin, mkt_total, mkt_p. Values already present in `frame` (e.g. the live
    pre-tip line in production) win over the closing lines."""
    out = frame.copy()
    m = frame[["game_id"]].merge(closing, on="game_id", how="left")
    for c in ("mkt_margin", "mkt_total", "mkt_p"):
        v = m[c].to_numpy(float)
        if c in frame.columns:
            cur = frame[c].to_numpy(float)
            v = np.where(np.isfinite(cur), cur, v)
        out[c] = v
    return out


# --- helpers ---------------------------------------------------------------------------------

def _col(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].to_numpy(float) if c in df.columns else np.full(len(df), np.nan)


def _league_trailing(day: np.ndarray, val: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Per game: mean of `val` over the last `n` masked games on days strictly before its day."""
    v = np.where(mask & np.isfinite(val), val, 0.0)
    c = (mask & np.isfinite(val)).astype(float)
    cs_v = np.concatenate([[0.0], np.cumsum(v)])
    cs_c = np.concatenate([[0.0], np.cumsum(c)])
    end = np.searchsorted(day, day, side="left")          # games before this day
    # walk back until n counted games: find start s with cs_c[end] - cs_c[s] >= n
    target = cs_c[end] - n
    start = np.searchsorted(cs_c, np.maximum(target, 0.0), side="right") - 1
    start = np.clip(np.where(target <= 0, 0, start), 0, None)
    cnt = cs_c[end] - cs_c[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs_v[end] - cs_v[start]) / cnt, np.nan)


def _continuity(frame: pd.DataFrame, players: pd.DataFrame | None) -> np.ndarray:
    """(n, 2) per game and side: share of the team's minutes in the game played by players who
    played for the same franchise in the previous season (outcome-free given who played, but
    used only through the shifted rolling mean below)."""
    out = np.full((len(frame), 2), np.nan)
    if players is None or len(players) == 0:
        return out
    gi = pd.Series(np.arange(len(frame)), index=frame["game_id"].to_numpy())
    p = players[players["game_id"].isin(gi.index) & (players["min"].fillna(0) > 0)]
    if p.empty:
        return out
    g = gi.loc[p["game_id"].to_numpy()].to_numpy()
    hr = frame["home_raw"].to_numpy()[g]
    side = np.where(p["team"].to_numpy() == hr, 0, 1)
    fr = np.where(side == 0, frame["home"].to_numpy()[g], frame["away"].to_numpy()[g])
    seasons = sorted(frame["season"].unique())
    sidx_map = {s: i for i, s in enumerate(seasons)}
    si = np.array([sidx_map[s] for s in frame["season"].to_numpy()])[g]
    d = pd.DataFrame({"g": g, "side": side, "fr": fr, "si": si,
                      "pid": p["player_id"].to_numpy(), "min": p["min"].to_numpy(float)})
    prev = d[["pid", "fr", "si"]].drop_duplicates().assign(si=lambda x: x["si"] + 1, ret=1.0)
    d = d.merge(prev, on=["pid", "fr", "si"], how="left")
    d["ret"] = d["ret"].fillna(0.0)
    d["rmin"] = d["ret"] * d["min"]
    agg = d.groupby(["g", "side"])[["rmin", "min"]].sum()
    share = (agg["rmin"] / agg["min"]).to_numpy()
    gg = agg.index.get_level_values(0).to_numpy()
    ss = agg.index.get_level_values(1).to_numpy()
    out[gg, ss] = share
    # first season in the frame has no previous season: unknown
    first = np.array([sidx_map[s] for s in frame["season"].to_numpy()]) == 0
    out[first] = np.nan
    return out


def _team_priors(long: pd.DataFrame, reg: np.ndarray) -> pd.DataFrame:
    """Per row of the long (team, season) table: last season's regular-season mean of each
    PRIOR_STATS column for that team, regressed toward the league mean by PRIOR_CARRY
    (`lg + carry * (team - lg)`; a team absent last season gets the league mean). NaN in the
    first season of the frame. Uses only the previous season, which is complete."""
    seasons = sorted(long["season"].unique())
    prev = dict(zip(seasons[1:], seasons[:-1]))
    key = pd.DataFrame({"team": long["team"].to_numpy(),
                        "season": long["season"].map(prev).to_numpy()}, index=long.index)
    lr = long[reg]
    out = pd.DataFrame(index=long.index)
    for c in PRIOR_STATS:
        tm = lr.groupby(["team", "season"])[c].mean().rename("tm").reset_index()
        lg = lr.groupby("season")[c].mean().rename("lg").reset_index()
        m = key.merge(tm, on=["team", "season"], how="left").merge(lg, on="season", how="left")
        out[c] = (m["lg"] + PRIOR_CARRY * (m["tm"].fillna(m["lg"]) - m["lg"])).to_numpy()
    return out


# --- main ------------------------------------------------------------------------------------

def build_features(frame: pd.DataFrame, eng: pd.DataFrame | None, elo: pd.DataFrame,
                   players: pd.DataFrame | None = None, ps_mode: str = "played",
                   overrides: bool = True) -> pd.DataFrame:
    """Feature matrix aligned with `frame` (same index). See module docstring.

    eng: player-engine output (`<mode>_home/away` for ps_mode and "roster" (or
    "roster_tight" for a *_tight ps_mode), `apm_hca`).
    elo: run_elo output (elo_home_pre, elo_away_pre).
    Production overrides (if overrides and present in frame): `lineup_home`, `lineup_away`
    (expected-lineup strengths from the injury report) replace the engine's available-lineup
    strength. Games with no player rows on a side (not played yet) and no override use the full
    recent roster (no absences known)."""
    n = len(frame)
    idx = frame.index
    F = pd.DataFrame(index=idx)
    hc = 1.0 - frame["is_neutral"].to_numpy(float)
    st = frame["season_type"].to_numpy()
    F["home_court"] = hc
    F["is_playoff"] = (st == "Playoffs").astype(float)
    F["is_playin"] = (st == "PlayIn").astype(float)

    # player strength and absences
    if eng is not None:
        ph = eng[f"{ps_mode}_home"].to_numpy(float).copy()
        pa = eng[f"{ps_mode}_away"].to_numpy(float).copy()
        rm = "roster_tight" if (ps_mode.endswith("_tight") and "roster_tight_home" in eng) \
            else "roster"
        rh = eng[f"{rm}_home"].to_numpy(float)
        ra = eng[f"{rm}_away"].to_numpy(float)
        if players is not None:
            have = players.loc[players["min"].fillna(0) > 0, ["game_id", "team"]]
            keys = set(zip(have["game_id"], have["team"]))
            nh = np.array([(g, t) not in keys for g, t in
                           zip(frame["game_id"], frame["home_raw"])])
            na = np.array([(g, t) not in keys for g, t in
                           zip(frame["game_id"], frame["away_raw"])])
            ph = np.where(nh, rh, ph)
            pa = np.where(na, ra, pa)
        if overrides:
            for arr, c in ((ph, "lineup_home"), (pa, "lineup_away")):
                if c in frame.columns:
                    o = frame[c].to_numpy(float)
                    arr[np.isfinite(o)] = o[np.isfinite(o)]
        F["ps_home"], F["ps_away"] = ph, pa
        F["ps_diff"] = ph - pa
        F["abs_loss_home"] = rh - ph
        F["abs_loss_away"] = ra - pa
        F["apm_hca"] = eng["apm_hca"].to_numpy(float)
    else:
        for c in ("ps_home", "ps_away", "ps_diff", "abs_loss_home", "abs_loss_away", "apm_hca"):
            F[c] = np.nan

    eh = elo["elo_home_pre"].to_numpy(float)
    ea = elo["elo_away_pre"].to_numpy(float)
    F["elo_diff"] = eh - ea

    # schedule
    for c in ("home_b2b", "away_b2b", "home_three_in_four", "away_three_in_four",
              "home_four_in_six", "away_four_in_six", "rest_diff"):
        F[c] = frame[c].to_numpy(float)
    F["games_last7_diff"] = (frame["home_games_last7"] - frame["away_games_last7"]).to_numpy(float)
    F["travel_diff_1000km"] = (frame["home_travel_km"] - frame["away_travel_km"]).to_numpy(float) / 1000
    F["tz_abs_diff"] = (frame["home_tz_change"].abs() - frame["away_tz_change"].abs()).to_numpy(float)

    # per-team long table for rolling form (outcomes of each team's own earlier games)
    hp, ap = frame["home_pts"].to_numpy(float), frame["away_pts"].to_numpy(float)
    gidx = np.arange(n)
    hs = np.where(hc > 0, 1.0, 0.0)
    rows = {}
    for side, sgn in (("h", 1.0), ("a", -1.0)):
        o = "a" if side == "h" else "h"
        poss = _col(frame, f"{side}_poss")
        ortg, drtg = _col(frame, f"{side}_ortg"), _col(frame, f"{side}_drtg")
        net = ortg - drtg
        own_pts, opp_pts = (hp, ap) if side == "h" else (ap, hp)
        net = np.where(np.isfinite(net), net, own_pts - opp_pts)   # no box: per-game margin
        opp_elo = ea if side == "h" else eh
        adj = net - sgn * hs * HCA_PER100 + (opp_elo - ELO_MEAN) / ELO_PTS
        rows[side] = pd.DataFrame({
            "g": gidx, "side": 0 if side == "h" else 1,
            "team": frame["home" if side == "h" else "away"].to_numpy(),
            "season": frame["season"].to_numpy(),
            "net": adj,
            "efg": _col(frame, f"{side}_efg") - _col(frame, f"{o}_efg"),
            "tov": _col(frame, f"{side}_tov") - _col(frame, f"{o}_tov"),
            "orb": _col(frame, f"{side}_orb") - _col(frame, f"{o}_orb"),
            "ftr": _col(frame, f"{side}_ftr") - _col(frame, f"{o}_ftr"),
            "pace": poss, "ortg": ortg, "drtg": drtg, "tot": own_pts + opp_pts,
        })
    cont = _continuity(frame, players)
    rows["h"]["cont"] = cont[:, 0]
    rows["a"]["cont"] = cont[:, 1]
    long = pd.concat([rows["h"], rows["a"]], ignore_index=True)
    long = long.sort_values(["g", "side"], kind="mergesort")   # chronological within team
    grp = long.groupby(["team", "season"], sort=False)
    long["gp"] = grp.cumcount().astype(float)
    roll = {}
    stats = {"net": WINDOWS, "efg": (10,), "tov": (10,), "orb": (10,), "ftr": (10,),
             "pace": (10,), "ortg": (10,), "drtg": (10,), "tot": (10,), "cont": (10,)}
    shifted = grp[list(stats)].shift(1)
    shifted[["team", "season"]] = long[["team", "season"]]
    g2 = shifted.groupby(["team", "season"], sort=False)
    prior = _team_priors(long, st[long["g"].to_numpy()] == "Regular Season")
    for c, ws in stats.items():
        for w in ws:
            r = g2[c].rolling(w, min_periods=1)
            if c in PRIOR_STATS and w == PRIOR_WINDOW:
                # window padded with the team's prior: all prior at game 1, all current by
                # game `w`; unknown prior (first season in the frame) keeps the plain mean
                tot = r.sum().reset_index(level=[0, 1], drop=True)
                cnt = r.count().reset_index(level=[0, 1], drop=True)
                tot = tot.fillna(0.0)                 # sum of an empty window is NaN
                pr = prior[c].reindex(tot.index)
                with np.errstate(invalid="ignore", divide="ignore"):
                    plain = tot / cnt.where(cnt > 0)
                    padded = (tot + (w - cnt) * pr) / w
                roll[f"{c}_r{w}"] = padded.where(pr.notna(), plain)
            else:
                roll[f"{c}_r{w}"] = r.mean().reset_index(level=[0, 1], drop=True)
    R = pd.DataFrame(roll).reindex(long.index)
    R["g"], R["side"], R["gp"] = long["g"], long["side"], long["gp"]
    Rh = R[R["side"] == 0].set_index("g").reindex(gidx)
    Ra = R[R["side"] == 1].set_index("g").reindex(gidx)
    for w in WINDOWS:
        F[f"net_diff_r{w}"] = Rh[f"net_r{w}"].to_numpy() - Ra[f"net_r{w}"].to_numpy()
    for c in ("efg", "tov", "orb", "ftr"):
        F[f"{c}_diff_r10"] = Rh[f"{c}_r10"].to_numpy() - Ra[f"{c}_r10"].to_numpy()
    F["cont_home"], F["cont_away"] = Rh["cont_r10"].to_numpy(), Ra["cont_r10"].to_numpy()
    F["gp_home"], F["gp_away"] = Rh["gp"].to_numpy(), Ra["gp"].to_numpy()
    F["pace_r10_sum"] = Rh["pace_r10"].to_numpy() + Ra["pace_r10"].to_numpy()
    F["ortg_r10_home"], F["drtg_r10_home"] = Rh["ortg_r10"].to_numpy(), Rh["drtg_r10"].to_numpy()
    F["ortg_r10_away"], F["drtg_r10_away"] = Ra["ortg_r10"].to_numpy(), Ra["drtg_r10"].to_numpy()
    F["tot_r10_home"], F["tot_r10_away"] = Rh["tot_r10"].to_numpy(), Ra["tot_r10"].to_numpy()

    # season phase (schedule only)
    day = frame["game_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    first_day = pd.Series(day).groupby(frame["season"].to_numpy()).transform("min").to_numpy()
    F["season_day"] = (day - first_day).astype(float)

    # league environment, strictly before the day
    reg = st == "Regular Season"
    F["lg_total"] = _league_trailing(day, hp + ap, np.ones(n, bool), LEAGUE_GAMES)
    pace_g = (_col(frame, "h_poss") + _col(frame, "a_poss")) / 2
    F["lg_pace"] = _league_trailing(day, pace_g, np.ones(n, bool), LEAGUE_GAMES)
    F["lg_hca"] = _league_trailing(day, hp - ap, reg & (hc > 0), LEAGUE_HCA_GAMES)

    # market (pre-game; NaN without a line)
    F["mkt_margin"] = _col(frame, "mkt_margin")
    F["mkt_total"] = _col(frame, "mkt_total")
    return F
