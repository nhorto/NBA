"""Team pulse: who is rising, who is sliding, and a power ranking. Leak-free via ``as_of``.

Pipeline (all on games strictly before ``as_of``, regular season):

1. ``fit_ratings``: ridge least squares on per-game offensive ratings,
   ``ORtg(t vs o) - L = off_t + dal_o + h * hca / 2``, where ``dal`` is defensive points allowed
   above average (positive = bad defence). Each team's (off, dal) is shrunk toward its prior-season
   value (regressed by ``RHO``) with weight ``K_PRIOR`` pseudo-games. Net = off - dal. Early in a
   season the prior dominates; by ~30 games the current season does.
2. ``adjust_games``: every game's raw ORtg/DRtg is opponent- and venue-adjusted using step 1.
3. ``trend_stats``: last-10 adjusted net vs a *baseline* of the team's earlier games blended with
   the prior. ``z`` uses the calibrated game-to-game noise ``SIGMA_NET``. ``trend`` is the delta
   shrunk by its reliability (``TAU``). Flags require ``z >= Z_FLAG``, ``|delta| >= MIN_DELTA``,
   and ``>= MIN_GAMES_FLAG`` games played.

Constants were fit with ``python -m courtside.trends.calibrate`` on 2001-02..2014-15 and
validated out of sample on 2015-16..2024-25;
see docs/TRENDS.md for the numbers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from courtside import teams as teams_mod
from courtside.trends.common import (
    as_date,
    franchise_key,
    jsonable,
    load_team_games,
    prior_season,
    r1,
    season_of,
)

# --- calibrated constants (docs/TRENDS.md section "Calibration") ------------------------------
LEAGUE_ORTG = 112.0      # only used when a season has no games yet
HCA_PRIOR = 2.5          # home edge in net pts/100, prior for the ridge fit
HCA_WEIGHT = 200.0       # pseudo-games behind HCA_PRIOR (keeps early-season hca sane)
RHO = 0.60               # prior-season adj net regressed toward 0 by this factor
K_PRIOR = 16.0           # prior-season weight in pseudo-games
SIGMA_NET = 12.2         # sd of one game's opponent-adjusted net rating around team strength
SIGMA_SIDE = 10.0        # same for adjusted ORtg or DRtg alone
TAU = 1.5                # sd of genuine 10-game form shifts (pts/100), for shrinking `trend`
Z_FLAG = 2.25            # rising/sliding threshold on z (calibrated by within-season shuffles)
MIN_DELTA = 4.0          # and the adjusted gap must be at least this many pts/100
MIN_GAMES_FLAG = 15      # no flags before 15 games: last 10 needs a real baseline
W_RECENT = 0.10          # power = rating + W_RECENT * (last-10 adj net - rating)
# Out-of-sample validity of the flags (2015-16..2024-25, weekly checkpoints, next 10 games),
# from `python -m courtside.trends.calibrate`. Shipped in the facts doc so the writer hedges.
FLAG_VALIDITY = {
    "sample": "2015-16..2024-25, 5,708 team-checkpoints, constants fit on 2001-02..2014-15",
    "rising": {"n": 133, "next10_beat_rating_pct": 61.7, "next10_vs_rating": 1.5},
    "sliding": {"n": 110, "next10_below_rating_pct": 64.5, "next10_vs_rating": -2.1},
    "base_rate_pct": 49.7,
    "share_expected_by_chance": 0.72,
    "reading": "a flag means real but modest carry-over: most of a 10-game gap is noise",
}
SIDE_Z = 1.5             # offense/defense reason thresholds (supporting detail, not flags)
SIDE_DELTA = 3.0
STREAK_MIN = 4
WINDOW = 10
SHORT = 5


@dataclass
class Ratings:
    off: dict[str, float]
    dal: dict[str, float]
    hca: float
    league_ortg: float

    def net(self, fr: str) -> float:
        return self.off.get(fr, 0.0) - self.dal.get(fr, 0.0)


@dataclass
class TeamTrend:
    team: str
    name: str
    season: str
    as_of: str
    games: int
    record: dict
    home: dict
    road: dict
    last10_record: dict
    streak: dict
    season_raw: dict          # {"off","def","net"} pts/100, possession-weighted
    last10_raw: dict
    last5_raw: dict
    season_adj: dict          # fitted ratings (shrunk toward prior), pts/100 vs average
    last10_adj: dict          # mean opponent/venue-adjusted ratings over the last 10
    last5_adj: dict
    prior_season_adj_net: float | None
    sos_last10: dict          # {"opp_adj_net", "home_games", "rank"} rank 1 = hardest
    trend: dict               # {"baseline","delta","z","score","flag","suppressed"}
    power: dict               # {"score","rank","rank_7d_ago","change_7d"}
    reasons: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


# --- ratings -----------------------------------------------------------------------------------
def fit_ratings(g: pd.DataFrame, prior: dict[str, tuple[float, float]] | None = None,
                k: float = K_PRIOR) -> Ratings:
    """Ridge SRS on offence/defence. ``prior`` maps franchise -> (off, dal) pull targets."""
    prior = prior or {}
    frs = sorted(set(g["fr"]) | set(g["opp_fr"]) | set(prior))
    if not frs:
        return Ratings({}, {}, HCA_PRIOR, LEAGUE_ORTG)
    idx = {f: i for i, f in enumerate(frs)}
    n = len(frs)
    p = 2 * n + 1
    lam = np.full(p, max(k, 0.5))
    target = np.zeros(p)
    for f, (o, d) in prior.items():
        target[idx[f]] = o
        target[n + idx[f]] = d
    lam[-1] = HCA_WEIGHT
    target[-1] = HCA_PRIOR
    ata = np.diag(lam)
    atb = lam * target
    L = float(g["ortg"].mean()) if len(g) else LEAGUE_ORTG
    if len(g):
        ti = g["fr"].map(idx).to_numpy()
        oi = g["opp_fr"].map(idx).to_numpy() + n
        hh = g["h"].to_numpy(dtype=float) / 2.0
        y = g["ortg"].to_numpy() - L
        m = len(g)
        A = np.zeros((m, p))
        A[np.arange(m), ti] = 1.0
        A[np.arange(m), oi] = 1.0
        A[:, -1] = hh
        ata += A.T @ A
        atb += A.T @ y
    x = np.linalg.solve(ata, atb)
    # center so the league averages 0 on both sides (ridge can drift the split slightly)
    off = x[:n] - x[:n].mean()
    dal = x[n:2 * n] - x[n:2 * n].mean()
    return Ratings({f: float(off[i]) for f, i in idx.items()},
                   {f: float(dal[i]) for f, i in idx.items()}, float(x[-1]), L)


def prior_from_season(g_prev: pd.DataFrame, rho: float = RHO) -> dict[str, tuple[float, float]]:
    """Prior-season fitted (off, dal), regressed toward 0 by ``rho``. Keyed by franchise."""
    if g_prev.empty:
        return {}
    r = fit_ratings(g_prev, None, k=1.0)
    return {f: (rho * r.off[f], rho * r.dal[f]) for f in r.off}


def adjust_games(g: pd.DataFrame, r: Ratings) -> pd.DataFrame:
    """Add adj_off / adj_def / adj_net (pts/100 vs league average) and opp_net to each game."""
    g = g.copy()
    opp_off = g["opp_fr"].map(r.off).fillna(0.0)
    opp_dal = g["opp_fr"].map(r.dal).fillna(0.0)
    half = g["h"] * r.hca / 2.0
    g["adj_off"] = g["ortg"] - r.league_ortg - opp_dal - half
    g["adj_def"] = g["drtg"] - r.league_ortg - opp_off + half
    g["adj_net"] = g["adj_off"] - g["adj_def"]
    g["opp_net"] = opp_off - opp_dal
    return g


# --- trend statistics --------------------------------------------------------------------------
def _window_stats(vals: np.ndarray, prior_val: float, window: int, k: float, sigma: float):
    """(recent mean, baseline, delta, z) for one team's chronological adjusted values."""
    n = len(vals)
    w = min(window, n)
    if w == 0:
        return np.nan, prior_val, np.nan, np.nan
    recent = vals[-w:].mean()
    prev = vals[:-w]
    base = (prev.sum() + k * prior_val) / (len(prev) + k)
    var = sigma ** 2 * (1.0 / w + 1.0 / (len(prev) + k))
    delta = recent - base
    return recent, base, delta, delta / np.sqrt(var)


def trend_stats(adj: pd.DataFrame, prior: dict[str, tuple[float, float]], window: int = WINDOW,
                k: float = K_PRIOR, sigma: float = SIGMA_NET) -> pd.DataFrame:
    """Per-team last-``window`` vs baseline for net, offense and defense (adjusted)."""
    out = []
    for fr, grp in adj.groupby("fr", sort=False):
        grp = grp.sort_values("game_date")
        po, pd_ = prior.get(fr, (0.0, 0.0))
        rn, bn, dn, zn = _window_stats(grp["adj_net"].to_numpy(), po - pd_, window, k, sigma)
        _, _, do, zo = _window_stats(grp["adj_off"].to_numpy(), po, window, k, SIGMA_SIDE)
        _, _, dd, zd = _window_stats(grp["adj_def"].to_numpy(), pd_, window, k, SIGMA_SIDE)
        w = min(window, len(grp))
        noise = sigma ** 2 * (1.0 / max(w, 1) + 1.0 / (len(grp) - w + k))
        out.append({"fr": fr, "n": len(grp), "recent": rn, "base": bn, "delta": dn, "z": zn,
                    "score": dn * TAU ** 2 / (TAU ** 2 + noise) if w else 0.0,
                    "off_delta": do, "off_z": zo, "def_delta": dd, "def_z": zd})
    return pd.DataFrame(out)


def flag_for(n: int, z: float, delta: float) -> str | None:
    if n < MIN_GAMES_FLAG or not np.isfinite(z):
        return None
    if z >= Z_FLAG and delta >= MIN_DELTA:
        return "rising"
    if z <= -Z_FLAG and delta <= -MIN_DELTA:
        return "sliding"
    return None


def power_scores(adj: pd.DataFrame, r: Ratings, frs: list[str]) -> dict[str, float]:
    """Rating (season fit, shrunk to prior) nudged toward last-10 adjusted form."""
    out = {}
    for fr in frs:
        rating = r.net(fr)
        g = adj[adj["fr"] == fr]
        recent = g["adj_net"].tail(WINDOW).mean() if len(g) else rating
        # the recent nudge is itself shrunk by how many recent games exist
        w = min(WINDOW, len(g)) / WINDOW
        out[fr] = rating + W_RECENT * w * (recent - rating)
    return out


# --- raw summaries -------------------------------------------------------------------------------
def _raw(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"off": None, "def": None, "net": None, "games": 0}
    poss = g["poss"].sum()
    off = 100.0 * g["pts"].sum() / poss
    de = 100.0 * g["opp_pts"].sum() / poss
    return {"off": r1(off), "def": r1(de), "net": r1(off - de), "games": len(g)}


def _adj(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"off": None, "def": None, "net": None, "games": 0}
    return {"off": r1(g["adj_off"].mean()), "def": r1(g["adj_def"].mean()),
            "net": r1(g["adj_net"].mean()), "games": len(g)}


def _wl(g: pd.DataFrame) -> dict:
    w = int(g["win"].sum())
    return {"w": w, "l": int(len(g) - w)}


def current_streak(wins: list[int] | np.ndarray) -> tuple[str | None, int]:
    """('W', 7) for seven straight wins at the end of the sequence."""
    wins = list(wins)
    if not wins:
        return None, 0
    last = wins[-1]
    n = 0
    for v in reversed(wins):
        if v != last:
            break
        n += 1
    return ("W" if last else "L"), n


def _reasons(t: dict, tr: pd.Series) -> list[dict]:
    rs: list[dict] = []
    flag = t["trend"]["flag"]
    if flag:
        rs.append({"kind": "net_trend", "value": r1(tr["delta"]), "z": r1(tr["z"], 2),
                   "detail": f"adj net {tr['recent']:+.1f} last 10 vs {tr['base']:+.1f} baseline "
                             f"(z {tr['z']:+.1f}); raw net {t['last10_raw']['net']:+.1f} vs "
                             f"{t['season_raw']['net']:+.1f} season"})
    big = t["games"] >= MIN_GAMES_FLAG
    s10, sea = t["last10_raw"], t["season_raw"]
    if big and tr["def_z"] >= SIDE_Z and tr["def_delta"] >= SIDE_DELTA:
        rs.append({"kind": "defense_collapse", "value": r1(tr["def_delta"]),
                   "detail": f"DRtg {s10['def']:.1f} last 10 vs {sea['def']:.1f} season"})
    if big and tr["def_z"] <= -SIDE_Z and tr["def_delta"] <= -SIDE_DELTA:
        rs.append({"kind": "defense_surge", "value": r1(tr["def_delta"]),
                   "detail": f"DRtg {s10['def']:.1f} last 10 vs {sea['def']:.1f} season"})
    if big and tr["off_z"] >= SIDE_Z and tr["off_delta"] >= SIDE_DELTA:
        rs.append({"kind": "offense_surge", "value": r1(tr["off_delta"]),
                   "detail": f"ORtg {s10['off']:.1f} last 10 vs {sea['off']:.1f} season"})
    if big and tr["off_z"] <= -SIDE_Z and tr["off_delta"] <= -SIDE_DELTA:
        rs.append({"kind": "offense_slump", "value": r1(tr["off_delta"]),
                   "detail": f"ORtg {s10['off']:.1f} last 10 vs {sea['off']:.1f} season"})
    st = t["streak"]
    if st["n"] >= STREAK_MIN:
        verb = "won" if st["kind"] == "W" else "lost"
        rs.append({"kind": "streak", "value": st["n"] if st["kind"] == "W" else -st["n"],
                   "detail": f"{verb} {st['n']} straight"})
    l10 = t["last10_record"]
    if l10["w"] + l10["l"] == WINDOW and (l10["w"] >= 8 or l10["l"] >= 8):
        rs.append({"kind": "last10_record", "value": l10["w"],
                   "detail": f"{l10['w']}-{l10['l']} in last 10"})
    sos = t["sos_last10"]
    if big and sos["opp_adj_net"] is not None and abs(sos["opp_adj_net"]) >= 2.0:
        hard = sos["opp_adj_net"] > 0
        pos = sos["rank"] if hard else sos["of"] + 1 - sos["rank"]
        rs.append({"kind": "schedule_tough" if hard else "schedule_soft",
                   "value": sos["opp_adj_net"],
                   "detail": f"last 10 opponents avg adj net {sos['opp_adj_net']:+.1f} "
                             f"({_ordinal(pos)} {'hardest' if hard else 'easiest'} of {sos['of']})"})
    hm, rd = t["home"], t["road"]
    if (hm["w"] + hm["l"] >= 10 and rd["w"] + rd["l"] >= 10 and hm["net"] is not None
            and rd["net"] is not None and abs(hm["net"] - rd["net"]) >= 15.0):
        rs.append({"kind": "home_road_split", "value": r1(hm["net"] - rd["net"]),
                   "detail": f"net {hm['net']:+.1f} at home ({hm['w']}-{hm['l']}) vs "
                             f"{rd['net']:+.1f} on the road ({rd['w']}-{rd['l']})"})
    return rs


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# --- entry points ----------------------------------------------------------------------------
def compute(cur: pd.DataFrame, prev: pd.DataFrame, as_of: date, season: str) -> list[TeamTrend]:
    """Team pulse from pre-loaded frames (``cur``: this season before as_of; ``prev``: prior)."""
    prior = prior_from_season(prev)
    r = fit_ratings(cur, prior)
    adj = adjust_games(cur, r).sort_values(["game_date", "game_id"])
    frs = sorted(set(cur["fr"])) if len(cur) else sorted(prior)
    trs = trend_stats(adj, prior).set_index("fr") if len(adj) else pd.DataFrame()
    power = power_scores(adj, r, frs)
    # power 7 days ago, for "rating change"
    cut = pd.Timestamp(as_of - timedelta(days=7))
    cur7 = cur[cur["game_date"] < cut]
    r7 = fit_ratings(cur7, prior)
    power7 = power_scores(adjust_games(cur7, r7).sort_values(["game_date", "game_id"]), r7, frs)
    rank = {f: i + 1 for i, f in enumerate(sorted(frs, key=lambda f: -power[f]))}
    rank7 = {f: i + 1 for i, f in enumerate(sorted(frs, key=lambda f: -power7[f]))}
    sos = {}
    for fr in frs:
        g = adj[adj["fr"] == fr].tail(WINDOW)
        sos[fr] = g["opp_net"].mean() if len(g) else np.nan
    sos_order = sorted([f for f in frs if np.isfinite(sos[f])], key=lambda f: -sos[f])
    sos_rank = {f: i + 1 for i, f in enumerate(sos_order)}

    out: list[TeamTrend] = []
    for fr in frs:
        g = adj[adj["fr"] == fr]
        code = g["team"].iloc[-1] if len(g) else fr
        try:
            name = f"{teams_mod.by_tricode(fr).city} {teams_mod.by_tricode(fr).name}"
        except KeyError:
            name = code
        st_kind, st_n = current_streak(g["win"].to_numpy())
        tr = trs.loc[fr] if fr in trs.index else pd.Series(
            {"n": 0, "recent": np.nan, "base": r.net(fr), "delta": np.nan, "z": np.nan,
             "score": 0.0, "off_delta": np.nan, "off_z": np.nan, "def_delta": np.nan,
             "def_z": np.nan})
        n = len(g)
        flag = flag_for(n, float(tr["z"]), float(tr["delta"]))
        suppressed = None
        if n < MIN_GAMES_FLAG:
            suppressed = f"early season: {n} games played (trend flags need {MIN_GAMES_FLAG})"
        home_g, road_g = g[g["h"] == 1], g[g["h"] == -1]
        po = prior.get(fr)
        t = {
            "team": code, "name": name, "season": season, "as_of": as_of.isoformat(), "games": n,
            "record": _wl(g), "home": {**_wl(home_g), "net": _raw(home_g)["net"]},
            "road": {**_wl(road_g), "net": _raw(road_g)["net"]},
            "last10_record": _wl(g.tail(WINDOW)),
            "streak": {"kind": st_kind, "n": st_n,
                       "label": f"{st_kind}{st_n}" if st_kind else None},
            "season_raw": _raw(g), "last10_raw": _raw(g.tail(WINDOW)),
            "last5_raw": _raw(g.tail(SHORT)),
            "season_adj": {"off": r1(r.off.get(fr, 0.0)), "def": r1(r.dal.get(fr, 0.0)),
                           "net": r1(r.net(fr))},
            "last10_adj": _adj(g.tail(WINDOW)), "last5_adj": _adj(g.tail(SHORT)),
            "prior_season_adj_net": r1((po[0] - po[1]) / RHO) if po else None,
            "sos_last10": {"opp_adj_net": r1(sos[fr]), "home_games": int((g.tail(WINDOW)["h"] == 1).sum()),
                           "rank": sos_rank.get(fr), "of": len(sos_rank)},
            "trend": {"baseline": r1(tr["base"]), "recent": r1(tr["recent"]),
                      "delta": r1(tr["delta"]), "z": r1(tr["z"], 2), "score": r1(tr["score"]),
                      "flag": flag, "suppressed": suppressed},
            "power": {"score": r1(power[fr]), "rank": rank[fr], "rank_7d_ago": rank7[fr],
                      "change_7d": rank7[fr] - rank[fr]},
        }
        t["reasons"] = _reasons(t, tr) if n else []
        out.append(TeamTrend(**t))
    out.sort(key=lambda x: x.power["rank"])
    return out


def load_frames(conn: sqlite3.Connection, as_of: date, season: str):
    cur = load_team_games(conn, [season], as_of)
    prev = load_team_games(conn, [prior_season(season)], as_of)
    return cur, prev


def team_pulse(conn: sqlite3.Connection, as_of: date, season: str | None = None) -> list[TeamTrend]:
    """Every team's pulse as of the morning of ``as_of`` (games before that date only).

    Sorted by power rank. ``season`` defaults to the season containing ``as_of``.
    """
    as_of = as_date(as_of)
    season = season or season_of(as_of)
    cur, prev = load_frames(conn, as_of, season)
    return compute(cur, prev, as_of, season)


__all__ = [
    "Ratings",
    "TeamTrend",
    "adjust_games",
    "compute",
    "current_streak",
    "fit_ratings",
    "flag_for",
    "franchise_key",
    "prior_from_season",
    "team_pulse",
    "trend_stats",
]
