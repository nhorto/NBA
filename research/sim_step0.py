"""T2.5: sim step 0 -- measure the targets a possession simulator must reproduce (docs/PLAYERS.md §4.5, M1a).

Fit window 2019-20 -> 2023-24 regular season; 2024-25 and 2025-26 are reported separately as an
out-of-sample check (nothing is fitted on them). Every quantity is a smooth function of per-game
sums, so the CIs come from one game-clustered bootstrap (games resampled within season, B
replicates, percentile intervals). Line-conditional quantities use the closing spread/total
(``closing_lines``, kaggle source preferred, espn as fallback).

Quantities
  line      Var(H | line), Var(A | line), Cov(H, A | line), sd(margin), sd(total) residuals
  pace      per-team possessions per game, Var(N | total line) (the pace part of Cov(H, A))
  poss      points per possession: mean, variance, distribution, and mean/variance/duration by how
            the possession started (period start, after a make, a defensive rebound, a steal, a
            dead-ball turnover)
  iid       the §4.5 arithmetic: the team variance an iid-possession sim would produce, and the
            decomposition given the game's possessions N: Var(total | N, line) vs 2 N sigma^2
            (room for a shared efficiency latent tau_g), points per extra possession (pace is
            endogenous), Cov(H, A | N, line)
  seg       eight 6-minute regulation segments: Var(lead vs line) and Var(total vs line) at each
            boundary; the restoring force rho_k = d E[dm over segment k | m at its start, line] / dm,
            k = 2..8 (and the same on the running total); leader's dm vs line by |m| bin
  b4        slope of the final margin on m3 (margin after Q3), with and without the line
  q4        Q4 pace and leader/trailer Q4 points per possession by |m3| bin
  reb       rebound outcome of missed FGs and missed final FTs (team vs player, off vs def; S7)
  ot        P(OT), overall and by |spread|

Usage (writes research/sim_step0_output.txt and research/sim_step0_targets.json):

    uv run python research/sim_step0.py [--workers 3] [--boot 1000] [--cache /tmp/step0.pkl]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from courtside import config
from courtside.analytics.pbp import AWAY, HOME, normalize
from courtside.analytics.stints_store import repair_legacy_pbp, season_of

FIT_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
OOS_SEASONS = ["2024-25", "2025-26"]
SEG = 360.0                                     # 6-minute segments
N_SEG = 8
REG_END = 2880.0
ABS_BINS = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 999)]
BIN_LABELS = ["0-5", "6-10", "11-15", "16-20", "21+"]
SPREAD_BINS = [(0, 3), (3.5, 6), (6.5, 10), (10.5, 99)]
SPREAD_LABELS = ["0-3", "3.5-6", "6.5-10", "10.5+"]
START_TYPES = ["period_start", "after_make", "after_dreb", "after_steal", "after_dead_tov"]
REB_CATS = ["p_dreb", "p_oreb", "t_dreb", "t_oreb", "none"]
OUT = Path(__file__).with_name("sim_step0_output.txt")
TARGETS = Path(__file__).with_name("sim_step0_targets.json")


# ------------------------------------------------------------------------- per-game extraction

def abs_bin(x: float) -> int:
    """|m| (integer points) -> index into ABS_BINS (0..4)."""
    a = abs(x)
    for i, (lo, hi) in enumerate(ABS_BINS):
        if a <= hi:
            return i
    return len(ABS_BINS) - 1


def boundary_margins(elapsed: np.ndarray, score_home: np.ndarray, score_away: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Home and away score at elapsed = 0, 360, ..., 2880 (score after the last event, in feed
    order, at or before the boundary)."""
    h, a = np.zeros(N_SEG + 1), np.zeros(N_SEG + 1)
    for k in range(1, N_SEG + 1):
        idx = np.flatnonzero(elapsed <= k * SEG + 1e-6)
        if len(idx):
            h[k], a[k] = score_home[idx[-1]], score_away[idx[-1]]
    return h, a


def segment_of(elapsed: float) -> int:
    """(0, 360] -> 0, ..., (2520, 2880] -> 7, overtime -> 8."""
    if elapsed > REG_END + 1e-6:
        return N_SEG
    return max(0, int(np.ceil(elapsed / SEG - 1e-9)) - 1)


def extract(events) -> dict:
    """One game's raw measurements from normalized events (home perspective)."""
    el = np.array([e.elapsed for e in events], dtype=float)
    sh = np.array([e.score_home for e in events], dtype=float)
    sa = np.array([e.score_away for e in events], dtype=float)
    bh, ba = boundary_margins(el, sh, sa)
    poss = {HOME: np.zeros(N_SEG + 1), AWAY: np.zeros(N_SEG + 1)}
    acc = {HOME: 0, AWAY: 0}
    cur_start = {HOME: "period_start", AWAY: "period_start"}
    pp = {t: [] for t in START_TYPES}               # points per possession by start type
    reb = {k: {c: 0 for c in REB_CATS + ["t_deadball"]} for k in ("fg", "ft")}
    dur = {t: 0.0 for t in START_TYPES}            # summed possession durations (s) by start type
    pending: str | None = None                      # "fg"/"ft" while a miss awaits its rebound
    t_start = 0.0                                   # elapsed when the current possession began
    for e in events:
        if e.is_period_start:
            cur_start = {HOME: "period_start", AWAY: "period_start"}
            t_start = e.elapsed
        # rebounds of misses
        if pending is not None:
            if e.is_rebound:
                kind = ("t_" if e.team_rebound else "p_") + ("oreb" if e.rebound_offensive else "dreb")
                reb[pending][kind] += 1
                if e.team_rebound and any(q.lower() == "deadball" for q in e.qualifiers):
                    reb[pending]["t_deadball"] += 1
                pending = None
            elif e.is_fga or e.is_ft or e.is_turnover or e.is_period_end or e.is_jumpball:
                reb[pending]["none"] += 1
                pending = None
        if e.is_fga and e.made is False:
            pending = "fg"
        elif e.is_ft and e.made is False and e.ft_num is not None and e.ft_num == e.ft_of \
                and not e.ft_retains_possession:
            pending = "ft"
        # possessions
        if e.side in acc and e.points:
            acc[e.side] += e.points
        if e.ends_possession and e.poss_side in acc:
            s = e.poss_side
            poss[s][segment_of(e.elapsed)] += 1
            pp[cur_start[s]].append(acc[s])
            dur[cur_start[s]] += max(e.elapsed - t_start, 0.0)
            t_start = e.elapsed
            acc[s] = 0
            if e.is_rebound:
                end = "after_dreb"
            elif e.is_turnover:
                end = "after_steal" if e.steal_id else "after_dead_tov"
            elif e.is_period_end:
                end = "period_start"
            else:
                end = "after_make"
            cur_start[HOME if s == AWAY else AWAY] = end
    if pending is not None:
        reb[pending]["none"] += 1
    return {"bh": bh, "ba": ba, "final_h": sh[-1], "final_a": sa[-1],
            "periods": max(e.period for e in events), "poss_h": poss[HOME], "poss_a": poss[AWAY],
            "pp": {k: np.array(v, dtype=float) for k, v in pp.items()}, "reb": reb, "dur": dur,
            "pp_leftover": acc[HOME] + acc[AWAY]}


def _load(task: tuple[str, str, str]) -> tuple[str, dict | None, str]:
    gid, pbp_path, box_path = task
    try:
        pbp = json.load(gzip.open(pbp_path))
        box = json.load(gzip.open(box_path)) if Path(box_path).exists() else None
        events = normalize(repair_legacy_pbp(pbp, box), box)
        if not events:
            return gid, None, "no_events"
        return gid, extract(events), ""
    except Exception as ex:  # noqa: BLE001 - one bad payload must not stop the run
        return gid, None, f"{type(ex).__name__}: {ex}"


def list_games(seasons: list[str]) -> list[tuple[str, str, str]]:
    root = config.data_dir() / "raw" / "nba_cdn"
    out = []
    for f in sorted((root / "playbyplay").glob("*/*.json.gz")):
        gid = f.name[:10]
        if gid.startswith("002") and season_of(gid) in seasons:
            out.append((gid, str(f), str(root / "boxscore" / f.parent.name / f.name)))
    return out


def load_lines(seasons: list[str]) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{config.db_path()}?mode=ro", uri=True)
    d = pd.read_sql("""
        SELECT h.game_id, h.season, h.pts AS h_pts, a.pts AS a_pts,
               c.source, c.home_spread_close AS spread, c.total_close AS total
        FROM team_games h JOIN team_games a ON a.game_id = h.game_id AND a.is_home = 0
        JOIN closing_lines c ON c.game_id = h.game_id
        WHERE h.is_home = 1 AND h.season_type = 'Regular Season'
          AND c.home_spread_close IS NOT NULL AND c.total_close IS NOT NULL""", conn)
    conn.close()
    d = d[d.season.isin(seasons)]
    d["pref"] = (d.source != "kaggle").astype(int)
    return d.sort_values(["game_id", "pref"]).drop_duplicates("game_id").drop(columns="pref")


# ------------------------------------------------------------------------- per-game features (sums)

def reg_feats(prefix: str, xs: list[float], y: float) -> dict[str, float]:
    """Contributions to X'X (upper triangle), X'y, y'y and n for OLS of y on [1, *xs]."""
    x = [1.0, *xs]
    f = {f"{prefix}n": 1.0, f"{prefix}yy": y * y}
    for i in range(len(x)):
        f[f"{prefix}xy{i}"] = x[i] * y
        for j in range(i, len(x)):
            f[f"{prefix}xx{i}{j}"] = x[i] * x[j]
    return f


def _acc(f: dict[str, float], d: dict[str, float]) -> None:
    for key, val in d.items():
        f[key] = f.get(key, 0.0) + val


def game_features(r: dict) -> dict[str, float]:
    """Flat per-game sums. ``r`` needs bh, ba, final_h, final_a, periods, poss_h, poss_a, pp, reb,
    spread, total (spread in betting convention: -6.5 = home favoured by 6.5)."""
    e = -float(r["spread"])                         # expected home margin
    tot = float(r["total"])
    eh, ea = (tot + e) / 2, (tot - e) / 2
    H, A = float(r["final_h"]), float(r["final_a"])
    m = np.asarray(r["bh"], float) - np.asarray(r["ba"], float)
    T = np.asarray(r["bh"], float) + np.asarray(r["ba"], float)
    final = H - A
    f: dict[str, float] = {"n": 1.0}
    # line residuals
    rh, ra = H - eh, A - ea
    f.update(rh=rh, ra=ra, rh2=rh * rh, ra2=ra * ra, rhra=rh * ra)
    # pace: per-team possessions (whole game) on the total line
    ph, pa = np.asarray(r["poss_h"], float), np.asarray(r["poss_a"], float)
    N = (ph.sum() + pa.sum()) / 2
    f["N"] = N
    f.update(reg_feats("pace_", [tot], N))
    for k in range(N_SEG + 1):
        f[f"segposs{k}"] = (ph[k] + pa[k]) / 2
    # team points given its own possessions and the line (home and away rows)
    _acc(f, reg_feats("hn_", [ph.sum(), eh], H))
    _acc(f, reg_feats("hn_", [pa.sum(), ea], A))
    # total and margin on the game's possessions and the line: Cov(H, A | N, line) = (VT - VM) / 4
    f.update(reg_feats("totN_", [N, tot], H + A))
    f.update(reg_feats("marN_", [N, e], H - A))
    # lead vs line at each boundary, and the variance curve
    for k in range(1, N_SEG + 1):
        L = m[k] - e * k / N_SEG
        f[f"L{k}"], f[f"L{k}sq"] = L, L * L
        LT = T[k] - tot * k / N_SEG
        f[f"LT{k}"], f[f"LT{k}sq"] = LT, LT * LT
    Lf = final - e
    f["Lf"], f["Lfsq"] = Lf, Lf * Lf
    # restoring force: dm over segment k+1 on m at its start (k = 1..7), with the line as control
    for k in range(1, N_SEG):
        f.update(reg_feats(f"rf{k + 1}_", [m[k], e], m[k + 1] - m[k]))
        # pooled over segments on the lead vs line; the e*k/8 coefficient tells whether the pull
        # is toward 0 (coef = rho) or toward the line's path (coef = 0)
        _acc(f, reg_feats("rfall_", [m[k] - e * k / N_SEG, e * k / N_SEG, e], m[k + 1] - m[k]))
        # the same on the running total (does a hot-scoring game cool down?)
        _acc(f, reg_feats(f"rt{k + 1}_", [T[k], tot], T[k + 1] - T[k]))
        _acc(f, reg_feats("rtall_", [T[k] - tot * k / N_SEG, tot * k / N_SEG, tot], T[k + 1] - T[k]))
        if m[k] != 0:                               # leader's change vs line, by |m| bin
            s = np.sign(m[k])
            grp = "q4" if k >= 6 else "q13"
            b = abs_bin(m[k])
            f[f"lead_{grp}_{b}_n"] = f.get(f"lead_{grp}_{b}_n", 0.0) + 1.0
            f[f"lead_{grp}_{b}_d"] = f.get(f"lead_{grp}_{b}_d", 0.0) + s * (m[k + 1] - m[k] - e / N_SEG)
    # b4: final margin on m3
    m3 = m[6]
    f.update(reg_feats("b4_", [m3, e], final))
    f.update(reg_feats("b4u_", [m3], final))
    f.update(reg_feats("b4r_", [m3, e], m[8]))      # regulation margin
    # Q4 pace and leader/trailer efficiency by |m3|
    b = abs_bin(m3)
    q4 = (ph[6] + ph[7] + pa[6] + pa[7]) / 2
    q13 = (ph[:6].sum() + pa[:6].sum()) / 2 / 3
    f[f"q4_{b}_n"], f[f"q4_{b}_poss"], f[f"q4_{b}_base"] = 1.0, q4, q13
    if m3 != 0:
        lead_home = m3 > 0
        hq4, aq4 = r["bh"][8] - r["bh"][6], r["ba"][8] - r["ba"][6]   # regulation Q4 only
        hp, ap = ph[6] + ph[7], pa[6] + pa[7]
        lp, lposs, tp, tposs = (hq4, hp, aq4, ap) if lead_home else (aq4, ap, hq4, hp)
        e_lead = e if lead_home else -e
        reg_poss = (ph[:8].sum() + pa[:8].sum()) / 2
        f.update({f"lt_{b}_n": 1.0, f"lt_{b}_lp": lp, f"lt_{b}_lposs": lposs,
                  f"lt_{b}_tp": tp, f"lt_{b}_tposs": tposs,
                  f"lt_{b}_exp100": 100 * e_lead / max(reg_poss, 1.0)})
    # points per possession
    for t in START_TYPES:
        x = r["pp"][t]
        f[f"pp_{t}_n"], f[f"pp_{t}_s"], f[f"pp_{t}_s2"] = len(x), x.sum(), (x * x).sum()
        f[f"pp_{t}_dur"] = float(r.get("dur", {}).get(t, 0.0))
    allx = np.concatenate([r["pp"][t] for t in START_TYPES])
    for v in range(5):
        f[f"pp_val{v}"] = float((np.minimum(allx, 4) == v).sum())
    # rebounds
    for kind in ("fg", "ft"):
        for c in REB_CATS + ["t_deadball"]:
            f[f"reb_{kind}_{c}"] = float(r["reb"][kind].get(c, 0))
    # overtime
    ot = float(r["periods"] > 4)
    f["ot"] = ot
    sb = next(i for i, (lo, hi) in enumerate(SPREAD_BINS) if abs(r["spread"]) <= hi)
    f[f"otb_{sb}_n"], f[f"otb_{sb}_ot"] = 1.0, ot
    return f


# ------------------------------------------------------------------------- estimators on sums

def _var(S, n, s, s2):
    return (S[s2] - S[s] ** 2 / S[n]) / (S[n] - 1)


def _cov(S, n, a, b, ab):
    return (S[ab] - S[a] * S[b] / S[n]) / (S[n] - 1)


def ols(S, prefix: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Batched OLS from summed X'X / X'y (p = number of columns incl. intercept).
    Returns (beta (R, p), residual variance (R,))."""
    R = len(S[f"{prefix}n"])
    XX, Xy = np.zeros((R, p, p)), np.zeros((R, p))
    for i in range(p):
        Xy[:, i] = S[f"{prefix}xy{i}"]
        for j in range(i, p):
            XX[:, i, j] = XX[:, j, i] = S[f"{prefix}xx{i}{j}"]
    beta = np.linalg.solve(XX, Xy[..., None])[..., 0]
    rss = S[f"{prefix}yy"] - (beta * Xy).sum(1)
    return beta, rss / (S[f"{prefix}n"] - p)


def _ratio(S, a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return S[a] / S[b]


def _pp_all(S, what):
    n = sum(S[f"pp_{t}_n"] for t in START_TYPES)
    s = sum(S[f"pp_{t}_s"] for t in START_TYPES)
    s2 = sum(S[f"pp_{t}_s2"] for t in START_TYPES)
    mu = s / n
    return {"n": n, "mu": mu, "var": (s2 - s * s / n) / (n - 1)}[what]


def _iid_team_var(S):
    """sigma^2 E[N] + mu^2 Var(N | total line): an iid-possession team's variance given the line."""
    _, vN = ols(S, "pace_", 2)
    return _pp_all(S, "var") * S["N"] / S["n"] + _pp_all(S, "mu") ** 2 * vN


def _pace_cov(S):
    _, vN = ols(S, "pace_", 2)
    return _pp_all(S, "mu") ** 2 * vN


def _reb(S, kind, num, den):
    return sum(S[f"reb_{kind}_{c}"] for c in num) / sum(S[f"reb_{kind}_{c}"] for c in den)


def estimators() -> "OrderedDict[str, tuple[str, callable]]":
    """name -> (group, f(S) -> (R,) array)."""
    E: OrderedDict = OrderedDict()
    v_h = lambda S: _var(S, "n", "rh", "rh2")
    v_a = lambda S: _var(S, "n", "ra", "ra2")
    c_ha = lambda S: _cov(S, "n", "rh", "ra", "rhra")
    E["var_H_line"] = ("line", v_h)
    E["var_A_line"] = ("line", v_a)
    E["var_team_line"] = ("line", lambda S: (v_h(S) + v_a(S)) / 2)
    E["cov_HA_line"] = ("line", c_ha)
    E["corr_HA_line"] = ("line", lambda S: c_ha(S) / np.sqrt(v_h(S) * v_a(S)))
    E["sd_margin_line"] = ("line", lambda S: np.sqrt(v_h(S) + v_a(S) - 2 * c_ha(S)))
    E["sd_total_line"] = ("line", lambda S: np.sqrt(v_h(S) + v_a(S) + 2 * c_ha(S)))
    E["bias_H"] = ("line", lambda S: S["rh"] / S["n"])
    E["bias_A"] = ("line", lambda S: S["ra"] / S["n"])
    E["poss_per_team"] = ("pace", lambda S: S["N"] / S["n"])
    E["sd_poss_given_total"] = ("pace", lambda S: np.sqrt(ols(S, "pace_", 2)[1]))
    E["poss_slope_on_total"] = ("pace", lambda S: ols(S, "pace_", 2)[0][:, 1])
    for k in range(N_SEG + 1):
        lab = f"seg{k + 1}" if k < N_SEG else "ot_all_games"
        E[f"poss_per_team_{lab}"] = ("pace", lambda S, k=k: S[f"segposs{k}"] / S["n"])
    E["ppp_mean"] = ("poss", lambda S: _pp_all(S, "mu"))
    E["ppp_var"] = ("poss", lambda S: _pp_all(S, "var"))
    for v in range(5):
        E[f"ppp_share_{v}{'+' if v == 4 else ''}"] = (
            "poss", lambda S, v=v: S[f"pp_val{v}"] / _pp_all(S, "n"))
    for t in START_TYPES:
        E[f"ppp_mean_{t}"] = ("poss", lambda S, t=t: S[f"pp_{t}_s"] / S[f"pp_{t}_n"])
        E[f"ppp_var_{t}"] = ("poss", lambda S, t=t: _var(S, f"pp_{t}_n", f"pp_{t}_s", f"pp_{t}_s2"))
        E[f"poss_seconds_{t}"] = ("poss", lambda S, t=t: S[f"pp_{t}_dur"] / S[f"pp_{t}_n"])
        E[f"share_{t}"] = ("poss", lambda S, t=t: S[f"pp_{t}_n"] / _pp_all(S, "n"))
    E["iid_var_team"] = ("iid", _iid_team_var)
    E["iid_poss_noise_var"] = ("iid", lambda S: _pp_all(S, "var") * S["N"] / S["n"])
    E["pace_cov"] = ("iid", _pace_cov)
    E["excess_var_team"] = ("iid", lambda S: (v_h(S) + v_a(S)) / 2 - _iid_team_var(S))
    E["efficiency_cov"] = ("iid", lambda S: c_ha(S) - _pace_cov(S))
    E["var_team_given_poss_line"] = ("iid", lambda S: ols(S, "hn_", 3)[1])
    E["pts_per_extra_poss"] = ("iid", lambda S: ols(S, "hn_", 3)[0][:, 1])
    E["pts_per_extra_game_poss"] = ("iid", lambda S: ols(S, "totN_", 3)[0][:, 1] / 2)
    E["cov_HA_given_poss_line"] = ("iid", lambda S: (ols(S, "totN_", 3)[1] - ols(S, "marN_", 3)[1]) / 4)
    E["var_team_given_game_poss_line"] = ("iid", lambda S: (ols(S, "totN_", 3)[1] + ols(S, "marN_", 3)[1]) / 4)
    E["var_total_given_poss_line"] = ("iid", lambda S: ols(S, "totN_", 3)[1])
    E["var_margin_given_poss_line"] = ("iid", lambda S: ols(S, "marN_", 3)[1])
    E["iid_var_total_given_poss"] = ("iid", lambda S: 2 * _pp_all(S, "var") * S["N"] / S["n"])
    # a shared efficiency latent adds 4 N^2 tau_g^2 to Var(total | N); what is left over the iid noise
    E["excess_var_total_given_poss"] = ("iid", lambda S: ols(S, "totN_", 3)[1] - 2 * _pp_all(S, "var") * S["N"] / S["n"])
    E["tau_g_ppp_implied"] = ("iid", lambda S: np.sqrt(np.maximum(
        ols(S, "totN_", 3)[1] - 2 * _pp_all(S, "var") * S["N"] / S["n"], 0) / 4) / (S["N"] / S["n"]))
    E["iid_sd_total"] = ("iid", lambda S: np.sqrt(2 * (_iid_team_var(S) + _pace_cov(S))))
    E["iid_sd_margin"] = ("iid", lambda S: np.sqrt(2 * (_iid_team_var(S) - _pace_cov(S))))
    for k in range(1, N_SEG + 1):
        E[f"var_lead_after_seg{k}"] = ("seg", lambda S, k=k: _var(S, "n", f"L{k}", f"L{k}sq"))
    E["var_lead_final"] = ("seg", lambda S: _var(S, "n", "Lf", "Lfsq"))
    for k in range(1, N_SEG + 1):
        E[f"var_total_vs_line_after_seg{k}"] = ("seg", lambda S, k=k: _var(S, "n", f"LT{k}", f"LT{k}sq"))
    for k in range(2, N_SEG + 1):
        E[f"rho_seg{k}"] = ("seg", lambda S, k=k: ols(S, f"rf{k}_", 3)[0][:, 1])
    E["rho_pooled"] = ("seg", lambda S: ols(S, "rfall_", 4)[0][:, 1])
    E["rho_pooled_pull_to_zero_coef"] = ("seg", lambda S: ols(S, "rfall_", 4)[0][:, 2])
    E["line_coef_pooled"] = ("seg", lambda S: ols(S, "rfall_", 4)[0][:, 3])
    for k in range(2, N_SEG + 1):
        E[f"rhoT_seg{k}"] = ("seg", lambda S, k=k: ols(S, f"rt{k}_", 3)[0][:, 1])
    E["rhoT_pooled"] = ("seg", lambda S: ols(S, "rtall_", 4)[0][:, 1])
    E["rhoT_pooled_pull_to_zero_coef"] = ("seg", lambda S: ols(S, "rtall_", 4)[0][:, 2])
    E["sd_dm_segment_resid"] = ("seg", lambda S: np.sqrt(ols(S, "rfall_", 4)[1]))
    for grp in ("q13", "q4"):
        for b, lab in enumerate(BIN_LABELS):
            E[f"leader_dm_vs_line_{grp}_m{lab}"] = (
                "seg", lambda S, g=grp, b=b: _ratio(S, f"lead_{g}_{b}_d", f"lead_{g}_{b}_n"))
    E["b4"] = ("b4", lambda S: ols(S, "b4_", 3)[0][:, 1])
    E["b4_line_coef"] = ("b4", lambda S: ols(S, "b4_", 3)[0][:, 2])
    E["b4_resid_sd"] = ("b4", lambda S: np.sqrt(ols(S, "b4_", 3)[1]))
    E["b4_no_line"] = ("b4", lambda S: ols(S, "b4u_", 2)[0][:, 1])
    E["b4_regulation"] = ("b4", lambda S: ols(S, "b4r_", 3)[0][:, 1])
    for b, lab in enumerate(BIN_LABELS):
        E[f"q4_poss_m{lab}"] = ("q4", lambda S, b=b: _ratio(S, f"q4_{b}_poss", f"q4_{b}_n"))
        E[f"q4_pace_ratio_m{lab}"] = ("q4", lambda S, b=b: _ratio(S, f"q4_{b}_poss", f"q4_{b}_base"))
        E[f"q4_share_games_m{lab}"] = ("q4", lambda S, b=b: S[f"q4_{b}_n"] / S["n"])
    for b, lab in enumerate(BIN_LABELS):
        E[f"q4_leader_ppp_m{lab}"] = ("q4", lambda S, b=b: _ratio(S, f"lt_{b}_lp", f"lt_{b}_lposs"))
        E[f"q4_trailer_ppp_m{lab}"] = ("q4", lambda S, b=b: _ratio(S, f"lt_{b}_tp", f"lt_{b}_tposs"))
        E[f"q4_leader_net100_m{lab}"] = ("q4", lambda S, b=b: 100 * (
            _ratio(S, f"lt_{b}_lp", f"lt_{b}_lposs") - _ratio(S, f"lt_{b}_tp", f"lt_{b}_tposs")))
        E[f"q4_leader_exp_net100_m{lab}"] = ("q4", lambda S, b=b: _ratio(S, f"lt_{b}_exp100", f"lt_{b}_n"))
    for kind in ("fg", "ft"):
        all_ = REB_CATS
        rebs = ["p_dreb", "p_oreb", "t_dreb", "t_oreb"]
        E[f"reb_{kind}_oreb_rate"] = ("reb", lambda S, k=kind: _reb(S, k, ["p_oreb", "t_oreb"], rebs))
        E[f"reb_{kind}_team_share"] = ("reb", lambda S, k=kind: _reb(S, k, ["t_dreb", "t_oreb"], rebs))
        E[f"reb_{kind}_team_share_of_dreb"] = ("reb", lambda S, k=kind: _reb(S, k, ["t_dreb"], ["t_dreb", "p_dreb"]))
        E[f"reb_{kind}_team_share_of_oreb"] = ("reb", lambda S, k=kind: _reb(S, k, ["t_oreb"], ["t_oreb", "p_oreb"]))
        E[f"reb_{kind}_deadball_share_of_team"] = ("reb", lambda S, k=kind: S[f"reb_{k}_t_deadball"] / (
            S[f"reb_{k}_t_dreb"] + S[f"reb_{k}_t_oreb"]))
        E[f"reb_{kind}_no_rebound"] = ("reb", lambda S, k=kind: _reb(S, k, ["none"], all_))
        E[f"reb_{kind}_misses_per_game"] = ("reb", lambda S, k=kind: sum(S[f"reb_{k}_{c}"] for c in all_) / S["n"])
    E["p_ot"] = ("ot", lambda S: S["ot"] / S["n"])
    for b, lab in enumerate(SPREAD_LABELS):
        E[f"p_ot_spread{lab}"] = ("ot", lambda S, b=b: _ratio(S, f"otb_{b}_ot", f"otb_{b}_n"))
    return E


# ------------------------------------------------------------------------- bootstrap

def boot_weights(seasons: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B+1, G) game weights; row 0 = the sample itself, rows 1.. resample games within season."""
    G = len(seasons)
    W = np.zeros((B + 1, G), dtype=np.float32)
    W[0] = 1
    for s in np.unique(seasons):
        idx = np.flatnonzero(seasons == s)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=B)
    return W


def evaluate(F: pd.DataFrame, seasons: np.ndarray, B: int, seed: int = 0) -> dict[str, tuple]:
    """name -> (estimate, lo95, hi95, group)."""
    W = boot_weights(seasons, B, np.random.default_rng(seed))
    M = W.astype(np.float64) @ F.to_numpy(np.float64)
    S = {c: M[:, i] for i, c in enumerate(F.columns)}
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, (grp, fn) in estimators().items():
            try:
                v = np.asarray(fn(S), float)
            except (KeyError, np.linalg.LinAlgError):
                continue
            lo, hi = np.nanpercentile(v[1:], [2.5, 97.5]) if B and np.isfinite(v[1:]).any() else (np.nan, np.nan)
            out[name] = (float(v[0]), float(lo), float(hi), grp)
    return out


# ------------------------------------------------------------------------- main

def build_table(workers: int, cache: str | None) -> tuple[pd.DataFrame, dict]:
    if cache and Path(cache).exists():
        print(f"loading per-game records from {cache}", file=sys.stderr)
        return pd.read_pickle(cache)
    seasons = FIT_SEASONS + OOS_SEASONS
    tasks = list_games(seasons)
    lines = load_lines(seasons).set_index("game_id")
    tasks = [t for t in tasks if t[0] in lines.index]
    print(f"{len(tasks)} games with PBP and a closing line", file=sys.stderr)
    rows, drops = [], {}
    with Pool(workers) as pool:
        for i, (gid, rec, err) in enumerate(pool.imap_unordered(_load, tasks, chunksize=16)):
            if rec is None:
                drops[err.split(":")[0]] = drops.get(err.split(":")[0], 0) + 1
                continue
            ln = lines.loc[gid]
            if (rec["final_h"], rec["final_a"]) != (ln.h_pts, ln.a_pts):
                drops["score_mismatch"] = drops.get("score_mismatch", 0) + 1
                continue
            if rec["periods"] < 4 or rec["bh"][8] + rec["ba"][8] < 100:
                drops["incomplete"] = drops.get("incomplete", 0) + 1
                continue
            rec.update(game_id=gid, season=ln.season, spread=ln.spread, total=ln.total)
            rows.append(rec)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(tasks)}", file=sys.stderr)
    recs = pd.DataFrame(rows).sort_values("game_id").reset_index(drop=True)
    meta = {"drops": drops, "n_tasks": len(tasks)}
    if cache:
        pd.to_pickle((recs, meta), cache)
    return recs, meta


def fmt(v: tuple) -> str:
    e, lo, hi, _ = v
    return f"{e:9.4f} [{lo:9.4f}, {hi:9.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cache", default=None, help="pickle of per-game records (reused if present)")
    a = ap.parse_args()
    t0 = time.time()
    recs, meta = build_table(a.workers, a.cache)
    t_parse = time.time() - t0
    F = pd.DataFrame([game_features(r) for r in recs.to_dict("records")]).fillna(0.0)
    seasons = recs.season.to_numpy()
    fit = np.isin(seasons, FIT_SEASONS)
    results: dict[str, dict] = {}
    results["fit"] = evaluate(F[fit], seasons[fit], a.boot, seed=1)
    for s in FIT_SEASONS + OOS_SEASONS:
        m = seasons == s
        results[s] = evaluate(F[m], seasons[m], a.boot, seed=2)
    oos = np.isin(seasons, OOS_SEASONS)
    results["oos"] = evaluate(F[oos], seasons[oos], a.boot, seed=3)

    names = list(results["fit"])
    lines = [f"sim step 0 (T2.5): fit {FIT_SEASONS[0]}..{FIT_SEASONS[-1]} regular season, "
             f"OOS {', '.join(OOS_SEASONS)}; B={a.boot} game-clustered bootstrap within season, 95% pct CI",
             f"games: {pd.Series(seasons).value_counts().sort_index().to_dict()}  "
             f"(of {meta['n_tasks']} with PBP+line; dropped {meta['drops']})", ""]
    grp = None
    for nme in names:
        g = results["fit"][nme][3]
        if g != grp:
            grp = g
            lines += ["", f"== {g}", f"{'quantity':38s} {'fit 2019-24 [95% CI]':>32s} {'OOS 2024-26 [95% CI]':>32s}  "
                      + " ".join(f"{s:>8s}" for s in FIT_SEASONS + OOS_SEASONS)]
        by = " ".join(f"{results[s].get(nme, (np.nan,) * 4)[0]:8.3f}" for s in FIT_SEASONS + OOS_SEASONS)
        lines.append(f"{nme:38s} {fmt(results['fit'][nme])} {fmt(results['oos'].get(nme, (np.nan,) * 4))}  {by}")
    pp_left = recs.pp_leftover.sum()
    lines += ["", f"points not attributed to a possession (period-end leftovers): {pp_left:.0f} "
              f"of {(recs.final_h + recs.final_a).sum():.0f}",
              f"runtime: parse {t_parse:.0f}s, total {time.time() - t0:.0f}s"]
    text = "\n".join(lines)
    OUT.write_text(text + "\n")
    print(text)

    tgt = {"meta": {"task": "T2.5 sim step 0", "fit_seasons": FIT_SEASONS, "oos_seasons": OOS_SEASONS,
                    "bootstrap": a.boot, "ci": "95% percentile, games resampled within season",
                    "games": {s: int((seasons == s).sum()) for s in FIT_SEASONS + OOS_SEASONS},
                    "dropped": meta["drops"], "segments": "eight 6-minute regulation segments",
                    "abs_m_bins": BIN_LABELS, "spread_bins": SPREAD_LABELS,
                    "conventions": {
                        "line": "E[H] = (total - spread)/2 with spread in betting convention (home -6.5 = favoured)",
                        "rho_segK": "OLS slope of dm over segment K on m at its start, controlling for -spread",
                        "rho_pooled": "pooled over segments 2-8: dm on [lead vs line L_k, e*k/8, e]; rho = coef on L_k",
                        "rho_pooled_pull_to_zero_coef": "coef on e*k/8: = rho if the pull is toward a 0 margin, 0 if toward the line's path",
                        "rhoT_segK": "OLS slope of the total's change over segment K on the running total at its start, controlling for the total line",
                        "var_team_given_poss_line": "residual var of team points on [1, own possessions, line E[team]]",
                        "lead_vs_line": "L_k = m_k - (-spread) k/8",
                        "leader_dm_vs_line": "leader's dm over the next segment minus line drift (e/8), by |m| at start",
                        "b4": "OLS slope of final margin (incl. OT) on m3, controlling for -spread",
                        "q4_pace_ratio": "Q4 possessions / mean Q1-Q3 quarter possessions",
                        "q4_leader_net100": "100 x (leader PPP - trailer PPP) in Q4; exp = line-implied net/100",
                        "ppp": "points per possession, counting method of pbp._mark_possessions",
                        "reb": "outcome of missed FGs / missed final FTs: player or team, off or def; none = no rebound logged",
                    }},
           "targets": {}}
    for nme in names:
        e, lo, hi, g = results["fit"][nme]
        tgt["targets"][nme] = {
            "group": g, "value": e, "ci95": [lo, hi],
            "by_season": {s: {"value": results[s][nme][0], "ci95": list(results[s][nme][1:3])}
                          for s in FIT_SEASONS if nme in results[s]},
            "oos": {s: {"value": results[s][nme][0], "ci95": list(results[s][nme][1:3])}
                    for s in OOS_SEASONS + ["oos"] if nme in results[s]}}
    TARGETS.write_text(json.dumps(tgt, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
