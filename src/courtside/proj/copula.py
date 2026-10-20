"""Path A v0: copula simulation of one game (docs/PLAYERS.md §4.4-4.5, T1.4).

Per sim s of a game (home H, away A):

1. **Environment.** (margin, total) = market (closing line in backtests, the latest book line
   live, else gbm_market) + a bivariate normal error with the per-season sd(margin), sd(total)
   and corr fit on closing-line residuals of seasons < S (`fit_env`). Team points
   H_s = (T_s + M_s)/2, A_s = (T_s - M_s)/2. One overtime period when |M_s| < 1 (~6%, the real
   OT rate at sd 13.7).
2. **Availability scenario.** Independent Bernoulli(P(available)) per player. Each distinct
   absent set re-runs the minutes engine (`ScenarioMinutes`, cached by set). Per team at most
   `MAX_SCENARIOS` (16) distinct sets are projected: the most frequent sets until they cover
   `SCENARIO_MASS` (95%) of the sims, plus the "everybody uncertain plays" set. A rarer set A
   is *collapsed* onto the kept set B ⊆ A with the most absentees (ties: the more frequent):
   it uses B's redistribution of vacated minutes, and its extra absentees are zeroed and the
   team re-water-filled to 240. Marginal P(play) is therefore exact; only the redistribution of
   rare multi-absence sets is approximate.
3. **Minutes** = `minutes.sample_minutes` at the sim's margin (team perspective) and OT.
4. **DK FP given minutes.** Mean = minutes x per-minute rate (T1.1 EB rate x the short-minutes
   multiplier f_stat(min/mpg) x the usage bump when a top-2 star of the team is absent in this
   sim's scenario) + E[DD/TD bonus | minutes] (independent-NB approximation, `state_eval`).
   The team environment scales points by g_s = H_s / E[H] and the other DK stats by g_s^alpha
   (alpha = elasticity of non-points DK to team points, fit on seasons < S); `reconcile` then
   scales each team so that sum_i E[pts_i] equals the implied team total exactly.
   Marginal: normal when the player's projected DK (played) >= 25, gamma below (Wilson-Hilferty
   cube transform of the copula normal, renormalised to the exact mean), sd = CV(mu) x mu with
   the CV-by-level table fit on seasons < S on residuals *given actual minutes and the actual
   team points* (the minutes and environment variance come from steps 1-3, not twice).
5. **Gaussian copula** over the game's players: teammate correlation by role-tier pair
   (star / role / bench) and one opponent value, estimated on seasons < S from the same
   conditional residuals (residuals vs our pregame rate x actual minutes x actual environment).

DD/TD: v0 draws DK FP directly, so the expected bonus is in the mean and the tails are only
approximately right (a DD is not a discrete jump in the draws).

Public API
  CopulaParams (JSON), default_copula_params(season), fit_env(conn, season), fit_copula(...)
  ScenarioMinutes(panel, minutes_params, shares, starters).project(absent_mask) -> DataFrame
  TeamSpec / GameSpec, simulate_game(game, cp, mp, sp, n_sims, seed) -> GameSims
  corr_matrix(team, tiers, cp), draw_copula(R, n, rng), marginal_fp(mu, z, cv, normal)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import minutes as M
from . import reconcile as RC
from . import state as ST
from .state_eval import DD_CATS, expected_bonus

MODEL_VERSION = "pathA-v0.1"
DK_W = dict(ST.DK_WEIGHTS)                     # pts 1, fg3m .5, reb 1.25, ast 1.5, ...
DK_STATS = tuple(DK_W)
OTHER_STATS = tuple(s for s in DK_STATS if s != "pts")
PROP_STATS = ("pts", "reb", "ast", "fg3m")
TIERS = ("star", "role", "bench")
STAR_MIN, ROLE_MIN = 28.0, 20.0                # projected minutes for star / role tiers
NORMAL_MIN = 25.0                              # projected DK >= 25 -> normal, else gamma
GRID_MAX = 60                                  # minutes grid 0..60 (1-minute steps)
MAX_SCENARIOS = 16
SCENARIO_MASS = 0.95
OT_BAND = 1.0
QUANTS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CV_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 1e9)
BUMP_ROLES = ("none", "star", "rotation")


def pair_key(a: str, b: str) -> str:
    a, b = sorted((a, b), key=TIERS.index)
    return f"{a}|{b}"


# ============================================================================== params

@dataclass
class CopulaParams:
    """Everything the v0 simulator learns, fit on seasons < `season` (JSON-serialisable)."""
    season: str
    train_seasons: list = field(default_factory=list)
    sd_margin: float = 13.7
    sd_total: float = 18.0
    corr_mt: float = 0.0
    alpha: float = 0.5
    cv_mu: list = field(default_factory=lambda: [3.0, 7.5, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0,
                                                 55.0])
    cv: list = field(default_factory=lambda: [0.9, 0.7, 0.55, 0.45, 0.38, 0.33, 0.29, 0.26,
                                              0.24])
    normal_min: float = NORMAL_MIN
    rho_team: dict = field(default_factory=lambda: {pair_key(a, b): 0.03 for a in TIERS
                                                    for b in TIERS})
    rho_opp: float = 0.01
    env_weight: float = 1.0          # 1 = scale to the market environment (0: ablation)
    diagnostics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CopulaParams":
        return cls(**json.loads(s))


def default_copula_params(season: str = "2099-00") -> CopulaParams:
    return CopulaParams(season=season)


def cv_of(cp: CopulaParams, mu) -> np.ndarray:
    return np.interp(np.asarray(mu, float), cp.cv_mu, cp.cv)


# ============================================================================== fitting

def _season_label(s: int) -> str:
    return f"{s}-{(s + 1) % 100:02d}"


def closing_env(conn: sqlite3.Connection) -> pd.DataFrame:
    """Regular-season games with a closing line: game_id, season, margin/total (home) and the
    closing spread/total (mean over sources)."""
    d = pd.read_sql_query(
        """SELECT h.game_id, h.season, h.pts - a.pts AS margin, h.pts + a.pts AS total,
                  AVG(c.home_spread_close) AS hs, AVG(c.total_close) AS tc
           FROM team_games h JOIN team_games a ON a.game_id = h.game_id AND a.is_home = 0
           JOIN closing_lines c ON c.game_id = h.game_id
           WHERE h.is_home = 1 AND h.season_type = 'Regular Season'
             AND c.home_spread_close IS NOT NULL AND c.total_close IS NOT NULL
           GROUP BY h.game_id""", conn)
    d["rm"] = d["margin"] + d["hs"]
    d["rt"] = d["total"] - d["tc"]
    return d


def fit_env(conn: sqlite3.Connection, season: str, n_seasons: int = 3) -> dict:
    """sd(margin), sd(total), corr of the closing-line residuals on the `n_seasons` regular
    seasons before `season` (the T0.8 numbers; drift 12.4 -> 13.7 is why it is per season)."""
    d = closing_env(conn)
    seasons = sorted(s for s in d["season"].unique() if s < season)[-n_seasons:]
    d = d[d["season"].isin(seasons)]
    if len(d) < 50:
        return {"sd_margin": 13.7, "sd_total": 18.0, "corr_mt": 0.0, "env_seasons": seasons,
                "env_n": int(len(d))}
    return {"sd_margin": float(d["rm"].std()), "sd_total": float(d["rt"].std()),
            "corr_mt": float(np.corrcoef(d["rm"], d["rt"])[0, 1]), "env_seasons": seasons,
            "env_n": int(len(d))}


def role_tiers(team: np.ndarray, dk: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """Tier index per player (0 star, 1 role, 2 bench): stars = the team's top 2 by projected
    DK among players projected >= STAR_MIN minutes; role = projected >= ROLE_MIN."""
    team = np.asarray(team)
    dk = np.asarray(dk, float)
    minutes = np.asarray(minutes, float)
    out = np.where(minutes >= ROLE_MIN, 1, 2)
    for t in np.unique(team):
        idx = np.flatnonzero((team == t) & (minutes >= STAR_MIN))
        top = idx[np.argsort(-dk[idx], kind="stable")[:2]]
        out[top] = 0
    return out


def conditional_frame(wf: pd.DataFrame, sp: dict, team_pts: pd.DataFrame,
                      alpha: float | None = None) -> tuple[pd.DataFrame, float]:
    """Conditional means given ACTUAL minutes and ACTUAL team points for a walk_forward frame.

    raw_pts = r_pts x min x f_pts; raw_oth = sum of the other DK stats likewise; g = team points
    / sum raw_pts of the team-game; mu = raw_pts g + raw_oth g^alpha + E[bonus | min].
    alpha (if None) = slope of log(actual other DK / raw other DK) on log g over team-games."""
    w = wf.merge(team_pts, on=["game_id", "team"], how="inner").copy()
    mins = w["min"].to_numpy(float)
    mpg = w["mpg"].to_numpy(float)
    mu = {s: w[f"r_{s}"].to_numpy(float) * mins * ST.minutes_multiplier(sp, s, mins, mpg)
          for s in DK_STATS}
    w["raw_pts"] = mu["pts"]
    w["raw_oth"] = sum(DK_W[s] * mu[s] for s in OTHER_STATS)
    w["act_oth"] = sum(DK_W[s] * w[s].to_numpy(float) for s in OTHER_STATS)
    tier = w["tier"].to_numpy(int)
    phi = {s: ST.phi_for(sp, s, tier) for s in DD_CATS}
    w["bonus_mu"] = expected_bonus({s: mu[s] for s in DD_CATS}, phi)
    cats = sum((w[s].to_numpy() >= 10).astype(int) for s in DD_CATS)
    w["dk_act"] = (sum(DK_W[s] * w[s].to_numpy(float) for s in DK_STATS)
                   + 1.5 * (cats >= 2) + 3.0 * (cats >= 3))
    g = w.groupby(["game_id", "team"]).agg(raw_pts=("raw_pts", "sum"), raw_oth=("raw_oth", "sum"),
                                            act_oth=("act_oth", "sum"),
                                            team_pts=("team_pts", "first"))
    g["g"] = g["team_pts"] / g["raw_pts"].clip(lower=1.0)
    if alpha is None:
        ok = (g["g"] > 0.6) & (g["g"] < 1.6) & (g["raw_oth"] > 20) & (g["act_oth"] > 0)
        x = np.log(g.loc[ok, "g"].to_numpy())
        y = np.log(g.loc[ok, "act_oth"].to_numpy() / g.loc[ok, "raw_oth"].to_numpy())
        x = x - x.mean()
        alpha = float(np.clip((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-9), 0.0, 1.5))
    w = w.merge(g[["g"]].reset_index(), on=["game_id", "team"])
    w["mu"] = w["raw_pts"] * w["g"] + w["raw_oth"] * w["g"] ** alpha + w["bonus_mu"]
    w["resid"] = w["dk_act"] - w["mu"]
    return w, alpha


def fit_cv(mu: np.ndarray, resid: np.ndarray) -> tuple[list, list]:
    """CV by level of the conditional mean (sd / mean per bin, monotone non-increasing)."""
    b = np.clip(np.searchsorted(CV_EDGES, mu, side="right") - 1, 0, len(CV_EDGES) - 2)
    centres, cvs = [], []
    for k in range(len(CV_EDGES) - 1):
        m = b == k
        if m.sum() < 200:
            continue
        centres.append(float(mu[m].mean()))
        cvs.append(float(np.sqrt(np.mean(resid[m] ** 2)) / max(mu[m].mean(), 0.5)))
    cvs = list(np.minimum.accumulate(np.asarray(cvs))) if cvs else [0.5]
    return centres or [20.0], [float(c) for c in cvs]


def tier_pair_corr(df: pd.DataFrame, z: str = "z") -> tuple[dict, float, dict]:
    """Pooled Pearson correlations of z (mean 0, sd ~1) over teammate pairs by tier pair and
    over opponent pairs. df: game_id, team, tier (0..2), z. Returns (rho_team, rho_opp, n)."""
    d = df[["game_id", "team", "tier", z]].copy()
    d["z2"] = d[z] ** 2
    by = d.groupby(["game_id", "team", "tier"]).agg(s=(z, "sum"), q=("z2", "sum"),
                                                    n=(z, "size")).reset_index()
    wide_s = by.pivot_table(index=["game_id", "team"], columns="tier", values="s",
                            fill_value=0.0).reindex(columns=range(3), fill_value=0.0)
    wide_q = by.pivot_table(index=["game_id", "team"], columns="tier", values="q",
                            fill_value=0.0).reindex(columns=range(3), fill_value=0.0)
    wide_n = by.pivot_table(index=["game_id", "team"], columns="tier", values="n",
                            fill_value=0).reindex(columns=range(3), fill_value=0)
    S, Q, N = wide_s.to_numpy(), wide_q.to_numpy(), wide_n.to_numpy()
    var = Q.sum(0) / np.maximum(N.sum(0), 1)          # E[z^2] per tier
    rho, npairs = {}, {}
    for a in range(3):
        for b in range(a, 3):
            if a == b:
                num = ((S[:, a] ** 2 - Q[:, a]) / 2).sum()
                cnt = (N[:, a] * (N[:, a] - 1) / 2).sum()
                den = var[a]
            else:
                num = (S[:, a] * S[:, b]).sum()
                cnt = (N[:, a] * N[:, b]).sum()
                den = np.sqrt(var[a] * var[b])
            k = pair_key(TIERS[a], TIERS[b])
            rho[k] = float(num / max(cnt, 1) / max(den, 1e-9)) if cnt > 0 else 0.0
            npairs[k] = int(cnt)
    # opponents: team sums per game
    t = d.groupby(["game_id", "team"]).agg(s=(z, "sum"), n=(z, "size")).reset_index()
    t = t.sort_values(["game_id", "team"])
    first = t.groupby("game_id").nth(0).set_index("game_id")
    second = t.groupby("game_id").nth(1).set_index("game_id")
    j = first.join(second, lsuffix="_a", rsuffix="_b", how="inner")
    cnt = (j["n_a"] * j["n_b"]).sum()
    v = float(d["z2"].mean())
    rho_opp = float((j["s_a"] * j["s_b"]).sum() / max(cnt, 1) / max(v, 1e-9))
    npairs["opp"] = int(cnt)
    return rho, rho_opp, npairs


def fit_copula(conn: sqlite3.Connection, season: str, sp: dict, *, n_seasons: int = 2,
               games: pd.DataFrame | None = None, log=None) -> CopulaParams:
    """Fit CopulaParams on the `n_seasons` regular seasons before `season`: environment sds
    (3 seasons of closing lines), alpha, CV-by-level and tier-pair / opponent correlations of
    the conditional residuals. `sp` = rate-state params fit on seasons < `season`."""
    say = log or (lambda m: None)
    s0 = int(season[:4])
    seasons = [_season_label(s0 - k) for k in range(n_seasons, 0, -1)]
    first = _season_label(s0 - n_seasons - sp.get("history_seasons", 6))
    if games is None:
        games = ST.load_games(conn, since_season=first, until_season=seasons[-1])
    wf = pd.concat([ST.walk_forward(conn, s, sp, games=games) for s in seasons],
                   ignore_index=True)
    wf = wf[wf["season_type"] == "Regular Season"]
    tp = pd.read_sql_query("SELECT game_id, team, pts AS team_pts FROM team_games WHERE "
                           "season >= ? AND season <= ?", conn, params=(seasons[0], seasons[-1]))
    w, alpha = conditional_frame(wf, sp, tp)
    w = w[w["mu"] > 0.5]
    cv_mu, cv = fit_cv(w["mu"].to_numpy(), w["resid"].to_numpy())
    cp = CopulaParams(season=season, train_seasons=seasons, alpha=alpha, cv_mu=cv_mu, cv=cv)
    cp.__dict__.update({k: v for k, v in fit_env(conn, season).items()
                        if k in ("sd_margin", "sd_total", "corr_mt")})
    w = w.copy()
    w["z"] = w["resid"] / (cv_of(cp, w["mu"]) * w["mu"])
    w["z"] = w["z"].clip(-6, 6)
    w["tier"] = role_tiers((w["game_id"] + "|" + w["team"]).to_numpy(),
                           (w["r_dk"] * w["mpg"]).to_numpy(), w["mpg"].to_numpy())
    rho, rho_opp, npairs = tier_pair_corr(w)
    cp.rho_team = rho
    cp.rho_opp = rho_opp
    env = fit_env(conn, season)
    cp.diagnostics = {"n": int(len(w)), "pairs": npairs, "z_sd": float(w["z"].std()),
                      "env_seasons": env["env_seasons"], "env_n": env["env_n"],
                      "bias_mean": float(w["resid"].mean())}
    say(f"copula fit {season}: alpha {alpha:.3f}, sd_m {cp.sd_margin:.2f}, sd_t "
        f"{cp.sd_total:.2f}, rho_team {json.dumps({k: round(v, 3) for k, v in rho.items()})}, "
        f"rho_opp {rho_opp:.3f}")
    return cp


# ============================================================================== copula pieces

def corr_matrix(team: np.ndarray, tiers: np.ndarray, cp: CopulaParams) -> np.ndarray:
    """Correlation matrix over a game's players: teammates by tier pair, opponents rho_opp;
    projected to the nearest PSD correlation matrix (eigenvalue clipping)."""
    team = np.asarray(team)
    tiers = np.asarray(tiers, int)
    n = len(team)
    R = np.full((n, n), float(cp.rho_opp))
    for i in range(n):
        for j in range(i + 1, n):
            if team[i] == team[j]:
                r = cp.rho_team.get(pair_key(TIERS[tiers[i]], TIERS[tiers[j]]), 0.0)
                R[i, j] = R[j, i] = r
    np.fill_diagonal(R, 1.0)
    return nearest_corr(R)


def nearest_corr(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, V = np.linalg.eigh(R)
    if w.min() >= eps:
        return R
    w = np.maximum(w, eps)
    A = (V * w) @ V.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def draw_copula(R: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, k) standard normals with correlation R."""
    L = np.linalg.cholesky(R)
    return rng.standard_normal((n, R.shape[0])) @ L.T


def marginal_fp(mu: np.ndarray, z: np.ndarray, cv: np.ndarray, normal: np.ndarray
                ) -> np.ndarray:
    """DK FP draws with mean mu and sd cv x mu from copula normals z: normal columns
    (`normal` mask) mu + sd z (floored at -5); gamma columns by the Wilson-Hilferty cube
    k theta (a + b z)^3, a = 1 - 1/(9k), b = 1/(3 sqrt k), k = 1/cv^2, renormalised to the
    exact mean (E[(a + b z)^3] = a^3 + 3 a b^2)."""
    mu = np.maximum(mu, 0.0)
    cv = np.maximum(cv, 0.05)
    k = 1.0 / cv ** 2
    a = 1.0 - 1.0 / (9.0 * k)
    b = 1.0 / (3.0 * np.sqrt(k))
    g = np.maximum(a + b * z, 0.0) ** 3 / (a ** 3 + 3.0 * a * b ** 2)
    out = np.where(normal, np.maximum(mu + cv * mu * z, -5.0), mu * g)
    return out


# ============================================================================== minutes scenarios

class ScenarioMinutes:
    """The minutes engine for one team-game, re-run per absent set (cached by the set).

    `panel` = `minutes.pregame_panel` rows of the team-game (one row per candidate; their
    order is the team's player order everywhere downstream). `starters`: confirmed starter
    player_ids (lock-10) or None."""

    def __init__(self, panel: pd.DataFrame, params: M.MinutesParams,
                 shares: M.PairShares | None = None, starters: Iterable[int] | None = None):
        p = panel.reset_index(drop=True).copy()
        if "absent" not in p:
            p["absent"] = False
        self.panel = p
        self.params = params
        self.shares = shares
        self.pids = p["player_id"].to_numpy(np.int64)
        self.date = int(p["date"].iloc[0]) if len(p) else 0
        self.g1 = p["g1"].to_numpy(int)
        self.g2 = p["g2"].to_numpy(int)
        self.st10 = p["st10"].to_numpy(float)
        self.ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
        self.tgt = M._team_target(p, params)
        self.starters = (None if starters is None
                         else p["player_id"].isin({int(s) for s in starters}).to_numpy())
        self._cache: dict[bytes, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.pids)

    def project(self, absent: np.ndarray) -> pd.DataFrame:
        absent = np.asarray(absent, bool)
        key = absent.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = self.panel
        base, pp = M.base_and_pplay(p, self.params, absent)
        share = (self.shares.matrix(self.date, self.pids, absent)
                 if self.shares is not None else None)
        mean, det = M.project_arrays(base, pp, absent, self.g1, self.g2, self.st10, self.ew,
                                     self.tgt, self.params, starters=self.starters,
                                     share=share, detail=True)
        out = pd.DataFrame({"player_id": self.pids, "mean": mean,
                            "p_play": np.where(absent, 0.0, pp),
                            "pf36": p["pf36"].to_numpy(float), "reserve": det["reserve"],
                            "base": base, "lift": det["lift"], "pstart": det["pstart"],
                            "promoted": det["promoted"], "exp_start": det["exp_start"]})
        self._cache[key] = out
        return out

    @property
    def n_projected(self) -> int:
        return len(self._cache)


# ============================================================================== game spec

@dataclass
class TeamSpec:
    """One team of a game. `players` (in `minutes` panel order): player_id, p_avail
    (P available for the scenario draw; 0 = certain out), r_<DK stat>, mpg, tier_state (rate
    state tier, for the NB dispersion of the bonus), star (top-2 regular star), rotation
    (mpg >= 15). `implied` = target team points (book team total / market split)."""
    team: str
    players: pd.DataFrame
    minutes: ScenarioMinutes
    implied: float


@dataclass
class GameSpec:
    game_id: str
    home: TeamSpec
    away: TeamSpec
    mu_margin: float            # expected home margin
    mu_total: float
    env_source: str = ""


@dataclass
class GameSims:
    game_id: str
    player_id: np.ndarray       # (n,)
    team: np.ndarray            # (n,) team codes
    fp: np.ndarray              # (S, n) float32 DK points (0 when not playing)
    minutes: np.ndarray         # (S, n) float32
    info: pd.DataFrame          # per player: p_avail, p_play, tier, normal, stat means ...
    teams: dict                 # team -> reconciliation / scenario info
    margin: np.ndarray          # (S,) home margin draws
    total: np.ndarray           # (S,)
    stats: dict | None = None   # v1 only: COMP_STATS -> (S, n) uint8 counts (0 when not playing)


# ============================================================================== grids

def _grids(players: pd.DataFrame, sp: dict) -> dict[str, np.ndarray]:
    """Per player, per bump role (none/star/rotation), per minute 0..GRID_MAX: expected pts,
    other DK, reb, ast, fg3m given minutes (rate x m x f(m/mpg) x bump) and the DD/TD bonus."""
    n = len(players)
    m = np.arange(GRID_MAX + 1, dtype=float)
    mpg = players["mpg"].to_numpy(float)
    mm = np.broadcast_to(m[None, :], (n, len(m)))
    mpgb = np.broadcast_to(mpg[:, None], (n, len(m)))
    base = {}
    for s in DK_STATS:
        r = players[f"r_{s}"].to_numpy(float)
        base[s] = r[:, None] * mm * ST.minutes_multiplier(sp, s, mm, mpgb)
    out = {k: np.zeros((3, n, len(m))) for k in ("pts", "oth", "reb", "ast", "fg3m")}
    for ri, role in enumerate(BUMP_ROLES):
        bump = {s: (ST.bump_for(sp, role, s) if role != "none" else 1.0) for s in DK_STATS}
        out["pts"][ri] = base["pts"] * bump["pts"]
        out["oth"][ri] = sum(DK_W[s] * base[s] * bump[s] for s in OTHER_STATS)
        for s in ("reb", "ast", "fg3m"):
            out[s][ri] = base[s] * bump[s]
    tier = players["tier_state"].to_numpy(int)
    phi = {s: np.repeat(ST.phi_for(sp, s, tier)[:, None], len(m), 1).ravel() for s in DD_CATS}
    bon = expected_bonus({s: base[s].ravel() for s in DD_CATS}, phi)
    out["bonus"] = bon.reshape(n, len(m))
    return out


def _lookup(grid: np.ndarray, role: np.ndarray, x: np.ndarray) -> np.ndarray:
    """grid (R, n, G) [or (n, G) with role None] at per-sim minutes x (S, n), linear in x."""
    x = np.clip(x, 0.0, GRID_MAX - 1e-9)
    lo = np.floor(x).astype(np.int64)
    w = x - lo
    n, G = grid.shape[-2], grid.shape[-1]
    col = np.arange(n)[None, :] * G
    if grid.ndim == 3:
        off = role.astype(np.int64) * (n * G) + col + lo
    else:
        off = col + lo
    flat = grid.ravel()
    return flat[off] * (1.0 - w) + flat[off + 1] * w


# ============================================================================== scenarios

def draw_scenarios(p_avail: np.ndarray, n: int, rng: np.random.Generator,
                   max_kept: int = MAX_SCENARIOS, mass: float = SCENARIO_MASS) -> dict:
    """Independent availability draws for one team and their collapse onto <= max_kept
    projected absent sets (see module doc). Returns absent (S, k) bool for every player,
    kept (list of (k,) bool absent masks), assign (S,) index into kept, and the realised
    distribution."""
    p = np.clip(np.asarray(p_avail, float), 0.0, 1.0)
    certain_out = p <= 0.0
    unc = np.flatnonzero((p > 0.0) & (p < 1.0))
    absent = np.repeat(certain_out[None, :], n, 0)
    if not len(unc):
        return {"absent": absent, "kept": [certain_out.copy()], "assign": np.zeros(n, int),
                "n_distinct": 1, "kept_mass": 1.0, "collapsed": 0}
    bits = rng.random((n, len(unc))) >= p[unc][None, :]
    absent[:, unc] = bits
    uniq, inv, cnt = np.unique(bits, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(-cnt, kind="stable")
    kept_u: list[int] = []
    cum = 0
    for u in order:
        if len(kept_u) >= max_kept - 1 or cum >= mass * n:
            break
        kept_u.append(int(u))
        cum += cnt[u]
    zero = np.zeros(len(unc), bool)
    kept_bits = [uniq[u] for u in kept_u]
    if not any((b == zero).all() for b in kept_bits):
        kept_bits.append(zero)              # the fallback every rarer set contains
    kept_cnt = [int(cnt[u]) for u in kept_u] + [0] * (len(kept_bits) - len(kept_u))
    KB = np.array(kept_bits)                # (K, u)
    pop = KB.sum(1)
    map_u = np.empty(len(uniq), int)
    collapsed = 0
    for u in range(len(uniq)):
        sub = np.flatnonzero(~(KB & ~uniq[u][None, :]).any(1))       # B subset of A
        exact = sub[(KB[sub] == uniq[u]).all(1)]
        if len(exact):
            map_u[u] = int(exact[0])
            continue
        collapsed += int(cnt[u])
        best = sorted(sub, key=lambda b: (-pop[b], -kept_cnt[b]))[0]
        map_u[u] = int(best)
    kept = []
    for b in KB:
        a = certain_out.copy()
        a[unc] = b
        kept.append(a)
    return {"absent": absent, "kept": kept, "assign": map_u[inv], "n_distinct": int(len(uniq)),
            "kept_mass": float(cum / n), "collapsed": collapsed}


# ============================================================================== simulation

def _seed_seq(seed) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, (list, tuple)):
        return np.random.SeedSequence([int(s) for s in seed])
    return np.random.SeedSequence(int(seed))


def _team_minutes(ts: TeamSpec, mp: M.MinutesParams, margin_team: np.ndarray,
                  n_ot: np.ndarray, rng_s: np.random.Generator, rng_m: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """(minutes (S, n), absent (S, n), scenario info) for one team."""
    S = len(margin_team)
    pl = ts.players
    sc = draw_scenarios(pl["p_avail"].to_numpy(float), S, rng_s)
    x = np.zeros((S, len(pl)))
    for b, kept in enumerate(sc["kept"]):
        idx = np.flatnonzero(sc["assign"] == b)
        if not len(idx):
            continue
        proj = ts.minutes.project(kept)
        xb = M.sample_minutes(proj, mp, margin_team[idx], rng_m, n_ot=n_ot[idx])
        extra = sc["absent"][idx] & ~kept[None, :]
        rows = extra.any(1)
        if rows.any():
            xr = np.where(extra[rows], 0.0, xb[rows])
            ot = n_ot[idx][rows]
            tot = M.TEAM_MIN + M.OT_TEAM_MIN * ot - M._reserve_of(proj)
            xb[rows] = M._renorm(xr, tot, M.CAP + 5.0 * ot)
        x[idx] = xb
    x = np.where(sc["absent"], 0.0, x)
    info = {"n_distinct": sc["n_distinct"], "n_projected": len(sc["kept"]),
            "kept_mass": sc["kept_mass"], "collapsed_sims": sc["collapsed"]}
    return x, sc["absent"], info


def simulate_game(game: GameSpec, cp: CopulaParams, mp: M.MinutesParams, sp: dict,
                  n_sims: int, seed) -> GameSims:
    """Joint DK FP sims for one game (module doc, steps 1-5). Deterministic in `seed`."""
    ss = _seed_seq(seed)
    r_env, r_sc, r_min, r_cop = (np.random.default_rng(s) for s in ss.spawn(4))
    S = int(n_sims)
    cov = np.array([[cp.sd_margin ** 2, cp.corr_mt * cp.sd_margin * cp.sd_total],
                    [cp.corr_mt * cp.sd_margin * cp.sd_total, cp.sd_total ** 2]])
    e = r_env.standard_normal((S, 2)) @ np.linalg.cholesky(cov).T
    margin = game.mu_margin + e[:, 0]
    total = game.mu_total + e[:, 1]
    n_ot = (np.abs(margin) < OT_BAND).astype(float)
    team_pts = {"home": (total + margin) / 2.0, "away": (total - margin) / 2.0}
    mean_pts = {"home": (game.mu_total + game.mu_margin) / 2.0,
                "away": (game.mu_total - game.mu_margin) / 2.0}
    sides = (("home", game.home, 1.0), ("away", game.away, -1.0))
    parts, teams_info = [], {}
    for side, ts, sign in sides:
        pl = ts.players.reset_index(drop=True)
        x, absent, sinfo = _team_minutes(ts, mp, sign * margin, n_ot, r_sc, r_min)
        played = x > 0
        star = pl["star"].to_numpy(bool)
        rot = pl["rotation"].to_numpy(bool) & ~star
        star_out = (absent & star[None, :]).any(1)
        role = np.where(star_out[:, None] & star[None, :] & ~absent, 1, 0)
        role = np.where(star_out[:, None] & rot[None, :] & ~absent, 2, role)
        G = _grids(pl, sp)
        raw_pts = _lookup(G["pts"], role, x) * played
        raw_oth = _lookup(G["oth"], role, x) * played
        bonus = _lookup(G["bonus"], None, x) * played
        g = team_pts[side] / max(mean_pts[side], 1.0)
        rec = RC.reconcile_team(raw_pts, raw_oth, g, cp.alpha, ts.implied, cp.env_weight)
        pts, oth, k_pace = rec.pop("pts"), rec.pop("oth"), rec.pop("pace")
        mu = pts + oth + bonus
        stats = {"pts": pts}
        for s in ("reb", "ast", "fg3m"):
            stats[s] = _lookup(G[s], role, x) * played * k_pace[:, None]
        rec.update(sinfo)
        rec["team"] = ts.team
        teams_info[ts.team] = rec
        parts.append((ts, pl, x, played, mu, stats))
    # tiers / families from the pre-noise projection
    pid = np.concatenate([p[1]["player_id"].to_numpy(np.int64) for p in parts])
    team = np.concatenate([np.full(len(p[1]), p[0].team, dtype=object) for p in parts])
    x = np.concatenate([p[2] for p in parts], axis=1)
    played = np.concatenate([p[3] for p in parts], axis=1)
    mu = np.concatenate([p[4] for p in parts], axis=1)
    npl = np.maximum(played.sum(0), 1)
    mu_played = mu.sum(0) / npl
    min_played = x.sum(0) / npl
    tiers = role_tiers(team, mu_played, min_played)
    normal = mu_played >= cp.normal_min
    R = corr_matrix(team, tiers, cp)
    z = draw_copula(R, S, r_cop)
    fp = marginal_fp(mu, z, cv_of(cp, mu), normal[None, :])
    fp = np.where(played, fp, 0.0)
    info = pd.DataFrame({"player_id": pid, "team": team,
                         "p_avail": np.concatenate([p[1]["p_avail"].to_numpy(float)
                                                    for p in parts]),
                         "p_play": played.mean(0), "tier": [TIERS[t] for t in tiers],
                         "normal": normal, "mu_played": mu_played, "min_played": min_played})
    for s in PROP_STATS:
        v = np.concatenate([p[5][s] for p in parts], axis=1)
        info[f"{s}_all"] = v.mean(0)
        info[f"{s}_played"] = v.sum(0) / npl
    return GameSims(game_id=game.game_id, player_id=pid, team=team, fp=fp.astype(np.float32),
                    minutes=x.astype(np.float32), info=info, teams=teams_info, margin=margin,
                    total=total)


# ============================================================================== summaries

def summarize_counts(x: np.ndarray, played: np.ndarray) -> pd.DataFrame:
    """`summarize` for uint8 count sims (S, n), vectorised through per-column histograms:
    mean, sd and **count quantiles** (the smallest k with P(Y <= k) >= q) over all sims and over
    played sims. Same frame layout as `summarize` (cond, col, mean, sd, q05..q95)."""
    S, n = x.shape
    idx = (x.astype(np.int64) + 256 * np.arange(n)[None, :]).ravel()
    k = np.arange(256, dtype=float)
    rows = []
    for cond, wts in (("all", None), ("played", played.ravel().astype(float))):
        h = np.bincount(idx, weights=wts, minlength=256 * n).reshape(n, 256)
        tot = h.sum(1)
        cdf = np.cumsum(h, 1) / np.maximum(tot, 1e-12)[:, None]
        mean = (h @ k) / np.maximum(tot, 1e-12)
        var = (h @ (k * k)) / np.maximum(tot, 1e-12) - mean ** 2
        qs = {f"q{int(round(q * 100)):02d}": (cdf >= q - 1e-12).argmax(1).astype(float)
              for q in QUANTS}
        for j in range(n):
            r = {"cond": cond, "col": j}
            if tot[j] > 0:
                r.update({"mean": float(mean[j]), "sd": float(np.sqrt(max(var[j], 0.0)))})
                r.update({c: v[j] for c, v in qs.items()})
            else:
                r.update({"mean": None, "sd": None})
                r.update({c: None for c in qs})
            rows.append(r)
    out = pd.DataFrame(rows)
    order = np.argsort(np.r_[np.arange(n) * 2, np.arange(n) * 2 + 1], kind="stable")
    return out.iloc[order].reset_index(drop=True)


def summarize(sims: np.ndarray, played: np.ndarray | None = None) -> pd.DataFrame:
    """Per column: mean, sd, q05..q95 over all sims ('all') and over played sims ('played')."""
    sims = np.asarray(sims, np.float64)
    if played is None:
        played = sims != 0
    rows = []
    qs = np.quantile(sims, QUANTS, axis=0) if len(sims) else np.zeros((len(QUANTS), 0))
    for j in range(sims.shape[1]):
        a = {"cond": "all", "col": j, "mean": float(sims[:, j].mean()),
             "sd": float(sims[:, j].std())}
        a.update({f"q{int(round(q * 100)):02d}": float(qs[k, j]) for k, q in enumerate(QUANTS)})
        rows.append(a)
        v = sims[played[:, j], j]
        b = {"cond": "played", "col": j}
        if len(v):
            b.update({"mean": float(v.mean()), "sd": float(v.std())})
            qv = np.quantile(v, QUANTS)
            b.update({f"q{int(round(q * 100)):02d}": float(qv[k]) for k, q in enumerate(QUANTS)})
        else:
            b.update({"mean": None, "sd": None})
            b.update({f"q{int(round(q * 100)):02d}": None for q in QUANTS})
        rows.append(b)
    return pd.DataFrame(rows)


def sim_corr(fp: np.ndarray, i: Sequence[int], j: Sequence[int]) -> np.ndarray:
    """Correlation of sim columns i[k], j[k] (for the copula recovery check)."""
    a = fp[:, list(i)] - fp[:, list(i)].mean(0)
    b = fp[:, list(j)] - fp[:, list(j)].mean(0)
    return (a * b).mean(0) / np.maximum(a.std(0) * b.std(0), 1e-12)


# ============================================================================== v1 (T2.10)
#
# Path A copula v1: the same environment, availability, minutes and reconciliation as v0, but
# per player and sim it draws the box components (COMP_STATS) instead of DK points:
#
#   * means: rate x minutes x f(min/mpg) x usage bump, pts scaled by the team factor g c and the
#     other stats by (g c)^alpha (exactly v0's reconciliation), then the **winner/loser shift**
#     f_{stat,tier}(signed margin of the sim) fit on seasons < S given actual minutes and actual
#     team points (points are re-normalised per team and sim, so the shift only moves points
#     between tiers and the market total is still met);
#   * marginals: NB2 counts, var = m + c2(m) m^2, with c2 by level per stat fit on seasons < S on
#     the same conditional residuals (the **CV-by-level fit** of v1). Draws are inverse-CDF table
#     lookups (`CountTables`: NU quantile cells x a mean grid, stochastic rounding between the two
#     neighbouring grid means so E[count] equals the mean exactly);
#   * within a player: a 7 x 7 latent Gaussian correlation per role tier (`r_within`), fit by
#     simulated moments so the simulated count covariances match the observed residual ones;
#   * across players: the T2.7 causal tier x position matrices (mode PM, `corr_<S>.json`, seasons
#     < S) on each player's latent **DK direction**: Z_i = b_i w_i + E_i with w ~ N(0, R_players),
#     b_i = R_w a_i / sqrt(a_i' R_w a_i), a_i = DK weight x Hermite-1 coefficient of the stat at
#     the player's level, E_i ~ N(0, R_w - b_i b_i'). Then Var(Z_i) = R_w and the DK direction of
#     player i is exactly w_i, so the player-level correlations are those of the matrix;
#   * DK points are scored exactly from the counts (DK Classic table of `dfs.rules`), so a
#     double-double / triple-double is a discrete +1.5 / +4.5 jump.
#
# The first four RNG streams (environment, scenarios, minutes, player copula) are v0's, so with
# the same seed v1 has the same margins, totals, availability and minutes as v0 (common random
# numbers); two extra streams draw the component noise and the grid rounding.

COMP_STATS = DK_STATS                          # pts, fg3m, reb, ast, stl, blk, tov
N_COMP = len(COMP_STATS)
DD_STATS = ("pts", "reb", "ast", "stl", "blk")
COPULA_V1_VERSION = "copula-v1.0"
NU = 4096                                      # quantile cells per mean grid row
N_MEAN = 321                                   # mean grid rows (0 .. MEAN_MAX)
KMAX = 255                                     # counts are uint8
MEAN_MAX = {"pts": 80.0, "fg3m": 12.0, "reb": 32.0, "ast": 24.0, "stl": 8.0, "blk": 10.0,
            "tov": 12.0}
C2_MIN = 1e-3
WL_EDGES = (-1e9, -20.0, -12.0, -6.0, -2.0, 2.0, 6.0, 12.0, 20.0, 1e9)
WL_K = 300.0                                   # shrinkage of a bin ratio toward 1 (rows)
WL_BOUNDS = (0.8, 1.25)
DISP_BINS = 16
_SCORING_KEYS = {"pts": "point", "fg3m": "three_pointer_made", "reb": "rebound",
                 "ast": "assist", "stl": "steal", "blk": "block", "tov": "turnover"}


def dk_scoring() -> dict:
    """DK Classic weights per component + the double / triple-double bonuses, from the table
    `dfs.rules.dk_points` scores with (`sources.dfs.CLASSIC_SCORING`)."""
    from .dfs import rules as R
    sc = R.dfs.CLASSIC_SCORING
    out = {s: float(sc[k]) for s, k in _SCORING_KEYS.items()}
    out["dd"] = float(sc["double_double"])
    out["td"] = float(sc["triple_double"])
    return out


def dk_from_components(stats: dict, scoring: dict | None = None) -> np.ndarray:
    """DK Classic points of count arrays {stat: array} (same shape), with the real DD / TD
    bonuses: 10+ in two of pts/reb/ast/stl/blk adds dd, in three adds td on top (= 4.5)."""
    sc = scoring or dk_scoring()
    tot = sum(sc[s] * np.asarray(stats[s], np.float32) for s in COMP_STATS)
    cats = sum((np.asarray(stats[s]) >= 10).astype(np.int8) for s in DD_STATS)
    return tot + sc["dd"] * (cats >= 2) + sc["td"] * (cats >= 3)


@dataclass
class CopulaV1Params:
    """What v1 learns on seasons < `season` (JSON). Defaults = a usable prior (NB1-like
    dispersion, a mild pts-fg3m link, no winner/loser shift, v0's tier-pair correlations)."""
    season: str
    train_seasons: list = field(default_factory=list)
    disp_mu: dict = field(default_factory=dict)        # stat -> level centres
    disp_c2: dict = field(default_factory=dict)        # stat -> NB2 excess c^2 at the centres
    r_within: dict = field(default_factory=dict)       # tier -> 7 x 7 (COMP_STATS order)
    wl_centres: list = field(default_factory=list)     # signed team margin
    wl_factor: dict = field(default_factory=dict)      # stat -> tier -> factor per centre
    corr: dict = field(default_factory=dict)           # T2.7 PM table (+ "source")
    diagnostics: dict = field(default_factory=dict)
    version: str = COPULA_V1_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CopulaV1Params":
        return cls(**json.loads(s))


_DEFAULT_PHI = {"pts": 2.3, "fg3m": 1.1, "reb": 1.15, "ast": 1.2, "stl": 1.05, "blk": 1.15,
                "tov": 1.05}


def default_copula_v1_params(season: str = "2099-00",
                             cp: CopulaParams | None = None) -> CopulaV1Params:
    lv = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    v = CopulaV1Params(season=season)
    for s in COMP_STATS:
        mu = [x for x in lv if x <= MEAN_MAX[s]]
        v.disp_mu[s] = mu
        v.disp_c2[s] = [max((_DEFAULT_PHI[s] - 1.0) / m, C2_MIN) for m in mu]
    R = np.eye(N_COMP)
    i, j = COMP_STATS.index("pts"), COMP_STATS.index("fg3m")
    R[i, j] = R[j, i] = 0.5
    v.r_within = {t: R.tolist() for t in TIERS}
    cp = cp or default_copula_params(season)
    v.corr = {"rho_team": dict(cp.rho_team), "rho_team_cell": {}, "rho_team_all": 0.0,
              "rho_opp": {pair_key(a, b): cp.rho_opp for a in TIERS for b in TIERS},
              "rho_opp_cell": {}, "rho_opp_all": cp.rho_opp, "source": "copula-v0"}
    return v


# ------------------------------------------------------------------------------ count tables

class CountTables:
    """Inverse-CDF tables of the NB2 marginals of a CopulaV1Params: per stat a mean grid
    (N_MEAN rows over 0..MEAN_MAX) x NU quantile cells (the quantile at each cell's midpoint),
    uint8. `draw` picks between the two grid rows around a mean with probability given by the
    position (exact mean), `h` is the Hermite-1 coefficient E[Y Z] of a row (Y = Q(Phi(Z)))."""

    def __init__(self, v1: CopulaV1Params):
        from scipy import stats as sps
        from scipy.special import ndtri
        self.step, self.tab, self.h_row, self.mean_row, self.var_row = {}, {}, {}, {}, {}
        edges = ndtri(np.arange(NU + 1) / NU)
        dens = np.exp(-0.5 * np.nan_to_num(edges, posinf=0.0, neginf=0.0) ** 2) / np.sqrt(2 * np.pi)
        dens[0] = dens[-1] = 0.0
        wz = dens[:-1] - dens[1:]                        # integral of z over each u cell
        u_mid = (np.arange(NU) + 0.5) / NU
        k = np.arange(KMAX + 1, dtype=float)
        for s in COMP_STATS:
            grid = np.linspace(0.0, MEAN_MAX[s], N_MEAN)
            c2 = np.maximum(np.interp(grid, v1.disp_mu[s], v1.disp_c2[s]), C2_MIN)
            r = 1.0 / c2
            m = np.maximum(grid, 1e-9)
            pmf = sps.nbinom.pmf(k[None, :], r[:, None], (r / (r + m))[:, None])
            pmf[0] = 0.0
            pmf[0, 0] = 1.0
            cdf = np.cumsum(pmf, axis=1)
            cdf[:, -1] = 1.0
            tab = np.empty((N_MEAN, NU), np.uint8)
            for i in range(N_MEAN):
                tab[i] = np.minimum(np.searchsorted(cdf[i], u_mid, side="left"), KMAX)
            self.step[s] = grid[1] - grid[0]
            self.tab[s] = tab.ravel()
            tf = tab.astype(float)
            self.h_row[s] = tf @ wz
            self.mean_row[s] = tf.mean(1)
            self.var_row[s] = tf.var(1)

    def _row(self, s: str, m: np.ndarray, rnd: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(m, float) / self.step[s], 0.0, N_MEAN - 1 - 1e-9)
        lo = np.floor(x)
        return (lo + (rnd < (x - lo))).astype(np.int64)

    def draw(self, s: str, m: np.ndarray, u: np.ndarray, rnd: np.ndarray) -> np.ndarray:
        """Counts with mean m (any shape) at copula uniforms u; rnd = U(0,1) for the rounding."""
        col = np.minimum((np.asarray(u) * NU).astype(np.int64), NU - 1)
        return self.tab[s][self._row(s, m, rnd) * NU + col]

    def h(self, s: str, m) -> np.ndarray:
        x = np.clip(np.asarray(m, float) / self.step[s], 0.0, N_MEAN - 1)
        return np.interp(x, np.arange(N_MEAN), self.h_row[s])

    def var(self, s: str, m) -> np.ndarray:
        x = np.clip(np.asarray(m, float) / self.step[s], 0.0, N_MEAN - 1)
        return np.interp(x, np.arange(N_MEAN), self.var_row[s])


_TABLES: dict[str, CountTables] = {}


def count_tables(v1: CopulaV1Params) -> CountTables:
    """CountTables of `v1`'s dispersion (built once per process per dispersion table)."""
    key = json.dumps([v1.disp_mu, v1.disp_c2], sort_keys=True)
    t = _TABLES.get(key)
    if t is None:
        if len(_TABLES) > 8:
            _TABLES.clear()
        t = _TABLES[key] = CountTables(v1)
    return t


# ------------------------------------------------------------------------------ v1 pieces

def _stat_grids(players: pd.DataFrame, sp: dict) -> dict[str, np.ndarray]:
    """Per stat (COMP_STATS): (3 bump roles, n, GRID_MAX + 1) expected count given minutes
    (rate x m x f(m/mpg) x bump), the per-stat version of `_grids`."""
    n = len(players)
    m = np.arange(GRID_MAX + 1, dtype=float)
    mpg = players["mpg"].to_numpy(float)
    mm = np.broadcast_to(m[None, :], (n, len(m)))
    mpgb = np.broadcast_to(mpg[:, None], (n, len(m)))
    out = {}
    for s in COMP_STATS:
        r = players[f"r_{s}"].to_numpy(float)
        base = r[:, None] * mm * ST.minutes_multiplier(sp, s, mm, mpgb)
        out[s] = np.stack([base * (ST.bump_for(sp, role, s) if role != "none" else 1.0)
                           for role in BUMP_ROLES])
    return out


def wl_factor(v1: CopulaV1Params, stat: str, tiers: np.ndarray, margin: np.ndarray
              ) -> np.ndarray:
    """Winner/loser per-minute factor (S, n) for players of one team: margin (S,) is the team's
    signed margin per sim, tiers (n,) 0..2. 1 everywhere when not fitted."""
    tab = v1.wl_factor.get(stat)
    if not tab or not v1.wl_centres:
        return np.ones((len(margin), len(tiers)))
    out = np.empty((len(margin), len(tiers)))
    for t in range(len(TIERS)):
        cols = np.flatnonzero(np.asarray(tiers) == t)
        if len(cols):
            f = np.interp(margin, v1.wl_centres, tab[TIERS[t]])
            out[:, cols] = f[:, None]
    return out


def corr_matrix_v1(team: np.ndarray, tiers: np.ndarray, pos: np.ndarray | None,
                   v1: CopulaV1Params) -> np.ndarray:
    """Player correlation matrix from v1's T2.7 table: teammates by tier x position cell when
    both positions are known (G/F/C), else by tier pair; opponents by tier pair. Nearest PSD."""
    c = v1.corr
    team = np.asarray(team)
    tiers = np.asarray(tiers, int)
    n = len(team)
    posl = ("G", "F", "C", "U")
    R = np.empty((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = team[i] == team[j]
            tp = pair_key(TIERS[tiers[i]], TIERS[tiers[j]])
            tab, cells, allv = ((c.get("rho_team", {}), c.get("rho_team_cell", {}),
                                 c.get("rho_team_all", 0.0)) if same else
                                (c.get("rho_opp", {}), c.get("rho_opp_cell", {}),
                                 c.get("rho_opp_all", 0.0)))
            r = tab.get(tp, allv)
            if same and pos is not None and pos[i] < 3 and pos[j] < 3 and cells:
                a = f"{TIERS[tiers[i]]}:{posl[pos[i]]}"
                b = f"{TIERS[tiers[j]]}:{posl[pos[j]]}"
                key = "|".join(sorted((a, b), key=lambda g: (TIERS.index(g.split(":")[0]),
                                                             posl.index(g.split(":")[1]))))
                r = cells.get(key, r)
            R[i, j] = R[j, i] = r
    np.fill_diagonal(R, 1.0)
    return nearest_corr(R)


def _loadings(a: np.ndarray, Rw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For one player: b = R_w a / sqrt(a' R_w a) and L with L L' = R_w - b b' (eigen square
    root, negative eigenvalues clipped). a = 0 -> b = 0, L = chol-like root of R_w."""
    q = float(a @ Rw @ a)
    b = Rw @ a / np.sqrt(q) if q > 1e-12 else np.zeros(len(a))
    Cm = Rw - np.outer(b, b)
    w, V = np.linalg.eigh((Cm + Cm.T) / 2.0)
    return b, V * np.sqrt(np.maximum(w, 0.0))[None, :]


def _r_within(v1: CopulaV1Params, tier: int) -> np.ndarray:
    R = v1.r_within.get(TIERS[tier])
    return np.eye(N_COMP) if R is None else np.asarray(R, float)


def draw_components(means: dict, played: np.ndarray, w: np.ndarray, tiers: np.ndarray,
                    v1: CopulaV1Params, tables: CountTables, r_cmp: np.random.Generator,
                    r_rnd: np.random.Generator) -> dict:
    """Counts {stat: (S, n) uint8} given per-sim means {stat: (S, n)}, the player-level copula
    normals w (S, n) and tiers (n,). Players who sit are 0."""
    S, n = played.shape
    npl = np.maximum(played.sum(0), 1)
    mbar = np.stack([means[s].sum(0) / npl for s in COMP_STATS], 1)          # (n, 7)
    sc = dk_scoring()
    wdk = np.array([sc[s] for s in COMP_STATS])
    hb = np.stack([tables.h(s, mbar[:, k]) for k, s in enumerate(COMP_STATS)], 1)
    B = np.zeros((n, N_COMP))
    L = np.zeros((n, N_COMP, N_COMP))
    for i in range(n):
        B[i], L[i] = _loadings(wdk * hb[i], _r_within(v1, int(tiers[i])))
    N = r_cmp.standard_normal((S, n, N_COMP), dtype=np.float32)
    Z = np.einsum("snk,njk->snj", N, L.astype(np.float32))
    Z += w.astype(np.float32)[:, :, None] * B.astype(np.float32)[None, :, :]
    from scipy.special import ndtr
    out = {}
    for k, s in enumerate(COMP_STATS):
        rnd = r_rnd.random((S, n), dtype=np.float32)
        y = tables.draw(s, means[s], ndtr(Z[:, :, k]), rnd)
        out[s] = np.where(played, y, 0).astype(np.uint8)
    return out


def simulate_game_v1(game: GameSpec, cp: CopulaParams, v1: CopulaV1Params,
                     mp: M.MinutesParams, sp: dict, n_sims: int, seed) -> GameSims:
    """Component sims for one game (module section v1). Deterministic in `seed`; margins,
    totals, availability and minutes equal `simulate_game`'s for the same seed."""
    ss = _seed_seq(seed)
    r_env, r_sc, r_min, r_cop = (np.random.default_rng(s) for s in ss.spawn(4))
    r_cmp, r_rnd = (np.random.default_rng(s) for s in ss.spawn(2))
    S = int(n_sims)
    cov = np.array([[cp.sd_margin ** 2, cp.corr_mt * cp.sd_margin * cp.sd_total],
                    [cp.corr_mt * cp.sd_margin * cp.sd_total, cp.sd_total ** 2]])
    e = r_env.standard_normal((S, 2)) @ np.linalg.cholesky(cov).T
    margin = game.mu_margin + e[:, 0]
    total = game.mu_total + e[:, 1]
    n_ot = (np.abs(margin) < OT_BAND).astype(float)
    team_pts = {"home": (total + margin) / 2.0, "away": (total - margin) / 2.0}
    mean_pts = {"home": (game.mu_total + game.mu_margin) / 2.0,
                "away": (game.mu_total - game.mu_margin) / 2.0}
    sides = (("home", game.home, 1.0), ("away", game.away, -1.0))
    parts, teams_info = [], {}
    for side, ts, sign in sides:
        pl = ts.players.reset_index(drop=True)
        x, absent, sinfo = _team_minutes(ts, mp, sign * margin, n_ot, r_sc, r_min)
        played = x > 0
        star = pl["star"].to_numpy(bool)
        rot = pl["rotation"].to_numpy(bool) & ~star
        star_out = (absent & star[None, :]).any(1)
        role = np.where(star_out[:, None] & star[None, :] & ~absent, 1, 0)
        role = np.where(star_out[:, None] & rot[None, :] & ~absent, 2, role)
        G = _grids(pl, sp)
        GS = _stat_grids(pl, sp)
        raw_pts = _lookup(G["pts"], role, x) * played
        raw_oth = _lookup(G["oth"], role, x) * played
        bonus = _lookup(G["bonus"], None, x) * played
        g = team_pts[side] / max(mean_pts[side], 1.0)
        rec = RC.reconcile_team(raw_pts, raw_oth, g, cp.alpha, ts.implied, cp.env_weight)
        pts, oth, k_pace = rec.pop("pts"), rec.pop("oth"), rec.pop("pace")
        means = {"pts": pts}
        for s in OTHER_STATS:
            means[s] = _lookup(GS[s], role, x) * played * k_pace[:, None]
        rec.update(sinfo)
        rec["team"] = ts.team
        teams_info[ts.team] = rec
        parts.append((ts, pl, x, played, pts + oth + bonus, means, sign))
    pid = np.concatenate([p[1]["player_id"].to_numpy(np.int64) for p in parts])
    team = np.concatenate([np.full(len(p[1]), p[0].team, dtype=object) for p in parts])
    x = np.concatenate([p[2] for p in parts], axis=1)
    played = np.concatenate([p[3] for p in parts], axis=1)
    mu = np.concatenate([p[4] for p in parts], axis=1)
    npl = np.maximum(played.sum(0), 1)
    mu_played = mu.sum(0) / npl
    min_played = x.sum(0) / npl
    tiers = role_tiers(team, mu_played, min_played)
    # winner/loser shift per team (points re-normalised per sim: the team's points are kept)
    means = {s: [] for s in COMP_STATS}
    off = 0
    for ts, pl, xt, pt, _, mt, sign in parts:
        tt = tiers[off:off + len(pl)]
        off += len(pl)
        for s in COMP_STATS:
            f = wl_factor(v1, s, tt, sign * margin)
            v = mt[s] * f
            if s == "pts":
                before = mt[s].sum(1)
                after = v.sum(1)
                v = v * np.where(after > 0, before / np.maximum(after, 1e-12), 1.0)[:, None]
            means[s].append(v)
    means = {s: np.concatenate(v, axis=1) for s, v in means.items()}
    pos = (np.concatenate([p[1]["grp"].to_numpy(int) for p in parts])
           if all("grp" in p[1] for p in parts) else None)
    R = corr_matrix_v1(team, tiers, pos, v1)
    w = draw_copula(R, S, r_cop)
    cnt = draw_components(means, played, w, tiers, v1, count_tables(v1), r_cmp, r_rnd)
    fp = np.where(played, dk_from_components(cnt), 0.0)
    info = pd.DataFrame({"player_id": pid, "team": team,
                         "p_avail": np.concatenate([p[1]["p_avail"].to_numpy(float)
                                                    for p in parts]),
                         "p_play": played.mean(0), "tier": [TIERS[t] for t in tiers],
                         "normal": mu_played >= cp.normal_min, "mu_played": mu_played,
                         "min_played": min_played})
    for s in PROP_STATS:
        v = cnt[s].astype(float)
        info[f"{s}_all"] = v.mean(0)
        info[f"{s}_played"] = v.sum(0) / npl
    return GameSims(game_id=game.game_id, player_id=pid, team=team, fp=fp.astype(np.float32),
                    minutes=x.astype(np.float32), info=info, teams=teams_info, margin=margin,
                    total=total, stats=cnt)


# ------------------------------------------------------------------------------ v1 fitting

def v1_frame(conn: sqlite3.Connection, season: str, sp: dict, alpha: float, *,
             n_seasons: int = 2, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-stat conditional means given ACTUAL minutes and ACTUAL team points (the v1 analogue
    of `conditional_frame`) on the `n_seasons` regular seasons before `season`: columns game_id,
    team, season, tier (0..2), margin (team perspective), y_<stat>, m_<stat> (pts x g, others x
    g^alpha, g = team points / sum of rate x minutes points of the team-game), dk_act, mu_dk."""
    s0 = int(season[:4])
    seasons = [_season_label(s0 - k) for k in range(n_seasons, 0, -1)]
    first = _season_label(s0 - n_seasons - sp.get("history_seasons", 6))
    if games is None:
        games = ST.load_games(conn, since_season=first, until_season=seasons[-1])
    wf = pd.concat([ST.walk_forward(conn, s, sp, games=games) for s in seasons],
                   ignore_index=True)
    wf = wf[wf["season_type"] == "Regular Season"]
    tp = pd.read_sql_query(
        "SELECT t.game_id, t.team, t.pts AS team_pts, t.pts - o.pts AS margin FROM team_games t "
        "JOIN team_games o ON o.game_id = t.game_id AND o.team != t.team "
        "WHERE t.season >= ? AND t.season <= ?", conn, params=(seasons[0], seasons[-1]))
    w = wf.merge(tp, on=["game_id", "team"], how="inner").reset_index(drop=True)
    mins = w["min"].to_numpy(float)
    mpg = w["mpg"].to_numpy(float)
    raw = {s: w[f"r_{s}"].to_numpy(float) * mins * ST.minutes_multiplier(sp, s, mins, mpg)
           for s in COMP_STATS}
    key = (w["game_id"] + "|" + w["team"]).to_numpy()
    rp = pd.Series(raw["pts"]).groupby(key).transform("sum").to_numpy()
    g = w["team_pts"].to_numpy(float) / np.maximum(rp, 1.0)
    out = pd.DataFrame({"game_id": w["game_id"].to_numpy(), "team": w["team"].to_numpy(),
                        "season": w["season"].to_numpy(),
                        "margin": w["margin"].to_numpy(float), "g": g})
    for s in COMP_STATS:
        out[f"y_{s}"] = w[s].to_numpy(float)
        out[f"m_{s}"] = raw[s] * (g if s == "pts" else g ** alpha)
    out["tier"] = role_tiers(key, (w["r_dk"] * w["mpg"]).to_numpy(), w["mpg"].to_numpy())
    out["dk_act"] = dk_from_components({s: out[f"y_{s}"].to_numpy() for s in COMP_STATS})
    out["mu_dk"] = sum(DK_W[s] * out[f"m_{s}"] for s in COMP_STATS)
    ok = (out["g"] > 0.5) & (out["g"] < 2.0) & (out["mu_dk"] > 0.5)
    return out[ok].reset_index(drop=True)


def fit_wl(f: pd.DataFrame, k: float = WL_K) -> tuple[list, dict, dict]:
    """Winner/loser factors: per stat and tier, the ratio sum(y) / sum(m) in each signed-margin
    bin over the tier's overall ratio, shrunk to 1 by n / (n + k), clipped to WL_BOUNDS.
    Returns (centres, {stat: {tier: [factor per bin]}}, n per tier and bin)."""
    b = np.clip(np.searchsorted(WL_EDGES, f["margin"].to_numpy(), side="right") - 1, 0,
                len(WL_EDGES) - 2)
    nb = len(WL_EDGES) - 1
    centres = [float(f["margin"].to_numpy()[b == i].mean()) if (b == i).any()
               else float(np.clip((WL_EDGES[i] + WL_EDGES[i + 1]) / 2, -30, 30))
               for i in range(nb)]
    tier = f["tier"].to_numpy(int)
    fac, ns = {}, {}
    for s in COMP_STATS:
        y, m = f[f"y_{s}"].to_numpy(), f[f"m_{s}"].to_numpy()
        fac[s] = {}
        for t in range(len(TIERS)):
            mt = tier == t
            base = y[mt].sum() / max(m[mt].sum(), 1e-9)
            row = []
            for i in range(nb):
                sel = mt & (b == i)
                n = int(sel.sum())
                ns[f"{TIERS[t]}:{i}"] = n
                if n == 0 or m[sel].sum() <= 0:
                    row.append(1.0)
                    continue
                r = (y[sel].sum() / m[sel].sum()) / max(base, 1e-9)
                row.append(float(np.clip(1.0 + (r - 1.0) * n / (n + k), *WL_BOUNDS)))
            fac[s][TIERS[t]] = row
    return centres, fac, ns


def apply_wl(f: pd.DataFrame, v1: CopulaV1Params) -> pd.DataFrame:
    """m_<stat> x the winner/loser factor at the actual margin; points re-normalised per
    team-game (the team's points are conditioned on)."""
    f = f.copy()
    tier = f["tier"].to_numpy(int)
    key = (f["game_id"] + "|" + f["team"]).to_numpy()
    for s in COMP_STATS:
        fac = np.ones(len(f))
        for t in range(len(TIERS)):
            sel = tier == t
            fac[sel] = np.interp(f["margin"].to_numpy()[sel], v1.wl_centres,
                                 v1.wl_factor[s][TIERS[t]])
        m = f[f"m_{s}"].to_numpy() * fac
        if s == "pts":
            before = pd.Series(f[f"m_{s}"].to_numpy()).groupby(key).transform("sum").to_numpy()
            after = pd.Series(m).groupby(key).transform("sum").to_numpy()
            m = m * before / np.maximum(after, 1e-12)
        f[f"m_{s}"] = m
    f["mu_dk"] = sum(DK_W[s] * f[f"m_{s}"] for s in COMP_STATS)
    return f


def fit_dispersion(f: pd.DataFrame, n_bins: int = DISP_BINS, min_n: int = 300
                   ) -> tuple[dict, dict]:
    """NB2 excess c^2 by level per stat: in quantile bins of m, c^2 = (Var(y - m) - E[m]) /
    E[m^2], floored at C2_MIN, smoothed over neighbouring bins (the v1 CV-by-level fit)."""
    mus, c2s = {}, {}
    for s in COMP_STATS:
        y, m = f[f"y_{s}"].to_numpy(), f[f"m_{s}"].to_numpy()
        ok = m > 0.01
        y, m = y[ok], m[ok]
        q = np.unique(np.quantile(m, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(q, m, side="right") - 1, 0, len(q) - 2)
        cen, val = [], []
        for i in range(len(q) - 1):
            sel = b == i
            if sel.sum() < min_n:
                continue
            r = y[sel] - m[sel]
            v = r.var()
            cen.append(float(m[sel].mean()))
            val.append(float((v - m[sel].mean()) / np.mean(m[sel] ** 2)))
        val = np.asarray(val)
        if len(val) >= 3:
            sm = val.copy()
            sm[1:-1] = (val[:-2] + val[1:-1] + val[2:]) / 3.0
            val = sm
        mus[s] = cen or [1.0]
        c2s[s] = [float(max(v, C2_MIN)) for v in val] or [0.1]
    return mus, c2s


def _cov_obs(f: pd.DataFrame, tier: int) -> tuple[np.ndarray, int]:
    d = f[f["tier"] == tier]
    r = np.stack([d[f"y_{s}"].to_numpy() - d[f"m_{s}"].to_numpy() for s in COMP_STATS], 1)
    r = r - r.mean(0)
    return r.T @ r / max(len(r), 1), len(r)


def _cov_sim(f: pd.DataFrame, tier: int, R: np.ndarray, tables: CountTables, reps: int,
             rng: np.random.Generator) -> np.ndarray:
    d = f[f["tier"] == tier]
    m = np.stack([d[f"m_{s}"].to_numpy() for s in COMP_STATS], 1)
    m = np.repeat(m, reps, axis=0)
    w, V = np.linalg.eigh(R)
    Lr = V * np.sqrt(np.maximum(w, 0.0))[None, :]
    z = rng.standard_normal(m.shape) @ Lr.T
    from scipy.special import ndtr
    u = ndtr(z)
    y = np.stack([tables.draw(s, m[:, k], u[:, k], rng.random(len(m))).astype(float)
                  for k, s in enumerate(COMP_STATS)], 1)
    r = y - m
    r = r - r.mean(0)
    return r.T @ r / max(len(r), 1)


def fit_within(f: pd.DataFrame, tables: CountTables, *, iters: int = 4, reps: int = 2,
               seed: int = 0) -> tuple[dict, dict]:
    """Latent within-player correlation per tier by simulated moments: start from the
    Hermite-1 estimate cov_obs / E[h_s h_t], then Newton-like steps on (cov_obs - cov_sim) /
    E[h_s h_t] until the simulated count covariances match the observed ones. Returns
    ({tier: 7x7}, diagnostics: observed vs simulated correlation per tier)."""
    rng = np.random.default_rng(seed)
    out, diag = {}, {}
    for t in range(len(TIERS)):
        d = f[f["tier"] == t]
        if len(d) < 500:
            out[TIERS[t]] = np.eye(N_COMP).tolist()
            continue
        C_obs, n = _cov_obs(f, t)
        H = np.stack([tables.h(s, d[f"m_{s}"].to_numpy()) for s in COMP_STATS], 1)
        HH = H.T @ H / len(H)
        R = np.clip(C_obs / np.maximum(HH, 1e-9), -0.95, 0.95)
        np.fill_diagonal(R, 1.0)
        R = nearest_corr(R)
        for _ in range(iters):
            C_sim = _cov_sim(f, t, R, tables, reps, rng)
            R = R + (C_obs - C_sim) / np.maximum(HH, 1e-9)
            R = np.clip((R + R.T) / 2.0, -0.95, 0.95)
            np.fill_diagonal(R, 1.0)
            R = nearest_corr(R)
        C_sim = _cov_sim(f, t, R, tables, reps, rng)
        sd_o, sd_s = np.sqrt(np.diag(C_obs)), np.sqrt(np.diag(C_sim))
        out[TIERS[t]] = R.tolist()
        diag[TIERS[t]] = {"n": n, "corr_obs": (C_obs / np.outer(sd_o, sd_o)).round(3).tolist(),
                          "corr_sim": (C_sim / np.outer(sd_s, sd_s)).round(3).tolist(),
                          "sd_obs": sd_o.round(3).tolist(), "sd_sim": sd_s.round(3).tolist()}
    return out, diag


def _dk_check(f: pd.DataFrame, v1: CopulaV1Params, tables: CountTables, reps: int = 2,
              seed: int = 1) -> dict:
    """Implied DK sd given minutes and environment by level band (v1 sims at the frame's means)
    vs the observed RMS of the DK residual (real bonuses), and the bonus rate."""
    from scipy.special import ndtr
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(len(TIERS)):
        d = f[f["tier"] == t]
        if not len(d):
            continue
        m = np.repeat(np.stack([d[f"m_{s}"].to_numpy() for s in COMP_STATS], 1), reps, 0)
        R = _r_within(v1, t)
        w, V = np.linalg.eigh(R)
        z = rng.standard_normal(m.shape) @ (V * np.sqrt(np.maximum(w, 0.0))[None, :]).T
        u = ndtr(z)
        y = {s: tables.draw(s, m[:, k], u[:, k], rng.random(len(m)))
             for k, s in enumerate(COMP_STATS)}
        rows.append(pd.DataFrame({"mu": np.repeat(d["mu_dk"].to_numpy(), reps),
                                  "sim": dk_from_components(y),
                                  "act": np.repeat(d["dk_act"].to_numpy(), reps)}))
    a = pd.concat(rows, ignore_index=True)
    a["band"] = pd.cut(a["mu"], [0, 10, 20, 30, 40, 50, 200])
    out = {}
    for band, g in a.groupby("band", observed=True):
        rs, ra = g["sim"] - g["mu"], g["act"] - g["mu"]
        out[str(band)] = {"n": int(len(g) // reps), "mu": round(float(g["mu"].mean()), 2),
                          "sd_sim": round(float(rs.std()), 3),
                          "sd_act": round(float(ra.std()), 3),
                          "mean_sim": round(float(g["sim"].mean()), 3),
                          "mean_act": round(float(g["act"].mean()), 3)}
    return out


def corr_from_t27(season: str, conn: sqlite3.Connection | None = None, log=None) -> dict:
    """The T2.7 causal PM table for season S (`corr_<S>.json`, fit on seasons < S; fitted from
    the cached extracts of the 3 seasons before S when the file is missing). {} when neither
    exists (the caller falls back to v0's tier pairs)."""
    from . import corr as CR
    say = log or (lambda m: None)
    p = None
    try:
        p = CR.load(season)
    except FileNotFoundError:
        if conn is not None:
            s0 = int(season[:4])
            seasons = [_season_label(s0 - k) for k in (3, 2, 1)]
            try:
                p = CR.fit(conn, seasons, season=season, k_shrink=2000.0)
                CR.save(p)
            except FileNotFoundError as e:
                say(f"copula v1 {season}: no T2.7 table ({e}); using v0 tier pairs")
                return {}
    if p is None or any(s >= season for s in p.train_seasons):
        return {}
    t = p.table("PM")
    return {"rho_team": t.rho_team, "rho_team_cell": t.rho_team_cell,
            "rho_team_all": t.rho_team_all, "rho_opp": t.rho_opp, "rho_opp_cell": {},
            "rho_opp_all": t.rho_opp_all,
            "source": f"corr {p.version} PM {season} train {p.train_seasons} k {p.k_shrink}"}


def fit_copula_v1(conn: sqlite3.Connection, season: str, sp: dict, cp: CopulaParams, *,
                  n_seasons: int = 2, games: pd.DataFrame | None = None, frame=None,
                  log=None) -> CopulaV1Params:
    """Fit CopulaV1Params on the regular seasons before `season` (module section v1): the
    winner/loser shift, the per-stat dispersion by level, the within-player correlations per
    tier, and the T2.7 player correlations. `cp` = the season's v0 params (alpha)."""
    say = log or (lambda m: None)
    f = frame if frame is not None else v1_frame(conn, season, sp, cp.alpha,
                                                 n_seasons=n_seasons, games=games)
    v1 = default_copula_v1_params(season, cp)
    v1.train_seasons = sorted(f["season"].unique().tolist())
    if any(s >= season for s in v1.train_seasons):
        raise ValueError(f"copula v1 {season}: training rows from {v1.train_seasons}")
    v1.wl_centres, v1.wl_factor, wl_n = fit_wl(f)
    f = apply_wl(f, v1)
    v1.disp_mu, v1.disp_c2 = fit_dispersion(f)
    tables = count_tables(v1)
    v1.r_within, wdiag = fit_within(f, tables)
    c = corr_from_t27(season, conn, log=say)
    if c:
        v1.corr = c
    v1.diagnostics = {"n": int(len(f)), "within": wdiag, "wl_n": wl_n,
                      "dk_by_level": _dk_check(f, v1, tables), "corr_source":
                      v1.corr.get("source")}
    say(f"copula v1 fit {season}: n {len(f)}, corr {v1.corr.get('source')}, "
        f"r(pts,fg3m) star {v1.r_within['star'][0][1]:.2f}")
    return v1
