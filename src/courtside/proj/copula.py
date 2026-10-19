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
