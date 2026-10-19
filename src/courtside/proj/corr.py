"""Causal DK-residual correlation parameters (docs/PLAYERS.md §4.5 Path A step 5, T2.7).

The v1 study (`research/dfs_correlations.py`) measured residuals against each player's own
in-season mean (hindsight). Here the residual is

    res = actual DK - OUR pregame projection mean,

where the projection comes from `pipeline.project_historical(protocol="avail")` fit on seasons
< S and uses only data before the game date (availability by the §7.1 L4 oracle). Role tiers are
the copula's own (`copula.role_tiers` on the projected mean and projected minutes), so a tier at
date d never uses d or later outcomes. `tiers_to_date` is an independent causal tiering from
the players' prior played games (robustness check, star-absence detection).

Residual modes (what the residual is conditioned on):

    U   unconditional: dk - mu                       (what a sim must reproduce end to end)
    P   given the actual team points: dk - mu (1 + b_tier (g - 1)), g = team pts / implied pts
    PM  given team points and actual minutes: P scaled by actual / projected minutes
        (the construction of T1.4's `conditional_frame`; the copula's input)

Correlations are pair-pooled moments of the (optionally CV-standardised) centred residual over
the 12 groups tier (star / role / bench) x position (G / F / C / unknown), cluster-bootstrapped
over games. `fit` stores teammate correlations by tier pair and by tier x position cell,
opponent correlations likewise, each shrunk toward its parent by pair count:

    r_cell -> r_tier_pair -> r_all.

Public API
  extract_season(conn, season, ...)        project_historical over a season -> cached extract
  load_frame(conn, seasons)                extract + actuals + positions -> residual frame
  fit(conn_or_frame, seasons) -> CorrParams ; save(p) ; load(season)
  CorrParams.corr_matrix(team, tiers, pos=None, ...) -> nearest-PSD matrix for a game
  PairStats(...)                           sufficient statistics + game-cluster bootstrap
  tiers_to_date(pg) ; shrink(r, n, prior, k) ; fit_cv(mu, res) ; nearest_corr(R)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

MODEL_VERSION = "corr-v2.0"
TIERS = ("star", "role", "bench")
POS = ("G", "F", "C", "U")
NPOS = len(POS)
NG = len(TIERS) * NPOS                      # 12 groups: tier * 4 + pos
MODES = ("U", "P", "PM")
DEFAULT_MODE = "PM"
STAR_MIN, ROLE_MIN = 28.0, 20.0             # same thresholds as copula.role_tiers
K_SHRINK = 4000.0                           # pair-count prior strength (research script picks it)
CV_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 1e9)
MIN_PROJ_MU = 0.5


def cache_dir() -> Path:
    return config.data_dir() / "proj"


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def next_season(season: str) -> str:
    return season_label(int(season[:4]) + 1)


def group_label(g: int) -> str:
    return f"{TIERS[g // NPOS]}:{POS[g % NPOS]}"


def _group_key(s: str) -> tuple:
    t, _, p = s.partition(":")
    return (TIERS.index(t), POS.index(p) if p else 0)


def pair_label(a: str, b: str) -> str:
    """Canonical 'x|y' for tiers ('star') or groups ('star:G'): higher tier first, then G<F<C<U."""
    a, b = sorted((a, b), key=_group_key)
    return f"{a}|{b}"


# ============================================================================== small pieces

def dk_points(df: pd.DataFrame) -> pd.Series:
    """DK Classic points; DD +1.5 and TD +3 on top (a triple-double is +4.5)."""
    cats = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    p = (df["pts"] + 0.5 * df["fg3m"] + 1.25 * df["reb"] + 1.5 * df["ast"] + 2 * df["stl"]
         + 2 * df["blk"] - 0.5 * df["tov"])
    return p + 1.5 * (cats >= 2) + 3.0 * (cats >= 3)


def shrink(r, n, prior, k: float):
    """Empirical-Bayes style pull toward `prior`: (n r + k prior) / (n + k). n = 0 -> prior,
    n >> k -> r. Vectorised."""
    r = np.asarray(r, float)
    n = np.asarray(n, float)
    prior = np.asarray(prior, float)
    w = n / (n + max(float(k), 0.0)) if k > 0 else np.where(n > 0, 1.0, 0.0)
    out = w * np.nan_to_num(r) + (1.0 - w) * prior
    return out if out.ndim else float(out)


def nearest_corr(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Nearest PSD correlation matrix (eigenvalue clipping, unit diagonal restored)."""
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    if w.min() >= eps and np.allclose(np.diag(R), 1.0):
        return R
    A = (V * np.maximum(w, eps)) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return (A + A.T) / 2.0


def fit_cv(mu: np.ndarray, resid: np.ndarray, min_n: int = 200) -> tuple[list, list]:
    """CV by level (RMS residual / mean level per bin of the reference mean), made monotone
    non-increasing. Same construction as copula.fit_cv."""
    mu = np.asarray(mu, float)
    resid = np.asarray(resid, float)
    b = np.clip(np.searchsorted(CV_EDGES, mu, side="right") - 1, 0, len(CV_EDGES) - 2)
    centres, cvs = [], []
    for k in range(len(CV_EDGES) - 1):
        m = b == k
        if m.sum() < min_n:
            continue
        centres.append(float(mu[m].mean()))
        cvs.append(float(np.sqrt(np.mean(resid[m] ** 2)) / max(mu[m].mean(), 0.5)))
    cvs = list(np.minimum.accumulate(np.asarray(cvs))) if cvs else [0.5]
    return centres or [20.0], [float(c) for c in cvs]


def cv_at(cv_mu, cv, mu) -> np.ndarray:
    return np.interp(np.asarray(mu, float), cv_mu, cv)


# ============================================================================== causal tiers

def tiers_to_date(pg: pd.DataFrame, window: int = 20, min_games: int = 5) -> pd.Series:
    """Role tier per player-game (0 star, 1 role, 2 bench) from the player's PRIOR played games
    only. pg: game_id, team, player_id, date (sortable), min, dk (played rows only).

    Player level: mean minutes and mean DK of the previous `window` played games (shift(1),
    min_periods `min_games`; NaN history -> bench). Team level: stars = the two highest prior
    DK mean among players with prior minutes >= STAR_MIN, role = prior minutes >= ROLE_MIN.
    Nothing at or after the row's own date enters its tier."""
    d = pg[["game_id", "team", "player_id", "date", "min", "dk"]].copy()
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["player_id", "date", "game_id"], kind="stable")
    g = d.groupby("player_id", sort=False)
    for c in ("min", "dk"):
        d["p_" + c] = g[c].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_games).mean())
    pm = d["p_min"].to_numpy(float)
    pk = d["p_dk"].to_numpy(float)
    tier = np.where(pm >= ROLE_MIN, 1, 2)
    d["_t"] = tier
    d["_k"] = np.where(pm >= STAR_MIN, pk, -np.inf)
    d["_rk"] = d.groupby(["game_id", "team"])["_k"].rank(ascending=False, method="first")
    d.loc[(d["_rk"] <= 2) & np.isfinite(d["_k"]), "_t"] = 0
    d = d.sort_values("_o")
    return pd.Series(d["_t"].to_numpy(int), index=pg.index, name="tier_td")


# ============================================================================== extraction

def _regular_days(ctx, season: str) -> list:
    tg = ctx.h.tg
    rs = tg[(tg["season"] == season) & (tg["season_type"] == "Regular Season")]
    return [date(1970, 1, 1) + timedelta(days=int(x)) for x in sorted(rs["date"].unique())]


def extract_path(season: str) -> Path:
    return cache_dir() / f"corr_extract_{season}.pkl"


def extract_season(conn: sqlite3.Connection, season: str, *, every: int = 1, n_sims: int = 500,
                   protocol: str = "avail", ctx=None, bundle=None, log=print,
                   checkpoint: int = 20, limit: int | None = None) -> pd.DataFrame:
    """project_historical over the regular-season dates of `season` (every `every`-th date),
    keeping only what the residual study needs: per projected player the exact mean given played
    (`mu_played`), projected minutes, the copula tier and P(play); per team-game the implied
    team points and expected margin / total. Resumable: the pickle at `extract_path(season)`
    holds the rows and the dates already done. n_sims is small on purpose (only means are used).
    The bundle (fit on seasons < S) is loaded or fitted by the pipeline."""
    from . import pipeline as P                    # lazy: pipeline imports copula
    t0 = time.time()
    path = extract_path(season)
    rows, done = [], set()
    if path.exists():
        saved = pd.read_pickle(path)
        rows = [saved["rows"]] if len(saved["rows"]) else []
        done = set(saved["done"])
    ctx = ctx or P.Context(conn, season, log=log)
    b = bundle or P.load_bundle(conn, season, ctx, log=log)
    days = _regular_days(ctx, season)[::every]
    if limit:
        days = days[:limit]
    todo = [d for d in days if d.isoformat() not in done]
    log(f"{season}: {len(days)} dates ({len(todo)} to do), n_sims {n_sims}, "
        f"setup {time.time() - t0:.0f}s")

    def flush():
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        pd.to_pickle({"rows": df, "done": sorted(done), "season": season, "every": every,
                      "n_sims": n_sims, "protocol": protocol, "version": MODEL_VERSION}, tmp)
        tmp.replace(path)

    for k, day in enumerate(todo):
        res = P.project_historical(conn, day, protocol=protocol, n_sims=n_sims, ctx=ctx,
                                   bundle=b)
        pl = res.players
        if len(pl):
            d = pl[["game_id", "team", "player_id", "p_play", "tier", "mu_played",
                    "min_played"]].copy()
            d["tier"] = d["tier"].map({t: i for i, t in enumerate(TIERS)}).astype(int)
            d = d.rename(columns={"mu_played": "mu", "min_played": "min_proj"})
            tm = pd.DataFrame([{"game_id": t["game_id"], "team": t["team"],
                                "target": t["target"]} for t in res.teams])
            d = d.merge(tm, on=["game_id", "team"], how="left")
            env = pd.DataFrame([{"game_id": g, "mu_margin": v["mu_margin"],
                                 "mu_total": v["mu_total"]} for g, v in res.meta["env"].items()])
            d = d.merge(env, on="game_id", how="left")
            d["date"] = day.isoformat()
            rows.append(d[d["p_play"] > 0])
        done.add(day.isoformat())
        if (k + 1) % checkpoint == 0 or k == len(todo) - 1:
            flush()
            log(f"  {season} {day} ({k + 1}/{len(todo)}) {time.time() - t0:.0f}s")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================================== residual frame

def _positions(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_id, season_start, pos 0..3): the season's roster position, else the current
    snapshot, else 3 (unknown). Static attributes, not performance."""
    tr = pd.read_sql_query("SELECT player_id, season, position FROM team_rosters", conn)
    pi = pd.read_sql_query("SELECT player_id, position FROM player_index", conn)

    def code(p):
        s = (p if isinstance(p, str) else "").split("-")[0].strip().upper()[:1]
        return {"G": 0, "F": 1, "C": 2}.get(s, 3)
    tr["sidx"] = tr["season"].str[:4].astype(int)
    tr["pos_s"] = [code(p) for p in tr["position"]]
    tr = tr[tr["pos_s"] < 3].drop_duplicates(["player_id", "sidx"])[["player_id", "sidx", "pos_s"]]
    pi["pos_p"] = [code(p) for p in pi["position"]]
    pi = pi[pi["pos_p"] < 3].drop_duplicates("player_id")[["player_id", "pos_p"]]
    return tr, pi


def load_frame(conn: sqlite3.Connection, seasons: list[str]) -> pd.DataFrame:
    """Cached extracts for `seasons` joined to the actuals: one row per projected player-game
    in which the player played. Columns: game_id, date, season, team, opp, player_id, tier
    (projection tier), pos, mu, min_proj, p_play, target, dk, min, team_pts, opp_pts, margin
    (team perspective), g (= team_pts / target)."""
    ex = []
    for s in seasons:
        p = extract_path(s)
        if not p.exists():
            raise FileNotFoundError(f"no extract for {s}: run extract_season ({p})")
        ex.append(pd.read_pickle(p)["rows"].assign(season=s))
    ex = pd.concat(ex, ignore_index=True)
    q = ",".join("?" * len(seasons))
    pg = pd.read_sql_query(
        f"""SELECT game_id, player_id, team, min, pts, fg3m, reb, ast, stl, blk, tov
            FROM player_games WHERE season_type='Regular Season' AND season IN ({q}) AND min>0""",
        conn, params=seasons)
    pg["dk"] = dk_points(pg)
    tg = pd.read_sql_query(
        f"""SELECT t.game_id, t.team, t.opp, t.pts AS team_pts, o.pts AS opp_pts
            FROM team_games t JOIN team_games o ON o.game_id=t.game_id AND o.team=t.opp
            WHERE t.season_type='Regular Season' AND t.season IN ({q})""", conn, params=seasons)
    d = ex.merge(pg[["game_id", "player_id", "min", "dk"]], on=["game_id", "player_id"])
    d = d.merge(tg, on=["game_id", "team"], how="inner")
    d["margin"] = d["team_pts"] - d["opp_pts"]
    d["g"] = d["team_pts"] / d["target"].clip(lower=50.0)
    tr, pi = _positions(conn)
    d["sidx"] = d["season"].str[:4].astype(int)
    d = d.merge(tr, on=["player_id", "sidx"], how="left").merge(pi, on="player_id", how="left")
    d["pos"] = d["pos_s"].fillna(d["pos_p"]).fillna(3).astype(int)
    d = d.drop(columns=["pos_s", "pos_p", "sidx"])
    d = d[(d["mu"] > MIN_PROJ_MU) & d["target"].notna()].reset_index(drop=True)
    d["grp"] = d["tier"] * NPOS + d["pos"]
    return d


def fit_elasticity(f: pd.DataFrame) -> list[float]:
    """b_tier: slope of the relative residual (dk - mu) / mu on (g - 1), per tier."""
    out = []
    for t in range(len(TIERS)):
        s = f[f["tier"] == t]
        if len(s) < 200:
            out.append(0.0)
            continue
        x = s["g"].to_numpy() - 1.0
        y = (s["dk"] - s["mu"]).to_numpy() / s["mu"].to_numpy()
        x = x - x.mean()
        out.append(float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-9)))
    return out


def residuals(f: pd.DataFrame, mode: str, b: list | None = None) -> tuple[pd.DataFrame, list]:
    """Add `ref` (the conditional mean) and `res` = dk - ref for a residual mode. b (tier
    elasticities) is fit on `f` when not given."""
    if mode not in MODES:
        raise ValueError(mode)
    b = list(b) if b is not None else (fit_elasticity(f) if mode != "U" else [0.0] * 3)
    ref = f["mu"].to_numpy(float).copy()
    if mode in ("P", "PM"):
        bt = np.asarray(b)[f["tier"].to_numpy(int)]
        ref = ref * np.clip(1.0 + bt * (f["g"].to_numpy(float) - 1.0), 0.2, None)
    if mode == "PM":
        ratio = np.clip(f["min"].to_numpy(float) / np.maximum(f["min_proj"].to_numpy(float), 1.0),
                        0.05, 4.0)
        ref = ref * ratio
    out = f.copy()
    out["ref"] = ref
    out["res"] = out["dk"] - ref
    return out, b


def standardise(f: pd.DataFrame, cvinfo: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Centre the residual by (season, tier) and divide by cv(ref) x ref. The CV-by-level table
    is fit on `f`'s centred residual unless `cvinfo` (from a training fold) is given. Adds `rc`
    (centred) and `z`."""
    f = f.copy()
    bias = f.groupby(["season", "tier"])["res"].transform("mean")
    f["rc"] = f["res"] - bias
    if cvinfo is None:
        cv_mu, cv = fit_cv(f["ref"].to_numpy(), f["rc"].to_numpy())
        cvinfo = {"cv_mu": cv_mu, "cv": cv}
    f["z"] = (f["rc"] / (cv_at(cvinfo["cv_mu"], cvinfo["cv"], f["ref"]) * f["ref"])).clip(-6, 6)
    bias_t = {TIERS[t]: float(f.loc[f["tier"] == t, "res"].mean()) for t in range(3)}
    return f, {"cv_mu": cvinfo["cv_mu"], "cv": cvinfo["cv"], "bias": bias_t}


# ============================================================================== pair statistics

_PAIRS = [(a, b) for a in range(NG) for b in range(a, NG)]
NP = len(_PAIRS)                                     # 78
_IA = np.array([p[0] for p in _PAIRS])
_IB = np.array([p[1] for p in _PAIRS])
_DIAG = _IA == _IB


def _known(g: int) -> bool:
    return g % NPOS != NPOS - 1


def _cat_defs() -> dict:
    """level -> (labels, M (ncat, NP) 0/1 membership of unordered group pairs)."""
    out = {}
    out["all"] = (["all"], np.ones((1, NP)))
    tl = [pair_label(TIERS[i], TIERS[j]) for i in range(3) for j in range(i, 3)]
    M = np.zeros((len(tl), NP))
    for p, (a, b) in enumerate(_PAIRS):
        ta, tb = sorted((a // NPOS, b // NPOS))
        M[tl.index(pair_label(TIERS[ta], TIERS[tb])), p] = 1
    out["tier"] = (tl, M)
    pl = [f"{POS[i]}|{POS[j]}" for i in range(3) for j in range(i, 3)]
    M = np.zeros((len(pl), NP))
    for p, (a, b) in enumerate(_PAIRS):
        if _known(a) and _known(b):
            pa, pb = sorted((a % NPOS, b % NPOS))
            M[pl.index(f"{POS[pa]}|{POS[pb]}"), p] = 1
    out["pos"] = (pl, M)
    cl = [f"{group_label(a)}|{group_label(b)}" for a, b in _PAIRS if _known(a) and _known(b)]
    M = np.zeros((len(cl), NP))
    k = 0
    for p, (a, b) in enumerate(_PAIRS):
        if _known(a) and _known(b):
            M[k, p] = 1
            k += 1
    out["cell"] = (cl, M)
    return out


CATS = _cat_defs()


class PairStats:
    """Sufficient statistics for pooled teammate / opponent correlations by group pair.

    Per team-game and group: S = sum x, Q = sum x^2, N = count. Teammate pair sums are
    S_a S_b (a != b) and (S_a^2 - Q_a) / 2 (a = b); opponent sums S^home_a S^away_b. All are
    linear per game, so a game-cluster bootstrap is a weighted sum of per-game feature rows.
    correlation(cat) = sum of pair products / sum over pairs of sqrt(v_a v_b), v_a = E[x^2]
    of group a (x is assumed centred)."""

    def __init__(self, game, team, grp, x):
        game = np.asarray(game)
        team = np.asarray(team)
        grp = np.asarray(grp, int)
        x = np.asarray(x, float)
        gi, _ = pd.factorize(game)
        ti, _ = pd.factorize(pd.Series(game).astype(str) + "|" + pd.Series(team).astype(str))
        T, G = ti.max() + 1, gi.max() + 1
        S = np.bincount(ti * NG + grp, x, T * NG).reshape(T, NG)
        Q = np.bincount(ti * NG + grp, x * x, T * NG).reshape(T, NG)
        N = np.bincount(ti * NG + grp, None, T * NG).reshape(T, NG)
        first = np.zeros(T, int)
        first[ti] = gi
        side = pd.Series(first).groupby(first).cumcount().to_numpy()
        self.n_games, self.n_team_games = G, T
        # teammates
        num = (S[:, :, None] * S[:, None, :])[:, _IA, _IB]
        num[:, _DIAG] = ((S ** 2 - Q) / 2.0)[:, _IA[_DIAG]]
        cnt = (N[:, :, None] * N[:, None, :])[:, _IA, _IB]
        cnt[:, _DIAG] = (N * (N - 1) / 2.0)[:, _IA[_DIAG]]
        tm_num = np.zeros((G, NP))
        tm_cnt = np.zeros((G, NP))
        np.add.at(tm_num, first, num)
        np.add.at(tm_cnt, first, cnt)
        # opponents (games with both teams)
        t0 = np.full(G, -1)
        t1 = np.full(G, -1)
        t0[first[side == 0]] = np.flatnonzero(side == 0)
        t1[first[side == 1]] = np.flatnonzero(side == 1)
        both = (t0 >= 0) & (t1 >= 0)
        op_num = np.zeros((G, NP))
        op_cnt = np.zeros((G, NP))
        if both.any():
            S0, S1, N0, N1 = S[t0[both]], S[t1[both]], N[t0[both]], N[t1[both]]
            o = S0[:, :, None] * S1[:, None, :]
            c = N0[:, :, None] * N1[:, None, :]
            sym = o + o.transpose(0, 2, 1)
            csym = c + c.transpose(0, 2, 1)
            n_ = sym[:, _IA, _IB]
            n_[:, _DIAG] = o[:, _IA[_DIAG], _IA[_DIAG]]
            c_ = csym[:, _IA, _IB]
            c_[:, _DIAG] = c[:, _IA[_DIAG], _IA[_DIAG]]
            op_num[both], op_cnt[both] = n_, c_
        gq = np.zeros((G, NG))
        gn = np.zeros((G, NG))
        np.add.at(gq, first, Q)
        np.add.at(gn, first, N)
        self.F = np.hstack([tm_num, tm_cnt, op_num, op_cnt, gq, gn])

    def totals(self, w: np.ndarray | None = None) -> np.ndarray:
        return self.F.sum(0) if w is None else w @ self.F

    def estimate(self, kind: str, level: str, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(r, n_pairs) per category for totals T of shape (F,) or (B, F)."""
        T = np.atleast_2d(T)
        off = 0 if kind == "team" else 2 * NP
        num, cnt = T[:, off:off + NP], T[:, off + NP:off + 2 * NP]
        Qs, Ns = T[:, 4 * NP:4 * NP + NG], T[:, 4 * NP + NG:]
        sd = np.sqrt(Qs / np.maximum(Ns, 1.0))
        den = cnt * sd[:, _IA] * sd[:, _IB]
        M = CATS[level][1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (num @ M.T) / (den @ M.T)
        return np.nan_to_num(r), cnt @ M.T

    def table(self, kind: str, level: str, nboot: int = 0, seed: int = 0,
              min_n: int = 1) -> pd.DataFrame:
        """DataFrame (index = category): r, n (pairs), se/lo/hi from a game-cluster bootstrap."""
        r, n = self.estimate(kind, level, self.totals())
        out = pd.DataFrame({"r": r[0], "n": n[0]}, index=CATS[level][0])
        if nboot:
            w = np.random.default_rng(seed).multinomial(
                self.n_games, np.ones(self.n_games) / self.n_games, size=nboot).astype(float)
            rb, _ = self.estimate(kind, level, self.totals(w))
            out["se"] = rb.std(0)
            out["lo"] = np.percentile(rb, 2.5, axis=0)
            out["hi"] = np.percentile(rb, 97.5, axis=0)
        return out[out["n"] >= min_n]


# ============================================================================== params

@dataclass
class CorrTable:
    """One residual mode. Teammate / opponent correlations of the CV-standardised residual."""
    mode: str
    elasticity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])    # b per tier (P, PM)
    cv_mu: list = field(default_factory=list)
    cv: list = field(default_factory=list)
    bias: dict = field(default_factory=dict)       # mean raw residual by tier
    rho_team_all: float = 0.0
    rho_team: dict = field(default_factory=dict)       # tier pair "star|role"
    rho_team_cell: dict = field(default_factory=dict)  # "star:G|role:C"
    rho_opp_all: float = 0.0
    rho_opp: dict = field(default_factory=dict)
    rho_opp_cell: dict = field(default_factory=dict)
    n: dict = field(default_factory=dict)              # pair counts, "team:star|role" etc.


@dataclass
class CorrParams:
    """JSON-serialisable output of `fit`; persisted as var/proj/corr_<season>.json."""
    season: str
    train_seasons: list = field(default_factory=list)
    k_shrink: float = K_SHRINK                         # tier pair -> all
    k_cell: float = K_SHRINK                           # tier x position cell -> tier pair
    default_mode: str = DEFAULT_MODE
    tables: dict = field(default_factory=dict)         # mode -> CorrTable (as dict in JSON)
    diagnostics: dict = field(default_factory=dict)
    version: str = MODEL_VERSION

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "CorrParams":
        d = json.loads(s)
        d["tables"] = {m: CorrTable(**t) for m, t in d["tables"].items()}
        return cls(**d)

    def table(self, mode: str | None = None) -> CorrTable:
        return self.tables[mode or self.default_mode]

    def corr_matrix(self, team, tiers, pos=None, *, mode: str | None = None,
                    opp_cells: bool = False) -> np.ndarray:
        """Correlation matrix for a game's players. team: team code per player; tiers: 0..2;
        pos: 0..3 (G, F, C, unknown) or None for tier-pair-only. Teammates use the tier x
        position cell when positions are known, else the tier pair; opponents the tier pair
        (or cell with opp_cells). Projected to the nearest PSD correlation matrix."""
        t = self.table(mode)
        team = np.asarray(team)
        tiers = np.asarray(tiers, int)
        n = len(team)
        pos = None if pos is None else np.asarray(pos, int)
        R = np.empty((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                same = team[i] == team[j]
                tp = pair_label(TIERS[tiers[i]], TIERS[tiers[j]])
                tab, cells, allv = ((t.rho_team, t.rho_team_cell, t.rho_team_all) if same
                                    else (t.rho_opp, t.rho_opp_cell, t.rho_opp_all))
                r = tab.get(tp, allv)
                if pos is not None and pos[i] < NPOS - 1 and pos[j] < NPOS - 1 and (
                        same or opp_cells):
                    key = pair_label(f"{TIERS[tiers[i]]}:{POS[pos[i]]}",
                                     f"{TIERS[tiers[j]]}:{POS[pos[j]]}")
                    r = cells.get(key, r)
                R[i, j] = R[j, i] = r
        np.fill_diagonal(R, 1.0)
        return nearest_corr(R)


def _table_from_stats(ps: PairStats, k: float, k_cell: float, mode: str, elasticity,
                      cvinfo) -> CorrTable:
    t = CorrTable(mode=mode, elasticity=[float(x) for x in elasticity],
                  cv_mu=cvinfo["cv_mu"], cv=cvinfo["cv"], bias=cvinfo["bias"])
    for kind in ("team", "opp"):
        ra, na = ps.estimate(kind, "all", ps.totals())
        r_all, n_all = float(ra[0, 0]), float(na[0, 0])
        rt, nt = ps.estimate(kind, "tier", ps.totals())
        rc, nc = ps.estimate(kind, "cell", ps.totals())
        tier_labels, cell_labels = CATS["tier"][0], CATS["cell"][0]
        tier_r = shrink(rt[0], nt[0], r_all, k)
        tier_map = dict(zip(tier_labels, [float(x) for x in tier_r]))
        cell_r = {}
        for lab, r, n in zip(cell_labels, rc[0], nc[0]):
            a, b = lab.split("|")
            parent = tier_map[pair_label(a.split(":")[0], b.split(":")[0])]
            cell_r[lab] = float(shrink(r, n, parent, k_cell))
        n_map = {f"{kind}:all": n_all}
        n_map.update({f"{kind}:{lab}": float(n) for lab, n in zip(tier_labels, nt[0])})
        n_map.update({f"{kind}:{lab}": float(n) for lab, n in zip(cell_labels, nc[0])})
        if kind == "team":
            t.rho_team_all, t.rho_team, t.rho_team_cell = r_all, tier_map, cell_r
        else:
            t.rho_opp_all, t.rho_opp, t.rho_opp_cell = r_all, tier_map, cell_r
        t.n.update(n_map)
    return t


def fit(data, seasons: list[str], *, season: str | None = None, k_shrink: float = K_SHRINK,
        k_cell: float | None = None, modes: tuple = MODES,
        default_mode: str = DEFAULT_MODE) -> CorrParams:
    """Fit CorrParams on the residual frame of `seasons` (training seasons, all < the target
    `season`, default the one after the last). `data` is a sqlite connection (cached extracts
    via `load_frame`) or a frame from `load_frame`/the same columns."""
    seasons = sorted(seasons)
    f = load_frame(data, seasons) if isinstance(data, sqlite3.Connection) else data
    f = f[f["season"].isin(seasons)]
    k_cell = k_shrink if k_cell is None else k_cell
    params = CorrParams(season=season or next_season(seasons[-1]), train_seasons=seasons,
                        k_shrink=k_shrink, k_cell=k_cell, default_mode=default_mode)
    for mode in modes:
        r, b = residuals(f, mode)
        z, cvinfo = standardise(r)
        ps = PairStats(z["game_id"], z["team"], z["grp"], z["z"])
        params.tables[mode] = _table_from_stats(ps, k_shrink, k_cell, mode, b, cvinfo)
    params.diagnostics = {"n_rows": int(len(f)), "n_games": int(f["game_id"].nunique()),
                          "rows_by_season": {s: int((f["season"] == s).sum()) for s in seasons}}
    return params


def params_path(season: str) -> Path:
    return cache_dir() / f"corr_{season}.json"


def save(p: CorrParams, path: Path | None = None) -> Path:
    path = path or params_path(p.season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(p.to_json())
    return path


def load(season: str, path: Path | None = None) -> CorrParams:
    return CorrParams.from_json((path or params_path(season)).read_text())
