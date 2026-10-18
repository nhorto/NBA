"""Rung 2: player-based team strength models (Model interface of model/backtest.py).

Every model here reads pre-game team strengths from one shared `EngineCache` (players.py), so
the expensive sequential pass runs once per distinct input. The pass is causal, so the result
for a prefix of the games equals the matching prefix of a pass over more games; the cache
exploits this by slicing a primed full-history run whenever the requested games are an exact
prefix of it (same ids and same content hash per row), and recomputes otherwise (e.g. when
check_no_lookahead scrambles future results). A test asserts the prefix property.

Heads (refit on each training window, first `burn_in` seasons dropped):
  logit   L2 logistic regression, no free intercept (antisymmetric design, like elo_sched)
  margin  OLS margin on the same design; P(home) = Phi(mu / sigma), sigma = training RMSE
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from . import baselines, elo as elo_mod, players as pl

_KEY_COLS = ("game_id", "game_date", "home", "away", "home_raw", "away_raw", "is_neutral",
             "season_type", "ot", "margin")


def _row_hashes(games: pd.DataFrame) -> np.ndarray:
    cols = [c for c in _KEY_COLS if c in games.columns]
    return pd.util.hash_pandas_object(games[cols], index=False).to_numpy()


class EngineCache:
    """Runs `players.run_player_engine` lazily; `players` may be a DataFrame or a loader."""

    def __init__(self, players: pd.DataFrame | Callable[[], pd.DataFrame],
                 params: pl.PlayerParams = pl.TUNED, modes=pl.ALL_MODES):
        self._players = players
        self.params = params
        self.modes = tuple(modes)
        self._runs: list[tuple[np.ndarray, pd.DataFrame]] = []
        self.n_runs = 0

    @property
    def players(self) -> pd.DataFrame:
        if callable(self._players):
            self._players = self._players()
        return self._players

    def prime(self, games: pd.DataFrame) -> None:
        self.get(games)

    def get(self, games: pd.DataFrame) -> pd.DataFrame:
        h = _row_hashes(games)
        for hc, res in self._runs:
            if len(h) <= len(hc) and np.array_equal(h, hc[:len(h)]):
                out = res.iloc[:len(h)].copy()
                out.index = games.index
                return out
        res = pl.run_player_engine(games, self.players, self.params, self.modes)
        self.n_runs += 1
        self._runs.append((h, res.reset_index(drop=True)))
        self._runs = self._runs[-3:]
        out = res.copy()
        out.index = games.index
        return out


FEATURE_SETS = {
    "ps": ("ps_diff", "home_court", "home_court_playoff"),
    "ps_elo": ("ps_diff", "elo_diff_logit", "home_court", "home_court_playoff"),
    "ps_elo_sched": ("ps_diff", "elo_diff_logit", "home_court", "home_court_playoff",
                     *baselines.SCHED_FEATURES),
    # market variants need frame["mkt_margin"] (market.add_market_columns); rows without a
    # line fall back to the same design minus the market column
    "mkt": ("mkt_margin", "home_court"),
    "elo_sched_mkt": ("elo_diff_logit", "home_court", "home_court_playoff",
                      *baselines.SCHED_FEATURES, "mkt_margin"),
    "ps_elo_sched_mkt": ("ps_diff", "elo_diff_logit", "home_court", "home_court_playoff",
                         *baselines.SCHED_FEATURES, "mkt_margin"),
}


def _burn_in(train_df: pd.DataFrame, n: int) -> np.ndarray:
    seasons = sorted(train_df["season"].unique())
    k = min(n, len(seasons) - 1)
    keep = set(seasons[k:])
    return train_df["season"].isin(keep).to_numpy()


class PlayerStrengthModel:
    """Team strength = sum(rating * expected minutes / 48) home minus away, plus optional Elo
    and schedule terms, through a logistic or Gaussian-margin head."""

    def __init__(self, cache: EngineCache, name: str, features: str = "ps_elo_sched",
                 mode: str = "played", head: str = "logit", C: float = 1.0, burn_in: int = 3):
        if head not in ("logit", "margin"):
            raise ValueError(head)
        self.cache, self.name, self.features = cache, name, features
        self.mode, self.head, self.C, self.burn_in = mode, head, C, burn_in
        self.cols = FEATURE_SETS[features]
        self._elo = baselines._EloRunner(elo_mod.TUNED) if "elo_diff_logit" in self.cols else None
        self._uses_ps = "ps_diff" in self.cols

    def _design(self, df: pd.DataFrame, eng: pd.DataFrame | None, e: pd.DataFrame | None
                ) -> pd.DataFrame:
        if e is not None:
            X = baselines.elo_sched_design(df, (e["elo_home_pre"] - e["elo_away_pre"]).to_numpy())
        else:
            hc = 1.0 - df["is_neutral"].to_numpy(float)
            po = (df["season_type"].to_numpy() != "Regular Season").astype(float)
            X = pd.DataFrame({"home_court": hc, "home_court_playoff": hc * po}, index=df.index)
        if eng is not None:
            X["ps_diff"] = (eng[f"{self.mode}_home"] - eng[f"{self.mode}_away"]).to_numpy()
        if "mkt_margin" in self.cols:
            X["mkt_margin"] = df["mkt_margin"].to_numpy(float)
        return X

    def _fit_head(self, X: np.ndarray, y: np.ndarray, mg: np.ndarray):
        reg = LinearRegression(fit_intercept=False).fit(X, mg)
        resid = mg - reg.predict(X)
        sigma = float(np.sqrt(np.sum(resid ** 2) / max(len(mg) - X.shape[1], 1)))
        clf = None
        if self.head == "logit":
            clf = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000).fit(X, y)
        return reg, sigma, clf

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_ = train_df
        eng = self.cache.get(train_df) if self._uses_ps else None
        e = self._elo.run(train_df, None) if self._elo else None
        X = self._design(train_df, eng, e)
        m = _burn_in(train_df, self.burn_in)
        y = train_df["home_win"].to_numpy()
        mg = train_df["margin"].to_numpy(float)
        main = list(self.cols)
        ok = m & np.isfinite(X[main].to_numpy()).all(axis=1)
        self.reg_, self.sigma_, self.clf_ = self._fit_head(X[main].to_numpy()[ok], y[ok], mg[ok])
        self.n_fit_ = int(ok.sum())
        self.fallback_ = None
        if "mkt_margin" in main:
            fb = [c for c in main if c != "mkt_margin"]
            self.fallback_ = (fb, *self._fit_head(X[fb].to_numpy()[m], y[m], mg[m]))

    def _Xdf(self, test_df):
        eng = None
        if self._uses_ps:
            eng = self.cache.get(pd.concat([self.train_, test_df])).iloc[len(self.train_):]
        e = self._elo.run(self.train_, test_df).iloc[len(self.train_):] if self._elo else None
        return self._design(test_df, eng, e)

    def _predict(self, test_df, what: str) -> np.ndarray:
        X = self._Xdf(test_df)
        heads = [(list(self.cols), self.reg_, self.sigma_, self.clf_)]
        if self.fallback_ is not None:
            heads.append(self.fallback_)
        out = np.full(len(test_df), np.nan)
        for cols, reg, sigma, clf in heads:
            A = X[cols].to_numpy()
            todo = np.isnan(out) & np.isfinite(A).all(axis=1)
            if not todo.any():
                continue
            if what == "margin":
                out[todo] = reg.predict(A[todo])
            elif clf is not None:
                out[todo] = clf.predict_proba(A[todo])[:, 1]
            else:
                out[todo] = norm.cdf(reg.predict(A[todo]) / sigma)
        return out

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._predict(test_df, "proba")

    def predict_margin(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._predict(test_df, "margin")

    def describe(self) -> dict:
        d = {"features": self.features, "mode": self.mode, "head": self.head,
             "sigma": self.sigma_, "n_fit": self.n_fit_, "params": self.cache.params.to_dict(),
             "margin_coef": dict(zip(self.cols, map(float, self.reg_.coef_)))}
        if self.clf_ is not None:
            d["logit_coef"] = dict(zip(self.cols, map(float, self.clf_.coef_[0])))
        return d


# name -> (features, mode, head)
SPECS = {
    "ps": ("ps", "played", "logit"),
    "ps_mh": ("ps", "played", "margin"),
    "ps_roster": ("ps", "roster", "logit"),
    "ps_elo": ("ps_elo", "played", "logit"),
    "ps_elo_sched": ("ps_elo_sched", "played", "logit"),
    "ps_elo_sched_mh": ("ps_elo_sched", "played", "margin"),
    "ps_elo_sched_roster": ("ps_elo_sched", "roster", "logit"),
    "ps_elo_sched_soft": ("ps_elo_sched", "soft", "logit"),
    "ps_elo_sched_flip": ("ps_elo_sched", "flip", "logit"),
    "ps_elo_sched_nofloor": ("ps_elo_sched", "played_nofloor", "logit"),
    "ps_elo_sched_box": ("ps_elo_sched", "box", "logit"),
    "ps_box": ("ps", "box", "logit"),
}
# Need frame["mkt_margin"]; registered by make_registry(..., market=True).
MARKET_SPECS = {
    "mkt": ("mkt", "played", "logit"),
    "elo_sched_mkt": ("elo_sched_mkt", "played", "logit"),
    "ps_elo_sched_mkt": ("ps_elo_sched_mkt", "played", "logit"),
}


def make_registry(players: pd.DataFrame | Callable[[], pd.DataFrame],
                  params: pl.PlayerParams = pl.TUNED, burn_in: int = 3, market: bool = False,
                  conn=None) -> tuple[dict[str, Callable], EngineCache]:
    """Rung-2 models; with `conn`, also the rung-3 GBMs (gbm.SPECS: gbm.PRODUCTION_WINPROB,
    gbm.PRODUCTION_OPINION), which read box scores and closing lines from `conn` lazily."""
    cache = EngineCache(players, params)
    reg = {}
    specs = {**SPECS, **(MARKET_SPECS if market else {})}
    for name, (fs, mode, head) in specs.items():
        reg[name] = (lambda n=name, f=fs, m=mode, h=head:
                     PlayerStrengthModel(cache, n, f, m, h, burn_in=burn_in))
    if conn is not None:
        from . import features as ft, gbm
        greg, _ = gbm.make_registry(players, lambda: ft.load_team_box(conn),
                                    lambda: ft.load_closing_full(conn), params)
        reg.update(greg)
    return reg, cache


# --- tuning (2000-01..2010-11 only) ---------------------------------------------------------

TUNE_SEASONS = ("2000-01", "2010-11")
TUNE_EVAL_FROM = "2003-04"


def tune_score(games: pd.DataFrame, players: pd.DataFrame, params: pl.PlayerParams) -> dict:
    """In-sample 3-parameter logistic (strength diff, home court, playoff home court) and
    margin fit on TUNE_EVAL_FROM..2010-11, using only games inside the tuning window."""
    from . import metrics
    lo, hi = TUNE_SEASONS
    w = games[(games["season"] >= lo) & (games["season"] <= hi)]
    eng = pl.run_player_engine(w, players, params, modes=("played",))
    ev = (w["season"] >= TUNE_EVAL_FROM).to_numpy()
    sd = (eng["played_home"] - eng["played_away"]).to_numpy()[ev]
    hc = 1.0 - w["is_neutral"].to_numpy(float)[ev]
    po = (w["season_type"].to_numpy()[ev] != "Regular Season").astype(float)
    X = np.column_stack([sd, hc, hc * po])
    y = w["home_win"].to_numpy()[ev]
    clf = LogisticRegression(C=1e4, fit_intercept=False, max_iter=2000).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    mg = w["margin"].to_numpy(float)[ev]
    reg = LinearRegression(fit_intercept=False).fit(X, mg)
    return {"log_loss": metrics.log_loss(y, p), "margin_rmse": metrics.rmse(mg, reg.predict(X)),
            "slope": float(reg.coef_[0])}


_TUNE_DATA: dict = {}


def _tune_init(db_path: str):
    import sqlite3
    from . import backtest
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    fr = backtest.build_frame(conn)
    fr = fr[fr["season"] <= TUNE_SEASONS[1]]
    p = pl.load_player_games(conn)
    _TUNE_DATA["games"] = fr
    _TUNE_DATA["players"] = p[p["game_id"].isin(set(fr["game_id"]))]


def _tune_one(prm):
    return tune_score(_TUNE_DATA["games"], _TUNE_DATA["players"], prm), prm


def tune_grid_stages(base: pl.PlayerParams):
    """Stage grids, each searched around the previous stage's winner."""
    from dataclasses import replace
    yield "apm", [replace(base, lam=l, apm_half_life=h, apm_offseason=o)
                  for l in (10.0, 30.0, 100.0, 300.0) for h in (180.0, 365.0, 730.0)
                  for o in (0.5, 0.8)]
    yield "box", lambda b: [replace(b, box_half_life=h, box_offseason=o, box_n0=n)
                            for h in (180.0, 365.0, 730.0) for o in (0.6, 0.9)
                            for n in (150.0, 600.0)]
    yield "minutes", lambda b: [replace(b, min_alpha_short=a, min_alpha_long=c, floor=f)
                                for a in (0.2, 0.35, 0.5) for c in (0.04, 0.1)
                                for f in (0.0, 6.0, 10.0)]
    # stage-1/3 winners sat on grid edges: extend past them
    yield "apm_refine", lambda b: [replace(b, lam=l, apm_half_life=h, apm_offseason=o)
                                   for l in (300.0, 600.0, 1200.0, 2500.0)
                                   for h in (730.0, 1460.0) for o in (0.8, 1.0)]
    yield "minutes_refine", lambda b: [replace(b, min_alpha_short=a, min_alpha_long=c,
                                               min_w_short=w)
                                       for a in (0.1, 0.2) for c in (0.02, 0.04)
                                       for w in (0.3, 0.5, 0.7)]


def tune_all(db_path: str, workers: int = 4, log=print) -> dict:
    """Staged search on 2000-01..2010-11 (scored 2003-04..2010-11, in-sample 3-param logit).
    Each config is one engine pass over 11 seasons."""
    from multiprocessing import get_context
    best = pl.PlayerParams()
    out = {}
    with get_context("spawn").Pool(workers, initializer=_tune_init, initargs=(db_path,)) as pool:
        for stage, grid in tune_grid_stages(best):
            grid = grid if isinstance(grid, list) else grid(best)
            res = sorted(pool.map(_tune_one, grid), key=lambda r: r[0]["log_loss"])
            out[stage] = [(r, p.to_dict()) for r, p in res]
            for r, p in res[:5]:
                log(f"  {stage} {r['log_loss']:.5f} rmse={r['margin_rmse']:.3f} "
                    f"slope={r['slope']:.3f} {_diff(p)}")
            if res[0][0]["log_loss"] < tune_score_cached(out, best):
                best = res[0][1]
            out[stage + "_best"] = best.to_dict()
    out["best"] = best.to_dict()
    return out


def _diff(p: pl.PlayerParams) -> str:
    d0 = pl.PlayerParams().to_dict()
    return " ".join(f"{k}={v:g}" for k, v in p.to_dict().items() if d0[k] != v) or "(defaults)"


def tune_score_cached(out: dict, prm: pl.PlayerParams) -> float:
    for k, rows in out.items():
        if isinstance(rows, list):
            for r, p in rows:
                if p == prm.to_dict():
                    return r["log_loss"]
    return float("inf")
