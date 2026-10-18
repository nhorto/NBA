"""Rung 3: stacked LightGBM margin/total models with a calibrated win probability.

Structure of one target head (margin or total), refit on every training window:

1. **Linear base** (weighted least squares). Margin: player-strength diff, Elo diff, home court
   (plus its playoff, league-drift and APM-HCA interactions) and the schedule terms, no
   intercept (antisymmetric, as in rung 2). Total: intercept, league scoring environment,
   the two teams' recent game totals and a pace x ratings projection. The market variant adds
   the closing spread/total; rows without a line use a second base fit without it.
2. **LightGBM on the residual**: the base prediction is the `init_score`, so the trees only
   learn what the linear stack misses (interactions, non-linearity, form, absences, ...).
   Zero trees (= the linear base) is a valid outcome of the inner validation.
3. **Inner time-ordered validation**: the last `n_inner` training seasons are each predicted by
   a fit on the seasons before them. The (config, number of trees) with the lowest pooled
   validation MSE is chosen (early stopping on time-ordered folds); the final model is refit on
   the whole window with that config.
4. **Calibration** of P(home win) from the predicted margin, fit on the inner out-of-fold
   predictions: Gaussian (Phi(mu/sigma)), Platt (logistic on mu and mu x postseason) or
   isotonic, picked by leave-one-inner-season-out log-loss.

**Availability**: player strength uses the engine's `avail_tight` mode. The backtest oracle
applies only to rotation regulars (projection >= players.REGULAR_MIN), and fringe players
count as available when they're on the recent roster. The postseason rotation is tighter.
The full "played" oracle leaks game flow (blowouts empty both benches), and a GBM turns that
into a fake 56-58% ATS; see MODEL.md, rung 3.

**Drift**: every fit is recency-weighted (season half-life `half_life`, part of the inner
search) and sees two online home-court estimates: the league's mean home margin over the last
1,230 non-neutral games before the day (`lg_hca`) and the player engine's decayed APM
home-court term (`apm_hca`).

Interface (backtest Model protocol plus production extras):

    m = factory();  m.fit(train_df)
    m.predict_proba(df), m.predict_margin(df), m.predict_total(df)
    m.explain(row, df) -> [{"factor", "team", "impact_pts", "home_margin_pts"}, ...]

`df` follows the backtest contract (rows in chronological order after train_df; online
inputs may use results of earlier days only). Production may add `mkt_margin`/`mkt_total`
(the live line) and `lineup_home`/`lineup_away` (injury-adjusted expected-lineup strength)
columns; see features.build_features.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import elo as elo_mod, features as ft, metrics, player_model, players as pl

ENGINE_MODES = ("played", "roster", "avail", "avail_tight", "roster_tight")

PRODUCTION_WINPROB = "gbm_market"    # best log-loss: uses the market line when there is one
PRODUCTION_OPINION = "gbm_opinion"   # our independent number: no market inputs at all


def cache_dir() -> Path | None:
    """$COURTSIDE_CACHE (scratch for engine passes and rung-3 reports), or None."""
    d = os.environ.get("COURTSIDE_CACHE")
    return Path(d).expanduser() if d else None


# --- caches ----------------------------------------------------------------------------------

def _frame_hash(df: pd.DataFrame, cols) -> np.ndarray:
    cols = [c for c in cols if c in df.columns]
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


class DiskEngineCache(player_model.EngineCache):
    """EngineCache that also persists full passes under `disk_dir` (keyed by the games' row
    hashes, the player rows, params and modes), so reruns skip the ~80 s engine pass."""

    def __init__(self, players, params: pl.PlayerParams = pl.TUNED, modes=ENGINE_MODES,
                 disk_dir: Path | None = None):
        super().__init__(players, params, modes)
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self._pfp = None

    def _players_fp(self) -> str:
        if self._pfp is None:
            p = self.players
            h = pd.util.hash_pandas_object(p, index=False).to_numpy()
            self._pfp = hashlib.sha1(h.tobytes()).hexdigest()[:16]
        return self._pfp

    def get(self, games: pd.DataFrame) -> pd.DataFrame:
        if self.disk_dir is None:
            return super().get(games)
        h = player_model._row_hashes(games)
        for hc, res in self._runs:
            if len(h) <= len(hc) and np.array_equal(h, hc[:len(h)]):
                out = res.iloc[:len(h)].copy()
                out.index = games.index
                return out
        key = hashlib.sha1(h.tobytes() + self._players_fp().encode()
                           + repr((self.params.to_dict(), self.modes, pl.REGULAR_MIN,
                                   pl.playoff_rotation())).encode()).hexdigest()[:20]
        path = self.disk_dir / f"engine_{key}.pkl"
        if path.exists():
            res = pickle.loads(path.read_bytes())
        else:
            res = pl.run_player_engine(games, self.players, self.params, self.modes)
            self.n_runs += 1
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(res.reset_index(drop=True)))
        res = res.reset_index(drop=True)
        self._runs.append((h, res))
        self._runs = self._runs[-3:]
        out = res.copy()
        out.index = games.index
        return out

    def state(self, games: pd.DataFrame) -> pl.PlayerState:
        """PlayerState at the end of `games` (ratings, minutes projections, rosters) from one
        pass, which is also stored as the cached run (so it doubles as `prime`)."""
        h = player_model._row_hashes(games)
        for hc, st in getattr(self, "_states", []):
            if np.array_equal(h, hc):
                return st
        res, st = pl.run_player_engine(games, self.players, self.params, self.modes,
                                       return_state=True)
        self.n_runs += 1
        self._runs.append((h, res.reset_index(drop=True)))
        self._runs = self._runs[-3:]
        self._states = [(h, st)]
        return st


_FEAT_KEY_COLS = (*player_model._KEY_COLS, "home_pts", "away_pts", "home_win", "season",
                  *ft.BOX_OUTCOME_COLUMNS, "mkt_margin", "mkt_total", "lineup_home",
                  "lineup_away")


class FeatureCache:
    """Feature matrices for game frames. Like EngineCache, a frame that is an exact prefix of a
    cached one (same row hashes over ids, results, box and market columns) is served by
    slicing, which is valid because every feature is causal."""

    def __init__(self, engine: player_model.EngineCache | None,
                 box: pd.DataFrame | Callable[[], pd.DataFrame] | None = None,
                 closing: pd.DataFrame | Callable[[], pd.DataFrame] | None = None):
        self.engine = engine
        self._box, self._closing = box, closing
        self._runs: list[tuple[np.ndarray, dict]] = []
        self.n_runs = 0

    def _get(self, attr):
        v = getattr(self, attr)
        if callable(v):
            v = v()
            setattr(self, attr, v)
        return v

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach box/market columns from the loaders where the frame lacks them."""
        box = self._get("_box")
        if box is not None and "h_poss" not in frame.columns:
            frame = ft.attach_box(frame, box)
        closing = self._get("_closing")
        if closing is not None:
            frame = ft.attach_market(frame, closing)
        return frame

    def prime(self, frame: pd.DataFrame) -> None:
        self.get(frame)

    def get(self, frame: pd.DataFrame, ps_mode: str = "avail_tight") -> pd.DataFrame:
        frame = self.prepare(frame)
        h = _frame_hash(frame, _FEAT_KEY_COLS)
        for hc, res in self._runs:
            if ps_mode in res and len(h) <= len(hc) and np.array_equal(h, hc[:len(h)]):
                out = res[ps_mode].iloc[:len(h)].copy()
                out.index = frame.index
                return out
        eng = self.engine.get(frame) if self.engine is not None else None
        players = self.engine.players if self.engine is not None else None
        e = elo_mod.run_elo(frame, elo_mod.TUNED)
        res = {}
        for mode in dict.fromkeys((ps_mode, "avail_tight", "avail")):
            if eng is not None and f"{mode}_home" not in eng:
                continue
            res[mode] = ft.build_features(frame, eng, e, players, ps_mode=mode).reset_index(
                drop=True)
        if ps_mode not in res:
            res[ps_mode] = ft.build_features(frame, None, e, players).reset_index(drop=True)
        self.n_runs += 1
        self._runs.append((h, res))
        self._runs = self._runs[-2:]
        out = res[ps_mode].copy()
        out.index = frame.index
        return out


# --- configuration ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GBMConfig:
    num_leaves: int = 7
    min_data_in_leaf: int = 300
    learning_rate: float = 0.05
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    lambda_l2: float = 10.0
    max_iter: int = 400
    half_life: float | None = None   # recency weight half-life in seasons (None: equal)

    def lgb_params(self, seed: int = 0, threads: int = 4) -> dict:
        return {"objective": "regression", "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves, "min_data_in_leaf": self.min_data_in_leaf,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction, "bagging_freq": 1,
                "lambda_l2": self.lambda_l2, "max_bin": 63, "verbose": -1, "seed": seed,
                "deterministic": True, "force_row_wise": True, "num_threads": threads}

    def to_dict(self) -> dict:
        return asdict(self)


GRID = tuple(GBMConfig(num_leaves=nl, half_life=hl)
             for nl in (7, 15) for hl in (None, 4.0))


def _season_num(s) -> np.ndarray:
    return np.array([int(str(x)[:4]) for x in np.asarray(s)])


def _weights(seasons: np.ndarray, ref: int, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(seasons))
    return 0.5 ** ((ref - seasons) / half_life)


# --- one target head --------------------------------------------------------------------------

def _base_design(F: pd.DataFrame, target: str, use_market: bool) -> pd.DataFrame:
    """Linear-base design. NaNs are filled with neutral values (the GBM sees the raw NaNs)."""
    hc = F["home_court"].to_numpy(float)
    if target == "margin":
        post = F["is_playoff"].to_numpy(float) + F["is_playin"].to_numpy(float)
        X = pd.DataFrame({
            "ps_diff": np.nan_to_num(F["ps_diff"].to_numpy(float)),
            "elo_pts": F["elo_diff"].to_numpy(float) / ft.ELO_PTS,
            "home_court": hc,
            "hc_post": hc * post,
            "hc_lg_hca": hc * np.nan_to_num(F["lg_hca"].to_numpy(float)),
            "hc_apm_hca": hc * np.nan_to_num(F["apm_hca"].to_numpy(float)),
            **{c: F[c].to_numpy(float) for c in ft.SCHED},
        }, index=F.index)
        if use_market:
            X["mkt_margin"] = F["mkt_margin"].to_numpy(float)
        return X
    lg = F["lg_total"].to_numpy(float)
    lg = np.where(np.isfinite(lg), lg, 200.0)
    th, ta = F["tot_r10_home"].to_numpy(float), F["tot_r10_away"].to_numpy(float)
    both = np.column_stack([th, ta])
    cnt = np.isfinite(both).sum(axis=1)
    tavg = np.where(cnt > 0, np.nansum(both, axis=1) / np.maximum(cnt, 1), lg)
    pace = F["pace_r10_sum"].to_numpy(float) / 2
    rt = (F["ortg_r10_home"].to_numpy(float) + F["drtg_r10_away"].to_numpy(float)
          + F["ortg_r10_away"].to_numpy(float) + F["drtg_r10_home"].to_numpy(float)) / 200
    proj = pace * rt
    proj = np.where(np.isfinite(proj), proj, tavg)
    X = pd.DataFrame({"const": 1.0, "lg_total": lg, "tot_r10_avg": tavg, "proj_pace_rtg": proj},
                     index=F.index)
    if use_market:
        X["mkt_total"] = F["mkt_total"].to_numpy(float)
    return X


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    sw = np.sqrt(w)
    A = X * sw[:, None]
    P = np.eye(X.shape[1]) * ridge
    return np.linalg.solve(A.T @ A + P, A.T @ (y * sw))


def _ps_split(F: pd.DataFrame) -> np.ndarray:
    """(full-roster diff, home absence loss, away absence loss), NaN -> 0, consistent with
    nan_to_num(ps_diff) in the base design."""
    ok = np.isfinite(F["ps_diff"].to_numpy(float))
    ah = np.where(ok, np.nan_to_num(F["abs_loss_home"].to_numpy(float)), 0.0)
    aa = np.where(ok, np.nan_to_num(F["abs_loss_away"].to_numpy(float)), 0.0)
    pd_ = np.nan_to_num(F["ps_diff"].to_numpy(float))
    return np.column_stack([pd_ + ah - aa, ah, aa])


class _Base:
    """Linear base with a market fallback: rows with a finite market column use the full fit."""

    def __init__(self, target: str, use_market: bool):
        self.target, self.use_market = target, use_market
        self.mcol = "mkt_margin" if target == "margin" else "mkt_total"

    def fit(self, F: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "_Base":
        X = _base_design(F, self.target, self.use_market)
        S = _ps_split(F)
        self.cols_fb = [c for c in X.columns if c != self.mcol]
        self.coef_fb = _wls(X[self.cols_fb].to_numpy(), y, w)
        self.mean_fb = np.average(X[self.cols_fb].to_numpy(), axis=0, weights=w)
        self.split_fb = np.average(S, axis=0, weights=w)
        self.coef_full = None
        if self.use_market:
            ok = np.isfinite(X[self.mcol].to_numpy())
            if ok.sum() > 50:
                self.cols_full = list(X.columns)
                A = X[self.cols_full].to_numpy()[ok]
                self.coef_full = _wls(A, y[ok], w[ok])
                self.mean_full = np.average(A, axis=0, weights=w[ok])
                self.split_full = np.average(S[ok], axis=0, weights=w[ok])
        return self

    def _has_full(self, X):
        if self.coef_full is None:
            return np.zeros(len(X), bool)
        return np.isfinite(X[self.mcol].to_numpy())

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        X = _base_design(F, self.target, self.use_market)
        out = X[self.cols_fb].to_numpy() @ self.coef_fb
        full = self._has_full(X)
        if full.any():
            out[full] = X[self.cols_full].to_numpy()[full] @ self.coef_full
        return out

    def contributions(self, F: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Per-row coef_j * (x_j - mean_j) by base column, and the per-row bias."""
        X = _base_design(F, self.target, self.use_market)
        cols = list(X.columns)
        C = pd.DataFrame(0.0, index=F.index, columns=cols)
        A = X[self.cols_fb].to_numpy()
        C[self.cols_fb] = (A - self.mean_fb) * self.coef_fb
        bias = np.full(len(F), float(self.mean_fb @ self.coef_fb))
        full = self._has_full(X)
        if full.any():
            B = X[self.cols_full].to_numpy()[full]
            C.loc[full, self.cols_full] = (B - self.mean_full) * self.coef_full
            bias[full] = float(self.mean_full @ self.coef_full)
        if "ps_diff" in cols:
            # ps_diff = full-roster diff - home absences + away absences (exactly), so its
            # linear term splits into player_strength and per-team absences
            b = np.full(len(F), self.coef_fb[self.cols_fb.index("ps_diff")])
            m = np.tile(self.split_fb, (len(F), 1))
            if full.any():
                b[full] = self.coef_full[self.cols_full.index("ps_diff")]
                m[full] = self.split_full
            S = _ps_split(F)
            # absences measured from a fully healthy roster (0 = nobody out), the strength
            # term from the training mean; b * (m_h - m_a) moves into the bias
            C["ps_diff"] = 0.0
            C["ps_full_diff"] = b * (S[:, 0] - m[:, 0])
            C["abs_loss_home_lin"] = -b * S[:, 1]
            C["abs_loss_away_lin"] = b * S[:, 2]
            bias = bias + b * (m[:, 1] - m[:, 2])
        return C, bias

    def describe(self) -> dict:
        d = {"fallback": dict(zip(self.cols_fb, map(float, self.coef_fb)))}
        if self.coef_full is not None:
            d["with_market"] = dict(zip(self.cols_full, map(float, self.coef_full)))
        return d


def _gbm_cols(use_market: bool) -> list[str]:
    return list(ft.CORE_FEATURES) + (list(ft.MARKET_FEATURES) if use_market else [])


@dataclass
class _Fit:
    base: _Base
    booster: object | None
    curve: np.ndarray | None = None      # validation MSE after k trees, k = 0..max_iter
    val_pred_base: np.ndarray | None = None
    val_X: np.ndarray | None = None


def _fit_head(F: pd.DataFrame, y: np.ndarray, w: np.ndarray, target: str, use_market: bool,
              cfg: GBMConfig, n_iter: int, Fv: pd.DataFrame | None = None,
              yv: np.ndarray | None = None, threads: int = 4) -> _Fit:
    import lightgbm as lgb
    base = _Base(target, use_market).fit(F, y, w)
    cols = _gbm_cols(use_market)
    b0 = base.predict(F)
    fit = _Fit(base, None)
    if n_iter <= 0:
        if Fv is not None:
            fit.curve = np.array([np.mean((yv - base.predict(Fv)) ** 2)])
        return fit
    dtr = lgb.Dataset(F[cols].to_numpy(float), label=y, weight=w, init_score=b0,
                      feature_name=cols, free_raw_data=False)
    params = cfg.lgb_params(threads=threads)
    if Fv is None:
        fit.booster = lgb.train(params, dtr, num_boost_round=n_iter)
        return fit
    bv = base.predict(Fv)
    Xv = Fv[cols].to_numpy(float)
    dva = lgb.Dataset(Xv, label=yv, init_score=bv, reference=dtr)
    ev: dict = {}
    fit.booster = lgb.train(params, dtr, num_boost_round=n_iter, valid_sets=[dva],
                            valid_names=["val"], callbacks=[lgb.record_evaluation(ev)])
    fit.curve = np.concatenate([[np.mean((yv - bv) ** 2)], ev["val"]["l2"]])
    fit.val_pred_base, fit.val_X = bv, Xv
    return fit


def _head_predict(fit: _Fit, F: pd.DataFrame, n_iter: int, use_market: bool) -> np.ndarray:
    out = fit.base.predict(F)
    if fit.booster is not None and n_iter > 0:
        out = out + fit.booster.predict(F[_gbm_cols(use_market)].to_numpy(float),
                                        num_iteration=n_iter, raw_score=True)
    return out


class FitMemo:
    """Inner-fold fits shared across outer folds (the fit for validation season V with a given
    config is the same for every outer fold that uses V). Keyed by the content of the data."""

    def __init__(self):
        self.d: dict = {}
        self.hits = self.misses = 0

    def get(self, key, make):
        if key in self.d:
            self.hits += 1
            return self.d[key]
        self.misses += 1
        v = make()
        self.d[key] = v
        return v


# --- calibration -----------------------------------------------------------------------------

class Calibrator:
    """P(home win) from the predicted margin. kind: gauss | platt | isotonic."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, mu: np.ndarray, y: np.ndarray, post: np.ndarray) -> "Calibrator":
        if self.kind == "gauss":
            # scale-only probit, P = Phi(mu / sigma), sigma by maximum likelihood
            from scipy.optimize import minimize_scalar

            def nll(ls):
                p = np.clip(norm.cdf(mu / np.exp(ls)), 1e-12, 1 - 1e-12)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.sigma = float(np.exp(minimize_scalar(nll, bounds=(0.0, 4.0),
                                                      method="bounded").x))
        elif self.kind == "platt":
            self.clf = LogisticRegression(C=1e6, max_iter=1000).fit(
                np.column_stack([mu, mu * post]), y)
        elif self.kind == "isotonic":
            self.iso = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip",
                                          increasing=True).fit(mu, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, mu: np.ndarray, post: np.ndarray) -> np.ndarray:
        if self.kind == "gauss":
            return norm.cdf(mu / self.sigma)
        if self.kind == "platt":
            return self.clf.predict_proba(np.column_stack([mu, mu * post]))[:, 1]
        return self.iso.predict(mu)

    def describe(self) -> dict:
        if self.kind == "gauss":
            return {"kind": "gauss", "sigma": self.sigma}
        if self.kind == "platt":
            return {"kind": "platt", "coef": list(map(float, self.clf.coef_[0])),
                    "intercept": float(self.clf.intercept_[0])}
        return {"kind": "isotonic", "n_knots": int(len(self.iso.X_thresholds_))}


CALIBRATORS = ("gauss", "platt", "isotonic")


def choose_calibrator(mu, y, post, season, kinds=CALIBRATORS) -> tuple[Calibrator, dict]:
    """Leave-one-season-out log-loss over the inner out-of-fold predictions; refit the winner
    on all of them."""
    seasons = np.unique(season)
    scores = {}
    for k in kinds:
        if len(seasons) < 2:
            scores[k] = 0.0
            continue
        ll = []
        for s in seasons:
            tr, te = season != s, season == s
            c = Calibrator(k).fit(mu[tr], y[tr], post[tr])
            ll.append(metrics.log_loss(y[te], c.predict(mu[te], post[te])) * te.sum())
        scores[k] = float(np.sum(ll) / len(y))
    best = min(kinds, key=lambda k: (scores[k], kinds.index(k)))
    return Calibrator(best).fit(mu, y, post), scores


# --- explain ---------------------------------------------------------------------------------

# feature -> (factor, side); side None = a home-minus-away quantity
FACTORS = {
    "ps_diff": ("player_strength", None), "ps_home": ("player_strength", None),
    "ps_away": ("player_strength", None),
    "abs_loss_home": ("absences", "home"), "abs_loss_away": ("absences", "away"),
    "abs_loss_home_lin": ("absences", "home"), "abs_loss_away_lin": ("absences", "away"),
    "ps_full_diff": ("player_strength", None),
    "elo_diff": ("team_rating", None), "elo_pts": ("team_rating", None),
    "home_court": ("home_court", "home"), "hc_post": ("home_court", "home"),
    "hc_lg_hca": ("home_court", "home"), "hc_apm_hca": ("home_court", "home"),
    "lg_hca": ("home_court", "home"), "apm_hca": ("home_court", "home"),
    "home_b2b": ("rest", "home"), "home_three_in_four": ("rest", "home"),
    "home_four_in_six": ("rest", "home"), "away_b2b": ("rest", "away"),
    "away_three_in_four": ("rest", "away"), "away_four_in_six": ("rest", "away"),
    "rest_diff": ("rest", None), "games_last7_diff": ("rest", None),
    "travel_diff_1000km": ("travel", None), "tz_abs_diff": ("travel", None),
    "net_diff_r5": ("recent_form", None), "net_diff_r10": ("recent_form", None),
    "net_diff_r20": ("recent_form", None), "efg_diff_r10": ("recent_form", None),
    "tov_diff_r10": ("recent_form", None), "orb_diff_r10": ("recent_form", None),
    "ftr_diff_r10": ("recent_form", None),
    "cont_home": ("continuity", "home"), "cont_away": ("continuity", "away"),
    "gp_home": ("season_phase", None), "gp_away": ("season_phase", None),
    "season_day": ("season_phase", None), "is_playoff": ("season_phase", None),
    "is_playin": ("season_phase", None),
    "mkt_margin": ("market", None), "mkt_total": ("market", None),
}


def factor_of(col: str) -> tuple[str, str | None]:
    return FACTORS.get(col, ("pace_scoring", None))


def reasons_from_contrib(contrib: pd.Series, home: str, away: str, top: int = 5) -> list[dict]:
    """Group per-feature home-margin contributions into factors. impact_pts is the effect on
    the named team's expected margin (absences of LAL: negative for LAL)."""
    agg: dict = {}
    for col, v in contrib.items():
        if col.startswith("_") or not np.isfinite(v):
            continue
        key = factor_of(col)
        agg[key] = agg.get(key, 0.0) + float(v)
    out = []
    for (factor, side), v in agg.items():
        if side == "home":
            team, impact = home, v
        elif side == "away":
            team, impact = away, -v
        else:
            team, impact = (home, v) if v >= 0 else (away, -v)
        out.append({"factor": factor, "team": team, "impact_pts": round(impact, 2),
                    "home_margin_pts": round(v, 2)})
    out.sort(key=lambda r: (-abs(r["home_margin_pts"]), r["factor"], r["team"]))
    return out[:top]


# --- the model -------------------------------------------------------------------------------

class GBMModel:
    """See module docstring. `market`: use closing/live spread and total as features."""

    def __init__(self, fcache: FeatureCache, name: str, market: bool, memo: FitMemo | None = None,
                 grid: tuple[GBMConfig, ...] = GRID,
                 fixed: dict[str, tuple[GBMConfig, int]] | None = None,
                 burn_in: int = 3, n_inner: int = 3, calibration: str = "auto",
                 ps_mode: str = "avail_tight", drift: bool = True,
                 targets: tuple[str, ...] = ("margin", "total"), linear_only: bool = False,
                 threads: int = 4):
        self.fc, self.name, self.market = fcache, name, market
        self.memo = memo if memo is not None else FitMemo()
        self.grid, self.fixed = grid, fixed
        self.burn_in, self.n_inner, self.calibration = burn_in, n_inner, calibration
        self.ps_mode, self.drift, self.targets = ps_mode, drift, targets
        self.linear_only, self.threads = linear_only, threads
        self._pred_cache: tuple | None = None

    # features
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        F = self.fc.get(df, self.ps_mode)
        if not self.drift:
            F = F.copy()
            F["lg_hca"] = np.nan
            F["apm_hca"] = np.nan
        return F

    def _grid(self):
        g = self.grid
        if self.linear_only:
            g = (replace(g[0], max_iter=0),)
        if not self.drift:
            g = tuple(dict.fromkeys(replace(c, half_life=None) for c in g))
        return g

    # fitting
    def _search(self, F, y, sn, target, keep):
        """Inner walk-forward over the last n_inner seasons. Returns (cfg, n_iter, oof)."""
        seasons = sorted(set(sn[keep]))
        vals = seasons[-self.n_inner:] if len(seasons) > self.n_inner else seasons[1:]
        grid = self._grid()
        fixed = (self.fixed or {}).get(target)
        if fixed is not None:   # frozen config: inner fits only for the calibrator
            grid = (replace(fixed[0], max_iter=fixed[1]),)
        results = []
        for cfg in grid:
            fits = []
            for V in vals:
                tr = keep & (sn < V)
                va = keep & (sn == V)
                if tr.sum() < 100 or va.sum() == 0:
                    continue
                key = (target, self.market, self.ps_mode, self.drift, cfg, int(V),
                       hashlib.sha1(np.ascontiguousarray(y[tr | va]).tobytes()
                                    + pd.util.hash_pandas_object(F[tr | va], index=False)
                                    .to_numpy().tobytes()).hexdigest())
                fits.append((V, va, self.memo.get(key, lambda tr=tr, va=va, V=V, cfg=cfg: _fit_head(
                    F[tr], y[tr], _weights(sn[tr], V, cfg.half_life), target, self.market, cfg,
                    cfg.max_iter, F[va], y[va], self.threads))))
            if not fits:
                continue
            L = min(len(f.curve) for _, _, f in fits)
            tot = sum(f.curve[:L] * va.sum() for _, va, f in fits)
            n = sum(va.sum() for _, va, _ in fits)
            k = int(np.argmin(tot)) if fixed is None else L - 1
            results.append((float(tot[k] / n), cfg, k, fits))
        if not results:
            return (fixed or (self._grid()[0], 0))[0], 0, None, []
        results.sort(key=lambda r: r[0])
        mse, cfg, k, fits = results[0]
        oof = []
        for V, va, f in fits:
            pred = f.val_pred_base.copy() if f.val_pred_base is not None else None
            if pred is None:
                pred = f.base.predict(F[va])
            if f.booster is not None and k > 0:
                pred = pred + f.booster.predict(f.val_X, num_iteration=k, raw_score=True)
            oof.append((np.flatnonzero(va), pred, V))
        table = [{"cfg": c.to_dict(), "n_iter": kk, "val_mse": m} for m, c, kk, _ in results]
        return cfg, k, oof, table

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_ = train_df
        F = self._features(train_df)
        sn = _season_num(train_df["season"])
        first = sorted(set(sn))
        keep = sn >= first[min(self.burn_in, len(first) - 1)]
        ref = int(sn.max()) + 1
        self.heads_ = {}
        self.search_ = {}
        for target in self.targets:
            y = (train_df["margin"].to_numpy(float) if target == "margin"
                 else (train_df["home_pts"] + train_df["away_pts"]).to_numpy(float))
            cfg, k, oof, table = self._search(F, y, sn, target, keep)
            w = _weights(sn[keep], ref, cfg.half_life)
            fit = _fit_head(F[keep], y[keep], w, target, self.market, cfg, k,
                            threads=self.threads)
            self.heads_[target] = (fit, k, cfg)
            self.search_[target] = table
            if target == "margin":
                self._fit_calibrator(train_df, oof, F, y)
        self._pred_cache = None

    def _fit_calibrator(self, train_df, oof, F, y):
        yb = train_df["home_win"].to_numpy()
        post = (train_df["season_type"].to_numpy() != "Regular Season").astype(float)
        if oof:
            idx = np.concatenate([i for i, _, _ in oof])
            mu = np.concatenate([p for _, p, _ in oof])
            ss = np.concatenate([np.full(len(i), V) for i, _, V in oof])
        else:   # no inner folds (tiny data): in-sample
            fit, k, _ = self.heads_["margin"]
            idx = np.arange(len(train_df))
            mu = _head_predict(fit, F, k, self.market)
            ss = np.zeros(len(idx))
        kinds = CALIBRATORS if self.calibration == "auto" else (self.calibration,)
        self.cal_, self.cal_scores_ = choose_calibrator(mu, yb[idx], post[idx], ss, kinds)
        self.oof_resid_sd_ = float(np.sqrt(np.mean((y[idx] - mu) ** 2)))

    # prediction
    def _test_features(self, df: pd.DataFrame) -> pd.DataFrame:
        full = pd.concat([self.train_, df])
        return self._features(full).iloc[len(self.train_):].set_axis(df.index)

    def _predict_all(self, df: pd.DataFrame) -> dict:
        key = (len(df), hashlib.sha1(_frame_hash(df, _FEAT_KEY_COLS).tobytes()).hexdigest())
        if self._pred_cache is not None and self._pred_cache[0] == key:
            return self._pred_cache[1]
        F = self._test_features(df)
        out = {"F": F}
        for target, (fit, k, _) in self.heads_.items():
            out[target] = _head_predict(fit, F, k, self.market)
        if "margin" in out:
            post = (df["season_type"].to_numpy() != "Regular Season").astype(float)
            out["p"] = np.clip(self.cal_.predict(out["margin"], post), 1e-4, 1 - 1e-4)
        self._pred_cache = (key, out)
        return out

    # production helpers
    def prepare(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Optional, before fit: one engine pass over train+test (fit and predict then both
        slice it), keeping the player state for `with_injuries`."""
        full = pd.concat([train_df, test_df])
        if isinstance(self.fc.engine, DiskEngineCache):
            self.fc.engine.state(full)
        self.fc.prime(full)

    def with_injuries(self, conn, test_df: pd.DataFrame, as_of: str,
                      sources=("official", "espn")) -> pd.DataFrame:
        """test_df + lineup_home / lineup_away: expected-lineup strength from the latest injury
        snapshot at or before `as_of` (players.availability_from_injuries; tighter rotation in
        the postseason). Pass the result to predict_* / explain. Teams without a snapshot get
        NaN (the model then assumes the recent roster is available)."""
        full = pd.concat([self.train_, test_df])
        st = self.fc.engine.state(full)
        out = test_df.copy()
        for side in ("home", "away"):
            vals = []
            for _, r in test_df.iterrows():
                av = pl.availability_from_injuries(conn, st, r[side], as_of, sources)
                if not av:
                    vals.append(np.nan)
                    continue
                lu = st.expected_lineup(r[side], None, av,
                                        playoff=r["season_type"] != "Regular Season")
                vals.append(float(lu.attrs["strength"]))
            out[f"lineup_{side}"] = vals
        return out

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_all(df)["margin"].copy()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_all(df)["p"].copy()

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        if "total" not in self.heads_:
            return np.full(len(df), np.nan)
        return self._predict_all(df)["total"].copy()

    def contributions(self, df: pd.DataFrame, target: str = "margin") -> pd.DataFrame:
        """Per-row additive decomposition of the prediction (points): linear-base terms
        coef*(x - mean) plus LightGBM SHAP values (pred_contrib), columns by feature, plus
        `_bias`. Rows sum exactly to the prediction."""
        F = self._predict_all(df)["F"]
        fit, k, _ = self.heads_[target]
        C, bias = fit.base.contributions(F)
        C = C.copy()
        if fit.booster is not None and k > 0:
            cols = _gbm_cols(self.market)
            S = fit.booster.predict(F[cols].to_numpy(float), num_iteration=k, pred_contrib=True)
            for j, c in enumerate(cols):
                C[c] = (C[c] if c in C else 0.0) + S[:, j]
            bias = bias + S[:, -1]
        C["_bias"] = bias
        return C

    def explain(self, row, df: pd.DataFrame | None = None, top: int = 5) -> list[dict]:
        """Top factors behind the predicted home margin of one game.

        row: a row (Series) or 1-row DataFrame of `df` (the frame passed to predict_*), or a
        game_id. Returns [{"factor", "team", "impact_pts", "home_margin_pts"}, ...] sorted by
        |home_margin_pts|; impact_pts is from the named team's point of view."""
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        gid = row if isinstance(row, str) else row["game_id"]
        if df is None:
            df = row.to_frame().T if not isinstance(row, str) else None
            if df is None:
                raise ValueError("pass df when explaining by game_id")
            df = df.astype({c: self.train_[c].dtype for c in df.columns
                            if c in self.train_.columns})
        C = self.contributions(df)
        i = int(np.flatnonzero(df["game_id"].to_numpy() == gid)[0])
        r = df.iloc[i]
        return reasons_from_contrib(C.iloc[i], r["home"], r["away"], top)

    def chosen(self) -> dict[str, tuple[GBMConfig, int]]:
        """{target: (config, n_trees)} picked by the inner search (the FROZEN format)."""
        return {t: (cfg, k) for t, (_, k, cfg) in self.heads_.items()}

    def describe(self) -> dict:
        d = {"market": self.market, "ps_mode": self.ps_mode, "drift": self.drift,
             "calibrator": self.cal_.describe(), "calibrator_scores": self.cal_scores_}
        for t, (fit, k, cfg) in self.heads_.items():
            d[t] = {"n_iter": k, "cfg": cfg.to_dict(), "base": fit.base.describe(),
                    "search": self.search_.get(t)}
            if fit.booster is not None and k > 0:
                gain = fit.booster.feature_importance("gain", iteration=k)
                d[t]["gain"] = dict(zip(_gbm_cols(self.market), map(float, gain)))
        return d


# --- registry --------------------------------------------------------------------------------

# Production: hyperparameters frozen from the walk-forward backtest (the config and tree count
# chosen most often by the inner search; see MODEL.md, rung 3), so the daily fit skips the
# search. None = run the inner search on every fit.
# Chosen by the inner search on all games through 2025-26 (`courtside model gbm-timing`,
# 2026-10-18); refresh once a season. The walk-forward backtest never uses these.
FROZEN: dict[str, dict[str, tuple[GBMConfig, int]] | None] = {
    PRODUCTION_WINPROB: {"margin": (GBMConfig(num_leaves=7), 22),
                         "total": (GBMConfig(num_leaves=7, half_life=4.0), 12)},
    PRODUCTION_OPINION: {"margin": (GBMConfig(num_leaves=15), 89),
                         "total": (GBMConfig(num_leaves=7, half_life=4.0), 31)},
}

SPECS = {
    # name: kwargs for GBMModel
    PRODUCTION_WINPROB: {"market": True},
    PRODUCTION_OPINION: {"market": False},
}
ABLATION_SPECS = {
    "gbm_opinion_linear": {"market": False, "linear_only": True, "targets": ("margin",)},
    "gbm_market_linear": {"market": True, "linear_only": True, "targets": ("margin",)},
    "gbm_opinion_nodrift": {"market": False, "drift": False, "targets": ("margin",)},
    "gbm_opinion_loose": {"market": False, "ps_mode": "avail", "targets": ("margin",)},
    # the leaky full oracle ("played"), kept to document the leak (never use in production)
    "gbm_opinion_oracle": {"market": False, "ps_mode": "played", "targets": ("margin",)},
}


def make_caches(players, box=None, closing=None, params: pl.PlayerParams = pl.TUNED,
                disk_dir: Path | None = None) -> tuple[DiskEngineCache, FeatureCache]:
    eng = DiskEngineCache(players, params, ENGINE_MODES, disk_dir if disk_dir else cache_dir())
    return eng, FeatureCache(eng, box, closing)


def make_registry(players, box=None, closing=None, params: pl.PlayerParams = pl.TUNED,
                  ablations: bool = False, fcache: FeatureCache | None = None,
                  memo: FitMemo | None = None, frozen: bool = True, **kw
                  ) -> tuple[dict[str, Callable], FeatureCache]:
    """{name: factory} for the rung-3 models, sharing one engine/feature cache and fit memo.
    players/box/closing: DataFrames or zero-argument loaders (players.load_player_games,
    features.load_team_box, features.load_closing_full)."""
    if fcache is None:
        _, fcache = make_caches(players, box, closing, params)
    memo = memo or FitMemo()
    specs = {**SPECS, **(ABLATION_SPECS if ablations else {})}
    reg = {}
    for name, spec in specs.items():
        fx = FROZEN.get(name) if frozen else None
        reg[name] = (lambda n=name, s=spec, f=fx:
                     GBMModel(fcache, n, memo=memo, fixed=f, **{**kw, **s}))
    return reg, fcache


def registry_for_conn(conn, **kw) -> tuple[dict[str, Callable], FeatureCache]:
    """Production entry point: rung-3 registry reading players, box scores and closing lines
    from `conn` lazily."""
    return make_registry(lambda: pl.load_player_games(conn), lambda: ft.load_team_box(conn),
                         lambda: ft.load_closing_full(conn), **kw)
