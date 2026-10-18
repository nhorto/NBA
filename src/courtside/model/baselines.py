"""Baseline models implementing the backtest Model interface (see model/backtest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from . import elo as elo_mod


class HomeModel:
    """Always pick the home team, with P = the training-set home win rate."""

    name = "home"

    def fit(self, train_df: pd.DataFrame) -> None:
        self.p_ = float(train_df["home_win"].mean())
        self.margin_ = float(train_df["margin"].mean())

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.p_)

    def predict_margin(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.margin_)

    def describe(self) -> dict:
        return {"p_home": self.p_, "margin": self.margin_}


def _burn_in_mask(train_df: pd.DataFrame) -> np.ndarray:
    """Drop the first season present (all teams start at the same rating), unless it is the
    only season available."""
    m = (train_df["season"] != train_df["season"].min()).to_numpy()
    return m if m.any() else np.ones(len(train_df), dtype=bool)


_KEY_COLS = ["game_id", "game_date", "home", "away", "margin", "is_neutral", "season_type"]


def _content_key(df: pd.DataFrame) -> int:
    cols = [c for c in _KEY_COLS + ["home_b2b", "away_b2b", "home_travel_km", "away_travel_km"]
            if c in df.columns]
    return int(pd.util.hash_pandas_object(df[cols], index=False).sum())


class _EloRunner:
    """Runs Elo over train + test (chronological) once per (train, test) pair."""

    def __init__(self, params: elo_mod.EloParams):
        self.params = params
        self._key = None
        self._res = None

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None):
        full = train_df if test_df is None else pd.concat([train_df, test_df])
        key = _content_key(full)
        if key != self._key:
            self._res = elo_mod.run_elo(full, self.params)
            self._key = key
        return self._res


class EloModel:
    """Elo with frozen hyperparameters (online: ratings update after every game, including
    test-season games once they are played). fit() only estimates the points-per-Elo scale for
    the implied margin from the training seasons."""

    def __init__(self, params: elo_mod.EloParams = elo_mod.TUNED, name: str = "elo"):
        self.params = params
        self.name = name
        self._runner = _EloRunner(params)

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_ = train_df
        e = self._runner.run(train_df, None)
        m = _burn_in_mask(train_df)
        self.pts_per_elo_ = elo_mod.fit_pts_per_elo(e["elo_dr"].to_numpy()[m],
                                                    train_df["margin"].to_numpy()[m])

    def _test_part(self, test_df):
        e = self._runner.run(self.train_, test_df)
        return e.iloc[len(self.train_):]

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._test_part(test_df)["elo_prob"].to_numpy()

    def predict_margin(self, test_df: pd.DataFrame) -> np.ndarray:
        return self._test_part(test_df)["elo_dr"].to_numpy() / self.pts_per_elo_

    def describe(self) -> dict:
        return {"params": self.params.to_dict(), "pts_per_elo_fit": self.pts_per_elo_}


# Schedule features for the logistic layer. Model is antisymmetric in home/away except for the
# explicit home-court terms, so there is no free intercept: a neutral-site game between equal,
# equally rested teams is exactly 50/50.
SCHED_FEATURES = (
    "home_b2b", "away_b2b", "home_three_in_four", "away_three_in_four",
    "home_four_in_six", "away_four_in_six",
    "rest_diff", "games_last7_diff", "travel_diff_1000km", "tz_abs_diff",
)


def elo_sched_design(df: pd.DataFrame, elo_rating_diff: np.ndarray) -> pd.DataFrame:
    """Design matrix: Elo rating difference (no HCA) in log-odds units, home-court indicators,
    schedule terms."""
    home_court = 1.0 - df["is_neutral"].to_numpy(dtype=float)
    playoff = (df["season_type"].to_numpy() != "Regular Season").astype(float)
    X = pd.DataFrame({
        "elo_diff_logit": elo_rating_diff * elo_mod.LN10_400,
        "home_court": home_court,
        "home_court_playoff": home_court * playoff,
        "home_b2b": df["home_b2b"].to_numpy(float),
        "away_b2b": df["away_b2b"].to_numpy(float),
        "home_three_in_four": df["home_three_in_four"].to_numpy(float),
        "away_three_in_four": df["away_three_in_four"].to_numpy(float),
        "home_four_in_six": df["home_four_in_six"].to_numpy(float),
        "away_four_in_six": df["away_four_in_six"].to_numpy(float),
        "rest_diff": df["rest_diff"].to_numpy(float),
        "games_last7_diff": (df["home_games_last7"] - df["away_games_last7"]).to_numpy(float),
        "travel_diff_1000km": (df["home_travel_km"] - df["away_travel_km"]).to_numpy(float) / 1000,
        "tz_abs_diff": (df["home_tz_change"].abs() - df["away_tz_change"].abs()).to_numpy(float),
    }, index=df.index)
    return X


class EloScheduleModel:
    """Logistic regression on the (frozen-parameter) Elo rating difference plus schedule
    features, refit walk-forward on each training window. Margin: linear regression on the same
    design (in points)."""

    def __init__(self, params: elo_mod.EloParams = elo_mod.TUNED, name: str = "elo_sched",
                 C: float = 1.0):
        # Elo used here must not include its own rest adjustment (the LR learns that).
        self.params = params
        self.name = name
        self.C = C
        self._runner = _EloRunner(params)

    def _diff(self, e: pd.DataFrame) -> np.ndarray:
        return (e["elo_home_pre"] - e["elo_away_pre"]).to_numpy()

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_ = train_df
        e = self._runner.run(train_df, None)
        m = _burn_in_mask(train_df)
        X = elo_sched_design(train_df, self._diff(e))[m]
        y = train_df["home_win"].to_numpy()[m]
        self.clf_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        self.clf_.fit(X.to_numpy(), y)
        self.reg_ = LinearRegression(fit_intercept=False)
        self.reg_.fit(X.to_numpy(), train_df["margin"].to_numpy(float)[m])
        self.columns_ = list(X.columns)

    def _X(self, test_df):
        e = self._runner.run(self.train_, test_df).iloc[len(self.train_):]
        return elo_sched_design(test_df, self._diff(e)).to_numpy()

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._X(test_df))[:, 1]

    def predict_margin(self, test_df: pd.DataFrame) -> np.ndarray:
        return self.reg_.predict(self._X(test_df))

    def describe(self) -> dict:
        return {"params": self.params.to_dict(), "C": self.C,
                "logit_coef": dict(zip(self.columns_, map(float, self.clf_.coef_[0]))),
                "margin_coef": dict(zip(self.columns_, map(float, self.reg_.coef_)))}


MODELS = {
    "home": HomeModel,
    "elo": lambda: EloModel(elo_mod.TUNED, "elo"),
    "elo_rest": lambda: EloModel(elo_mod.TUNED_REST, "elo_rest"),
    "elo_sched": lambda: EloScheduleModel(elo_mod.TUNED, "elo_sched"),
}
DEFAULT_MODELS = ("home", "elo", "elo_rest", "elo_sched")
