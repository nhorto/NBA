import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from courtside.model import backtest, baselines, elo


def _games(spec, seasons=None):
    """spec: list of (date, home, away, margin[, neutral]) -> minimal games frame."""
    rows = []
    for i, s in enumerate(spec):
        date, h, a, m = s[:4]
        rows.append({"game_id": f"g{i:04d}", "game_date": pd.Timestamp(date),
                     "season": seasons[i] if seasons else "2010-11",
                     "season_type": "Regular Season", "home": h, "away": a,
                     "margin": m, "home_win": int(m > 0),
                     "is_neutral": bool(s[4]) if len(s) > 4 else False})
    return pd.DataFrame(rows)


P = elo.EloParams(k=20, hca=100, carry=0.75)


def test_first_game_hand_computed():
    g = _games([("2010-11-01", "A", "B", 10), ("2010-11-03", "A", "B", -5)])
    e = elo.run_elo(g, P)
    assert e["elo_home_pre"].iloc[0] == 1500 and e["elo_away_pre"].iloc[0] == 1500
    exp_home = 1 / (1 + 10 ** (-100 / 400))
    assert e["elo_prob"].iloc[0] == pytest.approx(exp_home)
    mult = (10 + 3) ** 0.8 / (7.5 + 0.006 * 100)
    shift = 20 * mult * (1 - exp_home)
    assert e["elo_home_pre"].iloc[1] == pytest.approx(1500 + shift)
    assert e["elo_away_pre"].iloc[1] == pytest.approx(1500 - shift)
    assert e["elo_margin"].iloc[0] == pytest.approx(100 / 28)
    assert e["elo_spread"].iloc[0] == pytest.approx(-100 / 28)


def test_autocorrelation_discounts_favourite_wins():
    # same margin, but a heavy favourite gains less than an underdog would
    g_fav = _games([("2010-11-01", "A", "B", 10)])
    g_dog = _games([("2010-11-01", "A", "B", 10, True)])  # neutral: dr=0
    fav = elo.run_elo(pd.concat([g_fav, _games([("2010-11-02", "A", "C", 1)])],
                                ignore_index=True), P)
    dog = elo.run_elo(pd.concat([g_dog, _games([("2010-11-02", "A", "C", 1)])],
                                ignore_index=True), P)
    assert fav["elo_home_pre"].iloc[1] - 1500 < dog["elo_home_pre"].iloc[1] - 1500


def test_neutral_site_has_no_home_advantage():
    e = elo.run_elo(_games([("2010-11-01", "A", "B", 5, True)]), P)
    assert e["elo_prob"].iloc[0] == pytest.approx(0.5)


def test_season_regression_and_expansion():
    g = _games([("2010-11-01", "A", "B", 20), ("2011-11-01", "A", "B", 3),
                ("2011-11-02", "C", "A", 3)], seasons=["2010-11", "2011-12", "2011-12"])
    e = elo.run_elo(g, P)
    after1 = 1500 + (e["elo_home_pre"].iloc[1] - P.mean) / P.carry  # undo regression
    r_end = elo.run_elo(g.iloc[:1].assign(), P, return_ratings=True)[1]["A"]
    assert e["elo_home_pre"].iloc[1] == pytest.approx(P.mean + P.carry * (r_end - P.mean))
    assert after1 > 1500
    assert e["elo_home_pre"].iloc[2] == P.expansion   # C first seen after the first season


def test_rest_adjustment_moves_probability():
    g = _games([("2010-11-01", "A", "B", 5)]).assign(home_b2b=1, away_b2b=0,
                                                     home_travel_km=0.0, away_travel_km=0.0)
    base = elo.run_elo(g, P)["elo_prob"].iloc[0]
    adj = elo.run_elo(g, replace(P, b2b_elo=40))["elo_prob"].iloc[0]
    assert adj == pytest.approx(1 / (1 + 10 ** (-(100 - 40) / 400)))
    assert adj < base


def test_unsorted_games_rejected():
    g = _games([("2010-11-05", "A", "B", 5), ("2010-11-01", "A", "B", 5)])
    with pytest.raises(ValueError):
        elo.run_elo(g, P)


def test_changing_a_future_result_never_changes_earlier_predictions(synthetic_frame):
    f = synthetic_frame()
    base = elo.run_elo(f, P)
    j = len(f) // 2
    for new_margin in (-40, 1, 40):
        alt = f.copy()
        alt.loc[j, "margin"] = new_margin
        alt.loc[j, "home_win"] = int(new_margin > 0)
        e = elo.run_elo(alt, P)
        # games up to and including j are pre-game: identical
        np.testing.assert_array_equal(e["elo_prob"].iloc[:j + 1], base["elo_prob"].iloc[:j + 1])
    # and the change does propagate forward (the test is not vacuous)
    assert not np.allclose(e["elo_prob"].iloc[j + 1:], base["elo_prob"].iloc[j + 1:])


@pytest.mark.parametrize("name", sorted(baselines.MODELS))
def test_registered_models_have_no_lookahead(synthetic_frame, name):
    f = synthetic_frame()
    test_season = f["season"].max()
    train, test = f[f["season"] < test_season], f[f["season"] == test_season]
    cut = test["game_date"].iloc[len(test) // 2]
    assert backtest.check_no_lookahead(baselines.MODELS[name], train, test, cut) == 0.0


def test_check_no_lookahead_catches_a_cheater(synthetic_frame):
    class Cheater:
        name = "cheat"
        def fit(self, train_df): pass
        def predict_proba(self, test_df):
            return np.clip(test_df["home_win"].to_numpy() * 0.9 + 0.05, 0, 1)
    f = synthetic_frame()
    s = f["season"].max()
    d = backtest.check_no_lookahead(Cheater, f[f.season < s], f[f.season == s],
                                    f[f.season == s]["game_date"].iloc[0])
    assert d > 0


def test_tuning_uses_only_the_tuning_window(synthetic_frame):
    f = synthetic_frame(seasons=("2009-10", "2010-11", "2011-12", "2012-13"))
    r1 = elo.tune(f, elo.default_grid()[:3])
    later = f["season"] > elo.TUNE_SEASONS[1]
    f2 = f.copy()
    f2.loc[later, "margin"] = -f2.loc[later, "margin"]
    f2.loc[later, "home_win"] = 1 - f2.loc[later, "home_win"]
    r2 = elo.tune(f2, elo.default_grid()[:3])
    assert [x[0] for x in r1] == [x[0] for x in r2]


def test_fit_pts_per_elo():
    dr = np.array([28.0, 56.0, -28.0])
    assert elo.fit_pts_per_elo(dr, dr / 28) == pytest.approx(28.0)
