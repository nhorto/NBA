import json

import numpy as np
import pytest

from courtside.model import backtest, baselines, elo


def test_walk_forward_split_boundaries(synthetic_frame):
    f = synthetic_frame()
    seasons = backtest.seasons_in(f)
    tests = seasons[1:]
    splits = list(backtest.walk_forward_splits(f, tests))
    assert [s for s, _, _ in splits] == tests
    for s, train, test in splits:
        assert (train["season"] < s).all()
        assert (test["season"] == s).all()
        assert set(train.index).isdisjoint(test.index)
        assert len(train) == (f["season"] < s).sum()       # all prior seasons, nothing else
        assert train["game_date"].max() < test["game_date"].min()


def test_first_season_cannot_be_tested(synthetic_frame):
    f = synthetic_frame()
    with pytest.raises(ValueError):
        list(backtest.walk_forward_splits(f, [f["season"].min()]))


def test_run_backtest_and_report(synthetic_frame, tmp_path):
    f = synthetic_frame()
    tests = backtest.seasons_in(f)[2:]
    preds, desc = backtest.run_backtest(f, baselines.MODELS, tests)
    n_test = (f["season"].isin(tests)).sum()
    assert len(preds) == n_test * len(baselines.MODELS)
    assert preds["p"].between(0, 1).all()
    ev = backtest.evaluate(preds)
    for name in baselines.MODELS:
        agg = ev[name]["aggregate"]
        assert set(agg) == {"Regular Season", "Playoffs", "All"}
        assert agg["All"]["n"] == n_test
        assert len(agg["Regular Season"]["calibration"]) == 10
        assert set(ev[name]["per_season"]) == set(tests)
    assert "Regular Season" in backtest.format_tables(ev)
    path = backtest.write_report(ev, desc, f, tmp_path)
    doc = json.loads(path.read_text())
    assert doc["results"]["elo"]["aggregate"]["All"]["n"] == n_test


def test_home_model_uses_train_rate(synthetic_frame):
    f = synthetic_frame()
    s = f["season"].max()
    m = baselines.HomeModel()
    m.fit(f[f.season < s])
    p = m.predict_proba(f[f.season == s])
    assert np.allclose(p, f[f.season < s]["home_win"].mean())


def test_elo_walk_forward_equals_one_sequential_pass(synthetic_frame):
    """Online Elo in fold S must equal a single chronological pass restricted to S."""
    f = synthetic_frame()
    full = elo.run_elo(f, elo.TUNED)
    for s, train, test in backtest.walk_forward_splits(f, backtest.seasons_in(f)[1:]):
        m = baselines.EloModel(elo.TUNED)
        m.fit(train)
        np.testing.assert_allclose(m.predict_proba(test), full.loc[test.index, "elo_prob"])
