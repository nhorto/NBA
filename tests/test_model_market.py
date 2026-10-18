import numpy as np
import pandas as pd
import pytest

from courtside.model import market


def test_american_to_prob_and_devig():
    assert market.american_to_prob([-200, 200, 100]) == pytest.approx([2 / 3, 1 / 3, 0.5])
    ph, pa = market.american_to_prob([-150, 130])
    assert ph + pa > 1          # overround


def test_compare_ats_and_metrics():
    closing = pd.DataFrame({"game_id": ["a", "b", "c"], "source": "kaggle",
                            "home_spread_close": [-5.0, 3.0, -1.0],
                            "p_mkt": [0.66, 0.40, 0.52], "overround": 0.04})
    preds = pd.DataFrame({"model": "m", "game_id": ["a", "b", "c", "d"], "season": "2020-21",
                          "season_type": "Regular Season", "y": [1, 0, 1, 1],
                          "margin": [10, -1, -3, 5], "p": [0.7, 0.45, 0.6, 0.5],
                          "margin_pred": [8.0, -5.0, 2.0, 1.0]})
    r = market.compare(preds, closing)["m"]
    assert r["n_ml"] == 3 and r["n_spread"] == 3
    # a: pick home (8 > 5), home covered (10-5>0) win; b: pick away (-5 < -3), ats=-1+3=2>0 loss;
    # c: pick home (2 > 1), ats=-3+1=-2 loss
    assert r["ats"]["n"] == 3 and r["ats"]["win_rate"] == pytest.approx(1 / 3)
    assert r["margin_mae_market"] == pytest.approx(np.mean([5, 2, 4]))
