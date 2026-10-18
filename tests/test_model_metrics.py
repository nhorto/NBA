import math

import numpy as np
import pytest

from courtside.model import metrics


def test_log_loss_known_value():
    v = metrics.log_loss([1, 0], [0.8, 0.3])
    assert v == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert metrics.log_loss([1, 0, 1], [0.5, 0.5, 0.5]) == pytest.approx(math.log(2))


def test_log_loss_clips_extremes():
    assert np.isfinite(metrics.log_loss([1], [0.0]))


def test_brier_and_accuracy():
    assert metrics.brier([1, 0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2)
    assert metrics.brier([1, 1], [1.0, 1.0]) == 0.0
    # p >= 0.5 picks home; exact 0.5 counts as a home pick
    assert metrics.accuracy([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.5]) == pytest.approx(0.5)


def test_calibration_table_and_ece():
    y = [1, 0, 1, 1]
    p = [0.15, 0.15, 0.85, 0.95]
    tab = metrics.calibration_table(y, p, bins=10)
    assert len(tab) == 10
    assert tab[1]["n"] == 2 and tab[1]["mean_p"] == pytest.approx(0.15)
    assert tab[1]["frac_pos"] == pytest.approx(0.5)
    assert tab[8]["n"] == 1 and tab[9]["n"] == 1
    assert tab[0]["n"] == 0 and tab[0]["mean_p"] is None
    # ECE = 2/4*|0.15-0.5| + 1/4*|0.85-1| + 1/4*|0.95-1|
    assert metrics.ece(y, p) == pytest.approx(0.5 * 0.35 + 0.25 * 0.15 + 0.25 * 0.05)


def test_calibration_edges():
    tab = metrics.calibration_table([1, 0], [1.0, 0.0], bins=10)
    assert tab[9]["n"] == 1 and tab[0]["n"] == 1   # p=1.0 goes in the top bin


def test_mae_rmse():
    assert metrics.mae([3, -2], [1, 1]) == pytest.approx(2.5)
    assert metrics.rmse([3, -2], [1, 1]) == pytest.approx(math.sqrt((4 + 9) / 2))


def test_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.brier([1, 0], [0.5])
