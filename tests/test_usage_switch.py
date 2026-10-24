"""Usage v1 redistribution in the sim (`COURTSIDE_USAGE`, `+us`, IMPROVEMENTS.md P8).

OFF (TeamSpec.usage None) runs the tier bump exactly as before; ON reads usage v1's per-stat
factors of each sim's absent set."""

import numpy as np
import pandas as pd
import pytest

from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import state as ST
from courtside.proj import usage as U

from test_proj_copula import _league, _spec  # noqa: F401  (fixture helpers)
from test_proj_copula import game  # noqa: F401  (module fixture)


def _zero_L(n: int) -> dict:
    return {s: np.zeros((n, n)) for s in U.DK_STATS}


def test_default_off_and_version_tag():
    assert U.USAGE_DEFAULT is False
    assert U.USAGE is False                      # tests/conftest.py leaves it unset
    base = P.MODEL_VERSION
    try:
        P.set_usage(True)
        assert P.MODEL_VERSION == base + U.USAGE_TAG
        assert U.USAGE is True
    finally:
        P.set_usage(False)
    assert P.MODEL_VERSION == base and "+us" not in base


def test_teamspec_usage_defaults_none(game):
    spec, _, _ = _spec(game)
    assert spec.home.usage is None and spec.away.usage is None


def test_zero_factors_match_off_without_star_absence(game):
    """With all-zero log factors (every factor exactly 1) and no star absent, ON reads the
    'none' bump role everywhere, as OFF does: equal up to float summation order."""
    spec, mp, sp = _spec(game, p_out=[5])       # a non-star out: the bump does not fire
    cp = C.default_copula_params()
    off = C.simulate_game(spec, cp, mp, sp, 2000, 3)
    for ts in (spec.home, spec.away):
        ts.usage = _zero_L(len(ts.players))
    on = C.simulate_game(spec, cp, mp, sp, 2000, 3)
    assert np.array_equal(off.minutes, on.minutes)
    np.testing.assert_allclose(on.fp, off.fp, rtol=1e-4, atol=1e-3)


def test_positive_factor_moves_share_to_the_absorber(game):
    """A positive L[j, i] for an absent j raises i's share of the (reconciled) team points
    relative to OFF; the team total stays at the implied total."""
    spec, mp, sp = _spec(game, p_out=[5])
    cp = C.default_copula_params()
    off = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    pl = spec.home.players
    n = len(pl)
    L = _zero_L(n)
    i = 2
    for s in U.DK_STATS:
        L[s][5, i] = 0.3
    spec.home.usage = L
    spec.away.usage = _zero_L(len(spec.away.players))
    on = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    pid = pl["player_id"].to_numpy()
    ji = int(np.flatnonzero(on.player_id == pid[i])[0])
    home = on.info["team"].to_numpy() == spec.home.team
    share_off = off.info["pts_all"].to_numpy()[ji] / off.info.loc[home, "pts_all"].sum()
    share_on = on.info["pts_all"].to_numpy()[ji] / on.info.loc[home, "pts_all"].sum()
    assert share_on > share_off * 1.15
    assert on.info.loc[home, "pts_all"].sum() == pytest.approx(112.0, abs=0.5)


def test_factor_only_for_sims_where_absent(game):
    """j at P(available) 0.5: i's per-minute rate is lifted only in the sims j sits."""
    spec, mp, sp = _spec(game, p_half=[5])
    cp = C.default_copula_params()
    pl = spec.home.players
    L = _zero_L(len(pl))
    for s in U.DK_STATS:
        L[s][5, 2] = 0.3
    spec.home.usage = L
    spec.away.usage = _zero_L(len(spec.away.players))
    absent = np.zeros((4, len(pl)), bool)
    absent[1:3, 5] = True
    F = U.factor_matrix(L, absent, U.CAP)
    assert np.allclose(F["pts"][[0, 3], 2], 1.0)
    assert np.allclose(F["pts"][[1, 2], 2], np.exp(0.3))
    assert np.allclose(np.delete(F["pts"], 2, axis=1), 1.0)
    sims = C.simulate_game(spec, cp, mp, sp, 2000, 7)
    assert np.isfinite(sims.fp).all()


def test_live_params_path_keyed_by_rate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    a = U.live_params_path("2024-25", ST.default_params())
    sp = ST.default_params()
    sp["history_seasons"] = sp["history_seasons"] + 1
    b = U.live_params_path("2024-25", sp)
    assert a != b and a.name.startswith("usage_params_2024-25_")


def test_team_matrix_zero_without_evidence_rows():
    """`team_matrix` reads pair evidence strictly before the day; non-regulars get no row."""
    p = U.default_params()
    for s in U.DK_STATS:
        p.beta[s] = [0.5] * U.N_CELL

    class _D:
        def evidence(self, day, pids):
            return {}
    tp = pd.DataFrame({"player_id": [1, 2, 3], "mpg": [30.0, 25.0, 12.0],
                       "grp": [0, 0, 1], "star": [True, False, False],
                       "regular": [True, True, False]})
    for s in U.DK_STATS:
        tp[f"r_{s}"] = [0.5, 0.4, 0.3]
    L = U.team_matrix((p, _D()), tp, "2024-01-10")
    assert (L["pts"][2] == 0).all()                       # non-regular absentee: no effect
    assert L["pts"][0, 1] > 0 and L["pts"][0, 0] == 0     # star out lifts the teammate


def test_v1_path_zero_factors_match_off_without_star_absence(game):
    """Copula v1 / v1r (`simulate_game_v1`) reads the same factors: all-zero factors and no star
    out give the OFF component means (float summation order aside)."""
    spec, mp, sp = _spec(game, p_out=[5])
    cp = C.default_copula_params()
    v1 = C.default_copula_v1_params(cp=cp)
    off = C.simulate_game_v1(spec, cp, v1, mp, sp, 2000, 9)
    for ts in (spec.home, spec.away):
        ts.usage = _zero_L(len(ts.players))
    on = C.simulate_game_v1(spec, cp, v1, mp, sp, 2000, 9)
    assert np.array_equal(off.minutes, on.minutes)
    np.testing.assert_allclose(on.info["pts_all"], off.info["pts_all"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(on.fp.mean(0), off.fp.mean(0), rtol=1e-3, atol=1e-2)
