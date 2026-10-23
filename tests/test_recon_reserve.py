"""Reconciliation reserve (COURTSIDE_RECON_RESERVE, `+rr`; `reconcile.RECON_RESERVE`).

OFF by default and nothing of it runs when off (same draws, same team info, same versions); ON
the reconcile target is implied - E[reserve minutes] x the league bench rate of season S at both
call sites (copula v0 and v1), the draws before reconciliation are unchanged (common random
numbers) and the version gets `+rr` after `+pq`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import availability as AV
from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC


@pytest.fixture
def restore():
    state = (RC.RECON_RESERVE, AV.PPLAYQ)
    yield
    P.set_pplayq(state[1])
    P.set_recon_reserve(state[0])


def test_default_off_and_env_flag(monkeypatch):
    assert RC.RECON_RESERVE_DEFAULT is False
    monkeypatch.setenv(RC.RECON_RESERVE_ENV, "1")
    assert RC._flag(RC.RECON_RESERVE_ENV, False) is True
    monkeypatch.setenv(RC.RECON_RESERVE_ENV, "off")
    assert RC._flag(RC.RECON_RESERVE_ENV, True) is False
    monkeypatch.delenv(RC.RECON_RESERVE_ENV)
    assert RC._flag(RC.RECON_RESERVE_ENV, RC.RECON_RESERVE_DEFAULT) is False


def test_version_tag_composes(restore):
    P.set_pplayq(False)
    P.set_recon_reserve(False)
    off_vs, off_mv = P._VS, P.MODEL_VERSION
    assert "+rr" not in off_mv
    P.set_recon_reserve(True)
    assert P._VS == off_vs and P.MODEL_VERSION == off_mv + "+rr"     # model version only
    P.set_pplayq(True)
    assert P.MODEL_VERSION == off_mv + "+pq+rr"
    P.set_recon_reserve(False)
    assert P.MODEL_VERSION == off_mv + "+pq"
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                  # seeds unchanged


def test_bench_rate_table_is_walk_forward():
    t = json.loads(RC.RR_PATH.read_text())
    assert t["rates"]
    for s, r in t["rates"].items():
        assert r["train_seasons"] and all(x < s for x in r["train_seasons"])
        assert 0.3 < r["ppm"] < 0.55
        assert RC.bench_ppm(s) == pytest.approx(r["ppm"])
    last = max(t["rates"])
    assert RC.bench_ppm("2099-00") == pytest.approx(t["rates"][last]["ppm"])
    assert RC.bench_ppm("1990-91") == RC.RR_FALLBACK_PPM


def test_reserve_target():
    r = RC.bench_ppm("2024-25")
    assert RC.reserve_target(110.0, 3.0, "2024-25") == pytest.approx(110.0 - 3.0 * r)
    assert RC.reserve_target(110.0, -1.0, "2024-25") == 110.0
    assert np.isnan(RC.reserve_target(float("nan"), 3.0, "2024-25"))


def _big_reserve(mp):
    mp = M.MinutesParams.from_json(mp.to_json())
    mp.reserve_coef = [10.0] + [0.0] * (len(mp.reserve_coef) - 1)
    return mp


@pytest.mark.parametrize("v1", [False, True])
def test_simulate_off_unchanged_on_moves_only_the_target(game, restore, v1):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    mp = _big_reserve(mp)
    for side in ("home", "away"):             # the scenario engine with a 10-minute reserve
        getattr(spec, side).minutes = C.ScenarioMinutes(game[1][side][1], mp)
    cp = C.default_copula_params("2024-25")

    def run():
        if v1:
            return C.simulate_game_v1(spec, cp, C.default_copula_v1_params(cp=cp), mp, sp,
                                      3000, 11)
        return C.simulate_game(spec, cp, mp, sp, 3000, 11)

    P.set_recon_reserve(False)
    a = run()
    b = run()
    assert np.array_equal(a.fp, b.fp)
    for t, info in a.teams.items():           # OFF: the team info has no reserve keys
        assert not {"reserve_min", "reserve_pts", "implied"} & set(info)
    P.set_recon_reserve(True)
    c = run()
    assert np.array_equal(a.minutes, c.minutes)                      # same draws
    assert np.array_equal(a.margin, c.margin) and np.array_equal(a.total, c.total)
    assert not np.array_equal(a.fp, c.fp)
    ppm = RC.bench_ppm("2024-25")
    for side in (spec.home, spec.away):
        ia, ic = a.teams[side.team], c.teams[side.team]
        sel = c.team == side.team
        # E[reserve] = the minutes the candidates do not get (water-filling fills to the total)
        n_ot = (np.abs(c.margin) < C.OT_BAND).astype(float)
        gap = (M.TEAM_MIN + M.OT_TEAM_MIN * n_ot - c.minutes[:, sel].sum(1)).mean()
        assert ic["reserve_min"] == pytest.approx(gap, abs=0.05)
        assert ic["reserve_min"] == pytest.approx(10.0, abs=1.0)
        assert ic["target"] == pytest.approx(side.implied - ic["reserve_min"] * ppm)
        assert ic["reserve_pts"] == pytest.approx(ic["reserve_min"] * ppm)
        assert ia["target"] == pytest.approx(side.implied)
        assert ic["ok"] and ia["ok"]
        assert ic["baseline"] == pytest.approx(ia["baseline"])       # pre-reconciliation equal
        assert ic["scale"] < ia["scale"]


def test_on_with_zero_rate_draws_exactly_off(game, restore, monkeypatch):  # noqa: F811
    spec, mp, sp = _spec(game)
    cp = C.default_copula_params("2024-25")
    P.set_recon_reserve(False)
    a = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    monkeypatch.setattr(RC, "bench_ppm", lambda season: 0.0)
    P.set_recon_reserve(True)
    b = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    assert a.info.equals(b.info)
