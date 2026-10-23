"""Reserve extras (COURTSIDE_RESERVE_EXTRAS, `+rx`; `minutes.RESERVE_EXTRAS`).

OFF by default and nothing of it runs when off (no row flagged, same reserve, same draws); ON
the forced extras the reserve fit counted outside the candidates (`minutes.rx_rows`: roster ==
0, box history, not a regular) leave the reserve's input and their planned minutes come off the
reserve (floor 0); a panel without such rows (the avail oracle) is unchanged; the version gets
`+rx` before `+rr`, same bundle and seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC


@pytest.fixture
def restore():
    state = (M.RESERVE_EXTRAS, RC.RECON_RESERVE)
    yield
    P.set_recon_reserve(state[1])
    P.set_reserve_extras(state[0])


def _params(res0: float) -> M.MinutesParams:
    mp = M.default_params()
    mp.reserve_coef = [res0, 0.0, 0.0, 0.0]
    return mp


def _forced(panel: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """The team's panel with its two deepest bench rows turned into forced extras (roster 0,
    box history, below the regular line), as the DK pool / report puts them in live."""
    p = panel.copy().reset_index(drop=True)
    order = np.argsort(p["base_ew"].to_numpy(float), kind="stable")
    idx = [int(order[0]), int(order[1])]
    p.loc[idx, "roster"] = 0
    p.loc[idx, "base_ew"] = np.minimum(p.loc[idx, "base_ew"], 8.0)
    assert p.loc[idx, "last_date"].notna().all()
    return p, idx


def test_default_off_and_env_flag(monkeypatch):
    assert M.RESERVE_EXTRAS_DEFAULT is True                           # shipped 2026-10-23
    monkeypatch.setenv(M.RESERVE_EXTRAS_ENV, "1")
    assert M._rx_flag() is True
    monkeypatch.setenv(M.RESERVE_EXTRAS_ENV, "off")
    assert M._rx_flag() is False
    monkeypatch.delenv(M.RESERVE_EXTRAS_ENV)
    assert M._rx_flag() is M.RESERVE_EXTRAS_DEFAULT


def test_version_tag_composes(restore):
    P.set_recon_reserve(False)
    P.set_reserve_extras(False)
    off_vs, off_mv = P._VS, P.MODEL_VERSION
    assert "+rx" not in off_mv
    P.set_reserve_extras(True)
    assert M.RESERVE_EXTRAS
    assert P._VS == off_vs and P.MODEL_VERSION == off_mv + "+rx"     # model version only
    P.set_recon_reserve(True)
    assert P.MODEL_VERSION == off_mv + "+rx+rr"
    P.set_reserve_extras(False)
    assert P.MODEL_VERSION == off_mv + "+rr" and not M.RESERVE_EXTRAS
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                  # seeds unchanged


def test_rx_rows_flags_only_forced_non_regulars_with_history():
    p = pd.DataFrame({"roster": [1, 0, 0, 0, 1],
                      "last_date": [1.0, 2.0, np.nan, 3.0, 4.0],
                      "base_ew": [30.0, 6.0, 4.0, 25.0, 5.0]})
    # roster player / forced bench with history / no history / forced regular / roster bench
    assert M.rx_rows(p).tolist() == [False, True, False, False, False]
    assert M.rx_rows(p.drop(columns="roster")).tolist() == [False] * 5
    assert len(M.rx_rows(p.iloc[:0])) == 0


def test_off_flags_nothing_and_oracle_like_panel_is_unchanged(game, restore):  # noqa: F811
    h, teams, sp = game
    panel = teams["home"][1]
    forced, _ = _forced(panel)
    mp = _params(10.0)
    P.set_reserve_extras(False)
    base, _ = M.base_and_pplay(forced, mp, np.zeros(len(forced), bool))
    assert getattr(base, "rx_rows", None) is None
    off = C.ScenarioMinutes(forced, mp).project(np.zeros(len(forced), bool))
    assert off["reserve"].iloc[0] == pytest.approx(10.0)
    # a panel with no forced extras (the avail oracle: roster == 0 rows are played regulars)
    P.set_reserve_extras(False)
    a = C.ScenarioMinutes(panel, mp).project(np.zeros(len(panel), bool))
    P.set_reserve_extras(True)
    assert not M.rx_rows(panel).any()
    b = C.ScenarioMinutes(panel, mp).project(np.zeros(len(panel), bool))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("res0", [10.0, 0.8])
def test_on_subtracts_the_extras_plan_floor_zero(game, restore, res0):  # noqa: F811
    h, teams, sp = game
    forced, idx = _forced(teams["home"][1])
    mp = _params(res0)
    P.set_reserve_extras(True)
    absent = np.zeros(len(forced), bool)
    absent[idx[1]] = True                      # an absent extra is not subtracted
    base, pp = M.base_and_pplay(forced, mp, absent)
    rx = base.rx_rows
    assert rx.tolist() == [i in idx for i in range(len(forced))]
    p = forced
    ew = 0.5 * p["ew_f"].to_numpy(float) + 0.5 * p["ew_s"].to_numpy(float)
    args = (base, pp, absent, p["g1"].to_numpy(int), p["g2"].to_numpy(int),
            p["st10"].to_numpy(float), ew, M._team_target(p, mp), mp)
    mean, det = M.project_arrays(*args, detail=True)
    e = det["p"] * det["adj"]
    want = max(0.0, res0 - float(e[idx[0]]))
    assert e[idx[0]] > 0.5
    assert det["reserve"] == pytest.approx(want)
    # the candidates get 240 - the reserve
    assert float((det["p"] * mean).sum()) == pytest.approx(M.TEAM_MIN - want, abs=0.5)
    # OFF on the same inputs: the full fitted reserve
    P.set_reserve_extras(False)
    base0, pp0 = M.base_and_pplay(forced, mp, absent)
    _, det0 = M.project_arrays(base0, pp0, *args[2:], detail=True)
    assert det0["reserve"] == pytest.approx(res0)


def test_simulation_reads_the_rx_reserve(game, restore):  # noqa: F811
    spec, mp, sp = _spec(game)
    mp = _params(10.0)
    forced, idx = _forced(game[1]["home"][1])
    cp = C.default_copula_params("2024-25")

    def run(on):
        P.set_reserve_extras(on)
        spec.home.minutes = C.ScenarioMinutes(forced, mp)
        spec.away.minutes = C.ScenarioMinutes(game[1]["away"][1], mp)
        return C.simulate_game(spec, cp, mp, sp, 2000, 7)

    a, b, a2 = run(False), run(True), run(False)
    assert np.array_equal(a.fp, a2.fp) and np.array_equal(a.minutes, a2.minutes)
    assert np.array_equal(a.margin, b.margin)
    for side, moved in ((spec.home, True), (spec.away, False)):
        sel = a.team == side.team
        n_ot = (np.abs(a.margin) < C.OT_BAND).astype(float)
        gap = lambda r: float((M.TEAM_MIN + M.OT_TEAM_MIN * n_ot  # noqa: E731
                               - r.minutes[:, sel].sum(1)).mean())
        assert gap(a) == pytest.approx(10.0, abs=0.3)
        if moved:
            assert gap(b) < gap(a) - 0.5
        else:
            assert np.array_equal(a.minutes[:, sel], b.minutes[:, sel])


def test_oracle_input_counts_absent_non_regulars_as_available(game, restore):  # noqa: F811
    """The fit's oracle only knows regulars' absences: a listed-out bench player is an available
    candidate there, so ON keeps him in the reserve's input (sum p x base and the count)."""
    h, teams, sp = game
    p = teams["home"][1].reset_index(drop=True)
    mp = M.default_params()
    mp.reserve_coef = [1.0, 0.25, 0.0, 0.0]            # reserve = 1 + 0.25 (220 - s)+
    ew = p["base_ew"].to_numpy(float)
    bench = int(np.flatnonzero(ew < M.REGULAR_MIN)[0])
    reg = int(np.argmax(ew))
    assert ew[reg] >= M.REGULAR_MIN
    absent = np.zeros(len(p), bool)
    absent[[bench, reg]] = True
    P.set_reserve_extras(False)
    b0, pp0 = M.base_and_pplay(p, mp, absent)
    P.set_reserve_extras(True)
    b1, pp1 = M.base_and_pplay(p, mp, absent)
    assert np.array_equal(pp0, pp1) and np.array_equal(np.asarray(b0), np.asarray(b1))
    assert not b1.rx_rows.any()
    assert b1.rx_s[bench] > 0 and b1.rx_s[reg] == 0     # bench counted, the regular not
    assert b1.rx_n == len(p) - 1
    # the same as OFF with only the regular absent (what the oracle would have seen)
    only_reg = np.zeros(len(p), bool)
    only_reg[reg] = True
    P.set_reserve_extras(False)
    b2, pp2 = M.base_and_pplay(p, mp, only_reg)          # v0.5 params: s = sum p x base
    assert b1.rx_s.sum() == pytest.approx(float(np.sum(np.where(only_reg, 0.0, pp2 * b2))))


def test_pool_reserve_table_is_walk_forward():
    import json
    t = json.loads(M.RX_PATH.read_text())
    assert t["bins"] == list(M.DEBUT_BINS)
    for s, r in t["seasons"].items():
        assert r["train_seasons"] and all(x < s for x in r["train_seasons"])
        assert len(r["bins"]) == len(M.DEBUT_BINS)
        assert r["bins"][0] > 5.0 > r["bins"][1] > r["bins"][3] > 0.0   # openers: debutants
    assert M.pool_reserve("2024-25", 0) == pytest.approx(t["seasons"]["2024-25"]["bins"][0])
    assert M.pool_reserve("2024-25", 3) == pytest.approx(t["seasons"]["2024-25"]["bins"][1])
    assert M.pool_reserve("2024-25", 40) == pytest.approx(t["seasons"]["2024-25"]["bins"][3])
    last = max(t["seasons"])
    assert M.pool_reserve("2099-00", 40) == pytest.approx(t["seasons"][last]["bins"][3])


def test_dk_pool_candidates_reserve_only_the_debutants(game, restore):  # noqa: F811
    h, teams, sp = game
    p = teams["home"][1].reset_index(drop=True)
    mp = _params(10.0)
    absent = np.zeros(len(p), bool)
    r1 = p["roster"].to_numpy() == 1
    few = p.assign(forced=False)
    few.loc[np.flatnonzero(r1)[:2], "forced"] = True            # the report: a few listed
    pool = p.assign(forced=r1)                                   # the DK pool: the whole roster
    assert not M.rx_pool(few) and M.rx_pool(pool) and not M.rx_pool(p)
    P.set_reserve_extras(True)
    a = C.ScenarioMinutes(few, mp).project(absent)
    b = C.ScenarioMinutes(pool, mp).project(absent)
    want = M.pool_reserve(str(p["season"].iloc[0]), float(p["tgs"].iloc[0]))
    assert a["reserve"].iloc[0] == pytest.approx(10.0)
    assert b["reserve"].iloc[0] == pytest.approx(want)
    P.set_reserve_extras(False)
    c = C.ScenarioMinutes(pool, mp).project(absent)              # OFF: the column is ignored
    assert c["reserve"].iloc[0] == pytest.approx(10.0)
