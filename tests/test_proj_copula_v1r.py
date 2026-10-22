"""Copula v1 retune (v1r, `COURTSIDE_COPULA=v1r`, `+c1r`): the switch and versioning, the v1
params / tables staying exactly v1's without a shape map, the shape map (exact means, identity,
recovery of a known shape), the context-arm frame, the v1r fit and the pipeline end to end."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from courtside.proj import copula as C
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC
from courtside.proj import state as ST

from test_proj_copula import _spec, game  # noqa: F401  (module fixture)
from test_proj_copula_v1 import _frame


@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA)
    yield
    P.set_copula(state[2])
    P.set_context(state[0], state[1])


# ------------------------------------------------------------------------- switch, versions

def test_versions_and_flag(restore_switches, monkeypatch):
    base = P.MODEL_VERSION_BASE
    assert P._versions(False, False, "v1r")[:2] == (P._VS_BASE + "_c1r", base + "+c1r")
    assert P._versions(True, False, "v1r")[1] == P.CONTEXT_MODEL_VERSION + "+c1r"
    assert P._versions(True, False, "v1r", "v2")[1] == P.CONTEXT_MODEL_VERSION + "+c1r+p2"
    assert P._versions(True, False, "v1r", mf=True) == (P._VS_BASE + "_cx_mf_c1r",
                                                        P.CONTEXT_MODEL_VERSION + "+mf+c1r")
    # v1 and v0 are unchanged
    assert P._versions(True, False, "v1") == (P._VS_BASE + "_cx_c1",
                                              P.CONTEXT_MODEL_VERSION + "+c1")
    assert P._versions(True, False, "v0") == (P._VS_BASE + "_cx", P.CONTEXT_MODEL_VERSION)
    P.set_context(True, False)
    P.set_copula("v1r")
    assert P.COPULA == "v1r" and P._VS.endswith("_cx_c1r") and "+c1r" in P.MODEL_VERSION
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                 # common random numbers
    P.set_copula("v1")
    assert P.MODEL_VERSION.startswith(P.CONTEXT_MODEL_VERSION + "+c1") and "c1r" not in P._VS
    with pytest.raises(ValueError):
        P.set_copula("v2")
    for v, want in (("v1r", "v1r"), ("V1R", "v1r"), ("v1", "v1"), ("1", "v1"), ("0", "v0"),
                    ("", "v0")):
        monkeypatch.setenv("C1R_TEST_FLAG", v)
        assert P._copula_flag("C1R_TEST_FLAG") == want
    assert set(P.COPULA_V1_VERSIONS) == {"v1", "v1r"}


# ------------------------------------------------------------------------- v1 unchanged

def test_v1_params_json_and_tables_unchanged_without_map():
    v1 = C.default_copula_v1_params("2024-25")
    d = asdict(v1)
    d.pop("pit_map")
    assert v1.to_json() == json.dumps(d, sort_keys=True)          # the v1 format, byte for byte
    assert "pit_map" not in v1.to_json()
    assert C.CopulaV1Params.from_json(v1.to_json()) == v1
    T = C.CountTables(v1)
    assert not T.mono                                              # grid placement, as v1
    m = np.array([0.0, 0.3, 7.77, 25.0, 999.0])
    assert np.array_equal(T._pos("pts", m, C.N_MEAN - 1),
                          np.clip(m / T.step["pts"], 0.0, C.N_MEAN - 1))
    v1.pit_map = {"pts": {"mu": [1.0], "u": [0.0, 1.0], "q": [[0.0, 1.0]]}}
    back = C.CopulaV1Params.from_json(v1.to_json())
    assert back == v1 and "pit_map" in v1.to_json()


def _identity_map(stats=C.COMP_STATS, knots=41):
    u = np.linspace(0, 1, knots).tolist()
    return {s: {"mu": [1.0, 10.0], "u": u, "q": [u, u]} for s in stats}


def test_identity_map_keeps_the_nb_tables():
    v1 = C.default_copula_v1_params()
    T0 = C.CountTables(v1)
    v1.pit_map = _identity_map()
    T1 = C.CountTables(v1)
    for s in C.COMP_STATS:
        assert np.array_equal(T0.tab[s], T1.tab[s])
    assert C.count_tables(v1) is not C.count_tables(C.default_copula_v1_params())


def test_shape_map_keeps_the_mean_exactly():
    u = np.linspace(0, 1, 41)
    q = 0.5 + 0.7 * (u - 0.5)                                    # a narrower shape
    v1 = C.default_copula_v1_params()
    v1.pit_map = {"pts": {"mu": [1.0, 20.0], "u": u.tolist(), "q": [q.tolist(), q.tolist()]}}
    T = C.CountTables(v1)
    T0 = C.CountTables(C.default_copula_v1_params())
    for mm in (0.4, 3.0, 11.3, 27.9):
        x = T._pos("pts", mm, C.N_MEAN - 1)
        lo = int(np.floor(x))
        e = T.mean_row["pts"][lo] * (1 - (x - lo)) + T.mean_row["pts"][lo + 1] * (x - lo)
        assert e == pytest.approx(mm, abs=1e-6)
    # the map changes the shape, not the mean
    rng = np.random.default_rng(1)
    n = 200_000
    m = np.full(n, 15.0)
    a = T.draw("pts", m, rng.random(n), rng.random(n))
    b = T0.draw("pts", m, rng.random(n), rng.random(n))
    assert a.mean() == pytest.approx(15.0, abs=0.06) and b.mean() == pytest.approx(15.0, abs=0.06)
    assert a.var() < 0.8 * b.var()
    assert (T.h("pts", [15.0]) > 0).all() and (T.var("pts", [15.0]) > 0).all()


def test_nb_pit_uniform_under_the_model():
    rng = np.random.default_rng(2)
    n = 100_000
    m = rng.uniform(0.5, 20, n)
    c2 = np.full(n, 0.08)
    r = 1 / c2
    y = rng.negative_binomial(r, r / (r + m))
    u = C.nb_pit(y, m, c2, rng.random(n))
    h = np.histogram(u, bins=10, range=(0, 1))[0] / n
    assert np.abs(h - 0.1).max() < 0.006


def test_fit_pit_map_recovers_a_known_shape():
    """Counts drawn from a skew-mapped table: the NB tables under-cover them, the fitted shape
    map brings the 80 % coverage back (exact `_calib_check`)."""
    f = _frame(n=60_000, seed=4)
    v1 = C.default_copula_v1_params()
    for s in C.COMP_STATS:
        v1.disp_mu[s] = [1.0, 10.0]
        v1.disp_c2[s] = [0.05, 0.05]
    u = np.linspace(0, 1, 41)
    q = u - 0.08 * np.sin(2 * np.pi * u)                         # wider tails than the NB
    truth = C.CopulaV1Params.from_json(v1.to_json())
    truth.pit_map = {"pts": {"mu": [1.0], "u": u.tolist(), "q": [q.tolist()]}}
    Tt = C.CountTables(truth)
    rng = np.random.default_rng(5)
    m = f["m_pts"].to_numpy()
    f["y_pts"] = Tt.draw("pts", m, rng.random(len(m)), rng.random(len(m))).astype(float)
    before = C._calib_check(f, C.CountTables(v1))["pts"]["cov80"]
    v1.pit_map = C.fit_pit_map(f, v1)
    after = C._calib_check(f, C.CountTables(v1))["pts"]["cov80"]
    assert before < 0.785 and abs(after - 0.80) < abs(before - 0.80) / 2
    pm = v1.pit_map["pts"]
    qq = np.asarray(pm["q"])
    assert (np.diff(qq, axis=1) >= 0).all() and qq[:, 0].max() == 0 and qq[:, -1].min() == 1
    # stats drawn from the NB itself get a map close to the identity
    assert np.abs(np.asarray(v1.pit_map["reb"]["q"]) - np.asarray(v1.pit_map["reb"]["u"])
                  ).max() < 0.03


def test_fit_copula_v1r_on_a_frame(restore_switches):
    f = _frame(n=40_000, seed=6)
    v1 = C.fit_copula_v1r(None, "2022-23", ST.default_params(), C.default_copula_params(),
                          frame=f)
    assert v1.version == C.COPULA_V1R_VERSION
    assert not v1.wl_factor and not v1.diagnostics["wl"]     # no blowout minutes: no shift
    assert v1.train_seasons == ["2021-22"]
    assert set(v1.pit_map) == set(C.COMP_STATS)
    assert v1.diagnostics["rates"] == "eb"
    for s in C.COMP_STATS:
        assert v1.diagnostics["calib_map"][s]["cov80"] == pytest.approx(0.80, abs=0.02)
    back = C.CopulaV1Params.from_json(v1.to_json())
    assert back == v1
    mp = P.M.default_params("2022-23")
    mp.blowout = {"version": 1}                    # minutes that move with the margin
    v1b = C.fit_copula_v1r(None, "2022-23", ST.default_params(), C.default_copula_params(),
                           mp=mp, frame=f)
    assert v1b.wl_factor and v1b.diagnostics["wl"]
    lose = [k for k, c in enumerate(v1b.wl_centres) if c < -12]
    assert all(v1b.wl_factor["ast"]["role"][k] < 0.93 for k in lose)
    bad = f.assign(season="2022-23")
    with pytest.raises(ValueError):
        C.fit_copula_v1r(None, "2022-23", ST.default_params(), C.default_copula_params(),
                         frame=bad)


def test_simulate_v1_with_a_v1r_map(game):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    v1 = C.default_copula_v1_params(cp=cp)
    v1r = C.CopulaV1Params.from_json(v1.to_json())
    u = np.linspace(0, 1, 41)
    q = np.clip(u + 0.05 * np.sin(np.pi * u), 0, 1)
    v1r.pit_map = {"pts": {"mu": [1.0, 20.0], "u": u.tolist(), "q": [q.tolist(), q.tolist()]}}
    v1r.version = C.COPULA_V1R_VERSION
    a = C.simulate_game_v1(spec, cp, v1, mp, sp, 4000, 11)
    b = C.simulate_game_v1(spec, cp, v1r, mp, sp, 4000, 11)
    assert np.array_equal(a.minutes, b.minutes) and np.array_equal(a.margin, b.margin)
    # the pts map changes the pts draws (and, through the DK loadings, the latent normals of
    # the other stats), not the means
    assert not np.array_equal(a.stats["pts"], b.stats["pts"])
    for s in C.COMP_STATS:
        ma, mb = a.stats[s].mean(0), b.stats[s].mean(0)
        assert np.abs(ma - mb).max() < 0.08 * max(ma.max(), 1.0)
    home = b.info["team"].to_numpy() == spec.home.team
    assert b.info.loc[home, "pts_all"].sum() == pytest.approx(112.0, abs=0.8)
    for info in b.teams.values():
        assert abs(info["resid"]) <= RC.TOL


# ------------------------------------------------------------------------- pipeline

def test_pipeline_v1r_end_to_end(restore_switches, tmp_path, monkeypatch):
    import sqlite3

    from courtside import db
    from test_proj_pipeline import SEASON, _fill, _late_day
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    _fill(conn)
    b = P.Bundle(season=SEASON, minutes=P.M.default_params(SEASON), state=ST.default_params(),
                 copula=C.default_copula_params(SEASON))
    ctx = P.Context(conn, SEASON)
    day = _late_day(conn)
    P.set_copula("v1")
    b.copula_v1 = C.default_copula_v1_params(SEASON, b.copula)
    r1 = P.project_historical(conn, day, n_sims=1500, ctx=ctx, bundle=b)
    P.set_copula("v1r")
    v1r = C.CopulaV1Params.from_json(b.copula_v1.to_json())
    v1r.pit_map = _identity_map()
    v1r.version = C.COPULA_V1R_VERSION
    b2 = P.Bundle.from_json(b.to_json())
    b2.copula_v1 = v1r
    r2 = P.project_historical(conn, day, n_sims=1500, ctx=ctx, bundle=b2)
    assert P.MODEL_VERSION.endswith("+c1r")
    assert all(g.stats is not None for g in r2.sims)
    # an identity map is v1 up to the placement of the rows by their means (and the Hermite
    # loadings that follow): same minutes, the same per-player means up to Monte Carlo
    for x, y in zip(r1.sims, r2.sims):
        assert np.array_equal(x.minutes, y.minutes)
        assert np.abs(x.stats["pts"].mean(0) - y.stats["pts"].mean(0)).max() < 0.6
        assert x.fp.mean(0).sum() == pytest.approx(y.fp.mean(0).sum(), rel=0.01)
    # the cached-bundle check: a v1 bundle on the c1r path is refit, a v1r one is used
    monkeypatch.setattr(P, "cache_dir", lambda: tmp_path)
    path = tmp_path / f"bundle_{SEASON}{P._VS}.json"
    path.write_text(b2.to_json())
    got = P.load_bundle(conn, SEASON, ctx)
    assert got.copula_v1 == v1r

