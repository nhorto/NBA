"""Minutes-ratio arm (COURTSIDE_MINRATIO, `proj/minutes_ratio.py`): the switch is off by default
and changes nothing when off; the two-factor table separates planned from unplanned minute
changes; the sims read each scenario's plan; versions / bundle suffixes compose."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from courtside.proj import copula as C
from courtside.proj import minutes_ratio as MR
from courtside.proj import pipeline as P
from courtside.proj import state as ST
from courtside.proj.features import store as FS

from test_proj_copula import _spec, game  # noqa: F401  (module fixture)


def _old_table(lo=0.8, hi=1.1):
    """An f(m/mpg) table that is steep enough to matter."""
    c = [0.3, 0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.7]
    return {"edges": list(ST.RATIO_EDGES), "centers": c,
            "mult": {s: list(np.linspace(lo, hi, len(c))) for s in ST.STATS}}


def _mr_table(fu=None, gp=None):
    cu = [0.3, 0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.7]
    cp = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7]
    fu = fu if fu is not None else [1.0] * len(cu)
    gp = gp if gp is not None else [1.0] * len(cp)
    return {"version": 1, "centers_u": cu, "centers_p": cp,
            "mult_u": {s: list(fu) for s in ST.STATS}, "mult_p": {s: list(gp) for s in ST.STATS}}


# ------------------------------------------------------------------------- switch and versions

@pytest.fixture
def restore_switches():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO)
    yield
    P.set_copula(state[2])
    P.set_pplay(state[3])
    P.set_minratio(state[4])
    P.set_context(state[0], state[1])


def test_default_off_and_env_flag(monkeypatch):
    assert P.MINRATIO_DEFAULT is False
    monkeypatch.delenv(P.MINRATIO_ENV, raising=False)
    assert P._env_flag(P.MINRATIO_ENV, P.MINRATIO_DEFAULT) is False
    monkeypatch.setenv(P.MINRATIO_ENV, "1")
    assert P._env_flag(P.MINRATIO_ENV, P.MINRATIO_DEFAULT) is True
    monkeypatch.setenv(P.MINRATIO_ENV, "0")
    assert P._env_flag(P.MINRATIO_ENV, True) is False


def test_versions_compose(restore_switches):
    base = P.MODEL_VERSION_BASE
    assert P._versions(False, False, "v0", "v1", False) == P._versions(False, False, "v0")
    assert P._versions(False, False, "v0", "v1", True) == (P._VS_BASE + "_mr", base + "+mr")
    assert P._versions(True, False, "v0", "v2", True) == (P._VS_BASE + "_cx_mr",
                                                          P.CONTEXT_MODEL_VERSION + "+mr+p2")
    assert P._versions(True, False, "v1", "v1", True) == (P._VS_BASE + "_cx_mr_c1",
                                                          P.CONTEXT_MODEL_VERSION + "+mr+c1")
    P.set_context(True, False)
    P.set_pplay("v2")
    P.set_minratio(True)
    assert P.MODEL_VERSION == "pathA-v0.3+mr+p2" and P._VS.endswith("_cx_mr")
    assert P.SEED_VERSION == base                     # common random numbers with OFF
    P.set_minratio(False)
    assert P.MODEL_VERSION == "pathA-v0.3+p2" and P._VS.endswith("_cx")


# ------------------------------------------------------------------------- multiplier

def test_multiplier_off_identity():
    """Without a `minutes_adj_mr` table the `planned` argument is ignored bit for bit; with the
    table but no plan the old f(m/mpg) is used."""
    sp = ST.default_params()
    sp["minutes_adj"] = _old_table()
    rng = np.random.default_rng(0)
    m, mpg, pl = rng.uniform(0, 45, 500), rng.uniform(3, 38, 500), rng.uniform(1, 40, 500)
    for s in ("pts", "ast", "tov"):
        a = ST.minutes_multiplier(sp, s, m, mpg)
        assert np.array_equal(a, ST.minutes_multiplier(sp, s, m, mpg, planned=pl))
    assert not ST.has_mr(sp) and not ST.has_mr(None)
    sp["minutes_adj_mr"] = _mr_table(fu=np.linspace(0.7, 1.2, 9))
    assert ST.has_mr(sp)
    assert np.array_equal(ST.minutes_multiplier(sp, "pts", m, mpg),
                          ST.minutes_multiplier(dict(sp, minutes_adj_mr={}), "pts", m, mpg))
    assert not np.array_equal(ST.minutes_multiplier(sp, "pts", m, mpg),
                              ST.minutes_multiplier(sp, "pts", m, mpg, planned=pl))


def test_mr_multiplier_keys_on_plan():
    sp = ST.default_params()
    fu = [0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.12, 1.15]
    gp = [0.9, 0.95, 0.97, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05]
    sp["minutes_adj_mr"] = _mr_table(fu, gp)
    # a ramped returner: plan 0.5 x mpg, plays to plan -> f_u(1) x g_p(0.5) (no unplanned cut)
    got = ST.minutes_multiplier(sp, "pts", [15.0], [30.0], planned=[15.0])[0]
    assert got == pytest.approx(np.interp(1.0, sp["minutes_adj_mr"]["centers_u"], fu) * 0.9)
    # the same 15 minutes as an unplanned early exit (plan 30) -> f_u(0.5), g_p(1)
    got = ST.minutes_multiplier(sp, "pts", [15.0], [30.0], planned=[30.0])[0]
    assert got == pytest.approx(np.interp(0.5, sp["minutes_adj_mr"]["centers_u"], fu))
    # no plan (NaN / 0) -> plan = mpg
    a = ST.minutes_multiplier(sp, "pts", [20.0, 20.0], [25.0, 25.0], planned=[np.nan, 0.0])
    b = ST.minutes_multiplier(sp, "pts", [20.0], [25.0], planned=[25.0])
    assert np.allclose(a, b[0])
    assert ST.minutes_multiplier(dict(sp, minutes_adj_mr=dict(
        sp["minutes_adj_mr"], mult_u={}, mult_p={})), "pts", [10.0], [20.0], planned=[5.0])[0] == 1


def _synthetic_rows(n=120_000, seed=3):
    """Per-minute production depends ONLY on the unplanned deviation m / P (pulled early ->
    less per minute), never on the plan P / mpg (ramps and promotions play at their rate)."""
    rng = np.random.default_rng(seed)
    mpg = rng.uniform(8, 36, n)
    v = np.exp(rng.normal(0, 0.25, n))                    # planned change: ramps, promotions
    planned = np.clip(mpg * v, 2, 42)
    u = np.exp(rng.normal(0, 0.18, n))
    u = np.where(rng.random(n) < 0.08, u * rng.uniform(0.35, 0.85, n), u)   # early exits
    m = np.clip(planned * u, 0.5, 52)
    truth = np.interp(m / planned, [0.3, 0.6, 1.0, 1.4], [0.75, 0.88, 1.0, 1.04])
    r = rng.uniform(0.3, 0.8, n)
    rows = pd.DataFrame({"min": m, "planned": planned, "mpg": mpg, "r_pts": r,
                         "pts": rng.poisson(r * m * truth)})
    return rows, truth


def test_fit_table_separates_planned_from_unplanned():
    rows, _ = _synthetic_rows()
    t = MR.fit_table(rows, stats=("pts",), plan_factor=True)     # the research variant
    gp = np.asarray(t["mult_p"]["pts"])
    fu = np.asarray(t["mult_u"]["pts"])
    assert np.abs(gp - 1).max() < 0.03                   # the plan carries no per-minute change
    assert fu[1] < 0.93 and abs(np.interp(1.0, t["centers_u"], fu) - 1.0) < 0.02
    assert np.all(np.diff(fu[1:-1]) > -0.02)              # increasing in m / P
    # the old key (m / mpg) charges a planned ramp (plays 0.6 x mpg as planned) a penalty
    ratio = rows["min"] / rows["mpg"]
    b = np.searchsorted(ST.RATIO_EDGES, ratio, side="right")
    ramp = b == 2                                         # m / mpg in [0.55, 0.7)
    old = rows.loc[ramp, "pts"].sum() / (rows.loc[ramp, "r_pts"] * rows.loc[ramp, "min"]).sum()
    planned_ramp = ramp & (np.abs(rows["min"] / rows["planned"] - 1) < 0.1).to_numpy()
    real = (rows.loc[planned_ramp, "pts"].sum()
            / (rows.loc[planned_ramp, "r_pts"] * rows.loc[planned_ramp, "min"]).sum())
    sp = ST.default_params()
    sp["minutes_adj_mr"] = t
    mr = ST.minutes_multiplier(sp, "pts", rows.loc[planned_ramp, "min"].to_numpy(),
                               rows.loc[planned_ramp, "mpg"].to_numpy(),
                               planned=rows.loc[planned_ramp, "planned"].to_numpy())
    assert old < real - 0.03                              # f(m/mpg) under-projects the ramp
    assert abs(float(np.average(mr, weights=rows.loc[planned_ramp, "min"])) - real) < 0.02


def test_default_design_is_plain_rekey():
    """The switch runs design B: g_p = 1 (`mult_p` all ones), one pooled f_u = ratio of sums by
    m / P alone (no P tiers, version 1), and the cached-bundle guard tells the designs apart."""
    assert MR.PLAN_FACTOR is False and MR.PL_EDGES == ()
    rows, _ = _synthetic_rows(n=40_000, seed=5)
    t = MR.fit_table(rows, stats=("pts",))
    assert t["plan_factor"] is False
    assert t["version"] == 1 and "mult_up" not in t and "edges_pl" not in t    # design B
    assert all(x == 1.0 for x in t["mult_p"]["pts"])
    u = rows["min"].to_numpy() / np.maximum(rows["planned"].to_numpy(), 1.0)
    b = np.searchsorted(MR.U_EDGES, u, side="right")
    k = int(np.bincount(b).argmax())
    q = b == k
    want = rows["pts"].to_numpy()[q].sum() / (rows["r_pts"] * rows["min"]).to_numpy()[q].sum()
    assert t["mult_u"]["pts"][k] == pytest.approx(want)
    assert MR.table_matches(t)
    assert not MR.table_matches(MR.fit_table(rows, stats=("pts",), plan_factor=True))
    legacy = {k_: v for k_, v in t.items() if k_ != "plan_factor"}          # no field: inferred
    assert MR.table_matches(legacy)
    assert not MR.table_matches(None)


def _tiered_rows(n=160_000, seed=11):
    """Per-minute production by m / P whose steepness depends on the plan: bench plans (P < 12)
    lose little on short nights (spot minutes), rotation and starter plans a lot (early exits)."""
    rng = np.random.default_rng(seed)
    planned = rng.uniform(4, 38, n)
    u = np.exp(rng.normal(0, 0.3, n))
    u = np.where(rng.random(n) < 0.1, u * rng.uniform(0.3, 0.8, n), u)
    m = np.clip(planned * u, 0.5, 52)
    uu = m / planned
    flat = np.interp(uu, [0.3, 1.0, 1.6], [0.92, 1.0, 1.03])
    steep = np.interp(uu, [0.3, 1.0, 1.6], [0.65, 1.0, 1.10])
    truth = np.where(planned < 12, flat, steep)
    r = rng.uniform(0.3, 0.8, n)
    rows = pd.DataFrame({"min": m, "planned": planned, "mpg": planned * np.exp(
        rng.normal(0, 0.15, n)), "r_pts": r, "pts": rng.poisson(r * m * truth)})
    return rows, truth


def test_tiered_table_by_plan():
    """Design C (research option), version 2: f_u per tier of P (PL_TIERS_C), shrunk to the
    pooled f_u, read linearly in P between tier centres; recovers a flatter bench curve the
    pooled table cannot."""
    assert MR.TABLE_VERSION == 2
    rows, truth = _tiered_rows()
    t = MR.fit_table(rows, stats=("pts",), pl_edges=MR.PL_TIERS_C)
    assert t["version"] == 2 and t["edges_pl"] == list(MR.PL_TIERS_C)
    nt = len(MR.PL_TIERS_C) + 1
    assert len(t["centers_pl"]) == nt and len(t["mult_up"]["pts"]) == nt
    assert len(t["centers_up"][0]) == len(t["centers_u"])
    sp = ST.default_params()
    sp["minutes_adj_mr"] = t
    m, P, mpg = rows["min"].to_numpy(), rows["planned"].to_numpy(), rows["mpg"].to_numpy()
    f = ST.minutes_multiplier(sp, "pts", m, mpg, planned=P)
    pooled = dict(sp, minutes_adj_mr={k: v for k, v in t.items() if k != "mult_up"})
    f0 = ST.minutes_multiplier(pooled, "pts", m, mpg, planned=P)
    bench = (P < 8) & (m / P < 0.6)                # short nights on a bench plan (tier 0 only)
    starter = (P > 26) & (m / P < 0.6)
    y = rows["pts"].to_numpy()
    base = (rows["r_pts"] * rows["min"]).to_numpy()
    for q in (bench, starter):
        real = y[q].sum() / base[q].sum()
        tiered = float(np.average(f[q], weights=base[q]))
        assert abs(tiered - real) < 0.03
        assert abs(tiered - real) < abs(float(np.average(f0[q], weights=base[q])) - real)
    # continuous in P (linear between the tier centres), one value per tier at its centre
    pg = np.linspace(2, 45, 400)
    g = ST.minutes_multiplier(sp, "pts", 0.5 * pg, pg, planned=pg)
    assert np.abs(np.diff(g)).max() < 0.01
    c0 = t["centers_pl"][0]
    want = np.interp(0.5, t["centers_up"][0], t["mult_up"]["pts"][0])
    assert ST.minutes_multiplier(sp, "pts", [0.5 * c0], [c0], planned=[c0])[0] == \
        pytest.approx(want)
    # every tier equal to the pooled f_u (same centres) -> exactly the pooled multiplier
    flat_t = dict(t, centers_up=[t["centers_u"]] * nt, mult_up={"pts": [t["mult_u"]["pts"]] * nt})
    assert np.allclose(ST.minutes_multiplier(dict(sp, minutes_adj_mr=flat_t), "pts", m, mpg,
                                             planned=P), f0)


def test_tiered_shrinkage_and_guard(monkeypatch):
    """Sparse cells sit near the pooled value; no tiers = the version-1 table; a cached table
    of another design (pooled, other tiers) is refit."""
    rows, _ = _tiered_rows(n=30_000, seed=2)
    t = MR.fit_table(rows, stats=("pts",), pl_edges=MR.PL_TIERS_C)
    n_up = np.asarray(t["n_up"])
    fu = np.asarray(t["mult_u"]["pts"])
    ft = np.asarray(t["mult_up"]["pts"])
    assert n_up.sum() == t["n"]
    few = n_up < 30
    if few.any():
        assert np.abs(ft - fu[None, :])[few].max() < 0.05
    t1 = MR.fit_table(rows, stats=("pts",), pl_edges=())
    assert t1["version"] == 1 and "mult_up" not in t1
    assert t1["mult_u"] == t["mult_u"]                        # the pooled f_u is the same fit
    assert MR.table_matches(t1) and not MR.table_matches(t)  # the switch runs design B
    monkeypatch.setattr(MR, "PL_EDGES", MR.PL_TIERS_C)
    assert MR.table_matches(t) and not MR.table_matches(t1)
    assert not MR.table_matches(MR.fit_table(rows, stats=("pts",), pl_edges=(15.0,)))
    assert MR.fit_table(rows, stats=("pts",))["edges_pl"] == list(MR.PL_TIERS_C)


# ------------------------------------------------------------------------- sims

def test_team_minutes_plan_out_changes_no_draw(game):
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    ts = spec.home
    margin = np.linspace(-20, 20, 500)
    n_ot = np.zeros(500)
    args = [np.random.default_rng(5), np.random.default_rng(6)]
    x0, a0, i0 = C._team_minutes(ts, mp, margin, n_ot, *args)
    po: dict = {}
    args = [np.random.default_rng(5), np.random.default_rng(6)]
    x1, a1, i1 = C._team_minutes(ts, mp, margin, n_ot, *args, plan_out=po)
    assert np.array_equal(x0, x1) and np.array_equal(a0, a1) and i0 == i1
    assert set(np.unique(po["assign"])) == set(po["plan"])
    assert len(po["plan"]) >= 3                           # the two 50 % players -> scenarios


def test_neutral_mr_table_equals_off(game):
    """f_u = g_p = 1 and no f(m/mpg) table: the per-scenario path gives the OFF sims exactly."""
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    off = C.simulate_game(spec, cp, mp, sp, 2000, 7)
    sp_mr = dict(sp, minutes_adj_mr=_mr_table())
    on = C.simulate_game(spec, cp, mp, sp_mr, 2000, 7)
    assert np.array_equal(off.fp, on.fp) and np.array_equal(off.minutes, on.minutes)
    pd.testing.assert_frame_equal(off.info, on.info)


def test_sims_read_their_scenarios_plan(game):
    spec, mp, sp = _spec(game, p_half=[1, 3])
    pl = spec.home.players.reset_index(drop=True)
    sp_mr = dict(sp, minutes_adj_mr=_mr_table(fu=np.linspace(0.6, 1.3, 9)))
    plan = {0: np.full(len(pl), 20.0), 1: np.full(len(pl), 30.0)}
    assign = np.array([0, 1, 0, 1, 1])
    x = np.full((5, len(pl)), 24.0)
    role = np.zeros((5, len(pl)), int)
    got = C._PlanGrids(C._grids, pl, sp_mr, plan, assign).look("pts", role, x)
    g0 = C._grids(pl, sp_mr, planned=plan[0])
    g1 = C._grids(pl, sp_mr, planned=plan[1])
    assert np.allclose(got[0], C._lookup(g0["pts"], role[:1], x[:1])[0])
    assert np.allclose(got[1], C._lookup(g1["pts"], role[:1], x[:1])[0])
    # 24 min on a 20-min plan is above plan, on a 30-min plan below: more per minute on the first
    assert np.all(got[0] > got[1])
    # one player, by hand: rate x m x f_u(24 / 20)
    r = pl["r_pts"].to_numpy()[0]
    f = np.interp(24 / 20, sp_mr["minutes_adj_mr"]["centers_u"],
                  sp_mr["minutes_adj_mr"]["mult_u"]["pts"])
    assert got[0, 0] == pytest.approx(r * 24 * f)


def test_mr_sim_differs_from_old_table(game):
    """A steep f(m/mpg) vs the plan-keyed table: same draws (minutes), different means."""
    spec, mp, sp = _spec(game, p_out=[0])
    cp = C.default_copula_params()
    sp_old = dict(sp, minutes_adj=_old_table(0.7, 1.2))
    sp_mr = dict(sp_old, minutes_adj_mr=_mr_table())
    a = C.simulate_game(spec, cp, mp, sp_old, 1500, 3)
    b = C.simulate_game(spec, cp, mp, sp_mr, 1500, 3)
    assert np.array_equal(a.minutes, b.minutes)
    assert not np.allclose(a.info["mu_played"], b.info["mu_played"])


# ------------------------------------------------------------------------- offsets, frame key

def _env_db():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE closing_lines (game_id TEXT, home_spread_close REAL, "
              "total_close REAL)")
    c.execute("CREATE TABLE team_games (game_id TEXT, team TEXT, is_home INT, season TEXT)")
    c.executemany("INSERT INTO team_games VALUES (?,?,?,?)",
                  [("G1", "AAA", 1, "2022-23"), ("G1", "BBB", 0, "2022-23")])
    c.execute("INSERT INTO closing_lines VALUES ('G1', -4.0, 220.0)")
    return c


def test_adjust_offsets_rekeys_and_reconciles():
    from courtside.proj.features.common import ALL_STATS, ENV_ALPHA
    sp = ST.default_params()
    sp["minutes_adj"] = _old_table()
    sp["minutes_adj_mr"] = _mr_table(fu=np.linspace(0.8, 1.1, 9))
    n = 10
    fr = pd.DataFrame({"game_id": "G1", "team": ["AAA"] * 5 + ["BBB"] * 5,
                       "player_id": np.arange(n), "season": "2022-23",
                       "min": np.tile(np.linspace(14, 30, 5), 2), "mpg": np.full(n, 22.0),
                       "c_env": 1.0})
    for s in ALL_STATS:
        fr[f"E0_{s}"] = 1.0 * fr["min"]
        fr[f"E_{s}"] = fr[f"E0_{s}"]
    planned = pd.DataFrame({"game_id": "G1", "team": fr["team"], "player_id": fr["player_id"],
                            "planned": fr["min"].to_numpy() * 1.0})
    out = MR.adjust_offsets(_env_db(), fr, sp, planned)
    # every row played exactly to plan -> f_u(1) x g_p(P / mpg) over the old f(m / mpg)
    ratio = (ST.minutes_multiplier(sp, "pts", fr["min"], fr["mpg"], planned=fr["min"])
             / ST.minutes_multiplier(sp, "pts", fr["min"], fr["mpg"]))
    assert np.allclose(out["E0_pts"], fr["E0_pts"] * ratio)
    implied = {"AAA": (220 + 4) / 2, "BBB": (220 - 4) / 2}
    for team, d in out.groupby("team"):
        assert d["E_pts"].sum() == pytest.approx(implied[team])
        c = d["c_env"].iloc[0]
        assert np.allclose(d["E_ast"], d["E0_ast"] * c ** ENV_ALPHA)
    assert list(out.columns) == list(fr.columns) and len(out) == n


def test_frame_key_unchanged_off():
    sp = ST.default_params()
    blob = json.dumps({"v": FS.FRAME_VERSION, "season": "2024-25", "n": 6,
                       "sp": sp.get("fit_seasons"), "kappa": {s: sp["stats"][s]["kappa"]
                                                              for s in sp["stats"]},
                       "dvp": FS.opp_defense.DVP_PARAMS}, sort_keys=True)
    assert FS._key("2024-25", sp, 6, None) == hashlib.sha1(blob.encode()).hexdigest()[:10]
    sp_mr = dict(sp, minutes_adj_mr=_mr_table())
    assert FS._key("2024-25", sp_mr, 6, None) != FS._key("2024-25", sp, 6, None)


def test_plan_seasons():
    assert MR.plan_seasons("2024-25", 6) == ["2018-19", "2019-20", "2020-21", "2021-22",
                                             "2022-23", "2023-24"]
    assert MR.plan_seasons("2024-25", 0)[0] == "2019-20"
