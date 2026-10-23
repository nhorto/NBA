"""P(play) v2 path coverage beyond tests/test_proj_pplay_v2.py (E7): exact fringe values from
`pplay_v2_fix`, the packaged fringe params (range, monotonicity, clamp, file round trip), the
`+p2` version string composed with every other model-version tag, and v1 vs v2 on every caller:
they differ only on the live candidate paths (`project_games`, `backtest.live_like_day`); the
avail oracle (`historical_inputs` / `project_historical(protocol="avail")`) never reads the
switch. Small synthetic fixtures, no real DB (the suite conftest pins v1; tests flip it)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from courtside.proj import availability as AV
from courtside.proj import backtest as B
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC

from test_proj_backtest_live import _report, live  # noqa: F401  (module fixture)
from test_proj_pipeline import _copy, _late_day, bundle, conn, ctx  # noqa: F401


@pytest.fixture
def switches():
    """Snapshot every model-version switch; restore through the setters so MODEL_VERSION and
    the bundle suffix stay consistent for later tests."""
    s = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO, P.PRESEASON, P.MINFIX,
         P.BLOWOUT, AV.PPLAYQ, AV.INJ_GAMEDAY, M.RESERVE_EXTRAS, RC.RECON_RESERVE)
    yield
    P.set_context(s[0], s[1])
    P.set_copula(s[2])
    P.set_pplay(s[3])
    P.set_minratio(s[4])
    P.set_preseason(s[5])
    P.set_minfix(s[6])
    P.set_blowout(s[7])
    P.set_pplayq(s[8])
    P.set_inj_gameday(s[9])
    P.set_reserve_extras(s[10])
    P.set_recon_reserve(s[11])


def _all_off():
    P.set_context(False, False)
    P.set_copula("v0")
    P.set_pplay("v1")
    P.set_minratio(False)
    P.set_preseason(False)
    P.set_minfix(False)
    P.set_blowout(False)
    P.set_pplayq(False)
    P.set_inj_gameday(False)
    P.set_reserve_extras(False)
    P.set_recon_reserve(False)


# ------------------------------------------------------------------------- (e) version strings

def test_p2_composes_with_mf_and_the_model_version_only_tags(switches):
    _all_off()
    base = P.MODEL_VERSION_BASE
    assert P.MODEL_VERSION == base
    P.set_minfix(True)
    P.set_pplay("v2")
    assert P.MODEL_VERSION == f"{base}+mf+p2"                 # the shipped 2026-10-23 string
    vs_mf = P._VS
    assert vs_mf.endswith(P.MINFIX_SUFFIX)
    # +p2 / +pq / +gd / +rx / +rr are model-version tags only: the bundle suffix never moves
    P.set_pplayq(True)
    assert P.MODEL_VERSION == f"{base}+mf+p2+pq"
    P.set_inj_gameday(True)
    assert P.MODEL_VERSION == f"{base}+mf+p2+pq+gd"
    P.set_reserve_extras(True)
    P.set_recon_reserve(True)
    assert P.MODEL_VERSION == f"{base}+mf+p2+pq+gd+rx+rr"
    assert P._VS == vs_mf
    # dropping p2 removes only its tag; the others keep their order
    P.set_pplay("v1")
    assert P.MODEL_VERSION == f"{base}+mf+pq+gd+rx+rr"
    assert P._VS == vs_mf
    # seeds never depend on any of it
    assert P.SEED_VERSION == base


def test_each_tag_combination_matches_versions_and_is_order_independent(switches):
    _all_off()
    base = P.MODEL_VERSION_BASE
    tags = {"p2": lambda on: P.set_pplay("v2" if on else "v1"),
            "pq": P.set_pplayq, "gd": P.set_inj_gameday, "rr": P.set_recon_reserve}
    for names in (("p2", "pq"), ("p2", "gd"), ("p2", "rr"), ("pq", "p2"), ("rr", "gd", "p2"),
                  ("p2", "pq", "gd", "rr")):
        _all_off()
        P.set_minfix(True)
        for n in names:
            tags[n](True)
        want = base + "+mf" + "".join(f"+{n}" for n in ("p2", "pq", "gd", "rr") if n in names)
        assert P.MODEL_VERSION == want, names
        # the pure function agrees with the setter-maintained global
        assert P._versions(P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY, P.MINRATIO,
                           P.PRESEASON, P.MINFIX, P.BLOWOUT)[1] == want


def test_p2_composes_with_the_bundle_arms(switches):
    _all_off()
    P.set_minfix(True)
    P.set_pplay("v2")
    P.set_context(True, False)
    bundle_mv = P._versions_bundle(True, False, "v0", False, False, True, False)[1]
    assert P.MODEL_VERSION == bundle_mv + "+p2" and "+mf" in bundle_mv
    P.set_copula("v1")
    P.set_pplayq(True)
    bundle_mv = P._versions_bundle(True, False, "v1", False, False, True, False)[1]
    assert P.MODEL_VERSION == bundle_mv + "+p2+pq" and bundle_mv.endswith(P.COPULA_V1_TAG)
    # the explicit-argument form (no globals) is the same composition
    assert P._versions(True, False, "v1", "v2", False, False, True, False)[1] == P.MODEL_VERSION


# ------------------------------------------------------------------------- (a)/(b) fix values

def _team_panel(ctx, rosters, season="2022-23", date_=None):
    h = ctx.h
    last = h.tg[h.tg["season"] == season].sort_values("date").iloc[-1]
    team, day = last["team"], int(last["date"] if date_ is None else date_)
    pids = sorted(h.hist.loc[h.hist["team"] == team, "player_id"].unique())[:len(rosters)]
    return team, pd.DataFrame({"key": f"G|{team}", "player_id": pids, "team": team, "date": day,
                               "season": season, "roster": rosters})


def test_fringe_value_is_exactly_the_packaged_model(ctx):  # noqa: F811
    team, panel = _team_panel(ctx, [1, 0, 0, 0])
    sal = {int(panel["player_id"].iloc[1]): 3000, int(panel["player_id"].iloc[2]): 9000}
    fix = P.pplay_v2_fix(ctx, panel, {}, set(), salaries=sal)
    X = AV.fringe_features(panel.iloc[1:].reset_index(drop=True), ctx.h, sal)
    assert (X["regime"] == "mid").all()                    # mid-season: salary does not enter
    want = AV.load_fringe().p(X)
    got = [fix[(f"G|{team}", int(p))] for p in panel["player_id"].iloc[1:]]
    assert np.allclose(got, want, atol=0, rtol=0)
    assert all(AV.FRINGE_CLAMP[0] <= g <= AV.FRINGE_CLAMP[1] for g in got)
    assert (f"G|{team}", int(panel["player_id"].iloc[0])) not in fix     # rotation: the engine


def test_fix_rules_listed_reported_unreported_and_missing_roster(ctx):  # noqa: F811
    team, panel = _team_panel(ctx, [0, 0, 0, 0])
    p = [int(x) for x in panel["player_id"]]
    key = lambda i: (f"G|{team}", p[i])  # noqa: E731
    # 0: listed and reported -> the report cell is the total; 1: reported (e.g. in p_avail) but not
    # in `listed` and not a fringe row -> untouched; 2: out -> untouched; 3: unreported fringe
    fix = P.pplay_v2_fix(ctx, panel, {p[0]: 0.4, p[1]: 0.7, p[2]: 0.0}, {p[0]})
    assert fix[key(0)] == 1.0
    assert key(1) not in fix and key(2) not in fix
    assert 0.0 < fix[key(3)] < 1.0
    # a panel with no roster column is all rotation: never fringe
    fix = P.pplay_v2_fix(ctx, panel.drop(columns="roster"), {}, {p[0]})
    assert fix == {key(0): 1.0}
    # ids may arrive as numpy ints / strings-free ints in the sets
    fix = P.pplay_v2_fix(ctx, panel, {np.int64(p[0]): 0.4}, {np.int64(p[0])}, {np.int64(p[3])})
    assert fix[key(0)] == 1.0 and fix[key(3)] == AV.FRINGE_STARTER_P


def test_opener_regimes_and_starter_override(ctx):  # noqa: F811
    h = ctx.h
    first = int(h.tg.loc[h.tg["season"] == "2022-23", "date"].min())
    team, panel = _team_panel(ctx, [0, 0, 0, 0], date_=first)
    p = [int(x) for x in panel["player_id"]]
    key = lambda i: (f"G|{team}", p[i])  # noqa: E731
    sal = {p[0]: 3000, p[1]: AV.EARLY_ENGINE_SALARY, p[2]: AV.EARLY_ENGINE_SALARY + 3000,
           p[3]: 3000}
    fix = P.pplay_v2_fix(ctx, panel, {}, set(), starters={p[3], p[2]}, salaries=sal)
    X = AV.fringe_features(panel, h, sal)
    assert list(X["regime"]) == ["early"] * 4
    want = AV.load_fringe().p(X)
    assert fix[key(0)] == pytest.approx(want[0])            # min-priced: the early model
    assert key(1) not in fix                                 # priced at the cutoff: engine value
    assert fix[key(2)] == AV.FRINGE_STARTER_P                # priced AND a DK starter: 0.99
    assert fix[key(3)] == AV.FRINGE_STARTER_P                # min-priced starter: 0.99
    # without salaries every opener row is "early_nosal": all get the model value
    fx = P.pplay_v2_fix(ctx, panel, {}, set())
    Xn = AV.fringe_features(panel, h, None)
    assert set(Xn["regime"]) == {"early_nosal"}
    assert np.allclose([fx[key(i)] for i in range(4)], AV.load_fringe().p(Xn))


# ------------------------------------------------------------------------- (c) packaged params

def _rows(regime: str, n: int = 1, **over) -> pd.DataFrame:
    d = {"one": 1.0, "avg10": 15.0, "app365": 30.0, "salary": 5000.0, "l_app_season0": 0.0}
    d.update(over)
    r = pd.DataFrame({k: [v] * n if np.isscalar(v) else list(v) for k, v in d.items()})
    r["l_app365"] = np.log1p(r["app365"])
    r["l_sal"] = np.log(np.maximum(r["salary"], 3000.0) / 3000.0)
    r["min_sal"] = (r["salary"] <= 3000).astype(float)
    r["regime"] = regime
    return r


def test_packaged_fringe_file_loads_with_the_declared_features():
    assert AV.FRINGE_PATH.exists() and AV.FRINGE_PATH.parent.name == "proj"
    raw = json.loads(AV.FRINGE_PATH.read_text())
    fp = AV.load_fringe()
    assert set(fp.models) == set(AV.FRINGE_FEATURES) == set(raw["models"])
    for name, m in fp.models.items():
        assert tuple(m["features"]) == AV.FRINGE_FEATURES[name]
        assert len(m["coef"]) == len(m["features"]) and np.isfinite(m["coef"]).all()
        assert m["features"][0] == "one"
    assert AV.load_fringe() is fp                              # cached per path
    # every model column exists in fringe_features' output
    cols = set(_rows("mid").columns)
    assert all(set(m["features"]) <= cols for m in fp.models.values())


@pytest.mark.parametrize("regime", ["mid", "early", "early_nosal"])
def test_packaged_fringe_p_is_a_probability_over_a_grid(regime):
    fp = AV.load_fringe()
    g = pd.DataFrame([(a, n, s, z) for a in (0.0, 5.0, 12.0, 25.0, 38.0)
                      for n in (0, 3, 40, 82) for s in (3000.0, 3500.0, 4500.0, 9000.0)
                      for z in (0.0, 1.0)], columns=["avg10", "app365", "salary", "l_app_season0"])
    X = _rows(regime, len(g), avg10=g["avg10"], app365=g["app365"], salary=g["salary"],
              l_app_season0=g["l_app_season0"])
    p = fp.p(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert ((p >= AV.FRINGE_CLAMP[0]) & (p <= AV.FRINGE_CLAMP[1])).all()


def test_packaged_fringe_is_monotone_where_the_model_implies_it():
    fp = AV.load_fringe()
    mono = lambda r, **ax: np.diff(fp.p(_rows(r, len(next(iter(ax.values()))), **ax)))  # noqa: E731
    mins = np.linspace(0, 40, 41)
    # minutes level up -> P(plays) up (mid and no-salary opener models)
    assert (mono("mid", avg10=mins) >= -1e-12).all() and mono("mid", avg10=mins).sum() > 0.05
    assert (mono("early_nosal", avg10=mins) >= -1e-12).all()
    # more appearances in the last year -> up (openers)
    apps = np.arange(0, 83)
    assert (mono("early", app365=apps) >= -1e-12).all()
    assert (mono("early_nosal", app365=apps) >= -1e-12).all() \
        and mono("early_nosal", app365=apps).sum() > 0.1
    # a higher DK price -> up, including across the $3,000 minimum-salary step
    sal = np.arange(3000, 12001, 250, dtype=float)
    d = mono("early", salary=sal)
    assert (d >= -1e-12).all() and d.sum() > 0.2
    # the shipped sign facts the calibration note relies on
    assert fp.p(_rows("early", salary=3000.0))[0] < fp.p(_rows("early", salary=6000.0))[0]


def test_fringe_p_clamps_extremes_and_roundtrips_through_a_file(tmp_path):
    fp = AV.FringeParams(models={"mid": {"features": ["one", "avg10"], "coef": [-50.0, 0.0]},
                                 "early": {"features": ["one"], "coef": [50.0]}})
    X = pd.DataFrame({"one": [1.0, 1.0], "avg10": [10.0, 10.0], "regime": ["mid", "early"]})
    assert list(fp.p(X)) == [AV.FRINGE_CLAMP[0], AV.FRINGE_CLAMP[1]]
    assert np.isnan(fp.p(X.assign(regime="early_nosal"))).all()    # no model for the regime
    path = fp.save(tmp_path / "fringe.json")
    back = AV.load_fringe(path)
    assert back.models == fp.models and np.array_equal(back.p(X), fp.p(X))
    assert AV.load_fringe(path) is back                              # cache by path


def test_fringe_model_refit_reproduces_the_features_it_declares():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"one": 1.0, "avg10": rng.uniform(0, 30, n),
                       "l_app_season0": rng.integers(0, 2, n).astype(float)})
    z = -1.5 + 0.08 * df["avg10"] + 0.9 * df["l_app_season0"]
    df["played"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    m = AV.fit_fringe_model(df, "mid", ridge=0.1)
    assert m["features"] == list(AV.FRINGE_FEATURES["mid"]) and len(m["coef"]) == 3
    assert m["coef"][1] > 0 and m["coef"][2] > 0
    X = df.assign(regime="mid")
    p = AV.FringeParams(models={"mid": m}).p(X)
    assert abs(p.mean() - df["played"].mean()) < 0.02


# ------------------------------------------------------------------------- (d) v1 vs v2

def _bench_candidate(conn, ctx, day):  # noqa: F811
    """(games, panel, lowest-minutes rotation candidate of the first game)."""
    games = P._home_away(conn, day)
    panel = P.candidate_panel(ctx, day, games, None)
    g = games.iloc[0]["game_id"]
    rows = [(int(r[0]), r[1]) for r in conn.execute(
        "SELECT player_id, team FROM player_games WHERE game_id=? AND min>0 ORDER BY min", (g,))]
    in_panel = set(zip(panel["player_id"].astype(int), panel["team"]))
    pid = next(p for p, t in rows if (p, t) in in_panel)
    return games, panel, pid


def test_project_games_v1_vs_v2_differ_only_through_the_fix(conn, ctx, bundle, switches):  # noqa: F811
    day = _late_day(conn)
    games, panel, q = _bench_candidate(conn, ctx, day)
    assert (panel["roster"] == 1).all()                      # no fringe rows without extras
    kw = dict(ctx=ctx, bundle=bundle, n_sims=4000, seed=5, env_mode="closing")
    P.set_pplay("v1")
    a1 = P.project_games(conn, day, games, **kw)
    r1 = P.project_games(conn, day, games, p_avail={q: 0.5}, listed={q}, **kw)
    P.set_pplay("v2")
    a2 = P.project_games(conn, day, games, **kw)
    r2 = P.project_games(conn, day, games, p_avail={q: 0.5}, listed={q}, **kw)
    # no report, no fringe: v2 has nothing to fix -> bit-identical sims
    for s1, s2 in zip(a1.sims, a2.sims):
        assert np.array_equal(s1.fp, s2.fp) and np.array_equal(s1.minutes, s2.minutes)
    # a listed Questionable player: v2 plays him at the report cell, v1 double counts
    p1 = r1.players.set_index("player_id")["p_play"]
    p2 = r2.players.set_index("player_id")["p_play"]
    assert p2[q] == pytest.approx(0.5, abs=0.03)
    assert p1[q] < p2[q] - 0.01
    # not listed -> v2 leaves it to the engine even when he is in p_avail
    r3 = P.project_games(conn, day, games, p_avail={q: 0.5}, listed=set(), **kw)
    assert r3.players.set_index("player_id")["p_play"][q] == pytest.approx(p1[q], abs=0.03)


def test_project_games_v2_forced_pool_player_gets_the_fringe_value(conn, ctx, bundle,  # noqa: F811
                                                                  switches, monkeypatch):
    day = _late_day(conn)
    games = P._home_away(conn, day)
    base = P.candidate_panel(ctx, day, games, None)
    g = games.iloc[0]
    team = g["home"]
    assert len(games) >= 2
    # a player of another game's team, forced onto this team (a trade / DK pool mover): box
    # history, but not in this team's last 10 games -> roster 0 and not on the report
    mover = base[base["game_id"] == games.iloc[1]["game_id"]]
    extra = int(mover["player_id"].iloc[0])
    seen = {}
    real = P.pplay_v2_fix

    def spy(c, panel, *a, **k):
        out = real(c, panel, *a, **k)
        seen["panel"], seen["fix"] = panel.copy(), out
        return out
    monkeypatch.setattr(P, "pplay_v2_fix", spy)
    P.set_pplay("v2")
    P.project_games(conn, day, games, ctx=ctx, bundle=bundle, n_sims=500, seed=1,
                    env_mode="closing", extras={(g["game_id"], team): [extra]})
    pan, fix = seen["panel"], seen["fix"]
    row = pan[pan["player_id"] == extra]
    assert len(row) == 1 and int(row["roster"].iloc[0]) == 0
    X = AV.fringe_features(row.reset_index(drop=True), ctx.h, None)
    assert fix[(row["key"].iloc[0], extra)] == pytest.approx(float(AV.load_fringe().p(X)[0]))


def test_avail_oracle_is_identical_under_v1_and_v2(conn, ctx, bundle, switches, monkeypatch):  # noqa: F811
    day = _late_day(conn)

    def boom(*a, **k):
        raise AssertionError("pplay_v2_fix reached from the avail oracle")
    monkeypatch.setattr(P, "pplay_v2_fix", boom)
    out = {}
    for v in ("v1", "v2"):
        P.set_pplay(v)
        out[v] = P.project_historical(conn, day, protocol="avail", n_sims=1500, seed=11,
                                      ctx=ctx, bundle=bundle)
    for s1, s2 in zip(out["v1"].sims, out["v2"].sims):
        assert np.array_equal(s1.fp, s2.fp) and np.array_equal(s1.minutes, s2.minutes)
    cols = ["game_id", "player_id", "p_play", "dk_mean", "min_mean"]
    pd.testing.assert_frame_equal(out["v1"].players[cols], out["v2"].players[cols],
                                  check_exact=True)


def test_reports_protocol_is_a_live_candidate_path(conn, ctx, bundle, switches, monkeypatch):  # noqa: F811
    """project_historical(protocol="reports") goes through project_games, so v2 applies there
    (with the report's listed players); v1 never calls the fix."""
    day = _late_day(conn)
    rep = _report(conn, day)
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    calls = []
    real = P.pplay_v2_fix

    def spy(c, panel, p_avail, listed, *a, **k):
        calls.append((set(p_avail or {}), set(listed or ())))
        return real(c, panel, p_avail, listed, *a, **k)
    monkeypatch.setattr(P, "pplay_v2_fix", spy)
    P.set_pplay("v1")
    r1 = P.project_historical(conn, day, protocol="reports", n_sims=2500, seed=2, ctx=ctx,
                              bundle=bundle)
    assert calls == []
    P.set_pplay("v2")
    r2 = P.project_historical(conn, day, protocol="reports", n_sims=2500, seed=2, ctx=ctx,
                              bundle=bundle)
    assert len(calls) == 1
    out_pid, q_pid = int(rep["player_id"][0]), int(rep["player_id"][1])
    pa, listed = calls[0]
    assert {out_pid, q_pid} <= pa and listed == {q_pid}
    p1 = r1.players.set_index("player_id")["p_play"]
    p2 = r2.players.set_index("player_id")["p_play"]
    assert p2[q_pid] == pytest.approx(0.55, abs=0.04) and p1[q_pid] < p2[q_pid]
    assert p1[out_pid] == p2[out_pid] == 0.0


def test_live_like_day_v1_vs_v2(conn, ctx, bundle, live, switches, monkeypatch):  # noqa: F811
    day = _late_day(conn)
    rep = _report(conn, day)
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    calls = []
    real = P.pplay_v2_fix

    def spy(c, panel, p_avail, listed, starters=None, salaries=None):
        calls.append((set(listed or ()), salaries))
        return real(c, panel, p_avail, listed, starters, salaries)
    monkeypatch.setattr(P, "pplay_v2_fix", spy)
    P.set_pplay("v1")
    r1 = B.live_like_day(conn, ctx, bundle, live, day, offset_min=45, n_sims=2500, seed=9)
    assert calls == []
    P.set_pplay("v2")
    r2 = B.live_like_day(conn, ctx, bundle, live, day, offset_min=45, n_sims=2500, seed=9)
    assert len(calls) == r2.meta["groups"] == 1
    out_pid, q_pid = int(rep["player_id"][0]), int(rep["player_id"][1])
    assert calls[0][0] == {q_pid} and calls[0][1] is None      # listed, not Out; no salaries
    p1 = r1.players.set_index("player_id")["p_play"]
    p2 = r2.players.set_index("player_id")["p_play"]
    assert p2[q_pid] == pytest.approx(0.55, abs=0.04) and p1[q_pid] < p2[q_pid]
    assert p1[out_pid] == p2[out_pid] == 0.0
    assert not np.array_equal(r1.sims[0].fp, r2.sims[0].fp)


# ------------------------------------------------------------------------- salary routing

def _add_slate(c, day, sid, game_type_id, salary, captured="2023-01-01T00:00:00Z"):
    g = P._home_away(c, day).iloc[0]
    pl = [r for r in c.execute("SELECT player_id, team FROM player_games WHERE game_id=?",
                               (g["game_id"],))]
    with c:
        c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json) "
                  "VALUES (?, ?, ?, 'main', ?)", (sid, day.isoformat(), game_type_id,
                                                  f'["{g["game_id"]}"]'))
        for k, (pid, team) in enumerate(pl):
            c.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                      "game_type_id, draftable_id, player_name, site_player_id, team, positions,"
                      " roster_slot, salary, game_id, home, away, start_time, disabled) VALUES "
                      "(?, 'dk', ?, 'x', ?, ?, ?, ?, ?, 'SF', 'UTIL', ?, ?, ?, ?, "
                      "'2023-01-01T00:00:00Z', 0)",
                      (captured, sid, game_type_id, f"{sid}-{k}", f"P{pid}", f"dk{pid}", team,
                       salary, g["game_id"], g["home"], g["away"]))
            c.execute("INSERT OR IGNORE INTO player_ids (source, source_id, player_id) VALUES "
                      "('dk', ?, ?)", (f"dk{pid}", int(pid)))
    return [int(p) for p, _ in pl]


def test_classic_salaries_reads_only_classic_slates_latest_capture(conn):  # noqa: F811
    c = _copy(conn)
    day = _late_day(c)
    assert P.classic_salaries(c, day) == {}
    pids = _add_slate(c, day, "c1", P.CLASSIC_GAME_TYPE, 4000)
    _add_slate(c, day, "sd", 81, 11_000)                       # Showdown prices: ignored
    assert set(P.classic_salaries(c, day).values()) == {4000}
    _add_slate(c, day, "c2", P.CLASSIC_GAME_TYPE, 5200, captured="2023-01-02T00:00:00Z")
    got = P.classic_salaries(c, day)
    assert set(got) == set(pids) and set(got.values()) == {5200}      # latest capture wins
    assert P.classic_salaries(c, day + timedelta(days=30)) == {}


def test_project_slate_passes_salaries_to_v2_only(conn, ctx, bundle, switches, monkeypatch):  # noqa: F811
    c = _copy(conn)
    day = _late_day(c)
    _add_slate(c, day, "cl", P.CLASSIC_GAME_TYPE, 4000)
    _add_slate(c, day, "sd", 81, 11_000)
    seen = {}

    class Stop(Exception):
        pass

    def fake(conn_, d, at, **k):
        seen["salaries"] = k.get("salaries")
        raise Stop

    monkeypatch.setattr(P, "project_day", fake)
    at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=20)

    def run(sid):
        seen.clear()
        with pytest.raises(Stop):
            P.project_slate(c, sid, at, run_id=f"r-{sid}", n_sims=100, ctx=ctx, bundle=bundle,
                            store=False)
        return seen["salaries"]

    P.set_pplay("v1")
    assert run("cl") is None and run("sd") is None
    P.set_pplay("v2")
    assert set(run("cl").values()) == {4000}                   # the slate's own Classic prices
    assert set(run("sd").values()) == {4000}                   # Showdown: the day's Classic ones
