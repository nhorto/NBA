"""P(play) v2 (`COURTSIDE_PPLAY=v2`, pipeline.pplay_v2_fix): versioning, bit-identical when
off, listed players' total P(play) = the report cell, the fringe model (features without
lookahead, opener edge cases, shipped calibration)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from courtside.proj import availability as AV
from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P

from test_proj_copula import _spec, game  # noqa: F401  (module fixture)
from test_proj_pipeline import _copy, _late_day, bundle, conn, ctx  # noqa: F401


@pytest.fixture
def restore():
    state = (P.CONTEXT_ON, P.CONTEXT_EXT, P.COPULA, P.PPLAY)
    yield
    P.set_pplay(state[3])
    P.set_copula(state[2])
    P.set_context(state[0], state[1])


# ------------------------------------------------------------------------- switch, versions

def test_default_off_and_versions_compose(restore, monkeypatch):
    assert P.PPLAY_DEFAULT == "v1"
    for v, want in (("v2", "v2"), ("1", "v2"), ("on", "v2"), ("v1", "v1"), ("off", "v1"),
                    ("", "v1"), ("junk", "v1")):
        monkeypatch.setenv(P.PPLAY_ENV, v)
        assert P._pplay_flag(P.PPLAY_ENV, "v1") == want
    for on, ext, cop in ((False, False, "v0"), (True, False, "v0"), (True, True, "v0"),
                         (True, False, "v1"), (True, True, "v1"), (False, False, "v1")):
        vs, mv = P._versions(on, ext, cop)
        assert P._versions(on, ext, cop, "v1") == (vs, mv)          # off = the old version
        assert P._versions(on, ext, cop, "v2") == (vs, mv + "+p2")  # same bundle, new version
    P.set_context(True, False)
    P.set_copula("v0")
    P.set_pplay("v2")
    assert P.MODEL_VERSION == "pathA-v0.3+p2" and P._VS == P._VS_BASE + P.CONTEXT_SUFFIX
    P.set_copula("v1")
    assert P.MODEL_VERSION == "pathA-v0.3+c1+p2"
    P.set_pplay("v1")
    assert P.MODEL_VERSION == "pathA-v0.3+c1"
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                   # seeds never change
    with pytest.raises(ValueError):
        P.set_pplay("v3")


# ------------------------------------------------------------------------- bit-identical off

def test_scenario_minutes_without_fix_is_identical(game):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    a = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    for side in ("home", "away"):                     # an all-NaN override changes nothing
        ts = getattr(spec, side)
        pm = ts.minutes.panel.assign(pp_fix=np.nan)
        ts.minutes = C.ScenarioMinutes(pm, mp)
    b = C.simulate_game(spec, cp, mp, sp, 2000, 5)
    assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)


def test_off_never_calls_the_v2_path(conn, ctx, bundle, restore, monkeypatch):  # noqa: F811
    P.set_pplay("v1")

    def boom(*a, **k):
        raise AssertionError("pplay_v2_fix called with the switch off")
    monkeypatch.setattr(P, "pplay_v2_fix", boom)
    day = _late_day(conn)
    games = P._home_away(conn, day)
    r1 = P.project_games(conn, day, games, ctx=ctx, bundle=bundle, n_sims=800, seed=3,
                         p_avail={}, env_mode="closing", listed={1}, salaries={1: 3000})
    r0 = P.project_games(conn, day, games, ctx=ctx, bundle=bundle, n_sims=800, seed=3,
                         p_avail={}, env_mode="closing")
    for a, b in zip(r0.sims, r1.sims):
        assert np.array_equal(a.fp, b.fp)
    # build_games: no pp_fix == pp_fix None == empty
    games_, panel, pa = P.historical_inputs(conn, ctx, day, protocol="avail")
    env = {g: P.environment(conn, g, None, mode="closing") for g in games_["game_id"]}
    s0 = P.run_games(P.build_games(ctx, bundle, conn, day, games_, panel, pa, env), bundle,
                     600, 9, day)
    s1 = P.run_games(P.build_games(ctx, bundle, conn, day, games_, panel, pa, env, pp_fix={}),
                     bundle, 600, 9, day)
    for a, b in zip(s0.sims, s1.sims):
        assert np.array_equal(a.fp, b.fp)


# ------------------------------------------------------------------------- fix 1: listed

def test_listed_total_pplay_is_the_report_cell(game):  # noqa: F811
    spec, mp, sp = _spec(game)
    cp = C.default_copula_params()
    ts = spec.home
    proj = ts.minutes.project(np.zeros(len(ts.minutes), bool))
    j = int(np.argmin(proj["p_play"].to_numpy()))      # the most DNP-prone candidate
    pp = float(proj["p_play"].iloc[j])
    assert pp < 0.9
    ts.players.loc[j, "p_avail"] = 0.6                 # a listed player's report cell
    off = C.simulate_game(spec, cp, mp, sp, 20_000, 1)
    pm = ts.minutes.panel.copy()
    pm["pp_fix"] = np.nan
    pm.loc[j, "pp_fix"] = 1.0
    ts.minutes = C.ScenarioMinutes(pm, mp)
    on = C.simulate_game(spec, cp, mp, sp, 20_000, 1)
    pid = ts.players["player_id"].iloc[j]
    k = int(np.flatnonzero(on.player_id == pid)[0])
    assert (on.minutes[:, k] > 0).mean() == pytest.approx(0.6, abs=0.015)       # = the cell
    assert (off.minutes[:, k] > 0).mean() < 0.6 * pp + 0.05                    # double count
    # team minutes still reconcile in every sim
    home = on.team == ts.team
    assert np.allclose(on.minutes[:, home].sum(1), off.minutes[:, home].sum(1), atol=30)


def test_v2_fix_rules(ctx, restore):  # noqa: F811
    h = ctx.h
    last = h.tg[h.tg["season"] == "2022-23"].sort_values("date").iloc[-1]
    team, day = last["team"], int(last["date"])
    pids = sorted(h.hist.loc[h.hist["team"] == team, "player_id"].unique())[:6]
    panel = pd.DataFrame({"key": f"G|{team}", "player_id": pids, "team": team, "date": day,
                          "season": "2022-23", "roster": [1, 1, 0, 0, 0, 0]})
    fix = P.pplay_v2_fix(ctx, panel, p_avail={pids[1]: 0.6, pids[2]: 0.5, pids[5]: 0.005},
                         listed={pids[1], pids[2]}, starters={pids[4]})
    k = lambda i: (f"G|{team}", int(pids[i]))  # noqa: E731
    assert k(0) not in fix                                    # rotation, unlisted: engine
    assert fix[k(1)] == 1.0 and fix[k(2)] == 1.0              # listed, not Out
    assert 0.0 < fix[k(3)] < 1.0                              # fringe: the model
    assert fix[k(4)] == AV.FRINGE_STARTER_P                   # fringe DK starter
    assert k(5) not in fix                                    # listed Out: unchanged
    assert P.pplay_v2_fix(ctx, panel.iloc[:0], {}, set()) == {}
    # season start: a DK-priced fringe player keeps the engine's value, a min-priced one not
    first = int(h.tg.loc[h.tg["season"] == "2022-23", "date"].min())
    p0 = panel.assign(date=first)
    fx = P.pplay_v2_fix(ctx, p0, {}, set(), salaries={int(pids[3]): AV.EARLY_ENGINE_SALARY,
                                                     int(pids[4]): 3000})
    assert k(3) not in fx and 0.0 < fx[k(4)] < 0.5


def test_project_slate_v2_listed_and_version(conn, ctx, bundle, restore, monkeypatch):  # noqa: F811
    c = _copy(conn)
    day = _late_day(c)
    g = P._home_away(c, day).iloc[0]
    pl = [r for r in c.execute("SELECT player_id, team FROM player_games WHERE game_id=?",
                               (g["game_id"],))]
    sid = "778"
    with c:
        c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json) "
                  "VALUES (?, ?, 70, 'main', ?)", (sid, day.isoformat(), f'["{g["game_id"]}"]'))
        for k, (pid, team) in enumerate(pl):
            c.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                      "game_type_id, draftable_id, player_name, site_player_id, team, positions,"
                      " roster_slot, salary, game_id, home, away, start_time, disabled) VALUES "
                      "('2023-01-01T00:00:00Z', 'dk', ?, 'Classic', 70, ?, ?, ?, ?, 'SF', "
                      "'UTIL', 5000, ?, ?, ?, '2023-01-01T00:00:00Z', 0)",
                      (sid, f"d{k}", f"P{pid}", f"dk{pid}", team, g["game_id"], g["home"],
                       g["away"]))
            c.execute("INSERT INTO player_ids (source, source_id, player_id) VALUES "
                      "('dk', ?, ?)", (f"dk{pid}", int(pid)))
    q = int(pl[1][0])
    rep = pd.DataFrame({"player_id": [q], "team": [pl[1][1]], "game_id": [g["game_id"]],
                        "status": ["questionable"], "p_play": [0.5]})
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=20)
    P.set_pplay("v2")
    out = P.project_slate(c, sid, at, run_id="r-p2", n_sims=6000, ctx=ctx, bundle=bundle)
    assert out.meta["model_version"].endswith("+p2")
    res = out.result.players
    assert res.loc[res["player_id"] == q, "p_play"].iloc[0] == pytest.approx(0.5, abs=0.025)
    assert c.execute("SELECT model_version FROM dfs_runs WHERE run_id='r-p2'").fetchone()[0] \
        == P.MODEL_VERSION


# ------------------------------------------------------------------------- fix 2: fringe model

def test_fringe_features_no_lookahead(ctx):  # noqa: F811
    h = ctx.h
    days = sorted(h.tg.loc[h.tg["season"] == "2022-23", "date"].unique())
    day = int(days[len(days) // 2])
    teams = sorted(h.tg["team"].unique())
    pids = sorted(h.hist["player_id"].unique())
    panel = pd.DataFrame({"player_id": pids, "team": [teams[i % len(teams)] for i in
                                                      range(len(pids))],
                          "date": day, "season": "2022-23"})
    full = AV.fringe_features(panel, h, {int(pids[0]): 3500})
    cut = M.truncate(h, pd.Timestamp("1970-01-01") + pd.Timedelta(days=day))
    trunc = AV.fringe_features(panel, cut, {int(pids[0]): 3500})
    pd.testing.assert_frame_equal(full, trunc)
    assert (full["app_season"] <= full["team_gp"] + 1e-9).all()
    assert full.loc[0, "salary"] == 3500 and np.isnan(full.loc[1, "salary"])


def test_fringe_opener_edge_cases(ctx):  # noqa: F811
    h = ctx.h
    first = int(h.tg.loc[h.tg["season"] == "2022-23", "date"].min())
    team = sorted(h.tg["team"].unique())[0]
    pids = sorted(h.hist["player_id"].unique())[:3] + [999_999]       # last: no box history
    panel = pd.DataFrame({"player_id": pids, "team": team, "date": first, "season": "2022-23"})
    X = AV.fringe_features(panel, h, {int(pids[0]): 8000, int(pids[1]): 3000})
    assert (X["team_gp"] == 0).all() and (X["app_season"] == 0).all()
    assert list(X["regime"]) == ["early", "early", "early_nosal", "early_nosal"]
    assert X.loc[3, "avg10"] == 0 and X.loc[3, "app365"] == 0 and X.loc[3, "days"] == 400
    p = AV.load_fringe().p(X)
    assert np.isfinite(p).all() and (p >= AV.FRINGE_CLAMP[0]).all() \
        and (p <= AV.FRINGE_CLAMP[1]).all()
    # the shipped model on opener-like rows: a priced rotation mover plays, a min-salary
    # two-way does not, and salary moves P(play) up
    fp = AV.load_fringe()
    rows = pd.DataFrame({"avg10": [32.0, 5.0, 20.0, 20.0], "app365": [70, 3, 60, 60],
                         "salary": [8000.0, 3000.0, 3000.0, 6000.0]})
    rows["l_app365"] = np.log1p(rows["app365"])
    rows["l_sal"] = np.log(rows["salary"] / 3000.0)
    rows["min_sal"] = (rows["salary"] <= 3000).astype(float)
    rows["l_app_season0"] = 1.0
    rows["one"] = 1.0
    rows["regime"] = "early"
    q = fp.p(rows)
    assert q[0] >= 0.85 and q[1] <= 0.3 and q[3] > q[2]
    # mid season: a fringe player is far below the engine's ~0.7
    rows["regime"] = "mid"
    rows["l_app_season0"] = 0.0
    assert (fp.p(rows) < 0.45).all()


def test_fringe_shipped_params_are_calibrated():
    fp = AV.load_fringe()
    assert set(fp.models) == {"mid", "early", "early_nosal"}
    for name in ("mid", "early", "early_nosal"):
        cv = fp.meta["cv"][name]
        assert cv["p"] == pytest.approx(cv["played"], abs=0.02)      # out-of-fold mean
        assert cv["ll"] < cv["ll_const"]                              # beats a constant
    assert fp.meta["cv"]["mid"]["engine_pp"] > fp.meta["cv"]["mid"]["played"] + 0.25
    rt = AV.FringeParams.from_json(fp.to_json())
    assert rt.models == fp.models


def test_fit_logit_recovers_and_weights():
    rng = np.random.default_rng(0)
    n = 40_000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    w_true = np.array([-0.5, 1.0, -0.7])
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    w = AV.fit_logit(X, y, ridge=0.0)
    assert np.allclose(w, w_true, atol=0.05)
    # integer weights == duplicated rows
    k = rng.integers(1, 3, n).astype(float)
    wk = AV.fit_logit(X, y, ridge=0.0, weight=k)
    wd = AV.fit_logit(np.repeat(X, k.astype(int), 0), np.repeat(y, k.astype(int)), ridge=0.0)
    assert np.allclose(wk, wd, atol=1e-6)

