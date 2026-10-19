import copy

import numpy as np
import pandas as pd
import pytest

from courtside.proj import minutes as M

# 10-man rotations: (position, minutes when healthy, starter)
ROLES = [("G", 34, 1), ("G", 32, 1), ("F", 33, 1), ("F", 30, 1), ("C", 30, 1),
         ("G", 22, 0), ("F", 20, 0), ("C", 17, 0), ("G", 12, 0), ("F", 10, 0)]
TEAMS = ("BOS", "NYK", "LAL", "GSW")
SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")


def _league(seed=0, games_per_team=24, blowout=True):
    """Box rows for a 4-team league. Absences ~8% per player-game; the bench player of the
    absent player's group gets most of his minutes; in blowouts starters lose and bench gain."""
    rng = np.random.default_rng(seed)
    pg, tg, pos = [], [], []
    gid = 0
    pid_of = {}
    for t_i, team in enumerate(TEAMS):
        for k, (p, m, s) in enumerate(ROLES):
            pid = 1000 * (t_i + 1) + k
            pid_of[(team, k)] = pid
            pos.append({"player_id": pid, "season": None, "g1": "GFC".index(p),
                        "g2": "GFC".index(p)})
    for season in SEASONS:
        y = int(season[:4])
        day = pd.Timestamp(f"{y}-10-25")
        for r in range(games_per_team // 2 * len(TEAMS) // 2):
            pairs = [(TEAMS[0], TEAMS[1 + r % 3]), (TEAMS[1 + (r + 1) % 3], TEAMS[1 + (r + 2) % 3])]
            for h, a in pairs:
                gid += 1
                g = f"002{y % 100:02d}{gid:05d}"
                margin = int(rng.normal(0, 13)) or 1
                hp = 110 + max(margin, 0)
                ap = hp - margin
                for team, pts, opp_pts in ((h, hp, ap), (a, ap, hp)):
                    tm = margin if team == h else -margin
                    tg.append({"game_id": g, "team": team, "season": season,
                               "season_type": "Regular Season",
                               "game_date": day.strftime("%Y-%m-%d"), "min": 240.0, "pts": pts})
                    absent = rng.random(len(ROLES)) < 0.08
                    mins = np.array([m for _, m, _ in ROLES], float)
                    for k in np.flatnonzero(absent):
                        grp = ROLES[k][0]
                        sub = [i for i, (p, _, s) in enumerate(ROLES)
                               if p == grp and not s and not absent[i]]
                        if sub:
                            mins[sub[0]] += 0.7 * mins[k]
                        mins[k] = 0
                    if blowout and abs(tm) >= 20:
                        st = np.array([s for _, _, s in ROLES], bool)
                        mins[st & (mins > 0)] -= 5
                        mins[~st & (mins > 0)] += 25 / max((~st & (mins > 0)).sum(), 1)
                    mins = np.where(mins > 0, mins + rng.normal(0, 2, len(mins)), 0)
                    mins = np.clip(mins, 0, 46)
                    mins *= 240 / mins.sum()
                    starters = [i for i, (_, _, s) in enumerate(ROLES) if s and not absent[i]]
                    for i in np.flatnonzero(~absent & (mins > 0.5)):
                        if len(starters) < 5 and i not in starters and not ROLES[i][2]:
                            starters.append(i)
                    for i in np.flatnonzero(mins > 0.5):
                        pg.append({"game_id": g, "player_id": pid_of[(team, i)], "team": team,
                                   "season": season, "season_type": "Regular Season",
                                   "game_date": day.strftime("%Y-%m-%d"),
                                   "min": float(mins[i]), "pf": 2, "pts": int(mins[i] / 2),
                                   "fg3m": 1, "reb": 4, "ast": 2, "stl": 1, "blk": 0, "tov": 1,
                                   "started": int(i in starters[:5])})
            day += pd.Timedelta(days=1 + (r % 2))
    return pd.DataFrame(pg), pd.DataFrame(tg), pd.DataFrame(pos)


@pytest.fixture(scope="module")
def league():
    pg, tg, pos = _league()
    return pg, tg, pos, M.build_history(pg, tg, pos)


def _target(h, season="2022-23", i=40):
    t = M.eval_targets(h, [season])
    return t.iloc[[i]].reset_index(drop=True)


def test_no_lookahead(league):
    pg, tg, pos, h = league
    t = _target(h)
    day = pd.Timestamp(int(t["date"].iloc[0]), unit="D")
    full = M.pregame_panel(h, t)
    # scramble everything on/after the day and rebuild from scratch
    dd = pd.to_datetime(pg["game_date"])
    pg2 = pg.copy()
    late = (dd >= day).to_numpy()
    rng = np.random.default_rng(5)
    pg2.loc[late, "min"] = rng.uniform(1, 45, late.sum())
    pg2.loc[late, "started"] = rng.integers(0, 2, late.sum())
    tg2 = tg.copy()
    tg2.loc[(pd.to_datetime(tg2["game_date"]) >= day).to_numpy(), "pts"] = 1
    h2 = M.build_history(pg2, tg2, pos)
    h3 = M.build_history(pg[~late], tg[(pd.to_datetime(tg["game_date"]) < day).to_numpy()], pos)
    for other in (M.pregame_panel(h2, t), M.pregame_panel(h3, t), M.pregame_panel(M.truncate(h, day), t)):
        pd.testing.assert_frame_equal(full.drop(columns=["k"]), other.drop(columns=["k"]),
                                      check_dtype=False)
    prm = M.default_params()
    team = t["team"].iloc[0]
    a = M.project_team(h, prm, team, day, season="2022-23")
    b = M.project_team(h2, prm, team, day, season="2022-23")
    pd.testing.assert_frame_equal(a, b)


def test_fit_is_walk_forward_and_json(league):
    pg, tg, pos, h = league
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,))
    assert prm.train_seasons == ["2019-20", "2020-21", "2021-22"]
    assert all(s < "2022-23" for s in prm.train_seasons)
    again = M.MinutesParams.from_json(prm.to_json())
    assert again == prm
    # fitting with the test season's box scrambled gives identical params
    late = (pg["season"] == "2022-23").to_numpy()
    pg2 = pg.copy()
    pg2.loc[late, "min"] = 20.0
    h2 = M.build_history(pg2, tg, pos)
    prm2 = M.fit(h2, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,))
    assert prm2.to_json() == prm.to_json()


def _arrays(n_bench_c=1):
    roles = ROLES if n_bench_c else [r for r in ROLES if not (r[0] == "C" and not r[2])]
    base = np.array([m for _, m, _ in roles], float)
    g = np.array(["GFC".index(p) for p, _, _ in roles])
    st10 = np.array([float(s) for _, _, s in roles])
    return base, g, st10


@pytest.mark.parametrize("recon", ["variance", "proportional"])
def test_total_and_cap(recon):
    prm = M.default_params()
    prm.recon = recon
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 13
        base = rng.uniform(2, 44, n)
        # proportional mode (players.allocate_minutes) caps expected minutes, so it keeps the
        # total only with p = 1; variance mode caps the conditional mean at 40 for any p
        p = rng.uniform(0.6, 1.0, n) if recon == "variance" else np.ones(n)
        g = rng.integers(0, 3, n)
        absent = rng.random(n) < 0.2
        mean, det = M.project_arrays(base, p, absent, g, g, rng.random(n), base,
                                     np.tile(prm.starter_league, (n, 1)), prm, detail=True)
        pp = det["p"]
        target = min(240.0 - det["reserve"], M.CAP * pp.sum())
        assert np.isclose((pp * mean).sum(), target, atol=1e-6)
        assert mean.max() <= M.CAP + 1e-9
        assert (mean[absent] == 0).all()
    # one huge base hits the 40 cap; OT raises cap and total
    base = np.r_[60.0, np.full(9, 20.0)]
    one = np.ones(10)
    mean = M.project_arrays(base, one, np.zeros(10, bool), np.zeros(10, int),
                            np.zeros(10, int), np.zeros(10), base,
                            np.tile(prm.starter_league, (10, 1)), prm)
    assert np.isclose(mean[0], 40.0)
    mean = M.project_arrays(base, one, np.zeros(10, bool), np.zeros(10, int),
                            np.zeros(10, int), np.zeros(10), base,
                            np.tile(prm.starter_league, (10, 1)), prm, n_ot=1)
    assert np.isclose(mean.sum(), 265.0 - M.reserve_minutes(prm, base.sum(), 10)[0]) \
        and mean[0] <= 45.0 + 1e-9


def _project(absent_idx, mode="top", starters=None, n_bench_c=1):
    base, g, st10 = _arrays(n_bench_c)
    n = len(base)
    prm = M.default_params()
    prm.nmu_mode = mode
    absent = np.zeros(n, bool)
    absent[absent_idx] = True
    stt = None
    if starters is not None:
        stt = np.zeros(n, bool)
        stt[starters] = True
    return M.project_arrays(base, np.ones(n), absent, g, g, st10, base,
                            np.tile([32.0, 31.0, 30.0], (n, 1)), prm, starters=stt,
                            detail=True)


@pytest.mark.parametrize("mode", ["top", "softmax"])
def test_next_man_up_same_group(mode):
    healthy, _ = _project([], mode)
    mean, det = _project([4], mode)          # starting C out
    rise = mean - healthy
    assert int(np.argmax(det["pstart"])) == 7   # the bench C
    assert int(np.argmax(rise)) == 7
    assert rise[7] > 5.0
    assert mean[4] == 0.0
    # starting G out -> the best bench G (index 5), not the deeper G (8)
    mean, det = _project([0], mode)
    assert int(np.argmax(det["pstart"])) == 5
    assert det["pstart"][5] > det["pstart"][8]


def test_next_man_up_adjacent_group_and_confirmed_starters():
    # no bench C on the roster -> a bench F (adjacent group) is promoted
    mean, det = _project([4], "top", n_bench_c=0)
    base, g, _ = _arrays(0)
    up = int(np.argmax(det["pstart"]))
    assert g[up] == 1 and det["pstart"][up] == 1.0
    # confirmed starters: the deep F (index 9) starts for the absent C -> he gets the lift,
    # the bench C gets none
    mean, det = _project([4], "softmax", starters=[0, 1, 2, 3, 9])
    assert det["pstart"][9] == 1.0 and det["pstart"][7] == 0.0
    healthy, _ = _project([], "softmax")
    assert mean[9] - healthy[9] > 8.0


def _g_params():
    prm = M.default_params()
    nt, nb = len(M.TIER_EDGES) + 1, len(M.MARGIN_BINS) - 1
    tab = np.zeros((nt, 2, nb))
    ramp = np.clip((np.arange(nb) - 2) * 1.2, 0, None)     # 0 until 10-15, then growing
    tab[4] = -ramp          # stars lose minutes in blowouts (won and lost)
    tab[0] = +ramp          # deep bench gain
    prm.g_table = tab.tolist()
    return prm


def test_g_curve_sign_and_shape():
    prm = _g_params()
    tiers = np.array([4, 0])
    close, blow = M.g_value(prm, tiers, 2.0), M.g_value(prm, tiers, -30.0)
    assert blow[0] < close[0] - 3 and blow[1] > close[1] + 3
    ms = np.linspace(0, 40, 41)
    v = M.g_value(prm, tiers, ms)
    assert (np.diff(v[:, 0]) <= 1e-12).all() and (np.diff(v[:, 1]) >= -1e-12).all()
    # a big favourite's star expects fewer minutes than in a pick'em
    proj = pd.DataFrame({"mean": [36.0, 34, 33, 30, 30, 22, 20, 17, 11, 7],
                         "p_play": 1.0, "pf36": 3.0})
    e0 = M.expected_minutes(proj, prm, spread=0.0)
    e15 = M.expected_minutes(proj, prm, spread=15.0)
    assert e15[0] < e0[0] and e15[-1] > e0[-1]
    assert np.isclose(e0.sum(), 240 + 25 * prm.mean_n_ot)


def test_fitted_g_on_synthetic_blowouts(league):
    _, _, _, h = league
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,))
    t_star = int(M.tier_of([33.0])[0])
    t_bench = int(M.tier_of([11.0])[0])
    g = M.g_value(prm, np.array([t_star, t_bench]), np.array([2.0, 28.0, -28.0]))
    assert g[1, 0] < g[0, 0] - 1.5 and g[2, 0] < g[0, 0] - 1.5     # starters lose
    assert g[1, 1] > g[0, 1] and g[2, 1] > g[0, 1]                 # bench gains


def test_sampling_reproducible_and_constrained():
    prm = M.default_params()
    proj = pd.DataFrame({"mean": [36.0, 34, 33, 30, 30, 22, 20, 17, 11, 7, 0.0],
                         "p_play": [1, 1, 1, 1, 1, .98, .97, .9, .6, .3, 0.0],
                         "pf36": 3.5, "reserve": 1.0})
    margins = np.random.default_rng(0).normal(3, 12.4, 500)
    a = M.sample_minutes(proj, prm, margins, np.random.default_rng(42))
    b = M.sample_minutes(proj, prm, margins, np.random.default_rng(42))
    c = M.sample_minutes(proj, prm, margins, np.random.default_rng(43))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (500, 11)
    assert np.allclose(a.sum(1), 239.0)
    assert a.max() <= 40 + 1e-9 and (a[:, -1] == 0).all()
    ot = np.r_[np.ones(250), np.zeros(250)]
    d = M.sample_minutes(proj, prm, margins, np.random.default_rng(1), n_ot=ot)
    assert np.allclose(d[:250].sum(1), 264.0) and d.max() <= 45 + 1e-9
    # the sim mean is close to the margin-integrated expectation for rotation players
    e = M.expected_minutes(proj, prm, spread=3.0, n_ot=0)
    s = M.sample_minutes(proj, prm, np.random.default_rng(3).normal(3, 12.4, 4000),
                         np.random.default_rng(4), noise=False)
    played = np.where(s[:, :6] > 0, s[:, :6], np.nan)
    assert np.allclose(np.nanmean(played, 0), e[:6], atol=1.5)


def test_project_team_api(league):
    _, _, _, h = league
    prm = M.default_params()
    t = _target(h)
    day = pd.Timestamp(int(t["date"].iloc[0]), unit="D")
    team = t["team"].iloc[0]
    healthy = M.project_team(h, prm, team, day, season="2022-23")
    star = int(healthy.iloc[0]["player_id"])
    out = M.project_team(h, prm, team, day, absent=[star], season="2022-23", spread=5.0)
    assert out.set_index("player_id").loc[star, "mean"] == 0
    assert {"player_id", "p_play", "mean", "exp_min", "pstart", "g1"} <= set(out.columns)
    assert np.isclose((out["p_play"] * out["mean"]).sum(), 240.0 - out["reserve"].iloc[0])
    others = out[out["player_id"] != star].set_index("player_id")["mean"]
    assert (others - healthy.set_index("player_id")["mean"].reindex(others.index)).sum() > 20


# --- v1 (T2.2): LightGBM base, coach by date, S15 fallback -----------------------------------

def _stints():
    rows = [("BOS", "2019-10-01", 1), ("NYK", "2019-10-01", 2), ("NYK", "2021-11-20", 3),
            ("LAL", "2019-10-01", 4), ("LAL", "2022-10-01", 2), ("GSW", "2019-10-01", 5)]
    df = pd.DataFrame(rows, columns=["team", "day", "coach_id"])
    df["date"] = M._days(df["day"])
    return df[["team", "date", "coach_id"]]


V1_KW = {"max_rounds": 120, "gbm_params": {"min_data_in_leaf": 20}, "nmu_grid": (0.0, 1.0),
         "k_grid": (0.0,), "share_grid": (0.0, 0.15), "cv_grid": (0.2,)}


@pytest.fixture(scope="module")
def league_v1():
    pg, tg, pos = _league()
    h = M.build_history(pg, tg, pos, stints=_stints())
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                v1_kw=V1_KW)
    return pg, tg, pos, h, prm


def _day(s):
    return int(M._days(pd.Series([s]))[0])


def test_v1_fit_api_and_json(league_v1):
    _, _, _, h, prm = league_v1
    assert M.default_params().version == "v0.5"
    assert prm.version == "v1" and prm.gbm_model and prm.gbm_features
    assert prm.train_seasons == ["2019-20", "2020-21", "2021-22"]
    again = M.MinutesParams.from_json(prm.to_json())
    assert again == prm
    # the pipeline's Bundle rebuilds params with MinutesParams(**dict)
    import json
    assert M.MinutesParams(**json.loads(prm.to_json())) == prm
    t = _target(h)
    day = pd.Timestamp(int(t["date"].iloc[0]), unit="D")
    team = t["team"].iloc[0]
    healthy = M.project_team(h, prm, team, day, season="2022-23")
    star = int(healthy.iloc[0]["player_id"])
    out = M.project_team(h, prm, team, day, absent=[star], season="2022-23", spread=4.0)
    assert {"player_id", "p_play", "mean", "exp_min", "pstart", "g1", "reserve"} <= set(out.columns)
    assert out.set_index("player_id").loc[star, "mean"] == 0
    # reconciliation: sum of p x E[min | plays] is the team total minus the reserve
    for o in (healthy, out):
        assert np.isclose((o["p_play"] * o["mean"]).sum(), 240.0 - o["reserve"].iloc[0])
        assert o["mean"].max() <= M.CAP + 1e-9
    # learned promotion: the others gain most of the absent star's minutes
    others = out[out["player_id"] != star].set_index("player_id")["mean"]
    assert (others - healthy.set_index("player_id")["mean"].reindex(others.index)).sum() > 20


def test_v1_through_copula_scenarios(league_v1):
    from courtside.proj import copula as C
    _, _, _, h, prm = league_v1
    t = _target(h)
    panel = M.pregame_panel(h, t)
    sm = C.ScenarioMinutes(panel, prm)
    n = len(sm)
    a = sm.project(np.zeros(n, bool))
    ab = np.zeros(n, bool)
    ab[int(np.argmax(a["mean"]))] = True
    b = sm.project(ab)
    for o in (a, b):
        assert np.isclose((o["p_play"] * o["mean"]).sum(), 240.0 - o["reserve"].iloc[0])
    assert b["mean"].to_numpy()[ab][0] == 0.0
    # the scenario changes the GBM base of the teammates (scenario features)
    assert not np.allclose(a["base"].to_numpy()[~ab], b["base"].to_numpy()[~ab])


def test_v1_no_lookahead(league_v1):
    pg, tg, pos, h, prm = league_v1
    t = _target(h)
    day = pd.Timestamp(int(t["date"].iloc[0]), unit="D")
    dd = pd.to_datetime(pg["game_date"])
    late = (dd >= day).to_numpy()
    pg2 = pg.copy()
    rng = np.random.default_rng(11)
    pg2.loc[late, "min"] = rng.uniform(1, 45, late.sum())
    pg2.loc[late, "pf"] = rng.integers(0, 7, late.sum())
    pg2.loc[late, "started"] = rng.integers(0, 2, late.sum())
    tg2 = tg.copy()
    tg2.loc[(pd.to_datetime(tg2["game_date"]) >= day).to_numpy(), "pts"] = 1
    h2 = M.build_history(pg2, tg2, pos, stints=_stints())
    a = M.pregame_panel(h, t)
    for other in (M.pregame_panel(h2, t), M.pregame_panel(M.truncate(h, day), t)):
        pd.testing.assert_frame_equal(a.drop(columns=["k"]), other.drop(columns=["k"]),
                                      check_dtype=False)
    assert {"c_top7", "ew_vf", "abs_len_now", "team_rank", "tgs"} <= set(a.columns)
    team = t["team"].iloc[0]
    pa = M.project_team(h, prm, team, day, season="2022-23")
    pb = M.project_team(h2, prm, team, day, season="2022-23")
    pd.testing.assert_frame_equal(pa, pb)
    # refitting with the test season scrambled gives the same model
    late_s = (pg["season"] == "2022-23").to_numpy()
    pg3 = pg.copy()
    pg3.loc[late_s, "min"] = 20.0
    h3 = M.build_history(pg3, tg, pos, stints=_stints())
    prm3 = M.fit(h3, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=V1_KW)
    assert prm3.gbm_model == prm.gbm_model and prm3.to_json() == prm.to_json()


def test_coach_by_date_continuity(league_v1):
    _, _, _, h, _ = league_v1
    cs = h.coach
    nyk = cs.features(np.array(["NYK", "NYK"]), np.array([_day("2021-11-19"),
                                                        _day("2021-11-20")]))
    assert nyk["coach_id"].tolist() == [2, 3]
    assert nyk["c_games"].iloc[1] == 0.0          # new coach: no games yet ...
    assert np.isclose(nyk["c_top7"].iloc[1], cs.league[cs.league["date"] < _day("2021-11-20")]
                      ["l_top7"].iloc[-1])        # ... so his concentration is the league's
    # coach 2 moves NYK -> LAL: his history comes with him, his LAL count starts at 0
    first_lal = int(h.tg[(h.tg["team"] == "LAL") & (h.tg["season"] == "2022-23")]["date"].min())
    lal = cs.features(np.array(["LAL", "LAL"]), np.array([first_lal, first_lal + 60]))
    assert (lal["coach_id"] == 2).all()
    assert lal["c_games"].iloc[0] > 10 and lal["c_team_n"].iloc[0] == 0
    assert lal["c_team_n"].iloc[1] > 0 and lal["c_games"].iloc[1] > lal["c_games"].iloc[0]
    # strictly before the date: identical on the truncated state
    d = first_lal + 30
    a = cs.features(np.array(["LAL", "BOS"]), np.array([d, d]))
    b = M.truncate(h, pd.Timestamp(d, unit="D")).coach.features(np.array(["LAL", "BOS"]),
                                                              np.array([d, d]))
    pd.testing.assert_frame_equal(a, b)
    # no stints -> coach features missing, not wrong
    h0 = M.build_history(*_league()[:3])
    p0 = M.pregame_panel(h0, _target(h0))
    assert p0["c_top7"].isna().all() and p0["t_top7"].isna().all()


def test_curated_coach_stints():
    snap = pd.DataFrame({"season": ["2024-25", "2026-27", "2026-27"],
                         "team": ["SAC", "SAC", "CHI"], "coach_id": [57, 57, 201168],
                         "coach_name": ["Doug Christie", "Doug Christie", "Tiago Splitter"]})
    st = M.coach_stints(snap)
    for team, g in st.groupby("team"):
        assert g["date"].is_monotonic_increasing
    assert set(M.COACH_STINTS) == set(st["team"])
    # curated names resolve to snapshot ids; unknown coaches get a stable negative id
    sac = st[st["team"] == "SAC"]
    assert sac["coach_id"].iloc[-1] == 57 and len(sac[sac["coach_id"] == 57]) == 1
    assert (st[st["coach_name"] == "Larry Drew"]["coach_id"] < 0).all()
    assert M._coach_id("Larry Drew", {}) == M._coach_id("Larry Drew", {})
    # Splitter (POR interim in the curated 2025-26) keeps his snapshot id when he moves to CHI
    spl = st[st["coach_name"] == "Tiago Splitter"]
    assert set(spl["team"]) == {"POR", "CHI"} and (spl["coach_id"] == 201168).all()


def test_v1_fallback_s15(league_v1):
    _, _, _, h, prm = league_v1
    t = _target(h)
    q05 = M.MinutesParams.from_json(prm.to_json())
    q05.version = "v0.5"
    # a player without box history (extra) and one with < V1_MIN_GP games keep the v0.5 base
    ex = pd.DataFrame({"key": t["key"].iloc[0], "player_id": [999_999]})
    p = M.pregame_panel(h, t, extra=ex)
    p["absent"] = False
    p.loc[p.index[0], "n_gp"] = M.V1_MIN_GP - 1
    b1, pp1 = M.base_and_pplay(p, prm)
    b0, pp0 = M.base_and_pplay(p, q05)
    fb = ~M.v1_ok(p)
    assert fb.sum() == 2 and np.allclose(b1[fb], b0[fb]) and not np.allclose(b1[~fb], b0[~fb])
    assert np.allclose(pp1, pp0)             # P(plays | available) is the v0.5 model in both
    # a panel without the v1 columns falls back entirely (and is counted)
    M.V1_STATS.update(rows=0, fallback_rows=0, panel_fallback_calls=0)
    bare = p.drop(columns=["ew_vf", "abs_len_now", "team_rank", "c_top7", "tgs"])
    b2, _ = M.base_and_pplay(bare, prm)
    assert np.allclose(b2, b0) and M.V1_STATS["panel_fallback_calls"] == 1
