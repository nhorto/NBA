"""Preseason minutes features (item #2, `minutes.V1_PRESEASON`, switch COURTSIDE_PRESEASON).

No lookahead, missing preseason, scrimmage / non-NBA handling at load, the switch-off identity
(OFF never reads the tables and the minutes path is unchanged) and the live season read."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from test_proj_minutes import V1_KW, _league, _stints

PRE = list(M.V1_PRESEASON)
ALL_PRE = PRE + list(M.PRE_TEAM_COLS)


def _preseason(pg, tg, seed=3, skip=(), n_games=4):
    """Synthetic preseason logs before each season's first regular-season day: 4 games per team,
    minutes ~ 0.7 x the player's regular-season share with noise; players in `skip` sit out."""
    rng = np.random.default_rng(seed)
    pl, tm = [], []
    mpg = pg.groupby(["season", "team", "player_id"])["min"].mean()
    for season, d in tg.groupby("season"):
        first = pd.Timestamp(d["game_date"].min())
        for team in sorted(d["team"].unique()):
            roster = mpg.loc[season, team]
            for k in range(n_games):
                day = first - pd.Timedelta(days=3 * (n_games - k))
                gid = f"001{int(season[2:4]):02d}{abs(hash((team, k))) % 10**5:05d}"
                tm.append({"season": season, "team": team, "game_id": gid,
                           "date": int(M._days(pd.Series([day]))[0]), "min": 240.0})
                for pid, m in roster.items():
                    if pid in skip:
                        continue
                    mm = max(0.0, 0.7 * m + rng.normal(0, 3))
                    if mm < 1:
                        continue
                    pl.append({"season": season, "player_id": int(pid), "team": team,
                               "game_id": gid, "date": tm[-1]["date"], "min": float(round(mm))})
    return M.Preseason(pd.DataFrame(pl), pd.DataFrame(tm))


@pytest.fixture(scope="module")
def league_pre():
    pg, tg, pos = _league()
    skip = {1000 * 1 + 0}                       # BOS's top guard sits out every preseason
    h = M.build_history(pg, tg, pos, stints=_stints())
    h.pre = _preseason(pg, tg, skip=skip)
    return pg, tg, pos, h, skip


def _opener_targets(h, season="2022-23"):
    t = M.eval_targets(h, [season])
    return t[t["date"] == t["date"].min()].reset_index(drop=True)


def test_columns_and_values(league_pre):
    pg, tg, pos, h, skip = league_pre
    t = _opener_targets(h)
    p = M.pregame_panel(h, t)
    assert set(PRE) <= set(p.columns)
    q = p[p["roster"] == 1]
    # the player who sat out: missing indicator, 0 games, everything else NaN
    s = q[q["player_id"].isin(skip)]
    if len(s):
        assert (s["pre_none"] == 1).all() and (s["pre_gp"] == 0).all()
        assert s[["pre_mpg", "pre_share", "pre_rank"]].isna().all().all()
    o = q[~q["player_id"].isin(skip)]
    assert (o["pre_none"] == 0).all()
    # hand check of one player against the raw logs
    r = o.iloc[0]
    pl = h.pre.players
    mine = pl[(pl["player_id"] == r["player_id"]) & (pl["season"] == "2022-23")]
    team = h.pre.teams[(h.pre.teams["team"] == r["team"]) & (h.pre.teams["season"] == "2022-23")]
    assert r["pre_gp"] == len(mine)
    assert np.isclose(r["pre_mpg"], mine["min"].mean())
    assert np.isclose(r["pre_share"], 5 * mine["min"].sum() / team["min"].sum())
    l2 = set(team.sort_values("date")["game_id"].tail(2))
    assert np.isclose(r["pre_mpg_l2"], mine[mine["game_id"].isin(l2)]["min"].sum() / 2)
    # rank 1 = most preseason minutes on the team; prior-season ratio uses the season before
    tm_rows = o[o["team"] == r["team"]]
    assert tm_rows.loc[tm_rows["pre_share"].idxmax(), "pre_rank"] == 1
    prev = pg[(pg["player_id"] == r["player_id"]) & (pg["season"] == "2021-22")]
    prev_mpg = h.hist[(h.hist["player_id"] == r["player_id"])
                      & (h.hist["season"] == "2021-22")]["min_r"].mean()
    assert len(prev) and np.isclose(r["pre_rel_prior"], r["pre_mpg"] / max(prev_mpg, 5.0))


def test_team_without_preseason_is_unknown(league_pre):
    pg, tg, pos, h, _ = league_pre
    t = _opener_targets(h)
    pre = h.pre
    h2 = M.History(**{**h.__dict__, "pre": M.Preseason(
        pre.players[pre.players["team"] != "NYK"], pre.teams[pre.teams["team"] != "NYK"])})
    p = M.pregame_panel(h2, t)
    nyk = p[p["team"] == "NYK"]
    assert len(nyk) and nyk[PRE].isna().all().all()
    assert (p[p["team"] != "NYK"]["pre_none"].notna()).all()


def test_no_lookahead(league_pre):
    pg, tg, pos, h, _ = league_pre
    pre = h.pre
    # a row dated inside the preseason window only sees the games before it
    season = "2022-23"
    pt = pre.teams[pre.teams["season"] == season].sort_values("date")
    mid = int(pt["date"].iloc[len(pt) // 2])
    t = _opener_targets(h).iloc[:2].copy()
    t["date"] = mid
    full = M.pregame_panel(h, t)
    cut = M.pregame_panel(M.truncate(h, pd.Timestamp(mid, unit="D")), t)
    pd.testing.assert_frame_equal(full[PRE].reset_index(drop=True),
                                  cut[PRE].reset_index(drop=True), check_dtype=False)
    # scrambling every preseason row on/after the date changes nothing
    pl2 = pre.players.copy()
    late = pl2["date"] >= mid
    pl2.loc[late, "min"] = 40.0
    h2 = M.History(**{**h.__dict__, "pre": M.Preseason(pl2, pre.teams)})
    pd.testing.assert_frame_equal(full[PRE].reset_index(drop=True),
                                  M.pregame_panel(h2, t)[PRE].reset_index(drop=True))
    # ... while scrambling earlier rows does change them (the test has teeth)
    pl3 = pre.players.copy()
    pl3.loc[pl3["date"] < mid, "min"] = 40.0
    h3 = M.History(**{**h.__dict__, "pre": M.Preseason(pl3, pre.teams)})
    assert not M.pregame_panel(h3, t)["pre_mpg"].equals(full["pre_mpg"])
    # regular-season rows: the next season's preseason (after the date) is never used
    t2 = M.eval_targets(h, ["2021-22"]).iloc[[30]].reset_index(drop=True)
    a = M.pregame_panel(h, t2)
    pl4 = pre.players.copy()
    pl4.loc[pl4["season"] == "2022-23", "min"] = 1.0
    h4 = M.History(**{**h.__dict__, "pre": M.Preseason(pl4, pre.teams)})
    pd.testing.assert_frame_equal(a[PRE], M.pregame_panel(h4, t2)[PRE])


def _pre_db():
    conn = sqlite3.connect(":memory:")
    for sql in db.MIGRATIONS:
        if "preseason_player_games" in sql:
            conn.executescript(sql)
    conn.execute("CREATE TABLE IF NOT EXISTS team_games (game_id TEXT, team TEXT, season TEXT)")
    conn.executemany("INSERT INTO team_games VALUES (?,?,?)",
                     [("0022300001", "BOS", "2023-24"), ("0022300001", "NYK", "2023-24")])
    tg = [("0012300001", "BOS", "2023-24", "2023-10-08", 240, 0, 0),
          ("0012300001", "NYK", "2023-24", "2023-10-08", 240, 0, 0),
          ("0012300002", "BOS", "2023-24", "2023-10-12", 240, 0, 1),   # vs Real Madrid
          ("0012300002", "RMD", "2023-24", "2023-10-12", 240, 0, 1),
          ("0011900101", "BOS", "2019-20", "2020-07-22", 200, 1, 0),   # bubble scrimmage
          ("0011900101", "NYK", "2019-20", "2020-07-22", 200, 1, 0)]
    conn.executemany("INSERT INTO preseason_team_games (game_id, team, season, game_date, min, "
                     "scrimmage, vs_non_nba) VALUES (?,?,?,?,?,?,?)", tg)
    pl = [("0012300001", 1, "BOS", "2023-24", "2023-10-08", 30, 0, 0),
          ("0012300001", 2, "NYK", "2023-24", "2023-10-08", 20, 0, 0),
          ("0012300002", 1, "BOS", "2023-24", "2023-10-12", 24, 0, 1),
          ("0012300002", 9, "RMD", "2023-24", "2023-10-12", 35, 0, 1),
          ("0011900101", 1, "BOS", "2019-20", "2020-07-22", 22, 1, 0)]
    conn.executemany("INSERT INTO preseason_player_games (game_id, player_id, team, season, "
                     "game_date, min, scrimmage, vs_non_nba) VALUES (?,?,?,?,?,?,?,?)", pl)
    return conn


def test_load_excludes_scrimmages_and_foreign_sides():
    pre = M.load_preseason(_pre_db(), since="2010-11")
    assert set(pre.players["player_id"]) == {1, 2}            # no RMD player
    assert set(pre.teams["team"]) == {"BOS", "NYK"}
    assert set(pre.players["season"]) == {"2023-24"}          # the 2020 scrimmage is dropped
    # the NBA side of the game against an international club is kept
    assert (pre.players["game_id"] == "0012300002").sum() == 1
    assert len(pre.teams[pre.teams["team"] == "BOS"]) == 2


def test_switch_off_identity(league_pre, monkeypatch):
    pg, tg, pos, h, _ = league_pre
    assert P.PRESEASON_DEFAULT is False
    # version strings: OFF is unchanged for every arm, ON is a suffix before mr / c1 / p2
    for on in (False, True):
        for mr in (False, True):
            for cop in ("v0", "v1"):
                for pp in ("v1", "v2"):
                    assert P._versions(on, False, cop, pp, mr) == \
                        P._versions(on, False, cop, pp, mr, False)
    assert P._versions(True, False, "v0", "v2", False, True) == ("_v1_cx_pre", "pathA-v0.3+pre+p2")
    assert P._versions(True, False, "v1", "v1", True, True) == ("_v1_cx_pre_mr_c1",
                                                               "pathA-v0.3+pre+mr+c1")
    P.set_preseason(True)
    try:
        assert P.MODEL_VERSION.endswith("+pre") or "+pre+" in P.MODEL_VERSION
    finally:
        P.set_preseason(False)
    assert "+pre" not in P.MODEL_VERSION and "_pre" not in P._VS
    # OFF: a History without logs has no pre_* columns ...
    h0 = M.History(**{**h.__dict__, "pre": None})
    t = _opener_targets(h)
    p0 = M.pregame_panel(h0, t)
    assert not set(ALL_PRE) & set(p0.columns)
    p1 = M.pregame_panel(h, t)
    pd.testing.assert_frame_equal(p0, p1.drop(columns=ALL_PRE))
    # ... and OFF params project identically with or without the columns present
    prm = M.fit(h0, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                v1_kw=V1_KW)
    assert not M.uses_preseason(prm)
    a = M.run_projection(M.oracle_panel(h0, t), prm)
    b = M.run_projection(M.oracle_panel(h, t), prm)
    pd.testing.assert_frame_equal(a, b.drop(columns=ALL_PRE))


def test_preseason_fit_and_projection(league_pre):
    pg, tg, pos, h, _ = league_pre
    prm = M.fit(h, "2022-23", window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                v1_kw=dict(V1_KW, features=M.preseason_features()))
    assert M.uses_preseason(prm)
    assert M.MinutesParams.from_json(prm.to_json()) == prm
    t = _opener_targets(h)
    day = pd.Timestamp(int(t["date"].iloc[0]), unit="D")
    out = M.project_team(h, prm, t["team"].iloc[0], day, season="2022-23")
    assert np.isclose((out["p_play"] * out["mean"]).sum(), 240.0 - out["reserve"].iloc[0])
    # without the logs the ON params still run (pre_* missing -> NaN, LightGBM's missing branch)
    h0 = M.History(**{**h.__dict__, "pre": None})
    out0 = M.project_team(h0, prm, t["team"].iloc[0], day, season="2022-23")
    assert len(out0) == len(out)


def test_live_season_before_any_box_row(league_pre):
    """The live path on opening night: the season has preseason logs but no regular-season box
    rows yet; the panel still gets the current preseason and the prior-season ratio."""
    pg, tg, pos, h, _ = league_pre
    new = "2023-24"
    first = int(h.tg["date"].max()) + 160
    pre = h.pre
    roster = pg[pg["season"] == "2022-23"].groupby(["team", "player_id"]).size().reset_index()
    pl = [{"season": new, "player_id": int(r.player_id), "team": r.team, "game_id": f"g{r.team}",
           "date": first - 5, "min": 20.0} for r in roster.itertuples()]
    tm = [{"season": new, "team": t_, "game_id": f"g{t_}", "date": first - 5, "min": 240.0}
          for t_ in roster["team"].unique()]
    h2 = M.History(**{**h.__dict__, "pre": M.Preseason(
        pd.concat([pre.players, pd.DataFrame(pl)], ignore_index=True),
        pd.concat([pre.teams, pd.DataFrame(tm)], ignore_index=True))})
    t = pd.DataFrame({"key": ["x"], "team": ["BOS"], "date": [first], "season": [new]})
    p = M.pregame_panel(h2, t)
    assert len(p) and (p["pre_gp"] == 1).all() and (p["pre_mpg"] == 20.0).all()
    assert p["pre_rel_prior"].notna().all()


def test_features_stop_after_the_early_season(league_pre):
    """V1_PRESEASON is NaN from PRE_MAX_TGS team games on (stale camp evidence); the team-level
    debut input stays."""
    pg, tg, pos, h, _ = league_pre
    t = M.eval_targets(h, ["2022-23"])
    p = M.pregame_panel(h, t)
    late = p["tgs"] >= M.PRE_MAX_TGS
    assert late.any() and (~late).any()
    assert p.loc[late, PRE].isna().all().all()
    assert p.loc[~late & (p["roster"] == 1), "pre_none"].notna().all()
    assert p.loc[late, "pre_new_tg"].notna().all()


def test_debut_reserve_takes_team_minutes():
    """reserve_debut adds to the reserve: the candidates share 240 - reserve - debut minutes."""
    prm = M.default_params()
    n = 10
    base = np.linspace(34, 8, n)
    args = (base, np.ones(n), np.zeros(n, bool), np.zeros(n, int), np.zeros(n, int),
            np.r_[np.ones(5), np.zeros(5)], base, np.tile([32.0, 31.0, 28.0], (n, 1)), prm)
    m0, d0 = M.project_arrays(*args, detail=True)
    m1, d1 = M.project_arrays(*args, detail=True, reserve_add=12.0)
    assert np.isclose(d1["reserve"], d0["reserve"] + 12.0)
    assert np.isclose(m0.sum() - m1.sum(), 12.0)
    prm.reserve_debut = [15.0, 1.0, 0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 30.0]
    p = pd.DataFrame({"tgs": [0, 0, 3, 40, 0, 0], "pre_new_tg": [10.0, np.nan, 10.0, 10.0, 100.0,
                                                                  30.0]})
    # winsorised input (cap 30: 100 -> 30) and the DEBUT_MAX ceiling
    assert np.allclose(M.debut_reserve(prm, p), [17.0, 15.0, 1.0, 0.1, 21.0, 21.0])
    prm.reserve_debut = [50.0, 0, 0, 0, 0, 0, 0, 0, 30.0]
    assert M.debut_reserve(prm, p)[0] == M.DEBUT_MAX
    assert "reserve_debut" in prm.to_json()
    assert "reserve_debut" not in M.default_params().to_json()
