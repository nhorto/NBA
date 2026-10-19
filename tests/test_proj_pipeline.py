"""Slate projection pipeline (T1.4) on a synthetic league in a migrated in-memory DB:
reconciliation, absent players, cond all vs played, seeds, no lookahead, store round trip,
project_slate keyed by DK playerId."""

from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import availability as AV
from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC
from courtside.proj import state as ST
from courtside.proj import store as SO

from test_proj_minutes import _league

SEASON = "2022-23"


def _fill(conn: sqlite3.Connection, seed: int = 0) -> None:
    pg, tg, pos = _league(seed=seed)
    rng = np.random.default_rng(seed)
    home = tg.groupby("game_id").head(1).set_index("game_id")["team"]
    opp = tg.merge(tg[["game_id", "team"]], on="game_id", suffixes=("", "_o"))
    opp = opp[opp["team"] != opp["team_o"]].set_index(["game_id", "team"])["team_o"]
    conn.executemany(
        "INSERT INTO team_games (game_id, team, season, season_type, game_date, opp, is_home, "
        "min, pts) VALUES (?,?,?,?,?,?,?,?,?)",
        [(r.game_id, r.team, r.season, r.season_type, r.game_date, opp[(r.game_id, r.team)],
          int(home[r.game_id] == r.team), r.min, int(r.pts)) for r in tg.itertuples()])
    rows = []
    for r in pg.itertuples():
        m = float(r.min)
        reb = int(rng.poisson(0.18 * m))
        oreb = reb // 3
        pts = int(rng.poisson(0.45 * m))
        rows.append((r.game_id, int(r.player_id), f"P{r.player_id}", r.team, r.season,
                     r.season_type, r.game_date, m, int(0.38 * m), int(rng.poisson(0.06 * m)),
                     int(0.11 * m), oreb, reb - oreb, reb, int(rng.poisson(0.1 * m)),
                     int(rng.poisson(0.03 * m)), int(rng.poisson(0.02 * m)),
                     int(rng.poisson(0.06 * m)), int(r.pf), pts))
    conn.executemany(
        "INSERT INTO player_games (game_id, player_id, player_name, team, season, season_type, "
        "game_date, min, fga, fg3m, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, pts) VALUES "
        "(" + ",".join("?" * 20) + ")", rows)
    conn.executemany("INSERT INTO player_starts (game_id, player_id, team, source) VALUES "
                     "(?,?,?,'synthetic')",
                     [(r.game_id, int(r.player_id), r.team) for r in pg.itertuples()
                      if r.started])
    seasons = sorted(tg["season"].unique())
    team_of = pg.drop_duplicates("player_id").set_index("player_id")["team"]
    conn.executemany(
        "INSERT INTO team_rosters (season, team_id, team, player_id, position) VALUES "
        "(?,?,?,?,?)",
        [(s, 1, team_of[int(p.player_id)], int(p.player_id), "GFC"[int(p.g1)])
         for s in seasons for p in pos.itertuples()])
    g = tg.groupby("game_id").agg(total=("pts", "sum"))
    conn.executemany("INSERT INTO closing_lines (game_id, source, home_spread_close, "
                     "total_close) VALUES (?, 'synthetic', ?, ?)",
                     [(gid, float(rng.normal(-2, 4)), float(r.total + rng.normal(0, 10)))
                      for gid, r in g.iterrows()])
    conn.commit()


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    return c


@pytest.fixture(scope="module")
def bundle():
    return P.Bundle(season=SEASON, minutes=M.default_params(SEASON), state=ST.default_params(),
                    copula=C.default_copula_params(SEASON))


@pytest.fixture(scope="module")
def ctx(conn):
    return P.Context(conn, SEASON)


def _late_day(conn) -> date:
    d = conn.execute("SELECT game_date FROM team_games WHERE season=? GROUP BY game_date "
                     "ORDER BY game_date DESC LIMIT 1 OFFSET 3", (SEASON,)).fetchone()[0]
    return date.fromisoformat(d)


@pytest.fixture(scope="module")
def hist(conn, ctx, bundle):
    return P.project_historical(conn, _late_day(conn), n_sims=2000, ctx=ctx, bundle=bundle)


# ------------------------------------------------------------------------- reconciliation

def test_reconciliation_within_half_point(hist):
    assert hist.teams and hist.meta["recon"]["ok"]
    for t in hist.teams:
        assert abs(t["post"] - t["target"]) <= RC.TOL
    pl = hist.players
    for t in hist.teams:
        s = pl[(pl["game_id"] == t["game_id"]) & (pl["team"] == t["team"])]["pts_all"].sum()
        assert s == pytest.approx(t["target"], abs=RC.TOL)


def test_absent_player_all_zero_sims(conn, ctx, hist):
    day = hist.day
    _, panel, _ = P.historical_inputs(conn, ctx, day, protocol="avail")
    absent = panel.loc[panel["absent"], ["game_id", "player_id"]]
    assert len(absent) > 0, "the synthetic league has ~8% absences"
    for g in hist.sims:
        for pid in absent.loc[absent["game_id"] == g.game_id, "player_id"]:
            j = int(np.flatnonzero(g.player_id == pid)[0])
            assert (g.fp[:, j] == 0).all() and (g.minutes[:, j] == 0).all()
            assert g.info.at[j, "p_play"] == 0
    pr = hist.projections()
    for pid in absent["player_id"]:
        assert pr[int(pid)].mean == 0 and pr[int(pid)].p90 == 0


def test_cond_all_vs_played_consistency(hist):
    pl = hist.players
    act = pl[pl["p_play"] > 0]
    assert np.allclose(act["dk_mean"], act["p_play"] * act["dkp_mean"], rtol=1e-6, atol=1e-6)
    assert np.allclose(act["min_mean"], act["p_play"] * act["min_played"], rtol=1e-6, atol=1e-6)
    qs = ["dk_q05", "dk_q10", "dk_q25", "dk_q50", "dk_q75", "dk_q90", "dk_q95"]
    assert (np.diff(pl[qs].to_numpy(), axis=1) >= -1e-9).all()
    # 'all' quantiles never exceed the 'played' ones (mixture with a point mass at 0 <= FP)
    sure = act[act["p_play"] < 0.97]
    assert (sure["dk_q25"] <= sure["dkp_q25"] + 1e-6).all()
    pr = hist.projections()
    r = pl.iloc[0]
    p = pr[int(r["player_id"])]
    assert p.mean == pytest.approx(r["dk_mean"], abs=1e-3)
    assert p.p25 == pytest.approx(r["dk_q25"], abs=1e-3)
    assert p.p10 <= p.p25 <= p.p50 <= p.p90
    st = hist.stats
    assert set(st["stat"]) == {"dk", "min", "pts", "reb", "ast", "fg3m"}
    assert set(st["cond"]) == {"played", "all"}


# ------------------------------------------------------------------------- seeds

def test_seeds_reproducible(conn, ctx, bundle, hist):
    again = P.project_historical(conn, hist.day, n_sims=2000, ctx=ctx, bundle=bundle)
    for a, b in zip(hist.sims, again.sims):
        assert np.array_equal(a.fp, b.fp)
    other = P.project_historical(conn, hist.day, n_sims=2000, ctx=ctx, bundle=bundle, seed=7)
    assert not np.array_equal(hist.sims[0].fp, other.sims[0].fp)
    import hashlib
    h = int.from_bytes(hashlib.sha256(b"154575|lock45|" + P.MODEL_VERSION.encode())
                       .digest()[:8], "big") >> 1
    assert P.stable_seed("154575", "lock45", P.MODEL_VERSION) == h
    # a game's sims do not depend on the other games of the slate
    one = P.project_historical(conn, hist.day, n_sims=2000, ctx=ctx, bundle=bundle)
    g0 = hist.sims[0].game_id
    seed = P.stable_seed(hist.day.isoformat(), "hist-avail", P.MODEL_VERSION)
    games, panel, pa = P.historical_inputs(conn, ctx, hist.day, protocol="avail")
    env = {g0: P.environment(conn, g0, None, mode="closing")}
    specs = P.build_games(ctx, bundle, conn, hist.day, games[games["game_id"] == g0],
                          panel[panel["game_id"] == g0], pa, env)
    solo = P.run_games(specs, bundle, 2000, seed, hist.day)
    assert np.array_equal(solo.sims[0].fp, one.sims[0].fp)


# ------------------------------------------------------------------------- no lookahead

def _copy(conn) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    conn.backup(c)
    c.row_factory = sqlite3.Row
    return c


def test_no_lookahead(conn, bundle):
    day = _late_day(conn)
    games = P._home_away(conn, day)
    rng = np.random.default_rng(0)
    pids = [r[0] for r in conn.execute("SELECT DISTINCT player_id FROM player_games WHERE "
                                       "season=?", (SEASON,))]
    p_avail = {int(p): float(v) for p, v in zip(pids, rng.choice([0.0, 0.5, 1.0], len(pids),
                                                                  p=[.1, .2, .7]))}

    def run(c):
        ctx = P.Context(c, SEASON)
        return P.project_games(c, day, games, ctx=ctx, bundle=bundle, n_sims=1500, seed=42,
                               p_avail=p_avail, env_mode="closing")

    full = run(conn)
    cut = _copy(conn)
    with cut:
        for t in ("player_games", "team_games"):
            cut.execute(f"DELETE FROM {t} WHERE game_date >= ?", (day.isoformat(),))
        cut.execute("DELETE FROM player_starts WHERE game_id NOT IN "
                    "(SELECT game_id FROM team_games)")
        # scramble nothing before the day; closing lines of the day's games are inputs
    trunc = run(cut)
    assert len(full.sims) == len(trunc.sims) == len(games)
    for a, b in zip(full.sims, trunc.sims):
        assert np.array_equal(a.player_id, b.player_id)
        assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    # and the scenario draw used P(available): some players sat in some sims
    assert any(((g.minutes == 0).mean(0) > 0.3).any() for g in full.sims)


# ------------------------------------------------------------------------- store

def test_store_round_trip(conn, hist, tmp_path):
    c = _copy(conn)
    info = P.save_run(c, hist, run_id="run-a", slate_id="999", kind="manual", seconds=1.0)
    back = SO.read_projections(c, "run-a")
    assert len(back) == len(hist.players)
    m = back.merge(hist.players, on=["game_id", "player_id"])
    assert np.allclose(m["dk_all_mean"], m["dk_mean"])
    assert np.allclose(m["dk_all_q25"], m["dk_q25"])
    assert np.allclose(m["min_played_mean"].fillna(-1), m["min_played"].where(
        m["p_play_x"] > 0, -1).fillna(-1))
    assert np.allclose(m["pts_all_mean"], m["pts_all"])
    assert m["pts_all_sd"].isna().all()
    sims = SO.load_sims(info["sim_path"])
    pid, fp = hist.fp_matrix()
    assert sims["fp"].dtype == np.float16 and sims["fp"].shape == fp.shape
    assert np.array_equal(sims["player_id"], pid)
    assert np.allclose(sims["fp"].astype(np.float32), fp, atol=0.05, rtol=2e-3)
    run = c.execute("SELECT sim_path, model_version, path FROM dfs_runs WHERE run_id='run-a'"
                    ).fetchone()
    assert run[0] == info["sim_path"] and run[1] == P.MODEL_VERSION and run[2] == "A"
    # idempotent re-write
    P.save_run(c, hist, run_id="run-a", slate_id="999", kind="manual")
    n = c.execute("SELECT COUNT(*) FROM player_projections WHERE run_id='run-a'").fetchone()[0]
    assert n == len(hist.players)
    ns = c.execute("SELECT COUNT(*) FROM player_projection_stats").fetchone()[0]
    assert ns == len(hist.stats)
    # prune: files older than 8 days go
    assert SO.prune_sims(now=time.time() + 9 * 86400) >= 1
    assert not SO.sims_dir().joinpath("999").exists()


# ------------------------------------------------------------------------- project_slate

def test_project_slate_keyed_by_dk_id(conn, ctx, bundle, monkeypatch):
    c = _copy(conn)
    day = _late_day(c)
    games = P._home_away(c, day)
    g = games.iloc[0]
    pl = [r for r in c.execute("SELECT player_id, team FROM player_games WHERE game_id=? ",
                               (g["game_id"],))]
    sid = "777"
    tip = "2023-01-01T00:00:00Z"
    with c:
        c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, game_ids_json) "
                  "VALUES (?, ?, 70, 'main', ?)", (sid, day.isoformat(),
                                                    f'["{g["game_id"]}"]'))
        for k, (pid, team) in enumerate(pl):
            c.execute("INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                      "game_type_id, draftable_id, player_name, site_player_id, team, positions,"
                      " roster_slot, salary, game_id, home, away, start_time, disabled) VALUES "
                      "('2023-01-01T00:00:00Z', 'dk', ?, 'Classic', 70, ?, ?, ?, ?, 'SF', "
                      "'UTIL', 5000, ?, ?, ?, ?, 0)",
                      (sid, f"d{k}", f"P{pid}", f"dk{pid}", team, g["game_id"], g["home"],
                       g["away"], tip))
            c.execute("INSERT INTO player_ids (source, source_id, player_id) VALUES "
                      "('dk', ?, ?)", (f"dk{pid}", int(pid)))
    out_pid, q_pid = int(pl[0][0]), int(pl[1][0])
    rep = pd.DataFrame({"player_id": [out_pid, q_pid], "team": [pl[0][1], pl[1][1]],
                        "game_id": [g["game_id"]] * 2, "status": ["out", "questionable"],
                        "p_play": [0.0, 0.5]})
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=20)
    out = P.project_slate(c, sid, at, run_id="r-slate", n_sims=1500, ctx=ctx, bundle=bundle)
    assert out and all(isinstance(k, str) and k.startswith("dk") for k in out)
    assert out[f"dk{out_pid}"].mean == 0
    assert 0 < out[f"dk{q_pid}"].mean
    res = out.result
    j = res.players.index[res.players["player_id"] == q_pid][0]
    assert 0.35 < res.players.at[j, "p_play"] < 0.55
    assert out.meta["sim_path"].endswith("r-slate.npz")
    sims = SO.load_sims(out.meta["sim_path"])
    assert set(sims["dk_id"]) >= {f"dk{out_pid}", f"dk{q_pid}"}
    assert c.execute("SELECT COUNT(*) FROM player_projections WHERE run_id='r-slate' AND "
                     "slate_id='777'").fetchone()[0] == len(res.players)
    # same inputs + seed -> identical projections (common random numbers)
    again = P.project_slate(c, sid, at, run_id="r-slate-2", n_sims=1500, ctx=ctx,
                            bundle=bundle, kind="manual", store=False)
    assert {k: v.mean for k, v in again.items()} == {k: v.mean for k, v in out.items()}


def test_fit_copula_smoke(conn):
    cp = C.fit_copula(conn, SEASON, ST.default_params(), n_seasons=2)
    assert 0 <= cp.alpha <= 1.5 and len(cp.cv) == len(cp.cv_mu) >= 3
    assert np.all(np.diff(cp.cv) <= 1e-12)
    assert set(cp.rho_team) == {C.pair_key(a, b) for a in C.TIERS for b in C.TIERS}
    assert all(-1 < v < 1 for v in cp.rho_team.values()) and -1 < cp.rho_opp < 1
    env = C.fit_env(conn, SEASON)
    assert env["env_seasons"] and all(s < SEASON for s in env["env_seasons"])
    assert C.CopulaParams.from_json(cp.to_json()) == cp


def test_slate_day_is_the_eastern_day_of_the_earliest_tip():
    from types import SimpleNamespace as NS
    from datetime import date
    from courtside.proj import pipeline
    pool = NS(players=[NS(start_time="2026-10-21T01:30:00.0000000Z"),
                       NS(start_time="2026-10-20T19:00:00.0000000Z")])
    assert pipeline._slate_day(pool) == date(2026, 10, 20)
    late_only = NS(players=[NS(start_time="2026-10-21T01:30:00.0000000Z")])
    assert pipeline._slate_day(late_only) == date(2026, 10, 20)
