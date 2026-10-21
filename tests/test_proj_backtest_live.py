"""T2.4 live-like slice mode of the backtest harness (proj/backtest.py): the live lock path on a
past day equals `pipeline.project_day` (the live code) on the same inputs, the DK-pool stand-in,
out-of-month P(play) params, the environment rule, the slice metrics / SEs and the monitor's
baseline file."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import availability as AV
from courtside.proj import backtest as B
from courtside.proj import copula as C
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import state as ST

from test_proj_pipeline import SEASON, _fill, _late_day


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


@pytest.fixture(scope="module")
def live(conn):
    return B.LiveInputs(conn, SEASON, "shipped", log=lambda m: None)


def _report(conn, day, n_out=1, n_q=1):
    """A fake official report frame for the first game of `day` (out + questionable)."""
    g = P._home_away(conn, day).iloc[0]
    pl = [tuple(r) for r in conn.execute(
        "SELECT player_id, team FROM player_games WHERE game_id=? ORDER BY min DESC",
        (g["game_id"],))]
    rows = [(pl[k][0], pl[k][1], "out", 0.0) for k in range(n_out)]
    rows += [(pl[n_out + k][0], pl[n_out + k][1], "questionable", 0.55) for k in range(n_q)]
    return pd.DataFrame({"player_id": [int(r[0]) for r in rows], "team": [r[1] for r in rows],
                         "game_id": [g["game_id"]] * len(rows), "status": [r[2] for r in rows],
                         "p_play": [r[3] for r in rows]})


# ----------------------------------------------------------------------- live path parity

def test_live_like_day_equals_the_live_project_day_path(conn, ctx, bundle, live, monkeypatch):
    """Same report, same DK pool (the stand-in passed as `extras`), same seed: live_like_day is
    bit-identical to pipeline.project_day, the code the lock runs execute."""
    day = _late_day(conn)
    rep = _report(conn, day)
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    seed = 12345
    res = B.live_like_day(conn, ctx, bundle, live, day, offset_min=45, n_sims=600, seed=seed)
    tip = AV.day_games(conn, day)["tip"].min()
    pool = live.pool_extras(live.team_at(day), P._home_away(conn, day))
    # the live path's drop_moved_players reads team_rosters: make it the stand-in for the day
    cur = live.team_at(day)
    ref_conn = sqlite3.connect(":memory:")
    conn.backup(ref_conn)
    ref_conn.row_factory = sqlite3.Row
    with ref_conn:
        ref_conn.execute("DELETE FROM team_rosters WHERE season=?", (SEASON,))
        ref_conn.executemany("INSERT INTO team_rosters (season, team_id, team, player_id) "
                             "VALUES (?,1,?,?)", [(SEASON, t, p) for p, t in cur.items()])
    ref = P.project_day(ref_conn, day, tip - timedelta(minutes=45), n_sims=600, seed=seed,
                        ctx=ctx, bundle=bundle, extras=pool)
    cols = ["game_id", "player_id", "p_play", "p_avail", "dk_mean", "dk_q10", "dk_q90",
            "min_mean", "min_played"]
    a = res.players[cols].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    b = ref.players[cols].sort_values(["game_id", "player_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    out_pid, q_pid = int(rep["player_id"][0]), int(rep["player_id"][1])
    st = dict(zip(res.players["player_id"], res.players["status"]))
    assert st[out_pid] == "out" and st[q_pid] == "questionable"
    j = res.players.index[res.players["player_id"] == out_pid][0]
    assert res.players.at[j, "dk_mean"] == 0
    assert res.meta["env_sources"] == {"closing": len(P._home_away(conn, day))}
    assert res.meta["n_listed"] == 2 and res.meta["n_forced_report"] == 1


def test_tip_groups_use_each_games_own_lock_and_keep_the_draws(conn, ctx, bundle, live,
                                                               monkeypatch):
    """Games with different tips read the report at their own tip - 45; splitting the day into
    tip groups does not change a game's draws (per-game seed sequences)."""
    day = _late_day(conn)
    games = AV.day_games(conn, day)
    assert len(games) >= 2
    seen = []
    real = AV.p_play_slate

    def spy(conn_, d, at, params=None, **k):
        seen.append(at)
        return real(conn_, d, at, params, **k)
    monkeypatch.setattr(AV, "p_play_slate", spy)
    one = B.live_like_day(conn, ctx, bundle, live, day, n_sims=400, seed=7)
    assert len(seen) == 1
    shifted = games.copy()
    shifted.loc[shifted.index[0], "tip"] = shifted["tip"].iloc[0] - timedelta(hours=3)
    monkeypatch.setattr(AV, "day_games", lambda c, d: shifted)
    seen.clear()
    two = B.live_like_day(conn, ctx, bundle, live, day, n_sims=400, seed=7)
    assert sorted(seen) == sorted({t - timedelta(minutes=45) for t in shifted["tip"]})
    assert two.meta["groups"] == 2
    cols = ["game_id", "player_id", "dk_mean", "min_mean"]
    pd.testing.assert_frame_equal(
        one.players[cols].sort_values(cols[:2]).reset_index(drop=True),
        two.players[cols].sort_values(cols[:2]).reset_index(drop=True), check_exact=True)


def test_lock10_uses_actual_starters_and_lock45_never_does(conn, ctx, bundle, live, monkeypatch):
    day = _late_day(conn)
    got = {}
    real = AV.p_play_slate

    def spy(conn_, d, at, params=None, *, starters=None, **k):
        got.setdefault("starters", []).append(starters)
        return real(conn_, d, at, params, starters=starters, **k)
    monkeypatch.setattr(AV, "p_play_slate", spy)
    B.live_like_day(conn, ctx, bundle, live, day, offset_min=45, n_sims=300)
    assert got["starters"] == [None]
    got.clear()
    res = B.live_like_day(conn, ctx, bundle, live, day, offset_min=10, n_sims=300,
                          use_starters=True)
    gids = list(P._home_away(conn, day)["game_id"])
    want = {int(r[0]) for r in conn.execute(
        f"SELECT player_id FROM player_starts WHERE game_id IN ({','.join('?' * len(gids))})",
        gids)}
    assert got["starters"][0] == want and len(want) == 10 * len(gids)
    assert res.meta["starters"] is True


# ----------------------------------------------------------------------- stand-in, params, env

def _bare_live(by: dict) -> B.LiveInputs:
    li = object.__new__(B.LiveInputs)
    li.by = by
    return li


def test_pool_stand_in_follows_trades_and_drops_players_who_never_play_again():
    li = _bare_live({1: (["2026-01-02", "2026-02-10"], ["AAA", "BBB"]),     # traded in Feb
                     2: (["2026-01-02"], ["AAA"]),                          # gone after Jan 2
                     3: (["2026-01-01", "2026-03-01"], ["CCC", "CCC"]),     # back from injury
                     4: (["2026-03-01"], ["DDD"])})                         # signed in March
    assert li.team_at("2026-01-02") == {1: "AAA", 2: "AAA", 3: "CCC"}
    assert li.team_at("2026-02-01") == {1: "BBB", 3: "CCC"}
    assert li.team_at("2026-02-10") == {1: "BBB", 3: "CCC"}
    assert li.team_at("2026-02-27") == {3: "CCC", 4: "DDD"}                 # 2 days before debut
    assert li.team_at("2026-03-02") == {}
    g = pd.DataFrame({"game_id": ["G1"], "home": ["BBB"], "away": ["CCC"]})
    assert B.LiveInputs.pool_extras(li.team_at("2026-02-01"), g) == {("G1", "BBB"): [1],
                                                                     ("G1", "CCC"): [3]}


def test_loo_month_params_leave_the_dates_month_out(monkeypatch):
    s = pd.DataFrame({"day": [date(2026, 1, 5)] * 30 + [date(2026, 2, 5)] * 50,
                      "season_type": "Regular Season", "game_id": [f"g{k}" for k in range(80)],
                      "status": "questionable", "bucket": "2h_6h", "reason": "injury",
                      "role": "starter", "played": [1] * 30 + [0] * 50})
    monkeypatch.setattr(AV, "build_samples", lambda *a, **k: s)
    li = _bare_live({})
    li.conn, li.pplay, li.log, li.hist, li._params, li._samples = None, "loo_month", \
        (lambda m: None), None, {}, None
    jan, feb = li.params(date(2026, 1, 20)), li.params("2026-02-01")
    assert jan.meta["held_out_month"] == "2026-01" and jan.meta["rows"] == 50
    assert feb.meta["held_out_month"] == "2026-02" and feb.meta["rows"] == 30
    # January is scored with February's outcomes only (all DNP) and vice versa
    assert jan.p("questionable", "2h_6h", "injury", "starter") < \
        feb.p("questionable", "2h_6h", "injury", "starter")
    assert li.params(date(2026, 1, 3)) is jan                      # cached per month


def test_environment_books_at_the_run_else_closing_never_the_replayed_gbm_market(conn):
    c = sqlite3.connect(":memory:")
    conn.backup(c)
    c.row_factory = sqlite3.Row
    day = _late_day(c)
    gid = P._home_away(c, day)["game_id"].iloc[0]
    at = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=20)
    with c:
        c.execute("INSERT INTO predictions (game_id, model, created_at, home_margin, total) "
                  "VALUES (?, 'gbm_market', ?, 30.0, 300.0)", (gid, "2000-01-01T00:00:00Z"))
    li = _bare_live({})
    li.conn = c
    e = li.environment(gid, at)
    assert e["source"] == "closing" and e["mu_total"] != 300.0
    with c:
        c.execute("INSERT INTO line_snapshots (game_id, captured_at, source, book, home_spread, "
                  "total) VALUES (?, ?, 'test', 'b1', -3.5, 221.5)",
                  (gid, (at - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")))
        c.execute("INSERT INTO line_snapshots (game_id, captured_at, source, book, home_spread, "
                  "total) VALUES (?, ?, 'test', 'b2', -9.5, 240.5)",
                  (gid, (at + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")))   # after
    e = li.environment(gid, at)
    assert e["source"] == "books:1" and e["mu_total"] == 221.5 and e["mu_margin"] == 3.5


def test_report_inputs_mirror_pipeline_report_avail(conn, monkeypatch):
    day = _late_day(conn)
    rep = _report(conn, day, n_out=2, n_q=2)
    rep.loc[len(rep)] = [999, rep["team"][0], rep["game_id"][0], "unlisted", 0.99]
    monkeypatch.setattr(AV, "p_play_slate", lambda *a, **k: rep)
    games = P._home_away(conn, day)
    pa, ex, _ = P.report_avail(conn, day, datetime.now(timezone.utc), games)
    pa2, ex2, status = B.report_inputs(rep)
    assert pa == pa2 and ex == ex2 and status[999] == "unlisted"


# ----------------------------------------------------------------------- slice metrics

def _frame(n_days=6, per_day=40, shift=0.0, seed=0, status=True):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_days):
        day = (date(2026, 1, 1) + timedelta(days=k)).isoformat()
        for j in range(per_day):
            y = float(rng.gamma(4, 6))
            mins = float(rng.choice([0.0, rng.uniform(5, 38)], p=[0.1, 0.9]))
            mean = y + rng.normal(0, 6) + shift
            mp = mins + rng.normal(0, 4) + shift if mins > 0 else rng.uniform(10, 30)
            pa = [0.0, 0.3, 0.6, 0.95, 1.0][j % 5]
            rows.append({"date": day, "season": "2025-26", "player_id": j, "game_id": f"g{k}_{j % 4}",
                         "team": "T", "y": y, "minutes": mins, "has_proj": True, "mean": mean,
                         "q10": mean - 10, "q25": mean - 4, "q50": mean, "q75": mean + 4,
                         "q90": mean + 10, "exp_min": max(mp * 0.9, 0), "p_play": pa * 0.9,
                         "tier": ["bench", "role", "star"][j % 3], "crps": abs(mean - y) * 0.7,
                         "pit": rng.random(), "min_played": mp, "p_avail": pa,
                         "status": (["out", "doubtful", "questionable", "available", "unlisted"]
                                    [j % 5] if status else None), "env_source": "closing"})
    return pd.DataFrame(rows)


def test_metric_values_match_hand_computation_and_diff_ses():
    a = _frame()
    d = B.slice_rows(a)
    u = a[a["exp_min"] >= 10]
    assert B.metric_se(d, "dk_mae")["value"] == pytest.approx((u["mean"] - u["y"]).abs().mean())
    m = a[(a["minutes"] > 0) & (a["min_played"] >= 15)]
    assert B.metric_se(d, "min_mae_played15")["value"] == pytest.approx(
        (m["min_played"] - m["minutes"]).abs().mean())
    q = a[(a["p_avail"] > 0.015) & (a["p_avail"] < 0.985)]
    assert B.metric_se(d, "pplay_cal_err")["value"] == pytest.approx(
        (q["p_play"] - (q["minutes"] > 0)).mean())
    c = B.metric_se(d, "cov80")
    assert 0 < c["value"] <= 1 and c["se"] > 0
    # identical frames: zero difference, zero SE; a constant shift of the projection
    z = B.metric_diff(d, d, "dk_bias")
    assert z["diff"] == pytest.approx(0) and z["se"] == pytest.approx(0, abs=1e-12)
    b = a.assign(mean=a["mean"] + 2.0)
    s = B.metric_diff(B.slice_rows(b), d, "dk_bias")
    assert s["diff"] == pytest.approx(2.0) and s["se"] == pytest.approx(0, abs=1e-9)
    # different populations: the SE is positive and of the size of the separate SEs
    w = B.metric_diff(B.slice_rows(_frame(seed=1, shift=1.0)), d, "dk_mae")
    se_a, se_b = B.metric_se(B.slice_rows(_frame(seed=1, shift=1.0)), "dk_mae")["se"], \
        B.metric_se(d, "dk_mae")["se"]
    assert 0.3 * (se_a + se_b) < w["se"] < 2 * (se_a + se_b)


def test_pplay_by_status_and_tier_tables():
    d = B.slice_rows(_frame())
    t = {r["status"]: r for r in B.pplay_by_status(d)}
    assert set(t) == {"out", "doubtful", "questionable", "available", "unlisted"}
    assert t["out"]["p_play"] == 0 and t["questionable"]["p_play"] == pytest.approx(0.54)
    assert t["doubtful"]["gap"] == pytest.approx(0.27 - t["doubtful"]["played"])
    tiers = {r["tier"]: r for r in B.tier_bias(d)}
    assert set(tiers) == {"bench", "role", "star"} and tiers["star"]["dk_bias"]["n"] > 0


def test_daily_and_window_ranges():
    d = B.slice_rows(_frame(n_days=12, per_day=60))
    dv = B.daily_values(d)
    assert len(dv) == 12 and (dv["n_dk_mae"] > 0).all()
    r = B.ranges(dv, B.DAILY_METRICS)
    assert r["dk_mae"]["p10"] <= r["dk_mae"]["p50"] <= r["dk_mae"]["p90"]
    assert r["dk_mae"]["n_days"] == 12 and r["dk_mae"]["rows_median"] > 0
    wv = B.window_values(d, n=100)
    assert wv["cov80"].isna().iloc[0] and wv["cov80"].notna().iloc[-1]   # needs 100 rows first


def test_slice_report_and_baseline_file():
    frames = {("pathA-v0.2", "avail"): _frame(seed=0, status=False),
              ("pathA-v0.2", "live"): _frame(seed=1, shift=0.5),
              ("pathA-v0.3", "avail"): _frame(seed=2, status=False),
              ("pathA-v0.3", "live"): _frame(seed=3, shift=0.4)}
    rep = B.slice_report(frames, base_arm="pathA-v0.2")
    assert rep["n_dates"] == 6
    assert set(rep["vs_reference"]) == {"pathA-v0.2|live", "pathA-v0.3|live"}
    assert set(rep["vs_base_arm"]) == {"pathA-v0.3|avail", "pathA-v0.3|live"}
    assert "pathA-v0.2|avail" not in rep["pplay_status"] and "pathA-v0.2|live" in rep["pplay_status"]
    v = rep["vs_reference"]["pathA-v0.2|live"]["min_mae_played15"]
    assert v["diff"] == pytest.approx(rep["cells"]["pathA-v0.2|live"]["min_mae_played15"]["value"]
                                      - rep["cells"]["pathA-v0.2|avail"]["min_mae_played15"]["value"])
    base = B.live_baseline_json(rep, arm="pathA-v0.2")
    assert base["min_mae"] == rep["cells"]["pathA-v0.2|live"]["min_mae_played15"]["value"]
    assert set(base["per_model"]) == {"pathA-v0.2", "pathA-v0.3"}
    for m in ("dk_mae", "cov80", "min_mae_played15", "pplay_cal_err"):
        assert {"p10", "p50", "p90", "rows_median"} <= set(base["daily"][m])
    txt = B.format_slice_report(rep)
    assert "loss vs avail" in txt and "P(play) by report status" in txt


# ----------------------------------------------------------------------- settings, CLI

def test_pre_t24_cache_keys_are_unchanged_and_pplay_is_keyed(monkeypatch):
    monkeypatch.setenv("COURTSIDE_BT_CODE_TAG", "x")
    from dataclasses import asdict
    st = B.Settings()
    s = asdict(st)
    s.pop("pplay")
    want = "v1-" + B._hash({"s": s, "model": P.MODEL_VERSION,
                            "bundle": __import__("hashlib").sha1(b"{}").hexdigest(),
                            "rules": "none", "code": "tag:x"})
    assert B.config_key(st, "{}", None) == want
    k1 = B.config_key(B.Settings(protocol="live", pplay="loo_month"), "{}", None)
    k2 = B.config_key(B.Settings(protocol="live", pplay="shipped"), "{}", None)
    assert k1 != k2
    old = asdict(B.Settings(protocol="live", pplay="loo_month"))
    assert k1 != "v1-" + B._hash({"s": old, "model": P.MODEL_VERSION,
                                  "bundle": __import__("hashlib").sha1(b"{}").hexdigest(),
                                  "rules": "none", "code": "tag:x"})   # stand-in v1 caches


def test_live_protocols_need_the_report_season_and_cli_has_the_slice(conn):
    with pytest.raises(ValueError):
        B.run(conn, ["2019-20"], protocol="live")
    with pytest.raises(ValueError):
        B.run(conn, ["2025-26"], protocol="live", pplay="bogus")
    import argparse
    ap = argparse.ArgumentParser()
    B.add_commands(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["backtest-dfs", "--seasons", "2025-26", "--protocol", "live10", "--slice",
                       "--pplay", "shipped"])
    assert a.protocol == "live10" and a.slice and a.pplay == "shipped"
