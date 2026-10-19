"""Opening-week live-path check (proj/live_path_eval.py) on a tiny synthetic league: no DB.

The league is test_proj_minutes' four-team league with a mover (BOS -> NYK in 2022-23) and a
retiree, so the live-style candidate build (departed dropped, season roster forced in) is
exercised; v0.5 and v1 are fitted on 2019-22 and scored on the first 5 days of 2022-23."""

import numpy as np
import pandas as pd
import pytest

from courtside.proj import live_path_eval as L
from courtside.proj import minutes as M
from test_proj_minutes import V1_KW, _league, _stints

SEASON = "2022-23"
MOVER, RETIREE, STAYER = 1000, 1009, 1001


@pytest.fixture(scope="module")
def world():
    pg, tg, pos = _league()
    pg = pg.copy()
    pg.loc[(pg["player_id"] == MOVER) & (pg["season"] >= SEASON), "team"] = "NYK"
    pg = pg[~((pg["player_id"] == RETIREE) & (pg["season"] >= SEASON))]
    h = M.build_history(pg, tg, pos, stints=_stints())
    shares = L.default_shares(h, SEASON, window=3)
    prm05 = M.fit(h, SEASON, window=3, log=lambda s: None, f_grid=(0.6,))
    prm1 = M.fit(h, SEASON, window=3, log=lambda s: None, f_grid=(0.6,), version="v1",
                 v1_kw=V1_KW)
    return h, shares, {"v0.5": prm05, "v1": prm1}


def test_live_style_candidates(world):
    h, _, _ = world
    p = L.opening_panel(h, SEASON, ndays=5)
    assert 8 <= p["key"].nunique() <= 24
    assert p["date"].nunique() <= 5
    bos = p[p["team"] == "BOS"]
    nyk = p[p["team"] == "NYK"]
    # the mover is dropped from his old team's candidates and forced onto his new one
    assert MOVER not in set(bos["player_id"]) and MOVER in set(nyk["player_id"])
    first = nyk[nyk["player_id"] == MOVER].sort_values("date").iloc[0]
    assert first["roster"] == 0 and first["last_date"] == first["last_date"]   # forced extra
    # the retiree (no appearance this season) is gone everywhere; a stayer is a candidate
    assert RETIREE not in set(p["player_id"]) and STAYER in set(bos["player_id"])
    # actual availability: only regulars who did not play are absent
    assert not p.loc[p["played"], "absent"].any()
    assert p["act_min"].notna().sum() == p["played"].sum() > 0


def test_opening_week_live_path_invariants(world):
    """Fails if the live-path team minutes leave 240 - reserve by more than 2, or the shipped
    version's opening-week bias leaves +-1.5 minutes on the synthetic league."""
    h, shares, prms = world
    res = {"seasons": {SEASON: L.evaluate_season(h, SEASON, prms, shares,
                                                 pd.Series(dtype=float), ndays=5)}}
    for v, x in res["seasons"][SEASON]["versions"].items():
        assert x["n"] > 50, v
        t = x["team"]
        assert t["dev_max_abs"] <= L.MAX_TEAM_DEV, (v, t)
        assert abs(t["sum_mean"] - t["target_mean"]) <= L.MAX_TEAM_DEV, (v, t)
        assert t["reserve_mean"] < 15.0, (v, t)
        assert np.isfinite(x["mae"]) and x["mae"] < 8.0, (v, x["mae"])
        assert abs(x["bias"]) <= L.MAX_BIAS, (v, x["bias"], x["by_tier"])
    assert L.check(res) == []
    assert "PASS" in L.format_report({**res, "ndays": 5, "violations": []})
    # the check really trips: a too-tight bound is reported
    bad = L.check(res, max_team_dev=-1.0, max_bias=0.0)
    assert any("team minutes" in b for b in bad) and any("bias" in b for b in bad)


def test_check_flags_a_biased_or_short_team():
    base = {"n": 100, "mae": 5.0, "bias": 0.1, "by_tier": {},
            "team": {"dev_max_abs": 0.5, "sum_mean": 234.0, "target_mean": 234.0,
                     "reserve_mean": 6.0}}
    ok = {"seasons": {"S": {"versions": {"v1": base}}}}
    assert L.check(ok) == []
    short = {"seasons": {"S": {"versions": {"v1": {**base, "team": {**base["team"],
                                                                    "dev_max_abs": 25.0}}}}}}
    assert "team minutes off 240 - reserve" in L.check(short)[0]
    biased = {"seasons": {"S": {"versions": {"v1": {**base, "bias": 4.4}}}}}
    assert "bias +4.40" in L.check(biased)[0]
    # v0.5 is reference only (known +4 minute bias at the opener)
    ref = {"seasons": {"S": {"versions": {"v0.5": {**base, "bias": 4.4}}}}}
    assert L.check(ref) == []


def test_cli_registers():
    import argparse
    ap = argparse.ArgumentParser()
    handlers = L.add_commands(ap.add_subparsers(dest="c"))
    assert set(handlers) == {"live-path-eval"}
    a = ap.parse_args(["live-path-eval", "--seasons", "2024-25", "--days", "3"])
    assert a.seasons == ["2024-25"] and a.days == 3
