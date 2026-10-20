"""Extended context layer (Phase 2F second ablation: P + F2 + F3 + F6) behind pipeline.CONTEXT_EXT,
opt-in on top of CONTEXT_ON: own bundle suffix / MODEL_VERSION, P + F2 ON behaviour unchanged
with the extension off, multipliers inside the clip with no NaN, no lookahead through the
tracking tables, walk-forward bundle fit, JSON round trip."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import context as CX
from courtside.proj import copula as C
from courtside.proj import features as FX
from courtside.proj import minutes as M
from courtside.proj import pipeline as P
from courtside.proj import state as ST
from courtside.proj.features.common import DK_STATS

from test_proj_context_live import _specs, _team_rates, _with
from test_proj_features2 import add_tracking
from test_proj_pipeline import SEASON, _copy, _fill, _late_day

SMALL = {"min_data_in_leaf": 50, "min_sum_hessian_in_leaf": 1.0, "num_threads": 1}
N_SIMS = 1000


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    add_tracking(c)
    return c


@pytest.fixture(scope="module")
def base_bundle():
    return P.Bundle(season=SEASON, minutes=M.default_params(SEASON), state=ST.default_params(),
                    copula=C.default_copula_params(SEASON))


@pytest.fixture(scope="module")
def ext_layer(conn):
    """The shipped extended layer (P + F2 + F3 + F6) fit on seasons < S with outcomes planted
    on the F3 pass and defensive rebound-chance rates, so it acts on the F3 columns."""
    sp = ST.default_params()
    fr = FX.build_frame(conn, SEASON, sp, n_train=6, cache=False)
    fr = FX.extend_frame(conn, fr, ["F3", "F6"], cache=False)
    rng = np.random.default_rng(1)
    z = np.nan_to_num(fr["t3_pass"].to_numpy() + fr["t3_rbc_d"].to_numpy())
    for s in DK_STATS:
        fr[s] = rng.poisson(fr[f"E_{s}"].to_numpy() * np.exp(np.clip(4.0 * z, -1.0, 1.0)))
    return CX.fit_season(conn, SEASON, sp, groups=CX.LIVE_GROUPS_EXT, frame=fr, params=SMALL)


@pytest.fixture
def switch():
    was, was_ext = P.CONTEXT_ON, P.CONTEXT_EXT
    yield P.set_context
    P.set_context(was, was_ext)


def test_ext_versions(switch):
    assert P.CONTEXT_EXT_DEFAULT is False
    assert P._versions(True, True) == ("_v1_cx2", "pathA-v0.4")
    assert P._versions(True, False) == ("_v1_cx", "pathA-v0.3")
    assert P._versions(False, True) == ("_v1", "pathA-v0.2")      # ext alone does nothing
    switch(True, True)
    assert P._VS == "_v1_cx2" and P.MODEL_VERSION == "pathA-v0.4"
    assert P.context_groups() == CX.LIVE_GROUPS_EXT == ("P", "F2", "F3", "F6")
    switch(True)                                   # ext not given: unchanged
    assert P.MODEL_VERSION == "pathA-v0.4"
    switch(True, False)
    assert P._VS == "_v1_cx" and P.MODEL_VERSION == "pathA-v0.3"
    assert P.context_groups() == ("P", "F2")
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE


def test_p_f2_on_never_builds_extras(conn, base_bundle, switch, monkeypatch):
    """ON with a P + F2 layer: the extension path never runs (same code path as pathA-v0.3)."""
    sp = ST.default_params()
    fr = FX.build_frame(conn, SEASON, sp, n_train=6, cache=False)
    pf2 = CX.fit_season(conn, SEASON, sp, frame=fr, params=SMALL)

    def boom(*a, **k):
        raise AssertionError("extension inputs built for a P + F2 layer")
    monkeypatch.setattr(FX, "LiveExtras", boom)
    monkeypatch.setattr(P.Context, "context_extras", boom)
    switch(True, False)
    day = _late_day(conn)
    on = _team_rates(_specs(conn, P.Context(conn, SEASON), _with(base_bundle, pf2), day))
    assert np.isfinite(on["r_dk"]).all()


def test_ext_multipliers_within_clip(conn, base_bundle, ext_layer, switch):
    switch(False)
    day = _late_day(conn)
    b = _with(base_bundle, ext_layer)
    off = _team_rates(_specs(conn, P.Context(conn, SEASON), b, day))
    switch(True, True)
    on = _team_rates(_specs(conn, P.Context(conn, SEASON), b, day))
    assert np.array_equal(on["player_id"], off["player_id"])
    lo, hi = CX.CLIP
    moved = 0.0
    for s in DK_STATS:
        r = on[f"r_{s}"].to_numpy() / off[f"r_{s}"].to_numpy()
        assert np.isfinite(r).all(), s
        assert np.all(r >= lo - 1e-12) and np.all(r <= hi + 1e-12), s
        moved = max(moved, float(np.abs(np.log(r)).max()))
    assert moved > 0.1
    for c in ("mpg", "tier_state", "p_avail", "star"):
        assert np.array_equal(on[c].to_numpy(), off[c].to_numpy()), c


def test_ext_no_lookahead(conn, base_bundle, ext_layer, switch):
    """ON + ext: deleting the day's and later box scores, tracking, hustle and matchup rows
    changes nothing."""
    switch(True, True)
    day = _late_day(conn)
    b = _with(base_bundle, ext_layer)
    games = P._home_away(conn, day)

    def run(c):
        return P.project_games(c, day, games, ctx=P.Context(c, SEASON), bundle=b,
                               n_sims=N_SIMS, seed=42, env_mode="closing")
    full = run(conn)
    cut = _copy(conn)
    with cut:
        later = "SELECT game_id FROM team_games WHERE game_date >= ?"
        for t in ("player_tracking", "player_hustle", "player_matchups"):
            cut.execute(f"DELETE FROM {t} WHERE game_id IN ({later})", (day.isoformat(),))
        for t in ("player_games", "team_games"):
            cut.execute(f"DELETE FROM {t} WHERE game_date >= ?", (day.isoformat(),))
        cut.execute("DELETE FROM player_starts WHERE game_id NOT IN "
                    "(SELECT game_id FROM team_games)")
    trunc = run(cut)
    for a, c in zip(full.sims, trunc.sims):
        assert np.array_equal(a.fp, c.fp)


def test_ext_bundle_walk_forward(conn, base_bundle, switch, tmp_path, monkeypatch):
    """load_bundle with ON + ext: the OFF bundle plus a P + F2 + F3 + F6 layer fit on seasons < S,
    written as bundle_<S>_v1_cx2.json (the P + F2 file untouched); JSON round trip."""
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path))
    monkeypatch.setattr(P, "CONTEXT_FIT_PARAMS", dict(SMALL))
    switch(False)
    P.cache_dir().mkdir(parents=True, exist_ok=True)
    (P.cache_dir() / f"bundle_{SEASON}_v1.json").write_text(base_bundle.to_json())
    switch(True, True)
    b = P.load_bundle(conn, SEASON)
    assert (P.cache_dir() / f"bundle_{SEASON}_v1_cx2.json").exists()
    assert not (P.cache_dir() / f"bundle_{SEASON}_v1_cx.json").exists()
    assert b.context.groups == list(CX.LIVE_GROUPS_EXT)
    assert all(s < SEASON for s in b.context.train_seasons)
    d = json.loads(b.to_json())
    assert d["model_version"] == "pathA-v0.4"
    assert P.Bundle.from_json(b.to_json()).to_json() == b.to_json()


def test_f6_day_hook_matches_table_and_needs_no_result(conn):
    """F6 per day (the pipeline hook) equals the full-history table for completed games, and is
    unchanged when the day's and later games are not in team_games yet (the live case)."""
    from courtside.proj.features import schedule as F6
    day = _late_day(conn)
    tg = conn.execute("SELECT game_id, team, opp FROM team_games WHERE game_date = ? AND "
                      "is_home = 1", (day.isoformat(),)).fetchall()
    games = pd.DataFrame([dict(game_id=r[0], home=r[1], away=r[2])
                          for r in tg])
    full = F6.build(conn).set_index(["game_id", "team"])
    got = F6.History(conn).day_features(conn, day, games).set_index(["game_id", "team"])
    ref = full.loc[got.index, list(F6.TEAM_COLS)]
    np.testing.assert_allclose(got[list(F6.TEAM_COLS)].to_numpy(float), ref.to_numpy(float))
    cut = _copy(conn)
    with cut:
        cut.execute("DELETE FROM team_games WHERE game_date >= ?", (day.isoformat(),))
    live = F6.History(cut).day_features(cut, day, games).set_index(["game_id", "team"])
    np.testing.assert_allclose(live.loc[got.index, list(F6.TEAM_COLS)].to_numpy(float),
                               ref.to_numpy(float))
    assert np.isfinite(live[list(F6.TEAM_COLS)].to_numpy(float)).all()
