"""Context layer wired into the live pipeline (Phase 2F P + F2, behind pipeline.CONTEXT_ON):
switch OFF = bit-identical projections and bundle bytes; switch ON = multipliers inside the clip,
walk-forward bundle fit (seasons < S only), JSON round trip of the bundle with the layer, own
bundle file / MODEL_VERSION, common random numbers with OFF."""

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
from courtside.proj import reconcile as RC
from courtside.proj import state as ST
from courtside.proj.features.common import DK_STATS

from test_proj_pipeline import SEASON, _copy, _fill, _late_day

SMALL = {"min_data_in_leaf": 50, "min_sum_hessian_in_leaf": 1.0, "num_threads": 1}
N_SIMS = 1500


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c)
    return c


@pytest.fixture(scope="module")
def base_bundle():
    return P.Bundle(season=SEASON, minutes=M.default_params(SEASON), state=ST.default_params(),
                    copula=C.default_copula_params(SEASON))


@pytest.fixture(scope="module")
def strong(conn):
    """A P+F2 layer fit on seasons < S of the league's own feature frame with outcomes planted
    far from the EB expectation (x e^{+-2} by minutes state), so its raw multipliers go well past
    the clip on both sides."""
    sp = ST.default_params()
    fr = FX.build_frame(conn, SEASON, sp, n_train=6, cache=False)
    fr = fr.copy()
    rng = np.random.default_rng(0)
    z = (fr["p_mpg"] - fr["p_mpg"].median()) / fr["p_mpg"].std()
    for s in DK_STATS:
        fr[s] = rng.poisson(fr[f"E_{s}"].to_numpy() * np.exp(2.0 * np.clip(z, -1.5, 1.5)))
    return CX.fit_season(conn, SEASON, sp, frame=fr, params=SMALL)


@pytest.fixture
def switch():
    """Set the context switch for one test and restore the import-time state after it."""
    was = P.CONTEXT_ON
    yield P.set_context
    P.set_context(was)


def _with(bundle, context):
    return P.Bundle(season=bundle.season, minutes=bundle.minutes, state=bundle.state,
                    copula=bundle.copula, context=context)


def _specs(conn, ctx, bundle, day):
    games, panel, pa = P.historical_inputs(conn, ctx, day, protocol="avail")
    env = {g: P.environment(conn, g, None, mode="closing") for g in games["game_id"]}
    return P.build_games(ctx, bundle, conn, day, games, panel, pa, env)


def _team_rates(specs) -> pd.DataFrame:
    rows = []
    for gs in specs:
        for ts in (gs.home, gs.away):
            d = ts.players.copy()
            d["game_id"], d["team"] = gs.game_id, ts.team
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------- switch OFF

def test_context_cli_bundle(conn, base_bundle, switch, monkeypatch, capsys):
    """`python -m courtside.proj.context bundle --season S` writes the ON bundle and leaves the
    switch as it was."""
    switch(False)
    P.cache_dir().mkdir(parents=True, exist_ok=True)
    (P.cache_dir() / f"bundle_{SEASON}{P._VS}.json").write_text(base_bundle.to_json())
    monkeypatch.setattr(CX, "_conn", lambda path: conn)
    assert CX.main(["bundle", "--season", SEASON]) == 0
    assert "pathA-v0.3" in capsys.readouterr().out
    assert not P.CONTEXT_ON and P.MODEL_VERSION == "pathA-v0.2"
    b = P.Bundle.from_json((P.cache_dir() / f"bundle_{SEASON}_v1_cx.json").read_text())
    assert b.context is not None and all(s < SEASON for s in b.context.train_seasons)


def test_default_is_off():
    assert P._versions(False) == (P._VS_BASE, P.MODEL_VERSION_BASE)
    assert P.MODEL_VERSION_BASE == "pathA-v0.2" and P._VS_BASE == "_v1"


def test_off_is_bit_identical(conn, base_bundle, strong, switch, monkeypatch):
    """Switch OFF: a bundle carrying a layer projects exactly like one without (the hook never
    runs), and the seed / sims / summaries are unchanged."""
    switch(False)
    ctx = P.Context(conn, SEASON)
    assert ctx.cx_t is None                                  # no per-season context setup
    day = _late_day(conn)
    ref = P.project_historical(conn, day, n_sims=N_SIMS, ctx=ctx, bundle=base_bundle)

    def boom(*a, **k):
        raise AssertionError("context hook ran with the switch off")
    monkeypatch.setattr(P, "apply_context", boom)
    monkeypatch.setattr(CX, "apply_rates", boom)
    monkeypatch.setattr(FX, "day_features", boom)
    got = P.project_historical(conn, day, n_sims=N_SIMS, ctx=ctx,
                               bundle=_with(base_bundle, strong))
    assert got.meta["seed"] == ref.meta["seed"]
    assert len(got.sims) == len(ref.sims)
    for a, b in zip(ref.sims, got.sims):
        assert np.array_equal(a.player_id, b.player_id)
        assert np.array_equal(a.fp, b.fp) and np.array_equal(a.minutes, b.minutes)
    pd.testing.assert_frame_equal(ref.players, got.players, check_exact=True)
    pd.testing.assert_frame_equal(ref.stats, got.stats, check_exact=True)


def test_off_bundle_json_is_the_pre_context_format(base_bundle, switch):
    switch(False)
    legacy = json.dumps({"season": base_bundle.season,
                         "minutes": json.loads(base_bundle.minutes.to_json()),
                         "state": base_bundle.state,
                         "copula": json.loads(base_bundle.copula.to_json()),
                         "model_version": P.MODEL_VERSION_BASE})
    assert base_bundle.to_json() == legacy
    back = P.Bundle.from_json(legacy)
    assert back.context is None and back.to_json() == legacy


# ------------------------------------------------------------------------- switch ON

def test_versions_differ_on_and_off(switch, monkeypatch):
    switch(True)
    assert P.CONTEXT_ON and P._VS == "_v1_cx" and P.MODEL_VERSION == "pathA-v0.3"
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE              # common random numbers
    switch(False)
    assert P._VS == "_v1" and P.MODEL_VERSION == "pathA-v0.2"
    for v, on in (("1", True), ("true", True), ("ON", True), ("0", False), ("", False)):
        monkeypatch.setenv(P.CONTEXT_ENV, v)
        assert P._env_flag(P.CONTEXT_ENV) is on
    for v, on in (("", True), ("junk", True), ("0", False), ("off", False), ("1", True)):
        monkeypatch.setenv(P.CONTEXT_ENV, v)              # shipped default True: env still wins
        assert P._env_flag(P.CONTEXT_ENV, True) is on
    assert P.CONTEXT_DEFAULT is True                        # shipped 2026-10-21


def test_on_multipliers_within_clip(conn, base_bundle, strong, switch):
    switch(False)
    day = _late_day(conn)
    off = _team_rates(_specs(conn, P.Context(conn, SEASON), _with(base_bundle, strong), day))
    switch(True)
    ctx = P.Context(conn, SEASON)
    assert ctx.cx_t is not None and len(ctx.cx_t)
    on = _team_rates(_specs(conn, ctx, _with(base_bundle, strong), day))
    assert np.array_equal(on["player_id"], off["player_id"])
    assert np.array_equal(on["star"], off["star"])           # stars picked on the EB rates
    lo, hi = CX.CLIP
    moved = []
    for s in DK_STATS:
        r = on[f"r_{s}"].to_numpy() / off[f"r_{s}"].to_numpy()
        assert np.all(r >= lo - 1e-12) and np.all(r <= hi + 1e-12), s
        moved.append(np.abs(np.log(r)).max())
    assert max(moved) > 0.2                                  # the layer did act ...
    r_pts = on["r_pts"].to_numpy() / off["r_pts"].to_numpy()
    assert np.isclose(r_pts.min(), lo) or np.isclose(r_pts.max(), hi)   # ... up to the clip
    np.testing.assert_allclose(on["r_dk"].to_numpy(), ST.dk_rate(
        {s: on[f"r_{s}"].to_numpy() for s in ST.DK_WEIGHTS}))
    for c in ("mpg", "tier_state", "p_avail"):
        assert np.array_equal(on[c].to_numpy(), off[c].to_numpy()), c


def test_on_projection_reconciles_and_shares_seed(conn, base_bundle, strong, switch):
    """ON end to end: still reconciled to the market, same seed as OFF, means moved."""
    day = _late_day(conn)
    b = _with(base_bundle, strong)
    switch(False)
    off = P.project_historical(conn, day, n_sims=N_SIMS, ctx=P.Context(conn, SEASON), bundle=b)
    switch(True)
    on = P.project_historical(conn, day, n_sims=N_SIMS, ctx=P.Context(conn, SEASON), bundle=b)
    assert on.meta["seed"] == off.meta["seed"]
    assert on.meta["recon"]["ok"]
    for t in on.teams:
        assert abs(t["post"] - t["target"]) <= RC.TOL
    d = np.abs(on.players["dk_mean"].to_numpy() - off.players["dk_mean"].to_numpy())
    assert d.max() > 0.5
    # team DK totals move, but not wildly (points are re-anchored to the implied total)
    tot = lambda r: r.players.groupby(["game_id", "team"])["dk_mean"].sum()  # noqa: E731
    ratio = (tot(on) / tot(off)).to_numpy()
    assert np.all((ratio > 0.8) & (ratio < 1.25))


def test_on_no_lookahead(conn, base_bundle, strong, switch):
    """ON: deleting the day's and later box scores changes nothing (F2 / P strictly pre-day)."""
    switch(True)
    day = _late_day(conn)
    b = _with(base_bundle, strong)
    games = P._home_away(conn, day)

    def run(c):
        return P.project_games(c, day, games, ctx=P.Context(c, SEASON), bundle=b,
                               n_sims=N_SIMS, seed=42, env_mode="closing")
    full = run(conn)
    cut = _copy(conn)
    with cut:
        for t in ("player_games", "team_games"):
            cut.execute(f"DELETE FROM {t} WHERE game_date >= ?", (day.isoformat(),))
        cut.execute("DELETE FROM player_starts WHERE game_id NOT IN "
                    "(SELECT game_id FROM team_games)")
    trunc = run(cut)
    for a, c in zip(full.sims, trunc.sims):
        assert np.array_equal(a.fp, c.fp)


# ------------------------------------------------------------------------- bundle

def test_bundle_round_trip_keeps_context(conn, base_bundle, strong, switch):
    switch(True)
    b = _with(base_bundle, strong)
    js = b.to_json()
    d = json.loads(js)
    assert d["model_version"] == "pathA-v0.3" and d["context"]["groups"] == ["P", "F2"]
    back = P.Bundle.from_json(js)
    assert back.to_json() == js
    assert back.context.train_seasons == strong.train_seasons
    sp = ST.default_params()
    fr = FX.build_frame(conn, SEASON, sp, n_train=6, cache=False)
    test = fr[fr["season"] == SEASON]
    a, c = strong.predict(test), back.context.predict(test)
    for s in DK_STATS:
        np.testing.assert_array_equal(a[s], c[s])


def test_load_bundle_on_is_walk_forward(conn, base_bundle, switch, tmp_path, monkeypatch):
    """load_bundle with the switch ON: reuses the OFF bundle of S, adds a layer fit on seasons
    < S only (scrambling season S's box scores cannot change it), writes its own file."""
    def fit_in(c, data):
        monkeypatch.setenv("COURTSIDE_DATA", str(data))
        P.set_context(False)
        P.cache_dir().mkdir(parents=True, exist_ok=True)
        (P.cache_dir() / f"bundle_{SEASON}{P._VS}.json").write_text(base_bundle.to_json())
        P.set_context(True)
        b = P.load_bundle(c, SEASON)
        return b, P.cache_dir()

    b1, d1 = fit_in(conn, tmp_path / "a")
    assert (d1 / f"bundle_{SEASON}_v1_cx.json").exists()
    assert b1.context is not None and b1.context.season == SEASON
    assert b1.context.train_seasons and all(s < SEASON for s in b1.context.train_seasons)
    assert b1.context.groups == ["P", "F2"]
    on, off = json.loads(b1.to_json()), json.loads(base_bundle.to_json())
    for k in ("season", "minutes", "state", "copula"):
        assert on[k] == off[k], k                            # only the layer differs
    # cached: a second load reads the ON file as is
    assert P.load_bundle(conn, SEASON).to_json() == b1.to_json()
    # scramble season S outcomes (and add a later fake season): the layer is unchanged
    c2 = _copy(conn)
    with c2:
        c2.execute("UPDATE player_games SET pts = pts * 2 + 7, reb = reb + 5, ast = 0 "
                   "WHERE season = ?", (SEASON,))
    b2, _ = fit_in(c2, tmp_path / "b")
    assert b2.context.to_json() == b1.context.to_json()


def test_load_bundle_on_without_off_bundle_reuses_minutes_params(conn, base_bundle, switch,
                                                                  monkeypatch):
    """ON with no OFF bundle on disk: the minutes params cache of the OFF model
    (minutes_params_<S>_v1.json) is reused, never refit under an ON-only name."""
    switch(True)
    P.cache_dir().mkdir(parents=True, exist_ok=True)
    (P.cache_dir() / f"minutes_params_{SEASON}{P._VS_BASE}.json").write_text(
        base_bundle.minutes.to_json())
    (P.cache_dir() / f"state_params_{SEASON}.json").write_text(
        ST.params_to_json(base_bundle.state))

    def no_fit(*a, **k):
        raise AssertionError("minutes refit")
    monkeypatch.setattr(M, "fit", no_fit)
    b = P.load_bundle(conn, SEASON)
    assert b.context is not None and all(s < SEASON for s in b.context.train_seasons)
    assert json.loads(b.minutes.to_json()) == json.loads(base_bundle.minutes.to_json())
    assert (P.cache_dir() / f"bundle_{SEASON}_v1_cx.json").exists()
    assert not (P.cache_dir() / f"bundle_{SEASON}_v1.json").exists()   # OFF file untouched
