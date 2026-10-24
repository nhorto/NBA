"""Upper-tail quantile remap (COURTSIDE_TAIL_REMAP, `+tr`; `proj/tail_remap.py`, IMPROVEMENTS.md D6).

OFF by default and nothing of it runs when off (same sims, same frames, same versions); ON the
upper half of each player's played sims is re-labelled through g = h^-1 (monotone, identity at and
below the median): the draws are unchanged (common random numbers), minutes and DNP sims are
untouched, every sim keeps its rank, and the version gets `+tr` after `+tm`."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import pipeline as P
from courtside.proj import role_blend as RB
from courtside.proj import tail_remap as TR

DAY = "2024-01-10"
# a too-wide upper tail: the outcome's v-quantile sits at these levels of the sims (v = LEVELS)
WIDE = [0.73, 0.87, 0.93, 0.965, 0.978]


@pytest.fixture
def restore():
    state = (TR.TAIL_REMAP, RB.ROLE_BLEND, C.TIER_MATCH)
    yield
    P.set_tier_match(state[2])
    P.set_role_blend(state[1])
    P.set_tail_remap(state[0])


def _bundle(mp, sp):
    return SimpleNamespace(copula=C.default_copula_params("2023-24"), minutes=mp, state=sp,
                           season="2023-24", copula_v1=None)


def _table(h):
    return {f"{t}|*": list(h) for t in TR.TIERS}


def _digest(res) -> str:
    """sha1 of every sim array, the players / stats frames and the meta without timings."""
    x = hashlib.sha1()
    for g in res.sims:
        for a in (g.fp, g.minutes, np.asarray(g.player_id)):
            x.update(np.ascontiguousarray(a).tobytes())
    x.update(pd.util.hash_pandas_object(res.players, index=True).to_numpy().tobytes())
    x.update(pd.util.hash_pandas_object(res.stats, index=True).to_numpy().tobytes())
    meta = {k: v for k, v in res.meta.items() if not k.startswith("seconds")}
    x.update(json.dumps(meta, sort_keys=True, default=str).encode())
    return x.hexdigest()


# ------------------------------------------------------------------------------ switch

def test_default_on_and_env_flag(monkeypatch):
    assert TR.TAIL_REMAP_DEFAULT is True                # shipped 2026-10-24 (conftest pins it off)
    monkeypatch.setenv(TR.TAIL_REMAP_ENV, "1")
    assert TR._flag(TR.TAIL_REMAP_ENV, False) is True
    monkeypatch.setenv(TR.TAIL_REMAP_ENV, "off")
    assert TR._flag(TR.TAIL_REMAP_ENV, True) is False
    monkeypatch.delenv(TR.TAIL_REMAP_ENV)
    assert TR._flag(TR.TAIL_REMAP_ENV, TR.TAIL_REMAP_DEFAULT) is True


def test_version_tag_composes(restore):
    P.set_tail_remap(False)
    P.set_role_blend(False)
    P.set_tier_match(False)
    off_vs, off_mv = P._VS, P.MODEL_VERSION
    assert "+tr" not in off_mv
    P.set_tail_remap(True)
    assert P._VS == off_vs and P.MODEL_VERSION == off_mv + "+tr"     # model version only
    P.set_role_blend(True)
    P.set_tier_match(True)
    assert P.MODEL_VERSION == off_mv + "+rb+tm+tr"                   # after +rb and +tm
    P.set_tail_remap(False)
    assert P.MODEL_VERSION == off_mv + "+rb+tm"
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                  # seeds unchanged


# ------------------------------------------------------------------------------ the map

def test_mapping_monotone_and_median_fixed():
    V, T = TR.mapping(WIDE)
    assert T[0] == 0 and T[-1] == 1 and V[1] == T[1] == TR.MEDIAN
    assert (np.diff(V) > 0).all() and (np.diff(T) > 0).all()
    assert list(T[2:-1]) == WIDE and list(V[2:-1]) == list(TR.LEVELS)
    # degenerate input is repaired into a strictly increasing map
    _, T2 = TR.mapping([0.4, 0.4, 1.2, 0.99, 1.0])
    assert (np.diff(T2) > 0).all() and T2[-2] < 1


def test_remap_column_properties():
    rng = np.random.default_rng(3)
    x = np.round(rng.gamma(2.0, 8.0, 5000), 1).astype(np.float32)    # many ties
    V, T = TR.mapping(WIDE)
    y = TR.remap_column(x, V, T)
    med = np.median(x)
    assert np.array_equal(y[x <= med], x[x <= med].astype(np.float64))  # median and below kept
    o = np.argsort(x, kind="stable")
    assert (np.diff(y[o]) >= -1e-9).all()                              # order preserved
    for v in np.unique(x)[::50]:                                      # ties stay tied
        assert np.ptp(y[x == v]) == 0
    assert y.max() <= x.max() + 1e-6 and y.min() >= x.min() - 1e-6
    assert y.mean() < x.mean() and np.quantile(y, 0.99) < np.quantile(x, 0.99)
    # identity knots return the input exactly
    Vi, Ti = TR.mapping(TR.LEVELS)
    assert np.array_equal(TR.remap_column(x, Vi, Ti), x.astype(np.float64))


def test_fit_then_remap_calibrates_a_too_wide_model():
    """Model N(0, 1.4) for outcomes N(0, 1): fit h on the outcomes' PIT, remap the sims, and the
    outcomes' PIT in the remapped sims has the nominal upper-tail shares."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    fr = pd.DataFrame({"tier": "bench", "min_played": 20.0, "uc": norm.cdf(y / 1.4)})
    fit = TR.fit_cells(fr)
    q = fit["cells"]["bench|16-24"]["q"]
    assert q == pytest.approx([norm.cdf(norm.ppf(t) / 1.4) for t in TR.LEVELS], abs=0.003)
    sims = 1.4 * rng.standard_normal(40_000)
    V, T = TR.mapping(q)
    z = np.sort(TR.remap_column(sims, V, T))
    u = np.searchsorted(z, y[:50_000]) / len(z)
    for t in (0.90, 0.95, 0.99):
        assert (u > t).mean() == pytest.approx(1 - t, abs=0.003)
    assert (u > 0.5).mean() == pytest.approx(0.5, abs=0.01)


def test_fit_cells_shrinks_small_cells():
    rng = np.random.default_rng(1)
    big = pd.DataFrame({"tier": "role", "min_played": 28.0, "uc": rng.random(50_000) ** 1.3})
    small = pd.DataFrame({"tier": "role", "min_played": 34.0, "uc": rng.random(50)})
    fit = TR.fit_cells(pd.concat([big, small]))
    hs, ht = np.array(fit["cells"]["role|32+"]["q"]), np.array(fit["tiers"]["role"])
    assert np.abs(hs - ht).max() < 0.01                               # 50 rows vs 2,000 prior
    assert fit["cells"]["role|32+"]["n"] == 50 and fit["n"]["role|*"] == 50_050


def test_params_walk_forward_and_fallback(tmp_path, monkeypatch):
    p = tmp_path / "tr.json"
    p.write_text(json.dumps({"seasons": {
        "2020-21": {"tiers": {"bench": WIDE}, "cells": {}},
        "2023-24": {"tiers": {"bench": WIDE}, "cells": {"bench|16-24": {"n": 9, "q": TR.LEVELS}}}}}))
    monkeypatch.setattr(TR, "TR_PATH", p)
    TR.cells.cache_clear()
    try:
        assert TR.cells("2019-20") == {}                               # nothing earlier: no remap
        assert TR.cells("2021-22") == {"bench|*": WIDE}                 # latest earlier season
        t = TR.cells("2024-25")
        assert TR.lookup(t, "bench", 20.0) == list(TR.LEVELS)          # the cell
        assert TR.lookup(t, "bench", 30.0) == WIDE                     # its tier
        assert TR.lookup(t, "star", 30.0) is None                      # left as is
    finally:
        TR.cells.cache_clear()


def test_shipped_params_are_walk_forward():
    par = json.loads(TR.TR_PATH.read_text())
    assert par["levels"] == list(TR.LEVELS)
    for s, e in par["seasons"].items():
        assert e["train"] and all(x.split(":")[1] < s for x in e["train"])
        for h in [*e["tiers"].values(), *(c["q"] for c in e["cells"].values())]:
            assert len(h) == len(TR.LEVELS) and all(a < b for a, b in zip(h, h[1:]))
            assert h[0] > TR.MEDIAN and h[-1] < 1


# ------------------------------------------------------------------------------ run_games

def test_run_games_off_never_runs(game, restore, monkeypatch):  # noqa: F811
    spec, mp, sp = _spec(game, p_half=[1, 3])
    b = _bundle(mp, sp)
    P.set_tail_remap(False)
    monkeypatch.setattr(TR, "apply", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    a = P.run_games([spec], b, 2000, 7, P._as_date(DAY))
    assert "tail_remap" not in a.meta and "tr_shift" not in a.players


@pytest.mark.skipif(sys.platform != "linux", reason="exact float golden: Linux only")
def test_run_games_off_matches_pre_change_golden(game, restore):  # noqa: F811
    """OFF output == the pipeline before `+tr` (HEAD 102ed53; digest computed from that copy)."""
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    P.set_tail_remap(False)
    P.set_role_blend(False)
    a = P.run_games([spec], _bundle(mp, sp), 3000, 11, P._as_date(DAY))
    assert _digest(a) == GOLDEN_OFF


def test_run_games_on_relabels_upper_half_only(game, restore, monkeypatch):  # noqa: F811
    monkeypatch.setattr(TR, "cells", lambda season: _table(WIDE))
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    b = _bundle(mp, sp)
    P.set_tail_remap(False)
    off = P.run_games([spec], b, 3000, 11, P._as_date(DAY))
    P.set_tail_remap(True)
    on = P.run_games([spec], b, 3000, 11, P._as_date(DAY))
    g0, g1 = off.sims[0], on.sims[0]
    assert np.array_equal(g0.minutes, g1.minutes)                    # same draws
    played = g0.minutes > 0
    assert (g1.fp[~played] == 0).all() and (g0.fp[~played] == 0).all()
    sh = g1.info["tr_shift"].to_numpy(float)
    n_moved = 0
    for j in range(g0.fp.shape[1]):
        m = played[:, j]
        x, y = g0.fp[m, j].astype(float), g1.fp[m, j].astype(float)
        if m.sum() < TR.MIN_PLAYED:
            assert np.array_equal(x, y) and sh[j] == 0
            continue
        med = np.median(x)
        assert np.array_equal(x[x <= med], y[x <= med])               # lower half untouched
        o = np.argsort(x, kind="stable")
        assert (np.diff(y[o]) >= -1e-4).all()                          # ranks kept
        assert y.max() <= x.max() + 1e-4
        assert sh[j] == pytest.approx(y.mean() - x.mean(), abs=1e-4)
        assert sh[j] <= 1e-6
        n_moved += sh[j] < 0
    assert n_moved > 5
    assert on.meta["tail_remap"]["remapped"] > 0
    m0 = off.players.set_index("player_id")["dk_mean"]
    m1 = on.players.set_index("player_id")["dk_mean"]
    pp = off.players.set_index("player_id")["p_play"]
    shs = pd.Series(sh, index=g0.player_id).reindex(m0.index)
    assert np.allclose((m1 - m0).to_numpy(), (shs * pp).to_numpy(), atol=2e-3)
    assert (on.players["dk_q95"] <= off.players["dk_q95"] + 1e-4).all()
    mu0 = off.players.set_index("player_id")["mu_played"]
    mu1 = on.players.set_index("player_id")["mu_played"]
    assert np.allclose((mu1 - mu0).to_numpy(), shs.to_numpy(), atol=1e-6)


def test_on_with_identity_map_draws_exactly_off(game, restore, monkeypatch):  # noqa: F811
    monkeypatch.setattr(TR, "cells", lambda season: _table(TR.LEVELS))
    spec, mp, sp = _spec(game, p_half=[2])
    b = _bundle(mp, sp)
    P.set_tail_remap(False)
    a = P.run_games([spec], b, 2000, 5, P._as_date(DAY))
    P.set_tail_remap(True)
    c = P.run_games([spec], b, 2000, 5, P._as_date(DAY))
    assert np.array_equal(a.sims[0].fp, c.sims[0].fp)
    pd.testing.assert_frame_equal(a.players, c.players[a.players.columns], check_exact=True)
    pd.testing.assert_frame_equal(a.stats, c.stats, check_exact=True)


def test_on_after_role_blend(game, restore, monkeypatch):  # noqa: F811
    """With the blend on, the remap acts on the blended sims (it runs second)."""
    calls = []
    monkeypatch.setattr(RB, "apply", lambda sims, specs, day: calls.append("rb") or {})
    monkeypatch.setattr(TR, "apply", lambda sims, day: calls.append("tr") or {})
    spec, mp, sp = _spec(game)
    P.set_role_blend(True)
    P.set_tail_remap(True)
    P.run_games([spec], _bundle(mp, sp), 500, 3, P._as_date(DAY))
    assert calls == ["rb", "tr"]


# digest of `git show 102ed53:src/courtside/proj/pipeline.py` (the pipeline before `+tr`) on this
# fixture, Linux
GOLDEN_OFF = "e569a81b02de0df62fc91aa58fc0391a8e3b605b"
