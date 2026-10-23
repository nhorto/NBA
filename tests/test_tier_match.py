"""Tier match (COURTSIDE_TIER_MATCH, `+tm`; `copula.TIER_MATCH`).

OFF by default and nothing of it runs when off: the draw's tiers are role_tiers(mu_played,
min_played) exactly as before and the draws are bit-identical. ON the draw assigns the rho / CV
tiers from the fit's inputs (r_dk x mpg, mpg; players who never play are bench); the minutes
and the means are unchanged (same random draws), the version gets `+tm` (model version only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_proj_copula import _spec, game  # noqa: F401  (module fixture)

from courtside.proj import copula as C
from courtside.proj import pipeline as P
from courtside.proj import reconcile as RC


@pytest.fixture
def restore():
    state = C.TIER_MATCH
    yield
    P.set_tier_match(state)


def test_default_off_and_env_flag(monkeypatch):
    assert C.TIER_MATCH_DEFAULT is False
    monkeypatch.setenv(C.TIER_MATCH_ENV, "1")
    assert RC._flag(C.TIER_MATCH_ENV, False) is True
    monkeypatch.delenv(C.TIER_MATCH_ENV)
    assert RC._flag(C.TIER_MATCH_ENV, C.TIER_MATCH_DEFAULT) is False


def test_version_tag_composes(restore):
    P.set_tier_match(False)
    off_vs, off_mv = P._VS, P.MODEL_VERSION
    assert "+tm" not in off_mv
    P.set_tier_match(True)
    assert P._VS == off_vs and P.MODEL_VERSION == off_mv + "+tm"      # model version only
    assert P.SEED_VERSION == P.MODEL_VERSION_BASE                      # seeds unchanged
    P.set_tier_match(False)
    assert P.MODEL_VERSION == off_mv


def test_fit_style_tiers_is_the_fit_definition():
    team = np.array(["A"] * 6 + ["B"] * 3)
    pl = pd.DataFrame({"mpg": [34, 33, 31, 25, 12, 30, 36, 22, 9],
                       "r_dk": [1.5, 1.2, 1.4, 1.0, 1.0, 0.9, 1.3, 1.0, 0.8]})
    pp = np.ones(9)
    t = C.fit_style_tiers(team, pl, pp)
    # the fit's call, verbatim; team A: stars = top 2 of r_dk x mpg among mpg >= 28
    assert t.tolist() == C.role_tiers(team, (pl["r_dk"] * pl["mpg"]).to_numpy(),
                                      pl["mpg"].to_numpy()).tolist()
    assert t.tolist() == [0, 1, 0, 1, 2, 1, 0, 1, 2]
    # a player who is out (P(play) = 0) holds no star slot: the next one takes it
    pp2 = pp.copy()
    pp2[0] = 0.0
    t2 = C.fit_style_tiers(team, pl, pp2)
    assert t2[0] == 2 and (t2[:6] == 0).sum() == 2 and t2[1] == 0
    # a questionable player keeps his tier given he plays
    pp2[0] = 0.2
    assert C.fit_style_tiers(team, pl, pp2).tolist() == t.tolist()
    # falls back to the DK-weighted per-stat rates when r_dk is absent
    pl2 = pl.drop(columns="r_dk").assign(**{f"r_{s}": 0.1 for s in C.DK_STATS})
    assert C.fit_style_tiers(team, pl2, pp).shape == (9,)


def _info_tiers(spec, cp, mp, sp, on, seed=11):
    C.TIER_MATCH = on
    return C.simulate_game(spec, cp, mp, sp, 600, seed)


def test_off_is_unchanged_and_on_matches_the_fit(game, restore):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    cp = C.default_copula_params()
    off = _info_tiers(spec, cp, mp, sp, False)
    # OFF: the draw's tiers are the old mu_played / min_played definition, bit for bit
    inf = off.info
    old = C.role_tiers(inf["team"].to_numpy(), inf["mu_played"].to_numpy(),
                       inf["min_played"].to_numpy())
    assert [C.TIERS[t] for t in old] == inf["tier"].tolist()
    on = _info_tiers(spec, cp, mp, sp, True)
    pl = pd.concat([spec.home.players, spec.away.players], ignore_index=True)
    want = C.fit_style_tiers(on.info["team"].to_numpy(), pl, on.info["p_play"].to_numpy())
    assert on.info["tier"].tolist() == [C.TIERS[t] for t in want]
    # same minutes and the same mu_played: only the correlation / CV tiers of the draw move
    assert np.array_equal(on.minutes, off.minutes)
    assert np.allclose(on.info["mu_played"], off.info["mu_played"])
    # certain-out player is bench under the fit definition
    j = int(np.flatnonzero(on.player_id == spec.home.players["player_id"].iloc[0])[0])
    assert on.info["tier"].iloc[j] == "bench"
    # OFF twice is deterministic (and the switch left no state behind)
    off2 = _info_tiers(spec, cp, mp, sp, False)
    assert np.array_equal(off.fp, off2.fp)


def test_on_changes_draws_only_when_a_tier_moves(game, restore):  # noqa: F811
    spec, mp, sp = _spec(game)
    cp = C.default_copula_params()
    cp.rho_team = {k: 0.05 + 0.1 * i for i, k in enumerate(sorted(cp.rho_team))}   # tiers matter
    off = _info_tiers(spec, cp, mp, sp, False)
    on = _info_tiers(spec, cp, mp, sp, True)
    moved = off.info["tier"].to_numpy() != on.info["tier"].to_numpy()
    assert moved.any()                  # this fixture has tier moves
    assert not np.array_equal(on.fp, off.fp)
