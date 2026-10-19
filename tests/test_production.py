"""Production scoring: cached feature overrides equal the model's own, factor shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_model_gbm import _registry, _split, league  # noqa: F401  (league is a fixture)

from courtside import production
from courtside.model import gbm


def _fitted(league, name=gbm.PRODUCTION_WINPROB):
    f, p = league
    train, test = _split(f)
    reg, _ = _registry(p)
    m = reg[name]()
    m.fit(train)
    return m, train, test


def test_overrides_equal_the_models_own_lineup_and_market_columns(league):
    m, train, test = _fitted(league)
    F = m._test_features(test).reset_index(drop=True)
    rng = np.random.default_rng(4)
    n = len(test)
    lh = F["ps_home"].to_numpy() + rng.normal(0, 1.5, n)
    la = F["ps_away"].to_numpy() + rng.normal(0, 1.5, n)
    lh[::3] = np.nan                                         # teams with no report keep the engine's
    la[1::4] = np.nan
    mm = rng.normal(0, 6, n)
    mm[::5] = np.nan                                         # no live line: keep what F has
    mt = 215 + rng.normal(0, 5, n)
    t2 = test.copy()
    t2["lineup_home"], t2["lineup_away"] = lh, la
    t2["mkt_margin"] = np.where(np.isfinite(mm), mm, t2["mkt_margin"].to_numpy(float))
    t2["mkt_total"] = mt
    F2 = m._test_features(t2).reset_index(drop=True)         # the slow path: rebuild features
    G = production.apply_overrides(F, mm, mt, lh, la)       # the cached path
    for c in ("ps_home", "ps_away", "ps_diff", "abs_loss_home", "abs_loss_away",
              "mkt_margin", "mkt_total"):
        assert np.allclose(G[c], F2[c], equal_nan=True), c
    sc = production.Scorer.from_model(m)
    sc.set_features(G, (t2["season_type"].to_numpy() != "Regular Season").astype(float))
    t2 = t2.reset_index(drop=True)
    assert np.allclose(sc.predict_margin(t2), m.predict_margin(t2))
    assert np.allclose(sc.predict_total(t2), m.predict_total(t2))
    assert np.allclose(sc.predict_proba(t2), m.predict_proba(t2))
    # the explain rows come out the same too
    assert sc.explain(t2.iloc[0], t2) == m.explain(t2.iloc[0], t2)


def test_absence_effect_is_the_margin_cost_against_a_healthy_roster(league):
    m, train, test = _fitted(league, gbm.PRODUCTION_OPINION)
    F = m._test_features(test).reset_index(drop=True)
    sc = production.Scorer.from_model(m)
    # take the first game's home side to full roster minus 4 points of strength
    rh = F["ps_home"] + F["abs_loss_home"]
    G = production.apply_overrides(F, lineup_home=(rh - 4.0).to_numpy())
    sc.set_features(G, np.zeros(len(G)))
    eh, ea = production.absence_effects(sc, G)
    # the linear part alone makes the margin respond to ps_diff with a positive slope, so
    # losing strength costs the home side margin; the away side is at its own (unchanged) level
    assert (eh < 0).mean() > 0.9 and eh.mean() < -0.5
    assert np.isfinite(ea).all()
    sc.set_features(F, np.zeros(len(F)))
    eh0, _ = production.absence_effects(sc, F)
    assert eh.mean() < eh0.mean()


EXPLAIN = [
    {"factor": "market", "team": "BOS", "impact_pts": 9.0, "home_margin_pts": 9.0},
    {"factor": "home_court", "team": "BOS", "impact_pts": 2.0, "home_margin_pts": 2.0},
    {"factor": "absences", "team": "LAL", "impact_pts": -0.9, "home_margin_pts": 0.9},
    {"factor": "player_strength", "team": "BOS", "impact_pts": 1.4, "home_margin_pts": 1.4},
    {"factor": "rest", "team": "LAL", "impact_pts": -0.8, "home_margin_pts": 0.8},
    {"factor": "team_rating", "team": "BOS", "impact_pts": 0.6, "home_margin_pts": 0.6},
    {"factor": "recent_form", "team": "BOS", "impact_pts": 0.3, "home_margin_pts": 0.3},
    {"factor": "travel", "team": "BOS", "impact_pts": 0.1, "home_margin_pts": 0.1},
    {"factor": "pace_scoring", "team": "BOS", "impact_pts": 3.0, "home_margin_pts": 3.0},
]


def test_key_factors_shape_and_content():
    who = {"LAL": [{"player": "LeBron James", "status": "Out"},
                   {"player": "Austin Reaves", "status": "Questionable"}]}
    out = production.key_factors(EXPLAIN, "BOS", "LAL", who, {"LAL": -3.1, "BOS": 0.0})
    assert [f["factor"] for f in out] == ["absences", "player_strength", "rest", "team_rating"]
    assert len(out) <= production.FACTORS_PER_GAME
    for f in out:
        assert set(f) == {"kind", "factor", "team", "side", "value", "impact_pts", "favors",
                          "detail"}
        assert f["kind"] == "model_factor" and f["value"] == f["impact_pts"]
        assert f["side"] in ("home", "away") and f["favors"] in ("home", "away")
        assert isinstance(f["detail"], str) and f["detail"]
    ab = out[0]
    assert (ab["team"], ab["side"], ab["impact_pts"], ab["favors"]) == ("LAL", "away", -3.1, "home")
    assert ab["detail"] == "LAL absences: LeBron James (Out), Austin Reaves (Questionable)"
    assert out[1]["favors"] == "home" and out[2]["favors"] == "home"   # LAL rest penalty
    shown = {f["factor"] for f in out}
    assert not shown & {"market", "home_court", "pace_scoring"}


def test_key_factors_without_counterfactual_and_with_few_factors():
    out = production.key_factors(EXPLAIN, "BOS", "LAL")
    assert out[0]["factor"] == "player_strength"
    assert "absences" in {f["factor"] for f in out}      # explain's own number is kept
    tiny = [{"factor": "rest", "team": "BOS", "impact_pts": 0.1, "home_margin_pts": 0.1},
            {"factor": "travel", "team": "BOS", "impact_pts": 0.08, "home_margin_pts": 0.08},
            {"factor": "continuity", "team": "LAL", "impact_pts": 0.06, "home_margin_pts": 0.06}]
    out = production.key_factors(tiny, "BOS", "LAL")     # nothing above the noise floor: top 3
    assert [f["factor"] for f in out] == ["rest", "travel", "continuity"]
    assert production.key_factors([], "BOS", "LAL") == []
