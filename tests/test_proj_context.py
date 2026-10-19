"""Context layer (Phase 2F F0): neutrality, learning a planted effect, JSON round trip,
determinism, fit < S, the pipeline hook and the clustered SE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from courtside.proj import context as CX
from courtside.proj import features as FX
from courtside.proj import state as ST
from courtside.proj.features.common import DK_STATS

SMALL = {"min_data_in_leaf": 200, "min_sum_hessian_in_leaf": 5.0, "num_threads": 1}
GROUPS = ("P", "F1", "F2")


def synth_frame(n: int = 24000, effect: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Random features for every model column; y ~ Poisson(E x exp(effect x signal)), signal =
    the reb rows' own-group DvP index (sd 0.1) for reb, nothing for the other stats."""
    rng = np.random.default_rng(seed)
    cols = sorted({c for s in DK_STATS for c in FX.model_columns(s, GROUPS)})
    d = pd.DataFrame(rng.normal(0.0, 0.1, (n, len(cols))), columns=cols)
    d["p_grp"] = rng.integers(0, 3, n).astype(float)
    d["p_mpg"] = rng.uniform(10, 36, n)
    for c in cols:
        if c.startswith("f1_"):
            d[c] = np.where(rng.random(n) < 0.8, 0.0, np.abs(d[c]) * 10)
    seasons = np.array(["2020-21", "2021-22", "2022-23", "2023-24"])
    d["season"] = seasons[np.minimum(np.arange(n) * 4 // n, 3)]
    d["season_type"] = "Regular Season"
    d["game_id"] = (np.arange(n) // 10).astype(str)
    d["player_id"] = rng.integers(0, 400, n)
    rate = {"pts": .45, "fg3m": .05, "reb": .18, "ast": .1, "stl": .03, "blk": .02, "tov": .06}
    mins = rng.uniform(10, 38, n)
    for s in DK_STATS:
        E = rate[s] * mins
        lift = effect * d["f2_dvp_reb_grp"].to_numpy() / 0.1 if s == "reb" else 0.0
        d[f"E_{s}"] = E
        d[s] = rng.poisson(E * np.exp(lift))
        d[f"d_{s}"] = 0.05
    return d


@pytest.fixture(scope="module")
def null_model():
    d = synth_frame(effect=0.0)
    m = CX.fit(d, (d["season"] < "2023-24").to_numpy(), GROUPS, season="2023-24", params=SMALL)
    return d, m


def test_null_multiplier_is_one(null_model):
    d, m = null_model
    test = d[d["season"] == "2023-24"]
    mult = m.predict(test)
    for s in DK_STATS:
        E = test[f"E_{s}"].to_numpy()
        assert abs((mult[s] * E).sum() / E.sum() - 1.0) < 0.02, s
        assert np.std(np.log(mult[s])) < 0.03, s
    # at the league-average context the multiplier is ~1, and exactly 1 when centred
    for s in DK_STATS:
        ref = pd.DataFrame([m.reference[s]])
        assert abs(m.predict_stat(ref, s)[0] - 1.0) < 0.03, s


def test_centered_is_exactly_one_at_reference():
    d = synth_frame(effect=0.3, n=12000)
    m = CX.fit(d, (d["season"] < "2023-24").to_numpy(), GROUPS, params=SMALL, center=True,
               stats=("reb",))
    ref = pd.DataFrame([m.reference["reb"]])
    assert m.predict_stat(ref, "reb")[0] == pytest.approx(1.0, abs=1e-12)
    assert m.reference["reb"]["f2_dvp_reb_grp"] == 0.0      # neutral = league average
    assert m.reference["reb"]["f1_vac_min"] == 0.0           # neutral = nobody absent


def test_learns_planted_effect():
    d = synth_frame(effect=0.15, seed=3)
    m = CX.fit(d, (d["season"] < "2023-24").to_numpy(), ("P", "F2"), params=SMALL,
               stats=("reb", "pts"))
    test = d[d["season"] == "2023-24"]
    lm = np.log(m.predict(test)["reb"])
    truth = 0.15 * test["f2_dvp_reb_grp"].to_numpy() / 0.1
    assert np.corrcoef(lm, truth)[0, 1] > 0.8
    assert np.std(np.log(m.predict(test)["pts"])) < 0.03       # no effect planted on points
    # the own-index monotone constraint holds
    x = pd.DataFrame([m.reference["reb"]] * 2)
    x.loc[1, "f2_dvp_reb_grp"] = 0.3
    mm = m.predict_stat(x, "reb")
    assert mm[1] >= mm[0]


def test_json_round_trip_and_determinism(null_model, tmp_path):
    d, m = null_model
    test = d[d["season"] == "2023-24"]
    p = tmp_path / "ctx.json"
    CX.save(m, p)
    m2 = CX.load(p)
    a, b = m.predict(test), m2.predict(test)
    for s in DK_STATS:
        np.testing.assert_array_equal(a[s], b[s])
    m3 = CX.fit(d, (d["season"] < "2023-24").to_numpy(), GROUPS, season="2023-24", params=SMALL)
    assert m3.to_json() == m.to_json()


def test_fit_uses_only_seasons_before_s(null_model):
    """Scrambling the test season's outcomes cannot change the model (L2: fit < S)."""
    d, m = null_model
    d2 = d.copy()
    te = (d2["season"] == "2023-24").to_numpy()
    rng = np.random.default_rng(9)
    for s in DK_STATS:
        d2.loc[te, s] = rng.permutation(d2.loc[te, s].to_numpy()) + 3
    m2 = CX.fit(d2, (d2["season"] < "2023-24").to_numpy(), GROUPS, season="2023-24",
                params=SMALL)
    assert m2.to_json() == m.to_json()


def test_identity_and_apply_rates():
    rates = pd.DataFrame({"player_id": [1, 2], "grp": [0, 2], "mpg": [30.0, 20.0],
                          "g_eff": [40.0, 30.0], "r_dk": [1.0, 0.8], "p_dk": [0.9, 0.9]})
    for s in ST.STATS:
        rates[f"r_{s}"], rates[f"p_{s}"], rates[f"n_{s}"] = 0.1, 0.1, 500.0
    tg = pd.DataFrame({"game_id": ["g"], "team": ["AAA"]})
    same = CX.apply_rates(CX.identity(), rates, tg, ["g", "g"], ["AAA", "AAA"])
    pd.testing.assert_frame_equal(same, rates)
    d = synth_frame(n=6000, effect=0.2)
    m = CX.fit(d, (d["season"] < "2023-24").to_numpy(), ("P",), params=SMALL)
    out = CX.apply_rates(m, rates, tg, ["g", "g"], ["AAA", "AAA"])
    expect = ST.dk_rate({s: out[f"r_{s}"].to_numpy() for s in ST.DK_WEIGHTS})
    np.testing.assert_allclose(out["r_dk"].to_numpy(), expect)
    assert (out["r_pf"] == rates["r_pf"]).all()              # non-DK stats untouched


def test_two_way_se():
    rng = np.random.default_rng(0)
    n = 4000
    d = rng.normal(size=n)
    g_unique = np.arange(n)
    se_iid = d.std(ddof=1) / np.sqrt(n)
    assert CX.two_way_se(d, g_unique, g_unique) == pytest.approx(se_iid, rel=0.01)
    games = np.arange(n) // 20
    shared = rng.normal(size=n // 20)[games]                 # a common game shock
    d2 = d + 2.0 * shared
    one_way = CX.two_way_se(d2, g_unique, g_unique)
    two_way = CX.two_way_se(d2, games, g_unique)
    assert two_way > 2.0 * one_way


def test_on_synthetic_league():
    """Features from the synthetic league + the layer: centres facing DDD (which gives up
    rebounds) get a larger rebounding multiplier in the held-out season, and the layer beats
    the base on NB log-lik there."""
    from test_proj_features import feature_rows, make_league
    fr = feature_rows(make_league(seed=5))
    rs = (fr["season_type"] == "Regular Season").to_numpy()
    train = rs & (fr["season"] < "2023-24").to_numpy()
    test = fr[rs & (fr["season"] == "2023-24").to_numpy()].reset_index(drop=True)
    m = CX.fit(fr, train, ("P", "F2"), params=SMALL, stats=("reb", "tov"))
    mult = m.predict(test)
    mm = {s: mult[s] if s in ("reb", "tov") else np.ones(len(test)) for s in DK_STATS}
    c = (test["grp"] == 2).to_numpy()
    vs_d = (test["opp"] == "DDD").to_numpy()
    assert mult["reb"][c & vs_d].mean() > mult["reb"][c & ~vs_d].mean() + 0.03
    ev = CX.evaluate(test, {"base": None, "P+F2": mm}, CX.cv_table(fr[train]),
                     pairs=[("P+F2", "base")])
    assert ev["pairs"]["P+F2 vs base"]["all"]["ll"]["diff"] > 0
