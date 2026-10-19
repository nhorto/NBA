"""Causal correlation parameters (T2.7): shrinkage, JSON round trip, PSD matrices, pair
statistics vs brute force, causal tiering."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from courtside.proj import corr as K


def _frame(n_games=400, seed=0, rho_team=0.3, season="2020-21"):
    """Synthetic residual frame: two teams of 8 (tiers 2/3/3), teammate residuals share a team
    factor so the within-team correlation is ~rho_team; opponents independent."""
    rng = np.random.default_rng(seed)
    tiers = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    rows = []
    for gi in range(n_games):
        for ti, team in enumerate(("AAA", "BBB")):
            f = rng.standard_normal()
            for p in range(8):
                mu = (38.0, 38.0, 24.0, 24.0, 24.0, 12.0, 12.0, 12.0)[p]
                e = np.sqrt(rho_team) * f + np.sqrt(1 - rho_team) * rng.standard_normal()
                dk = mu + 0.3 * mu * e
                rows.append(dict(game_id=f"g{gi:04d}", season=season, team=team,
                                 player_id=ti * 100 + p, tier=int(tiers[p]),
                                 pos=int(rng.integers(0, 3)), mu=mu, min_proj=30.0 - 2 * tiers[p],
                                 dk=dk, min=30.0 - 2 * tiers[p], g=1.0))
    d = pd.DataFrame(rows)
    d["grp"] = d["tier"] * K.NPOS + d["pos"]
    return d


# ------------------------------------------------------------------------------ shrinkage

def test_shrink_limits_and_monotone():
    assert K.shrink(0.2, 0, 0.05, 1000) == pytest.approx(0.05)
    assert K.shrink(0.2, 1e9, 0.05, 1000) == pytest.approx(0.2, abs=1e-6)
    assert K.shrink(0.2, 500, 0.05, 0) == pytest.approx(0.2)           # k = 0: raw if n > 0
    assert K.shrink(0.2, 0, 0.05, 0) == pytest.approx(0.05)
    ns = np.array([0, 100, 1000, 10000, 100000])
    out = K.shrink(np.full(5, 0.2), ns, 0.05, 2000)
    assert np.all(np.diff(out) > 0) and out[0] == pytest.approx(0.05) and out[-1] < 0.2
    # exact weight n / (n + k)
    assert K.shrink(0.3, 2000, 0.0, 2000) == pytest.approx(0.15)


def test_fit_shrinks_small_cells_toward_tier_pair():
    f = _frame(200, seed=1)
    big = K.fit(f, ["2020-21"], k_shrink=1e12, k_cell=1e12, modes=("U",))
    t = big.table("U")
    # infinite shrinkage: every cell equals its tier pair equals the pooled value
    assert all(v == pytest.approx(t.rho_team_all) for v in t.rho_team.values())
    assert all(v == pytest.approx(t.rho_team_all) for v in t.rho_team_cell.values())
    none = K.fit(f, ["2020-21"], k_shrink=0.0, k_cell=0.0, modes=("U",)).table("U")
    spread0 = np.std(list(none.rho_team_cell.values()))
    mid = K.fit(f, ["2020-21"], k_shrink=3000, k_cell=3000, modes=("U",)).table("U")
    spread1 = np.std(list(mid.rho_team_cell.values()))
    assert spread1 < spread0


# ------------------------------------------------------------------------------ JSON

def test_params_json_round_trip(tmp_path):
    f = _frame(150, seed=2)
    p = K.fit(f, ["2020-21"], k_shrink=2000)
    s = p.to_json()
    p2 = K.CorrParams.from_json(s)
    assert p2.to_json() == s
    assert p2.season == "2021-22" and p2.train_seasons == ["2020-21"]
    assert set(p2.tables) == set(K.MODES)
    path = K.save(p, tmp_path / "corr_2021-22.json")
    p3 = K.load("2021-22", path)
    assert p3.table("U").rho_team == p.table("U").rho_team
    assert p3.table().rho_team_cell == p.table().rho_team_cell      # default mode


# ------------------------------------------------------------------------------ PSD

def test_corr_matrix_is_psd_unit_diagonal():
    f = _frame(120, seed=3)
    p = K.fit(f, ["2020-21"])
    team = np.array(["A"] * 9 + ["B"] * 9)
    tiers = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2] * 2)
    pos = np.array([0, 2, 1, 1, 0, 2, 2, 0, 3] * 2)
    for m in K.MODES:
        R = p.corr_matrix(team, tiers, pos, mode=m)
        assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)
        assert np.linalg.eigvalsh(R).min() > -1e-9
        np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
    # an adversarial table (strong negative teammate correlation) must still be repaired
    t = p.table("U")
    for k in t.rho_team:
        t.rho_team[k] = -0.6
    for k in t.rho_team_cell:
        t.rho_team_cell[k] = -0.6
    R = p.corr_matrix(team, tiers, pos, mode="U")
    assert np.linalg.eigvalsh(R).min() > -1e-9 and np.allclose(np.diag(R), 1.0)
    # an already valid matrix passes through unchanged
    A = np.array([[1, .2], [.2, 1]])
    assert np.array_equal(K.nearest_corr(A), A)


# ------------------------------------------------------------------------------ pair stats

def test_pairstats_matches_brute_force():
    f = _frame(40, seed=4, rho_team=0.2)
    f["x"] = f["dk"] - f["mu"]
    f["x"] -= f["x"].mean()
    ps = K.PairStats(f["game_id"], f["team"], f["grp"], f["x"])
    got = ps.table("team", "tier")
    # brute force: same pooled-moment definition on every ordered-once pair
    var = f.groupby("grp")["x"].apply(lambda s: (s ** 2).mean()).to_dict()
    num, den = {}, {}
    for (gid, team), d in f.groupby(["game_id", "team"]):
        for (i, a), (j, b) in itertools.combinations(list(d.iterrows()), 2):
            lab = K.pair_label(K.TIERS[int(a.tier)], K.TIERS[int(b.tier)])
            num[lab] = num.get(lab, 0.0) + a.x * b.x
            den[lab] = den.get(lab, 0.0) + np.sqrt(var[a.grp] * var[b.grp])
    for lab in got.index:
        assert got.loc[lab, "r"] == pytest.approx(num[lab] / den[lab], rel=1e-9)
    # opponents
    opp_num = opp_den = 0.0
    for gid, d in f.groupby("game_id"):
        a, b = d[d.team == "AAA"], d[d.team == "BBB"]
        for _, x in a.iterrows():
            for _, y in b.iterrows():
                opp_num += x.x * y.x
                opp_den += np.sqrt(var[x.grp] * var[y.grp])
    assert ps.table("opp", "all").loc["all", "r"] == pytest.approx(opp_num / opp_den, rel=1e-9)


def test_recovers_planted_correlation_and_bootstrap_ci():
    f = _frame(900, seed=5, rho_team=0.25)
    z, _ = K.standardise(K.residuals(f, "U")[0])
    ps = K.PairStats(z["game_id"], z["team"], z["grp"], z["z"])
    t = ps.table("team", "all", nboot=100, seed=1)
    assert t.loc["all", "r"] == pytest.approx(0.25, abs=0.03)
    assert t.loc["all", "lo"] < 0.25 < t.loc["all", "hi"]
    o = ps.table("opp", "all", nboot=100)
    assert abs(o.loc["all", "r"]) < 0.03
    # conditioning on minutes etc. does not change a frame with g = 1 and min = min_proj
    assert np.allclose(K.residuals(f, "PM")[0]["res"], K.residuals(f, "U")[0]["res"])


# ------------------------------------------------------------------------------ causal tiers

def _player_games(seed=0, n_players=12, n_games=60):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_games):
        for p in range(n_players):
            if rng.random() < 0.1:
                continue
            m = (34, 33, 30, 28, 24, 22, 18, 16, 12, 9, 6, 4)[p] + rng.normal(0, 2)
            rows.append(dict(game_id=f"g{gi:03d}", team="AAA", player_id=p, date=gi,
                             min=max(m, 1.0), dk=max(m * (1.6 - 0.05 * p) + rng.normal(0, 6), 0)))
    return pd.DataFrame(rows)


def test_tiers_to_date_uses_only_prior_games():
    pg = _player_games()
    base = K.tiers_to_date(pg)
    assert set(base.unique()) <= {0, 1, 2}
    d_cut = 40
    # 1) changing the game at d (and everything after) never changes the tier at d
    pg2 = pg.copy()
    later = pg2["date"] >= d_cut
    pg2.loc[later, "min"] = pg2.loc[later, "min"].iloc[::-1].to_numpy()
    pg2.loc[later, "dk"] = 80.0
    t2 = K.tiers_to_date(pg2)
    at_d = (pg["date"] == d_cut).to_numpy()
    assert np.array_equal(base[at_d].to_numpy(), t2[at_d].to_numpy())
    before = (pg["date"] < d_cut).to_numpy()
    assert np.array_equal(base[before].to_numpy(), t2[before].to_numpy())
    # 2) dropping every game at/after d gives the same tiers up to d (no look-ahead)
    t3 = K.tiers_to_date(pg[pg["date"] <= d_cut].reset_index(drop=True))
    cut = pg[pg["date"] <= d_cut]
    assert np.array_equal(base.loc[cut.index].to_numpy(), t3.to_numpy())
    # 3) it does depend on the past (a past game changes a later tier)
    pg4 = pg.copy()
    pg4.loc[pg4["player_id"] == 0, "min"] = 5.0
    pg4.loc[pg4["player_id"] == 0, "dk"] = 2.0
    t4 = K.tiers_to_date(pg4)
    last = pg.index[pg["player_id"] == 0][-1]
    assert t4[last] == 2 and base[last] == 0
    # no team has more than two stars in a game
    assert (pd.concat([pg["game_id"], base], axis=1).groupby("game_id")["tier_td"]
            .apply(lambda s: (s == 0).sum()).max() <= 2)


def test_projection_tiers_are_the_copula_tiers():
    from courtside.proj import copula as C
    team = np.array(["A"] * 6)
    dk = np.array([40, 35, 30, 22, 10, 5.0])
    mins = np.array([35, 30, 29, 22, 15, 8.0])
    assert C.role_tiers(team, dk, mins).tolist() == [0, 0, 1, 1, 2, 2]
    assert (K.STAR_MIN, K.ROLE_MIN) == (C.STAR_MIN, C.ROLE_MIN)
