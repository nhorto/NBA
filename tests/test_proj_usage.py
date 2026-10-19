"""Usage v1 with/without factors (T2.3): shrinkage, no lookahead, bounded multi-absence,
factors = 1 with nobody absent, JSON round-trip, sim consistency."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj import state, usage as U

SEASONS = {"2021-22": date(2021, 10, 20), "2022-23": date(2022, 10, 20),
           "2023-24": date(2023, 10, 20)}
N_GAMES = 50
STAR = 1                      # team AAA's star; sits ~25% of games
LIFT = 1.25                   # true per-minute lift of AAA's other players when STAR sits
EMPTY_TALLIES = {"pairs": pd.DataFrame(), "totals": pd.DataFrame()}


def _fill(c: sqlite3.Connection, seed: int = 0, scramble_from: date | None = None) -> None:
    rng = np.random.default_rng(seed)
    pids = list(range(1, 21))
    mins = {p: float(rng.integers(16, 34)) for p in pids}
    mins[STAR] = 35.0
    sits = rng.random(len(SEASONS) * N_GAMES) < 0.25
    rows, gid = [], 0
    rs = np.random.default_rng(seed + 99)           # independent stream for scrambled rows
    for season, start in SEASONS.items():
        for k in range(N_GAMES):
            day = start + timedelta(days=2 * k)
            star_out = bool(sits[gid]) and k >= 3
            gid += 1
            for p in pids:
                team = "AAA" if p <= 10 else "BBB"
                if p == STAR and star_out:
                    continue
                lift = LIFT if (team == "AAA" and star_out) else 1.0
                base = 0.8 if p == STAR else 0.4
                g = rs if (scramble_from and day >= scramble_from) else rng
                m = float(np.clip(g.normal(mins[p], 3), 6, 42))
                pts = int(g.poisson(base * lift * m))
                o = {s: int(g.poisson(r * lift * m)) for s, r in
                     (("fg3m", 0.06), ("oreb", 0.05), ("dreb", 0.13), ("ast", 0.1),
                      ("stl", 0.03), ("blk", 0.02), ("tov", 0.06), ("fga", 0.35), ("fta", 0.1),
                      ("pf", 0.09))}
                rows.append((f"G{gid:05d}", p, f"P{p}", team, season, "Regular Season",
                             day.isoformat(), m, 0, o["fga"], o["fg3m"], 0, 0, o["fta"],
                             o["oreb"], o["dreb"], o["oreb"] + o["dreb"], o["ast"], o["stl"],
                             o["blk"], o["tov"], o["pf"], pts, 0.0))
    c.executemany(
        "INSERT INTO player_games(game_id, player_id, player_name, team, season, season_type, "
        "game_date, min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, "
        "pts, plus_minus) VALUES (" + ",".join("?" * 24) + ")", rows)
    pos = ["G", "F", "C"]
    for season in SEASONS:
        c.executemany("INSERT INTO team_rosters(season, team_id, team, player_id, position) "
                      "VALUES (?,?,?,?,?)",
                      [(season, 1 if p <= 10 else 2, "AAA" if p <= 10 else "BBB", p, pos[p % 3])
                       for p in pids])
    c.commit()


def _conn(**kw) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.migrate(c)
    _fill(c, **kw)
    return c


def _sp():
    p = state.default_params()
    p["history_seasons"] = 1
    return p


def _data(c) -> U.UsageData:
    return U.UsageData(c, list(SEASONS), _sp(), tallies=EMPTY_TALLIES)


@pytest.fixture(scope="module")
def data():
    return _data(_conn())


# ------------------------------------------------------------------------------ shrinkage

def test_shrink_term_limits_and_monotone():
    assert np.allclose(U.shrink_term(1.6, 0.0, 100.0), 0.0)
    assert np.allclose(U.shrink_term(1.6, 500.0, float("inf")), 0.0)
    big = U.shrink_term(1.6, 1e9, 100.0)
    assert np.isclose(big, np.log(1.6), atol=1e-6)
    ns = np.array([0, 10, 100, 1000, 10000.0])
    v = U.shrink_term(np.full(5, 1.6), ns, 200.0)
    assert np.all(np.diff(v) > 0) and v[-1] < np.log(1.6)
    # spec form: phi = 1 + (phi_emp - 1) n / (n + K)
    assert np.isclose(np.exp(U.shrink_term(1.6, 100.0, 300.0)), 1 + 0.6 * 100 / 400)


def test_stint_ratio_pseudocount_shrinks_small_samples():
    q = pd.DataFrame({"po": [2.0, 2000.0, 0.0], "pt": [40.0, 40000.0, 40.0]})
    for s in U.DK_STATS:
        q[f"yo_{s}"] = [2.0, 2000.0, 0.0]          # off: 1 per poss
        q[f"yt_{s}"] = [21.0, 21000.0, 20.0]       # on: 19/38 = 0.5 per poss
    r = U.stint_ratios(q, np.ones(3))
    assert 1.0 < r["pts"][0] < r["pts"][1]          # tiny sample pulled towards 1
    assert np.isclose(r["pts"][1], 2.0, rtol=0.01)
    assert r["pts"][2] == 1.0                       # no off exposure -> no evidence


# ------------------------------------------------------------------------------ fit / factors

def test_fit_recovers_lift_and_beats_none(data):
    train = (data.rows["season"] < "2023-24").to_numpy()
    p = U.fit_frame(data, train, k_grid=(100.0, float("inf")))
    lg = U.predict_log(data, p)
    r = data.rows
    sel = ((r["season"] == "2023-24") & (r["team"] == "AAA")).to_numpy()
    out = U._pair_sum(data, data.pairs["star_j"].to_numpy(float)) > 0
    m = sel & out
    assert m.sum() > 20
    assert np.exp(lg["pts"][m]).mean() > 1.08               # true lift 1.25 (shrunk)
    ll_m = U.row_ll(r, {s: np.exp(lg[s]) for s in U.DK_STATS})
    ll_0 = U.row_ll(r, None)
    assert sum(ll_m[s][m].sum() - ll_0[s][m].sum() for s in U.DK_STATS) > 0
    # healthy rows are untouched
    assert np.all(lg["pts"][~(U._pair_sum(data, np.ones(len(data.pairs))) > 0)] == 0.0)


def test_no_lookahead_scrambled_future():
    cut = date(2023, 12, 1)
    a = _data(_conn())
    b = _data(_conn(scramble_from=cut))
    cut_day = state._day_of(cut)
    ra = a.rows[a.rows["day"] < cut_day].reset_index(drop=True)
    rb = b.rows[b.rows["day"] < cut_day].reset_index(drop=True)
    cols = ["game_id", "player_id", "sat", "bump_role"] + [f"E_{s}" for s in U.DK_STATS]
    pd.testing.assert_frame_equal(ra[cols], rb[cols])
    pa = a.pairs[a.pairs["day"] < cut_day].reset_index(drop=True)
    pb = b.pairs[b.pairs["day"] < cut_day].reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb)
    # params fit on seasons before the cut season are identical; box evidence before the cut too
    pa_ = U.fit_frame(a, (a.rows["season"] < "2023-24").to_numpy(), k_grid=(100.0,))
    pb_ = U.fit_frame(b, (b.rows["season"] < "2023-24").to_numpy(), k_grid=(100.0,))
    assert pa_.to_json() == pb_.to_json()
    la, lb = U.predict_log(a, pa_), U.predict_log(b, pb_)
    ma = (a.rows["day"] < cut_day).to_numpy()
    mb = (b.rows["day"] < cut_day).to_numpy()
    for s in U.DK_STATS:
        assert np.array_equal(la[s][ma], lb[s][mb])
    ev_a = a.evidence(cut, [1, 2, 3])
    ev_b = b.evidence(cut, [1, 2, 3])
    for k in ev_a:
        assert np.array_equal(ev_a[k], ev_b[k])


def _team(n=8, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tp = pd.DataFrame({"player_id": np.arange(1, n + 1), "mpg": rng.uniform(12, 36, n),
                       "grp": rng.integers(0, 3, n), "regular": True})
    tp["star"] = tp["mpg"].rank(ascending=False) <= 2
    for s in U.DK_STATS:
        tp[f"r_{s}"] = rng.uniform(0.01, 0.5, n)
    return tp


def _strong_params() -> U.UsageParams:
    p = U.default_params()
    p.beta = {s: [3.0] * U.N_CELL for s in U.DK_STATS}
    p.alpha = {s: [0.2] * len(U.ROLES) for s in U.DK_STATS}
    return p


def test_factors_one_when_nobody_absent():
    tp = _team()
    f = U.factors(_strong_params(), tp, np.zeros(len(tp), bool))
    assert all(v == 1.0 for d in f.values() for v in d.values())
    F = U.factor_matrix(U.pair_matrix(_strong_params(), tp), np.zeros((5, len(tp)), bool))
    assert all(np.all(v == 1.0) for v in F.values())
    f0 = U.factors(U.default_params(), tp, [1, 2, 3])
    assert all(v == 1.0 for d in f0.values() for v in d.values())


def test_multi_absence_bounded_and_additive():
    tp = _team(12)
    p = _strong_params()
    L = U.pair_matrix(p, tp)
    ab = np.zeros(12, bool)
    ab[:6] = True
    F = U.factor_matrix(L, ab, p.cap)
    for s in U.DK_STATS:
        assert np.all(F[s] <= np.exp(p.cap) + 1e-12) and np.all(F[s] >= np.exp(-p.cap) - 1e-12)
    # two small absences add on the log scale (below the cap)
    p2 = U.default_params()
    p2.beta = {s: [0.1] * U.N_CELL for s in U.DK_STATS}
    L2 = U.pair_matrix(p2, tp)
    one = np.log(U.factor_matrix(L2, np.eye(12, dtype=bool)[0])["pts"])
    two = np.log(U.factor_matrix(L2, np.eye(12, dtype=bool)[1])["pts"])
    both = np.log(U.factor_matrix(L2, np.eye(12, dtype=bool)[0] | np.eye(12, dtype=bool)[1])["pts"])
    assert np.allclose(both[0, 2:], one[0, 2:] + two[0, 2:])


def test_non_regular_absence_does_not_trigger():
    tp = _team()
    tp.loc[0, "regular"] = False
    ab = np.zeros(len(tp), bool)
    ab[0] = True
    F = U.factor_matrix(U.pair_matrix(_strong_params(), tp), ab)
    assert all(np.all(v == 1.0) for v in F.values())


def test_params_json_round_trip(data):
    p = U.fit_frame(data, (data.rows["season"] < "2023-24").to_numpy(),
                    k_grid=(50.0, float("inf")))
    q = U.UsageParams.from_json(p.to_json())
    assert q.to_json() == p.to_json()
    assert q.k_stint == p.k_stint and q.beta == p.beta
    d = U.default_params()
    assert U.UsageParams.from_json(d.to_json()).k_box["pts"] == float("inf")


def test_sim_factors_match_fitted_rows(data):
    p = U.fit_frame(data, (data.rows["season"] < "2023-24").to_numpy(), k_grid=(100.0,))
    lg = U.predict_log(data, p)
    r = data.rows
    tgs = r[(r["season"] == "2023-24") & r["sat"]][["game_id", "team"]].drop_duplicates()
    assert len(tgs)
    for g, t in tgs.head(5).itertuples(index=False):
        tp = data.regs[(data.regs["game_id"] == g) & (data.regs["team"] == t)]
        tp = tp.reset_index(drop=True)
        ev = data.evidence(int(tp["day"].iloc[0]), tp["player_id"])
        f = U.factors(p, tp, ~tp["played"].to_numpy(bool), ev)
        for row in r[(r["game_id"] == g) & (r["team"] == t)].itertuples():
            if row.player_id in f:
                assert np.isclose(np.log(f[row.player_id]["pts"]), lg["pts"][row.r])


def test_sim_means_matches_rate_times_minutes():
    tp = _team(4)
    sp = state.default_params()
    x = np.array([[30.0, 20.0, 0.0, 10.0]])
    played = x > 0
    F = {s: np.full((1, 4), 1.1) for s in U.DK_STATS}
    pts, oth, st = U.sim_means(tp, sp, x, played, F)
    assert np.allclose(pts, tp["r_pts"].to_numpy() * x * 1.1)
    assert np.allclose(st["reb"], tp["r_reb"].to_numpy() * x * 1.1)
    exp_oth = sum(U.DK_W[s] * tp[f"r_{s}"].to_numpy() * x * 1.1 for s in U.OTHER_STATS)
    assert np.allclose(oth, exp_oth)
