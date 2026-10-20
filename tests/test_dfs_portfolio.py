"""GPP contest portfolio (proj/dfs/portfolio.py, T2.12).

Synthetic pools (the field tests' helpers) and hand-built per-sim inputs: tie-splitting ladder,
joint ranks of our own entries, the greedy marginal (it telescopes to the joint expected payout
and matches brute force), exact exposure / overlap / lock / exclude / minimum caps on a real
build, the refill path, `field.evaluate(per_sim=True)` consistency, storage (migration 13) and
the realised (actual points) score.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF
from courtside.proj.dfs import rules as R

from test_dfs_field import hand_field, inputs_for, mk, sims_for, synth_pool, tiny_rules


def contest(m=10, tiers=((1, 1, 50.0), (2, 3, 20.0), (4, 5, 10.0)), fee=10.0):
    return F.Contest("c", "synthetic", fee, m,
                     [{"min_pos": a, "max_pos": b, "payout": p} for a, b, p in tiers],
                     "small_gpp", "preset")


# ============================================================================== ladder / ranks

def test_ladder_splits_ties_and_pays_nothing_past_the_field():
    lad = PF.Ladder(contest())
    assert lad.paid == 5 and lad.top1 == 1
    assert lad.g(np.array([1]), np.array([0]))[0] == 50.0
    assert lad.g(np.array([1]), np.array([1]))[0] == pytest.approx(35.0)     # ranks 1-2
    assert lad.g(np.array([5]), np.array([1]))[0] == pytest.approx(5.0)      # ranks 5-6
    assert lad.g(np.array([11]), np.array([0]))[0] == 0.0
    assert lad.g(np.array([30]), np.array([3]))[0] == 0.0


def test_joint_ranks_our_entries_push_each_other_down():
    score = np.array([[100, 90, 80], [50, 60, 70]])
    above = np.array([[0, 0, 0], [2, 1, 0]])
    r = PF.joint_ranks(score, above)
    assert r.tolist() == [[1, 2, 3], [5, 3, 1]]
    # equal scores: the later entry ranks first, the pair still fills two consecutive ranks
    r = PF.joint_ranks(np.array([[70, 70]]), np.array([[0, 0]]))
    assert sorted(r[0].tolist()) == [1, 2] and r[0, 1] == 1


def _brute(lad, P, idx):
    pay, _ = PF.joint_payouts(lad, P.score[:, idx], P.above[:, idx], P.tied[:, idx])
    return pay.sum(1).mean()


def _random_pool(S=400, C=12, seed=0, m=10):
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 60, (S, C)).astype(np.int32)
    # 9 other entries per sim with scores 0..60: above = how many beat each candidate
    others = rng.integers(0, 60, (S, m - 1))
    above = (others[:, None, :] > score[:, :, None]).sum(2).astype(np.int32)
    tied = (others[:, None, :] == score[:, :, None]).sum(2).astype(np.int32)
    return PF._Pool(score, above, tied)


def test_greedy_marginals_telescope_to_the_joint_payout():
    lad = PF.Ladder(contest())
    P = _random_pool()
    C = P.score.shape[1]
    inc = np.eye(C, dtype=np.int16)                # disjoint "lineups": no caps bind
    chosen, info = PF.greedy(lad, PF.CandSet(P, inc), 5, gamma=0, cap=np.full(C, 5),
                             low=np.zeros(C, int))
    assert len(chosen) == 5 and info["stopped"] is None
    assert sum(info["marginals"]) == pytest.approx(_brute(lad, P, chosen), rel=1e-9)
    # every step picks the best marginal by brute force
    for k in range(5):
        base = _brute(lad, P, chosen[:k]) if k else 0.0
        best = max(_brute(lad, P, chosen[:k] + [c]) - base for c in range(C) if c not in chosen[:k])
        assert info["marginals"][k] == pytest.approx(best, rel=1e-9)


def test_greedy_beats_individual_ranking_when_entries_collide():
    """Two near-identical strong entries cannibalise each other; the greedy takes one of them and
    a different second entry, the individual ranking takes both."""
    lad = PF.Ladder(contest(m=10, tiers=((1, 1, 100.0),)))
    S = 1000
    rng = np.random.default_rng(1)
    a = rng.integers(40, 60, S)
    score = np.stack([a, a - 1, rng.integers(30, 62, S)], 1).astype(np.int32)
    above = (score < 50).astype(np.int32)          # one other entry scores 50
    P = PF._Pool(score, above, np.zeros_like(score))
    ind = np.array([lad.g(above[:, c] + 1, np.zeros(S, int)).mean() for c in range(3)])
    assert ind[1] > ind[2]                         # individually the twin looks second best
    chosen, _ = PF.greedy(lad, PF.CandSet(P, np.eye(3, dtype=np.int16)), 2, gamma=0,
                          cap=np.full(3, 2), low=np.zeros(3, int))
    assert sorted(chosen) == [0, 2]
    assert _brute(lad, P, [0, 2]) > _brute(lad, P, [0, 1])


# ============================================================================== evaluate extension

def test_evaluate_per_sim_matches_single_entry_payout():
    pool = synth_pool(seed=3)
    inp = inputs_for(pool)
    con = F.preset_contest("gpp", 300, 2.0)
    f = F.build_field(pool, inp, con, n=2000, n_solves=20, seed=1, threads=1)
    fp, _ = F.align_sims(sims_for(pool, S=600), f.pids)
    lus = [[(s, f.pids[i]) for s, i in zip(f.slots, row)] for row in f.lineups[:5]]
    ev = F.evaluate(f, fp, con, lus, per_sim=True)
    ps = ev.per_sim
    lad = PF.Ladder(con)
    for j in range(5):
        g = lad.g(np.rint(ps["above"][:, j]).astype(int) + 1, np.rint(ps["tied"][:, j]).astype(int))
        assert g.mean() == pytest.approx(ev.lineups[j]["e_payout"], rel=1e-9, abs=1e-9)
    ev2 = F.evaluate(f, fp, con, lus, own_entries=20)
    assert ev2.info["opponents"] == 280 and ev.info["opponents"] == 299
    assert ev.per_sim is not None and F.evaluate(f, fp, con, lus).per_sim is None


# ============================================================================== real builds

SMALL = dict(n_scenario=30, n_ownpen=20, n_noise=10, n_topmean=10, n_sims=None, n_field=2000,
             field_solves=30, threads=1, solve_limit_s=2.0)


@pytest.fixture(scope="module")
def slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1500, seed=2)


def _counts(pf):
    cnt = {}
    for lu in pf.slots():
        for s in lu:
            cnt[s["player_id"]] = cnt.get(s["player_id"], 0) + 1
    return cnt


def test_build_respects_caps_exactly(slate):
    pool, inp, sims = slate
    by_val = sorted(pool.players, key=lambda p: -(p.fppg / p.salary))
    lock, excl, mn = by_val[5].dk_player_id, by_val[0].dk_player_id, by_val[9].dk_player_id
    s = PF.Settings(n=10, gamma=4, max_exposure=0.3, locks=(lock,), excludes=(excl,),
                    min_exp={mn: 0.4}, **SMALL)
    pf = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s)
    assert len(pf.lineups) == 10
    cnt = _counts(pf)
    assert cnt[lock] == 10                          # locked: every lineup
    assert excl not in cnt                          # excluded: never
    assert cnt[mn] >= 4                             # min 40 % of 10
    assert cnt[mn] <= 4                             # his own minimum raises his cap to 4
    assert all(c <= 3 for p, c in cnt.items() if p not in (lock, mn))   # max 30 % of 10 = 3
    sets = [{s["player_id"] for s in lu} for lu in pf.slots()]
    assert all(len(a & b) <= 4 for a, b in itertools.combinations(sets, 2))
    # lineups are legal DK lineups
    by_id = pool.by_id()
    for lu in pf.slots():
        assert [x["slot"] for x in lu] == list(pool.rules.slot_names)
        assert sum(by_id[x["player_id"]].salary for x in lu) <= pool.rules.salary_cap
        assert all(x["slot"] in by_id[x["player_id"]].slots for x in lu)
    sm = pf.summary
    assert sm["e_payout"] == pytest.approx(sum(x["e_payout"] for x in pf.lineups))
    assert sm["roi"] == pytest.approx(sm["e_payout"] / (10 * 3.0) - 1)
    assert 0 <= sm["p_any_cash"] <= 1 and sm["p_any_cash"] >= max(x["p_cash"] for x in pf.lineups)
    ex = {r["player_id"]: r for r in pf.exposure}
    assert ex[lock]["locked"] and ex[lock]["count"] == 10 and ex[lock]["share"] == 1.0
    assert pf.label == PF.LABEL and "not real results" in PF.LABEL


def test_refill_or_clean_stop_when_candidates_run_out(slate):
    pool, inp, sims = slate
    s = PF.Settings(n=12, gamma=3, max_exposure=0.25, **{**SMALL, "n_scenario": 4,
                                                          "n_ownpen": 2, "n_noise": 2,
                                                          "n_topmean": 2})
    pf = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s)
    assert pf.info["refills"] >= 1
    sets = [{x["player_id"] for x in lu} for lu in pf.slots()]
    assert all(len(a & b) <= 3 for a, b in itertools.combinations(sets, 2))
    assert all(c <= 3 for c in _counts(pf).values())
    assert len(pf.lineups) == 12 or pf.summary["stopped"] == "exhausted"


def test_exposure_limits_and_validation():
    s = PF.Settings(n=3, max_exposure=0.6)
    cap, low = PF.exposure_limits(s, ["a", "b"])
    assert cap == {"a": 3, "b": 3} and low == {}          # default cap only from N >= 5
    s = PF.Settings(n=20, max_exposure=0.6, max_exp={"b": 0.25}, min_exp={"a": 0.3},
                    locks=("c",))
    cap, low = PF.exposure_limits(s, ["a", "b", "c"])
    assert cap == {"a": 12, "b": 5, "c": 20} and low == {"a": 6, "c": 20}
    with pytest.raises(PF.PortfolioError):
        PF.exposure_limits(PF.Settings(n=10, min_exp={"a": 0.8}, max_exp={"a": 0.5}), ["a"])


def test_bad_settings_raise(slate):
    pool, inp, sims = slate
    con = F.preset_contest("gpp", 2000, 3.0)
    p = pool.players[0].dk_player_id
    with pytest.raises(PF.PortfolioError):
        PF.build_portfolio(pool, inp, sims, con, PF.Settings(n=2, locks=(p,), excludes=(p,)))
    with pytest.raises(PF.PortfolioError):
        PF.build_portfolio(pool, inp, sims, con, PF.Settings(n=2, locks=("nope",)))
    with pytest.raises(PF.PortfolioError):
        PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 5, 3.0), PF.Settings(n=5))


def test_realised_scores_actual_points_jointly():
    rules = tiny_rules(("UTIL",), 100_000)
    players = [mk(i, "A" if i % 2 else "B", "B" if i % 2 else "A", "PG", 5000, "G1", rules,
                  fppg=10.0) for i in range(1, 12)]
    pool = R.SlatePool("t", None, "classic", "x", rules, tuple(players))
    f = hand_field(pool, rules, [[str(i)] for i in range(1, 9)], field_size=10)
    pts = {str(i): float(i * 10) for i in range(1, 12)}
    pts["10"], pts["11"] = 95.0, 85.0                      # ours: ranks 1 and 2
    act = np.array([pts[p] for p in f.pids], np.float32)
    r = PF.realised(f, contest(), [[("UTIL", "10")], [("UTIL", "11")]], act)
    assert r["payout"] == pytest.approx(70.0) and r["cashed"] == 2 and r["best_rank"] == 1
    assert r["cost"] == 20.0 and r["roi"] == pytest.approx(2.5)


def test_save_load_roundtrip(tmp_path, slate):
    pool, inp, sims = slate
    s = PF.Settings(n=5, **SMALL)
    pf = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s,
                            slate_id="synth")
    conn = db.connect(tmp_path / "c.db")
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='dfs_portfolios'").fetchone()
    pid = PF.save(conn, pf)
    assert pid.startswith(PF.ID_PREFIX)
    d = PF.load(conn, pid)
    assert d["status"] == "ok" and d["n_built"] == 5 and len(d["lineups"]) == 5
    assert d["lineups"][0]["mode"] == "gpp" and d["lineups"][0]["slots"][0]["draftable_id"]
    assert d["data"]["summary"]["e_payout"] == pytest.approx(pf.summary["e_payout"])
    assert d["settings"]["n"] == 5
    assert [r["portfolio_id"] for r in PF.recent(conn, "synth")] == [pid]
    # portfolios stay out of dfs_runs (runner, grading and monitoring read every run)
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == 0
    PF.fail_row(conn, pid, "boom")
    assert PF.load(conn, pid)["status"] == "error"


# ============================================================================== truth / bands

def test_truth_shift_only_where_the_player_plays():
    rng = np.random.default_rng(0)
    fp = np.array([[10.0, 0.0, 30.0], [20.0, 0.0, 0.0], [30.0, 0.0, 30.0]], np.float32)
    inp = __import__("pandas").DataFrame({"pid": ["a", "b", "c"], "fppg": [30.0, 25.0, None]})
    same, info = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.0, 0.0, rng)
    assert same is fp and info["beta"] == 0.0
    x, info = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.5, 0.0, rng)
    assert np.allclose(x[:, 0], fp[:, 0] + 0.5 * (30 - 20))      # played mean 20 -> +5
    assert np.all(x[:, 1] == 0)                                    # never plays: untouched
    assert np.allclose(x[:, 2], fp[:, 2])                          # no public number
    assert info["shifted"] == 1
    y, _ = PF.truth_sims(fp, ["a", "b", "c"], inp, 0.0, 1.0, rng)
    assert y[1, 2] == 0 and y[0, 0] != 10 and np.all(y[:, 1] == 0)   # noise only when playing


def test_prob_band():
    assert PF.prob_band(0.2) == "~15\u201325%" and PF.prob_band(0.004) == "below 1%"
    assert PF.prob_band(0.99) == "above 97%" and PF.prob_band(0.5) == "~50\u201365%"
    assert PF.prob_band(None) == "n/a"


def test_relative_score_and_strengths_by_information(slate):
    pool, inp, sims = slate
    s = PF.Settings(n=4, strengths=True, strength_sims=800, **SMALL)
    pf = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s)
    sm = pf.summary
    assert sm["relative"] > 0 and sm["relative_se"] >= 0
    assert sm["baseline"]["e_payout"] > 0
    assert pf.info["truth"]["beta"] == PF.TRUTH_BETA and pf.info["truth"]["tau"] == PF.TRUTH_TAU
    inp = inp.assign(proj=inp["fppg"] * np.linspace(0.7, 1.3, len(inp)))   # ours != public
    soft, _ = PF.strength_field(pool.players, inp, F.preset_contest("gpp", 2000, 3.0), pool.rules,
                                "soft", s, seed=0)
    sharp, _ = PF.strength_field(pool.players, inp, F.preset_contest("gpp", 2000, 3.0),
                                 pool.rules, "sharp", s, seed=0)
    a = soft.ownership.inputs["pub_proj"].to_numpy()
    b = sharp.ownership.inputs["pub_proj"].to_numpy()
    ours = inp.set_index("pid").loc[soft.ownership.pids, "proj"].to_numpy(float)
    assert np.abs(b - ours).mean() < np.abs(a - ours).mean()      # sharper = closer to ours
    assert soft.mix == sharp.mix
