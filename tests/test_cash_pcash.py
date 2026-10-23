"""D2: cash lineup by P(cash) and Top N by mean then P(cash) (proj/dfs/cash_pcash.py,
topn.rank_by_mean; switch COURTSIDE_CASH_PCASH, default OFF).

OFF is bit-identical: no pick, no field reuse, the runner's lineups are the plain solves.
ON: candidates (Top N + λ grid, deduplicated), the pick is the max P(cash) on the select rows,
P(cash) is reported on the disjoint report rows, the Top N set is kept and only reordered within
0.5 DK near-ties, and the run's `score_run` reuses the pick's cash field."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest

from courtside.proj.dfs import cash_pcash as CP
from courtside.proj.dfs import field as F
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import runner
from courtside.proj.dfs import topn as T

from test_dfs_field import inputs_for, sims_for, synth_pool
from test_dfs_runner import G1, avail, conn, pipe, slate3  # noqa: F401  (fixtures)
from test_dfs_runner import T as TS


# --------------------------------------------------------------------------- switch

def test_switch_default_off_and_parsing(monkeypatch):
    assert CP.CASH_PCASH_DEFAULT == "off"
    assert CP.VERSION == "cash-pcash-1"
    assert CP.pcash_mode(None) == "off"
    for v in ("1", "true", "YES", "on", True):
        assert CP.pcash_mode(v) == "on"
    for v in ("0", "false", "no", "off", False):
        assert CP.pcash_mode(v, "on") == "off"
    assert CP.pcash_mode("junk", "off") == "off"
    monkeypatch.setattr(CP, "CASH_PCASH", False)
    CP.set_cash_pcash("on")
    assert CP.CASH_PCASH is True
    CP.set_cash_pcash(False)
    assert CP.CASH_PCASH is False


def test_reuse_fields_is_a_noop_without_a_pending_field():
    orig = F.field_for
    with CP.reuse_fields("nothing-pending"):
        assert F.field_for is orig
    assert F.field_for is orig


# --------------------------------------------------------------------------- rank_by_mean

def _fake_res(means, p=None):
    pool = synth_pool(1)
    by = list(pool.players)
    lus = []
    for k, m in enumerate(means):
        ents = tuple(O.LineupEntry("UTIL", by[k], 1000, m, m) for _ in range(1))
        lus.append(O.Lineup(ents, 1000, m, m))
    ranked = tuple(T.RankedLineup(i + 1, lu, T.diff_vs(lus[0], lu)) for i, lu in enumerate(lus))
    return T.TopNResult(ranked, len(means), None, 0.1)


def test_rank_by_mean_breaks_near_ties_by_pcash_only():
    res = _fake_res([300.0, 299.8, 299.6, 298.0, 297.9])
    out = T.rank_by_mean(res, [0.50, 0.60, 0.40, 0.30, 0.35])
    # groups: {300, 299.8} (299.6 is 0.4 from the leader: also in), then {298.0, 297.9}
    assert [r.lineup.projected for r in out] == [299.8, 300.0, 299.6, 297.9, 298.0]
    assert [r.rank for r in out] == [1, 2, 3, 4, 5]
    assert {id(r.lineup) for r in out} == {id(r.lineup) for r in res}       # same set
    assert out.lineups[0].diff.n_different == 0
    assert out.lineups[1].diff.d_projected == pytest.approx(0.2)


def test_rank_by_mean_without_ties_is_solve_order_and_none_goes_last():
    res = _fake_res([300.0, 299.0, 298.0])
    out = T.rank_by_mean(res, [0.1, 0.9, 0.5])
    assert [r.lineup for r in out] == [r.lineup for r in res]
    out2 = T.rank_by_mean(_fake_res([300.0, 300.0]), [None, 0.2])
    assert [r.lineup.entries[0].player.dk_player_id for r in out2] == ["2", "1"]
    with pytest.raises(ValueError):
        T.rank_by_mean(res, [0.1])


# --------------------------------------------------------------------------- choose

@pytest.fixture(scope="module")
def slate():
    pool = synth_pool(11, games=4)
    projs = {p.dk_player_id: O.Projection(p.fppg, p25=p.fppg * (0.55 + 0.4 * ((i * 7) % 5) / 4))
             for i, p in enumerate(pool.players)}
    con = F.preset_contest("double_up", 500, 5)
    f = F.build_field(pool, inputs_for(pool), con, n=5000, n_solves=20, seed=3)
    fp, _ = F.align_sims(sims_for(pool, 1200, seed=4), f.pids)
    return pool, projs, con, f, fp


def _solves(pool, projs, n=6):
    cash = O.optimize(pool, projs, mode="cash", lam=0.3)
    res = T.top_n(pool, projs, n=n, mode="mean")
    return cash, res


def test_candidates_dedupe_and_grid(slate):
    pool, projs, *_ = slate
    cash, res = _solves(pool, projs)
    c = CP.candidates(O, pool, projs, pool.rules, set(), None, cash, res)
    srcs = [s for s, _ in c]
    assert srcs[0] == "lam0.3" and c[0][1] is cash
    assert "lam0" not in srcs                         # λ = 0 is Top N #1
    ids = [CP._ident(pool.rules, lu) for _, lu in c]
    assert len(ids) == len(set(ids))
    for lam in (0.15, 0.5):
        lu = O.optimize(pool, projs, mode="cash", lam=lam)
        assert CP._ident(pool.rules, lu) in ids


def test_choose_picks_max_select_pcash_reports_on_disjoint_rows(slate, monkeypatch):
    pool, projs, con, f, fp = slate
    cash, res = _solves(pool, projs)
    monkeypatch.setattr(CP, "cash_field", lambda *a, **k: ({"info": {"run_id": "r1"}}, con, f, fp))
    new_cash, new_res, info = CP.choose(None, "synth", "r1", O, T, pool, projs, pool.rules,
                                        set(), None, cash, res)
    assert info["version"] == "cash-pcash-1" and info["status"] == "ok"
    ps = [c["p_cash_select"] for c in info["candidates"]]
    k = int(np.argmax(ps))
    assert info["pick"] == info["candidates"][k]["src"]
    # recompute both numbers by hand on the split rows
    rs, rr = CP.split(fp.shape[0])
    assert len(set(rs) & set(rr)) == 0 and len(rs) + len(rr) == fp.shape[0]
    assert len(rr) == round(0.45 * fp.shape[0])
    ev_s = F.evaluate(f, fp[rs], con, [new_cash.slot_pairs()], seed=0, tiers=False)
    ev_r = F.evaluate(f, fp[rr], con, [new_cash.slot_pairs()], seed=1, tiers=False)
    assert ev_s.lineups[0]["p_cash"] == pytest.approx(max(ps), abs=1e-4)
    assert info["p_cash_report"] == pytest.approx(ev_r.lineups[0]["p_cash"], abs=1e-4)
    # Top N: same set, mean order up to 0.5 DK near-ties
    assert {r.lineup.player_ids for r in new_res} == {r.lineup.player_ids for r in res}
    m = [r.lineup.projected for r in new_res]
    assert all(m[i + 1] <= m[i] + T.NEAR_TIE_DK + 1e-9 for i in range(len(m) - 1))
    assert sorted(info["topn_order"]) == list(range(1, len(res) + 1))
    # the cash field is pending for the run's score_run, and handed out exactly once
    with CP.reuse_fields("r1"):
        got = F.field_for({"info": {"run_id": "r1"}}, con, n=CP.FIELD_N, mix=None,
                          n_solves=F.N_SOLVES, seed=0)
        assert got is f
    assert "r1" not in CP._PENDING


def test_pick_ties_go_to_the_rule_lineup():
    assert CP.pick([("lam0.3", 1), ("topn1", 2), ("topn2", 3)], [0.6, 0.6, 0.59]) == 0
    assert CP.pick([("lam0.3", 1), ("topn1", 2)], [0.6, 0.61]) == 1


def test_choose_safe_keeps_off_lineups_on_error(slate, monkeypatch):
    pool, projs, *_ = slate
    cash, res = _solves(pool, projs, 3)

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(CP, "cash_field", boom)
    c2, r2, info = CP.choose_safe(None, "s", "r", O, T, pool, projs, pool.rules, set(), None,
                                  cash, res)
    assert c2 is cash and r2 is res and info["status"].startswith("error")
    monkeypatch.setattr(CP, "cash_field", lambda *a, **k: None)
    c3, r3, info3 = CP.choose(None, "s", "r", O, T, pool, projs, pool.rules, set(), None, cash,
                              res)
    assert c3 is cash and r3 is res and info3["status"] == "no_sims"


# --------------------------------------------------------------------------- runner

def _saved(conn, kind=None):
    rid = conn.execute("SELECT run_id, detail FROM dfs_runs WHERE ok=1 ORDER BY created_at DESC "
                       "LIMIT 1").fetchone()
    rows = conn.execute("SELECT mode, rank, slots_json, salary, mean FROM dfs_lineups WHERE "
                        "run_id=? ORDER BY mode, rank", (rid[0],)).fetchall()
    return json.loads(rid[1]), [tuple(r) for r in rows]


def test_runner_off_is_the_plain_solves_and_never_calls_the_pick(conn, slate3, pipe, avail,
                                                                 monkeypatch):
    monkeypatch.setattr(CP, "CASH_PCASH", False)

    def boom(*a, **k):
        raise AssertionError("pick called with the switch OFF")
    monkeypatch.setattr(CP, "choose_safe", boom)
    monkeypatch.setattr(CP, "choose", boom)
    now = TS(G1) - timedelta(minutes=20)
    runner.run_slate(conn, "S1", "manual", now=now, refresh=False, n=4)
    detail, rows = _saved(conn)
    assert CP.DETAIL_KEY not in detail
    pool = R.build_pool(conn, "S1")
    projs = {str(k): v for k, v in pipe.project_slate(conn, "S1", now, run_id="x").items()}
    bans, _ = runner.availability_bans(pool, {p.dk_player_id: {"p": 1.0, "status": "",
                                                               "source": "stub"}
                                              for p in pool.players}, None, now)
    cash, res = runner._build(O, T, pool, projs, pool.rules, bans, now, 4, 1, {})
    want = [("cash", 1, cash.slots_json(), cash.salary, cash.projected)] + [
        ("topn", r.rank, r.lineup.slots_json(), r.lineup.salary, r.lineup.projected) for r in res]
    assert rows == want


def test_runner_on_without_sims_keeps_lineups_and_records_status(conn, slate3, pipe, avail,
                                                                 monkeypatch):
    now = TS(G1) - timedelta(minutes=20)
    monkeypatch.setattr(CP, "CASH_PCASH", False)
    runner.run_slate(conn, "S1", "manual", now=now, refresh=False, n=4)
    _, off_rows = _saved(conn)
    monkeypatch.setattr(CP, "CASH_PCASH", True)
    runner.run_slate(conn, "S1", "manual", now=now + timedelta(seconds=1), refresh=False, n=4,
                     force=True)
    detail, on_rows = _saved(conn)
    assert detail[CP.DETAIL_KEY]["status"] == "no_sims"
    assert on_rows == off_rows


def test_runner_on_saves_the_pick_and_reorders_topn(conn, slate3, pipe, avail, monkeypatch):
    now = TS(G1) - timedelta(minutes=20)
    pool = R.build_pool(conn, "S1")
    con = F.preset_contest("double_up", 400, 5)
    f = F.build_field(pool, inputs_for(pool), con, n=4000, n_solves=10, seed=2)
    # sims that favour low-salary players (so the P(cash) pick can differ from the mean pick)
    rng = np.random.default_rng(3)
    mu = np.array([(p.fppg or 0) for p in pool.players])
    fp = np.maximum(0, rng.normal(mu, 0.5 * mu + 2, (900, len(mu)))).astype(np.float32)
    X, _ = F.align_sims({"fp": fp, "dk_id": np.array([p.dk_player_id for p in pool.players])},
                        f.pids)
    seen = {}

    def fake_field(conn_, sid, run_id, **k):
        seen["run"] = run_id
        return {"info": {"run_id": run_id}}, con, f, X
    monkeypatch.setattr(CP, "cash_field", fake_field)
    monkeypatch.setattr(CP, "CASH_PCASH", True)
    runner.run_slate(conn, "S1", "manual", now=now, refresh=False, n=5)
    detail, rows = _saved(conn)
    info = detail[CP.DETAIL_KEY]
    assert info["status"] == "ok" and info["version"] == CP.VERSION
    cands = {c["src"]: c for c in info["candidates"]}
    best = max(info["candidates"], key=lambda c: c["p_cash_select"])
    assert cands[info["pick"]]["p_cash_select"] == best["p_cash_select"]
    cash_row = next(r for r in rows if r[0] == "cash")
    assert cash_row[4] == pytest.approx(cands[info["pick"]]["mean"], abs=0.01)
    top = [r for r in rows if r[0] == "topn"]
    assert [r[1] for r in top] == list(range(1, len(top) + 1))
    assert sorted(info["topn_order"]) == list(range(1, len(top) + 1))
    assert seen["run"] not in CP._PENDING              # score_run consumed / cleared it
