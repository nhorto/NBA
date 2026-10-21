"""The contest builder's "how many entries" curve (proj/dfs/portfolio.entries_curve).

Hand-built per-sim pools (a shared "core" shock makes lineups bust together) and a real build
on the synthetic slate: chances are non-decreasing in N, the N = 1 point is the single-lineup
summary, correlated entries fall below the independence formula, the curve stops at the
contest's max entries, each point keeps its own N's caps and nests, and the display is banded
(odds for rare events) until payouts are calibrated.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import portfolio as PF
from courtside.web import dfs_contest

from test_dfs_field import inputs_for, sims_for, synth_pool
from test_dfs_portfolio import SMALL, _fake_prepare

METRICS = ("p_any_cash", "p_any_top1", "p_any_top01", "p_first")


def gpp(m=2000, fee=3.0):
    return F.Contest("c", "synthetic GPP", fee, m,
                     [{"min_pos": 1, "max_pos": 1, "payout": 500.0},
                      {"min_pos": 2, "max_pos": 20, "payout": 30.0},
                      {"min_pos": 21, "max_pos": 400, "payout": 5.0}], "large_gpp", "preset")


def correlated_pool(S=4000, C=30, m=2000, rho=0.8, seed=0):
    """Per-sim inputs for C lineups whose scores share a common shock (weight rho): the field's
    distribution is fixed, so on a bad night for the shared core every lineup sinks together."""
    rng = np.random.default_rng(seed)
    core = rng.standard_normal((S, 1))
    z = rho * core + np.sqrt(1 - rho ** 2) * rng.standard_normal((S, C))
    q = 1 / (1 + np.exp(-1.7 * (z + 0.4)))              # quantile of the field beaten (0..1)
    above = np.rint((1 - q) * (m - 1)).astype(np.int32)
    score = np.rint(q * 10_000).astype(np.int32)
    tied = np.zeros((S, C), np.int32)
    return PF._Pool(score, above, tied)


def _curve(P, contest, chosen, top, s=None, inc=None):
    C = P.score.shape[1]
    inc = np.eye(C, dtype=np.int16) if inc is None else inc
    pids = [f"p{i}" for i in range(inc.shape[1])]
    s = s or PF.Settings(n=len(chosen), curve_to=top)
    lad = PF.Ladder(contest)
    return PF.entries_curve(lad, PF.CandSet(P, inc), chosen, s, pids, contest,
                            m_o=contest.field_size - len(chosen), gamma=8)


def _greedy(P, contest, n):
    C = P.score.shape[1]
    ch, _ = PF.greedy(PF.Ladder(contest), PF.CandSet(P, np.eye(C, dtype=np.int16)), n, gamma=8,
                      cap=np.full(C, n), low=np.zeros(C, int))
    return ch


# ============================================================================== pure curve

def test_points_and_cap_at_contest_max():
    assert PF.curve_points(20, 150) == [1, 3, 5, 10, 20, 50, 150]
    assert PF.curve_points(12, 150) == [1, 3, 5, 10, 12, 20, 50, 150]     # the build's own N
    assert PF.curve_points(20, 25) == [1, 3, 5, 10, 20, 25]               # contest max 25
    assert PF.curve_points(3, 3) == [1, 3]
    assert PF.curve_points(1, 1) == [1]
    assert PF.curve_points(5, None) == [1, 3, 5]                          # no max: up to N
    assert PF.curve_points(20, 500) == [1, 3, 5, 10, 20, 50, 150]         # never past 150
    con = gpp()
    P = correlated_pool(C=20)
    cv = _curve(P, con, _greedy(P, con, 5), top=12)
    assert [r["requested"] for r in cv["points"]] == [1, 3, 5, 10, 12]   # 12 max: no 20 point
    assert cv["complete"] is True and cv["points"][-1]["entries"] == 12
    # a pool too small for the largest point: the curve stops short and says so
    cv = _curve(P, con, _greedy(P, con, 5), top=50)
    last = cv["points"][-1]
    assert last["entries"] == 20 and last["requested"] == 50 and cv["complete"] is False
    assert last["stopped"] == "exhausted"


def test_monotone_nested_and_n1_is_the_single_lineup_summary():
    con = gpp()
    P = correlated_pool()
    chosen = _greedy(P, con, 10)
    cv = _curve(P, con, chosen, top=20)
    pts = cv["points"]
    assert [r["entries"] for r in pts] == [1, 3, 5, 10, 20]
    for m in METRICS:
        v = [r[m] for r in pts]
        assert all(b >= a for a, b in zip(v, v[1:])), (m, v)
    assert pts[-1]["p_any_cash"] > pts[0]["p_any_cash"]
    assert [r["cost"] for r in pts] == [3.0, 9.0, 15.0, 30.0, 60.0]
    # sources: subsets of the build below N, the build at N, the build plus picks above
    assert [r["source"] for r in pts] == ["subset"] * 3 + ["build", "extended"]
    assert all(set(a["build_ranks"]) <= set(b["build_ranks"]) for a, b in zip(pts[:3], pts[1:4]))
    # N = 1 is exactly the single-lineup summary of that lineup (and of a 1-entry build)
    lad = PF.Ladder(con)
    j = chosen[pts[0]["build_ranks"][0] - 1]
    one = _curve(P, con, [j], top=1)["points"][0]
    pay, rank = PF.joint_payouts(lad, P.score[:, [j]], P.above[:, [j]], P.tied[:, [j]])
    sm = PF._summary(lad, pay, rank, con)
    assert one["p_any_cash"] == pytest.approx(sm["p_any_cash"])
    assert one["p_any_top1"] == pytest.approx(sm["p_any_top1"])
    assert one["e_payout"] == pytest.approx(sm["e_payout"])
    assert pts[0]["p_any_cash"] == pytest.approx(sm["p_any_cash"], abs=0.01)   # 9 more others
    # the build's own point equals the joint summary of the whole set
    pay, rank = PF.joint_payouts(lad, P.score[:, chosen], P.above[:, chosen], P.tied[:, chosen])
    sm = PF._summary(lad, pay, rank, con)
    mine = pts[3]
    assert mine["p_any_cash"] == pytest.approx(sm["p_any_cash"])
    assert mine["p_any_top1"] == pytest.approx(sm["p_any_top1"])
    assert mine["e_payout"] == pytest.approx(sm["e_payout"])


def test_correlated_entries_fall_below_the_independence_formula():
    con = gpp()
    P = correlated_pool(rho=0.85)
    cv = _curve(P, con, _greedy(P, con, 20), top=20)
    for r in cv["points"][1:]:                                    # N >= 3
        assert r["p_any_cash"] < r["if_independent"]["p_any_cash"] - 0.02, r
        assert r["p_any_top1"] < r["if_independent"]["p_any_top1"]
    # 1 - (1 - p)^N with every lineup's own chance is what the sims would give if independent
    P0 = correlated_pool(rho=0.0, S=20000, C=10, seed=3)
    r = _curve(P0, con, list(range(10)), top=10)["points"][-1]
    assert r["p_any_cash"] == pytest.approx(r["if_independent"]["p_any_cash"], abs=0.01)


def test_each_point_keeps_its_own_exposure_caps():
    """Player 0 is in every strong lineup: the 10-entry set may use him 6 times, the 5-entry set
    only 3, so the first 5 picks are NOT the 5-entry set (greedy is not prefix-nested)."""
    con = gpp()
    P = correlated_pool(C=24, rho=0.3, seed=5)
    C = P.score.shape[1]
    inc = np.zeros((C, C + 1), np.int16)
    inc[np.arange(C), np.arange(C) + 1] = 1
    strong = np.argsort(-P.score.mean(0))[:12]
    inc[strong, 0] = 1
    lad = PF.Ladder(con)
    pids = [f"p{i}" for i in range(C + 1)]
    s = PF.Settings(n=10, max_exposure=0.6, curve_to=20, curve_select_sims=1500)
    cap, low = PF._caps_for(s, 10, pids)
    chosen, _ = PF.greedy(lad, PF.CandSet(P, inc), 10, gamma=8, cap=cap, low=low)
    assert inc[chosen[:5], 0].sum() > 3                    # the prefix breaks the 5-entry cap
    cv = PF.entries_curve(lad, PF.CandSet(P, inc), chosen, s, pids, con,
                          m_o=con.field_size - 10, gamma=8)
    by_n = {r["requested"]: r for r in cv["points"]}
    use = lambda r: sum(inc[chosen[k - 1], 0] for k in r["build_ranks"])  # noqa: E731
    assert use(by_n[5]) <= 3 and not by_n[5]["is_prefix"] and cv["nested_prefix"] is False
    assert use(by_n[3]) <= 3 and use(by_n[10]) <= 6
    assert by_n[20]["source"] == "extended" and by_n[20]["entries"] == 20
    assert cv["select_sims"] == 1500 and by_n[20]["n_sims"] == P.score.shape[0]   # picked on
    #                                                     a subsample, scored on every sim
    for m in METRICS:
        v = [r[m] for r in cv["points"]]
        assert all(b >= a for a, b in zip(v, v[1:])), (m, v)


def test_cap_subset_meets_caps_when_greedy_gets_stuck():
    # players A, B: lineups 0-2 hold both, 3 only A, 4 only B, 5-6 neither
    inc = np.array([[1, 1], [1, 1], [1, 1], [1, 0], [0, 1], [0, 0], [0, 0]], np.int16)
    parent = list(range(7))
    cap = np.array([2, 2])
    got = PF._cap_subset(parent, [0, 1, 2], 5, inc, cap, np.zeros(2, int))
    assert got is not None and len(got) == 5
    assert (inc[got].sum(0) <= cap).all() and 0 in got          # keeps the earliest picks
    assert PF._cap_subset(parent, [], 6, inc, np.array([1, 1]), np.zeros(2, int)) is None
    fill = PF._fill_least_over([0], parent, 6, inc, np.array([1, 1]))
    assert len(fill) == 6 and inc[fill].sum(0).tolist() == [3, 3]  # leaves out one of 1-2


# ============================================================================== real build

@pytest.fixture(scope="module")
def slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1500, seed=2)


def test_real_build_curve_matches_summary_and_stays_capped(slate):
    pool, inp, sims = slate
    s = PF.Settings(n=10, max_exposure=0.5, curve_to=20, **SMALL)
    pf = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s)
    cv = pf.summary["curve"]
    pts = cv["points"]
    assert [r["requested"] for r in pts] == [1, 3, 5, 10, 20]
    mine = next(r for r in pts if r["source"] == "build")
    assert mine["entries"] == 10 and mine["build_ranks"] == list(range(1, 11))
    assert mine["p_any_cash"] == pytest.approx(pf.summary["p_any_cash"])
    assert mine["p_any_top1"] == pytest.approx(pf.summary["p_any_top1"])
    assert mine["e_payout"] == pytest.approx(pf.summary["e_payout"])
    for m in METRICS:
        v = [r[m] for r in pts]
        assert all(b >= a for a, b in zip(v, v[1:])), (m, v)
    slots = pf.slots()
    for r in pts:
        if r["source"] != "subset":
            continue
        cnt: dict = {}
        for k in r["build_ranks"]:
            for x in slots[k - 1]:
                cnt[x["player_id"]] = cnt.get(x["player_id"], 0) + 1
        k = r["entries"]
        lim = k if k < 5 else int(np.floor(0.5 * k + 1e-9))
        assert max(cnt.values()) <= lim or r["caps_relaxed"], (k, max(cnt.values()))
    assert pf.info["timing"]["curve"] >= 0
    # no curve when switched off; the build itself is unchanged
    off = PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0),
                             PF.Settings(n=10, max_exposure=0.5, curve=False, **SMALL))
    assert off.summary["curve"] is None
    assert [x["slots_json"] for x in off.lineups] == [x["slots_json"] for x in pf.lineups]
    assert off.summary["e_payout"] == pytest.approx(pf.summary["e_payout"])


def test_live_build_caps_the_curve_at_the_contests_max_entries(slate, tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "c.db")
    con = F.preset_contest("gpp", 2000, 3.0)
    conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, entry_fee, field_size, "
                 "max_per_user, payout_json, captured_at) VALUES (?,?,?,?,?,?,?,?)",
                 ("9", "synth", "Synthetic 7-max", 3.0, 2000, 7, json.dumps(con.payouts),
                  "2026-10-20T12:00:00Z"))
    conn.commit()
    monkeypatch.setattr(F, "prepare", _fake_prepare(slate))
    pf = PF.build_for_slate(conn, "synth", "9", PF.Settings(n=3, **SMALL))
    assert pf.settings.curve_to == 7
    assert [r["requested"] for r in pf.summary["curve"]["points"]] == [1, 3, 5, 7]
    pid = PF.save(conn, pf)
    stored = PF.load(conn, pid)["data"]["summary"]["curve"]
    assert [r["entries"] for r in stored["points"]] == [1, 3, 5, 7]


# ============================================================================== display

def test_chance_text_bands_until_calibrated():
    assert PF.chance_text(0.27) == "~25–35%"
    assert PF.chance_text(0.27, calibrated=True) == "27%"
    assert PF.chance_text(0.00037) == "~1 in 3,000"                 # 1 significant figure
    assert PF.chance_text(0.00037, calibrated=True) == "~1 in 2,700"
    assert PF.chance_text(0.004) == "~1 in 300"
    assert PF.chance_text(0.0, n_sims=25000) == "under 1 in 25,000"
    assert PF.chance_text(None) == "n/a"


def test_curve_view_has_no_payout_and_bands():
    cv = {"n_build": 3, "complete": True, "points": [
        {"entries": 1, "requested": 1, "cost": 20.0, "p_any_cash": 0.21, "p_any_top1": 0.012,
         "p_any_top01": 0.0011, "p_first": 0.00004, "e_payout": 31.0,
         "e_payout_per_dollar": 1.55, "source": "subset", "n_sims": 25000},
        {"entries": 3, "requested": 3, "cost": 60.0, "p_any_cash": 0.45, "p_any_top1": 0.03,
         "p_any_top01": 0.003, "p_first": 0.0, "e_payout": 80.0, "e_payout_per_dollar": 1.33,
         "source": "build", "n_sims": 25000}]}
    v = dfs_contest.curve_view(cv, calibrated=False, cal_slates=2)
    r0, r1 = v["rows"]
    assert (r0["cash"], r0["top1"], r0["top01"], r0["first"]) == ("~15–25%", "~1–3%",
                                                                 "~1 in 900", "~1 in 30,000")
    assert r1["this_set"] and r1["first"] == "under 1 in 25,000" and r1["cost"] == "$60"
    flat = json.dumps(v).lower()
    assert "1.55" not in flat and "e_payout" not in flat and "roi" not in flat
    assert v["cal_slates"] == 2 and v["cal_needed"] == PF.MIN_CAL_SLATES
    cal = dfs_contest.curve_view(cv, calibrated=True)
    assert cal["rows"][0]["cash"] == "21%" and cal["rows"][0]["first"] == "~1 in 25,000"
    assert dfs_contest.curve_view(None, False) is None


# ============================================================================== page + JSON

from test_web_dfs_contest import CID, SAME, URL, client, wait_done  # noqa: E402,F401
from test_web_dfs import BANNED, text_of  # noqa: E402

PAGE_CURVE = {"version": PF.CURVE_VERSION, "n_build": 2, "complete": True, "points": [
    {"entries": 1, "requested": 1, "cost": 3.0, "p_any_cash": 0.21, "p_any_top1": 0.012,
     "p_any_top01": 0.0011, "p_first": 0.0004, "e_payout": 2.9, "e_payout_per_dollar": 0.97,
     "source": "subset", "n_sims": 25000, "build_ranks": [1]},
    {"entries": 2, "requested": 2, "cost": 6.0, "p_any_cash": 0.33, "p_any_top1": 0.02,
     "p_any_top01": 0.002, "p_first": 0.0007, "e_payout": 5.1, "e_payout_per_dollar": 0.85,
     "source": "build", "n_sims": 25000, "build_ranks": [1, 2]},
    {"entries": 20, "requested": 20, "cost": 60.0, "p_any_cash": 0.71, "p_any_top1": 0.09,
     "p_any_top01": 0.011, "p_first": 0.004, "e_payout": 41.0, "e_payout_per_dollar": 0.68,
     "source": "extended", "n_sims": 25000}]}


def test_page_shows_the_curve_table_and_json(client):
    r = client.post(URL, data={"contest": CID, "n": "2", "max_exp": "60", "gamma": "5"},
                    headers=SAME, follow_redirects=False)
    wait_done(client)
    pfid = r.headers["location"].split("p=")[1]
    page = client.get(r.headers["location"]).text
    assert 'class="table dfc-curve"' not in page               # older builds: no curve, no table
    conn = db.connect(client.app_.state.store.db_path)
    row = conn.execute("SELECT summary_json FROM dfs_portfolios WHERE portfolio_id=?",
                       (pfid,)).fetchone()
    data = json.loads(row[0])
    data["summary"]["curve"] = PAGE_CURVE
    conn.execute("UPDATE dfs_portfolios SET summary_json=? WHERE portfolio_id=?",
                 (json.dumps(data), pfid))
    conn.commit()
    conn.close()
    page = client.get(r.headers["location"]).text
    assert 'class="table dfc-curve"' in page
    sec = page.split('id="curve-title"')[1].split('class="dfs-actions dfc-dl"')[0]
    t = text_of(sec)
    for want in ("Chances by number of entries", "In the money", "Top 0.1%", "1st",
                 "~15–25%", "~25–35%", "~65–80%", "~1 in 900", "~1 in 3,000", "~1 in 300",
                 "this set", "$60", "same simulated nights", "Rough ranges",
                 "not in its CSV", "0 so far"):
        assert want in t, want
    low = t.lower()
    assert "0.97" not in t and "$41" not in t and "roi" not in low and "expected" not in low
    assert "should" not in low and "kelly" not in low and "recommend" not in low
    for pat in BANNED:
        assert not re.search(pat, low), pat
    j = client.get(f"{URL}/{pfid}.json").json()
    assert [p["entries"] for p in j["curve"]["points"]] == [1, 2, 20]
    assert j["summary"]["curve"]["points"][2]["source"] == "extended"
