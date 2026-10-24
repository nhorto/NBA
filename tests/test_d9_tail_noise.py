"""Tail-estimation noise (IMPROVEMENTS.md D9; `proj/dfs/portfolio.py`).

* The entries curve's points carry simulation SEs (binomial for the chances, per-sim sd for
  E[payout]); display only: every number of a point is unchanged (the builder goldens in
  test_portfolio_split.py / test_xfactor_v2.py drop only the new "se" key).
* The selection objective switch (COURTSIDE_PORTFOLIO_OBJ, default OFF): the greedy picks on a
  smoothed ladder ('flat' / 'blend', same total money), the report keeps the real ladder, the
  version string is tagged. OFF passes the real ladder object itself.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj.dfs import field as F            # noqa: E402
from courtside.proj.dfs import portfolio as PF       # noqa: E402
from courtside.web import dfs_contest                # noqa: E402

from test_dfs_entries_curve import _curve, _greedy, correlated_pool, gpp   # noqa: E402
from test_dfs_field import inputs_for, sims_for, synth_pool               # noqa: E402

SMALL = dict(n_scenario=20, n_ownpen=10, n_noise=6, n_topmean=6, n_sims=None, n_field=2000,
             field_solves=20, threads=1, solve_limit_s=2.0, curve_to=8, curve_select_sims=600)


# ============================================================================== curve SEs

def test_curve_points_carry_simulation_ses():
    con = gpp()
    P = correlated_pool()
    cv = _curve(P, con, _greedy(P, con, 5), top=10)
    lad = PF.Ladder(con)
    for r in cv["points"]:
        se, n = r["se"], r["n_sims"]
        for k in ("p_any_cash", "p_any_top1", "p_any_top01", "p_first"):
            assert se[k] == pytest.approx(math.sqrt(r[k] * (1 - r[k]) / n))
        assert se["e_payout"] > 0
        assert se["e_payout_per_dollar"] == pytest.approx(se["e_payout"] / r["cost"])
    # E[payout] SE = sd of the per-sim total / sqrt(S) on the point's own set
    st = PF.curve_stats(lad, P, _greedy(P, con, 10), con, m_o=con.field_size - 5)
    f = max(1, con.field_size - 10) / (con.field_size - 5)
    ch = _greedy(P, con, 10)
    pay, _ = PF.joint_payouts(lad, P.score[:, ch], PF._rescaled(P.above[:, ch], f),
                              PF._rescaled(P.tied[:, ch], f))
    tot = pay.sum(1)
    assert st["se"]["e_payout"] == pytest.approx(tot.std() / math.sqrt(len(tot)))
    assert st["e_payout"] == pytest.approx(tot.mean())


def test_binom_se_and_se_text():
    assert PF.binom_se(0.25, 10_000) == pytest.approx(math.sqrt(0.25 * 0.75 / 10_000))
    assert PF.binom_se(None, 100) is None and PF.binom_se(0.3, 0) is None
    assert PF.se_text(0.25, 0.0043) == "± 0.4 pt"
    assert PF.se_text(0.25, 0.0001) == "± <0.1 pt"
    assert PF.se_text(0.0004, 0.00012) == "± 30%"            # rare: relative
    assert PF.se_text(0.0, 0.0) == "" and PF.se_text(0.2, None) == ""


def test_curve_view_shows_ses_and_falls_back_to_binomial():
    pt = {"entries": 1, "requested": 1, "cost": 20.0, "p_any_cash": 0.21, "p_any_top1": 0.012,
          "p_any_top01": 0.0011, "p_first": 0.00004, "e_payout": 31.0,
          "e_payout_per_dollar": 1.55, "source": "build", "n_sims": 25000}
    old = dfs_contest.curve_view({"n_build": 1, "points": [pt]}, calibrated=False)
    r = old["rows"][0]
    assert r["cash"] == "~15–25%" and r["cash_se"] == "± 0.3 pt"   # sqrt(.21*.79/25k)
    assert r["first_se"] == "± 100%"                                  # one sim in 25,000
    assert old["n_sims"] == 25000
    new = dict(pt, se={"p_any_cash": 0.01, "p_any_top1": None, "p_any_top01": 0.0001,
                       "p_first": 0.00002, "e_payout": 3.0})
    r = dfs_contest.curve_view({"n_build": 1, "points": [new]}, calibrated=True)["rows"][0]
    assert r["cash"] == "21%" and r["cash_se"] == "± 1.0 pt" and r["top1_se"] == ""
    assert r["first_se"] == "± 50%"
    flat = json.dumps(r).lower()
    assert "e_payout" not in flat and "3.0" not in flat                  # still no payout


# ============================================================================== objective

def test_objective_mode_parsing_and_default_blend():
    assert PF.OBJ_DEFAULT == "blend"                       # shipped 2026-10-24
    assert PF.PORTFOLIO_OBJ == "off"                       # tests/conftest.py pins it off
    for v, want in (("flat", "flat"), ("blend", "blend"), ("1", "flat"), ("on", "flat"),
                    ("0", "off"), ("off", "off"), ("", "off"), ("junk", "blend"),
                    (None, "blend")):
        assert PF.objective_mode(v) == want, v
    saved = PF.PORTFOLIO_OBJ
    try:
        PF.set_objective("blend")
        assert PF.PORTFOLIO_OBJ == "blend"
    finally:
        PF.PORTFOLIO_OBJ = saved


def test_selection_ladder_smooths_the_top_and_keeps_the_money():
    con = F.Contest("c", "g", 5.0, 5000,
                    [{"min_pos": 1, "max_pos": 1, "payout": 2000.0},
                     {"min_pos": 2, "max_pos": 3, "payout": 500.0},
                     {"min_pos": 4, "max_pos": 10, "payout": 100.0},
                     {"min_pos": 11, "max_pos": 50, "payout": 30.0},
                     {"min_pos": 51, "max_pos": 1000, "payout": 8.0}], "large_gpp", "preset")
    lad = PF.Ladder(con)
    assert PF.selection_ladder(lad, "off") is lad
    flat, blend = PF.selection_ladder(lad, "flat"), PF.selection_ladder(lad, "blend")
    for x in (flat, blend):
        assert x.cum[-1] == pytest.approx(lad.cum[-1])      # same total money
        assert (x.paid, x.top1, x.M, x.top) == (lad.paid, lad.top1, lad.M, lad.top)
        assert lad.cum is not x.cum
    pay, fp, bp = lad.pay(), flat.pay(), blend.pay()
    k = 5                                                   # ceil(0.001 x 5,000)
    assert np.allclose(fp[:k], pay[:k].mean()) and np.array_equal(fp[k:], pay[k:])
    t1 = lad.top1                                           # 50 ranks
    assert np.allclose(bp[:t1], 0.5 * pay[:t1] + 0.5 * pay[:t1].mean())
    assert np.array_equal(bp[t1:], pay[t1:])
    # g() is the tie-split mean of the smoothed ranks
    assert flat.g(np.array([1]), np.array([0]))[0] == pytest.approx(pay[:k].mean())
    with pytest.raises(PF.PortfolioError):
        PF.selection_ladder(lad, "nope")


@pytest.fixture(scope="module")
def slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1500, seed=2)


def _build(slate, **kw):
    pool, inp, sims = slate
    return PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 1500, 3.0),
                              PF.Settings(n=5, gamma=5, max_exposure=0.6, **SMALL), **kw)


def test_objective_on_picks_on_the_smoothed_ladder_and_reports_on_the_real_one(slate,
                                                                               monkeypatch):
    seen = []
    real_greedy = PF.greedy

    def spy(lad, *a, **k):
        seen.append(getattr(lad, "mode", "real"))
        return real_greedy(lad, *a, **k)
    monkeypatch.setattr(PF, "greedy", spy)
    off = _build(slate, split="off")
    assert set(seen) == {"real"} and "+obj" not in off.info["version"]
    seen.clear()
    on = _build(slate, split="off", objective="flat")
    assert set(seen) == {"flat"}                     # the build and the curve's picks
    assert on.info["version"] == PF.VERSION + "+obj-flat"
    seen.clear()
    bl = _build(slate, split="gs", objective="blend")
    assert set(seen) == {"blend"} and bl.info["version"] == PF.VERSION + "+split-gs+obj-blend"
    # the report is the real ladder: the summary is the joint payout of the chosen set
    for pf in (on, bl):
        assert pf.summary["e_payout"] == pytest.approx(sum(x["e_payout"] for x in pf.lineups))
        pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in pf.slots()]
        assert len({frozenset(p for _, p in lu) for lu in pairs}) == len(pairs)
    # off via the argument == the process default (same picks and numbers)
    off2 = _build(slate, split="off", objective="off")
    assert [x["slots_json"] for x in off2.lineups] == [x["slots_json"] for x in off.lineups]
    assert off2.summary["e_payout"] == off.summary["e_payout"]
