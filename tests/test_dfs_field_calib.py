"""Field / ownership proxy calibration from real standings (field-proxy-1.1, own-proxy-1.1).

docs/research/real-standings-2026-10-20.md: per-archetype concentration, cash optimizer noise
(duplication), salary use, the large-GPP default mix, double-ups uploaded without a payout table
(inferred paid places for the grader and the contest classifier), inputs as of a capture, and the
partial calibration report. Synthetic pools and DB rows only.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from dfs_results_helpers import (CONTEST, SLATE, player_block, populate, standings_csv,
                                 ten_entries)
from test_dfs_field import CLASSIC, inputs_for, synth_pool

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import grade
from courtside.proj.dfs import standings as ST

FNAME = f"contest-standings-{CONTEST}.csv"


# --------------------------------------------------------------------------- versions / params

def test_versions_bumped_and_archetypes_ordered():
    assert F.MODEL == "own-proxy-1.1" and F.FIELD_VERSION == "field-proxy-1.1"
    assert F.HANDSET_VERSION == "own-proxy-1.1-handset"
    a = F.ARCHETYPES
    # double-ups: more concentrated and duplicated than GPPs; the cash / GPP ratio ~ 1.8
    assert a["cash"]["conc"] / a["large_gpp"]["conc"] == pytest.approx(1.78, abs=0.05)
    assert a["cash"]["noise"] < a["small_gpp"]["noise"] <= a["large_gpp"]["noise"]
    assert a["large_gpp"]["mix"] < a["cash"]["mix"]
    assert all(0.9 <= v["full_spend"] <= 1.0 for v in a.values())
    assert F.MIN_FIT_SLATES == 10                     # bands stay until 10 uploaded slates


def test_spend_target():
    pool = synth_pool(1)
    op = F.ownership_proxy(inputs_for(pool), pool)
    assert op.spend == pytest.approx(0.997 * CLASSIC.salary_cap, rel=0.01)
    op2 = F.ownership_proxy(inputs_for(pool), pool, spend=0.99)
    assert op2.spend == pytest.approx(0.99 * CLASSIC.salary_cap, rel=0.01)


def test_cash_ownership_more_concentrated():
    pool = synth_pool(3, games=5)
    inp = inputs_for(pool)
    top = {a: np.sort(F.ownership_proxy(inp, pool, archetype=a).own)[::-1][:10].sum()
           for a in ("cash", "large_gpp")}
    assert top["cash"] > top["large_gpp"] + 0.3


# --------------------------------------------------------------------------- field

def test_default_mix_is_the_archetypes():
    pool = synth_pool(5, games=5)
    inp = inputs_for(pool)
    gpp = F.preset_contest("gpp", 20_000, 5)
    du = F.preset_contest("double_up", 300, 5)
    assert F._strength(None, "large_gpp") == F.ARCHETYPES["large_gpp"]["mix"] == 0.20
    assert F._strength("medium", "cash") == F.ARCHETYPES["cash"]["mix"] == 0.40
    assert F._strength("sharp", "large_gpp") == 0.70 and F._strength(0.33, "cash") == 0.33
    f = F.build_field(pool, inp, gpp, n=3000, n_solves=20, seed=1)
    assert f.opt_counts.sum() == round(0.20 * 3000)
    g = F.build_field(pool, inp, du, n=3000, n_solves=20, seed=1)
    assert g.opt_counts.sum() == round(0.40 * 3000) and g.mix == 0.40
    h = F.build_field(pool, inp, gpp, n=3000, n_solves=20, seed=1, mix="sharp")
    assert h.opt_counts.sum() == round(0.70 * 3000)


def test_cash_field_duplicates_like_double_ups():
    """Low optimizer noise: the cash field repeats the same few optimal lineups (real
    double-ups put 18-34 % of entries on one lineup)."""
    pool = synth_pool(5, games=5)
    inp = inputs_for(pool)
    du = F.preset_contest("double_up", 5000, 5)
    cash = F.build_field(pool, inp, du, n=5000, n_solves=60, seed=2, mix=0.4)
    gpp = F.build_field(pool, inp, du, n=5000, n_solves=60, seed=2, mix=0.4,
                        archetype="large_gpp")
    assert cash.info["opt_unique"] < gpp.info["opt_unique"]
    assert cash.counts.max() > gpp.counts.max()
    for row in cash.lineups[:200]:
        assert len(set(row.tolist())) == CLASSIC.n_slots


def test_salary_use_matches_real_fields():
    pool = synth_pool(5, games=5)
    inp = inputs_for(pool)
    f = F.build_field(pool, inp, F.preset_contest("double_up", 1000, 5), n=4000, n_solves=20,
                      seed=4)
    su = f.salary_use()
    assert su["share_ge_98"] >= 0.9 and su["min"] >= 0.9 * CLASSIC.salary_cap


# --------------------------------------------------------------------------- double-ups

def test_du_paid_places_matches_dk():
    assert [F.du_paid_places(m) for m in (287, 1149, 459)] == [125, 500, 200]
    assert F.du_paid_places(1) == 1
    assert F.is_double_up("NBA $2 Double Up") and F.is_double_up("NBA $1 50/50")
    assert not F.is_double_up("NBA $100K Four Point Play") and not F.is_double_up(None)


def _upload_only(tmp_path, note):
    c = db.connect(tmp_path / "u.db")
    populate(c, contest=False)
    ST.import_upload(c, standings_csv(ten_entries(), player_block()), FNAME, owner_names=[],
                     slate_id=SLATE)
    c.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, points, "
              "rank, field_size, payout, source, created_at, note) VALUES "
              "(?,?,?,?,?,?,?,?,?,?,?)", (CONTEST, SLATE, "1", 2.0, 300.0, 1, 10, 4.0, "upload",
                                         "2026-10-21T00:00:00Z", note))
    c.commit()
    return c


def test_uploaded_double_up_gets_inferred_table(tmp_path):
    c = _upload_only(tmp_path, "history: NBA $2 Double Up (1/3)")
    con = F.load_contest(c, CONTEST)
    assert con.archetype == "cash" and con.source == "inferred"
    assert con.name == "NBA $2 Double Up" and con.entry_fee == 2.0
    assert con.payouts == [{"min_pos": 1, "max_pos": 4, "payout": 4.0}]    # round(0.435 x 10)
    [r] = grade.contest_results(c, SLATE)
    assert r["cash_line"] == 270 and r["line_source"] == "inferred double-up"   # 4th of 300..210
    assert r["name"] == "NBA $2 Double Up"
    cash, per = grade._cash_fields(275.0, [r], {"mode": "cash", "contest_id": None})
    assert cash["cash_line_source"] == "real" and cash["would_cash"] == 1
    assert cash["est_payout"] == 4.0 and per[0]["line_source"] == "inferred double-up"


def test_uploaded_gpp_without_table_stays_unknown(tmp_path):
    c = _upload_only(tmp_path, "history: NBA $5K Pick and Roll")
    assert F.load_contest(c, CONTEST).source == "captured"
    assert grade.contest_results(c, SLATE)[0]["cash_line"] is None      # nothing invented


# --------------------------------------------------------------------------- inputs / report

def test_capture_at(tmp_path):
    c = db.connect(tmp_path / "c.db")
    populate(c)
    assert F.capture_at(c, SLATE, "2026-10-20T21:00:00Z") == "2026-10-20T20:00:00Z"
    assert F.capture_at(c, SLATE, "2026-10-20T19:00:00Z") is None
    assert F.capture_at(c, SLATE, None) is None


def test_sharpen_exponent_recovers_k():
    rng = np.random.default_rng(0)
    g = {str(i): float(x) for i, x in enumerate(np.clip(rng.gamma(1.0, 0.1, 60), 0, 0.9))}
    tot = 8.0 * sum(g.values()) / sum(g.values())
    vals = F.sharpen(np.array(list(g.values())), 2.0, tot)
    cash = dict(zip(g, vals))
    assert F.sharpen_exponent(g, cash) == pytest.approx(2.0, abs=0.06)
    assert F.sharpen(np.array([0.5, 0.3, 0.2]), 1.0, 1.0).sum() == pytest.approx(1.0)


def test_calibration_report(tmp_path):
    c = db.connect(tmp_path / "r.db")
    populate(c)
    ST.import_upload(c, standings_csv(ten_entries(), player_block()), FNAME, owner_names=[])
    rep = F.calibration_report(c)
    assert rep["n_slates"] == 1 and rep["needed"] == F.MIN_FIT_SLATES
    [r] = rep["contests"]
    assert r["contest_id"] == CONTEST and r["entries"] == 10
    assert r["unique_share"] == pytest.approx(0.7)          # 6 distinct "others" + owner
    assert 0 < r["dupe_entry_share"] <= 1 and r["own_top10"] > 0
    json.dumps(rep, default=float)
