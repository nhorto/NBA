"""Ownership-proxy recalibration on the real standings (re-evaluation H8; switch
COURTSIDE_OWN_CAL, `field.OWN_CAL`; docs/research/reeval-C.md).

OFF (the default) must be bit-identical to own-proxy-1.2 / field-proxy-1.2: the golden file
`fixtures/own_cal_off_golden.json` was written by `python tests/test_own_cal.py golden FIELD
PORTFOLIO` with the pre-change field.py and portfolio.py (HEAD 94b9712, `git show`), and the OFF
path must reproduce it exactly (floats
by repr; Linux only). It covers the proxy at every archetype on a slate with Out / Q players and
injury replacements, a small field, and the contest builder's three strength proxies. The D5
golden (tests/test_field_k.py) pins the rest of the OFF field path. Platform-free checks: the
switch, versions, the 1.2 values written as a spec reproduce OFF exactly, the shipped spec's
effect (GPP flatter, cash unchanged), the builder strengths around the refit w_ours, the K-field
noise table, the recent-DK helper and the form / news terms.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj.dfs import field as F            # noqa: E402
from courtside.proj.dfs import portfolio as PF       # noqa: E402

from test_dfs_field import inputs_for, synth_pool   # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "own_cal_off_golden.json"
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="float rounding differs off Linux")
SPEC_1_2 = {"params": {"b_value": 1.10, "b_proj": 0.70, "w_ours": 0.50, "b_sal": 0.0,
                       "b_form": 0.0, "b_news": 0.0},
            "arch_params": {}, "conc": {"cash": 1.60, "small_gpp": 1.00, "large_gpp": 0.90}}


def _slate():
    """A 4-game slate with our projection != DK FPPG, statuses (Out / Q) and minutes so the
    replacement, bench and news terms all fire."""
    pool = synth_pool(seed=11, games=4, per_team=9)
    rng = np.random.default_rng(5)
    players = list(pool.players)
    for i in (0, 20, 40):
        players[i] = replace(players[i], status="O")
    for i in (5, 30):
        players[i] = replace(players[i], status="Q")
    pool = replace(pool, players=tuple(players))
    proj = {p.dk_player_id: float(max(3.0, (p.fppg or 10) * rng.uniform(0.7, 1.4)))
            for p in pool.players}
    rec = {p.dk_player_id: float(rng.uniform(10, 34)) for p in pool.players}
    pmin = {k: v + float(rng.choice([0.0, 0.0, 4.0, 8.0])) for k, v in rec.items()}
    inp = F.pool_inputs(pool, proj=proj, proj_min=pmin, recent_min=rec,
                        totals={p.dk_player_id: 110.0 + (int(p.dk_player_id) % 7) for p in
                                pool.players})
    return pool, inp


def _r(x):
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items() if k not in ("seconds",)}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_r(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
    return x


def _off_fingerprint(Fm, PFm=None) -> dict:
    pool, inp = _slate()
    out = {}
    for arch in ("cash", "small_gpp", "large_gpp"):
        op = Fm.ownership_proxy(inp, pool, archetype=arch)
        out[arch] = {"own": op.own, "by_slot": op.by_slot, "lam": op.lam, "spend": op.spend,
                     "version": op.version, "pub": op.inputs["pub_proj"].to_numpy()}
    c = Fm.preset_contest("gpp", 3000, 3.0)
    f = Fm.build_field(pool, inp, c, n=2000, n_solves=0, seed=3, threads=1)
    out["field"] = {"lineups": f.lineups, "counts": f.counts, "version": f.version,
                    "realized": f.realized_ownership()}
    if PFm is not None:
        st = PFm.Settings(n_field=1500, field_solves=0, threads=1)
        for name in ("soft", "medium", "sharp"):
            fld, op = PFm.strength_field(pool.players, inp, c, pool.rules, name, st, seed=2)
            out[f"strength_{name}"] = {"own": op.own, "version": op.version,
                                       "field_version": fld.version, "counts": fld.counts}
    return _r(out)


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("courtside.proj.dfs._field_cal_golden", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cal_off():
    saved = F.OWN_CAL
    F.set_own_cal(False)
    yield
    F.OWN_CAL = saved


@pytest.fixture
def cal_on():
    saved = F.OWN_CAL
    F.set_own_cal(True)
    yield
    F.OWN_CAL = saved


# --------------------------------------------------------------------------- the switch

def test_switch_parsing_and_default():
    assert F.OWN_CAL_DEFAULT is False and F.OWN_CAL_ENV == "COURTSIDE_OWN_CAL"
    assert F.own_cal_mode(None) is False and F.own_cal_mode("") is False
    for v in ("1", "on", "true", "yes", True):
        assert F.own_cal_mode(v) is True
    for v in ("0", "off", "false", "no", False):
        assert F.own_cal_mode(v, True) is False
    assert F.own_cal_mode("banana") is False


# --------------------------------------------------------------------------- OFF

@LINUX
def test_off_bit_identical_to_pre_change_golden(cal_off):
    assert GOLDEN.exists(), "golden missing: python tests/test_own_cal.py golden <old field.py>"
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(_off_fingerprint(F, PF)))
    assert got == want


def test_off_paths_carry_no_cal(cal_off):
    pool, inp = _slate()
    op = F.ownership_proxy(inp, pool, archetype="large_gpp")
    assert op.version.startswith(F.MODEL + "-") and F.MODEL_CAL not in op.version
    assert "b_sal" not in op.params and F.field_version(op) == F.FIELD_VERSION
    assert F.own_cal_w("large_gpp") is None
    assert F.own_sd_for(op)[0] == F.OWN_SD["large_gpp"]


def test_spec_of_the_1_2_values_reproduces_off_exactly(cal_off):
    """cal_utility with the 1.2 weights and zero new terms is the 1.2 utility, bit for bit."""
    pool, inp = _slate()
    for arch in ("cash", "small_gpp", "large_gpp"):
        a = F.ownership_proxy(inp, pool, archetype=arch, cal=False)
        b = F.ownership_proxy(inp, pool, archetype=arch, cal=SPEC_1_2)
        assert np.array_equal(a.own, b.own) and np.array_equal(a.by_slot, b.by_slot)
        assert a.lam == b.lam
        assert b.version == a.version.replace(F.MODEL, F.MODEL_CAL)


# --------------------------------------------------------------------------- ON

def test_on_versions_and_field_tags(cal_on):
    pool, inp = _slate()
    op = F.ownership_proxy(inp, pool, archetype="large_gpp")
    assert op.version.startswith(F.MODEL_CAL) and op.version.endswith("/large_gpp")
    assert F.field_version(op) == F.FIELD_VERSION_CAL == "field-proxy-1.3"
    assert F.field_version(op, k=True) == "field-proxy-1.3+k"
    c = F.preset_contest("gpp", 3000, 3.0)
    f = F.build_field(pool, inp, c, n=1500, n_solves=0, seed=1, threads=1)
    assert f.version == "field-proxy-1.3"
    fs = F.build_fields(pool, inp, c, k=2, ownership=op, n=1500, n_solves=0, seed=1, threads=1)
    assert fs.version == "field-proxy-1.3+k"
    assert all(x.version == "field-proxy-1.3+k" for x in fs.fields)
    assert F.own_sd_for(op)[0] == F.OWN_SD_CAL["large_gpp"]


def test_on_gpp_is_flatter_and_cash_is_unchanged(cal_on):
    pool, inp = _slate()
    for arch in ("large_gpp", "small_gpp"):
        off = F.ownership_proxy(inp, pool, archetype=arch, cal=False)
        on = F.ownership_proxy(inp, pool, archetype=arch)
        assert abs(on.own.sum() - off.own.sum()) < 1e-6      # still 800 %
        assert (on.own ** 2).sum() < (off.own ** 2).sum()    # conc 0.65 < 0.90
        assert on.params["w_ours"] == pytest.approx(0.70)
    off = F.ownership_proxy(inp, pool, archetype="cash", cal=False)
    on = F.ownership_proxy(inp, pool, archetype="cash")
    assert np.array_equal(off.own, on.own) and on.params["w_ours"] == 0.5
    assert on.version != off.version


def test_builder_strengths_shift_around_the_refit_w(cal_on):
    pool, inp = _slate()
    c = F.preset_contest("gpp", 3000, 3.0)
    st = PF.Settings(n_field=1500, field_solves=0, threads=1)
    ws = {}
    for name in ("soft", "medium", "sharp"):
        fld, op = PF.strength_field(pool.players, inp, c, pool.rules, name, st, seed=2)
        ws[name] = op.params["w_ours"]
        assert op.version.startswith(F.MODEL_CAL) and fld.version == F.FIELD_VERSION_CAL
    w0 = F.own_cal_w("large_gpp")
    assert ws["medium"] == pytest.approx(w0)
    assert ws["soft"] == pytest.approx(w0 + PF.STRENGTH_W["soft"])
    assert ws["sharp"] == pytest.approx(min(1.0, w0 + PF.STRENGTH_W["sharp"]))
    # the medium strength equals the plain ON proxy
    plain = F.ownership_proxy(inp, pool, archetype="large_gpp")
    _, med = PF.strength_field(pool.players, inp, c, pool.rules, "medium", st, seed=2)
    assert np.allclose(plain.own, med.own, atol=1e-12)


def test_form_and_news_terms_move_the_right_players():
    pool, inp = _slate()
    base = json.loads(json.dumps(SPEC_1_2))
    i = int(np.argmax(inp["fppg"].fillna(0).to_numpy() < 30))     # a mid player
    inp2 = inp.copy()
    inp2["recent_dk"] = np.nan
    inp2.loc[i, "recent_dk"] = 60.0                                 # hot recent form
    sp = json.loads(json.dumps(base))
    sp["params"]["b_form"] = 1.0
    a = F.ownership_proxy(inp2, pool, archetype="large_gpp", cal=base)
    b = F.ownership_proxy(inp2, pool, archetype="large_gpp", cal=sp)
    assert b.own[i] > a.own[i]
    # news: teammates of an OUT rotation player who gain minutes
    sp = json.loads(json.dumps(base))
    sp["params"]["b_news"] = 1.0
    _, _, feats = F.cal_utility(inp, {}, "large_gpp", F._cal_spec(sp))
    out = (inp["status"] == "O") & (inp["recent_min"] >= F.OWN_CAL_NEWS_MIN)
    teams = set(inp.loc[out, "team"])
    gain = (inp["proj_min"] - inp["recent_min"]).to_numpy()
    hit = inp["team"].isin(teams).to_numpy() & (gain > 0)
    assert (feats["news"][hit] > 0).all() and (feats["news"][~inp["team"].isin(teams)] == 0).all()


def test_recent_dk_reads_the_last_played_games():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE player_games (player_id INT, game_date TEXT, pts, fg3m, reb, ast, "
                 "stl, blk, tov, min)")
    rows = [(1, "2026-10-10", 10, 0, 0, 0, 0, 0, 0, 30), (1, "2026-10-12", 20, 0, 0, 0, 0, 0, 0, 30),
            (1, "2026-10-14", 0, 0, 0, 0, 0, 0, 0, 0),          # DNP: skipped
            (1, "2026-10-16", 30, 2, 10, 0, 0, 0, 2, 32),       # 30+1+12.5-1+1.5 (DD) = 44
            (1, "2026-10-20", 99, 0, 0, 0, 0, 0, 0, 30)]        # on / after the day: excluded
    conn.executemany("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    got = F.recent_dk(conn, [1, 2], "2026-10-20", n_games=3)
    assert got == {1: pytest.approx((10 + 20 + 44) / 3)}


def test_calibration_report_classifies_upload_only_contests_by_entries(tmp_path, monkeypatch):
    """An uploaded contest with no stored field size or payout table read as a 0-entry small
    GPP in `calibration_report` (4 of the 10-23 GPPs); it is now classified by its standings'
    entry count."""
    from courtside import db
    from dfs_results_helpers import (CONTEST, SLATE, player_block, populate, standings_csv,
                                     ten_entries)
    from courtside.proj.dfs import standings as ST
    fix = Path(__file__).parent / "fixtures" / "dfs"
    c = db.connect(tmp_path / "r.db")
    populate(c)
    with c:
        c.execute("UPDATE dfs_slates SET rules_json=? WHERE slate_id=?",
                  ((fix / "rules_70.json").read_text(), SLATE))
        c.execute("UPDATE dfs_salaries SET roster_slot=NULL WHERE slate_id=?", (SLATE,))
    ST.import_upload(c, standings_csv(ten_entries(), player_block()),
                     f"contest-standings-{CONTEST}.csv", owner_names=[])
    with c:
        c.execute("UPDATE dfs_contests SET field_size=NULL, payout_json=NULL WHERE contest_id=?",
                  (CONTEST,))
    monkeypatch.setattr(F, "SMALL_GPP_MAX", 5)          # 10 entries = a "large" GPP here
    [r] = F.calibration_report(c)["contests"]
    assert r["entries"] == 10 and r["archetype"] == "large_gpp"


if __name__ == "__main__":
    # python tests/test_own_cal.py golden PRE_CHANGE_FIELD_PY PRE_CHANGE_PORTFOLIO_PY
    if len(sys.argv) == 4 and sys.argv[1] == "golden":
        old = _load_module(sys.argv[2])
        spec = importlib.util.spec_from_file_location("courtside.proj.dfs._pf_cal_golden",
                                                      sys.argv[3])
        old_pf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old_pf
        spec.loader.exec_module(old_pf)
        old_pf.F = old                   # the old builder on the old field module
        GOLDEN.write_text(json.dumps(_off_fingerprint(old, old_pf)))
        print(f"wrote {GOLDEN}")
