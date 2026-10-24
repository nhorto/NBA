"""X factor v2 = top-1 % leverage against the modelled GPP field (IMPROVEMENTS.md D3;
`xfactor.py` xf-v2, switch COURTSIDE_XF_V2; docs/research/xfactor-v2.md).

OFF (the default) must leave xf-v1 and the contest builder bit-identical. The golden file
`fixtures/xfactor_v2_off_golden.json` was written from the PRE-CHANGE modules loaded in place:

    python tests/test_xfactor_v2.py golden PRE_CHANGE_PORTFOLIO_PY      # builder part
    XF2_GOLDEN_FROM=PRE_CHANGE_XFACTOR_PY pytest tests/test_xfactor_v2.py -k golden_table
                                                                        # run_table part

The exact checks are Linux-only (goldens recorded on geekom1 / OpenBLAS).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj.dfs import field as F            # noqa: E402
from courtside.proj.dfs import portfolio as PF       # noqa: E402
from courtside.proj.dfs import xfactor as XF         # noqa: E402

from test_dfs_field import inputs_for, sims_for, synth_pool   # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "xfactor_v2_off_golden.json"
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="goldens recorded on Linux")
SMALL = dict(n_scenario=20, n_ownpen=12, n_noise=6, n_topmean=6, n_sims=None, n_field=2000,
             field_solves=20, threads=1, solve_limit_s=2.0, curve_to=8, curve_select_sims=600)
BANNED = ("edge", "lean", "over", "play")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(f"courtside.proj.dfs.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _strip(x):
    if isinstance(x, dict):
        return {str(k): _strip(v) for k, v in x.items() if k not in ("seconds", "timing")}
    if isinstance(x, (list, tuple)):
        return [_strip(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


@pytest.fixture
def v2_off():
    saved = XF.XF_V2
    XF.set_xf_v2(False)
    yield
    XF.XF_V2 = saved


@pytest.fixture
def v2_on():
    saved = XF.XF_V2
    XF.set_xf_v2(True)
    XF._CACHE.clear()
    yield
    XF.XF_V2 = saved
    XF._CACHE.clear()


# ============================================================================== switch

def test_switch_default_on_and_parsing(monkeypatch):
    # shipped 2026-10-24 (replaces xf-v1); tests/conftest.py pins it off for the suite
    assert XF.XF_V2_DEFAULT is True and XF.VERSION == "xf-v1" and XF.VERSION_V2 == "xf-v2"
    saved = XF.XF_V2
    try:
        for v, want in (("1", True), ("on", True), ("v2", True), ("0", False), ("off", False),
                        ("", False), (True, True), (False, False)):
            XF.set_xf_v2(v)
            assert XF.xf_v2() is want, v
    finally:
        XF.XF_V2 = saved


# ============================================================================== leverage math

def _toy_field(lineups, counts, n_players, slots=("UTIL", "UTIL")):
    return SimpleNamespace(pids=[f"p{i}" for i in range(n_players)], slots=slots,
                           lineups=np.array(lineups), counts=np.array(counts))


def test_top1_counts_share_and_ownership_by_hand():
    # 100 entries: (0,1) x50, (2,3) x30, (0,2) x20; top 1 % = the single best entry
    f = _toy_field([[0, 1], [2, 3], [0, 2]], [50, 30, 20], 4)
    fp = np.array([[10, 10, 0, 0],       # (0,1) 20 best -> players 0, 1
                   [0, 0, 9, 9],         # (2,3) 18 best -> players 2, 3
                   [0, 0, 0, 0]], np.float32)   # all tie at 0: every entry is "in" the line
    share, own, r = XF.top1_counts(f, fp, top=0.01)
    assert r == 3
    np.testing.assert_allclose(own, [0.7, 0.5, 0.5, 0.3])
    # row 3: everybody tied -> shares = ownership
    np.testing.assert_allclose(share, [1 + 0 + 0.7, 1 + 0 + 0.5, 0 + 1 + 0.5, 0 + 1 + 0.3])


def test_top1_counts_uses_the_count_weighted_line():
    # top 1 % of 200 entries = 2 entries: the best lineup has 1 entry, so the line is the
    # second best lineup's score and both are in
    f = _toy_field([[0, 1], [2, 3], [0, 3]], [1, 150, 49], 4)
    fp = np.array([[10, 10, 5, 5]], np.float32)
    share, own, _ = XF.top1_counts(f, fp, top=0.01)
    # in the top: (0,1) x1 [20], (0,3) x49 [15] -> 50 entries; (2,3) [10] is below the line
    np.testing.assert_allclose(share, [50 / 50, 1 / 50, 0.0, 49 / 50])


def test_leverage_shrinks_toward_one_and_routes_rows_to_fields():
    f1 = _toy_field([[0, 1], [2, 3]], [90, 10], 4)
    f2 = _toy_field([[0, 2], [1, 3]], [50, 50], 4)
    fp = np.array([[0, 0, 10, 10], [10, 0, 10, 0]] * 2, np.float32)   # rows 0, 2 -> f1; 1, 3 -> f2
    lv = XF.leverage([f1, f2], fp, top=0.01, prior=0.0)
    # f1 rows: (2,3) wins; f2 rows: (0,2) wins
    np.testing.assert_allclose(lv["share_k"][0], [0, 0, 1, 1])
    np.testing.assert_allclose(lv["share_k"][1], [1, 0, 1, 0])
    np.testing.assert_allclose(lv["share"], [0.5, 0, 1, 0.5])
    np.testing.assert_allclose(lv["own"], [(0.9 + 0.5) / 2, (0.9 + 0.5) / 2, (0.1 + 0.5) / 2,
                                           (0.1 + 0.5) / 2])
    np.testing.assert_allclose(lv["lev"], lv["lev_raw"])
    a = 0.0025
    lv2 = XF.leverage([f1, f2], fp, top=0.01, prior=a)
    np.testing.assert_allclose(lv2["lev"], (lv["share"] + a) / (lv["own"] + a))
    # shrinkage: a player at share = own stays at 1; a rare one moves most toward 1
    rare = (0.001 * 5 + a) / (0.001 + a)
    common = (0.05 * 5 + a) / (0.05 + a)
    assert 1 < rare < common < 5 and rare == pytest.approx(1 + 4 * 0.001 / 0.0035)


def test_leverage_on_a_real_field_set_sums_to_the_roster():
    pool = synth_pool(seed=3, games=3, per_team=8)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    fs = F.build_fields(pool, inp, F.preset_contest("gpp", 3000), k=2, n=3000, n_solves=10,
                        threads=1)
    fp, _ = F.align_sims(sims_for(pool, S=400, seed=1), fs.pids)
    lv = XF.leverage(fs, fp)
    assert lv["share"].sum() == pytest.approx(8.0)          # 8 roster slots per entry
    assert lv["own"].sum() == pytest.approx(8.0)
    assert np.all(lv["lev"] > 0) and lv["share_k"].shape == (2, len(fs.pids))


def test_noisy_fields_reuse_a_field_set_or_the_solves_of_one_field():
    pool = synth_pool(seed=4, games=3, per_team=8)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    c = F.preset_contest("gpp", 3000)
    fs = F.build_fields(pool, inp, c, k=4, n=2000, n_solves=8, threads=1)
    assert XF.noisy_fields(pool.players, inp, c, pool.rules, base=fs) is fs
    one = F.build_field(pool, inp, c, n=2000, n_solves=8, threads=1)
    ks = XF.noisy_fields(pool.players, inp, c, pool.rules, base=one, k=4)
    assert ks.k == 4 and all(f.opt_solves == one.opt_solves for f in ks.fields)
    assert all(f.n == one.n for f in ks.fields)
    assert ks.version.endswith(F.FIELD_K_TAG)


# ============================================================================== tag rule

def test_is_xfactor_v2_rule():
    ok = dict(lev=2.0, share=0.05, own=0.03, p_play=0.95, pub_rank=20)
    assert XF.is_xfactor_v2(**ok)
    for k, bad in (("lev", 1.4), ("share", 0.01), ("own", 0.12), ("p_play", 0.8),
                   ("pub_rank", 10), ("lev", None), ("p_play", None), ("own", float("nan"))):
        assert not XF.is_xfactor_v2(**{**ok, k: bad}), (k, bad)
    assert XF.is_xfactor_v2(**{**ok, "pub_rank": None})        # not ranked: not excluded


def test_cap_v2_keeps_the_highest_leverage():
    t = {str(i): {"lev": 1.5 + 0.1 * i, "own": 1.0, "x": True} for i in range(8)}
    t["9"] = {"lev": 9.0, "own": 1.0, "x": False}
    XF.apply_cap_v2(t)
    assert {d for d, r in t.items() if r["x"]} == {"3", "4", "5", "6", "7"}


def test_public_value_ranks_use_the_public_number():
    import pandas as pd
    op = SimpleNamespace(pids=["a", "b", "c", "d"], own=np.array([0.1, 0.2, 0.0, 0.3]),
                         inputs=pd.DataFrame({"pub_proj": [20.0, 30.0, 50.0, 10.0],
                                              "salary": [4000, 5000, 5000, 1000]}))
    assert XF.public_value_ranks(op) == {"d": 1, "b": 2, "a": 3}     # c: not drafted


def test_tilt_scores():
    base = np.array([10.0, 20.0, 30.0])
    t = XF.tilt_scores(["a", "b", "c"], base, {"a": math.e, "b": 1e-9}, 2.0)
    assert t["a"] == pytest.approx(12.0)
    assert t["b"] == pytest.approx(20.0 - 2.0 * XF.LEV_LOG_CLIP)     # clipped
    assert t["c"] == pytest.approx(30.0)                              # no leverage: lev 1


def test_v2_title_and_labels_are_information_only():
    row = {"share": 0.061, "lev_own": 0.025, "lev": 2.4,
           "split_text": "With A out: big-night chance 1 in 3 vs 1 in 5 with him available.",
           "late_news": "x"}
    s = XF.v2_title(row)
    assert "6.1 %" in s and "2.5 %" in s and "2.4 times" in s and "not advice" in s
    for text in (s, XF.LABEL_V2):
        low = text.lower()
        for w in BANNED:
            assert w not in low, (w, text)


# ============================================================================== run_table

from test_dfs_xfactor import HOT, xf_db    # noqa: E402,F401  (fixture)
from test_web_dfs import DAY, NOW, SLATE, players, text_of  # noqa: E402


def _table_fp(t: dict) -> dict:
    return _strip({d: {k: v for k, v in r.items()} for d, r in sorted(t.items())})


@LINUX_ONLY
def test_golden_table_off_is_bit_identical_to_xf_v1(xf_db, v2_off):
    _, conn = xf_db
    XF._CACHE.clear()
    src = os.environ.get("XF2_GOLDEN_FROM")
    if src:                                     # write the run_table part from the old module
        old = _load_module(src, "_xfactor_golden")
        old.STRETCH, old.VALUE_TOP_EXCLUDE = {}, 0
        gold = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        gold["run_table"] = json.loads(json.dumps(_table_fp(old.run_table(conn, "run1"))))
        GOLDEN.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")
    gold = json.loads(GOLDEN.read_text())["run_table"]
    t = XF.run_table(conn, "run1")
    assert json.loads(json.dumps(_table_fp(t))) == gold
    assert not any(k in r for r in t.values() for k in ("lev", "x_v1", "title"))


def _fake_leverage(ps):
    hot, other = ps[HOT]["id"], ps[9]["id"]

    def run_leverage(conn, run_id, **kw):
        pl = {p["id"]: {"share": 0.01, "lev_own": 0.02, "lev": 0.5, "lev_raw": 0.5,
                        "proxy_own": 0.02, "pub_value_rank": 50} for p in ps}
        pl[other] = {"share": 0.08, "lev_own": 0.03, "lev": 2.5, "lev_raw": 2.7,
                     "proxy_own": 0.02, "pub_value_rank": 40}
        return {"version": XF.VERSION_V2, "players": pl, "k": 4, "contest": {},
                "own_unc": {}, "n_sims": 100, "seconds": 0.0}
    return run_leverage, hot, other


def test_run_table_on_uses_the_v2_rule_and_keeps_v1(xf_db, v2_on, monkeypatch, tmp_path):
    _, conn = xf_db
    ps = players()
    fake, hot, other = _fake_leverage(ps)
    calls = []
    monkeypatch.setattr(XF, "run_leverage", lambda *a, **k: calls.append(1) or fake(*a, **k))
    monkeypatch.setattr(XF, "_cache_path", lambda run_id: tmp_path / f"{run_id}.json")
    for p in ps:                                   # everybody likely to suit up
        conn.execute("UPDATE player_projections SET p_play=0.95 WHERE run_id='run1' AND "
                     "player_id=?", (500 + p["i"],))
    conn.commit()
    t = XF.run_table(conn, "run1")
    assert t[hot]["x_v1"] and not t[hot]["x"]               # v1 tag, low leverage
    assert t[other]["x"] and not t[other]["x_v1"]
    assert t[other]["lev"] == 2.5 and t[other]["version"] == XF.VERSION_V2
    assert "8.0 %" in t[other]["title"] and "not advice" in t[other]["title"]
    assert XF.xfactor_ids(t) == {other}
    # disk cache: a fresh process (memory cache cleared) does not rebuild the fields
    XF._CACHE.clear()
    XF.run_table(conn, "run1")
    assert len(calls) == 1


def test_run_table_on_falls_back_to_v1_when_v2_fails(xf_db, v2_on, monkeypatch, tmp_path):
    _, conn = xf_db

    def boom(*a, **k):
        raise RuntimeError("no field")
    monkeypatch.setattr(XF, "run_leverage", boom)
    monkeypatch.setattr(XF, "_cache_path", lambda run_id: tmp_path / f"{run_id}.json")
    t = XF.run_table(conn, "run1")
    assert XF.xfactor_ids(t) == {players()[HOT]["id"]}


def test_pool_page_on_shows_the_v2_text(xf_db, v2_on, monkeypatch, tmp_path):
    d, conn = xf_db
    ps = players()
    fake, _, other = _fake_leverage(ps)
    monkeypatch.setattr(XF, "run_leverage", fake)
    monkeypatch.setattr(XF, "_cache_path", lambda run_id: tmp_path / f"{run_id}.json")
    conn.execute("UPDATE player_projections SET p_play=0.95 WHERE run_id='run1'")
    conn.commit()
    from courtside.web.app import create_app
    from fastapi.testclient import TestClient
    app = create_app(d)
    app.state.dfs_now = lambda: NOW
    html = TestClient(app).get(f"/dfs/{DAY}/{SLATE}?all=1").text
    t = text_of(html)
    assert "best 1 % of entries" in t and "1 on this slate" in t and "not advice" in t
    assert "8.0 % of the best 1 %" in html
    foot = t[t.index("X factor marks"):]
    foot = foot[:foot.index("not advice")]
    for w in BANNED:
        assert w not in foot.lower(), w


# ============================================================================== builder

def _slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1200, seed=2)


def _cases():
    return {"off": dict(split="off"), "gs": dict(split="gs"),
            "gs_k": dict(split="gs", field_k=2)}


def _build(mod, pool, inp, sims, case: dict):
    saved = F.FIELD_K
    F.set_field_k(case.get("field_k", 0))
    try:
        s = mod.Settings(n=5, gamma=5, max_exposure=0.6, **SMALL)
        return mod.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 1500, 3.0), s,
                                   split=case["split"], blended=False)
    finally:
        F.FIELD_K = saved


def _fingerprint(pf) -> dict:
    return _strip({"lineups": pf.lineups, "summary": pf.summary, "exposure": pf.exposure,
                   "info": pf.info})


@pytest.fixture(scope="module")
def slate():
    return _slate()


@LINUX_ONLY
def test_golden_builder_off_is_bit_identical(slate, v2_off):
    gold = json.loads(GOLDEN.read_text())["builder"]
    pool, inp, sims = slate
    for name, case in _cases().items():
        assert json.loads(json.dumps(_fingerprint(_build(PF, pool, inp, sims, case)))) \
            == gold[name], name


def test_builder_on_adds_leverage_tilted_solves(slate, monkeypatch):
    pool, inp, sims = slate
    saved = XF.XF_V2
    try:
        XF.set_xf_v2(False)
        off = _build(PF, pool, inp, sims, {"split": "gs"})
        XF.set_xf_v2(True)
        on = _build(PF, pool, inp, sims, {"split": "gs"})
    finally:
        XF.XF_V2 = saved
    assert "+xf2" in on.info["version"] and "+xf2" not in off.info["version"]
    assert "xfactor_v2" not in off.info and "leverage" not in off.info["timing"]
    x = on.info["xfactor_v2"]
    assert x["version"] == XF.VERSION_V2 and x["k"] == XF.LEV_K and x["rows"] > 0
    src_on, src_off = on.info["candidates"]["by_source"], off.info["candidates"]["by_source"]
    assert on.info["candidates"]["solves"] - off.info["candidates"]["solves"] \
        == round(XF.LEV_N_SOLVES * PF.Settings(n=5).scale)
    # the tilted jobs come after every other job and draw from their own stream: the other
    # sources' unique candidates are exactly OFF's
    assert {k: v for k, v in src_on.items() if k != "levtilt"} == src_off
    assert x["selected_levtilt"] == on.info["split"]["by_source_selected"].get("levtilt", 0)
    assert len(on.lineups) == 5


def test_builder_on_tilted_solves_find_new_lineups(slate, monkeypatch):
    pool, inp, sims = slate
    ids = sorted(p.dk_player_id for p in pool.players)
    lev = {p: (4.0 if i % 3 == 0 else 0.3) for i, p in enumerate(ids)}
    monkeypatch.setattr(XF, "builder_leverage",
                        lambda *a, **k: (lev, {"version": XF.VERSION_V2, "k": 4, "rows": 1}))
    saved = XF.XF_V2
    try:
        XF.set_xf_v2(True)
        on = _build(PF, pool, inp, sims, {"split": "gs"})
    finally:
        XF.XF_V2 = saved
    assert on.info["candidates"]["by_source"].get("levtilt", 0) > 0


def test_builder_on_reuses_a_k4_field_set(slate):
    pool, inp, sims = slate
    saved = XF.XF_V2
    try:
        XF.set_xf_v2(True)
        on = _build(PF, pool, inp, sims, {"split": "gs", "field_k": 4})
    finally:
        XF.XF_V2 = saved
    assert on.info["xfactor_v2"]["reused_fields"] is True


def test_builder_on_survives_a_leverage_failure(slate, monkeypatch):
    pool, inp, sims = slate

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(XF, "builder_leverage", boom)
    saved = XF.XF_V2
    try:
        XF.set_xf_v2(True)
        on = _build(PF, pool, inp, sims, {"split": "gs"})
    finally:
        XF.XF_V2 = saved
    assert "error" in on.info["xfactor_v2"] and "levtilt" not in on.info["candidates"]["by_source"]


if __name__ == "__main__":
    # python tests/test_xfactor_v2.py golden PATH_TO_PRE_CHANGE_PORTFOLIO_PY
    if len(sys.argv) == 3 and sys.argv[1] == "golden":
        mod = _load_module(sys.argv[2], "_portfolio_golden")
        pool, inp, sims = _slate()
        gold = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        gold["builder"] = {name: json.loads(json.dumps(_fingerprint(_build(mod, pool, inp, sims,
                                                                          case))))
                           for name, case in _cases().items()}
        GOLDEN.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN} from {mod.__file__}")
    else:
        print(__doc__)
