"""X factor ship prep (D3 / D6): ``xfactor.STRETCH_TR`` (the big-night map refit on `+tr` sims, used
only while the tail remap is on) and the xf-v2 leverage table precomputed in the DFS run path
(`runner._execute` -> `xfactor.store_safe`, read back by `xfactor.apply_v2`). Docs: the "ship prep"
addenda of docs/research/xfactor-v2.md and docs/research/tail-remap.md.

OFF (both switches) must be bit-identical. The golden `fixtures/xf_ship_prep_off_golden.json` was
written from the HEAD (102ed53) xfactor module loaded in place:

    XFSP_GOLDEN_FROM=<HEAD xfactor.py> pytest tests/test_xf_ship_prep.py -k golden

The exact checks are Linux-only (recorded on geekom1 / OpenBLAS)."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj import tail_remap as TRM          # noqa: E402
from courtside.proj.dfs import runner                 # noqa: E402
from courtside.proj.dfs import xfactor as XF          # noqa: E402

from test_dfs_runner import G1, T, avail, conn, pipe, slate3   # noqa: E402,F401  (fixtures)
from test_dfs_xfactor import xf_db                     # noqa: E402,F401  (fixture)
from test_xfactor_v2 import _fake_leverage, _load_module, _strip   # noqa: E402
from test_web_dfs import players                       # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "xf_ship_prep_off_golden.json"
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="goldens recorded on Linux")
GRID = [0.0, 1e-5, 0.003, 0.02, 0.05, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.5, 0.6, 0.8, 0.97, 1.0,
        float("nan")]
TIERS = ["bench", "role", "star", None, "other"]
SHIPPED = {"bench": (-0.2902, 0.8275), "role": (-0.2890, 0.7789), "star": (-0.3186, 0.7724)}


@pytest.fixture
def tr_off():
    saved = TRM.TAIL_REMAP
    TRM.set_tail_remap(False)
    XF._CACHE.clear()
    yield
    TRM.set_tail_remap(saved)
    XF._CACHE.clear()


@pytest.fixture
def v2():
    saved = XF.XF_V2
    XF._CACHE.clear()
    yield XF.set_xf_v2
    XF.XF_V2 = saved
    XF._CACHE.clear()


def _grid():
    p = np.array([g for g in GRID for _ in TIERS])
    t = [x for _ in GRID for x in TIERS]
    return p, t


# ============================================================================== STRETCH_TR

def test_stretch_tr_is_a_separate_table_with_the_same_shape():
    assert set(XF.STRETCH_TR) == set(XF.STRETCH) == {"bench", "role", "star"}
    for t, (a, b) in XF.STRETCH_TR.items():
        assert math.isfinite(a) and 0.5 < b < 1.2, t
    # the shipped map is unchanged (fit on raw sims, xfactor.md §3)
    assert XF.STRETCH == SHIPPED
    assert XF.STRETCH_TR != XF.STRETCH


def test_stretch_table_follows_the_tail_remap_switch(tr_off):
    assert TRM.TAIL_REMAP is False and XF.stretch_table() is XF.STRETCH
    TRM.set_tail_remap(True)
    assert XF.stretch_table() is XF.STRETCH_TR
    TRM.set_tail_remap(False)
    assert XF.stretch_table() is XF.STRETCH


def _by_hand(p, tiers, tab):
    out = []
    for x, t in zip(p, tiers):
        if not np.isfinite(x) or t not in tab:
            out.append(x)
            continue
        z = math.log(min(max(x, XF._EPS), 1 - XF._EPS) / (1 - min(max(x, XF._EPS), 1 - XF._EPS)))
        a, b = tab[t]
        out.append(1 / (1 + math.exp(-(a + b * z))))
    return np.array(out)


def test_calibrate_uses_stretch_off_and_stretch_tr_on(tr_off):
    p, t = _grid()
    off = XF.calibrate(p, t)
    np.testing.assert_allclose(off, _by_hand(p, t, XF.STRETCH), rtol=0, atol=1e-15)
    TRM.set_tail_remap(True)
    on = XF.calibrate(p, t)
    np.testing.assert_allclose(on, _by_hand(p, t, XF.STRETCH_TR), rtol=0, atol=1e-15)
    known = np.array([x in XF.STRETCH for x in t]) & np.isfinite(p)
    assert np.array_equal(on[~known], off[~known], equal_nan=True)   # unknown tier / NaN: as is
    # +tr sims have narrower tails: the refit lifts a mid-range chance (about 1 pt at 1 in 3)
    for tier in ("bench", "role", "star"):
        lo, hi = XF.calibrate([0.33], [tier])[0], None
        TRM.set_tail_remap(False)
        hi = XF.calibrate([0.33], [tier])[0]
        TRM.set_tail_remap(True)
        assert lo > hi, tier


def test_run_table_on_tr_uses_stretch_tr_and_its_own_cache_key(xf_db, tr_off, monkeypatch):
    _, conn = xf_db                    # xf_db sets STRETCH = {} (raw chances)
    monkeypatch.setattr(XF, "STRETCH_TR", {"bench": (0.5, 1.0), "role": (0.5, 1.0),
                                           "star": (0.5, 1.0)})
    off = XF.run_table(conn, "run1")
    TRM.set_tail_remap(True)
    on = XF.run_table(conn, "run1")    # not the OFF entry of the memory cache
    assert on is not off
    for d, r in off.items():
        assert on[d]["raw"] == r["raw"]
        if r["tier"] in ("bench", "role", "star") and 0 < r["raw"] < 1:
            assert on[d]["chance"] > r["chance"]
    TRM.set_tail_remap(False)
    assert XF.run_table(conn, "run1") is off


# ============================================================================== OFF goldens

def _golden_payload(mod, conn) -> dict:
    p, t = _grid()
    return {"calibrate": _strip(list(mod.calibrate(p, t))),
            "run_table": json.loads(json.dumps(_strip(
                {d: dict(r) for d, r in sorted(mod.run_table(conn, "run1").items())})))}


@LINUX_ONLY
def test_golden_off_is_bit_identical_to_head(xf_db, tr_off, monkeypatch):
    """calibrate on a grid and run_table on the web fixture WITH the real STRETCH, both switches
    OFF, equal the HEAD module's output exactly (floats by repr)."""
    _, conn = xf_db
    monkeypatch.setattr(XF, "STRETCH", dict(SHIPPED))
    src = os.environ.get("XFSP_GOLDEN_FROM")
    if src:
        old = _load_module(src, "_xfactor_ship_golden")
        old.VALUE_TOP_EXCLUDE = 0
        old._CACHE.clear()
        GOLDEN.write_text(json.dumps(_golden_payload(old, conn), indent=1, sort_keys=True) + "\n")
    gold = json.loads(GOLDEN.read_text())
    saved = XF.XF_V2
    try:
        XF.set_xf_v2(False)
        XF._CACHE.clear()
        assert _golden_payload(XF, conn) == gold
    finally:
        XF.XF_V2 = saved


# ============================================================================== precompute

def _ok_res(players_: dict) -> dict:
    return {"version": XF.VERSION_V2, "players": players_, "k": 4, "contest": {}, "own_unc": {},
            "n_sims": 100, "seconds": 1.0}


def test_store_safe_off_never_computes_and_leaves_the_detail(v2, monkeypatch):
    v2(False)

    def boom(*a, **k):
        raise AssertionError("precompute called with the switch OFF")
    monkeypatch.setattr(XF, "precompute_run", boom)
    detail = {"a": 1}
    XF.store_safe(detail, None, "S", "r", "2026-10-20T22:50:00Z", {"flags": []})
    assert detail == {"a": 1}


def test_store_safe_on_stores_and_survives_failures(v2, monkeypatch):
    v2(True)
    seen = {}

    def fake(conn, run_id, *, as_of=None, late_news=None):
        seen.update(run_id=run_id, as_of=as_of, late_news=late_news)
        return {"version": XF.VERSION_V2, "players": {}, "mtime": 1, "params": "p"}
    monkeypatch.setattr(XF, "precompute_run", fake)
    d: dict = {}
    XF.store_safe(d, None, "S", "r1", "2026-10-20T22:50:00Z", {"flags": []})
    assert d[XF.DETAIL_KEY]["mtime"] == 1
    assert seen == {"run_id": "r1", "as_of": "2026-10-20T22:50:00Z", "late_news": {"flags": []}}
    monkeypatch.setattr(XF, "precompute_run", lambda *a, **k: None)      # showdown / no sims
    d = {}
    XF.store_safe(d, None, "S", "r1")
    assert d == {}

    def raise_(*a, **k):
        raise RuntimeError("field")
    monkeypatch.setattr(XF, "precompute_run", raise_)
    XF.store_safe(d, None, "S", "r1")
    assert d == {}


def test_precompute_run_passes_the_runs_time_and_late_news(xf_db, monkeypatch):
    _, conn = xf_db
    seen = {}

    def fake(conn_, run_id, **kw):
        seen.update(kw)
        return _ok_res({})
    monkeypatch.setattr(XF, "run_leverage", fake)
    late = {"flags": [{"dk_id": "123", "own_uncertainty": 2.0}]}
    res = XF.precompute_run(conn, "run1", as_of="2026-10-20T22:50:00Z", late_news=late)
    from courtside.proj.dfs import late_news_tag as LNT
    assert seen == {"as_of": "2026-10-20T22:50:00Z", "own_unc": LNT.own_uncertainty(late)}
    assert res["params"] == XF._params_key() and res["mtime"] == XF._sims_mtime(conn, "run1")
    assert XF.precompute_run(conn, "nope") is None


def test_run_leverage_takes_as_of_and_own_unc(monkeypatch, xf_db):
    """The runner passes the run's `now` (the created_at it saves) and its late-news tag, so
    the stored table is computed from the inputs a later page view reads."""
    _, conn = xf_db
    from courtside.proj.dfs import field as F
    seen = {}

    def prepare(conn_, sid, run_id, *, n_sims=None, seed=0, as_of=None):
        seen["as_of"] = as_of
        return {"sims": None}
    monkeypatch.setattr(F, "prepare", prepare)
    assert XF.run_leverage(conn, "run1", as_of="X") == {}
    assert seen["as_of"] == "X"
    at = conn.execute("SELECT created_at FROM dfs_runs WHERE run_id='run1'").fetchone()[0]
    XF.run_leverage(conn, "run1")
    assert seen["as_of"] == at


def test_page_reads_the_stored_table_and_falls_back(xf_db, v2, monkeypatch, tmp_path):
    _, conn = xf_db
    v2(True)
    ps = players()
    fake, hot, other = _fake_leverage(ps)
    conn.execute("UPDATE player_projections SET p_play=0.95 WHERE run_id='run1'")
    calls = []
    monkeypatch.setattr(XF, "run_leverage", lambda *a, **k: calls.append(1) or fake(*a, **k))
    monkeypatch.setattr(XF, "_cache_path", lambda run_id: tmp_path / f"{run_id}.json")
    res = XF.precompute_run(conn, "run1")
    assert calls == [1]
    conn.execute("UPDATE dfs_runs SET detail=? WHERE run_id='run1'",
                 (json.dumps({"x": 1, XF.DETAIL_KEY: res}),))
    conn.commit()
    calls.clear()
    t = XF.run_table(conn, "run1")
    assert calls == [] and not (tmp_path / "run1.json").exists()     # stored: no field build
    assert t[other]["x"] and t[other]["lev"] == 2.5
    # a stale stored table (other params / sims mtime) is ignored: computed once, then cached
    for bad in ({**res, "params": "old"}, {**res, "mtime": res["mtime"] + 1},
                {**res, "version": "xf-v1"}):
        assert XF.stored_leverage(conn, "run1", res["mtime"]) is not None
        conn.execute("UPDATE dfs_runs SET detail=? WHERE run_id='run1'",
                     (json.dumps({XF.DETAIL_KEY: bad}),))
        assert XF.stored_leverage(conn, "run1", res["mtime"]) is None
        conn.execute("UPDATE dfs_runs SET detail=? WHERE run_id='run1'",
                     (json.dumps({XF.DETAIL_KEY: res}),))
    conn.execute("UPDATE dfs_runs SET detail=? WHERE run_id='run1'", (json.dumps({}),))
    conn.commit()
    XF._CACHE.clear()
    t2 = XF.run_table(conn, "run1")
    assert calls == [1] and t2[other]["x"]
    assert json.loads(json.dumps(_strip(t2))) == json.loads(json.dumps(_strip(t)))


# ============================================================================== runner hook

def _run_fp(conn, run_id) -> dict:
    row = dict(conn.execute("SELECT * FROM dfs_runs WHERE run_id=?", (run_id,)).fetchone())
    lus = [tuple(r) for r in conn.execute(
        "SELECT mode, rank, slots_json, salary, mean, p10, p50, p90 FROM dfs_lineups WHERE "
        "run_id=? ORDER BY mode, rank", (run_id,))]
    det = json.loads(row["detail"].replace(run_id, "RUN"))
    return {"detail": det, "lineups": lus, "late": row["late_news_json"],
            "mv": row["model_version"], "created_at": row["created_at"]}


def test_runner_hook_off_adds_nothing_on_stores_the_table(tmp_path, pipe, avail, v2,
                                                         monkeypatch):
    """The same lock10 build in two fresh databases, switch OFF and ON: identical lineups,
    late news, model version and detail (seconds aside), except the stored table ON."""
    from courtside import db
    from test_dfs_runner import G2, G3, capture, payload
    now = T(G1) - timedelta(minutes=10)
    seen = []

    def fake(conn_, run_id, *, as_of=None, late_news=None):
        seen.append((run_id, as_of, late_news))
        return {"version": XF.VERSION_V2, "players": {"1": {"lev": 2.0}}, "mtime": 7,
                "params": XF._params_key()}
    monkeypatch.setattr(XF, "precompute_run", fake)
    fps = {}
    for on in (False, True):
        v2(on)
        c = db.connect(tmp_path / f"{on}.db")
        capture(c, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
        r = runner.run_slate(c, "S1", "lock10", now=now, refresh=False, n=3)
        assert r["status"] == "ok"
        fps[on] = _run_fp(c, r["run_id"])
        if on:
            from courtside.proj.dfs import late_news_tag as LNT
            det = json.loads(c.execute("SELECT detail FROM dfs_runs WHERE run_id=?",
                                       (r["run_id"],)).fetchone()[0])
            assert seen == [(r["run_id"], runner.iso(now), det.get(LNT.DETAIL_KEY))]
        c.close()
    off, on = fps[False], fps[True]
    assert XF.DETAIL_KEY not in off["detail"]
    assert on["detail"].pop(XF.DETAIL_KEY)["players"] == {"1": {"lev": 2.0}}
    assert off == on                                   # lineups, detail, late news, version


def test_runner_hook_failure_never_fails_the_run(conn, slate3, pipe, avail, v2, monkeypatch):
    v2(True)

    def boom(*a, **k):
        raise RuntimeError("fields")
    monkeypatch.setattr(XF, "precompute_run", boom)
    r = runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10),
                         refresh=False, n=2)
    assert r["status"] == "ok"
