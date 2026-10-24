"""+c1r ship prep: the tail-remap params and the X factor STRETCH map follow the active copula.

`tail_remap.params_path` / `cells` read `tail_remap_params_c1r.json` only while
`pipeline.COPULA == "v1r"` (missing file -> the v0 file), and `xfactor.stretch_table` returns
``STRETCH_TR_C1R`` only with the tail remap on and copula v1r. Copula v0 (live until the flip) is
bit-identical to the tree before this change: the golden below is the digest of `run_games` with
`+tr` ON (shipped v0 params) computed with HEAD ce02661's `tail_remap.py` (unchanged since
91d131d) loaded in place of the current one:

    C1R_TR_GOLDEN_FROM=<git show ce02661:src/courtside/proj/tail_remap.py > f; f> \\
        pytest tests/test_c1r_ship_prep.py -k golden -s

Exact floats: Linux only (recorded on geekom1 / OpenBLAS). Docs: docs/research/tail-remap.md
(addendum 2026-10-24, c1r)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_proj_copula import game  # noqa: E402,F401  (module fixture)
from test_tail_remap import DAY, _bundle, _digest  # noqa: E402
from test_proj_copula import _spec  # noqa: E402

from courtside.proj import pipeline as P  # noqa: E402
from courtside.proj import role_blend as RB  # noqa: E402
from courtside.proj import tail_remap as TR  # noqa: E402
from courtside.proj.dfs import xfactor as XF  # noqa: E402

LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="goldens recorded on Linux")

# run_games, +tr ON, copula v0, shipped tail_remap_params.json, role blend off, on the
# test_tail_remap fixture (p_out [0], p_half [1, 3], 3,000 sims, seed 11): digest with HEAD
# ce02661's tail_remap.py (before params_path), Linux
GOLDEN_ON_V0 = "73e77a92dd00ff2204643443dfda2d5aab76e200"


@pytest.fixture
def restore():
    state = (TR.TAIL_REMAP, RB.ROLE_BLEND, P.COPULA)
    TR.cells.cache_clear()
    XF._CACHE.clear()
    yield
    P.set_copula(state[2])
    P.set_role_blend(state[1])
    P.set_tail_remap(state[0])
    TR.cells.cache_clear()
    XF._CACHE.clear()


# ------------------------------------------------------------------------------ params selection

@pytest.fixture
def c1r_selected(monkeypatch):
    """The c1r refit mapped in (it ships unselected; see tail_remap.TR_C1R_CANDIDATE)."""
    monkeypatch.setitem(TR.TR_PATHS_BY_COPULA, "v1r", TR.TR_C1R_CANDIDATE)


def test_c1r_refit_ships_unselected(restore):
    assert TR.TR_PATHS_BY_COPULA == {} and TR.TR_C1R_CANDIDATE.exists()
    P.set_copula("v1r")
    assert TR.params_path() == TR.TR_PATH
    P.set_tail_remap(True)
    assert XF.stretch_table() is XF.STRETCH_TR


def test_params_path_follows_the_copula(restore, c1r_selected):
    c1r = TR.TR_PATHS_BY_COPULA["v1r"]
    assert c1r.name == "tail_remap_params_c1r.json" and c1r.exists()
    assert TR.params_path("v0") == TR.TR_PATH
    assert TR.params_path("v1") == TR.TR_PATH
    assert TR.params_path("v1r") == c1r
    P.set_copula("v0")
    assert TR.params_path() == TR.TR_PATH
    P.set_copula("v1r")
    assert TR.params_path() == c1r


def test_missing_c1r_file_falls_back_to_v0(restore, monkeypatch, tmp_path):
    monkeypatch.setitem(TR.TR_PATHS_BY_COPULA, "v1r", tmp_path / "absent.json")
    P.set_copula("v1r")
    assert TR.params_path() == TR.TR_PATH
    assert TR.cells("2024-25") == TR.cells("2024-25", "v0")


def test_cells_read_the_active_copula_file(restore, c1r_selected):
    v0 = json.loads(TR.TR_PATH.read_text())["seasons"]["2024-25"]
    c1 = json.loads(TR.TR_PATHS_BY_COPULA["v1r"].read_text())["seasons"]["2024-25"]
    P.set_copula("v0")
    t0 = TR.cells("2024-25")
    assert t0["bench|*"] == v0["tiers"]["bench"]
    assert t0["role|24-32"] == v0["cells"]["role|24-32"]["q"]
    P.set_copula("v1r")
    t1 = TR.cells("2024-25")
    assert t1["bench|*"] == c1["tiers"]["bench"]
    assert t1["role|24-32"] == c1["cells"]["role|24-32"]["q"]
    assert t1 != t0                                       # the c1r fit is a different table
    P.set_copula("v0")
    assert TR.cells("2024-25") == t0                      # cached per file, no cross-talk


def test_shipped_c1r_params_are_walk_forward():
    par = json.loads(TR.TR_C1R_CANDIDATE.read_text())
    v0 = json.loads(TR.TR_PATH.read_text())
    assert par["copula"] == "v1r" and "+c1r" in par["arm"] and "+tr" not in par["arm"]
    assert par["levels"] == list(TR.LEVELS) and par["shrink"] == TR.SHRINK
    assert par["min_bands"] == v0["min_bands"] and par["fit_pplay_min"] == v0["fit_pplay_min"]
    assert sorted(par["seasons"]) == sorted(v0["seasons"])           # same targets
    for s, e in par["seasons"].items():
        assert e["train"] == v0["seasons"][s]["train"]                # same datasets per target
        assert e["train"] and all(x.split(":")[1] < s for x in e["train"])
        for h in [*e["tiers"].values(), *(c["q"] for c in e["cells"].values())]:
            assert len(h) == len(TR.LEVELS) and all(a < b for a, b in zip(h, h[1:]))
            assert h[0] > TR.MEDIAN and h[-1] < 1


def test_apply_uses_the_c1r_table_under_v1r(restore, monkeypatch, c1r_selected):
    seen = []
    real = TR._cells

    def spy(season, path):
        seen.append(Path(path).name)
        return real(season, path)
    monkeypatch.setattr(TR, "_cells", spy)
    P.set_copula("v0")
    TR.apply([], DAY)
    P.set_copula("v1r")
    TR.apply([], DAY)
    assert seen == ["tail_remap_params.json", "tail_remap_params_c1r.json"]


# ------------------------------------------------------------------------------ v0 bit-identical

def _on_v0(game):  # noqa: F811
    spec, mp, sp = _spec(game, p_out=[0], p_half=[1, 3])
    P.set_copula("v0")
    P.set_role_blend(False)
    P.set_tail_remap(True)
    return P.run_games([spec], _bundle(mp, sp), 3000, 11, P._as_date(DAY))


@LINUX_ONLY
def test_v0_on_matches_head_golden(game, restore):  # noqa: F811
    """+tr ON with copula v0 == HEAD's tail_remap (C1R_TR_GOLDEN_FROM prints the HEAD digest)."""
    src = os.environ.get("C1R_TR_GOLDEN_FROM")
    if src:
        spec = importlib.util.spec_from_file_location("courtside.proj._tail_remap_head", src)
        head = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(head)
        head.TR_PATH = TR.TR_PATH                    # the shipped v0 params file
        saved = P.TR
        P.TR = head
        try:
            head.set_tail_remap(True)
            a = _on_v0(game)
        finally:
            P.TR = saved
        print(f"\nHEAD digest {_digest(a)} remapped {a.meta['tail_remap']}")
        assert a.meta["tail_remap"]["remapped"] > 0
    a = _on_v0(game)
    assert a.meta["tail_remap"]["remapped"] > 0
    assert _digest(a) == GOLDEN_ON_V0


# ------------------------------------------------------------------------------ X factor

def test_stretch_table_selection(restore, c1r_selected):
    P.set_tail_remap(False)
    for c in ("v0", "v1", "v1r"):
        P.set_copula(c)
        assert XF.stretch_table() is XF.STRETCH                 # remap off: the raw-sims map
    P.set_tail_remap(True)
    P.set_copula("v0")
    assert XF.stretch_table() is XF.STRETCH_TR                  # live (v0) unchanged
    P.set_copula("v1")
    assert XF.stretch_table() is XF.STRETCH_TR
    P.set_copula("v1r")
    assert XF.stretch_table() is XF.STRETCH_TR_C1R


def test_stretch_follows_the_params_file_in_use(restore, monkeypatch, tmp_path, c1r_selected):
    """Without the c1r params file the remap reads the v0 params, and the X factor uses
    STRETCH_TR (the state E3 tested `+c1r` in)."""
    P.set_tail_remap(True)
    P.set_copula("v1r")
    assert XF.stretch_table() is XF.STRETCH_TR_C1R
    monkeypatch.setitem(TR.TR_PATHS_BY_COPULA, "v1r", tmp_path / "absent.json")
    assert XF.stretch_table() is XF.STRETCH_TR


def test_stretch_tr_c1r_shape():
    assert set(XF.STRETCH_TR_C1R) == {"bench", "role", "star"}
    for t, (a, b) in XF.STRETCH_TR_C1R.items():
        assert math.isfinite(a) and 0.5 < b < 1.2, t
    assert XF.STRETCH_TR_C1R != XF.STRETCH_TR


def test_calibrate_v0_unchanged_and_v1r_uses_c1r_map(restore, c1r_selected):
    p = np.array([0.02, 0.1, 0.25, 1 / 3, 0.5, 0.8] * 3)
    t = ["bench"] * 6 + ["role"] * 6 + ["star"] * 6
    P.set_tail_remap(True)
    P.set_copula("v0")
    v0 = XF.calibrate(p, t)
    z = np.log(p / (1 - p))
    ab = np.array([XF.STRETCH_TR[x] for x in t])
    np.testing.assert_allclose(v0, 1 / (1 + np.exp(-(ab[:, 0] + ab[:, 1] * z))), atol=1e-15)
    P.set_copula("v1r")
    v1 = XF.calibrate(p, t)
    ab = np.array([XF.STRETCH_TR_C1R[x] for x in t])
    np.testing.assert_allclose(v1, 1 / (1 + np.exp(-(ab[:, 0] + ab[:, 1] * z))), atol=1e-15)
