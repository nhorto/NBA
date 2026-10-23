"""Contest-type aware ownership row cap (B5): 0.97 for cash / double-up, 0.92 for GPPs.

GPP output must be bit-identical to the pre-B5 code. The goldens below are SHA-256 hashes of the
float64 ownership vector and of the sampled field (lineups + counts) of `HEAD:field.py` before the
change, on `synth_pool(1)`; if `var/tmp/b5/field_head.py` exists (that HEAD copy) it is also
compared live.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_dfs_field import inputs_for, synth_pool

from courtside.proj.dfs import field as F

GOLDEN = {   # archetype -> (sha256 of own float64 bytes, sha256 of field lineups + counts int64)
    "large_gpp": ("5d5d6da55da14e5d016a757000cd38ba0db28d2b37b61b10071e69d309c3b39b",
                  "f3e3f52d9e101db2055144ced2dfb86798e5910ee962231b2c822f14013f4e04"),
    "small_gpp": ("243c0316b55d19a4601536f5708536449c3cc992074fb58294714fcf7851aa40",
                  "cc975b34f9cb8832a7d38c42d36c05966ff7fe9ee08515bc0eaaa66ec5844fcf"),
}
HEAD_COPY = Path(__file__).parents[1] / "var" / "tmp" / "b5" / "field_head.py"


def _h(a: np.ndarray, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def _build(mod, arch):
    pool = synth_pool(1)
    inp = inputs_for(pool)
    op = mod.ownership_proxy(inp, pool, archetype=arch)
    f = mod.build_field(pool, inp, mod.preset_contest("gpp", 2000), n=3000, n_solves=20, seed=3,
                        archetype=arch)
    return op, f


def test_constants_and_versions():
    assert F.ROW_CAP_GPP == 0.92 and F.ROW_CAP_CASH == 0.97 and F.ROW_CAP == F.ROW_CAP_GPP
    assert F.row_cap_for("cash") == 0.97
    assert F.row_cap_for("small_gpp") == F.row_cap_for("large_gpp") == 0.92
    assert F.FIELD_VERSION == "field-proxy-1.2" and F.MODEL == "own-proxy-1.2"


@pytest.mark.parametrize("arch", ["large_gpp", "small_gpp"])
def test_gpp_bit_identical_to_head(arch):
    op, f = _build(F, arch)
    own_hash, field_hash = GOLDEN[arch]
    assert _h(op.own, "<f8") == own_hash
    assert _h(np.concatenate([f.lineups.ravel(), f.counts]), "<i8") == field_hash
    assert op.own.max() == pytest.approx(F.ROW_CAP_GPP) and op.cap == F.ROW_CAP_GPP


@pytest.mark.skipif(not HEAD_COPY.exists(), reason="HEAD copy of field.py not present")
@pytest.mark.parametrize("arch", ["large_gpp", "small_gpp"])
def test_gpp_matches_live_head_copy(arch):
    spec = importlib.util.spec_from_file_location("courtside.proj.dfs.field_head", HEAD_COPY)
    head = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = head
    spec.loader.exec_module(head)
    op0, f0 = _build(head, arch)
    op1, f1 = _build(F, arch)
    assert np.array_equal(op0.own, op1.own)
    assert np.array_equal(f0.lineups, f1.lineups) and np.array_equal(f0.counts, f1.counts)


def test_cash_can_exceed_gpp_cap_but_never_the_cash_cap():
    pool = synth_pool(1)
    op = F.ownership_proxy(inputs_for(pool), pool, archetype="cash")
    assert F.ROW_CAP_GPP + 1e-6 < op.own.max() <= F.ROW_CAP_CASH + 1e-9
    assert op.cap == F.ROW_CAP_CASH and op.own.sum() == pytest.approx(8.0, abs=1e-6)
    # the same weights as a GPP are clipped at 0.92
    gpp = F.ownership_proxy(inputs_for(pool), pool, archetype="cash", row_cap=F.ROW_CAP_GPP)
    assert gpp.own.max() <= F.ROW_CAP_GPP + 1e-9


def test_cash_field_targets_use_the_cash_cap():
    pool = synth_pool(1)
    f = F.build_field(pool, inputs_for(pool), F.preset_contest("double_up", 1000), n=3000,
                      n_solves=20, seed=3)
    assert f.archetype == "cash" and f.ownership.cap == F.ROW_CAP_CASH
    assert f.realized_ownership().max() <= F.ROW_CAP_CASH + 1e-9


def test_sinkhorn_cap_binds_at_the_given_value():
    w = np.ones((6, 1))
    w[0, 0] = 1e3                       # one player demands ~99.5 % of the slot
    counts = np.array([1.0])
    for cap in (F.ROW_CAP_GPP, F.ROW_CAP_CASH):
        x = F.sinkhorn(w, counts, cap)
        assert x.sum() == pytest.approx(1.0) and x[0, 0] == pytest.approx(cap, abs=1e-4)
        assert x.sum(1).max() <= cap + 1e-4   # sinkhorn ends on a column step
