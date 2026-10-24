"""B6: the Showdown captain ownership term (`field.cpt_utility`, switch COURTSIDE_SD_CPT, OFF).

OFF is bit-identical to HEAD (goldens recorded from the pre-B6 tree, Linux only: float hashes);
ON changes only the CPT column of Showdown slates (Σ 600 %, CPT column Σ 100 %, row cap kept),
captains move toward the best public projections and away from cheap high-value players, the
"+cpt" tag reaches the ownership and field versions, and Classic is untouched. Also the B6
evaluation metrics (`showdown.own_metrics`) and the no-data path of `showdown.ownership_eval`.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import showdown as SD

FIX = Path(__file__).parent / "fixtures" / "dfs"
CLASSIC = R.Rules.from_payload(json.loads((FIX / "rules_70.json").read_text()))
SHOWDOWN = R.Rules.from_payload(json.loads((FIX / "rules_81.json").read_text()))
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="goldens recorded on Linux")

# Recorded from HEAD 91d131d (before B6) with the pools below: sha256(by_slot)[:16], lambda,
# version; the field: sha256(lineups + counts)[:16] and version.
HEAD_GOLDEN = {
    "sd/small_gpp": "ccc1a94747528895 lam=0.182373046875 v=own-proxy-1.2-handset/small_gpp",
    "sd/large_gpp": "3ae98c9497086e18 lam=0.157470703125 v=own-proxy-1.2-handset/large_gpp",
    "sd/cash": "2e706b12530b4db9 lam=0.305419921875 v=own-proxy-1.2-handset/cash",
    "sd/field": "22b16baf94cdbdf1 field-proxy-1.2",
    "cl/small_gpp": "91219969e0006320 lam=0.706787109375 v=own-proxy-1.2-handset/small_gpp",
    "cl/large_gpp": "f0aff9585496e9de lam=0.645263671875 v=own-proxy-1.2-handset/large_gpp",
    "cl/cash": "70aaf4e0af352aac lam=1.034912109375 v=own-proxy-1.2-handset/cash",
    "cl/field": "526886372bd52f83 field-proxy-1.2",
}


def mk(pid, team, opp, pos, salary, game, rules, fppg=None, status=None, starter=None):
    positions = tuple(pos.split("/"))
    slots = rules.eligible_slots(positions)
    sbs = {s: int(round(salary * rules.slot_multiplier(s))) for s in slots}
    return R.PoolPlayer(
        dk_player_id=str(pid), dk_dk_id=None, name=f"P{pid}", team=team, positions=positions,
        salary=salary, slots=slots, salary_by_slot=sbs,
        draftable_ids={s: f"d{pid}{s}" for s in slots}, game_id=game, competition_id=None,
        game_label=game, home=team, away=opp, start_time=None, status=status, news_status=None,
        starter=starter, swappable=None, fppg=fppg)


def sd_pool(seed=0, n=18):
    rnd = random.Random(seed)
    players = []
    for k in range(n):
        t, o = ("AAA", "BBB") if k % 2 else ("BBB", "AAA")
        sal = rnd.randrange(5, 60) * 200
        players.append(mk(k + 1, t, o, ["PG", "SG", "SF", "PF", "C"][k % 5], sal, "G1", SHOWDOWN,
                          fppg=round(max(3.0, sal / 1000 * 4.6 + rnd.gauss(0, 4)), 1),
                          status="Q" if k == 3 else None))
    return R.SlatePool("sd", 81, "showdown", "x", SHOWDOWN, tuple(players))


def cl_pool(seed=1):
    rnd = random.Random(seed)
    poss = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C", "SG/SF"]
    players, pid = [], 0
    for g in range(4):
        a, b = f"T{g}A", f"T{g}B"
        for t, o in ((a, b), (b, a)):
            for k in range(9):
                pid += 1
                sal = rnd.randrange(30, 110) * 100
                players.append(mk(pid, t, o, poss[k % 9], sal, f"G{g}", CLASSIC,
                                  round(max(4.0, sal / 1000 * 5 + rnd.gauss(0, 5)), 1)))
    return R.SlatePool("cl", 70, "classic", "x", CLASSIC, tuple(players))


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@pytest.fixture(autouse=True)
def _off():
    F.set_sd_cpt(False)
    yield
    F.set_sd_cpt(F.sd_cpt_mode(None))


def test_switch_parsing_and_default_off():
    assert F.SD_CPT_DEFAULT is False
    assert F.sd_cpt_mode(None) is False and F.sd_cpt_mode("") is False
    for v in ("1", "on", "true", "YES", True):
        assert F.sd_cpt_mode(v) is True
    for v in ("0", "off", "false", "no", False, "bogus"):
        assert F.sd_cpt_mode(v) is False
    F.set_sd_cpt("1")
    assert F.sd_cpt() is True
    F.set_sd_cpt("0")
    assert F.sd_cpt() is False


@LINUX_ONLY
def test_off_is_bit_identical_to_head():
    got = {}
    for name, pool in (("sd", sd_pool()), ("cl", cl_pool())):
        for arch in ("small_gpp", "large_gpp", "cash"):
            op = F.ownership_proxy(F.pool_inputs(pool), pool, archetype=arch)
            got[f"{name}/{arch}"] = f"{_h(op.by_slot)} lam={op.lam!r} v={op.version}"
        f = F.build_field(pool, F.pool_inputs(pool), F.preset_contest("gpp", 2000), n=3000,
                          n_solves=8, seed=3, threads=1)
        got[f"{name}/field"] = (hashlib.sha256(np.ascontiguousarray(f.lineups).tobytes()
                                               + np.ascontiguousarray(f.counts).tobytes())
                                .hexdigest()[:16] + " " + f.version)
    assert got == HEAD_GOLDEN


def test_off_default_equals_explicit_off():
    pool = sd_pool(2)
    inp = F.pool_inputs(pool)
    a = F.ownership_proxy(inp, pool, archetype="large_gpp")
    b = F.ownership_proxy(inp, pool, archetype="large_gpp", cpt=False)
    assert np.array_equal(a.by_slot, b.by_slot) and a.lam == b.lam and a.version == b.version
    assert F.SD_CPT_TAG not in a.version and set(a.params) == set(F.DEFAULT_PARAMS)


@pytest.mark.parametrize("arch", ["cash", "small_gpp", "large_gpp"])
def test_on_showdown_sums_caps_and_tag(arch):
    pool = sd_pool(4)
    op = F.ownership_proxy(F.pool_inputs(pool), pool, archetype=arch, cpt=True)
    assert op.version.endswith(F.SD_CPT_TAG)
    assert op.own.sum() == pytest.approx(6.0, abs=1e-6)
    assert np.allclose(op.by_slot.sum(0), op.slot_counts, atol=1e-6)
    assert sum(op.captain().values()) == pytest.approx(1.0, abs=1e-6)
    assert op.own.max() <= op.cap + 1e-9 and op.by_slot.min() >= 0
    assert {"c_value", "c_proj", "c_star"} <= set(op.params)


def test_on_moves_captains_to_stars_and_off_cheap_value():
    """A star (best projection, expensive) and a cheap high-value player: ON raises the star's
    CPT share of his ownership and lowers the cheap player's."""
    players = [mk(1, "AAA", "BBB", "PG", 11800, "G1", SHOWDOWN, fppg=52.0),
               mk(2, "BBB", "AAA", "SF", 3000, "G1", SHOWDOWN, fppg=24.0)]
    rnd = random.Random(7)
    for k in range(3, 19):
        t, o = ("AAA", "BBB") if k % 2 else ("BBB", "AAA")
        sal = rnd.randrange(10, 50) * 200
        players.append(mk(k, t, o, "C", sal, "G1", SHOWDOWN, fppg=round(sal / 1000 * 4.3, 1)))
    pool = R.SlatePool("sd", 81, "showdown", "x", SHOWDOWN, tuple(players))
    inp = F.pool_inputs(pool)
    off = F.ownership_proxy(inp, pool, archetype="large_gpp", cpt=False)
    on = F.ownership_proxy(inp, pool, archetype="large_gpp", cpt=True)
    share = lambda op, p: op.captain()[p] / op.as_dict()[p]          # noqa: E731
    assert share(on, "1") > share(off, "1")
    assert share(on, "2") < share(off, "2")
    assert on.captain()["1"] > off.captain()["1"]


def test_on_leaves_classic_untouched():
    pool = cl_pool(3)
    inp = F.pool_inputs(pool)
    a = F.ownership_proxy(inp, pool, archetype="large_gpp", cpt=False)
    F.set_sd_cpt(True)
    b = F.ownership_proxy(inp, pool, archetype="large_gpp")
    assert np.array_equal(a.by_slot, b.by_slot) and a.version == b.version
    f = F.build_field(pool, inp, F.preset_contest("gpp", 2000), n=2000, n_solves=4, seed=1,
                      threads=1, ownership=b)
    assert f.version == F.FIELD_VERSION


def test_on_field_valid_and_tagged():
    pool = sd_pool(5)
    inp = F.pool_inputs(pool)
    F.set_sd_cpt(True)
    op = F.ownership_proxy(inp, pool, archetype="small_gpp")
    assert F.SD_CPT_TAG in op.version
    f = F.build_field(pool, inp, F.preset_contest("gpp", 2000), n=3000, n_solves=8, seed=1,
                      threads=1, ownership=op)
    assert f.version == F.FIELD_VERSION + F.SD_CPT_TAG
    by_id = pool.by_id()
    for row in f.lineups[:300]:
        ps = [by_id[f.pids[i]] for i in row]
        assert len(set(row)) == 6 and f.slots[0] == "CPT"
        assert sum(p.salary_for(s, SHOWDOWN) for s, p in zip(f.slots, ps)) <= 50000
        assert len({p.team for p in ps}) >= 2
    # the sampled field reproduces the captain split roughly
    cpt_real = np.bincount(f.lineups[:, 0], weights=f.counts, minlength=len(f.pids)) / f.counts.sum()
    top = int(np.argmax(op.by_slot[:, op.slot_names.index("CPT")]))
    assert cpt_real[top] > 0.5 * op.by_slot[top, op.slot_names.index("CPT")]
    fs = F.build_fields(pool, inp, F.preset_contest("gpp", 2000), k=2, n=2000, n_solves=4,
                        seed=1, threads=1, ownership=op)
    assert fs.version == F.FIELD_VERSION + F.FIELD_K_TAG + F.SD_CPT_TAG
    assert all(x.version == fs.version for x in fs.fields)
    assert F.field_version(op, k=False) == F.FIELD_VERSION + F.SD_CPT_TAG
    assert F.field_version(None, k=True) == F.FIELD_VERSION + F.FIELD_K_TAG


def test_own_metrics_by_slot_and_paired():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(3):
        for p in range(15):
            for slot, scale in (("CPT", 0.15), ("UTIL", 0.5)):
                act = float(rng.uniform(0, scale))
                rows.append({"contest_id": f"c{c}", "slate_id": f"s{c}", "player_id": 100 + p,
                             "slot": slot, "actual": act,
                             "off": float(np.clip(act + rng.normal(0, 0.05), 0, 1)),
                             "cpt": act})
    m = SD.own_metrics(pd.DataFrame(rows), ("off", "cpt"), boot=50)
    for slot in ("CPT", "UTIL"):
        r = m[slot]
        assert r["contests"] == 3 and r["slates"] == 3
        assert r["cpt"]["mae_pp"] == pytest.approx(0.0, abs=1e-12)
        assert r["cpt"]["rank_corr"] == pytest.approx(1.0)
        assert r["off"]["mae_pp"] > 0 and np.isfinite(r["off"]["mae_se"])
        d = r["cpt-off"]
        assert d["mae_pp"] == pytest.approx(-r["off"]["mae_pp"])
        assert np.isfinite(d["mae_se"]) and d["rank_corr"] > 0


def test_ownership_eval_without_showdown_uploads(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(data_dir / "courtside.db")
    rep = SD.ownership_eval(conn)
    assert rep["status"] == "no_data" and rep["contests"] == 0
