"""Contest-aware late swap (IMPROVEMENTS.md D8; `proj/dfs/swap_contest.py`, switch
COURTSIDE_SWAP_CONTEST, default OFF).

OFF must leave `swap.plan` bit-identical: the golden `fixtures/swap_off_golden.json` was written
from the PRE-CHANGE swap.py loaded in place (`python tests/test_swap_contest.py golden
PRE_CHANGE_SWAP_PY`). ON: locked slots never change, ruled-out players leave, each entry's pick
is the best joint E[payout] (GPP) / P(cash) (cash) candidate on the conditioned sims, and an
entry only changes for a gain above the noise. Synthetic slate (test_dfs_swap.py) and sims
(test_dfs_field.py); the context is built in place of `swap_contest.prepare`.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside import db                                         # noqa: E402
from courtside.proj.dfs import field as F                        # noqa: E402
from courtside.proj.dfs import portfolio as PF                   # noqa: E402
from courtside.proj.dfs import rules as R                        # noqa: E402
from courtside.proj.dfs import swap                              # noqa: E402
from courtside.proj.dfs import swap_contest as SC                # noqa: E402
from courtside.proj.dfs import topn as TN                        # noqa: E402

from test_dfs_swap import (CLASSIC, G1, G2, G3, T, assert_locked_untouched,   # noqa: E402
                           capture, entry_of, fppg, payload)

GOLDEN = Path(__file__).parent / "fixtures" / "swap_off_golden.json"
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="golden recorded on Linux")


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def slate(conn):
    capture(conn, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
    return "S1"


# ============================================================================== OFF

def _strip(x):
    if isinstance(x, dict):
        return {str(k): _strip(v) for k, v in x.items() if k not in ("seconds",)}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_strip(v) for v in (sorted(x) if isinstance(x, (set, frozenset)) else x)]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
    return x


def _fingerprint(sp) -> dict:
    return _strip({
        "results": [{"key": r.entry.key, "status": r.status, "objective": r.objective,
                     "after": r.after_pairs, "notes": r.notes, "before_mean": r.before_mean,
                     "after_mean": r.after_mean, "after_salary": r.after_salary,
                     "slots": [[s.slot, s.fixed, s.fixed_why, s.changed, s.reason]
                               for s in r.slots]} for r in sp.results],
        "warnings": sp.warnings, "portfolio": sp.portfolio, "ruled_out": sp.ruled_out,
        "summary": {k: v for k, v in sp.summary().items() if k != "seconds"}})


def _off_cases(mod, conn, slate) -> dict:
    """A few plans covering the solve levels, cuts, caps, min_gain, scope and identical
    entries (`mod` = the swap module under test)."""
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    rnd = random.Random(11)
    out = {}
    for trial in range(5):
        noisy = {k: v * rnd.uniform(0.6, 1.4) for k, v in base.items()}
        res = TN.top_n(pool, noisy, CLASSIC, 4, min_diff=2)
        entries = [entry_of(r.lineup, f"t{trial}-{r.rank}",
                            contest=rnd.choice([None, "gpp1", "cash1"])) for r in res]
        now = rnd.choice([T(G1), T(G1) + timedelta(minutes=30), T(G2) + timedelta(seconds=1)])
        open_ids = sorted({pid for e in entries for _, pid in e.pairs
                           if not pool.by_id()[pid].is_locked(now)})
        out_ = {pid: "test: out" for pid in rnd.sample(open_ids, min(2, len(open_ids)))}
        shock = {k: v * rnd.uniform(0.5, 1.5) for k, v in base.items()}
        sp = mod.plan(conn, slate, now=now, entries=entries, projections=shock, extra_out=out_,
                      min_gain=rnd.choice([0.0, 0.5]), max_exposure=rnd.choice([None, 0.75]),
                      max_overlap=rnd.choice([None, 6]))
        out[f"random{trial}"] = _fingerprint(sp)
    res = TN.top_n(pool, base, CLASSIC, 3, min_diff=1)
    es = [entry_of(r.lineup, f"o{r.rank}", contest="gpp1") for r in res]
    es.append(entry_of(res.lineups[0].lineup, "o1b", contest="gpp1"))
    sp = mod.plan(conn, slate, now=T("2026-10-20T20:30:00Z"), entries=es, projections=base,
                  max_overlap=5, min_gain=-1e9)
    out["overlap"] = _fingerprint(sp)
    victim = next(pid for _, pid in res.lineups[0].lineup.slot_pairs()
                  if not pool.by_id()[pid].is_locked(T(G1) + timedelta(minutes=1)))
    sp = mod.plan(conn, slate, now=T(G1) + timedelta(minutes=1), entries=es[:3],
                  projections=base, scope="news", extra_out={victim: "test"})
    out["news"] = _fingerprint(sp)
    return out


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("courtside.proj.dfs.swap", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get(spec.name)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules[spec.name] = saved
    return mod


@LINUX_ONLY
def test_off_is_bit_identical_to_the_pre_change_swap(conn, slate):
    assert SC.contest_swap() is False                     # the default
    gold = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(_off_cases(swap, conn, slate)))
    assert got.keys() == gold.keys()
    for k in gold:
        assert got[k] == gold[k], k


def test_switch_parsing():
    saved = SC.CONTEST_SWAP
    try:
        for v, want in (("1", True), ("on", True), ("0", False), ("off", False), ("", False)):
            SC.set_contest_swap(v)
            assert SC.contest_swap() is want
        SC.set_contest_swap(True)
        assert SC.contest_swap() is True
    finally:
        SC.CONTEST_SWAP = saved
    assert SC._on(None, False) is False and SC._on("junk", True) is True


# ============================================================================== pieces

def _scen_sims(S=4000, n=6, seed=0):
    """A sims dict with P10 scenario arrays: column 0 drawn out in ~30 % of rows, column 1 in
    ~2 %; column 2 (his teammate) scores 10 more when column 0 is out."""
    rng = np.random.default_rng(seed)
    absent = np.zeros((S, n), bool)
    absent[:, 0] = rng.random(S) < 0.3
    absent[:, 1] = rng.random(S) < 0.02
    fp = rng.normal(20, 5, (S, n)).astype(np.float32)
    fp[absent] = 0
    fp[:, 2] += 10 * absent[:, 0]
    z = {"fp": fp, "dk_id": np.array([f"d{j}" for j in range(n)]),
         "player_id": np.arange(n), "team": np.array(["A"] * n),
         "min_t": np.zeros((n, S), np.uint8), "absent_bits": np.packbits(absent, axis=1),
         "scen_team": np.array(["A"]), "scen_idx": np.zeros((S, 1), np.uint8),
         "scen_off": np.array([0, 1]), "scen_kept": np.zeros((1, n), bool)}
    return z, absent


def test_condition_rows_keeps_the_drawn_out_rows():
    z, absent = _scen_sims()
    rows, info = SC.condition_rows(z, {"d0"}, min_rows=500)
    assert info["scenarios"] and [c["player_id"] for c in info["conditioned"]] == ["d0"]
    assert np.array_equal(rows, np.flatnonzero(absent[:, 0]))
    assert z["fp"][rows, 2].mean() > z["fp"][:, 2].mean() + 5        # his teammate's bump
    # too few rows: skipped (the column is only zeroed later)
    rows, info = SC.condition_rows(z, {"d0", "d1"}, min_rows=500)
    assert [c["player_id"] for c in info["skipped"]] == ["d1"]
    assert np.array_equal(rows, np.flatnonzero(absent[:, 0]))
    # within given rows; unknown ids ignored
    sub = np.arange(1000)
    rows, info = SC.condition_rows(z, {"d0", "zz"}, rows=sub, min_rows=10)
    assert set(rows) <= set(sub) and absent[rows, 0].all()
    # no scenario arrays (runs before P10): rows unchanged
    old = {"fp": z["fp"], "dk_id": z["dk_id"]}
    rows, info = SC.condition_rows(old, {"d0"})
    assert not info["scenarios"] and len(rows) == len(z["fp"])


def test_cash_ladder_prices_the_chance_to_cash():
    c = F.preset_contest("double_up", 200, 5.0)
    lad = SC.cash_ladder(PF.Ladder(c))
    assert lad.g(np.array([1, c.paid, c.paid + 1]), np.array([0, 0, 0])).tolist() == [1, 1, 0]
    # a tie across the cash line splits: 2 paid of 4 tied places
    assert lad.g(np.array([c.paid - 1]), np.array([3]))[0] == pytest.approx(0.5)


# ============================================================================== ON

def _context(conn, slate, *, locked_points=None, outs=(), S=3000, seed=0):
    pool = R.build_pool(conn, slate)
    inputs = F.pool_inputs(pool, proj=fppg(pool))
    rng = np.random.default_rng(seed)
    mu = np.array([p.fppg or 0 for p in pool.players])
    fp = np.maximum(0, rng.normal(mu, 0.35 * mu + 2, (S, len(mu)))).astype(np.float32)
    sims = {"fp": fp, "dk_id": np.array([p.dk_player_id for p in pool.players])}
    return SC.Context(run_id="r-test", pool=pool, inputs=inputs, sims=sims,
                      rows_info={"scenarios": False, "rows": S}, outs=set(outs),
                      locked_points=dict(locked_points or {}), blended=False, seed=seed,
                      conn=conn, info={"slate_id": slate})


@pytest.fixture
def small_contests(monkeypatch):
    """Small presets for every entry (fast fields)."""
    gpp, cash = F.preset_contest("gpp", 400, 3.0), F.preset_contest("double_up", 200, 5.0)
    monkeypatch.setattr(SC.Context, "contest_for",
                        lambda self, cid, mode: cash if mode == "cash" else gpp)
    monkeypatch.setattr(SC, "FIELD_SOLVES", 20)
    return gpp, cash


def _on_plan(conn, slate, ctx, monkeypatch, **kw):
    monkeypatch.setattr(SC, "prepare", lambda *a, **k: (ctx, None))
    return swap.plan(conn, slate, contest_aware=True, **kw)


def test_on_keeps_locked_slots_and_removes_ruled_out_players(conn, slate, small_contests,
                                                             monkeypatch):
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    now = T(G1) + timedelta(minutes=30)
    res = TN.top_n(pool, base, CLASSIC, 4, min_diff=2)
    es = [entry_of(r.lineup, f"g{r.rank}", contest="gpp1") for r in res.lineups[:3]] + \
        [entry_of(res.lineups[3].lineup, "c1", contest="cash1")]
    open_ids = sorted({pid for e in es for _, pid in e.pairs
                       if not pool.by_id()[pid].is_locked(now)})
    out = {open_ids[0]: "test: out", open_ids[-1]: "test: out"}
    locked = {pid: 55.0 for e in es for _, pid in e.pairs if pool.by_id()[pid].is_locked(now)}
    ctx = _context(conn, slate, locked_points=locked, outs=out)
    sp = _on_plan(conn, slate, ctx, monkeypatch, now=now, entries=es, projections=base,
                  extra_out=out)
    assert_locked_untouched(sp, pool, now)
    for r in sp.results:
        after = r.after_pairs if r.status == "swapped" else r.entry.pairs
        fixed = {pid for _, pid in r.entry.pairs if pool.by_id()[pid].is_locked(now)}
        assert not ({p for _, p in after} & set(out)) - fixed, r.entry.key
        assert any(n.startswith("Contest-aware") for n in r.notes), r.notes
    sm = sp.summary()["contest"]
    assert sm["version"] == SC.VERSION and sm["locked_points"] == len(locked)
    assert set(sm["decisions"]) == {e.key for e in es}
    objs = {k: d["objective"] for k, d in sm["decisions"].items()}
    assert objs["c1"] == "p_cash" and objs["g1"] == "e_payout"
    assert not [w for w in sp.warnings if "safety check" in w or "failed" in w]
    # the locked players' columns carry their realised points, the Outs score 0
    cs = next(iter(ctx.contests.values()))
    for pid, v in locked.items():
        if pid in cs.col:
            assert np.all(cs.fp[:, cs.col[pid]] == np.float32(v))
    for pid in out:
        if pid in cs.col:
            assert not cs.fp[:, cs.col[pid]].any()


def test_on_pick_is_the_best_joint_e_payout_candidate(conn, slate, small_contests):
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    ctx = _context(conn, slate)
    gpp = small_contests[0]
    res = TN.top_n(pool, base, CLASSIC, 12, min_diff=1)
    others = [r.lineup.slot_pairs() for r in res.lineups[:2]]
    cands = [r.lineup for r in res.lineups[2:]]
    it = SC.Pending(gk=("x",), mode="mean", contest=gpp, copies=1, before=others[0],
                    keep_ok=False, cands=cands, group="g")
    d = SC.decide(ctx, [it], {"g": [(others[1], 1)]}, n_slots=8)[("x",)]
    assert d["lineup"] is not None and d["objective"] == "e_payout"
    # brute force: the joint E[payout] of (other, candidate) on the same sims, field and dupes
    cs = ctx.get(gpp)
    lus = [list(others[1])] + [list(others[0])] + [c.slot_pairs() for c in cands]
    ev = F.evaluate(cs.field, cs.fp, gpp, lus, seed=ctx.seed, tiers=False, own_entries=2,
                    per_sim=True)
    P = PF.dupe_pool(cs.field, ev, lus, gpp, gpp.field_size - 2, np.random.default_rng(3))
    vals = [SC._joint(cs.lad, P, [0, 2 + i]).sum(1).mean() for i in range(len(cands))]
    assert d["value"] == pytest.approx(max(vals))
    assert cands.index(d["lineup"]) == int(np.argmax(vals))


def test_on_keeps_an_entry_unless_the_gain_clears_the_noise(conn, slate, small_contests):
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    ctx = _context(conn, slate)
    gpp = small_contests[0]
    res = TN.top_n(pool, base, CLASSIC, 6, min_diff=1)
    before = res.lineups[0].lineup.slot_pairs()
    # the only candidate is the entry itself: no gain, kept
    it = SC.Pending(gk=("k",), mode="mean", contest=gpp, copies=1, before=before, keep_ok=True,
                    cands=[res.lineups[0].lineup], group="g")
    d = SC.decide(ctx, [it], {}, n_slots=8)[("k",)]
    assert d["lineup"] is None and d["gain"] == pytest.approx(0.0)
    # with a high GAIN_Z nothing changes; with -inf the best candidate is taken
    it = SC.Pending(gk=("k",), mode="mean", contest=gpp, copies=1, before=before, keep_ok=True,
                    cands=[r.lineup for r in res.lineups[1:]], group="g")
    saved = SC.GAIN_Z
    try:
        SC.GAIN_Z = 1e9
        assert SC.decide(ctx, [it], {}, n_slots=8)[("k",)]["lineup"] is None
        SC.GAIN_Z = -1e9
        d = SC.decide(ctx, [it], {}, n_slots=8)[("k",)]
        if d["gain"] > 0:
            assert d["lineup"] is not None
    finally:
        SC.GAIN_Z = saved
    assert "Contest-aware" in SC.note(d)


def test_on_without_sims_falls_back_to_the_mean_resolve(conn, slate, monkeypatch):
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    now = T(G1) + timedelta(minutes=1)
    res = TN.top_n(pool, base, CLASSIC, 3, min_diff=1)
    es = [entry_of(r.lineup, f"f{r.rank}", contest="gpp1") for r in res]
    out = {next(p for _, p in es[0].pairs if not pool.by_id()[p].is_locked(now)): "test"}
    off = swap.plan(conn, slate, now=now, entries=es, projections=base, extra_out=out)
    monkeypatch.setattr(SC, "prepare", lambda *a, **k: (None, "the run has no stored sims"))
    on = swap.plan(conn, slate, now=now, entries=es, projections=base, extra_out=out,
                   contest_aware=True)
    assert any("no stored sims" in w for w in on.warnings)
    assert [r.after_pairs for r in on.results] == [r.after_pairs for r in off.results]
    assert "contest" not in off.summary() and "contest" not in on.summary()


if __name__ == "__main__":
    # regenerate the golden ONLY from the module it pins: the pre-change swap.py
    if len(sys.argv) < 3 or sys.argv[1] != "golden":
        raise SystemExit("usage: python tests/test_swap_contest.py golden PRE_CHANGE_SWAP_PY")
    import tempfile
    mod = _load_module(sys.argv[2])
    with tempfile.TemporaryDirectory() as d:
        import os
        os.environ["COURTSIDE_DATA"] = d
        os.environ["COURTSIDE_NO_WRITER"] = "1"
        c = db.connect(Path(d) / "t.db")
        capture(c, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
        gold = json.loads(json.dumps(_off_cases(mod, c, "S1")))
    GOLDEN.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN, "from", sys.argv[2])
