"""Ownership uncertainty in the field (IMPROVEMENTS.md D5; switch COURTSIDE_FIELD_K, `field.FIELD_K`).

OFF (the default) must be bit-identical to field-proxy-1.2 before D5: the golden file
`fixtures/field_k_off_golden.json` was written by `python tests/test_field_k.py golden PATH` with
PATH = the pre-change field.py (commit 3c40fbf) and the OFF ownership / field / evaluate / score
path here must reproduce it exactly (floats by repr; Linux only, macOS rounds differently). The
contest builder's OFF path is pinned by tests/test_portfolio_split.py's golden. Platform-free
checks: the switch, the ownership draw (slot totals, cap, zeros, centred on the proxy, the
late-news multiplier), the
K fields (shared solves, own draws), evaluate's row-wise mixture (= per-field scores on their
rows, exactly), the builder ON (bands bracket the means) and realised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj.dfs import field as F            # noqa: E402
from courtside.proj.dfs import portfolio as PF       # noqa: E402

from test_dfs_field import inputs_for, sims_for, synth_pool   # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "field_k_off_golden.json"
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="float rounding differs off Linux")


def _slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1203, seed=2)


def _cands(pool, k=6, Fm=F):
    """A few legal lineups: the k best-projected unique lineups of a small sampled field
    (deterministic, no optimizer dependency)."""
    f = Fm.build_field(pool, inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players}),
                       Fm.preset_contest("gpp", 3000, 3.0), n=1500, n_solves=0, seed=9,
                       threads=1)
    proj = np.array([pool.by_id()[p].fppg for p in f.pids])
    order = np.argsort(-proj[f.lineups].sum(1), kind="stable")[:k]
    return [[(s, f.pids[int(i)]) for s, i in zip(f.slots, f.lineups[j])] for j in order]


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


def _off_fingerprint(Fm) -> dict:
    """Ownership (cash / large GPP), a small cash field and a GPP field, their evaluation of a
    few lineups, all through module `Fm` (the current field.py or the pre-change one)."""
    pool, inp, sims = _slate()
    cands = _cands(pool, 6, Fm)
    out = {}
    for arch, contest in (("cash", Fm.preset_contest("double_up", 300, 2.0)),
                          ("large_gpp", Fm.preset_contest("gpp", 3000, 3.0))):
        op = Fm.ownership_proxy(inp, pool, archetype=arch)
        f = Fm.build_field(pool, inp, contest, n=2500, ownership=op, n_solves=25, seed=3,
                           threads=1)
        fp, _ = Fm.align_sims(sims, f.pids, 1000, 1)
        ev = Fm.evaluate(f, fp, contest, cands, seed=4, per_sim=True)
        out[arch] = {"own": op.own, "by_slot_sum": op.by_slot.sum(0), "lam": op.lam,
                     "lineups": f.lineups, "counts": f.counts, "opt_counts": f.opt_counts,
                     "dupe_c": f.dupe_c, "summary": f.summary(),
                     "cash_line": ev.cash_line, "top1_line": ev.top1_line,
                     "tier_lines": ev.tier_lines, "lineups_eval": ev.lineups,
                     "per_sim_score": ev.per_sim["score"][:50],
                     "per_sim_above": ev.per_sim["above"][:50]}
    return _r(out)


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("courtside.proj.dfs._field_golden", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def slate():
    return _slate()


@pytest.fixture
def k_off():
    saved = F.FIELD_K
    F.set_field_k(0)
    yield
    F.FIELD_K = saved


@pytest.fixture
def k_on():
    saved = F.FIELD_K
    F.set_field_k(4)
    yield 4
    F.FIELD_K = saved


# --------------------------------------------------------------------------- the switch

def test_switch_parsing_and_default():
    assert F.FIELD_K_DEFAULT == 0
    assert F.field_k_mode(None) == 0 and F.field_k_mode("") == 0
    for v in ("0", "off", "false", "no", False):
        assert F.field_k_mode(v, 4) == 0
    for v in ("1", "on", "true", "yes", True, 1):
        assert F.field_k_mode(v) == F.FIELD_K_ON == 4
    assert F.field_k_mode("8") == 8 and F.field_k_mode("k6") == 6
    assert F.field_k_mode("banana") == 0


# --------------------------------------------------------------------------- OFF

@LINUX
def test_off_bit_identical_to_pre_change_golden(k_off):
    assert GOLDEN.exists(), "golden missing: python tests/test_field_k.py golden <old field.py>"
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(_off_fingerprint(F)))
    assert got == want


def test_off_paths_untouched(slate, k_off):
    pool, inp, sims = slate
    c = F.preset_contest("gpp", 3000, 3.0)
    ctx = {"pool": pool, "inputs": inp}
    f = F.field_for(ctx, c, n=2000, n_solves=10)
    assert type(f) is F.Field and f.version == F.FIELD_VERSION
    ev = F.evaluate(f, F.align_sims(sims, f.pids, 300, 0)[0], c, _cands(pool, 2))
    assert "p_cash_k" not in ev.lineups[0] and "field_k" not in ev.info


def test_build_field_reuses_given_solves(slate):
    pool, inp, _ = slate
    c = F.preset_contest("gpp", 3000, 3.0)
    a = F.build_field(pool, inp, c, n=2000, n_solves=12, seed=1, threads=1)
    b = F.build_field(pool, inp, c, n=2000, n_solves=12, seed=5, threads=1,
                      opt_pairs=a.opt_solves)
    assert b.opt_solves == a.opt_solves and b.n_solves == a.n_solves
    assert set(b.opt_keys) == set(a.opt_keys)


# --------------------------------------------------------------------------- ownership draws

@pytest.mark.parametrize("arch", ["cash", "large_gpp"])
def test_perturbed_ownership_keeps_slot_totals_cap_and_zeros(slate, arch):
    pool, inp, _ = slate
    op = F.ownership_proxy(inp, pool, archetype=arch)
    rng = np.random.default_rng(0)
    sd = F.own_sd_for(op)
    assert np.allclose(sd, F.OWN_SD[arch])
    for _ in range(5):
        q = F.perturb_ownership(op, sd, rng)
        assert np.allclose(q.by_slot.sum(0), op.slot_counts, atol=1e-6)
        assert q.own.max() <= op.cap + 1e-9
        assert np.all(q.own[op.own == 0] == 0)
        assert q.version.endswith(F.FIELD_K_TAG)
        assert not np.allclose(q.own, op.own)
    # sd 0: the proxy back (Sinkhorn of an already balanced allocation)
    q0 = F.perturb_ownership(op, np.zeros(len(op.own)), rng)
    assert np.allclose(q0.own, op.own, atol=1e-6)


def test_noise_is_centred_on_the_proxy(slate):
    pool, inp, _ = slate
    p = np.array([0.002, 0.03, 0.2, 0.5, 0.8])
    for sd in (0.86, 2.65):
        c = F.mean_shift(p, np.full(5, sd))
        z = np.random.default_rng(1).standard_normal((200_000, 1))
        m = (1 / (1 + np.exp(-(np.log(p / (1 - p)) + c + sd * z)))).mean(0)
        assert np.allclose(m, p, rtol=0.02, atol=5e-4)
        assert c[0] < 0 < c[-1]          # pulls the long tail down and the chalk up
    op = F.ownership_proxy(inp, pool, archetype="large_gpp")
    rng = np.random.default_rng(3)
    D = np.array([F.perturb_ownership(op, F.own_sd_for(op), rng).own for _ in range(300)])
    act = op.own > 0
    assert np.abs(D.mean(0) - op.own)[act].mean() < 0.006
    top = np.argsort(-op.own)[:8]
    assert abs(D.mean(0)[top].sum() - op.own[top].sum()) < 0.1


def test_late_news_multiplier_widens_that_player(slate):
    pool, inp, _ = slate
    op = F.ownership_proxy(inp, pool, archetype="large_gpp")
    mid = int(np.argsort(-op.own)[8])           # a mid-owned player
    pid = op.pids[mid]
    sd1 = F.own_sd_for(op)
    sd2 = F.own_sd_for(op, {pid: 2.0})
    assert sd2[mid] == pytest.approx(2 * sd1[mid])
    assert np.allclose(np.delete(sd2, mid), np.delete(sd1, mid))
    draws = {}
    for name, sd in (("one", sd1), ("two", sd2)):
        rng = np.random.default_rng(11)
        x = np.array([F.perturb_ownership(op, sd, rng).own[mid] for _ in range(300)])
        draws[name] = np.std(np.log(x / (1 - x)))
    assert draws["two"] > 1.6 * draws["one"]


# --------------------------------------------------------------------------- K fields

def test_build_fields_share_solves_and_differ(slate):
    pool, inp, _ = slate
    c = F.preset_contest("double_up", 300, 2.0)
    fs = F.build_fields(pool, inp, c, k=4, n=2500, n_solves=15, seed=2, threads=1)
    assert fs.k == 4 and fs.version == F.FIELD_VERSION + F.FIELD_K_TAG
    assert all(f.opt_solves == fs.fields[0].opt_solves for f in fs.fields)
    owns = np.array([f.ownership.own for f in fs.fields])
    assert np.ptp(owns, axis=0).max() > 0.05
    assert fs.pids == fs.fields[0].pids and fs.field_size == 300     # delegated
    s = fs.summary()
    assert s["field_k"]["k"] == 4 and len(s["field_k"]["realized_mae_pp"]) == 4
    assert np.allclose(fs.ownership.own, F.ownership_proxy(inp, pool, archetype="cash").own)
    # the same seed: the same fields
    fs2 = F.build_fields(pool, inp, c, k=4, n=2500, n_solves=15, seed=2, threads=1)
    assert all(np.array_equal(a.lineups, b.lineups) and np.array_equal(a.counts, b.counts)
               for a, b in zip(fs.fields, fs2.fields))


def test_evaluate_k_is_the_row_mixture_of_the_fields(slate):
    pool, inp, sims = slate
    c = F.preset_contest("gpp", 3000, 3.0)
    fs = F.build_fields(pool, inp, c, k=3, n=2000, n_solves=10, seed=0, threads=1)
    fp, _ = F.align_sims(sims, fs.pids, None, 0)            # 1203 rows: not a multiple of 3
    cands = _cands(pool, 4)
    ev = F.evaluate(fs, fp, c, cands, seed=5, per_sim=True)
    S = fp.shape[0]
    assert ev.cash_line.shape == (S,) and ev.per_sim["score"].shape == (S, 4)
    assert ev.info["field_k"] == 3 and ev.info["k_mode"] == "rows"
    pc = np.zeros(4)
    for j, f in enumerate(fs.fields):
        r = fs.rows_of(S, j)
        e = F.evaluate(f, fp[r], c, cands, seed=5 + j, per_sim=True)
        assert np.array_equal(ev.cash_line[r], e.cash_line)
        assert np.array_equal(ev.per_sim["above"][r], e.per_sim["above"])
        for i in range(4):
            assert ev.lineups[i]["p_cash_k"][j] == e.lineups[i]["p_cash"]
            pc[i] += e.lineups[i]["p_cash"] * len(r) / S
    for i, x in enumerate(ev.lineups):
        assert x["p_cash"] == pytest.approx(pc[i], abs=1e-12)
        # the row mixture = the plain mean over all rows
        sc, line = ev.per_sim["score"][:, i], (ev.cash_line * F.UNIT * F.WUNIT + F.BIAS)
        assert x["p_cash"] == pytest.approx(float(np.mean(sc >= np.rint(line))), abs=1e-12)
        assert x["p_cash_lo"] <= x["p_cash"] <= x["p_cash_hi"]
        assert x["p_top1_lo"] <= x["p_top1"] <= x["p_top1_hi"]
        assert len(x["dupes_k"]) == 3


def test_evaluate_k_fewer_rows_than_fields(slate):
    pool, inp, sims = slate
    c = F.preset_contest("double_up", 300, 2.0)
    fs = F.build_fields(pool, inp, c, k=4, n=2000, n_solves=10, seed=0, threads=1)
    fp, _ = F.align_sims(sims, fs.pids, 2, 0)
    ev = F.evaluate(fs, fp, c, _cands(pool, 2), per_sim=True)
    assert ev.info["k_mode"] == "all" and ev.cash_line.shape == (2,)
    assert all(len(x["p_cash_k"]) == 4 for x in ev.lineups)
    actual = fp[0]
    out = PF.realised(fs, c, _cands(pool, 2), actual)
    assert out["field_k"] == 4 and len(out["payout_k"]) == 4
    assert out["payout"] == pytest.approx(np.mean(out["payout_k"]))


# --------------------------------------------------------------------------- builder ON

SMALL = dict(n_scenario=20, n_ownpen=10, n_noise=6, n_topmean=6, n_sims=None, n_field=2000,
             field_solves=15, threads=1, solve_limit_s=2.0)


def test_builder_on_mixes_fields_and_reports_bands(slate, k_on):
    pool, inp, sims = slate
    c = F.preset_contest("gpp", 2000, 3.0)
    pid = pool.players[3].dk_player_id
    pf = PF.build_portfolio(pool, inp, sims, c, PF.Settings(n=4, curve_to=6, **SMALL),
                            split="off", own_unc={pid: 2.0})
    assert isinstance(pf.field, F.FieldSet) and pf.field.k == k_on
    assert pf.field.own_unc == {pid: 2.0}
    assert pf.info["field"]["version"] == F.FIELD_VERSION + F.FIELD_K_TAG
    fk = pf.summary["field_k"]
    assert fk["k"] == k_on and fk["own_unc_players"] == 1
    assert fk["p_any_cash_lo"] <= pf.summary["p_any_cash"] <= fk["p_any_cash_hi"]
    for lu in pf.lineups:
        assert lu["p_cash_lo"] <= lu["p_cash"] <= lu["p_cash_hi"]
        assert lu["p_top1_lo"] <= lu["p_top1"] <= lu["p_top1_hi"]
    # the reported P(cash) is the joint rank over all report rows (mixture of the K fields)
    sc = pf.fp.shape[0]
    assert pf.info["n_sims"] == sc


def test_builder_on_split_gs_runs(slate, k_on):
    pool, inp, sims = slate
    c = F.preset_contest("gpp", 2000, 3.0)
    pf = PF.build_portfolio(pool, inp, sims, c, PF.Settings(n=3, curve=False, **SMALL),
                            split="gs")
    assert pf.summary["field_k"]["k"] == k_on
    assert pf.info["version"].endswith(PF.SPLIT_GS_TAG)


def test_builder_off_has_no_k_keys(slate, k_off):
    pool, inp, sims = slate
    c = F.preset_contest("gpp", 2000, 3.0)
    pf = PF.build_portfolio(pool, inp, sims, c, PF.Settings(n=3, curve=False, **SMALL),
                            split="off", own_unc={pool.players[0].dk_player_id: 2.0})
    assert type(pf.field) is F.Field and "field_k" not in pf.summary
    assert not any(k.endswith(("_lo", "_hi")) for k in pf.lineups[0])


# --------------------------------------------------------------------------- score_run ON

def test_score_run_on_tags_version_and_stores_means(slate, tmp_path, monkeypatch, k_on):
    from courtside import db
    pool, inp, sims = slate
    conn = db.connect(tmp_path / "c.db")

    def prepare(conn, slate_id, run_id=None, *, n_sims=None, seed=0):
        return {"pool": pool, "inputs": inp, "info": {"run_id": run_id}, "sims": sims,
                "n_sims": n_sims, "seed": seed}
    monkeypatch.setattr(F, "prepare", prepare)
    monkeypatch.setattr(F, "default_contests", lambda c, s: (
        F.preset_contest("double_up", 300, 5.0), F.preset_contest("gpp", 2000, 3.0)))
    for r, lu in enumerate(_cands(pool, 3)):
        conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json) VALUES (?,?,?,?)",
                     ("r1", "topn", r + 1, json.dumps([{"slot": a, "player_id": b}
                                                       for a, b in lu])))
    conn.commit()
    pid = pool.players[5].dk_player_id
    out = F.score_run(conn, "r1", slate_id="synth", n_field=2000, n_solves=10, own_unc={pid: 2.0})
    assert out["field_version"] == F.FIELD_VERSION + F.FIELD_K_TAG
    assert out["field_k"] == {"k": k_on, "own_unc": {pid: 2.0}}
    assert out["cash"]["field"]["field_k"]["k"] == k_on
    stored = conn.execute("SELECT p_cash, p_top1 FROM dfs_lineups WHERE run_id='r1' ORDER BY "
                          "rank").fetchall()
    for (pc, pt), d in zip(stored, out["per_lineup"]):
        assert pc == round(d["p_cash"], 4) and pt == round(d["p_top1"], 4)
        assert d["p_cash_lo"] <= d["p_cash"] <= d["p_cash_hi"]
        assert d["p_top1_lo"] <= d["p_top1"] <= d["p_top1_hi"]
    # an unsaved run with no late news: everybody 1.0
    assert F.run_own_uncertainty(conn, "r1", slate_id="synth") == {}


if __name__ == "__main__":
    # python tests/test_field_k.py golden PATH_TO_PRE_CHANGE_FIELD_PY
    if len(sys.argv) == 3 and sys.argv[1] == "golden":
        mod = _load_module(sys.argv[2])
        GOLDEN.write_text(json.dumps(_off_fingerprint(mod)))
        print(f"wrote {GOLDEN}")
    else:
        print(__doc__)
