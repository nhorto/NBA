"""Contest builder sim split (IMPROVEMENTS.md B1; `portfolio.PORTFOLIO_SPLIT`).

OFF (the default) must be bit-identical to the builder before the split existed: the golden file
`fixtures/portfolio_split_off_golden.json` was written by `python tests/test_portfolio_split.py`
from the pre-change portfolio.py (commit 4d7df36) and every OFF build here must reproduce it
exactly (floats compared by repr). ON: generation / selection / reporting use disjoint sim rows
and every reported number comes from the report rows only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from courtside.proj.dfs import field as F            # noqa: E402
from courtside.proj.dfs import portfolio as PF       # noqa: E402

from test_dfs_field import inputs_for, sims_for, synth_pool   # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "portfolio_split_off_golden.json"
# mode "split" (COURTSIDE_PORTFOLIO_SPLIT=1) as built before mode "gs" existed (written by
# `python tests/test_portfolio_split.py split` from that portfolio.py): adding "gs" must not move it
GOLDEN_SPLIT = Path(__file__).parent / "fixtures" / "portfolio_split_on_golden.json"
SMALL = dict(n_scenario=30, n_ownpen=20, n_noise=10, n_topmean=10, n_sims=None, n_field=2000,
             field_solves=30, threads=1, solve_limit_s=2.0)


def _slate():
    pool = synth_pool(seed=7, games=4, per_team=9)
    inp = inputs_for(pool, proj={p.dk_player_id: p.fppg for p in pool.players})
    return pool, inp, sims_for(pool, S=1500, seed=2)


def _cases(pool):
    by_val = sorted(pool.players, key=lambda p: -(p.fppg / p.salary))
    mn = by_val[9].dk_player_id
    return {
        # min-exposure (min_locked jobs), the curve above N (extended picks + curve refills),
        # soft / sharp strengths and the research baselines
        "full": (PF.Settings(n=6, gamma=5, max_exposure=0.5, min_exp={mn: 0.3},
                             strengths=True, strength_sims=600, curve_to=12,
                             curve_select_sims=700, **SMALL), True),
        # a pool that runs dry: build refills
        "refill": (PF.Settings(n=8, gamma=3, max_exposure=0.25,
                               **{**SMALL, "n_scenario": 4, "n_ownpen": 2, "n_noise": 2,
                                  "n_topmean": 2}), False),
    }


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in ("seconds", "timing")}
    if isinstance(x, (list, tuple)):
        return [_strip(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
    return x


def _fingerprint(pf) -> dict:
    return _strip({"lineups": pf.lineups, "summary": pf.summary, "exposure": pf.exposure,
                   "info": pf.info})


def _build(pool, inp, sims, s, baselines, **kw):
    return PF.build_portfolio(pool, inp, sims, F.preset_contest("gpp", 2000, 3.0), s,
                              baselines=baselines, **kw)


@pytest.fixture(scope="module")
def slate():
    return _slate()


@pytest.fixture(scope="module")
def built(slate):
    pool, inp, sims = slate
    out = {}
    for name, (s, bl) in _cases(pool).items():
        out[name] = (_build(pool, inp, sims, s, bl, split=False),
                     _build(pool, inp, sims, s, bl, split=True))
    return out


@pytest.fixture(scope="module")
def built_gs(slate):
    pool, inp, sims = slate
    return {name: _build(pool, inp, sims, s, bl, split="gs")
            for name, (s, bl) in _cases(pool).items()}


# ============================================================================== OFF

# The goldens were recorded on geekom1 (Linux / OpenBLAS). On macOS (Accelerate) float64 rounding in
# the field / Sinkhorn steps changes which lineups the solver finds, so the exact goldens only
# hold on Linux; the other tests here (disjoint rows, report == re-score on R) are platform-free.
LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="goldens recorded on Linux")


@LINUX_ONLY
def test_off_is_bit_identical_to_the_pre_change_builder(built):
    gold = json.loads(GOLDEN.read_text())
    for name, (off, _) in built.items():
        assert json.loads(json.dumps(_fingerprint(off))) == gold[name], name


def test_split_mode_parsing_and_switch():
    for v, want in (("1", "split"), ("on", "split"), ("true", "split"), ("gs", "gs"),
                    (" GS ", "gs"), ("0", "off"), ("off", "off"), ("", "off"), (None, "off"),
                    ("junk", "off"), (True, "split"), (False, "off"), ("split", "split")):
        assert PF.split_mode(v, "off") == want, v
    assert PF.SPLIT_DEFAULT == "gs"                                   # shipped 2026-10-23
    assert PF.split_mode(None, PF.SPLIT_DEFAULT) == "gs"
    old = PF.PORTFOLIO_SPLIT
    try:
        PF.set_split(True)
        assert PF.PORTFOLIO_SPLIT == "split"
        PF.set_split("gs")
        assert PF.PORTFOLIO_SPLIT == "gs"
    finally:
        PF.set_split(old)


@LINUX_ONLY
def test_mode_split_is_unchanged_by_the_gs_mode(built):
    gold = json.loads(GOLDEN_SPLIT.read_text())
    for name, (_, on) in built.items():
        assert json.loads(json.dumps(_fingerprint(on))) == gold[name], name


# ============================================================================== ON

def test_split_rows_are_disjoint_and_cover_the_sims():
    for S in (25_000, 1500, 7):
        r = PF.split_rows(S, seed=3)
        g, s_, rep = (set(r[k].tolist()) for k in ("G", "S", "R"))
        assert not (g & s_) and not (g & rep) and not (s_ & rep)
        assert g | s_ | rep == set(range(S))
        assert all(len(r[k]) >= 1 for k in ("G", "S", "R"))
    r = PF.split_rows(25_000, seed=0)
    assert (len(r["G"]), len(r["S"]), len(r["R"])) == (2_500, 11_250, 11_250)
    assert all((np.diff(r[k]) > 0).all() for k in r)                 # sorted
    assert not np.array_equal(r["G"], PF.split_rows(25_000, seed=1)["G"])


def test_generation_rows_do_not_repeat_until_g_is_used_up():
    d = PF._cycler(50, seed=4)
    first = [d() for _ in range(50)]
    assert sorted(first) == list(range(50))
    second = [d() for _ in range(50)]
    assert sorted(second) == list(range(50)) and second != first


def test_on_tags_the_version_and_records_the_split(built):
    for name, (off, on) in built.items():
        assert off.info["version"] == PF.VERSION and "split" not in off.info
        assert on.info["version"] == PF.VERSION + PF.SPLIT_TAG
        sp = on.info["split"]
        assert sp["G"] + sp["S"] + sp["R"] == off.info["n_sims"] == 1500
        assert on.summary["n_sims"] == sp["R"]
        assert on.summary["curve"]["points"][0]["n_sims"] == sp["R"]


def test_on_reported_numbers_come_from_the_report_rows_only(slate, built, monkeypatch):
    """Every reported number equals a re-score of the chosen lineups on the R rows of the
    scoring truth, and generation only ever reads G rows."""
    pool, inp, sims = slate
    s, bl = _cases(pool)["full"]
    seen = []
    real = PF.candidates

    def spy(pool_players, rules, mu, fp, *a, **k):
        seen.append(fp)
        return real(pool_players, rules, mu, fp, *a, **k)

    monkeypatch.setattr(PF, "candidates", spy)
    on = _build(pool, inp, sims, s, bl, split=True)
    rows = PF.split_rows(on.fp.shape[0], s.seed)
    assert np.array_equal(seen[0], on.fp[rows["G"]])                # generation: G rows only
    fp_r = on.fp[rows["R"]]
    pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in on.slots()]
    con = F.preset_contest("gpp", 2000, 3.0)
    ev = F.evaluate(on.field, fp_r, con, pairs, seed=s.seed, tiers=False, own_entries=s.n,
                    per_sim=True)
    lad = PF.Ladder(con)
    sc = ev.per_sim["score"]
    tot = (sc.astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
    for k, lu in enumerate(on.lineups):
        assert lu["mean"] == pytest.approx(tot[:, k].mean(), abs=1e-9)
        assert lu["p90"] == pytest.approx(np.percentile(tot[:, k], 90), abs=1e-9)
    # the score is noise-free per row; above / tied carry the dupe Poisson draws, so compare the
    # payouts on the R rows within the dupe noise and the chances exactly where no dupes apply
    pay, rank = PF.joint_payouts(lad, sc, np.rint(ev.per_sim["above"]).astype(np.int64),
                                 np.rint(ev.per_sim["tied"]).astype(np.int64))
    for k, lu in enumerate(on.lineups):          # ranks do not depend on ties: exact
        assert lu["p_cash"] == (rank[:, k] <= lad.paid).mean()
        assert lu["p_top1"] == (rank[:, k] <= lad.top1).mean()
    assert on.summary["p_any_cash"] == (rank <= lad.paid).any(1).mean()
    assert on.summary["e_payout"] == pytest.approx(pay.sum(1).mean(), rel=0.05, abs=0.05)
    assert on.summary["n_sims"] == len(rows["R"])
    # the selection rows score differently: a stat computed on S would not match
    fp_s = on.fp[rows["S"]]
    ev_s = F.evaluate(on.field, fp_s, con, pairs, seed=s.seed, tiers=False, own_entries=s.n,
                      per_sim=True)
    tot_s = (ev_s.per_sim["score"].astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
    assert on.lineups[0]["mean"] != pytest.approx(tot_s[:, 0].mean(), abs=1e-9)


def test_on_builds_a_legal_portfolio_under_the_same_caps(built):
    for name, (off, on) in built.items():
        assert len(on.lineups) == len(off.lineups) or on.summary["stopped"] == "exhausted"
        sm = on.summary
        assert sm["e_payout"] == pytest.approx(sum(x["e_payout"] for x in on.lineups))
        assert 0 <= sm["p_any_cash"] <= 1
        sets = [{x["player_id"] for x in lu} for lu in on.slots()]
        g = on.info["gamma"]
        assert all(len(a & b) <= g for i, a in enumerate(sets) for b in sets[i + 1:])
    on = built["full"][1]
    assert on.summary["strengths"]["soft"]["n_sims"] <= on.info["split"]["R"]
    assert set(on.info["baselines"]) >= {"top_individual", "top_mean"}


# ============================================================================== mode "gs"

def test_gs_rows_merge_g_and_s_and_keep_the_same_report_rows():
    for S in (25_000, 1500, 7):
        a, b = PF.split_rows(S, seed=3), PF.split_rows(S, seed=3, mode="gs")
        assert np.array_equal(b["G"], b["S"]) and np.array_equal(b["R"], a["R"])
        assert np.array_equal(b["G"], np.union1d(a["G"], a["S"]))
        assert not set(b["G"].tolist()) & set(b["R"].tolist())
    r = PF.split_rows(25_000, seed=0, mode="gs")
    assert (len(r["G"]), len(r["R"])) == (13_750, 11_250)


def test_gs_generates_and_selects_on_g_plus_s_and_reports_on_r(slate, built_gs, monkeypatch):
    pool, inp, sims = slate
    s, bl = _cases(pool)["full"]
    seen = []
    real = PF.candidates

    def spy(pool_players, rules, mu, fp, *a, **k):
        seen.append(fp)
        return real(pool_players, rules, mu, fp, *a, **k)

    monkeypatch.setattr(PF, "candidates", spy)
    on = _build(pool, inp, sims, s, bl, split="gs")
    rows = PF.split_rows(on.fp.shape[0], s.seed, "gs")
    assert np.array_equal(seen[0], on.fp[rows["G"]])
    assert on.info["version"] == PF.VERSION + PF.SPLIT_GS_TAG
    sp = on.info["split"]
    assert sp["mode"] == "gs" and sp["G"] == sp["S"] and sp["G"] + sp["R"] == 1500
    assert on.summary["n_sims"] == sp["R"]
    pairs = [[(x["slot"], x["player_id"]) for x in lu] for lu in on.slots()]
    con = F.preset_contest("gpp", 2000, 3.0)
    ev = F.evaluate(on.field, on.fp[rows["R"]], con, pairs, seed=s.seed, tiers=False,
                    own_entries=s.n, per_sim=True)
    lad = PF.Ladder(con)
    sc = ev.per_sim["score"]
    tot = (sc.astype(np.float64) - F.BIAS) / (F.UNIT * F.WUNIT)
    _, rank = PF.joint_payouts(lad, sc, np.rint(ev.per_sim["above"]).astype(np.int64),
                               np.rint(ev.per_sim["tied"]).astype(np.int64))
    for k, lu in enumerate(on.lineups):
        assert lu["mean"] == pytest.approx(tot[:, k].mean(), abs=1e-9)
        assert lu["p_cash"] == (rank[:, k] <= lad.paid).mean()
        assert lu["p_top1"] == (rank[:, k] <= lad.top1).mean()
    assert on.summary["p_any_cash"] == (rank <= lad.paid).any(1).mean()
    assert on.summary["curve"]["points"][0]["n_sims"] == sp["R"]
    assert on.summary["strengths"]["soft"]["n_sims"] <= sp["R"]


def test_gs_builds_a_legal_portfolio(built, built_gs):
    for name, on in built_gs.items():
        off = built[name][0]
        assert len(on.lineups) == len(off.lineups) or on.summary["stopped"] == "exhausted"
        assert on.summary["e_payout"] == pytest.approx(sum(x["e_payout"] for x in on.lineups))
        sets = [{x["player_id"] for x in lu} for lu in on.slots()]
        g = on.info["gamma"]
        assert all(len(a & b) <= g for i, a in enumerate(sets) for b in sets[i + 1:])


if __name__ == "__main__":
    # regenerate a golden ONLY from the builder it pins: no arg = OFF (pre-change portfolio.py),
    # "split" = mode "split" (portfolio.py before mode "gs")
    which = sys.argv[1] if len(sys.argv) > 1 else "off"
    pool, inp, sims = _slate()
    out = {name: json.loads(json.dumps(_fingerprint(_build(pool, inp, sims, s, bl,
                                                           split=which == "split"))))
           for name, (s, bl) in _cases(pool).items()}
    path = GOLDEN_SPLIT if which == "split" else GOLDEN
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", path, "from", PF.__file__)
