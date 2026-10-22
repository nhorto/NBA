"""Ownership proxy and field proxy (proj/dfs/field.py, T2.11).

Synthetic pools and the archived DK rules fixtures: Σ ownership (Classic 800 %, Showdown 600 %
with the CPT split), field lineup validity (slots, cap, games, teams, distinct players), salary
use, determinism by seed, the payout ladder on a hand-built contest, P(cash) monotone in lineup
strength, the dupe estimate on a tiny pool, persistence and `fit_from_uploads` gating.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import rules as R

FIX = Path(__file__).parent / "fixtures" / "dfs"
CLASSIC = R.Rules.from_payload(json.loads((FIX / "rules_70.json").read_text()))
SHOWDOWN = R.Rules.from_payload(json.loads((FIX / "rules_81.json").read_text()))


def mk(pid, team, opp, pos, salary, game, rules=CLASSIC, fppg=None, status=None, starter=None):
    positions = tuple(pos.split("/"))
    slots = rules.eligible_slots(positions)
    sbs = {s: int(round(salary * rules.slot_multiplier(s))) for s in slots}
    return R.PoolPlayer(
        dk_player_id=str(pid), dk_dk_id=None, name=f"P{pid}", team=team, positions=positions,
        salary=salary, slots=slots, salary_by_slot=sbs,
        draftable_ids={s: f"d{pid}{s}" for s in slots},
        game_id=game, competition_id=None, game_label=game, home=team, away=opp, start_time=None,
        status=status, news_status=None, starter=starter, swappable=None, fppg=fppg)


def synth_pool(seed=0, games=4, per_team=9, rules=CLASSIC):
    rnd = random.Random(seed)
    poss = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C", "SG/SF"]
    players = []
    pid = 0
    for g in range(games):
        a, b = f"T{g}A", f"T{g}B"
        for t, o in ((a, b), (b, a)):
            for k in range(per_team):
                pid += 1
                sal = rnd.randrange(30, 110) * 100
                fp = max(4.0, sal / 1000 * 5 + rnd.gauss(0, 5))
                players.append(mk(pid, t, o, poss[k % len(poss)], sal, f"G{g}", rules,
                                  round(fp, 1)))
    return R.SlatePool("synth", rules.game_type_id, "classic", "2026-01-01", rules, tuple(players))


def inputs_for(pool, proj=None):
    return F.pool_inputs(pool, proj=proj)


def sims_for(pool, S=2000, seed=0, sd=0.3):
    rng = np.random.default_rng(seed)
    mu = np.array([p.fppg or 0 for p in pool.players])
    fp = np.maximum(0, rng.normal(mu, sd * mu + 2, (S, len(mu)))).astype(np.float16)
    return {"fp": fp, "dk_id": np.array([p.dk_player_id for p in pool.players])}


def check_lineup(row, field: F.Field, pool, rules):
    by_id = pool.by_id()
    players = [by_id[field.pids[i]] for i in row]
    assert len(set(row)) == len(row), "a player twice"
    sal = 0
    for slot, p in zip(field.slots, players):
        assert slot in p.slots, f"{p.name} not eligible for {slot}"
        sal += p.salary_for(slot, rules)
    assert sal <= rules.salary_cap
    if rules.min_games:
        assert len({F._game(p) for p in players}) >= rules.min_games
    if rules.min_teams:
        assert len({p.team for p in players}) >= rules.min_teams
    return sal


# --------------------------------------------------------------------------- ownership

@pytest.mark.parametrize("arch", ["cash", "small_gpp", "large_gpp"])
def test_classic_ownership_sums_to_800(arch):
    pool = synth_pool(1)
    op = F.ownership_proxy(inputs_for(pool), pool, archetype=arch)
    assert op.own.sum() == pytest.approx(8.0, abs=1e-6)
    assert np.allclose(op.by_slot.sum(0), op.slot_counts, atol=1e-6)
    assert op.own.max() <= F.ROW_CAP + 1e-9 and op.own.min() >= 0
    assert "proxy" == op.label and op.version.startswith(F.MODEL)
    # expected spend matches the field's salary use
    assert op.spend == pytest.approx(F.SPEND * CLASSIC.salary_cap, rel=0.01)
    assert set(op.bands) <= {"Low", "Med", "High"}


def test_cash_archetype_is_chalkier():
    pool = synth_pool(2)
    inp = inputs_for(pool)
    cash = F.ownership_proxy(inp, pool, archetype="cash").own
    big = F.ownership_proxy(inp, pool, archetype="large_gpp").own
    top = lambda o: np.sort(o)[-5:].sum()                  # noqa: E731
    assert top(cash) > top(big)


def test_showdown_ownership_600_with_captain_split():
    players = []
    for k in range(12):
        t, o = ("AAA", "BBB") if k % 2 else ("BBB", "AAA")
        players.append(mk(k + 1, t, o, ["PG", "SG", "SF", "PF", "C"][k % 5],
                          random.Random(k).randrange(20, 120) * 100, "G1", SHOWDOWN,
                          fppg=10 + 3 * k))
    pool = R.SlatePool("sd", 81, "showdown", "x", SHOWDOWN, tuple(players))
    op = F.ownership_proxy(inputs_for(pool), pool, archetype="small_gpp")
    assert op.own.sum() == pytest.approx(6.0, abs=1e-6)
    cpt = op.captain()
    assert sum(cpt.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(cpt[p] <= o + 1e-9 for p, o in op.as_dict().items())
    f = F.build_field(pool, inputs_for(pool), F.preset_contest("gpp", 2000), n=3000,
                      n_solves=10, seed=1, ownership=op)
    for row in f.lineups[:300]:
        check_lineup(row, f, pool, SHOWDOWN)


def test_out_players_get_zero_and_q_is_faded():
    pool = synth_pool(3)
    base = pool.players[0]
    import dataclasses
    out = dataclasses.replace(base, status="OUT")
    pl = (out,) + pool.players[1:]
    pool_o = R.SlatePool("x", 70, "classic", "x", CLASSIC, pl)
    op = F.ownership_proxy(inputs_for(pool_o), pool_o)
    assert op.as_dict()[base.dk_player_id] == 0
    q = dataclasses.replace(base, status="Q")
    pool_q = R.SlatePool("x", 70, "classic", "x", CLASSIC, (q,) + pool.players[1:])
    a = F.ownership_proxy(inputs_for(pool), pool).as_dict()[base.dk_player_id]
    b = F.ownership_proxy(inputs_for(pool_q), pool_q).as_dict()[base.dk_player_id]
    assert b < a


def test_value_drives_chalk():
    pool = synth_pool(4)
    inp = inputs_for(pool)
    op = F.ownership_proxy(inp, pool)
    t = op.table()
    hi = t.head(10)["pub_value"].mean()
    lo = t[t["own"] > 0].tail(10)["pub_value"].mean()
    assert hi > lo


# --------------------------------------------------------------------------- field

@pytest.fixture(scope="module")
def built():
    pool = synth_pool(5, games=5)
    inp = inputs_for(pool)
    con = F.preset_contest("gpp", 20_000, 5)
    f = F.build_field(pool, inp, con, n=6000, n_solves=40, seed=7)
    return pool, inp, con, f


def test_field_lineups_valid(built):
    pool, inp, con, f = built
    assert f.n == 6000 and f.label == "proxy"
    mix = F.ARCHETYPES[con.archetype]["mix"]          # the archetype's default (medium) share
    assert f.n_solves > 0 and f.opt_counts.sum() == round(mix * 6000)
    for row in f.lineups:
        check_lineup(row, f, pool, CLASSIC)


def test_salary_use(built):
    _, _, _, f = built
    su = f.salary_use()
    assert su["share_ge_98"] >= 0.7             # most of the field spends >= 98 % of the cap
    assert su["max"] <= CLASSIC.salary_cap
    assert su["min"] >= 0.9 * CLASSIC.salary_cap


def test_field_reproduces_proxy_ownership(built):
    _, _, _, f = built
    real = f.realized_ownership()
    assert real.sum() == pytest.approx(8.0)
    assert f.summary()["own_target_mae_pp"] < 3.0


def test_determinism_by_seed():
    pool = synth_pool(6)
    inp = inputs_for(pool)
    con = F.preset_contest("double_up", 300, 5)
    a = F.build_field(pool, inp, con, n=3000, n_solves=20, seed=3)
    b = F.build_field(pool, inp, con, n=3000, n_solves=20, seed=3)
    c = F.build_field(pool, inp, con, n=3000, n_solves=20, seed=4)
    assert np.array_equal(a.lineups, b.lineups) and np.array_equal(a.counts, b.counts)
    assert not (len(a.lineups) == len(c.lineups) and np.array_equal(a.lineups, c.lineups))
    sims = sims_for(pool, 500)
    X, _ = F.align_sims(sims, a.pids)
    lus = [[(s, a.pids[i]) for s, i in zip(a.slots, a.lineups[k])] for k in range(5)]
    e1 = F.evaluate(a, X, con, lus, seed=9)
    e2 = F.evaluate(b, X, con, lus, seed=9)
    assert np.array_equal(e1.cash_line, e2.cash_line)
    assert [m["e_payout"] for m in e1.lineups] == [m["e_payout"] for m in e2.lineups]


# --------------------------------------------------------------------------- scoring

def tiny_rules(names, cap, min_games=None, min_teams=None):
    slots = tuple(R.Slot(None, n, i + 1, R.SLOT_POSITIONS[n]) for i, n in enumerate(names))
    return R.Rules(None, "tiny", cap, min_games, min_teams, slots)


def hand_field(pool, rules, lineups, field_size):
    """A Field built by hand (no sampling): `lineups` are lists of player ids."""
    inp = inputs_for(pool)
    op = F.ownership_proxy(inp, pool, rules)
    idx = {p: i for i, p in enumerate(op.pids)}
    L = np.array([[idx[p] for p in lu] for lu in lineups])
    canon = np.sort(L, axis=1)
    uniq, first, cnt = np.unique(canon, axis=0, return_index=True, return_counts=True)
    _, _, _, S = F._eligibility(op.inputs, pool.by_id(), rules)
    sal = np.array([sum(pool.by_id()[op.pids[i]].salary for i in r) for r in L[first]])
    return F.Field(pids=op.pids, slots=rules.slot_names, lineups=L[first], counts=cnt,
                   opt_counts=np.zeros(len(cnt), int), salary=sal, field_size=field_size,
                   mix=0.0, archetype="small_gpp", ownership=op, n_solves=0, opt_keys={},
                   dupe_c=float("nan"), cap=rules.salary_cap, slot_salary=S)


def test_payout_ladder_expected_payout_exact():
    rules = tiny_rules(("UTIL",), 100_000)
    players = [mk(i, "A" if i % 2 else "B", "B" if i % 2 else "A", "PG", 5000, "G1", rules,
                  fppg=10.0) for i in range(1, 12)]
    pool = R.SlatePool("t", None, "classic", "x", rules, tuple(players))
    # the other 9 entries: players 1..9 (one each); candidates: player 10, 11 and a dupe of 5
    f = hand_field(pool, rules, [[str(i)] for i in range(1, 10)], field_size=10)
    pts = {str(i): float(i * 10) for i in range(1, 10)}
    pts["10"], pts["11"] = 55.0, 95.0                      # 55 beats 5 others -> rank 5
    fp = np.array([[pts[p] for p in f.pids]], np.float32)
    con = F.Contest("c", "synthetic", 10.0, 10,
                    [{"min_pos": 1, "max_pos": 1, "payout": 50.0},
                     {"min_pos": 2, "max_pos": 3, "payout": 20.0},
                     {"min_pos": 4, "max_pos": 5, "payout": 10.0}], "small_gpp", "preset")
    ev = F.evaluate(f, fp, con, [[("UTIL", "10")], [("UTIL", "11")], [("UTIL", "5")]])
    assert ev.info["mode"] == "exact"
    a, b, c = ev.lineups
    assert a["e_payout"] == pytest.approx(10.0)          # rank 5
    assert b["e_payout"] == pytest.approx(50.0)          # rank 1
    # score 50 ties entry "5": others above = 4 (60..90) -> ranks 5 and 6 split (10 + 0) / 2
    assert c["e_payout"] == pytest.approx(5.0)
    assert ev.cash_line[0] == pytest.approx(50.0)        # 5th best of the others: 90,80,70,60,50
    assert a["p_cash"] == 1.0 and c["p_cash"] == 1.0
    assert b["e_roi"] == pytest.approx(4.0)


def test_cash_line_is_paid_rank_score_and_pcash_definition():
    pool = synth_pool(8)
    inp = inputs_for(pool)
    con = F.preset_contest("double_up", 1001, 1)          # pays 444
    f = F.build_field(pool, inp, con, n=1000, n_solves=10, seed=2)
    assert f.n == 1000                                    # exact mode: sample = other entries
    sims = sims_for(pool, 300, seed=1)
    X, _ = F.align_sims(sims, f.pids)
    lus = [[(s, f.pids[i]) for s, i in zip(f.slots, f.lineups[k])] for k in range(3)]
    ev = F.evaluate(f, X, con, lus)
    # recompute the cash line by brute force: paid-th best of the 1000 other entries
    Au = np.zeros((len(f.lineups), len(f.pids)))
    for j in range(f.lineups.shape[1]):
        np.add.at(Au, (np.arange(len(f.lineups)), f.lineups[:, j]), 1.0)
    su = (np.rint(X.astype(np.float64) * F.UNIT) / F.UNIT) @ Au.T          # (S, U)
    scores = np.repeat(su, f.counts, axis=1)                                 # (S, 1000)
    brute = -np.sort(-scores, axis=1)[:, con.paid - 1]
    assert np.allclose(ev.cash_line, brute, atol=1e-6)
    for k, m in enumerate(ev.lineups):
        assert m["p_cash"] == pytest.approx(np.mean(su[:, k] >= brute))


def test_pcash_monotone_in_lineup_strength():
    pool = synth_pool(9)
    inp = inputs_for(pool)
    con = F.preset_contest("double_up", 200, 5)
    f = F.build_field(pool, inp, con, n=3000, n_solves=20, seed=5)
    sims = sims_for(pool, 3000, seed=2)
    X, _ = F.align_sims(sims, f.pids)
    base = [(s, f.pids[i]) for s, i in zip(f.slots, f.lineups[0])]
    # strengthen the UTIL player in steps: same lineup, his points + k in every sim
    util = next(i for i, (s, _) in enumerate(base) if s == "UTIL")
    j = f.pids.index(base[util][1])
    evs = []
    for k in range(6):
        Xk = X.copy()
        Xk[:, j] += 4.0 * k
        evs.append(F.evaluate(f, Xk, con, [base], seed=11).lineups[0])
    pc = [e["p_cash"] for e in evs]
    ep = [e["e_payout"] for e in evs]
    assert all(b >= a for a, b in zip(pc, pc[1:])) and pc[-1] > pc[0]
    assert all(b >= a - 1e-9 for a, b in zip(ep, ep[1:]))
    assert all(b["p_top1"] >= a["p_top1"] for a, b in zip(evs, evs[1:]))


def test_dupe_estimate_on_tiny_pool():
    rules = tiny_rules(("UTIL", "UTIL", "UTIL"), 30_000)
    rnd = random.Random(0)
    players = [mk(i, "A" if i % 2 else "B", "B" if i % 2 else "A", "PG", 10_000, "G1", rules,
                  fppg=10 + 2 * i + rnd.random()) for i in range(1, 15)]
    pool = R.SlatePool("t", None, "classic", "x", rules, tuple(players))
    inp = inputs_for(pool)
    con = F.preset_contest("gpp", 50_000)
    f = F.build_field(pool, inp, con, n=4000, mix=0.0, seed=1)
    big = F.build_field(pool, inp, con, n=200_000, mix=0.0, seed=2)
    freq = {tuple(np.sort(r)): c / big.n for r, c in zip(big.lineups, big.counts)}
    combos = [c for c in itertools.combinations(range(len(f.pids)), 3)
              if f.ownership.own[list(c)].min() > 0]
    rows = np.array(combos)
    est = F.dupes(f, rows) / (con.field_size - 1)
    emp = np.array([freq.get(tuple(c), 0.0) for c in combos])
    assert math.isfinite(f.dupe_c)
    # calibrated product form: total mass and ranking agree with a 50x bigger independent sample
    assert est.sum() == pytest.approx(1.0, rel=0.35)
    keep = emp > 0
    r = np.corrcoef(np.log(est[keep]), np.log(emp[keep]))[0, 1]
    assert r > 0.8
    common = np.argsort(-emp)[:10]
    assert np.allclose(est[common], emp[common], rtol=0.35)
    assert F.dupe_band(0.1) == "Low" and F.dupe_band(1) == "Med" and F.dupe_band(3) == "High"


def test_optimizer_dupes_exact_frequency():
    pool = synth_pool(10)
    con = F.preset_contest("gpp", 10_001)
    f = F.build_field(pool, inputs_for(pool), con, n=5000, n_solves=30, seed=3)
    k, q = max(f.opt_keys.items(), key=lambda kv: kv[1])
    row = np.array(k)
    d = F.dupes(f, row[None, :])[0]
    assert d >= (con.field_size - 1) * f.mix * q - 1e-6


def test_undominated_pruning_is_exact():
    rnd = np.random.default_rng(0)
    n = 40
    sal = rnd.integers(30, 100, n) * 100.0
    proj = sal / 200 + rnd.normal(0, 4, n)
    masks = rnd.integers(1, 8, n)
    keep = F.undominated(proj, sal, masks, 3)
    for i in np.flatnonzero(~keep):
        dom = [j for j in range(n) if j != i and sal[j] <= sal[i] and proj[j] >= proj[i]
               and (masks[j] & masks[i]) == masks[i]]
        assert len(dom) >= 3


# --------------------------------------------------------------------------- contests / io

def test_classify_contest():
    du = {"attrs": {"lobby": {"IsDoubleUp": "true"}}, "field_size": 124,
          "payouts": [{"min_pos": 1, "max_pos": 54, "payout": 20}]}
    assert F.classify_contest(du) == "cash"
    gpp = {"attrs": {}, "field_size": 35294,
           "payouts": [{"min_pos": 1, "max_pos": 1, "payout": 9},
                       {"min_pos": 2, "max_pos": 9, "payout": 1}]}
    assert F.classify_contest(gpp) == "large_gpp"
    assert F.classify_contest({**gpp, "field_size": 444}) == "small_gpp"
    p = F.preset_contest("double_up", 1000)
    assert p.paid == 444 and p.archetype == "cash" and p.source == "preset"
    g = F.preset_contest("gpp", 10_000, 1.0)
    assert sum((t["max_pos"] - t["min_pos"] + 1) * t["payout"] for t in g.payouts) == \
        pytest.approx(8600, rel=0.01)


def test_align_sims_handles_missing_and_nan_ids():
    sims = {"fp": np.arange(12, dtype=np.float16).reshape(3, 4),
            "dk_id": np.array(["a", "nan", "b", "c"])}
    X, info = F.align_sims(sims, ["c", "a", "zz"])
    assert X[:, 0].tolist() == [3, 7, 11] and X[:, 1].tolist() == [0, 4, 8]
    assert X[:, 2].tolist() == [0, 0, 0] and info["missing"] == ["zz"]


def test_store_ownership_and_fit_gate(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    pool = synth_pool(11)
    op = F.ownership_proxy(inputs_for(pool), pool)
    n = F.store_ownership(conn, "run1", op)
    rows = conn.execute("SELECT player_id, own, band, version FROM dfs_ownership_proxy "
                        "WHERE run_id='run1'").fetchall()
    assert len(rows) == n and sum(r["own"] for r in rows) == pytest.approx(800, abs=0.5)
    assert all(r["version"].startswith(F.HANDSET_VERSION) for r in rows)
    rep = F.fit_from_uploads(conn)                          # nothing uploaded
    assert rep["status"] == "insufficient" and rep["n_slates"] == 0
    assert F.load_params() == (F.DEFAULT_PARAMS, F.HANDSET_VERSION)


def test_fit_from_uploads_recovers_params(tmp_path, monkeypatch):
    """With >= 10 synthetic 'uploaded' slates generated from known weights, the refit lowers the
    MAE vs the hand-set proxy, reports rank correlation, and saves a versioned parameter file."""
    from courtside.proj.dfs import standings as ST
    truth = {**F.DEFAULT_PARAMS, "b_value": 1.8, "b_proj": 0.2}
    pools, rows = {}, []
    for s in range(10):
        pool = synth_pool(100 + s, games=3, per_team=5)
        inp = inputs_for(pool)
        pools[str(s)] = (pool, inp)
        op = F.ownership_proxy(inp, pool, archetype="large_gpp", params=truth, version="truth")
        for pid, o in zip(op.pids, op.own):
            rows.append({"contest_id": f"c{s}", "slate_id": str(s), "run_id": None, "dk_id": pid,
                         "pct_drafted": o * 100})
    monkeypatch.setattr(ST, "ownership_dataset", lambda conn: pd.DataFrame(rows))
    monkeypatch.setattr(F, "slate_inputs",
                        lambda conn, sid, run_id=None, **kw: (*pools[sid], {}))
    monkeypatch.setattr(F, "load_contest", lambda conn, cid: None)
    rep = F.fit_from_uploads(None, maxiter=25, loso=1)
    assert rep["status"] == "fitted" and rep["n_slates"] == 10
    assert rep["fitted_in_sample"]["mae_pp"] < rep["handset"]["mae_pp"]
    assert rep["fitted_loso"]["mae_pp"] < rep["handset"]["mae_pp"]
    assert rep["fitted_in_sample"]["spearman"] > 0.9
    params, version = F.load_params()
    assert version == rep["version"] and version.startswith(f"{F.MODEL}-fit-")
    assert F.calibrated()
