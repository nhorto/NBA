"""CP-SAT lineup engine, Top N and persistence (proj/dfs/optimizer.py, topn.py, optimizer_cli.py).

Synthetic pools check the constraints against brute force; the pydfs-lineup-optimizer oracle
checks equal optimal objectives on RotoGuru history (5 fixture dates always, every date in the
real ``var/courtside.db`` when at least 50 exist) and on the real archived 2026-10 draftables.
"""

from __future__ import annotations

import dataclasses
import gzip
import itertools
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from courtside import db
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import optimizer_cli
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import topn as T
from courtside.sources import dfs

FIX = Path(__file__).parent / "fixtures" / "dfs"
REAL_DB = Path(__file__).parents[1] / "var" / "courtside.db"


def J(name):
    return json.loads((FIX / name).read_text())


CLASSIC = R.Rules.from_payload(J("rules_70.json"))
SHOWDOWN = R.Rules.from_payload(J("rules_81.json"))
Q = 0.005     # the objective is scaled x100: one rounding step per player


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


# --------------------------------------------------------------------------- builders

def mk(pid, team, pos, salary, game, rules=CLASSIC, name=None, start=None):
    positions = tuple(pos.split("/"))
    slots = rules.eligible_slots(positions)
    sbs = {s: int(round(salary * rules.slot_multiplier(s))) for s in slots}
    return R.PoolPlayer(
        dk_player_id=str(pid), dk_dk_id=None, name=name or f"P{pid}", team=team,
        positions=positions, salary=salary, slots=slots, salary_by_slot=sbs,
        draftable_ids={s: f"d{pid}{s}" for s in slots}, game_id=game, competition_id=None,
        game_label=game, home=team, away=None, start_time=start, status=None,
        news_status=None, starter=None, swappable=None, fppg=None)


def synth_pool(seed=0, games=4, per_team=8, rules=CLASSIC):
    rnd = random.Random(seed)
    poss = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C", "SG/SF"]
    players, projs = [], {}
    pid = 0
    for g in range(games):
        for t in (f"T{g}A", f"T{g}B"):
            for _ in range(per_team):
                pid += 1
                sal = rnd.randrange(30, 105) * 100
                players.append(mk(pid, t, rnd.choice(poss), sal, f"G{g}", rules))
                projs[str(pid)] = round(sal / 200 + rnd.gauss(0, 5), 2)
    return players, projs


def tiny_rules(base, names, cap, **kw):
    slots = tuple(R.Slot(None, n, i + 1, R.SLOT_POSITIONS[n],
                         base.slot_multiplier(n) if n in base.slot_names else 1.0)
                  for i, n in enumerate(names))
    return dataclasses.replace(base, slots=slots, salary_cap=cap, **kw)


def brute(players, projs, rules):
    """Exhaustive best objective over legal lineups (tiny pools only)."""
    best = None
    names = rules.slot_names
    pool = [p for p in players if p.dk_player_id in projs]
    for combo in itertools.permutations(pool, len(names)):
        if any(n not in p.slots for n, p in zip(names, combo)):
            continue
        sal = sum(p.salary_for(n, rules) for n, p in zip(names, combo))
        if sal > rules.salary_cap:
            continue
        if rules.min_teams and len({p.team for p in combo}) < rules.min_teams:
            continue
        if rules.min_games and len({O.game_key(p) for p in combo}) < rules.min_games:
            continue
        v = sum(projs[p.dk_player_id] * rules.slot_multiplier(n) for n, p in zip(names, combo))
        if best is None or v > best:
            best = v
    return best


def check_legal(lu, rules, cap=None):
    assert [e.slot for e in lu.entries] == list(rules.slot_names)
    assert all(e.slot in e.player.slots for e in lu.entries)
    assert len(lu.player_ids) == rules.n_slots
    assert lu.salary == sum(e.salary for e in lu.entries) <= (cap or rules.salary_cap)
    if rules.min_teams:
        assert len({p.team for p in lu.players}) >= rules.min_teams
    if rules.min_games:
        assert len({O.game_key(p) for p in lu.players}) >= rules.min_games


def insert_slate(conn, slate_id=154575):
    """The real 2026-10-19 captures (154575 and 154827 Classic, 154825 Showdown), all rows of
    their latest capture, as stored in ``dfs_salaries``."""
    data = json.loads(gzip.decompress((FIX / "real_slates_20261019.json.gz").read_bytes()))
    db.insert_rows(conn, "dfs_salaries", data[str(slate_id)])


def fppg_projs(pool):
    return {p.dk_player_id: p.fppg for p in pool.players if p.fppg is not None}


@pytest.fixture
def classic_pool(conn):
    insert_slate(conn)
    return R.build_pool(conn, 154575, rules=CLASSIC)


@pytest.fixture
def showdown_pool(conn):
    insert_slate(conn, 154825)
    return R.build_pool(conn, 154825, rules=SHOWDOWN)


# --------------------------------------------------------------------------- constraints

def test_lineup_is_legal_and_optimal_on_synthetic_pools():
    for seed in range(5):
        players, projs = synth_pool(seed)
        lu = O.optimize(players, projs, CLASSIC)
        assert lu is not None and lu.optimal
        check_legal(lu, CLASSIC)
        assert lu.projected == pytest.approx(sum(projs[i] for i in lu.player_ids))


def test_matches_brute_force_on_tiny_rules():
    rules = tiny_rules(CLASSIC, ["PG", "G", "F", "UTIL"], 21000, min_games=2, min_teams=2)
    for seed in range(12):
        rnd = random.Random(seed)
        players = [mk(i, f"T{i % 3}", rnd.choice(["PG", "SG", "SF", "PF", "C", "PG/SG", "PF/C"]),
                      rnd.randrange(30, 90) * 100, f"G{i % 2}", rules) for i in range(9)]
        projs = {str(p.dk_player_id): round(rnd.uniform(5, 50), 2) for p in players}
        want = brute(players, projs, rules)
        lu = O.optimize(players, projs, rules)
        if want is None:
            assert lu is None
        else:
            assert lu.projected == pytest.approx(want, abs=Q)


def test_cap_binds():
    players, projs = synth_pool(1)
    full = O.optimize(players, projs, CLASSIC)
    rules = dataclasses.replace(CLASSIC, salary_cap=40000)
    lu = O.optimize(players, projs, rules)
    check_legal(lu, rules)
    assert lu.salary <= 40000 and lu.projected < full.projected
    assert O.optimize(players, projs, dataclasses.replace(CLASSIC, salary_cap=10000)) is None


def test_min_salary_option():
    players, projs = synth_pool(2)
    base = O.optimize(players, projs, CLASSIC)
    lu = O.optimize(players, projs, CLASSIC, min_salary=base.salary + 1) if base.salary < 50000 \
        else None
    if lu is not None:
        assert lu.salary > base.salary


def test_one_team_pool_is_infeasible_for_team_count():
    players = [mk(i, "AAA", p, 5000, "G1") for i, p in enumerate(
        ["PG", "SG", "SF", "PF", "C", "PG", "SF", "C", "PF", "SG"])]
    projs = {p.dk_player_id: 30.0 for p in players}
    model = O.model_for(players, projs, CLASSIC)
    assert model.solve() is None and model.status == "infeasible"
    # the same pool is fine once the rule is relaxed: the minimum is read from Rules
    assert O.optimize(players, projs, dataclasses.replace(CLASSIC, min_teams=1, min_games=1))


def test_single_game_pool_is_infeasible_for_classic_game_count():
    players = []
    for i, p in enumerate(["PG", "SG", "SF", "PF", "C", "PG", "SF", "C", "PF", "SG"]):
        players.append(mk(i, "AAA" if i % 2 else "BBB", p, 5000, "G1"))
    projs = {p.dk_player_id: 30.0 for p in players}
    assert O.optimize(players, projs, CLASSIC) is None
    lu = O.optimize(players, projs, dataclasses.replace(CLASSIC, min_games=None))
    assert lu is not None
    assert O.optimize(players, projs, dataclasses.replace(CLASSIC, min_games=1)) is not None


def test_min_games_and_teams_read_from_rules():
    players, projs = synth_pool(3)
    strict = dataclasses.replace(CLASSIC, min_games=4, min_teams=6)
    lu = O.optimize(players, projs, strict)
    assert len({O.game_key(p) for p in lu.players}) >= 4
    assert len({p.team for p in lu.players}) >= 6
    assert O.optimize(players, projs, dataclasses.replace(CLASSIC, min_teams=9)) is None


def test_locks_bans_and_slot_locks():
    players, projs = synth_pool(4)
    best = O.optimize(players, projs, CLASSIC)
    top = max(best.players, key=lambda p: p.salary)
    cheap = min((p for p in players if p.dk_player_id not in best.player_ids),
                key=lambda p: projs[p.dk_player_id])
    banned = O.optimize(players, projs, CLASSIC, bans=[top.dk_player_id])
    assert top.dk_player_id not in banned.player_ids
    assert banned.projected <= best.projected
    locked = O.optimize(players, projs, CLASSIC, locks=[cheap.dk_player_id])
    assert cheap.dk_player_id in locked.player_ids and locked.projected <= best.projected
    slot = cheap.slots[-1]                       # UTIL
    pinned = O.optimize(players, projs, CLASSIC, locks={cheap.dk_player_id: slot})
    assert dict((e.player.dk_player_id, e.slot) for e in pinned.entries)[cheap.dk_player_id] == slot
    with pytest.raises(ValueError):
        O.optimize(players, projs, CLASSIC, locks={cheap.dk_player_id: "C"} if "C" not in
                   cheap.slots else {cheap.dk_player_id: "PG"})
    with pytest.raises(ValueError):
        O.optimize(players, projs, CLASSIC, locks=[top.dk_player_id], bans=[top.dk_player_id])
    with pytest.raises(ValueError):
        O.optimize(players, projs, CLASSIC, locks=["nope"])
    # locking everyone expensive is infeasible, not an error
    stars = sorted(players, key=lambda p: -p.salary)[:8]
    assert O.optimize(players, projs, CLASSIC, locks=[p.dk_player_id for p in stars]) is None


def test_players_without_projection_are_not_picked():
    players, projs = synth_pool(5)
    best = O.optimize(players, projs, CLASSIC)
    gone = next(iter(best.player_ids))
    del projs[gone]
    assert gone not in O.optimize(players, projs, CLASSIC).player_ids


def test_deterministic():
    players, projs = synth_pool(6)
    runs = [O.optimize(players, projs, CLASSIC).slot_pairs() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# --------------------------------------------------------------------------- cash objective

def test_cash_objective_trades_mean_for_floor():
    rules = tiny_rules(CLASSIC, ["UTIL"], 10000, min_games=None, min_teams=None)
    a = mk(1, "A", "PG", 5000, "G1", rules)
    b = mk(2, "B", "PG", 5000, "G1", rules)
    projs = {"1": O.Projection(40.0, p25=20.0), "2": O.Projection(36.0, p25=32.0)}
    assert O.optimize([a, b], projs, rules, mode="mean").players[0].dk_player_id == "1"
    # 40 - .3*20 = 34 < 36 - .3*4 = 34.8
    cash = O.optimize([a, b], projs, rules, mode="cash")
    assert cash.players[0].dk_player_id == "2"
    assert cash.objective == pytest.approx(34.8) and cash.projected == pytest.approx(36.0)
    assert O.optimize([a, b], projs, rules, mode="cash", lam=0.0).players[0].dk_player_id == "1"
    # a missing p25 falls back to mu
    assert O.objective_score(O.Projection(40.0), "cash") == 40.0
    assert O.objective_score(O.as_projection({"mean": 10, "p25": 6}), "cash", 0.5) == 8.0
    assert O.as_projection(12).mean == 12.0


# --------------------------------------------------------------------------- Showdown

def test_showdown_captain_multiplier_on_synthetic_pool():
    rules = tiny_rules(SHOWDOWN, ["CPT", "UTIL", "UTIL"], 25000, min_games=None, min_teams=2)
    for seed in range(8):
        rnd = random.Random(seed)
        players = [mk(i, "A" if i % 2 else "B", "PG", rnd.randrange(30, 90) * 100, "G1", rules)
                   for i in range(7)]
        projs = {p.dk_player_id: round(rnd.uniform(5, 50), 2) for p in players}
        want = brute(players, projs, rules)
        lu = O.optimize(players, projs, rules)
        assert lu.projected == pytest.approx(want, abs=Q)
        cpt = lu.entries[0]
        assert cpt.slot == "CPT" and cpt.salary == round(1.5 * cpt.player.salary)
        assert cpt.mean == pytest.approx(1.5 * projs[cpt.player.dk_player_id])
        assert len(lu.player_ids) == 3                    # never both CPT and UTIL


def test_showdown_real_slate(showdown_pool):
    projs = fppg_projs(showdown_pool)
    lu = O.optimize(showdown_pool, projs)
    check_legal(lu, SHOWDOWN)
    cpt, *utils = lu.entries
    assert cpt.slot == "CPT" and [e.slot for e in utils] == ["UTIL"] * 5
    assert cpt.salary == cpt.player.salary_by_slot["CPT"] > cpt.player.salary
    assert lu.projected == pytest.approx(1.5 * projs[cpt.player.dk_player_id]
                                         + sum(projs[e.player.dk_player_id] for e in utils))
    row = lu.to_dk_row()
    assert len(row) == 6 and row[0] == cpt.player.draftable_ids["CPT"]
    assert lu.dk_header() == ["CPT"] + ["UTIL"] * 5
    # captain swaps are different lineups for Top N, so the 2nd best can share all 6 players
    res = T.top_n(showdown_pool, projs, n=3, min_diff=1)
    assert len(res) == 3
    ids = [r.lineup.slot_pairs() for r in res]
    assert len({frozenset(i) for i in ids}) == 3


# --------------------------------------------------------------------------- real classic slate

def test_real_classic_slate_and_dk_row(classic_pool):
    projs = fppg_projs(classic_pool)
    lu = O.optimize(classic_pool, projs)
    check_legal(lu, CLASSIC)
    row = lu.to_dk_row()
    assert lu.dk_header() == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    assert row == [e.player.draftable_ids[e.slot] for e in lu.entries]
    assert all(r.isdigit() for r in row)
    assert len({e.player.team for e in lu.entries}) >= 2
    assert lu.salary <= 50000


def test_late_swap_fixes_started_players(classic_pool):
    projs = fppg_projs(classic_pool)
    base = O.optimize(classic_pool, projs)
    starts = sorted({p.start for p in classic_pool.players if p.start})
    assert len(starts) >= 2
    now = starts[0]                                # the first game has just started
    started = classic_pool.locked_ids(now)
    in_game1 = [e for e in base.entries if e.player.dk_player_id in started]
    # a lineup built before lock; then re-solve at `now` with a projection shock
    shock = dict(projs)
    for e in base.entries:
        if e.player.dk_player_id not in started:
            shock[e.player.dk_player_id] = 0.0     # news kills the unlocked players
    swapped = O.optimize(classic_pool, shock, existing=base, now=now)
    check_legal(swapped, CLASSIC)
    pairs = dict(swapped.slot_pairs())
    for e in in_game1:                             # started players stay in their slots
        assert pairs[e.slot] == e.player.dk_player_id
    for e in swapped.entries:                      # nobody else from a started game
        if e.player.dk_player_id in started:
            assert e.player.dk_player_id in {x.player.dk_player_id for x in in_game1}
    assert any(e.player.dk_player_id not in base.player_ids for e in swapped.entries) \
        or len(in_game1) == 8
    # without an existing lineup, started players are simply banned
    fresh = O.optimize(classic_pool, projs, now=now)
    assert not (fresh.player_ids & started)
    # an explicit lock lets a started player in
    one = next(i for i in sorted(started) if i in projs)
    assert one in O.optimize(classic_pool, projs, now=now, locks=[one]).player_ids
    # before any tip nothing is fixed
    assert O.optimize(classic_pool, projs, now=starts[0].replace(year=2020)).slot_pairs() \
        == base.slot_pairs()


# --------------------------------------------------------------------------- Top N

def test_top_n_distinct_ranked_and_min_diff():
    players, projs = synth_pool(7)
    for d in (1, 2, 4):
        res = T.top_n(players, projs, CLASSIC, n=8, min_diff=d)
        assert res.complete and res.stopped is None
        objs = [r.lineup.objective for r in res]
        assert objs == sorted(objs, reverse=True)
        assert [r.rank for r in res] == list(range(1, 9))
        sets = [r.lineup.player_ids for r in res]
        for a, b in itertools.combinations(sets, 2):
            assert len(a - b) >= d
        for r in res:
            check_legal(r.lineup, CLASSIC)
        assert res.lineups[0].diff.n_different == 0
        for r in res.lineups[1:]:
            assert r.diff.n_different == len(r.diff.players_in) == len(r.diff.players_out) >= 0
            assert r.diff.d_projected == pytest.approx(r.lineup.projected
                                                       - res.lineups[0].lineup.projected)
    top1 = T.top_n(players, projs, CLASSIC, n=1).lineups[0].lineup
    assert top1.slot_pairs() == O.optimize(players, projs, CLASSIC).slot_pairs()


def test_top_n_matches_exhaustive_ranking():
    rules = tiny_rules(CLASSIC, ["PG", "G", "UTIL"], 16000, min_games=None, min_teams=None)
    rnd = random.Random(3)
    players = [mk(i, f"T{i % 2}", rnd.choice(["PG", "SG", "PG/SG"]), rnd.randrange(30, 70) * 100,
                  f"G{i % 2}", rules) for i in range(6)]
    projs = {p.dk_player_id: round(rnd.uniform(10, 40), 2) for p in players}
    sets = {}
    for combo in itertools.permutations(players, 3):
        if all(n in p.slots for n, p in zip(rules.slot_names, combo)) \
                and sum(p.salary for p in combo) <= 16000:
            k = frozenset(p.dk_player_id for p in combo)
            sets[k] = max(sets.get(k, 0), sum(projs[p.dk_player_id] for p in combo))
    ranked = sorted(sets.values(), reverse=True)
    res = T.top_n(players, projs, rules, n=50)
    assert res.stopped == "exhausted" and not res.complete       # fewer than N exist
    assert len(res) == len(sets)
    assert [r.lineup.projected for r in res] == pytest.approx(ranked, abs=Q)


def test_top_n_infeasible_pool_returns_empty():
    players = [mk(i, "AAA", "PG", 5000, "G1") for i in range(10)]
    projs = {p.dk_player_id: 10.0 for p in players}
    res = T.top_n(players, projs, CLASSIC, n=5)
    assert len(res) == 0 and res.stopped == "exhausted"


def test_top_n_exposure_caps():
    players, projs = synth_pool(8)
    free = T.top_n(players, projs, CLASSIC, n=10)
    star = max(free.exposure().items(), key=lambda kv: kv[1])
    assert star[1] > 0.5
    res = T.top_n(players, projs, CLASSIC, n=10, max_exposure=0.3)
    assert res.complete
    assert max(res.exposure().values()) <= 0.3 + 1e-9
    per = T.top_n(players, projs, CLASSIC, n=10, max_exposure={star[0]: 0.0})
    assert star[0] not in set().union(*(r.lineup.player_ids for r in per))
    capped = T.top_n(players, projs, CLASSIC, n=10, max_exposure={star[0]: 0.2})
    assert sum(star[0] in r.lineup.player_ids for r in capped) <= 2
    # a lock beats an exposure cap
    lock = T.top_n(players, projs, CLASSIC, n=4, max_exposure=0.25, locks=[star[0]])
    assert all(star[0] in r.lineup.player_ids for r in lock)


def test_top_n_with_locks_bans_min_salary():
    players, projs = synth_pool(9)
    base = T.top_n(players, projs, CLASSIC, n=3)
    ban = next(iter(base.lineups[0].lineup.player_ids))
    res = T.top_n(players, projs, CLASSIC, n=5, bans=[ban], min_salary=48000)
    for r in res:
        assert ban not in r.lineup.player_ids and 48000 <= r.lineup.salary <= 50000


def test_top_n_total_time_limit():
    players, projs = synth_pool(10)
    res = T.top_n(players, projs, CLASSIC, n=50, total_time_limit_s=1e-6)
    assert res.stopped == "timeout" and len(res) < 50


# --------------------------------------------------------------------------- persistence

def test_save_run_and_lineups(conn, classic_pool):
    projs = fppg_projs(classic_pool)
    res = T.top_n(classic_pool, projs, n=3)
    run = O.save_run(conn, slate_id="154575", kind="topn", seconds=res.seconds,
                     detail={"n": 3, "proj": "fppg"})
    assert O.save_lineups(conn, run, "topn", res.lineups) == 3
    r = conn.execute("SELECT * FROM dfs_runs WHERE run_id=?", (run,)).fetchone()
    assert r["slate_id"] == "154575" and r["kind"] == "topn" and r["ok"] == 1
    assert json.loads(r["detail"]) == {"n": 3, "proj": "fppg"}
    rows = conn.execute("SELECT * FROM dfs_lineups WHERE run_id=? ORDER BY rank", (run,)).fetchall()
    assert [x["rank"] for x in rows] == [1, 2, 3]
    first = rows[0]
    assert first["mode"] == "topn" and first["salary"] == res.lineups[0].lineup.salary
    assert first["mean"] == pytest.approx(res.lineups[0].lineup.projected)
    assert first["p10"] is None and first["p_cash"] is None and first["contest_id"] is None
    slots = json.loads(first["slots_json"])
    assert [s["slot"] for s in slots] == ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    assert all(s["draftable_id"] for s in slots)
    # plain lineups are ranked by position
    O.save_lineups(conn, run, "cash", [res.lineups[0].lineup])
    assert conn.execute("SELECT rank FROM dfs_lineups WHERE mode='cash'").fetchone()[0] == 1


# --------------------------------------------------------------------------- history pools

def load_hist_fixture(conn, days=None):
    data = json.loads(gzip.decompress((FIX / "hist_slates.json.gz").read_bytes()))
    rows = []
    for day, rs in data.items():
        if days and day not in days:
            continue
        for rg, name, pos, team, opp, home, starter, sal, pts in rs:
            rows.append({"game_date": day, "rg_id": rg, "name": name, "pos": pos, "team": team,
                         "opp": opp, "home": home, "starter": starter, "salary": sal,
                         "dk_pts": pts, "on_slate": 1})
    db.insert_rows(conn, "dfs_hist_salaries", rows)
    return sorted(data)


def test_hist_pool_eligibility_and_projections(conn):
    days = load_hist_fixture(conn)
    assert O.hist_dates(conn, min_players=1) == days
    pool = O.hist_pool(conn, days[0], CLASSIC)
    assert pool.rules is CLASSIC and len(pool.games) >= 2
    pg = next(p for p in pool.players if p.positions == ("PG",))
    assert pg.slots == ("PG", "G", "UTIL")
    c = next(p for p in pool.players if p.positions == ("C",))
    assert c.slots == ("C", "UTIL")
    dual = next(p for d in days for p in O.hist_pool(conn, d, CLASSIC).players
                if len(p.positions) == 2)
    assert len(dual.slots) >= 4 and {"G", "F", "UTIL"} & set(dual.slots)
    act = O.hist_projections(conn, pool, "actual")
    sal = O.hist_projections(conn, pool, "salary")
    assert sal[pg.dk_player_id].mean == pytest.approx(5 * pg.salary / 1000, abs=0.01)
    best_hind = O.optimize(pool, act)
    best_sal = O.optimize(pool, sal)
    check_legal(best_hind, CLASSIC)
    realised = sum(act[i].mean for i in best_sal.player_ids if i in act)
    assert best_hind.projected >= realised - Q                   # hindsight is an upper bound
    assert best_hind.projected > 250                             # a real DK slate
    with pytest.raises(ValueError):
        O.hist_projections(conn, pool, "nope")


# --------------------------------------------------------------------------- pydfs oracle

pydfs = pytest.importorskip("pydfs_lineup_optimizer")


def pydfs_best(players, projs, rules):
    """Best objective pydfs finds on the same pool, projections and eligibility."""
    from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
    from pydfs_lineup_optimizer.player import GameInfo
    opt = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
    assert opt.settings.budget == rules.salary_cap
    assert opt.settings.max_from_one_team is None            # no hard-coded team cap
    assert opt.settings.min_games == rules.min_games
    pp, infos = [], {}
    for p in players:
        if p.dk_player_id not in projs:
            continue
        pp.append(Player(p.dk_player_id, p.name, "x", list(p.positions), p.team, p.salary,
                         projs[p.dk_player_id],
                         game_info=infos.setdefault(O.game_key(p), GameInfo(
                             p.home or p.team, p.away or "?", None))))
    opt.player_pool.load_players(pp)
    opt.set_total_teams(min_teams=rules.min_teams)    # after the pool: pydfs checks it
    return next(iter(opt.optimize(1))).fantasy_points_projection


def mean_of(projs):
    return {k: (v.mean if isinstance(v, O.Projection) else float(v)) for k, v in projs.items()}


def agree(players, projs, rules, tag=""):
    lu = O.optimize(players, projs, rules)
    want = pydfs_best(players, mean_of(projs), rules)
    assert lu.projected == pytest.approx(want, abs=0.01), tag
    return lu


def test_pydfs_oracle_on_fixture_slates(conn):
    days = load_hist_fixture(conn)
    n = 0
    for day in days:
        pool = O.hist_pool(conn, day, CLASSIC)
        for kind in ("actual", "salary"):
            projs = O.hist_projections(conn, pool, kind)
            agree(pool.players, projs, CLASSIC, f"{day} {kind}")
            n += 1
    assert n == 10


def test_pydfs_oracle_on_real_draftables(conn, classic_pool):
    insert_slate(conn, 154827)
    other = R.build_pool(conn, 154827, rules=CLASSIC)
    n = 0
    for pool in (classic_pool, other):
        projs = fppg_projs(pool)
        agree(pool.players, projs, CLASSIC, f"{pool.slate_id} fppg")
        rnd = random.Random(0)
        for k in range(3):                                # more "slates": noisy variants
            noisy = {i: round(max(0.0, v * rnd.uniform(0.7, 1.3)), 2) for i, v in projs.items()}
            agree(pool.players, noisy, CLASSIC, f"{pool.slate_id} noisy {k}")
            n += 1
        n += 1
    assert n == 8


def test_pydfs_oracle_on_every_rotoguru_date_in_the_real_db():
    if not REAL_DB.exists():
        pytest.skip("no var/courtside.db")
    ro = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        dates = O.hist_dates(ro)
    except sqlite3.OperationalError:
        pytest.skip("no dfs_hist_salaries table")
    if len(dates) < 50:
        pytest.skip(f"only {len(dates)} RotoGuru slates")
    rnd = random.Random(20261019)
    sample = sorted(rnd.sample(dates, 50))
    n = 0
    for day in sample:
        pool = O.hist_pool(ro, day, CLASSIC)
        projs = O.hist_projections(ro, pool, "actual" if n % 2 == 0 else "salary")
        agree(pool.players, projs, CLASSIC, day)
        n += 1
    assert n == 50
    ro.close()


# --------------------------------------------------------------------------- runtime

def test_top_20_on_a_full_main_slate_is_fast(classic_pool):
    assert 100 <= len(classic_pool.players) <= 160       # the real 125-player main slate
    projs = fppg_projs(classic_pool)
    t = time.perf_counter()
    res = T.top_n(classic_pool, projs, n=20, min_diff=1)
    dt = time.perf_counter() - t
    assert res.complete and len(res) == 20
    print(f"top-20 on {len(classic_pool.players)} players: {dt:.2f}s")
    assert dt <= 5.0


# --------------------------------------------------------------------------- CLI

def test_cli_build_topn_on_a_stored_slate(conn, capsys, monkeypatch):
    insert_slate(conn)
    monkeypatch.setattr(optimizer_cli, "_conn", lambda: conn)
    monkeypatch.setattr(R, "load_rules", lambda *a, **k: CLASSIC)
    import argparse
    p = argparse.ArgumentParser()
    handlers = optimizer_cli.add_commands(p.add_subparsers(dest="c"))
    args = p.parse_args(["build", "--slate", "154575", "--mode", "topn", "--n", "3",
                         "--proj", "fppg", "--save", "--now", "2026-10-20T12:00:00Z"])
    assert handlers["build"](args) == 0           # (pinned before the slate's 19:00Z lock)
    out = capsys.readouterr().out
    assert "#1" in out and "#3" in out and "UTIL" in out
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups").fetchone()[0] == 3
    # lock / ban by name
    name = R.build_pool(conn, 154575, rules=CLASSIC).players[40].name
    args = p.parse_args(["build", "--slate", "154575", "--mode", "cash", "--lock", name])
    assert handlers["build"](args) == 0
    assert name in capsys.readouterr().out


def test_cli_build_historical_date(conn, capsys, monkeypatch):
    days = load_hist_fixture(conn)
    monkeypatch.setattr(optimizer_cli, "_conn", lambda: conn)
    monkeypatch.setattr(O, "classic_rules", lambda *a, **k: CLASSIC)
    import argparse
    p = argparse.ArgumentParser()
    handlers = optimizer_cli.add_commands(p.add_subparsers(dest="c"))
    args = p.parse_args(["build", "--date", days[0], "--proj", "actual", "--mode", "topn",
                         "--n", "2"])
    assert handlers["build"](args) == 0
    assert "HINDSIGHT" in capsys.readouterr().out


def test_save_run_completes_a_row_the_pipeline_wrote():
    from courtside import db
    from courtside.proj.dfs import optimizer as O
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO dfs_runs (run_id, slate_id, kind, created_at, model_version, "
                 "sim_path, path) VALUES ('r1', '1', 'lock45', 'x', 'pathA-v0', 'sims.npz', 'A')")
    O.save_run(conn, slate_id="1", kind="lock45", model_version=None, run_id="r1", seconds=3.0)
    row = conn.execute("SELECT model_version, sim_path, path, seconds, ok FROM dfs_runs").fetchone()
    assert tuple(row) == ("pathA-v0", "sims.npz", "A", 3.0, 1)
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == 1
