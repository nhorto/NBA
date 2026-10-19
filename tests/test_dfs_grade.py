"""Nightly DFS grading (proj/dfs/grade.py, T3.0): actual DK incl. DNP, PIT/coverage/CRPS,
hindsight >= every lineup, random percentile, idempotent re-grade, non-final games skipped,
snapshot windows and SEs, real vs simulated entries, real cash lines only from uploads.
Synthetic slates only: no network, no real DB."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from courtside import collect, db
from courtside.proj import metrics as MT
from courtside.proj import store as SO
from courtside.proj.dfs import grade as G
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import topn as T

FIX = Path(__file__).parent / "fixtures" / "dfs"
RULES70 = json.loads((FIX / "rules_70.json").read_text())
RULES81 = json.loads((FIX / "rules_81.json").read_text())
DAY = date(2026, 10, 20)
POS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C"]
TEAMS = [("BOS", "DET"), ("PHI", "NYK"), ("OKC", "SAS")]      # (away, home)
RULES = R.Rules.from_payload(RULES70)


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


# --------------------------------------------------------------------------- the synthetic world

def dk_id(gi, side, k):
    return 1000 * (gi + 1) + 100 * side + k + 1


def nba_id(gi, side, k):
    return 90000 + dk_id(gi, side, k)


def stat_line(k, side):
    return dict(pts=8 + 3 * k + side, reb=2 + k % 4, ast=1 + k % 3, stl=k % 2, blk=0,
                fg3m=k % 3, tov=k % 3)


def game_id(d, gi):
    return f"0022{d:02d}{gi + 1:04d}"


def build_day(conn, d=0, *, dnp=((0, 0, 7),), final=True, box=True, run=True, qwidth=12.0,
              sims=None):
    """One Classic slate `S<d>` on DAY+d: 3 games x 2 teams x 8 players, a captured pool with
    DK ids, ids, final games with box scores (the players in `dnp` have no box row), one
    lock10 run with projections + stats and cash / Top-3 lineups. Returns the slate id."""
    day = DAY + timedelta(days=d)
    sid = f"S{d}"
    gids = [game_id(d, gi) for gi in range(3)]
    conn.execute("INSERT OR REPLACE INTO dfs_slates (slate_id, site, nba_day, game_type_id, tag, "
                 "game_ids_json, lock_utc, rules_json, captured_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (sid, "dk", day.isoformat(), 70, "main", json.dumps(gids),
                  f"{day.isoformat()}T23:00:00Z", json.dumps(RULES70), f"{day.isoformat()}T12:00:00Z"))
    for gi, (away, home) in enumerate(TEAMS):
        db.upsert_game(conn, game_id=gids[gi], season="2026-27", nba_day=day.isoformat(),
                       tipoff_utc=f"{day.isoformat()}T23:00:00Z", home=home, away=away,
                       status="final" if final else "scheduled", home_score=100, away_score=99)
        for side, team in enumerate((away, home)):
            for k, pos in enumerate(POS):
                did, nid = dk_id(gi, side, k), nba_id(gi, side, k)
                salary = 3500 + 700 * ((k * 3 + gi + side) % 8)
                fppg = 14.0 + 3.2 * ((k * 5 + gi * 2 + side) % 9) + salary / 2000
                for slot in RULES.eligible_slots(pos):
                    conn.execute(
                        "INSERT INTO dfs_salaries (captured_at, site, slate_id, game_type, "
                        "game_type_id, draftable_id, player_name, site_player_id, team, "
                        "positions, roster_slot, salary, fppg, disabled, game_id, home, away, "
                        "start_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"{day.isoformat()}T12:00:00Z", "dk", sid, "classic", 70,
                         f"{did}-{slot}", f"{team} P{k}", str(did), team, pos, slot, salary,
                         fppg, 0, gids[gi], home, away, f"{day.isoformat()}T23:00:00Z"))
                conn.execute("INSERT OR REPLACE INTO player_ids (source, source_id, player_id, "
                             "name, team, first_seen, last_seen, how, confidence) VALUES "
                             "('dk',?,?,?,?,?,?,?,?)", (str(did), nid, f"{team} P{k}", team,
                                                        "2026-10-01", "2026-10-20", "exact", 1.0))
                if box and (gi, side, k) not in dnp:
                    st = stat_line(k, side)
                    conn.execute(
                        "INSERT OR REPLACE INTO player_games (game_id, player_id, player_name, "
                        "team, season, season_type, game_date, min, pts, fg3m, reb, ast, stl, "
                        "blk, tov) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (gids[gi], nid, f"{team} P{k}", team, "2026-27", "Regular Season",
                         day.isoformat(), 20.0 + 2 * k, st["pts"], st["fg3m"], st["reb"],
                         st["ast"], st["stl"], st["blk"], st["tov"]))
    conn.commit()
    if not run:
        return sid
    O.save_run(conn, slate_id=sid, kind="lock10", run_id=f"run{d}",
               created_at=f"{day.isoformat()}T22:50:00Z", model_version="synth")
    mean_of = {}
    for gi, (away, home) in enumerate(TEAMS):
        for side, team in enumerate((away, home)):
            for k in range(len(POS)):
                did, nid = dk_id(gi, side, k), nba_id(gi, side, k)
                mean = 18.0 + 2.5 * ((k * 5 + gi * 2 + side) % 9)
                mean_of[str(did)] = mean
                cur = conn.execute(
                    "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, team, "
                    "run_id, created_at, model_version, role, p_play, inputs_json) VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?)",
                    (gids[gi], day.isoformat(), sid, nba_id(gi, side, k), team, f"run{d}",
                     f"{day.isoformat()}T22:50:00Z", "synth", "starter", 0.95,
                     json.dumps({"dk_id": str(did)})))
                pid = cur.lastrowid
                q = lambda m, w: [max(0.0, m + w * z) for z in (-1.64, -1.28, -0.67, 0, .67, 1.28, 1.64)]
                for cond, m in (("all", mean * 0.95), ("played", mean)):
                    qs = q(m, qwidth)
                    conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, "
                                 "mean, sd, q05, q10, q25, q50, q75, q90, q95) VALUES "
                                 "(?,?,?,?,?,?,?,?,?,?,?,?)", (pid, "dk", cond, m, qwidth, *qs))
                    conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, "
                                 "mean) VALUES (?,?,?,?)",
                                 (pid, "min", cond, 28.0 if cond == "played" else 26.6))
    conn.commit()
    pool = R.build_pool(conn, sid)
    projs = {k: O.Projection(v, p25=0.8 * v) for k, v in mean_of.items()}
    cash = O.optimize(pool, projs, RULES, mode="cash")
    top = T.top_n(pool, projs, RULES, 3, min_diff=1, mode="mean")
    O.save_lineups(conn, f"run{d}", "cash", [cash])
    O.save_lineups(conn, f"run{d}", "topn", list(top.lineups))
    return sid


def actual_of(k, side):
    return R.dk_points(stat_line(k, side))


def grades(conn, table, key="lineup_id"):
    return {r[0]: dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY {key}")}


# --------------------------------------------------------------------------- players

def test_actual_dk_includes_dnp_and_bonuses(conn):
    build_day(conn)
    res = G.grade_day(conn, DAY)
    assert res["status"] == "ok" and res["players"]["graded"] == 48
    rows = {(r["game_id"], r["player_id"]): dict(r)
            for r in conn.execute("SELECT * FROM player_projection_grades")}
    gid0 = game_id(0, 0)
    dnp = rows[(gid0, nba_id(0, 0, 7))]
    assert dnp["played"] == 0 and dnp["actual_dk"] == 0.0 and dnp["actual_min"] == 0.0
    star = rows[(gid0, nba_id(0, 1, 6))]
    assert star["played"] == 1
    assert star["actual_dk"] == pytest.approx(actual_of(6, 1))
    assert star["actual_min"] == 32.0
    # baselines recorded from the capture
    assert star["dk_fppg"] is not None and star["salary"] is not None
    # projected means are the unconditional ("all") ones
    assert star["proj_dk"] == pytest.approx(0.95 * (18.0 + 2.5 * ((6 * 5 + 0 + 1) % 9)))
    assert star["proj_min"] == pytest.approx(26.6)


def test_triple_double_bonus_in_grade(conn):
    sid = build_day(conn, box=False)
    gid = game_id(0, 0)
    conn.execute("INSERT INTO player_games (game_id, player_id, player_name, team, season, "
                 "season_type, game_date, min, pts, reb, ast, stl, blk, fg3m, tov) VALUES "
                 "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (gid, nba_id(0, 0, 0), "x", "BOS", "2026-27",
                                                     "Regular Season", DAY.isoformat(), 35, 20, 11, 10, 0, 0, 2, 3))
    conn.commit()
    G.grade_players(conn, DAY)
    r = conn.execute("SELECT actual_dk FROM player_projection_grades WHERE player_id=?",
                     (nba_id(0, 0, 0),)).fetchone()
    assert r[0] == pytest.approx(20 + 2 * 0.5 + 11 * 1.25 + 10 * 1.5 - 3 * 0.5 + 1.5 + 3.0)


def test_pit_and_coverage_from_quantiles(conn):
    qp = np.array([10, 13, 18, 25, 32, 37, 40], float)
    # the median maps to u = 0.5 (given he plays, p_play = 1)
    assert G.pit_from_quantiles(25.0, 1.0, qp, qp, 0.3) == pytest.approx(0.5)
    assert G.pit_from_quantiles(13.0, 1.0, qp, qp, 0.3) == pytest.approx(0.10)
    assert G.pit_from_quantiles(1000.0, 1.0, qp, qp, 0.3) == pytest.approx(1.0)
    # a DNP (y = 0) with P(DNP) = 0.2: the atom is spread over [0, 0.2] by V
    us = [G.pit_from_quantiles(0.0, 0.8, qp, qp, v) for v in (0.0, 0.25, 0.5, 0.999)]
    assert us == sorted(us) and 0.0 <= us[0] and us[-1] <= 0.2 + 1e-12
    assert us[1] == pytest.approx(0.05)
    # a player who plays sits above the atom
    assert G.pit_from_quantiles(25.0, 0.8, qp, qp, 0.3) == pytest.approx(0.2 + 0.8 * 0.5)
    # no played quantiles: falls back to the unconditional ones
    assert G.pit_from_quantiles(25.0, None, None, qp, 0.3) == pytest.approx(0.5)
    assert G.pit_from_quantiles(25.0, None, None, None, 0.3) is None


def test_grade_flags_interval_hits_and_pit_range(conn):
    build_day(conn, qwidth=12.0)
    G.grade_day(conn, DAY)
    rows = [dict(r) for r in conn.execute("SELECT * FROM player_projection_grades")]
    assert rows and all(0.0 <= r["pit"] <= 1.0 for r in rows)
    assert all(r["in80"] in (0, 1) and r["in50"] in (0, 1) for r in rows)
    assert all(r["crps"] >= 0 for r in rows)
    for r in rows:                                    # in50 implies in80
        assert r["in80"] >= r["in50"]
    # the interval flag is the stored q10-q90 test of the unconditional distribution
    pid = rows[0]["projection_id"]
    q = conn.execute("SELECT q10, q90, q25, q75 FROM player_projection_stats WHERE "
                     "projection_id=? AND stat='dk' AND cond='all'", (pid,)).fetchone()
    assert rows[0]["in80"] == int(q["q10"] <= rows[0]["actual_dk"] <= q["q90"])
    assert rows[0]["in50"] == int(q["q25"] <= rows[0]["actual_dk"] <= q["q75"])


def test_pit_and_crps_from_sims_when_present(conn, tmp_path):
    build_day(conn)
    pid = nba_id(0, 1, 6)
    x = np.concatenate([np.zeros(200), np.linspace(10, 60, 800)])         # 20% DNP mass
    others = nba_id(0, 0, 0)
    fp = np.stack([x, np.full_like(x, 20.0)], axis=1)
    path = SO.save_sims("S0", "run0", [pid, others], fp, root=tmp_path / "sims")
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id='run0'", (str(path),))
    conn.commit()
    G.grade_players(conn, DAY)
    r = conn.execute("SELECT * FROM player_projection_grades WHERE player_id=?", (pid,)).fetchone()
    y = actual_of(6, 1)
    f_lo, f_hi = (x < y).mean(), (x <= y).mean()
    assert f_lo - 1e-9 <= r["pit"] <= f_hi + 1e-9
    assert r["crps"] == pytest.approx(MT.crps_samples(x.astype(np.float16).astype(float), y),
                                      rel=1e-3)
    # the constant-20 column: CRPS = |20 - y|; its PIT is the randomised step at 20
    r2 = conn.execute("SELECT * FROM player_projection_grades WHERE player_id=?", (others,)).fetchone()
    assert r2["crps"] == pytest.approx(abs(20.0 - r2["actual_dk"]), abs=1e-3)


# --------------------------------------------------------------------------- lineups

def test_lineup_grades_hindsight_and_random_percentile(conn):
    sid = build_day(conn)
    res = G.grade_day(conn, DAY)
    assert res["slates"][0]["status"] == "graded" and res["slates"][0]["lineups"] == 4
    g = grades(conn, "dfs_lineup_grades")
    assert len(g) == 4
    pool = R.build_pool(conn, sid)
    actual = {str(dk_id(gi, side, k)): (0.0 if (gi, side, k) == (0, 0, 7) else actual_of(k, side))
              for gi in range(3) for side in range(2) for k in range(8)}
    hs = O.optimize(pool, {k: O.Projection(v) for k, v in actual.items()}, RULES)
    hind = sum(actual[p] for p in hs.player_ids)
    for lid, r in g.items():
        row = conn.execute("SELECT slots_json, actual, ratio_hindsight FROM dfs_lineups WHERE id=?",
                           (lid,)).fetchone()
        pts = sum(actual[str(s["player_id"])] for s in json.loads(row["slots_json"]))
        assert r["actual"] == pytest.approx(pts)
        assert r["hindsight"] == pytest.approx(hind)
        assert r["actual"] <= r["hindsight"] + 1e-9               # hindsight >= any lineup
        assert 0.0 < r["ratio_hindsight"] <= 1.0 + 1e-9
        assert 0.0 <= r["pct_random"] <= 1.0
        assert row["actual"] == pytest.approx(pts)                # dfs_lineups columns filled
        assert row["ratio_hindsight"] == pytest.approx(pts / hind)
        assert r["cash_line_source"] == "none" and r["cash_line"] is None
        assert r["would_cash"] is None and r["est_rank"] is None and r["est_payout"] is None
        d = json.loads(r["detail_json"])
        assert d["day"] == DAY.isoformat() and d["unresolved"] == []
    # the baselines are real optimizer lineups scored with the actuals, never above hindsight
    snap = json.loads(conn.execute("SELECT details_json FROM dfs_metric_snapshots WHERE "
                                   "model='slate:S0' AND metric='baselines'").fetchone()[0])
    assert snap["hindsight"]["actual"] == pytest.approx(hind)
    fp = {p.dk_player_id: O.Projection(p.fppg) for p in pool.players}
    f_lu = O.optimize(pool, fp, RULES)
    assert snap["fppg"]["actual"] == pytest.approx(sum(actual[p] for p in f_lu.player_ids))
    for k in ("fppg", "salary"):
        assert snap[k]["actual"] <= hind + 1e-9
        assert 0.0 <= snap[k]["pct_random"] <= 1.0
    assert snap["random"]["n"] == G.N_RANDOM and 0 < snap["random"]["mean"] < snap["random"]["p99"]


def test_random_lineups_are_valid_seeded_and_showdown_aware():
    rules81 = R.Rules.from_payload(RULES81)
    players = []
    for i in range(14):
        sal = 3000 + 500 * (i % 7)
        slots = rules81.eligible_slots(("PG",))
        players.append(R.PoolPlayer(
            dk_player_id=str(i), dk_dk_id=None, name=f"P{i}", team="AAA" if i < 7 else "BBB",
            positions=("PG",), salary=sal, slots=slots,
            salary_by_slot={s: int(sal * rules81.slot_multiplier(s)) for s in slots},
            draftable_ids={}, game_id="g", competition_id=None, game_label="g", home="AAA",
            away="BBB", start_time=None, status=None, news_status=None, starter=None,
            swappable=None, fppg=None))
    actual = {str(i): float(i + 1) for i in range(14)}
    a = G.random_lineup_scores(players, rules81, actual, 500, seed=7)
    b = G.random_lineup_scores(players, rules81, actual, 500, seed=7)
    assert len(a) == 500 and np.array_equal(a, b)
    best = O.optimize(players, {k: O.Projection(v) for k, v in actual.items()}, rules81)
    top = G.score_lineup(best, actual, rules81)
    assert a.max() <= top + 1e-9 and a.min() > 0
    # CPT carries x1.5 in the stored-lineup score
    slots = [{"slot": "CPT", "player_id": "13", "name": "x"}] + [
        {"slot": "UTIL", "player_id": str(i), "name": "y"} for i in range(5)]
    pts, det, miss = G.lineup_actual(slots, actual, rules81)
    assert pts == pytest.approx(14 * 1.5 + sum(range(1, 6))) and not miss


def test_real_cash_line_from_uploaded_standings_only(conn):
    sid = build_day(conn)
    G.grade_day(conn, DAY)
    assert {r[0] for r in conn.execute("SELECT cash_line_source FROM dfs_lineup_grades")} == {"none"}
    pays = [{"min_pos": 1, "max_pos": 1, "payout": 100.0}, {"min_pos": 2, "max_pos": 3, "payout": 20.0}]
    conn.execute("INSERT INTO dfs_contests (contest_id, slate_id, name, entry_fee, field_size, "
                 "payout_json, captured_at) VALUES ('C1', ?, 'NBA Double Up', 5, 5, ?, 'x')",
                 (sid, json.dumps(pays)))
    for rank, (pts, pay) in enumerate([(400, 100), (300, 20), (280, 20), (100, 0), (50, 0)], 1):
        conn.execute("INSERT INTO dfs_contest_results (contest_id, rank, entry_hash, points, "
                     "payout, uploaded_at) VALUES ('C1', ?, ?, ?, ?, 'now')", (rank, f"h{rank}", pts, pay))
    conn.commit()
    res = G.grade_day(conn, DAY)                 # an upload makes the cached grades stale
    assert res["slates"][0]["status"] == "graded" and res["slates"][0]["real_contests"] == 1
    for r in conn.execute("SELECT * FROM dfs_lineup_grades"):
        assert r["cash_line_source"] == "real" and r["cash_line"] == 280
        assert r["would_cash"] == int(r["actual"] >= 280)
        rank = 1 + sum(1 for p in (400, 300, 280, 100, 50) if p > r["actual"])
        assert r["est_rank"] == rank
        assert r["est_payout"] == G._payout_at([{"min_pos": 1, "max_pos": 1, "payout": 100.0},
                                                {"min_pos": 2, "max_pos": 3, "payout": 20.0}], rank)
    assert G.grade_day(conn, DAY)["slates"][0]["status"] == "cached"


# --------------------------------------------------------------------------- idempotence, finality

def _strip(rows):
    return [{k: v for k, v in dict(r).items() if k != "graded_at"} for r in rows]


def test_regrade_is_idempotent(conn):
    build_day(conn)
    G.grade_day(conn, DAY)
    tables = ("player_projection_grades", "dfs_lineup_grades", "dfs_metric_snapshots",
              "player_metric_snapshots")
    first = {t: _strip(conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3, 4").fetchall())
             for t in tables}
    again = G.grade_day(conn, DAY)
    assert again["status"] == "cached" and again["players"]["graded"] == 0
    forced = G.grade_day(conn, DAY, force=True)
    assert forced["status"] == "ok"
    for t in tables:
        now = _strip(conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3, 4").fetchall())
        assert len(now) == len(first[t]), t
        for a, b in zip(first[t], now):
            for k in a:
                if isinstance(a[k], float):
                    assert a[k] == pytest.approx(b[k], abs=1e-9), (t, k)
                elif k != "details_json" or "graded_at" not in str(a[k]):
                    assert a[k] == b[k], (t, k)


def test_non_final_games_are_skipped_then_graded_later(conn):
    sid = build_day(conn)
    g1 = game_id(0, 1)
    conn.execute("UPDATE games SET status='live' WHERE game_id=?", (g1,))
    conn.commit()
    res = G.grade_day(conn, DAY)
    assert res["players"]["graded"] == 32 and res["players"]["pending"] == 16
    assert res["slates"][0]["status"] == "pending"
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineup_grades").fetchone()[0] == 0
    assert not conn.execute("SELECT 1 FROM player_projection_grades WHERE game_id=?", (g1,)).fetchone()
    conn.execute("UPDATE games SET status='final' WHERE game_id=?", (g1,))
    conn.commit()
    res = G.grade_day(conn, DAY)
    assert res["players"]["graded"] == 16 and res["slates"][0]["status"] == "graded"
    assert conn.execute("SELECT COUNT(*) FROM player_projection_grades").fetchone()[0] == 48


def test_final_without_box_scores_is_not_graded(conn):
    build_day(conn, box=False)
    res = G.grade_day(conn, DAY)
    assert res["status"] == "pending"
    assert conn.execute("SELECT COUNT(*) FROM player_projection_grades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineup_grades").fetchone()[0] == 0


def test_no_runs_is_a_noop_and_writes_nothing(conn):
    assert G.grade_day(conn, DAY)["status"] == "skipped"
    build_day(conn, final=False)                      # runs exist, games not final
    before = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in (
        "player_projection_grades", "dfs_lineup_grades", "dfs_metric_snapshots",
        "player_metric_snapshots")]
    res = G.grade_day(conn, DAY)
    assert res["status"] == "pending"
    after = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in (
        "player_projection_grades", "dfs_lineup_grades", "dfs_metric_snapshots",
        "player_metric_snapshots")]
    assert before == after == [0, 0, 0, 0]
    assert collect._dfs_grade_step(conn, DAY)["status"] == "skipped"
    assert collect._dfs_grade_step(conn, DAY + timedelta(days=30))["status"] == "skipped"


def test_postgame_step_grades_and_catches_up(conn):
    build_day(conn)
    out = collect._dfs_grade_step(conn, DAY)
    assert out["graded_players"] == 48 and out["slates"] == {"S0": "graded"}
    # the next night: yesterday is cached, nothing breaks, no new work
    build_day(conn, d=1)
    out = collect._dfs_grade_step(conn, DAY + timedelta(days=1))
    assert out["graded_players"] == 48
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineup_grades").fetchone()[0] == 8


# --------------------------------------------------------------------------- snapshots

def _snap(conn, table, model, metric, window, as_of):
    r = conn.execute(f"SELECT * FROM {table} WHERE as_of=? AND model=? AND metric=? AND "
                     f"\"window\"=?", (as_of.isoformat(), model, metric, window)).fetchone()
    return dict(r) if r else None


def test_snapshot_windows_and_paired_differences(conn):
    days = 8
    for d in range(days):
        build_day(conn, d=d, dnp=((0, 0, 7), (1, 1, d % 8)))
        G.grade_day(conn, DAY + timedelta(days=d))
    last = DAY + timedelta(days=days - 1)
    cash = {}
    for r in conn.execute("SELECT g.actual, g.detail_json, g.slate_id FROM dfs_lineup_grades g"):
        det = json.loads(r["detail_json"])
        if det["role"] == "cash":
            cash[det["day"]] = (r["actual"], r["slate_id"])
    base = {}
    for r in conn.execute("SELECT as_of, details_json FROM dfs_metric_snapshots WHERE "
                          "model LIKE 'slate:%'"):
        base[r["as_of"]] = json.loads(r["details_json"])
    n_of = {"1d": 1, "7d": 7, "30d": 8, "season": 8}
    for w, n in n_of.items():
        s = _snap(conn, "dfs_metric_snapshots", "cash", "actual_mean", w, last)
        assert s["n"] == n, w
        days_in = sorted(cash)[-n:]
        assert s["value"] == pytest.approx(np.mean([cash[x][0] for x in days_in]))
        dv = _snap(conn, "dfs_metric_snapshots", "cash", "diff_vs_fppg", w, last)
        diffs = [cash[x][0] - base[x]["fppg"]["actual"] for x in days_in]
        assert dv["n"] == n and dv["value"] == pytest.approx(np.mean(diffs))
        ds = _snap(conn, "dfs_metric_snapshots", "cash", "diff_vs_salary", w, last)
        assert ds["value"] == pytest.approx(np.mean(
            [cash[x][0] - base[x]["salary"]["actual"] for x in days_in]))
        if n >= G.MIN_DAYS_SE:
            assert dv["se"] is not None and dv["se"] >= 0
        else:
            assert dv["se"] is None
        assert _snap(conn, "dfs_metric_snapshots", "hindsight", "actual_mean", w, last)["n"] == n
        assert 0 < _snap(conn, "dfs_metric_snapshots", "cash", "ratio_hindsight", w, last)["value"] <= 1
    # 1-day window of the first day: a single day never gets an SE
    assert _snap(conn, "dfs_metric_snapshots", "cash", "diff_vs_fppg", "1d", DAY)["se"] is None
    # player snapshots: ours vs fppg vs salary-implied on the same table
    for w in G.WINDOWS:
        for model in ("ours", "fppg", "salary"):
            assert _snap(conn, "player_metric_snapshots", model, "dk_mae", w, last)["n"] > 0
    ours = _snap(conn, "player_metric_snapshots", "ours", "dk_mae", "1d", last)
    rows = conn.execute("SELECT proj_dk, actual_dk, proj_min FROM player_projection_grades g "
                        "JOIN player_projections pp ON pp.id=g.projection_id WHERE pp.nba_day=?",
                        (last.isoformat(),)).fetchall()
    pop = [abs(r[0] - r[1]) for r in rows if r[2] >= 10]
    assert ours["n"] == len(pop) and ours["value"] == pytest.approx(np.mean(pop))
    cov = _snap(conn, "player_metric_snapshots", "ours", "dk_cov80", "season", last)
    assert 0.0 <= cov["value"] <= 1.0 and cov["se"] is not None
    gain = _snap(conn, "player_metric_snapshots", "ours", "dk_mae_gain_vs_fppg", "season", last)
    assert gain["n"] > 0 and gain["value"] is not None
    # the windows are keyed per as_of: earlier days keep their own rows
    assert _snap(conn, "dfs_metric_snapshots", "cash", "actual_mean", "season", DAY)["n"] == 1


def test_real_entries_stay_separate_from_simulated(conn):
    sid = build_day(conn)
    G.grade_day(conn, DAY)
    rows = [("c1", 5.0, 12.0, "real"), ("c2", 20.0, 0.0, "real"), ("c3", 3.0, None, "real"),
            ("c4", 10.0, 500.0, "simulated"), ("c5", 10.0, 90.0, "proxy")]
    for i, (cid, fee, pay, src) in enumerate(rows):
        conn.execute("INSERT INTO dfs_entries (contest_id, slate_id, dk_entry_id, entry_fee, "
                     "payout, source, created_at) VALUES (?,?,?,?,?,?,?)",
                     (cid, sid, f"e{i}", fee, pay, src, "2026-10-21T00:00:00Z"))
    conn.commit()
    G.grade_day(conn, DAY, force=True)
    get = lambda m: _snap(conn, "dfs_metric_snapshots", "real", m, "1d", DAY)
    assert get("entries")["value"] == 3                       # only the owner's real entries
    assert get("fees")["value"] == pytest.approx(28.0)
    assert get("winnings")["value"] == pytest.approx(12.0)
    assert get("net")["value"] == pytest.approx(12.0 - 25.0)  # the unpaid entry has no result yet
    assert json.loads(get("entries")["details_json"])["pending_entries"] == 1
    # simulated lineup metrics never mix with them
    assert _snap(conn, "dfs_metric_snapshots", "cash", "actual_mean", "1d", DAY)["value"] > 0
    assert not conn.execute("SELECT 1 FROM dfs_metric_snapshots WHERE model='real' AND "
                            "metric='actual_mean'").fetchone()


def test_report_prints_latest_snapshot(conn, capsys, monkeypatch):
    import argparse
    build_day(conn)
    G.grade_day(conn, DAY)
    monkeypatch.setattr(G, "_conn", lambda: conn)
    assert G._report_cmd(argparse.Namespace(window="7d", as_of=None)) == 0
    out = capsys.readouterr().out
    assert "DFS tracking as of 2026-10-20" in out and "dk_mae" in out and "S0" in out
    assert G._grade_cmd(argparse.Namespace(date="2026-10-20", force=False)) == 0
