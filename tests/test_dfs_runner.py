"""Lock-relative DFS runner (T1.6): tick windows, idempotency, late news, the availability rule,
late-swap re-solve, the pipeline stub and the FPPG fallback. Offline: no DK / ESPN / NBA request.

Slates are built from captured DraftKings payloads (tests/fixtures/dfs/ and, when present, the
archived draft groups under var/raw/draftkings/) or from synthetic draftables with the same
shape (`payload`)."""

from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from courtside import archive, db
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import runner, runner_cli
from courtside.sources import dfs, dfs_capture as cap

FIX = Path(__file__).parent / "fixtures" / "dfs"
REAL_RAW = Path(__file__).parents[1] / "var" / "raw" / "draftkings" / "draftgroup" / "2026-10-20"
DAY = date(2026, 10, 20)
UTC = timezone.utc
RULES70 = json.loads((FIX / "rules_70.json").read_text())
RULES81 = json.loads((FIX / "rules_81.json").read_text())
G1, G2, G3 = "2026-10-20T23:00:00.0000000Z", "2026-10-21T01:00:00.0000000Z", "2026-10-21T03:00:00.0000000Z"


def T(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def J(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


# --------------------------------------------------------------------------- synthetic slates

POS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF"]
TEAMS = [("BOS", "DET"), ("PHI", "NYK"), ("OKC", "SAS"), ("LAL", "DEN")]


def pid_of(gi: int, side: int, k: int) -> int:
    return 1000 * (gi + 1) + 100 * side + k + 1


def payload(starts, *, status=None, starters=(), disabled=(), rules=RULES70):
    """A draftables payload of len(starts) games (14 players each, Classic eligibility)."""
    slot_id = {s["rosterSlot"]["name"]: s["rosterSlot"]["id"] for s in rules["lineupTemplate"]}
    cs = R.Rules.from_payload(rules)
    status = status or {}
    comps, drafts = [], []
    for gi, start in enumerate(starts):
        away, home = TEAMS[gi]
        cid = 7000 + gi
        comps.append({"competitionId": cid, "homeTeam": {"abbreviation": home},
                      "awayTeam": {"abbreviation": away}, "startTime": start,
                      "name": f"{away} @ {home}", "startingLineupsAvailable": False})
        for side, team in enumerate((away, home)):
            for k, pos in enumerate(POS):
                pid = pid_of(gi, side, k)
                salary = 3500 + 700 * ((k * 3 + gi + side) % 8)
                fppg = 14.0 + 3.2 * ((k * 5 + gi * 2 + side) % 9) + salary / 2000
                attrs = [{"id": 100, "value": "true"}] if pid in starters else []
                for slot in cs.eligible_slots(pos):
                    drafts.append({
                        "draftableId": pid * 10 + slot_id[slot] % 10 + 100000 * slot_id[slot],
                        "firstName": team, "lastName": f"P{k}", "displayName": f"{team} P{k}",
                        "playerId": pid, "playerDkId": pid + 500000, "position": pos,
                        "rosterSlotId": slot_id[slot], "salary": salary,
                        "status": status.get(pid, "None"), "isSwappable": True,
                        "isDisabled": pid in disabled, "newsStatus": "None",
                        "competition": {"competitionId": cid, "name": f"{away} @ {home}",
                                        "startTime": start},
                        "draftStatAttributes": [{"id": 219, "value": f"{fppg:.1f}"}],
                        "teamAbbreviation": team, "playerGameAttributes": attrs})
    return {"draftables": drafts, "competitions": comps}


def capture(conn, sid, pl, *, at: str, tag="main", gt=70, rules=RULES70):
    now = T(at)
    cap.store_draftables(conn, pl, slate_id=sid, game_type_id=gt, day=DAY, tag=tag, rules=rules,
                         lobby=None, start_utc=None, captured_at=cap.iso(now), now=now)


def by_name(conn, sid, name):
    return conn.execute("SELECT site_player_id FROM dfs_salaries WHERE slate_id=? AND "
                        "player_name=? LIMIT 1", (str(sid), name)).fetchone()[0]


class Pipe:
    """Stand-in for courtside.proj.pipeline: mean = fppg, p25 = 80% of it."""
    MODEL_VERSION = "stub-1"

    def __init__(self, conn_pool_fppg, scale=None):
        self.calls = []
        self.fppg = conn_pool_fppg
        self.scale = scale or {}

    def project_slate(self, conn, slate_id, at, *, run_id, n_sims=25_000, seed=None):
        self.calls.append((slate_id, at, run_id))
        rows = conn.execute("SELECT DISTINCT site_player_id, fppg FROM dfs_salaries WHERE "
                            "slate_id=? AND fppg IS NOT NULL", (str(slate_id),)).fetchall()
        return {int(r[0]): O.Projection(r[1] * self.scale.get(str(r[0]), 1.0),
                                        p25=0.8 * r[1] * self.scale.get(str(r[0]), 1.0))
                for r in rows}


@pytest.fixture
def pipe(monkeypatch):
    p = Pipe(None)
    mod = types.ModuleType("courtside.proj.pipeline")
    mod.project_slate = p.project_slate
    mod.MODEL_VERSION = Pipe.MODEL_VERSION
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", mod)
    return p


@pytest.fixture
def avail(monkeypatch):
    """availability_for stub: p_play from a dict {dk id: p} (default 1.0)."""
    table: dict[str, float] = {}

    def fake(conn, slate, pool, now):
        return {p.dk_player_id: {"p": 0.0 if p.status == "O" else table.get(p.dk_player_id, 1.0),
                                 "status": p.status or "", "source": "stub"}
                for p in pool.players}
    monkeypatch.setattr(runner, "availability_for", fake)
    return table


@pytest.fixture
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network refresh called")
    monkeypatch.setattr(runner, "refresh_inputs", boom)
    monkeypatch.setattr(runner.dfs_capture, "refresh_slate", boom)


@pytest.fixture
def slate3(conn):
    """One Classic slate: games at 23:00Z / 01:00Z / 03:00Z, first capture at 20:00Z."""
    capture(conn, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
    return "S1"


# --------------------------------------------------------------------------- tick windows

def captured_slates():
    """Three captured payloads: the Classic fixture, the Showdown fixture and the archived
    opening-night main slate (the Classic fixture again when the archive is not on this
    machine)."""
    real = sorted(REAL_RAW.glob("draftgroup_154575__*.json.gz"))
    third = archive.load_raw(real[-1]) if real else J("draftables_classic.json")
    return [("C1", J("draftables_classic.json"), 70, "main", RULES70),
            ("SD", J("draftables_showdown.json"), 81, "showdown", RULES81),
            ("C2", third, 70, "main", RULES70)]


def seed_captured(conn):
    now = T("2026-10-20T12:00:00Z")
    for sid, pl, gt, tag, rules in captured_slates():
        cap.store_draftables(conn, pl, slate_id=sid, game_type_id=gt, day=DAY, tag=tag,
                             rules=rules, lobby=None, start_utc=None,
                             captured_at=cap.iso(now), now=now)


def save_fake_lineups(conn, sid):
    """A saved lock10 build for the swap path of tick (one dummy lineup)."""
    rid = O.save_run(conn, slate_id=sid, kind="manual", ok=True, detail={},
                     created_at="2026-10-20T10:00:00Z")
    conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean) "
                 "VALUES (?,?,?,?,?,?)",
                 (rid, "cash", 1, json.dumps([{"slot": "UTIL", "player_id": "1"}]), 1, 1.0))
    conn.commit()


def test_tick_fires_inside_windows_only(conn, monkeypatch):
    seed_captured(conn)
    for sid in ("C1", "C2"):
        save_fake_lineups(conn, sid)
    fired = []

    def fake_run(c, slate_id, kind, *, now, tip=None, **kw):
        trig = runner.iso(tip) if tip else None
        O.save_run(c, slate_id=slate_id, kind=kind, ok=True, detail={"trigger": trig},
                   created_at=runner.iso(now))
        fired.append((slate_id, kind, trig, now))
        return {"status": "ok"}

    refreshed = []
    monkeypatch.setattr(runner, "run_slate", fake_run)
    monkeypatch.setattr(runner, "refresh_inputs", lambda c, n: refreshed.append(n) or {})
    start = T("2026-10-20T12:00:00Z")
    outside_calls = 0
    t = start
    while t <= start + timedelta(hours=36):
        before = len(refreshed)
        res = runner.tick(conn, t)
        if not [r for r in res if r["action"] != "already_done"]:
            outside_calls += len(refreshed) - before
        t += timedelta(minutes=5)
    assert outside_calls == 0          # a tick with nothing due makes no request

    # every firing is inside its window, and each (slate, kind, tip) fired exactly once
    keys = [(s, k, tr) for s, k, tr, _ in fired]
    assert len(keys) == len(set(keys))
    for sid, kind, trig, now in fired:
        sl = runner._slate(conn, sid)
        tips = runner.slate_tips(conn, sid)
        lock = runner.slate_lock(sl, tips)
        ev = T(trig) if trig else lock
        win = {"lock45": runner.LOCK45_WINDOW, "lock10": runner.LOCK10_WINDOW,
               "swap": runner.SWAP_WINDOW}[kind]
        d = (ev - now).total_seconds()
        assert win[1] - runner.TOLERANCE_S <= d <= win[0], (sid, kind, d)
    # and every expected firing happened
    expect = set()
    for sid in ("C1", "C2"):
        sl = runner._slate(conn, sid)
        tips = runner.slate_tips(conn, sid)
        lock = runner.slate_lock(sl, tips)
        expect |= {(sid, "lock45", None), (sid, "lock10", None)}
        expect |= {(sid, "swap", runner.iso(x)) for x in tips if x > lock}
    sd_lock = runner.slate_lock(runner._slate(conn, "SD"), runner.slate_tips(conn, "SD"))
    expect |= {("SD", "lock45", None), ("SD", "lock10", None)}     # Showdown: no late swap
    assert set(keys) == expect
    assert not [k for k in keys if k[0] == "SD" and k[1] == "swap"]
    assert sd_lock is not None


def test_tick_window_edges(conn):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    lock = T(G1)

    def due(minutes_before, sec=0):
        return [(d["kind"]) for d in runner.due_runs(conn, lock - timedelta(minutes=minutes_before,
                                                                          seconds=sec))]
    assert due(47) == ["lock45"] and due(40) == ["lock45"]
    assert due(47, 1) == [] and due(48) == []
    assert due(39) == [] and due(30) == [] and due(13) == []
    assert due(12) == ["lock10"] and due(7) == ["lock10"]
    assert due(6) == [] and due(1) == [] and due(0) == []
    assert due(-60) == []                      # an hour after the first lock: nothing until swap
    swap = runner.due_runs(conn, T(G2) - timedelta(minutes=10))
    assert [d["done"] for d in swap] == [True]        # nothing saved to swap: nothing to do
    save_fake_lineups(conn, "S1")
    swap = runner.due_runs(conn, T(G2) - timedelta(minutes=10))
    assert [(d["kind"], d["trigger"]) for d in swap] == [("swap", runner.iso(T(G2)))]
    assert runner.due_runs(conn, T(G2) - timedelta(minutes=30)) == []


def test_tick_makes_no_request_outside_windows(conn, no_net):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    save_fake_lineups(conn, "S1")
    for minutes in (300, 120, 60, 30, 20, 3, 0, -30):
        assert runner.tick(conn, T(G1) - timedelta(minutes=minutes)) == []
    # the no-op path does not even write
    n = conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0]
    runner.tick(conn, T(G1) - timedelta(minutes=100))
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == n


def test_tick_refreshes_once_per_tick_and_slate(conn, monkeypatch, pipe, avail):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    capture(conn, "S2", payload([G1, G3]), at="2026-10-20T20:00:00Z", tag="late")
    inputs, drafts = [], []
    monkeypatch.setattr(runner, "refresh_inputs", lambda c, n: inputs.append(n) or {})
    monkeypatch.setattr(runner.dfs_capture, "refresh_slate",
                        lambda c, s, *, reason, now=None, **k: drafts.append((s, reason)) or {})
    res = runner.tick(conn, T(G1) - timedelta(minutes=45))
    assert sorted(r["slate_id"] for r in res if r["action"] == "ran") == ["S1", "S2"]
    assert len(inputs) == 1 and sorted(drafts) == [("S1", "lock45"), ("S2", "lock45")]
    # the same window again: already done, nothing is fetched
    res = runner.tick(conn, T(G1) - timedelta(minutes=42))
    assert {r["action"] for r in res} == {"already_done"}
    assert len(inputs) == 1 and len(drafts) == 2


def test_tick_dry_run_touches_nothing(conn, no_net):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    res = runner.tick(conn, T(G1) - timedelta(minutes=10), dry_run=True)
    assert [(r["slate_id"], r["kind"], r["action"]) for r in res] == [("S1", "lock10", "would_run")]
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == 0


# --------------------------------------------------------------------------- run_slate

def test_run_slate_saves_run_and_lineups_and_is_idempotent(conn, slate3, pipe, avail):
    now = T(G1) - timedelta(minutes=45)
    r = runner.run_slate(conn, "S1", "lock45", now=now, refresh=False, n=4)
    assert r["status"] == "ok" and r["lineups"] == {"cash": 1, "topn": 4}
    row = dict(conn.execute("SELECT * FROM dfs_runs").fetchone())
    assert (row["slate_id"], row["kind"], row["ok"], row["path"]) == ("S1", "lock45", 1, "A")
    assert row["model_version"] == "stub-1" and row["created_at"] == runner.iso(now)
    assert row["seconds"] >= 0 and json.loads(row["late_news_json"]) == []
    assert pipe.calls and pipe.calls[0][2] == row["run_id"] and pipe.calls[0][1] == now
    modes = [tuple(x) for x in conn.execute(
        "SELECT mode, rank FROM dfs_lineups WHERE run_id=? ORDER BY mode, rank", (row["run_id"],))]
    assert modes == [("cash", 1), ("topn", 1), ("topn", 2), ("topn", 3), ("topn", 4)]
    top = conn.execute("SELECT mean FROM dfs_lineups WHERE mode='topn' ORDER BY rank").fetchall()
    assert [round(x[0], 6) for x in top] == sorted((round(x[0], 6) for x in top), reverse=True)
    # a second call is a no-op ...
    r2 = runner.run_slate(conn, "S1", "lock45", now=now + timedelta(minutes=2), refresh=False)
    assert r2["status"] == "already_done" and r2["run_id"] == row["run_id"]
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == 1
    # ... unless forced; another kind of the same slate is independent
    r3 = runner.run_slate(conn, "S1", "lock45", now=now, refresh=False, n=2, force=True)
    assert r3["status"] == "ok" and r3["run_id"] != row["run_id"]
    assert runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10),
                            refresh=False, n=2)["status"] == "ok"
    assert conn.execute("SELECT COUNT(*) FROM dfs_runs").fetchone()[0] == 3


def test_cash_uses_p25_and_topn_uses_mean(conn, slate3, pipe, avail):
    # a boom-or-bust star: a huge mean but a terrible floor
    sid = by_name(conn, "S1", "BOS P4")
    projs: dict = {}

    def spiky(conn_, slate_id, at, *, run_id, **kw):
        out = Pipe.project_slate(pipe, conn_, slate_id, at, run_id=run_id)
        out[int(sid)] = O.Projection(55.0, p25=-60.0)
        projs.update(out)
        return out
    sys.modules["courtside.proj.pipeline"].project_slate = spiky
    runner.run_slate(conn, "S1", "manual", now=T(G1) - timedelta(minutes=20), refresh=False, n=3)
    cash = {s["player_id"] for s in json.loads(conn.execute(
        "SELECT slots_json FROM dfs_lineups WHERE mode='cash'").fetchone()[0])}
    top1 = {s["player_id"] for s in json.loads(conn.execute(
        "SELECT slots_json FROM dfs_lineups WHERE mode='topn' AND rank=1").fetchone()[0])}
    pool = R.build_pool(conn, "S1")
    p = {str(k): v for k, v in projs.items()}
    assert top1 == O.optimize(pool, p, mode="mean").player_ids
    assert cash == O.optimize(pool, p, mode="cash", lam=0.3).player_ids
    assert sid in top1 and cash != top1


def test_pipeline_missing_falls_back_to_fppg(conn, slate3, avail, monkeypatch):
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", None)
    r = runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False,
                         n=3)
    assert r["status"] == "ok" and r["model_version"] == "fppg-fallback"
    row = conn.execute("SELECT model_version, detail FROM dfs_runs").fetchone()
    assert row[0] == "fppg-fallback" and "unavailable" in json.loads(row[1])["pipeline"]
    # the fallback lineup is the plain FPPG optimum (the same as `dfs build --proj fppg`)
    pool = R.build_pool(conn, "S1")
    want = O.optimize(pool, {p.dk_player_id: O.Projection(p.fppg) for p in pool.players},
                      mode="cash")
    got = json.loads(conn.execute("SELECT slots_json FROM dfs_lineups WHERE mode='cash'"
                                  ).fetchone()[0])
    assert {s["player_id"] for s in got} == want.player_ids


def test_pipeline_crash_falls_back_and_is_recorded(conn, slate3, avail, monkeypatch):
    mod = types.ModuleType("courtside.proj.pipeline")

    def boom(*a, **k):
        raise RuntimeError("copula exploded")
    mod.project_slate = boom
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", mod)
    r = runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False,
                         n=2)
    assert r["status"] == "ok" and r["model_version"] == "fppg-fallback"
    assert "copula exploded" in conn.execute("SELECT detail FROM dfs_runs").fetchone()[0]


def test_failed_run_is_recorded_and_retried(conn, slate3, pipe, avail, monkeypatch):
    # the model returns nothing for anyone; the fppg fallback has no fppg either -> error
    conn.execute("UPDATE dfs_salaries SET fppg=NULL")
    conn.commit()
    pipe.project_slate = lambda *a, **k: {}
    sys.modules["courtside.proj.pipeline"].project_slate = lambda *a, **k: {}
    r = runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False)
    assert r["status"] == "error"
    row = conn.execute("SELECT ok, detail FROM dfs_runs").fetchone()
    assert row[0] == 0 and "error" in json.loads(row[1])
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups").fetchone()[0] == 0
    # a failed run does not make the window idempotent: the next tick tries again
    assert runner.due_runs(conn, T(G1) - timedelta(minutes=10))[0]["done"] is False


def test_unknown_slate_and_bad_kind(conn):
    assert runner.run_slate(conn, "nope", "lock10", refresh=False)["status"] == "unknown_slate"
    with pytest.raises(ValueError):
        runner.run_slate(conn, "nope", "lock5")


def test_refresh_uses_budgeted_refresh_slate(conn, slate3, pipe, avail, monkeypatch):
    seen = []
    monkeypatch.setattr(runner.dfs_capture, "refresh_slate",
                        lambda c, s, *, reason, now=None, **k:
                        seen.append((s, reason)) or {"status": "budget_exhausted"})
    monkeypatch.setattr(runner, "refresh_inputs", lambda c, n: {"espn_injuries": {"status": "ok"}})
    r = runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), n=2)
    assert r["status"] == "ok" and seen == [("S1", "lock10")]       # a spent budget is not fatal
    d = json.loads(conn.execute("SELECT detail FROM dfs_runs").fetchone()[0])
    assert d["refresh"]["draftables"]["status"] == "budget_exhausted"
    assert "espn_injuries" in d["refresh"]["inputs"]


# --------------------------------------------------------------------------- availability rule

def test_next_run_and_availability_bans(conn, slate3):
    sl = runner._slate(conn, "S1")
    tips = runner.slate_tips(conn, "S1")
    now = T(G1) - timedelta(minutes=45)
    assert runner.next_run_at("lock45", sl, tips, now) == T(G1) - timedelta(minutes=10)
    assert runner.next_run_at("lock10", sl, tips, T(G1) - timedelta(minutes=10)) \
        == T(G2) - timedelta(minutes=10)
    assert runner.next_run_at("swap", sl, tips, T(G2) - timedelta(minutes=10)) \
        == T(G3) - timedelta(minutes=10)
    assert runner.next_run_at("swap", sl, tips, T(G3) - timedelta(minutes=10)) is None
    assert runner.next_run_at("lock10", {**sl, "tag": "showdown"}, tips, now) is None

    pool = R.build_pool(conn, "S1")
    g1 = next(p for p in pool.players if p.start == T(G1))
    g2 = next(p for p in pool.players if p.start == T(G2))
    av = {p.dk_player_id: {"p": 1.0, "status": "", "source": "t"} for p in pool.players}
    av[g1.dk_player_id]["p"] = 0.5
    av[g2.dk_player_id]["p"] = 0.5
    # lock-10: the game-1 Q player locks before the next run (G2 tip - 10) and is out; the
    # game-2 Q player is not
    nxt = runner.next_run_at("lock10", sl, tips, T(G1) - timedelta(minutes=10))
    bans, ex = runner.availability_bans(pool, av, nxt, T(G1))
    assert bans == {g1.dk_player_id} and ex[0]["name"] == g1.name
    # lock-45: the next run (lock-10) comes before every lock, so a Q player stays
    nxt = runner.next_run_at("lock45", sl, tips, now)
    assert runner.availability_bans(pool, av, nxt, now)[0] == set()
    # P(play) at the threshold is fine; no later run means every Q player is out
    av[g2.dk_player_id]["p"] = 0.85
    assert runner.availability_bans(pool, av, T(G2), T(G1))[0] == {g1.dk_player_id}
    av[g2.dk_player_id]["p"] = 0.84
    assert runner.availability_bans(pool, av, None, T(G1))[0] == {g1.dk_player_id, g2.dk_player_id}


def test_run_applies_availability_rule_and_records_swap_candidates(conn, slate3, pipe, avail):
    pool = R.build_pool(conn, "S1")
    # the best FPPG player of game 1 and of game 2 are both Questionable (p = 0.5)
    best = {st: max((p for p in pool.players if p.start == T(st)), key=lambda p: p.fppg)
            for st in (G1, G2)}
    pipe.scale = {best[G1].dk_player_id: 2.0, best[G2].dk_player_id: 2.0}   # make both attractive
    for p in best.values():
        avail[p.dk_player_id] = 0.5
    runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False, n=3)
    cash = {s["player_id"] for s in json.loads(conn.execute(
        "SELECT slots_json FROM dfs_lineups WHERE mode='cash'").fetchone()[0])}
    assert best[G1].dk_player_id not in cash          # game 1 locks before the next run
    assert best[G2].dk_player_id in cash              # a later game: allowed ...
    d = json.loads(conn.execute("SELECT detail FROM dfs_runs").fetchone()[0])
    assert [e["player_id"] for e in d["availability"]["excluded"]] == [best[G1].dk_player_id]
    cands = d["swap_candidates"]                       # ... and a swap candidate is recorded
    assert [c["player_id"] for c in cands] == [best[G2].dk_player_id]
    assert cands[0]["replacement"] and cands[0]["p_play"] == 0.5
    # the lock-45 run keeps the game-1 Q player: lock-10 will still decide
    runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False, n=2)
    cash45 = {s["player_id"] for s in json.loads(conn.execute(
        "SELECT slots_json FROM dfs_lineups WHERE mode='cash' ORDER BY id DESC").fetchone()[0])}
    assert best[G1].dk_player_id in cash45


def test_dk_out_is_always_excluded(conn, pipe, avail):
    capture(conn, "S1", payload([G1, G2], status={1001: "O"}), at="2026-10-20T20:00:00Z")
    pipe.scale = {"1001": 3.0}
    runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False, n=3)
    for (js,) in conn.execute("SELECT slots_json FROM dfs_lineups"):
        assert "1001" not in {s["player_id"] for s in json.loads(js)}


# --------------------------------------------------------------------------- late news

def S(dk, status=None, starter=None, start=G2, name="X"):
    return runner.PState(dk, name, "BOS", status, starter, start)


NOW = T("2026-10-20T22:50:00Z")


def test_late_news_status_change():
    prev = {"1": S("1"), "2": S("2", "Q"), "3": S("3", "Q")}
    cur = {"1": S("1", "O"), "2": S("2", "Q"), "3": S("3", "GTD")}
    news = runner.diff_news(prev, cur, now=NOW)
    assert {n["player_id"] for n in news} == {"1", "3"}
    assert news[0]["reasons"] == ["status"]
    assert {n["player_id"]: n["status"]["to"] for n in news} == {"1": "O", "3": "GTD"}


def test_late_news_report_status_change():
    cur = {"1": S("1"), "2": S("2"), "3": S("3")}
    news = runner.diff_news(cur, cur, now=NOW,
                            report_prev={"1": "Questionable", "2": "Probable", "3": ""},
                            report_cur={"1": "Out", "2": "Available", "3": ""})
    assert [n["player_id"] for n in news] == ["1"]
    assert news[0]["report_status"] == {"from": "Questionable", "to": "Out"}


def test_late_news_unexpected_starter_and_non_starter():
    prev = {"1": S("1"), "2": S("2"), "3": S("3"), "4": S("4"), "5": S("5")}
    cur = {"1": S("1", starter=1), "2": S("2", starter=1), "3": S("3", starter=0),
           "4": S("4", starter=0), "5": S("5", starter=1)}
    share = {"1": 0.0, "2": 0.9, "3": 1.0, "4": 0.1, "5": 0.5}
    news = {n["player_id"]: n for n in runner.diff_news(prev, cur, now=NOW, p_start=share)}
    assert news["1"]["reasons"] == ["unexpected_starter"]          # bench player named a starter
    assert news["3"]["reasons"] == ["unexpected_non_starter"]      # regular starter not starting
    assert set(news) == {"1", "3"}                                  # the expected ones are not news
    # a flag that was already there at the previous run is not new
    prev2 = {"1": S("1", starter=1)}
    assert runner.diff_news(prev2, {"1": S("1", starter=1)}, now=NOW, p_start={"1": 0.0}) == []
    # no start history, no expectation
    assert runner.diff_news(prev, cur, now=NOW, p_start={}) == []


def test_late_news_minutes_move():
    cur = {"1": S("1"), "2": S("2"), "3": S("3"), "4": S("4")}
    news = runner.diff_news(cur, cur, now=NOW,
                            minutes_prev={"1": 20.0, "2": 20.0, "3": 20.0},
                            minutes_now={"1": 23.0, "2": 22.9, "3": 16.5, "4": 30.0})
    assert [(n["player_id"], n["minutes"]["delta"]) for n in news] == [("1", 3.0), ("3", -3.5)]
    assert news[0]["reasons"] == ["minutes"]


def test_late_news_nothing_is_empty_and_started_players_are_skipped():
    cur = {"1": S("1", "Q"), "2": S("2")}
    assert runner.diff_news(cur, cur, now=NOW, p_start={"1": 0.5}, minutes_prev={"1": 20.0},
                            minutes_now={"1": 20.4}) == []
    old = {"1": S("1", start=G1)}
    new = {"1": S("1", "O", start=G1)}
    assert runner.diff_news(old, new, now=T(G1) + timedelta(minutes=1)) == []   # already started
    assert len(runner.diff_news(old, new, now=T(G1) - timedelta(minutes=10))) == 1


def test_late_news_from_database_captures(conn):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    capture(conn, "S1", payload([G1, G2], status={1002: "Q"}, starters={1003}),
            at="2026-10-20T22:00:00Z")
    prev = {"run_id": "r0", "created_at": "2026-10-20T21:00:00Z"}
    news = runner.late_news(conn, "S1", T("2026-10-20T22:30:00Z"), prev, p_start={"1003": 0.0},
                            minutes={}, prev_minutes={})
    byid = {n["player_id"]: n for n in news}
    assert byid["1002"]["reasons"] == ["status"] and byid["1002"]["status"]["to"] == "Q"
    assert byid["1003"]["reasons"] == ["unexpected_starter"]
    assert byid["1002"]["since"] == "2026-10-20T21:00:00Z" and len(news) == 2
    # against a previous run taken AFTER the change: nothing is news any more
    prev2 = {"run_id": "r1", "created_at": "2026-10-20T22:10:00Z"}
    assert runner.late_news(conn, "S1", T("2026-10-20T22:30:00Z"), prev2, p_start={"1003": 0.0},
                            minutes={}, prev_minutes={}) == []
    # with no previous run the baseline is 90 minutes before lock (21:30Z): same two players
    first = runner.late_news(conn, "S1", T("2026-10-20T22:30:00Z"), None, p_start={"1003": 0.0},
                             minutes={}, prev_minutes={})
    assert {n["player_id"] for n in first} == {"1002", "1003"}
    assert first[0]["since"] == "2026-10-20T21:30:00Z"


def test_late_news_from_injury_report_snapshots(conn):
    capture(conn, "S1", payload([G1, G2]), at="2026-10-20T20:00:00Z")
    rows = [("2026-10-20T20:30:00Z", "BOS", "BOS P1", "Questionable"),
            ("2026-10-20T22:20:00Z", "BOS", "BOS P1", "Out"),
            ("2026-10-20T22:20:00Z", "BOS", "BOS P2", "Questionable")]
    conn.executemany("INSERT INTO injury_snapshots (captured_at, source, team, player_name, "
                     "status) VALUES (?,?,?,?,?)", [(a, "nba_official", t, n, s)
                                                     for a, t, n, s in rows])
    conn.commit()
    prev = {"run_id": "r0", "created_at": "2026-10-20T21:00:00Z"}
    news = runner.late_news(conn, "S1", T("2026-10-20T22:30:00Z"), prev, p_start={}, minutes={},
                            prev_minutes={})
    assert {(n["name"], n["report_status"]["to"]) for n in news} == {("BOS P1", "Out"),
                                                                    ("BOS P2", "Questionable")}


def test_run_stores_late_news_and_banner_helper(conn, slate3, pipe, avail):
    runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False, n=3)
    cash = json.loads(conn.execute("SELECT slots_json FROM dfs_lineups WHERE mode='cash'"
                                   ).fetchone()[0])
    star = cash[0]["player_id"]
    other = next(str(i) for i in (1001, 1002, 2001) if str(i) not in {c["player_id"] for c in cash})
    # one lineup player turns Questionable after the lock45 run, one outsider too
    sts = {int(star): "Q", int(other): "Q"}
    capture(conn, "S1", payload([G1, G2, G3], status=sts), at="2026-10-20T22:40:00Z")
    runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False, n=3)
    news = json.loads(conn.execute("SELECT late_news_json FROM dfs_runs WHERE kind='lock10'"
                                   ).fetchone()[0])
    assert {n["player_id"] for n in news} >= {star, other}
    hit = runner.lineups_with_news(conn, "S1")
    assert hit and all(set(h["players"]) <= {star, other} for h in hit)
    assert all(star in h["players"] or other in h["players"] for h in hit)
    assert runner.lineups_with_news(conn, "S1", [])  == []


# --------------------------------------------------------------------------- late swap

def test_swap_keeps_started_players_and_replaces_news_player(conn, slate3, pipe, avail):
    runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False, n=3)
    saved = {(d["mode"], d["rank"]): d for d in runner.current_lineups(conn, "S1")}
    pool = R.build_pool(conn, "S1")
    start_of = {p.dk_player_id: p.start for p in pool.players}
    # a game-3 player of the saved cash lineup is ruled out
    victim = next(pid for _, pid in saved[("cash", 1)]["pairs"] if start_of[pid] == T(G3))
    capture(conn, "S1", payload([G1, G2, G3], status={int(victim): "O"}),
            at="2026-10-21T02:30:00Z")
    now = T(G3) - timedelta(minutes=10)                     # game 1 and 2 have started
    r = runner.run_slate(conn, "S1", "swap", now=now, refresh=False)
    assert r["status"] == "ok" and r["lineups"]["swap"] >= 1
    sw = conn.execute("SELECT * FROM dfs_lineups WHERE mode='swap' AND rank=1").fetchone()
    new = [(s["slot"], s["player_id"]) for s in json.loads(sw["slots_json"])]
    old = saved[("cash", 1)]["pairs"]
    assert victim not in {p for _, p in new}
    for slot, pid in old:                                   # every started player stays, in place
        if start_of[pid] < now:
            assert (slot, pid) in new
    assert len(new) == len(old) == 8
    why = json.loads(sw["why_json"])
    assert why["from"]["mode"] == "cash" and why["from"]["rank"] == 1
    assert pool.by_id()[victim].name in why["late_news"] and why["out"]
    # only lineups that contain a news player are re-solved
    hit = [k for k, d in saved.items() if victim in d["player_ids"]]
    assert hit and r["lineups"]["swap"] == len(hit)
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups WHERE mode='swap'").fetchone()[0] \
        == len(hit)
    # the saved lineup now reads as the swapped one
    cur = {(d["mode"], d["rank"]): d for d in runner.current_lineups(conn, "S1")}
    assert cur[("cash", 1)]["via"] == "swap" and victim not in cur[("cash", 1)]["player_ids"]
    # idempotent per game tip
    assert runner.run_slate(conn, "S1", "swap", now=now, refresh=False,
                            tip=T(G3))["status"] == "already_done"


def test_swap_skipped_when_no_news_touches_a_saved_lineup(conn, slate3, pipe, avail):
    runner.run_slate(conn, "S1", "lock10", now=T(G1) - timedelta(minutes=10), refresh=False, n=3)
    used = set()
    for d in runner.current_lineups(conn, "S1"):
        used |= set(d["player_ids"])
    out = next(str(i) for i in (3001, 3002, 3003, 3004, 3005, 3006, 3007) if str(i) not in used)
    capture(conn, "S1", payload([G1, G2, G3], status={int(out): "O"}), at="2026-10-21T02:00:00Z")
    now = T(G3) - timedelta(minutes=10)
    r = runner.run_slate(conn, "S1", "swap", now=now, refresh=False)
    assert r["status"] == "skipped_no_news" and r["late_news"] >= 1
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups WHERE mode='swap'").fetchone()[0] == 0
    row = conn.execute("SELECT ok, kind, detail, late_news_json FROM dfs_runs WHERE kind='swap'"
                       ).fetchone()
    assert row[0] == 1 and "skipped" in json.loads(row[2]) and json.loads(row[3])
    # recorded, so the next tick of the window does not look again
    assert runner.run_slate(conn, "S1", "swap", now=now, refresh=False)["status"] == "already_done"
    # --all re-solves the saved lineups regardless
    r = runner.run_slate(conn, "S1", "swap", now=now, refresh=False, force=True, resolve_all=True)
    assert r["status"] == "ok" and r["lineups"]["swap"] == len(runner.current_lineups(conn, "S1"))


def test_showdown_has_no_late_swap(conn, pipe, avail):
    capture(conn, "SD", J("draftables_showdown.json"), at="2026-10-20T12:00:00Z", tag="showdown",
            gt=81, rules=RULES81)
    r = runner.run_slate(conn, "SD", "swap", now=T("2026-10-20T20:00:00Z"), refresh=False)
    assert r["status"] == "nothing_to_swap"


def test_tick_runs_swap_end_to_end(conn, slate3, pipe, avail, monkeypatch):
    monkeypatch.setattr(runner, "refresh_inputs", lambda c, n: {})
    monkeypatch.setattr(runner.dfs_capture, "refresh_slate", lambda c, s, **k: {})
    runner.tick(conn, T(G1) - timedelta(minutes=10))        # builds the lock10 lineups
    cash = [d for d in runner.current_lineups(conn, "S1") if d["mode"] == "cash"][0]
    pool = R.build_pool(conn, "S1")
    start_of = {p.dk_player_id: p.start for p in pool.players}
    victim = next(pid for pid in cash["player_ids"] if start_of[pid] == T(G2))
    capture(conn, "S1", payload([G1, G2, G3], status={int(victim): "O"}), at="2026-10-20T23:30:00Z")
    res = runner.tick(conn, T(G2) - timedelta(minutes=10))
    assert [(r["kind"], r["action"], r["result"]["status"]) for r in res] == \
        [("swap", "ran", "ok")]
    assert victim not in [d for d in runner.current_lineups(conn, "S1")
                          if d["mode"] == "cash"][0]["player_ids"]


# --------------------------------------------------------------------------- CLI

def test_cli_run_and_tick(conn, tmp_path, slate3, pipe, avail, capsys):
    # db.connect() in the CLI reads $COURTSIDE_DATA/courtside.db: copy our slate into it
    from courtside import config
    c2 = db.connect()
    capture(c2, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
    c2.close()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="sub_cmd", required=True)
    handlers = runner_cli.add_commands(sub)
    assert set(handlers) == {"run", "tick"}
    args = ap.parse_args(["tick", "--now", "2026-10-20T22:15:00Z", "--dry-run"])
    assert handlers["tick"](args) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert (out["slate_id"], out["kind"], out["action"]) == ("S1", "lock45", "would_run")
    args = ap.parse_args(["tick", "--now", "2026-10-20T20:00:00Z"])
    assert handlers["tick"](args) == 0 and capsys.readouterr().out == ""      # nothing due: silent
    args = ap.parse_args(["run", "--slate", "S1", "--kind", "lock45", "--no-refresh", "--n", "2",
                          "--now", "2026-10-20T22:15:00Z"])
    assert handlers["run"](args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert handlers["run"](args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_done"
    assert config.data_dir().exists()


# --------------------------------------------------------------------------- lineup sim ranges

def write_sims(conn, run_id, dk_ids, fp, tmp_path):
    """Store a synthetic sims npz and point dfs_runs(run_id).sim_path at it."""
    import numpy as np
    from courtside.proj import store
    path = store.save_sims("S", run_id, np.arange(len(dk_ids)) + 1, fp, dk_id=dk_ids,
                           root=tmp_path / "sims")
    conn.execute("UPDATE dfs_runs SET sim_path=? WHERE run_id=?", (str(path), run_id))
    conn.commit()
    return path


def sim_matrix(dk_ids, n=2000, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    mu = 12 + 2.5 * np.arange(len(dk_ids))
    return (mu + 6 * rng.standard_normal((n, len(dk_ids)))).clip(0).astype(np.float16)


def test_run_fills_lineup_sim_stats(conn, slate3, avail, tmp_path, monkeypatch):
    import numpy as np
    pool = R.build_pool(conn, "S1")
    ids = [p.dk_player_id for p in pool.players]
    fp = sim_matrix(ids)
    pipe = Pipe(None)
    store_to = {}

    def project_slate(c, slate_id, at, *, run_id, **kw):
        write_run = pipe.project_slate(c, slate_id, at, run_id=run_id)
        O.save_run(c, slate_id=slate_id, kind="manual", ok=True, run_id=run_id,
                   created_at=runner.iso(at))
        store_to["path"] = write_sims(c, run_id, ids, fp, tmp_path)
        return write_run
    mod = types.ModuleType("courtside.proj.pipeline")
    mod.project_slate = project_slate
    mod.MODEL_VERSION = "stub-1"
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", mod)
    r = runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False,
                         n=3)
    assert r["status"] == "ok"
    rows = conn.execute("SELECT slots_json, mean, p10, p50, p90, why_json FROM dfs_lineups "
                        "WHERE run_id=?", (r["run_id"],)).fetchall()
    assert len(rows) == 4
    col = {d: i for i, d in enumerate(ids)}
    for slots_json, mean, p10, p50, p90, why_json in rows:
        slots = json.loads(slots_json)
        tot = sum(fp[:, col[s["player_id"]]].astype(np.float32) for s in slots)
        assert mean == pytest.approx(float(tot.mean()), rel=1e-5)
        assert (p10, p50, p90) == pytest.approx(tuple(np.percentile(tot, [10, 50, 90])), rel=1e-5)
        assert p10 < p50 < p90
        assert json.loads(why_json)["sim"] == {"sd": pytest.approx(float(tot.std()), abs=0.01),
                                               "n_sims": 2000}
    detail = json.loads(conn.execute("SELECT detail FROM dfs_runs WHERE run_id=?",
                                     (r["run_id"],)).fetchone()[0])
    assert detail["lineup_sims"] == 4


def test_lineup_totals_showdown_captain_is_x1_5():
    import numpy as np
    fp = np.array([[10, 20, 30], [20, 40, 60]], np.float16)
    sims = runner.RunSims(fp, ["a", "b", "c"])
    slots = [{"slot": "CPT", "player_id": "a"}, {"slot": "UTIL", "player_id": "b"},
             {"slot": "UTIL", "player_id": "c"}]
    assert sims.lineup_totals(slots).tolist() == [15 + 20 + 30, 30 + 40 + 60]
    st = runner.lineup_stats(sims.lineup_totals(slots))
    assert st["mean"] == pytest.approx(97.5) and st["p50"] == pytest.approx(97.5)
    assert st["p10"] < st["p50"] < st["p90"]
    # a player without a column: no stats rather than a lineup that is short a player
    assert sims.lineup_totals(slots + [{"slot": "UTIL", "player_id": "zz"}]) is None


def test_annotate_showdown_lineup_uses_captain_multiplier(conn, tmp_path):
    import numpy as np
    ids = ["1", "2", "3", "4", "5", "6"]
    fp = sim_matrix(ids, n=500)
    O.save_run(conn, slate_id="SD", kind="manual", ok=True, run_id="r1",
               created_at="2026-10-20T10:00:00Z")
    write_sims(conn, "r1", ids, fp, tmp_path)
    slots = [{"slot": "CPT" if i == "3" else "UTIL", "player_id": i} for i in ids]
    conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, mean, why_json) "
                 "VALUES ('r1','cash',1,?,1.0,?)", (json.dumps(slots), json.dumps({"from": "x"})))
    assert runner.annotate_lineups(conn, "r1") == 1
    mean, p10, p90, why = conn.execute("SELECT mean, p10, p90, why_json FROM dfs_lineups"
                                       ).fetchone()
    f = fp.astype(np.float32)
    want = f.sum(axis=1) + 0.5 * f[:, 2]
    assert mean == pytest.approx(float(want.mean()), rel=1e-5)
    assert p10 == pytest.approx(float(np.percentile(want, 10)), rel=1e-5)
    assert json.loads(why)["from"] == "x" and "sd" in json.loads(why)["sim"]   # merged, not lost


def test_fppg_fallback_leaves_lineup_stats_null(conn, slate3, avail, monkeypatch):
    monkeypatch.setitem(sys.modules, "courtside.proj.pipeline", None)     # import fails
    r = runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False,
                         n=2)
    assert r["status"] == "ok" and r["model_version"] == runner.FALLBACK_VERSION
    rows = conn.execute("SELECT mean, p10, p50, p90, why_json FROM dfs_lineups").fetchall()
    assert rows and all(x[0] is not None and x[1:4] == (None, None, None) for x in rows)
    assert all(x[4] is None or "sim" not in json.loads(x[4]) for x in rows)


def test_missing_sim_file_does_not_fail_the_run(conn, slate3, pipe, avail):
    # a dfs_runs row pointing at a file that is not there: stats stay NULL, the run is ok
    orig = pipe.project_slate

    def ps(c, slate_id, at, *, run_id, **kw):
        out = orig(c, slate_id, at, run_id=run_id)
        O.save_run(c, slate_id=slate_id, kind="manual", ok=True, run_id=run_id,
                   created_at=runner.iso(at))
        c.execute("UPDATE dfs_runs SET sim_path='/nonexistent/x.npz' WHERE run_id=?", (run_id,))
        return out
    pipe.project_slate = ps
    sys.modules["courtside.proj.pipeline"].project_slate = ps
    r = runner.run_slate(conn, "S1", "lock45", now=T(G1) - timedelta(minutes=45), refresh=False,
                         n=2)
    assert r["status"] == "ok"
    assert conn.execute("SELECT COUNT(*) FROM dfs_lineups WHERE p10 IS NOT NULL").fetchone()[0] == 0


def test_backfill_lineup_ranges(conn, tmp_path):
    import numpy as np
    ids = ["11", "22", "33"]
    fp = sim_matrix(ids, n=300)
    O.save_run(conn, slate_id="S", kind="manual", ok=True, run_id="r1",
               created_at="2026-10-20T10:00:00Z")
    O.save_run(conn, slate_id="S", kind="manual", ok=True, run_id="r2",
               created_at="2026-10-20T11:00:00Z")                  # no sims: untouched
    write_sims(conn, "r1", ids, fp, tmp_path)
    slots = json.dumps([{"slot": "UTIL", "player_id": i} for i in ids])
    for rid in ("r1", "r2"):
        conn.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, salary, mean) "
                     "VALUES (?,?,?,?,?,?)", (rid, "cash", 1, slots, 1, 1.0))
    conn.commit()
    assert runner.backfill_lineup_ranges(conn) == {"runs": 1, "lineups": 1, "skipped": 0}
    r1 = conn.execute("SELECT mean, p10, p50, p90 FROM dfs_lineups WHERE run_id='r1'").fetchone()
    assert r1[0] == pytest.approx(float(fp.astype(np.float32).sum(axis=1).mean()), rel=1e-5)
    assert r1[1] < r1[2] < r1[3]
    assert conn.execute("SELECT p10 FROM dfs_lineups WHERE run_id='r2'").fetchone()[0] is None
    assert runner.backfill_lineup_ranges(conn) == {"runs": 0, "lineups": 0, "skipped": 0}
