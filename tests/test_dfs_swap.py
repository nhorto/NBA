"""Late-swap re-solver (proj/dfs/swap.py, T2.14): locked slots never change, flex slots, the DK
entries CSV in and the DK edit CSV out, portfolio checks, the runner's loop, timing, CLI.

Offline. Slates are synthetic draftables with DK's shape (as in test_dfs_runner.py) or the real
2026-10-19 main-slate capture (tests/fixtures/dfs/real_slates_20261019.json.gz)."""

from __future__ import annotations

import copy
import csv
import gzip
import io
import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from courtside import db
from courtside.proj.dfs import export as X
from courtside.proj.dfs import optimizer as O
from courtside.proj.dfs import rules as R
from courtside.proj.dfs import runner, swap
from courtside.proj.dfs import topn as TN
from courtside.sources import dfs_capture as cap

FIX = Path(__file__).parent / "fixtures" / "dfs"
DAY = date(2026, 10, 20)
RULES70 = json.loads((FIX / "rules_70.json").read_text())
RULES81 = json.loads((FIX / "rules_81.json").read_text())
CLASSIC = R.Rules.from_payload(RULES70)
G1, G2, G3 = ("2026-10-20T23:00:00.0000000Z", "2026-10-21T01:00:00.0000000Z",
              "2026-10-21T03:00:00.0000000Z")
POS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF"]
TEAMS = [("BOS", "DET"), ("PHI", "NYK"), ("OKC", "SAS"), ("LAL", "DEN")]


def T(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def pid_of(gi: int, side: int, k: int) -> str:
    return str(1000 * (gi + 1) + 100 * side + k + 1)


def payload(starts, *, status=None, disabled=()):
    """len(starts) games x 14 players, one draftable per eligible slot (DK's layout)."""
    slot_id = {s["rosterSlot"]["name"]: s["rosterSlot"]["id"] for s in RULES70["lineupTemplate"]}
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
                pid = int(pid_of(gi, side, k))
                salary = 3500 + 700 * ((k * 3 + gi + side) % 8)
                fppg = 14.0 + 3.2 * ((k * 5 + gi * 2 + side) % 9) + salary / 2000
                for slot in CLASSIC.eligible_slots(pos):
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
                        "teamAbbreviation": team, "playerGameAttributes": []})
    return {"draftables": drafts, "competitions": comps}


def capture(conn, sid, pl, *, at, tag="main", gt=70, rules=RULES70):
    now = T(at)
    cap.store_draftables(conn, pl, slate_id=sid, game_type_id=gt, day=DAY, tag=tag, rules=rules,
                         lobby=None, start_utc=None, captured_at=cap.iso(now), now=now)


@pytest.fixture
def slate(conn):
    capture(conn, "S1", payload([G1, G2, G3]), at="2026-10-20T20:00:00Z")
    return "S1"


def fppg(pool):
    return {p.dk_player_id: p.fppg for p in pool.players if p.fppg is not None}


def entry_of(lu, key, *, contest=None, mode="topn"):
    pairs = lu.slot_pairs()
    return swap.SwapEntry(key=key, label=key, pairs=pairs, names=[e.player.name for e in
                                                                   lu.entries],
                          mode=None if contest else mode, contest_id=contest,
                          contest_name="NBA $5 Double Up" if contest == "cash1" else
                          ("NBA GPP" if contest else None), entry_id=key if contest else None)


def started_slots(entry, pool, now):
    by = pool.by_id()
    return {k: pid for k, (s, pid) in enumerate(entry.pairs) if by[pid].is_locked(now)}


def assert_locked_untouched(sp, pool, now):
    """The hard invariant, checked on every result and on the CSV rows."""
    by = pool.by_id()
    rows = list(csv.reader(io.StringIO(sp.to_csv(style="name_id"))))
    head = rows[0]
    assert head[:4] == list(X.EDIT_HEADER) and head[4:] == list(CLASSIC.slot_names)
    for r, row in zip(sp.results, rows[1:]):
        e = r.entry
        after = r.after_pairs if r.status == "swapped" else e.pairs
        for k, (slot, pid) in enumerate(e.pairs):
            p = by.get(pid)
            if p is not None and p.is_locked(now) or k in e.dk_locked:
                assert after[k] == (slot, pid), (e.key, slot, pid, after[k])
                assert r.slots[k].fixed and not r.slots[k].changed
                if e.cells:
                    assert row[4 + k] == e.cells[k]          # DK's own cell, byte for byte
                else:
                    assert f"({p.draftable_ids[slot]})" in row[4 + k]
        if r.status == "swapped":
            fixed = {k: pid for k, (_, pid) in enumerate(e.pairs)
                     if (by.get(pid) and by[pid].is_locked(now)) or k in e.dk_locked}
            swap.check_entry(CLASSIC, sp.by_id, e.pairs, fixed, after,
                             started={p for p in sp.by_id if sp.by_id[p].is_locked(now)}
                             - set(fixed.values()), bans=set(sp.ruled_out) - set(fixed.values()))


# =========================================================================== locked slots

def test_started_players_never_change_random(conn, slate):
    """Many random lineups, Out players, projection shocks and clock times: every started
    player stays in his exact slot, nobody else from a started game comes in, ruled-out players
    leave, and every swapped lineup is legal."""
    pool = R.build_pool(conn, slate)
    base = fppg(pool)
    rnd = random.Random(7)
    checked = swapped = no_legal = 0
    for trial in range(40):
        noisy = {k: v * rnd.uniform(0.6, 1.4) for k, v in base.items()}
        res = TN.top_n(pool, noisy, CLASSIC, 4, min_diff=2)
        entries = [entry_of(r.lineup, f"t{trial}-{r.rank}",
                            contest=rnd.choice([None, "gpp1", "cash1"])) for r in res]
        now = rnd.choice([T(G1), T(G1) + timedelta(minutes=30), T(G2) + timedelta(seconds=1)])
        open_ids = sorted({pid for e in entries for _, pid in e.pairs
                           if not pool.by_id()[pid].is_locked(now)})
        fixed_ids = sorted({pid for e in entries for _, pid in e.pairs
                            if pool.by_id()[pid].is_locked(now)})
        out = {pid: "test: out" for pid in rnd.sample(open_ids, min(2, len(open_ids)))}
        if fixed_ids and trial % 3 == 0:          # a started player ruled out: still fixed
            out[rnd.choice(fixed_ids)] = "test: out after tip"
        shock = {k: v * rnd.uniform(0.5, 1.5) for k, v in base.items()}
        sp = swap.plan(conn, slate, now=now, entries=entries, projections=shock,
                       extra_out=out, min_gain=rnd.choice([0.0, 0.5]),
                       max_exposure=rnd.choice([None, 0.75]))
        assert_locked_untouched(sp, pool, now)
        for r in sp.results:
            after = r.after_pairs if r.status == "swapped" else r.entry.pairs
            fixed_ids_e = {pid for k, pid in started_slots(r.entry, pool, now).items()}
            if r.status == "no_legal":                 # nothing proposed, entry kept as is
                no_legal += 1
                continue
            for _, pid in after:
                if pid in out and pid not in fixed_ids_e:
                    pytest.fail(f"ruled-out {pid} still in an open slot ({r.status})")
            checked += 1
            swapped += r.status == "swapped"
        assert not [w for w in sp.warnings if "safety check" in w]
    assert checked >= 120 and swapped >= 40 and no_legal <= 30


def test_early_player_in_flex_slot_stays_in_that_flex_slot(conn, slate):
    """Game-1 players sitting in UTIL, G and F stay there, even with their natural slot (PG)
    open and better for the lineup."""
    pool = R.build_pool(conn, slate)
    by = pool.by_id()
    projs = fppg(pool)
    pg1 = pid_of(0, 0, 0)          # BOS PG, game 1
    g1 = pid_of(0, 1, 5)           # DET PG/SG, game 1
    f1 = pid_of(0, 0, 6)           # BOS SF/PF, game 1
    pg3 = pid_of(2, 0, 0)          # OKC PG, game 3
    lu = O.optimize(pool, projs, CLASSIC, locks={pg1: "UTIL", g1: "G", f1: "F", pg3: "PG"})
    e = entry_of(lu, "flex")
    now = T(G1) + timedelta(minutes=5)
    shock = dict(projs)
    shock[pg1] = 99.0                      # moving him to PG would look great; it is not allowed
    sp = swap.plan(conn, slate, now=now, entries=[e], projections=shock,
                   extra_out={pg3: "test: ruled out"}, min_gain=0)
    r = sp.results[0]
    assert r.status == "swapped"
    after = dict((s, p) for s, p in r.after_pairs)
    assert after["UTIL"] == pg1 and after["G"] == g1 and after["F"] == f1
    assert after["PG"] != pg1 and after["PG"] != pg3
    assert not by[after["PG"]].is_locked(now) and "PG" in by[after["PG"]].slots
    assert [s.fixed for s in r.slots] == [k in started_slots(e, pool, now)
                                          for k in range(8)]
    assert_locked_untouched(sp, pool, now)
    pg_slot = next(s for s in r.slots if s.slot == "PG")
    assert pg_slot.reason == "test: ruled out" and pg_slot.changed


def test_positional_flex_moves_only_unlocked_players(conn, slate):
    """A later-game PG/SG in G must move to PG (the only PG left), a SG takes G; locked slots
    do not move and no locked player is used to fill the hole."""
    pool = R.build_pool(conn, slate)
    by = pool.by_id()
    projs = fppg(pool)
    x = pid_of(1, 0, 5)                    # PHI PG/SG, game 2
    pg_out = pid_of(2, 1, 0)               # SAS PG, game 3
    pg1 = pid_of(0, 0, 0)                  # BOS PG, game 1 (started): eligible but locked out
    lu = O.optimize(pool, projs, CLASSIC, locks={x: "G", pg_out: "PG"})
    e = entry_of(lu, "move")
    now = T(G1) + timedelta(minutes=1)
    # every other open PG-eligible player is ruled out, so PG can only be X
    others = {p.dk_player_id: "test: out" for p in pool.players
              if "PG" in p.slots and not p.is_locked(now) and p.dk_player_id != x}
    assert pg_out in others
    sp = swap.plan(conn, slate, now=now, entries=[e], projections=projs, extra_out=others,
                   min_gain=0)
    r = sp.results[0]
    assert r.status == "swapped", r.notes
    after = dict(r.after_pairs)
    assert after["PG"] == x                               # moved G -> PG
    assert after["G"] != x and "SG" in by[after["G"]].positions
    assert pg1 not in after.values() or pg1 in dict(e.pairs).values()
    g_slot = next(s for s in r.slots if s.slot == "G")
    assert g_slot.changed and g_slot.reason == "Moved to PG"
    fixed = started_slots(e, pool, now)
    for k, pid in fixed.items():
        assert r.after_pairs[k] == e.pairs[k]
    assert_locked_untouched(sp, pool, now)


def test_unchanged_players_keep_their_seat():
    """The re-seat step undoes a solver shuffle of players who stay."""
    rules = CLASSIC
    mk = lambda pid, pos: R.PoolPlayer(  # noqa: E731
        pid, None, pid, "T" + pid[-1], tuple(pos.split("/")), 4000, rules.eligible_slots(pos),
        {}, {}, "g" + pid[-1], None, None, None, None, None, None, None, None, None, None)
    ps = {p.dk_player_id: p for p in [mk("a1", "PG/SG"), mk("b2", "PG/SG"), mk("c3", "SF"),
                                      mk("d4", "PF"), mk("e5", "C"), mk("f6", "PG"),
                                      mk("g7", "SF/PF"), mk("h8", "C")]}
    before = [("PG", "a1"), ("SG", "b2"), ("SF", "c3"), ("PF", "d4"), ("C", "e5"), ("G", "x9"),
              ("F", "g7"), ("UTIL", "h8")]
    shuffled = [("PG", "f6"), ("SG", "a1"), ("SF", "c3"), ("PF", "d4"), ("C", "e5"),
                ("G", "b2"), ("F", "g7"), ("UTIL", "h8")]
    out = swap._keep_slots(rules, ps, {2: "c3"}, before, shuffled)
    assert out[0] == ("PG", "a1") and out[1] == ("SG", "b2") and out[5] == ("G", "f6")
    assert out[2] == ("SF", "c3")


def test_dk_locked_cell_is_fixed_before_our_clock_says_so(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    now = T(G1) - timedelta(minutes=30)            # nothing has tipped by our clock
    cells = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    cells[3] += " (LOCKED)"
    data = dk_file([("111", "NBA GPP", "c9", "$5", cells)], pool)
    out = {e.player.dk_player_id: "test" for e in lu.entries}      # all open players out
    sp = swap.plan(conn, slate, now=now, csv_data=data, projections=projs, extra_out=out)
    r = sp.results[0]
    assert r.slots[3].fixed and r.slots[3].fixed_why == "DraftKings shows it locked"
    assert r.status == "swapped" and r.after_pairs[3] == r.entry.pairs[3]
    assert sum(s.changed for s in r.slots) == 7
    row = list(csv.reader(io.StringIO(sp.to_csv())))[1]
    assert row[4 + 3] == cells[3]


def test_dk_in_progress_game_fixes_its_players_and_bans_the_rest(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    now = T(G1) - timedelta(minutes=2)             # DK says game 1 is under way; our clock not
    g1_ids = {p.dk_player_id for p in pool.players if p.start == T(G1)}
    cells = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    data = dk_file([("222", "NBA GPP", "c9", "$5", cells)], pool, in_progress=g1_ids)
    out = {pid: "test" for _, pid in lu.slot_pairs() if pid not in g1_ids}
    sp = swap.plan(conn, slate, now=now, csv_data=data, projections=projs, extra_out=out)
    r = sp.results[0]
    for k, (_, pid) in enumerate(r.entry.pairs):
        if pid in g1_ids:
            assert r.slots[k].fixed and r.after_pairs[k] == r.entry.pairs[k]
    new = {pid for _, pid in r.after_pairs} - {pid for _, pid in r.entry.pairs}
    assert new and not (new & g1_ids)
    assert sp.games[0]["started"] and not sp.games[1]["started"]


def test_fixed_player_dk_greyed_out_after_tip_stays(conn, slate):
    """A started player whose draftables DK disabled later is still pinned (the pool has lost
    him, the stored rows have not)."""
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    e = entry_of(lu, "grey")
    now = T(G1) + timedelta(minutes=10)
    gone = next(pid for _, pid in e.pairs if pool.by_id()[pid].start == T(G1))
    capture(conn, "S1", payload([G1, G2, G3], disabled={int(gone)}), at="2026-10-20T23:05:00Z")
    assert gone not in R.build_pool(conn, slate).by_id()
    victim = next(pid for _, pid in e.pairs if pool.by_id()[pid].start == T(G3))
    sp = swap.plan(conn, slate, now=now, entries=[e], projections=projs,
                   extra_out={victim: "test"})
    r = sp.results[0]
    k = [p for _, p in e.pairs].index(gone)
    assert r.status == "swapped" and r.after_pairs[k] == e.pairs[k] and r.slots[k].fixed
    assert victim not in {p for _, p in r.after_pairs}
    assert f"({lu.entries[k].draftable_id})" in list(csv.reader(io.StringIO(sp.to_csv())))[1][4 + k]


def test_all_locked_and_no_legal_keep_the_entry(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    e = entry_of(lu, "late")
    sp = swap.plan(conn, slate, now=T(G3) + timedelta(minutes=1), entries=[e], projections=projs)
    assert sp.results[0].status == "all_locked" and sp.results[0].after_pairs == e.pairs
    # every open player is ruled out: nothing legal can fill the open slots
    now = T(G2) + timedelta(minutes=1)
    out = {p.dk_player_id: "test" for p in pool.players if not p.is_locked(now)}
    sp = swap.plan(conn, slate, now=now, entries=[e], projections=projs, extra_out=out)
    r = sp.results[0]
    assert r.status == "no_legal" and r.after_pairs is None
    row = list(csv.reader(io.StringIO(sp.to_csv())))[1]
    assert row[4:] == [f"{x.player.name} ({x.draftable_id})" for x in lu.entries]


def test_scope_news_and_min_gain(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    a = O.optimize(pool, projs, CLASSIC)
    res = TN.top_n(pool, projs, CLASSIC, 3)
    es = [entry_of(r.lineup, f"n{r.rank}") for r in res]
    now = T(G1) + timedelta(minutes=1)
    # same projections, nothing out: re-solving changes nothing
    sp = swap.plan(conn, slate, now=now, entries=es, projections=projs)
    assert all(r.status == "unchanged" for r in sp.results)
    victim = next(pid for _, pid in a.slot_pairs() if not pool.by_id()[pid].is_locked(now))
    sp = swap.plan(conn, slate, now=now, entries=es, projections=projs, scope="news",
                   extra_out={victim: "test"})
    for r in sp.results:
        hit = victim in {p for _, p in r.entry.pairs}
        assert (r.status == "swapped") == hit
        if not hit:
            assert "No late news" in r.notes[0]
    # a small projection bump below min_gain does not churn an entry
    bump = dict(projs)
    alt = next(p.dk_player_id for p in pool.players
               if p.dk_player_id not in a.player_ids and not p.is_locked(now))
    bump[alt] += 0.3
    sp = swap.plan(conn, slate, now=now, entries=[entry_of(a, "g")], projections=bump,
                   min_gain=5.0)
    assert sp.results[0].status == "unchanged"


def test_showdown_is_refused(conn):
    capture(conn, "SD", json.loads((FIX / "draftables_showdown.json").read_text()),
            at="2026-10-20T12:00:00Z", tag="showdown", gt=81, rules=RULES81)
    with pytest.raises(swap.SwapError, match="Showdown"):
        swap.plan(conn, "SD", now=T("2026-10-20T20:00:00Z"))


# =========================================================================== portfolio

def test_identical_entries_stay_identical_distinct_stay_distinct(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    res = TN.top_n(pool, projs, CLASSIC, 30, min_diff=1)
    now = T(G1) + timedelta(minutes=1)
    a = res.lineups[0].lineup
    # b differs from a in open players only; those are ruled out, so both want the same fill
    b = next(r.lineup for r in res.lineups[1:]
             if all(not pool.by_id()[p].is_locked(now) for p in a.player_ids ^ r.lineup.player_ids))
    out = {pid: "test" for pid in a.player_ids ^ b.player_ids}
    es = [entry_of(a, "a1", contest="gpp1"), entry_of(a, "a2", contest="gpp1"),
          entry_of(b, "b1", contest="gpp1")]
    sp = swap.plan(conn, slate, now=now, entries=es, projections=projs, extra_out=out)
    r1, r2, r3 = sp.results
    p1 = r1.after_pairs if r1.status == "swapped" else es[0].pairs
    p2 = r2.after_pairs if r2.status == "swapped" else es[1].pairs
    p3 = r3.after_pairs if r3.status == "swapped" else es[2].pairs
    assert p1 == p2                                         # identical entries: one solve
    assert {x for _, x in p1} != {x for _, x in p3}         # distinct entries stay distinct
    g = sp.portfolio["groups"]["c:gpp1"]
    assert g["identical_after"] == 1                        # only the deliberate pair
    assert_locked_untouched(sp, pool, now)


def test_exposure_cap_counts_locked_parts(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    res = TN.top_n(pool, projs, CLASSIC, 6, min_diff=3)
    es = [entry_of(r.lineup, f"e{r.rank}") for r in res]
    now = T(G1) + timedelta(minutes=1)
    star = max((p for p in pool.players if not p.is_locked(now)), key=lambda p: projs[p.dk_player_id] / p.salary)
    boost = dict(projs)
    boost[star.dk_player_id] = 80.0                         # everyone wants him
    sp = swap.plan(conn, slate, now=now, entries=es, projections=boost, max_exposure=0.5,
                   min_gain=0)
    n_star = sum(star.dk_player_id in {p for _, p in (r.after_pairs or r.entry.pairs)}
                 for r in sp.results)
    assert n_star <= 3
    exp = {d["player_id"]: d for d in sp.portfolio["exposure"]}
    assert exp[star.dk_player_id]["after"] <= 0.5 + 1e-9
    # a locked player above the cap is reported, not "fixed"
    same = [swap.SwapEntry(key=f"x{i}", label=f"x{i}", pairs=list(es[0].pairs),
                           names=list(es[0].names), contest_id=f"c{i}") for i in range(4)]
    sp = swap.plan(conn, slate, now=now, entries=same, projections=projs, max_exposure=0.25)
    assert any("locked into 4 entries" in w for w in sp.warnings)
    assert_locked_untouched(sp, pool, now)


def test_max_overlap_between_entries_of_a_contest(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    res = TN.top_n(pool, projs, CLASSIC, 3, min_diff=1)
    es = [entry_of(r.lineup, f"o{r.rank}", contest="gpp1") for r in res]
    now = T("2026-10-20T20:30:00Z")                          # before any tip: all open
    sp = swap.plan(conn, slate, now=now, entries=es, projections=projs, max_overlap=5,
                   min_gain=-1e9)
    finals = [{p for _, p in (r.after_pairs if r.status == "swapped" else r.entry.pairs)}
              for r in sp.results]
    for i in range(3):
        for j in range(i + 1, 3):
            assert len(finals[i] & finals[j]) <= 5
    assert sp.portfolio["groups"]["c:gpp1"]["max_shared_after"] <= 5


# =========================================================================== DK files

def dk_file(rows, pool, *, in_progress=(), name_only=False, bare=False) -> str:
    """A DK entries CSV: 4 entry columns, the slots, a blank column, Instructions, then DK's
    player list further right (header on the 2nd row, like DK's own download)."""
    head = list(X.EDIT_HEADER) + list(CLASSIC.slot_names) + ["", "Instructions"]
    plist_head = ["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
                  "Game Info", "TeamAbbrev", "AvgPointsPerGame"]
    out = [head]
    for eid, cname, cid, fee, cells in rows:
        out.append([eid, cname, cid, fee] + list(cells) + ["", "1. Column A lists all of your "
                                                                "contest entries"])
    players = list(pool.players)
    out.append([""] * len(head) + plist_head)
    for p in players:
        did = next(iter(p.draftable_ids.values()))
        gi = "In Progress" if p.dk_player_id in in_progress else \
            f"{p.game_label.replace(' @ ', '@')} 10/20/2026 07:00PM ET"
        out.append([""] * len(head) + ["/".join(p.positions), f"{p.name} ({did})", p.name, did,
                                       "/".join(p.slots), str(p.salary), gi, p.team or "",
                                       str(p.fppg)])
    buf = io.StringIO()
    csv.writer(buf).writerows(out)
    return buf.getvalue()


def test_parse_dk_entries_csv_formats(conn, slate):
    pool = R.build_pool(conn, slate)
    idx = swap.slate_index(conn, slate)
    lu = O.optimize(pool, fppg(pool), CLASSIC)
    named = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    bare = [str(e.draftable_id) for e in lu.entries]
    names = [e.player.name for e in lu.entries]
    other_slot = [f"{e.player.name} ({next(iter(e.player.draftable_ids.values()))})"
                  for e in lu.entries]                       # any of his draftable ids
    text = dk_file([("1", "NBA $5 Double Up", "c1", "$5", named),
                    ("2", "NBA GPP", "c2", "$3", bare),
                    ("3", "NBA GPP", "c2", "$3", names),
                    ("4", "NBA GPP", "c2", "$3", other_slot),
                    ("5", "NBA GPP", "c2", "$3", [""] * 8),
                    ("6", "NBA GPP", "c2", "$3", ["Nobody Here (999999)"] + named[1:])], pool)
    pf = swap.parse_entries_csv(text.encode("utf-8-sig"), idx, CLASSIC)
    assert [e.entry_id for e in pf.entries] == ["1", "2", "3", "4", "5", "6"]
    want = lu.slot_pairs()
    for e in pf.entries[:4]:
        assert e.pairs == want, e.entry_id
    assert pf.entries[0].contest_name == "NBA $5 Double Up" and pf.entries[0].entry_fee == "$5"
    assert pf.formats["name_id"] >= 16 and pf.formats["id"] == 8 and pf.formats["name"] == 8
    assert all(pid is None for _, pid in pf.entries[4].pairs)
    assert pf.entries[5].problems and "not on this slate" in pf.entries[5].problems[0]
    assert len(pf.dk_ids) == len(pool.players)
    assert swap.objective_for("NBA $5 Double Up") == "cash"
    assert swap.objective_for("NBA $20K Shootaround") == "mean"
    # the plan: an empty entry is filled, an unknown cell is left alone
    sp = swap.plan(conn, slate, now=T(G1) + timedelta(minutes=1), csv_data=text,
                   projections=fppg(pool))
    st = {r.entry.entry_id: r.status for r in sp.results}
    assert st["6"] == "unusable" and st["5"] == "swapped"
    assert sp.results[0].objective == "cash" and sp.results[1].objective == "mean"
    rows = list(csv.reader(io.StringIO(sp.to_csv())))
    assert rows[6][4:] == ["Nobody Here (999999)"] + named[1:]
    assert all(c for c in rows[5][4:])
    assert_locked_untouched(sp, pool, T(G1) + timedelta(minutes=1))


def test_parse_rejects_other_files(conn, slate):
    idx = swap.slate_index(conn, slate)
    with pytest.raises(swap.SwapError, match="Entry ID"):          # a standings file
        swap.parse_entries_csv("Rank,EntryId,Points\n1,2,3\n", idx, CLASSIC)
    with pytest.raises(swap.SwapError, match="do not match"):
        swap.parse_entries_csv("Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,SF\n"
                               "1,a,b,c,x,y,z\n", idx, CLASSIC)
    with pytest.raises(swap.SwapError, match="No entries"):
        swap.parse_entries_csv(",".join(X.EDIT_HEADER + tuple(CLASSIC.slot_names)) + "\n",
                               idx, CLASSIC)
    sd = "Entry ID,Contest Name,Contest ID,Entry Fee,CPT,UTIL,UTIL,UTIL,UTIL,UTIL\n1,a,b,c,,,,,,\n"
    pf = swap.parse_entries_csv(sd, idx, R.Rules.from_payload(RULES81))
    assert pf.showdown
    with pytest.raises(swap.SwapError, match="Showdown"):
        swap.plan(conn, slate, csv_data=sd)


def test_edit_csv_cells_and_styles(conn, slate):
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    cells = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    text = dk_file([("77", "NBA GPP", "c2", "$3", cells)], pool)
    now = T(G2) + timedelta(minutes=1)
    victim = next(pid for _, pid in lu.slot_pairs() if not pool.by_id()[pid].is_locked(now))
    sp = swap.plan(conn, slate, now=now, csv_data=text, projections=projs,
                   extra_out={victim: "test"})
    r = sp.results[0]
    assert r.status == "swapped"
    pf = swap.parse_entries_csv(text, swap.slate_index(conn, slate), CLASSIC)
    for style in ("name_id", "id"):
        rows = list(csv.reader(io.StringIO(sp.to_csv(style=style))))
        assert rows[0] == list(X.EDIT_HEADER) + list(CLASSIC.slot_names)
        assert rows[1][:4] == ["77", "NBA GPP", "c2", "$3"]
        for k, s in enumerate(r.slots):
            if not s.changed:
                assert rows[1][4 + k] == cells[k]
            else:
                did = pf.dk_ids[s.after["player_id"]]               # DK's own ID for him
                assert rows[1][4 + k] == (f"{s.after['name']} ({did})" if style == "name_id"
                                          else did)
    # the written file reads back as the swapped lineup
    back = swap.parse_entries_csv(sp.to_csv(), swap.slate_index(conn, slate), CLASSIC)
    assert [p for _, p in back.entries[0].pairs] == [p for _, p in r.after_pairs]
    with pytest.raises(X.ExportError):
        X.to_edit_csv([{"entry_id": "1", "slots": [{"cell": "x"}]}], header=["PG", "SG"])


def test_saved_lineups_path_reads_current_lineups(conn, slate):
    """Our saved lineups (the latest build, swap runs overlaid), objective by mode."""
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    rid = O.save_run(conn, slate_id=slate, kind="lock10", ok=True, detail={},
                     created_at="2026-10-20T22:50:00Z")
    O.save_lineups(conn, rid, "cash", [O.optimize(pool, projs, CLASSIC, mode="cash")])
    O.save_lineups(conn, rid, "topn", TN.top_n(pool, projs, CLASSIC, 3))
    es = swap.saved_entries(conn, slate)
    assert [e.label for e in es] == ["Cash lineup", "Top N #1", "Top N #2", "Top N #3"]
    now = T(G1) + timedelta(minutes=1)
    victim = next(pid for _, pid in es[0].pairs if not pool.by_id()[pid].is_locked(now))
    sp = swap.plan(conn, slate, now=now, projections=projs, extra_out={victim: "test"})
    assert sp.source == "saved" and sp.results[0].objective == "cash"
    assert {r.objective for r in sp.results[1:]} == {"mean"}
    rows = list(csv.reader(io.StringIO(sp.to_csv())))
    assert len(rows) == 5 and rows[1][0] == ""              # no DK entry ids for saved lineups
    assert_locked_untouched(sp, pool, now)


# =========================================================================== runner layer

def _legacy_resolve(O_, pool, projs, rules, targets, bans, now, min_diff):
    """runner._resolve_swaps as deployed before the move (verbatim)."""
    by_id = pool.by_id()
    started = pool.locked_ids(now)
    free_bans = set(bans) - started
    produced = {"cash": [], "topn": []}
    out = []
    for t in targets:
        pairs = [(s, pid) for s, pid in t["pairs"] if pid in by_id]
        for s, pid in pairs:
            if pid in started and pid not in projs:
                p = by_id[pid]
                projs[pid] = O_.Projection(p.fppg if p.fppg is not None else 0.0)
        mode = "cash" if t["mode"] == "cash" else "mean"
        lu = None
        for cuts in (True, False):
            model = O_.model_for(pool, projs, rules, mode=mode, lam=0.3, bans=free_bans,
                                 now=now, existing=pairs, time_limit_s=20.0)
            if cuts:
                for prev in produced[t["mode"]]:
                    model.add_no_good(prev, min_diff)
            lu = model.solve()
            if lu is not None:
                break
        if lu is None:
            out.append({"target": t, "lineup": None})
            continue
        produced[t["mode"]].append(lu)
        before = set(t["player_ids"])
        out.append({"target": t, "lineup": lu,
                    "in": [by_id[i].name for i in sorted(lu.player_ids - before)],
                    "out": [by_id[i].name for i in sorted(before - lu.player_ids)]})
    return out


def test_runner_loop_is_unchanged(conn, slate):
    pool = R.build_pool(conn, slate)
    base = {k: O.Projection(v, p25=0.8 * v) for k, v in fppg(pool).items()}
    rnd = random.Random(3)
    for trial in range(12):
        projs = {k: O.Projection(v.mean * rnd.uniform(0.7, 1.3), p25=v.p25)
                 for k, v in base.items()}
        res = TN.top_n(pool, projs, CLASSIC, 4)
        cash = O.optimize(pool, projs, CLASSIC, mode="cash")
        targets = [{"mode": "cash", "rank": 1, "id": 0, "pairs": cash.slot_pairs(),
                    "player_ids": [p for _, p in cash.slot_pairs()]}] + [
            {"mode": "topn", "rank": r.rank, "id": r.rank, "pairs": r.lineup.slot_pairs(),
             "player_ids": [p for _, p in r.lineup.slot_pairs()]} for r in res]
        now = rnd.choice([T(G1) + timedelta(minutes=1), T(G2) + timedelta(minutes=1)])
        bans = set(rnd.sample(sorted(p.dk_player_id for p in pool.players), 6))
        shock = {k: O.Projection(v.mean * rnd.uniform(0.8, 1.2), p25=v.p25)
                 for k, v in projs.items()}
        a = swap.resolve_targets(pool, copy.deepcopy(shock), CLASSIC, targets, bans, now, 1)
        b = _legacy_resolve(O, pool, copy.deepcopy(shock), CLASSIC, targets, bans, now, 1)
        assert [(x["lineup"].slot_pairs() if x["lineup"] else None, x.get("in"), x.get("out"))
                for x in a] == \
               [(x["lineup"].slot_pairs() if x["lineup"] else None, x.get("in"), x.get("out"))
                for x in b]
    assert runner._resolve_swaps.__module__ == runner.__name__   # still callable from runner


# =========================================================================== timing + CLI

def test_timing_on_the_real_main_slate(conn):
    """150 distinct entries on the real 125-player main slate, two games started: <= 30 s."""
    data = json.loads(gzip.decompress((FIX / "real_slates_20261019.json.gz").read_bytes()))
    db.insert_rows(conn, "dfs_salaries", data["154575"])
    conn.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, tag, rules_json) "
                 "VALUES ('154575','2026-10-20',70,'main',?)", (json.dumps(RULES70),))
    conn.commit()
    pool = R.build_pool(conn, "154575")
    projs = fppg(pool)
    res = TN.top_n(pool, projs, CLASSIC, 150, min_diff=1, total_time_limit_s=120)
    assert len(res) >= 100
    es = [entry_of(r.lineup, f"r{r.rank}", contest="gpp1") for r in res]
    tips = sorted({p.start for p in pool.players if p.start})
    now = tips[1] + timedelta(minutes=1)
    rnd = random.Random(1)
    late = [p.dk_player_id for p in pool.players if not p.is_locked(now)]
    shock = {k: v * rnd.uniform(0.8, 1.2) for k, v in projs.items()}
    out = {pid: "test" for pid in rnd.sample(late, 3)}
    t = time.perf_counter()
    sp = swap.plan(conn, "154575", now=now, entries=es, projections=shock, extra_out=out)
    secs = time.perf_counter() - t
    print(f"\nswap timing: {len(es)} entries, {sp.n_swapped} swapped, {secs:.2f} s")
    assert secs <= 30.0
    assert not [r for r in sp.results if "budget ran out" in " ".join(r.notes)]
    assert_locked_untouched_real(sp, pool, now)


def assert_locked_untouched_real(sp, pool, now):
    by = pool.by_id()
    for r in sp.results:
        after = r.after_pairs if r.status == "swapped" else r.entry.pairs
        for k, (slot, pid) in enumerate(r.entry.pairs):
            if by[pid].is_locked(now):
                assert after[k] == (slot, pid)


def test_cli_swap_writes_an_edit_csv(conn, slate, tmp_path, capsys, monkeypatch):
    from courtside import config
    from courtside.proj import cli
    pool = R.build_pool(conn, slate)
    projs = fppg(pool)
    lu = O.optimize(pool, projs, CLASSIC)
    cells = [f"{e.player.name} ({e.draftable_id})" for e in lu.entries]
    f = tmp_path / "DKEntries.csv"
    f.write_text(dk_file([("55", "NBA GPP", "c2", "$3", cells)], pool))
    conn.close()
    db_path = config.data_dir() / "courtside.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c2 = db.connect(db_path)
    capture(c2, "S1", payload([G1, G2, G3], status={int(lu.slot_pairs()[7][1]): "O"}),
            at="2026-10-20T20:00:00Z")
    c2.close()
    out = tmp_path / "edit.csv"
    rc = cli.dfs_main(["swap", "--slate", "S1", "--entries", str(f), "--now",
                       "2026-10-20T23:30:00Z", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "1 entries" in text and out.exists()
    rows = list(csv.reader(io.StringIO(out.read_text())))
    assert rows[0][:4] == list(X.EDIT_HEADER) and rows[1][0] == "55"
    rc = cli.dfs_main(["swap", "--slate", "S1", "--now", "2026-10-20T23:30:00Z", "--json",
                       "--out", str(tmp_path / "x.csv")])
    assert rc == 1                                          # no saved lineups yet
