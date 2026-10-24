"""E5: contest-history slate inference (standings.import_history), the per-slate
real-standings calibration report (proj/dfs/real_calib.py), its /health/dfs card, and the
field.calibration_report argument-order fix. Synthetic DB rows only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dfs_results_helpers import (CONTEST, DAY, SLATE, player_block, populate, standings_csv,
                                 ten_entries)

from courtside import db
from courtside.proj.dfs import field as F
from courtside.proj.dfs import real_calib as RC
from courtside.proj.dfs import standings as ST

HDR = ("Sport,Game_Type,Entry_Key,Entry,Contest_Key,Contest_Date_EST,Place,Points,"
       "Winnings_Non_Ticket,Winnings_Ticket,Contest_Entries,Entry_Fee,Prize_Pool,Places_Paid\r\n")
FIX = Path(__file__).parent / "fixtures" / "dfs"


def _slates(c):
    with c:
        for sid, day, gt, lock in (("500", "2026-10-23", 70, "2026-10-23T23:00:00Z"),
                                   ("501", "2026-10-23", 70, "2026-10-24T01:30:00Z"),
                                   ("502", "2026-10-23", 81, "2026-10-23T23:00:00Z"),
                                   ("503", "2026-10-23", 81, "2026-10-24T01:30:00Z"),
                                   ("504", "2026-10-23", 81, "2026-10-24T01:30:00Z"),
                                   ("505", "2027-01-06", 70, "2027-01-07T00:00:00Z")):
            c.execute("INSERT INTO dfs_slates (slate_id, nba_day, game_type_id, lock_utc) "
                      "VALUES (?,?,?,?)", (sid, day, gt, lock))


# ============================================================================== slate inference

def test_history_lock_utc_is_eastern_wall_time():
    assert ST.history_lock_utc("2026-10-23 19:00:00") == ("2026-10-23", "2026-10-23T23:00:00Z")
    assert ST.history_lock_utc("2027-01-06 19:00:00") == ("2027-01-06", "2027-01-07T00:00:00Z")
    assert ST.history_lock_utc("2026-10-20 15:00:00") == ("2026-10-20", "2026-10-20T19:00:00Z")
    assert ST.history_lock_utc("garbage") == (None, None)
    assert ST.history_lock_utc("") == (None, None)


def test_infer_history_slate(tmp_path):
    c = db.connect(tmp_path / "s.db")
    _slates(c)
    assert ST.infer_history_slate(c, "2026-10-23 19:00:00", "Classic")[0] == "500"
    assert ST.infer_history_slate(c, "2026-10-23 21:30:00", "Classic")[0] == "501"
    assert ST.infer_history_slate(c, "2026-10-23 19:00:00", "Showdown Captain Mode")[0] == "502"
    sid, how = ST.infer_history_slate(c, "2026-10-23 21:30:00", "Showdown Captain Mode")
    assert sid is None and "2 slates" in how                    # two showdowns: no guess
    assert ST.infer_history_slate(c, "2026-10-23 20:00:00", "Classic")[0] is None
    assert ST.infer_history_slate(c, "2027-01-06 19:00:00", "Classic")[0] == "505"   # EST
    assert ST.infer_history_slate(c, "2026-10-23 19:00:00", "Tiers")[0] is None


def _hist(rows):
    return (HDR + "".join(rows)).encode()


def test_import_history_links_slates_and_keeps_places_paid(tmp_path, data_dir):
    c = db.connect(tmp_path / "h.db")
    _slates(c)
    with c:     # a captured contest keeps its own slate even if the time says otherwise
        c.execute("INSERT INTO dfs_contests (contest_id, slate_id, captured_at) VALUES "
                  "('900', '501', 'x')")
    data = _hist([
        "NBA,Classic,11,NBA $2.1K mini-MAX (1/2),700,2026-10-23 19:00:00,14,317.75,$8.00,$0.00,"
        "5112,$0.50,\"$2,150.00\",1025\r\n",
        "NBA,Classic,12,NBA $2.1K mini-MAX (2/2),700,2026-10-23 19:00:00,1286,253.5,$0.00,$0.00,"
        "5112,$0.50,\"$2,150.00\",1025\r\n",
        "NBA,Classic,13,NBA Captured,900,2026-10-23 19:00:00,5,300,$5.00,$0.00,100,$1.00,$80.00,"
        "20\r\n",
        "NBA,Showdown Captain Mode,14,NBA Showdown,800,2026-10-23 21:30:00,9,200,$0.00,$0.00,"
        "100,$1.00,$80.00,20\r\n",
        "NBA,Classic,15,NBA Off-hours,801,2026-10-23 20:00:00,9,200,$0.00,$0.00,100,$1.00,"
        "$80.00,20\r\n"])
    res = ST.import_upload(c, data, "draftkings-contest-entry-history.csv", owner_names=[])
    assert res["status"] == "ok" and res["slates_inferred"] == {"700": "500"}
    sl = {r[0]: r[1] for r in c.execute("SELECT dk_entry_id, slate_id FROM dfs_entries")}
    assert sl == {"11": "500", "12": "500", "13": "501", "14": None, "15": None}
    # no contest rows are created (the grader's reference contests stay as they were)
    assert c.execute("SELECT COUNT(*) FROM dfs_contests").fetchone()[0] == 1
    det = json.loads(c.execute("SELECT detail FROM dfs_uploads WHERE kind='history'")
                     .fetchone()[0])
    assert det["contests"]["700"]["places_paid"] == 1025
    assert det["contests"]["700"]["name"] == "NBA $2.1K mini-MAX"
    meta = ST.history_contests(c)
    assert meta["700"]["entries"] == 5112 and meta["700"]["prize_pool"] == 2150.0


def test_history_contests_reads_old_uploads_from_the_archive(tmp_path):
    c = db.connect(tmp_path / "o.db")
    raw = tmp_path / "hist.csv"
    raw.write_text(HDR + "NBA,Classic,11,NBA X (1/3),700,2026-10-23 19:00:00,14,317.75,$8.00,"
                         "$0.00,5112,$0.50,$2150.00,1025\r\n")
    with c:      # an import from before 2026-10-24: no "contests" in the detail
        c.execute("INSERT INTO dfs_uploads (kind, sha256, raw_path, uploaded_at, detail) VALUES "
                  "('history', 'abc', ?, 'x', '{\"entries\": 1}')", (str(raw),))
    m = ST.history_contests(c)
    assert m["700"]["places_paid"] == 1025 and m["700"]["name"] == "NBA X"


def test_existing_history_test_slate_stays_ambiguous(tmp_path):
    """dfs_results_helpers has two Classic slates locking at 23:00Z on DAY (main and early):
    an uncaptured contest of that time is NOT linked."""
    c = db.connect(tmp_path / "a.db")
    populate(c)
    data = _hist([f"NBA,Classic,99,NBA Other,777,{DAY} 19:00:00,50,180.5,$0.00,,100,$5.00,"
                  "$400.00,20\r\n"])
    res = ST.import_upload(c, data, "h.csv", owner_names=[])
    assert res["slates_inferred"] == {}
    assert c.execute("SELECT slate_id FROM dfs_entries").fetchone()[0] is None


# ============================================================================== the report

def _world(tmp_path):
    """Helpers' slate 9001 with: an uploaded standings file (10 entries, owner rank 4 holds the
    stored cash lineup), and a history-only GPP with three of the owner's entries (two matched
    to stored lineups by points, one cashing, one not), plus the lineups' grades."""
    c = db.connect(tmp_path / "w.db")
    populate(c)
    ST.import_upload(c, standings_csv(ten_entries(), player_block()),
                     f"contest-standings-{CONTEST}.csv", owner_names=["nhorto"])
    lid = c.execute("SELECT id FROM dfs_lineups").fetchone()[0]
    with c:
        c.execute("UPDATE dfs_lineups SET actual=270.0 WHERE id=?", (lid,))
        for rank, pts, players in ((2, 281.5, range(1, 9)), (3, 253.5, range(2, 10))):
            slots = [{"slot": s, "player_id": str(5000 + i), "name": f"x{i}"}
                     for s, i in zip(ST.CLASSIC_SLOTS, players)]
            c.execute("INSERT INTO dfs_lineups (run_id, mode, rank, slots_json, actual) VALUES "
                      "('run1', 'topn', ?, ?, ?)", (rank, json.dumps(slots), pts))
    hist = _hist([
        f"NBA,Classic,21,NBA $20K And-One (1/3),888,{DAY} 19:00:00,222,281.5,$6.00,$0.00,23781,"
        "$1.00,\"$20,000.00\",6420\r\n",
        f"NBA,Classic,22,NBA $20K And-One (2/3),888,{DAY} 19:00:00,7000,253.5,$0.00,$0.00,23781,"
        "$1.00,\"$20,000.00\",6420\r\n",
        f"NBA,Classic,23,NBA $20K And-One (3/3),888,{DAY} 19:00:00,9000,240.0,$0.00,$0.00,23781,"
        "$1.00,\"$20,000.00\",6420\r\n"])
    ST.import_upload(c, hist, "draftkings-contest-entry-history.csv", owner_names=[])
    with c:     # two Classic slates lock at that time in the helpers: link by hand, as on 10-23
        c.execute("UPDATE dfs_entries SET slate_id=? WHERE contest_id='888'", (SLATE,))
    return c


def test_report_contests_lines_and_matching(tmp_path):
    c = _world(tmp_path)
    rep = RC.build_report(c, since="2026-10-01", until="2026-10-31", ownership=False)
    [b] = [b for b in rep["slates"] if b["slate_id"] == SLATE]
    cs = {x["contest_id"]: x for x in b["contests"]}
    std = cs[CONTEST]
    # payout table pays 5 places: the line is the 5th best score of the standings
    assert std["standings"] and std["cash_line"]["value"] == 260.0
    assert std["places_paid"] == 5 and std["top1_line"]["value"] == 300.0
    [own] = std["entries"]
    assert own["rank"] == 4 and own["cashed"] and own["matched_by"] == "lineup id"
    g = cs["888"]
    assert not g["standings"] and g["type"] == "large_gpp" and g["field"] == 23781
    assert g["places_paid"] == 6420 and g["places_how"] == "contest history"
    assert g["cash_line"] == {"lo": 253.5, "hi": 281.5,
                              "source": "bracketed by the owner's entries"}
    by_pts = {e["points"]: e for e in g["entries"]}
    assert by_pts[281.5]["matched_by"] == "points" and by_pts[281.5]["lineup"] == "lock10 topn #2"
    assert by_pts[240.0]["lineup_id"] is None
    assert by_pts[281.5]["beat"] == pytest.approx(1 - 221.5 / 23781)
    s = rep["summary"]
    assert s["gpp_cash_rate"]["mean"] == pytest.approx(2 / 4) and s["gpp_cash_rate"]["n"] == 1
    assert s["slates_with_standings"] == 1
    assert any("888" in t for t in rep["needs"])
    assert "(253.50, 281.50]" in RC.format_report(rep)


def test_ambiguous_points_are_not_matched():
    lus = [{"id": 1, "run_id": "a", "mode": "topn", "rank": 1, "actual": 250.0, "key": ("1",),
            "kind": "manual", "p_cash": None, "slots_json": None},
           {"id": 2, "run_id": "b", "mode": "topn", "rank": 1, "actual": 250.0, "key": ("2",),
            "kind": "lock10", "p_cash": None, "slots_json": None},
           {"id": 3, "run_id": "a", "mode": "topn", "rank": 2, "actual": 260.0, "key": ("3",),
            "kind": "manual", "p_cash": None, "slots_json": None}]
    es = [{"entry_id": 1, "points": 250.0}]
    RC.match_entries(es, lus)
    assert es[0]["lineup"] is None and es[0]["matched_by"].startswith("ambiguous")
    # with another entry explaining run "a", the tie goes to the run the owner played
    es = [{"entry_id": 1, "points": 250.0}, {"entry_id": 2, "points": 260.0}]
    RC.match_entries(es, lus)
    assert es[0]["lineup"]["id"] == 1 and "explains most" in es[0]["matched_by"]


def test_summary_unit_is_the_slate():
    def slate(sid, pits, cashed):
        ents = [{"cashed": x, "beat": 0.5} for x in cashed]
        return {"slate_id": sid, "size": "5+ games", "ownership": [], "nightly": None,
                "contests": [{"type": "cash", "standings": True, "entries": ents,
                              "sim": {"lineups": {str(i): {"pit_line": p, "p_cash": 0.6}
                                                  for i, p in enumerate(pits)}}}]}
    rep = {"slates": [slate("a", [0.1, 0.2, 0.3], [0, 0, 1]), slate("b", [0.9], [1])]}
    s = RC.summarize(rep)
    assert s["du_pit_line"]["n"] == 2
    assert s["du_pit_line"]["mean"] == pytest.approx((0.2 + 0.9) / 2)
    assert s["du_pit_line"]["se"] == pytest.approx(np.std([0.2, 0.9], ddof=1) / np.sqrt(2))
    assert s["du_cash_rate"]["mean"] == pytest.approx((1 / 3 + 1) / 2)


def test_contest_sims_pits(monkeypatch):
    """The line PIT and the place PIT from a stubbed field evaluation."""
    S = 1000
    rng = np.random.default_rng(0)
    score = np.round((250 + rng.normal(0, 20, S)) * 64) / 64     # on the 1/64-point grid
    line = 245 + rng.normal(0, 5, S)
    above = np.clip(np.round((260 - score) * 20), 0, 999).astype(np.float32)

    def evaluate(f, fp, contest, lineups, **kw):
        units = (score * F.UNIT * F.WUNIT + F.BIAS).astype(np.int64)
        return SimpleNamespace(
            per_sim={"score": units[:, None], "above": above[:, None],
                     "tied": np.zeros((S, 1), np.float32)},
            cash_line=line, lineups=[{"p_cash": float((score >= line).mean())}],
            cash_line_summary=lambda: {"cash_p50": float(np.median(line))})
    monkeypatch.setattr(F, "prepare", lambda *a, **k: {"sims": {"fp": np.zeros((S, 1)),
                                                               "dk_id": np.array(["1"])},
                                                       "inputs": None})
    monkeypatch.setattr(F, "field_for", lambda *a, **k: SimpleNamespace(pids=["1"]))
    from courtside.proj.dfs import portfolio as PF
    monkeypatch.setattr(PF, "truth_sims", lambda fp, *a, **k: (fp, {}))
    monkeypatch.setattr(F, "evaluate", evaluate)
    lu = {"id": 7, "run_id": "r", "actual": 255.0, "slots_json": "[]"}
    cb = {"contest_id": "1", "name": "x", "fee": 1.0, "field": 1000, "places_paid": 200,
          "type": "large_gpp", "cash_line": {"value": 250.0}}
    ent = [{"lineup": lu, "beat": 0.9}]
    out = RC.contest_sims(None, "s", cb, ent, [], n_sims=S, ctx_cache={})
    d = out["lineups"]["7"]
    assert d["pit_line"] == pytest.approx(((score - line) < 5.0).mean())
    rank = above + 1.0
    beat = 1 - (rank - 0.5) / 1000
    assert d["pit_rank"] == pytest.approx((beat < 0.9).mean() + 0.5 * (beat == 0.9).mean())
    cb2 = {**cb, "cash_line": {"lo": 240.0, "hi": 252.0}}
    d2 = RC.contest_sims(None, "s", cb2, ent, [], n_sims=S, ctx_cache={})["lineups"]["7"]
    assert d2["pit_line_lo"] == pytest.approx(((score - line) < 3.0).mean())
    assert d2["pit_line_hi"] == pytest.approx(((score - line) < 15.0).mean())
    assert d2["pit_line_lo"] <= d["pit_line"] <= d2["pit_line_hi"]


def test_write_load_and_health_card(tmp_path, data_dir):
    import warnings
    from datetime import date
    warnings.filterwarnings("ignore", message="Using `httpx`")
    from fastapi.testclient import TestClient
    from courtside.web import health_dfs as H
    from courtside.web.app import create_app
    c = _world(tmp_path)
    rep = RC.build_report(c, since="2026-10-01", until="2026-10-31", ownership=False)
    c.close()
    d = data_dir.parent / "web"
    d.mkdir(parents=True)
    conn = db.connect(d / "courtside.db")
    assert H.build(conn, date(2026, 10, 25))["real_calib"] == {}
    p = RC.write_report(rep, d)
    assert p.exists() and (p.parent / "real_calib_2026-10-31.json").exists()
    assert RC.load_report(conn)["to"] == "2026-10-31"
    rc = H.build(conn, date(2026, 10, 25))["real_calib"]
    [row] = rc["rows"]
    assert row["slate"] == SLATE and row["money"] == "2 of 4" and row["entries"] == 4
    assert row["line"] == "260.00"                  # an exact (standings) line comes first
    conn.close()
    r = TestClient(create_app(d)).get("/health/dfs")
    assert r.status_code == 200 and 'id="real-calib"' in r.text
    card = r.text.split('id="real-calib"')[1].split("</div>\n  {% endif")[0]
    assert "None" not in card.split("</table>")[0]


def test_cli_registered():
    from courtside.proj import cli
    assert "courtside.proj.dfs.real_calib" in cli.DFS_MODULES
    assert "courtside.proj.value_eval" in cli.PROJ_MODULES


# ============================================================================== field fix

def test_calibration_report_compares_the_proxy(tmp_path):
    """calibration_report passed (pool, inputs) to ownership_proxy(inputs, pool): every contest
    lost its proxy row ("no rules"). With the order fixed the proxy errors are reported."""
    c = db.connect(tmp_path / "r.db")
    populate(c)
    with c:
        c.execute("UPDATE dfs_slates SET rules_json=? WHERE slate_id=?",
                  ((FIX / "rules_70.json").read_text(), SLATE))
        c.execute("UPDATE dfs_salaries SET roster_slot=NULL WHERE slate_id=?", (SLATE,))
    ST.import_upload(c, standings_csv(ten_entries(), player_block()),
                     f"contest-standings-{CONTEST}.csv", owner_names=[])
    rep = F.calibration_report(c)
    [r] = rep["contests"]
    assert "proxy" in r and r["proxy"]["mae_pp"] >= 0 and -1 <= r["proxy"]["spearman"] <= 1
