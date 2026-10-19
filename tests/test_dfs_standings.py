"""T3.2: DK standings importer, owner entries, real P&L, ownership dataset (synthetic files)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from dfs_results_helpers import (CONTEST, DAY, LOCK, NAMES, SD_SLATE, SLATE, TABLE, dk_id,
                                 lineup_text, nba_id, player_block, populate, standings_csv,
                                 ten_entries)

from courtside import db
from courtside.proj.dfs import grade
from courtside.proj.dfs import standings as ST

FNAME = f"contest-standings-{CONTEST}.csv"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    populate(c)
    return c


def count(conn, table, where="1=1", args=()):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]


# ------------------------------------------------------------------ parsing

def test_parse_lineup_tricky_names():
    names = ["Jaren Jackson Jr.", "Nikola Jokić", "P.J. Washington", "De'Aaron Fox",
             "Karl-Anthony Towns", "Alperen Şengün", "Kelly Oubre Jr.", "Luka Dončić"]
    text = " ".join(f"{s} {n}" for s, n in zip(ST.CLASSIC_SLOTS, names))
    got = ST.parse_lineup(text)
    assert [n for _, n in got] == names
    assert [s for s, _ in got] == list(ST.CLASSIC_SLOTS)
    assert ST.parse_lineup("LOCKED") is None and ST.parse_lineup("") is None
    assert ST.parse_lineup("Joel Embiid C") is None            # does not start with a slot
    sd = ST.parse_lineup("CPT Luka Doncic UTIL LeBron James UTIL Nic Claxton")
    assert sd[0] == ("CPT", "Luka Doncic") and len(sd) == 3


def test_parse_standings_blocks_and_quirks():
    ents = [(1, 11, "a", 0, "321.5", "LOCKED"), (2, 12, "b (1/3)", 0, 300, lineup_text(range(8))),
            ("", 13, "noname", 0, "", "")]
    players = [("Joel Embiid", "C", "45.20%", "63.5"), ("Nikola Jokić", "C", "12.00%", "")]
    st = ST.parse_standings(standings_csv(ents, players), "contest-standings-123456.csv")
    assert st.contest_id == "123456" and len(st.entries) == 2 and st.locked == 1
    assert st.skipped == 1 and not st.live
    assert st.entries[0].lineup is None and st.entries[0].points == 321.5
    assert [p.pct_drafted for p in st.players] == [45.2, 12.0]
    assert st.players[1].fpts is None
    # quoted field with a comma, accents in cp1252, BOM
    raw = standings_csv([(1, 5, 'x, "y"', 0, 10, lineup_text(range(8)))]).decode()
    st = ST.parse_standings(("\ufeff" + raw).encode("utf-8"), "f.csv")
    assert st.entries[0].entry_name == 'x, "y"'
    old = standings_csv([(1, 5, "Jos\u00e9", 0, 10, "C Nikola Joki\u0107")]).decode()
    st = ST.parse_standings(old.encode("cp1252", "replace"), "f.csv")     # not utf-8: cp1252
    assert st.entries[0].entry_name == "Jos\u00e9"
    # fractions, no percent sign
    st = ST.parse_standings(standings_csv(ten_entries(), [("A B", "C", "0.25", "1")]), "f.csv")
    assert st.players[0].pct_drafted == 25.0
    with pytest.raises(ST.StandingsError):
        ST.parse_standings(b"a,b\n1,2\n", "x.csv")
    with pytest.raises(ST.StandingsError, match="No ranked entries"):
        ST.parse_standings(standings_csv([("", 1, "a", 0, "", "LOCKED")]), "x.csv")


# ------------------------------------------------------------------ import

def test_import_basic_contest(conn, data_dir):
    data = standings_csv(ten_entries(), player_block())
    res = ST.import_upload(conn, data, FNAME, owner_names=[])
    assert res["status"] == "ok" and res["contest_id"] == CONTEST and res["entries"] == 10
    assert res["slate_id"] == SLATE and res["slate_how"] == "contest was captured"
    assert res["ownership_rows"] == len(NAMES) and res["unmatched"] == []
    # archive first: the raw bytes are in var/raw/draftkings/uploads, named by sha
    files = list((data_dir / "raw" / "draftkings" / "uploads").glob("*"))
    assert len(files) == 1 and files[0].read_bytes() == data
    # results: DK points and ranks; payouts from the captured payout table; no entry names stored
    rows = conn.execute("SELECT rank, points, payout, entry_hash, lineup_text FROM "
                        "dfs_contest_results WHERE contest_id=? ORDER BY rank", (CONTEST,)).fetchall()
    assert [r["points"] for r in rows] == [300 - 10 * k for k in range(10)]
    assert [r["payout"] for r in rows] == [100, 20, 20, 6, 6, 0, 0, 0, 0, 0]
    assert len({r["entry_hash"] for r in rows}) == 10
    assert not any("user0" in (r["lineup_text"] or "") for r in rows)
    dump = " ".join(str(tuple(r)) for r in conn.execute("SELECT * FROM dfs_contest_results"))
    assert "user0" not in dump
    # ownership: percent points keyed by NBA id; the FPTS
    o = conn.execute("SELECT * FROM dfs_contest_ownership WHERE contest_id=? AND player_id=?",
                     (CONTEST, nba_id(0))).fetchone()
    assert o["pct_drafted"] == 60.0 and o["fpts"] == 30.0 and o["name"] == "Joel Embiid"
    assert count(conn, "dfs_contest_ownership") == len(NAMES)
    # derived numbers
    s = ST.contest_summary(conn, CONTEST)
    assert s["field_size"] == 10 and s["winning_score"] == 300 and s["top1_score"] == 300
    assert s["cash_line"] == 260 and s["cash_line_source"] == "payout table"   # 5th place
    assert s["paid_places"] == 5 and s["median_score"] == 255
    assert s["ownership"]["top"][0]["name"] == "Joel Embiid" and s["ownership"]["players"] == 18
    assert ST.list_contests(conn)[0]["contest_id"] == CONTEST


def test_grade_reads_the_stored_cash_line(conn):
    ST.import_upload(conn, standings_csv(ten_entries(), player_block()), FNAME, owner_names=[])
    [c] = grade.contest_results(conn, SLATE)
    assert c["contest_id"] == CONTEST and c["cash_line"] == 260 and c["field_size"] == 10
    assert list(c["points"][:2]) == [300, 290]
    # the lineup-level fields the grader derives from it
    cash, per = grade._cash_fields(281.0, [c], {"mode": "cash", "contest_id": None})
    assert cash["cash_line_source"] == "real" and cash["would_cash"] == 1
    assert cash["est_rank"] == 3 and cash["est_payout"] == 20.0


def test_cash_line_without_payout_table_is_unknown(tmp_path):
    c = db.connect(tmp_path / "n.db")
    populate(c, payout=False)
    ST.import_upload(c, standings_csv(ten_entries(), player_block()), FNAME, owner_names=[])
    assert ST.contest_summary(c, CONTEST)["cash_line"] is None
    assert grade.contest_results(c, SLATE)[0]["cash_line"] is None      # nothing invented
    # the table arrives later: payouts are re-derived
    c.execute("UPDATE dfs_contests SET payout_json=?", (json.dumps(TABLE),))
    c.commit()
    assert ST.refresh_contest_payouts(c, CONTEST) == 10
    assert grade.contest_results(c, SLATE)[0]["cash_line"] == 260


def test_duplicate_upload_is_a_noop(conn, data_dir):
    data = standings_csv(ten_entries(), player_block())
    first = ST.import_upload(conn, data, FNAME, owner_names=[])
    again = ST.import_upload(conn, data, "renamed-" + FNAME, owner_names=[])
    assert again["status"] == "duplicate" and again["upload_id"] == first["upload_id"]
    assert count(conn, "dfs_uploads") == 1 and count(conn, "dfs_contest_results") == 10
    # a newer file for the same contest replaces the rows (latest wins), both uploads are kept
    newer = standings_csv(ten_entries()[:8], player_block())
    ST.import_upload(conn, newer, FNAME, owner_names=[])
    assert count(conn, "dfs_uploads") == 2 and count(conn, "dfs_contest_results") == 8
    assert len(list((data_dir / "raw" / "draftkings" / "uploads").glob("*"))) == 2


def test_contest_without_a_captured_row_infers_the_slate(tmp_path):
    c = db.connect(tmp_path / "i.db")
    populate(c, contest=False)
    res = ST.import_upload(c, standings_csv(ten_entries(), player_block()),
                           "contest-standings-777000111.csv", owner_names=[])
    assert res["slate_id"] == SLATE and res["slate_how"] == "inferred from the players"
    row = c.execute("SELECT slate_id, field_size FROM dfs_contests WHERE contest_id='777000111'"
                    ).fetchone()
    assert row["slate_id"] == SLATE and row["field_size"] == 10
    assert [x["contest_id"] for x in grade.contest_results(c, SLATE)] == ["777000111"]
    # a file of players in no slate: imported, but unlinked and reported
    odd = player_block()
    odd = [("Nobody Atall%d" % i, "C", "1.0%", "1") for i in range(20)]
    ents = [(r, 9000 + r, f"u{r}", 0, 100 - r, "C Nobody Atall0 F Nobody Atall1") for r in range(1, 4)]
    res = ST.import_upload(c, standings_csv(ents, odd), "contest-standings-888000222.csv",
                           owner_names=[])
    assert res["slate_id"] is None and len(res["unmatched"]) == 20
    # ... until the owner links it
    ST.link_contest_slate(c, "888000222", SLATE)
    assert c.execute("SELECT slate_id FROM dfs_contests WHERE contest_id='888000222'"
                     ).fetchone()[0] == SLATE


def test_unmatched_players_are_reported_not_stored(conn):
    pb = player_block() + [("Mystery Guest", "C", "0.5%", "10")]
    res = ST.import_upload(conn, standings_csv(ten_entries(), pb), FNAME, owner_names=[])
    assert res["unmatched"] == [{"name": "Mystery Guest", "pct": 0.5}]
    assert res["ownership_rows"] == len(NAMES)


def test_accented_names_match_the_pool(conn):
    # the file has plain-ASCII spellings of accented DK names
    pb = [("Nikola Jokic", "C", "30.0%", "55.0"), ("Luka Doncic", "PG", "25.0%", "50.0"),
          ("Alperen Sengun", "C", "5.0%", "30.0")]
    res = ST.import_upload(conn, standings_csv(ten_entries(), pb), FNAME, owner_names=[])
    assert res["unmatched"] == [] and res["ownership_rows"] == 3


def test_showdown_contest(tmp_path):
    c = db.connect(tmp_path / "s.db")
    populate(c, contest=False)
    ents = [(1, 21, "a", 0, 250.5, "CPT Joel Embiid UTIL LeBron James UTIL Tyrese Maxey "
                                  "UTIL Nikola Jokić UTIL Jaren Jackson Jr. UTIL P.J. Washington"),
            (2, 22, "b", 0, 200.0, "CPT LeBron James UTIL Joel Embiid UTIL Tyrese Maxey "
                                  "UTIL Nikola Jokić UTIL Jaren Jackson Jr. UTIL P.J. Washington")]
    pl = [("Joel Embiid", "CPT", "40.0%", "90.0"), ("Joel Embiid", "UTIL", "55.0%", "60.0"),
          ("LeBron James", "UTIL", "50.0%", "40.0"), ("LeBron James", "CPT", "10.0%", "60.0"),
          ("Tyrese Maxey", "UTIL", "30.0%", "30.0"), ("Nikola Jokić", "UTIL", "30.0%", "30.0"),
          ("Jaren Jackson Jr.", "UTIL", "30.0%", "30.0"), ("P.J. Washington", "UTIL", "30.0%", "30.0")]
    res = ST.import_upload(c, standings_csv(ents, pl), "contest-standings-555000333.csv",
                           owner_names=[])
    assert res["game_type"] == "showdown" and res["slate_id"] == SD_SLATE
    o = c.execute("SELECT pct_drafted FROM dfs_contest_ownership WHERE player_id=?",
                  (nba_id(0),)).fetchone()
    assert o[0] == 55.0                                        # the UTIL row is stored
    det = json.loads(c.execute("SELECT detail FROM dfs_uploads").fetchone()[0])
    assert det["cpt_pct_drafted"][str(nba_id(0))]["pct"] == 40.0
    assert count(c, "dfs_contest_ownership") == 6


def test_live_and_idless_files_are_refused(conn):
    live = [(1, 1, "a", 120, 50, lineup_text(range(8)))] + ten_entries()[1:3]
    with pytest.raises(ST.StandingsError, match="still in progress"):
        ST.import_upload(conn, standings_csv(live, player_block()), FNAME, owner_names=[])
    with pytest.raises(ST.StandingsError, match="contest id"):
        ST.import_upload(conn, standings_csv(ten_entries(), player_block()), "standings.csv",
                         owner_names=[])
    res = ST.import_upload(conn, standings_csv(ten_entries(), player_block()), "standings.csv",
                           contest_id="123456789", owner_names=[])
    assert res["contest_id"] == "123456789"
    with pytest.raises(ST.StandingsError, match="empty"):
        ST.import_upload(conn, b"", "x.csv")
    with pytest.raises(ST.StandingsError, match="Unrecognised"):
        ST.import_upload(conn, b"hello,world\n1,2\n", "x.csv")
    assert count(conn, "dfs_contest_results", "contest_id=?", (CONTEST,)) == 0   # refused: no rows


def test_zip_of_standings(conn):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(FNAME, standings_csv(ten_entries(), player_block()))
        z.writestr("notes.txt", "ignored")
    res = ST.import_upload(conn, buf.getvalue(), "standings.zip", owner_names=[])
    assert res["kind"] == "zip" and res["members"][0]["contest_id"] == CONTEST
    assert count(conn, "dfs_contest_results") == 10
    assert ST.import_upload(conn, buf.getvalue(), "standings.zip")["status"] == "duplicate"
    with pytest.raises(ST.StandingsError, match="damaged"):
        ST.import_upload(conn, b"PKnot a zip", "x.zip")


# ------------------------------------------------------------------ the owner

def test_owner_detection_by_username(conn):
    data = standings_csv(ten_entries(), player_block())
    res = ST.import_upload(conn, data, FNAME, owner_names=["NHorto"])        # case-insensitive
    assert res["owner_entries"] == 1
    e = conn.execute("SELECT * FROM dfs_entries").fetchone()
    assert e["contest_id"] == CONTEST and e["slate_id"] == SLATE and e["source"] == "upload"
    assert e["rank"] == 4 and e["points"] == 270 and e["field_size"] == 10
    assert e["entry_fee"] == 3.0 and e["payout"] == 6.0                     # table payout, 4th place
    assert e["dk_entry_id"] == "4600000004" and "nhorto" in e["note"]
    # the lineup equals our stored one -> linked
    lid = conn.execute("SELECT id FROM dfs_lineups").fetchone()[0]
    assert e["lineup_id"] == lid
    slots = json.loads(e["slots_json"])
    assert slots[0]["slot"] == "PG" and slots[0]["name"] == "Joel Embiid"
    assert slots[0]["player_id"] == dk_id(0) and slots[0]["nba_player_id"] == nba_id(0)
    # nobody else's name is kept anywhere
    names = " ".join(str(tuple(r)) for r in conn.execute("SELECT * FROM dfs_entries"))
    assert "user0" not in names
    # real P&L
    r = ST.my_results(conn)
    t = r["totals"]
    assert t["entries"] == 1 and t["fees"] == 3.0 and t["winnings"] == 6.0 and t["net"] == 3.0
    assert t["roi"] == 1.0 and t["cashed"] == 1 and t["pending"] == 0
    assert r["entries"][0]["percentile"] == pytest.approx(100 * 3.5 / 10)


def test_owner_username_from_env_or_file(conn, monkeypatch, data_dir):
    assert ST.owner_usernames() == []
    ST.set_owner_username("nhorto")
    assert ST.owner_usernames() == ["nhorto"]
    monkeypatch.setenv("COURTSIDE_DK_USERNAME", "alpha, beta")
    assert ST.owner_usernames() == ["alpha", "beta"]
    monkeypatch.delenv("COURTSIDE_DK_USERNAME")
    res = ST.import_upload(conn, standings_csv(ten_entries(), player_block()), FNAME)
    assert res["owner_entries"] == 1 and res["owner_configured"]            # from the file


def test_no_username_then_same_file_with_one(conn):
    data = standings_csv(ten_entries(), player_block())
    res = ST.import_upload(conn, data, FNAME, owner_names=[])
    assert res["owner_entries"] == 0 and not res["owner_configured"]
    again = ST.import_upload(conn, data, FNAME, owner_names=["nhorto"])     # duplicate file
    assert again["status"] == "duplicate" and again["owner_entries"] == 1
    assert count(conn, "dfs_entries") == 1 and count(conn, "dfs_uploads") == 1
    ST.import_upload(conn, data, FNAME, owner_names=["nhorto"])             # idempotent
    assert count(conn, "dfs_entries") == 1


def test_manual_entry_merge_and_winnings(conn):
    eid = ST.add_manual_entry(conn, contest_id=CONTEST, entry_fee=3.0, points=270, rank=4,
                              lineup_text=lineup_text(range(8)))
    e = conn.execute("SELECT * FROM dfs_entries WHERE entry_id=?", (eid,)).fetchone()
    assert e["source"] == "manual" and e["payout"] is None and e["slate_id"] == SLATE
    assert e["lineup_id"] is not None                                        # equals our lineup
    assert ST.my_results(conn)["totals"]["pending"] == 1
    ST.update_entry(conn, eid, payout=6.0)
    assert ST.my_results(conn)["totals"]["net"] == 3.0
    # the standings file arrives: the manual entry is claimed, not duplicated
    ST.import_upload(conn, standings_csv(ten_entries(), player_block()), FNAME,
                     owner_names=["nhorto"])
    rows = conn.execute("SELECT * FROM dfs_entries").fetchall()
    assert len(rows) == 1 and rows[0]["entry_id"] == eid and rows[0]["dk_entry_id"] == "4600000004"
    assert rows[0]["source"] == "manual" and rows[0]["payout"] == 6.0
    for bad in (dict(contest_id=""), dict(contest_id="1", entry_fee=-1),
                dict(contest_id="1", lineup_text="nonsense")):
        with pytest.raises(ST.StandingsError):
            ST.add_manual_entry(conn, **bad)
    pasted = "\n".join(NAMES[:8])
    e2 = ST.add_manual_entry(conn, contest_id="42", lineup_text=pasted, entry_fee=5)
    assert len(json.loads(conn.execute("SELECT slots_json FROM dfs_entries WHERE entry_id=?",
                                       (e2,)).fetchone()[0])) == 8


def test_manual_entry_from_one_of_our_lineups(conn):
    lid = conn.execute("SELECT id FROM dfs_lineups").fetchone()[0]
    eid = ST.add_manual_entry(conn, contest_id=CONTEST, lineup_id=lid, entry_fee=3, payout=0,
                              points=210, rank=10)
    e = conn.execute("SELECT * FROM dfs_entries WHERE entry_id=?", (eid,)).fetchone()
    assert e["lineup_id"] == lid and e["slate_id"] == SLATE and e["payout"] == 0
    t = ST.my_results(conn)["totals"]
    assert t["net"] == -3.0 and t["roi"] == -1.0 and t["cashed"] == 0


def test_history_csv(conn):
    hist = ("Sport,Game_Type,Entry_Key,Entry,Contest_Key,Contest_Date_EST,Place,Points,"
            "Winnings_Non_Ticket,Winnings_Ticket,Contest_Entries,Entry_Fee,Prize_Pool,Places_Paid\r\n"
            f"NBA,Classic,4600000004,NBA $20K Test (2/20),{CONTEST},2026-10-20 19:00:00,4,270.0,"
            "$6.00,,10,$3.00,$200.00,5\r\n"
            "NFL,Classic,9,NFL thing,12,2026-10-18 13:00:00,1,100,$0.00,,5,$1.00,$5.00,2\r\n"
            "NBA,Classic,4600000099,NBA Other,777,2026-10-20 19:00:00,50,180.5,$0.00,,100,$5.00,"
            "$400.00,20\r\n").encode()
    res = ST.import_upload(conn, hist, "contest-history.csv", owner_names=[])
    assert res["kind"] == "history" and res["entries"] == 2 and res["skipped"] == 1
    rows = {r["dk_entry_id"]: r for r in conn.execute("SELECT * FROM dfs_entries")}
    assert rows["4600000004"]["payout"] == 6.0 and rows["4600000004"]["entry_fee"] == 3.0
    assert rows["4600000004"]["slate_id"] == SLATE and rows["4600000099"]["payout"] == 0.0
    t = ST.my_results(conn)["totals"]
    assert t["entries"] == 2 and t["fees"] == 8.0 and t["net"] == -2.0
    assert ST.import_upload(conn, hist, "again.csv")["status"] == "duplicate"


# ------------------------------------------------------------------ ownership dataset

def test_ownership_dataset_joins_pre_lock_inputs(conn):
    # a lock10 projection run for the slate, made before lock
    for i, name in enumerate(NAMES):
        cur = conn.execute(
            "INSERT INTO player_projections (game_id, nba_day, slate_id, player_id, run_id, "
            "created_at, model_version, p_play) VALUES ('G0', ?, ?, ?, 'run1', "
            "'2026-10-20T22:50:00Z', 'pathA-v0.1', 0.95)", (DAY, SLATE, nba_id(i)))
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean, sd) "
                     "VALUES (?, 'dk', 'all', ?, 7.0)", (cur.lastrowid, 25.0 + i))
        conn.execute("INSERT INTO player_projection_stats (projection_id, stat, cond, mean, sd) "
                     "VALUES (?, 'min', 'all', ?, 5.0)", (cur.lastrowid, 20.0 + i))
    conn.commit()
    empty = ST.ownership_dataset(conn)
    assert len(empty) == 0 and "pct_drafted" in empty.columns
    ST.import_upload(conn, standings_csv(ten_entries(), player_block()), FNAME, owner_names=[])
    df = ST.ownership_dataset(conn)
    assert len(df) == len(NAMES) and set(df["contest_id"]) == {CONTEST}
    r = df[df["player_id"] == nba_id(2)].iloc[0]
    assert r["pct_drafted"] == 54.0 and r["fpts"] == 32.0
    assert r["salary"] == 3600 and r["fppg"] == 22.0 and r["dk_id"] == dk_id(2)
    assert r["proj_dk"] == 27.0 and r["proj_min"] == 22.0 and r["p_play"] == 0.95
    assert r["run_kind"] == "lock10" and r["pool_size"] == len(NAMES)
    assert r["salary_rank"] == len(NAMES) - 2 and r["field_size"] == 10
    assert r["value_proj"] == pytest.approx(27.0 / 3.6)
    assert r["game_type"] == "classic" and r["nba_day"] == DAY
