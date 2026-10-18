import io
import json
import math
import zipfile
from pathlib import Path

import pytest

from courtside import db, odds_history as oh

FIX = Path(__file__).parent / "fixtures" / "odds_history"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def _game(conn, gid, day, home, away, season="2007-08", hp=100, ap=90, neutral=False,
          season_type="Regular Season"):
    rows = [
        {"game_id": gid, "team": home, "opp": away, "game_date": day, "season": season,
         "season_type": season_type, "is_home": 0 if neutral else 1, "pts": hp},
        {"game_id": gid, "team": away, "opp": home, "game_date": day, "season": season,
         "season_type": season_type, "is_home": 0, "pts": ap},
    ]
    db.upsert_rows(conn, "team_games", rows)


def _kaggle_rows():
    return list(oh.iter_kaggle_rows((FIX / "kaggle_sample.csv").read_text()))


# ---------------------------------------------------------------------------- Kaggle parsing

def test_kaggle_spread_sign_and_fields():
    r = _kaggle_rows()
    assert r[0]["home"] == "SAS" and r[0]["away"] == "POR"
    assert r[0]["home_spread_close"] == -13.0          # home favored -> negative
    assert (r[0]["home_ml_close"], r[0]["away_ml_close"]) == (-1400.0, 900.0)
    assert r[0]["total_close"] == 189.5 and r[0]["season"] == "2007-08"
    assert r[0]["provider"] == "sbr"
    assert r[1]["home"] == "GSW" and r[1]["away"] == "UTA"
    assert r[2]["home"] == "NOP" and r[2]["away"] == "BKN"   # no / nj aliases
    assert r[2]["home_spread_close"] == 2.5            # away favored -> positive
    assert r[3]["home_spread_close"] == 0.0 and math.copysign(1, r[3]["home_spread_close"]) == 1
    assert r[4]["home_ml_close"] is None and r[4]["away_ml_close"] is None
    assert r[4]["provider"] == "espnbet" and r[4]["home"] == "PHX" and r[4]["away"] == "WAS"
    assert r[5]["total_close"] is None and r[5]["away"] == "OKC" and r[5]["home"] == "CHA"


def test_norm_team_aliases():
    for src, dst in [("NJN", "BKN"), ("nj", "BKN"), ("NOH", "NOP"), ("NOK", "NOP"), ("sea", "OKC"),
                     ("utah", "UTA"), ("wsh", "WAS"), ("gs", "GSW"), ("ny", "NYK"), ("sa", "SAS"),
                     ("PHO", "PHX"), ("CHO", "CHA"), ("BOS", "BOS")]:
        assert oh.norm_team(src) == dst


# ---------------------------------------------------------------------------- ESPN parsing

def _load(name):
    return json.loads((FIX / name).read_text())


def test_espn_parse_open_close():
    p = oh.parse_espn_odds(_load("espn_odds_draftkings.json"))
    assert p["provider"] == "DraftKings"
    assert (p["home_spread_open"], p["home_spread_close"]) == (-2.5, -6.5)   # home-signed
    assert (p["total_open"], p["total_close"]) == (222.5, 225.5)
    assert (p["home_ml_open"], p["home_ml_close"]) == (-135.0, -218.0)
    assert (p["away_ml_open"], p["away_ml_close"]) == (114.0, 180.0)


def test_espn_parse_skips_live_and_juice_in_close_block():
    # ESPN BET's close.pointSpread/total hold juice (-105/-110): must fall back to `current`;
    # DraftKings (current only) is lower priority than a provider with a close block; the
    # 'Live Odds' provider (spread 10.5) is never chosen.
    p = oh.parse_espn_odds(_load("espn_odds_multi.json"))
    assert p["provider"] == "ESPN BET [current]"
    assert p["home_spread_close"] == -5.5 and p["total_close"] == 225.5
    assert (p["home_ml_close"], p["away_ml_close"]) == (-220.0, 170.0)
    assert p["home_spread_open"] is None and p["home_ml_open"] is None


def test_espn_only_live_or_empty_is_none():
    d = _load("espn_odds_multi.json")
    d["items"] = [i for i in d["items"] if "Live Odds" in i["provider"]["name"]]
    assert oh.parse_espn_odds(d) is None
    assert oh.parse_espn_odds({"count": 0, "items": []}) is None
    assert oh.parse_espn_odds({}) is None


# ---------------------------------------------------------------------------- matching

def test_match_exact_alias_tolerance_swap(conn):
    _game(conn, "g1", "2007-10-30", "SAS", "POR")
    _game(conn, "g2", "2007-11-02", "NOH", "NJN")        # historical tricodes in team_games
    _game(conn, "g3", "2024-01-10", "PHX", "WAS", season="2023-24")
    idx = oh.GameIndex(conn)
    assert idx.match("sa", "por", "2007-10-30") == ("g1", "exact", 0)
    assert idx.match("no", "bkn", "2007-11-02") == ("g2", "exact", 0)    # alias on both sides
    assert idx.match("phx", "wsh", "2024-01-11") == ("g3", "tolerance", -1)   # Kaggle +1 day
    assert idx.match("phx", "wsh", "2024-01-09") == ("g3", "tolerance", 1)
    assert idx.match("phx", "wsh", "2024-01-13")[0] is None
    assert idx.match("wsh", "phx", "2024-01-10") == ("g3", "swapped", 0)
    assert idx.match("lal", "bos", "2024-01-10") == (None, "no_game", 0)


def test_match_neutral_site_both_orientations(conn):
    _game(conn, "gp", "2025-01-23", "IND", "SAS", season="2024-25", neutral=True)
    idx = oh.GameIndex(conn)
    assert idx.match("ind", "sa", "2025-01-23")[:2] == ("gp", "exact")
    assert idx.match("sa", "ind", "2025-01-23")[:2] == ("gp", "exact")


def _zip_bytes(csv_text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nba_2008-2026.csv", csv_text)
    return buf.getvalue()


def test_import_kaggle_end_to_end(conn, tmp_path, data_dir):
    _game(conn, "g1", "2007-10-30", "SAS", "POR", hp=106, ap=97)
    _game(conn, "g2", "2007-10-31", "GSW", "UTA")        # Kaggle says 10-30: +/-1 day tolerance
    _game(conn, "g3", "2007-11-02", "NOH", "NJN", hp=95, ap=90)
    _game(conn, "g4", "2023-11-20", "PHX", "WAS", season="2023-24", hp=110, ap=100)
    # Kaggle rows 4 (bkn/ny 2014) and 6 (cha/sea) have no game -> unmatched
    zp = tmp_path / "k.zip"
    zp.write_bytes(_zip_bytes((FIX / "kaggle_sample.csv").read_text()))
    res = oh.import_kaggle(conn, zip_path=zp)
    assert res["rows"] == 6 and res["matched"] == 4
    assert res["how"]["exact"] == 3 and res["how"]["tolerance"] == 1 and res["tolerance_used"] == 1
    assert res["per_season"]["2007-08"]["matched"] == 3
    assert res["per_season"]["2007-08"]["tolerance"] == 1
    rows = {r["game_id"]: dict(r) for r in conn.execute("SELECT * FROM closing_lines")}
    assert set(rows) == {"g1", "g2", "g3", "g4"}
    assert rows["g1"]["home_spread_close"] == -13.0 and rows["g1"]["source"] == "kaggle"
    assert rows["g3"]["home_spread_close"] == 2.5 and rows["g3"]["provider"] == "sbr"
    assert rows["g4"]["provider"] == "espnbet" and rows["g4"]["home_ml_close"] is None
    csv_text = (data_dir / "reports" / "odds_unmatched.csv").read_text()
    assert csv_text.count("no_game") == 2 and "2014-12-01" in csv_text
    # idempotent
    oh.import_kaggle(conn, zip_path=zp)
    assert conn.execute("SELECT COUNT(*) FROM closing_lines").fetchone()[0] == 4


# ---------------------------------------------------------------------------- consensus, math

def test_consensus_prefers_espn_for_recent_seasons(conn):
    _game(conn, "old", "2010-01-05", "BOS", "NYK", season="2009-10")
    _game(conn, "new", "2025-01-05", "BOS", "NYK", season="2024-25")
    base = dict(home_spread_open=None, total_open=None, home_ml_open=None, away_ml_open=None)
    db.upsert_rows(conn, "closing_lines", [
        {"game_id": "old", "source": "kaggle", "provider": "sbr", "home_spread_close": -4.0,
         "total_close": 200.0, "home_ml_close": -180.0, "away_ml_close": 160.0, **base},
        {"game_id": "old", "source": "espn", "provider": "X", "home_spread_close": -9.0, **base},
        {"game_id": "new", "source": "kaggle", "provider": "espnbet", "home_spread_close": -5.0, **base},
        {"game_id": "new", "source": "espn", "provider": "DraftKings", "home_spread_close": -6.5,
         "home_spread_open": -2.5},
    ])
    old, new = oh.consensus_line(conn, "old"), oh.consensus_line(conn, "new")
    assert old["source"] == "kaggle" and old["home_spread"] == -4.0 and old["home_ml"] == -180.0
    assert new["source"] == "espn" and new["home_spread"] == -6.5 and new["home_spread_open"] == -2.5
    assert oh.consensus_line(conn, "missing") is None


def test_devig_and_sanity(conn):
    assert oh.devig_home_prob(-110, -110) == pytest.approx(0.5)
    p = oh.devig_home_prob(-200, 170)
    assert 0.63 < p < 0.66
    _game(conn, "a", "2015-01-01", "BOS", "NYK", season="2014-15", hp=100, ap=90)
    _game(conn, "b", "2015-01-02", "BOS", "NYK", season="2014-15", hp=90, ap=100)
    _game(conn, "c", "2015-01-03", "BOS", "NYK", season="2014-15", hp=95, ap=80)
    for gid, sp in [("a", -5.0), ("b", -5.0), ("c", 3.0)]:
        db.upsert_rows(conn, "closing_lines", [{
            "game_id": gid, "source": "kaggle", "provider": "sbr", "home_spread_close": sp,
            "home_ml_close": -200.0, "away_ml_close": 170.0}])
    s = oh.sanity(conn, "kaggle")["2014-15"]
    assert s["n"] == 3 and s["fav_win_rate"] == pytest.approx(1 / 3, abs=1e-3)   # a won, b lost, c upset
    assert s["n_ml"] == 3 and 0 < s["brier"] < 1 and s["logloss"] > 0


# ---------------------------------------------------------------------------- migration

def test_closing_lines_migration_idempotent(tmp_path):
    c = db.connect(tmp_path / "m.db")
    v = db.migrate(c)
    assert db.migrate(c) == v == len(db.MIGRATIONS)
    cols = [r[1] for r in c.execute("PRAGMA table_info(closing_lines)")]
    assert cols == list(oh.CL_COLUMNS)
    pk = [r[1] for r in c.execute("PRAGMA table_info(closing_lines)") if r[5]]
    assert pk == ["game_id", "source"]
    n = c.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert n == len(db.MIGRATIONS)
    c.close()
