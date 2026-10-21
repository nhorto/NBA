"""Preseason game-log backfill (no network): trimmed real 2019-20 'Pre Season' fixtures.

Fixture games: 0011900001 SDS @ HOU (Shanghai, non-NBA), 0011900002 HOU @ LAC,
0011900003 IND @ SAC, 0011900101 ORL @ LAC (2020-07-22 bubble scrimmage).
"""

import json
from pathlib import Path

import pytest

from courtside import archive, backfill_preseason as bp, cli, db
from courtside.sources import nba_stats as ns

FIX = Path(__file__).parent / "fixtures" / "nba_stats"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []

    def lg(season, stype, pot):
        calls.append((season, stype, pot))
        return load("preseason_team_gamelog" if pot == "T" else "preseason_player_gamelog")
    monkeypatch.setattr(ns, "fetch_league_gamelog", lg)
    return calls


def test_parser_on_real_fixture():
    teams = bp.team_rows(load("preseason_team_gamelog"), "2019-20")
    assert len(teams) == 8
    by = {(r["game_id"], r["team"]): r for r in teams}
    hou = by[("0011900002", "HOU")]
    assert hou["season_type"] == "Pre Season" and hou["season"] == "2019-20"
    assert hou["game_date"] == "2019-10-04" and hou["opp"] == "LAC" and hou["is_home"] == 0
    assert hou["scrimmage"] == 0 and hou["vs_non_nba"] == 0
    assert by[("0011900001", "HOU")]["vs_non_nba"] == 1
    assert by[("0011900001", "SDS")]["vs_non_nba"] == 1
    assert {r["min"] for r in teams if not r["scrimmage"]} <= {240.0, 265.0, 290.0}
    assert {r["min"] for r in teams if r["scrimmage"]} == {200.0}   # 40-minute bubble scrimmage

    players = bp.player_rows(load("preseason_player_gamelog"), "2019-20")
    assert len(players) == 16
    assert all(isinstance(r["min"], float) and r["player_id"] for r in players)
    assert all(r["season_type"] == "Pre Season" for r in players)
    assert {r["vs_non_nba"] for r in players if r["game_id"] == "0011900001"} == {1}


def test_scrimmage_flag():
    assert bp.is_scrimmage("2019-20", "2020-07-22")
    assert not bp.is_scrimmage("2019-20", "2019-10-04")
    assert not bp.is_scrimmage("2020-21", "2020-12-19")   # Dec 2020 preseason is real preseason
    assert not bp.is_scrimmage("2011-12", "2011-12-16")   # lockout preseason
    assert not bp.is_scrimmage("2019-20", None)
    teams = bp.team_rows(load("preseason_team_gamelog"), "2019-20")
    assert {r["game_id"] for r in teams if r["scrimmage"]} == {"0011900101"}
    players = bp.player_rows(load("preseason_player_gamelog"), "2019-20")
    assert {r["game_id"] for r in players if r["scrimmage"]} == {"0011900101"}


def test_non_nba_from_team_payload_reaches_players():
    pl = load("preseason_player_gamelog")
    rs = pl["resultSets"][0]
    ti = rs["headers"].index("TEAM_ID")
    rs["rowSet"] = [r for r in rs["rowSet"] if ns.is_nba_team_id(r[ti])]  # drop SDS players
    alone = bp.player_rows(pl, "2019-20")
    assert {r["vs_non_nba"] for r in alone if r["game_id"] == "0011900001"} == {0}
    non_nba = bp.non_nba_games(load("preseason_team_gamelog"))
    assert non_nba == {"0011900001"}
    fixed = bp.player_rows(pl, "2019-20", non_nba)
    assert {r["vs_non_nba"] for r in fixed if r["game_id"] == "0011900001"} == {1}


def test_migration_idempotent(tmp_path):
    path = tmp_path / "m.db"
    conn = db.connect(path)
    v = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert v == len(db.MIGRATIONS)
    conn.close()
    conn = db.connect(path)          # re-run: nothing re-applied, no error
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == len(db.MIGRATIONS)
    assert db.migrate(conn) == len(db.MIGRATIONS)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(preseason_player_games)")]
    base = [r[1] for r in conn.execute("PRAGMA table_info(player_games)")]
    assert cols == base + ["scrimmage", "vs_non_nba"]
    tcols = [r[1] for r in conn.execute("PRAGMA table_info(preseason_team_games)")]
    assert tcols == [r[1] for r in conn.execute("PRAGMA table_info(team_games)")] + [
        "scrimmage", "vs_non_nba"]
    # the migration SQL itself is re-runnable (IF NOT EXISTS)
    conn.executescript(db.MIGRATIONS[-1])


def test_backfill_skips_archived_and_never_touches_player_games(fake_fetch):
    s1 = bp.backfill_preseason(["2019-20"])
    assert s1["fetched"] == 2 and s1["cached"] == 0 and s1["failed"] == 0
    assert fake_fetch == [("2019-20", "Pre Season", "T"), ("2019-20", "Pre Season", "P")]
    assert s1["rows"] == {"preseason_team_games": 8, "preseason_player_games": 16}
    assert archive.raw_path("nba_stats", "player_gamelog", "2019-10-01",
                            "2019-20_pre_season").exists()

    s2 = bp.backfill_preseason(["2019-20"])
    assert s2["fetched"] == 0 and s2["cached"] == 2 and len(fake_fetch) == 2

    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM team_games").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM preseason_player_games").fetchone()[0] == 16
    assert conn.execute("SELECT SUM(scrimmage) FROM preseason_team_games").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(DISTINCT game_id) FROM preseason_player_games "
                        "WHERE vs_non_nba = 1").fetchone()[0] == 1
    runs = [(r["job"], r["ok"]) for r in conn.execute("SELECT job, ok FROM collect_runs")]
    assert runs == [("backfill_preseason", 1)] * 2

    s3 = bp.backfill_preseason(["2019-20"], force=True)
    assert s3["fetched"] == 2 and len(fake_fetch) == 4


def test_cli_backfill_preseason(fake_fetch, capsys):
    assert cli.main(["backfill", "preseason", "--from", "2019-20", "--to", "2019-20"]) == 0
    assert "fetched=2" in capsys.readouterr().out
    assert cli.main(["backfill", "preseason", "--from", "2019-20", "--to", "2018-19"]) == 2
    args = cli.build_parser().parse_args(["backfill", "preseason"])
    assert (args.start, args.end, args.force) == ("2010-11", "2026-27", False)
