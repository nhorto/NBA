import json
from pathlib import Path

import pytest

from courtside import archive, backfill, cli, db
from courtside.sources import nba_stats as ns

FIX = Path(__file__).parent / "fixtures" / "nba_stats"


def load(name):
    return json.loads((FIX / f"{name}.json").read_text())


def test_seasons_range():
    assert backfill.seasons_range("2010-11", "2012-13") == ["2010-11", "2011-12", "2012-13"]
    assert backfill.seasons_range("1999-00", "2000-01") == ["1999-00", "2000-01"]
    assert len(backfill.seasons_range("2010-11", "2025-26")) == 16
    with pytest.raises(ValueError):
        backfill.seasons_range("2012-13", "2010-11")
    with pytest.raises(ValueError):
        backfill.seasons_range("2010-12", "2012-13")


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []
    def team(season, stype, pot):
        calls.append(("lg", pot, season, stype))
        return load("team_gamelog" if pot == "T" else "player_gamelog")
    def adv(season, stype, entity):
        calls.append(("adv", entity, season, stype))
        return load("team_adv" if entity == "team" else "player_adv")
    monkeypatch.setattr(ns, "fetch_league_gamelog", team)
    monkeypatch.setattr(ns, "fetch_gamelogs_advanced", adv)
    return calls


def test_backfill_skips_archived_and_is_idempotent(fake_fetch):
    s1 = backfill.backfill_gamelogs(["2025-26"], season_types=("Regular Season",))
    assert s1["fetched"] == 4 and s1["cached"] == 0 and s1["failed"] == 0
    assert len(fake_fetch) == 4
    assert s1["rows"] == {"team_games": 10, "player_games": 10,
                          "team_games_adv": 10, "player_games_adv": 10}
    assert archive.raw_path("nba_stats", "team_gamelog", "2025-10-01",
                            "2025-26_regular_season").exists()

    s2 = backfill.backfill_gamelogs(["2025-26"], season_types=("Regular Season",))
    assert s2["fetched"] == 0 and s2["cached"] == 4
    assert len(fake_fetch) == 4  # no new network calls

    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM team_games").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM player_games_adv").fetchone()[0] == 10
    runs = conn.execute("SELECT job, ok FROM collect_runs").fetchall()
    assert [(r["job"], r["ok"]) for r in runs] == [("backfill_gamelogs", 1)] * 2

    s3 = backfill.backfill_gamelogs(["2025-26"], season_types=("Regular Season",), force=True)
    assert s3["fetched"] == 4 and len(fake_fetch) == 8


def test_backfill_no_advanced(fake_fetch):
    s = backfill.backfill_gamelogs(["2025-26"], season_types=("Playoffs",), include_advanced=False)
    assert s["fetched"] == 2 and all(c[0] == "lg" for c in fake_fetch)


def test_backfill_records_failure_and_resumes(monkeypatch, fake_fetch):
    real = ns.fetch_gamelogs_advanced
    monkeypatch.setattr(ns, "fetch_gamelogs_advanced",
                        lambda *a: (_ for _ in ()).throw(ns.BlockedError("blocked")))
    s = backfill.backfill_gamelogs(["2025-26"], season_types=("Regular Season",))
    assert s["failed"] == 2 and s["fetched"] == 2
    assert db.connect().execute("SELECT ok FROM collect_runs").fetchone()[0] == 0
    monkeypatch.setattr(ns, "fetch_gamelogs_advanced", real)
    s = backfill.backfill_gamelogs(["2025-26"], season_types=("Regular Season",))
    assert s["failed"] == 0 and s["cached"] == 2 and s["fetched"] == 2


def test_cli_backfill(fake_fetch, capsys):
    assert cli.main(["backfill", "gamelogs", "--from", "2025-26", "--to", "2025-26",
                     "--no-advanced"]) == 0
    assert "team_games" in capsys.readouterr().out
    assert cli.main(["backfill", "gamelogs", "--from", "bad"]) == 2
