from datetime import date, datetime, timezone

import pytest

from courtside import config


def test_nba_day_late_utc_is_previous_eastern_day():
    assert config.nba_day(datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)) == date(2026, 10, 20)


def test_nba_day_midday():
    assert config.nba_day(datetime(2026, 10, 21, 16, 0, tzinfo=timezone.utc)) == date(2026, 10, 21)


def test_nba_day_rejects_naive():
    with pytest.raises(ValueError):
        config.nba_day(datetime(2026, 10, 21, 2, 30))


def test_data_dir_reads_env_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSIDE_DATA", str(tmp_path / "x"))
    assert config.data_dir() == tmp_path / "x"
    assert config.db_path() == tmp_path / "x" / "courtside.db"
    assert not (tmp_path / "x").exists()
    config.ensure_dirs()
    assert (tmp_path / "x" / "raw").is_dir()


def test_default_data_dir(monkeypatch):
    monkeypatch.delenv("COURTSIDE_DATA")
    assert config.data_dir() == config.REPO_ROOT / "var"


def test_today_yesterday():
    assert config.today_nba_day() - config.yesterday_nba_day() == (date(2026, 1, 2) - date(2026, 1, 1))
