import pandas as pd
import pytest

from courtside.analytics.pbp import Event
from courtside.analytics.stints import Stint
from courtside.sources import sportsdataverse as sv

API = {
    "nba_stats_game_lineups": {"assets": [
        {"name": "nba_lineups_2024.parquet", "size": 5, "browser_download_url": "http://x/l24"},
        {"name": "nba_lineups_2024.csv.gz", "size": 9, "browser_download_url": "http://x/l24c"},
        {"name": "nba_lineups_1997.parquet", "size": 3, "browser_download_url": "http://x/l97"},
        {"name": "README.md", "size": 1, "browser_download_url": "http://x/r"}]},
}


def test_season_end_year():
    assert sv.season_end_year("2023-24") == 2024 and sv.season_end_year(2024) == 2024
    assert sv.season_end_year("2024") == 2024
    with pytest.raises(ValueError):
        sv.season_end_year("23-24")


def test_download_only_parquet_in_range_and_skips_existing():
    got = []

    def get(url, **kw):
        return API[url.rsplit("/", 1)[1]]

    def get_bytes(url, **kw):
        got.append(url)
        return b"x" * {"http://x/l24": 5, "http://x/l97": 3}[url]

    r = sv.download(["lineups"], since=2000, until=2026, get=get, get_bytes=get_bytes,
                    log=lambda m: None)
    assert got == ["http://x/l24"] and r["downloaded"] == 1 and r["failed"] == 0
    assert sv.asset_path("nba_stats_game_lineups", "nba_lineups_2024.parquet").read_bytes() == b"xxxxx"

    r2 = sv.download(["lineups"], since=2000, until=2026, get=get, get_bytes=get_bytes,
                     log=lambda m: None)
    assert got == ["http://x/l24"] and r2["skipped"] == 1 and r2["downloaded"] == 0

    r3 = sv.download(["lineups"], since=1990, until=2026, get=get, get_bytes=get_bytes,
                     log=lambda m: None)
    assert r3["downloaded"] == 1 and got[-1] == "http://x/l97"   # only the missing season


def test_size_mismatch_is_a_failure_and_leaves_no_file():
    r = sv.download(["lineups"], since=2024, until=2024, get=lambda url, **kw: API[url.rsplit("/", 1)[1]],
                    get_bytes=lambda url, **kw: b"sh", log=lambda m: None)
    assert r["failed"] == 1 and r["downloaded"] == 0
    assert not sv.asset_path("nba_stats_game_lineups", "nba_lineups_2024.parquet").exists()


def _lineup_frame():
    rows = []
    for a in (2, 4, 7):
        rows.append({"game_id": "0022300001", "action_number": a, "period": 1,
                     **{f"home_player_{i}": i for i in range(1, 6)},
                     **{f"away_player_{i}": 10 + i for i in range(1, 6)}, "season": 2024})
    rows.append(rows[-1])   # real files hold exact duplicate rows
    return pd.DataFrame(rows)


def test_read_lineups_and_row_count():
    path = sv.asset_path("nba_stats_game_lineups", "nba_lineups_2024.parquet")
    path.parent.mkdir(parents=True)
    _lineup_frame().to_parquet(path)
    df = sv.read_lineups("2023-24")
    assert len(df) == 4 and df.game_id.nunique() == 1
    assert sv.read_lineups(2024).equals(df)
    assert sv.row_count("lineups", "2023-24") == 4
    with pytest.raises(FileNotFoundError):
        sv.read_lineups("2022-23")


def _event(idx, action):
    return Event(idx=idx, action_number=action, order=idx, period=1, clock=600.0, elapsed=10.0 * idx,
                 team_id=None, team=None, side=None, player_id=None, action_type="2pt",
                 sub_type="", descriptor=None, qualifiers=(), description="", points=0,
                 score_home=0, score_away=0)


def test_compare_with_stints_agreement_and_duplicates():
    home, away = (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)
    wrong = Stint(1, 0, 10, 2, 3, (1, 2, 3, 4, 9), away, 0, 0, 0, 0)
    stints = [Stint(1, 0, 10, 0, 2, home, away, 0, 0, 0, 0), wrong]
    events = [_event(0, 2), _event(1, 4), _event(2, 7), _event(3, 99)]
    res = sv.compare_with_stints(_lineup_frame(), events, stints)
    assert res["matched"] == 3 and res["agree"] == 2 and res["rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert res["no_stint"] == 0 and len(res["examples"]) == 1
    assert res["examples"][0]["action"] == 7
