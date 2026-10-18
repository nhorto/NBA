import pandas as pd
import pytest

from courtside import teams
from courtside.model import dataset, schedule


def _rows(games, season="2015-16"):
    out = []
    for i, (date, h, a) in enumerate(games, start=1):
        gid = f"00215{i:05d}"
        for team, opp, is_home, pts in ((h, a, 1, 100), (a, h, 0, 90)):
            out.append({"game_id": gid, "team": team, "season": season,
                        "season_type": "Regular Season", "game_date": date, "opp": opp,
                        "is_home": is_home, "wl": "W" if pts == 100 else "L", "min": 240.0,
                        "pts": pts})
    return pd.DataFrame(out)


@pytest.fixture
def bos():
    g = dataset.build_games(_rows([
        ("2015-11-01", "BOS", "NYK"),
        ("2015-11-02", "NYK", "BOS"),
        ("2015-11-04", "BOS", "MIA"),
        ("2015-11-05", "BOS", "LAL"),
        ("2015-11-06", "DEN", "BOS"),
    ]))
    f = schedule.add_schedule_features(g)
    side = lambda r, col: r[f"home_{col}"] if r["home"] == "BOS" else r[f"away_{col}"]
    return {c: [side(r, c) for _, r in f.iterrows()] for c in schedule.TEAM_FEATURES}


def test_rest_and_b2b(bos):
    cap = schedule.REST_CAP
    assert bos["rest"] == [cap, 0, 1, 0, 0]
    assert bos["b2b"] == [0, 1, 0, 1, 1]
    assert bos["season_opener"] == [1, 0, 0, 0, 0]


def test_density_windows(bos):
    assert bos["three_in_four"] == [0, 0, 1, 1, 1]
    assert bos["four_in_six"] == [0, 0, 0, 1, 1]
    assert bos["games_last7"] == [0, 1, 2, 3, 4]


def test_travel_and_timezone(bos):
    b, n, d = (teams.by_tricode(x) for x in ("BOS", "NYK", "DEN"))
    bn = float(schedule.haversine_km(b.lat, b.lon, n.lat, n.lon))
    bd = float(schedule.haversine_km(b.lat, b.lon, d.lat, d.lon))
    assert 280 < bn < 320 and 2700 < bd < 2900
    assert bos["travel_km"] == pytest.approx([0.0, bn, bn, 0.0, bd])
    assert bos["tz_change"] == [0, 0, 0, 0, -2]


def test_rest_diff_and_season_boundary():
    g = dataset.build_games(pd.concat([
        _rows([("2015-11-01", "BOS", "NYK"), ("2015-11-02", "BOS", "MIA")]),
        _rows([("2016-10-30", "BOS", "NYK")], season="2016-17").assign(
            game_id=lambda d: "0021600001")]))
    f = schedule.add_schedule_features(g)
    assert list(f["home_rest"]) == [schedule.REST_CAP, 0, schedule.REST_CAP]
    assert list(f["home_b2b"]) == [0, 1, 0]
    assert list(f["rest_diff"]) == [0, 0 - schedule.REST_CAP, 0]
    assert list(f["home_season_opener"]) == [1, 0, 1]


def test_utc_offset_dst():
    assert schedule.utc_offset_hours("America/New_York", "2016-01-15") == -5
    assert schedule.utc_offset_hours("America/New_York", "2016-07-15") == -4
    assert schedule.utc_offset_hours("America/Phoenix", "2016-07-15") == -7


def test_features_use_no_results(synthetic_rows):
    rows = synthetic_rows()
    g1 = dataset.build_games(rows)
    rows2 = rows.copy()
    rows2["pts"] = rows2["is_home"].map({1: 50, 0: 60})  # every result flipped to an away win
    g2 = dataset.build_games(rows2)
    assert (g2["home_win"] == 0).all() and (g1["home_win"] == 1).any()
    pd.testing.assert_frame_equal(schedule.schedule_features(g1), schedule.schedule_features(g2))
