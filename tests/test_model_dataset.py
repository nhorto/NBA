import pandas as pd
import pytest

from courtside.model import dataset


def _row(gid, team, opp, is_home, pts, date="2015-11-01", season="2015-16",
         stype="Regular Season", mins=240.0, opp_pts=None):
    return {"game_id": gid, "team": team, "season": season, "season_type": stype,
            "game_date": date, "opp": opp, "is_home": is_home,
            "wl": None if opp_pts is None else ("W" if pts > opp_pts else "L"),
            "min": mins, "pts": pts}


def _game(gid, h, a, hp, ap, **kw):
    return [_row(gid, h, a, 1, hp, opp_pts=ap, **kw), _row(gid, a, h, 0, ap, opp_pts=hp, **kw)]


def test_pairs_rows_and_maps_franchises():
    rows = pd.DataFrame(_game("0020700001", "SEA", "NJN", 100, 90, season="2007-08",
                              date="2007-11-01")
                        + _game("0020700002", "BOS", "NOK", 95, 99, season="2007-08",
                                date="2007-11-02"))
    g = dataset.build_games(rows)
    assert list(g["home"]) == ["OKC", "BOS"]
    assert list(g["away"]) == ["BKN", "NOP"]
    assert list(g["home_raw"]) == ["SEA", "BOS"]
    assert list(g["margin"]) == [10, -4]
    assert list(g["home_win"]) == [1, 0]
    assert g["site"].iloc[0].startswith("Seattle")  # era-aware home arena
    assert not g["is_neutral"].any()


def test_zero_minute_games_dropped():
    rows = pd.DataFrame(_game("0021201214", "BOS", "IND", 0, 0, mins=0.0, date="2013-04-16",
                              season="2012-13")
                        + _game("0021200001", "BOS", "IND", 100, 90, season="2012-13"))
    g = dataset.build_games(rows)
    assert list(g["game_id"]) == ["0021200001"]
    assert g.attrs["anomalies"]["junk_games_dropped"] == ["0021201214"]


def test_partial_junk_raises():
    r = _game("0021500001", "BOS", "IND", 100, 90)
    r[1]["min"] = 0.0
    with pytest.raises(dataset.DataError):
        dataset.build_games(pd.DataFrame(r))


def test_not_two_rows_raises():
    r = _game("0021500001", "BOS", "IND", 100, 90)
    with pytest.raises(dataset.DataError):
        dataset.build_games(pd.DataFrame(r[:1]))
    r3 = r + [_row("0021500001", "LAL", "BOS", 0, 80)]
    with pytest.raises(dataset.DataError):
        dataset.build_games(pd.DataFrame(r3))


def test_two_home_rows_raises():
    r = _game("0021500001", "BOS", "IND", 100, 90)
    r[1]["is_home"] = 1
    with pytest.raises(dataset.DataError):
        dataset.build_games(pd.DataFrame(r))


def test_neutral_detection():
    rows = (
        # both "@" (stats.nba.com from 2024-25): flagged neutral, nominal home alphabetical
        _game("0022400147", "WAS", "MIA", 98, 118, date="2024-11-02", season="2024-25")
        # curated London game
        + _game("0021800665", "NYK", "WAS", 100, 101, date="2019-01-17", season="2018-19")
        # bubble seeding game and bubble playoffs
        + _game("0021901300", "LAL", "LAC", 103, 101, date="2020-07-30", season="2019-20")
        + _game("0041900101", "MIL", "ORL", 110, 122, date="2020-08-18", season="2019-20",
                stype="Playoffs")
        # pre-bubble 2019-20 game is a normal home game
        + _game("0021900900", "LAL", "LAC", 103, 111, date="2020-03-08", season="2019-20"))
    rows = pd.DataFrame(rows)
    rows.loc[(rows["game_id"] == "0022400147"), "is_home"] = 0
    g = dataset.build_games(rows).set_index("game_id")
    assert g.loc["0022400147", "neutral_reason"] == "flagged"
    assert g.loc["0022400147", "home"] == "MIA"           # alphabetical nominal home
    assert g.loc["0022400147", "margin"] == 20
    assert g.loc["0022400147", "site"].startswith("Mexico City")
    assert g.loc["0021800665", "neutral_reason"] == "curated"
    assert g.loc["0021800665", "site"].startswith("London")
    assert g.loc["0021901300", "neutral_reason"] == "bubble"
    assert g.loc["0041900101", "neutral_reason"] == "bubble"
    assert not g.loc["0021900900", "is_neutral"]


def test_overtime_and_order():
    rows = pd.DataFrame(_game("0021500002", "BOS", "IND", 120, 118, mins=265.0,
                              date="2015-11-02")
                        + _game("0021500001", "LAL", "GSW", 100, 90, date="2015-11-01"))
    g = dataset.build_games(rows)
    assert list(g["game_id"]) == ["0021500001", "0021500002"]
    assert list(g["ot"]) == [0, 1]


def test_synthetic_league_builds(synthetic_rows):
    g = dataset.build_games(synthetic_rows())
    assert len(g) == len(synthetic_rows()) // 2
    assert g["game_date"].is_monotonic_increasing
