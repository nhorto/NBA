import numpy as np
import pandas as pd

from courtside import db
from courtside.proj import starters as S

H = S.HOME_COLS
A = S.AWAY_COLS


def _lineups():
    """Two games; rows out of order; game 2 has a 0 id on the away side."""
    rows = [
        # game 1: action 5 is the first period-1 row (action 2 is period 2 noise)
        {"game_id": "0022400001", "action_number": 9, "period": 1,
         **dict(zip(H, [1, 2, 3, 4, 99])), **dict(zip(A, [11, 12, 13, 14, 15]))},
        {"game_id": "0022400001", "action_number": 5, "period": 1,
         **dict(zip(H, [1, 2, 3, 4, 5])), **dict(zip(A, [11, 12, 13, 14, 15]))},
        {"game_id": "0022400001", "action_number": 2, "period": 2,
         **dict(zip(H, [6, 7, 8, 9, 10])), **dict(zip(A, [16, 17, 18, 19, 20]))},
        {"game_id": "0022400002", "action_number": 1, "period": 1,
         **dict(zip(H, [21, 22, 23, 24, 25])), **dict(zip(A, [31, 32, 33, 34, 0]))},
    ]
    return pd.DataFrame(rows)


def test_extract_first_lineups():
    f = S.extract_first_lineups(_lineups())
    g1 = f[f["game_id"] == "0022400001"]
    assert sorted(g1[g1["side"] == "home"]["player_id"]) == [1, 2, 3, 4, 5]
    assert sorted(g1[g1["side"] == "away"]["player_id"]) == [11, 12, 13, 14, 15]
    g2 = f[f["game_id"] == "0022400002"]
    assert set(g2["side"]) == {"home"}          # away side had a 0 id -> dropped
    assert len(f) == 15


def _box():
    rows = []
    # game 1: BOS (home in sdv) players 1..8, NYK players 11..18 -- neutral site (is_home 0/0)
    for pid, m in zip(range(1, 9), [36, 34, 30, 28, 0, 20, 18, 26]):
        rows.append(("0022400001", pid, "BOS", m))   # player 5 started per sdv but DNP
    for pid, m in zip(range(11, 19), [35, 33, 31, 29, 27, 20, 18, 7]):
        rows.append(("0022400001", pid, "NYK", m))
    return pd.DataFrame(rows, columns=["game_id", "player_id", "team", "min"])


def test_starts_for_season_maps_sides_by_overlap_and_repairs():
    box = _box()
    ha = pd.DataFrame({"game_id": ["0022400001"] * 2, "team": ["BOS", "NYK"],
                       "is_home": [0, 0]})      # neutral site: no explicit home
    first = S.extract_first_lineups(_lineups())
    st = S.starts_for_season(box, ha, first)
    bos = st[st["team"] == "BOS"]
    nyk = st[st["team"] == "NYK"]
    assert len(bos) == 5 and len(nyk) == 5
    assert sorted(nyk["player_id"]) == [11, 12, 13, 14, 15]
    assert set(nyk["source"]) == {"sdv"}
    # player 5 had no minutes -> replaced by the top-minutes non-starter (8, 26 min) as proxy
    assert 5 not in set(bos["player_id"])
    assert bos.set_index("player_id").loc[8, "source"] == "proxy"
    assert (bos["source"] == "sdv").sum() == 4
    assert st.attrs["sdv_valid_rows"] == 9


def test_build_load_validate_roundtrip():
    conn = db.connect()
    box = _box()
    for gid, pid, team, m in box.itertuples(index=False):
        conn.execute("INSERT INTO player_games(game_id, player_id, team, season, season_type, "
                     "game_date, min) VALUES (?,?,?,?,?,?,?)",
                     (gid, pid, team, "2024-25", "Regular Season", "2024-10-22", m))
    for team, home in (("BOS", 1), ("NYK", 0)):
        conn.execute("INSERT INTO team_games(game_id, team, season, season_type, game_date, "
                     "is_home, min) VALUES (?,?,?,?,?,?,?)",
                     ("0022400001", team, "2024-25", "Regular Season", "2024-10-22", home, 240))
    # a game without sdv lineups -> all proxy
    for pid in range(41, 49):
        conn.execute("INSERT INTO player_games(game_id, player_id, team, season, season_type, "
                     "game_date, min) VALUES (?,?,?,?,?,?,?)",
                     ("0022400003", pid, "LAL", "2024-25", "Regular Season", "2024-10-23",
                      50 - pid))
    conn.commit()
    lu = _lineups()

    def fake_read(season, columns=None):
        assert season == "2024-25"
        return lu[columns] if columns else lu

    out = S.build_starts(conn, ["2024-25"], read_lineups=fake_read, log=lambda s: None)
    assert out["rows"] == 15
    v = out["validation"]
    assert v["box_check_rate"] == 1.0 and v["team_games_with_5"] == 3
    again = S.build_starts(conn, ["2024-25"], read_lineups=fake_read, log=lambda s: None,
                           validate=False)
    assert again["rows"] == 15
    st = S.load_starts(conn)
    assert len(st) == 15
    lal = st[st["game_id"] == "0022400003"]
    assert sorted(lal["player_id"]) == [41, 42, 43, 44, 45]
    assert set(lal["source"]) == {"proxy"}
    assert np.issubdtype(st["player_id"].dtype, np.integer)
