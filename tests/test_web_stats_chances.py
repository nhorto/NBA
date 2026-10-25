"""The player page's "Chances" panel (templates/_player_opportunity.html, web/stats_chances.py)
and the /leaders "Running hot / running cold" card, on a tiny fixture DB.

Fixture (previous season): Jokic (C) 30 regular-season games, 10 of 20 rebound chances in the
first 20 and 16 of 20 in the last 10, then 4 playoff games at 4 of 20; Beta (C) 10 of 20 every
game sets the league rate. One regular-season game has an implausible 44 potential assists on 5
assists. Current season: one Jokic game, no tracking (the empty state).
"""

from __future__ import annotations

import re
import warnings

import pytest

warnings.filterwarnings("ignore", message="Using `httpx`")
from fastapi.testclient import TestClient  # noqa: E402

from courtside import config, db  # noqa: E402
from courtside.stats import db as sdb  # noqa: E402
from courtside.stats import opportunity as op  # noqa: E402
from courtside.stats import seasons as S  # noqa: E402
from courtside.web.app import create_app  # noqa: E402

TODAY = config.today_nba_day()
CUR = S.current_season(TODAY)
PREV = S.prev_season(CUR)
PY = S.start_year(PREV)
JOKIC, BETA = 203999, 2
BAD_GAME = 27                       # regular-season game with 44 potential assists on 5 assists


def _day(i: int) -> str:
    m, d = divmod(i - 1, 28)
    return f"{PY}-{11 + m:02d}-{d + 1:02d}"


def _game(conn, gid, pid, name, team, season, stype, day, *, reb, chances, ast=8, pot=None,
          tracking=True):
    db.upsert_rows(conn, "player_games", [{
        "game_id": gid, "player_id": pid, "player_name": name, "team": team, "season": season,
        "season_type": stype, "game_date": day, "min": 34, "reb": reb, "oreb": 2,
        "dreb": reb - 2, "ast": ast, "stl": 1, "blk": 1, "fgm": 8, "fga": 16, "pts": 20}])
    if not tracking:
        return
    db.upsert_rows(conn, "player_tracking", [{
        "game_id": gid, "player_id": pid, "team": team, "minutes": 34,
        "reb_chances_off": 5, "reb_chances_def": chances - 5, "reb_chances_total": chances,
        "passes": 60, "touches": 80, "def_rim_fgm": 3, "def_rim_fga": 6,
        "cont_fgm": 3, "cont_fga": 8, "uncont_fgm": 5, "uncont_fga": 8}])
    db.upsert_rows(conn, "player_hustle", [{
        "game_id": gid, "player_id": pid, "team": team, "deflections": 2,
        "contested_shots": 7, "contested_2pt": 5, "contested_3pt": 2}])
    if pot is not None:
        db.upsert_rows(conn, "player_tracking_ext", [{
            "game_id": gid, "player_id": pid, "team": team, "game_date": day, "season": season,
            "pot_ast": pot}])


def populate(conn) -> None:
    conn.executemany("INSERT INTO player_index (player_id, first_name, last_name, team, position)"
                     " VALUES (?,?,?,?,?)", [(JOKIC, "Nikola", "Jokic", "DEN", "C"),
                                             (BETA, "Beta", "Big", "GSW", "C")])
    for i in range(1, 31):
        reb = 16 if i > 20 else 10
        bad = i == BAD_GAME
        _game(conn, f"r{i}", JOKIC, "Nikola Jokic", "DEN", PREV, S.REGULAR, _day(i), reb=reb,
              chances=20, ast=5 if bad else 8, pot=44 if bad else 12)
        _game(conn, f"r{i}", BETA, "Beta Big", "GSW", PREV, S.REGULAR, _day(i), reb=10,
              chances=20, ast=2)
    for i in range(1, 5):
        _game(conn, f"po{i}", JOKIC, "Nikola Jokic", "DEN", PREV, S.PLAYOFFS, f"{PY + 1}-04-2{i}",
              reb=4, chances=20, pot=12)
    _game(conn, "c1", JOKIC, "Nikola Jokic", "DEN", CUR, S.REGULAR, f"{S.start_year(CUR)}-10-22",
          reb=12, chances=0, tracking=False)
    conn.commit()


@pytest.fixture
def client(data_dir):
    sdb.clear()
    conn = db.connect()
    populate(conn)
    conn.close()
    return TestClient(create_app(data_dir))


def _panel(html: str) -> str:
    i = html.index('id="chances"')
    return html[i:html.index("</section>", i)]


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ------------------------------------------------------------------------------- panel

def test_panel_renders_numbers_and_flags(client):
    r = client.get(f"/player/{JOKIC}?season={PREV}")
    assert r.status_code == 200
    p = _panel(r.text)
    t = _text(p)
    assert "Chances" in t and "opportunity vs conversion" in t
    # windows with game counts; L5 / L10 = 16 of 20, season = (20x10 + 10x16) / 600
    assert "Last 5 5 g" in t and "Last 10 10 g" in t and "Season 30 g" in t
    assert "80.0% 80.0% 60.0%" in t
    assert "League centers" in t
    # the signal: plain English, information only
    assert "Running hot" in t
    assert "Converting 80% of rebound chances over his last 10 games vs his usual" in t
    assert "likely to cool off" in t
    assert not re.search(r"\b(under|edges?|lean|bets?|betting|wager|the over)\b", t.lower())
    # the conversion-vs-baseline bar: markers by shape class + the numbers printed
    assert 'class="ch-m ch-l10"' in p and 'class="ch-m ch-own"' in p and 'ch-m ch-league' in p
    assert 'role="img" aria-label="Rebound chances converted: Last 5 80.0%' in p
    assert "/static/pro-chances.css" in r.text
    # contested share from player_tracking (8 contested of 16), regular-season games only
    from courtside.web import stats_chances as CH
    conn = db.connect()
    k = CH._contested(conn, JOKIC, PREV, S.REGULAR)
    conn.close()
    assert k["season"]["games"] == 30 and k["l5"]["cont_share"] == 0.5
    assert k["l10"]["cont_fg"] == 3 / 8 and k["l10"]["open_fg"] == 5 / 8


def test_empty_state_when_tracking_missing(client):
    t = _text(_panel(client.get(f"/player/{JOKIC}").text))        # current season, no tracking
    assert "Tracking not collected yet." in t
    assert ("Tracking stats for this season are collected nightly; first games appear after "
            "the next update.") in t
    t2 = _text(_panel(client.get(f"/player/{BETA}?season={CUR}").text))
    assert "No regular season games in " + CUR in t2


def test_plausibility_guard(client):
    assert op.implausible_pot_ast(44, 5) and op.implausible_pot_ast(31, 10)
    assert op.implausible_pot_ast(13, 2)
    assert not op.implausible_pot_ast(12, 8) and not op.implausible_pot_ast(9, 1)
    assert not op.implausible_pot_ast(None, 0)
    conn = db.connect()
    o = op.player_opportunity(conn, JOKIC, PREV, S.REGULAR)
    conn.close()
    a5 = o["assists"]["l5"]
    assert a5["pot_ast_excluded"] == 1 and a5["pot_ast_games"] == 4
    assert a5["pot_ast_pg"] == 12.0 and a5["ast_per_pot_ast"] == round(32 / 48, 3)
    assert o["assists"]["season"]["pot_ast_excluded"] == 1
    t = _text(_panel(client.get(f"/player/{JOKIC}?season={PREV}").text))
    assert "data check" in t and "1 game with an implausible potential-assist count" in t
    assert "12.0 12.0 12.0" in t                                   # not (4x12 + 44) / 5
    assert "Potential assists 12.0 12.0 12.0 " in t
    assert "18.4" not in t


def test_playoffs_stay_out_of_regular_season_windows(client):
    conn = db.connect()
    reg = op.player_opportunity(conn, JOKIC, PREV, S.REGULAR)
    po = op.player_opportunity(conn, JOKIC, PREV, S.PLAYOFFS)
    flags = op.regression_flags(conn, PREV, 10, metrics=("reb",), season_type=S.REGULAR)
    conn.close()
    assert reg["rebounds"]["l10"]["conv"] == 0.8 and reg["window_games"]["season"] == 30
    (f,) = [x for x in reg["flags"] if x["metric"] == "reb"]
    assert f["rate"] == 0.8 and f["games"] == 10 and f["z_shrunk"] > 0      # no playoff games
    assert [x["rate"] for x in flags["hot"] if x["player_id"] == JOKIC] == [0.8]
    assert po["rebounds"]["season"]["games"] == 4 and po["rebounds"]["l10"]["conv"] == 0.2
    (fp,) = [x for x in po["flags"] if x["metric"] == "reb"]
    assert fp["rate"] == 0.2 and fp["games"] == 4 and fp["z_shrunk"] < 0
    t = _text(_panel(client.get(f"/player/{JOKIC}?season={PREV}&type=playoffs").text))
    assert "Season 4 g" in t and "Running cold" in t
    assert "Converting 20% of rebound chances over his last 4 games" in t


def test_leaders_running_hot_cold_card(client):
    r = client.get(f"/leaders?season={PREV}")
    assert r.status_code == 200
    i = r.text.index("Running hot / running cold")
    card = r.text[i:r.text.index("</section>", i)]
    t = _text(card)
    assert "How this works" in card and "data-tip=" in card and "shrunk" in card
    assert f'href="/player/{JOKIC}"' in card
    assert "Running hot" in t and "Rebounds" in t and "80.0%" in t
    assert 'data-seg-group="chstat"' in card
    assert "/static/pro-chances.css" in r.text
    # a season with no tracking: the shared empty state
    t2 = _text(client.get(f"/leaders?season={CUR}").text)
    assert "Nobody to list yet." in t2
