"""IMPROVEMENTS B3: past-day replays price only pre-tip injury news; live days unchanged.
Plus the opt-in game-day rows for the game model (COURTSIDE_INJ_GAMEDAY_GAME): on a
back-to-back the official report lists a player twice (tonight, tomorrow), and only tonight's
row may price tonight's game."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from synth import day_date, synth_conn

from courtside import config, injuries, predict
from courtside.model import players as pl
from courtside.proj import availability as pav


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)
    injuries.set_inj_gameday_game(False)
    yield
    injuries.set_inj_gameday_game(injuries._env_flag(injuries.INJ_GAMEDAY_GAME_ENV,
                                                     injuries.INJ_GAMEDAY_GAME_DEFAULT))


DAY = day_date("2024-25", 22)
NEXT = DAY + timedelta(days=1)
INS = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, reason) "
       "VALUES (?,?,?,?,?,?)")


def _z(d: date, hhmm: str) -> str:
    return f"{d.isoformat()}T{hhmm}:00Z"


# --- injury_cutoff --------------------------------------------------------------------------

def test_cutoff_past_day_caps_at_tip_and_end_of_day(monkeypatch):
    monkeypatch.setattr(config, "today_nba_day", lambda: date(2025, 3, 1))
    d = date(2025, 1, 10)
    eod = "2025-01-11T05:00:00Z"
    assert predict.injury_cutoff(d, _z(d, "12:00"), "2025-01-11T00:30:00Z") == \
        "2025-01-11T00:30:00Z"
    assert predict.injury_cutoff(d, _z(d, "12:00"), "2025-01-11T00:30Z") == \
        "2025-01-11T00:30:00Z"                                    # ESPN-style, no seconds
    assert predict.injury_cutoff(d, _z(d, "12:00"), "2025-01-11T09:00:00Z") == eod
    assert predict.injury_cutoff(d, _z(d, "12:00")) == _z(d, "12:00")     # no tip: virtual time
    assert predict.injury_cutoff(d, "2025-02-01T10:00:00Z") == eod        # Brief: wall clock


def test_cutoff_live_day_is_created_at_whatever_the_tip(monkeypatch):
    today = date(2025, 1, 10)
    monkeypatch.setattr(config, "today_nba_day", lambda: today)
    now = "2025-01-11T02:00:00Z"                                   # after an early tip
    for d in (today, today + timedelta(days=1)):
        assert predict.injury_cutoff(d, now) == now
        assert predict.injury_cutoff(d, now, "2025-01-11T00:00:00Z") == now


# --- a replayed past day ----------------------------------------------------------------------

def _slate(conn):
    gs = conn.execute("SELECT game_id, team, opp, is_home FROM team_games WHERE game_date=? "
                      "AND is_home=1 ORDER BY game_id", (DAY.isoformat(),)).fetchall()
    assert len(gs) >= 2
    return [(g["game_id"], g["team"], g["opp"]) for g in gs]


def _past_setup(monkeypatch):
    """Two games on a past day with known tips (via the archived-scoreboard path that the
    2025-26 replay uses): game A at 00:00Z, game B at 02:30Z, the rest at 03:00Z. Official
    batches at 22:00Z (pre-tip for all), 01:00Z (after A, before B) and 04:00Z (after all)."""
    conn = synth_conn()
    sl = _slate(conn)
    tips = {g: _z(NEXT, "03:00") for g, _, _ in sl}
    tips[sl[0][0]], tips[sl[1][0]] = _z(NEXT, "00:00"), _z(NEXT, "02:30")
    by_pair = {(h, a): tips[g] for g, h, a in sl}
    monkeypatch.setattr(pav, "espn_scoreboard_tips", lambda d: [
        {"home": h, "away": a, "tipoff_utc": t} for (h, a), t in by_pair.items()]
        if d == DAY else [])
    for cap, status in ((_z(DAY, "22:00"), "Questionable"), (_z(NEXT, "01:00"), "Doubtful"),
                        (_z(NEXT, "04:00"), "Out")):
        for _, h, a in sl:
            conn.execute(INS, (cap, "nba_official", h, f"{h} Player 0", status, "knee"))
            conn.execute(INS, (cap, "nba_official", a, f"{a} Player 1", "Available", ""))
    conn.commit()
    return conn, sl, tips


def _read_spy(monkeypatch):
    seen: list[str] = []
    real = injuries._batch_rows

    def spy(conn, source, captured_at, *a, **kw):
        seen.append(captured_at)
        return real(conn, source, captured_at, *a, **kw)
    monkeypatch.setattr(injuries, "_batch_rows", spy)
    return seen


@pytest.mark.parametrize("path", ["predict_day", "predict_days"])
def test_past_day_replay_never_reads_post_tip_rows(monkeypatch, path):
    conn, sl, tips = _past_setup(monkeypatch)
    seen = _read_spy(monkeypatch)
    if path == "predict_day":
        rows = predict.predict_day(conn, DAY, store=False)
    else:
        predict.predict_days(conn, DAY, DAY)
        rows = list(predict.latest_pretip(conn, day=DAY.isoformat()).values())
    assert seen and max(seen) <= max(tips.values())               # 04:00Z is never read
    assert _z(NEXT, "04:00") not in seen
    by = {r["game_id"]: r for r in rows}
    assert set(by) == set(tips)
    for gid, tip in tips.items():
        inp = by[gid]["inputs"]
        assert inp["injuries_at"] <= tip
        assert all(c <= tip for c in inp["injuries_batches"].values())
        for side in ("home", "away"):
            cap = inp["availability"][side]["captured_at"]
            assert cap is None or cap <= tip
    a, b = sl[0][0], sl[1][0]
    assert by[a]["inputs"]["injuries_batches"]["nba_official"] == _z(DAY, "22:00")
    assert by[b]["inputs"]["injuries_batches"]["nba_official"] == _z(NEXT, "01:00")
    assert [x["status"] for x in by[a]["inputs"]["injuries"]["home"]] == ["Questionable"]
    assert [x["status"] for x in by[b]["inputs"]["injuries"]["home"]] == ["Doubtful"]


def test_past_day_one_model_path_is_capped_too(monkeypatch):
    conn, sl, tips = _past_setup(monkeypatch)
    seen = _read_spy(monkeypatch)
    rows = {r["game_id"]: r for r in predict.predict_day(conn, DAY, model="elo", store=False)}
    assert _z(NEXT, "04:00") not in seen
    assert rows[sl[0][0]]["inputs"]["injuries_batches"]["nba_official"] == _z(DAY, "22:00")


def test_live_day_reads_up_to_created_at_and_never_looks_up_tips(monkeypatch):
    """A live day is priced exactly as before B3: one slate-wide cutoff = created_at (here
    after game A's tip, so A sees the post-tip batch, as a live run at that time would), no tip
    lookup, no extra stored keys."""
    conn, sl, tips = _past_setup(monkeypatch)
    monkeypatch.setattr(config, "today_nba_day", lambda: DAY)
    monkeypatch.setattr(predict, "_past_tips", lambda *a: pytest.fail("tip lookup on live day"))
    ats: list[str] = []
    real = injuries.resolve
    monkeypatch.setattr(injuries, "resolve",
                        lambda conn, at, **kw: ats.append(at) or real(conn, at, **kw))
    now = _z(NEXT, "01:30")
    rows = predict.predict_day(conn, DAY, created_at=now, store=False)
    assert ats and set(ats) == {now}
    for r in rows:
        assert "injuries_at" not in r["inputs"] and "injuries_gameday" not in r["inputs"]
        assert r["inputs"]["injuries_batches"]["nba_official"] == _z(NEXT, "01:00")


# --- game-day rows (COURTSIDE_INJ_GAMEDAY_GAME) -----------------------------------------------

def _b2b_conn():
    """One official batch listing the home team's Player 0 twice: Questionable for DAY,
    Out for the next day's game; plus an ESPN row without a game date."""
    conn = synth_conn()
    conn.execute("ALTER TABLE injury_snapshots ADD COLUMN game_date TEXT")
    (gid, h, a), *_ = _slate(conn)
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
           "reason, game_date) VALUES (?,?,?,?,?,?,?)")
    cap = _z(DAY, "21:00")
    conn.execute(ins, (cap, "nba_official", h, f"{h} Player 0", "Questionable", "knee",
                       DAY.isoformat()))
    conn.execute(ins, (cap, "nba_official", h, f"{h} Player 0", "Out", "knee", NEXT.isoformat()))
    conn.execute(ins, (cap, "espn", a, f"{a} Player 2", "Out", "ankle", None))
    conn.commit()
    return conn, gid, h, a


def test_gameday_off_is_the_old_most_severe_merge():
    conn, gid, h, a = _b2b_conn()
    at = _z(DAY, "23:00")
    assert injuries.resolve(conn, at, game_day=DAY) == injuries.resolve(conn, at)
    assert injuries.current(conn, at, game_day=DAY) == injuries.current(conn, at)
    assert [x["status"] for x in injuries.current(conn, at)["by_team"][h]] == ["Out"]


def test_gameday_on_prices_only_tonights_row():
    conn, gid, h, a = _b2b_conn()
    at = _z(DAY, "23:00")
    injuries.set_inj_gameday_game(True)
    cur = injuries.current(conn, at, game_day=DAY)["by_team"]
    assert [x["status"] for x in cur[h]] == ["Questionable"]
    assert [x["status"] for x in cur[a]] == ["Out"]                       # NULL game_date kept
    tomorrow = injuries.current(conn, at, game_day=NEXT)["by_team"]
    assert [x["status"] for x in tomorrow[h]] == ["Out"]
    # the priced availability (P(plays)) follows the same rows
    av = injuries.availability(conn, at, DAY, {h, a})
    assert sorted(p["status"] for p in av[h]["players"]) == ["Questionable"]
    # no game_day: unchanged even when ON (DFS callers of resolve)
    assert injuries.resolve(conn, at) == (injuries.set_inj_gameday_game(False)
                                          or injuries.resolve(conn, at))


def test_gameday_on_reaches_the_stored_prediction():
    conn, gid, h, a = _b2b_conn()
    off = {r["game_id"]: r for r in predict.predict_day(conn, DAY, store=False)}[gid]
    injuries.set_inj_gameday_game(True)
    on = {r["game_id"]: r for r in predict.predict_day(conn, DAY, store=False)}[gid]
    assert [x["status"] for x in off["injuries"]["home"]] == ["Out"]
    assert [x["status"] for x in on["injuries"]["home"]] == ["Questionable"]
    assert on["inputs"]["injuries_gameday"] is True and "injuries_gameday" not in off["inputs"]
    assert on["home_win_prob"] != off["home_win_prob"]


class _State:
    def __init__(self, names):
        self.player_ids = list(range(len(names)))
        self.names = names


def test_players_availability_from_injuries_b2b():
    conn, gid, h, a = _b2b_conn()
    st = _State([f"{h} Player 0"])
    at = _z(DAY, "23:00")
    off = pl.availability_from_injuries(conn, st, h, at)
    assert off[0] in (0.0, 0.5)                    # OFF: whichever row came last (unchanged)
    injuries.set_inj_gameday_game(True)
    assert pl.availability_from_injuries(conn, st, h, at) == {0: 0.5}        # NBA day of `at`
    assert pl.availability_from_injuries(conn, st, h, at, game_day=NEXT) == {0: 0.0}
