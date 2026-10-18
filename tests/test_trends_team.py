"""Team pulse on synthetic leagues: lookahead, shrinkage, streaks, flags."""

import sqlite3

import pytest

from courtside.trends import team as T
from courtside.trends.synthetic import day_date, make_league


def league(**kw):
    conn = sqlite3.connect(":memory:")
    make_league(conn, **kw)
    return conn


def by_team(res):
    return {t.team: t for t in res}


def test_current_streak():
    assert T.current_streak([1, 0, 1, 1, 1]) == ("W", 3)
    assert T.current_streak([1, 1, 0, 0]) == ("L", 2)
    assert T.current_streak([]) == (None, 0)


def test_no_lookahead_games_on_or_after_as_of_are_ignored():
    conn = league()
    as_of = day_date("2024-25", 20)
    before = [t.to_dict() for t in T.team_pulse(conn, as_of)]
    # vandalise everything from as_of onward: blowouts, flipped results
    conn.execute("UPDATE team_games SET pts = pts + 60 WHERE game_date >= ? AND is_home = 0",
                 (as_of.isoformat(),))
    conn.execute("UPDATE team_games SET wl = 'W' WHERE game_date >= ? AND is_home = 0",
                 (as_of.isoformat(),))
    after = [t.to_dict() for t in T.team_pulse(conn, as_of)]
    assert before == after
    # the day after, the as_of-day games count
    later = by_team(T.team_pulse(conn, day_date("2024-25", 21)))
    assert sum(t.games for t in later.values()) == sum(t["games"] for t in before) + 6


def test_records_and_ratings_are_consistent():
    conn = league()
    res = T.team_pulse(conn, day_date("2024-25", 20))
    assert all(t.games == 20 for t in res)
    assert sum(t.record["w"] for t in res) == sum(t.record["l"] for t in res) == 60
    for t in res:
        assert t.home["w"] + t.home["l"] + t.road["w"] + t.road["l"] == 20
        assert t.season_raw["net"] == pytest.approx(t.season_raw["off"] - t.season_raw["def"],
                                                    abs=0.11)
    # strongest synthetic team (MIA, +5) should rank above the weakest (BOS, -5)
    ranks = {t.team: t.power["rank"] for t in res}
    assert ranks["MIA"] < ranks["BOS"]


def test_early_season_shrinkage_and_suppression():
    # BOS is the weakest team last season, then wins its first 3 games by 30
    def blowouts(season, day, h, a, m):
        if season == "2024-25" and "BOS" in (h, a):
            return 30 if h == "BOS" else -30
        return m

    conn = league(margin_fn=blowouts)
    early = by_team(T.team_pulse(conn, day_date("2024-25", 3)))["BOS"]
    assert early.games == 3
    assert early.season_raw["net"] == pytest.approx(30.0, abs=0.5)
    assert early.trend["flag"] is None
    assert "early season" in early.trend["suppressed"]
    # the prior (weak last season) and K_PRIOR pseudo-games keep the rating modest
    assert early.season_adj["net"] < 0.4 * early.season_raw["net"]
    assert not any(r["kind"] == "net_trend" for r in early.reasons)
    late = by_team(T.team_pulse(conn, day_date("2024-25", 28)))["BOS"]
    assert late.season_adj["net"] > early.season_adj["net"] + 5
    assert late.trend["suppressed"] is None


def test_streak_reason():
    def bos_streak(season, day, h, a, m):
        if season == "2024-25" and 13 <= day <= 19 and "BOS" in (h, a):
            return 6 if h == "BOS" else -6
        if season == "2024-25" and day == 12 and "BOS" in (h, a):
            return -4 if h == "BOS" else 4
        return m

    conn = league(margin_fn=bos_streak)
    bos = by_team(T.team_pulse(conn, day_date("2024-25", 20)))["BOS"]
    assert bos.streak == {"kind": "W", "n": 7, "label": "W7"}
    st = [r for r in bos.reasons if r["kind"] == "streak"]
    assert st and st[0]["detail"] == "won 7 straight"


def test_rising_and_sliding_flags():
    def form(season, day, h, a, m):
        if season == "2024-25" and day >= 20:
            if "BOS" in (h, a):           # weakest team gets hot
                return m + (22 if h == "BOS" else -22)
            if "MIA" in (h, a):           # strongest team falls apart
                return m + (-22 if h == "MIA" else 22)
        return m

    conn = league(seasons={"2023-24": 30, "2024-25": 30}, margin_fn=form)
    res = by_team(T.team_pulse(conn, day_date("2024-25", 30)))
    assert res["BOS"].trend["flag"] == "rising"
    assert res["MIA"].trend["flag"] == "sliding"
    assert res["BOS"].reasons[0]["kind"] == "net_trend"
    assert res["BOS"].trend["z"] >= T.Z_FLAG and res["BOS"].trend["delta"] >= T.MIN_DELTA
    # shrunk score is smaller than the raw gap but has the same sign
    assert 0 < res["BOS"].trend["score"] < res["BOS"].trend["delta"]
    flagged = [t for t in res.values() if t.trend["flag"]]
    assert {t.team for t in flagged} <= {"BOS", "MIA", "NYK", "LAL", "GSW", "DEN"}


def test_no_games_yet_uses_prior():
    conn = league()
    res = T.team_pulse(conn, day_date("2024-25", 0), season="2024-25")
    assert len(res) == 6 and all(t.games == 0 for t in res)
    assert res[0].team == "MIA"          # strongest last season
    assert all(t.trend["flag"] is None for t in res)


def test_fit_ratings_recovers_strengths():
    conn = league(seasons={"2024-25": 60}, noise=4.0)
    from courtside.trends.common import load_team_games
    g = load_team_games(conn, ["2024-25"], day_date("2024-25", 60))
    r = T.fit_ratings(g, None, k=1.0)
    # synthetic strengths are margins (pts per game, 100 poss) from -5 to +5
    assert r.net("MIA") - r.net("BOS") == pytest.approx(10.0, abs=2.5)
    assert r.hca == pytest.approx(2.0, abs=1.0)
