"""build_brief on a synthetic league: sections, determinism, scorecard math, upset logic."""

from __future__ import annotations

import json
import math
from datetime import timedelta

import numpy as np
import pytest
from synth import day_date, synth_conn

from courtside import config, predict
from courtside.brief import assemble, show
from courtside.model import metrics


@pytest.fixture(autouse=True)
def small_history(monkeypatch):
    monkeypatch.setattr(predict, "MIN_TRAIN_GAMES", 20)


DAY = day_date("2024-25", 30)
SECTIONS = {"meta", "headline", "last_night", "team_pulse", "player_pulse", "injury_watch",
            "tonight", "scorecard"}


def seed(conn):
    """Predictions for the whole 2024-25 run-in, two injury snapshots, facts for one game."""
    predict.predict_days(conn, day_date("2024-25", 8), DAY)
    ln = (DAY - timedelta(days=1)).isoformat()
    gid = conn.execute("SELECT game_id FROM team_games WHERE game_date=? LIMIT 1",
                       (ln,)).fetchone()[0]
    conn.execute("INSERT INTO game_facts VALUES (?,?,?,?)",
                 (gid, ln, json.dumps({"game_id": gid, "turning_point": {"run": "10-0"}}), "x"))
    ins = ("INSERT INTO injury_snapshots (captured_at, source, team, player_name, status, "
           "reason) VALUES (?,?,?,?,?,?)")
    conn.execute(ins, (f"{DAY - timedelta(days=1)}T22:00:00Z", "nba_official", "BOS",
                       "BOS Player 0", "Questionable", "x"))
    conn.execute(ins, (f"{DAY}T22:00:00Z", "nba_official", "BOS", "BOS Player 0", "Out", "knee"))
    conn.execute(ins, (f"{DAY}T22:00:00Z", "nba_official", "BOS", "BOS Player 5", "Out", "x"))
    conn.commit()
    return gid


def test_sections_present_json_clean_and_stored():
    conn = synth_conn()
    gid = seed(conn)
    b = assemble.build_brief(conn, DAY)
    assert set(b) == SECTIONS
    json.dumps(b, allow_nan=False)
    m = b["meta"]
    assert m["schema_version"] == assemble.SCHEMA_VERSION == "1.0"
    assert m["day"] == DAY.isoformat() and m["last_night_day"] == (DAY - timedelta(1)).isoformat()
    assert m["text_filled"] is False and b["headline"] == {"text": None}
    assert m["data_freshness"]["team_games_through"] == m["last_night_day"]

    ln = b["last_night"]
    assert len(ln["games"]) == 3 and ln["summary"]["final"] == 3
    card = next(c for c in ln["games"] if c["game_id"] == gid)
    assert card["facts_available"] and card["facts"]["turning_point"]["run"] == "10-0"
    assert card["text"] == {"headline": None, "story": None}
    hs, as_ = card["result"]["home_score"], card["result"]["away_score"]
    assert card["result"]["winner"] == (card["home"]["team"] if hs > as_ else card["away"]["team"])
    assert card["prediction"]["pick"] in (card["home"]["team"], card["away"]["team"])
    assert card["correct"] == (card["prediction"]["pick"] == card["result"]["winner"])
    others = [c for c in ln["games"] if c["game_id"] != gid]
    assert all(not c["facts_available"] and c["facts"] is None for c in others)
    assert any("game facts missing" in n for n in m["notes"])

    for sec in ("team_pulse", "player_pulse"):
        assert b[sec]["available"] and b[sec]["text"] is None and "as_of" in b[sec]
    assert {"rising", "sliding", "power_top10"} <= set(b["team_pulse"])
    assert len(b["team_pulse"]["power_top10"]) == 6
    assert {"hot", "cold", "leaders_last10", "role_changes"} <= set(b["player_pulse"])

    t = b["tonight"]
    assert len(t["games"]) == 3 and t["summary"]["with_prediction"] == 3
    g = t["games"][0]
    assert {"prediction", "market", "edge", "factors", "injuries", "upset_watch", "text"} <= set(g)
    assert g["text"] == {"preview": None}
    assert g["prediction"]["home_win_prob"] + g["prediction"]["away_win_prob"] == pytest.approx(1)
    assert any(f["kind"] == "form" for f in g["factors"])
    for f in g["factors"]:
        assert {"kind", "team", "side", "value", "favors", "detail"} <= set(f)

    sc = b["scorecard"]
    assert sc["n"] > 0 and sc["through"] == m["last_night_day"]
    assert sc["correct"] + sc["incorrect"] == sc["n"] and sc["text"] is None

    # stored in the table and on disk, identical
    row = conn.execute("SELECT brief_json, version FROM briefs WHERE nba_day=?",
                       (DAY.isoformat(),)).fetchone()
    assert json.loads(row["brief_json"]) == b and row["version"] == "1.0"
    on_disk = json.loads((config.data_dir() / m["path"]).read_text())
    assert on_disk == b
    assert assemble.load_brief(conn, DAY) == b
    assert "COURTSIDE MORNING BRIEF" in show.format_brief(b)


def test_injury_watch_ranks_and_diffs():
    conn = synth_conn()
    seed(conn)
    b = assemble.build_brief(conn, DAY)
    iw = b["injury_watch"]
    assert iw["available"] and iw["diff_available"]
    assert [c["player"] for c in iw["changes"]["status_changes"]] == ["BOS Player 0"]
    assert [c["player"] for c in iw["changes"]["new_absences"]] == ["BOS Player 5"]
    outs = iw["out_tonight"]
    assert {e["player"] for e in outs} == {"BOS Player 0", "BOS Player 5"}
    # Player 0 plays 36 mpg with the biggest scoring share: highest impact ranks first
    assert outs[0]["player"] == "BOS Player 0" and outs[0]["mpg"] > outs[1]["mpg"]
    assert outs[0]["impact"] > outs[1]["impact"] and outs[0]["model_effect_margin"] is None
    bos = [g for g in b["tonight"]["games"] if "BOS" in (g["home"]["team"], g["away"]["team"])]
    assert bos and any(f["kind"] == "injury" for f in bos[0]["factors"])


def test_empty_database_still_builds_a_complete_brief():
    conn = synth_conn({"2024-25": 1})
    conn.execute("DELETE FROM team_games"); conn.execute("DELETE FROM player_games")
    b = assemble.build_brief(conn, DAY, write=False)
    assert set(b) == SECTIONS
    assert b["last_night"]["games"] == [] and b["tonight"]["games"] == []
    assert b["scorecard"]["n"] == 0 and b["scorecard"]["accuracy"] is None
    assert b["injury_watch"]["available"] is False
    json.dumps(b, allow_nan=False)
    assert show.format_brief(b)


def test_writer_failure_leaves_nulls_and_success_is_used(monkeypatch):
    import sys
    import types
    conn = synth_conn()
    monkeypatch.delenv("COURTSIDE_NO_WRITER")

    def boom(brief):
        brief["headline"]["text"] = "partial"      # mutated in place, then fails
        raise RuntimeError("llm down")
    w = types.ModuleType("courtside.brief.writer")
    w.fill = boom
    monkeypatch.setitem(sys.modules, "courtside.brief.writer", w)
    monkeypatch.setattr("courtside.brief.writer", w, raising=False)
    b = assemble.build_brief(conn, DAY, write=False)
    assert b["headline"]["text"] is None and b["meta"]["text_filled"] is False

    def ok(brief):
        brief["headline"]["text"] = "Quiet night."
        return brief
    w.fill = ok
    b = assemble.build_brief(conn, DAY, write=False)
    assert b["headline"]["text"] == "Quiet night." and b["meta"]["text_filled"] is True


# ------------------------------------------------------------------ scorecard math

def test_score_rows_math_by_hand():
    rows = [
        {"p": 0.8, "y": 1, "p_mkt": 0.7},
        {"p": 0.6, "y": 0, "p_mkt": 0.55},
        {"p": 0.3, "y": 0, "p_mkt": None},
        {"p": 0.5, "y": 1, "p_mkt": 0.4},
    ]
    s = assemble.score_rows(rows)
    assert (s["n"], s["correct"], s["incorrect"]) == (4, 3, 1)       # 0.5 counts as a home pick
    assert s["accuracy"] == 0.75
    assert s["brier"] == pytest.approx((0.04 + 0.36 + 0.09 + 0.25) / 4, abs=1e-4)
    ll = -(math.log(0.8) + math.log(0.4) + math.log(0.7) + math.log(0.5)) / 4
    assert s["log_loss"] == pytest.approx(ll, abs=1e-4)
    v = s["vs_market"]
    assert v["n"] == 3                                               # market on its own games
    ym, pm, pk = np.array([1, 0, 1]), np.array([0.8, 0.6, 0.5]), np.array([0.7, 0.55, 0.4])
    assert v["model"]["brier"] == pytest.approx(metrics.brier(ym, pm), abs=1e-4)
    assert v["market"]["brier"] == pytest.approx(metrics.brier(ym, pk), abs=1e-4)
    assert v["model"]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert v["market"]["accuracy"] == pytest.approx(1 / 3, abs=1e-4)   # 0.7 right, 0.55 and 0.4 wrong
    assert v["model_beats_market_brier"] is (v["brier_diff"] < 0)
    assert sum(c["n"] for c in s["calibration"]) == 4 and len(s["calibration"]) == 5
    assert sum(t["n"] for t in s["by_confidence"]) == 4


def test_score_rows_margin_and_total_vs_market():
    rows = [
        {"p": 0.7, "y": 1, "p_mkt": 0.6, "margin_pred": 5.0, "margin": 10, "spread": -3.0,
         "total_pred": 220.0, "total": 230, "market_total": 225.0},
        {"p": 0.4, "y": 0, "p_mkt": 0.45, "margin_pred": -2.0, "margin": -6, "spread": 1.0,
         "total_pred": 210.0, "total": 200, "market_total": 205.0},
    ]
    v = assemble.score_rows(rows)["vs_market"]
    # model errors 5 and 4; market margins +3 and -1 give errors 7 and 5
    assert v["margin_mae"] == {"n": 2, "model": 4.5, "market": 6.0}
    assert v["total_mae"] == {"n": 2, "model": 10.0, "market": 5.0}


def test_score_rows_empty_and_no_market():
    e = assemble.score_rows([])
    assert e["n"] == 0 and e["accuracy"] is None and e["calibration"] == []
    s = assemble.score_rows([{"p": 0.9, "y": 1}, {"p": 0.2, "y": 0}])
    assert s["vs_market"] is None and s["correct"] == 2


def test_scorecard_uses_only_pretip_predictions_before_day():
    conn = synth_conn()
    seed(conn)
    sc = assemble.scorecard_section(conn, DAY)
    rows = assemble.resolved_predictions(conn, "2024-07-01", DAY.isoformat(), predict.PRODUCTION_MODEL)
    assert sc["n"] == len(rows) == 66                      # 22 days x 3 games, DAY excluded
    assert max(r["nba_day"] for r in rows) < DAY.isoformat()
    # recompute straight from the tables
    q = conn.execute("""SELECT p.home_win_prob p, h.pts hp, a.pts ap FROM predictions p
                        JOIN team_games h ON h.game_id=p.game_id AND h.team=p.home
                        JOIN team_games a ON a.game_id=p.game_id AND a.team=p.away
                        WHERE p.nba_day < ?""", (DAY.isoformat(),)).fetchall()
    y = np.array([r["hp"] > r["ap"] for r in q], float)
    p = np.array([r["p"] for r in q])
    assert sc["brier"] == pytest.approx(metrics.brier(y, p), abs=1e-4)
    assert sc["correct"] == int(((p >= 0.5) == (y == 1)).sum())
    # a prediction created after tip-off is ignored in favour of the earlier one
    gid = rows[-1]["game_id"]
    conn.execute("INSERT INTO games (game_id, season, nba_day, tipoff_utc, home, away, status) "
                 "VALUES (?,?,?,?,?,?,?)", (gid, "2024-25", rows[-1]["nba_day"],
                                            f"{rows[-1]['nba_day']}T23:00:00Z", "X", "Y", "final"))
    conn.execute("UPDATE games SET home_score=NULL WHERE game_id=?", (gid,))
    conn.execute("INSERT INTO predictions (game_id, model, created_at, home_win_prob, nba_day, "
                 "home, away) SELECT game_id, model, ?, 0.001, nba_day, home, away "
                 "FROM predictions WHERE game_id=? LIMIT 1",
                 (f"{rows[-1]['nba_day']}T23:59:00Z", gid))
    conn.commit()
    again = assemble.resolved_predictions(conn, "2024-07-01", DAY.isoformat(),
                                          predict.PRODUCTION_MODEL)
    assert next(r["p"] for r in again if r["game_id"] == gid) == rows[-1]["p"]


# ------------------------------------------------------------------ upset watch

def test_upset_assessment_rules():
    f = assemble.upset_assessment
    # market says NYK (away) is a 70% favourite; we give BOS (home) 45%: flag
    u = f("BOS", "NYK", 0.45, 0.30)
    assert u["flag"] and u["underdog"] == "BOS" and u["favorite"] == "NYK"
    assert u["edge"] == pytest.approx(0.15) and u["model_picks_underdog"] is False
    # favourite only 55% in the market: not a real favourite
    assert not f("BOS", "NYK", 0.50, 0.45)["flag"]
    # model agrees with the market: no flag
    assert not f("BOS", "NYK", 0.32, 0.30)["flag"]
    # model barely gives the underdog a chance
    assert not f("BOS", "NYK", 0.20, 0.10)["flag"]
    # home favourite, away underdog
    u = f("BOS", "NYK", 0.58, 0.80)
    assert u["flag"] and u["underdog"] == "NYK" and u["model_prob"] == pytest.approx(0.42)
    # no market -> no assessment
    assert f("BOS", "NYK", 0.6, None) is None and f("BOS", "NYK", None, 0.6) is None
    assert assemble.confidence(0.55) == "toss-up" and assemble.confidence(0.9) == "strong"


def test_rebuild_carries_prose_whose_content_is_unchanged(monkeypatch):
    conn = synth_conn()
    gid = seed(conn)
    first = assemble.build_brief(conn, DAY, write=False)
    # pretend a writer filled things in
    stored = json.loads(conn.execute("SELECT brief_json FROM briefs").fetchone()[0])
    card = next(c for c in stored["last_night"]["games"] if c["game_id"] == gid)
    card["text"] = {"headline": "H", "story": "S"}
    stored["tonight"]["games"][0]["text"]["preview"] = "P0"
    stored["tonight"]["games"][1]["text"]["preview"] = "P1"
    stored["headline"]["text"] = "Top"
    conn.execute("UPDATE briefs SET brief_json=?", (json.dumps(stored),))
    # change the first tonight game's market so its preview is stale; everything else same
    g0 = stored["tonight"]["games"][0]["game_id"]
    conn.execute("INSERT INTO closing_lines (game_id, source, provider, home_spread_close, "
                 "total_close, home_ml_close, away_ml_close) VALUES (?, 'espn', 'x', -13.5, 250, "
                 "-900, 600)", (g0,))
    conn.execute("UPDATE predictions SET market_home_spread=-13.5, market_total=250, "
                 "market_home_prob=0.88, inputs_json=json_remove(inputs_json, '$.market') "
                 "WHERE game_id=?", (g0,))
    conn.commit()
    second = assemble.build_brief(conn, DAY, write=False)
    c2 = next(c for c in second["last_night"]["games"] if c["game_id"] == gid)
    assert c2["text"] == {"headline": "H", "story": "S"}          # facts unchanged: kept
    t = {g["game_id"]: g["text"]["preview"] for g in second["tonight"]["games"]}
    assert t[g0] is None                                          # market moved: stale, dropped
    assert t[stored["tonight"]["games"][1]["game_id"]] == "P1"    # unchanged: kept
    assert second["headline"]["text"] is None                     # a tonight game changed
    assert first["meta"]["built_at"] <= second["meta"]["built_at"]
