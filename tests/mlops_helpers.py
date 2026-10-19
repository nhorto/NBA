"""Tiny builders for the MLOps monitoring tests (synthetic rows, no network)."""

from __future__ import annotations

import json

from courtside import db


def conn_for(tmp_path):
    return db.connect(tmp_path / "t.db")


def game(conn, gid, day, hs, as_, home="BOS", away="NYK", tip=None, status="final"):
    db.upsert_game(conn, game_id=gid, season="2025-26", nba_day=day, tipoff_utc=tip,
                   home=home, away=away, status=status, home_score=hs, away_score=as_)


def pred(conn, gid, day, p, *, model="gbm_market", created="2026-01-01T12:00:00Z", margin=None,
         total=None, role=None, version=None, inputs=None, mkt_p=None, mkt_spread=None,
         mkt_total=None, home="BOS", away="NYK"):
    conn.execute(
        "INSERT OR REPLACE INTO predictions (game_id, model, created_at, home_win_prob, "
        "home_margin, total, market_home_spread, market_total, market_home_prob, inputs_json, "
        "nba_day, home, away, model_version, role) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (gid, model, created, p, margin, total, mkt_spread, mkt_total, mkt_p,
         json.dumps(inputs or {}), day, home, away, version, role))
    conn.commit()


def odds(conn, gid, home_ml, away_ml, *, captured="2026-01-01T10:00:00Z", spread=None, total=None):
    db.insert_odds_snapshot(conn, game_id=gid, captured_at=captured, source="espn",
                            provider="x", home_ml=home_ml, away_ml=away_ml,
                            home_spread=spread, total=total)
