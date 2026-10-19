"""Assemble the daily Brief: one JSON document per NBA day (schema: docs/BRIEF.md).

`build_brief(conn, day)` is the *morning-of* Brief for `day`: `last_night` covers `day - 1`,
the pulses are computed from games before `day`, `tonight` is `day`'s slate. Everything
numeric is computed here from stored facts; prose lives only in the `text` slots, which an
optional writer (`courtside.brief.writer.fill`) fills afterwards. Without it, or when it
fails, the slots stay `null` and the Brief is still complete.

Sections: meta, headline, last_night, team_pulse, player_pulse, injury_watch, tonight,
scorecard.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, injuries, predict, teams
from ..model import metrics
from ..trends import player as trends_player
from ..trends import team as trends_team
from ..trends.common import as_date, load_player_games, prior_season, season_of
from ..trends.summarize import league_facts

log = logging.getLogger("courtside.brief")

SCHEMA_VERSION = "1.1"
MIN_REAL_TRAIN = 500
CONFIDENCE_TIERS = ((0.56, "toss-up"), (0.65, "lean"), (0.75, "solid"), (1.01, "strong"))
UPSET_MIN_FAVORITE = 0.60      # the market favourite must be at least this likely to win ...
UPSET_MIN_PROB = 0.40          # ... the model gives the underdog at least this chance ...
UPSET_MIN_EDGE = 0.05          # ... and that is at least this much above the market's price
EDGE_NOTABLE = 0.06            # |model - market| win-prob gap worth a `model_vs_market` factor
LONG_TRAVEL_KM = 2000.0
TZ_SHIFT_HOURS = 2.0
INJURY_FACTOR_MIN_MPG = 18.0
INJURY_FACTORS_PER_TEAM = 3      # highest-impact absences per team
OUT_TONIGHT_MAX = 15
CHANGES_MAX = 40
REAL_IDS = ("002", "004", "005")
# Models whose win probability ignores who is injured (results-only). Injuries are still shown
# as factors; a player-strength model should not be listed here.
RESULTS_ONLY_MODELS = ("home", "elo", "elo_rest", "elo_sched")


# ------------------------------------------------------------------ JSON hygiene

def clean(obj):
    """Plain JSON types only: numpy -> python, NaN/inf -> None, dates -> ISO strings."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _r(x, nd=3):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _team_ref(tri: str | None) -> dict:
    try:
        t = teams.by_tricode(tri)
        return {"team": t.tricode, "name": f"{t.city} {t.name}"}
    except (KeyError, AttributeError):
        return {"team": tri, "name": tri}


def _fr(code: str | None, season: str | None) -> str | None:
    if not code:
        return code
    try:
        return teams.franchise(code, season)
    except KeyError:
        return code


# ------------------------------------------------------------------ results of a day

def day_results(conn: sqlite3.Connection, day: date) -> dict[str, dict]:
    """Finished (or scheduled) games of an NBA day, merged from `team_games` and `games`.

    game_id -> {game_id, home, away, home_score, away_score, season_type, ot, status, final,
    tipoff_utc}. `games` wins for status/tip-off; scores come from whichever has them."""
    d = day.isoformat()
    out: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT game_id, team, season, season_type, is_home, pts, min FROM team_games "
        "WHERE game_date=? ORDER BY game_id, is_home DESC, team", (d,)).fetchall()
    by_game: dict[str, list] = {}
    for r in rows:
        by_game.setdefault(r["game_id"], []).append(r)
    for gid, pair in by_game.items():
        if len(pair) != 2 or not pair[0]["min"]:
            continue
        h, a = pair  # is_home DESC puts the home row first; neutral sites: alphabetical
        season = h["season"]
        out[gid] = {"game_id": gid, "home": _fr(h["team"], season), "away": _fr(a["team"], season),
                    "home_score": h["pts"], "away_score": a["pts"],
                    "season_type": h["season_type"],
                    "ot": max(0, int(round(((h["min"] or 240) - 240) / 25))),
                    "status": "final", "final": True, "tipoff_utc": None}
    for g in conn.execute("SELECT * FROM games WHERE nba_day=? ORDER BY game_id", (d,)):
        gid = g["game_id"]
        if gid[:3] not in REAL_IDS or g["status"] == "postponed":
            continue
        cur = out.get(gid)
        if cur is None:
            cur = out[gid] = {
                "game_id": gid, "home": g["home"], "away": g["away"],
                "home_score": None, "away_score": None,
                "season_type": {"002": "Regular Season", "004": "Playoffs",
                                "005": "PlayIn"}[gid[:3]],
                "ot": 0, "status": g["status"], "final": False, "tipoff_utc": None}
        cur["tipoff_utc"] = g["tipoff_utc"]
        if g["status"] == "final" and g["home_score"] is not None:
            cur.update(status="final", final=True, home_score=g["home_score"],
                       away_score=g["away_score"])
        elif not cur["final"]:
            cur["status"] = g["status"]
    return out


def _sorted_games(games: dict[str, dict]) -> list[dict]:
    return sorted(games.values(), key=lambda g: (g["tipoff_utc"] or "9", g["game_id"]))


# ------------------------------------------------------------------ market / pick helpers

def confidence(p_pick: float) -> str:
    for hi, label in CONFIDENCE_TIERS:
        if p_pick < hi:
            return label
    return CONFIDENCE_TIERS[-1][1]


def _market_block(pred: dict | None, line: dict | None) -> dict | None:
    """Prefer the market stored with the prediction (what we saw at the time)."""
    if pred and pred.get("inputs", {}).get("market"):
        m = pred["inputs"]["market"]
    elif line:
        m = line
    else:
        return None
    return {k: m.get(k) for k in ("source", "provider", "captured_at", "kind", "home_spread",
                                  "total", "home_ml", "away_ml", "home_prob")}


def _prediction_block(pred: dict | None, home: str, away: str) -> dict | None:
    if not pred or pred.get("home_win_prob") is None:
        return None
    p = float(pred["home_win_prob"])
    pick = home if p >= 0.5 else away
    pp = max(p, 1 - p)
    return {"model": pred["model"], "created_at": pred["created_at"],
            "home_win_prob": _r(p, 4), "away_win_prob": _r(1 - p, 4),
            "home_margin": _r(pred.get("home_margin"), 2), "total": _r(pred.get("total"), 1),
            "pick": pick, "pick_prob": _r(pp, 4), "confidence": confidence(pp)}


# ------------------------------------------------------------------ last night

def _load_facts(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    return {r["game_id"]: json.loads(r["facts_json"]) for r in conn.execute(
        f"SELECT game_id, facts_json FROM game_facts WHERE game_id IN ({q})", ids)}


def last_night_section(conn: sqlite3.Connection, ln_day: date, notes: list[str]) -> dict:
    games = _sorted_games(day_results(conn, ln_day))
    ids = [g["game_id"] for g in games]
    facts = _load_facts(conn, ids)
    preds = predict.latest_pretip(conn, ids)
    cards, correct, scored, fav_ok, fav_n = [], 0, 0, 0, 0
    for g in games:
        gid, h, a = g["game_id"], g["home"], g["away"]
        pred = preds.get(gid)
        pblock = _prediction_block(pred, h, a)
        line = None if pred and pred.get("inputs", {}).get("market") else \
            predict.market_line(conn, gid, g["tipoff_utc"])
        mk = _market_block(pred, line)
        result, ok, mres = None, None, None
        if g["final"] and g["home_score"] is not None:
            hs, as_ = int(g["home_score"]), int(g["away_score"])
            winner = h if hs > as_ else a
            result = {"home_score": hs, "away_score": as_, "winner": winner,
                      "margin": abs(hs - as_), "total": hs + as_, "overtimes": g["ot"]}
            if pblock:
                ok = pblock["pick"] == winner
                scored += 1
                correct += int(ok)
            if mk:
                mres = {"home_covered": None, "went_over": None, "favorite": None,
                        "favorite_won": None}
                sp = mk.get("home_spread")
                if sp is not None and (hs - as_) + sp != 0:
                    mres["home_covered"] = (hs - as_) + sp > 0
                if mk.get("total") is not None and hs + as_ != mk["total"]:
                    mres["went_over"] = hs + as_ > mk["total"]
                hp = mk.get("home_prob")
                fav = None
                if hp is not None and hp != 0.5:
                    fav = h if hp > 0.5 else a
                elif sp is not None and sp != 0:
                    fav = h if sp < 0 else a
                if fav:
                    mres["favorite"], mres["favorite_won"] = fav, fav == winner
                    fav_n += 1
                    fav_ok += int(fav == winner)
        cards.append({
            "game_id": gid, "status": g["status"], "final": g["final"],
            "season_type": g["season_type"],
            "home": {**_team_ref(h), "score": g["home_score"]},
            "away": {**_team_ref(a), "score": g["away_score"]},
            "result": result,
            "prediction": pblock,
            "correct": ok,
            "market": ({**mk, **(mres or {})} if mk else None),
            "facts": facts.get(gid),
            "facts_available": gid in facts,
            "text": {"headline": None, "story": None},
        })
    n_final = sum(1 for c in cards if c["final"])
    missing = [c["game_id"] for c in cards if c["final"] and not c["facts_available"]]
    if missing:
        notes.append(f"game facts missing for {len(missing)} of {n_final} finished games "
                     f"({', '.join(missing[:4])}{'...' if len(missing) > 4 else ''})")
    if cards and scored < n_final:
        notes.append(f"no stored pre-tip prediction for {n_final - scored} of {n_final} "
                     f"finished games")
    return {"day": ln_day.isoformat(), "games": cards,
            "summary": {"games": len(cards), "final": n_final, "picks_scored": scored,
                        "picks_correct": correct, "market_favorites_scored": fav_n,
                        "market_favorites_correct": fav_ok},
            "text": None}


# ------------------------------------------------------------------ pulses

def _with_text(items: list[dict]) -> list[dict]:
    return [{**i, "text": None} for i in items]


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason, "text": None}


def pulses(conn: sqlite3.Connection, day: date, notes: list[str]) -> tuple[dict, dict]:
    try:
        lf = league_facts(conn, as_of=day)
    except Exception as e:  # noqa: BLE001 - a broken trends engine must not sink the Brief
        log.exception("league_facts failed")
        notes.append(f"pulse sections unavailable: {type(e).__name__}: {e}")
        return _unavailable("league_facts failed"), _unavailable("league_facts failed")
    t, p = lf["teams"], lf["players"]
    common = {"available": True, "as_of": lf["as_of"], "season": lf["season"],
              "data_through": lf["data_through"], "early_season": lf["early_season"],
              "games_per_team_median": lf["games_per_team_median"], "notes": lf["notes"]}
    team_pulse = {**common,
                  "method": {k: lf["method"][k] for k in
                             ("net_ratings", "flag_rule", "flag_validity")},
                  "rising": _with_text(t["rising"]), "sliding": _with_text(t["sliding"]),
                  "power_top10": t["power_top10"], "power_movers_7d": t["power_movers_7d"],
                  "streaks": t["streaks"], "text": None}
    player_pulse = {**common,
                    "method": {k: lf["method"][k] for k in
                               ("impact_metric", "hot_cold_rule", "hot_cold_validity")},
                    "hot": _with_text(p["hot"]), "cold": _with_text(p["cold"]),
                    "role_changes": p["role_changes"], "leaders_last10": p["leaders_last10"],
                    "streaks": p["streaks"], "career_highs_last7d": p["career_highs_last7d"],
                    "text": None}
    return team_pulse, player_pulse


# ------------------------------------------------------------------ player impact index

def player_index(conn: sqlite3.Connection, day: date) -> dict[str, list[dict]]:
    """normalised player name -> [{team, mpg, impact, games, last_played}] from player_games
    before `day` (last 15 appearances this season, or including last season when the player has
    fewer than 5 this season). `impact` is the trends Courtside Impact metric."""
    season = season_of(day)
    try:
        pg = load_player_games(conn, [prior_season(season), season], day)
    except Exception:  # noqa: BLE001
        log.exception("player_index: could not load player games")
        return {}
    pg = pg[pg["min"] > 0]
    if pg.empty:
        return {}
    pg = pg.assign(ci=trends_player.impact_per_game(pg)).sort_values(["game_date", "game_id"])
    out: dict[str, list[dict]] = {}
    for _, g in pg.groupby("player_id", sort=False):
        cur = g[g["season"] == season]
        use = (cur if len(cur) >= 5 else g).tail(15)
        last = g.iloc[-1]
        out.setdefault(injuries.norm_name(last["player_name"]), []).append({
            "team": _fr(last["team"], last["season"]), "mpg": _r(use["min"].mean(), 1),
            "impact": _r(use["ci"].mean(), 1), "games": int(len(use)),
            "last_played": last["game_date"].strftime("%Y-%m-%d")})
    return out


def _impact_for(index: dict[str, list[dict]], player: str, team: str) -> dict | None:
    cands = index.get(injuries.norm_name(player)) or []
    same = [c for c in cands if c["team"] == team]
    pick = same[0] if same else (cands[0] if len(cands) == 1 else None)
    return pick


def model_effects(preds: dict[str, dict]) -> dict[tuple[str, str], float]:
    """(team, normalised player name) -> what the opinion model's lineup pricing says that
    player's status is worth to his team's expected margin, in points (negative = hurts his
    team). From the `inputs.model_effects` stored with the day's opinion predictions."""
    out = {}
    for p in preds.values():
        for e in (p.get("inputs") or {}).get("model_effects") or []:
            out[(e["team"], injuries.norm_name(e["player"]))] = e["effect_margin"]
    return out


def _enrich(entry: dict, index: dict, effects: dict | None = None) -> dict:
    imp = _impact_for(index, entry["player"], entry["team"])
    return {**entry, "mpg": imp["mpg"] if imp else None,
            "impact": imp["impact"] if imp else None,
            "impact_games": imp["games"] if imp else None,
            "last_played": imp["last_played"] if imp else None,
            "model_effect_margin": (effects or {}).get(
                (entry["team"], injuries.norm_name(entry["player"])))}


def _impact_key(e: dict):
    return (e.get("impact") is None, -(e.get("impact") or 0.0), e.get("player") or "")


# ------------------------------------------------------------------ tonight

def _team_form(conn: sqlite3.Connection, day: date, notes: list[str]) -> dict[str, dict]:
    try:
        return {t.team: {"record": f"{t.record['w']}-{t.record['l']}",
                         "last10": f"{t.last10_record['w']}-{t.last10_record['l']}",
                         "streak": t.streak["label"], "power_rank": t.power["rank"],
                         "last10_adj_net": t.last10_adj["net"], "games": t.games}
                for t in trends_team.team_pulse(conn, day)}
    except Exception as e:  # noqa: BLE001
        log.exception("team_pulse failed")
        notes.append(f"team form unavailable: {type(e).__name__}: {e}")
        return {}


def _factors(sched: dict, home: str, away: str, inj: dict, form: dict, market: dict | None,
             pred: dict, index: dict, model_factors: list[dict] | None = None) -> list[dict]:
    """Machine-readable reasons. Every factor: kind, team (whom it is about), side, value,
    favors ('home'|'away'|None), detail. The model's own top factors (kind "model_factor",
    from the opinion model's explain) come first; `pred` is the prediction the
    `model_vs_market` gap is measured with (the opinion model when there is one)."""
    out: list[dict] = [dict(f) for f in model_factors or []]
    sides = {"home": home, "away": away}
    opp = {"home": "away", "away": "home"}
    for s in ("home", "away"):
        f = sched[s]
        if f["b2b"]:
            out.append({"kind": "back_to_back", "team": sides[s], "side": s, "value": True,
                        "favors": opp[s], "detail": "second night of a back-to-back"})
        elif f["four_in_six"]:
            out.append({"kind": "four_in_six", "team": sides[s], "side": s, "value": True,
                        "favors": opp[s], "detail": "fourth game in six nights"})
        elif f["three_in_four"]:
            out.append({"kind": "three_in_four", "team": sides[s], "side": s, "value": True,
                        "favors": opp[s], "detail": "third game in four nights"})
        if (f["travel_km"] or 0) >= LONG_TRAVEL_KM:
            out.append({"kind": "long_travel", "team": sides[s], "side": s,
                        "value": _r(f["travel_km"], 0), "favors": opp[s],
                        "detail": f"{f['travel_km']:.0f} km since the last game"})
        if abs(f["tz_change"] or 0) >= TZ_SHIFT_HOURS:
            out.append({"kind": "time_zone", "team": sides[s], "side": s,
                        "value": _r(f["tz_change"], 0), "favors": opp[s],
                        "detail": f"{abs(f['tz_change']):.0f}-hour time-zone change"})
    rd = sched.get("rest_diff") or 0
    if abs(rd) >= 2:
        better = "home" if rd > 0 else "away"
        out.append({"kind": "rest_edge", "team": sides[better], "side": better,
                    "value": int(abs(rd)), "favors": better,
                    "detail": f"{abs(rd):.0f} more days of rest than the opponent"})
    if sched.get("is_neutral"):
        out.append({"kind": "neutral_site", "team": None, "side": None, "value": True,
                    "favors": None, "detail": "no home-court advantage"})
    for s in ("home", "away"):
        key = [e for e in inj.get(s, []) if (e.get("mpg") or 0) >= INJURY_FACTOR_MIN_MPG]
        for e in sorted(key, key=_impact_key)[:INJURY_FACTORS_PER_TEAM]:
            out.append({"kind": "injury", "team": sides[s], "side": s,
                        "value": {"player": e["player"], "status": e["status"],
                                  "mpg": e["mpg"], "impact": e["impact"]},
                        "favors": opp[s], "detail": f"{e['player']} ({e['status']})"})
    for s in ("home", "away"):
        fm = form.get(sides[s])
        if fm:
            out.append({"kind": "form", "team": sides[s], "side": s,
                        "value": {"record": fm["record"], "last10": fm["last10"],
                                  "streak": fm["streak"], "power_rank": fm["power_rank"]},
                        "favors": None,
                        "detail": f"{fm['last10']} in the last 10, power rank {fm['power_rank']}"})
    if market and market.get("home_prob") is not None and pred:
        gap = pred["home_win_prob"] - market["home_prob"]
        if abs(gap) >= EDGE_NOTABLE:
            side = "home" if gap > 0 else "away"
            out.append({"kind": "model_vs_market", "team": sides[side], "side": side,
                        "value": _r(abs(gap), 3), "favors": side,
                        "detail": f"model is {abs(gap) * 100:.0f} points more bullish on "
                                  f"{sides[side]} than the market"})
    return out


def upset_assessment(home: str, away: str, p_home: float | None,
                     mkt_home: float | None) -> dict | None:
    """Upset watch: a clear market favourite (>= UPSET_MIN_FAVORITE) that our model rates as
    meaningfully more beatable than the market does. Requires a market probability."""
    if p_home is None or mkt_home is None or mkt_home == 0.5:
        return None
    dog, dog_model, dog_mkt = (away, 1 - p_home, 1 - mkt_home) if mkt_home > 0.5 else \
        (home, p_home, mkt_home)
    fav = home if dog == away else away
    flag = (1 - dog_mkt >= UPSET_MIN_FAVORITE and dog_model >= UPSET_MIN_PROB
            and dog_model - dog_mkt >= UPSET_MIN_EDGE)
    return {"flag": bool(flag), "favorite": fav, "underdog": dog,
            "model_prob": _r(dog_model, 4), "market_prob": _r(dog_mkt, 4),
            "edge": _r(dog_model - dog_mkt, 4), "model_picks_underdog": bool(dog_model > 0.5)}


def _availability_view(pred: dict | None) -> dict | None:
    av = ((pred or {}).get("inputs") or {}).get("availability")
    if not av:
        return None
    return {s: {k: av[s].get(k) for k in ("basis", "source", "captured_at")}
            for s in ("home", "away") if s in av}


def tonight_section(conn: sqlite3.Connection, day: date, notes: list[str], index: dict,
                    form: dict, inj_now: dict) -> dict:
    base = day_results(conn, day)
    preds = predict.latest_pretip(conn, day=day.isoformat())
    if not preds and predict._enough_history(conn):
        try:
            predict.predict_day(conn, day)
            preds = predict.latest_pretip(conn, day=day.isoformat())
        except Exception as e:  # noqa: BLE001
            log.exception("predict_day failed")
            notes.append(f"tonight's predictions unavailable: {type(e).__name__}: {e}")
    opinions = predict.latest_pretip(conn, day=day.isoformat(), model=predict.PRODUCTION_OPINION) \
        if predict.PRODUCTION_OPINION != predict.PRODUCTION_MODEL else {}
    effects = model_effects(opinions)
    games = dict(base)
    for gid, p in preds.items():
        games.setdefault(gid, {"game_id": gid, "home": p["home"], "away": p["away"],
                               "season_type": p["inputs"].get("schedule", {}).get(
                                   "season_type", "Regular Season"),
                               "status": "scheduled", "final": False, "tipoff_utc": None})
    out, upsets = [], []
    for g in _sorted_games(games):
        gid, h, a = g["game_id"], g["home"], g["away"]
        pred = preds.get(gid)
        opin = opinions.get(gid)
        pblock = _prediction_block(pred, h, a)
        oblock = _prediction_block(opin, h, a)
        line = None if pred and pred.get("inputs", {}).get("market") else \
            predict.market_line(conn, gid, g.get("tipoff_utc"))
        mk = _market_block(pred, line)
        if mk and mk.get("home_prob") is None and mk.get("home_spread") is None \
                and mk.get("total") is None:
            mk = None
        sched = (pred or {}).get("inputs", {}).get("schedule")
        inj = {"home": [_enrich(e, index, effects) for e in inj_now["by_team"].get(h, [])],
               "away": [_enrich(e, index, effects) for e in inj_now["by_team"].get(a, [])]}
        for s in inj:
            inj[s].sort(key=lambda e: (-e["severity"],) + _impact_key(e))
        # "our number vs the market" is the independent model when we have it
        ours, ours_pred = (oblock, opin) if oblock else (pblock, pred)
        edge = None
        if ours and mk:
            edge = {"model": ours["model"],
                    "win_prob": (_r(ours["home_win_prob"] - mk["home_prob"], 4)
                                 if mk.get("home_prob") is not None else None),
                    "margin": (_r(ours["home_margin"] + mk["home_spread"], 2)
                               if mk.get("home_spread") is not None
                               and ours["home_margin"] is not None else None),
                    "total": (_r(ours["total"] - mk["total"], 1)
                              if mk.get("total") is not None and ours["total"] is not None
                              else None)}
        mfac = ((ours_pred or pred or {}).get("inputs") or {}).get("factors") or []
        factors = _factors(sched, h, a, inj, form, mk, ours_pred or pred, index, mfac) \
            if sched else []
        up = upset_assessment(h, a, ours["home_win_prob"] if ours else None,
                              mk.get("home_prob") if mk else None)
        if up and up["flag"]:
            upsets.append(gid)
        out.append({
            "game_id": gid, "tipoff_utc": g.get("tipoff_utc"), "status": g["status"],
            "season_type": g["season_type"],
            "home": {**_team_ref(h), **{k: form.get(h, {}).get(k) for k in
                                        ("record", "last10", "streak", "power_rank")}},
            "away": {**_team_ref(a), **{k: form.get(a, {}).get(k) for k in
                                        ("record", "last10", "streak", "power_rank")}},
            "prediction": pblock, "opinion": oblock, "market": mk, "edge": edge,
            "factors": factors, "availability": _availability_view(pred or opin),
            "injuries": inj, "upset_watch": bool(up and up["flag"]), "upset": up,
            "text": {"preview": None},
        })
    return {"day": day.isoformat(), "games": out, "upset_watch": upsets,
            "summary": {"games": len(out), "with_prediction": sum(1 for g in out
                                                                  if g["prediction"]),
                        "with_market": sum(1 for g in out if g["market"]),
                        "upset_watch": len(upsets)},
            "text": None}


# ------------------------------------------------------------------ injury watch

def injury_watch_section(conn: sqlite3.Connection, day: date, at: str, tonight: dict,
                         index: dict, inj_now: dict) -> dict:
    d = injuries.diff(conn, at)
    effects = model_effects(predict.latest_pretip(conn, day=day.isoformat(),
                                                  model=predict.PRODUCTION_OPINION))
    playing = {g["home"]["team"]: g for g in tonight["games"]}
    playing.update({g["away"]["team"]: g for g in tonight["games"]})

    def enrich(c: dict) -> dict:
        e = _enrich(c, index, effects)
        g = playing.get(c["team"])
        e["playing_tonight"] = g is not None
        e["game_id"] = g["game_id"] if g else None
        return e

    changes = {}
    for key in ("new_absences", "returns", "status_changes"):
        lst = sorted((enrich(c) for c in d[key]),
                     key=lambda e: (not e["playing_tonight"],) + _impact_key(e))
        changes[key] = lst[:CHANGES_MAX]
    out_tonight = []
    for team, lst in inj_now["by_team"].items():
        g = playing.get(team)
        if g is None:
            continue
        for e in lst:
            en = _enrich(e, index, effects)
            en["game_id"] = g["game_id"]
            en["opponent"] = g["away"]["team"] if g["home"]["team"] == team else g["home"]["team"]
            out_tonight.append(en)
    out_tonight.sort(key=lambda e: (e["severity"] < 2,) + _impact_key(e))
    return {"available": bool(inj_now["by_team"] or d["available"]),
            "as_of": at, "latest_snapshot": d["latest"] or inj_now.get("latest_captured_at"),
            "baseline_snapshot": d["baseline"], "diff_available": d["available"],
            "sources": d["sources"] or sorted(inj_now.get("batches", {})),
            "changes": changes, "out_tonight": out_tonight[:OUT_TONIGHT_MAX],
            "text": None}


# ------------------------------------------------------------------ scorecard

def _block(y, p) -> dict:
    return {"accuracy": _r(metrics.accuracy(y, p), 4), "brier": _r(metrics.brier(y, p), 4),
            "log_loss": _r(metrics.log_loss(y, p), 4)}


def score_rows(rows: list[dict]) -> dict:
    """Scorecard math on resolved predictions.

    Each row: p (P home win), y (1 = home won), optional p_mkt, margin_pred, margin (actual
    home margin), spread (home spread, negative = home favoured), total_pred, total (actual),
    market_total. Returns record/Brier/log-loss/accuracy/calibration, plus the market's numbers
    on exactly the games that carry a market probability."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "correct": 0, "incorrect": 0, "accuracy": None, "brier": None,
                "log_loss": None, "ece": None, "calibration": [], "vs_market": None,
                "by_confidence": []}
    y = np.array([r["y"] for r in rows], float)
    p = np.array([r["p"] for r in rows], float)
    correct = int(((p >= 0.5) == (y == 1)).sum())
    bins = 10 if n >= 200 else 5
    cal = [c for c in metrics.calibration_table(y, p, bins)]
    out = {"n": n, "correct": correct, "incorrect": n - correct,
           **_block(y, p), "ece": _r(metrics.ece(y, p, bins), 4),
           "mean_pick_prob": _r(np.maximum(p, 1 - p).mean(), 4),
           "calibration": [{"lo": _r(c["lo"], 3), "hi": _r(c["hi"], 3), "n": c["n"],
                            "mean_p": _r(c["mean_p"], 4), "frac_home_won": _r(c["frac_pos"], 4)}
                           for c in cal]}
    tiers = []
    pp = np.maximum(p, 1 - p)
    lo = 0.0
    for hi, label in CONFIDENCE_TIERS:
        m = (pp >= lo) & (pp < hi)
        if m.any():
            ok = ((p >= 0.5) == (y == 1))[m]
            tiers.append({"tier": label, "n": int(m.sum()), "correct": int(ok.sum()),
                          "accuracy": _r(ok.mean(), 4)})
        lo = hi
    out["by_confidence"] = tiers
    mk = [i for i, r in enumerate(rows) if r.get("p_mkt") is not None]
    if mk:
        ym, pm, pmk = y[mk], p[mk], np.array([rows[i]["p_mkt"] for i in mk], float)
        vs = {"n": len(mk), "model": _block(ym, pm), "market": _block(ym, pmk),
              "model_beats_market_brier": bool(metrics.brier(ym, pm) < metrics.brier(ym, pmk)),
              "brier_diff": _r(metrics.brier(ym, pm) - metrics.brier(ym, pmk), 4)}
        ms = [i for i in mk if rows[i].get("spread") is not None
              and rows[i].get("margin_pred") is not None and rows[i].get("margin") is not None]
        if ms:
            act = np.array([rows[i]["margin"] for i in ms], float)
            vs["margin_mae"] = {"n": len(ms),
                                "model": _r(np.mean(np.abs(act - [rows[i]["margin_pred"]
                                                                  for i in ms])), 2),
                                "market": _r(np.mean(np.abs(act + [rows[i]["spread"]
                                                                   for i in ms])), 2)}
        mt = [i for i in mk if rows[i].get("market_total") is not None
              and rows[i].get("total_pred") is not None and rows[i].get("total") is not None]
        if mt:
            act = np.array([rows[i]["total"] for i in mt], float)
            vs["total_mae"] = {"n": len(mt),
                               "model": _r(np.mean(np.abs(act - [rows[i]["total_pred"]
                                                                 for i in mt])), 2),
                               "market": _r(np.mean(np.abs(act - [rows[i]["market_total"]
                                                                  for i in mt])), 2)}
        out["vs_market"] = vs
    else:
        out["vs_market"] = None
    return out


def resolved_predictions(conn: sqlite3.Connection, start: str, end_excl: str, model: str
                         ) -> list[dict]:
    """Latest pre-tip predictions with nba_day in [start, end_excl) joined to final results."""
    preds = conn.execute(
        "SELECT * FROM latest_pretip_predictions WHERE model=? AND nba_day>=? AND nba_day<? "
        "ORDER BY nba_day, game_id", (model, start, end_excl)).fetchall()
    if not preds:
        return []
    ids = [r["game_id"] for r in preds]
    results: dict[str, tuple[int, int]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ",".join("?" * len(chunk))
        for g in conn.execute(f"SELECT game_id, home_score, away_score FROM games "
                              f"WHERE status='final' AND home_score IS NOT NULL "
                              f"AND game_id IN ({q})", chunk):
            results[g["game_id"]] = (g["home_score"], g["away_score"])
        pts: dict[str, dict[str, int]] = {}
        for t in conn.execute(f"SELECT game_id, team, season, pts, min FROM team_games "
                              f"WHERE game_id IN ({q})", chunk):
            if t["min"]:
                pts.setdefault(t["game_id"], {})[_fr(t["team"], t["season"])] = t["pts"]
        for r in preds[i:i + 500]:
            gid = r["game_id"]
            if gid not in results and r["home"] in pts.get(gid, {}) \
                    and r["away"] in pts.get(gid, {}):
                results[gid] = (pts[gid][r["home"]], pts[gid][r["away"]])
    rows = []
    for r in preds:
        res = results.get(r["game_id"])
        if res is None or r["home_win_prob"] is None or res[0] == res[1]:
            continue
        rows.append({"game_id": r["game_id"], "nba_day": r["nba_day"],
                     "p": r["home_win_prob"], "y": int(res[0] > res[1]),
                     "p_mkt": r["market_home_prob"], "margin_pred": r["home_margin"],
                     "margin": res[0] - res[1], "spread": r["market_home_spread"],
                     "total_pred": r["total"], "total": res[0] + res[1],
                     "market_total": r["market_total"]})
    return rows


def scorecard_section(conn: sqlite3.Connection, day: date, model: str | None = None) -> dict:
    model = model or predict.PRODUCTION_MODEL
    season = season_of(day)
    start = f"{season[:4]}-07-01"
    rows = resolved_predictions(conn, start, day.isoformat(), model)
    week = (day - timedelta(days=7)).isoformat()
    last = (day - timedelta(days=1)).isoformat()
    def rec(rs):
        c = sum(1 for r in rs if (r["p"] >= 0.5) == (r["y"] == 1))
        return {"n": len(rs), "correct": c}
    return {"model": model, "season": season, "through": last if rows else None,
            "scope": "latest pre-tip prediction per game vs final result, season to date",
            **score_rows(rows),
            "recent": {"last_night": rec([r for r in rows if r["nba_day"] == last]),
                       "last_7_days": rec([r for r in rows if r["nba_day"] >= week])},
            "opinion": _opinion_scorecard(conn, season, start, day),
            "text": None}


def _opinion_scorecard(conn: sqlite3.Connection, season: str, start: str, day: date
                       ) -> dict | None:
    """The independent (no-market) model on the same footing, season to date: the honest
    "us vs the market" line. Null while it has no resolved predictions."""
    model = predict.PRODUCTION_OPINION
    if model == predict.PRODUCTION_MODEL:
        return None
    rows = resolved_predictions(conn, start, day.isoformat(), model)
    if not rows:
        return None
    sc = score_rows(rows)
    return {"model": model, "season": season,
            **{k: sc[k] for k in ("n", "correct", "incorrect", "accuracy", "brier", "log_loss",
                                  "ece", "vs_market")}}


# ------------------------------------------------------------------ meta / freshness

def _freshness(conn: sqlite3.Connection, day: date, tonight: dict, last_night: dict) -> dict:
    def one(sql, *a):
        r = conn.execute(sql, a).fetchone()
        return r[0] if r else None
    tids = [g["game_id"] for g in tonight["games"]]
    odds = None
    if tids:
        odds = one(f"SELECT MAX(captured_at) FROM odds_snapshots WHERE game_id IN "
                   f"({','.join('?' * len(tids))})", *tids)
    mp = [g["prediction"]["created_at"] for g in tonight["games"] if g["prediction"]]
    return {
        "team_games_through": one("SELECT MAX(game_date) FROM team_games WHERE game_date<?",
                                  day.isoformat()),
        "player_games_through": one("SELECT MAX(game_date) FROM player_games WHERE game_date<?",
                                    day.isoformat()),
        "odds_latest_captured_at": odds,
        "injuries_latest_captured_at": one("SELECT MAX(captured_at) FROM injury_snapshots "
                                           "WHERE captured_at<=?",
                                           predict.injury_cutoff(day, predict._iso_now())),
        "predictions_latest_created_at": max(mp) if mp else None,
        "last_night_games": last_night["summary"]["games"],
        "last_night_games_with_facts": sum(1 for g in last_night["games"]
                                           if g["facts_available"]),
    }


# ------------------------------------------------------------------ build

def brief_path(day: date) -> Path:
    return config.data_dir() / "briefs" / f"{day.isoformat()}.json"


def _fill_text(brief: dict) -> dict:
    """Run the optional writer; any failure leaves the null slots."""
    if os.environ.get("COURTSIDE_NO_WRITER"):
        return brief
    try:
        from courtside.brief import writer
    except ImportError:
        return brief
    except Exception:  # noqa: BLE001
        log.exception("brief.writer import failed; text slots stay null")
        return brief
    snapshot = copy.deepcopy(brief)
    try:
        res = writer.fill(brief)
        out = res if isinstance(res, dict) else brief
        json.dumps(out, allow_nan=False)
        return out
    except Exception:  # noqa: BLE001
        log.exception("writer.fill failed; text slots stay null")
        return snapshot


def _without_text(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "text"}


def _tonight_sig(g: dict) -> dict:
    """What a preview is written from: change any of it and the prose is stale."""
    p, m = g.get("prediction") or {}, g.get("market") or {}
    return {"pick": p.get("pick"), "prob": round(p.get("home_win_prob") or 0, 2),
            "margin": round(p.get("home_margin") or 0), "spread": m.get("home_spread"),
            "total": m.get("total"), "upset": g.get("upset_watch"),
            "factors": [(f["kind"], f["team"], str(f["value"])) for f in g.get("factors", [])]}


def carry_text(old: dict | None, new: dict) -> int:
    """Copy already-written prose from the previously stored Brief of the same day into slots
    whose underlying content is unchanged, so a rebuild (8 per day) does not re-call the LLM
    for text that is still true. Returns the number of slots carried."""
    if not old or old.get("meta", {}).get("schema_version") != new["meta"]["schema_version"]:
        return 0
    n = 0

    def put(dst: dict, key: str, val) -> None:
        nonlocal n
        if val and not dst.get(key):
            dst[key] = val
            n += 1

    old_cards = {c["game_id"]: c for c in old.get("last_night", {}).get("games", [])}
    for c in new["last_night"]["games"]:
        o = old_cards.get(c["game_id"])
        if o and o.get("facts") == c.get("facts") and c.get("facts") is not None:
            for k in ("headline", "story"):
                put(c["text"], k, (o.get("text") or {}).get(k))
    for sec, lists in (("team_pulse", ("rising", "sliding")), ("player_pulse", ("hot", "cold"))):
        if not new[sec].get("available") or not old.get(sec, {}).get("available"):
            continue
        idk = "team" if sec == "team_pulse" else "player_id"
        for lst in lists:
            olds = {e[idk]: e for e in old[sec].get(lst, [])}
            for e in new[sec][lst]:
                o = olds.get(e[idk])
                if o and _without_text(o) == _without_text(e):
                    put(e, "text", o.get("text"))
    old_games = {g["game_id"]: g for g in old.get("tonight", {}).get("games", [])}
    for g in new["tonight"]["games"]:
        o = old_games.get(g["game_id"])
        if o and _tonight_sig(o) == _tonight_sig(g):
            put(g["text"], "preview", (o.get("text") or {}).get("preview"))
    # section-level prose: only when nothing under it changed
    same_pulse = {k: new[k].get("available") and old.get(k, {}).get("available")
                  and [_without_text(e) for e in new[k].get("rising", new[k].get("hot", []))]
                  == [_without_text(e) for e in old[k].get("rising", old[k].get("hot", []))]
                  for k in ("team_pulse", "player_pulse")}
    for k in ("team_pulse", "player_pulse"):
        if same_pulse[k]:
            put(new[k], "text", old[k].get("text"))
    all_same = (all(same_pulse.values())
                and [c["game_id"] for c in new["last_night"]["games"]]
                == [c["game_id"] for c in old.get("last_night", {}).get("games", [])]
                and [_tonight_sig(g) for g in new["tonight"]["games"]]
                == [_tonight_sig(g) for g in old.get("tonight", {}).get("games", [])])
    if all_same:
        put(new["headline"], "text", (old.get("headline") or {}).get("text"))
    return n


def _availability_counts(tonight: dict) -> dict:
    """How many team-sides of tonight's slate were priced from each availability basis:
    injury_report | regulars_played | full_roster (see predict.py)."""
    out: dict[str, int] = {}
    for g in tonight["games"]:
        for side in ((g.get("availability") or {}).values()):
            out[side["basis"]] = out.get(side["basis"], 0) + 1
    return out


def build_brief(conn: sqlite3.Connection, day, *, write: bool = True,
                run_writer: bool = True, model: str | None = None) -> dict:
    """Build (and store) the morning-of Brief for `day`. See docs/BRIEF.md."""
    day = as_date(day)
    ln_day = day - timedelta(days=1)
    notes: list[str] = []
    t0 = datetime.now(timezone.utc)

    last_night = last_night_section(conn, ln_day, notes)
    team_pulse, player_pulse = pulses(conn, day, notes)
    index = player_index(conn, day)
    form = _team_form(conn, day, notes)
    at = predict.injury_cutoff(day, predict._iso_now())
    # tonight's teams (for the injury snapshot filter) come from the slate itself
    slate_teams = set()
    for r in conn.execute("SELECT home, away FROM latest_pretip_predictions WHERE nba_day=?",
                          (day.isoformat(),)):
        slate_teams |= {r["home"], r["away"]}
    for g in day_results(conn, day).values():
        slate_teams |= {g["home"], g["away"]}
    inj_now = injuries.current(conn, at, slate_teams or None)
    tonight = tonight_section(conn, day, notes, index, form, inj_now)
    if not tonight["games"]:
        notes.append(f"no games scheduled for {day.isoformat()}")
    injury_watch = injury_watch_section(conn, day, at, tonight, index, inj_now)
    scorecard = scorecard_section(conn, day, model)
    freshness = _freshness(conn, day, tonight, last_night)
    through = freshness["team_games_through"]
    if through and through < ln_day.isoformat() and last_night["games"]:
        notes.append(f"team_games ends {through}, before last night ({ln_day}); trends and "
                     f"ratings do not include it")
    if not injury_watch["available"]:
        notes.append("no injury snapshots available")
    elif not injury_watch["diff_available"]:
        notes.append("injury changes unavailable: need two snapshots at least 12 hours apart")
    model_name = model or predict.PRODUCTION_MODEL
    basis = _availability_counts(tonight)
    if basis.get("regulars_played"):
        notes.append(f"availability for {basis['regulars_played']} teams comes from the "
                     f"rotation regulars who played (backfilled/demo day), not an injury report")
    if basis.get("full_roster"):
        notes.append(f"no injury report covers {basis['full_roster']} teams tonight: the model "
                     f"assumes their full recent roster is available")
    if model_name in RESULTS_ONLY_MODELS:
        notes.append(f"model {model_name} is results-only: injuries are listed as factors but "
                     f"not priced into win probability or margin")
    if scorecard["n"] == 0:
        notes.append("scorecard is empty: no stored pre-tip predictions with results yet")

    brief = {
        "meta": {
            "schema_version": SCHEMA_VERSION, "day": day.isoformat(),
            "last_night_day": ln_day.isoformat(), "season": season_of(day),
            "built_at": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "model": model_name,
            "model_uses_injuries": model_name not in RESULTS_ONLY_MODELS,
            "opinion_model": (predict.PRODUCTION_OPINION
                              if model_name == predict.PRODUCTION_MODEL else None),
            "model_uses_market": model_name == predict.PRODUCTION_MODEL,
            "availability_basis": basis,
            "data_freshness": freshness, "notes": notes, "text_filled": False,
            "path": f"briefs/{day.isoformat()}.json",   # relative to the data dir
        },
        "headline": {"text": None},
        "last_night": last_night,
        "team_pulse": team_pulse,
        "player_pulse": player_pulse,
        "injury_watch": injury_watch,
        "tonight": tonight,
        "scorecard": scorecard,
    }
    brief = clean(brief)
    carried = carry_text(load_brief(conn, day), brief)
    if carried:
        log.info("brief %s: carried %d text slots from the previous build", day, carried)
    if run_writer:
        brief = clean(_fill_text(brief))
    brief["meta"]["text_filled"] = bool((brief.get("headline") or {}).get("text"))
    payload = json.dumps(brief, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
    with conn:
        conn.execute("INSERT OR REPLACE INTO briefs (nba_day, brief_json, built_at, version) "
                     "VALUES (?,?,?,?)",
                     (day.isoformat(), payload, brief["meta"]["built_at"], SCHEMA_VERSION))
    if write:
        path = brief_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(brief, allow_nan=False, indent=1, ensure_ascii=False))
        os.replace(tmp, path)
    return brief


def load_brief(conn: sqlite3.Connection, day) -> dict | None:
    r = conn.execute("SELECT brief_json FROM briefs WHERE nba_day=?",
                     (as_date(day).isoformat(),)).fetchone()
    return json.loads(r[0]) if r else None
