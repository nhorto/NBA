"""``league_facts``: the compact, JSON-serializable trends document for the Brief writer.

Every claim carries the numbers behind it (and the window it covers), so the writer can quote
rather than invent and the fact-check pass can verify. Built only from games before ``as_of``.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any

import pandas as pd

from courtside.trends import player as P
from courtside.trends import team as T
from courtside.trends.common import (
    as_date,
    jsonable,
    load_player_games,
    load_team_games,
    prior_season,
    season_of,
)

SCHEMA_VERSION = 1
POWER_TOP = 10
ROLE_MAX = 20
ROLE_PER_KIND = 6
ABSENCE_MPG = 24.0           # key absence for team context
ROLE_OUT_MAX_MISSED = 10       # only *recent* disappearances; long-term absences aren't news
STREAKS_MAX = 10
HIGHS_DAYS = 7
HIGHS_MAX = 10
TEAM_STREAK_MIN = 5
MOVER_MIN = 3                  # power-rank change over 7 days

ROLE_PRIORITY = ["starter_minutes", "new_in_rotation", "minutes_up", "usage_up", "new_team",
                 "new_arrival", "lost_starter_minutes", "minutes_down", "usage_down",
                 "out_of_rotation"]


def _team_entry(t: T.TeamTrend, roles: list[dict] | None = None) -> dict:
    return {
        "team": t.team, "name": t.name, "record": f"{t.record['w']}-{t.record['l']}",
        "last10": f"{t.last10_record['w']}-{t.last10_record['l']}", "streak": t.streak["label"],
        "power_rank": t.power["rank"],
        "net": {"season_raw": t.season_raw["net"], "last10_raw": t.last10_raw["net"],
                "season_adj": t.season_adj["net"], "last10_adj": t.last10_adj["net"]},
        "off": {"season": t.season_raw["off"], "last10": t.last10_raw["off"]},
        "def": {"season": t.season_raw["def"], "last10": t.last10_raw["def"]},
        "trend": {k: t.trend[k] for k in ("delta", "z", "baseline", "score")},
        "sos_last10": t.sos_last10,
        "reasons": t.reasons,
        "key_absences": _absences(roles or [], t.team),
    }


def _power_entry(t: T.TeamTrend) -> dict:
    return {"rank": t.power["rank"], "team": t.team, "record": f"{t.record['w']}-{t.record['l']}",
            "power": t.power["score"], "adj_net": t.season_adj["net"],
            "last10_adj_net": t.last10_adj["net"], "change_7d": t.power["change_7d"]}


def _group_roles(roles: list[dict]) -> list[dict]:
    keep = [r for r in roles
            if r["kind"] != "out_of_rotation" or r["team_games_missed"] <= ROLE_OUT_MAX_MISSED]
    by: dict[int, dict] = {}
    for r in keep:
        e = by.setdefault(r["player_id"], {"player_id": r["player_id"], "player": r["player"],
                                           "team": r["team"], "kinds": [], "details": [],
                                           "numbers": {}})
        e["kinds"].append(r["kind"])
        e["details"].append(r["detail"])
        for k, v in r.items():
            if k not in ("player_id", "player", "team", "kind", "detail"):
                e["numbers"][f"{r['kind']}.{k}"] = v
    pri = {k: i for i, k in enumerate(ROLE_PRIORITY)}
    ranked = sorted(by.values(), key=lambda e: (min(pri[k] for k in e["kinds"]),
                                                -len(e["kinds"])))
    out, per_kind = [], {}
    for e in ranked:  # cap each headline kind so one kind can't crowd out the rest
        head = min(e["kinds"], key=lambda k: pri[k])
        if per_kind.get(head, 0) >= ROLE_PER_KIND:
            continue
        per_kind[head] = per_kind.get(head, 0) + 1
        out.append(e)
    return out[:ROLE_MAX]


def _absences(roles: list[dict], team: str) -> list[dict]:
    """Rotation players currently missing for ``team`` (context for a rising/sliding claim)."""
    return [{"player": r["player"], "team_games_missed": r["team_games_missed"],
             "mpg_before": r["mpg_before"], "last_played": r["last_played"]}
            for r in roles if r["kind"] == "out_of_rotation" and r["team"] == team
            and r["mpg_before"] >= ABSENCE_MPG]


def _leaders_compact(lead: dict[str, list[dict]]) -> dict:
    out = {}
    for cat, rows in lead.items():
        out[cat] = [{"player": r["player"], "team": r["team"], "value": r["value"],
                     "games": r["games"], "line": f"{r['pts']}/{r['reb']}/{r['ast']}",
                     "ts_pct": r["ts_pct"], "min": r["min"], "impact": r["impact"]}
                    for r in rows]
    return out


def compute(conn: sqlite3.Connection | None, tcur: pd.DataFrame, tprev: pd.DataFrame,
            pg: pd.DataFrame, as_of: date, season: str) -> dict[str, Any]:
    trends = T.compute(tcur, tprev, as_of, season)
    pp = P.compute(conn, pg, as_of, season)
    gp = tcur.groupby("team").size() if len(tcur) else pd.Series(dtype=int)
    median_gp = int(gp.median()) if len(gp) else 0
    early = median_gp < T.MIN_GAMES_FLAG
    notes = []
    if median_gp == 0:
        notes.append(f"no {season} regular-season games before {as_of}; power ranking is the "
                     f"prior-season rating regressed {int(100 * (1 - T.RHO))}% to the mean")
    elif early:
        notes.append(f"early season ({median_gp} games per team): rising/sliding flags are "
                     f"suppressed until {T.MIN_GAMES_FLAG} games; power ranking still leans on "
                     f"the prior season ({T.K_PRIOR:.0f} pseudo-games)")
    notes += pp.get("notes", [])
    if 0 < median_gp < P.LEAD_WINDOW:
        notes.append(f"leaders cover each team's first {median_gp} games, not a full 10")
    cutoff = (as_of - timedelta(days=HIGHS_DAYS)).isoformat()
    teams_streaks = [{"team": t.team, "streak": t.streak["label"], "record": f"{t.record['w']}-"
                      f"{t.record['l']}", "detail": f"{'won' if t.streak['kind'] == 'W' else 'lost'}"
                      f" {t.streak['n']} straight"}
                     for t in trends if t.streak["n"] >= TEAM_STREAK_MIN]
    teams_streaks.sort(key=lambda x: -int(x["streak"][1:]))
    movers = [{"team": t.team, "rank": t.power["rank"], "rank_7d_ago": t.power["rank_7d_ago"],
               "change_7d": t.power["change_7d"]}
              for t in trends if abs(t.power["change_7d"]) >= MOVER_MIN and not early]
    movers.sort(key=lambda x: -abs(x["change_7d"]))
    last_game = tcur["game_date"].max() if len(tcur) else None
    doc = {
        "schema": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "season": season,
        "data_through": last_game.strftime("%Y-%m-%d") if last_game is not None else None,
        "games_per_team_median": median_gp,
        "early_season": early,
        "notes": notes,
        "method": {
            "net_ratings": "points per 100 possessions; adj = opponent- and venue-adjusted",
            "flag_rule": f"last-10 adj net vs baseline (earlier games + {T.K_PRIOR:.0f} "
                         f"pseudo-games of prior season), |z| >= {T.Z_FLAG}, "
                         f"|gap| >= {T.MIN_DELTA}, >= {T.MIN_GAMES_FLAG} games",
            "flag_validity": T.FLAG_VALIDITY,
            "impact_metric": pp["impact_metric"],
            "hot_cold_validity": P.HOT_VALIDITY,
            "hot_cold_rule": f"|z| >= {P.HOT_Z} vs own baseline, rotation players "
                             f"(>= {P.HOT_MIN_MPG:.0f} mpg, >= {P.HOT_MIN_GAMES} games)",
        },
        "teams": {
            "rising": [_team_entry(t, pp["role_changes"]) for t in sorted(trends, key=lambda t: -(t.trend["z"] or 0))
                       if t.trend["flag"] == "rising"],
            "sliding": [_team_entry(t, pp["role_changes"]) for t in sorted(trends, key=lambda t: t.trend["z"] or 0)
                        if t.trend["flag"] == "sliding"],
            "power_top10": [_power_entry(t) for t in trends[:POWER_TOP]],
            "power_movers_7d": movers,
            "streaks": teams_streaks,
        },
        "players": {
            "hot": pp["hot"], "cold": pp["cold"],
            "role_changes": _group_roles(pp["role_changes"]) if not early else [],
            "leaders_last10": _leaders_compact(pp["leaders_last10"]),
            "streaks": pp["streaks"][:STREAKS_MAX],
            "career_highs_last7d": [h for h in pp["career_highs"] if h["date"] >= cutoff]
            [:HIGHS_MAX],
        },
    }
    return jsonable(doc)


def league_facts(conn: sqlite3.Connection, as_of: date, season: str | None = None) -> dict:
    """The trends facts document for the morning of ``as_of`` (games before it only)."""
    as_of = as_date(as_of)
    season = season or season_of(as_of)
    tcur = load_team_games(conn, [season], as_of)
    tprev = load_team_games(conn, [prior_season(season)], as_of)
    pg = load_player_games(conn, [prior_season(season), season], as_of)
    return compute(conn, tcur, tprev, pg, as_of, season)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - demo helper
    import argparse
    import json
    import time

    from courtside.trends.common import connect_ro
    ap = argparse.ArgumentParser(description="print league_facts JSON for a date")
    ap.add_argument("as_of")
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    doc = league_facts(connect_ro(a.db), date.fromisoformat(a.as_of))
    print(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"# {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
