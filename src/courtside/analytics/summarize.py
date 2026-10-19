"""``game_facts``: the compact, JSON-serializable facts document the Brief writer consumes.

Every number in here is derived from the raw boxscore + play-by-play payloads and nothing else, so
the writer can be fact-checked against it. Keys are stable; times are given as
``{"label": "Q4", "clock": "3:52", "score": {"OKC": 91, "HOU": 97}}``. Scores in the document are
always keyed by tricode to avoid home/away ambiguity.

Rules (documented in docs/ANALYTICS.md):

* **Top performers**: two per team by Hollinger Game Score.
* **Unsung hero**: see ``unsung_score`` / ``UNSUNG_*``.
* **Notable lines**: see ``NOTABLE`` thresholds.
"""

from __future__ import annotations

import re
from typing import Any

from courtside.analytics import game_story as gs
from courtside.analytics import winprob
from courtside.analytics.pbp import (
    AWAY, HOME, REG_PERIODS, Event, final_score, fmt_clock, normalize, period_end, period_label,
)
from courtside.analytics.stints import SIDES, box_players, build_rotation, validate_stints

SCHEMA_VERSION = 1

# --- unsung hero -----------------------------------------------------------------------------
UNSUNG_MIN_MINUTES = 12.0       # non-garbage minutes
UNSUNG_MAX_POINTS = 20          # "impact outstripped points": not a big scorer ...
UNSUNG_EXCLUDE_TOP_SCORERS = 2  # ... and not one of his team's top-2 scorers
LEAGUE_TS = 0.57                # par true-shooting for "points over par"
ONOFF_SHRINK_POSS = 40.0        # on/off shrinkage: x * poss / (poss + 40)
UNSUNG_BENCH_BONUS = 1.0

UNSUNG_MIN_SCORE = 2.0         # candidates below this are not offered at all
LATE_DEFENSE_CLOCK = 120.0     # clutch steals/blocks listed as key plays only inside the last 2:00
MAX_KEY_PLAYS = 8               # keep the latest 8 clutch key plays

# --- notable lines ---------------------------------------------------------------------------
NOTABLE = {
    "pts_30": 30, "pts_40": 40, "pts_50": 50,
    "reb_15": 15, "ast_12": 12, "threes_5": 5, "stocks_5": 5, "tov_7": 7,
    "perfect_fg_min_fga": 5, "perfect_ft_min_fta": 8,
    "efficient_min_pts": 25, "efficient_ts": 0.70,
    "cold_min_fga": 15, "cold_fg_pct": 0.30,
    "bench_pts": 20,
}

# --- "did not return" heuristic --------------------------------------------------------------
INJURY_CUTOFF_ELAPSED = 2 * 720.0 + 6 * 60.0   # last on the floor before 6:00 of the 3rd
INJURY_MAX_MINUTES = 20.0


# ------------------------------------------------------------------------- helpers

def _tricodes(box: dict) -> dict[str, str]:
    g = box.get("game") or {}
    return {HOME: (g.get("homeTeam") or {}).get("teamTricode"),
            AWAY: (g.get("awayTeam") or {}).get("teamTricode")}


def _score(tc: dict[str, str], score: list[int] | tuple[int, int]) -> dict[str, int]:
    return {tc[HOME]: score[0], tc[AWAY]: score[1]}


def _when(tc: dict[str, str], m: dict[str, Any] | None) -> dict[str, Any] | None:
    if not m:
        return None
    return {"label": m["label"], "clock": m["clock"], "score": _score(tc, m["score"])}


def _ref(p: dict[str, Any]) -> dict[str, Any]:
    return {"id": p["player_id"], "name": p["name"], "team": p["team"]}


def _line(p: dict[str, Any]) -> str:
    parts = [f"{p['pts']} pts ({p['fgm']}-{p['fga']} FG"
             + (f", {p['fg3m']}-{p['fg3a']} 3P" if p["fg3a"] else "")
             + (f", {p['ftm']}-{p['fta']} FT" if p["fta"] else "") + ")"]
    for k, lab in (("reb", "reb"), ("ast", "ast"), ("stl", "stl"), ("blk", "blk")):
        if p[k] >= (3 if k in ("reb", "ast") else 2):
            parts.append(f"{p[k]} {lab}")
    return ", ".join(parts)


def _run_fact(tc, events, r: gs.Run, garbage: gs.Span, core: gs.Run | None = None) -> dict[str, Any]:
    d = r.to_dict(events)
    out = {"team": tc[r.side], "run": d["label"], "unanswered": r.unanswered, "minutes": d["minutes"],
           "from": _when(tc, d["start"]), "to": _when(tc, d["end"])}
    if core is not None:
        out["includes_unanswered"] = core.label
    if r.end_idx in garbage.events:
        out["garbage_time"] = True
    return out


def notable_flags(p: dict[str, Any]) -> list[str]:
    n = NOTABLE
    flags: list[str] = []
    for k in ("pts_50", "pts_40", "pts_30"):
        if p["pts"] >= n[k]:
            flags.append(f"{n[k]}_point_game")
            break
    tens = sum(p[k] >= 10 for k in ("pts", "reb", "ast", "stl", "blk"))
    if tens >= 3:
        flags.append("triple_double")
    if p["pts"] >= 20 and (p["reb"] >= 10 or p["ast"] >= 10):
        flags.append("20_10")
    if p["reb"] >= n["reb_15"]:
        flags.append(f"{p['reb']}_rebounds")
    if p["ast"] >= n["ast_12"]:
        flags.append(f"{p['ast']}_assists")
    if p["fg3m"] >= n["threes_5"]:
        flags.append(f"{p['fg3m']}_threes")
    if p["stocks"] >= n["stocks_5"]:
        flags.append(f"{p['stocks']}_stocks")
    if p["fga"] >= n["perfect_fg_min_fga"] and p["fgm"] == p["fga"]:
        flags.append("perfect_fg")
    if p["fta"] >= n["perfect_ft_min_fta"] and p["ftm"] == p["fta"]:
        flags.append("perfect_ft")
    if p["pts"] >= n["efficient_min_pts"] and (p["ts_pct"] or 0) >= n["efficient_ts"]:
        flags.append("efficient_scoring")
    if p["fga"] >= n["cold_min_fga"] and p["fgm"] / p["fga"] <= n["cold_fg_pct"]:
        flags.append("cold_shooting")
    if not p["starter"] and p["pts"] >= n["bench_pts"]:
        flags.append("bench_scoring")
    if p["tov"] >= n["tov_7"]:
        flags.append(f"{p['tov']}_turnovers")
    return flags


def unsung_score(p: dict[str, Any]) -> float:
    """Unsung-hero score (higher = more impact not reflected in points).

        0.5 * non-garbage on-court +/-
      + 0.1 * shrunk on/off net rating     (on_off * on_poss / (on_poss + 40))
      + hustle                             (stl + 0.7 blk + 0.7 ast + 0.7 oreb + 0.3 dreb - tov)
      + 0.5 * points over par              (pts - 2 * 0.57 * (FGA + 0.44 FTA))
      + 1.0 if he came off the bench
    """
    onoff = p["on_off"] or 0.0
    shrunk = onoff * p["on_poss"] / (p["on_poss"] + ONOFF_SHRINK_POSS)
    hustle = (p["stl"] + 0.7 * p["blk"] + 0.7 * p["ast"] + 0.7 * p["oreb"]
              + 0.3 * (p["reb"] - p["oreb"]) - p["tov"])
    over_par = p["pts"] - 2 * LEAGUE_TS * (p["fga"] + 0.44 * p["fta"])
    score = (0.5 * p["plus_minus_non_garbage"] + 0.1 * shrunk + hustle + 0.5 * over_par
             + (UNSUNG_BENCH_BONUS if not p["starter"] else 0.0))
    return round(score, 2)


def unsung_candidates(impact: list[dict[str, Any]], winner_side: str | None, n: int = 3) -> list[dict[str, Any]]:
    top_scorers: set[int] = set()
    for s in SIDES:
        team = sorted((p for p in impact if p["side"] == s), key=lambda p: (-p["pts"], p["player_id"]))
        top_scorers |= {p["player_id"] for p in team[:UNSUNG_EXCLUDE_TOP_SCORERS]}
    pool = [p for p in impact
            if p["minutes_non_garbage"] >= UNSUNG_MIN_MINUTES and p["pts"] <= UNSUNG_MAX_POINTS
            and p["player_id"] not in top_scorers and unsung_score(p) >= UNSUNG_MIN_SCORE]
    ranked = sorted(pool, key=lambda p: (-(p["side"] == winner_side), -unsung_score(p), p["player_id"]))
    out = []
    for p in ranked[:n]:
        why = []
        if p["plus_minus_non_garbage"] > 0:
            why.append(f"+{p['plus_minus_non_garbage']} on the floor")
        if p["on_off"] is not None and p["on_off"] > 0:
            why.append(f"team +{p['on_off']:.1f} per 100 better with him on")
        if p["stocks"] >= 3:
            why.append(f"{p['stocks']} stocks")
        if p["ts_pct"] is not None and p["ts_pct"] >= 0.65 and p["fga"] >= 4:
            why.append(f"{p['ts_pct']:.0%} TS")
        if p["oreb"] >= 3:
            why.append(f"{p['oreb']} offensive rebounds")
        if p["ast"] >= 5:
            why.append(f"{p['ast']} assists")
        if p["winning_time"]["seconds"] > 0 and p["winning_time"]["plus_minus"] > 0:
            why.append(f"+{p['winning_time']['plus_minus']} during the decisive run")
        out.append({**_ref(p), "bench": not p["starter"], "winner": p["side"] == winner_side,
                    "score": unsung_score(p), "minutes": p["minutes"], "line": _line(p),
                    "plus_minus": p["plus_minus_non_garbage"], "on_off": p["on_off"],
                    "ts_pct": p["ts_pct"], "stocks": p["stocks"], "why": why})
    return out


# ------------------------------------------------------------------------- injuries / ejections

_INJ = re.compile(r"injur", re.I)


def incidents(events: list[Event], stints, impact: list[dict[str, Any]], garbage: gs.Span,
              names: dict[int, str], tc: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Ejections (explicit ejection events, 2 technicals, flagrant 2), foul-outs, explicit injury
    mentions in the feed, and the heuristic "left the game and did not return" for starters."""
    ejections, foul_outs, injuries = [], [], []
    techs: dict[int, int] = {}
    pfs: dict[int, int] = {}
    ejected: set[int] = set()

    def who(e: Event) -> dict[str, Any]:
        return {"id": e.player_id, "name": names.get(e.player_id or 0), "team": e.team,
                "label": period_label(e.period), "clock": fmt_clock(e.clock)}

    for e in events:
        st, desc = e.sub_type.lower(), (e.descriptor or "").lower()
        if e.is_ejection and e.player_id and e.player_id not in ejected:
            ejected.add(e.player_id)
            ejections.append({**who(e), "reason": e.sub_type or "ejection"})
        if e.is_foul and e.player_id:
            if e.foul_kind == "technical" and "3-second" not in desc and "delay" not in desc:
                techs[e.player_id] = techs.get(e.player_id, 0) + 1
                if techs[e.player_id] == 2 and e.player_id not in ejected:
                    ejected.add(e.player_id)
                    ejections.append({**who(e), "reason": "two technical fouls"})
            if e.foul_kind == "flagrant" and ("2" in st or "type2" in desc.replace(" ", "")) \
                    and e.player_id not in ejected:
                ejected.add(e.player_id)
                ejections.append({**who(e), "reason": "flagrant 2"})
            if e.foul_kind in ("personal", "offensive", "flagrant"):
                pfs[e.player_id] = pfs.get(e.player_id, 0) + 1
                if pfs[e.player_id] == 6:
                    foul_outs.append(who(e))
        if _INJ.search(e.sub_type) or _INJ.search(e.descriptor or "") or _INJ.search(e.description):
            injuries.append({**who(e), "source": "feed", "description": e.description})

    # Heuristic: a starter who played under 20 minutes and whose last stint ended before 6:00 of
    # the 3rd quarter, outside garbage time, while at least three other starters on his team kept
    # playing well after -> "did not return" (possible injury; verify against news / ESPN).
    last_end: dict[int, float] = {}
    for st in stints:
        for pid in st.home + st.away:
            last_end[pid] = max(last_end.get(pid, 0.0), st.end)
    cutoff = INJURY_CUTOFF_ELAPSED
    starters = {p["player_id"]: p for p in impact if p["starter"]}
    for pid, p in starters.items():
        t = last_end.get(pid)
        if t is None or t >= cutoff or p["minutes"] >= INJURY_MAX_MINUTES or pid in ejected \
                or garbage.contains_time(t + 1):
            continue
        mates = [last_end.get(q, 0) for q, o in starters.items() if o["side"] == p["side"] and q != pid]
        if sum(1 for m in mates if m > t + 120) >= 3:
            injuries.append({"id": pid, "name": p["name"], "team": p["team"], "source": "heuristic",
                             "note": "left the game and did not return",
                             "last_on_floor": _elapsed_label(t), "minutes": p["minutes"]})
    return {"ejections": ejections, "foul_outs": foul_outs, "possible_injuries": injuries}


def _elapsed_label(t: float, start: bool = False) -> dict[str, str]:
    """Elapsed seconds -> period label + clock. At a period boundary, ``start=True`` names the
    period that begins there (Q4 12:00), otherwise the one that ends there (Q3 0:00)."""
    p = 1
    while (period_end(p) <= t + 1e-9) if start else (period_end(p) < t - 1e-9):
        p += 1
    return {"label": period_label(p), "clock": fmt_clock(round(period_end(p) - t, 1))}


# ------------------------------------------------------------------------- facts

WP_TIMELINE_POINTS = 100
WP_TOP_PLAYS = 3
_NOT_PLAYS = ("substitution", "timeout", "period", "game")


def wp_facts(events: list[Event], points: list[dict[str, Any]], tc: dict[str, str],
             names: dict[int, str], w: str | None, pregame_spread: float | None) -> dict[str, Any]:
    """Win-probability keys for the facts document (all from the ``winprob.series`` points)."""
    by_idx = {e.idx: e for e in events}
    plays = []
    for a, b in zip(points, points[1:]):
        e = by_idx[b["idx"]]
        if e.action_type in _NOT_PLAYS:
            continue
        plays.append((b["home_wp"] - a["home_wp"], a["home_wp"], e))
    top = sorted(plays, key=lambda x: (-abs(x[0]), x[2].idx))[:WP_TOP_PLAYS]
    biggest = []
    for d, before, e in top:
        if d == 0:
            continue
        biggest.append({
            "description": e.description, "player": names.get(e.player_id or 0),
            "team": e.team, "toward": tc[HOME] if d > 0 else tc[AWAY],
            "wp_swing": round(abs(d), 3), "home_wp_before": round(before, 3),
            "home_wp_after": round(before + d, 3), **(_when(tc, gs.moment(e)) or {})})
    comeback = None
    if w is not None:
        mine = [(p["home_wp"] if w == HOME else 1.0 - p["home_wp"], p["idx"]) for p in points[1:]]
        if mine:
            lo, idx = min(mine, key=lambda m: (m[0], m[1]))
            comeback = {"team": tc[w], "min_wp": round(lo, 3), "at": _when(tc, gs.moment(by_idx[idx]))}
    return {
        "wp_model": {"version": "winprob_v1", "pregame_spread": pregame_spread},
        "wp_timeline": winprob.timeline(points, WP_TIMELINE_POINTS),
        "biggest_wp_plays": biggest,
        "excitement": round(sum(abs(d) for d, _, _ in plays), 2),
        "comeback_wp": comeback,
    }


def game_facts(box_payload: dict, pbp_payload: dict, win_prob: list[tuple[int, float]] | None = None,
               pregame_spread: float | None = None) -> dict[str, Any]:
    """The per-game facts document. Deterministic for a given pair of payloads.

    ``pregame_spread`` is the home spread (negative = home favoured, as in ``closing_lines``); it is
    the prior of the win-probability model. ``win_prob`` overrides the model series used for the
    turning point (``[(event_idx, p_home)]``)."""
    g = box_payload.get("game") or {}
    tc = _tricodes(box_payload)
    events = normalize(pbp_payload, box_payload)
    rot = build_rotation(events, box_payload)
    stints = rot.stints
    report = validate_stints(box_payload, stints, rot.anomalies)
    players = box_players(box_payload)
    names = {pid: p["name"] for pid, p in players.items()}
    starters = [pid for pid, p in players.items() if p["starter"]]

    garbage = gs.garbage_time(events, stints, starters)
    runs = gs.scoring_runs(events)
    wp_points = winprob.series(events, pregame_spread)
    tp = gs.turning_point(events, runs, win_prob=win_prob or winprob.wp_pairs(wp_points),
                          exclude=garbage)
    impact = gs.player_game_impact(box_payload, stints, events, garbage=garbage, runs=runs, tp=tp)
    by_id = {p["player_id"]: p for p in impact}
    leads = gs.lead_changes_and_ties(events)
    largest = gs.largest_lead(events)
    cb = gs.comeback(events)
    cl = gs.clutch(events, stints)

    h, a = final_score(events)
    bh, ba = (g.get("homeTeam") or {}).get("score"), (g.get("awayTeam") or {}).get("score")
    if bh is not None and ba is not None:
        h, a = int(bh), int(ba)
    w = HOME if h > a else AWAY if a > h else None
    n_per = max((e.period for e in events), default=REG_PERIODS)

    def team_block(side: str) -> dict[str, Any]:
        t = g.get("homeTeam" if side == HOME else "awayTeam") or {}
        return {"team": t.get("teamTricode"), "name": f"{t.get('teamCity', '')} {t.get('teamName', '')}".strip(),
                "score": h if side == HOME else a,
                "periods": [int(p.get("score") or 0) for p in t.get("periods") or []]}

    # top performers: 2 per team by game score
    top = []
    for s in SIDES:
        team = sorted((p for p in impact if p["side"] == s), key=lambda p: (-p["game_score"], p["player_id"]))
        for p in team[:2]:
            top.append({**_ref(p), "game_score": p["game_score"], "line": _line(p),
                        "minutes": p["minutes"], "plus_minus": p["plus_minus"], "ts_pct": p["ts_pct"]})

    # runs: stretch runs with any unanswered core folded in; top 5 by net, chronological
    un = [r for r in runs if r.unanswered]
    stretches = [r for r in runs if not r.unanswered]
    shown = []
    used_cores: set[int] = set()
    for r in stretches:
        core = max((u for u in un if u.side == r.side and r.start_idx <= u.start_idx and u.end_idx <= r.end_idx),
                   key=lambda u: u.pts, default=None)
        if core is not None:
            used_cores.add(core.start_idx)
        shown.append((r, core))
    shown += [(u, None) for u in un if u.start_idx not in used_cores
              and not any(r.start_idx <= u.start_idx and u.end_idx <= r.end_idx for r in stretches)]
    shown = sorted(shown, key=lambda rc: (-rc[0].net, rc[0].start_idx))[:5]
    shown.sort(key=lambda rc: rc[0].start_idx)

    # turning point
    tp_fact = None
    if tp:
        tp_fact = {"method": tp["method"], "team": tc[tp["side"]], "reason": tp.get("reason")}
        if tp.get("run"):
            r = tp["run"]
            tp_fact.update(run=r["label"], minutes=r["minutes"], **{"from": _when(tc, r["start"]), "to": _when(tc, r["end"])})
        if tp.get("wp_swing") is not None:
            tp_fact.update(wp_swing=tp["wp_swing"], at=_when(tc, tp["at"]), description=tp["description"],
                           wp_before=tp.get("wp_before"), wp_after=tp.get("wp_after"))
            kp = tp.get("key_play")
            if kp:
                tp_fact["key_play"] = {"description": kp["description"],
                                       "player": names.get(kp["player_id"] or 0),
                                       "wp_swing": kp["wp_swing"], **_when(tc, kp)}

    # clutch
    clutch_fact: dict[str, Any] = {"seconds": cl["seconds"], "had_clutch": cl["seconds"] > 0}
    if cl["seconds"] > 0:
        clutch_fact["points"] = {tc[s]: cl["points"][s] for s in SIDES}
        cp = []
        for r in cl["players"]:
            if not (r.get("pts") or r.get("fga") or r.get("stl") or r.get("blk") or r.get("ast")):
                continue
            p = by_id.get(r["player_id"])
            if p is None:
                continue
            cp.append({**_ref(p), "pts": r.get("pts", 0), "fg": f"{r.get('fgm', 0)}-{r.get('fga', 0)}",
                       "ft": f"{r.get('ftm', 0)}-{r.get('fta', 0)}", "ast": r.get("ast", 0),
                       "stl": r.get("stl", 0), "blk": r.get("blk", 0), "tov": r.get("tov", 0),
                       "plus_minus": r["plus_minus"]})
        clutch_fact["players"] = cp[:6]
        kp = []
        for k in cl["key_plays"]:
            if k["kind"] in ("steal", "block") and k["clock_s"] > LATE_DEFENSE_CLOCK:
                continue  # defensive key plays: only the final 2:00 of the 4th / any OT
            kp.append({"kind": k["kind"], "player": names.get(k["player_id"] or 0),
                       "team": tc.get(k["side"]) if k["side"] else None,
                       **(_when(tc, k) or {}), **({"decisive": True} if k.get("decisive") else {})})
        clutch_fact["key_plays"] = kp[-MAX_KEY_PLAYS:]

    notable = []
    for p in sorted(impact, key=lambda p: (-p["game_score"], p["player_id"])):
        flags = notable_flags(p)
        if flags:
            notable.append({**_ref(p), "flags": flags, "line": _line(p)})

    def lead_fact(side: str) -> dict[str, Any] | None:
        m = largest[side]
        return {"lead": m["lead"], **_when(tc, m)} if m else None

    gt_fact: dict[str, Any] = {"seconds": garbage.seconds}
    if garbage.intervals:
        gt_fact["from"] = _elapsed_label(garbage.intervals[0][0], start=True)

    return {
        "schema": SCHEMA_VERSION,
        "game_id": g.get("gameId"),
        "date": (g.get("gameEt") or "")[:10] or None,
        "arena": (g.get("arena") or {}).get("arenaName"),
        "home": team_block(HOME),
        "away": team_block(AWAY),
        "winner": tc[w] if w else None,
        "margin": abs(h - a),
        "overtimes": max(0, n_per - REG_PERIODS),
        "top_performers": top,
        "unsung_hero": unsung_candidates(impact, w),
        "runs": [_run_fact(tc, events, r, garbage, core) for r, core in shown],
        "turning_point": tp_fact,
        "clutch": clutch_fact,
        "lead_changes": leads["lead_changes"],
        "times_tied": leads["times_tied"],
        "largest_lead": {tc[s]: lead_fact(s) for s in SIDES},
        "comeback": ({"team": tc[cb["side"]], "deficit": cb["deficit"], "at": _when(tc, cb.get("at"))}
                     if cb and cb["deficit"] > 0 else None),
        "garbage_time": gt_fact,
        "notable_lines": notable,
        "incidents": incidents(events, stints, impact, garbage, names, tc),
        "validation": report.summary(),
        **wp_facts(events, wp_points, tc, names, w, pregame_spread),
    }
