"""Brief JSON -> view models for the templates.

Everything a template prints comes through here: writer text when present, otherwise a
fallback built only from the structured facts (never invented). Every section tolerates missing
keys, nulls and empty lists, so a schema-valid Brief always renders.
"""

from __future__ import annotations

import re

from . import charts, fmt
from .teams import identity

NUMBER_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
                "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen"]
QUARTER_NAMES = {"Q1": "first quarter", "Q2": "second quarter", "Q3": "third quarter",
                 "Q4": "fourth quarter", "OT": "overtime", "2OT": "second overtime",
                 "3OT": "third overtime", "4OT": "fourth overtime"}
SHORT_Q = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}


def g(d, *keys, default=None):
    """Safe nested get: g(brief, 'last_night', 'games') -> value or default."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list) and isinstance(k, int) and -len(cur) <= k < len(cur):
            cur = cur[k]
        else:
            return default
        if cur is None:
            return default
    return cur


def text_of(slot) -> str | None:
    """A writer text slot (string or {"text": ...}) -> non-empty string or None."""
    if isinstance(slot, dict):
        slot = slot.get("text")
    if isinstance(slot, str) and slot.strip() and slot.strip().lower() not in ("none", "null"):
        return slot.strip()
    return None


def count_word(n: int, lower: bool = False) -> str:
    w = NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)
    return w.lower() if lower else w


def plural(n: int, word: str, pl: str | None = None) -> str:
    return word if n == 1 else (pl or word + "s")


# ------------------------------------------------------------------ box-line parsing

_LINE_STAT = re.compile(r"(\d+)\s+(pts|reb|ast|stl|blk|tov|to)\b")
_SHOOT = re.compile(r"(\d+)-(\d+)\s+(FG|3P|FT)")
STAT_WORDS = {"pts": ("point", "points"), "reb": ("rebound", "rebounds"),
              "ast": ("assist", "assists"), "stl": ("steal", "steals"),
              "blk": ("block", "blocks")}


def parse_line(line: str | None) -> dict:
    """'35 pts (14-20 FG, 2-4 3P, 5-5 FT), 8 reb, 4 ast, 3 blk' -> {pts: 35, fg: (14, 20), ...}."""
    out: dict = {}
    if not line:
        return out
    for v, k in _LINE_STAT.findall(line):
        out[k] = int(v)
    for m, a, k in _SHOOT.findall(line):
        out[k.lower()] = (int(m), int(a))
    return out


def short_line(line: str | None) -> str:
    """Drop the shooting split: '35 pts, 8 reb, 4 ast, 3 blk'."""
    if not line:
        return ""
    return re.sub(r"\s*\([^)]*\)", "", line).strip()


def prose_line(line: str | None) -> str:
    """'35 points on 14-of-20 shooting, 8 rebounds, 4 assists and 3 blocks'."""
    p = parse_line(line)
    parts = []
    for k in ("pts", "reb", "ast", "stl", "blk"):
        if k in p:
            one, many = STAT_WORDS[k]
            s = f"{p[k]} {one if p[k] == 1 else many}"
            if k == "pts" and "fg" in p and p["fg"][1] >= 5:
                s += f" on {p['fg'][0]}\u2011of\u2011{p['fg'][1]} shooting"
            parts.append(s)
    if not parts:
        return short_line(line)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


FLAG_LABELS = {"triple_double": "Triple-double", "20_10": "20–10", "perfect_fg": "Perfect FG",
               "perfect_ft": "Perfect FT", "efficient_scoring": "Efficient scoring",
               "cold_shooting": "Cold shooting", "bench_scoring": "Bench scoring"}
NEG_FLAGS = {"cold_shooting"}


def flag_label(flag: str) -> tuple[str, bool]:
    """('40-point game', is_negative)."""
    if flag in FLAG_LABELS:
        return FLAG_LABELS[flag], flag in NEG_FLAGS
    m = re.fullmatch(r"(\d+)_point_game", flag)
    if m:
        return f"{m.group(1)}-point game", False
    m = re.fullmatch(r"(\d+)_(rebounds|assists|threes|stocks|turnovers|steals|blocks)", flag)
    if m:
        return f"{m.group(1)} {m.group(2)}", m.group(2) == "turnovers"
    return flag.replace("_", " ").capitalize(), False


# ------------------------------------------------------------------ teams & scores

def team_side(t: dict | None, *, score=None, winner: str | None = None) -> dict:
    t = t or {}
    ident = identity(t.get("team"))
    return {**ident, "score": score if score is not None else t.get("score"),
            "winner": bool(winner) and winner == ident["code"],
            "record": t.get("record"), "last10": t.get("last10"), "streak": t.get("streak"),
            "power_rank": t.get("power_rank"), "full_name": t.get("name") or ident["name"]}


def period_name(label: str | None) -> str:
    return QUARTER_NAMES.get(label or "", label or "")


def span_name(a: str | None, b: str | None) -> str:
    if not a:
        return ""
    if not b or a == b:
        return f"the {period_name(a)}"
    if a in SHORT_Q and b in SHORT_Q:
        return f"the {SHORT_Q[a]} and {SHORT_Q[b]} quarters"
    return f"the {period_name(a)} and {period_name(b)}"


def score_pair(score: dict | None, first: str, second: str) -> tuple[int, int] | None:
    if not isinstance(score, dict) or score.get(first) is None or score.get(second) is None:
        return None
    return int(score[first]), int(score[second])


def a_n(word) -> str:
    """Indefinite article for a word or number: 'an 18-point', 'a 17-2'."""
    w = str(word)
    if w[:1] == "8" or re.match(r"(11|18)(\D|$)", w) or w[:1].lower() in "aeio":
        return f"an {w}"
    return f"a {w}"


def run_effect(team: str, opp: str, frm: dict | None, to: dict | None) -> str:
    """Plain-English effect of a run on the score, from the run team's point of view."""
    b = score_pair(frm, team, opp)
    a = score_pair(to, team, opp)
    if not b or not a:
        return ""
    (tb, ob), (ta, oa) = b, a
    d = fmt.DASH
    if tb < ob and ta > oa:
        return f"turned {a_n(f'{ob}{d}{tb}')} deficit into {a_n(f'{ta}{d}{oa}')} lead"
    if tb < ob and ta == oa:
        return f"erased {a_n(f'{ob - tb}-point')} deficit to tie it at {ta}"
    if tb < ob:
        return f"cut {a_n(f'{ob - tb}-point')} deficit to {oa - ta}"
    if tb == ob and ta > oa:
        return f"broke {a_n(f'{tb}{d}{ob}')} tie and opened {a_n(f'{ta}{d}{oa}')} lead"
    if tb > ob:
        return f"stretched {a_n(f'{tb - ob}-point')} lead to {ta - oa}"
    return ""


def turning_view(facts: dict | None) -> dict | None:
    tp = g(facts, "turning_point")
    if not isinstance(tp, dict) or not tp.get("run") or not tp.get("team"):
        return None
    team = tp["team"]
    home, away = g(facts, "home", "team"), g(facts, "away", "team")
    opp = away if team == home else home
    frm, to = tp.get("from") or {}, tp.get("to") or {}
    effect = run_effect(team, opp, frm.get("score"), to.get("score")) if opp else ""
    tail = ""
    b = score_pair(frm.get("score"), team, opp) if opp else None
    if tp.get("reason") == "created_winning_margin" and b and b[0] <= b[1]:
        tail = "and they never trailed again" if b[0] < b[1] else "and they never looked back"

    def sc(e):
        p = score_pair(e.get("score"), team, opp) if opp else None
        return f"{p[0]}{fmt.DASH}{p[1]}" if p else ""

    return {"team": identity(team), "opp": identity(opp), "run": fmt.run(tp["run"]),
            "from_label": frm.get("label"), "from_clock": frm.get("clock"),
            "to_label": to.get("label"), "to_clock": to.get("clock"),
            "from_score": sc(frm), "to_score": sc(to), "minutes": tp.get("minutes"),
            "effect": effect, "tail": tail, "span": span_name(frm.get("label"), to.get("label")),
            "reason": tp.get("reason"), "method": tp.get("method"),
            "wp_swing": tp.get("wp_swing")}


def hero_view(c: dict | None) -> dict | None:
    if not isinstance(c, dict) or not c.get("name"):
        return None
    return {"name": c["name"], "team": identity(c.get("team")), "line": c.get("line") or "",
            "short_line": short_line(c.get("line")), "bench": bool(c.get("bench")),
            "winner": bool(c.get("winner")), "minutes": c.get("minutes"),
            "plus_minus": c.get("plus_minus"), "on_off": c.get("on_off"),
            "ts_pct": c.get("ts_pct"), "stocks": c.get("stocks"), "score": c.get("score"),
            "why": [w for w in (c.get("why") or []) if isinstance(w, str) and w.strip()]}


# A candidate is only *featured* on the compact home card when he did something you can see in
# the box score: single-game on/off is noisy, and "1 pt on 0-2 shooting in 13 minutes" should not
# headline a card. The game page still lists every candidate.
HERO_MIN_MINUTES = 15
HERO_MIN_PRODUCTION = 10      # pts + reb + ast + stocks
HERO_MIN_PLUS_MINUS = 8


def _number(x) -> float | None:
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else float(x)


def hero_production(c: dict) -> int:
    """pts + reb + ast + stocks from a candidate (stocks field preferred, else stl + blk)."""
    pl = parse_line(c.get("line"))
    stocks = _number(c.get("stocks"))
    if stocks is None:
        stocks = pl.get("stl", 0) + pl.get("blk", 0)
    return int(pl.get("pts", 0) + pl.get("reb", 0) + pl.get("ast", 0) + stocks)


def hero_qualifies(c: dict | None) -> bool:
    """Minimum real contribution to feature an unsung-hero candidate on the home card."""
    if not isinstance(c, dict) or not c.get("name"):
        return False
    mins = _number(c.get("minutes"))
    if mins is None or mins < HERO_MIN_MINUTES:
        return False
    pm = _number(c.get("plus_minus"))
    return hero_production(c) >= HERO_MIN_PRODUCTION or (pm is not None
                                                          and pm >= HERO_MIN_PLUS_MINUS)


def featured_player(facts: dict | None, winner: str | None) -> dict | None:
    """The one player the compact card names: the first unsung-hero candidate who clears the
    contribution bar, otherwise the winning side's top performer."""
    for c in g(facts, "unsung_hero", default=[]) or []:
        if hero_qualifies(c):
            return {**hero_view(c), "kind": "unsung", "label": "Unsung hero"}
    tops = [p for p in g(facts, "top_performers", default=[]) or []
            if isinstance(p, dict) and p.get("name")]
    if not tops:
        return None
    mine = [p for p in tops if p.get("team") == winner] or tops
    best = max(mine, key=lambda p: _number(p.get("game_score")) or 0)
    return {**hero_view(best), "kind": "top", "label": "Top performer"}


def player_beat(h: dict | None) -> str:
    """'2 pts, 8 reb, 8 ast, 2 stl · +2 in 29 min'."""
    if not h:
        return ""
    parts = [h["short_line"]] if h.get("short_line") else []
    pm, mins = _number(h.get("plus_minus")), _number(h.get("minutes"))
    if pm is not None and mins is not None:
        parts.append(f"{fmt.signed_int(pm)} in {mins:.0f} min")
    elif pm is not None:
        parts.append(fmt.signed_int(pm))
    return " · ".join(parts)


def turning_beat_text(t: dict | None) -> str:
    """'LAC 17–5 run · Q3 4:41 – Q4 11:02'."""
    if not t:
        return ""
    s = f"{t['team']['code']} {t['run']} run"
    when = " ".join(x for x in (t.get("from_label"), t.get("from_clock")) if x)
    if when and t.get("to_clock"):
        to = t["to_clock"] if t.get("to_label") == t.get("from_label") else \
            " ".join(x for x in (t.get("to_label"), t["to_clock"]) if x)
        when = f"{when} {fmt.DASH} {to}"
    return f"{s} · {when}" if when else s


KEY_PLAY = {"go_ahead": "Go-ahead", "tying": "Tying", "steal": "Steal", "block": "Block"}


def clutch_view(facts: dict | None, winner: str | None) -> dict | None:
    c = g(facts, "clutch")
    if not isinstance(c, dict):
        return None
    had = bool(c.get("had_clutch"))
    plays = []
    for kp in c.get("key_plays") or []:
        if not isinstance(kp, dict):
            continue
        home, away = g(facts, "home", "team"), g(facts, "away", "team")
        sc = score_pair(kp.get("score"), away, home) if home and away else None
        plays.append({"kind": KEY_PLAY.get(kp.get("kind"), (kp.get("kind") or "").title()),
                      "player": kp.get("player") or "", "team": identity(kp.get("team")),
                      "label": kp.get("label") or "", "clock": kp.get("clock") or "",
                      "score": f"{away} {sc[0]}{fmt.DASH}{sc[1]} {home}" if sc else "",
                      "decisive": bool(kp.get("decisive"))})
    players = [p for p in (c.get("players") or []) if isinstance(p, dict)]
    top = max(players, key=lambda p: (p.get("pts") or 0), default=None)
    decisive = next((p for p in plays if p["decisive"]), None)
    pts = c.get("points") or {}
    return {"had": had, "seconds": c.get("seconds") or 0, "points": pts,
            "players": players, "plays": plays, "top": top, "decisive": decisive,
            "points_text": " · ".join(f"{k} {v}" for k, v in pts.items()) if pts else ""}


def notable_view(facts: dict | None) -> list[dict]:
    out = []
    for n in g(facts, "notable_lines", default=[]) or []:
        if not isinstance(n, dict) or not n.get("name"):
            continue
        flags = [flag_label(f) for f in (n.get("flags") or [])]
        out.append({"name": n["name"], "team": identity(n.get("team")),
                    "line": n.get("line") or "",
                    "flags": [{"label": lab, "neg": neg} for lab, neg in flags]})
    return out


def incidents_view(facts: dict | None) -> list[str]:
    inc = g(facts, "incidents") or {}
    out = []
    for e in inc.get("ejections") or []:
        if isinstance(e, dict) and e.get("name"):
            when = " ".join(x for x in ((e.get("at") or {}).get("label"),
                                        (e.get("at") or {}).get("clock")) if x)
            out.append(f"{e['name']} ({e.get('team', '')}) was ejected" + (f", {when}" if when
                                                                           else ""))
    for e in inc.get("foul_outs") or []:
        if isinstance(e, dict) and e.get("name"):
            out.append(f"{e['name']} ({e.get('team', '')}) fouled out")
    for e in inc.get("possible_injuries") or []:
        if isinstance(e, dict) and e.get("name"):
            lof = e.get("last_on_floor") or {}
            when = " ".join(x for x in (lof.get("label"), lof.get("clock")) if x)
            note = e.get("note") or "left the game and did not return"
            out.append(f"{e['name']} ({e.get('team', '')}) {note}"
                       + (f" (last on the floor {when})" if when else ""))
    for e in inc.get("injury_mentions") or inc.get("injuries") or []:
        if isinstance(e, dict) and e.get("name") and e.get("note"):
            out.append(f"{e['name']}: {e['note']}")
    return out


# ------------------------------------------------------------------ card fallbacks

def _verb(margin: int, overtimes: int) -> str:
    if overtimes:
        return "outlast"
    if margin <= 3:
        return "edge"
    if margin >= 25:
        return "rout"
    if margin >= 15:
        return "roll past"
    return "beat"


def card_headline(card: dict, facts: dict | None, win: dict, lose: dict) -> str:
    res = card.get("result") or {}
    margin = int(res.get("margin") or abs((win.get("score") or 0) - (lose.get("score") or 0)))
    ots = int(res.get("overtimes") or g(facts, "overtimes", default=0) or 0)
    wn, ln = win["nickname"], lose["nickname"]
    cb = g(facts, "comeback") or {}
    if cb.get("team") == win["code"] and (cb.get("deficit") or 0) >= 15:
        return f"{wn} rally from {cb['deficit']} down to beat the {ln}"
    head = f"{wn} {_verb(margin, ots)} {ln}"
    if ots:
        head += " in overtime" if ots == 1 else f" in {count_word(ots, True)} overtimes"
    star = None
    for n in g(facts, "notable_lines", default=[]) or []:
        if n.get("team") == win["code"] and any(f in (n.get("flags") or [])
                                                for f in ("triple_double", "50_point_game",
                                                          "40_point_game")):
            star = n
            break
    if star and star.get("name"):
        fl = star.get("flags") or []
        what = "triple-double" if "triple_double" in fl else str(parse_line(star.get("line"))
                                                                    .get("pts", ""))
        if what:
            return f"{head} behind {fmt.possessive(fmt.last_name(star['name']))} {what}"
    tops = [p for p in g(facts, "top_performers", default=[]) or [] if p.get("team") == win["code"]]
    if tops:
        pts = parse_line(tops[0].get("line")).get("pts")
        if pts and pts >= 25:
            return f"{head} behind {fmt.possessive(fmt.last_name(tops[0]['name']))} {pts}"
    return head


def card_story(card: dict, facts: dict | None, win: dict, lose: dict,
               turning: dict | None) -> str:
    """2-3 sentences from the facts: the turning point, then the best lines."""
    sents = []
    if turning and turning["effect"]:
        team = turning["team"]
        lead = f"{a_n(turning['run'])} {team['nickname']} run in {turning['span']}"
        lead = lead[:1].upper() + lead[1:]
        clock = ""
        if turning["from_clock"] and turning["to_clock"]:
            clock = f" ({turning['from_label']} {turning['from_clock']} to " \
                    f"{turning['to_label']} {turning['to_clock']})" \
                if turning["from_label"] != turning["to_label"] else \
                f" ({turning['from_clock']} to {turning['to_clock']})"
        s = f"{lead}{clock} {turning['effect']}"
        if turning["tail"] and team["code"] == win["code"]:
            s += f", {turning['tail']}"
        sents.append(s + ".")
    tops = g(facts, "top_performers", default=[]) or []
    wt = next((p for p in tops if p.get("team") == win["code"]), None)
    lt = next((p for p in tops if p.get("team") == lose["code"]), None)
    if wt:
        sents.append(f"{wt['name']} had {prose_line(wt.get('line'))} for the {win['nickname']}.")
    if lt:
        p = parse_line(lt.get("line")).get("pts")
        if p is not None:
            sents.append(f"{lt['name']} led the {lose['nickname']} with {p} "
                         f"{'point' if p == 1 else 'points'}.")
    if not sents:
        if win.get("score") is not None and lose.get("score") is not None:
            sents.append(f"The {win['nickname']} beat the {lose['nickname']} "
                         f"{win['score']}{fmt.DASH}{lose['score']}.")
    return " ".join(sents)


# ------------------------------------------------------------------ last night

def pick_view(card: dict) -> dict | None:
    p = card.get("prediction")
    if not isinstance(p, dict) or not p.get("pick"):
        return None
    return {"team": identity(p["pick"]), "prob": p.get("pick_prob"),
            "confidence": p.get("confidence"), "correct": card.get("correct"),
            "margin": fmt.margin_text(p.get("home_margin"), g(card, "home", "team"),
                                      g(card, "away", "team")),
            "total": p.get("total")}


def market_view(m: dict | None, home: str | None, away: str | None) -> dict | None:
    if not isinstance(m, dict):
        return None
    hp = m.get("home_prob")
    fav = None
    if hp is not None:
        fav = home if hp >= 0.5 else away
    elif m.get("home_spread") is not None:
        fav = home if m["home_spread"] < 0 else away
    fav = m.get("favorite") or fav
    fav_prob = (hp if fav == home else (1 - hp)) if (hp is not None and fav) else None
    covered = None
    if m.get("home_covered") is not None and m.get("home_spread") not in (None, 0):
        fav_is_home = m["home_spread"] < 0
        covered = m["home_covered"] if fav_is_home else not m["home_covered"]
    return {"spread": fmt.spread_text(None, m.get("home_spread"), home, away),
            "total": m.get("total"), "home_prob": hp, "fav": fav, "fav_prob": fav_prob,
            "home_ml": m.get("home_ml"), "away_ml": m.get("away_ml"),
            "provider": m.get("provider") or "", "kind": m.get("kind") or "",
            "favorite_won": m.get("favorite_won"), "fav_covered": covered,
            "went_over": m.get("went_over"), "captured_at": m.get("captured_at")}


def card_view(card: dict) -> dict:
    facts = card.get("facts") if card.get("facts_available") is not False else None
    facts = facts if isinstance(facts, dict) else None
    res = card.get("result") if isinstance(card.get("result"), dict) else {}
    winner = res.get("winner") or g(facts, "winner")
    home = team_side(card.get("home"), score=res.get("home_score"), winner=winner)
    away = team_side(card.get("away"), score=res.get("away_score"), winner=winner)
    final = bool(card.get("final")) or card.get("status") == "final"
    ots = int(res.get("overtimes") or g(facts, "overtimes", default=0) or 0)
    status = "Final" if final else (card.get("status") or "").replace("_", " ").title()
    if final and ots:
        status = "Final/OT" if ots == 1 else f"Final/{ots}OT"
    win, lose = (home, away) if winner == home["code"] else (away, home)
    turning = turning_view(facts)
    text = card.get("text") if isinstance(card.get("text"), dict) else {}
    head, story = text_of(text.get("headline")), text_of(text.get("story"))
    have_result = final and winner and home["score"] is not None and away["score"] is not None
    if not head and have_result:
        head = card_headline(card, facts, win, lose)
    if not story and have_result:
        story = card_story(card, facts, win, lose, turning)
    heroes = [h for h in (hero_view(c) for c in g(facts, "unsung_hero", default=[]) or []) if h]
    stype = card.get("season_type")
    featured = featured_player(facts, winner) if facts else None
    vm = {
        "game_id": card.get("game_id"), "final": final, "status": status,
        "season_type": stype if stype and stype != "Regular Season" else None,
        "home": home, "away": away, "winner": winner, "overtimes": ots,
        "margin": res.get("margin"), "total": res.get("total"),
        "pick": pick_view(card),
        "market": market_view(card.get("market"), home["code"], away["code"]),
        "headline": head, "headline_auto": not text_of(text.get("headline")),
        "story": story, "story_auto": not text_of(text.get("story")),
        "turning": turning, "hero": heroes[0] if heroes else None, "heroes": heroes,
        "clutch": clutch_view(facts, winner), "notable": notable_view(facts),
        "incidents": incidents_view(facts), "facts": facts,
        "has_facts": facts is not None, "arena": g(facts, "arena"),
        "featured": featured, "featured_beat": player_beat(featured),
        "turning_beat": turning_beat_text(turning),
    }
    vm["story_lead"], vm["story_rest"] = story_lead(story)
    vm["interest"], vm["tags"], vm["blowout"] = interest(vm)
    return vm


# ------------------------------------------------------------------ story lead

_SENT_END = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“(]?[A-Z0-9])")
_ABBREV = re.compile(r"(?:\b(?:Jr|Sr|St|vs|No|Mt|Dr|Mr)|\b[A-Z])\.$")


def split_sentences(text: str | None) -> list[str]:
    """Plain sentence split that keeps 'Jr.', 'vs.', initials and decimals ('35.6') intact."""
    if not text:
        return []
    out: list[str] = []
    for piece in _SENT_END.split(text.strip()):
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [p for p in out if p]


STORY_SHORT = 100        # a first sentence shorter than this gets company
STORY_LEAD_MAX = 260     # ...as long as the two together stay under this


def story_lead(text: str | None) -> tuple[str, str]:
    """(lead, rest): the first sentence, plus the second when the first is short. The home card
    shows the lead; the rest sits behind 'Read more' (whole sentences only, never a cut)."""
    sents = split_sentences(text)
    if len(sents) <= 1:
        return (text or "").strip(), ""
    n = 1
    if len(sents) > 2 and len(sents[0]) < STORY_SHORT \
            and len(sents[0]) + len(sents[1]) <= STORY_LEAD_MAX:
        n = 2
    return " ".join(sents[:n]), " ".join(sents[n:])


# ------------------------------------------------------------------ card ordering

BLOWOUT_MARGIN = 20
STAR_FLAGS = (("50_point_game", 5.0, "50-point night"), ("40_point_game", 3.5, "40-point night"),
              ("triple_double", 3.0, "Triple-double"))


def interest(c: dict) -> tuple[float, list[str], bool]:
    """(score, tags, is_blowout) for ordering Last Night by how much a reader wants each game.

    Uses the win-probability facts when the analytics provide them (`excitement` = sum of |ΔWP|,
    `comeback_wp.min_wp` = the winner's lowest win probability); otherwise margin, overtime and
    the score-based comeback. Upsets (market favourite lost, or our pick missed) and big
    individual nights add to either. Blowouts (20+ points without a 40-point night or a
    triple-double) sort last and render as compact rows.
    """
    f = c.get("facts") or {}
    if not c.get("final") or not c.get("winner"):
        return -100.0, [], False
    tags: list[str] = []
    s = 0.0
    margin = _number(c.get("margin"))
    ots = int(c.get("overtimes") or 0)
    exc = _number(f.get("excitement"))
    cwp = f.get("comeback_wp") if isinstance(f.get("comeback_wp"), dict) else {}
    min_wp = _number(cwp.get("min_wp")) if cwp.get("team") in (None, c["winner"]) else None
    cb = f.get("comeback") if isinstance(f.get("comeback"), dict) else {}
    deficit = _number(cb.get("deficit")) if cb.get("team") == c["winner"] else None
    if exc is not None:
        s += 2 * min(exc, 5.0)
    elif margin is not None:
        s += 6 if margin <= 3 else 4 if margin <= 6 else 2 if margin <= 10 else 0
    if ots:
        s += min(3 * ots, 5)
        tags.append("Overtime" if ots == 1 else f"{ots} OT")
    elif margin is not None and margin <= 3:
        tags.append("Thriller")
    if min_wp is not None:
        if min_wp < 0.25:
            s += (0.25 - min_wp) * 16
        if min_wp <= 0.15:
            tags.append("Comeback")
    elif deficit:
        s += 4 if deficit >= 15 else 2.5 if deficit >= 10 else 1 if deficit >= 6 else 0
        if deficit >= 10:
            tags.append("Comeback")
    mk = c.get("market") or {}
    if mk.get("favorite_won") is False:
        s += 2.5
        tags.append("Upset")
    if (c.get("pick") or {}).get("correct") is False:
        s += 1.5
    flags = {fl for n in f.get("notable_lines") or [] if isinstance(n, dict)
             for fl in (n.get("flags") or [])}
    star = next(((w, lab) for k, w, lab in STAR_FLAGS if k in flags), None)
    if star:
        s += star[0]
        tags.append(star[1])
    elif "30_point_game" in flags:
        s += 0.5
    blowout = margin is not None and margin >= BLOWOUT_MARGIN and not star and not ots
    if blowout:
        s -= 6
    return round(s, 2), tags[:2], blowout


FEATURED_MAX = 4


def card_tiers(cards: list[dict]) -> dict:
    """featured (full cards), also (compact rows), blowouts (compact rows, 'Decided early').
    Cards are already in interest order. With five or fewer competitive games all get a card."""
    close = [c for c in cards if not c["blowout"]]
    n = len(close) if len(close) <= FEATURED_MAX + 1 else FEATURED_MAX
    return {"featured": close[:n], "also": close[n:],
            "blowouts": [c for c in cards if c["blowout"]]}


def order_cards(cards: list[dict]) -> list[dict]:
    """Most interesting first; ties keep tip-off order."""
    return [c for _, c in sorted(enumerate(cards), key=lambda ic: (-ic[1]["interest"], ic[0]))]


# ------------------------------------------------------------------ tonight

FACTOR_KINDS = {"back_to_back": "Back-to-back", "four_in_six": "4 in 6 nights",
                "three_in_four": "3 in 4 nights"}


def factor_chip(f: dict) -> dict | None:
    kind, team, v = f.get("kind"), f.get("team"), f.get("value")
    t = identity(team) if team else None
    if kind in FACTOR_KINDS:
        label = FACTOR_KINDS[kind]
    elif kind == "long_travel" and isinstance(v, (int, float)):
        label = f"{v:,.0f} km travel"
    elif kind == "time_zone" and isinstance(v, (int, float)):
        label = f"{abs(v):.0f}-hour time-zone shift"
    elif kind == "rest_edge" and isinstance(v, (int, float)):
        label = f"+{v:.0f} {'day' if v == 1 else 'days'} rest"
    elif kind == "neutral_site":
        label, t = "Neutral site", None
    elif kind == "injury" and isinstance(v, dict):
        st = v.get("status") or "Out"
        label = f"{v.get('player', '')} {st.lower()}"
        mpg = v.get("mpg")
        return {"team": t, "label": label, "kind": "injury",
                "sub": f"{mpg:.1f} mpg" if isinstance(mpg, (int, float)) else "",
                "detail": f.get("detail") or label, "status": st}
    elif kind in ("form", "model_vs_market"):
        return None
    else:
        label = f.get("detail") or (kind or "").replace("_", " ")
    return {"team": t, "label": label, "kind": kind or "", "sub": "",
            "detail": f.get("detail") or label}


SCHED_SHORT = {"back_to_back": "on a back-to-back", "four_in_six": "on 4 in 6 nights",
               "three_in_four": "on 3 in 4 nights"}


def key_factors(game: dict) -> list[str]:
    """One or two short phrases for the compact Tonight row: the side missing the most minutes,
    then the strongest schedule factor."""
    out = []
    by_team: dict[str, list[dict]] = {}
    for f in game.get("factors") or []:
        if isinstance(f, dict) and f.get("kind") == "injury" and isinstance(f.get("value"), dict) \
                and f.get("team") and f["value"].get("player"):
            by_team.setdefault(f["team"], []).append(f["value"])
    if by_team:
        team, rows = max(by_team.items(),
                         key=lambda kv: sum(_number(v.get("mpg")) or 0 for v in kv[1]))
        gone = [fmt.last_name(v["player"]) for v in rows
                if (v.get("status") or "Out").lower() in ("out", "doubtful")]
        q = [fmt.last_name(v["player"]) for v in rows
             if (v.get("status") or "").lower() == "questionable"]
        if gone:
            s = f"{team} without {', '.join(gone)}"
            if q:
                s += f"; {', '.join(q)} questionable"
            out.append(s)
        elif q:
            out.append(f"{team}: {', '.join(q)} questionable")
    sched: dict[str, list[str]] = {}
    for f in game.get("factors") or []:
        if isinstance(f, dict) and f.get("kind") in SCHED_SHORT and f.get("team"):
            sched.setdefault(f["kind"], []).append(f["team"])
    for kind in ("back_to_back", "three_in_four", "four_in_six"):
        teams = sched.get(kind)
        if teams:
            out.append(f"Both {SCHED_SHORT[kind]}" if len(set(teams)) > 1
                       else f"{teams[0]} {SCHED_SHORT[kind]}")
            break
    else:
        for f in game.get("factors") or []:
            if isinstance(f, dict) and f.get("kind") == "rest_edge" and f.get("team") \
                    and _number(f.get("value")):
                v = _number(f["value"])
                out.append(f"{f['team']} +{v:.0f} {'day' if v == 1 else 'days'} rest")
                break
    return out[:2]


def tonight_view(game: dict) -> dict:
    home = team_side(game.get("home"))
    away = team_side(game.get("away"))
    p = game.get("prediction") if isinstance(game.get("prediction"), dict) else None
    m = market_view(game.get("market"), home["code"], away["code"])
    pred = None
    if p and p.get("home_win_prob") is not None:
        hp = float(p["home_win_prob"])
        pred = {"home_prob": hp, "away_prob": 1 - hp,
                "pick": identity(p.get("pick") or (home["code"] if hp >= 0.5 else away["code"])),
                "pick_prob": p.get("pick_prob") if p.get("pick_prob") is not None
                else max(hp, 1 - hp),
                "confidence": p.get("confidence"),
                "margin": fmt.margin_text(p.get("home_margin"), home["code"], away["code"]),
                "total": p.get("total")}
    edge = game.get("edge") if isinstance(game.get("edge"), dict) else {}
    edge_v = None
    if edge.get("win_prob") is not None and m and m["home_prob"] is not None:
        wp = float(edge["win_prob"])
        side = home if wp >= 0 else away
        mg = edge.get("margin")
        tot = edge.get("total")
        edge_v = {"team": side, "win_prob": abs(wp),
                  "margin": abs(mg) if isinstance(mg, (int, float)) else None,
                  "total": tot, "big": abs(wp) >= 0.06}
    chips = [c for c in (factor_chip(f) for f in game.get("factors") or []
                         if isinstance(f, dict)) if c]
    sched = [c for c in chips if c["kind"] != "injury"]
    inj_chips = []
    for side in (away, home):
        mine = [c for c in chips if c["kind"] == "injury" and c["team"]
                and c["team"]["code"] == side["code"]]
        if mine:
            names = [c["label"].rsplit(" ", 1)[0] + ("" if c["status"].lower() == "out"
                                                     else f" ({c['status'].lower()})")
                     for c in mine]
            inj_chips.append({"team": identity(side["code"]), "label": ", ".join(names),
                              "detail": "; ".join(c["detail"] for c in mine)})
    injuries = g(game, "injuries") or {}
    up = game.get("upset") if isinstance(game.get("upset"), dict) else None
    upset = None
    if game.get("upset_watch") or (up and up.get("flag")):
        if up:
            upset = {"favorite": identity(up.get("favorite")),
                     "underdog": identity(up.get("underdog")),
                     "model_prob": up.get("model_prob"), "market_prob": up.get("market_prob"),
                     "edge": up.get("edge"), "picks_dog": bool(up.get("model_picks_underdog"))}
        else:
            upset = {}
    text = game.get("text") if isinstance(game.get("text"), dict) else {}
    preview = text_of(text.get("preview"))
    return {"game_id": game.get("game_id"), "home": home, "away": away,
            "tipoff": fmt.tipoff(game.get("tipoff_utc")), "status": game.get("status"),
            "season_type": game.get("season_type") if game.get("season_type") not in
            (None, "Regular Season") else None,
            "pred": pred, "market": m, "edge": edge_v, "sched": sched, "inj_chips": inj_chips,
            "injuries": {"home": [i for i in injuries.get("home") or [] if isinstance(i, dict)],
                         "away": [i for i in injuries.get("away") or [] if isinstance(i, dict)]},
            "upset": upset,
            "key": key_factors(game),
            "preview": preview,
            "preview_auto": preview is None}


# ------------------------------------------------------------------ pulses

REASON_SKIP = {"net_trend", "streak", "last10_record"}


def polish(s: str | None) -> str:
    """Typographic minus for negative numbers in machine-written detail strings."""
    return re.sub(r"(?<![\w-])-(?=\d)", fmt.MINUS, s or "")


def team_trend_view(t: dict, direction: str) -> dict:
    ident = identity(t.get("team"))
    net = t.get("net") or {}
    tr = t.get("trend") or {}
    reasons = [r for r in t.get("reasons") or [] if isinstance(r, dict)]
    chips = [polish(r.get("detail")) for r in reasons if r.get("kind") not in REASON_SKIP
             and r.get("detail")]
    blurb = text_of(t.get("text"))
    if not blurb:
        bits = []
        if tr.get("delta") is not None and tr.get("baseline") is not None:
            bits.append(f"{abs(tr['delta']):.1f} points per 100 possessions "
                        f"{'better' if tr['delta'] >= 0 else 'worse'} than their "
                        f"{fmt.signed(tr['baseline'])} baseline")
        sos = t.get("sos_last10") or {}
        has_sched = any(r.get("kind", "").startswith("schedule_") for r in reasons)
        if sos.get("rank") and sos.get("of") and not has_sched and (
                sos["rank"] <= 8 or sos["rank"] > sos["of"] - 8):
            r = sos["rank"]
            hard = r <= sos["of"] / 2
            bits.append(f"done against the {ordinal(r if hard else sos['of'] + 1 - r)}-"
                        f"{'hardest' if hard else 'easiest'} schedule in the league")
        blurb = "; ".join(bits)
        blurb = blurb[:1].upper() + blurb[1:] + "." if blurb else ""
    abs_ = [a for a in t.get("key_absences") or [] if isinstance(a, dict) and a.get("player")]
    return {"team": ident, "record": t.get("record"), "last10": t.get("last10"),
            "z": _number(tr.get("z")), "why": team_why(t, reasons),
            "streak": t.get("streak"), "power_rank": t.get("power_rank"),
            "net": net, "off": t.get("off") or {}, "def": t.get("def") or {}, "trend": tr,
            "direction": direction, "chips": chips, "blurb": blurb,
            "absences": abs_,
            "spark": charts.dumbbell(net.get("season_adj"), net.get("last10_adj"), -20, 20,
                                     label=f"Adjusted net rating: season "
                                           f"{fmt.signed(net.get('season_adj'))}, last 10 "
                                           f"{fmt.signed(net.get('last10_adj'))}")}


def team_why(t: dict, reasons: list[dict]) -> str:
    """A one-line reason for the compact row: the offense/defense split that moved most."""
    off, de = t.get("off") or {}, t.get("def") or {}
    cands = []
    for label, d, good_low in (("Offense", off, False), ("Defense", de, True)):
        a, b = _number(d.get("last10")), _number(d.get("season"))
        if a is not None and b is not None:
            cands.append((abs(a - b), label, a, b))
    if cands:
        _, label, a, b = max(cands)
        return f"{label} {fmt.num(a)} last 10 (season {fmt.num(b)})"
    r = next((r for r in reasons if r.get("kind") not in REASON_SKIP and r.get("detail")), None)
    return polish(r["detail"]) if r else ""


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def player_trend_view(p: dict, hot: bool) -> dict:
    rec, sea = p.get("recent") or {}, p.get("season") or {}

    def delta(k, scale=1.0):
        a, b = rec.get(k), sea.get(k)
        if a is None or b is None:
            return None
        return (a - b) * scale

    lo = 0
    vals = [v for v in (p.get("impact_baseline"), p.get("impact_recent")) if v is not None]
    hi = max(30, *(v + 2 for v in vals)) if vals else 30
    lo = min(0, *(v - 2 for v in vals)) if vals else 0
    return {"player": p.get("player") or "", "team": identity(p.get("team")),
            "window": p.get("window") or "", "z": p.get("z"), "hot": hot,
            "impact_recent": p.get("impact_recent"), "impact_baseline": p.get("impact_baseline"),
            "recent": rec, "season": sea,
            "d_pts": delta("pts"), "d_ts": delta("ts_pct", 100), "d_min": delta("min"),
            "blurb": text_of(p.get("text")),
            "spark": charts.dumbbell(p.get("impact_baseline"), p.get("impact_recent"), lo, hi,
                                     label=f"Impact: baseline {fmt.num(p.get('impact_baseline'))}"
                                           f", {p.get('window') or 'recent'} "
                                           f"{fmt.num(p.get('impact_recent'))}")}


ROLE_LABELS = {"minutes_up": "Minutes up", "minutes_down": "Minutes down",
               "usage_up": "Usage up", "usage_down": "Usage down",
               "new_in_rotation": "New in rotation", "starter_minutes": "Starter minutes",
               "lost_starter_minutes": "Lost starter minutes", "new_team": "New team",
               "new_arrival": "New arrival", "out_of_rotation": "Out of rotation"}
ROLE_NEG = {"minutes_down", "usage_down", "lost_starter_minutes", "out_of_rotation"}


def role_view(r: dict) -> dict:
    kinds = [k for k in r.get("kinds") or [] if isinstance(k, str)]
    details = [re.sub(r"\s*\((?:proxy|likely)[^)]*\)", "", d).split(":")[0].strip()
               for d in r.get("details") or [] if isinstance(d, str)]
    return {"player": r.get("player") or "", "team": identity(r.get("team")),
            "kinds": [{"label": ROLE_LABELS.get(k, k.replace("_", " ").capitalize()),
                       "neg": k in ROLE_NEG} for k in kinds],
            "detail": polish(details[0]) if details else ""}


LEADER_CATS = [("impact", "Impact", 1), ("pts", "Points", 1), ("reb", "Rebounds", 1),
               ("ast", "Assists", 1), ("stocks", "Stocks", 1), ("fg3m", "3PM", 1),
               ("ts_pct", "TS%", None)]


def leaders_view(ld: dict | None) -> list[dict]:
    out = []
    for key, label, nd in LEADER_CATS:
        rows = [r for r in (ld or {}).get(key) or [] if isinstance(r, dict)]
        if not rows:
            continue
        out.append({"key": key, "label": label, "rows": [
            {"player": r.get("player") or "", "team": identity(r.get("team")),
             "value": fmt.ts(r.get("value")) if nd is None else fmt.num(r.get("value"), nd),
             "games": r.get("games"), "line": (r.get("line") or "").replace("/", " / "),
             "min": r.get("min")} for r in rows]})
    return out


# ------------------------------------------------------------------ injuries

STATUS_CLASS = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
                "probable": "probable", "healthy": "healthy", "available": "healthy"}


def status_class(s: str | None) -> str:
    return STATUS_CLASS.get((s or "").lower(), "unknown")


def injury_row(e: dict) -> dict:
    return {"player": e.get("player") or "", "team": identity(e.get("team")),
            "status": e.get("status") or e.get("to_status") or "",
            "from_status": e.get("from_status"), "to_status": e.get("to_status"),
            "reason": fmt.injury_reason(e.get("reason")), "mpg": e.get("mpg"),
            "impact": e.get("impact"), "opponent": e.get("opponent"),
            "tonight": bool(e.get("playing_tonight")), "game_id": e.get("game_id"),
            "last_played": e.get("last_played"), "effect": e.get("model_effect_margin"),
            "direction": e.get("direction")}


IMPACT_MIN = 10          # Courtside Impact (Game Score units) for an "impact player"
IMPACT_MIN_MPG = 24      # fallback when impact is unknown


def is_impact(e: dict) -> bool:
    imp, mpg = _number(e.get("impact")), _number(e.get("mpg"))
    return (imp is not None and imp >= IMPACT_MIN) or (imp is None and mpg is not None
                                                       and mpg >= IMPACT_MIN_MPG)


def injury_split(inj: dict) -> dict:
    """What Injury Watch shows before 'See all': up to five impact players out tonight and the
    three biggest changes since yesterday (by impact)."""
    outs = inj["out_tonight"]
    lead = [e for e in outs if is_impact(e)][:5] or outs[:3]
    rest = [e for e in outs if e not in lead]
    changes = inj["new"] + inj["returns"] + inj["status"]
    top_changes = sorted([e for e in changes if is_impact(e)],
                         key=lambda e: (not e["tonight"], -(_number(e.get("impact")) or 0)))[:3]
    return {"lead": lead, "rest": rest, "top_changes": top_changes,
            "n_changes": len(changes)}


# ------------------------------------------------------------------ whole brief

def masthead(brief: dict, cards: list[dict], tonight: list[dict]) -> dict:
    head = text_of(brief.get("headline"))
    auto = head is None
    ln = g(brief, "last_night", "summary") or {}
    if auto:
        finished = [c for c in cards if c["headline"] and c["final"]]
        if finished:
            best = max(finished, key=lambda c: c.get("interest") or 0)
            head = best["headline"]
        elif tonight:
            n = len(tonight)
            head = f"{count_word(n)} {plural(n, 'game')} on tonight’s slate"
        else:
            head = "A quiet day around the league"
    # the one-line "what you need to know"
    parts = []
    n = len(cards)
    if n:
        s = f"{count_word(n)} {plural(n, 'game')} last night"
        if ln.get("picks_scored"):
            c, sc = ln.get("picks_correct") or 0, ln["picks_scored"]
            s += f"; our picks went {c}\u2060{fmt.DASH}\u2060{sc - c}"
        parts.append(s)
    else:
        parts.append("No games last night")
    t = len(tonight)
    if t:
        ups = sum(1 for x in tonight if x["upset"] is not None)
        s = f"{count_word(t)} on tonight’s slate"
        if ups:
            s += f", {count_word(ups, True)} on upset watch"
        parts.append(s)
    else:
        parts.append("nothing scheduled tonight")
    outs = [e for e in g(brief, "injury_watch", "out_tonight", default=[]) or []
            if isinstance(e, dict) and (e.get("status") or "").lower() == "out"
            and e.get("player")]
    if outs:
        o = outs[0]
        parts.append(f"{o['player']} is out for {o.get('team', '')}")
    dek = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
    return {"headline": head, "auto": auto, "dek": dek}


def section_text(sec) -> str | None:
    return text_of(g(sec, "text"))


# ------------------------------------------------------------------ the 60-second version

def _streak_words(streak: str | None) -> str:
    m = re.fullmatch(r"([WL])(\d+)", streak or "")
    if not m or int(m.group(2)) < 3:
        return ""
    return f"{'won' if m.group(1) == 'W' else 'lost'} {m.group(2)} straight"


def _game_bullet(c: dict) -> str:
    """'Clippers 115, Nuggets 114: the underdog Clippers came back from 10 down, and Bennedict
    Mathurin scored 38.' Every number comes from the card's facts."""
    win, lose = (c["home"], c["away"]) if c["home"]["winner"] else (c["away"], c["home"])
    s = f"{win['nickname']} {win['score']}, {lose['nickname']} {lose['score']}"
    dog = (c.get("market") or {}).get("favorite_won") is False
    verbs = []
    cb = g(c, "facts", "comeback") or {}
    if cb.get("team") == win["code"] and (cb.get("deficit") or 0) >= 8:
        verbs.append(f"came back from {cb['deficit']} down")
    if c.get("overtimes"):
        verbs.append("won in overtime" if c["overtimes"] == 1 else
                     f"won in {count_word(c['overtimes'], True)} overtimes")
    if dog and not verbs:
        verbs.append("won as the betting underdog")
        dog = False
    clauses = []
    if verbs:
        clauses.append(f"the {'underdog ' if dog else ''}{win['nickname']} {' and '.join(verbs)}")
    tops = [p for p in g(c, "facts", "top_performers", default=[]) or []
            if isinstance(p, dict) and p.get("team") == win["code"] and p.get("name")]
    if tops:
        pts = parse_line(tops[0].get("line")).get("pts")
        if pts and pts >= 25:
            clauses.append(f"{tops[0]['name']} scored {pts}")
    if clauses:
        s += ": " + ", and ".join(clauses)
    return s + "."


def _trend_bullet(team: dict, player: dict) -> tuple[str, str] | None:
    ts = [t for t in team.get("trending", []) if t.get("z") is not None]
    if ts:
        t = max(ts, key=lambda t: abs(t["z"]))
        s = f"The {t['team']['nickname']}"
        if t.get("last10"):
            s += f" are {fmt.record(t['last10'])} in their last 10"
        sw = _streak_words(t.get("streak"))
        if sw:
            s += f" and have {sw}" if t.get("last10") else f" have {sw}"
        net = t.get("net") or {}
        if net.get("last10_adj") is not None and net.get("season_adj") is not None:
            s += (f" (net rating {fmt.signed(net['last10_adj'])}, up from "
                  f"{fmt.signed(net['season_adj'])})" if net["last10_adj"] >= net["season_adj"]
                  else f" (net rating {fmt.signed(net['last10_adj'])}, down from "
                  f"{fmt.signed(net['season_adj'])})")
        if s == f"The {t['team']['nickname']}":
            return None
        return ("Rising" if t["direction"] == "up" else "Sliding"), s + "."
    hot = player.get("hot") or []
    if hot:
        p = hot[0]
        a, b = (p.get("recent") or {}).get("pts"), (p.get("season") or {}).get("pts")
        if a is not None and b is not None:
            return "Hot", (f"{p['player']} is averaging {fmt.num(a)} points over his "
                           f"{p['window'] or 'recent games'}, up from {fmt.num(b)} on the season.")
    return None


def _injury_bullet(inj: dict) -> str | None:
    outs = inj.get("out_tonight") or []
    parts = []
    lead = next((e for e in outs if (e.get("status") or "").lower() == "out"), None) or \
        (outs[0] if outs else None)
    if lead and lead.get("player"):
        st = (lead.get("status") or "out").lower()
        s = f"{lead['player']} ({lead['team']['code']}) is {st} tonight"
        if lead.get("opponent"):
            s += f" against the {identity(lead['opponent'])['nickname']}"
        if lead.get("reason"):
            s += f" ({lead['reason'][:1].lower() + lead['reason'][1:]})"
        parts.append(s)
    new = next((e for e in inj.get("new") or [] if e.get("tonight") and e.get("player")
                and (lead is None or e["player"] != lead.get("player"))
                and (_number(e.get("impact")) or 0) >= 12), None)
    if new and new.get("to_status"):
        parts.append(f"{new['player']} ({new['team']['code']}) is newly "
                     f"{new['to_status'].lower()}")
    return "; ".join(parts) + "." if parts else None


def _marquee(tonight: list[dict]) -> dict | None:
    def key(t):
        ranks = [s.get("power_rank") for s in (t["home"], t["away"])]
        avg = sum(ranks) / 2 if all(isinstance(r, int) for r in ranks) else 31
        close = abs(t["pred"]["home_prob"] - 0.5) if t.get("pred") else 0.5
        return avg + 30 * close
    with_pred = [t for t in tonight if t.get("pred")]
    return min(with_pred or tonight, key=key) if tonight else None


def _marquee_bullet(t: dict) -> str:
    s = f"{t['away']['nickname']} at {t['home']['nickname']}"
    if t.get("tipoff"):
        s += f", {t['tipoff']}"
    ra, rh = t["away"].get("power_rank"), t["home"].get("power_rank")
    bits = []
    if isinstance(ra, int) and isinstance(rh, int):
        bits.append(f"#{ra} vs #{rh} in our power ranking")
    if t.get("pred"):
        bits.append(f"we have the {t['pred']['pick']['nickname']} at "
                    f"{fmt.pct(t['pred']['pick_prob'])}")
    return s + (": " + ", and ".join(bits) if bits else "") + "."


def _upset_bullet(t: dict, others: int) -> str | None:
    u = t.get("upset") or {}
    if not u.get("underdog") or u.get("model_prob") is None or u.get("market_prob") is None:
        return None
    s = (f"We give the {u['underdog']['nickname']} {fmt.pct(u['model_prob'])} against the "
         f"{u['favorite']['nickname']}; the market has them at {fmt.pct(u['market_prob'])}.")
    if others:
        s += f" {count_word(others)} more {plural(others, 'game')} on the watch list."
    return s


def sixty_view(brief: dict, cards: list[dict], tonight: list[dict], team: dict, player: dict,
               inj: dict, score: dict) -> dict:
    """The 60-second version: 3-5 takeaways written from the facts, no LLM.

    Hook: a writer may later supply `brief["sixty_seconds"] = {"bullets": [str, ...]}`; when
    present (and non-empty) those bullets are shown instead of the deterministic ones.
    """
    sec = brief.get("sixty_seconds") if isinstance(brief.get("sixty_seconds"), dict) else {}
    written = [text_of(b) for b in sec.get("bullets") or []]
    written = [b for b in written if b]
    if written:
        return {"auto": False, "bullets": [{"label": None, "href": None, "text": b}
                                           for b in written[:5]]}
    items = []
    finals = [c for c in cards if c["final"] and c.get("winner")
              and c["home"]["score"] is not None and c["away"]["score"] is not None]
    if finals:
        best = finals[0]
        items.append({"label": "Game of the night", "href": f"#g{best['game_id']}",
                      "text": _game_bullet(best)})
    elif not cards:
        items.append({"label": "Last night", "href": "#last-night",
                      "text": "No games last night, so there is nothing to recap."})
    tb = _trend_bullet(team, player) if (team.get("available") or player.get("available")) \
        else None
    if tb:
        items.append({"label": tb[0], "href": "#teams" if tb[0] != "Hot" else "#players",
                      "text": tb[1]})
    ib = _injury_bullet(inj) if inj.get("available") else None
    if ib:
        items.append({"label": "Injury", "href": "#injuries", "text": ib})
    ups = sorted([t for t in tonight if t.get("upset")],
                 key=lambda t: -abs(_number((t["upset"] or {}).get("edge")) or 0))
    mq = _marquee([t for t in tonight if not ups or t is not ups[0]] or tonight)
    if mq:
        items.append({"label": "Tonight", "href": "#tonight", "text": _marquee_bullet(mq)})
    elif not tonight:
        items.append({"label": "Tonight", "href": "#tonight",
                      "text": "Nothing on the schedule tonight."})
    if ups:
        ub = _upset_bullet(ups[0], len(ups) - 1)
        if ub:
            items.append({"label": "Upset watch", "href": "#tonight", "text": ub})
    if len(items) < 5 and score.get("n"):
        items.append({"label": "Scorecard", "href": "#scorecard",
                      "text": f"Our picks are {score['correct']}{fmt.DASH}{score['incorrect']} "
                              f"({fmt.pct(score['accuracy'], 1)}) this season."})
    return {"auto": True, "bullets": items[:5]}


def brief_view(brief: dict) -> dict:
    meta = brief.get("meta") or {}
    ln = brief.get("last_night") or {}
    cards = order_cards([card_view(c) for c in ln.get("games") or [] if isinstance(c, dict)])
    tn = brief.get("tonight") or {}
    tonight = [tonight_view(x) for x in tn.get("games") or [] if isinstance(x, dict)]
    tp = brief.get("team_pulse") or {}
    pp = brief.get("player_pulse") or {}
    iw = brief.get("injury_watch") or {}
    sc = brief.get("scorecard") or {}
    lsum = ln.get("summary") or {}

    team = {"available": bool(tp.get("available")), "reason": tp.get("reason"),
            "early": bool(tp.get("early_season")), "text": section_text(tp),
            "rising": [team_trend_view(t, "up") for t in tp.get("rising") or []
                       if isinstance(t, dict)],
            "sliding": [team_trend_view(t, "down") for t in tp.get("sliding") or []
                        if isinstance(t, dict)],
            "power": [{**r, "team_id": identity(r.get("team"))}
                      for r in tp.get("power_top10") or [] if isinstance(r, dict)],
            "movers": [{**r, "team_id": identity(r.get("team"))}
                       for r in tp.get("power_movers_7d") or [] if isinstance(r, dict)],
            "streaks": [{**r, "team_id": identity(r.get("team"))}
                        for r in tp.get("streaks") or [] if isinstance(r, dict)],
            "validity": g(tp, "method", "flag_validity") or {},
            "data_through": tp.get("data_through")}
    team["trending"] = sorted(team["rising"] + team["sliding"],
                           key=lambda t: -abs(t["z"]) if t["z"] is not None else 0)
    player = {"available": bool(pp.get("available")), "reason": pp.get("reason"),
              "early": bool(pp.get("early_season")), "text": section_text(pp),
              "hot": [player_trend_view(p, True) for p in pp.get("hot") or []
                      if isinstance(p, dict)],
              "cold": [player_trend_view(p, False) for p in pp.get("cold") or []
                       if isinstance(p, dict)],
              "roles": [role_view(r) for r in pp.get("role_changes") or [] if isinstance(r, dict)],
              "leaders": leaders_view(pp.get("leaders_last10")),
              "streaks": [s for s in pp.get("streaks") or [] if isinstance(s, dict)],
              "career_highs": [s for s in pp.get("career_highs_last7d") or []
                               if isinstance(s, dict)],
              "validity": g(pp, "method", "hot_cold_validity") or {},
              "impact_metric": g(pp, "method", "impact_metric")}
    ch = iw.get("changes") or {}
    injuries = {"available": bool(iw.get("available")), "diff": bool(iw.get("diff_available")),
                "text": section_text(iw),
                "out_tonight": [injury_row(e) for e in iw.get("out_tonight") or []
                                if isinstance(e, dict)],
                "new": [injury_row(e) for e in ch.get("new_absences") or [] if isinstance(e, dict)],
                "returns": [injury_row(e) for e in ch.get("returns") or [] if isinstance(e, dict)],
                "status": [injury_row(e) for e in ch.get("status_changes") or []
                           if isinstance(e, dict)],
                "latest": iw.get("latest_snapshot"), "baseline": iw.get("baseline_snapshot"),
                "sources": iw.get("sources") or []}
    injuries.update(injury_split(injuries))
    vm = sc.get("vs_market") if isinstance(sc.get("vs_market"), dict) else None
    score = {"n": sc.get("n") or 0, "correct": sc.get("correct") or 0,
             "incorrect": sc.get("incorrect") or 0, "accuracy": sc.get("accuracy"),
             "brier": sc.get("brier"), "log_loss": sc.get("log_loss"), "ece": sc.get("ece"),
             "model": sc.get("model") or meta.get("model"), "through": sc.get("through"),
             "season": sc.get("season") or meta.get("season"), "vs_market": vm,
             "calibration": [b for b in sc.get("calibration") or [] if isinstance(b, dict)],
             "calib_svg": charts.calibration_chart(sc.get("calibration") or []),
             "by_conf": [b for b in sc.get("by_confidence") or [] if isinstance(b, dict)],
             "recent": sc.get("recent") or {}, "text": section_text(sc)}
    picks = None
    if lsum.get("picks_scored"):
        picks = {"correct": lsum.get("picks_correct") or 0, "n": lsum["picks_scored"]}
    return {
        "meta": meta, "day": meta.get("day"), "masthead": masthead(brief, cards, tonight),
        "sixty": sixty_view(brief, cards, tonight, team, player, injuries, score),
        "last_night": {"day": ln.get("day") or meta.get("last_night_day"), "cards": cards,
                       **card_tiers(cards),
                       "picks": picks, "text": section_text(ln), "summary": lsum},
        "team": team, "player": player, "injuries": injuries,
        "tonight": {"day": tn.get("day") or meta.get("day"), "games": tonight,
                    "text": section_text(tn), "upsets": sum(1 for x in tonight if x["upset"]
                                                            is not None)},
        "score": score,
        "notes": [n for n in meta.get("notes") or [] if isinstance(n, str) and n.strip()],
    }


# ------------------------------------------------------------------ game page

def impact_rows(facts: dict | None) -> list[dict]:
    """Merge top performers and unsung-hero candidates into one per-player table."""
    rows: dict = {}
    for src in ("top_performers", "unsung_hero"):
        for p in g(facts, src, default=[]) or []:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            key = p.get("id") or (p["name"], p.get("team"))
            r = rows.setdefault(key, {"name": p["name"], "team": identity(p.get("team")),
                                      "tags": []})
            for k in ("minutes", "plus_minus", "ts_pct", "on_off", "stocks", "game_score",
                      "line", "bench"):
                if p.get(k) is not None and r.get(k) is None:
                    r[k] = p[k]
            r["tags"].append("Top performer" if src == "top_performers" else "Unsung")
    out = []
    for r in rows.values():
        pl = parse_line(r.get("line"))
        r["pts"], r["reb"], r["ast"] = pl.get("pts", 0), pl.get("reb", 0), pl.get("ast", 0)
        if r.get("stocks") is None:
            r["stocks"] = pl.get("stl", 0) + pl.get("blk", 0)
        fg = pl.get("fg")
        r["fg"] = f"{fg[0]}-{fg[1]}" if fg else ""
        out.append(r)
    win = g(facts, "winner")
    out.sort(key=lambda r: (r["team"]["code"] != win, r["team"]["code"],
                            -(r.get("game_score") or 0), -(r["pts"] or 0)))
    return out


def runs_view(facts: dict | None) -> list[dict]:
    out = []
    home, away = g(facts, "home", "team"), g(facts, "away", "team")
    for r in g(facts, "runs", default=[]) or []:
        if not isinstance(r, dict) or not r.get("team"):
            continue
        team = r["team"]
        opp = away if team == home else home
        frm, to = r.get("from") or {}, r.get("to") or {}
        out.append({"team": identity(team), "run": fmt.run(r.get("run")),
                    "unanswered": bool(r.get("unanswered")), "minutes": r.get("minutes"),
                    "from": f"{frm.get('label', '')} {frm.get('clock', '')}".strip(),
                    "to": f"{to.get('label', '')} {to.get('clock', '')}".strip(),
                    "effect": run_effect(team, opp, frm.get("score"), to.get("score"))
                    if opp else "",
                    "includes": r.get("includes")})
    return out


def game_view(card: dict) -> dict:
    cv = card_view(card)
    f = cv["facts"] or {}
    hp = g(f, "home", "periods", default=[]) or []
    ap = g(f, "away", "periods", default=[]) or []
    labels = [lab for lab, _, _ in charts.period_labels(max(0, len(hp) - 4))][:len(hp)]
    ll = []
    for code, x in (g(f, "largest_lead") or {}).items():
        if isinstance(x, dict) and x.get("lead") is not None:
            ll.append({"team": identity(code), "lead": x["lead"],
                       "when": f"{x.get('label', '')} {x.get('clock', '')}".strip()})
    cb = g(f, "comeback")
    comeback = None
    if isinstance(cb, dict) and cb.get("team") and cb.get("deficit"):
        at = cb.get("at") or {}
        comeback = {"team": identity(cb["team"]), "deficit": cb["deficit"],
                    "when": f"{at.get('label', '')} {at.get('clock', '')}".strip()}
    gt = g(f, "garbage_time") or {}
    gts = gt.get("seconds") or 0
    val = g(f, "validation") or {}
    return {**cv, "periods": labels, "home_periods": hp, "away_periods": ap,
            "arena": f.get("arena"), "date": f.get("date"),
            "lead_chart": charts.lead_chart(f) if f else None,
            "runs": runs_view(f), "lead_changes": f.get("lead_changes"),
            "times_tied": f.get("times_tied"), "largest_leads": ll, "comeback": comeback,
            "garbage": {"seconds": gts, "from": gt.get("from")} if gts else None,
            "impact": impact_rows(f), "validation": val}
