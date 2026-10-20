"""Deterministic fallback prose, built only from the facts documents.

Used when the LLM is unavailable or two attempts fail the fact-check. These sentences are
plainer than the model's, but every number and name comes straight from the facts, and the
tests run them through ``factcheck`` for every fixture.

    game(facts)            -> {"headline", "story"}
    team_blurb(entry)      -> str            (entry = a compact pulse team entity)
    player_blurb(entry)    -> str
    pulse_headline(teams, players) -> str
    tonight(pred)          -> str            (pred = a normalised prediction, see writer)
    brief_headline(summary)-> str
    player_story(entry)    -> str            (entry = a Brief ``player_stories.stories[i]``)
"""

from __future__ import annotations

import re
from courtside.teams import TEAMS

_BY_TRI = {t.tricode: t for t in TEAMS}
_QUARTER = {"Q1": "first", "Q2": "second", "Q3": "third", "Q4": "fourth"}
_STAT_WORDS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes",
               "stocks": "steals plus blocks", "stl": "steals", "blk": "blocks"}


def nick(code_or_name: str | None) -> str:
    """'LAC' -> 'Clippers'; 'LA Clippers' -> 'Clippers'; unknown strings pass through."""
    if not code_or_name:
        return "the team"
    t = _BY_TRI.get(code_or_name)
    if t:
        return t.name
    for t in TEAMS:
        if code_or_name == f"{t.city} {t.name}" or code_or_name.endswith(t.name):
            return t.name
    return code_or_name


def _fmt(x: float, nd: int = 1) -> str:
    s = f"{abs(x):.{nd}f}"
    return s


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _quarter(label: str | None) -> str:
    if not label:
        return "game"
    if label in _QUARTER:
        return f"{_QUARTER[label]} quarter"
    return "overtime"


def _ot_phrase(n: int) -> str:
    return {1: "overtime", 2: "double overtime", 3: "triple overtime"}.get(n, f"{n} overtimes")


def _score_str(scores: dict, first: str) -> str:
    other = [k for k in scores if k != first]
    return f"{scores[first]}-{scores[other[0]]}" if other else str(scores[first])


# ---------------------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------------------

def _sides(f: dict) -> tuple[dict, dict]:
    home, away = f.get("home") or {}, f.get("away") or {}
    return (home, away) if f.get("winner") == home.get("team") else (away, home)


def game_headline(f: dict) -> str:
    w, l = _sides(f)
    W, L = nick(w["team"]), nick(l["team"])
    margin, ot = int(f.get("margin") or 0), int(f.get("overtimes") or 0)
    cb = f.get("comeback") or {}
    if cb.get("team") == w["team"] and (cb.get("deficit") or 0) >= 15:
        return f"{W} erase {cb['deficit']}-point deficit to beat {L}"
    if ot:
        return f"{W} outlast {L} in {_ot_phrase(ot)}"
    if margin >= 20:
        return f"{W} rout {L} by {margin}"
    if margin <= 3:
        return f"{W} edge {L} by {'a point' if margin == 1 else str(margin)}"
    if margin >= 12:
        return f"{W} pull away from {L}"
    return f"{W} beat {L} by {margin}"


def _first_sentence(f: dict) -> str:
    w, l = _sides(f)
    W, L = nick(w["team"]), nick(l["team"])
    ws, ls = w["score"], l["score"]
    margin, ot = int(f.get("margin") or 0), int(f.get("overtimes") or 0)
    cb = f.get("comeback") or {}
    if cb.get("team") == w["team"] and (cb.get("deficit") or 0) >= 15:
        q = _quarter((cb.get("at") or {}).get("label"))
        return (f"The {L} led by {cb['deficit']} in the {q}, but the {W} came all the way "
                f"back to win {ws}-{ls}.")
    if ot:
        s = f"The {W} needed {_ot_phrase(ot)} to beat the {L} {ws}-{ls}."
        return s
    if margin >= 20:
        ll = (f.get("largest_lead") or {}).get(w["team"]) or {}
        if ll.get("lead", 0) > margin:
            return f"The {W} led by as many as {ll['lead']} on the way to a {ws}-{ls} win over the {L}."
        return f"The {W} rolled past the {L} {ws}-{ls}."
    lc = int(f.get("lead_changes") or 0)
    if margin <= 5 and lc >= 3:
        return f"The {W} edged the {L} {ws}-{ls} in a game where the lead changed hands {lc} times."
    if margin <= 5:
        return f"The {W} edged the {L} {ws}-{ls}."
    return f"The {W} beat the {L} {ws}-{ls}."


def _turning_point_sentence(f: dict) -> str | None:
    tp = f.get("turning_point")
    if not tp or not tp.get("run") or not tp.get("team"):
        return None
    T = nick(tp["team"])
    q = _quarter((tp.get("from") or {}).get("label"))
    to = tp.get("to") or {}
    reason = tp.get("reason")
    end = f" with {to['clock']} left" if to.get("clock") and to.get("label") in _QUARTER else ""
    score = _score_str(to["score"], tp["team"]) if to.get("score") else ""
    if reason == "levelled" and score:
        return (f"The turning point came in the {q}, when the {T} went on a {tp['run']} run "
                f"to level it at {score}{end}.")
    if reason == "flipped_lead" and score:
        return (f"The turning point came in the {q}, when a {tp['run']} run put the {T} "
                f"ahead {score}{end}.")
    if reason == "created_winning_margin" and score:
        return (f"The game turned in the {q}, when a {tp['run']} run built a "
                f"lead of {score} that the {T} never gave back.")
    return f"The {T} had the game's biggest swing in the {q}, a {tp['run']} run."


def _line_prose(line: str) -> tuple[str, str]:
    """'28 pts (13-26 FG, 1-6 3P, 1-2 FT), 8 reb, 4 ast' -> ('28 points', 'on 13-of-26 shooting...')"""
    pts = re.match(r"(\d+) pts", line)
    fg = re.search(r"\((\d+)-(\d+) FG", line)
    others = []
    for n, k in re.findall(r"(\d+) (reb|ast|stl|blk)\b", line):
        others.append(f"{n} {_STAT_WORDS[k] if int(n) != 1 else _STAT_WORDS[k].rstrip('s')}")
    head = f"{pts.group(1)} points" if pts else line
    if fg:
        head += f" on {fg.group(1)}-of-{fg.group(2)} shooting"
    return head, ", ".join(others)


def _performer_sentence(f: dict) -> str | None:
    tps = f.get("top_performers") or []
    tps = [p for p in tps if isinstance(p, dict) and p.get("name") and p.get("line")]
    if not tps:
        return None
    best = max(tps, key=lambda p: p.get("game_score") or 0)
    head, rest = _line_prose(best["line"])
    s = f"{best['name']} posted {head}"
    if rest:
        parts = rest.split(", ")
        s += ", " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1])
    if best.get("team") != f.get("winner"):
        s += " in defeat"
    return s + "."


def _why_prose(h: dict) -> str | None:
    bits = []
    team = nick(h.get("team"))
    for w in h.get("why") or []:
        m = re.match(r"([+-]?\d+) on the floor", w)
        if m:
            bits.append(f"a {m.group(1)} in {h['minutes']:.0f} minutes" if h.get("minutes")
                        else f"a {m.group(1)} on the floor")
            continue
        m = re.match(r"team \+([\d.]+) per 100 better with him on", w)
        if m:
            bits.append(f"the {team} {m.group(1)} points per 100 possessions better when he played")
            continue
        m = re.match(r"(\d+)% TS", w)
        if m:
            bits.append(f"{m.group(1)}% true shooting")
            continue
        m = re.match(r"([+-]\d+) during the decisive run", w)
        if m:
            bits.append(f"a {m.group(1)} during the decisive run")
            continue
        m = re.match(r"(\d+) (assists|steals|blocks|rebounds)", w)
        if m:
            bits.append(f"{m.group(1)} {m.group(2)}")
    return ", ".join(bits[:3]) if bits else None


def _hero_sentence(f: dict) -> str | None:
    heroes = [h for h in f.get("unsung_hero") or [] if isinstance(h, dict) and h.get("name")]
    pick = next((h for h in heroes if h.get("winner")), heroes[0] if heroes else None)
    if not pick:
        return None
    why = _why_prose(pick)
    if not why:
        return None
    return f"{pick['name']} flew under the radar: {why}."


def _incident_sentence(f: dict) -> str | None:
    inc = f.get("incidents") or {}
    for e in inc.get("ejections") or []:
        who = e.get("name") or e.get("player") if isinstance(e, dict) else None
        if who:
            return f"{who} was ejected."
    for p in inc.get("possible_injuries") or []:
        if not isinstance(p, dict):
            continue
        lo = p.get("last_on_floor") or {}
        if p.get("name") and lo.get("label"):
            return f"{p['name']} left in the {_quarter(lo['label'])} and did not return."
    return None


def game(f: dict) -> dict[str, str]:
    sents = [_first_sentence(f)]
    ok = (f.get("validation") or {}).get("ok", True)
    if ok:
        for fn in (_turning_point_sentence, _performer_sentence, _hero_sentence):
            s = fn(f)
            if s:
                sents.append(s)
    else:
        s = _performer_sentence(f)
        if s:
            sents.append(s)
    if len(sents) < 4:
        s = _incident_sentence(f)
        if s:
            sents.append(s)
    return {"headline": game_headline(f), "story": " ".join(sents[:4])}


# ---------------------------------------------------------------------------------------
# pulse
# ---------------------------------------------------------------------------------------

def team_blurb(e: dict) -> str:
    T = nick(e.get("team") or e.get("name"))
    flag = e.get("flag")
    net = e.get("net") or {}
    last10, rec = e.get("last10"), e.get("record")
    if flag in ("rising", "sliding"):
        word = "climbing" if flag == "rising" else "sliding"
        s = f"The {T} are {last10} over their last 10"
        if net.get("last10_adj") is not None and (e.get("trend") or {}).get("baseline") is not None:
            s += (f", with an adjusted net rating of {net['last10_adj']:+.1f} against "
                  f"{e['trend']['baseline']:+.1f} before that")
        s += f". It's a {word} signal, real but modest: most of a 10-game gap is noise."
        return s
    if e.get("streak_detail"):
        return f"The {T} have {e['streak_detail']} and are {rec}."
    if e.get("rank_7d_ago") is not None and e.get("rank") is not None:
        return (f"The {T} moved from No. {e['rank_7d_ago']} to No. {e['rank']} in our power "
                f"ranking over the past week.")
    if e.get("power_rank") is not None:
        return f"The {T} are {rec} and No. {e['power_rank']} in our power ranking."
    return f"The {T} are {rec}." if rec else f"The {T} are in the pulse."


def player_blurb(e: dict) -> str:
    n = e["player"]
    tags = e.get("tags") or []
    recent, season = e.get("recent") or {}, e.get("season") or {}
    if ("hot" in tags or "cold" in tags) and recent and season:
        direction = "up" if "hot" in tags else "down"
        s = (f"{n} is averaging {recent['pts']} points over his last {recent['games']} games, "
             f"{direction} from {season['pts']} for the season")
        if recent.get("ts_pct"):
            s += f", on {recent['ts_pct'] * 100:.1f}% true shooting"
        return s + "."
    if e.get("role_details"):
        return f"{n} has a changed role: {e['role_details'][0]}."
    if e.get("career_high"):
        ch = e["career_high"]
        word = _STAT_WORDS.get(ch.get("stat"), ch.get("stat"))
        return f"{n} set a career high of {ch['value']} {word} (previous best {ch['previous_high']})."
    if e.get("streak_detail"):
        return f"{n} has {e['streak_detail']}."
    return f"{n} is in the pulse."


def pulse_headline(teams: list[dict], players: list[dict]) -> str:
    rising = [t for t in teams if t.get("flag") == "rising"]
    sliding = [t for t in teams if t.get("flag") == "sliding"]
    if rising and sliding:
        return f"{nick(rising[0]['team'])} surge while {nick(sliding[0]['team'])} slide"
    if rising:
        return f"{nick(rising[0]['team'])} are the league's biggest riser"
    if sliding:
        return f"{nick(sliding[0]['team'])} are the league's biggest slider"
    top = next((t for t in teams if t.get("power_rank") == 1), None)
    if top:
        return f"{nick(top['team'])} sit atop our power ranking"
    return "The pulse of the league"


# ---------------------------------------------------------------------------------------
# tonight / brief headline
# ---------------------------------------------------------------------------------------

def tonight(p: dict) -> str:
    home, away = nick(p.get("home")), nick(p.get("away"))
    prob = p.get("home_win_prob")
    margin = p.get("home_margin")
    if prob is None and margin is None:
        return f"The {away} visit the {home}."
    home_fav = (prob if prob is not None else 0.5 + (margin or 0)) >= 0.5
    fav, dog = (home, away) if home_fav else (away, home)
    fp = (prob if home_fav else 1 - prob) if prob is not None else None
    s = f"The {fav} are "
    s += f"{round(fp * 100)}% favorites" if fp is not None else "favorites"
    s += f" over the {dog} in our model"
    if margin is not None:
        s += f", by a projected {abs(margin):.1f}"
    s += "."
    sp = p.get("market_home_spread")
    if sp is not None and sp != 0:
        mfav = home if sp < 0 else away
        s += f" The market line has the {mfav} by {abs(sp):g}."
    elif p.get("reasons"):
        r = str(p["reasons"][0]).rstrip(".")
        s += f" Key factor: {r}."
    return s


def brief_headline(summary: dict) -> str:
    games = summary.get("games") or []
    if games:
        def interest(g: dict) -> tuple:
            return (int(g.get("overtimes") or 0) > 0, g.get("comeback_deficit") or 0,
                    -int(g.get("margin") or 99))
        best = max(games, key=interest)
        if best.get("headline"):
            return str(best["headline"]).rstrip(".")
    p = (summary.get("pulse") or {}).get("headline")
    if p:
        return str(p).rstrip(".")
    return "The Courtside Morning Brief"


# ---------------------------------------------------------------------------------------
# player stories (last night vs our projection, in DK points)
# ---------------------------------------------------------------------------------------

def _count(x) -> str:
    """A projected count as shown: one decimal, a trailing .0 dropped (24.0 -> 24, 1.8 -> 1.8)."""
    s = f"{float(x):.1f}"
    return s[:-2] if s.endswith(".0") else s


def _dk(x) -> str:
    return f"{float(x or 0):.1f}"


_STORY_NOUN = {"pts": ("point", "points"), "reb": ("rebound", "rebounds"),
               "ast": ("assist", "assists"), "fg3m": ("three", "threes"),
               "stl": ("steal", "steals"), "blk": ("block", "blocks"),
               "tov": ("turnover", "turnovers")}
_STORY_VERB = {"pts": "had", "reb": "grabbed", "ast": "handed out", "fg3m": "made", "stl": "had",
               "blk": "had", "tov": "committed"}


def _shooting(line: dict) -> str:
    m = re.match(r"(\d+)-(\d+)", str(line.get("fg") or ""))
    return f" on {m.group(1)}-of-{m.group(2)} shooting" if m else ""


def story_reason(r: dict, e: dict, only: bool = True) -> str | None:
    """One reason as box-score English: 'played 41 minutes (32 expected)'. `only` marks a
    figure below the projection ('played only 22 minutes'); used once per line."""
    k = r.get("kind")
    if k in ("double_double", "triple_double"):
        return f"picked up the {k.replace('_', '-')} bonus"
    a, p = r.get("actual"), r.get("projected")
    if a is None or p is None:
        return None
    if k == "minutes":
        return (f"played {'only ' if only and a < p else ''}{a} {_plural(int(a), 'minute')} "
                f"({p} expected)")
    if k not in _STORY_NOUN:
        return None
    one, many = _STORY_NOUN[k]
    s = f"{_STORY_VERB[k]} {'only ' if only and a < p else ''}{a} {one if int(a) == 1 else many} ({_count(p)} projected)"
    if k == "pts":
        s += _shooting(e.get("line") or {})
    return s


def _story_context(e: dict) -> str | None:
    c = e.get("context") or {}
    kinds = {r.get("kind"): r for r in e.get("reasons") or []}
    mins = kinds.get("minutes")
    bits = []
    if c.get("fouled_out"):
        bits.append("he fouled out")
    if c.get("ejected"):
        bits.append("he was ejected")
    if c.get("blowout_margin") and mins:          # cut a starter's minutes or padded a bench's
        bits.append(f"the game was a {c['blowout_margin']}-point blowout")
    if c.get("overtime") and mins and mins["actual"] > mins["projected"] \
            and not c.get("blowout_margin"):
        bits.append(f"the game went to {_ot_phrase(int(c['overtime']))}")
    tm = [n for n in c.get("teammates_not_playing") or [] if isinstance(n, str)][:2]
    if tm and (e.get("dk") or {}).get("diff", 0) > 0 and mins and mins["actual"] > mins["projected"]:
        bits.append(f"{' and '.join(tm)} did not play")
    if not bits:
        return None
    s = "; ".join(bits[:2])
    return s[0].upper() + s[1:] + "."


def player_story(e: dict) -> str:
    """Deterministic story line: '<Player> scored 52.3 DK points vs 38.0 projected: he played
    41 minutes (32 expected) and made 6 threes (1.8 projected).' Box-score English only, never
    a cause the facts do not state; a player who did not play gets no reason at all."""
    name = e.get("player") or "He"
    dk = e.get("dk") or {}
    if e.get("did_not_play"):
        m = (e.get("minutes") or {}).get("expected")
        tail = f" and about {m} {_plural(int(m), 'minute')}" if m else ""
        return f"{name} did not play; we had projected {_dk(dk.get('expected'))} DK points{tail}."
    head = f"{name} scored {_dk(dk.get('actual'))} DK points vs {_dk(dk.get('expected'))} projected"
    parts = [t for t in (story_reason(r, e, only=i == 0)
                         for i, r in enumerate(e.get("reasons") or [])) if t]
    if parts:
        body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        s = f"{head}: he {body}."
    else:
        s = f"{head}, with the gap spread across his line."
    ctx = _story_context(e)
    return s + (" " + ctx if ctx else "")
