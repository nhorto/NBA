"""The Brief's prose: LLM drafts, fact-checked against the facts, template fallback.

Public API (all take an optional ``complete=`` callable for tests; default ``llm.complete``):

    write_game_stories(facts_list)   -> {game_id: {"headline", "story", "source"}}
    write_pulse(league_facts)        -> {"headline", "teams": {tricode: text},
                                         "players": {player_key: text}, "sources": {...}}
    write_tonight(predictions)       -> {game_id: text}
    write_brief_headline(summary)    -> {"headline", "source"}
    fill(brief)                      -> brief with every ``"text": null`` slot written

Every function makes ONE batched LLM call (plus at most one retry call for the items that
failed the fact-check). Each item ends with a ``source``:

    llm_ok          first draft passed ``factcheck``
    llm_retry_ok    passed after one retry that was told what was wrong
    template        both drafts failed, or no LLM backend worked: ``templates.py`` wrote it

Pass ``report={}`` to any ``write_*`` to receive ``{item_key: {"source", "problems",
"warnings"}}`` (problems = the violations of the last failed attempt).

------------------------------------------------------------------------------------------
Input contracts (tolerant: extra keys are ignored, missing optional keys are omitted)
------------------------------------------------------------------------------------------

``facts_list``  list of ``analytics.summarize.game_facts`` documents. Games whose
    ``validation.ok`` is false are not sent to the LLM (template only).

``league_facts``  the ``trends.summarize.league_facts`` document. The writer picks the
    entities worth a blurb: flagged teams, biggest power movers, long streaks, hot/cold players,
    role changes, streaks, career highs (<= 8 teams, <= 12 players).

``predictions``  list of dicts: either the docs/BRIEF.md ``tonight.games[i]`` objects as they
    are (``prediction``, ``market``, ``edge``, ``factors``, ``injuries {home, away}``, ``upset``)
    or flat dicts, each roughly::

    {"game_id": "0022500123", "home": "BOS", "away": "MIA",    # tricode, name, or {"team": ..}
     "home_win_prob": 0.71,           # fraction (a value > 1 is read as a percent)
     "home_margin": 7.2,              # model's projected home margin, + = home wins
     "total": 228.5,                  # model's projected total points
     "market_home_spread": -4.5,      # market line, betting convention (- = home favored)
     "market_total": 226.0,
     "edge": 2.7,                     # optional, passed through if present
     "reasons": ["Miami on the second night of a back-to-back", {"detail": "..."}],
     "injuries": ["Bam Adebayo (out)", {"player": "..", "team": "MIA", "status": "out"}]}

    Aliases accepted: p_home/win_prob (prob), margin/proj_margin (home_margin),
    proj_total (total), spread/market_spread (market_home_spread), home_team/away_team.

``brief_summary``  ``{"date", "games": [{"game_id", "headline", "home", "away", "home_score",
    "away_score", "winner", "margin", "overtimes", "comeback_deficit"}], "pulse":
    {"headline", "rising": [...], "sliding": [...]}, "tonight_games": int}``. Whatever is in it
    is the entire universe the headline may draw from.

------------------------------------------------------------------------------------------
``fill(brief)``: the docs/BRIEF.md schema (see ``fill``'s docstring for exactly which slots)
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from courtside.brief import factcheck, llm, templates
from courtside.brief.factcheck import FactIndex
from courtside.teams import TEAMS

PROMPT_DIR = Path(__file__).parent / "prompts"
_TRICODES = {t.tricode for t in TEAMS}
Complete = Callable[..., str]


# ---------------------------------------------------------------------------------------
# prompt plumbing
# ---------------------------------------------------------------------------------------

def _read(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text().strip()


def render(name: str, **subs: str) -> str:
    """Load prompts/<name>.md and fill ``{{KEY}}`` placeholders (``{{STYLE}}`` is automatic)."""
    text = _read(name).replace("{{STYLE}}", _read("_style"))
    for k, v in subs.items():
        text = text.replace("{{" + k.upper() + "}}", v)
    return text


def system_prompt() -> str:
    return _read("_system")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text``: tolerates code fences, preamble, trailing
    commentary, smart quotes and trailing commas. Raises ValueError if nothing parses."""
    if not isinstance(text, str):
        raise ValueError("not a string")
    t = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", t, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(t)
    dec = json.JSONDecoder()
    for cand in candidates:
        for fixed in (cand, _repair(cand)):
            for m in re.finditer(r"[{\[]", fixed):
                try:
                    obj, _ = dec.raw_decode(fixed[m.start():])
                except ValueError:
                    continue
                if isinstance(obj, (dict, list)):
                    return obj
    raise ValueError(f"no JSON found in model output: {t[:120]!r}")


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r",\s*([}\]])", r"\1", s)


# ---------------------------------------------------------------------------------------
# generic generate -> check -> retry -> template engine
# ---------------------------------------------------------------------------------------

@dataclass
class Item:
    key: str
    fields: dict[str, str]                 # output field -> factcheck kind
    scope: Any                             # facts this item is checked against
    template: Callable[[], dict[str, str]]
    _ix: FactIndex | None = field(default=None, repr=False)

    @property
    def index(self) -> FactIndex:
        if self._ix is None:
            self._ix = FactIndex.build(self.scope)
        return self._ix


@dataclass
class Outcome:
    fields: dict[str, str]
    source: str
    problems: list[str]
    warnings: list[str]


def _check_item(it: Item, cand: dict[str, str] | None) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    for fld, kind in it.fields.items():
        text = (cand or {}).get(fld)
        if text is None:
            problems.append(f"{fld}: missing from your answer")
            continue
        res = factcheck.check(text, it.index, kind=kind)
        problems += [f"{fld}: {m}" for m in res.messages()]
        warnings += [f"{fld}: {w}" for w in res.warnings]
    return problems, warnings


def _complete(complete: Complete | None, prompt: str, timeout: int) -> str:
    fn = complete or llm.complete
    return fn(prompt, system=system_prompt(), timeout=timeout)


def _normalize(items: list[Item], cands: dict[str, dict[str, str]] | None) -> None:
    """Headlines: sentence case, no trailing period (the model sometimes Title Cases them)."""
    for it in items:
        cand = (cands or {}).get(it.key)
        for fld, kind in it.fields.items():
            if cand and kind.endswith("headline") and isinstance(cand.get(fld), str):
                cand[fld] = factcheck.normalize_headline(cand[fld])


def _try_call(complete: Complete | None, prompt: str, timeout: int,
              extract: Callable[[Any], dict[str, dict[str, str]]]) -> tuple[dict | None, str | None]:
    """-> (candidates by item key, error). error is set when nothing usable came back."""
    try:
        text = _complete(complete, prompt, timeout)
    except llm.LLMError as e:
        return None, f"llm unavailable: {e}"
    try:
        return extract(extract_json(text)), None
    except (ValueError, TypeError, AttributeError) as e:
        return {}, f"unparseable model output: {e}"


def generate(prompt: str, items: list[Item], extract: Callable[[Any], dict[str, dict[str, str]]],
             *, complete: Complete | None = None, timeout: int = 180) -> dict[str, Outcome]:
    """One batched call, fact-check every field, one targeted retry, then templates."""
    out: dict[str, Outcome] = {}
    if not items:
        return out
    cands, err = _try_call(complete, prompt, timeout, extract)
    _normalize(items, cands)
    llm_down = cands is None
    pending: dict[str, list[str]] = {}
    prior: dict[str, dict[str, str]] = {}
    for it in items:
        cand = (cands or {}).get(it.key)
        problems, warnings = _check_item(it, cand)
        if llm_down:
            problems = [err or "llm unavailable"]
        if not problems:
            out[it.key] = Outcome(cand, "llm_ok", [], warnings)
        else:
            pending[it.key] = problems if cand is not None or llm_down else (
                problems + ([err] if err else []))
            prior[it.key] = cand or {}

    if pending and not llm_down:
        by_key = {it.key: it for it in items}
        listing = "\n".join(
            f"- {k}: {'; '.join(p)}\n  your answer was: {_dumps(prior[k])}" for k, p in pending.items())
        retry = prompt + "\n\n" + render("retry", problems=listing)
        cands2, err2 = _try_call(complete, retry, timeout, extract)
        _normalize(items, cands2)
        for k in list(pending):
            it = by_key[k]
            cand = (cands2 or {}).get(k)
            problems, warnings = _check_item(it, cand)
            if cands2 is not None and not problems:
                out[k] = Outcome(cand, "llm_retry_ok", pending[k], warnings)   # problems = 1st attempt
                del pending[k]
            else:
                pending[k] = problems or [err2 or "retry failed"]

    for it in items:
        if it.key in out:
            continue
        out[it.key] = Outcome(it.template(), "template", pending.get(it.key, []), [])
    return out


def _record(report: dict | None, outcomes: dict[str, Outcome], prefix: str = "") -> None:
    if report is None:
        return
    for k, o in outcomes.items():
        report[prefix + k] = {"source": o.source, "problems": o.problems, "warnings": o.warnings}


# ---------------------------------------------------------------------------------------
# 1. game stories
# ---------------------------------------------------------------------------------------

def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, [], {})}


def _snap(t: Any) -> dict | None:
    if not isinstance(t, dict):
        return None
    return _drop_none({"quarter": t.get("label"), "clock": t.get("clock"), "score": t.get("score")}) or None


def _d(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x: Any) -> list[dict]:
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []


def compact_game(f: dict) -> dict:
    """The part of a game_facts document the writer needs (about 40% of its size).

    Defensive: any field may be null or missing (a team that never led has a null
    ``largest_lead`` entry, a blowout has no ``turning_point`` worth naming, ...)."""
    def side(s: Any) -> dict:
        s = _d(s)
        return _drop_none({"team": s.get("team"),
                           "nickname": templates.nick(s["team"]) if s.get("team") else None,
                           "name": s.get("name"), "score": s.get("score"),
                           "by_period": s.get("periods")})
    tp = _d(f.get("turning_point"))
    runs = sorted((r for r in _l(f.get("runs")) if r.get("team") and r.get("run")),
                  key=lambda r: -_run_net(r.get("run")))[:4]
    clutch = _d(f.get("clutch"))
    inc = _d(f.get("incidents"))
    cb = _d(f.get("comeback"))
    gt = _d(f.get("garbage_time"))
    out = {
        "game_id": str(f.get("game_id")),
        "arena": f.get("arena"),
        "away": side(f.get("away")), "home": side(f.get("home")),
        "winner": f.get("winner"), "margin": f.get("margin"),
        "overtimes": f.get("overtimes") or None,
        "top_performers": [_drop_none({
            "name": p.get("name"), "team": p.get("team"), "line": p.get("line"),
            "minutes": p.get("minutes"), "plus_minus": p.get("plus_minus")})
            for p in _l(f.get("top_performers")) if p.get("name")],
        "unsung_hero": [_drop_none({
            "name": h.get("name"), "team": h.get("team"), "on_winning_team": h.get("winner"),
            "off_bench": h.get("bench"), "line": h.get("line"), "minutes": h.get("minutes"),
            "why": h.get("why")}) for h in _l(f.get("unsung_hero")) if h.get("name")],
        "turning_point": _drop_none({
            "team": tp.get("team"), "run": tp.get("run"), "reason": tp.get("reason"),
            "minutes": tp.get("minutes"), "start": _snap(tp.get("from")), "end": _snap(tp.get("to")),
        }) if tp.get("run") else None,
        "biggest_runs": [_drop_none({
            "team": r["team"], "run": r["run"], "uninterrupted_burst": r.get("includes_unanswered"),
            "minutes": r.get("minutes"), "start": _snap(r.get("from")), "end": _snap(r.get("to"))})
            for r in runs],
        "late_key_plays": [_drop_none({
            "team": k.get("team"), "player": k.get("player"), "kind": k.get("kind"),
            "quarter": k.get("label"), "clock": k.get("clock"), "score": k.get("score"),
            "decisive": k.get("decisive") or None}) for k in _l(clutch.get("key_plays"))]
        if clutch.get("had_clutch") else None,
        "lead_changes": f.get("lead_changes"), "times_tied": f.get("times_tied"),
        "largest_lead": {t: _drop_none({"lead": _d(v).get("lead"), "quarter": _d(v).get("label"),
                                        "clock": _d(v).get("clock")})
                         for t, v in _d(f.get("largest_lead")).items() if _d(v).get("lead")},
        "comeback": _drop_none({"team": cb.get("team"), "deficit": cb.get("deficit"),
                                "at": _snap(cb.get("at"))})
        if isinstance(cb.get("deficit"), (int, float)) and cb["deficit"] >= 10 else None,
        "garbage_time_seconds": gt.get("seconds") or None,
        "notable_lines": [_drop_none({"name": n.get("name"), "team": n.get("team"),
                                      "line": n.get("line"), "flags": n.get("flags")})
                          for n in _l(f.get("notable_lines")) if n.get("name")],
        "ejections": inc.get("ejections") or None,
        "foul_outs": inc.get("foul_outs") or None,
        "left_and_did_not_return_unverified": [_drop_none({
            "name": p.get("name"), "team": p.get("team"), "minutes": p.get("minutes"),
            "last_on_floor": _snap(p.get("last_on_floor"))})
            for p in _l(inc.get("possible_injuries")) if p.get("name")] or None,
    }
    return _drop_none(out)


def _run_net(run: Any) -> int:
    m = re.match(r"(\d+)-(\d+)", run if isinstance(run, str) else "")
    return int(m.group(1)) - int(m.group(2)) if m else 0


def write_game_stories(facts_list: list[dict], *, context: dict[str, dict] | None = None,
                       complete: Complete | None = None, report: dict | None = None,
                       timeout: int = 240) -> dict[str, dict]:
    """ONE call for the night's games -> {game_id: {"headline", "story", "source"}}.

    ``context`` optionally maps game_id -> extra facts about the game that don't live in
    game_facts, e.g. ``{"our_pregame_model": {"pick": "IND", "win_prob": 0.68, "correct": False}}``.
    It is shown to the writer and counts as facts for the checker."""
    games = {str(f["game_id"]): f for f in facts_list or [] if isinstance(f, dict) and f.get("game_id")}
    if not games:
        return {}
    result: dict[str, dict] = {}
    llm_items: list[Item] = []
    for gid, f in games.items():
        ctx = (context or {}).get(gid)
        item = Item(gid, {"headline": "game_headline", "story": "game_story"},
                    {"game": f, "context": ctx} if ctx else f, (lambda f=f: _safe_game_template(f)))
        if (f.get("validation") or {}).get("ok", True):
            llm_items.append(item)
        else:  # stint gate failed: don't let the model narrate suspect numbers
            result[gid] = {**_safe_game_template(f), "source": "template"}
            _record(report, {gid: Outcome({}, "template", ["validation.ok is false"], [])})
    if llm_items:
        compact = [{**compact_game(games[it.key]), **_drop_none({"context": (context or {}).get(it.key)})}
                   for it in llm_items]
        prompt = render("game_stories", facts=_dumps(compact))

        def extract(parsed: Any) -> dict[str, dict[str, str]]:
            body = parsed.get("games", parsed) if isinstance(parsed, dict) else {}
            if isinstance(body, list):
                body = {str(g.get("game_id")): g for g in body if isinstance(g, dict)}
            return {str(k): {"headline": _s(v.get("headline")), "story": _s(v.get("story"))}
                    for k, v in body.items() if isinstance(v, dict)}

        outcomes = generate(prompt, llm_items, extract, complete=complete, timeout=timeout)
        _record(report, outcomes)
        for gid, o in outcomes.items():
            result[gid] = {"headline": o.fields["headline"], "story": o.fields["story"],
                           "source": o.source}
    return {gid: result[gid] for gid in games}


def _safe_game_template(f: dict) -> dict[str, str]:
    try:
        return templates.game(f)
    except Exception:  # noqa: BLE001 - facts are untrusted; the fallback must not raise
        home, away = _d(f.get("home")), _d(f.get("away"))
        h, a = templates.nick(home.get("team")), templates.nick(away.get("team"))
        score = (f" {away['score']}-{home['score']}" if isinstance(home.get("score"), int)
                 and isinstance(away.get("score"), int) else "")
        return {"headline": f"{a} at {h}", "story": f"The {a} played the {h}{score}."}


def _s(x: Any) -> str | None:
    return x.strip() if isinstance(x, str) and x.strip() else None


# ---------------------------------------------------------------------------------------
# 2. pulse
# ---------------------------------------------------------------------------------------

_MAX_TEAMS, _MAX_PLAYERS = 8, 12


def _streak_len(code: str | None) -> int:
    m = re.match(r"[WL](\d+)", code or "")
    return int(m.group(1)) if m else 0


def compact_team(entry: dict, flag: str | None = None) -> dict:
    t = entry.get("team")
    e = {
        "team": t, "nickname": templates.nick(t), "flag": flag or entry.get("flag"),
        "record": entry.get("record"), "last10": entry.get("last10"), "streak": entry.get("streak"),
        "power_rank": entry.get("power_rank"), "net": entry.get("net"), "off": entry.get("off"),
        "def": entry.get("def"),
        "trend": _drop_none({k: (entry.get("trend") or {}).get(k) for k in ("delta", "baseline")}),
        "opponent_difficulty_last10": (_drop_none({"rank_hardest_is_1_of_30": (entry.get("sos_last10") or {}).get("rank"),
                                                   "opp_adj_net": (entry.get("sos_last10") or {}).get("opp_adj_net")})
                                       if entry.get("sos_last10") else None),
        "reasons": [r.get("detail") for r in entry.get("reasons") or [] if r.get("kind") != "net_trend"
                    and r.get("detail")] or None,
        "key_absences": [_drop_none({"player": a["player"], "games_missed": a.get("team_games_missed"),
                                     "mpg_before": a.get("mpg_before")})
                         for a in entry.get("key_absences") or []] or None,
        "streak_detail": entry.get("streak_detail"),
        "rank": entry.get("rank"), "rank_7d_ago": entry.get("rank_7d_ago"),
        "change_7d": entry.get("change_7d"),
    }
    return _drop_none(e)


def _team_entities(lf: dict, only: set[str] | None = None) -> list[dict]:
    """Teams worth a blurb. ``only`` (tricodes) restricts to those teams, uncapped, and never adds
    teams just because they have a streak or moved in the rankings."""
    teams = lf.get("teams") or {}
    ents: dict[str, dict] = {}
    for flag in ("rising", "sliding"):
        for t in teams.get(flag) or []:
            ents[t["team"]] = compact_team(t, flag)
    for s in sorted(teams.get("streaks") or [], key=lambda s: -_streak_len(s.get("streak")))[:3 if only is None else None]:
        if only is not None and s["team"] not in ents:
            continue
        ents.setdefault(s["team"], {"team": s["team"], "nickname": templates.nick(s["team"])})
        ents[s["team"]].update(_drop_none({"record": s.get("record"), "streak": s.get("streak"),
                                           "streak_detail": s.get("detail")}))
    for m in sorted(teams.get("power_movers_7d") or [], key=lambda m: -abs(m.get("change_7d") or 0))[:2 if only is None else None]:
        if only is not None and m["team"] not in ents:
            continue
        if m.get("change_7d"):
            ents.setdefault(m["team"], {"team": m["team"], "nickname": templates.nick(m["team"])})
            ents[m["team"]].update(_drop_none({"rank": m.get("rank"), "rank_7d_ago": m.get("rank_7d_ago"),
                                               "change_7d": m.get("change_7d")}))
    if only is not None:
        return [e for k, e in ents.items() if k in only]
    if not ents:
        for t in (teams.get("power_top10") or [])[:1]:
            ents[t["team"]] = _drop_none({"team": t["team"], "nickname": templates.nick(t["team"]),
                                          "record": t.get("record"), "power_rank": t.get("rank")})
    return list(ents.values())[:_MAX_TEAMS]


def _stat_block(b: dict | None) -> dict | None:
    if not b:
        return None
    return _drop_none({k: b.get(k) for k in ("games", "min", "pts", "reb", "ast", "ts_pct")})


def _player_entities(lf: dict, only: set[str] | None = None) -> list[dict]:
    """Players worth a blurb (keys are str(player_id)). ``only`` restricts and uncaps."""
    pl = lf.get("players") or {}
    caps = only is None
    ents: dict[str, dict] = {}

    def lim(n: int) -> int | None:
        return n if caps else None

    def ent(p: dict) -> dict | None:
        key = str(p.get("player_id") or p["player"])
        if only is not None and key not in only:
            return None
        return ents.setdefault(key, {"key": key, "player": p["player"], "team": p.get("team"),
                                     "tags": []})
    for tag in ("hot", "cold"):
        for p in (pl.get(tag) or [])[:lim(3)]:
            e = ent(p)
            if e is None:
                continue
            e["tags"].append(tag)
            e.update(_drop_none({
                "window": p.get("window"), "recent": _stat_block(p.get("recent")),
                "season": _stat_block(p.get("season")), "impact_recent": p.get("impact_recent"),
                "impact_baseline": p.get("impact_baseline")}))
    pri = {"starter_minutes": 0, "new_in_rotation": 1, "lost_starter_minutes": 0, "new_team": 2,
           "out_of_rotation": 2, "new_arrival": 2, "minutes_up": 3, "minutes_down": 3,
           "usage_up": 4, "usage_down": 4}
    rcs = sorted(pl.get("role_changes") or [], key=lambda r: min(pri.get(k, 9) for k in r["kinds"]))
    for p in rcs[:lim(3)]:
        e = ent(p)
        if e is not None:
            e["tags"].append("role_change")
            e["role_details"] = p.get("details")
    for p in sorted(pl.get("streaks") or [], key=lambda s: -(s.get("games") or 0))[:lim(1)]:
        e = ent(p)
        if e is not None:
            e["tags"].append("streak")
            e["streak_detail"] = p.get("detail")
    for p in (pl.get("career_highs_last7d") or [])[:lim(2)]:
        e = ent(p)
        if e is not None:
            e["tags"].append("career_high")
            e["career_high"] = _drop_none({"stat": p.get("stat"), "value": p.get("value"),
                                           "previous_high": p.get("previous_high")})
    out = list(ents.values())
    return out[:_MAX_PLAYERS] if caps else out


def _method_block(lf: dict) -> dict:
    m = lf.get("method") or {}
    fv, hv = m.get("flag_validity") or {}, m.get("hot_cold_validity") or {}
    return _drop_none({
        "team_flag_validity": _drop_none({
            "rising": fv.get("rising"), "sliding": fv.get("sliding"),
            "base_rate_pct": fv.get("base_rate_pct"), "reading": fv.get("reading")}),
        "player_hot_cold_validity": _drop_none({
            "hot": hv.get("hot"), "cold": hv.get("cold"), "base_rate_pct": hv.get("base_rate_pct")}),
        "as_of": lf.get("as_of"), "games_per_team_median": lf.get("games_per_team_median"),
        "early_season": lf.get("early_season") or None, "notes": lf.get("notes") or None,
    })


def write_pulse(league_facts: dict, *, complete: Complete | None = None,
                report: dict | None = None, timeout: int = 240) -> dict:
    """-> {"headline", "teams": {tricode: text}, "players": {key: text}, "sources": {...}}"""
    return _pulse(_team_entities(league_facts), _player_entities(league_facts),
                  _method_block(league_facts), complete=complete, report=report, timeout=timeout)


def _pulse(teams: list[dict], players: list[dict], method: dict, *, complete: Complete | None,
           report: dict | None, timeout: int, want_headline: bool = True) -> dict:
    items: list[Item] = []
    for e in teams:
        items.append(Item(f"team:{e['team']}", {"text": "blurb"}, {"entity": e, "method": method},
                          (lambda e=e: {"text": templates.team_blurb(e)})))
    for e in players:
        items.append(Item(f"player:{e['key']}", {"text": "blurb"}, {"entity": e, "method": method},
                          (lambda e=e: {"text": templates.player_blurb(e)})))
    if want_headline:
        items.append(Item("headline", {"text": "pulse_headline"},
                          {"teams": teams, "players": players, "method": method},
                          lambda: {"text": templates.pulse_headline(teams, players)}))
    result = {"headline": None, "teams": {}, "players": {}, "sources": {}}
    if not (teams or players):
        if want_headline:
            result["headline"] = templates.pulse_headline([], [])
            result["sources"]["headline"] = "template"
        return result
    prompt = render("pulse", method=_dumps(method), teams=_dumps(teams) if teams else "(none)",
                    players=_dumps(players) if players else "(none)")

    def extract(parsed: Any) -> dict[str, dict[str, str]]:
        d: dict[str, dict[str, str]] = {}
        if not isinstance(parsed, dict):
            return d
        if _s(parsed.get("headline")):
            d["headline"] = {"text": _s(parsed["headline"])}
        for sect, prefix in (("teams", "team:"), ("players", "player:")):
            body = parsed.get(sect)
            if isinstance(body, dict):
                for k, v in body.items():
                    if _s(v):
                        d[prefix + str(k)] = {"text": _s(v)}
        return d

    outcomes = generate(prompt, items, extract, complete=complete, timeout=timeout)
    _record(report, outcomes)
    for k, o in outcomes.items():
        text = o.fields["text"]
        if k == "headline":
            result["headline"] = text
        else:
            sect, key = k.split(":", 1)
            result["teams" if sect == "team" else "players"][key] = text
        result["sources"][k] = o.source
    return result


# ---------------------------------------------------------------------------------------
# 3. tonight
# ---------------------------------------------------------------------------------------

def _team_code(x: Any) -> str | None:
    if isinstance(x, dict):
        for k in ("team", "tricode", "abbr", "name"):
            if x.get(k):
                return _team_code(x[k])
        return None
    if not isinstance(x, str):
        return None
    if x in _TRICODES:
        return x
    for t in TEAMS:
        if x in (t.name, f"{t.city} {t.name}", t.city):
            return t.tricode
    return x


def _first(p: dict, *keys: str) -> Any:
    for k in keys:
        if p.get(k) is not None:
            return p[k]
    return None


def _text_of(x: Any) -> str | None:
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, dict):
        if x.get("player"):
            team = f" ({templates.nick(_team_code(x['team']))})" if x.get("team") else ""
            st = f": {x['status']}" if x.get("status") else ""
            return f"{x['player']}{team}{st}"
        for k in ("detail", "text", "reason", "description", "note"):
            if isinstance(x.get(k), str) and x[k].strip():
                return x[k].strip()
    return None


def normalize_prediction(p: dict) -> dict:
    """Flat prediction dicts and docs/BRIEF.md ``tonight.games[i]`` objects both work."""
    pred = p.get("prediction") if isinstance(p.get("prediction"), dict) else {}
    mkt = p.get("market") if isinstance(p.get("market"), dict) else {}
    edge = p.get("edge")
    src = {**pred, **{k: v for k, v in p.items() if v is not None}}
    home = _team_code(_first(p, "home", "home_team", "home_tricode"))
    away = _team_code(_first(p, "away", "away_team", "away_tricode"))
    prob = _first(src, "home_win_prob", "p_home", "win_prob", "home_prob")
    if prob is not None and prob > 1:
        prob = prob / 100.0
    margin = _first(src, "home_margin", "margin", "proj_margin", "projected_margin")
    spread = _first(src, "market_home_spread", "market_spread", "spread", "home_spread")
    if spread is None:
        spread = mkt.get("home_spread")
    gap = None
    if isinstance(edge, dict):
        gap = edge.get("margin")
    if gap is None and margin is not None and spread is not None:
        gap = margin + spread
    reasons = p.get("reasons") if p.get("reasons") is not None else p.get("factors")
    has_inj_list = bool(p.get("injuries"))
    reasons = [r for r in (reasons or []) if not (isinstance(r, dict) and (
        r.get("kind") == "form" or (r.get("kind") == "injury" and has_inj_list)))]
    inj = p.get("injuries") or []
    if isinstance(inj, dict):
        inj = list(inj.get("home") or [])[:3] + list(inj.get("away") or [])[:3]
    up = p.get("upset") if isinstance(p.get("upset"), dict) else {}

    def form(side: Any) -> dict | None:
        if not isinstance(side, dict):
            return None
        return _drop_none({k: side.get(k) for k in ("record", "last10", "streak", "power_rank")}) or None

    out = {
        "game_id": str(p.get("game_id")),
        "home": {"team": home, "nickname": templates.nick(home),
                 **(form(p.get("home")) or {})},
        "away": {"team": away, "nickname": templates.nick(away),
                 **(form(p.get("away")) or {})},
        "home_win_prob": round(prob, 3) if prob is not None else None,
        "away_win_prob": round(1 - prob, 3) if prob is not None else None,
        "projected_home_margin": margin,
        "projected_total": _first(src, "total", "proj_total", "projected_total"),
        "market_home_spread": spread,
        "market_total": _first(p, "market_total") if _first(p, "market_total") is not None else mkt.get("total"),
        "our_number_minus_market_pts": round(gap, 1) if gap is not None else None,
        "edge": edge if isinstance(edge, (int, float)) else None,
        "confidence": src.get("confidence"),
        "model_prices_injuries": p.get("model_uses_injuries"),
        "reasons": [t for t in (_text_of(r) for r in reasons[:8]) if t] or None,
        "injuries": [t for t in (_text_of(r) for r in inj) if t] or None,
        "upset_watch": _drop_none({
            "favorite": up.get("favorite"), "underdog": up.get("underdog"),
            "model_prob": up.get("model_prob"), "market_prob": up.get("market_prob"),
            "model_picks_underdog": up.get("model_picks_underdog")}) if up.get("flag") else None,
    }
    return _drop_none(out)


def _tonight_template_input(n: dict) -> dict:
    return {"home": n["home"]["team"], "away": n["away"]["team"],
            "home_win_prob": n.get("home_win_prob"), "home_margin": n.get("projected_home_margin"),
            "market_home_spread": n.get("market_home_spread"), "reasons": n.get("reasons")}


def write_tonight(predictions: list[dict], *, complete: Complete | None = None,
                  report: dict | None = None, timeout: int = 180) -> dict[str, str]:
    """-> {game_id: preview text (<= 2 sentences)}"""
    norm = [normalize_prediction(p) for p in predictions or []]
    if not norm:
        return {}
    items = [Item(n["game_id"], {"text": "tonight"}, n,
                  (lambda n=n: {"text": templates.tonight(_tonight_template_input(n))}))
             for n in norm]
    prompt = render("tonight", facts=_dumps(norm))

    def extract(parsed: Any) -> dict[str, dict[str, str]]:
        body = parsed.get("games", parsed) if isinstance(parsed, dict) else {}
        if isinstance(body, list):
            body = {str(g.get("game_id")): g.get("text") or g.get("preview") for g in body
                    if isinstance(g, dict)}
        return {str(k): {"text": _s(v if isinstance(v, str) else (v or {}).get("text"))}
                for k, v in body.items() if v}

    outcomes = generate(prompt, items, extract, complete=complete, timeout=timeout)
    _record(report, outcomes)
    return {n["game_id"]: outcomes[n["game_id"]].fields["text"] for n in norm}


# ---------------------------------------------------------------------------------------
# 4. Brief headline
# ---------------------------------------------------------------------------------------

def write_brief_headline(brief_summary: dict, *, complete: Complete | None = None,
                         report: dict | None = None, timeout: int = 120) -> dict:
    """-> {"headline", "source"}. The summary is the whole universe of allowed facts."""
    item = Item("headline", {"text": "brief_headline"}, brief_summary,
                lambda: {"text": templates.brief_headline(brief_summary)})
    prompt = render("headline", facts=_dumps(brief_summary))

    def extract(parsed: Any) -> dict[str, dict[str, str]]:
        return {"headline": {"text": _s(parsed.get("headline"))}} if isinstance(parsed, dict) else {}

    o = generate(prompt, [item], extract, complete=complete, timeout=timeout)["headline"]
    _record(report, {"headline": o})
    return {"headline": o.fields["text"], "source": o.source}


# ---------------------------------------------------------------------------------------
# 5. fill(brief)   (schema: docs/BRIEF.md)
# ---------------------------------------------------------------------------------------

def _slot_empty(container: Any, key: str) -> bool:
    """True when ``container[key]`` exists and is null/blank (we never create slots)."""
    return isinstance(container, dict) and key in container and not (
        isinstance(container[key], str) and container[key].strip())


def _dicts(x: Any) -> list[dict]:
    return [i for i in x if isinstance(i, dict)] if isinstance(x, list) else []


def _game_context(card: dict) -> dict | None:
    pred = card.get("prediction") if isinstance(card.get("prediction"), dict) else None
    if not pred or pred.get("pick") is None:
        return None
    mkt = card.get("market") if isinstance(card.get("market"), dict) else {}
    return {"our_pregame_model": _drop_none({
        "team": pred.get("pick"), "win_prob": pred.get("pick_prob"),
        "confidence": pred.get("confidence"), "correct": card.get("correct"),
        "market_favorite": mkt.get("favorite"), "market_favorite_won": mkt.get("favorite_won")})}


def fill(brief: dict, *, complete: Complete | None = None) -> dict:
    """Return a copy of ``brief`` with the writer's text slots filled (docs/BRIEF.md).

    Filled: ``headline.text``; ``last_night.games[i].text.{headline, story}`` (needs ``facts``);
    ``team_pulse.{rising,sliding,...}[i].text`` and ``team_pulse.text`` (the pulse headline);
    ``player_pulse.{hot,cold,...}[i].text``; ``tonight.games[i].text.preview``.
    Left null: ``last_night.text``, ``player_pulse.text``, ``injury_watch.text``,
    ``tonight.text``, ``scorecard.text`` (no writer for them yet). Slots that already hold text
    are never overwritten, and slots are never created. Per-item outcomes (llm_ok /
    llm_retry_ok / template) are recorded in ``meta.writer`` = ``{"counts", "outcomes"}``.
    """
    b = copy.deepcopy(brief)
    if complete is None and os.environ.get("COURTSIDE_NO_WRITER"):
        return b                       # the pipeline's off switch (tests, dry runs)
    counts: Counter = Counter()
    outcomes: dict[str, str] = {}

    def tally(key: str, source: str) -> None:
        counts[source] += 1
        outcomes[key] = source

    # -- last night ---------------------------------------------------------------------
    cards = _dicts((b.get("last_night") or {}).get("games"))
    todo = [c for c in cards if isinstance(c.get("facts"), dict) and isinstance(c.get("text"), dict)
            and (_slot_empty(c["text"], "headline") or _slot_empty(c["text"], "story"))]
    if todo:
        ctx = {str(c["facts"]["game_id"]): g for c in todo if (g := _game_context(c))}
        stories = write_game_stories([c["facts"] for c in todo], context=ctx, complete=complete)
        for c in todo:
            gid = str(c["facts"]["game_id"])
            s = stories.get(gid)
            if not s:
                continue
            for k in ("headline", "story"):
                if _slot_empty(c["text"], k):
                    c["text"][k] = s[k]
            tally(f"game:{gid}", s["source"])

    # -- pulse --------------------------------------------------------------------------
    tp = b.get("team_pulse") if isinstance(b.get("team_pulse"), dict) else {}
    pp = b.get("player_pulse") if isinstance(b.get("player_pulse"), dict) else {}
    team_slots: dict[str, list[dict]] = {}
    for val in tp.values():
        for e in _dicts(val):
            if e.get("team") and _slot_empty(e, "text"):
                team_slots.setdefault(e["team"], []).append(e)
    player_slots: dict[str, list[dict]] = {}
    for val in pp.values():
        for e in _dicts(val):
            if e.get("player") and _slot_empty(e, "text"):
                player_slots.setdefault(str(e.get("player_id") or e["player"]), []).append(e)
    want_head = tp.get("available", True) is not False and _slot_empty(tp, "text")
    if team_slots or player_slots or want_head:
        method = {**(tp.get("method") or {}), **(pp.get("method") or {})}
        lf = {
            "as_of": tp.get("as_of") or pp.get("as_of"), "method": method,
            "games_per_team_median": tp.get("games_per_team_median"),
            "early_season": tp.get("early_season") or pp.get("early_season"),
            "notes": tp.get("notes") or pp.get("notes"),
            "teams": {k: tp.get(k) for k in ("rising", "sliding", "streaks", "power_movers_7d", "power_top10")},
            "players": {k: pp.get(k) for k in ("hot", "cold", "role_changes", "streaks", "career_highs_last7d")},
        }
        if want_head and not (team_slots or player_slots):
            teams, players = _team_entities(lf), _player_entities(lf)   # headline only
        else:
            teams, players = _team_entities(lf, set(team_slots)), _player_entities(lf, set(player_slots))
        if teams or players:
            res = _pulse(teams, players, _method_block(lf), complete=complete, report=None,
                         timeout=240, want_headline=want_head)
            for code, text in res["teams"].items():
                for e in team_slots.get(code, []):
                    e["text"] = text
                tally(f"team:{code}", res["sources"][f"team:{code}"])
            for key, text in res["players"].items():
                for e in player_slots.get(key, []):
                    e["text"] = text
                tally(f"player:{key}", res["sources"][f"player:{key}"])
            if want_head and res.get("headline"):
                tp["text"] = res["headline"]
                tally("pulse:headline", res["sources"]["headline"])

    # -- tonight ------------------------------------------------------------------------
    games = [g for g in _dicts((b.get("tonight") or {}).get("games"))
             if isinstance(g.get("text"), dict) and _slot_empty(g["text"], "preview")]
    if games:
        rep: dict = {}
        uses_inj = (b.get("meta") or {}).get("model_uses_injuries") if isinstance(b.get("meta"), dict) else None
        texts = write_tonight([{**g, "model_uses_injuries": uses_inj} for g in games],
                              complete=complete, report=rep)
        for g in games:
            gid = str(g.get("game_id"))
            if gid in texts:
                g["text"]["preview"] = texts[gid]
                tally(f"tonight:{gid}", rep[gid]["source"])

    # -- Brief headline (built from the sections written above) --------------------------
    if isinstance(b.get("headline"), dict) and _slot_empty(b["headline"], "text"):
        r = write_brief_headline(brief_summary(b), complete=complete)
        b["headline"]["text"] = r["headline"]
        tally("brief:headline", r["source"])

    if counts:
        meta = b.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["writer"] = {"counts": dict(counts), "outcomes": outcomes}
    return b


def brief_summary(b: dict) -> dict:
    """Compact facts for the Brief headline, from the (already written) sections of ``b``."""
    games = []
    for c in _dicts((b.get("last_night") or {}).get("games")):
        res = c.get("result") if isinstance(c.get("result"), dict) else None
        if not res or not c.get("final"):
            continue
        f = c.get("facts") or {}
        cb = f.get("comeback") or {}
        text = c.get("text") if isinstance(c.get("text"), dict) else {}
        games.append(_drop_none({
            "game_id": str(c.get("game_id")), "headline": text.get("headline"),
            "home": (c.get("home") or {}).get("team"), "away": (c.get("away") or {}).get("team"),
            "home_score": res.get("home_score"), "away_score": res.get("away_score"),
            "winner": res.get("winner"), "margin": res.get("margin"),
            "overtimes": res.get("overtimes") or None,
            "comeback_deficit": cb.get("deficit") if cb.get("deficit", 0) >= 10 else None}))
    tonight = _dicts((b.get("tonight") or {}).get("games"))
    upsets = [g for g in tonight if g.get("upset_watch")]
    meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}
    tp = b.get("team_pulse") if isinstance(b.get("team_pulse"), dict) else {}
    return _drop_none({
        "date": meta.get("day"),
        "games": games,
        "pulse": _drop_none({
            "headline": tp.get("text") if isinstance(tp.get("text"), str) else None,
            "rising": [e["team"] for e in _dicts(tp.get("rising"))] or None,
            "sliding": [e["team"] for e in _dicts(tp.get("sliding"))] or None}),
        "tonight_games": len(tonight) or None,
        "tonight_upset_watch": [{"home": (g.get("home") or {}).get("team"),
                                 "away": (g.get("away") or {}).get("team")} for g in upsets] or None,
    })
