"""Offline stand-ins for the pre-tip archive sources (props, referees, Action Network, DFS).

`stub_pretip_sources(monkeypatch)` patches the fetch functions of the four source modules with
the saved fixtures and pins `collect._utcnow` to a pre-tip moment of opening night, so the
steps behave the same whenever the suite runs. Returns a `Calls` counter.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from courtside import collect
from courtside.sources import action_network, dfs, espn_props, referees

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)   # 10:00 ET, 5 h before first tip


def J(rel: str):
    return json.loads((FIX / rel).read_text())


def referee_page_for(day_text: str = "October 20, 2026") -> str:
    """The plain-markup filled fixture re-dated and re-teamed to opening night's three games."""
    html = (FIX / "referees/filled_plain_20260420.html").read_text()
    html = html.replace("April 20, 2026", day_text)
    for old, new in (("Atlanta @ New York", "Boston @ Detroit"),
                     ("Toronto @ Cleveland", "Philadelphia @ New York"),
                     ("Minnesota @ Denver", "Oklahoma City @ San Antonio")):
        html = html.replace(old, new)
    return html


def stub_pretip_sources(monkeypatch, *, referee_html: str | None = None) -> Counter:
    calls: Counter = Counter()

    def count(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapper

    monkeypatch.setattr(collect, "_utcnow", lambda: NOW)

    # ESPN props
    monkeypatch.setattr(espn_props, "fetch_providers",
                        count("props.providers", lambda eid: J("espn_props/odds_list.json")))
    monkeypatch.setattr(espn_props, "fetch_propbets_page", count(
        "props.page", lambda eid, pid, page=1, limit=1000:
        J(f"espn_props/propbets_page{page}.json")))
    rosters = {"BOS": "roster_BOS", "DET": "roster_DET"}
    monkeypatch.setattr(espn_props, "fetch_roster", count(
        "props.roster", lambda abbr: J(f"espn_props/{rosters.get(abbr, 'roster_BOS')}.json")))
    monkeypatch.setattr(espn_props, "fetch_athlete", count(
        "props.athlete", lambda aid, season=None: {"displayName": f"Athlete {aid}"}))

    # referees
    page = (referee_html if referee_html is not None else referee_page_for()).encode()
    monkeypatch.setattr(referees, "fetch_page", count("referees.page", lambda *a, **k: page))

    # Action Network
    monkeypatch.setattr(action_network, "fetch_scoreboard", count(
        "an.scoreboard", lambda day, **k: copy.deepcopy(J("action_network/scoreboard_20261020.json"))))

    # DraftKings
    monkeypatch.setattr(dfs, "fetch_lobby", count("dfs.lobby", lambda: J("dfs/lobby.json")))
    monkeypatch.setattr(dfs, "fetch_draftables", count(
        "dfs.draftables", lambda sid: J("dfs/draftables_classic.json" if int(sid) == 154575
                                        else "dfs/draftables_showdown.json")))
    monkeypatch.setattr(dfs, "fetch_rules", count(
        "dfs.rules", lambda gt: J(f"dfs/rules_{int(gt)}.json")))
    return calls
