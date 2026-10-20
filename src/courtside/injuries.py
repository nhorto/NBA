"""Injury snapshots -> "who is out right now" and "what changed since yesterday".

`injury_snapshots` is append-only: ESPN writes one batch per pregame run (every row of a batch
shares `captured_at`), the official NBA report writes one batch per published PDF (keyed by
the publication time). A *batch* is therefore (source, captured_at).

* `current(conn, at)`   union of the newest ESPN batch and the newest official batch at/before
  `at` (the official row wins for a player on both), keeping only availability-relevant
  statuses.
* `diff(conn, at)`      newest batch vs. the batch ~24 h earlier, per source (official wins),
  split into new absences / returns / status changes. Official reports only list the teams
  playing that day, so for them only teams present in *both* batches are compared.

Pricing injuries (the prediction models), same module so the Brief and the model agree:

* `resolve(conn, at)`             per team, which snapshot speaks for it: the newest listing
  batch, the official report winning ties and anything it lists while it is the newer one.
* `play_probability(status)`      Out 0, Doubtful 0.1, Questionable 0.5, Day-To-Day 0.6,
  Probable 0.9, Available / not listed 1 (see `PLAY_PROB` for the reasoning).
* `recent_roster` / `match_player`  names (ESPN "LeBron James", official "Jackson Jr., Jaren",
  accents, suffixes, nicknames) -> `player_games.player_id`, restricted to the team's recent
  roster so fuzzy matching cannot pick the wrong player.
* `team_availability(...)`        {player_id: P(plays)} + the per-name match record.
* `match_report(conn, ...)`       match rates over archived snapshots (+ the unmatched CSV).
"""

from __future__ import annotations

import csv
import difflib
import re
import sqlite3
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

OFFICIAL = "nba_official"
ESPN = "espn"

# Normalised status -> severity. Anything not listed (Probable, Available, "") is healthy.
SEVERITY = {"out": 3, "doubtful": 2, "questionable": 1, "day-to-day": 1}
LABEL = {3: "Out", 2: "Doubtful", 1: "Questionable"}
OFFICIAL_MAX_AGE_H = 48        # an older official batch is ignored for "current"
BASELINE_TARGET_H = 24
BASELINE_MIN_GAP_H = 12        # the comparison batch must be at least this much older

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str | None) -> str:
    """'Jaren Jackson Jr.' / 'Jackson Jr., Jaren' / 'Nikola Jokić' -> 'jaren jackson'."""
    if not name:
        return ""
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def severity(status: str | None) -> int:
    return SEVERITY.get((status or "").strip().lower(), 0)


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def batches(conn: sqlite3.Connection, source: str, at: str | None = None) -> list[str]:
    """captured_at of every batch of `source` (ascending), optionally only those <= `at`."""
    sql = "SELECT DISTINCT captured_at FROM injury_snapshots WHERE source=?"
    args: list = [source]
    if at is not None:
        sql += " AND captured_at<=?"
        args.append(at)
    return [r[0] for r in conn.execute(sql + " ORDER BY captured_at", args)]


def _batch_rows(conn: sqlite3.Connection, source: str, captured_at: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT team, player_name, player_id, status, reason FROM injury_snapshots "
        "WHERE source=? AND captured_at=?", (source, captured_at))]


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(team, normalised name) -> row; the most severe row wins on a duplicate."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if not r.get("player_name"):
            continue
        k = (r["team"], norm_name(r["player_name"]))
        if k not in out or severity(r["status"]) > severity(out[k]["status"]):
            out[k] = r
    return out


def _entry(r: dict, source: str, captured_at: str) -> dict:
    sev = severity(r["status"])
    return {"team": r["team"], "player": r["player_name"], "player_id": r.get("player_id"),
            "status": LABEL.get(sev, r["status"]), "severity": sev,
            "reason": (r.get("reason") or "").split("|")[0].strip() or None,
            "source": source, "captured_at": captured_at}


ESPN_MAX_AGE_H = 36            # pricing only: an older ESPN batch is not "the current state"


def resolve(conn: sqlite3.Connection, at: str, *, espn_max_age_h: float | None = None) -> dict:
    """Which snapshot speaks for each team at `at` (ISO UTC).

    Per team the newest batch that *lists* the team wins and the official report wins ties, so
    "official over ESPN when official is the newer one"; a team only one source lists uses that
    source. An official batch older than OFFICIAL_MAX_AGE_H (and, if given, an ESPN batch older
    than `espn_max_age_h`) is ignored.

    Returns {"teams": {team: {"source", "captured_at", "rows": [raw snapshot rows]}},
    "batches": {source: captured_at}, "fresh": bool}. `fresh` says at least one usable batch
    exists: a team missing from `teams` is then healthy as far as the reports go."""
    used: dict[str, str] = {}
    rows_by: dict[str, dict[str, list[dict]]] = {}
    for src, max_age in ((ESPN, espn_max_age_h), (OFFICIAL, OFFICIAL_MAX_AGE_H)):
        b = batches(conn, src, at)
        if not b:
            continue
        if max_age is not None and "T" in at and _ts(at) - _ts(b[-1]) > timedelta(hours=max_age):
            continue
        used[src] = b[-1]
        by: dict[str, list[dict]] = {}
        for r in _batch_rows(conn, src, b[-1]):
            if r.get("player_name"):
                by.setdefault(r["team"], []).append(r)
        rows_by[src] = by
    teams: dict[str, dict] = {}
    for team in set().union(*(set(v) for v in rows_by.values())) if rows_by else ():
        cands = [(used[src], 1 if src == OFFICIAL else 0, src) for src in rows_by
                 if team in rows_by[src]]
        cap, _, src = max(cands)
        teams[team] = {"source": src, "captured_at": cap, "rows": rows_by[src][team]}
    return {"teams": teams, "batches": used, "fresh": bool(used)}


def current(conn: sqlite3.Connection, at: str, teams: set[str] | None = None) -> dict:
    """Relevant (Out/Doubtful/Questionable) injuries known at or before `at`.

    Uses `resolve`: per team the newest listing batch, official winning ties.
    Returns {"by_team": {team: [entry,...]}, "batches": {source: captured_at},
    "latest_captured_at": str|None}. Entries are ordered most severe first."""
    res = resolve(conn, at)
    by_team: dict[str, list[dict]] = {}
    for team, t in res["teams"].items():
        if teams is not None and team not in teams:
            continue
        for k, r in _index(t["rows"]).items():
            e = _entry(r, t["source"], t["captured_at"])
            if e["severity"] >= 1:
                by_team.setdefault(team, []).append(e)
    for lst in by_team.values():
        lst.sort(key=lambda e: (-e["severity"], e["player"]))
    used = res["batches"]
    return {"by_team": by_team, "batches": used,
            "latest_captured_at": max(used.values()) if used else None}


def _pick_baseline(cands: list[str], latest: str) -> str | None:
    """The batch closest to latest-24h among those at least BASELINE_MIN_GAP_H older."""
    lt = _ts(latest)
    ok = [c for c in cands if lt - _ts(c) >= timedelta(hours=BASELINE_MIN_GAP_H)]
    if not ok:
        return None
    target = lt - timedelta(hours=BASELINE_TARGET_H)
    return min(ok, key=lambda c: abs(_ts(c) - target))


def _diff_states(old: dict, new: dict, comparable_teams: set[str] | None) -> list[dict]:
    changes = []
    for k in set(old) | set(new):
        team, _ = k
        if comparable_teams is not None and team not in comparable_teams:
            continue
        o, n = old.get(k), new.get(k)
        so = severity(o["status"]) if o else 0
        sn = severity(n["status"]) if n else 0
        if so == sn:
            continue
        ref = n or o
        if so == 0:
            kind = "new_absence"
        elif sn == 0:
            kind = "return"
        else:
            kind = "status_change"
        changes.append({
            "kind": kind, "team": team, "player": ref["player"], "player_id": ref.get("player_id"),
            "from_status": LABEL.get(so, "Healthy") if o else None,
            "to_status": LABEL.get(sn, "Healthy") if n else "Healthy",
            "direction": "worse" if sn > so else "better",
            "reason": (n or o).get("reason"),
            "source": ref["source"]})
    return changes


def diff(conn: sqlite3.Connection, at: str) -> dict:
    """Changes between the newest batch at/before `at` and the batch ~24 h earlier.

    {"available": bool, "latest": ts|None, "baseline": ts|None,
     "new_absences": [...], "returns": [...], "status_changes": [...], "sources": [...]}"""
    out: dict = {"available": False, "latest": None, "baseline": None,
                 "new_absences": [], "returns": [], "status_changes": [], "sources": []}
    per_source: dict[str, list[dict]] = {}
    spans = []
    for source in (OFFICIAL, ESPN):
        b = batches(conn, source, at)
        if not b:
            continue
        latest = b[-1]
        if source == OFFICIAL and _ts(at) - _ts(latest) > timedelta(hours=OFFICIAL_MAX_AGE_H):
            continue
        base = _pick_baseline(b[:-1], latest)
        if base is None:
            continue
        def state(cap):
            rows = _batch_rows(conn, source, cap)
            idx = _index(rows)
            return {k: _entry(r, source, cap) for k, r in idx.items()}, {r["team"] for r in rows}
        new_s, new_t = state(latest)
        old_s, old_t = state(base)
        comparable = (new_t & old_t) if source == OFFICIAL else None
        per_source[source] = _diff_states(old_s, new_s, comparable)
        spans.append((source, latest, base))
    if not per_source:
        return out
    # Official wins per team where it produced a diff; ESPN fills the remaining teams.
    chosen: list[dict] = []
    covered: set[str] = set()
    if OFFICIAL in per_source:
        chosen += per_source[OFFICIAL]
        # teams the official comparison covered (even with zero changes) are not re-reported
        sp = next(s for s in spans if s[0] == OFFICIAL)
        covered = {r["team"] for r in _batch_rows(conn, OFFICIAL, sp[1])} & \
                  {r["team"] for r in _batch_rows(conn, OFFICIAL, sp[2])}
    if ESPN in per_source:
        chosen += [c for c in per_source[ESPN] if c["team"] not in covered]
    out["available"] = True
    out["sources"] = [s[0] for s in spans]
    out["latest"] = max(s[1] for s in spans)
    out["baseline"] = min(s[2] for s in spans)
    order = {"new_absences": "new_absence", "returns": "return", "status_changes": "status_change"}
    for key, kind in order.items():
        out[key] = sorted((c for c in chosen if c["kind"] == kind),
                          key=lambda c: (c["team"], c["player"]))
    return out


# ====================================================================================
# Pricing injuries: status -> P(plays), names -> player_games.player_id
# ====================================================================================

# P(plays) by report status. Out/Doubtful/Questionable/Probable are the report's own words.
# "Day-To-Day" is ESPN's catch-all (it covers everything from a sore ankle to a multi-week
# knee), kept at the production value of docs/MODEL.md. Doubtful is 0.1 (the historical share
# of Doubtful players who suit up is about one in ten); Questionable 0.5 is the value the
# rung-2 ablation tested (ps_elo_sched_soft: status-change players at 0.5 cost only 0.0014
# log-loss against the oracle, and flipping a quarter of them wrongly 0.0030, so the exact
# constants hardly matter). A name that is not on the report at all is 1.
PLAY_PROB = {
    "out": 0.0, "out for season": 0.0, "suspended": 0.0, "inactive": 0.0,
    "doubtful": 0.10,
    "questionable": 0.5, "game time decision": 0.5, "gtd": 0.5,
    "day-to-day": 0.6, "day to day": 0.6,
    "probable": 0.9,
    "available": 1.0, "healthy": 1.0, "active": 1.0,
}
UNKNOWN_STATUS_P = 0.5        # a status we have never seen: treat like Questionable


def play_probability(status: str | None) -> float:
    """P(the player suits up) for a report status. Empty -> 1, unknown wording -> 0.5."""
    s = (status or "").strip().lower()
    if not s:
        return 1.0
    return PLAY_PROB.get(s, UNKNOWN_STATUS_P)


# --- names -------------------------------------------------------------------------------

# First names that are the same person under different spellings (token sets, after
# norm_name). Plain prefixes (Cam / Cameron, Herb / Herbert, Zach / Zachary) need no entry.
_NICK_GROUPS = (
    {"nic", "nick", "nico", "nicolas", "nicholas"}, {"moe", "mo", "moritz"},
    {"bobby", "bob", "rob", "robbie", "robert"}, {"jake", "jacob"}, {"kenny", "kenneth"},
    {"ray", "rayj"}, {"trey", "tre"}, {"tim", "timothy"}, {"jeff", "jeffrey"},
    {"mike", "michael"}, {"matt", "matthew"}, {"will", "william", "bill", "billy"},
    {"dan", "danny", "daniel"}, {"ron", "ronald"}, {"sam", "samuel"},
    {"alex", "alexander", "alexandre"}, {"tony", "anthony"}, {"jon", "jonathan", "jonathon"},
    {"pat", "patrick"}, {"joe", "joseph", "joey"}, {"nate", "nathan", "nathaniel"},
    {"cam", "cameron"}, {"herb", "herbert"}, {"zach", "zack", "zachary"},
)
_NICK = {n: i for i, g in enumerate(_NICK_GROUPS) for n in g}
FUZZY_MIN = 0.86            # SequenceMatcher ratio on the compacted full name
FUZZY_MARGIN = 0.05         # ... and it must beat the runner-up by this much
ROSTER_LOOKBACK_DAYS = 300


def _tok(name: str | None) -> list[str]:
    return norm_name(name).split()


def _first_ok(a: str, b: str) -> bool:
    """Are two first-name tokens plausibly the same person?"""
    if a == b:
        return True
    if _NICK.get(a) is not None and _NICK.get(a) == _NICK.get(b):
        return True
    if len(a) == 1 or len(b) == 1:                    # "J." vs "James"
        return a[0] == b[0]
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short)


def match_player(name: str | None, roster: list[tuple[int, str]],
                 player_id: str | int | None = None,
                 league: list[tuple[int, str]] | None = None
                 ) -> tuple[int | None, str | None]:
    """Map a report name to a (player_id, player_name) of `roster` -> (player_id, how).

    `roster` is the team's recent players, which is what makes the loose rules below safe.
    A plain-integer `player_id` that is on the roster wins ("id"); ESPN ids ("espn:1966") carry
    no NBA id and are ignored. Then, in order: "exact" (same normalised name: accents, case,
    punctuation, Jr./III and "Last, First" all normalised away); "compact" (spaces/hyphens
    ignored: P.J. = PJ, De'Aaron = DeAaron); "nickname" (same surname, first names that are
    nicknames, prefixes or initials of each other: Cam/Cameron, Nic/Nicolas); "fuzzy" (typos,
    unambiguous only). Returns (None, "ambiguous") when two roster players fit equally well and
    (None, None) when nobody does.

    `league` (everybody active recently, any team) is a last resort for players who changed
    team (traded, signed) and so are not on the roster yet: only an exact/compact name match
    that is unique league-wide counts ("other_team")."""
    pid, how = _match_roster(name, roster, player_id)
    if pid is None and how is None and league:
        lp, lhow = _match_roster(name, league, None, loose=False)
        if lp is not None:
            return lp, "other_team"
    return pid, how


def _match_roster(name, roster, player_id, loose: bool = True):
    if player_id is not None and str(player_id).strip().isdigit():
        pid = int(str(player_id))
        if any(p == pid for p, _ in roster):
            return pid, "id"
    toks = _tok(name)
    if not toks:
        return None, None
    key = " ".join(toks)
    compact = "".join(toks)
    cand = [(p, _tok(n)) for p, n in roster if n]
    for how, test in (("exact", lambda t: " ".join(t) == key),
                      ("compact", lambda t: "".join(t) == compact)):
        hit = {p for p, t in cand if test(t)}
        if len(hit) == 1:
            return hit.pop(), how
        if len(hit) > 1:
            return None, "ambiguous"
    if not loose:
        return None, None
    if len(toks) >= 2:
        rest = "".join(toks[1:])
        hit = {p for p, t in cand if len(t) >= 2 and "".join(t[1:]) == rest
               and _first_ok(toks[0], t[0])}
        if len(hit) == 1:
            return hit.pop(), "nickname"
        if len(hit) > 1:
            return None, "ambiguous"
    scored = sorted(((difflib.SequenceMatcher(None, compact, "".join(t)).ratio(), p)
                     for p, t in cand), reverse=True)
    if scored and scored[0][0] >= FUZZY_MIN:
        if len(scored) == 1 or scored[0][0] - scored[1][0] >= FUZZY_MARGIN:
            return scored[0][1], "fuzzy"
        return None, "ambiguous"
    return None, None


def nba_day_of(captured_at: str) -> date:
    """US/Eastern calendar day of an ISO UTC timestamp."""
    from . import config
    return config.nba_day(_ts(captured_at))


def active_players(conn: sqlite3.Connection, before_day: date
                   ) -> list[tuple[int, str, str]]:
    """[(player_id, player_name, team)] for every player who logged minutes in the
    ROSTER_LOOKBACK_DAYS before `before_day`, with the team of his most recent game."""
    lo = (before_day - timedelta(days=ROSTER_LOOKBACK_DAYS)).isoformat()
    last: dict[int, tuple[str, str, str]] = {}
    for r in conn.execute(
            "SELECT player_id, player_name, team, MAX(game_date) AS d FROM player_games "
            "WHERE game_date < ? AND game_date >= ? AND min > 0 GROUP BY player_id, team",
            (before_day.isoformat(), lo)):
        pid = int(r["player_id"])
        if pid not in last or r["d"] > last[pid][0]:
            last[pid] = (r["d"], r["player_name"], r["team"])
    return [(pid, name, team) for pid, (_, name, team) in last.items()]


def recent_rosters(conn: sqlite3.Connection, before_day: date, teams: set[str] | None = None,
                   active: list[tuple[int, str, str]] | None = None
                   ) -> dict[str, list[tuple[int, str]]]:
    """team -> [(player_id, player_name)]: players whose most recent game (within
    ROSTER_LOOKBACK_DAYS before `before_day`) was for that team. Wider than the engine's
    10-game rotation on purpose: a star back from a month out is still "his" team's player and
    must match (the engine adds a listed, rated player with P > 0 to the lineup). Matching is
    still safe: ~25 candidates per team. In the off-season this is last season's end."""
    out: dict[str, list[tuple[int, str]]] = {}
    for pid, name, team in active if active is not None else active_players(conn, before_day):
        if teams is None or team in teams:
            out.setdefault(team, []).append((pid, name))
    return out


def team_availability(rows: list[dict], roster: list[tuple[int, str]],
                      source: str | None = None, league: list[tuple[int, str]] | None = None
                      ) -> tuple[dict[int, float], list[dict]]:
    """({player_id: P(plays)} for matched players with P < 1, one record per report row).

    A record: player, status, p_play, player_id (None if unmatched), how (match rule or None),
    reason, source. Unmatched rows are ignored by the model: a name that is not on the team's
    recent roster is not in the projected lineup anyway. Duplicate rows keep the lowest P."""
    avail: dict[int, float] = {}
    recs = []
    for r in rows:
        pid, how = match_player(r.get("player_name"), roster, r.get("player_id"), league)
        p = play_probability(r.get("status"))
        recs.append({"player": r.get("player_name"), "status": r.get("status"), "p_play": p,
                     "player_id": pid, "how": how, "reason": (r.get("reason") or "")
                     .split("|")[0].strip() or None, "source": source})
        if pid is not None and p < 1.0:
            avail[pid] = min(p, avail.get(pid, 1.0))
    return avail, recs


def availability(conn: sqlite3.Connection, at: str, day: date, teams: set[str]) -> dict:
    """Injury-report availability for `teams` playing on `day`, as known at `at`.

    {team: {"basis": "injury_report", "source", "captured_at", "avail": {pid: p},
            "players": [records of team_availability]}} for every team in `teams` when at least
    one usable snapshot exists (a team the report does not list is healthy: avail {}, source
    None); an empty dict when no snapshot is usable, so the model keeps its own default."""
    res = resolve(conn, at, espn_max_age_h=ESPN_MAX_AGE_H)
    if not res["fresh"]:
        return {}
    act = active_players(conn, day)
    rosters = recent_rosters(conn, day, teams, act)
    league = [(p, n) for p, n, _ in act]
    out = {}
    for team in teams:
        t = res["teams"].get(team)
        avail, recs = team_availability(t["rows"] if t else [], rosters.get(team, []),
                                        t["source"] if t else None, league)
        out[team] = {"basis": "injury_report", "source": t["source"] if t else None,
                     "captured_at": t["captured_at"] if t else None,
                     "avail": avail, "players": recs}
    return out


# --- match quality over the archive ----------------------------------------------------------

def _league_names(conn: sqlite3.Connection) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for pid, name in conn.execute("SELECT DISTINCT player_id, player_name FROM player_games"):
        out.setdefault(norm_name(name), set()).add(int(pid))
    return out


def _season_rosters(conn: sqlite3.Connection, day: date) -> set[tuple[str, str]]:
    """{(team, normalised name)} of the season's official roster snapshot (`team_rosters`)."""
    y = day.year if day.month >= 8 else day.year - 1
    season = f"{y}-{(y + 1) % 100:02d}"
    try:
        rows = conn.execute("SELECT team, player FROM team_rosters WHERE season=?", (season,))
        return {(r[0], norm_name(r[1])) for r in rows}
    except sqlite3.OperationalError:
        return set()


def match_report(conn: sqlite3.Connection, batches_: list[tuple[str, str]] | None = None,
                 csv_path: Path | str | None = None) -> dict:
    """Match every row of the given (source, captured_at) snapshot batches (default: all) to
    its team's roster (see `recent_rosters`); write the unmatched rows to `csv_path` if given.

    Per batch and overall: rows, matched, rate, and the unmatched split into `off_roster`
    (a player_games name that is not active for any team within the lookback: G League, out for
    the season before it began, retired), `unknown` (never in player_games: rookies, new
    signings) and `ambiguous`; `no_history` is an unmatched name that IS on his team's current
    official roster (`team_rosters`) but has no NBA minutes in the lookback (rookies, two-way and
    camp players): the model has nothing to price for him, so he is left out of the rates; `other_team` matches (traded/signed, found league-wide) count as
    matched and are also counted separately. `relevant_*` repeats the counts for Out/Doubtful/Questionable rows only,
    the ones that can move a prediction."""
    if batches_ is None:
        batches_ = [(r[0], r[1]) for r in conn.execute(
            "SELECT DISTINCT source, captured_at FROM injury_snapshots "
            "ORDER BY captured_at, source")]
    known = _league_names(conn)
    per, unmatched = [], []
    keys = ("rows", "matched", "other_team", "off_roster", "unknown", "ambiguous", "no_history",
            "relevant_rows", "relevant_matched", "relevant_no_history")
    tot = dict.fromkeys(keys, 0)
    how_count: dict[str, int] = {}
    for src, cap in batches_:
        rows = _batch_rows(conn, src, cap)
        act = active_players(conn, nba_day_of(cap))
        rosters = recent_rosters(conn, nba_day_of(cap), None, act)
        league = [(p, n) for p, n, _ in act]
        official = _season_rosters(conn, nba_day_of(cap))
        b = dict.fromkeys(keys, 0)
        for r in rows:
            roster = rosters.get(r["team"], [])
            pid, how = match_player(r["player_name"], roster, r.get("player_id"), league)
            rel = int(severity(r["status"]) >= 1)
            b["rows"] += 1
            b["relevant_rows"] += rel
            if pid is not None:
                b["matched"] += 1
                b["relevant_matched"] += rel
                b["other_team"] += how == "other_team"
                how_count[how] = how_count.get(how, 0) + 1
                continue
            cat = ("ambiguous" if how == "ambiguous" else
                   "no_history" if (r["team"], norm_name(r["player_name"])) in official else
                   "off_roster" if norm_name(r["player_name"]) in known else "unknown")
            b["relevant_no_history"] += rel and cat == "no_history"
            b[cat] += 1
            unmatched.append({"source": src, "captured_at": cap, "team": r["team"],
                              "player_name": r["player_name"], "player_id": r.get("player_id"),
                              "status": r["status"], "relevant": rel, "category": cat,
                              "roster_size": len(roster)})
        for k in keys:
            tot[k] += b[k]
        per.append({"source": src, "captured_at": cap, **b, **_rates(b)})
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["source", "captured_at", "team", "player_name", "player_id", "status",
                "relevant", "category", "roster_size"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(unmatched)
    return {"batches": per, "total": {**tot, **_rates(tot)}, "matched_by": how_count,
            "unmatched": unmatched}


def _rates(b: dict) -> dict:
    """Match rates over the rows the model could price (`no_history` rows are excluded)."""
    n = b["rows"] - b.get("no_history", 0)
    nr = b["relevant_rows"] - b.get("relevant_no_history", 0)
    return {"rate": round(b["matched"] / n, 4) if n else None,
            "relevant_rate": round(b["relevant_matched"] / nr, 4) if nr else None}
