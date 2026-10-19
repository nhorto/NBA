"""DraftKings / RotoGuru player ids -> NBA `player_id` (docs/PLAYERS.md §3.4, T0.5).

Persisted in `player_ids(source, source_id, player_id, ...)`: source "dk" (DK `playerId`, stable
across slates; the per-slate `draftableId` is never used) and "rotoguru" (the RotoGuru id).

Matching ladder, per player (a name, a team and, for RotoGuru, a date):

  alias         `player_aliases` (source, normalised name, team or '' for any team): hand-made
                and seeded from `aliases_seed.csv`; always wins
  exact_team    same normalised name on the team's pool (accents, case, Jr./III, "Last, First"
                and spaces/hyphens all normalised away by `injuries.norm_name`), unique
  exact_league  the same name, unique league-wide (traded / newly signed: DK knows the team
                before our rosters do)
  fuzzy         `injuries.match_player` on the team's pool: nicknames (Cam/Cameron) and typos
                (FUZZY_MIN 0.86, 0.05 margin), with the near-duplicate guard below

Near-duplicate guard: a *fuzzy* match is refused when another player of the same pool shares the
matched player's last name and the first name is not an exact token match (Jaylen Williams must
not become Jalen or Jaylin Williams: a human decides, via an alias).

DK pool (today): current-season `team_rosters`, the `player_index` snapshot (has the rookie
class and current teams) and `injuries.recent_rosters` (last team of everybody who played in
the last 300 days), unioned per team. RotoGuru (history): the (date, team, name) key against
`player_games` of that day, then the team's recent roster as of that date (players with a
salary who did not play are not in that day's box score).

An alias / manual mapping is never replaced by an automatic one; automatic mappings (exact_team >
exact_league > fuzzy) are recomputed every run, so they stay deterministic and a matcher fix
repairs old rows. A player nobody matched is stored with how='unmatched' and player_id NULL so
the gap is queryable.
Every `resolve_*` writes a `data_checks` row (`ids_dk`, `ids_rotoguru`) listing the unmatched.

Public API
  resolve_dk(conn, *, since=None, as_of=None) -> summary dict
  resolve_rotoguru(conn, *, since=None) -> summary dict
  add_alias(conn, source, name, team, player_id, note=None)
  nba_id(conn, source, source_id) -> int | None;  nba_ids(conn, source, source_ids=None) -> dict
  add_commands(sub) -> {"ids": handler, "alias": handler}      (`courtside proj ids|alias`)
"""

from __future__ import annotations

import csv
import difflib
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .. import config, teams
from ..injuries import FUZZY_MIN, _first_ok, active_players, match_player
from ..injuries import norm_name as _injuries_norm_name

DK = "dk"
ROTOGURU = "rotoguru"
SOURCES = (DK, ROTOGURU)
SEED_PATH = Path(__file__).with_name("aliases_seed.csv")

# How strong a mapping is; a weaker one never replaces a stronger one.
RANK = {"alias": 4, "manual": 4, "exact_team": 3, "exact_league": 2, "fuzzy": 1, "unmatched": 0}
CONF = {"alias": 1.0, "exact_team": 1.0, "exact_team_compact": 0.98, "exact_league": 0.9,
        "nickname": 0.9}
LIKELY_MIN = 10.0              # "likely to play" proxy: averaged this many minutes last season
CHECK_WARN = {DK: 0.98, ROTOGURU: 0.99}
CHECK_FAIL = {DK: 0.75, ROTOGURU: 0.95}   # DK also fails on any unmatched likely rotation player
UNMATCHED_LISTED = 25          # names put in the data-check detail

# RotoGuru's own team codes that `teams.franchise` does not know
_RG_TEAMS = {"BRO": "BKN", "NOR": "NOP", "SAN": "SAS", "GOL": "GSW", "PHO": "PHX"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _team(code: str | None, season: str | int | None = None) -> str:
    """Franchise tricode of a team code; unknown codes are kept upper-cased, None -> ''."""
    c = (code or "").strip().upper()
    if not c:
        return ""
    c = _RG_TEAMS.get(c, c)
    try:
        return teams.franchise(c, season)
    except KeyError:
        return c


def _season_of(day: str) -> int:
    y, m = int(day[:4]), int(day[5:7])
    return y if m >= 8 else y - 1


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str | None) -> str:
    """`injuries.norm_name`, except that a *leading* "JR" is a first name, not a suffix:
    injuries.norm_name turns "JR Smith" into "smith" (and so does not match "J.R. Smith")."""
    base = _injuries_norm_name(name)
    raw = (name or "").strip()
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    words = raw.split()
    if len(words) >= 2 and words[0].lower() in _SUFFIXES:
        return f"{words[0].lower()} {base}"
    return base


def _tok(name: str | None) -> list[str]:
    return norm_name(name).split()


def _compact(name: str | None) -> str:
    return "".join(_tok(name))


# ---------------------------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------------------------

def _player_known(conn: sqlite3.Connection, player_id: int) -> bool:
    for sql in ("SELECT 1 FROM player_index WHERE player_id=?",
                "SELECT 1 FROM team_rosters WHERE player_id=? LIMIT 1",
                "SELECT 1 FROM player_games WHERE player_id=? LIMIT 1"):
        if conn.execute(sql, (player_id,)).fetchone():
            return True
    return False


def add_alias(conn: sqlite3.Connection, source: str, name: str, team: str | None,
              player_id: int, note: str | None = None, *, check: bool = True) -> None:
    """Pin (source, name, team) to an NBA player_id. `team` '' / None / '-' = any team.

    The id must exist in player_index / team_rosters / player_games (guards typos) unless
    `check` is False. Picked up by the next `resolve_*` run, which then also writes the
    player_ids row (how='alias')."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    key = norm_name(name)
    if not key:
        raise ValueError("empty name")
    player_id = int(player_id)
    if check and not _player_known(conn, player_id):
        raise ValueError(f"player_id {player_id} is not in player_index/team_rosters/player_games")
    t = "" if team in (None, "", "-") else _team(team)
    conn.execute(
        "INSERT INTO player_aliases (source, name_norm, team, player_id, note) VALUES (?,?,?,?,?) "
        "ON CONFLICT(source, name_norm, team) DO UPDATE SET player_id=excluded.player_id, "
        "note=excluded.note", (source, key, t, player_id, note))
    conn.commit()


def load_seed(conn: sqlite3.Connection, path: Path | str | None = None) -> int:
    """Insert the rows of `aliases_seed.csv` (source,name,team,player_id,note) that are not in
    `player_aliases` yet; rows already there (hand edits) are left alone. Returns rows added."""
    p = Path(path) if path else SEED_PATH
    if not p.exists():
        return 0
    added = 0
    with p.open(newline="", encoding="utf-8") as f:
        lines = (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
        for r in csv.DictReader(lines):
            key = norm_name(r["name"])
            t = "" if (r.get("team") or "-") in ("", "-") else _team(r["team"])
            cur = conn.execute(
                "INSERT OR IGNORE INTO player_aliases (source, name_norm, team, player_id, note) "
                "VALUES (?,?,?,?,?)", (r["source"].strip(), key, t, int(r["player_id"]),
                                       (r.get("note") or "").strip() or None))
            added += cur.rowcount
    conn.commit()
    return added


def _aliases(conn: sqlite3.Connection, source: str) -> dict[tuple[str, str], int]:
    return {(r["name_norm"], r["team"] or ""): int(r["player_id"]) for r in conn.execute(
        "SELECT name_norm, team, player_id FROM player_aliases WHERE source=?", (source,))}


# ---------------------------------------------------------------------------------------------
# candidate pools
# ---------------------------------------------------------------------------------------------

class Pool:
    """Candidate players: per team, and league-wide in two tiers (recent, then everybody).

    Every player id keeps all name variants seen (rosters say "Nic Claxton", player_index
    "Nicolas Claxton"); the first one is the primary name used for fuzzy matching."""

    def __init__(self) -> None:
        self.teams: dict[str, dict[int, list[str]]] = defaultdict(dict)
        self.league: dict[int, list[str]] = {}
        self.league_all: dict[int, list[str]] = {}

    @staticmethod
    def _put(d: dict[int, list[str]], pid: int, name: str | None) -> None:
        names = d.setdefault(int(pid), [])
        if name and name not in names:
            names.append(name)

    def add(self, pid: int, name: str | None, team: str | None = None, recent: bool = True) -> None:
        if team:
            self._put(self.teams[team], pid, name)
        if recent:
            self._put(self.league, pid, name)
        self._put(self.league_all, pid, name)

    def roster(self, team: str) -> list[tuple[int, str]]:
        return [(p, n[0]) for p, n in self.teams.get(team, {}).items() if n]

    def name(self, pid: int) -> str | None:
        for d in (self.league, self.league_all):
            if pid in d and d[pid]:
                return d[pid][0]
        return None


def current_pool(conn: sqlite3.Connection, as_of: date | None = None) -> Pool:
    """Pool for today's DK captures. 2026-27 rosters may not be loaded yet: every source is
    used, so a player is a candidate for a team if *any* of them says so."""
    as_of = as_of or config.today_nba_day()
    pool = Pool()
    row = conn.execute("SELECT MAX(season) FROM team_rosters").fetchone()
    if row and row[0]:
        for r in conn.execute("SELECT team, player_id, player FROM team_rosters WHERE season=?",
                              (row[0],)):
            pool.add(r["player_id"], r["player"], _team(r["team"]))
    top = conn.execute("SELECT MAX(to_year) FROM player_index").fetchone()[0]
    for r in conn.execute("SELECT player_id, first_name, last_name, team, to_year "
                          "FROM player_index"):
        name = f"{r['first_name'] or ''} {r['last_name'] or ''}".strip()
        cur = top is not None and (r["to_year"] or 0) >= top
        recent = top is not None and (r["to_year"] or 0) >= top - 1
        pool.add(r["player_id"], name, _team(r["team"]) if cur and r["team"] else None, recent)
    for pid, name, team in active_players(conn, as_of + timedelta(days=1)):
        pool.add(pid, name, _team(team))
    for pid, name in conn.execute("SELECT DISTINCT player_id, player_name FROM player_games "
                                  "WHERE player_name IS NOT NULL"):
        pool.add(pid, name, None, recent=False)
    return pool


def _pool_from_active(act: list[tuple[int, str, str]]) -> Pool:
    pool = Pool()
    for pid, name, team in act:
        pool.add(pid, name, _team(team))
    return pool


# ---------------------------------------------------------------------------------------------
# matching one name
# ---------------------------------------------------------------------------------------------

def _exact_hits(name: str, cands: dict[int, list[str]]) -> tuple[set[int], str]:
    """Players of `cands` whose name equals `name` once normalised ('exact'), else once spaces
    and hyphens are dropped too ('compact': P.J. = PJ). Returns (ids, rule)."""
    key = norm_name(name)
    comp = _compact(name)
    if not key:
        return set(), ""
    hit = {p for p, ns in cands.items() if any(norm_name(n) == key for n in ns)}
    if hit:
        return hit, "exact"
    return {p for p, ns in cands.items() if any(_compact(n) == comp for n in ns)}, "compact"


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv)\.?$", re.I)


def _suffix(name: str | None) -> str:
    """The Jr./Sr./II/III/IV ending of a name ('' if none); "Jackson Jr., Jaren" counts."""
    s = (name or "").strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    m = _SUFFIX_RE.search(s)
    return m.group(1).lower() if m else ""


def _same_suffix(name: str, cands: dict[int, list[str]], hit: set[int]) -> set[int]:
    """Drop players of `hit` who carry no variant with the query's own suffix. norm_name strips
    Jr./Sr., so "Jameer Nelson Jr." would otherwise match his father, retired in 2019."""
    want = _suffix(name)
    if not want:
        return hit
    return {p for p in hit if any(_suffix(n) == want for n in cands[p])}


def _ratio(a: str | None, b: str | None) -> float:
    return difflib.SequenceMatcher(None, _compact(a), _compact(b)).ratio()


def _fuzzy(name: str, roster: list[tuple[int, str]]) -> tuple[int | None, str, float]:
    """`injuries.match_player` on the team's pool + the near-duplicate guard.
    -> (player_id, rule, confidence); player_id None carries the failure reason in `rule`."""
    pid, how = match_player(name, roster)
    if pid is None:
        return None, (how or "no_match"), 0.0
    if how in ("exact", "compact", "id"):           # normalised-equal: handled by _exact_hits
        return pid, "exact", 1.0
    matched = next(n for p, n in roster if p == pid)
    if how == "fuzzy":
        qt, mt = _tok(name), _tok(matched)
        sharing = [p for p, n in roster if p != pid and _tok(n)[-1:] == mt[-1:]]
        if sharing and qt[:1] != mt[:1]:
            return None, "near_duplicate", 0.0
        return pid, "fuzzy", round(min(0.89, max(FUZZY_MIN, _ratio(name, matched))), 3)
    return pid, "nickname", CONF["nickname"]


def resolve_name(name: str, team: str, pool: Pool,
                 aliases: dict[tuple[str, str], int] | None = None
                 ) -> tuple[int | None, str, float]:
    """The ladder in the module docstring -> (player_id, how, confidence).

    how is alias / exact_team / exact_league / fuzzy for a match; for no match player_id is None
    and how is the reason (no_match, ambiguous, near_duplicate)."""
    key = norm_name(name)
    if not key:
        return None, "no_name", 0.0
    if aliases:
        for k in ((key, team), (key, "")):
            if k in aliases:
                return aliases[k], "alias", CONF["alias"]
    reason = "no_match"
    team_pool = pool.teams.get(team, {})
    hit, rule = _exact_hits(name, team_pool)
    if len(hit) == 1:
        return next(iter(hit)), "exact_team", CONF["exact_team" if rule == "exact"
                                                    else "exact_team_compact"]
    if len(hit) > 1:
        return None, "ambiguous", 0.0
    for tier in (pool.league, pool.league_all):
        hit, _ = _exact_hits(name, tier)
        if tier is pool.league_all:                  # long-retired namesakes (Sr. vs Jr.)
            hit = _same_suffix(name, tier, hit)
        if len(hit) == 1:
            return next(iter(hit)), "exact_league", CONF["exact_league"]
        if len(hit) > 1:
            reason = "ambiguous"
            break
    pid, rule, conf = _fuzzy(name, pool.roster(team))
    if pid is not None:
        return pid, ("fuzzy" if rule in ("fuzzy", "nickname") else "exact_team"), conf
    return None, (reason if reason != "no_match" else rule), 0.0


def candidates(name: str, pool: Pool, avg_min: dict[int, float], n: int = 3) -> list[dict]:
    """Likely NBA players for an unmatched name, for a human: same last name or a close
    spelling, best first. [{player_id, name, ratio, avg_min, plausible}]; `plausible` = the
    spelling is very close, or same surname and a first name that could be the same person
    (nickname / initial: EJ ~ Elijah), which is what makes an unmatched player "likely"."""
    toks = _tok(name)
    last = toks[-1] if toks else ""
    out = []
    for pid, names in pool.league_all.items():
        best = max((_ratio(name, nm) for nm in names), default=0.0)
        same_last = any(_tok(nm)[-1:] == [last] for nm in names) if last else False
        if best >= 0.75 or (same_last and best >= 0.5):
            plausible = best >= 0.88 or (same_last and any(
                _tok(nm) and toks and (_first_ok(toks[0], _tok(nm)[0])
                                       or toks[0][0] == _tok(nm)[0][0]
                                       and toks[0] != _tok(nm)[0] and len(toks[0]) <= 2)
                for nm in names))
            out.append({"player_id": pid, "name": names[0], "ratio": round(best, 2),
                        "avg_min": round(avg_min.get(pid, 0.0), 1), "plausible": plausible})
    out.sort(key=lambda c: (-c["ratio"], -c["avg_min"]))
    return out[:n]


# ---------------------------------------------------------------------------------------------
# persistence + lookup
# ---------------------------------------------------------------------------------------------

def _upsert_id(conn: sqlite3.Connection, source: str, source_id: str, player_id: int | None,
               name: str | None, team: str | None, first_seen: str | None, last_seen: str | None,
               how: str, confidence: float) -> None:
    """Write a mapping. An alias / manual mapping is never replaced by an automatic one (then only
    first_seen/last_seen move); automatic mappings are simply recomputed, so a fixed matcher
    also fixes what an earlier run got wrong."""
    old = conn.execute("SELECT player_id, how, first_seen, last_seen FROM player_ids "
                       "WHERE source=? AND source_id=?", (source, source_id)).fetchone()
    if old is None:
        conn.execute("INSERT INTO player_ids (source, source_id, player_id, name, team, "
                     "first_seen, last_seen, how, confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                     (source, source_id, player_id, name, team, first_seen, last_seen, how,
                      confidence))
        return
    first = min(x for x in (old["first_seen"], first_seen) if x) if (
        old["first_seen"] or first_seen) else None
    last = max(x for x in (old["last_seen"], last_seen) if x) if (
        old["last_seen"] or last_seen) else None
    if old["player_id"] is not None and RANK.get(old["how"] or "", 0) > RANK.get(how, 0) \
            and RANK.get(old["how"] or "", 0) >= RANK["alias"]:
        conn.execute("UPDATE player_ids SET first_seen=?, last_seen=? WHERE source=? AND "
                     "source_id=?", (first, last, source, source_id))
        return
    conn.execute("UPDATE player_ids SET player_id=?, name=?, team=?, first_seen=?, last_seen=?, "
                 "how=?, confidence=? WHERE source=? AND source_id=?",
                 (player_id, name, team, first, last, how, confidence, source, source_id))


def nba_id(conn: sqlite3.Connection, source: str, source_id: str | int) -> int | None:
    """NBA player_id of a DK playerId / RotoGuru id, None if unknown or unmatched."""
    r = conn.execute("SELECT player_id FROM player_ids WHERE source=? AND source_id=?",
                     (source, str(source_id))).fetchone()
    return int(r[0]) if r and r[0] is not None else None


def nba_ids(conn: sqlite3.Connection, source: str,
            source_ids: list | set | tuple | None = None) -> dict[str, int]:
    """{source_id: player_id} for every matched id of `source` (restricted to `source_ids`
    when given). Keys are strings."""
    rows = conn.execute("SELECT source_id, player_id FROM player_ids WHERE source=? AND "
                        "player_id IS NOT NULL", (source,))
    out = {str(r[0]): int(r[1]) for r in rows}
    if source_ids is not None:
        want = {str(s) for s in source_ids}
        out = {k: v for k, v in out.items() if k in want}
    return out


# ---------------------------------------------------------------------------------------------
# summary + data check
# ---------------------------------------------------------------------------------------------

def _last_season_minutes(conn: sqlite3.Connection) -> dict[int, float]:
    """player_id -> average minutes (games played) in the latest complete regular season."""
    row = conn.execute(
        "SELECT season FROM player_games WHERE season_type='Regular Season' GROUP BY season "
        "HAVING COUNT(*) >= 20000 ORDER BY season DESC LIMIT 1").fetchone()
    if not row:
        return {}
    return {int(r[0]): float(r[1]) for r in conn.execute(
        "SELECT player_id, AVG(min) FROM player_games WHERE season=? AND "
        "season_type='Regular Season' AND min>0 GROUP BY player_id", (row[0],))}


def _write_check(conn: sqlite3.Connection, source: str, summary: dict) -> dict:
    from ..mlops import checks
    name = f"ids_{source}"
    n, rate = summary["n"], summary["rate"]
    um = summary["unmatched"]
    listed = ", ".join(f"{u['name']} ({u['team'] or '?'})" + (f" x{u['rows']}" if "rows" in u else "")
                       for u in um[:UNMATCHED_LISTED])
    more = f" (+{len(um) - UNMATCHED_LISTED} more)" if len(um) > UNMATCHED_LISTED else ""
    base = (f"{summary['matched']}/{n} {source} ids matched ({rate:.1%}, auto "
            f"{summary['auto_rate']:.1%})")
    bad = f"{base}; unmatched: {listed}{more}. Add rows with `courtside proj alias`."
    if n == 0:
        res = checks._res(name, "skip", None, CHECK_WARN[source], f"no {source} rows")
    else:
        res = checks._ladder(name, rate, CHECK_WARN[source], CHECK_FAIL[source],
                             base + (f"; unmatched: {listed}{more}" if um else ""), bad)
        if res["status"] != "fail" and summary.get("likely_unmatched"):
            lu = ", ".join(u["name"] for u in summary["likely_unmatched"])
            res = checks._res(name, "fail", rate, CHECK_WARN[source],
                              f"{base}; likely rotation players unmatched: {lu}. "
                              "Add rows with `courtside proj alias`.")
    res["run_at"] = _now()
    with conn:
        conn.execute("INSERT OR REPLACE INTO data_checks (run_at, check_name, status, value, "
                     "threshold, detail) VALUES (:run_at, :check_name, :status, :value, "
                     ":threshold, :detail)", res)
    return res


# ---------------------------------------------------------------------------------------------
# DraftKings
# ---------------------------------------------------------------------------------------------

def resolve_dk(conn: sqlite3.Connection, *, since: str | None = None,
               as_of: date | None = None, write_check: bool = True) -> dict:
    """Map every distinct DK `site_player_id` of `dfs_salaries` (captured_at >= `since`) to an
    NBA player_id, persist it in `player_ids(source='dk')` and write the `ids_dk` data check.

    Summary: n (players), matched, auto (matched without an alias), rate, auto_rate, by_how,
    unmatched [{source_id, name, team, reason, candidates, likely}], likely_total,
    likely_matched, likely_unmatched (unmatched with a candidate who averaged >= 10 min last
    season: the ones that must be aliased before a slate)."""
    load_seed(conn)
    sql = ("SELECT site_player_id, player_name, team, captured_at FROM dfs_salaries "
           "WHERE site_player_id IS NOT NULL AND player_name IS NOT NULL")
    args: list = []
    if since:
        sql += " AND captured_at >= ?"
        args.append(since)
    seen: dict[str, dict] = {}
    for r in conn.execute(sql + " ORDER BY captured_at", args):
        sid = str(r["site_player_id"])
        e = seen.setdefault(sid, {"first": r["captured_at"]})
        e.update(name=r["player_name"], team=_team(r["team"]), last=r["captured_at"])
    pool = current_pool(conn, as_of)
    aliases = _aliases(conn, DK)
    avg_min = _last_season_minutes(conn)
    by_how: Counter = Counter()
    unmatched: list[dict] = []
    likely_total = likely_matched = 0
    with conn:
        for sid, e in sorted(seen.items()):
            pid, how, conf = resolve_name(e["name"], e["team"], pool, aliases)
            if pid is None:
                _upsert_id(conn, DK, sid, None, e["name"], e["team"], e["first"], e["last"],
                           "unmatched", 0.0)
                cands = candidates(e["name"], pool, avg_min)
                likely = any(c["plausible"] and c["avg_min"] >= LIKELY_MIN for c in cands)
                unmatched.append({"source_id": sid, "name": e["name"], "team": e["team"],
                                  "reason": how, "candidates": cands, "likely": likely})
                likely_total += likely
                continue
            _upsert_id(conn, DK, sid, pid, e["name"], e["team"], e["first"], e["last"], how, conf)
            by_how[how] += 1
            if avg_min.get(pid, 0.0) >= LIKELY_MIN:
                likely_total += 1
                likely_matched += 1
    # report what is stored (a stronger earlier mapping may have survived a weaker new one)
    stored = {r["source_id"]: r for r in conn.execute(
        "SELECT source_id, player_id, how FROM player_ids WHERE source=?", (DK,))}
    still = [u for u in unmatched if stored[u["source_id"]]["player_id"] is None]
    n = len(seen)
    matched = n - len(still)
    auto = sum(1 for sid in seen if stored[sid]["player_id"] is not None
               and stored[sid]["how"] not in ("alias", "manual"))
    summary = {"source": DK, "n": n, "matched": matched, "auto": auto,
               "rate": matched / n if n else 1.0, "auto_rate": auto / n if n else 1.0,
               "by_how": dict(by_how), "unmatched": still,
               "likely_total": likely_total, "likely_matched": likely_matched,
               "likely_unmatched": [u for u in still if u["likely"]]}
    if write_check:
        summary["check"] = _write_check(conn, DK, summary)
    return summary


# ---------------------------------------------------------------------------------------------
# RotoGuru
# ---------------------------------------------------------------------------------------------

def resolve_rotoguru(conn: sqlite3.Connection, *, since: str | None = None,
                     write_check: bool = True) -> dict:
    """Fill `dfs_hist_salaries.player_id` and `player_ids(source='rotoguru')`.

    Per row the best key is (game_date, team, normalised name) against that day's
    `player_games` ("Last, First" is handled by norm_name); a name alone on that day is the
    next best, then the team's recent roster as of that date (salaried players who did not
    play are not in the box score). A row left over takes the id its rg_id resolved to on other
    days. `player_ids` holds one row per rg_id: the majority player, confidence scaled by the
    share of that rg_id's rows that agree.

    Summary: n (salaried player-games), matched, auto, rate, auto_rate, by_how, rg_ids,
    rg_conflicts, unmatched [{source_id, name, team, rows, first_date, reason}]."""
    load_seed(conn)
    sql = "SELECT game_date, rg_id, name, team, player_id FROM dfs_hist_salaries"
    args: list = []
    if since:
        sql += " WHERE game_date >= ?"
        args.append(since)
    rows = [dict(r) for r in conn.execute(sql + " ORDER BY game_date", args)]
    if not rows:
        summary = {"source": ROTOGURU, "n": 0, "matched": 0, "auto": 0, "rate": 1.0,
                   "auto_rate": 1.0, "by_how": {}, "rg_ids": 0, "rg_conflicts": 0,
                   "unmatched": []}
        if write_check:
            summary["check"] = _write_check(conn, ROTOGURU, summary)
        return summary
    aliases = _aliases(conn, ROTOGURU)
    lo, hi = rows[0]["game_date"], rows[-1]["game_date"]

    # (date, team, name) / (date, name) -> player ids, from the box scores of those days
    by_team: dict[tuple, set[int]] = defaultdict(set)
    by_day: dict[tuple, set[int]] = defaultdict(set)
    for r in conn.execute("SELECT game_date, team, player_id, player_name FROM player_games "
                          "WHERE game_date BETWEEN ? AND ? AND min > 0", (lo, hi)):
        t = _team(r["team"], _season_of(r["game_date"]))
        for k in (norm_name(r["player_name"]), _compact(r["player_name"])):
            by_team[(r["game_date"], t, k)].add(r["player_id"])
            by_day[(r["game_date"], k)].add(r["player_id"])

    res: list[tuple[int | None, str, float]] = []
    left: dict[str, list[int]] = defaultdict(list)       # date -> row indices
    for i, r in enumerate(rows):
        t = _team(r["team"], _season_of(r["game_date"]))
        key, comp = norm_name(r["name"]), _compact(r["name"])
        out: tuple[int | None, str, float] = (None, "no_match", 0.0)
        a = aliases.get((key, t)) or aliases.get((key, ""))
        if a is not None:
            out = (a, "alias", 1.0)
        else:
            for k, conf in ((key, 1.0), (comp, 0.98)):
                hit = by_team.get((r["game_date"], t, k), set())
                if len(hit) == 1:
                    out = (next(iter(hit)), "exact_team", conf)
                    break
            if out[0] is None:
                for k in (key, comp):
                    hit = by_day.get((r["game_date"], k), set())
                    if len(hit) == 1:
                        out = (next(iter(hit)), "exact_league", CONF["exact_league"])
                        break
        res.append(out)
        if out[0] is None:
            left[r["game_date"]].append(i)

    # leftovers: the team's recent roster as of that date (DNPs, nicknames, typos)
    for day, idxs in left.items():
        d = date.fromisoformat(day)
        pool = _pool_from_active(active_players(conn, d + timedelta(days=1)))
        for i in idxs:
            r = rows[i]
            res[i] = resolve_name(r["name"], _team(r["team"], _season_of(day)), pool, aliases)

    # an rg_id is one person: vote across its rows, fill the leftovers from the vote
    votes: dict[str, Counter] = defaultdict(Counter)
    for r, (pid, how, _) in zip(rows, res):
        if pid is not None:
            votes[r["rg_id"]][pid] += 1
    filled = 0
    for i, r in enumerate(rows):
        if res[i][0] is None and votes.get(r["rg_id"]):
            pid, cnt = votes[r["rg_id"]].most_common(1)[0]
            res[i] = (pid, "fuzzy", round(0.85 * cnt / sum(votes[r["rg_id"]].values()), 3))
            filled += 1

    by_how: Counter = Counter(h for p, h, _ in res if p is not None)
    per_id: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        per_id[r["rg_id"]].append(i)
    conflicts = 0
    with conn:
        conn.executemany(
            "UPDATE dfs_hist_salaries SET player_id=? WHERE game_date=? AND rg_id=? AND "
            "player_id IS NOT ?",
            [(res[i][0], r["game_date"], r["rg_id"], res[i][0]) for i, r in enumerate(rows)])
        for rg, idxs in per_id.items():
            last = rows[idxs[-1]]
            dates = [rows[i]["game_date"] for i in idxs]
            hit = [i for i in idxs if res[i][0] is not None]
            if not hit:
                _upsert_id(conn, ROTOGURU, rg, None, last["name"], _team(last["team"]), dates[0],
                           dates[-1], "unmatched", 0.0)
                continue
            cnt = Counter(res[i][0] for i in hit)
            pid, k = cnt.most_common(1)[0]
            conflicts += len(cnt) > 1
            agree = [i for i in hit if res[i][0] == pid]
            best = max((res[i][1] for i in agree), key=lambda h: RANK.get(h, 0))
            conf = round(max(res[i][2] for i in agree) * k / len(idxs) if best != "alias"
                         else 1.0, 3)
            _upsert_id(conn, ROTOGURU, rg, pid, last["name"], _team(last["team"]), dates[0],
                       dates[-1], best, conf)
    n = len(rows)
    matched = sum(1 for p, _, _ in res if p is not None)
    auto = sum(1 for p, h, _ in res if p is not None and h != "alias")
    um: dict[tuple, dict] = {}
    for i, r in enumerate(rows):
        if res[i][0] is None:
            u = um.setdefault((r["rg_id"], r["name"]), {
                "source_id": r["rg_id"], "name": r["name"], "team": _team(r["team"]), "rows": 0,
                "first_date": r["game_date"], "reason": res[i][1]})
            u["rows"] += 1
    unmatched = sorted(um.values(), key=lambda u: -u["rows"])
    summary = {"source": ROTOGURU, "n": n, "matched": matched, "auto": auto,
               "rate": matched / n, "auto_rate": auto / n, "by_how": dict(by_how),
               "rg_ids": len(per_id), "rg_conflicts": conflicts, "filled_by_rg_id": filled,
               "unmatched": unmatched}
    if write_check:
        summary["check"] = _write_check(conn, ROTOGURU, summary)
    return summary


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def add_commands(sub) -> dict:
    p = sub.add_parser("ids", help="map DraftKings / RotoGuru player ids to NBA player_id")
    p.add_argument("--source", choices=("dk", "rotoguru", "all"), default="all")
    p.add_argument("--since", help="only DK captures / RotoGuru dates from here (ISO)")

    p = sub.add_parser("alias", help="pin a DK/RotoGuru name (+team) to an NBA player_id")
    p.add_argument("source", choices=SOURCES)
    p.add_argument("name")
    p.add_argument("team", help="team tricode, or - for any team")
    p.add_argument("player_id", type=int)
    p.add_argument("--note")
    return {"ids": _ids, "alias": _alias}


def _conn():
    from .. import db
    return db.connect()


def _print_summary(s: dict) -> None:
    print(f"== {s['source']}: {s['matched']}/{s['n']} matched ({s['rate']:.1%}); "
          f"auto {s['auto']} ({s['auto_rate']:.1%}); by rule {s['by_how']}")
    if s["source"] == DK:
        print(f"   likely rotation players (>= {LIKELY_MIN:.0f} min last season): "
              f"{s['likely_matched']}/{s['likely_total']} matched")
    else:
        print(f"   rg ids {s['rg_ids']}, conflicting {s['rg_conflicts']}")
    for u in s["unmatched"]:
        extra = f" x{u['rows']} from {u['first_date']}" if "rows" in u else ""
        cands = "; ".join(f"{c['name']} #{c['player_id']} ({c['avg_min']} min)"
                          for c in u.get("candidates", []))
        flag = " [LIKELY]" if u.get("likely") else ""
        print(f"   unmatched {u['source_id']:>9} {u['name']} ({u['team'] or '?'}) "
              f"[{u['reason']}]{extra}{flag}" + (f" ? {cands}" if cands else ""))
    chk = s.get("check")
    if chk:
        print(f"   check {chk['check_name']}: {chk['status']}")


def _ids(args) -> int:
    conn = _conn()
    bad = 0
    if args.source in ("dk", "all"):
        s = resolve_dk(conn, since=args.since)
        _print_summary(s)
        bad += s["check"]["status"] == "fail"
    if args.source in ("rotoguru", "all"):
        s = resolve_rotoguru(conn, since=args.since)
        _print_summary(s)
        bad += s["check"]["status"] == "fail"
    return 1 if bad else 0


def _alias(args) -> int:
    conn = _conn()
    add_alias(conn, args.source, args.name, args.team, args.player_id, args.note)
    s = resolve_dk(conn) if args.source == DK else resolve_rotoguru(conn)
    print(f"alias saved; {args.source}: {s['matched']}/{s['n']} matched ({s['rate']:.1%})")
    return 0
