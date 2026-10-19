"""DraftKings rules, scoring and the scoring verification checks (docs/PLAYERS.md §5.1, T0.6, M4).

Three parts:

* `Rules` is built from an archived DK ``rules_json`` payload (``api.draftkings.com/lineups/v1/
  gametypes/{id}/rules``): lineup slots, salary cap, ``gameCount.min``, ``teamCount.min``, late
  swap and the Showdown captain multiplier. Nothing about the roster is hard-coded; the
  constants below only say which positions a *named* slot accepts (G = PG/SG, F = SF/PF).
* `build_pool` turns the latest ``dfs_salaries`` capture of a slate into one `PoolPlayer` per DK
  ``playerId`` (salary, positions, eligible slots, game, start time, status), without disabled
  rows. `SlatePool.locked_ids(now)` marks the players whose game has started.
* `dk_points` scores a stat line with the DK Classic table (``sources.dfs.CLASSIC_SCORING``),
  including the double-double (+1.5) and triple-double (+3.0 on top = +4.5 total) bonuses over
  points, rebounds, assists, steals and blocks. `check_fppg` and `check_rotoguru` verify that
  table against DK's own numbers and are stored in `data_checks` (`dfs_scoring_fppg`,
  `dfs_scoring_rotoguru`) by `verify`, so they rerun weekly.

What DK's FPPG (draftStatAttributes id 219) is, measured on the 2026-10-20 opening-night
Classic slates (draftables captured 2026-10-19, 92 players matched to ``player_games``):

* It is a plain per-game mean of the DK points above over **every 2025-26 game the player has a
  row for: regular season, play-in and playoffs** (before the new season has ~5 games DK shows
  last season; `MIN_CURRENT_GAMES`). Regular season only misses by 1-3 FP for anyone who
  played in the playoffs (22% within 0.1). Zero-minute rows (a handful of end-of-bench
  players) are ambiguous: including them matches 6 of the 8 players that have one, so they stay in.
* 76 of 92 (82.6%) land within 0.1, most of them within 0.05: the scoring table is exact for
  them (Hukporti 7.6, Dean Wade 15.1, SGA 49.9 ...). The other 16 are 0.1-1.0 FP off. The ten
  worst are all stars with deep playoff runs and DK is **below** our mean (Maxey -1.02, Brown
  -0.87, LeBron -0.87, Cunningham -0.79, Wembanyama -0.79, Towns -0.73, Holmgren -0.69,
  Brunson -0.69, Fox -0.66, Duren -0.59), a near-constant ~0.98x. That is not a scoring
  difference: it does not track double- or triple-double rate, and SGA, Castle, Hart, White
  (same bonuses, same teams) match exactly. The likeliest reason is games outside
  ``player_games`` that DK averages in (2026 preseason), which the database cannot confirm.
  The check reports the hit rate as is: ok >= 95%, warn >= 80%, fail below.
* Triple-double bonus: no opening-night draftable has >= 10 triple-doubles in the window (best:
  Castle 5), so FPPG cannot separate +4.5 from +3.0 yet (they differ by < 0.1 FPPG for every
  player here). If a player with >= `TD_PLAYER_MIN` triple-doubles is in the sample, the
  FPPG check fails when the +3.0 total fits better than +4.5. The discriminating evidence
  today is RotoGuru (`check_rotoguru`): on RotoGuru's own stat lines the +4.5 total is exact
  for the triple-doubles and the +3.0 total is not. The join to ``player_games`` is exact
  for ~98.6% of 2014-15 player-games; the rest are one-stat box-score differences between the
  two feeds (often a whole game), so the scoring verdict uses RotoGuru's own lines
  (>= 99.5% exact) and the join only has to reach `RG_JOIN_WARN`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ... import archive, config, injuries, teams
from ...mlops.checks import _res
from ...sources import dfs

log = logging.getLogger("courtside.proj.dfs.rules")

FPPG_CHECK = "dfs_scoring_fppg"
ROTOGURU_CHECK = "dfs_scoring_rotoguru"

MIN_CURRENT_GAMES = 5          # before this many games DK's FPPG is last season's mean
FPPG_TOL = 0.1                 # |recomputed - DK| for a "match"
FPPG_MIN_N = 50
FPPG_OK, FPPG_WARN = 0.95, 0.80
TD_PLAYER_MIN = 10             # triple-doubles in the window to discriminate the TD bonus
TD_MARGIN = 0.02               # MAE advantage (FP) the other TD total needs to flag a fail
RG_TOL = 0.01
RG_OK, RG_WARN = 0.995, 0.99
RG_JOIN_WARN = 0.98
RG_MIN_TD = 20

# Which positions a named lineup slot accepts. None = any player (UTIL, Showdown CPT/FLEX).
# This is DK's naming convention, not roster construction: slots, counts and the cap always
# come from the rules payload.
SLOT_POSITIONS: dict[str, frozenset[str] | None] = {
    "PG": frozenset({"PG"}), "SG": frozenset({"SG"}), "SF": frozenset({"SF"}),
    "PF": frozenset({"PF"}), "C": frozenset({"C"}),
    "G": frozenset({"PG", "SG"}), "F": frozenset({"SF", "PF"}),
    "UTIL": None, "CPT": None, "FLEX": None,
}


# ============================================================================== rules

@dataclass(frozen=True)
class Slot:
    """One lineup slot: DK roster-slot id, name, lineup order, accepted positions (None = any)
    and the multiplier applied to the occupant's salary and points (Showdown CPT: 1.5)."""
    id: int | None
    name: str
    order: int
    positions: frozenset[str] | None
    multiplier: float = 1.0

    def accepts(self, positions: tuple[str, ...] | list[str]) -> bool:
        return self.positions is None or bool(self.positions.intersection(positions))


@dataclass(frozen=True)
class Rules:
    """Lineup rules of one DK game type, read from an archived rules payload."""
    game_type_id: int | None
    name: str | None
    salary_cap: int
    min_games: int | None          # gameCount.min (None when DK disables it: Showdown)
    min_teams: int | None          # teamCount.min
    slots: tuple[Slot, ...]        # lineup order; a name repeats (Showdown UTIL x5)
    allow_late_swap: bool | None = None
    rules_url: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Rules":
        """From a raw rules payload (``lineupTemplate`` ...) or `sources.dfs.parse_rules`
        output. Unknown slot names raise ValueError (the eligibility would be a guess)."""
        parsed = payload if "slots" in payload else dfs.parse_rules(dict(payload))
        slots = []
        for s in parsed["slots"]:
            nm = s["name"]
            if nm not in SLOT_POSITIONS:
                raise ValueError(f"unknown DK roster slot {nm!r}")
            mult = 1.0
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x", s.get("note") or "", re.I)
            if m:
                mult = float(m.group(1))
            elif nm == "CPT":
                mult = dfs.SHOWDOWN_CAPTAIN_MULTIPLIER
            slots.append(Slot(s.get("id"), nm, int(s.get("order") or len(slots) + 1),
                              SLOT_POSITIONS[nm], mult))
        if not slots or not parsed.get("salary_cap"):
            raise ValueError("rules payload has no lineup slots or salary cap")
        return cls(game_type_id=parsed.get("game_type_id"), name=parsed.get("name"),
                   salary_cap=int(parsed["salary_cap"]), min_games=parsed.get("min_games"),
                   min_teams=parsed.get("min_teams"), slots=tuple(slots),
                   allow_late_swap=payload.get("allowLateSwap"),
                   rules_url=parsed.get("rules_url"))

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)

    @property
    def distinct_slot_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.slot_names))

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def is_showdown(self) -> bool:
        return any(s.name == "CPT" for s in self.slots)

    @property
    def captain_multiplier(self) -> float:
        return max((s.multiplier for s in self.slots), default=1.0)

    def slot_multiplier(self, slot_name: str) -> float:
        for s in self.slots:
            if s.name == slot_name:
                return s.multiplier
        raise KeyError(slot_name)

    def eligible_slots(self, positions: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Distinct slot names (lineup order) a player of `positions` ("PG/SG" or a list)
        may fill: PG -> PG, G, UTIL; PG/SG -> PG, SG, G, UTIL; C -> C, UTIL."""
        pos = split_positions(positions)
        if not pos:
            return ()
        return tuple(dict.fromkeys(s.name for s in self.slots if s.accepts(pos)))


def split_positions(positions: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not positions:
        return ()
    if isinstance(positions, str):
        positions = positions.split("/")
    return tuple(p.strip().upper() for p in positions if p and p.strip())


def _rules_dir(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    return p.parents[2] / "rules" / p.parent.name if len(p.parents) > 2 else None


def load_rules(conn: sqlite3.Connection, slate_id: str | int,
               game_type_id: int | None = None) -> Rules | None:
    """Rules of a slate: ``dfs_slates.rules_json`` when stored, otherwise the archived
    ``var/raw/draftkings/rules/<day>/gametype_<id>.json.gz`` next to the slate's draftables
    (falling back to the newest archive of that game type). None when neither exists."""
    sid = str(slate_id)
    try:
        r = conn.execute("SELECT rules_json, game_type_id FROM dfs_slates WHERE slate_id=?",
                         (sid,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    if r is not None:
        game_type_id = game_type_id or r[1]
        if r[0]:
            return Rules.from_payload(json.loads(r[0]))
    row = conn.execute("SELECT game_type_id, raw_path FROM dfs_salaries WHERE slate_id=? "
                       "ORDER BY captured_at DESC LIMIT 1", (sid,)).fetchone()
    if row is None:
        return None
    game_type_id = game_type_id or row[0]
    cands: list[Path] = []
    d = _rules_dir(row[1])
    if d is not None:
        cands.append(d / f"gametype_{game_type_id}.json.gz")
    root = config.data_dir() / "raw" / "draftkings" / "rules"
    cands += sorted(root.glob(f"*/gametype_{game_type_id}.json.gz"), reverse=True)
    for p in cands:
        if p.exists():
            return Rules.from_payload(archive.load_raw(p))
    return None


# ============================================================================== pool

def _parse_ts(s: str | None) -> datetime | None:
    """DK start times ('2026-10-21T01:30:00.0000000Z', 7 fractional digits) -> aware UTC."""
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", str(s))
    if not m:
        return None
    return datetime.strptime(f"{m[1]}T{m[2]}", "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _aware(now: datetime) -> datetime:
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, eq=False)
class PoolPlayer:
    """One DK player of a slate (all of his draftable rows merged by ``playerId``)."""
    dk_player_id: str
    dk_dk_id: str | None
    name: str
    team: str | None
    positions: tuple[str, ...]
    salary: int                             # non-captain salary
    slots: tuple[str, ...]                  # eligible slot names, lineup order
    salary_by_slot: Mapping[str, int]       # CPT carries 1.5x in Showdown
    draftable_ids: Mapping[str, str]        # slot name -> draftableId
    game_id: str | None
    competition_id: str | None
    game_label: str | None
    home: str | None
    away: str | None
    start_time: str | None
    status: str | None                      # None | Q | O | GTD
    news_status: str | None
    starter: bool | None
    swappable: bool | None
    fppg: float | None

    @property
    def start(self) -> datetime | None:
        return _parse_ts(self.start_time)

    @property
    def opp(self) -> str | None:
        return self.away if self.team == self.home else self.home

    def is_locked(self, now: datetime) -> bool:
        """True once his game's start time has passed (`now` must be timezone-aware or UTC)."""
        s = self.start
        return s is not None and s <= _aware(now)

    def salary_for(self, slot: str, rules: Rules | None = None) -> int:
        if slot in self.salary_by_slot:
            return self.salary_by_slot[slot]
        mult = rules.slot_multiplier(slot) if rules else 1.0
        return int(round(self.salary * mult))


@dataclass(frozen=True, eq=False)
class SlatePool:
    slate_id: str
    game_type_id: int | None
    game_type: str | None
    captured_at: str
    rules: Rules | None
    players: tuple[PoolPlayer, ...]

    def by_id(self) -> dict[str, PoolPlayer]:
        return {p.dk_player_id: p for p in self.players}

    def locked_ids(self, now: datetime) -> frozenset[str]:
        return frozenset(p.dk_player_id for p in self.players if p.is_locked(now))

    def open_players(self, now: datetime) -> tuple[PoolPlayer, ...]:
        return tuple(p for p in self.players if not p.is_locked(now))

    def eligible(self, slot: str) -> tuple[PoolPlayer, ...]:
        return tuple(p for p in self.players if slot in p.slots)

    @property
    def games(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in self.players:
            key = p.competition_id or p.game_id or p.game_label or ""
            out.setdefault(key, {"competition_id": p.competition_id, "game_id": p.game_id,
                                 "label": p.game_label, "home": p.home, "away": p.away,
                                 "start_time": p.start_time})
        return out


def latest_slates(conn: sqlite3.Connection, *, game_type_id: int | None = None,
                  day: date | str | None = None) -> list[dict]:
    """Slates at their latest capture: {slate_id, game_type_id, game_type, captured_at,
    n_players, first_start}, ordered by first game start. `day` keeps slates whose first game
    starts on that Eastern day."""
    sql = ("SELECT slate_id, game_type_id, game_type, captured_at, "
           "COUNT(DISTINCT site_player_id) AS n_players, MIN(start_time) AS first_start "
           "FROM dfs_salaries d WHERE captured_at = "
           "(SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id = d.slate_id)")
    args: list = []
    if game_type_id is not None:
        sql += " AND game_type_id = ?"
        args.append(game_type_id)
    sql += " GROUP BY slate_id ORDER BY first_start, slate_id"
    out = [dict(r) for r in conn.execute(sql, args)]
    if day is not None:
        d = day.isoformat() if isinstance(day, date) else str(day)[:10]
        out = [s for s in out if s["first_start"] and _slate_day(s["first_start"]).isoformat() == d]
    return out


def _slate_day(start_time: str) -> date:
    return config.nba_day(_parse_ts(start_time))


def _s(v) -> str | None:
    return None if v is None else str(v)


def build_pool(conn: sqlite3.Connection, slate_id: str | int, *,
               rules: Rules | None = None, captured_at: str | None = None) -> SlatePool:
    """The player pool of a slate from its latest ``dfs_salaries`` capture (or `captured_at`).

    One `PoolPlayer` per DK ``playerId``. Disabled rows are dropped (a player with only
    disabled rows is excluded). Eligible slots are the slots of his remaining draftable rows,
    DK's own listing; rows with no resolved slot name fall back to `Rules.eligible_slots` of
    his positions. `rules` default to `load_rules`."""
    sid = str(slate_id)
    if captured_at is None:
        captured_at = conn.execute("SELECT MAX(captured_at) FROM dfs_salaries WHERE slate_id=?",
                                   (sid,)).fetchone()[0]
    if captured_at is None:
        raise LookupError(f"no dfs_salaries rows for slate {sid}")
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM dfs_salaries WHERE slate_id=? AND captured_at=? AND disabled=0 "
        "ORDER BY id", (sid, captured_at))]
    gt = conn.execute("SELECT game_type_id, game_type FROM dfs_salaries WHERE slate_id=? LIMIT 1",
                      (sid,)).fetchone()
    if rules is None:
        rules = load_rules(conn, sid, gt[0] if gt else None)
    by_player: dict[str, list[dict]] = {}
    for r in rows:
        key = r["site_player_id"] or f"name:{r['player_name']}"
        by_player.setdefault(key, []).append(r)
    players = []
    for pid, rs in by_player.items():
        first = rs[0]
        positions = split_positions(first["positions"])
        listed = [r["roster_slot"] for r in rs if r["roster_slot"]]
        if listed:
            slots = tuple(dict.fromkeys(listed))
            if rules is not None:
                order = rules.distinct_slot_names
                slots = tuple(sorted(slots, key=lambda x: order.index(x) if x in order
                                     else len(order)))
        elif rules is not None:
            slots = rules.eligible_slots(positions)
        else:
            slots = ()
        sal = {r["roster_slot"]: r["salary"] for r in rs if r["roster_slot"]}
        base = next((r["salary"] for r in rs if r["roster_slot"] != "CPT"), first["salary"])
        st = first.get("starter")
        sw = first.get("swappable")
        players.append(PoolPlayer(
            dk_player_id=pid, dk_dk_id=first["site_player_dk_id"], name=first["player_name"],
            team=first["team"], positions=positions, salary=int(base), slots=slots,
            salary_by_slot=sal,
            draftable_ids={r["roster_slot"]: r["draftable_id"] for r in rs if r["roster_slot"]},
            game_id=first["game_id"], competition_id=first["competition_id"],
            game_label=first["game_label"], home=first["home"], away=first["away"],
            start_time=first["start_time"], status=_status(first["status"]),
            news_status=first["news_status"], starter=None if st is None else bool(st),
            swappable=None if sw is None else bool(sw), fppg=first["fppg"]))
    return SlatePool(slate_id=sid, game_type_id=gt[0] if gt else None,
                     game_type=gt[1] if gt else None, captured_at=captured_at, rules=rules,
                     players=tuple(players))


def _status(s: str | None) -> str | None:
    return None if s in (None, "", "None") else s


# ============================================================================== scoring

def _stat(row: Mapping | sqlite3.Row, key: str) -> float:
    try:
        v = row[key]
    except (KeyError, IndexError):
        return 0.0
    return 0.0 if v is None else float(v)


def double_double_cats(row: Mapping | sqlite3.Row) -> int:
    """How many of points, rebounds, assists, steals, blocks are >= 10."""
    return sum(1 for k in ("pts", "reb", "ast", "stl", "blk") if _stat(row, k) >= 10)


def dk_points(row: Mapping | sqlite3.Row, scoring: Mapping[str, float] | None = None) -> float:
    """DraftKings Classic fantasy points of a stat line (`player_games` columns pts, fg3m, reb,
    ast, stl, blk, tov; missing/None = 0). A double-double adds ``double_double`` (1.5) and a
    triple-double adds ``triple_double`` (3.0) on top of it: 4.5 in total. `scoring` overrides
    the table (default `sources.dfs.CLASSIC_SCORING`)."""
    sc = scoring or dfs.CLASSIC_SCORING
    total = (_stat(row, "pts") * sc["point"] + _stat(row, "fg3m") * sc["three_pointer_made"]
             + _stat(row, "reb") * sc["rebound"] + _stat(row, "ast") * sc["assist"]
             + _stat(row, "stl") * sc["steal"] + _stat(row, "blk") * sc["block"]
             + _stat(row, "tov") * sc["turnover"])
    cats = double_double_cats(row)
    if cats >= 2:
        total += sc["double_double"]
    if cats >= 3:
        total += sc["triple_double"]
    return total


# ============================================================================== shared match

_STAT_COLS = "pts, fg3m, reb, ast, stl, blk, tov, min"


def season_of(day: date) -> str:
    """NBA season label of a calendar day ('2026-10-20' -> '2026-27')."""
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{(y + 1) % 100:02d}"


def _franchise(code: str | None, season: str | None = None) -> str | None:
    if not code:
        return None
    try:
        return teams.franchise(code, season)
    except KeyError:
        return None


def _compact(name: str | None) -> str:
    return injuries.norm_name(name).replace(" ", "")


# ============================================================================== check 1: FPPG

def check_fppg(conn: sqlite3.Connection, *, day: date | None = None,
               slate_ids: list[str] | None = None, n_worst: int = 8) -> dict:
    """Recompute DK's FPPG for the draftables of the latest captured Classic slate(s) from
    `player_games` and compare (see the module docstring for the window).

    Players match by name within the team's last-game roster, then league-wide exact names
    (`injuries.match_player`). Window: all rows of the slate's season (any season type,
    zero-minute rows included) when the player has >= `MIN_CURRENT_GAMES` of them before the
    slate day, else all rows of the previous season. Returns a report dict: n, within,
    frac, window counts, worst misses, unmatched, and the triple-double evidence."""
    if slate_ids is None:
        slates = latest_slates(conn, game_type_id=70)
        if day is None and slates:
            last = max(_slate_day(s["first_start"]) for s in slates if s["first_start"])
            slates = [s for s in slates if s["first_start"]
                      and _slate_day(s["first_start"]) == last]
        elif day is not None:
            slates = [s for s in slates if s["first_start"]
                      and _slate_day(s["first_start"]) == day]
        slate_ids = [s["slate_id"] for s in slates]
        if slates and day is None:
            day = max(_slate_day(s["first_start"]) for s in slates if s["first_start"])
    if not slate_ids:
        return {"n": 0, "within": 0, "frac": None, "slates": [], "misses": [], "unmatched": [],
                "td": {"players": 0}, "window": None, "day": day.isoformat() if day else None}
    if day is None:
        day = config.today_nba_day()
    ph = ",".join("?" * len(slate_ids))
    caps = {r[0]: r[1] for r in conn.execute(
        f"SELECT slate_id, MAX(captured_at) FROM dfs_salaries WHERE slate_id IN ({ph}) "
        f"GROUP BY slate_id", slate_ids)}
    draftables: dict[str, dict] = {}
    for sid, cap in caps.items():
        for r in conn.execute(
                "SELECT site_player_id AS pid, player_name AS name, team, fppg FROM dfs_salaries "
                "WHERE slate_id=? AND captured_at=? AND disabled=0 AND fppg IS NOT NULL "
                "AND site_player_id IS NOT NULL", (sid, cap)):
            draftables.setdefault(r["pid"], dict(r))

    cur = season_of(day)
    prev = prev_season(cur)
    games: dict[int, dict[str, list[sqlite3.Row]]] = {}
    last: dict[int, tuple[str, str, str]] = {}      # pid -> (date, name, team)
    for r in conn.execute(
            f"SELECT player_id, player_name, team, season, game_date, {_STAT_COLS} "
            f"FROM player_games WHERE season IN (?, ?) AND game_date < ?",
            (cur, prev, day.isoformat())):
        pid = int(r["player_id"])
        games.setdefault(pid, {}).setdefault(r["season"], []).append(r)
        if pid not in last or r["game_date"] > last[pid][0]:
            last[pid] = (r["game_date"], r["player_name"], _franchise(r["team"], r["season"]))
    roster: dict[str, list[tuple[int, str]]] = {}
    for pid, (_, name, team) in last.items():
        roster.setdefault(team, []).append((pid, name))
    league = [(pid, v[1]) for pid, v in last.items()]

    td_variants = {"td_4.5": dfs.CLASSIC_SCORING,
                   "td_3.0": {**dfs.CLASSIC_SCORING, "triple_double": 1.5}}
    n = within = 0
    misses, unmatched = [], []
    td_rows = []
    for d in draftables.values():
        pid, how = injuries.match_player(d["name"], roster.get(d["team"], []), None, league)
        if pid is None or pid not in games:
            unmatched.append({"name": d["name"], "team": d["team"], "fppg": d["fppg"],
                              "why": how or "no match"})
            continue
        by_season = games[pid]
        rows = by_season.get(cur, [])
        used = cur
        if len(rows) < MIN_CURRENT_GAMES:
            rows, used = by_season.get(prev, []), prev
        if not rows:
            unmatched.append({"name": d["name"], "team": d["team"], "fppg": d["fppg"],
                              "why": "no games in window"})
            continue
        mean = sum(dk_points(r) for r in rows) / len(rows)
        err = mean - d["fppg"]
        n += 1
        ok = abs(err) <= FPPG_TOL + 1e-9
        within += ok
        tds = sum(1 for r in rows if double_double_cats(r) >= 3)
        if not ok:
            misses.append({"name": d["name"], "team": d["team"], "fppg": d["fppg"],
                           "mean": round(mean, 2), "delta": round(err, 2), "games": len(rows),
                           "season": used, "tds": tds})
        if tds >= 1:
            td_rows.append({"name": d["name"], "tds": tds, "games": len(rows), "fppg": d["fppg"],
                            **{k: sum(dk_points(r, sc) for r in rows) / len(rows)
                               for k, sc in td_variants.items()}})
    big = [t for t in td_rows if t["tds"] >= TD_PLAYER_MIN]
    mae = {k: (sum(abs(t[k] - t["fppg"]) for t in big) / len(big)) if big else None
           for k in td_variants}
    td = {"players": len(big), "min_tds": TD_PLAYER_MIN, "mae": mae,
          "players_with_any_td": len(td_rows),
          "max_tds": max((t["tds"] for t in td_rows), default=0),
          "best_td_player": max(td_rows, key=lambda t: t["tds"], default=None),
          "verdict": None}
    if big:
        better_45 = mae["td_4.5"] <= mae["td_3.0"] - TD_MARGIN
        better_30 = mae["td_3.0"] <= mae["td_4.5"] - TD_MARGIN
        td["verdict"] = "4.5" if better_45 else "3.0" if better_30 else "inconclusive"
    misses.sort(key=lambda m: -abs(m["delta"]))
    return {"n": n, "within": within, "frac": within / n if n else None, "slates": slate_ids,
            "day": day.isoformat(), "window": f"{cur} (>= {MIN_CURRENT_GAMES} games) else {prev}",
            "misses": misses[:n_worst], "n_miss": len(misses), "unmatched": unmatched,
            "td": td}


def fppg_check_row(rep: dict) -> dict:
    """`data_checks` row for a `check_fppg` report."""
    n, frac = rep["n"], rep["frac"]
    td = rep["td"]
    if n == 0:
        return _res(FPPG_CHECK, "skip", detail="no captured Classic slate with DK FPPG")
    bits = [f"{rep['within']}/{n} within {FPPG_TOL} FP of DK FPPG ({frac:.1%}), "
            f"window {rep['window']}, {len(rep['unmatched'])} unmatched"]
    if rep["misses"]:
        w = rep["misses"][0]
        bits.append(f"worst {w['name']} {w['delta']:+.2f}")
    if td["players"]:
        bits.append(f"TD bonus ({td['players']} players with >= {td['min_tds']} TDs): "
                    f"MAE 4.5={td['mae']['td_4.5']:.3f} vs 3.0={td['mae']['td_3.0']:.3f} "
                    f"-> {td['verdict']}")
    else:
        bits.append(f"TD bonus not testable: max {td['max_tds']} TDs for any draftable "
                    f"(need {td['min_tds']})")
    if td["verdict"] == "3.0":
        status = "fail"
    elif n < FPPG_MIN_N:
        status = "warn"
        bits.append(f"only {n} draftables (< {FPPG_MIN_N})")
    elif frac >= FPPG_OK:
        status = "ok"
    elif frac >= FPPG_WARN:
        status = "warn"
    else:
        status = "fail"
    return _res(FPPG_CHECK, status, frac, FPPG_OK, "; ".join(bits))


# ============================================================================== check 2: RotoGuru

_RG_STAT = re.compile(r"(\d+)(pt|rb|as|st|bl|to|trey)\b")
_RG_KEYS = {"pt": "pts", "rb": "reb", "as": "ast", "st": "stl", "bl": "blk", "to": "tov",
            "trey": "fg3m"}
TD_ALT = {**dfs.CLASSIC_SCORING, "triple_double": 1.5}       # +3.0 in total instead of +4.5


def parse_rg_stats(raw: str | None) -> dict[str, int] | None:
    """RotoGuru's stat string ('24pt 10rb 5as 6st 4to 2trey 7-17fg') -> player_games keys."""
    if not raw:
        return None
    out = {k: 0 for k in _RG_KEYS.values()}
    found = False
    for v, k in _RG_STAT.findall(raw):
        out[_RG_KEYS[k]] = int(v)
        found = True
    return out if found else None


def check_rotoguru(conn: sqlite3.Connection, *, since: str | None = None,
                   until: str | None = None, n_worst: int = 8) -> dict:
    """Compare RotoGuru's ``dk_pts`` (`dfs_hist_salaries`) with `dk_points` of the matching
    ``player_games`` row: same game date, same franchise and same normalised name (a unique
    same-day name match without the team, and the row's ``player_id`` when set, are fallbacks
    for traded and oddly spelled players). Exact = within `RG_TOL`. Works on whatever rows
    exist; returns n, exact, frac, triple-doubles among the joined rows, worst misses.

    ``own``: the same comparison using RotoGuru's *own* stat string (``raw_stats``), which needs
    no join and so isolates the scoring table from box-score differences between the two feeds
    (a rebound or assist credited differently moves the points by 1.25-1.5). ``td``: among
    triple-doubles, how many are exact with the +4.5 total and with +3.0."""
    try:
        sql = ("SELECT game_date, rg_id, player_id, name, team, dk_pts, raw_stats "
               "FROM dfs_hist_salaries "
               "WHERE dk_pts IS NOT NULL")
        args: list = []
        if since:
            sql += " AND game_date >= ?"
            args.append(since)
        if until:
            sql += " AND game_date <= ?"
            args.append(until)
        hist = [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.OperationalError:
        hist = []
    own = {"n": 0, "exact": 0, "frac": None, "tds": 0, "tds_exact_45": 0, "tds_exact_30": 0,
           "misses": []}
    base = {"n_rows": len(hist), "n": 0, "exact": 0, "frac": None, "tds": 0, "tds_exact": 0,
            "tds_exact_30": 0, "unmatched": 0, "misses": [], "n_miss": 0, "dates": None,
            "by": {}, "own": own}
    if not hist:
        return base
    for h in hist:
        st = parse_rg_stats(h["raw_stats"])
        if st is None:
            continue
        mine, rg = dk_points(st), float(h["dk_pts"])
        ok = abs(mine - rg) <= RG_TOL + 1e-9
        own["n"] += 1
        own["exact"] += ok
        if double_double_cats(st) >= 3:
            own["tds"] += 1
            own["tds_exact_45"] += ok
            own["tds_exact_30"] += abs(dk_points(st, TD_ALT) - rg) <= RG_TOL + 1e-9
        elif not ok and len(own["misses"]) < 8:
            own["misses"].append({"date": h["game_date"], "name": h["name"], "rg": rg,
                                  "mine": round(mine, 2), "stats": h["raw_stats"]})
    own["frac"] = own["exact"] / own["n"] if own["n"] else None
    lo, hi = min(h["game_date"] for h in hist), max(h["game_date"] for h in hist)
    base["dates"] = [lo, hi]
    by_team: dict[tuple[str, str | None, str], list] = {}
    by_day: dict[tuple[str, str], list] = {}
    by_pid: dict[tuple[str, int], sqlite3.Row] = {}
    for r in conn.execute(
            f"SELECT game_id, player_id, player_name, team, season, game_date, {_STAT_COLS} "
            f"FROM player_games WHERE game_date BETWEEN ? AND ?", (lo, hi)):
        fr = _franchise(r["team"], r["season"])
        for nm in {injuries.norm_name(r["player_name"]), _compact(r["player_name"])}:
            by_team.setdefault((r["game_date"], fr, nm), []).append(r)
            by_day.setdefault((r["game_date"], nm), []).append(r)
        by_pid[(r["game_date"], int(r["player_id"]))] = r
    season_hint = lambda d: season_of(date.fromisoformat(d))    # noqa: E731
    n = exact = tds = tds_exact = tds_exact_30 = unmatched = 0
    by_how: dict[str, int] = {}
    misses = []
    for h in hist:
        fr = _franchise(h["team"], season_hint(h["game_date"]))
        cand, how = None, None
        for nm in (injuries.norm_name(h["name"]), _compact(h["name"])):
            c = by_team.get((h["game_date"], fr, nm))
            if c and len(c) == 1:
                cand, how = c[0], "team+name"
                break
        if cand is None:
            for nm in (injuries.norm_name(h["name"]), _compact(h["name"])):
                c = by_day.get((h["game_date"], nm))
                if c and len(c) == 1:
                    cand, how = c[0], "day+name"
                    break
        if cand is None and h["player_id"] is not None:
            cand = by_pid.get((h["game_date"], int(h["player_id"])))
            how = "player_id" if cand is not None else None
        if cand is None:
            unmatched += 1
            continue
        by_how[how] = by_how.get(how, 0) + 1
        mine = dk_points(cand)
        ok = abs(mine - float(h["dk_pts"])) <= RG_TOL + 1e-9
        td = double_double_cats(cand) >= 3
        n += 1
        exact += ok
        tds += td
        tds_exact += td and ok
        tds_exact_30 += td and abs(dk_points(cand, TD_ALT) - float(h["dk_pts"])) <= RG_TOL + 1e-9
        if not ok:
            misses.append({"date": h["game_date"], "name": h["name"], "team": h["team"],
                           "rg": float(h["dk_pts"]), "mine": round(mine, 2),
                           "delta": round(mine - float(h["dk_pts"]), 2), "td": td})
    misses.sort(key=lambda m: -abs(m["delta"]))
    base.update({"n": n, "exact": exact, "frac": exact / n if n else None, "tds": tds,
                 "tds_exact": tds_exact, "tds_exact_30": tds_exact_30, "unmatched": unmatched, "misses": misses[:n_worst],
                 "n_miss": len(misses), "by": by_how})
    return base


def rotoguru_check_row(rep: dict) -> dict:
    """`data_checks` row for a `check_rotoguru` report.

    The scoring table is judged on RotoGuru's own stat strings (`RG_OK` exact, `RG_WARN`
    floor) because the join to `player_games` also carries box-score differences between the
    two feeds (a rebound or assist credited to someone else: ~1.4% of 2014-15 player-games, in
    bursts of whole games), which a scoring change cannot cause: the join is judged against
    `RG_JOIN_WARN`. Without stat strings the join is judged on `RG_OK` / `RG_WARN`.
    fail: below `RG_WARN`, or the +3.0 triple-double total fits better than +4.5;
    warn: below `RG_OK` / `RG_JOIN_WARN`, or fewer than `RG_MIN_TD` triple-doubles."""
    n, frac, own = rep["n"], rep["frac"], rep["own"]
    if n == 0 and not own["n"]:
        return _res(ROTOGURU_CHECK, "skip",
                    detail=f"no dfs_hist_salaries rows ({rep['n_rows']} rows, none joinable)")
    bits = []
    if n:
        bits.append(f"{rep['exact']}/{n} joined player-games equal (+-{RG_TOL}) ({frac:.2%}), "
                    f"{rep['tds']} triple-doubles ({rep['tds_exact']} exact at +4.5, "
                    f"{rep['tds_exact_30']} at +3.0); {rep['unmatched']} of {rep['n_rows']} "
                    f"rows unmatched; {rep['dates'][0]} -> {rep['dates'][1]}")
    if own["n"]:
        bits.append(f"RotoGuru's own stat lines: {own['exact']}/{own['n']} equal "
                    f"({own['frac']:.2%}), {own['tds']} triple-doubles "
                    f"({own['tds_exact_45']} exact at +4.5, {own['tds_exact_30']} at +3.0)")
    if rep["misses"]:
        w = rep["misses"][0]
        bits.append(f"worst {w['date']} {w['name']} rg {w['rg']} vs {w['mine']}")
    tds = max(rep["tds"], own["tds"])
    ok_floor, warn_floor = (RG_OK, RG_WARN)
    main = own["frac"] if own["n"] else frac
    status = "ok"
    if main < warn_floor:
        status = "fail"
    elif main < ok_floor or (own["n"] and n and frac < RG_JOIN_WARN):
        status = "warn"
    if status == "ok" and tds < RG_MIN_TD:
        status = "warn"
        bits.append(f"only {tds} triple-doubles (< {RG_MIN_TD})")
    if own["tds"] and own["tds_exact_30"] > own["tds_exact_45"]:
        status = "fail"
        bits.append("the +3.0 triple-double total fits better than +4.5")
    return _res(ROTOGURU_CHECK, status, main, RG_OK, "; ".join(bits))


# ============================================================================== runner

def store_checks(conn: sqlite3.Connection, rows: list[dict], now: datetime | None = None) -> None:
    """Write check rows into `data_checks` (the same insert `mlops.checks.run_checks` does)."""
    run_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO data_checks (run_at, check_name, status, value, threshold, "
            "detail) VALUES (?, ?, ?, ?, ?, ?)",
            [(run_at, r["check_name"], r["status"], r["value"], r["threshold"], r["detail"])
             for r in rows])


def verify(conn: sqlite3.Connection, *, now: datetime | None = None, store: bool = True,
           day: date | None = None) -> tuple[list[dict], dict]:
    """Run both scoring checks -> (data_check rows, {"fppg": report, "rotoguru": report})."""
    fppg = check_fppg(conn, day=day)
    rg = check_rotoguru(conn)
    rows = [fppg_check_row(fppg), rotoguru_check_row(rg)]
    if store:
        store_checks(conn, rows, now)
    return rows, {"fppg": fppg, "rotoguru": rg}


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("rules", help="print the parsed DK rules and the pool of a slate")
    p.add_argument("--slate", help="draft group id (default: every slate of the latest capture)")
    p = sub.add_parser("check-scoring",
                       help="verify the DK scoring table (FPPG reconciliation + RotoGuru)")
    p.add_argument("--no-store", action="store_true", help="do not write data_checks rows")
    return {"rules": _rules_cmd, "check-scoring": _check_scoring_cmd}


def _conn():
    from ... import db
    return db.connect()


def _print_rules(r: Rules) -> None:
    mults = {s.name: s.multiplier for s in r.slots if s.multiplier != 1.0}
    print(f"  {r.name} (gameType {r.game_type_id}): cap ${r.salary_cap:,}, "
          f"min games {r.min_games}, min teams {r.min_teams}, late swap {r.allow_late_swap}")
    print(f"  slots: {' '.join(r.slot_names)}" + (f"   multipliers {mults}" if mults else ""))


def _rules_cmd(args) -> int:
    conn = _conn()
    if args.slate:
        slates = [{"slate_id": args.slate}]
    else:
        slates = latest_slates(conn)
    if not slates:
        print("no dfs_salaries captured yet")
        return 1
    for s in slates:
        pool = build_pool(conn, s["slate_id"])
        print(f"slate {pool.slate_id} ({pool.game_type}) captured {pool.captured_at}")
        if pool.rules is None:
            print("  rules: not found (no dfs_slates.rules_json, no archive)")
        else:
            _print_rules(pool.rules)
        now = datetime.now(timezone.utc)
        locked = pool.locked_ids(now)
        statuses: dict[str, int] = {}
        for p in pool.players:
            statuses[p.status or "healthy"] = statuses.get(p.status or "healthy", 0) + 1
        print(f"  pool: {len(pool.players)} players, {len(pool.games)} games, "
              f"{len(locked)} locked now; status {statuses}")
        if pool.rules:
            for nm in pool.rules.distinct_slot_names:
                print(f"    {nm:<5} {len(pool.eligible(nm)):>3} eligible")
        for g in pool.games.values():
            print(f"    {g['label']}  {g['start_time']}")
    return 0


def _print_fppg(rep: dict) -> None:
    if not rep["n"]:
        print("  FPPG: nothing to compare")
        return
    print(f"  FPPG: {rep['within']}/{rep['n']} within {FPPG_TOL} ({rep['frac']:.1%}) "
          f"slates {rep['slates']} day {rep['day']}; window {rep['window']}; "
          f"unmatched {len(rep['unmatched'])}")
    for m in rep["misses"]:
        print(f"    miss {m['name']:<24} dk {m['fppg']:>5} ours {m['mean']:>6} "
              f"({m['delta']:+.2f}) {m['games']} games {m['season']} {m['tds']} TD")
    td = rep["td"]
    if td["players"]:
        print(f"  TD bonus: {td['players']} players with >= {td['min_tds']} TDs: "
              f"MAE total 4.5 = {td['mae']['td_4.5']:.3f}, total 3.0 = {td['mae']['td_3.0']:.3f}"
              f" -> {td['verdict']}")
    else:
        b = td["best_td_player"]
        print(f"  TD bonus: no draftable with >= {td['min_tds']} TDs "
              f"(most: {b['name']} {b['tds']})" if b else "  TD bonus: no draftable with a TD")


def _print_rg(rep: dict) -> None:
    own = rep["own"]
    if not rep["n"] and not own["n"]:
        print(f"  RotoGuru: no usable rows ({rep['n_rows']} in dfs_hist_salaries)")
        return
    if rep["n"]:
        print(f"  RotoGuru vs player_games: {rep['exact']}/{rep['n']} exact ({rep['frac']:.3%}), "
              f"{rep['tds']} triple-doubles ({rep['tds_exact']} exact at +4.5, "
              f"{rep['tds_exact_30']} at +3.0), {rep['unmatched']} of {rep['n_rows']} "
              f"unmatched, {rep['dates'][0]} -> {rep['dates'][1]}, joins {rep['by']}")
        for m in rep["misses"]:
            print(f"    miss {m['date']} {m['name']:<24} rg {m['rg']:>6} ours {m['mine']:>6} "
                  f"({m['delta']:+.2f}){' TD' if m['td'] else ''}")
    if own["n"]:
        print(f"  RotoGuru vs its own stat lines: {own['exact']}/{own['n']} exact "
              f"({own['frac']:.3%}), {own['tds']} triple-doubles ({own['tds_exact_45']} exact at "
              f"+4.5, {own['tds_exact_30']} at +3.0)")
        for m in own["misses"][:5]:
            print(f"    miss {m['date']} {m['name']:<24} rg {m['rg']:>6} ours {m['mine']:>6} "
                  f"{m['stats']}")


def _check_scoring_cmd(args) -> int:
    conn = _conn()
    rows, reps = verify(conn, store=not args.no_store)
    print("DraftKings scoring verification")
    for r in rows:
        print(f"  [{r['status']:<4}] {r['check_name']:<22} {r['detail']}")
    _print_fppg(reps["fppg"])
    _print_rg(reps["rotoguru"])
    return 1 if any(r["status"] == "fail" for r in rows) else 0
