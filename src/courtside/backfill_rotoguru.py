"""RotoGuru DraftKings history backfill (2014-10-29 .. 2021-05) into ``dfs_hist_salaries``.

Same pattern as backfill_players.py: every page is archived (gzipped HTML at
``raw/rotoguru/dk/<YYYY-MM-DD>.html.gz``) before it is parsed, an archived date is never fetched
again, and parsing always works from the archive, so ``--reparse`` rebuilds the table with no
network and a killed run simply resumes. One sequential request per game day, 1 request/s
(``sources.rotoguru``); days come from ``team_games`` (regular season, plus Playoffs / PlayIn with
``playoffs=True``), so off days are never requested.

Per row ``game_id`` comes from ``team_games`` by (date, team) and ``on_slate`` from the slate rule
(PLAYERS.md section 3.1): per team, the share of rotation players (``player_games`` minutes >= 10)
that carry a salary on the page; a game is on the slate when both teams reach 80%. RotoGuru lists
off-slate games too, but with salary N/A, so the rule is a direct test of "was this game on DK".
A secondary proxy (tip at or after 19:00 ET, from ``games.tipoff_utc``) is only logged: the
``games`` table starts in 2026, so it has no tip times for this history and the comparison counts
zero games unless that gets backfilled. ``player_id`` is left NULL (T0.5 fills it) and survives a
reparse.

Failures: a page that is not the complete page for its date is retried, then counted as failed
and left for the next run; after MAX_CONSECUTIVE_FAILS days in a row the run aborts (the site is
down or blocking us). Progress goes to ``var/logs/backfill_rotoguru.log``.
"""

from __future__ import annotations

import gzip
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from . import archive, config, db
from .injuries import norm_name
from .sources import rotoguru as rg

SOURCE, KIND = rg.SOURCE, rg.KIND
LOG_NAME = "backfill_rotoguru.log"
TABLE = "dfs_hist_salaries"
DEFAULT_START = rg.FIRST_DAY
DEFAULT_END = "2021-07-31"
SEASON_TYPES = ("Regular Season",)
PLAYOFF_TYPES = ("Playoffs", "PlayIn")
ROTATION_MIN = 10.0        # minutes for a "rotation" player
SLATE_SHARE = 0.80         # share of a team's rotation that must carry a salary
MAIN_SLATE_TIP_ET = 19     # secondary proxy: tip at or after 19:00 ET
MAX_CONSECUTIVE_FAILS = 5
ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------------ logging

def log_path() -> Path:
    return config.data_dir() / "logs" / LOG_NAME


def _log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------------ archive

def page_path(day: str) -> Path:
    """raw/rotoguru/dk/<day>.html.gz (an empty ``nba_day`` drops the per-day directory)."""
    return archive.raw_path(SOURCE, KIND, "", day, ext=".html.gz")


def save_page(day: str, data: bytes) -> Path:
    return archive.save_raw_bytes(SOURCE, KIND, "", day, gzip.compress(data), ext=".html.gz")


def load_page(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def archived_days() -> dict[str, Path]:
    """day -> archived page path."""
    root = page_path("x").parent
    return {p.name[: -len(".html.gz")]: p for p in root.glob("*.html.gz")} if root.exists() else {}


def pending_days(days: list[str], archived) -> list[str]:
    """The days that still need a request: everything not archived yet."""
    return [d for d in days if d not in archived]


def game_days(conn, start: str | None = None, end: str | None = None,
              playoffs: bool = False) -> list[str]:
    """Days with at least one game in ``team_games`` (what RotoGuru can have a page for)."""
    types = SEASON_TYPES + (PLAYOFF_TYPES if playoffs else ())
    q = ",".join("?" * len(types))
    rows = conn.execute(
        f"SELECT DISTINCT game_date FROM team_games WHERE season_type IN ({q}) "
        "AND game_date BETWEEN ? AND ? ORDER BY game_date",
        (*types, max(start or DEFAULT_START, DEFAULT_START), end or DEFAULT_END))
    return [r[0] for r in rows]


# ------------------------------------------------------------------------ enrich

def pair_names(left: set[str], right: set[str]) -> dict[str, tuple[str, int]]:
    """One-to-one pairing of two sets of ``norm_name`` strings from the same team and day:
    {left: (right, tier)}. Tier 0 = equal; 1 = equal ignoring spaces ("j r smith" / "jr smith");
    2 = same last token and first letter ("wes matthews" / "wesley matthews"); 3 = same first or
    last token among the leftovers ("enes kanter" / "enes freedom", "nene hilario" / "nene").
    Alias gaps beyond that are left to the T0.5 id matcher."""
    out: dict[str, tuple[str, int]] = {}
    left, right = set(left), set(right)

    def sweep(tier: int, same) -> None:
        for a in sorted(left):
            hits = [b for b in sorted(right) if same(a, b)]
            if len(hits) == 1 or (tier == 0 and hits):
                out[a] = (hits[0], tier)
                right.discard(hits[0])
        left.difference_update(out)

    sweep(0, lambda a, b: a == b)
    sweep(1, lambda a, b: a.replace(" ", "") == b.replace(" ", ""))
    sweep(2, lambda a, b: a.split()[-1:] == b.split()[-1:] and a[:1] == b[:1])
    sweep(3, lambda a, b: a.split()[:1] == b.split()[:1] or a.split()[-1:] == b.split()[-1:])
    return out


def tip_proxy(conn, game_id: str) -> bool | None:
    """True when the game tips at or after 19:00 ET (``games.tipoff_utc``); None if unknown."""
    r = conn.execute("SELECT tipoff_utc FROM games WHERE game_id=?", (game_id,)).fetchone()
    if not r or not r[0]:
        return None
    tip = datetime.fromisoformat(r[0].replace("Z", "+00:00")).astimezone(ET)
    return tip.hour >= MAIN_SLATE_TIP_ET


def slate_flags(conn, day: str, rows: list[dict]) -> tuple[dict[str, str], dict[str, dict]]:
    """({team: game_id}, {game_id: {"on_slate", "share": {team: share}, "tip_et"}}) for ``day``.

    share = rotation players (minutes >= ROTATION_MIN) who pair (``pair_names``) with a salaried
    row of that team on the page, over all rotation players of the team."""
    game_of: dict[str, str] = {}
    sides: dict[str, list[str]] = defaultdict(list)
    for r in conn.execute("SELECT game_id, team FROM team_games WHERE game_date=?", (day,)):
        game_of[r["team"]] = r["game_id"]
        sides[r["game_id"]].append(r["team"])
    salaried: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["salary"] is not None and r["team"]:
            salaried[r["team"]].add(norm_name(r["name"]))
    rotation: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in conn.execute("SELECT game_id, team, player_name FROM player_games "
                          "WHERE game_date=? AND min >= ?", (day, ROTATION_MIN)):
        rotation[(r["game_id"], r["team"])].add(norm_name(r["player_name"]))
    games: dict[str, dict] = {}
    for gid, tms in sides.items():
        share = {}
        for t in tms:
            rot = rotation[(gid, t)]
            share[t] = len(pair_names(rot, salaried[t])) / len(rot) if rot else 0.0
        games[gid] = {"share": share, "tip_et": tip_proxy(conn, gid),
                      "on_slate": int(len(tms) == 2 and all(v >= SLATE_SHARE
                                                            for v in share.values()))}
    return game_of, games


def annotate(conn, day: str, rows: list[dict], path: Path | str | None) -> dict[str, dict]:
    """Fill game_id / on_slate / raw_path in place and carry over an existing player_id.
    Returns the per-game slate info (``slate_flags``)."""
    game_of, games = slate_flags(conn, day, rows)
    known = {r["rg_id"]: r["player_id"] for r in conn.execute(
        f"SELECT rg_id, player_id FROM {TABLE} WHERE game_date=? AND player_id IS NOT NULL",
        (day,))}
    for r in rows:
        gid = game_of.get(r["team"])
        r["game_id"] = gid
        r["on_slate"] = games[gid]["on_slate"] if gid else None
        r["player_id"] = known.get(r["rg_id"])
        r["raw_path"] = str(path) if path is not None else None
    return games


def parse_day(conn, day: str, data: bytes, path: Path | str | None) -> tuple[list[dict], dict]:
    """Parse one archived page into annotated rows (duplicate rg_ids dropped) + slate info."""
    seen: set[str] = set()
    rows = []
    for r in rg.parse_page(data, day):
        if r["rg_id"] not in seen:
            seen.add(r["rg_id"])
            rows.append(r)
    return rows, annotate(conn, day, rows, path)


# ------------------------------------------------------------------------ run

def run(start: str | None = None, end: str | None = None, reparse: bool = False,
        playoffs: bool = False, conn=None, fetch: Callable[[str], bytes] | None = None,
        log_every: int = 50) -> dict:
    """Fetch (archive first) and parse the game days in [start, end].

    ``reparse`` parses every archived day in range and never touches the network; otherwise missing
    days are fetched and archived, and archived days with no rows yet are parsed. Returns
    {"days", "fetched", "cached", "reparsed", "missing", "empty", "failed", "rows", "games",
    "games_on_slate", "tip_compared", "tip_agree", "errors", "aborted", "run_id"}."""
    own = conn is None
    conn = conn or db.connect()
    fetch = fetch or rg.fetch_page
    run_id = db.start_run(conn, "backfill_rotoguru")
    s: dict = {"days": 0, "fetched": 0, "cached": 0, "reparsed": 0, "missing": 0, "empty": 0,
               "failed": 0, "rows": 0, "games": 0, "games_on_slate": 0, "tip_compared": 0,
               "tip_agree": 0, "errors": [], "aborted": False, "run_id": run_id}
    try:
        days = game_days(conn, start, end, playoffs)
        have = archived_days()
        done = {r[0] for r in conn.execute(f"SELECT DISTINCT game_date FROM {TABLE}")}
        s["days"] = len(days)
        todo = 0 if reparse else len(pending_days(days, have))
        _log(f"rotoguru: {len(days)} game days {days[0] if days else '-'}..{days[-1] if days else '-'}"
             f", {len(days) - len(pending_days(days, have))} archived, "
             f"{'reparse only' if reparse else f'{todo} to fetch'}")
        t0, fetched, fails = time.monotonic(), 0, 0
        for day in days:
            just_fetched = False
            try:
                path = have.get(day)
                if path is not None:
                    if day in done and not reparse:
                        s["cached"] += 1
                        continue
                    data = load_page(path)
                    s["reparsed"] += 1
                elif reparse:
                    s["missing"] += 1
                    continue
                else:
                    data = fetch(day)
                    rg.validate_page(data, day)
                    path = save_page(day, data)
                    have[day] = path
                    fetched += 1
                    s["fetched"] += 1
                    just_fetched, fails = True, 0
                rows, games = parse_day(conn, day, data, path)
                if rows:
                    s["rows"] += db.upsert_rows(conn, TABLE, rows)
                else:
                    s["empty"] += 1
                    _log(f"  no rows on {day}")
                _tally(s, games)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - keep going; a rerun resumes the failures
                s["failed"] += 1
                fails += 1
                s["errors"].append(f"{day}: {type(e).__name__}: {e}")
                _log(f"  FAILED {day}: {type(e).__name__}: {e}")
                if fails >= MAX_CONSECUTIVE_FAILS:
                    s["aborted"] = True
                    _log(f"  aborting after {fails} consecutive failures; rerun to resume")
                    break
                continue
            if just_fetched and fetched % log_every == 0:
                el = time.monotonic() - t0
                eta = (todo - fetched) * el / fetched / 60
                _log(f"  {day} fetched={fetched}/{todo} rows={s['rows']} failed={s['failed']} "
                     f"{fetched / el * 60:.0f}/min eta {eta:.1f} min")
        db.finish_run(conn, run_id, not s["failed"] and not s["aborted"], _detail(s))
    except BaseException as e:
        db.finish_run(conn, run_id, False, f"aborted: {type(e).__name__}: {e}")
        raise
    finally:
        if own:
            conn.close()
    _log(f"rotoguru done: {_detail(s)}")
    return s


def _tally(s: dict, games: dict[str, dict]) -> None:
    for g in games.values():
        s["games"] += 1
        s["games_on_slate"] += g["on_slate"]
        if g["tip_et"] is not None:
            s["tip_compared"] += 1
            s["tip_agree"] += int(g["tip_et"] == bool(g["on_slate"]))


def _detail(s: dict) -> str:
    return (f"days={s['days']} fetched={s['fetched']} cached={s['cached']} "
            f"reparsed={s['reparsed']} missing={s['missing']} empty={s['empty']} "
            f"failed={s['failed']} rows={s['rows']} games={s['games']} "
            f"on_slate={s['games_on_slate']} tip_compared={s['tip_compared']} "
            f"tip_agree={s['tip_agree']}"
            + (f" errors={s['errors'][:5]}" if s["errors"] else ""))


# ------------------------------------------------------------------------ summary

def summarize(conn, start: str | None = None, end: str | None = None,
              playoffs: bool = False, examples: int = 12) -> dict:
    """Quality report over what is in ``dfs_hist_salaries`` for [start, end]: coverage of game
    days, game_id resolution, join to ``player_games`` by (date, team, norm_name), DK-points
    recompute, triple-doubles and slate stats."""
    days = game_days(conn, start, end, playoffs)
    lo, hi = (days[0], days[-1]) if days else ("", "")
    have = {d for d in archived_days() if lo <= d <= hi}
    out: dict = {"days_expected": len(days), "days_archived": len(have & set(days)),
                 "days_with_rows": 0}
    rows = conn.execute(
        f"SELECT * FROM {TABLE} WHERE game_date BETWEEN ? AND ?", (lo, hi)).fetchall()
    out["days_with_rows"] = len({r["game_date"] for r in rows} & set(days))
    out["rows"] = len(rows)
    out["game_id_resolved"] = sum(r["game_id"] is not None for r in rows)
    box: dict[tuple, set] = defaultdict(set)
    for r in conn.execute("SELECT game_date, team, player_name FROM player_games "
                          "WHERE game_date BETWEEN ? AND ?", (lo, hi)):
        box[(r["game_date"], r["team"])].add(norm_name(r["player_name"]))
    played = [r for r in rows if (r["minutes"] or 0) > 0]   # DNP / inactive have no box row
    groups: dict[tuple, list] = defaultdict(list)
    for r in played:
        groups[(r["game_date"], r["team"])].append(r)
    tiers: Counter = Counter()
    causes: Counter = Counter()
    missed: dict[str, Counter] = defaultdict(Counter)
    for key, grp in groups.items():
        names = box.get(key, set())
        pairs = pair_names({norm_name(r["name"]) for r in grp}, names)
        for r in grp:
            hit = pairs.get(norm_name(r["name"]))
            if hit:
                tiers[hit[1]] += 1
                continue
            why = "not in box score" if names else "team has no box score that day"
            causes[why] += 1
            missed[why][f"{r['name']} ({r['team']})"] += 1
    out["played_rows"] = len(played)
    out["played_matched"] = sum(tiers.values())
    out["match_tiers"] = {"exact": tiers[0], "ignoring spaces": tiers[1],
                          "last name + first initial": tiers[2], "same first or last token": tiers[3]}
    out["unmatched_causes"] = dict(causes)
    out["unmatched_examples"] = {k: c.most_common(examples) for k, c in missed.items()}
    pts = [r for r in rows if r["dk_pts"] is not None]
    bad = [r for r in pts if abs(rg.recompute_points(r["raw_stats"]) - r["dk_pts"]) >= 1e-6]
    out["points_rows"], out["points_exact"] = len(pts), len(pts) - len(bad)
    out["points_mismatch_examples"] = [
        f"{r['game_date']} {r['name']}: page {r['dk_pts']} vs recomputed "
        f"{rg.recompute_points(r['raw_stats'])} [{r['raw_stats']}]" for r in bad[:examples]]
    out["triple_doubles"] = sum(rg.is_triple_double(r["raw_stats"]) for r in rows)
    out["salaried_rows"] = sum(r["salary"] is not None for r in rows)
    out["rows_on_slate"] = sum(bool(r["on_slate"]) for r in rows)
    games = {(r["game_date"], r["game_id"]): r["on_slate"] for r in rows if r["game_id"]}
    out["games"], out["games_on_slate"] = len(games), sum(bool(v) for v in games.values())
    per_day = Counter(d for (d, _), v in games.items() if v)
    out["days_with_slate"] = len(per_day)
    out["games_on_slate_per_day_avg"] = (sum(per_day.values()) / len(per_day)) if per_day else 0.0
    return out
