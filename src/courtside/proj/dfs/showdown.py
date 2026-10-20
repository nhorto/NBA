"""Showdown Captain Mode checks (docs/PLAYERS.md §2.6, §5.7; task T2.15).

The optimizer already handles Showdown from ``rules_json`` (CPT slot with a 1.5x multiplier on
salary and points, UTIL x5, each player once, ``teamCount.min`` = 2, no late swap). This module
verifies it against pydfs-lineup-optimizer (the T1.5 test oracle) and checks DK's Showdown
pricing against what we capture.

**Slates.** Few real Showdown salary files exist (the light DK capture started 2026-10-19), so
the check builds Showdown slates from RotoGuru single games (``dfs_hist_salaries``, DK Classic
salaries 2014-21): every priced player of the two teams, UTIL salary = his DK salary, CPT
salary = `captain_salary` (1.5x), projections = his actual DK points. Real captured Showdown
slates (``dfs_slates`` tag showdown / game type 81) are added with DK's own CPT and UTIL rows and
FPPG as the projection.

**Pass criterion.** The same optimal objective as pydfs, exactly: actual DK points are multiples
of 0.25 and FPPG has one decimal, so 1.5x is exact at the solver's 1/100 scale (or, for quarter
points, off by at most 0.005 while lineup totals differ by multiples of 0.125: rounding can never
reorder two lineups). Lineups may differ only on an exact tie; both are checked against the DK
rules (one CPT, five UTIL, six different players, >= 2 teams, salary <= cap, CPT salary 1.5x).

**What pydfs encodes differently.** pydfs hard-codes the Showdown roster (``max_from_one_team =
5``, budget 50,000) and needs the CPT as a separate player row; ``max 5 from one team`` is the
same rule as ">= 2 teams" in a two-team game. We read ``teamCount.min`` from DK's payload.

**CPT salary rounding.** DK's CPT salary is exactly 1.5x the UTIL salary on every captured row:
DK prices Showdown UTIL in $200 steps, so 1.5x is always a whole $300 step and no rounding ever
happens (`check_cpt_salaries`). Classic salaries ($100 steps) can give $x50 captains; we keep
them exact (integer dollars) because DK never needs a rounding rule for its own prices.

CLI::

    courtside dfs showdown-check [--n 30] [--seed 2026] [--json]
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from . import optimizer as O
from . import rules as R

SHOWDOWN_GAME_TYPE = 81
CAPTAIN = "CPT"
FLEX = "UTIL"
MIN_PLAYERS = 14               # a RotoGuru game needs this many priced players with points
MIN_PER_TEAM = 5
EPS = 1e-6


def captain_salary(salary: int, multiplier: float = 1.5) -> int:
    """DK's CPT salary for a UTIL salary. Exact for every DK price (multiples of $100); a
    fractional result is rounded half up to the dollar (never observed: see the module doc)."""
    v = salary * multiplier
    return int(v + 0.5) if v != int(v) else int(v)


def showdown_rules(conn: sqlite3.Connection | None = None) -> R.Rules:
    """DK's Showdown rules: a stored slate's ``rules_json``, else the archived gametype_81
    payload."""
    if conn is not None:
        try:
            r = conn.execute("SELECT rules_json FROM dfs_slates WHERE game_type_id=? AND "
                             "rules_json IS NOT NULL ORDER BY lock_utc DESC LIMIT 1",
                             (SHOWDOWN_GAME_TYPE,)).fetchone()
            if r and r[0]:
                return R.Rules.from_payload(json.loads(r[0]))
        except sqlite3.OperationalError:
            pass
    rules = O.classic_rules(SHOWDOWN_GAME_TYPE)
    if rules is None:
        raise LookupError("no Showdown rules payload stored or archived")
    return rules


# ============================================================================== slates

def hist_games(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """RotoGuru games usable as Showdown slates: (date, team A, team B), A < B."""
    rows = conn.execute(
        "SELECT game_date, team, opp, COUNT(*) FROM dfs_hist_salaries WHERE salary>0 AND "
        "dk_pts IS NOT NULL AND team IS NOT NULL AND opp IS NOT NULL "
        "GROUP BY game_date, team, opp").fetchall()
    per = {(d, t, o): n for d, t, o, n in rows}
    out = []
    for (d, t, o), n in per.items():
        if t < o and n >= MIN_PER_TEAM and per.get((d, o, t), 0) >= MIN_PER_TEAM \
                and n + per[(d, o, t)] >= MIN_PLAYERS:
            out.append((d, t, o))
    return sorted(out)


def hist_showdown_pool(conn: sqlite3.Connection, day: str, team_a: str, team_b: str,
                       rules: R.Rules) -> tuple[R.SlatePool, dict[str, O.Projection]]:
    """A Showdown pool from one RotoGuru game: CPT and UTIL eligibility for every priced player,
    CPT salary `captain_salary`, projections = actual DK points."""
    mult = rules.captain_multiplier
    rows = conn.execute(
        "SELECT rg_id, name, pos, team, opp, home, salary, dk_pts FROM dfs_hist_salaries "
        "WHERE game_date=? AND team IN (?, ?) AND opp IN (?, ?) AND salary>0 AND dk_pts IS NOT "
        "NULL ORDER BY rg_id", (day, team_a, team_b, team_a, team_b)).fetchall()
    players, projs = [], {}
    label = None
    for rg, name, pos, team, opp, home, sal, pts in rows:
        h, a = (team, opp) if home else (opp, team)
        label = label or f"{a} @ {h}"
        pid = str(rg)
        players.append(R.PoolPlayer(
            dk_player_id=pid, dk_dk_id=None, name=name, team=team,
            positions=R.split_positions(pos), salary=int(sal), slots=(CAPTAIN, FLEX),
            salary_by_slot={CAPTAIN: captain_salary(int(sal), mult), FLEX: int(sal)},
            draftable_ids={CAPTAIN: f"9{rg}", FLEX: f"8{rg}"}, game_id=f"{day}:{team_a}-{team_b}",
            competition_id=None, game_label=label, home=h, away=a, start_time=None, status=None,
            news_status=None, starter=None, swappable=None, fppg=None))
        projs[pid] = O.Projection(float(pts))
    pool = R.SlatePool(slate_id=f"rg:{day}:{team_a}-{team_b}", game_type_id=SHOWDOWN_GAME_TYPE,
                       game_type="showdown", captured_at=day, rules=rules, players=tuple(players))
    return pool, projs


def real_showdown_slates(conn: sqlite3.Connection) -> list[str]:
    return [str(r[0]) for r in conn.execute(
        "SELECT slate_id FROM dfs_slates WHERE tag='showdown' OR game_type_id=? "
        "ORDER BY lock_utc, slate_id", (SHOWDOWN_GAME_TYPE,))
        if conn.execute("SELECT 1 FROM dfs_salaries WHERE slate_id=? LIMIT 1",
                        (str(r[0]),)).fetchone()]


# ============================================================================== CPT pricing

def check_cpt_salaries(conn: sqlite3.Connection) -> dict:
    """Every captured Showdown player: is the CPT draftable's salary exactly
    `captain_salary(UTIL salary)`? Also how many UTIL prices would need rounding at all."""
    n = exact = needs_rounding = 0
    bad = []
    steps: Counter = Counter()
    for sid, pid, cap, cpt, util, name in conn.execute(
            "SELECT slate_id, site_player_id, captured_at, "
            "MAX(CASE WHEN roster_slot='CPT' THEN salary END), "
            "MAX(CASE WHEN roster_slot<>'CPT' THEN salary END), MAX(player_name) "
            "FROM dfs_salaries WHERE game_type_id=? GROUP BY slate_id, site_player_id, "
            "captured_at", (SHOWDOWN_GAME_TYPE,)):
        if cpt is None or util is None:
            continue
        n += 1
        want = captain_salary(int(util))
        exact += int(cpt) == want
        needs_rounding += (int(util) * 3) % 2 != 0
        steps[200 if util % 200 == 0 else 100 if util % 100 == 0 else 1] += 1
        if int(cpt) != want and len(bad) < 10:
            bad.append({"slate": sid, "name": name, "util": util, "cpt": cpt, "want": want})
    return {"n": n, "exact": exact, "needs_rounding": needs_rounding,
            "util_price_steps": dict(steps), "mismatches": bad,
            "slates": len(real_showdown_slates(conn))}


# ============================================================================== pydfs oracle

def pydfs_best(players: Sequence[R.PoolPlayer], projs: Mapping[str, float], rules: R.Rules
               ) -> tuple[float, list[tuple[str, str]], dict]:
    """(objective, [(slot, player id)], settings) of pydfs's DraftKings Captain Mode optimum on
    the same pool: each player as a CPT row (CPT salary, 1.5x points) and a UTIL row."""
    from pydfs_lineup_optimizer import Player, Site, Sport, get_optimizer
    from pydfs_lineup_optimizer.player import GameInfo
    opt = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
    settings = {"budget": opt.settings.budget, "max_from_one_team": opt.settings.max_from_one_team,
                "positions": [p.name for p in opt.settings.positions]}
    mult = rules.captain_multiplier
    info = None
    pp = []
    for p in players:
        if p.dk_player_id not in projs:
            continue
        info = info or GameInfo(p.home or p.team, p.away or "?", None)
        v = float(projs[p.dk_player_id])
        pp.append(Player(f"{p.dk_player_id}|{CAPTAIN}", p.dk_player_id, "", [CAPTAIN], p.team,
                         p.salary_for(CAPTAIN, rules), v * mult, game_info=info))
        pp.append(Player(f"{p.dk_player_id}|{FLEX}", p.dk_player_id, "", [FLEX], p.team,
                         p.salary_for(FLEX, rules), v, game_info=info))
    opt.player_pool.load_players(pp)
    if rules.min_teams:
        opt.set_total_teams(min_teams=rules.min_teams)
    lu = next(iter(opt.optimize(1)))
    pairs = []
    for lp in lu.players:
        pid, slot = lp.id.split("|")
        pairs.append((slot, pid))
    return float(lu.fantasy_points_projection), pairs, settings


def rule_problems(rules: R.Rules, by_id: Mapping[str, R.PoolPlayer],
                  pairs: Sequence[tuple[str, str]]) -> list[str]:
    """DK Showdown rules a lineup breaks (empty = legal)."""
    out = []
    slots = Counter(s for s, _ in pairs)
    if slots != Counter(rules.slot_names):
        out.append(f"slots {dict(slots)}")
    ids = [p for _, p in pairs]
    if len(set(ids)) != len(ids):
        out.append("a player is both CPT and UTIL")
    sal = sum(by_id[p].salary_for(s, rules) for s, p in pairs)
    if sal > rules.salary_cap:
        out.append(f"salary {sal} > {rules.salary_cap}")
    if rules.min_teams and len({by_id[p].team for p in ids}) < rules.min_teams:
        out.append("fewer than 2 teams")
    for s, p in pairs:
        if s == CAPTAIN and by_id[p].salary_for(s, rules) != captain_salary(
                by_id[p].salary, rules.captain_multiplier):
            out.append(f"CPT salary {by_id[p].salary_for(s, rules)} != 1.5 x {by_id[p].salary}")
    return out


def points(rules: R.Rules, projs: Mapping[str, float], pairs) -> float:
    return sum(float(projs[p]) * rules.slot_multiplier(s) for s, p in pairs)


@dataclass
class SlateCheck:
    slate: str
    source: str                      # rotoguru | captured
    players: int
    ours: float | None
    pydfs: float | None
    same_lineup: bool
    tie: bool
    problems: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return (self.ours is not None and self.pydfs is not None
                and abs(self.ours - self.pydfs) <= EPS and not self.problems)


def check_pool(pool: R.SlatePool, projs: Mapping, rules: R.Rules, source: str) -> SlateCheck:
    t0 = time.perf_counter()
    means = {k: O.as_projection(v).mean for k, v in projs.items()}
    by_id = pool.by_id()
    lu = O.optimize(pool, means, rules)
    ours_pairs = lu.slot_pairs() if lu else None
    ours = points(rules, means, ours_pairs) if lu else None
    best, theirs, _ = pydfs_best(pool.players, means, rules)
    problems = []
    if ours_pairs is not None:
        problems += [f"ours: {p}" for p in rule_problems(rules, by_id, ours_pairs)]
        if abs(lu.projected - ours) > 0.01:
            problems.append(f"ours: stored projection {lu.projected} != {ours}")
    problems += [f"pydfs: {p}" for p in rule_problems(rules, by_id, theirs)]
    same = ours_pairs is not None and sorted(ours_pairs) == sorted(theirs)
    tie = (not same and ours is not None and abs(ours - points(rules, means, theirs)) <= EPS)
    if ours is not None and abs(ours - best) > EPS:
        problems.append(f"objective {ours:.3f} vs pydfs {best:.3f}")
    return SlateCheck(pool.slate_id, source, len(pool.players), ours, best, same, tie, problems,
                      time.perf_counter() - t0)


def run_check(conn: sqlite3.Connection, n: int = 30, *, seed: int = 2026,
              include_real: bool = True, noisy: bool = True) -> dict:
    """The T2.15 comparison: `n` RotoGuru single games (sampled with `seed`; each with actual
    points and, `noisy`, a noisy projection) plus every real captured Showdown slate. Returns
    {checks, passed, total, slates, slates_passed, ties, cpt_salaries, pydfs}."""
    rules = showdown_rules(conn)
    games = hist_games(conn)
    rnd = random.Random(seed)
    pick = sorted(rnd.sample(games, min(n, len(games))))
    checks: list[SlateCheck] = []
    for day, a, b in pick:
        pool, projs = hist_showdown_pool(conn, day, a, b, rules)
        checks.append(check_pool(pool, projs, rules, "rotoguru"))
        if noisy:      # a "projection": actual x U(0.6, 1.4), kept on DK's quarter-point grid
            nz = {k: round(v.mean * rnd.uniform(0.6, 1.4) * 4) / 4 for k, v in projs.items()}
            checks.append(check_pool(pool, nz, rules, "rotoguru-noisy"))
    if include_real:
        for sid in real_showdown_slates(conn):
            pool = R.build_pool(conn, sid)
            r = pool.rules or rules
            projs = {p.dk_player_id: p.fppg for p in pool.players if p.fppg is not None}
            checks.append(check_pool(pool, projs, r, "captured"))
    from pydfs_lineup_optimizer import Site, Sport, get_optimizer
    st = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL).settings
    bad = {c.slate for c in checks if not c.ok}
    names = list(dict.fromkeys(c.slate for c in checks))
    return {"checks": checks, "passed": sum(c.ok for c in checks), "total": len(checks),
            "slates": len(names), "slates_passed": sum(s not in bad for s in names),
            "rotoguru": sum(c.source == "rotoguru" for c in checks),
            "captured": sum(c.source == "captured" for c in checks),
            "ties": sum(c.tie for c in checks), "games_available": len(games),
            "cpt_salaries": check_cpt_salaries(conn),
            "rules": {"cap": rules.salary_cap, "min_teams": rules.min_teams,
                      "captain_multiplier": rules.captain_multiplier,
                      "late_swap": rules.allow_late_swap},
            "pydfs": {"budget": st.budget, "max_from_one_team": st.max_from_one_team}}


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("showdown-check", help="T2.15: our Showdown optimum vs pydfs on RotoGuru "
                                              "single games + captured Showdown slates")
    p.add_argument("--n", type=int, default=30, help="RotoGuru games (default 30)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-real", action="store_true", help="skip captured Showdown slates")
    p.add_argument("--no-noisy", action="store_true", help="actual points only")
    p.add_argument("--json", action="store_true")
    return {"showdown-check": _check_cmd}


def _check_cmd(args) -> int:
    from ... import db
    conn = db.connect()
    rep = run_check(conn, args.n, seed=args.seed, include_real=not args.no_real,
                    noisy=not args.no_noisy)
    if args.json:
        out = {**rep, "checks": [c.__dict__ | {"ok": c.ok} for c in rep["checks"]]}
        print(json.dumps(out, indent=2, default=str))
        return 0 if rep["passed"] == rep["total"] else 1
    print(f"Showdown vs pydfs-lineup-optimizer: {rep['slates_passed']}/{rep['slates']} slates, "
          f"{rep['passed']}/{rep['total']} comparisons with the same optimal objective "
          f"({rep['rotoguru']} RotoGuru games of {rep['games_available']}, actual + noisy "
          f"points; {rep['captured']} captured slates, FPPG; {rep['ties']} exact ties with a "
          f"different lineup)")
    for c in rep["checks"]:
        flag = "ok  " if c.ok else "FAIL"
        lu = "same lineup" if c.same_lineup else ("tie" if c.tie else "different")
        print(f"  [{flag}] {c.slate:<28} {c.source:<14} {c.players:>3} players  ours "
              f"{c.ours:8.2f}  pydfs {c.pydfs:8.2f}  {lu}  {c.seconds:.2f}s")
        for p in c.problems:
            print(f"         {p}")
    cs = rep["cpt_salaries"]
    print(f"CPT salary = 1.5 x UTIL on captured rows: {cs['exact']}/{cs['n']} "
          f"({cs['slates']} slates; UTIL price steps {cs['util_price_steps']}; "
          f"{cs['needs_rounding']} would need rounding)")
    for m in cs["mismatches"]:
        print(f"  mismatch {m}")
    print(f"rules: {rep['rules']}; pydfs settings: {rep['pydfs']}")
    return 0 if rep["passed"] == rep["total"] else 1
