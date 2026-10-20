"""Post-game "why" v0 (docs/PLAYERS.md §2.8, T1.8): why did he beat / miss our number?

For every player-game with a stored pregame projection and an actual box, the DraftKings points
he scored are compared with the projection's expected DK points and the gap is split EXACTLY::

    actual - expected = minutes term + sum of per-stat rate terms (+ residual, which is 0)

Write DK points as ``DK = m * d`` with ``m`` minutes and ``d`` the DK points per minute, made of
components ``c_k`` (``d_k = c_k / m``): points, threes (0.5 each), rebounds, assists, steals,
blocks, turnovers and the double-double / triple-double bonus. With ``e`` for expected and ``a``
for actual::

    minutes term   = (m_a - m_e) * D_e                 D_e = sum_k d_e,k  (expected DK per minute)
    rate term k    = m_a * (d_a,k - d_e,k)             (his actual vs expected rate, at his minutes)
    sum            = m_a*D_e - m_e*D_e + m_a*D_a - m_a*D_e = m_a*D_a - m_e*D_e = actual - expected

so the terms add up to the gap by algebra, not by fitting, and a player who did not play
(``m_a = 0``) is a pure minutes term of ``-expected``. ``residual`` is computed as
``(actual - expected) - sum(terms)`` and is exactly 0 except for the degenerate case of a player
with no expected minutes (``m_e ~ 0``), where his expected rate is taken to be his actual rate.

Expected components. The expected minutes and DK points are the stored ``min`` / ``dk`` means
(cond ``all``, i.e. including the chance he sits, the number the lineup optimizer saw). Points,
threes, rebounds and assists come from the stored stat means when the projection carries them
(``player_projection_stats`` rows ``pts/fg3m/reb/ast``, or ``inputs_json['exp']``), otherwise from
the EB per-minute rate state (``proj/state.py``, the state before the game day) times expected
minutes; steals, blocks and turnovers come from the rate state. The expected bonus is the
remainder ``expected DK - sum(weight * expected count)``, so the components always sum to the
projection's own DK mean and the bonus term also absorbs any small calibration gap between the
simulation's DK mean and the rate-state stat means (flagged by ``approx`` in ``terms_json`` when a
stat had to be filled in).

Context facts are data only, with no causal claims: regulars of his previous five team games who
did not play, final margin (blowout at 20+), overtime, started or not.

    build(conn, nba_day, *, llm=True) -> summary      writes ``postgame_why_v0``
    top_stories(conn, nba_day, n=5) -> [dict]         for the Brief to call later
    courtside proj why --date D [--no-llm] [--historical]

Text: a template sentence is always written first; with ``llm=True`` the top stories (<= 30) are
rewritten by the subscription CLI through ``brief.llm`` and kept only if the numeric fact-check
(``brief.factcheck``) and the sign guard (`check_signs`) both pass, otherwise the template stays.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from ..brief import factcheck, llm, writer
from ..sources import dfs as _dfs
from .dfs.rules import dk_points

BLOWOUT_MARGIN = 20
NOTABLE_DK = 6.0              # |actual - expected| DK points ...
NOTABLE_PTS = 5.0             # ... or |actual - expected| points (docs/PLAYERS.md §2.8)
MIN_MINUTES = 10.0            # played at least this long (or was expected to, and did not play)
CAP = 30                      # notable players per night
LLM_MAX = 30                  # LLM rewrites per night (S9)
TERM_FLOOR = 1.0              # a term has to be this big to be named in text
MAX_TERMS_IN_TEXT = 3
EPS = 1e-9

SC = _dfs.CLASSIC_SCORING
WEIGHTS = {"pts": SC["point"], "fg3m": SC["three_pointer_made"], "reb": SC["rebound"],
           "ast": SC["assist"], "stl": SC["steal"], "blk": SC["block"], "tov": SC["turnover"]}
STATS = tuple(WEIGHTS)
COMPONENTS = (*STATS, "bonus")
TERM_KEYS = ("minutes", *COMPONENTS)
PROJ_STATS = ("pts", "fg3m", "reb", "ast")          # means the projection store carries
SOURCE_LABEL = {"minutes": "minutes", "pts": "points", "fg3m": "threes", "reb": "rebounds",
                "ast": "assists", "stl": "steals", "blk": "blocks", "tov": "turnovers",
                "bonus": "bonus"}


# ============================================================================== decomposition

@dataclass
class Decomp:
    expected_min: float
    actual_min: float
    expected_dk: float
    actual_dk: float
    terms: dict[str, float]           # TERM_KEYS -> signed DK points
    residual: float                   # delta - sum(terms), 0 up to float noise
    exp_counts: dict[str, float]      # expected stat counts (pts, fg3m, ...)
    act_counts: dict[str, float]
    exp_comps: dict[str, float]       # expected DK points by component (incl. bonus)
    act_comps: dict[str, float]
    approx: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.actual_dk - self.expected_dk

    @property
    def exp_counts_at_min(self) -> dict[str, float]:
        """Expected stat counts scaled to the minutes he actually played: what his expected
        rates imply for that many minutes (the figure a rate term compares his line with)."""
        if self.expected_min > 1e-6:
            k = self.actual_min / self.expected_min
            return {s: v * k for s, v in self.exp_counts.items()}
        return dict(self.act_counts)


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0


def actual_components(row) -> tuple[dict[str, float], dict[str, float], float]:
    """(DK points by component incl. bonus, raw counts, DK total) of a `player_games`-like row."""
    counts = {s: _f(row[s]) for s in STATS}
    comps = {s: WEIGHTS[s] * counts[s] for s in STATS}
    total = dk_points({s: counts[s] for s in STATS})
    comps["bonus"] = total - sum(comps[s] for s in STATS)
    return comps, counts, total


def decompose(exp_min: float, exp_dk: float, exp_counts: dict[str, float], act_min: float,
              act_row, *, approx: list[str] | None = None) -> Decomp:
    """The exact split of `actual - expected` (see the module docstring)."""
    exp_min, exp_dk, act_min = _f(exp_min), _f(exp_dk), _f(act_min)
    ce = {s: WEIGHTS[s] * _f(exp_counts.get(s)) for s in STATS}
    ce["bonus"] = exp_dk - sum(ce[s] for s in STATS)
    ca, counts, act_dk = actual_components(act_row)
    if exp_min > 1e-6:
        de = {k: v / exp_min for k, v in ce.items()}
    elif act_min > 1e-6:                                 # not expected to play at all
        de = {k: v / act_min for k, v in ca.items()}
    else:
        de = {k: 0.0 for k in ce}
    da = {k: v / act_min for k, v in ca.items()} if act_min > 1e-6 else dict(de)
    d_exp = sum(de.values())
    terms = {"minutes": (act_min - exp_min) * d_exp}
    for k in COMPONENTS:
        terms[k] = act_min * (da[k] - de[k])
    resid = (act_dk - exp_dk) - sum(terms.values())
    return Decomp(exp_min, act_min, exp_dk, act_dk, terms, resid,
                  {s: _f(exp_counts.get(s)) for s in STATS}, counts, ce, ca, list(approx or []))


# ============================================================================== reading the DB

def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _as_date(d) -> date:
    return d if isinstance(d, date) and not isinstance(d, datetime) else date.fromisoformat(str(d)[:10])


def _json(s) -> dict:
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(s) if s else {}
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _game_info(conn: sqlite3.Connection, nba_day: str) -> dict[str, dict]:
    """game_id -> {tip, home, away} for the day (games table, else team_games)."""
    out: dict[str, dict] = {}
    for r in conn.execute("SELECT game_id, tipoff_utc, home, away FROM games WHERE nba_day=?",
                          (nba_day,)):
        out[r[0]] = {"tip": _parse_ts(r[1]), "home": r[2], "away": r[3]}
    return out


def select_runs(conn: sqlite3.Connection, nba_day: str, *, allow_post_tip: bool = False
                ) -> dict[str, dict]:
    """game_id -> the latest projection run created at or before that game's tip.

    With `allow_post_tip` (historical backfills, whose rows are created after the game) a game
    with no earlier run takes its latest run, flagged ``pre_tip = False``."""
    info = _game_info(conn, nba_day)
    cand: dict[str, list[dict]] = defaultdict(list)
    for r in conn.execute(
            "SELECT DISTINCT game_id, run_id, model_version, created_at FROM player_projections "
            "WHERE nba_day=?", (nba_day,)):
        cand[r[0]].append({"run_id": r[1], "model_version": r[2], "created_at": r[3],
                           "ts": _parse_ts(r[3])})
    out: dict[str, dict] = {}
    for gid, runs in cand.items():
        tip = (info.get(gid) or {}).get("tip")
        runs.sort(key=lambda x: (x["ts"] or datetime.min.replace(tzinfo=timezone.utc),
                                 str(x["run_id"])))
        ok = [x for x in runs if tip is None or (x["ts"] is not None and x["ts"] <= tip)]
        if ok:
            out[gid] = {**ok[-1], "pre_tip": True if tip is not None else None}
        elif allow_post_tip and runs:
            out[gid] = {**runs[-1], "pre_tip": False}
    return out


def _proj_rows(conn: sqlite3.Connection, gid: str, run: dict) -> list[dict]:
    rows = _dicts(
        conn, "SELECT * FROM player_projections WHERE game_id=? AND run_id IS ? AND "
        "model_version IS ?", (gid, run["run_id"], run["model_version"]))
    out = []
    for d in rows:
        st: dict[tuple[str, str], float | None] = {}
        for s in conn.execute("SELECT stat, cond, mean FROM player_projection_stats "
                              "WHERE projection_id=?", (d["id"],)):
            st[(s[0], s[1])] = s[2]
        d["stats"] = st
        d["inputs"] = _json(d.get("inputs_json"))
        d["minutes"] = _json(d.get("minutes_json"))
        out.append(d)
    return out


def _opt(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def expected_inputs(p: dict) -> tuple[float | None, float | None, dict[str, float | None]]:
    """(expected minutes, expected DK, stored stat means) of a projection row (cond 'all')."""
    st = p["stats"]
    m = _opt(st.get(("min", "all")))
    if m is None:
        m = _opt(p["minutes"].get("mean_all"))
    dk = _opt(st.get(("dk", "all")))
    exp = p["inputs"].get("exp") if isinstance(p["inputs"].get("exp"), dict) else {}
    stored = {s: _opt(st.get((s, "all"))) for s in STATS}
    for s in STATS:
        if stored[s] is None:
            stored[s] = _opt(exp.get(s))
    return m, dk, stored


def load_rates(conn: sqlite3.Connection, day: date, pids: list[int]) -> dict[int, dict[str, float]]:
    """Per-minute EB rates before `day`: the stored `rate_state` when it is recent, else
    `state.rate_frame` computed on the fly. {player_id: {stat: rate}}."""
    from . import state as ST
    out: dict[int, dict[str, float]] = {}
    if not pids:
        return out
    wide = ST.read_state(conn, day)
    fresh = False
    if len(wide):
        try:
            as_of = _as_date(wide.attrs.get("as_of"))
            fresh = timedelta(0) <= day - as_of <= timedelta(days=7)
        except (TypeError, ValueError):
            fresh = False
    if not fresh:
        import pandas as pd
        q = pd.DataFrame({"player_id": [int(p) for p in pids],
                          "game_date": [day.isoformat()] * len(pids)})
        wide = ST.rate_frame(conn, q).set_index("player_id")
    for pid in pids:
        row = wide.loc[pid] if pid in wide.index else None
        rates = {}
        for s in STATS:
            v = _opt(row.get(f"r_{s}")) if row is not None else None
            rates[s] = v if v is not None else ST.DEFAULT_RATE[s]
        out[int(pid)] = rates
    return out


def _dicts(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, args)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _box(conn: sqlite3.Connection, gid: str) -> dict[int, dict]:
    return {int(r["player_id"]): r for r in _dicts(
        conn, "SELECT * FROM player_games WHERE game_id=?", (gid,))}


def _game_context(conn: sqlite3.Connection, gid: str) -> dict:
    """{'margin': final margin, 'n_ot': overtimes, 'opp': {team: opponent}} from team_games."""
    rows = conn.execute("SELECT team, opp, pts, min FROM team_games WHERE game_id=?",
                        (gid,)).fetchall()
    ctx: dict = {"margin": None, "n_ot": 0, "opp": {}}
    for r in rows:
        ctx["opp"][r[0]] = r[1]
    if len(rows) == 2 and rows[0][2] is not None and rows[1][2] is not None:
        ctx["margin"] = abs(int(rows[0][2]) - int(rows[1][2]))
    mins = [float(r[3]) for r in rows if r[3] is not None]
    if mins:
        ctx["n_ot"] = max(0, int(round((max(mins) - 240.0) / 25.0)))
    return ctx


def absent_regulars(conn: sqlite3.Connection, gid: str, team: str, day: str,
                    box: dict[int, dict]) -> list[str]:
    """Players of `team` who played in >= 3 of its previous 5 games for >= 20 min on average
    and did not play in this one (a heuristic fact; never a claim about why)."""
    prev = [r[0] for r in conn.execute(
        "SELECT game_id FROM team_games WHERE team=? AND game_date<? "
        "ORDER BY game_date DESC LIMIT 5", (team, day))]
    if len(prev) < 3:
        return []
    q = ",".join("?" * len(prev))
    agg: dict[int, list] = defaultdict(lambda: [None, 0, 0.0])
    for pid, name, mn in conn.execute(
            f"SELECT player_id, player_name, min FROM player_games WHERE team=? AND "
            f"game_id IN ({q})", (team, *prev)):
        if mn and mn > 0:
            a = agg[int(pid)]
            a[0], a[1], a[2] = name, a[1] + 1, a[2] + float(mn)
    played = {pid for pid, r in box.items() if (r.get("min") or 0) > 0 and r.get("team") == team}
    regs = [(a[2] / a[1], a[0]) for pid, a in agg.items()
            if a[1] >= 3 and a[2] / a[1] >= 20.0 and pid not in played and a[0]]
    regs.sort(reverse=True)
    return [n for _, n in regs[:3]]


def _starters(conn: sqlite3.Connection, gid: str) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT player_id FROM player_starts WHERE game_id=?",
                                            (gid,))}


# ============================================================================== records

def _r1(x: float) -> float:
    v = round(float(x), 1)
    return 0.0 if v == 0 else v


def build_record(p: dict, gid: str, box_row: dict | None, rates: dict[str, float] | None,
                 ctx: dict, *, name: str, run: dict, absent: list[str], started: bool | None
                 ) -> dict | None:
    """One `postgame_why_v0` record (no text yet) for a projection row, or None if it cannot be
    decomposed (no expected minutes / DK in the projection)."""
    m_e, dk_e, stored = expected_inputs(p)
    if m_e is None or dk_e is None:
        return None
    approx: list[str] = []
    counts: dict[str, float] = {}
    for s in STATS:
        v = stored.get(s)
        if v is None and rates is not None:
            v = rates[s] * m_e
            if s in PROJ_STATS:
                approx.append(s)            # the projection should carry these
        elif v is None:
            v = 0.0
            approx.append(s)
        counts[s] = v
    if rates is not None:
        for s in ("stl", "blk", "tov"):
            if stored.get(s) is None:
                counts[s] = rates[s] * m_e
    act_row = box_row or {s: 0 for s in STATS}
    act_min = _f((box_row or {}).get("min"))
    d = decompose(m_e, dk_e, counts, act_min, act_row, approx=approx)
    team = (box_row or {}).get("team") or p.get("team")
    cats = sum(1 for s in ("pts", "reb", "ast", "stl", "blk") if d.act_counts[s] >= 10)
    started_exp = p["inputs"].get("p_start")
    facts = {
        "overtime": ctx.get("n_ot") or 0,
        "blowout_margin": ctx["margin"] if (ctx.get("margin") or 0) >= BLOWOUT_MARGIN else None,
        "margin": ctx.get("margin"),
        "started": started,
        "p_start": _opt(started_exp),
        "exp_min_played": _opt(p["minutes"].get("mean_played")),   # minutes if he plays
        "teammates_not_playing": absent,
        "p_play": _opt(p.get("p_play")),
        "double_double_cats": cats,
    }
    return {
        "game_id": gid, "player_id": int(p["player_id"]), "name": name, "team": team,
        "opponent": (ctx.get("opp") or {}).get(team), "run": run, "decomp": d, "facts": facts,
    }


# ============================================================================== text: templates

def _signed(v: float) -> str:
    return f"{_r1(v):+.1f}"


def _min_str(x: float) -> str:
    return f"{int(round(x))}"


def _bonus_label(d: Decomp) -> str:
    cats = sum(1 for s in ("pts", "reb", "ast", "stl", "blk") if d.act_counts[s] >= 10)
    if cats >= 3:
        return "the triple-double bonus"
    if cats == 2:
        return "the double-double bonus"
    return "bonus points"


def term_phrase(key: str, v: float, d: Decomp) -> str:
    """'+9.0 from rebounds (15)': the signed figure first, so the sign is never ambiguous."""
    if key == "minutes":
        return (f"{_signed(v)} from minutes ({_min_str(d.actual_min)} vs "
                f"{_min_str(d.expected_min)} expected)")
    if key == "bonus":
        return f"{_signed(v)} from {_bonus_label(d)}"
    n = int(round(d.act_counts[key]))
    if key == "tov":
        return f"{_signed(v)} from {'fewer ' if v > 0 else ''}turnovers ({n})"
    if key == "fg3m":
        return f"{_signed(v)} from threes ({n} made)"
    return f"{_signed(v)} from {SOURCE_LABEL[key]} ({n})"


def shown_terms(d: Decomp) -> list[tuple[str, float]]:
    """Terms named in text: the largest (<= 3) with |v| >= TERM_FLOOR, else the single largest."""
    ranked = sorted(((k, v) for k, v in d.terms.items()), key=lambda kv: -abs(kv[1]))
    big = [(k, v) for k, v in ranked if abs(v) >= TERM_FLOOR][:MAX_TERMS_IN_TEXT]
    if big:
        return big
    return [(k, v) for k, v in ranked[:1] if abs(v) >= 0.3]


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 2 else " and ".join(parts)


def _names(ns: list[str]) -> str:
    return " and ".join(ns) if len(ns) <= 2 else ", ".join(ns[:-1]) + ", and " + ns[-1]


def context_sentence(rec: dict) -> str | None:
    f, bits = rec["facts"], []
    if f["overtime"]:
        n = f["overtime"]
        bits.append("the game went to " + {1: "overtime", 2: "double overtime",
                                           3: "triple overtime"}.get(n, f"{n} overtimes"))
    if f["blowout_margin"]:
        bits.append(f"it was a {f['blowout_margin']}-point blowout")
    if f["teammates_not_playing"]:
        bits.append(f"{_names(f['teammates_not_playing'])} did not play")
    if not bits:
        return None
    s = "; ".join(bits)
    return s[0].upper() + s[1:] + "."


def template_text(rec: dict) -> str:
    d: Decomp = rec["decomp"]
    head = (f"{rec['name']}: {d.actual_dk:.1f} DK vs {d.expected_dk:.1f} expected "
            f"({_signed(d.delta)})")
    if d.actual_min <= 0:
        return (f"{head}: did not play (about {_min_str(d.expected_min)} minutes were "
                f"projected).")
    shown = shown_terms(d)
    if shown:
        text = head + ": " + _join([term_phrase(k, v, d) for k, v in shown]) + "."
    else:
        text = head + "."
    ctx = context_sentence(rec)
    return text + (" " + ctx if ctx else "")


# ============================================================================== LLM guard

_HELP = re.compile(r"\b(?:add(?:ed|s|ing)?|boost(?:ed|s|ing)?|help(?:ed|s|ing)?|lift(?:ed|s|ing)?|"
                   r"gain(?:ed|s|ing)?|padd(?:ed|ing)|plus)\b", re.I)
_LOSS = re.compile(r"\b(?:cost(?:s|ing)?|lost|lose|loses|fewer|fell|fall|falls|drag(?:ged|s|ging)?|"
                   r"drop(?:ped|s|ping)?|short|worse|less|minus|shav(?:ed|ing)|trimm(?:ed|ing)|"
                   r"sank|hurt(?:ing)?|below|subtract(?:ed|ing)?|missing|lacked)\b", re.I)
_NUM_TOKEN = re.compile(r"(?<![\w.])([+\-]?)(\d+(?:\.\d+)?)(?![\w]|\.\d)")
_BEAT = re.compile(r"\b(?:outperform\w*|exceeded|surpass\w*|beat (?:our|his) (?:number|projection))\b",
                   re.I)
_MISS = re.compile(r"\b(?:underperform\w*|fell short|disappoint\w*|missed (?:our|his) "
                   r"(?:number|projection))\b", re.I)


_LABELS = {"minutes": "minutes", "points": "pts", "threes": "fg3m", "three-pointers": "fg3m",
           "rebounds": "reb", "assists": "ast", "steals": "stl", "blocks": "blk",
           "turnovers": "tov", "bonus": "bonus", "double-double": "bonus",
           "triple-double": "bonus"}
_LABEL_RE = "|".join(sorted((re.escape(k) for k in _LABELS), key=len, reverse=True))
_AFTER = re.compile(rf"^\s*(?:DK\s+)?(?:points?\s+)?(?:from|in|on|for)\s+(?:the\s+|fewer\s+|more\s+)?"
                    rf"({_LABEL_RE})\b", re.I)
_BEFORE = re.compile(rf"({_LABEL_RE})\b(?:[^\d.;]{{0,28}})$", re.I)


def _label_of(t: str, start: int, end: int) -> str | None:
    """The term key a figure is attached to: 'from <source>' right after it, else the nearest
    source word just before it (within its clause)."""
    m = _AFTER.match(t[end:end + 48])
    if m:
        return _LABELS[m.group(1).lower()]
    before = re.split(r"[.;,]", t[max(0, start - 40):start])[-1]
    if re.search(r"\bDK$|\bfantasy$", before.strip(), re.I):
        return None
    m = _BEFORE.search(before)
    return _LABELS[m.group(1).lower()] if m else None


def check_signs(text: str, signed, terms: dict[str, float] | None = None) -> list[str]:
    """Sign violations of `text` against the signed DK-point figures it may mention.

    `signed` is the list of displayed term values with the total gap (actual - expected) last;
    `terms` optionally maps a source key (minutes, pts, fg3m, reb, ast, stl, blk, tov, bonus) to
    its value so a figure attached to a named source ("+6.1 from minutes", "rebounds added 3.7")
    is checked against that source alone.

    * a figure written with an explicit sign ("+9.0", "-3.2") must carry the sign of the term it
      matches;
    * an unsigned figure for a negative term must not sit next to words that say it helped
      (added, boosted ...), and one for a positive term must not sit next to words that say it
      cost him (cost, fewer, dragged ...);
    * "outperformed" is not allowed when he missed our number, nor "fell short" when he beat it."""
    t = text.replace("\u2212", "-").replace("\u2013", "-")
    sv = [float(x) for x in signed]
    tv = {k: float(v) for k, v in (terms or {}).items()}
    problems: list[str] = []

    def hit(s: float, v: float, dec: int) -> bool:
        if abs(s) <= 1e-9:
            return False
        return abs(abs(_r1(s)) - v) < 0.051 if dec else int(round(abs(s))) == int(v)

    for m in _NUM_TOKEN.finditer(t):
        sign, raw = m.group(1), m.group(2)
        v = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        match = [s for s in sv if hit(s, v, dec)]
        if not match:
            continue
        label = _label_of(t, m.start(), m.end())
        if label in tv and hit(tv[label], v, dec):
            match = [tv[label]]
        if sign:
            want = 1 if sign == "+" else -1
            if not any((s > 0) == (want > 0) for s in match):
                problems.append(f"sign: '{sign}{raw}' has the wrong sign")
            continue
        a, b = m.start(), m.end()
        left = max(t.rfind(c, 0, a) for c in ".;")
        right = [i for i in (t.find(c, b) for c in ".;") if i >= 0]
        ctx = t[left + 1:(min(right) if right else len(t))]
        helps, loses = bool(_HELP.search(ctx)), bool(_LOSS.search(ctx))
        if all(s < 0 for s in match) and helps and not loses:
            problems.append(f"sign: '{raw}' is a negative contribution described as helping")
        if all(s > 0 for s in match) and loses and not helps:
            problems.append(f"sign: '{raw}' is a positive contribution described as costing")
    if sv:
        total = sv[-1]
        if total < 0 and _BEAT.search(t):
            problems.append("sign: says he beat our number but he fell short")
        if total > 0 and _MISS.search(t):
            problems.append("sign: says he fell short but he beat our number")
    return problems


def llm_facts(rec: dict) -> dict:
    """The facts one story may use: every figure is the displayed (rounded) value."""
    d: Decomp = rec["decomp"]
    contrib = []
    for k, v in sorted(d.terms.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) < 0.05:
            continue
        c = {"source": SOURCE_LABEL[k], "dk_points": _r1(v)}
        if k == "minutes":
            c.update(actual=int(round(d.actual_min)), expected=int(round(d.expected_min)))
        elif k == "bonus":
            c["kind"] = _bonus_label(d)
        else:
            c.update(actual=int(round(d.act_counts[k])),
                     expected_in_his_minutes=_r1(d.exp_counts_at_min[k]))
        contrib.append(c)
    f = rec["facts"]
    ctx = {k: v for k, v in {
        "overtime": f["overtime"] or None, "blowout_margin": f["blowout_margin"],
        "started": f["started"], "teammates_not_playing": f["teammates_not_playing"] or None,
    }.items() if v is not None}
    out = {"player": rec["name"], "team": rec["team"], "opponent": rec["opponent"],
           "unit": "DK fantasy points",
           "dk": {"actual": _r1(d.actual_dk), "expected": _r1(d.expected_dk),
                  "diff": _r1(d.delta)},
           "contributions": contrib, "context": ctx}
    return {k: v for k, v in out.items() if v is not None}


def signed_figures(rec: dict) -> list[float]:
    d: Decomp = rec["decomp"]
    return [_r1(v) for v in d.terms.values()] + [_r1(d.delta)]


def rounded_terms(rec: dict) -> dict[str, float]:
    return {k: _r1(v) for k, v in rec["decomp"].terms.items()}


def validate_story(text, rec: dict, facts: dict | None = None) -> list[str]:
    """Problems with an LLM story (empty = keep it): numbers, names, banned words, length and
    the sign guard. Anything in the list means the template text is used instead."""
    if not isinstance(text, str) or not text.strip():
        return ["empty text"]
    facts = facts if facts is not None else llm_facts(rec)
    res = factcheck.check(text, facts, kind="blurb")
    out = res.messages()
    out += check_signs(text, signed_figures(rec), rounded_terms(rec))
    if re.search(r"[#*`]|https?:", text):
        out.append("format: no markdown")
    return out


def rewrite(recs: list[dict], *, complete=None, timeout: int = 240,
            report: dict | None = None) -> dict[str, str]:
    """Ask the LLM for one story per record (one batched call plus one retry for the items that
    failed), keep only those that pass `validate_story`. Returns {key: text}; missing keys keep
    their template. `complete(prompt, system=..., timeout=...)` defaults to ``brief.llm``."""
    keyed = {f"{r['game_id']}:{r['player_id']}": r for r in recs}
    if not keyed:
        return {}
    facts = {k: llm_facts(r) for k, r in keyed.items()}
    fn = complete or llm.complete

    def ask(prompt: str) -> dict[str, str] | None:
        try:
            raw = fn(prompt, system=writer.system_prompt(), timeout=timeout)
            obj = writer.extract_json(raw)
            st = obj.get("stories", obj) if isinstance(obj, dict) else {}
            return {str(k): v for k, v in st.items() if isinstance(v, str)}
        except (llm.LLMError, ValueError, TypeError, AttributeError) as e:
            if report is not None:
                report.setdefault("errors", []).append(str(e)[:200])
            return None

    def body(ks: list[str]) -> str:
        return json.dumps([{"id": k, **facts[k]} for k in ks], ensure_ascii=False,
                          separators=(",", ":"))

    out: dict[str, str] = {}
    problems: dict[str, list[str]] = {}
    prior: dict[str, str] = {}
    cands = ask(writer.render("player_why", facts=body(list(keyed))))
    for k, r in keyed.items():
        t = (cands or {}).get(k)
        pr = validate_story(t, r, facts[k])
        if t is not None and not pr:
            out[k] = t.strip()
        else:
            problems[k] = pr or ["missing from your answer"]
            prior[k] = t or ""
    if problems and cands is not None:
        listing = "\n".join(f"- {k}: {'; '.join(p)}\n  your answer was: {json.dumps(prior[k])}"
                            for k, p in problems.items())
        retry = (writer.render("player_why", facts=body(list(problems)))
                 + "\n\n" + writer.render("retry", problems=listing))
        cands2 = ask(retry)
        for k in list(problems):
            t = (cands2 or {}).get(k)
            if t is not None and not validate_story(t, keyed[k], facts[k]):
                out[k] = t.strip()
                del problems[k]
    if report is not None:
        report["kept"], report["fell_back"] = sorted(out), sorted(problems)
        report["problems"] = problems
    return out


# ============================================================================== build

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def notable(recs: list[dict], cap: int = CAP) -> list[dict]:
    """Players worth a line: a big gap in DK points (or in points) and a real role (played >= 10
    minutes, or was projected for >= 15 and did not play); biggest absolute gap first."""
    keep = []
    for r in recs:
        d: Decomp = r["decomp"]
        played = d.actual_min >= MIN_MINUTES
        scratched = d.actual_min <= 0 and d.expected_min >= 15.0
        if not (played or scratched):
            continue
        dpts = d.act_counts["pts"] - d.exp_counts["pts"]
        if abs(d.delta) >= NOTABLE_DK or (played and abs(dpts) >= NOTABLE_PTS):
            keep.append(r)
    keep.sort(key=lambda r: (-abs(r["decomp"].delta), r["game_id"], r["player_id"]))
    return keep[:cap]


def terms_blob(rec: dict, text_source: str) -> dict:
    d: Decomp = rec["decomp"]
    run = rec["run"]
    return {
        "basis": "all", "name": rec["name"], "team": rec["team"], "opponent": rec["opponent"],
        "run_id": run.get("run_id"), "model_version": run.get("model_version"),
        "run_created_at": run.get("created_at"), "pre_tip": run.get("pre_tip"),
        "expected": {"min": d.expected_min, "dk": d.expected_dk, "counts": d.exp_counts,
                     "components": d.exp_comps},
        "actual": {"min": d.actual_min, "dk": d.actual_dk, "counts": d.act_counts,
                   "components": d.act_comps},
        "delta": d.delta, "terms": d.terms, "residual": d.residual, "approx": d.approx,
        "facts": rec["facts"], "text_source": text_source,
    }


def gather(conn: sqlite3.Connection, nba_day, *, allow_post_tip: bool = False,
           rates: dict[int, dict[str, float]] | None = None,
           log=None) -> tuple[list[dict], dict]:
    """Decomposed records of every projected player-game on the day with a box, and stats."""
    day = str(nba_day)[:10]
    runs = select_runs(conn, day, allow_post_tip=allow_post_tip)
    stats = {"games_projected": len(set(r[0] for r in conn.execute(
        "SELECT DISTINCT game_id FROM player_projections WHERE nba_day=?", (day,)))),
        "games_with_run": len(runs), "games_with_box": 0, "players": 0}
    pending = []
    for gid, run in sorted(runs.items()):
        box = _box(conn, gid)
        if not box:
            continue
        stats["games_with_box"] += 1
        ctx = _game_context(conn, gid)
        starters = _starters(conn, gid)
        for p in _proj_rows(conn, gid, run):
            pending.append((gid, run, box, ctx, starters, p))
    if rates is None and pending:
        rates = load_rates(conn, _as_date(day), sorted({int(x[5]["player_id"]) for x in pending}))
    names: dict[int, str] = {}
    absent_cache: dict[tuple[str, str], list[str]] = {}
    recs = []
    for gid, run, box, ctx, starters, p in pending:
        pid = int(p["player_id"])
        brow = box.get(pid)
        team = (brow or {}).get("team") or p.get("team")
        if (gid, team) not in absent_cache:
            absent_cache[(gid, team)] = (
                [str(n) for n in p["inputs"].get("absent_teammates") or []]
                or absent_regulars(conn, gid, team, day, box))
        name = (brow or {}).get("player_name") or names.get(pid) or _name_of(conn, pid)
        names[pid] = name
        rec = build_record(p, gid, brow, (rates or {}).get(pid), ctx, name=name, run=run,
                           absent=absent_cache[(gid, team)],
                           started=(pid in starters) if starters else None)
        if rec is not None:
            recs.append(rec)
    stats["players"] = len(recs)
    return recs, stats


def _name_of(conn: sqlite3.Connection, pid: int) -> str:
    r = conn.execute("SELECT MAX(player_name) FROM player_games WHERE player_id=?",
                     (pid,)).fetchone()
    return (r[0] if r and r[0] else str(pid))


def build(conn: sqlite3.Connection, nba_day, *, llm: bool = True, complete=None,
          cap: int = CAP, llm_max: int = LLM_MAX, allow_post_tip: bool = False,
          rates: dict[int, dict[str, float]] | None = None, log=None) -> dict:
    """Decompose the day's notable players, write `postgame_why_v0`, return a summary."""
    day = str(nba_day)[:10]
    recs, stats = gather(conn, day, allow_post_tip=allow_post_tip, rates=rates, log=log)
    picks = notable(recs, cap)
    for r in picks:
        r["text"], r["source"] = template_text(r), "template"
    report: dict = {}
    if llm and picks:
        todo = [r for r in picks if r["decomp"].actual_min > 0][:llm_max]
        got = rewrite(todo, complete=complete, report=report)
        for r in todo:
            t = got.get(f"{r['game_id']}:{r['player_id']}")
            if t:
                r["text"], r["source"] = t, "llm"
    built = _now()
    gids = sorted({r["game_id"] for r in recs} | set(select_runs(
        conn, day, allow_post_tip=allow_post_tip)))
    with conn:
        if gids:
            q = ",".join("?" * len(gids))
            conn.execute(f"DELETE FROM postgame_why_v0 WHERE game_id IN ({q})", gids)
        conn.executemany(
            "INSERT OR REPLACE INTO postgame_why_v0 (game_id, player_id, expected, actual, "
            "terms_json, text, built_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r["game_id"], r["player_id"], r["decomp"].expected_dk, r["decomp"].actual_dk,
              json.dumps(terms_blob(r, r["source"]), default=float), r["text"], built)
             for r in picks])
    return {
        "nba_day": day, **stats, "notable": len(picks), "written": len(picks),
        "max_abs_residual": max((abs(r["decomp"].residual) for r in recs), default=0.0),
        "llm": {"asked": len(todo) if (llm and picks) else 0,
                "kept": sum(1 for r in picks if r["source"] == "llm"),
                "fell_back": len(report.get("fell_back", [])),
                "errors": report.get("errors", [])},
        "rows": [{"game_id": r["game_id"], "player_id": r["player_id"], "name": r["name"],
                  "delta": r["decomp"].delta, "source": r["source"], "text": r["text"]}
                 for r in picks],
    }


# ============================================================================== for the Brief

def is_rotation(expected_dk, expected_min, facts: dict | None, *, did_not_play: bool,
                min_dk: float, min_minutes: float) -> bool:
    """A story about a rotation player: projected DK >= `min_dk` or projected minutes >=
    `min_minutes` (both cond 'all'). A player who did not play also qualifies when he was a
    projected starter (p_start >= 0.5, or >= `min_minutes` minutes if he played), so a late
    scratch whose 'all' numbers were cut by a Questionable tag still counts."""
    if _f(expected_dk) >= min_dk or _f(expected_min) >= min_minutes:
        return True
    if did_not_play:
        f = facts or {}
        ps, mp = _opt(f.get("p_start")), _opt(f.get("exp_min_played"))
        return (ps is not None and ps >= 0.5) or (mp is not None and mp >= min_minutes)
    return False


def top_stories(conn: sqlite3.Connection, nba_day, n: int = 5, *, per_game: int = 2,
                max_dnp: int | None = None, min_expected_dk: float | None = None,
                min_expected_min: float | None = None) -> list[dict]:
    """The biggest stored stories of an NBA day for the Brief: [{game_id, player_id, player,
    team, opponent, expected, actual, diff, text, source, terms, minutes, expected_counts,
    actual_counts, facts, pre_tip}] by |diff|, at most `per_game` from one game and at most
    `max_dnp` players who did not play (None = no limit). With `min_expected_dk` /
    `min_expected_min` only rotation players count (`is_rotation`), before the caps apply.
    `terms` is the signed DK-point split
    (minutes, pts, ..., bonus); `minutes` is {expected, actual}; `expected_counts` are the
    projection's full-game stat means; `facts` the stored context facts."""
    day = str(nba_day)[:10]
    gids = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM player_projections "
                                       "WHERE nba_day=?", (day,))}
    gids |= {r[0] for r in conn.execute("SELECT game_id FROM games WHERE nba_day=?", (day,))}
    gids |= {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM team_games WHERE game_date=?",
                                        (day,))}
    if not gids:
        return []
    q = ",".join("?" * len(gids))
    rows = conn.execute(
        f"SELECT game_id, player_id, expected, actual, terms_json, text FROM postgame_why_v0 "
        f"WHERE game_id IN ({q})", sorted(gids)).fetchall()
    rows = sorted(rows, key=lambda r: (-abs((r[3] or 0) - (r[2] or 0)), r[0], r[1]))
    out, seen, dnp = [], defaultdict(int), 0
    for r in rows:
        if seen[r[0]] >= per_game:
            continue
        tj = _json(r[4])
        exp, act = _json(tj.get("expected")), _json(tj.get("actual"))
        did_not_play = _f(act.get("min")) <= 0
        if did_not_play and max_dnp is not None and dnp >= max_dnp:
            continue
        if (min_expected_dk is not None or min_expected_min is not None) and not is_rotation(
                r[2], exp.get("min"), _json(tj.get("facts")), did_not_play=did_not_play,
                min_dk=min_expected_dk if min_expected_dk is not None else float("inf"),
                min_minutes=min_expected_min if min_expected_min is not None else float("inf")):
            continue
        seen[r[0]] += 1
        dnp += int(did_not_play)
        out.append({"game_id": r[0], "player_id": r[1], "player": tj.get("name"),
                    "team": tj.get("team"), "opponent": tj.get("opponent"),
                    "expected": r[2], "actual": r[3], "diff": (r[3] or 0) - (r[2] or 0),
                    "text": r[5], "source": tj.get("text_source"), "terms": tj.get("terms"),
                    "minutes": {"expected": _opt(exp.get("min")), "actual": _opt(act.get("min"))},
                    "expected_counts": _json(exp.get("counts")),
                    "actual_counts": _json(act.get("counts")),
                    "facts": _json(tj.get("facts")), "pre_tip": tj.get("pre_tip")})
        if len(out) >= n:
            break
    return out


# ============================================================================== CLI

def add_commands(sub) -> dict:
    p = sub.add_parser("why", help="post-game why v0: decompose actual vs projected DK points")
    p.add_argument("--date", required=True, help="NBA day, YYYY-MM-DD")
    p.add_argument("--no-llm", action="store_true", help="templates only (no LLM calls)")
    p.add_argument("--historical", action="store_true",
                   help="accept projection runs created after tip (backtest backfills)")
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--top", type=int, default=0, help="rows to print (0 = all written)")
    return {"why": _why_cmd}


def _why_cmd(args: argparse.Namespace) -> int:
    from .. import db
    conn = db.connect()
    s = build(conn, args.date, llm=not args.no_llm, cap=args.cap,
              allow_post_tip=args.historical, log=print)
    print(f"{s['nba_day']}: {s['games_with_run']}/{s['games_projected']} projected games with a "
          f"pre-tip run, {s['games_with_box']} with a box, {s['players']} player-games "
          f"decomposed (max |residual| {s['max_abs_residual']:.2e}), {s['written']} written; "
          f"LLM kept {s['llm']['kept']}/{s['llm']['asked']}")
    for r in (s["rows"][:args.top] if args.top else s["rows"]):
        print(f"[{r['source']:<8}] {r['text']}")
    return 0
