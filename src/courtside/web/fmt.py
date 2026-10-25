"""Number and text formatting used by every template, so numbers look the same everywhere.

All helpers accept None (and non-numbers) and return "" (or a dash where noted), so a template
never prints "None".
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

MINUS = "−"
DASH = "–"          # en dash: scores, records, ranges
EM = "—"
ET = ZoneInfo("America/New_York")


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except Exception:  # noqa: BLE001 - None, strings, jinja Undefined: render as missing
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _minus(s: str) -> str:
    return s.replace("-", MINUS)


def num(x, nd: int = 1, empty: str = EM) -> str:
    """Fixed decimals, typographic minus: 117.4, −3.8."""
    v = _num(x)
    if v is None:
        return empty
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
    return _minus(s)


def signed(x, nd: int = 1, empty: str = EM) -> str:
    """+14.5 / −3.8 / 0.0 (zero carries no sign)."""
    v = _num(x)
    if v is None:
        return empty
    s = f"{v:+.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return _minus(s)


def signed_int(x, empty: str = EM) -> str:
    return signed(x, 0, empty)


def pct(x, nd: int = 0, empty: str = EM) -> str:
    """Probability/fraction in [0, 1] -> '60%' / '64.4%'."""
    v = _num(x)
    if v is None:
        return empty
    return _minus(f"{100 * v:.{nd}f}%")


def pct_points(x, nd: int = 0, empty: str = EM) -> str:
    """Probability difference -> '+19 pts' (signed percentage points)."""
    v = _num(x)
    if v is None:
        return empty
    s = signed(100 * v, nd)
    return f"{s} pts"


def ts(x, empty: str = EM) -> str:
    """True shooting fraction -> '59.2%'."""
    return pct(x, 1, empty)


def prob3(x, empty: str = EM) -> str:
    """Brier/log-loss style: 0.224."""
    v = _num(x)
    return empty if v is None else f"{v:.3f}"


def ml(x, empty: str = EM) -> str:
    """American moneyline: +160 / −192."""
    v = _num(x)
    if v is None:
        return empty
    return _minus(f"{v:+.0f}")


def spread_text(team: str | None, home_spread, home: str | None, away: str | None,
                empty: str = EM) -> str:
    """'CLE −5.5' (always quoted from the favourite's side) or 'Pick’em'."""
    v = _num(home_spread)
    if v is None or not home or not away:
        return empty
    if v == 0:
        return "Pick’em"
    fav, line = (home, v) if v < 0 else (away, -v)
    return f"{fav} {num(line, 1)}"


def margin_text(home_margin, home: str | None, away: str | None, empty: str = EM) -> str:
    """Projected home margin -> 'CHA by 1.1'."""
    v = _num(home_margin)
    if v is None or not home or not away:
        return empty
    if abs(v) < 0.05:
        return "Even"
    return f"{home if v > 0 else away} by {abs(v):.1f}"


def record(r: str | None, empty: str = "") -> str:
    return r.replace("-", DASH) if r else empty


def score(a, b) -> str:
    return f"{a}{DASH}{b}"


def run(r: str | None) -> str:
    return r.replace("-", DASH) if r else ""


def int_(x, empty: str = EM) -> str:
    v = _num(x)
    return empty if v is None else f"{v:,.0f}"


def minutes(x, empty: str = EM) -> str:
    """Decimal minutes -> '29:18'."""
    v = _num(x)
    if v is None:
        return empty
    m = int(v)
    s = int(round((v - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


def seconds_clock(x, empty: str = EM) -> str:
    v = _num(x)
    if v is None:
        return empty
    return f"{int(v) // 60}:{int(v) % 60:02d}"


def parse_day(s) -> date | None:
    if isinstance(s, date):
        return s
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def long_date(s) -> str:
    """'Friday, February 20, 2026'."""
    d = parse_day(s)
    return f"{d:%A}, {d:%B} {d.day}, {d.year}" if d else ""


def short_date(s) -> str:
    """'Feb 20'."""
    d = parse_day(s)
    return f"{d:%b} {d.day}" if d else ""


def medium_date(s) -> str:
    """'Fri, Feb 20'."""
    d = parse_day(s)
    return f"{d:%a}, {d:%b} {d.day}" if d else ""


def weekday(s) -> str:
    """'Fri'."""
    d = parse_day(s)
    return f"{d:%a}" if d else ""


def _dt(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def tipoff(s, empty: str = "") -> str:
    """UTC ISO -> '7:30 PM ET'."""
    dt = _dt(s)
    if dt is None or dt.tzinfo is None:
        return empty
    t = dt.astimezone(ET)
    return f"{t:%I:%M}".lstrip("0") + f" {t:%p} ET"


def stamp(s, empty: str = "") -> str:
    """UTC ISO -> 'Feb 20, 5:00 PM ET'."""
    dt = _dt(s)
    if dt is None or dt.tzinfo is None:
        return empty
    t = dt.astimezone(ET)
    return f"{t:%b} {t.day}, " + f"{t:%I:%M}".lstrip("0") + f" {t:%p} ET"


_INJ_PREFIX = re.compile(r"^\s*injury/illness\s*-\s*", re.I)


def injury_reason(s: str | None) -> str:
    """'Injury/Illness - Right Hamstring; Tightness' -> 'Right hamstring tightness'."""
    if not s:
        return ""
    s = _INJ_PREFIX.sub("", s).strip()
    parts = [p.strip() for p in s.split(";") if p.strip() and p.strip().lower() != "n/a"]
    if not parts:
        return ""
    out = parts[0] + (f" {parts[1]}" if len(parts) > 1 else "")
    if len(parts) > 2:
        out += ", " + ", ".join(parts[2:])
    words = out.split()
    words = [w if i == 0 or (w.isupper() and len(w) > 1) else w.lower()
             for i, w in enumerate(words)]
    return " ".join(words)


def possessive(name: str) -> str:
    return f"{name}’" if name.endswith("s") else f"{name}’s"


def last_name(name: str | None) -> str:
    if not name:
        return ""
    parts = name.split()
    suffixes = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "v"}
    while len(parts) > 1 and parts[-1].lower() in suffixes:
        parts = parts[:-1]
    return parts[-1] if parts else name


def ordinal(x, empty: str = "") -> str:
    """1 -> '1st', 2 -> '2nd', 13 -> '13th', 22 -> '22nd'."""
    v = _num(x)
    if v is None:
        return empty
    n = int(round(v))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


RANK_BEST, RANK_MID, RANK_WORST = "#2a6fc4", "#3a3f4b", "#c44a4a"


def _mix(a: str, b: str, t: float) -> str:
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def rank_color(rank, n=30) -> str:
    """League-rank chip colour, diverging: 1st blue -> middle gray -> last red. Rank 1 is
    always the best (callers rank "lower is better" stats ascending). Pair it with the rank
    text: colour is never the only signal."""
    r, nn = _num(rank), _num(n)
    if r is None:
        return RANK_MID
    nn = max(nn or 30, 2)
    t = min(max((r - 1) / (nn - 1), 0.0), 1.0)
    return _mix(RANK_BEST, RANK_MID, t * 2) if t < .5 else _mix(RANK_MID, RANK_WORST, (t - .5) * 2)


def dollars(x, empty: str = EM) -> str:
    """11400 -> '$11,400'."""
    v = _num(x)
    if v is None:
        return empty
    s = f"${abs(v):,.0f}"
    return MINUS + s if v < 0 else s


FILTERS = {
    "num": num, "signed": signed, "signed_int": signed_int, "pct": pct,
    "pct_points": pct_points, "ts": ts, "prob3": prob3, "ml": ml, "record": record, "run": run,
    "int": int_, "minutes": minutes, "long_date": long_date, "short_date": short_date,
    "medium_date": medium_date, "weekday": weekday, "tipoff": tipoff, "stamp": stamp,
    "injury_reason": injury_reason, "last_name": last_name, "possessive": possessive,
    "seconds_clock": seconds_clock, "ordinal": ordinal, "rank_color": rank_color,
    "dollars": dollars,
}
