"""Inline SVG charts, rendered server-side. Colours come from CSS classes/variables so every
chart follows the light/dark theme; team colours are passed as inline custom properties.

Every chart has a text equivalent nearby in the page (table or labels); `<title>` elements give
native hover tooltips.
"""

from __future__ import annotations

import math
import re
from html import escape

from . import fmt
from .teams import identity

# ------------------------------------------------------------------ helpers


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else str(int(round(x)))


def period_seconds(label: str | None) -> tuple[int, int] | None:
    """'Q3' -> (start elapsed, length); 'OT' / '2OT' -> overtime periods."""
    if not label:
        return None
    m = re.fullmatch(r"Q([1-4])", label)
    if m:
        q = int(m.group(1))
        return (q - 1) * 720, 720
    m = re.fullmatch(r"(\d*)OT", label)
    if m:
        n = int(m.group(1) or 1)
        return 2880 + (n - 1) * 300, 300
    return None


def clock_seconds(clock: str | None) -> float | None:
    if not clock:
        return None
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", str(clock).strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def elapsed(label: str | None, clock: str | None) -> float | None:
    p = period_seconds(label)
    c = clock_seconds(clock)
    if p is None or c is None:
        return None
    start, length = p
    return start + max(0.0, min(length, length - c))


def period_labels(overtimes: int) -> list[tuple[str, int, int]]:
    out = [(f"Q{q}", (q - 1) * 720, 720) for q in range(1, 5)]
    for n in range(1, (overtimes or 0) + 1):
        out.append(("OT" if n == 1 else f"{n}OT", 2880 + (n - 1) * 300, 300))
    return out


# ------------------------------------------------------------------ score timeline

def score_timeline(facts: dict) -> tuple[list[tuple[float, int, int]], str]:
    """[(elapsed s, home pts, away pts)] and its provenance ('timeline' | 'checkpoints').

    Uses a real score timeline if the facts carry one (`score_timeline`: [[t, home, away]] or
    [{"t"/"elapsed", "home", "away"}]); otherwise reconstructs checkpoints from quarter scores,
    runs, largest leads, the comeback point and clutch key plays.
    """
    home = (facts.get("home") or {}).get("team")
    away = (facts.get("away") or {}).get("team")
    if not home or not away:
        return [], "none"
    raw = facts.get("score_timeline") or facts.get("timeline")
    pts: list[tuple[float, int, int]] = []
    if isinstance(raw, list) and raw:
        for r in raw:
            try:
                if isinstance(r, dict):
                    t = r.get("t", r.get("elapsed"))
                    pts.append((float(t), int(r["home"]), int(r["away"])))
                else:
                    pts.append((float(r[0]), int(r[1]), int(r[2])))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        if len(pts) >= 2:
            return sorted(pts), "timeline"
        pts = []

    def add(label, clock, score):
        if not isinstance(score, dict):
            return
        t = elapsed(label, clock)
        h, a = score.get(home), score.get(away)
        if t is None or h is None or a is None:
            return
        pts.append((t, int(h), int(a)))

    pts.append((0.0, 0, 0))
    hp = (facts.get("home") or {}).get("periods") or []
    ap = (facts.get("away") or {}).get("periods") or []
    labels = period_labels(max(0, len(hp) - 4))
    ch = ca = 0
    for (lab, start, length), h, a in zip(labels, hp, ap):
        if h is None or a is None:
            break
        ch, ca = ch + h, ca + a
        pts.append((float(start + length), ch, ca))
    for r in facts.get("runs") or []:
        for end in ("from", "to"):
            e = r.get(end) or {}
            add(e.get("label"), e.get("clock"), e.get("score"))
    tp = facts.get("turning_point") or {}
    for end in ("from", "to"):
        e = tp.get(end) or {}
        add(e.get("label"), e.get("clock"), e.get("score"))
    for ll in (facts.get("largest_lead") or {}).values():
        if isinstance(ll, dict):
            add(ll.get("label"), ll.get("clock"), ll.get("score"))
    cb = (facts.get("comeback") or {}).get("at") or {}
    add(cb.get("label"), cb.get("clock"), cb.get("score"))
    for kp in ((facts.get("clutch") or {}).get("key_plays") or []):
        add(kp.get("label"), kp.get("clock"), kp.get("score"))
    # de-duplicate and keep only checkpoints consistent with monotone running scores
    pts = sorted(set(pts))
    clean: list[tuple[float, int, int]] = []
    for t, h, a in pts:
        if clean and t == clean[-1][0]:
            if h >= clean[-1][1] and a >= clean[-1][2]:
                clean[-1] = (t, h, a)
            continue
        if clean and (h < clean[-1][1] or a < clean[-1][2]):
            continue
        clean.append((t, h, a))
    return clean, "checkpoints"


def lead_chart(facts: dict) -> dict | None:
    """Lead-margin chart for a game: {'svg' (wide), 'svg_narrow' (phone), 'source', 'n'}.

    None if there are too few score points to draw a meaningful line.
    """
    pts, source = score_timeline(facts)
    if len(pts) < 4:
        return None
    return {"svg": _lead_svg(facts, pts, 1000, 300, "is-wide"),
            "svg_narrow": _lead_svg(facts, pts, 360, 250, "is-narrow"),
            "source": source, "n": len(pts)}


def _lead_svg(facts: dict, pts: list, W: int, H: int, variant: str) -> str:
    home = identity((facts.get("home") or {}).get("team"))
    away = identity((facts.get("away") or {}).get("team"))
    ot = max(0, len((facts.get("home") or {}).get("periods") or []) - 4,
             facts.get("overtimes") or 0)
    periods = period_labels(ot)
    t_end = periods[-1][1] + periods[-1][2]
    L, R, T, B = (30, 10, 16, 28) if W < 600 else (36, 14, 16, 30)
    pw, ph = W - L - R, H - T - B
    peak = max(abs(h - a) for _, h, a in pts)
    ymax = next((y for y in (10, 15, 20, 30, 40, 50, 60) if y >= peak + 2), 70)
    ticks = {15: (5, 10, 15)}.get(ymax, (ymax // 2, ymax))

    def X(t):
        return L + pw * min(max(t, 0), t_end) / t_end

    def Y(m):
        return T + ph / 2 - (ph / 2) * m / ymax

    out = [f'<svg class="chart lead-chart {variant}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Lead margin by game time, {escape(home["code"])} above the line, '
           f'{escape(away["code"])} below" style="--home:{home["bar"]};--away:{away["bar"]}">']
    # grid: horizontal at +-ymax/2, ymax ; vertical period boundaries
    for m in [t for t in ticks] + [-t for t in ticks]:
        y = Y(m)
        out.append(f'<line class="ch-grid" x1="{L}" x2="{W - R}" y1="{_f(y)}" y2="{_f(y)}"/>')
        out.append(f'<text class="ch-tick" x="{L - 6}" y="{_f(y + 3.5)}" text-anchor="end">'
                   f'{abs(m)}</text>')
    for lab, start, length in periods:
        x0 = X(start)
        if start > 0:
            out.append(f'<line class="ch-grid" x1="{_f(x0)}" x2="{_f(x0)}" y1="{T}" '
                       f'y2="{H - B}"/>')
        out.append(f'<text class="ch-tick" x="{_f(X(start + length / 2))}" y="{H - 10}" '
                   f'text-anchor="middle">{lab}</text>')
    # turning point band
    tp = facts.get("turning_point") or {}
    t0 = elapsed((tp.get("from") or {}).get("label"), (tp.get("from") or {}).get("clock"))
    t1 = elapsed((tp.get("to") or {}).get("label"), (tp.get("to") or {}).get("clock"))
    if t0 is not None and t1 is not None and t1 > t0:
        x0, x1 = X(t0), X(t1)
        out.append(f'<rect class="ch-band" x="{_f(x0)}" y="{T}" width="{_f(max(2, x1 - x0))}" '
                   f'height="{ph}"><title>Turning point: {escape(fmt.run(tp.get("run")))} '
                   f'{escape(tp.get("team") or "")} run</title></rect>')
        out.append(f'<text class="ch-band-label" x="{_f((x0 + x1) / 2)}" y="{T + 11}" '
                   f'text-anchor="middle">Turning point</text>')
    # area (clipped above/below zero) + line
    zero = Y(0)
    line = " ".join(f"{_f(X(t))},{_f(Y(h - a))}" for t, h, a in pts)
    last_t = pts[-1][0]
    area = f"M{_f(X(pts[0][0]))},{_f(zero)} L" + line.replace(" ", " L") + \
        f" L{_f(X(last_t))},{_f(zero)} Z"
    cid = f"lc{variant}{abs(hash(line)) % 10_000_000}"
    out.append(f'<clipPath id="{cid}a"><rect x="0" y="0" width="{W}" height="{_f(zero)}"/>'
               f'</clipPath><clipPath id="{cid}b"><rect x="0" y="{_f(zero)}" width="{W}" '
               f'height="{_f(H - zero)}"/></clipPath>')
    out.append(f'<path class="ch-area-home" clip-path="url(#{cid}a)" d="{area}"/>')
    out.append(f'<path class="ch-area-away" clip-path="url(#{cid}b)" d="{area}"/>')
    out.append(f'<line class="ch-zero" x1="{L}" x2="{W - R}" y1="{_f(zero)}" y2="{_f(zero)}"/>')
    out.append(f'<polyline class="ch-line" points="{line}"/>')
    # largest-lead markers
    for code, ll in (facts.get("largest_lead") or {}).items():
        if not isinstance(ll, dict) or not ll.get("lead"):
            continue
        t = elapsed(ll.get("label"), ll.get("clock"))
        if t is None:
            continue
        m = ll["lead"] if code == home["code"] else -ll["lead"]
        x, y = X(t), Y(m)
        out.append(f'<circle class="ch-dot" cx="{_f(x)}" cy="{_f(y)}" r="4.5">'
                   f'<title>{escape(code)} largest lead: {ll["lead"]} '
                   f'({ll.get("label")} {ll.get("clock")})</title></circle>')
        dy = -9 if m > 0 else 17
        anchor = "end" if x > W - 60 else ("start" if x < 70 else "middle")
        out.append(f'<text class="ch-label" x="{_f(x)}" y="{_f(y + dy)}" '
                   f'text-anchor="{anchor}">{escape(code)} +{ll["lead"]}</text>')
    # end label
    t, h, a = pts[-1]
    out.append(f'<circle class="ch-end" cx="{_f(X(t))}" cy="{_f(Y(h - a))}" r="4"/>')
    # side legend
    out.append(f'<text class="ch-side" x="{L + 6}" y="{T + 12}">{escape(home["code"])} ahead'
               f'</text>')
    out.append(f'<text class="ch-side" x="{L + 6}" y="{H - B - 6}">{escape(away["code"])} '
               f'ahead</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ calibration

def calibration_chart(bins: list[dict]) -> str | None:
    pts = [b for b in bins or [] if b.get("n") and b.get("mean_p") is not None
           and b.get("frac_home_won") is not None]
    if len(pts) < 2:
        return None
    W, H = 340, 270
    L, R, T, B = 40, 14, 12, 40
    pw, ph = W - L - R, H - T - B
    nmax = max(b["n"] for b in pts)

    def X(p):
        return L + pw * p

    def Y(p):
        return T + ph * (1 - p)

    out = [f'<svg class="chart calib-chart" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Calibration: predicted home win probability against how often the home '
           f'team won">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="ch-grid" x1="{L}" x2="{W - R}" y1="{_f(Y(v))}" '
                   f'y2="{_f(Y(v))}"/>')
        out.append(f'<text class="ch-tick" x="{L - 6}" y="{_f(Y(v) + 3.5)}" '
                   f'text-anchor="end">{int(v * 100)}%</text>')
        out.append(f'<text class="ch-tick" x="{_f(X(v))}" y="{H - 20}" '
                   f'text-anchor="middle">{int(v * 100)}%</text>')
    out.append(f'<text class="ch-axis-title" x="{_f(L + pw / 2)}" y="{H - 3}" '
               f'text-anchor="middle">Our home win probability</text>')
    out.append(f'<line class="ch-diag" x1="{_f(X(0))}" y1="{_f(Y(0))}" x2="{_f(X(1))}" '
               f'y2="{_f(Y(1))}"/>')
    out.append(f'<text class="ch-diag-label" x="{_f(X(0.70))}" y="{_f(Y(0.78))}" '
               f'text-anchor="end">perfect calibration</text>')
    line = " ".join(f"{_f(X(b['mean_p']))},{_f(Y(b['frac_home_won']))}" for b in pts)
    out.append(f'<polyline class="ch-line ch-accent" points="{line}"/>')
    for b in pts:
        r = 3.5 + 4.5 * math.sqrt(b["n"] / nmax)
        out.append(f'<g class="ch-hit"><circle class="ch-hit-area" cx="{_f(X(b["mean_p"]))}" '
                   f'cy="{_f(Y(b["frac_home_won"]))}" r="12"/>'
                   f'<circle class="ch-dot ch-accent" cx="{_f(X(b["mean_p"]))}" '
                   f'cy="{_f(Y(b["frac_home_won"]))}" r="{_f(r)}"/>'
                   f'<title>We said {fmt.pct(b["mean_p"])}: home won {fmt.pct(b["frac_home_won"])}'
                   f' of {b["n"]} games</title></g>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ dumbbell (then -> now)

def dumbbell(before, after, lo: float, hi: float, *, good_up: bool = True,
             width: int = 240, label: str = "") -> str:
    """Two dots on a shared scale joined by a bar: before (hollow) -> after (filled)."""
    try:
        b, a = float(before), float(after)
    except (TypeError, ValueError):
        return ""
    W, H, P = width, 22, 6

    def X(v):
        v = min(max(v, lo), hi)
        return P + (W - 2 * P) * (v - lo) / (hi - lo)

    up = a >= b
    cls = "pos" if up == good_up else "neg"
    xb, xa = X(b), X(a)
    zero = ""
    if lo < 0 < hi:
        zero = (f'<line class="db-zero" x1="{_f(X(0))}" x2="{_f(X(0))}" y1="4" '
                f'y2="{H - 4}"/>')
    return (f'<svg class="dumbbell {cls}" viewBox="0 0 {W} {H}" '
            f'role="img" aria-label="{escape(label)}">'
            f'<line class="db-track" x1="{P}" x2="{W - P}" y1="{H / 2}" y2="{H / 2}"/>{zero}'
            f'<line class="db-bar" x1="{_f(xb)}" x2="{_f(xa)}" y1="{H / 2}" y2="{H / 2}"/>'
            f'<circle class="db-before" cx="{_f(xb)}" cy="{H / 2}" r="4"/>'
            f'<circle class="db-after" cx="{_f(xa)}" cy="{H / 2}" r="5"/>'
            f'<title>{escape(label)}</title></svg>')
