"""Server-side SVG for the Model Health page: multi-series time lines. Colours come from CSS
classes (`.hm-line.s1` ...) so the charts follow the light/dark theme like the rest of the site.

Every chart has a text equivalent on the page (scoreboard, tables); `<title>` gives hover text.
"""

from __future__ import annotations

import math
from datetime import date
from html import escape

from . import fmt


def _f(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def nice_ticks(lo: float, hi: float, want: int = 4) -> list[float]:
    """Round tick values spanning [lo, hi] (inclusive of the covering multiples)."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi - lo < 1e-9:
        hi, lo = hi + 0.5, lo - 0.5
    raw = (hi - lo) / max(1, want)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step
    out, v = [], start
    while v <= hi + step * 0.999 and len(out) < 12:
        out.append(round(v, 10))
        v += step
    return out


def _day(s) -> date | None:
    return fmt.parse_day(s)


def line_chart(series: list[dict], *, W: int, H: int, variant: str, ytick, aria: str,
               zero: bool = False, min_span: float = 0.0, pad_frac: float = 0.12) -> str | None:
    """Lines over calendar time.

    series: [{"key", "label", "cls" (s1 | s2 | s3 | mk), "pts": [(day, value, n)]}]
    ytick: value -> axis label. Returns None when no series has two points.
    """
    prepared = []
    for s in series:
        pts = []
        for d, v, n in s.get("pts") or []:
            dd = _day(d)
            if dd is not None and v is not None and math.isfinite(v):
                pts.append((dd, float(v), n))
        pts.sort(key=lambda p: p[0])
        if pts:
            prepared.append({**s, "pts": pts})
    if not any(len(s["pts"]) >= 2 for s in prepared):
        return None

    days = [p[0] for s in prepared for p in s["pts"]]
    d0, d1 = min(days), max(days)
    span_days = max(1, (d1 - d0).days)
    vals = [p[1] for s in prepared for p in s["pts"]]
    lo, hi = min(vals), max(vals)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < min_span:
        mid = (hi + lo) / 2
        lo, hi = mid - min_span / 2, mid + min_span / 2
    pad = (hi - lo) * pad_frac or 0.01
    lo, hi = lo - pad, hi + pad
    ticks = nice_ticks(lo, hi, 4)
    if ticks:
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
        ticks = [t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]

    narrow = variant == "is-narrow"
    L, R, T, B = (44, 12, 12, 26) if narrow else (50, 16, 12, 28)
    pw, ph = W - L - R, H - T - B

    def X(d: date) -> float:
        return L + pw * (d - d0).days / span_days

    def Y(v: float) -> float:
        return T + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg class="chart hm-chart {variant}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{escape(aria)}">']
    for t in ticks:
        out.append(f'<line class="ch-grid" x1="{L}" x2="{W - R}" y1="{_f(Y(t))}" y2="{_f(Y(t))}"/>')
        out.append(f'<text class="ch-tick" x="{L - 7}" y="{_f(Y(t) + 3.5)}" text-anchor="end">'
                   f'{escape(ytick(t))}</text>')
    if zero and lo < 0 < hi:
        out.append(f'<line class="ch-zero" x1="{L}" x2="{W - R}" y1="{_f(Y(0))}" y2="{_f(Y(0))}"/>')
    # x ticks: first of each month (thinned to fit), always labelled by month
    months, y, m = [], d0.year, d0.month
    while True:
        first = date(y, m, 1)
        if first > d1:
            break
        if first >= d0:
            months.append(first)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    max_labels = 4 if narrow else 7
    if len(months) > max_labels:
        k = math.ceil(len(months) / max_labels)
        months = months[::k]
    for first in months:
        x = X(first)
        out.append(f'<line class="ch-grid ch-vgrid" x1="{_f(x)}" x2="{_f(x)}" y1="{T}" '
                   f'y2="{T + ph}"/>')
        out.append(f'<text class="ch-tick" x="{_f(x)}" y="{H - 8}" text-anchor="middle">'
                   f'{first:%b}</text>')
    if not months:  # a window shorter than a month: label the ends
        out.append(f'<text class="ch-tick" x="{L}" y="{H - 8}" text-anchor="start">'
                   f'{escape(fmt.short_date(d0))}</text>')
        out.append(f'<text class="ch-tick" x="{W - R}" y="{H - 8}" text-anchor="end">'
                   f'{escape(fmt.short_date(d1))}</text>')

    # draw the benchmark (market) first so model lines sit on top
    for s in sorted(prepared, key=lambda s: s.get("cls") != "mk"):
        pts = s["pts"]
        poly = " ".join(f"{_f(X(d))},{_f(Y(v))}" for d, v, _ in pts)
        cls = s.get("cls", "s1")
        if len(pts) >= 2:
            out.append(f'<polyline class="hm-line {cls}" points="{poly}"/>')
        step = max(1, len(pts) // 28)
        for i, (d, v, n) in enumerate(pts):
            if i % step and i != len(pts) - 1:
                continue
            tip = f'{s["label"]}, {fmt.short_date(d)}: {ytick(v)}' + (f" ({n} games)" if n else "")
            last = i == len(pts) - 1
            out.append(f'<g class="ch-hit"><circle class="ch-hit-area" cx="{_f(X(d))}" '
                       f'cy="{_f(Y(v))}" r="9"/>'
                       + (f'<circle class="hm-dot {cls}" cx="{_f(X(d))}" cy="{_f(Y(v))}" r="4"/>'
                          if last else
                          f'<circle class="hm-pt {cls}" cx="{_f(X(d))}" cy="{_f(Y(v))}" r="2.6"/>')
                       + f'<title>{escape(tip)}</title></g>')
    out.append("</svg>")
    return "".join(out)


def chart_pair(series: list[dict], *, ytick, aria: str, zero: bool = False,
               min_span: float = 0.0) -> dict | None:
    """Wide + narrow variants (CSS shows one), like the game lead chart."""
    wide = line_chart(series, W=600, H=250, variant="is-wide", ytick=ytick, aria=aria,
                      zero=zero, min_span=min_span)
    if wide is None:
        return None
    narrow = line_chart(series, W=360, H=230, variant="is-narrow", ytick=ytick, aria=aria,
                        zero=zero, min_span=min_span)
    return {"wide": wide, "narrow": narrow}


def small_chart(series: list[dict], *, ytick, aria: str, zero: bool = False,
                min_span: float = 0.0) -> str | None:
    return line_chart(series, W=360, H=200, variant="is-small", ytick=ytick, aria=aria,
                      zero=zero, min_span=min_span)
