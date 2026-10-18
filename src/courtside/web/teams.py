"""Team identity for the web UI: primary/secondary colours and derived badge colours.

Colours only (no logos: they are trademarked). A badge is the tricode set on the primary
colour with a thin stripe of the secondary; `bar` is a mid-luminance colour that stays visible
on both light and dark surfaces (used for probability bars and chart fills).
"""

from __future__ import annotations

from functools import lru_cache

from .. import teams as _teams

# tricode: (primary, secondary)
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ATL": ("#E03A3E", "#C1D32F"),
    "BOS": ("#007A33", "#BA9653"),
    "BKN": ("#000000", "#FFFFFF"),
    "CHA": ("#1D1160", "#00788C"),
    "CHI": ("#CE1141", "#000000"),
    "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"),
    "DEN": ("#0E2240", "#FEC524"),
    "DET": ("#C8102E", "#1D42BA"),
    "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#C4CED4"),
    "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"),
    "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#12173F"),
    "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"),
    "MIN": ("#0C2340", "#78BE20"),
    "NOP": ("#0C2340", "#C8102E"),
    "NYK": ("#006BB6", "#F58426"),
    "OKC": ("#007AC1", "#EF3B24"),
    "ORL": ("#0077C0", "#C4CED4"),
    "PHI": ("#006BB6", "#ED174C"),
    "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#000000"),
    "SAC": ("#5A2D81", "#63727A"),
    "SAS": ("#000000", "#C4CED4"),
    "TOR": ("#CE1141", "#000000"),
    "UTA": ("#002B5C", "#F9A01B"),
    "WAS": ("#002B5C", "#E31837"),
}

_FALLBACK = ("#5B6170", "#A3A9B5")


def _lum(hex_: str) -> float:
    """WCAG relative luminance."""
    h = hex_.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = out
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@lru_cache(maxsize=64)
def identity(code: str | None) -> dict:
    """{code, primary, secondary, ink, bar, city, nickname, name} for a tricode."""
    code = (code or "").upper()
    primary, secondary = TEAM_COLORS.get(code, _FALLBACK)
    ink = "#FFFFFF" if contrast(primary, "#FFFFFF") >= contrast(primary, "#111111") else "#111111"
    # a bar colour that reads on light (#fff) and dark (#14171c) surfaces alike
    if _lum(primary) >= 0.025:
        bar = primary
    elif _lum(secondary) <= 0.7:
        bar = secondary
    else:
        bar = "#7A808C"
    try:
        t = _teams.by_tricode(code)
        city, nickname = t.city, t.name
    except KeyError:
        city, nickname = "", code
    return {"code": code, "primary": primary, "secondary": secondary, "ink": ink, "bar": bar,
            "city": city, "nickname": nickname,
            "name": f"{city} {nickname}".strip()}
