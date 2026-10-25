"""The "Chances" panel (player page) and the "Running hot / running cold" card (/leaders).

View-model glue over `courtside.stats.opportunity` (formulas in its docstring):
  panel(conn, pid, season, stype, opp)   the player page's panel from player_opportunity(...)
  league_chances(conn, season, stype)    top hot / cold per stat from regression_flags(...)

Information only: the copy says what a rate is and whether it tends to drift back toward
normal; never over / under / edge / lean / bet wording.
"""

from __future__ import annotations

import sqlite3

from ..stats import opportunity as OPP

WINDOWS = (("l5", "Last 5"), ("l10", "Last 10"), ("season", "Season"))
POS_LABEL = {"G": "guards", "F": "forwards", "C": "centers"}
FLAG_LABEL = {"reb": "Rebounds", "ast_pot": "Assists", "fg": "Shooting"}
LEADER_TOP = 10
# |shrunk z| at or above this reads "well above / below"; regression_flags lists from 1.5
STRONG_Z = 2.5
METHOD_TIP = ("Each player's last 10 games of this season type against his own earlier games."
              "\nRebounds: rebounds per rebound chance. Assists: assists per potential"
              " assist. Shooting: makes vs the makes expected from his shot locations"
              " (xFG) plus his usual shot-making."
              "\nHis usual rate is pulled toward the league rate for his position, and the"
              " gap (z) is shrunk when the sample is small, so a few lucky games do not rank."
              "\nRates far from normal tend to drift back toward it.")


def _pct(x, nd=0) -> str:
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def _div(a, b):
    return a / b if a is not None and b else None


# ------------------------------------------------------------------------------- bars

def _bar(label: str, points: list[tuple[str, str, float | None]], *, kind: str = "pct",
         center: float | None = None, note: str | None = None) -> dict | None:
    """A 'conversion vs baseline' strip: markers placed on a padded axis around the values.
    points: (key, label, value) with key in l5 / l10 / season / own / league."""
    vals = [v for _, _, v in points if v is not None]
    if not vals or all(v is None for k, _, v in points if k in ("l5", "l10", "season")):
        return None
    if center is not None:                       # diverging (shot making): symmetric around 0
        half = max(abs(v - center) for v in vals)
        half = max(half * 1.25, 0.05)
        lo, hi = center - half, center + half
    else:
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.3, 0.06)
        lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
        if hi - lo < 0.12:
            hi = min(1.0, lo + 0.12)
            lo = max(0.0, hi - 0.12)
    span = (hi - lo) or 1.0

    def fmt(v):
        if v is None:
            return "–"
        if kind == "pts":
            s = f"{100 * v:+.1f}"
            return "0.0 pts" if s in ("+0.0", "-0.0") else s.replace("-", "−") + " pts"
        if kind == "per100":
            return f"{100 * v:.1f}"
        return _pct(v, 1)

    marks = []
    for key, lab, v in points:
        if v is None:
            continue
        marks.append({"key": key, "label": lab, "value": v, "text": fmt(v),
                      "x": round(min(max((v - lo) / span * 100, 0), 100), 1)})
    aria = f"{label}: " + "; ".join(f"{m['label']} {m['text']}" for m in marks)
    return {"label": label, "marks": marks, "lo": fmt(lo), "hi": fmt(hi), "aria": aria,
            "note": note, "zero": (round((center - lo) / span * 100, 1)
                                   if center is not None else None)}


# ----------------------------------------------------------------------------- signals

def _signal(flag: dict) -> dict:
    m, z = flag["metric"], flag["z_shrunk"]
    hot = z > 0
    strong = abs(z) >= STRONG_Z
    rel = ("well above" if strong else "above") if hot else ("well below" if strong else "below")
    g = flag["games"]
    rate, base = _pct(flag["rate"], 0), _pct(flag["baseline"], 0)
    n, d = flag["n"], flag["d"]
    if m == "reb":
        what = (f"Converting {rate} of rebound chances over his last {g} games vs his usual "
                f"{base}")
        detail = f"{n:.0f} rebounds on {d:.0f} chances"
    elif m == "ast_pot":
        what = (f"Turning {rate} of potential assists into assists over his last {g} games vs "
                f"his usual {base}")
        detail = f"{n:.0f} assists on {d:.0f} potential assists"
    else:
        what = (f"Making {rate} of his shots over his last {g} games vs {base} expected from "
                f"his shot locations and usual shot-making")
        detail = f"{n:.0f} of {d:.0f} field goals, {flag['expected']:.0f} expected"
    tail = "likely to cool off" if hot else "likely to pick back up"
    return {"metric": m, "label": FLAG_LABEL.get(m, m), "dir": "hot" if hot else "cold",
            "strong": strong, "rel": rel,
            "text": f"{what}: {tail}.",
            "tip": (f"{detail}. Shrunk z {z:+.1f}".replace("-", "−")
                    + f" ({rel} his normal rate; |z| of {STRONG_Z:g}+ reads 'well').")}


def _quiet(metric: str, enough: bool) -> dict:
    lab = {"reb": "Rebound conversion", "ast_pot": "Assists per potential assist",
           "fg": "Shot-making"}[metric]
    text = (f"{lab} over the last 10 games is within his normal range." if enough else
            f"Too few chances in the last 10 games to judge {lab[0].lower() + lab[1:]}.")
    return {"metric": metric, "label": FLAG_LABEL[metric], "dir": "flat", "strong": False,
            "rel": "", "text": text, "tip": None,
            "tag": "Normal" if enough else "Small sample"}


# ------------------------------------------------------------------- contested shots

def _contested(conn, pid: int, season: str, stype: str) -> dict:
    """{window: {cont_fga, uncont_fga, cont_share, cont_fg, open_fg, games}} from
    player_tracking, windows as player_opportunity (games of the season type, by date)."""
    out = {}
    try:
        gids = [r[0] for r in conn.execute(
            "SELECT game_id FROM player_games WHERE player_id=? AND season=? AND season_type=? "
            "ORDER BY game_date, game_id", (pid, season, stype))]
        rows = {}
        if gids:
            q = ",".join("?" * len(gids))
            for g, cm, ca, um, ua in conn.execute(
                    f"SELECT game_id, cont_fgm, cont_fga, uncont_fgm, uncont_fga FROM "
                    f"player_tracking WHERE player_id=? AND game_id IN ({q})", (pid, *gids)):
                if ca is not None or ua is not None:
                    rows[g] = (cm or 0, ca or 0, um or 0, ua or 0)
    except sqlite3.OperationalError:
        return out
    for key, n in (("l5", 5), ("l10", 10), ("season", None)):
        sel = [rows[g] for g in (gids[-n:] if n else gids) if g in rows]
        if not sel:
            out[key] = None
            continue
        cm, ca, um, ua = (sum(x[i] for x in sel) for i in range(4))
        out[key] = {"games": len(sel), "cont_fga": ca, "uncont_fga": ua,
                    "cont_share": _div(ca, ca + ua), "cont_fg": _div(cm, ca),
                    "open_fg": _div(um, ua)}
    return out


# ------------------------------------------------------------------------------ panel

def panel(conn, pid: int, season: str, stype: str, opp: dict | None) -> dict | None:
    """The template's view of player_opportunity(...) (None = nothing collected)."""
    if not opp:
        return None
    has = opp["has"]
    pos = opp.get("position")
    R, A, S, D = opp["rebounds"], opp["assists"], opp["shooting"], opp["defense"]
    league_lab = f"League {POS_LABEL[pos]}" if pos in POS_LABEL else "League"

    # plausibility guard, window level (per-game rows are already guarded in opportunity.py)
    excluded = A["season"].get("pot_ast_excluded") or 0
    pot_ok = {}
    for w, _ in WINDOWS:
        a = A[w]
        bad = ((a.get("pot_ast_pg") or 0) > OPP.POT_AST_MAX
               or (a.get("ast_per_pot_ast") is not None
                   and a["ast_per_pot_ast"] < 1 / OPP.POT_AST_RATIO))
        pot_ok[w] = not bad and a.get("pot_ast_games", 0) > 0
    pot_check = excluded > 0 or any(not pot_ok[w] and A[w].get("pot_ast_games", 0) > 0
                                    for w, _ in WINDOWS)
    pot_any = any(pot_ok.values())

    def wpts(block, field, *, own=None, league=None, ok=None, lg_lab=None):
        pts = [(w, lab, (block[w][field] if (ok is None or ok[w]) else None))
               for w, lab in WINDOWS]
        return pts + [("own", "His season", own), ("league", lg_lab or league_lab, league)]

    bars = {}
    if has["tracking"]:
        bars["reb"] = _bar("Rebound chances converted",
                           wpts(R, "conv", own=R["baseline"]["own"],
                                league=R["baseline"]["league"]))
        bars["pass"] = _bar("Assists per 100 passes",
                            wpts(A, "ast_per_pass", own=A["baseline"]["own_per_pass"],
                                 league=A["baseline"]["league_per_pass"]), kind="per100")
    if pot_any:
        bars["pot"] = _bar("Assists per potential assist",
                           wpts(A, "ast_per_pot_ast", own=A["baseline"]["own"],
                                league=A["baseline"]["league"], ok=pot_ok))
    if has["shots"]:
        bars["fg"] = _bar("Shot-making: FG% minus expected FG%",
                          wpts(S, "making_fg_pct", own=S["baseline"]["own"], league=0.0,
                               lg_lab="League"),
                          kind="pts", center=0.0,
                          note="0 = exactly what his shot locations predict (the league "
                               "average by construction)")
    if any(D[w]["rim_games"] for w, _ in WINDOWS):
        bars["rim"] = _bar("Rim FG% allowed (lower is better)",
                           wpts(D, "rim_fg_allowed", own=D["baseline"]["own"],
                                league=D["baseline"]["league"]))

    # signals: one line per conversion the page has data for
    flags = {f["metric"]: f for f in opp.get("flags") or []}
    signals = []
    checks = (("reb", has["tracking"], (R["l10"]["chances"] or 0) >= OPP.MIN_WINDOW["reb"]),
              ("ast_pot", pot_any, (A["l10"]["pot_ast"] or 0) >= OPP.MIN_WINDOW["ast_pot"]),
              ("fg", has["shots"], (S["l10"]["fga"] or 0) >= OPP.MIN_WINDOW["fg"]))
    for m, show, enough in checks:
        if not show:
            continue
        signals.append(_signal(flags[m]) if m in flags else _quiet(m, enough))

    return {"raw": opp, "has": has, "pos": pos, "league_lab": league_lab,
            "windows": WINDOWS, "tracking": has["tracking"] or has["hustle"] or has["pot_ast"],
            "pot_ok": pot_ok, "pot_any": pot_any, "pot_check": pot_check,
            "pot_excluded": excluded, "pot_max": OPP.POT_AST_MAX,
            "pot_ratio": OPP.POT_AST_RATIO,
            "contested": _contested(conn, pid, season, stype) if has["tracking"] else {},
            "bars": bars, "signals": signals}


# ---------------------------------------------------------------------------- leaders

def league_flags(conn, season: str, stype: str) -> dict:
    """regression_flags for the season type, cached under the key player_opportunity uses."""
    return OPP.flags_cached(conn, season, stype) or {}


def league_chances(conn, season: str, stype: str, top: int = LEADER_TOP) -> dict:
    """{"metrics": [{key, label, unit, hot: [...], cold: [...]}], "any": bool}."""
    try:
        fl = league_flags(conn, season, stype)
    except sqlite3.OperationalError:
        fl = {}
    units = {"reb": "of rebound chances", "ast_pot": "of potential assists",
             "fg": "FG% (usual = expected)"}
    out = []
    for m in OPP.FLAG_METRICS:
        hot = [f for f in fl.get("hot", []) if f["metric"] == m][:top]
        cold = [f for f in fl.get("cold", []) if f["metric"] == m][:top]
        out.append({"key": m, "label": FLAG_LABEL[m], "unit": units[m], "hot": hot,
                    "cold": cold})
    return {"metrics": out, "any": any(x["hot"] or x["cold"] for x in out),
            "tip": METHOD_TIP}
