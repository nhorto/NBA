"""``shot_xfg``: a transparent per-season expected make probability for every field-goal attempt
in ``shots`` (stats-site "shot quality"; formulas also in stats/opportunity.py).

Cell of a shot (same 2 / 3 split, distance bins and shot families as the projection model's F4
features, ``proj/features/shots.py``: ``dist_bin`` and ``SUBTYPES``):
  zone         ``shots.zone`` (Restricted Area, Paint (Non-RA), Mid-Range, Left / Right Corner 3,
               Above the Break 3, Backcourt; 'Unknown' without coordinates)
  dist_band    '<2|3>pt <ft>': 1-ft bands '00' .. '29' (feet, floored), then '30-35', '36-46',
               '47+' (the official 2 / 3 ruling is part of the band)
  action_type  shot family from ``sub_type``: jumpshot | layup | dunk | hook | other

Fit, per season, on every attempt with a known result (regular season, play-in and playoffs):
  p_zone            = (FGM_z + K_ZONE * p_league) / (FGA_z + K_ZONE)
  p_zone,band       = (FGM_zb + K_BAND * p_zone) / (FGA_zb + K_BAND)
  p_zone,band,type  = (FGM_zbt + K_TYPE * p_zone,band) / (FGA_zbt + K_TYPE)
then a per-zone calibration so the expected makes of each zone equal its actual makes:
  xfg = clip(p * FGM_z / sum_cells(FGA * p), 0.005, 0.995)
(at both the band and the band x type level). So the league's (and each zone's) expected FG%
equals its actual FG% for the season, by construction; a player's xFG% is the league's make rate
on his shot diet, and "shot making" is his FG% above that.

Rows: one per (zone, dist_band, action_type) cell with attempts, plus the pooled parents used as
fallbacks for a cell never seen in the fit: (zone, dist_band, '*') and (zone, '*', '*').

This deliberately is not the projection model's LightGBM xFG (``proj/features/shots.py``): that
one is walk-forward (fit on seasons < S, so league xFG drifts from the season's actual FG% and
2019-20 has no model), needs features the ``shots`` table does not keep (score margin, play
qualifiers, via its parquet store) and is a heavy fit; the stats pages want an in-season,
league-calibrated reference that a nightly step can rebuild in about a second.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Iterable

K_ZONE, K_BAND, K_TYPE = 50.0, 100.0, 50.0
P_MIN, P_MAX = 0.005, 0.995
FAMILIES = ("jumpshot", "layup", "dunk", "hook")
ANY = "*"


def dist_band(distance: float | None, shot_type: str | None) -> str:
    """'2pt 00' .. '2pt 29', '2pt 30-35', '2pt 36-46', '2pt 47+' (and 3pt likewise)."""
    d = distance if distance is not None and not (isinstance(distance, float)
                                                  and math.isnan(distance)) else 0.0
    d = min(max(float(d), 0.0), 47.0)
    if d < 30:
        b = f"{int(math.floor(d)):02d}"
    elif d < 36:
        b = "30-35"
    elif d < 47:
        b = "36-46"
    else:
        b = "47+"
    return f"{'3pt' if shot_type == '3pt' else '2pt'} {b}"


def family(sub_type: str | None) -> str:
    s = str(sub_type or "").lower().replace(" ", "")
    return s if s in FAMILIES else "other"


def cell(zone: str | None, distance: float | None, shot_type: str | None,
         sub_type: str | None) -> tuple[str, str, str]:
    return (zone or "Unknown", dist_band(distance, shot_type), family(sub_type))


# ------------------------------------------------------------------------------------ fit

def _agg(rows: Iterable[tuple]) -> dict[tuple[str, str, str], list[int]]:
    cells: dict[tuple[str, str, str], list[int]] = {}
    for zone, dist, stype, sub, made in rows:
        c = cells.setdefault(cell(zone, dist, stype, sub), [0, 0])
        c[0] += 1
        c[1] += int(made)
    return cells


def fit_cells(cells: dict[tuple[str, str, str], list[int]]) -> list[dict]:
    """``shot_xfg`` rows (without season) from {(zone, band, type): [fga, fgm]}. Pure."""
    fga_l = sum(v[0] for v in cells.values())
    if not fga_l:
        return []
    p_league = sum(v[1] for v in cells.values()) / fga_l
    zone: dict[str, list[int]] = {}
    band: dict[tuple[str, str], list[int]] = {}
    for (z, b, _t), (a, m) in cells.items():
        zz = zone.setdefault(z, [0, 0])
        zz[0] += a
        zz[1] += m
        bb = band.setdefault((z, b), [0, 0])
        bb[0] += a
        bb[1] += m
    p_z = {z: (m + K_ZONE * p_league) / (a + K_ZONE) for z, (a, m) in zone.items()}
    p_zb = {k: (m + K_BAND * p_z[k[0]]) / (a + K_BAND) for k, (a, m) in band.items()}
    p_zbt = {k: (m + K_TYPE * p_zb[k[:2]]) / (a + K_TYPE) for k, (a, m) in cells.items()}

    def calib(p: dict, counts: dict) -> dict:
        exp: dict[str, float] = {}
        for k, pv in p.items():
            exp[k[0]] = exp.get(k[0], 0.0) + counts[k][0] * pv
        return {k: min(max(pv * (zone[k[0]][1] / exp[k[0]] if exp[k[0]] else 1.0), P_MIN), P_MAX)
                for k, pv in p.items()}

    x_zb, x_zbt = calib(p_zb, band), calib(p_zbt, cells)
    out = [{"zone": z, "dist_band": ANY, "action_type": ANY, "fga": a, "fgm": m,
            "xfg": min(max(m / a if a else p_z[z], P_MIN), P_MAX)} for z, (a, m) in zone.items()]
    out += [{"zone": z, "dist_band": b, "action_type": ANY, "fga": a, "fgm": m,
             "xfg": x_zb[(z, b)]} for (z, b), (a, m) in band.items()]
    out += [{"zone": z, "dist_band": b, "action_type": t, "fga": a, "fgm": m,
             "xfg": x_zbt[(z, b, t)]} for (z, b, t), (a, m) in cells.items()]
    return out


def fit_season(conn: sqlite3.Connection, season: str) -> dict:
    """Refit and replace one season's ``shot_xfg`` rows from ``shots``."""
    cells = _agg(conn.execute(
        "SELECT zone, distance, shot_type, sub_type, made FROM shots "
        "WHERE season=? AND made IS NOT NULL", (season,)))
    rows = fit_cells(cells)
    with conn:
        conn.execute("DELETE FROM shot_xfg WHERE season=?", (season,))
        conn.executemany(
            "INSERT INTO shot_xfg (season, zone, dist_band, action_type, fga, fgm, xfg) "
            "VALUES (?,?,?,?,?,?,?)",
            [(season, r["zone"], r["dist_band"], r["action_type"], r["fga"], r["fgm"], r["xfg"])
             for r in rows])
    return {"season": season, "fga": sum(v[0] for v in cells.values()), "cells": len(cells),
            "rows": len(rows)}


def backfill(conn: sqlite3.Connection, seasons: Iterable[str] | str | None = None) -> list[dict]:
    """Fit every season present in ``shots`` (or the given ones). No network; ~1 s a season."""
    if isinstance(seasons, str):
        seasons = [seasons]
    if seasons is None:
        seasons = [r[0] for r in conn.execute(
            "SELECT DISTINCT season FROM shots WHERE season IS NOT NULL ORDER BY season")]
    return [fit_season(conn, s) for s in seasons]


def has_season(conn: sqlite3.Connection, season: str) -> bool:
    return conn.execute("SELECT 1 FROM shot_xfg WHERE season=? LIMIT 1",
                        (season,)).fetchone() is not None


# --------------------------------------------------------------------------------- lookup

class XFGTable:
    """Lookup of xFG per shot, with the pooled fallbacks: (zone, band, type) -> (zone, band, '*')
    -> (zone, '*', '*') -> the season's league FG%."""

    def __init__(self, conn: sqlite3.Connection, seasons: Iterable[str]):
        seasons = sorted(set(seasons))
        self.t: dict[tuple[str, str, str, str], float] = {}
        self.league: dict[str, float] = {}
        if not seasons:
            return
        q = ",".join("?" * len(seasons))
        for s, z, b, a, fga, fgm, x in conn.execute(
                f"SELECT season, zone, dist_band, action_type, fga, fgm, xfg FROM shot_xfg "
                f"WHERE season IN ({q})", seasons):
            self.t[(s, z, b, a)] = x
            if b == ANY and a == ANY:
                tot = self.league.setdefault(s, [0, 0])
                tot[0] += fga or 0
                tot[1] += fgm or 0
        self.league = {s: (m / a if a else None) for s, (a, m) in self.league.items()}

    def seasons(self) -> set[str]:
        return set(self.league)

    def xfg(self, season: str, zone: str | None, distance: float | None, shot_type: str | None,
            sub_type: str | None) -> float | None:
        z, b, f = cell(zone, distance, shot_type, sub_type)
        t = self.t
        for k in ((season, z, b, f), (season, z, b, ANY), (season, z, ANY, ANY)):
            if k in t:
                return t[k]
        return self.league.get(season)
