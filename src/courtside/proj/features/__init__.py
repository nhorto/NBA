"""Feature store for the learned context layer (docs/PLAYERS.md Phase 2F, F0-F2).

One row per (player, game) with strictly pre-tip features, built from as-of joins on dates
before the game date (season-to-date + recency-weighted + shrunk, `common.ew_asof`). Feature
*groups* are registered in `GROUPS` so a model can be fit with any subset (the ablation):

  P   player descriptors from the EB state (always carried by a non-base model)   player.py
  F1  opponent availability (absent rim protector / rebounder / ball-handler /
      perimeter defender, vacated minutes by position group)                       opp_avail.py
  F2  opponent defense per stat x position group (residual DvP, team box rates,
      pace beyond the market total)                                                opp_defense.py
  F3  tracking role signals (rebound chances, passes, touches, deflections, ...:
      shrunk per-minute rates, conversions, tracking-implied rate vs the EB rate)  tracking.py
  F5  matchups (likely defenders x shrunk defender quality; whom he guards)        matchups.py
  F6  schedule context (home, rest, density, travel, time zones, altitude, stage) schedule.py
  F4  shot quality / shooting luck (xFG from the cdn PBP: quality, diet, skill,
      luck inside the EB rates, opponent quality / luck allowed)                  shots.py
  F9  market player-prop lines as of a lock cutoff (de-vigged implied mean vs the
      EB rate, has-line flags, moves). Registered, OFF: in no live group set;
      props exist only from 2026-10-19 (`market_props.readiness`)                 market_props.py

F3 / F4 / F5 / F6 are not in the cached base frame: `extend_frame(conn, frame, groups)` adds them
(cached per season and group). F9 too, never cached (the prop log grows every day).

Public API
  GROUPS / Group                       registry (columns(stat), monotone(stat), categorical)
  model_columns(stat, groups)          feature list of a per-stat model
  monotone_for(stat, groups)           {column: +1/-1} monotone constraints
  build_frame(conn, season, sp, ...)   training/eval frame for test season S (store.py)
  team_game_features(conn, wf, ...)    F1 + F2 team-game table (store.py)
  day_features(conn, wf, t, day, games) F2 rows for one day's games (the pipeline hook)
  attach(rows, team_feats)             join team-game features to player rows + row-level cols
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import (market_props, matchups, opp_avail, opp_defense, player, schedule, shots,
               tracking)


@dataclass(frozen=True)
class Group:
    name: str
    title: str
    columns: Callable[[str], list]
    monotone: Callable[[str], dict]
    categorical: tuple = field(default_factory=tuple)
    neutral: dict = field(default_factory=dict)       # column -> league-average value


GROUPS: dict[str, Group] = {
    "P": Group("P", "player descriptors (EB state)", player.columns, player.monotone,
               player.CATEGORICAL),
    "F1": Group("F1", "opponent availability", opp_avail.columns, opp_avail.monotone,
                neutral={c: 0.0 for c in opp_avail.COLS if c != "f1_depth"}),
    "F2": Group("F2", "opponent defense per stat x position group", opp_defense.columns,
                opp_defense.monotone,
                neutral={c: 0.0 for c in opp_defense.TEAM_COLS + opp_defense.ROW_COLS}),
    "F3": Group("F3", "tracking role signals", tracking.columns, tracking.monotone,
                neutral=dict(tracking.NEUTRAL)),
    "F5": Group("F5", "matchups: likely defenders x defender quality", matchups.columns,
                matchups.monotone, neutral=dict(matchups.NEUTRAL)),
    "F6": Group("F6", "schedule context", schedule.columns, schedule.monotone,
                neutral=dict(schedule.NEUTRAL)),
    "F4": Group("F4", "shot quality and shooting luck (xFG)", shots.columns, shots.monotone,
                neutral=dict(shots.NEUTRAL)),
    "F9": Group("F9", "market player-prop lines (de-vigged, as of the lock cutoff)",
                market_props.columns, market_props.monotone,
                neutral=dict(market_props.NEUTRAL)),
}
EXTENSION_GROUPS = ("F3", "F5", "F6", "F4")
# market groups: added by extend_frame like the extensions, but never by LiveExtras (the live
# path reads them with `market_props.live_lines`); in no live group set (off by default)
MARKET_GROUPS = ("F9",)


def model_columns(stat: str, groups: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for g in groups:
        for c in GROUPS[g].columns(stat):
            if c not in cols:
                cols.append(c)
    return cols


def monotone_for(stat: str, groups: Sequence[str]) -> dict:
    out: dict = {}
    for g in groups:
        out.update(GROUPS[g].monotone(stat))
    return out


def categorical_for(groups: Sequence[str]) -> list[str]:
    return [c for g in groups for c in GROUPS[g].categorical]


from .store import (LiveExtras, attach, build_frame, day_features,  # noqa: E402
                    extend_frame, team_game_features)

__all__ = ["GROUPS", "Group", "model_columns", "monotone_for", "categorical_for", "build_frame",
           "team_game_features", "day_features", "attach", "extend_frame", "LiveExtras",
           "EXTENSION_GROUPS", "MARKET_GROUPS",
           "market_props", "opp_avail", "opp_defense", "player", "tracking", "matchups", "schedule", "shots"]
