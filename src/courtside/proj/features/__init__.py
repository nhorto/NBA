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
  F7  referee crew tendencies (FTA / fouls / pace vs expected, shrunk)       referees.py
  F8  player vs this opponent (EB-shrunk residual vs EB x F2; F8r: light fixed
      shrinkage; F8c: vs the opponent's head coach)                           opp_history.py
  F3x tracking-dashboard opportunity (potential assists, drives, touches)     tracking_ext.py
      F7 / F8 / F8r / F8c / F3x: re-evaluation workstream B (docs/research/reeval-B.md). Registered
      only with the switch ON (`CONTEXT_B_DEFAULT` = False, env COURTSIDE_CONTEXT_B=1 or
      `set_context_b(True)`); OFF the registry is exactly the pre-reeval one. In no live group
      set and not wired into the pipeline.
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

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import (market_props, matchups, opp_avail, opp_defense, opp_history, player, referees,
               schedule, shots, tracking, tracking_ext)


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

# re-evaluation workstream B (H1 F8 / F8c, H3 F7, H4 F3x): behind a switch, OFF by default
CONTEXT_B_DEFAULT = False
B_GROUPS: dict[str, Group] = {
    "F7": Group("F7", "referee crew tendencies", referees.columns, referees.monotone,
                neutral=dict(referees.NEUTRAL)),
    "F8": Group("F8", "player vs this opponent (EB-shrunk residual)", opp_history.columns,
                opp_history.monotone, neutral=dict(opp_history.NEUTRAL)),
    "F8r": Group("F8r", "player vs this opponent (light fixed shrinkage)", opp_history.columns_r,
                 opp_history.monotone_r, neutral=dict(opp_history.NEUTRAL_R)),
    "F8c": Group("F8c", "player vs this opponent's head coach", opp_history.columns_c,
                 opp_history.monotone_c, neutral=dict(opp_history.NEUTRAL_C)),
    "F3x": Group("F3x", "tracking-dashboard opportunity", tracking_ext.columns,
                 tracking_ext.monotone, neutral=dict(tracking_ext.NEUTRAL)),
}
REEVAL_GROUPS: tuple = ()


def set_context_b(on: bool) -> None:
    """Register (ON) or remove (OFF) the workstream-B groups in GROUPS / REEVAL_GROUPS."""
    global REEVAL_GROUPS
    if on:
        GROUPS.update(B_GROUPS)
        REEVAL_GROUPS = tuple(B_GROUPS)
    else:
        for g in B_GROUPS:
            GROUPS.pop(g, None)
        REEVAL_GROUPS = ()


def context_b_on() -> bool:
    return bool(REEVAL_GROUPS)


set_context_b(os.environ.get("COURTSIDE_CONTEXT_B", "1" if CONTEXT_B_DEFAULT else "0")
              not in ("", "0", "false", "False"))


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
           "EXTENSION_GROUPS", "MARKET_GROUPS", "REEVAL_GROUPS", "CONTEXT_B_DEFAULT",
           "set_context_b", "context_b_on", "B_GROUPS",
           "opp_history", "referees", "tracking_ext", "market_props", "opp_avail", "opp_defense", "player", "tracking", "matchups", "schedule", "shots"]
