"""Feature store for the learned context layer (docs/PLAYERS.md Phase 2F, F0-F2).

One row per (player, game) with strictly pre-tip features, built from as-of joins on dates
before the game date (season-to-date + recency-weighted + shrunk, `common.ew_asof`). Feature
*groups* are registered in `GROUPS` so a model can be fit with any subset (the ablation):

  P   player descriptors from the EB state (always carried by a non-base model)   player.py
  F1  opponent availability (absent rim protector / rebounder / ball-handler /
      perimeter defender, vacated minutes by position group)                       opp_avail.py
  F2  opponent defense per stat x position group (residual DvP, team box rates,
      pace beyond the market total)                                                opp_defense.py

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

from . import opp_avail, opp_defense, player


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
}


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


from .store import attach, build_frame, day_features, team_game_features  # noqa: E402

__all__ = ["GROUPS", "Group", "model_columns", "monotone_for", "categorical_for", "build_frame",
           "team_game_features", "day_features", "attach", "opp_avail", "opp_defense", "player"]
