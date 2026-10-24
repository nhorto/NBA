"""Absence scale (IMPROVEMENTS G4): COURTSIDE_ABSENCE_SCALE, default OFF.

G1 found the ESPN open->close spread move regresses on the opinion model's injury-news value
with slope 0.45 ("our absence values are 2x the market's"). This switch scales the absence
component of both game models' features by s:

    lineup strength used = roster strength - s * (roster strength - expected-lineup strength)

where roster strength is the full recent roster (the feature row's ps + abs_loss) and the
expected lineup is the injury-report one (`production.lineup_strengths`). s = 1 is production;
s = 0 ignores the injury report. It is applied by wrapping the engine state that
`predict.price_day` hands to `lineup_strengths`, `injury_effects` and the G1 news-since-line
code, so the priced lineups, the per-player "injury moves the odds" effects and the capture-time
lineups are all scaled the same way. Days without a usable report are unchanged (no lineup
override, nothing to scale).

Value: unset / "" / off / false / no -> OFF; on / true / yes -> `SCALE`; a number -> that s.

`SCALE` = 1.0: the evaluation (`research/as_eval.py`, docs/research/absence-scale.md) found the
absence values are NOT too big against outcomes (2025-26, 899 games: margin slope on the absence
component 1.29, SE 0.17; the close prices it at 0.92, SE 0.04). The 0.45 is timing: the open
already prices about a third of the later news. The walk-forward fit of s ends at the grid cap
1.5 and is flat, so there is no fitted shrink to ship; the switch stays for re-tests (set a
number).

ON tags both models' rows' model_version with `TAG` and stores `inputs.absence_scale`. OFF
(default) `price_day` never calls into this module: bit-identical to before.
"""

from __future__ import annotations

import os

import numpy as np

ENV = "COURTSIDE_ABSENCE_SCALE"
DEFAULT = None
TAG = "+as"
SCALE = 1.0
S_RANGE = (0.0, 2.0)

_ON = ("on", "true", "yes")
_OFF = ("", "off", "false", "no")


def parse(value: str | None) -> float | None:
    """The scale for an env value (None = OFF). Unparseable values are OFF."""
    v = (value or "").strip().lower()
    if v in _OFF:
        return None
    if v in _ON:
        return SCALE
    try:
        s = float(v)
    except ValueError:
        return None
    if not np.isfinite(s):
        return None
    return float(min(max(s, S_RANGE[0]), S_RANGE[1]))


S: float | None = parse(os.environ.get(ENV))
ON = S is not None


def set_absence_scale(s: float | None) -> None:
    """Flip the switch at run time (evaluation scripts, tests): None = OFF, else the scale."""
    global S, ON
    S = None if s is None else float(s)
    ON = S is not None


def tagged(version_id: str | None) -> str | None:
    """`version_id` + TAG (None stays None: an unregistered model has no version to tag)."""
    return f"{version_id}{TAG}" if version_id else None


def scale_strength(roster: float, lineup: float, s: float) -> float:
    """roster - s * (roster - lineup); s = 1 returns `lineup` exactly."""
    if s == 1.0 or not np.isfinite(roster):
        return lineup
    return float(roster - s * (roster - lineup))


def roster_strengths(F, rows) -> dict:
    """{franchise: full recent-roster strength} for the home/away teams of feature rows F."""
    rh = F["ps_home"].to_numpy(float) + F["abs_loss_home"].to_numpy(float)
    ra = F["ps_away"].to_numpy(float) + F["abs_loss_away"].to_numpy(float)
    out = {}
    for i, (h, a) in enumerate(zip(rows["home"], rows["away"])):
        out[h], out[a] = float(rh[i]), float(ra[i])
    return out


class ScaledState:
    """Engine-state proxy whose `expected_lineup` strength is scaled toward the roster strength
    by s. Everything else is the wrapped state's."""

    def __init__(self, state, roster: dict, s: float):
        self._state, self._roster, self._s = state, dict(roster), float(s)

    def __getattr__(self, name):
        return getattr(self._state, name)

    def expected_lineup(self, team, game=None, availability=None, playoff=False):
        lu = self._state.expected_lineup(team, game, availability, playoff=playoff)
        r = self._roster.get(team)
        if r is not None:
            lu.attrs["strength"] = scale_strength(r, float(lu.attrs["strength"]), self._s)
        return lu


def wrap(state, F, rows, s: float | None = None):
    """The state `price_day` prices with when ON."""
    return ScaledState(state, roster_strengths(F, rows), S if s is None else s)
