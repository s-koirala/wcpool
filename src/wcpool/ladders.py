"""Scoring ladders: terminal points awarded by the furthest knockout round a team reaches.

A team's tournament outcome is summarised by a single ordinal *stage index* — the
furthest stage it reached — and a ladder maps that index to points.

Stage indices (``Stage``)::

    0 GROUP     eliminated in the group stage (did not reach the Round of 32)
    1 R32       reached the Round of 32, lost there
    2 R16       reached the Round of 16, lost there
    3 QF        reached the quarter-final, lost there
    4 SF        reached the semi-final, lost there
    5 FINAL     reached the final, lost it (runner-up)
    6 CHAMPION  won the final

All ladders award 0 for a group-stage exit (stage 0); the task specifies points only
for stages R32..Champion. Each ladder is therefore a length-7 vector indexed by stage.

The three ladders are taken verbatim from the task specification:

* ``linear``     R32..Champ = 1, 2, 3, 4, 5, 6        (arithmetic, step 1)
* ``triangular`` R32..Champ = 1, 3, 6, 10, 15, 21     (triangular numbers T_k = k(k+1)/2)
* ``geometric``  R32..Champ = 1, 2, 4, 8, 16, 32      (powers of two, 2^(k-1))

These are exact, prescribed designs, not tuned parameters, so they carry no free
constants to calibrate.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Stage(IntEnum):
    """Furthest stage reached by a team. Ordinal; higher is further."""

    GROUP = 0
    R32 = 1
    R16 = 2
    QF = 3
    SF = 4
    FINAL = 5
    CHAMPION = 6


N_STAGES = len(Stage)  # 7
N_SCORING_STAGES = 6  # R32..Champion


def _ladder(scoring_stage_points: list[int]) -> np.ndarray:
    """Build a length-7 stage->points vector from the 6 scoring-stage values.

    ``scoring_stage_points`` is [R32, R16, QF, SF, FINAL, CHAMPION]; stage GROUP gets 0.
    """
    if len(scoring_stage_points) != N_SCORING_STAGES:
        raise ValueError(
            f"expected {N_SCORING_STAGES} scoring-stage values, got {len(scoring_stage_points)}"
        )
    vec = np.zeros(N_STAGES, dtype=np.float64)
    vec[Stage.R32 :] = scoring_stage_points
    return vec


# --- Prescribed ladders (exact, from the task spec) -------------------------------------

LINEAR = _ladder([1, 2, 3, 4, 5, 6])
TRIANGULAR = _ladder([1, 3, 6, 10, 15, 21])  # cumulative increments 1..6
GEOMETRIC = _ladder([1, 2, 4, 8, 16, 32])  # 2^0 .. 2^5

LADDERS: dict[str, np.ndarray] = {
    "linear": LINEAR,
    "triangular": TRIANGULAR,
    "geometric": GEOMETRIC,
}


def get_ladder(name: str) -> np.ndarray:
    """Return the length-7 stage->points vector for a named ladder."""
    try:
        return LADDERS[name].copy()
    except KeyError as exc:
        raise KeyError(f"unknown ladder {name!r}; choices: {sorted(LADDERS)}") from exc


def points_for_stages(stages: np.ndarray, ladder: np.ndarray) -> np.ndarray:
    """Vectorised lookup: map an int array of stage indices to points under ``ladder``.

    ``stages`` may be any shape; output has the same shape.
    """
    return ladder[stages]
