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


# --- Group-W/D/L + progression-bonus variant --------------------------------------------
#
# A scoring variant requested on top of the three terminal-stage ladders above. Unlike
# them, a team's score here has TWO additive components:
#
#   (1) Group-stage match points, scored exactly like the FIFA group table — win 3,
#       draw 1, loss 0 — summed over a team's three group matches (integer 0..9). These
#       depend on the realised group *results*, not just on the furthest stage reached,
#       so the tournament engine supplies them separately
#       (``tournament.simulate_tournament(..., return_group_points=True)``).
#
#   (2) Progression bonuses, awarded once per team as it *reaches* each milestone:
#       reach the Round of 32 / knockouts +5, reach the quarter-final +8, reach the
#       semi-final +12, win the Cup +20. Because each milestone is passed at most once on
#       the way to a team's furthest stage, the total bonus a team earns is the cumulative
#       sum of the milestones at or below that stage — i.e. a furthest-stage lookup vector.
#
# The requested spec defines NO milestone for reaching the Round of 16, nor for *reaching*
# the Final (only for *winning* it). Stages R16 and FINAL therefore add nothing and inherit
# the running total, making the vector flat across (R32, R16) and across (SF, FINAL):
#
#       stage     GROUP  R32  R16  QF   SF  FINAL  CHAMPION
#       bonus       0     5    5   13   25   25     45
#
# A team's total variant points = group_match_points + PROGRESSION_BONUS[furthest_stage].
# The milestone increments are the verbatim task spec, so they carry no free constants; the
# vector is derived from them rather than transcribed, so the two cannot drift apart.

PROGRESSION_BONUS_MILESTONES: dict[str, int] = {
    "R32": 5,  # reach the Round of 32 / knockout stage
    "QF": 8,  # reach the quarter-final
    "SF": 12,  # reach the semi-final
    "CHAMPION": 20,  # win the final
}


def build_cumulative_bonus(milestones: dict[str, int]) -> np.ndarray:
    """Cumulative furthest-stage bonus vector from per-milestone award *increments*.

    ``milestones`` maps a :class:`Stage` *name* to the points awarded for reaching that
    stage. A team keeps every milestone it passes, so the value at stage ``s`` is the sum
    of the increments of all stages ``<= s``. Stages omitted from ``milestones`` add
    nothing and inherit the running total. Returns a length-``N_STAGES`` float vector
    indexed by ``Stage`` (so it composes with :func:`points_for_stages`).
    """
    unknown = set(milestones) - {s.name for s in Stage}
    if unknown:
        raise KeyError(f"unknown stage name(s) in milestones: {sorted(unknown)}")
    vec = np.zeros(N_STAGES, dtype=np.float64)
    running = 0.0
    for s in Stage:  # ascending: GROUP(0) .. CHAMPION(6)
        running += milestones.get(s.name, 0)
        vec[int(s)] = running
    return vec


PROGRESSION_BONUS = build_cumulative_bonus(PROGRESSION_BONUS_MILESTONES)  # [0,5,5,13,25,25,45]


def variant_team_points(
    group_points: np.ndarray, stages: np.ndarray, bonus: np.ndarray = PROGRESSION_BONUS
) -> np.ndarray:
    """Total per-team points under the group-W/D/L + progression-bonus variant.

    ``group_points`` (summed 3/1/0 group-match points) and ``stages`` (furthest
    :class:`Stage`) are same-shaped arrays, e.g. ``(n_sims, n_teams)``. Returns
    ``group_points + bonus[stages]`` as float (so downstream pool scoring is uniform with
    the terminal-stage ladders).
    """
    return group_points.astype(np.float64) + bonus[stages]
