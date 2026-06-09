"""Two-layer pool scoring: progressive-knockout advancement plus group win/draw points.

The prior study scored a drafted team solely by the furthest knockout stage it reached
(a terminal :mod:`~wcpool.ladders` lookup; a group-stage exit scores 0). This module
generalises that to a convex mixture of two layers::

    team_points = (1 - mix) * A(stage)                 # knockout layer (advancement shape)
                +      mix  * (w_pts * wins + d_pts * draws)   # group layer (win/draw only)

``A(.)`` is a length-``N_STAGES`` advancement shape — a named ladder from
:mod:`~wcpool.ladders`, with ``A(GROUP) = 0``. ``wins``/``draws`` are each team's group-match
win/draw counts (each in ``0..3``; there are no points for goals or losses). ``mix`` is the
single cross-layer mixing knob in ``[0, 1]``.

``mix = 0`` recovers the validated terminal-only baseline *exactly* (the group layer
vanishes). To keep that legacy path bit-identical to ``ladders.points_for_stages`` — rather
than merely equal up to floating-point reordering of ``(1 - 0) * ko + 0 * grp`` — both
:func:`team_points` and :func:`running_scores` short-circuit ``mix == 0.0`` to the bare
knockout lookup.

The scorer is a pure array post-processor (mirroring :mod:`~wcpool.metrics`): it consumes
the ``stages`` array and the group win/draw tallies emitted by
:func:`wcpool.tournament.simulate_tournament` and never re-simulates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from .ladders import N_STAGES, Stage, get_ladder

# Group matches each team plays in the single round-robin (``GROUP_SIZE - 1 == 3``); a
# structural constant of the 4-team group, NOT a tunable. Used by :func:`gamma_match_mix`
# (a team winning all three group games == reaching the Round of 32).
GROUP_MATCHES_PER_TEAM = 3


class GroupTallies(TypedDict):
    """Per-team group-match tallies, each ``(n_sims, n_teams)`` ``int32`` and shaped like
    ``stages``: the ``{"wins", "draws"}`` contract emitted by
    ``simulate_tournament(..., return_group_results=True)`` and consumed by the group layer.
    """

    wins: np.ndarray
    draws: np.ndarray


# The three draw/win point candidates, as ``(w_pts, d_pts)``. ``linear_2_1`` sits on the
# constant-sum line ``2 * d_pts == w_pts`` (every match distributes exactly ``w_pts``, so the
# 72-match group total is fixed at ``72 * w_pts`` regardless of the draw count); the other two
# lie off it. Names mirror the on-screen meaning (``fifa_3_1`` is the FIFA 3/1 standings rule).
GROUP_POINT_SCHEMES: dict[str, tuple[float, float]] = {
    "fifa_3_1": (3.0, 1.0),
    "linear_2_1": (2.0, 1.0),
    "wins_only_1_0": (1.0, 0.0),
}


@dataclass(frozen=True)
class ScoringScheme:
    """A fully specified two-layer scoring rule.

    Parameters
    ----------
    knockout_ladder : str
        Name of the advancement shape in :data:`wcpool.ladders.LADDERS`
        (``"linear"``/``"triangular"``/``"geometric"``).
    w_pts : float
        Pool points awarded per group-stage win.
    d_pts : float
        Pool points awarded per group-stage draw.
    mix : float, optional
        Convex weight on the group layer in ``[0, 1]``. ``mix = 0`` (default) is the prior
        terminal-only study; ``mix = 1`` would delete the knockout layer.

    Raises
    ------
    KeyError
        If ``knockout_ladder`` is not a known ladder name (deferred to
        :func:`wcpool.ladders.get_ladder` at :meth:`knockout_vector`).
    ValueError
        If ``mix`` is outside ``[0, 1]``, or either group rate ``w_pts``/``d_pts`` is negative.
    """

    knockout_ladder: str
    w_pts: float
    d_pts: float
    mix: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError(f"mix must lie in [0, 1]; got {self.mix}")
        # Group points are awards, never penalties: a negative w_pts/d_pts would let a losing
        # team out-score a winning one in the group layer. Reject at construction (e.g. a
        # fat-fingered ``--w-pts -3``) rather than silently building a garbage board. All
        # legitimate schemes are non-negative: GROUP_POINT_SCHEMES (3/1, 2/1, 1/0) and the
        # recommended (3, 1). (mix == 0 leaves these inert, but the rule still holds.)
        if self.w_pts < 0.0:
            raise ValueError(f"w_pts must be >= 0; got {self.w_pts}")
        if self.d_pts < 0.0:
            raise ValueError(f"d_pts must be >= 0; got {self.d_pts}")

    def knockout_vector(self) -> np.ndarray:
        """Return the length-``N_STAGES`` stage->points advancement vector.

        This is a FRESH copy of the shared ladder constant (``get_ladder`` copies
        :data:`wcpool.ladders.LADDERS`), so it is safe to index/mutate and is *not*
        identity-equal to ``wcpool.ladders.LADDERS[knockout_ladder]`` — do not "optimise" this
        to alias the shared constant.

        Raises
        ------
        KeyError
            If :attr:`knockout_ladder` is not a known ladder name.
        """
        return get_ladder(self.knockout_ladder)


def _validate_group(group: GroupTallies, stages: np.ndarray) -> None:
    """Guard the group layer's contract before it is used (``mix > 0`` only).

    ``group`` must carry both ``"wins"`` and ``"draws"``, each shaped exactly like ``stages``
    (``(n_sims, n_teams)``). Mirrors ``tournament.random_pot_draw``'s shape guard; the
    ``mix == 0`` short-circuit never reaches here, so it tolerates an absent/ill-shaped group.
    """
    for key in ("wins", "draws"):
        if key not in group:
            raise ValueError(f"group must contain {key!r}; got keys {sorted(group)}")
        if group[key].shape != stages.shape:
            raise ValueError(
                f"group[{key!r}] shape {group[key].shape} != stages shape {stages.shape}"
            )


def team_points(stages: np.ndarray, group: GroupTallies, scheme: ScoringScheme) -> np.ndarray:
    """Per-team two-layer points: ``(n_sims, n_teams)``.

    Parameters
    ----------
    stages : numpy.ndarray
        Integer furthest-:class:`~wcpool.ladders.Stage` array, any shape; used to index the
        knockout vector.
    group : GroupTallies
        ``{"wins": ..., "draws": ...}`` group-match tallies, each of shape ``(n_sims, n_teams)``
        matching ``stages`` (as emitted by ``simulate_tournament(..., return_group_results=True)``).
        Validated only when ``scheme.mix > 0``; ignored on the ``mix == 0`` short-circuit.
    scheme : ScoringScheme
        The scoring rule.

    Returns
    -------
    numpy.ndarray
        Points per team, same shape as ``stages``, ``float64``.

    Raises
    ------
    ValueError
        When ``scheme.mix > 0`` and ``group`` lacks ``"wins"``/``"draws"`` or either tally's
        shape differs from ``stages``.

    Notes
    -----
    When ``scheme.mix == 0.0`` the bare knockout lookup ``ko_vec[stages]`` is returned
    unaltered, so the result is element-wise identical to
    ``ladders.points_for_stages(stages, get_ladder(scheme.knockout_ladder))``.

    The group layer promotes ``int32`` tallies to ``float64``: ``int32`` ``wins``/``draws``
    times the ``float`` ``w_pts``/``d_pts`` yields ``float64``, matching the ladders' ``float64``
    knockout vectors, so the mixed result is uniformly ``float64``.
    """
    ko = scheme.knockout_vector()[stages]
    if scheme.mix == 0.0:
        return ko
    _validate_group(group, stages)
    grp = scheme.w_pts * group["wins"] + scheme.d_pts * group["draws"]
    return (1.0 - scheme.mix) * ko + scheme.mix * grp


def running_scores(
    stages: np.ndarray,
    group: GroupTallies,
    scheme: ScoringScheme,
    rosters: np.ndarray,
) -> np.ndarray:
    """Per-drafter cumulative pool score at each stage boundary: ``(n_sims, n_drafters, N_STAGES)``.

    At boundary ``b`` (``0 == GROUP`` ... ``N_STAGES - 1 == CHAMPION``) results through stage
    ``b`` are known. The group layer is fully realised at ``b = 0`` and constant thereafter;
    the knockout layer at boundary ``b`` credits a team for being at stage
    ``min(furthest_stage, b)`` — i.e. ``ko_vec[min(stages, b)]``. Each drafter's score sums
    the two-layer team points over their roster.

    Parameters
    ----------
    stages : numpy.ndarray
        ``(n_sims, n_teams)`` integer furthest-:class:`~wcpool.ladders.Stage` array.
    group : GroupTallies
        ``{"wins": ..., "draws": ...}`` each of shape ``(n_sims, n_teams)`` matching ``stages``
        (the same contract as :func:`team_points`). Validated only when ``scheme.mix > 0``;
        ignored on the ``mix == 0`` short-circuit.
    scheme : ScoringScheme
        The scoring rule.
    rosters : numpy.ndarray
        ``(n_drafters, teams_per_drafter)`` global team indices each drafter holds.

    Returns
    -------
    numpy.ndarray
        Cumulative per-drafter score at each boundary, ``(n_sims, n_drafters, N_STAGES)``,
        ``float64``.

    Raises
    ------
    ValueError
        When ``scheme.mix > 0`` and ``group`` lacks ``"wins"``/``"draws"`` or either tally's
        shape differs from ``stages`` (delegated to the same guard as :func:`team_points`).

    Notes
    -----
    The terminal boundary (index ``N_STAGES - 1 == CHAMPION``) reconciles with the one-shot
    scorer::

        running_scores(...)[..., N_STAGES - 1] == pool_scores(team_points(...), rosters)

    because ``min(stages, CHAMPION) == stages``. The reconciliation is *exact*
    (``np.array_equal``) only at ``mix in {0, 1}`` — where one layer vanishes and no convex
    blend occurs; for ``0 < mix < 1`` the running and one-shot paths blend the two layers in a
    different floating-point op-order, so they agree only to floating-point tolerance
    (``np.allclose``). As with :func:`team_points`, the ``mix == 0`` short-circuit returns the
    bare knockout contribution so the legacy path is bit-identical.
    """
    ko_vec = scheme.knockout_vector()
    boundaries = np.arange(N_STAGES)
    # (n_sims, n_teams, N_STAGES): knockout points for being at stage min(furthest, b).
    ko_at_boundary = ko_vec[np.minimum(stages[..., None], boundaries)]
    # Gather each roster's teams (axis 1) and sum over teams -> (n_sims, n_drafters, N_STAGES).
    ko_running = ko_at_boundary[:, rosters, :].sum(axis=2)
    if scheme.mix == 0.0:
        return ko_running
    _validate_group(group, stages)
    grp_team = scheme.w_pts * group["wins"] + scheme.d_pts * group["draws"]
    grp_running = grp_team[:, rosters].sum(axis=2)  # (n_sims, n_drafters), constant over b
    return (1.0 - scheme.mix) * ko_running + scheme.mix * grp_running[..., None]


# --- gamma <-> mix solver (plan section 4) ----------------------------------------------
#
# ``mix`` is the convex implementation knob; the *reported/targeted* quantity is the expected
# group-layer SHARE of total points on the real-Elo field,
#
#     gamma(mix) = mix * G / ( mix * G + (1 - mix) * K ),
#
# with ``G = E[sum_teams group_layer] = w_pts * E[sum wins] + d_pts * E[sum draws]`` and
# ``K = E[sum_teams A(stage)]`` (the field-averaged group/knockout purses). gamma is dimensionless,
# scale-free, monotone in mix, and 0 at mix=0. ``G`` and ``K`` are estimated once from a calibration
# batch (``simulate.calibrate_group_knockout``); these helpers are then pure closed-form arithmetic
# with no simulation and no tunable constants.


def gamma_of_mix(mix: float, group_purse: float, knockout_purse: float) -> float:
    """Expected group-layer share ``gamma`` of total points at convex weight ``mix``.

    ``gamma(mix) = mix * G / (mix * G + (1 - mix) * K)`` for group purse ``G = group_purse`` and
    knockout purse ``K = knockout_purse`` (both ``>= 0``; the field-averaged expected per-team
    group / knockout points, summed over teams). Monotone non-decreasing in ``mix`` on ``[0, 1]``
    for ``G, K > 0``, with ``gamma(0) = 0``, ``gamma(1) = 1``. Returns ``0.0`` at ``mix == 0`` for
    any purses (group layer absent); the limit ``mix -> 1`` collapses to ``1`` when ``G > 0``.

    Parameters
    ----------
    mix : float
        Convex weight on the group layer, ``mix in [0, 1]``.
    group_purse : float
        ``G`` -- expected sum over teams of the group-layer points (``w_pts * E[sum wins] +
        d_pts * E[sum draws]``), ``>= 0``.
    knockout_purse : float
        ``K`` -- expected sum over teams of the knockout-ladder points ``A(stage)``, ``>= 0``.

    Returns
    -------
    float
        The group-layer share in ``[0, 1]``; ``nan`` only if both purses are 0 at ``mix > 0``.
    """
    if mix == 0.0:
        # Early return for symmetry with ``solve_mix_for_gamma(0.0, ...) == 0.0`` and to make the
        # "0.0 at mix == 0 for any purses" contract literally true: at ``mix == 0`` the group layer
        # is absent, so the share is exactly 0 regardless of the (possibly both-zero) purses.
        # Without it the all-zero-purse case (K == G == 0) would hit den == 0 and return nan.
        return 0.0
    num = mix * group_purse
    den = num + (1.0 - mix) * knockout_purse
    if den == 0.0:
        return float("nan")
    return float(num / den)


def solve_mix_for_gamma(target_gamma: float, group_purse: float, knockout_purse: float) -> float:
    """Closed-form convex weight ``mix`` achieving a target group-layer share ``target_gamma``.

    The exact inverse of :func:`gamma_of_mix` in ``mix``::

        mix = gamma * K / ( (1 - gamma) * G + gamma * K )

    (solve ``gamma = mix G / (mix G + (1 - mix) K)`` for ``mix``). ``target_gamma == 0`` returns
    ``0.0`` exactly (the mix=0 baseline, group layer absent); a target approaching ``1`` drives
    ``mix -> 1``. Round-trips to floating point against :func:`gamma_of_mix`.

    Parameters
    ----------
    target_gamma : float
        Desired group-layer share, ``target_gamma in [0, 1)`` (``gamma == 1`` requires ``mix == 1``,
        which deletes the knockout layer and is excluded by the swept grid; passing exactly ``1.0``
        with ``K > 0`` still returns ``1.0``).
    group_purse, knockout_purse : float
        ``G`` and ``K`` as in :func:`gamma_of_mix` (``>= 0``).

    Returns
    -------
    float
        The convex weight ``mix in [0, 1]`` solving ``gamma_of_mix(mix, G, K) == target_gamma``.

    Raises
    ------
    ValueError
        If ``target_gamma`` is outside ``[0, 1]``, or both purses are 0 at ``target_gamma > 0``
        (no finite mix can hit a positive share when there is no group OR knockout purse).
    """
    if not 0.0 <= target_gamma <= 1.0:
        raise ValueError(f"target_gamma must lie in [0, 1]; got {target_gamma}")
    if target_gamma == 0.0:
        return 0.0
    den = (1.0 - target_gamma) * group_purse + target_gamma * knockout_purse
    if den == 0.0:
        raise ValueError(
            "cannot solve mix for a positive gamma when both group and knockout purses are 0"
        )
    return float(target_gamma * knockout_purse / den)


def gamma_match_mix(w_pts: float) -> float:
    """Convex weight ``mix*`` at the per-cell *commensurability* landmark (plan section 4).

    ``mix*`` is the weight at which a team winning all three group games earns the same as a team
    reaching the Round of 32, i.e. ``mix* * (3 * w_pts) == (1 - mix*) * A(R32)`` with ``A(R32) = 1``
    for all three shapes (:mod:`wcpool.ladders`). Solving::

        mix* = A(R32) / (3 * w_pts + A(R32)) = 1 / (3 * w_pts + 1)

    (the ``3`` is :data:`GROUP_MATCHES_PER_TEAM`, the round-robin match count, not a tunable). This
    is a property of the *scheme algebra* and depends only on ``w_pts``; the corresponding share
    ``gamma_match = gamma_of_mix(mix*, G, K)`` is field-dependent through the calibrated purses.

    Parameters
    ----------
    w_pts : float
        Pool points per group win (``> 0``); the win-rate of the ``(w_pts, d_pts)`` candidate.

    Returns
    -------
    float
        ``mix* in (0, 1)`` -- the commensurability weight.

    Raises
    ------
    ValueError
        If ``w_pts <= 0`` (no win purse, so the commensurability point is undefined).
    """
    if w_pts <= 0.0:
        raise ValueError(f"w_pts must be > 0 for the gamma_match landmark; got {w_pts}")
    a_r32 = float(get_ladder("linear")[int(Stage.R32)])  # == 1 for every shape (A(R32)=1)
    return float(a_r32 / (GROUP_MATCHES_PER_TEAM * w_pts + a_r32))
