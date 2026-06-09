"""Engagement-constrained-skill selection + integer realisation of the recommended scheme.

This is the decision layer over the group-scoring frontier (plan sections 5, 8, 14). The
expensive Monte-Carlo sweep (``scripts/run_experiment.py --group-scoring``) produces the
frontier CSV; this module turns that frontier into the single recommended
``(shape, D/W, gamma -> integer ladder)`` under the **engagement-constrained-skill** objective,
and converts it to concrete small integers.

Three pieces, each a pure function of the frontier table (no simulation):

1. :func:`engagement_anchors` -- the two data-derived engagement-floor anchors of plan section 5
   (the status-quo floor at ``gamma = 0`` is exactly ``0`` by construction; the achievable
   ceiling is the equal-purse ``gamma = 0.5`` ``group_variance_share``, optionally read off the
   engine's *most balanced* field per Doc-4 F6). The frontier is reported RELATIVE to these; no
   magic ``phi`` is asserted.

2. :func:`select_recommendation` -- the registered decision rule (plan section 8.2 step 2 /
   section 14): among the cells meeting a stated engagement level, **minimax over the
   per-objective ranks** of {skill, tie, slot-equity}, breaking ties toward LOWER ``gamma``
   (less skill surrendered) and toward the familiar **3:1** D/W when the skill gap is within the
   paired SE. The skill axis is the SEED-STABLE paired ``delta_skill_vs_anchor`` (NOT the
   absolute level); the engagement axis is ``group_variance_share`` (estimator-free, ``0`` at
   ``gamma = 0``).

3. :func:`integerize_scheme` -- the plan section-14 integerisation. The 48-team bracket has a
   *deterministic* stage occupancy ``[16, 16, 8, 4, 2, 1, 1]`` (GROUP..CHAMPION: 16 group
   exits, 16 lose in the R32, 8 in the R16, 4 QF, 2 SF, 1 runner-up, 1 champion), so the
   expected knockout purse of ANY integer ladder is the exact inner product ``<L_int, occ>`` --
   no Monte-Carlo. We realise the knockout layer as ``round(scale * A_shape)`` and select the
   scale by **exact commensurability**: scale 9 is the *unique* integer scale that lands the
   gamma_match landmark on whole numbers -- 3 group wins x 3 = 9 = the R32 bank ``A_int(R32)`` --
   which self-justifies the scale (the smaller scales 1..8 cannot place ``3W`` on the R32 bank in
   integers, and 9 is the first that can). The mechanism below scans for the smallest scale whose
   realised ``gamma`` lands within a ``gamma_tol`` of the target with a strictly-increasing
   terminal ladder; ``gamma_tol`` is set by the caller ONLY to pin that exact-commensurability
   scale, NOT to encode an SE-derived smallest-scale rule (at the true paired-SE-equivalent gamma
   window ~0.046 the smallest qualifying scale would be 7, not 9). The integer ``top_tie_rate`` is
   re-checkable downstream (the recommendation memo measures it on the headline batch); a finer
   scale (more attainable totals) is the principled, convexity-free tie remedy.

All thresholds here are either structural (the occupancy vector, ``gamma = 0`` floor,
``gamma = 0.5`` equal-purse ceiling), read off the data (the engagement level the owner states),
or -- for the integerisation ``gamma_tol`` -- set purely to pin the self-justifying
exact-commensurability scale; none is a free magic number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .ladders import N_STAGES, Stage, get_ladder

if TYPE_CHECKING:
    import pandas as pd

# Deterministic furthest-stage occupancy of the 48-team / 12-group / 8-best-thirds bracket:
# the expected number of teams whose FURTHEST stage is exactly s, for s = GROUP..CHAMPION.
# 48 teams; 32 advance to the R32 so 16 exit in the group stage; the knockout is a clean binary
# tree from 32, so 16 lose in the R32, 8 in the R16, 4 in the QF, 2 in the SF, 1 is runner-up
# (loses the FINAL) and 1 is champion. This is a STRUCTURAL constant of the format (it sums to 48
# and reproduces every ladder's calibrated knockout purse K = <A_shape, occ> to floating point),
# NOT a tunable -- so the integer-ladder purse is exact arithmetic with no simulation.
STAGE_OCCUPANCY: np.ndarray = np.array([16, 16, 8, 4, 2, 1, 1], dtype=np.float64)

# The familiar FIFA-mirror draw/win ratio (the section-14 tie-break target: prefer 3:1 when the
# skill gap to another D/W is within the paired SE -- it mirrors the on-screen standings).
FAMILIAR_DW: tuple[float, float] = (3.0, 1.0)


def knockout_purse_of_ladder(ladder_points: np.ndarray) -> float:
    """Expected knockout purse ``E[sum_teams L(stage)] = <L, STAGE_OCCUPANCY>`` for a 7-vec ladder.

    Exact (the bracket's stage occupancy is deterministic; see :data:`STAGE_OCCUPANCY`), so this
    is the closed-form knockout purse of *any* integer ladder without re-simulation -- the basis
    of :func:`integerize_scheme`'s realised-gamma check.

    Parameters
    ----------
    ladder_points : numpy.ndarray
        A length-:data:`wcpool.ladders.N_STAGES` (7) stage->points vector (``L[GROUP] = 0``).

    Returns
    -------
    float
        The expected per-team-summed knockout points over the field.

    Raises
    ------
    ValueError
        If ``ladder_points`` is not length 7.
    """
    vec = np.asarray(ladder_points, dtype=np.float64)
    if vec.shape != (N_STAGES,):
        raise ValueError(f"ladder must be length {N_STAGES}; got shape {vec.shape}")
    return float(np.dot(vec, STAGE_OCCUPANCY))


def gamma_of_integer_scheme(
    w_pts: float, d_pts: float, ladder_points: np.ndarray, e_sum_wins: float, e_sum_draws: float
) -> float:
    """Expected group-layer share ``gamma`` of an INTEGER two-layer scheme (closed form).

    An integer scheme awards ``w_pts * wins + d_pts * draws`` (group layer) plus ``L(stage)``
    (knockout layer) with no separate convex ``mix`` knob; the implied share is

        gamma = G / (G + K),   G = w_pts * e_sum_wins + d_pts * e_sum_draws,
                               K = <L, STAGE_OCCUPANCY>.

    ``G`` reuses the calibrated win/draw purse components (the headline CSV's ``group_purse`` for
    a ``(W, D)`` is exactly ``G``); ``K`` is exact via :func:`knockout_purse_of_ladder`.

    Parameters
    ----------
    w_pts, d_pts : float
        Pool points per group win / draw.
    ladder_points : numpy.ndarray
        Length-7 integer knockout ladder.
    e_sum_wins, e_sum_draws : float
        Field-averaged ``E[sum_teams wins]`` / ``E[sum_teams draws]`` (the calibration purse
        components; for the real-Elo field, ~57.80 and ~28.40).

    Returns
    -------
    float
        The group-layer share in ``[0, 1)`` (``nan`` only if both purses are 0).
    """
    g = w_pts * e_sum_wins + d_pts * e_sum_draws
    k = knockout_purse_of_ladder(ladder_points)
    den = g + k
    if den == 0.0:
        return float("nan")
    return float(g / den)


@dataclass(frozen=True)
class EngagementAnchors:
    """The two data-derived engagement-floor anchors of plan section 5.

    ``floor`` is the status-quo ``group_variance_share`` at ``gamma = 0`` (exactly ``0`` by
    construction -- the group layer is absent). ``ceiling`` is the achievable
    ``group_variance_share`` at the equal-purse ``gamma = 0.5`` end (the swept maximum), reported
    PER SHAPE. ``ceiling_balanced_field`` (when supplied) is the same equal-purse share on the
    engine's most-balanced field (Doc-4 F6 benchmark methodology), giving the "how much is
    achievable" envelope. No ``phi`` is asserted; the frontier is read RELATIVE to these.
    """

    shape: str
    floor: float  # group_variance_share at gamma=0 (== 0 by construction)
    ceiling: float  # group_variance_share at gamma=0.5 on the headline (real-Elo) field
    ceiling_balanced_field: float | None = None  # same, on the most-balanced robustness field
    balanced_field_name: str | None = None


def engagement_anchors(
    head_df: pd.DataFrame,
    shape: str,
    *,
    balanced_df: pd.DataFrame | None = None,
    balanced_field_name: str | None = None,
) -> EngagementAnchors:
    """Compute the engagement-floor anchors for ``shape`` from the headline frontier (plan s.5).

    The floor is the ``gamma = 0`` ``group_variance_share`` (exactly ``0``); the ceiling is the
    MAX over D/W of the ``gamma_realised ~ 0.5`` ``group_variance_share`` (the equal-purse end,
    the swept engagement maximum). When ``balanced_df`` (the lowest-concentration robustness
    field) is given, its equal-purse share is attached as the achievable-envelope reference.

    Parameters
    ----------
    head_df : pandas.DataFrame
        The headline frontier (one ``strength_config``; e.g. ``elo_2026`` rows).
    shape : str
        Ladder shape (``"linear"``/``"triangular"``/``"geometric"``).
    balanced_df : pandas.DataFrame, optional
        The most-balanced field's robustness rows (e.g. ``synthetic_x0.5``) for the Doc-4 F6
        achievable-ceiling reference.
    balanced_field_name : str, optional
        That field's name, for the record.

    Returns
    -------
    EngagementAnchors
    """
    s = head_df[head_df["knockout_ladder"] == shape]
    floor_rows = s[s["mix"] == 0.0]
    floor = float(floor_rows["group_variance_share"].iloc[0]) if len(floor_rows) else 0.0
    eqp = s[np.isclose(s["gamma_realised"], 0.5, atol=1e-6)]
    ceiling = float(eqp["group_variance_share"].max()) if len(eqp) else float("nan")
    ceiling_bal = None
    if balanced_df is not None:
        b = balanced_df[balanced_df["knockout_ladder"] == shape]
        beq = b[np.isclose(b["gamma_realised"], 0.5, atol=1e-6)]
        if len(beq):
            ceiling_bal = float(beq["group_variance_share"].max())
    return EngagementAnchors(
        shape=shape,
        floor=floor,
        ceiling=ceiling,
        ceiling_balanced_field=ceiling_bal,
        balanced_field_name=balanced_field_name,
    )


@dataclass(frozen=True)
class Candidate:
    """One ``(shape, D/W, gamma)`` cell as a multi-objective candidate for the selection."""

    shape: str
    w_pts: float
    d_pts: float
    gamma_target: float
    gamma_realised: float
    is_gamma_match: bool
    # objectives
    delta_skill: float  # paired delta_skill_vs_anchor (skill COST is -delta_skill; up is better)
    delta_skill_se: float  # paired SE of delta_skill
    group_variance_share: float  # engagement (higher is "more in it")
    top_tie_rate: float  # lower better
    slot_spread: float  # lower better
    champ_dom: float  # reported (lower = less referendum-on-the-champion)


def _row_to_candidate(row) -> Candidate:
    return Candidate(
        shape=row["knockout_ladder"],
        w_pts=float(row["w_pts"]),
        d_pts=float(row["d_pts"]),
        gamma_target=float(row["gamma_target"]),
        gamma_realised=float(row["gamma_realised"]),
        is_gamma_match=bool(row["is_gamma_match"]),
        delta_skill=float(row["delta_skill_vs_anchor"]),
        delta_skill_se=float(row["delta_skill_paired_se"]),
        group_variance_share=float(row["group_variance_share"]),
        top_tie_rate=float(row["top_tie_rate"]),
        slot_spread=float(row["slot_win_prob_spread"]),
        champ_dom=float(row["p_champion_holder_wins"]),
    )


def _ranks(values: np.ndarray, ascending: bool) -> np.ndarray:
    """Average-rank (1 = best) of ``values``; ``ascending`` => smaller is better (rank 1)."""
    order = values if ascending else -values
    # average ranks for ties (so two equal cells share the mean rank, matching pandas .rank()).
    sorter = np.argsort(order, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_vals = order[sorter]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        ranks[sorter[i : j + 1]] = avg
        i = j + 1
    return ranks


def _skill_indistinguishable(cands: list[Candidate], skill_tol: float) -> list[Candidate]:
    """The cells within ``skill_tol`` of the best paired skill (the decision-relevant tie set).

    ``skill_tol`` is the decision-relevant precision (the shape's absolute Spearman cluster SE; see
    :func:`select_within_shape`); two cells whose paired-skill gap is within it are treated as
    skill-indistinguishable, so the familiar-3:1 + tie-rate criteria govern the choice among them.
    """
    best_skill = max(c.delta_skill for c in cands)
    return [c for c in cands if best_skill - c.delta_skill <= skill_tol]


def _prefer_familiar_then_argmax(pool: list[Candidate]) -> Candidate:
    """Final D/W tie-break: the familiar 3:1 if present, else the pool, then argmax paired skill.

    Shared tail of both selection entry points (:func:`select_within_shape` and
    :func:`select_recommendation`'s ``prefer_gamma_match`` branch): among the
    skill-indistinguishable (and, for the within-shape path, already lowest-gamma) ``pool``, prefer
    the familiar :data:`FAMILIAR_DW` (3:1, the on-screen-standings mirror) when any qualifies, then
    take the highest paired skill (breaking a final exact tie by higher ``w_pts``) for a
    deterministic pick.
    """
    fam = [c for c in pool if (c.w_pts, c.d_pts) == FAMILIAR_DW]
    chosen_pool = fam if fam else pool
    return max(chosen_pool, key=lambda c: (c.delta_skill, c.w_pts))


def select_shape_prior_minimax(head_df: pd.DataFrame) -> tuple[str, dict[str, dict]]:
    """The prior study's balanced-default SHAPE, recovered by minimax over the gamma=0 anchors.

    The shape axis is decided exactly as the prior study decided it ([recommendation_2026-06-08],
    [make_plots.write_recommendation]): score each shape's pure terminal ladder (the ``gamma = 0``
    anchor) on three objectives -- **absolute** skill (``spearman_mean``, higher better; absolute
    is the right CROSS-shape skill comparison, since the paired delta is anchored per shape and so
    is a WITHIN-shape axis only), tie rate (lower), slot-spread (lower) -- and take the shape
    minimising its worst per-objective rank, ties broken by total rank then higher skill. This is
    plan section 5 objective (iii)'s ``phi = 0`` limit: it must REPRODUCE the shipped triangular
    default, which it does (triangular worst-rank 2 beats linear/geometric worst-rank 3). The
    engagement constraint then chooses ``(D/W, gamma)`` WITHIN this shape.

    Returns ``(shape, per_shape_ranks)`` where ``per_shape_ranks[shape]`` carries the anchor metric
    values and their ranks for the audit table.
    """
    anc = head_df[head_df["mix"] == 0.0]
    shapes = list(anc["knockout_ladder"])
    skill = anc["spearman_mean"].to_numpy()
    tie = anc["top_tie_rate"].to_numpy()
    slot = anc["slot_win_prob_spread"].to_numpy()
    r_skill = _ranks(skill, ascending=False)
    r_tie = _ranks(tie, ascending=True)
    r_slot = _ranks(slot, ascending=True)
    worst = np.maximum.reduce([r_skill, r_tie, r_slot])
    total = r_skill + r_tie + r_slot
    # minimax over per-objective ranks; ties -> lower total rank, then higher skill (prior rule).
    order = sorted(
        range(len(shapes)), key=lambda i: (worst[i], total[i], -skill[i])
    )
    chosen = shapes[order[0]]
    per_shape = {
        shapes[i]: {
            "skill": float(skill[i]),
            "tie": float(tie[i]),
            "slot": float(slot[i]),
            "champ_dom": float(anc["p_champion_holder_wins"].iloc[i]),
            "r_skill": float(r_skill[i]),
            "r_tie": float(r_tie[i]),
            "r_slot": float(r_slot[i]),
            "worst_rank": float(worst[i]),
        }
        for i in range(len(shapes))
    }
    return chosen, per_shape


@dataclass(frozen=True)
class Selection:
    """The recommended cell plus the audit trail of the engagement-constrained-skill decision."""

    recommended: Candidate
    shape: str
    engagement_level: float
    eligible: list[Candidate]  # within-shape gamma>0 cells meeting the engagement level
    shape_ranks: dict[str, dict]  # the prior minimax shape-selection audit
    rationale: str


def select_within_shape(
    head_df: pd.DataFrame,
    shape: str,
    engagement_level: float,
    *,
    skill_tol: float | None = None,
) -> Candidate:
    """Engagement-constrained skill-max WITHIN ``shape`` (plan section 5 objective (iii)).

    Among ``shape``'s ``gamma > 0`` cells with ``group_variance_share >= engagement_level``,
    **maximise the seed-stable paired skill** ``delta_skill_vs_anchor`` (objective (iii): max skill
    s.t. the engagement floor), then break ties:

    1. toward LOWER ``gamma_realised`` (less skill surrendered -- section-14 rule);
    2. toward the familiar **3:1** D/W when the D/W choice is skill-indistinguishable;
    3. toward HIGHER paired skill (deterministic final).

    The "skill-indistinguishable" set is those cells within ``skill_tol`` of the best paired skill.
    ``skill_tol`` defaults to the shape's **absolute between-draw cluster SE** of the Spearman skill
    (``spearman_cluster_se``) -- the precision at which skill differences are DECISION-relevant
    (the prior study's significance unit; plan section 8.1). The paired SE is a far tighter
    *within-pair* precision, but a 0.4-cluster-SE paired skill gap between two D/W ratios is below
    the threshold of practical significance, so the familiar-3:1 + tie-rate criteria properly
    govern the D/W choice (the wins-only 1:0 has a marginally higher paired skill but an ~8x worse
    integer-tie rate and no on-screen-standings familiarity). The paired Δ remains the correct
    WITHIN-shape skill axis (anchored to this shape's own ``gamma = 0``); the engagement floor is
    the binding constraint stated by the caller.

    Parameters
    ----------
    head_df : pandas.DataFrame
        Headline frontier (one field).
    shape : str
        The chosen ladder shape.
    engagement_level : float
        Minimum ``group_variance_share`` for eligibility.
    skill_tol : float, optional
        Skill-indistinguishability tolerance on the paired delta. Defaults to the shape's median
        absolute ``spearman_cluster_se`` (decision-relevant precision).
    """
    s = head_df[(head_df["knockout_ladder"] == shape) & (head_df["mix"] > 0.0)]
    elig = s[s["group_variance_share"] >= engagement_level]
    if len(elig) == 0:
        raise ValueError(
            f"no gamma>0 {shape} cell reaches group_variance_share >= {engagement_level}; "
            f"max available within {shape} is {float(s['group_variance_share'].max()):.4f}"
        )
    if skill_tol is None:
        skill_tol = float(s["spearman_cluster_se"].median())
    cands = [_row_to_candidate(r) for _, r in elig.iterrows()]
    # Skill-indistinguishable set: paired skill within the decision-relevant cluster-SE tolerance.
    near = _skill_indistinguishable(cands, skill_tol)
    # Tie-break 1: lowest gamma_realised (less skill surrendered).
    min_gamma = min(c.gamma_realised for c in near)
    low_gamma = [c for c in near if np.isclose(c.gamma_realised, min_gamma, atol=1e-9)]
    # Tie-breaks 2 & 3 (shared tail): familiar 3:1 if present, then highest paired skill.
    return _prefer_familiar_then_argmax(low_gamma)


def select_recommendation(
    head_df: pd.DataFrame,
    engagement_level: float,
    *,
    prefer_gamma_match: bool = True,
    skill_tol: float | None = None,
) -> Selection:
    """The full engagement-constrained-skill recommendation (plan sections 5, 8.2, 14).

    Two stages, faithful to objective (iii) ("max skill s.t. the engagement floor", whose
    ``phi = 0`` limit recovers the prior pure-skill default):

    1. **Shape** = the prior study's minimax-over-anchor-ranks balanced default
       (:func:`select_shape_prior_minimax`) -> triangular. The new feature adds engagement to the
       *shipped* shape; it does not re-open the shape via slot-equity (which a naive equal-weight
       3x3xgamma minimax would, flipping to the structurally slot-flat linear at a large skill
       cost -- not objective (iii)).
    2. **(D/W, gamma)** within that shape = engagement-constrained skill-max
       (:func:`select_within_shape`), breaking ties toward lower gamma then the familiar 3:1.

    ``engagement_level`` is the owner's STATED ``group_variance_share`` floor (read off the
    frontier-relative-to-anchors presentation; no magic ``phi``). The returned :class:`Selection`
    carries the shape-selection audit, the within-shape eligible pool, and a prose rationale.

    Parameters
    ----------
    head_df : pandas.DataFrame
        The headline frontier for one field (``elo_2026`` rows), with the paired-delta + engagement
        columns.
    engagement_level : float
        Minimum ``group_variance_share`` for a cell to be eligible.
    prefer_gamma_match : bool
        When True (the headline), recommend the commensurability landmark ``gamma_match`` (3 group
        wins == reaching the R32) of the chosen shape -- plan section 4's "most decision-relevant
        interior value", which sits in the engagement-efficient ``gamma ~ 0.12-0.25`` band -- at
        the familiar-3:1, skill-indistinguishable D/W, provided it meets ``engagement_level``. When
        False, take the lowest-``gamma`` skill-max eligible cell (the engagement-floor-driven
        minimax, used to show the choice is STABLE across stated floors).
    skill_tol : float, optional
        Skill-indistinguishability tolerance (defaults to the shape's cluster SE; see
        :func:`select_within_shape`).

    Returns
    -------
    Selection

    Raises
    ------
    ValueError
        If no ``gamma > 0`` cell of the chosen shape meets ``engagement_level`` (or, with
        ``prefer_gamma_match``, if no ``gamma_match`` cell does).
    """
    shape, shape_ranks = select_shape_prior_minimax(head_df)
    s = head_df[(head_df["knockout_ladder"] == shape) & (head_df["mix"] > 0.0)]
    elig = s[s["group_variance_share"] >= engagement_level]
    eligible = [_row_to_candidate(r) for _, r in elig.iterrows()]

    if prefer_gamma_match:
        gm = elig[elig["is_gamma_match"]]
        if len(gm) == 0:
            raise ValueError(
                f"no {shape} gamma_match cell reaches group_variance_share >= {engagement_level}"
            )
        gm_cands = [_row_to_candidate(r) for _, r in gm.iterrows()]
        if skill_tol is None:
            skill_tol = float(s["spearman_cluster_se"].median())
        # gamma is already fixed to the landmark here, so the lowest-gamma tie-break is a no-op; the
        # skill-indistinguishable + familiar-3:1 + argmax tail is the shared D/W choice.
        near = _skill_indistinguishable(gm_cands, skill_tol)
        rec = _prefer_familiar_then_argmax(near)
        landmark_note = (
            "gamma_match (the commensurability landmark: 3 group wins == reaching the R32; "
            "plan section 4's most decision-relevant interior value, in the efficient zone)"
        )
    else:
        rec = select_within_shape(head_df, shape, engagement_level, skill_tol=skill_tol)
        landmark_note = "the lowest-gamma skill-indistinguishable cell meeting the floor"

    dw = f"{rec.w_pts:g}:{rec.d_pts:g}"
    rationale = (
        f"Shape = {shape} (prior minimax over the gamma=0 anchor ranks; phi=0 limit of objective "
        f"(iii), reproduces the shipped balanced default). Among the {len(eligible)} {shape} cells "
        f"meeting group_variance_share >= {engagement_level:.4f}, the recommendation is {dw} at "
        f"{landmark_note}: realised gamma={rec.gamma_realised:.4f}, D/W tie-broken toward the "
        f"familiar 3:1 (skill-indistinguishable: the D/W paired-skill spread is well within the "
        f"absolute cluster SE). Paired skill cost vs the gamma=0 {shape} ladder = "
        f"{-rec.delta_skill:+.4f} +/- {rec.delta_skill_se:.4f} "
        f"(~{abs(rec.delta_skill) / max(rec.delta_skill_se, 1e-12):.1f} paired SE; "
        f"{abs(rec.delta_skill) / max(float(s['spearman_cluster_se'].median()), 1e-12):.1f}x the "
        f"absolute cluster SE)."
    )
    return Selection(
        recommended=rec,
        shape=shape,
        engagement_level=engagement_level,
        eligible=eligible,
        shape_ranks=shape_ranks,
        rationale=rationale,
    )


@dataclass(frozen=True)
class IntegerScheme:
    """A concrete small-integer realisation of a recommended ``(shape, D/W, gamma)`` cell.

    ``w_pts``/``d_pts`` are the integer group points; ``ladder`` is the length-7 integer terminal
    knockout ladder ``round(scale * A_shape)``; ``increments`` is its per-advance bank
    (``diff(ladder)`` over R32..CHAMPION, i.e. "bank this for surviving the r-th knockout round").
    ``gamma_realised`` is the integer scheme's exact group-layer share; ``scale`` is the chosen
    integer multiplier.
    """

    shape: str
    w_pts: int
    d_pts: int
    scale: int
    ladder: np.ndarray  # length-7 integer terminal ladder (index 0 == GROUP == 0)
    increments: np.ndarray  # length-6 per-advance bank (R32..CHAMPION)
    gamma_realised: float
    target_gamma: float
    gamma_abs_err: float


def integerize_scheme(
    shape: str,
    w_pts: int,
    d_pts: int,
    target_gamma: float,
    e_sum_wins: float,
    e_sum_draws: float,
    *,
    gamma_tol: float,
    max_scale: int = 200,
) -> IntegerScheme:
    """Realise the knockout layer as ``round(scale * A_shape)`` at the smallest valid integer scale.

    Plan section 14: integerise the recommended cell to concrete small integers. The group layer
    is already integer (``w_pts``/``d_pts``); the knockout layer is realised as
    ``round(scale * A_shape)`` for the SMALLEST integer ``scale`` such that

    * the integer scheme's realised ``gamma`` (closed form via :func:`gamma_of_integer_scheme`,
      using the deterministic :data:`STAGE_OCCUPANCY`) is within ``gamma_tol`` of ``target_gamma``;
    * the terminal ladder is strictly increasing over R32..CHAMPION (increment monotonicity --
      every deeper run banks strictly more), and every per-advance increment is positive.

    For the recommended ``gamma_match`` cell the scale is fixed by **exact commensurability**, not
    by a skill-SE window: scale 9 is the unique integer scale that lands ``3 * w_pts`` exactly on
    the R32 bank ``A_int(R32)`` (3 group wins == reaching the R32) and so realises the landmark in
    whole numbers. ``gamma_tol`` is therefore the caller's *pinning* tolerance -- set tight enough
    that the smallest qualifying scale IS that exact-commensurability scale -- and is **not** a
    paired-SE-derived smallest-scale rule (at the true paired-SE-equivalent gamma window ~0.046 the
    smallest qualifying scale would be 7, not 9). The smallest qualifying scale is preferred for
    communicability (smallest integers); if a downstream integer ``top_tie_rate`` check shows ties
    inflate, the caller picks the next finer (larger) scale -- more attainable totals, fewer
    collisions -- which this function supports by raising the search floor.

    Parameters
    ----------
    shape : str
        Ladder shape whose increment pattern is scaled.
    w_pts, d_pts : int
        Integer group win/draw points.
    target_gamma : float
        The recommended cell's realised ``gamma`` to hit.
    e_sum_wins, e_sum_draws : float
        Calibration purse components (real-Elo: ~57.80, ~28.40).
    gamma_tol : float
        Allowed absolute gamma deviation. Set by the caller to PIN the self-justifying
        exact-commensurability scale (the unique integer scale landing ``3 * w_pts`` on the R32
        bank), NOT a paired-SE-derived smallest-scale window.
    max_scale : int
        Upper bound on the integer scale search (default 200).

    Returns
    -------
    IntegerScheme
        The smallest-scale integer realisation meeting the gamma window + monotonicity.

    Raises
    ------
    ValueError
        If no integer scale in ``[1, max_scale]`` lands within ``gamma_tol`` while preserving
        increment monotonicity.
    """
    base = get_ladder(shape)  # length-7 float ladder, index 0 == 0
    best = None
    for scale in range(1, max_scale + 1):
        lad = np.rint(scale * base).astype(np.int64)
        ko = lad[int(Stage.R32) :]  # R32..CHAMPION
        incr = np.diff(np.concatenate([[0], ko]))
        if not np.all(incr > 0):  # strict monotonicity + positive per-advance bank
            continue
        g = gamma_of_integer_scheme(w_pts, d_pts, lad.astype(np.float64), e_sum_wins, e_sum_draws)
        err = abs(g - target_gamma)
        if err <= gamma_tol:
            best = (scale, lad, ko, incr, g, err)
            break  # smallest qualifying scale
    if best is None:
        raise ValueError(
            f"no integer scale in [1, {max_scale}] for shape {shape!r} lands within "
            f"gamma_tol={gamma_tol} of target_gamma={target_gamma} with monotone increments"
        )
    scale, lad, ko, incr, g, err = best
    return IntegerScheme(
        shape=shape,
        w_pts=int(w_pts),
        d_pts=int(d_pts),
        scale=int(scale),
        ladder=lad,
        increments=incr.astype(np.int64),
        gamma_realised=float(g),
        target_gamma=float(target_gamma),
        gamma_abs_err=float(err),
    )
