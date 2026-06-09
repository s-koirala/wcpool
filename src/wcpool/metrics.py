"""Pool-scoring helpers, the five evaluation-metric families, and the engagement suite.

All functions operate on plain arrays so they are independent of how the draft was run:

* ``points`` : (n_sims, n_teams) terminal points each team earned under a ladder.
* ``rosters``: (n_drafters, teams_per_drafter) global team indices each drafter holds.
* ``drafter_of_team``: (n_teams,) owning drafter index, or -1 if undrafted.
* ``running``: (n_sims, n_drafters, N_STAGES) cumulative per-drafter score at each stage
  boundary (``wcpool.scoring.running_scores``), consumed by the engagement metrics.

The metric families correspond 1:1 to the task's list:

1. skill vs luck  -> ``spearman_ev_vs_placing`` + ``skill_variance_share``
2. slot equity    -> ``slot_win_probability`` + ``slot_equity_imbalance``
3. ties at top    -> ``top_tie_rate`` + ``winning_score_summary``
4. champion dom.  -> ``champion_dominance``
5. robustness     -> assembled across favoritism levels by the experiment runner.

The engagement suite (added for the group-stage scoring extension) is a second block of
pure array post-processors over the ``running`` trajectory:

* ``stage_variance_share`` -- law-of-total-variance split of final-score variance into the
  group-phase and knockout-phase increments (estimator-free; the robust headline).
* ``alive_fraction`` -- fraction of (replicate, drafter) mathematically able to still finish
  first entering the knockouts (estimator-free feasibility bound).
* ``conditional_win_prob`` -- a numpy/scipy-only k-NN surrogate for the conditional
  pool-win-probability martingale ``Wcond`` (the Ely-Frankel-Kamenica belief path).
* ``pool_suspense`` / ``pool_surprise`` -- the EFK variance-kernel suspense/surprise built on
  ``Wcond`` (provisional until the nested-simulation gate of the plan's section 6 passes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .ladders import N_STAGES, Stage

if TYPE_CHECKING:
    from .scoring import ScoringScheme


def pool_scores(points: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Per-drafter pool score each replicate: (n_sims, n_drafters)."""
    return points[:, rosters].sum(axis=2)


def win_probability(scores: np.ndarray) -> np.ndarray:
    """Win probability per drafter, splitting a tie at the top equally among those tied."""
    top = scores.max(axis=1, keepdims=True)
    is_top = (scores == top).astype(float)
    credit = is_top / is_top.sum(axis=1, keepdims=True)
    return credit.mean(axis=0)


def top_tie_rate(scores: np.ndarray) -> float:
    """Fraction of replicates in which two or more drafters share the top score."""
    top = scores.max(axis=1, keepdims=True)
    n_tied = (scores == top).sum(axis=1)
    return float(np.mean(n_tied > 1))


def winning_score_summary(scores: np.ndarray) -> dict[str, float]:
    """Distribution summary of the *winning* (maximum) score across replicates."""
    winning = scores.max(axis=1)
    return {
        "mean": float(np.mean(winning)),
        "sd": float(np.std(winning)),
        "p05": float(np.percentile(winning, 5)),
        "median": float(np.percentile(winning, 50)),
        "p95": float(np.percentile(winning, 95)),
    }


def roster_ev(team_ev: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Pre-draft expected pool score per drafter: (n_drafters,)."""
    return team_ev[rosters].sum(axis=1)


def spearman_per_sim(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Per-replicate Spearman rho between roster-EV rank and realised-score rank.

    With only n_drafters items, Spearman rho is Pearson on the rank vectors. Replicates in
    which every drafter scores identically (undefined ordering) yield NaN.
    """
    ev_per_drafter = roster_ev(team_ev, rosters)
    ev_ranks = rankdata(ev_per_drafter)  # (D,)
    score_ranks = rankdata(scores, axis=1)  # (S, D), average method

    b = ev_ranks - ev_ranks.mean()
    a = score_ranks - score_ranks.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman_ev_vs_placing(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> dict:
    """Spearman correlation between pre-draft roster-EV rank and realised placing.

    Computed per replicate, then summarised over replicates. Replicates in which every
    drafter scores identically (undefined ordering) are dropped.
    """
    rho = spearman_per_sim(scores, team_ev, rosters)
    valid = rho[~np.isnan(rho)]
    return {
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "sd": float(np.std(valid)) if valid.size else float("nan"),
        "undefined_frac": float(np.mean(np.isnan(rho))),
    }


def skill_variance_share(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> dict:
    """Share of pool-score variance attributable to roster composition (skill) vs noise.

    Variance decomposition over the (drafter, replicate) score panel::

        sigma2_between = Var over drafters of their expected score  (roster differences)
        sigma2_within  = mean over drafters of Var over replicates  (tournament noise)
        skill_share    = sigma2_between / (sigma2_between + sigma2_within)

    The between-component uses the *independent* roster EV (estimated on a separate batch,
    passed in as ``team_ev``) so it is not inflated by in-sample estimation noise; the
    within-component uses the realised eval-batch ``scores``. This is an
    intraclass-correlation-style index: 1 = placing is fully determined by the draft,
    0 = placing is pure simulation luck.
    """
    sigma2_between = float(np.var(roster_ev(team_ev, rosters)))
    sigma2_within = float(np.mean(np.var(scores, axis=0)))
    total = sigma2_between + sigma2_within
    return {
        "sigma2_between": sigma2_between,
        "sigma2_within": sigma2_within,
        "skill_share": sigma2_between / total if total > 0 else float("nan"),
    }


def slot_win_probability(scores: np.ndarray) -> np.ndarray:
    """Win probability by snake slot (drafter index == pick slot): (n_drafters,)."""
    return win_probability(scores)


def slot_equity_imbalance(scores: np.ndarray) -> dict:
    """How far slot win-probabilities depart from the equitable 1/n_drafters.

    ``max_minus_min`` is the spread. ``imbalance_index`` is a scaled sum of squared
    departures from uniform, ``sum((wp - 1/n)**2 / (1/n))`` — a chi-square-*like*
    discrepancy on probabilities (NOT a chi-square test statistic: it is not multiplied by
    a sample size and has no associated p-value). Both are 0 iff every slot wins equally.
    """
    wp = win_probability(scores)
    n = len(wp)
    uniform = 1.0 / n
    return {
        "win_prob_by_slot": wp.tolist(),
        "max_minus_min": float(wp.max() - wp.min()),
        "imbalance_index": float(np.sum((wp - uniform) ** 2 / uniform)),
    }


def champion_dominance(
    scores: np.ndarray, champion_team: np.ndarray, drafter_of_team: np.ndarray
) -> dict:
    """P(the drafter holding the eventual champion wins the pool).

    Conditioned on the champion being drafted (with N<8 teams/drafter some teams are
    undrafted, so the champion may be held by nobody); the undrafted rate is reported.
    A tie at the top is credited fractionally to the champion's holder.
    """
    holder = drafter_of_team[champion_team]  # (n_sims,), -1 if undrafted
    drafted = holder >= 0
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    n_tied = is_top.sum(axis=1)
    rows = np.arange(scores.shape[0])
    holder_is_top = np.zeros(scores.shape[0], dtype=bool)
    holder_is_top[drafted] = is_top[rows[drafted], holder[drafted]]
    credit = np.where(holder_is_top, 1.0 / n_tied, 0.0)
    p_holder = float(np.mean(credit[drafted])) if drafted.any() else float("nan")
    return {
        "champion_undrafted_rate": float(np.mean(~drafted)),
        "p_champion_holder_wins": p_holder,
    }


# --- Engagement suite -------------------------------------------------------------------
#
# Post-processors over the ``running`` trajectory ``(n_sims, n_drafters, N_STAGES)`` from
# ``wcpool.scoring.running_scores``. Boundary index ``b`` runs ``0 == Stage.GROUP`` (results
# through the group stage are known) ... ``N_STAGES - 1 == Stage.CHAMPION`` (the whole
# tournament is resolved). The estimator-free metrics (``stage_variance_share``,
# ``alive_fraction``) are the robust engagement headline; the ``Wcond``-based suspense/surprise
# are provisional until the plan's section-6 nested-simulation gate passes.


def terminal_win_credit(scores: np.ndarray) -> np.ndarray:
    """Per-replicate fractional first-place credit: ``(n_sims, n_drafters)``.

    The replicate-level summand behind :func:`win_probability`: each top-scoring drafter in a
    replicate receives ``1 / (number tied at the top)`` and everyone else ``0`` (the same tie
    convention as :func:`win_probability`/:func:`champion_dominance`). Each row sums to ``1``,
    and ``terminal_win_credit(scores).mean(axis=0) == win_probability(scores)`` exactly.

    This is the regression target the conditional-win-probability surrogate
    (:func:`conditional_win_prob`) smooths over the running-score state.

    Parameters
    ----------
    scores : numpy.ndarray
        ``(n_sims, n_drafters)`` final pool scores.

    Returns
    -------
    numpy.ndarray
        ``(n_sims, n_drafters)`` ``float64`` fractional win credit; each row sums to ``1``.
    """
    top = scores.max(axis=1, keepdims=True)
    is_top = (scores == top).astype(float)
    return is_top / is_top.sum(axis=1, keepdims=True)


def stage_variance_share(running: np.ndarray) -> dict:
    """Split final-score variance into its group-phase and knockout-phase contributions.

    Writes each drafter's final pool score as the sum of two cumulative-score increments, the
    group-phase increment ``G`` and the knockout-phase increment ``K``::

        G = running[:, :, GROUP]                          # points banked through the group stage
        K = running[:, :, CHAMPION] - running[:, :, GROUP]  # points added across the knockouts
        final = running[:, :, CHAMPION] = G + K

    and applies the law of total variance to the *final-score* variance over replicates, per
    drafter::

        Var(final) = Var(G) + Var(K) + 2 * Cov(G, K)

    so ``var_group = Var(G)``, ``var_knockout = Var(K)`` and ``cov_term = 2 * Cov(G, K)`` sum
    to ``var_total = Var(final)`` exactly. The covariance term is reported explicitly (rather
    than folded into either phase) so the three shares ``group_share``, ``knockout_share`` and
    ``cov_share`` sum to ``1``. The increments are *not* assumed uncorrelated: a steep ladder
    can make a strong group showing predict deep knockout runs, giving ``cov_term != 0``; the
    explicit covariance share keeps the decomposition exact and the trade-off honest.

    Pooling across drafters is a ratio of means -- the summed component variances over the
    drafter mirror :func:`skill_variance_share`'s ``sigma2_between``/``sigma2_within`` pooling
    and the runner's pooled ``skill_share`` -- not a (biased) mean of per-drafter ratios. The
    common denominators cancel, so the pooled shares still sum to ``1`` exactly.

    A *higher* ``group_share`` means more of the final-standings uncertainty is resolved during
    the group phase (more early engagement) and, per the uncertainty-of-outcome literature,
    typically less skill: the metric surfaces the engagement-vs-skill trade-off directly. It is
    threshold-free and estimator-free.

    Parameters
    ----------
    running : numpy.ndarray
        ``(n_sims, n_drafters, N_STAGES)`` cumulative per-drafter score at each stage boundary,
        as returned by :func:`wcpool.scoring.running_scores`.

    Returns
    -------
    dict
        ``var_group``, ``var_knockout``, ``cov_term`` (drafter-summed component variances) and
        the shares ``group_share``, ``knockout_share``, ``cov_share`` (each a pooled ratio of
        means summing to ``1``; ``nan`` when the total variance is ``0``).
    """
    group_inc = running[:, :, int(Stage.GROUP)]  # (n_sims, n_drafters)
    final = running[:, :, int(Stage.CHAMPION)]
    knock_inc = final - group_inc

    # Per-drafter variances/covariance over the replicate axis, then summed over drafters
    # (ratio-of-means pooling). cov_term = 2 * Cov(G, K).
    var_g = np.var(group_inc, axis=0)
    var_k = np.var(knock_inc, axis=0)
    mean_g = group_inc.mean(axis=0)
    mean_k = knock_inc.mean(axis=0)
    cov_gk = ((group_inc - mean_g) * (knock_inc - mean_k)).mean(axis=0)

    var_group = float(var_g.sum())
    var_knockout = float(var_k.sum())
    cov_term = float((2.0 * cov_gk).sum())
    var_total = var_group + var_knockout + cov_term  # == sum_d Var(final_d), by construction

    def _share(part: float) -> float:
        return part / var_total if var_total > 0 else float("nan")

    return {
        "var_group": var_group,
        "var_knockout": var_knockout,
        "cov_term": cov_term,
        "var_total": float(var_total),
        "group_share": _share(var_group),
        "knockout_share": _share(var_knockout),
        "cov_share": _share(cov_term),
    }


def alive_fraction(
    running: np.ndarray,
    stages: np.ndarray,
    scheme: ScoringScheme,
    rosters: np.ndarray,
) -> dict:
    """Fraction of (replicate, drafter) still able to finish first entering the knockouts.

    Drafter ``d`` is *alive* in replicate ``s`` when, with results known through the group
    stage, an optimistic accounting of the points its qualified teams could still earn keeps it
    in reach of the current group-phase leader::

        running[s, d, GROUP] + maxgain[s, d] >= max_{d' != d} running[s, d', GROUP]

    ``maxgain[s, d]`` is a *feasibility upper bound* on ``d``'s still-earnable points entering
    the knockouts. The group layer is already fully realised at the GROUP boundary, so only the
    knockout layer can still grow; for each of ``d``'s teams that *qualified* for the knockouts
    (``stages >= Stage.R32``) the most it can still add is the run from the group floor to the
    title::

        (1 - mix) * (ko_vec[CHAMPION] - ko_vec[GROUP])    # ko_vec[GROUP] == 0 for every ladder

    summed over ``d``'s qualified teams. This bound is deliberately *loose*: it lets every one
    of a drafter's qualified teams become champion simultaneously, which is impossible (a single
    bracket has one champion), so ``maxgain`` over-counts. A loose upper bound is exactly what is
    wanted for a *not-yet-eliminated* test -- it can only ever call a truly dead drafter alive,
    never the reverse, so the reported fraction is a conservative (upper) bound on engagement and
    never falsely eliminates anyone.

    The comparison is against ``max_{d' != d}`` (each drafter excludes itself), so a lone leader
    is always alive. The metric is threshold-free (a pure feasibility inequality) and is
    invariant to the affine point rescalings discussed in the plan.

    .. note::
       **Uninformative over the studied grid -- do NOT read as a discriminating engagement axis.**
       The bound is deliberately loose (it lets every one of a drafter's qualified teams win the
       title at once), and at the studied 8x6 field with ``mix <= 0.5`` essentially every drafter
       is still mathematically alive entering the knockouts: the swept ``alive_fraction`` saturates
       at ``>= 0.999`` with a between-draw cluster SE of ``0``, so it does not separate the scoring
       schemes. This is by design -- the loose bound can only ever call a dead drafter alive, never
       the reverse, so a saturated value is the *correct* (and conservative) reading, not a defect.
       It is retained as a layperson-facing feasibility sanity check; the **estimator-free
       engagement headline is** :func:`stage_variance_share` (the group-phase variance share). The
       bound is intentionally not tightened (a tighter, bracket-aware ceiling would forfeit the
       never-falsely-eliminates guarantee).

    Parameters
    ----------
    running : numpy.ndarray
        ``(n_sims, n_drafters, N_STAGES)`` cumulative per-drafter score
        (:func:`wcpool.scoring.running_scores`); only the GROUP boundary is used.
    stages : numpy.ndarray
        ``(n_sims, n_teams)`` integer furthest-:class:`~wcpool.ladders.Stage` array; a team is
        treated as qualified for the knockouts iff ``stages >= Stage.R32``.
    scheme : ScoringScheme
        The scoring rule. Supplies ``mix`` and the knockout vector for ``maxgain``.
    rosters : numpy.ndarray
        ``(n_drafters, teams_per_drafter)`` global team indices each drafter holds.

    Returns
    -------
    dict
        ``alive_fraction`` (mean over all (replicate, drafter) pairs),
        ``alive_fraction_by_slot`` (per-drafter mean over replicates, as a list), and
        ``n_drafters`` (the number of drafters/snake slots, i.e. ``running.shape[1]`` and the length
        of ``alive_fraction_by_slot`` — carried so a downstream consumer reading the dict alone
        knows the slot count without re-deriving it from the arrays).
    """
    group_now = running[:, :, int(Stage.GROUP)]  # (n_sims, n_drafters)

    ko_vec = scheme.knockout_vector()
    # Per-qualified-team feasibility ceiling on still-earnable knockout points.
    per_team_ceiling = (1.0 - scheme.mix) * (ko_vec[int(Stage.CHAMPION)] - ko_vec[int(Stage.GROUP)])
    qualified = stages >= int(Stage.R32)  # (n_sims, n_teams)
    # maxgain[s, d] = ceiling * (number of d's teams that qualified), summed over the roster.
    maxgain = per_team_ceiling * qualified[:, rosters].sum(axis=2)  # (n_sims, n_drafters)

    # max_{d' != d} running[s, d', GROUP]: the overall row max, except the unique argmax holder
    # competes against the second-highest score in its row.
    n_drafters = group_now.shape[1]
    top2 = np.sort(group_now, axis=1)[:, -2:]  # ascending: [..., second, max]
    row_max = top2[:, -1]
    row_second = top2[:, -2]
    is_unique_top = group_now == row_max[:, None]
    is_unique_top &= is_unique_top.sum(axis=1, keepdims=True) == 1
    competitor_max = np.where(is_unique_top, row_second[:, None], row_max[:, None])

    alive = group_now + maxgain >= competitor_max  # (n_sims, n_drafters)
    by_slot = alive.mean(axis=0)
    return {
        "alive_fraction": float(alive.mean()),
        "alive_fraction_by_slot": [float(x) for x in by_slot],
        "n_drafters": int(n_drafters),
    }


def knn_k_grid(n_sims: int) -> list[int]:
    """Data-derived neighbour-count grid for the :func:`conditional_win_prob` surrogate.

    The candidate ``k`` values are the powers of two from ``1`` up to ``floor(sqrt(n_sims))``.
    The ``sqrt(n)`` ceiling is the standard consistency scaling for k-NN regression -- ``k`` must
    grow with ``n`` for consistency but ``k / n -> 0`` to keep bias vanishing
    (Devroye, Gyorfi & Lugosi 1996, *A Probabilistic Theory of Pattern Recognition*, ch. 6, the
    ``k_n -> infinity``, ``k_n / n -> 0`` condition; ``k ~ sqrt(n)`` satisfies both). The
    geometric (powers-of-two) spacing is a log-uniform search over that admissible range, so no
    single ``k`` is hand-picked -- it is selected by leave-one-out CV in
    :func:`conditional_win_prob`. Always includes ``1``; for very small ``n_sims`` the grid
    collapses to ``[1]``.

    Parameters
    ----------
    n_sims : int
        Number of replicates available to the surrogate.

    Returns
    -------
    list of int
        Ascending powers of two in ``[1, floor(sqrt(n_sims))]``.
    """
    k_max = int(np.floor(np.sqrt(max(n_sims, 1))))
    grid: list[int] = []
    k = 1
    while k <= k_max:
        grid.append(k)
        k *= 2
    return grid or [1]


def _knn_loo_predict(state: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out k-NN regression of ``target`` on ``state`` (deterministic, no RNG).

    For each row ``s`` the prediction is the mean of ``target`` over the ``k`` nearest *other*
    rows of ``state`` (self excluded), found with :class:`scipy.spatial.cKDTree`. The tree query
    is deterministic, so the whole routine is deterministic for fixed inputs -- there is no fold
    shuffle and no random seed (the determinism contract of the plan's section 6).

    Self-exclusion is robust to exact duplicate states. ``cKDTree.query`` does *not* guarantee the
    self-match sits at column 0 when several rows share an identical state (a common case at coarse
    boundaries where running scores take few distinct values), so ``k + 1`` neighbours are queried
    and the self index is removed *by identity* (stable-sorted to the end, then the first ``k``
    kept) rather than by blindly dropping column 0. If a state has more than ``k + 1`` exact
    duplicates so self falls outside the queried window, the first ``k`` candidates are already all
    non-self, which is still a valid leave-one-out neighbourhood.

    Parameters
    ----------
    state : numpy.ndarray
        ``(n_sims, n_drafters)`` predictor vectors (the running-score state at one boundary).
    target : numpy.ndarray
        ``(n_sims, n_drafters)`` regression target (per-replicate win credit).
    k : int
        Number of neighbours, ``1 <= k <= n_sims - 1``.

    Returns
    -------
    numpy.ndarray
        ``(n_sims, n_drafters)`` leave-one-out neighbour-mean predictions.
    """
    n_sims = state.shape[0]
    tree = cKDTree(state)
    # Query k + 1 so the self-match (distance 0) can be removed and still leave k neighbours.
    _, idx = tree.query(state, k=k + 1)
    idx = np.atleast_2d(idx)  # (n_sims, k + 1)
    # Push each row's self-match to the end (stable, so non-self order is preserved), then keep the
    # first k. This excludes self by identity even when query returns it off column 0.
    is_self = idx == np.arange(n_sims)[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")
    neigh = np.take_along_axis(idx, order, axis=1)[:, :k]  # (n_sims, k) nearest OTHER replicates
    return target[neigh].mean(axis=1)


def _within_state_mean(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact within-state mean of ``target`` over rows sharing an identical predictor ``state``.

    Groups the rows by *exact* equality of their predictor vector (``numpy.unique`` along the row
    axis) and assigns every row the mean of ``target`` over **all** rows in its state group (self
    included), so the prediction is constant within each group. This is the conditional expectation
    ``E[target | state]`` at the empirical distribution -- the exact, smoother-free estimate the
    k-NN regression only approximates. It is deterministic (a sort-based group-by, no RNG, no tree
    query) and so preserves the section-6 determinism contract.

    It is used by :func:`conditional_win_prob` for *degenerate* boundaries -- those whose predictor
    takes too few distinct values for the k-NN neighbourhood to localise (the ``n_unique <= k`` test
    there). The fully-constant boundary (one unique state, e.g. the GROUP boundary at ``mix == 0``,
    every replicate's running score the all-zero knockout floor) is the limiting case: every row
    shares the single state, so the prediction is ``target.mean(axis=0)`` for every row -- *exactly*
    the prior. The resulting prior -> boundary belief increment is then identically zero (no
    spurious sampling-variance jump), which a k-NN sub-sample of ``k`` arbitrary other replicates
    would not give.

    Parameters
    ----------
    state : numpy.ndarray
        ``(n_sims, n_drafters)`` predictor vectors (the running-score state at one boundary).
    target : numpy.ndarray
        ``(n_sims, n_drafters)`` regression target (per-replicate win credit).

    Returns
    -------
    numpy.ndarray
        ``(n_sims, n_drafters)`` within-state means; rows sharing an identical ``state`` are equal.
    """
    # Group rows by exact predictor equality. ``inverse`` maps each row to its unique-state id.
    _, inverse = np.unique(state, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_states = int(inverse.max()) + 1 if inverse.size else 0
    if n_states <= 1:
        # Fully-constant predictor: every row shares the one state, so the within-state mean is the
        # global mean. Return it via ``target.mean(axis=0)`` (broadcast) so it is BIT-IDENTICAL to
        # the unconditional prior, which ``_belief_path_with_prior`` forms the same way from the
        # CHAMPION boundary (``terminal_credit.mean(axis=0)``). Bit-identity makes the prior->bound
        # increment EXACTLY 0.0 (not merely ~1e-33), so the ``mix == 0`` group-share is exactly 0.
        return np.broadcast_to(target.mean(axis=0), target.shape)
    sums = np.zeros((n_states, target.shape[1]), dtype=float)
    counts = np.zeros(n_states, dtype=float)
    np.add.at(sums, inverse, target)
    np.add.at(counts, inverse, 1.0)
    means = sums / counts[:, None]  # counts >= 1 for every realised state, so no divide-by-zero
    return means[inverse]


# Sentinel ``k`` recorded for a boundary estimated by the exact within-state mean rather than the
# k-NN smoother (a degenerate ``n_unique <= k`` boundary; see ``conditional_win_prob``). It is 0
# because the within-state mean uses *no* neighbourhood -- it is the exact ``E[credit | state]``,
# the k -> within-group-size limit -- so no positive neighbour count describes it.
WITHIN_STATE_MEAN_K = 0


def conditional_win_prob(
    running: np.ndarray,
    terminal_credit: np.ndarray,
    *,
    selected_k_out: dict[int, int] | None = None,
    selected_k_in: dict[int, int] | None = None,
) -> np.ndarray:
    """k-NN surrogate for the conditional pool-win-probability martingale ``Wcond``.

    Estimates ``Wcond[s, d, b] = P(drafter d ultimately wins the pool | running-score vector at
    boundary b)`` for every boundary ``b``, by regressing the terminal fractional win credit on
    the running-score state ``running[:, :, b]`` across the existing replicates -- no
    re-simulation. The conditional expectation of the (simplex-valued) win credit *is* the
    conditional win-probability vector, and by the tower property it is a martingale in ``b``,
    which is what licenses the Ely-Frankel-Kamenica suspense/surprise machinery
    (:func:`pool_suspense`/:func:`pool_surprise`).

    The estimator is numpy/scipy only (``scipy.spatial.cKDTree`` + a hand-rolled leave-one-out
    loop); scikit-learn is *not* a dependency. ``k`` is chosen **per non-terminal boundary** by
    deterministic leave-one-out CV: for each boundary ``b`` independently, the candidate ``k`` in
    :func:`knn_k_grid` minimising that boundary's leave-one-out squared error is used. A single
    shared ``k`` is suboptimal because the boundaries differ sharply in state granularity -- the
    GROUP boundary has heavy running-score ties (empirically ~1285 unique states of 3000 at the
    8x6 field, since only the realised group W/D tallies vary), so it wants a larger neighbourhood,
    whereas R16 onward are essentially state-unique (the knockout layer separates replicates) and
    tolerate a smaller ``k``. Selecting ``k`` per boundary lets the coarse boundary smooth more and
    the fine boundaries smooth less; one shared ``k`` cannot do both. There is no fold shuffle and
    no RNG, so the output is bit-identical across runs at the same inputs. The chosen ``k`` per
    boundary is recorded in ``selected_k_out`` (when supplied) for the reproducibility log.

    **Degenerate (heavily-tied) boundaries use the EXACT within-state mean, not k-NN.** When a
    boundary's predictor takes too few distinct values for the smoother to localise -- formally when
    ``n_unique <= max(knn_k_grid)``, i.e. even the *largest* admissible neighbourhood would span all
    distinct states, so no candidate ``k`` can resolve one state from another -- that boundary's fit
    is replaced by :func:`_within_state_mean`: each replicate gets the mean terminal credit over
    *all* replicates sharing its exact predictor row (the smoother-free ``E[credit | state]``). The
    limiting case is a *fully constant* predictor (one unique state, e.g. the GROUP boundary at
    ``mix == 0``, every replicate's running score the all-zero knockout floor): there the
    within-state mean is ``terminal_credit.mean(axis=0)`` for every row -- *exactly* the prior -- so
    the prior -> boundary belief increment is **identically zero**. The previous k-NN fit instead
    returned the mean of ``k`` *arbitrary* other replicates -- a noisy ``k``-sub-sample of the prior
    whose sim-to-sim sampling variance manufactured a spurious prior -> boundary jump (hence a
    non-zero ``pool_suspense``/``pool_surprise`` group-share at ``mix == 0``, which must be exactly
    zero). The cKDTree k-NN is kept for genuinely high-cardinality boundaries (R16 onward, and the
    GROUP boundary once ``mix > 0`` makes the per-team W/D points separate replicates), where exact
    duplicates are too sparse to estimate ``E[credit | state]`` directly. Degenerate boundaries are
    recorded in ``selected_k_out`` with the sentinel :data:`WITHIN_STATE_MEAN_K` (``0``).

    The CHAMPION boundary is special-cased: with the whole tournament resolved, the running score
    equals the final score, which *determines* the winner, so ``Wcond[:, :, CHAMPION]`` is set to
    ``terminal_credit`` exactly (the conditional expectation collapses onto the realised credit).
    k-NN smoothing there would wrongly average over distinct outcomes, so it is excluded from both
    the fit and the k-selection. Each ``Wcond[s, :, b]`` lies on the probability simplex (a convex
    average of simplex points), so the rows sum to ``1``.

    **Bias caveat (one-sided).** k-NN smoothing attenuates the variance of the belief increments,
    so it biases :func:`pool_suspense`/:func:`pool_surprise` *downward*. Per-boundary CV still
    targets terminal-credit prediction error, which is *not* increment-variance error; the
    suspense/surprise numbers are therefore provisional until validated against the
    nested-simulation ground truth (the plan's section-6 gate). The martingale check
    (``mean_s Wcond[:, d, b] ~= win_probability``) is a *necessary, not sufficient* sanity test --
    a degenerate zero-suspense smoother also passes it.

    Parameters
    ----------
    running : numpy.ndarray
        ``(n_sims, n_drafters, N_STAGES)`` cumulative per-drafter score at each boundary
        (:func:`wcpool.scoring.running_scores`).
    terminal_credit : numpy.ndarray
        ``(n_sims, n_drafters)`` fractional first-place credit per replicate
        (:func:`terminal_win_credit`); each row sums to ``1``.
    selected_k_out : dict of int to int, optional
        If supplied, it is cleared and populated in place with ``{boundary: chosen_k}`` for each
        non-terminal boundary (the deterministic LOO minimiser, or :data:`WITHIN_STATE_MEAN_K` for a
        degenerate within-state-mean boundary), so the per-boundary neighbour counts can be recorded
        in the reproducibility log. The CHAMPION boundary is exact and is not entered. Returning
        ``k`` this way keeps the array return type and every existing call site unchanged.
    selected_k_in : dict of int to int, optional
        If supplied and **non-empty**, the per-boundary CV scan is skipped and these
        ``{boundary: k}`` are used directly (``k == WITHIN_STATE_MEAN_K`` selects the within-state
        mean). A repeated call over a *structurally identical* batch -- same ``n_sims`` and boundary
        granularity, e.g. successive draws of one sweep cell -- then skips the k-grid scan (the
        dominant cost). The caller's contract is that the cached ``k`` came from a batch with the
        same boundary structure; determinism and bit-identity still hold per cached ``k``. An empty
        dict is treated as absent (CV runs); pass the same dict as ``selected_k_out`` on the first
        call to populate it, then as ``selected_k_in`` thereafter. ``selected_k_in`` and
        ``selected_k_out`` may be the same object.

    Returns
    -------
    numpy.ndarray
        ``Wcond``, ``(n_sims, n_drafters, N_STAGES)`` ``float64``; ``Wcond[:, :, CHAMPION]`` is
        exactly ``terminal_credit`` and every ``Wcond[s, :, b]`` row sums to ``1``.

    Raises
    ------
    ValueError
        If ``running`` is not 3-D with a trailing ``N_STAGES`` axis, or if ``terminal_credit``'s
        shape does not match ``running``'s first two axes.
    """
    if running.ndim != 3 or running.shape[2] != N_STAGES:
        raise ValueError(f"running must be (n_sims, n_drafters, {N_STAGES}); got {running.shape}")
    if terminal_credit.shape != running.shape[:2]:
        raise ValueError(
            f"terminal_credit shape {terminal_credit.shape} != running[:, :, 0] shape "
            f"{running.shape[:2]}"
        )
    n_sims = running.shape[0]
    champion = int(Stage.CHAMPION)
    non_terminal = [b for b in range(N_STAGES) if b != champion]
    grid = [k for k in knn_k_grid(n_sims) if k <= n_sims - 1] or [1]
    # A boundary whose predictor has at most this many distinct states cannot be localised by ANY
    # admissible neighbourhood (the largest candidate k already spans every state) -- it is
    # degenerate and gets the exact within-state mean. This is the data-driven reading of the
    # ``n_unique <= k`` rule, taken at the most generous neighbourhood ``k = max(grid)``.
    degenerate_max_unique = max(grid)
    cached = bool(selected_k_in)  # non-empty -> reuse; empty/None -> run CV

    if selected_k_out is not None and selected_k_out is not selected_k_in:
        selected_k_out.clear()
    wcond = np.empty_like(running, dtype=float)
    for b in non_terminal:
        state_b = running[:, :, b]
        if cached:
            k = selected_k_in[b]
            if k == WITHIN_STATE_MEAN_K:
                wcond[:, :, b] = _within_state_mean(state_b, terminal_credit)
            else:
                wcond[:, :, b] = _knn_loo_predict(state_b, terminal_credit, k)
            best_k = k
        elif np.unique(state_b, axis=0).shape[0] <= degenerate_max_unique:
            # Degenerate boundary: exact within-state mean (fully-constant => exactly the prior =>
            # zero prior->boundary increment). No k-grid scan, no neighbourhood.
            wcond[:, :, b] = _within_state_mean(state_b, terminal_credit)
            best_k = WITHIN_STATE_MEAN_K
        else:
            # High-cardinality boundary: per-boundary k-selection minimising THIS boundary's
            # leave-one-out squared error over the grid (deterministic; no RNG).
            best_k, best_err, best_pred = grid[0], np.inf, None
            for k in grid:
                pred = _knn_loo_predict(state_b, terminal_credit, k)
                err = float(((pred - terminal_credit) ** 2).sum())
                if err < best_err:
                    best_k, best_err, best_pred = k, err, pred
            wcond[:, :, b] = best_pred
        if selected_k_out is not None:
            selected_k_out[b] = best_k
    wcond[:, :, champion] = terminal_credit  # state fully determines the outcome
    return wcond


def _belief_path_with_prior(wcond: np.ndarray) -> np.ndarray:
    """Prepend the unconditional prior to the ``Wcond`` boundary path.

    Returns ``(n_sims, n_drafters, N_STAGES + 1)``: index ``0`` is the pre-tournament prior
    (the unconditional win probability ``mean_s Wcond[:, :, CHAMPION]``, broadcast across
    replicates) and indices ``1 .. N_STAGES`` are the boundary beliefs ``Wcond[:, :, 0 .. 6]``.

    The prior is needed so the *group phase* appears as a belief increment: ``Wcond[:, :, GROUP]``
    already conditions on the realised group results, so the movement the group stage induces is
    the jump from the prior to the GROUP boundary -- the leading increment of this augmented path.
    """
    prior = wcond[:, :, int(Stage.CHAMPION)].mean(axis=0)  # (n_drafters,) unconditional win prob
    n_sims, n_drafters, _ = wcond.shape
    path = np.empty((n_sims, n_drafters, N_STAGES + 1), dtype=float)
    path[:, :, 0] = prior  # broadcast over replicates
    path[:, :, 1:] = wcond
    return path


def _efk_increment_summary(wcond: np.ndarray) -> dict:
    """Shared EFK squared-increment summary for :func:`pool_suspense`/:func:`pool_surprise`.

    Builds the prior-augmented belief path and sums the squared Euclidean belief increments over
    the ``N_STAGES`` transitions (prior -> GROUP, GROUP -> R32, ..., FINAL -> CHAMPION), averaged
    over replicates. The leading increment (prior -> GROUP) is the *group-phase* contribution; the
    remaining transitions are the *knockout-phase* contribution.
    """
    path = _belief_path_with_prior(wcond)
    steps = np.diff(path, axis=2)  # (n_sims, n_drafters, N_STAGES)
    sq = (steps**2).sum(axis=1)  # (n_sims, N_STAGES): squared-norm increment per transition
    per_transition = sq.mean(axis=0)  # mean over replicates
    total = float(per_transition.sum())
    group_phase = float(per_transition[0])  # prior -> GROUP boundary
    knockout_phase = float(per_transition[1:].sum())
    return {
        "total": total,
        "group_phase": group_phase,
        "knockout_phase": knockout_phase,
        "group_share": group_phase / total if total > 0 else float("nan"),
        "per_transition": [float(x) for x in per_transition],
    }


def pool_surprise(wcond: np.ndarray) -> dict:
    """Ely-Frankel-Kamenica *surprise* of the pool win-probability trajectory.

    Surprise is the realised squared belief jump, summed over the boundary path and averaged over
    replicates (the squared-norm / variance-kernel variant, the plan's section 6 and Flepp et al.
    2025 Eq. 1)::

        surprise[s] = sum_b || Wcond[s, :, b] - Wcond[s, :, b - 1] ||^2
        pool_surprise = mean_s surprise[s]

    with the pre-tournament prior prepended (see :func:`_belief_path_with_prior`) so the group
    phase enters as the leading prior -> GROUP increment. The ``group_share`` is that leading
    increment over the total.

    Surprise is reported **separately** from :func:`pool_suspense` and the two are never collapsed
    into one number (Ely-Frankel-Kamenica treat them as distinct preferences). On this
    *realised-path surrogate* they evaluate to the *same statistic* -- the summed squared increments
    over the one observed belief path are identical whether each step is read backward (surprise,
    ``b-1 -> b``) or forward (suspense, ``b -> b+1``) -- so the two totals are exactly equal here,
    not merely equal in expectation. The EFK definitions genuinely differ only when suspense uses
    the *branched* next-period belief variance; that the branched forward version would still agree
    with surprise in expectation is the martingale increment identity, and measuring the genuine
    forward/backward split is exactly the job of the nested-simulation gate (the plan's section 6).
    The absolute value is provisional pending that gate -- k-NN smoothing biases it downward (see
    :func:`conditional_win_prob`).

    Parameters
    ----------
    wcond : numpy.ndarray
        ``(n_sims, n_drafters, N_STAGES)`` conditional win-probability path
        (:func:`conditional_win_prob`).

    Returns
    -------
    dict
        ``total`` surprise, its ``group_phase`` / ``knockout_phase`` split, the ``group_share``,
        and the ``per_transition`` increments.
    """
    return _efk_increment_summary(wcond)


def pool_suspense(wcond: np.ndarray) -> dict:
    """Ely-Frankel-Kamenica *suspense* of the pool win-probability trajectory.

    Suspense is the conditional expected squared change of the *next* belief, summed over the
    boundary path and averaged over replicates (squared-norm / variance-kernel variant, the plan's
    section 6 and Flepp et al. 2025 Eq. 2)::

        suspense[s] = sum_b E[ || Wcond_next - Wcond_now ||^2 | state at b ]
        pool_suspense = mean_s suspense[s]

    On this realised-path surrogate the conditional expectation is estimated by the realised
    squared step ``|| Wcond[s, :, b + 1] - Wcond[s, :, b] ||^2`` (an unbiased single-draw estimate
    of ``E||mu_{b+1} - mu_b||^2`` once averaged over replicates), with the pre-tournament prior
    prepended so the group phase enters as the leading prior -> GROUP increment. The ``group_share``
    is that leading increment over the total -- the single headline engagement number of the plan's
    section 6.

    Suspense is reported **separately** from :func:`pool_surprise`; see that function for why the
    two evaluate to the same statistic on this realised-path surrogate (hence are exactly equal
    here) and why genuine forward/backward separation -- and the absolute calibration of these
    numbers -- requires the nested-simulation gate. The value is provisional and one-sidedly biased
    downward by k-NN smoothing (see :func:`conditional_win_prob`) until that gate passes.

    Parameters
    ----------
    wcond : numpy.ndarray
        ``(n_sims, n_drafters, N_STAGES)`` conditional win-probability path
        (:func:`conditional_win_prob`).

    Returns
    -------
    dict
        ``total`` suspense, its ``group_phase`` / ``knockout_phase`` split, the ``group_share``,
        and the ``per_transition`` increments.
    """
    return _efk_increment_summary(wcond)
