"""Objective and roster-scoring primitives.

The pool pays 1st and refunds 2nd (break-even); everyone else loses their buy-in ``b``.
With ``n`` drafters and a winner-takes-rest split (1st gets ``(n-1)*b``, 2nd gets ``b``
back), expected utility in units of ``b`` is

    E[U]/b = (n-1)*P(1st) + 1*P(2nd) - 1,

so the quantity to maximise for our drafter is

    W = (n-1)*P(1st) + P(2nd)                      # weight w = n-1

This is *near* winner-take-all (the (n-1) weight on 1st) with a genuine loss-avoidance
floor on 2nd. The full derivation is in ``docs/objective_derivation.md``.

Placement probabilities use **fractional tie credit**: within a tie group of size ``g``
whose best competition rank is ``r``, the members jointly occupy finishing positions
``r .. r+g-1`` and split each equally. For the 1st-place position this reduces *exactly* to
:func:`wcpool.metrics.win_probability` (verified in ``tests/test_objective.py``); it extends
the same convention to 2nd place.

Tier-cliff detection (:func:`detect_cliffs`) flags a gap that is a **local spike** in the
descending-``W`` sequence: a drop larger than *both* its neighbouring gaps by more than
Monte-Carlo noise (one-sided Bonferroni across the interior gaps at ``CLIFF_ALPHA``). This is
a discrete second-difference / changepoint test, so a smoothly declining board -- including a
*convex* one whose gaps grow monotonically toward the top -- produces no spurious cliffs.
(Earlier attempts failed here: a Tukey fence flagged ~50% of smooth declines, and a
linear-null bootstrap still flagged convex ones; only a genuine kink is a spike above both
neighbours.) The decision logic does not depend on cliffs -- they are a display aid; the
robustness sweep and reach/wait look-ahead use fixed caps.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

# Number of ceiling bands (low / medium / high upper-tail character). Tertiles are the
# minimum interpretable risk grouping; shared with the CLI glyph map.
N_CEILING_BANDS = 3
# Family-wise level for the tier-cliff spike test (conventional 0.05; one-sided Bonferroni
# across the interior gaps). The test is a local curvature spike, robust to a convex board.
CLIFF_ALPHA = 0.05


def objective_weight(n_drafters: int) -> float:
    """Weight on P(1st) relative to P(2nd) implied by the payout: ``w = n_drafters - 1``."""
    if n_drafters < 2:
        raise ValueError(f"n_drafters must be >= 2; got {n_drafters}")
    return float(n_drafters - 1)


def _position_credit(scores: np.ndarray, drafter: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-simulation fractional credit that ``drafter`` finishes 1st and 2nd."""
    our = scores[:, drafter][:, None]
    strictly_above = (scores > our).sum(axis=1)  # our own column contributes 0 (not > itself)
    group = (scores == our).sum(axis=1)  # tie-group size INCLUDING us (== contributes 1)
    r = strictly_above + 1  # best competition rank held by our tie group
    inv_g = 1.0 / group

    def credit_for(position: int) -> np.ndarray:
        in_group = (r <= position) & (position <= r + group - 1)
        return np.where(in_group, inv_g, 0.0)

    return credit_for(1), credit_for(2)


def first_place_credit(scores: np.ndarray, drafter: int) -> np.ndarray:
    """Per-simulation fractional 1st-place credit for ``drafter`` (public accessor)."""
    return _position_credit(scores, drafter)[0]


def placement_probs(scores: np.ndarray, drafter: int) -> dict[str, float]:
    """P(1st), P(2nd), and P(top-2) for ``drafter``. P1 matches ``win_probability``."""
    c1, c2 = _position_credit(scores, drafter)
    p1 = float(c1.mean())
    p2 = float(c2.mean())
    return {"p1": p1, "p2": p2, "p_money": p1 + p2}


def objective_W(scores: np.ndarray, drafter: int, n_drafters: int | None = None) -> float:
    """The objective ``W = (n-1)*P(1st) + P(2nd)`` for ``drafter``."""
    n = scores.shape[1] if n_drafters is None else n_drafters
    pp = placement_probs(scores, drafter)
    return objective_weight(n) * pp["p1"] + pp["p2"]


def objective_W_se(scores: np.ndarray, drafter: int, n_drafters: int | None = None) -> float:
    """Monte-Carlo standard error of the :func:`objective_W` estimate for ``drafter``.

    ``W`` is the sample mean of the per-simulation quantity
    ``w_i = (n-1)*credit1_i + credit2_i``; its SE is the ordinary MC SE of a sample mean,
    ``sd(w_i)/sqrt(n_sims)``, computed from the realised ``w_i`` so it captures the empirical
    covariance between the 1st- and 2nd-place credits (usually negative, but jointly positive
    in tie-for-first replicates).
    """
    n = scores.shape[1] if n_drafters is None else n_drafters
    c1, c2 = _position_credit(scores, drafter)
    w_i = objective_weight(n) * c1 + c2
    return float(np.std(w_i, ddof=1) / np.sqrt(w_i.shape[0]))


def ceiling_deep_run(points: np.ndarray, stages: np.ndarray, deep_stage: int) -> np.ndarray:
    """Per-team upper-tail ("ceiling") value: expected points earned from deep runs only.

    ``ceiling[t] = E[ points[:, t] * 1{ stage[:, t] >= deep_stage } ]`` -- the part of a
    team's expected points that comes from reaching ``deep_stage`` (semi-final) or beyond,
    which under a convex ladder is where the big, pool-winning scores live. Roster-independent.
    """
    deep = (stages >= deep_stage).astype(np.float64)
    return (points * deep).mean(axis=0)


def ceiling_bands(ceiling_values: np.ndarray, n_bands: int = N_CEILING_BANDS) -> np.ndarray:
    """Map ceiling values to integer bands ``0..n_bands-1`` by quantile (high = better).

    Band edges are empirical quantiles of the supplied values (no arbitrary cutoff). Ties are
    resolved with average ranks, so identical ceilings receive identical bands and the output
    is invariant to input order. Band over the *displayed* candidate set for a useful spread.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    vals = np.asarray(ceiling_values, dtype=float)
    if vals.size == 0:
        return np.empty(0, dtype=int)
    q = (rankdata(vals, method="average") - 0.5) / vals.size  # average-rank quantile in (0,1)
    return np.clip((q * n_bands).astype(int), 0, n_bands - 1)


def detect_cliffs(
    w_sorted_desc: np.ndarray,
    se_sorted: np.ndarray,
    alpha: float = CLIFF_ALPHA,
) -> list[int]:
    """Gap indices ``i`` where a tier cliff sits between candidate ``i`` and ``i+1``.

    A cliff is a **local spike**: gap ``i`` exceeds *both* neighbouring gaps (``i-1`` and
    ``i+1``) by more than Monte-Carlo noise. The contrast is a discrete second difference of
    ``W``; its SE combines the four candidates involved, and the threshold is a one-sided
    Bonferroni ``z`` across the ``k-3`` interior gaps at level ``alpha``. Because it tests
    local curvature rather than gap magnitude, a smoothly (even convexly) declining board --
    where gaps change monotonically -- yields no cliffs; only a genuine kink is flagged.
    ``se_sorted`` must align with ``w_sorted_desc``. Boundary gaps (first/last) are not tested.
    """
    w = np.asarray(w_sorted_desc, dtype=float)
    se = np.asarray(se_sorted, dtype=float)
    k = w.size
    if k < 4:
        return []
    gaps = -np.diff(w)  # length k-1
    n_tests = k - 3  # interior gap indices 1 .. k-3
    z = float(norm.ppf(1 - alpha / n_tests))  # one-sided Bonferroni
    cliffs = []
    for i in range(1, k - 2):
        excess = gaps[i] - max(gaps[i - 1], gaps[i + 1])
        se_contrast = float(np.sqrt(se[i - 1] ** 2 + se[i] ** 2 + se[i + 1] ** 2 + se[i + 2] ** 2))
        if (excess > z * se_contrast) if se_contrast > 0 else (excess > 0):
            cliffs.append(i)
    return cliffs
