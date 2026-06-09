"""Tests for the engagement metric suite (:mod:`wcpool.metrics`, group-stage extension).

Organised in the plan's priority order:

* Priority 1 -- the estimator-free engagement headline: :func:`~wcpool.metrics.stage_variance_share`
  (law-of-total-variance split) and :func:`~wcpool.metrics.alive_fraction` (feasibility bound).
* Priority 2 -- the numpy/scipy-only k-NN ``Wcond`` surrogate
  (:func:`~wcpool.metrics.conditional_win_prob`) plus the EFK
  :func:`~wcpool.metrics.pool_suspense`/:func:`~wcpool.metrics.pool_surprise`, the martingale
  sanity check, and the determinism check.
* Priority 3 -- the nested-simulation validation gate: a ground-truth ``Wcond`` from re-simulating
  the remaining knockout rounds (:func:`wcpool.tournament.replay_knockout_from_round`), compared to
  the surrogate's suspense/surprise within Monte-Carlo error.
"""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import norm

from wcpool import metrics, scoring
from wcpool import tournament as T
from wcpool.ladders import N_STAGES, Stage
from wcpool.scoring import ScoringScheme
from wcpool.strength import StrengthModel

# --- shared fixtures --------------------------------------------------------------------

ROSTERS_8x6 = np.arange(T.N_TEAMS).reshape(8, 6)  # 8 drafters x 6 teams = the full 48-team field


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    ratings = 1800 + 120 * rng.standard_normal(T.N_TEAMS)
    return StrengthModel(ratings)


@pytest.fixture
def groups():
    return np.arange(T.N_TEAMS).reshape(T.N_GROUPS, T.GROUP_SIZE)


def _simulate(model, groups, n_sims, seed):
    return T.simulate_tournament(
        model, groups, n_sims, np.random.default_rng(seed), return_group_results=True
    )


# --- Priority 1: estimator-free metrics -------------------------------------------------


def test_stage_variance_share_shares_sum_to_one_and_match_components():
    # Hand-built running trajectory: 2 drafters, controlled group and knockout increments so the
    # decomposition is checkable by hand. running[:, :, GROUP] = group increment G;
    # running[:, :, CHAMPION] = G + K (the final score). Interior boundaries are irrelevant here.
    n_sims = 400
    rng = np.random.default_rng(0)
    g = rng.normal(size=(n_sims, 2)) + np.array([1.0, 5.0])  # group increments
    k = rng.normal(size=(n_sims, 2)) * 2.0  # knockout increments (independent of g here)
    running = np.zeros((n_sims, 2, N_STAGES))
    running[:, :, int(Stage.GROUP)] = g
    running[:, :, int(Stage.CHAMPION)] = g + k

    out = metrics.stage_variance_share(running)
    # The three shares sum to exactly 1 (covariance share explicit).
    assert out["group_share"] + out["knockout_share"] + out["cov_share"] == pytest.approx(1.0)
    # Component variances reconcile with a direct drafter-summed final-score variance.
    var_final_direct = float(np.var(g + k, axis=0).sum())
    assert out["var_total"] == pytest.approx(var_final_direct)
    assert out["var_group"] + out["var_knockout"] + out["cov_term"] == pytest.approx(
        out["var_total"]
    )
    # G and K independent by construction -> cov_term ~ 0, group_share < knockout_share here.
    assert abs(out["cov_share"]) < 0.05
    assert out["group_share"] < out["knockout_share"]


def test_stage_variance_share_zero_total_is_nan():
    # All drafters flat at the same score every replicate -> zero variance -> nan shares.
    running = np.ones((10, 3, N_STAGES))
    out = metrics.stage_variance_share(running)
    assert math.isnan(out["group_share"])
    assert out["var_total"] == 0.0


def test_stage_variance_share_pooling_is_ratio_of_means(model, groups):
    # The pooled shares are a ratio of drafter-summed variances, not a mean of per-drafter ratios.
    # Verify group_share == sum_d Var(G_d) / sum_d Var(final_d) on a real trajectory.
    stages, group = _simulate(model, groups, 1500, seed=1)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.2)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    out = metrics.stage_variance_share(running)

    g = running[:, :, int(Stage.GROUP)]
    final = running[:, :, int(Stage.CHAMPION)]
    pooled_group = float(np.var(g, axis=0).sum())
    pooled_total = float(np.var(final, axis=0).sum())
    assert out["group_share"] == pytest.approx(pooled_group / pooled_total)
    # A real triangular/low-mix field resolves most variance in the knockouts.
    assert out["knockout_share"] > out["group_share"]


def test_alive_fraction_eliminated_drafter_not_alive_unless_leading():
    # Hand-built n_sims=1, 2 drafters x 2 teams. Drafter 0 holds two GROUP-eliminated teams; drafter
    # 1 holds two qualified teams and leads on group points. With a steep ladder, drafter 0 has
    # maxgain = 0 (no qualified teams) and trails -> NOT alive; drafter 1 leads -> alive.
    stages = np.array([[Stage.GROUP, Stage.GROUP, Stage.R16, Stage.QF]], dtype=np.int8)
    group = {
        "wins": np.array([[0, 0, 3, 3]], dtype=np.int32),  # drafter 1's teams won their groups
        "draws": np.array([[0, 0, 0, 0]], dtype=np.int32),
    }
    rosters = np.array([[0, 1], [2, 3]])  # drafter 0 = {GROUP, GROUP}; drafter 1 = {R16, QF}
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.3)
    running = scoring.running_scores(stages, group, scheme, rosters)
    out = metrics.alive_fraction(running, stages, scheme, rosters)

    by_slot = out["alive_fraction_by_slot"]
    assert by_slot[0] == 0.0  # eliminated + trailing -> dead
    assert by_slot[1] == 1.0  # leader -> always alive
    assert out["alive_fraction"] == pytest.approx(0.5)


def test_alive_fraction_live_team_can_overtake_is_alive():
    # Drafter 0 trails on group points but holds a qualified team whose feasible knockout upside
    # (a deep run) exceeds the deficit -> alive. The whole construction uses one team each so the
    # maxgain arithmetic is transparent.
    # group points (mix=1 component): drafter0 team has 0 wins/draws; drafter1 team has 3 wins.
    stages = np.array([[Stage.QF, Stage.GROUP]], dtype=np.int8)  # d0 team qualified (QF), d1 out
    group = {
        "wins": np.array([[0, 3]], dtype=np.int32),
        "draws": np.array([[0, 0]], dtype=np.int32),
    }
    rosters = np.array([[0], [1]])
    # Choose a mix where the knockout upside of d0's qualified team overtakes d1's group lead.
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.3)
    running = scoring.running_scores(stages, group, scheme, rosters)
    # d0 group-now = (1-mix)*ko[QF] + mix*0 ; d1 group-now = (1-mix)*0 + mix*(2*3)
    # maxgain[d0] = (1-mix)*(ko[CHAMPION]-ko[GROUP]) = 0.7*21; clearly >= d1's lead -> d0 alive.
    out = metrics.alive_fraction(running, stages, scheme, rosters)
    assert out["alive_fraction_by_slot"][0] == 1.0  # can still overtake
    assert out["alive_fraction_by_slot"][1] == 1.0  # leader on group points is alive
    assert out["alive_fraction"] == pytest.approx(1.0)


def test_alive_fraction_high_group_mass_eliminates_some(model, groups):
    # A flat ladder with heavy group mass gives little knockout upside, so trailing drafters become
    # mathematically dead -> alive_fraction strictly below 1 and bounded in [0, 1].
    stages, group = _simulate(model, groups, 2000, seed=2)
    scheme = ScoringScheme("linear", 3.0, 1.0, mix=0.9)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    out = metrics.alive_fraction(running, stages, scheme, ROSTERS_8x6)
    assert 0.0 <= out["alive_fraction"] < 1.0
    assert all(0.0 <= x <= 1.0 for x in out["alive_fraction_by_slot"])


def test_alive_fraction_affine_invariance(model, groups):
    # The feasibility inequality is preserved by a positive affine rescale of the point lattice
    # (a > 0, common offset), per the plan's invariance argument: rescaling w_pts/d_pts and the
    # ladder by the same factor leaves the alive set unchanged.
    stages, group = _simulate(model, groups, 1500, seed=3)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.4)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    base = metrics.alive_fraction(running, stages, scheme, ROSTERS_8x6)
    scaled = ScoringScheme("triangular", 2.0 * 3.0, 1.0 * 3.0, mix=0.4)
    running_scaled = scoring.running_scores(stages, group, scaled, ROSTERS_8x6)
    out = metrics.alive_fraction(running_scaled, stages, scaled, ROSTERS_8x6)
    assert out["alive_fraction"] == pytest.approx(base["alive_fraction"])


# --- Priority 2: Wcond surrogate + EFK metrics ------------------------------------------


def test_terminal_win_credit_matches_win_probability_convention():
    scores = np.array([[1.0, 2.0], [3.0, 3.0]])  # sim2 is a tie
    tc = metrics.terminal_win_credit(scores)
    assert np.allclose(tc.sum(axis=1), 1.0)
    assert np.allclose(tc, np.array([[0.0, 1.0], [0.5, 0.5]]))
    # Mean over replicates reproduces win_probability exactly.
    assert np.allclose(tc.mean(axis=0), metrics.win_probability(scores))


def test_knn_k_grid_is_powers_of_two_up_to_sqrt_n():
    assert metrics.knn_k_grid(2000) == [1, 2, 4, 8, 16, 32]  # floor(sqrt(2000)) = 44
    assert metrics.knn_k_grid(100) == [1, 2, 4, 8]  # floor(sqrt(100)) = 10
    assert metrics.knn_k_grid(2) == [1]  # floor(sqrt(2)) = 1
    assert metrics.knn_k_grid(1) == [1]  # degenerate guard


def test_conditional_win_prob_shape_simplex_and_terminal(model, groups):
    stages, group = _simulate(model, groups, 1200, seed=4)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.15)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)

    wcond = metrics.conditional_win_prob(running, tc)
    assert wcond.shape == (1200, 8, N_STAGES)
    # Every boundary's belief lies on the probability simplex (rows sum to 1, non-negative).
    assert np.allclose(wcond.sum(axis=1), 1.0)
    assert np.all(wcond >= -1e-12)
    # The CHAMPION boundary is the realised credit exactly (state determines the outcome).
    assert np.array_equal(wcond[:, :, int(Stage.CHAMPION)], tc)


def test_conditional_win_prob_validates_inputs():
    bad_running = np.zeros((10, 4, N_STAGES - 1))
    with pytest.raises(ValueError, match="running must be"):
        metrics.conditional_win_prob(bad_running, np.zeros((10, 4)))
    running = np.zeros((10, 4, N_STAGES))
    with pytest.raises(ValueError, match="terminal_credit shape"):
        metrics.conditional_win_prob(running, np.zeros((10, 3)))


def test_conditional_win_prob_is_deterministic(model, groups):
    # The leave-one-out k-selection and the cKDTree query carry no RNG, so two runs at identical
    # inputs produce bit-identical output (the plan's determinism contract).
    stages, group = _simulate(model, groups, 1000, seed=5)
    scheme = ScoringScheme("geometric", 2.0, 1.0, mix=0.25)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)
    w1 = metrics.conditional_win_prob(running, tc)
    w2 = metrics.conditional_win_prob(running, tc)
    assert np.array_equal(w1, w2)


def test_knn_loo_excludes_self_under_exact_duplicate_states():
    # Regression guard: cKDTree.query does not pin the self-match to column 0 when several states
    # are identical (pervasive at coarse boundaries -- running scores take few distinct values), so
    # the leave-one-out loop must exclude self BY IDENTITY, never include a replicate's own target.
    # Four identical states + two distinct ones; targets are distinct so leakage is detectable.
    state = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    target = np.eye(2)[[0, 1, 0, 1, 0, 1]].astype(float)  # alternating one-hot rows
    for k in (1, 2, 3):
        tree = cKDTree(state)
        _, idx = tree.query(state, k=k + 1)
        idx = np.atleast_2d(idx)
        is_self = idx == np.arange(state.shape[0])[:, None]
        order = np.argsort(is_self, axis=1, kind="stable")
        neigh = np.take_along_axis(idx, order, axis=1)[:, :k]
        assert not (neigh == np.arange(state.shape[0])[:, None]).any(), f"self leaked at k={k}"
    # And the public surrogate stays deterministic on a duplicate-heavy state.
    pred1 = metrics._knn_loo_predict(state, target, 2)
    pred2 = metrics._knn_loo_predict(state, target, 2)
    assert np.array_equal(pred1, pred2)


def test_wcond_martingale_property_necessary_check(model, groups):
    # NECESSARY-BUT-NOT-SUFFICIENT sanity test (plan section 6): for a belief martingale,
    # mean_s Wcond[:, d, b] is flat across b and equals the unconditional win_probability(scores).
    # This catches gross mis-normalisation/leakage but is ALSO passed by a degenerate zero-suspense
    # smoother, so it does NOT validate conditional calibration -- that is the nested-sim gate's job
    # (test_nested_simulation_gate below).
    stages, group = _simulate(model, groups, 3000, seed=6)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.15)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)
    wcond = metrics.conditional_win_prob(running, tc)

    wp = metrics.win_probability(scores)  # (n_drafters,)
    mean_by_b = wcond.mean(axis=0)  # (n_drafters, N_STAGES)
    # Flat across b and equal to the unconditional win prob, within k-NN leave-one-out smoothing
    # noise. The CHAMPION boundary is exact (== tc), so it pins the endpoint; interior boundaries
    # carry the smoothing wobble. Tolerance is generous relative to the ~1/n_drafters scale.
    assert np.max(np.abs(mean_by_b - wp[:, None])) < 0.05


def test_pool_suspense_surprise_structure_and_group_share(model, groups):
    stages, group = _simulate(model, groups, 2000, seed=7)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.15)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)
    wcond = metrics.conditional_win_prob(running, tc)

    susp = metrics.pool_suspense(wcond)
    surp = metrics.pool_surprise(wcond)
    for out in (susp, surp):
        assert out["total"] > 0.0
        # group + knockout phases partition the total; group_share in [0, 1].
        assert out["group_phase"] + out["knockout_phase"] == pytest.approx(out["total"])
        assert 0.0 <= out["group_share"] <= 1.0
        assert len(out["per_transition"]) == N_STAGES
        assert sum(out["per_transition"]) == pytest.approx(out["total"])
    # Reported SEPARATELY but, on this realised-path surrogate, their replicate means coincide by
    # the martingale increment identity (documented in pool_suspense/pool_surprise).
    assert susp["total"] == pytest.approx(surp["total"])


def test_pool_suspense_rises_with_group_mass(model, groups):
    # More group mass moves more belief during the group phase -> larger group-phase suspense share.
    # A monotone, threshold-free directional check of the headline number.
    stages, group = _simulate(model, groups, 2500, seed=8)
    shares = []
    for mix in (0.05, 0.5):
        scheme = ScoringScheme("linear", 3.0, 1.0, mix=mix)
        running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
        scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
        tc = metrics.terminal_win_credit(scores)
        wcond = metrics.conditional_win_prob(running, tc)
        shares.append(metrics.pool_suspense(wcond)["group_share"])
    assert shares[1] > shares[0]


def test_constant_predictor_yields_prior_and_zero_group_share_at_mix0(model, groups):
    # REGRESSION (critical audit finding 1): at a degenerate (near-)constant predictor the surrogate
    # must collapse onto the exact within-state mean, NOT a noisy k-NN sub-sample of the prior whose
    # sim-to-sim sampling variance manufactures a phantom prior->boundary belief jump.
    #
    # (i) A *fully constant* running trajectory => every boundary's Wcond is the unconditional prior
    #     BIT-FOR-BIT, so the EFK belief path is flat and suspense/surprise are exactly zero.
    rng = np.random.default_rng(0)
    tc_const = metrics.terminal_win_credit(rng.normal(size=(200, 6)))
    running_const = np.zeros((200, 6, N_STAGES))  # constant predictor at every boundary
    wc = metrics.conditional_win_prob(running_const, tc_const)
    prior = tc_const.mean(axis=0)
    for b in range(N_STAGES - 1):  # CHAMPION boundary is the realised credit by construction
        assert np.array_equal(wc[:, :, b], np.broadcast_to(prior, (200, 6))), (
            f"boundary {b} not bit-identical to the prior under a constant predictor"
        )
    susp_c, surp_c = metrics.pool_suspense(wc), metrics.pool_surprise(wc)
    assert susp_c["group_phase"] == 0.0 and surp_c["group_phase"] == 0.0
    assert math.isnan(susp_c["group_share"]) or susp_c["group_share"] == 0.0  # 0/0 total => nan ok

    # (ii) On the real engine the GROUP boundary at mix=0 is fully constant (the knockout floor is 0
    #      for every replicate and the group layer is off), so the suspense/surprise GROUP-PHASE
    #      share must be EXACTLY 0.0 -- the phantom previously sat near 0.04.
    stages, group = _simulate(model, groups, 2000, seed=11)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.0)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)
    sel: dict[int, int] = {}
    wcond = metrics.conditional_win_prob(running, tc, selected_k_out=sel)
    assert sel[int(Stage.GROUP)] == metrics.WITHIN_STATE_MEAN_K  # GROUP fell to within-state mean
    for fn in (metrics.pool_suspense, metrics.pool_surprise):
        out = fn(wcond)
        assert out["group_phase"] == 0.0
        assert out["group_share"] == 0.0
        assert out["per_transition"][0] == 0.0  # the prior->GROUP increment is identically zero


def test_conditional_win_prob_selected_k_cache_is_bit_identical(model, groups):
    # The selected-k cache (sweep optimisation, finding 6) must change NOTHING numerically: priming
    # selected_k_in with the CV-chosen k reproduces the from-scratch fit bit-for-bit on one batch.
    stages, group = _simulate(model, groups, 1500, seed=12)
    scheme = ScoringScheme("triangular", 2.0, 1.0, mix=0.12)
    running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
    scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
    tc = metrics.terminal_win_credit(scores)
    cache: dict[int, int] = {}
    w_cv = metrics.conditional_win_prob(running, tc, selected_k_out=cache)
    w_cached = metrics.conditional_win_prob(running, tc, selected_k_in=cache)
    assert np.array_equal(w_cv, w_cached)
    # An empty cache is treated as absent (CV runs); same-object in/out populates then reuses.
    same: dict[int, int] = {}
    w_a = metrics.conditional_win_prob(running, tc, selected_k_out=same, selected_k_in=same)
    w_b = metrics.conditional_win_prob(running, tc, selected_k_out=same, selected_k_in=same)
    assert np.array_equal(w_a, w_b) and np.array_equal(w_a, w_cv)


# --- Priority 3: nested-simulation validation gate --------------------------------------


def _total_increment(wcond, prior):
    """Per-replicate summed squared belief increment with the prior prepended (EFK kernel).

    This DELIBERATELY re-derives ``metrics._belief_path_with_prior`` / the squared-increment sum
    independently of the production code (it does not call the metrics helper) so the test is an
    INDEPENDENT cross-check of the surrogate's suspense kernel, not a tautological re-run of it. Do
    not DRY this against :func:`wcpool.metrics._efk_increment_summary` -- the duplication is the
    point.
    """
    n, d, _ = wcond.shape
    path = np.empty((n, d, N_STAGES + 1))
    path[:, :, 0] = prior
    path[:, :, 1:] = wcond
    return ((np.diff(path, axis=2)) ** 2).sum(axis=1).sum(axis=1)


def _crossterm_increment(half_a, half_b, prior):
    """Bias-corrected per-replicate total squared belief increment (fully-paired cross-term).

    The naive plug-in ``sum_b ||mu_hat_b - mu_hat_{b-1}||^2`` over-estimates the true
    ``sum_b ||mu_b - mu_{b-1}||^2`` by ``sum_b tr Var(mu_hat_b) ~ 1/n_inner`` (squaring a noisy
    belief inflates its increment). With two INDEPENDENT inner-sim halves at each boundary
    (``half_a``, ``half_b``), the fully-paired cross-term

        sum_b < mu_hat_b^A - mu_hat_{b-1}^A , mu_hat_b^B - mu_hat_{b-1}^B >

    is unbiased for the true increment: the A- and B-halves are independent given the realised
    conditioning bracket, so each transition's cross-product has expectation
    ``||mu_b - mu_{b-1}||^2`` with the variance terms cancelling. The prior (transition-0
    "previous") is the full-sample unconditional win prob, with negligible variance, and the
    CHAMPION boundary is exact, so both halves coincide there and match the naive estimator.
    """
    n, d, _ = half_a.shape

    def aug(x):
        p = np.empty((n, d, N_STAGES + 1))
        p[:, :, 0] = prior  # transition-0 "previous" = the (essentially exact) prior
        p[:, :, 1:] = x
        return p

    return (np.diff(aug(half_a), axis=2) * np.diff(aug(half_b), axis=2)).sum(axis=1).sum(axis=1)


def _within_mc_equivalence(m_surrogate, se_surrogate, m_gt, se_gt):
    """One-sided equivalence (non-inferiority) decision for the KNOWN one-sided smoothing bias.

    The surrogate's k-NN smoothing biases suspense DOWNWARD (documented, one-sided), so the
    appropriate confirmatory check (plan section 8.3, "confirmatory equivalence test") is NOT a
    symmetric TOST centred at zero but a one-sided non-inferiority test: the surrogate is acceptable
    iff it does not UNDER-state the ground-truth increment by more than the equivalence margin
    ``delta``. There is no hand-picked threshold (CLAUDE.md): ``delta`` is DERIVED entirely from the
    combined Monte-Carlo SE budget --

        delta = z_{0.995} * sqrt(se_surrogate^2 + se_gt^2)

    -- i.e. the two-sided 99% confirmatory resolution of the joint surrogate-increment +
    bias-corrected-GT-increment standard error. Anything below ``delta`` is indistinguishable from
    Monte-Carlo noise plus the bounded one-sided smoothing bias; ``delta`` shrinks as the budget
    grows. Decision (one-sided alpha = 0.05): conclude within-margin equivalence iff the lower 95%
    confidence bound on the (positive) understatement gap ``m_gt - m_surrogate`` is below
    ``delta``::

        (m_gt - m_surrogate) - z_{0.95} * sqrt(se_surrogate^2 + se_gt^2) < delta

    Returns ``(equivalent, diff, delta, se_combined)``.
    """
    se_combined = math.sqrt(se_surrogate**2 + se_gt**2)
    delta = norm.ppf(0.995) * se_combined  # 99% two-sided confirmatory resolution of the SE budget
    diff = m_gt - m_surrogate  # positive => surrogate understates the GT (the expected direction)
    lower_cb = diff - norm.ppf(0.95) * se_combined  # one-sided 95% lower bound on the gap
    return (lower_cb < delta), diff, delta, se_combined


def test_nested_simulation_gate(model, groups):
    # PRIORITY-3 GATE (plan section 6, spec budget). Compare the surrogate's suspense/surprise
    # increment to a nested-simulation ground truth at the spec subset size (>= 1000 replicates,
    # n_inner >= 800), PER MIX, and make the conclusion honest rather than a blanket pass:
    #
    #   * Low-mix headline cell (triangular, mix ~ 0.15): the surrogate is EXPECTED to agree with
    #     the ground truth -- its score-vector conditioning is nearly as informative as the full
    #     bracket when the group layer carries little weight. This agreement is the ENCODED gate
    #     assertion (within the SE-derived equivalence margin delta; see _within_mc_equivalence).
    #   * High-mix cell (triangular, mix ~ 0.5): the surrogate is EXPECTED to REJECT -- a heavy
    #     group layer moves belief in ways the coarse score-vector cannot resolve, so the surrogate
    #     materially under-states suspense. This rejection is RECORDED as the documented reliability
    #     boundary of the surrogate, NOT asserted as a pass.
    #
    # Both cells share ONE nested simulation (the knockout replays are scheme-independent), so the
    # comparison is paired on identical realised inner brackets.
    seed = 20260609
    n_sims, subset_n, n_inner = 1200, 1000, 800
    stages, group, r32_pairs, round_winners = T.simulate_tournament_trace(
        model, groups, n_sims, np.random.default_rng(1)
    )
    schemes = {
        "low": ScoringScheme("triangular", 2.0, 1.0, mix=0.15),  # headline cell -> expect PASS
        "high": ScoringScheme("triangular", 2.0, 1.0, mix=0.5),  # reliability boundary -> REJECT
    }
    subset = np.arange(subset_n)
    # One shared nested simulation scored under both schemes (cfg_id/draw threaded as 0, 0 in-test).
    wcn_list, ha_list, hb_list = T.nested_conditional_win_prob_multi(
        model, stages, group, r32_pairs, round_winners, list(schemes.values()),
        ROSTERS_8x6, subset, n_inner, seed, cfg_id=0, draw=0,
    )

    results = {}
    for (name, scheme), wcn, ha, hb in zip(
        schemes.items(), wcn_list, ha_list, hb_list, strict=True
    ):
        running = scoring.running_scores(stages, group, scheme, ROSTERS_8x6)
        scores = metrics.pool_scores(scoring.team_points(stages, group, scheme), ROSTERS_8x6)
        tc = metrics.terminal_win_credit(scores)
        wcond = metrics.conditional_win_prob(running, tc)
        prior = tc.mean(axis=0)

        # --- Ground-truth structural guarantees -----------------------------------------
        assert np.allclose(wcn.sum(axis=1), 1.0)  # every boundary on the simplex
        # CHAMPION boundary is EXACT: the resolved state determines the champion.
        assert np.allclose(wcn[:, :, int(Stage.CHAMPION)], tc[subset])
        # FINAL boundary is NOT the realised one-hot credit: the running state does not reveal the
        # champion (both finalists sit at ko[FINAL]), so the GT here is a genuine BLEND over who
        # wins the replayed final. It must still be a valid simplex per replicate, and on average
        # satisfy the martingale-mean property (mean over replicates ~ prior), but the per-replicate
        # FINAL belief must DIFFER materially from the one-hot terminal credit (otherwise the old
        # over-conditioning bug -- forcing FINAL == terminal credit -- would have returned).
        fb = wcn[:, :, int(Stage.FINAL)]
        assert np.allclose(fb.sum(axis=1), 1.0)  # FINAL on the simplex per replicate
        assert np.all(fb >= -1e-12)
        # Some replicate's FINAL belief is a non-degenerate two-way split (the typical case: two
        # distinct drafters hold the two finalists), so it is NOT the realised one-hot vector.
        assert np.max(np.abs(fb - tc[subset])) > 0.1, "FINAL collapsed onto terminal credit"
        nested_mean = wcn.mean(axis=0)
        assert np.max(np.abs(nested_mean - prior[:, None])) < 0.05  # nested martingale-mean

        # --- Increment statistics --------------------------------------------------------
        tot_surrogate = _total_increment(wcond[subset], prior)  # independent cross-check kernel
        tot_naive = _total_increment(wcn, prior)  # UPPER estimate (1/n_inner inflated)
        tot_cross = _crossterm_increment(ha, hb, prior)  # bias-corrected (unbiased) GT increment
        # The bias-corrected increment must be BELOW the naive plug-in (it removes a positive
        # variance inflation), confirming the correction acts in the right direction.
        assert tot_cross.mean() < tot_naive.mean()

        m_s, m_n = float(tot_surrogate.mean()), float(tot_cross.mean())
        se_s = float(tot_surrogate.std(ddof=1) / math.sqrt(subset_n))
        se_n = float(tot_cross.std(ddof=1) / math.sqrt(subset_n))
        equiv, diff, delta, se_comb = _within_mc_equivalence(m_s, se_s, m_n, se_n)
        results[name] = dict(
            m_s=m_s, m_n=m_n, naive=float(tot_naive.mean()), se_s=se_s, se_n=se_n,
            diff=diff, delta=delta, se_comb=se_comb, equiv=equiv,
        )

    lo, hi = results["low"], results["high"]
    # ENCODED GATE: the headline low-mix cell agrees with the GT within the SE-derived margin.
    assert lo["equiv"], (
        f"headline low-mix gate FAILED: surrogate {lo['m_s']:.4f} vs GT {lo['m_n']:.4f} "
        f"(diff {lo['diff']:+.4f}, delta {lo['delta']:.4f}, combined SE {lo['se_comb']:.4f})"
    )
    # The surrogate is not biased UPWARD beyond noise (k-NN smoothing cannot inflate suspense).
    assert lo["m_s"] <= lo["m_n"] + norm.ppf(0.995) * lo["se_comb"]
    # DOCUMENTED RELIABILITY BOUNDARY: the high-mix cell REJECTS (not a pass). The surrogate's
    # understatement there exceeds the SE-derived margin -- a genuine breakdown, recorded not masked
    assert not hi["equiv"], (
        f"high-mix cell unexpectedly within margin: surrogate {hi['m_s']:.4f} vs GT "
        f"{hi['m_n']:.4f} (diff {hi['diff']:+.4f}, delta {hi['delta']:.4f}) -- the documented "
        "reliability boundary (surrogate under-states suspense at high mass) did not reproduce"
    )
