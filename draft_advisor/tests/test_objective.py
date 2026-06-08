import numpy as np

from draft_advisor import objective as O


def test_placement_clean_order():
    scores = np.array([[10, 5, 1], [10, 5, 1]], dtype=float)
    assert O.placement_probs(scores, 0) == {"p1": 1.0, "p2": 0.0, "p_money": 1.0}
    assert O.placement_probs(scores, 1) == {"p1": 0.0, "p2": 1.0, "p_money": 1.0}
    assert O.placement_probs(scores, 2)["p_money"] == 0.0


def test_tie_for_first_splits_credit():
    scores = np.array([[10, 10, 1]], dtype=float)
    pp = O.placement_probs(scores, 0)
    assert pp["p1"] == 0.5 and pp["p2"] == 0.5  # two tied -> half 1st, half 2nd


def test_p1_matches_engine_win_probability():
    from wcpool.metrics import win_probability

    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=(500, 4)).astype(float)
    wp = win_probability(scores)
    for d in range(4):
        assert abs(O.placement_probs(scores, d)["p1"] - wp[d]) < 1e-12


def test_objective_weight():
    assert O.objective_weight(8) == 7.0
    assert O.objective_weight(7) == 6.0


def test_objective_W_is_weighted_sum():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 6, size=(400, 5)).astype(float)
    pp = O.placement_probs(scores, 2)
    assert abs(O.objective_W(scores, 2, 5) - (4 * pp["p1"] + pp["p2"])) < 1e-12


def test_W_se_zero_for_constant_scores():
    scores = np.tile([10.0, 5.0, 1.0], (50, 1))
    assert O.objective_W_se(scores, 0, 3) == 0.0


def test_ceiling_bands_monotone():
    bands = O.ceiling_bands(np.array([1.0, 2, 3, 4, 5, 6]), n_bands=3)
    assert bands[0] == 0 and bands[-1] == 2
    assert list(bands) == sorted(bands)


def test_detect_cliffs_finds_a_midboard_kink():
    w = np.array([1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # gaps .1 .1 .1 .4 .1 .1 -> spike at 3
    assert O.detect_cliffs(w, np.zeros_like(w)) == [3]


def test_detect_cliffs_respects_noise():
    # uniform gaps -> no local spike regardless of noise scale
    w = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    assert O.detect_cliffs(w, np.full_like(w, 1.0)) == []


def test_detect_cliffs_no_false_cliff_on_smooth_decline():
    # a smooth linear descent (uniform gaps) has no tier structure -> no cliff
    w = np.linspace(1.0, 0.0, 12)
    assert O.detect_cliffs(w, np.full_like(w, 1e-6)) == []


def test_detect_cliffs_no_false_cliff_on_convex_decline():
    # convex board: gaps grow monotonically toward the top -> no local spike (round-2 failure)
    w = np.array([10.0, 6, 3.3, 1.8, 1.0, 0.6, 0.4, 0.3])
    assert O.detect_cliffs(w, np.full_like(w, 1e-6)) == []


def test_ceiling_bands_ties_get_equal_bands():
    bands = O.ceiling_bands(np.array([5.0, 5, 5, 5, 5, 5]), n_bands=3)
    assert len(set(bands)) == 1  # identical values -> identical band, order-invariant


def test_first_place_credit_is_per_sim_and_means_to_p1():
    scores = np.array([[10.0, 5, 1], [1, 5, 10]])
    c1 = O.first_place_credit(scores, 0)
    assert c1.shape == (2,)
    assert abs(c1.mean() - O.placement_probs(scores, 0)["p1"]) < 1e-12
