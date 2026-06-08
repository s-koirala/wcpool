import numpy as np

from draft_advisor import opponent as Opp


def test_softmax_argmax_at_zero_temperature():
    p = Opp.softmax_pick_probs(np.array([1.0, 5.0, 2.0]), 0.0)
    assert p[1] == 1.0 and p.sum() == 1.0


def test_softmax_uniform_at_infinite_temperature():
    p = Opp.softmax_pick_probs(np.array([1.0, 5.0, 2.0]), np.inf)
    assert np.allclose(p, 1 / 3)


def test_softmax_monotone_in_ev():
    p = Opp.softmax_pick_probs(np.array([1.0, 2.0, 3.0]), 1.0)
    assert p[0] < p[1] < p[2]


def test_temperature_grid_endpoints():
    grid = Opp.temperature_grid(np.array([1.0, 2.0, 3.0, 4.0]))
    assert grid[0] == 0.0 and np.isinf(grid[-1]) and len(grid) >= 3


def test_sample_pick_stays_available():
    rng = np.random.default_rng(0)
    ev = np.arange(5.0)
    idx = np.array([1, 3, 4])
    for _ in range(50):
        assert Opp.sample_pick(ev, idx, 1.0, rng) in idx


def test_greedy_pick_is_argmax_over_available():
    ev = np.array([9.0, 1.0, 8.0, 2.0])
    assert Opp.greedy_pick(ev, np.array([1, 2, 3])) == 2
