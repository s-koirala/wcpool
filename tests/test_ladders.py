import numpy as np

from wcpool import ladders
from wcpool.ladders import Stage


def test_ladder_values_match_spec():
    assert ladders.LINEAR.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert ladders.TRIANGULAR.tolist() == [0, 1, 3, 6, 10, 15, 21]
    assert ladders.GEOMETRIC.tolist() == [0, 1, 2, 4, 8, 16, 32]


def test_group_stage_scores_zero_for_all_ladders():
    for ladder in ladders.LADDERS.values():
        assert ladder[Stage.GROUP] == 0


def test_triangular_is_cumulative_increments():
    # T_k = 1+2+...+k
    expected = np.cumsum([1, 2, 3, 4, 5, 6])
    assert ladders.TRIANGULAR[1:].tolist() == expected.tolist()


def test_geometric_is_powers_of_two():
    assert ladders.GEOMETRIC[1:].tolist() == [2**k for k in range(6)]


def test_points_for_stages_vectorised():
    stages = np.array([[0, 6], [1, 3]])
    pts = ladders.points_for_stages(stages, ladders.GEOMETRIC)
    assert pts.tolist() == [[0, 32], [1, 4]]


def test_get_ladder_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        ladders.get_ladder("nope")
