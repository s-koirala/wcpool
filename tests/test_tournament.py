import numpy as np
import pytest

from wcpool import tournament as T
from wcpool.ladders import Stage
from wcpool.strength import StrengthModel


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    ratings = 1800 + 120 * rng.standard_normal(T.N_TEAMS)
    return StrengthModel(ratings)


@pytest.fixture
def groups():
    # simple draw: teams 0..47 laid out group-major
    return np.arange(T.N_TEAMS).reshape(T.N_GROUPS, T.GROUP_SIZE)


def test_thirdplace_table_complete_and_feasible():
    table = T.thirdplace_assignment_table()
    from math import comb

    assert len(table) == comb(12, 8)  # 495
    for mask, slot_group in table.items():
        groups_in = [g for g in range(12) if mask & (1 << g)]
        assert len(groups_in) == 8
        # assignment is a bijection of the 8 qualifying groups onto the 8 slots
        assert sorted(slot_group) == sorted(groups_in)
        # every slot's source group is eligible for that slot
        for s, g in enumerate(slot_group):
            assert g in T.THIRD_SLOT_ELIGIBILITY[s]


def test_random_pot_draw_structure():
    rng = np.random.default_rng(3)
    pots = np.arange(T.N_TEAMS).reshape(T.GROUP_SIZE, T.N_GROUPS)  # 4x12
    groups = T.random_pot_draw(pots, rng)
    assert groups.shape == (T.N_GROUPS, T.GROUP_SIZE)
    # all 48 teams present exactly once
    assert sorted(groups.ravel().tolist()) == list(range(T.N_TEAMS))
    # each within-group position p draws from pot p
    for p in range(T.GROUP_SIZE):
        assert set(groups[:, p].tolist()) == set(pots[p].tolist())


def test_group_stage_goal_and_point_conservation(model, groups):
    pts, gd, gf = T.simulate_group_stage(model, groups, n_sims=200, rng=np.random.default_rng(5))
    ga = gf - gd
    for g in range(T.N_GROUPS):
        gidx = groups[g]
        # goals scored in a group equal goals conceded in that group
        assert np.array_equal(gf[:, gidx].sum(axis=1), ga[:, gidx].sum(axis=1))
        # 6 matches -> total group points between 12 (all draws) and 18 (no draws)
        tot = pts[:, gidx].sum(axis=1)
        assert np.all((tot >= 12) & (tot <= 18))


def test_tournament_stage_counts_invariant(model, groups):
    stages = T.simulate_tournament(model, groups, n_sims=500, rng=np.random.default_rng(7))
    assert stages.shape == (500, T.N_TEAMS)
    expected = np.array([16, 16, 8, 4, 2, 1, 1])  # stages 0..6
    for row in stages:
        counts = np.bincount(row, minlength=7)
        assert np.array_equal(counts, expected)


def test_champion_is_unique(model, groups):
    stages = T.simulate_tournament(model, groups, n_sims=300, rng=np.random.default_rng(9))
    n_champ = (stages == int(Stage.CHAMPION)).sum(axis=1)
    assert np.all(n_champ == 1)


def test_stronger_field_advances_more_often(groups):
    # team 0 made hugely strong; it should reach late stages far more than an average team
    ratings = np.full(T.N_TEAMS, 1800.0)
    ratings[0] = 2400.0
    model = StrengthModel(ratings)
    stages = T.simulate_tournament(model, groups, n_sims=2000, rng=np.random.default_rng(11))
    assert stages[:, 0].mean() > stages[:, 1:].mean()
