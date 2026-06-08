import numpy as np
import pytest

from wcpool import metrics


def test_pool_scores_sums_roster():
    points = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    rosters = np.array([[0, 1], [2, 3]])  # 2 drafters, 2 teams each
    s = metrics.pool_scores(points, rosters)
    assert s.tolist() == [[3.0, 7.0], [30.0, 70.0]]


def test_win_probability_sums_to_one_and_handles_ties():
    scores = np.array([[1.0, 2.0], [3.0, 3.0]])  # sim2 is a tie
    wp = metrics.win_probability(scores)
    assert wp.sum() == pytest.approx(1.0)
    # sim1 drafter1 wins, sim2 split: drafter0 = 0.5*1/2 over 2 sims = 0.25; drafter1 = 0.75
    assert wp[0] == pytest.approx(0.25)
    assert wp[1] == pytest.approx(0.75)


def test_top_tie_rate():
    scores = np.array([[1.0, 2.0], [3.0, 3.0], [5.0, 1.0]])
    assert metrics.top_tie_rate(scores) == pytest.approx(1 / 3)


def test_spearman_perfect_correlation():
    # roster EV ordering exactly matches realised ordering in every sim
    rosters = np.array([[0], [1], [2]])
    team_ev = np.array([1.0, 2.0, 3.0])
    scores = np.array([[1.0, 2.0, 3.0], [0.5, 0.6, 0.7]])
    rho = metrics.spearman_per_sim(scores, team_ev, rosters)
    assert np.allclose(rho, 1.0)


def test_skill_variance_share_bounds():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(500, 6)) + np.arange(6)  # drafters differ in mean
    team_ev = np.arange(6, dtype=float)
    rosters = np.arange(6).reshape(6, 1)
    out = metrics.skill_variance_share(scores, team_ev, rosters)
    assert 0.0 <= out["skill_share"] <= 1.0


def test_champion_dominance_toy():
    # 2 sims, 2 drafters; champion team index per sim; drafter_of_team mapping
    scores = np.array([[5.0, 1.0], [1.0, 5.0]])
    champion_team = np.array([0, 3])
    drafter_of_team = np.array([0, 0, 1, 1])  # teams 0,1 -> d0 ; 2,3 -> d1
    out = metrics.champion_dominance(scores, champion_team, drafter_of_team)
    # sim1 champion held by d0 who wins; sim2 champion held by d1 who wins -> 1.0
    assert out["p_champion_holder_wins"] == pytest.approx(1.0)
    assert out["champion_undrafted_rate"] == pytest.approx(0.0)


def test_champion_undrafted_detected():
    scores = np.array([[5.0, 1.0]])
    champion_team = np.array([2])
    drafter_of_team = np.array([0, 1, -1])  # team 2 undrafted
    out = metrics.champion_dominance(scores, champion_team, drafter_of_team)
    assert out["champion_undrafted_rate"] == pytest.approx(1.0)
    assert np.isnan(out["p_champion_holder_wins"])
