import numpy as np
import pytest

from wcpool import draft, metrics


def test_snake_sequence_pattern():
    seq = draft.snake_sequence(6, 3)
    assert seq[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert seq[6:12].tolist() == [5, 4, 3, 2, 1, 0]
    assert seq[12:18].tolist() == [0, 1, 2, 3, 4, 5]
    assert len(seq) == 18


def test_ev_greedy_picks_in_value_order():
    ev = np.arange(48, dtype=float)  # team 47 best, 46 next, ...
    res = draft.draft_ev_greedy(ev, n_drafters=6, n_rounds=4)
    rosters = res["rosters"]
    # snake: drafter 0 picks the very best (47), drafter 5 picks 42 in round 1
    assert rosters[0, 0] == 47
    assert rosters[5, 0] == 42
    # no team drafted twice, exactly 24 drafted
    drafted = rosters[rosters >= 0]
    assert len(drafted) == 24
    assert len(set(drafted.tolist())) == 24


def test_drafter_of_team_consistency():
    ev = np.random.default_rng(0).random(48)
    res = draft.draft_ev_greedy(ev, 6, 6)
    owner = res["drafter_of_team"]
    for d in range(6):
        for t in res["rosters"][d]:
            assert owner[t] == d
    assert (owner == -1).sum() == 48 - 36  # 12 undrafted


def test_variance_policy_targets_variance():
    var = np.zeros(48)
    var[10] = 100.0  # single high-variance team
    res = draft.draft_variance(var, 6, 4)
    assert res["rosters"][0, 0] == 10  # first overall pick grabs it


@pytest.mark.parametrize("p", [6, 7, 8])
def test_draft_generalises_to_participant_count(p):
    rng = np.random.default_rng(0)
    ev = rng.random(48)
    res = draft.draft_ev_greedy(ev, n_drafters=p, n_rounds=6)
    rosters = res["rosters"]
    drafted = rosters[rosters >= 0]
    assert len(drafted) == p * 6  # all picks made
    assert len(set(drafted.tolist())) == p * 6  # no team drafted twice
    wp = metrics.win_probability(metrics.pool_scores(rng.random((200, 48)), rosters))
    assert wp.shape == (p,)
    assert wp.sum() == pytest.approx(1.0)  # win prob distributes over all participants


def test_best_response_not_worse_than_ev_greedy():
    rng = np.random.default_rng(2)
    n_teams = 48
    # points matrix with a few high-ceiling teams
    points = rng.gamma(shape=1.5, scale=2.0, size=(800, n_teams))
    team_ev = points.mean(axis=0)
    br_slot = 0
    n_rounds = 4

    greedy = draft.draft_ev_greedy(team_ev, 6, n_rounds)
    greedy_wp = metrics.win_probability(metrics.pool_scores(points, greedy["rosters"]))[br_slot]

    br = draft.draft_best_response(points, team_ev, br_slot, 6, n_rounds)
    br_wp = metrics.win_probability(metrics.pool_scores(points, br["rosters"]))[br_slot]

    # In-sample guarantee: when the decision batch (points) IS the scoring batch and the
    # BR drafter picks first, its choice set is a superset of the greedy pick, so it weakly
    # dominates. (Out-of-sample, scored on an independent batch, this need NOT hold.)
    assert br_wp >= greedy_wp - 1e-9
