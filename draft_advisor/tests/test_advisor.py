import numpy as np
import pytest

from draft_advisor import advisor as A


def test_snake_auto_attribution():
    st = A.DraftState(3, 2, our_seat=0, picks=[5, 4, 3, 2])  # seq [0,1,2,2,1,0]
    assert st.roster_of(0) == [5]
    assert st.roster_of(2) == [3, 2]
    assert st.our_roster() == [5]
    assert st.our_pick_numbers() == [1, 6]


def test_turn_and_gap():
    st = A.DraftState(3, 2, 0, picks=[])
    assert st.is_our_turn()
    assert st.picks_until_our_next() == 4
    done = A.DraftState(3, 2, 0, picks=[9, 8, 7, 6, 5, 4])
    assert done.is_complete and not done.is_our_turn()


def test_available_mask_and_validation(make_board):
    bd = make_board(np.ones((10, 6)))
    st = A.DraftState(2, 2, 0, picks=[1, 3])
    mask = st.available_mask(bd.n_teams)
    assert not mask[1] and not mask[3] and mask[0]
    with pytest.raises(ValueError):
        A.DraftState(2, 2, 0, picks=[1, 1])


def test_recommend_ranks_dominant_first(make_board):
    rng = np.random.default_rng(0)
    base = np.array([21.0, 15, 10, 6, 3, 1])
    pts = base + rng.normal(0, 0.5, size=(300, 6))
    bd = make_board(pts)
    st = A.DraftState(2, 2, 0, picks=[])
    rec = A.recommend(st, bd, r_samples=4, rng=rng)
    assert rec.rows[0].team == 0
    ws = [r.w0 for r in rec.rows]
    assert ws == sorted(ws, reverse=True)
    assert len(rec.rows) == 6
    assert isinstance(rec.rows[0].robust_top, bool)


def test_recommend_ranking_is_deterministic(make_board):
    # the tau=0 ranking uses no RNG -> identical order on repeated/independent calls
    rng = np.random.default_rng(3)
    pts = np.abs(rng.normal(5, 3, size=(200, 6)))
    bd = make_board(pts)
    st = A.DraftState(2, 2, 0, picks=[])
    order_a = [r.team for r in A.recommend(st, bd, rng=np.random.default_rng(1)).rows]
    order_b = [r.team for r in A.recommend(st, bd, rng=np.random.default_rng(99)).rows]
    assert order_a == order_b


def test_recommend_single_candidate(make_board):
    bd = make_board(np.abs(np.random.default_rng(4).normal(5, 2, size=(120, 4))))
    st = A.DraftState(2, 2, 0, picks=[1, 2, 0])  # seq [0,1,1,0]; our turn, one team left
    rec = A.recommend(st, bd)
    assert len(rec.rows) == 1 and rec.rows[0].team == 3
    assert rec.cliffs == []


def test_score_rejects_partial_roster(make_board):
    bd = make_board(np.ones((10, 6)))
    st = A.DraftState(2, 2, 0, picks=[1, 3])  # incomplete -> rosters() has -1
    with pytest.raises(ValueError):
        A._score(bd, st.rosters())


def test_recommend_works_off_turn(make_board):
    # snake order may drift ("muffled"); recommend must still produce our next-pick rec off-turn
    rng = np.random.default_rng(5)
    bd = make_board(np.abs(rng.normal(5, 2, size=(150, 6))))
    st = A.DraftState(2, 2, 0, picks=[0])  # we own team 0; snake now points to seat 2
    assert not st.is_our_turn()
    rec = A.recommend(st, bd)
    assert len(rec.rows) == 5  # 6 teams - 1 owned
    assert all(r.team != 0 for r in rec.rows)  # owned team excluded as a candidate


def test_forward_order_is_snake_over_remaining():
    assert A._forward_order({0: 1, 1: 2}, 2) == [0, 1, 1]
    assert A._forward_order({0: 2, 1: 2}, 2) == [0, 1, 1, 0]


def test_complete_and_score_shape(make_board):
    rng = np.random.default_rng(1)
    bd = make_board(np.abs(rng.normal(5, 2, size=(120, 6))))
    st = A.DraftState(2, 2, 0, picks=[])
    scores = A.complete_and_score(st.picks, st, bd, 0.0, rng)
    assert scores.shape == (120, 2)


def test_post_draft_summary(make_board):
    rng = np.random.default_rng(2)
    bd = make_board(np.abs(rng.normal(5, 3, size=(150, 6))))
    st = A.DraftState(2, 2, 0, picks=[0, 1, 2, 3])
    s = A.post_draft_summary(st, bd)
    assert len(s["roster"]) == 2
    assert s["swing_team"] in s["roster"]
    assert s["score_p05"] <= s["score_p50"] <= s["score_p95"]
