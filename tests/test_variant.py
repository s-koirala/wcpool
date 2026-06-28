"""The group-W/D/L + progression-bonus scoring variant.

Covers the cumulative-bonus construction, the two-component team scoring, the engine flag
that exposes group-match points, and an 8-participant end-to-end pool score.
"""

from pathlib import Path

import numpy as np
import pytest

from wcpool import draft as draft_mod
from wcpool import ladders, metrics
from wcpool.config import load_field
from wcpool.ladders import Stage
from wcpool.simulate import champion_team, make_elo_config
from wcpool.tournament import N_TEAMS, simulate_tournament

CONFIG = Path(__file__).resolve().parents[1] / "config" / "ratings_elo_2026.yaml"


# --- Cumulative bonus vector ------------------------------------------------------------


def test_progression_bonus_vector_matches_spec():
    # reach R32 +5, QF +8, SF +12, win +20; no milestone at R16 or at reaching the Final.
    # Cumulative furthest-stage totals: GROUP R32 R16 QF SF FINAL CHAMPION.
    assert ladders.PROGRESSION_BONUS.tolist() == [0, 5, 5, 13, 25, 25, 45]


def test_bonus_is_cumulative_and_nondecreasing():
    b = ladders.PROGRESSION_BONUS
    assert np.all(np.diff(b) >= 0)  # reaching a further stage never loses points
    # Champion total is exactly the sum of every milestone increment, each counted once.
    assert b[Stage.CHAMPION] == sum(ladders.PROGRESSION_BONUS_MILESTONES.values())
    # Flat segments where the spec defines no milestone.
    assert b[Stage.R16] == b[Stage.R32]  # no R16 milestone
    assert b[Stage.FINAL] == b[Stage.SF]  # no "reach the final" milestone (only winning)


def test_build_cumulative_bonus_is_general():
    # A dense ladder with a milestone at every stage accumulates the increments in order.
    inc = {"R32": 1, "R16": 1, "QF": 1, "SF": 1, "FINAL": 1, "CHAMPION": 1}
    assert ladders.build_cumulative_bonus(inc).tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_build_cumulative_bonus_rejects_unknown_stage():
    with pytest.raises(KeyError):
        ladders.build_cumulative_bonus({"R64": 3})


# --- Two-component team scoring ---------------------------------------------------------


def test_variant_team_points_combines_group_and_bonus():
    # One champion with a perfect group (9 pts), one group-stage exit (4 pts), one
    # semi-finalist (7 pts). Totals = group + bonus[stage].
    group = np.array([[9, 4, 7]])
    stages = np.array([[Stage.CHAMPION, Stage.GROUP, Stage.SF]])
    pts = ladders.variant_team_points(group, stages)
    assert pts.tolist() == [[9 + 45, 4 + 0, 7 + 25]]


def test_group_exit_scores_only_group_points():
    # The whole point of the variant: a team that fails to advance is no longer a dead
    # pick — it still banks its group-match points (here 0..9), unlike the terminal ladders.
    group = np.array([[0, 3, 6]])
    stages = np.zeros((1, 3), dtype=np.int64)  # all GROUP
    assert ladders.variant_team_points(group, stages).tolist() == [[0, 3, 6]]


# --- Engine flag: group points exposed --------------------------------------------------


def test_simulate_tournament_returns_group_points():
    field = load_field(CONFIG)
    model = make_elo_config(field).model
    groups = field.fixed_groups
    stages, group_pts = simulate_tournament(
        model, groups, n_sims=256, rng=np.random.default_rng(0), return_group_points=True
    )
    assert stages.shape == group_pts.shape == (256, N_TEAMS)
    # Three group matches per team: points are integers in [0, 9].
    assert group_pts.min() >= 0 and group_pts.max() <= 9
    # Exactly two teams per group win it and reach the knockouts -> 24 reach R32 each sim
    # (the 12 winners + 12 runners-up), plus 8 best thirds = 32 in the knockout stage.
    reached_ko = (stages >= Stage.R32).sum(axis=1)
    assert np.all(reached_ko == 32)


def test_return_group_points_flag_does_not_change_draws():
    # The flag only changes what is returned, not the simulated tournament.
    field = load_field(CONFIG)
    model = make_elo_config(field).model
    groups = field.fixed_groups
    stages_only = simulate_tournament(model, groups, 256, np.random.default_rng(7))
    stages_both, _ = simulate_tournament(
        model, groups, 256, np.random.default_rng(7), return_group_points=True
    )
    assert np.array_equal(stages_only, stages_both)


# --- 8-participant pool, end to end -----------------------------------------------------


def test_eight_player_pool_under_variant():
    field = load_field(CONFIG)
    model = make_elo_config(field).model
    groups = field.fixed_groups
    n_drafters, teams_each = 8, 6  # 8 * 6 = 48 fills the field exactly

    # EV batch -> team expected variant points -> snake draft; eval batch -> scoring.
    s_ev, g_ev = simulate_tournament(
        model, groups, 1500, np.random.default_rng(1), return_group_points=True
    )
    s_eval, g_eval = simulate_tournament(
        model, groups, 1500, np.random.default_rng(2), return_group_points=True
    )
    pts_ev = ladders.variant_team_points(g_ev, s_ev)
    pts_eval = ladders.variant_team_points(g_eval, s_eval)

    team_ev = pts_ev.mean(axis=0)
    res = draft_mod.draft_ev_greedy(team_ev, n_drafters, teams_each)
    assert res["rosters"].shape == (n_drafters, teams_each)
    assert sorted(res["rosters"].ravel().tolist()) == list(range(N_TEAMS))  # whole field used

    scores = metrics.pool_scores(pts_eval, res["rosters"])
    assert scores.shape == (1500, n_drafters)
    wp = metrics.win_probability(scores)
    assert wp.shape == (n_drafters,)
    assert wp.sum() == pytest.approx(1.0)

    # Champion is always owned when the field is fully drafted.
    champ = champion_team(s_eval)
    assert np.all(res["drafter_of_team"][champ] >= 0)
