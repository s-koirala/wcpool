"""End-to-end: a tiny experiment cell runs and produces sane metric records."""

from pathlib import Path

import numpy as np
import pytest

from wcpool import simulate
from wcpool.config import load_field

CONFIG = Path(__file__).resolve().parents[1] / "config" / "ratings_elo_2026.yaml"


def test_load_field_structure():
    field = load_field(CONFIG)
    assert len(field.names) == 48
    assert field.elo.shape == (48,)
    assert field.fixed_groups.shape == (12, 4)
    assert sorted(field.fixed_groups.ravel().tolist()) == list(range(48))
    assert field.pots.shape == (4, 12)


def test_run_strength_config_small():
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field)
    records = simulate.run_strength_config(
        cfg,
        ladders=["linear", "geometric"],
        n_values=[4, 6],
        policies=["ev_greedy", "variance"],
        n_draws=3,
        sims_per_draw=200,
        seed=123,
        regime="resampled",
    )
    # 2 ladders x 2 N x 2 policies = 8 cells
    assert len(records) == 8
    for r in records:
        assert -1.0 <= r["spearman_mean"] <= 1.0
        assert 0.0 <= r["skill_variance_share"] <= 1.0
        assert 0.0 <= r["top_tie_rate"] <= 1.0
        assert 0.0 <= r["champion_undrafted_rate"] <= 1.0
        slot_probs = [r[f"slot{i + 1}_win_prob"] for i in range(6)]
        assert sum(slot_probs) == pytest.approx(1.0)
        assert r["n_eval_tournaments"] == 3 * 200


def test_participant_count_field_boundary():
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field)
    # 8 participants x 6 teams = 48 exactly fills the field -> champion always owned
    recs = simulate.run_strength_config(
        cfg, ladders=["linear"], n_values=[6], policies=["ev_greedy"],
        n_draws=2, sims_per_draw=200, seed=1, n_drafters=8,
    )
    assert recs[0]["champion_undrafted_rate"] == 0.0
    # 8 x 7 = 56 > 48 -> guard raises
    with pytest.raises(ValueError):
        simulate.run_strength_config(
            cfg, ladders=["linear"], n_values=[7], policies=["ev_greedy"],
            n_draws=1, sims_per_draw=50, seed=1, n_drafters=8,
        )


def test_synthetic_config_concentration_monotone():
    # higher rating spread -> more title probability concentrated in the top 8
    # same seed + rating_stream -> same underlying z draw, so this isolates the spread effect
    low = simulate.make_synthetic_config(spread=60.0, name="lo", seed=1)
    high = simulate.make_synthetic_config(spread=240.0, name="hi", seed=1)
    rng_l = np.random.default_rng(0)
    rng_h = np.random.default_rng(0)
    share_low = simulate.topk_ev_share(
        simulate.team_title_prob(
            simulate.simulate_tournament(low.model, low.fixed_groups, 2000, rng_l)
        ),
        8,
    )
    share_high = simulate.topk_ev_share(
        simulate.team_title_prob(
            simulate.simulate_tournament(high.model, high.fixed_groups, 2000, rng_h)
        ),
        8,
    )
    assert share_high > share_low
