"""Tests for the INTEGER-scheme tie measurement (``measure_integer_and_anchor_tie``).

Audit MAJOR 1: the recommended ladder we ship is the INTEGER scheme
``points = 3*wins + 1*draws + LADDER_INT[stage]`` (LADDER_INT = scale * A_triangular), whose set of
attainable totals is a coarser lattice than the swept *continuous convex-mix* scheme, so its top-tie
rate is lattice-dependent and must be MEASURED on the headline batch -- it cannot be read off the
continuous-mix gamma_match cell. The headline run measures it on 100 draws x 2000 sims; these tests
run a SMALL batch (fast, deterministic) and assert the two load-bearing properties:

1. **Replay correctness** -- the within-run mix=0 triangular ANCHOR measured by
   ``measure_integer_and_anchor_tie`` reproduces, to floating point, the SAME anchor metrics that
   ``simulate.run_strength_config`` produces for the matched batch (the section-8.4 reproduction
   contract). This guards that the integer number is trusted only because the replay matches the
   sweep.
2. **Integer-scheme invariants** -- the integer tie is a valid probability strictly below the
   dead-group anchor tie, and the integer scheme's measured skill matches the anchor's to within
   sampling noise (skill is affine invariant; the integer scheme adds a non-affine group layer but
   its skill stays in the same neighbourhood as the continuous gamma_match cell).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_recommendation as br  # noqa: E402

from wcpool import select, simulate  # noqa: E402
from wcpool.config import load_field  # noqa: E402
from wcpool.scoring import ScoringScheme  # noqa: E402

CONFIG_PATH = ROOT / "config" / "ratings_elo_2026.yaml"
HEAD_CSV = ROOT / "docs" / "tables" / "results_groupscoring_2026-06-09.csv"

# Small batch: fast + deterministic. Asserts the REPLAY EQUIVALENCE and the lattice invariants, not
# the absolute headline level (which uses 100 draws x 2000 sims).
SEED = br.GROUPSCORING_SEED
DRAWS = 3
SIMS = 300


@pytest.fixture(scope="module")
def recommended_integer_scheme():
    """The integerised recommended cell (triangular 3:1 gamma_match -> [9,27,54,90,135,189])."""
    head = pd.read_csv(HEAD_CSV)
    head = head[head["strength_config"] == "elo_2026"].copy()
    e_wins, e_draws = br._purse_components(head)
    sel = select.select_recommendation(
        head, engagement_level=br.ENGAGEMENT_FLOOR, prefer_gamma_match=True
    )
    rec = sel.recommended
    iscm = select.integerize_scheme(
        rec.shape, int(rec.w_pts), int(rec.d_pts), rec.gamma_realised, e_wins, e_draws,
        gamma_tol=br.INTEGER_GAMMA_TOL,
    )
    return iscm


@pytest.fixture(scope="module")
def measured(recommended_integer_scheme):
    return br.measure_integer_and_anchor_tie(
        recommended_integer_scheme, seed=SEED, n_draws=DRAWS, sims_per_draw=SIMS
    )


def test_recommended_integer_ladder_is_the_expected_scale9(recommended_integer_scheme):
    # Sanity on the scheme under test: the exact-commensurability scale-9 triangular ladder.
    iscm = recommended_integer_scheme
    assert iscm.ladder.tolist() == [0, 9, 27, 54, 90, 135, 189]
    assert (iscm.w_pts, iscm.d_pts) == (3, 1)


def test_within_run_anchor_reproduces_run_strength_config(measured):
    # REPLAY CORRECTNESS (audit MAJOR 1): the within-run mix=0 triangular anchor measured on the
    # replayed batch must equal run_strength_config's anchor metrics on the SAME batch to FP -- the
    # guard that the integer-tie measurement uses the exact headline-run streams.
    _integer_cell, anchor_cell = measured
    cfg = simulate.make_elo_config(load_field(CONFIG_PATH), name="elo_2026")
    recs = simulate.run_strength_config(
        cfg,
        schemes=[ScoringScheme("triangular", 1.0, 0.0, 0.0)],
        n_values=[6],
        policies=["ev_greedy"],
        n_draws=DRAWS,
        sims_per_draw=SIMS,
        seed=SEED,
        regime="resampled",
        cfg_id=0,
        n_drafters=8,
    )
    r0 = recs[0]
    assert anchor_cell.tie == pytest.approx(r0["top_tie_rate"], abs=1e-12)
    assert anchor_cell.skill == pytest.approx(r0["spearman_mean"], abs=1e-9)
    assert anchor_cell.champ_dom == pytest.approx(r0["p_champion_holder_wins"], abs=1e-9)
    assert anchor_cell.slot_spread == pytest.approx(r0["slot_win_prob_spread"], abs=1e-9)


def test_integer_tie_is_below_anchor_and_a_valid_probability(measured):
    # Lattice invariant: the integer scheme ties LESS than the dead-group anchor (the group layer
    # decorrelates totals), and the tie is a valid probability with a finite between-draw cluster.
    integer_cell, anchor_cell = measured
    assert 0.0 <= integer_cell.tie < anchor_cell.tie
    assert integer_cell.tie_cluster_se >= 0.0
    assert np.isfinite(integer_cell.tie_cluster_se)


def test_integer_skill_matches_anchor_within_noise(measured):
    # Skill is affine invariant; the integer scheme's non-affine group layer keeps skill in the same
    # neighbourhood as the gamma=0 anchor (the recommended cell costs << 1 cluster SE). On the small
    # batch the two skills agree to within a loose sampling tolerance.
    integer_cell, anchor_cell = measured
    assert integer_cell.skill == pytest.approx(anchor_cell.skill, abs=0.02)
