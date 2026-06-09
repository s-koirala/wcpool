"""Tests for the engagement-constrained-skill selection + integerisation (:mod:`wcpool.select`).

Covers the three decision-layer pieces (plan sections 5, 8, 14): the deterministic bracket
stage-occupancy purse identity, the closed-form integer-scheme gamma, the prior-minimax shape
choice (must recover the shipped triangular default), the engagement-constrained-skill within-shape
recommendation, and the integerisation (smallest monotone integer scale hitting the target gamma,
with the exact-commensurability property of the headline gamma_match landmark).

These run on small hand-built frames + the committed frontier CSV (no simulation), so they are fast
and deterministic.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wcpool import select
from wcpool.ladders import N_STAGES, Stage, get_ladder

TABLES = Path(__file__).resolve().parents[1] / "docs" / "tables"
HEAD_CSV = TABLES / "results_groupscoring_2026-06-09.csv"
ROBUST_CSV = TABLES / "results_groupscoring_robustness_2026-06-09.csv"

# Calibration purse components of the real-Elo field (recovered from the committed frontier's
# group_purse columns; (1,0)->E[sum wins], (2,1)->2E[wins]+E[draws]). Structural, not magic.
E_SUM_WINS = 57.80053710937
E_SUM_DRAWS = 28.39893750000


@pytest.fixture(scope="module")
def head_df():
    df = pd.read_csv(HEAD_CSV)
    return df[df["strength_config"] == "elo_2026"].copy()


# --- structural purse identity ----------------------------------------------------------


def test_stage_occupancy_is_the_structural_bracket_constant():
    # 48 teams; 32 advance so 16 group exits; the knockout is a clean binary tree from 32 ->
    # 16/8/4/2/1/1 losers per round + the champion. The vector sums to 48 and is GROUP..CHAMPION.
    occ = select.STAGE_OCCUPANCY
    assert occ.shape == (N_STAGES,)
    assert occ.tolist() == [16, 16, 8, 4, 2, 1, 1]
    assert occ.sum() == 48


@pytest.mark.parametrize("shape", ["linear", "triangular", "geometric"])
def test_knockout_purse_reproduces_calibrated_K(shape, head_df):
    # The closed-form purse <A_shape, occupancy> must equal the sweep's calibrated knockout_purse
    # column for every shape (the gamma=0 anchor carries it). This is the basis for exact, no-sim
    # integer-scheme gamma.
    k_closed = select.knockout_purse_of_ladder(get_ladder(shape))
    k_calib = float(
        head_df[(head_df["knockout_ladder"] == shape) & (head_df["mix"] == 0.0)][
            "knockout_purse"
        ].iloc[0]
    )
    assert k_closed == pytest.approx(k_calib, rel=1e-9)


def test_knockout_purse_rejects_wrong_length():
    with pytest.raises(ValueError, match="length 7"):
        select.knockout_purse_of_ladder(np.array([1, 2, 3]))


def test_gamma_of_integer_scheme_matches_frontier_gamma_for_a_known_cell(head_df):
    # For the recommended triangular 3:1 gamma_match cell, the closed-form integer-scheme gamma at
    # the EXACT-scale ladder (scale=9 -> [9,27,54,90,135,189]) must reproduce the cell's realised
    # gamma to FP (both are G/(G+K) on the same purses).
    lad = np.zeros(N_STAGES)
    lad[1:] = 9 * np.array([1, 3, 6, 10, 15, 21])
    g = select.gamma_of_integer_scheme(3, 1, lad, E_SUM_WINS, E_SUM_DRAWS)
    cell = head_df[
        (head_df["knockout_ladder"] == "triangular")
        & (head_df["w_pts"] == 3)
        & (head_df["d_pts"] == 1)
        & (head_df["is_gamma_match"])
    ]
    assert g == pytest.approx(float(cell["gamma_realised"].iloc[0]), abs=1e-4)


# --- engagement anchors -----------------------------------------------------------------


@pytest.mark.parametrize("shape", ["linear", "triangular", "geometric"])
def test_engagement_floor_is_exactly_zero_and_ceiling_is_equal_purse(shape, head_df):
    # Plan section 5: the status-quo floor (gamma=0 group_variance_share) is EXACTLY 0 by
    # construction; the ceiling is the equal-purse (gamma=0.5) share, > 0 and the swept maximum.
    a = select.engagement_anchors(head_df, shape)
    assert a.floor == 0.0
    assert a.ceiling > a.floor
    # ceiling must be the gamma=0.5 max-over-D/W share for that shape
    eqp = head_df[
        (head_df["knockout_ladder"] == shape)
        & np.isclose(head_df["gamma_realised"], 0.5, atol=1e-6)
    ]
    assert a.ceiling == pytest.approx(float(eqp["group_variance_share"].max()))


def test_engagement_anchor_attaches_balanced_field_ceiling():
    head = pd.read_csv(HEAD_CSV)
    head = head[head["strength_config"] == "elo_2026"]
    rb = pd.read_csv(ROBUST_CSV)
    bal = rb[rb["strength_config"] == "synthetic_x0.5"]
    a = select.engagement_anchors(
        head, "triangular", balanced_df=bal, balanced_field_name="synthetic_x0.5"
    )
    assert a.ceiling_balanced_field is not None
    assert 0.0 < a.ceiling_balanced_field < 1.0
    assert a.balanced_field_name == "synthetic_x0.5"


# --- shape selection (prior minimax) ----------------------------------------------------


def test_prior_minimax_shape_recovers_triangular(head_df):
    # Plan section 5 objective (iii) phi=0 limit: the prior minimax over the gamma=0 anchor ranks
    # must reproduce the shipped balanced default = triangular (worst-rank 2 beats lin/geo's 3).
    shape, ranks = select.select_shape_prior_minimax(head_df)
    assert shape == "triangular"
    assert ranks["triangular"]["worst_rank"] < ranks["linear"]["worst_rank"]
    assert ranks["triangular"]["worst_rank"] < ranks["geometric"]["worst_rank"]


def test_prior_minimax_shape_is_stable_across_robustness_fields():
    # The shape choice must be triangular in every concentration config (recommendation stability).
    rb = pd.read_csv(ROBUST_CSV)
    for cfg in ["synthetic_x0.5", "synthetic_x1", "synthetic_x2"]:
        sub = rb[rb["strength_config"] == cfg]
        shape, _ = select.select_shape_prior_minimax(sub)
        assert shape == "triangular", f"shape flipped in {cfg}"


# --- the headline recommendation --------------------------------------------------------


def test_recommendation_is_triangular_3to1_gamma_match(head_df):
    # The headline engagement-constrained-skill recommendation: triangular, 3:1, gamma_match.
    sel = select.select_recommendation(head_df, engagement_level=0.0001, prefer_gamma_match=True)
    r = sel.recommended
    assert sel.shape == "triangular"
    assert (r.w_pts, r.d_pts) == (3.0, 1.0)
    assert r.is_gamma_match
    assert r.gamma_realised == pytest.approx(0.1574, abs=2e-3)
    # The skill cost is tiny: < 1 absolute cluster SE (decision-irrelevant) though a few paired SE.
    assert -0.01 < r.delta_skill < 0.0  # a small but negative paired skill cost


def test_recommendation_3to1_is_skill_indistinguishable_from_other_dw(head_df):
    # The D/W tie-break toward 3:1 is justified: at gamma_match the paired-skill spread across the
    # three D/W is well within the absolute cluster SE (so 3:1's familiarity + low tie rate govern).
    tri = head_df[(head_df["knockout_ladder"] == "triangular") & (head_df["is_gamma_match"])]
    spread = tri["delta_skill_vs_anchor"].max() - tri["delta_skill_vs_anchor"].min()
    abs_se = float(
        head_df[head_df["knockout_ladder"] == "triangular"]["spearman_cluster_se"].median()
    )
    assert spread <= abs_se  # skill-indistinguishable across D/W at the decision-relevant precision


def test_recommendation_stable_across_engagement_floors(head_df):
    # Sweep the stated engagement floor over a wide band; the gamma_match recommendation stays
    # triangular 3:1 wherever gamma_match remains eligible (its own engagement ~0.0044).
    for lvl in [0.0001, 0.001, 0.002, 0.004]:
        sel = select.select_recommendation(head_df, engagement_level=lvl, prefer_gamma_match=True)
        assert sel.shape == "triangular"
        assert (sel.recommended.w_pts, sel.recommended.d_pts) == (3.0, 1.0)


def test_recommendation_raises_when_engagement_floor_unreachable_at_gamma_match(head_df):
    # gamma_match engagement (~0.0044) cannot meet an absurdly high floor -> ValueError.
    with pytest.raises(ValueError, match="gamma_match"):
        select.select_recommendation(head_df, engagement_level=0.9, prefer_gamma_match=True)


# --- select_within_shape + the prefer_gamma_match=False floor-driven path ----------------


def test_select_within_shape_is_lowest_gamma_familiar_3to1(head_df):
    # Direct cover of select_within_shape (the engagement-floor-driven skill-max WITHIN a shape):
    # among triangular gamma>0 cells meeting a trivial floor it returns the LOWEST-gamma,
    # skill-indistinguishable, familiar 3:1 cell (tie-break: skill-tol -> lowest gamma -> 3:1).
    c = select.select_within_shape(head_df, "triangular", 1e-4)
    assert c.shape == "triangular"
    assert (c.w_pts, c.d_pts) == (3.0, 1.0)  # familiar 3:1 wins the D/W tie-break
    assert not c.is_gamma_match  # the floor-driven path takes the lowest gamma, not the landmark
    assert c.group_variance_share >= 1e-4  # meets the stated floor
    # It is the lowest gamma among the eligible triangular cells (less skill surrendered).
    elig = head_df[
        (head_df["knockout_ladder"] == "triangular")
        & (head_df["mix"] > 0.0)
        & (head_df["group_variance_share"] >= 1e-4)
    ]
    assert c.gamma_realised == pytest.approx(float(elig["gamma_realised"].min()), abs=1e-9)


def test_select_within_shape_floor_binds_raises_lowest_gamma(head_df):
    # Raising the floor pushes the within-shape pick to a HIGHER lowest-eligible gamma (the floor
    # binds): floor 1e-3 excludes the gamma=0.05 cell (gvs < 1e-3), so the pick moves to gamma=0.12.
    c_low = select.select_within_shape(head_df, "triangular", 1e-4)
    c_high = select.select_within_shape(head_df, "triangular", 1e-3)
    assert c_high.gamma_realised > c_low.gamma_realised
    assert (c_high.w_pts, c_high.d_pts) == (3.0, 1.0)


def test_select_within_shape_raises_when_floor_unreachable(head_df):
    # No gamma>0 triangular cell reaches an absurd floor -> ValueError naming the shape + the max.
    with pytest.raises(ValueError, match="no gamma>0 triangular cell"):
        select.select_within_shape(head_df, "triangular", 0.9)


def test_recommendation_prefer_gamma_match_false_takes_lowest_gamma_cell(head_df):
    # The prefer_gamma_match=False branch (used to show the choice is STABLE across stated floors)
    # delegates to select_within_shape: it must return the lowest-gamma skill-indistinguishable 3:1
    # cell meeting the floor, NOT the gamma_match landmark.
    sel = select.select_recommendation(head_df, engagement_level=1e-4, prefer_gamma_match=False)
    r = sel.recommended
    assert sel.shape == "triangular"
    assert (r.w_pts, r.d_pts) == (3.0, 1.0)
    assert not r.is_gamma_match
    # Identical to the direct select_within_shape pick at the same floor (the branch just wraps it).
    direct = select.select_within_shape(head_df, "triangular", 1e-4)
    assert r.gamma_realised == pytest.approx(direct.gamma_realised, abs=1e-9)
    assert (r.w_pts, r.d_pts) == (direct.w_pts, direct.d_pts)


def test_select_within_shape_helpers_match_inline_logic(head_df):
    # The refactored shared tail (_skill_indistinguishable + _prefer_familiar_then_argmax) must
    # reproduce the original inline tie-break: build the candidate list, take the skill-near set,
    # lowest gamma, then familiar 3:1 + argmax, and confirm it equals select_within_shape's output.
    s = head_df[(head_df["knockout_ladder"] == "triangular") & (head_df["mix"] > 0.0)]
    elig = s[s["group_variance_share"] >= 1e-4]
    skill_tol = float(s["spearman_cluster_se"].median())
    cands = [select._row_to_candidate(r) for _, r in elig.iterrows()]
    near = select._skill_indistinguishable(cands, skill_tol)
    min_gamma = min(c.gamma_realised for c in near)
    low_gamma = [c for c in near if abs(c.gamma_realised - min_gamma) < 1e-9]
    expect = select._prefer_familiar_then_argmax(low_gamma)
    got = select.select_within_shape(head_df, "triangular", 1e-4)
    assert (got.w_pts, got.d_pts, got.gamma_realised) == (
        expect.w_pts,
        expect.d_pts,
        expect.gamma_realised,
    )


# --- integerisation ---------------------------------------------------------------------


def test_integerize_triangular_gamma_match_is_scale_9_exact_commensurability():
    # Plan section 14: smallest monotone integer scale hitting gamma_match within a tight window is
    # scale=9, which hits gamma EXACTLY and realises the commensurability in integers: 3 group wins
    # (3*W=9) == the R32 bank (A_int(R32)=9). Ladder [9,27,54,90,135,189]; bank [9,18,27,36,45,54].
    iscm = select.integerize_scheme(
        "triangular", 3, 1, 0.1574, E_SUM_WINS, E_SUM_DRAWS, gamma_tol=0.012
    )
    assert iscm.scale == 9
    assert iscm.ladder.tolist() == [0, 9, 27, 54, 90, 135, 189]
    assert iscm.increments.tolist() == [9, 18, 27, 36, 45, 54]
    assert iscm.gamma_realised == pytest.approx(0.1574, abs=1e-3)
    # exact integer commensurability: 3*W_group == A_int(R32)
    assert 3 * iscm.w_pts == int(iscm.ladder[int(Stage.R32)])


def test_integerize_preserves_increment_monotonicity():
    # Every realised ladder is strictly increasing over R32..CHAMPION (deeper runs bank more).
    for shape in ["linear", "triangular", "geometric"]:
        target = select.gamma_of_integer_scheme(
            3, 1, np.concatenate([[0], 5 * get_ladder(shape)[1:]]), E_SUM_WINS, E_SUM_DRAWS
        )
        iscm = select.integerize_scheme(
            shape, 3, 1, target, E_SUM_WINS, E_SUM_DRAWS, gamma_tol=0.02
        )
        ko = iscm.ladder[int(Stage.R32) :]
        assert np.all(np.diff(ko) > 0)
        assert np.all(iscm.increments > 0)
        # increments are the successive differences of the terminal ladder (plan s.2 bijection)
        assert iscm.increments.tolist() == np.diff(np.concatenate([[0], ko])).tolist()


def test_integerize_smaller_tolerance_picks_a_finer_scale_for_ties():
    # Plan section 14: a finer (larger) scale gives more attainable totals (the convexity-free tie
    # remedy). Raising the search floor (max_scale up, but forcing a smaller gamma_tol around a
    # different target) yields a strictly larger scale; here just assert a smaller integer scale
    # gives a coarser lattice (larger gamma step), monotone in scale.
    g_small = select.gamma_of_integer_scheme(
        3, 1, np.concatenate([[0], 9 * get_ladder("triangular")[1:]]), E_SUM_WINS, E_SUM_DRAWS
    )
    g_big = select.gamma_of_integer_scheme(
        3, 1, np.concatenate([[0], 18 * get_ladder("triangular")[1:]]), E_SUM_WINS, E_SUM_DRAWS
    )
    assert g_big < g_small  # larger knockout scale -> smaller group share (more knockout mass)


def test_integerize_raises_if_no_scale_in_window():
    # An impossibly tight tolerance around a gamma no integer scale reaches -> ValueError.
    with pytest.raises(ValueError, match="no integer scale"):
        select.integerize_scheme(
            "triangular", 3, 1, 0.1574, E_SUM_WINS, E_SUM_DRAWS, gamma_tol=1e-9, max_scale=50
        )


def test_calibration_purse_components_reproduce_frontier_group_purse(head_df):
    # Guard the recovered E[sum wins]/E[sum draws] constants against the committed CSV (so a future
    # re-run that changes them fails loudly rather than silently mis-integerising).
    gp10 = float(head_df[(head_df["w_pts"] == 1) & (head_df["d_pts"] == 0)]["group_purse"].iloc[0])
    gp31 = float(head_df[(head_df["w_pts"] == 3) & (head_df["d_pts"] == 1)]["group_purse"].iloc[0])
    assert gp10 == pytest.approx(E_SUM_WINS, rel=1e-6)
    assert gp31 == pytest.approx(3 * E_SUM_WINS + E_SUM_DRAWS, rel=1e-6)
    assert not math.isnan(gp31)
