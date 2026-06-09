"""Tests for the two-layer pool scorer (:mod:`wcpool.scoring`)."""

import math
from pathlib import Path

import numpy as np
import pytest

from wcpool import ladders, metrics, scoring, simulate
from wcpool.config import load_field
from wcpool.ladders import LADDERS, N_STAGES, Stage, get_ladder
from wcpool.scoring import GROUP_POINT_SCHEMES, ScoringScheme
from wcpool.tournament import (
    GROUP_MATCH_PAIRS,
    GROUP_SIZE,
    N_GROUPS,
    N_TEAMS,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "ratings_elo_2026.yaml"

N_GROUP_MATCHES = N_GROUPS * len(GROUP_MATCH_PAIRS)  # 72
GROUP_TOTAL_WINS = N_GROUP_MATCHES  # exactly one win allocated per decisive match


def _synthetic_group_panel(
    n_sims: int, is_draw: np.ndarray, winner_is_a: np.ndarray
) -> dict[str, np.ndarray]:
    """Build a valid (wins, draws) panel honouring the group round-robin structure.

    Lays the 48 teams out group-major (group ``g`` = teams ``4g..4g+3``) and plays each of
    the 72 round-robin matches per :data:`wcpool.tournament.GROUP_MATCH_PAIRS`. Each match is
    either a draw (both teams +1 draw) or decisive (the winner +1 win; the loser gets
    neither). Every team plays exactly 3 matches, so ``wins + draws + losses == 3`` and hence
    ``wins + draws <= 3`` per team in every replicate (matching the ``<= GROUP_SIZE - 1``
    assertion below).

    Parameters
    ----------
    n_sims : int
        Number of replicates.
    is_draw : numpy.ndarray
        ``(n_sims, 72)`` boolean: whether each match (group-major, pair-order) is drawn.
    winner_is_a : numpy.ndarray
        ``(n_sims, 72)`` boolean: for decisive matches, whether the lower-local-index team
        (``la``) won; ignored where ``is_draw`` is ``True``.
    """
    wins = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    draws = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    m = 0
    for g in range(N_GROUPS):
        for la, lb in GROUP_MATCH_PAIRS:
            a = g * GROUP_SIZE + la
            b = g * GROUP_SIZE + lb
            d = is_draw[:, m]
            decisive = ~d
            draws[:, a] += d
            draws[:, b] += d
            wins[:, a] += decisive & winner_is_a[:, m]
            wins[:, b] += decisive & ~winner_is_a[:, m]
            m += 1
    return {"wins": wins, "draws": draws}


def test_group_point_schemes_are_the_three_dw_candidates():
    assert GROUP_POINT_SCHEMES == {
        "fifa_3_1": (3.0, 1.0),
        "linear_2_1": (2.0, 1.0),
        "wins_only_1_0": (1.0, 0.0),
    }


def test_scoring_scheme_rejects_out_of_range_mix():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="mix"):
            ScoringScheme("linear", 3.0, 1.0, mix=bad)


def test_scoring_scheme_unknown_ladder_raises():
    scheme = ScoringScheme("nope", 3.0, 1.0)
    with pytest.raises(KeyError):
        scheme.knockout_vector()


def test_group_layer_validates_keys_and_shape_when_mix_positive():
    # When the group layer is active (mix > 0), both scorers require {"wins", "draws"} each
    # shaped exactly like stages, raising a clear ValueError otherwise.
    stages = np.zeros((3, 4), dtype=np.int8)
    rosters = np.array([[0, 1], [2, 3]])
    good = {
        "wins": np.zeros((3, 4), dtype=np.int32),
        "draws": np.zeros((3, 4), dtype=np.int32),
    }
    scheme = ScoringScheme("linear", 2.0, 1.0, mix=0.3)

    for scorer in (
        lambda g: scoring.team_points(stages, g, scheme),
        lambda g: scoring.running_scores(stages, g, scheme, rosters),
    ):
        with pytest.raises(ValueError, match="draws"):
            scorer({"wins": good["wins"]})  # missing "draws"
        with pytest.raises(ValueError, match=r"shape"):
            scorer({"wins": good["wins"], "draws": np.zeros((3, 5), dtype=np.int32)})


def test_group_layer_unvalidated_on_mix_zero_short_circuit():
    # The mix == 0 short-circuit ignores group entirely, so an absent/ill-shaped group must
    # NOT raise (and the result is the bare knockout lookup).
    stages = np.array([[Stage.R16, Stage.GROUP, Stage.QF, Stage.R32]], dtype=np.int8)
    rosters = np.array([[0, 1], [2, 3]])
    scheme = ScoringScheme("linear", 2.0, 1.0, mix=0.0)
    bad_group: dict = {}  # missing both keys
    expected_tp = ladders.points_for_stages(stages, get_ladder("linear"))
    assert np.array_equal(scoring.team_points(stages, bad_group, scheme), expected_tp)
    # running_scores likewise tolerates the bad group on the mix == 0 path.
    run = scoring.running_scores(stages, bad_group, scheme, rosters)
    assert run.shape == (1, 2, N_STAGES)


def test_constant_sum_identity_on_line_and_variation_off_line():
    # 2 * d_pts == w_pts -> the group-layer total over all 48 teams is 72 * w_pts in EVERY
    # replicate, independent of how many matches were drawn. Off-line schemes vary with draws.
    rng = np.random.default_rng(0)
    n_sims = 64
    DRAW_RATE = 0.4  # in (0, 1): exercises both drawn and decisive matches.
    FAIR_WIN = 0.5  # symmetric: neither local position wins systematically.
    is_draw = rng.random((n_sims, N_GROUP_MATCHES)) < DRAW_RATE
    winner_is_a = rng.random((n_sims, N_GROUP_MATCHES)) < FAIR_WIN
    group = _synthetic_group_panel(n_sims, is_draw, winner_is_a)

    # Sanity: the panel is valid -- each team plays 3 matches, so wins + draws <= 3 (the
    # remainder are losses, which carry neither a win nor a draw).
    assert np.all(group["wins"] + group["draws"] <= GROUP_SIZE - 1)
    # Conservation: across all teams, wins = #decisive matches and draws = 2 * #drawn matches.
    n_drawn = is_draw.sum(axis=1)
    assert np.array_equal(group["wins"].sum(axis=1), N_GROUP_MATCHES - n_drawn)
    assert np.array_equal(group["draws"].sum(axis=1), 2 * n_drawn)

    stages = np.zeros((n_sims, N_TEAMS), dtype=np.int8)  # group layer is stage-independent

    # On the constant-sum line (2, 1): total is exactly 72 * w_pts every replicate.
    w_pts, d_pts = GROUP_POINT_SCHEMES["linear_2_1"]
    on_line = ScoringScheme("linear", w_pts, d_pts, mix=1.0)
    total_on = scoring.team_points(stages, group, on_line).sum(axis=1)
    assert np.allclose(total_on, GROUP_TOTAL_WINS * w_pts)
    assert np.ptp(total_on) == pytest.approx(0.0)

    # Off the line (3, 1) and (1, 0): totals co-vary with the number of drawn matches.
    for name in ("fifa_3_1", "wins_only_1_0"):
        w_off, d_off = GROUP_POINT_SCHEMES[name]
        scheme_off = ScoringScheme("linear", w_off, d_off, mix=1.0)
        total_off = scoring.team_points(stages, group, scheme_off).sum(axis=1)
        # Closed form: w * (#decisive) + 2 * d * (#drawn).
        expected = w_off * (N_GROUP_MATCHES - n_drawn) + 2.0 * d_off * n_drawn
        assert np.allclose(total_off, expected)
        # Regression guard: the total genuinely varies replicate-to-replicate with draws.
        assert np.ptp(total_off) > 0.0


def test_mix_zero_reproduces_terminal_ladders_exactly():
    rng = np.random.default_rng(1)
    shape = (50, N_TEAMS)
    stages = rng.integers(0, N_STAGES, size=shape).astype(np.int8)
    group = {
        "wins": rng.integers(0, GROUP_SIZE, size=shape).astype(np.int32),
        "draws": rng.integers(0, GROUP_SIZE, size=shape).astype(np.int32),
    }
    for name in ladders.LADDERS:
        scheme = ScoringScheme(name, 3.0, 1.0, mix=0.0)
        expected = ladders.points_for_stages(stages, get_ladder(name))
        assert np.array_equal(scoring.team_points(stages, group, scheme), expected)


@pytest.mark.parametrize("mix", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("name", list(ladders.LADDERS))
def test_team_points_monotone_nondecreasing_in_stage(name, mix):
    # For fixed group counts, raising every team's stage cannot lower its two-layer points
    # (all ladders are non-decreasing in stage). mix in {0.0, 0.3} is the real coverage;
    # mix=1.0 is a vacuous guard here (knockout weight is 0, so the score has no stage
    # dependence at all) — kept only as a non-load-bearing boundary case.
    rng = np.random.default_rng(2)
    shape = (40, N_TEAMS)
    s1 = rng.integers(0, N_STAGES, size=shape).astype(np.int8)
    bump = rng.integers(0, N_STAGES, size=shape).astype(np.int8)
    s2 = np.minimum(s1.astype(np.int16) + bump, int(Stage.CHAMPION)).astype(np.int8)
    assert np.all(s2 >= s1)
    group = {
        "wins": rng.integers(0, GROUP_SIZE, size=shape).astype(np.int32),
        "draws": rng.integers(0, GROUP_SIZE, size=shape).astype(np.int32),
    }
    scheme = ScoringScheme(name, 2.0, 1.0, mix=mix)
    tp1 = scoring.team_points(s1, group, scheme)
    tp2 = scoring.team_points(s2, group, scheme)
    assert np.all(tp2 >= tp1 - 1e-9)


@pytest.mark.parametrize("mix", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("name", list(ladders.LADDERS))
def test_running_scores_reconciles_and_is_cumulative(name, mix):
    rng = np.random.default_rng(3)
    n_sims, n_drafters, per = 60, 8, 6
    n_teams = n_drafters * per
    rosters = rng.permutation(n_teams).reshape(n_drafters, per)
    stages = rng.integers(0, N_STAGES, size=(n_sims, n_teams)).astype(np.int8)
    group = {
        "wins": rng.integers(0, GROUP_SIZE, size=(n_sims, n_teams)).astype(np.int32),
        "draws": rng.integers(0, GROUP_SIZE, size=(n_sims, n_teams)).astype(np.int32),
    }
    scheme = ScoringScheme(name, 2.0, 1.0, mix=mix)

    running = scoring.running_scores(stages, group, scheme, rosters)
    assert running.shape == (n_sims, n_drafters, N_STAGES)

    # Terminal boundary == one-shot pool scores on the two-layer team points.
    terminal = metrics.pool_scores(scoring.team_points(stages, group, scheme), rosters)
    assert np.allclose(running[..., int(Stage.CHAMPION)], terminal)

    # Cumulative: non-decreasing along the boundary axis (knockout credit only grows; the
    # group layer is constant).
    assert np.all(np.diff(running, axis=-1) >= -1e-9)


def test_running_scores_deterministic_fixture_full_profile():
    # Hand-built n_sims=1 case with KNOWN furthest stages and group tallies; assert the FULL
    # length-7 running vector for each drafter against a hand-computed profile (group layer
    # flat from b=0; knockout = ko_vec[min(stage, b)] summed over the roster). Uses the
    # "linear" ladder ko_vec = [0, 1, 2, 3, 4, 5, 6] so the interior boundaries are transparent.
    #
    # 4 teams: team0 -> QF (stage 3), team1 -> R16 (stage 2), team2 -> eliminated at GROUP
    # (stage 0), team3 -> R32 (stage 1). Two drafters of two teams each.
    stages = np.array([[Stage.QF, Stage.R16, Stage.GROUP, Stage.R32]], dtype=np.int8)
    group = {
        "wins": np.array([[2, 1, 0, 1]], dtype=np.int32),
        "draws": np.array([[1, 2, 0, 1]], dtype=np.int32),
    }
    rosters = np.array([[0, 2], [1, 3]])  # drafter0 = {QF, GROUP}; drafter1 = {R16, R32}
    w_pts, d_pts = 2.0, 1.0

    # Knockout credit at each boundary b = ko_vec[min(stage, b)], summed over the roster.
    # drafter0: ko_vec[min(3,b)] + ko_vec[min(0,b)] = [0, 1, 2, 3, 3, 3, 3]
    # drafter1: ko_vec[min(2,b)] + ko_vec[min(1,b)] = [0, 2, 3, 3, 3, 3, 3]
    ko_d0 = np.array([0.0, 1, 2, 3, 3, 3, 3])
    ko_d1 = np.array([0.0, 2, 3, 3, 3, 3, 3])

    # mix == 0: bare cumulative knockout sum, no group layer.
    sc0 = ScoringScheme("linear", w_pts, d_pts, mix=0.0)
    run0 = scoring.running_scores(stages, group, sc0, rosters)
    assert np.array_equal(run0[0, 0], ko_d0)
    assert np.array_equal(run0[0, 1], ko_d1)

    # mix == 0.3: blend with a FLAT group layer (constant over all 7 boundaries).
    # grp_team = 2*wins + 1*draws -> team0=5, team1=4, team2=0, team3=3.
    # grp_running: drafter0 = 5 + 0 = 5; drafter1 = 4 + 3 = 7.
    mix = 0.3
    exp_d0 = (1.0 - mix) * ko_d0 + mix * 5.0  # [1.5, 2.2, 2.9, 3.6, 3.6, 3.6, 3.6]
    exp_d1 = (1.0 - mix) * ko_d1 + mix * 7.0  # [2.1, 3.5, 4.2, 4.2, 4.2, 4.2, 4.2]
    sc3 = ScoringScheme("linear", w_pts, d_pts, mix=mix)
    run3 = scoring.running_scores(stages, group, sc3, rosters)
    assert np.allclose(run3[0, 0], exp_d0)
    assert np.allclose(run3[0, 1], exp_d1)
    # Spot-check the literal interior profile, independent of the formula above.
    assert np.allclose(run3[0, 0], [1.5, 2.2, 2.9, 3.6, 3.6, 3.6, 3.6])
    assert np.allclose(run3[0, 1], [2.1, 3.5, 4.2, 4.2, 4.2, 4.2, 4.2])


@pytest.mark.parametrize("name", list(ladders.LADDERS))
def test_running_scores_mix_zero_matches_pool_scores_at_every_boundary(name):
    # mix == 0 ALL-BOUNDARY identity: for every boundary b, running_scores[..., b] equals the
    # bare cumulative knockout pool score pool_scores(ladder[min(stages, b)], rosters). This
    # locks the interior boundaries, not just CHAMPION.
    rng = np.random.default_rng(4)
    n_sims, n_drafters, per = 30, 6, 5
    n_teams = n_drafters * per
    rosters = rng.permutation(n_teams).reshape(n_drafters, per)
    stages = rng.integers(0, N_STAGES, size=(n_sims, n_teams)).astype(np.int8)
    group = {
        "wins": rng.integers(0, GROUP_SIZE, size=(n_sims, n_teams)).astype(np.int32),
        "draws": rng.integers(0, GROUP_SIZE, size=(n_sims, n_teams)).astype(np.int32),
    }
    ladder = get_ladder(name)
    scheme = ScoringScheme(name, 2.0, 1.0, mix=0.0)
    running = scoring.running_scores(stages, group, scheme, rosters)
    for b in range(N_STAGES):
        expected_b = metrics.pool_scores(ladder[np.minimum(stages, b)], rosters)
        assert np.array_equal(running[..., b], expected_b)


# --- gamma <-> mix solver (plan section 4) ----------------------------------------------

# Representative non-degenerate purses for the solver round-trip / monotonicity checks. The
# exact values are immaterial (the identities hold for any positive G, K); these mirror the
# scale of the real-Elo calibration (group purse ~ tens-hundreds, knockout purse ~ tens).
_SOLVER_GK = [(58.0, 60.0), (120.0, 32.0), (29.0, 100.0), (1.0, 1.0)]


def test_gamma_of_mix_zero_and_one_endpoints():
    # gamma(0) == 0 exactly (group layer absent) for any purses; gamma(1) == 1 when G > 0.
    for G, K in _SOLVER_GK:
        assert scoring.gamma_of_mix(0.0, G, K) == 0.0
        assert scoring.gamma_of_mix(1.0, G, K) == pytest.approx(1.0)


def test_gamma_of_mix_is_zero_at_mix_zero_for_any_purses_including_both_zero():
    # The mix==0 early-return makes the "0.0 at mix==0 for ANY purses" docstring contract literally
    # true and symmetric with solve_mix_for_gamma(0.0, ...) == 0.0. Crucially the degenerate
    # both-purses-zero case returns 0.0 (the group layer is absent), NOT nan (the old den==0 path).
    assert scoring.gamma_of_mix(0.0, 0.0, 0.0) == 0.0
    assert scoring.gamma_of_mix(0.0, 0.0, 50.0) == 0.0
    assert scoring.gamma_of_mix(0.0, 50.0, 0.0) == 0.0
    # mix>0 with both purses zero remains nan (genuinely undefined: no group OR knockout points).
    assert math.isnan(scoring.gamma_of_mix(0.5, 0.0, 0.0))


def test_solve_mix_for_gamma_is_exact_inverse_round_trip_to_fp():
    # solve_mix_for_gamma is the exact closed-form inverse of gamma_of_mix: round-tripping a target
    # gamma through mix and back reproduces it to floating point, and mapping a mix -> gamma -> mix
    # likewise returns the original mix.
    for G, K in _SOLVER_GK:
        for gamma in np.linspace(0.0, 0.95, 20):
            mix = scoring.solve_mix_for_gamma(float(gamma), G, K)
            assert 0.0 <= mix <= 1.0
            assert scoring.gamma_of_mix(mix, G, K) == pytest.approx(float(gamma), abs=1e-12)
        for mix in np.linspace(0.0, 1.0, 21):
            gamma = scoring.gamma_of_mix(float(mix), G, K)
            assert scoring.solve_mix_for_gamma(gamma, G, K) == pytest.approx(float(mix), abs=1e-12)


def test_gamma_of_mix_strictly_monotone_increasing_in_mix():
    # For positive purses gamma is strictly increasing in mix on [0, 1].
    for G, K in _SOLVER_GK:
        gammas = [scoring.gamma_of_mix(float(m), G, K) for m in np.linspace(0.0, 1.0, 50)]
        assert np.all(np.diff(gammas) > 0.0)


def test_solve_mix_for_gamma_zero_is_exactly_zero():
    # gamma == 0 must return mix == 0 EXACTLY (not ~1e-17), so the gamma=0 anchor short-circuits to
    # the bit-identical bare-ladder path.
    for G, K in _SOLVER_GK:
        assert scoring.solve_mix_for_gamma(0.0, G, K) == 0.0


def test_solve_mix_for_gamma_rejects_out_of_range_and_degenerate():
    with pytest.raises(ValueError, match="target_gamma"):
        scoring.solve_mix_for_gamma(-0.01, 50.0, 50.0)
    with pytest.raises(ValueError, match="target_gamma"):
        scoring.solve_mix_for_gamma(1.01, 50.0, 50.0)
    # Positive gamma with both purses 0 is unsolvable.
    with pytest.raises(ValueError, match="purses are 0"):
        scoring.solve_mix_for_gamma(0.3, 0.0, 0.0)


def test_gamma_match_mix_closed_form_and_commensurability():
    # mix* = 1 / (3 W + 1) and at mix* a 3-win team's group points equal A(R32) * (1 - mix*).
    a_r32 = float(get_ladder("linear")[int(Stage.R32)])
    assert a_r32 == 1.0
    for _name, (w_pts, _d_pts) in GROUP_POINT_SCHEMES.items():
        mix_star = scoring.gamma_match_mix(w_pts)
        assert mix_star == pytest.approx(1.0 / (3.0 * w_pts + 1.0))
        # Commensurability: winning all three group games == reaching the Round of 32.
        group_points_3wins = mix_star * (3.0 * w_pts)
        r32_points = (1.0 - mix_star) * a_r32
        assert group_points_3wins == pytest.approx(r32_points)
    with pytest.raises(ValueError, match="w_pts"):
        scoring.gamma_match_mix(0.0)


# --- gamma calibration + scheme-grid construction (plan section 4) -----------------------


@pytest.fixture(scope="module")
def _calib():
    # Small but real calibration on the Elo field (fast; the purses are very stable, so a reduced
    # n_sims still pins them well enough for the structural assertions below).
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field, name="elo_2026")
    return simulate.calibrate_group_knockout(
        cfg, seed=20260609, regime="resampled", n_sims=1200, n_draws=4
    )


def test_calibration_purses_are_in_physical_range(_calib):
    # E[sum wins] == #decisive matches, E[sum draws] == 2 * #drawn, summed over 72 group matches.
    # Decisive + drawn == 72, so E[sum wins] + E[sum draws]/2 == 72 exactly (a conservation check).
    assert _calib.e_sum_wins + _calib.e_sum_draws / 2.0 == pytest.approx(72.0, abs=1e-6)
    assert 0.0 < _calib.e_sum_wins < 72.0
    assert 0.0 < _calib.e_sum_draws < 72.0
    # Knockout purse ordering: triangular (champ 21), geometric (champ 32) dwarf linear (champ 6).
    assert _calib.e_sum_A["triangular"] > _calib.e_sum_A["linear"]
    assert _calib.e_sum_A["geometric"] > _calib.e_sum_A["linear"]
    # group_purse / knockout_purse helpers reproduce the closed form.
    assert _calib.group_purse(3.0, 1.0) == pytest.approx(
        3.0 * _calib.e_sum_wins + 1.0 * _calib.e_sum_draws
    )
    assert _calib.knockout_purse("geometric") == _calib.e_sum_A["geometric"]


def test_calibrate_group_knockout_rejects_nonpositive_budget():
    # Defensive guards: a non-positive n_sims or n_draws is a caller error (it would otherwise
    # divide by zero / loop zero times and silently return meaningless purses), so reject it.
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field, name="elo_2026")
    with pytest.raises(ValueError, match="n_sims must be positive"):
        simulate.calibrate_group_knockout(cfg, seed=0, n_sims=0)
    with pytest.raises(ValueError, match="n_sims must be positive"):
        simulate.calibrate_group_knockout(cfg, seed=0, n_sims=-10)
    with pytest.raises(ValueError, match="n_draws must be positive"):
        simulate.calibrate_group_knockout(cfg, seed=0, n_sims=100, n_draws=0)


def test_build_gamma_schemes_grid_structure_and_realised_gamma(_calib):
    gamma_grid = [0.0, 0.05, 0.12, 0.25, 0.50]
    cells = simulate.build_gamma_schemes(_calib, gamma_grid)

    # gamma == 0 collapses the 3x3 (D/W x shape) anchor slice to ONE cell per shape (mix=0 base).
    anchors = [c for c in cells if c.scheme.mix == 0.0]
    assert len(anchors) == len(LADDERS)
    for c in anchors:
        assert c.scheme.w_pts == 1.0 and c.scheme.d_pts == 0.0  # pure terminal ladder identity
        assert c.gamma_realised == 0.0
        assert c.gamma_target == 0.0
        assert not c.is_gamma_match

    # Distinct-cell count: 3 anchors + 3 shapes x 3 D/W x (4 positive levels + 1 gamma_match) = 48.
    n_pos_levels = len([g for g in gamma_grid if g != 0.0])
    expected = len(LADDERS) + len(LADDERS) * len(GROUP_POINT_SCHEMES) * (n_pos_levels + 1)
    assert len(cells) == expected == 48
    # No duplicate (shape, w, d, mix) identities survive.
    idents = {
        (c.scheme.knockout_ladder, c.scheme.w_pts, c.scheme.d_pts, c.scheme.mix) for c in cells
    }
    assert len(idents) == len(cells)

    # Realised gamma reproduces the requested target to FP for every positive grid level.
    for c in cells:
        if c.gamma_target >= 0.0 and not c.is_gamma_match:
            assert c.gamma_realised == pytest.approx(c.gamma_target, abs=1e-12)

    # gamma_match: exactly one per (shape, D/W), mix* == 1/(3W+1), 0 < gamma_match < 1.
    matches = [c for c in cells if c.is_gamma_match]
    assert len(matches) == len(LADDERS) * len(GROUP_POINT_SCHEMES)
    for c in matches:
        assert c.scheme.mix == pytest.approx(1.0 / (3.0 * c.scheme.w_pts + 1.0))
        assert 0.0 < c.gamma_realised < 1.0
        assert c.gamma_target == -1.0


def test_build_gamma_schemes_without_zero_anchor_has_no_collapsed_cell(_calib):
    # If gamma=0 is absent the grid carries no collapsed mix=0 anchor (every cell has mix > 0).
    cells = simulate.build_gamma_schemes(_calib, [0.05, 0.5])
    assert all(c.scheme.mix > 0.0 for c in cells)


def test_build_gamma_schemes_rejects_negative_gamma_level(_calib):
    # A negative requested level must be rejected: it would alias the -1.0 gamma_target reserved for
    # the per-cell gamma_match landmark, so the sentinel cannot be impersonated by a grid value.
    with pytest.raises(ValueError, match="gamma_match sentinel"):
        simulate.build_gamma_schemes(_calib, [0.0, -1.0, 0.5])
    with pytest.raises(ValueError, match="gamma_match sentinel"):
        simulate.build_gamma_schemes(_calib, [-0.01])


def test_gamma_grid_runs_end_to_end_with_bounded_metrics(_calib):
    # Integration (plan section 10.6): run a couple of grid schemes through run_strength_config at a
    # tiny budget, 8x6 full field; assert bounded metrics, 0 <= group_share <= 1, champion always
    # owned, and that a mix=0 anchor cell carries the bare-ladder identity columns.
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field, name="elo_2026")
    cells = simulate.build_gamma_schemes(_calib, [0.0, 0.25])
    # Keep the run small: one shape's anchor + its three 0.25 D/W cells.
    subset = [
        c for c in cells if c.scheme.knockout_ladder == "triangular" and not c.is_gamma_match
    ]
    recs = simulate.run_strength_config(
        cfg,
        schemes=[c.scheme for c in subset],
        n_values=[6],
        policies=["ev_greedy"],
        n_draws=2,
        sims_per_draw=300,
        seed=20260609,
        regime="resampled",
        n_drafters=8,
        cfg_id=0,
    )
    assert len(recs) == len(subset)
    for r in recs:
        assert -1.0 <= r["spearman_mean"] <= 1.0
        assert 0.0 <= r["skill_variance_share"] <= 1.0
        assert 0.0 <= r["top_tie_rate"] <= 1.0
        assert r["champion_undrafted_rate"] == 0.0  # full 8x6 field
        gvs = r["group_variance_share"]
        assert np.isnan(gvs) or 0.0 <= gvs <= 1.0
        assert 0.0 <= r["alive_fraction"] <= 1.0
        assert 0.0 <= r["pool_suspense_group_share"] <= 1.0
    # The mix=0 anchor keys to the bare ladder; its group-share is exactly 0 (constant GROUP layer).
    anchor = next(r for r in recs if r["mix"] == 0.0)
    assert anchor["knockout_ladder"] == "triangular"
    assert anchor["w_pts"] == 1.0 and anchor["d_pts"] == 0.0
    assert anchor["group_variance_share"] == 0.0
    assert anchor["pool_suspense_group_share"] == 0.0


def test_paired_delta_skill_matches_manual_paired_statistics():
    # The paired-delta kernel (audit item 1): delta = mean(cell - anchor) formed PER DRAW, and a
    # paired cluster SE = std(per-draw delta, ddof=1) / sqrt(n). Cross-check against an independent
    # hand computation on a small per-draw series, and confirm it is NOT the unpaired difference of
    # means' SE (the pairing cancels the shared between-draw component, which is the whole point).
    rng = np.random.default_rng(0)
    n = 25
    bracket = rng.normal(size=n)  # large shared between-draw component (cancels under pairing)
    anchor = 0.30 + bracket + 0.01 * rng.normal(size=n)
    cell = 0.27 + bracket + 0.01 * rng.normal(size=n)  # same brackets, small idiosyncratic noise

    delta, se = simulate.paired_delta_skill(cell, anchor)
    per_draw = cell - anchor
    assert delta == pytest.approx(float(np.mean(per_draw)))
    assert se == pytest.approx(float(np.std(per_draw, ddof=1) / np.sqrt(n)))
    # Pairing slashes the SE: the unpaired (independent-sample) SE of the difference of means is far
    # larger because it does not net out the shared bracket term.
    unpaired_se = np.sqrt(np.var(cell, ddof=1) / n + np.var(anchor, ddof=1) / n)
    assert se < 0.2 * unpaired_se


def test_paired_delta_skill_anchor_against_itself_is_exact_zero():
    # An anchor paired with itself yields identically-zero per-draw deltas -> (0.0, 0.0) EXACTLY,
    # which is what gives the gamma=0 anchor rows delta 0 and paired SE 0 in the emitted frontier.
    series = np.array([0.31, 0.29, 0.33, 0.30, 0.28])
    delta, se = simulate.paired_delta_skill(series, series)
    assert delta == 0.0
    assert se == 0.0


def test_paired_delta_skill_length_mismatch_and_short_series():
    # A length mismatch is a caller error (unpairable draws) -> ValueError.
    with pytest.raises(ValueError, match="equal-length"):
        simulate.paired_delta_skill([0.1, 0.2, 0.3], [0.1, 0.2])
    # Fewer than two paired draws -> SE is NaN (cluster SE undefined); the point delta is defined.
    delta, se = simulate.paired_delta_skill([0.25], [0.30])
    assert delta == pytest.approx(-0.05)
    assert math.isnan(se)


def test_run_strength_config_rejects_nonpositive_suspense_subsample(_calib):
    # Defensive guard: a non-positive suspense_subsample is a caller error; reject it rather than
    # silently treating it as "no cap" (which would run the full, slow surrogate). None disables the
    # cap and is the documented sentinel; 0 / negative raise. The guard fires before any simulation.
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field, name="elo_2026")
    base = dict(
        ladders=["linear"], n_values=[6], policies=["ev_greedy"], n_draws=1, sims_per_draw=50,
        seed=20260609, regime="resampled", n_drafters=8,
    )
    for bad in (0, -1, -512):
        with pytest.raises(ValueError, match="suspense_subsample must be positive"):
            simulate.run_strength_config(cfg, suspense_subsample=bad, **base)
    # None (no cap) and a positive cap are both accepted (smoke; just must not raise).
    simulate.run_strength_config(cfg, suspense_subsample=None, **base)
    simulate.run_strength_config(cfg, suspense_subsample=16, **base)
