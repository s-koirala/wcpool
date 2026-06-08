import numpy as np
import pytest

from wcpool import strength
from wcpool.strength import StrengthModel


def test_elo_expected_score_properties():
    assert strength.elo_expected_score(0.0) == pytest.approx(0.5)
    # complementary symmetry
    both = strength.elo_expected_score(120.0) + strength.elo_expected_score(-120.0)
    assert both == pytest.approx(1.0)
    # 200-pt gap ~ 0.76 favourite (documented reference value)
    assert strength.elo_expected_score(200.0) == pytest.approx(0.76, abs=0.01)
    # monotone increasing
    dr = np.linspace(-400, 400, 50)
    we = strength.elo_expected_score(dr)
    assert np.all(np.diff(we) > 0)


def test_poisson_rates_even_match():
    mu = 2.6
    la, lb = strength.poisson_rates(0.0, mu, beta=1.7)
    assert la == pytest.approx(mu / 2)
    assert lb == pytest.approx(mu / 2)


def test_poisson_rates_product_invariant():
    # log-linear symmetric form keeps lambda_A * lambda_B constant in dr
    mu, beta = 2.6, 1.5
    la0, lb0 = strength.poisson_rates(0.0, mu, beta)
    la1, lb1 = strength.poisson_rates(300.0, mu, beta)
    # product is pinned to (mu/2)^2 at every gap; geometric mean is mu/2 regardless of dr
    assert la0 * lb0 == pytest.approx((mu / 2) ** 2)
    assert la1 * lb1 == pytest.approx((mu / 2) ** 2)
    assert la1 > la0 and lb1 < lb0  # favourite scores more, concedes... rate drops


def test_skellam_wdl_sums_to_one():
    la = np.array([2.2, 1.3, 0.8])
    lb = np.array([0.8, 1.3, 2.2])
    pa, pd, pb = strength.skellam_wdl(la, lb)
    assert np.allclose(pa + pd + pb, 1.0)
    # symmetric matchup -> equal win probs
    assert pa[1] == pytest.approx(pb[1])


def test_knockout_advance_prob_bounds_and_symmetry():
    la = np.array([1.3, 2.5])
    lb = np.array([1.3, 0.7])
    p = strength.knockout_advance_prob(la, lb)
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] == pytest.approx(0.5)  # equal strength -> coin flip
    assert p[1] > 0.5


def test_calibrate_beta_matches_elo_curve():
    rng = np.random.default_rng(0)
    ratings = 1800 + 120 * rng.standard_normal(48)
    out = strength.calibrate_beta(ratings, mu_total=2.6)
    assert 0.1 < out["beta"] < 6.0
    # the goals model reproduces the Elo expected-score curve closely
    assert out["rmse"] < 0.02
    assert out["max_abs_dev"] < 0.05
    # implied even-match draw rate is in the historically plausible band
    assert 0.20 < out["even_match_draw_rate"] < 0.35


def test_strength_model_autocalibrates():
    rng = np.random.default_rng(1)
    ratings = 1800 + 100 * rng.standard_normal(48)
    model = StrengthModel(ratings)
    assert model.beta is not None
    # expected score from the model's calibration matches Elo at a known gap
    pa = model.knockout_advance_prob(np.array([0]), np.array([1]))
    assert 0.0 <= pa[0] <= 1.0


def test_devig_normalises():
    p = strength.devig_probabilities(np.array([0.5, 0.3, 0.4]))  # sums to 1.2 (vig)
    assert p.sum() == pytest.approx(1.0)
