"""Team-strength model: ratings in, match-outcome probabilities out.

Design
------
The *rating source* is the pluggable input the task calls for (World Football Elo,
bookmaker-implied title odds, or a synthetic favoritism generator). All sources emit a
length-``n_teams`` vector on the **Elo scale** so downstream code is source-agnostic.

The *outcome model* converting a rating difference into match results is fixed and
literature-grounded:

* **Win probability (Elo expected score).** The World Football Elo system defines the
  pre-match *expected score* (points fraction, a draw counting as 0.5) as a logistic in
  the rating difference ``dr``::

      W_e(dr) = 1 / (1 + 10**(-dr/400))

  (eloratings.net system description; Wikipedia "World Football Elo Ratings"). This is
  the quantity the goals model below is calibrated against.

* **Goals (independent Poisson).** Group-stage results need win/draw/loss *and* goal
  difference / goals-for to break third-place ties, so we model goals, not just the
  win/draw/loss class. Each team's goals are independent Poisson with a log-rate linear in
  the rating difference::

      log lambda_A = log(mu_total/2) + (beta/2) * (dr/400)
      log lambda_B = log(mu_total/2) - (beta/2) * (dr/400)

  At ``dr=0`` both rates equal ``mu_total/2``, so the expected total is ``mu_total`` for
  even matches. ``mu_total`` is pinned to the historical World-Cup goals-per-match rate
  (not a free constant); ``beta`` (the supremacy slope) is **calibrated** so the model's
  implied expected score ``P(win)+0.5*P(draw)`` reproduces the Elo logistic ``W_e(dr)``
  over the realised distribution of rating gaps. No supremacy constant is hand-set.

  The *independent*-Poisson goals model is Maher (1982). Dixon & Coles (1997) is the
  canonical reference for the supremacy/attack-defence Poisson framework and additionally
  introduced a low-score *dependence* correction (a tau adjustment for 0-0/1-0/0-1/1-1)
  that we deliberately do NOT implement — this code uses pure independent Poisson.

  Maher, M.J. (1982), "Modelling association football scores", Statistica Neerlandica
  36(3):109-118. https://doi.org/10.1111/j.1467-9574.1982.tb00782.x
  Dixon, M.J. & Coles, S.G. (1997), "Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market", J. R. Statist. Soc. C 46(2):265-280.
  https://doi.org/10.1111/1467-9876.00065

The win/draw/loss probabilities of two independent Poissons are read off the **Skellam**
distribution of the goal difference ``D = g_A - g_B``: ``P(draw)=P(D=0)``,
``P(A win)=P(D>0)``, ``P(B win)=P(D<0)``. Knockout matches admit no draw, so a tie is
resolved (extra-time / shootout proxy) in proportion to win strength:
``P(A advances) = P(A win) + P(draw) * P(A win)/(P(A win)+P(B win))``.

  Skellam, J.G. (1946), "The frequency distribution of the difference between two Poisson
  variates belonging to different populations", J. R. Statist. Soc. A 109(3):296.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

# --- Constants pinned to data (not free parameters) -------------------------------------

ELO_DIVISOR = 400.0  # definitional constant of the Elo logistic (base-10, 400-pt scale)

# Historical men's World Cup goals per match, used to pin ``mu_total``. Each entry is
# (total goals, matches) for the tournament; the default mu_total is their pooled mean.
# Sources: official FIFA tournament statistics / Wikipedia tournament summary pages.
WC_GOALS_PER_MATCH_HISTORY = {
    "2014": (171, 64),
    "2018": (169, 64),
    "2022": (172, 64),
}


def default_mu_total() -> float:
    """Pooled goals-per-match over recent World Cups (see ``WC_GOALS_PER_MATCH_HISTORY``)."""
    goals = sum(g for g, _ in WC_GOALS_PER_MATCH_HISTORY.values())
    matches = sum(m for _, m in WC_GOALS_PER_MATCH_HISTORY.values())
    return goals / matches


def elo_expected_score(dr: np.ndarray | float) -> np.ndarray | float:
    """World Football Elo expected score (draw = 0.5) for rating difference ``dr``."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(dr, dtype=float) / ELO_DIVISOR))


def poisson_rates(
    dr: np.ndarray | float, mu_total: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent-Poisson goal rates ``(lambda_A, lambda_B)`` for rating gap ``dr``."""
    x = np.asarray(dr, dtype=float) / ELO_DIVISOR
    half = mu_total / 2.0
    lam_a = half * np.exp(beta * x / 2.0)
    lam_b = half * np.exp(-beta * x / 2.0)
    return lam_a, lam_b


def skellam_wdl(lam_a: np.ndarray, lam_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(P(A win), P(draw), P(B win))`` for independent Poisson goal rates.

    Uses the Skellam law of ``D = g_A - g_B``: draw = pmf(0); A-win = P(D>=1) = sf(0);
    B-win = P(D<=-1) = cdf(-1). The three are mutually exclusive and sum to 1.
    """
    lam_a = np.asarray(lam_a, dtype=float)
    lam_b = np.asarray(lam_b, dtype=float)
    p_draw = stats.skellam.pmf(0, lam_a, lam_b)
    p_a = stats.skellam.sf(0, lam_a, lam_b)  # P(D >= 1)
    p_b = stats.skellam.cdf(-1, lam_a, lam_b)  # P(D <= -1)
    return p_a, p_draw, p_b


def knockout_advance_prob(lam_a: np.ndarray, lam_b: np.ndarray) -> np.ndarray:
    """P(A advances) in a no-draw knockout: win outright, or win the strength-weighted tie."""
    p_a, p_draw, p_b = skellam_wdl(lam_a, lam_b)
    denom = p_a + p_b
    # When both rates are equal the tie split is 0.5; guard the 0/0 limit.
    share = np.where(denom > 0, p_a / np.where(denom > 0, denom, 1.0), 0.5)
    return p_a + p_draw * share


def calibrate_beta(
    ratings: np.ndarray,
    mu_total: float,
    beta_bounds: tuple[float, float] = (0.1, 6.0),
) -> dict:
    """Fit the *neutral* supremacy slope ``beta`` so the goals model matches the Elo curve.

    Target ``W_e(dr)`` is evaluated on every realised neutral pairwise rating gap (so the
    fit is weighted by the matchups that actually occur for this field). ``beta`` is the
    neutral supremacy slope; home advantage, if enabled, simply shifts the effective gap
    ``dr`` at simulation time (the standard Elo treatment) and is deliberately excluded
    here so it does not get folded into the slope. Returns the fitted beta plus
    diagnostics: RMSE/max-abs deviation from the Elo curve and the implied even-match draw
    rate (sanity-checked against historical World-Cup draw frequencies).
    """
    ratings = np.asarray(ratings, dtype=float)
    diffs = ratings[:, None] - ratings[None, :]
    dr = diffs[~np.eye(len(ratings), dtype=bool)]  # all ordered i!=j gaps
    target = elo_expected_score(dr)

    def loss(beta: float) -> float:
        lam_a, lam_b = poisson_rates(dr, mu_total, beta)
        p_a, p_draw, _ = skellam_wdl(lam_a, lam_b)
        model_score = p_a + 0.5 * p_draw
        return float(np.mean((model_score - target) ** 2))

    res = optimize.minimize_scalar(loss, bounds=beta_bounds, method="bounded")
    beta = float(res.x)

    lam_a, lam_b = poisson_rates(dr, mu_total, beta)
    p_a, p_draw, _ = skellam_wdl(lam_a, lam_b)
    model_score = p_a + 0.5 * p_draw
    even_lam = mu_total / 2.0
    return {
        "beta": beta,
        "rmse": float(np.sqrt(np.mean((model_score - target) ** 2))),
        "max_abs_dev": float(np.max(np.abs(model_score - target))),
        "even_match_draw_rate": float(stats.skellam.pmf(0, even_lam, even_lam)),
    }


@dataclass
class StrengthModel:
    """Bundles a rating vector with its calibrated outcome model.

    ``home_idx``/``home_advantage`` add Elo points to designated host teams in every match
    (default off: a neutral-venue field, which is the cleaner basis for the design study).
    """

    ratings: np.ndarray
    mu_total: float = field(default_factory=default_mu_total)
    beta: float | None = None
    home_advantage: float = 0.0
    home_idx: np.ndarray | None = None
    calibration: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ratings = np.asarray(self.ratings, dtype=float)
        if self.beta is None:
            self.calibration = calibrate_beta(self.ratings, self.mu_total)
            self.beta = self.calibration["beta"]

    @property
    def n_teams(self) -> int:
        return len(self.ratings)

    def _adj_rating(self, idx: np.ndarray) -> np.ndarray:
        r = self.ratings[idx]
        if self.home_advantage and self.home_idx is not None:
            is_home = np.isin(idx, self.home_idx)
            r = r + is_home * self.home_advantage
        return r

    def goal_rates(self, idx_a: np.ndarray, idx_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised goal rates for teams ``idx_a`` vs ``idx_b`` (same-shape index arrays)."""
        dr = self._adj_rating(np.asarray(idx_a)) - self._adj_rating(np.asarray(idx_b))
        return poisson_rates(dr, self.mu_total, self.beta)

    def knockout_advance_prob(self, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
        lam_a, lam_b = self.goal_rates(idx_a, idx_b)
        return knockout_advance_prob(lam_a, lam_b)

    def expected_score(self, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
        """Elo expected score (draw=0.5) — used for calibration diagnostics, not sampling."""
        dr = self._adj_rating(np.asarray(idx_a)) - self._adj_rating(np.asarray(idx_b))
        return elo_expected_score(dr)


# --- Rating sources (the pluggable input) -----------------------------------------------


def synthetic_ratings(
    n_teams: int,
    spread: float,
    rng: np.random.Generator,
    mean_rating: float = 1800.0,
) -> np.ndarray:
    """Draw ``n_teams`` i.i.d. Gaussian Elo ratings with standard deviation ``spread``.

    ``spread`` is the single favoritism knob: small spread -> near-parity (luck-dominated);
    large spread -> a few dominant teams hold most of the advancement mass. The realised
    top-k EV concentration that this induces is measured downstream (it is the *outcome*
    of ``spread``, not a separately tuned quantity). ``mean_rating`` only sets the scale
    origin and cancels in every rating difference, so its value is immaterial.
    """
    return mean_rating + spread * rng.standard_normal(n_teams)


def devig_probabilities(implied: np.ndarray) -> np.ndarray:
    """Normalise raw implied probabilities (e.g. from decimal odds) to sum to 1 (remove vig)."""
    implied = np.asarray(implied, dtype=float)
    if np.any(implied <= 0):
        raise ValueError("implied probabilities must be positive")
    return implied / implied.sum()


def fit_ratings_to_title_probs(
    target_probs: np.ndarray,
    champion_prob_fn,
    spread_seed: float,
    mean_rating: float = 1800.0,
    temp_bounds: tuple[float, float] = (0.2, 5.0),
) -> dict:
    """Calibrate an Elo rating vector to bookmaker-implied **title** probabilities.

    Ranks teams by ``target_probs``, lays them on a Gaussian rating template, and fits a
    single *temperature* scaling the spread so the model's *simulated* champion
    distribution best matches ``target_probs`` (squared error). ``champion_prob_fn(ratings)``
    must return per-team simulated champion probabilities. The temperature is fitted (one
    scalar), not hand-set; the calibration error is reported for verification.

    Returns ``{"ratings", "temperature", "rmse", "achieved_probs"}``.
    """
    target_probs = devig_probabilities(target_probs)
    n = len(target_probs)
    # Rank-based Gaussian template: strongest implied prob -> highest quantile.
    order = np.argsort(np.argsort(target_probs))  # 0..n-1 ascending by implied prob
    quantiles = (order + 0.5) / n
    template = mean_rating + spread_seed * stats.norm.ppf(quantiles)

    def loss(temp: float) -> float:
        ratings = mean_rating + temp * (template - mean_rating)
        achieved = champion_prob_fn(ratings)
        return float(np.sum((achieved - target_probs) ** 2))

    res = optimize.minimize_scalar(loss, bounds=temp_bounds, method="bounded")
    temp = float(res.x)
    ratings = mean_rating + temp * (template - mean_rating)
    achieved = champion_prob_fn(ratings)
    return {
        "ratings": ratings,
        "temperature": temp,
        "rmse": float(np.sqrt(np.mean((achieved - target_probs) ** 2))),
        "achieved_probs": achieved,
    }
