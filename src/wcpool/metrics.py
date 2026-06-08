"""Pool-scoring helpers and the five evaluation-metric families.

All functions operate on plain arrays so they are independent of how the draft was run:

* ``points`` : (n_sims, n_teams) terminal points each team earned under a ladder.
* ``rosters``: (n_drafters, teams_per_drafter) global team indices each drafter holds.
* ``drafter_of_team``: (n_teams,) owning drafter index, or -1 if undrafted.

The metric families correspond 1:1 to the task's list:

1. skill vs luck  -> ``spearman_ev_vs_placing`` + ``skill_variance_share``
2. slot equity    -> ``slot_win_probability`` + ``slot_equity_imbalance``
3. ties at top    -> ``top_tie_rate`` + ``winning_score_summary``
4. champion dom.  -> ``champion_dominance``
5. robustness     -> assembled across favoritism levels by the experiment runner.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def pool_scores(points: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Per-drafter pool score each replicate: (n_sims, n_drafters)."""
    return points[:, rosters].sum(axis=2)


def win_probability(scores: np.ndarray) -> np.ndarray:
    """Win probability per drafter, splitting a tie at the top equally among those tied."""
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    credit = is_top / is_top.sum(axis=1, keepdims=True)
    return credit.mean(axis=0)


def top_tie_rate(scores: np.ndarray) -> float:
    """Fraction of replicates in which two or more drafters share the top score."""
    top = scores.max(axis=1, keepdims=True)
    n_tied = (scores == top).sum(axis=1)
    return float(np.mean(n_tied > 1))


def winning_score_summary(scores: np.ndarray) -> dict[str, float]:
    """Distribution summary of the *winning* (maximum) score across replicates."""
    winning = scores.max(axis=1)
    return {
        "mean": float(np.mean(winning)),
        "sd": float(np.std(winning)),
        "p05": float(np.percentile(winning, 5)),
        "median": float(np.percentile(winning, 50)),
        "p95": float(np.percentile(winning, 95)),
    }


def roster_ev(team_ev: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Pre-draft expected pool score per drafter: (n_drafters,)."""
    return team_ev[rosters].sum(axis=1)


def spearman_per_sim(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> np.ndarray:
    """Per-replicate Spearman rho between roster-EV rank and realised-score rank.

    With only n_drafters items, Spearman rho is Pearson on the rank vectors. Replicates in
    which every drafter scores identically (undefined ordering) yield NaN.
    """
    ev_per_drafter = roster_ev(team_ev, rosters)
    ev_ranks = rankdata(ev_per_drafter)  # (D,)
    score_ranks = rankdata(scores, axis=1)  # (S, D), average method

    b = ev_ranks - ev_ranks.mean()
    a = score_ranks - score_ranks.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman_ev_vs_placing(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> dict:
    """Spearman correlation between pre-draft roster-EV rank and realised placing.

    Computed per replicate, then summarised over replicates. Replicates in which every
    drafter scores identically (undefined ordering) are dropped.
    """
    rho = spearman_per_sim(scores, team_ev, rosters)
    valid = rho[~np.isnan(rho)]
    return {
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "sd": float(np.std(valid)) if valid.size else float("nan"),
        "undefined_frac": float(np.mean(np.isnan(rho))),
    }


def skill_variance_share(scores: np.ndarray, team_ev: np.ndarray, rosters: np.ndarray) -> dict:
    """Share of pool-score variance attributable to roster composition (skill) vs noise.

    Variance decomposition over the (drafter, replicate) score panel::

        sigma2_between = Var over drafters of their expected score  (roster differences)
        sigma2_within  = mean over drafters of Var over replicates  (tournament noise)
        skill_share    = sigma2_between / (sigma2_between + sigma2_within)

    The between-component uses the *independent* roster EV (estimated on a separate batch,
    passed in as ``team_ev``) so it is not inflated by in-sample estimation noise; the
    within-component uses the realised eval-batch ``scores``. This is an
    intraclass-correlation-style index: 1 = placing is fully determined by the draft,
    0 = placing is pure simulation luck.
    """
    sigma2_between = float(np.var(roster_ev(team_ev, rosters)))
    sigma2_within = float(np.mean(np.var(scores, axis=0)))
    total = sigma2_between + sigma2_within
    return {
        "sigma2_between": sigma2_between,
        "sigma2_within": sigma2_within,
        "skill_share": sigma2_between / total if total > 0 else float("nan"),
    }


def slot_win_probability(scores: np.ndarray) -> np.ndarray:
    """Win probability by snake slot (drafter index == pick slot): (n_drafters,)."""
    return win_probability(scores)


def slot_equity_imbalance(scores: np.ndarray) -> dict:
    """How far slot win-probabilities depart from the equitable 1/n_drafters.

    ``max_minus_min`` is the spread; ``chi2_uniform`` is the chi-square statistic against a
    uniform allocation (0 = perfectly balanced). Both are 0 iff every slot wins equally.
    """
    wp = win_probability(scores)
    n = len(wp)
    uniform = 1.0 / n
    return {
        "win_prob_by_slot": wp.tolist(),
        "max_minus_min": float(wp.max() - wp.min()),
        "chi2_uniform": float(np.sum((wp - uniform) ** 2 / uniform)),
    }


def champion_dominance(
    scores: np.ndarray, champion_team: np.ndarray, drafter_of_team: np.ndarray
) -> dict:
    """P(the drafter holding the eventual champion wins the pool).

    Conditioned on the champion being drafted (with N<8 teams/drafter some teams are
    undrafted, so the champion may be held by nobody); the undrafted rate is reported.
    A tie at the top is credited fractionally to the champion's holder.
    """
    holder = drafter_of_team[champion_team]  # (n_sims,), -1 if undrafted
    drafted = holder >= 0
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    n_tied = is_top.sum(axis=1)
    rows = np.arange(scores.shape[0])
    holder_is_top = np.zeros(scores.shape[0], dtype=bool)
    holder_is_top[drafted] = is_top[rows[drafted], holder[drafted]]
    credit = np.where(holder_is_top, 1.0 / n_tied, 0.0)
    p_holder = float(np.mean(credit[drafted])) if drafted.any() else float("nan")
    return {
        "champion_undrafted_rate": float(np.mean(~drafted)),
        "p_champion_holder_wins": p_holder,
    }
