"""Orchestration: build a strength config, run the draws, and assemble cell metrics.

A *cell* is a ``(strength_config, ladder, N, policy)`` combination. The expensive object —
the simulated tournament — is **independent of the ladder, N, and policy**, so each draw's
tournament batch is simulated once and reused across the whole ladder x N x policy grid:
the ladder only re-weights the per-team stage outcomes, and the draft only re-partitions
teams among drafters.

Two independent batches are drawn per group draw: an *EV batch* (estimates each team's
expected/variance points used by the drafters) and an *eval batch* (the held-out
tournaments the rosters are actually scored on). Using independent batches keeps the
skill-vs-luck estimates free of in-sample optimism.

Primary regime: ``resampled`` draws — each replicate redraws groups (pot-constrained) and
re-drafts, marginalising single-draw luck so conclusions describe the *design*, not one
bracket. ``fixed`` uses the official draw throughout (a sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from . import draft as draft_mod
from . import metrics as M
from .ladders import LADDERS, Stage
from .strength import StrengthModel, synthetic_ratings
from .tournament import N_TEAMS, random_pot_draw, simulate_tournament

N_DRAFTERS = draft_mod.N_DRAFTERS


def champion_team(stages: np.ndarray) -> np.ndarray:
    """Global index of the champion (stage == CHAMPION) in each replicate: (n_sims,)."""
    return (stages == int(Stage.CHAMPION)).argmax(axis=1)


def team_title_prob(stages: np.ndarray) -> np.ndarray:
    """Per-team probability of being champion, estimated over the batch: (n_teams,)."""
    return (stages == int(Stage.CHAMPION)).mean(axis=0)


def topk_ev_share(title_prob: np.ndarray, k: int) -> float:
    """Share of total title probability held by the top-k teams (concentration diagnostic)."""
    return float(np.sort(title_prob)[::-1][:k].sum())


def rank_pots(ratings: np.ndarray) -> np.ndarray:
    """Form (4, 12) draw pots by rating rank (strongest 12 -> pot 0, etc.)."""
    order = np.argsort(ratings)[::-1]  # strongest first
    return order.reshape(4, 12)


@dataclass
class StrengthConfig:
    """A named strength scenario: a rating vector plus the pots used for random draws."""

    name: str
    model: StrengthModel
    pots: np.ndarray
    fixed_groups: np.ndarray
    top8_title_share: float = float("nan")
    diagnostics: dict = dc_field(default_factory=dict)


def make_elo_config(field, name: str = "elo_2026") -> StrengthConfig:
    model = StrengthModel(field.elo.copy())
    return StrengthConfig(
        name=name,
        model=model,
        pots=field.pots,
        fixed_groups=field.fixed_groups,
        diagnostics={"calibration": model.calibration, "elo_spread": field.elo_spread},
    )


def make_synthetic_config(spread: float, name: str, seed: int) -> StrengthConfig:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(7, int(spread))))
    ratings = synthetic_ratings(N_TEAMS, spread=spread, rng=rng)
    model = StrengthModel(ratings)
    pots = rank_pots(ratings)
    fixed_groups = pots.T.copy()
    return StrengthConfig(
        name=name,
        model=model,
        pots=pots,
        fixed_groups=fixed_groups,
        diagnostics={"calibration": model.calibration, "spread": spread},
    )


def _draw_groups(cfg: StrengthConfig, regime: str, rng: np.random.Generator) -> np.ndarray:
    if regime == "fixed":
        return cfg.fixed_groups
    if regime == "resampled":
        return random_pot_draw(cfg.pots, rng)
    raise ValueError(f"unknown regime {regime!r}")


@dataclass
class CellAccumulator:
    scores: list = dc_field(default_factory=list)
    rho: list = dc_field(default_factory=list)
    skill_share: list = dc_field(default_factory=list)
    sigma_between: list = dc_field(default_factory=list)
    sigma_within: list = dc_field(default_factory=list)
    champ_credit: list = dc_field(default_factory=list)
    champ_drafted: list = dc_field(default_factory=list)


def _run_draft(policy: str, n_rounds: int, team_ev, team_var, points_eval, br_slot: int):
    if policy == "ev_greedy":
        return draft_mod.draft_ev_greedy(team_ev, N_DRAFTERS, n_rounds)
    if policy == "variance":
        return draft_mod.draft_variance(team_var, N_DRAFTERS, n_rounds)
    if policy == "best_response":
        return draft_mod.draft_best_response(points_eval, team_ev, br_slot, N_DRAFTERS, n_rounds)
    raise ValueError(f"unknown policy {policy!r}")


def _accumulate_champion(scores, champ, drafter_of_team):
    holder = drafter_of_team[champ]
    drafted = holder >= 0
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    n_tied = is_top.sum(axis=1)
    rows = np.arange(scores.shape[0])
    holder_is_top = np.zeros(scores.shape[0], dtype=bool)
    holder_is_top[drafted] = is_top[rows[drafted], holder[drafted]]
    credit = np.where(holder_is_top, 1.0 / n_tied, 0.0)
    return credit, drafted


def run_strength_config(
    cfg: StrengthConfig,
    ladders: list[str],
    n_values: list[int],
    policies: list[str],
    n_draws: int,
    sims_per_draw: int,
    seed: int,
    regime: str = "resampled",
    br_slot: int = 0,
    cfg_id: int = 0,
) -> list[dict]:
    """Run all cells for one strength config; return tidy per-cell metric records."""
    acc: dict[tuple, CellAccumulator] = {}

    # In the fixed regime the draw never changes, so collapse to a single larger batch.
    if regime == "fixed":
        draws, per_draw = 1, n_draws * sims_per_draw
    else:
        draws, per_draw = n_draws, sims_per_draw

    title_shares = []
    for d in range(draws):
        rng_draw = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 0)))
        rng_ev = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 1)))
        rng_eval = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 2)))
        groups = _draw_groups(cfg, regime, rng_draw)

        stages_ev = simulate_tournament(cfg.model, groups, per_draw, rng_ev)
        stages_eval = simulate_tournament(cfg.model, groups, per_draw, rng_eval)
        champ_eval = champion_team(stages_eval)
        title_shares.append(topk_ev_share(team_title_prob(stages_ev), 8))

        for ladder_name in ladders:
            ladder = LADDERS[ladder_name]
            points_ev = ladder[stages_ev]
            points_eval = ladder[stages_eval]
            team_ev = points_ev.mean(axis=0)
            team_var = points_ev.var(axis=0)
            for n in n_values:
                for policy in policies:
                    key = (ladder_name, n, policy)
                    res = _run_draft(policy, n, team_ev, team_var, points_eval, br_slot)
                    rosters = res["rosters"]
                    scores = M.pool_scores(points_eval, rosters)
                    a = acc.setdefault(key, CellAccumulator())
                    a.scores.append(scores)
                    a.rho.append(M.spearman_per_sim(scores, team_ev, rosters))
                    sv = M.skill_variance_share(scores, team_ev, rosters)
                    a.skill_share.append(sv["skill_share"])
                    a.sigma_between.append(sv["sigma2_between"])
                    a.sigma_within.append(sv["sigma2_within"])
                    credit, drafted = _accumulate_champion(
                        scores, champ_eval, res["drafter_of_team"]
                    )
                    a.champ_credit.append(credit)
                    a.champ_drafted.append(drafted)

    cfg.top8_title_share = float(np.mean(title_shares))
    return _finalize(acc, cfg, regime, draws * per_draw, br_slot)


def _finalize(acc, cfg, regime, n_eval, br_slot) -> list[dict]:
    records = []
    for (ladder_name, n, policy), a in acc.items():
        scores = np.vstack(a.scores)
        rho = np.concatenate(a.rho)
        valid_rho = rho[~np.isnan(rho)]
        credit = np.concatenate(a.champ_credit)
        drafted = np.concatenate(a.champ_drafted)
        slot_wp = M.win_probability(scores)
        imbalance = M.slot_equity_imbalance(scores)
        ws = M.winning_score_summary(scores)
        records.append(
            {
                "strength_config": cfg.name,
                "regime": regime,
                "top8_title_share": cfg.top8_title_share,
                "ladder": ladder_name,
                "teams_per_drafter": n,
                "policy": policy,
                "n_eval_tournaments": int(n_eval),
                "spearman_mean": float(np.mean(valid_rho)) if valid_rho.size else float("nan"),
                "spearman_sd": float(np.std(valid_rho)) if valid_rho.size else float("nan"),
                "spearman_undefined_frac": float(np.mean(np.isnan(rho))),
                "skill_variance_share": float(np.mean(a.skill_share)),
                "sigma2_between": float(np.mean(a.sigma_between)),
                "sigma2_within": float(np.mean(a.sigma_within)),
                "top_tie_rate": M.top_tie_rate(scores),
                "winning_score_mean": ws["mean"],
                "winning_score_sd": ws["sd"],
                "winning_score_p05": ws["p05"],
                "winning_score_p95": ws["p95"],
                "slot_win_prob_spread": imbalance["max_minus_min"],
                "slot_chi2_uniform": imbalance["chi2_uniform"],
                "champion_undrafted_rate": float(np.mean(~drafted)),
                "p_champion_holder_wins": float(np.mean(credit[drafted]))
                if drafted.any()
                else float("nan"),
                "br_slot": br_slot if policy == "best_response" else -1,
                **{f"slot{i + 1}_win_prob": float(slot_wp[i]) for i in range(len(slot_wp))},
            }
        )
    return records


def champion_probabilities(model, pots, n_sims, rng, n_draws=1):
    """Per-team champion probability under random draws (used by the bookmaker fit)."""
    shares = np.zeros(N_TEAMS)
    for _ in range(n_draws):
        groups = random_pot_draw(pots, rng)
        stages = simulate_tournament(model, groups, n_sims, rng)
        shares += team_title_prob(stages)
    return shares / n_draws
