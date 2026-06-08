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

# Fixed namespace tag for synthetic-config RNG streams, keeping them disjoint from the
# main grid's spawn_key=(cfg_id, draw, batch) scheme. Arbitrary but documented.
SYNTH_SEED_NAMESPACE = 7


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


def make_synthetic_config(
    spread: float, name: str, seed: int, rating_stream: int = 0
) -> StrengthConfig:
    # The favoritism sweep is a CONTROLLED concentration ladder: every synthetic config
    # shares one standard-normal draw (fixed rating_stream) and differs only by `spread`,
    # so the induced top-k concentration is monotone in spread by construction and the
    # sweep isolates the concentration effect. (rating_stream lets callers vary the field.)
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(SYNTH_SEED_NAMESPACE, rating_stream))
    )
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
        return cfg.fixed_groups.copy()  # copy so callers cannot mutate the shared config
    if regime == "resampled":
        return random_pot_draw(cfg.pots, rng)
    raise ValueError(f"unknown regime {regime!r}")


@dataclass
class CellAccumulator:
    scores: list = dc_field(default_factory=list)
    rho: list = dc_field(default_factory=list)
    sigma_between: list = dc_field(default_factory=list)
    sigma_within: list = dc_field(default_factory=list)
    champ_credit: list = dc_field(default_factory=list)
    champ_drafted: list = dc_field(default_factory=list)
    # Per-draw metric values: the draw is the independent replication unit, so cluster
    # (between-draw) standard errors are computed from these, NOT from the 50k pooled sims.
    draw_spearman: list = dc_field(default_factory=list)
    draw_tie: list = dc_field(default_factory=list)
    draw_slot_spread: list = dc_field(default_factory=list)


def _run_draft(policy, n_rounds, team_ev, team_var, points_decision, br_slot, n_drafters):
    """Build rosters. Best-response decides on ``points_decision`` (the EV/model batch),
    never on the held-out eval batch it is later scored against."""
    if policy == "ev_greedy":
        return draft_mod.draft_ev_greedy(team_ev, n_drafters, n_rounds)
    if policy == "variance":
        return draft_mod.draft_variance(team_var, n_drafters, n_rounds)
    if policy == "best_response":
        return draft_mod.draft_best_response(
            points_decision, team_ev, br_slot, n_drafters, n_rounds
        )
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
    n_drafters: int = N_DRAFTERS,
    return_per_draw: bool = False,
):
    """Run all cells for one strength config; return tidy per-cell metric records.

    ``n_drafters`` (default 6) is the number of pool participants. Requires
    ``n_drafters * max(n_values) <= 48`` (cannot draft more teams than exist). Each record
    carries ``n_drafters`` ``slot{i}_win_prob`` columns, so records produced with different
    ``n_drafters`` are ragged — align/fill before concatenating across participant counts.

    If ``return_per_draw`` is True, also return ``{cell_key: per-draw spearman array}`` so
    callers can compute *paired* between-draw statistics across runs that share ``cfg_id``
    (and therefore the same simulated tournaments per draw).
    """
    if n_drafters * max(n_values) > N_TEAMS:
        raise ValueError(
            f"{n_drafters} drafters x {max(n_values)} teams exceeds the {N_TEAMS}-team field"
        )
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
                    # Decisions use the EV (model) batch; scoring uses the eval batch.
                    res = _run_draft(
                        policy, n, team_ev, team_var, points_ev, br_slot, n_drafters
                    )
                    rosters = res["rosters"]
                    scores = M.pool_scores(points_eval, rosters)
                    a = acc.setdefault(key, CellAccumulator())
                    a.scores.append(scores)
                    rho = M.spearman_per_sim(scores, team_ev, rosters)
                    a.rho.append(rho)
                    sv = M.skill_variance_share(scores, team_ev, rosters)
                    a.sigma_between.append(sv["sigma2_between"])
                    a.sigma_within.append(sv["sigma2_within"])
                    credit, drafted = _accumulate_champion(
                        scores, champ_eval, res["drafter_of_team"]
                    )
                    a.champ_credit.append(credit)
                    a.champ_drafted.append(drafted)
                    # per-draw values for between-draw cluster SE
                    a.draw_spearman.append(float(np.nanmean(rho)))
                    a.draw_tie.append(M.top_tie_rate(scores))
                    a.draw_slot_spread.append(M.slot_equity_imbalance(scores)["max_minus_min"])

    cfg.top8_title_share = float(np.mean(title_shares))
    records = _finalize(acc, cfg, regime, draws * per_draw, draws, br_slot)
    if return_per_draw:
        per_draw_skill = {key: np.array(a.draw_spearman) for key, a in acc.items()}
        return records, per_draw_skill
    return records


def _cluster_se(per_draw_values) -> float:
    """Between-draw (cluster) SE of a metric's per-draw values; NaN if < 2 draws."""
    v = np.asarray(per_draw_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def _finalize(acc, cfg, regime, n_eval, n_draws, br_slot) -> list[dict]:
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
        # skill variance share as a pooled ratio-of-means (not a biased mean-of-ratios)
        mb, mw = float(np.mean(a.sigma_between)), float(np.mean(a.sigma_within))
        skill_share = mb / (mb + mw) if (mb + mw) > 0 else float("nan")
        records.append(
            {
                "strength_config": cfg.name,
                "regime": regime,
                "top8_title_share": cfg.top8_title_share,
                "ladder": ladder_name,
                "teams_per_drafter": n,
                "policy": policy,
                "n_eval_tournaments": int(n_eval),
                "n_draws": int(n_draws),
                "spearman_mean": float(np.mean(valid_rho)) if valid_rho.size else float("nan"),
                "spearman_sd": float(np.std(valid_rho)) if valid_rho.size else float("nan"),
                # between-draw cluster SE (the draw is the independent unit, not the sim)
                "spearman_cluster_se": _cluster_se(a.draw_spearman),
                "spearman_undefined_frac": float(np.mean(np.isnan(rho))),
                "skill_variance_share": skill_share,
                "sigma2_between": mb,
                "sigma2_within": mw,
                "top_tie_rate": M.top_tie_rate(scores),
                "top_tie_rate_cluster_se": _cluster_se(a.draw_tie),
                "winning_score_mean": ws["mean"],
                "winning_score_sd": ws["sd"],
                "winning_score_p05": ws["p05"],
                "winning_score_p95": ws["p95"],
                "slot_win_prob_spread": imbalance["max_minus_min"],
                "slot_win_prob_spread_cluster_se": _cluster_se(a.draw_slot_spread),
                "slot_imbalance_index": imbalance["imbalance_index"],
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
    """Per-team champion probability under random draws (used by the bookmaker fit).

    ``n_sims`` should be large enough that the per-team title probabilities are stable to
    the precision the temperature fit needs; the headline study uses the Elo + synthetic
    paths instead, so no fixed sizing is mandated here (the caller chooses ``n_sims``).
    """
    shares = np.zeros(N_TEAMS)
    for _ in range(n_draws):
        groups = random_pot_draw(pots, rng)
        stages = simulate_tournament(model, groups, n_sims, rng)
        shares += team_title_prob(stages)
    return shares / n_draws
