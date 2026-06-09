"""Orchestration: build a strength config, run the draws, and assemble cell metrics.

A *cell* is a ``(scheme, N, policy)`` combination within a strength config, where ``scheme``
is a :class:`wcpool.scoring.ScoringScheme` (the two-layer knockout-advancement + group-W/D
rule). The expensive object — the simulated tournament — is **independent of the scheme, N,
and policy**, so each draw's tournament batch (group W/D tallies + furthest stages) is
simulated once and re-scored across the whole scheme x N x policy grid: the scheme only
re-weights the per-team outcome into points, and the draft only re-partitions teams among
drafters. The scheme loop therefore sits *inside* the per-draw loop, reusing the one set of
``stages``/``group`` arrays.

Two independent batches are drawn per group draw: an *EV batch* (estimates each team's
expected/variance points used by the drafters) and an *eval batch* (the held-out
tournaments the rosters are actually scored on). Using independent batches keeps the
skill-vs-luck estimates free of in-sample optimism.

``mix = 0`` schemes (the legacy ``ladders=`` path maps to these) recover the prior
terminal-only study *exactly*: ``scoring.team_points`` short-circuits to the bare ladder
lookup, so the per-cell metrics reproduce the prior run to floating-point tolerance (the
section-8.4 validation gate).

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
from . import scoring
from .ladders import LADDERS, Stage, get_ladder
from .scoring import GROUP_POINT_SCHEMES, ScoringScheme
from .strength import StrengthModel, synthetic_ratings
from .tournament import N_TEAMS, random_pot_draw, simulate_tournament

N_DRAFTERS = draft_mod.N_DRAFTERS

# Fixed namespace tag for synthetic-config RNG streams, keeping them disjoint from the
# main grid's spawn_key=(cfg_id, draw, batch) scheme. Arbitrary but documented.
SYNTH_SEED_NAMESPACE = 7

# Disjoint namespace tag for the gamma->mix calibration batch's RNG streams (the per-team
# group/knockout purse expectations G, K of plan section 4). Kept distinct from the main grid's
# batches {0, 1, 2}, the synthetic namespace 7, and the nested-sim namespace 3, so the calibration
# never shares a stream with any scored cell.
GAMMA_CALIB_SEED_NAMESPACE = 9

# Default number of replicates for the gamma->mix purse calibration (plan section 4: "a few thousand
# sims, fixed seed"). The calibrated quantities are field-averaged per-team SUMS -- E[sum wins],
# E[sum draws], E[sum A(stage)] -- which are very stable: total group wins == #decisive matches
# (~58/72) and draws == 2 * #drawn (~29) have tiny replicate-to-replicate spread, and even the
# geometric knockout purse (champion worth 32) is a sum over ~58 advancing-team contributions, so
# its mean over a few thousand replicates has an SE far below the gamma grid spacing (>= 0.05). 4096
# (a power of two) is comfortably in the "few thousand" band; split over a handful of resampled
# brackets so the purse marginalises the particular draw, matching the sweep's resampled regime.
_DEFAULT_GAMMA_CALIB_SIMS = 4096
_DEFAULT_GAMMA_CALIB_DRAWS = 8

# Default cap on the number of eval replicates fed to the conditional-win-probability surrogate
# (`metrics.conditional_win_prob`) per cell-draw, set by `run_strength_config`'s
# `suspense_subsample` argument.
#
# The k-NN suspense/surprise estimator is ~98% of per-cell-draw engagement compute (~0.5-0.7 s at
# 2000 sims vs a few ms for the estimator-free stage_variance_share/alive_fraction), so it dominates
# the sweep. It is also the *provisional* metric: its absolute level is one-sidedly biased by k-NN
# smoothing and is only adopted once the section-6 nested-simulation gate passes; the estimator-free
# stage_variance_share is the reported engagement headline (Doc-4 / plan section 6 fallback). So the
# suspense surrogate may run on a CAPPED replicate subset while the estimator-free metrics use the
# FULL per-cell-draw budget every draw.
#
# Value (512) is data-driven, not arbitrary: on the headline triangular/mix=0.15 cell the subsampled
# pool_suspense group-share converges toward the full-2000-sim value as O(1/sqrt(n)) (measured abs
# deviations: n=128 -> 0.039, 256 -> 0.023, 512 -> 0.016, 1024 -> 0.003); 512 is the smallest
# power-of-two giving an ~9x speedup on the dominant cost while the residual group-share distortion
# (~0.016) sits at the scale of the metric's own between-draw cluster SE (the reported uncertainty).
# Since the suspense level is provisional pending the nested-sim gate regardless, this is an
# acceptable provisional/compute trade-off. Pass `suspense_subsample=None` to disable the cap (full,
# exact). The cap never applies below its own size (a draw with <= 512 eval sims is unaffected) and
# does not touch the mix=0 zero-group-share guarantee (the GROUP boundary is fully constant on any
# subset, so its within-state mean is still exactly the prior).
_DEFAULT_SUSPENSE_SUBSAMPLE = 512


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


# --- gamma->mix calibration + scheme-grid construction (plan section 4) ------------------


@dataclass(frozen=True)
class GammaCalibration:
    """Field-averaged purse expectations for the gamma<->mix solver (plan section 4).

    A single calibration batch on the real-Elo field estimates the per-team-summed expectations the
    group-layer share ``gamma`` is defined against::

        e_sum_wins  = E[ sum_teams group wins ]
        e_sum_draws = E[ sum_teams group draws ]
        e_sum_A[shape] = E[ sum_teams A_shape(furthest stage) ]   (one per ladder shape)

    For a ``(w_pts, d_pts, shape)`` cell the purses are then closed-form arithmetic (no further
    simulation): group purse ``G = w_pts * e_sum_wins + d_pts * e_sum_draws`` (depends on W, D) and
    knockout purse ``K = e_sum_A[shape]`` (depends on shape). ``regime``/``seed``/``n_sims`` are
    retained for the ReproLog fingerprint.
    """

    e_sum_wins: float
    e_sum_draws: float
    e_sum_A: dict[str, float]
    regime: str
    seed: int
    n_sims: int

    def group_purse(self, w_pts: float, d_pts: float) -> float:
        """Group purse ``G = w_pts * E[sum wins] + d_pts * E[sum draws]`` for a ``(W, D)`` cell."""
        return w_pts * self.e_sum_wins + d_pts * self.e_sum_draws

    def knockout_purse(self, shape: str) -> float:
        """Knockout purse ``K = E[sum_teams A_shape(stage)]`` for a ladder ``shape``."""
        return self.e_sum_A[shape]


def calibrate_group_knockout(
    cfg: StrengthConfig,
    shapes: list[str] | None = None,
    *,
    seed: int = 0,
    regime: str = "resampled",
    n_sims: int = _DEFAULT_GAMMA_CALIB_SIMS,
    n_draws: int = _DEFAULT_GAMMA_CALIB_DRAWS,
    cfg_id: int = 0,
) -> GammaCalibration:
    """Estimate the gamma<->mix purse expectations from ONE calibration batch (plan section 4).

    Simulates ``n_sims`` tournaments (split over ``n_draws`` resampled brackets so the purses
    marginalise the particular draw, matching the sweep's resampled regime) and averages the
    per-replicate team sums of wins, draws, and each shape's ladder points. No drafting, no metrics
    -- just the field-level expectations the solver needs. The RNG draws from the disjoint
    :data:`GAMMA_CALIB_SEED_NAMESPACE` slot so the calibration never shares a stream with any scored
    cell. The result is the input to :func:`build_gamma_schemes`.

    Parameters
    ----------
    cfg : StrengthConfig
        The strength config (the real-Elo field for the headline calibration).
    shapes : list of str, optional
        Ladder shapes to compute ``K`` for; defaults to all of :data:`wcpool.ladders.LADDERS`.
    seed : int
        Master seed for the calibration ``SeedSequence``.
    regime : {"resampled", "fixed"}
        Draw regime (``"resampled"`` marginalises the bracket; ``"fixed"`` uses the official draw).
    n_sims : int
        Total replicates (split as evenly as possible over ``n_draws``).
    n_draws : int
        Number of (resampled) brackets to pool the calibration over (ignored in ``"fixed"``).
    cfg_id : int
        Configuration index threaded into the calibration ``spawn_key`` (default ``0``).

    Returns
    -------
    GammaCalibration
        The purse expectations plus the calibration provenance (regime/seed/n_sims).
    """
    if n_sims <= 0:
        raise ValueError(f"n_sims must be positive; got {n_sims}")
    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive; got {n_draws}")
    if shapes is None:
        shapes = list(LADDERS)
    ladder_vecs = {name: get_ladder(name) for name in shapes}
    if regime == "fixed":
        n_draws = 1
    per = max(n_sims // n_draws, 1)

    sum_wins = sum_draws = 0.0
    sum_A = dict.fromkeys(shapes, 0.0)
    total = 0
    for d in range(n_draws):
        rng_draw = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(GAMMA_CALIB_SEED_NAMESPACE, cfg_id, d, 0))
        )
        rng_sim = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(GAMMA_CALIB_SEED_NAMESPACE, cfg_id, d, 1))
        )
        groups = _draw_groups(cfg, regime, rng_draw)
        stages, grp = simulate_tournament(
            cfg.model, groups, per, rng_sim, return_group_results=True
        )
        # Per-replicate team sums, accumulated so the final divide is the overall mean over n_sims.
        sum_wins += float(grp["wins"].sum())
        sum_draws += float(grp["draws"].sum())
        for name, vec in ladder_vecs.items():
            sum_A[name] += float(vec[stages].sum())
        total += per

    return GammaCalibration(
        e_sum_wins=sum_wins / total,
        e_sum_draws=sum_draws / total,
        e_sum_A={name: sum_A[name] / total for name in shapes},
        regime=regime,
        seed=seed,
        n_sims=total,
    )


@dataclass(frozen=True)
class GammaCell:
    """One resolved cell of the gamma scheme grid: a :class:`ScoringScheme` + its design metadata.

    ``gamma_target`` is the requested share (a grid level, or ``-1.0`` for the per-cell
    ``gamma_match`` landmark); ``gamma_realised`` is ``gamma_of_mix(scheme.mix, G, K)`` at the G/K
    (equal to ``gamma_target`` to FP for the grid levels, and the landmark's own value for
    ``gamma_match``). ``is_gamma_match`` flags the commensurability landmark.
    """

    scheme: ScoringScheme
    gamma_target: float
    gamma_realised: float
    group_purse: float
    knockout_purse: float
    is_gamma_match: bool = False


def build_gamma_schemes(
    calib: GammaCalibration,
    gamma_grid: list[float],
    dw_candidates: dict[str, tuple[float, float]] | None = None,
    shapes: list[str] | None = None,
) -> list[GammaCell]:
    """Build the ``(D/W) x shape x gamma`` scheme grid of plan section 4 from a calibration.

    For each ``(w_pts, d_pts)`` candidate and ladder ``shape``, solve ``mix`` for every target
    ``gamma`` in ``gamma_grid`` plus the cell's own ``gamma_match`` landmark (``mix* = 1/(3W+1)``),
    using calibrated purses ``G = calib.group_purse(W, D)``, ``K = calib.knockout_purse(shape)``.

    The ``gamma == 0`` anchor is special: there the group layer is absent, so the entire ``3 x 3``
    (D/W x shape) anchor slice collapses to ONE cell per shape -- the validated ``mix == 0`` base,
    emitted as ``ScoringScheme(shape, 1.0, 0.0, 0.0)`` (the pure terminal ladder, whose ``(w_pts,
    d_pts)`` are irrelevant at ``mix == 0``; this keys to the historic bare-ladder cell so it
    self-validates against the frozen golden). Cells that resolve to an identical
    ``(shape, w_pts, d_pts, mix)`` within floating point are de-duplicated, so the net count is the
    documented ~39 distinct schemes (3 shapes at gamma=0 + 3 D/W x 3 shapes x the gamma>0 levels +
    gamma_match, minus FP-identical collisions).

    Parameters
    ----------
    calib : GammaCalibration
        The purse expectations from :func:`calibrate_group_knockout`.
    gamma_grid : list of float
        The target group-layer shares (e.g. ``[0, 0.05, 0.12, 0.25, 0.50]``); ``gamma_match`` is
        added per cell automatically and need not appear here.
    dw_candidates : dict, optional
        ``{name: (w_pts, d_pts)}`` draw/win candidates; defaults to
        :data:`wcpool.scoring.GROUP_POINT_SCHEMES` (the three plan-section-4 ratios).
    shapes : list of str, optional
        Ladder shapes; defaults to all of :data:`wcpool.ladders.LADDERS`.

    Returns
    -------
    list of GammaCell
        The de-duplicated grid cells. Ordered ``shape`` (outer) then ``(D/W)`` then ``gamma`` for a
        stable, readable sweep order, with the collapsed ``gamma == 0`` anchors emitted once per
        shape first.
    """
    if dw_candidates is None:
        dw_candidates = GROUP_POINT_SCHEMES
    if shapes is None:
        shapes = list(LADDERS)
    # A negative requested level would collide with the ``-1.0`` ``gamma_target`` reserved for the
    # per-cell gamma_match landmark (``solve_mix_for_gamma`` rejects gamma < 0 anyway), so reject it
    # up front: the sentinel must not be aliasable by a user-supplied grid value.
    bad = [g for g in gamma_grid if g < 0.0]
    if bad:
        raise ValueError(f"gamma_grid levels must be >= 0 (the -1.0 gamma_match sentinel is "
                         f"reserved); got negative {bad}")

    cells: list[GammaCell] = []
    seen: set[tuple[str, float, float, float]] = set()

    def _add(scheme: ScoringScheme, g_target: float, G: float, K: float, is_match: bool) -> None:
        ident = (scheme.knockout_ladder, scheme.w_pts, scheme.d_pts, scheme.mix)
        if ident in seen:
            return
        seen.add(ident)
        cells.append(
            GammaCell(
                scheme=scheme,
                gamma_target=g_target,
                gamma_realised=scoring.gamma_of_mix(scheme.mix, G, K),
                group_purse=G,
                knockout_purse=K,
                is_gamma_match=is_match,
            )
        )

    has_zero = any(g == 0.0 for g in gamma_grid)
    pos_gammas = [g for g in gamma_grid if g != 0.0]

    for shape in shapes:
        K = calib.knockout_purse(shape)
        # gamma == 0: the group layer is absent -> one collapsed anchor per shape (mix=0 baseline).
        if has_zero:
            _add(ScoringScheme(shape, 1.0, 0.0, 0.0), 0.0, calib.group_purse(1.0, 0.0), K, False)
        for w_pts, d_pts in dw_candidates.values():
            G = calib.group_purse(w_pts, d_pts)
            for g_target in pos_gammas:
                mix = scoring.solve_mix_for_gamma(g_target, G, K)
                _add(ScoringScheme(shape, w_pts, d_pts, mix), g_target, G, K, False)
            # Per-cell gamma_match landmark (mix* depends only on w_pts; share is field-derived).
            mix_star = scoring.gamma_match_mix(w_pts)
            _add(ScoringScheme(shape, w_pts, d_pts, mix_star), -1.0, G, K, True)

    return cells


@dataclass
class CellAccumulator:
    # Per-draw arrays appended once per draw (each a full eval-batch result for this cell): the
    # (n_sims, n_drafters) pool-score / champion-credit panels and the (n_sims,) per-sim spearman.
    scores: list[np.ndarray] = dc_field(default_factory=list)
    rho: list[np.ndarray] = dc_field(default_factory=list)
    sigma_between: list[float] = dc_field(default_factory=list)
    sigma_within: list[float] = dc_field(default_factory=list)
    champ_credit: list[np.ndarray] = dc_field(default_factory=list)
    champ_drafted: list[np.ndarray] = dc_field(default_factory=list)
    # Per-draw metric values: the draw is the independent replication unit, so cluster
    # (between-draw) standard errors are computed from these, NOT from the 50k pooled sims.
    draw_spearman: list[float] = dc_field(default_factory=list)
    draw_tie: list[float] = dc_field(default_factory=list)
    draw_slot_spread: list[float] = dc_field(default_factory=list)
    # Engagement metrics (group-stage extension), also accumulated per draw so each carries a
    # between-draw cluster SE: the group-phase variance share, the alive fraction entering the
    # knockouts, and the EFK group-phase suspense/surprise shares + totals.
    #
    # group_variance_share is pooled as a RATIO OF MEANS (mean_draws(var_group) over
    # mean_draws(var_total)), mirroring skill_variance_share's mb/(mb+mw) pooling, so the two
    # variance-share headlines are side-by-side comparable. The raw per-draw stage_variance_share
    # components are therefore accumulated for the numerator/denominator; the per-draw scalar SHARE
    # list is kept ONLY for the between-draw cluster SE (a mean-of-ratios SE is the right
    # uncertainty for the per-draw statistic, though the point estimate uses the ratio-of-means).
    draw_group_share: list[float] = dc_field(default_factory=list)
    draw_var_group: list[float] = dc_field(default_factory=list)
    draw_var_total: list[float] = dc_field(default_factory=list)
    draw_alive: list[float] = dc_field(default_factory=list)
    draw_suspense: list[float] = dc_field(default_factory=list)
    draw_surprise: list[float] = dc_field(default_factory=list)
    draw_suspense_group_share: list[float] = dc_field(default_factory=list)
    draw_surprise_group_share: list[float] = dc_field(default_factory=list)
    # Per-cell cache of the surrogate's CV-selected neighbour count per boundary. All draws within a
    # cell feed the surrogate a structurally identical batch (same effective replicate count, same
    # scheme/boundary granularity), so the expensive k-grid CV scan runs ONCE on the first draw and
    # is reused thereafter (finding 6). Populated and consumed by `conditional_win_prob`.
    suspense_k_cache: dict[int, int] = dc_field(default_factory=dict)


def _run_draft(
    policy: str,
    n_rounds: int,
    team_ev: np.ndarray,
    team_var: np.ndarray,
    points_decision: np.ndarray,
    br_slot: int,
    n_drafters: int,
) -> dict:
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


def _scheme_key(scheme: ScoringScheme) -> str | tuple[str, float, float, float]:
    """Canonical cell-key fragment identifying a scoring scheme.

    A pure terminal-ladder shim (``w_pts == 1``, ``d_pts == 0``, ``mix == 0`` — exactly what
    the legacy ``ladders=[...]`` path constructs, and the ``mix == 0`` baseline the validation
    gate replays) collapses to the *bare ladder name string*, so its cell key is the historic
    ``(ladder_name, n, policy)`` 3-tuple — keeping ``return_per_draw`` consumers (e.g.
    ``participant_sweep.py``) byte-compatible. This collapse is semantically exact: at
    ``mix == 0`` the group ``(w_pts, d_pts)`` never enter ``team_points`` (it short-circuits to
    the knockout lookup), so the scheme *is* the bare ladder regardless of ``w_pts``/``d_pts``.

    Any scheme with ``mix > 0`` (or a non-default group ``(w_pts, d_pts)`` at ``mix == 0``,
    which scores identically but is a distinct request) keys on the full identity tuple
    ``(knockout_ladder, w_pts, d_pts, mix)`` so distinct schemes never alias.
    """
    if scheme.mix == 0.0 and scheme.w_pts == 1.0 and scheme.d_pts == 0.0:
        return scheme.knockout_ladder
    return (scheme.knockout_ladder, scheme.w_pts, scheme.d_pts, scheme.mix)


def _resolve_schemes(
    schemes: list[ScoringScheme] | None, ladders: list[str] | None
) -> list[ScoringScheme]:
    """Resolve the scoring schemes from the new ``schemes`` arg or the legacy ``ladders`` shim.

    Back-compat: a caller passing ``ladders=[name, ...]`` (the prior API) is mapped to
    ``ScoringScheme(name, w_pts=1.0, d_pts=0.0, mix=0.0)`` per name — a pure terminal ladder
    whose ``(w_pts, d_pts)`` are irrelevant at ``mix == 0``. This reproduces the prior study
    bit-for-bit. Exactly one of ``schemes``/``ladders`` must be supplied.
    """
    if (schemes is None) == (ladders is None):
        raise ValueError("pass exactly one of `schemes` or `ladders`")
    if schemes is not None:
        return schemes
    return [ScoringScheme(name, w_pts=1.0, d_pts=0.0, mix=0.0) for name in ladders]


def _accumulate_champion(
    scores: np.ndarray, champ: np.ndarray, drafter_of_team: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def _accumulate_engagement(
    a: CellAccumulator,
    running: np.ndarray,
    stages_eval: np.ndarray,
    scheme: ScoringScheme,
    rosters: np.ndarray,
    scores: np.ndarray,
    suspense_subsample: int | None,
) -> None:
    """Append this draw's engagement metrics to the cell accumulator (like `_accumulate_champion`).

    The estimator-free headline metrics -- `stage_variance_share` (its ratio-of-means components +
    per-draw share) and `alive_fraction` -- are computed on the FULL per-cell-draw budget. The
    provisional, compute-dominant EFK suspense/surprise surrogate is computed on the first
    ``suspense_subsample`` replicates (a deterministic head slice, no RNG) when that cap is smaller
    than the batch, with the per-cell CV ``k`` reused via ``a.suspense_k_cache`` (finding 6). The
    ``mix == 0`` zero-group-share guarantee survives subsampling: the GROUP boundary is fully
    constant on any row subset, so its within-state mean is still exactly the prior.
    """
    svs = M.stage_variance_share(running)  # FULL budget (estimator-free headline)
    a.draw_group_share.append(svs["group_share"])  # per-draw share -> cluster SE only
    a.draw_var_group.append(svs["var_group"])  # ratio-of-means numerator component
    a.draw_var_total.append(svs["var_total"])  # ratio-of-means denominator component
    a.draw_alive.append(
        M.alive_fraction(running, stages_eval, scheme, rosters)["alive_fraction"]  # FULL budget
    )

    # Provisional EFK surrogate on the (optionally) capped subset; reuse the per-cell selected k.
    terminal_credit = M.terminal_win_credit(scores)
    if suspense_subsample is not None and running.shape[0] > suspense_subsample:
        running_s = running[:suspense_subsample]
        terminal_s = terminal_credit[:suspense_subsample]
    else:
        running_s, terminal_s = running, terminal_credit
    wcond = M.conditional_win_prob(
        running_s, terminal_s,
        selected_k_out=a.suspense_k_cache, selected_k_in=a.suspense_k_cache,
    )
    susp = M.pool_suspense(wcond)
    surp = M.pool_surprise(wcond)
    a.draw_suspense.append(susp["total"])
    a.draw_surprise.append(surp["total"])
    a.draw_suspense_group_share.append(susp["group_share"])
    a.draw_surprise_group_share.append(surp["group_share"])


def run_strength_config(
    cfg: StrengthConfig,
    ladders: list[str] | None = None,
    n_values: list[int] | None = None,
    policies: list[str] | None = None,
    n_draws: int = 1,
    sims_per_draw: int = 1000,
    seed: int = 0,
    regime: str = "resampled",
    br_slot: int = 0,
    cfg_id: int = 0,
    n_drafters: int = N_DRAFTERS,
    return_per_draw: bool = False,
    *,
    schemes: list[ScoringScheme] | None = None,
    suspense_subsample: int | None = _DEFAULT_SUSPENSE_SUBSAMPLE,
) -> list[dict] | tuple[list[dict], dict]:
    """Run all cells for one strength config; return tidy per-cell metric records.

    A *cell* is a ``(scheme, N, policy)`` combination. Supply the scoring rules either as the
    new ``schemes`` (a list of :class:`wcpool.scoring.ScoringScheme`) or via the legacy
    ``ladders`` (a list of ladder names) — exactly one. The ``ladders`` shim maps each name to
    ``ScoringScheme(name, w_pts=1.0, d_pts=0.0, mix=0.0)``, a pure terminal ladder that
    reproduces the prior study (``team_points`` short-circuits ``mix == 0`` to the bare ladder
    lookup, so the metrics match the prior CSV to floating-point tolerance).

    ``n_drafters`` (default 6) is the number of pool participants. Requires
    ``n_drafters * max(n_values) <= 48`` (cannot draft more teams than exist). Each record
    carries ``n_drafters`` ``slot{i}_win_prob`` columns, so records produced with different
    ``n_drafters`` are ragged — align/fill before concatenating across participant counts.

    Each record also carries the scheme columns (``knockout_ladder``/``w_pts``/``d_pts``/``mix``;
    the legacy ``ladder`` value mirrors ``knockout_ladder`` so existing plot filters keep
    working) and the engagement metrics computed on the eval batch — each with its between-draw
    cluster SE. **How to read the engagement columns:**

    * ``group_variance_share`` — the estimator-free **engagement headline**: the group-phase share
      of final-score variance (:func:`wcpool.metrics.stage_variance_share`). Pooled as a **ratio of
      means** (``mean_draws(var_group) / mean_draws(var_total)``), the *same* pooling as
      ``skill_variance_share`` above, so the two variance shares are directly comparable; its
      cluster SE is the between-draw spread of the per-draw share. Computed at the **full**
      per-cell-draw budget every draw.
    * ``alive_fraction`` — a layperson-facing feasibility sanity check, **NOT a discriminating
      engagement axis**: over the studied 8x6 grid with ``mix <= 0.5`` it saturates at ``>= 0.999``
      (cluster SE ``0``), so it does not separate the scoring schemes (see
      :func:`wcpool.metrics.alive_fraction`). Also full-budget every draw.
    * ``pool_suspense`` (+ ``pool_suspense_group_share``) — the **reported** EFK engagement metric,
      *provisional* (one-sidedly biased by the k-NN smoother; adopted only once the plan's section-6
      nested-simulation gate passes). ``pool_surprise`` (+ ``pool_surprise_group_share``) is emitted
      too but on this realised-path surrogate is **identical** to ``pool_suspense`` byte-for-byte
      (the summed squared belief increments read forward or backward are the same statistic), so the
      ``pool_surprise*`` columns are **not** independent confirmation of ``pool_suspense`` — the
      genuine forward/backward (suspense vs surprise) split is gated on the section-6 nested
      simulation. At ``mix == 0`` the suspense/surprise group-share is exactly ``0`` (the GROUP
      boundary is fully constant, so its conditional belief is exactly the prior).

    If ``return_per_draw`` is True, also return ``{cell_key: per-draw spearman array}`` so
    callers can compute *paired* between-draw statistics across runs that share ``cfg_id``
    (and therefore the same simulated tournaments per draw). For ``mix == 0`` ladder shims the
    ``cell_key`` is the historic ``(ladder_name, N, policy)`` 3-tuple (see :func:`_scheme_key`).

    ``suspense_subsample`` caps the number of eval replicates fed to the (compute-dominant,
    provisional) ``pool_suspense``/``pool_surprise`` surrogate per cell-draw; it defaults to
    :data:`_DEFAULT_SUSPENSE_SUBSAMPLE` and is documented there. The estimator-free
    ``group_variance_share``/``alive_fraction`` always use the full per-cell-draw budget. Pass
    ``None`` to disable the cap (full, exact suspense); a non-positive value raises ``ValueError``
    (it is almost always a caller error, and silently treating it as "no cap" would mask the typo).
    It does not perturb the ``mix == 0`` zero-group-share guarantee.

    Raises
    ------
    ValueError
        If neither/both of ``schemes``/``ladders`` is given, ``n_values``/``policies`` is missing,
        ``n_drafters * max(n_values) > N_TEAMS``, or ``suspense_subsample <= 0`` (use ``None`` to
        disable the cap).
    """
    resolved_schemes = _resolve_schemes(schemes, ladders)
    if n_values is None or policies is None:
        raise ValueError("n_values and policies are required")
    if n_drafters * max(n_values) > N_TEAMS:
        raise ValueError(
            f"{n_drafters} drafters x {max(n_values)} teams exceeds the {N_TEAMS}-team field"
        )
    # Reject a non-positive suspense cap rather than silently treating it as "no cap": a 0/negative
    # value is almost always a caller error (a typo'd small cap), and masking it would quietly run
    # the full, slow surrogate. Pass ``None`` (the documented sentinel) to disable the cap.
    if suspense_subsample is not None and suspense_subsample <= 0:
        raise ValueError(
            f"suspense_subsample must be positive (or None for no cap); got {suspense_subsample}"
        )
    acc: dict[tuple, CellAccumulator] = {}
    # Canonical scheme per cell key, so _finalize can emit the scheme columns. The key's scheme
    # fragment may be a bare ladder name (mix=0 shim) or the full identity tuple; this maps it
    # back to the ScoringScheme that produced the cell.
    key_scheme: dict[tuple, ScoringScheme] = {}

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

        # Capture the group W/D tallies ONCE per batch alongside the stages (negligible extra
        # memory): the scorer re-weights this single batch per scheme, so the tournament is
        # simulated once per draw and reused across the whole scheme x N x policy grid. The
        # `rng_*` streams are unchanged, so `stages_ev`/`stages_eval` are byte-identical to the
        # prior (single-return) call — the section-8.4 reproduction rests on this.
        stages_ev, grp_ev = simulate_tournament(
            cfg.model, groups, per_draw, rng_ev, return_group_results=True
        )
        stages_eval, grp_eval = simulate_tournament(
            cfg.model, groups, per_draw, rng_eval, return_group_results=True
        )
        champ_eval = champion_team(stages_eval)
        title_shares.append(topk_ev_share(team_title_prob(stages_ev), 8))

        for scheme in resolved_schemes:
            # Re-score the SAME EV/eval batches under this scheme (mix=0 -> bare ladder lookup).
            points_ev = scoring.team_points(stages_ev, grp_ev, scheme)
            points_eval = scoring.team_points(stages_eval, grp_eval, scheme)
            team_ev = points_ev.mean(axis=0)
            team_var = points_ev.var(axis=0)
            scheme_key = _scheme_key(scheme)
            for n in n_values:
                for policy in policies:
                    key = (scheme_key, n, policy)
                    key_scheme.setdefault(key, scheme)
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
                    # Engagement metrics on the EVAL batch (per draw within this cell, mirroring the
                    # SE'd metrics above): the running-score panel for THIS scheme/roster, then the
                    # estimator-free stage-variance + alive-fraction (FULL budget) and the EFK
                    # suspense/surprise over the conditional-win-prob path (optionally subsampled).
                    running = scoring.running_scores(stages_eval, grp_eval, scheme, rosters)
                    _accumulate_engagement(
                        a, running, stages_eval, scheme, rosters, scores, suspense_subsample
                    )

    cfg.top8_title_share = float(np.mean(title_shares))
    records = _finalize(acc, key_scheme, cfg, regime, draws * per_draw, draws, br_slot)
    if return_per_draw:
        per_draw_skill = {key: np.array(a.draw_spearman) for key, a in acc.items()}
        return records, per_draw_skill
    return records


def _cluster_se(per_draw_values: list[float] | np.ndarray) -> float:
    """Between-draw (cluster) SE of a metric's per-draw values; NaN if < 2 draws."""
    v = np.asarray(per_draw_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def paired_delta_skill(
    cell_per_draw: list[float] | np.ndarray,
    anchor_per_draw: list[float] | np.ndarray,
) -> tuple[float, float]:
    """PAIRED between-draw skill delta of a cell vs its anchor, with a paired cluster SE.

    Both inputs are the per-draw skill series (the per-draw mean Spearman, one value per draw) for a
    cell and its gamma=0 anchor, computed on the SAME tournament brackets per draw (the producer
    runs every scheme on one shared per-draw batch under one ``cfg_id``; see ``run_strength_config(
    ..., return_per_draw=True)``). The delta is formed PER DRAW and then averaged::

        delta_per_draw[d] = cell[d] - anchor[d]      # paired on the same bracket d
        delta             = mean_d delta_per_draw
        paired_se         = std_d(delta_per_draw, ddof=1) / sqrt(n_draws)   (:func:`_cluster_se`)

    The pairing cancels the dominant between-draw bracket variance, making this ~7x more seed-stable
    than the absolute level (the basis the selection must rest on; plan section 8 / 14). NaN
    per-draw deltas are dropped pairwise (the point estimate is the NaN-dropped mean, and
    ``_cluster_se`` drops them too). An anchor against itself gives identically-zero deltas, so the
    result is ``(0.0, 0.0)``.

    Parameters
    ----------
    cell_per_draw, anchor_per_draw : array-like of float
        The cell's and the anchor's per-draw skill series. Must be the same length (paired draws);
        a length mismatch raises ``ValueError``.

    Returns
    -------
    (float, float)
        ``(delta_skill_vs_anchor, delta_skill_paired_se)``. The SE is ``0.0`` when every paired
        delta is identically zero (the anchor-vs-itself case) and ``nan`` when fewer than two draws
        survive the NaN drop.
    """
    cell = np.asarray(cell_per_draw, dtype=float)
    anchor = np.asarray(anchor_per_draw, dtype=float)
    if cell.shape != anchor.shape:
        raise ValueError(
            f"paired delta needs equal-length per-draw series; got {cell.shape} vs {anchor.shape}"
        )
    per_draw_delta = cell - anchor
    valid = per_draw_delta[~np.isnan(per_draw_delta)]
    delta = float(np.mean(valid)) if valid.size else float("nan")
    return delta, _cluster_se(per_draw_delta)


def _ratio_of_means(numerators: list[float], denominators: list[float]) -> float:
    """Pooled ratio of means ``mean(numerators) / mean(denominators)``; NaN if the denominator is 0.

    The pooling used for the variance-share headlines (``skill_variance_share``,
    ``group_variance_share``): average the per-draw numerator and denominator components separately,
    then divide. This is comparable across metrics and avoids the small-sample bias of a mean of
    per-draw ratios. NaN per-draw components (none expected here) are dropped pairwise first.
    """
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    ok = ~(np.isnan(num) | np.isnan(den))
    if not ok.any():
        return float("nan")
    mean_den = float(np.mean(den[ok]))
    return float(np.mean(num[ok]) / mean_den) if mean_den > 0 else float("nan")


def _finalize(
    acc: dict[tuple, CellAccumulator],
    key_scheme: dict[tuple, ScoringScheme],
    cfg: StrengthConfig,
    regime: str,
    n_eval: int,
    n_draws: int,
    br_slot: int,
) -> list[dict]:
    records = []
    for key, a in acc.items():
        _, n, policy = key
        scheme = key_scheme[key]
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
                # `ladder` mirrors the knockout-ladder name so existing plot/sweep filters keep
                # working; the explicit scheme columns carry the full two-layer identity.
                "ladder": scheme.knockout_ladder,
                "knockout_ladder": scheme.knockout_ladder,
                "w_pts": float(scheme.w_pts),
                "d_pts": float(scheme.d_pts),
                "mix": float(scheme.mix),
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
                # Engagement metrics (eval batch): per-draw means + between-draw cluster SEs.
                # group_variance_share is a pooled RATIO OF MEANS (mean_draws(var_group) /
                # mean_draws(var_total)) -- the same pooling as skill_variance_share above, so the
                # two variance shares are directly comparable -- NOT a (biased) mean of per-draw
                # ratios. Its cluster SE still comes from the per-draw share list (the per-draw
                # statistic's between-draw spread).
                "group_variance_share": _ratio_of_means(a.draw_var_group, a.draw_var_total),
                "group_variance_share_cluster_se": _cluster_se(a.draw_group_share),
                "alive_fraction": float(np.mean(a.draw_alive)),
                "alive_fraction_cluster_se": _cluster_se(a.draw_alive),
                "pool_suspense": float(np.mean(a.draw_suspense)),
                "pool_suspense_cluster_se": _cluster_se(a.draw_suspense),
                "pool_suspense_group_share": float(np.mean(a.draw_suspense_group_share)),
                "pool_suspense_group_share_cluster_se": _cluster_se(a.draw_suspense_group_share),
                "pool_surprise": float(np.mean(a.draw_surprise)),
                "pool_surprise_cluster_se": _cluster_se(a.draw_surprise),
                "pool_surprise_group_share": float(np.mean(a.draw_surprise_group_share)),
                "pool_surprise_group_share_cluster_se": _cluster_se(a.draw_surprise_group_share),
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
