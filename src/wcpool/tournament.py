"""2026 FIFA World Cup tournament structure and vectorised Monte-Carlo simulation.

Structure (verified against the official bracket; see ``docs/assumptions.md`` for the
primary-source citations):

* 48 teams, 12 groups (A..L) of 4, single round-robin (6 matches/group, 3 each).
* Group ranking: points (3/1/0), then goal difference, then goals for, then a random
  draw of lots (FIFA tie-break order; head-to-head and fair-play are omitted — see the
  assumptions doc).
* Advance to the Round of 32: the 12 group winners, the 12 runners-up, and the 8 best
  third-placed teams ranked across groups by the same points -> GD -> GF -> lots order.
* The eight qualifying third-placed teams are assigned to the eight "winner vs third"
  Round-of-32 slots using the official eligibility template (FIFA Annex C, 495
  combinations), solved here as a bipartite matching that respects each slot's eligible
  group set.
* Knockout: R32 -> R16 -> QF -> SF -> Final, single elimination. Drawn knockout matches
  are decided by the strength-weighted tie rule in ``strength.knockout_advance_prob``.

The whole tournament is simulated vectorised over ``n_sims`` replicates with NumPy; the
output is an ``(n_sims, n_teams)`` array of *furthest stage reached* (``ladders.Stage``).
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ladders import N_STAGES, Stage
from .scoring import GroupTallies

# --- Group-stage layout -----------------------------------------------------------------

N_GROUPS = 12
GROUP_SIZE = 4
N_TEAMS = N_GROUPS * GROUP_SIZE  # 48
GROUP_LABELS = [chr(ord("A") + g) for g in range(N_GROUPS)]

# The 6 round-robin pairings among 4 local positions in a group.
GROUP_MATCH_PAIRS: list[tuple[int, int]] = list(itertools.combinations(range(GROUP_SIZE), 2))

# FIFA points (Laws of the competition): win 3, draw 1, loss 0.
WIN_POINTS, DRAW_POINTS, LOSS_POINTS = 3, 1, 0


def _letters(s: str) -> frozenset[int]:
    return frozenset(ord(c) - ord("A") for c in s)


# Eligible third-place source groups for each of the 8 "winner vs third" R32 slots,
# in R32 match-number order (M74, M77, M79, M80, M81, M82, M85, M87). Taken verbatim
# from the official R32 pairing labels "3rd (X/Y/Z/...)".
THIRD_SLOT_ELIGIBILITY: list[frozenset[int]] = [
    _letters("ABCDF"),  # slot 0 -> M74, opponent 1E
    _letters("CDFGH"),  # slot 1 -> M77, opponent 1I
    _letters("CEFHI"),  # slot 2 -> M79, opponent 1A
    _letters("EHIJK"),  # slot 3 -> M80, opponent 1L
    _letters("BEFIJ"),  # slot 4 -> M81, opponent 1D
    _letters("AEHIJ"),  # slot 5 -> M82, opponent 1G
    _letters("EFGIJ"),  # slot 6 -> M85, opponent 1B
    _letters("DEIJL"),  # slot 7 -> M87, opponent 1K
]
N_THIRD_SLOTS = len(THIRD_SLOT_ELIGIBILITY)  # 8

# R32 matches in match-number order 73..88. Each spec is ("W"|"R", group_idx) for a group
# winner / runner-up, or ("T", slot_idx) for a third-place slot.
R32_MATCHES: list[tuple[tuple[str, int], tuple[str, int]]] = [
    (("R", 0), ("R", 1)),  # M73  2A v 2B
    (("W", 4), ("T", 0)),  # M74  1E v 3rd
    (("W", 5), ("R", 2)),  # M75  1F v 2C
    (("W", 2), ("R", 5)),  # M76  1C v 2F
    (("W", 8), ("T", 1)),  # M77  1I v 3rd
    (("R", 4), ("R", 8)),  # M78  2E v 2I
    (("W", 0), ("T", 2)),  # M79  1A v 3rd
    (("W", 11), ("T", 3)),  # M80  1L v 3rd
    (("W", 3), ("T", 4)),  # M81  1D v 3rd
    (("W", 6), ("T", 5)),  # M82  1G v 3rd
    (("R", 10), ("R", 11)),  # M83  2K v 2L
    (("W", 7), ("R", 9)),  # M84  1H v 2J
    (("W", 1), ("T", 6)),  # M85  1B v 3rd
    (("W", 9), ("R", 7)),  # M86  1J v 2H
    (("W", 10), ("T", 7)),  # M87  1K v 3rd
    (("R", 3), ("R", 6)),  # M88  2D v 2G
]

# Knockout tree: each round pairs winners of the previous round by their index. R32
# winners are indexed 0..15 in the R32_MATCHES order above; each subsequent list pairs
# indices into the previous round's winners. Verified against the official bracket
# (R16 M89-96, QF M97-100, SF M101-102, Final M104).
KNOCKOUT_TREE: list[list[tuple[int, int]]] = [
    [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)],  # R16
    [(0, 1), (4, 5), (2, 3), (6, 7)],  # QF
    [(0, 1), (2, 3)],  # SF
    [(0, 1)],  # Final
]


@lru_cache(maxsize=1)
def thirdplace_assignment_table() -> dict[int, tuple[int, ...]]:
    """Map each set of 8 qualifying groups to a slot->source-group assignment.

    Key is a 12-bit mask of the qualifying group indices (popcount 8). Value is a length-8
    tuple ``slot_group`` where ``slot_group[s]`` is the group index feeding R32 third-slot
    ``s``. Solved as a min-cost bipartite matching (cost 0 for an eligible (group, slot)
    edge, 1 otherwise); a feasible assignment uses only eligible edges (total cost 0).

    Raises if any of the C(12,8)=495 combinations is infeasible under the eligibility
    template, which would signal a transcription error in ``THIRD_SLOT_ELIGIBILITY``.
    """
    eligibility = THIRD_SLOT_ELIGIBILITY
    table: dict[int, tuple[int, ...]] = {}
    infeasible: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(N_GROUPS), N_THIRD_SLOTS):
        rows = list(combo)
        cost = np.ones((N_THIRD_SLOTS, N_THIRD_SLOTS), dtype=np.int64)
        for i, g in enumerate(rows):
            for s in range(N_THIRD_SLOTS):
                if g in eligibility[s]:
                    cost[i, s] = 0
        r_ind, c_ind = linear_sum_assignment(cost)
        if cost[r_ind, c_ind].sum() != 0:
            infeasible.append(combo)
            continue
        slot_group = [0] * N_THIRD_SLOTS
        for i, s in zip(r_ind, c_ind, strict=True):
            slot_group[s] = rows[i]
        mask = 0
        for g in combo:
            mask |= 1 << g
        table[mask] = tuple(slot_group)
    if infeasible:
        raise ValueError(
            f"{len(infeasible)} of 495 third-place combinations are infeasible under the "
            f"eligibility template; first: {infeasible[0]}"
        )
    return table


# --- Draws ------------------------------------------------------------------------------


def random_pot_draw(pots: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pot-constrained random draw: one team from each pot into each group.

    ``pots`` is shape ``(4, 12)`` of global team indices (pot x slot). Returns a
    ``(12, 4)`` array of global team indices (group x within-group position by pot). This
    enforces the pot constraint only; confederation spread constraints are not modelled
    (documented simplification).
    """
    pots = np.asarray(pots)
    if pots.shape != (GROUP_SIZE, N_GROUPS):
        raise ValueError(f"pots must be ({GROUP_SIZE}, {N_GROUPS}); got {pots.shape}")
    groups = np.empty((N_GROUPS, GROUP_SIZE), dtype=np.int64)
    for p in range(GROUP_SIZE):
        groups[:, p] = rng.permutation(pots[p])
    return groups


# --- Group stage ------------------------------------------------------------------------


def simulate_group_stage(
    model, groups: np.ndarray, n_sims: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate every group match; return per-team group-stage tallies.

    Parameters
    ----------
    model
        Strength model exposing ``goal_rates`` (Poisson scoring rates per match-up).
    groups : numpy.ndarray
        ``(N_GROUPS, GROUP_SIZE)`` array of global team indices laid out group-major.
    n_sims : int
        Number of independent replicates.
    rng : numpy.random.Generator
        Source of randomness for the Poisson goal draws.

    Returns
    -------
    pts : numpy.ndarray
        FIFA group points (3/1/0), ``int32`` shape ``(n_sims, N_TEAMS)``.
    gd : numpy.ndarray
        Goal difference (goals for minus against), ``int32`` shape ``(n_sims, N_TEAMS)``.
    gf : numpy.ndarray
        Goals for, ``int32`` shape ``(n_sims, N_TEAMS)``.
    wins : numpy.ndarray
        Group-match wins per team, ``int32`` shape ``(n_sims, N_TEAMS)`` (each in ``0..3``).
    draws : numpy.ndarray
        Group-match draws per team, ``int32`` shape ``(n_sims, N_TEAMS)`` (each in ``0..3``).
    """
    pts = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    gf = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    ga = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    wins = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    draws = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    for g in range(N_GROUPS):
        for la, lb in GROUP_MATCH_PAIRS:
            a = int(groups[g, la])
            b = int(groups[g, lb])
            lam_a, lam_b = model.goal_rates(np.array([a]), np.array([b]))
            goals_a = rng.poisson(float(lam_a[0]), n_sims)
            goals_b = rng.poisson(float(lam_b[0]), n_sims)
            a_win = goals_a > goals_b
            b_win = goals_a < goals_b
            draw = ~(a_win | b_win)
            pts[:, a] += np.where(a_win, WIN_POINTS, np.where(draw, DRAW_POINTS, LOSS_POINTS))
            pts[:, b] += np.where(b_win, WIN_POINTS, np.where(draw, DRAW_POINTS, LOSS_POINTS))
            gf[:, a] += goals_a
            gf[:, b] += goals_b
            ga[:, a] += goals_b
            ga[:, b] += goals_a
            wins[:, a] += a_win
            wins[:, b] += b_win
            draws[:, a] += draw
            draws[:, b] += draw
    return pts, gf - ga, gf, wins, draws


def _group_finishing_order(keys_local: tuple[np.ndarray, ...]) -> np.ndarray:
    """Best-to-worst local ordering from (pts, gd, gf, lots) arrays each shape (n_sims, k)."""
    pts, gd, gf, lots = keys_local
    order_asc = np.lexsort((lots, gf, gd, pts), axis=-1)  # primary key = pts, ascending
    return order_asc[:, ::-1]  # best first


def standings(
    pts: np.ndarray, gd: np.ndarray, gf: np.ndarray, groups: np.ndarray, lots: np.ndarray
) -> dict[str, np.ndarray]:
    """Resolve group standings.

    Returns global team indices for each finishing position: ``winner``/``runner``/
    ``third`` each shape (n_sims, 12), plus the third-placed teams' (pts, gd, gf) for the
    cross-group ranking.
    """
    n_sims = pts.shape[0]
    winner = np.empty((n_sims, N_GROUPS), dtype=np.int64)
    runner = np.empty((n_sims, N_GROUPS), dtype=np.int64)
    third = np.empty((n_sims, N_GROUPS), dtype=np.int64)
    for g in range(N_GROUPS):
        gidx = groups[g]  # (4,) global indices
        order = _group_finishing_order((pts[:, gidx], gd[:, gidx], gf[:, gidx], lots[:, gidx]))
        winner[:, g] = gidx[order[:, 0]]
        runner[:, g] = gidx[order[:, 1]]
        third[:, g] = gidx[order[:, 2]]
    return {"winner": winner, "runner": runner, "third": third}


# --- Knockout ---------------------------------------------------------------------------


def _resolve_third_slots(
    third_global: np.ndarray,
    pts: np.ndarray,
    gd: np.ndarray,
    gf: np.ndarray,
    lots: np.ndarray,
) -> np.ndarray:
    """Fill the 8 third-place R32 slots. Returns (n_sims, 8) of global team indices.

    Ranks the 12 group third-placed teams per replicate (pts -> GD -> GF -> lots), takes
    the best 8, then maps the qualifying group set through the Annex-C assignment table.
    """
    n_sims = third_global.shape[0]
    rows = np.arange(n_sims)
    # third-placed teams' keys, per group (n_sims, 12)
    k_pts = np.take_along_axis(pts, third_global, axis=1)
    k_gd = np.take_along_axis(gd, third_global, axis=1)
    k_gf = np.take_along_axis(gf, third_global, axis=1)
    k_lots = np.take_along_axis(lots, third_global, axis=1)
    order_asc = np.lexsort((k_lots, k_gf, k_gd, k_pts), axis=-1)  # ascending by key
    qualifying_groups = order_asc[:, N_GROUPS - N_THIRD_SLOTS :]  # best 8 group indices

    masks = np.zeros(n_sims, dtype=np.int64)
    for c in range(N_THIRD_SLOTS):
        masks |= (1 << qualifying_groups[:, c]).astype(np.int64)

    table = thirdplace_assignment_table()
    unique_masks, inverse = np.unique(masks, return_inverse=True)
    assign_unique = np.array([table[int(m)] for m in unique_masks], dtype=np.int64)  # (U,8)
    slot_group = assign_unique[inverse]  # (n_sims, 8): source group feeding each slot

    slots = np.empty((n_sims, N_THIRD_SLOTS), dtype=np.int64)
    for s in range(N_THIRD_SLOTS):
        slots[:, s] = third_global[rows, slot_group[:, s]]
    return slots


def _r32_participants(st: dict, third_slots: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the 16 R32 (teamA, teamB) global-index pairs from standings + third slots."""

    def resolve(spec: tuple[str, int], sim_rows: np.ndarray) -> np.ndarray:
        kind, idx = spec
        if kind == "W":
            return st["winner"][:, idx]
        if kind == "R":
            return st["runner"][:, idx]
        return third_slots[:, idx]  # "T"

    rows = np.arange(next(iter(st.values())).shape[0])
    return [(resolve(a, rows), resolve(b, rows)) for a, b in R32_MATCHES]


def _play_round(
    model, pairs: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> list[np.ndarray]:
    """Play a list of knockout matches; return the list of winner index arrays."""
    winners = []
    for a, b in pairs:
        p_adv = model.knockout_advance_prob(a, b)
        coin = rng.random(len(a))
        winners.append(np.where(coin < p_adv, a, b))
    return winners


def simulate_tournament(
    model,
    groups: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    *,
    return_group_results: bool = False,
) -> np.ndarray | tuple[np.ndarray, GroupTallies]:
    """Run ``n_sims`` full tournaments on a fixed ``groups`` draw.

    Parameters
    ----------
    model
        Strength model exposing ``goal_rates`` and ``knockout_advance_prob``.
    groups : numpy.ndarray
        ``(N_GROUPS, GROUP_SIZE)`` array of global team indices.
    n_sims : int
        Number of independent replicates.
    rng : numpy.random.Generator
        Source of randomness. The same stream feeds the group, lots, and knockout draws,
        so the returned ``stages`` are unaffected by ``return_group_results``.
    return_group_results : bool, optional
        When ``False`` (default) return only ``stages`` — the historical contract, so all
        existing callers are byte-for-byte unchanged. When ``True`` additionally return the
        group win/draw tallies for the group-layer scorer (``scoring.team_points``).

    Returns
    -------
    stages : numpy.ndarray
        ``(n_sims, N_TEAMS)`` ``int8`` array of the furthest ``Stage`` each team reached.
    group : GroupTallies, optional
        Only when ``return_group_results`` is ``True``: ``{"wins": ..., "draws": ...}`` each
        ``int32`` shape ``(n_sims, N_TEAMS)`` (group-match wins/draws per team).
    """
    pts, gd, gf, wins, draws = simulate_group_stage(model, groups, n_sims, rng)
    lots = rng.random((n_sims, N_TEAMS))  # per-team "drawing of lots" tie-break
    st = standings(pts, gd, gf, groups, lots)
    third_slots = _resolve_third_slots(st["third"], pts, gd, gf, lots)

    stages = np.zeros((n_sims, N_TEAMS), dtype=np.int8)
    rows = np.arange(n_sims)

    r32 = _r32_participants(st, third_slots)
    for a, b in r32:
        stages[rows, a] = Stage.R32
        stages[rows, b] = Stage.R32

    # R32 results -> winners reach R16 (Stage 2). Subsequent rounds add one stage each.
    winners = _play_round(model, r32, rng)
    stage_value = int(Stage.R16)
    for w in winners:
        stages[rows, w] = stage_value
    for round_pairs in KNOCKOUT_TREE:
        stage_value += 1
        pairs = [(winners[i], winners[j]) for i, j in round_pairs]
        winners = _play_round(model, pairs, rng)
        for w in winners:
            stages[rows, w] = stage_value
    if return_group_results:
        group: GroupTallies = {"wins": wins, "draws": draws}
        return stages, group
    return stages


# --- Validation-only: bracket trace + conditional knockout replay -----------------------
#
# The functions below are NOT on the single-pass hot path used by the experiment runner; they
# exist solely to build a nested-simulation ground truth for the conditional-win-probability
# surrogate (``metrics.conditional_win_prob``), per the plan's section-6 gate.
# ``simulate_tournament`` above is untouched and byte-for-byte unchanged.


def simulate_tournament_trace(
    model, groups: np.ndarray, n_sims: int, rng: np.random.Generator
) -> tuple[np.ndarray, GroupTallies, np.ndarray, list[list[np.ndarray]]]:
    """Like :func:`simulate_tournament` but also return the realised knockout bracket trace.

    Validation-only. Re-runs the full tournament (reusing the same helpers, so a fixed ``rng``
    stream reproduces ``simulate_tournament``'s ``stages`` exactly) and additionally captures the
    bracket occupancy needed to *replay the knockout from any boundary*: the 16 Round-of-32 pairs
    and, after each played round, the surviving teams (the winners list that the next round pairs
    over). This is what :func:`replay_knockout_from_round` consumes to condition on a realised
    state and Monte-Carlo the remainder.

    Parameters
    ----------
    model
        Strength model exposing ``goal_rates`` and ``knockout_advance_prob``.
    groups : numpy.ndarray
        ``(N_GROUPS, GROUP_SIZE)`` array of global team indices.
    n_sims : int
        Number of independent replicates.
    rng : numpy.random.Generator
        Source of randomness (same stream as :func:`simulate_tournament`).

    Returns
    -------
    stages : numpy.ndarray
        ``(n_sims, N_TEAMS)`` ``int8`` furthest stage reached (identical to
        :func:`simulate_tournament`).
    group : GroupTallies
        ``{"wins", "draws"}`` group tallies.
    r32_pairs : numpy.ndarray
        ``(n_sims, 16, 2)`` global team indices of the two sides of each R32 match.
    round_winners : list of list of numpy.ndarray
        One entry per played knockout round in the order ``[R32, *KNOCKOUT_TREE]`` (i.e. R32
        producing the R16 occupants, then R16->QF, QF->SF, SF->Final). Entry ``r`` is the list of
        ``(n_sims,)`` winner arrays of that round -- the occupants of the *next* stage.
    """
    pts, gd, gf, wins, draws = simulate_group_stage(model, groups, n_sims, rng)
    lots = rng.random((n_sims, N_TEAMS))
    st = standings(pts, gd, gf, groups, lots)
    third_slots = _resolve_third_slots(st["third"], pts, gd, gf, lots)

    stages = np.zeros((n_sims, N_TEAMS), dtype=np.int8)
    rows = np.arange(n_sims)

    r32 = _r32_participants(st, third_slots)
    for a, b in r32:
        stages[rows, a] = Stage.R32
        stages[rows, b] = Stage.R32
    r32_pairs = np.stack([np.stack([a, b], axis=1) for a, b in r32], axis=1)  # (n_sims, 16, 2)

    winners = _play_round(model, r32, rng)
    round_winners: list[list[np.ndarray]] = [list(winners)]
    stage_value = int(Stage.R16)
    for w in winners:
        stages[rows, w] = stage_value
    for round_pairs in KNOCKOUT_TREE:
        stage_value += 1
        pairs = [(winners[i], winners[j]) for i, j in round_pairs]
        winners = _play_round(model, pairs, rng)
        round_winners.append(list(winners))
        for w in winners:
            stages[rows, w] = stage_value
    group: GroupTallies = {"wins": wins, "draws": draws}
    return stages, group, r32_pairs, round_winners


def replay_knockout_from_round(
    model,
    occupants: list[np.ndarray],
    start_stage: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample the knockout from a known set of occupants; return furthest stages.

    Validation-only. Given the teams occupying a stage's slots (``occupants`` -- e.g. the 32 R32
    participants for ``start_stage == Stage.R32``, the 16 R16 occupants for ``Stage.R16``, the 8 QF
    occupants for ``Stage.QF``, ...), play the remaining rounds with a *fresh* ``rng`` and return
    each occupant's resampled furthest :class:`~wcpool.ladders.Stage`. ``occupants[i]`` is an
    ``(n_inner,)`` array (the same team broadcast across inner replicates, or genuinely varying --
    the routine is agnostic).

    The bracket suffix is selected by ``start_stage`` over the round list ``[R32, *KNOCKOUT_TREE]``,
    where the R32 round pairs participants ``(0,1),(2,3),...`` so winner ``k`` lands in position
    ``k`` (matching how ``simulate_tournament`` feeds R32 winners into ``KNOCKOUT_TREE``). Starting
    at ``R32`` resamples the R32 round onward (the GROUP-boundary case); ``R16`` resamples the whole
    ``KNOCKOUT_TREE``; ``QF`` drops the first tree round; and so on. Occupants carry ``start_stage``
    as their floor; survivors are promoted one stage per round they win.

    Parameters
    ----------
    model
        Strength model exposing ``knockout_advance_prob``.
    occupants : list of numpy.ndarray
        Global team indices in bracket order at ``start_stage``; each entry shape ``(n_inner,)``.
        Length must be ``2 ** (number of rounds remaining + 1)``.
    start_stage : int
        The :class:`~wcpool.ladders.Stage` the occupants currently hold (``R32`` ... ``FINAL``).
    rng : numpy.random.Generator
        Fresh randomness for the resampled rounds.

    Returns
    -------
    numpy.ndarray
        ``(len(occupants), n_inner)`` furthest stage reached by each *bracket slot's* team; row
        ``i`` corresponds to ``occupants[i]``. Eliminated slots keep ``start_stage`` as their floor.
    """
    n_slots = len(occupants)
    occ = np.stack(occupants, axis=0)  # (n_slots, n_inner) team index per slot per replicate
    n_inner = occ.shape[1]
    inner_rows = np.arange(n_inner)
    furthest = np.full((n_slots, n_inner), int(start_stage), dtype=np.int8)

    # Round list keyed from R32: the R32 round (16 self-pairs of adjacent participants) followed by
    # KNOCKOUT_TREE. ``first_round = start_stage - R32`` selects the suffix to resample.
    r32_pairing = [(2 * k, 2 * k + 1) for k in range(len(R32_MATCHES))]  # 16 self-pairs
    all_rounds = [r32_pairing, *KNOCKOUT_TREE]
    first_round = int(start_stage) - int(Stage.R32)

    # ``pos_slot[p]`` = (n_inner,) original occupant index currently in bracket position ``p``;
    # ``occ[pos_slot[p], inner_rows]`` is that position's team. Slot identity is preserved across
    # rounds so ``furthest`` is indexed by the original occupant slot.
    pos_slot = [np.full(n_inner, i, dtype=np.int64) for i in range(n_slots)]
    stage_value = int(start_stage)
    for round_pairs in all_rounds[first_round:]:
        stage_value += 1
        next_pos_slot: list[np.ndarray] = []
        for i, j in round_pairs:
            slot_a, slot_b = pos_slot[i], pos_slot[j]  # each (n_inner,)
            team_a = occ[slot_a, inner_rows]
            team_b = occ[slot_b, inner_rows]
            p_adv = model.knockout_advance_prob(team_a, team_b)
            a_adv = rng.random(n_inner) < p_adv
            winner_slot = np.where(a_adv, slot_a, slot_b)
            furthest[winner_slot, inner_rows] = stage_value  # promote the surviving slot
            next_pos_slot.append(winner_slot)
        pos_slot = next_pos_slot
    return furthest


def _inner_credit_for_boundary(
    stages_s: np.ndarray,
    group_s: GroupTallies,
    schemes: list,
    rosters: np.ndarray,
    furthest: np.ndarray,
    teams: np.ndarray,
) -> list[np.ndarray]:
    """Inner-sim per-replicate win credit for one outer replicate/boundary, per scheme.

    Validation-only helper. Given already-replayed ``furthest`` stages (shape
    ``(n_slots, n_inner)`` from :func:`replay_knockout_from_round`), overlay them onto the realised
    ``stages_s`` (the eliminated/group layer is held fixed) and score the resulting fractional
    pool-win credit under EACH ``scheme`` in turn. The expensive knockout replay is scheme-
    independent (it produces furthest stages, not points), so it is done ONCE by the caller and the
    cheap per-scheme ``team_points`` scoring is looped here -- this lets one nested simulation serve
    several schemes (e.g. the low- and high-mix gate cells) without re-replaying. Imports of
    :mod:`wcpool.scoring`/:mod:`wcpool.metrics` are local so the single-pass hot path's module-level
    imports are untouched (no import cycle: neither imports this module).
    """
    from . import metrics, scoring  # local import: keep the hot-path import block unchanged

    n_inner = furthest.shape[1]
    inner_stages = np.broadcast_to(stages_s, (n_inner, N_TEAMS)).copy()
    for slot, team in enumerate(teams):
        inner_stages[:, int(team)] = furthest[slot]
    inner_group: GroupTallies = {
        "wins": np.broadcast_to(group_s["wins"], (n_inner, N_TEAMS)),
        "draws": np.broadcast_to(group_s["draws"], (n_inner, N_TEAMS)),
    }
    out = []
    for scheme in schemes:
        inner_scores = metrics.pool_scores(
            scoring.team_points(inner_stages, inner_group, scheme), rosters
        )
        out.append(metrics.terminal_win_credit(inner_scores))  # (n_inner, n_drafters)
    return out


def nested_conditional_win_prob(
    model,
    stages: np.ndarray,
    group: GroupTallies,
    r32_pairs: np.ndarray,
    round_winners: list[list[np.ndarray]],
    scheme,
    rosters: np.ndarray,
    subset: np.ndarray,
    n_inner: int,
    seed: int,
    *,
    cfg_id: int = 0,
    draw: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nested-simulation ground-truth ``Wcond`` for a subset of replicates (validation-only).

    For each outer replicate ``s`` in ``subset`` and each boundary ``b``, condition on the realised
    bracket through boundary ``b`` and Monte-Carlo the REMAINING knockout rounds (``n_inner`` inner
    draws), holding the realised group tallies fixed (the group layer is resolved at every knockout
    boundary). This targets the plan's definition ``P(d wins | all results through stage b)`` -- a
    RICHER conditioning set than the surrogate's drafter-score vector, hence the surrogate's
    documented one-sided downward bias. It is the section-6 gate's ground truth, NOT a hot-path
    routine (it is ``O(boundaries x n_inner)`` per replicate).

    Boundary handling (the two ends differ from the interior):

    * ``b == CHAMPION``: the whole tournament is resolved, so the running state *determines* the
      champion. The belief is the realised one-hot terminal credit -- exact, no inner sim.
    * ``b == FINAL``: the running state does **not** reveal the champion (both finalists sit at
      ``ko[FINAL]``; the champion bump is added only at CHAMPION). So the final is genuinely
      undecided here: Monte-Carlo the single remaining match between the two realised finalists
      (``round_winners[3]`` -- the SF->Final winners that feed the final) from ``start_stage =
      Stage.FINAL`` (exactly one round replayed), giving the blended belief. This fixes the
      FINAL-boundary over-conditioning that would otherwise force ``Wcond[:, :, FINAL]`` onto the
      realised one-hot credit and zero out the final's true suspense.
    * ``b == GROUP``: resample the R32 round onward from the 32 realised participants.
    * interior ``b``: occupants = ``round_winners[b - 1]`` (the occupants of stage ``b``); resample
      from ``b + 1``.

    **Finite-``n_inner`` bias correction.** The squared-increment estimator
    ``E||mu_b - mu_{b-1}||^2``
    used by the suspense/surprise kernel is inflated by ``Var(mu_hat_b) ~ 1/n_inner``. To support an
    *unbiased* cross-term estimate, each inner-sim boundary's belief is returned as TWO independent
    halves ``mu_hat_b^A``, ``mu_hat_b^B`` (the ``n_inner`` inner replicates partitioned into the
    first/last ``n_inner // 2``; the replay draws each inner replicate's outcome independently, so
    the two halves are independent given the realised conditioning bracket). The caller forms the
    unbiased squared increment ``<mu_hat_b^A - mu_hat_{b-1}, mu_hat_b^B - mu_hat_{b-1}>`` (or the
    fully-paired cross of both boundaries' halves); the plain mean of the two halves is the
    UPPER-biased plug-in. The exact (CHAMPION) and prior boundaries have zero inner-sim variance, so
    their two "halves" are identical. Use ``n_inner >= 800``.

    **Seeding (plan section 11).** Each ``(replicate, boundary)`` inner replay draws from the
    reserved 5-element slot ``spawn_key=(cfg_id, draw, 3, replicate_idx, boundary)`` (namespace 3,
    disjoint from the main grid's batches {0, 1, 2} and the synthetic namespace 7). The two halves
    are partitioned from this single per-boundary stream, so the slot stays 5-element (no separate
    half axis). ``cfg_id``/``draw`` are threaded in rather than hardcoded, so when this is wired
    into the experiment runner each cell/draw uses an independent inner stream; the validation test
    may pass ``cfg_id = draw = 0``.

    Parameters
    ----------
    model
        Strength model exposing ``knockout_advance_prob``.
    stages, group, r32_pairs, round_winners
        The realised-bracket trace from :func:`simulate_tournament_trace`.
    scheme : wcpool.scoring.ScoringScheme
        The scoring rule (drives ``team_points`` for the inner credit).
    rosters : numpy.ndarray
        ``(n_drafters, teams_per_drafter)`` global team indices each drafter holds.
    subset : numpy.ndarray
        Outer-replicate indices to evaluate the ground truth on (``>= 1000`` for the spec gate).
    n_inner : int
        Inner Monte-Carlo replicates per boundary (``>= 800``; split into two halves).
    seed : int
        Master seed for the inner ``SeedSequence``.
    cfg_id, draw : int, optional
        Configuration / draw indices threaded into the inner ``spawn_key`` (default ``0``).

    Returns
    -------
    wcn : numpy.ndarray
        ``(len(subset), n_drafters, N_STAGES)`` mean ground-truth belief per boundary (the
        UPPER-biased plug-in path; on the simplex; CHAMPION == realised credit).
    half_a, half_b : numpy.ndarray
        ``(len(subset), n_drafters, N_STAGES)`` the two independent-half belief estimates per
        boundary (identical to ``wcn`` at the exact CHAMPION boundary), for the unbiased cross-term.
    """
    wcn, ha, hb = nested_conditional_win_prob_multi(
        model, stages, group, r32_pairs, round_winners, [scheme], rosters, subset, n_inner, seed,
        cfg_id=cfg_id, draw=draw,
    )
    return wcn[0], ha[0], hb[0]


def nested_conditional_win_prob_multi(
    model,
    stages: np.ndarray,
    group: GroupTallies,
    r32_pairs: np.ndarray,
    round_winners: list[list[np.ndarray]],
    schemes: list,
    rosters: np.ndarray,
    subset: np.ndarray,
    n_inner: int,
    seed: int,
    *,
    cfg_id: int = 0,
    draw: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Multi-scheme ground-truth ``Wcond`` sharing one nested simulation (validation-only).

    Identical to :func:`nested_conditional_win_prob` but scores the SAME inner replays under each
    scheme in ``schemes``, returning parallel lists ``([wcn_i], [half_a_i], [half_b_i])``. The
    knockout replay (the expensive part) is scheme-independent -- it produces furthest stages, not
    points -- so replaying once and scoring per scheme keeps the gate's low- and high-mix cells on
    *exactly* the same realised inner brackets (a paired comparison) at the cost of one nested
    simulation rather than ``len(schemes)``. The inner ``spawn_key`` is the same 5-element
    ``(cfg_id, draw, 3, replicate_idx, boundary)`` slot (independent of the scheme), so a single
    scheme reproduces :func:`nested_conditional_win_prob` bit-for-bit.
    """
    from . import metrics, scoring  # local import: keep the hot-path import block unchanged

    n_d = rosters.shape[0]
    n_half = n_inner // 2
    final_st = int(Stage.FINAL)
    champ_st = int(Stage.CHAMPION)
    group_st = int(Stage.GROUP)
    ns = len(schemes)

    wcn = [np.zeros((len(subset), n_d, N_STAGES)) for _ in range(ns)]
    half_a = [np.zeros_like(wcn[0]) for _ in range(ns)]
    half_b = [np.zeros_like(wcn[0]) for _ in range(ns)]
    for ii, s in enumerate(subset):
        s = int(s)
        group_s: GroupTallies = {"wins": group["wins"][s], "draws": group["draws"][s]}
        for b in range(N_STAGES):
            if b == champ_st:  # whole tournament resolved -> realised one-hot credit (exact)
                group_row = {"wins": group["wins"][s][None], "draws": group["draws"][s][None]}
                for m, scheme in enumerate(schemes):
                    inner_scores = metrics.pool_scores(
                        scoring.team_points(stages[s][None], group_row, scheme), rosters
                    )
                    tc = metrics.terminal_win_credit(inner_scores)[0]
                    wcn[m][ii, :, b] = tc
                    half_a[m][ii, :, b] = tc
                    half_b[m][ii, :, b] = tc
                continue
            if b == group_st:  # resample R32 onward from the 32 participants
                teams = r32_pairs[s].reshape(-1).astype(np.int64)
                start = int(Stage.R32)
            elif b == final_st:  # Monte-Carlo the final between the two realised finalists
                rw = round_winners[3]  # SF->Final winners == the Final's two occupants
                teams = np.array([int(rw[k][s]) for k in range(len(rw))], dtype=np.int64)
                start = final_st
            else:  # occupants = winners of round b (stage-b occupants); resample from b + 1
                rw = round_winners[b - 1]
                teams = np.array([int(rw[k][s]) for k in range(len(rw))], dtype=np.int64)
                start = b + 1
            # One 5-element-slot stream per (replicate, boundary); n_inner inner replicates drawn
            # together, then partitioned into two independent halves. occupants[i] is the SAME team
            # broadcast across inner replicates (broadcast singletons), which the half partition and
            # the scheme-independent replay both rely on.
            rng_inner = np.random.default_rng(
                np.random.SeedSequence(seed, spawn_key=(int(cfg_id), int(draw), 3, s, b))
            )
            occupants = [np.full(n_inner, int(t)) for t in teams]
            furthest = replay_knockout_from_round(model, occupants, start, rng_inner)
            credits = _inner_credit_for_boundary(
                stages[s], group_s, schemes, rosters, furthest, teams
            )
            for m, credit in enumerate(credits):
                wcn[m][ii, :, b] = credit.mean(axis=0)
                half_a[m][ii, :, b] = credit[:n_half].mean(axis=0)
                half_b[m][ii, :, b] = credit[n_half : 2 * n_half].mean(axis=0)
    return wcn, half_a, half_b
