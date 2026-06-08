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

from .ladders import Stage

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate every group match; return ``(points, gd, gf)`` arrays shape (n_sims, 48)."""
    pts = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    gf = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
    ga = np.zeros((n_sims, N_TEAMS), dtype=np.int32)
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
    return pts, gf - ga, gf


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
    model, groups: np.ndarray, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Run ``n_sims`` full tournaments on a fixed ``groups`` draw.

    Returns an ``(n_sims, 48)`` int array of the furthest ``Stage`` each team reached.
    """
    pts, gd, gf = simulate_group_stage(model, groups, n_sims, rng)
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
    return stages
