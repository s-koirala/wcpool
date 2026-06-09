"""Draft state, the per-pick recommendation, reach/wait, live standing, and summary.

The state is a single ordered list of picks (global team indices) in snake order. Because
the snake schedule is deterministic, ownership is auto-attributed from pick position -- the
user enters only the team as each pick happens, never who took it.

Ranking design (post-audit). Candidates are ranked by the objective ``W`` under the
**EV-greedy opponent baseline (temperature 0)** -- the regime under which ``wcpool``'s
exploitability probe validated EV-greedy as unbeatable, and the same regime whose P(1st)/
P(2nd) we display, so the sort key and the shown levels are one coherent quantity. The
temperature *sweep* (EV-greedy -> uniform) is used only to flag whether the top pick is
robust to opponent behaviour, never to average the magnitude. The τ=0 ranking is
deterministic (no RNG, candidate-order-independent) and fast.

Every recommendation is scored on the joint value board, so bracket collisions are priced
into ``W`` automatically (a roster that doubles up a bracket region scores a lower P(1st));
the collision flag merely names that already-priced effect.

Scoring scheme. The board's ``points`` carry the published recommendation
(``docs/tables/recommendation_2026-06-09.md``): the *triangular* knockout shape with a 3:1 group
win:draw layer at the gamma_match landmark (``ScoringScheme("triangular", 3, 1, mix=0.1)`` —
``draft_advisor.board.RECOMMENDED_SCHEME``). No logic here depends on the scheme: ``recommend`` /
``_score`` route entirely through ``board.points`` / ``board.team_ev``. The objective
``W = (n-1)*P(1st) + P(2nd)`` is moreover a *rank* functional of the per-drafter pool scores, hence
**affine-invariant** — under any ``p -> a*p + c`` (``a > 0``) with equal roster sizes (the snake
draft guarantees this) every drafter's score, and so the argmax/rank ``W`` reads, is unchanged.
Therefore scoring under that continuous blend yields pick advice *point-for-point identical* to the
published integer ladder [9, 27, 54, 90, 135, 189] with 3/1 group points (the two differ only by a
positive affine map): the live advisor matches the integer recommendation exactly.

Approximation (documented residual, see ``docs/acceptance_criteria.md`` A3): our own future
picks are completed EV-greedily inside candidate scoring, a one-ply proxy consistent with
``wcpool.draft.draft_best_response``. We re-optimise at each real turn, so displayed absolute
W/P1 are conservative; the ranking distortion is quantified in the acceptance doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wcpool import metrics as M
from wcpool.draft import snake_sequence

from . import objective, opponent
from .board import DEEP_STAGE, Board

# Opponent-completion Monte-Carlo reps per stochastic temperature, used ONLY for the
# robustness flag (the ranking itself is the deterministic τ=0 baseline). A modest count
# suffices: it gauges rank stability, not a precise magnitude.
DEFAULT_OPP_SAMPLES = 8
# Cap on how many top candidates the opponent-robustness sweep covers when no cliff bounds it.
# A candidate far down the τ=0 board will not become the top pick under another opponent model,
# so the realistic contenders for the #1 slot are the leaders; the cap also matches the CLI
# shortlist length.
ROBUSTNESS_CAP = 12
# Cap on how many next-tier teams the reach/wait look-ahead reports survival for.
NEXT_TIER_CAP = 8
# Disjoint RNG stream ids so the robustness sweep and the reach/wait look-ahead never draw the
# same sequence (passed as SeedSequence entropy, not arithmetic offsets).
_SWEEP_STREAM = 0
_REACH_STREAM = 1


@dataclass
class DraftState:
    """The live draft: who plays, our seat, and the ordered picks so far."""

    n_drafters: int
    n_rounds: int
    our_seat: int  # 0-indexed snake slot
    picks: list[int] = field(default_factory=list)  # global team indices in pick order

    def __post_init__(self) -> None:
        if not (0 <= self.our_seat < self.n_drafters):
            raise ValueError(f"our_seat {self.our_seat} out of range [0, {self.n_drafters})")
        if len(self.picks) > self.n_drafters * self.n_rounds:
            raise ValueError("more picks than the draft has slots")
        if len(set(self.picks)) != len(self.picks):
            raise ValueError("duplicate team in picks")

    # --- schedule ----------------------------------------------------------------------
    @property
    def total_slots(self) -> int:
        return self.n_drafters * self.n_rounds

    @property
    def sequence(self) -> np.ndarray:
        return snake_sequence(self.n_drafters, self.n_rounds)

    @property
    def is_complete(self) -> bool:
        return len(self.picks) >= self.total_slots

    def is_our_turn(self) -> bool:
        p = len(self.picks)
        return (not self.is_complete) and int(self.sequence[p]) == self.our_seat

    def our_pick_numbers(self) -> list[int]:
        """1-indexed overall pick numbers that belong to our seat."""
        return [i + 1 for i, d in enumerate(self.sequence) if d == self.our_seat]

    def picks_until_our_next(self) -> int | None:
        """Opponent picks between the slot about to be filled and our next pick.

        Counts slots strictly between position ``len(picks)`` and our next-after-that slot.
        At our turn this is the number of opponent picks after we pick; ``None`` if we have no
        further pick.
        """
        seq = self.sequence
        p = len(self.picks)
        for idx in range(p + 1, len(seq)):
            if int(seq[idx]) == self.our_seat:
                return idx - p - 1
        return None

    # --- ownership ---------------------------------------------------------------------
    def rosters(self) -> np.ndarray:
        """(n_drafters, n_rounds) of global team indices, -1 for unfilled slots.

        WARNING: a -1 must never reach ``wcpool.metrics.pool_scores`` (NumPy would alias it to
        the last team). Use the advisor's scoring paths, which complete the draft first; the
        internal scorer asserts a fully-filled roster.
        """
        return _rosters_from_picks(self.picks, self.n_drafters, self.n_rounds)

    def our_roster(self) -> list[int]:
        return self.roster_of(self.our_seat)

    def roster_of(self, drafter: int) -> list[int]:
        seq = self.sequence
        return [
            t for d, t in zip(seq[: len(self.picks)], self.picks, strict=True) if int(d) == drafter
        ]

    def available_mask(self, n_teams: int) -> np.ndarray:
        mask = np.ones(n_teams, dtype=bool)
        if self.picks:
            mask[self.picks] = False
        return mask


def _rosters_from_picks(picks: list[int], n_drafters: int, n_rounds: int) -> np.ndarray:
    seq = snake_sequence(n_drafters, n_rounds)
    rosters = np.full((n_drafters, n_rounds), -1, dtype=np.int64)
    counts = np.zeros(n_drafters, dtype=int)
    for d, t in zip(seq[: len(picks)], picks, strict=True):
        rosters[int(d), counts[int(d)]] = t
        counts[int(d)] += 1
    return rosters


def _score(board: Board, rosters: np.ndarray) -> np.ndarray:
    """Pool scores for fully-filled rosters; refuses any -1 to avoid silent team aliasing."""
    if (rosters < 0).any():
        raise ValueError("cannot score an incomplete roster (contains -1 slots)")
    return M.pool_scores(board.points, rosters)


def complete_and_score(
    picks_prefix: list[int], state: DraftState, board: Board, temperature: float, rng
) -> np.ndarray:
    """Complete the draft from ``picks_prefix`` and score every drafter on the board.

    Our seat and (at ``temperature == 0``) every opponent complete EV-greedily -- a single
    deterministic completion; at ``temperature > 0`` opponents sample the softmax-over-EV.
    Returns ``(n_sims, n_drafters)`` pool scores.
    """
    seq = snake_sequence(state.n_drafters, state.n_rounds)
    picks = list(picks_prefix)
    available = np.ones(board.n_teams, dtype=bool)
    if picks:
        available[picks] = False
    ev = board.team_ev
    for i in range(len(picks), len(seq)):
        avail_idx = np.where(available)[0]
        if int(seq[i]) == state.our_seat or temperature == 0:
            choice = opponent.greedy_pick(ev, avail_idx)
        else:
            choice = opponent.sample_pick(ev, avail_idx, temperature, rng)
        picks.append(choice)
        available[choice] = False
    return _score(board, _rosters_from_picks(picks, state.n_drafters, state.n_rounds))


def _forward_order(remaining: dict[int, int], n_drafters: int) -> list[int]:
    """Snake-ordered drafter ids for the remaining picks, given each drafter's remaining count.

    Position-agnostic: derived purely from how many picks each drafter still needs (read off
    their current roster sizes), so it does not depend on the live order matching the assumed
    snake schedule. Used when the order has drifted ("muffled") or a rec is requested off-turn.
    """
    rem = dict(remaining)
    order: list[int] = []
    fwd = True
    while sum(rem.values()) > 0:
        seq = range(n_drafters) if fwd else range(n_drafters - 1, -1, -1)
        for d in seq:
            if rem[d] > 0:
                order.append(d)
                rem[d] -= 1
        fwd = not fwd
    return order


def complete_for_our_pick(
    state: DraftState, board: Board, candidate: int, temperature: float, rng
) -> np.ndarray:
    """Score the pool with ``candidate`` assigned to OUR seat, completing the rest by remaining
    counts (snake-ordered). Position-agnostic, so it yields a recommendation even when the live
    order has drifted from the snake schedule."""
    n_d, n_r = state.n_drafters, state.n_rounds
    rosters_l = {d: list(state.roster_of(d)) for d in range(n_d)}
    rosters_l[state.our_seat].append(candidate)
    available = np.ones(board.n_teams, dtype=bool)
    for teams in rosters_l.values():
        for t in teams:
            available[t] = False
    remaining = {d: n_r - len(rosters_l[d]) for d in range(n_d)}
    ev = board.team_ev
    for d in _forward_order(remaining, n_d):
        avail_idx = np.where(available)[0]
        if avail_idx.size == 0:
            break
        if d == state.our_seat or temperature == 0:
            choice = opponent.greedy_pick(ev, avail_idx)
        else:
            choice = opponent.sample_pick(ev, avail_idx, temperature, rng)
        rosters_l[d].append(int(choice))
        available[int(choice)] = False
    rosters = np.full((n_d, n_r), -1, dtype=np.int64)
    for d in range(n_d):
        for i, t in enumerate(rosters_l[d]):
            rosters[d, i] = t
    return _score(board, rosters)


def _our_pick_scores(
    state: DraftState, board: Board, candidate: int, temperature: float, rng
) -> np.ndarray:
    """Pool scores with ``candidate`` as our next pick: exact snake completion at a genuine
    our-turn, else the position-agnostic completion (so a rec is always available)."""
    if state.is_our_turn():
        return complete_and_score(state.picks + [candidate], state, board, temperature, rng)
    return complete_for_our_pick(state, board, candidate, temperature, rng)


# --- recommendation --------------------------------------------------------------------


@dataclass
class CandidateRow:
    team: int
    name: str
    w0: float  # objective W at the EV-greedy (τ=0) baseline -- the sort key
    p1: float  # resulting P(1st) at τ=0
    p2: float  # resulting P(2nd) at τ=0
    p_money: float  # resulting P(top-2) at τ=0
    se: float  # Monte-Carlo SE of w0
    ceiling_value: float  # raw upper-tail (deep-run) contribution; banded at display time
    collision_partner: str | None  # owned team whose deep runs most trade off with this one
    collision_cov: float  # covariance with that partner (negative => collision)
    rank_by_temp: list[int]  # rank (1=best) across the swept temperatures (empty if not swept)
    robust_top: bool  # True iff swept and rank 1 at every temperature


@dataclass
class Recommendation:
    rows: list[CandidateRow]  # sorted best-first by w0
    temps: list[float]
    cliffs: list[int]  # positions i: a tier cliff sits between rows[i] and rows[i+1]
    reach_wait: dict | None  # survival look-ahead to our next pick (None if no next pick)


def _collision(board: Board, candidate: int, owned: list[int]) -> tuple[str | None, float]:
    """Owned team whose points covary most negatively with the candidate's (deep-run trade-off).

    Negative covariance means the two seldom go deep together (shared bracket region; they can
    eliminate each other). Sign-based, so no magic threshold. Already priced into ``W``.
    """
    if not owned:
        return None, 0.0
    pc = board.points[:, candidate]
    pc = pc - pc.mean()
    best_partner, best_cov = None, np.inf
    for o in owned:
        po = board.points[:, o]
        cov = float(np.mean(pc * (po - po.mean())))
        if cov < best_cov:
            best_cov, best_partner = cov, o
    return board.names[best_partner], best_cov


def recommend(
    state: DraftState,
    board: Board,
    r_samples: int = DEFAULT_OPP_SAMPLES,
    rng: np.random.Generator | None = None,
) -> Recommendation:
    """Rank every available team by ``W`` at the τ=0 baseline, with tiers, ceiling, look-ahead.

    Recommends our *next* pick. It is valid at a genuine our-turn (exact snake completion) and
    off-turn (position-agnostic completion), so it can be requested on demand even if the live
    order has drifted from the snake schedule. The ranking is deterministic; ``rng`` (and
    ``r_samples``) drive only the opponent-robustness sweep and the reach/wait look-ahead.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    temps = opponent.temperature_grid(board.team_ev)
    avail = np.where(state.available_mask(board.n_teams))[0]
    if avail.size == 0 or len(state.our_roster()) >= state.n_rounds:
        return Recommendation(rows=[], temps=temps, cliffs=[], reach_wait=None)
    ceil_all = objective.ceiling_deep_run(board.points, board.stages, DEEP_STAGE)
    owned = state.our_roster()

    # τ=0 baseline: one deterministic completion per candidate -> the sort key.
    base = []
    for c in avail:
        c = int(c)
        scores = _our_pick_scores(state, board, c, 0.0, rng)
        pp = objective.placement_probs(scores, state.our_seat)
        base.append(
            {
                "team": c,
                "w0": objective.objective_weight(state.n_drafters) * pp["p1"] + pp["p2"],
                "p1": pp["p1"],
                "p2": pp["p2"],
                "p_money": pp["p_money"],
                "se": objective.objective_W_se(scores, state.our_seat, state.n_drafters),
            }
        )
    base.sort(key=lambda e: e["w0"], reverse=True)

    cliffs = objective.detect_cliffs(
        np.array([e["w0"] for e in base]), np.array([e["se"] for e in base])
    )

    # Opponent-robustness sweep over the realistic contenders for the #1 slot (a capped
    # shortlist -- cliffs are display-only and do NOT drive this), common random numbers.
    m = max(min(len(base), ROBUSTNESS_CAP), 1)
    rank_by_temp = _robustness_sweep(state, board, [e["team"] for e in base[:m]], temps, r_samples)

    rows: list[CandidateRow] = []
    for e in base:
        partner, cov = _collision(board, e["team"], owned)
        ranks = rank_by_temp.get(e["team"], [])
        rows.append(
            CandidateRow(
                team=e["team"],
                name=board.names[e["team"]],
                w0=e["w0"],
                p1=e["p1"],
                p2=e["p2"],
                p_money=e["p_money"],
                se=e["se"],
                ceiling_value=float(ceil_all[e["team"]]),
                collision_partner=partner if cov < 0 else None,
                collision_cov=cov,
                rank_by_temp=ranks,
                robust_top=bool(ranks) and all(r == 1 for r in ranks),
            )
        )

    rw = _reach_wait(state, board, rows, temps, r_samples)
    return Recommendation(rows=rows, temps=temps, cliffs=cliffs, reach_wait=rw)


def _robustness_sweep(
    state: DraftState, board: Board, teams: list[int], temps: list[float], r_samples: int
) -> dict[int, list[int]]:
    """W of each shortlisted team at every temperature -> rank (1=best) per temperature.

    Common random numbers: at each (temperature, rep) all candidates are scored from the same
    seeded RNG state, so cross-candidate comparisons are not polluted by independent noise.
    """
    w_by_temp: dict[float, dict[int, float]] = {}
    for ti, tau in enumerate(temps):
        reps = 1 if tau == 0 else r_samples
        totals = dict.fromkeys(teams, 0.0)
        for rep in range(reps):
            # one seed per (temp, rep), reused across candidates -> common random numbers
            seed = np.random.SeedSequence([ti, rep, _SWEEP_STREAM])
            for t in teams:
                rng = np.random.default_rng(seed)
                scores = _our_pick_scores(state, board, t, tau, rng)
                totals[t] += objective.objective_W(scores, state.our_seat, state.n_drafters)
        w_by_temp[tau] = {t: totals[t] / reps for t in teams}

    ranks: dict[int, list[int]] = {t: [] for t in teams}
    for tau in temps:
        order = sorted(teams, key=lambda t: w_by_temp[tau][t], reverse=True)
        pos = {t: i + 1 for i, t in enumerate(order)}
        for t in teams:
            ranks[t].append(pos[t])
    return ranks


def _reach_wait(
    state: DraftState,
    board: Board,
    rows: list[CandidateRow],
    temps: list[float],
    r_samples: int,
) -> dict | None:
    """Probability each next-tier team survives to our next pick (now-vs-wait).

    After we take our #1, ``gap`` opponent picks occur before our next turn; we simulate them
    across the temperature sweep. ``verdict`` is driven by the survival probability of our best
    *alternative* (rows[1]): if it is more likely than not to be gone, take the scarce tier
    now. The reported set is the next ``NEXT_TIER_CAP`` candidates (not cliff-derived).
    """
    gap = state.picks_until_our_next()
    if not gap:  # None (no further pick) or 0 (back-to-back): no opponents to wait through
        return {"gap": gap or 0, "survival": {}, "verdict": "last pick / back-to-back"}
    top_k = [r.team for r in rows[1 : 1 + NEXT_TIER_CAP]]
    if not top_k:
        return {"gap": gap, "survival": {}, "verdict": "no alternatives"}
    base_prefix = state.picks + [rows[0].team]
    survive = dict.fromkeys(top_k, 0)
    trials = 0
    for ti, tau in enumerate(temps):
        reps = 1 if tau == 0 else r_samples
        for rep in range(reps):
            rng_rep = np.random.default_rng(np.random.SeedSequence([ti, rep, _REACH_STREAM]))
            available = np.ones(board.n_teams, dtype=bool)
            available[base_prefix] = False
            for _g in range(gap):
                avail_idx = np.where(available)[0]
                choice = (
                    opponent.greedy_pick(board.team_ev, avail_idx)
                    if tau == 0
                    else opponent.sample_pick(board.team_ev, avail_idx, tau, rng_rep)
                )
                available[choice] = False
            for t in top_k:
                survive[t] += int(available[t])
            trials += 1
    survival = {board.names[t]: survive[t] / trials for t in top_k}
    best_alt_survival = survive[top_k[0]] / trials
    # 0.5 = "more likely than not" gone -> scarcity verdict (definitional midpoint, not tuned)
    verdict = "take now (best alternative unlikely to survive)" if best_alt_survival < 0.5 else (
        "safe to wait (best alternative likely survives)"
    )
    return {"gap": gap, "survival": survival, "verdict": verdict}


# --- standing & summary ----------------------------------------------------------------


def standing(state: DraftState, board: Board) -> dict:
    """Our projected P(1st)/P(top-2) at the τ=0 baseline, plus a PUSH/PROTECT mode.

    Uses the same EV-greedy baseline as the ranking, so the figures reconcile with the
    recommendation panel. Mode is a relative display aid (the optimiser always maximises ``W``):
    PROTECT when our projected P(top-2) already leads the field (the floor is the binding
    margin); PUSH otherwise (we still need ceiling to climb).
    """
    scores = complete_and_score(state.picks, state, board, 0.0, np.random.default_rng(0))
    pp = objective.placement_probs(scores, state.our_seat)
    money = np.array(
        [objective.placement_probs(scores, d)["p_money"] for d in range(state.n_drafters)]
    )
    leader = int(np.argmax(money)) == state.our_seat
    return {"p1": pp["p1"], "p2": pp["p2"], "p_money": pp["p_money"],
            "mode": "PROTECT" if leader else "PUSH"}


def post_draft_summary(state: DraftState, board: Board) -> dict:
    """Final-roster report: placements, score distribution, swing team, bracket map.

    Deterministic on the realised rosters. ``swing_team`` is the team whose points most
    strongly co-move with our finishing 1st (Pearson correlation with our 1st-place credit).
    """
    if not state.is_complete:
        raise ValueError("post_draft_summary() requires a complete draft")
    scores = _score(board, state.rosters())
    pp = objective.placement_probs(scores, state.our_seat)
    our = state.our_roster()
    our_score = board.points[:, our].sum(axis=1)

    c1 = objective.first_place_credit(scores, state.our_seat)
    swing_team, swing_corr = None, -np.inf
    for t in our:
        pt = board.points[:, t]
        denom = pt.std() * c1.std()
        corr = float(np.mean((pt - pt.mean()) * (c1 - c1.mean())) / denom) if denom > 0 else 0.0
        if corr > swing_corr:
            swing_corr, swing_team = corr, t

    return {
        "roster": [board.names[t] for t in our],
        "roster_groups": [chr(ord("A") + int(board.group_of[t])) for t in our],
        "p1": pp["p1"],
        "p2": pp["p2"],
        "p_money": pp["p_money"],
        "score_mean": float(our_score.mean()),
        "score_p05": float(np.percentile(our_score, 5)),
        "score_p50": float(np.percentile(our_score, 50)),
        "score_p95": float(np.percentile(our_score, 95)),
        "swing_team": board.names[swing_team] if swing_team is not None else None,
        "swing_corr": float(swing_corr),
    }
