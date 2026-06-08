"""Snake-draft execution and the three draft policies.

Six drafters pick teams in *snake* order (1..6, 6..1, 1..6, ...) for ``teams_per_drafter``
rounds. A policy decides which available team an acting drafter takes:

* ``ev_greedy``      : take the highest expected-points team available (the default).
* ``variance``       : take the highest points-*variance* team available (variance-seeking).
* ``best_response``   : one designated drafter takes the team maximising its own pool-win
  probability assuming every remaining pick (its own and the others') is filled
  EV-greedily; the other five drafters play EV-greedy. This is a one-ply greedy
  best-response used to test whether EV-greedy is exploitable.

A draft returns ``rosters`` (n_drafters, teams_per_drafter) of global team indices and
``drafter_of_team`` (n_teams,), with -1 for any undrafted team.
"""

from __future__ import annotations

import numpy as np

from .metrics import pool_scores, win_probability

N_DRAFTERS = 6


def snake_sequence(n_drafters: int, n_rounds: int) -> np.ndarray:
    """Pick order as a flat array of drafter ids, length ``n_drafters * n_rounds``."""
    base = np.arange(n_drafters)
    seq = []
    for r in range(n_rounds):
        seq.append(base if r % 2 == 0 else base[::-1])
    return np.concatenate(seq)


def _assign(seq: np.ndarray, picks: list[int], n_drafters: int, n_rounds: int) -> dict:
    rosters = np.full((n_drafters, n_rounds), -1, dtype=np.int64)
    counts = np.zeros(n_drafters, dtype=np.int64)
    for drafter, team in zip(seq, picks, strict=True):
        rosters[drafter, counts[drafter]] = team
        counts[drafter] += 1
    return {"rosters": rosters}


def _greedy_picks(
    key: np.ndarray, seq: np.ndarray, available: np.ndarray, start: int, picks: list[int]
) -> list[int]:
    """Fill picks from index ``start`` onward by taking argmax ``key`` among available."""
    picks = list(picks)
    candidates = np.where(available)[0]
    for _ in range(start, len(seq)):
        # argmax key over currently available teams
        best = candidates[np.argmax(key[candidates])]
        picks.append(int(best))
        candidates = candidates[candidates != best]
    return picks


def draft_by_key(key: np.ndarray, n_drafters: int, n_rounds: int) -> dict:
    """Snake draft where every drafter greedily maximises a fixed per-team ``key``."""
    n_teams = len(key)
    seq = snake_sequence(n_drafters, n_rounds)
    available = np.ones(n_teams, dtype=bool)
    picks = _greedy_picks(key, seq, available, 0, [])
    out = _assign(seq, picks, n_drafters, n_rounds)
    out["drafter_of_team"] = _drafter_of_team(out["rosters"], n_teams)
    return out


def _drafter_of_team(rosters: np.ndarray, n_teams: int) -> np.ndarray:
    owner = np.full(n_teams, -1, dtype=np.int64)
    for d in range(rosters.shape[0]):
        for t in rosters[d]:
            if t >= 0:
                owner[t] = d
    return owner


def draft_ev_greedy(team_ev: np.ndarray, n_drafters: int, n_rounds: int) -> dict:
    """Default policy: every drafter takes the highest-EV available team."""
    return draft_by_key(team_ev, n_drafters, n_rounds)


def draft_variance(team_var: np.ndarray, n_drafters: int, n_rounds: int) -> dict:
    """Variance-seeking policy: every drafter takes the highest points-variance team."""
    return draft_by_key(team_var, n_drafters, n_rounds)


def draft_best_response(
    points: np.ndarray,
    team_ev: np.ndarray,
    br_slot: int,
    n_drafters: int,
    n_rounds: int,
) -> dict:
    """One drafter (``br_slot``) best-responds to EV-greedy opponents.

    At each of the best-responder's picks, every available team is trialled: assign it,
    complete the entire remaining draft EV-greedily (all drafters), score the pool over
    ``points`` and keep the team that maximises the best-responder's win probability.
    Opponents' own picks use EV-greedy throughout.

    ``points`` must be the best-responder's *belief* about outcomes (the model/EV-batch
    sample), NOT the held-out tournaments it is later scored on — otherwise the policy
    optimises in-sample and the measured exploitability is spurious. The caller scores the
    returned roster on an independent eval batch.
    """
    n_teams = len(team_ev)
    seq = snake_sequence(n_drafters, n_rounds)
    available = np.ones(n_teams, dtype=bool)
    picks: list[int] = []

    for i, drafter in enumerate(seq):
        avail_idx = np.where(available)[0]
        if drafter != br_slot:
            choice = int(avail_idx[np.argmax(team_ev[avail_idx])])
        else:
            best_choice, best_wp = None, -1.0
            for cand in avail_idx:
                trial_avail = available.copy()
                trial_avail[cand] = False
                completed = _greedy_picks(team_ev, seq, trial_avail, i + 1, picks + [int(cand)])
                rosters = _assign(seq, completed, n_drafters, n_rounds)["rosters"]
                wp = win_probability(pool_scores(points, rosters))[br_slot]
                if wp > best_wp:
                    best_wp, best_choice = wp, int(cand)
            choice = best_choice
        picks.append(choice)
        available[choice] = False

    out = _assign(seq, picks, n_drafters, n_rounds)
    out["drafter_of_team"] = _drafter_of_team(out["rosters"], n_teams)
    out["br_slot"] = br_slot
    return out
