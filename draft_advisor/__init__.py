"""Live draft-pick advisor for the 2026 World Cup team-draft pool.

A thin decision layer over the ``wcpool`` Monte-Carlo engine. The engine answers a
*design* question (which ladder/N a pool organiser should pick); this package answers a
*participant* question: at our turn in the snake draft, which available team maximises our
own probability of winning the pool under the payout we actually face.

Payout (settled): 1st place wins, 2nd breaks even, everyone else loses the buy-in. The
objective that follows from it is ``W = (n-1)*P(1st) + P(2nd)`` (derivation in
``docs/objective_derivation.md``).

Modules
-------
* :mod:`draft_advisor.board`     -- precompute/cache the value board (points per team per
  simulated tournament) on the *official* 2026 draw, with a reproducibility stamp.
* :mod:`draft_advisor.objective` -- placement probabilities, the objective ``W``, ceiling
  (upper-tail contribution), Monte-Carlo standard error, and data-driven tier-cliff
  detection.
* :mod:`draft_advisor.opponent`  -- the softmax-over-value opponent model and the
  temperature sweep (EV-greedy <-> uniform) used for robustness.
* :mod:`draft_advisor.advisor`   -- draft state (snake auto-attribution), the per-pick
  recommendation, reach/wait look-ahead, live standing, and post-draft summary.
* :mod:`draft_advisor.cli`       -- the live REPL implementing the [A]-[J] interface.
"""

from __future__ import annotations

__all__ = ["advisor", "board", "cli", "objective", "opponent"]
