"""Opponent model: how co-drafters pick, and the robustness sweep over their behaviour.

We do not know how the other drafters behave, and their picks are highly variable. We
therefore model an opponent pick as a **softmax over the value board** with a temperature
``tau`` (in the same units as ``team_ev``):

    P(team t) proportional to exp(team_ev[t] / tau)   over available teams.

The two endpoints are the decision-relevant extremes, and the recommendation is reported as
robust across the whole grid:

* ``tau -> 0``   : argmax -- a perfectly EV-rational field (the wcpool engine's baseline,
  and the regime under which its exploitability probe found EV-greedy unbeatable).
* ``tau -> inf`` : uniform -- a maximally unpredictable field.

Because we log every pick as it happens, the opponent model is only ever used for the short
LOOK-AHEAD to our next pick (reach/wait) and to complete the modelled draft when scoring a
candidate -- never to reconstruct the past. The interior grid points are spaced at multiples
of the EV spread (data-scaled, not absolute magic numbers); exact interior spacing does not
drive the answer, since we report the recommendation across the full sweep.
"""

from __future__ import annotations

import numpy as np

# Interior temperature grid points, as multiples of the value board's standard deviation.
# The endpoints 0 (argmax) and inf (uniform) are added by ``temperature_grid``. The interior
# multiples span the argmax->uniform path; the recommendation's robustness is reported across
# the whole grid, so these are samples of a path, not tuned thresholds.
SIGMA_MULTIPLES = (0.5, 1.0, 2.0)


def softmax_pick_probs(ev_available: np.ndarray, temperature: float) -> np.ndarray:
    """Pick-probability over available teams given their EV and a temperature.

    ``temperature == 0`` returns a one-hot argmax (ties split equally); ``temperature`` inf
    returns the uniform distribution.
    """
    ev = np.asarray(ev_available, dtype=float)
    if ev.size == 0:
        raise ValueError("no available teams to pick from")
    if np.isinf(temperature):
        return np.full(ev.size, 1.0 / ev.size)
    if temperature <= 0:
        top = ev == ev.max()
        return top / top.sum()
    z = ev / temperature
    z -= z.max()  # numerical stability
    e = np.exp(z)
    return e / e.sum()


def sample_pick(
    team_ev: np.ndarray, available_idx: np.ndarray, temperature: float, rng: np.random.Generator
) -> int:
    """Sample one opponent pick (a global team index) from the softmax over availables."""
    p = softmax_pick_probs(team_ev[available_idx], temperature)
    return int(rng.choice(available_idx, p=p))


def greedy_pick(team_ev: np.ndarray, available_idx: np.ndarray) -> int:
    """The EV-greedy pick (global team index): argmax team_ev among availables."""
    return int(available_idx[np.argmax(team_ev[available_idx])])


def temperature_grid(team_ev: np.ndarray) -> list[float]:
    """The robustness sweep: ``[0, 0.5*sd, 1*sd, 2*sd, inf]`` in EV units.

    ``0`` is EV-greedy (argmax) and ``inf`` is uniform -- the two extremes we want the
    recommendation to be robust across. Interior points scale with the EV spread so the grid
    is meaningful regardless of ladder.
    """
    sd = float(np.std(team_ev))
    interior = [m * sd for m in SIGMA_MULTIPLES] if sd > 0 else []
    return [0.0, *interior, np.inf]
