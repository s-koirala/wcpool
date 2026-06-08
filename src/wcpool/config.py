"""Load the team field (ratings, pots, official draw) from the YAML config."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .tournament import GROUP_SIZE, N_GROUPS, N_TEAMS


@dataclass
class Field:
    """The 48-team field: names, Elo ratings, and the official pot/group structure."""

    names: list[str]
    elo: np.ndarray  # (48,)
    group_of: np.ndarray  # (48,) group index 0..11
    pot_of: np.ndarray  # (48,) pot index 0..3
    sources: list[str]
    config_sha256: str

    @property
    def fixed_groups(self) -> np.ndarray:
        """Official draw as (12, 4) of global team indices, position == pot index."""
        groups = np.full((N_GROUPS, GROUP_SIZE), -1, dtype=np.int64)
        for t in range(N_TEAMS):
            groups[self.group_of[t], self.pot_of[t]] = t
        if (groups < 0).any():
            raise ValueError("incomplete draw: some (group, pot) cell is unfilled")
        return groups

    @property
    def pots(self) -> np.ndarray:
        """Pots as (4, 12) of global team indices (pot x group). Feeds random draws."""
        return self.fixed_groups.T.copy()

    @property
    def elo_spread(self) -> float:
        """Population SD (ddof=0) of the field's Elo ratings; the synthetic-sweep anchor.

        ddof=0 is used because the 48-team field is the whole population of interest here,
        not a sample from a larger one; the sweep multiplies this by a documented range.
        """
        return float(np.std(self.elo))


def load_field(path: str | Path) -> Field:
    raw = Path(path).read_bytes()
    config_sha256 = hashlib.sha256(raw).hexdigest()
    data = yaml.safe_load(raw)
    teams = data["teams"]
    if len(teams) != N_TEAMS:
        raise ValueError(f"expected {N_TEAMS} teams, got {len(teams)}")

    names = [t["name"] for t in teams]
    elo = np.array([float(t["elo"]) for t in teams])
    group_of = np.array([ord(t["group"]) - ord("A") for t in teams])
    pot_of = np.array([int(t["pot"]) - 1 for t in teams])
    sources = [t.get("source", "unknown") for t in teams]

    # Structural validation: every group has exactly one team from each pot.
    for g in range(N_GROUPS):
        pots_in_group = sorted(pot_of[group_of == g].tolist())
        if pots_in_group != list(range(GROUP_SIZE)):
            raise ValueError(f"group {chr(ord('A') + g)} pot composition invalid: {pots_in_group}")

    return Field(
        names=names,
        elo=elo,
        group_of=group_of,
        pot_of=pot_of,
        sources=sources,
        config_sha256=config_sha256,
    )
