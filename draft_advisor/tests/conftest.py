import numpy as np
import pytest

from draft_advisor import board as B


@pytest.fixture
def make_board():
    """Factory for a small synthetic :class:`~draft_advisor.board.Board` for fast logic tests."""

    def _make(points, stages=None, names=None, group_of=None):
        points = np.asarray(points, dtype=float)
        n_sims, n_teams = points.shape
        if stages is None:
            mx = points.max() or 1.0
            stages = np.clip(np.round(points / mx * 6).astype(np.int8), 0, 6)
        if names is None:
            names = [f"T{i}" for i in range(n_teams)]
        if group_of is None:
            group_of = np.arange(n_teams) % 12
        return B.Board(
            points=points,
            stages=np.asarray(stages, dtype=np.int8),
            team_ev=points.mean(axis=0),
            names=list(names),
            group_of=np.asarray(group_of),
            ladder="test",
            n_sims=n_sims,
            seed=0,
            use_official_draw=True,
            config_sha256="x",
            board_sha256="y",
        )

    return _make
