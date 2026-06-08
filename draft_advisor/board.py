"""Value board: the precomputed Monte-Carlo outcome distribution our picks are scored on.

For the *design* study, ``wcpool`` resamples the group draw every replicate to describe the
design rather than one bracket. For OUR live draft the groups are already known (the official
2025-12-05 draw, in ``config/ratings_elo_2026.yaml``), so the board is computed on the
**fixed official draw**: every simulation plays the real 2026 bracket. This is what makes the
bracket-collision structure real -- two teams' actual potential meeting round is baked into
the joint distribution, so a roster that doubles up a bracket region prices out automatically
as a lower P(1st).

The board is expensive (one full tournament batch) but is computed **once, offline**, and
cached. At the table a recommendation runs a few hundred roster scorings over this board (a
few seconds at the default sim count), with no network -- fast enough for live use, though not
the "microseconds per roster" of a single array-sum. The cache is written atomically and a
reproducibility record is written to ``logs/reproducibility/`` before the cache is saved.

Default simulation count is justified by the Monte-Carlo precision of the quantity actually
ranked, the objective ``W``: see :data:`DEFAULT_N_SIMS`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from wcpool.config import load_field
from wcpool.ladders import Stage, get_ladder
from wcpool.strength import StrengthModel
from wcpool.tournament import random_pot_draw, simulate_tournament

# One draw of the official bracket -> all sims share one group assignment, so a large batch is
# cheap. The advisor ranks on W = (n-1)*P1 + P2; with n=8, a contender p1 ~ 0.15 and
# S = 100_000, the MC SE of W is ~ (n-1)*sqrt(p1(1-p1)/S) ~ 7*0.00113 ~ 0.008, well under the
# ~0.05-0.35 W-gaps between draftable tiers, so 100k resolves the ranking comfortably.
DEFAULT_N_SIMS = 100_000
# Fixed RNG seed for the board (recorded in the repro stamp; override per run if desired).
DEFAULT_SEED = 20260608

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FIELD = _REPO_ROOT / "config" / "ratings_elo_2026.yaml"
_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_REPRO_DIR = _REPO_ROOT / "logs" / "reproducibility"

DEEP_STAGE = int(Stage.SF)  # "ceiling" counts points from the semi-final onward


@dataclass
class Board:
    """Precomputed value board for one (field, ladder, n_sims, seed, draw) combination."""

    points: np.ndarray  # (n_sims, 48) float64 -- terminal points each team earned
    stages: np.ndarray  # (n_sims, 48) int8   -- furthest stage each team reached
    team_ev: np.ndarray  # (48,)              -- mean points per team
    names: list[str]
    group_of: np.ndarray  # (48,) group index 0..11
    ladder: str
    n_sims: int
    seed: int
    use_official_draw: bool
    config_sha256: str
    board_sha256: str

    @property
    def n_teams(self) -> int:
        return self.points.shape[1]

    def index_of(self, name: str) -> int:
        """Global team index for an exact team name (case-insensitive)."""
        lower = [s.lower() for s in self.names]
        return lower.index(name.lower())


def _board_sha256(points: np.ndarray, stages: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(points).tobytes())
    h.update(np.ascontiguousarray(stages).tobytes())
    return h.hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _pip_freeze_sha256() -> str:
    """SHA-256 of the project venv's frozen dependency set (env fingerprint)."""
    for cmd in (["uv", "pip", "freeze"], ["python", "-m", "pip", "freeze"]):
        try:
            out = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
            return hashlib.sha256(out.stdout.encode()).hexdigest()
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    return "unknown"


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def build_board(
    field_path: str | Path = _DEFAULT_FIELD,
    ladder: str = "triangular",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    use_official_draw: bool = True,
) -> Board:
    """Simulate the tournament ``n_sims`` times and assemble the value board.

    ``ladder`` defaults to ``triangular`` (the report's recommended "Building" ladder).
    ``use_official_draw`` keeps the real 2026 groups fixed across sims (the correct basis for
    a live draft); ``False`` resamples groups (matches the design study, for testing).
    """
    field = load_field(field_path)
    model = StrengthModel(field.elo.copy())
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    groups = field.fixed_groups if use_official_draw else random_pot_draw(field.pots, rng)
    stages = simulate_tournament(model, groups, n_sims, rng)
    points = get_ladder(ladder)[stages]
    return Board(
        points=points,
        stages=stages,
        team_ev=points.mean(axis=0),
        names=field.names,
        group_of=field.group_of,
        ladder=ladder,
        n_sims=n_sims,
        seed=seed,
        use_official_draw=use_official_draw,
        config_sha256=field.config_sha256,
        board_sha256=_board_sha256(points, stages),
    )


def _cache_path(ladder: str, n_sims: int, seed: int, official: bool) -> Path:
    tag = "official" if official else "resampled"
    return _ARTIFACTS / f"board_{ladder}_{tag}_n{n_sims}_s{seed}.npz"


def _write_repro_log(board: Board, cache_path: Path) -> Path:
    """Write a reproducibility record next to the board cache (CLAUDE.md mandate).

    Captures git HEAD (code/model commit), pip-freeze hash + lock fingerprint (environment),
    field-config checksum + board content hash (data), RNG seed, sim count, library versions,
    host, and a UTC timestamp -- the five mandated anchors plus replay context.
    """
    _REPRO_DIR.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    try:  # repo-relative path only -> never commit an OS username / home directory
        cache_rel = Path(cache_path).resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        cache_rel = Path(cache_path).name
    record = {
        "artifact": "draft_advisor.board",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_head": head,
        "model_hash": head,  # the model is defined by the engine code at this commit
        "pip_freeze_sha256": _pip_freeze_sha256(),
        "env_id": _file_sha256(_REPO_ROOT / "uv.lock"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "host": platform.platform(),
        "python": platform.python_version(),
        "dataset_checksums": {
            "field_config_sha256": board.config_sha256,
            "board_sha256": board.board_sha256,
        },
        "rng_seed": board.seed,
        "ladder": board.ladder,
        "n_sims": board.n_sims,
        "use_official_draw": board.use_official_draw,
        "cache_path": cache_rel,
    }
    out = _REPRO_DIR / f"repro_board_{board.board_sha256}.json"
    out.write_text(json.dumps(record, indent=2))
    return out


def save_board(board: Board) -> Path:
    """Persist a board to ``artifacts/`` atomically, after writing its repro record."""
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(board.ladder, board.n_sims, board.seed, board.use_official_draw)
    _write_repro_log(board, cache_path)  # repro record BEFORE the artifact write
    meta = {
        k: getattr(board, k)
        for k in ("ladder", "n_sims", "seed", "use_official_draw", "config_sha256", "board_sha256")
    }
    tmp = tempfile.NamedTemporaryFile(dir=_ARTIFACTS, suffix=".npz.tmp", delete=False)
    try:
        np.savez_compressed(
            tmp,
            points=board.points,
            stages=board.stages,
            team_ev=board.team_ev,
            group_of=board.group_of,
            names=np.array(board.names),
            meta=json.dumps(meta),
        )
        tmp.close()
        os.replace(tmp.name, cache_path)  # atomic publish: no partial cache is ever visible
    except BaseException:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return cache_path


def load_board(cache_path: str | Path) -> Board:
    """Load a board previously written by :func:`save_board`."""
    data = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return Board(
        points=data["points"],
        stages=data["stages"],
        team_ev=data["team_ev"],
        names=[str(s) for s in data["names"]],
        group_of=data["group_of"],
        **meta,
    )


def load_or_build(
    ladder: str = "triangular",
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    use_official_draw: bool = True,
    field_path: str | Path = _DEFAULT_FIELD,
) -> Board:
    """Return a cached board if present, valid, and current, else build, cache, and return it.

    A cache that is unreadable (e.g. a partial write) or stale (built against a different
    field config) is ignored and rebuilt, so a corrupt cache can never poison the session.
    """
    cache_path = _cache_path(ladder, n_sims, seed, use_official_draw)
    if cache_path.exists():
        try:
            board = load_board(cache_path)
            if board.config_sha256 == load_field(field_path).config_sha256:
                return board
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # corrupt (incl. truncated .npz -> BadZipFile) / stale -> rebuild
    board = build_board(field_path, ladder, n_sims, seed, use_official_draw)
    save_board(board)
    return board
