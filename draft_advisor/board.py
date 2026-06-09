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
from wcpool.ladders import Stage
from wcpool.scoring import ScoringScheme, team_points
from wcpool.strength import StrengthModel
from wcpool.tournament import random_pot_draw, simulate_tournament

# One draw of the official bracket -> all sims share one group assignment, so a large batch is
# cheap. The advisor ranks on W = (n-1)*P1 + P2; with n=8, a contender p1 ~ 0.15 and
# S = 100_000, the MC SE of W is ~ (n-1)*sqrt(p1(1-p1)/S) ~ 7*0.00113 ~ 0.008, well under the
# ~0.05-0.35 W-gaps between draftable tiers, so 100k resolves the ranking comfortably.
DEFAULT_N_SIMS = 100_000
# Fixed RNG seed for the board (recorded in the repro stamp; override per run if desired).
DEFAULT_SEED = 20260608

# The published group-stage scoring recommendation (docs/tables/recommendation_2026-06-09.md):
# the *triangular* knockout shape with a 3:1 group win:draw layer at the commensurability
# landmark gamma_match -- whose convex weight is mix = 1 / (3*w_pts + 1) = 1 / (3*3 + 1) = 0.1
# (realised group share gamma = 0.1574). The advisor's objective W = (n-1)*P(1st) + P(2nd) is a
# rank functional of the per-drafter pool scores, hence *affine-invariant*: with equal roster
# sizes (the snake draft guarantees this), scoring the board under this ScoringScheme -- a convex
# blend (1-mix)*A(stage) + mix*(3*wins + 1*draws) -- gives pick advice POINT-FOR-POINT IDENTICAL
# to the published integer ladder [9, 27, 54, 90, 135, 189] with 3/1 group points. That ladder is
# `9 * triangular + 3*wins + 1*draws`, and the live blend equals `0.1 *` it exactly -- the two are
# one scoring rule up to a positive affine map a*p + c, under which every drafter's score (and so
# the argmax/rank W reads) is unchanged. So the live advisor matches the integer recommendation.
RECOMMENDED_SCHEME = ScoringScheme("triangular", w_pts=3.0, d_pts=1.0, mix=0.1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FIELD = _REPO_ROOT / "config" / "ratings_elo_2026.yaml"
_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_REPRO_DIR = _REPO_ROOT / "logs" / "reproducibility"

DEEP_STAGE = int(Stage.SF)  # "ceiling" counts points from the semi-final onward


@dataclass
class Board:
    """Precomputed value board for one (field, scheme, n_sims, seed, draw) combination.

    The board is scored under a :class:`~wcpool.scoring.ScoringScheme` — the two-layer
    ``(1 - mix)*A(stage) + mix*(w_pts*wins + d_pts*draws)`` rule — carried here as its four
    scalar fields ``(knockout_ladder, w_pts, d_pts, mix)`` rather than the bare ladder name, so a
    cache built under a *different* scheme (e.g. the old terminal ``mix == 0`` scoring) is detected
    as stale and rebuilt. ``mix == 0`` recovers the legacy terminal-ladder board exactly.
    """

    points: np.ndarray  # (n_sims, 48) float64 -- two-layer points each team earned
    stages: np.ndarray  # (n_sims, 48) int8   -- furthest stage each team reached
    team_ev: np.ndarray  # (48,)              -- mean points per team
    names: list[str]
    group_of: np.ndarray  # (48,) group index 0..11
    knockout_ladder: str  # advancement-shape name (the scheme's ladder)
    w_pts: float  # group points per win
    d_pts: float  # group points per draw
    mix: float  # convex weight on the group layer (0 == legacy terminal scoring)
    n_sims: int
    seed: int
    use_official_draw: bool
    config_sha256: str
    board_sha256: str

    @property
    def n_teams(self) -> int:
        return self.points.shape[1]

    @property
    def ladder(self) -> str:
        """The knockout advancement shape (back-compat alias for displays/old call sites)."""
        return self.knockout_ladder

    @property
    def scheme(self) -> ScoringScheme:
        """Reconstruct the :class:`~wcpool.scoring.ScoringScheme` this board was scored under."""
        return ScoringScheme(self.knockout_ladder, self.w_pts, self.d_pts, self.mix)

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


def _resolve_scheme(scheme: ScoringScheme | None, ladder: str | None) -> ScoringScheme:
    """Pick the board's scoring scheme from the (scheme, ladder) call signature.

    Precedence, so both the new scheme axis and the legacy ``ladder=`` knob keep working:

    * an explicit ``scheme`` is used verbatim (the new, group-stage-aware path);
    * else an explicit ``ladder`` maps to the *terminal* ``ScoringScheme(ladder, 1, 0, mix=0)`` —
      the pre-refactor behaviour (group layer absent), so ``build_board(ladder="linear")`` still
      yields the old bare-ladder board ``get_ladder("linear")[stages]`` bit-for-bit;
    * else the published :data:`RECOMMENDED_SCHEME` (triangular 3:1 @ gamma_match).
    """
    if scheme is not None:
        return scheme
    if ladder is not None:
        return ScoringScheme(ladder, w_pts=1.0, d_pts=0.0, mix=0.0)
    return RECOMMENDED_SCHEME


def _canonical_scheme(scheme: ScoringScheme) -> ScoringScheme:
    """Canonical cache-identity of a scheme: collapse every ``mix == 0`` scheme to one rep.

    At ``mix == 0`` the group layer vanishes (:func:`wcpool.scoring.team_points` short-circuits to
    the bare ``ko_vec[stages]``), so ``(w_pts, d_pts)`` never enter the board's ``points`` — every
    ``mix == 0`` scheme on a given ladder produces a bit-identical board. :func:`_scheme_tag` folds
    them all onto the bare-ladder filename already, but ``save_board``/``load_or_build`` compare the
    *stored* scheme field-by-field, so a non-canonical ``mix == 0`` scheme (reachable via
    ``--ladder linear --w-pts 5 --mix 0``) would write/guard mismatched ``(w_pts, d_pts)`` against
    that shared filename, defeating the cache (rebuild + a fresh repro-log on every alternation).
    Mapping every ``mix == 0`` scheme to the inert ``(ladder, 1, 0, 0)`` at the store/guard boundary
    makes the cache round-trip a fixpoint: a ``mix == 0`` board is a true cache hit under any w/d.
    ``mix > 0`` is returned unchanged, preserving the full four-field scheme distinction.
    """
    if scheme.mix == 0.0:
        return ScoringScheme(scheme.knockout_ladder, w_pts=1.0, d_pts=0.0, mix=0.0)
    return scheme


def build_board(
    field_path: str | Path = _DEFAULT_FIELD,
    ladder: str | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    use_official_draw: bool = True,
    scheme: ScoringScheme | None = None,
) -> Board:
    """Simulate the tournament ``n_sims`` times and assemble the value board.

    The board is scored under ``scheme`` (a :class:`~wcpool.scoring.ScoringScheme`); when omitted
    it is resolved by :func:`_resolve_scheme` — an explicit ``ladder`` selects the legacy terminal
    scoring (``mix == 0``), otherwise the published :data:`RECOMMENDED_SCHEME` (triangular 3:1 at
    gamma_match) is used. Because the objective W is affine-invariant, that recommended blend ranks
    picks identically to the published integer ladder [9, 27, 54, 90, 135, 189] + 3/1 group points.

    ``use_official_draw`` keeps the real 2026 groups fixed across sims (the correct basis for
    a live draft); ``False`` resamples groups (matches the design study, for testing).
    """
    scheme = _resolve_scheme(scheme, ladder)
    field = load_field(field_path)
    model = StrengthModel(field.elo.copy())
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    groups = field.fixed_groups if use_official_draw else random_pot_draw(field.pots, rng)
    # One tournament batch, with the group W/D tallies the group layer needs. The `rng` stream is
    # unchanged by `return_group_results`, so `stages` (hence the legacy mix=0 board) is identical.
    stages, group = simulate_tournament(model, groups, n_sims, rng, return_group_results=True)
    points = team_points(stages, group, scheme)
    return Board(
        points=points,
        stages=stages,
        team_ev=points.mean(axis=0),
        names=field.names,
        group_of=field.group_of,
        knockout_ladder=scheme.knockout_ladder,
        w_pts=scheme.w_pts,
        d_pts=scheme.d_pts,
        mix=scheme.mix,
        n_sims=n_sims,
        seed=seed,
        use_official_draw=use_official_draw,
        config_sha256=field.config_sha256,
        board_sha256=_board_sha256(points, stages),
    )


def _scheme_tag(scheme: ScoringScheme) -> str:
    """Filesystem-safe cache-key fragment for the full scoring scheme.

    Encodes every field that changes the board's ``points`` — ``(knockout_ladder, w_pts, d_pts,
    mix)`` — so a cache built under one scheme cannot be reused for another (a scheme change yields
    a different filename, hence a clean rebuild). ``mix == 0`` collapses to the bare ladder name
    (the group ``(w_pts, d_pts)`` never enter ``team_points`` there), keeping legacy terminal-board
    filenames stable. Decimals are written as ``p`` so the tag has no path-hostile ``.``.
    """
    if scheme.mix == 0.0:
        return scheme.knockout_ladder

    def _num(x: float) -> str:
        return f"{x:g}".replace(".", "p").replace("-", "m")

    return (f"{scheme.knockout_ladder}"
            f"_w{_num(scheme.w_pts)}_d{_num(scheme.d_pts)}_m{_num(scheme.mix)}")


def _cache_path(scheme: ScoringScheme, n_sims: int, seed: int, official: bool) -> Path:
    tag = "official" if official else "resampled"
    return _ARTIFACTS / f"board_{_scheme_tag(scheme)}_{tag}_n{n_sims}_s{seed}.npz"


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
        "scoring_scheme": {  # the two-layer rule the board was scored under
            "knockout_ladder": board.knockout_ladder,
            "w_pts": board.w_pts,
            "d_pts": board.d_pts,
            "mix": board.mix,
        },
        "ladder": board.ladder,  # back-compat alias (== knockout_ladder)
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
    cache_path = _cache_path(board.scheme, board.n_sims, board.seed, board.use_official_draw)
    _write_repro_log(board, cache_path)  # repro record BEFORE the artifact write
    # Persist the CANONICAL scheme (knockout_ladder/w_pts/d_pts/mix) so a reload reconstructs the
    # exact scoring rule and a scheme change is detectable; `ladder` is a derived property, not
    # stored. Canonicalising the stored form means a mix==0 board built under any (w_pts, d_pts)
    # round-trips to the same scheme the load_or_build guard compares against (and to which
    # `_scheme_tag` already folded the filename), so it is a true cache hit -- no rebuild, no new
    # repro-log on every alternation. mix>0 is stored verbatim (full four-field distinction).
    canon = _canonical_scheme(board.scheme)
    meta = {
        "knockout_ladder": canon.knockout_ladder,
        "w_pts": canon.w_pts,
        "d_pts": canon.d_pts,
        "mix": canon.mix,
        **{
            k: getattr(board, k)
            for k in ("n_sims", "seed", "use_official_draw", "config_sha256", "board_sha256")
        },
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
    ladder: str | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    use_official_draw: bool = True,
    field_path: str | Path = _DEFAULT_FIELD,
    scheme: ScoringScheme | None = None,
) -> Board:
    """Return a cached board if present, valid, and current, else build, cache, and return it.

    The board is scored under ``scheme`` (resolved by :func:`_resolve_scheme`: an explicit
    ``ladder`` selects legacy terminal scoring, else the published :data:`RECOMMENDED_SCHEME`).
    A cache that is unreadable (e.g. a partial write) or stale — built against a *different field
    config OR a different scoring scheme* — is ignored and rebuilt, so a corrupt cache can never
    poison the session and a scheme change always forces a fresh board. (The cache filename already
    encodes the scheme via :func:`_scheme_tag`, so a stale-scheme file is normally not even found;
    the explicit scheme-equality guard additionally hardens against a hand-edited or collided meta.)
    """
    scheme = _resolve_scheme(scheme, ladder)
    cache_path = _cache_path(scheme, n_sims, seed, use_official_draw)
    if cache_path.exists():
        try:
            board = load_board(cache_path)
            fresh_config = board.config_sha256 == load_field(field_path).config_sha256
            # Compare CANONICAL forms: a mix==0 board is cache-identical under any (w_pts, d_pts)
            # (the group layer is inert), so this is a true hit across w/d differences rather than a
            # spurious mismatch -> rebuild. mix>0 canonicalises to itself (full four-field guard).
            if fresh_config and _canonical_scheme(board.scheme) == _canonical_scheme(scheme):
                return board
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            # corrupt (incl. truncated .npz -> BadZipFile) / stale -> rebuild. TypeError covers a
            # PRE-REFACTOR cache whose meta carries the old bare `ladder` key (no scheme fields):
            # `load_board`'s `Board(**meta)` rejects it, which is exactly the stale-scheme signal.
            pass
    # scheme is already resolved above (it takes precedence over ladder); pass it alone so
    # build_board does not re-run _resolve_scheme on the raw ladder (single source of truth).
    board = build_board(field_path, ladder=None, n_sims=n_sims, seed=seed,
                        use_official_draw=use_official_draw, scheme=scheme)
    save_board(board)
    return board
