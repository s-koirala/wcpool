"""Run the (strength_config x ladder x N x policy) experiment grid and write tidy tables.

Outputs (under docs/tables/):
  results_main_{date}.csv          EV-greedy + variance policies over the full grid.
  results_exploitability_{date}.csv  EV-greedy vs best-response at a matched subset.

Emits a 13-field ReproLog (logs/reproducibility/) before writing, per the CLAUDE.md
reproducibility mandate, using the canonical emit-repro-log asset when available.

Sim sizing: a "cell" sees ``n_draws * sims_per_draw`` eval tournaments. The default
(25 x 2000 = 50,000) meets the >=50k-per-cell requirement; --quick is a smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpool import simulate  # noqa: E402
from wcpool.config import load_field  # noqa: E402
from wcpool.scoring import GROUP_POINT_SCHEMES, ScoringScheme  # noqa: E402

# --- Experiment grid (design choices documented in docs/assumptions.md) -----------------
N_VALUES = [4, 5, 6, 8]
LADDERS = ["linear", "triangular", "geometric"]
MAIN_POLICIES = ["ev_greedy", "variance"]
# Synthetic favoritism sweep: spread as a multiple of the empirical Elo SD (half / equal /
# double the real field's dispersion) — a documented stress range, not a tuned value.
SYNTH_SPREAD_MULTIPLIERS = [0.5, 1.0, 2.0]
DEFAULT_SEED = 20260608  # fixed master seed = the 2026-06-08 run date (YYYYMMDD)

# Best-response (exploitability) probe runs at a reduced budget because each pick is
# O(n_teams) more expensive than EV-greedy; the resulting between-draw SE is reported so
# the small exploitability margins remain interpretable. (full / --quick)
BR_DRAWS_FULL, BR_SIMS_FULL = 8, 1500
BR_DRAWS_QUICK, BR_SIMS_QUICK = 3, 250

CONFIG_PATH = ROOT / "config" / "ratings_elo_2026.yaml"

# --- Group-scoring (gamma) sweep grid (plan sections 4, 11, 12) --------------------------
# The NEW grid seed (run date 2026-06-09); the gamma=0 anchor self-validates against the frozen
# golden, which deliberately reuses 20260608.
GROUPSCORING_SEED = 20260609
# Headline grid: 8 drafters x 6 teams, real Elo, EV-greedy, N=6, resampled (plan section 4).
GS_N_DRAFTERS = 8
GS_N_VALUES = [6]
GS_POLICIES = ["ev_greedy"]
# Swept group-layer share grid (plan section 4: denser toward 0, gaps widen toward the equal-purse
# end). gamma=0 is the mix=0 anchor; gamma_match is added PER CELL by build_gamma_schemes.
GAMMA_GRID = [0.0, 0.05, 0.12, 0.25, 0.50]
# Cap on replicates fed to the (compute-dominant, provisional) suspense surrogate per cell-draw
# (data-driven default documented in simulate._DEFAULT_SUSPENSE_SUBSAMPLE).
GS_SUSPENSE_SUBSAMPLE = 512
# Robustness slice (EXISTING metrics only): the synthetic-concentration sweep at a couple of
# representative gamma so the shape-ordering stability can be checked. Cheaper than the headline.
GS_ROBUSTNESS_GAMMAS = [0.0, 0.5]  # gamma=0 anchor, equal-purse end; gamma_match added per cell
GS_ROBUSTNESS_SPREADS = SYNTH_SPREAD_MULTIPLIERS  # {0.5, 1, 2} x Elo SD

# Headline draw count (plan section 8.2: "raise draws not sims"). The cross-seed level SD of the
# ABSOLUTE skill ~ 1/sqrt(n_draws); 25 draws -> SD ~ 0.012, 100 draws -> SD ~ 0.006, i.e.
# seed-stable at the gamma-grid spacing scale. The paired Delta-skill frontier (item 1) is already
# ~7x more stable than the level, but the headline level itself is pinned by raising draws to 100.
# sims stay at 2000 (deepening per-draw resolution does not reduce the BETWEEN-draw level SD).
GS_HEADLINE_DRAWS = 100
# Three consecutive seeds for the gamma=0 anchor cross-seed envelope (the run-date seed +/- 1): a
# small, documented self-consistency band that brackets GROUPSCORING_SEED, used only to REPORT the
# cross-seed level SD (confirming stability), not to tune anything.
GS_ANCHOR_ENVELOPE_SEEDS = [20260608, 20260609, 20260610]


def _load_reprolog_module():
    asset = Path.home() / ".claude" / "skills" / "emit-repro-log" / "assets" / "emit_repro_log.py"
    if not asset.exists():
        return None
    spec = importlib.util.spec_from_file_location("emit_repro_log", asset)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # let the frozen dataclass resolve its own annotations
    spec.loader.exec_module(mod)
    return mod


def _resolve_git_head() -> str | None:
    """Project git HEAD via the repro asset's ``_git_head`` (None if the asset/symbol is absent).

    Guarded with ``hasattr`` so an asset refactor that renames/removes the private ``_git_head``
    symbol degrades gracefully (the sidecar records ``git_head: null``) rather than raising
    AttributeError (audit item 8).
    """
    mod = _load_reprolog_module()
    if mod is None or not hasattr(mod, "_git_head"):
        return None
    return mod._git_head(mod.ProjectPaths.discover().root)


def emit_repro_log(
    seed: int, config_sha256: str, grid_meta: dict, run_tag: str = "n/a"
) -> tuple[str, str | None]:
    """Write a ReproLog; return ``(path_or_note, run_id)`` (run_id None when the asset is absent).

    ``model_hash`` is set to the git HEAD (the plan section-11 / audit-R6 parity fix: the design
    path previously left it ``null`` while ``board.py`` already populated it). The resolved-config
    SHA fingerprints ``grid_meta`` -- so the scoring axes added to it (D/W set, shapes, gamma grid,
    gamma_match values, seed, suspense_subsample, run_tag) are hashed into the record.

    ``run_tag`` (e.g. ``"groupscoring_headline_100draws"`` vs ``"groupscoring_quick"``) is recorded
    in the ReproLog's ``hypothesis_id`` field so two otherwise-identical group-scoring runs are
    distinguishable by reading the JSON, not just by recomputing the opaque config hash (audit item
    7). The returned ``run_id`` lets the caller link the emitted CSV to this exact log in a sidecar.
    """
    mod = _load_reprolog_module()
    config_resolved = hashlib.sha256(json.dumps(grid_meta, sort_keys=True).encode()).hexdigest()
    if mod is None:
        return "emit-repro-log asset unavailable; ReproLog skipped", None
    paths = mod.ProjectPaths.discover()
    # Populate model_hash with the git HEAD for parity with board.py (was null on the design path).
    # `_git_head` is a private asset symbol; guard its use with `hasattr` so a future asset refactor
    # that renames/removes it degrades gracefully (model_hash falls back to the log's own git_head,
    # which `capture` always records) rather than raising AttributeError.
    git_head = mod._git_head(paths.root) if hasattr(mod, "_git_head") else None
    log = mod.capture(
        phase="inference",
        hypothesis_id=run_tag,  # human-readable run identity (audit item 7); was always "n/a"
        rng_seed=seed,
        dataset_checksums={"config/ratings_elo_2026.yaml": config_sha256},
        model_hash=git_head if git_head is not None else None,
        config_resolved_sha256=config_resolved,
    )
    if git_head is None and log.model_hash is None:
        # Asset lacked `_git_head`; mirror capture's own git_head into model_hash (board.py parity).
        log = mod.with_model_hash(log, log.git_head) if hasattr(mod, "with_model_hash") else log
    out = paths.logs_reproducibility / f"repro_log_{log.run_id}.json"
    log.write(out)
    return str(out.relative_to(ROOT)), log.run_id


def build_configs(field, seed: int):
    configs = [(0, simulate.make_elo_config(field, name="elo_2026"))]
    spread0 = field.elo_spread
    for i, mult in enumerate(SYNTH_SPREAD_MULTIPLIERS, start=1):
        spread = mult * spread0
        name = f"synthetic_x{mult:g}"
        # rating_stream fixed (default 0): all synthetic configs share one z-draw and differ
        # only by spread -> a controlled, monotone concentration sweep. cfg_id (i) still
        # gives each config independent tournament-sim streams.
        cfg = simulate.make_synthetic_config(spread, name=name, seed=seed)
        configs.append((i, cfg))
    return configs


def run_main_grid(field, n_draws, sims_per_draw, seed, regime):
    rows = []
    for cfg_id, cfg in build_configs(field, seed):
        recs = simulate.run_strength_config(
            cfg,
            ladders=LADDERS,
            n_values=N_VALUES,
            policies=MAIN_POLICIES,
            n_draws=n_draws,
            sims_per_draw=sims_per_draw,
            seed=seed,
            regime=regime,
            cfg_id=cfg_id,
        )
        rows.extend(recs)
        print(
            f"  [{cfg.name}] top-8 title share={cfg.top8_title_share:.3f}, "
            f"beta={cfg.model.beta:.3f}"
        )
    return pd.DataFrame(rows)


def run_exploitability(field, n_draws, sims_per_draw, seed, regime):
    """EV-greedy vs best-response at a matched subset (the exploitability probe)."""
    rows = []
    probe_configs = [(0, simulate.make_elo_config(field, name="elo_2026"))]
    spread = 1.0 * field.elo_spread
    probe_configs.append((1, simulate.make_synthetic_config(spread, "synthetic_x1", seed)))
    for cfg_id, cfg in probe_configs:
        recs = simulate.run_strength_config(
            cfg,
            ladders=LADDERS,
            n_values=[4, 6],
            policies=["ev_greedy", "best_response"],
            n_draws=n_draws,
            sims_per_draw=sims_per_draw,
            seed=seed + 1,  # distinct stream from the main grid
            regime=regime,
            br_slot=0,
            cfg_id=cfg_id,
        )
        rows.extend(recs)
        print(f"  [probe:{cfg.name}] done")
    return pd.DataFrame(rows)


# --- Group-scoring (gamma) sweep --------------------------------------------------------


def _cell_lookup(cells):
    """Map a record's ``(knockout_ladder, w_pts, d_pts, mix)`` identity -> its ``GammaCell``.

    The scheme columns ``run_strength_config`` emits are exactly this 4-tuple (the gamma=0 anchor
    collapses to ``(shape, 1.0, 0.0, 0.0)``), so the realised records join back to the design grid
    by identity. ``mix`` is rounded to a fixed precision for a robust float key (it round-trips
    bit-for-bit, so this only guards against accidental representation drift).
    """
    return {
        (c.scheme.knockout_ladder, c.scheme.w_pts, c.scheme.d_pts, round(c.scheme.mix, 12)): c
        for c in cells
    }


def _attach_gamma_meta(recs, cells):
    """Annotate each metric record with its design-grid gamma metadata (in place), return ``recs``.

    Adds ``gamma_target`` (the requested level, ``-1`` for the gamma_match landmark),
    ``gamma_realised`` (``gamma_of_mix`` at the calibrated purses), ``is_gamma_match``,
    ``gamma_match_value`` (the (shape, D/W) family's gamma_match share, attached to every row of the
    family for easy pivoting), and the calibrated ``group_purse``/``knockout_purse``.
    """
    lookup = _cell_lookup(cells)
    # gamma_match share per (shape, w, d) family (the is_gamma_match cell's realised gamma).
    match_value = {
        (c.scheme.knockout_ladder, c.scheme.w_pts, c.scheme.d_pts): c.gamma_realised
        for c in cells
        if c.is_gamma_match
    }
    for r in recs:
        key = (r["knockout_ladder"], r["w_pts"], r["d_pts"], round(r["mix"], 12))
        c = lookup[key]
        r["gamma_target"] = c.gamma_target
        r["gamma_realised"] = c.gamma_realised
        r["is_gamma_match"] = c.is_gamma_match
        r["gamma_match_value"] = match_value.get(
            (c.scheme.knockout_ladder, c.scheme.w_pts, c.scheme.d_pts), float("nan")
        )
        r["group_purse"] = c.group_purse
        r["knockout_purse"] = c.knockout_purse
    return recs


def _attach_paired_delta_skill(df, per_draw_skill):
    """Attach the SEED-STABLE paired Delta-skill-vs-gamma frontier columns (audit item 1).

    For each (shape, D/W, gamma) cell, the recommendation basis is the PAIRED skill change relative
    to that shape's gamma=0 anchor, formed PER DRAW on the shared tournament brackets and then
    averaged::

        delta_per_draw[d] = skill_cell[d] - skill_anchor[d]      # paired on the SAME bracket d
        delta_skill_vs_anchor = mean_d delta_per_draw
        delta_skill_paired_se = std_d(delta_per_draw, ddof=1) / sqrt(n_draws)

    The pairing cancels the large between-draw bracket variance that dominates the ABSOLUTE skill
    level, so this quantity is ~7x more seed-stable than the level (plan section 8 / 14 -- it is
    what the selection must rest on). The anchor is the ``(shape, 1.0, 0.0, 0.0)`` mix=0 cell, whose
    per-draw series is keyed by the bare ladder-name string (``wcpool.simulate._scheme_key``); every
    cell of every shape was scored on the SAME per-draw brackets (one ``run_strength_config`` call,
    shared ``cfg_id``), so cell and anchor series are paired draw-for-draw. Anchor cells get delta 0
    and paired SE 0 (their series minus itself is identically zero).

    Adds columns ``delta_skill_vs_anchor`` and ``delta_skill_paired_se`` in place; returns ``df``.
    Assumes the single-(N, policy) headline grid (``GS_N_VALUES``/``GS_POLICIES``); the (N, policy)
    of each row is read from the row itself, so a multi-(N, policy) grid is handled correctly too.
    """
    deltas: list[float] = []
    paired_ses: list[float] = []
    for row in df.itertuples(index=False):
        shape = row.knockout_ladder
        n, policy = int(row.teams_per_drafter), row.policy
        cell_scheme = ScoringScheme(shape, float(row.w_pts), float(row.d_pts), float(row.mix))
        cell_key = (simulate._scheme_key(cell_scheme), n, policy)
        # The shape's gamma=0 anchor is the pure terminal ladder, whose scheme key is the bare name.
        anchor_key = (simulate._scheme_key(ScoringScheme(shape, 1.0, 0.0, 0.0)), n, policy)
        # The paired delta + paired cluster SE are the tested library kernel (wcpool.simulate); the
        # cell and anchor series are paired draw-for-draw (shared per-draw brackets, one cfg_id).
        delta, paired_se = simulate.paired_delta_skill(
            per_draw_skill[cell_key], per_draw_skill[anchor_key]
        )
        deltas.append(delta)
        paired_ses.append(paired_se)
    df["delta_skill_vs_anchor"] = deltas
    df["delta_skill_paired_se"] = paired_ses
    return df


def run_groupscoring_grid(field, calib, n_draws, sims_per_draw, seed, regime, suspense_subsample):
    """Headline gamma sweep: real Elo, the (D/W x shape x gamma) schemes, EV-greedy, 8 x 6.

    One cell per (shape, D/W, gamma) -- the skill-vs-engagement frontier. Every scheme is scored on
    the SAME per-draw tournament batch (the simulation is scheme-independent and reused across the
    whole grid in one ``run_strength_config`` call with a shared ``cfg_id``), so the per-draw skill
    series returned via ``return_per_draw=True`` are PAIRED across all cells -- the basis for the
    seed-stable Delta-skill-vs-gamma frontier (:func:`_attach_paired_delta_skill`, audit item 1).

    Returns ``(DataFrame, cells, per_draw_skill)`` where ``per_draw_skill`` maps each cell key
    (``(scheme_key, N, policy)``; the gamma=0 anchor's key is the bare ladder-name string per
    :func:`wcpool.simulate._scheme_key`) to its ``(n_draws,)`` per-draw mean-Spearman array. The
    DataFrame carries the existing metrics + engagement columns + the gamma/mix/knockout_ladder/
    w_pts/d_pts design columns + cluster SEs (all from ``run_strength_config``; gamma metadata
    joined on) and, after :func:`_attach_paired_delta_skill`, the paired Delta-skill columns.
    """
    cells = simulate.build_gamma_schemes(calib, GAMMA_GRID)
    schemes = [c.scheme for c in cells]
    cfg = simulate.make_elo_config(field, name="elo_2026")
    recs, per_draw_skill = simulate.run_strength_config(
        cfg,
        schemes=schemes,
        n_values=GS_N_VALUES,
        policies=GS_POLICIES,
        n_draws=n_draws,
        sims_per_draw=sims_per_draw,
        seed=seed,
        regime=regime,
        cfg_id=0,
        n_drafters=GS_N_DRAFTERS,
        suspense_subsample=suspense_subsample,
        return_per_draw=True,
    )
    _attach_gamma_meta(recs, cells)
    print(
        f"  [elo_2026] {len(cells)} schemes x {len(GS_N_VALUES)}N x {len(GS_POLICIES)}pol "
        f"= {len(recs)} cells, top-8 title share={cfg.top8_title_share:.3f}"
    )
    return pd.DataFrame(recs), cells, per_draw_skill


def run_groupscoring_robustness(field, n_draws, sims_per_draw, seed, regime, suspense_subsample):
    """Robustness slice (EXISTING metrics) across concentration configs at representative gamma.

    Re-runs the synthetic concentration sweep (spread in {0.5, 1, 2} x Elo SD) at gamma in
    {0, gamma_match, 0.5} so the shape-ordering stability (geometric > triangular > linear on skill)
    can be checked off the real headline. Each synthetic config is calibrated separately (its purses
    differ from the real field), so gamma is comparable WITHIN a config. Engagement metrics are
    computed too (cheap to carry) but are not the point here; this stays cheaper than the headline
    by spanning fewer gamma levels. Returns a single tidy DataFrame tagged by ``strength_config``.
    """
    rows = []
    for i, mult in enumerate(GS_ROBUSTNESS_SPREADS, start=1):
        spread = mult * field.elo_spread
        name = f"synthetic_x{mult:g}"
        cfg = simulate.make_synthetic_config(spread, name=name, seed=seed)
        # Per-config calibration (purses are field-specific), on the SAME synthetic config / seed.
        calib = simulate.calibrate_group_knockout(cfg, seed=seed, regime=regime, cfg_id=i)
        cells = simulate.build_gamma_schemes(calib, GS_ROBUSTNESS_GAMMAS)
        recs = simulate.run_strength_config(
            cfg,
            schemes=[c.scheme for c in cells],
            n_values=GS_N_VALUES,
            policies=GS_POLICIES,
            n_draws=n_draws,
            sims_per_draw=sims_per_draw,
            seed=seed,
            regime=regime,
            cfg_id=i,
            n_drafters=GS_N_DRAFTERS,
            suspense_subsample=suspense_subsample,
        )
        _attach_gamma_meta(recs, cells)
        rows.extend(recs)
        print(f"  [robust:{name}] {len(cells)} schemes -> {len(recs)} cells")
    return pd.DataFrame(rows)


def _groupscoring_grid_meta(
    calib, n_draws, sims_per_draw, regime, seed, suspense_subsample, run_tag
):
    """ReproLog ``grid_meta`` for the gamma sweep: the scoring axes are fingerprinted (plan s.11).

    Includes the D/W set, shapes, the gamma grid, the calibrated purses, the per-cell gamma_match
    landmark shares, the seed, the suspense subsample, and the ``run_tag`` (so ``emit_repro_log``
    hashes them into ``config_resolved_sha256`` and the full vs ``--quick`` runs fingerprint
    distinctly). ``run_tag`` is also surfaced verbatim in the ReproLog's ``hypothesis_id`` field so
    the two logs are distinguishable by inspection without recomputing the hash (audit item 7).
    """
    cells = simulate.build_gamma_schemes(calib, GAMMA_GRID)

    def _mkey(c):
        return f"{c.scheme.knockout_ladder}|{c.scheme.w_pts:g}:{c.scheme.d_pts:g}"

    gamma_match = {
        _mkey(c): round(c.gamma_realised, 8) for c in cells if c.is_gamma_match
    }
    return {
        "sweep": "groupscoring_gamma",
        "run_tag": run_tag,
        "GROUP_POINT_SCHEMES": {k: list(v) for k, v in GROUP_POINT_SCHEMES.items()},
        "SHAPES": list(LADDERS),
        "GAMMA_GRID": GAMMA_GRID,
        "gamma_match_shares": gamma_match,
        "calibration": {
            "e_sum_wins": round(calib.e_sum_wins, 6),
            "e_sum_draws": round(calib.e_sum_draws, 6),
            "e_sum_A": {k: round(v, 6) for k, v in calib.e_sum_A.items()},
            "regime": calib.regime,
            "seed": calib.seed,
            "n_sims": calib.n_sims,
        },
        "n_distinct_cells": len(cells),
        "GS_N_DRAFTERS": GS_N_DRAFTERS,
        "GS_N_VALUES": GS_N_VALUES,
        "GS_POLICIES": GS_POLICIES,
        "robustness_gammas": GS_ROBUSTNESS_GAMMAS,
        "robustness_spreads": GS_ROBUSTNESS_SPREADS,
        "n_draws": n_draws,
        "sims_per_draw": sims_per_draw,
        "regime": regime,
        "seed": seed,
        "suspense_subsample": suspense_subsample,
    }


def _sha256_file(path: Path) -> str:
    """Stream the SHA-256 of a file's bytes (the emitted CSV, read back after the write)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# Provenance: the engagement-column reliability annotations carried in every group-scoring sidecar
# (audit item 4). These document -- in a machine-readable, byte-pinned place -- which engagement
# columns are gated/reportable vs provisional, so a downstream reader does not over-trust the
# un-gated suspense surrogate. The text mirrors the run_strength_config / metrics docstrings.
_ENGAGEMENT_COLUMN_NOTES = {
    "group_variance_share": (
        "HEADLINE engagement axis: estimator-free, monotone in group mass, exactly 0 at gamma=0. "
        "Pooled ratio-of-means (comparable to skill_variance_share). Reportable."
    ),
    "alive_fraction": (
        "Feasibility sanity check only, NOT a discriminating axis: saturates >= 0.999 (cluster SE "
        "0) over the studied 8x6 grid with mix <= 0.5."
    ),
    "pool_suspense": (
        "PROVISIONAL / un-gated: k-NN surrogate, one-sidedly biased DOWNWARD by smoothing. The "
        "section-6 nested-sim gate validates ONLY the low-mix headline cell (triangular, "
        "mix~0.15); it REJECTS at high mass, and the share is non-monotone for geometric where the "
        "share is tiny (below the surrogate reliability floor). Do not adopt the level until gated."
    ),
    "pool_suspense_group_share": (
        "PROVISIONAL: group-phase share of pool_suspense (see pool_suspense)."
    ),
    "pool_surprise": (
        "PROVISIONAL and NOT independent confirmation of pool_suspense: on this realised-path "
        "surrogate pool_surprise is BYTE-IDENTICAL to pool_suspense (the summed squared belief "
        "increments are the same statistic read forward vs backward). The genuine forward/backward "
        "split is gated on the section-6 nested simulation."
    ),
    "pool_surprise_group_share": (
        "PROVISIONAL: byte-identical to pool_suspense_group_share on the surrogate "
        "(see pool_surprise)."
    ),
}


def _emit_results_sidecar(
    csv_path: Path,
    repro_path: str,
    run_id: str | None,
    seed: int,
    n_draws: int,
    sims_per_draw: int,
    regime: str,
    suspense_subsample,
    run_tag: str,
    git_head: str | None,
) -> Path:
    """Write ``<csv>.repro.json`` linking an emitted CSV to its ReproLog + provenance (item 3).

    Records the SHA-256 of the emitted CSV (read back from disk so the digest is of the exact bytes
    written), the ReproLog run_id + log filename, git_head, seed, n_draws/sims_per_draw, regime, the
    suspense subsample, and the run_tag. Also carries the engagement-column reliability annotations
    (:data:`_ENGAGEMENT_COLUMN_NOTES`, audit item 4): the suspense/surprise columns are flagged
    PROVISIONAL/un-gated, ``group_variance_share`` is named the engagement HEADLINE, and
    pool_surprise* is recorded as byte-identical to pool_suspense* on the surrogate. NOT
    git-committed (that is user-authorized) -- this just creates the sidecar next to the CSV.
    """
    csv_sha = _sha256_file(csv_path)
    repro_log_file = Path(repro_path).name if run_id is not None else None
    sidecar = {
        "artifact": csv_path.name,
        "artifact_sha256": csv_sha,
        "reprolog": {
            "run_id": run_id,
            "log_file": repro_log_file,
            "log_path": repro_path if run_id is not None else None,
            "git_head": git_head,
            "run_tag": run_tag,
        },
        "run": {
            "seed": seed,
            "n_draws": n_draws,
            "sims_per_draw": sims_per_draw,
            "n_eval_tournaments_per_cell": n_draws * sims_per_draw,
            "regime": regime,
            "suspense_subsample": suspense_subsample,
        },
        "engagement_column_provenance": _ENGAGEMENT_COLUMN_NOTES,
        "note": (
            "Recommendation rests on the SEED-STABLE PAIRED delta_skill_vs_anchor (+ "
            "delta_skill_paired_se) columns, NOT the absolute spearman_mean level. See "
            "_attach_paired_delta_skill / audit item 1."
        ),
    }
    out = csv_path.with_suffix(".repro.json")
    out.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _run_anchor_seed_envelope(field, seeds, regime, suspense_subsample, n_draws, sims_per_draw):
    """Cross-seed envelope on the gamma=0 ANCHORS, confirming the absolute level is seed-stable.

    Re-runs ONLY the three (shape) gamma=0 anchor cells (the pure terminal ladders) at each seed in
    ``seeds`` and reports, per shape, the across-seed mean and sample SD of ``spearman_mean``. The
    cross-seed level SD ~ 1/sqrt(n_draws) (plan section 8.2), so at the headline ``n_draws`` it
    should be small (~0.006 at 100 draws) -- the check that "raise draws not sims" delivered a
    seed-stable level. Anchors are cheap (3 cells) and isolate the level (gamma=0 has no group
    layer), so this adds little cost. Returns ``{shape: {"mean", "sd", "values": [...]}}``.

    No calibration is needed: the anchors are the pure terminal ladders (mix=0), whose scores never
    touch the calibrated purses, so each seed's anchor depends only on its RNG path.
    """
    per_shape_by_seed: dict[str, list[float]] = {s: [] for s in LADDERS}
    cfg = simulate.make_elo_config(field, name="elo_2026")
    # The gamma=0 anchors are exactly the pure terminal ladders ``(shape, 1.0, 0.0, 0.0)`` -- the
    # SAME mix=0 cell the headline frontier anchors on. Construct them directly (NOT via
    # ``build_gamma_schemes(calib, [0.0])``, which also emits the per-D/W gamma_match landmarks, so
    # multiple records per shape would be returned and a by-shape dict would silently keep the wrong
    # one). No calibration is needed: at mix=0 the purses never enter the anchor's score.
    anchor_schemes = [ScoringScheme(shape, 1.0, 0.0, 0.0) for shape in LADDERS]
    for sd in seeds:
        recs = simulate.run_strength_config(
            cfg,
            schemes=anchor_schemes,
            n_values=GS_N_VALUES,
            policies=GS_POLICIES,
            n_draws=n_draws,
            sims_per_draw=sims_per_draw,
            seed=sd,
            regime=regime,
            cfg_id=0,
            n_drafters=GS_N_DRAFTERS,
            suspense_subsample=suspense_subsample,
        )
        # Each record is a distinct mix=0 anchor (one per shape); key by shape unambiguously.
        by_shape = {r["knockout_ladder"]: r for r in recs if r["mix"] == 0.0}
        for shape in LADDERS:
            per_shape_by_seed[shape].append(float(by_shape[shape]["spearman_mean"]))
    out = {}
    for shape, vals in per_shape_by_seed.items():
        arr = np.asarray(vals, dtype=float)
        out[shape] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "values": [float(x) for x in arr],
        }
    return out


def _print_paired_frontier(df):
    """Print the paired Delta-skill-vs-gamma frontier (per shape x D/W), the seed-stable basis."""
    print("\nPAIRED Delta-skill vs gamma frontier (delta_skill_vs_anchor +/- paired SE; "
          "anchor = gamma=0 of each shape):")
    print(f"  {'shape':11s} {'D/W':10s} {'gamma_t':>8s} {'gamma_real':>10s} "
          f"{'skill':>8s} {'d_skill':>9s} {'paired_se':>10s}")
    sub = df.sort_values(["knockout_ladder", "w_pts", "d_pts", "gamma_target"])
    for r in sub.itertuples(index=False):
        dw = "anchor" if r.mix == 0.0 else f"{r.w_pts:g}/{r.d_pts:g}"
        gt = "match" if r.is_gamma_match else f"{r.gamma_target:.2f}"
        print(f"  {r.knockout_ladder:11s} {dw:10s} {gt:>8s} {r.gamma_realised:10.4f} "
              f"{r.spearman_mean:8.4f} {r.delta_skill_vs_anchor:+9.4f} "
              f"{r.delta_skill_paired_se:10.5f}")


def _print_anchor_envelope(envelope, seeds):
    """Print the gamma=0 anchor cross-seed envelope (per-shape mean + cross-seed SD)."""
    print(f"\ngamma=0 ANCHOR cross-seed envelope over seeds {seeds} "
          f"(confirms absolute-level stability; SD ~ 1/sqrt(n_draws)):")
    print(f"  {'shape':11s} {'cross_seed_mean':>16s} {'cross_seed_sd':>14s}  values")
    for shape in LADDERS:
        e = envelope[shape]
        vals = ", ".join(f"{v:.4f}" for v in e["values"])
        print(f"  {shape:11s} {e['mean']:16.4f} {e['sd']:14.5f}  [{vals}]")


def run_groupscoring(args):
    """The NEW group-scoring entry point (separate from the existing prior-study grid)."""
    if args.quick:
        n_draws, sims_per_draw = (args.draws or 3), (args.sims or 300)
        run_tag = "groupscoring_quick"
    else:
        # Headline default is 100 draws (plan section 8.2 "raise draws not sims") -- pins the
        # absolute level to a seed-stable cross-seed SD (~0.006). --draws overrides it.
        n_draws, sims_per_draw = (args.draws or GS_HEADLINE_DRAWS), (args.sims or 2000)
        run_tag = f"groupscoring_headline_{n_draws}draws"
    suspense_subsample = GS_SUSPENSE_SUBSAMPLE
    seed = args.seed if args.seed is not None else GROUPSCORING_SEED

    field = load_field(CONFIG_PATH)
    cfg_real = simulate.make_elo_config(field, name="elo_2026")
    print("Calibrating gamma<->mix purses (real Elo)...")
    calib = simulate.calibrate_group_knockout(cfg_real, seed=seed, regime=args.regime, cfg_id=0)
    print(
        f"  E[sum wins]={calib.e_sum_wins:.3f} E[sum draws]={calib.e_sum_draws:.3f} "
        f"K={ {k: round(v, 2) for k, v in calib.e_sum_A.items()} }"
    )

    grid_meta = _groupscoring_grid_meta(
        calib, n_draws, sims_per_draw, args.regime, seed, suspense_subsample, run_tag
    )
    repro, run_id = emit_repro_log(seed, field.config_sha256, grid_meta, run_tag=run_tag)
    git_head = _resolve_git_head()
    print(f"ReproLog: {repro} (run_id={run_id}, tag={run_tag})")
    print(
        f"Grid: {n_draws} draws x {sims_per_draw} sims = {n_draws * sims_per_draw} "
        f"eval tournaments/cell, regime={args.regime}, seed={seed}, "
        f"distinct cells={grid_meta['n_distinct_cells']}"
    )

    out_dir = ROOT / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    tag = "_quick" if args.quick else ""

    print("Headline gamma sweep:")
    df_head, _cells, per_draw_skill = run_groupscoring_grid(
        field, calib, n_draws, sims_per_draw, seed, args.regime, suspense_subsample
    )
    # Attach the SEED-STABLE paired Delta-skill frontier (audit item 1) -- the recommendation basis.
    _attach_paired_delta_skill(df_head, per_draw_skill)
    head_path = out_dir / f"results_groupscoring{tag}_{today}.csv"
    df_head.to_csv(head_path, index=False)
    print(f"  wrote {head_path.relative_to(ROOT)} ({len(df_head)} rows)")
    _print_paired_frontier(df_head)

    # Provenance sidecar (audit item 3 + the item-4 provisional labels) for the headline CSV.
    head_sidecar = _emit_results_sidecar(
        head_path, repro, run_id, seed, n_draws, sims_per_draw, args.regime,
        suspense_subsample, run_tag, git_head,
    )
    print(f"  sidecar: {head_sidecar.relative_to(ROOT)}")

    # 3-seed cross-seed envelope on the gamma=0 ANCHORS only -> confirm the absolute level is stable
    # at the headline draw count (audit item 2). Skipped for --quick (the small budget is not the
    # seed-stability claim) and when --no-anchor-envelope is passed.
    if not args.quick and not args.no_anchor_envelope:
        print(f"gamma=0 anchor cross-seed envelope ({len(GS_ANCHOR_ENVELOPE_SEEDS)} seeds x "
              f"{n_draws} draws x {sims_per_draw} sims):")
        envelope = _run_anchor_seed_envelope(
            field, GS_ANCHOR_ENVELOPE_SEEDS, args.regime, suspense_subsample, n_draws, sims_per_draw
        )
        _print_anchor_envelope(envelope, GS_ANCHOR_ENVELOPE_SEEDS)

    if not args.no_robustness:
        # Robustness is cheaper: fewer gamma levels, and (unless overridden) a reduced budget so it
        # stays well under the headline cost while still resolving the shape ordering. It confirms
        # ORDERING, not levels, so it stays at its lower budget (audit item 2).
        rb_draws = args.robust_draws or max(n_draws // 4, 2)
        rb_sims = args.robust_sims or sims_per_draw
        print(f"Robustness slice ({rb_draws} draws x {rb_sims} sims):")
        df_rb = run_groupscoring_robustness(
            field, rb_draws, rb_sims, seed, args.regime, suspense_subsample
        )
        rb_path = out_dir / f"results_groupscoring_robustness{tag}_{today}.csv"
        df_rb.to_csv(rb_path, index=False)
        print(f"  wrote {rb_path.relative_to(ROOT)} ({len(df_rb)} rows)")
        rb_tag = f"{run_tag}_robustness"
        rb_sidecar = _emit_results_sidecar(
            rb_path, repro, run_id, seed, rb_draws, rb_sims, args.regime,
            suspense_subsample, rb_tag, git_head,
        )
        print(f"  sidecar: {rb_sidecar.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run (small sim counts)")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"master seed (prior-study default {DEFAULT_SEED}; group-scoring {GROUPSCORING_SEED})",
    )
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--regime", choices=["resampled", "fixed"], default="resampled")
    ap.add_argument("--no-bestresponse", action="store_true")
    # NEW group-scoring (gamma) sweep -- separate entry point; does NOT touch the prior-study grid.
    ap.add_argument(
        "--group-scoring",
        action="store_true",
        help="run the NEW group-stage (W/D + gamma) sweep instead of the prior terminal-only study",
    )
    ap.add_argument("--no-robustness", action="store_true", help="skip group-scoring robustness")
    ap.add_argument("--robust-draws", type=int, default=None, help="override robustness draws")
    ap.add_argument("--robust-sims", type=int, default=None, help="override robustness sims")
    ap.add_argument(
        "--no-anchor-envelope",
        action="store_true",
        help="skip the gamma=0 anchor cross-seed envelope (group-scoring level-stability check)",
    )
    args = ap.parse_args()

    if args.group_scoring:
        run_groupscoring(args)
        return

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    if args.quick:
        n_draws, sims_per_draw = (args.draws or 4), (args.sims or 250)
        br_draws, br_sims = BR_DRAWS_QUICK, BR_SIMS_QUICK
    else:
        n_draws, sims_per_draw = (args.draws or 25), (args.sims or 2000)
        br_draws, br_sims = BR_DRAWS_FULL, BR_SIMS_FULL

    field = load_field(CONFIG_PATH)
    grid_meta = {
        "N_VALUES": N_VALUES,
        "LADDERS": LADDERS,
        "MAIN_POLICIES": MAIN_POLICIES,
        "SYNTH_SPREAD_MULTIPLIERS": SYNTH_SPREAD_MULTIPLIERS,
        "n_draws": n_draws,
        "sims_per_draw": sims_per_draw,
        "regime": args.regime,
        "seed": seed,
    }
    repro, _run_id = emit_repro_log(seed, field.config_sha256, grid_meta)
    print(f"ReproLog: {repro}")
    print(
        f"Grid: {n_draws} draws x {sims_per_draw} sims = "
        f"{n_draws * sims_per_draw} eval tournaments/cell, regime={args.regime}"
    )

    out_dir = ROOT / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    # Tag smoke runs so a --quick run never overwrites the full 50k deliverable CSV.
    tag = "_quick" if args.quick else ""

    print("Main grid:")
    df_main = run_main_grid(field, n_draws, sims_per_draw, seed, args.regime)
    main_path = out_dir / f"results_main{tag}_{today}.csv"
    df_main.to_csv(main_path, index=False)
    print(f"  wrote {main_path.relative_to(ROOT)} ({len(df_main)} rows)")

    if not args.no_bestresponse:
        print("Exploitability probe:")
        df_exp = run_exploitability(field, br_draws, br_sims, seed, args.regime)
        exp_path = out_dir / f"results_exploitability{tag}_{today}.csv"
        df_exp.to_csv(exp_path, index=False)
        print(f"  wrote {exp_path.relative_to(ROOT)} ({len(df_exp)} rows)")


if __name__ == "__main__":
    main()
