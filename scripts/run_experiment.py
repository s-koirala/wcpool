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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpool import simulate  # noqa: E402
from wcpool.config import load_field  # noqa: E402

# --- Experiment grid (design choices documented in docs/assumptions.md) -----------------
N_VALUES = [4, 5, 6, 8]
LADDERS = ["linear", "triangular", "geometric"]
MAIN_POLICIES = ["ev_greedy", "variance"]
# Synthetic favoritism sweep: spread as a multiple of the empirical Elo SD (half / equal /
# double the real field's dispersion) — a documented stress range, not a tuned value.
SYNTH_SPREAD_MULTIPLIERS = [0.5, 1.0, 2.0]
DEFAULT_SEED = 20260608  # today's date as the master seed (documented, not arbitrary)

CONFIG_PATH = ROOT / "config" / "ratings_elo_2026.yaml"


def _load_reprolog_module():
    asset = Path.home() / ".claude" / "skills" / "emit-repro-log" / "assets" / "emit_repro_log.py"
    if not asset.exists():
        return None
    spec = importlib.util.spec_from_file_location("emit_repro_log", asset)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # let the frozen dataclass resolve its own annotations
    spec.loader.exec_module(mod)
    return mod


def emit_repro_log(seed: int, config_sha256: str, grid_meta: dict) -> str:
    """Write a ReproLog; return its path (or a note if the canonical asset is absent)."""
    mod = _load_reprolog_module()
    config_resolved = hashlib.sha256(json.dumps(grid_meta, sort_keys=True).encode()).hexdigest()
    if mod is None:
        return "emit-repro-log asset unavailable; ReproLog skipped"
    log = mod.capture(
        phase="inference",
        hypothesis_id="n/a",
        rng_seed=seed,
        dataset_checksums={"config/ratings_elo_2026.yaml": config_sha256},
        config_resolved_sha256=config_resolved,
    )
    out = mod.ProjectPaths.discover().logs_reproducibility / f"repro_log_{log.run_id}.json"
    log.write(out)
    return str(out.relative_to(ROOT))


def build_configs(field, seed: int):
    configs = [(0, simulate.make_elo_config(field, name="elo_2026"))]
    spread0 = field.elo_spread
    for i, mult in enumerate(SYNTH_SPREAD_MULTIPLIERS, start=1):
        spread = mult * spread0
        name = f"synthetic_x{mult:g}"
        configs.append((i, simulate.make_synthetic_config(spread, name=name, seed=seed)))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run (small sim counts)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--regime", choices=["resampled", "fixed"], default="resampled")
    ap.add_argument("--no-bestresponse", action="store_true")
    args = ap.parse_args()

    if args.quick:
        n_draws, sims_per_draw = (args.draws or 4), (args.sims or 250)
        br_draws, br_sims = 3, 250
    else:
        n_draws, sims_per_draw = (args.draws or 25), (args.sims or 2000)
        br_draws, br_sims = 8, 1500

    field = load_field(CONFIG_PATH)
    grid_meta = {
        "N_VALUES": N_VALUES,
        "LADDERS": LADDERS,
        "MAIN_POLICIES": MAIN_POLICIES,
        "SYNTH_SPREAD_MULTIPLIERS": SYNTH_SPREAD_MULTIPLIERS,
        "n_draws": n_draws,
        "sims_per_draw": sims_per_draw,
        "regime": args.regime,
        "seed": args.seed,
    }
    repro = emit_repro_log(args.seed, field.config_sha256, grid_meta)
    print(f"ReproLog: {repro}")
    print(
        f"Grid: {n_draws} draws x {sims_per_draw} sims = "
        f"{n_draws * sims_per_draw} eval tournaments/cell, regime={args.regime}"
    )

    out_dir = ROOT / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    print("Main grid:")
    df_main = run_main_grid(field, n_draws, sims_per_draw, args.seed, args.regime)
    main_path = out_dir / f"results_main_{today}.csv"
    df_main.to_csv(main_path, index=False)
    print(f"  wrote {main_path.relative_to(ROOT)} ({len(df_main)} rows)")

    if not args.no_bestresponse:
        print("Exploitability probe:")
        df_exp = run_exploitability(field, br_draws, br_sims, args.seed, args.regime)
        exp_path = out_dir / f"results_exploitability_{today}.csv"
        df_exp.to_csv(exp_path, index=False)
        print(f"  wrote {exp_path.relative_to(ROOT)} ({len(df_exp)} rows)")


if __name__ == "__main__":
    main()
