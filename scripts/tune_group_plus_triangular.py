"""Choose the scale that marries group W/D/L (3/1/0) to the triangular knockout ladder.

The combined team score is::

    total = group_match_points(3/1/0, 0..9)  +  s * TRIANGULAR[furthest_stage]

with TRIANGULAR = [0,1,3,6,10,15,21] (GROUP..CHAMPION). Group W/D/L is fixed at the FIFA
3/1/0 standard (the one citable, non-arbitrary anchor); the only free parameter is the
triangular multiplier ``s`` (points per triangular unit), which sets how much group form
matters relative to knockout progression. This script sweeps ``s`` and reports, for an
8-player pool:

  * coherence -- the cross-stage INVERSION rate: P(a team that exited earlier outscores a
    team that went further), purely on group points. ``inv_any`` counts any earlier-stage
    pair; ``inv_gross`` counts only gaps of >=2 stages (e.g. a group-stage team outscoring a
    quarter-finalist) -- the egregious kind. Analytic fact: ``s >= 10`` => both are 0.
  * design -- tie rate, draft skill (Spearman), slot-win-probability spread, champion
    dominance, and the first-seat win probability (assumptions.md sec.7 definitions).

Tournaments are simulated ONCE per draw and re-scored for every ``s`` (the sims do not depend
on ``s``), so the comparison is paired and cheap. Emits a ReproLog; writes a tidy CSV.
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
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpool import draft as draft_mod  # noqa: E402
from wcpool import metrics  # noqa: E402
from wcpool import simulate as sim_mod  # noqa: E402
from wcpool.config import load_field  # noqa: E402
from wcpool.ladders import TRIANGULAR  # noqa: E402
from wcpool.tournament import N_TEAMS, random_pot_draw, simulate_tournament  # noqa: E402

CONFIG_PATH = ROOT / "config" / "ratings_elo_2026.yaml"
DEFAULT_SEED = 20260608
N_DRAFTERS = 8
TEAMS_EACH = N_TEAMS // N_DRAFTERS  # 6
S_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10]  # triangular multipliers; s>=10 is the analytic bound
MAX_GROUP_POINTS = 9  # 3 wins; the coherence bound is s * min_increment(=1) > this


def _load_reprolog_module():
    asset = Path.home() / ".claude" / "skills" / "emit-repro-log" / "assets" / "emit_repro_log.py"
    if not asset.exists():
        return None
    spec = importlib.util.spec_from_file_location("emit_repro_log", asset)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def emit_repro_log(seed, config_sha256, run_meta) -> str:
    mod = _load_reprolog_module()
    config_resolved = hashlib.sha256(json.dumps(run_meta, sort_keys=True).encode()).hexdigest()
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


# --- Coherence: cross-stage inversion rate ----------------------------------------------


def inversion_rates(stages: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    """(inv_any, inv_gross) over a (n_sims, n_teams) batch.

    inv_any   = P(total_i > total_j | stage_i  <  stage_j)          any earlier-stage pair
    inv_gross = P(total_i > total_j | stage_i <= stage_j - 2)       a >=2-round-earlier exit
    """
    s_i, s_j = stages[:, :, None], stages[:, None, :]
    t_i, t_j = totals[:, :, None], totals[:, None, :]
    earlier = s_i < s_j
    gross = s_i <= s_j - 2
    outscores = t_i > t_j
    inv_any = float((earlier & outscores).sum() / max(earlier.sum(), 1))
    inv_gross = float((gross & outscores).sum() / max(gross.sum(), 1))
    return inv_any, inv_gross


def coherence_sweep(cfg, n_sims, seed):
    """Inversion rates per s under the official draw (one simulated batch, re-scored)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(99, 0, 0)))
    stages, group_pts = simulate_tournament(
        cfg.model, cfg.fixed_groups, n_sims, rng, return_group_points=True
    )
    group_pts = group_pts.astype(np.float64)
    out = {}
    for s in S_GRID:
        totals = group_pts + s * TRIANGULAR[stages]
        out[s] = inversion_rates(stages, totals)
    return out


# --- Design metrics: resampled 8-player pool, sims shared across s -----------------------


def _champ_credit(scores, champ, drafter_of_team):
    holder = drafter_of_team[champ]
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    n_tied = is_top.sum(axis=1)
    rows = np.arange(scores.shape[0])
    return np.where(is_top[rows, holder], 1.0 / n_tied, 0.0)


def design_sweep(cfg, draws, sims, seed):
    """Per-s pooled design metrics; tournaments simulated once per draw, re-scored per s."""
    acc = {s: {"scores": [], "rho": [], "credit": []} for s in S_GRID}
    for d in range(draws):
        rng_ev = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, d, 1)))
        rng_eval = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, d, 2)))
        rng_draw = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, d, 0)))
        groups = random_pot_draw(cfg.pots, rng_draw)
        s_ev, g_ev = simulate_tournament(cfg.model, groups, sims, rng_ev, return_group_points=True)
        s_eval, g_eval = simulate_tournament(
            cfg.model, groups, sims, rng_eval, return_group_points=True
        )
        g_ev, g_eval = g_ev.astype(np.float64), g_eval.astype(np.float64)
        for s in S_GRID:
            tri = s * TRIANGULAR
            team_ev = (g_ev + tri[s_ev]).mean(axis=0)
            res = draft_mod.draft_ev_greedy(team_ev, N_DRAFTERS, TEAMS_EACH)
            rosters = res["rosters"]
            scores = metrics.pool_scores(g_eval + tri[s_eval], rosters)
            acc[s]["scores"].append(scores)
            acc[s]["rho"].append(metrics.spearman_per_sim(scores, team_ev, rosters))
            champ = sim_mod.champion_team(s_eval)
            acc[s]["credit"].append(_champ_credit(scores, champ, res["drafter_of_team"]))

    rows = {}
    for s in S_GRID:
        scores = np.vstack(acc[s]["scores"])
        rho = np.concatenate(acc[s]["rho"])
        valid = rho[~np.isnan(rho)]
        wp = metrics.win_probability(scores)
        rows[s] = {
            "skill_rho": float(np.mean(valid)) if valid.size else float("nan"),
            "tie_rate": metrics.top_tie_rate(scores),
            "slot_spread": metrics.slot_equity_imbalance(scores)["max_minus_min"],
            "champ_dom": float(np.mean(np.concatenate(acc[s]["credit"]))),
            "slot1_win_prob": float(wp[0]),
        }
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--coh-sims", type=int, default=None)
    args = ap.parse_args()

    if args.quick:
        draws, sims, coh = (args.draws or 4), (args.sims or 600), (args.coh_sims or 1500)
    else:
        draws, sims, coh = (args.draws or 15), (args.sims or 1500), (args.coh_sims or 5000)

    field = load_field(CONFIG_PATH)
    cfg = sim_mod.make_elo_config(field, name="elo_2026")
    run_meta = {
        "group_points": "3/1/0",
        "knockout": "s * triangular[0,1,3,6,10,15,21]",
        "s_grid": S_GRID,
        "n_drafters": N_DRAFTERS,
        "draws": draws,
        "sims": sims,
        "coh_sims": coh,
        "seed": args.seed,
    }
    repro = emit_repro_log(args.seed, field.config_sha256, run_meta)

    coh_rows = coherence_sweep(cfg, coh, args.seed)
    des_rows = design_sweep(cfg, draws, sims, args.seed)

    table = []
    for s in S_GRID:
        ko = (s * TRIANGULAR[1:]).astype(int).tolist()  # R32..CHAMPION points
        inv_any, inv_gross = coh_rows[s]
        d = des_rows[s]
        table.append(
            {
                "s": s,
                "KO pts R32..Champ": ",".join(map(str, ko)),
                "inv_any": inv_any,
                "inv_gross": inv_gross,
                "tie_rate": d["tie_rate"],
                "skill_rho": d["skill_rho"],
                "slot_spread": d["slot_spread"],
                "champ_dom": d["champ_dom"],
                "slot1_wp": d["slot1_win_prob"],
            }
        )
    df = pd.DataFrame(table)

    print("=" * 100)
    print("GROUP 3/1/0  +  s x TRIANGULAR[1,3,6,10,15,21]  -- choosing the scale s (8 players)")
    print("=" * 100)
    print(f"ReproLog: {repro}")
    print(f"Coherence: official draw, {coh:,} sims | Design: resampled {draws}x{sims}="
          f"{draws * sims:,}/cell | fair share 1/8={1 / N_DRAFTERS:.3f}")
    print(f"Analytic bound: s*1 > {MAX_GROUP_POINTS} (max group haul) => s>=10 gives zero "
          f"cross-stage inversions.\n")
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="github",
            floatfmt=("", "", "", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"),
            showindex=False,
        )
    )

    out_dir = ROOT / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    path = out_dir / f"tune_group_triangular_scale{tag}_{date.today().isoformat()}.csv"
    df.to_csv(path, index=False)
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
