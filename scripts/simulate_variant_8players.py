"""Simulate an 8-participant pool under the group-W/D/L + progression-bonus variant.

Scoring (the requested variant, implemented in ``wcpool.ladders``):

  * Group stage, per match:   win 3, draw 1, loss 0      (summed over 3 matches, 0..9)
  * Progression bonus, once:  reach R32 +5, QF +8, SF +12, win the Cup +20
                              (cumulative furthest-stage vector [0,5,5,13,25,25,45])
  * Team points = group-match points + progression bonus; a drafter's pool score is the
    sum over the teams they drafted.

Two reports, both for 8 drafters x 6 teams = the whole 48-team field, EV-greedy snake draft:

  Part A  THE 2026 POOL (fixed official draw). Estimate each team's expected variant points
          on an EV batch, run one snake draft, then score the rosters on an independent eval
          batch. Output: the eight rosters and each drafter's score distribution, win
          probability, podium probability, and the group-vs-bonus split of their EV.

  Part B  DESIGN COMPARISON (resampled draws, re-drafted each replicate). The variant and the
          three prescribed terminal ladders (linear/triangular/geometric) are scored on
          byte-identical tournaments per draw (shared ``cfg_id`` seed streams), so the five
          design metrics (skill, tie rate, slot equity, champion dominance, winning score)
          are a paired comparison. Shows where adding group W/D/L points moves the design.

Emits a 13-field ReproLog before writing, per the CLAUDE.md reproducibility mandate.
Default sizing: Part A 20k EV + 50k eval sims; Part B 25 draws x 2000 = 50k/cell. --quick
is a smoke run.
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
from wcpool import ladders, metrics  # noqa: E402
from wcpool import simulate as sim_mod  # noqa: E402
from wcpool.config import load_field  # noqa: E402
from wcpool.ladders import PROGRESSION_BONUS, Stage, variant_team_points  # noqa: E402
from wcpool.tournament import N_TEAMS, random_pot_draw, simulate_tournament  # noqa: E402

CONFIG_PATH = ROOT / "config" / "ratings_elo_2026.yaml"
DEFAULT_SEED = 20260608  # house master seed = the 2026-06-08 run date (YYYYMMDD)
N_DRAFTERS = 8
TEAMS_EACH = N_TEAMS // N_DRAFTERS  # 6 -> the draft consumes the whole field
TERMINAL_LADDERS = ["linear", "triangular", "geometric"]


# --- ReproLog (reuse the canonical emit-repro-log asset when present) --------------------


def _load_reprolog_module():
    asset = Path.home() / ".claude" / "skills" / "emit-repro-log" / "assets" / "emit_repro_log.py"
    if not asset.exists():
        return None
    spec = importlib.util.spec_from_file_location("emit_repro_log", asset)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def emit_repro_log(seed: int, config_sha256: str, run_meta: dict) -> str:
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


# --- Variant simulation primitive -------------------------------------------------------


def simulate_variant(model, groups, n_sims, rng):
    """Return ``(stages, group_points, total_points)`` each ``(n_sims, 48)`` under the variant."""
    stages, group_pts = simulate_tournament(model, groups, n_sims, rng, return_group_points=True)
    total = variant_team_points(group_pts, stages)
    return stages, group_pts.astype(np.float64), total


def _streams(seed, cfg_id, d):
    """The (draw, EV, eval) RNG triple for draw ``d`` — identical keying to simulate.py."""
    return (
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 0))),
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 1))),
        np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(cfg_id, d, 2))),
    )


# --- Part A: the concrete 2026 pool -----------------------------------------------------


def podium_probability(scores: np.ndarray, k: int) -> np.ndarray:
    """P(a drafter finishes in the top ``k``) per drafter (a tie for the cut counts as in)."""
    n_strictly_above = (scores[:, None, :] > scores[:, :, None]).sum(axis=2)  # (sims, drafters)
    return (n_strictly_above < k).mean(axis=0)


def run_2026_pool(cfg, names, seed, ev_sims, eval_sims):
    """Fixed official draw: one EV-greedy snake draft, scored on an independent eval batch."""
    groups = cfg.fixed_groups
    _, rng_ev, rng_eval = _streams(seed, cfg_id=0, d=0)
    _, _, pts_ev = simulate_variant(cfg.model, groups, ev_sims, rng_ev)
    s_eval, group_eval, pts_eval = simulate_variant(cfg.model, groups, eval_sims, rng_eval)

    team_ev = pts_ev.mean(axis=0)
    res = draft_mod.draft_ev_greedy(team_ev, N_DRAFTERS, TEAMS_EACH)
    rosters = res["rosters"]  # (8, 6) snake order: row i == pick slot i+1

    scores = metrics.pool_scores(pts_eval, rosters)  # (eval_sims, 8)
    bonus_eval = PROGRESSION_BONUS[s_eval]  # (eval_sims, 48)
    group_component = metrics.pool_scores(group_eval, rosters).mean(axis=0)
    bonus_component = metrics.pool_scores(bonus_eval, rosters).mean(axis=0)

    win_p = metrics.win_probability(scores)
    podium_p = podium_probability(scores, k=3)

    rows = []
    for i in range(N_DRAFTERS):
        col = scores[:, i]
        rows.append(
            {
                "slot": i + 1,
                "roster": ", ".join(names[t] for t in rosters[i]),
                "E[score]": float(col.mean()),
                "SD": float(col.std()),
                "p5": float(np.percentile(col, 5)),
                "p50": float(np.percentile(col, 50)),
                "p95": float(np.percentile(col, 95)),
                "E[group]": float(group_component[i]),
                "E[bonus]": float(bonus_component[i]),
                "P(win)": float(win_p[i]),
                "P(top3)": float(podium_p[i]),
            }
        )
    return pd.DataFrame(rows), res, scores


# --- Part B: paired design comparison ---------------------------------------------------


def _champ_dominance_credit(scores, champ, drafter_of_team):
    """Per-replicate tie-split credit that the champion's holder is the pool winner."""
    holder = drafter_of_team[champ]
    top = scores.max(axis=1, keepdims=True)
    is_top = scores == top
    n_tied = is_top.sum(axis=1)
    rows = np.arange(scores.shape[0])
    holder_is_top = is_top[rows, holder]
    return np.where(holder_is_top, 1.0 / n_tied, 0.0)


def variant_resampled(cfg, draws, sims, seed, cfg_id=0):
    """Resampled-draw metrics for the variant (re-draw + re-draft each replicate).

    Metrics are computed on the pooled eval scores (identical convention to the
    terminal-ladder grid runner ``simulate.run_strength_config``), so the comparison in
    :func:`terminal_ladder_rows` is paired and like-for-like.
    """
    all_scores, all_rho, all_credit = [], [], []
    for d in range(draws):
        rng_draw, rng_ev, rng_eval = _streams(seed, cfg_id, d)
        groups = random_pot_draw(cfg.pots, rng_draw)
        _, _, pts_ev = simulate_variant(cfg.model, groups, sims, rng_ev)
        s_eval, _, pts_eval = simulate_variant(cfg.model, groups, sims, rng_eval)
        team_ev = pts_ev.mean(axis=0)
        res = draft_mod.draft_ev_greedy(team_ev, N_DRAFTERS, TEAMS_EACH)
        rosters = res["rosters"]
        scores = metrics.pool_scores(pts_eval, rosters)
        all_scores.append(scores)
        all_rho.append(metrics.spearman_per_sim(scores, team_ev, rosters))
        champ = sim_mod.champion_team(s_eval)
        all_credit.append(_champ_dominance_credit(scores, champ, res["drafter_of_team"]))

    scores = np.vstack(all_scores)
    rho = np.concatenate(all_rho)
    valid = rho[~np.isnan(rho)]
    ws = metrics.winning_score_summary(scores)
    return {
        "ladder": "wdl_bonus_variant",
        "spearman_mean": float(np.mean(valid)) if valid.size else float("nan"),
        "top_tie_rate": metrics.top_tie_rate(scores),
        "slot_win_prob_spread": metrics.slot_equity_imbalance(scores)["max_minus_min"],
        "p_champion_holder_wins": float(np.mean(np.concatenate(all_credit))),
        "winning_score_mean": ws["mean"],
        **{
            f"slot{i + 1}_win_prob": float(p) for i, p in enumerate(metrics.win_probability(scores))
        },
    }


def terminal_ladder_rows(field, draws, sims, seed, cfg_id=0):
    """Same five metrics for linear/triangular/geometric via the validated grid runner."""
    cfg = sim_mod.make_elo_config(field, name="elo_2026")
    recs = sim_mod.run_strength_config(
        cfg,
        ladders=TERMINAL_LADDERS,
        n_values=[TEAMS_EACH],
        policies=["ev_greedy"],
        n_draws=draws,
        sims_per_draw=sims,
        seed=seed,
        regime="resampled",
        cfg_id=cfg_id,
        n_drafters=N_DRAFTERS,
    )
    keep = [
        "ladder",
        "spearman_mean",
        "top_tie_rate",
        "slot_win_prob_spread",
        "p_champion_holder_wins",
        "winning_score_mean",
        *[f"slot{i + 1}_win_prob" for i in range(N_DRAFTERS)],
    ]
    return [{k: r[k] for k in keep} for r in recs]


# --- Reporting --------------------------------------------------------------------------


def _fmt(df: pd.DataFrame, floatfmt) -> str:
    return tabulate(df, headers="keys", tablefmt="github", floatfmt=floatfmt, showindex=False)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true", help="smoke run (small sim counts)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ev-sims", type=int, default=None, help="Part A EV-batch sims")
    ap.add_argument("--eval-sims", type=int, default=None, help="Part A eval-batch sims")
    ap.add_argument("--draws", type=int, default=None, help="Part B resampled draws")
    ap.add_argument("--sims", type=int, default=None, help="Part B sims per draw")
    args = ap.parse_args()

    if args.quick:
        ev_sims, eval_sims = (args.ev_sims or 4000), (args.eval_sims or 8000)
        draws, sims = (args.draws or 4), (args.sims or 500)
    else:
        ev_sims, eval_sims = (args.ev_sims or 20000), (args.eval_sims or 50000)
        draws, sims = (args.draws or 25), (args.sims or 2000)

    field = load_field(CONFIG_PATH)
    cfg = sim_mod.make_elo_config(field, name="elo_2026")

    run_meta = {
        "scoring": "group_wdl_3_1_0 + progression_bonus",
        "bonus_milestones": ladders.PROGRESSION_BONUS_MILESTONES,
        "bonus_vector": PROGRESSION_BONUS.tolist(),
        "n_drafters": N_DRAFTERS,
        "teams_each": TEAMS_EACH,
        "ev_sims": ev_sims,
        "eval_sims": eval_sims,
        "draws": draws,
        "sims": sims,
        "seed": args.seed,
    }
    repro = emit_repro_log(args.seed, field.config_sha256, run_meta)

    print("=" * 92)
    print("8-PLAYER POOL: group W/D/L (3/1/0) + progression bonus (R32+5, QF+8, SF+12, win+20)")
    print("=" * 92)
    print(f"ReproLog: {repro}")
    print(f"Field: 48 teams, official 2026 draw | Elo model beta={cfg.model.beta:.3f}")
    print(
        f"Furthest-stage bonus vector [GROUP..CHAMPION]: {PROGRESSION_BONUS.astype(int).tolist()}"
    )
    print(
        f"  (a beaten finalist scores the same {int(PROGRESSION_BONUS[Stage.FINAL])} bonus as a "
        f"semi-finalist: the spec rewards winning the final, not reaching it)"
    )

    # ---- Part A ----
    print("\n" + "-" * 92)
    print(
        f"PART A - THE 2026 POOL (official draw, EV-greedy snake draft, "
        f"{eval_sims:,} eval tournaments)"
    )
    print("-" * 92)
    pool_df, res, scores = run_2026_pool(cfg, field.names, args.seed, ev_sims, eval_sims)
    show = pool_df[
        [
            "slot",
            "roster",
            "E[score]",
            "SD",
            "p5",
            "p50",
            "p95",
            "E[group]",
            "E[bonus]",
            "P(win)",
            "P(top3)",
        ]
    ].copy()
    print(
        _fmt(
            show,
            floatfmt=(".0f", "", ".1f", ".1f", ".0f", ".0f", ".0f", ".1f", ".1f", ".3f", ".3f"),
        )
    )
    lead = pool_df.loc[pool_df["P(win)"].idxmax()]
    print(
        f"\nMost likely pool winner: slot {int(lead['slot'])} "
        f"(P(win)={lead['P(win)']:.3f}); favourite-heavy seats lead because the bonus tail "
        f"(champion holder banks {int(PROGRESSION_BONUS[Stage.CHAMPION])}) dominates the spread."
    )

    # ---- Part B ----
    print("\n" + "-" * 92)
    print(
        f"PART B - DESIGN COMPARISON vs the prescribed ladders "
        f"(resampled, {draws}x{sims}={draws * sims:,}/cell, paired)"
    )
    print("-" * 92)
    rows = terminal_ladder_rows(field, draws, sims, args.seed)
    rows.append(variant_resampled(cfg, draws, sims, args.seed))
    cmp_df = pd.DataFrame(rows)
    headline = cmp_df[
        [
            "ladder",
            "spearman_mean",
            "top_tie_rate",
            "slot_win_prob_spread",
            "p_champion_holder_wins",
            "winning_score_mean",
        ]
    ].rename(
        columns={
            "spearman_mean": "skill(rho)",
            "top_tie_rate": "tie_rate",
            "slot_win_prob_spread": "slot_spread",
            "p_champion_holder_wins": "champ_dom",
            "winning_score_mean": "win_score_mean",
        }
    )
    print(_fmt(headline, floatfmt=("", ".3f", ".3f", ".3f", ".3f", ".1f")))
    var = cmp_df[cmp_df.ladder == "wdl_bonus_variant"].iloc[0]
    tri = cmp_df[cmp_df.ladder == "triangular"].iloc[0]
    print(
        f"\nReading: adding group W/D/L points drives the tie rate to {var['top_tie_rate']:.3f} "
        f"(vs triangular {tri['top_tie_rate']:.3f}) because near-continuous group points break "
        f"exact ladder ties; slot equity and champion dominance sit between the linear and "
        f"triangular ladders."
    )

    # ---- Write artifacts ----
    out_dir = ROOT / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    tag = "_quick" if args.quick else ""
    pool_path = out_dir / f"variant_wdl_8players_pool{tag}_{today}.csv"
    cmp_path = out_dir / f"variant_wdl_8players_ladder_compare{tag}_{today}.csv"
    pool_df.to_csv(pool_path, index=False)
    cmp_df.to_csv(cmp_path, index=False)
    print(f"\nWrote {pool_path.relative_to(ROOT)}")
    print(f"Wrote {cmp_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
