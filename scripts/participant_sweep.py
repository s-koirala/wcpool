"""Compare 6 vs 7 vs 8 participants under otherwise-identical conditions.

Holds the field (real 2026 Elo), seed, draws, roster size (6 teams each), and policy
(EV-greedy) fixed; varies only the number of participants. cfg_id is fixed so the simulated
tournaments are IDENTICAL across participant counts (a controlled, *paired* comparison):
each draw is the same tournament for every P, so the skill change vs 6 players is tested
with a paired between-draw standard error, not the weaker independent-sample SE.

Reports, per (participants, ladder): the skill correlation and its per-P cluster SE; the
variance components (between-roster vs within-tournament-noise) and the skill-variance share
that drive the skill change; the paired skill delta vs 6 players with its paired SE and
z-score; tie rate; first-seat win probability and its multiple of the fair share (1/P);
seat spread; champion-undrafted rate; and champion-holder-wins.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpool import simulate  # noqa: E402
from wcpool.config import load_field  # noqa: E402

PARTICIPANTS = [6, 7, 8]
BASELINE = 6
TEAMS_EACH = 6  # constant roster size (P=8 x 6 exactly fills the 48-team field)
LADDERS = ["linear", "triangular", "geometric"]
SEED = 20260608


def main() -> None:
    field = load_field(ROOT / "config" / "ratings_elo_2026.yaml")
    by_record: dict[int, dict] = {}  # p -> {ladder: record}
    by_draw: dict[int, dict] = {}  # p -> {ladder: per-draw spearman array}
    for p in PARTICIPANTS:
        cfg = simulate.make_elo_config(field, name="elo_2026")
        recs, per_draw = simulate.run_strength_config(
            cfg,
            ladders=LADDERS,
            n_values=[TEAMS_EACH],
            policies=["ev_greedy"],
            n_draws=25,
            sims_per_draw=2000,
            seed=SEED,
            regime="resampled",
            cfg_id=0,  # identical tournaments across P -> paired comparison
            n_drafters=p,
            return_per_draw=True,
        )
        by_record[p] = {r["ladder"]: r for r in recs}
        by_draw[p] = {ladder: per_draw[(ladder, TEAMS_EACH, "ev_greedy")] for ladder in LADDERS}
        print(f"  done P={p}")

    rows = []
    for ladder in LADDERS:
        base_draw = by_draw[BASELINE][ladder]
        for p in PARTICIPANTS:
            r = by_record[p][ladder]
            fair = 1.0 / p
            # paired skill delta vs baseline (same tournaments per draw -> per-draw diffs)
            diffs = by_draw[p][ladder] - base_draw
            n = len(diffs)
            delta = float(np.mean(diffs))
            se_paired = float(np.std(diffs, ddof=1) / np.sqrt(n)) if p != BASELINE else float("nan")
            z = delta / se_paired if (p != BASELINE and se_paired > 0) else float("nan")
            rows.append(
                {
                    "participants": p,
                    "ladder": ladder,
                    "skill": round(r["spearman_mean"], 3),
                    "skill_se": round(r["spearman_cluster_se"], 3),
                    "var_between": round(r["sigma2_between"], 2),
                    "var_within": round(r["sigma2_within"], 2),
                    "skill_var_share": round(r["skill_variance_share"], 3),
                    "skill_delta_vs6": round(delta, 3),
                    "delta_se_paired": round(se_paired, 4),
                    "z_vs6": round(z, 1) if np.isfinite(z) else float("nan"),
                    "tie_rate": round(r["top_tie_rate"], 4),
                    "first_seat": round(r["slot1_win_prob"], 3),
                    "fair_share": round(fair, 3),
                    "first_seat_vs_fair": round(r["slot1_win_prob"] / fair, 2),
                    "seat_spread": round(r["slot_win_prob_spread"], 3),
                    "champ_undrafted": round(r["champion_undrafted_rate"], 4),
                    "champ_holder_wins": round(r["p_champion_holder_wins"], 3),
                }
            )
    df = pd.DataFrame(rows)
    out = ROOT / "docs" / "tables" / f"participant_sweep_{date.today().isoformat()}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
