"""Compare 6 vs 7 vs 8 participants under otherwise-identical conditions.

Holds the field (real 2026 Elo), seed, draws, roster size (6 teams each), and policy
(EV-greedy) fixed; varies only the number of participants. Uses the same resampled-draw +
cluster-SE machinery as the main study (cfg_id fixed so the simulated tournaments are
identical across participant counts — a controlled comparison). Writes a tidy table.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from wcpool import simulate  # noqa: E402
from wcpool.config import load_field  # noqa: E402

PARTICIPANTS = [6, 7, 8]
TEAMS_EACH = 6  # constant roster size (P=8 x 6 exactly fills the 48-team field)
LADDERS = ["linear", "triangular", "geometric"]
SEED = 20260608


def main() -> None:
    field = load_field(ROOT / "config" / "ratings_elo_2026.yaml")
    rows = []
    for p in PARTICIPANTS:
        cfg = simulate.make_elo_config(field, name="elo_2026")
        recs = simulate.run_strength_config(
            cfg,
            ladders=LADDERS,
            n_values=[TEAMS_EACH],
            policies=["ev_greedy"],
            n_draws=25,
            sims_per_draw=2000,
            seed=SEED,
            regime="resampled",
            cfg_id=0,  # identical tournaments across P -> controlled comparison
            n_drafters=p,
        )
        for r in recs:
            fair = 1.0 / p
            rows.append(
                {
                    "participants": p,
                    "ladder": r["ladder"],
                    "skill": round(r["spearman_mean"], 3),
                    "skill_se": round(r["spearman_cluster_se"], 3),
                    "tie_rate": round(r["top_tie_rate"], 4),
                    "1st_seat_winprob": round(r["slot1_win_prob"], 3),
                    "fair_share": round(fair, 3),
                    "1st_seat_vs_fair": round(r["slot1_win_prob"] / fair, 2),
                    "seat_spread": round(r["slot_win_prob_spread"], 3),
                    "champ_undrafted": round(r["champion_undrafted_rate"], 4),
                    "champ_holder_wins": round(r["p_champion_holder_wins"], 3),
                }
            )
        print(f"  done P={p}")
    df = pd.DataFrame(rows).sort_values(["ladder", "participants"])
    out = ROOT / "docs" / "tables" / f"participant_sweep_{date.today().isoformat()}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
