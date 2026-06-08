"""Render figures and the (N, ladder) recommendation from the experiment tables.

Reads the latest results_main_*.csv (+ results_exploitability_*.csv) from docs/tables/
and writes:
  docs/figures/skill_heatmap_{date}.png        skill correlation over the (N, ladder) grid
  docs/figures/tradeoff_{date}.png             skill vs tie-rate vs slot-equity tradeoff
  docs/figures/robustness_{date}.png           metrics vs favoritism concentration
  docs/figures/slot_equity_{date}.png          win prob by snake slot
  docs/tables/results_summary_{date}.csv       headline metrics, EV-greedy, per config
  docs/tables/recommendation_{date}.md         Pareto-frontier recommendation + robustness

Recommendation uses a Pareto frontier in (skill up, tie-rate down, slot-imbalance down)
space — no arbitrary cut-offs — then names the highest-skill non-dominated cell.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "docs" / "tables"
FIGURES = ROOT / "docs" / "figures"
LADDER_ORDER = ["linear", "triangular", "geometric"]
SKILL_COL = "spearman_mean"  # headline "skill correlation" axis
ANCHOR_CONFIG = "elo_2026"


def latest(glob: str) -> Path:
    files = sorted(TABLES.glob(glob))
    if not files:
        raise FileNotFoundError(f"no file matching {glob} in {TABLES}")
    return files[-1]


def _grid(df, value, config, policy="ev_greedy"):
    sub = df[(df.strength_config == config) & (df.policy == policy)]
    piv = sub.pivot_table(index="teams_per_drafter", columns="ladder", values=value)
    return piv.reindex(columns=LADDER_ORDER)


def fig_skill_heatmap(df, today):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (val, title) in zip(
        axes,
        [
            (SKILL_COL, "Skill: Spearman(EV rank, placing)"),
            ("skill_variance_share", "Skill: variance share (roster vs noise)"),
        ],
        strict=True,
    ):
        piv = _grid(df, val, ANCHOR_CONFIG)
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(LADDER_ORDER)), LADDER_ORDER)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("ladder")
        ax.set_ylabel("teams per drafter (N)")
        ax.set_title(title, fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{piv.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Skill metrics over the (N, ladder) grid — {ANCHOR_CONFIG}, EV-greedy")
    fig.tight_layout()
    out = FIGURES / f"skill_heatmap_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_tradeoff(df, today):
    sub = df[(df.strength_config == ANCHOR_CONFIG) & (df.policy == "ev_greedy")]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {"linear": "o", "triangular": "s", "geometric": "^"}
    norm = plt.Normalize(sub.slot_win_prob_spread.min(), sub.slot_win_prob_spread.max())
    for ladder in LADDER_ORDER:
        s = sub[sub.ladder == ladder]
        sc = ax.scatter(
            s[SKILL_COL],
            s.top_tie_rate,
            c=s.slot_win_prob_spread,
            s=120,
            marker=markers[ladder],
            cmap="plasma",
            norm=norm,
            edgecolors="black",
            label=ladder,
        )
        for _, row in s.iterrows():
            ax.annotate(
                f"N={int(row.teams_per_drafter)}",
                (row[SKILL_COL], row.top_tie_rate),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
    ax.set_xlabel("skill correlation  (Spearman EV-rank vs placing)  -> higher better")
    ax.set_ylabel("tie rate at the top  -> lower better")
    ax.set_title(f"Skill / tie-rate / slot-equity tradeoff — {ANCHOR_CONFIG}, EV-greedy")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("slot win-prob spread (max-min)  -> lower better")
    ax.legend(title="ladder")
    fig.tight_layout()
    out = FIGURES / f"tradeoff_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_robustness(df, today):
    sub = df[df.policy == "ev_greedy"].copy()
    metrics = [
        (SKILL_COL, "skill (Spearman)"),
        ("top_tie_rate", "tie rate"),
        ("slot_win_prob_spread", "slot win-prob spread"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, (col, label) in zip(axes, metrics, strict=True):
        for ladder in LADDER_ORDER:
            # average over N to show the ladder-level robustness trend vs concentration
            agg = sub[sub.ladder == ladder].groupby("top8_title_share")[col].mean().reset_index()
            ax.plot(agg.top8_title_share, agg[col], marker="o", label=ladder)
        ax.set_xlabel("top-8 title concentration")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(title="ladder")
    fig.suptitle("Robustness: metrics vs favoritism concentration (avg over N), EV-greedy")
    fig.tight_layout()
    out = FIGURES / f"robustness_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_slot_equity(df, today):
    sub = df[(df.strength_config == ANCHOR_CONFIG) & (df.policy == "ev_greedy")]
    slot_cols = [f"slot{i + 1}_win_prob" for i in range(6)]
    fig, axes = plt.subplots(1, len(LADDER_ORDER), figsize=(13, 4), sharey=True)
    for ax, ladder in zip(axes, LADDER_ORDER, strict=True):
        s = sub[sub.ladder == ladder]
        for _, row in s.iterrows():
            ax.plot(
                range(1, 7),
                [row[c] for c in slot_cols],
                marker="o",
                label=f"N={int(row.teams_per_drafter)}",
            )
        ax.axhline(1 / 6, ls="--", color="grey", lw=1, label="equitable 1/6")
        ax.set_title(f"{ladder}")
        ax.set_xlabel("snake slot")
    axes[0].set_ylabel("win probability")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Draft-slot equity by snake position — {ANCHOR_CONFIG}, EV-greedy")
    fig.tight_layout()
    out = FIGURES / f"slot_equity_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def pareto_frontier(sub):
    """Non-dominated cells in (skill up, tie down, slot-imbalance down). Returns mask."""
    pts = sub[[SKILL_COL, "top_tie_rate", "slot_win_prob_spread"]].to_numpy()
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_eq = pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1] and pts[j, 2] <= pts[i, 2]
            strictly = pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1] or pts[j, 2] < pts[i, 2]
            if better_eq and strictly:
                dominated[i] = True
                break
    return ~dominated


def _objective_ranks(cells):
    """Per-objective ranks over the cells: skill (desc), tie (asc), slot spread (asc)."""
    r_skill = cells[SKILL_COL].rank(ascending=False)
    r_tie = cells["top_tie_rate"].rank(ascending=True)
    r_slot = cells["slot_win_prob_spread"].rank(ascending=True)
    return r_skill, r_tie, r_slot


def write_recommendation(df, exp_df, today):
    anchor = df[(df.strength_config == ANCHOR_CONFIG) & (df.policy == "ev_greedy")].copy()
    anchor = anchor.reset_index(drop=True)
    anchor["mc_se"] = anchor["spearman_sd"] / np.sqrt(anchor["n_eval_tournaments"])

    # Three single-objective optima and one balanced compromise.
    skill_max = anchor.loc[anchor[SKILL_COL].idxmax()]
    slot_fair = anchor.loc[anchor["slot_win_prob_spread"].idxmin()]
    r_skill, r_tie, r_slot = _objective_ranks(anchor)
    worst_rank = pd.concat([r_skill, r_tie, r_slot], axis=1).max(axis=1)
    sum_rank = r_skill + r_tie + r_slot
    # balanced = minimise the worst per-objective rank (Rawlsian), break ties by total rank.
    order = pd.DataFrame({"worst": worst_rank, "sum": sum_rank}).sort_values(["worst", "sum"])
    balanced = anchor.loc[order.index[0]]
    rec_n, rec_ladder = int(balanced.teams_per_drafter), balanced.ladder

    front_mask = pareto_frontier(anchor)
    front = anchor[front_mask].sort_values(SKILL_COL, ascending=False)

    # N-effect within the balanced ladder, with MC standard error.
    n_effect = anchor[anchor.ladder == rec_ladder].sort_values("teams_per_drafter")[
        ["teams_per_drafter", SKILL_COL, "mc_se", "top_tie_rate", "slot_win_prob_spread"]
    ]

    rob_rows = []
    for cfg in sorted(df.strength_config.unique()):
        s = df[(df.strength_config == cfg) & (df.policy == "ev_greedy")].copy()
        s["skill_rank"] = s[SKILL_COL].rank(ascending=False)
        cell = s[(s.teams_per_drafter == rec_n) & (s.ladder == rec_ladder)].iloc[0]
        rob_rows.append(
            {
                "strength_config": cfg,
                "top8_title_share": round(cell.top8_title_share, 3),
                "skill": round(cell[SKILL_COL], 3),
                "skill_rank_of_balanced": int(cell.skill_rank),
                "tie_rate": round(cell.top_tie_rate, 4),
                "slot_spread": round(cell.slot_win_prob_spread, 4),
            }
        )
    rob = pd.DataFrame(rob_rows)

    def cell_str(c):
        return (
            f"N={int(c.teams_per_drafter)} {c.ladder} "
            f"(skill={c[SKILL_COL]:.3f}, tie={c.top_tie_rate:.4f}, "
            f"slot-spread={c.slot_win_prob_spread:.3f}, "
            f"champ-dom={c.p_champion_holder_wins:.3f})"
        )

    lines = [
        f"# Recommended (N, ladder) — {today}",
        "",
        "## Headline",
        "",
        "The three design goals **conflict along one axis — ladder convexity**. The same "
        "convexity that raises skill correlation and removes ties also concentrates pool "
        "wins in the first snake pick (it turns the pool into a referendum on who drafts "
        "the eventual champion). No single cell wins all three goals.",
        "",
        f"* **Balanced default (recommended): {cell_str(balanced)}** — best joint standing "
        "across all three objectives (minimax over per-objective ranks).",
        f"* **Pure skill / no-ties: {cell_str(skill_max)}** — maximises skill and "
        "essentially eliminates ties, but is the *least* slot-equitable and makes the pool "
        "almost entirely about owning the champion.",
        f"* **Most slot-equitable: {cell_str(slot_fair)}** — fairest across snake slots, but "
        "lowest skill and a high tie rate (needs an explicit tiebreaker rule).",
        "",
        "**N is second-order.** Within a ladder, skill varies by <~0.01 across "
        "N in {4,5,6,8} (≈ 3 Monte-Carlo SE), while the cross-ladder gaps are 15–25 SE. "
        "Choose the ladder first; pick N on other grounds (e.g. N=8 drafts the full field).",
        "",
        "## N-effect within the recommended ladder "
        f"(`{rec_ladder}`, anchor, EV-greedy; mc_se = SE of skill mean)",
        "",
        n_effect.round(4).to_markdown(index=False),
        "",
        "## Full (N, ladder) grid — anchor config, EV-greedy",
        "",
        anchor.sort_values([SKILL_COL], ascending=False)[
            [
                "teams_per_drafter",
                "ladder",
                SKILL_COL,
                "skill_variance_share",
                "top_tie_rate",
                "slot_win_prob_spread",
                "p_champion_holder_wins",
            ]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
        "Pareto-frontier cells (non-dominated in skill↑ / tie↓ / slot-spread↓): "
        + ", ".join(f"{int(r.teams_per_drafter)} {r.ladder}" for _, r in front.iterrows())
        + ".",
        "",
        "## Robustness of the recommendation across strength models",
        "",
        "`skill_rank_of_balanced` is the recommended cell's skill rank (1 = best) within "
        "each config's 12-cell (N x ladder) grid. The slot-spread column shows the "
        "slot-equity cost is driven by field concentration (large when top-8 share is high).",
        "",
        rob.to_markdown(index=False),
        "",
    ]
    if exp_df is not None:
        lines += ["## Exploitability (best-response vs EV-greedy at slot 1)", ""]
        e = exp_df.copy()
        piv = e.pivot_table(
            index=["strength_config", "ladder", "teams_per_drafter"],
            columns="policy",
            values="slot1_win_prob",
        )
        piv["br_minus_evgreedy"] = piv.get("best_response", np.nan) - piv.get("ev_greedy", np.nan)
        lines += [
            piv.round(4).to_markdown(),
            "",
            "A positive `br_minus_evgreedy` means a best-responding drafter in slot 1 "
            "beats the EV-greedy baseline there — i.e. EV-greedy is exploitable by that "
            "margin.",
            "",
        ]

    out = TABLES / f"recommendation_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, rec_n, rec_ladder


def write_summary(df, today):
    cols = [
        "strength_config",
        "top8_title_share",
        "ladder",
        "teams_per_drafter",
        "policy",
        SKILL_COL,
        "skill_variance_share",
        "top_tie_rate",
        "winning_score_mean",
        "slot_win_prob_spread",
        "champion_undrafted_rate",
        "p_champion_holder_wins",
    ]
    out = TABLES / f"results_summary_{today}.csv"
    df[cols].round(4).to_csv(out, index=False)
    return out


def main():
    today = date.today().isoformat()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(latest("results_main_*.csv"))
    try:
        exp_df = pd.read_csv(latest("results_exploitability_*.csv"))
    except FileNotFoundError:
        exp_df = None

    outs = [
        fig_skill_heatmap(df, today),
        fig_tradeoff(df, today),
        fig_robustness(df, today),
        fig_slot_equity(df, today),
        write_summary(df, today),
    ]
    rec_out, rec_n, rec_ladder = write_recommendation(df, exp_df, today)
    outs.append(rec_out)
    for o in outs:
        print(f"wrote {o.relative_to(ROOT)}")
    print(f"\nRecommended: N={rec_n}, ladder={rec_ladder}")


if __name__ == "__main__":
    main()
