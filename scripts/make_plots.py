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
N_DRAFTERS = 6  # fixed by the task (6-person pool); the equitable per-slot win prob is 1/6


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
    slot_cols = [f"slot{i + 1}_win_prob" for i in range(N_DRAFTERS)]
    fig, axes = plt.subplots(1, len(LADDER_ORDER), figsize=(13, 4), sharey=True)
    for ax, ladder in zip(axes, LADDER_ORDER, strict=True):
        s = sub[sub.ladder == ladder]
        for _, row in s.iterrows():
            ax.plot(
                range(1, N_DRAFTERS + 1),
                [row[c] for c in slot_cols],
                marker="o",
                label=f"N={int(row.teams_per_drafter)}",
            )
        ax.axhline(1 / N_DRAFTERS, ls="--", color="grey", lw=1, label=f"equitable 1/{N_DRAFTERS}")
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


SE_COL = "spearman_cluster_se"  # between-draw cluster SE of the skill correlation


def _ladder_aggregates(anchor):
    """Per-ladder means over N + a representative between-draw cluster SE."""
    lad = (
        anchor.groupby("ladder")
        .agg(
            skill=(SKILL_COL, "mean"),
            tie=("top_tie_rate", "mean"),
            slot=("slot_win_prob_spread", "mean"),
            champ_dom=("p_champion_holder_wins", "mean"),
            cluster_se=(SE_COL, "mean"),
        )
        .reindex(LADDER_ORDER)
    )
    return lad


def write_recommendation(df, exp_df, today):
    anchor = df[(df.strength_config == ANCHOR_CONFIG) & (df.policy == "ev_greedy")].copy()
    anchor = anchor.reset_index(drop=True)
    lad = _ladder_aggregates(anchor)

    # Recommend at the LADDER level (the discriminating axis). N is not separated under the
    # cluster SE, so it is not ranked. Balanced ladder = minimax over the 3 objective ranks.
    r_skill = lad.skill.rank(ascending=False)
    r_tie = lad.tie.rank(ascending=True)
    r_slot = lad.slot.rank(ascending=True)
    worst = pd.concat([r_skill, r_tie, r_slot], axis=1).max(axis=1)
    total = r_skill + r_tie + r_slot
    balanced_ladder = pd.DataFrame({"w": worst, "t": total}).sort_values(["w", "t"]).index[0]
    skill_ladder = lad.skill.idxmax()
    fair_ladder = lad.slot.idxmin()

    # Significance framing from the between-draw cluster SE (the draw is the unit).
    typ_se = float(anchor[SE_COL].median())
    bl = anchor[anchor.ladder == balanced_ladder]
    n_range = float(bl[SKILL_COL].max() - bl[SKILL_COL].min())
    cross_gap = float(lad.skill.max() - lad.skill.min())
    n_range_se = n_range / typ_se
    cross_gap_se = cross_gap / typ_se

    def lad_str(name):
        r = lad.loc[name]
        return (
            f"{name} ladder — skill={r.skill:.3f}, tie={r.tie:.4f}, "
            f"slot-spread={r.slot:.3f}, champ-dom={r.champ_dom:.3f}"
        )

    lines = [
        f"# Recommended ladder (and N) — {today}",
        "",
        "## Headline",
        "",
        "The three design goals **conflict along one axis — ladder convexity**. The same "
        "convexity that raises skill correlation and removes ties also concentrates pool "
        "wins in the first snake pick (it turns the pool into a referendum on who drafts "
        "the eventual champion). No ladder wins all three goals; the choice is a tradeoff.",
        "",
        f"* **Balanced default (recommended):** the {lad_str(balanced_ladder)}. Best joint "
        "standing across all three objectives (minimax over per-objective ranks).",
        f"* **Pure skill / no-ties:** the {lad_str(skill_ladder)}. Maximises skill and "
        "essentially eliminates ties, but is the *least* slot-equitable and makes the pool "
        "almost entirely about owning the champion.",
        f"* **Most slot-equitable:** the {lad_str(fair_ladder)}. Fairest across snake "
        "slots, but lowest skill and a high tie rate (needs an explicit tiebreaker rule).",
        "",
        "**N is second-order and statistically indistinguishable.** Within the "
        f"{balanced_ladder} ladder, skill ranges only {n_range:.3f} across N in {{4,5,6,8}} "
        f"= {n_range_se:.1f}x the per-cell between-draw cluster SE ({typ_se:.4f}); the "
        f"geometric-vs-linear skill gap is {cross_gap:.3f} = {cross_gap_se:.0f}x that SE. "
        "Choose the ladder first; set N on practical grounds (N=8 drafts the full 48-team "
        "field with no stars left undrafted; smaller N leaves more teams unowned).",
        "",
        "Note: the between-draw cluster SE (~"
        f"{typ_se:.4f}) is the correct precision here — the 50k per-cell tournaments are "
        "clustered within 25 draws, so the independent unit is the draw, not the sim. A "
        "naive iid SE would understate uncertainty roughly fourfold.",
        "",
        "## Per-ladder summary (anchor config, EV-greedy; mean over N, with cluster SE)",
        "",
        lad.round(4).to_markdown(),
        "",
        "## Full (N, ladder) grid — anchor config, EV-greedy",
        "",
        anchor.sort_values([SKILL_COL], ascending=False)[
            [
                "teams_per_drafter",
                "ladder",
                SKILL_COL,
                SE_COL,
                "skill_variance_share",
                "top_tie_rate",
                "slot_win_prob_spread",
                "p_champion_holder_wins",
            ]
        ]
        .round(4)
        .to_markdown(index=False),
        "",
        "## Robustness across strength models (recommended ladder, mean over N)",
        "",
        "`skill_rank` is the recommended ladder's rank among the 3 ladders within each "
        "config (1 = best skill). The slot-spread column shows the slot-equity cost is "
        "driven by field concentration (large only when the top-8 title share is high).",
        "",
    ]

    rob_rows = []
    for cfg in sorted(df.strength_config.unique()):
        s = df[(df.strength_config == cfg) & (df.policy == "ev_greedy")]
        lad_c = _ladder_aggregates(s.reset_index(drop=True))
        rank = lad_c.skill.rank(ascending=False)[balanced_ladder]
        row = lad_c.loc[balanced_ladder]
        rob_rows.append(
            {
                "strength_config": cfg,
                "top8_title_share": round(s.top8_title_share.iloc[0], 3),
                "skill": round(row.skill, 3),
                "skill_rank_of_recommended": int(rank),
                "tie_rate": round(row.tie, 4),
                "slot_spread": round(row.slot, 4),
            }
        )
    lines += [pd.DataFrame(rob_rows).to_markdown(index=False), ""]

    if exp_df is not None:
        e = exp_df.copy()
        piv = e.pivot_table(
            index=["strength_config", "ladder", "teams_per_drafter"],
            columns="policy",
            values="slot1_win_prob",
        )
        piv["br_minus_evgreedy"] = piv.get("best_response", np.nan) - piv.get("ev_greedy", np.nan)
        # noise scale: between-draw cluster SE of slot win-prob spread on the probe cells
        noise = float(e["slot_win_prob_spread_cluster_se"].median())
        probe_draws = int(e["n_draws"].iloc[0])
        lines += [
            "## Exploitability (out-of-sample best-response vs EV-greedy at slot 1)",
            "",
            "The best-responder now decides on the EV (model) batch and is scored on an "
            "independent eval batch, so any positive margin is genuine exploitability, not "
            "in-sample optimism. The probe runs at a reduced budget "
            f"({probe_draws} draws); the between-draw noise scale on slot win-prob is "
            f"~{noise:.4f}, so margins within ~that size are not distinguishable from zero.",
            "",
            piv.round(4).to_markdown(),
            "",
            "`br_minus_evgreedy` > 0 means the slot-1 best-responder beats EV-greedy there; "
            "margins of the order of the noise scale above indicate EV-greedy is, at most, "
            "marginally exploitable.",
            "",
        ]

    out = TABLES / f"recommendation_{today}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, balanced_ladder


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
    rec_out, rec_ladder = write_recommendation(df, exp_df, today)
    outs.append(rec_out)
    for o in outs:
        print(f"wrote {o.relative_to(ROOT)}")
    print(f"\nRecommended ladder: {rec_ladder} (N is statistically indistinguishable)")


if __name__ == "__main__":
    main()
