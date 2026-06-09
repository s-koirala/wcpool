"""Render figures and the (N, ladder) recommendation from the experiment tables.

Reads the latest results_main_*.csv (+ results_exploitability_*.csv) from docs/tables/
and writes:
  docs/figures/skill_heatmap_{date}.png        skill correlation over the (N, ladder) grid
  docs/figures/tradeoff_{date}.png             skill vs tie-rate vs slot-equity tradeoff
  docs/figures/robustness_{date}.png           metrics vs favoritism concentration
  docs/figures/slot_equity_{date}.png          win prob by snake slot
  docs/tables/results_summary_{date}.csv       headline metrics, EV-greedy, per config
  docs/tables/recommendation_{date}.md         ladder recommendation + robustness

The recommendation is made at the LADDER level (the statistically discriminating axis):
each ladder is scored on three objectives (skill up, tie-rate down, slot-spread down) and
the balanced default is the ladder minimising its worst per-objective rank (minimax, no
arbitrary cut-offs). N is reported as second-order — its within-ladder skill differences
are below the between-draw cluster SE, so it is not ranked.
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
    # Ignore "_quick" smoke-run files so plots always reflect the full 50k deliverable.
    files = sorted(f for f in TABLES.glob(glob) if "_quick" not in f.name)
    if not files:
        raise FileNotFoundError(
            f"no full-run file matching {glob} in {TABLES} "
            f"(run scripts/run_experiment.py without --quick first)"
        )
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
    # minimax over per-objective ranks; ties broken by total rank, then higher skill.
    order = pd.DataFrame({"w": worst, "t": total, "neg_skill": -lad.skill})
    balanced_ladder = order.sort_values(["w", "t", "neg_skill"]).index[0]
    skill_ladder = lad.skill.idxmax()
    fair_ladder = lad.slot.idxmin()

    # Significance framing from the between-draw cluster SE (the draw is the unit). Under
    # the fixed-draw regime there is a single draw, so the cluster SE is undefined (NaN).
    typ_se = float(anchor[SE_COL].median())
    bl = anchor[anchor.ladder == balanced_ladder]
    n_range = float(bl[SKILL_COL].max() - bl[SKILL_COL].min())
    cross_gap = float(lad.skill.max() - lad.skill.min())
    se_known = np.isfinite(typ_se) and typ_se > 0
    if se_known:
        sig_text = (
            f"Within the {balanced_ladder} ladder, skill ranges only {n_range:.3f} across "
            f"N in {{4,5,6,8}} = {n_range / typ_se:.1f}x the per-cell between-draw cluster "
            f"SE ({typ_se:.4f}); the geometric-vs-linear skill gap is {cross_gap:.3f} = "
            f"{cross_gap / typ_se:.0f}x that SE."
        )
    else:
        sig_text = (
            f"Within the {balanced_ladder} ladder, skill ranges only {n_range:.3f} across "
            f"N in {{4,5,6,8}} vs a cross-ladder gap of {cross_gap:.3f}. (Between-draw "
            "cluster SE is undefined under the single-draw fixed regime; run the resampled "
            "regime for the significance multiples.)"
        )

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
        "**N is second-order and statistically indistinguishable.** " + sig_text + " "
        "Choose the ladder first; set N on practical grounds (N=8 drafts the full 48-team "
        "field with no stars left undrafted; smaller N leaves more teams unowned).",
        "",
        "Note: the between-draw cluster SE is the correct precision here — the 50k per-cell "
        "tournaments are clustered within the draws, so the independent unit is the draw, "
        "not the sim. A naive iid SE would understate uncertainty roughly fourfold.",
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


# --- Group-scoring (gamma) figures (plan section 12) ------------------------------------
# These read the group-scoring frontier (results_groupscoring_*.csv on the elo_2026 field) and the
# robustness slice. The skill axis is the SEED-STABLE paired delta_skill_vs_anchor (not the absolute
# level); the engagement axis is the estimator-free group_variance_share. The recommended cell
# (triangular, 3:1, gamma_match) is annotated. Style mirrors the prior-study figures.
GS_SHAPE_MARKERS = {"linear": "o", "triangular": "s", "geometric": "^"}
GS_SHAPE_COLORS = {"linear": "#1f77b4", "triangular": "#2ca02c", "geometric": "#d62728"}
GS_DW_LABEL = {
    (1.0, 0.0): "1:0 (wins-only)",
    (2.0, 1.0): "2:1 (constant-sum)",
    (3.0, 1.0): "3:1 (FIFA)",
}


def _gs_grid_only(head):
    """The swept gamma grid (anchor + interior levels), excluding the gamma_match landmark."""
    return head[~head.is_gamma_match].sort_values("gamma_realised")


def _gs_shape_31(head, shape):
    """A shape's gamma=0 anchor + its 3:1 D/W cells, gamma-grid only (per-shape 3:1 trajectory)."""
    mask = (head.knockout_ladder == shape) & (
        (head.mix == 0) | ((head.w_pts == 3) & (head.d_pts == 1))
    )
    return _gs_grid_only(head[mask])


def fig_gs_paired_skill_frontier(head, today, rec=None):
    """Paired delta-skill (seed-stable) vs realised gamma, per shape at the familiar 3:1 D/W."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for shape in LADDER_ORDER:
        s = _gs_shape_31(head, shape)
        ax.errorbar(
            s.gamma_realised,
            s.delta_skill_vs_anchor,
            yerr=s.delta_skill_paired_se,
            marker=GS_SHAPE_MARKERS[shape],
            color=GS_SHAPE_COLORS[shape],
            capsize=3,
            label=shape,
        )
    ax.axhline(0.0, ls="--", color="grey", lw=1, label="gamma=0 anchor (no skill cost)")
    if rec is not None:
        ax.scatter(
            [rec.gamma_realised],
            [rec.delta_skill],
            s=240,
            marker="*",
            color="black",
            zorder=5,
            label=f"recommended ({rec.shape} 3:1 gamma_match)",
        )
    ax.set_xlabel("realised group share  gamma  (group stage = this fraction of the points)")
    ax.set_ylabel("paired skill change vs the gamma=0 ladder  (0 = no skill surrendered)")
    ax.set_title("Skill cost of the group layer (seed-stable paired delta) — elo_2026, 3:1 D/W")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"groupscoring_paired_skill_frontier_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gs_engagement_vs_gamma(head, balanced, today, rec=None):
    """group_variance_share (engagement) vs gamma per shape + the balanced-field benchmark band."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for shape in LADDER_ORDER:
        s = _gs_shape_31(head, shape)
        ax.plot(
            s.gamma_realised,
            s.group_variance_share,
            marker=GS_SHAPE_MARKERS[shape],
            color=GS_SHAPE_COLORS[shape],
            label=shape,
        )
        # balanced-field benchmark band: floor (gamma=0, == 0) to ceiling (equal-purse gamma=0.5).
        if balanced is not None:
            b = balanced[balanced.knockout_ladder == shape]
            beq = b[np.isclose(b.gamma_realised, 0.5, atol=1e-6)]
            if len(beq):
                ax.axhspan(
                    0.0,
                    float(beq.group_variance_share.max()),
                    color=GS_SHAPE_COLORS[shape],
                    alpha=0.06,
                )
    if rec is not None:
        ax.scatter(
            [rec.gamma_realised],
            [rec.group_variance_share],
            s=240,
            marker="*",
            color="black",
            zorder=5,
            label="recommended (gamma_match)",
        )
    ax.set_xlabel("realised group share  gamma")
    ax.set_ylabel("engagement: group_variance_share  (0 = dead group stage)")
    ax.set_title(
        "Engagement vs gamma per shape (3:1 D/W); shaded = achievable band on the balanced field"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"groupscoring_engagement_vs_gamma_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gs_skill_engagement_knee(head, today, rec=None):
    """The skill-vs-engagement tradeoff (the knee): skill COST vs engagement, gamma along curves."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for shape in LADDER_ORDER:
        s = _gs_shape_31(head, shape)
        ax.plot(
            s.group_variance_share,
            -s.delta_skill_vs_anchor,  # skill COST (positive)
            marker=GS_SHAPE_MARKERS[shape],
            color=GS_SHAPE_COLORS[shape],
            label=shape,
        )
        for _, row in s.iterrows():
            if row.gamma_realised > 0:
                ax.annotate(
                    f"{row.gamma_realised:.2f}",
                    (row.group_variance_share, -row.delta_skill_vs_anchor),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=7,
                    color=GS_SHAPE_COLORS[shape],
                )
    if rec is not None:
        ax.scatter(
            [rec.group_variance_share],
            [-rec.delta_skill],
            s=240,
            marker="*",
            color="black",
            zorder=5,
            label="recommended (gamma_match)",
        )
    ax.set_xlabel("engagement bought: group_variance_share  -> more in it")
    ax.set_ylabel("skill cost (paired)  -> more skill surrendered")
    ax.set_title("The skill-vs-engagement tradeoff (the knee) — elo_2026, 3:1 D/W (gamma labelled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"groupscoring_skill_engagement_knee_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gs_dw_tie_comparison(head, today):
    """The D/W comparison: tie rate (+ engagement) vs gamma for the three D/W (constant-sum 2:1)."""
    tri = head[head.knockout_ladder == "triangular"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dw, lbl in GS_DW_LABEL.items():
        s = tri[(tri.mix == 0) | ((tri.w_pts == dw[0]) & (tri.d_pts == dw[1]))]
        s = _gs_grid_only(s)
        axes[0].plot(s.gamma_realised, s.top_tie_rate, marker="o", label=lbl)
        axes[1].plot(s.gamma_realised, s.group_variance_share, marker="o", label=lbl)
    axes[0].set_xlabel("realised group share  gamma")
    axes[0].set_ylabel("tie rate at the top  -> lower better")
    axes[0].set_title("Ties by D/W (triangular)")
    axes[1].set_xlabel("realised group share  gamma")
    axes[1].set_ylabel("engagement: group_variance_share")
    axes[1].set_title("Engagement by D/W (triangular)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Win:draw effect — constant-sum 2:1 vs FIFA 3:1 vs wins-only 1:0 (triangular)")
    fig.tight_layout()
    out = FIGURES / f"groupscoring_wd_tradeoff_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_gs_metrics_vs_gamma(head, today):
    """skill / tie / slot / champ-dom vs gamma per shape (3:1 D/W) — the four-metric trajectory."""
    metrics = [
        ("delta_skill_vs_anchor", "paired skill change (0 = no cost)"),
        ("top_tie_rate", "tie rate"),
        ("slot_win_prob_spread", "slot win-prob spread"),
        ("p_champion_holder_wins", "champion-dominance"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (col, label) in zip(axes, metrics, strict=True):
        for shape in LADDER_ORDER:
            s = _gs_shape_31(head, shape)
            ax.plot(
                s.gamma_realised,
                s[col],
                marker=GS_SHAPE_MARKERS[shape],
                color=GS_SHAPE_COLORS[shape],
                label=shape,
            )
        ax.set_xlabel("gamma")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Metrics vs group share gamma per shape (3:1 D/W) — elo_2026")
    fig.tight_layout()
    out = FIGURES / f"groupscoring_metrics_vs_gamma_{today}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _latest_groupscoring_headline():
    """The latest headline group-scoring frontier CSV, excluding the robustness/exploitability/quick
    siblings (whose names share the ``results_groupscoring_`` prefix)."""
    files = sorted(
        f
        for f in TABLES.glob("results_groupscoring_*.csv")
        if "_quick" not in f.name
        and "robustness" not in f.name
        and "exploitability" not in f.name
    )
    if not files:
        raise FileNotFoundError(
            f"no headline results_groupscoring_*.csv in {TABLES} "
            f"(run scripts/run_experiment.py --group-scoring first)"
        )
    return files[-1]


def make_group_plots(today):
    """Render the five group-scoring figures (plan section 12) from the committed frontier CSVs."""
    from wcpool import select  # local import: only the group-scoring path needs the selector

    head_all = pd.read_csv(_latest_groupscoring_headline())
    head = head_all[head_all.strength_config == ANCHOR_CONFIG].copy()
    try:
        rob = pd.read_csv(latest("results_groupscoring_robustness_*.csv"))
        balanced = rob[rob.strength_config == "synthetic_x0.5"]
    except FileNotFoundError:
        balanced = None
    sel = select.select_recommendation(head, engagement_level=1e-4, prefer_gamma_match=True)
    rec = sel.recommended
    outs = [
        fig_gs_paired_skill_frontier(head, today, rec),
        fig_gs_engagement_vs_gamma(head, balanced, today, rec),
        fig_gs_skill_engagement_knee(head, today, rec),
        fig_gs_dw_tie_comparison(head, today),
        fig_gs_metrics_vs_gamma(head, today),
    ]
    return outs


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group-scoring",
        action="store_true",
        help="render the group-stage (gamma) figures from results_groupscoring_*.csv",
    )
    args = ap.parse_args()
    today = date.today().isoformat()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.group_scoring:
        outs = make_group_plots(today)
        for o in outs:
            print(f"wrote {o.relative_to(ROOT)}")
        return

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
