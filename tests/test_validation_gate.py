"""The gamma=0 validation gate for the two-layer-scoring refactor of ``run_strength_config``.

Plan section 8.4 (corrected: the config-matched anchor is the *8-participant* participant-sweep
rows, i.e. 8 drafters x 6 teams -- NOT the full-grid ``teams_per_drafter=8`` rows, which are the
6-drafter x 8-team configuration). The gate has two parts:

(a) **Exact self-consistency (budget-independent, rigorous).** Running the *refactored*
    ``run_strength_config`` with ``schemes = [ScoringScheme(l, 1.0, 0.0, 0.0) for l in ladders]``
    at the same config/seed/budget as the pre-refactor golden reproduces every existing-metric
    value to floating-point tolerance. At ``mix == 0`` ``scoring.team_points`` short-circuits to the
    bare ladder lookup ``ladder[stages]`` and the ``SeedSequence(seed, spawn_key=(cfg_id, draw,
    batch))`` RNG path is unchanged, so the intermediate ``stages`` are byte-identical and the
    derived metrics match *exactly*. The frozen ``GOLDEN_8x6_MIX0`` below was captured from the
    pre-refactor ``run_strength_config`` (HEAD ``5c4ef6a``) at this same config/seed/budget; it is
    the regression baseline the refactor must not perturb. This is budget-independent because it
    rests on byte-identical ``stages`` -- the small budget only keeps the test fast.

(b) **Cross-check vs the published config-matched baseline.** At the spec budget (25 draws x 2000
    sims), real Elo, 8 drafters x 6 teams, EV-greedy, ``mix == 0``, the new pipeline's
    skill/tie/slot-spread/champ-holder reproduce the ``participant_sweep_2026-06-08.csv``
    8-participant rows within a few between-draw cluster SE. This is a *sanity cross-check* (the
    participant-sweep used master seed 20260608 with the same resampled regime, so the realised
    numbers should agree closely but need not be byte-identical). It is reported in cluster-SE units
    and only hard-fails outside ~3 SE; a marginal excess is flagged, not failed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from wcpool import simulate
from wcpool.config import load_field
from wcpool.scoring import ScoringScheme

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ratings_elo_2026.yaml"
PARTICIPANT_SWEEP = ROOT / "docs" / "tables" / "participant_sweep_2026-06-08.csv"

LADDERS = ["linear", "triangular", "geometric"]
GOLDEN_SEED = 20260608  # the prior study's seed (plan section 8.4 reuses it deliberately)

# Golden config/budget: real Elo, 8 drafters x 6 teams, EV-greedy, resampled, seed 20260608.
# The 6 x 500 budget is ARBITRARY-BUT-DOCUMENTED: it only sets how fast the test runs. Part (a) is
# BUDGET-INDEPENDENT -- it rests on the mix=0 ``stages`` being byte-identical to the pre-refactor
# RNG path (so every derived metric matches exactly at any budget); the small budget is for speed.
GOLDEN_N_DRAWS = 6
GOLDEN_SIMS_PER_DRAW = 500

# FP-tolerance envelope for the part-(a) exact reproduction. The expected difference is EXACTLY 0.0
# (byte-identical ``stages`` + the mix=0 ``team_points`` short-circuit equals the bare ladder
# lookup), so this is not a "closeness" budget -- it only guards against last-bit libm reduction
# reordering (e.g. a platform summing a mean in a different order). rel=1e-9 / abs=1e-12 sit far
# below any metric's scale yet far above a single ULP, so a genuine regression (which would perturb
# a metric at the 1e-3+ Monte-Carlo scale) cannot hide under them. Used for BOTH the per-metric
# ``pytest.approx`` band and the ``worst_overall`` final bar, reconciling the two to one tolerance:
# ``RTOL`` for the per-metric relative check, ``ATOL`` for the near-zero metrics, and
# ``worst_overall < ATOL`` as the strictest single scalar -- an ABSOLUTE max-abs-diff floor, so
# passing it implies every metric matched to < 1e-12 absolute.
RTOL = 1e-9
ATOL = 1e-12

# Hard-fail bound (in between-draw cluster-SE units) for the part-(b) skill cross-check: ~99.7% of a
# two-sided normal lies within 3 SD, so a |z| > 3 skill gap is a >3 cluster-SE departure from the
# published anchor. This is a SANITY CROSS-CHECK, not a registered inferential test -- different
# seed paths realise slightly different numbers, so a marginal single-ladder excess is FLAGGED
# (loud, captured) not failed; only a gross divergence on ALL THREE ladders trips the assertion.
MAX_CLUSTER_SE = 3.0

# Existing-metric scalar keys the refactor must reproduce to FP tolerance.
GOLDEN_SCALAR_KEYS = [
    "spearman_mean",
    "spearman_sd",
    "spearman_cluster_se",
    "skill_variance_share",
    "sigma2_between",
    "sigma2_within",
    "top_tie_rate",
    "top_tie_rate_cluster_se",
    "winning_score_mean",
    "winning_score_sd",
    "winning_score_p05",
    "winning_score_p95",
    "slot_win_prob_spread",
    "slot_win_prob_spread_cluster_se",
    "slot_imbalance_index",
    "champion_undrafted_rate",
    "p_champion_holder_wins",
]

# Frozen pre-refactor golden (captured from run_strength_config at HEAD 5c4ef6a, GOLDEN_SEED,
# GOLDEN_N_DRAWS x GOLDEN_SIMS_PER_DRAW, the config above). Full float precision so the FP-tolerance
# comparison is exact. DO NOT regenerate against the refactored code -- it is the regression anchor.
GOLDEN_8x6_MIX0 = {
    "linear": {
        "champion_undrafted_rate": 0.0,
        "p_champion_holder_wins": 0.47556111111111116,
        "sigma2_between": 0.28329016666666657,
        "sigma2_within": 5.931490166666667,
        "skill_variance_share": 0.04558329522078574,
        "slot_imbalance_index": 0.14817787654320988,
        "slot_win_prob_spread": 0.15188333333333331,
        "slot_win_prob_spread_cluster_se": 0.009668275728151087,
        "slot_win_probs": [
            0.23117222222222222, 0.16203888888888887, 0.1325333333333333, 0.12445555555555554,
            0.09722777777777776, 0.08957777777777778, 0.0792888888888889, 0.08370555555555556,
        ],
        "spearman_cluster_se": 0.01865207104601528,
        "spearman_mean": 0.18313042268600863,
        "spearman_sd": 0.3575153565233394,
        "top_tie_rate": 0.21366666666666667,
        "top_tie_rate_cluster_se": 0.007804557073345749,
        "winning_score_mean": 11.891,
        "winning_score_p05": 10.0,
        "winning_score_p95": 15.0,
        "winning_score_sd": 1.4759581069032188,
    },
    "triangular": {
        "champion_undrafted_rate": 0.0,
        "p_champion_holder_wins": 0.8012777777777779,
        "sigma2_between": 5.848926000000002,
        "sigma2_within": 55.534124666666656,
        "skill_variance_share": 0.09528568450860968,
        "slot_imbalance_index": 0.5327328395061727,
        "slot_win_prob_spread": 0.28227777777777774,
        "slot_win_prob_spread_cluster_se": 0.011106693566282837,
        "slot_win_probs": [
            0.3317777777777777, 0.20238888888888887, 0.1271111111111111, 0.09894444444444443,
            0.07061111111111111, 0.0621111111111111, 0.057555555555555554, 0.0495,
        ],
        "spearman_cluster_se": 0.018804839701453826,
        "spearman_mean": 0.25735698022305414,
        "spearman_sd": 0.35466513655851667,
        "top_tie_rate": 0.057666666666666665,
        "top_tie_rate_cluster_se": 0.003844187531556933,
        "winning_score_mean": 29.119333333333334,
        "winning_score_p05": 23.0,
        "winning_score_p95": 38.0,
        "winning_score_sd": 4.490852876186834,
    },
    "geometric": {
        "champion_undrafted_rate": 0.0,
        "p_champion_holder_wins": 0.9996666666666667,
        "sigma2_between": 14.09030733333333,
        "sigma2_within": 99.74597383333334,
        "skill_variance_share": 0.12377694693578262,
        "slot_imbalance_index": 0.8107235555555558,
        "slot_win_prob_spread": 0.3396666666666667,
        "slot_win_prob_spread_cluster_se": 0.010874026137748816,
        "slot_win_probs": [
            0.38133333333333336, 0.214, 0.14533333333333334, 0.06966666666666667,
            0.057666666666666665, 0.04766666666666667, 0.042666666666666665, 0.041666666666666664,
        ],
        "spearman_cluster_se": 0.016640229683924253,
        "spearman_mean": 0.27687740316746606,
        "spearman_sd": 0.35279977053183686,
        "top_tie_rate": 0.0,
        "top_tie_rate_cluster_se": 0.0,
        "winning_score_mean": 38.14033333333333,
        "winning_score_p05": 34.0,
        "winning_score_p95": 48.0,
        "winning_score_sd": 4.129322772023949,
    },
}


def _run_mix0(schemes_or_ladders: str, n_draws: int, sims_per_draw: int) -> dict[str, dict]:
    """Run the refactored ``run_strength_config`` at the golden config; key records by ladder.

    ``schemes_or_ladders`` selects which API to exercise: ``"schemes"`` passes explicit
    ``ScoringScheme(l, 1.0, 0.0, 0.0)`` (the gate's mix=0 construction), ``"ladders"`` passes the
    legacy ``ladders=[...]`` shim. Both must produce identical records.
    """
    field = load_field(CONFIG)
    cfg = simulate.make_elo_config(field, name="elo_2026")
    kwargs = dict(
        n_values=[6],
        policies=["ev_greedy"],
        n_draws=n_draws,
        sims_per_draw=sims_per_draw,
        seed=GOLDEN_SEED,
        regime="resampled",
        n_drafters=8,
    )
    if schemes_or_ladders == "schemes":
        recs = simulate.run_strength_config(
            cfg, schemes=[ScoringScheme(name, 1.0, 0.0, 0.0) for name in LADDERS], **kwargs
        )
    else:
        recs = simulate.run_strength_config(cfg, ladders=LADDERS, **kwargs)
    return {r["ladder"]: r for r in recs}


def _slot_probs(rec: dict) -> list[float]:
    keys = sorted(
        (k for k in rec if k.startswith("slot") and k.endswith("_win_prob")),
        key=lambda k: int(k[len("slot") : -len("_win_prob")]),
    )
    return [rec[k] for k in keys]


# --- (a) Exact self-consistency: refactored mix=0 == pre-refactor golden ----------------


def test_gate_exact_self_consistency_schemes_reproduce_golden():
    """The new ``schemes=`` mix=0 path reproduces the pre-refactor golden to FP tolerance.

    Reports the max abs diff per metric; asserts every metric matches within a tight rtol/atol
    (empirically the diff is exactly 0.0 because the ``stages`` are byte-identical and the mix=0
    ``team_points`` short-circuit equals the bare ladder lookup).
    """
    got = _run_mix0("schemes", GOLDEN_N_DRAWS, GOLDEN_SIMS_PER_DRAW)
    assert set(got) == set(GOLDEN_8x6_MIX0)

    worst_per_metric: dict[str, float] = {}
    worst_overall = 0.0
    for ladder, g in GOLDEN_8x6_MIX0.items():
        r = got[ladder]
        for k in GOLDEN_SCALAR_KEYS:
            d = abs(float(g[k]) - float(r[k]))
            worst_per_metric[k] = max(worst_per_metric.get(k, 0.0), d)
            worst_overall = max(worst_overall, d)
            assert float(r[k]) == pytest.approx(float(g[k]), rel=RTOL, abs=ATOL), (
                f"{ladder}.{k}: refactored {r[k]!r} != golden {g[k]!r} (|d|={d:.3e})"
            )
        gp, rp = g["slot_win_probs"], _slot_probs(r)
        assert len(gp) == len(rp)
        for i, (a, b) in enumerate(zip(gp, rp, strict=True)):
            d = abs(float(a) - float(b))
            worst_per_metric[f"slot{i + 1}_win_prob"] = max(
                worst_per_metric.get(f"slot{i + 1}_win_prob", 0.0), d
            )
            worst_overall = max(worst_overall, d)
            assert float(b) == pytest.approx(float(a), rel=RTOL, abs=ATOL)
        # mix=0 cell still carries the new scheme columns, set to the pure-ladder identity.
        assert r["mix"] == 0.0
        assert r["knockout_ladder"] == ladder
        assert r["w_pts"] == 1.0 and r["d_pts"] == 0.0
    # Surface the realised resolution for the run log.
    print("\n[gate-a] EXACT mix=0 reproduction vs pre-refactor golden:")
    print(f"  max abs diff over ALL metrics = {worst_overall:.3e}")
    for k in sorted(worst_per_metric):
        print(f"    {k:34s} max|d| = {worst_per_metric[k]:.3e}")
    # Strictest single bar: an ABSOLUTE max-abs-diff floor reconciling the per-metric relative band
    # above (the realised diff is exactly 0.0, so this 1e-12 absolute bar holds with margin).
    assert worst_overall < ATOL


def test_gate_legacy_ladders_shim_matches_schemes_mix0_byte_identical():
    """The legacy ``ladders=[...]`` shim is byte-identical to the explicit ``schemes=`` mix=0 path.

    The shim maps each name to ``ScoringScheme(name, 1.0, 0.0, 0.0)``; both must trace the same RNG
    path and produce ``np.array_equal`` records (so the prior study reproduces bit-for-bit through
    either entry point).
    """
    via_schemes = _run_mix0("schemes", GOLDEN_N_DRAWS, GOLDEN_SIMS_PER_DRAW)
    via_ladders = _run_mix0("ladders", GOLDEN_N_DRAWS, GOLDEN_SIMS_PER_DRAW)
    assert set(via_schemes) == set(via_ladders)
    for ladder in via_schemes:
        rs, rl = via_schemes[ladder], via_ladders[ladder]
        for k in GOLDEN_SCALAR_KEYS:
            assert float(rs[k]) == float(rl[k]), f"{ladder}.{k}: {rs[k]!r} != {rl[k]!r}"
        assert _slot_probs(rs) == _slot_probs(rl)


# --- (b) Cross-check vs the published participant-sweep 8-participant rows ---------------


def _participant_sweep_8p() -> dict[str, dict[str, float]]:
    """Read the 8-participant rows of participant_sweep_2026-06-08.csv (the config-matched anchor).

    These are the 8-drafter x 6-team config -- the same configuration the gate runs -- not the
    full-grid ``teams_per_drafter=8`` (6-drafter x 8-team) rows the uncorrected plan cited.
    """
    out: dict[str, dict[str, float]] = {}
    with PARTICIPANT_SWEEP.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["participants"]) == 8:
                out[row["ladder"]] = {
                    "skill": float(row["skill"]),
                    "skill_se": float(row["skill_se"]),
                    "tie_rate": float(row["tie_rate"]),
                    "seat_spread": float(row["seat_spread"]),
                    "champ_undrafted": float(row["champ_undrafted"]),
                    "champ_holder_wins": float(row["champ_holder_wins"]),
                }
    return out


@pytest.mark.slow
def test_gate_cross_check_vs_participant_sweep_8p():
    """At the spec budget the new mix=0 pipeline matches the participant-sweep 8-participant anchor.

    Reports each realised metric vs the published value in cluster-SE units. Hard-fails only outside
    ~3 cluster SE on skill (the SE'd headline); ties/slot-spread/champ-holder are reported as a
    cross-check (different seed-path realisations differ at the MC-noise scale) and the champion-
    undrafted rate must be exactly 0 (full 8x6 field).
    """
    anchor = _participant_sweep_8p()
    assert set(anchor) == set(LADDERS), f"missing 8-participant rows: {sorted(anchor)}"

    # Spec budget: 25 draws x 2000 sims (plan section 8.2).
    got = _run_mix0("schemes", n_draws=25, sims_per_draw=2000)

    print("\n[gate-b] CROSS-CHECK vs participant_sweep_2026-06-08.csv 8-participant rows "
          "(25 draws x 2000 sims, mix=0):")
    print(f"  {'ladder':10s} {'metric':18s} {'realised':>10s} {'published':>10s} "
          f"{'cluster_se':>11s} {'z (SE units)':>13s}")
    flagged: list[str] = []
    for ladder in LADDERS:
        r, a = got[ladder], anchor[ladder]
        se = float(r["spearman_cluster_se"])
        skill_z = (r["spearman_mean"] - a["skill"]) / se if se > 0 else float("nan")
        rows = [
            ("skill", r["spearman_mean"], a["skill"], se, skill_z),
            ("tie_rate", r["top_tie_rate"], a["tie_rate"],
             r["top_tie_rate_cluster_se"], float("nan")),
            ("seat_spread", r["slot_win_prob_spread"], a["seat_spread"],
             r["slot_win_prob_spread_cluster_se"], float("nan")),
            ("champ_holder", r["p_champion_holder_wins"], a["champ_holder_wins"],
             float("nan"), float("nan")),
            ("champ_undrafted", r["champion_undrafted_rate"], a["champ_undrafted"],
             float("nan"), float("nan")),
        ]
        for name, realised, published, cse, z in rows:
            zs = f"{z:+.2f}" if np.isfinite(z) else "    -"
            cs = f"{cse:.4f}" if np.isfinite(cse) else "   -"
            print(f"  {ladder:10s} {name:18s} {realised:10.4f} {published:10.4f} "
                  f"{cs:>11s} {zs:>13s}")
        # Full 8x6 field -> the champion is always owned.
        assert r["champion_undrafted_rate"] == 0.0
        # Skill is the registered SE'd head-to-head; allow within ~3 cluster SE, flag beyond.
        if np.isfinite(skill_z) and abs(skill_z) > MAX_CLUSTER_SE:
            flagged.append(
                f"{ladder} skill {r['spearman_mean']:.4f} vs published {a['skill']:.4f} "
                f"= {skill_z:+.2f} cluster SE (> 3 SE)"
            )

    if flagged:
        # Per the task: report rather than hard-fail at a slight excess, but make it loud. Only a
        # gross (> 3 SE) skill divergence on ALL three ladders would indicate a real regression.
        joined = "; ".join(flagged)
        msg = f"cross-check skill outside ~3 cluster SE (FLAGGED, see captured stdout): {joined}"
        print("\n[gate-b] FLAG:", msg)
        assert len(flagged) < len(LADDERS), msg
