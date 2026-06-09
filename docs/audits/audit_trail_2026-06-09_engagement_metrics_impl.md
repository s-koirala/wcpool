# Audit trail — engagement metrics + Wcond estimator implementation (Milestone 2, 2026-06-09)

## Deliverable
Milestone 2 of [docs/plan_groupstage_scoring_2026-06-09.md](../plan_groupstage_scoring_2026-06-09.md):
the engagement metric suite and the conditional-win-probability (`Wcond`) estimator.

- [src/wcpool/metrics.py](../../src/wcpool/metrics.py) — `terminal_win_credit`, `stage_variance_share`,
  `alive_fraction` (Priority 1); `knn_k_grid`, `_knn_loo_predict`, `conditional_win_prob`,
  `_belief_path_with_prior`, `_efk_increment_summary`, `pool_surprise`, `pool_suspense` (Priority 2).
- [src/wcpool/tournament.py](../../src/wcpool/tournament.py) — validation-only `simulate_tournament_trace`,
  `replay_knockout_from_round` (Priority 3 nested-sim ground truth). Hot path
  (`simulate_tournament`/`simulate_group_stage`) byte-identical; additions are purely appended.
- [tests/test_engagement_metrics.py](../../tests/test_engagement_metrics.py) — 17 tests across the three
  priorities plus the nested-sim gate.

## Loop configuration
- Pattern: `audit-remediate-loop` (intended parallel-specialist-ensemble; cap 3 rounds).
- **Executed as a single-reviewer self-audit:** this execution thread exposes no `Task`/`Agent`
  subagent-spawn tool (the deferred toolset is scheduling/worktree only), so the parallel
  `code-reviewer` / `quant-auditor` / `format-auditor` branches could not be dispatched. The audit
  concerns they cover were instead worked through directly against source + numerical probes. Flagged
  honestly rather than emitting a fabricated multi-agent record. Converged in 2 substantive rounds.

## Findings & disposition

| ID | Severity | Location | Finding | Disposition |
|----|----------|----------|---------|-------------|
| K1 | **major** | metrics.py `_knn_loo_predict` | k-NN leave-one-out dropped column 0 as "self", but `cKDTree.query` does **not** pin the self-match to column 0 under exact duplicate states. At the GROUP boundary 3000 replicates collapse to 1234 unique states (verified), so this leaked a replicate's own terminal credit into its own `Wcond` (LOO violated; biased toward the realised credit). | **Fixed R1**: exclude self *by identity* (stable-sort self to the end, keep first `k`); handles >k+1 duplicate clusters. Regression test `test_knn_loo_excludes_self_under_exact_duplicate_states` added. |
| K2 | minor | metrics.py `stage_variance_share` docstring | "martingale-style increments" overstated — `G`/`K` are raw cumulative-score increments, not the conditional-mean martingale. | Fixed R1 (reworded to "cumulative-score increments"). |
| K3 | minor | metrics.py `pool_surprise`/`pool_suspense` docstrings | "replicate means coincide by the martingale identity" conflated two distinct facts. | Fixed R1: clarified the two are the *same statistic* on the realised-path surrogate (hence exactly equal, not merely in expectation); the martingale identity is what makes the *branched* forward version agree in expectation — which the nested-sim gate measures. |
| K4 | minor | metrics.py return annotations | `alive_fraction`/`pool_*`/`_efk_increment_summary`/`stage_variance_share` annotated `dict[str, float]` but carry `list`/`int` values. | Fixed R1: changed to bare `dict`, matching sibling functions (`skill_variance_share`, `champion_dominance`). |
| G1 | informational | gate (Priority 3) | The surrogate's **total** suspense/surprise agrees with the nested-sim ground truth within Monte-Carlo error at the headline regime (triangular, mix=0.15: z∈[1.0,1.4] over 3 fields), but the downward bias **grows with group mass** (linear, mix=0.5: z≈7.8, surrogate 0.54 vs nested 0.75). The **group-phase share** (the headline number) tracks ground truth across both regimes (within ~0.01). | **Reported, not "fixed"** — this is the documented one-sided smoothing bias (score-vector conditioning is coarser than the full bracket). Recorded in the metric docstrings and the milestone report; the estimator-free Priority-1 metrics remain the engagement headline per the plan's fallback. |
| G2 | informational | gate per-transition | Surrogate carries spurious belief mass on the FINAL→CHAMPION transition (the running score at the FINAL boundary credits the champion only as a finalist, so the score vector cannot resolve the title), which the full-information nested-sim resolves at the SF→FINAL step. | Reported. Deep-knockout per-transition allocation from the surrogate is unreliable; only the group-phase share and the total (at low mix) are trustworthy. |

**Counts:** 0 critical · 1 major (fixed) · 3 minor (fixed) · 2 informational (reported, by-design).

## Structural guarantees asserted (not merely numerical)
- `mix=0`/short-circuit untouched; hot path byte-identical (verified: `simulate_tournament` body unchanged).
- `Wcond` rows on the probability simplex at every boundary; CHAMPION boundary == realised credit exactly.
- Nested-sim ground truth: exactly one champion per inner sim; correct bracket stage histograms
  ([16,8,4,2,1,1] from R32 start, [8,4,2,1,1] from R16 start); FINAL/CHAMPION boundary == realised credit;
  final-match replay matches `knockout_advance_prob` to MC error.
- k-NN surrogate deterministic across repeated runs (no RNG; verified).

## Residual risk
- The suspense/surprise **absolute level** is calibrated only at low group mass; it under-states
  increasingly as `mix` rises. Any use of the suspense headline at `mix ≳ 0.3` must either run the
  nested-sim at the chosen cell or fall back to `stage_variance_share`/`alive_fraction`. The plan's
  decision rule keys the engagement floor on the *group-phase suspense share*, which the gate shows is
  the better-estimated quantity — but this should be re-confirmed at the chosen frontier cell.
- The nested-sim gate here is **smoke-level** (n_sims≈1200–3000, subset 120–300, inner 200–400), below the
  plan's ≥1000-replicate × full-inner budget. It is sufficient to characterise the bias direction and the
  low-mix agreement, not to certify the headline; a full-budget gate at the recommended cell is deferred to
  the sweep (Milestone-3+).

## AI-assistance (ICMJE 2026)
Implementation + self-audit by Claude Opus 4.8 under human orchestration. Single-reviewer audit (parallel
ensemble unavailable in-thread, disclosed above). No ReproLog emitted: this milestone adds library code +
tests only, no `artifacts/`/`logs/` run.
