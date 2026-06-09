# Audit trail — engagement-metrics audit remediation (Milestone 3, 2026-06-09)

## Deliverable
Apply the 9-item Milestone-3 audit remediation to the engagement-metrics surface (numpy/scipy
only), preserving the existing skill/tie/slot/champ metrics and the RNG path byte-identically and
keeping the γ=0 exact validation gate byte-identical.

- [src/wcpool/metrics.py](../../src/wcpool/metrics.py) — critical `conditional_win_prob`
  degenerate-boundary fix (`_within_state_mean`, `WITHIN_STATE_MEAN_K`, `selected_k_in` cache);
  `alive_fraction` docstring labelling.
- [src/wcpool/simulate.py](../../src/wcpool/simulate.py) — `group_variance_share` ratio-of-means
  pooling (`_ratio_of_means`); `suspense_subsample` run param + `_DEFAULT_SUSPENSE_SUBSAMPLE`;
  `_accumulate_engagement` extraction; type hints on the touched public surface; engagement-column
  documentation (items 2-6, 9).
- [tests/test_engagement_metrics.py](../../tests/test_engagement_metrics.py) — regression tests
  (constant-predictor → prior + mix=0 zero group-share; selected-k cache bit-identity).
- [tests/test_integration.py](../../tests/test_integration.py) — `_scheme_key` aliasing invariant
  (item 8).
- [tests/test_validation_gate.py](../../tests/test_validation_gate.py) — named/justified gate
  constants `RTOL`/`ATOL`/`MAX_CLUSTER_SE`, budget-independence note, per-metric↔worst_overall
  reconciliation (item 7).

## Loop configuration
- Pattern: `audit-remediate-loop` (intended parallel-specialist-ensemble; cap 3 rounds).
- **Executed as a single-reviewer self-audit:** this execution thread exposes no `Task`/`Agent`
  subagent-spawn tool, so the parallel `code-reviewer`/`quant-auditor`/`format-auditor` branches
  could not be dispatched (same constraint disclosed in the Milestone-2 trail). The audit concerns
  were worked through directly against source + numerical probes. Converged in 1 substantive round
  (the findings were supplied pre-triaged by the task).

## Findings & disposition

| ID | Severity | Location | Finding | Disposition |
|----|----------|----------|---------|-------------|
| 1 | **critical** | metrics.py `conditional_win_prob` | Phantom prior→boundary increment from a k-NN LOO on a (near-)constant predictor: at the GROUP boundary at small mix (1 unique state at mix=0), the smoother returned the mean of `k` arbitrary other replicates — a noisy sub-sample of the prior — yielding `pool_suspense_group_share ≈ 0.008` at mix=0 (must be 0). | **Fixed.** Degenerate boundaries (`n_unique <= max(knn_k_grid)`) use the exact within-state mean `E[credit\|state]`; the fully-constant case returns `terminal_credit.mean(axis=0)` **bit-identically** to the prior → prior→GROUP increment EXACTLY 0.0. cKDTree k-NN retained for genuinely high-cardinality boundaries. Determinism + CHAMPION exact case preserved. Regression test added. **Deviation flagged** — see below. |
| 2 | major | simulate.py `_finalize` | `group_variance_share` was a mean-of-ratios while `skill_variance_share` is a ratio-of-means; the `stage_variance_share` docstring claimed they mirror. | **Fixed.** Accumulate raw per-draw `var_group`/`var_total`; `_finalize` forms `mean_draws(var_group)/mean_draws(var_total)` via `_ratio_of_means`. Per-draw share list kept only for the cluster SE. |
| 3 | major | simulate.py public surface | Missing type hints. | **Fixed.** `_scheme_key`, `run_strength_config` (full sig + `-> list[dict] \| tuple[list[dict], dict]`), `_finalize`, `_accumulate_champion`, `_cluster_se`, `_run_draft`, `_ratio_of_means`, `_accumulate_engagement`. |
| 4 | minor | metrics.py `alive_fraction` | Saturated (≥0.999, cluster SE 0) over the studied grid. | **Fixed (labelled).** Docstring + `run_strength_config` engagement doc note it is uninformative, not a discriminating axis; bound deliberately left loose. |
| 5 | minor | metrics.py suspense/surprise | `pool_surprise` byte-identical to `pool_suspense` on the realised-path surrogate. | **Fixed (labelled).** Documented that surprise is not independent confirmation; genuine split gated on §6 nested sim. Both functions retained. |
| 6 | minor | simulate.py run path | Suspense estimator ≈98% of engagement compute. | **Fixed.** `suspense_subsample` (data-driven default 512) caps eval sims fed to the surrogate; estimator-free metrics at full budget; per-cell selected-k cached. mix=0 group-share stays 0 after subsampling. |
| 7 | minor | test_validation_gate.py | Bare literals. | **Fixed.** `RTOL`/`ATOL`/`MAX_CLUSTER_SE` named + justified; budget-independence + per-metric↔worst_overall reconciliation documented. |
| 8 | minor | simulate.py `_scheme_key` | No aliasing-invariant lock. | **Fixed.** Test asserts `ScoringScheme("linear",2.0,1.0,0.0)` → 4-tuple, not the bare-name 3-tuple. |
| 9 | minor | simulate.py hot loop | Hot-loop length / "per draw" comment. | **Fixed.** `_accumulate_engagement` extracted; comment tightened to "per draw within this cell". |

**Counts:** 1 critical (fixed) · 2 major (fixed) · 6 minor (fixed). 0 residual blocking.

## Deviation from the task brief (flagged per CLAUDE.md "flag any discrepancy")
The task's *descriptive* claim that the critical fix also removes "79–85% of the reported
group-phase suspense at mix∈{0.05,0.12,0.15}" is **not realisable by the prescribed mechanism** and
was not reproduced. Evidence (triangular, 3000 sims, seed 42):

| mix | group_share BEFORE | group_share AFTER | group-phase reduction |
|-----|-----|-----|-----|
| 0.00 | 0.007944 | **0.000000** | **100%** |
| 0.05 | 0.036837 | 0.036837 | 0% |
| 0.12 | 0.037458 | 0.037458 | 0% |
| 0.15 | 0.037228 | 0.037228 | 0% |

The prescribed trigger is exact-duplicate degeneracy (`n_unique <= k`). The mix=0 GROUP boundary has
**1** unique state → fixed exactly. The low-mix GROUP boundary has **~400–1335** unique states (of
2000–3000) → genuinely high-cardinality by exact-duplicate count, so the prompt's own directive
("keep the cKDTree k-NN for genuinely high-cardinality boundaries") leaves it on k-NN. The exact
within-state mean is mathematically *incapable* of reducing the low-mix group-share: applied
per-state (LOO) at low mix it **inflates** the group-phase ~10× (singleton-dominated states collapse
to one-hot credit), the over-conditioning direction — measured at -960% to -977%. Any reduction at
low mix would require an *over-smoothing* estimator (larger k / a value-axis kernel), which the task
does not prescribe and which would attenuate genuine signal. The 0.037 low-mix group-share is
therefore the genuine k-NN signal and is consistent with the Milestone-2 nested-sim finding (G1: the
group-phase **share** tracks the bias-corrected ground truth within ~0.01). Faithful implementation
of the prescribed mechanism satisfies every coded acceptance criterion (mix=0 group-share == 0 to FP;
constant predictor → Wcond == prior; high-cardinality kept on k-NN; determinism; CHAMPION exact).

## Structural guarantees asserted
- γ=0 exact gate byte-identical: every existing-metric `max|d| = 0.000e+00`; legacy-shim byte-identity
  test passes; cross-check within ~0.03 cluster SE.
- mix=0 `pool_suspense`/`pool_surprise` group-share == 0.0 **exactly** (suspense, surprise, and after
  subsampling); fully-constant predictor → `Wcond == prior` bit-for-bit at every non-terminal boundary.
- Selected-k cache and `suspense_subsample` head-slice are deterministic and bit-identical to the
  uncached/unsubsampled fit on the same batch; whole-run determinism verified.
- `group_variance_share` ratio-of-means matches a manual reconstruction bit-for-bit.
- M4 consumers (run_experiment.py, participant_sweep.py) untouched and exercised via both call
  patterns; new param is keyword-only with a default.

## Verification
- `uv run pytest -q`: 133 passed (130 baseline + 3 new). `uv run ruff check`: All checks passed.

## Residual risk
- The suspense **absolute level** remains provisional/one-sidedly biased (Milestone-2 G1); subsampling
  adds a small documented upward distortion (~0.016 at sub=512 on the headline cell) absorbed in the
  metric's own cluster SE. The estimator-free `stage_variance_share` is the engagement headline
  regardless. The selected-k cache assumes a structurally stable batch within a cell (holds: all draws
  in a cell share `sims_per_draw` and scheme).

## AI-assistance (ICMJE 2026)
Remediation + single-reviewer self-audit by Claude Opus 4.8 under human orchestration (parallel
auditor ensemble unavailable in-thread, disclosed above). No ReproLog emitted: library code + tests
only, no `artifacts/`/`logs/` run.
