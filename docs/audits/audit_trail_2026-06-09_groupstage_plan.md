# Audit trail — group-stage scoring execution plan (2026-06-09)

## Deliverable
[docs/plan_groupstage_scoring_2026-06-09.md](../plan_groupstage_scoring_2026-06-09.md) — execution plan for the
group-stage (W/D) + progressive-knockout scoring extension (8 drafters, full 48-team draft), synthesised from the four
[research notes](../research) and the [prior study](../tables/recommendation_2026-06-08.md).

## Loop configuration
- Pattern: `audit-remediate-loop`. Cap 3 rounds; **converged in 1**.
- Round-1 auditors (parallel): `quant-auditor` (experimental/statistical design + independent verification of engine claims, with Bash), `reproducibility-verifier` (repro mechanisms + seeds + validation anchor, with Bash), `format-auditor` (magic numbers, identity hygiene, format, scope completeness).
- `literature-check` not re-spawned: the plan cites the already-verified Phase-1 research notes, not new primary sources.

## Findings & disposition (Round 1)

| ID | Severity | Finding | Disposition |
|----|----------|---------|-------------|
| R1 | **critical** | §8.4↔§11 seed contradiction (reuse `20260608` vs bump to `20260609`); "byte-comparable" overclaimed — only `stages` arrays are byte-identical, mix=0 metrics match to FP tolerance | Fixed R1 (anchor reuses `20260608` → stages byte-identical + FP-tol on metrics; new grid uses `20260609`; both logged) |
| R2 | **critical** | headline `Wcond` estimator ("CV-selected k-NN/kernel regression") would pull in scikit-learn, which is **not** a pinned dependency — breaks env fingerprint | Fixed R1 (estimator is numpy/scipy-only: `scipy.spatial.cKDTree` + hand-rolled LOO; no sklearn without `==` pin + uv.lock regen) |
| Q1 | major | §3 conflates "full draft" with "equal roster sizes"; affine invariance needs only equal rosters (snake guarantees it); full draft only buys `champion_undrafted_rate=0` | Fixed R1 (restated; verified numerically by the auditor) |
| Q3 | major | the martingale unit test is necessary-not-sufficient (a zero-suspense smoother passes it); the nested-sim **distribution** gate is the sole guard against smoothing bias | Fixed R1 (re-scoped; gate compares distributions, not stage means) |
| R3 | major | CV nondeterminism + false `calibrate_beta` analogy (that fit is deterministic, no CV/RNG) | Fixed R1 (deterministic LOO specified; analogy dropped; determinism unit test added; seeded-fold escape hatch) |
| R4 | major | nested-sim "named entropy slot" unspecified + collision hazard with batches {0,1,2} and synthetic namespace 7 | Fixed R1 (pinned to `spawn_key=(cfg_id,draw,3,replicate_idx)`, recorded in grid_meta) |
| R5 | major | new artifacts (result CSVs, figures, report, `participant_sweep`) lack provenance binding + clobber-guards | Fixed R1 (§12 provenance bullet: provenance column/sidecar; make_plots+report ReproLog; parametrise+guard participant_sweep) |
| Q2 | minor | §4 γ grid mislabeled "geometric/denser toward 0"; gaps were sparser toward 0 | Fixed R1 (grid → {0,0.05,0.12,0.25,0.50}, gaps widen toward equal-purse end) |
| Q4 | minor | k-NN smoothing biases suspense/surprise **downward** (one-sided); CV-for-prediction ≠ CV-for-increment-variance | Fixed R1 (bias direction stated; select k against the suspense discrepancy on the nested-sim subset) |
| Q5 | minor | γ=1 boundary semantics; φ knee-rule + suspense-share estimator not pre-registered | Fixed R1 (γ=1 excluded by construction noted; estimator + cluster SE + knee rule pre-registered) |
| R6 | minor | `model_hash` passed null on the design path (board.py sets it) | Fixed R1 (pass `model_hash=git_head` in run_experiment for parity) |
| R7 | minor | §9 mix=0 not bit-identical (FP reorder) vs legacy `ladder[stages]` | Fixed R1 (short-circuit `mix==0` to the bare `ko` lookup) |
| F1 | minor | (== R1) seed-provenance clarification | Fixed via R1 |

**Counts:** 2 critical (fixed) · 5 major (fixed) · 5 minor (fixed) · 0 remaining.

## Independently verified CORRECT (auditors re-checked against code/tables, did not take prose on trust)
- **Engine claims (§9):** `simulate_group_stage` computes per-match win/draw and discards them (tournament.py L177-181); `simulate_tournament` returns the `(n_sims,48)` furthest-stage array with "alive after round k ⇔ stages≥k"; `mix=0` reproduces `ladders.points_for_stages`; `objective_W=(n−1)P1+P2` is scoring-agnostic (acts on `pool_scores`).
- **Prior numbers (§8.4):** triangular 0.2184 ± 0.0117, geometric 0.2462 ± 0.0123, linear 0.1515 ± 0.0095 (8-drafter rows) match [recommendation_2026-06-08.md](../tables/recommendation_2026-06-08.md); §5 floors (0.169 skill, 0.186 tie) match the per-ladder summary.
- **Math:** γ(mix) monotone; equal-purse landmark γ=0.5; `1:0.5 ≡ 2:1` affine-redundant (ratio std 1.7e-15); cluster SE ∝ 1/√n_draws (`_cluster_se`, ddof=1).
- **Repro infra present:** `emit_repro_log` (13-field, hashes grid_meta, atomic); `board._write_repro_log` (atomic publish); `SeedSequence(seed, spawn_key=(cfg_id,draw,batch))` + `SYNTH_SEED_NAMESPACE=7`; pinned `uv.lock`.
- **Format:** identity hygiene clean (no real name / git email / OS username / absolute user path; repo-relative links; sole author string is the SKIE/AI-assistance line); every numeric value sourced; filename conforms; all 13 sections present.

## Residual risk
1. **The headline `pool_suspense`/`pool_surprise` remain the least-reproducible, most-uncertain numbers** even after remediation: the k-NN surrogate's one-sided downward smoothing bias is bounded — not eliminated — by the nested-sim distribution gate. The estimator-free `stage_variance_share` + `alive_fraction` are the robust engagement fallback, so the deliverable is not gated on the suspense estimator.
2. **The objective (§5) is the one non-derivable decision** (skill-max / engagement-primary / engagement-constrained-skill, recommended). The concrete `W_pts`, `D_pts`, shape, and γ — hence the final integer ladder — are functions of that choice and the engagement floor φ. Pre-registration (estimator, cluster SE, knee rule) makes φ-selection reproducible once the objective class is fixed.
3. Carried from the [research audit trail](audit_trail_2026-06-09_groupstage_research.md): paywalled primaries verified at abstract/metadata level; no published precedent for team-exclusive draft-pool / group-stage scoring (first-principles extrapolation, flagged for Monte-Carlo).

## Provenance
git HEAD at audit: `5c4ef6a`. Auditors: `quant-auditor`, `reproducibility-verifier`, `format-auditor`. Producers (2 parallel planning leads) + remediation: `general-purpose`. AI-assistance (ICMJE 2026): Claude Opus 4.8 — plan synthesis, independent design/repro audit, remediation — under human orchestration.
