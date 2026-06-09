# Audit trail — group-stage scoring execution (2026-06-09)

Consolidated record of executing [plan_groupstage_scoring_2026-06-09.md](../plan_groupstage_scoring_2026-06-09.md) in full.
Each milestone ran an `audit-remediate-loop` (parallel specialists; ≤3 rounds). Per-phase trails:
[research](audit_trail_2026-06-09_groupstage_research.md), [plan](audit_trail_2026-06-09_groupstage_plan.md),
[engagement remediation](audit_trail_2026-06-09_engagement_metrics_remediation.md), [selection](audit_trail_2026-06-09_groupscoring_selection.md).

## Objective used
**Engagement-constrained skill** (plan §5 (iii), the recommended default) — the frontier deliverable lets the owner override the engagement level.

## Milestones, audits, and the substantive bugs caught
| M | Deliverable | Auditors | Key issues caught & fixed |
|---|---|---|---|
| 1 | Two-layer scoring engine (`scoring.py`; `tournament.py` group W/D emission); identity tests | quant-auditor, code-reviewer | input validation + interior-boundary `running_scores` tests added; `GroupTallies` contract; mix=0 bit-identity proven |
| 2 | Engagement metrics + numpy/scipy `Wcond` k-NN estimator + nested-sim gate | quant-auditor, code-reviewer | **LOO self-leak under duplicate states** (fixed); **FINAL-boundary over-conditioning** in the GT (fixed); **finite-`n_inner` GT bias** → unbiased cross-term; gate moved to spec budget (1000×800); magic `z<4.0` → MC-derived non-inferiority margin |
| 3 | `simulate.py` threading + the γ=0 validation gate | quant-auditor, code-reviewer | **γ=0 gate passes byte-identically** (max diff 0.0 vs pre-refactor); **phantom suspense increment from a degenerate GROUP-boundary k-NN** (fixed → mix=0 group-share exactly 0); group-share pooling → ratio-of-means |
| 4 | γ→mix solver, 48-cell grid, headline sweep, ReproLog | quant-auditor, reproducibility-verifier, code-reviewer | **absolute skill level seed-fragile at 25 draws** → re-run at 100 draws + emit the **seed-stable paired Δskill frontier** (7× tighter); provenance sidecars; `model_hash=git_head` parity |
| 5 | Selection + concrete recommendation + layperson report + sensitivities | quant-auditor, format-auditor, literature-check, code-reviewer | **integer-scheme tie rate measured** (0.5%, not the continuous 0.2%); integerization justified by commensurability (not a reverse-fitted tolerance); higher-γ options honestly costed; engagement framing corrected; report re-rendered |
| 6 | Live `draft_advisor` board rewire | code-reviewer | affine-equivalence to the integer ladder proven; pre-refactor-cache crash fixed; mix=0 cache canonicalised to a fixpoint; non-negative group points validated |

## Validation gates (both passed)
- **Scoring identities** (M1): constant-sum `D=W/2`, mix=0 ≡ terminal ladder (bit-identical), monotonicity, `running_scores` reconciliation.
- **γ=0 reproduction** (M3/M4): the refactored mix=0 path reproduces the pre-refactor output byte-identically, and the 100-draw γ=0 anchors match the participant-sweep 8-participant baseline (linear 0.211 / triangular 0.283 / geometric 0.295; 3-seed cross-seed sd ≈ 0.006).

## Final recommendation (deliverable)
Triangular knockout shape, group **3:1**, at the commensurability landmark **γ_match (≈0.16)**. Integers: **group 3/win, 1/draw; knockout bank 9, 18, 27, 36, 45, 54** (terminal 9/27/54/90/135/189). Measured vs the dead-group-stage baseline (within the 100-draw run): skill 0.283→**0.281** (paired Δ −0.0025, 0.6 cluster-SE, statistically free), tie **4.9%→0.5%** (integer scheme, measured), champion-dominance 0.81→**0.80**, champion-undrafted 0 (full 8×6 field). The frontier + menu (γ_match / 0.25 / 0.5) are in [recommendation_2026-06-09.md](../tables/recommendation_2026-06-09.md) and [report_draft_pool_layperson_2026-06-09.md](../report_draft_pool_layperson_2026-06-09.md).

## Residual risk
1. **Engagement level is the owner's normative call.** γ_match is the *skill-preserving* operating point (group phase = 0.4% of outcome variance — structural engagement, minimal outcome shift). γ=0.25 gives ~3× the engagement at still-<1-SE skill cost; γ=0.5 gives ~23× (group_variance_share 0.10, champ-dominance 0.82→0.70) at ~4 cluster-SE skill cost. The report presents all three; the recommended default is γ_match.
2. **`pool_suspense` is provisional** — the k-NN surrogate is validated (MC-non-inferiority) only at the low-mix headline cell and is non-monotone for geometric where the share is tiny. The **estimator-free `group_variance_share`** (exact, monotone, 0 at γ=0) is the engagement headline; suspense is labeled provisional in the sidecars.
3. **Absolute skill levels are seed-sensitive at low draw counts** (cross-seed sd ≈0.015 at 25 draws); the recommendation rests on the **paired** Δskill (seed-stable) and the 100-draw levels. Comparisons to the prior study's absolute anchors carry this caveat.
4. **Draw model**: independent-Poisson field draw rate (0.1975) already matches empirical WC group (0.1944); matched-ρ≈0, recommendation invariant (confirmatory, not re-implemented).
5. **Open owner items:** the report title "Knights'" is a proper noun — confirm it is not deanonymising before any pseudonymous (SKIE) release; and all execution artifacts are **uncommitted** working-tree files (no git provenance anchor yet).

## State
`uv run pytest -q` → **190 passed**; `uv run ruff check` → clean. git HEAD `5c4ef6a` (nothing committed). numpy/scipy/matplotlib only (no new dependencies; scikit-learn deliberately avoided).

**AI-assistance (ICMJE 2026):** engine, metrics, sweep, selection, report, and this audit trail produced by Claude Opus 4.8 (1M context) under human orchestration, via per-milestone parallel-specialist audit-remediate loops. AI is not an author.
