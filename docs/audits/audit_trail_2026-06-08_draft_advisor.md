# Audit-remediate trail — draft_advisor — 2026-06-08

Deliverable: live World Cup draft-pick advisor (`draft_advisor/`), built on the `wcpool`
engine under the `audit-remediate-loop` pattern. Two full parallel-auditor rounds plus a
round-3 remediation. Auditors per round: code-reviewer, quant-auditor, literature-check,
reproducibility-verifier, format-auditor (round 1: all five; round 2: the four with
critical/major findings).

## Round 1 — findings and dispositions

| ID | Sev | Auditor | Finding | Disposition |
|----|-----|---------|---------|-------------|
| R1-C1 | critical | code/quant | `recommend`/`standing` too slow for live use (sweep×reps×Python completions) | FIXED — rank on τ=0 baseline (deterministic, 1 completion/candidate); sweep only for robustness flag (~200× fewer completions) |
| R1-C2 | critical | code | Shared sequential RNG → no common random numbers; order-dependent ranking | FIXED — τ=0 ranking uses no RNG; sweep uses per-(temp,rep) seeded streams (CRN) |
| R1-C3 | critical | quant/code | A1: ranking by sweep-mean `W` (incl. uniform endpoint) reorders candidates; inconsistent with displayed τ=0 levels | FIXED — rank + display both τ=0 |
| R1-C4 | critical | quant | Tukey-fence cliff detector ~50% false positives on smooth declines | FIXED (superseded R2-C1) |
| R1-M1 | major | code/quant | Cliff SE units mismatch (single-temp SE applied to sweep-mean gaps) | FIXED — cliffs on τ=0 `W` with per-candidate SE |
| R1-M2 | major | code | `-1` roster sentinel silently aliases to last team in `pool_scores` | FIXED — `advisor._score` refuses any `-1`; regression test |
| R1-M3 | major | code | `post_draft_summary` used private `objective._position_credit` | FIXED — public `first_place_credit` |
| R1-M4 | major | quant | A3: EV-greedy self-completion can reorder candidates | CARRIED (residual, documented + magnitude noted) |
| R1-M5 | major | repro | Non-atomic board cache write → poison-cache risk | FIXED — temp-file + `os.replace` |
| R1-M6 | major | repro | Repro record misses 13-field mandate (pip-freeze, env, model hash) | FIXED — enriched record (git HEAD, model hash, pip-freeze SHA, uv.lock id, numpy/scipy, host, ts) |
| R1-M7 | major | format | Magic numbers (`9`, `n_bands=3`) | FIXED — named constants with rationale |
| R1-m* | minor | all | A2 ceiling saturation; `objective_W_se` docstring; `n_sims` justification; interface/pyproject inconsistencies; greedy tie rule; CLI undo | FIXED (banding over shown set + average-rank ties; docstring; doc/pyproject; τ=0 routing; undo re-render) |

## Round 2 — findings and dispositions

| ID | Sev | Auditor | Finding | Disposition |
|----|-----|---------|---------|-------------|
| R2-C1 | critical | quant | Cliff bootstrap used a **linear** no-tier null; real boards are **convex** → still ~100% false positives | FIXED (round 3) — replaced with local second-difference (curvature) spike test; verified 0 FP on convex p=1–3, positive control on tiered board |
| R2-M1 | major | quant | Cliff regression test only covered linear decline | FIXED — added convex-decline and mid-board-kink tests |
| R2-M2 | major | quant | Docs overstate the cliff guarantee | FIXED — docs rewritten to the curvature-spike method; cliffs marked display-only |
| R2-M3 | major | repro | Corrupt-cache guard missed `zipfile.BadZipFile` (truncated `.npz`, the dominant mode) | FIXED — added to except tuple; truncated-cache rebuild test |
| R2-M4 | major | format | Stale cliff docs (`TUKEY_K`/`GAP_Z`, item 4, `[E]`/`[H]` samples) | FIXED — acceptance criteria + interface spec updated |
| R2-m* | minor | code/repro/format | CLI undo/cap test coverage; RNG stream namespacing (+7 magic); gap None/0 conflation; `__all__` omits `cli`; `model_hash==git_head` alias; `0.5` inline marker | MOSTLY FIXED — stream ids via SeedSequence entropy; `cli` in `__all__`; `0.5` marker; gap comment. Carried: deeper CLI-undo test |

Verified-by-reproduction (round 3): the R2-C1 fix was checked by re-running the round-2
auditor's convex experiment (`(1-r/k)**p`, p∈{1,1.5,2,3}) → `cliffs=[]` for all; tiered control
→ correct boundaries. Decision logic (robustness sweep size, reach/wait set) was **decoupled**
from cliffs (now fixed caps), so no recommendation depends on the cliff list.

## Residual risk (carried after round 3)

1. **A3 self-completion (R1-M4).** Our future picks complete EV-greedily (one-ply proxy, per
   `wcpool.draft_best_response`); we re-optimise at each real turn, so displayed absolute
   `W`/`P1` are conservative and the quant-auditor showed it can reorder *mid-board* candidates.
   Top-of-board ranking (where decisions are made) is far less sensitive. Mitigation for a
   future round: complete our tail with `draft_best_response`.
2. **Opponent model is assumed, not fitted.** Softmax-over-EV with a σ-scaled temperature grid
   is plausible, not estimated; the sweep is a robustness check, not a calibrated belief.
3. **Neutral venue.** Hosts (USA/Mexico/Canada) get no advantage; `home_advantage` hook exists.
4. **Strength-input calibration (out of scope).** Market-implied/supercomputer calibration is
   the highest-ROI model upgrade but needs external odds; `fit_ratings_to_title_probs` is the hook.
5. **Cliff display.** Boundary gaps (the top team's separation) are intentionally not tested by
   the curvature spike test; that lead is shown in the `W` column.
6. **CLI test depth.** Undo/ambiguity REPL paths are exercised once (full scripted run) but not
   per-branch.

Status: all critical and major findings resolved or explicitly carried with rationale; 75
tests pass; ruff clean.
