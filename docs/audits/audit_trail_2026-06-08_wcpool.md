# Audit trail — wcpool Monte Carlo deliverable (2026-06-08)

Pattern: `audit-remediate-loop`, 5 parallel specialist auditors (quant, code, literature,
reproducibility, format). Two rounds run; exited after round 2 with only documented minor
residuals. Severity key: **C**ritical / **M**ajor / **m**inor. Disposition: FIXED /
DOCUMENTED / VERIFIED-OK / WONTFIX.

## Round 1

| # | Sev | Auditor | Finding | Disposition |
|---|-----|---------|---------|-------------|
| 1 | C | quant | Best-response optimised win-prob on the **same eval batch** it was scored on → in-sample leakage; positive exploitability spurious | FIXED — BR decides on the EV/model batch, scored on a held-out eval batch ([draft.py](../../src/wcpool/draft.py), [simulate.py](../../src/wcpool/simulate.py)); out-of-sample margin now ~0 |
| 2 | C | quant | Monte-Carlo SE computed iid over 50k sims, but sims are **clustered by draw**; SE understated ~4× | FIXED — between-draw cluster SE (`_cluster_se`, `*_cluster_se` cols); recommendation moved to ladder level |
| 3 | M | quant | `skill_variance_share` was a biased mean-of-ratios across draws | FIXED — pooled ratio-of-means in `_finalize` |
| 4 | M | quant | Pareto step gave no discrimination; N ranked on sub-SE differences | FIXED — recommend the ladder (discriminating axis); N reported indistinguishable |
| 5 | M | code | `home_advantage` calibration (symmetric ±ha) ≠ application (per-team) | FIXED — `calibrate_beta` calibrates neutral β only |
| 6 | M | code | `_draw_groups` returned `fixed_groups` by reference (aliasing) | FIXED — returns `.copy()` |
| 7 | M | format | `spawn_key=(7, int(spread))` magic salt + lossy float cast | FIXED — `SYNTH_SEED_NAMESPACE` + integer `rating_stream` |
| 8 | M | format | Best-response probe sizing (8×1500) undocumented, below 50k, no SE | FIXED — named constants + documented + noise scale reported |
| 9 | M | format | `champion_probabilities` sizing undocumented | FIXED — docstring note |
| 10 | m | quant | `chi2_uniform` is not a chi-square statistic | FIXED — renamed `imbalance_index`, docstring corrected |
| 11 | m | quant | Variance-seeking skill anomaly in concentrated fields unflagged | DOCUMENTED — assumptions §10 |
| 12 | m | code | Dead `avail` array in `_greedy_picks`; weak product-invariant test; `win_probability` untested for 3-way tie; BR "monotone" comment overstated | FIXED — dead code removed, tests strengthened/added, comment scoped |
| 13 | m | code | `elo_spread` ddof undocumented; scalar-return type union never returns float | FIXED ddof note; WONTFIX type union (cosmetic) |
| 14 | m | literature | Dixon-Coles cited for the independent-Poisson form, but D&C's contribution is a low-score *dependence* correction (not implemented); Skellam ref underspecified | FIXED — attribute independent Poisson to Maher; note D&C correction not implemented; full Skellam cite + DOIs |
| 15 | m | reproducibility | Committed repro-log `git_head` one commit stale; freeze sidecar lacked `tabulate`; date-stamped CSV overwrites | FIXED — `tabulate` pinned; final re-run refreshes HEAD; `--quick` now writes `_quick`-tagged files |
| 16 | m | format | README overstated best-response; `1/6` hardcode; `DEFAULT_SEED` "today's date"; "15–25 SE" hardcoded; mu_total per-tournament cites; citation-format drift | FIXED — README reworded; `1/N_DRAFTERS`; seed comment fixed; SE multiples computed from data; per-tournament cites + DOIs added |
| 17 | m | format | "14 approx ratings" count | VERIFIED-OK — exactly 14 (`source: approx`); legend lines inflated a naive grep |

## Round 2

| # | Sev | Auditor | Finding | Disposition |
|---|-----|---------|---------|-------------|
| 18 | C | quant | Committed `results_main` looked like a 4-draw quick run, mismatching the 25-draw recommendation md ("7× SE") | VERIFIED-OK + FIXED ROOT CAUSE — the **committed** CSV is 25-draw and the md matches it exactly (gap = 6.8× cluster SE); the auditor read a working-tree file transiently overwritten by **parallel auditors' own `--quick` runs** (same date-stamped path). Root-cause fixed: `--quick` now writes `_quick`-tagged files; `make_plots` ignores them |
| 19 | M | format | Committed pip-freeze env sidecars embed the editable-install path with the **OS username** | FIXED — `logs/reproducibility/env/` git-ignored + untracked; repro-log JSON (freeze SHA, no path) stays tracked; env pinned by `uv.lock` |
| 20 | m | code | Fixed-draw regime → single draw → cluster SE NaN → recommendation rendered literal "nan" multipliers | FIXED — `se_known` guard emits an explicit "undefined under single-draw regime" message |
| 21 | m | code | `make_plots` module docstring still said "Pareto frontier" | FIXED — docstring updated to ladder-level minimax |
| 22 | m | code | `balanced_ladder` tie broken by incidental sort/index order | FIXED — explicit documented tiebreak (worst-rank → total-rank → higher skill) |
| 23 | m | code | `StrengthModel.home_advantage` path unexercised/untested | DOCUMENTED — docstring notes it is an unexercised extensibility hook |
| 24 | m | quant | `spearman_mean` (pooled) vs cluster-SE centre (mean of per-draw means) differ when per-draw NaN counts differ | DOCUMENTED — negligible (undefined-rho fraction ≈0 except minor for linear); noted as residual |
| 25 | m | quant/repro | Cluster SE at n_draws=4 has 3 dof (noisy); pyproject uses floor specifiers not `==` | WONTFIX — only affects `--quick`; the deliverable is 25 draws; `uv.lock` is the authoritative pin |
| 26 | m | code | Redundant `slot_equity_imbalance` dict build per draw | WONTFIX — negligible at ≤25 draws |

## Residual risk (carried)

See [assumptions §11](../assumptions.md). Principal items: knockout strength-proportional
tie rule not sensitivity-tested; skill metric sensitive to mean–variance coupling in
concentrated fields (headline fixes EV-greedy); Annex-C third-place matching uses one valid
assignment (argued, not proven, immaterial to aggregates); 14/48 Elo ratings are approximate
placeholders (conclusions rest on the spread + synthetic sweep); confederation draw
constraints and host home-advantage not modelled.

## Verification at exit

- 36 unit/integration tests pass; ruff lint + format clean.
- Structural invariants asserted in tests: per-sim stage counts `[16,16,8,4,2,1,1]`, unique
  champion, group goal conservation, all 495 Annex-C third-place combinations feasible,
  goals-model calibration reproduces the Elo curve (RMSE < 0.02).
- Determinism: identical seed → byte-identical results (verified by reproducibility-verifier).
- Committed recommendation md numerically matches the committed 25-draw results CSV.
