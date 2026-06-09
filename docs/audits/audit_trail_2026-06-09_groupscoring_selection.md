# Audit trail — group-scoring selection + report (Milestone 5)

**Date:** 2026-06-09 · **Deliverable:** the engagement-constrained-skill recommendation,
integerisation, figures, and layperson report for the group-stage scoring extension.
**Loop:** audit-remediate-loop, 1 round (no critical/major findings; inline self-audit across the
calculations / code / reproducibility / format specialist concerns, as nested subagent spawning was
unavailable in this execution context).

## Scope audited
- `src/wcpool/select.py` (selection rule + integerisation)
- `scripts/build_recommendation.py`, `scripts/make_plots.py` (group-scoring figures)
- `tests/test_select.py`
- `docs/tables/recommendation_2026-06-09.md`, `docs/tables/results_summary_2026-06-09.csv`
- `docs/report_draft_pool_layperson_2026-06-09.md`

## Findings + disposition

| # | Severity | Location | Finding | Disposition |
|---|---|---|---|---|
| 1 | (verify) | select.py / recommendation.md | Recommendation (triangular, 3:1, gamma_match, ints 9-27-54-90-135-189) is the OUTPUT of the decision rule, not hand-picked | VERIFIED: `select_recommendation` reproduces it from the CSV; invariant across engagement floors {1e-4..4e-3}. No action. |
| 2 | (verify) | select.py:select_shape_prior_minimax | phi=0 limit recovers the shipped triangular default | VERIFIED: prior minimax on the gamma=0 anchors returns triangular (worst-rank 2 < lin/geo 3), matching recommendation_2026-06-08. No action. |
| 3 | (verify) | select.py | Skill axis: paired delta (within-shape, per-shape-anchored) for D/W+gamma; absolute Spearman (cross-shape) for the shape choice | VERIFIED: correct separation; the paired delta cannot compare across shapes (each anchored to its own gamma=0), so absolute is used for the shape and paired within. No action. |
| 4 | (verify) | select.py:integerize_scheme | scale=9 hits gamma exactly via the deterministic occupancy [16,16,8,4,2,1,1] inner product; monotone; exact integer commensurability 3W=A_int(R32)=9 | VERIFIED: gamma=0.157435 occupancy-exact; ladder strictly increasing; 3*3==9==R32 bank. Occupancy reproduces every shape's calibrated knockout purse. No action. |
| 5 | (verify) | exploitability CSV | br-evg=-0.0009, var-evg=-0.0056, noise 0.0090 -> non-exploitable | VERIFIED: margins negative and within the between-draw noise scale; matches the prior gamma=0 finding and Doc-3 prediction. No action. |
| 6 | (verify) | build_recommendation.py | No magic phi: engagement floor 1e-4 only excludes the gamma=0 status quo; frontier presented relative to data-derived anchors (floor=0, equal-purse ceiling) | VERIFIED: recommendation is floor-invariant (landmark-driven); anchors are the gamma=0 floor and gamma=0.5 ceiling, both read off the data. No action. |
| 7 | (verify) | report + recommendation.md | All lay-facing numbers match the frontier CSV | VERIFIED: anchor/rec skill (0.283/0.281), tie (4.9%/0.2%), champ-dom (81%/80%) all match the CSV exactly; draw-rate claim (19.8%/19.4%) matches research_draw_lowscore (0.1975/0.1944). No action. |
| 8 | minor | select.py:_ranks | Hand-rolled average-rank helper | VERIFIED equal to pandas.rank over 200 random tie-heavy cases (both directions). No action. |

## Residual risk
- **Integerisation `round()` is a latent no-op for the three integer-base ladders.** All of
  linear/triangular/geometric have integer points, so `round(scale * A)` never actually rounds for
  the studied shapes; the rounding-direction (round-half-to-even via `np.rint`) is therefore
  exercised only if a future non-integer-base shape is added. Correct for all current shapes; the
  rounding-active path is untested because no current shape triggers it. Low risk (the three shapes
  are prescribed and integer).
- **The engagement axis `group_variance_share` at gamma_match is small (~0.0044).** The group layer
  contributes a modest *share* of final-standings variance by design (skill is preserved); the
  engagement benefit is principally the qualitative "every drafter scores from match 1", honestly
  stated as such in the report. The provisional EFK suspense surrogate is NOT used (it is un-gated
  at this cell per the existing sidecar provenance); the estimator-free `group_variance_share`
  carries the engagement story.
- **Absolute skill level offset.** The 100-draw headline anchors sit ~3.7-4.1 cluster SE above the
  plan section-8.4 participant-sweep targets (seed/budget difference: 100 draws @ 20260609 vs 25 @
  20260608). The recommendation rests on the SEED-STABLE paired delta (not the absolute level), so
  this does not affect the choice; documented in the result sidecars.

**Conclusion:** no critical or major findings. The recommendation is the verified output of the
stated decision rule; all reported numbers trace to the committed frontier CSV; the integerisation
is occupancy-exact with the commensurability landing on whole numbers. Deliverable accepted.
