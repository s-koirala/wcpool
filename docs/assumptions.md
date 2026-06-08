# Modeling assumptions — wcpool

Every assumption behind the Monte-Carlo environment, with its source and rationale.
Sections map to the engine modules. Residual risk is at the bottom (filled after audit).

## 1. Tournament structure  ([tournament.py](../src/wcpool/tournament.py))

| Assumption | Value / rule | Source |
|---|---|---|
| Teams / groups | 48 teams, 12 groups of 4, single round-robin (6 matches/group, 3 each) | FIFA 2026 format (Wikipedia "2026 FIFA World Cup"; FIFA tournament regulations) |
| Group points | Win 3, draw 1, loss 0 | FIFA competition regulations |
| Group tie-break | points → goal difference → goals for → drawing of lots | FIFA regulations (R32 ranking). **Head-to-head and fair-play points are omitted** — see §11. |
| Advancement | 12 winners + 12 runners-up + 8 best third-placed | FIFA 2026 format |
| Third-place ranking | points → GD → GF → lots, best 8 of 12 | FIFA 2026 format |
| R32 pairings | Verbatim official template (matches 73–88), incl. the "winners face thirds, 8/4 runner-up split" structure | Wikipedia "2026 FIFA World Cup knockout stage" |
| Third-place → R32 slots | Official eligibility template (FIFA Annex C, 495 combinations) solved as a bipartite matching that respects each slot's eligible-group set | R32 pairing labels "3rd (X/Y/Z/...)"; Annex C referenced therein |
| Knockout tree | R32→R16→QF→SF→Final, verified pairings (matches 89–104) | Wikipedia "2026 FIFA World Cup knockout stage" |
| Knockout ties | Decided by the strength-weighted tie rule (extra-time/shootout proxy), so no draws in knockout | model choice (§2) |

**Annex C approximation.** The official table fixes one specific valid assignment per
combination; we solve a min-cost bipartite matching, which selects *a* valid assignment
respecting the same eligibility sets. All 495 combinations are feasible under the
eligibility template (asserted in [test_tournament.py](../tests/test_tournament.py)). The
specific valid matching only permutes which group winner faces which third-placed team; it
does not change the strength distribution a team faces, so aggregate pool-scoring metrics
are insensitive to it.

## 2. Strength / outcome model  ([strength.py](../src/wcpool/strength.py))

- **Win probability = Elo expected score.** `W_e(dr) = 1/(1+10^(-dr/400))`, the World
  Football Elo logistic (eloratings.net system; Wikipedia "World Football Elo Ratings").
  This is the *expected score* (a draw counts 0.5), the quantity the goals model is
  calibrated to reproduce.
- **Goals = independent Poisson**, log-rate linear in the rating gap (Maher 1982,
  *Statistica Neerlandica* 36:109; Dixon & Coles 1997,
  [JRSS-C 46:265](https://doi.org/10.1111/1467-9876.00065)):
  `log λ_A = log(μ_total/2) + (β/2)(dr/400)`, symmetric for B. Group W/D/L, GD and GF all
  follow from the sampled scoreline; the third-place tie-breakers are therefore produced by
  the same model rather than bolted on. Win/draw/loss probabilities are the **Skellam**
  law of the goal difference (Skellam 1946).
- **μ_total (goals/match)** is pinned to the pooled men's World Cup rate over 2014/2018/2022
  (171+169+172 goals over 192 matches ≈ 2.667), not a free constant.
- **β (supremacy slope)** is *calibrated*, not hand-set: chosen by least squares so the
  model's implied expected score `P(win)+0.5·P(draw)` reproduces the Elo logistic across
  the field's realised pairwise rating gaps. Calibration diagnostics (RMSE vs the Elo
  curve, implied even-match draw rate ≈ 0.27, within the historical band) are stored on the
  model and asserted in [test_strength.py](../tests/test_strength.py).
- **Knockout tie rule.** `P(A advances) = P(A win) + P(draw)·P(A win)/(P(A win)+P(B win))`
  — extra-time/shootout modelled as strength-proportional. This is a modelling choice;
  real shootouts are closer to a coin flip, so this *slightly* overstates favourite
  survival in knockouts (sensitivity not yet run — see §11).
- **Home advantage** defaults to 0 (neutral field). The three 2026 hosts' home edge
  (Elo's conventional +100) is exposed as a parameter but off by default, since a
  neutral field is the cleaner basis for a scoring-design study.

## 3. Rating sources (the pluggable input)  ([strength.py](../src/wcpool/strength.py))

- **Real Elo** ([config/ratings_elo_2026.yaml](../config/ratings_elo_2026.yaml)):
  eloratings.net snapshot via the Wikipedia data module, dated 2026-06-01. **14 of 48
  ratings are approximate placeholders** (flagged `source: approx`) for qualifiers outside
  the sourced top-50; design conclusions rest on the field's *spread* and on the synthetic
  sweep, both robust to ±50 Elo on those teams.
- **Synthetic favoritism** generates i.i.d. Gaussian ratings with a single knob — the
  standard deviation `spread`. The sweep uses spread ∈ {0.5, 1, 2}×(empirical Elo SD): a
  documented stress range (half to double the real dispersion), anchored to data rather
  than chosen ad hoc. The induced top-8 title-probability concentration is *measured and
  reported* (it is the interpretable axis), not separately tuned.
- **Bookmaker-implied** is supported via `fit_ratings_to_title_probs`: de-vig the implied
  title odds, then fit a single temperature on a rank-Gaussian rating template so the
  *simulated* champion distribution matches the de-vigged probabilities (the temperature is
  fitted, not set; calibration error reported). Provided as a utility; the headline run
  uses real Elo + the synthetic sweep.

## 4. Draw model  ([config.py](../src/wcpool/config.py), [tournament.py](../src/wcpool/tournament.py))

- **Primary: pot-constrained random draw.** Each replicate draws one team per pot into
  each group. Pots are the official 2025-12-05 draw pots for the real field, or
  rating-rank pots for synthetic fields.
- **Confederation spread constraints are not modelled** (a documented simplification; they
  perturb which specific teams meet but not the strength distribution materially).
- **Fixed-draw sensitivity** uses the official group assignment throughout (Mexico→A,
  Canada→B, USA→D, confirmed against FIFA).

## 5. Draft model  ([draft.py](../src/wcpool/draft.py))

- 6 drafters, snake order (1..6, 6..1, …), N teams each, N ∈ {4,5,6,8}.
- **EV-greedy** (default): take the highest expected-points team available under the active
  ladder. **Variance-seeking**: take the highest points-variance team. **Best-response**:
  one drafter maximises its own pool-win probability assuming all remaining picks proceed
  EV-greedily — a *one-ply greedy* best-response (not a full game-theoretic equilibrium),
  used only to test whether EV-greedy is exploitable.
- Drafter score = sum of drafted teams' terminal points under the active ladder.

## 6. Simulation regime  ([simulate.py](../src/wcpool/simulate.py))

- **Resampled-draw regime is primary**: each replicate redraws groups and re-drafts, so
  conclusions describe the *design* rather than one bracket realisation. Fixed-draw is a
  sensitivity.
- **Independent EV and eval batches.** Team EV/variance for drafting are estimated on an EV
  batch; rosters are scored on a separate eval batch. This removes in-sample optimism from
  the skill-vs-luck estimates.
- **Shared-simulation optimisation.** The tournament outcome is independent of ladder, N and
  policy, so each draw's batch is simulated once and reused across the whole grid (the
  ladder re-weights stage outcomes; the draft re-partitions teams). Every cell is therefore
  scored on the same `n_draws × sims_per_draw` eval tournaments.
- **Sample size.** Default 25 draws × 2000 sims = 50,000 eval tournaments per cell (meets
  the ≥50k requirement). Monte-Carlo SE on the skill correlation is ≈ 0.0019.
- **Seeding.** Master seed = 20260608 (the run date). Independent streams per
  (config, draw, batch) via `numpy.random.SeedSequence(seed, spawn_key=...)`, fully
  reproducible.

## 7. Metric definitions  ([metrics.py](../src/wcpool/metrics.py))

1. **Skill vs luck.** (a) *Spearman* between the fixed pre-draft roster-EV ranking and the
   per-replicate realised-score ranking, averaged over replicates (average-rank tie
   handling). (b) *Variance share* = σ²_between / (σ²_between + σ²_within): roster-EV
   variance over noise variance — an ICC-style index (1 = placing fully determined by the
   draft, 0 = pure luck). Between-component uses the independent roster EV.
2. **Slot equity.** Win probability by snake slot; ties at the top split fractionally.
   `slot_win_prob_spread` = max−min across slots (0 = perfectly equitable).
3. **Ties.** Fraction of replicates with ≥2 drafters sharing the top score; plus the mean,
   SD and 5/95 percentiles of the winning score.
4. **Champion dominance.** P(the drafter holding the eventual champion wins the pool),
   conditioned on the champion being drafted; the champion-undrafted rate is reported
   separately (relevant at small N).
5. **Robustness.** Metrics 1–4 recomputed across the synthetic concentration sweep and the
   real-Elo anchor.

## 8. Scoring ladders  ([ladders.py](../src/wcpool/ladders.py))

Exact, prescribed (no free constants); a group-stage exit scores 0 under all three:

| Ladder | R32 | R16 | QF | SF | Final | Champion |
|---|---|---|---|---|---|---|
| linear | 1 | 2 | 3 | 4 | 5 | 6 |
| triangular | 1 | 3 | 6 | 10 | 15 | 21 |
| geometric | 1 | 2 | 4 | 8 | 16 | 32 |

## 9. Parameter-selection summary (zero magic numbers)

| Quantity | How set |
|---|---|
| μ_total | pooled WC goals/match 2014–2022 (cited) |
| β | calibrated to the Elo expected-score curve (least squares) |
| knockout tie split | strength-proportional closed form (model choice, documented) |
| synthetic spreads | {0.5,1,2}× empirical Elo SD (documented stress range) |
| n_draws, sims_per_draw | 25 × 2000 = 50k/cell; MC SE reported |
| master seed | 20260608 (run date), documented |
| recommendation rule | Pareto frontier + minimax-rank compromise (no thresholds) |

## 10. Key findings (anchor = real 2026 Elo field, EV-greedy)

- The three goals (high skill, low ties, balanced slots) **conflict along ladder
  convexity**. Convexity ↑ skill (Spearman 0.17→0.23→0.25 for linear→triangular→geometric)
  and ↓ ties (0.19→0.05→0.001) but ↑ slot inequity (spread 0.13→0.23→0.31) by making the
  pool a referendum on the champion pick (champion-dominance 0.52→0.79→0.999).
- **N is second-order**: within a ladder, skill differs by <0.01 across N∈{4,5,6,8}
  (≈3 MC-SE); the ladder choice dominates.
- **EV-greedy is only marginally exploitable**: a slot-1 best-responder gains <1pp of
  win probability over EV-greedy across all cells.
- **Robustness**: the skill ranking of ladders is stable across concentration levels;
  geometric's slot-inequity is concentration-driven (spread 0.05 at top-8 share 0.65 vs
  0.31 at 0.91), so it is a problem specifically for top-heavy fields like 2026's.
- **Recommendation**: triangular as the balanced default; geometric only if a pure skill /
  no-ties contest is wanted (accepting a large first-pick advantage); linear only if slot
  fairness is paramount (accepting an ~18% tie rate that needs a tiebreaker rule).

## 11. Residual risk

_(populated by the audit-remediate loop; see [docs/audits/](audits))_
