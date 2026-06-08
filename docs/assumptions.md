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
- **Goals = independent Poisson**, log-rate linear in the rating gap:
  `log λ_A = log(μ_total/2) + (β/2)(dr/400)`, symmetric for B. Group W/D/L, GD and GF all
  follow from the sampled scoreline; the third-place tie-breakers are therefore produced by
  the same model rather than bolted on. Win/draw/loss probabilities are the **Skellam**
  law of the goal difference.
  - The *independent*-Poisson model is **Maher (1982)**,
    [*Statistica Neerlandica* 36(3):109–118](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x).
    **Dixon & Coles (1997)**,
    [*JRSS-C* 46(2):265–280](https://doi.org/10.1111/1467-9876.00065), is the canonical
    reference for the supremacy/attack-defence Poisson framework and additionally introduced
    a low-score *dependence* correction (a τ adjustment for 0-0/1-0/0-1/1-1) that we
    **deliberately do not implement** — this code uses pure independent Poisson.
  - Skellam (1946),
    *J. R. Statist. Soc. A* 109(3):296, for the difference of two independent Poissons.
- **μ_total (goals/match)** is pinned to the pooled men's World Cup rate over 2014/2018/2022,
  not a free constant: 171/64 ([2014](https://en.wikipedia.org/wiki/2014_FIFA_World_Cup)) +
  169/64 ([2018](https://en.wikipedia.org/wiki/2018_FIFA_World_Cup)) +
  172/64 ([2022](https://en.wikipedia.org/wiki/2022_FIFA_World_Cup)) = 512 goals / 192
  matches ≈ 2.667.
- **β (supremacy slope)** is *calibrated*, not hand-set: chosen by least squares so the
  model's implied expected score `P(win)+0.5·P(draw)` reproduces the Elo logistic across
  the field's realised pairwise rating gaps. β is the **neutral** supremacy slope
  (calibrated on neutral gaps only, so home advantage is never folded into it). Calibration
  diagnostics (RMSE vs the Elo curve, implied even-match draw rate ≈ 0.27, within the
  historical band) are stored on the model and asserted in
  [test_strength.py](../tests/test_strength.py).
- **Knockout tie rule.** `P(A advances) = P(A win) + P(draw)·P(A win)/(P(A win)+P(B win))`
  — extra-time/shootout modelled as strength-proportional. This is a modelling choice;
  real shootouts are closer to a coin flip, so this *slightly* overstates favourite
  survival in knockouts (sensitivity not run — listed in §11 residual risk).
- **Home advantage** defaults to 0 (neutral field). The three 2026 hosts' home edge
  (Elo's conventional +100) is exposed as a parameter but off by default, since a neutral
  field is the cleaner basis for a scoring-design study. When enabled it shifts the
  effective gap `dr` at simulation time only (standard Elo treatment), leaving β neutral.

## 3. Rating sources (the pluggable input)  ([strength.py](../src/wcpool/strength.py))

- **Real Elo** ([config/ratings_elo_2026.yaml](../config/ratings_elo_2026.yaml)):
  eloratings.net snapshot via the Wikipedia data module, dated 2026-06-01. **14 of 48
  ratings are approximate placeholders** (flagged `source: approx`) for qualifiers outside
  the sourced top-50; design conclusions rest on the field's *spread* and on the synthetic
  sweep, both robust to ±50 Elo on those teams.
- **Synthetic favoritism** generates i.i.d. Gaussian ratings with a single knob — the
  standard deviation `spread`. The sweep uses spread ∈ {0.5, 1, 2}×(empirical Elo SD): a
  documented stress range (half to double the real dispersion), anchored to data rather
  than chosen ad hoc. It is a **controlled** sweep: all three synthetic configs share one
  standard-normal draw (fixed RNG `rating_stream`) and differ only by `spread`, so the
  induced top-8 title-probability concentration is monotone in the multiplier by
  construction. That concentration is *measured and reported* (it is the interpretable
  axis), not separately tuned.
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
  ladder. **Variance-seeking**: take the highest points-variance team. EV-greedy and
  variance-seeking run on the full `(ladder × N × config)` grid. **Best-response**: one
  drafter maximises its own pool-win probability (estimated on its model/EV batch) assuming
  all remaining picks proceed EV-greedily — a *one-ply greedy* best-response (not a full
  game-theoretic equilibrium), run on a matched subset purely as an exploitability probe.
- Drafter score = sum of drafted teams' terminal points under the active ladder.

## 6. Simulation regime  ([simulate.py](../src/wcpool/simulate.py))

- **Resampled-draw regime is primary**: each replicate redraws groups and re-drafts, so
  conclusions describe the *design* rather than one bracket realisation. Fixed-draw is a
  sensitivity.
- **Independent EV and eval batches.** Team EV/variance for drafting (and the
  best-responder's win-probability estimates) are computed on an EV batch; rosters are
  scored on a separate eval batch. This removes in-sample optimism from the skill-vs-luck
  estimates **and** from the exploitability probe — the best-responder optimises against its
  model belief (EV batch), never against the held-out tournaments it is graded on.
- **Shared-simulation optimisation.** The tournament outcome is independent of ladder, N and
  policy, so each draw's batch is simulated once and reused across the whole grid (the
  ladder re-weights stage outcomes; the draft re-partitions teams). Every cell is therefore
  scored on the same `n_draws × sims_per_draw` eval tournaments.
- **Sample size and standard error.** Default 25 draws × 2000 sims = 50,000 eval
  tournaments per cell (meets the ≥50k requirement). **The draw — not the sim — is the
  independent replication unit** (all sims within a draw share the groups, EVs and
  rosters), so the reported precision is a **between-draw cluster SE** (std of per-draw
  metric means / √n_draws), ≈ 0.012 on the skill correlation. A naive iid SE
  (≈ 0.0019) understates uncertainty ~4×; tightening it requires more *draws*, not more
  sims. The exploitability probe runs at a reduced budget (8 draws × 1500 sims) because
  best-response is O(n_teams) costlier per pick; its win-prob noise scale is reported with
  the result.
- **Seeding.** Master seed = 20260608 (the 2026-06-08 run date, YYYYMMDD). Independent
  streams per (config, draw, batch) via `SeedSequence(seed, spawn_key=(cfg_id, draw, batch))`;
  synthetic rating draws use a disjoint namespace `(SYNTH_SEED_NAMESPACE, rating_stream)`.
  Fully reproducible (verified byte-identical on re-run).

## 7. Metric definitions  ([metrics.py](../src/wcpool/metrics.py))

1. **Skill vs luck.** (a) *Spearman* between the fixed pre-draft roster-EV ranking and the
   per-replicate realised-score ranking, averaged over replicates (average-rank tie
   handling). (b) *Variance share* = σ²_between / (σ²_between + σ²_within): roster-EV
   variance over noise variance — an ICC-style index (1 = placing fully determined by the
   draft, 0 = pure luck). Between-component uses the independent roster EV.
2. **Slot equity.** Win probability by snake slot; ties at the top split fractionally.
   `slot_win_prob_spread` = max−min across slots (0 = perfectly equitable). A companion
   `imbalance_index` = Σ(wpᵢ−1/n)²/(1/n) is a chi-square-*like* discrepancy on
   probabilities (NOT a chi-square test statistic — no sample-size factor, no p-value).
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
| μ_total | pooled WC goals/match 2014–2022 (per-tournament sources cited in §2) |
| β | calibrated to the Elo expected-score curve, neutral gaps only (least squares) |
| knockout tie split | strength-proportional closed form (model choice, documented) |
| synthetic spreads | {0.5,1,2}× empirical Elo SD (documented stress range, controlled sweep) |
| n_draws, sims_per_draw | 25 × 2000 = 50k/cell; between-draw cluster SE reported |
| best-response probe budget | 8 draws × 1500 sims (reduced: BR is O(n_teams) costlier); noise scale reported |
| synthetic RNG namespace | `SYNTH_SEED_NAMESPACE` (arbitrary tag, documented) to keep streams disjoint |
| master seed | 20260608 (the 2026-06-08 run date), documented |
| recommendation rule | recommend the ladder (the discriminating axis) via minimax over per-objective ranks; N reported as indistinguishable under the cluster SE (no thresholds) |

## 10. Key findings (anchor = real 2026 Elo field, EV-greedy; ladder means over N)

- The three goals (high skill, low ties, balanced slots) **conflict along ladder
  convexity**. Convexity ↑ skill (Spearman 0.169→0.224→0.249 for
  linear→triangular→geometric) and ↓ ties (0.186→0.049→0.001) but ↑ slot inequity
  (spread 0.132→0.229→0.314) by making the pool a referendum on the champion pick
  (champion-dominance 0.51→0.77→0.999).
- **N is second-order and not statistically separable**: within a ladder, skill varies
  ≈0.014 across N∈{4,5,6,8} ≈ 1.2× the between-draw cluster SE (≈0.012), whereas the
  cross-ladder gap (≈0.080) is ≈7× that SE. The ladder is the decisive lever; N is a free
  choice (N=8 drafts the full 48-team field).
- **Participant count (6 vs 7 vs 8)** — [participant_sweep](tables/participant_sweep_2026-06-08.csv),
  controlled comparison (fixed `cfg_id` → identical tournaments per draw across P; 6 teams
  each, EV-greedy). The ladder ordering and recommendation are unchanged. Two real shifts:
  (i) **the first-seat advantage grows** because the fair share falls (1/6→1/8) while the
  absolute first-seat win prob is ~flat — slot-1 ÷ fair-share rises 1.9→2.3→2.6 (triangular)
  and 2.3→2.7→3.0 (geometric). (ii) **Skill rises modestly** (triangular 0.222→0.248→0.266).
  This is significant under the **paired** between-draw test the controlled design enables
  (Δ vs 6 players: triangular z≈6.7, geometric z≈5.1, linear z≈3.3 at P=8) — the naive
  independent-sample SE understates it. The driver is the **variance ratio**: per-drafter
  within-tournament-noise variance falls ~22% (more of the field owned at fixed roster size)
  while between-roster variance rises ~13–27%, so `skill_variance_share` rises
  (0.092→0.128 geometric). The Spearman rise is **not a rank-cardinality artifact** — with
  the variance ratio held fixed, cardinality alone moves ρ by <0.01 (negative for
  geometric), and the cardinality-free `skill_variance_share` rises identically. Tie rate
  rises with P only for linear (0.18→0.22); champion-dominance is flat for geometric, rises
  ~5 pp for triangular, ~flat for linear.
- **EV-greedy is *not* exploitable** by a one-ply greedy best-response: scored
  out-of-sample (best-responder decides on its model/EV batch, scored on a held-out batch),
  the slot-1 win-probability margin is uniformly slightly negative (−0.001 to −0.007),
  within the probe's noise scale (~0.011). (An earlier in-sample formulation produced
  spurious positive margins; corrected per audit round 1.)
- **Variance-seeking caveat.** In highly concentrated fields the variance-seeking policy
  can rival or beat EV-greedy on the skill metric, because there the strongest teams have
  both the highest mean *and* the highest points variance, so variance becomes a partial EV
  proxy. The headline skill comparison therefore fixes the policy (EV-greedy); the metric is
  sensitive to mean–variance coupling, not a pure measure of draft skill.
- **Robustness**: the ladder *skill ranking* (geometric > triangular > linear) is stable
  across every concentration level and the real-Elo anchor (recommended triangular is rank 2
  on skill in all configs). The slot-inequity is **concentration-driven** for every ladder
  (e.g. triangular slot-spread 0.18 → 0.40 → 0.69 as top-8 title share rises 0.71 → 0.93 →
  1.00), so it bites specifically for top-heavy fields like 2026's (top-8 ≈ 0.91).
- **Recommendation**: triangular as the balanced default; geometric only if a pure skill /
  no-ties contest is wanted (accepting a ~38% first-pick win rate); linear only if slot
  fairness is paramount (accepting an ~18% tie rate that needs a tiebreaker rule).

## 11. Residual risk

Recorded after audit round 1 (see [docs/audits/](audits) for the full trail):

- **Knockout tie rule is strength-proportional, not coin-flip, and is not sensitivity-
  tested.** Real shootouts are closer to 50/50, so the model slightly overstates favourite
  survival in knockouts. The direction of any bias on the ladder *ordering* is expected to
  be small (it would mildly compress skill differences), but this has not been quantified.
- **Skill metric sensitivity to mean–variance coupling** in concentrated fields (see §10),
  which is why the headline comparison fixes EV-greedy.
- **Annex-C third-place assignment** uses one valid bipartite matching, not FIFA's exact
  published table; aggregate metrics are argued (not formally proven) insensitive to which
  valid matching is used.
- **14 of 48 Elo ratings are approximate placeholders** (§3). The synthetic sweep and the
  real-field *spread* carry the conclusions; ±50 Elo on lower-ranked qualifiers does not
  move the ladder ordering, but the absolute real-Elo skill magnitudes are approximate.
- **Confederation draw constraints not modelled** (§4); pot constraint only.
- **Home-advantage path is unexercised** (default neutral) and lacks a dedicated test.
