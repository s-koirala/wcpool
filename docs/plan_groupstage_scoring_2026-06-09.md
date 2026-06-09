# Execution plan — group-stage (W/D) + progressive-knockout scoring extension

**Project:** wcpool (FIFA 2026 draft pool) · **Date:** 2026-06-09 · **Engine HEAD:** `5c4ef6a`
**Fixed parameters (owner):** `n_drafters = 8`, `teams_per_drafter = 6`, snake draft, **all 48 teams drafted** (full field).
**One decision deliberately deferred to the owner:** the *objective* (§5). Everything else here is determined.

Synthesised from the four verified research notes dated 2026-06-09 in [docs/research/](research) and the prior study
([recommendation_2026-06-08.md](tables/recommendation_2026-06-08.md), [report](report_draft_pool_layperson_2026-06-08.md),
[assumptions.md](assumptions.md)). Audit trail for the research: [audit_trail_2026-06-09_groupstage_research.md](audits/audit_trail_2026-06-09_groupstage_research.md).

Notation follows [ladders.py](../src/wcpool/ladders.py): `Stage` 0=GROUP … 6=CHAMPION. **Disambiguation of three "W"s** (kept distinct in code and prose): `W_pts`/`D_pts` = pool points per group win/draw (this task); `objective_W` = the payout objective `(n−1)·P1+P2` in [objective.py](../draft_advisor/objective.py); `Wcond` = the Ely–Frankel–Kamenica conditional pool-win-probability martingale (§6).

---

## 1. What we are building

Replace the prior **terminal-only** scoring (points awarded solely for the furthest knockout stage a drafted team reaches; group exit = 0) with a **two-layer** scheme:

```
team_points[i] = (1 − mix)·A(stage_i)              # knockout layer  (progressive advancement)
               +      mix ·(W_pts·wins_i + D_pts·draws_i)   # group layer (win/draw, no goals/losses)
```

- `wins_i, draws_i ∈ {0..3}` — team i's group win/draw counts (each team plays 3 group matches).
- `A(·)` — a length-7 **advancement shape** (a terminal ladder from [ladders.py](../src/wcpool/ladders.py)); `A(GROUP)=0`.
- `mix ∈ [0,1]` — the single cross-layer mixing knob (§4). `mix=0` ⇒ pure terminal ladder = the validated prior study.

The end product is a layperson recommendation (format of [report_draft_pool_layperson_2026-06-08.md](report_draft_pool_layperson_2026-06-08.md)) stating **concrete integer points** for a group win, a group draw, and each knockout-stage advancement. Those integers are an *output of the sweep conditioned on the §5 objective* — they cannot be pinned a priori (§14).

---

## 2. Scoring model — formal definition and the terminal≡progressive equivalence

**Group layer.** `W_pts·wins + D_pts·draws`. The pool's `(W_pts, D_pts)` are a design choice **independent** of the FIFA 3/1/0 the tournament uses internally to rank groups ([tournament.py](../src/wcpool/tournament.py) L44; [Doc 1 §1](research/research_points_system_incentives_2026-06-09.md)). No points for goals or losses (per spec).

**Knockout layer — "progressive points for advancing" is a terminal ladder.** Awarding a per-advance **increment** `δ(r)` each time a team survives into knockout round `r` and banking the running sum gives a team reaching furthest stage `s` a total `A_cum(s) = Σ_{r=R32..s} δ(r) = cumsum(δ)[s]`. A terminal ladder **is** the cumulative sum of an increment schedule, and the increment schedule is its successive difference (`δ = diff(A)`) — a bijection. The three existing shapes ([ladders.py](../src/wcpool/ladders.py) L68-70):

| Shape | Terminal `A` (R32…Champ) | Per-advance increment `δ = diff(A)` |
|---|---|---|
| linear | 1, 2, 3, 4, 5, 6 | 1, 1, 1, 1, 1, 1 |
| triangular | 1, 3, 6, 10, 15, 21 | **1, 2, 3, 4, 5, 6** |
| geometric | 1, 2, 4, 8, 16, 32 | 1, 1, 2, 4, 8, 16 |

So "bank `r` points for surviving the r-th knockout round" reproduces the **triangular** ladder exactly. **Decision:** score via the terminal lookup `A(stage_i)` (cheap, vectorised, `ladders.points_for_stages`), since outcomes are exogenous and the two forms are point-for-point identical; the banked/running form is needed only for the stage-trajectory engagement metrics (§6, `running_scores`). All three shapes have `δ ≥ 0`, so `A` is non-decreasing in `s` — deeper runs never lower a team's score (required for the EV-rank metrics to be coherent; tested in §10). The layperson doc states **both** representations ("you bank δ(r) per round survived" is the intuitive phrasing; the terminal table is the audit form).

**Per-drafter score** is the existing `pool_scores(team_points, rosters)` ([metrics.py](../src/wcpool/metrics.py) L24) on the two-layer points — no change to the draft or metric harness signatures.

---

## 3. Identifiability — why the free parameters reduce to two ratios + one shape

**With equal roster sizes — which the snake draft guarantees at any `teams_per_drafter` — the rank-based metrics are invariant to positive affine transforms of the team-point vector.** Under `p ↦ a·p + c·1` (`a>0`), every drafter's score maps identically `S_d ↦ a·S_d + (teams_per_drafter)·c` (the offset is common *because rosters are equal-sized*, not because the field is fully drafted):

- **Spearman EV-vs-placing** ([metrics.py](../src/wcpool/metrics.py) `spearman_per_sim`) — rank-based, invariant to monotone maps. Invariant.
- **skill_variance_share** (L94-116) — ratio of variances; `a²` and `+6c` cancel. Invariant.
- **slot win-prob / slot equity / champion-dominance / tie rate** — all key off `argmax`/equality of `S_d`. Invariant. **Sole exception:** the *integer* tie rate depends on the point **lattice** (how often two drafters hit the identical integer total), so absolute integer granularity matters for `top_tie_rate` even though ratios suffice for everything else ([Doc 1 §3d](research/research_points_system_incentives_2026-06-09.md)).
- **Engagement metrics (§6)** — `pool_suspense`/`pool_surprise` build on win-probability orderings (invariant); `stage_variance_share` is a variance ratio (invariant); `alive_fraction` is a feasibility inequality preserved by `a>0, +6c`. Invariant.

**Consequence — the only identified, decision-relevant degrees of freedom are:**
1. **`D_pts/W_pts`** — the draw/win ratio (equivalently the draw discount `1 − D/W`).
2. **The advancement SHAPE** (linear/triangular/geometric) — a shape, not a scale.
3. **The mixing weight** between layers — *not* absorbed by global affine invariance because it reweights the two layers non-uniformly. Parameterised by `mix` (§4).

We therefore sweep `(D/W) × shape × mix` only; `W_pts` and a separate knockout scale `K` are **not** swept independently (that would re-scan a 1-D identified manifold). The integer granularity is fixed downstream per recommended cell (§14), not as a primary factor. This reduction is a derived invariance (from equal roster sizes), not a convenience cut. Full draft additionally makes `champion_undrafted_rate = 0` (every team owned), cleanly defining champion-dominance — a benefit distinct from the affine invariance.

---

## 4. Mixing parameter: code knob `mix`, reported/targeted as group-share `γ`

`mix` is the implementation knob (convex, bounded `[0,1]`, makes the §10 identities exact). But `mix` is not directly interpretable, so the swept and **reported** quantity is the **expected group-layer share of total points** on the real-2026-Elo field:

```
γ(mix) = mix·E[Σ group_layer] / ( mix·E[Σ group_layer] + (1−mix)·E[Σ A(stage)] ),
```

a dimensionless, scale-free quantity in `[0,1)`, monotone in `mix`. Both expectations are closed-form under the engine's outcome model (E[wins], E[draws] over 72 group matches from the field's Skellam W/D/L; E[Σ A(stage)] from per-team stage probabilities) — **no simulation needed to solve `mix` for a target `γ`**. "γ = the group stage is worth this fraction of the points on the board" is the phrase the layperson doc uses.

**Swept γ grid (justified; zero magic numbers):**
- **`γ = 0` (mix=0) — mandatory lower anchor.** Pure terminal ladder = the prior study identically; the **validation gate** (§8.4). The degenerate boundary that makes this extension a strict superset of the validated baseline.
- **`γ = 0.5` — derived upper landmark.** The unique *equal-purse* point (expected group purse = expected knockout purse); beyond it the group stage dominates, contradicting the design intent that the knockout carries the title. A structural equality, not a chosen cap.
- **`γ_match` — derived commensurability landmark.** The `γ` at the `mix*` solving `mix*·3·W_pts = (1−mix*)·A(R32)` — i.e. a team winning all three group games earns the same as a team reaching the Round of 32 ([Doc 1 §4](research/research_points_system_incentives_2026-06-09.md)). Computed per `(D/W, shape)` cell and added as an explicit grid point (the most decision-relevant interior value).
- **Interior spacing denser toward 0** (where metric sensitivity is expected strongly nonlinear — any group mass breaks the champion-referendum regime): `γ ∈ {0, 0.05, 0.12, 0.25, 0.50}` (gaps widen monotonically toward the equal-purse end) plus `γ_match`. Smoothness/monotonicity of metric trajectories between adjacent γ is checked; refine locally if a metric is non-monotone in γ (the spacing is justified by expected nonlinearity, not convenience).

**Total grid:** `3 (D/W) × 3 (shape) × {γ>0 levels}` + **3 anchor cells** (at γ=0 the group layer — hence D/W — is absent, so the 3×3 anchor slice collapses to one cell per shape = the prior study). Net ≈ **39 distinct cells**. Fixed: `n_drafters=8`, `teams_per_drafter=6`, resampled regime (primary), EV-greedy headline policy (exploitability re-test policies in §7.3). (γ=1 is excluded by construction — it requires mix=1, which deletes the knockout layer; the swept maximum is γ=0.5.)

**D/W candidates — exactly the three from [Doc 1](research/research_points_system_incentives_2026-06-09.md), no others:**

| `D/W` | (W_pts, D_pts) | Rationale | Constant-sum (`2D=W`)? |
|---|---|---|---|
| **1:0** wins-only | (1, 0) | Max draw discount; max group skill signal; max aggregate-purse noise | below |
| **2:1** (≡1:0.5) | (2, 1) | **On the constant-sum line `D=W/2`**: every match distributes exactly `W_pts` ⇒ group total fixed at `72·W_pts` ⇒ pool→constant-sum. Integer representative of the 1:0.5 class | **on** |
| **3:1** FIFA-mirror | (3, 1) | Mirrors the on-screen standings; familiar to laypeople; sits *below* the line | below |

`1:0.5` is omitted as affine-redundant with `2:1` (identical on every metric but integer granularity; `2:1` keeps integers). The **shapes** reuse [ladders.py](../src/wcpool/ladders.py) verbatim (no new shape without empirical/derivational justification).

---

## 5. Objective — the one decision for the owner

The metrics trade off in **opposite directions in γ**: adding group mass (`γ↑`) *lowers* skill (3 group games are a small, noisy, weakly-strength-correlated sample — [Doc 4 §1](research/research_engagement_competitive_balance_2026-06-09.md)) while *raising* engagement (it breaks the champion-referendum, keeps drafters alive and the standings in suspense through the group phase — [Doc 4 §3-4](research/research_engagement_competitive_balance_2026-06-09.md); [Doc 3 §4](research/research_pool_scoring_design_2026-06-09.md) "decorrelator"). The prior study optimised **skill alone**; the new feature exists *because engagement is now in scope*. Three candidate objectives:

1. **Skill-maximisation** (prior goal). Extend the minimax-over-per-objective-ranks rule ([assumptions §9](assumptions.md)) to the new grid. *Predicted outcome:* pushes `γ→0`, reproduces triangular, **rejects the group layer as skill-dilutive** — i.e. choosing (i) makes the extension moot.
2. **Engagement-primary.** Maximise the group-phase share of `pool_suspense` ([Doc 4 §4](research/research_engagement_competitive_balance_2026-06-09.md)) subject to a skill floor and tie ceiling.
3. **Engagement-constrained skill (recommended default).** `max skill s.t. group-phase suspense share ≥ φ` (with the usual tie/equity reporting). Report the **skill-vs-engagement efficient frontier over γ** and pick the knee. Nests both extremes: `φ=0` recovers (i); a binding `φ` forces a positive γ. It states honestly that the group layer *costs* skill and asks how much engagement is bought per unit skill surrendered — the exact trade-off Docs 3-4 identify, and the minimal auditable extension of the already-accepted methodology.

**Floors set from data, not fiat (a flagged sub-decision):**
- *Engagement floor `φ`* — anchored to [Doc 4 F6](research/research_engagement_competitive_balance_2026-06-09.md)'s benchmark methodology: the group-phase suspense share attainable in the engine's own perfectly-balanced field (`make_synthetic_config(spread≈0)`) is the achievable ceiling; the status-quo (γ=0) is the floor. **The deliverable is the frontier plus these two data-derived anchors**, so the owner reads off the φ that costs an acceptable amount of skill; no single φ is asserted here.
- *Skill floor / tie ceiling* (for objective ii) — the γ=0 **linear-ladder** skill (0.169) and tie rate (0.186) from the [prior recommendation](tables/recommendation_2026-06-08.md): "no worse than the flattest already-shipped option." Read off the baseline, not chosen.

**This is the single non-derivable call** — a normative choice about whether the pool is a skill contest or an engagement product. The plan determines the grid, metrics, sensitivities, statistics, and selection *rule* completely; the owner supplies the objective.

---

## 6. Metric suite

**Existing five** ([metrics.py](../src/wcpool/metrics.py)), recomputed on the two-layer `team_points` (only the points array changes): (1) skill — `spearman_ev_vs_placing` + `skill_variance_share`; (2) slot equity — `slot_win_probability` + `slot_equity_imbalance`; (3) ties — `top_tie_rate` + `winning_score_summary`; (4) `champion_dominance` (champion-undrafted rate ≈ 0 at full draft — a clean side-benefit of 8×6); (5) robustness across configs (§7.2).

**Engagement metrics (added, [Doc 4](research/research_engagement_competitive_balance_2026-06-09.md)).** All are post-processors over a per-replicate **stage trajectory** + the group W/D counts. The trajectory needs **no new tournament array** — `simulate_tournament`'s `stages` already encodes "alive after round k" as `stages ≥ k`. The one new object is `running_scores(stages, wins, draws, scheme, rosters) → R[s, d, b]` `(n_sims, n_drafters, 7)` = drafter d's cumulative score once results through stage boundary `b` are known (group layer realised at b=0; knockout via `A(min(stage_i, b))`).

- **`stage_variance_share(R)`** — law-of-total-variance split of final pool score into group-phase vs knockout-phase ([Doc 4 §3(iii)](research/research_engagement_competitive_balance_2026-06-09.md)); reuses the `skill_variance_share` pattern recoloured by stage. Threshold-free. The cheap, robust companion — the direct answer to "does the group layer matter?".
- **`pool_suspense` / `pool_surprise`** (EFK variance kernel; headline) — treat the conditional pool-win-probability vector `Wcond[s,·,b]` as the EFK belief martingale over the 6 stage boundaries: `pool_surprise = Σ_b ‖Wcond_b − Wcond_{b−1}‖²` (squared-norm variant, [Doc 4 F6](research/research_engagement_competitive_balance_2026-06-09.md) fidelity note), `pool_suspense = Σ_b E[‖W̃_{b+1} − Wcond_b‖² | state_b]`. Headline number = **group-phase share** of suspense + the **paired Δ vs the γ=0 ladder** + a **benchmark range** from the Elo-equal field. Report suspense and surprise **separately, never collapsed** ([Doc 4 §5](research/research_engagement_competitive_balance_2026-06-09.md)). Threshold-free.
- **`alive_fraction(R, stages, scheme)`** (optional, layperson-facing) — fraction of (replicate, drafter) mathematically able to win entering the knockouts: `R[s,d,GROUP] + maxgain[s,d] ≥` the leader's guaranteed floor, `maxgain` a pure feasibility quantity from the ladder's max-remaining points. Threshold-free.

**`Wcond` estimator (the top implementation risk — [Doc 4 §4-5](research/research_engagement_competitive_balance_2026-06-09.md)).** Production estimator = **k-NN regression of the terminal fractional 1st-place credit on the running-score vector `R[·,·,b]`** across the existing replicates (no re-simulation), implemented on **numpy/scipy only** (`scipy.spatial.cKDTree` + a hand-rolled leave-one-out loop) — **scikit-learn is not a project dependency and must not be added without a `==` pin + `uv.lock` regeneration**. `k` is chosen by **deterministic leave-one-out CV (no fold shuffle, no new RNG)**; if a shuffled k-fold is ever substituted, its fold assignment must draw from a named seed slot (§11) and be logged, and a determinism unit test must assert identical estimator output across two runs at the same seed. **Bias is one-sided:** k-NN smoothing attenuates the variance of belief increments, biasing `pool_suspense`/`pool_surprise` **downward**; since CV-for-prediction-error is not CV-for-increment-variance, select `k` (or bias-correct) against the suspense/surprise discrepancy on the nested-sim subset, not only against terminal-credit prediction error. Nested simulation is exact but `O(stages×inner_sims)` per replicate and incompatible with the single-pass budget, so it is the **validation ground truth only** (`tournament.nested_conditional_win_prob`). **Ground-truth construction (two corrections over the naive nested sim):** (a) only the **CHAMPION** boundary is set to the realised one-hot terminal credit (the resolved state determines the champion); the **FINAL** boundary does *not* reveal the champion (both finalists sit at `ko[FINAL]`; the champion bump enters only at CHAMPION), so the final is Monte-Carlo'd between the two realised finalists (`round_winners[3]`) from `start_stage=Stage.FINAL` — exactly one replayed round — giving the genuine blended belief rather than forcing FINAL onto the one-hot credit. (b) The squared-increment estimator `E‖μ̂_b−μ_{b−1}‖²` is inflated by `Var(μ̂_b)∼1/n_inner`; this is bias-corrected by splitting each boundary's inner sim into two independent halves `μ̂_b^A,μ̂_b^B` and forming the unbiased cross-term `⟨μ̂_b^A−μ̂_{b−1}^A, μ̂_b^B−μ̂_{b−1}^B⟩` (the plain two-half mean is reported as the UPPER estimate). Run at `n_inner ≥ 800`. **Gate (the sole guard against the smoothing bias):** run nested simulation on a ≥1,000-replicate subset, PER MIX, and adopt the surrogate where the `pool_suspense`/`pool_surprise` increment agrees with the bias-corrected ground truth within a **confirmatory equivalence margin δ derived from the combined Monte-Carlo SE budget** (no hand-picked threshold; one-sided non-inferiority against the documented downward smoothing bias, §8.3). The conclusion is per-mix and honest: agreement holds at the headline low-mix cell (triangular, `mix≈0.15`) — the encoded gate — and the surrogate **rejects** at high mix (`mix≈0.5`, recorded as its documented reliability boundary), not a blanket pass. **Necessary-but-not-sufficient sanity check:** the martingale property `E[Wcond_{b+1}|b]=Wcond_b` ⇒ `mean_s Wcond[:,d,b]` flat across `b`, equal to the unconditional `win_probability` — this catches gross mis-normalisation/leakage but is passed by a degenerate zero-suspense smoother, so it does **not** validate conditional calibration. **Documented fallback:** if the surrogate fails the gate, run nested-sim at reduced budget for the suspense headline only; the estimator-free `stage_variance_share` + `alive_fraction` carry the engagement story regardless, so the project is not gated on the suspense estimator.

Secondary: HHI of final pool-score shares as a concentration diagnostic (largely restates `champion_dominance`). Noll-Scully/Gini **not** headline (small-N bias regime — [Doc 4 F3-F4](research/research_engagement_competitive_balance_2026-06-09.md)).

---

## 7. Sensitivities (each tests a stated hypothesis so the run confirms, not discovers)

**7.1 Draw model — matched-ρ Dixon-Coles (confirmatory).** Draws now score, so `P(draw)` is scoring-relevant for the first time. Preferred, self-calibrating ([Doc 2 F-quant-3B](research/research_draw_lowscore_modeling_2026-06-09.md)): solve the single DC τ ρ that matches the engine's field-averaged draw rate to the empirical pooled WC group rate **0.1944**; since the independent engine is already at **0.1975**, matched ρ ≈ 0 — no deficit to correct. Bracket with the literature band ρ ∈ {−0.05, −0.13, −0.18} as a worst-case envelope. **Hypothesis ([Doc 2 F-quant-4](research/research_draw_lowscore_modeling_2026-06-09.md)):** the recommendation is invariant across the entire ρ envelope (worst case ≈ 1 extra draw per tournament, ≪ skill cluster SE ≈ 0.012; the shape/γ ordering is knockout-driven, untouched by τ). Run only on the chosen frontier's neighbourhood. Guard: assert all τ-cells' draw probs > 0.

**7.2 Robustness across concentration + real Elo** (reuse [simulate.py](../src/wcpool/simulate.py) synthetic sweep, spread ∈ {0.5,1,2}× Elo SD). **Hypothesis:** the shape skill-ordering (geometric > triangular > linear) is stable at every γ; slot-inequity remains concentration-driven; and the *engagement* benefit of γ↑ is **larger** in concentrated fields (where the champion-referendum is strongest, so the decorrelating group layer has more to undo).

**7.3 Exploitability re-test** (best-response + variance-seeking under the new scoring; [draft.py](../src/wcpool/draft.py); reduced probe budget). **Hypothesis ([Doc 3 §4-5](research/research_pool_scoring_design_2026-06-09.md)):** EV-greedy stays at-most-marginally exploitable — small N (8 drafters) sits at the favorites-optimal end of Clair-Letscher eq. 3.1, and the **snake draft voids crowd-avoidance** (each team owned exactly once), so the duplication that drives contrarian value is structurally absent; the dense group points further decorrelate scores, so margins should be ≤ the prior γ=0 margins (within the ≈0.011 noise scale). Note the variance-seeking caveat ([assumptions §10-11](assumptions.md)).

---

## 8. Statistical plan

**8.1 Cluster SE.** The **draw, not the sim, is the independent unit** — all headline metrics carry a between-draw cluster SE (`_cluster_se` in [simulate.py](../src/wcpool/simulate.py); naive iid SE understates ≈4×). The new engagement metrics are accumulated per-draw (extend `CellAccumulator` with `draw_stage_share`, `draw_suspense`, `draw_surprise`, `draw_alive`, mirroring `draw_spearman`/`draw_tie`/`draw_slot_spread`).

**8.2 Budget.** Baseline 25 draws × 2000 sims = **50k eval tournaments/cell** ([assumptions §6](assumptions.md)). If the recommendation hinges on a γ-to-γ gap **within** the cluster SE (ambiguous frontier knee), **raise the number of DRAWS, not sims** (cluster SE ∝ 1/√n_draws; more sims don't shrink it) — sized sequentially to the observed gap, not a fixed inflation. The nested-sim `Wcond` validation runs at its own ≥1,000-replicate subset.

**8.3 Multiple testing.** The 39-cell grid is design *enumeration*, not itself a test family. But the recommendation's load-bearing **pairwise inferential claims** ("shape X > Y on skill at the chosen γ", "γ meets the engagement floor", "the chosen cell is non-exploitable") are tests and must be **registered via the `multipletest-gate` skill** in `config/multipletest_family.yaml` *before* the headline run, with the project's clustered correction: Ledoit-Wolf-style studentized paired-draw bootstrap for the skill head-to-heads, Benjamini-Hochberg FDR across the family of cell-level claims. The §8.4 cross-check anchor is registered as a confirmatory *equivalence* test.

**8.4 Validation anchor (a gate — blocks the run if it fails).** The config-matched baseline for the **8-drafter × 6-team** study is the **8-participant rows of [participant_sweep_2026-06-08.csv](tables/participant_sweep_2026-06-08.csv)** — that file is the only prior artifact at *this exact configuration* (`n_drafters = 8`, `teams_per_drafter = 6`, EV-greedy, real Elo, resampled, seed `20260608`, 25 draws × 2000 sims). **The anchor values are:**

| shape | skill (`spearman_mean`) ± cluster SE | `top_tie_rate` | `slot_win_prob_spread` (seat) | `p_champion_holder_wins` | `champion_undrafted_rate` |
|---|---|---|---|---|---|
| linear | 0.196 ± 0.009 | 0.2153 | 0.151 | 0.479 | 0.0 |
| triangular | 0.266 ± 0.010 | 0.0501 | 0.279 | 0.812 | 0.0 |
| geometric | 0.278 ± 0.011 | 0.0002 | 0.338 | 1.000 | 0.0 |

**The gate has two parts ([test_validation_gate.py](../tests/test_validation_gate.py)):**

- **(i) Exact self-consistency — new mix=0 ≡ pre-refactor.** Run the refactored `run_strength_config` with `schemes = [ScoringScheme(l, 1.0, 0.0, 0.0) for l in ladders]` at the same config/seed/budget as a captured pre-refactor golden and require every existing-metric value to reproduce to **floating-point tolerance**. With the RNG path unchanged (§9e preserves `SeedSequence(seed, spawn_key=(cfg_id, draw, batch))`) and the `mix=0` short-circuit returning the bare ladder lookup `ladder[stages]`, the `stages` arrays are byte-identical and the derived metrics match **exactly** (realised: max abs diff `0.0` over all metrics, all three shapes). This proves the refactor did not perturb the validated baseline. (Budget-independent: it rests on byte-identical `stages`, so a small smoke budget suffices for the assertion.)
- **(ii) Participant-sweep cross-check.** At the spec budget (25 draws × 2000 sims) the new mix=0 pipeline's skill / tie / seat-spread / champ-holder reproduce the participant-sweep 8-participant rows above. Reported in cluster-SE units; hard-fail only outside ~3 cluster SE on the SE'd skill head; `champion_undrafted_rate` must be exactly 0 (full 8×6 field). Realised: all three skill z-scores |z| < 0.05 cluster SE (effectively exact, because the participant sweep calls this same function at the same seed/regime).

**Not the anchor.** The full-grid [recommendation_2026-06-08.md](tables/recommendation_2026-06-08.md) `teams_per_drafter=8` rows (triangular 0.2184, geometric 0.2462, linear 0.1515) are the **6-drafter × 8-team** configuration — a *different* study (8 teams each over 6 participants), not the 8×6 design fixed in this plan. Citing those as the validation anchor would compare against the wrong configuration; the participant-sweep 8-participant rows are the correct config-matched target. The rest of the analysis is not trusted until both gate parts pass.

**8.5 Pre-registration & reproducibility.** Freeze the design + the §5 objective choice + the floors via `pre-register-hypothesis` before the headline run. Every run emits the 13-field ReproLog (§11). Pre-register the suspense-share estimator and its cluster SE, and the frontier knee rule (e.g. maximum curvature of the skill-vs-suspense-share frontier), so φ selection is reproducible once the owner fixes the objective class.

---

## 9. Implementation (minimal-diff, idiomatic; references real files/lines)

**9a. [tournament.py](../src/wcpool/tournament.py) — emit group W/D counts.** `simulate_group_stage` (L163-186) already derives `a_win`/`b_win`/`draw` (L177-181); add `wins`/`draws` `int32 (n_sims,48)` accumulators and return a 5-tuple `(pts, gd, gf, wins, draws)`. Two internal callers (`simulate_tournament`, [test_tournament.py](../tests/test_tournament.py) L50) update their unpack (`pts,gd,gf,_,_`). `simulate_tournament` (L285-316) gains an opt-in flag, preserving its `(n_sims,48)` contract for all current callers:
```python
def simulate_tournament(model, groups, n_sims, rng, *, return_group_results=False):
    pts, gd, gf, wins, draws = simulate_group_stage(...)
    ...                                   # unchanged knockout logic -> stages
    return (stages, {"wins": wins, "draws": draws}) if return_group_results else stages
```
Same `rng` stream ⇒ reproducibility unchanged. No change to `KNOCKOUT_TREE`, third-place logic, or stage encoding. "Alive after round k" = `stages ≥ k` directly.

**9b. New module `src/wcpool/scoring.py`** (keep [ladders.py](../src/wcpool/ladders.py) untouched; `scoring.py` imports from it — mirrors how `metrics.py` consumes arrays):
```python
@dataclass(frozen=True)
class ScoringScheme:
    knockout_ladder: str        # name in ladders.LADDERS (the progressive shape)
    w_pts: float; d_pts: float  # group win/draw points (no goals/losses)
    mix: float = 0.0            # convex weight on the group layer; mix=0 -> prior study
    def knockout_vector(self): return get_ladder(self.knockout_ladder)   # length-7

def team_points(stages, group, scheme):     # (n_sims, n_teams)
    ko  = scheme.knockout_vector()[stages]                       # == ladders.points_for_stages
    grp = scheme.w_pts*group["wins"] + scheme.d_pts*group["draws"]
    return (1.0 - scheme.mix)*ko + scheme.mix*grp

GROUP_POINT_SCHEMES = {"fifa_3_1": (3.,1.), "linear_2_1": (2.,1.), "wins_only_1_0": (1.,0.)}  # the 3 D/W candidates (§4)
```
Short-circuit `mix==0` to return the bare `ko` lookup so the legacy path is bit-identical to `ladder[stages]` (asserted by §10 test 2). `mix` is solved per cell to hit the target γ / the `γ_match` landmark (§4); reported alongside its resolved value.

**9c. `running_scores(stages, group, scheme, rosters) → (n_sims, n_drafters, 7)`** (in `scoring.py`): group component constant across boundaries; knockout component at boundary `b` = `ko_vec[min(stages,b)]` summed per roster; combined with `(1-mix)`/`mix`. Invariant (tested): `running[...,CHAMPION] == pool_scores(team_points(...), rosters)`. No re-simulation.

**9d. [metrics.py](../src/wcpool/metrics.py) additions** (pure array post-processors, consistent with the module contract): `stage_variance_share`, `alive_fraction`, `pool_suspense`/`pool_surprise`, and the CV-selected k-NN `Wcond` estimator + martingale unit test (§6).

**9e. [simulate.py](../src/wcpool/simulate.py) — thread params, preserve the shared-simulation optimisation** (the single hot edit). A *cell* becomes `(scheme, n, policy)`. Pass `schemes: list[ScoringScheme]`; keep a back-compat shim mapping legacy `ladders=[...]` strings to `ScoringScheme(name, w, d, mix=0.0)` so the prior study reproduces bit-for-bit. Capture group results once per batch (`...return_group_results=True`); replace `points_ev = ladder[stages_ev]` (L207) with `scoring.team_points(stages_ev, grp_ev, scheme)` (likewise eval). `team_ev`, `team_var`, `_run_draft`, `pool_scores`, champion accumulation, per-draw cluster-SE arrays, the EV/eval split, and the `SeedSequence(seed, spawn_key=(cfg_id,draw,batch))` scheme are **unchanged**. `_finalize` keeps existing keys, renames `ladder`→carries `knockout_ladder` + adds `w_pts`/`d_pts`/`mix` columns (so existing plot filters still work), and accumulates the engagement metrics from `running_scores` computed inside the cell loop.

**9f. [draft_advisor](../draft_advisor) — score the live advisor under the new scheme.** Change is localised to **board construction** ([board.py](../draft_advisor/board.py) L120-151): build with `return_group_results=True` and `points = scoring.team_points(...)` under the recommended `ScoringScheme`; extend the `Board` dataclass + cache meta/key + repro record (L159-195) with the scheme fields so a stale cache rebuilds (the `config_sha256` staleness pattern at L257 is the template). **[objective.py](../draft_advisor/objective.py) needs no change** — `objective_W=(n−1)P1+P2` is scoring-agnostic (acts on `pool_scores`). **[advisor.py](../draft_advisor/advisor.py)** needs no logic change (routes through `board.points`/`board.team_ev`); update the docstring's "recommended ladder" reference. Expose scheme selection in [cli.py](../draft_advisor/cli.py) as `--ladder` is today.

---

## 10. Test plan (new `tests/test_scoring.py`, mirrors [test_ladders.py](../tests/test_ladders.py)/[test_metrics.py](../tests/test_metrics.py))

1. **Constant-sum identity:** with `2·D_pts = W_pts` (e.g. (2,1)), the group-layer total over all 48 teams = `72·W_pts` in *every* replicate, independent of draw count; off-line cases (3:1, 1:0) vary with draws (regression guard on the algebra).
2. **`mix=0` reproduces terminal ladders exactly:** `team_points(stages, group, ScoringScheme(ladder,w,d,0.0)) == ladders.points_for_stages(stages, get_ladder(ladder))` element-wise (`np.array_equal`) — the bit-for-bit back-compat anchor.
3. **Monotonicity:** for fixed group counts and any `mix∈[0,1]`, `team_points` non-decreasing in `stages` (all ladders non-decreasing).
4. **`running_scores` reconciliation:** terminal slice == `pool_scores(team_points,...)`; cumulative (non-decreasing in `b`).
5. **`Wcond` validation gate + martingale unit test** (§6) — the gate for the suspense headline. Runs at the spec subset size (≥1000 replicates, `n_inner≥800`) against the bias-corrected nested-sim ground truth, PER MIX: encodes the headline low-mix (`mix≈0.15`) agreement within the SE-derived equivalence margin δ as the assertion, and records the high-mix (`mix≈0.5`) rejection as the documented reliability boundary. Asserts the GT's CHAMPION boundary is exact, its FINAL boundary is a valid simplex satisfying the martingale-mean property but NOT equal to the one-hot terminal credit (the FINAL-blend correction), and that the bias-corrected increment sits below the naive plug-in.
6. **Integration test** mirroring [test_integration.py](../tests/test_integration.py): run `run_strength_config` with a couple of schemes, `n_drafters=8`, small budget; assert grid cell count, bounded metrics (incl. `0≤group_share≤1`, `0≤alive_fraction≤1`), champion-undrafted = 0 (full field), and a `mix=0` cell matches `results_main_2026-06-08.csv` within MC noise.

Run: `uv run pytest -q`; lint `uv run ruff check`. Parallel assertion in [draft_advisor/tests/test_board.py](../draft_advisor/tests/test_board.py) that a scheme-built board's `points` match `scoring.team_points`.

---

## 11. Reproducibility (reuse existing infra; no new machinery)

- **Design runs** (`run_experiment.py`): the wired `emit_repro_log` (13-field: git HEAD, pip-freeze SHA-256, dataset checksum, RNG seed, `config_resolved_sha256`, model hash) already hashes `grid_meta`; add the scoring axes (`GROUP_POINT_SCHEMES`, shapes, mix/γ targets) to `grid_meta` so they are fingerprinted. Logs → `logs/reproducibility/`. Pass `model_hash=git_head` in `run_experiment.py`'s `emit_repro_log` call for parity with `board.py` (currently null on the design path).
- **Advisor board**: `_write_repro_log` (board.py L159-195) gains the scheme fields; atomic-publish discipline retained.
- **Seeding**: reuse `SeedSequence(seed, spawn_key=(cfg_id,draw,batch))` and the synthetic disjoint namespace; engagement metrics ride the existing eval batch (no new stream) except the nested-sim spike, whose inner RNG draws from the **5-element** slot `SeedSequence(master_seed, spawn_key=(cfg_id, draw, 3, replicate_idx, boundary))` — reserving batch index 3, disjoint from batches {0,1,2} and the synthetic namespace 7 — recorded in `grid_meta`. **The boundary axis is required**: each of the (up to) six non-terminal boundaries of a given replicate replays an *independent* knockout suffix, so each needs its own stream; a 4-element slot would reuse one stream across boundaries and correlate the per-boundary belief increments. The two independent inner-sim halves used for the finite-`n_inner` bias correction (§6) are partitioned *within* this single per-(replicate, boundary) stream — the replay draws each inner replicate independently — so no separate half axis is needed and the slot stays 5-element. The production entry point (`tournament.nested_conditional_win_prob`) threads the real `cfg_id`/`draw` into the slot rather than hardcoding `0,0`; the §6 validation test may pass `cfg_id = draw = 0`. The **new group-scoring grid uses master seed `20260609`** (run date, per convention); the §8.4 validation anchor deliberately **reuses `20260608`**. Record both seeds in the ReproLog.
- Invoke `emit-repro-log` at the start of the sweep; `pre-register-hypothesis` to freeze the design (§8.5).

---

## 12. Artifact pipeline (naming `{type}_{description}_2026-06-09.{csv,png,md}`; existing subfolders)

- `scripts/run_experiment.py` → `docs/tables/results_groupscoring_2026-06-09.csv` (grid: shape × D/W × γ/mix × N × policy × config); `_quick` smoke tag preserved so smoke runs never clobber the 50k deliverable; exploitability probe takes `schemes` instead of `ladders`.
- `scripts/make_plots.py` → `figure_engagement_suspense_2026-06-09.png` (suspense / group-phase share vs γ, with Elo-equal benchmark band), `figure_stage_variance_share_2026-06-09.png` (group-vs-knockout variance, the skill-engagement trade-off), `figure_wd_tradeoff_2026-06-09.png` (skill/tie/equity by D/W, constant-sum vs not); `results_summary_2026-06-09.csv` + `recommendation_2026-06-09.md` extend the minimax-over-ranks logic to recommend `(shape, D/W, γ)` and surface the constant-sum finding.
- `scripts/participant_sweep.py` re-points at the recommended scheme (8-participant headline structurally unchanged).
- **Layperson report:** new `docs/report_draft_pool_layperson_2026-06-09.md` (do not edit the dated 2026-06-08 deliverable in place) with a "points during the group stage" section (W:D in plain terms), the "everyone stays in it" engagement framing, the realized-ladder table in **both** representations, the exogenous tiebreaker recommendation ("most teams reaching the Final"), and the AI-assistance + repro-log statement. Render via the unchanged `build_report_html.py`/`build_report_pdf.py` (point `DEFAULT_MD` or pass the path).
- **Provenance binding (new artifacts).** The prior CSVs carry no seed/git/repro columns; add a provenance column (ReproLog `run_id` or `git_head`+seed) or a sidecar `results_groupscoring_2026-06-09.repro.json` to the new result CSVs; have `make_plots.py` and the report builder emit/append a ReproLog (git HEAD + input-CSV SHA-256) so figures/report trace to their source table; parametrise `participant_sweep.py`'s seed (argparse, default = run-date seed) and add the `_quick` clobber-guard it currently lacks.

---

## 13. Sequencing & risk (build validation anchors before the expensive sweep)

| # | Task | Risk |
|---|------|------|
| 1 | tournament.py group W/D emission + flag (§9a) | Low (2-line ripple; tests stay green) |
| 2 | scoring.py `team_points` + `running_scores` + identities 1-4 (§10) | Low, high value — the correctness anchor |
| 3 | **`Wcond` estimator + validation spike** (§6, §10.5) | **TOP RISK** — surrogate smoothing bias untested; mitigated by the nested-sim gate + martingale test + documented fallback; do not gate the project on it |
| 4 | metrics.py additions (§9d) | Low for stage-variance/alive; medium for suspense (depends on 3) |
| 5 | simulate.py threading (§9e) | Most invasive but mechanical; de-risked by the `mix=0` reproduction (integration test 6) |
| 6 | full sweep + ReproLog (§11-12) | Compute, not correctness |
| 7 | draft_advisor board rewire (§9f) | Low-medium (cache-key/meta the only fiddly part) |
| 8 | make_plots + layperson report + html/pdf (§12) | Low (presentation) |

Run the `audit-remediate-loop` (3-round cap) over steps 2-5 and over the final recommendation doc.

---

## 14. Deliverable & what cannot be stated until the objective is fixed

Running this plan (conditional on the §5 objective) yields the concrete recommendation:
1. Metric suite + cluster SEs over the 39-cell grid → the **skill-vs-engagement frontier over γ** per (D/W, shape).
2. Apply the registered decision rule: among cells meeting the engagement floor φ, minimax over per-objective ranks on skill/tie/equity; break ties toward lower γ (less skill sacrificed) and toward the familiar 3:1 D/W when within cluster SE.
3. Confirm stability across the robustness sweep (§7.2), invariance to the draw model (§7.1), and non-exploitability (§7.3) at the chosen cell.
4. **Translate to small integers** (§3 invariance ⇒ integers chosen for communicability + tie granularity): fix `(W_pts,D_pts)` to the ratio's integer rep; realise the knockout layer as `round(K·A_shape)` at the smallest integer scale that hits the target γ within cluster SE and preserves increment monotonicity; if integerisation inflates ties, pick the next-finer scale (more attainable totals → fewer collisions — the principled, convexity-free way to manage ties). State both representations (terminal table + per-round bank).

**The concrete W, D, shape, and γ — hence the integer ladder — are functions of the §5 objective choice and its floor φ.** This plan fixes the grid, metrics, sensitivities, statistics, selection rule, implementation, and tests completely; the one remaining input is the owner's objective decision.

---

**AI-assistance (ICMJE 2026):** plan synthesised by Claude Opus 4.8 from four agent-compiled, independently citation-audited research notes under human orchestration; reproducibility log path per §11.
