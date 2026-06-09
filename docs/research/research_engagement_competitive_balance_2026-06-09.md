# Measuring engagement / competitive balance / suspense for the wcpool group-stage layer

Research note. Author: SKIE. Date: 2026-06-09. Repo HEAD at writing: `5c4ef6a`.

## 1. Scope & question

The prior study ([report](../report_draft_pool_layperson_2026-06-08.md)) optimised the scoring
ladder to **maximise the influence of draft skill** on the standings (metrics: `skill_variance_share`,
`spearman_ev_vs_placing`, `slot_equity_imbalance`, `top_tie_rate`, `champion_dominance` in
[metrics.py](../../src/wcpool/metrics.py)). All terminal ladders award **0 for a group-stage exit**
([ladders.py](../../src/wcpool/ladders.py): `vec[Stage.R32:] = …`), so under the current design every
drafter sits at exactly 0 until the Round of 32 — the entire group stage (48 of 104 matches, ≈46% of
the tournament's calendar) is dead for the pool.

The proposed feature — awarding pool points for **group-stage wins/draws** — will, on the skill axis,
almost certainly *lose*: three group matches per team are a small, high-variance, weakly-strength-correlated
sample, so adding them dilutes the share of final-score variance explained by roster quality. The
plausible real objective is therefore **engagement**: keep all participants scoring (and plausibly able
to lead) through the group phase, and reward drafters whose teams over-perform early. **None of the five
existing metric families measures this.** This note surveys how sports-economics and tournament-design
research quantifies outcome uncertainty / competitive balance / suspense, and translates the usable
constructs into metrics computable on the wcpool simulation output.

**Engine facts that constrain any metric.** `simulate_tournament` returns `stages` of shape
`(n_sims, 48)`, each entry the furthest `Stage` (0=GROUP … 6=CHAMPION) a team reached
([tournament.py](../../src/wcpool/tournament.py)). A drafter's **terminal** pool score is
`pool_scores(points, rosters)` = `points[:, rosters].sum(axis=2)`. To get a **running** score after
each stage boundary we need a *cumulative-by-stage* points array: for a ladder `L` (length-7, indexed
by stage) define the after-stage-`s` score of a team that reached furthest stage `f` as
`L[min(f, s)]` when the ladder is read as "points for *currently* being at this stage", or — for the
group-stage feature, where points are *additive per result* — as the sum of stage-credits earned up to
`s`. Both are pure functions of `stages` and the (existing or extended) ladder, so every metric below is
a thin post-processor over the panel the simulator already produces. No new tournament code is required;
only a `running_scores(stages, ladder, rosters)` helper of shape `(n_sims, n_drafters, n_stage_boundaries)`.

## 2. Key findings (claim + citation + evidence)

**F1 — Outcome uncertainty is the foundational demand driver in sports economics ("uncertainty of
outcome hypothesis", UOH).** Rottenberg first argued fans prefer closer, less-predictable contests, and
that demand falls when outcomes are foregone conclusions ([Rottenberg 1956, *J. Political Economy*
64:242-258, doi:10.1086/257790](https://doi.org/10.1086/257790)). This is the canonical justification for
caring about "is the pool still in doubt?" as a design objective distinct from skill. [VERIFIED — citation
+ DOI confirmed; full text behind JSTOR/UCP paywall, claim corroborated by multiple secondary surveys.]

**F2 — Competitive balance is a measurable construct with a standard toolkit, and Szymanski's survey is
the authoritative entry point.** [Szymanski 2003, *J. Economic Literature* 41(4):1137-1187,
doi:10.1257/002205103771800004](https://doi.org/10.1257/002205103771800004) reviews the contest-design and
competitive-balance literature, covering prize structure, the dispersion-of-win-percentage family, and
concentration indices. It frames three dimensions of balance — **match** uncertainty (one game),
**within-season** (spread of the final table), **between-season** (persistence of champions) — the
relevant one here being *within-season*. [VERIFIED — citation + DOI confirmed via AEA/RePEc.]

**F3 — The workhorse within-season balance measure is the Noll–Scully ratio: actual SD of win
percentage divided by the *idealised* SD under a coin-flip league.** The idealised SD is
`σ_ideal = 0.5 / √N` where `N` = games per competitor (the SD of a Binomial(N, 0.5) win proportion);
Noll–Scully = `σ_actual / σ_ideal`, equal to 1 for a perfectly balanced league and rising with imbalance.
Originated by Noll (1988) and Scully (1989), popularised by Quirk & Fort (1992). [Verified secondary:
[Wages of Wins — Noll-Scully](https://wagesofwins.com/noll-scully/);
[Booth, RePEc HCS Session08.3](https://hcapps.holycross.edu/hcs/RePEc/fek/Session08.3-Booth.pdf).]
**Caveat (load-bearing):** for short seasons / few games the *ratio* is biased and unstable — [Owen,
Ryan & Weatherston 2007, *Review of Industrial Organization* 31:289-302, doi:10.1007/s11151-008-9157-0](https://doi.org/10.1007/s11151-008-9157-0) ("Measuring Competitive Balance …
Using the HHI") and the Utt–Fort "pitfalls" critique
([Utt & Fort 2002, *J. Sports Economics* 3(4):367-373, doi:10.1177/152700202237501](https://doi.org/10.1177/152700202237501))
show the idealised-SD denominator and the HHI both need care with small N or unequal games. This matters
because **a pool has only ~6–8 participants and 3 group games each** — Noll–Scully on *pool standings*
would sit on exactly the small-sample regime these papers warn about, so it is informative as an *analogy*
but not recommended as the headline pool metric. [F3 VERIFIED for the formula; small-N caveat VERIFIED.]

**F4 — HHI and Gini are the concentration/inequality alternatives, but Gini has documented pitfalls for
balance.** The Herfindahl–Hirschman Index (`HHI = Σ s_i²`, `s_i` = a competitor's share of wins/titles)
and the Gini coefficient (area between the Lorenz curve of win-shares and the 45° line) are imported from
industrial organisation and income-inequality respectively. They are **concentration**, not *uncertainty*,
measures — high concentration ⇒ low balance. Owen et al. 2007 (above) formalise a normalised
`HHI* = (HHI − 1/n)` style decomposition for leagues. Gini specifically misbehaves as a balance index
(insensitive to where in the distribution inequality sits; non-zero even for an ideal league):
[Utt & Fort 2002, doi:10.1177/152700202237501](https://doi.org/10.1177/152700202237501). **Relevance:**
HHI of *final pool-score shares* is a clean one-number summary of "did one drafter run away with it",
but it is a champion-dominance restatement, partly covered already by `champion_dominance`. [VERIFIED.]

**F5 — Suspense and surprise have a rigorous, computable formalisation (Ely–Frankel–Kamenica).** A
Bayesian audience watches beliefs `μ_t` (a martingale) evolve. EFK define, per period, with `‖·‖` the
Euclidean norm over states:

- **Suspense** (forward-looking, the conditional variance of the *next* belief):
  `suspense_t = E_t[ ‖μ̃_{t+1} − μ_t‖² ]` — verbatim utility kernel `E_t‖μ̃_{t+1} − μ_t‖²`
  (baseline `u(x)=x`; the "standard deviation" baseline takes `√` of this).
- **Surprise** (backward-looking, realised squared belief jump): `surprise_t = ‖μ_t − μ_{t−1}‖²`
  (baseline = Euclidean distance, i.e. `√` of this).

Source: [Ely, Frankel & Kamenica 2015, *J. Political Economy* 123(1):215-260,
doi:10.1086/677350](https://doi.org/10.1086/677350) §2.1 — "suspense is induced by variance over the next
period's beliefs, and surprise by change from the previous belief to the current one … the baseline
specification where suspense is the standard deviation of `μ̃_{t+1}` (aggregated across states) and
surprise is the Euclidean distance between `μ_t` and `μ_{t−1}`." Their motivating Figure 1 is literally a
**win-probability path of a tennis match**: the dramatic match (lead changes, blown match points) is
labelled higher-suspense/higher-surprise than the one-sided one. [VERIFIED — the EFK definitions are
corroborated across indexed sources (Google Scholar / Semantic Scholar / EconPapers) and the Flepp
implementation; the published JPE text (doi:10.1086/677350) is paywalled.]

**F6 — EFK is already operationalised on football using the win-probability trajectory, with a
benchmark-range methodology that avoids a magic threshold.** [Flepp, Pawlowski & Richardson 2025,
"Suspense and Surprise in European Football", arXiv:2506.21253,
doi:10.48550/arXiv.2506.21253](https://doi.org/10.48550/arXiv.2506.21253) compute, minute-by-minute over
25,000+ matches, with `p^j_t` the probability of outcome `j∈{home,draw,away}` at minute `t` (the
**squared-norm (variance-kernel) variant** of Flepp et al. Eq. 1/Eq. 2):

- `Surprise = Σ_{t=1}^{90} Σ_{j} (p^j_t − p^j_{t−1})²`  (their Eq. 1)
- `Suspense = Σ_{t=1}^{90} [ p^{HS}_{t+1} Σ_j (p^j_{t+1|HS} − p^j_t)² + p^{AS}_{t+1} Σ_j (p^j_{t+1|AS} − p^j_t)² ]`
  (their Eq. 2), weighting each hypothetical next-step belief change by the probability of the event
  (home/away scoring) that would cause it. **Note on fidelity:** the *published* Flepp equations carry an
  outer square root over each per-minute term (a standard-deviation baseline); the pool metric here
  deliberately uses the squared (variance) form, which is the EFK quadratic/variance kernel — so the
  recommended `pool_suspense`/`pool_surprise` definition in §3 is internally consistent.

Two findings transfer directly: (a) they **calibrate against a benchmark range** computed from
*perfectly balanced* simulated matches (suspense ∈ [6.03, 6.89], surprise ∈ [1.17, 1.74]) rather than
asserting an arbitrary cutoff — the data-driven way to say "is this enough suspense?"; (b) they document
that **suspense ≠ competitive balance**: a 0–0 between equal teams is perfectly balanced yet near-zero
suspense, "as also noted by Scarf et al. (2019)". For the pool this means a balance index (everyone tied
at 0 all through the group stage) can look *maximally balanced* while delivering *zero* suspense — the
status quo. Suspense is the construct that captures the engagement motive; balance alone does not.
[VERIFIED — primary PDF read directly; Eqs. 1–2 reproduced as the squared-norm (variance-kernel) variant
(the published equations carry an outer square root; see the fidelity note above).]

**F7 — Tournament-design research already quantifies "matches that matter" / dead rubbers by Monte
Carlo, in the exact 2026 format, and this is the closest analogue to the pool's engagement goal.**

- [Csató & Gyimesi 2025, "Increasing competitiveness by imbalanced groups: the example of the 48-team
  FIFA World Cup", arXiv:2502.08565](https://arxiv.org/abs/2502.08565) define a last-round match as
  **stakeless** for a team if its outcome does not change the probability of {being eliminated,
  qualifying for the R32, qualifying for the R16}, and estimate `P(stakeless)` over **1,000 random group
  draws × 1,000 simulations** — the *same Monte-Carlo design as wcpool*. They refine it to a
  **win-expectancy-weighted** stakeless measure `S^W_i = S_i · W_i` (Elo win prob of the indifferent
  team), because a dead rubber is costlier when the unmotivated team is strong. [VERIFIED — arXiv HTML
  read; definitions quoted.]
- [Guajardo & Krumer 2023, "Format and schedule proposals for a FIFA World Cup with 12 four-team
  groups", NHH Discussion Paper 2023/2, handle RePEc:hhs:nhhfms:2023_002](https://ideas.repec.org/p/hhs/nhhfms/2023_002.html)
  (book-chapter version "… Every Win Matters", [Springer 2024,
  doi:10.1007/978-3-031-63581-6_11](https://doi.org/10.1007/978-3-031-63581-6_11)) list **"no dead rubbers"**
  as an explicit design criterion for the 12×4 format. [VERIFIED — RePEc record confirmed; note the
  "Every Win Matters" title is **Guajardo & Krumer**, *not* Csató, a common mis-attribution.]

The pool analogue of a "dead rubber" is a **participant who is mathematically eliminated from winning the
pool** before the final round, or whose marginal pool-position is unaffected by remaining results — i.e.
exactly EFK-zero-suspense for that participant.

## 3. Candidate metrics with exact definitions

Notation: `S` = `n_sims` replicates; `D` = `n_drafters`; stage boundaries
`B = (GROUP, R32, R16, QF, SF, FINAL, CHAMPION)` indexed `b = 0…6`. Let
`R[s, d, b]` = drafter `d`'s **cumulative** pool score in replicate `s` after boundary `b`
(from the proposed `running_scores` helper). Let `W[s, d, b]` = the **conditional win probability** of
drafter `d` after boundary `b` — i.e. `P(d finishes top of the pool | all results through stage b)`,
estimated by the engine's *own forward simulation* from that point (see §4 for the estimator). By the
tower property `W` is a martingale in `b`, which is what licenses the EFK machinery.

**(i) Fraction still "alive" / within reach entering the knockouts (`alive_fraction`).** After the group
boundary (`b=0` post-group-points), drafter `d` is **live** in replicate `s` if their *maximum
attainable* final score `R[s,d,0] + maxgain[s,d]` ≥ the *minimum* final score the current leader can
guarantee, where `maxgain[s,d]` is the largest additional score `d`'s still-active teams could
accumulate. Report `mean over (s,d)` of the live indicator, and the simpler **within-threshold** variant:
fraction of drafters whose group-phase deficit to the leader is ≤ `τ`. Mathematically-alive needs **no**
threshold (it is a feasibility computation from the ladder's maximum remaining points). The
within-threshold variant needs `τ`; set it *not* by fiat but as **the largest single-round point swing
available in the knockout phase** under the chosen ladder (e.g. a team going champion adds `L[CHAMPION]`),
so "within reach" means "one decisive result from the lead" — a ladder-derived, magic-number-free `τ`.

**(ii) Number of lead changes across stages (`lead_changes`).** Let `leader[s,b] = argmax_d R[s,d,b]`.
`lead_changes[s] = Σ_{b=1}^{6} 1{ leader[s,b] ≠ leader[s,b−1] }`. Report `mean_s`, and the share of
replicates with ≥1 change. Engagement-relevant refinement: count changes **occurring at or before the
group boundary vs after**, to show how much "action" the group layer injects. No threshold. (Ties at the
top: break by `argmax` deterministically, or count a change only when the *set* of co-leaders changes —
the latter is more robust and is the recommended variant given the pool's known tie behaviour,
`top_tie_rate` ≈ 5–19%.)

**(iii) Stage-wise variance decomposition of the final score (`stage_variance_share`).** Decompose the
total variance of the final pool score into the part **resolved during the group phase** vs the
**knockout phase**, via the martingale-increment (law-of-total-variance / ANOVA-by-stage) identity. With
`R[s,d,B_final]` the terminal score and `R[s,d,0]` the post-group score, and using that increments of a
martingale are uncorrelated:

`Var_s(R[·,d,final]) = Var_s(R[·,d,0]) + Var_s(R[·,d,final] − R[·,d,0])` *(holds for the conditional-mean
trajectory; for raw cumulative scores compute the two component variances directly and report both plus
their ratio).* Define `group_share_d = Var(group-phase increment) / Var(total)`, averaged over `d`. This
is the **direct quantitative answer to "does the group layer matter?"**: it is the fraction of the
final-standings uncertainty that the group stage, rather than the knockouts, resolves. It reuses the
exact variance-decomposition logic already in `skill_variance_share`
([metrics.py](../../src/wcpool/metrics.py) lines 94-116), so it is idiomatic to the codebase. **No
threshold.** Interpretation guard: a *higher* group share means more early engagement but (per F1/F3) less
skill, surfacing the engagement-vs-skill trade-off the study must price.

**(iv) EFK suspense/surprise on the pool win-probability trajectory (`pool_suspense`, `pool_surprise`).**
Treat `W[s,·,b]` ∈ Δ^{D−1} (the vector of conditional pool-win probabilities) as the EFK belief
martingale over the 6 stage boundaries. Per replicate:

- `surprise[s] = Σ_{b=1}^{6} ‖W[s,·,b] − W[s,·,b−1]‖²`  (EFK realised squared belief jump, F5/F6 Eq. 1).
- `suspense[s] = Σ_{b=0}^{5} E[ ‖W̃[·,·,b+1] − W[s,·,b]‖² | state at b ]`  (EFK conditional next-step
  variance, F5/F6 Eq. 2). The inner conditional expectation is estimated by branching the engine forward
  from the state at boundary `b`.

Aggregate over replicates (`mean_s`). To decide whether the group layer adds suspense, compute these for
**(a)** the status-quo terminal ladder (group stage contributes a degenerate `W` increment of ~0) and
**(b)** the group-points ladder, and report the **difference**, plus — following F6 — a **benchmark
range** from a *perfectly balanced* synthetic field (all 48 Elo-equal, via
[strength.py](../../src/wcpool/strength.py)/`make_synthetic_config(spread=0)`), which the engine can
already generate, to anchor "how much suspense is achievable" without an arbitrary cutoff. The
*group-phase share of total suspense*, `Σ_{b≤group} / Σ_{all b}`, is the single headline number.

## 4. Direct recommendation

**Implement two metrics: (iv) `pool_suspense`/`pool_surprise` (EFK) as the headline engagement metric,
and (iii) `stage_variance_share` as the interpretable, cheap companion.** Optionally add (i)
`alive_fraction` (mathematically-alive variant) as a layperson-facing one-liner. Rationale:

1. **(iv) is the construct that actually matches the stated motive.** The engagement claim is "keep
   everyone *in suspense* through the group stage", and EFK is the peer-reviewed, primary-source
   formalisation of exactly that, already operationalised on win-probability trajectories in football
   (F5, F6). It distinguishes the status quo (group stage = zero suspense, everyone flat at 0) from the
   proposal in a way **balance indices cannot** (F6: balanced-but-dead is the failure mode of the status
   quo). Reporting the *group-phase share of suspense* and its *delta* between ladders is a direct,
   falsifiable test of whether the feature delivers its purpose.

2. **(iii) is near-free and reuses existing code.** It is the same law-of-total-variance decomposition as
   `skill_variance_share`, recoloured by *stage* instead of *skill vs noise*, and answers the same
   question in a unit (share of variance) the report already uses. It also explicitly exposes the
   engagement-vs-skill trade-off (F1/F3): variance pushed into the group phase is variance taken away from
   the strength-correlated knockouts.

3. **Why not Noll–Scully / HHI / Gini as the headline.** They measure *balance/concentration of the final
   table*, not *trajectory suspense*, and on a 6–8-participant, 3-group-game pool they sit squarely in the
   small-N bias regime that Owen et al. 2007 and Utt & Fort 2002 warn against (F3, F4); Gini additionally
   has documented pitfalls as a balance index. HHI of final pool-score shares is worth reporting as a
   *secondary* concentration check but largely restates `champion_dominance`.

**Threshold-selection rationale (zero magic numbers).**
- (iv) and (iii) need **no threshold** (variances and squared belief-changes are parameter-free). The
  "is it enough?" question is answered by a **benchmark range** computed from the engine's own
  Elo-equal field (F6 method), and by the **paired delta** between the status-quo and group-points
  ladders — both data-derived, not asserted.
- (i)'s *mathematically-alive* variant needs no threshold (pure feasibility from the ladder's maximum
  remaining points). Its *within-reach* variant, if reported, sets `τ` = the **single largest knockout
  round point increment** under the active ladder, i.e. "one decisive result from the lead" — a value
  read off the ladder, not chosen.

**Estimator note for `W` (the conditional win probability).** The clean estimator is **nested simulation**:
at each stage boundary, fix the realised results up to that boundary and Monte-Carlo the remainder to get
`P(d wins pool | state)`. This is `O(stages × inner_sims)` per outer replicate and is the only nontrivial
cost. A cheaper, bias-controlled alternative that fits the current single-pass architecture: estimate `W`
by **kernel/k-NN regression of the terminal win indicator on the running score vector `R[·,·,b]`** across
the existing `n_sims` replicates (no re-simulation) — i.e. "among replicates that looked like this after
stage b, how often did d win?". Validate the cheap estimator against a small nested-simulation ground
truth before using it in the headline (cross-check mandate). The martingale property (`E[W_{b+1}|W_b]=W_b`)
is a free unit test: the mean of `W` must be flat across `b` and equal the unconditional win probability.

## 5. Open questions / residual uncertainty

- **`W` estimator choice is unresolved and is the main implementation risk.** Nested simulation is exact
  but multiplies cost; the regression surrogate is cheap but introduces smoothing bias whose size on a
  6–8-dim score vector with ~50k replicates is untested. Recommend a small spike: nested simulation on a
  1,000-replicate subset as ground truth, then pick the surrogate only if its suspense/surprise estimates
  agree within Monte-Carlo error.
- **Group-points ladder is unspecified.** Every metric here is defined relative to *a* group-points rule
  (e.g. +x per group win, +y per draw), but no such rule exists in [ladders.py](../../src/wcpool/ladders.py)
  yet, and the per-result additive scoring differs structurally from the current terminal/furthest-stage
  ladder. The metric code needs the `running_scores` helper *and* a decision on whether group points are
  additive-per-result or a flat "qualified" bonus. The choice will itself need the same
  data-driven/grid treatment the ladder steepness received.
- **Suspense vs surprise weighting.** EFK treat them as distinct preferences; the football paper finds
  they can move in opposite directions (their Fig. 1 trade-off). Which one the pool's organiser actually
  values is a normative call — recommend reporting both and not collapsing to one composite.
- **Tie handling in `W` and `lead_changes`.** The pool's non-trivial tie rate (5–19%,
  [report §3b](../report_draft_pool_layperson_2026-06-08.md)) means `argmax`-based leadership and the
  win-probability simplex both need the fractional-credit convention already used in
  `win_probability`/`champion_dominance`; not yet specified for the running/trajectory case.
- **Unread primaries.** Rottenberg 1956 and Szymanski 2003 full texts are paywalled; claims F1/F2 rest on
  verified citations + multiple secondary surveys, not on reading the originals. EFK's published JPE text
  is likewise paywalled — its definitions are corroborated across indexed sources and the Flepp
  implementation (F5). The two FIFA preprints (arXiv) were read directly.
- **ISD formula transcription error spotted in the wild.** One secondary source rendered the idealised SD
  as "0.5 divided by the square of games"; the correct, universally-cited form is `0.5/√N` (SD of a
  fair-coin Binomial proportion). Flagged so it is not propagated.

## 6. References

[VERIFIED] [Rottenberg, S. (1956). The Baseball Players' Labor Market. *Journal of Political Economy*
64(3):242-258. doi:10.1086/257790.](https://doi.org/10.1086/257790) — UOH origin (F1). Full text
paywalled; citation/DOI confirmed.

[VERIFIED] [Szymanski, S. (2003). The Economic Design of Sporting Contests. *Journal of Economic
Literature* 41(4):1137-1187. doi:10.1257/002205103771800004.](https://doi.org/10.1257/002205103771800004)
— competitive-balance survey (F2). Full text paywalled; citation/DOI confirmed via AEA + RePEc.

[VERIFIED] [Ely, J., Frankel, A., & Kamenica, E. (2015). Suspense and Surprise. *Journal of Political
Economy* 123(1):215-260. doi:10.1086/677350.](https://doi.org/10.1086/677350) — suspense/surprise
formal definitions (F5). Definitions corroborated across indexed sources (Google Scholar / Semantic
Scholar / EconPapers) and the Flepp implementation; the published JPE text (doi:10.1086/677350) is paywalled.

[VERIFIED] [Flepp, R., Pawlowski, T., & Richardson, T. (2025). Suspense and Surprise in European
Football. arXiv:2506.21253. doi:10.48550/arXiv.2506.21253.](https://doi.org/10.48550/arXiv.2506.21253)
— EFK operationalised on win-probability trajectories; benchmark-range method; suspense≠balance (F6).
PDF read directly; Eqs. 1–2 reproduced as the squared-norm (variance-kernel) variant (the published
equations carry an outer square root; the pool metric deliberately uses the variance form).

[VERIFIED] [Csató, L., & Gyimesi, A. (2025). Increasing competitiveness by imbalanced groups: the
example of the 48-team FIFA World Cup. arXiv:2502.08565.](https://arxiv.org/abs/2502.08565) — stakeless
("dead rubber") match definition + win-expectancy-weighted metric, same Monte-Carlo design as wcpool
(F7). arXiv HTML read; definitions quoted.

[VERIFIED] [Guajardo, M., & Krumer, A. (2023). Format and schedule proposals for a FIFA World Cup with 12
four-team groups. NHH Dept. of Business and Management Science, Discussion Paper 2023/2. Handle
RePEc:hhs:nhhfms:2023_002.](https://ideas.repec.org/p/hhs/nhhfms/2023_002.html) — "no dead rubbers" as a
design criterion (F7). Book-chapter version: [Guajardo & Krumer (2024), "… Every Win Matters", in *The
Palgrave Handbook on the Economics of Manipulation in Sport*, Springer,
doi:10.1007/978-3-031-63581-6_11.](https://doi.org/10.1007/978-3-031-63581-6_11) RePEc record confirmed;
chapter DOI confirmed; **attribution note: "Every Win Matters" is Guajardo & Krumer, not Csató.**

[VERIFIED] [Owen, P. D., Ryan, M., & Weatherston, C. R. (2007). Measuring Competitive Balance in
Professional Team Sports Using the Herfindahl-Hirschman Index. *Review of Industrial Organization*
31(4):289-302. doi:10.1007/s11151-008-9157-0.](https://doi.org/10.1007/s11151-008-9157-0) — HHI for
balance + small-N caveats on the SD-ratio (F3, F4).

[VERIFIED] [Utt, J., & Fort, R. (2002). Pitfalls to Measuring Competitive Balance with Gini Coefficients.
*Journal of Sports Economics* 3(4):367-373. doi:10.1177/152700202237501.](https://doi.org/10.1177/152700202237501)
— Gini/SD-ratio pitfalls (F4). Citation/DOI confirmed via secondary indices; abstract read.

[UNVERIFIED] Noll, R. (1988), unpublished/working material, and Scully, G. (1989), *The Business of Major
League Baseball* (Univ. Chicago Press) — original Noll–Scully ratio (F3). Cited universally as the origin
but the primary 1988/1989 sources were not located in full; the *formula* `σ_actual / (0.5/√N)` is
verified via multiple secondary sources and Quirk & Fort (1992), *Pay Dirt* (Princeton Univ. Press).

[UNVERIFIED] Scarf, P., et al. (2019) — cited inside F6 (Flepp et al.) for "competitive balance ≠ outcome
uncertainty". Not independently retrieved; the claim is reproduced from the verified F6 source, not from
the original.
