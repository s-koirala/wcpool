# Acceptance criteria & audit contract — draft_advisor

The deliverable is a live draft-pick advisor for our 2026 World Cup pool, built on the
`wcpool` engine. This file is the spec the audit-remediate loop drives to.

## Pool parameters (fixed)
Snake draft · 4 teams per drafter · 7 or 8 participants (parameterized) · random seat ·
payout 1st wins / 2nd breaks even → objective `W = (n−1)·P(1st) + P(2nd)` (see
`objective_derivation.md`).

## Functional criteria
1. **Board** built on the **official** 2026 draw (groups fixed across sims) so bracket
   collisions are real; cached with a reproducibility record written *before* the artifact.
2. **Recommendation** ranks every available team by `W`, scored on the joint sim so
   collisions are priced automatically.
3. **Opponent uncertainty** handled by a softmax-over-EV sweep from EV-greedy (τ→0) to
   uniform (τ→∞); the recommendation's robustness across the sweep is reported.
4. **Tier cliffs** flagged as local curvature spikes (a `W` drop exceeding *both* neighbouring
   drops beyond MC noise, one-sided Bonferroni); display-only, robust to a convex board.
5. **Ceiling** = quantile-banded upper-tail (deep-run) contribution.
6. **Reach/wait** look-ahead from the deterministic snake gap.
7. **Standing** with PUSH/PROTECT; **post-draft summary** with swing team + bracket map.
8. **CLI** takes incremental pick input in snake order, auto-attributes ownership, runs
   offline and fast (board precomputed).

## Non-negotiable constraints (CLAUDE.md)
- **No magic numbers.** Every tunable must be data-driven or a cited/definitional constant:
  - `DEFAULT_N_SIMS` — justified by Monte-Carlo SE of a title probability (`board.py`).
  - cliff `CLIFF_ALPHA=0.05` — family-wise level for the local-spike Bonferroni test.
  - opponent `SIGMA_MULTIPLES` — σ-scaled grid spanning the two principled endpoints.
  - `DEFAULT_OPP_SAMPLES` — compute/precision knob, not a decision threshold.
- **Reproducibility.** git HEAD, field checksum, board hash, seed, NumPy version logged.
- **Reuse `wcpool`** — no duplicated engine logic.
- **Tests**: objective math, ΔW/ranking, snake auto-attribution, ceiling, collision, cliff
  detection (unit); full draft loop (integration). All green.

## Audit dispositions (round 1)
Five parallel auditors ran (code-reviewer, quant-auditor, literature-check,
reproducibility-verifier, format-auditor). Resolved this round:

- **A1 / aggregation (critical, FIXED).** Ranking now uses the **τ=0 baseline `W`** (the
  EV-greedy regime under which the engine's exploitability probe validated EV-greedy); the
  temperature sweep is used only for the robustness flag. The displayed `P1`/`top2`/`W` are
  all τ=0, so the sort key and shown levels reconcile. This also made the ranking
  deterministic (no RNG), removing the order-dependence and common-random-numbers concerns,
  and cut the per-pick cost ~200x (no sweep×reps in the ranking).
- **Cliff detector (critical, FIXED over rounds 1-3).** The Tukey-fence heuristic (~50% false
  positives on smooth declines) was first replaced by a linear-null bootstrap, which round 2
  showed still false-positived on *convex* boards (real boards are convex). Final method: a
  **local second-difference (curvature) spike test** — a gap is a cliff only if it exceeds
  *both* neighbouring gaps beyond MC noise (one-sided Bonferroni across interior gaps). A
  monotone-gap (convex or linear) decline has no spikes, so it yields no false cliffs. Cliffs
  are now **display-only**: the robustness sweep and reach/wait use fixed caps
  (`ROBUSTNESS_CAP`, `NEXT_TIER_CAP`), so no decision depends on the cliff list. Boundary gaps
  (the top team's separation) are not tested — that lead is visible in the `W` column.
- **`-1` roster sentinel (major, FIXED).** All scoring routes through `advisor._score`, which
  refuses any roster containing `-1` (prevents NumPy aliasing to the last team). Regression
  test added.
- **Private accessor (major, FIXED).** `objective.first_place_credit` is now public; the
  summary uses it instead of `_position_credit`.
- **Atomic cache + repro completeness (major, FIXED).** Board cache writes via temp-file +
  `os.replace` (no poison cache); the repro record now carries git HEAD, model hash,
  pip-freeze SHA, lock fingerprint, NumPy/scipy versions, host, and a UTC timestamp.
- **A2 ceiling banding (minor, FIXED).** Bands computed over the *displayed* candidate set;
  ties get equal bands (average-rank).
- **Magic numbers (major/minor, FIXED).** `N_CEILING_BANDS`, `ROBUSTNESS_CAP`,
  `NEXT_TIER_CAP`, `_MAX_ROWS`, `_MAX_AMBIGUOUS`, `CLIFF_ALPHA`, `CLIFF_N_BOOT` named with
  rationale; reach/wait verdict now driven by the best alternative's survival probability.

## Residual risk (carried, with rationale)
- **A3 self-completion (quantify, carried).** Our future picks complete EV-greedily (one-ply
  proxy, per `wcpool.draft_best_response`). We re-optimise at each real turn, so displayed
  absolute `W`/`P1` are conservative; the quant-auditor showed it *can* reorder mid-board
  candidates. The top-of-board ranking (where decisions are made) is far less sensitive.
  Mitigation option for a later round: complete our tail with `draft_best_response`.
- **A4 neutral venue.** Hosts (USA/Mexico/Canada) get no advantage; the `home_advantage` hook
  in `wcpool.strength` is the integration point.
- **Opponent model is assumed, not fitted.** The softmax-over-EV family with a σ-scaled
  temperature grid is a plausible, not estimated, model of human drafters; the sweep is a
  robustness check, not a calibrated belief.
- **Strength-input calibration (out of scope).** Market-implied / supercomputer calibration is
  the highest-ROI model upgrade but needs external odds; `strength.fit_ratings_to_title_probs`
  is the integration point.
