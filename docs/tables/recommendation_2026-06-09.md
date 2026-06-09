# Recommended group-stage + knockout scoring — 2026-06-09

## Headline (engagement-constrained skill)

The prior study optimised **skill alone** and shipped the *triangular* knockout ladder. This extension adds a **group-stage win/draw layer** because engagement is now in scope: we want every drafter still scoring -- and plausibly able to lead -- through the group phase, instead of everyone flat at 0 until the Round of 32. The objective is therefore **maximise drafting skill subject to an engagement floor** (plan section 5 (iii)); the group layer *costs* a little skill, and we price exactly how much.

**Recommended scheme: the *triangular* knockout shape with a *3:1* group win:draw layer at the commensurability landmark gamma_match (realised group share gamma = 0.1574, occupancy-exact; approx 0.16 hereafter).** In concrete small integers:

- **Group stage: 3 points per win, 1 point per draw** (no points for goals or losses).
- **Knockout — terminal table:** reaching the Round of 32 = 9, Round of 16 = 27, Quarter-final = 54, Semi-final = 90, Final = 135, **winning the Cup = 189** (i.e. 9, 27, 54, 90, 135, 189).
- **Knockout — per-advance bank (the same thing):** bank **9, 18, 27, 36, 45, 54** for surviving each successive knockout round (Round of 32, Round of 16, Quarter-final, Semi-final, Final, Champion). Banking these increments reproduces the terminal table exactly (9, 18, 27, 36, 45, 54 cumulate to 9, 27, 54, 90, 135, 189).

**Why scale 9 (directly, by exact commensurability).** Scale 9 is the *unique* integer scale that realises the gamma_match landmark exactly in whole numbers: a team winning all three group games earns 9 group points = exactly the 9 points for reaching the Round of 32 (9 = 3 group wins x 3 = the R32 bank). This self-justifies the scale -- no reverse-fitted tolerance is invoked: scales 1..8 cannot place 3W on the R32 bank in integers, and 9 is the first that can. The realised group share is then gamma = 0.1574 (occupancy-exact, approx 0.16); it matches the swept gamma_match cell (0.1574) to floating point because both are the same G/(G+K) on the same purses.

### Why this cell

- **Shape = triangular** by the prior study's minimax over the gamma=0 anchor ranks (skill / tie / slot-equity) -- the phi=0 limit of the engagement-constrained objective, so it reproduces the shipped balanced default. Triangular's worst objective rank (2) beats both linear's and geometric's (3).
- **gamma = gamma_match (approx 0.16)** is the most decision-relevant value (plan section 4): it sits in the engagement-efficient band where group mass buys engagement at a low, roughly constant skill cost, before the gamma=0.5 equal-purse end where the skill cost accelerates. The skill cost here is only **+0.0025** (paired vs the gamma=0 triangular ladder, +/- 0.0007) = **0.6x the absolute between-draw cluster SE** (0.0043) -- i.e. practically free.
- **D/W = 3:1** (the FIFA-mirror, familiar from the on-screen standings) is chosen over 2:1 and wins-only because all three D/W are **skill-indistinguishable** at gamma_match (their paired-skill spread is well within the absolute cluster SE), and 3:1 has the lowest top-tie rate of the three. (The TIE figures in the candidate table below are the continuous-mix cells'; the **integer** ladder we ship ties at a MEASURED 0.5% -- see the integer-realisation note in the tiebreaker section.)

### Engagement-floor anchors (group_variance_share; the engagement axis)

The frontier is read RELATIVE to two data-derived anchors (plan section 5; Doc-4 F6 benchmark method): the **status-quo floor** is exactly 0 at gamma=0 (the group layer is absent -- the dead group stage), and the **achievable ceiling** is the equal-purse (gamma=0.5) share. No single engagement target phi is asserted; the recommended gamma_match sits low on this axis (engagement = 0.0044) by design: the group stage decides only a small, deliberately near-zero share (<0.5%) of who ultimately wins, so drafting skill is preserved, while every player still has a live, moving score from match one. The achievable ceiling on the engine's most-balanced field (synthetic_x0.5) brackets the same range.

| shape      |   floor (gamma=0) |   ceiling (gamma=0.5, real Elo) |   ceiling (synthetic_x0.5) |
|:-----------|------------------:|--------------------------------:|---------------------------:|
| linear     |                 0 |                          0.2762 |                     0.2615 |
| triangular |                 0 |                          0.163  |                     0.1595 |
| geometric  |                 0 |                          0.0951 |                     0.1007 |

## How much should the group stage matter? (the honest menu)

gamma_match is the recommended DEFAULT -- it is self-explaining (3 group wins == reaching the R32) and skill-preserving -- but it is not the only sensible point, and the alternatives are cheap. The table below prices the three decision-relevant group-mass settings of the 3:1 triangular shape with the REAL numbers, so a more-engaging point can be chosen knowingly. **Skill cost is in ABSOLUTE between-draw cluster-SE units** (the decision-relevant precision; < 1 SE is below the threshold of practical significance):

| how much the group stage matters   |   gamma |   engagement (gvs) |   skill cost (abs cluster-SE) |   champ-dominance |
|:-----------------------------------|--------:|-------------------:|------------------------------:|------------------:|
| gamma_match (recommended)          |   0.157 |             0.0044 |                           0.6 |             0.797 |
| 0.25                               |   0.25  |             0.0137 |                           0.9 |             0.782 |
| 0.5                                |   0.5   |             0.102  |                           4.3 |             0.7   |

Reading it: gamma_match costs 0.6 absolute cluster-SE of skill for engagement gvs 0.0044. **gamma = 0.25 (triangular 3:1)** costs about 0.9 absolute cluster-SE -- still sub-significance, the same regime as gamma_match's ~0.6 SE -- while delivering roughly 3x the engagement (gvs 0.0137 vs 0.0044) with champion-dominance essentially unchanged (0.797 -> 0.782). gamma = 0.5 is where the skill cost starts to accelerate (several cluster-SE). An owner wanting the group stage to carry visibly more of the standings can move to gamma = 0.25 at a still-negligible, honestly-stated skill cost; gamma_match stays the default for its clean self-explaining design, not because the higher-gamma points are unacceptable.

## Shape selection (prior minimax over the gamma=0 anchors)

| shape      |   skill (Spearman) |    tie |   slot-spread |   champ-dom |   worst rank |
|:-----------|-------------------:|-------:|--------------:|------------:|-------------:|
| linear     |             0.2114 | 0.2125 |        0.1563 |      0.4763 |            3 |
| triangular |             0.2833 | 0.0491 |        0.2832 |      0.8136 |            2 |
| geometric  |             0.2948 | 0.0002 |        0.3425 |      0.9997 |            3 |

Minimax over per-objective ranks selects **triangular** (lowest worst-rank). This is the engagement-constrained objective's phi=0 limit and reproduces the prior study's shipped default exactly.

## Multi-objective candidate table (all gamma>0 cells)

Skill is the seed-stable **paired** delta_skill_vs_anchor (the cell minus its own shape's gamma=0 ladder, formed per draw on shared brackets) -- the right within-shape skill axis; engagement_gvs is the estimator-free group_variance_share (0 at gamma=0). The **tie_rate column is the CONTINUOUS-mix cell's** (the swept convex blend); the shipped INTEGER ladder ties at a MEASURED 0.5% (lattice-dependent; see the tiebreaker section), not the continuous-mix value.

| shape      | D/W   | gamma       |   gamma_realised |   paired_dskill |   paired_se |   engagement_gvs |   tie_rate |   slot_spread |   champ_dom |
|:-----------|:------|:------------|-----------------:|----------------:|------------:|-----------------:|-----------:|--------------:|------------:|
| geometric  | 1:0   | 0.05        |           0.05   |         -0.0011 |      0.0004 |           0.0003 |     0      |        0.3426 |      0.9996 |
| geometric  | 1:0   | 0.12        |           0.12   |         -0.0006 |      0.0007 |           0.0021 |     0      |        0.3432 |      0.9994 |
| geometric  | 1:0   | gamma_match |           0.1468 |          0.0007 |      0.0008 |           0.0033 |     0.0001 |        0.343  |      0.9994 |
| geometric  | 1:0   | 0.25        |           0.25   |         -0.0006 |      0.0012 |           0.0121 |     0      |        0.3433 |      0.9986 |
| geometric  | 1:0   | 0.5         |           0.5    |         -0.0073 |      0.0019 |           0.0951 |     0      |        0.3389 |      0.9837 |
| geometric  | 2:1   | 0.05        |           0.05   |         -0.0025 |      0.0004 |           0.0001 |     0      |        0.3426 |      0.9996 |
| geometric  | 2:1   | 0.12        |           0.12   |         -0.0025 |      0.0006 |           0.001  |     0      |        0.3433 |      0.9996 |
| geometric  | 2:1   | gamma_match |           0.1765 |         -0.0029 |      0.0008 |           0.0024 |     0      |        0.3432 |      0.9994 |
| geometric  | 2:1   | 0.25        |           0.25   |         -0.0039 |      0.001  |           0.0059 |     0      |        0.3434 |      0.999  |
| geometric  | 2:1   | 0.5         |           0.5    |         -0.0132 |      0.0018 |           0.0486 |     0.0005 |        0.3405 |      0.9925 |
| geometric  | 3:1   | 0.05        |           0.05   |         -0.0022 |      0.0004 |           0.0002 |     0      |        0.3426 |      0.9996 |
| geometric  | 3:1   | 0.12        |           0.12   |         -0.0018 |      0.0006 |           0.0012 |     0      |        0.3432 |      0.9995 |
| geometric  | 3:1   | gamma_match |           0.1668 |         -0.0018 |      0.0008 |           0.0025 |     0      |        0.343  |      0.9994 |
| geometric  | 3:1   | 0.25        |           0.25   |         -0.0025 |      0.0011 |           0.007  |     0      |        0.3433 |      0.999  |
| geometric  | 3:1   | 0.5         |           0.5    |         -0.0109 |      0.0019 |           0.0568 |     0      |        0.3404 |      0.9911 |
| linear     | 1:0   | 0.05        |           0.05   |         -0.0058 |      0.0007 |           0.0016 |     0.0224 |        0.1539 |      0.4634 |
| linear     | 1:0   | 0.12        |           0.12   |         -0.0055 |      0.0008 |           0.0102 |     0.0215 |        0.1552 |      0.4668 |
| linear     | 1:0   | gamma_match |           0.2342 |         -0.0095 |      0.0012 |           0.0455 |     0.071  |        0.1476 |      0.4438 |
| linear     | 1:0   | 0.25        |           0.25   |         -0.0115 |      0.0013 |           0.0529 |     0.0307 |        0.1449 |      0.4356 |
| linear     | 1:0   | 0.5         |           0.5    |         -0.0403 |      0.0025 |           0.2762 |     0.0306 |        0.1123 |      0.3426 |
| linear     | 2:1   | 0.05        |           0.05   |         -0.0108 |      0.0008 |           0.0008 |     0.0097 |        0.1498 |      0.4597 |
| linear     | 2:1   | 0.12        |           0.12   |         -0.0114 |      0.0009 |           0.005  |     0.0096 |        0.1487 |      0.4603 |
| linear     | 2:1   | 0.25        |           0.25   |         -0.0151 |      0.0012 |           0.0269 |     0.0099 |        0.1439 |      0.4465 |
| linear     | 2:1   | gamma_match |           0.2759 |         -0.0158 |      0.0012 |           0.0343 |     0.0151 |        0.1416 |      0.4426 |
| linear     | 2:1   | 0.5         |           0.5    |         -0.0381 |      0.0022 |           0.1654 |     0.0123 |        0.1184 |      0.3838 |
| linear     | 3:1   | 0.05        |           0.05   |         -0.0095 |      0.0008 |           0.0009 |     0.0065 |        0.1502 |      0.4576 |
| linear     | 3:1   | 0.12        |           0.12   |         -0.0097 |      0.0008 |           0.0059 |     0.0135 |        0.1504 |      0.4589 |
| linear     | 3:1   | 0.25        |           0.25   |         -0.0134 |      0.0012 |           0.0315 |     0.0068 |        0.1457 |      0.4476 |
| linear     | 3:1   | gamma_match |           0.2625 |         -0.0138 |      0.0012 |           0.0355 |     0.0106 |        0.1452 |      0.4448 |
| linear     | 3:1   | 0.5         |           0.5    |         -0.0356 |      0.0023 |           0.187  |     0.0066 |        0.1222 |      0.3806 |
| triangular | 1:0   | 0.05        |           0.05   |         -0.0015 |      0.0004 |           0.0006 |     0.004  |        0.2813 |      0.8064 |
| triangular | 1:0   | 0.12        |           0.12   |         -0.0007 |      0.0006 |           0.0042 |     0.0029 |        0.2792 |      0.7997 |
| triangular | 1:0   | gamma_match |           0.1383 |         -0.0007 |      0.0006 |           0.0058 |     0.017  |        0.2782 |      0.7955 |
| triangular | 1:0   | 0.25        |           0.25   |         -0.003  |      0.0011 |           0.0237 |     0.0042 |        0.2721 |      0.7708 |
| triangular | 1:0   | 0.5         |           0.5    |         -0.0196 |      0.0021 |           0.163  |     0.0041 |        0.2376 |      0.6563 |
| triangular | 2:1   | 0.05        |           0.05   |         -0.0026 |      0.0004 |           0.0003 |     0.0018 |        0.2802 |      0.8035 |
| triangular | 2:1   | 0.12        |           0.12   |         -0.0026 |      0.0005 |           0.002  |     0.0013 |        0.2788 |      0.8027 |
| triangular | 2:1   | gamma_match |           0.1667 |         -0.0031 |      0.0006 |           0.0043 |     0.0035 |        0.2761 |      0.7945 |
| triangular | 2:1   | 0.25        |           0.25   |         -0.0048 |      0.0009 |           0.0116 |     0.0014 |        0.2731 |      0.7834 |
| triangular | 2:1   | 0.5         |           0.5    |         -0.0198 |      0.0019 |           0.0883 |     0.0043 |        0.2485 |      0.707  |
| triangular | 3:1   | 0.05        |           0.05   |         -0.0023 |      0.0004 |           0.0004 |     0.0012 |        0.2808 |      0.8054 |
| triangular | 3:1   | 0.12        |           0.12   |         -0.002  |      0.0005 |           0.0024 |     0.001  |        0.2785 |      0.8009 |
| triangular | 3:1   | gamma_match |           0.1574 |         -0.0025 |      0.0007 |           0.0044 |     0.0021 |        0.2777 |      0.7972 |
| triangular | 3:1   | 0.25        |           0.25   |         -0.0039 |      0.0009 |           0.0137 |     0.001  |        0.2738 |      0.782  |
| triangular | 3:1   | 0.5         |           0.5    |         -0.0185 |      0.0019 |           0.102  |     0.0011 |        0.2466 |      0.6996 |

## Stability across robustness concentrations

The shape choice (triangular by the prior minimax) is **stable in every concentration config** (synthetic spread 0.5x / 1x / 2x the Elo SD): triangular is the minimax winner in all three. The shape skill-ordering geometric > triangular > linear also holds at every gamma in every field, and the engagement benefit of raising gamma is present throughout, so the recommendation is invariant to field concentration.

## Draw-model invariance (confirmatory)

Draws now score, so P(draw) is scoring-relevant. The draw-model research ([research_draw_lowscore_modeling_2026-06-09.md](research/research_draw_lowscore_modeling_2026-06-09.md)) established that the engine's independent-Poisson field draw rate (0.1975) already matches the empirical pooled World-Cup group rate (0.1944), so the matched Dixon-Coles correlation is rho ~ 0 -- there is no draw deficit to correct, and the worst-case literature envelope is ~1 extra draw tournament, far below the skill cluster SE. The recommendation is **invariant to the draw model**; no re-implementation is needed (confirmatory per plan section 7.1).

## Exploitability re-test at the recommended cell (out-of-sample)

At the recommended (triangular, 3:1, gamma_match) scheme, a one-ply greedy best-response (slot 1) and a variance-seeking policy were scored OUT-OF-SAMPLE (deciding on the model/EV batch, graded on a held-out batch) against EV-greedy, at a reduced probe budget (8 draws). The between-draw noise scale on the slot win-prob is ~0.0090.

| policy        |   slot1_win_prob |   slot_win_prob_spread |   spearman_mean |
|:--------------|-----------------:|-----------------------:|----------------:|
| ev_greedy     |           0.3303 |                 0.2823 |          0.289  |
| best_response |           0.3295 |                 0.2814 |          0.2892 |
| variance      |           0.3247 |                 0.2755 |          0.2979 |

- best_response - ev_greedy (slot 1) = **-0.0009** -- within the noise the best-responder does not beat EV-greedy.
- variance - ev_greedy (slot 1) = **-0.0056** -- variance-seeking is no better either.

EV-greedy is **non-exploitable** at the recommended cell, matching the prior gamma=0 finding and Doc-3's prediction: the snake draft voids the crowd-avoidance mechanism (each team owned once), and the dense group layer further decorrelates scores.

## Integer-realisation tie rate (measured) + tiebreaker

The headline candidate table's tie column is the *continuous convex-mix* scheme's; the ladder we actually ship is the INTEGER scheme (points = 3*wins + 1*draws + the integer knockout ladder [9, 27, 54, 90, 135, 189]), whose set of attainable totals is a coarser lattice than the continuous mix, so its top-tie rate is **lattice-dependent and must be measured**. Scored on the SAME 100-draw headline batch (real Elo, 8 drafters x 6 teams, EV-greedy, seed 20260609, the 100 draws x 2000 sims), the integer scheme's measured top-tie rate is **0.5%** (+/- 0.02% between-draw cluster SE). The within-run mix=0 triangular ANCHOR scored on the same batch ties at **4.9%** (reproducing the frontier CSV's gamma=0 row) with skill 0.2833 and champ-dominance 0.8136. So the honest old-pool -> recommended tie comparison is **4.9% -> 0.5%** (both from the same run); the integer ladder ties MORE than the continuous-mix cell's 0.2% (lattice is coarser) but still far below the dead-group anchor. Skill is affine-invariant (EV-greedy is unchanged by an order-preserving rescale), so the integer scheme's measured skill 0.2808 matches the continuous gamma_match cell to rounding.

Ties at the top remain rare under this scheme (~0.5% of tournaments) but should still be resolved by an **exogenous tiebreaker** -- the convention Clair & Letscher (2007) assume and document for winner-take-all pools (e.g. an independent Monday-night-game prediction); they document the exogenous-tiebreaker practice, they do not prescribe it. Recommended: **most teams reaching the Final** (a knockout-depth criterion independent of the group layer).

**Calibration purses (real-Elo field):** E[sum group wins] = 57.801, E[sum group draws] = 28.399; bracket stage occupancy (GROUP..CHAMPION) = [16, 16, 8, 4, 2, 1, 1] (structural). The integer scheme's gamma is exact closed-form arithmetic on these.

---

**AI-assistance (ICMJE 2026):** selection rule, integerisation, the integer-tie measurement, and this recommendation synthesised by Claude Opus 4.8 (1M context) under human orchestration, over the audit-verified frontier sweep. AI is not an author. Reproducibility log: [logs/reproducibility/repro_log_7417f44086ec442287cf0648caac9b5b.json](../../logs/reproducibility/repro_log_7417f44086ec442287cf0648caac9b5b.json) (per the results_groupscoring_2026-06-09.csv .repro.json sidecar).