# Group-W/D/L + progression-bonus scoring variant — 8-player pool

**Date:** 2026-06-08 · **Engine:** [wcpool](../README.md) · **Field:** official 2026 draw, 48
teams, World-Football-Elo strength model (β = 1.776). Modeling assumptions and citations:
[assumptions.md](assumptions.md) (§8a for this variant).

## Scoring

| Component | Rule |
|---|---|
| Group match points | win **3**, draw **1**, loss **0**, summed over a team's 3 group matches (0..9) |
| Progression bonus (once per team) | reach R32 **+5**, QF **+8**, SF **+12**, win the Cup **+20** |

Bonus as a cumulative furthest-stage vector (derived from the milestones, not transcribed):

| stage | GROUP | R32 | R16 | QF | SF | FINAL | CHAMPION |
|---|---|---|---|---|---|---|---|
| bonus | 0 | 5 | 5 | 13 | 25 | 25 | 45 |

A team's points = group points + bonus[furthest stage]; a drafter's pool score is the sum
over their drafted teams. The spec sets **no milestone at R16 or at reaching the Final** (only
at winning), so those stages inherit the prior total — a beaten finalist scores the same 25
bonus as a beaten semi-finalist. Group W/D/L is scored for the **group stage only**; knockout
progression is rewarded through the bonus. Both are spec-faithful choices, flagged in §8a, and
each is a one-line change to flip.

**Pool:** 8 drafters × 6 teams = the entire 48-team field, EV-greedy snake draft (every drafter
takes the highest expected-points team available under this variant).

---

## Part A — the 2026 pool (official draw, 50,000 eval tournaments)

Team expected points are estimated on an independent EV batch; the snake draft runs once; rosters
are scored on a held-out eval batch (house EV/eval split — no in-sample optimism).

| Seat | Roster (snake order) | E[score] | SD | p5 | p50 | p95 | E[group] | E[bonus] | P(win) | P(top-3) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Spain, Mexico, Norway, Bosnia, Algeria, Curaçao | 73.7 | 20.9 | 40 | 74 | 108 | 25.9 | 47.8 | **0.342** | 0.634 |
| 2 | Argentina, Croatia, Uruguay, S. Africa, USA, Haiti | 68.7 | 18.3 | 41 | 67 | 101 | 26.0 | 42.8 | 0.209 | 0.545 |
| 3 | France, Japan, Morocco, Czechia, Qatar, Iraq | 65.0 | 17.9 | 37 | 64 | 96 | 24.9 | 40.1 | 0.151 | 0.481 |
| 4 | England, Belgium, Paraguay, Egypt, Sweden, Cape Verde | 58.6 | 16.4 | 35 | 57 | 89 | 25.3 | 33.3 | 0.076 | 0.314 |
| 5 | Brazil, Turkey, Canada, Australia, Panama, Jordan | 57.2 | 15.9 | 35 | 55 | 86 | 25.1 | 32.1 | 0.059 | 0.277 |
| 6 | Netherlands, Switzerland, Senegal, Côte d'Ivoire, Uzbekistan, Ghana | 56.6 | 15.7 | 34 | 55 | 85 | 24.2 | 32.3 | 0.050 | 0.269 |
| 7 | Portugal, Germany, Iran, S. Korea, Tunisia, DR Congo | 57.0 | 15.9 | 35 | 55 | 86 | 25.2 | 31.7 | 0.057 | 0.273 |
| 8 | Colombia, Ecuador, Austria, Scotland, Saudi Arabia, N. Zealand | 57.0 | 16.0 | 35 | 55 | 87 | 25.1 | 31.9 | 0.058 | 0.270 |

Source: [variant_wdl_8players_pool_2026-06-08.csv](tables/variant_wdl_8players_pool_2026-06-08.csv).

**Group points are a near-constant floor; the bonus is the lever.** Because each drafter holds 6
teams and group points are roughly i.i.d. across teams, the group component concentrates near
**25 for every seat** (range 24.2–26.0, spread 1.7). The bonus component ranges **31.7–47.8**
(spread 16.1, ≈ 9× the group spread). So adding W/D/L points lifts everyone's floor and breaks
ties, but does **not** change who wins versus a bonus-only scheme — the deep-run picks still
decide it.

**First-seat advantage is large.** Seat 1 (drafts Spain first) wins **0.342** of the time, **2.7×**
the fair share of 1/8 = 0.125, and reaches the podium 63%. Seats 5–8 are statistically
indistinguishable (P(win) ≈ 0.05–0.06) — by the back half of the snake the remaining teams are
near-interchangeable on EV. This is the known "first-pick advantage grows with pool size" effect
(assumptions §10), amplified here by the convex 45-point champion tail.

---

## Part B — design comparison vs the prescribed ladders (resampled, 25 × 2 000 = 50 000/cell, paired)

The variant and the three terminal ladders are scored on **byte-identical tournaments per draw**
(shared seed streams), so differences are the scoring rule, not sampling noise.

| Ladder | Skill (ρ) | Tie rate | Slot spread | Champion dominance | Winning score (mean) |
|---|---:|---:|---:|---:|---:|
| linear | 0.196 | 0.215 | 0.151 | 0.479 | 11.9 |
| triangular | 0.266 | 0.050 | 0.279 | 0.812 | 29.2 |
| geometric | 0.278 | **0.000** | 0.338 | 1.000 | 38.0 |
| **wdl_bonus_variant** | **0.238** | **0.025** | **0.249** | **0.752** | 93.0 |

Source: [variant_wdl_8players_ladder_compare_2026-06-08.csv](tables/variant_wdl_8players_ladder_compare_2026-06-08.csv).
Metrics defined in assumptions §7; skill = Spearman ρ(roster-EV rank, realised-placing rank);
tie rate = P(≥2 drafters share the top score); slot spread = max−min of per-seat win probability
(0 = equitable); champion dominance = P(the champion's holder wins the pool).

**Where the variant sits.** Adding near-continuous group points gives it the **second-lowest tie
rate (0.025)** — close to geometric's zero and 8.6× better than linear — without geometric's
extreme convexity. It pairs that with **triangular-grade skill** (ρ 0.238 vs 0.266) and **better
slot equity than triangular** (spread 0.249 vs 0.279). Champion dominance is 0.752: holding the
eventual champion wins the pool three times in four — high (the bonus tail is convex) but short of
geometric's winner-take-all 1.000. The winning-score mean (93) is simply on a larger scale (group
points + larger bonus magnitudes) and is not comparable across rows.

**Per-seat win probability (resampled):**

| Seat | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P(win) | 0.308 | 0.189 | 0.154 | 0.088 | 0.071 | 0.065 | 0.065 | 0.059 |

Seat 1 ≈ 2.5× the fair share (0.125); the tail seats sit just below it.

---

## Takeaways

- **The variant is a balanced design.** It buys near-geometric tie suppression and triangular-level
  skill while staying more slot-equitable than triangular — a sensible default if the pool wants
  few ties without making the contest pure winner-take-all.
- **Group W/D/L mostly removes dead picks and breaks ties; it does not rebalance the seats.** Every
  team now scores its group form (≈ 25-point near-constant floor per drafter), but the ~16-point
  bonus spread still decides standings. If the goal of adding W/D/L was to *reduce* the first-pick
  advantage, it does not — seat 1 still wins ≈ 0.31–0.34.
- **Two spec gaps to decide on** (both one-line edits to `PROGRESSION_BONUS_MILESTONES`): no R16
  milestone, and no reward for *reaching* the Final (runner-up = semi-finalist on bonus). Add a
  "reach the Final +k" milestone if a beaten finalist should outscore a beaten semi-finalist.
- **Lever, if seat fairness matters:** the first-seat edge is structural to snake + convex bonus.
  An auction/serpentine-with-compensation draft, or a flatter bonus, would shrink it more than any
  change to the group-points rule.

## Reproduce

```bash
uv run pytest tests/test_variant.py                 # variant unit + 8-player integration tests
uv run python scripts/simulate_variant_8players.py  # full run (this report's numbers)
uv run python scripts/simulate_variant_8players.py --quick   # smoke run
```

Master seed 20260608; results are byte-identical on re-run. A 13-field ReproLog (git HEAD, pip
freeze, dataset checksum, seed) is written to [logs/reproducibility/](../logs/reproducibility) on
every run.
