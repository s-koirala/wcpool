# Knights' FIFA 2026 Draft

## The bottom line

Six players draft World Cup teams and score points as their teams advance; most points wins.
Two settings decide whether it's fair and fun:

1. **Teams per player → 8.** That uses all 48 teams, so everyone has a full squad and the
   champion is always owned. 4, 5, or 6 work equally well — the number barely affects who wins.
2. **Points per round → "Building": 1, 3, 6, 10, 15, 21** (for reaching the Round of 32,
   Round of 16, Quarter-final, Semi-final, Final, and winning).

This keeps draft seats reasonably fair, makes ties rare, and rewards good drafting — without
reducing the pool to "who got the champion." Basis: the tournament simulated 50,000 times
under each rule set.

## 1. How we tested it

- Give each of the 48 teams a strength rating (real published ratings; bigger gap = bigger
  favourite).
- Play the full 104-match tournament on the computer — stronger teams win more often, upsets
  still happen. The group stage, "8 best third-placed teams" rule, and knockout bracket
  follow FIFA's 2026 format.
- Replay it 50,000 times per rule set to see what usually happens.
- Draft teams for the 6 players and total everyone's points in every tournament.

Scoring example (Building ladder): a team knocked out in the quarter-final earns 6 points; a
team that wins the World Cup earns 21. Your score is the sum across your teams.

## 2. The three scoring systems

Each gives 0 points for a team that doesn't escape the group stage.

| Reach → | R32 | R16 | QF | SF | Final | Champion |
|---|---|---|---|---|---|---|
| Steady (linear) | 1 | 2 | 3 | 4 | 5 | 6 |
| Building (triangular) | 1 | 3 | 6 | 10 | 15 | 21 |
| Doubling (geometric) | 1 | 2 | 4 | 8 | 16 | 32 |

They differ in how much the late rounds matter: Steady makes a champion worth 6× a
first-round team; Doubling makes it 32×.

## 3. Results

### 3a. The core trade-off

![Trade-off between skill, ties, and seat fairness](figures/tradeoff_2026-06-08.png)

Each marker is one rule set: right = more skill, down = fewer ties, colour = seat fairness
(dark = fair, yellow = unfair). The markers best for skill and ties (bottom-right) are the
worst for fairness. You can't win all three at once.

### 3b. The numbers (real 2026 field)

| | Steady | Building | Doubling |
|---|---|---|---|
| Better drafter finishes higher? (0 = chance, 1 = always) | 0.17 | 0.22 | 0.25 |
| Share of the result the draft explains (rest is luck) | ~4% | ~7% | ~9% |
| Tie at the top | 19% | 5% | 0.1% |
| 1st-seat win chance (fair = 16.7%) | 26% | 33% | 38% |
| Last-seats win chance (5th/6th) | 12–13% | 10% | 7% |
| Champion's owner wins the pool | 51% | 77% | 99.9% |

The draft is a weak lever — even under Doubling it explains under 10% of who wins; the rest
is luck. The steeper the scoring, the more the first seat dominates and the more the pool
becomes "who owns the champion."

### 3c. Seat fairness

![Win probability by draft seat](figures/slot_equity_2026-06-08.png)

Win chance by draft seat (1 picks first … 6 last); dashed line = fair 16.7%. Doubling is most
lop-sided (first seat ~38%, late seats ~7%); Building is gentler; Steady is flattest but
tie-prone.

### 3d. Does it hold if favourites are stronger or weaker?

![Robustness across strength gaps](figures/robustness_2026-06-08.png)

Re-run from an even field to a top-heavy one (left to right). The order never changes:
Doubling gives the most skill and fewest ties, Steady the fairest seats. Seat unfairness
grows with a top-heavy field — and the 2026 field is top-heavy.

### 3e. You can't game the draft

A "shark" drafting cleverly did no better than just taking the best available team, under
every system. Simple drafting is fine.

## 4. What it means

One dial controls everything: how steep the scoring is.

- Steeper (Doubling): more skill, almost no ties — but a big first-pick advantage and a pool
  that's really about the champion.
- Flatter (Steady): fairer seats — but frequent ties and more luck.

Either way the pool is mostly luck: the best scoring still leaves over 90% of the result to
chance. The number of teams per player doesn't change this.

## 5. How other pools score

- **March Madness brackets:** ~81% use a fixed points-per-round system; the most common is
  1-2-4-8-16-32 (~70%) — our "Doubling"
  ([TeamRankings](https://www.teamrankings.com/blog/ncaa-tournament/bracket-pool-scoring)).
  Many use gentler ladders to stop the champion dominating — Fibonacci 2-3-5-8-13-21, or
  1-3-6-10-15-20 ([PrintYourBrackets](https://www.printyourbrackets.com/bracket-scoring.html)),
  nearly identical to our "Building." Some add upset bonuses.
- **World Cup sweepstakes:** teams drawn from a hat — pure luck
  ([FourFourTwo](https://www.fourfourtwo.com/competition/world-cup-2026-sweepstakes-kit-download-and-print-our-sweepstake-template)).
- **Pick'em / progressive pick'em:** predict match results; later rounds pay more, in linear
  (1,2,3…) or doubling (1,2,4…) variants
  ([PoolTracker](https://www.pooltracker.com/game_info/world-cup.asp)).
- **Auction / Calcutta:** bid money for teams; anyone can win any team, so the draft-seat
  advantage disappears — at the cost of more complexity.

## 6. Recommendations

**Teams per player: 8.** Uses all 48 teams; everyone has a full squad; the champion is always
owned. 4–6 are equally competitive — pick for convenience.

**Scoring: Building (1, 3, 6, 10, 15, 21).** Near the skill of steeper scoring, ties rare
(~5%), without the first-pick blowout or "all about the champion."

| Want | Use | Cost |
|---|---|---|
| Balanced (recommended) | Building 1-3-6-10-15-21 | mild first-seat edge; ~5% ties |
| Pure skill, no ties | Doubling 1-2-4-8-16-32 | first seat wins ~38%; pool ≈ champion |
| Fairest seats | Steady 1-2-3-4-5-6 | ~19% ties; needs a tiebreaker |

Set a tiebreaker in advance (e.g., most teams reaching the Final). If equal seats matter
most, use an auction instead of a snake draft.
