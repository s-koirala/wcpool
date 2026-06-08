# Objective derivation: why we maximize W = (n−1)·P(1st) + P(2nd)

## Payout

The pool pays **1st place** and **refunds 2nd place** (break-even); everyone else loses
their buy-in. Let the buy-in be `b`, with `n` drafters, and a winner-takes-rest split: 1st
receives the pot net of the 2nd-place refund, `(n−1)·b`, and 2nd receives `b` back.

| Finish | Gross payout | Net utility |
|---|---|---|
| 1st | `(n−1)·b` | `(n−2)·b` |
| 2nd | `b` | `0` |
| 3rd … last | `0` | `−b` |

## Expected utility

With `P1 = P(1st)`, `P2 = P(2nd)`:

```
E[U] = (n−2)·b·P1 + 0·P2 − b·(1 − P1 − P2)
     = b·[ (n−2)·P1 + P1 + P2 − 1 ]
     = b·[ (n−1)·P1 + P2 − 1 ]
```

`b` and the `−1` are constants, so maximizing `E[U]` is maximizing

```
W = (n−1)·P(1st) + P(2nd).
```

The weight on 1st relative to 2nd is **n−1**: **6:1 at n=7, 7:1 at n=8**. This is *near*
winner-take-all (so a ceiling/variance tilt to reach 1st is rewarded), with a genuine
loss-avoidance floor (2nd has zero downside). No threshold is hand-chosen — the weight falls
out of the payout. If the pool's split differs (e.g., 1st does not take the entire residual
pot), the weight rescales but `P(1st)` stays dominant; `objective.objective_weight` is the
single point to change.

## Placement probabilities (tie handling)

`P1` and `P2` use **fractional tie credit**: within a tie group of size `g` whose best
competition rank is `r`, the members jointly occupy finishing positions `r … r+g−1` and
split each position equally. For position 1 this reduces *exactly* to
`wcpool.metrics.win_probability` (verified in `tests/test_objective.py`), so `P1` here is
consistent with the engine; the same convention defines `P2`. Examples:

- Clean order (no ties): position = (#strictly above) + 1; `P2` is the mass on exactly one
  drafter strictly above us.
- Two tied for 1st: each holds position 1 with prob ½ and position 2 with prob ½ → `P1 = P2
  = ½`.

A real pool needs an explicit tiebreaker rule (the report recommends "most teams reaching
the Final"); fractional credit is the unbiased modelling counterpart, not a substitute for
that rule.
