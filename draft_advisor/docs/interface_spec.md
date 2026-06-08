# Live interface spec — what each region tells you

The CLI prints labelled regions `[A]`–`[J]`. Sample (8 players, our seat 5, after the top
four favourites are gone):

```
[A] draft_advisor | players 8 | rounds 4 | ladder triangular | seat 5/8
    objective W = 7*P1 + P2 | field official draw, board sha f75f2b54 | n_sims 100000
[B] your picks (snake 5/8): 5, 12, 21, 28
[C] board: 4 gone . 44 left
    seat 1: Spain   ...   seat 5 (you): -   ...
  -- [D] YOUR PICK . overall 5 . round 1 ------------------------
    #  team                   W      P1    top2  ceiling  bracket
    1  England            1.610    7.7%   19.0%    HHH   clear
       -- tier cliff (+0.188) ------------
    2  Portugal           1.422    4.6%   13.5%    HHH   clear
   ...
  [F] -> take ENGLAND   [G] rank-1 across opponent sweep
  [H] ~6 picks before your next -> safe to wait (best alternative likely survives)
      P(Portugal survives to your next pick) =  78.8%
[I] your standing (projected): P1 7.7% . top2 19.0% . mode: PUSH
[J] POST-DRAFT SUMMARY  (prints when the draft completes)
```

## [A] Session banner — *is it scoring the right game, reproducibly*
Players, rounds, ladder, our seat, and the objective weight `W = (n−1)·P1 + P2` (verify it
reads `7*` at 8 players, `6*` at 7). `field official draw` confirms the board is built on the
real 2026 groups; `board sha` + `n_sims` stamp the exact value board. If this line is wrong,
every number below is wrong.

## [B] Your pick schedule — *when you're up, and the gaps*
The 1-indexed overall pick numbers belonging to our seat. The gaps between them drive every
reach/wait call in `[H]`; with a random seat this is the one thing you enter (once).

## [C] Board / pick log — *live ownership and forming threats*
Count gone/left and each seat's roster, auto-attributed from the snake order — you enter only
the team as each pick happens. Watch for an opponent hoarding contenders (a threat to
out-ceiling) or stacking one bracket region (value they are leaking to you).

## [D] Recommendation panel — *ranked best pick now*
One row per available team, sorted by `W`.

| Field | Meaning | How to read |
|---|---|---|
| `W` | The objective if you take this team and the modelled draft completes. The sort key. | Higher = more win-equity. |
| `P1` / `top2` | Resulting P(1st) and P(top-2) levels. | `top2` is your break-even floor; `P1` is the prize. |
| `ceiling` | Banded contribution to the upper tail of your roster score (deep-run/title equity), quantile-banded `H..` / `HH.` / `HHH`. | Risk character: `HHH` lifts your best case (what wins a near-winner-take-all pool); `H..` is a floor pick. Use with `[I]`: PUSH → favour `HHH`; PROTECT → a safe pick at similar `W` is fine. |
| `bracket` | `clear`, or `shares <team>` when this team's deep runs trade off with one you already own (sim-derived, sign of covariance). | Explanatory only — the collision is **already priced into `W`**; this names *why* a strong team may rank low. |

## [E] Tier cliff — *scarcity (display aid)*
A break printed where a `W` drop is a **local spike** — larger than *both* neighbouring drops
beyond Monte-Carlo noise (a curvature/changepoint test, robust to a smoothly convex board, so
it does not flag the gradual decline that a convex board produces). Teams above a cliff are
materially separated from those below. Cliffs are informational only — the recommendation,
robustness sweep, and reach/wait do not depend on them.

## [F] Recommended pick — *the single answer*
The argmax of `W`.

## [G] Robustness — *confidence under unknown opponents*
`rank-1 across opponent sweep` means the pick is the top choice at every modelled opponent
behaviour (EV-greedy ↔ uniform), so your read of the table is not needed. If it instead
reports a rank range, the pick is contingent — that is where human judgement of the room adds
value.

## [H] Reach/wait — *take now vs. punt*
Uses the deterministic snake gap plus the opponent sampler to estimate the probability your
best *alternative* survives to your next pick. `take now` = it is unlikely to survive (grab the
scarce tier); `safe to wait` = it is more likely than not to still be there next round.

## [I] Standing — *push or protect*
After each of your picks: projected P(1st)/P(top-2) and a mode read off the objective.
`PROTECT` when your projected P(top-2) already leads the field (the floor is the binding
margin); `PUSH` otherwise (you still need ceiling to climb). The optimiser always maximizes
`W`; mode is a display aid.

## [J] Post-draft summary — *what you're rooting for, and your range*
Final roster (with group letters), P(1st)/P(2nd)/P(in-money), score distribution
(mean/P5/P50/P95), the single **swing team** (the one whose results most determine your
finish), and the repro stamp.
