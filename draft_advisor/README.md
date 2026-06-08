# draft_advisor — live pick advisor for the 2026 World Cup draft pool

A participant-side decision aid layered on the `wcpool` Monte-Carlo engine. Where `wcpool`
answers the *organiser's* design question, this answers ours: **at our turn in the snake
draft, which available team maximizes our probability of winning the pool?**

Pool: snake draft, 4 teams each, 7–8 players, random seat, **1st wins / 2nd breaks even** →
maximize `W = (n−1)·P(1st) + P(2nd)` (derivation in
[docs/objective_derivation.md](docs/objective_derivation.md)).

## How it works
1. **Board** ([board.py](board.py)) — simulate the **official** 2026 bracket `n_sims` times,
   producing `points[sim, team]` under the chosen ladder. One draw, fixed groups → real
   bracket-collision structure. Computed once, cached, with a reproducibility record.
2. **Advisor** ([advisor.py](advisor.py)) — at our pick, rank available teams by `W`, scored
   on the joint sim by completing the modelled draft (collisions priced automatically).
   Opponents modelled as softmax-over-value across a temperature sweep (EV-greedy ↔ uniform);
   the ranking's robustness is reported. Plus tier cliffs, ceiling bands, reach/wait,
   live standing, and a post-draft summary.
3. **CLI** ([cli.py](cli.py)) — log each team **as it is picked** in snake order; ownership
   is auto-attributed. Offline and instant (the board is precomputed). Interface regions are
   documented in [docs/interface_spec.md](docs/interface_spec.md).

## Quickstart (from the repo root)
```bash
uv run pytest draft_advisor/tests            # unit + integration tests
uv run python -m draft_advisor.cli --players 8 --rounds 4 --ladder triangular --seat 5
```
First launch builds and caches the board (slower); subsequent launches load the cache.

## Layout
| Path | Contents |
|---|---|
| [board.py](board.py) | value-board precompute + cache + repro stamp |
| [objective.py](objective.py) | placement probabilities, `W`, ceiling, cliff detection |
| [opponent.py](opponent.py) | softmax opponent model + temperature sweep |
| [advisor.py](advisor.py) | draft state, recommendation, reach/wait, standing, summary |
| [cli.py](cli.py) | the live REPL |
| [docs/](docs) | objective derivation, interface spec, acceptance criteria |
| [tests/](tests) | unit + integration tests |
| artifacts/, ../logs/reproducibility/ | board cache, repro records |

## Status
Built under the `audit-remediate-loop` QC pattern (≤3 rounds). Open items and residual risk
are tracked in [docs/acceptance_criteria.md](docs/acceptance_criteria.md).

## AI-assistance statement
Design discussion, implementation, documentation, and audit orchestration were produced with
Claude Opus 4.8 (claude-opus-4-8) under the audit-remediate-loop pattern; the human directed
scope and decisions. Reproducibility records: [../logs/reproducibility/](../logs/reproducibility).
