# Recommended (N, ladder) — 2026-06-08

## Headline

The three design goals **conflict along one axis — ladder convexity**. The same convexity that raises skill correlation and removes ties also concentrates pool wins in the first snake pick (it turns the pool into a referendum on who drafts the eventual champion). No single cell wins all three goals.

* **Balanced default (recommended): N=4 triangular (skill=0.223, tie=0.0463, slot-spread=0.235, champ-dom=0.793)** — best joint standing across all three objectives (minimax over per-objective ranks).
* **Pure skill / no-ties: N=5 geometric (skill=0.254, tie=0.0007, slot-spread=0.315, champ-dom=0.999)** — maximises skill and essentially eliminates ties, but is the *least* slot-equitable and makes the pool almost entirely about owning the champion.
* **Most slot-equitable: N=8 linear (skill=0.151, tie=0.1776, slot-spread=0.118, champ-dom=0.479)** — fairest across snake slots, but lowest skill and a high tie rate (needs an explicit tiebreaker rule).

**N is second-order.** Within a ladder, skill varies by <~0.01 across N in {4,5,6,8} (≈ 3 Monte-Carlo SE), while the cross-ladder gaps are 15–25 SE. Choose the ladder first; pick N on other grounds (e.g. N=8 drafts the full field).

## N-effect within the recommended ladder (`triangular`, anchor, EV-greedy; mc_se = SE of skill mean)

|   teams_per_drafter |   spearman_mean |   mc_se |   top_tie_rate |   slot_win_prob_spread |
|--------------------:|----------------:|--------:|---------------:|-----------------------:|
|                   4 |          0.2234 |  0.0019 |         0.0463 |                 0.2345 |
|                   5 |          0.232  |  0.0019 |         0.0502 |                 0.2357 |
|                   6 |          0.2219 |  0.0019 |         0.0495 |                 0.2253 |
|                   8 |          0.2184 |  0.0019 |         0.0506 |                 0.2208 |

## Full (N, ladder) grid — anchor config, EV-greedy

|   teams_per_drafter | ladder     |   spearman_mean |   skill_variance_share |   top_tie_rate |   slot_win_prob_spread |   p_champion_holder_wins |
|--------------------:|:-----------|----------------:|-----------------------:|---------------:|-----------------------:|-------------------------:|
|                   5 | geometric  |          0.2544 |                 0.095  |         0.0007 |                 0.3145 |                   0.9991 |
|                   6 | geometric  |          0.2487 |                 0.0919 |         0.0008 |                 0.3142 |                   0.9988 |
|                   4 | geometric  |          0.247  |                 0.0911 |         0.0005 |                 0.3146 |                   0.9994 |
|                   8 | geometric  |          0.2462 |                 0.0908 |         0.0011 |                 0.3141 |                   0.9982 |
|                   5 | triangular |          0.232  |                 0.0697 |         0.0502 |                 0.2357 |                   0.7733 |
|                   4 | triangular |          0.2234 |                 0.0643 |         0.0463 |                 0.2345 |                   0.7928 |
|                   6 | triangular |          0.2219 |                 0.0645 |         0.0495 |                 0.2253 |                   0.759  |
|                   8 | triangular |          0.2184 |                 0.0629 |         0.0506 |                 0.2208 |                   0.748  |
|                   5 | linear     |          0.1896 |                 0.046  |         0.1859 |                 0.1483 |                   0.5199 |
|                   4 | linear     |          0.1708 |                 0.0373 |         0.1985 |                 0.1336 |                   0.5365 |
|                   6 | linear     |          0.1642 |                 0.0359 |         0.1828 |                 0.1285 |                   0.4989 |
|                   8 | linear     |          0.1515 |                 0.0318 |         0.1776 |                 0.1184 |                   0.4792 |

Pareto-frontier cells (non-dominated in skill↑ / tie↓ / slot-spread↓): 5 geometric, 6 geometric, 4 geometric, 8 geometric, 5 triangular, 4 triangular, 6 triangular, 8 triangular, 5 linear, 4 linear, 6 linear, 8 linear.

## Robustness of the recommendation across strength models

`skill_rank_of_balanced` is the recommended cell's skill rank (1 = best) within each config's 12-cell (N x ladder) grid. The slot-spread column shows the slot-equity cost is driven by field concentration (large when top-8 share is high).

| strength_config   |   top8_title_share |   skill |   skill_rank_of_balanced |   tie_rate |   slot_spread |
|:------------------|-------------------:|--------:|-------------------------:|-----------:|--------------:|
| elo_2026          |              0.913 |   0.223 |                        6 |     0.0463 |        0.2345 |
| synthetic_x0.5    |              0.648 |   0.028 |                        8 |     0.0377 |        0.0362 |
| synthetic_x1      |              0.837 |   0.083 |                        6 |     0.043  |        0.0797 |
| synthetic_x2      |              0.977 |   0.111 |                        7 |     0.0372 |        0.0955 |

## Exploitability (best-response vs EV-greedy at slot 1)

|                                   |   best_response |   ev_greedy |   br_minus_evgreedy |
|:----------------------------------|----------------:|------------:|--------------------:|
| ('elo_2026', 'geometric', 4)      |          0.3953 |      0.3926 |              0.0027 |
| ('elo_2026', 'geometric', 6)      |          0.3957 |      0.3923 |              0.0034 |
| ('elo_2026', 'linear', 4)         |          0.2652 |      0.2593 |              0.0058 |
| ('elo_2026', 'linear', 6)         |          0.2622 |      0.2525 |              0.0097 |
| ('elo_2026', 'triangular', 4)     |          0.3419 |      0.3367 |              0.0052 |
| ('elo_2026', 'triangular', 6)     |          0.3385 |      0.3318 |              0.0067 |
| ('synthetic_x1', 'geometric', 4)  |          0.2463 |      0.2439 |              0.0024 |
| ('synthetic_x1', 'geometric', 6)  |          0.2475 |      0.2445 |              0.003  |
| ('synthetic_x1', 'linear', 4)     |          0.2034 |      0.2001 |              0.0032 |
| ('synthetic_x1', 'linear', 6)     |          0.1938 |      0.1881 |              0.0057 |
| ('synthetic_x1', 'triangular', 4) |          0.2247 |      0.2211 |              0.0036 |
| ('synthetic_x1', 'triangular', 6) |          0.2167 |      0.2127 |              0.004  |

A positive `br_minus_evgreedy` means a best-responding drafter in slot 1 beats the EV-greedy baseline there — i.e. EV-greedy is exploitable by that margin.
