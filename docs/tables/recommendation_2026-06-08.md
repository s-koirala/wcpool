# Recommended ladder (and N) — 2026-06-08

## Headline

The three design goals **conflict along one axis — ladder convexity**. The same convexity that raises skill correlation and removes ties also concentrates pool wins in the first snake pick (it turns the pool into a referendum on who drafts the eventual champion). No ladder wins all three goals; the choice is a tradeoff.

* **Balanced default (recommended):** the triangular ladder — skill=0.224, tie=0.0492, slot-spread=0.229, champ-dom=0.768. Best joint standing across all three objectives (minimax over per-objective ranks).
* **Pure skill / no-ties:** the geometric ladder — skill=0.249, tie=0.0008, slot-spread=0.314, champ-dom=0.999. Maximises skill and essentially eliminates ties, but is the *least* slot-equitable and makes the pool almost entirely about owning the champion.
* **Most slot-equitable:** the linear ladder — skill=0.169, tie=0.1862, slot-spread=0.132, champ-dom=0.509. Fairest across snake slots, but lowest skill and a high tie rate (needs an explicit tiebreaker rule).

**N is second-order and statistically indistinguishable.** Within the triangular ladder, skill ranges only 0.014 across N in {4,5,6,8} = 1.2x the per-cell between-draw cluster SE (0.0117); the geometric-vs-linear skill gap is 0.080 = 7x that SE. Choose the ladder first; set N on practical grounds (N=8 drafts the full 48-team field with no stars left undrafted; smaller N leaves more teams unowned).

Note: the between-draw cluster SE (~0.0117) is the correct precision here — the 50k per-cell tournaments are clustered within 25 draws, so the independent unit is the draw, not the sim. A naive iid SE would understate uncertainty roughly fourfold.

## Per-ladder summary (anchor config, EV-greedy; mean over N, with cluster SE)

| ladder     |   skill |    tie |   slot |   champ_dom |   cluster_se |
|:-----------|--------:|-------:|-------:|------------:|-------------:|
| linear     |  0.169  | 0.1862 | 0.1322 |      0.5086 |       0.0099 |
| triangular |  0.2239 | 0.0492 | 0.2291 |      0.7683 |       0.0118 |
| geometric  |  0.2491 | 0.0008 | 0.3144 |      0.9989 |       0.0122 |

## Full (N, ladder) grid — anchor config, EV-greedy

|   teams_per_drafter | ladder     |   spearman_mean |   spearman_cluster_se |   skill_variance_share |   top_tie_rate |   slot_win_prob_spread |   p_champion_holder_wins |
|--------------------:|:-----------|----------------:|----------------------:|-----------------------:|---------------:|-----------------------:|-------------------------:|
|                   5 | geometric  |          0.2544 |                0.0121 |                 0.0948 |         0.0007 |                 0.3145 |                   0.9991 |
|                   6 | geometric  |          0.2487 |                0.0121 |                 0.0916 |         0.0008 |                 0.3142 |                   0.9988 |
|                   4 | geometric  |          0.247  |                0.0123 |                 0.0909 |         0.0005 |                 0.3146 |                   0.9994 |
|                   8 | geometric  |          0.2462 |                0.0123 |                 0.0905 |         0.0011 |                 0.3141 |                   0.9982 |
|                   5 | triangular |          0.232  |                0.0117 |                 0.0695 |         0.0502 |                 0.2357 |                   0.7733 |
|                   4 | triangular |          0.2234 |                0.0122 |                 0.0642 |         0.0463 |                 0.2345 |                   0.7928 |
|                   6 | triangular |          0.2219 |                0.0116 |                 0.0643 |         0.0495 |                 0.2253 |                   0.759  |
|                   8 | triangular |          0.2184 |                0.0117 |                 0.0627 |         0.0506 |                 0.2208 |                   0.748  |
|                   5 | linear     |          0.1896 |                0.0101 |                 0.0459 |         0.1859 |                 0.1483 |                   0.5199 |
|                   4 | linear     |          0.1708 |                0.0101 |                 0.0373 |         0.1985 |                 0.1336 |                   0.5365 |
|                   6 | linear     |          0.1642 |                0.0098 |                 0.0359 |         0.1828 |                 0.1285 |                   0.4989 |
|                   8 | linear     |          0.1515 |                0.0095 |                 0.0318 |         0.1776 |                 0.1184 |                   0.4792 |

## Robustness across strength models (recommended ladder, mean over N)

`skill_rank` is the recommended ladder's rank among the 3 ladders within each config (1 = best skill). The slot-spread column shows the slot-equity cost is driven by field concentration (large only when the top-8 title share is high).

| strength_config   |   top8_title_share |   skill |   skill_rank_of_recommended |   tie_rate |   slot_spread |
|:------------------|-------------------:|--------:|----------------------------:|-----------:|--------------:|
| elo_2026          |              0.913 |   0.224 |                           2 |     0.0492 |        0.2291 |
| synthetic_x0.5    |              0.709 |   0.172 |                           2 |     0.0426 |        0.1833 |
| synthetic_x1      |              0.931 |   0.354 |                           2 |     0.0464 |        0.3977 |
| synthetic_x2      |              0.998 |   0.545 |                           2 |     0.043  |        0.6856 |

## Exploitability (out-of-sample best-response vs EV-greedy at slot 1)

The best-responder now decides on the EV (model) batch and is scored on an independent eval batch, so any positive margin is genuine exploitability, not in-sample optimism. The probe runs at a reduced budget (8 draws); the between-draw noise scale on slot win-prob is ~0.0107, so margins within ~that size are not distinguishable from zero.

|                                   |   best_response |   ev_greedy |   br_minus_evgreedy |
|:----------------------------------|----------------:|------------:|--------------------:|
| ('elo_2026', 'geometric', 4)      |          0.3913 |      0.3926 |             -0.0013 |
| ('elo_2026', 'geometric', 6)      |          0.3917 |      0.3923 |             -0.0006 |
| ('elo_2026', 'linear', 4)         |          0.2581 |      0.2593 |             -0.0013 |
| ('elo_2026', 'linear', 6)         |          0.2499 |      0.2525 |             -0.0025 |
| ('elo_2026', 'triangular', 4)     |          0.3344 |      0.3367 |             -0.0022 |
| ('elo_2026', 'triangular', 6)     |          0.3283 |      0.3318 |             -0.0036 |
| ('synthetic_x1', 'geometric', 4)  |          0.5408 |      0.5417 |             -0.001  |
| ('synthetic_x1', 'geometric', 6)  |          0.5401 |      0.5416 |             -0.0015 |
| ('synthetic_x1', 'linear', 4)     |          0.3464 |      0.35   |             -0.0036 |
| ('synthetic_x1', 'linear', 6)     |          0.3157 |      0.3228 |             -0.0071 |
| ('synthetic_x1', 'triangular', 4) |          0.4571 |      0.4603 |             -0.0032 |
| ('synthetic_x1', 'triangular', 6) |          0.4362 |      0.4392 |             -0.0031 |

`br_minus_evgreedy` > 0 means the slot-1 best-responder beats EV-greedy there; margins of the order of the noise scale above indicate EV-greedy is, at most, marginally exploitable.
