# wcpool — Monte Carlo for a 6-person 2026 FIFA World Cup draft pool

A simulation environment to evaluate **scoring ladders** and **draft designs** for a
6-person team-draft pool over the 2026 FIFA World Cup. It answers: *which `(teams-per-drafter N, scoring ladder)` choice maximizes the influence of draft skill on the final standings while keeping ties rare and snake-draft slots equitable — and is that choice robust to the team-strength model?*

## What it does

1. Simulates the **official 2026 tournament** (48 teams, 12 groups, top-2 + 8 best thirds, single-elim R32→Final) from a pluggable team-strength rating.
2. Runs a **6-person snake draft** under three policies (EV-greedy, greedy best-response, variance-seeking).
3. Scores each drafter under three **terminal-points ladders** (linear, triangular, geometric).
4. Reports five metric families over the `(N, ladder, strength-model)` grid: skill-vs-luck, slot equity, tie rate, champion dominance, robustness.

## Layout

| Path | Contents |
|------|----------|
| [src/wcpool/](src/wcpool) | engine modules (strength, tournament, draft, ladders, metrics, simulate) |
| [scripts/](scripts) | experiment runner + plotting entrypoints |
| [config/](config) | team ratings + experiment grid configs |
| [tests/](tests) | unit + integration tests |
| [docs/assumptions.md](docs/assumptions.md) | every modeling assumption, stated explicitly |
| [docs/tables/](docs/tables), [docs/figures/](docs/figures) | tidy results + plots |
| [docs/audits/](docs/audits) | audit-remediate-loop trail |
| [logs/](logs) | ReproLog records (one per run) |

## Quickstart

```bash
uv sync --extra dev
uv run pytest                     # unit + integration tests
uv run python scripts/run_experiment.py --quick   # smoke run (small sim count)
uv run python scripts/run_experiment.py           # full run (>=50k tournaments/cell)
uv run python scripts/make_plots.py
```

## Status

Built under the `audit-remediate-loop` QC pattern. Residual risk recorded at the bottom of
[docs/assumptions.md](docs/assumptions.md) after the final audit round.

## AI-assistance

Built with Claude Code (Opus 4.8). Role: scaffolding, implementation, audit. See
[docs/assumptions.md](docs/assumptions.md) for modeling provenance and the reproducibility log in [logs/](logs).
