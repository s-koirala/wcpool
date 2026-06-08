# wcpool — Monte Carlo for a 6–8-person 2026 FIFA World Cup draft pool

A simulation environment to evaluate **scoring ladders** and **draft designs** for a
team-draft pool (6–8 participants) over the 2026 FIFA World Cup. It answers: *which `(teams-per-drafter N, scoring ladder)` choice maximizes the influence of draft skill on the final standings while keeping ties rare and snake-draft slots equitable — is that choice robust to the team-strength model, and does it hold as the pool grows from 6 to 7–8 participants?*

## What it does

1. Simulates the **official 2026 tournament** (48 teams, 12 groups, top-2 + 8 best thirds, single-elim R32→Final) from a pluggable team-strength rating.
2. Runs a **snake draft** (participant count is a parameter, default 6; swept over 6/7/8). EV-greedy and variance-seeking cover the full grid; a one-ply greedy best-response is a matched-subset exploitability probe.
3. Scores each drafter under three **terminal-points ladders** (linear, triangular, geometric).
4. Reports five metric families over the `(N, ladder, strength-model)` grid: skill-vs-luck, slot equity, tie rate, champion dominance, robustness.
5. **Participant-count sensitivity (6 vs 7 vs 8):** a controlled, paired comparison showing the recommendation holds, skill rises modestly, and the first-pick advantage grows as the pool grows.

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
uv run python scripts/make_plots.py               # tables, figures, recommendation
uv run python scripts/build_report_html.py        # self-contained report (HTML)
uv run python scripts/build_report_pdf.py          # report as PDF (via headless Chrome/Edge)
uv run python scripts/participant_sweep.py         # 6 vs 7 vs 8 participants comparison
```

## Plain-English report

A non-technical write-up for pool organisers (definitions, results, figures, the cons of
arbitrary scoring, and the final answer on teams-per-person and scoring) is at
[docs/report_draft_pool_layperson_2026-06-08.md](docs/report_draft_pool_layperson_2026-06-08.md)
(rendered to a shareable, images-embedded `.html` and `.pdf` of the same name).

## Status

Built under the `audit-remediate-loop` QC pattern. Residual risk recorded at the bottom of
[docs/assumptions.md](docs/assumptions.md) after the final audit round.

