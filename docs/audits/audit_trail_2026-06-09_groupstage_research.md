# Audit trail — group-stage scoring research docs (2026-06-09)

## Deliverable
Four literature/research docs in [docs/research/](../research) supporting the group-stage
**win/draw + progressive-advancement** scoring extension (8 drafters, full 48-team draft, 6 teams each):

- [research_points_system_incentives_2026-06-09.md](../research/research_points_system_incentives_2026-06-09.md)
- [research_draw_lowscore_modeling_2026-06-09.md](../research/research_draw_lowscore_modeling_2026-06-09.md)
- [research_pool_scoring_design_2026-06-09.md](../research/research_pool_scoring_design_2026-06-09.md)
- [research_engagement_competitive_balance_2026-06-09.md](../research/research_engagement_competitive_balance_2026-06-09.md)

## Loop configuration
- Pattern: `audit-remediate-loop` (parallel-specialist-ensemble / Mixture-of-Agents). Cap 3 rounds; **converged in 1**.
- Round-1 auditors (parallel): `literature-check` ×4 (one per doc, independent primary-source verification; instructed not to trust producers' self-tags) + `format-auditor` ×1 (all docs).
- `quant-auditor` / `code-reviewer` / `reproducibility-verifier` not spawned: these are pure literature-synthesis docs (no code, statistical calculations, or ReproLog artifacts). They re-enter in Phase 2 (execution plan / implementation).

## Findings & disposition (Round 1)

| ID | Doc | Severity | Finding | Disposition |
|----|-----|----------|---------|-------------|
| P1 | points | minor | Garicano & Palacios-Huerta book chapter cited as "ch. 9"; correct = **ch. 8** (DOI/pages correct) | Fixed R1 |
| P2 | points | minor | CEPR DP 5231 / SSRN form missing full title "Sabotage in Tournaments: …" | Fixed R1 |
| P3 | points | minor | even-match draw "≈0.27 asserted in test_strength.py"; test asserts only band 0.20–0.35 | Fixed R1 (reworded) |
| D1 | draw | **major** | five cross-league values mislabeled "bivariate λ̂₃ (arXiv:2103.07272 T3)"; are **Mar-Co θ̂₃** (Petretta, Schiavon & Diquigiovanni 2021). Values/CIs correct, conclusion unchanged | Fixed R1 (relabeled; λ₃ reserved for Karlis-Ntzoufras) |
| D2 | draw | minor | enumerate all nine 2018 drawn matches inline | Dropped (count = 9 verified first-hand; cosmetic) |
| S1 | pool | minor | "Modeling Madness" (OR/MS Today 2001) misattributed to Kaplan & Garstka; author = **Peter R. Horner** (secondary feature) | Fixed R1 (re-cited Horner; evidence tier ~4) |
| S2 | pool | minor | ESPN per-round sequence "10,20,40,80,120,160" wrong; = **160,320** at rounds 5–6 | Fixed R1 |
| S3 | pool | minor | eq-3.1 lower endpoint OCR-read as a=1; analytically a=½ | Dropped (no doc change; a=½ already stated + analytically confirmed) |
| E1 | engagement | **major** | Owen, Ryan & Weatherston (2007) DOI `10.1007/s11151-007-9159-3` = HTTP 404; correct = `10.1007/s11151-008-9157-0` (load-bearing: anchors the small-N competitive-balance caveat that rejects Noll-Scully as headline metric) | Fixed R1 (all 3 sites) |
| E2 | engagement | minor | Flepp et al. Eq.1/2 labeled "verbatim"; published eqs carry outer √ (SD baseline) vs doc's squared (variance) form | Fixed R1 (relabeled variance-kernel variant + fidelity note; recommended metric unchanged & internally consistent) |
| E3 | engagement | minor | EFK "primary PDF read directly … verbatim" provenance overstated (JPE paywalled) | Fixed R1 (softened to indexed-source corroboration) |
| E4 | engagement | minor | Noll/Scully primary + Scarf et al. 2019 unverified | Dropped (honestly `[UNVERIFIED]`-tagged; not load-bearing) |
| F1 | all | minor | DOI prefix style inconsistent (engagement "DOI 10.x" vs others "doi:10.x") | Fixed R1 (normalized to `doi:`) |
| F2 | engagement | minor | one bare `http` link (wagesofwins) | Fixed R1 (→ https) |
| F3 | points | minor | References visible-label style inconsistent (`[doi:…]` vs `[link]`) | Fixed R1 (uniform labels) |
| F4 | all | minor | tag vocabulary not uniform across docs (`[PARTIAL]` etc.) | Dropped (per-doc legends defensible; each internally consistent) |

**Counts:** 0 critical · 2 major (both fixed) · 9 minor fixed · 4 minor dropped (cosmetic / honestly-disclosed).

## Format-auditor non-finding confirmations
- **Identity hygiene: PASS** — no real name, git email, OS username, or absolute user path (`skoir` / `C:\Users`) in any doc body; internal links repo-relative; only the intended `SKIE` pseudonym appears.
- **Magic numbers: PASS** — every recommended design value is a sweep grid or a data-/literature-anchored quantity, never a bare constant.
- **Filename convention: PASS** — `research_{description}_2026-06-09.md`.
- **Section completeness: PASS** — all four docs carry the six required sections.

## Residual risk (carried to Phase 2)
1. **Paywalled primaries verified at abstract/metadata level** (directions + identifiers confirmed via CrossRef + corroborating sources; per-coefficient effect sizes not extracted from paywalled tables): Moschini 2010, Dilger-Geyer 2009, Karlis-Ntzoufras 2003 (abstract + Serie A dataset), original Dixon-Coles ρ̂ value/SE, EFK 2015 JPE published text. Disclosed in-doc.
2. **Garicano & Palacios-Huerta**: no published JEEA journal article confirmable; the work exists as CEPR DP 5231 (2005) / SSRN 831964 / *Beautiful Game Theory* ch. 8 (2014). Tagged `[PARTIAL]` in-doc.
3. **No published source** designs a scoring rule against a multi-objective skill/equity/tie/concentration frontier, studies team-*exclusive* draft pools, or studies group-stage pool scoring. Those mappings are first-principles extrapolation flagged for direct Monte-Carlo, not citation (Doc 3).
4. **EFK suspense estimator** — the conditional pool-win-probability vector `W` (nested simulation vs k-NN/regression surrogate) is unimplemented and is the main Phase-2 implementation risk (Doc 4); needs a validation spike.
5. **Draw-model adequacy** conclusion (independent Poisson already on-target; matched-ρ Dixon-Coles as a *confirmatory* sensitivity) rests on a first-hand WC group-stage draw parse (28/144 = 0.1944, verified) + paywalled DC/KN abstracts (Doc 2).

## Provenance
git HEAD at audit: `5c4ef6a`. Auditors: `literature-check`, `format-auditor`. Producers + remediation: `general-purpose`. AI-assistance (ICMJE 2026): Claude Opus 4.8 — research synthesis, independent citation audit, surgical remediation — under human orchestration.
