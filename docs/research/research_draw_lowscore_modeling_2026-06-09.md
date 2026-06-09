# Adequacy of the draw / low-score model now that group-stage draws are scored

**Date:** 2026-06-09 · **Project:** wcpool (FIFA 2026 draft pool) · **Author:** SKIE

## 1. Scope & question

The new scoring layer awards points for **group-stage draws** (FIFA 3/1/0), so the engine's
**draw probability** `P(D=0)` is now directly scoring-relevant. It was previously a tiebreak
input only; the prior ladders gave 0 for the whole group stage
([assumptions §8](../assumptions.md)). The goals model
([strength.py](../../src/wcpool/strength.py)) is **independent Poisson** (Maher 1982): each
team's goals ~ Poisson with log-rate linear in the Elo gap, β calibrated so
`P(win)+0.5·P(draw)` reproduces the World-Football-Elo logistic, μ_total pinned to the pooled
2014/2018/2022 men's-WC goals-per-match (2.6667). Win/draw/loss are the **Skellam** law of the
goal difference. The model **deliberately omits** the Dixon–Coles (1997) τ low-score
dependence correction — flagged in [assumptions §2](../assumptions.md) (lines 42–45) and as
residual risk.

**Question.** Is independent Poisson adequate for the **draw** probability, or is a Dixon–Coles
τ correction or a bivariate-Poisson covariance term needed before we trust group-stage draw
points? Decision-relevant output: whether a draw-model **sensitivity** is warranted, which
correction, and how to parameterize it without magic numbers.

**Method note.** Engine quantities (even-match draw rate, calibrated β, field-averaged draw
rate, DC-τ lift) are **reproduced from the live engine** at git HEAD `5c4ef6a` on the real
2026 Elo field ([config](../../config/ratings_elo_2026.yaml)). Empirical WC group-stage draw
counts were **derived first-hand** from the per-group Wikipedia match boxes (24 group pages,
2014/2018/2022) via the MediaWiki API and parsed programmatically; the parse is cross-validated
(see §3). Dixon–Coles 1997, Karlis–Ntzoufras 2003 and Maher 1982 bibliographic records were
field-verified against CrossRef (§6). The DC τ formula is corroborated by **three independent**
re-implementations; the original paper's ρ̂ and the Karlis–Ntzoufras abstract are paywalled and
are tagged accordingly.

## 2. Key findings (claim → citation → evidence)

**F1 — Independent (double) Poisson systematically *underestimates* the proportion of draws;
this is the canonical motivation for both DC-τ and bivariate Poisson.** The dependence-modelling
literature states it directly: Maher's own bivariate extension was "driven by the fact that the
initial proposal [independent Poisson] tends to **underestimate the proportion of draws**"
([Petretta, Schiavon & Diquigiovanni 2021, arXiv:2103.07272](https://arxiv.org/abs/2103.07272),
literature review). The two standard fixes both add a small positive dependence in the joint
goal distribution that lifts the diagonal (equal-score) cells. **Direction is unambiguous: real
football has more draws than independent Poisson predicts, for a *fixed* pair of marginal
goal rates.**

**F2 — Dixon–Coles τ raises 0-0 and 1-1 (and lowers 1-0/0-1) via a small *negative* ρ; it only
touches the four lowest cells.** DC multiply the independent-Poisson joint pmf on four cells by
([Dixon & Coles 1997, JRSS-C 46(2):265-280](https://doi.org/10.1111/1467-9876.00065)):
τ(0,0)=1−λμρ, τ(0,1)=1+λρ, τ(1,0)=1+μρ, τ(1,1)=1−ρ, τ=1 elsewhere. With the empirically-fitted
**ρ<0**, τ(0,0)>1 and τ(1,1)>1 (draws up) while τ(1,0),τ(0,1)<1 (1-0/0-1 down), so the **net
effect raises total draw probability**. Formula and sign confirmed by three independent
re-implementations with fitted ρ̂ all small-negative:
−0.13 ([opisthokonta 2011-12 EPL](https://opisthokonta.net/?p=890)),
−0.1285 ([dashee87 2017-18 EPL](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)),
−0.2242 ([datafc, Turkish league](https://urazakgul.github.io/datafc-blog/posts/en/post3/better-predictions-for-football-matches-how-does-the-dixon-coles-model-work.html)).
The cross-league **Mar-Co model dependence parameter θ̂₃** ([Petretta, Schiavon &
Diquigiovanni 2021, arXiv:2103.07272](https://arxiv.org/abs/2103.07272), Table 3) — the
dependence parameter of those authors' own CDF/copula-style "Mar-Co" construction, **not** a
bivariate-Poisson λ₃ — is likewise small and mostly straddles zero (England −0.013 [−0.038,
0.015]; France −0.027 [−0.051, 0.000]; Germany −0.078 [−0.119, −0.040]; Italy 0.003 [−0.044,
0.043]; Spain 0.015 [−0.011, 0.046]) — i.e. the goal-level dependence is **second-order and
weakly/inconsistently signed across competitions**.

**F3 — Karlis–Ntzoufras (2003) bivariate Poisson adds a covariance term λ₃ (≥0) that raises
the correlation of the two scores; the *diagonal-inflated* variant specifically improves draw
estimation and induces overdispersion.** The bivariate Poisson "allows for correlation between
the two scores"; an "inflation factor for diagonal terms… improves in precision the estimation
of draws and, at the same time, allows for overdispersed… marginal distributions"
([Karlis & Ntzoufras 2003, JRSS-D 52(3):381-393](https://doi.org/10.1111/1467-9884.00366);
bibliographic record CrossRef-verified, abstract wording [UNVERIFIED]). A constant λ₃>0 in the
plain bivariate Poisson, however, leaves the *win/draw/loss split largely unchanged* relative
to a draw-targeted inflation — which is why DC's diagonal-only τ and KN's diagonal inflation,
not the plain covariance, are the draw-relevant tools. Cross-tournament evidence for the sign:
Karlis & Ntzoufras (2000), pooling 24 European championships, found a **small positive**
goal correlation (cited in arXiv:2103.07272) — consistent with the small-negative DC ρ
(opposite sign convention, same "extra draws" direction).

**F4 — The bias is real but *small in magnitude*, and it is a property of the joint
distribution at *fixed marginals* — not a free addition once μ_total is pinned to the goal
rate.** This is the load-bearing distinction for wcpool (developed quantitatively in §3–§4).
DC/KN re-fit the attack/defence (marginal-rate) parameters *jointly* with ρ/λ₃; the dependence
term and the marginal rates are not separable. wcpool already pins μ_total to the **realised**
men's-WC goal rate and calibrates β to the Elo expected-score curve, so the engine's marginals
are anchored to data. Bolting a draw-inflating τ onto data-anchored marginals **double-counts**
and would push the engine's draw rate *above* the empirical WC rate (§4, F-quant-3).

## 3. Quantitative values

### 3a. Empirical men's-WC group-stage draw rates (first-hand parse)

Counted from the per-group Wikipedia match boxes (6 matches × 8 groups = 48 per tournament).
**Cross-validation:** the parser's 2014 group-stage goal total (136, 2.83 gpm) reproduces
Wikipedia's stated "136 goals… 2.83 per match" exactly; the 2018 Group B draw count (3) was
independently confirmed match-by-match. The 2018 "8/48" figure on footballhistory.org is
**erroneous** (the true count is 9 — Portugal–Spain 3-3, Iran–Portugal 1-1, Spain–Morocco 2-2,
plus 6 more across groups C/D/E/H).

| Tournament | Group matches | Draws | Draw rate | Group-stage goals | gpm |
|---|---|---|---|---|---|
| 2014 | 48 | 9 | 0.1875 | 136 | 2.833 |
| 2018 | 48 | 9 | 0.1875 | 122 | 2.542 |
| 2022 | 48 | 10 | 0.2083 | 120 | 2.500 |
| **Pooled** | **144** | **28** | **0.1944** | **378** | **2.625** |

Pooled **field** draw rate = **28/144 = 0.1944**, Wilson 95% CI **[0.138, 0.267]**. (Sources:
per-group pages e.g. [2018 Group B](https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_Group_B);
tournament pages [2014](https://en.wikipedia.org/wiki/2014_FIFA_World_Cup),
[2018](https://en.wikipedia.org/wiki/2018_FIFA_World_Cup),
[2022](https://en.wikipedia.org/wiki/2022_FIFA_World_Cup).)

### 3b. Model vs empirical — the right comparison is the *field-averaged* rate, not even-match

The engine's calibration diagnostic reports the **even-match** draw rate
(`stats.skellam.pmf(0, μ/2, μ/2)`), but that is the rate for two *equal* teams. The WC field is
full of mismatches, which have lower draw probability, so the decision-relevant model quantity
is the **field-averaged** draw rate over the matchups that actually occur. Computed on the live
engine (real 2026 Elo, μ_total=2.6667, calibrated β=1.776, calibration RMSE vs Elo curve 0.0061):

| Quantity | Value | Note |
|---|---|---|
| Engine **even-match** draw rate `P(D=0 \| λ=μ/2)` | **0.2600** | calibration diagnostic; **not** the field rate |
| Engine field-avg over all 48×47 ordered gaps | 0.2066 | whole-field analogue |
| Engine field-avg over the **72 actual 2026 group matchups** | **0.1975** | the decision-relevant model number |
| **Empirical pooled WC group-stage field rate** | **0.1944** | §3a; CI [0.138, 0.267] |
| Δ (engine actual-field − empirical) | **+0.0031** | inside the CI; ≈1/6 of a single draw over 144 |
| Δ (engine **even-match** − empirical) | +0.0656 | the misleading comparison |

**The engine's independent-Poisson field draw rate (0.1975) matches the empirical WC field rate
(0.1944) to within +0.003** — far inside sampling noise (the CI half-width is ±0.064). The
apparent 0.27-vs-0.19 "gap" cited from the even-match diagnostic is an **artifact of comparing
an even-match rate to a mismatch-laden field average**; once β-weighted by the real matchup
distribution (median |gap| 174 Elo, 42 of 72 matches within 200 Elo), independent Poisson is
already on target.

### 3c. Size and sign of a DC-τ correction *if applied* (engine, field-averaged over 72 matchups)

| ρ (DC convention) | Even-match DC draw | Field-avg DC draw (72) | Lift vs indep | vs empirical 0.1944 |
|---|---|---|---|---|
| 0 (independent) | 0.2600 | 0.1975 | — | +0.0031 |
| −0.05 | 0.2724 | 0.2069 | +0.0094 | +0.0125 |
| −0.08 | 0.2798 | 0.2125 | +0.0150 | +0.0181 |
| −0.13 (EPL literature) | 0.2922 | 0.2219 | +0.0244 | +0.0275 |
| −0.18 | 0.3045 | 0.2313 | +0.0338 | +0.0369 |

τ touches only the four lowest cells, so its **field-averaged** lift is diluted to roughly half
its even-match lift. Critically, applying any literature ρ moves the engine draw rate **away
from** the empirical 0.1944 (overshoot +0.013 to +0.037): on the present data-anchored μ_total,
**adding DC-τ makes the fit worse, not better** — exactly because the published draw inflation
was fitted *jointly with* the marginal rates, not on top of an already-realised marginal rate
(F4).

## 4. Direct implications

**F-quant-1 — Is a draw-model sensitivity warranted? Yes, but as a *robustness check on the
scoring conclusion*, not a model fix.** The point estimate says independent Poisson is adequate
for WC draws (Δ ≈ +0.003). But (i) the conclusion that draws earn points is **new**, (ii) the
DC omission is the standing residual risk, and (iii) the even-match vs field-rate confusion is
an easy way to mis-read the model. A bounded sensitivity converts "looks fine" into "demonstrably
does not change the ranking", which is the publishable claim.

**F-quant-2 — Which correction: Dixon–Coles τ, run as a one-parameter *draw-rate* sensitivity,
NOT a fresh bivariate-Poisson fit.** Rationale: (a) τ is the minimal, draw-targeted modification
of the *existing* independent-Poisson engine — it leaves Skellam-based W/D/L machinery and the
β/μ calibration intact and perturbs only the four low cells; (b) a full Karlis–Ntzoufras bivariate
Poisson would require re-deriving the W/D/L probabilities and re-calibrating β, large surface
area for a second-order effect; (c) the cross-league Mar-Co θ̂₃ CIs (F2) show the dependence is
weak and inconsistently signed, so a heavy model is unjustified.

**F-quant-3 — How to parameterize without magic numbers.** Two defensible, citation-grounded
choices — run **both** as the sensitivity envelope:
- **(A) Literature-ρ band.** Sweep ρ over the published fitted range, anchored to peer/vetted
  estimates: ρ ∈ {−0.13 (EPL, two sources), −0.05, −0.18}, i.e. the empirical spread of fitted
  DC ρ̂. No value is hand-picked; the band *is* the literature's reported dispersion. Report the
  scoring metrics at each endpoint.
- **(B) Draw-rate-matched ρ (preferred, self-calibrating).** Solve for the single ρ that makes
  the engine's **field-averaged** draw rate equal the **empirical** pooled WC rate 0.1944. From
  §3b the independent engine is already at 0.1975 > 0.1944, so the matched ρ is **≈0 or slightly
  positive** (independent Poisson is, if anything, marginally *over*-drawing the 2026 field at
  this μ). This is the data-driven, zero-free-parameter calibration: ρ is *determined* by the
  empirical target, not assumed. It makes the key point quantitatively — there is **no negative-ρ
  draw deficit to correct** for the 2026 WC field once marginals are data-anchored.

**F-quant-4 — Expected effect on the scoring-design conclusion: negligible.** The draw-points
layer changes each drawn group match by 1 point per team. The worst-case draw-rate perturbation
in the literature band is ≈ +0.034 field-averaged (ρ=−0.18, §3c), i.e. ≈ 0.034×3 ≈ 0.1 extra
drawn matches per group, ≈ 1 extra draw per *tournament* across 12 groups, distributed across
drafters. This is far below the between-draw cluster SE that governs the headline metrics
(skill-correlation cluster SE ≈ 0.012, [assumptions §6](../assumptions.md)). The ladder-convexity
ordering (geometric > triangular > linear on skill) is driven by the **knockout** payoff
structure, which the draw model does not touch. **Predicted conclusion: the recommendation
(triangular default) is invariant to the draw model across the entire literature-ρ envelope.**
The sensitivity should *confirm* this, not be expected to overturn it.

## 5. Open questions / residual uncertainty

- **WC ≠ club-league dependence.** Every fitted ρ/λ₃ above is from domestic leagues
  (EPL/Serie A/etc.). WC group matches differ (neutral-ish venues, one-off stakes, wider talent
  gaps). No published DC/bivariate ρ̂ on **World-Cup-only group** data was located; the matched-ρ
  approach (F-quant-3B) side-steps this by calibrating to WC draws directly, but the *shape* of
  the low-score dependence on WC data is unverified. **Gap.**
- **Original DC ρ̂ value [UNVERIFIED].** The 1997 paper's own estimate (English league 1992–95)
  is paywalled; secondary sources give the *typical* fitted range (−0.03 to −0.15) and the
  formula/sign, all consistent, but the exact original ρ̂ and its SE are not confirmed verbatim.
- **Karlis–Ntzoufras (2003) abstract wording [UNVERIFIED].** Bibliographic record CrossRef-verified;
  the "diagonal inflation improves draw estimation / induces overdispersion" wording is from
  publisher search abstracts and corroborated by the arXiv review's description, not read from
  the paywalled full text. The Serie A 1991-92 dataset attribution is plausible but unconfirmed.
- **μ_total sensitivity of the matched ρ.** The matched-ρ≈0 result depends on μ_total=2.6667
  (full-tournament gpm). The *group-stage-only* gpm is 2.625 (§3a, lower); re-pinning μ to the
  group rate would *lower* the engine draw rate slightly and could yield a small-negative matched
  ρ. Worth noting in the sensitivity but immaterial to F-quant-4's magnitude.
- **Dixon–Coles τ admissibility.** For large λ,μ the τ(0,0)=1−λμρ factor can go negative; standard
  practice constrains ρ to keep all four τ>0. At WC rates (λ≈1.0–1.6) and |ρ|≤0.18 this is not
  binding, but a production implementation must assert it.

## 6. References

**Primary (peer-reviewed) — bibliographic records CrossRef-verified:**

- [VERIFIED] Maher, M.J. (1982). *Modelling association football scores.* Statistica Neerlandica
  36(3):109–118. [doi:10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x)
  — the independent-Poisson goals model the engine implements; its bivariate extension was
  motivated by independent Poisson under-predicting draws.
- [VERIFIED] Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* JRSS-C (Applied Statistics) 46(2):265–280.
  [doi:10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) — τ low-score
  correction (0-0/1-0/0-1/1-1). *DOI, journal, vol/iss/pp confirmed; original ρ̂ value
  [UNVERIFIED] (paywalled).*
- [VERIFIED] Karlis, D. & Ntzoufras, I. (2003). *Analysis of sports data by using bivariate
  Poisson models.* JRSS-D (The Statistician) 52(3):381–393.
  [doi:10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366) — bivariate Poisson
  covariance + diagonal inflation for draws. *DOI/journal/vol/iss/pp confirmed; abstract wording
  and dataset [UNVERIFIED] (paywalled).* **Note:** the project brief mislabeled this DOI as
  `…00065` (that is the DC DOI); the correct KN DOI is `…00366`.
- [VERIFIED] Skellam, J.G. (1946). *The frequency distribution of the difference between two
  Poisson variates…* JRSS-A 109(3):296. — distribution of the goal difference used for W/D/L.

**Supporting (preprint / vetted technical):**

- [VERIFIED] Petretta, M., Schiavon, L. & Diquigiovanni, J. (2021). *On the dependence in
  football match outcomes: traditional model assumptions and an alternative proposal.*
  [arXiv:2103.07272](https://arxiv.org/abs/2103.07272) — full text text-extracted locally;
  states independent Poisson "tends to underestimate the proportion of draws"; Table 3 reports
  the **Mar-Co model dependence parameter θ̂₃** (the authors' own CDF/copula-style construction,
  not a bivariate-Poisson λ₃) per league with bootstrap 95% CIs (all small, mostly span 0).
  *Preprint (stat.ME); not journal-published as of retrieval.*
- [VERIFIED] DC τ formula + negative-ρ direction, three independent re-implementations (vetted
  technical): [opisthokonta ρ̂=−0.13](https://opisthokonta.net/?p=890);
  [dashee87 ρ̂=−0.1285](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/);
  [datafc ρ̂=−0.2242](https://urazakgul.github.io/datafc-blog/posts/en/post3/better-predictions-for-football-matches-how-does-the-dixon-coles-model-work.html).

**Empirical WC draw rates (official/tertiary) — first-hand parse, cross-validated:**

- [VERIFIED] Per-group match boxes, 2014/2018/2022, MediaWiki API (24 pages). 2014 goal-total
  cross-check vs [2014 FIFA World Cup](https://en.wikipedia.org/wiki/2014_FIFA_World_Cup)
  (136 goals, 2.83 gpm) exact; 2018 Group B draws (3) confirmed match-by-match against
  [2018 Group B](https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_Group_B). Pooled field draw
  rate 28/144 = 0.1944.
- [UNVERIFIED — superseded] footballhistory.org states 2018 group phase "8/48 drawn (16%)"; the
  first-hand parse gives 9/48 (0.1875). Discrepancy resolved in favour of the parse.

**Engine reproductions (this study, git HEAD `5c4ef6a`):** even-match draw rate 0.2600;
calibrated β 1.776 (real 2026 field); field-avg draw over 72 actual group matchups 0.1975;
DC-τ field-avg lift +0.009 to +0.034 over ρ∈[−0.05,−0.18]. All computed against
[strength.py](../../src/wcpool/strength.py) and [config](../../config/ratings_elo_2026.yaml).
