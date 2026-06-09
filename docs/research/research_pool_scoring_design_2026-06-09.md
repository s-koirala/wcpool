# Scoring-system design & strategy in sports/office/bracket pools — research synthesis

**Date:** 2026-06-09 · **Project:** wcpool (FIFA 2026 draft pool) · **Author:** SKIE

## 1. Scope & question

We are extending the wcpool scoring ladder (terminal knockout-round points) with a
group-stage win/draw layer plus progressive knockout-advancement points. The prior study
([report](../report_draft_pool_layperson_2026-06-08.md);
[recommendation](../tables/recommendation_2026-06-08.md)) established that one design axis —
**ladder convexity** (how steeply points rise per round) — jointly governs four engine
metrics ([metrics.py](../../src/wcpool/metrics.py)): skill-vs-luck (Spearman EV-vs-placing +
between/within variance share `skill_variance_share`), snake-slot equity
(`slot_equity_imbalance`), tie rate (`top_tie_rate`), and champion dominance
(`champion_dominance`, P(champion's owner wins)). Steeper ladders raise skill and kill ties
but concentrate wins on the champion's owner (geometric → `p_champion_holder_wins`=0.999) and
worsen slot equity. The pool pays 1st and refunds 2nd, so the strategic objective is
W = (n−1)·P(1st) + P(2nd) ([objective_derivation](../../draft_advisor/docs/objective_derivation.md));
our pool is **small** (n = 6–8 drafters).

This memo asks what the operations-research / sports-economics literature on betting, office,
and bracket pools establishes about: (a) designing the scoring structure; (b) the skill/luck
balance; (c) round-progressive ("doubling-per-round") scoring and its concentration effect;
(d) tie-handling; (e) whether favorite-loading is exploitable — and extracts transferable,
quantitative design principles. The five papers named in the brief are the spine; two
open-access reviews and one practitioner tool corroborate.

**Method note.** The Clair & Letscher (2007) full text was obtained as a publisher PDF and
text-extracted locally (16 pp, 79,815 chars); quotes below are verbatim from that extraction.
Kaplan & Garstka, Metrick, Breiter & Carlin, and Niemi et al. are paywalled; their claims are
sourced from publisher abstracts, Horner's (2001) secondary INFORMS *OR/MS Today* feature on
Kaplan & Garstka, and two peer-reviewed/preprint reviews that cite them. All five DOIs were resolved and field-verified
against CrossRef (§6).

## 2. Key findings (claim → citation → evidence)

**F1 — Maximizing expected score is the wrong objective in a competitive pool; you must
maximize P(win), which depends on the *opponents*.** Kaplan & Garstka (2001) give an exact
dynamic program for the entry of *highest expected score* (and the mean **and variance** of
total points for any slate), building 64 team-anchored brackets backward round-by-round and
keeping the best. Under simple "count correct winners" scoring this DP just reproduces chalk
(the seeds) and "[does] not surpass the simple strategy of picking the seeds"; "for a more
sophisticated point structure, their models do outperform picking the seeds"
([Kaplan & Garstka 2001](https://doi.org/10.1287/mnsc.47.3.369.9769); method reported in
[Horner, P. R. (2001), "Modeling Madness," INFORMS *OR/MS Today* 28(1)](https://www.informs.org/ORMS-Today/Archived-Issues/2001/orms-2-01/Modeling-Madness),
a secondary feature on Kaplan & Garstka (2001)).
The limitation they expose: picking favorites "doesn't give you an advantage over other
participants, since the seedings are known to all pool players in advance" — i.e. an
expected-score-optimal entry is *correlated with everyone else's*, so it rarely wins outright.
This is the gap the next four papers close.

**F2 — Pool participants systematically over-back heavy favorites relative to game-theoretic
equilibrium; the bias shrinks only slightly in larger pools.** Metrick (1996) treats pools as
a natural experiment: observed behavior "differs from equilibrium behavior in a robust
manner; bettors overback the heaviest favorites," and "the size of this bias falls slightly in
larger pools, where the induced profit opportunities are higher," consistent with only a small
fraction of players adjusting to the strategic situation
([Metrick 1996](https://doi.org/10.1016/S0167-2681(96)00855-4), abstract; corroborated by
[Clair & Letscher 2007](https://doi.org/10.1287/opre.1070.0448), §3.3). This over-backing is
the market inefficiency that contrarian strategies exploit — it is a property of the *crowd*,
not of the scoring rule.

**F3 — Optimal strategy is pool-size-dependent: bet favorites in small pools, shift to upsets
("crowd avoidance") only as the pool grows large.** This is the central, directly transferable
result. Clair & Letscher (2007) build an exact model of opponent behavior (each opponent picks
the favorite in game *i* with "pool probability" *pᵢ*; true win probability *aᵢ*) and an exact
formula for expected monetary return. Verbatim: optimal picks are "very sensitive to the
number of opponents in the pool, generally **picking more conservatively in small pools and
choosing more upsets in larger pools**." For a single game vs *N* opponents the favorite/underdog
threshold is

> a = ( 1 + ((1−p)/p)·((1−(1−p)^N)/(1−p^N)) )^(−1)   (their eq. 3.1)

"which interpolates between a = ½ and a = p … for N = 1,…,15." Limits: at **N = 1** "one should
bet the actual favorite"; as **N → ∞** expected return → a/p for the favorite vs (1−a)/(1−p)
for the underdog, so you "bet the edge" — favorite iff true prob a exceeds pool prob p. The
practical upshot they state: "In larger (thousands of players) pools, crowd avoidance is
essential — picking all favorites is one of the few losing bets"
([Clair & Letscher 2007](https://doi.org/10.1287/opre.1070.0448)). The contrarian edge is a
**large-N** phenomenon; for small N favorites are optimal or near-optimal.

**F4 — The mechanism that makes favorite-loading lose in large pools is *correlation of your
score with opponents'*, not low win probability per se.** Clair & Letscher's worked NCAA example
(N = 5,000,000, ESPN scoring, Massey ratings): optimal contrarian picks have expected return
**798.8 with 0.15 correlation** to opponent scores, whereas the maximum-expected-score entry
returns only **32.7 because of 0.37 correlation** with opponents — "often by orders of
magnitude" better
([Clair & Letscher 2007](https://doi.org/10.1287/opre.1070.0448)). They derive the covariance
between two opponents' scores and between your score and an opponent's explicitly (their App.
A14). **Lowering correlation with the field is what buys win probability.** This is the same
quantity our `champion_dominance` metric measures: a steep ladder forces every drafter's score
to be dominated by the single champion term, maximizing inter-drafter score correlation —
exactly the high-correlation, low-edge regime Clair & Letscher penalize.

**F5 — Contrarian play = pick an undervalued champion, then fill the rest with favorites; it
raises ROI but only matters when the pool is large.** Breiter & Carlin (1997) first framed
office-pool play as picking to *win the pool* (a min-cost / max-return problem) rather than to
maximize points
([Breiter & Carlin 1997](https://doi.org/10.1080/09332480.1997.10554789)). Niemi, Carlin &
Alexander (2008) operationalize the contrarian recipe: because most pools use simple scoring
that does not reward upsets, the expected-points-maximizing sheet "has too much in common with
many other players' sheets to be profitable," so pick an *undervalued champion* (e.g. the 3rd
or 4th #1-seed, or a top #2) and complete the bracket with published odds, demonstrating
improved ROI
([Niemi, Carlin & Alexander 2008](https://doi.org/10.1080/09332480.2008.10722884); thesis
basis [Niemi 2005](https://www.semanticscholar.org/paper/a7953d248fe4b5fc90e713a74341f810db8b56cb)).
*Caveat:* the champion term dominates the score precisely *because* of progressive scoring (F6),
so the "differentiate on the champion" lever and the "round-doubling concentrates on the
champion" effect are two faces of one structural fact.

**F6 — Round-progressive ("doubling") scoring is the dominant real-world structure, and it
concentrates the score on the late rounds / the champion.** The de-facto standard is ESPN
Tournament Challenge: a round-*r* correct pick scores **10·2^(r−1)** (10,20,40,80,160,320 across
rounds 1–6 in the era Clair & Letscher used; pure 2^(r−1) = 1,2,4,8,16,32 in the normalized
form). Two peer-reviewed/preprint reviews state the structure and its consequence:
[Brill, Wyner & Barnett (2024), *Entropy* (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354004/)
define ESPN score as "10·2^(rd−1) points … in round rd ∈ {1,…,6}" with equal max points (320)
per round; [Decary et al. (2024), "The Madness of Multiple Entries"](https://arxiv.org/html/2407.13438v1)
use 2^(r−1) (max 192 over a single bracket). The doubling makes each later correct pick worth
as much as all earlier picks combined; combined with the champion being the unique team that
survives every round, the realized winning entry is overwhelmingly a function of who called the
title team — the bracket-pool analogue of our "referendum on the champion." (Neither review
uses that exact phrase; the concentration is stated structurally, and we connect it to our
`champion_dominance` finding, where geometric → 0.999.)

**F7 — Scoring-rule taxonomy and the upset-reward dial.** The design space documented across
the literature and practitioner tools:
(i) **constant / count-correct** — every correct pick = 1 (Kaplan & Garstka's simplest case;
flattest; our *linear*-like end);
(ii) **round-multiplier / progressive** — points scale with round, geometric 2^(r−1) (ESPN) or
gentler triangular / Fibonacci (1-3-6-10-15-21, 2-3-5-8-13-21) seen in practice;
(iii) **seed-value (upset-rewarding)** — points = (seed of correctly-picked winner) × round
factor: CBS Sportsline gave a correct #1-seed in round 4 **8 points** but a correct #8-seed
**64 points** ([Horner 2001, "Modeling Madness," INFORMS *OR/MS Today* 28(1)](https://www.informs.org/ORMS-Today/Archived-Issues/2001/orms-2-01/Modeling-Madness),
reporting on Kaplan & Garstka 2001);
(iv) **seed-difference / underdog bonus** — extra points proportional to (loser seed − winner
seed), or a flat upset bonus
([Poologic calculator help](https://www.poologic.com/pchelp.htm), which implements seed,
underdog, useed, upset-bonus, and bonus variants and cites Breiter & Carlin and Kaplan &
Garstka). The key dial: **only seed-based / upset-reward terms make the expected-score-optimal
entry diverge from chalk endogenously**; pure round-multipliers leave it chalk and push all
contrarian burden onto the player's deviation (F1, F5).

**F8 — Ties: the literature largely sidesteps them by design (winner-take-all + an independent
tiebreaker), rather than tuning the ladder to avoid them.** Clair & Letscher state their
analysis is "for a winner-take-all pool with a tiebreaker that is reasonably independent from
the game picks. For example, many football pools break ties with predictions about scoring in
the Monday night game"
([Clair & Letscher 2007](https://doi.org/10.1287/opre.1070.0448)). The standard practitioner
remedy is an exogenous numeric tiebreaker (predicted total points of the final), not a steeper
ladder. This is evidence *against* using convexity as a tie-breaking instrument when an
explicit tiebreaker is available.

## 3. Quantitative / structural design principles relevant to us

1. **Objective = win, not score.** Maximizing expected pool points is provably suboptimal for
   winning a competitive pool (F1, F5). Our objective W = (n−1)·P(1st)+P(2nd) is already the
   correct target; the literature endorses it and explains *why* EV-of-score is a trap (it
   maximizes correlation with the field).
2. **The exploitable quantity is correlation-with-opponents, and it is governed by the scoring
   rule.** Clair & Letscher's 798.8/0.15 vs 32.7/0.37 result (F4) shows return rises as
   score-to-field correlation falls. A scoring rule that forces all entries to load on the same
   dominant term (the champion) maximizes that correlation. Our `champion_dominance` metric is
   the direct in-engine measurement of this; **steepening the ladder is equivalent to raising
   the very correlation the OR literature says destroys edge.**
3. **Contrarianism scales with N; favorites are optimal at small N (eq. 3.1, F3).** The
   favorite/underdog threshold interpolates from a = ½ at N = 1 to a = p as N → ∞. For our pool
   (N = 6–8 → 5–7 opponents), the threshold sits near the small-N end: you deviate from a
   favorite only if it is *heavily* over-owned. Greedy favorite-loading is therefore expected
   to be close to optimal, and only weakly exploitable, in our size class — which matches our
   out-of-sample result that one-ply greedy best-response did **not** beat EV-greedy
   ([recommendation](../tables/recommendation_2026-06-08.md), margins ≈ the between-draw noise
   scale ~0.011).
4. **The upset-reward dial (seed-based scoring) is the only structural lever that makes "take
   the favorites" endogenously suboptimal (F7).** Pure round-multipliers (linear/triangular/
   geometric — our three ladders) all leave EV-greedy = chalk; they shift skill/luck and
   concentration but do **not** create an internal incentive to draft upsets. If we wanted to
   *reward* good upset-calling rather than mere advancement, we would need a seed/strength-gap
   term, not a steeper round ladder.
5. **Use an exogenous tiebreaker, not convexity, to remove ties (F8).** The field's standard is
   an independent numeric tiebreaker. The report's "most teams reaching the Final" rule is the
   correct instrument; do not pay the slot-equity/concentration cost of a steeper ladder merely
   to suppress `top_tie_rate`.

## 4. Direct implications for our group + progressive-advancement scheme

**Skill / luck.** The literature gives no closed form for our `skill_variance_share`, but the
qualitative direction is consistent: progressive scoring raises the weight of late, rare,
high-information events (deep runs, the title), which increases the between-roster (skill)
signal *and* the variance — Kaplan & Garstka quantify both the mean and the **variance** of
total points precisely because variance is decision-relevant. A **group-stage win/draw layer
adds many low-variance, high-frequency points** (every team plays 3 group games), which is the
opposite of progressive concentration: it should *raise* the within-roster signal floor, shrink
ties, and — by giving every drafter a non-trivial score independent of the champion — *reduce*
champion dominance and improve slot equity. This is a genuinely new lever the bracket
literature does not study (their pools score only knockout games), so we cannot import a number;
flag for direct simulation.

**Concentration ("referendum on the champion").** F4/F6 give the mechanism: concentration =
high score-to-field correlation = the champion term dominating every roster. **Progressive
knockout-advancement points that double per round will, on their own, recreate the
geometric-ladder concentration** (`p_champion_holder_wins` → ~1) and the literature predicts
this is *bad for competitive balance* (it makes favorite-loading the only viable play and
collapses the pool onto one event). The group-stage layer is the natural decorrelator: it puts
a meaningful, champion-independent component into each score. Design the two layers so the
group-stage mass is large enough to keep `champion_dominance` off its ceiling — the analogue of
Clair & Letscher's "keep correlation low."

**Ties.** F8: add an explicit independent tiebreaker; do not steepen to suppress ties. A dense
group-stage layer (many distinct attainable totals) will itself cut `top_tie_rate` far more
cheaply than convexity, the same way the linear ladder's tie problem (0.186) is a *flatness*
artifact, not an argument for geometric.

**Exploitability of greedy "take-the-favorites."** Strongly corroborated. Clair & Letscher
(F3) prove favorites are optimal at small N; Metrick (F2) shows the crowd over-backs favorites
(an edge that only pays in large pools); our engine's out-of-sample best-response test found
EV-greedy is at most marginally exploitable
([recommendation](../tables/recommendation_2026-06-08.md)). **In a 6–8-person draft, greedy
EV-maximization is near-optimal and the contrarian literature's gains do not transfer** —
because (i) N is small (threshold near ½→p only barely shifts) and (ii) a snake *draft* removes
the duplication problem that drives contrarian value in *prediction* pools (each team is owned
by exactly one drafter, so two entries cannot collide on the same champion). This second point
is a structural disanalogy worth stating: the entire Clair–Letscher / Niemi mechanism rests on
many entrants picking the *same* favorite; in a draft, the favorite is owned once, so the
"crowd avoidance" edge is mechanically absent. The residual strategic content is purely the
snake-slot/EV tradeoff our engine already models.

## 5. Open questions / residual uncertainty

- **No literature on draft (auction/snake) pools over a tournament.** All five papers concern
  *prediction* pools (everyone forecasts the same bracket; entries collide). Our setting —
  exclusive ownership via a snake draft — voids the duplication mechanism that drives every
  contrarian result. Transfer of F3–F5 is therefore *qualitative*, not quantitative; the slot
  equity and champion-dominance numbers must come from our own Monte Carlo.
- **Group-stage win/draw scoring is unstudied here.** Bracket pools score knockout games only.
  The decorrelating effect of a high-frequency group layer on skill/concentration/ties is a
  prediction from first principles (F4 mechanism), not a cited result. Needs direct simulation
  in the engine.
- **No published number maps "ladder convexity" → skill_variance_share or slot equity.** The
  OR literature optimizes ROI given a fixed scoring rule; it does not *design* the rule against
  a multi-objective skill/equity/tie/concentration frontier as we do. Our frontier is, as far
  as this search found, novel.
- **Author-attribution discrepancy (resolved, flagged):** the brief cited "Niemi, Carlin &
  Sekhon (2008)." CrossRef lists the *CHANCE* 21(1):35–42 paper's authors as **Niemi, Carlin &
  Alexander**. Cited here as Niemi, Carlin & Alexander; if a distinct Sekhon co-authored work
  exists it was not located.
- **Paywalled primaries (KG, Metrick, Breiter–Carlin, Niemi):** scoring-system specifics and
  numbers for these four are from abstracts, Horner's (2001) secondary *OR/MS Today* feature, two
  citing reviews, and a practitioner tool — not the full texts. The Clair & Letscher quantitative
  claims (eq. 3.1, the 798.8/0.15 vs 32.7/0.37 result, the small-vs-large-N rule) are from the
  full extracted text and are the most reliable. Confidence is correspondingly highest on F3/F4.

## 6. References — [VERIFIED]/[UNVERIFIED]

DOIs field-verified against CrossRef (title, venue, volume/issue/pages, year) on 2026-06-09.

- **[VERIFIED]** Kaplan, E. H., & Garstka, S. J. (2001). *March Madness and the Office Pool.*
  Management Science 47(3):369–382. [doi:10.1287/mnsc.47.3.369.9769](https://doi.org/10.1287/mnsc.47.3.369.9769).
  CrossRef-confirmed. Full text paywalled; claims (F1, F7) from abstract + Horner's (2001)
  secondary INFORMS *OR/MS Today* feature (see below).
- **[VERIFIED]** Clair, B., & Letscher, D. (2007). *Optimal Strategies for Sports Betting Pools.*
  Operations Research 55(6):1163–1177. [doi:10.1287/opre.1070.0448](https://doi.org/10.1287/opre.1070.0448).
  CrossRef-confirmed. **Full text obtained and extracted**; all quotes/numbers (F3, F4, F8,
  eq. 3.1) verbatim from the PDF.
- **[VERIFIED]** Metrick, A. (1996). *March madness? Strategic behavior in NCAA basketball
  tournament betting pools.* Journal of Economic Behavior & Organization 30(2):159–172.
  [doi:10.1016/S0167-2681(96)00855-4](https://doi.org/10.1016/S0167-2681(96)00855-4).
  CrossRef-confirmed. Claim (F2) from abstract.
- **[VERIFIED]** Breiter, D. J., & Carlin, B. P. (1997). *How to Play Office Pools If You Must.*
  CHANCE 10(1):5–11. [doi:10.1080/09332480.1997.10554789](https://doi.org/10.1080/09332480.1997.10554789).
  CrossRef-confirmed. Claim (F5) from abstract + Poologic citation.
- **[VERIFIED]** Niemi, J. B., Carlin, B. P., & Alexander, J. M. (2008). *Contrarian Strategies
  for NCAA Tournament Pools: A Cure for March Madness?* CHANCE 21(1):35–42.
  [doi:10.1080/09332480.2008.10722884](https://doi.org/10.1080/09332480.2008.10722884).
  CrossRef-confirmed (authors = Niemi, Carlin, **Alexander**; brief said "Sekhon" — see §5).
  Claim (F5) from abstract + thesis.
- **[VERIFIED]** Brill, R. S., Wyner, A. J., & Barnett, I. J. (2024). *Entropy-Based Strategies
  for Multi-Bracket Pools.* Entropy (Basel) (peer-reviewed; PMC11354004).
  [https://pmc.ncbi.nlm.nih.gov/articles/PMC11354004/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354004/).
  Used for ESPN 10·2^(rd−1) formula and pool-size→entropy/contrarian relationship (F6, F3).
- **[VERIFIED]** Decary, J., Bergman, D., Cardonha, C., Imbrogno, J., & Lodi, A. (2024). *The
  Madness of Multiple Entries in March Madness.* arXiv preprint (not yet peer-reviewed).
  [arXiv:2407.13438](https://arxiv.org/html/2407.13438v1). Used for 2^(r−1) scoring and
  multi-source over-betting consensus (F6, F2). Preprint status noted.
- **[VERIFIED — secondary journalism, evidence tier ~4]** Horner, P. R. (2001).
  *Modeling Madness.* INFORMS *OR/MS Today* 28(1), Feb 2001.
  [https://www.informs.org/ORMS-Today/Archived-Issues/2001/orms-2-01/Modeling-Madness](https://www.informs.org/ORMS-Today/Archived-Issues/2001/orms-2-01/Modeling-Madness).
  Third-party feature by the *OR/MS Today* editor reporting on the primary Kaplan & Garstka (2001)
  *Management Science* paper; source for CBS seed-value scoring example (F7).
- **[VERIFIED — practitioner tool, evidence tier 4]** Poologic Calculator help.
  [https://www.poologic.com/pchelp.htm](https://www.poologic.com/pchelp.htm). Documents the
  implemented scoring-rule taxonomy (seed/underdog/upset-bonus) and cites Breiter–Carlin and
  Kaplan–Garstka (F7). Lower evidence tier; used only for the design taxonomy, not for numbers.
