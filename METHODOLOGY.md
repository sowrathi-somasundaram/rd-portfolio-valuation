### Model

**Timeline.** Each candidate runs from its current phase through Phase I → II → III → Filing, with a
duration and total cost per phase. Year 0 is 2026, and cash flows are annual and discounted at
(1 + r)^t.

**Sales.** After approval, sales ramp linearly to peak over *years to peak*. From the patent expiry
year on, sales fall by the *LOE erosion* share each year. Operating profit = sales × operating margin,
where the margin is net of COGS, SG&A and tax.

**Success rates.** Each candidate is benchmarked to a BIO disease area and uses that area's
phase-transition rates (Figure 2 of the report; vaccines use the modality table, Figure 9). For the
defaults: ONC-117 uses Oncology, IMM-204 Autoimmune, NEU-310 Neurology, RES-402 Respiratory and
VAC-115 Vaccine. A candidate benchmarked to "All indications" uses the sidebar rates instead.

**NPV vs rNPV.** Plain NPV assumes every phase succeeds. rNPV multiplies each year's cash flow by the
probability the drug is still alive then:

- costs in phase *k*: P(reach phase *k*) = product of the earlier phases' success rates
- sales: P(approval) = product of all remaining success rates (the LOA)

**Scenarios.** These are multipliers on peak sales, success rates and R&D cost, plus a launch delay.
The discount rate stays fixed, since it's a hurdle rate, not a scenario. You can edit the scenarios
on their tab.

**Tornado.** One-at-a-time sensitivity: each input is moved to its low and high value while the
others stay at base.

**Monte Carlo.** For each run and each drug:

1. Draw pass/fail for each remaining phase (Bernoulli at the phase success rate). Costs accrue up to
   and including the phase that fails.
2. Draw peak sales from a lognormal with mean = base peak sales and the chosen coefficient of variation.
3. Draw a launch delay: 0 years with probability 1 − *q*, 1 year with 0.7*q*, 2 years with 0.3*q*.
   Delays extend Phase III at the same annual cost, and patent expiry doesn't move, so a delay
   shortens the exclusivity period.

A run's value is the plain NPV of what happened. The mean over runs equals rNPV when there are no
delays, which the test suite checks. Drugs are independent, and the same simulated runs are reused
for every portfolio so comparisons aren't distorted by noise.

**Stage-gating.** The same simulated runs are also valued under a second policy. Before each later
phase starts, the expected value of continuing is re-checked, and the programme is stopped if it's
negative. The Phase II readout is treated as proof of concept: from the Phase III gate on, the
decision knows the drug's true peak-sales draw, while earlier gates use the forecast. Launch timing
isn't known when deciding, so costs and profits are averaged over the delay distribution.

**Prioritisation.** Candidates are ranked by rNPV ÷ risk-adjusted PV of R&D cost (a profitability
index). The budget is checked against *expected* R&D spend to approval (probability-weighted,
undiscounted). The recommended set is the one with the highest total rNPV that fits the budget,
found by checking every combination. When funding in rank order would choose differently, the app
shows both. The app also shows what happens if the same rule ranks by plain NPV instead.

**Robustness study.** To check that the ranking result isn't specific to the five defaults, the app
generates 1,000 random five-drug portfolios (fixed seed). Each drug gets a random phase (I, II or
III), a random BIO disease area, phase durations (1–2, 2–3 and 3–4 years), phase costs ($20–60M,
$60–200M, $250–800M, $5–15M), lognormal peak sales (median $1B, clipped to $150M–$5B), 4–7 years to
peak, a 30–45% margin, 30–80% post-patent erosion and 7–15 years of exclusivity. The budget is a
random 30–70% of the cost of funding everything. Portfolios where no drug has positive rNPV are
excluded. These ranges are illustrative and bracket the default candidates.

### Simplifications and what I'd improve

- Success rates vary by disease area but not by modality, target or company. The BIO numbers are
  industry averages from 2011–2020.
- Drug outcomes are independent. In reality, programmes sharing a target or platform are correlated.
- The budget is one pot covering spend to approval, and portfolio selection uses committed values.
  Stage-gating is valued separately and only learns about peak sales. In reality a Phase II readout
  also changes the odds of Phase III success, which would make gates worth more.
- There are no launch costs, working capital or terminal value beyond the erosion tail.

### Sources

- Thomas, D. et al. *Clinical Development Success Rates and Contributing Factors 2011–2020.*
  BIO, Informa Pharma Intelligence & QLS Advisors, February 2021. Phase transition rates across all
  indications: Phase I → II 52.0%, II → III 28.9%, III → NDA/BLA 57.8%, NDA/BLA → approval 90.6%;
  likelihood of approval from Phase I 7.9%. Disease-area rates from Figure 2, vaccine rates from
  Figure 9.
- Stewart, J. J., Allison, P. N. & Johnson, R. S. "Putting a price on biotechnology." *Nature
  Biotechnology* 19, 813–817 (2001). This is the standard reference for the rNPV method.

All five candidates are hypothetical. Costs, durations and sales are illustrative and sized to
typical industry ranges.
