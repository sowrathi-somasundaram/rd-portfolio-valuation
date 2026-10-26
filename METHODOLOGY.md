### Model

**Timeline.** Each candidate runs from its current phase through Phase I → II → III → Filing, with a
duration and total cost per phase. Year 0 is 2026, and cash flows are annual and discounted at
(1 + r)^t.

**Sales.** After approval, sales ramp linearly to peak over *years to peak*. From the patent expiry
year on, sales fall by the *LOE erosion* share each year. Operating profit = sales × operating margin,
where the margin is net of COGS, SG&A and tax.

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

**Prioritisation.** Candidates are ranked by rNPV ÷ risk-adjusted PV of R&D cost (a profitability
index). The budget is checked against *expected* R&D spend to approval (probability-weighted,
undiscounted). The recommended set is the one with the highest total rNPV that fits the budget,
found by checking every combination. When funding in rank order would choose differently, the app
shows both.

### Simplifications and what I'd improve

- One success rate per phase for all candidates. Real rates vary a lot by therapy area (oncology is
  well below average, haematology and vaccines are above), and you can override them in the sidebar.
- Drug outcomes are independent. In reality, programmes sharing a target or platform are correlated.
- The budget is one pot covering spend to approval. Real portfolios fund phase by phase, and a
  Phase II readout is a decision point (option value) that this model doesn't capture.
- There are no launch costs, working capital or terminal value beyond the erosion tail.

### Sources

- Thomas, D. et al. *Clinical Development Success Rates and Contributing Factors 2011–2020.*
  BIO, Informa Pharma Intelligence & QLS Advisors, February 2021. Phase transition rates across all
  indications: Phase I → II 52.0%, II → III 28.9%, III → NDA/BLA 57.8%, NDA/BLA → approval 90.6%;
  likelihood of approval from Phase I 7.9%.
- Stewart, J. J., Allison, P. N. & Johnson, R. S. "Putting a price on biotechnology." *Nature
  Biotechnology* 19, 813–817 (2001). This is the standard reference for the rNPV method.

All five candidates are hypothetical. Costs, durations and sales are illustrative and sized to
typical industry ranges.
