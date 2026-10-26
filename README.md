# R&D Portfolio Valuation

**Live app:** https://rd-portfolio-valuation-wswk5umxshbbxrlunf94hi.streamlit.app/

I built this to get hands-on with how pharma companies decide which drug programmes to fund. It takes
five made-up drug candidates at different stages of development, values each one with risk-adjusted
NPV (rNPV), and then works out which ones to fund with a fixed R&D budget.

The basic problem: a drug in Phase I might be worth $1B+ if it gets to market, but only about 8% of
Phase I drugs ever do. Plain NPV ignores that. rNPV weights each year's cash flow by the chance the
drug is still alive at that point, which gives a much more honest number.

## What's in the app

- **Summary:** the recommendation and key findings on one page.
- **rNPV:** pick a drug and see NPV vs rNPV side by side, with the year-by-year cash flows behind them.
- **Scenarios and sensitivity:** pessimistic, base and optimistic cases, plus a tornado chart showing
  which inputs move the value most.
- **Monte Carlo:** 10,000 simulated outcomes where each drug passes or fails each trial phase, peak
  sales vary, and launches can slip. You get the probability of losing money, a P10/P50/P90 range,
  and a comparison of committing to a drug vs reviewing it at each phase gate.
- **Portfolio:** drugs ranked by rNPV per R&D dollar, the best combination that fits the budget, what
  you'd get by ranking on plain NPV instead, and checkboxes to try your own picks.
- **Robustness:** the same ranking comparison on 1,000 random portfolios, to check the result isn't
  a one-off.

Everything is editable in the app: drug inputs, success rates, discount rate, budget and scenarios.

## What I found

With the default inputs ($800M budget, 10% discount rate, BIO disease-area success rates):

- Risk adjustment removes most of the value from early-stage drugs. ONC-117 (oncology, Phase I) is
  worth about $1.25B on plain NPV but −$38M on rNPV.
- Ranking by rNPV per R&D dollar funds RES-402 and NEU-310: $383M of rNPV, with a 44.4% chance
  of losing money. Ranking by plain NPV spends about the same but funds NEU-310, ONC-117 and
  VAC-115, which is worth 64.5% less ($136M), with a 76.8% chance of losing money.
- That isn't specific to these five drugs. Across 901 random portfolios (out of 1,000; the rest
  had nothing worth funding), ranking by rNPV per dollar did better in 65.4%, picked the same drugs
  in 33.6%, and did worse in 1.0%. It captured a median 100% of the best achievable value, vs 89.5%
  for plain NPV.
- Peak sales is the biggest driver of portfolio value (a ±30% change swings it by $861M), with
  launch timing close behind ($825M for ±1 year).
- Reviewing programmes at each phase gate mostly matters for weak ones: it stops ONC-117 on 49% of
  paths. For strong drugs that clear Phase II, carrying on is almost always worth it.

## Assumptions

- Phase success rates come from BIO's *Clinical Development Success Rates 2011–2020* report, using
  each drug's disease area (oncology, neurology and so on). The all-indications average is
  Phase I→II 52.0%, II→III 28.9%, III→filing 57.8%, filing→approval 90.6%, or 7.9% from Phase I to
  approval.
- 10% discount rate, annual nominal cash flows, starting in 2026.
- Sales ramp up linearly to peak, then drop off each year after the patent expires.
- In the simulation, peak sales are lognormal around the base case and launches can slip 1–2 years.
  Drugs are treated as independent.
- The budget is compared with each drug's expected R&D spend to approval (weighted by the chance of
  reaching each phase).

The drugs are hypothetical, and the costs and sales figures are rough industry-typical numbers.
[METHODOLOGY.md](METHODOLOGY.md) has the full details and what I'd improve.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests:

```bash
pip install pytest
pytest -q
```

One of the tests checks that the average of the Monte Carlo runs matches the rNPV formula, which was
a useful way to catch mistakes in either one.

## Code

- `app.py`: the Streamlit app
- `rnpv/model.py`: cash flows, NPV and rNPV
- `rnpv/analysis.py`: scenarios, tornado, Monte Carlo, stage-gating and portfolio selection
- `rnpv/robustness.py`: the random-portfolio study (`data/robustness_study.csv` holds its saved results)
- `rnpv/data.py`: the five example drugs
- `tests/`: tests

Built with Python, pandas, NumPy, Plotly and Streamlit.
