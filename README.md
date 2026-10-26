# R&D Portfolio Valuation

**Live app:** https://rd-portfolio-valuation-wswk5umxshbbxrlunf94hi.streamlit.app/

I built this to get hands-on with how pharma companies decide which drug programmes to fund. It takes
five made-up drug candidates at different stages of development, values each one with risk-adjusted
NPV (rNPV), and then works out which ones to fund with a fixed R&D budget.

The basic problem: a drug in Phase I might be worth $1B+ if it gets to market, but only about 8% of
Phase I drugs ever do. Plain NPV ignores that. rNPV weights each year's cash flow by the chance the
drug is still alive at that point, which gives a much more honest number.

## What's in the app

- **rNPV:** pick a drug and see NPV vs rNPV side by side, with the year-by-year cash flows behind them.
- **Scenarios and sensitivity:** pessimistic, base and optimistic cases, plus a tornado chart showing
  which inputs move the value most.
- **Monte Carlo:** 10,000 simulated outcomes where each drug passes or fails each trial phase, peak
  sales vary, and launches can slip. You get the probability of losing money and a P10/P50/P90 range.
- **Portfolio:** drugs ranked by rNPV per R&D dollar, the best combination that fits the budget, and
  checkboxes to try your own picks and see how value and risk change.

Everything is editable in the app: drug inputs, success rates, discount rate, budget and scenarios.

## Assumptions

- Phase success rates come from BIO's *Clinical Development Success Rates 2011–2020* report:
  Phase I→II 52.0%, II→III 28.9%, III→filing 57.8%, filing→approval 90.6%. That works out to 7.9%
  from Phase I to approval.
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
- `rnpv/analysis.py`: scenarios, tornado, Monte Carlo and portfolio selection
- `rnpv/data.py`: the five example drugs
- `tests/`: tests

Built with Python, pandas, NumPy, Plotly and Streamlit.
