"""Scenario analysis, tornado sensitivity, Monte Carlo and portfolio selection."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd

from .model import PHASES, Drug, value, pv_components

# ---------------------------------------------------------------- scenarios

SCENARIOS = pd.DataFrame(
    {
        "Peak sales ×": [0.70, 1.00, 1.25],
        "Success rates ×": [0.85, 1.00, 1.10],
        "R&D cost ×": [1.20, 1.00, 0.90],
        "Launch delay (yrs)": [1, 0, 0],
    },
    index=["Pessimistic", "Base", "Optimistic"],
)


def scale_p(p: tuple, mult: float, phases=None) -> tuple:
    """Multiply success rates (optionally only some phases), capped at 99%."""
    return tuple(min(0.99, x * mult) if phases is None or i in phases else x
                 for i, x in enumerate(p))


def scenario_values(drug: Drug, p: tuple, rate: float, scenarios: pd.DataFrame = SCENARIOS) -> pd.DataFrame:
    rows = []
    for name, s in scenarios.iterrows():
        d = drug.scaled(peak=s["Peak sales ×"], cost=s["R&D cost ×"])
        v = value(d, scale_p(p, s["Success rates ×"]), rate, delay=int(s["Launch delay (yrs)"]))
        rows.append({"Scenario": name, "NPV": v.npv, "rNPV": v.rnpv, "LOA": v.loa})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tornado

def tornado(drug: Drug, p: tuple, rate: float) -> tuple[float, pd.DataFrame]:
    """One-at-a-time sensitivity. Returns (base rNPV, table sorted by swing)."""
    base = value(drug, p, rate).rnpv
    cases = [
        ("Peak sales", "−30%", "+30%",
         lambda: value(drug.scaled(peak=0.7), p, rate), lambda: value(drug.scaled(peak=1.3), p, rate)),
        ("Discount rate", f"{rate + 0.02:.0%}", f"{rate - 0.02:.0%}",
         lambda: value(drug, p, rate + 0.02), lambda: value(drug, p, max(rate - 0.02, 0.0))),
        ("R&D cost", "+25%", "−25%",
         lambda: value(drug.scaled(cost=1.25), p, rate), lambda: value(drug.scaled(cost=0.75), p, rate)),
        ("Operating margin", f"{drug.margin - 0.10:.0%}", f"{drug.margin + 0.10:.0%}",
         lambda: value(replace(drug, margin=drug.margin - 0.10), p, rate),
         lambda: value(replace(drug, margin=drug.margin + 0.10), p, rate)),
        ("Patent expiry", f"{drug.patent_expiry - 2}", f"{drug.patent_expiry + 2}",
         lambda: value(replace(drug, patent_expiry=drug.patent_expiry - 2), p, rate),
         lambda: value(replace(drug, patent_expiry=drug.patent_expiry + 2), p, rate)),
        ("Launch timing", "1 yr late", "1 yr early",
         lambda: value(drug, p, rate, delay=1), lambda: value(drug, p, rate, delay=-1)),
        ("Years to peak", f"{drug.years_to_peak + 2}", f"{max(drug.years_to_peak - 2, 1)}",
         lambda: value(replace(drug, years_to_peak=drug.years_to_peak + 2), p, rate),
         lambda: value(replace(drug, years_to_peak=max(drug.years_to_peak - 2, 1)), p, rate)),
    ]
    for i in range(drug.phase_idx, len(PHASES)):
        lo, hi = scale_p(p, 0.75, {i}), scale_p(p, 1.25, {i})
        cases.append((f"{PHASES[i]} success rate", f"{lo[i]:.0%}", f"{hi[i]:.0%}",
                      lambda lo=lo: value(drug, lo, rate), lambda hi=hi: value(drug, hi, rate)))

    rows = []
    for name, lo_lbl, hi_lbl, lo_fn, hi_fn in cases:
        lo, hi = lo_fn().rnpv, hi_fn().rnpv
        rows.append({"Input": name, "Downside": lo_lbl, "Upside": hi_lbl,
                     "rNPV downside": lo, "rNPV upside": hi, "Swing": abs(hi - lo)})
    return base, pd.DataFrame(rows).sort_values("Swing", ascending=False, ignore_index=True)


def portfolio_tornado(drugs: list[Drug], p: tuple, rate: float) -> tuple[float, pd.DataFrame]:
    """Tornado on total portfolio rNPV: each input moves for every drug at once.

    rNPV is additive across drugs, so the portfolio swing is the sum of each drug's change.
    A phase success rate only affects drugs that still have that phase ahead of them.
    """
    base, deltas = 0.0, {}
    for d in drugs:
        b, t = tornado(d, p, rate)
        base += b
        for _, r in t.iterrows():
            lo, hi = deltas.get(r["Input"], (0.0, 0.0))
            deltas[r["Input"]] = (lo + r["rNPV downside"] - b, hi + r["rNPV upside"] - b)
    rows = [{"Input": k, "rNPV downside": base + lo, "rNPV upside": base + hi, "Swing": abs(hi - lo)}
            for k, (lo, hi) in deltas.items()]
    return base, pd.DataFrame(rows).sort_values("Swing", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- Monte Carlo

DELAYS = np.array([0, 1, 2])


def delay_probs(delay_prob: float) -> np.ndarray:
    """A slip is 1 year 70% of the time, 2 years 30% of the time."""
    return np.array([1 - delay_prob, 0.7 * delay_prob, 0.3 * delay_prob])


def simulate_drug(drug: Drug, p: tuple, rate: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate n development paths for one drug. Each path's NPV is a plain (unrisked)
    NPV of what actually happened, so the mean across paths approximates rNPV."""
    probs = np.asarray(p[drug.phase_idx:])
    m = len(probs)
    failed = rng.random((n, m)) >= probs
    stage = np.where(failed.any(axis=1), failed.argmax(axis=1), m)    # m means approved
    approved = stage == m

    delay = rng.choice(DELAYS, size=n, p=delay_probs(drug.delay_prob))
    if drug.peak_cv > 0:
        sigma = np.sqrt(np.log1p(drug.peak_cv ** 2))
        peak = rng.lognormal(np.log(drug.peak_sales) - sigma ** 2 / 2, sigma, n)   # mean = peak_sales
    else:
        peak = np.full(n, drug.peak_sales)

    comps = [pv_components(drug, p, rate, int(d)) for d in DELAYS]
    cum_cost = np.array([np.cumsum(c) for c, _ in comps])           # (delays, phases)
    unit = np.array([u for _, u in comps])
    cost = cum_cost[delay, np.minimum(stage, m - 1)]
    npv = -cost + approved * peak * unit[delay]
    return pd.DataFrame({"npv": npv, "approved": approved, "stage": stage,
                         "peak": np.where(approved, peak, np.nan), "delay": delay})


def simulate_portfolio(drugs: list[Drug], p: tuple, rate: float, n: int = 10_000, seed: int = 7) -> pd.DataFrame:
    """NPV per path for every drug (columns = drug names). Drugs are independent."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({d.name: simulate_drug(d, p, rate, n, rng)["npv"].to_numpy() for d in drugs})


def summarize(values: np.ndarray) -> dict:
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"mean": float(values.mean()), "std": float(values.std()), "p10": p10, "p50": p50,
            "p90": p90, "p_loss": float((values < 0).mean())}


# ---------------------------------------------------------------- portfolio

def portfolio_table(drugs: list[Drug], p: tuple, rate: float) -> pd.DataFrame:
    rows = []
    for d in drugs:
        v = value(d, p, rate)
        rows.append({"Drug": d.name, "Area": d.area, "Phase": d.phase, "NPV": v.npv, "rNPV": v.rnpv,
                     "LOA": v.loa, "Launch": v.launch_year, "Expected R&D spend": v.exp_rd_spend,
                     "rNPV per R&D $": v.rnpv / v.risk_pv_rd if v.risk_pv_rd else np.nan})
    df = pd.DataFrame(rows).sort_values("rNPV per R&D $", ascending=False, ignore_index=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def greedy_selection(table: pd.DataFrame, budget: float, rank_by: str = "rNPV per R&D $") -> list[str]:
    """Fund in order of ``rank_by`` (highest first), skipping anything that would break the
    budget or that the ranking metric itself says destroys value."""
    chosen, spent = [], 0.0
    for _, r in table.sort_values(rank_by, ascending=False).iterrows():
        if r[rank_by] > 0 and spent + r["Expected R&D spend"] <= budget:
            chosen.append(r["Drug"])
            spent += r["Expected R&D spend"]
    return chosen


# How each selection approach ranks candidates before funding them within the budget.
SELECTION_METHODS = {"Rank by rNPV per R&D $": "rNPV per R&D $", "Rank by plain NPV": "NPV"}


def compare_selections(drugs: list[Drug], p: tuple, rate: float, budget: float,
                       sims: pd.DataFrame) -> pd.DataFrame:
    """Run the same budget-constrained selection under each ranking in SELECTION_METHODS and
    score every resulting portfolio on risk-adjusted value and simulated outcomes.

    ``sims`` is the output of ``simulate_portfolio``; reusing one set of simulated paths keeps
    the comparison free of sampling noise between methods.
    """
    table = portfolio_table(drugs, p, rate).set_index("Drug")
    rows = []
    for method, col in SELECTION_METHODS.items():
        chosen = greedy_selection(table.reset_index(), budget, rank_by=col)
        s = summarize(sims[chosen].sum(axis=1).to_numpy() if chosen else np.zeros(len(sims)))
        rows.append({"Method": method, "Drugs funded": chosen,
                     "Total NPV": table.loc[chosen, "NPV"].sum(),
                     "Total rNPV": table.loc[chosen, "rNPV"].sum(),
                     "Expected R&D spend": table.loc[chosen, "Expected R&D spend"].sum(),
                     "MC mean": s["mean"], "P(loss)": s["p_loss"],
                     "P10": s["p10"], "P50": s["p50"], "P90": s["p90"]})
    return pd.DataFrame(rows).set_index("Method")


@dataclass
class Subset:
    drugs: tuple
    spend: float
    rnpv: float


def all_subsets(table: pd.DataFrame) -> list[Subset]:
    names = table["Drug"].tolist()
    spend = dict(zip(names, table["Expected R&D spend"]))
    rnpv = dict(zip(names, table["rNPV"]))
    return [Subset(c, sum(spend[x] for x in c), sum(rnpv[x] for x in c))
            for k in range(len(names) + 1) for c in combinations(names, k)]


def optimal_selection(table: pd.DataFrame, budget: float) -> list[str]:
    """Exhaustive search: the affordable set with the highest total rNPV (fine for ≤ 15 drugs)."""
    if len(table) > 15:
        return greedy_selection(table, budget)
    feasible = [s for s in all_subsets(table) if s.spend <= budget]
    return list(max(feasible, key=lambda s: s.rnpv).drugs)
