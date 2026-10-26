"""Does ranking by rNPV per R&D dollar beat ranking by plain NPV in general, or only for the
five default candidates? Generate many random portfolios and compare the two rules on each.

Input ranges are illustrative and chosen to bracket the default candidates; success rates
come from the BIO disease-area tables. Everything is deterministic for a given seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import greedy_selection, optimal_selection, portfolio_table
from .model import ALL_INDICATIONS, BIO_BY_AREA, BIO_SUCCESS, PHASES, START_YEAR, Drug

AREAS = [a for a in BIO_BY_AREA if a != ALL_INDICATIONS]


def random_drug(rng: np.random.Generator, name: str) -> Drug:
    phase = PHASES[rng.choice(3, p=[0.40, 0.35, 0.25])]          # Phase I, II or III
    durations = (int(rng.integers(1, 3)), int(rng.integers(2, 4)), int(rng.integers(3, 5)), 1)
    costs = (rng.uniform(20, 60), rng.uniform(60, 200), rng.uniform(250, 800), rng.uniform(5, 15))
    launch = START_YEAR + sum(durations[PHASES.index(phase):])
    return Drug(name, "", phase, durations=durations, costs=costs,
                peak_sales=float(np.clip(rng.lognormal(np.log(1_000), 0.7), 150, 5_000)),
                years_to_peak=int(rng.integers(4, 8)), patent_expiry=launch + int(rng.integers(7, 16)),
                margin=rng.uniform(0.30, 0.45), loe_erosion=rng.uniform(0.3, 0.8),
                benchmark=AREAS[rng.integers(len(AREAS))])


def run_study(n_portfolios: int = 1_000, n_drugs: int = 5, rate: float = 0.10, seed: int = 2026) -> pd.DataFrame:
    """One row per random portfolio: the budget, and the total rNPV each selection rule achieves.

    The budget is a random 30–70% of what funding every candidate would cost (in expected
    R&D spend), so the constraint always binds.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_portfolios):
        drugs = [random_drug(rng, f"D{k + 1}") for k in range(n_drugs)]
        table = portfolio_table(drugs, BIO_SUCCESS, rate)
        frac = rng.uniform(0.3, 0.7)
        budget = frac * table["Expected R&D spend"].sum()
        rnpv = table.set_index("Drug")["rNPV"]
        npv_pick = greedy_selection(table, budget, rank_by="NPV")
        rows.append({
            "Budget share": frac,
            "Best possible": rnpv[optimal_selection(table, budget)].sum(),
            "Rank by rNPV per R&D $": rnpv[greedy_selection(table, budget)].sum(),
            "Rank by plain NPV": rnpv[npv_pick].sum(),
            "NPV pick funds a value-destroying drug": bool((rnpv[npv_pick] < 0).any()),
        })
    return pd.DataFrame(rows)


def summarize_study(df: pd.DataFrame) -> dict:
    """Headline statistics. Value captured = a rule's total rNPV ÷ the best possible total."""
    ok = df[df["Best possible"] > 0]
    a, b = ok["Rank by rNPV per R&D $"], ok["Rank by plain NPV"]
    diff = a - b
    return {
        "n": len(ok),
        "win": float((diff > 1e-9).mean()),
        "tie": float((diff.abs() <= 1e-9).mean()),
        "loss": float((diff < -1e-9).mean()),
        "capture_rnpv_median": float((a / ok["Best possible"]).median()),
        "capture_npv_median": float((b / ok["Best possible"]).median()),
        "capture_rnpv_mean": float((a / ok["Best possible"]).mean()),
        "capture_npv_mean": float((b / ok["Best possible"]).mean()),
        "uplift_median": float(diff.median()),
        "uplift_mean": float(diff.mean()),
        "rnpv_rank_is_optimal": float(((ok["Best possible"] - a).abs() <= 1e-9).mean()),
        "npv_funds_destroyer": float(ok["NPV pick funds a value-destroying drug"].mean()),
    }
