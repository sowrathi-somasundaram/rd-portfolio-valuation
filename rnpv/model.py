"""Core rNPV model for a single drug candidate.

Conventions
-----------
* Money is in $ millions, nominal.
* Year index ``t = 0`` is ``START_YEAR``; a cash flow in year t is discounted by (1 + r) ** t.
* Development runs phase by phase from the drug's *current* phase. R&D cost is spread
  evenly over each phase's duration.
* rNPV weights every cash flow by the probability the drug is still alive when that
  cash flow happens: phase-k costs by P(reach phase k), commercial cash flows by the
  probability of approval (likelihood of approval, LOA).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

START_YEAR = 2026
PHASES = ("Phase I", "Phase II", "Phase III", "Filing")

# BIO / Informa Pharma Intelligence / QLS, "Clinical Development Success Rates 2011-2020"
# (Feb 2021), all indications. Phase-transition probabilities; LOA from Phase I = 7.9%.
BIO_SUCCESS = (0.520, 0.289, 0.578, 0.906)

# Sales stop being modelled once post-patent sales fall below this share of peak.
_TAIL_CUTOFF = 0.01
_MAX_YEARS = 45


@dataclass(frozen=True)
class Drug:
    name: str
    area: str
    phase: str                  # current phase, one of PHASES
    durations: tuple            # years per phase, all four phases (completed ones ignored)
    costs: tuple                # total R&D cost per phase, $M
    peak_sales: float           # $M per year at peak
    years_to_peak: int          # linear ramp from launch to peak
    patent_expiry: int          # calendar year of loss of exclusivity
    margin: float               # operating margin on sales (after COGS, SG&A, tax)
    loe_erosion: float          # share of sales lost each year after patent expiry
    peak_cv: float = 0.4        # Monte Carlo: coefficient of variation of peak sales
    delay_prob: float = 0.25    # Monte Carlo: probability of a launch slip

    @property
    def phase_idx(self) -> int:
        return PHASES.index(self.phase)

    @property
    def remaining(self) -> tuple:
        return PHASES[self.phase_idx:]

    def scaled(self, peak: float = 1.0, cost: float = 1.0) -> "Drug":
        return replace(self, peak_sales=self.peak_sales * peak,
                       costs=tuple(c * cost for c in self.costs))


def _delay_index(n_remaining: int) -> int:
    """Delays stretch the last clinical phase (Phase III), or Filing if already filed."""
    return max(n_remaining - 2, 0)


def cash_flows(drug: Drug, p: tuple, rate: float, delay: int = 0) -> pd.DataFrame:
    """Year-by-year cash flow schedule with unrisked and risk-adjusted values.

    ``p`` holds the success probability of each of the four phases. ``delay`` adds (or,
    if negative, removes) years from the last clinical phase at the same annual cost.
    """
    i0 = drug.phase_idx
    durs = [int(d) for d in drug.durations[i0:]]
    costs = [float(c) for c in drug.costs[i0:]]
    j = _delay_index(len(durs))
    annual = costs[j] / durs[j]
    durs[j] = max(1, durs[j] + delay)
    costs[j] = annual * durs[j]

    rows = []
    t, reach = 0, 1.0
    for k, (d, c) in enumerate(zip(durs, costs)):
        for _ in range(d):
            rows.append((t, PHASES[i0 + k], -c / d, 0.0, reach))
            t += 1
        reach *= p[i0 + k]
    loa = reach

    # Commercial phase: linear ramp to peak, then erosion after loss of exclusivity.
    launch, sales = t, 0.0
    for t in range(launch, _MAX_YEARS):
        year = START_YEAR + t
        if year < drug.patent_expiry:
            sales = drug.peak_sales * min(1.0, (t - launch + 1) / max(drug.years_to_peak, 1))
        else:
            sales *= 1 - drug.loe_erosion
            if sales < _TAIL_CUTOFF * drug.peak_sales:
                break
        rows.append((t, "Market", 0.0, sales, loa))

    df = pd.DataFrame(rows, columns=["t", "stage", "rd_cost", "revenue", "p_alive"])
    df["year"] = START_YEAR + df["t"]
    df["op_profit"] = df["revenue"] * drug.margin
    df["cash_flow"] = df["rd_cost"] + df["op_profit"]
    df["risk_cash_flow"] = df["cash_flow"] * df["p_alive"]
    df["discount"] = (1 + rate) ** -df["t"]
    df["pv"] = df["cash_flow"] * df["discount"]
    df["risk_pv"] = df["risk_cash_flow"] * df["discount"]
    return df


@dataclass(frozen=True)
class Valuation:
    npv: float              # unrisked: assumes every phase succeeds
    rnpv: float             # risk-adjusted
    loa: float              # probability of approval from the current phase
    exp_rd_spend: float     # probability-weighted R&D spend, undiscounted ($M)
    risk_pv_rd: float       # probability-weighted R&D spend, discounted ($M)
    launch_year: int


def loa(drug: Drug, p: tuple) -> float:
    return float(np.prod(p[drug.phase_idx:]))


def value(drug: Drug, p: tuple, rate: float, delay: int = 0) -> Valuation:
    df = cash_flows(drug, p, rate, delay)
    rd = df[df["stage"] != "Market"]
    market = df[df["stage"] == "Market"]
    launch = int(market["year"].iloc[0]) if len(market) else int(df["year"].iloc[-1]) + 1
    return Valuation(
        npv=df["pv"].sum(),
        rnpv=df["risk_pv"].sum(),
        loa=loa(drug, p),
        exp_rd_spend=-(rd["rd_cost"] * rd["p_alive"]).sum(),
        risk_pv_rd=-(rd["rd_cost"] * rd["p_alive"] * rd["discount"]).sum(),
        launch_year=launch,
    )


def pv_components(drug: Drug, p: tuple, rate: float, delay: int = 0):
    """Decompose PV into pieces the Monte Carlo can recombine cheaply.

    Returns (PV of R&D cost for each remaining phase, PV of operating profit per $1M of
    peak sales). Revenue is linear in peak sales, so NPV for any peak-sales draw is
    ``-cost + approved * peak * unit``.
    """
    df = cash_flows(drug, p, rate, delay)
    rd = df[df["stage"] != "Market"]
    cost = np.array([-(rd.loc[rd["stage"] == ph, "rd_cost"] * rd.loc[rd["stage"] == ph, "discount"]).sum()
                     for ph in drug.remaining])
    mkt = df[df["stage"] == "Market"]
    unit = (mkt["op_profit"] * mkt["discount"]).sum() / drug.peak_sales if drug.peak_sales else 0.0
    return cost, unit
