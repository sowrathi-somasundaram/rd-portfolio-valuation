from dataclasses import replace

import numpy as np
import pytest

from rnpv.analysis import (compare_selections, optimal_selection, portfolio_table, portfolio_tornado,
                           simulate_drug, simulate_portfolio, tornado)
from rnpv.data import DEFAULT_BUDGET, DEFAULT_DRUGS, DEFAULT_RATE
from rnpv.model import BIO_SUCCESS, loa, value

P, R = BIO_SUCCESS, DEFAULT_RATE


def test_loa_from_phase_one_matches_bio():
    assert loa(DEFAULT_DRUGS[0], P) == pytest.approx(0.079, abs=0.001)


@pytest.mark.parametrize("drug", DEFAULT_DRUGS, ids=lambda d: d.name)
def test_risk_adjustment_lowers_value(drug):
    v = value(drug, P, R)
    assert v.rnpv < v.npv


def test_certain_success_makes_rnpv_equal_npv():
    v = value(DEFAULT_DRUGS[0], (1.0, 1.0, 1.0, 1.0), R)
    assert v.rnpv == pytest.approx(v.npv)


@pytest.mark.parametrize("drug", DEFAULT_DRUGS, ids=lambda d: d.name)
def test_monte_carlo_mean_converges_to_rnpv(drug):
    # With no timing risk, E[NPV of a simulated path] is exactly the rNPV.
    d = replace(drug, delay_prob=0.0)
    sims = simulate_drug(d, P, R, 200_000, np.random.default_rng(0))
    expected = value(d, P, R).rnpv
    se = sims["npv"].std() / np.sqrt(len(sims))
    assert abs(sims["npv"].mean() - expected) < 4 * se
    assert sims["approved"].mean() == pytest.approx(loa(d, P), abs=0.005)


def test_tornado_bars_bracket_base():
    base, t = tornado(DEFAULT_DRUGS[1], P, R)
    assert (t["Swing"] > 0).all()
    assert t["Swing"].is_monotonic_decreasing


def test_optimal_selection_respects_budget():
    table = portfolio_table(DEFAULT_DRUGS, P, R)
    for budget in (0, 300, 600, 900, 5000):
        chosen = optimal_selection(table, budget)
        assert table.set_index("Drug").loc[chosen, "Expected R&D spend"].sum() <= budget


def test_npv_ranking_vs_rnpv_ranking():
    sims = simulate_portfolio(DEFAULT_DRUGS, P, R, 10_000, seed=7)
    c = compare_selections(DEFAULT_DRUGS, P, R, DEFAULT_BUDGET, sims)
    rnpv_pick, npv_pick = c.loc["Rank by rNPV per R&D $"], c.loc["Rank by plain NPV"]

    # Both respect the same budget.
    assert (c["Expected R&D spend"] <= DEFAULT_BUDGET).all()
    # Ranking by risk-adjusted efficiency finds the true optimum here; ranking by NPV does not.
    table = portfolio_table(DEFAULT_DRUGS, P, R)
    assert set(rnpv_pick["Drugs funded"]) == set(optimal_selection(table, DEFAULT_BUDGET))
    assert rnpv_pick["Total rNPV"] > npv_pick["Total rNPV"]
    # NPV ranking looks better unrisked but is worse once risk is priced in.
    assert npv_pick["Total NPV"] > rnpv_pick["Total NPV"]
    assert npv_pick["P(loss)"] > rnpv_pick["P(loss)"]
    # Plain NPV can't see that ONC-117 destroys value after risk adjustment.
    assert "ONC-117" in npv_pick["Drugs funded"] and "ONC-117" not in rnpv_pick["Drugs funded"]


def test_comparison_is_reproducible_with_fixed_seed():
    a = compare_selections(DEFAULT_DRUGS, P, R, DEFAULT_BUDGET, simulate_portfolio(DEFAULT_DRUGS, P, R, 10_000, seed=7))
    b = compare_selections(DEFAULT_DRUGS, P, R, DEFAULT_BUDGET, simulate_portfolio(DEFAULT_DRUGS, P, R, 10_000, seed=7))
    assert a.drop(columns="Drugs funded").equals(b.drop(columns="Drugs funded"))


def test_portfolio_tornado_is_sum_of_drug_tornados():
    base, t = portfolio_tornado(DEFAULT_DRUGS, P, R)
    assert base == pytest.approx(sum(value(d, P, R).rnpv for d in DEFAULT_DRUGS))
    peak = t.set_index("Input").loc["Peak sales"]
    expected = sum(tornado(d, P, R)[1].set_index("Input").loc["Peak sales", "Swing"] for d in DEFAULT_DRUGS)
    assert peak["Swing"] == pytest.approx(expected)
