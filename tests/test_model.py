from dataclasses import replace

import numpy as np
import pytest

from rnpv.analysis import optimal_selection, portfolio_table, simulate_drug, tornado
from rnpv.data import DEFAULT_DRUGS, DEFAULT_RATE
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
