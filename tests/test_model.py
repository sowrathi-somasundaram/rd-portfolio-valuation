from dataclasses import replace

import numpy as np
import pytest

from rnpv.analysis import (compare_selections, optimal_selection, portfolio_table, portfolio_tornado,
                           simulate_drug, simulate_portfolio, stage_gate_table, tornado)
from rnpv.robustness import run_study, summarize_study
from rnpv.data import DEFAULT_BUDGET, DEFAULT_DRUGS, DEFAULT_RATE
from rnpv.model import ALL_INDICATIONS, BIO_SUCCESS, loa, value

P, R = BIO_SUCCESS, DEFAULT_RATE


def test_loa_from_phase_one_matches_bio():
    generic = replace(DEFAULT_DRUGS[0], benchmark=ALL_INDICATIONS)
    assert loa(generic, P) == pytest.approx(0.079, abs=0.001)


@pytest.mark.parametrize("area, expected", [("Oncology", 0.053), ("Hematology", 0.239), ("Neurology", 0.059),
                                            ("Respiratory", 0.075), ("Vaccine", 0.097)])
def test_disease_area_loa_matches_bio(area, expected):
    # BIO 2011-2020, Figure 5b (and Figure 9 for vaccines): LOA from Phase I by area.
    drug = replace(DEFAULT_DRUGS[0], benchmark=area)
    assert loa(drug, P) == pytest.approx(expected, abs=0.0015)


def test_area_benchmark_ignores_shared_rates():
    onc = DEFAULT_DRUGS[0]
    assert value(onc, P, R).rnpv == pytest.approx(value(onc, (0.9, 0.9, 0.9, 0.9), R).rnpv)


@pytest.mark.parametrize("drug", DEFAULT_DRUGS, ids=lambda d: d.name)
def test_risk_adjustment_lowers_value(drug):
    v = value(drug, P, R)
    assert v.rnpv < v.npv


def test_certain_success_makes_rnpv_equal_npv():
    v = value(replace(DEFAULT_DRUGS[0], benchmark=ALL_INDICATIONS), (1.0, 1.0, 1.0, 1.0), R)
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


def test_stage_gating_never_hurts_and_stops_weak_programmes():
    g = stage_gate_table(DEFAULT_DRUGS, P, R, 20_000, seed=1).set_index("Drug")
    # Same draws, and the gate only stops when continuing has negative expected value.
    se = 400 / np.sqrt(20_000)
    assert (g["Stage-gated"] >= g["Committed"] - 2 * se).all()
    # ONC-117's continuation value is negative before Phase II, so every path that gets there stops.
    assert g.loc["ONC-117", "Stopped at a gate"] == pytest.approx(0.488, abs=0.015)
    assert g.loc["ONC-117", "Value added"] > 0
    # A drug already in Phase III has no later gate with new information worth acting on.
    assert g.loc["RES-402", "Stopped at a gate"] == 0


def test_committed_simulation_unchanged_by_gating():
    a = simulate_portfolio(DEFAULT_DRUGS, P, R, 5_000, seed=3)
    b = stage_gate_table(DEFAULT_DRUGS, P, R, 5_000, seed=3).set_index("Drug")["Committed"]
    assert np.allclose(a.mean().to_numpy(), b.to_numpy())


def test_robustness_study():
    df = run_study(n_portfolios=150, seed=11)
    s = summarize_study(df)
    assert s["win"] + s["tie"] + s["loss"] == pytest.approx(1.0)
    assert s["capture_rnpv_median"] >= s["capture_npv_median"]
    assert s["win"] > s["loss"]
    assert run_study(n_portfolios=20, seed=11).equals(run_study(n_portfolios=20, seed=11))
