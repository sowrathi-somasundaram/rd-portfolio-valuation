"""R&D Portfolio Valuation — Streamlit app.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rnpv.analysis import (SCENARIOS, all_subsets, compare_selections, greedy_selection, optimal_selection,
                           portfolio_table, scenario_values, simulate_portfolio, summarize, tornado)
from rnpv.data import DEFAULT_BUDGET, DEFAULT_DRUGS, DEFAULT_RATE
from rnpv.model import BIO_SUCCESS, PHASES, START_YEAR, Drug, cash_flows, value

st.set_page_config(page_title="R&D Portfolio Valuation", page_icon="💊", layout="wide")

# Colours: categorical slots 1–2 for paired series, a one-hue ramp for ordered scenarios.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
SCENARIO_COLOURS = {"Pessimistic": "#86b6ef", "Base": "#2a78d6", "Optimistic": "#104281"}


def money(x: float) -> str:
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1000:,.2f}B" if x >= 1000 else f"{sign}${x:,.0f}M"


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=10, r=10, t=80, b=10), hovermode="closest",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_layout(**(layout | kw))
    return fig


# ---------------------------------------------------------------- inputs

COLS = ["Name", "Area", "Phase",
        *[f"{ph} yrs" for ph in PHASES], *[f"{ph} cost $M" for ph in PHASES],
        "Peak sales $M", "Years to peak", "Patent expiry", "Op. margin", "LOE erosion",
        "Peak sales CV", "Delay prob."]


def drugs_to_df(drugs: list[Drug]) -> pd.DataFrame:
    return pd.DataFrame([[d.name, d.area, d.phase, *d.durations, *d.costs, d.peak_sales, d.years_to_peak,
                          d.patent_expiry, d.margin, d.loe_erosion, d.peak_cv, d.delay_prob]
                         for d in drugs], columns=COLS)


def df_to_drugs(df: pd.DataFrame) -> list[Drug]:
    out = []
    for _, r in df.dropna(subset=["Name", "Phase"]).iterrows():
        out.append(Drug(str(r["Name"]), str(r["Area"]), r["Phase"],
                        durations=tuple(max(1, int(r[f"{ph} yrs"])) for ph in PHASES),
                        costs=tuple(float(r[f"{ph} cost $M"]) for ph in PHASES),
                        peak_sales=float(r["Peak sales $M"]), years_to_peak=int(r["Years to peak"]),
                        patent_expiry=int(r["Patent expiry"]), margin=float(r["Op. margin"]),
                        loe_erosion=float(r["LOE erosion"]), peak_cv=float(r["Peak sales CV"]),
                        delay_prob=float(r["Delay prob."])))
    return out


if "drug_df" not in st.session_state:
    st.session_state.drug_df = drugs_to_df(DEFAULT_DRUGS)

with st.sidebar:
    st.header("Assumptions")
    rate = st.slider("Discount rate (WACC)", 0.04, 0.16, DEFAULT_RATE, 0.005, format="%.3f")
    budget = st.number_input("R&D budget ($M, expected spend)", 0.0, 10_000.0, DEFAULT_BUDGET, 50.0)
    n_sims = st.select_slider("Monte Carlo runs", [1_000, 5_000, 10_000, 25_000, 50_000], 10_000)
    seed = st.number_input("Random seed", 0, 10_000, 7)

    st.subheader("Phase success rates")
    st.caption("Defaults: BIO / Informa / QLS, *Clinical Development Success Rates 2011–2020*, all indications.")
    p = tuple(st.number_input(f"{ph} → next", 0.01, 1.0, BIO_SUCCESS[i], 0.01, key=f"p{i}")
              for i, ph in enumerate(PHASES))
    st.caption(f"Likelihood of approval from Phase I: **{np.prod(p):.1%}**")

st.title("R&D Portfolio Valuation")
st.caption("Risk-adjusted NPV, scenario and sensitivity analysis, Monte Carlo simulation, and budget-"
           "constrained prioritisation for five hypothetical drug candidates. All figures are nominal $.")

tabs = st.tabs(["Candidates", "1 · rNPV", "2 · Scenarios & sensitivity", "3 · Monte Carlo",
                "4 · Portfolio", "Method & sources"])

with tabs[0]:
    st.markdown("Edit any input; every tab recalculates. Costs are totals per phase, spread evenly over "
                "its duration. Phases before a drug's current phase are ignored.")
    edited = st.data_editor(
        st.session_state.drug_df, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Phase": st.column_config.SelectboxColumn(options=list(PHASES), required=True),
            "Op. margin": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=1.0),
            "LOE erosion": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                         help="Share of sales lost each year after patent expiry"),
            "Peak sales CV": st.column_config.NumberColumn(format="%.2f", min_value=0.0,
                                                           help="Monte Carlo: spread of peak sales (lognormal)"),
            "Delay prob.": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                         help="Monte Carlo: chance of a 1–2 year launch slip"),
        })
    if st.button("Reset to defaults"):
        st.session_state.drug_df = drugs_to_df(DEFAULT_DRUGS)
        st.rerun()

drugs = df_to_drugs(edited)
if not drugs:
    st.warning("Add at least one candidate on the Candidates tab.")
    st.stop()
names = [d.name for d in drugs]
by_name = dict(zip(names, drugs))


@st.cache_data(show_spinner="Running simulations…")
def run_mc(drugs: list[Drug], p: tuple, rate: float, n: int, seed: int) -> pd.DataFrame:
    return simulate_portfolio(drugs, p, rate, n, seed)


sims = run_mc(drugs, p, rate, n_sims, seed)
table = portfolio_table(drugs, p, rate)

# ---------------------------------------------------------------- 1 · rNPV

with tabs[1]:
    drug = by_name[st.selectbox("Candidate", names, key="t1")]
    v = value(drug, p, rate)
    cf = cash_flows(drug, p, rate)

    c = st.columns(4)
    c[0].metric("NPV (unrisked)", money(v.npv), help="Assumes every remaining phase succeeds")
    c[1].metric("rNPV (risk-adjusted)", money(v.rnpv), help="Each cash flow weighted by P(drug still alive)")
    c[2].metric(f"P(approval) from {drug.phase}", f"{v.loa:.1%}")
    c[3].metric("Launch year", v.launch_year, help=f"Patent expiry {drug.patent_expiry}")
    st.caption(f"Risk adjustment removes **{money(v.npv - v.rnpv)}** of value. Patent expiry {drug.patent_expiry} "
               f"leaves {drug.patent_expiry - v.launch_year} years of exclusivity.")

    hover = "%{x}<br>%{customdata[0]}<br>%{y:,.0f} $M<br>P(alive) %{customdata[1]:.1%}<extra>%{fullData.name}</extra>"
    cd = np.column_stack([cf["stage"], cf["p_alive"]])
    fig = go.Figure([
        go.Bar(x=cf["year"], y=cf["cash_flow"], name="Unrisked cash flow", marker_color=BLUE,
               customdata=cd, hovertemplate=hover),
        go.Bar(x=cf["year"], y=cf["risk_cash_flow"], name="Risk-adjusted cash flow", marker_color=ORANGE,
               customdata=cd, hovertemplate=hover),
    ])
    st.plotly_chart(style(fig, title="Annual cash flow ($M)", barmode="group", bargap=0.25), use_container_width=True)

    fig = go.Figure([
        go.Scatter(x=cf["year"], y=cf["pv"].cumsum(), name="Cumulative NPV", line=dict(color=BLUE, width=2),
                   hovertemplate="%{x}: %{y:,.0f} $M<extra>NPV</extra>"),
        go.Scatter(x=cf["year"], y=cf["risk_pv"].cumsum(), name="Cumulative rNPV", line=dict(color=ORANGE, width=2),
                   hovertemplate="%{x}: %{y:,.0f} $M<extra>rNPV</extra>"),
    ])
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    st.plotly_chart(style(fig, title="Cumulative discounted value ($M)", hovermode="x unified"), use_container_width=True)
    st.caption("rNPV weights each year's cash flow by the probability the drug is still alive then: "
               "development costs by P(reaching that phase), sales by P(approval). Early losses are "
               "near-certain, while sales are heavily discounted for risk. That gap is the whole point of rNPV.")

    with st.expander("Year-by-year schedule"):
        show = cf[["year", "stage", "rd_cost", "revenue", "op_profit", "cash_flow", "p_alive",
                   "discount", "pv", "risk_pv"]]
        st.dataframe(show.style.format({c: "{:,.1f}" for c in ["rd_cost", "revenue", "op_profit", "cash_flow",
                                                                "pv", "risk_pv"]}
                                       | {"p_alive": "{:.1%}", "discount": "{:.3f}"}),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 2 · Scenarios & sensitivity

with tabs[2]:
    st.subheader("Scenarios")
    scen = st.data_editor(SCENARIOS, use_container_width=True, key="scen")
    sv = pd.concat([scenario_values(d, p, rate, scen).assign(Drug=d.name) for d in drugs])
    fig = go.Figure([go.Bar(x=g["Drug"], y=g["rNPV"], name=s, marker_color=SCENARIO_COLOURS.get(s, GREY),
                            hovertemplate="%{x}<br>%{y:,.0f} $M<extra>" + s + "</extra>")
                     for s, g in sv.groupby("Scenario", sort=False)])
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    st.plotly_chart(style(fig, title="rNPV by scenario ($M)", barmode="group", bargap=0.25), use_container_width=True)
    pivot = sv.pivot(index="Drug", columns="Scenario", values="rNPV")[list(scen.index)]
    pivot.loc["Portfolio total"] = pivot.sum()
    st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)

    st.subheader("Tornado: what moves rNPV most?")
    drug = by_name[st.selectbox("Candidate", names, key="t2")]
    base, tor = tornado(drug, p, rate)
    tor = tor.iloc[::-1]  # largest swing at the top
    fig = go.Figure([
        go.Bar(y=tor["Input"], x=tor["rNPV downside"] - base, base=base, orientation="h", name="Downside case",
               marker_color=ORANGE, customdata=tor[["Downside", "rNPV downside"]],
               hovertemplate="%{y}: %{customdata[0]}<br>rNPV %{customdata[1]:,.0f} $M<extra></extra>"),
        go.Bar(y=tor["Input"], x=tor["rNPV upside"] - base, base=base, orientation="h", name="Upside case",
               marker_color=BLUE, customdata=tor[["Upside", "rNPV upside"]],
               hovertemplate="%{y}: %{customdata[0]}<br>rNPV %{customdata[1]:,.0f} $M<extra></extra>"),
    ])
    fig.add_vline(x=base, line_color=GREY, line_width=1,
                  annotation_text=f"base {money(base)}", annotation_position="top")
    st.plotly_chart(style(fig, height=440, title=f"{drug.name}: rNPV sensitivity ($M)", barmode="overlay",
                          bargap=0.35), use_container_width=True)
    st.caption("Each input moves alone while the rest stay at base. Ranges: peak sales ±30%, discount rate ±2pp, "
               "R&D cost ±25%, margin ±10pp, patent expiry ±2 yrs, launch ±1 yr, years-to-peak ±2, "
               "each phase success rate ±25% (relative, capped at 99%).")

# ---------------------------------------------------------------- 3 · Monte Carlo

def histogram(values: np.ndarray, title: str) -> go.Figure:
    s = summarize(values)
    lo, hi = np.percentile(values, [0.5, 99.5])   # trim extreme tails so the body is readable
    fig = go.Figure(go.Histogram(x=values[(values >= lo) & (values <= hi)], nbinsx=80, marker_color=BLUE, marker_line_width=0,
                                 hovertemplate="%{x} $M<br>%{y} runs<extra></extra>", name="Runs"))
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    for k, dash, pos in (("p10", "dot", "top left"), ("p50", "dash", "top"), ("p90", "dot", "top right")):
        fig.add_vline(x=s[k], line_color=ORANGE, line_width=2, line_dash=dash,
                      annotation_text=k.upper(), annotation_position=pos)
    return style(fig, title=title, showlegend=False, bargap=0.02, yaxis_title="Runs",
                 xaxis_title="NPV ($M) · extreme 0.5% tails trimmed from view")


def stat_row(s: dict, rnpv: float | None = None):
    c = st.columns(5)
    c[0].metric("Mean", money(s["mean"]), help=f"Deterministic rNPV: {money(rnpv)}" if rnpv is not None else None)
    c[1].metric("P(loss)", f"{s['p_loss']:.1%}")
    c[2].metric("P10", money(s["p10"]))
    c[3].metric("P50", money(s["p50"]))
    c[4].metric("P90", money(s["p90"]))


with tabs[3]:
    st.markdown(f"**{n_sims:,} simulated futures.** In each one, every drug passes or fails each phase "
                "(Bernoulli draws at the phase success rates), peak sales are drawn from a lognormal whose mean "
                "is the base case, and launch may slip 1–2 years. A path's value is the plain NPV of what happened.")
    choice = st.selectbox("Show", ["All candidates (portfolio)", *names], key="t3")
    if choice.startswith("All"):
        vals, rn = sims.sum(axis=1).to_numpy(), table["rNPV"].sum()
    else:
        vals, rn = sims[choice].to_numpy(), table.set_index("Drug").loc[choice, "rNPV"]
    stat_row(summarize(vals), rn)
    st.plotly_chart(histogram(vals, f"Distribution of NPV: {choice}"), use_container_width=True)
    st.caption("The simulated mean sits below rNPV because the simulation also includes launch delays, which the "
               "deterministic rNPV doesn't. Patent expiry is fixed, so every year of slip is a year of lost exclusivity. With delay probability set to 0 the two converge, and the test suite "
               "checks this. A single early-stage drug is a lottery ticket: most paths lose the development spend, "
               "and a few pay off hugely. Pooling drugs is what makes the distribution usable.")

# ---------------------------------------------------------------- 4 · Portfolio

with tabs[4]:
    optimal = optimal_selection(table, budget)
    greedy = greedy_selection(table, budget)

    st.subheader("Ranking")
    st.dataframe(table.style.format({"NPV": "{:,.0f}", "rNPV": "{:,.0f}", "LOA": "{:.1%}",
                                     "Expected R&D spend": "{:,.0f}", "rNPV per R&D $": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("**rNPV per R&D $** = rNPV ÷ risk-adjusted PV of remaining R&D spend (a profitability index). "
               "**Expected R&D spend** is the probability-weighted, undiscounted spend to approval, "
               "which is what the budget is checked against.")

    msg = f"**Recommended (max total rNPV within {money(budget)}):** {', '.join(optimal) or 'none'}."
    if set(greedy) != set(optimal):
        msg += f" Funding in rank order alone would pick {', '.join(greedy)}, which is worse, so the budget is lumpy."
    st.info(msg)

    st.subheader("What if we ranked by plain NPV?")
    comp = compare_selections(drugs, p, rate, budget, sims)
    a, b = comp["Total rNPV"]
    show = comp.assign(**{"Drugs funded": comp["Drugs funded"].map(lambda x: ", ".join(x) or "none")})
    st.dataframe(show[["Drugs funded", "Total NPV", "Total rNPV", "Expected R&D spend", "P(loss)", "P10", "P50", "P90"]]
                 .style.format({"Total NPV": "{:,.0f}", "Total rNPV": "{:,.0f}", "Expected R&D spend": "{:,.0f}",
                                "P(loss)": "{:.1%}", "P10": "{:,.0f}", "P50": "{:,.0f}", "P90": "{:,.0f}"}),
                 use_container_width=True)
    if a > 0 and a != b:
        st.caption(f"Same budget, same funding rule, only the ranking changes. Ranking by plain NPV gives "
                   f"**{(b - a) / a:+.1%}** risk-adjusted value versus ranking by rNPV per R&D dollar. "
                   f"Plain NPV assumes every trial succeeds, so it favours big, early-stage bets. "
                   f"P(loss) and P10/P50/P90 come from the Monte Carlo runs above ($M).")

    st.subheader("Choose what to fund")
    cols = st.columns(len(names))
    funded = [n for n, col in zip(table["Drug"], cols)
              if col.checkbox(n, value=n in optimal, key=f"fund_{n}_{budget}_{rate}")]
    t = table.set_index("Drug")
    spend = t.loc[funded, "Expected R&D spend"].sum()
    port = sims[funded].sum(axis=1).to_numpy() if funded else np.zeros(len(sims))
    s = summarize(port)

    c = st.columns(6)
    c[0].metric("Total rNPV", money(t.loc[funded, "rNPV"].sum()))
    c[1].metric("Expected R&D spend", money(spend), f"{money(budget - spend)} headroom" if spend <= budget
                else f"{money(spend - budget)} over budget", delta_color="normal" if spend <= budget else "inverse")
    c[2].metric("P(portfolio loses money)", f"{s['p_loss']:.1%}")
    for col, k in zip(c[3:], ("p10", "p50", "p90")):
        col.metric(k.upper(), money(s[k]))
    if spend > budget:
        st.warning("This selection exceeds the R&D budget.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(histogram(port, "Funded portfolio: NPV distribution"), use_container_width=True)
    with right:
        # Every possible portfolio: mean vs risk, from the same simulated paths.
        subs = all_subsets(table)
        mean = [sims[list(x.drugs)].sum(axis=1).mean() if x.drugs else 0 for x in subs]
        sd = [sims[list(x.drugs)].sum(axis=1).std() if x.drugs else 0 for x in subs]
        ok = np.array([x.spend <= budget for x in subs])
        label = [", ".join(x.drugs) or "(nothing)" for x in subs]
        cd = np.column_stack([label, [x.spend for x in subs]])
        hov = "%{customdata[0]}<br>mean %{y:,.0f} $M · risk %{x:,.0f} $M<br>spend %{customdata[1]:,.0f} $M<extra></extra>"
        fig = go.Figure([
            go.Scatter(x=np.array(sd)[~ok], y=np.array(mean)[~ok], mode="markers", name="Over budget",
                       marker=dict(size=9, color=GREY, opacity=0.45), customdata=cd[~ok], hovertemplate=hov),
            go.Scatter(x=np.array(sd)[ok], y=np.array(mean)[ok], mode="markers", name="Affordable",
                       marker=dict(size=9, color=BLUE, line=dict(width=2, color="white")),
                       customdata=cd[ok], hovertemplate=hov),
            go.Scatter(x=[s["std"]], y=[s["mean"]], mode="markers", name="Your selection",
                       marker=dict(size=16, color=ORANGE, symbol="diamond", line=dict(width=2, color="white")),
                       hovertemplate="Your selection<br>mean %{y:,.0f} $M · risk %{x:,.0f} $M<extra></extra>"),
        ])
        st.plotly_chart(style(fig, title="Every possible portfolio: value vs risk",
                              xaxis_title="Risk: std. dev. of NPV ($M)", yaxis_title="Expected NPV ($M)"),
                        use_container_width=True)

    if funded:
        st.subheader("Development timeline of funded programmes")
        fig, seen = go.Figure(), set()
        colours = dict(zip(PHASES, ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"]))
        for name in funded:
            d = by_name[name]
            sched = cash_flows(d, p, rate)
            for ph, g in sched[sched["stage"] != "Market"].groupby("stage", sort=False):
                fig.add_trace(go.Bar(y=[name], x=[len(g)], base=[g["year"].min()], orientation="h",
                                     name=ph, legendgroup=ph, marker_color=colours[ph],
                                     showlegend=ph not in seen,
                                     hovertemplate=f"{name} · {ph}<br>{g['year'].min()}–{g['year'].max()}<extra></extra>"))
                seen.add(ph)
        fig.update_layout(barmode="overlay", xaxis=dict(dtick=1, range=[START_YEAR, None]))
        st.plotly_chart(style(fig, height=110 + 50 * len(funded), title_text=""), use_container_width=True)

# ---------------------------------------------------------------- Method

with tabs[5]:
    st.markdown(open("METHODOLOGY.md", encoding="utf-8").read())
