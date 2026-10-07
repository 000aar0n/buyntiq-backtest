import pandas as pd
import plotly.express as px
import streamlit as st

from backtest import BacktestConfig, DEFAULT_UNIVERSE, run_backtest

st.set_page_config(page_title="Buyntiq Backtest", layout="wide")
st.title("Buyntiq · 5-Year Portfolio Backtest")
st.caption("Point-in-time backtest: signals use only information available before each trade. Rebalances execute on the next market session.")

with st.sidebar:
    st.header("Settings")
    start = st.date_input("Start date", pd.Timestamp.today().date() - pd.DateOffset(years=5))
    end = st.date_input("End date", pd.Timestamp.today().date())
    starting_cash = st.number_input("Starting cash", min_value=1000.0, value=100000.0, step=10000.0)
    holdings = st.slider("Holdings", 3, 25, 15)
    profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ"], index=0)
    positive = st.checkbox("Require positive 3-month forecast", value=True)
    costs = st.number_input("Transaction cost (bps per trade)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    whole = st.checkbox("Whole shares", value=False)

symbols_text = st.text_area(
    "Universe (comma-separated)",
    value=", ".join(DEFAULT_UNIVERSE),
    height=110,
    help="Use current long-lived stocks for a clean first test. A current-membership universe still has survivorship bias; see README.",
)

if st.button("Run 5-year backtest", type="primary", use_container_width=True):
    symbols = [x.strip().upper() for x in symbols_text.replace("\n", ",").split(",") if x.strip()]
    cfg = BacktestConfig(
        start=str(start), end=str(end), holdings=holdings, starting_cash=starting_cash,
        profile=profile, benchmark=benchmark, positive_forecast_only=positive,
        transaction_cost_bps=costs, whole_shares=whole,
    )
    with st.spinner("Downloading history and rebuilding the portfolio at each quarter..."):
        result = run_backtest(symbols, cfg)
    s = result["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ending value", f"${s['end_value']:,.0f}", f"{s['total_return']:+.1%}")
    c2.metric(f"{benchmark} ending value", f"${s['benchmark_end']:,.0f}", f"{s['benchmark_return']:+.1%}")
    c3.metric("Return vs benchmark", f"{s['alpha_vs_benchmark']:+.1%}")
    c4.metric("Max drawdown", f"{s['max_drawdown']:.1%}")

    chart = result["equity"][["portfolio", "benchmark"]].rename(columns={"portfolio": "Buyntiq", "benchmark": benchmark})
    st.plotly_chart(px.line(chart, labels={"value": "Portfolio value", "date": "Date", "variable": "Series"}), use_container_width=True)

    st.subheader("Stats")
    stats = pd.DataFrame({
        "Metric": ["Total return", "Annualized return", "Annualized volatility", "Sharpe (rf=0)", "Max drawdown", "Trading costs", "Rebalances"],
        "Value": [
            f"{s['total_return']:.2%}", f"{s['annualized_return']:.2%}", f"{s['annualized_volatility']:.2%}",
            f"{s['sharpe_no_rf']:.2f}", f"{s['max_drawdown']:.2%}", f"${s['transaction_costs']:,.2f}", str(s['rebalances'])
        ]
    })
    st.dataframe(stats, hide_index=True, use_container_width=True)

    st.subheader("Quarterly holdings")
    st.dataframe(result["holdings"], hide_index=True, use_container_width=True)
    st.download_button("Download holdings CSV", result["holdings"].to_csv(index=False), "buyntiq_backtest_holdings.csv", "text/csv")

    st.subheader("Trade log")
    st.dataframe(result["trades"], hide_index=True, use_container_width=True)
    st.download_button("Download trades CSV", result["trades"].to_csv(index=False), "buyntiq_backtest_trades.csv", "text/csv")

st.info("Historical fundamentals are intentionally not pulled from today's Yahoo company snapshot. Using today's fundamentals in 2021 would leak future information. Add a point-in-time fundamentals dataset later if you want the full production Company score in every historical rebalance.")
