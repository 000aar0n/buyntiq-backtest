import pandas as pd
import plotly.express as px
import streamlit as st

from backtest import BacktestConfig, COMPARISON_LABELS, DEFAULT_UNIVERSE, run_backtest

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
    benchmark = st.selectbox(
        "Primary benchmark for alpha",
        ["SPY", "QQQ", "FWD", "AB_INTL_TECH"],
        index=0,
        format_func=lambda x: COMPARISON_LABELS.get(x, x),
        help="Choose SPY, QQQ, AB Disruptors ETF (FWD), or AB International Technology.",
    )
    positive = st.checkbox("Require positive 3-month forecast", value=True)
    costs = st.number_input("Transaction cost (bps per trade)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    whole = st.checkbox("Whole shares", value=False)
    speed = st.selectbox(
        "Model mode",
        ["Fast (recommended)", "Full validation (slow)"],
        index=0,
        help="Fast keeps the same three model families but uses one chronological validation block and smaller tree/boosting models. Full uses three validation folds and larger models.",
    )
    finalists = st.number_input(
        "ML finalists per rebalance",
        min_value=int(holdings),
        max_value=100,
        value=max(int(holdings), 18),
        step=1,
        help="Every stock gets the cheap technical screen. Only these top candidates get the expensive ML retraining.",
    )

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
        model_mode="fast" if speed.startswith("Fast") else "full",
        finalists=int(finalists),
    )
    bar = st.progress(0.0, text="Preparing backtest")
    try:
        result = run_backtest(
            symbols,
            cfg,
            progress=lambda fraction, message: bar.progress(
                min(max(float(fraction), 0.0), 1.0), text=message
            ),
        )
    finally:
        bar.empty()
    s = result["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ending value", f"${s['end_value']:,.0f}", f"{s['total_return']:+.1%}")
    benchmark_label = COMPARISON_LABELS.get(benchmark, benchmark)
    c2.metric(f"{benchmark_label} ending value", f"${s['benchmark_end']:,.0f}", f"{s['benchmark_return']:+.1%}")
    c3.metric("Return vs benchmark", f"{s['alpha_vs_benchmark']:+.1%}")
    c4.metric("Max drawdown", f"{s['max_drawdown']:.1%}")

    chart = result["comparisons"].copy()
    chart.insert(0, "Buyntiq", result["equity"]["portfolio"])
    st.plotly_chart(
        px.line(
            chart,
            labels={"value": "Comparable portfolio value", "date": "Date", "variable": "Series"},
        ),
        use_container_width=True,
    )
    st.caption(
        "Later-starting benchmarks are anchored to Buyntiq's value on their first available date. "
        "FWD therefore starts at its actual history instead of being backfilled."
    )

    st.subheader("Stats")
    stats = pd.DataFrame({
        "Metric": ["Total return", "Annualized return", "Annualized volatility", "Sharpe (rf=0)", "Max drawdown", "Trading costs", "Rebalances"],
        "Value": [
            f"{s['total_return']:.2%}", f"{s['annualized_return']:.2%}", f"{s['annualized_volatility']:.2%}",
            f"{s['sharpe_no_rf']:.2f}", f"{s['max_drawdown']:.2%}", f"${s['transaction_costs']:,.2f}", str(s['rebalances'])
        ]
    })
    st.dataframe(stats, hide_index=True, use_container_width=True)

    st.subheader("Benchmark comparisons")
    comparisons = result["comparison_stats"].copy()
    if not comparisons.empty:
        comparisons["Start"] = pd.to_datetime(comparisons["start_date"]).dt.strftime("%Y-%m-%d")
        comparisons["End"] = pd.to_datetime(comparisons["end_date"]).dt.strftime("%Y-%m-%d")
        comparisons["Benchmark return"] = comparisons["benchmark_return"].map(lambda x: f"{x:+.2%}")
        comparisons["Buyntiq same period"] = comparisons["buyntiq_return_same_period"].map(lambda x: f"{x:+.2%}")
        comparisons["Buyntiq edge"] = comparisons["alpha_same_period"].map(lambda x: f"{x:+.2%}")
        show_cols = ["name", "Start", "End", "Benchmark return", "Buyntiq same period", "Buyntiq edge"]
        st.dataframe(
            comparisons[show_cols].rename(columns={"name": "Benchmark"}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("No comparison benchmark histories were available.")

    if "AB International Technology" not in result["comparisons"].columns:
        st.warning(
            "AB International Technology (LU0060230025 / XAY5) could not be downloaded from Yahoo on this run. "
            "SPY, QQQ, and FWD comparisons are still valid."
        )

    st.subheader("Quarterly holdings")
    st.dataframe(result["holdings"], hide_index=True, use_container_width=True)
    st.download_button("Download holdings CSV", result["holdings"].to_csv(index=False), "buyntiq_backtest_holdings.csv", "text/csv")

    st.subheader("Trade log")
    st.dataframe(result["trades"], hide_index=True, use_container_width=True)
    st.download_button("Download trades CSV", result["trades"].to_csv(index=False), "buyntiq_backtest_trades.csv", "text/csv")

st.info(
    "Historical fundamentals are intentionally not pulled from today's Yahoo company snapshot. "
    "Using today's fundamentals in 2021 would leak future information. Fast mode is recommended on throttled CPUs; "
    "Full validation is intentionally much slower. AB International Technology uses the LU0060230025/XAY5 exchange "
    "quote when available and converts the EUR quote to USD for a closer share-class comparison."
)
