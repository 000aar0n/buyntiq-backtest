import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend_loader import load_backend

st.set_page_config(page_title="Buyntiq Backtest", layout="wide")
try:
    backend = load_backend()
except (RuntimeError, OSError) as exc:
    st.error(f"Backtest startup could not finish: {exc}")
    st.stop()
BacktestConfig = backend.engine.BacktestConfig
COMPARISON_LABELS = backend.engine.COMPARISON_LABELS
DEFAULT_UNIVERSE = backend.engine.DEFAULT_UNIVERSE
run_backtest = backend.engine.run_backtest
load_us_universe = backend.universe.load_us_universe
st.title("Buyntiq · Portfolio Backtest")
st.caption("Choose your stock universe, historical period, and how often the portfolio is rebuilt.")


@st.cache_data(ttl=86400, show_spinner=False)
def us_listings(engine_build):
    return load_us_universe()


now = pd.Timestamp.now(tz="America/New_York")
# Avoid using an unfinished US daily bar. Holidays/weekends are trimmed by SPY.
latest_day = (now.normalize() if now.hour >= 18 else now.normalize()-pd.Timedelta(days=1)).date()
frequencies = {
    "Every week": (3, 1, 5),
    "Every month": (1, None, 21),
    "Every 2 months": (2, None, 42),
    "Every 3 months": (3, None, 63),
    "Every 6 months": (6, None, 126),
    "Every year": (12, None, 252),
}

with st.sidebar:
    st.header("Backtest settings")
    period = st.radio("Historical period", ["Years back", "Custom dates"], horizontal=True)
    end = st.date_input("End date", value=latest_day, max_value=latest_day, min_value=pd.Timestamp("1980-01-02").date())
    if period == "Years back":
        years = st.number_input("Years back", min_value=1, max_value=20, value=5, step=1)
        start = (pd.Timestamp(end)-pd.DateOffset(years=int(years))).date()
        st.caption(f"Start date: {start:%B %d, %Y}")
    else:
        start = st.date_input("Start date", value=(pd.Timestamp(end)-pd.DateOffset(years=5)).date(), min_value=pd.Timestamp("1980-01-01").date(), max_value=latest_day)
    frequency = st.selectbox("Rebuild portfolio", list(frequencies), index=3,
                             help="At each rebalance, refresh historical inputs, retrain forecasts, and simulate buys and sells. This does not schedule live trades.")
    months, weeks, matching_horizon = frequencies[frequency]
    horizon_choice = st.selectbox("Forecast horizon", ["Match rebalance frequency", "1 month", "3 months", "6 months", "1 year"])
    horizon = matching_horizon if horizon_choice.startswith("Match") else {"1 month":21,"3 months":63,"6 months":126,"1 year":252}[horizon_choice]
    starting_cash = st.number_input("Starting cash", min_value=1000.0, value=100000.0, step=10000.0)
    holdings = st.slider("Holdings", 3, 50, 15)
    profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    benchmark = st.selectbox("Primary benchmark", ["SPY", "QQQ", "FWD", "AB_INTL_TECH"],
                            format_func=lambda x: "AB International Technology · A USD (INTTECHA)" if x == "AB_INTL_TECH" else COMPARISON_LABELS[x])
    positive = st.checkbox("Require positive forecast", value=True)
    with st.expander("Stock eligibility", expanded=True):
        min_cap = st.number_input("Minimum estimated market cap ($ billions)", min_value=0.0, max_value=1000.0, value=2.0, step=0.5,
            help="Uses original SEC share counts published before each signal date, adjusted for splits, times that day's price. Counts older than 180 days and unverifiable share classes/ADRs are excluded. Set 0 to disable.")
        min_price = st.number_input("Minimum historical share price ($)", min_value=0.0, max_value=1000.0, value=5.0, step=1.0,
            help="Price in the share units traded at that time, not today's dividend/split-adjusted price. Set 0 to disable.")
        min_liquidity = st.number_input("Minimum daily trading value ($ millions)", min_value=0.0, max_value=10000.0, value=10.0, step=1.0,
            help="Median dollar volume over the 60 sessions before each rebalance. Set 0 to disable.")
        st.caption("Filters apply at every rebalance. Unknown enabled-filter inputs are excluded. SEC size estimates have incomplete historical coverage.")
    with st.expander("Costs and model settings"):
        costs = st.number_input("Transaction cost (bps per trade)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        whole = st.checkbox("Whole shares", value=False)
        speed = st.selectbox("Model mode", ["Fast (recommended)", "Full validation (slow)"],
                             help="Fast uses one selection block; Full uses three selection folds. Both use a separate purged holdout to gate ML influence.")
        finalists = st.number_input("Initial ML finalists per rebalance", min_value=int(holdings), max_value=200,
                                    value=max(int(holdings),18), step=1,
                                    help="All available stocks receive the technical screen. These top candidates receive ML analysis first; the pool expands if too few qualify.")

scope = st.selectbox("Stock universe", ["Entire US stock universe", "Technology sample", "Custom tickers"])
if scope == "Entire US stock universe":
    st.caption("All current common shares and ADRs in Nasdaq Trader's Nasdaq and other-US-exchange directories. ETFs, preferred shares, warrants, rights and units are excluded. No top-100 cap. The eligibility controls determine which stocks can be selected. Historical delisted companies are not included.")
    symbols_text = None
else:
    symbols_text = st.text_area("Tickers (comma-separated)", value=", ".join(DEFAULT_UNIVERSE) if scope == "Technology sample" else "", height=110)
st.caption("The full-universe first run can take substantially longer. Historical size checks are cached and only requested for candidates that pass price/liquidity filters. Downloads are batched and cached for 24 hours; historical price files are read from disk as needed. Weekly rebalancing requires many more model fits than quarterly rebalancing.")

if st.button("Run backtest", type="primary", use_container_width=True):
    bar = st.progress(0.0, text="Preparing backtest")
    try:
        if start >= end:
            raise ValueError("Start date must be before end date.")
        if scope == "Entire US stock universe":
            bar.progress(.01, text="Loading current US exchange listings")
            listings = us_listings(backend.build)
            symbols = listings.ticker.tolist()
        else:
            symbols = list(dict.fromkeys(x.strip().upper().replace(".", "-") for x in symbols_text.replace("\n", ",").split(",") if x.strip()))
        cfg = BacktestConfig(start=str(start), end=str(end), rebalance_months=months, rebalance_weeks=weeks,
            horizon=horizon, holdings=holdings, starting_cash=starting_cash, profile=profile,
            benchmark=benchmark, positive_forecast_only=positive, transaction_cost_bps=costs,
            whole_shares=whole, model_mode="fast" if speed.startswith("Fast") else "full", finalists=int(finalists),
            min_market_cap=float(min_cap)*1e9, min_price=float(min_price), min_dollar_volume=float(min_liquidity)*1e6)
        result = run_backtest(symbols, cfg, progress=lambda f,m:bar.progress(min(max(float(f),0),1),text=m))
        st.session_state["backtest_result"] = result
        st.session_state["backtest_config"] = asdict(cfg)
        st.session_state["backtest_build"] = backend.build
        st.session_state["backtest_scope"] = scope
        st.session_state["backtest_symbols"] = symbols
    except Exception as exc:
        st.error(f"Backtest could not finish: {exc}")
    finally:
        bar.empty()

result = st.session_state.get("backtest_result")
if result is not None:
    saved = st.session_state["backtest_config"]
    s = result["summary"]
    interval = f"{saved['rebalance_weeks']} week(s)" if saved.get("rebalance_weeks") else f"{saved['rebalance_months']} month(s)"
    st.subheader("Saved run results")
    st.caption(f"{saved['start']} to {saved['end']} · rebuild every {interval} · {st.session_state['backtest_scope']} · {result['universe_downloaded']:,}/{result['universe_requested']:,} stock histories loaded. Changes above apply when you run again.")
    st.caption(f"Run filters: estimated market cap ≥ ${saved.get('min_market_cap',0)/1e9:g}B · historical price ≥ ${saved.get('min_price',0):g} · median daily trading value ≥ ${saved.get('min_dollar_volume',0)/1e6:g}M. A zero threshold means that filter was disabled.")
    for message in result.get("warnings",[]):
        st.warning(message)
    for name,error in result.get("benchmark_errors",{}).items():
        st.warning(f"{name}: {error}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ending value", f"${s['end_value']:,.0f}", f"{s['total_return']:+.1%}")
    label = COMPARISON_LABELS.get(saved["benchmark"], saved["benchmark"])
    if np.isfinite(s["benchmark_end"]):
        c2.metric(f"{label} ending value", f"${s['benchmark_end']:,.0f}", f"{s['benchmark_return']:+.1%}")
        c3.metric("Same-period return edge", f"{s['alpha_vs_benchmark']*100:+.1f} pp")
    else:
        c2.metric(f"{label} ending value", "Unavailable")
        c3.metric("Same-period return edge", "Unavailable")
    c4.metric("Max drawdown", f"{s['max_drawdown']:.1%}")

    chart=result["comparisons"].copy()
    chart.insert(0,"Buyntiq",result["equity"]["portfolio"])
    st.plotly_chart(px.line(chart,labels={"value":"Comparable portfolio value","date":"Date","variable":"Series"}),use_container_width=True)
    st.caption("Benchmark lines start at Buyntiq's closing value on their first available date. The comparison table uses matching dates; newer funds are not backfilled. AB uses official A USD NAV (LU0060230025), through its last reported date.")

    st.subheader("Benchmark comparisons")
    comparisons=result["comparison_stats"].copy()
    if not comparisons.empty:
        comparisons["Start"]=pd.to_datetime(comparisons.start_date).dt.strftime("%Y-%m-%d")
        comparisons["End"]=pd.to_datetime(comparisons.end_date).dt.strftime("%Y-%m-%d")
        comparisons["Benchmark return"]=comparisons.benchmark_return.map(lambda x:f"{x:+.2%}")
        comparisons["Buyntiq same period"]=comparisons.buyntiq_return_same_period.map(lambda x:f"{x:+.2%}")
        comparisons["Buyntiq edge (pp)"]=comparisons.alpha_same_period.map(lambda x:f"{100*x:+.2f}")
        st.dataframe(comparisons[["name","Start","End","Benchmark return","Buyntiq same period","Buyntiq edge (pp)"]].rename(columns={"name":"Benchmark"}),hide_index=True,use_container_width=True)
    st.subheader("Stats")
    st.dataframe(pd.DataFrame({"Metric":["Total return","Annualized return","Annualized volatility","Sharpe (rf=0)","Max drawdown","Trading costs","Rebalances"],
        "Value":[f"{s['total_return']:.2%}",f"{s['annualized_return']:.2%}",f"{s['annualized_volatility']:.2%}",f"{s['sharpe_no_rf']:.2f}",f"{s['max_drawdown']:.2%}",f"${s['transaction_costs']:,.2f}",str(s['rebalances'])]}),hide_index=True,use_container_width=True)

    excluded = result.get("eligibility_exclusions", pd.DataFrame())
    if not excluded.empty:
        with st.expander("Why stocks were excluded"):
            st.dataframe(excluded, hide_index=True, use_container_width=True)
            st.caption("Counts are per rebalance. Price/liquidity checks cover the universe; size/forecast checks cover ranked candidates considered for selection. A stock can appear on multiple dates.")
            st.download_button("Download eligibility exclusions", excluded.to_csv(index=False), "buyntiq_eligibility_exclusions.csv", "text/csv")
    st.subheader("Holdings by rebalance")
    st.dataframe(result["holdings"],hide_index=True,use_container_width=True)
    st.download_button("Download holdings CSV",result["holdings"].to_csv(index=False),"buyntiq_backtest_holdings.csv","text/csv")
    with st.expander("Trade log and downloads"):
        st.caption(f"Engine build: {st.session_state.get('backtest_build', 'older saved run')}")
        st.dataframe(result["trades"],hide_index=True,use_container_width=True)
        st.download_button("Download trades CSV",result["trades"].to_csv(index=False),"buyntiq_backtest_trades.csv","text/csv")
        st.download_button("Download daily equity CSV",result["equity"].to_csv(),"buyntiq_backtest_equity.csv","text/csv")
        st.download_button("Download benchmark comparison CSV",result["comparison_stats"].to_csv(index=False),"buyntiq_backtest_comparisons.csv","text/csv")
        st.download_button("Download benchmark curves CSV",chart.to_csv(),"buyntiq_backtest_curves.csv","text/csv")
        st.download_button("Download run settings",json.dumps({**saved,"scope":st.session_state["backtest_scope"],"universe":st.session_state["backtest_symbols"]},indent=2),"buyntiq_backtest_settings.json","application/json")
    if result.get("download_errors"):
        with st.expander(f"Unavailable price histories ({len(result['download_errors']):,})"):
            errors=pd.DataFrame(result["download_errors"].items(),columns=["Ticker","Reason"])
            st.dataframe(errors,hide_index=True,use_container_width=True)
            st.download_button("Download missing histories",errors.to_csv(index=False),"buyntiq_missing_histories.csv","text/csv")

st.info("This backtest uses historical technical and ML inputs, not today's company fundamentals. Current listing membership introduces survivorship bias and is not a historical listing universe. Adjusted prices model splits and reinvested dividends; execution costs and fractional/whole-share fills are approximations. Past outperformance does not establish future returns.")
