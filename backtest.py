from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import technical_analysis, forecast_return, combined_score, allocate


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "TXN", "AMAT", "MU", "ADI",
    "KLAC", "LRCX", "ORCL", "CRM", "CSCO", "INTU", "ADBE", "NOW", "PANW", "ANET",
    "IBM", "AMZN", "GOOGL", "META", "NFLX",
]


@dataclass
class BacktestConfig:
    start: str = "2021-10-07"
    end: Optional[str] = None
    rebalance_months: int = 3
    holdings: int = 15
    starting_cash: float = 100_000.0
    profile: str = "Balanced"
    horizon: int = 63
    benchmark: str = "SPY"
    positive_forecast_only: bool = True
    transaction_cost_bps: float = 10.0
    whole_shares: bool = False


def _download(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        group_by="column",
        threads=True,
        progress=False,
    )
    out = {}
    if raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            frame = None
            if s in raw.columns.get_level_values(1):
                try:
                    frame = raw.xs(s, axis=1, level=1, drop_level=True)
                except Exception:
                    pass
            elif s in raw.columns.get_level_values(0):
                try:
                    frame = raw.xs(s, axis=1, level=0, drop_level=True)
                except Exception:
                    pass
            if frame is not None and "Close" in frame and frame.Close.notna().any():
                out[s] = frame.dropna(subset=["Close"]).copy()
    else:
        if len(symbols) == 1:
            s = symbols[0]
            out[s] = raw.dropna(subset=["Close"]).copy()
    return out


def _quarterly_schedule(start: pd.Timestamp, end: pd.Timestamp, months: int) -> List[pd.Timestamp]:
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d = d + pd.DateOffset(months=months)
    return dates


def _trade_date(index: pd.DatetimeIndex, scheduled: pd.Timestamp) -> Optional[pd.Timestamp]:
    # First market session on/after scheduled date.
    loc = index.searchsorted(scheduled)
    if loc >= len(index):
        return None
    return pd.Timestamp(index[loc])


def _signal_date(index: pd.DatetimeIndex, trade_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    loc = index.get_indexer([trade_date])[0]
    if loc <= 0:
        return None
    return pd.Timestamp(index[loc - 1])


def _execution_price(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    row = frame.loc[date]
    value = row.get("Open", np.nan)
    if pd.isna(value) or value <= 0:
        value = row["Close"]
    return float(value)


def _mark_price(frame: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
    hist = frame.loc[:date, "Close"].dropna()
    return float(hist.iloc[-1]) if len(hist) else None


def rank_universe(price_data: Dict[str, pd.DataFrame], as_of: pd.Timestamp, horizon: int, positive_only: bool) -> pd.DataFrame:
    rows = []
    for symbol, frame in price_data.items():
        hist = frame.loc[:as_of].copy()
        if len(hist) < 260:
            continue
        tech = technical_analysis(hist)
        if not tech:
            continue
        forecast = forecast_return(hist, horizon=horizon)
        pred = float(forecast.get("predicted_return", np.nan)) if forecast else np.nan
        if positive_only and (not np.isfinite(pred) or pred <= 0):
            continue
        score, components = combined_score(tech["technical_score"], forecast)
        rows.append({
            "ticker": symbol,
            "score": score,
            "technical_score": tech["technical_score"],
            "forecast_return": pred,
            "forecast_kind": forecast.get("kind") if forecast else "Unavailable",
            "ml_evidence_weight": forecast.get("evidence_weight", 0.0) if forecast else 0.0,
            "annualized_volatility": tech["annualized_volatility"],
            "price_asof": tech["price"],
            "rsi": tech["rsi"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def run_backtest(universe: Iterable[str], cfg: BacktestConfig):
    universe = list(dict.fromkeys(s.upper().strip() for s in universe if s.strip()))
    if not universe:
        raise ValueError("Universe is empty")
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end) if cfg.end else pd.Timestamp.today().normalize()
    # Extra history is required so a 2021 decision can train only on pre-2021 rows.
    history_start = start - pd.DateOffset(years=8)
    all_symbols = list(dict.fromkeys(universe + [cfg.benchmark]))
    data = _download(all_symbols, history_start, end)
    if cfg.benchmark not in data:
        raise ValueError(f"Benchmark {cfg.benchmark} could not be downloaded")
    benchmark = data[cfg.benchmark]
    stock_data = {s: data[s] for s in universe if s in data}
    if len(stock_data) < cfg.holdings:
        raise ValueError(f"Only {len(stock_data)} symbols downloaded successfully; need at least {cfg.holdings}.")

    calendar = benchmark.loc[start:end].index
    if len(calendar) < 2:
        raise ValueError("No benchmark trading sessions in the requested period")
    schedule = _quarterly_schedule(start, end, cfg.rebalance_months)
    trades = []
    holdings_log = []
    shares = {}
    cash = float(cfg.starting_cash)
    costs_total = 0.0
    equity_rows = []

    # Precompute actual trade dates.
    rebalance_map = {}
    for scheduled in schedule:
        td = _trade_date(calendar, scheduled)
        if td is not None:
            rebalance_map[td] = scheduled

    for date in calendar:
        date = pd.Timestamp(date)
        if date in rebalance_map:
            sig = _signal_date(calendar, date)
            if sig is not None:
                ranking = rank_universe(stock_data, sig, cfg.horizon, cfg.positive_forecast_only)
                if not ranking.empty:
                    chosen = allocate(ranking, min(cfg.holdings, len(ranking)), cfg.profile)
                    # Liquidate everything at next-session open.
                    for s, qty in list(shares.items()):
                        frame = stock_data[s]
                        if date not in frame.index:
                            continue
                        px = _execution_price(frame, date)
                        gross = qty * px
                        fee = abs(gross) * cfg.transaction_cost_bps / 10_000
                        cash += gross - fee
                        costs_total += fee
                        trades.append({"date": date, "signal_date": sig, "ticker": s, "side": "SELL", "shares": qty, "price": px, "fee": fee})
                    shares = {}

                    portfolio_value = cash
                    # Buy target basket.
                    for row in chosen.itertuples(index=False):
                        s = row.ticker
                        frame = stock_data[s]
                        if date not in frame.index:
                            continue
                        px = _execution_price(frame, date)
                        target = portfolio_value * float(row.target_weight)
                        if cfg.whole_shares:
                            qty = math.floor(target / px)
                        else:
                            qty = target / px
                        if qty <= 0:
                            continue
                        gross = qty * px
                        fee = gross * cfg.transaction_cost_bps / 10_000
                        if gross + fee > cash:
                            qty = max(0.0, cash / (px * (1 + cfg.transaction_cost_bps / 10_000)))
                            if cfg.whole_shares:
                                qty = math.floor(qty)
                            gross = qty * px
                            fee = gross * cfg.transaction_cost_bps / 10_000
                        if qty <= 0:
                            continue
                        cash -= gross + fee
                        costs_total += fee
                        shares[s] = qty
                        trades.append({"date": date, "signal_date": sig, "ticker": s, "side": "BUY", "shares": qty, "price": px, "fee": fee})
                        holdings_log.append({
                            "rebalance_date": date,
                            "signal_date": sig,
                            "ticker": s,
                            "score": row.score,
                            "technical_score": row.technical_score,
                            "forecast_return": row.forecast_return,
                            "target_weight": row.target_weight,
                            "execution_price": px,
                            "shares": qty,
                        })

        value = cash
        for s, qty in shares.items():
            px = _mark_price(stock_data[s], date)
            if px is not None:
                value += qty * px
        equity_rows.append({"date": date, "portfolio": value})

    equity = pd.DataFrame(equity_rows).set_index("date")
    # Benchmark normalized to first tradable close in the period.
    b = benchmark.loc[equity.index.min():equity.index.max(), "Close"].reindex(equity.index).ffill().dropna()
    b = b / b.iloc[0] * cfg.starting_cash
    equity["benchmark"] = b
    equity["portfolio_return"] = equity.portfolio / cfg.starting_cash - 1
    equity["benchmark_return"] = equity.benchmark / cfg.starting_cash - 1

    daily = equity.portfolio.pct_change().dropna()
    ann_return = (equity.portfolio.iloc[-1] / equity.portfolio.iloc[0]) ** (252 / max(len(daily), 1)) - 1
    ann_vol = daily.std() * np.sqrt(252) if len(daily) else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    dd = equity.portfolio / equity.portfolio.cummax() - 1

    summary = {
        "start_value": cfg.starting_cash,
        "end_value": float(equity.portfolio.iloc[-1]),
        "total_return": float(equity.portfolio.iloc[-1] / cfg.starting_cash - 1),
        "benchmark_end": float(equity.benchmark.iloc[-1]),
        "benchmark_return": float(equity.benchmark.iloc[-1] / cfg.starting_cash - 1),
        "alpha_vs_benchmark": float(equity.portfolio.iloc[-1] / cfg.starting_cash - equity.benchmark.iloc[-1] / cfg.starting_cash),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe_no_rf": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(dd.min()),
        "transaction_costs": float(costs_total),
        "rebalances": int(pd.DataFrame(holdings_log).rebalance_date.nunique()) if holdings_log else 0,
    }
    return {
        "summary": summary,
        "equity": equity,
        "holdings": pd.DataFrame(holdings_log),
        "trades": pd.DataFrame(trades),
    }
