from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ab_benchmark import load_ab_intl_tech
from price_store import download_prices
from strategy import technical_analysis, forecast_return, combined_score, allocate
from threadpoolctl import threadpool_limits


DEFAULT_UNIVERSE = [
    # Mega-cap / platforms / enterprise
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "IBM", "AMZN", "GOOGL",
    "META", "NFLX", "ADBE", "INTU", "NOW", "ADSK", "CDNS", "SNPS", "FICO", "PLTR",

    # Semiconductors / equipment
    "QCOM", "TXN", "AMAT", "MU", "ADI", "KLAC", "LRCX", "ASML", "TSM", "NXPI",
    "MCHP", "MPWR", "ON", "GFS", "MRVL", "SWKS", "TER", "ENTG", "COHR", "QRVO",
    "ACLS", "AEHR", "AMKR", "FORM", "MKSI", "IPGP", "SITM", "ALGM", "POWI", "DIOD",
    "SLAB", "CRUS", "NVMI", "CAMT",

    # Networking / hardware / infrastructure
    "CSCO", "ANET", "DELL", "HPE", "HPQ", "NTAP", "PSTG", "GLW", "AKAM", "FFIV",
    "KEYS", "LOGI", "CIEN", "CDW", "JBL", "FLEX", "SMCI", "VRT",

    # Cybersecurity / cloud / software
    "PANW", "FTNT", "CRWD", "ZS", "OKTA", "DDOG", "NET", "MDB", "HUBS", "TEAM",
    "DOCU", "U", "PATH", "APP", "SHOP", "WDAY", "SNOW", "MNDY", "GTLB", "CFLT",
    "ESTC", "DT", "IOT", "TWLO", "ZM", "DOCN", "BILL", "GEN", "CHKP", "CYBR",
    "NICE",

    # IT services / international tech listed in the U.S.
    "ACN", "CTSH", "EPAM", "PAYC", "PAYX", "SAP", "INFY",

    # Storage
    "WDC", "STX",
]

COMPARISON_LABELS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "FWD": "AB Disruptors ETF (FWD)",
    "AB_INTL_TECH": "AB International Technology",
}

@dataclass
class BacktestConfig:
    start: str = "2021-10-07"
    end: Optional[str] = None
    rebalance_months: int = 3
    rebalance_weeks: Optional[int] = None
    holdings: int = 15
    starting_cash: float = 100_000.0
    profile: str = "Balanced"
    horizon: int = 63
    benchmark: str = "SPY"
    positive_forecast_only: bool = True
    transaction_cost_bps: float = 10.0
    whole_shares: bool = False
    model_mode: str = "fast"
    finalists: int = 18


def _download(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    raw = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        group_by="column",
        threads=4,
        timeout=15,
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


def _anchor_comparison(
    close: pd.Series,
    equity: pd.DataFrame,
) -> Optional[pd.Series]:
    """Align a benchmark to the portfolio value on its first available date."""
    if close is None or close.empty or equity.empty:
        return None
    close = close[~close.index.duplicated(keep="last")].sort_index()
    # Fill only interior calendar gaps, never extend a benchmark past its
    # last published observation or across an arbitrarily long missing period.
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]
    if len(close) < 2:return None
    aligned = close.reindex(equity.index, method="ffill", tolerance=pd.Timedelta(days=5))
    aligned.loc[aligned.index > close.index[-1]] = np.nan
    if aligned.notna().sum() < 2:return None
    first = aligned.first_valid_index()
    if first is None:
        return None
    base = float(aligned.loc[first])
    if not np.isfinite(base) or base <= 0:
        return None
    anchor = float(equity.loc[first, "portfolio"])
    result = aligned / base * anchor
    result.loc[result.index < first] = np.nan
    return result


def _comparison_stats(name: str, series: pd.Series, equity: pd.DataFrame) -> Optional[dict]:
    valid = series.dropna()
    if len(valid) < 2:
        return None
    first, last = valid.index[0], valid.index[-1]
    benchmark_return = float(valid.iloc[-1] / valid.iloc[0] - 1)
    p = equity.loc[first:last, "portfolio"].dropna()
    if p.empty:
        return None
    portfolio_return = float(p.iloc[-1] / p.iloc[0] - 1)
    return {
        "name": name,
        "start_date": pd.Timestamp(first),
        "end_date": pd.Timestamp(last),
        "benchmark_return": benchmark_return,
        "buyntiq_return_same_period": portfolio_return,
        "alpha_same_period": portfolio_return - benchmark_return,
    }


def _quarterly_schedule(start, end, months=3, weeks=None):
    if weeks is not None:
        if not isinstance(weeks, int) or weeks < 1:raise ValueError("Rebalance weeks must be a positive integer")
    elif not isinstance(months, int) or months < 1:
        raise ValueError("Rebalance months must be a positive integer")
    dates=[];i=0
    while True:
        d=start+(pd.DateOffset(weeks=weeks*i) if weeks is not None else pd.DateOffset(months=months*i))
        if d>end:break
        dates.append(d);i+=1
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
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Missing opening execution price on {date.date()}; no closing-price substitution was made.")
    return float(value)


def _mark_price(frame: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
    hist = frame.loc[:date, "Close"].dropna()
    if not len(hist) or (date-hist.index[-1]).days > 7:
        raise ValueError(f"A held stock has no recent closing price on {date.date()}; cannot value it safely.")
    return float(hist.iloc[-1])


def rank_universe(
    price_data: Dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    horizon: int,
    positive_only: bool,
    count: int,
    finalists: int,
    model_mode: str,
    progress=None,
    progress_base: float = 0.0,
    progress_span: float = 0.0,
) -> pd.DataFrame:
    """Fast two-stage screen: technical score first, ML only for finalists.

    This mirrors the live Buyntiq builder's architecture and avoids spending
    CPU on deep ML for names that cannot make the final portfolio.
    """
    technical_rows = []
    for symbol, frame in price_data.items():
        hist = frame.loc[:as_of].copy()
        if len(hist) < 260 or hist.index[-1] != as_of:
            continue
        tech = technical_analysis(hist)
        if not tech:
            continue
        technical_rows.append({
            "ticker": symbol,
            "technical_score": tech["technical_score"],
            "annualized_volatility": tech["annualized_volatility"],
            "price_asof": tech["price"],
            "rsi": tech["rsi"],
        })

    if not technical_rows:
        return pd.DataFrame()

    screen = pd.DataFrame(technical_rows).sort_values(
        ["technical_score", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)

    target_finalists = min(len(screen), max(count, int(finalists)))
    analyzed = []
    next_idx = 0
    batch_size = max(3, min(6, count // 2 or 3))

    while next_idx < len(screen) and (next_idx < target_finalists or len(analyzed) < count):
        stop = min(len(screen), max(target_finalists, next_idx + batch_size))
        batch = screen.iloc[next_idx:stop]
        for row in batch.itertuples(index=False):
            symbol = row.ticker
            with threadpool_limits(limits=1):
                forecast = forecast_return(price_data[symbol].loc[:as_of], horizon=horizon, mode=model_mode)
            pred = float(forecast.get("predicted_return", np.nan)) if forecast else np.nan
            if not np.isfinite(pred) or not forecast.get("available") or (positive_only and pred <= 0):
                next_idx += 1
                continue
            score, _ = combined_score(row.technical_score, forecast)
            analyzed.append({
                "ticker": symbol,
                "score": score,
                "technical_score": row.technical_score,
                "forecast_return": pred,
                "forecast_kind": forecast.get("kind") if forecast else "Unavailable",
                "ml_evidence_weight": forecast.get("evidence_weight", 0.0) if forecast else 0.0,
                "annualized_volatility": row.annualized_volatility,
                "price_asof": row.price_asof,
                "rsi": row.rsi,
            })
            next_idx += 1
            if progress and progress_span:
                frac = min(1.0, next_idx / max(target_finalists, 1))
                progress(progress_base + progress_span * frac, f"ML finalists · {next_idx}/{target_finalists}")
        if len(analyzed) >= count and next_idx >= target_finalists:
            break
        if next_idx >= len(screen):
            break

    if not analyzed:
        return pd.DataFrame()
    return pd.DataFrame(analyzed).sort_values(
        ["score", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)

def run_backtest(universe: Iterable[str], cfg: BacktestConfig, progress=None):
    universe = list(dict.fromkeys(s.upper().strip() for s in universe if s.strip()))
    if not universe:
        raise ValueError("Universe is empty")
    start = pd.Timestamp(cfg.start).normalize()
    end = pd.Timestamp(cfg.end).normalize() if cfg.end else pd.Timestamp.today().normalize()
    if start >= end:raise ValueError("Start date must be before end date")
    if start < pd.Timestamp("1980-01-01"):raise ValueError("Choose a start date in 1980 or later")
    if not 1 <= cfg.holdings <= 100:raise ValueError("Holdings must be between 1 and 100")
    if not np.isfinite(cfg.starting_cash) or cfg.starting_cash <= 0:raise ValueError("Starting cash must be positive")
    if not np.isfinite(cfg.transaction_cost_bps) or not 0 <= cfg.transaction_cost_bps <= 100:raise ValueError("Trading costs must be between 0 and 100 bps")
    if not isinstance(cfg.horizon,int) or not 1 <= cfg.horizon <= 252:raise ValueError("Forecast horizon must be between 1 and 252 sessions")
    schedule = _quarterly_schedule(start,end,cfg.rebalance_months,cfg.rebalance_weeks)
    cfg.benchmark = {"INTTECHA":"AB_INTL_TECH", "LU0060230025":"AB_INTL_TECH"}.get(cfg.benchmark.upper(),cfg.benchmark.upper())
    warnings=[];benchmark_errors={}
    # Fetch the optional fund before the expensive simulation and retain a clear
    # error alongside successful results if its provider is unavailable.
    try:
        ab_close=load_ab_intl_tech(start,end)
        if ab_close.attrs.get("warning"):warnings.append(ab_close.attrs["warning"])
    except ValueError as exc:
        ab_close=None;benchmark_errors["AB International Technology"]=str(exc)
    # Extra history is required so a 2021 decision can train only on pre-2021 rows.
    history_start = start - pd.DateOffset(years=8)
    comparison_symbols = ["SPY", "QQQ", "FWD"]
    direct_primary = cfg.benchmark if cfg.benchmark in comparison_symbols else None
    all_symbols = list(dict.fromkeys(universe + comparison_symbols + ([direct_primary] if direct_primary else [])))
    if progress:
        progress(.01, "Downloading historical prices")
    data, download_errors = download_prices(all_symbols, history_start, end, _download, progress)
    if "SPY" not in data:
        raise ValueError("Yahoo SPY history is unavailable for the trading calendar. The provider may be rate-limiting requests; retry later. Completed downloads are cached.")
    benchmark = data["SPY"]
    stock_data = data.subset(universe)
    if len(stock_data) < cfg.holdings:
        raise ValueError(f"Only {len(stock_data)} symbols downloaded successfully; need at least {cfg.holdings}.")

    calendar = benchmark.loc[start:end].index
    if len(calendar) < 2:
        raise ValueError("No benchmark trading sessions in the requested period")
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

    rebalance_dates = sorted(rebalance_map)
    total_rebalances = max(len(rebalance_dates), 1)
    completed_rebalances = 0

    for date in calendar:
        date = pd.Timestamp(date)
        if date in rebalance_map:
            sig = _signal_date(benchmark.index, date)
            if sig is not None:
                base = .18 + .75 * completed_rebalances / total_rebalances
                span = .75 / total_rebalances
                if progress:
                    progress(base, f"Rebalance {completed_rebalances + 1}/{total_rebalances} · technical screen")
                ranking = rank_universe(
                    stock_data, sig, cfg.horizon, cfg.positive_forecast_only,
                    count=cfg.holdings, finalists=cfg.finalists, model_mode=cfg.model_mode,
                    progress=progress, progress_base=base, progress_span=span * .9,
                )
                if not ranking.empty:
                    chosen = allocate(ranking, min(cfg.holdings, len(ranking)), cfg.profile)
                    if len(chosen)<cfg.holdings:
                        warnings.append(f"{date.date()}: only {len(chosen)} stocks passed the forecast/history rules (requested {cfg.holdings}).")
                    # Validate every execution before mutating the account.
                    for symbol in set(shares)|set(chosen.ticker):
                        if date not in stock_data[symbol].index:
                            raise ValueError(f"No execution bar for {symbol} on {date.date()}; refusing to drop a holding or invent a fill.")
                        _execution_price(stock_data[symbol],date)
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

                    portfolio_value = cash / (1 + cfg.transaction_cost_bps / 10_000)
                    # Reserve buy-side costs before sizing every position.
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
                            "forecast_kind": row.forecast_kind,
                            "ml_evidence_weight": row.ml_evidence_weight,
                            "target_weight": row.target_weight,
                            "execution_price": px,
                            "shares": qty,
                        })

                else:
                    warnings.append(f"{date.date()}: no eligible stocks; existing holdings/cash retained.")

            completed_rebalances += 1
            if progress:
                progress(.18 + .75 * completed_rebalances / total_rebalances,
                         f"Completed rebalance {completed_rebalances}/{total_rebalances}")

        value = cash
        for s, qty in shares.items():
            px = _mark_price(stock_data[s], date)
            if px is not None:
                value += qty * px
        equity_rows.append({"date": date, "portfolio": value})

    equity = pd.DataFrame(equity_rows).set_index("date")

    # Build all comparison lines. Each line is anchored to Buyntiq's portfolio
    # value on that benchmark's first available date, so later launches such as
    # FWD are compared fairly instead of being backfilled into 2021.
    comparisons = pd.DataFrame(index=equity.index)
    comparison_meta = []
    for symbol in ("SPY", "QQQ", "FWD"):
        frame = data.get(symbol)
        if frame is None or frame.empty or "Close" not in frame:
            continue
        series = _anchor_comparison(frame["Close"].dropna().astype(float), equity)
        if series is None:
            continue
        label = COMPARISON_LABELS[symbol]
        comparisons[label] = series
        stats = _comparison_stats(label, series, equity)
        if stats:
            comparison_meta.append(stats)

    if ab_close is not None and not ab_close.empty:
        ab_series = _anchor_comparison(ab_close, equity)
        if ab_series is not None:
            label = COMPARISON_LABELS["AB_INTL_TECH"]
            comparisons[label] = ab_series
            stats = _comparison_stats(label, ab_series, equity)
            if stats:
                stats["source_symbol"] = ab_close.attrs.get("source", "Official AB A USD NAV")
                comparison_meta.append(stats)
        else:benchmark_errors["AB International Technology"]="AB NAV does not overlap this backtest sufficiently."

    primary_label = COMPARISON_LABELS.get(cfg.benchmark, cfg.benchmark)
    if primary_label not in comparisons:
        benchmark_errors.setdefault(primary_label,"Benchmark history is unavailable for the requested period.")
        warnings.append(f"Primary benchmark {primary_label} is unavailable; the portfolio results are still shown.")
        equity["benchmark"]=np.nan
    else:equity["benchmark"]=comparisons[primary_label]
    equity["portfolio_return"] = equity.portfolio / cfg.starting_cash - 1
    primary_valid = equity["benchmark"].dropna()
    equity["benchmark_return"] = np.nan
    if not primary_valid.empty:
        first_primary = primary_valid.index[0]
        equity.loc[first_primary:, "benchmark_return"] = (
            equity.loc[first_primary:, "benchmark"] / float(primary_valid.iloc[0]) - 1
        )

    daily = equity.portfolio.pct_change()
    daily.iloc[0] = equity.portfolio.iloc[0] / cfg.starting_cash - 1
    daily = daily.dropna()
    ann_return = (equity.portfolio.iloc[-1] / cfg.starting_cash) ** (365.25 / max((equity.index[-1]-start).days,1)) - 1
    ann_vol = daily.std() * np.sqrt(252) if len(daily) else np.nan
    sharpe = daily.mean() * 252 / ann_vol if ann_vol and ann_vol > 0 else np.nan
    dd = equity.portfolio / equity.portfolio.cummax().clip(lower=cfg.starting_cash) - 1

    summary = {
        "start_value": cfg.starting_cash,
        "end_value": float(equity.portfolio.iloc[-1]),
        "total_return": float(equity.portfolio.iloc[-1] / cfg.starting_cash - 1),
        "benchmark_end": float(primary_valid.iloc[-1]) if len(primary_valid)>1 else np.nan,
        "benchmark_return": float(primary_valid.iloc[-1]/primary_valid.iloc[0]-1) if len(primary_valid)>1 else np.nan,
        "alpha_vs_benchmark": float(equity.portfolio.loc[primary_valid.index[-1]] / equity.portfolio.loc[primary_valid.index[0]] - primary_valid.iloc[-1]/primary_valid.iloc[0]) if len(primary_valid)>1 else np.nan,
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe_no_rf": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(dd.min()),
        "transaction_costs": float(costs_total),
        "rebalances": int(pd.DataFrame(holdings_log).rebalance_date.nunique()) if holdings_log else 0,
    }
    if progress:
        progress(1.0, "Backtest complete")
    return {
        "summary": summary,
        "equity": equity,
        "comparisons": comparisons,
        "comparison_stats": pd.DataFrame(comparison_meta),
        "holdings": pd.DataFrame(holdings_log),
        "trades": pd.DataFrame(trades),
        "warnings": warnings,
        "download_errors": download_errors,
        "benchmark_errors": benchmark_errors,
        "universe_requested": len(universe),
        "universe_downloaded": len(stock_data),
    }
