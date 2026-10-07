# Buyntiq Backtest

A standalone, point-in-time portfolio backtester based on the current Buyntiq portfolio-builder rules.

## What it does

- Starts from a historical date (default: 5 years ago).
- Rebuilds the portfolio every 3 months.
- Uses only price/volume rows available before each trade.
- Mirrors Buyntiq's technical score.
- Uses the same Buyntiq-style ML model family: Ridge, Extra Trees, and histogram gradient boosting.
- Uses Buyntiq's score/volatility allocation and Conservative/Balanced/Aggressive position caps.
- Can require a positive 3-month forecast.
- Trades on the next market session after the signal date to avoid same-close look-ahead.
- Tracks transaction costs, holdings, trades, equity curve, drawdown, and benchmark return.
- Compares against SPY or QQQ.

## Why company fundamentals are not included by default

Buyntiq's live app calls a company-data provider for today's fundamentals. Reusing today's revenue growth, margins, forward P/E, debt/equity, etc. at a 2021 rebalance would be look-ahead bias.

This backtester therefore uses technical + ML scoring unless you later add a true **point-in-time fundamentals** dataset (for example SEC filing data keyed by filing date).

That makes the historical test less flashy, but much more honest.

## Important backtest caveats

1. **Survivorship bias** — a hand-entered list of today's stocks excludes companies that disappeared during the period.
2. **Corporate actions** — yfinance auto-adjusted OHLC handles most splits/dividends, but not every real-world execution detail.
3. **Execution** — trades are simulated at the next session's adjusted open (falling back to close).
4. **Slippage/fees** — configurable in basis points; default 10 bps per trade.
5. **ML compute** — the app retrains candidates at each rebalance, so a large universe can be slow.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Suggested first test

- Start: five years ago
- End: today
- Rebalance: every 3 months
- Holdings: 15
- Risk: Balanced
- Benchmark: SPY and then QQQ
- Require positive forecast: on
- Transaction costs: 10 bps

## Repo layout

- `app.py` — Streamlit UI
- `backtest.py` — point-in-time rebalance/trading engine
- `strategy.py` — Buyntiq technical score, ML forecast, combined score, and allocation rules
- `requirements.txt` — dependencies
