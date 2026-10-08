# Buyntiq Backtest

A standalone historical simulator using Buyntiq-style technical scoring, Ridge/Extra Trees/gradient-boosting forecasts, and score/volatility portfolio weights. This does not place live trades.

## Controls

- **Universe:** Entire US stock universe (default), the existing technology sample, or custom tickers.
- **Period:** 1–20 years back from the selected end date, or custom start/end dates.
- **Rebuild portfolio:** weekly, every 1/2/3/6 months, or annually. Model inputs and forecasts are refreshed at each historical rebalance.
- **Forecast horizon:** match the rebalance interval or separately choose 1m/3m/6m/1y. Matching weekly uses five trading sessions.
- **Holdings:** 3–50, with Conservative/Balanced/Aggressive weighting, optional positive forecasts, fractional/whole shares, and trading costs.
- **Model mode:** Fast uses smaller models and one chronological validation block; Full uses three folds. Both are historical approximations to the live app, not an exact replay of every production feature.

Results remain visible after settings change or CSV downloads. The saved run is labeled with its actual settings. Run again to apply changes. Export holdings, trades, daily equity, benchmark comparisons/curves, settings, and unavailable histories.

## Full US universe

`universe.py` reads both official Nasdaq Trader symbol directories:

- https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
- https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt

The universe covers current US exchange-listed stocks/ADRs, including non-tech sectors. Known test issues, ETFs, preferred shares, warrants, rights and units are excluded. Class-share dots are mapped to Yahoo hyphens (BRK.B → BRK-B). There is no 100-stock cap, market-cap ranking, or hidden sample. This is not an OTC or historical delisted-security database.

Listing data are cached for 24 hours. Price requests run in batches of 48 with at most four Yahoo download threads. Adjusted histories are cached per ticker/date range on disk for 24 hours; at most 64 histories are retained in memory by each price store. Every available stock receives the technical screen, and only the initial top finalists receive expensive ML analysis, expanding when too few qualify. No-history and provider failures are reported with counts and a CSV. Three consecutive empty download batches stop the run instead of silently testing a truncated universe. Completed downloads can be reused on retry.

The first full-universe download and long/weekly backtests can take a long time on Streamlit Cloud. A small finalist count limits model training, not the universe screened. Yahoo may rate-limit or lack individual symbols; the app cannot guarantee provider availability.

## AB International Technology / INTTECHA

The benchmark uses **AB International Technology Portfolio, Class A USD**, ISIN **LU0060230025**. `INTTECHA` and the ISIN are accepted as configuration aliases for `AB_INTL_TECH`.

Official share-class page:
https://www.alliancebernstein.com/americas/en/investor/funds/equities/ab-international-technology.a.LU0060230025.html

The loader uses the same public historical NAV feed as AB's fund page:
https://webapi.alliancebernstein.com/v2/funds/americas/en/investor/LU0060230025/historical-navs?freq=daily

This is the actual USD share-class NAV, not a German exchange quote and not an unconverted EUR proxy. The old Morningstar portal lookup and XAY5 fallbacks have been removed. NAV responses are cached for six hours. A provider failure may use a saved response with an explicit warning and last reported date; the curve never extends past the last published observation. If no data are available, the app displays the reason and retains the portfolio results and other benchmarks.

SPY, QQQ, FWD and AB are compared over their available overlapping dates. Each curve is anchored to Buyntiq's closing value on its first available day. The return-edge calculation uses that same start/end for both series, and is expressed in percentage points. FWD is not backfilled before its available history. The first-day intraday portfolio return is included in total account return, but not in close-to-close benchmark comparisons.

## Timing and execution

Signals use price/volume history through the preceding market session, including for the initial purchase. Scheduled dates roll forward to the next SPY session. Month schedules remain anchored to the original date, avoiding month-end drift. Transactions use adjusted opening prices; missing openings or missing held-stock execution bars stop the run rather than inventing a fill or deleting a position.

The simulator liquidates and rebuilds at each rebalance, charging costs on both buys and sells. Buy costs are reserved before sizing the basket. Whole-share mode can leave cash or fewer positions when allocation budgets cannot buy a share. If too few stocks pass the history/forecast rules, the smaller portfolio is reported; if none qualify, existing holdings/cash are retained with a warning. No unavailable forecasts are recommended, even when the positive-only filter is disabled.

## Limits

- Current membership has **survivorship bias**: delisted companies and historical membership changes are missing.
- Today's fundamentals are not reused in historical selections. This app scores technicals plus ML, not historical company financials.
- Models are retrained using data available at each historical decision, but the strategy rules were designed later. This is a retrospective simulation, not a live record.
- Yahoo-adjusted prices approximate split/dividend total returns. Fractional adjusted units and fills are not exact broker execution. Taxes, liquidity constraints and market impact are not modeled beyond the configured cost allowance.
- An apparent win over a benchmark does not establish future outperformance.

## Run and test

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py

# Development regression tests (deterministic fixtures, no network required):
python -m pip install pytest
python -m pytest -q
```

Tests cover first-session investing, monthly/weekly schedules, fee accounting, missing execution data, AB NAV parsing and overlap, listing filters, batched caching, and persistent Streamlit results. Fixtures are isolated test inputs; the application does not generate synthetic market data.
