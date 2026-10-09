# Buyntiq Backtest

A standalone historical simulator using Buyntiq-style technical scoring, Ridge/Extra Trees/gradient-boosting forecasts, and score/volatility portfolio weights. This does not place live trades.

## Controls

- **Universe:** Entire US stock universe (default), the existing technology sample, or custom tickers.
- **Period:** 1–20 years back from the selected end date, or custom start/end dates.
- **Rebuild portfolio:** weekly, every 1/2/3/6 months, or annually. Model inputs and forecasts are refreshed at each historical rebalance.
- **Forecast horizon:** match the rebalance interval or separately choose 1m/3m/6m/1y. Matching weekly uses five trading sessions.
- **Eligibility:** minimum estimated historical market cap $2B, nominal historical share price $5, and trailing 60-session median dollar volume $10M. All three thresholds are adjustable; 0 explicitly disables a gate.
- **Holdings:** 3–50, with Conservative/Balanced/Aggressive weighting, optional positive forecasts, fractional/whole shares, and trading costs.
- **Model mode:** Fast uses smaller models and one selection block; Full uses three selection folds. Both now gate ML influence on a separate chronological holdout with a full-horizon purge. Baselines use only each training window’s matured outcomes. Both are historical approximations to the live app, not an exact replay of every production feature.

Results remain visible after settings change or CSV downloads. The saved run is labeled with its actual settings. Run again to apply changes. Export holdings, trades, daily equity, benchmark comparisons/curves, settings, and unavailable histories.

## Full US universe

`universe.py` reads both official Nasdaq Trader symbol directories:

- https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
- https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt

The universe covers current US exchange-listed stocks/ADRs, including non-tech sectors. Known test issues, ETFs, preferred shares, warrants, rights and units are excluded. Class-share dots are mapped to Yahoo hyphens (BRK.B → BRK-B). There is no 100-stock cap, market-cap ranking, or hidden sample in the listing download. Selection then applies the explicit eligibility filters. This is not an OTC or historical delisted-security database.

Listing data are cached for 24 hours. Price requests run in batches of 48 with at most four Yahoo download threads. Adjusted histories are cached per ticker/date range on disk for 24 hours; at most 64 histories are retained in memory by each price store. Every available stock is checked for price/liquidity eligibility before the technical screen, and only the initial top finalists receive expensive ML analysis, expanding when too few qualify. No-history and provider failures are reported with counts and a CSV. Three consecutive empty download batches stop the run instead of silently testing a truncated universe. Completed downloads can be reused on retry.

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

The simulator liquidates and rebuilds at each rebalance, charging costs on both buys and sells. Buy costs are reserved before sizing the basket. Whole-share mode can leave cash or fewer positions when allocation budgets cannot buy a share. If too few stocks pass the history/forecast rules, the smaller portfolio is reported; if none qualify, existing holdings are closed at the rebalance open and the account holds cash with a warning. No unavailable forecasts are recommended, even when the positive-only filter is disabled.

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

## October 2026 validation audit

Fixed: the stock directory filter admitted named closed-end funds (including DHY and FTHY); it now excludes fund/portfolio/investment-trust security names while preserving ordinary REITs. This is name-based classification, not a complete security master: unusually named vehicles and historical SPAC/predecessor identities remain a limitation.

Fixed: validation baselines previously used outcomes later than the validation training cutoff. Full mode reserved but never evaluated its holdout. Ensemble weights are now selected before the holdout, and ML influence is gated by holdout performance against a training-only baseline. Overlapping horizon labels mean holdout rows are not independent trials.

The supplied losing run had 245 of 315 position rows with zero ML influence. That reflects baseline/historical-median fallbacks; it is not proof of an execution defect. Selection is still technical pre-screening followed by forecasts for finalists, not ML ranking of every listing. Low liquidity, sector exposure, current-membership survivorship bias, and full liquidation/repurchase costs can materially affect results. These fixes do not establish that the previous loss was caused entirely by bugs or promise positive returns. Previous exports must be rerun to reflect the fixes.

## Historical size, price and liquidity filters

The default eligibility floor is **$2 billion estimated market cap**, **$5 nominal share price**, and **$10 million median daily dollar volume over 60 trading sessions**. These are initial research settings, not optimized thresholds or evidence of outperformance. All are evaluated using the prior session's information before new positions are selected. Size checks occur on technically ranked candidates before ML; no current market-cap shortcut or silent relaxation of the floors is allowed. The US listing universe is still uncapped.

`eligibility.py` downloads the SEC ticker map and the `dei/EntityCommonStockSharesOutstanding` concept on demand. It takes the latest report observation, using its first published count, from a domestic 10-K/10-Q filed strictly before the signal day. Counts older than 180 calendar days are rejected. The count is adjusted for stock splits since its observation date and multiplied by the historical nominal price. This is a **historical estimate**, not an exact daily capitalization: issuance/buybacks between reports, incomplete corporate-action data and current ticker mappings can affect it. Current share counts and future filings are never substituted. This does not add fundamentals to the ranking model.

The SEC API aggregates entity-wide facts. Multiple listed tickers for one issuer, ambiguous share counts, foreign-only filings/ADRs with unverified share ratios, missing mappings, and missing data are conservatively excluded when the size gate is enabled. Coverage is incomplete and current mappings retain survivorship/identity limitations. A pre-2009 start with the size filter enabled is rejected. HTTP/provider failures stop the run rather than silently disabling the filter. SEC responses are cached for 24 hours and requests are limited to at most four per second per process. An optional `SEC_USER_AGENT` environment variable can supply the operator's SEC contact identifier; no API key is required.

Yahoo prices are fetched with `auto_adjust=False, actions=True`. Total-return OHLC is reconstructed using `Adj Close / Close`, preserving the previous return/execution basis. Separate `As Traded Close`, `Split Factor` and `Dollar Volume` fields support eligibility. Histories/actions are requested through today even for older backtest end dates because Yahoo's price/share units reflect subsequent splits. Subsequent split factors only undo that unit conversion; prices after the requested end are discarded before the simulation and never drive a selection. Price-cache keys are versioned, so old adjusted-only files cannot masquerade as nominal-price data. The first run after this update needs new price downloads.

Holdings now export the historical price, median dollar volume, estimated cap, share-count observation and filing dates, and the size source. **Why stocks were excluded** displays downloadable counts per rebalance. Price/liquidity counts cover the universe; size/forecast counts cover candidates actually considered, not every listing. Threshold changes require a new run; saved results retain their original settings. If too few stocks qualify, the app reports the smaller basket; if none qualify, it holds cash. The cap gate does not establish that poor earlier performance came from penny stocks, nor does it replace the proposed shared ranking model.
