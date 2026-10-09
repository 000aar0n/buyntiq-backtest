# Ranking experiment — 2026-10-09

Rules recorded before evaluating the new model. This is research, not a claim of a profitable trading system.

## Frozen comparison

- Current US common-share/ADR listing universe, acknowledging missing delisted stocks and historical membership (survivorship bias).
- 2021-10-08 through 2026-10-08; 15 holdings; quarterly rebalancing; 63-session horizon; Balanced allocation; positive estimates required.
- Historical eligibility: estimated capitalization $2B, nominal share price $5, median 60-session dollar volume $10M. Unknown enabled-filter values excluded.
- 10 basis points per transaction, next-session opening fills, identical accounting for every strategy.
- Compare legacy technical/per-stock ML, a fixed 12-minus-1-month momentum baseline, and a fixed shared Ridge/gradient-boosting ensemble. No parameter sweep or selecting a favorable start date after results.
- Report SPY and QQQ comparisons, costs, maximum drawdown, and calendar-year returns. These dates have already been examined, so this is not an untouched investment holdout.

## Candidate design

The shared model learns 63-session log returns in excess of SPY from monthly observations across liquid stocks. Features use trailing adjusted prices, volume, and SPY context, never today's fundamentals. Price/liquidity rules also apply to training rows; the historical capitalization check applies to portfolio selection. Each date has equal total fitting weight.

Fixed ensemble: standardized Ridge (alpha 100) and shallow histogram gradient boosting (60 iterations, 7 leaves, learning rate .05, L2 10, minimum leaf 100), averaged equally. Features and targets are clipped using training data only. Training uses at most the latest 60 monthly dates, with at most 800 deterministically sampled securities per date.

At every rebalance, the last 12 matured monthly dates form a separate validation block. Training labels must end strictly before that block starts. ML is enabled only when average per-date rank correlation is positive and exceeds 12-minus-1 momentum by .02, with at least 8 evaluable dates. Otherwise the ranking explicitly falls back to momentum. A fallback return estimate is the historical median realized return of the corresponding momentum quintile; it is not labeled ML. No future outcomes choose hyperparameters.

## Promotion gate

A candidate must beat the legacy and momentum strategies and SPY after costs, show improvement in more than half of full calendar years, and not increase maximum drawdown by more than 5 percentage points versus legacy. The website must finish a full-universe run without errors. This gate is deliberately not a guarantee of future performance, and survivorship bias limits all findings. Failed gates mean no replacement of the production algorithm.
