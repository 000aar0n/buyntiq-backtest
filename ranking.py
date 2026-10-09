"""Experimental pooled ranking, with calendar-based label purging.

Every prediction is reconstructed using only observations available by its
signal close. This module has no network access and never substitutes data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ["mom21", "mom63", "mom126", "mom12_1", "relative63",
            "relative126", "vol21", "vol63", "downside63", "drawdown126",
            "ma50", "ma200", "volume_ratio", "log_liquidity",
            "market63", "market_ma200", "market_vol63"]
STRATEGIES = {"legacy": "Current technical + per-stock ML",
              "momentum": "12–1 month momentum baseline",
              "pooled": "Shared ML ranking (experimental)"}


def features(frame, market):
    """Trailing features only; no forward filling across missing stock bars."""
    c = frame.Close.reindex(market.index)
    ret = c.pct_change(fill_method=None)
    mr = market.pct_change(fill_method=None)
    f = pd.DataFrame(index=market.index)
    for n in (21, 63, 126):
        f[f"mom{n}"] = c / c.shift(n) - 1
    f["mom12_1"] = c.shift(21) / c.shift(252) - 1
    for n in (63, 126):
        f[f"relative{n}"] = f[f"mom{n}"] - (market / market.shift(n) - 1)
    for n in (21, 63):
        f[f"vol{n}"] = ret.rolling(n).std() * np.sqrt(252)
    f["downside63"] = ret.clip(upper=0).pow(2).rolling(63).mean().pow(.5) * np.sqrt(252)
    f["drawdown126"] = c / c.rolling(126).max() - 1
    for n in (50, 200):
        f[f"ma{n}"] = c / c.rolling(n).mean() - 1
    v = frame.Volume.reindex(market.index)
    f["volume_ratio"] = v.rolling(21).mean() / v.rolling(126).mean()
    dv = frame.get("Dollar Volume", pd.Series(index=frame.index, dtype=float)).reindex(market.index)
    f["median_daily_dollar_volume"] = dv.rolling(60).median()
    f["log_liquidity"] = np.log1p(f.median_daily_dollar_volume)
    f["market63"] = market / market.shift(63) - 1
    f["market_ma200"] = market / market.rolling(200).mean() - 1
    f["market_vol63"] = mr.rolling(63).std() * np.sqrt(252)
    f["historical_price"] = frame.get("As Traded Close", pd.Series(index=frame.index, dtype=float))
    f["price_asof"] = c
    # Rolling count prevents an isolated missing liquidity observation passing.
    f["liquidity_complete"] = dv.rolling(60).count().eq(60)
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(price_data, market, signals, horizon, min_price, min_dollar_volume, progress=None):
    calendar = market.index
    monthly = pd.DatetimeIndex(pd.Series(calendar, index=calendar).resample("ME").last().dropna())
    dates = monthly.union(pd.DatetimeIndex(signals)).intersection(calendar)
    future_dates = pd.Series(calendar, index=calendar).shift(-horizon)
    benchmark_target = np.log(market.shift(-horizon) / market)
    rows = []
    for i, (symbol, frame) in enumerate(price_data.items(), 1):
        f = features(frame, market).reindex(dates)
        f = f.dropna(subset=FEATURES + ["price_asof"])
        if min_price:
            f = f[f.historical_price.ge(min_price)]
        if min_dollar_volume:
            f = f[f.median_daily_dollar_volume.ge(min_dollar_volume) & f.liquidity_complete]
        if not f.empty:
            c = frame.Close.reindex(calendar)
            target = np.log(c.shift(-horizon) / c)
            f["target"] = target - benchmark_target
            f["absolute_target"] = target
            f["market_target"] = benchmark_target
            f["target_end"] = future_dates
            f["training_date"] = f.index.isin(monthly)
            f["ticker"] = symbol
            f.index.name = "date"
            rows.append(f.reset_index())
        if progress and (i % 100 == 0 or i == len(price_data)):
            progress(.18 + .10 * i / max(len(price_data), 1),
                     f"Preparing shared features · {i:,}/{len(price_data):,} stocks")
    if not rows:
        raise ValueError("No stocks have enough verified history for the shared ranking experiment.")
    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"])
    for c in FEATURES + ["target", "absolute_target", "market_target"]:
        panel[c] = panel[c].astype("float32")
    # Deterministic sampling is independent of future returns and row ordering.
    panel["sample_key"] = pd.util.hash_pandas_object(panel.ticker, index=False).to_numpy()
    return panel


def matured(panel, as_of):
    return panel[panel.training_date & panel.date.lt(as_of) & panel.target_end.le(as_of)
                 & np.isfinite(panel.target) & np.isfinite(panel.absolute_target)].copy()


def validation_split(history):
    dates = sorted(history.date.unique())
    if len(dates) < 36:
        return history.iloc[:0], history.iloc[:0]
    hold_dates = dates[-12:]
    hold = history[history.date.isin(hold_dates)].copy()
    # Across pooled securities, row-number gaps are not valid time purges.
    train = history[history.target_end.lt(hold_dates[0])].copy()
    return train, hold


def training_sample(frame):
    dates = sorted(frame.date.unique())[-60:]
    sampled = frame[frame.date.isin(dates)].sort_values(["date", "sample_key"])
    return sampled.groupby("date", sort=False).head(800)


@dataclass
class Ensemble:
    low: np.ndarray
    high: np.ndarray
    scale: StandardScaler
    ridge: Ridge
    tree: HistGradientBoostingRegressor

    def predict(self, frame):
        x = np.clip(frame[FEATURES].to_numpy(dtype=float), self.low, self.high)
        return .5 * self.ridge.predict(self.scale.transform(x)) + .5 * self.tree.predict(x)


def fit_model(train):
    train = training_sample(train)
    if len(train) < 1000 or train.date.nunique() < 18:
        return None
    x = train[FEATURES].to_numpy(dtype=float)
    low, high = np.quantile(x, [.01, .99], axis=0)
    x = np.clip(x, low, high)
    y = train.target.to_numpy(dtype=float)
    y = np.clip(y, *np.quantile(y, [.05, .95]))
    weights = 1 / train.groupby("date").date.transform("size").to_numpy()
    weights *= len(weights) / weights.sum()
    scale = StandardScaler().fit(x, sample_weight=weights)
    ridge = Ridge(alpha=100).fit(scale.transform(x), y, sample_weight=weights)
    tree = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=7,
        learning_rate=.05, l2_regularization=10, min_samples_leaf=100,
        max_bins=63, early_stopping=False, random_state=42).fit(x, y, sample_weight=weights)
    return Ensemble(low, high, scale, ridge, tree)


def rank_ic(frame, values):
    f = frame[["date", "target"]].copy()
    f["prediction"] = np.asarray(values)
    values = []
    for _, group in f.groupby("date"):
        if len(group) >= 20 and group.target.nunique() > 1 and group.prediction.nunique() > 1:
            values.append(group.target.rank().corr(group.prediction.rank()))
    return float(np.mean(values)) if values else np.nan, len(values)


def predict_panel(panel, as_of, strategy):
    current = panel[panel.date.eq(as_of)].copy()
    history = matured(panel, as_of)
    diagnostics = {"signal_date": as_of, "strategy": strategy, "ml_enabled": False,
                   "training_rows": len(history), "holdout_dates": 0,
                   "ml_rank_ic": np.nan, "momentum_rank_ic": np.nan}
    if current.empty or history.empty:
        diagnostics["reason"] = "Insufficient matured training observations"
        return current.iloc[:0], diagnostics
    model = None
    if strategy == "pooled":
        train, hold = validation_split(history)
        with threadpool_limits(limits=1):
            candidate = fit_model(train)
            if candidate is not None and not hold.empty:
                ml_ic, n = rank_ic(hold, candidate.predict(hold))
                base_ic, _ = rank_ic(hold, hold.mom12_1)
                diagnostics.update(ml_rank_ic=ml_ic, momentum_rank_ic=base_ic, holdout_dates=n)
                if n >= 8 and np.isfinite(ml_ic) and np.isfinite(base_ic) and ml_ic > 0 and ml_ic > base_ic + .02:
                    model = fit_model(history)
    if model is not None:
        with threadpool_limits(limits=1):
            raw = model.predict(current)
        # Each monthly date contributes once to the market-return estimate.
        market_return = history.groupby("date").market_target.first().median()
        current["forecast_return"] = np.expm1(np.clip(raw + market_return, -2, 2))
        current["score"] = pd.Series(raw, index=current.index).rank(pct=True) * 100
        current["forecast_kind"] = "Shared ML ensemble"
        current["ml_evidence_weight"] = 1.0
        diagnostics.update(ml_enabled=True, reason="Passed purged ranking validation")
    else:
        # Historical, observable momentum-bucket returns; never call this ML.
        hist = history.copy()
        hist["bucket"] = np.minimum(4, (hist.groupby("date").mom12_1.rank(pct=True) * 5).astype(int))
        bucket_returns = hist.groupby(["date", "bucket"]).absolute_target.median().groupby("bucket").median()
        percentile = current.mom12_1.rank(pct=True)
        buckets = np.minimum(4, (percentile * 5).astype(int))
        current["forecast_return"] = np.expm1(buckets.map(bucket_returns))
        current["score"] = percentile * 100
        current["forecast_kind"] = "Historical momentum bucket (not ML)"
        current["ml_evidence_weight"] = 0.0
        diagnostics["reason"] = "Momentum baseline" if strategy == "momentum" else "ML did not establish a ranking advantage; momentum fallback"
    current["technical_score"] = current.mom12_1.rank(pct=True) * 100
    current["annualized_volatility"] = current.vol63
    current["rsi"] = np.nan
    return current.sort_values(["score", "ticker"], ascending=[False, True]), diagnostics


def eligible_ranking(predictions, price_data, as_of, cfg, size_store, exclusions, progress=None):
    chosen = []
    for row in predictions.to_dict("records"):
        pred = row["forecast_return"]
        if not np.isfinite(pred) or (cfg.positive_forecast_only and pred <= 0):
            exclusions["Unavailable or non-positive forecast"] += 1
            continue
        if cfg.min_market_cap:
            if progress:
                progress(f"Checking historical size · {row['ticker']} · {len(chosen)}/{cfg.holdings} eligible")
            size, reason = size_store.estimate(row["ticker"], price_data[row["ticker"]].loc[:as_of], as_of)
            if reason or not size or size["estimated_market_cap"] < cfg.min_market_cap:
                exclusions[reason or "Below minimum estimated market cap"] += 1
                continue
            row.update(size)
        chosen.append(row)
        if len(chosen) >= cfg.holdings:
            break
    return pd.DataFrame(chosen)
