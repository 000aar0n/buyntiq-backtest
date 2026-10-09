from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


PROFILES = {
    "Conservative": {"power": 1.30, "cap": 0.25},
    "Balanced": {"power": 0.80, "cap": 0.35},
    "Aggressive": {"power": 0.35, "cap": 0.45},
}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(value, low, high))


def scaled_points(value, bad_value, good_value, points):
    if value is None or pd.isna(value):
        return 0.0
    if value <= bad_value:
        return 0.0
    if value >= good_value:
        return float(points)
    return float((value - bad_value) / (good_value - bad_value) * points)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def _pct_return(close: pd.Series, days: int):
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1)


def _max_drawdown(close: pd.Series):
    peak = close.cummax()
    return float((close / peak - 1).min())


def _rsi_points(rsi):
    if pd.isna(rsi): return 0
    if 50 <= rsi <= 65: return 12
    if 45 <= rsi < 50: return 9
    if 65 < rsi <= 72: return 8
    if 40 <= rsi < 45: return 6
    if 30 <= rsi < 40: return 3
    if 72 < rsi <= 80: return 4
    return 0


def _volatility_points(vol):
    if vol <= 0.18: return 8
    if vol <= 0.28: return 6
    if vol <= 0.40: return 3
    if vol <= 0.55: return 1
    return 0


def _drawdown_points(dd):
    if dd >= -0.10: return 8
    if dd >= -0.20: return 6
    if dd >= -0.30: return 3
    if dd >= -0.45: return 1
    return 0


def technical_analysis(prices: pd.DataFrame) -> Optional[dict]:
    """Replicates Buyntiq's current technical scoring, using only data available as-of the slice end."""
    if prices is None or prices.empty or "Close" not in prices:
        return None
    close = prices["Close"].dropna().astype(float)
    if len(close) < 63:
        return None

    price = float(close.iloc[-1])
    ma20s = close.rolling(20).mean()
    ma50s = close.rolling(50).mean()
    ma200s = close.rolling(200).mean()
    ma20 = float(ma20s.iloc[-1]) if len(close) >= 20 else None
    ma50 = float(ma50s.iloc[-1]) if len(close) >= 50 else None
    ma200 = float(ma200s.iloc[-1]) if len(close) >= 200 else None
    r1, r3, r6 = _pct_return(close, 21), _pct_return(close, 63), _pct_return(close, 126)
    rsi = float(calculate_rsi(close).iloc[-1])
    daily = close.pct_change().dropna()
    ann_vol = float(daily.std() * np.sqrt(252)) if len(daily) >= 20 else 0.5
    dd = _max_drawdown(close)

    score = possible = 0.0
    for value, bad, good, points in [
        (r1, -0.10, 0.10, 8),
        (r3, -0.20, 0.20, 12),
        (r6, -0.30, 0.30, 16),
    ]:
        if value is not None:
            score += scaled_points(value, bad, good, points)
            possible += points
    if ma20 is not None:
        possible += 6; score += 6 if price > ma20 else 0
    if ma50 is not None:
        possible += 10; score += 10 if price > ma50 else 0
    if ma200 is not None:
        possible += 10; score += 10 if price > ma200 else 0
    if ma50 is not None and ma200 is not None:
        possible += 10; score += 10 if ma50 > ma200 else 0
    score += _rsi_points(rsi); possible += 12
    score += _volatility_points(ann_vol); possible += 8
    score += _drawdown_points(dd); possible += 8

    return {
        "price": price,
        "technical_score": clamp(score / possible * 100.0),
        "return_1m": r1,
        "return_3m": r3,
        "return_6m": r6,
        "rsi": rsi,
        "annualized_volatility": ann_vol,
        "max_drawdown": dd,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
    }


def feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal stock-only feature set mirrored from Buyntiq's production model."""
    c = prices.Close.astype(float)
    daily = np.log(c).diff()
    f = pd.DataFrame(index=c.index)
    for n in (5, 21, 63, 126):
        f[f"momentum_{n}"] = np.log(c / c.shift(n))
    for n in (20, 50, 200):
        f[f"distance_ma_{n}"] = c / c.rolling(n).mean() - 1
    for n in (21, 63):
        f[f"volatility_{n}"] = daily.rolling(n).std()
    f["volatility_ratio"] = f.volatility_21 / f.volatility_63.clip(lower=.001)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    f["rsi"] = (gain / (gain + loss).replace(0, np.nan)).fillna(.5)
    f["drawdown_126"] = c / c.rolling(126).max() - 1
    f["trend_acceleration"] = f.momentum_21 - f.momentum_63 / 3
    volume = prices.get("Volume", pd.Series(0.0, index=prices.index)).astype(float).fillna(0)
    f["relative_volume"] = (volume / volume.rolling(21).mean().replace(0, np.nan)).fillna(1).clip(0, 10)
    f["volume_trend"] = (volume.rolling(5).mean() / volume.rolling(63).mean().replace(0, np.nan)).fillna(1).clip(0, 10)
    f["downside_volatility"] = daily.clip(upper=0).rolling(63).std()
    f["return_autocorrelation"] = daily.rolling(63).corr(daily.shift(1)).fillna(0)
    return f.replace([np.inf, -np.inf], np.nan)


def _models(mode: str = "fast"):
    """Same Buyntiq model family, with a much smaller fast backtest configuration."""
    fast = mode == "fast"
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=40)),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=20 if fast else 56,
            max_depth=5 if fast else 6,
            min_samples_leaf=22 if fast else 18,
            max_features=.75 if fast else .8,
            n_jobs=1,
            random_state=42,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=35 if fast else 70,
            max_leaf_nodes=7 if fast else 9,
            l2_regularization=12,
            learning_rate=.05 if fast else .045,
            min_samples_leaf=28 if fast else 25,
            early_stopping=False,
            random_state=42,
        ),
    }


def _fit_member_predictions(x, y, scale, train, test_x, test_scale, mode: str):
    target_scaled = y.iloc[train].to_numpy() / scale.iloc[train].to_numpy()
    lo, hi = np.quantile(target_scaled, [.01, .99])
    preds = {}
    for name, model in _models(mode).items():
        model.fit(x.iloc[train], np.clip(target_scaled, lo, hi))
        preds[name] = np.clip(model.predict(test_x), -8, 8) * np.asarray(test_scale)
    return preds


def forecast_return(prices: pd.DataFrame, horizon: int = 63, mode: str = "fast") -> dict:
    """Point-in-time Buyntiq-style stock forecast.

    fast:
        One selection block plus a separate purged holdout + smaller versions of the same three
        production model families. Designed for repeated quarterly backtests.

    full:
        Three time-series selection folds, a separate purged holdout and larger
        model settings. Much slower on throttled CPUs.
    """
    if mode not in ("fast", "full"):
        raise ValueError("mode must be 'fast' or 'full'")

    f = feature_frame(prices)
    target = np.log(prices.Close.shift(-horizon) / prices.Close)
    scale = f.volatility_63.clip(lower=.003) * np.sqrt(horizon)
    joined = f.assign(_target=target, _scale=scale).dropna()

    if len(joined) < max(560 if mode == "fast" else 700, 3 * horizon + 300):
        c = prices.Close.dropna().astype(float)
        if len(c) < horizon + 252:
            return {"available": False, "predicted_return": np.nan, "evidence_weight": 0.0, "kind": "Unavailable"}
        hist = np.log(c / c.shift(horizon)).dropna().tail(504)
        pred = float(hist.median()) if len(hist) else 0.0
        return {"available": True, "predicted_return": float(np.expm1(pred)),
                "evidence_weight": 0.0, "kind": "Historical median"}

    x = joined[f.columns]
    y = joined._target
    sc = joined._scale
    current = f.iloc[[-1]].dropna(axis=1)
    cols = [c for c in x.columns if c in current.columns]
    x, current = x[cols], current[cols]
    if current.empty or not np.isfinite(current.to_numpy()).all():
        return {"available": False, "predicted_return": np.nan, "evidence_weight": 0.0, "kind": "Unavailable"}

    names = list(_models(mode))
    actual_parts = []
    pred_parts = {name: [] for name in names}

    # Keep model selection and evidence evaluation separate. Purge a full
    # horizon so no training label reaches the next evaluation period.
    holdout_n = min(252, max(horizon, 84 if mode == "fast" else 126))
    holdout_start = len(x) - holdout_n
    dev_end = holdout_start - horizon
    validation_size = min(84, max(42, horizon))
    if dev_end - horizon - validation_size < 250:
        baseline_log = float(y.tail(252).median())
        return {"available": True, "predicted_return": float(np.expm1(baseline_log)),
                "evidence_weight": 0.0, "kind": "Historical median"}
    if mode == "fast":
        val_start = dev_end - validation_size
        splits = [(np.arange(val_start - horizon), np.arange(val_start, dev_end))]
    else:
        test_size = min(max(84, horizon), (dev_end - horizon - 250) // 3)
        splits = TimeSeriesSplit(n_splits=3, test_size=test_size, gap=horizon).split(x.iloc[:dev_end])
    baseline_parts = []
    for train, test in splits:
        member = _fit_member_predictions(x, y, sc, train, x.iloc[test], sc.iloc[test].to_numpy(), mode)
        actual_parts.append(y.iloc[test].to_numpy())
        baseline_parts.append(np.full(len(test), float(y.iloc[train].tail(252).median())))
        for name in names:
            pred_parts[name].append(member[name])

    actual = np.concatenate(actual_parts)
    predictions = {name: np.concatenate(pred_parts[name]) for name in names}
    losses = {
        name: float(np.mean(np.abs(np.expm1(actual) - np.expm1(predictions[name]))))
        for name in names
    }
    inv = {k: 1 / max(v, 1e-4) for k, v in losses.items()}
    weights = {k: .5 / len(names) + .5 * inv[k] / sum(inv.values()) for k in names}
    blended = sum(weights[k] * predictions[k] for k in names)

    dev_baseline = np.concatenate(baseline_parts)
    dev_base_mae = float(np.mean(np.abs(np.expm1(actual) - np.expm1(dev_baseline))))
    dev_blend_mae = float(np.mean(np.abs(np.expm1(actual) - np.expm1(blended))))

    # Weights are now frozen. The untouched holdout alone gates ML influence.
    holdout_train = np.arange(dev_end)
    holdout = np.arange(holdout_start, len(x))
    held = _fit_member_predictions(x, y, sc, holdout_train, x.iloc[holdout], sc.iloc[holdout].to_numpy(), mode)
    held_blend = sum(weights[k] * held[k] for k in names)
    held_baseline = float(y.iloc[holdout_train].tail(252).median())
    held_actual = np.expm1(y.iloc[holdout].to_numpy())
    blend_mae = float(np.mean(np.abs(held_actual - np.expm1(held_blend))))
    base_mae = float(np.mean(np.abs(held_actual - np.expm1(held_baseline))))
    alpha = 1.0 if blend_mae < 0.97 * base_mae else 0.0
    baseline_log = float(y.tail(252).median())

    # Only refit when the independently tested ensemble will actually be used.
    # A rejected ensemble contributes nothing to ranking or the forecast.
    raw_log = np.nan
    predicted_log = baseline_log
    if alpha:
        train = np.arange(len(x))
        current_scale = max(float(f.volatility_63.iloc[-1]), .003) * np.sqrt(horizon)
        latest = _fit_member_predictions(
            x, y, sc, train, current, np.asarray([current_scale]), mode
        )
        raw_log = float(sum(weights[k] * latest[k][0] for k in names))
        predicted_log = raw_log

    skill = 0.0 if base_mae <= 1e-9 else max(0.0, 1 - blend_mae / base_mae)
    evidence = min(skill, .25) / .25 * alpha * (.55 if mode == "fast" else .8)
    return {
        "available": True,
        "predicted_return": float(np.expm1(predicted_log)),
        "raw_ml_return": float(np.expm1(raw_log)),
        "baseline_return": float(np.expm1(baseline_log)),
        "evidence_weight": float(np.clip(evidence, 0, .8)),
        "kind": ("Fast validated ensemble" if mode == "fast" else "Validated ensemble") if alpha else "Baseline fallback",
        "ml_blend": alpha,
        "dev_skill": 0.0 if dev_base_mae <= 1e-9 else 1 - dev_blend_mae / dev_base_mae,
        "holdout_skill": skill,
        "holdout_mae": blend_mae,
        "holdout_baseline_mae": base_mae,
        "holdout_rows": len(holdout),
        "mode": mode,
    }


def combined_score(technical_score: float, forecast: Optional[dict], fundamental_score: Optional[float] = None, fundamental_coverage: int = 0):
    """Buyntiq combined score. Fundamentals are optional for point-in-time safety."""
    parts = {"Technical": (float(technical_score), .55)}
    if fundamental_score is not None:
        parts["Company"] = (float(fundamental_score), .45 * min(1, max(0, fundamental_coverage) / 6))
    total = sum(w for _, w in parts.values())
    ml_weight = 0.0
    if forecast and forecast.get("available"):
        ml_weight = .25 * float(forecast.get("evidence_weight", 0.0))
    components = {name: {"score": score, "weight": weight / total * (1 - ml_weight)} for name, (score, weight) in parts.items()}
    if ml_weight:
        raw = 50 + 50 * np.tanh(float(forecast["predicted_return"]) / .20)
        components["ML"] = {"score": float(raw), "weight": ml_weight}
    score = sum(v["score"] * v["weight"] for v in components.values())
    return float(score), components


def capped_weights(raw, cap):
    raw = np.maximum(np.asarray(raw, dtype=float), 1e-12)
    if len(raw) == 0:
        raise ValueError("Cannot allocate an empty portfolio")
    cap = max(float(cap), 1 / len(raw))
    result = np.zeros(len(raw))
    free = np.ones(len(raw), dtype=bool)
    remaining = 1.0
    for _ in range(len(raw) + 1):
        proposed = raw[free] / raw[free].sum() * remaining
        high = proposed > cap + 1e-12
        if not high.any():
            result[free] = proposed
            break
        idx = np.where(free)[0][high]
        result[idx] = cap
        free[idx] = False
        remaining = 1 - result.sum()
    return result, cap


def allocate(ranked: pd.DataFrame, count: int, profile: str = "Balanced") -> pd.DataFrame:
    chosen = ranked.head(count).copy()
    if chosen.empty:
        return chosen
    settings = PROFILES[profile]
    raw = (chosen["score"].clip(lower=10) / 100.0) / chosen["annualized_volatility"].clip(lower=.10) ** settings["power"]
    weights, effective_cap = capped_weights(raw.to_numpy(), settings["cap"])
    chosen["target_weight"] = weights
    chosen["effective_cap"] = effective_cap
    return chosen
