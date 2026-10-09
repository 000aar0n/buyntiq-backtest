"""Deterministic fixtures verify timing, not claimed investment performance."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import ranking as rk


def history_fixture():
    idx = pd.bdate_range('2015-01-01', periods=2100)
    rng = np.random.default_rng(7)
    close = 50 * np.exp(np.cumsum(rng.normal(.0003, .01, len(idx))))
    frame = pd.DataFrame({'Close': close, 'Volume': 2e6,
                          'As Traded Close': close, 'Dollar Volume': close * 2e6}, index=idx)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0002, .008, len(idx)))), index=idx)
    return frame, market


def test_future_bars_cannot_change_features_or_matured_labels():
    frame, market = history_fixture()
    cutoff = frame.index[1400]
    a = rk.build_panel({'AAA': frame}, market, [cutoff], 63, 5, 1e7)
    changed = frame.copy()
    changed.loc[changed.index > cutoff, ['Close', 'As Traded Close', 'Dollar Volume']] *= 11
    changed_market = market.copy()
    changed_market.loc[changed_market.index > cutoff] *= 3
    b = rk.build_panel({'AAA': changed}, changed_market, [cutoff], 63, 5, 1e7)
    pd.testing.assert_frame_equal(rk.matured(a, cutoff), rk.matured(b, cutoff))
    pd.testing.assert_frame_equal(a.loc[a.date.eq(cutoff), rk.FEATURES],
                                  b.loc[b.date.eq(cutoff), rk.FEATURES])


def test_validation_purges_label_dates_across_all_symbols():
    dates = pd.date_range('2015-01-31', periods=60, freq='ME')
    history = pd.DataFrame([{'date': d, 'ticker': s, 'target_end': d + pd.Timedelta(days=95)}
                            for d in dates for s in ['AAA', 'BBB', 'CCC']])
    train, hold = rk.validation_split(history)
    assert hold.date.nunique() == 12
    assert train.target_end.max() < hold.date.min()
    assert set(train.ticker) == set(hold.ticker)


def test_training_sample_is_deterministic_and_date_balanced():
    f = pd.DataFrame({'date': np.repeat(pd.date_range('2010-01-31', periods=70, freq='ME'), 1000),
                      'sample_key': np.tile(np.arange(1000), 70)})
    sample = rk.training_sample(f.sample(frac=1, random_state=2))
    assert sample.date.nunique() == 60
    assert sample.groupby('date').size().eq(800).all()
    assert sample.sample_key.max() == 799


def test_failed_ml_gate_explicitly_uses_baseline(monkeypatch):
    f, m = history_fixture()
    cutoff = f.index[-1]
    panel = rk.build_panel({'AAA': f, 'BBB': f * 1.01}, m, [cutoff], 63, 5, 1e7)
    monkeypatch.setattr(rk, 'fit_model', lambda _: None)
    predictions, diagnostics = rk.predict_panel(panel, cutoff, 'pooled')
    assert not predictions.empty
    assert not diagnostics['ml_enabled']
    assert predictions.ml_evidence_weight.eq(0).all()
    assert predictions.forecast_kind.str.contains('not ML').all()
    assert np.isfinite(predictions.forecast_return).all()


def test_enabled_size_filter_excludes_unknown_even_for_highest_score():
    rows = pd.DataFrame([{'ticker': s, 'score': 100-i, 'forecast_return': .1}
                         for i, s in enumerate(['UNKNOWN', 'SMALL', 'GOOD'])])
    class Size:
        def estimate(self, s, *args):
            return (None, 'Unknown size') if s == 'UNKNOWN' else ({'estimated_market_cap': 1e8 if s == 'SMALL' else 3e9}, None)
    cfg = SimpleNamespace(positive_forecast_only=True, min_market_cap=2e9, holdings=1)
    excluded = Counter()
    out = rk.eligible_ranking(rows, {s: pd.DataFrame() for s in rows.ticker},
                             pd.Timestamp('2024-01-01'), cfg, Size(), excluded)
    assert out.ticker.tolist() == ['GOOD']
    assert sum(excluded.values()) == 2
