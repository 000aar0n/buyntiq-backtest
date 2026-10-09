from dataclasses import asdict
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest
import backend_loader

ROOT = Path(__file__).resolve().parents[1]


def test_stale_import_reproduces_error_but_isolated_backend_accepts_filters(monkeypatch):
    stale = ModuleType('backtest')
    class OldConfig:
        def __init__(self, start='2021-10-07'):
            self.start = start
    stale.BacktestConfig = OldConfig
    monkeypatch.setitem(sys.modules, 'backtest', stale)
    with pytest.raises(TypeError, match='min_market_cap'):
        stale.BacktestConfig(min_market_cap=2e9)
    # Deliberately poison the old dependency imports too.
    for name in ['price_store', 'eligibility', 'strategy', 'universe', 'ab_benchmark']:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    bundle = backend_loader.load_backend()
    cfg = bundle.engine.BacktestConfig(min_market_cap=2e9, min_price=5., min_dollar_volume=1e7)
    assert asdict(cfg)['min_market_cap'] == 2e9
    assert bundle.engine.HistoricalSizeStore.__module__.startswith('_buyntiq_engine_')
    assert bundle.engine.download_prices.__module__.startswith('_buyntiq_engine_')
    assert bundle.engine.forecast_return.__module__.startswith('_buyntiq_engine_')
    assert sys.modules['backtest'] is stale  # No reload mutates an older run.
    assert backend_loader.load_backend() is bundle


def test_source_change_loads_new_bundle_and_keeps_old_bundle_intact(tmp_path):
    for name in backend_loader._MODULES:
        shutil.copyfile(ROOT / f'{name}.py', tmp_path / f'{name}.py')
    original = backend_loader.load_backend(tmp_path)
    p = tmp_path / 'backtest.py'
    p.write_text(p.read_text() + '\nTEST_BUILD_MARKER = "updated"\n')
    updated = backend_loader.load_backend(tmp_path)
    assert updated.build != original.build
    assert updated.engine.TEST_BUILD_MARKER == 'updated'
    assert not hasattr(original.engine, 'TEST_BUILD_MARKER')


def test_incomplete_backend_is_rejected_without_dropping_filters(tmp_path):
    for name in backend_loader._MODULES:
        shutil.copyfile(ROOT / f'{name}.py', tmp_path / f'{name}.py')
    p = tmp_path / 'backtest.py'
    p.write_text(p.read_text().replace('    min_market_cap: float = 2_000_000_000.0\n', ''))
    with pytest.raises(RuntimeError, match='not been bypassed'):
        backend_loader.load_backend(tmp_path)


def test_streamlit_starts_with_stale_global_backend(monkeypatch):
    from streamlit.testing.v1 import AppTest
    stale = ModuleType('backtest')
    monkeypatch.setitem(sys.modules, 'backtest', stale)
    bundle = backend_loader.load_backend()
    captured = []
    def checked_run(symbols, cfg, progress):
        captured.append(asdict(cfg))
        raise ValueError('fixture reached current engine without network')
    monkeypatch.setattr(bundle.engine, 'run_backtest', checked_run)
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=30).run()
    assert not app.exception and not app.error
    assert any(x.label == 'Minimum estimated market cap ($ billions)' for x in app.number_input)
    assert any(x.label == 'Run backtest' for x in app.button)

    next(x for x in app.selectbox if x.label == 'Stock universe').select('Custom tickers').run()
    app.text_area[0].set_value('AAPL,MSFT,NVDA').run()
    next(x for x in app.button if x.label == 'Run backtest').click().run()
    assert captured[0]['min_market_cap'] == 2e9
    assert captured[0]['min_price'] == 5
    assert captured[0]['min_dollar_volume'] == 1e7
    assert not app.exception
    assert 'fixture reached current engine' in app.error[0].value
