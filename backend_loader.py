"""Load one coherent source snapshot, isolated from stale Streamlit imports.

Each source digest gets its own package, so a running older session keeps its
own globals. We do not reload shared modules or silently discard new settings.
"""
from dataclasses import fields
import hashlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

_MODULES = ('price_store', 'eligibility', 'strategy', 'ranking', 'ab_benchmark', 'universe', 'backtest')
_LOCK = threading.RLock()
_REQUIRED_FILTERS = {'min_market_cap', 'min_price', 'min_dollar_volume'}


def load_backend(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    # Compile the bytes we hash, not a timestamp-based .pyc or a module imported
    # by a different app. Relative imports bind exclusively within this package.
    sources = {name: (root / f'{name}.py').read_bytes() for name in _MODULES}
    digest = hashlib.sha256()
    for name, source in sources.items():
        digest.update(name.encode() + b'\0' + source + b'\0')
    build = digest.hexdigest()
    package_name = '_buyntiq_engine_' + build
    with _LOCK:
        cached = sys.modules.get(package_name)
        if cached is not None and hasattr(cached, '_bundle'):
            return cached._bundle
        package = ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
        package.__package__ = package_name
        sys.modules[package_name] = package
        loaded = []
        try:
            for name, source in sources.items():
                full_name = package_name + '.' + name
                module = ModuleType(full_name)
                module.__package__ = package_name
                module.__file__ = str(root / f'{name}.py')
                module.__spec__ = ModuleSpec(full_name, loader=None, origin=module.__file__)
                sys.modules[full_name] = module
                loaded.append(full_name)
                setattr(package, name, module)
                exec(compile(source, module.__file__, 'exec'), module.__dict__)
            engine = package.backtest
            supported = {f.name for f in fields(engine.BacktestConfig)}
            if engine.BACKTEST_API_VERSION != 2 or not _REQUIRED_FILTERS.issubset(supported):
                raise RuntimeError('The backend source does not support the current eligibility settings.')
            bundle = SimpleNamespace(engine=engine, universe=package.universe, build=build[:12])
            package._bundle = bundle
            return bundle
        except Exception as exc:
            for name in loaded:
                sys.modules.pop(name, None)
            sys.modules.pop(package_name, None)
            raise RuntimeError('The deployed backtest files are incomplete or incompatible. '
                               'Wait for the GitHub update to finish, then reboot the Streamlit app. '
                               'The eligibility filters have not been bypassed.') from exc
