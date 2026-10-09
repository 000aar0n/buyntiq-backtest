"""Evaluate preserved real Yahoo/SEC responses; never generates market prices.

Usage: python studies/run_cached.py /path/to/raw /path/to/output momentum pooled
The cached universe is a technology subset, not the full-US promotion test.
"""
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import backtest as bt
import eligibility
import price_store

raw, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
price_store.CACHE = output / 'cache'
frames = {}
for path in sorted((raw / 'prices').glob('*.json.gz')):
    try:
        z = json.load(gzip.open(path, 'rt'))['chart']['result'][0]
        idx = pd.to_datetime(z['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        f = pd.DataFrame(z['indicators']['quote'][0], index=idx).rename(columns=str.title)
        f['Adj Close'] = z['indicators']['adjclose'][0]['adjclose']
        f['Stock Splits'] = 0.
        for ev in z.get('events', {}).get('splits', {}).values():
            d = pd.Timestamp(ev['date'], unit='s', tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
            if d in f.index:
                f.loc[d, 'Stock Splits'] = float(ev['numerator']) / float(ev['denominator'])
        frames[path.name.removesuffix('.json.gz')] = price_store.prepare_yahoo_history(f)
    except (KeyError, IndexError, TypeError):
        continue
mapping = json.loads((raw / 'sec_tickers.json').read_text())
def fetch(url):
    if 'company_tickers' in url:
        return mapping
    cik = int(url.split('CIK')[1].split('/')[0])
    path = raw / 'sec' / f'{cik}.json.gz'
    if not path.exists():
        return None
    return json.load(gzip.open(path, 'rt')).get('facts', {}).get('dei', {}).get('EntityCommonStockSharesOutstanding')
bt.HistoricalSizeStore = lambda: eligibility.HistoricalSizeStore(fetch)
bt._download = lambda symbols, start, end: {s: frames[s].loc[start:end] for s in symbols if s in frames}
bt.load_ab_intl_tech = lambda *args: pd.Series(dtype=float)
universe = sorted(set(frames) - {'SPY', 'QQQ', 'XLK', 'FWD'})
last = [0.]
def progress(fraction, message):
    if time.monotonic() - last[0] > 10:
        print(round(fraction*100), message, flush=True)
        last[0] = time.monotonic()
for strategy in sys.argv[3:]:
    cfg = bt.BacktestConfig(start='2021-10-08', end='2026-10-06', ranking_strategy=strategy)
    started = time.monotonic()
    result = bt.run_backtest(universe, cfg, progress)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f'{strategy}_{key}.csv', index=key in ('equity', 'comparisons'))
    details = {**result['summary'], 'seconds': time.monotonic()-started, 'scope': 'Cached current technology stocks only',
               'universe_requested': len(universe), 'universe_downloaded': result['universe_downloaded']}
    (output / f'{strategy}_summary.json').write_text(json.dumps(details, indent=2))
    print(strategy, json.dumps(details), flush=True)
