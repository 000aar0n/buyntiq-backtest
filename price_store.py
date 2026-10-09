"""Bounded-memory price downloads, cached per symbol/date range on local disk."""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get('BUYNTIQ_BACKTEST_CACHE',Path(tempfile.gettempdir())/'buyntiq-backtest-cache-v1'))


class PriceStore(Mapping):
    def __init__(self, paths):
        self.paths=dict(paths)
        self._recent=OrderedDict()
    def __len__(self):return len(self.paths)
    def __iter__(self):return iter(self.paths)
    def __contains__(self,symbol):return symbol in self.paths
    def __getitem__(self,symbol):
        if symbol in self._recent:
            self._recent.move_to_end(symbol)
            return self._recent[symbol]
        frame=pd.read_csv(self.paths[symbol],index_col=0,parse_dates=True)
        self._recent[symbol]=frame
        if len(self._recent)>64:self._recent.popitem(last=False)
        return frame
    def subset(self,symbols):return PriceStore({s:self.paths[s] for s in symbols if s in self.paths})


def prepare_yahoo_history(frame):
    """Preserve total-return OHLC; add price/volume in historical trading units.

    Input is Yahoo auto_adjust=False with corporate actions THROUGH TODAY,
    even for a backtest ending earlier. Future splits only undo Yahoo's unit
    conversion; future returns never enter features or eligibility.
    """
    required = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Stock Splits'}
    if not required.issubset(frame):
        raise ValueError('Yahoo did not return prices and corporate actions needed for historical filters')
    f = frame.copy().sort_index()
    splits = pd.to_numeric(f['Stock Splits'], errors='coerce').fillna(0)
    if not np.isfinite(splits).all() or (splits < 0).any():
        raise ValueError('Invalid Yahoo stock-split history')
    factors = splits.mask(splits == 0, 1)
    # Exclude the event on this row: its closing price is already post-split.
    future = factors.iloc[::-1].cumprod().iloc[::-1] / factors
    f['Split Factor'] = future
    f['As Traded Close'] = f.Close * future
    # Yahoo Close and Volume share the same split basis; dividends are excluded.
    f['Dollar Volume'] = f.Close * f.Volume
    adjustment = f['Adj Close'] / f.Close
    for col in ['Open', 'High', 'Low', 'Close']:
        f[col] = f[col] * adjustment
    f = f.dropna(subset=['Close'])
    return f


def download_prices(symbols,start,end,download,progress=None):
    CACHE.mkdir(parents=True,exist_ok=True)
    paths={};missing=[];failures={}
    for symbol in symbols:
        key=hashlib.sha256(f'{symbol}|{start.date()}|{end.date()}|historical-units-v2'.encode()).hexdigest()
        path=CACHE/f'{key}.csv.gz'
        if path.exists() and time.time()-path.stat().st_mtime<86400:
            try:
                probe=pd.read_csv(path,nrows=1)
                if probe.empty or 'Close' not in probe:raise ValueError('Invalid cached history')
                paths[symbol]=path
            except (OSError,ValueError,EOFError):missing.append((symbol,path))
        else:missing.append((symbol,path))
    empty_batches=0
    for offset in range(0,len(missing),48):
        batch=missing[offset:offset+48]
        try:
            frames=download([s for s,_ in batch],start,end)
        except Exception as exc:
            frames={}
            for symbol,_ in batch:failures[symbol]=f'Download failed: {type(exc).__name__}'
        empty_batches = empty_batches + 1 if not frames else 0
        if empty_batches >= 3:
            raise ValueError("Yahoo returned no histories for three consecutive batches, possibly due to rate limits. Please retry later; completed downloads are cached. No partial-universe backtest was run.")
        for symbol,path in batch:
            frame=frames.get(symbol)
            if frame is None or frame.empty:
                failures.setdefault(symbol,'No usable Yahoo history returned')
                continue
            frame=frame.copy()
            index=pd.to_datetime(frame.index)
            if index.tz is not None:index=index.tz_localize(None)
            frame.index=index.normalize()
            frame=frame[~frame.index.duplicated(keep='last')].sort_index().loc[:end]
            frame=frame.dropna(subset=['Close'])
            frame=frame[np.isfinite(frame.Close) & (frame.Close>0)]
            if frame.empty:
                failures[symbol]='No positive closing prices in requested period'
                continue
            temporary=path.with_name(path.name+'.'+uuid.uuid4().hex)
            frame.to_csv(temporary,compression='gzip')
            temporary.replace(path)
            paths[symbol]=path
        if progress:
            done=len(symbols)-len(missing)+min(offset+len(batch),len(missing))
            progress(.02+.16*done/max(len(symbols),1),f'Price histories · {done:,}/{len(symbols):,} checked · {len(paths):,} available')
    return PriceStore(paths),failures
