"""Historical eligibility gates. Unknown size never passes an enabled cap floor.

Market cap is an estimate from the latest original SEC cover-page share count
public before the signal day (one-day publication lag), adjusted for intervening
splits. Only domestic 10-K/10-Q entity-wide, single-listed-class counts are used.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

import numpy as np
import pandas as pd
if __package__:
    from . import price_store
else:
    import price_store

_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


class EligibilityProviderError(ValueError):
    pass


def _cached_json(url):
    global _LAST_REQUEST
    cache = price_store.CACHE / 'sec-size-v1'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (hashlib.sha256(url.encode()).hexdigest() + '.json')
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            pass
    headers = {'User-Agent': os.environ.get('SEC_USER_AGENT',
               'BuyntiqBacktest/1.0 (https://github.com/000aar0n/buyntiq-backtest)')}
    try:
        # A shared process lock limits all sessions to at most four starts/sec.
        with _LOCK:
            pause = .25 - (time.monotonic() - _LAST_REQUEST)
            if pause > 0:
                time.sleep(pause)
            _LAST_REQUEST = time.monotonic()
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                result = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        endpoint = 'ticker directory' if 'company_tickers' in url else 'historical share-count API'
        raise EligibilityProviderError(f'SEC {endpoint} unavailable (HTTP {exc.code}). The run stopped; no current-market-cap substitute was used.') from exc
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        raise EligibilityProviderError('SEC size data could not be verified. Retry later; completed requests are cached.') from exc
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return result


def share_observation(concept, as_of, max_age_days=180):
    """Return a mature original observation; never use future/restated counts."""
    rows = (concept or {}).get('units', {}).get('shares', [])
    if not rows:
        return None
    f = pd.DataFrame(rows)
    if not {'end', 'filed', 'val', 'form'}.issubset(f):
        return None
    f['end'] = pd.to_datetime(f.end, errors='coerce')
    f['filed'] = pd.to_datetime(f.filed, errors='coerce')
    f['val'] = pd.to_numeric(f.val, errors='coerce')
    f = f[f.form.isin(['10-K', '10-Q', '10-K/A', '10-Q/A']) &
          (f.end <= as_of) & (f.filed < as_of) & (f.end <= f.filed) &
          (f.end >= as_of - pd.Timedelta(days=max_age_days)) &
          np.isfinite(f.val) & (f.val > 0)]
    if f.empty:
        return None
    f = f[f.end == f.end.max()]
    # First-published count preserves the share units at that report date.
    f = f[f.filed == f.filed.min()]
    if f.val.nunique() != 1:
        return None  # Ambiguous share-class facts: do not sum or guess.
    row = f.iloc[0]
    return {'shares': float(row.val), 'shares_report_date': row.end,
            'shares_filed_date': row.filed}


class HistoricalSizeStore:
    def __init__(self, fetch=None):
        self.fetch = fetch or _cached_json
        self._mapping = None
        self._classes = defaultdict(set)
        self._concepts = {}
        self.mapping_note = None

    def _load_mapping(self):
        if self._mapping is None:
            try:
                raw = self.fetch('https://www.sec.gov/files/company_tickers.json')
            except EligibilityProviderError:
                # This is an identifier directory, never a substitute for
                # historical fundamentals. Every share count is still dated.
                snapshot = Path(__file__).resolve().parent / 'data' / 'sec_tickers.json'
                if not snapshot.exists():
                    raise
                saved = json.loads(snapshot.read_text())
                retrieved = pd.Timestamp(saved['retrieved_at'])
                if pd.Timestamp.now(tz='UTC') - retrieved > pd.Timedelta(days=30):
                    raise EligibilityProviderError('SEC ticker directory is blocked and the bundled directory is over 30 days old. Refresh the directory snapshot; the size filter remains enabled.')
                raw = {s:{'ticker':s,'cik_str':cik} for s,cik in saved['tickers'].items()}
                self.mapping_note = f"SEC live ticker directory unavailable; using official directory snapshot from {retrieved.date()}. Historical share counts still require dated filings."
            if not isinstance(raw, dict) or not raw:
                raise EligibilityProviderError('SEC ticker mapping is unavailable; cannot apply the market-cap floor.')
            self._mapping = {}
            for row in raw.values():
                if not isinstance(row, dict) or not {'ticker', 'cik_str'}.issubset(row):
                    continue
                ticker = str(row['ticker']).upper().replace('.', '-')
                cik = int(row['cik_str'])
                self._mapping[ticker] = cik
                self._classes[cik].add(ticker)
            if not self._mapping:
                raise EligibilityProviderError('SEC ticker mapping is invalid; cannot apply the market-cap floor.')

    def preflight(self):
        """Check required data before minutes of price downloads and model work."""
        self._load_mapping()
        cik = self._mapping.get('AAPL')
        if cik is None:
            raise EligibilityProviderError('SEC directory failed its identifier check.')
        self._concepts[cik] = self.fetch(
            f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json')
        if not self._concepts[cik]:
            raise EligibilityProviderError('SEC historical share-count API returned no data during its availability check.')

    def estimate(self, symbol, history, as_of):
        self._load_mapping()
        cik = self._mapping.get(symbol)
        if cik is None:
            return None, 'No SEC ticker mapping'
        if len(self._classes[cik]) != 1:
            return None, 'Ambiguous issuer/share-class mapping'
        if cik not in self._concepts:
            self._concepts[cik] = self.fetch(
                f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json')
        obs = share_observation(self._concepts[cik], as_of)
        if obs is None:
            return None, 'No recent, unambiguous historical shares'
        if not {'As Traded Close', 'Split Factor'}.issubset(history):
            return None, 'Missing historical price/share units'
        prior = history.loc[:obs['shares_report_date'], 'Split Factor'].dropna()
        if prior.empty or (obs['shares_report_date'] - prior.index[-1]).days > 7:
            return None, 'Missing price history at shares report date'
        report_factor = float(prior.iloc[-1])
        signal_factor = float(history.loc[as_of, 'Split Factor'])
        price = float(history.loc[as_of, 'As Traded Close'])
        if not all(np.isfinite(v) and v > 0 for v in [price, report_factor, signal_factor]):
            return None, 'Invalid historical price/share units'
        adjusted_shares = obs['shares'] * report_factor / signal_factor
        cap = adjusted_shares * price
        if not np.isfinite(cap) or cap <= 0:
            return None, 'Invalid market-cap estimate'
        return {**obs, 'estimated_market_cap': cap,
                'size_source': 'SEC original cover-page shares × split-normalized historical price'}, None


def price_liquidity_check(history, min_price, min_dollar_volume):
    result = {'historical_price': np.nan, 'median_daily_dollar_volume': np.nan}
    if 'As Traded Close' in history:
        result['historical_price'] = float(history['As Traded Close'].iloc[-1])
    if min_price > 0:
        if not np.isfinite(result['historical_price']) or result['historical_price'] <= 0:
            return result, 'Unknown historical share price'
        if result['historical_price'] < min_price:
            return result, 'Below minimum share price'
    if 'Dollar Volume' in history:
        dv = pd.to_numeric(history['Dollar Volume'].tail(60), errors='coerce')
        if len(dv) == 60 and np.isfinite(dv).all() and (dv >= 0).all():
            result['median_daily_dollar_volume'] = float(dv.median())
    if min_dollar_volume > 0:
        if not np.isfinite(result['median_daily_dollar_volume']):
            return result, 'Unknown 60-session trading liquidity'
        if result['median_daily_dollar_volume'] < min_dollar_volume:
            return result, 'Below minimum trading liquidity'
    return result, None
