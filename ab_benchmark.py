"""Official daily USD NAV for AB International Technology Portfolio A USD."""
from datetime import date
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import uuid

import numpy as np
import pandas as pd
from price_store import CACHE

FUND_NAME = 'AB International Technology'
ISIN = 'LU0060230025'
OFFICIAL_URL = ('https://webapi.alliancebernstein.com/v2/funds/americas/en/investor/'
                + ISIN + '/historical-navs?freq=daily')
SOURCE = 'AllianceBernstein official A USD NAV (LU0060230025 / INTTECHA)'


def _extract_history(payload):
    rows=payload.get('navs',[])
    if not isinstance(rows,list) or not rows:
        raise ValueError('AB returned no NAV observations')
    frame=pd.DataFrame(rows)
    if not {'date','price'}.issubset(frame):raise ValueError('Unrecognized AB NAV response')
    dates=pd.to_datetime(frame.date,errors='coerce',utc=True,format='mixed').dt.tz_localize(None).dt.normalize()
    values=pd.to_numeric(frame.price.astype(str).str.replace(',','',regex=False),errors='coerce')
    series=pd.Series(values.to_numpy(),index=pd.DatetimeIndex(dates),name=FUND_NAME)
    series=series[series.index.notna() & np.isfinite(series) & (series>0)]
    series=series[~series.index.duplicated(keep='last')].sort_index()
    if series.empty:raise ValueError('AB returned no valid NAV values')
    series.attrs.update(source=SOURCE,currency='USD',isin=ISIN,source_url=OFFICIAL_URL)
    return series


def load_ab_intl_tech(start,end=None):
    start=pd.Timestamp(start).normalize();end=pd.Timestamp(end or date.today()).normalize()
    path=CACHE/'ab-LU0060230025-nav.json'
    payload=None;cached=None
    if path.exists():
        try:
            cached=json.loads(path.read_text())
            _extract_history(cached)
            if time.time()-path.stat().st_mtime<21600:payload=cached
        except (ValueError,OSError,KeyError):cached=None
    warning=None
    if payload is None:
        try:
            request=Request(OFFICIAL_URL,headers={'User-Agent':'BuyntiqBacktest/1.0','Accept':'application/json'})
            with urlopen(request,timeout=30) as response:payload=json.load(response)
            _extract_history(payload)
            CACHE.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+'.'+uuid.uuid4().hex)
            temp.write_text(json.dumps(payload));temp.replace(path)
        except Exception as exc:
            if cached is None:
                raise ValueError(f'AB official NAV could not be loaded ({type(exc).__name__}). Please retry.') from exc
            payload=cached
            warning='AB provider unavailable; showing the saved NAV history through its last reported date.'
    series=_extract_history(payload)
    series=series.loc[(series.index>=start)&(series.index<=end)]
    if len(series)<2:raise ValueError('AB A USD NAV has fewer than two observations in this date range.')
    if warning:series.attrs['warning']=warning
    return series
