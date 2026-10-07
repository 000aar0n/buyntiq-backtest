from __future__ import annotations

import json
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


MORNINGSTAR_SECID = "F0GBR04I8U"
FUND_NAME = "AB International Technology"
PORTAL_TOKEN = "t92wz0sj7c"

BASE_URLS = [
    "https://tools.morningstar.co.uk/api/rest.svc/timeseries_price/" + PORTAL_TOKEN,
    "https://tools.morningstar.es/api/rest.svc/timeseries_price/" + PORTAL_TOKEN,
]


def _extract_history(payload: dict) -> pd.Series:
    timeseries = payload.get("TimeSeries") or {}
    securities = timeseries.get("Security") or []
    if isinstance(securities, dict):
        securities = [securities]

    rows = []
    for security in securities:
        history = security.get("HistoryDetail") or []
        if isinstance(history, dict):
            history = [history]
        for item in history:
            value = item.get("Value")
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, dict):
                value = value.get("value", value.get("Value"))
            try:
                px = float(str(value).replace(",", ""))
                dt = pd.Timestamp(item.get("EndDate")).normalize()
            except Exception:
                continue
            if px > 0:
                rows.append((dt, px))

    if not rows:
        return pd.Series(dtype=float, name=FUND_NAME)

    frame = (
        pd.DataFrame(rows, columns=["date", "close"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    return pd.Series(
        frame["close"].to_numpy(),
        index=pd.DatetimeIndex(frame["date"]),
        name=FUND_NAME,
    )


def load_ab_intl_tech(start, end=None) -> pd.Series:
    """Load daily USD NAV for AB International Technology A USD.

    ISIN: LU0060230025
    Morningstar security ID: F0GBR04I8U
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end or date.today()).normalize()

    params = {
        "currencyId": "USD",
        "idtype": "Morningstar",
        "frequency": "daily",
        "outputType": "JSON",
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "id": MORNINGSTAR_SECID + "]2]0]FOGBR$$ALL",
        "applyTrackRecordExtension": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
    }

    for base in BASE_URLS:
        try:
            request = Request(base + "?" + urlencode(params), headers=headers)
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            series = _extract_history(payload)
            if not series.empty:
                return series.loc[
                    (series.index >= start) & (series.index <= end)
                ]
        except Exception:
            continue

    return pd.Series(dtype=float, name=FUND_NAME)
