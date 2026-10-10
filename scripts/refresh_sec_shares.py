"""Generate a reproducible SEC dated-share-count cache for Streamlit deployment.

Uses official SEC DEI companyconcept filings and the app's uncapped US
listing universe. The snapshot contains dated observations, NOT current caps.
No market-cap estimates are invented for unavailable/ambiguous issuers.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from universe import load_us_universe
DATA = ROOT / "data"
BUNDLE = DATA / "sec_shares.json.gz"
DIRECTORY = DATA / "sec_tickers.json"
API = "https://data.sec.gov/api/xbrl/companyconcept/CIK{:010d}/dei/EntityCommonStockSharesOutstanding.json"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
_lock = threading.Lock()
_next_allowed = 0.0
AGENT = os.environ.get("SEC_USER_AGENT", "BuyntiqBacktest research (https://github.com/000aar0n/buyntiq-backtest)")
MAX_REQUESTS_PER_SECOND = 7


def get_json(url):
    global _next_allowed
    last_error = None
    for attempt in range(5):
        try:
            with _lock:
                delay = _next_allowed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                _next_allowed = time.monotonic() + 1.0 / MAX_REQUESTS_PER_SECOND
            request = Request(url, headers={
                "User-Agent": AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            })
            with urlopen(request, timeout=25) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
            if error.code not in (403, 429, 500, 502, 503, 504):
                break
        except (URLError, OSError, TimeoutError, ValueError) as error:
            last_error = error
        time.sleep(min(30.0, 2 ** attempt + random.random()))
    raise RuntimeError(str(last_error))


def load_prior():
    if not BUNDLE.exists():
        return {}
    with gzip.open(BUNDLE, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("concepts", {})


def fetch_concept(cik):
    payload = get_json(API.format(cik))
    if payload is None:
        return None
    rows = payload.get("units", {}).get("shares", [])
    # Keep complete original-filing dates and values. Never replace historic
    # filed facts with a current company-size figure.
    records = []
    for row in rows:
        if row.get("form") not in FORMS or not row.get("end") or not row.get("filed"):
            continue
        if row["end"] < "2008-01-01":
            continue
        try:
            val = float(row["val"])
        except (ValueError, TypeError, KeyError):
            continue
        if not (0 < val < float("inf")):
            continue
        records.append({
            "end": row["end"], "filed": row["filed"],
            "val": val, "form": row["form"]
        })
    return {"units": {"shares": records}} if records else None


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # Every ticker in the uncapped Nasdaq and other-exchange common-share
    # directories is considered. Maps to official CIK identifiers.
    listings = load_us_universe()
    tickers = sorted(set(str(s).upper().replace(".", "-") for s in listings.ticker))
    if len(tickers) < 3000:
        raise RuntimeError("Unexpectedly small US listing universe; refusing a partial cache.")

    saved = json.loads(DIRECTORY.read_text(encoding="utf-8"))
    mapping = saved["tickers"]
    ciks = sorted(set(int(mapping[s]) for s in tickers if s in mapping))
    if len(ciks) < 2000:
        raise RuntimeError(f"Only {len(ciks)} CIKs mapped; refusing a partial cache.")

    old = load_prior()
    concepts = dict(old)
    failures = []
    started = time.monotonic()
    print(f"Refreshing {len(ciks)} SEC issuers for {len(tickers)} US listings", flush=True)

    def refresh(cik):
        try:
            return cik, fetch_concept(cik), None
        except Exception as exc:
            return cik, None, str(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(refresh, cik): cik for cik in ciks}
        for completed, future in enumerate(as_completed(jobs), 1):
            cik, concept, err = future.result()
            if concept:
                concepts[str(cik)] = concept
            elif err:
                failures.append((cik, err))
            # A provider's 404 or missing concept means "unknown", not zero cap.
            if completed % 250 == 0 or completed == len(ciks):
                print(f"SEC issuers checked: {completed}/{len(ciks)}; records: {len(concepts)}; failures: {len(failures)}; elapsed: {time.monotonic()-started:.0f}s", flush=True)

    # Prevent a heavily blocked runner from replacing a usable cache
    # with fabricated or tiny coverage.
    covered = sum(str(cik) in concepts for cik in ciks)
    required = max(1000, int(len(ciks) * .40))
    if covered < required or str(mapping.get("AAPL", "")) not in concepts:
        raise RuntimeError(f"Official SEC cache incomplete: {covered}/{len(ciks)} covered, require {required} plus AAPL; no bundle published.")
    if not old and failures and len(failures) >= len(ciks) // 2:
        raise RuntimeError("SEC blocked most companyconcept requests. Not publishing an unverified partial bundle.")

    result = {
        "source": "SEC EDGAR DEI EntityCommonStockSharesOutstanding companyconcept dated filings",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "target_listing_count": len(tickers),
        "target_cik_count": len(ciks),
        "covered_target_ciks": covered,
        "failed_requests": len(failures),
        "concepts": concepts,
    }
    path = BUNDLE.with_suffix(".json.gz.tmp")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(result, f, separators=(",", ":"), allow_nan=False)
    os.replace(path, BUNDLE)
    print(f"Published {BUNDLE} ({BUNDLE.stat().st_size:,} bytes), {covered}/{len(ciks)} CIKs", flush=True)


if __name__ == "__main__":
    main()
