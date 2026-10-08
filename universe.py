"""Current US exchange-listed common shares and ADRs; never a capped sample."""
from io import StringIO
import re
from urllib.request import Request, urlopen

import pandas as pd

DIRECTORIES = {
    'Nasdaq': 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
    'Other US exchanges': 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',
}
EXCLUDED = re.compile(
    r"\b(?:preferred (?:stock|shares|securities)|preference shares|ETF|ETN)\b"
    r"|\b(?:warrants?|rights?|units?|debentures?|notes|bonds)\b(?!.*\b(?:common stock|ordinary shares)\b)"
    r"|\bunits?,?\s+(?:each|consisting)\b"
    r"|depositary shares.*(?:series|interest)"
    r"|\b(?:fund|closed[ -]end|portfolio|income trust|(?<!real estate )investment trust)\b", re.I
)


def parse_directory(text, source):
    lines = [line for line in text.splitlines() if line and not line.startswith('File Creation Time')]
    frame = pd.read_csv(StringIO('\n'.join(lines)), sep='|', dtype=str, keep_default_na=False)
    symbol_col = 'Symbol' if 'Symbol' in frame else 'ACT Symbol'
    required = {symbol_col, 'Security Name', 'Test Issue', 'ETF'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{source} returned an invalid listing directory')
    frame = frame[(frame['Test Issue']=='N') & (frame['ETF']=='N')].copy()
    frame = frame[~frame['Security Name'].str.contains(EXCLUDED)]
    # Dots denote share classes (BRK.B -> BRK-B); punctuation used for
    # preferred shares and units is deliberately rejected before conversion.
    frame = frame[frame[symbol_col].str.fullmatch(r'[A-Z]{1,6}(?:\.[A-Z])?')]
    frame = frame[~frame[symbol_col].str.contains(r'\.[UWR]$', regex=True)]
    return pd.DataFrame({'ticker':frame[symbol_col].str.replace('.', '-', regex=False),
                         'company':frame['Security Name'], 'source':source})


def load_us_universe():
    frames=[]
    for name,url in DIRECTORIES.items():
        request=Request(url,headers={'User-Agent':'BuyntiqBacktest/1.0'})
        with urlopen(request,timeout=30) as response:
            frames.append(parse_directory(response.read().decode('utf-8'),name))
    result=pd.concat(frames,ignore_index=True).drop_duplicates('ticker').sort_values('ticker').reset_index(drop=True)
    if len(result)<1000:
        raise ValueError('US listing directories returned an unexpectedly small universe; refusing a partial list.')
    return result
