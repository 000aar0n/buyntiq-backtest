import numpy as np
import pandas as pd
import pytest
import backtest as bt
from eligibility import (HistoricalSizeStore, EligibilityProviderError,
                         share_observation, price_liquidity_check)
from price_store import prepare_yahoo_history


def fact(end='2024-04-25', filed='2024-05-01', val=100_000_000, form='10-Q'):
    return {'end':end,'filed':filed,'val':val,'form':form}


def concept(*rows):
    return {'units':{'shares':list(rows)}}


def test_filing_lag_original_units_and_unknowns():
    data=concept(fact(),fact(filed='2024-06-01',val=400_000_000),
                 fact(end='2024-05-20',filed='2024-05-31',val=200_000_000))
    assert share_observation(data,pd.Timestamp('2024-05-01')) is None
    obs=share_observation(data,pd.Timestamp('2024-05-30'))
    assert obs['shares']==100_000_000
    assert obs['shares_report_date']==pd.Timestamp('2024-04-25')
    # A future restatement cannot overwrite the original report's share units.
    assert share_observation(concept(fact(),fact(filed='2024-06-01',val=400_000_000)),pd.Timestamp('2024-06-05'))['shares']==100_000_000
    assert share_observation(data,pd.Timestamp('2025-05-01')) is None
    assert share_observation(concept(fact(form='20-F')),pd.Timestamp('2024-05-05')) is None
    assert share_observation(concept(fact(),fact(val=200_000_000)),pd.Timestamp('2024-05-05')) is None


@pytest.mark.parametrize('split',[4.,.1])
def test_size_estimate_keeps_price_and_shares_in_same_units_and_caches(split):
    idx=pd.bdate_range('2024-04-01','2024-06-01')
    history=pd.DataFrame({'As Traded Close':40.,'Split Factor':split*10},index=idx)
    history.loc['2024-05-15':,'As Traded Close']=40/split
    history.loc['2024-05-15':,'Split Factor']=10.
    calls=[]
    def fetch(url):
        calls.append(url)
        if 'company_tickers' in url:return {'0':{'ticker':'AAA','cik_str':123}}
        return concept(fact())
    store=HistoricalSizeStore(fetch)
    for day in ['2024-05-14','2024-05-30']:
        result,reason=store.estimate('AAA',history.loc[:day],pd.Timestamp(day))
        assert reason is None
        assert result['estimated_market_cap']==pytest.approx(4e9)
        assert result['shares_filed_date']<pd.Timestamp(day)
    assert len(calls)==2


def test_ambiguous_share_classes_do_not_guess_cap():
    store=HistoricalSizeStore(lambda url:{'0':{'ticker':'AAA','cik_str':123},'1':{'ticker':'AAB','cik_str':123}})
    size,reason=store.estimate('AAA',pd.DataFrame(),pd.Timestamp('2024-05-30'))
    assert size is None and 'share-class' in reason


def test_yahoo_units_and_unchanged_total_return_prices():
    idx=pd.bdate_range('2024-05-13',periods=4)
    f=pd.DataFrame({'Open':[10,10,11,12],'High':[10,10,11,12],
                    'Low':[10,10,11,12],'Close':[10,10,11,12],
                    'Adj Close':[9,9,9.9,12],'Volume':[4e6]*4,
                    'Stock Splits':[0,0,4,0]},index=idx)
    prepared=prepare_yahoo_history(f)
    assert prepared['As Traded Close'].tolist()==[40,40,11,12]
    assert prepared.Close.tolist()==pytest.approx(f['Adj Close'].tolist())
    assert prepared['Dollar Volume'].tolist()==[40e6,40e6,44e6,48e6]
    changed=f.copy();changed.loc[idx[-1],['Open','High','Low','Close','Adj Close']]=999
    pd.testing.assert_frame_equal(prepare_yahoo_history(changed).iloc[:2],prepared.iloc[:2])
    with pytest.raises(ValueError):prepare_yahoo_history(f.drop(columns='Stock Splits'))


def history():
    return pd.DataFrame({'Open':100.,'Close':100.,'Volume':1e6,
        'As Traded Close':100.,'Split Factor':1.,'Dollar Volume':100e6},
        index=pd.bdate_range('2023-01-01',periods=330))


def test_price_and_liquidity_use_historical_trading_units():
    f=history();f['Close']=1.  # adjusted price is not the nominal-price test
    values,reason=price_liquidity_check(f,5,10e6)
    assert reason is None and values['historical_price']==100
    f['As Traded Close']=4.
    assert 'share price' in price_liquidity_check(f,5,10e6)[1]
    f['As Traded Close']=100.;f['Dollar Volume']=1e6
    assert 'liquidity' in price_liquidity_check(f,5,10e6)[1]
    f.loc[f.index[-1],'Dollar Volume']=np.nan
    assert 'Unknown' in price_liquidity_check(f,5,10e6)[1]


def test_all_enabled_gates_apply_before_ml(monkeypatch):
    data={s:history() for s in ['PENNY','SMALL','THIN','UNKNOWN','GOOD']}
    data['PENNY']['As Traded Close']=2.
    data['THIN']['Dollar Volume']=1000.
    checked=[];fitted=[]
    class Size:
        def estimate(self,symbol,hist,asof):
            checked.append(symbol)
            if symbol=='UNKNOWN':return None,'No recent, unambiguous historical shares'
            return {'estimated_market_cap':1e8 if symbol=='SMALL' else 3e9},None
    def forecast(frame,**kw):
        fitted.append(frame)
        return {'available':True,'predicted_return':.1,'kind':'fixture','evidence_weight':0}
    monkeypatch.setattr(bt,'forecast_return',forecast)
    from collections import Counter
    excluded=Counter()
    ranked=bt.rank_universe(data,data['GOOD'].index[-1],63,True,1,5,'fast',
        min_market_cap=2e9,min_price=5,min_dollar_volume=10e6,size_store=Size(),exclusions=excluded)
    assert ranked.ticker.tolist()==['GOOD']
    assert len(fitted)==1
    assert set(checked)=={'GOOD','SMALL','UNKNOWN'}
    assert sum(excluded.values())==4
    assert ranked.estimated_market_cap.iloc[0]>=2e9


def test_provider_failure_never_disables_filter(monkeypatch):
    class Failed:
        def estimate(self,*args):raise EligibilityProviderError('SEC unavailable')
    f=history()
    monkeypatch.setattr(bt,'forecast_return',lambda *a,**kw:pytest.fail('Must not run ML'))
    with pytest.raises(EligibilityProviderError):
        bt.rank_universe({'AAA':f},f.index[-1],63,True,1,1,'fast',min_market_cap=2e9,size_store=Failed())


def test_rebalance_rechecks_size_and_sells_to_cash_when_nobody_qualifies(tmp_path,monkeypatch):
    import price_store
    idx=pd.bdate_range('2022-01-01','2024-04-05')
    # Explicit constant-price fixture; no simulated data are used by the app.
    f=pd.DataFrame({'Open':100.,'High':100.,'Low':100.,'Close':100.,'Volume':1e6,
                    'As Traded Close':100.,'Split Factor':1.,'Dollar Volume':100e6},index=idx)
    monkeypatch.setattr(price_store,'CACHE',tmp_path)
    monkeypatch.setattr(bt,'_download',lambda syms,start,end:{s:f.loc[start:end].copy() for s in syms})
    monkeypatch.setattr(bt,'load_ab_intl_tech',lambda *a:pd.Series(dtype=float))
    monkeypatch.setattr(bt,'forecast_return',lambda *a,**kw:{'available':True,'predicted_return':.1,'evidence_weight':0,'kind':'fixture'})
    class Size:
        def estimate(self,symbol,hist,asof):
            return {'estimated_market_cap':3e9 if asof < pd.Timestamp('2024-02-01') else 1e8,
                    'shares_report_date':asof-pd.Timedelta(days=30),
                    'shares_filed_date':asof-pd.Timedelta(days=10),'size_source':'fixture'},None
    monkeypatch.setattr(bt,'HistoricalSizeStore',Size)
    result=bt.run_backtest(['AAA'],bt.BacktestConfig(start='2024-01-02',end='2024-04-05',holdings=1))
    assert result['trades'].side.tolist()==['BUY','SELL']
    assert result['trades'].date.iloc[-1]==pd.Timestamp('2024-04-02')
    assert result['holdings'].estimated_market_cap.min()>=2e9
    assert result['eligibility_exclusions']['count'].sum()==1
    assert 'cash' in result['warnings'][-1]
    assert result['summary']['end_value']+result['trades'].fee.sum()==pytest.approx(100000.)
