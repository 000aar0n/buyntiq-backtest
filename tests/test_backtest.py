from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import backtest as bt
import price_store
from ab_benchmark import _extract_history
from universe import parse_directory


def frames():
    index=pd.bdate_range('2019-01-01','2022-05-10')
    return {s:pd.DataFrame({'Open':100.,'High':100.,'Low':100.,'Close':100.,'Volume':1000000.,'As Traded Close':100.,'Split Factor':1.,'Dollar Volume':100000000.},index=index) for s in ['AAA','BBB','CCC','SPY','QQQ','FWD']}


@pytest.fixture
def market(monkeypatch,tmp_path):
    data=frames()
    monkeypatch.setattr(price_store,'CACHE',tmp_path)
    monkeypatch.setattr(bt,'_download',lambda symbols,start,end:{s:data[s].loc[start:end] for s in symbols if s in data})
    monkeypatch.setattr(bt,'load_ab_intl_tech',lambda start,end:pd.Series([100.,110.],index=pd.to_datetime(['2022-01-03','2022-04-29'])))
    def ranking(stock_data,asof,horizon,positive_only,**kwargs):
        assert asof in data['SPY'].index
        rows=[]
        for s in ['AAA','BBB','CCC']:
            rows.append({'ticker':s,'score':80.,'technical_score':80.,'forecast_return':.1,'forecast_kind':'test fixture', 'ml_evidence_weight':0.,'annualized_volatility':.2})
        return pd.DataFrame(rows)
    monkeypatch.setattr(bt,'rank_universe',ranking)
    return data


def test_month_anniversaries_and_weekly():
    days=bt._quarterly_schedule(pd.Timestamp('2022-01-31'),pd.Timestamp('2022-04-30'),1)
    assert [str(x.date()) for x in days]==['2022-01-31','2022-02-28','2022-03-31','2022-04-30']
    assert len(bt._quarterly_schedule(pd.Timestamp('2022-01-03'),pd.Timestamp('2022-01-31'),weeks=1))==5
    with pytest.raises(ValueError):bt._quarterly_schedule(pd.Timestamp('2022-01-01'),pd.Timestamp('2023-01-01'),0)


@pytest.mark.parametrize('months,weeks,expected',[(1,None,4),(3,None,2),(6,None,1),(12,None,1),(3,1,17)])
def test_frequency_first_trade_and_accounting(market,months,weeks,expected):
    cfg=bt.BacktestConfig(start='2022-01-03',end='2022-04-29',holdings=3,rebalance_months=months,rebalance_weeks=weeks)
    r=bt.run_backtest(['AAA','BBB','CCC'],cfg)
    h=r['holdings'];t=r['trades']
    assert h.rebalance_date.min()==pd.Timestamp('2022-01-03')
    assert h.iloc[0].signal_date==pd.Timestamp('2021-12-31')
    assert r['summary']['rebalances']==expected
    assert (h.signal_date<h.rebalance_date).all()
    assert h.groupby('rebalance_date').ticker.nunique().eq(3).all()
    assert np.allclose(h.groupby('rebalance_date').target_weight.sum(),1)
    assert np.isclose(r['summary']['end_value']+t.fee.sum(),cfg.starting_cash)
    # All stock prices are flat. Every round trip has precisely the stated fees.
    expected_value=cfg.starting_cash/(1.001)*((1-.001)/(1+.001))**(expected-1)
    assert np.isclose(r['summary']['end_value'],expected_value)
    assert 'AB International Technology' in r['comparisons']


def test_missing_open_never_uses_future_close(market):
    market['AAA'].loc['2022-04-04','Open']=np.nan
    cfg=bt.BacktestConfig(start='2022-01-03',end='2022-04-29',holdings=3)
    with pytest.raises(ValueError,match='opening execution'):bt.run_backtest(['AAA','BBB','CCC'],cfg)


def test_missing_held_bar_is_not_deleted(market):
    market['AAA'].drop(pd.Timestamp('2022-04-04'),inplace=True)
    cfg=bt.BacktestConfig(start='2022-01-03',end='2022-04-29',holdings=3)
    with pytest.raises(ValueError,match='No execution bar'):bt.run_backtest(['AAA','BBB','CCC'],cfg)


def test_unavailable_primary_keeps_portfolio(market,monkeypatch):
    def unavailable(*args):raise ValueError('Provider unavailable')
    monkeypatch.setattr(bt,'load_ab_intl_tech',unavailable)
    r=bt.run_backtest(['AAA','BBB','CCC'],bt.BacktestConfig(start='2022-01-03',end='2022-04-29',holdings=3,benchmark='INTTECHA'))
    assert r['summary']['end_value']>0
    assert np.isnan(r['summary']['benchmark_end'])
    assert 'AB International Technology' in r['benchmark_errors']
    assert 'SPY' in r['comparisons']


def test_benchmark_no_backfill_or_stale_tail():
    index=pd.bdate_range('2022-01-03','2022-01-14')
    equity=pd.DataFrame({'portfolio':100.},index=index)
    nav=pd.Series([10.,11.],index=pd.to_datetime(['2022-01-05','2022-01-11']))
    aligned=bt._anchor_comparison(nav,equity)
    assert aligned.loc[:'2022-01-04'].isna().all()
    assert aligned.loc['2022-01-12':].isna().all()
    assert aligned['2022-01-11']==pytest.approx(110.)
    stats=bt._comparison_stats('AB',aligned,equity)
    assert stats['start_date']==pd.Timestamp('2022-01-05')
    assert stats['end_date']==pd.Timestamp('2022-01-11')


def test_official_nav_parser():
    s=_extract_history({'navs':[{'date':'2022-01-03T00:00:00','price':'1,000.25'},{'date':'2022-01-04','price':'bad'},{'date':'bad','price':'12'},{'date':'2022-01-05','price':'-1'},{'date':'2022-01-06','price':'1100.25'}]})
    assert len(s)==2
    assert s.iloc[0]==1000.25
    assert s.attrs['currency']=='USD'
    assert s.attrs['isin']=='LU0060230025'
    with pytest.raises(ValueError):_extract_history({'data':[]})


def test_exchange_universe_filters_not_capped():
    header='ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n'
    rows=['BRK.B|Berkshire Class B|N|BRK.B|N|100|N|BRK.B','NA|Nano Labs ADR|Q|NA|N|100|N|NA','SPY|SPDR ETF|P|SPY|Y|100|N|SPY','TEST|Test Common Stock|N|TEST|N|100|Y|TEST','BAD|Some Series A Preferred Stock|N|BAD|N|100|N|BAD','WARR|Some Warrants|N|WARR|N|100|N|WARR','UNIT.U|Some Units|N|UNIT.U|N|100|N|UNIT.U','PFBC|Preferred Bank - Common Stock|Q|PFBC|N|100|N|PFBC','UNT|Unit Corporation Common Stock|N|UNT|N|100|N|UNT']
    # More than 100 valid listings must survive unchanged.
    names=['AA'+chr(65+i//26)+chr(65+i%26) for i in range(150)]
    rows += [f'{s}|Example Common Stock|N|{s}|N|100|N|{s}' for s in names]
    parsed=parse_directory(header+'\n'.join(rows)+'\nFile Creation Time: 20221007','Other US exchanges')
    assert len(parsed)==154
    assert {'NA','BRK-B'}.issubset(set(parsed.ticker))
    assert not {'SPY','TEST','BAD','WARR','UNIT-U'} & set(parsed.ticker)


def test_price_cache_batching_and_second_run(tmp_path,monkeypatch):
    monkeypatch.setattr(price_store,'CACHE',tmp_path)
    calls=[];symbols=[f'T{i}' for i in range(101)]
    def download(batch,start,end):
        calls.append(batch)
        return {s:pd.DataFrame({'Open':[1.,2.],'Close':[1.,2.],'Volume':[100,100]},index=pd.to_datetime(['2022-01-03','2022-01-04'])) for s in batch}
    args=(symbols,pd.Timestamp('2022-01-01'),pd.Timestamp('2022-01-05'),download)
    store,errors=price_store.download_prices(*args)
    assert [len(x) for x in calls]==[48,48,5]
    assert len(store)==101 and not errors
    assert store['T0'].Close.iloc[-1]==2
    price_store.download_prices(*args)
    assert len(calls)==3


def test_invalid_config_rejected_before_network(monkeypatch):
    monkeypatch.setattr(bt,'load_ab_intl_tech',lambda *args:pytest.fail('Network should not be called'))
    with pytest.raises(ValueError):bt.run_backtest(['AAA'],bt.BacktestConfig(start='2022-02-01',end='2022-01-01'))
    with pytest.raises(ValueError):bt.run_backtest(['AAA'],bt.BacktestConfig(rebalance_months=0))


def test_unknown_forecast_always_excluded(monkeypatch):
    data=frames()
    monkeypatch.setattr(bt,'forecast_return',lambda *args,**kw:{'available':False,'predicted_return':np.nan})
    ranked=bt.rank_universe({'AAA':data['AAA']},pd.Timestamp('2022-01-03'),21,False,1,1,'fast')
    assert ranked.empty


def test_streamlit_controls_and_saved_results(market,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import universe
    monkeypatch.setattr(universe,'load_us_universe',lambda:pd.DataFrame({'ticker':['AAA','BBB','CCC']}))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    assert not app.exception
    app.date_input[0].set_value(pd.Timestamp('2022-04-29').date())
    next(x for x in app.slider if x.label=='Holdings').set_value(3)
    next(x for x in app.number_input if x.label=='Years back').set_value(1)
    next(x for x in app.selectbox if x.label=='Rebuild portfolio').select('Every month')
    app.run()
    next(x for x in app.button if x.label=='Run backtest').click().run()
    assert not app.exception
    assert len(app.metric)==4
    assert 'backtest_result' in app.session_state
    before=app.session_state['backtest_config'].copy()
    next(x for x in app.number_input if x.label=='Years back').set_value(2).run()
    assert len(app.metric)==4
    assert app.session_state['backtest_config']==before
    assert before['rebalance_months']==1 and before['horizon']==21
    assert before['min_market_cap']==2e9 and before['min_price']==5 and before['min_dollar_volume']==1e7
    next(x for x in app.number_input if x.label=='Minimum estimated market cap ($ billions)').set_value(5).run()
    assert app.session_state['backtest_config']['min_market_cap']==2e9
    next(x for x in app.button if x.label=='Run backtest').click().run()
    assert not app.exception and app.session_state['backtest_config']['min_market_cap']==5e9
