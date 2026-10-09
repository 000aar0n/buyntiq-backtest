import numpy as np
import pandas as pd
import pytest
import strategy as st
from universe import parse_directory


def test_named_funds_excluded_without_removing_reits_or_banks():
    names = {'DHY':'Credit Suisse High Yield Bond Fund Common Stock',
             'FTHY':'First Trust High Yield Opportunities 2027 Term Fund Common Stock',
             'CPT':'Camden Property Trust Common Stock',
             'REIT':'Example Real Estate Investment Trust Common Stock',
             'PFBC':'Preferred Bank Common Stock',
             'UNT':'Unit Corporation Common Stock'}
    text = 'Symbol|Security Name|Test Issue|ETF\n' + '\n'.join(f'{t}|{n}|N|N' for t,n in names.items())
    assert set(parse_directory(text,'fixture').ticker) == {'CPT','PFBC','UNT','REIT'}


@pytest.mark.parametrize('mode', ['fast','full'])
def test_holdout_is_purged_and_independent_of_weight_selection(monkeypatch, mode):
    # Synthetic fixtures exercise chronology, not investment performance.
    n, horizon = 1500, 63
    index = pd.bdate_range('2010-01-01',periods=n)
    c = 100 * np.exp(np.arange(n)*.0005 + .03*np.sin(np.arange(n)/37))
    prices = pd.DataFrame({'Close':c,'Volume':10000},index=index)
    calls = []
    def fit(x,y,scale,train,test_x,test_scale,mode):
        calls.append((np.array(train), x.index.get_indexer(test_x.index)))
        if len(test_x)==1:  # current forecast, outside matured x
            return {k:np.array([.2]) for k in st._models(mode)}
        assert train[-1] + horizon < x.index.get_indexer(test_x.index)[0]
        # Selection is perfect, but the independently evaluated holdout fails.
        selection_calls = 1 if mode=='fast' else 3
        pred = y.loc[test_x.index].to_numpy() if len(calls)<=selection_calls else np.ones(len(test_x))
        return {k:pred.copy() for k in st._models(mode)}
    monkeypatch.setattr(st,'_fit_member_predictions',fit)
    result = st.forecast_return(prices,horizon,mode)
    assert len(calls)==(2 if mode=='fast' else 4)
    assert result['kind']=='Baseline fallback'
    assert result['evidence_weight']==0
    assert np.isnan(result['raw_ml_return'])
    assert result['holdout_mae']>result['holdout_baseline_mae']
    features=st.feature_frame(prices)
    target=np.log(prices.Close.shift(-horizon)/prices.Close)
    joined=features.assign(_target=target,_scale=features.volatility_63.clip(lower=.003)*np.sqrt(horizon)).dropna()
    train,test=calls[-1]
    baseline=joined._target.iloc[train].tail(252).median()
    expected_mae=np.abs(np.expm1(joined._target.iloc[test])-np.expm1(baseline)).mean()
    assert result['holdout_baseline_mae']==pytest.approx(expected_mae)
    expected=np.expm1(np.log(prices.Close.shift(-horizon)/prices.Close).dropna().tail(252).median())
    assert result['predicted_return']==pytest.approx(expected)


def test_accepted_ensemble_still_refits(monkeypatch):
    n=1500
    prices=pd.DataFrame({'Close':100*np.exp(np.arange(n)*.0005+.1*np.sin(np.arange(n)/37)),
                         'Volume':10000},index=pd.bdate_range('2010-01-01',periods=n))
    calls=[]
    def fit(x,y,scale,train,test_x,test_scale,mode):
        calls.append(len(test_x))
        pred=np.array([.12]) if len(test_x)==1 else y.loc[test_x.index].to_numpy()
        return {k:pred.copy() for k in st._models(mode)}
    monkeypatch.setattr(st,'_fit_member_predictions',fit)
    result=st.forecast_return(prices,63,'fast')
    assert calls[-1]==1 and len(calls)==3
    assert result['predicted_return']==pytest.approx(np.expm1(.12))
    assert result['evidence_weight']>0
