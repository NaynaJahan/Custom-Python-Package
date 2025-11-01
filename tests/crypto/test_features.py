# tests/crypto/test_features.py

import pandas as pd
from amla_at1.crypto.features import add_lagged_features, add_rolling_features

def test_add_lagged_features_and_roll():
    df = pd.DataFrame({
        "high":[10,11,12,13],
        "low":[8,9,10,11],
        "close":[9,10,11,12],
        "volume":[100,120,110,130],
    }, index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))

    out = add_lagged_features(df, lags=(1,2))
    assert "high_lag1" in out.columns
    assert "close_lag2" in out.columns

    out2 = add_rolling_features(out, col="close", windows=(2,3))
    assert "close_rollmean2" in out2.columns
    assert "close_rollstd3" in out2.columns
