# tests/crypto/test_etl.py

import pandas as pd
import numpy as np
from amla_at1.crypto.etl import build_nextday_high_supervised, latest_feature_row

def test_build_nextday_high_supervised_shapes():
    # fake ETH daily candles
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open":[1,2,3,4,5],
        "high":[2,3,4,5,6],
        "low":[0.5,1.5,2.5,3.5,4.5],
        "close":[1.5,2.5,3.5,4.5,5.5],
        "volume":[100,110,120,130,140],
        "marketCap":[1000,1100,1200,1300,1400],
    }, index=idx)

    X, y = build_nextday_high_supervised(df)

    # We had 5 rows in, but last row can't form y (needs t+1),
    # so expect length 4
    assert len(X) == 4
    assert len(y) == 4

    # y name is correct
    assert y.name == "next_day_high"

    # latest_feature_row() returns the most recent row of engineered features
    row = latest_feature_row(df)
    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
