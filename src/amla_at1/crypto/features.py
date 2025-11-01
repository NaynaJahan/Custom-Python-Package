from __future__ import annotations
import pandas as pd


def add_lagged_features(df: pd.DataFrame, cols=("high", "low", "close", "volume"), lags=(1, 2, 3)):
    """
    For each column in cols, add lag features col_lag{k} = col.shift(k).
    Assumes df is sorted by index ascending (oldest -> newest).

    Returns a new DataFrame with added lag columns.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, col: str = "close", windows=(3, 7)):
    """
    Add simple rolling mean/std features on 'close' (or any numeric column).
    rolling_mean_{w}, rolling_std_{w}
    """
    out = df.copy()
    if col in out.columns:
        for w in windows:
            out[f"{col}_rollmean{w}"] = out[col].rolling(window=w, min_periods=1).mean()
            out[f"{col}_rollstd{w}"] = out[col].rolling(window=w, min_periods=1).std().fillna(0.0)
    return out
