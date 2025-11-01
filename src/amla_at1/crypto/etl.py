from __future__ import annotations
import glob
import pandas as pd
from typing import List, Tuple
from .features import add_lagged_features, add_rolling_features


def load_local_history_csvs(pattern: str = "data/*.csv") -> pd.DataFrame:
    """
    Load and concatenate ETH history CSVs you were given (2015...2025).
    Expects columns like: timeOpen, high, low, open, close, volume, etc.

    Parameters
    ----------
    pattern : str
        Glob pattern to find all the CSVs, e.g. "data/Ethereum_*.csv"

    Returns
    -------
    hist : pd.DataFrame
        index = date (UTC-normalized day)
        columns at least ['open','high','low','close','volume','marketCap']
    """
    frames: List[pd.DataFrame] = []
    for p in glob.glob(pattern):
        df = pd.read_csv(p)
        # We assume there's at least a timeOpen or timestamp column we can parse.
        # We'll prefer 'timeOpen' as the candle start.
        if "timeOpen" in df.columns:
            idx = pd.to_datetime(df["timeOpen"], utc=True)
        elif "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"], utc=True)
        else:
            raise KeyError(f"No timeOpen/timestamp column in {p}")

        df = df.assign(ts=idx).set_index("ts").sort_index()

        # Keep only the columns we care about
        keep_cols = [
            "open", "high", "low", "close", "volume", "marketCap"
        ]
        present = [c for c in keep_cols if c in df.columns]
        df = df[present]

        frames.append(df)

    if not frames:
        raise RuntimeError("No CSVs matched pattern for ETH history")

    hist = pd.concat(frames).sort_index()
    # Drop duplicate timestamps if any overlap between files
    hist = hist[~hist.index.duplicated(keep="last")]

    return hist


def _make_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper:
    - assumes index is datetime (daily candles)
    - adds lag + rolling stats
    """
    # Always sort ascending
    df = raw_df.sort_index().copy()

    df_feat = add_lagged_features(df, cols=("high", "low", "close", "volume"), lags=(1, 2, 3))
    df_feat = add_rolling_features(df_feat, col="close", windows=(3, 7))

    return df_feat


def build_nextday_high_supervised(
    hist_df: pd.DataFrame,
    target_col: str = "high",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Takes ETH daily history -> returns (X, y) for regression.
    y[t] = next day's HIGH price (i.e. shift(-1) of 'high').

    Output index is the SAME date as the features (today),
    but note that the last day won't have a y because next day is unknown.
    """
    df_feat = _make_feature_table(hist_df)

    # our target is tomorrow's high
    if target_col not in hist_df.columns:
        raise KeyError(f"{target_col} not in hist_df columns")

    y = hist_df[target_col].shift(-1).rename("next_day_high")

    # align X and y (drop rows where y is NaN -> last day)
    mask = ~y.isna()
    X = df_feat.loc[mask].copy()
    y = y.loc[mask].copy()

    return X, y


def latest_feature_row(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the single most recent row of model features to be scored by FastAPI.

    Steps:
    - create engineered features on full history
    - take the LAST timestamp row (that's "today's" features)
    - return it as a 1-row DataFrame (still with column names the model expects)

    FastAPI will do:
        row = latest_feature_row(...)
        pred = model.predict(row)[0]
    """
    df_feat = _make_feature_table(hist_df)
    last_row = df_feat.tail(1).copy()
    return last_row
