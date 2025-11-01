"""
Crypto utilities for Advanced Machine Learning Application (AT3 / Lab4).

This submodule focuses on Ethereum (ETH) pricing:
- Pull recent OHLC data from public APIs (Kraken, CoinGecko)
- Engineer features for next-day HIGH prediction
- Build supervised tables for training regression models
"""

from .ohlc_api import fetch_kraken_ohlc, fetch_coingecko_ohlc
from .etl import (
    load_local_history_csvs,
    build_nextday_high_supervised,
    latest_feature_row,
)
from .features import add_lagged_features, add_rolling_features

__all__ = [
    "fetch_kraken_ohlc",
    "fetch_coingecko_ohlc",
    "load_local_history_csvs",
    "build_nextday_high_supervised",
    "latest_feature_row",
    "add_lagged_features",
    "add_rolling_features",
]
