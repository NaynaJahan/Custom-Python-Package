from __future__ import annotations
from typing import Optional
import time
import requests
import pandas as pd


def fetch_kraken_ohlc(
    pair: str = "ETHUSD",
    interval: int = 1440,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLC candles from Kraken public API.
    Docs: https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/

    Parameters
    ----------
    pair : str
        Trading pair (e.g. "ETHUSD").
    interval : int
        Candle size in minutes (1440 = 1 day).
    since : int or None
        Unix timestamp (seconds). If provided, Kraken returns candles since that time.

    Returns
    -------
    df : pd.DataFrame
        index = UTC datetime (ns precision)
        columns = ['open','high','low','close','vwap','volume','count']
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if "error" in payload and payload["error"]:
        raise RuntimeError(f"Kraken API error: {payload['error']}")

    result = payload["result"]
    # Kraken nests data under the pair symbol key, e.g. result["ETHUSD"]
    # plus result["last"] which we don't need for now.
    # Each row: [time, open, high, low, close, vwap, volume, count]
    rows = None
    for k, v in result.items():
        if k == "last":
            continue
        rows = v
        break

    if rows is None:
        raise RuntimeError("No OHLC rows found in Kraken response")

    cols = ["ts", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(rows, columns=cols)

    # Timestamp comes in seconds -> convert to UTC datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    df = df.set_index("ts").sort_index()

    # Convert numeric cols from string to float/int
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype(int)

    return df


def fetch_coingecko_ohlc(
    coin_id: str = "ethereum",
    vs_currency: str = "usd",
    days: str = "30",
) -> pd.DataFrame:
    """
    Fetch OHLC candles from CoinGecko demo OHLC endpoint.
    Docs: /coins/{id}/ohlc
    We assume public/demo usage (no paid key).

    Parameters
    ----------
    coin_id : str
        CoinGecko coin ID (e.g. "ethereum").
    vs_currency : str
        Quote currency (e.g. "usd").
    days : str
        How many days of history. Must be one of ['1','7','14','30','90','180','365'].

    Returns
    -------
    df : pd.DataFrame
        index = UTC datetime from ms timestamp
        columns = ['open','high','low','close']
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        # if you have/need demo key later you can pass headers or ?x_cg_demo_api_key=...
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Response is a list like:
    # [
    #   [timestamp_ms, open, high, low, close],
    #   ...
    # ]

    cols = ["ts_ms", "open", "high", "low", "close"]
    df = pd.DataFrame(data, columns=cols)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"]).set_index("ts").sort_index()

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)

    return df
