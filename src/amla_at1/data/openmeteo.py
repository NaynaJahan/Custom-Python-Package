from __future__ import annotations
import datetime as dt
import io, gzip, json, time
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import requests

SYDNEY = {"latitude": -33.8678, "longitude": 151.2073, "timezone": "Australia/Sydney"}

DAILY_VARS = [
    "weather_code",
    "temperature_2m_max","temperature_2m_min",
    "apparent_temperature_max","apparent_temperature_min",
    "precipitation_sum","rain_sum","snowfall_sum",
    "precipitation_hours",
    "sunshine_duration","daylight_duration",
    "wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant",
    "shortwave_radiation_sum","et0_fao_evapotranspiration"
]

def fetch_daily_archive(
    start_date: str, end_date: str,
    latitude: float = SYDNEY["latitude"],
    longitude: float = SYDNEY["longitude"],
    timezone: str = SYDNEY["timezone"],
    daily_vars: List[str] = DAILY_VARS,
    session: requests.Session | None = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Download daily archive from Open-Meteo /v1/archive and return a tidy DataFrame indexed by date.
    """
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone
    }
    sess = session or requests.Session()
    last_err = None
    for i in range(retries):
        try:
            r = sess.get(base, params=params, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {r.text[:120]}")
            data = r.json()
            days = data["daily"]["time"]
            out = pd.DataFrame({"date": pd.to_datetime(days)})
            for v in daily_vars:
                out[v] = data["daily"].get(v, [np.nan]*len(days))
            out["date"] = out["date"].dt.date
            return out.set_index("date").sort_index()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i+1))
    raise RuntimeError(f"Failed to fetch archive: {last_err}")

def make_supervised_tables(
    daily_df: pd.DataFrame,
    *,
    rain_label_lag_days: int = 7,
    precip_window_days: int = 3,
    feature_lookback_days: int = 14,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build supervised datasets for:
      (A) classification: will_rain_on(D+rain_label_lag_days) using features up to D
      (B) regression: precip_sum over D+1..D+precip_window_days using features up to D
    Returns: (X_cls, y_cls, X_reg, y_reg)
    """
    df = daily_df.copy().sort_index()
    df["rain_next7"] = df["rain_sum"].shift(-rain_label_lag_days).fillna(0.0) > 0.0
    df["precip_3d_sum"] = df["precipitation_sum"].shift(-1).rolling(precip_window_days, min_periods=precip_window_days).sum()

    # rolling features ending at Day D (no future leakage)
    roll = df.rolling(window=feature_lookback_days, min_periods=3)
    feats = pd.DataFrame(index=df.index)
    for col in [
        "temperature_2m_max","temperature_2m_min",
        "apparent_temperature_max","apparent_temperature_min",
        "precipitation_sum","rain_sum",
        "wind_speed_10m_max","wind_gusts_10m_max",
        "shortwave_radiation_sum","et0_fao_evapotranspiration",
        "sunshine_duration"
    ]:
        if col in df:
            feats[f"{col}_mean_{feature_lookback_days}"] = roll[col].mean()
            feats[f"{col}_std_{feature_lookback_days}"]  = roll[col].std()
            feats[f"{col}_sum_{feature_lookback_days}"]  = roll[col].sum()

    # calendar features (seasonality)
    cal = pd.DataFrame(index=df.index)
    idx_dt = pd.to_datetime(feats.index)
    cal["month"] = idx_dt.month
    cal["dayofyear"] = idx_dt.dayofyear
    cal["is_summer"] = cal["month"].isin([12,1,2]).astype(int)  # Southern Hemisphere
    X = pd.concat([feats, cal], axis=1).dropna()

    y_cls = df.loc[X.index, "rain_next7"]
    y_reg = df.loc[X.index, "precip_3d_sum"]

    # Drop rows without targets (tail due to shifting)
    mask_cls = y_cls.notna()
    mask_reg = y_reg.notna()
    return X[mask_cls], y_cls[mask_cls].astype(int), X[mask_reg], y_reg[mask_reg]
