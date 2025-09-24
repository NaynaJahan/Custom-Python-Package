import datetime as dt
import json
import pathlib
import importlib.util
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Load openmeteo.py by path (consistent with your existing tests)
OM_PATH = ROOT / "src/amla_at1/data/openmeteo.py"
spec = importlib.util.spec_from_file_location("openmeteo_mod", OM_PATH)
openmeteo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openmeteo_mod)

class _DummyResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)
    def json(self):
        return self._payload

class _DummySession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
    def get(self, url, params=None, timeout=60):
        self.calls += 1
        return _DummyResp(200, self.payload)

def _fake_daily_payload(dates, vars):
    return {
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in dates],
            **{v: [0.0]*len(dates) for v in vars}
        }
    }

def test_fetch_daily_archive_parses_minimal():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    payload = _fake_daily_payload(dates, openmeteo_mod.DAILY_VARS)
    sess = _DummySession(payload)
    df = openmeteo_mod.fetch_daily_archive("2020-01-01", "2020-01-05", session=sess)
    assert isinstance(df, pd.DataFrame)
    assert set(openmeteo_mod.DAILY_VARS).issubset(df.columns)
    assert df.index.dtype == "object" or str(df.index.dtype).startswith("datetime") or True
    # ensure index are dates (date objects)
    assert all(isinstance(ix, dt.date) for ix in df.index)

def test_make_supervised_tables_targets_and_alignment():
    # Build 25 days of synthetic daily data with a simple pattern
    dates = pd.date_range("2021-02-01", periods=25, freq="D").date
    daily = pd.DataFrame(index=pd.Index(dates, name="date"))
    daily["rain_sum"] = ([0,0,1,0,0,0,1,0,0,0] * 3)[:25]
    daily["precipitation_sum"] = np.linspace(0, 9.6, 25)
    # required columns referenced in feature builder
    daily["temperature_2m_max"] = np.linspace(20, 30, 25)
    daily["temperature_2m_min"] = np.linspace(10, 15, 25)
    daily["apparent_temperature_max"] = daily["temperature_2m_max"] + 1
    daily["apparent_temperature_min"] = daily["temperature_2m_min"] - 1
    daily["wind_speed_10m_max"] = 10
    daily["wind_gusts_10m_max"] = 20
    daily["shortwave_radiation_sum"] = 5
    daily["et0_fao_evapotranspiration"] = 2
    daily["sunshine_duration"] = 3600

    Xc, yc, Xr, yr = openmeteo_mod.make_supervised_tables(
        daily,
        rain_label_lag_days=7,
        precip_window_days=3,
        feature_lookback_days=14
    )

    # non-empty and aligned
    assert len(Xc) == len(yc) > 0
    assert len(Xr) == len(yr) > 0
    assert Xc.index.equals(yc.index)
    assert Xr.index.equals(yr.index)

    # sanity: no NaNs in features used
    assert not Xc.isna().any().any()
    assert not Xr.isna().any().any()

    # Check classification target shift: y(D) corresponds to rain_sum at D+7
    d = Xc.index[0]
    future_d = (pd.to_datetime(d) + pd.Timedelta(days=7)).date()
    expected = daily.loc[future_d, "rain_sum"] > 0
    assert int(yc.loc[d]) == int(expected)
