import pathlib, importlib.util
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "src/amla_at1/features/weather.py"
spec = importlib.util.spec_from_file_location("w_mod", PATH)
w_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w_mod)

def test_clip_and_fill_interpolates_and_cleans():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [np.inf, 2.0, -np.inf],
    })
    out = w_mod.clip_and_fill(df)
    assert not np.isinf(out.values).any()
    assert not out.isna().any().any()
    # interpolation should bring middle near 2.0
    assert abs(out.loc[1, "a"] - 2.0) < 1e-6

def test_normalize_cols_mean0_std1():
    df = pd.DataFrame({"x":[1,2,3,4,5], "y":[10,10,10,10,10]})
    out = w_mod.normalize_cols(df, ["x","y"])
    # x normalized ~ mean 0 std 1
    assert abs(out["x"].mean()) < 1e-6
    assert abs(out["x"].std(ddof=0) - 1.0) < 1e-6
    # y has zero variance → stays 0 after (x - mu)/sigma with sigma forced to 1
    assert out["y"].eq(0.0).all()
