import pathlib, importlib.util
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TS_PATH = ROOT / "src/amla_at1/data/time_split.py"
spec = importlib.util.spec_from_file_location("ts_mod", TS_PATH)
ts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts_mod)

def test_split_by_date_boundaries():
    dates = pd.date_range("2024-09-25", periods=72, freq="D").date
    X = pd.DataFrame({"x": np.arange(len(dates))}, index=dates)
    y = pd.Series(np.arange(len(dates)), index=dates)

    Xtr, ytr, Xval, yval, Xte, yte = ts_mod.split_by_date(X, y, train_end="2024-10-31", val_end="2024-11-30")

    assert len(Xtr) > 0 and len(Xval) > 0 and len(Xte) > 0
    assert Xtr.index.max().isoformat() == "2024-10-31"
    assert Xval.index.min() > Xtr.index.max()
    assert Xval.index.max().isoformat() == "2024-11-30"
    assert Xte.index.min() > Xval.index.max()
    # alignment
    assert Xtr.index.equals(ytr.index)
    assert Xval.index.equals(yval.index)
    assert Xte.index.equals(yte.index)
