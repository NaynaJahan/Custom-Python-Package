# tests/models/test_metrics_export.py
import pathlib, importlib.util, json
import numpy as np
from sklearn.dummy import DummyRegressor
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]

# metrics_extra
M_PATH = ROOT / "src/amla_at1/models/metrics_extra.py"
m_spec = importlib.util.spec_from_file_location("mx_mod", M_PATH)
mx_mod = importlib.util.module_from_spec(m_spec)
m_spec.loader.exec_module(mx_mod)

# export
E_PATH = ROOT / "src/amla_at1/models/export.py"
e_spec = importlib.util.spec_from_file_location("exp_mod", E_PATH)
exp_mod = importlib.util.module_from_spec(e_spec)
e_spec.loader.exec_module(exp_mod)

def test_cls_scores_keys_and_ranges():
    y = np.array([0,1,1,0,1,0,1,0])
    yhat = np.array([0,1,1,0,0,0,1,1])
    s = mx_mod.cls_scores(y, yhat)
    for k in ["f1","balanced_acc","precision","recall"]:
        assert k in s
        assert 0.0 <= s[k] <= 1.0

def test_reg_scores_numbers():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    yhat = np.array([0.0, 1.5, 2.5, 2.5])
    s = mx_mod.reg_scores(y, yhat)
    assert set(["rmse","mae","r2"]).issubset(s.keys())
    assert s["rmse"] > 0 and s["mae"] > 0
    assert -1.0 <= s["r2"] <= 1.0

def test_save_model_writes_joblib_and_meta(tmp_path):
    # simple fitted model
    X = pd.DataFrame({"x":[0,1,2,3]})
    y = np.array([0.0, 1.0, 1.0, 2.0])
    m = DummyRegressor(strategy="mean").fit(X, y)

    out_dir = tmp_path / "models"
    out_path = out_dir / "model.joblib"
    meta = {"task":"demo","features":["x"],"train_end":"2024-12-31"}

    exp_mod.save_model(m, out_path.as_posix(), meta)

    assert out_path.exists()
    meta_path = out_dir / "meta.json"
    assert meta_path.exists()
    j = json.loads(meta_path.read_text())
    assert j["task"] == "demo"
    assert j["features"] == ["x"]
