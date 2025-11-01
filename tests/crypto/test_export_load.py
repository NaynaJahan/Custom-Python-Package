# tests/crypto/test_export_load.py

import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from amla_at1.models.export import save_model, load_model

def test_save_and_load_roundtrip(tmp_path):
    X = pd.DataFrame({"x":[0,1,2,3]})
    y = np.array([0.0, 1.0, 1.0, 2.0])

    m = DummyRegressor(strategy="mean").fit(X, y)

    out_dir = tmp_path / "models"
    out_path = out_dir / "rf_eth.joblib"
    meta = {
        "task":"eth_next_day_high",
        "features":["x"],
        "train_end":"2025-01-01"
    }

    save_model(m, out_path.as_posix(), meta)

    reloaded = load_model(out_path.as_posix())
    yhat_orig = m.predict(X)
    yhat_new  = reloaded.predict(X)
    assert np.allclose(yhat_orig, yhat_new)

    meta_path = out_dir / "meta.json"
    assert meta_path.exists()
    j = json.loads(meta_path.read_text())
    assert j["task"] == "eth_next_day_high"
