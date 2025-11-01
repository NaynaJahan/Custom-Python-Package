# from __future__ import annotations
# import json, os
# from typing import Any, Dict
# from joblib import dump

# def save_model(model, out_path: str, meta: Dict[str, Any]):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     dump(model, out_path)
#     with open(os.path.join(os.path.dirname(out_path), "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)

from __future__ import annotations
import json, os
from typing import Any, Dict
from joblib import dump, load


def save_model(model, out_path: str, meta: Dict[str, Any]):
    """
    Save a trained model plus a small JSON metadata file alongside it.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump(model, out_path)
    with open(os.path.join(os.path.dirname(out_path), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_model(model_path: str):
    """
    Load a model that was saved with save_model().
    """
    return load(model_path)
