from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score

def cls_scores(y_true, y_pred) -> Dict[str, float]:
    f1 = float(f1_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    p,r,_,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"f1": f1, "balanced_acc": bacc, "precision": float(p), "recall": float(r)}

def reg_scores(y_true, y_pred) -> Dict[str, float]:
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}
