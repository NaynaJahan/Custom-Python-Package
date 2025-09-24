from __future__ import annotations
import pandas as pd
from typing import Tuple

def split_by_date(
    X: pd.DataFrame, y: pd.Series,
    train_end: str, val_end: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Time-ordered split using inclusive date boundaries (index must be date).
    train: index <= train_end
    val:   train_end < index <= val_end
    test:  val_end < index (optional; may be empty in experiments)
    """
    Xtr = X.loc[:train_end]; ytr = y.loc[:train_end]
    Xval = X.loc[train_end:].loc[:val_end].iloc[1:]; yval = y.loc[train_end:].loc[:val_end].iloc[1:]
    Xte = X.loc[val_end:].iloc[1:]; yte = y.loc[val_end:].iloc[1:]
    return Xtr, ytr, Xval, yval, Xte, yte
