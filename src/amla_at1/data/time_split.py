from __future__ import annotations
import pandas as pd
from typing import Tuple

def split_by_date(
    X: pd.DataFrame, y: pd.Series,
    train_end: str, val_end: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Time-ordered split using inclusive date boundaries.
      - train: idx <= train_end
      - val:   train_end < idx <= val_end
      - test:  idx > val_end

    Notes:
      * Internally we use a DatetimeIndex for robust comparisons,
        then we convert back to `datetime.date` indexes to match tests.
    """
    X = X.copy(); y = y.copy()
    X.index = pd.to_datetime(X.index)
    y.index = pd.to_datetime(y.index)
    X = X.sort_index(); y = y.sort_index()

    te = pd.to_datetime(train_end)
    ve = pd.to_datetime(val_end)

    idx = X.index
    m_tr  = idx <= te
    m_val = (idx > te) & (idx <= ve)
    m_te  = idx > ve

    Xtr, ytr   = X.loc[m_tr],   y.loc[m_tr]
    Xval, yval = X.loc[m_val],  y.loc[m_val]
    Xte, yte   = X.loc[m_te],   y.loc[m_te]

    # Convert indexes back to `datetime.date` so `isoformat()` -> 'YYYY-MM-DD'
    Xtr.index   = Xtr.index.date;   ytr.index   = ytr.index.date
    Xval.index  = Xval.index.date;  yval.index  = yval.index.date
    Xte.index   = Xte.index.date;   yte.index   = yte.index.date

    return Xtr, ytr, Xval, yval, Xte, yte
