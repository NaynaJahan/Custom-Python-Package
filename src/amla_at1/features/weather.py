import pandas as pd
import numpy as np

def clip_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: replace inf with nan, forward/backward fill small gaps."""
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    return out.interpolate(limit_direction="both").ffill().bfill()

def normalize_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            x = out[c]
            mu, sigma = x.mean(), x.std() or 1.0
            out[c] = (x - mu) / sigma
    return out
