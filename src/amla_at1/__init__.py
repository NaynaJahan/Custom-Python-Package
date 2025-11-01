"""
Top-level convenience exports for amla_at1.
Bump the version here when you publish a new build.
"""

__version__ = "2025.0.3.1"  

# -----------------
# Existing imports (from AT1 weather / general utils)
# -----------------
from .data.openmeteo import fetch_daily_archive, make_supervised_tables
from .data.time_split import split_by_date
from .features.weather import clip_and_fill, normalize_cols
from .models.null import NullModel
from .models.performance import metrics_from_proba, weighted_blend
from .models.metrics_extra import cls_scores, reg_scores
from .models.export import save_model, load_model

# -----------------
# New crypto imports for AT3
# -----------------
from .crypto.ohlc_api import fetch_kraken_ohlc, fetch_coingecko_ohlc
from .crypto.etl import (
    load_local_history_csvs,
    build_nextday_high_supervised,
    latest_feature_row,
)
from .crypto.features import add_lagged_features, add_rolling_features

__all__ = [
    "__version__",

    # --- data (AT1 legacy) ---
    "fetch_daily_archive",
    "make_supervised_tables",
    "split_by_date",

    # --- features / cleaning (AT1 legacy) ---
    "clip_and_fill",
    "normalize_cols",

    # --- model utils (AT1 legacy) ---
    "NullModel",
    "metrics_from_proba",
    "weighted_blend",
    "cls_scores",
    "reg_scores",
    "save_model",
    "load_model",

    # --- crypto (new for AT3) ---
    "fetch_kraken_ohlc",
    "fetch_coingecko_ohlc",
    "load_local_history_csvs",
    "build_nextday_high_supervised",
    "latest_feature_row",
    "add_lagged_features",
    "add_rolling_features",
]