# __all__ = []
# __version__ = "2025.0.1.9"

# # sed -i '' 's/^__version__ = ".*"/__version__ = "2025.0.1.9"/' src/amla_at1/__init__.py

"""
Top-level convenience exports for amla_at1.
Bump the version here when you publish a new build.
"""

__version__ = "2025.0.2.0"

# Curated, convenient re-exports (so users can do `from amla_at1 import ...`)
from .data.openmeteo import fetch_daily_archive, make_supervised_tables
from .data.time_split import split_by_date

from .features.weather import clip_and_fill, normalize_cols

from .models.null import NullModel
from .models.performance import metrics_from_proba, weighted_blend
from .models.metrics_extra import cls_scores, reg_scores
from .models.export import save_model

__all__ = [
    "__version__",
    # Data
    "fetch_daily_archive",
    "make_supervised_tables",
    "split_by_date",
    # Features
    "clip_and_fill",
    "normalize_cols",
    # Models + metrics + export
    "NullModel",
    "metrics_from_proba",
    "weighted_blend",
    "cls_scores",
    "reg_scores",
    "save_model",
]
