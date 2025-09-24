# from .sets import pop_target, save_sets, load_sets, subset_x_y, stratified_split
# __all__ = ["pop_target", "save_sets", "load_sets", "subset_x_y", "stratified_split"]


# from .sets import pop_target, save_sets, load_sets, subset_x_y, stratified_split
# from .openmeteo import fetch_daily_archive, make_supervised_tables

# __all__ = [
#     "pop_target", "save_sets", "load_sets", "subset_x_y", "stratified_split",
#     "fetch_daily_archive", "make_supervised_tables",
# ]


"""
Data utilities: dataset IO, splits, Open-Meteo fetchers.
"""

from .sets import pop_target, save_sets, load_sets, subset_x_y, stratified_split
from .time_split import split_by_date
from .openmeteo import fetch_daily_archive, make_supervised_tables

__all__ = [
    # sets.py
    "pop_target",
    "save_sets",
    "load_sets",
    "subset_x_y",
    "stratified_split",
    # time_split.py
    "split_by_date",
    # openmeteo.py
    "fetch_daily_archive",
    "make_supervised_tables",
]