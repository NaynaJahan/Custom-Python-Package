# Custom Python Package - Utilities & baselines 

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project description
`amla_at1` is a lightweight Python package that bundles **data utilities**, **feature-engineering helpers**, **baseline models**, and **evaluation helpers**. It includes:
- **Data utils**: train/val/test splits, date-based splits, save/load sets.
- **Weather I/O**: Open-Meteo historical API wrapper for Sydney
- **Crypto utilities**: Fetches the latest Ethereum OHLC data from Kraken / CoinGecko, prepares historical/engineered features, builds supervised data to predict next_day_high.
- **Feature engineering**: calendar & ratios (draft dataset), weather cleaning/normalization
- **Baselines & metrics**: a constant-probability model, AUROC/Brier, extra cls/reg scores
- **Model export**: convenience function for saving model + metadata
- Used by Advanced Machine Learning Application's Assignment 1, Assignment 2 and Assignment 3 experimentation repo to train models, and by the FastAPI repositories of Assignment 2 and Assignment 3 to serve predictions.

The package is tested via `pytest` and designed to be imported directly in notebooks or Python scripts. The classes and functions are assessed using these test cases. 

---

## Project Structure

```markdown

amla_at1_python_pkg/
├─ src/
│  └─ amla_at1/
│     ├─ __init__.py
│     ├─ crypto/
│     │  ├─ __init__.py
│     │  ├─ etl.py
│     │  ├─ features.py
│     │  └─ ohlc_api.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  ├─ sets.py
│     │  ├─ openmeteo.py
│     │  └─ time_split.py
│     ├─ features/
│     │  ├─ __init__.py
│     │  ├─ dates.py
│     │  └─ weather.py
│     └─ models/
│        ├─ __init__.py
│        ├─ null.py
│        ├─ performance.py
│        ├─ export.py
│        └─ metrics_extra.py
├─ tests/
│  ├─ crypto/
│  │  ├─ test_etl.py
│  │  ├─ test_export_load.py
│  │  └─ test_features.py
│  ├─ data/
│  │  ├─ test_openmeteo.py
│  │  ├─ test_time_split.py
│  │  └─ test_sets.py
│  ├─ features/
│  │  ├─ test_weather.py
│  │  └─ test_dates.py
│  └─ models/
│  │  ├─ test_metrics_export.py
│  │  ├─ test_null.py
│  │  └─ test_performance.py
│  ├─ __init__.py
│  ├─ conftest.py
│  └─ test_data.py
├─ pyproject.toml
└─ README.md

```

--------

---

## Requirements

- Python **3.11.4**
- [Poetry](https://python-poetry.org/) (for dependency and build management)
- `pytest` (installed as a dev dependency) to run tests

---

## Setup (local development)

1. **Clone the repo**
   ```bash
   git clone https://github.com/naynajn/amla_at1_python_pkg.git
   cd amla_at1_python_pkg
   ```
---

2. **Pin Python 3.11.4 with pyenv**
   ```bash
    pyenv install 3.11.4
    pyenv local 3.11.4
   ```
---

3. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```
---

4. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

---

5. **Run the test suite**
   ```bash
   pytest -q
   ```
---

6. **(Optional) Launch Jupyter Lab / Notebook**
   ```bash
   poetry run jupyter lab
   ```

---

7. **Import necessary package in the notebook according to project needs,such as:**
   ```bash
   from amla_at1.features.dates import add_domain_features
   from amla_at1.models.performance import metrics_from_proba
   from amla_at1.data.openmeteo import fetch_daily_archive, make_supervised_tables
   from amla_at1.data.time_split import split_by_date
   from amla_at1.features.weather import clip_and_fill, normalize_cols
   from amla_at1.models.metrics_extra import cls_scores
   from amla_at1.models.export import save_model
   from amla_at1.data import save_sets
   from amla_at1.crypto.etl import load_local_history_csvs, build_nextday_high_supervised, latest_feature_row
   from amla_at1.crypto.ohlc_api import fetch_kraken_ohlc
   ```
---

### Installing from TestPyPI
   ```bash
   pip install --extra-index-url https://test.pypi.org/simple/ amla-at1==2025.0.3.1
   ```
---

### Versioning & releases
- Version and metadata are managed in pyproject.toml.
- Use poetry version <patch|minor|major> to bump versions, then poetry build and poetry publish.

### Attribution
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api#json_return_object)
- [Coin OHLC Chart by ID - CoinGecko API](https://docs.coingecko.com/v3.0.1/reference/coins-id-ohlc)
- [Get OHLC Data | Kraken API Center](https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/)