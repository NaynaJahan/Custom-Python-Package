# Utilities & baselines for the Advanced Machine Learning Application AT1 AT1 - Kaggle Competition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project description
`amla_at1` is a lightweight Python package that bundles **data utilities**, **feature-engineering helpers**, **baseline models**, and **evaluation helpers** used in the AMLA AT1 Kaggle Competition. It includes:
- A simple **NullModel** baseline with `fit()` / `predict_proba()` that returns calibrated constant probabilities.
- **Performance helpers** to compute AUROC, Brier score, and to ensemble multiple probability vectors.
- **Feature engineering** for the draft dataset (height parsing, year ordinal, type flags, shooting ratios, per-minute rates), which are relevant and important features of the project.
- **Dataset helpers** to pop the target, save/load splits, and run a stratified train/validation/test split.

The package is tested via `pytest` and designed to be imported directly in notebooks or Python scripts. The classes and functions are assessed using these test cases. 

---

## Project Structure

```
amla_at1_python_pkg/
├─ src/
│ └─ amla_at1/
│   ├─ __init__.py
│   ├─ data/
│   │    └─ __init__.py
|   |    └─ sets.py
│   ├─ features/
│   │    └─ __init__.py
|   |    └─ dates.py
│   ├─ models/
│   |    ├─ __init__.py
│   |    ├─ null.py
|   └─———└─ performance.py
├─ tests/
│ ├─ data/
| |    └─ test_sets.py
│ ├─ features/
| |    └─ test_dates.py
│ ├─ models/
│ |    ├─ test_null.py
| |    └─ test_performance.py
│ └─ test_data.py
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
---

2. **Pin Python 3.11.4 with pyenv**
   ```bash
    pyenv install 3.11.4
    pyenv local 3.11.4

---

3. **Install dependencies with Poetry**
   ```bash
   poetry install
---

4. **Activate the virtual environment**
   ```bash
   poetry shell

---

5. **Run the test suite**
   ```bash
   pytest -q

---

6. **(Optional) Launch Jupyter Lab / Notebook**
   ```bash
   poetry run jupyter lab

---

7. **Import necessary package in the notebook according to project needs,such as:**
   ```bash
   from amla_at1.features.dates import add_domain_features
   from amla_at1.models.performance import metrics_from_proba
---

### Installing from TestPyPI
   ```
   bash
   pip install -i https://test.pypi.org/simple/ amla-at1
   import amla_at1
   ```
---

### Versioning & releases
- Version and metadata are managed in pyproject.toml.
- Use poetry version <patch|minor|major> to bump versions, then poetry build and poetry publish.