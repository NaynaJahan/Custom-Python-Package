# tests/conftest.py
import sys
import pathlib

# Root of the project = parent of "tests"
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Our package lives in ROOT / "src"
SRC = ROOT / "src"

# Add "<repo>/src" to sys.path so `import amla_at1` works
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
