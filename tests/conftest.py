"""pytest set-up: make the project's code importable from the tests."""
import sys
from pathlib import Path

# The code lives in src/, which Python does not search by default; add it to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
