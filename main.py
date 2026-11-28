"""What Makes a Steam Game Successful? Run the whole analysis:

    python main.py

Options (none are needed):
    python main.py --list          show the steps
    python main.py --steps 3-4     run only some steps
    python main.py --download      regenerate the source data from Hugging Face and Kaggle first
    python main.py --slim-games    shrink data/source/games.parquet so it fits on GitHub
    python main.py --scrub         delete everything the program has generated (the input data is kept)

Every run also saves a copy of its live console output to a file in logs/.
"""
import sys
from pathlib import Path

# A clean-up should not leave new files behind. Normally Python saves compiled copies of the code in
# __pycache__ folders as it imports it; for --scrub that is switched off, otherwise the command that
# removes those folders would immediately create them again.
if "--scrub" in sys.argv:
    sys.dont_write_bytecode = True

# The code lives in src/, which Python does not search by default, so add it to the search path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from steam_success.pipeline import main  # noqa: E402  (must come after the path line above)

if __name__ == "__main__":
    main()
