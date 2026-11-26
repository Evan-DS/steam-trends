"""What Makes a Steam Game Successful? Run the whole analysis:

    python main.py

Options (none are needed):
    python main.py --list          show the steps
    python main.py --steps 3-4     run only some steps
    python main.py --download      regenerate the source data from Hugging Face and Kaggle first
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from steam_success.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
