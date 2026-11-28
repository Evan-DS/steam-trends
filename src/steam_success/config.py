"""Project-wide settings: folders, random seed, model settings and shared names."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders: all relative to the repository, wherever the program is run from.
# Each can be overridden with an environment variable.
# ---------------------------------------------------------------------------
# parents[2] climbs from src/steam_success/config.py up to the project folder
PROJECT_ROOT = Path(os.environ.get("STEAM_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_DIR = Path(os.environ.get("STEAM_SOURCE_DIR", PROJECT_ROOT / "data" / "source"))     # the two input files
DERIVED_DIR = Path(os.environ.get("STEAM_DERIVED_DIR", PROJECT_ROOT / "data" / "derived"))  # everything the steps make
OUTPUT_DIR = Path(os.environ.get("STEAM_OUTPUT_DIR", PROJECT_ROOT / "output"))              # report.html
FIG_DIR = OUTPUT_DIR / "figures"           # every chart is saved here as a .png
DOWNLOAD_DIR = DERIVED_DIR / "downloads"   # temporary review files when regenerating the source data
LOG_DIR = Path(os.environ.get("STEAM_LOG_DIR", PROJECT_ROOT / "logs"))                      # one log file per run

# The two source files everything is built from
GAMES_FILE = "games.parquet"                 # full Steam catalogue (Steam Games Dataset)
# Long text and media columns of the catalogue that the analysis never uses. They are left out of
# games.parquet, which keeps it as one file under GitHub's 100 MB limit without changing any result.
UNUSED_GAME_COLUMNS = ["About the game", "Header image", "Website", "Support url", "Support email",
                       "Metacritic url", "Screenshots", "Movies", "Reviews", "Notes", "Tags"]
GITHUB_FILE_LIMIT_MB = 100                         # GitHub refuses files larger than this
REVIEW_FEATURES_FILE = "review_features.parquet"   # early-review features of the Model B sample

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
GAMES_REPO = "FronkonGames/steam-games-dataset"            # Hugging Face dataset
REVIEWS_DATASET = "artermiloff/steam-games-reviews-2024"   # Kaggle dataset

# ---------------------------------------------------------------------------
# Analysis population (which games are kept)
# ---------------------------------------------------------------------------
MIN_REVIEWS = 20       # games with fewer user reviews have too noisy a "% positive"
YEAR_MIN, YEAR_MAX = 2015, 2023
PLAYTIME_MAX_H = 500   # median playtimes above this are idle / card-farming titles

# ---------------------------------------------------------------------------
# Reach tiers: the estimated number of owners is grouped into four tiers.
# REACH_BINS are the cut points applied to the lower end of the owner range.
# ---------------------------------------------------------------------------
REACH_BINS = [-1, 19_999, 99_999, 499_999, float("inf")]
REACH_LABELS = ["Low (<20k)", "Moderate (20k–100k)", "High (100k–500k)", "Very high (500k+)"]
REACH_SHORT = ["Low", "Moderate", "High", "Very high"]   # short names for chart axes
HIGH_REACH = 2             # tier code from which a game counts as high reach (≥ 100k owners)
ENGAGEMENT_MIN_TIER = 1    # engagement is modelled for tier ≥ Moderate (≥ 20k owners)

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
SEED = 42                  # fixed random seed: the same data always gives the same results
N_SPLITS = 5               # cross-validation: the data is cut into 5 parts ("folds")
N_REPEATS = 3              # Model A vs Model B: 5 folds × 3 repeats = 15 paired folds

# Default XGBoost settings (the "default" models; tuning.py searches for better ones)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                  random_state=SEED, n_jobs=-1)

# Columns of launch_features.parquet that are not model inputs (raw versions of logged features)
NON_FEATURE_COLUMNS = ("AppID", "list_price", "publisher_prior_games")

# ---------------------------------------------------------------------------
# Early-review features (Model B)
# ---------------------------------------------------------------------------
WINDOW_DAYS = 30           # only reviews written in a game's first 30 days are used
MIN_EARLY_REVIEWS = 10     # games with fewer early reviews are left out of Model B
SAMPLE_PER_TIER = 500      # Model B uses 500 random games from each reach tier
PLAYTIME_AT_REVIEW = "early_median_playtime_at_review_h"
REVIEW_FEATURES = ["early_pct_positive", PLAYTIME_AT_REVIEW, "early_pct_early_access",
                   "early_pct_received_free", "early_pct_steam_purchase", "early_pct_english",
                   "early_median_reviewer_games", "early_median_reviewer_reviews"]
# Playtime at review would make the engagement model circular (engagement IS playtime)
REVIEW_FEATURES_ENGAGEMENT = [c for c in REVIEW_FEATURES if c != PLAYTIME_AT_REVIEW]


def source_file(name: str) -> Path:
    """Path of a source file (data/source); the folder is created if needed."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    return SOURCE_DIR / name


def derived_file(name: str) -> Path:
    """Path of a file the pipeline makes (data/derived); the folder is created if needed."""
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    return DERIVED_DIR / name
