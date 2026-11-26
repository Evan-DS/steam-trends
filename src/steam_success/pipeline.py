"""Runs the analysis from start to finish, printing progress as it goes.

Normal run (no options): starts from the two files in data/source and needs no internet.
"""
import argparse
import importlib
import sys
import time
import traceback

from steam_success import config
from steam_success.data import MissingDataError
from steam_success.progress import banner, status

# (module, title) in running order. load_and_join only runs with --download.
STEPS = [
    ("clean", "Cleaning and the three success measures"),
    ("features", "The 42 launch features"),
    ("model_a", "Model A: launch features only"),
    ("model_b", "Model B: adding early-review features"),
    ("explain", "What drives the predictions (SHAP)"),
    ("tuning", "Tuning and learning curves (the slow step, about 30 minutes)"),
    ("report", "The results report"),
]
NAMES = [name for name, _ in STEPS]


def parse_steps(text: str) -> list[str]:
    """'4', '2-5' or '3,6,7' (numbers as listed by --list) into step names."""
    chosen = []
    for part in text.split(","):
        if "-" in part:
            first, last = (int(x) for x in part.split("-"))
            chosen += NAMES[first - 1:last]
        elif part.strip().isdigit():
            chosen.append(NAMES[int(part) - 1])
        elif part.strip() in NAMES:
            chosen.append(part.strip())
        else:
            raise SystemExit(f"Unknown step '{part}'. See: python main.py --list")
    return chosen


def check_sources(download: bool, steps: list[str]):
    """Stop early, with a clear message, if a source file the chosen steps need is missing."""
    if download:
        return
    needed = []
    if {"clean", "features"} & set(steps):
        needed.append(config.GAMES_FILE)
    if {"model_b", "explain", "tuning"} & set(steps):
        needed.append(config.REVIEW_FEATURES_FILE)
    missing = [f for f in needed if not (config.SOURCE_DIR / f).exists()]
    if missing:
        sys.exit(f"\nMissing source data in {config.SOURCE_DIR}: {', '.join(missing)}\n"
                 "Copy the project's data files there (see README), or regenerate them from the original\n"
                 "sources with:  python main.py --download")


def slim_games():
    """Rewrite data/source/games.parquet without the unused text columns, so it fits on GitHub."""
    import pandas as pd

    from steam_success.data import save_games

    path = config.SOURCE_DIR / config.GAMES_FILE
    if not path.exists():
        sys.exit(f"{path} not found.")
    before = path.stat().st_size / 1e6
    status(f"Reading {path.name} ({before:.0f} MB) ...")
    after = save_games(pd.read_parquet(path), path)
    status(f"Saved {path.name}: {before:.0f} MB -> {after:.0f} MB "
           f"({'fits' if after < config.GITHUB_FILE_LIMIT_MB else 'still over'} GitHub's "
           f"{config.GITHUB_FILE_LIMIT_MB} MB file limit). Results are unchanged: the analysis never reads "
           "the columns that were dropped.")


def warn_if_too_big():
    path = config.SOURCE_DIR / config.GAMES_FILE
    if path.exists() and path.stat().st_size / 1e6 > config.GITHUB_FILE_LIMIT_MB:
        status(f"Note: {path.name} is over GitHub's {config.GITHUB_FILE_LIMIT_MB} MB limit. "
               "Shrink it once with: python main.py --slim-games")


def run(steps: list[str], download: bool = False):
    check_sources(download, steps)
    banner("Steam game success: analysis pipeline")
    status(f"Source data : {config.SOURCE_DIR}")
    status(f"Derived data: {config.DERIVED_DIR}")
    status(f"Output      : {config.OUTPUT_DIR}")
    warn_if_too_big()

    todo = list(steps)
    if download:
        todo = ["load_and_join"] + todo
        status("Regenerating the source data from Hugging Face and Kaggle (--download)")

    t_all = time.time()
    for i, name in enumerate(todo, 1):
        title = dict(STEPS).get(name, "Download and join the source data")
        banner(f"Step {i}/{len(todo)} · {title}")
        t0 = time.time()
        module = importlib.import_module(f"steam_success.steps.{name}")
        try:
            module.main(allow_download=download) if name == "model_b" else module.main()
        except MissingDataError as e:
            sys.exit(f"\nStopped at step '{name}': {e}")
        except Exception:
            traceback.print_exc()
            sys.exit(f"\nStopped at step '{name}' because of the error above. "
                     f"Steps before it finished, and their files are in {config.DERIVED_DIR}.")
        status(f"Step {i}/{len(todo)} done in {(time.time() - t0) / 60:.1f} min")

    banner(f"Finished in {(time.time() - t_all) / 60:.1f} min")
    report = config.OUTPUT_DIR / "report.html"
    if report.exists():
        status(f"Open the results: {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="What makes a Steam game successful? Runs the whole analysis with no options.")
    parser.add_argument("--steps", help="run only some steps, e.g. 3 or 2-4 or 3,7 (see --list)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--download", action="store_true",
                        help="regenerate data/source from Hugging Face and Kaggle first (needs a Kaggle API token)")
    parser.add_argument("--slim-games", action="store_true",
                        help="shrink data/source/games.parquet under 100 MB (drops unused text columns) and exit")
    args = parser.parse_args(argv)

    if args.slim_games:
        slim_games()
        return

    if args.list:
        for i, (name, title) in enumerate(STEPS, 1):
            print(f"{i}. {name:<9} {title}")
        return
    run(parse_steps(args.steps) if args.steps else NAMES, download=args.download)
