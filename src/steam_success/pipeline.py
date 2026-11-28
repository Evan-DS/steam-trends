"""Runs the analysis from start to finish, printing progress as it goes.

Normal run (no options): starts from the two files in data/source and needs no internet.
Everything the run prints is also saved to a log file in logs/ (see runlog.py).

Other commands handled here:
    --list        show the steps
    --steps       run only some of the steps
    --download    rebuild data/source from Hugging Face and Kaggle first
    --slim-games  shrink data/source/games.parquet so it fits on GitHub
    --scrub       delete everything the program has generated (see scrub.py)
"""
import argparse
import importlib
import sys
import time
import traceback

from steam_success import config, runlog
from steam_success.data import MissingDataError
from steam_success.progress import banner, status

# The steps in running order, as (module name inside steps/, title shown to the user).
# The numbers shown by --list and accepted by --steps are the positions in this list.
# load_and_join is not in the list: it only runs when --download is given.
STEPS = [
    ("clean", "Cleaning and the three success measures"),
    ("features", "The 42 launch features"),
    ("model_a", "Model A: launch features only"),
    ("model_b", "Model B: adding early-review features"),
    ("explain", "What drives the predictions (SHAP)"),
    ("tuning", "Tuning and learning curves (the slow step, about 30 minutes)"),
    ("report", "The results report"),
]
NAMES = [name for name, _ in STEPS]   # just the module names, in order


# ---------------------------------------------------------------------------
# Choosing and checking the steps
# ---------------------------------------------------------------------------
def parse_steps(text: str) -> list[str]:
    """Turn the --steps text into step names.

    Accepts a single number ('4'), a range ('2-5'), a comma list ('3,6,7') or step names
    ('model_a'). The numbers are the ones shown by --list.
    """
    chosen = []
    for part in text.split(","):
        if "-" in part:
            # a range such as "2-5" means steps 2, 3, 4 and 5 (list positions start at 0, so subtract 1)
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
        return   # --download creates the source files itself, so there is nothing to check
    needed = []
    if {"clean", "features"} & set(steps):                    # these read the game catalogue
        needed.append(config.GAMES_FILE)
    if {"model_b", "explain", "tuning"} & set(steps):         # these read the early-review features
        needed.append(config.REVIEW_FEATURES_FILE)
    missing = [f for f in needed if not (config.SOURCE_DIR / f).exists()]
    if missing:
        sys.exit(f"\nMissing source data in {config.SOURCE_DIR}: {', '.join(missing)}\n"
                 "Copy the project's data files there (see README), or regenerate them from the original\n"
                 "sources with:  python main.py --download")


# ---------------------------------------------------------------------------
# --slim-games
# ---------------------------------------------------------------------------
def slim_games():
    """Rewrite data/source/games.parquet without the unused text columns, so it fits on GitHub."""
    import pandas as pd   # imported here because only this option needs it

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
    """Print a hint if games.parquet is too large to be uploaded to GitHub."""
    path = config.SOURCE_DIR / config.GAMES_FILE
    if path.exists() and path.stat().st_size / 1e6 > config.GITHUB_FILE_LIMIT_MB:
        status(f"Note: {path.name} is over GitHub's {config.GITHUB_FILE_LIMIT_MB} MB limit. "
               "Shrink it once with: python main.py --slim-games")


# ---------------------------------------------------------------------------
# Running the steps
# ---------------------------------------------------------------------------
def run(steps: list[str], download: bool = False):
    """Run the chosen steps one after another, announcing and timing each of them."""
    check_sources(download, steps)
    banner("Steam game success: analysis pipeline")
    status(f"Source data : {config.SOURCE_DIR}")
    status(f"Derived data: {config.DERIVED_DIR}")
    status(f"Output      : {config.OUTPUT_DIR}")
    if runlog.log_path():   # None when run without a log (for example from the tests)
        status(f"Log file    : {runlog.log_path()}")
    warn_if_too_big()

    todo = list(steps)
    if download:
        todo = ["load_and_join"] + todo   # rebuilding the source data always comes first
        status("Regenerating the source data from Hugging Face and Kaggle (--download)")

    t_all = time.time()
    for i, name in enumerate(todo, 1):
        title = dict(STEPS).get(name, "Download and join the source data")
        banner(f"Step {i}/{len(todo)} · {title}")
        t0 = time.time()
        # Each step is a module in steps/ with a main() function; import it only when it is needed,
        # so a short run does not wait for the libraries of steps it will not run.
        module = importlib.import_module(f"steam_success.steps.{name}")
        try:
            # model_b is the only step that can download review data, so it is told whether it may
            module.main(allow_download=download) if name == "model_b" else module.main()
        except MissingDataError as e:
            sys.exit(f"\nStopped at step '{name}': {e}")
        except Exception:
            traceback.print_exc()   # show the full error, then stop with a short summary
            sys.exit(f"\nStopped at step '{name}' because of the error above. "
                     f"Steps before it finished, and their files are in {config.DERIVED_DIR}.")
        status(f"Step {i}/{len(todo)} done in {(time.time() - t0) / 60:.1f} min")

    banner(f"Finished in {(time.time() - t_all) / 60:.1f} min")
    report = config.OUTPUT_DIR / "report.html"
    if report.exists():
        status(f"Open the results: {report}")
    if runlog.log_path():
        status(f"Log saved   : {runlog.log_path()}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The command-line options (python main.py --help lists them)."""
    parser = argparse.ArgumentParser(
        description="What makes a Steam game successful? Runs the whole analysis with no options.")
    parser.add_argument("--steps", help="run only some steps, e.g. 3 or 2-4 or 3,7 (see --list)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    parser.add_argument("--download", action="store_true",
                        help="regenerate data/source from Hugging Face and Kaggle first (needs a Kaggle API token)")
    parser.add_argument("--slim-games", action="store_true",
                        help="shrink data/source/games.parquet under 100 MB (drops unused text columns) and exit")
    parser.add_argument("--scrub", action="store_true",
                        help="delete everything the program has generated (data/derived, output, logs, caches) "
                             "and exit; the input data in data/source is kept")
    return parser


def main(argv=None):
    """Entry point: read the options, then scrub, list, or run (with a log file)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # --scrub deletes files, so it must never be combined with anything that creates them
    if args.scrub:
        if args.steps or args.list or args.download or args.slim_games:
            parser.error("--scrub cannot be combined with other options")
        from steam_success.scrub import scrub   # imported here: only this option needs it
        if not scrub():
            sys.exit(1)   # some files could not be removed
        return

    if args.list and not args.slim_games:
        for i, (name, title) in enumerate(STEPS, 1):
            print(f"{i}. {name:<9} {title}")
        return

    # Everything from here on does real work, so its output is also saved to a log file.
    command = " ".join(["python main.py"] + list(sys.argv[1:] if argv is None else argv))
    with runlog.RunLog(command):
        try:
            if args.slim_games:
                slim_games()
            else:
                run(parse_steps(args.steps) if args.steps else NAMES, download=args.download)
        except SystemExit as stop:
            # sys.exit("some message") prints its message only after the program has fully stopped,
            # which is too late for the log. Print it now instead (it reaches the screen and the log),
            # then stop with the same "failed" exit code.
            if isinstance(stop.code, str):
                print(stop.code, file=sys.stderr)
                raise SystemExit(1) from None
            raise
