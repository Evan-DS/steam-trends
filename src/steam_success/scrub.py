"""`python main.py --scrub`: remove everything the program has made, so the project is clean again.

What is removed
    data/derived/   every file the pipeline derived from the source data (and any temporary downloads)
    output/         report.html and the figures
    logs/           the run logs
    caches          every __pycache__ folder and .pytest_cache inside the project

What is NOT touched
    data/source/    the two input files. Scrubbing must never delete the data the analysis starts
                    from: getting it back means the multi-hour --download (or the original zip).
    .gitkeep        the empty placeholder files that keep data/derived and output in git
    anything outside the project folder (for example the Hugging Face / Kaggle download caches
    in your home directory, which other projects may share)

Everything that is removed can be rebuilt with `python main.py`.
"""
import os
import shutil
from pathlib import Path

from steam_success import config

KEEP_FILES = {".gitkeep"}                                   # placeholders that must survive
SKIP_WHEN_SEARCHING = {".git", ".venv", "venv", "node_modules"}   # never look inside these for caches


# ---------------------------------------------------------------------------
# Safety checks and small helpers
# ---------------------------------------------------------------------------
def _why_not_safe(folder: Path) -> str | None:
    """Return the reason a folder must NOT be emptied, or None when emptying it is safe.

    The folders can be moved with environment variables (STEAM_DERIVED_DIR and friends). A
    typo there must never turn --scrub into "delete my documents", so only folders strictly
    inside the project, and not containing the source data, are ever emptied.
    """
    root = config.PROJECT_ROOT.resolve()
    target = folder.resolve()
    if target == root or root not in target.parents:
        return "it is not inside the project folder"
    source = config.SOURCE_DIR.resolve()
    if target == source or target in source.parents:
        return "it contains the source data"
    return None


def _show(path: Path) -> str:
    """A path as shown to the user: relative to the project when possible."""
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _file_stats(path: Path) -> tuple[int, int]:
    """(number of files, total bytes) of a file or of everything inside a folder."""
    if path.is_symlink() or path.is_file():
        return 1, path.lstat().st_size
    files = size = 0
    for sub in path.rglob("*"):
        if sub.is_file() and not sub.is_symlink():
            files += 1
            size += sub.stat().st_size
    return files, size


def _remove(path: Path, failures: list[str]) -> tuple[int, int]:
    """Delete a file or folder. Returns (files, bytes) removed; problems are added to `failures`."""
    files, size = _file_stats(path)
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()                  # a link is removed, never followed
        else:
            shutil.rmtree(path)
    except OSError as e:
        failures.append(f"{_show(path)} ({e.strerror or e})")   # e.g. the file is open in another program
        return 0, 0
    return files, size


def _clear_folder(folder: Path, failures: list[str], drop_folder_if_empty: bool = False) -> tuple[int, int]:
    """Delete everything inside a folder except the .gitkeep placeholder. Returns (files, bytes)."""
    if not folder.exists():
        return 0, 0
    reason = _why_not_safe(folder)
    if reason:
        print(f"  Skipped {folder}: {reason}.")
        return 0, 0
    files = size = 0
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.name in KEEP_FILES:
            continue
        f, b = _remove(entry, failures)
        files, size = files + f, size + b
    if drop_folder_if_empty and not any(folder.iterdir()):
        folder.rmdir()   # for a folder that git does not track and the program makes on demand
    return files, size


def _find_caches() -> list[Path]:
    """Every __pycache__ folder and .pytest_cache inside the project (virtual environments skipped)."""
    found = []
    for current, subfolders, _ in os.walk(config.PROJECT_ROOT):
        subfolders[:] = [d for d in subfolders if d not in SKIP_WHEN_SEARCHING]   # do not descend into these
        for name in list(subfolders):
            if name in ("__pycache__", ".pytest_cache"):
                found.append(Path(current) / name)
                subfolders.remove(name)   # no need to look inside a cache folder
    return found


def _megabytes(size: int) -> str:
    """Bytes as a short 'x.x MB' text."""
    return f"{size / 1e6:.1f} MB"


def _plural(count: int, word: str) -> str:
    """'1 file' / '3 files' (just adds an s: enough for the words used here)."""
    return f"{count} {word}" + ("" if count == 1 else "s")


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def scrub() -> bool:
    """Remove everything the program has generated and print a summary.

    Returns True if everything was removed, False if some items could not be deleted.
    Nothing is written to a log file: a log of the clean-up would itself be a leftover file.
    """
    failures: list[str] = []
    removed_anything = False

    print("\nScrubbing: removing everything the program has generated ...")
    # (what to show, folder, delete the folder itself once empty?)
    areas = [("Derived data", config.DERIVED_DIR, False),
             ("Output (report and figures)", config.OUTPUT_DIR, False),
             ("Logs", config.LOG_DIR, True)]
    for label, folder, drop_folder in areas:
        files, size = _clear_folder(folder, failures, drop_folder)
        if files:
            removed_anything = True
            print(f"  {label:<28} {_show(folder):<14} {_plural(files, 'file'):>9}, {_megabytes(size)}")

    # Python's compiled-code caches (they appear after any run, and after running pytest)
    caches = _find_caches()
    cache_files = 0
    for cache in caches:
        f, _ = _remove(cache, failures)
        cache_files += f
    if caches:
        removed_anything = True
        print(f"  {'Python caches':<28} {_plural(len(caches), 'folder')} ({_plural(cache_files, 'file')})")

    if failures:
        print("\nCould not remove (is it open in another program?):")
        for item in failures:
            print(f"  - {item}")
    if not removed_anything and not failures:
        print("  Nothing to remove: the project is already clean.")
    elif not failures:
        print(f"\nDone. Kept the input data in {_show(config.SOURCE_DIR)}. "
              "Everything removed can be rebuilt with: python main.py")
    return not failures
