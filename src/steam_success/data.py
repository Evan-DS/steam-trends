"""Reading the source files and the files the pipeline derives from them."""
import pandas as pd

from steam_success import config


class MissingDataError(Exception):
    """A file the step needs is not there."""


def _read(path, hint: str, **kwargs) -> pd.DataFrame:
    """Read a parquet file, or stop with a helpful message (`hint`) if it does not exist.

    Extra keyword arguments (such as columns=[...]) are passed on to pandas.read_parquet.
    """
    if not path.exists():
        raise MissingDataError(f"{path} not found. {hint}")
    return pd.read_parquet(path, **kwargs)


def read_source(name: str, **kwargs) -> pd.DataFrame:
    """Read one of the two input files from data/source."""
    return _read(config.SOURCE_DIR / name,
                 f"Put {name} in {config.SOURCE_DIR} (see README), or regenerate it with: python main.py --download",
                 **kwargs)


def read_derived(name: str, **kwargs) -> pd.DataFrame:
    """Read a file made by an earlier step from data/derived."""
    return _read(config.DERIVED_DIR / name, "Run the earlier steps first: python main.py", **kwargs)


def load_games_clean() -> pd.DataFrame:
    """The filtered games with their three success measures (made by the clean step)."""
    return read_derived("games_clean.parquet")


def load_launch_features() -> pd.DataFrame:
    """The 42 launch features per game (made by the features step)."""
    return read_derived("launch_features.parquet")


def load_review_features() -> pd.DataFrame:
    """The early-review features of the Model B sample (an input file in data/source)."""
    return read_source(config.REVIEW_FEATURES_FILE)


def without_unused_columns(games: pd.DataFrame) -> pd.DataFrame:
    """The game catalogue without the text and media columns the analysis never reads."""
    return games.drop(columns=[c for c in config.UNUSED_GAME_COLUMNS if c in games.columns])


def save_games(games: pd.DataFrame, path) -> float:
    """Save the catalogue compactly (unused columns dropped, zstd compression); returns its size in MB."""
    tmp = path.with_suffix(".tmp")   # write to a temporary file first ...
    without_unused_columns(games).to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)   # ... and replace the real file only once the new one is complete
    return path.stat().st_size / 1e6


def feature_columns(features: pd.DataFrame) -> list[str]:
    """The model inputs among the launch-feature columns (log versions instead of raw counts)."""
    return [c for c in features.columns if c not in config.NON_FEATURE_COLUMNS]


def with_int_appid(df: pd.DataFrame) -> pd.DataFrame:
    """A copy with AppID as a plain integer, so tables from different files can be joined on it."""
    return df.assign(AppID=df["AppID"].astype(int))
