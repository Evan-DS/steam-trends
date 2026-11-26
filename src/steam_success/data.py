"""Reading the source files and the files the pipeline derives from them."""
import pandas as pd

from steam_success import config


class MissingDataError(Exception):
    """A file the step needs is not there."""


def _read(path, hint: str, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise MissingDataError(f"{path} not found. {hint}")
    return pd.read_parquet(path, **kwargs)


def read_source(name: str, **kwargs) -> pd.DataFrame:
    return _read(config.SOURCE_DIR / name,
                 f"Put {name} in {config.SOURCE_DIR} (see README), or regenerate it with: python main.py --download",
                 **kwargs)


def read_derived(name: str, **kwargs) -> pd.DataFrame:
    return _read(config.DERIVED_DIR / name, "Run the earlier steps first: python main.py", **kwargs)


def load_games_clean() -> pd.DataFrame:
    return read_derived("games_clean.parquet")


def load_launch_features() -> pd.DataFrame:
    return read_derived("launch_features.parquet")


def load_review_features() -> pd.DataFrame:
    return read_source(config.REVIEW_FEATURES_FILE)


def without_unused_columns(games: pd.DataFrame) -> pd.DataFrame:
    """The game catalogue without the text and media columns the analysis never reads."""
    return games.drop(columns=[c for c in config.UNUSED_GAME_COLUMNS if c in games.columns])


def save_games(games: pd.DataFrame, path) -> float:
    """Save the catalogue compactly (unused columns dropped, zstd compression); returns its size in MB."""
    tmp = path.with_suffix(".tmp")
    without_unused_columns(games).to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)   # replace only once the new file is complete
    return path.stat().st_size / 1e6


def feature_columns(features: pd.DataFrame) -> list[str]:
    """The model inputs among the launch-feature columns (log versions instead of raw counts)."""
    return [c for c in features.columns if c not in config.NON_FEATURE_COLUMNS]


def with_int_appid(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(AppID=df["AppID"].astype(int))
