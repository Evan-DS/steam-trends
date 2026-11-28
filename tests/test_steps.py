"""Small checks of the data-handling functions (no downloads, runs in seconds)."""
import numpy as np
import pandas as pd
import pytest

from steam_success.pipeline import parse_steps
from steam_success.steps.clean import add_targets, split_list
from steam_success.steps.features import first_name, parse_list
from steam_success.steps.load_and_join import load_games
from steam_success.steps.model_b import early_features


def test_split_list():
    """A comma-separated field becomes a clean list; a missing value becomes an empty list."""
    assert split_list("Action, Indie ,") == ["Action", "Indie"]
    assert split_list(np.nan) == []


def test_parse_list_handles_python_lists_and_plain_text():
    """List fields can be written as a Python list or as plain comma-separated text."""
    assert parse_list("['English', 'French']") == ["English", "French"]
    assert parse_list("[]") == []
    assert parse_list("English, French") == ["English", "French"]


def test_first_name_normalises():
    """Only the first publisher is kept, in lower case; an empty field has no name."""
    assert first_name("Valve, Hidden Path") == "valve"
    assert pd.isna(first_name(""))


def test_header_fix(tmp_path):
    """The glued 'DiscountDLC count' header is split in two, so every later column lines up with its data."""
    path = tmp_path / "games.csv"
    path.write_text("AppID,Name,Price,DiscountDLC count,Positive\n10,Game,9.99,0,5,100\n")
    games = load_games(str(path))
    assert list(games.columns) == ["AppID", "Name", "Price", "Discount", "DLC count", "Positive"]
    assert games.loc[0, "Positive"] == 100


def test_targets():
    """The three success measures and the regular price are worked out correctly for two example games."""
    df = pd.DataFrame({"Estimated owners": ["20000 - 50000", "0 - 20000"], "Positive": [80, 5],
                       "n_reviews": [100, 20], "Median playtime forever": [120, 0],
                       "Price": [7.49, 0.0], "Discount": [50, 0]})
    out = add_targets(df)
    assert out["pct_positive"].tolist() == [0.8, 0.25]         # 80 of 100 reviews, 5 of 20 reviews
    assert out["reach_code"].tolist() == [1, 0]                # 20k owners = tier 1, 0 owners = tier 0
    # 120 minutes = 2 hours; a playtime of 0 means "no data", so it is missing
    assert out.loc[0, "median_playtime_h"] == 2 and pd.isna(out.loc[1, "median_playtime_h"])
    assert out.loc[0, "list_price"] == pytest.approx(14.98)    # 7.49 at 50% off was 14.98 before the discount


def test_early_features_window():
    """Only reviews inside the first 30 days count towards the early-review features."""
    day = 86_400   # seconds in a day (review times are stored in seconds)
    reviews = pd.DataFrame({"timestamp_created": [0, day, 40 * day], "voted_up": [True, False, True],
                            "author_playtime_at_review": [60, 180, 600], "language": ["english"] * 3})
    f = early_features(reviews, pd.Timestamp(0))
    assert f["n_early_reviews"] == 2           # the review on day 40 is outside the window
    assert f["early_pct_positive"] == 0.5
    assert f["early_median_playtime_at_review_h"] == 2.0   # median of 60 and 180 minutes = 120 minutes = 2 hours


def test_parse_steps():
    """--steps accepts ranges, comma lists and step names."""
    assert parse_steps("2-4") == ["features", "model_a", "model_b"]
    assert parse_steps("3,7") == ["model_a", "report"]
    assert parse_steps("model_a") == ["model_a"]


def test_unused_columns_dropped_and_saved(tmp_path):
    """The unused text columns are dropped, both in memory and in the saved file."""
    from steam_success import config
    from steam_success.data import save_games, without_unused_columns

    games = pd.DataFrame({"AppID": [1, 2], "Price": [9.99, 0.0], "About the game": ["long text", "more"],
                          "Screenshots": ["a.jpg", "b.jpg"]})
    slim = without_unused_columns(games)
    assert list(slim.columns) == ["AppID", "Price"]
    assert not set(config.UNUSED_GAME_COLUMNS) & set(slim.columns)
    path = tmp_path / "games.parquet"
    save_games(games, path)
    assert list(pd.read_parquet(path).columns) == ["AppID", "Price"]
