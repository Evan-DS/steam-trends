"""Regeneration only: download the game metadata and check that review files join to it.

Not part of the normal run, which starts from data/source. Used by: python main.py --download

Inputs : Steam Games Dataset (Hugging Face), Steam Games Reviews 2024 (Kaggle)
Outputs: data/source/games.parquet; data/derived/review_counts.parquet, review_schema.csv

"Join" means matching rows of two tables through a shared key; here, the game's AppID, which appears
in both the game catalogue and the review files.
"""
import csv
import glob
import os
import re
import time
import zipfile
from collections import Counter

import pandas as pd

from steam_success import config
from steam_success.data import save_games
from steam_success.kaggle_client import connect, with_retry
from steam_success.printing import print_table
from steam_success.progress import status

SAMPLE_GB = 1      # size of the review sample used to check the join
MAX_FILES = 60     # upper bound on review files downloaded


# ---------------------------------------------------------------------------
# Game metadata
# ---------------------------------------------------------------------------
def download_games_csv() -> str:
    """Download games.csv from Hugging Face (cached on disk after the first time) and return its path."""
    from huggingface_hub import hf_hub_download   # imported here: only needed when regenerating the data

    path = hf_hub_download(config.GAMES_REPO, "games.csv", repo_type="dataset")
    # the cache stores the file under a snapshot folder; search for it so the exact layout does not matter
    return glob.glob(f"{os.path.dirname(path)}/**/games.csv", recursive=True)[0]


def load_games(path: str) -> pd.DataFrame:
    """Load games.csv, splitting the merged 'DiscountDLC count' header field.

    The file's header glues two column names together, which shifts every later
    name one column to the left. Splitting it puts each name back on its data.
    """
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))   # just the first line: the column names
    if "DiscountDLC count" in header:
        i = header.index("DiscountDLC count")
        header[i:i + 1] = ["Discount", "DLC count"]   # replace the one glued name with the two real ones
    # Read the data with our corrected names (skiprows=1 skips the broken header line)
    df = pd.read_csv(path, names=header, skiprows=1, index_col=False, low_memory=False)
    df["AppID"] = pd.to_numeric(df["AppID"], errors="coerce").astype("Int64")
    for c in ["Price", "Positive", "Negative", "Median playtime forever"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def check_alignment(games: pd.DataFrame):
    """After the header fix, numeric columns must actually hold numbers."""
    numeric_share = {c: games[c].notna().mean() for c in ["Price", "Positive"]}
    # if the columns were still shifted, these would be mostly text (missing after conversion)
    assert all(v > 0.95 for v in numeric_share.values()), f"Column alignment check failed: {numeric_share}"


# ---------------------------------------------------------------------------
# Review sample
# ---------------------------------------------------------------------------
def select_sample(api, dataset: str, limit_bytes: float, max_files: int):
    """Page through the dataset's file list until the sample budget is reached."""
    picked, used, token = [], 0, None
    while True:
        resp = with_retry(lambda: api.dataset_list_files(dataset, page_token=token, page_size=100), "file listing")
        for f in resp.files or []:
            size = f.total_bytes or 0
            if not f.name.endswith((".csv", ".parquet")):
                continue   # skip anything that is not a data file
            if len(picked) >= max_files or (picked and used + size > limit_bytes):
                return picked, used   # the budget (number of files or total size) is used up
            picked.append(f.name)
            used += size
        token = resp.next_page_token
        if not token:
            return picked, used   # no more pages
        time.sleep(2)   # a short pause between pages, to stay polite to Kaggle


def download_file(api, dataset: str, name: str, fallback_dir=None) -> str:
    """Download one dataset file via kagglehub, falling back to the Kaggle API."""
    import kagglehub   # imported here: only needed when regenerating the data

    fallback_dir = fallback_dir or str(config.DOWNLOAD_DIR / "join_check")
    try:
        return with_retry(lambda: kagglehub.dataset_download(dataset, path=name), name)
    except Exception as e:
        if "429" in str(e):
            raise   # rate-limited: the fallback would be blocked too
    # kagglehub failed for another reason: try the plain Kaggle API instead
    os.makedirs(fallback_dir, exist_ok=True)
    with_retry(lambda: api.dataset_download_file(dataset, name, path=fallback_dir, quiet=True), name)
    path = os.path.join(fallback_dir, os.path.basename(name))
    if not os.path.exists(path) and os.path.exists(path + ".zip"):
        zipfile.ZipFile(path + ".zip").extractall(fallback_dir)   # the API may deliver the file zipped
    return path


def read_head(path: str, n: int = 5) -> pd.DataFrame:
    """The first n rows of a review file (parquet or CSV)."""
    return pd.read_parquet(path).head(n) if path.endswith(".parquet") else pd.read_csv(path, nrows=n)


def find_col(cols, candidates):
    """The first of the candidate names that exists among the columns (ignoring upper/lower case), or None."""
    lower = {c.lower(): c for c in cols}
    return next((lower[c] for c in candidates if c in lower), None)


def detect_schema(sample: pd.DataFrame) -> dict:
    """Find the review files' column names for the fields we need."""
    return {
        "app_id": find_col(sample.columns, ["app_id", "appid", "steam_appid", "game_id", "gameid"]),
        "text": find_col(sample.columns, ["review", "review_text", "text", "content"]),
        "created": find_col(sample.columns, ["timestamp_created", "created", "date", "timestamp", "review_date"]),
        "vote": find_col(sample.columns, ["voted_up", "recommended", "is_recommended", "is_positive"]),
    }


def app_id_from_filename(path: str):
    """The first number in a file's name, taken as the game's AppID (None if there is no number)."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def count_reviews(files, app_col) -> pd.DataFrame:
    """Number of reviews per game in the sampled files."""
    counts = Counter()
    for p in files:
        if app_col is None:   # one file per game: the App ID is in the file name
            n = (len(pd.read_parquet(p, columns=[])) if p.endswith(".parquet")
                 else sum(len(c) for c in pd.read_csv(p, usecols=[0], chunksize=2_000_000)))   # read in chunks: files can be huge
            counts[app_id_from_filename(p)] += n
        elif p.endswith(".parquet"):
            counts.update(pd.read_parquet(p, columns=[app_col])[app_col].dropna().astype("int64"))
        else:
            for chunk in pd.read_csv(p, usecols=[app_col], chunksize=2_000_000):
                counts.update(pd.to_numeric(chunk[app_col], errors="coerce").dropna().astype("int64"))
    out = pd.Series(counts, name="n_review_texts").rename_axis("AppID").reset_index()
    out["AppID"] = out["AppID"].astype("Int64")
    return out


# ---------------------------------------------------------------------------
# Join check
# ---------------------------------------------------------------------------
def join_summary(games: pd.DataFrame, rev_counts: pd.DataFrame) -> pd.DataFrame:
    """How well the review sample lines up with the game catalogue."""
    eligible = games[(games["Positive"] + games["Negative"]) >= config.MIN_REVIEWS]
    resolved = rev_counts["AppID"].isin(games["AppID"])      # review games that exist in the catalogue
    covered = eligible["AppID"].isin(rev_counts["AppID"])    # catalogue games that the sample covers
    return pd.DataFrame({
        "value": [len(games), len(eligible), len(rev_counts), resolved.sum(), covered.sum()],
        "share": [None, None, None, resolved.mean(), covered.mean()],
    }, index=["Games (metadata)", f"Games with ≥ {config.MIN_REVIEWS} reviews", "Games in review sample",
              "Review games resolved in metadata", "Eligible games covered by sample"])


def save_outputs(games: pd.DataFrame, rev_counts: pd.DataFrame, schema: dict):
    """Save games.parquet (the source file the pipeline starts from) and the join-check tables."""
    games_out = games.copy()
    for c in games_out.select_dtypes("object").columns:
        games_out[c] = games_out[c].astype("string")   # a text type that parquet stores reliably
    size_mb = save_games(games_out, config.source_file(config.GAMES_FILE))
    rev_counts.to_parquet(config.derived_file("review_counts.parquet"), index=False)
    pd.Series(schema, name="column").to_csv(config.derived_file("review_schema.csv"))
    status(f"Saved {config.GAMES_FILE} ({size_mb:.0f} MB) to {config.SOURCE_DIR}")


def main():
    """Run this step: download the catalogue and a review sample, check the join, save games.parquet."""
    api = connect()

    status("Downloading the Steam Games Dataset from Hugging Face ...")
    games = load_games(download_games_csv())
    check_alignment(games)
    print(f"{len(games):,} games × {games.shape[1]} columns")

    names, sample_bytes = select_sample(api, config.REVIEWS_DATASET, SAMPLE_GB * 1e9, MAX_FILES)
    status(f"Downloading {len(names)} review files to check the join ...")
    rev_files = []
    for name in names:
        rev_files.append(download_file(api, config.REVIEWS_DATASET, name))
        time.sleep(1)   # a short pause between downloads, to stay polite to Kaggle
    print(f"Review sample: {len(rev_files)} files, {sample_bytes / 1e9:.2f} GB")

    schema = detect_schema(read_head(rev_files[0]))
    print_table(pd.Series(schema, name="column"), "Review file columns")

    rev_counts = count_reviews(rev_files, schema["app_id"])
    print(f"{rev_counts['n_review_texts'].sum():,} reviews across {len(rev_counts):,} games")
    print_table(join_summary(games, rev_counts), "Join check",
                {"value": "{:,.0f}", "share": lambda v: "" if pd.isna(v) else f"{v:.1%}"})

    save_outputs(games, rev_counts, schema)


if __name__ == "__main__":
    main()
