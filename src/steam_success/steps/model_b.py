"""Step 4: Model B asks whether the first 30 days of reviews improve the predictions.

A stratified sample of games gets eight early-review features. Model A (launch
features) and Model B (launch + early-review features) are trained and tested on
exactly the same games and the same 15 folds (5-fold CV repeated 3 times).

Inputs : data/source/review_features.parquet, data/derived/games_clean.parquet, launch_features.parquet
         (review_features is rebuilt from the Kaggle review files only with: python main.py --download)
Outputs: data/derived/model_b_comparison.csv, model_b_fold_scores.csv; figure 05_model_a_vs_b.png

"Stratified" means the sample takes the same number of games from every reach tier, so rare tiers
are not drowned out. "Paired folds" means both models are scored on exactly the same test games,
so any difference between them is not down to luck in how the data was split.
"""
import contextlib
import io
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from steam_success import config
from steam_success.data import (MissingDataError, feature_columns, load_games_clean, load_launch_features,
                                with_int_appid)
from steam_success.kaggle_client import RateLimited, connect, with_retry
from steam_success.modeling import CLASSIFICATION, REGRESSION, fit, score, xgb_model
from steam_success.plotting import COLOR_A, COLOR_B, INK, INK_2, save_figure
from steam_success.printing import print_table
from steam_success.progress import status

# The columns of a Kaggle review file that are used (the files hold many more)
REVIEW_COLUMNS = ["timestamp_created", "voted_up", "author_playtime_at_review", "written_during_early_access",
                  "received_for_free", "steam_purchase", "language", "author_num_games_owned",
                  "author_num_reviews"]
SAVE_EVERY = 50   # games between progress saves (the download can be resumed)


# ---------------------------------------------------------------------------
# Sample and review download
# ---------------------------------------------------------------------------
def sample_games(games: pd.DataFrame) -> pd.DataFrame:
    """The same number of random games from each reach tier."""
    # min(...) takes every game of a tier that has fewer games than the target size; the fixed seed makes
    # the sample identical on every run
    sample = pd.concat([g.sample(min(len(g), config.SAMPLE_PER_TIER), random_state=config.SEED)
                        for _, g in games.groupby("reach_code")])[["AppID", "Name", "release_date", "reach_code"]]
    return with_int_appid(sample)


def infer_template(names: list[str]) -> str:
    """Turn a listed name such as 'reviews/10.csv' into a pattern such as 'reviews/{}.csv'."""
    for n in names:
        digits = list(re.finditer(r"\d+", n))
        if not digits:
            continue
        d = digits[-1]   # the last number in the name is taken to be the game's AppID
        template = n[:d.start()] + "{}" + n[d.end():]
        # Accept the pattern only if at least 80% of the listed files follow it
        pattern = re.compile("^" + re.escape(template).replace(r"\{\}", r"\d+") + "$")
        if sum(bool(pattern.match(x)) for x in names) >= 0.8 * len(names):
            return template
    raise ValueError(f"Could not work out the file naming from: {names[:5]}")


class ReviewDownloader:
    """Downloads one game's review file at a time from the Kaggle reviews dataset."""

    def __init__(self):
        import kagglehub   # imported here: only needed (and only installed) when regenerating the data

        self._kagglehub = kagglehub
        api = connect()
        # Look at the first page of the file listing to learn how the review files are named
        first_page = with_retry(lambda: api.dataset_list_files(config.REVIEWS_DATASET, page_size=100),
                                "file listing")
        self.template = infer_template([f.name for f in first_page.files or []])
        print("Review files are named like:", self.template.format(570))

    def fetch(self, app_id: int):
        """Download one game's review file; return its local path, or None if the game has no file."""
        name = self.template.format(app_id)
        quiet = io.StringIO()   # kagglehub prints progress bars; send them nowhere
        try:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                return with_retry(lambda: self._kagglehub.dataset_download(config.REVIEWS_DATASET, path=name), name)
        except RateLimited:
            raise   # being rate-limited for good is a real problem: stop the run
        except Exception:
            return None   # most likely this game simply has no review file


# ---------------------------------------------------------------------------
# Early-review features
# ---------------------------------------------------------------------------
def as_bool(s: pd.Series) -> pd.Series:
    """Read a column of true/false written in various ways ('True', 'true', 1, '1.0') as real booleans."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def early_features(reviews: pd.DataFrame, release_date) -> dict:
    """Eight features from the reviews written in a game's first WINDOW_DAYS days.

    The window starts on the release date, or on the first review if that is earlier
    (Early Access games are reviewed before release). Only shares and medians are used:
    review counts would largely restate reach.
    """
    def col(c):
        """A column of the review table, or a column of blanks if the file does not have it."""
        return reviews[c] if c in reviews.columns else pd.Series(np.nan, index=reviews.index)

    t = pd.to_datetime(pd.to_numeric(col("timestamp_created"), errors="coerce"), unit="s")   # when each review was written
    if t.notna().sum() == 0:
        return {"n_early_reviews": 0}
    start = min(t.min(), release_date) if pd.notna(release_date) else t.min()
    w = (t >= start) & (t < start + pd.Timedelta(days=config.WINDOW_DAYS))   # True for reviews inside the window
    if w.sum() == 0:
        return {"n_early_reviews": 0}

    def num(c):
        """A column of the window's reviews as numbers."""
        return pd.to_numeric(col(c)[w], errors="coerce")

    return {
        "n_early_reviews": int(w.sum()),   # used only to select games, never as a feature
        "early_pct_positive": as_bool(col("voted_up")[w]).mean(),
        config.PLAYTIME_AT_REVIEW: num("author_playtime_at_review").median() / 60,   # minutes -> hours
        "early_pct_early_access": as_bool(col("written_during_early_access")[w]).mean(),
        "early_pct_received_free": as_bool(col("received_for_free")[w]).mean(),
        "early_pct_steam_purchase": as_bool(col("steam_purchase")[w]).mean(),
        "early_pct_english": (col("language")[w].astype(str).str.lower() == "english").mean(),
        "early_median_reviewer_games": num("author_num_games_owned").median(),
        "early_median_reviewer_reviews": num("author_num_reviews").median(),
    }


def read_reviews(path: str) -> pd.DataFrame:
    """Read a downloaded review file (parquet or CSV), keeping only the columns that are used."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, usecols=lambda c: c in REVIEW_COLUMNS, low_memory=False)


def collect_review_features(sample: pd.DataFrame, allow_download: bool = False) -> pd.DataFrame:
    """Early-review features for the sampled games.

    Normally they are read from data/source/review_features.parquet. With allow_download=True,
    games missing from that file are downloaded from Kaggle (resuming where it stopped).
    """
    out = config.source_file(config.REVIEW_FEATURES_FILE)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame({"AppID": pd.Series(dtype=int)})
    todo = sample[~sample["AppID"].isin(done["AppID"])]   # sampled games that are not in the file yet
    if todo.empty:
        status(f"Early-review features found for all {len(sample):,} sampled games in {out.name}")
        return done
    if not allow_download:
        raise MissingDataError(
            f"{out} is missing or covers only {len(sample) - len(todo):,} of the {len(sample):,} sampled games. "
            f"Put the project's {config.REVIEW_FEATURES_FILE} in {config.SOURCE_DIR}, "
            "or regenerate it with: python main.py --download")
    status(f"{len(done):,} games already done · {len(todo):,} to download from Kaggle")

    downloader = ReviewDownloader()
    rows, t0 = [], time.time()
    for i, (app_id, released) in enumerate(zip(todo["AppID"], todo["release_date"]), 1):
        path = downloader.fetch(app_id)
        row = {"AppID": app_id, "has_file": path is not None}
        if path:
            try:
                row.update(early_features(read_reviews(path), released))
            except Exception as e:  # one bad file should not stop the whole download
                row["error"] = str(e)[:120]
            finally:
                with contextlib.suppress(OSError):
                    os.remove(path)   # free disk space
        rows.append(row)
        if i % SAVE_EVERY == 0 or i == len(todo):
            # Save progress every SAVE_EVERY games, so a stopped run can be resumed instead of restarted
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(out, index=False)
            rows = []
            minutes_left = (time.time() - t0) / i * (len(todo) - i) / 60
            status(f"  {len(done):,}/{len(sample):,} games saved · about {minutes_left:.0f} min left")
        time.sleep(0.5)   # a short pause between downloads, to stay polite to Kaggle
    return done


# ---------------------------------------------------------------------------
# Model A vs Model B
# ---------------------------------------------------------------------------
def compare(X_a, X_b, y, task: str) -> pd.DataFrame:
    """Score Model A and Model B on the same 15 folds; one row per fold."""
    if task == CLASSIFICATION:
        splits = RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                                         random_state=config.SEED).split(X_a, y)
    else:
        splits = RepeatedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                               random_state=config.SEED).split(X_a)
    rows = []
    n_folds = config.N_SPLITS * config.N_REPEATS
    for fold, (train, test) in enumerate(splits):
        status(f"  fold {fold + 1}/{n_folds}")
        # Both models are trained and tested on exactly the same rows of this fold
        for name, X in (("Model A", X_a), ("Model B", X_b)):
            model = fit(xgb_model(task, **config.XGB_PARAMS), X.iloc[train], y.iloc[train], task, balanced=True)
            rows.append({"fold": fold, "model": name, "score": score(task, y.iloc[test], model.predict(X.iloc[test]))})
    return pd.DataFrame(rows).pivot(index="fold", columns="model", values="score")   # one row per fold, one column per model


def summarise(folds: pd.DataFrame, measure: str, score_name: str, n: int) -> dict:
    """One row of Table IV: mean ± standard deviation over the folds, and how often Model B won."""
    gap = folds["Model B"] - folds["Model A"]   # per fold; positive = Model B scored higher
    return {"Measure": measure, "Score": score_name, "Games": n,
            "Model A": f"{folds['Model A'].mean():.3f} ± {folds['Model A'].std():.3f}",
            "Model B": f"{folds['Model B'].mean():.3f} ± {folds['Model B'].std():.3f}",
            "Gap (B − A)": f"{gap.mean():+.3f} ± {gap.std():.3f}",
            "B better in": f"{(gap > 0).sum()} of {len(gap)} folds"}


def plot_comparison(reach_folds: pd.DataFrame, eng_folds: pd.DataFrame):
    """Figure 05_model_a_vs_b: each model's mean ± SD, with the single-fold scores as small dots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, (title, folds) in zip(axes, [("Reach: macro F1", reach_folds), ("Engagement: R²", eng_folds)]):
        for i, (name, color) in enumerate([("Model A", COLOR_A), ("Model B", COLOR_B)]):
            s = folds[name]
            # the small dots: one per fold, nudged up/down by a fixed random amount so they do not overlap
            ax.scatter(s, np.full(len(s), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(s)),
                       s=14, color=color, alpha=0.45, linewidths=0)
            # the big dot with an error bar: the mean and its standard deviation
            ax.errorbar(s.mean(), i, xerr=s.std(), fmt="o", color=color, ms=9, capsize=4, lw=1.5,
                        mec="white", mew=1.5, zorder=3)
            ax.annotate(f"{s.mean():.3f}", (s.mean(), i), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.set_yticks([0, 1], ["Model A\n(launch only)", "Model B\n(+ early reviews)"])
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="y", visible=False)
        gap = (folds["Model B"] - folds["Model A"]).mean()
        ax.set_title(f"{title}   (gap {gap:+.3f})")
    fig.suptitle("Model A vs Model B on the same games: mean ± SD over 15 folds; small dots are single folds",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save_figure(fig, "05_model_a_vs_b")


def main(allow_download: bool = False):
    """Run this step: get the early-review features, compare Model A and Model B, save table and figure."""
    games = load_games_clean()
    sample = sample_games(games)
    status(f"Model B sample: {len(sample):,} games ({config.SAMPLE_PER_TIER} per reach tier)")
    print_table(sample["reach_code"].map(dict(enumerate(config.REACH_SHORT))).value_counts()
                .reindex(config.REACH_SHORT).to_frame("games"), "Sample per reach tier")

    rf = collect_review_features(sample, allow_download)
    status(f"Review file found for {rf['has_file'].mean():.0%} of sampled games; "
          f"{(rf['n_early_reviews'] >= config.MIN_EARLY_REVIEWS).sum():,} have at least "
          f"{config.MIN_EARLY_REVIEWS} reviews in the first {config.WINDOW_DAYS} days")

    # Join the success measures, the launch features and the early-review features into one table,
    # keeping only games with enough early reviews
    launch = load_launch_features()
    features_a = feature_columns(launch)
    data = (with_int_appid(games[["AppID", "reach_code", "median_playtime_h"]])
            .merge(with_int_appid(launch), on="AppID")
            .merge(rf, on="AppID"))
    data = data[data["n_early_reviews"] >= config.MIN_EARLY_REVIEWS].reset_index(drop=True)
    print_table(data.groupby(data["reach_code"].map(dict(enumerate(config.REACH_SHORT))))[config.REVIEW_FEATURES]
                .median().reindex(config.REACH_SHORT).T.round(2), "Early-review features: median per tier")

    # Reach: Model A uses the launch features; Model B adds the eight early-review features
    y_reach = data["reach_code"].astype(int)
    status(f"Reach: Model A vs Model B on {len(data):,} games, 15 paired folds ...")
    reach_folds = compare(data[features_a].astype(float),
                          data[features_a + config.REVIEW_FEATURES].astype(float), y_reach, CLASSIFICATION)

    # Engagement: only games with playtime and ≥ 20k owners; Model B leaves out playtime-at-review
    # (it would be circular, because engagement IS playtime)
    eng = data[(data["reach_code"] >= config.ENGAGEMENT_MIN_TIER) & data["median_playtime_h"].notna()]
    eng = eng.reset_index(drop=True)
    status(f"Engagement: Model A vs Model B on {len(eng):,} games, 15 paired folds ...")
    eng_folds = compare(eng[features_a].astype(float),
                        eng[features_a + config.REVIEW_FEATURES_ENGAGEMENT].astype(float),
                        np.log(eng["median_playtime_h"]), REGRESSION)

    comparison = pd.DataFrame([summarise(reach_folds, "Reach", "Macro F1", len(data)),
                               summarise(eng_folds, "Engagement (≥ 20k owners)", "R²", len(eng))]).set_index("Measure")
    print_table(comparison, "Model A vs Model B (Table IV)")
    plot_comparison(reach_folds, eng_folds)

    comparison.to_csv(config.derived_file("model_b_comparison.csv"))
    pd.concat({"Reach": reach_folds, "Engagement": eng_folds}, names=["Measure"]).to_csv(
        config.derived_file("model_b_fold_scores.csv"))
    status("Saved Model B comparison")


if __name__ == "__main__":
    main()
