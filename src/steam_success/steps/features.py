"""Step 2: build the 42 launch features, meaning what a developer sets at or before launch.

Inputs : data/derived/games_clean.parquet, data/source/games.parquet (for publisher history)
Output : data/derived/launch_features.parquet, figure 03_feature_reach_gap.png

A "feature" is one input the models use to make a prediction (price, genre, number of languages ...).
Most are yes/no flags (1 = the game has it, 0 = it does not).
"""
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_success import config
from steam_success.data import load_games_clean, read_source
from steam_success.plotting import COLOR, INK, NEG, save_figure, signed_pp
from steam_success.printing import print_table
from steam_success.progress import status

# The twelve genres that get their own yes/no feature
GENRES = ["Indie", "Action", "Adventure", "Casual", "Simulation", "Strategy", "RPG", "Sports",
          "Racing", "Massively Multiplayer", "Early Access", "Free To Play"]
MATURE_GENRES = {"Violent", "Gore", "Nudity", "Sexual Content"}   # any of these = "mature content"
# Steam store categories collapsed into 15 flags: each flag is 1 if the game has ANY of the listed categories
CATEGORY_GROUPS = {
    "single_player":      {"Single-player"},
    "multiplayer":        {"Multi-player", "MMO", "PvP", "Online PvP", "Co-op", "Online Co-op", "Shared/Split Screen",
                           "Shared/Split Screen PvP", "Shared/Split Screen Co-op", "LAN PvP", "LAN Co-op",
                           "Cross-Platform Multiplayer"},
    "online_multiplayer": {"Online PvP", "Online Co-op", "MMO", "Cross-Platform Multiplayer"},
    "local_multiplayer":  {"Shared/Split Screen", "Shared/Split Screen PvP", "Shared/Split Screen Co-op",
                           "LAN PvP", "LAN Co-op"},
    "coop":               {"Co-op", "Online Co-op", "Shared/Split Screen Co-op", "LAN Co-op"},
    "pvp":                {"PvP", "Online PvP", "Shared/Split Screen PvP", "LAN PvP"},
    "steam_achievements": {"Steam Achievements"},
    "steam_cloud":        {"Steam Cloud"},
    "leaderboards":       {"Steam Leaderboards"},
    "full_controller":    {"Full controller support"},
    "partial_controller": {"Partial Controller Support"},
    "vr_only":            {"VR Only"},
    "vr_supported":       {"VR Supported", "VR Support", "Tracked Controller Support"},
    "level_editor":       {"Includes level editor"},
    "in_app_purchases":   {"In-App Purchases"},
}


def parse_list(value) -> list[str]:
    """Parse a Steam list field such as "['English', 'French']" into a Python list."""
    if pd.isna(value) or str(value).strip() in ("", "[]"):
        return []
    try:
        out = ast.literal_eval(str(value))   # safely read text that looks like a Python list
        return [str(x).strip() for x in out] if isinstance(out, (list, tuple)) else []
    except (ValueError, SyntaxError):
        # The text was not a clean list: fall back to splitting on commas and stripping brackets and quotes
        return [x.strip(" '\"[]") for x in str(value).split(",") if x.strip(" '\"[]")]


def first_name(value):
    """First developer or publisher in a comma-separated field, normalised for matching."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    return str(value).split(",")[0].strip().lower()   # lower case, so "Valve" and "valve" match


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Price, timing, genres, store categories, platforms, localisation, content and publisher features."""
    f = pd.DataFrame({"AppID": games["AppID"]})

    # Price: the regular (list) price with any discount undone. log1p = log(1 + price), which stops a few
    # very expensive games from dominating while keeping free games at exactly 0.
    discount = games["Discount"].fillna(0)
    f["list_price"] = (games["Price"] / (1 - discount / 100)).where(discount < 100).round(2)
    f["log_price"] = np.log1p(f["list_price"])
    f["is_free"] = (f["list_price"] == 0).astype(int)

    # Release timing: year, month, and whether it came out in the Q4 holiday season (Oct-Dec)
    f["release_year"] = games["release_year"].astype(int)
    f["release_month"] = games["release_month"].astype(int)
    f["release_q4"] = f["release_month"].isin([10, 11, 12]).astype(int)

    # One yes/no column per genre (g=g fixes the genre inside the lambda for each loop turn)
    genre_sets = games["genre_list"].map(set)
    for g in GENRES:
        f["genre_" + g.lower().replace(" ", "_")] = genre_sets.map(lambda s, g=g: int(g in s))
    f["mature_content"] = genre_sets.map(lambda s: int(bool(s & MATURE_GENRES)))   # s & MATURE = any in common
    f["mature_rating"] = (games["Required age"].fillna(0) >= 17).astype(int)

    # One yes/no column per store-category group
    category_sets = games["category_list"].map(set)
    for name, members in CATEGORY_GROUPS.items():
        f[name] = category_sets.map(lambda s, m=members: int(bool(s & m)))

    # Platforms and localisation: how many languages the text and the voice-over cover
    f["mac"] = games["Mac"].astype(int)
    f["linux"] = games["Linux"].astype(int)
    f["n_languages"] = games["Supported languages"].map(lambda v: len(parse_list(v)))
    f["n_audio_languages"] = games["Full audio languages"].map(lambda v: len(parse_list(v)))
    f["log_achievements"] = np.log1p(games["Achievements"].fillna(0).clip(lower=0))

    # Publisher: does the game have one, and is it the developer itself?
    dev = games["Developers"].map(first_name)
    pub = games["Publishers"].map(first_name)
    f["has_publisher"] = pub.notna().astype(int)
    f["self_published"] = (dev.notna() & (dev == pub)).astype(int)
    return f


def add_publisher_history(f: pd.DataFrame, games: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Number of games the publisher released BEFORE each game, counted over the full catalogue."""
    cat = catalog.assign(pub=catalog["Publishers"].map(first_name),
                         date=pd.to_datetime(catalog["Release date"].astype("string"), format="mixed", errors="coerce"))
    cat = cat.dropna(subset=["pub", "date"])
    # For every publisher, a sorted list of all its release dates
    release_dates = {p: np.sort(d.values) for p, d in cat.groupby("pub")["date"]}

    def prior_releases(publisher, date):
        """How many of the publisher's releases came strictly before `date`."""
        dates = release_dates.get(publisher)
        if dates is None or pd.isna(date):
            return 0
        # searchsorted on a sorted list = how many dates are earlier than this one
        return int(np.searchsorted(dates, np.datetime64(date), side="left"))

    pub = games["Publishers"].map(first_name)
    f = f.copy()
    f["publisher_prior_games"] = [prior_releases(p, d) for p, d in zip(pub, games["release_date"])]
    f["log_publisher_prior"] = np.log1p(f["publisher_prior_games"])
    return f


def feature_summary(f: pd.DataFrame) -> pd.DataFrame:
    """One row per column: yes/no or numeric, how common it is (or its median), and how much is missing."""
    cols = [c for c in f.columns if c != "AppID"]
    binary = [c for c in cols if set(f[c].dropna().unique()) <= {0, 1}]   # only 0s and 1s = a yes/no flag
    return pd.DataFrame({
        "type": ["binary" if c in binary else "numeric" for c in cols],
        "share = 1 / median": [f"{f[c].mean():.1%}" if c in binary else f"{f[c].median():g}" for c in cols],
        "missing": [f"{f[c].isna().mean():.1%}" for c in cols],
    }, index=cols)


def plot_reach_gap(f: pd.DataFrame, games: pd.DataFrame):
    """For each yes/no feature: share of high-reach games with it minus without it."""
    cols = [c for c in f.columns if c != "AppID"]
    binary = [c for c in cols if set(f[c].dropna().unique()) <= {0, 1}]
    hi = games["high_reach"].astype(float).values   # 1.0 = at least 100k owners
    # Only features that are neither nearly universal nor nearly absent (1%-99%) give a meaningful comparison
    diff = pd.Series({c: hi[f[c].values == 1].mean() - hi[f[c].values == 0].mean()
                      for c in binary if 0.01 < f[c].mean() < 0.99}).sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(diff) + 1.2))   # height grows with the number of bars
    ax.barh(diff.index, diff.values * 100, color=[COLOR if v > 0 else NEG for v in diff.values], height=0.65)
    ax.axvline(0, color=INK, lw=1)
    ax.xaxis.set_major_formatter(signed_pp)
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of games with ≥ 100k owners: with the feature minus without it")
    fig.tight_layout()
    save_figure(fig, "03_feature_reach_gap")


def main():
    """Run this step: build the features, add publisher history, draw the chart and save the table."""
    games = load_games_clean()
    # Publisher history needs the WHOLE catalogue (not only the analysed games), but only three columns of it
    catalog = read_source(config.GAMES_FILE, columns=["AppID", "Release date", "Publishers"])
    status(f"Analysis games: {len(games):,} · catalogue for publisher history: {len(catalog):,}")

    status("Building launch features ...")
    f = build_features(games)
    status("Counting each publisher's earlier releases ...")
    f = add_publisher_history(f, games, catalog)
    print_table(f["publisher_prior_games"].describe().round(1).to_frame("prior releases").T, "Publisher history")
    summary = feature_summary(f)
    # Note: this count is 44 because the table also holds list_price and publisher_prior_games, the raw
    # versions of log_price and log_publisher_prior. They are not model inputs: the models use 42 features.
    print_table(summary, f"{len(summary)} launch features for {len(f):,} games")
    plot_reach_gap(f, games)

    f.to_parquet(config.derived_file("launch_features.parquet"), index=False)
    status(f"Saved launch_features.parquet: {len(f):,} games × {len(summary)} features")


if __name__ == "__main__":
    main()
