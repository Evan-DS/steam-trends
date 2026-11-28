"""Step 1: filter the analysis population, build the three success measures, and explore them.

Input : data/source/games.parquet
Output: data/derived/games_clean.parquet, population_funnel.csv, target_correlations.csv;
        figures 02_*.png

The three success measures ("targets"):
    reception   = share of a game's user reviews that are positive
    reach       = how many people own the game (grouped into four tiers)
    engagement  = the game's median playtime in hours
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_success import config
from steam_success.data import read_source, without_unused_columns
from steam_success.plotting import COLOR, INK, INK_2, INK_3, SURFACE, hours, nice_ticks, percent, save_figure
from steam_success.printing import print_table
from steam_success.progress import status

TOP_GENRES = 12   # the genre chart shows the 12 genres with the most games
# Steam lists software (not games) under these genres; they are removed from the population
NON_GAME_GENRES = {"Utilities", "Design & Illustration", "Animation & Modeling", "Video Production",
                   "Audio Production", "Photo Editing", "Web Publishing", "Software Training",
                   "Accounting", "Game Development"}
# Price groups for the price chart: the cut points (in dollars) and the names of the groups they make
PRICE_BINS = [-0.01, 0, 4.99, 9.99, 19.99, 39.99, np.inf]
PRICE_LABELS = ["Free", "Under $5", "$5–9.99", "$10–19.99", "$20–39.99", "$40+"]
# Columns that must exist in the source file, otherwise parse_games stops with a clear message
REQUIRED_COLUMNS = ["AppID", "Name", "Release date", "Estimated owners", "Price", "Positive", "Negative",
                    "Median playtime forever", "Genres", "Categories"]


# ---------------------------------------------------------------------------
# Parsing and filtering
# ---------------------------------------------------------------------------
def split_list(value) -> list[str]:
    """Split a comma-separated Steam field into a clean list."""
    if pd.isna(value) or str(value).strip() == "":
        return []   # a missing or empty field means "no entries"
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw catalogue into proper numbers, dates and lists (nothing is dropped yet)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    assert not missing, f"Missing columns: {missing}"   # stop early if the file is not the expected one
    df = raw.copy()   # never change the caller's table
    numeric = ["Price", "Discount", "Positive", "Negative", "Median playtime forever", "Average playtime forever",
               "Peak CCU", "Achievements", "DLC count", "Required age"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")   # text that is not a number becomes missing (NaN)
    df["AppID"] = pd.to_numeric(df["AppID"], errors="coerce").astype("Int64")
    # Release dates come in several written formats ("Mar 5, 2020", "2020-03-05", ...), hence format="mixed"
    df["release_date"] = pd.to_datetime(df["Release date"].astype("string"), format="mixed", errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["release_month"] = df["release_date"].dt.month.astype("Int64")
    df["genre_list"] = df["Genres"].map(split_list)
    df["category_list"] = df["Categories"].map(split_list)
    df["n_reviews"] = df["Positive"].fillna(0) + df["Negative"].fillna(0)   # all user reviews, good and bad
    return df


def filter_population(games: pd.DataFrame):
    """Apply the four filters (Table I in the report). Returns the kept games and the step counts."""
    # `steps` records the table after each filter so the report can show how many games each one removed
    steps = [("All titles in metadata", games)]
    g = games.drop_duplicates("AppID")   # keep one row per game
    steps.append(("Unique App IDs", g))
    g = g[~g["genre_list"].map(lambda gs: bool(NON_GAME_GENRES.intersection(gs)))]   # drop software
    steps.append(("Games only (software genres removed)", g))
    g = g[g["n_reviews"] >= config.MIN_REVIEWS]   # enough reviews for a trustworthy % positive
    steps.append((f"At least {config.MIN_REVIEWS} user reviews", g))
    g = g[g["release_year"].between(config.YEAR_MIN, config.YEAR_MAX).fillna(False)]   # unknown years are dropped
    steps.append((f"Released {config.YEAR_MIN}–{config.YEAR_MAX}", g))

    counts = pd.Series({label: len(x) for label, x in steps}, name="Titles")
    funnel = pd.DataFrame({
        "Titles": counts.map("{:,}".format),
        # each count divided by the one before it = the share that survived this filter
        "Kept from previous step": (counts / counts.shift()).map(lambda v: "" if pd.isna(v) else f"{v:.1%}"),
    })
    return g.copy(), funnel


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Reception (share positive), reach (owner tier) and engagement (median playtime, hours)."""
    df = df.copy()
    # "Estimated owners" is text such as "20000 - 50000"; pull out the two numbers
    owners = df["Estimated owners"].astype("string").str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    df["owners_low"], df["owners_high"] = owners[0], owners[1]
    df["pct_positive"] = df["Positive"] / df["n_reviews"]   # reception
    # reach: cut the lower end of the owner range into the four tiers; reach_code is the tier as a number 0-3
    df["reach_tier"] = pd.cut(df["owners_low"], bins=config.REACH_BINS, labels=config.REACH_LABELS, ordered=True)
    df["reach_code"] = df["reach_tier"].cat.codes.where(df["reach_tier"].notna()).astype("Int64")
    df["high_reach"] = (df["reach_code"] >= config.HIGH_REACH).astype("boolean")
    # SteamSpy writes 0 when it has no data, so 0 means "missing"
    df["playtime_raw_h"] = df["Median playtime forever"].where(df["Median playtime forever"] > 0) / 60   # minutes -> hours
    df["median_playtime_h"] = df["playtime_raw_h"].where(df["playtime_raw_h"] <= config.PLAYTIME_MAX_H)   # engagement
    # Steam stores the discounted price; undo the discount to get the regular (list) price
    discount = df["Discount"].fillna(0) if "Discount" in df.columns else pd.Series(0, index=df.index)
    df["list_price"] = (df["Price"] / (1 - discount / 100)).where(discount < 100).round(2)
    df["price_bucket"] = pd.cut(df["list_price"], bins=PRICE_BINS, labels=PRICE_LABELS, ordered=True)
    return df


def describe(s: pd.Series, fmt) -> dict:
    """Availability, median and interquartile range (the middle half of the values) of one measure."""
    q = s.quantile([0.25, 0.5, 0.75])
    return {"Available": f"{s.notna().mean():.1%}", "Median": fmt(q[0.5]),
            "Interquartile range": f"{fmt(q[0.25])} – {fmt(q[0.75])}"}


def print_target_summaries(df: pd.DataFrame):
    """Print the summary tables for the three measures, and how complete the playtime data is."""
    print_table(pd.DataFrame({
        "Reception (% positive)": describe(df["pct_positive"], lambda v: f"{v:.0%}"),
        "Engagement (median hours)": describe(df["median_playtime_h"], lambda v: f"{v:.1f} h"),
    }).T, "Targets")

    reach = df["reach_tier"].value_counts().reindex(config.REACH_LABELS)
    print_table(pd.DataFrame({"Games": reach.map("{:,}".format),
                              "Share": (reach / reach.sum()).map("{:.1%}".format)}).rename_axis("Reach tier"),
                "Reach tiers")

    # How many games have playtime data, by tier and by year (missing playtime is not random)
    has_playtime = df["median_playtime_h"].notna()
    print_table(pd.DataFrame({
        "Games": df.groupby("reach_tier", observed=True).size().map("{:,}".format),
        "With playtime data": has_playtime.groupby(df["reach_tier"], observed=True).mean().map("{:.0%}".format),
    }).rename_axis("Reach tier"), "Playtime coverage by tier")

    by_year = has_playtime.groupby(df["release_year"]).mean().map("{:.0%}".format).to_frame("With playtime data").T
    by_year.columns = by_year.columns.astype(int)
    print_table(by_year, "Playtime coverage by release year")

    # Games above the playtime cap are treated as missing; show the longest ones as a sanity check
    long_play = df[df["playtime_raw_h"] > config.PLAYTIME_MAX_H]
    print(f"{len(long_play):,} games report a median playtime above {config.PLAYTIME_MAX_H} hours "
          "(treated as missing). Longest:")
    print_table(long_play.nlargest(10, "playtime_raw_h")
                [["Name", "release_year", "reach_tier", "n_reviews", "playtime_raw_h"]].round(1))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def median_marker(ax, value, label):
    """A dashed vertical line at the median, with a small text label at the top."""
    ax.axvline(value, color=INK, lw=1, ls=(0, (4, 3)))
    ax.annotate(label, xy=(value, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                textcoords="offset points", va="top", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))


def plot_target_distributions(df: pd.DataFrame):
    """Figure 02_target_distributions: one panel per measure (histogram, bars, histogram)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    # Reception: a histogram of the share of positive reviews
    ax = axes[0]
    ax.hist(df["pct_positive"].dropna(), bins=np.linspace(0, 1, 41), color=COLOR, edgecolor=SURFACE, linewidth=0.8)
    median_marker(ax, df["pct_positive"].median(), f"median {df['pct_positive'].median():.0%}")
    ax.xaxis.set_major_formatter(percent)
    ax.set(title="Reception", xlabel="Positive reviews", ylabel="Games")

    # Reach: the share of games in each tier
    ax = axes[1]
    share = df["reach_tier"].value_counts(normalize=True).reindex(config.REACH_LABELS)
    bars = ax.bar(range(len(share)), share.values, width=0.62, color=COLOR)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in share.values], padding=3, fontsize=8.5, color=INK)
    ax.set_xticks(range(len(share)), [lab.replace(" (", "\n(") for lab in config.REACH_LABELS])
    ax.yaxis.set_major_formatter(percent)
    ax.grid(axis="x", visible=False)
    ax.set(title="Reach", ylabel="Share of games")

    # Engagement: playtime is very skewed, so a log scale; the longest 0.5% are cut off for readability
    ax = axes[2]
    h = df["median_playtime_h"].dropna()
    cap = min(h.quantile(0.995), 1000)
    ax.hist(h[h <= cap], bins=np.logspace(np.log10(max(h.min(), 0.05)), np.log10(cap), 40),
            color=COLOR, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(hours)
    median_marker(ax, h.median(), f"median {h.median():.1f} h")
    ax.text(0.98, 0.72, f"{(h > cap).sum():,} games above\n{cap:,.0f} h not shown", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK_2,
            bbox=dict(boxstyle="square,pad=0.2", fc=SURFACE, ec="none"))
    ax.set(title="Engagement", xlabel="Median playtime (log scale)", ylabel="Games")
    fig.tight_layout()
    save_figure(fig, "02_target_distributions")


def plot_trends_by_year(df: pd.DataFrame):
    """Figure 02_trends_by_year: games released, and the typical value of each measure, per release year."""
    by_year = df.groupby("release_year").agg(
        games=("AppID", "size"), pct_positive=("pct_positive", "median"),
        high_reach=("high_reach", "mean"), playtime=("median_playtime_h", "median"))
    years = by_year.index.astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    axes[0].bar(years, by_year["games"], width=0.65, color=COLOR)
    axes[0].set(title="Games released")
    axes[0].grid(axis="x", visible=False)
    panels = [("pct_positive", "Median % positive", percent),
              ("high_reach", "Share with ≥ 100k owners", percent),
              ("playtime", "Median playtime (hours)", hours)]
    for ax, (col, title, fmt) in zip(axes[1:], panels):
        ax.plot(years, by_year[col], color=COLOR, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
        ax.set_ylim(bottom=0)
        nice_ticks(ax.yaxis)
        ax.yaxis.set_major_formatter(fmt)
        ax.set(title=title)
    for ax in axes:
        ax.set_xticks(years[::2])   # label every second year so the labels do not overlap
    fig.tight_layout()
    save_figure(fig, "02_trends_by_year")


def group_stats(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Number of games and the typical value of each measure for every group in column `key`."""
    return frame.groupby(key, observed=True).agg(
        n=("AppID", "size"), pct_positive=("pct_positive", "median"),
        high_reach=("high_reach", "mean"), playtime=("median_playtime_h", "median"))


def plot_dot_panels(df: pd.DataFrame, stats: pd.DataFrame, title: str, name: str):
    """Three aligned dot plots: one row per group, one panel per target."""
    # The dashed reference line in each panel is the value for all games together
    overall = {"pct_positive": df["pct_positive"].median(), "high_reach": df["high_reach"].mean(),
               "playtime": df["median_playtime_h"].median()}
    panels = [("pct_positive", "Median % positive", percent),
              ("high_reach", "Share with ≥ 100k owners", percent),
              ("playtime", "Median playtime", hours)]
    y = np.arange(len(stats))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.36 * len(stats) + 1.4), sharey=True)   # height grows with rows
    for ax, (col, label, fmt) in zip(axes, panels):
        ax.axvline(overall[col], color=INK_3, lw=1, ls=(0, (4, 3)))
        ax.plot(stats[col], y, "o", color=COLOR, ms=7, mec=SURFACE, mew=1.5)
        nice_ticks(ax.xaxis)
        ax.xaxis.set_major_formatter(fmt)
        ax.set_title(label)
        ax.grid(axis="y", visible=False)
        lo, hi = stats[col].min(), stats[col].max()
        pad = (hi - lo) * 0.15 or 0.05   # a little room around the dots (a fixed amount if they are all equal)
        ax.set_xlim(min(lo, overall[col]) - pad, max(hi, overall[col]) + pad)
    axes[0].set_yticks(y, [f"{k}  (n={v:,})" for k, v in stats["n"].items()])
    axes[0].invert_yaxis()   # first group at the top
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save_figure(fig, name)


def plot_by_genre_and_price(df: pd.DataFrame):
    """Figures 02_targets_by_genre and 02_targets_by_price."""
    # A game can have several genres, so it counts once in each of its genres (one row per game-genre pair)
    by_genre = df.explode("genre_list").dropna(subset=["genre_list"])
    top = by_genre["genre_list"].value_counts().head(TOP_GENRES).index   # the most common genres, largest first
    plot_dot_panels(df, group_stats(by_genre[by_genre["genre_list"].isin(top)], "genre_list").loc[top],
                    f"Targets by genre (top {TOP_GENRES} by number of games)", "02_targets_by_genre")
    plot_dot_panels(df, group_stats(df, "price_bucket"), "Targets by regular (list) price", "02_targets_by_price")


def target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman (rank) correlation between the three measures, rounded to two decimals."""
    corr = df[["pct_positive", "reach_code", "median_playtime_h"]].astype(float).corr(method="spearman")
    corr.index = corr.columns = ["Reception", "Reach", "Engagement"]
    return corr.round(2)


def main():
    """Run this step: read, filter, build the measures, draw the charts and save the cleaned games."""
    status(f"Reading {config.GAMES_FILE} ...")
    games = parse_games(read_source(config.GAMES_FILE))
    status(f"{len(games):,} titles; release date parsed for {games['release_date'].notna().mean():.1%}")

    df, funnel = filter_population(games)
    print_table(funnel, "Analysis population (Table I)")
    funnel.to_csv(config.derived_file("population_funnel.csv"))

    status("Building the three success measures ...")
    df = add_targets(df)
    print_target_summaries(df)

    status("Drawing exploration charts ...")
    plot_target_distributions(df)
    plot_trends_by_year(df)
    plot_by_genre_and_price(df)
    corr = target_correlations(df)
    print_table(corr, "Spearman correlations between the targets")
    corr.to_csv(config.derived_file("target_correlations.csv"))

    # Save the cleaned games for the later steps (without the unused text columns, to keep the file small)
    games_clean = without_unused_columns(df).reset_index(drop=True)
    games_clean.to_parquet(config.derived_file("games_clean.parquet"), index=False)
    status(f"Saved games_clean.parquet: {len(games_clean):,} games × {games_clean.shape[1]} columns")


if __name__ == "__main__":
    main()
