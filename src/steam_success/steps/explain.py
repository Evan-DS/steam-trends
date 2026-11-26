"""Step 6 — What drives the predictions (SHAP), and the high-reach, low-engagement group.

Inputs : data/derived/games_clean.parquet, launch_features.parquet; data/source/review_features.parquet
Outputs: data/derived/shap_importance.csv, high_reach_low_engagement.csv, high_reach_low_engagement_summary.csv;
         figures 06_*.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap

from steam_success import config
from steam_success.data import (feature_columns, load_games_clean, load_launch_features, load_review_features,
                                with_int_appid)
from steam_success.modeling import CLASSIFICATION, REGRESSION, fit, xgb_model
from steam_success.plotting import COLOR, COLOR_REVIEW, INK, INK_2, save_figure, signed_pp
from steam_success.printing import print_table
from steam_success.progress import status

TOP_N = 12                   # features shown per chart
PLOT_POINTS = 3000           # games drawn per beeswarm
LOW_ENGAGEMENT_QUANTILE = 0.25
VALUE_CMAP = LinearSegmentedColormap.from_list("low_high", ["#2a78d6", "#d6d4ce", "#e34948"])

LABELS = {
    "log_price": "Price (log)", "is_free": "Free to play (price $0)", "release_year": "Release year",
    "release_month": "Release month", "release_q4": "Released Oct–Dec", "mature_content": "Mature content",
    "mature_rating": "Age rating 17+", "single_player": "Single-player", "multiplayer": "Multiplayer",
    "online_multiplayer": "Online multiplayer", "local_multiplayer": "Local multiplayer", "coop": "Co-op",
    "pvp": "PvP", "steam_achievements": "Steam Achievements", "steam_cloud": "Steam Cloud",
    "leaderboards": "Leaderboards", "full_controller": "Full controller support",
    "partial_controller": "Partial controller support", "vr_only": "VR only", "vr_supported": "VR supported",
    "level_editor": "Level editor", "in_app_purchases": "In-app purchases", "mac": "Mac version",
    "linux": "Linux version", "n_languages": "Languages supported", "n_audio_languages": "Languages with full audio",
    "log_achievements": "Number of achievements (log)", "has_publisher": "Has a publisher",
    "self_published": "Self-published", "log_publisher_prior": "Publisher's earlier games (log)",
    "early_pct_positive": "Early reviews: % positive",
    "early_median_playtime_at_review_h": "Early reviews: playtime at review",
    "early_pct_early_access": "Early reviews: % in Early Access",
    "early_pct_received_free": "Early reviews: % free copies",
    "early_pct_steam_purchase": "Early reviews: % bought on Steam", "early_pct_english": "Early reviews: % English",
    "early_median_reviewer_games": "Early reviewers: games owned",
    "early_median_reviewer_reviews": "Early reviewers: reviews written",
    "genre_rpg": "Genre: RPG", "genre_free_to_play": "Genre: Free to Play",
}


def label(column: str) -> str:
    if column in LABELS:
        return LABELS[column]
    if column.startswith("genre_"):
        return "Genre: " + column[6:].replace("_", " ").title()
    return column


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_values(model, X) -> np.ndarray:
    """Exact TreeSHAP from XGBoost: (games × features), or (games × classes × features). Bias column dropped."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contrib[..., :-1]


def direction(sv: np.ndarray, X: pd.DataFrame) -> pd.Series:
    """How a feature's value moves the prediction: rank correlation between value and SHAP value."""
    out = {}
    for j, c in enumerate(X.columns):
        v = X[c].astype(float)
        out[c] = np.nan if v.nunique() < 2 else pd.Series(v.values).corr(pd.Series(sv[:, j]), method="spearman")
    return pd.Series(out)


def importance_table(sv: np.ndarray, X: pd.DataFrame, target: str) -> pd.DataFrame:
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns)
    t = pd.DataFrame({"target": target, "feature": X.columns, "label": [label(c) for c in X.columns],
                      "mean_abs_shap": imp.values, "direction": direction(sv, X).values})
    return t.sort_values("mean_abs_shap", ascending=False)


def beeswarm(sv, X, title, unit, ax, n=TOP_N):
    """SHAP summary plot: one row per feature, one dot per game, colour = the game's feature value."""
    rng = np.random.default_rng(config.SEED)
    keep = rng.choice(len(X), size=min(PLOT_POINTS, len(X)), replace=False)
    order = np.argsort(np.abs(sv).mean(0))[::-1][:n][::-1]
    for row, j in enumerate(order):
        v = X.iloc[keep, j].astype(float).values
        colour = v if set(np.unique(v[~np.isnan(v)])) <= {0, 1} else pd.Series(v).rank(pct=True).values
        y = row + rng.uniform(-0.32, 0.32, len(keep))
        ax.scatter(sv[keep, j], y, c=colour, cmap=VALUE_CMAP, vmin=0, vmax=1, s=5, alpha=0.6, linewidths=0,
                   rasterized=True)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(order)), [label(X.columns[j]) for j in order])
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=f"SHAP value: effect on the prediction ({unit})")


def value_legend(fig):
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=VALUE_CMAP), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["low / no", "high / yes"], fontsize=8.5)
    cb.outline.set_visible(False)
    cb.set_label("Feature value", fontsize=8.5, color=INK_2)


def engagement_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["reach_code"] >= config.ENGAGEMENT_MIN_TIER) & df["median_playtime_h"].notna()].reset_index(drop=True)


def explain_model_a(data: pd.DataFrame, features: list[str]) -> list[pd.DataFrame]:
    tables = []
    X = data[features].astype(float)

    rec_model = fit(xgb_model(REGRESSION), X, data["pct_positive"], REGRESSION)
    sv_rec = shap_values(rec_model, X) * 100   # percentage points
    tables.append(importance_table(sv_rec, X, "Reception"))

    eng = engagement_rows(data)
    X_eng = eng[features].astype(float)
    eng_model = fit(xgb_model(REGRESSION), X_eng, np.log(eng["median_playtime_h"]), REGRESSION)
    sv_eng = shap_values(eng_model, X_eng)
    tables.append(importance_table(sv_eng, X_eng, "Engagement"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    beeswarm(sv_rec, X, "Reception: what moves % positive", "percentage points", axes[0])
    beeswarm(sv_eng, X_eng, "Engagement: what moves median playtime", "log hours", axes[1])
    fig.subplots_adjust(left=0.17, right=0.9, wspace=0.75)
    value_legend(fig)
    save_figure(fig, "06_shap_reception_engagement")

    y_reach = data["reach_code"].astype(int)
    reach_model = fit(xgb_model(CLASSIFICATION), X, y_reach, CLASSIFICATION, balanced=True)
    sv_reach = shap_values(reach_model, X)       # games × 4 tiers × features
    sv_reach_mean = np.abs(sv_reach).mean(1)      # overall importance per game and feature
    reach_table = importance_table(sv_reach[:, 3, :], X, "Reach (Very high tier)")
    reach_table["mean_abs_shap"] = (pd.Series(sv_reach_mean.mean(0), index=X.columns)
                                    .reindex(reach_table["feature"]).values)
    tables.append(reach_table.sort_values("mean_abs_shap", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    beeswarm(sv_reach[:, 3, :], X, "Reach: what pushes a game toward 500k+ owners", "log-odds", axes[0])
    top = pd.Series(sv_reach_mean.mean(0), index=X.columns).sort_values().tail(TOP_N)
    axes[1].barh([label(c) for c in top.index], top.values, color=COLOR, height=0.65)
    axes[1].grid(axis="y", visible=False)
    axes[1].set(title="Reach: overall importance (all tiers)", xlabel="mean |SHAP value| (log-odds)")
    fig.subplots_adjust(left=0.17, right=0.9, wspace=0.75)
    value_legend(fig)
    save_figure(fig, "06_shap_reach")
    return tables


def explain_model_b(data: pd.DataFrame, features: list[str]) -> list[pd.DataFrame]:
    """Share of Model B's importance that comes from the early-review features."""
    rf = with_int_appid(load_review_features())
    b = data.merge(rf, on="AppID")
    b = b[b["n_early_reviews"] >= config.MIN_EARLY_REVIEWS].reset_index(drop=True)

    X_b = b[features + config.REVIEW_FEATURES].astype(float)
    y_b = b["reach_code"].astype(int)
    model = fit(xgb_model(CLASSIFICATION), X_b, y_b, CLASSIFICATION, balanced=True)
    imp_reach = pd.Series(np.abs(shap_values(model, X_b)).mean((0, 1)), index=X_b.columns)

    be = engagement_rows(b)
    X_be = be[features + config.REVIEW_FEATURES_ENGAGEMENT].astype(float)
    model = fit(xgb_model(REGRESSION), X_be, np.log(be["median_playtime_h"]), REGRESSION)
    imp_eng = pd.Series(np.abs(shap_values(model, X_be)).mean(0), index=X_be.columns)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (name, imp, n) in zip(axes, [("Reach", imp_reach, len(X_b)), ("Engagement", imp_eng, len(X_be))]):
        share = imp[[c for c in imp.index if c.startswith("early_")]].sum() / imp.sum()
        top = imp.sort_values().tail(TOP_N)
        ax.barh([label(c) for c in top.index], top.values, height=0.65,
                color=[COLOR_REVIEW if c.startswith("early_") else COLOR for c in top.index])
        ax.grid(axis="y", visible=False)
        ax.set(title=f"Model B, {name}: early reviews = {share:.0%} of total importance", xlabel="mean |SHAP value|")
        status(f"{name}: {n:,} games; early-review features carry {share:.0%} of the importance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR), plt.Rectangle((0, 0), 1, 1, color=COLOR_REVIEW)]
    fig.legend(handles, ["Launch feature", "Early-review feature"], loc="upper right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, wspace=0.75)
    save_figure(fig, "06_shap_model_b")

    return [pd.DataFrame({"target": name, "feature": imp.index, "label": [label(c) for c in imp.index],
                          "mean_abs_shap": imp.values, "direction": np.nan}).sort_values("mean_abs_shap",
                                                                                         ascending=False)
            for name, imp in [("Model B Reach", imp_reach), ("Model B Engagement", imp_eng)]]


# ---------------------------------------------------------------------------
# High reach, low engagement
# ---------------------------------------------------------------------------
def high_reach_low_engagement(data: pd.DataFrame, features: list[str]):
    """Games with ≥ 100k owners whose median playtime is in the bottom quarter, against the rest."""
    hr = data[(data["reach_code"] >= config.HIGH_REACH) & data["median_playtime_h"].notna()].copy()
    cut = hr["median_playtime_h"].quantile(LOW_ENGAGEMENT_QUANTILE)
    hr["low_engagement"] = hr["median_playtime_h"] <= cut
    status(f"{len(hr):,} high-reach games with playtime data; "
          f"{hr['low_engagement'].sum():,} have median playtime ≤ {cut:.1f} h")

    binary = [c for c in features if set(hr[c].dropna().unique()) <= {0, 1} and hr[c].mean() >= 0.02]
    low, rest = hr[hr["low_engagement"]], hr[~hr["low_engagement"]]
    comp = pd.DataFrame({"low-engagement group": low[binary].mean(), "other high-reach games": rest[binary].mean()})
    comp["difference (pp)"] = (comp.iloc[:, 0] - comp.iloc[:, 1]) * 100
    comp.index = [label(c) for c in comp.index]
    comp = comp.sort_values("difference (pp)")

    fig, ax = plt.subplots(figsize=(9, 0.26 * len(comp) + 1.2))
    ax.barh(comp.index, comp["difference (pp)"], height=0.65,
            color=[COLOR_REVIEW if v > 0 else COLOR for v in comp["difference (pp)"]])
    ax.axvline(0, color=INK, lw=1)
    ax.xaxis.set_major_formatter(signed_pp)
    ax.grid(axis="y", visible=False)
    ax.set_title("High-reach, low-engagement games vs other high-reach games: share with each feature")
    fig.tight_layout()
    save_figure(fig, "06_high_reach_low_engagement")

    medians = pd.DataFrame({"low-engagement group": low[["log_price", "n_languages", "release_year"]].median(),
                            "other high-reach games": rest[["log_price", "n_languages", "release_year"]].median()})
    medians.loc["log_price"] = np.expm1(medians.loc["log_price"]).round(2)
    medians.index = ["Regular price ($)", "Languages supported", "Release year"]
    print_table(medians, "Medians: low-engagement group vs other high-reach games")
    summary = pd.DataFrame({"low-engagement group": [len(low), low["median_playtime_h"].median()],
                            "other high-reach games": [len(rest), rest["median_playtime_h"].median()]},
                           index=["Games", "Median playtime (h)"])
    summary = pd.concat([summary, medians, comp[["low-engagement group", "other high-reach games"]]
                         .loc[[c for c in ["Free to play (price $0)", "Genre: Free to Play"] if c in comp.index]]])
    summary.to_csv(config.derived_file("high_reach_low_engagement_summary.csv"))

    examples = (low.sort_values(["owners_low", "median_playtime_h"], ascending=[False, True])
                [["Name", "owners_low", "owners_high", "median_playtime_h"]].head(15)
                .assign(owners=lambda d: d["owners_low"].map("{:,.0f}".format) + " – "
                        + d["owners_high"].map("{:,.0f}".format))
                [["Name", "owners", "median_playtime_h"]].round(1))
    print_table(examples, "Examples")
    return low


def main():
    games = with_int_appid(load_games_clean())
    launch = with_int_appid(load_launch_features())
    features = feature_columns(launch)
    data = games[["AppID", "Name", "pct_positive", "reach_code", "median_playtime_h",
                  "owners_low", "owners_high"]].merge(launch, on="AppID")
    status(f"{len(data):,} games × {len(features)} launch features")

    status("SHAP values for Model A (reception, engagement, reach) ...")
    tables = explain_model_a(data, features)
    for target, g in pd.concat(tables).groupby("target", sort=False):
        print_table(g.head(8)[["label", "mean_abs_shap", "direction"]].round(3).set_index("label"),
                    f"Top features: {target}")
    status("SHAP values for Model B ...")
    tables += explain_model_b(data, features)

    status("Profiling high-reach, low-engagement games ...")
    low = high_reach_low_engagement(data, features)

    pd.concat(tables).to_csv(config.derived_file("shap_importance.csv"), index=False)
    low[["AppID", "Name", "owners_low", "owners_high", "median_playtime_h"]].to_csv(
        config.derived_file("high_reach_low_engagement.csv"), index=False)
    status("Saved SHAP importance and the high-reach, low-engagement list")


if __name__ == "__main__":
    main()
