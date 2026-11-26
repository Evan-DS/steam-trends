"""Step 4 — Model A: predict each success measure from launch features only.

A baseline (ridge / logistic regression) and XGBoost are compared with 5-fold
cross-validation; every game is predicted by a model that never saw it.

Inputs : data/derived/games_clean.parquet, launch_features.parquet
Outputs: data/derived/model_a_results.csv, model_a_summary.csv, model_a_robustness.csv,
         model_a_predictions.parquet; figures 04_*.png
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from steam_success import config
from steam_success.data import feature_columns, load_games_clean, load_launch_features
from steam_success.modeling import (CLASSIFICATION, REGRESSION, baseline_model, classification_scores,
                                    out_of_fold_predictions, regression_scores, score, xgb_model)
from steam_success.plotting import COLOR, INK, percent, save_figure
from steam_success.printing import print_table
from steam_success.progress import status

# Launch features that a developer can still add after release (robustness check)
LATE_FEATURES = ["n_languages", "n_audio_languages", "log_achievements"]


@dataclass
class ModelAData:
    data: pd.DataFrame       # all games with targets and features
    features: list[str]
    X: pd.DataFrame          # all games
    y_reception: pd.Series
    y_reach: pd.Series
    engagement_rows: pd.Series   # boolean mask: games used for engagement
    X_eng: pd.DataFrame
    y_eng: pd.Series         # log of median playtime (hours)


def load_data() -> ModelAData:
    games, features = load_games_clean(), load_launch_features()
    targets = games[["AppID", "Name", "pct_positive", "reach_code", "median_playtime_h"]]
    data = targets.merge(features, on="AppID", how="inner", validate="one_to_one")
    cols = feature_columns(features)
    X = data[cols].astype(float)
    eng = data["median_playtime_h"].notna() & (data["reach_code"] >= config.ENGAGEMENT_MIN_TIER)
    return ModelAData(
        data=data, features=cols, X=X,
        y_reception=data["pct_positive"],
        y_reach=data["reach_code"].astype(int),
        engagement_rows=eng,
        X_eng=X[eng].reset_index(drop=True),
        y_eng=np.log(data.loc[eng, "median_playtime_h"]).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# The three measures
# ---------------------------------------------------------------------------
def evaluate_reception(d: ModelAData):
    preds = {
        "Ridge (baseline)": out_of_fold_predictions(lambda: baseline_model(REGRESSION), d.X, d.y_reception, REGRESSION),
        "XGBoost": out_of_fold_predictions(lambda: xgb_model(REGRESSION), d.X, d.y_reception, REGRESSION),
    }
    naive = np.full(len(d.y_reception), d.y_reception.mean())
    table = pd.DataFrame({"Predict the average": regression_scores(d.y_reception, naive, 100),
                          **{k: regression_scores(d.y_reception, v, 100) for k, v in preds.items()}}).T
    return table, preds


def evaluate_reach(d: ModelAData):
    preds = {
        "Logistic (baseline)": out_of_fold_predictions(lambda: baseline_model(CLASSIFICATION), d.X, d.y_reach,
                                                       CLASSIFICATION),
        "XGBoost": out_of_fold_predictions(lambda: xgb_model(CLASSIFICATION), d.X, d.y_reach, CLASSIFICATION,
                                           balanced=True),
    }
    majority = np.zeros(len(d.y_reach), dtype=int)   # always "Low"
    table = pd.DataFrame({"Always predict Low": classification_scores(d.y_reach, majority),
                          **{k: classification_scores(d.y_reach, v) for k, v in preds.items()}}).T
    return table, preds


def evaluate_engagement(d: ModelAData):
    preds = {
        "Ridge (baseline)": out_of_fold_predictions(lambda: baseline_model(REGRESSION), d.X_eng, d.y_eng, REGRESSION),
        "XGBoost": out_of_fold_predictions(lambda: xgb_model(REGRESSION), d.X_eng, d.y_eng, REGRESSION),
    }
    naive = np.full(len(d.y_eng), d.y_eng.mean())
    table = pd.DataFrame({"Predict the average": regression_scores(d.y_eng, naive),
                          **{k: regression_scores(d.y_eng, v) for k, v in preds.items()}}).T
    table["Typical factor off"] = np.exp(table["RMSE"])
    return table, preds


def summary_table(results: dict) -> pd.DataFrame:
    rec, reach, eng = results["Reception"], results["Reach"], results["Engagement"]
    return pd.DataFrame({
        "Measure": ["Reception", "Reach", "Engagement"],
        "Score": ["R²", "Macro F1", "R²"],
        "Reference": [rec.loc["Predict the average", "R²"], reach.loc["Always predict Low", "Macro F1"],
                      eng.loc["Predict the average", "R²"]],
        "Baseline": [rec.loc["Ridge (baseline)", "R²"], reach.loc["Logistic (baseline)", "Macro F1"],
                     eng.loc["Ridge (baseline)", "R²"]],
        "XGBoost": [rec.loc["XGBoost", "R²"], reach.loc["XGBoost", "Macro F1"], eng.loc["XGBoost", "R²"]],
    }).set_index("Measure")


# ---------------------------------------------------------------------------
# Feature importance and robustness
# ---------------------------------------------------------------------------
def top_importance(model, X, y, n=10, **fit_kw) -> pd.Series:
    """Share of XGBoost's total gain per feature (top n)."""
    model.fit(X, y, **fit_kw)
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    return (gain / gain.sum()).reindex(X.columns).fillna(0).sort_values(ascending=False).head(n)


def robustness_check(d: ModelAData) -> pd.DataFrame:
    """XGBoost scores without the features that can change after launch."""
    def xgb_scores(cols):
        rec = score(REGRESSION, d.y_reception,
                    out_of_fold_predictions(lambda: xgb_model(REGRESSION), d.X[cols], d.y_reception, REGRESSION))
        reach = score(CLASSIFICATION, d.y_reach,
                      out_of_fold_predictions(lambda: xgb_model(CLASSIFICATION), d.X[cols], d.y_reach,
                                              CLASSIFICATION, balanced=True))
        eng = score(REGRESSION, d.y_eng,
                    out_of_fold_predictions(lambda: xgb_model(REGRESSION), d.X_eng[cols], d.y_eng, REGRESSION))
        return {"Reception R²": rec, "Reach macro F1": reach, "Engagement R²": eng}

    return pd.DataFrame({
        "All launch features": xgb_scores(d.features),
        "Without language counts": xgb_scores([c for c in d.features if c not in LATE_FEATURES[:2]]),
        "Without languages and achievements": xgb_scores([c for c in d.features if c not in LATE_FEATURES]),
    }).T


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > 0.5 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(4), config.REACH_SHORT)
    ax.set_yticks(range(4), config.REACH_SHORT)
    ax.set(xlabel="Predicted tier", ylabel="Actual tier", title="Reach, XGBoost: where each actual tier is predicted")
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, "04_reach_confusion_matrix")


def plot_importance(importance: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (name, s) in zip(axes, importance.items()):
        s = s[::-1]
        ax.barh(s.index, s.values, color=COLOR, height=0.65)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(percent)
        ax.grid(axis="y", visible=False)
        ax.set_title(name)
    fig.suptitle("Share of total gain, top 10 features", x=0.01, ha="left", fontsize=11.5, fontweight="bold",
                 color=INK)
    fig.tight_layout()
    save_figure(fig, "04_feature_importance")


def save_outputs(d: ModelAData, results: dict, preds: dict):
    pd.concat(results, names=["Measure", "Model"]).to_csv(config.derived_file("model_a_results.csv"))
    oof = d.data[["AppID"]].copy()
    oof["pct_positive"], oof["pred_reception_xgb"] = d.y_reception.values, preds["Reception"]["XGBoost"]
    oof["reach_code"], oof["pred_reach_xgb"] = d.y_reach.values, preds["Reach"]["XGBoost"]
    oof["log_playtime"] = np.nan
    oof["pred_log_playtime_xgb"] = np.nan
    oof.loc[d.engagement_rows.values, "log_playtime"] = d.y_eng.values
    oof.loc[d.engagement_rows.values, "pred_log_playtime_xgb"] = preds["Engagement"]["XGBoost"]
    oof.to_parquet(config.derived_file("model_a_predictions.parquet"), index=False)
    status("Saved Model A results and predictions")


def main():
    d = load_data()
    status(f"{len(d.data):,} games × {len(d.features)} launch features; "
          f"engagement models use {d.engagement_rows.sum():,} games")

    results, preds = {}, {}
    status("Reception: 5-fold cross-validation of ridge and XGBoost ...")
    results["Reception"], preds["Reception"] = evaluate_reception(d)
    print_table(results["Reception"], "Reception (percentage points)",
                {"RMSE": "{:.1f} pp", "MAE": "{:.1f} pp", "R²": "{:.3f}"})

    status("Reach: 5-fold cross-validation of logistic regression and XGBoost ...")
    results["Reach"], preds["Reach"] = evaluate_reach(d)
    print_table(results["Reach"], "Reach (4 tiers)", "{:.3f}")
    plot_confusion(d.y_reach, preds["Reach"]["XGBoost"])

    status("Engagement: 5-fold cross-validation of ridge and XGBoost ...")
    results["Engagement"], preds["Engagement"] = evaluate_engagement(d)
    print_table(results["Engagement"], "Engagement (log hours)",
                {"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.3f}", "Typical factor off": "×{:.2f}"})

    summary = summary_table(results)
    print_table(summary, "Model A summary (Table III)", {c: "{:.3f}" for c in ["Reference", "Baseline", "XGBoost"]})
    summary.to_csv(config.derived_file("model_a_summary.csv"))

    status("Feature importance ...")
    plot_importance({
        "Reception": top_importance(xgb_model(REGRESSION), d.X, d.y_reception),
        "Reach": top_importance(xgb_model(CLASSIFICATION), d.X, d.y_reach,
                                sample_weight=compute_sample_weight("balanced", d.y_reach)),
        "Engagement": top_importance(xgb_model(REGRESSION), d.X_eng, d.y_eng),
    })
    status("Robustness check: refitting XGBoost without languages and achievements (9 more runs) ...")
    robust = robustness_check(d)
    print_table(robust, "Robustness: XGBoost without late-changeable features", "{:.3f}")
    robust.to_csv(config.derived_file("model_a_robustness.csv"))

    save_outputs(d, results, preds)


if __name__ == "__main__":
    main()
