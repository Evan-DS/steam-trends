"""Step 6: were the fixed settings good enough, and would more data help?

1. Hold out 20% of games (stratified by reach tier).
2. On the other 80%: grid-search the baseline penalties and random-search XGBoost (3-fold CV).
3. Compare default and tuned models once on the hold-out set, with a bootstrap interval.
4. Learning curves for Model A, and for Model A vs Model B on the review sample.

Inputs : data/derived/games_clean.parquet, launch_features.parquet; data/source/review_features.parquet
Outputs: data/derived/tuning_trials.csv, tuned_settings.csv, holdout_comparison.csv, learning_curves.csv,
         learning_curve_growth.csv; figures 07_*.png
This step takes about 30 minutes.

Vocabulary:
    tuning            trying different model settings ("hyperparameters") to find ones that score better
    grid search       trying every value from a fixed list
    random search     trying randomly drawn combinations of settings
    hold-out set      games kept out of tuning entirely, so the final comparison is honest
    bootstrap         re-scoring many random re-draws of the hold-out games to see how much a difference could be luck
    learning curve    how the score changes as the model is given more training games
"""
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.utils.class_weight import compute_sample_weight

from steam_success import config
from steam_success.data import feature_columns, load_games_clean, load_launch_features, load_review_features
from steam_success.modeling import CLASSIFICATION, REGRESSION, baseline_model, fit, folds, score, xgb_model
from steam_success.plotting import COLOR, COLOR_A, COLOR_B, INK, INK_2, INK_3, log_axis, save_figure, thousands
from steam_success.printing import print_table
from steam_success.progress import status

TEST_SIZE = 0.20          # share of games held out from tuning
INNER_FOLDS = 3           # cross-validation folds inside the searches
N_TRIALS = 30             # random settings tried per target
N_BOOTSTRAP = 1000        # resamples for the hold-out difference
TRAIN_FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]   # shares of the training games used for the learning curves
FIXED = dict(tree_method="hist", random_state=config.SEED, n_jobs=-1)   # XGBoost settings that are never searched

# The three things being predicted, and how each is scored
TARGETS = {
    "Reception": dict(task=REGRESSION, score="R²", scoring="r2"),
    "Reach": dict(task=CLASSIFICATION, score="Macro F1", scoring="f1_macro"),
    "Engagement": dict(task=REGRESSION, score="R²", scoring="r2"),
}
BASELINE_GRID = {   # parameter, values, symbol
    REGRESSION: ("ridge__alpha", np.logspace(-3, 3, 13), "α"),             # penalty strength, 0.001 to 1000
    CLASSIFICATION: ("logisticregression__C", np.logspace(-3, 2, 11), "C"),   # 0.001 to 100 (smaller C = stronger penalty)
}
# The ranges the random search draws from; "log-uniform" means small values are tried as often as large ones
SEARCH_SPACE = {
    "n_estimators": randint(100, 1001),
    "learning_rate": loguniform(0.01, 0.3),
    "max_depth": randint(3, 11),
    "min_child_weight": loguniform(1, 50),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(0.1, 10),
}
# The same ranges written for people (Table V in the report), next to the default setting
SEARCH_RANGES = pd.DataFrame([
    ("n_estimators", "Number of trees", "100 – 1,000", "uniform, integers", 400),
    ("learning_rate", "Shrinkage applied to each new tree", "0.01 – 0.3", "log-uniform", 0.05),
    ("max_depth", "Maximum depth of each tree", "3 – 10", "uniform, integers", 6),
    ("min_child_weight", "Minimum weight needed to make a leaf", "1 – 50", "log-uniform", 5),
    ("subsample", "Share of games used for each tree", "0.6 – 1.0", "uniform", 0.8),
    ("colsample_bytree", "Share of features used for each tree", "0.5 – 1.0", "uniform", 0.8),
    ("reg_lambda", "L2 penalty on leaf weights", "0.1 – 10", "log-uniform", 1.0),
], columns=["Setting", "Meaning", "Range searched", "Sampled as", "Default"]).set_index("Setting")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """All games with their success measures and launch features, plus the list of feature columns."""
    games, launch = load_games_clean(), load_launch_features()
    features = feature_columns(launch)
    data = (games[["AppID", "pct_positive", "reach_code", "median_playtime_h"]]
            .merge(launch, on="AppID", how="inner", validate="one_to_one").reset_index(drop=True))
    data["reach_code"] = data["reach_code"].astype(int)
    data["eligible_engagement"] = data["median_playtime_h"].notna() & (data["reach_code"] >= config.ENGAGEMENT_MIN_TIER)
    return data, features


def target_data(data, features, name, rows):
    """X and y for one target, restricted to the given rows."""
    d = data.iloc[np.sort(rows)]
    if name == "Engagement":
        d = d[d["eligible_engagement"]]   # engagement is only defined for some games
    y = {"Reception": d["pct_positive"], "Reach": d["reach_code"],
         "Engagement": np.log(d["median_playtime_h"])}[name]
    return d[features].astype(float).reset_index(drop=True), y.reset_index(drop=True)


def hold_out_split(data, features):
    """Split off 20% of the games as the hold-out set; returns train and test data for each target."""
    # Stratified by reach tier, so the hold-out set has the same tier mix as the whole data
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=TEST_SIZE,
                                           stratify=data["reach_code"], random_state=config.SEED)
    return {name: {"train": target_data(data, features, name, train_idx),
                   "test": target_data(data, features, name, test_idx)} for name in TARGETS}


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_baselines(split):
    """Grid search for the baseline penalty of each target, scored by 3-fold cross-validation."""
    searches, t0 = {}, time.time()
    for name, cfg in TARGETS.items():
        param, grid, symbol = BASELINE_GRID[cfg["task"]]
        X_tr, y_tr = split[name]["train"]
        search = GridSearchCV(baseline_model(cfg["task"]), {param: grid}, scoring=cfg["scoring"],
                              cv=folds(cfg["task"], INNER_FOLDS), n_jobs=-1)
        searches[name] = search.fit(X_tr, y_tr)
        status(f"{name:<11} best {symbol} = {search.best_params_[param]:g}  "
              f"(CV {cfg['score']} {search.best_score_:.3f})")
    status(f"Baselines tuned in {(time.time() - t0) / 60:.1f} min")
    return searches


def tune_xgboost(split):
    """Random search per target; the default settings are scored on the same folds for reference."""
    searches, default_cv, t0 = {}, {}, time.time()
    for name, cfg in TARGETS.items():
        task = cfg["task"]
        X_tr, y_tr = split[name]["train"]
        status(f"{name}: random search, {N_TRIALS} settings × {INNER_FOLDS} folds ...")
        search = RandomizedSearchCV(xgb_model(task, **FIXED), SEARCH_SPACE, n_iter=N_TRIALS,
                                    scoring=cfg["scoring"], cv=folds(task, INNER_FOLDS),
                                    random_state=config.SEED, n_jobs=1, refit=True)
        # for reach, give every tier equal total weight while training
        weights = {"sample_weight": compute_sample_weight("balanced", y_tr)} if task == CLASSIFICATION else {}
        searches[name] = search.fit(X_tr, y_tr, **weights)

        # Score the default settings on the same folds, so "tuned vs default" is a fair comparison
        fold_scores = []
        for tr, va in folds(task, INNER_FOLDS).split(X_tr, y_tr):
            m = fit(xgb_model(task, **config.XGB_PARAMS), X_tr.iloc[tr], y_tr.iloc[tr], task, balanced=True)
            fold_scores.append(score(task, y_tr.iloc[va], m.predict(X_tr.iloc[va])))
        default_cv[name] = np.mean(fold_scores)
        status(f"{name:<11} best CV {cfg['score']} {search.best_score_:.3f} vs default {default_cv[name]:.3f}"
              f"   ({(time.time() - t0) / 60:.1f} min so far)")
    return searches, default_cv


def selected_settings(baseline_search, xgb_search) -> pd.DataFrame:
    """A table of the settings the searches chose for each target, next to the defaults."""
    selected = pd.DataFrame({"Default": SEARCH_RANGES["Default"],
                             **{name: xgb_search[name].best_params_ for name in TARGETS}})
    for name, cfg in TARGETS.items():
        param, _, symbol = BASELINE_GRID[cfg["task"]]
        selected.loc[f"Baseline {symbol}", name] = baseline_search[name].best_params_[param]
    selected.loc[["Baseline α", "Baseline C"], "Default"] = 1.0

    def tidy(v):
        """Whole numbers with thousands separators, other values to three significant digits."""
        if pd.isna(v):
            return "–"
        return f"{int(v):,}" if float(v).is_integer() else f"{float(v):.3g}"

    return selected.reindex(list(SEARCH_RANGES.index) + ["Baseline α", "Baseline C"]).astype(float).map(tidy)


def bootstrap_interval(task, y, pred_a, pred_b):
    """95% interval of score(b) − score(a), resampling the test games with replacement."""
    rng = np.random.default_rng(config.SEED)
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))   # draw len(y) games at random, some more than once
        diffs.append(score(task, y[i], pred_b[i]) - score(task, y[i], pred_a[i]))
    return np.percentile(diffs, [2.5, 97.5])   # the middle 95% of the differences


def hold_out_comparison(split, baseline_search, xgb_search) -> pd.DataFrame:
    """Table VI: default and tuned baseline and XGBoost, scored once on the hold-out games."""
    rows = []
    for name, cfg in TARGETS.items():
        task = cfg["task"]
        (X_tr, y_tr), (X_te, y_te) = split[name]["train"], split[name]["test"]
        param = BASELINE_GRID[task][0]
        # name -> (model, give every class equal weight while training?)
        models = {
            "Baseline (default)": (baseline_model(task), False),
            "Baseline (tuned)": (baseline_model(task, baseline_search[name].best_params_[param]), False),
            "XGBoost (default)": (xgb_model(task, **config.XGB_PARAMS), True),
            "XGBoost (tuned)": (xgb_model(task, **FIXED, **xgb_search[name].best_params_), True),
        }
        preds = {k: fit(m, X_tr, y_tr, task, balanced).predict(X_te) for k, (m, balanced) in models.items()}
        row = {"Target": name, "Score": cfg["score"], "Hold-out games": len(y_te),
               **{k: score(task, y_te, p) for k, p in preds.items()}}
        row["XGBoost: tuned − default"] = row["XGBoost (tuned)"] - row["XGBoost (default)"]
        row["95% low"], row["95% high"] = bootstrap_interval(task, y_te, preds["XGBoost (default)"],
                                                             preds["XGBoost (tuned)"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Target")


def print_holdout(holdout: pd.DataFrame):
    """Print Table VI, then one plain sentence per target saying whether the bootstrap interval includes 0."""
    print_table(holdout, "Default vs tuned on the hold-out set (Table VI)",
                {**{c: "{:.3f}" for c in ["Baseline (default)", "Baseline (tuned)", "XGBoost (default)",
                                          "XGBoost (tuned)"]},
                 **{c: "{:+.3f}" for c in ["XGBoost: tuned − default", "95% low", "95% high"]},
                 "Hold-out games": "{:,}"})
    for name, r in holdout.iterrows():
        lo, hi = r["95% low"], r["95% high"]
        # an interval that includes 0 means the improvement could be down to chance
        verdict = "includes 0" if lo <= 0 <= hi else ("is entirely above 0" if lo > 0 else "is entirely below 0")
        print(f"{name:<11} XGBoost {r['Score']}: default {r['XGBoost (default)']:.3f} → tuned "
              f"{r['XGBoost (tuned)']:.3f} ({r['XGBoost: tuned − default']:+.3f}); "
              f"the 95% interval [{lo:+.3f}, {hi:+.3f}] {verdict}")


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------
def subsample(train, frac, y, task, seed):
    """A random share of the training rows; stratified by class for classification."""
    if frac >= 1:
        return train   # 100%: use everything
    strat = y.iloc[train] if task == CLASSIFICATION else None
    sub, _ = train_test_split(train, train_size=frac, stratify=strat, random_state=seed)
    return np.sort(sub)


def learning_curve(X, y, task, models, splits) -> pd.DataFrame:
    """Train and test scores for each model at each training share, for every fold.

    models: {name: (make_model, columns, balanced)}
    """
    rows = []
    for fold, (train, test) in enumerate(splits):
        status(f"  fold {fold + 1}")
        for frac in TRAIN_FRACTIONS:
            sub = subsample(train, frac, y, task, seed=fold)
            for name, (make_model, cols, balanced) in models.items():
                m = fit(make_model(), X.iloc[sub][cols], y.iloc[sub], task, balanced)
                rows.append({"fold": fold, "fraction": frac, "n_train": len(sub), "model": name,
                             "train": score(task, y.iloc[sub], m.predict(X.iloc[sub][cols])),   # score on games it learned from
                             "test": score(task, y.iloc[test], m.predict(X.iloc[test][cols]))})   # score on unseen games
    return pd.DataFrame(rows)


def model_a_curves(data, features) -> dict:
    """Learning curves of XGBoost and the baseline on Model A's launch features, for all three targets."""
    curves, t0 = {}, time.time()
    for name, cfg in TARGETS.items():
        task = cfg["task"]
        X_all, y_all = target_data(data, features, name, np.arange(len(data)))
        # t=task fixes the task inside each lambda (otherwise every lambda would see the last loop value)
        models = {"XGBoost": (lambda t=task: xgb_model(t, **config.XGB_PARAMS), features, True),
                  "Baseline": (lambda t=task: baseline_model(t), features, False)}
        status(f"{name}: learning curve, {len(TRAIN_FRACTIONS)} training sizes × {config.N_SPLITS} folds ...")
        curves[name] = learning_curve(X_all, y_all, task, models, folds(task).split(X_all, y_all))
        status(f"{name:<11} learning curve done   ({(time.time() - t0) / 60:.1f} min so far)")
    return curves


def model_b_curves(data, features) -> dict:
    """Learning curves of Model A vs Model B on the early-review sample (reach and engagement)."""
    rf = load_review_features()
    sample = (data.assign(AppID=data["AppID"].astype(int))
              .merge(rf.assign(AppID=rf["AppID"].astype(int)), on="AppID"))
    sample = sample[sample["n_early_reviews"] >= config.MIN_EARLY_REVIEWS].reset_index(drop=True)
    eng_rows = sample[sample["eligible_engagement"]].reset_index(drop=True)
    setup = {
        "Reach": dict(rows=sample, y=sample["reach_code"], task=CLASSIFICATION, review=config.REVIEW_FEATURES),
        "Engagement": dict(rows=eng_rows, y=np.log(eng_rows["median_playtime_h"]), task=REGRESSION,
                           review=config.REVIEW_FEATURES_ENGAGEMENT),
    }
    curves, t0 = {}, time.time()
    for name, s in setup.items():
        X_b, y_b = s["rows"][features + s["review"]].astype(float), s["y"].reset_index(drop=True)
        # 5 folds repeated 3 times = the same 15 folds as the Model A vs Model B comparison
        splitter = (RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS,
                                            random_state=config.SEED)
                    if s["task"] == CLASSIFICATION
                    else RepeatedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS, random_state=config.SEED))
        models = {"Model A": (lambda t=s["task"]: xgb_model(t, **config.XGB_PARAMS), features, True),
                  "Model B": (lambda t=s["task"]: xgb_model(t, **config.XGB_PARAMS), features + s["review"], True)}
        status(f"{name}: Model A vs B learning curve on {len(y_b):,} games, 15 folds ...")
        curves[name] = learning_curve(X_b, y_b, s["task"], models, splitter.split(X_b, y_b))
        status(f"{name:<11} {len(y_b):,} games, done   ({(time.time() - t0) / 60:.1f} min so far)")
    return curves


def growth_table(curves: dict, model_names: list[str]) -> pd.DataFrame:
    """What doubling the training data adds: test score at 50% and at 100%, and the train-test gap at 100%."""
    rows = []
    for name, df in curves.items():
        for model in model_names:
            g = df[df["model"] == model].groupby("fraction")[["train", "test"]].mean()   # average over the folds
            rows.append({"Target": name, "Model": model,
                         "Test score, 50% of data": g.loc[0.5, "test"],
                         "Test score, 100% of data": g.loc[1.0, "test"],
                         "Gain from doubling": g.loc[1.0, "test"] - g.loc[0.5, "test"],
                         "Training − test gap, 100%": g.loc[1.0, "train"] - g.loc[1.0, "test"]})
    return pd.DataFrame(rows).set_index(["Target", "Model"])


# How to print the growth tables in the terminal
GROWTH_FORMATS = {"Test score, 50% of data": "{:.3f}", "Test score, 100% of data": "{:.3f}",
                  "Gain from doubling": "{:+.3f}", "Training − test gap, 100%": "{:.3f}"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_baseline_tuning(baseline_search):
    """Figure 07_baseline_tuning: score against penalty strength, one panel per target."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, (name, cfg) in zip(axes, TARGETS.items()):
        _, grid, symbol = BASELINE_GRID[cfg["task"]]
        res = baseline_search[name].cv_results_
        mean, sd = res["mean_test_score"], res["std_test_score"]
        ax.fill_between(grid, mean - sd, mean + sd, color=COLOR, alpha=0.15, linewidth=0)   # the ±1 SD band
        ax.plot(grid, mean, color=COLOR, lw=2, marker="o", ms=4)
        best = baseline_search[name].best_index_
        ax.plot(grid[best], mean[best], "o", ms=10, mfc="none", mec=INK, mew=1.5)   # a ring around the chosen value
        ax.axvline(1.0, color=INK_3, lw=1, ls="--")   # the default value
        ax.annotate(f"selected {symbol} = {grid[best]:.3g}", (grid[best], mean[best]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK, fontweight="bold")
        log_axis(ax.xaxis, subs=[1])
        ax.set(xlabel=f"{symbol} (log scale)", ylabel=f"CV {cfg['score']}")
        ax.set_title(f"{name}: {'Ridge' if cfg['task'] == REGRESSION else 'Logistic'} {symbol}")
    fig.suptitle("Baseline tuning: mean ± SD over 3 folds; dashed line = default value",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save_figure(fig, "07_baseline_tuning")


def plot_search_progress(xgb_search, default_cv):
    """Figure 07_search_progress: the score of each random-search trial, the best so far, and the default."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (name, cfg) in zip(axes, TARGETS.items()):
        s = xgb_search[name].cv_results_["mean_test_score"]
        trial = np.arange(1, len(s) + 1)
        ax.scatter(trial, s, s=22, color=COLOR, alpha=0.65, linewidths=0, label="One combination")
        ax.step(trial, np.maximum.accumulate(s), where="post", color=INK, lw=1.6, label="Best so far")
        ax.axhline(default_cv[name], color=INK_3, lw=1.2, ls="--", label="Default settings")
        # Zoom in on the interesting part: start the vertical axis at the 10th percentile of the trial scores
        lo = np.percentile(s, 10)
        ax.set_ylim(lo - 0.15 * (s.max() - lo), s.max() + 0.12 * (s.max() - lo))
        ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        ax.set(xlabel="Trial", ylabel=f"CV {cfg['score']}")
        ax.set_title(name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"XGBoost random search: {N_TRIALS} trials, 3-fold CV on the tuning set "
                 "(frame starts at the 10th percentile; poorer trials sit below it)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, "07_search_progress")


def plot_search_landscape(xgb_search):
    """Figure 07_search_landscape: each trial's score against four of the searched settings."""
    show = [("learning_rate", True), ("max_depth", False), ("n_estimators", False), ("min_child_weight", True)]   # (setting, log axis?)
    fig, axes = plt.subplots(3, len(show), figsize=(13, 7.6), sharey="row")
    for r, (name, cfg) in enumerate(TARGETS.items()):
        res = xgb_search[name].cv_results_
        s, best = res["mean_test_score"], xgb_search[name].best_index_
        lo = np.percentile(s, 10)
        for c, (param, log) in enumerate(show):
            ax = axes[r, c]
            v = np.array(res[f"param_{param}"], dtype=float)
            ax.scatter(v, s, s=20, color=COLOR, alpha=0.65, linewidths=0)
            ax.plot(v[best], s[best], "o", ms=10, mfc="none", mec=INK, mew=1.5)   # a ring around the selected trial
            if log:
                log_axis(ax.xaxis)
            ax.set_ylim(lo - 0.15 * (s.max() - lo), s.max() + 0.12 * (s.max() - lo))
            if r == 2:
                ax.set_xlabel(param)   # setting names only under the bottom row
            if c == 0:
                ax.set_ylabel(f"{name}\nCV {cfg['score']}")   # target names only on the left column
    fig.suptitle("Score against each setting (one dot per trial; ring = selected; frame starts at the 10th percentile)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save_figure(fig, "07_search_landscape")


def with_mean_size(df):
    """Add n_train_mean: the average number of training games at each training share (for the x axis)."""
    return df.assign(n_train_mean=df.groupby("fraction")["n_train"].transform("mean").round())


def band(ax, df, model, col, color, ls="-", label=None):
    """Draw one learning-curve line (the mean over folds), with a ±1 SD band for solid lines. Returns the table."""
    g = df[df["model"] == model].groupby("n_train_mean")[col].agg(["mean", "std"])
    ax.plot(g.index, g["mean"], color=color, lw=2, ls=ls, marker="o" if ls == "-" else None, ms=3.5, label=label)
    if ls == "-":
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, linewidth=0)
    return g


def plot_curves_a(curves):
    """Figure 07_learning_curves_model_a: top row = test scores of both models, bottom = XGBoost train vs test."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex="col")
    for c, (name, cfg) in enumerate(TARGETS.items()):
        df = with_mean_size(curves[name])
        top, bottom = axes[0, c], axes[1, c]
        band(top, df, "XGBoost", "test", COLOR, label="XGBoost")
        band(top, df, "Baseline", "test", INK_3, label="Baseline")
        top.set(ylabel=f"Test {cfg['score']}")
        top.set_title(name)
        band(bottom, df, "XGBoost", "test", COLOR, label="Test score")
        band(bottom, df, "XGBoost", "train", COLOR, ls="--", label="Training score")
        bottom.set(xlabel="Training games", ylabel=f"XGBoost {cfg['score']}")
        bottom.xaxis.set_major_formatter(thousands)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[1, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Learning curves, Model A (mean ± SD over 5 folds). Top: test score of both models. "
                 "Bottom: XGBoost training vs test score.", x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save_figure(fig, "07_learning_curves_model_a")


def plot_curves_b(curves_b):
    """Figure 07_learning_curves_model_b: Model A vs Model B test score against training size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (name, df) in zip(axes, curves_b.items()):
        df = with_mean_size(df)
        a = band(ax, df, "Model A", "test", COLOR_A, label="Model A (launch only)")
        b = band(ax, df, "Model B", "test", COLOR_B, label="Model B (+ early reviews)")
        gap = b["mean"].iloc[-1] - a["mean"].iloc[-1]   # Model B minus Model A at the largest training size
        ax.set(xlabel="Training games", ylabel="Macro F1" if name == "Reach" else "R²")
        ax.set_title(f"{name}   (gap at full size {gap:+.3f})")
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Learning curves, Model A vs Model B on the review sample: mean ± SD over 15 folds",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    save_figure(fig, "07_learning_curves_model_b")


def main():
    """Run this step: tune, compare on the hold-out set, draw the learning curves and save all the results."""
    data, features = load_data()
    split = hold_out_split(data, features)
    print_table(pd.DataFrame({name: {"Tuning set (80%)": len(s["train"][1]), "Hold-out set (20%)": len(s["test"][1])}
                              for name, s in split.items()}).T, "Hold-out split", "{:,}")

    status("Tuning the baselines (grid search) ...")
    baseline_search = tune_baselines(split)
    plot_baseline_tuning(baseline_search)

    print_table(SEARCH_RANGES, "XGBoost search ranges (Table V)", {"Default": "{:g}"})
    xgb_search, default_cv = tune_xgboost(split)
    plot_search_progress(xgb_search, default_cv)
    plot_search_landscape(xgb_search)

    selected = selected_settings(baseline_search, xgb_search)
    print_table(selected, "Selected settings")

    status("Comparing default and tuned models on the hold-out set (with bootstrap) ...")
    t0 = time.time()
    holdout = hold_out_comparison(split, baseline_search, xgb_search)
    status(f"Hold-out comparison done in {(time.time() - t0) / 60:.1f} min")
    print_holdout(holdout)

    curves = model_a_curves(data, features)
    plot_curves_a(curves)
    growth_a = growth_table(curves, ["Baseline", "XGBoost"])
    print_table(growth_a, "Model A: gain from doubling the data", GROWTH_FORMATS)

    curves_b = model_b_curves(data, features)
    plot_curves_b(curves_b)
    for name, df in curves_b.items():
        # the Model B − Model A gap on each fold, averaged per training share
        g = df.pivot_table(index=["fold", "fraction"], columns="model", values="test")
        gap = (g["Model B"] - g["Model A"]).groupby("fraction").mean()
        print(f"{name:<11} B − A gap: {gap.loc[0.5]:+.3f} with 50% of the data → {gap.loc[1.0]:+.3f} with 100%")
    growth_b = growth_table(curves_b, ["Model A", "Model B"])
    print_table(growth_b, "Model B sample: gain from doubling the data", GROWTH_FORMATS)
    pd.concat({"Model A": growth_a, "Model B sample": growth_b}, names=["Curves"]).to_csv(
        config.derived_file("learning_curve_growth.csv"))

    # Save every trial of the random searches, the chosen settings, the hold-out table and the raw curves
    trials = pd.concat({name: pd.DataFrame(s.cv_results_).filter(regex="^(param_|mean_test|std_test|rank)")
                        for name, s in xgb_search.items()}, names=["Target", "Trial"])
    trials.to_csv(config.derived_file("tuning_trials.csv"))
    selected.to_csv(config.derived_file("tuned_settings.csv"))
    holdout.to_csv(config.derived_file("holdout_comparison.csv"))
    pd.concat({**{f"Model A · {k}": v for k, v in curves.items()},
               **{f"Model B sample · {k}": v for k, v in curves_b.items()}},
              names=["Curve"]).to_csv(config.derived_file("learning_curves.csv"))
    status("Saved tuning_trials.csv, tuned_settings.csv, holdout_comparison.csv and learning_curves.csv")


if __name__ == "__main__":
    main()
