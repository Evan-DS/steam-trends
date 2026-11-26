"""Models, cross-validation and scores shared by the modelling steps."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from steam_success import config

REGRESSION, CLASSIFICATION = "regression", "classification"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def baseline_model(task: str, strength: float = 1.0):
    """Ridge regression (alpha = strength) or class-balanced logistic regression (C = strength)."""
    if task == CLASSIFICATION:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=strength, class_weight="balanced", max_iter=3000))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=strength))


def xgb_model(task: str, **params):
    """XGBoost with the project's default settings unless other settings are given."""
    params = params or config.XGB_PARAMS
    if task == CLASSIFICATION:
        return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **params)
    return XGBRegressor(**params)


def fit(model, X, y, task: str, balanced: bool = False):
    """Fit a model; balanced=True gives each class equal total weight (used for XGBoost on reach)."""
    if task == CLASSIFICATION and balanced:
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def folds(task: str, n_splits: int = config.N_SPLITS, seed: int = config.SEED):
    if task == CLASSIFICATION:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits, shuffle=True, random_state=seed)


def out_of_fold_predictions(make_model, X, y, task: str, balanced: bool = False):
    """Predict every row with a model trained on the other folds."""
    oof = np.zeros(len(y), dtype=int if task == CLASSIFICATION else float)
    for train, test in folds(task).split(X, y):
        model = fit(make_model(), X.iloc[train], y.iloc[train], task, balanced)
        oof[test] = model.predict(X.iloc[test])
    return oof


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def score(task: str, y, pred) -> float:
    """The main score: R² for regression, macro F1 for classification."""
    return f1_score(y, pred, average="macro") if task == CLASSIFICATION else r2_score(y, pred)


def regression_scores(y, pred, scale: float = 1.0) -> dict:
    return {"RMSE": np.sqrt(mean_squared_error(y, pred)) * scale,
            "MAE": mean_absolute_error(y, pred) * scale,
            "R²": r2_score(y, pred)}


def classification_scores(y, pred) -> dict:
    return {"Macro F1": f1_score(y, pred, average="macro"),
            "Balanced accuracy": balanced_accuracy_score(y, pred),
            "Accuracy": accuracy_score(y, pred)}
