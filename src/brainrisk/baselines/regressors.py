"""
ROI-based regression pipelines.

Mirrors the classification module's API but for continuous targets.
Supported model families:

- **DummyRegressor**: predicts the training-set mean (lower bound).
- **HistGradientBoostingRegressor**: tree-based, handles mixed features.
- **Ridge**: L2-regularised linear regression.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MODEL_TYPES = {"dummy", "hist_boosting", "ridge"}


def build_pipeline(model_type: str, seed: int = 42) -> Pipeline:
    """
    Construct a regression pipeline with a ``StandardScaler`` front-end.

    Parameters
    ----------
    model_type : str
        One of ``"dummy"``, ``"hist_boosting"``, or ``"ridge"``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline.

    Raises
    ------
    ValueError
        If *model_type* is not recognised.
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {_MODEL_TYPES}.")

    estimators: dict[str, Any] = {
        "dummy": DummyRegressor(strategy="mean"),
        "hist_boosting": HistGradientBoostingRegressor(random_state=seed),
        "ridge": Ridge(random_state=seed),
    }

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", estimators[model_type]),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``r2``, ``mae``, ``mse``.
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def _param_dist(model_type: str) -> dict[str, Any]:
    """Return the hyperparameter search space for a given model type."""
    if model_type == "hist_boosting":
        return {
            "reg__max_depth": [3, 5, 7, None],
            "reg__learning_rate": [0.01, 0.05, 0.1],
            "reg__max_iter": [100, 200, 300],
            "reg__l2_regularization": [0.0, 0.1, 1.0],
        }
    if model_type == "ridge":
        return {
            "reg__alpha": uniform(0.01, 10),
        }
    return {}


def tune_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "hist_boosting",
    n_iter: int = 10,
    cv: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Tune hyperparameters with randomised search, then evaluate on the test set.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices for train and test splits.
    y_train, y_test : np.ndarray
        Target arrays for train and test splits.
    model_type : str
        Model family (see :func:`build_pipeline`).
    n_iter : int
        Number of random search iterations (ignored for dummy).
    cv : int
        Cross-validation folds inside the search.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Keys: ``model_type``, ``best_params``, ``metrics``,
        ``fitted_pipeline``.
    """
    pipe = build_pipeline(model_type, seed=seed)
    params = _param_dist(model_type)

    if params:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=seed,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        best_params = search.best_params_
    else:
        pipe.fit(X_train, y_train)
        best_pipe = pipe
        best_params = {}

    y_pred = best_pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    logger.info(
        "Regression [%s]: R²=%.3f, MAE=%.3f",
        model_type,
        metrics["r2"],
        metrics["mae"],
    )

    return {
        "model_type": model_type,
        "best_params": best_params,
        "metrics": metrics,
        "fitted_pipeline": best_pipe,
    }


def run_regression_suite(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
    n_iter: int = 10,
    cv: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run Dummy, HistGradientBoosting, and Ridge regressors and compare.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Target arrays.
    task_name : str
        Human-readable label for this task.
    n_iter : int
        Number of random-search iterations for tuned models.
    cv : int
        Cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    list[dict[str, Any]]
        One entry per model.
    """
    results: list[dict[str, Any]] = []

    for model_type in ("dummy", "hist_boosting", "ridge"):
        result = tune_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            model_type=model_type,
            n_iter=n_iter,
            cv=cv,
            seed=seed,
        )
        result["task"] = task_name
        results.append(result)

    return results
