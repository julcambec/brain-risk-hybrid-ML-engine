"""
ROI-based classification pipelines.

Provides a uniform interface for building, tuning, and evaluating
classification models on FreeSurfer ROI features. Three model families
are supported:

- **DummyClassifier**: always-majority baseline (lower bound).
- **HistGradientBoostingClassifier**: tree-based, handles mixed
  feature types and scales well.
- **LogisticRegression**: linear baseline with L2 regularisation and
  balanced class weights.

All models are wrapped in sklearn ``Pipeline`` objects with a
``StandardScaler`` first stage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MODEL_TYPES = {"dummy", "hist_boosting", "logistic"}


def build_pipeline(model_type: str, seed: int = 42) -> Pipeline:
    """
    Construct a classification pipeline with a ``StandardScaler`` front-end.

    Parameters
    ----------
    model_type : str
        One of ``"dummy"``, ``"hist_boosting"``, or ``"logistic"``.
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
        "dummy": DummyClassifier(strategy="most_frequent", random_state=seed),
        "hist_boosting": HistGradientBoostingClassifier(random_state=seed),
        "logistic": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        ),
    }

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", estimators[model_type]),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``accuracy``, ``f1_macro``,
        ``precision_macro``, ``recall_macro``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _param_dist(model_type: str) -> dict[str, Any]:
    """Return the hyperparameter search space for a given model type."""
    if model_type == "hist_boosting":
        return {
            "clf__max_depth": [3, 5, 7, None],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_iter": [100, 200, 300],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
        }
    if model_type == "logistic":
        return {
            "clf__C": uniform(0.1, 10),
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

    For ``model_type="dummy"`` no tuning is performed.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices for train and test splits.
    y_train, y_test : np.ndarray
        Label arrays for train and test splits.
    model_type : str
        Model family (see :func:`build_pipeline`).
    n_iter : int
        Number of random search iterations (ignored for dummy).
    cv : int
        Cross-validation folds inside the search.
    seed : int
        Random seed for reproducibility.

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
            scoring="f1_macro",
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
        "Classification [%s]: F1-macro=%.3f, Accuracy=%.3f",
        model_type,
        metrics["f1_macro"],
        metrics["accuracy"],
    )

    return {
        "model_type": model_type,
        "best_params": best_params,
        "metrics": metrics,
        "fitted_pipeline": best_pipe,
    }


def run_classification_suite(
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
    Run Dummy, HistGradientBoosting, and Logistic classifiers and compare.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Label arrays.
    task_name : str
        Human-readable label for this task (e.g. ``"HYDRA subtype"``).
    n_iter : int
        Number of random-search iterations for tuned models.
    cv : int
        Cross-validation folds inside the search.
    seed : int
        Random seed.

    Returns
    -------
    list[dict[str, Any]]
        One entry per model, each containing ``task``, ``model_type``,
        ``metrics``, and ``best_params``.
    """
    results: list[dict[str, Any]] = []

    for model_type in ("dummy", "hist_boosting", "logistic"):
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
