"""
Feature importance extraction and brain-atlas mapping.

Extracts importances from fitted sklearn pipelines (linear coefficients
or permutation-based) and parses FreeSurfer-style ROI column names into
structured region records suitable for downstream visualization.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def extract_feature_importance(
    fitted_pipeline: Pipeline,
    feature_names: list[str],
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    method: str = "auto",
    seed: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Extract feature importances from a fitted sklearn pipeline.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted pipeline whose last step is a classifier or regressor.
    feature_names : list[str]
        Feature names corresponding to the columns of the training data.
    X_test, y_test : np.ndarray | None
        Test data required when ``method="permutation"``.
    method : str
        ``"auto"`` inspects the estimator for ``coef_`` or
        ``feature_importances_`` attributes and falls back to permutation
        importance if neither is found.  ``"permutation"`` forces
        permutation importance (requires *X_test* and *y_test*).
        ``"model"`` forces model-intrinsic extraction.
    seed : int
        Random seed for permutation importance.
    n_repeats : int
        Number of permutation repeats.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature``, ``importance``,
        ``abs_importance``, sorted by descending absolute importance.

    Raises
    ------
    ValueError
        If permutation importance is requested but test data is missing.
    """
    estimator = fitted_pipeline.named_steps.get("clf", fitted_pipeline.named_steps.get("reg"))

    importances: np.ndarray | None = None

    if method in ("auto", "model"):
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            # For multiclass, coef_ is (n_classes, n_features); average across classes.
            importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_

    if importances is None:
        if method == "model":
            raise ValueError(
                f"Estimator {type(estimator).__name__} has no coef_ or "
                "feature_importances_ attribute."
            )
        if X_test is None or y_test is None:
            raise ValueError("Permutation importance requires X_test and y_test.")

        logger.info("Using permutation importance (n_repeats=%d).", n_repeats)
        result = permutation_importance(
            fitted_pipeline,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=-1,
        )
        importances = result.importances_mean

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "abs_importance": np.abs(importances),
        }
    )
    return df.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def parse_roi_name(feature_name: str) -> dict[str, str | None]:
    """
    Parse a FreeSurfer-style feature name into structured components.

    Examples
    --------
    >>> parse_roi_name("lh_bankssts_thickness")
    {'hemisphere': 'lh', 'region': 'bankssts', 'measure': 'thickness'}
    >>> parse_roi_name("Left-Hippocampus_volume")
    {'hemisphere': 'Left', 'region': 'Hippocampus', 'measure': 'volume'}
    >>> parse_roi_name("morphometry_feature_001")
    {'hemisphere': None, 'region': None, 'measure': None}

    Parameters
    ----------
    feature_name : str
        Column name from the ROI feature table.

    Returns
    -------
    dict[str, str | None]
        Keys: ``hemisphere``, ``region``, ``measure``.
    """
    # Cortical: lh_region_measure or rh_region_measure
    cortical = re.match(r"^(lh|rh)_(.+)_(thickness|area|volume|gwc|ndi|curv)$", feature_name)
    if cortical:
        return {
            "hemisphere": cortical.group(1),
            "region": cortical.group(2),
            "measure": cortical.group(3),
        }

    # Subcortical: Left-Region_volume or Right-Region_volume
    subcortical = re.match(r"^(Left|Right)-(.+)_(volume)$", feature_name)
    if subcortical:
        return {
            "hemisphere": subcortical.group(1),
            "region": subcortical.group(2),
            "measure": subcortical.group(3),
        }

    return {"hemisphere": None, "region": None, "measure": None}


def map_to_atlas(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Annotate top-N important features with brain-atlas information.

    Parses each feature name into hemisphere, region, and measure, then
    returns a DataFrame suitable for atlas-based plotting.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Output of :func:`extract_feature_importance`.
    top_n : int
        Number of top features to annotate.

    Returns
    -------
    pd.DataFrame
        Top-N features with additional ``hemisphere``, ``region``,
        ``measure`` columns.
    """
    top = importance_df.head(top_n).copy()
    parsed = top["feature"].apply(parse_roi_name).apply(pd.Series)
    return pd.concat([top, parsed], axis=1)
