"""
Clinical variable loading and task preparation utilities.

Provides helpers to merge ROI feature tables with clinical/demographic
data and extract ``(X, y)`` arrays ready for sklearn pipelines. The
merging key is always ``subject_id``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are metadata or targets; never used as predictive features.
_METADATA_COLUMNS: set[str] = {
    "subject_id",
    "site",
    "site_clinical",
    "age_years",
    "sex",
    "risk_group",
    "hydra_subtype",
    "cbcl_internalizing",
    "cbcl_externalizing",
    "family_income_k",
    "maternal_substance_use",
    "peer_support",
}


def load_and_merge(
    roi_path: str | Path,
    clinical_path: str | Path,
) -> pd.DataFrame:
    """
    Load ROI features and clinical data, merge on ``subject_id``.

    Parameters
    ----------
    roi_path : str | Path
        Path to the ROI features CSV (from preprocessing or synthetic).
    clinical_path : str | Path
        Path to the clinical variables CSV.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ROI features and clinical variables.
    """
    roi_df = pd.read_csv(str(roi_path))
    clinical_df = pd.read_csv(str(clinical_path))
    merged = roi_df.merge(clinical_df, on="subject_id", suffixes=("", "_clinical"))
    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return column names that are ROI features (excluding metadata/targets).

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from :func:`load_and_merge`.

    Returns
    -------
    list[str]
        Sorted list of numeric feature column names.
    """
    return sorted(c for c in df.columns if c not in _METADATA_COLUMNS)


def prepare_task(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract ``(X, y, feature_names)`` arrays for a prediction task.

    Rows where the target is NaN are dropped automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from :func:`load_and_merge`.
    target_col : str
        Name of the target column.
    feature_cols : list[str] | None
        Feature columns to use. If ``None``, auto-detected via
        :func:`get_feature_columns`.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target array (n_samples,).
    feature_names : list[str]
        Column names corresponding to the columns of *X*.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols].to_numpy(dtype=np.float64)
    y = df.loc[mask, target_col].to_numpy()

    return X, y, feature_cols
