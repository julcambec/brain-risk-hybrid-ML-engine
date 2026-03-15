"""
Subject-level train/test splitting utilities.

In medical imaging datasets a single participant may contribute multiple
scans or repeated measures. Splitting at the *subject* level (rather
than the row level) prevents information leakage between train and test
sets. All functions in this module operate on subject IDs, guaranteeing
that every observation from a given participant lands in exactly one
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def subject_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets by subject ID.

    Every row belonging to a given ``subject_id`` is placed entirely in
    either the training or the test set — never both.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain a ``subject_id`` column.
    test_size : float
        Fraction of *subjects* (not rows) to hold out for testing.
    stratify_col : str | None
        Column name used for stratified splitting. When provided, the
        class distribution of this column is preserved in both splits.
        The column must have one unique value per subject.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame
        Training partition.
    test_df : pd.DataFrame
        Test partition.

    Raises
    ------
    KeyError
        If ``subject_id`` is missing from the DataFrame.
    """
    if "subject_id" not in df.columns:
        raise KeyError("DataFrame must contain a 'subject_id' column.")

    subjects = df["subject_id"].unique()

    stratify_values: np.ndarray | None = None
    if stratify_col is not None:
        # Take the first value per subject for stratification.
        subject_labels = df.groupby("subject_id")[stratify_col].first().reindex(subjects)
        stratify_values = subject_labels.to_numpy()

    train_ids, test_ids = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_values,
    )

    train_df = df[df["subject_id"].isin(set(train_ids))].copy()
    test_df = df[df["subject_id"].isin(set(test_ids))].copy()

    return train_df, test_df
