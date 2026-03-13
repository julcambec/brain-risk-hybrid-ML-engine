"""ROI feature table construction from FreeSurfer stats outputs.

FreeSurfer's ``aparcstats2table`` and ``asegstats2table`` commands produce
tab-separated tables with one row per subject and one column per region.
This module parses those tables, combines them into a single wide-format
feature DataFrame, validates the schema, and provides a placeholder hook
for site-harmonization (e.g. ComBat).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------
# Parsers for FreeSurfer stats tables
# --------------------------------------


def parse_aparc_stats(stats_file: str | Path, measure: str) -> pd.DataFrame:
    """Read an ``aparcstats2table``-produced file into a tidy DataFrame.

    The input is a tab-separated table where the first column is the subject
    ID and subsequent columns are region names.

    Parameters
    ----------
    stats_file : str | Path
        Path to the stats table (e.g. ``aparc_thickness_lh.txt``).
    measure : str
        Name of the morphometric measure (e.g. ``"thickness"``, ``"area"``).
        Used to suffix the column names.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``subject_id`` and ``{region}_{measure}``
        columns.
    """
    df = pd.read_csv(str(stats_file), sep="\t")
    first_col = df.columns[0]
    rename_map: dict[str, str] = {first_col: "subject_id"}
    for col in df.columns[1:]:
        rename_map[col] = f"{col}_{measure}"
    return df.rename(columns=rename_map)


def parse_aseg_stats(stats_file: str | Path) -> pd.DataFrame:
    """Read an ``asegstats2table``-produced file into a tidy DataFrame.

    Parameters
    ----------
    stats_file : str | Path
        Path to the aseg stats table (e.g. ``aseg_stats.txt``).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with ``subject_id`` and volume columns.
    """
    df = pd.read_csv(str(stats_file), sep="\t")
    first_col = df.columns[0]
    return df.rename(columns={first_col: "subject_id"})


# ---------------
# Table builder
# ---------------


def build_roi_table(
    stats_dir: str | Path,
    measures: list[str] | None = None,
) -> pd.DataFrame:
    """Combine multiple FreeSurfer stats tables into a single wide-format ROI table.

    If *stats_dir* contains a pre-built ``roi_stats.csv`` (e.g. from demo
    mode), that file is loaded directly.

    Parameters
    ----------
    stats_dir : str | Path
        Directory containing FreeSurfer stats files **or** a pre-built
        ``roi_stats.csv``.
    measures : list[str] | None
        Morphometric measures to look for (default: ``["thickness", "area"]``).
        Ignored when loading a pre-built CSV.

    Returns
    -------
    pd.DataFrame
        Wide-format ROI feature table with ``subject_id`` as the first column.
    """
    stats_dir = Path(stats_dir)

    # Fast path: pre-built CSV (demo mode / pre-extracted features)
    prebuilt = stats_dir / "roi_stats.csv"
    if prebuilt.exists():
        return pd.read_csv(str(prebuilt))

    if measures is None:
        measures = ["thickness", "area"]

    frames: list[pd.DataFrame] = []
    for measure in measures:
        for hemi in ("lh", "rh"):
            pattern = f"aparc_{measure}_{hemi}.txt"
            path = stats_dir / pattern
            if path.exists():
                frames.append(parse_aparc_stats(path, measure))

    aseg_path = stats_dir / "aseg_stats.txt"
    if aseg_path.exists():
        frames.append(parse_aseg_stats(aseg_path))

    if not frames:
        raise FileNotFoundError(f"No FreeSurfer stats files found in {stats_dir}")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="subject_id", how="outer")
    return merged


# ------------
# Validation
# ------------


def validate_roi_schema(
    df: pd.DataFrame,
    expected_n_features: int | None = None,
) -> list[str]:
    """Check that a ROI feature DataFrame has the expected structure.

    Parameters
    ----------
    df : pd.DataFrame
        ROI feature table.
    expected_n_features : int | None
        If provided, warn when the number of feature columns (i.e. all columns
        except ``subject_id``) does not match.

    Returns
    -------
    list[str]
        Warning messages. An empty list means all checks passed.
    """
    warnings: list[str] = []

    if "subject_id" not in df.columns:
        warnings.append("Missing 'subject_id' column")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        warnings.append("No numeric feature columns found")

    n_nan = int(df[numeric_cols].isna().sum().sum()) if numeric_cols else 0
    if n_nan > 0:
        warnings.append(f"{n_nan} NaN value(s) in numeric features")

    metadata_columns = {"subject_id", "site"}
    feature_columns = [c for c in df.columns if c not in metadata_columns]

    if expected_n_features is not None:
        n_feat = len(feature_columns)
        if n_feat != expected_n_features:
            warnings.append(f"Expected {expected_n_features} features, found {n_feat}")

    return warnings


# --------------------------
# Harmonization hook (stub)
# --------------------------


def harmonize_sites(
    df: pd.DataFrame,
    site_labels: np.ndarray,
    method: str = "combat",
) -> pd.DataFrame:
    """Harmonize ROI features across acquisition sites.

    Note::

    This is a placeholder. In a production pipeline, site
    harmonization would be performed using ComBat (Johnson et al., 2007)
    or neuroCombat (Fortin et al., 2018) to remove site-related batch
    effects from multi-site neuroimaging data while preserving biological
    variability. This stub returns the input unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        ROI feature table.
    site_labels : np.ndarray
        Array of site identifiers, one per row in *df*.
    method : str
        Harmonization method name (currently only ``"combat"`` is recognized,
        but not implemented).

    Returns
    -------
    pd.DataFrame
        The input DataFrame, unchanged.
    """
    logger.warning(
        "harmonize_sites() is a stub — returning data unchanged. "
        "Integrate neuroCombat for real site harmonization."
    )
    return df
