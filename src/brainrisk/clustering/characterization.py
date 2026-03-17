"""
Subtype characterization: statistical comparisons between HYDRA-derived
subtypes and controls across imaging, clinical, and longitudinal measures.

Implements the full characterization workflow used in my MSc analyses:

- Imaging ANCOVAs with FDR correction and effect sizes.
- Non-imaging comparisons (ANCOVAs for continuous, chi-squared for
  categorical variables) with effect sizes.
- Longitudinal analysis via Reliable Change Index and proportion
  comparison across subtypes.

All implementations use scipy, numpy, and pandas only (no statsmodels
dependency). ANCOVA is computed via OLS (numpy least-squares) comparing
a full model (group + covariates) against a reduced model (covariates
only).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ------------------
# Internal helpers
# ------------------


def _encode_covariates(covariates: pd.DataFrame) -> np.ndarray:
    """
    Build a numeric design matrix from a covariate DataFrame.

    Numeric columns pass through unchanged. Categorical / object columns
    are one-hot encoded with the first level dropped.

    Returns a 2-D float64 array with one row per observation.
    """
    parts: list[np.ndarray] = []

    numeric = covariates.select_dtypes(include="number")
    if not numeric.empty:
        parts.append(numeric.to_numpy(dtype=np.float64))

    categorical = covariates.select_dtypes(exclude="number")
    if not categorical.empty:
        dummies = pd.get_dummies(categorical, drop_first=True)
        parts.append(dummies.to_numpy(dtype=np.float64))

    if not parts:
        raise ValueError("Covariate DataFrame has no usable columns.")

    return np.column_stack(parts)


def _ols_ss_residual(X: np.ndarray, y: np.ndarray) -> float:
    """Return the residual sum of squares from an OLS fit of *y* on *X*."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return float(np.dot(residuals, residuals))


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction (step-up procedure).

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (1-D).

    Returns
    -------
    np.ndarray
        Adjusted p-values, same length as input.
    """
    n = len(p_values)
    if n == 0:
        return p_values.copy()

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.empty(n, dtype=np.float64)
    adjusted[-1] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(sorted_p[i] * n / (i + 1), adjusted[i + 1])
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result


def _bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Bonferroni correction."""
    return np.clip(p_values * len(p_values), 0.0, 1.0)


def _correct_pvalues(p_values: np.ndarray, method: str) -> np.ndarray:
    """Apply multiple-comparison correction."""
    if method == "fdr":
        return _fdr_bh(p_values)
    if method == "bonferroni":
        return _bonferroni(p_values)
    return p_values.copy()


def _covariate_adjusted_means(
    y: np.ndarray,
    group_labels: np.ndarray,
    cov_matrix: np.ndarray,
) -> dict[int, float]:
    """
    Compute least-squares (marginal) means per group.

    Fits a full OLS model with group dummies + covariates, then evaluates
    each group at the mean covariate values.
    """
    unique_groups = np.sort(np.unique(group_labels))
    n = len(y)

    # Build full design matrix
    intercept = np.ones((n, 1))
    group_dummies = pd.get_dummies(pd.Series(group_labels), drop_first=True).to_numpy(
        dtype=np.float64
    )
    X = np.column_stack([intercept, group_dummies, cov_matrix])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Predict at mean covariates for each group
    mean_covs = cov_matrix.mean(axis=0)
    means: dict[int, float] = {}
    for g in unique_groups:
        row = np.zeros(X.shape[1])
        row[0] = 1.0  # intercept
        # Set the appropriate group dummy
        for i, col in enumerate(pd.get_dummies(pd.Series(unique_groups), drop_first=True).columns):
            idx = 1 + i  # offset by intercept
            row[idx] = 1.0 if col == g else 0.0
        # Set covariates to their means
        cov_start = 1 + group_dummies.shape[1]
        row[cov_start:] = mean_covs
        means[int(g)] = float(row @ beta)

    return means


# ------------
# Public API
# ------------


def imaging_ancova(
    features: pd.DataFrame,
    subtype_labels: np.ndarray,
    covariates: pd.DataFrame,
    alpha: float = 0.05,
    correction: str = "fdr",
) -> pd.DataFrame:
    """
    Compare HYDRA subtypes vs controls on imaging features using ANCOVA.

    For each imaging feature, fits a full OLS model (group + covariates)
    and a reduced model (covariates only), then tests the group factor
    with an incremental F-test.  Reports F-statistics, raw and corrected
    p-values, and partial eta-squared effect sizes.

    Parameters
    ----------
    features : pd.DataFrame
        Imaging feature matrix (n_subjects rows, feature columns).
    subtype_labels : np.ndarray
        Integer array of subtype assignments (e.g. 1, 2, 3 for patients;
        0 or -1 for controls).
    covariates : pd.DataFrame
        Covariate matrix with columns such as ``age``, ``sex``, ``site``.
    alpha : float
        Significance threshold (default 0.05).
    correction : str
        Multiple-comparison correction method (``"fdr"`` or ``"bonferroni"``).

    Returns
    -------
    pd.DataFrame
        One row per feature with columns: ``feature``, ``F_stat``,
        ``p_value``, ``p_corrected``, ``partial_eta_sq``, ``significant``.
    """
    cov_matrix = _encode_covariates(covariates)
    n = len(subtype_labels)
    unique_groups = np.sort(np.unique(subtype_labels))
    n_groups = len(unique_groups)

    intercept = np.ones((n, 1))
    group_dummies = pd.get_dummies(pd.Series(subtype_labels), drop_first=True).to_numpy(
        dtype=np.float64
    )

    X_full = np.column_stack([intercept, group_dummies, cov_matrix])
    X_reduced = np.column_stack([intercept, cov_matrix])

    df_num = n_groups - 1
    df_den = n - X_full.shape[1]

    records: list[dict[str, Any]] = []
    for col in features.columns:
        y = features[col].to_numpy(dtype=np.float64)

        ss_full = _ols_ss_residual(X_full, y)
        ss_reduced = _ols_ss_residual(X_reduced, y)
        ss_group = ss_reduced - ss_full

        if df_den > 0 and ss_full > 1e-12:
            f_stat = (ss_group / df_num) / (ss_full / df_den)
            p_value = float(1.0 - sp_stats.f.cdf(f_stat, df_num, df_den))
        else:
            f_stat = 0.0
            p_value = 1.0

        partial_eta_sq = ss_group / (ss_group + ss_full) if (ss_group + ss_full) > 1e-12 else 0.0

        records.append(
            {
                "feature": col,
                "F_stat": float(f_stat),
                "p_value": float(p_value),
                "partial_eta_sq": float(partial_eta_sq),
            }
        )

    result_df = pd.DataFrame(records)

    raw_p = result_df["p_value"].to_numpy()
    result_df["p_corrected"] = _correct_pvalues(raw_p, correction)
    result_df["significant"] = result_df["p_corrected"] < alpha

    return result_df


def non_imaging_comparisons(
    clinical_df: pd.DataFrame,
    subtype_labels: np.ndarray,
    continuous_vars: list[str],
    categorical_vars: list[str],
    covariates: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    Compare subtypes on non-imaging (clinical / environmental) variables.

    Continuous variables are tested with ANCOVA (if covariates are provided)
    or one-way ANOVA, and reported with Cohen's d effect sizes for each
    pairwise contrast. Categorical variables are tested with chi-squared
    tests and reported with Cramér's V.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Clinical / demographic data with one row per subject.
    subtype_labels : np.ndarray
        Integer subtype assignments.
    continuous_vars : list[str]
        Column names for continuous variables.
    categorical_vars : list[str]
        Column names for categorical variables.
    covariates : pd.DataFrame | None
        Optional covariate matrix for ANCOVA on continuous variables.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``"continuous"`` : DataFrame with ``variable``, ``F_stat``,
        ``p_value``, ``p_corrected``, ``cohens_d`` per variable.
        ``"categorical"`` : DataFrame with ``variable``, ``chi2``,
        ``p_value``, ``p_corrected``, ``cramers_v`` per variable.
    """
    unique_groups = np.sort(np.unique(subtype_labels))

    # Continuous variables (ANCOVA / ANOVA)
    cont_records: list[dict[str, Any]] = []
    for var in continuous_vars:
        y = clinical_df[var].to_numpy(dtype=np.float64)
        n = len(y)

        if covariates is not None:
            cov_matrix = _encode_covariates(covariates)
            intercept = np.ones((n, 1))
            group_dummies = pd.get_dummies(pd.Series(subtype_labels), drop_first=True).to_numpy(
                dtype=np.float64
            )
            X_full = np.column_stack([intercept, group_dummies, cov_matrix])
            X_reduced = np.column_stack([intercept, cov_matrix])

            ss_full = _ols_ss_residual(X_full, y)
            ss_reduced = _ols_ss_residual(X_reduced, y)
            ss_group = ss_reduced - ss_full

            df_num = len(unique_groups) - 1
            df_den = n - X_full.shape[1]

            if df_den > 0 and ss_full > 1e-12:
                f_stat = (ss_group / df_num) / (ss_full / df_den)
                p_value = float(1.0 - sp_stats.f.cdf(f_stat, df_num, df_den))
                s_residual = np.sqrt(ss_full / df_den)
            else:
                f_stat, p_value, s_residual = 0.0, 1.0, 1.0
        else:
            groups_data = [y[subtype_labels == g] for g in unique_groups]
            f_stat, p_value = sp_stats.f_oneway(*groups_data)
            f_stat = float(f_stat)
            p_value = float(p_value)
            s_residual = float(np.std(y, ddof=1))

        # Pairwise Cohen's d (using residual SD as denominator)
        group_means = {int(g): float(y[subtype_labels == g].mean()) for g in unique_groups}
        if len(unique_groups) >= 2:
            g1, g2 = int(unique_groups[0]), int(unique_groups[-1])
            cohens_d = (group_means[g2] - group_means[g1]) / max(s_residual, 1e-8)
        else:
            cohens_d = 0.0

        cont_records.append(
            {
                "variable": var,
                "F_stat": f_stat,
                "p_value": p_value,
                "cohens_d": float(cohens_d),
            }
        )

    cont_df = pd.DataFrame(cont_records)
    if not cont_df.empty:
        cont_df["p_corrected"] = _correct_pvalues(cont_df["p_value"].to_numpy(), "fdr")
    else:
        cont_df["p_corrected"] = pd.Series(dtype=float)

    # Categorical variables (chi-squared)
    cat_records: list[dict[str, Any]] = []
    for var in categorical_vars:
        contingency = pd.crosstab(subtype_labels, clinical_df[var])
        chi2, p_value, dof, _ = sp_stats.chi2_contingency(contingency)

        n_obs = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_obs * max(min_dim, 1))) if n_obs > 0 else 0.0

        cat_records.append(
            {
                "variable": var,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "cramers_v": float(cramers_v),
            }
        )

    cat_df = pd.DataFrame(cat_records)
    if not cat_df.empty:
        cat_df["p_corrected"] = _correct_pvalues(cat_df["p_value"].to_numpy(), "fdr")
    else:
        cat_df["p_corrected"] = pd.Series(dtype=float)

    return {"continuous": cont_df, "categorical": cat_df}


def longitudinal_analysis(
    baseline_scores: pd.DataFrame,
    followup_scores: pd.DataFrame,
    subtype_labels: np.ndarray,
    covariates: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Assess subtype-level trajectories in psychopathology over time.

    Computes the Reliable Change Index (RCI) for each subject to classify
    individual-level change as improved, stable, or worsened. Then compares
    proportions across subtypes using chi-squared tests.

    The RCI formula is::

        RCI = (followup - baseline) / SE_diff

    where ``SE_diff = SD_baseline * sqrt(2 * (1 - r))`` and *r* is the
    Pearson correlation between baseline and follow-up scores (a proxy
    for test-retest reliability).

    Parameters
    ----------
    baseline_scores : pd.DataFrame
        Baseline scores. Must contain ``subject_id`` and one or more
        numeric outcome columns.
    followup_scores : pd.DataFrame
        Follow-up scores (same columns as baseline).
    subtype_labels : np.ndarray
        Integer subtype assignments aligned with the score DataFrames.
    covariates : pd.DataFrame | None
        Optional covariates (reserved for future mixed-model extension;
        currently unused).

    Returns
    -------
    dict[str, Any]
        ``"rci"`` : pd.DataFrame; per-subject RCI values and categories
        for each outcome variable.
        ``"proportion_test"`` : dict; chi-squared test of RCI categories
        across subtypes for each outcome.
        ``"trajectory_test"`` : dict; group-level mean-change comparison
        (one-way ANOVA on change scores) for each outcome.
    """
    outcome_cols = [c for c in baseline_scores.columns if c != "subject_id"]

    rci_records: list[dict[str, Any]] = []
    proportion_tests: dict[str, dict[str, Any]] = {}
    trajectory_tests: dict[str, dict[str, Any]] = {}

    for outcome in outcome_cols:
        bl = baseline_scores[outcome].to_numpy(dtype=np.float64)
        fu = followup_scores[outcome].to_numpy(dtype=np.float64)
        change = fu - bl

        # Test-retest reliability proxy: Pearson r between baseline and
        # follow-up. Clip to avoid negative SE_diff under very low r.
        r = float(np.corrcoef(bl, fu)[0, 1])
        r = max(r, 0.1)

        sd_bl = float(np.std(bl, ddof=1))
        se_diff = sd_bl * np.sqrt(2.0 * (1.0 - r))
        se_diff = max(se_diff, 1e-8)

        rci = change / se_diff

        category = np.where(
            rci > 1.96,
            "worsened",
            np.where(rci < -1.96, "improved", "no_change"),
        )

        for i in range(len(bl)):
            rci_records.append(
                {
                    "outcome": outcome,
                    "subject_idx": i,
                    "subtype": int(subtype_labels[i]),
                    "rci": float(rci[i]),
                    "category": category[i],
                }
            )

        # Chi-squared test on RCI categories across subtypes
        ct = pd.crosstab(
            pd.Series(subtype_labels, name="subtype"),
            pd.Series(category, name="category"),
        )
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = sp_stats.chi2_contingency(ct)
        else:
            chi2, p, dof = 0.0, 1.0, 0

        proportion_tests[outcome] = {
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "contingency_table": ct.to_dict(),
        }

        # Group-level trajectory comparison (one-way ANOVA on change scores)
        unique_groups = np.unique(subtype_labels)
        groups_change = [change[subtype_labels == g] for g in unique_groups]
        if len(groups_change) >= 2 and all(len(gc) > 0 for gc in groups_change):
            f_stat, f_p = sp_stats.f_oneway(*groups_change)
        else:
            f_stat, f_p = 0.0, 1.0

        trajectory_tests[outcome] = {
            "F_stat": float(f_stat),
            "p_value": float(f_p),
            "mean_change_per_group": {
                int(g): float(change[subtype_labels == g].mean()) for g in unique_groups
            },
        }

    rci_df = pd.DataFrame(rci_records)

    return {
        "rci": rci_df,
        "proportion_test": proportion_tests,
        "trajectory_test": trajectory_tests,
    }
