"""
HYDRA clustering: sklearn-based stand-in and documented MATLAB interface.

HYDRA (Heterogeneity through Discriminative Analysis) is a semi-supervised
clustering algorithm that identifies patient subtypes relative to a control
group using multiple max-margin discriminative hyperplanes (Varol et al.,
2017). The key insight is that different patient subtypes may deviate from
the control distribution in *different directions*, so a single hyperplane
(standard SVM) cannot capture the heterogeneity.

Production workflow
-------------------
In my MSc research, the real HYDRA clustering was executed via the
original MATLAB implementation on HPC.

This module provides:

1. A **Python stand-in** (``HydraClusterer``) that approximates the
   semi-supervised logic using scikit-learn for demo and testing purposes.
2. A **documented stub** (``run_hydra_matlab``) describing how to invoke
   the MATLAB implementation as an external step.

Reference
---------
Varol, E., Sotiras, A., & Davatzikos, C. (2017). HYDRA: Revealing
heterogeneity of imaging and genetic patterns through a multiple
max-margin discriminative analysis framework. *NeuroImage*, 145, 346–364.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HydraClusterer:
    """
    Sklearn-based stand-in for HYDRA semi-supervised clustering.

    This approximation captures the core idea of HYDRA (clustering patients
    *relative to a control distribution*) without the multi-hyperplane SVM
    machinery. The steps are:

    1. **Standardize** features using control-group statistics (mean, std).
       This centers the feature space on the control distribution, so
       patient deviations become the signal.
    2. **Residualize** covariates (optional, via ordinary least squares)
       to remove confounds such as age, sex, and site.
    3. **Cluster** the residualized patient features using KMeans.

    This is adequate for demo/testing. For real analyses, use the MATLAB
    HYDRA implementation (see :func:`run_hydra_matlab`).

    Parameters
    ----------
    n_clusters : int
        Number of patient subtypes to discover (default 3).
    covariate_correction : bool
        Whether to regress out covariates before clustering (default True).
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : np.ndarray | None
        Cluster assignments for the patient group after fitting.
        Labels are 1-indexed (1, 2, …, k) to match HYDRA convention.
    scaler_ : StandardScaler | None
        Fitted scaler (from control group statistics).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        covariate_correction: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.covariate_correction = covariate_correction
        self.seed = seed

        self.labels_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self._kmeans: KMeans | None = None

    def fit(
        self,
        patient_features: np.ndarray,
        control_features: np.ndarray,
        patient_covariates: np.ndarray | None = None,
        control_covariates: np.ndarray | None = None,
    ) -> HydraClusterer:
        """
        Fit the clustering model.

        Parameters
        ----------
        patient_features : np.ndarray
            Feature matrix for the patient group (n_patients, n_features).
        control_features : np.ndarray
            Feature matrix for the control group (n_controls, n_features).
        patient_covariates : np.ndarray | None
            Covariate matrix for patients (n_patients, n_covariates).
            Required if ``covariate_correction=True``.
        control_covariates : np.ndarray | None
            Covariate matrix for controls (n_controls, n_covariates).
            Required if ``covariate_correction=True``.

        Returns
        -------
        HydraClusterer
            The fitted instance (for method chaining).
        """
        # Standardize using control-group statistics
        self.scaler_ = StandardScaler()
        self.scaler_.fit(control_features)
        patient_z = self.scaler_.transform(patient_features)

        # Optional covariate correction via OLS residualization
        if self.covariate_correction and patient_covariates is not None:
            patient_z = self._residualize(
                patient_z,
                patient_covariates,
                control_features,
                control_covariates,
            )

        # KMeans clustering on residualized patient features
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.seed,
        )
        zero_indexed = self._kmeans.fit_predict(patient_z)
        self.labels_ = zero_indexed + 1  # 1-indexed to match HYDRA convention

        logger.info(
            "HydraClusterer fit: k=%d, n_patients=%d, n_controls=%d",
            self.n_clusters,
            len(patient_features),
            len(control_features),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Assign new patient samples to the nearest cluster.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix for new samples (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1-indexed cluster assignments.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self._kmeans is None or self.scaler_ is None:
            raise RuntimeError("Call fit() before predict().")
        z = self.scaler_.transform(features)
        return self._kmeans.predict(z) + 1

    def _residualize(
        self,
        patient_z: np.ndarray,
        patient_covariates: np.ndarray,
        control_features: np.ndarray,
        control_covariates: np.ndarray | None,
    ) -> np.ndarray:
        """
        Remove covariate effects via OLS regression fitted on controls.

        Fits a linear model ``features ~ covariates`` on the combined
        (control + patient) data, then returns the patient residuals.
        This removes variance explained by confounds (e.g. age, sex, site)
        while preserving subtype-related signal.
        """
        if control_covariates is None:
            logger.warning("No control covariates provided; skipping residualization.")
            return patient_z

        control_z = self.scaler_.transform(control_features)  # type: ignore[union-attr]

        # Combine for a single OLS fit
        all_features = np.vstack([control_z, patient_z])
        all_covariates = np.vstack([control_covariates, patient_covariates])

        # Add intercept
        X = np.column_stack([np.ones(len(all_covariates)), all_covariates])

        # OLS: beta = (X'X)^-1 X'Y
        beta = np.linalg.lstsq(X, all_features, rcond=None)[0]

        # Patient residuals only
        n_controls = len(control_z)
        X_patient = X[n_controls:]
        predicted = X_patient @ beta
        residuals = patient_z - predicted

        return residuals

    def get_cluster_fn(self) -> Any:
        """
        Return a stateless clustering callable for evaluation utilities.

        The returned function has signature ``fn(features) -> labels``
        and creates a fresh KMeans fit each time it is called. This is
        used by :func:`brainrisk.clustering.evaluation.permutation_test`
        and :func:`brainrisk.clustering.evaluation.ari_across_k`.
        """
        n_clusters = self.n_clusters
        seed = self.seed

        def _cluster_fn(features: np.ndarray) -> np.ndarray:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            return km.fit_predict(features) + 1

        return _cluster_fn

    def get_cluster_factory(self) -> Any:
        """
        Return a factory callable for evaluation across k values.

        The returned function has signature ``factory(k) -> cluster_fn``
        where ``cluster_fn(features) -> labels``.
        """
        seed = self.seed

        def _factory(k: int) -> Any:
            def _cluster_fn(features: np.ndarray) -> np.ndarray:
                km = KMeans(n_clusters=k, n_init=10, random_state=seed)
                return km.fit_predict(features) + 1

            return _cluster_fn

        return _factory


# ------------------------------------------
# Documented stub for the real MATLAB HYDRA
# ------------------------------------------


def run_hydra_matlab(
    feature_csv: str | Path,
    covariate_csv: str | Path,
    output_dir: str | Path,
    k_max: int = 15,
    n_folds: int = 10,
    regularization: float = 1.0,
    matlab_executable: str = "matlab",
) -> None:
    """
    Invoke the MATLAB HYDRA implementation as an external process.

    .. warning::

       This function is a **documented stub**. It describes the interface
       contract for calling the original MATLAB HYDRA implementation but
       does not execute it. For actual HYDRA clustering, use the original
       MATLAB-based implementation with MATLAB available.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the input feature matrix.
    covariate_csv : str | Path
        Path to the covariate matrix.
    output_dir : str | Path
        Directory for HYDRA output files.
    k_max : int
        Maximum number of clusters to evaluate (solutions k=2..k_max).
    n_folds : int
        Number of cross-validation folds for ARI-based model selection.
    regularization : float
        SVM regularization parameter (C).
    matlab_executable : str
        Path to the MATLAB executable.

    Raises
    ------
    NotImplementedError
        Always: this is a documented interface stub.
    """
    raise NotImplementedError(
        "run_hydra_matlab() is a documented stub. "
        "The real HYDRA implementation requires MATLAB. "
        "For demo/testing, use HydraClusterer (the sklearn-based stand-in)."
    )
