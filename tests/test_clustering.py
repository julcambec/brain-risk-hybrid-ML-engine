"""
Unit tests for the clustering module: HYDRA stand-in, evaluation, and
characterization stubs.
"""

from __future__ import annotations

import numpy as np
import pytest

from brainrisk.clustering.evaluation import ari_across_k, compute_ari, permutation_test
from brainrisk.clustering.hydra import HydraClusterer, run_hydra_matlab

# ----------------------------
# evaluation.py — compute_ari
# ----------------------------


def test_compute_ari_perfect_agreement() -> None:
    labels = np.array([1, 1, 2, 2, 3, 3])
    assert compute_ari(labels, labels) == 1.0


def test_compute_ari_no_agreement_is_low() -> None:
    a = np.array([1, 1, 1, 2, 2, 2])
    b = np.array([2, 2, 2, 1, 1, 1])
    # Swapped labels still have perfect structure → ARI = 1.0
    assert compute_ari(a, b) == 1.0

    # Random vs structured should give low ARI
    rng = np.random.default_rng(42)
    random_labels = rng.integers(1, 4, size=100)
    structured = np.repeat([1, 2, 3, 1], 25)
    ari = compute_ari(random_labels, structured)
    assert ari < 0.3


def test_compute_ari_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_ari(np.array([1, 2]), np.array([1, 2, 3]))


# ---------------------------------
# evaluation.py — permutation_test
# ---------------------------------


def test_permutation_test_returns_valid_structure() -> None:
    rng = np.random.default_rng(42)
    # Create features with some cluster structure
    features = np.vstack(
        [
            rng.normal(0, 1, (30, 10)),
            rng.normal(3, 1, (30, 10)),
            rng.normal(-3, 1, (30, 10)),
        ]
    )
    labels = np.array([1] * 30 + [2] * 30 + [3] * 30)

    from sklearn.cluster import KMeans

    def cluster_fn(feats: np.ndarray) -> np.ndarray:
        return KMeans(n_clusters=3, n_init=5, random_state=42).fit_predict(feats) + 1

    result = permutation_test(
        features=features,
        labels=labels,
        cluster_fn=cluster_fn,
        n_permutations=20,
        seed=42,
    )

    assert "observed_ari" in result
    assert "null_distribution" in result
    assert "p_value" in result
    assert len(result["null_distribution"]) == 20
    assert 0.0 <= result["p_value"] <= 1.0


# -----------------------------
# evaluation.py — ari_across_k
# -----------------------------


def test_ari_across_k_returns_valid_structure() -> None:
    rng = np.random.default_rng(42)
    patient_features = np.vstack(
        [
            rng.normal(0, 1, (40, 10)),
            rng.normal(3, 1, (40, 10)),
        ]
    )

    from sklearn.cluster import KMeans

    def factory(k: int):
        def cluster_fn(feats: np.ndarray) -> np.ndarray:
            return KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(feats) + 1

        return cluster_fn

    result = ari_across_k(
        features=patient_features,
        k_range=[2, 3, 4],
        cluster_factory=factory,
        n_folds=3,
        seed=42,
    )

    assert result["k_values"] == [2, 3, 4]
    assert len(result["mean_ari"]) == 3
    assert len(result["std_ari"]) == 3
    assert result["best_k"] in [2, 3, 4]
    # All ARIs should be finite
    assert all(np.isfinite(a) for a in result["mean_ari"])


# ---------------------------
# hydra.py — HydraClusterer
# ---------------------------


def test_hydra_clusterer_fit_returns_labels() -> None:
    rng = np.random.default_rng(42)
    patient = rng.normal(0, 1, (60, 20))
    control = rng.normal(0, 1, (40, 20))

    clusterer = HydraClusterer(n_clusters=3, covariate_correction=False, seed=42)
    clusterer.fit(patient, control)

    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == 60
    assert set(np.unique(clusterer.labels_)) <= {1, 2, 3}


def test_hydra_clusterer_labels_are_one_indexed() -> None:
    rng = np.random.default_rng(42)
    patient = rng.normal(0, 1, (60, 20))
    control = rng.normal(0, 1, (40, 20))

    clusterer = HydraClusterer(n_clusters=2, covariate_correction=False, seed=42)
    clusterer.fit(patient, control)

    assert clusterer.labels_ is not None
    assert int(clusterer.labels_.min()) >= 1


def test_hydra_clusterer_with_covariate_correction() -> None:
    rng = np.random.default_rng(42)
    patient = rng.normal(0, 1, (60, 20))
    control = rng.normal(0, 1, (40, 20))
    patient_cov = rng.normal(0, 1, (60, 3))
    control_cov = rng.normal(0, 1, (40, 3))

    clusterer = HydraClusterer(n_clusters=3, covariate_correction=True, seed=42)
    clusterer.fit(patient, control, patient_cov, control_cov)

    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == 60


def test_hydra_clusterer_predict_after_fit() -> None:
    rng = np.random.default_rng(42)
    patient = rng.normal(0, 1, (60, 20))
    control = rng.normal(0, 1, (40, 20))

    clusterer = HydraClusterer(n_clusters=3, covariate_correction=False, seed=42)
    clusterer.fit(patient, control)

    new_samples = rng.normal(0, 1, (10, 20))
    predictions = clusterer.predict(new_samples)
    assert len(predictions) == 10
    assert all(p >= 1 for p in predictions)


def test_hydra_clusterer_predict_before_fit_raises() -> None:
    clusterer = HydraClusterer(n_clusters=3)
    with pytest.raises(RuntimeError, match="fit"):
        clusterer.predict(np.ones((5, 10)))


def test_hydra_clusterer_get_cluster_fn() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (60, 10))

    clusterer = HydraClusterer(n_clusters=3, seed=42)
    cluster_fn = clusterer.get_cluster_fn()

    labels = cluster_fn(features)
    assert len(labels) == 60
    assert set(np.unique(labels)) <= {1, 2, 3}


def test_hydra_clusterer_get_cluster_factory() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (60, 10))

    clusterer = HydraClusterer(seed=42)
    factory = clusterer.get_cluster_factory()

    for k in [2, 3, 4]:
        fn = factory(k)
        labels = fn(features)
        assert len(labels) == 60
        assert len(np.unique(labels)) <= k


def test_hydra_clusterer_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    patient = rng.normal(0, 1, (60, 20))
    control = rng.normal(0, 1, (40, 20))

    c1 = HydraClusterer(n_clusters=3, covariate_correction=False, seed=42)
    c1.fit(patient, control)

    c2 = HydraClusterer(n_clusters=3, covariate_correction=False, seed=42)
    c2.fit(patient, control)

    assert c1.labels_ is not None
    assert c2.labels_ is not None
    np.testing.assert_array_equal(c1.labels_, c2.labels_)


# ------------------------------------
# hydra.py — run_hydra_matlab (stub)
# ------------------------------------


def test_run_hydra_matlab_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="documented stub"):
        run_hydra_matlab(
            feature_csv="data.csv",
            covariate_csv="covar.csv",
            output_dir="results/",
        )


# --------------------------------------------
# characterization.py — stubs raise correctly
# --------------------------------------------


def test_imaging_ancova_stub_raises() -> None:
    import pandas as pd

    from brainrisk.clustering.characterization import imaging_ancova

    with pytest.raises(NotImplementedError, match="stubbed"):
        imaging_ancova(
            features=pd.DataFrame({"f1": [1.0]}),
            subtype_labels=np.array([1]),
            covariates=pd.DataFrame({"age": [10.0]}),
        )


def test_non_imaging_comparisons_stub_raises() -> None:
    import pandas as pd

    from brainrisk.clustering.characterization import non_imaging_comparisons

    with pytest.raises(NotImplementedError, match="stubbed"):
        non_imaging_comparisons(
            clinical_df=pd.DataFrame({"x": [1]}),
            subtype_labels=np.array([1]),
            continuous_vars=["x"],
            categorical_vars=[],
        )


def test_longitudinal_analysis_stub_raises() -> None:
    import pandas as pd

    from brainrisk.clustering.characterization import longitudinal_analysis

    with pytest.raises(NotImplementedError, match="stubbed"):
        longitudinal_analysis(
            baseline_scores=pd.DataFrame({"subject_id": ["s1"], "score": [50]}),
            followup_scores=pd.DataFrame({"subject_id": ["s1"], "score": [55]}),
            subtype_labels=np.array([1]),
        )
