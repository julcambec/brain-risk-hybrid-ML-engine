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
# characterization.py
# --------------------------------------------


def test_imaging_ancova_returns_valid_dataframe() -> None:
    """imaging_ancova produces correct columns and reasonable values."""
    import pandas as pd

    from brainrisk.clustering.characterization import imaging_ancova

    rng = np.random.default_rng(42)
    n = 80

    # Create features with some group structure
    labels = np.array([0] * 40 + [1] * 20 + [2] * 20)
    features = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n) + 0.5 * labels,
            "feat_b": rng.normal(0, 1, n),
        }
    )
    covariates = pd.DataFrame(
        {
            "age": rng.uniform(9, 11, n),
            "sex": rng.choice(["M", "F"], n),
        }
    )

    result = imaging_ancova(features, labels, covariates, alpha=0.05, correction="fdr")

    assert "feature" in result.columns
    assert "F_stat" in result.columns
    assert "p_value" in result.columns
    assert "p_corrected" in result.columns
    assert "partial_eta_sq" in result.columns
    assert "significant" in result.columns
    assert len(result) == 2
    assert (result["F_stat"] >= 0).all()
    assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()
    assert (result["partial_eta_sq"] >= 0).all()
    # feat_a should have a larger F-stat than feat_b (has signal)
    f_a = float(result.loc[result["feature"] == "feat_a", "F_stat"].iloc[0])
    f_b = float(result.loc[result["feature"] == "feat_b", "F_stat"].iloc[0])
    assert f_a > f_b


def test_imaging_ancova_bonferroni_correction() -> None:
    """Bonferroni correction produces valid adjusted p-values."""
    import pandas as pd

    from brainrisk.clustering.characterization import imaging_ancova

    rng = np.random.default_rng(42)
    n = 60
    labels = np.array([0] * 30 + [1] * 30)
    features = pd.DataFrame({"x": rng.normal(0, 1, n)})
    covariates = pd.DataFrame({"age": rng.uniform(9, 11, n)})

    result = imaging_ancova(features, labels, covariates, correction="bonferroni")
    assert (result["p_corrected"] >= result["p_value"]).all()


def test_non_imaging_comparisons_returns_expected_structure() -> None:
    """non_imaging_comparisons returns continuous and categorical DataFrames."""
    import pandas as pd

    from brainrisk.clustering.characterization import non_imaging_comparisons

    rng = np.random.default_rng(42)
    n = 60
    labels = np.array([1] * 20 + [2] * 20 + [3] * 20)

    clinical_df = pd.DataFrame(
        {
            "income": rng.normal(90, 20, n) + 5 * labels,
            "category": rng.choice(["low", "high"], n),
        }
    )

    result = non_imaging_comparisons(
        clinical_df=clinical_df,
        subtype_labels=labels,
        continuous_vars=["income"],
        categorical_vars=["category"],
    )

    assert "continuous" in result
    assert "categorical" in result

    cont = result["continuous"]
    assert "variable" in cont.columns
    assert "F_stat" in cont.columns
    assert "cohens_d" in cont.columns
    assert len(cont) == 1

    cat = result["categorical"]
    assert "variable" in cat.columns
    assert "chi2" in cat.columns
    assert "cramers_v" in cat.columns
    assert len(cat) == 1
    assert (cat["cramers_v"] >= 0).all()


def test_non_imaging_with_covariates() -> None:
    """non_imaging_comparisons runs correctly when covariates are provided."""
    import pandas as pd

    from brainrisk.clustering.characterization import non_imaging_comparisons

    rng = np.random.default_rng(42)
    n = 60
    labels = np.array([1] * 30 + [2] * 30)
    clinical_df = pd.DataFrame({"score": rng.normal(50, 10, n)})
    covariates = pd.DataFrame({"age": rng.uniform(9, 11, n)})

    result = non_imaging_comparisons(
        clinical_df=clinical_df,
        subtype_labels=labels,
        continuous_vars=["score"],
        categorical_vars=[],
        covariates=covariates,
    )
    assert len(result["continuous"]) == 1
    assert result["categorical"].empty


def test_longitudinal_analysis_returns_rci_and_tests() -> None:
    """longitudinal_analysis returns RCI DataFrame, proportion test, and trajectory test."""
    import pandas as pd

    from brainrisk.clustering.characterization import longitudinal_analysis

    rng = np.random.default_rng(42)
    n = 60
    labels = np.array([1] * 20 + [2] * 20 + [3] * 20)

    baseline = pd.DataFrame(
        {
            "subject_id": [f"sub-{i:03d}" for i in range(n)],
            "internalizing": rng.normal(50, 8, n),
        }
    )
    followup = pd.DataFrame(
        {
            "subject_id": [f"sub-{i:03d}" for i in range(n)],
            "internalizing": baseline["internalizing"]
            + rng.normal(0, 3, n)
            + np.where(labels == 1, 4.0, 0.0),  # subtype 1 worsens
        }
    )

    result = longitudinal_analysis(baseline, followup, labels)

    assert "rci" in result
    assert "proportion_test" in result
    assert "trajectory_test" in result

    rci_df = result["rci"]
    assert "outcome" in rci_df.columns
    assert "rci" in rci_df.columns
    assert "category" in rci_df.columns
    assert set(rci_df["category"].unique()) <= {"improved", "no_change", "worsened"}
    assert len(rci_df) == n  # one row per subject per outcome

    prop = result["proportion_test"]
    assert "internalizing" in prop
    assert "chi2" in prop["internalizing"]
    assert "p_value" in prop["internalizing"]

    traj = result["trajectory_test"]
    assert "internalizing" in traj
    assert "F_stat" in traj["internalizing"]
    assert "mean_change_per_group" in traj["internalizing"]


# -------------------------------------------------------
# Integration: HydraClusterer on synthetic pipeline data
# -------------------------------------------------------


def test_hydra_clusterer_on_synthetic_bundle(synthetic_bundle: dict) -> None:
    """Integration test: cluster synthetic ROI features from the pipeline fixtures."""
    import pandas as pd

    roi_df = pd.read_csv(str(synthetic_bundle["roi"]))
    labels_df = pd.read_csv(str(synthetic_bundle["labels"]))

    # Merge to get subtype labels aligned with features
    merged = roi_df.merge(labels_df, on="subject_id")

    # Split into patient (subtypes 1-3) and control features
    # In real data, controls are PH- (group=-1). In synthetic data,
    # all subjects are PH+ with subtypes 1-3. For this integration test,
    # I designate subtype 3 as "pseudo-controls" to exercise the interface.
    control_mask = merged["hydra_subtype"] == 3
    patient_mask = ~control_mask

    feature_cols = [c for c in roi_df.columns if c not in {"subject_id", "site"}]
    patient_features = merged.loc[patient_mask, feature_cols].to_numpy()
    control_features = merged.loc[control_mask, feature_cols].to_numpy()

    clusterer = HydraClusterer(n_clusters=2, covariate_correction=False, seed=42)
    clusterer.fit(patient_features, control_features)

    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == int(patient_mask.sum())
    assert set(np.unique(clusterer.labels_)) <= {1, 2}


def test_evaluation_with_hydra_helpers() -> None:
    """Verify evaluation functions work with HydraClusterer's callable helpers."""
    rng = np.random.default_rng(42)
    # Create features with clear cluster structure
    patient_features = np.vstack(
        [
            rng.normal(-2, 0.5, (30, 10)),
            rng.normal(2, 0.5, (30, 10)),
            rng.normal(0, 0.5, (30, 10)),
        ]
    )
    control_features = rng.normal(0, 1, (40, 10))

    clusterer = HydraClusterer(n_clusters=3, covariate_correction=False, seed=42)
    clusterer.fit(patient_features, control_features)

    # Test permutation_test with get_cluster_fn
    cluster_fn = clusterer.get_cluster_fn()

    assert clusterer.labels_ is not None

    perm_result = permutation_test(
        features=patient_features,
        labels=clusterer.labels_,
        cluster_fn=cluster_fn,
        n_permutations=10,
        seed=42,
    )
    assert "p_value" in perm_result

    # Test ari_across_k with get_cluster_factory
    factory = clusterer.get_cluster_factory()
    ari_result = ari_across_k(
        features=patient_features,
        k_range=[2, 3, 4],
        cluster_factory=factory,
        n_folds=3,
        seed=42,
    )
    assert ari_result["best_k"] in [2, 3, 4]
