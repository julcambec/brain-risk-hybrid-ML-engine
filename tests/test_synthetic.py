from __future__ import annotations

import numpy as np
import pandas as pd

from brainrisk.data.synthetic import (
    generate_clinical_data,
    generate_labels,
    generate_roi_features,
    generate_volumetric_data,
)


def test_generate_roi_features_writes_expected_schema(temp_output_dir) -> None:
    df = generate_roi_features(
        n_subjects=12,
        n_features=40,
        n_sites=3,
        seed=123,
        output_dir=temp_output_dir,
    )

    assert df.shape == (12, 42)
    assert list(df.columns[:2]) == ["subject_id", "site"]
    assert df["subject_id"].is_unique
    assert set(df["site"].unique()) <= {"site_01", "site_02", "site_03"}

    numeric_df = df.drop(columns=["subject_id", "site"])
    assert numeric_df.notna().all().all()
    assert np.isfinite(numeric_df.to_numpy()).all()

    thickness_cols = [col for col in numeric_df.columns if col.endswith("_thickness")]
    assert thickness_cols
    assert numeric_df[thickness_cols].ge(1.0).all().all()
    assert numeric_df[thickness_cols].le(5.0).all().all()

    assert (temp_output_dir / "roi" / "features.csv").exists()


def test_generate_labels_has_reasonable_distribution(temp_output_dir) -> None:
    df = generate_labels(
        n_subjects=300,
        seed=123,
        output_dir=temp_output_dir,
    )

    proportions = df["hydra_subtype"].value_counts(normalize=True).sort_index()
    expected = {1: 0.34, 2: 0.36, 3: 0.30}

    for subtype, target in expected.items():
        assert abs(float(proportions.loc[subtype]) - target) < 0.10

    assert (temp_output_dir / "labels" / "subtype_labels.csv").exists()


def test_generate_clinical_data_writes_expected_columns(temp_output_dir) -> None:
    df = generate_clinical_data(
        n_subjects=12,
        n_sites=3,
        seed=123,
        output_dir=temp_output_dir,
    )

    expected_columns = {
        "subject_id",
        "site",
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

    assert set(df.columns) == expected_columns
    assert df["subject_id"].is_unique
    assert set(df["sex"].unique()) <= {"F", "M"}
    assert set(df["risk_group"].unique()) <= {"PH+", "PH-"}
    assert set(df["hydra_subtype"].unique()) <= {1, 2, 3}
    assert pd.api.types.is_numeric_dtype(df["cbcl_internalizing"])
    assert pd.api.types.is_numeric_dtype(df["family_income_k"])

    assert (temp_output_dir / "labels" / "clinical.csv").exists()


def test_generate_volumetric_data_writes_npy_files_and_manifest(
    temp_output_dir,
    small_volume_shape,
) -> None:
    manifest_df = generate_volumetric_data(
        n_subjects=12,
        shape=small_volume_shape,
        seed=123,
        output_dir=temp_output_dir,
    )

    npy_files = sorted((temp_output_dir / "volumes").glob("*.npy"))

    assert len(npy_files) == 12
    assert manifest_df.shape[0] == 12
    assert {"subject_id", "hydra_subtype", "volume_path"} <= set(manifest_df.columns)

    sample_volume = np.load(npy_files[0])
    assert sample_volume.shape == small_volume_shape
    assert sample_volume.dtype == np.float32
    assert float(sample_volume.min()) >= 0.0
    assert float(sample_volume.max()) <= 1.0
    assert float(sample_volume.sum()) > 0.0

    assert (temp_output_dir / "volumes" / "manifest.csv").exists()


def test_fixture_bundle_creates_all_expected_outputs(synthetic_bundle) -> None:
    assert synthetic_bundle["roi"].exists()
    assert synthetic_bundle["labels"].exists()
    assert synthetic_bundle["clinical"].exists()
    assert synthetic_bundle["manifest"].exists()
    assert synthetic_bundle["volumes_dir"].exists()
