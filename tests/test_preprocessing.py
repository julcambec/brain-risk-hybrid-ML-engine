"""
Unit tests for individual preprocessing modules.
Each section tests one module in isolation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainrisk.preprocessing.freesurfer import generate_mock_outputs, is_freesurfer_available
from brainrisk.preprocessing.mni305 import apply_affine_warp, demo_affine_warp
from brainrisk.preprocessing.nifti_io import load_npy_as_volume, validate_volume
from brainrisk.preprocessing.normalization import minmax_normalize, validate_normalized
from brainrisk.preprocessing.orientation import reorient_volume
from brainrisk.preprocessing.resampling import compute_zoom_factors, resample_to_shape
from brainrisk.preprocessing.roi_extraction import harmonize_sites, validate_roi_schema
from brainrisk.utils.config import load_config, merge_configs
from brainrisk.utils.logging import setup_logger

# ----------------
# utils/config.py
# ----------------


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("batch_size: 32\nmodel:\n  depth: 6\n  heads: 8\n")
    result = load_config(cfg_path)
    assert result["batch_size"] == 32
    assert result["model"]["depth"] == 6
    assert result["model"]["heads"] == 8


def test_load_config_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_raises_on_empty_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(cfg_path)


def test_merge_configs_overrides_win() -> None:
    base = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    overrides = {"a": 99, "b": {"y": 99}}
    merged = merge_configs(base, overrides)
    assert merged["a"] == 99
    assert merged["b"]["x"] == 10
    assert merged["b"]["y"] == 99
    assert merged["c"] == 3


# -----------------
# utils/logging.py
# -----------------


def test_setup_logger_returns_logger() -> None:
    logger = setup_logger(name="brainrisk.test_unit", level="DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "brainrisk.test_unit"
    assert logger.level == logging.DEBUG


# --------------------------
# preprocessing/nifti_io.py
# --------------------------


def test_load_npy_as_volume_returns_correct_shape_and_dtype(tmp_path: Path) -> None:
    vol = np.random.default_rng(0).uniform(0, 1, (16, 16, 16)).astype(np.float32)
    path = tmp_path / "test_vol.npy"
    np.save(path, vol)

    loaded_vol, affine = load_npy_as_volume(path)
    assert loaded_vol.shape == (16, 16, 16)
    assert loaded_vol.dtype == np.float32
    np.testing.assert_array_equal(affine, np.eye(4))


def test_validate_volume_passes_clean_volume() -> None:
    vol = np.random.default_rng(0).uniform(0.1, 1.0, (16, 16, 16)).astype(np.float32)
    affine = np.eye(4)
    warnings = validate_volume(vol, affine)
    assert warnings == []


def test_validate_volume_catches_nan() -> None:
    vol = np.ones((16, 16, 16), dtype=np.float32)
    vol[5, 5, 5] = np.nan
    warnings = validate_volume(vol, np.eye(4))
    assert any("NaN" in w for w in warnings)


def test_validate_volume_catches_flat_intensity() -> None:
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    warnings = validate_volume(vol, np.eye(4))
    assert any("intensity range" in w for w in warnings)


# --------------------------------
# preprocessing/normalization.py
# --------------------------------


def test_minmax_normalize_scales_to_unit_range() -> None:
    vol = np.random.default_rng(0).uniform(50, 200, (16, 16, 16)).astype(np.float32)
    mask = np.zeros_like(vol, dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    vol[~mask] = 0.0

    normed = minmax_normalize(vol, mask=mask)
    brain_vals = normed[mask]
    assert float(brain_vals.min()) >= 0.0
    assert float(brain_vals.max()) <= 1.0 + 1e-6
    assert float(brain_vals.max()) > 0.99


def test_minmax_normalize_preserves_background() -> None:
    vol = np.random.default_rng(0).uniform(50, 200, (16, 16, 16)).astype(np.float32)
    mask = np.zeros_like(vol, dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    vol[~mask] = 0.0

    normed = minmax_normalize(vol, mask=mask)
    assert float(normed[~mask].sum()) == 0.0


def test_validate_normalized_passes_clean_volume() -> None:
    vol = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    warnings = validate_normalized(vol)
    assert warnings == []


# -----------------------------
# preprocessing/resampling.py
# -----------------------------


def test_resample_to_shape_produces_target_shape() -> None:
    vol = np.random.default_rng(0).uniform(0, 1, (16, 16, 16)).astype(np.float32)
    resampled = resample_to_shape(vol, target_shape=(64, 64, 64))
    assert resampled.shape == (64, 64, 64)


def test_resample_to_shape_preserves_value_range() -> None:
    vol = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16, 16)).astype(np.float32)
    resampled = resample_to_shape(vol, target_shape=(32, 32, 32))
    assert float(resampled.min()) >= -0.1
    assert float(resampled.max()) <= 1.1


def test_compute_zoom_factors_correctness() -> None:
    factors = compute_zoom_factors((16, 16, 16), (64, 64, 64))
    assert factors == (4.0, 4.0, 4.0)

    factors_asym = compute_zoom_factors((10, 20, 30), (20, 40, 60))
    assert factors_asym == (2.0, 2.0, 2.0)


def test_compute_zoom_factors_raises_on_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="Dimension mismatch"):
        compute_zoom_factors((16, 16), (64, 64, 64))


# ------------------------------
# preprocessing/orientation.py
# ------------------------------


def test_reorient_volume_produces_same_shape() -> None:
    vol = np.random.default_rng(0).uniform(0, 1, (16, 16, 16)).astype(np.float32)
    reoriented = reorient_volume(vol)
    assert reoriented.shape == vol.shape


def test_reorient_volume_is_not_identity() -> None:
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[0, 0, 0] = 1.0
    reoriented = reorient_volume(vol)
    assert reoriented[0, 0, 0] != 1.0


# ----------------------------
# preprocessing/freesurfer.py
# ----------------------------


def test_is_freesurfer_available_returns_bool() -> None:
    result = is_freesurfer_available()
    assert isinstance(result, bool)


def test_generate_mock_outputs_creates_expected_files_and_rewrites_subject_id(
    tmp_path: Path,
) -> None:
    requested_subject_id = "sub-demo-123"

    outputs = generate_mock_outputs(
        subject_id=requested_subject_id,
        output_dir=tmp_path,
        volume_shape=(32, 32, 32),
        seed=42,
    )

    brainmask_path = outputs["brainmask"]
    roi_stats_path = outputs["roi_stats"]

    assert brainmask_path.exists()
    assert roi_stats_path.exists()

    assert brainmask_path == tmp_path / requested_subject_id / "mri" / "brainmask.npy"
    assert roi_stats_path == tmp_path / requested_subject_id / "stats" / "roi_stats.csv"

    vol = np.load(brainmask_path)
    assert vol.shape == (32, 32, 32)
    assert vol.dtype == np.float32

    df = pd.read_csv(roi_stats_path)
    assert "subject_id" in df.columns
    assert df.shape[0] == 1
    assert df.loc[0, "subject_id"] == requested_subject_id

    numeric_cols = df.select_dtypes(include="number").columns
    assert len(numeric_cols) > 0

    # The adapter should expose the requested ID, not the staged synthetic one.
    assert "sub-00001" not in df["subject_id"].tolist()


# ---------------------------------
# preprocessing/roi_extraction.py
# ---------------------------------


def test_validate_roi_schema_passes_clean_dataframe() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-002"],
            "lh_bankssts_thickness": [2.5, 2.7],
            "rh_cuneus_area": [1500.0, 1600.0],
        }
    )
    warnings = validate_roi_schema(df)
    assert warnings == []


def test_validate_roi_schema_catches_missing_subject_id() -> None:
    df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})
    warnings = validate_roi_schema(df)
    assert any("subject_id" in w for w in warnings)


def test_validate_roi_schema_catches_nan() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-002"],
            "feat_a": [1.0, np.nan],
        }
    )
    warnings = validate_roi_schema(df)
    assert any("NaN" in w for w in warnings)


def test_harmonize_sites_returns_unchanged() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-002"],
            "feat_a": [1.0, 2.0],
        }
    )
    sites = np.array(["site_01", "site_02"])
    result = harmonize_sites(df, sites)
    pd.testing.assert_frame_equal(result, df)


# ------------------------
# preprocessing/mni305.py
# ------------------------


def test_apply_affine_warp_preserves_shape_with_identity() -> None:
    vol = np.random.default_rng(0).uniform(0, 1, (16, 16, 16)).astype(np.float32)
    identity = np.eye(4)
    warped = apply_affine_warp(vol, identity, target_shape=(16, 16, 16))
    assert warped.shape == (16, 16, 16)
    np.testing.assert_allclose(warped, vol, atol=1e-5)


def test_demo_affine_warp_returns_same_shape() -> None:
    vol = np.random.default_rng(0).uniform(0, 1, (16, 16, 16)).astype(np.float32)
    warped = demo_affine_warp(vol, seed=42)
    assert warped.shape == vol.shape
    assert warped.dtype == np.float32


# ----------------------------------------
# preprocessing/volume_standardization.py
# ----------------------------------------


def test_standardize_volume_produces_target_shape() -> None:
    """End-to-end: standardize a synthetic volume to (32, 32, 32)."""
    from brainrisk.preprocessing.volume_standardization import standardize_volume

    rng = np.random.default_rng(0)
    # Create a volume with a non-trivial brain-like region
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[3:13, 3:13, 3:13] = rng.uniform(0.2, 1.0, (10, 10, 10)).astype(np.float32)

    config = {
        "target_shape": [32, 32, 32],
        "mask_threshold": 0.05,
        "apply_warp": False,  # Skip warp for unit test speed
    }
    result = standardize_volume(vol, config)
    assert result.shape == (32, 32, 32)
    assert result.dtype == np.float32


def test_standardize_volume_output_is_normalized() -> None:
    """Standardized volume should have values in a reasonable range."""
    from brainrisk.preprocessing.volume_standardization import standardize_volume

    rng = np.random.default_rng(1)
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[2:14, 2:14, 2:14] = rng.uniform(0.3, 1.0, (12, 12, 12)).astype(np.float32)

    config = {"target_shape": [32, 32, 32], "apply_warp": False}
    result = standardize_volume(vol, config)

    # After min-max normalization and resampling, values should be >= 0
    # (some slight interpolation undershoot is acceptable)
    assert float(result.min()) >= -0.1
    # Should have some non-zero content
    assert float(result.sum()) > 0.0


def test_standardize_volume_raises_on_empty_mask() -> None:
    """If the volume is all zeros, the brain mask is empty → error."""
    from brainrisk.preprocessing.volume_standardization import standardize_volume

    vol = np.zeros((16, 16, 16), dtype=np.float32)
    config = {"target_shape": [32, 32, 32], "apply_warp": False}
    with pytest.raises(ValueError, match="[Bb]rain mask is empty"):
        standardize_volume(vol, config)


# ----------------------------------------
# preprocessing/pipeline.py (integration)
# ----------------------------------------


def test_run_pipeline_demo_mode_produces_expected_artifacts(tmp_path: Path) -> None:
    """Full integration: run the demo pipeline and verify artifacts on disk."""
    from brainrisk.preprocessing.pipeline import run_pipeline

    output_dir = tmp_path / "pipeline_output"
    config_overrides = {
        "mode": "demo",
        "n_subjects": 6,
        "n_features": 20,
        "n_sites": 2,
        "volume_shape": [16, 16, 16],
        "n_demo_subjects": 2,
        "dl_branch": {
            "target_shape": [16, 16, 16],
            "mask_threshold": 0.01,
            "apply_warp": False,
        },
    }

    report = run_pipeline(config_overrides=config_overrides, output_dir=output_dir)

    # ROI branch artifacts
    roi_path = output_dir / "roi" / "features.csv"
    assert roi_path.exists()
    roi_df = pd.read_csv(str(roi_path))
    assert roi_df.shape[0] == 6
    assert "subject_id" in roi_df.columns

    # DL branch artifacts
    dl_dir = output_dir / "dl"
    assert dl_dir.exists()
    dl_manifest = pd.read_csv(str(dl_dir / "manifest.csv"))
    assert dl_manifest.shape[0] == 6
    npy_files = list(dl_dir.glob("*.npy"))
    assert len(npy_files) == 6
    sample = np.load(npy_files[0])
    assert sample.shape == (16, 16, 16)

    # FreeSurfer demo artifacts
    fs_dir = output_dir / "freesurfer"
    assert fs_dir.exists()
    assert (fs_dir / "sub-demo-001" / "mri" / "brainmask.npy").exists()
    assert (fs_dir / "sub-demo-001" / "stats" / "roi_stats.csv").exists()

    # QC report
    report_path = output_dir / "reports" / "preprocessing_qc.json"
    assert report_path.exists()

    # Report structure
    assert report["mode"] == "demo"
    assert "roi_branch" in report
    assert "dl_branch" in report
    assert report["dl_branch"]["n_subjects"] == 6


def test_run_pipeline_report_captures_config(tmp_path: Path) -> None:
    """The QC report should include the pipeline configuration."""
    from brainrisk.preprocessing.pipeline import run_pipeline

    config_overrides = {
        "mode": "demo",
        "n_subjects": 4,
        "n_features": 10,
        "n_sites": 1,
        "volume_shape": [16, 16, 16],
        "n_demo_subjects": 1,
        "dl_branch": {
            "target_shape": [16, 16, 16],
            "mask_threshold": 0.01,
            "apply_warp": False,
        },
    }
    report = run_pipeline(config_overrides=config_overrides, output_dir=tmp_path / "out")
    assert report["config"]["n_subjects"] == 4
    assert report["config"]["mode"] == "demo"
