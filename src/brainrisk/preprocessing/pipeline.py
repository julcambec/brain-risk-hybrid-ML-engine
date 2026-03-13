"""
Top-level preprocessing orchestrator.

Coordinates the full preprocessing pipeline from raw data (or synthetic
stand-ins) through to analysis-ready outputs for both the **traditional ML**
track (FreeSurfer ROI feature tables) and the **deep learning** track
(standardized 64³ volumetric arrays).

Pipeline structure
------------------
::

    Shared core
    ├─ NIfTI ingestion / validation  (or synthetic data generation)
    └─ FreeSurfer recon-all          (or demo mock outputs)
         │
         ├── Traditional ML branch
         │   └─ ROI table extraction → schema validation → [harmonization hook]
         │
         └── Deep Learning branch
             └─ Volume standardization (warp → crop → normalize → pad → resample → reorient)

All paths are configurable via a YAML config file (see ``configs/preprocessing.yaml``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from brainrisk.data.synthetic import (
    generate_clinical_data,
    generate_labels,
    generate_roi_features,
    generate_volumetric_data,
)
from brainrisk.preprocessing.freesurfer import generate_mock_outputs
from brainrisk.preprocessing.nifti_io import load_npy_as_volume, validate_volume
from brainrisk.preprocessing.normalization import validate_normalized
from brainrisk.preprocessing.roi_extraction import validate_roi_schema
from brainrisk.preprocessing.volume_standardization import standardize_volume
from brainrisk.utils.config import load_config

logger = logging.getLogger(__name__)


# ────────────
# Shared core
# ────────────


def _generate_synthetic_inputs(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """
    Generate synthetic data as a stand-in for real NIfTI + FreeSurfer inputs.

    Returns
    -------
    dict[str, Path]
        Mapping with keys ``"roi"``, ``"labels"``, ``"clinical"``,
        ``"volumes_dir"``, and ``"manifest"``.
    """
    seed: int = config.get("seed", 42)
    n_subjects: int = config.get("n_subjects", 50)
    n_features: int = config.get("n_features", 400)
    n_sites: int = config.get("n_sites", 3)
    volume_shape = cast(tuple[int, int, int], tuple(config.get("volume_shape", (64, 64, 64))))

    logger.info("Generating synthetic inputs (n=%d, seed=%d) ...", n_subjects, seed)

    generate_roi_features(
        n_subjects=n_subjects,
        n_features=n_features,
        n_sites=n_sites,
        seed=seed,
        output_dir=output_dir,
    )
    labels_df = generate_labels(
        n_subjects=n_subjects,
        seed=seed,
        output_dir=output_dir,
    )
    generate_clinical_data(
        n_subjects=n_subjects,
        n_sites=n_sites,
        seed=seed,
        output_dir=output_dir,
    )
    generate_volumetric_data(
        n_subjects=n_subjects,
        shape=volume_shape,
        seed=seed,
        output_dir=output_dir,
        labels=labels_df,
    )

    return {
        "roi": output_dir / "roi" / "features.csv",
        "labels": output_dir / "labels" / "subtype_labels.csv",
        "clinical": output_dir / "labels" / "clinical.csv",
        "volumes_dir": output_dir / "volumes",
        "manifest": output_dir / "volumes" / "manifest.csv",
    }


def _run_freesurfer_demo(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """
    Run demo-mode FreeSurfer mock outputs for a small number of subjects.

    In demo mode the real FreeSurfer ``recon-all`` is not called; instead,
    ``generate_mock_outputs`` materializes synthetic volumes and ROI stats
    in the expected FreeSurfer directory layout.

    Returns
    -------
    dict[str, dict[str, Path]]
        ``{subject_id: {"brainmask": Path, "roi_stats": Path}, ...}``
    """
    seed: int = config.get("seed", 42)
    n_demo_subjects: int = config.get("n_demo_subjects", 3)
    volume_shape = cast(tuple[int, int, int], tuple(config.get("volume_shape", (64, 64, 64))))

    fs_dir = output_dir / "freesurfer"
    results: dict[str, dict[str, Path]] = {}

    logger.info("Generating %d FreeSurfer demo subjects ...", n_demo_subjects)
    for i in range(1, n_demo_subjects + 1):
        subject_id = f"sub-demo-{i:03d}"
        outputs = generate_mock_outputs(
            subject_id=subject_id,
            output_dir=fs_dir,
            volume_shape=volume_shape,
            seed=seed + i,
        )
        results[subject_id] = outputs

    return results


# ──────────────────────
# Traditional ML branch
# ──────────────────────


def _run_roi_branch(
    synthetic_paths: dict[str, Path],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract, validate, and write the ROI feature table.

    Returns a QC summary dict for inclusion in the pipeline report.
    """
    roi_output_dir = output_dir / "roi"
    roi_output_dir.mkdir(parents=True, exist_ok=True)

    # Load the synthetic ROI features (or build from FreeSurfer stats)
    roi_source = synthetic_paths["roi"]
    roi_df = pd.read_csv(str(roi_source))

    # Schema validation
    expected_n = config.get("n_features", None)
    schema_warnings = validate_roi_schema(roi_df, expected_n_features=expected_n)
    if schema_warnings:
        for w in schema_warnings:
            logger.warning("ROI schema: %s", w)

    # Write the validated table
    out_path = roi_output_dir / "features.csv"
    roi_df.to_csv(out_path, index=False)
    logger.info("ROI features written → %s  (%d subjects × %d cols)", out_path, *roi_df.shape)

    return {
        "roi_shape": list(roi_df.shape),
        "roi_path": str(out_path),
        "schema_warnings": schema_warnings,
    }


# ─────────────────────
# Deep Learning branch
# ─────────────────────


def _run_dl_branch(
    synthetic_paths: dict[str, Path],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Standardize volumes through the DL pipeline and write outputs.

    Returns a QC summary dict for inclusion in the pipeline report.
    """
    dl_output_dir = output_dir / "dl"
    dl_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = synthetic_paths["manifest"]
    manifest_df = pd.read_csv(str(manifest_path))

    dl_config: dict[str, Any] = config.get("dl_branch", {})
    target_shape = cast(tuple[int, int, int], tuple(dl_config.get("target_shape", (64, 64, 64))))

    records: list[dict[str, Any]] = []
    volume_warnings: list[dict[str, Any]] = []

    for _, row in manifest_df.iterrows():
        subject_id = str(row["subject_id"])
        volume_path = str(row["volume_path"])

        # Load
        volume, affine = load_npy_as_volume(volume_path)
        load_warnings = validate_volume(volume, affine)
        if load_warnings:
            volume_warnings.append({"subject_id": subject_id, "warnings": load_warnings})
            for w in load_warnings:
                logger.warning("Volume QC [%s]: %s", subject_id, w)

        # Standardize
        standardized = standardize_volume(volume, dl_config)

        # Post-standardization QC
        norm_warnings = validate_normalized(standardized)
        if norm_warnings:
            volume_warnings.append({"subject_id": subject_id, "post_norm_warnings": norm_warnings})

        # Save
        out_name = f"{subject_id}_standardized.npy"
        out_path = dl_output_dir / out_name
        np.save(out_path, standardized)

        records.append(
            {
                "subject_id": subject_id,
                "hydra_subtype": int(row["hydra_subtype"]),
                "standardized_path": str(out_path.as_posix()),
                "shape": list(standardized.shape),
            }
        )

    # Write DL manifest
    dl_manifest = pd.DataFrame(records)
    dl_manifest_path = dl_output_dir / "manifest.csv"
    dl_manifest.to_csv(dl_manifest_path, index=False)
    logger.info(
        "DL volumes written → %s  (%d subjects, target %s)",
        dl_output_dir,
        len(records),
        target_shape,
    )

    return {
        "n_subjects": len(records),
        "target_shape": list(target_shape),
        "dl_manifest_path": str(dl_manifest_path),
        "volume_warnings": volume_warnings,
    }


# ──────────
# QC report
# ──────────


def _write_qc_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Serialize the QC report as JSON."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "preprocessing_qc.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("QC report written → %s", report_path)
    return report_path


# ───────────
# Public API
# ───────────


def run_pipeline(
    config_path: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Execute the full preprocessing pipeline.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a YAML config file.  If ``None``, built-in defaults are used.
    config_overrides : dict | None
        Key-value overrides applied on top of the loaded config.
    output_dir : str | Path | None
        Root directory for all outputs.  Overrides ``output_dir`` in the
        config if provided.

    Returns
    -------
    dict[str, Any]
        QC report summarizing both branches.
    """
    # ── Load configuration ──
    config: dict[str, Any] = {}
    if config_path is not None:
        config = load_config(config_path)
    if config_overrides:
        config.update(config_overrides)

    resolved_output_dir = Path(output_dir or config.get("output_dir", "artifacts"))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared core: generate or load inputs ──
    mode: str = config.get("mode", "demo")

    if mode == "demo":
        synthetic_paths = _generate_synthetic_inputs(config, resolved_output_dir / "synthetic")

        # Also demonstrate FreeSurfer demo outputs (separate from the
        # synthetic branch, to exercise the FreeSurfer adapter code).
        fs_results = _run_freesurfer_demo(config, resolved_output_dir)
        fs_summary = {
            sid: {k: str(v) for k, v in paths.items()} for sid, paths in fs_results.items()
        }
    else:
        raise NotImplementedError(
            f"Pipeline mode '{mode}' is not yet implemented. Use mode='demo'."
        )

    # ── Traditional ML branch ──
    roi_qc = _run_roi_branch(synthetic_paths, resolved_output_dir, config)

    # ── Deep Learning branch ──
    dl_qc = _run_dl_branch(synthetic_paths, resolved_output_dir, config)

    # ── QC report ──
    report: dict[str, Any] = {
        "mode": mode,
        "config": {k: v for k, v in config.items() if k != "output_dir"},
        "freesurfer_demo": fs_summary if mode == "demo" else {},
        "roi_branch": roi_qc,
        "dl_branch": dl_qc,
    }
    report_path = _write_qc_report(report, resolved_output_dir)
    report["report_path"] = str(report_path)

    logger.info("Pipeline complete.")
    return report
