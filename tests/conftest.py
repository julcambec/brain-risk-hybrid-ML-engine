from __future__ import annotations

from pathlib import Path

import pytest

from brainrisk.data.synthetic import (
    generate_clinical_data,
    generate_labels,
    generate_roi_features,
    generate_volumetric_data,
)


@pytest.fixture()
def temp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "synthetic_output"


@pytest.fixture()
def small_subject_count() -> int:
    return 12


@pytest.fixture()
def small_volume_shape() -> tuple[int, int, int]:
    return (16, 16, 16)


@pytest.fixture()
def synthetic_bundle(
    temp_output_dir: Path,
    small_subject_count: int,
    small_volume_shape: tuple[int, int, int],
) -> dict[str, Path]:
    generate_roi_features(
        n_subjects=small_subject_count,
        n_features=40,
        n_sites=3,
        seed=123,
        output_dir=temp_output_dir,
    )
    generate_labels(
        n_subjects=small_subject_count,
        seed=123,
        output_dir=temp_output_dir,
    )
    generate_clinical_data(
        n_subjects=small_subject_count,
        n_sites=3,
        seed=123,
        output_dir=temp_output_dir,
    )
    generate_volumetric_data(
        n_subjects=small_subject_count,
        shape=small_volume_shape,
        seed=123,
        output_dir=temp_output_dir,
    )

    return {
        "root": temp_output_dir,
        "roi": temp_output_dir / "roi" / "features.csv",
        "labels": temp_output_dir / "labels" / "subtype_labels.csv",
        "clinical": temp_output_dir / "labels" / "clinical.csv",
        "volumes_dir": temp_output_dir / "volumes",
        "manifest": temp_output_dir / "volumes" / "manifest.csv",
    }
