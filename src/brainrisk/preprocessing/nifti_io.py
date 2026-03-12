"""NIfTI file loading, validation, and basic QC.

This module is the entry point for all raw MRI data ingestion. It handles
both NIfTI files (via nibabel) and pre-processed ``.npy`` volumes (e.g. from
the synthetic data generator or prior preprocessing steps).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file, reorient to closest canonical (RAS), and return data + affine.

    Parameters
    ----------
    path : str | Path
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    volume : np.ndarray
        3-D float32 volume in RAS-like orientation.
    affine : np.ndarray
        4x4 affine matrix mapping voxel indices to scanner (mm) coordinates.
    """
    img = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    volume = np.asarray(img_canonical.dataobj, dtype=np.float32)
    affine: np.ndarray = img_canonical.affine
    return volume, affine


def load_npy_as_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a ``.npy`` file and return it with an identity affine.

    Useful for synthetic or already-preprocessed volumes that have no NIfTI
    header information.

    Parameters
    ----------
    path : str | Path
        Path to a ``.npy`` file containing a 3-D array.

    Returns
    -------
    volume : np.ndarray
        3-D float32 volume.
    affine : np.ndarray
        4x4 identity matrix (no spatial metadata available).
    """
    volume = np.load(str(path)).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    return volume, affine


def validate_volume(volume: np.ndarray, affine: np.ndarray) -> list[str]:
    """Run QC checks on a loaded volume.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of voxel intensities.
    affine : np.ndarray
        4x4 affine matrix.

    Returns
    -------
    list[str]
        Warning messages. An empty list means all checks passed.
    """
    warnings: list[str] = []

    if volume.ndim != 3:
        warnings.append(f"Expected 3-D volume, got {volume.ndim}-D (shape {volume.shape})")

    if np.any(np.isnan(volume)):
        warnings.append("Volume contains NaN values")

    if np.any(np.isinf(volume)):
        warnings.append("Volume contains Inf values")

    intensity_range = float(volume.max()) - float(volume.min())
    if intensity_range < 1e-8:
        warnings.append(
            f"Volume has near-zero intensity range ({intensity_range:.2e}); "
            "may be empty or constant"
        )

    det = float(np.linalg.det(affine[:3, :3]))
    if abs(det) < 1e-8:
        warnings.append(f"Affine matrix is degenerate (det={det:.2e})")

    return warnings


def get_orientation(affine: np.ndarray) -> str:
    """Return the orientation code string (e.g. ``'RAS'``) from an affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine matrix.

    Returns
    -------
    str
        Three-character orientation code derived from the affine.
    """
    codes = nib.aff2axcodes(affine)
    return "".join(codes)
