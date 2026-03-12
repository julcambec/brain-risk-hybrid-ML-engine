"""
Intensity normalization for volumetric MRI data.

Provides min-max and z-score normalization with optional brain-mask support.
Based on the normalization logic in my original MNI305 preprocessing script
(``(cropped - vmin) / (vmax - vmin)``).
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Scale brain-voxel intensities to [0, 1], preserving background at zero.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of voxel intensities.
    mask : np.ndarray | None
        Boolean mask indicating brain voxels. If ``None``, voxels with
        intensity > 0 are treated as brain.

    Returns
    -------
    np.ndarray
        Normalized float32 volume.

    Raises
    ------
    ValueError
        If the brain region has zero intensity range.
    """
    volume = volume.astype(np.float32, copy=True)
    if mask is None:
        mask = volume > 0

    brain_vals = volume[mask]
    if brain_vals.size == 0:
        return volume

    vmin = float(brain_vals.min())
    vmax = float(brain_vals.max())
    if vmax - vmin < 1e-8:
        raise ValueError("Zero intensity range within brain mask; cannot normalize.")

    volume[mask] = (volume[mask] - vmin) / (vmax - vmin)
    volume[~mask] = 0.0
    return volume


def zscore_normalize(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Z-score normalize brain-voxel intensities, preserving background at zero.
    Note: My original implementation only used minmax.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of voxel intensities.
    mask : np.ndarray | None
        Boolean mask indicating brain voxels. If ``None``, voxels with
        intensity > 0 are treated as brain.

    Returns
    -------
    np.ndarray
        Z-score normalized float32 volume (background stays 0).

    Raises
    ------
    ValueError
        If the brain region has zero standard deviation.
    """
    volume = volume.astype(np.float32, copy=True)
    if mask is None:
        mask = volume > 0

    brain_vals = volume[mask]
    if brain_vals.size == 0:
        return volume

    std = float(brain_vals.std())
    if std < 1e-8:
        raise ValueError("Zero standard deviation within brain mask; cannot z-score normalize.")

    mean = float(brain_vals.mean())
    volume[mask] = (volume[mask] - mean) / std
    volume[~mask] = 0.0
    return volume


def validate_normalized(volume: np.ndarray) -> list[str]:
    """
    Check that a min-max normalized volume has expected properties.

    Parameters
    ----------
    volume : np.ndarray
        Volume expected to be in [0, 1] range after min-max normalization.

    Returns
    -------
    list[str]
        Warning messages. An empty list means all checks passed.
    """
    warnings: list[str] = []

    if np.any(np.isnan(volume)):
        warnings.append("Normalized volume contains NaN values")

    if float(volume.min()) < -1e-6:
        warnings.append(f"Normalized volume has negative values (min={volume.min():.4f})")

    if float(volume.max()) > 1.0 + 1e-6:
        warnings.append(f"Normalized volume exceeds 1.0 (max={volume.max():.4f})")

    if float(volume.sum()) < 1e-8:
        warnings.append("Normalized volume has no brain content (all zeros)")

    return warnings
