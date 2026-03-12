"""Volume resampling to a fixed isotropic grid.

Refactored from ``scipy.ndimage.zoom`` calls in my original preprocessing
script. Provides trilinear (order=1) resampling to a target shape such as
64x64x64.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def compute_zoom_factors(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[float, ...]:
    """Compute per-axis zoom factors to go from *source_shape* to *target_shape*.

    Parameters
    ----------
    source_shape : tuple[int, ...]
        Shape of the input array.
    target_shape : tuple[int, ...]
        Desired output shape.

    Returns
    -------
    tuple[float, ...]
        Zoom factor for each axis.

    Raises
    ------
    ValueError
        If the shapes have different numbers of dimensions.
    """
    if len(source_shape) != len(target_shape):
        raise ValueError(
            f"Dimension mismatch: source has {len(source_shape)}D, target has {len(target_shape)}D"
        )
    return tuple(t / s for s, t in zip(source_shape, target_shape, strict=True))


def resample_to_shape(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3-D volume to *target_shape* via ``scipy.ndimage.zoom``.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume.
    target_shape : tuple[int, int, int]
        Desired ``(X, Y, Z)`` output shape.
    order : int
        Spline interpolation order. ``1`` = trilinear (default, matches my
        original preprocessing pipeline).

    Returns
    -------
    np.ndarray
        Resampled float32 volume with shape *target_shape*.
    """
    factors = compute_zoom_factors(volume.shape, target_shape)
    resampled: np.ndarray = zoom(volume.astype(np.float32), factors, order=order)
    return resampled
