"""
DL-branch volume standardization orchestrator.

Chains the individual preprocessing steps into a single callable that
transforms a raw or MNI305-warped brain volume into a fixed-size,
normalized, canonically oriented array ready for deep-learning ingestion.

The full pipeline mirrors my original preprocessing workflow::

    load → brain-mask → tight-crop → min–max normalize
        → centered-pad → resample to 64³ → reorient (RAS-like)

Each step delegates to an existing preprocessing module so that every
operation remains independently testable and configurable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from brainrisk.preprocessing.mni305 import demo_affine_warp
from brainrisk.preprocessing.normalization import minmax_normalize
from brainrisk.preprocessing.orientation import reorient_volume
from brainrisk.preprocessing.resampling import resample_to_shape


def _build_brain_mask(volume: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Create a boolean brain mask by thresholding voxel intensities.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume.
    threshold : float
        Voxels with intensity above this value are considered brain.

    Returns
    -------
    np.ndarray
        Boolean mask with the same shape as *volume*.
    """
    return volume > threshold


def _tight_crop(
    volume: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a volume to the bounding box of a brain mask.

    Parameters
    ----------
    volume : np.ndarray
        3-D volume.
    mask : np.ndarray
        Boolean brain mask (same shape as *volume*).

    Returns
    -------
    cropped : np.ndarray
        Cropped volume.
    brain_dims : np.ndarray
        Per-axis size of the bounding box (length-3 integer array).
    """
    coords = np.array(np.where(mask))
    min_c = coords.min(axis=1)
    max_c = coords.max(axis=1) + 1
    cropped = volume[min_c[0] : max_c[0], min_c[1] : max_c[1], min_c[2] : max_c[2]]
    brain_dims = max_c - min_c
    return cropped, brain_dims


def _centered_pad(
    cropped: np.ndarray,
    brain_dims: np.ndarray,
    occupancy: float = 0.97,
) -> np.ndarray:
    """
    Pad a cropped volume to a near-cubic field of view.

    The target side length is chosen so the brain occupies approximately
    *occupancy* of the cube along its longest axis.

    Parameters
    ----------
    cropped : np.ndarray
        Tight-cropped volume.
    brain_dims : np.ndarray
        Per-axis bounding-box sizes (from :func:`_tight_crop`).
    occupancy : float
        Fraction of the cube's side length the brain should fill.

    Returns
    -------
    np.ndarray
        Zero-padded volume with equal dimensions along each axis.
    """
    max_dim = int(brain_dims.max())
    target_side = int(np.ceil(max_dim / occupancy))
    pad_total = target_side - brain_dims
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = np.pad(
        cropped,
        [
            (int(pad_before[0]), int(pad_after[0])),
            (int(pad_before[1]), int(pad_after[1])),
            (int(pad_before[2]), int(pad_after[2])),
        ],
        mode="constant",
        constant_values=0,
    )
    return padded


def standardize_volume(
    volume: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Run the full DL-branch standardization pipeline on a single volume.

    The pipeline steps are:

    1. **Affine warp** (demo mode applies a small perturbation; real mode
       uses the Talairach transform to MNI305).
    2. **Brain masking** via intensity thresholding.
    3. **Tight cropping** to the mask bounding box.
    4. **Min–max normalization** of cropped voxels to [0, 1].
    5. **Centered padding** to a near-cubic field of view.
    6. **Resampling** to a fixed isotropic grid (default 64³).
    7. **Reorientation** to a canonical axis order (RAS-like).

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume (float32).
    config : dict[str, Any]
        Pipeline configuration. Relevant keys (all optional, with defaults):

        - ``target_shape`` – output voxel grid, default ``(64, 64, 64)``.
        - ``mask_threshold`` – intensity threshold for brain masking,
          default ``0.05``.
        - ``occupancy`` – brain-to-cube ratio for padding, default ``0.97``.
        - ``resample_order`` – spline interpolation order, default ``1``.
        - ``apply_warp`` – whether to run the affine-warp step,
          default ``True``.
        - ``warp_seed`` – random seed for the demo affine warp,
          default ``42``.

    Returns
    -------
    np.ndarray
        Standardized float32 volume with shape *target_shape*.

    Raises
    ------
    ValueError
        If the brain mask is empty or normalization fails.
    """
    target_shape: tuple[int, int, int] = tuple(config.get("target_shape", (64, 64, 64)))
    mask_threshold: float = config.get("mask_threshold", 0.05)
    occupancy: float = config.get("occupancy", 0.97)
    resample_order: int = config.get("resample_order", 1)
    apply_warp: bool = config.get("apply_warp", True)
    warp_seed: int = config.get("warp_seed", 42)

    vol = volume.astype(np.float32, copy=True)

    # Step 1: Affine warp (demo mode — small perturbation)
    if apply_warp:
        vol = demo_affine_warp(vol, seed=warp_seed)

    # Step 2: Brain mask
    mask = _build_brain_mask(vol, threshold=mask_threshold)
    if not mask.any():
        raise ValueError("Brain mask is empty after thresholding. Adjust mask_threshold.")

    # Step 3: Tight crop
    cropped, brain_dims = _tight_crop(vol, mask)

    # Step 4: Min–max normalize (on the cropped volume, brain voxels only)
    cropped_mask = _build_brain_mask(cropped, threshold=mask_threshold)
    cropped = minmax_normalize(cropped, mask=cropped_mask)

    # Step 5: Centered pad
    padded = _centered_pad(cropped, brain_dims, occupancy=occupancy)

    # Step 6: Resample to target shape
    resampled = resample_to_shape(padded, target_shape=target_shape, order=resample_order)

    # Step 7: Reorient to canonical orientation
    standardized = reorient_volume(resampled, source="raw", target="RAS")

    return standardized
