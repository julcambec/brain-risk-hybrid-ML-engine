"""Affine warp to MNI305 space via Talairach transform.

In my original preprocessing pipeline, FreeSurfer's ``mri_vol2vol`` applies
the subject-specific Talairach affine (``talairach.xfm.lta``) to register
the skull-stripped brainmask into MNI305 space.  This module reimplements
the affine application in pure Python / SciPy so the pipeline can run without
FreeSurfer installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform


def load_talairach_lta(lta_path: str | Path) -> np.ndarray:
    """Parse a FreeSurfer ``.lta`` file and extract the 4x4 affine matrix.

    The ``.lta`` (Linear Transform Array) format stores one or more affine
    matrices. This function reads the first ``1 4 4`` block.

    Parameters
    ----------
    lta_path : str | Path
        Path to a FreeSurfer ``.lta`` file.

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix.

    Raises
    ------
    ValueError
        If the file does not contain a recognizable 4x4 matrix block.
    """
    lines = Path(lta_path).read_text().splitlines()
    matrix_lines: list[str] = []
    reading = False

    for line in lines:
        stripped = line.strip()
        if stripped == "1 4 4":
            reading = True
            continue
        if reading:
            parts = stripped.split()
            if len(parts) == 4:
                matrix_lines.append(stripped)
            if len(matrix_lines) == 4:
                break

    if len(matrix_lines) != 4:
        raise ValueError(f"Could not find a 4x4 matrix block in {lta_path}")

    matrix = np.array(
        [[float(x) for x in row.split()] for row in matrix_lines],
        dtype=np.float64,
    )
    return matrix


def apply_affine_warp(
    volume: np.ndarray,
    transform: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Apply a 4x4 affine transform to resample a volume into target space.

    Uses ``scipy.ndimage.affine_transform`` which expects the *inverse*
    mapping (target → source).  The provided *transform* is assumed to be
    source → target, so it is inverted internally.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume.
    transform : np.ndarray
        4x4 affine matrix (source → target).
    target_shape : tuple[int, int, int]
        Shape of the output volume.
    order : int
        Spline interpolation order (default 1 = trilinear).

    Returns
    -------
    np.ndarray
        Warped float32 volume with shape *target_shape*.
    """
    inv = np.linalg.inv(transform)
    warped: np.ndarray = affine_transform(
        volume.astype(np.float32),
        matrix=inv[:3, :3],
        offset=inv[:3, 3],
        output_shape=target_shape,
        order=order,
    )
    return warped


def warp_to_mni305(
    volume: np.ndarray,
    source_affine: np.ndarray,
    talairach_matrix: np.ndarray,
    target_shape: tuple[int, int, int] = (256, 256, 256),
) -> np.ndarray:
    """Warp a volume into MNI305 space using a Talairach affine.

    Composes the voxel-to-scanner affine with the Talairach matrix and
    applies the combined transform.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume in native space.
    source_affine : np.ndarray
        4x4 voxel-to-scanner affine of the input volume.
    talairach_matrix : np.ndarray
        4x4 Talairach affine (scanner → MNI305).
    target_shape : tuple[int, int, int]
        Output shape (default 256^3 to match MNI305 grid).

    Returns
    -------
    np.ndarray
        Volume in MNI305 space.
    """
    combined = talairach_matrix @ source_affine
    return apply_affine_warp(volume, combined, target_shape)


def demo_affine_warp(
    volume: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Apply a small, plausible affine perturbation for demo purposes.

    Simulates the effect of a Talairach warp without requiring actual
    FreeSurfer outputs. Applies a slight rotation (~2 degrees) and
    translation (~3 voxels) then resamples back to the original shape.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Perturbed float32 volume with the same shape as input.
    """
    rng = np.random.default_rng(seed)

    # Small rotation angles (radians) around each axis
    angles = rng.uniform(-0.035, 0.035, size=3)  # ~2 degrees max
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = rot_x @ rot_y @ rot_z

    translation = rng.uniform(-3.0, 3.0, size=3)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return apply_affine_warp(volume, transform, volume.shape)
