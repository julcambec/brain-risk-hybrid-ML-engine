"""
Volume reorientation / canonicalization utilities.

Provides axis transposition and flipping to bring volumes into a canonical
orientation. The default transform replicates the reorientation applied in
my original MNI305 preprocessing script:

    vol = vol.transpose((0, 2, 1))   # swap Y <-> Z
    vol = vol[:, ::-1, :]            # flip new-Y
    vol = vol[::-1, ::-1, ::-1]      # flip all axes (180-degree rotation)
"""

from __future__ import annotations

import numpy as np


def get_reorientation_transform(
    source: str = "raw",
    target: str = "RAS",
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Return the transpose permutation and per-axis flip flags for a known transform.

    Currently supports one named transform — the specific reorientation used
    in my original MNI305 preprocessing script (``source="raw"``,
    ``target="RAS"``).  Additional named transforms can be added as needed.

    Parameters
    ----------
    source : str
        Source orientation label.
    target : str
        Target orientation label.

    Returns
    -------
    axes : tuple[int, ...]
        Axis permutation for ``np.transpose``.
    flips : tuple[bool, ...]
        Per-axis flip flags (``True`` = flip that axis).

    Raises
    ------
    NotImplementedError
        If the requested transform is not registered.
    """
    if source == "raw" and target == "RAS":
        # Matches my original script:
        #   transpose (0,2,1)  → swap Y<->Z
        #   flip axis 1        → correct new-Y
        #   flip all axes       → 180-degree in-plane rotation
        # Net effect after combining:
        axes = (0, 2, 1)
        flips = (True, False, True)  # axes 0 and 2 end up flipped
        return axes, flips

    raise NotImplementedError(
        f"No registered reorientation from '{source}' to '{target}'. "
        "Add the mapping in get_reorientation_transform() if needed."
    )


def reorient_volume(
    volume: np.ndarray,
    source: str = "raw",
    target: str = "RAS",
) -> np.ndarray:
    """Reorient a 3-D volume using transpose and flip operations.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume.
    source : str
        Source orientation label (default ``"raw"``).
    target : str
        Target orientation label (default ``"RAS"``).

    Returns
    -------
    np.ndarray
        Reoriented volume (contiguous float32 copy).
    """
    axes, flips = get_reorientation_transform(source, target)
    vol = volume.transpose(axes)
    for axis, should_flip in enumerate(flips):
        if should_flip:
            vol = np.flip(vol, axis=axis)
    return np.ascontiguousarray(vol, dtype=np.float32)
