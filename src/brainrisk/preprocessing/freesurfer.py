"""FreeSurfer ``recon-all`` orchestration wrapper.

In a production environment, FreeSurfer's ``recon-all`` performs cortical
reconstruction in several stages:

* **autorecon1** — motion correction, Talairach alignment, bias-field
  correction, skull stripping (produces ``brainmask.mgz``).
* **autorecon2** — white/pial surface tessellation, inflation, registration.
* **autorecon3** — cortical parcellation (Desikan–Killiany atlas), thickness
  and area statistics.
* **-all** — runs all three stages end-to-end.

This module provides a Python interface for invoking ``recon-all`` and
extracting its outputs.  A **demo mode** (``generate_mock_outputs``) reshapes
the outputs from ``brainrisk.data.synthetic`` into the minimal directory/file
structure that downstream FreeSurfer-oriented code expects
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from brainrisk.data.synthetic import generate_roi_features, generate_volumetric_data

# ---------------------------
# Real FreeSurfer interface
# ---------------------------


def is_freesurfer_available() -> bool:
    """Check whether ``recon-all`` is on the system PATH."""
    return shutil.which("recon-all") is not None


def run_recon_all(
    subject_id: str,
    t1_path: str | Path,
    subjects_dir: str | Path,
    flags: str = "-all",
) -> Path:
    """Run FreeSurfer ``recon-all`` for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (used as the FreeSurfer subject directory name).
    t1_path : str | Path
        Path to the input T1-weighted NIfTI file.
    subjects_dir : str | Path
        FreeSurfer ``SUBJECTS_DIR`` (where output directories are written).
    flags : str
        ``recon-all`` flags (default ``"-all"``). Use ``"-autorecon1"`` for
        skull-stripping only.

    Returns
    -------
    Path
        Path to the subject's FreeSurfer output directory.

    Raises
    ------
    RuntimeError
        If FreeSurfer is not available or the command fails.
    """
    if not is_freesurfer_available():
        raise RuntimeError(
            "FreeSurfer is not installed or not on PATH. Use generate_mock_outputs() for demo mode."
        )

    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "recon-all",
        "-i",
        str(t1_path),
        "-s",
        subject_id,
        "-sd",
        str(subjects_dir),
        *flags.split(),
    ]
    subprocess.run(cmd, check=True)
    return subjects_dir / subject_id


def convert_brainmask(
    subjects_dir: str | Path,
    subject_id: str,
    output_dir: str | Path,
) -> Path:
    """Convert ``brainmask.mgz`` to NIfTI using ``mri_convert``.

    Parameters
    ----------
    subjects_dir : str | Path
        FreeSurfer ``SUBJECTS_DIR``.
    subject_id : str
        Subject identifier.
    output_dir : str | Path
        Directory for the output ``.nii`` file.

    Returns
    -------
    Path
        Path to the converted NIfTI file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mgz_path = Path(subjects_dir) / subject_id / "mri" / "brainmask.mgz"
    out_path = output_dir / f"{subject_id}_brainmask.nii"

    subprocess.run(
        ["mri_convert", str(mgz_path), str(out_path)],
        check=True,
    )
    return out_path


# ------------------
# Demo / mock mode
# ------------------


def generate_mock_outputs(
    subject_id: str,
    output_dir: str | Path,
    volume_shape: tuple[int, int, int] = (256, 256, 256),
    seed: int = 42,
    *,
    n_features: int = 400,
    n_sites: int = 1,
) -> dict[str, Path]:
    """Materialize synthetic artifacts in a FreeSurfer-like folder layout.

    This function is intentionally an *adapter*, not a synthetic data
    generator. It reuses the shared synthetic generators in
    ``brainrisk.data.synthetic`` and reshapes their outputs into the minimal
    directory/file structure that downstream FreeSurfer-oriented code expects.

    Output layout
    -------------
    <output_dir>/<subject_id>/
        mri/
            brainmask.npy
        stats/
            roi_stats.csv

    Parameters
    ----------
    subject_id : str
        Subject identifier to expose in the FreeSurfer-like output tree.
    output_dir : str | Path
        Root directory under which the subject directory is created.
    volume_shape : tuple[int, int, int]
        Shape of the synthetic brainmask volume.
    seed : int
        Random seed for reproducibility.
    n_features : int, keyword-only
        Number of ROI features to request from the shared synthetic generator.
    n_sites : int, keyword-only
        Number of synthetic sites to request from the shared ROI generator.

    Returns
    -------
    dict[str, Path]
        Mapping with keys ``"brainmask"`` and ``"roi_stats"``.

    Notes
    -----
    The shared synthetic generators auto-create canonical subject IDs
    (e.g. ``sub-00001``). For demo-mode FreeSurfer compatibility, this adapter
    stages one synthetic subject in a temporary directory and rewrites the
    public-facing ``subject_id`` in the copied outputs.
    """
    output_dir = Path(output_dir)
    subject_dir = output_dir / subject_id
    mri_dir = subject_dir / "mri"
    stats_dir = subject_dir / "stats"

    mri_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="brainrisk_mock_fs_") as tmpdir:
        staging_dir = Path(tmpdir)

        # Generate one synthetic subject using the shared generators.
        manifest_df = generate_volumetric_data(
            n_subjects=1,
            shape=volume_shape,
            seed=seed,
            output_dir=staging_dir,
        )
        roi_df = generate_roi_features(
            n_subjects=1,
            n_features=n_features,
            n_sites=n_sites,
            seed=seed,
            output_dir=staging_dir,
        )

        staged_subject_id = str(manifest_df.loc[0, "subject_id"])
        staged_volume_path = Path(str(manifest_df.loc[0, "volume_path"]))

        if not staged_volume_path.exists():
            raise FileNotFoundError(
                f"Expected staged synthetic volume at {staged_volume_path}, but it was not found."
            )

        # Copy the staged volume into a FreeSurfer-like MRI location.
        brainmask_path = mri_dir / "brainmask.npy"
        shutil.copy2(staged_volume_path, brainmask_path)

        # Extract the matching ROI row, rewrite subject_id, and save it into
        # a FreeSurfer-like stats location.
        subject_roi_df = roi_df.loc[roi_df["subject_id"] == staged_subject_id].copy()
        if subject_roi_df.empty:
            raise ValueError(
                f"No ROI features found for staged synthetic subject {staged_subject_id}."
            )

        subject_roi_df.loc[:, "subject_id"] = subject_id
        roi_stats_path = stats_dir / "roi_stats.csv"
        subject_roi_df.to_csv(roi_stats_path, index=False)

    return {"brainmask": brainmask_path, "roi_stats": roi_stats_path}
