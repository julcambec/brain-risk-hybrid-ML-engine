from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SUBTYPE_PROPORTIONS = np.array([0.34, 0.36, 0.30], dtype=float)

CORTICAL_REGIONS = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
]

SUBCORTICAL_STRUCTURES = [
    "Thalamus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Hippocampus",
    "Amygdala",
    "Accumbens-area",
]


FAMILY_PARAMS: dict[str, dict[str, float]] = {
    "thickness": {"mean": 2.7, "scale": 0.25, "min": 1.4, "max": 4.7},
    "area": {"mean": 2400.0, "scale": 350.0, "min": 500.0, "max": 5000.0},
    "volume": {"mean": 5500.0, "scale": 1200.0, "min": 200.0, "max": 20000.0},
    "gwc": {"mean": 0.32, "scale": 0.04, "min": 0.12, "max": 0.60},
    "ndi": {"mean": 0.46, "scale": 0.06, "min": 0.12, "max": 0.85},
    "curv": {"mean": 0.0, "scale": 0.08, "min": -0.35, "max": 0.35},
    "generic": {"mean": 0.50, "scale": 0.12, "min": 0.0, "max": 1.0},
}


def _subject_ids(n_subjects: int) -> list[str]:
    if n_subjects <= 0:
        raise ValueError("n_subjects must be a positive integer.")
    return [f"sub-{idx:05d}" for idx in range(1, n_subjects + 1)]


def _ensure_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_subtypes(n_subjects: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 23)
    counts = rng.multinomial(n_subjects, DEFAULT_SUBTYPE_PROPORTIONS)
    labels = np.concatenate(
        [np.full(count, subtype, dtype=int) for subtype, count in enumerate(counts, start=1)]
    )
    rng.shuffle(labels)
    return labels


def _sample_sites(n_subjects: int, n_sites: int, seed: int) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("n_sites must be a positive integer.")
    rng = np.random.default_rng(seed + 17)
    site_idx = rng.integers(0, n_sites, size=n_subjects)
    return np.array([f"site_{idx + 1:02d}" for idx in site_idx], dtype=object)


def _feature_schema(n_features: int) -> list[tuple[str, str]]:
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer.")

    schema: list[tuple[str, str]] = []

    for measure in ["thickness", "area", "gwc", "ndi", "curv"]:
        for hemi in ["lh", "rh"]:
            for region in CORTICAL_REGIONS:
                schema.append((f"{hemi}_{region}_{measure}", measure))

    for hemi in ["Left", "Right"]:
        for structure in SUBCORTICAL_STRUCTURES:
            schema.append((f"{hemi}-{structure}_volume", "volume"))

    extra_idx = 1
    while len(schema) < n_features:
        schema.append((f"morphometry_feature_{extra_idx:03d}", "generic"))
        extra_idx += 1

    return schema[:n_features]


def generate_labels(
    n_subjects: int,
    seed: int,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Generate synthetic HYDRA-like subtype labels and write them to disk.
    """
    subject_ids = _subject_ids(n_subjects)
    labels = _sample_subtypes(n_subjects=n_subjects, seed=seed)

    labels_dir = _ensure_dir(Path(output_dir) / "labels")
    labels_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "hydra_subtype": labels.astype(int),
        }
    )
    labels_df.to_csv(labels_dir / "subtype_labels.csv", index=False)
    return labels_df


def generate_clinical_data(
    n_subjects: int,
    seed: int,
    output_dir: str | Path,
    n_sites: int = 3,
    labels: Sequence[int] | np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic clinical and demographic covariates with subtype-linked signal.

    If ``labels`` is provided, it is treated as the canonical subtype assignment.
    Otherwise subtype labels are generated from ``seed`` for backwards-compatible
    behavior.
    """
    rng = np.random.default_rng(seed + 31)
    subject_ids = _subject_ids(n_subjects)
    subtype = _coerce_labels(labels=labels, n_subjects=n_subjects, seed=seed)
    site = _sample_sites(n_subjects=n_subjects, n_sites=n_sites, seed=seed)

    age_years = rng.uniform(9.0, 10.99, size=n_subjects).round(2)
    sex = rng.choice(["F", "M"], size=n_subjects)
    risk_group = rng.choice(["PH+", "PH-"], size=n_subjects, p=[0.55, 0.45])

    internalizing_effect = np.select(
        [subtype == 1, subtype == 2, subtype == 3],
        [7.0, 4.0, 1.0],
        default=0.0,
    )
    externalizing_effect = np.select(
        [subtype == 1, subtype == 2, subtype == 3],
        [6.0, 3.0, 0.5],
        default=0.0,
    )
    socioeconomic_effect = np.select(
        [subtype == 1, subtype == 2, subtype == 3],
        [-5.0, -15.0, 8.0],
        default=0.0,
    )

    risk_boost = np.where(risk_group == "PH+", 3.0, 0.0)

    cbcl_internalizing = (
        50.0 + internalizing_effect + risk_boost + rng.normal(0.0, 5.0, size=n_subjects)
    ).round(2)
    cbcl_externalizing = (
        48.0 + externalizing_effect + risk_boost + rng.normal(0.0, 5.0, size=n_subjects)
    ).round(2)
    family_income_k = np.clip(
        90.0 + socioeconomic_effect + rng.normal(0.0, 20.0, size=n_subjects),
        20.0,
        250.0,
    ).round(2)
    maternal_substance_use = rng.binomial(
        n=1,
        p=np.clip(0.20 + 0.08 * (subtype == 2) + 0.05 * (risk_group == "PH+"), 0.0, 0.95),
        size=n_subjects,
    )
    peer_support = np.clip(
        3.0
        + np.select([subtype == 1, subtype == 2, subtype == 3], [-0.3, -0.4, 0.4], default=0.0)
        + rng.normal(0.0, 0.5, size=n_subjects),
        1.0,
        5.0,
    ).round(2)

    clinical_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "site": site,
            "age_years": age_years,
            "sex": sex,
            "risk_group": risk_group,
            "hydra_subtype": subtype.astype(int),
            "cbcl_internalizing": cbcl_internalizing,
            "cbcl_externalizing": cbcl_externalizing,
            "family_income_k": family_income_k,
            "maternal_substance_use": maternal_substance_use.astype(int),
            "peer_support": peer_support,
        }
    )

    labels_dir = _ensure_dir(Path(output_dir) / "labels")
    clinical_df.to_csv(labels_dir / "clinical.csv", index=False)
    return clinical_df


def generate_roi_features(
    n_subjects: int,
    n_features: int,
    n_sites: int,
    seed: int,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Generate a FreeSurfer-like ROI feature table with plausible ranges and site effects.
    """
    rng = np.random.default_rng(seed + 47)
    subject_ids = _subject_ids(n_subjects)
    site = _sample_sites(n_subjects=n_subjects, n_sites=n_sites, seed=seed)
    schema = _feature_schema(n_features=n_features)

    latent = rng.normal(0.0, 1.0, size=(n_subjects, 6))
    site_codes = np.array([int(site_name.split("_")[1]) - 1 for site_name in site], dtype=int)
    site_latent = rng.normal(0.0, 0.15, size=(n_sites, 6))

    data: dict[str, object] = {
        "subject_id": subject_ids,
        "site": site,
    }

    for feature_name, family in schema:
        params = FAMILY_PARAMS[family]
        weights = rng.normal(0.0, 0.55, size=6)
        signal = latent @ weights
        site_shift = site_latent[site_codes, hash(feature_name) % 6] * params["scale"]

        values = (
            params["mean"]
            + 0.50 * params["scale"] * signal
            + site_shift
            + rng.normal(0.0, params["scale"] * 0.35, size=n_subjects)
        )
        clipped = np.clip(values, params["min"], params["max"])
        data[feature_name] = clipped.round(6)

    roi_dir = _ensure_dir(Path(output_dir) / "roi")
    roi_df = pd.DataFrame(data)
    roi_df.to_csv(roi_dir / "features.csv", index=False)
    return roi_df


def _coerce_labels(
    labels: Sequence[int] | np.ndarray | pd.Series | pd.DataFrame | None,
    n_subjects: int,
    seed: int,
) -> np.ndarray:
    if labels is None:
        return _sample_subtypes(n_subjects=n_subjects, seed=seed)

    if isinstance(labels, pd.DataFrame):
        if "hydra_subtype" not in labels.columns:
            raise ValueError("DataFrame labels must contain a 'hydra_subtype' column.")
        label_array = labels["hydra_subtype"].to_numpy()
    elif isinstance(labels, pd.Series):
        label_array = labels.to_numpy()
    else:
        label_array = np.asarray(labels)

    if len(label_array) != n_subjects:
        raise ValueError("labels length must match n_subjects.")

    return label_array.astype(int)


def generate_volumetric_data(
    n_subjects: int,
    shape: tuple[int, int, int],
    seed: int,
    output_dir: str | Path,
    labels: Sequence[int] | np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic 3D brain-like volumes and write one .npy file per subject.
    """
    if len(shape) != 3:
        raise ValueError("shape must be a 3-tuple, e.g. (64, 64, 64).")

    rng = np.random.default_rng(seed + 59)
    subject_ids = _subject_ids(n_subjects)
    subtype_labels = _coerce_labels(labels=labels, n_subjects=n_subjects, seed=seed)

    x = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    y = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
    z = np.linspace(-1.0, 1.0, shape[2], dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    brain_mask = (xx**2 / 0.85**2 + yy**2 / 0.75**2 + zz**2 / 0.65**2) <= 1.0
    subtype_shift = {1: -0.10, 2: 0.00, 3: 0.10}

    volumes_dir = _ensure_dir(Path(output_dir) / "volumes")
    records: list[dict[str, object]] = []

    for subject_id, subtype in zip(subject_ids, subtype_labels, strict=True):
        shift_x = subtype_shift[int(subtype)]

        blob_1 = np.exp(
            -((xx - shift_x) ** 2 / (2 * 0.18**2) + yy**2 / (2 * 0.22**2) + zz**2 / (2 * 0.20**2))
        )
        blob_2 = 0.80 * np.exp(
            -(
                (xx + 0.25) ** 2 / (2 * 0.16**2)
                + (yy - 0.15) ** 2 / (2 * 0.16**2)
                + (zz + 0.10) ** 2 / (2 * 0.18**2)
            )
        )
        gradient = 0.15 * (zz + 1.0)
        noise = rng.normal(0.0, 0.03, size=shape)

        volume = (blob_1 + blob_2 + gradient + noise).astype(np.float32)
        volume = np.where(brain_mask, volume, 0.0).astype(np.float32)
        volume = np.clip(volume, 0.0, None)

        brain_values = volume[brain_mask]
        denom = max(float(np.ptp(brain_values)), 1e-8)
        volume[brain_mask] = (brain_values - float(brain_values.min())) / denom
        volume = volume.astype(np.float32)

        file_name = f"{subject_id}_64iso.npy"
        file_path = volumes_dir / file_name
        np.save(file_path, volume)

        records.append(
            {
                "subject_id": subject_id,
                "hydra_subtype": int(subtype),
                "volume_path": str(file_path.as_posix()),
            }
        )

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(volumes_dir / "manifest.csv", index=False)
    return manifest_df
