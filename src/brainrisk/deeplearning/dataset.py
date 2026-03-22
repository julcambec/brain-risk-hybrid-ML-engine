"""
PyTorch dataset for volumetric MRI classification.

Loads pre-processed 3D volumes (as ``.npy`` files) paired with integer
class labels. Designed to consume outputs from the ``brainrisk``
preprocessing pipeline or the synthetic data generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VolumeClassificationDataset(Dataset):
    """
    Dataset of 3D brain volumes and integer class labels.

    Each sample is a ``(volume, label)`` pair where:

    - ``volume`` is a float32 tensor of shape ``(1, L, W, H)``
      (single-channel, matching the model's expected input format).
    - ``label`` is a long tensor containing the 0-indexed class index.

    Parameters
    ----------
    volume_paths : list[str]
        File paths to ``.npy`` volumes. Each file should contain a 3-D
        float32 array of shape ``(L, W, H)``.
    labels : list[int]
        Integer class labels, one per volume. Must be 0-indexed
        (i.e. values in ``{0, 1, ..., C-1}``).
    """

    def __init__(
        self,
        volume_paths: list[str],
        labels: list[int],
    ) -> None:
        if len(volume_paths) != len(labels):
            raise ValueError(f"Mismatch: {len(volume_paths)} volume paths vs {len(labels)} labels.")
        self.volume_paths = volume_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single (volume, label) pair.

        The volume is loaded from disk on each access (no in-memory caching)
        to keep memory usage manageable for large datasets.

        Returns
        -------
        volume : torch.Tensor
            Shape ``(1, L, W, H)``, dtype float32.
        label : torch.Tensor
            Scalar long tensor with the class index.
        """
        volume = np.load(self.volume_paths[idx]).astype(np.float32)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, L, W, H)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return volume_tensor, label_tensor


def build_datasets_from_config(
    config: dict,
) -> tuple[VolumeClassificationDataset, VolumeClassificationDataset]:
    """
    Build train and validation datasets from a training configuration.

    Loads the volume manifest and clinical data, encodes the target labels,
    performs a subject-level split, and returns two dataset objects.

    Parameters
    ----------
    config : dict
        Full training configuration. Expected keys under ``data``::

            manifest_path : str   —> path to volumes manifest CSV
            clinical_path : str   —> path to clinical variables CSV
            label_column  : str   —> target column name
            test_size     : float —> fraction of subjects for validation

    Returns
    -------
    train_dataset : VolumeClassificationDataset
    val_dataset : VolumeClassificationDataset
    """
    import pandas as pd

    from brainrisk.data.splits import subject_split

    data_cfg = config.get("data", {})
    manifest_path = data_cfg["manifest_path"]
    clinical_path = data_cfg["clinical_path"]
    label_column = data_cfg.get("label_column", "hydra_subtype")
    test_size = data_cfg.get("test_size", 0.2)
    seed = config.get("seed", 42)

    # Load and merge manifest with clinical data
    manifest_df = pd.read_csv(manifest_path)
    clinical_df = pd.read_csv(clinical_path)
    merged = manifest_df.merge(clinical_df, on="subject_id", suffixes=("", "_clinical"))

    # Determine volume path column (handle both raw and standardized manifests)
    if "standardized_path" in merged.columns:
        path_col = "standardized_path"
    elif "volume_path" in merged.columns:
        path_col = "volume_path"
    else:
        raise KeyError("Manifest must contain 'volume_path' or 'standardized_path'.")

    # Encode labels to 0-indexed integers
    encoded_col = _encode_labels(merged, label_column)

    # Subject-level split
    train_df, val_df = subject_split(
        merged,
        test_size=test_size,
        stratify_col=encoded_col,
        seed=seed,
    )

    train_dataset = VolumeClassificationDataset(
        volume_paths=train_df[path_col].tolist(),
        labels=train_df[encoded_col].tolist(),
    )
    val_dataset = VolumeClassificationDataset(
        volume_paths=val_df[path_col].tolist(),
        labels=val_df[encoded_col].tolist(),
    )

    return train_dataset, val_dataset


def _encode_labels(df: pd.DataFrame, label_column: str) -> str:
    """
    Encode a label column to 0-indexed integers, adding a new column.

    For ``sex``: maps ``"M"`` → 1, ``"F"`` → 0.
    For ``hydra_subtype``: subtracts 1 (paper convention is 1-indexed).
    For other columns: assumes already integer and subtracts the minimum.

    Returns the name of the new encoded column.
    """
    encoded_col = f"{label_column}_encoded"

    if label_column == "sex":
        df[encoded_col] = (df["sex"] == "M").astype(int)
    elif label_column == "hydra_subtype":
        # HYDRA labels are 1-indexed (1, 2, 3) → convert to 0-indexed
        df[encoded_col] = df["hydra_subtype"].astype(int) - 1
    else:
        vals = df[label_column].astype(int)
        df[encoded_col] = vals - vals.min()

    return encoded_col
