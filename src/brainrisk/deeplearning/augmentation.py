"""
Data augmentation for 3D volumetric classification.

Implements MixUp augmentation adapted for volumetric inputs, as described for
the MINiT training procedure. The paper references both MixUp (Zhang et al.,
2017) and CutMix (Yun et al., 2019) as run-time augmentations.

MixUp generates virtual training samples by linearly interpolating between
pairs of training volumes and their labels, which acts as a regularizer
and encourages linear behavior between training examples.
"""

from __future__ import annotations

import numpy as np
import torch


def mixup_data(
    volumes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation to a batch of volumes and labels.

    Generates mixed samples by linearly interpolating between randomly
    paired volumes::

        mixed_x = lambda * x_i + (1 - lambda) * x_j
        mixed_y = lambda * y_i + (1 - lambda) * y_j  (handled in loss)

    where ``lambda ~ Beta(alpha, alpha)``.

    Parameters
    ----------
    volumes : torch.Tensor
        Batch of volumes, shape ``(N, I, L, W, H)``.
    labels : torch.Tensor
        Integer class labels, shape ``(N,)``.
    alpha : float
        Shape parameter for the Beta distribution. Higher values produce
        more aggressive mixing. Set to 0 to disable MixUp (returns
        ``lambda = 1.0``, i.e. no mixing).
    device : torch.device | None
        Device for the shuffle index tensor.

    Returns
    -------
    mixed_volumes : torch.Tensor
        Mixed input volumes, shape ``(N, I, L, W, H)``.
    labels_a : torch.Tensor
        Original labels for the first element of each pair.
    labels_b : torch.Tensor
        Labels for the shuffled (second) element of each pair.
    lam : float
        Mixing coefficient drawn from ``Beta(alpha, alpha)``.
    """
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = volumes.size(0)
    index = torch.randperm(batch_size, device=device or volumes.device)

    mixed_volumes = lam * volumes + (1.0 - lam) * volumes[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_volumes, labels_a, labels_b, lam


def mixup_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute the MixUp loss as a convex combination of per-label losses.

    Parameters
    ----------
    criterion : torch.nn.Module
        Loss function (e.g. ``nn.CrossEntropyLoss``).
    logits : torch.Tensor
        Model output logits, shape ``(N, C)``.
    labels_a : torch.Tensor
        First set of labels.
    labels_b : torch.Tensor
        Second set of labels (from the shuffled pairing).
    lam : float
        Mixing coefficient.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
