"""
Utility functions for the deep learning track.

Provides reproducibility controls (seed setting), device detection,
and distributed training setup (DDP initialization with automatic
backend selection).
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all relevant RNGs.

    Configures Python's ``random``, NumPy, and PyTorch (CPU + CUDA) random
    number generators. Also sets CuDNN to deterministic mode, which may
    reduce performance but ensures reproducible results.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic CuDNN for reproducibility (at some speed cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("Random seed set to %d.", seed)


def get_device() -> torch.device:
    """
    Detect and return the best available device.

    Returns ``cuda:<local_rank>`` if CUDA is available (respecting the
    ``LOCAL_RANK`` environment variable for DDP), otherwise ``cpu``.

    Returns
    -------
    torch.device
        The selected compute device.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        logger.info("Using CUDA device: %s", device)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    return device


def setup_distributed() -> tuple[int, int, bool]:
    """
    Initialize PyTorch distributed training if a distributed environment is detected.

    Checks for the ``RANK`` and ``WORLD_SIZE`` environment variables (set by
    ``torchrun`` or SLURM launchers). If present, initializes the process
    group with automatic backend selection:

    - **NCCL** when CUDA is available (optimal for GPU-to-GPU communication).
    - **Gloo** otherwise (CPU-only fallback, e.g. for local testing).

    Returns
    -------
    rank : int
        Global rank of this process (0 if not distributed).
    world_size : int
        Total number of processes (1 if not distributed).
    is_distributed : bool
        Whether distributed training is active.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.info("No distributed environment detected. Running in single-process mode.")
        return 0, 1, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ENG-DECISION: Explicitly select backend based on hardware availability.
    # NCCL is required for efficient multi-GPU training; Gloo is the CPU fallback.
    # Without this check, DDP crashes on CPU-only machines if NCCL is selected.
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info(
        "Distributed training initialized: rank=%d/%d, backend=%s.",
        rank,
        world_size,
        backend,
    )
    return rank, world_size, True


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def is_main_process(rank: int) -> bool:
    """Return True if this is the main (rank 0) process."""
    return rank == 0
