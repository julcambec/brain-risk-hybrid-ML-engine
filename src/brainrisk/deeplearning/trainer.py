"""
Training loop for MINiT volumetric classification.

Provides the complete training pipeline including:

- Single-GPU and multi-GPU (DDP) training with automatic backend selection.
- MixUp augmentation with configurable intensity.
- Cosine annealing learning rate schedule with optional linear warmup.
- Checkpoint save/load/resume with best-model tracking.
- Structured JSON-lines logging (one entry per epoch) for experiment tracking.
- Optional Weights & Biases integration (disabled by default).
- Early stopping based on validation accuracy.

The primary entry point is :func:`run_training`, which accepts a configuration
dictionary and orchestrates the full training run.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from brainrisk.deeplearning.augmentation import mixup_criterion, mixup_data
from brainrisk.deeplearning.dataset import build_datasets_from_config
from brainrisk.deeplearning.freeze import apply_freeze_strategy
from brainrisk.deeplearning.model import MINiT
from brainrisk.deeplearning.utils import (
    cleanup_distributed,
    get_device,
    is_main_process,
    set_seed,
    setup_distributed,
)

logger = logging.getLogger(__name__)


# ----------------------
# Unwrapping helper
# ----------------------


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model if wrapped in DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


# ----------------------
# Optimizer & Scheduler
# ----------------------


def _build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer from config.

    Only parameters with ``requires_grad=True`` are included, so frozen
    parameters are automatically excluded.
    """
    training_cfg = config.get("training", {})
    lr = training_cfg.get("learning_rate", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 0.01)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a cosine annealing scheduler with optional linear warmup.

    ENG-DECISION: The paper uses cosine decay with gradual warmup (Goyal
    et al., 2017). I implement this as a LambdaLR that linearly ramps
    during warmup epochs and then follows a cosine decay schedule.
    """
    training_cfg = config.get("training", {})
    total_epochs = training_cfg.get("epochs", 200)
    warmup_epochs = training_cfg.get("warmup_epochs", 10)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs and warmup_epochs > 0:
            # Linear warmup from 0 to 1
            return (epoch + 1) / warmup_epochs
        # Cosine decay from 1 to ~0
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------
# Checkpointing
# --------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_metric: float,
    config: dict[str, Any],
    filepath: str | Path,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        The model (handles DDP-wrapped models by accessing ``.module``).
    optimizer : torch.optim.Optimizer
        Optimizer state.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Scheduler state.
    epoch : int
        Current epoch number (0-indexed).
    best_val_metric : float
        Best validation accuracy seen so far.
    config : dict
        Full training configuration (saved for reproducibility).
    filepath : str | Path
        Output path for the checkpoint file.
    """
    # Handle DDP-wrapped models
    model_state = _unwrap_model(model).state_dict()

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "config": config,
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.info("Checkpoint saved → %s (epoch %d)", filepath, epoch)


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """
    Load a training checkpoint and restore state.

    Parameters
    ----------
    filepath : str | Path
        Path to the checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer | None
        Optimizer to restore (if provided).
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        Scheduler to restore (if provided).
    device : torch.device | None
        Device to map tensors to.

    Returns
    -------
    epoch : int
        The epoch to resume from (checkpoint epoch + 1).
    best_val_metric : float
        Best validation metric from the checkpoint.
    """
    map_location = device or torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    # Handle DDP-wrapped models
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
    best_val_metric = checkpoint.get("best_val_metric", 0.0)

    logger.info("Checkpoint loaded ← %s (resuming from epoch %d)", filepath, epoch)
    return epoch, best_val_metric


# ----------------------
# Training & Validation
# ----------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_alpha: float = 0.0,
) -> dict[str, float]:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    dataloader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : nn.Module
        Loss function (e.g. ``nn.CrossEntropyLoss``).
    device : torch.device
        Compute device.
    mixup_alpha : float
        MixUp alpha parameter. Set to 0 to disable.

    Returns
    -------
    dict[str, float]
        ``train_loss`` and ``train_acc`` for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for volumes, labels in dataloader:
        volumes = volumes.to(device, non_blocking=True)  # (N, 1, L, W, H)
        labels = labels.to(device, non_blocking=True)  # (N,)

        if mixup_alpha > 0.0:
            volumes, labels_a, labels_b, lam = mixup_data(
                volumes, labels, alpha=mixup_alpha, device=device
            )
            logits = model(volumes)  # (N, C)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

            # Accuracy: use the dominant label for tracking purposes
            preds = logits.argmax(dim=1)
            correct += int(
                (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float())
                .sum()
                .item()
            )
        else:
            logits = model(volumes)  # (N, C)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())

        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"train_loss": avg_loss, "train_acc": accuracy}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on the validation set.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate (set to eval mode internally).
    dataloader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.

    Returns
    -------
    dict[str, float]
        ``val_loss`` and ``val_acc`` for the validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for volumes, labels in dataloader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(volumes)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"val_loss": avg_loss, "val_acc": accuracy}


# ---------
# Logging
# ---------


def _log_epoch(
    log_path: str | Path,
    entry: dict[str, Any],
) -> None:
    """Append a JSON-lines entry to the training log file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _init_wandb(config: dict[str, Any]) -> Any:
    """
    Initialize Weights & Biases if enabled and available.

    Returns the ``wandb`` module if active, or ``None`` if disabled or
    unavailable (degrades gracefully).
    """
    log_cfg = config.get("logging", {})
    if not log_cfg.get("wandb_enabled", False):
        return None

    try:
        import wandb

        wandb.init(
            project=log_cfg.get("wandb_project", "brainrisk-minit"),
            config=config,
        )
        logger.info("Weights & Biases initialized.")
        return wandb
    except ImportError:
        logger.warning("wandb_enabled=True but wandb is not installed. Skipping.")
        return None


# ==========================
# Main training entry point
# ==========================


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a complete MINiT training run.

    This is the primary entry point for training. It handles:

    1. Seed setting and device detection
    2. Distributed training setup (if applicable)
    3. Dataset construction and data loading
    4. Model instantiation and optional layer freezing
    5. Optimizer and scheduler construction
    6. Optional checkpoint resume
    7. Training loop with validation, checkpointing, and logging
    8. Cleanup

    Parameters
    ----------
    config : dict
        Full training configuration. See ``configs/demo_dl.yaml`` for
        the expected structure and available keys.

    Returns
    -------
    dict[str, Any]
        Summary of the training run, including final and best metrics.
    """
    # --- Setup ---
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device()
    rank, world_size, is_distributed = setup_distributed()

    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})
    aug_cfg = config.get("augmentation", {})

    epochs = training_cfg.get("epochs", 200)
    batch_size = training_cfg.get("batch_size", 4)
    num_workers = training_cfg.get("num_workers", 0)
    mixup_alpha = aug_cfg.get("mixup_alpha", 0.0)
    patience = training_cfg.get("early_stopping_patience", 0)  # 0 = disabled

    log_path = log_cfg.get("log_path", "artifacts/dl/training_log.jsonl")
    save_dir = Path(checkpoint_cfg.get("save_dir", "artifacts/dl/checkpoints"))
    save_every = checkpoint_cfg.get("save_every", 10)

    # --- Data ---
    if is_main_process(rank):
        logger.info("Building datasets from config ...")
    train_dataset, val_dataset = build_datasets_from_config(config)

    # Distributed sampler for DDP; standard sequential access otherwise
    train_sampler: DistributedSampler[Any] | None = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if is_main_process(rank):
        logger.info(
            "Data loaded: %d train, %d val samples.",
            len(train_dataset),
            len(val_dataset),
        )

    # --- Model ---
    model: nn.Module = MINiT.from_config(config)
    freeze_strategy = config.get("freeze_strategy", "full")
    apply_freeze_strategy(model, freeze_strategy)
    model = model.to(device)

    if is_main_process(rank):
        logger.info(
            "MINiT instantiated: %d total params, %d trainable.",
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    # DDP wrapping
    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[device] if device.type == "cuda" else None
        )

    # --- Optimizer & Scheduler ---
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss()

    # --- Resume ---
    start_epoch = 0
    best_val_acc = 0.0
    resume_path = config.get("resume_path")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_acc = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )

    # --- W&B ---
    wandb_module = None
    if is_main_process(rank):
        wandb_module = _init_wandb(config)

    # --- Training loop ---
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mixup_alpha
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        current_lr = float(scheduler.get_last_lr()[0])
        epoch_time = time.time() - epoch_start

        # -- Logging (rank 0 only) --
        if is_main_process(rank):
            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_metrics["train_loss"], 5),
                "train_acc": round(train_metrics["train_acc"], 4),
                "val_loss": round(val_metrics["val_loss"], 5),
                "val_acc": round(val_metrics["val_acc"], 4),
                "lr": round(current_lr, 8),
                "time_seconds": round(epoch_time, 2),
            }
            _log_epoch(log_path, log_entry)

            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f train_acc=%.3f | "
                "val_loss=%.4f val_acc=%.3f | lr=%.2e | %.1fs",
                epoch,
                epochs,
                train_metrics["train_loss"],
                train_metrics["train_acc"],
                val_metrics["val_loss"],
                val_metrics["val_acc"],
                current_lr,
                epoch_time,
            )

            # W&B logging
            if wandb_module is not None:
                wandb_module.log(log_entry)

            # -- Checkpointing --
            is_best = val_metrics["val_acc"] > best_val_acc
            if is_best:
                best_val_acc = val_metrics["val_acc"]
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_acc,
                    config,
                    save_dir / "checkpoint_best.pt",
                )
            else:
                epochs_without_improvement += 1

            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_acc,
                    config,
                    save_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                )

        # -- Early stopping --
        if patience > 0 and epochs_without_improvement >= patience:
            if is_main_process(rank):
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
            break

    # --- Cleanup ---
    if is_distributed:
        cleanup_distributed()

    if wandb_module is not None:
        wandb_module.finish()

    summary = {
        "best_val_acc": best_val_acc,
        "final_epoch": epoch,
        "log_path": str(log_path),
        "checkpoint_dir": str(save_dir),
    }

    if is_main_process(rank):
        logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)

    return summary
