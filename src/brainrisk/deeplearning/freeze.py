"""
Layer freezing strategies for MINiT fine-tuning.

Provides a simple mechanism to freeze subsets of model parameters before
training begins. Three granularities are supported:

- **head_only**: Freeze everything except the per-block classification head
  and the aggregation layer. Use when fine-tuning a pretrained encoder on a
  new task with minimal data.
- **partial**: Freeze early transformer layers and all embedding parameters;
  train the later encoder layers, the classification head, and the
  aggregation layer. Offers a middle ground between full fine-tuning and
  head-only training.
- **full**: Train all layers (no freezing). Default for training from scratch.

The boundary for "partial" freezing is the midpoint of the encoder layers
(i.e. the first half is frozen, the second half is trainable). This is a
reasonable heuristic: early layers learn generic features while later layers
specialize for the task.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


_VALID_STRATEGIES = {"head_only", "partial", "full"}


def apply_freeze_strategy(model: nn.Module, strategy: str = "full") -> None:
    """
    Apply a freezing strategy to a MINiT model.

    Parameters
    ----------
    model : nn.Module
        The MINiT model instance. Expected to have attributes:
        ``encoder``, ``block_head``, ``block_head_norm``, ``aggregation``,
        ``patch_projection``, ``cls_token``, ``pos_embedding``,
        ``block_embedding``, ``embed_dropout``.
    strategy : str
        One of ``"head_only"``, ``"partial"``, or ``"full"``.

    Raises
    ------
    ValueError
        If the strategy name is not recognized.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown freeze strategy '{strategy}'. Choose from {_VALID_STRATEGIES}.")

    if strategy == "full":
        # Train everything — ensure all params are unfrozen
        for param in model.parameters():
            param.requires_grad_(True)
        logger.info("Freeze strategy 'full': all %d parameters trainable.", _count(model, True))
        return

    if strategy == "head_only":
        _freeze_head_only(model)
        return

    if strategy == "partial":
        _freeze_partial(model)
        return


def _freeze_head_only(model: nn.Module) -> None:
    """
    Freeze all parameters except the classification head and aggregation.

    Trainable:
        - block_head_norm, block_head (per-block classification head)
        - aggregation (final prediction layer)

    Frozen:
        - patch_projection, cls_token, pos_embedding, block_embedding
        - All encoder layers
        - embed_dropout (no learnable params, but listed for clarity)
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Then unfreeze the heads
    for name in ("block_head_norm", "block_head", "aggregation"):
        submodule = getattr(model, name, None)
        if submodule is not None:
            for param in submodule.parameters():
                param.requires_grad_(True)

    trainable = _count(model, True)
    frozen = _count(model, False)
    logger.info(
        "Freeze strategy 'head_only': %d trainable, %d frozen parameters.",
        trainable,
        frozen,
    )


def _freeze_partial(model: nn.Module) -> None:
    """
    Freeze early layers; train later encoder layers and heads.

    Trainable:
        - Second half of encoder layers (layers[midpoint:])
        - block_head_norm, block_head, aggregation

    Frozen:
        - patch_projection, cls_token, pos_embedding, block_embedding
        - First half of encoder layers (layers[:midpoint])
        - embed_dropout

    The midpoint is ``num_layers // 2``. For odd layer counts, the
    extra layer goes to the trainable (unfrozen) group.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze later encoder layers
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers"):
        num_layers = len(encoder.layers)
        midpoint = num_layers // 2
        for layer in encoder.layers[midpoint:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        logger.debug(
            "Partial freeze: encoder layers [%d:%d] trainable (of %d total).",
            midpoint,
            num_layers,
            num_layers,
        )

    # Unfreeze heads
    for name in ("block_head_norm", "block_head", "aggregation"):
        submodule = getattr(model, name, None)
        if submodule is not None:
            for param in submodule.parameters():
                param.requires_grad_(True)

    trainable = _count(model, True)
    frozen = _count(model, False)
    logger.info(
        "Freeze strategy 'partial': %d trainable, %d frozen parameters.",
        trainable,
        frozen,
    )


def _count(model: nn.Module, requires_grad: bool) -> int:
    """Count parameters by their requires_grad status."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)
