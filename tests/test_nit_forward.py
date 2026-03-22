"""
Tests for the MINiT deep learning architecture.

Covers:
- Forward pass output shape verification across configurations
- Gradient flow through all trainable parameters
- Model instantiation with varying hyperparameters
- Reproducibility (deterministic output given same seed and input)
- Freezing strategy correctness
- MixUp augmentation mechanics
"""

# ruff: noqa: E402

from __future__ import annotations

from typing import Any

import pytest

# Skip the entire module if PyTorch is not installed (it's an optional dep).
torch = pytest.importorskip("torch", reason="PyTorch required for DL tests")

import numpy as np

from brainrisk.deeplearning.augmentation import mixup_criterion, mixup_data
from brainrisk.deeplearning.freeze import apply_freeze_strategy
from brainrisk.deeplearning.model import MINiT

# ---------
# Helpers
# ---------

# Use tiny configurations for fast tests (no GPU needed).
_TINY_CFG: dict[str, Any] = dict(
    volume_size=16,
    block_size=8,
    patch_size=4,
    in_channels=1,
    num_classes=2,
    embed_dim=32,
    num_layers=1,
    num_heads=4,
    mlp_dim=64,
    dropout=0.0,
)


def _make_input(batch_size: int = 2, volume_size: int = 16) -> Any:
    """Create a random input tensor."""
    return torch.randn(batch_size, 1, volume_size, volume_size, volume_size)


# --------------------
# Forward pass tests
# --------------------


def test_forward_pass_output_shape() -> None:
    """The model should produce (N, C) logits for a batch of volumes."""
    model = MINiT(**_TINY_CFG)
    model.eval()
    x = _make_input(batch_size=2, volume_size=16)
    logits = model(x)
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"


def test_forward_pass_single_sample() -> None:
    """Single-sample batch should work without errors."""
    model = MINiT(**_TINY_CFG)
    model.eval()
    x = _make_input(batch_size=1, volume_size=16)
    logits = model(x)
    assert logits.shape == (1, 2)


def test_forward_pass_three_class() -> None:
    """Three-class output for HYDRA subtype classification."""
    cfg = {**_TINY_CFG, "num_classes": 3}
    model = MINiT(**cfg)
    model.eval()
    x = _make_input(batch_size=3, volume_size=16)
    logits = model(x)
    assert logits.shape == (3, 3)


# ---------------------
# Gradient flow tests
# ---------------------


def test_gradient_flow_all_parameters() -> None:
    """All trainable parameters should receive gradients after backward pass."""
    model = MINiT(**_TINY_CFG)
    model.train()
    x = _make_input(batch_size=2, volume_size=16)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.all(param.grad == 0), f"All-zero gradient for: {name}"


def test_gradient_with_cross_entropy_loss() -> None:
    """Verify gradients flow through a realistic cross-entropy loss."""
    model = MINiT(**_TINY_CFG)
    model.train()
    x = _make_input(batch_size=4, volume_size=16)
    labels = torch.tensor([0, 1, 0, 1])

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    # At least the aggregation layer should have gradients
    assert model.aggregation.weight.grad is not None
    assert model.block_head.weight.grad is not None


# -------------------------------
# Configuration variation tests
# -------------------------------


def test_different_configurations() -> None:
    """The model should instantiate and run with various valid configurations."""
    configs: list[dict[str, Any]] = [
        dict(
            volume_size=16,
            block_size=8,
            patch_size=4,
            embed_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_dim=64,
            num_classes=2,
        ),
        dict(
            volume_size=16,
            block_size=8,
            patch_size=2,
            embed_dim=64,
            num_layers=2,
            num_heads=8,
            mlp_dim=128,
            num_classes=5,
        ),
        dict(
            volume_size=32,
            block_size=16,
            patch_size=4,
            embed_dim=48,
            num_layers=3,
            num_heads=6,
            mlp_dim=96,
            num_classes=3,
        ),
        dict(
            volume_size=16,
            block_size=4,
            patch_size=2,
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            mlp_dim=32,
            num_classes=2,
        ),
    ]

    for cfg in configs:
        model = MINiT(**cfg)
        model.eval()
        x = _make_input(batch_size=1, volume_size=cfg["volume_size"])
        logits = model(x)
        assert logits.shape == (
            1,
            cfg["num_classes"],
        ), f"Config {cfg}: expected (1, {cfg['num_classes']}), got {logits.shape}"


def test_from_config_classmethod() -> None:
    """MINiT.from_config should produce a working model from a config dict."""
    config = {
        "model": {
            "volume_size": 16,
            "block_size": 8,
            "patch_size": 4,
            "num_classes": 3,
            "embed_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_dim": 64,
        }
    }
    model = MINiT.from_config(config)
    model.eval()
    x = _make_input(batch_size=1, volume_size=16)
    logits = model(x)
    assert logits.shape == (1, 3)


def test_invalid_divisibility_raises() -> None:
    """Non-divisible volume/block/patch sizes should raise ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        MINiT(volume_size=64, block_size=15)  # 64 % 15 != 0

    with pytest.raises(ValueError, match="divisible"):
        MINiT(volume_size=64, block_size=16, patch_size=5)  # 16 % 5 != 0


# -----------------------
# Reproducibility tests
# -----------------------


def test_reproducibility() -> None:
    """Two forward passes with the same input should produce identical outputs."""
    model = MINiT(**_TINY_CFG)
    model.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16, 16)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_deterministic_with_seed() -> None:
    """Two independently constructed models with the same seed and input match."""
    torch.manual_seed(42)
    m1 = MINiT(**_TINY_CFG)
    m1.eval()

    torch.manual_seed(42)
    m2 = MINiT(**_TINY_CFG)
    m2.eval()

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(m1(x), m2(x))


# ----------------------
# Parameter count test
# ----------------------


def test_count_parameters() -> None:
    """count_parameters should return a positive integer."""
    model = MINiT(**_TINY_CFG)
    n_params = model.count_parameters()
    assert isinstance(n_params, int)
    assert n_params > 0


# -------------------------
# Freezing strategy tests
# -------------------------


def test_freeze_head_only() -> None:
    """head_only: only head and aggregation parameters should be trainable."""
    model = MINiT(**_TINY_CFG)
    apply_freeze_strategy(model, "head_only")

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}

    # Head and aggregation should be trainable
    assert any("block_head" in n for n in trainable_names)
    assert any("aggregation" in n for n in trainable_names)

    # Encoder and embeddings should be frozen
    assert any("encoder" in n for n in frozen_names)
    assert any("pos_embedding" in n or "block_embedding" in n for n in frozen_names)


def test_freeze_partial() -> None:
    """partial: later encoder layers and heads should be trainable."""
    cfg = {**_TINY_CFG, "num_layers": 4}
    model = MINiT(**cfg)
    apply_freeze_strategy(model, "partial")

    # First 2 layers should be frozen, last 2 trainable
    for i, layer in enumerate(model.encoder.layers):
        for param in layer.parameters():
            if i < 2:
                assert not param.requires_grad, f"Layer {i} should be frozen"
            else:
                assert param.requires_grad, f"Layer {i} should be trainable"

    # Head should be trainable
    assert model.block_head.weight.requires_grad
    assert model.aggregation.weight.requires_grad


def test_freeze_full() -> None:
    """full: all parameters should be trainable."""
    model = MINiT(**_TINY_CFG)
    apply_freeze_strategy(model, "full")

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should be trainable"


def test_freeze_invalid_strategy_raises() -> None:
    """Invalid strategy name should raise ValueError."""
    model = MINiT(**_TINY_CFG)
    with pytest.raises(ValueError, match="Unknown freeze strategy"):
        apply_freeze_strategy(model, "calgary-cold_freeze")


# --------------------------
# MixUp augmentation tests
# --------------------------


def test_mixup_data_shapes() -> None:
    """MixUp should preserve tensor shapes."""
    volumes = torch.randn(4, 1, 16, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])

    mixed, labels_a, labels_b, lam = mixup_data(volumes, labels, alpha=1.0)
    assert mixed.shape == volumes.shape
    assert labels_a.shape == labels.shape
    assert labels_b.shape == labels.shape
    assert 0.0 <= lam <= 1.0


def test_mixup_alpha_zero_is_identity() -> None:
    """With alpha=0, MixUp should return the original data unchanged."""
    volumes = torch.randn(4, 1, 16, 16, 16)
    labels = torch.tensor([0, 1, 0, 1])

    mixed, labels_a, labels_b, lam = mixup_data(volumes, labels, alpha=0.0)
    assert lam == 1.0
    torch.testing.assert_close(mixed, volumes)


def test_mixup_criterion_computes() -> None:
    """MixUp criterion should produce a scalar loss."""
    criterion = torch.nn.CrossEntropyLoss()
    logits = torch.randn(4, 3)
    labels_a = torch.tensor([0, 1, 2, 0])
    labels_b = torch.tensor([1, 2, 0, 1])

    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam=0.7)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


# ---------------
# Dataset test
# ---------------


def test_volume_dataset(tmp_path) -> None:
    """VolumeClassificationDataset should load .npy files and return correct shapes."""
    from brainrisk.deeplearning.dataset import VolumeClassificationDataset

    # Create test volumes
    paths = []
    for i in range(3):
        vol = np.random.rand(16, 16, 16).astype(np.float32)
        p = tmp_path / f"vol_{i}.npy"
        np.save(p, vol)
        paths.append(str(p))

    dataset = VolumeClassificationDataset(paths, [0, 1, 0])
    assert len(dataset) == 3

    vol, label = dataset[0]
    assert vol.shape == (1, 16, 16, 16)
    assert vol.dtype == torch.float32
    assert label.dtype == torch.long
