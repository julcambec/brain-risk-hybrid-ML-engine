"""
Multiple Instance Neuroimage Transformer (MINiT) architecture.

Implements a hierarchical, convolution-free transformer for classifying 3D
volumetric neuroimages. The architecture processes inputs in two levels
inspired by multiple instance learning (MIL):

1. **Block decomposition**: the input volume is partitioned into a regular
   grid of non-overlapping cubic blocks (the "bags" in MIL terminology).
2. **Patch-level processing**: each block is independently processed by a
   shared NiT encoder that decomposes it into non-overlapping 3D patches
   (the "instances"), embeds them, and applies multi-headed self-attention.
3. **Prediction aggregation**: per-block class predictions from the NiT
   encoder are concatenated and linearly projected to yield the final output.

Key design properties:

- **Convolution-free**: All spatial feature extraction is done via linear
  patch projection and self-attention, making attention kernels dynamically
  computed rather than fixed after training.
- **Hierarchical attention**: Block embeddings inject positional information
  about each block's location within the full volume, enabling the model to
  learn both local (within-block) and global (across-block) spatial patterns.
- **Shared encoder**: A single NiT encoder processes all blocks with shared
  weights, keeping the parameter count manageable.

Reference
---------
Singla, A. et al. "Multiple Instance Neuroimage Transformer" (2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brainrisk.deeplearning.layers import TransformerEncoder


class MINiT(nn.Module):
    """
    Multiple Instance Neuroimage Transformer.

    Processes a batch of 3D volumetric neuroimages through a hierarchical
    block → patch → transformer pipeline and produces class logits.

    Input contract
    --------------
    Tensor of shape ``(N, I, L, W, H)`` where:

    - *N* is the batch size
    - *I* is the number of input channels (typically 1 for T1w MRI)
    - *L*, *W*, *H* are spatial dimensions (must all equal ``volume_size``)

    Output contract
    ---------------
    Tensor of shape ``(N, C)``: raw (unnormalized) class logits.

    Tensor shape trace (reference configuration from the paper)
    -----------------------------------------------------------
    With ``volume_size=64, block_size=16, patch_size=4, embed_dim=256,
    num_layers=6, num_heads=8, mlp_dim=309, num_classes=2``::

        Input                        (N, 1, 64, 64, 64)
        Block decomposition          (N*64, 1, 16, 16, 16)
        Patch flatten                (N*64, 64, 64)
        Linear projection            (N*64, 64, 256)
        + CLS token                  (N*64, 65, 256)
        + positional embedding       (N*64, 65, 256)
        + block embedding            (N*64, 65, 256)
        Transformer output           (N*64, 65, 256)
        CLS extraction               (N*64, 256)
        Per-block head               (N*64, 2)
        Reshape + concatenate        (N, 128)
        Aggregation                  (N, 2)

    Parameters
    ----------
    volume_size : int
        Spatial dimension of the cubic input volume (L = W = H).
    block_size : int
        Side length *B* of each cubic block. Must divide ``volume_size``.
    patch_size : int
        Side length *Q* of each cubic patch within a block. Must divide
        ``block_size``.
    in_channels : int
        Number of input channels *I* (default 1 for single-channel T1w MRI).
    num_classes : int
        Number of output classes *C*.
    embed_dim : int
        Transformer model dimension *D*.
    num_layers : int
        Number of transformer encoder layers *L_enc*.
    num_heads : int
        Number of attention heads *N_H*.
    mlp_dim : int
        Hidden dimension *D_MLP* of the feed-forward network (post-GEGLU).
    dropout : float
        Dropout probability applied throughout the model.
    """

    def __init__(
        self,
        volume_size: int = 64,
        block_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 309,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # --- Validate divisibility constraints ---
        if volume_size % block_size != 0:
            raise ValueError(
                f"volume_size ({volume_size}) must be divisible by block_size ({block_size})."
            )
        if block_size % patch_size != 0:
            raise ValueError(
                f"block_size ({block_size}) must be divisible by patch_size ({patch_size})."
            )

        self.volume_size = volume_size
        self.block_size = block_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # --- Derived quantities ---
        self.blocks_per_dim: int = volume_size // block_size  # n_B
        self.num_blocks: int = self.blocks_per_dim**3  # nu_B = n_B^3
        self.patches_per_dim: int = block_size // patch_size  # n_P
        self.num_patches: int = self.patches_per_dim**3  # nu_P = n_P^3
        self.patch_volume: int = in_channels * patch_size**3  # I * Q^3
        self.seq_len: int = self.num_patches + 1  # nu_P + 1 (incl. CLS)

        # --- Patch embedding ---
        # Linear projection E: R^(I*Q^3) → R^D
        self.patch_projection = nn.Linear(self.patch_volume, embed_dim)

        # --- Classification token ---
        # Learned token t_cls ∈ R^D, prepended to each patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional embedding ---
        # Phi ∈ R^(nu_P+1, D), shared across all blocks, encodes
        # patch arrangement within a block
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        # --- Block embedding ---
        # Omega ∈ R^(nu_B, nu_P+1, D), encodes each block's position
        # within the full input volume. This is prob a key MINiT innovation:
        # injects global positional information into the per-block
        # processing, loosely emulating hierarchical attention.
        self.block_embedding = nn.Parameter(torch.zeros(self.num_blocks, self.seq_len, embed_dim))

        # --- Transformer encoder ---
        # Shared across all blocks (each block is an "instance" in MIL)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # --- Per-block classification head ---
        # LayerNorm → Linear(D, C) applied to the CLS token output
        self.block_head_norm = nn.LayerNorm(embed_dim)
        self.block_head = nn.Linear(embed_dim, num_classes)

        # --- Prediction aggregation ---
        # Linear(nu_B * C, C): combines all per-block predictions
        self.aggregation = nn.Linear(self.num_blocks * num_classes, num_classes)

        # ENG-DECISION: Applying dropout to patch embeddings before the
        # transformer, consistent with standard ViT practice. The paper
        # mentions dropout as a regularization technique but does not
        # specify exact placement within the pipeline.
        self.embed_dropout = nn.Dropout(dropout)

        # Initialize all parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        Initialize learned parameters.

        ENG-DECISION: The paper does not specify an initialization scheme.
        I use truncated normal (std=0.02) for embedding parameters and
        Xavier uniform for linear layers, following standard ViT practice
        (Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020).
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.block_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _decompose_into_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partition a batch of volumes into non-overlapping cubic blocks.

        Reshapes and permutes the spatial dimensions to extract a regular
        grid of cubic blocks along all three axes.

        Parameters
        ----------
        x : torch.Tensor
            Input volumes of shape ``(N, I, L, W, H)``.

        Returns
        -------
        torch.Tensor
            Blocks of shape ``(N * nu_B, I, B, B, B)``.
        """
        N = x.shape[0]
        I = self.in_channels  # noqa: E741
        nB = self.blocks_per_dim
        B = self.block_size

        # Reshape spatial dims into (n_B, B) pairs along each axis
        x = x.reshape(N, I, nB, B, nB, B, nB, B)  # (N, I, nB, B, nB, B, nB, B)
        # Group block indices together, channel and local coords together
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (N, nB, nB, nB, I, B, B, B)
        # Merge batch and block dimensions
        x = x.reshape(N * self.num_blocks, I, B, B, B)  # (N*nu_B, I, B, B, B)
        return x

    def _extract_patches(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Extract and flatten non-overlapping cubic patches from blocks.

        Parameters
        ----------
        blocks : torch.Tensor
            Block volumes of shape ``(N * nu_B, I, B, B, B)``.

        Returns
        -------
        torch.Tensor
            Flattened patches of shape ``(N * nu_B, nu_P, I * Q^3)``.
        """
        NB = blocks.shape[0]
        I = self.in_channels  # noqa: E741
        nP = self.patches_per_dim
        Q = self.patch_size

        # Reshape spatial dims into (n_P, Q) pairs along each axis
        x = blocks.reshape(NB, I, nP, Q, nP, Q, nP, Q)  # (NB, I, nP, Q, nP, Q, nP, Q)
        # Group patch indices together, flatten channel and local coords
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # (NB, nP, nP, nP, I, Q, Q, Q)
        # Flatten into patch vectors
        x = x.reshape(NB, self.num_patches, self.patch_volume)  # (NB, nu_P, I*Q^3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full MINiT architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input volumes of shape ``(N, I, L, W, H)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(N, C)``.
        """
        N = x.shape[0]

        # Step 1: Block decomposition
        blocks = self._decompose_into_blocks(x)  # (N*nu_B, I, B, B, B)

        # Step 2: Patch extraction and flattening
        patches = self._extract_patches(blocks)  # (N*nu_B, nu_P, I*Q^3)

        # Step 3: Linear projection to model dimension
        tokens = self.patch_projection(patches)  # (N*nu_B, nu_P, D)

        # Step 4: Prepend classification token
        cls_expanded = self.cls_token.expand(N * self.num_blocks, -1, -1)  # (N*nu_B, 1, D)
        tokens = torch.cat([cls_expanded, tokens], dim=1)  # (N*nu_B, nu_P+1, D)

        # Step 5: Add positional embedding
        tokens = tokens + self.pos_embedding  # (N*nu_B, nu_P+1, D)

        # Step 6: Add block embedding
        # Omega has shape (nu_B, nu_P+1, D). I need to broadcast it
        # across the batch dimension: repeat the block embedding pattern
        # N times so that sample i's j-th block gets Omega[j].
        block_emb = self.block_embedding.unsqueeze(0)  # (1, nu_B, nu_P+1, D)
        block_emb = block_emb.expand(N, -1, -1, -1)  # (N, nu_B, nu_P+1, D)
        block_emb = block_emb.reshape(
            N * self.num_blocks, self.seq_len, self.embed_dim
        )  # (N*nu_B, nu_P+1, D)
        tokens = tokens + block_emb  # (N*nu_B, nu_P+1, D)

        # ENG-DECISION: Dropout on the combined embeddings before the
        # transformer, following standard ViT practice.
        tokens = self.embed_dropout(tokens)

        # Step 7: Transformer encoder
        encoded = self.encoder(tokens)  # (N*nu_B, nu_P+1, D)

        # Step 8: Extract CLS token representation
        cls_output = encoded[:, 0]  # (N*nu_B, D)

        # Step 9: Per-block classification head
        cls_output = self.block_head_norm(cls_output)  # (N*nu_B, D)
        block_logits = self.block_head(cls_output)  # (N*nu_B, C)

        # Step 10: Reshape and concatenate
        block_logits = block_logits.reshape(N, self.num_blocks * self.num_classes)  # (N, nu_B * C)

        # Step 11: Prediction aggregation
        logits = self.aggregation(block_logits)  # (N, C)

        return logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> MINiT:
        """
        Construct a MINiT instance from a configuration dictionary.

        The dictionary should contain a ``model`` key with the constructor
        keyword arguments. Missing keys fall back to the constructor defaults.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Expected structure::

                {"model": {"volume_size": 64, "block_size": 16, ...}}

        Returns
        -------
        MINiT
            A new MINiT instance.
        """
        model_cfg = config.get("model", {})
        return cls(
            volume_size=model_cfg.get("volume_size", 64),
            block_size=model_cfg.get("block_size", 16),
            patch_size=model_cfg.get("patch_size", 4),
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 2),
            embed_dim=model_cfg.get("embed_dim", 256),
            num_layers=model_cfg.get("num_layers", 6),
            num_heads=model_cfg.get("num_heads", 8),
            mlp_dim=model_cfg.get("mlp_dim", 309),
            dropout=model_cfg.get("dropout", 0.0),
        )
