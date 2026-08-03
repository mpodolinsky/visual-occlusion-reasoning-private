"""Trainable perception-uncertainty probe on top of frozen pi0.5 prefix features.

Pure PyTorch, no dependency on openpi/JAX -- operates on the raw per-token
prefix representations (`base_image`, `wrist_image`, `language`) that
`serve_pi05_with_features.py` extracts from the frozen backbone and sends
over the websocket. See scripts/perception_probe/README.md for the full
architecture this implements.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_DIM = 2048


class AttentionPool(nn.Module):
    """Learned single-query attention pooling over a variable-length token set.

    Collapses (batch, num_tokens, dim) -> (batch, dim) with a softmax-weighted
    sum instead of a flat average, so the probe can learn to weight the tokens
    that actually matter (e.g. the occluded region of an image) rather than
    diluting them across every patch.
    """

    def __init__(self, dim: int = FEATURE_DIM, key_dim: int = 128):
        super().__init__()
        self.key_proj = nn.Linear(dim, key_dim)
        self.query = nn.Parameter(torch.randn(key_dim) * key_dim**-0.5) # torch.Size([key_dim])

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (batch, num_tokens, dim). mask: (batch, num_tokens) bool, True = valid."""
        keys = self.key_proj(tokens)  # (b, t, key_dim)
        scores = keys @ self.query  # (b, t)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (b, t) # We learn the weights that will serve for the weighted combination.
        return torch.einsum("bt,btd->bd", weights, tokens) 


class PerceptionSuccessProbe(nn.Module):
    """AttnPool(base image) + AttnPool(wrist image) + AttnPool(language) -> MLP -> p(success).

    Everything here is trainable; the pi0.5 backbone that produced `tokens` is
    frozen and not part of this module at all.
    """

    def __init__(self, dim: int = FEATURE_DIM, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pool_base = AttentionPool(dim)
        self.pool_wrist = AttentionPool(dim)
        self.pool_lang = AttentionPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        base_image: torch.Tensor,
        wrist_image: torch.Tensor,
        language: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw logits, shape (batch,). Apply sigmoid for p(success)."""
        pooled = torch.cat(
            [
                self.pool_base(base_image),
                self.pool_wrist(wrist_image),
                self.pool_lang(language, language_mask),
            ],
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)
