"""Trainable perception-uncertainty probe on top of frozen pi0.5 prefix features.

Pure PyTorch, no dependency on openpi/JAX -- operates on the raw per-token
prefix representations (`base_image`, `wrist_image`, `language`) that
`serve_pi05_with_features.py` extracts from the frozen backbone and sends
over the websocket. See scripts/perception_probe/README.md for the full
architecture this implements.

The default construction -- PerceptionSuccessProbe() with no arguments -- is
byte-for-byte the same architecture as before the sweep-support rewrite:
three independent single-query attention pools (key_dim=128) over base /
wrist / language tokens, concatenated, then LayerNorm -> Linear(6144, 256) ->
GELU -> Dropout(0.1) -> Linear(256, 1). All the extra knobs below default to
that; they exist for scripts/perception_probe/sweep.py.
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_DIM = 2048

CANONICAL_MODALITIES = ("base", "wrist", "lang")

# pool types that collapse (b, t, dim) -> (b, dim); every one outputs `dim`
# so the head input stays dim * n_modalities regardless of choice.
POOL_TYPES = ("attention", "gated", "topk", "mean", "max")


class TokenPool(nn.Module):
    """Collapses a variable-length token set (batch, num_tokens, dim) -> (batch, dim).

    pool_type:
      - "attention": learned query attention (the original AttentionPool). With
        n_queries > 1, several independent query vectors each produce a pooled
        vector and the results are averaged (keeps output dim == dim).
      - "gated": gated-attention MIL pooling (Ilse et al. 2018) -- attention
        weights a_t come from tanh(V h_t) * sigmoid(U h_t), which lets the pool
        both select AND veto tokens. n_queries adds parallel gated heads (mean-
        reduced like "attention").
      - "topk": same scoring as "attention" but softmax is taken over only the
        top-k highest-scoring valid tokens per query (the rest masked to -inf) --
        an explicit "only a few patches matter" pool.
      - "mean" / "max": parameter-free baselines (mask-aware).

    temperature divides the pre-softmax scores (attention/gated/topk only);
    < 1 sharpens the selection, > 1 softens it toward a mean.
    """

    def __init__(
        self,
        dim: int = FEATURE_DIM,
        key_dim: int = 128,
        pool_type: str = "attention",
        n_queries: int = 1,
        topk: int | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        if pool_type not in POOL_TYPES:
            raise ValueError(f"pool_type must be one of {POOL_TYPES}, got {pool_type!r}")
        self.pool_type = pool_type
        self.n_queries = n_queries
        self.topk = topk
        self.temperature = temperature

        if pool_type in ("attention", "topk"):
            self.key_proj = nn.Linear(dim, key_dim)
            # keep the 1-query case 1-D so pre-rewrite checkpoints still load
            if n_queries == 1:
                self.query = nn.Parameter(torch.randn(key_dim) * key_dim**-0.5)
            else:
                self.query = nn.Parameter(torch.randn(n_queries, key_dim) * key_dim**-0.5)
        elif pool_type == "gated":
            self.gate_v = nn.Linear(dim, key_dim)
            self.gate_u = nn.Linear(dim, key_dim)
            self.gate_w = nn.Linear(key_dim, n_queries)

    def _scores(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns pre-softmax scores, shape (b, n_queries, t)."""
        if self.pool_type == "gated":
            gated = torch.tanh(self.gate_v(tokens)) * torch.sigmoid(self.gate_u(tokens))  # (b, t, key_dim)
            return self.gate_w(gated).transpose(1, 2)  # (b, n_queries, t)
        keys = self.key_proj(tokens)  # (b, t, key_dim)
        query = self.query if self.query.ndim == 2 else self.query.unsqueeze(0)  # (n_queries, key_dim)
        return torch.einsum("qk,btk->bqt", query, keys)  # (b, n_queries, t)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (batch, num_tokens, dim). mask: (batch, num_tokens) bool, True = valid."""
        if self.pool_type == "mean":
            if mask is None:
                return tokens.mean(dim=1)
            m = mask.unsqueeze(-1).to(tokens.dtype)
            return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.pool_type == "max":
            if mask is not None:
                tokens = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return tokens.max(dim=1).values

        scores = self._scores(tokens) / self.temperature  # (b, q, t)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if self.pool_type == "topk" and self.topk is not None:
            k = min(self.topk, scores.shape[-1])
            kth = scores.topk(k, dim=-1).values[..., -1:].detach()  # (b, q, 1)
            scores = scores.masked_fill(scores < kth, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (b, q, t)
        pooled = torch.einsum("bqt,btd->bqd", weights, tokens)  # (b, q, dim)
        return pooled.mean(dim=1)  # (b, dim)


class AttentionPool(TokenPool):
    """Backwards-compatible alias: the original single-query attention pool."""

    def __init__(self, dim: int = FEATURE_DIM, key_dim: int = 128):
        super().__init__(dim=dim, key_dim=key_dim, pool_type="attention", n_queries=1)


class PerceptionSuccessProbe(nn.Module):
    """Pool(base) [+ Pool(wrist)] [+ Pool(language)] -> MLP -> failure/success logit.

    Everything here is trainable; the pi0.5 backbone that produced `tokens` is
    frozen and not part of this module at all.

    Sweep knobs (all default to the original architecture):
      - hidden_dim, dropout, n_hidden_layers: the MLP head.
      - key_dim, pool_type, n_queries, topk, pool_temperature: the token pools.
      - modalities: subset of ("base", "wrist", "lang") to actually use; the
        head input shrinks to dim * len(modalities).
      - share_image_pool: base and wrist share one pool module.
      - input_proj_dim: if set, a FIXED (non-trained, seeded) random projection
        dim -> input_proj_dim is applied to every token before pooling, which
        is the cheapest way to shrink the head's parameter count.
    """

    def __init__(
        self,
        dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        *,
        n_hidden_layers: int = 1,
        key_dim: int = 128,
        pool_type: str = "attention",
        n_queries: int = 1,
        topk: int | None = None,
        pool_temperature: float = 1.0,
        modalities: tuple[str, ...] = CANONICAL_MODALITIES,
        share_image_pool: bool = False,
        input_proj_dim: int | None = None,
        embed_dim: int | None = None,
        embed_hidden: int | None = None,
    ):
        super().__init__()
        bad = set(modalities) - set(CANONICAL_MODALITIES)
        if bad:
            raise ValueError(f"unknown modalities {sorted(bad)}; pick from {CANONICAL_MODALITIES}")
        if not modalities:
            raise ValueError("need at least one modality")
        # canonical order regardless of how the caller passed them
        self.modalities = tuple(m for m in CANONICAL_MODALITIES if m in modalities)

        pool_dim = dim if input_proj_dim is None else input_proj_dim
        if input_proj_dim is not None:
            gen = torch.Generator().manual_seed(0)
            proj = torch.randn(dim, input_proj_dim, generator=gen) * dim**-0.5
            self.register_buffer("input_proj", proj)
        else:
            self.input_proj = None

        def make_pool() -> TokenPool:
            return TokenPool(pool_dim, key_dim, pool_type, n_queries, topk, pool_temperature)

        self.pool_base = make_pool() if "base" in self.modalities else None
        if "wrist" in self.modalities:
            self.pool_wrist = (
                self.pool_base if share_image_pool and self.pool_base is not None else make_pool()
            )
        else:
            self.pool_wrist = None
        self.pool_lang = make_pool() if "lang" in self.modalities else None

        head_in = pool_dim * len(self.modalities)
        self.embed_dim = embed_dim
        if embed_dim is None:
            # unchanged: LayerNorm -> Linear(head_in, hidden) -> GELU -> Dropout [-> ...] -> Linear(hidden, 1)
            layers: list[nn.Module] = [
                nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            ]
            for _ in range(max(0, n_hidden_layers - 1)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden_dim, 1))
            self.head = nn.Sequential(*layers)
            self.embed = self.classifier = None
        else:
            # embed block ends in a bare Linear -> z (no activation on z, so it can
            # take any direction for cosine/InfoNCE alignment). embed_hidden adds a
            # compression bottleneck first: head_in -> embed_hidden -> embed_dim, which
            # caps capacity (a bare head_in -> 512 doubles params vs the default 256
            # head and memorises the ~210 train episodes -> unseen collapses).
            eb: list[nn.Module] = [nn.LayerNorm(head_in)]
            if embed_hidden is not None:
                eb += [nn.Linear(head_in, embed_hidden), nn.GELU(), nn.Dropout(dropout),
                       nn.Linear(embed_hidden, embed_dim)]
            else:
                eb += [nn.Linear(head_in, embed_dim)]
            self.embed = nn.Sequential(*eb)
            clf: list[nn.Module] = [nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(max(0, n_hidden_layers - 1)):
                clf += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            clf.append(nn.Linear(hidden_dim, 1))
            self.classifier = nn.Sequential(*clf)
            self.head = None

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.input_proj is None else x @ self.input_proj

    def forward(
        self,
        base_image: torch.Tensor,
        wrist_image: torch.Tensor,
        language: torch.Tensor,
        language_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Raw logits, shape (batch,). Apply sigmoid for a probability.

        With embed_dim set and return_embedding=True, returns (logits, z) where
        z is the (batch, embed_dim) latent -- the target for CLIP alignment.
        """
        parts = []
        if self.pool_base is not None:
            parts.append(self.pool_base(self._proj(base_image)))
        if self.pool_wrist is not None:
            parts.append(self.pool_wrist(self._proj(wrist_image)))
        if self.pool_lang is not None:
            parts.append(self.pool_lang(self._proj(language), language_mask))
        pooled = torch.cat(parts, dim=-1)

        if self.embed_dim is None:
            logits = self.head(pooled).squeeze(-1)
            return (logits, None) if return_embedding else logits
        z = self.embed(pooled)
        logits = self.classifier(z).squeeze(-1)
        return (logits, z) if return_embedding else logits
