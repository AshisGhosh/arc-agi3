"""
Temporal Transformer for cross-transition reasoning.

Takes a sequence of transition embeddings and produces an "understanding state"
via DETR-style learnable query tokens that attend to the sequence.

Input: [B, L, 256] sequence of transition embeddings (L up to 200)
Output: [B, 16, 256] understanding state (16 query tokens)

~2M parameters. Runs in ~15ms every 100 actions on RTX 5090.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """Cross-attention transformer for temporal reasoning over transitions."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_queries: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Positional encoding for the transition sequence
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        # Learnable query tokens (DETR-style)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Transformer decoder: queries attend to transition sequence
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Layer norm on output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        transition_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process transition sequence and produce understanding state.

        Args:
            transition_embeddings: [B, L, d_model] sequence of embeddings
            mask: [B, L] bool tensor, True for valid positions

        Returns:
            understanding: [B, num_queries, d_model] query token outputs
        """
        B, L, D = transition_embeddings.shape

        # Add positional encoding
        positions = torch.arange(L, device=transition_embeddings.device)
        pos_emb = self.pos_encoding(positions)  # [L, D]
        memory = transition_embeddings + pos_emb.unsqueeze(0)  # [B, L, D]

        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        # Create padding mask if provided
        memory_key_padding_mask = None
        if mask is not None:
            memory_key_padding_mask = ~mask  # transformer expects True for padding

        # Cross-attention: queries attend to transition sequence
        output = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        output = self.output_norm(output)
        return output  # [B, num_queries, d_model]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
