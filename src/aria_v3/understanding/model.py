"""
Full Understanding Model.

Combines the CNN transition encoder, temporal transformer, and decoder heads
into a single module with a unified forward pass.

Total: ~3.1M params, ~6MB VRAM, ~1.8ms/action amortized.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransitionEncoder, FramePredictor
from .temporal import TemporalTransformer
from .decoder import UnderstandingDecoder


class UnderstandingModel(nn.Module):
    """Full understanding pipeline: encode transitions → attend → decode understanding."""

    def __init__(
        self,
        d_model: int = 256,
        num_actions: int = 8,
        nhead: int = 4,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_queries: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        # Stage 1: Per-transition CNN encoder
        self.encoder = TransitionEncoder(num_actions=num_actions, embed_dim=d_model)

        # Stage 2: Temporal transformer with query tokens
        self.temporal = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_queries=num_queries,
        )

        # Stage 3: Decoder heads
        self.decoder = UnderstandingDecoder(d_model=d_model)

        # TTT prediction head (used only during test-time training)
        self.frame_predictor = FramePredictor(embed_dim=d_model)

    def encode_transitions(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        next_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of transitions.

        Args:
            frames: [B, L, 64, 64] sequence of frames
            actions: [B, L] sequence of actions
            next_frames: [B, L, 64, 64] sequence of next frames

        Returns:
            embeddings: [B, L, d_model] transition embeddings
            spatial_features: [B, L, d_model, 4, 4] spatial features for TTT
        """
        B, L = frames.shape[:2]

        # Flatten batch and sequence dimensions
        frames_flat = frames.reshape(B * L, 64, 64)
        actions_flat = actions.reshape(B * L)
        next_flat = next_frames.reshape(B * L, 64, 64)

        # Encode all transitions
        global_emb, spatial = self.encoder(frames_flat, actions_flat, next_flat)

        # Reshape back
        embeddings = global_emb.reshape(B, L, self.d_model)
        spatial_features = spatial.reshape(B, L, self.d_model, 4, 4)

        return embeddings, spatial_features

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        next_frames: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode → attend → decode.

        Args:
            frames: [B, L, 64, 64] frame sequence
            actions: [B, L] action sequence
            next_frames: [B, L, 64, 64] next frame sequence
            mask: [B, L] bool, True for valid positions

        Returns:
            dict with all understanding predictions + transition embeddings
        """
        # Encode transitions
        embeddings, spatial_features = self.encode_transitions(
            frames, actions, next_frames
        )

        # Temporal attention
        understanding = self.temporal(embeddings, mask=mask)

        # Decode structured predictions
        # Pass raw data to decoder for direct-path heads (action grouping, frame stats)
        predictions = self.decoder(
            understanding,
            embeddings=embeddings,
            actions=actions,
            frames=frames,
            next_frames=next_frames,
            mask=mask,
        )

        # Include raw data for TTT and analysis
        predictions["embeddings"] = embeddings
        predictions["spatial_features"] = spatial_features
        predictions["understanding"] = understanding

        return predictions

    def predict_frames(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Predict next frames from spatial features (for TTT).

        Args:
            spatial_features: [B, L, 256, 4, 4] or [B*L, 256, 4, 4]

        Returns:
            logits: [*, 16, 64, 64] per-pixel color logits
        """
        shape = spatial_features.shape
        if len(shape) == 5:
            B, L = shape[:2]
            flat = spatial_features.reshape(B * L, self.d_model, 4, 4)
            logits = self.frame_predictor(flat)
            return logits.reshape(B, L, 16, 64, 64)
        else:
            return self.frame_predictor(spatial_features)

    def count_params(self) -> dict[str, int]:
        """Count parameters per component."""
        return {
            "encoder": self.encoder.count_params(),
            "temporal": self.temporal.count_params(),
            "decoder": self.decoder.count_params(),
            "frame_predictor": sum(
                p.numel() for p in self.frame_predictor.parameters()
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
