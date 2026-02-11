"""
Understanding Decoder Heads.

Two input paths:
1. Understanding state (query tokens) from temporal transformer
   → Game type classification, confidence
2. Per-action aggregated embeddings + shift classification
   → Action effects
3. Per-color frame statistics → entity roles

Key design: shift prediction uses CLASSIFICATION (7 discrete bins per axis)
instead of regression, avoiding regression-to-mean on randomized action mappings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_GAME_TYPES = 8
NUM_ENTITY_ROLES = 5  # player, wall, collectible, background, counter
NUM_ACTIONS = 8
NUM_COLORS = 16

# Discrete shift bins: [-16, -8, -4, 0, +4, +8, +16]
SHIFT_BINS = [-16, -8, -4, 0, 4, 8, 16]
NUM_SHIFT_BINS = len(SHIFT_BINS)


def shift_to_class(shift_value: float) -> int:
    """Convert continuous shift to class index."""
    for i, v in enumerate(SHIFT_BINS):
        if abs(shift_value - v) < 2:
            return i
    return 3  # default: 0 shift


def class_to_shift(class_idx: int) -> float:
    """Convert class index back to shift value."""
    return float(SHIFT_BINS[class_idx])


class ActionEffectHead(nn.Module):
    """Predict per-action effects by grouping transition embeddings by action type.

    Shift prediction uses 7-way classification per axis instead of regression.
    """

    def __init__(self, d_model: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions

        # Shift classifier: per-axis 7-way classification
        self.shift_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * NUM_SHIFT_BINS),  # 2 axes × 7 bins
        )

        # Other effects: change_prob, blocked_prob, affected_color
        self.effects_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict action effects by grouping embeddings by action type."""
        B, L, D = embeddings.shape
        device = embeddings.device

        # Group embeddings by action type
        action_features = torch.zeros(B, self.num_actions, D, device=device)
        action_counts = torch.zeros(B, self.num_actions, device=device)

        for a in range(self.num_actions):
            action_mask = (actions == a)
            if mask is not None:
                action_mask = action_mask & mask
            count = action_mask.float().sum(dim=1, keepdim=True)
            expanded_mask = action_mask.unsqueeze(-1).float()
            summed = (embeddings * expanded_mask).sum(dim=1)
            safe_count = count.squeeze(-1).clamp(min=1)
            action_features[:, a, :] = summed / safe_count.unsqueeze(-1)
            action_counts[:, a] = count.squeeze(-1)

        # Shift classification
        shift_logits = self.shift_mlp(action_features)  # [B, A, 14]
        shift_dx_logits = shift_logits[:, :, :NUM_SHIFT_BINS]  # [B, A, 7]
        shift_dy_logits = shift_logits[:, :, NUM_SHIFT_BINS:]  # [B, A, 7]

        # Convert to continuous shift values (for compatibility)
        bins = torch.tensor(SHIFT_BINS, dtype=torch.float32, device=device)
        shift_dx = (F.softmax(shift_dx_logits, dim=-1) * bins).sum(dim=-1)  # [B, A]
        shift_dy = (F.softmax(shift_dy_logits, dim=-1) * bins).sum(dim=-1)  # [B, A]
        shift = torch.stack([shift_dx, shift_dy], dim=-1)  # [B, A, 2]

        # Other effects
        effects = self.effects_mlp(action_features)  # [B, A, 3]

        # Zero out unseen actions
        seen = (action_counts > 0).unsqueeze(-1).float()

        return {
            "shift": shift * seen,
            "shift_dx_logits": shift_dx_logits,
            "shift_dy_logits": shift_dy_logits,
            "change_prob": torch.sigmoid(effects[:, :, 0]) * seen.squeeze(-1),
            "blocked_prob": torch.sigmoid(effects[:, :, 1]) * seen.squeeze(-1),
            "affected_color": effects[:, :, 2] * seen.squeeze(-1),
        }


class EntityRoleHead(nn.Module):
    """Predict entity roles from per-color frame statistics.

    Computes 5 statistics per color from frame diffs, then classifies roles.
    """

    def __init__(self, stats_dim: int = 5, num_roles: int = NUM_ENTITY_ROLES):
        super().__init__()
        self.num_roles = num_roles
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_roles),
        )

    def forward(
        self,
        frames: torch.Tensor,
        next_frames: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict entity roles from frame statistics."""
        B, L = frames.shape[:2]
        device = frames.device

        stats = self._compute_stats(frames, next_frames, mask)
        return self.mlp(stats)  # [B, 16, num_roles]

    def _compute_stats(
        self,
        frames: torch.Tensor,
        next_frames: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-color statistics efficiently."""
        B, L = frames.shape[:2]
        device = frames.device
        npix = 64 * 64

        # One-hot encode both frames: [B, L, 16, 64, 64]
        frames_oh = F.one_hot(frames.long(), NUM_COLORS).permute(0, 1, 4, 2, 3).float()
        next_oh = F.one_hot(next_frames.long(), NUM_COLORS).permute(0, 1, 4, 2, 3).float()

        if mask is not None:
            m = mask.float().view(B, L, 1, 1, 1)
            frames_oh = frames_oh * m
            next_oh = next_oh * m
            valid_count = mask.float().sum(dim=1).clamp(min=1)  # [B]
        else:
            valid_count = torch.tensor(L, device=device, dtype=torch.float32).expand(B)

        # Area: average fraction of pixels per color [B, 16]
        area = frames_oh.sum(dim=(1, 3, 4)) / (valid_count.unsqueeze(-1) * npix)

        # Appear: pixels that become this color [B, 16]
        appeared = ((1 - frames_oh) * next_oh).sum(dim=(1, 3, 4)) / (valid_count.unsqueeze(-1) * npix)

        # Disappear: pixels that were this color and aren't anymore [B, 16]
        disappeared = (frames_oh * (1 - next_oh)).sum(dim=(1, 3, 4)) / (valid_count.unsqueeze(-1) * npix)

        # Volatility
        volatility = appeared + disappeared

        # Static ratio: fraction of color's pixels that don't change
        same = (frames_oh * next_oh).sum(dim=(1, 3, 4))  # [B, 16]
        total_c = frames_oh.sum(dim=(1, 3, 4)).clamp(min=1)  # [B, 16]
        static_ratio = same / total_c

        # Stack: [B, 16, 5]
        stats = torch.stack([area, appeared, disappeared, volatility, static_ratio], dim=-1)
        return stats


class GameTypeHead(nn.Module):
    """Classify game type from transformer understanding state."""

    def __init__(self, d_model: int = 256, num_types: int = NUM_GAME_TYPES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_types),
        )

    def forward(self, understanding: torch.Tensor) -> torch.Tensor:
        pooled = understanding.mean(dim=1)
        return self.mlp(pooled)


class ConfidenceHead(nn.Module):
    """Predict understanding confidence (0-1)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, understanding: torch.Tensor) -> torch.Tensor:
        pooled = understanding.mean(dim=1)
        return torch.sigmoid(self.mlp(pooled).squeeze(-1))


class UnderstandingDecoder(nn.Module):
    """Combined decoder: direct paths for actions/entities, transformer for game type."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.action_head = ActionEffectHead(d_model)
        self.entity_head = EntityRoleHead()
        self.game_type_head = GameTypeHead(d_model)
        self.confidence_head = ConfidenceHead(d_model)

    def forward(
        self,
        understanding: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        frames: torch.Tensor | None = None,
        next_frames: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        result = {}

        if embeddings is not None and actions is not None:
            action_effects = self.action_head(embeddings, actions, mask)
            result.update(action_effects)

        if frames is not None and next_frames is not None:
            result["entity_roles"] = self.entity_head(frames, next_frames, mask)

        result["game_type"] = self.game_type_head(understanding)
        result["confidence"] = self.confidence_head(understanding)

        return result

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
