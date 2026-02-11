"""
v4 CNN model: predicts P(frame_change | state, action).

Two output heads:
  - Action head: P(frame_change) for simple actions 1-5
  - Coordinate head: P(frame_change) for click at each pixel (64x64)

Learns online during gameplay. No pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameChangeCNN(nn.Module):
    """CNN predicting P(frame_change | state, action).

    Input: one-hot encoded frame [B, 16, 64, 64]
    Output: action_logits [B, 5], coord_logits [B, 1, 64, 64]
    """

    def __init__(self):
        super().__init__()

        # Shared backbone: 16 → 32 → 64 → 128 → 256
        self.backbone = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [64, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [128, 16, 16]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [256, 8, 8]
        )

        # Action head: predict P(frame_change) for actions 1-5
        # Uses flattened spatial features (16384) for spatial awareness
        # (StochasticGoose uses 65536; our backbone gives 256×8×8=16384)
        self.action_head = nn.Sequential(
            nn.Flatten(),  # → [16384]
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 5),  # 5 simple actions
        )

        # Coordinate head: predict P(frame_change) for click at each pixel
        self.coord_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # → [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # → [1, 64, 64]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: one-hot encoded frame [B, 16, 64, 64]

        Returns:
            action_logits: [B, 5] raw logits for actions 1-5
            coord_logits: [B, 64, 64] raw logits for click at each pixel
        """
        features = self.backbone(x)
        action_logits = self.action_head(features)
        coord_logits = self.coord_head(features).squeeze(1)  # [B, 64, 64]
        return action_logits, coord_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def frame_to_onehot(frame: torch.Tensor) -> torch.Tensor:
    """Convert frame [B, 64, 64] (values 0-15) to one-hot [B, 16, 64, 64]."""
    return F.one_hot(frame.long(), 16).permute(0, 3, 1, 2).float()
