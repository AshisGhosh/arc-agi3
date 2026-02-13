"""
v4 CNN model: predicts P(state_novelty | state, action).

Two output heads:
  - Action head: P(novelty) for simple actions 1-5
  - Coordinate head: P(novelty) for click at each pixel (64x64)

Learns online during gameplay. No pretraining.
Uses GroupNorm instead of BatchNorm (works correctly with batch_size=1 inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with sensible group count for our channel sizes."""
    # 8 groups works for 32, 64, 128, 256, 512 channels
    return nn.GroupNorm(min(8, channels), channels)


class ResBlock(nn.Module):
    """Residual block with optional channel projection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _gn(out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class FrameChangeCNN(nn.Module):
    """CNN predicting P(state_novelty | state, action).

    Input: one-hot encoded frame [B, 16, 64, 64]
    Output: action_logits [B, 5], coord_logits [B, 64, 64]

    Sizes:
      - "goose": StochasticGoose architecture — 4 conv at full res, MaxPool4x4,
        flatten+FC. Preserves spatial info for directional actions. ~34M params.
      - "small": ResNet with GAP, ~1.9M params (mostly in conv)
      - "medium": Wider ResNet with GAP, ~7.7M params
      - "large": 5 conv layers, ~35M params (legacy flatten+linear)
    """

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size

        if size == "goose":
            # StochasticGoose architecture: 4 conv at full 64x64 resolution,
            # then MaxPool 4x4 → 16x16, flatten → FC.
            # Key: spatial info preserved through large FC layer.
            self.backbone = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self._feat_ch = 256
            self._feat_h = 64  # full resolution

            self.action_head = nn.Sequential(
                nn.MaxPool2d(4),  # 64x64 → 16x16
                nn.Flatten(),
                nn.Linear(256 * 16 * 16, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 5),
            )

            self.coord_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )

        elif size == "large":
            # Legacy large model with flatten+linear
            self.backbone = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                _gn(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # -> [64, 32, 32]
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                _gn(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                _gn(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # -> [256, 16, 16]
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                _gn(256),
                nn.ReLU(inplace=True),
            )
            self._feat_ch = 256
            self._feat_h = 16

            self.action_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 16 * 16, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, 5),
            )

            self.coord_head = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                _gn(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

        elif size == "medium":
            # Wider ResNet backbone with GAP action head
            # 16→64→128→256→512, 3 maxpools → [512, 8, 8]
            self.backbone = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=3, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                ResBlock(64, 64),
                nn.MaxPool2d(2),      # -> [64, 32, 32]
                ResBlock(64, 128),
                nn.MaxPool2d(2),      # -> [128, 16, 16]
                ResBlock(128, 256),
                nn.MaxPool2d(2),      # -> [256, 8, 8]
                ResBlock(256, 512),
            )
            self._feat_ch = 512
            self._feat_h = 8

            # GAP + MLP for action head — capacity in conv, not linear
            self.action_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 5),
            )

            # Coord head: upsample from 8x8 to 64x64
            self.coord_head = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                _gn(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                _gn(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

        else:
            # Small ResNet backbone with GAP action head
            # 16→64→128→256, 3 maxpools → [256, 8, 8]
            self.backbone = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=3, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                ResBlock(64, 64),
                nn.MaxPool2d(2),      # -> [64, 32, 32]
                ResBlock(64, 128),
                nn.MaxPool2d(2),      # -> [128, 16, 16]
                ResBlock(128, 256),
                nn.MaxPool2d(2),      # -> [256, 8, 8]
            )
            self._feat_ch = 256
            self._feat_h = 8

            # GAP + MLP for action head
            self.action_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 5),
            )

            # Coord head: upsample from 8x8 to 64x64
            self.coord_head = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                _gn(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                _gn(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                _gn(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )

    def forward(
        self, x: torch.Tensor, need_coord: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: one-hot encoded frame [B, 16, 64, 64]
            need_coord: if False, skip coordinate head (faster for non-click games)

        Returns:
            action_logits: [B, 5] raw logits for actions 1-5
            coord_logits: [B, 64, 64] raw logits for click at each pixel, or None
        """
        features = self.backbone(x)
        action_logits = self.action_head(features)
        if need_coord:
            coord_logits = self.coord_head(features).squeeze(1)  # [B, 64, 64]
        else:
            coord_logits = None
        return action_logits, coord_logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def frame_to_onehot(frame: torch.Tensor) -> torch.Tensor:
    """Convert frame [B, 64, 64] (values 0-15) to one-hot [B, 16, 64, 64]."""
    return F.one_hot(frame.long(), 16).permute(0, 3, 1, 2).float()
