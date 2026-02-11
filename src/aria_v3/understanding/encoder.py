"""
CNN Transition Encoder.

Takes (frame_t, action, frame_t+1) and produces a 256-dim transition embedding.
The encoder learns what changed between frames and how the action relates to the change.

Input: [B, 40, 64, 64] = one_hot(frame_t)[16] + one_hot(frame_t+1)[16] + action_embed[8]
Output: [B, 256] global embedding + [B, 256, 4, 4] spatial features

~580K parameters. Runs in ~0.3ms per transition on RTX 5090.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionEncoder(nn.Module):
    """CNN that encodes a (frame, action, next_frame) transition."""

    def __init__(self, num_actions: int = 8, embed_dim: int = 256):
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # Action embedding: action_id → 8-dim, broadcast to spatial
        self.action_embed = nn.Embedding(num_actions, 8)

        # Input: 16 (frame_t one-hot) + 16 (frame_t+1 one-hot) + 8 (action) = 40 channels
        self.conv1 = nn.Conv2d(40, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # → 32x32
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # → 16x16
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # → 8x8
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, embed_dim, 3, stride=2, padding=1)  # → 4x4
        self.bn5 = nn.BatchNorm2d(embed_dim)

    def forward(
        self,
        frame: torch.Tensor,
        action: torch.Tensor,
        next_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a transition.

        Args:
            frame: [B, 64, 64] long tensor (values 0-15)
            action: [B] long tensor (values 0-7)
            next_frame: [B, 64, 64] long tensor (values 0-15)

        Returns:
            global_embed: [B, 256] global transition embedding
            spatial_features: [B, 256, 4, 4] spatial feature map
        """
        B = frame.shape[0]

        # One-hot encode frames: [B, 16, 64, 64]
        frame_oh = F.one_hot(frame.long(), 16).permute(0, 3, 1, 2).float()
        next_oh = F.one_hot(next_frame.long(), 16).permute(0, 3, 1, 2).float()

        # Action embedding broadcast to spatial: [B, 8, 64, 64]
        act_emb = self.action_embed(action.long())  # [B, 8]
        act_spatial = act_emb.unsqueeze(-1).unsqueeze(-1).expand(B, 8, 64, 64)

        # Concatenate: [B, 40, 64, 64]
        x = torch.cat([frame_oh, next_oh, act_spatial], dim=1)

        # CNN forward
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 256, 4, 4]

        spatial_features = x
        global_embed = x.mean(dim=(2, 3))  # [B, 256]

        return global_embed, spatial_features

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FramePredictor(nn.Module):
    """Prediction head for TTT: predict next frame from (frame, action).

    Used during test-time training. Shares the encoder but adds a decoder
    that reconstructs the next frame from spatial features.
    """

    def __init__(self, embed_dim: int = 256, num_colors: int = 16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 4x4 → 8x8
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 8x8 → 16x16
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 16x16 → 32x32
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32x32 → 64x64
            nn.Conv2d(16, num_colors, 1),  # → [B, 16, 64, 64] logits
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Predict next frame from spatial features.

        Args:
            spatial_features: [B, 256, 4, 4] from encoder

        Returns:
            logits: [B, 16, 64, 64] per-pixel color logits
        """
        return self.decoder(spatial_features)
