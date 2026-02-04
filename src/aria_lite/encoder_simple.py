"""
Simple Grid Encoder for BC validation.

A minimal encoder that doesn't aggressively downsample,
suitable for small (10x10) grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGridEncoder(nn.Module):
    """
    Simple encoder for small grids.

    No aggressive downsampling - preserves spatial info.
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.output_dim = output_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Simple CNN without downsampling
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        # Position-aware pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed 4x4 output

        # Output projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        grid: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode grid to latent vector.

        Args:
            grid: [B, H, W] int tensor with cell types
            mask: [B, H, W] bool tensor, True = invalid (unused for now)

        Returns:
            [B, output_dim] latent vectors
        """
        # Embed colors: [B, H, W] -> [B, H, W, E] -> [B, E, H, W]
        x = self.color_embed(grid)
        x = x.permute(0, 3, 1, 2)

        # CNN (no downsampling)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Pool to fixed size
        x = self.pool(x)

        # Project to output
        x = self.fc(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
