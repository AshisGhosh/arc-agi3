"""
Position-Aware Encoder for Spatial ARC Tasks

Key insight: For copy/reflect, the model needs to:
1. Know WHERE in the output grid it's filling
2. ATTEND to relevant input positions
3. Learn position→position mappings

Architecture:
- 2D position embeddings for input grid
- Current cell position embedding for output
- Cross-attention: output position queries input grid
- This lets model learn "output[2,3] should look at input[2,7]" (for reflect)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """Learnable 2D position embeddings."""

    def __init__(self, grid_size: int, embed_dim: int):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # Separate row and column embeddings (factorized)
        self.row_embed = nn.Embedding(grid_size, embed_dim // 2)
        self.col_embed = nn.Embedding(grid_size, embed_dim // 2)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns position embeddings for full grid.

        Returns: [B, H, W, embed_dim]
        """
        rows = torch.arange(self.grid_size, device=device)
        cols = torch.arange(self.grid_size, device=device)

        row_emb = self.row_embed(rows)  # [H, D/2]
        col_emb = self.col_embed(cols)  # [W, D/2]

        # Combine: each position gets (row_emb, col_emb)
        # Create grid of embeddings
        pos_emb = torch.zeros(self.grid_size, self.grid_size, self.embed_dim, device=device)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos_emb[i, j] = torch.cat([row_emb[i], col_emb[j]])

        # Expand for batch
        return pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def get_position_embedding(self, row: int, col: int, device: torch.device) -> torch.Tensor:
        """Get embedding for a single position. Returns [embed_dim]."""
        row_t = torch.tensor([row], device=device)
        col_t = torch.tensor([col], device=device)
        return torch.cat([self.row_embed(row_t), self.col_embed(col_t)], dim=-1).squeeze(0)


class CrossAttention(nn.Module):
    """Cross-attention: query attends to key-value pairs."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, Q, D] - query positions
            key: [B, K, D] - key positions (input grid)
            value: [B, K, D] - values (input features)

        Returns: [B, Q, D]
        """
        B, Q, D = query.shape
        K = key.shape[1]

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Q, D)
        return self.out_proj(out)


class PositionAwareARCEncoder(nn.Module):
    """
    Position-aware encoder for spatial ARC tasks.

    Uses cross-attention to let current output position attend to input grid.
    """

    def __init__(
        self,
        num_colors: int = 4,
        grid_size: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Position embeddings
        self.pos_embed = PositionalEncoding2D(grid_size, embed_dim)

        # Process input grid
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # color + position
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cross-attention: current position queries input
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        # Process output so far
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # color + position
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Combine everything
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # attended + output_context + current_pos
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation with position-aware attention.

        Args:
            obs: [B, 3, H, W] - input_grid, output_grid, current_mask

        Returns: [B, output_dim]
        """
        B, _, H, W = obs.shape
        device = obs.device

        input_grid = obs[:, 0]   # [B, H, W]
        output_grid = obs[:, 1]  # [B, H, W]
        mask = obs[:, 2]         # [B, H, W] - 1 at current cell

        # Get position embeddings
        pos_emb = self.pos_embed(B, device)  # [B, H, W, D]

        # Embed input colors and combine with positions
        input_colors = self.color_embed(input_grid)  # [B, H, W, D]
        input_features = torch.cat([input_colors, pos_emb], dim=-1)  # [B, H, W, 2D]
        input_features = self.input_proj(input_features)  # [B, H, W, D]

        # Flatten input for attention
        input_flat = input_features.view(B, H * W, -1)  # [B, H*W, D]

        # Find current position
        current_pos = mask.nonzero(as_tuple=False)  # [N, 3] = (batch, row, col)

        # Get current position embedding for each batch element
        current_pos_emb = torch.zeros(B, self.embed_dim, device=device)
        for i in range(B):
            batch_mask = current_pos[:, 0] == i
            if batch_mask.any():
                idx = current_pos[batch_mask][0]
                row, col = idx[1].item(), idx[2].item()
                current_pos_emb[i] = self.pos_embed.get_position_embedding(row, col, device)

        # Cross-attention: current position queries input grid
        query = current_pos_emb.unsqueeze(1)  # [B, 1, D]
        attended = self.cross_attn(query, input_flat, input_flat)  # [B, 1, D]
        attended = attended.squeeze(1)  # [B, D]

        # Process output grid context (what we've filled so far)
        output_colors = self.color_embed(output_grid)  # [B, H, W, D]
        output_features = torch.cat([output_colors, pos_emb], dim=-1)
        output_features = self.output_proj(output_features)  # [B, H, W, D]
        output_context = output_features.mean(dim=(1, 2))  # [B, D] - average over grid

        # Combine: attended input + output context + current position
        combined = torch.cat([attended, output_context, current_pos_emb], dim=-1)
        return self.combine(combined)
