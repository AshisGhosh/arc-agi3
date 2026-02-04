"""
Template-aware encoders for pattern matching.

Two approaches:
1. Cross-attention (queries attend to grid) - works but slow to learn
2. Convolutional (slide template over grid) - fast and perfect generalization

The convolutional approach is preferred as template matching is inherently
a sliding window comparison operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.d_model = d_model

        # Create position encodings
        pe = torch.zeros(max_h, max_w, d_model)

        y_pos = torch.arange(max_h).unsqueeze(1).unsqueeze(2)
        x_pos = torch.arange(max_w).unsqueeze(0).unsqueeze(2)

        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2) * (-math.log(10000.0) / (d_model // 2))
        )

        # Y position encoding (first quarter of channels)
        pe[:, :, 0::4] = torch.sin(y_pos * div_term)
        pe[:, :, 1::4] = torch.cos(y_pos * div_term)
        # X position encoding (second quarter of channels)
        pe[:, :, 2::4] = torch.sin(x_pos * div_term)
        pe[:, :, 3::4] = torch.cos(x_pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [B, H, W, D] tensor

        Returns:
            [B, H, W, D] tensor with positional encoding added
        """
        B, H, W, D = x.shape
        return x + self.pe[:H, :W, :D].unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to keys/values."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, Nq, D] query tokens
            key: [B, Nk, D] key tokens
            value: [B, Nk, D] value tokens

        Returns:
            output: [B, Nq, D] attended output
            attn_weights: [B, n_heads, Nq, Nk] attention weights
        """
        B, Nq, D = query.shape
        _, Nk, _ = key.shape

        # Project
        q = self.q_proj(query).view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, D)
        attn_output = self.out_proj(attn_output)

        # Residual + norm
        query = self.norm1(query + attn_output)

        # FFN
        query = self.norm2(query + self.ffn(query))

        return query, attn_weights


class TemplateMatchingEncoder(nn.Module):
    """
    Encoder that uses cross-attention to find template in grid.

    Architecture:
    1. Embed grid cells and template cells separately
    2. Add 2D positional encoding to grid
    3. Use cross-attention: template queries grid
    4. Pool attention weights to predict match location
    """

    def __init__(
        self,
        num_colors: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_grid: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_grid = max_grid

        # Cell embedding
        self.cell_embed = nn.Embedding(num_colors, d_model)

        # Positional encoding for grid (not template - template is position-agnostic)
        self.pos_encoding = PositionalEncoding2D(d_model, max_grid, max_grid)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # Template self-attention to build template representation
        self.template_self_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )

        # Output heads for coordinate prediction
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.x_head = nn.Linear(d_model, max_grid)
        self.y_head = nn.Linear(d_model, max_grid)

    def forward(
        self,
        grid: torch.Tensor,
        template: torch.Tensor,
        grid_size: int = 10,
    ) -> dict:
        """
        Find template location in grid.

        Args:
            grid: [B, H, W] main grid
            template: [B, Th, Tw] template to find
            grid_size: actual grid size for output masking

        Returns:
            dict with x_logits, y_logits, attention_map
        """
        B = grid.shape[0]
        H, W = grid.shape[1], grid.shape[2]
        Th, Tw = template.shape[1], template.shape[2]

        # Embed grid: [B, H, W, D]
        grid_embed = self.cell_embed(grid.long().clamp(0, 15))
        grid_embed = self.pos_encoding(grid_embed)
        grid_flat = grid_embed.view(B, H * W, self.d_model)  # [B, H*W, D]

        # Embed template: [B, Th, Tw, D]
        template_embed = self.cell_embed(template.long().clamp(0, 15))
        template_flat = template_embed.view(B, Th * Tw, self.d_model)  # [B, Th*Tw, D]

        # Build template representation with self-attention
        template_repr = self.template_self_attn(template_flat)  # [B, Th*Tw, D]

        # Cross-attention: template queries grid
        query = template_repr
        key = value = grid_flat

        all_attn_weights = []
        for layer in self.cross_attn_layers:
            query, attn_weights = layer(query, key, value)
            all_attn_weights.append(attn_weights)

        # Average attention across template tokens and heads
        # attn_weights: [B, n_heads, Th*Tw, H*W]
        final_attn = all_attn_weights[-1]
        attn_map = final_attn.mean(dim=1).mean(dim=1)  # [B, H*W]
        attn_map = attn_map.view(B, H, W)  # [B, H, W]

        # Pool attended features
        attended_features = query.mean(dim=1)  # [B, D]
        features = self.coord_head(attended_features)

        # Predict coordinates
        x_logits = self.x_head(features)[:, :grid_size]
        y_logits = self.y_head(features)[:, :grid_size]

        # Also use attention map directly for coordinate prediction
        # Find argmax of attention map
        attn_y = attn_map.sum(dim=2)  # [B, H] - sum across x
        attn_x = attn_map.sum(dim=1)  # [B, W] - sum across y

        # Combine learned and attention-based predictions
        x_logits = x_logits + attn_x[:, :grid_size]
        y_logits = y_logits + attn_y[:, :grid_size]

        return {
            "x_logits": x_logits,
            "y_logits": y_logits,
            "attention_map": attn_map,
            "features": features,
        }


class PatternMatchingPolicy(nn.Module):
    """Policy for pattern matching using template-aware encoder."""

    def __init__(
        self,
        num_colors: int = 16,
        d_model: int = 64,
        max_grid: int = 20,
    ):
        super().__init__()
        self.encoder = TemplateMatchingEncoder(
            num_colors=num_colors,
            d_model=d_model,
            max_grid=max_grid,
        )

        # Action head (usually just CLICK for pattern matching)
        self.action_head = nn.Linear(d_model, 9)

    def forward(
        self,
        obs: torch.Tensor,
        template: torch.Tensor,
        grid_size: int = 10,
    ) -> dict:
        """
        Args:
            obs: [B, H, W] observation (main grid)
            template: [B, Th, Tw] template to match
            grid_size: grid size for coordinate masking

        Returns:
            dict with action_logits, x_logits, y_logits
        """
        out = self.encoder(obs, template, grid_size)

        action_logits = self.action_head(out["features"])

        return {
            "action_logits": action_logits,
            "x_logits": out["x_logits"],
            "y_logits": out["y_logits"],
            "attention_map": out["attention_map"],
        }


class ConvTemplateMatching(nn.Module):
    """
    Template matching using convolution - the preferred approach.

    Key insight: Template matching is inherently a convolution operation.
    1. Embed grid and template to feature space
    2. Use template features as conv kernel
    3. Slide over grid to get match scores
    4. Argmax of match scores = template location

    This achieves 100% generalization because the operation (convolution)
    naturally captures what template matching is.
    """

    def __init__(self, num_colors: int = 16, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding for cell values
        self.embed = nn.Embedding(num_colors, embed_dim)

        # Project embeddings to matching space
        self.grid_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

        self.template_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, grid: torch.Tensor, template: torch.Tensor) -> dict:
        """
        Find template in grid using convolution.

        Args:
            grid: [B, H, W] main grid
            template: [B, Th, Tw] template

        Returns:
            dict with match_scores [B, H-Th+1, W-Tw+1] and predicted x, y
        """
        B = grid.shape[0]
        H, W = grid.shape[1], grid.shape[2]
        Th, Tw = template.shape[1], template.shape[2]

        # Embed
        grid_embed = self.embed(grid.long().clamp(0, 15))  # [B, H, W, D]
        grid_embed = grid_embed.permute(0, 3, 1, 2)  # [B, D, H, W]

        template_embed = self.embed(template.long().clamp(0, 15))  # [B, Th, Tw, D]
        template_embed = template_embed.permute(0, 3, 1, 2)  # [B, D, Th, Tw]

        # Project
        grid_feat = self.grid_proj(grid_embed)  # [B, D, H, W]
        template_feat = self.template_proj(template_embed)  # [B, D, Th, Tw]

        # Compute match scores via convolution
        # For each sample, use its template as the conv kernel
        match_scores = []
        for b in range(B):
            g = grid_feat[b:b+1]  # [1, D, H, W]
            t = template_feat[b]  # [D, Th, Tw]

            # Convolution: slide template over grid
            score = F.conv2d(g, t.unsqueeze(0))
            match_scores.append(score)

        match_scores = torch.cat(match_scores, dim=0)  # [B, 1, H-Th+1, W-Tw+1]
        match_scores = match_scores.squeeze(1)  # [B, H-Th+1, W-Tw+1]

        # Find argmax
        flat_scores = match_scores.view(B, -1)
        max_idx = flat_scores.argmax(dim=-1)

        out_h = H - Th + 1
        out_w = W - Tw + 1

        pred_y = max_idx // out_w
        pred_x = max_idx % out_w

        return {
            "match_scores": match_scores,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "x_logits": match_scores.sum(dim=1),  # [B, W'] for compatibility
            "y_logits": match_scores.sum(dim=2),  # [B, H'] for compatibility
        }


class DifferenceMatchingEncoder(nn.Module):
    """
    Encoder for spot-the-difference task.

    Uses self-attention across both grids to find differences.
    """

    def __init__(
        self,
        num_colors: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        max_grid: int = 20,
    ):
        super().__init__()
        self.d_model = d_model

        self.cell_embed = nn.Embedding(num_colors, d_model)
        self.pos_encoding = PositionalEncoding2D(d_model, max_grid, max_grid)

        # Grid indicator embedding (which grid: 0 or 1)
        self.grid_embed = nn.Embedding(2, d_model)

        # Self-attention across both grids
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Difference detection head
        self.diff_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),  # Score per cell
        )

    def forward(self, obs: torch.Tensor, grid_size: int = 8) -> dict:
        """
        Find difference between two grids stacked in obs.

        Args:
            obs: [B, 2*H, W] two grids stacked vertically

        Returns:
            dict with x_logits, y_logits
        """
        B, total_H, W = obs.shape
        H = total_H // 2

        # Split into two grids
        grid1 = obs[:, :H, :]  # [B, H, W]
        grid2 = obs[:, H:, :]  # [B, H, W]

        # Embed both grids
        embed1 = self.cell_embed(grid1.long().clamp(0, 15))  # [B, H, W, D]
        embed2 = self.cell_embed(grid2.long().clamp(0, 15))

        # Add positional encoding
        embed1 = self.pos_encoding(embed1)
        embed2 = self.pos_encoding(embed2)

        # Add grid indicator
        embed1 = embed1 + self.grid_embed(torch.zeros(1, dtype=torch.long, device=obs.device))
        embed2 = embed2 + self.grid_embed(torch.ones(1, dtype=torch.long, device=obs.device))

        # Flatten and concatenate
        flat1 = embed1.view(B, H * W, self.d_model)
        flat2 = embed2.view(B, H * W, self.d_model)
        combined = torch.cat([flat1, flat2], dim=1)  # [B, 2*H*W, D]

        # Self-attention
        features = self.transformer(combined)

        # Get features for second grid (where difference is marked)
        grid2_features = features[:, H * W:, :]  # [B, H*W, D]

        # Score each cell for being the difference
        scores = self.diff_head(grid2_features).squeeze(-1)  # [B, H*W]
        scores = scores.view(B, H, W)  # [B, H, W]

        # Convert to coordinate logits
        x_logits = scores.sum(dim=1)[:, :grid_size]  # [B, W] -> [B, grid_size]
        y_logits = scores.sum(dim=2)[:, :grid_size]  # [B, H] -> [B, grid_size]

        return {
            "x_logits": x_logits,
            "y_logits": y_logits,
            "diff_scores": scores,
        }
