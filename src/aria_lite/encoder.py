"""
ARIA-Lite Grid Encoder

Encodes 2D grid observations (H, W) with integer colors (0-15) into
a compact latent representation of shape [B, 256].

Architecture:
    Input: [B, H, W] int tensor
    → Color Embedding (16 → 64)
    → 2D Positional Encoding (64)
    → CNN: (128 → 256 → 512) with GroupNorm + GELU
    → Transformer: 3 layers, 8 heads, 512 dim
    → Pooling + Projection → [B, 256]

Target: ~5M parameters
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARIALiteConfig, EncoderConfig


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for grid inputs."""

    def __init__(self, dim: int, max_size: int = 64):
        super().__init__()
        self.dim = dim
        self.max_size = max_size

        # Precompute encodings
        pe = self._create_encoding(max_size, dim)
        self.register_buffer("pe", pe)

    def _create_encoding(self, max_size: int, dim: int) -> torch.Tensor:
        """Create 2D sinusoidal encoding."""
        half_dim = dim // 2

        # Position indices
        y_pos = torch.arange(max_size).unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(max_size).unsqueeze(0)  # [1, W]

        # Frequency bands
        div_term = torch.exp(
            torch.arange(0, half_dim, 2) * (-math.log(10000.0) / half_dim)
        )

        # Y encoding
        pe_y = torch.zeros(max_size, half_dim)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        # X encoding
        pe_x = torch.zeros(max_size, half_dim)
        pe_x[:, 0::2] = torch.sin(x_pos.T * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos.T * div_term)

        # Combine: [H, W, dim]
        pe = torch.zeros(max_size, max_size, dim)
        pe[:, :, :half_dim] = pe_y.unsqueeze(1).expand(-1, max_size, -1)
        pe[:, :, half_dim:] = pe_x.unsqueeze(0).expand(max_size, -1, -1)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: [B, C, H, W] tensor

        Returns:
            [B, C + dim, H, W] tensor with positional encoding concatenated
        """
        B, C, H, W = x.shape
        # Get relevant portion of precomputed encoding
        pe = self.pe[:H, :W, :]  # [H, W, dim]
        pe = pe.permute(2, 0, 1)  # [dim, H, W]
        pe = pe.unsqueeze(0).expand(B, -1, -1, -1)  # [B, dim, H, W]

        return torch.cat([x, pe], dim=1)


class CNNBlock(nn.Module):
    """CNN block with GroupNorm and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_residual: bool = True,
        downsample: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and not downsample

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # GroupNorm with 8 groups (or fewer if channels < 8)
        num_groups = min(8, out_channels)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()

        self.downsample_flag = downsample

        # Projection for residual if dimensions change
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        if self.downsample_flag:
            # Use adaptive pooling to handle any input size
            H, W = out.shape[-2:]
            target_H = max(1, H // 2)
            target_W = max(1, W // 2)
            out = F.adaptive_max_pool2d(out, (target_H, target_W))
            if identity.shape[-1] != out.shape[-1]:
                identity = F.adaptive_avg_pool2d(identity, out.shape[-2:])

        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            out = out + identity

        return out


class TransformerBlock(nn.Module):
    """Standard transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _attn_weights = self.attn(normed, normed, normed, key_padding_mask=mask)
        del _attn_weights  # Not used
        x = x + attn_out

        # Feedforward with pre-norm
        x = x + self.ff(self.norm2(x))

        return x


class GridEncoderLite(nn.Module):
    """
    Grid encoder for ARIA-Lite.

    Converts grid observations [B, H, W] with integer colors (0-15) into
    latent state vectors [B, output_dim].

    Architecture:
        1. Color embedding: 16 colors → color_embed_dim
        2. Positional encoding: 2D sinusoidal → pos_embed_dim
        3. CNN backbone: Progressive downsampling
        4. Transformer: Global attention
        5. Pooling + projection: → output_dim
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()

        if config is None:
            config = EncoderConfig()

        self.config = config
        self.output_dim = config.output_dim

        # Color embedding
        self.color_embed = nn.Embedding(config.num_colors, config.color_embed_dim)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding2D(
            config.pos_embed_dim, config.max_grid_size
        )

        # Build CNN backbone
        cnn_input_dim = config.color_embed_dim + config.pos_embed_dim
        cnn_layers = []

        in_channels = cnn_input_dim
        for i, out_channels in enumerate(config.cnn_channels):
            # Downsample on first two blocks
            downsample = i < 2
            cnn_layers.append(
                CNNBlock(
                    in_channels,
                    out_channels,
                    config.cnn_kernel_size,
                    use_residual=True,
                    downsample=downsample,
                )
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Projection from CNN to transformer dim
        self.cnn_to_transformer = nn.Linear(
            config.cnn_channels[-1], config.transformer_dim
        )

        # Transformer encoder
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.transformer_dim,
                    config.transformer_heads,
                    config.transformer_ff_dim,
                    config.transformer_dropout,
                )
                for _ in range(config.transformer_layers)
            ]
        )

        # Output projection
        self.output_norm = nn.LayerNorm(config.transformer_dim)
        self.output_proj = nn.Linear(config.transformer_dim, config.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        grid: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode grid observation to latent state.

        Args:
            grid: [B, H, W] int tensor with values 0-15
            mask: Optional [B, H, W] bool tensor (True = masked/invalid)

        Returns:
            [B, output_dim] latent state tensor
        """
        # Color embedding: [B, H, W] → [B, H, W, C] → [B, C, H, W]
        x = self.color_embed(grid)  # [B, H, W, color_embed_dim]
        x = x.permute(0, 3, 1, 2)  # [B, color_embed_dim, H, W]

        # Add positional encoding: [B, C, H, W] → [B, C + pos_dim, H, W]
        x = self.pos_encoding(x)

        # CNN: [B, C_in, H, W] → [B, C_out, H', W']
        x = self.cnn(x)

        # Reshape for transformer: [B, C, H', W'] → [B, H'*W', C]
        _, _, H_out, W_out = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, H'*W', C]

        # Project to transformer dim
        x = self.cnn_to_transformer(x)  # [B, seq_len, transformer_dim]

        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            # Downsample mask to match CNN output size
            mask_downsampled = F.adaptive_max_pool2d(
                mask.float().unsqueeze(1), (H_out, W_out)
            )
            attn_mask = mask_downsampled.squeeze(1).flatten(1).bool()  # [B, seq_len]

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask=attn_mask)

        # Global average pooling over sequence
        if attn_mask is not None:
            # Masked mean pooling
            mask_expanded = (~attn_mask).unsqueeze(-1).float()  # [B, seq_len, 1]
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # [B, transformer_dim]

        # Output projection
        x = self.output_norm(x)
        x = self.output_proj(x)  # [B, output_dim]

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_encoder(config: Optional[ARIALiteConfig] = None) -> GridEncoderLite:
    """Factory function to create encoder from full config."""
    if config is None:
        config = ARIALiteConfig()
    return GridEncoderLite(config.encoder)
