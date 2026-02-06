"""
VQ-VAE Frame Tokenizer for ARC-AGI-3 game frames.

Converts 64x64 16-color grid frames into 64 discrete tokens (8x8 spatial grid,
each token from a 512-code codebook). Uses EMA codebook updates for stability.

Architecture:
    Input: [B, 64, 64] int tensor (0-15)
    → nn.Embedding(16, 32) → [B, 64, 64, 32]
    → Encoder (4 conv layers) → [B, 128, 8, 8]
    → VectorQuantizer (512 codes, 128-dim, EMA) → [B, 8, 8] indices
    → Decoder (4 deconv layers) → [B, 16, 64, 64] logits
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    num_colors: int = 16
    color_embed_dim: int = 32
    hidden_dim: int = 128
    codebook_size: int = 512
    code_dim: int = 128
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: int = 2  # Reset codes unused for this many batches


class VectorQuantizer(nn.Module):
    """Vector quantization with EMA codebook updates and dead code reset."""

    def __init__(self, num_codes: int, code_dim: int, ema_decay: float = 0.99,
                 commitment_beta: float = 0.25, dead_code_threshold: int = 2):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.commitment_beta = commitment_beta
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
        self.register_buffer("usage_count", torch.zeros(num_codes))  # For dead code detection

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D, H, W] continuous latent features

        Returns:
            z_q: [B, D, H, W] quantized features (straight-through)
            indices: [B, H, W] codebook indices
            loss: commitment loss
        """
        B, D, H, W = z.shape

        # Reshape to [BHW, D] for distance computation
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 - 2*z@e^T + ||e||^2
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Find nearest codes
        indices = d.argmin(dim=1)  # [BHW]
        z_q_flat = self.embedding(indices)  # [BHW, D]

        # EMA updates during training (detach from graph to prevent memory leak)
        if self.training:
            with torch.no_grad():
                z_flat_detached = z_flat.detach()

                # Count code usage
                one_hot = F.one_hot(indices, self.num_codes).float()  # [BHW, K]
                cluster_size = one_hot.sum(0)  # [K]
                embed_sum = one_hot.t() @ z_flat_detached  # [K, D]

                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
                self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size_smooth = (
                    (self.ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                )

                # Update codebook
                self.embedding.weight.data.copy_(
                    self.ema_embed_sum / cluster_size_smooth.unsqueeze(1)
                )

                # Track usage for dead code detection
                used = (cluster_size > 0).float()
                self.usage_count.mul_(used)
                self.usage_count.add_(1 - used)

                # Reset dead codes
                dead_mask = self.usage_count >= self.dead_code_threshold
                if dead_mask.any():
                    n_dead = dead_mask.sum().item()
                    random_idx = torch.randint(
                        0, z_flat_detached.shape[0], (int(n_dead),), device=z.device
                    )
                    self.embedding.weight.data[dead_mask] = z_flat_detached[random_idx]
                    self.ema_embed_sum[dead_mask] = z_flat_detached[random_idx]
                    self.ema_cluster_size[dead_mask] = 1.0
                    self.usage_count[dead_mask] = 0

        # Commitment loss
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        indices = indices.reshape(B, H, W)

        return z_q, indices, self.commitment_beta * commitment_loss

    def get_utilization(self) -> float:
        """Fraction of codebook entries actively used."""
        return (self.ema_cluster_size > 1e-3).float().mean().item()


class Encoder(nn.Module):
    """Encode 64x64 color-embedded frames to 8x8 latent grid."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(in_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 → 16x16
            nn.Conv2d(64, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 → 8x8
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 8x8 → 8x8 (refine)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Decode 8x8 latent grid back to 64x64 color logits."""

    def __init__(self, hidden_dim: int = 128, num_colors: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            # 8x8 → 8x8 (refine)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(64, num_colors, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameVQVAE(nn.Module):
    """
    VQ-VAE for tokenizing 64x64 16-color game frames into 64 discrete codes.

    Usage:
        model = FrameVQVAE()
        # Training: returns logits, indices, losses
        logits, indices, recon_loss, vq_loss = model(frames)
        # Encoding only: returns 64 discrete codes per frame
        indices = model.encode(frames)
        # Decoding: returns reconstructed frame
        frames = model.decode(indices)
    """

    def __init__(self, config: VQVAEConfig | None = None):
        super().__init__()
        config = config or VQVAEConfig()
        self.config = config

        # Color embedding: int(0-15) → vector
        self.color_embed = nn.Embedding(config.num_colors, config.color_embed_dim)

        # Encoder/decoder
        self.encoder = Encoder(config.color_embed_dim, config.hidden_dim)
        self.decoder = Decoder(config.hidden_dim, config.num_colors)

        # Vector quantizer
        self.vq = VectorQuantizer(
            num_codes=config.codebook_size,
            code_dim=config.code_dim,
            ema_decay=config.ema_decay,
            commitment_beta=config.commitment_beta,
            dead_code_threshold=config.dead_code_threshold,
        )

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames: [B, 64, 64] int tensor with values 0-15

        Returns:
            logits: [B, 16, 64, 64] reconstruction logits
            indices: [B, 8, 8] codebook indices
            recon_loss: cross-entropy reconstruction loss
            vq_loss: commitment loss
        """
        # Embed colors: [B, 64, 64] → [B, 64, 64, 32] → [B, 32, 64, 64]
        x = self.color_embed(frames)
        x = x.permute(0, 3, 1, 2)

        # Encode
        z = self.encoder(x)  # [B, 128, 8, 8]

        # Quantize
        z_q, indices, vq_loss = self.vq(z)  # [B, 128, 8, 8], [B, 8, 8], scalar

        # Decode
        logits = self.decoder(z_q)  # [B, 16, 64, 64]

        # Reconstruction loss
        recon_loss = F.cross_entropy(logits, frames)

        return logits, indices, recon_loss, vq_loss

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to discrete codes. Returns [B, 8, 8] indices."""
        x = self.color_embed(frames).permute(0, 3, 1, 2)
        z = self.encoder(x)
        _, indices, _ = self.vq(z)
        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to frames. Returns [B, 64, 64] int tensor."""
        z_q = self.vq.embedding(indices)  # [B, 8, 8, 128]
        z_q = z_q.permute(0, 3, 1, 2)  # [B, 128, 8, 8]
        logits = self.decoder(z_q)  # [B, 16, 64, 64]
        return logits.argmax(dim=1)  # [B, 64, 64]

    def get_codebook_utilization(self) -> float:
        """Fraction of codebook entries actively used."""
        return self.vq.get_utilization()
