"""
Test-Time Training (TTT) for the understanding model.

During live gameplay, the CNN encoder is fine-tuned via LoRA on a self-supervised
next-frame prediction task. This adapts the pretrained model to the specific
game being played.

Configuration:
- LoRA rank 4 on last 2 conv layers of TransitionEncoder (~10K trainable params)
- SGD with momentum 0.9, lr=0.01
- Update every 10 transitions (~1ms amortized)
- Reset LoRA at each level start
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import UnderstandingModel


class LoRAConv2d(nn.Module):
    """LoRA adapter for Conv2d layers."""

    def __init__(self, conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.conv = conv
        self.rank = rank

        # LoRA: W = W0 + BA where B: [out, rank, 1, 1], A: [rank, in, k, k]
        out_c = conv.out_channels
        in_c = conv.in_channels
        k = conv.kernel_size[0]

        self.lora_A = nn.Parameter(torch.zeros(rank, in_c, k, k))
        self.lora_B = nn.Parameter(torch.zeros(out_c, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A)
        # B starts at zero so LoRA contribution is zero at initialization

        self.scaling = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convolution
        out = self.conv(x)

        # LoRA addition: conv with A then conv with B (1x1)
        lora_out = F.conv2d(x, self.lora_A, padding=self.conv.padding, stride=self.conv.stride)
        lora_out = F.conv2d(lora_out, self.lora_B)
        return out + self.scaling * lora_out

    def reset_lora(self) -> None:
        """Reset LoRA parameters to zero (clean start for new level)."""
        nn.init.kaiming_uniform_(self.lora_A)
        self.lora_B.data.zero_()


class TTTEngine:
    """Test-time training engine for online adaptation."""

    def __init__(
        self,
        model: UnderstandingModel,
        lora_rank: int = 4,
        lr: float = 0.01,
        momentum: float = 0.9,
        update_interval: int = 10,
        buffer_size: int = 200,
    ):
        self.model = model
        self.lora_rank = lora_rank
        self.update_interval = update_interval
        self.buffer_size = buffer_size

        # Apply LoRA to last 2 conv layers of encoder
        self.lora_layers: list[LoRAConv2d] = []
        self._apply_lora()

        # Collect LoRA parameters
        lora_params = []
        for lora in self.lora_layers:
            lora_params.extend([lora.lora_A, lora.lora_B])

        self.optimizer = torch.optim.SGD(lora_params, lr=lr, momentum=momentum)

        # Rolling buffer of recent transitions
        self.frame_buffer: list[torch.Tensor] = []
        self.action_buffer: list[int] = []
        self.next_frame_buffer: list[torch.Tensor] = []
        self.step_count = 0

        # Save initial LoRA state for reset
        self._initial_state = {
            name: param.data.clone()
            for name, param in self._lora_named_params()
        }

    def _apply_lora(self) -> None:
        """Replace last 2 conv layers with LoRA-wrapped versions."""
        encoder = self.model.encoder

        # Wrap conv4 and conv5 (last 2 layers)
        lora4 = LoRAConv2d(encoder.conv4, rank=self.lora_rank)
        lora5 = LoRAConv2d(encoder.conv5, rank=self.lora_rank)

        # Replace in encoder (keep original conv frozen inside LoRA)
        encoder.conv4 = lora4
        encoder.conv5 = lora5

        self.lora_layers = [lora4, lora5]

        # Freeze all encoder parameters except LoRA
        for param in encoder.parameters():
            param.requires_grad = False
        for lora in self.lora_layers:
            lora.lora_A.requires_grad = True
            lora.lora_B.requires_grad = True

    def _lora_named_params(self):
        """Yield (name, param) for all LoRA parameters."""
        for i, lora in enumerate(self.lora_layers):
            yield f"lora_{i}_A", lora.lora_A
            yield f"lora_{i}_B", lora.lora_B

    def observe(
        self,
        frame: torch.Tensor,
        action: int,
        next_frame: torch.Tensor,
    ) -> float | None:
        """Add a transition to the buffer and potentially run a TTT update.

        Args:
            frame: [64, 64] long tensor
            action: action ID (0-7)
            next_frame: [64, 64] long tensor

        Returns:
            TTT loss if an update was performed, None otherwise
        """
        self.frame_buffer.append(frame.cpu())
        self.action_buffer.append(action)
        self.next_frame_buffer.append(next_frame.cpu())
        self.step_count += 1

        # Trim buffer to max size
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.action_buffer.pop(0)
            self.next_frame_buffer.pop(0)

        # Run TTT update every N steps
        if self.step_count % self.update_interval == 0 and len(self.frame_buffer) >= 5:
            return self._update()

        return None

    def _update(self) -> float:
        """Run a single TTT gradient step on the buffer."""
        device = next(self.model.parameters()).device
        self.model.train()

        # Sample a mini-batch from buffer
        n = len(self.frame_buffer)
        batch_size = min(n, 32)
        indices = torch.randperm(n)[:batch_size]

        frames = torch.stack([self.frame_buffer[i] for i in indices]).to(device)
        actions = torch.tensor(
            [self.action_buffer[i] for i in indices], dtype=torch.long, device=device
        )
        next_frames = torch.stack(
            [self.next_frame_buffer[i] for i in indices]
        ).to(device)

        # Encode transitions
        global_emb, spatial = self.model.encoder(frames, actions, next_frames)

        # Predict next frames
        logits = self.model.frame_predictor(spatial)  # [B, 16, 64, 64]

        # Cross-entropy loss
        loss = F.cross_entropy(logits, next_frames.long())

        # Backward (only updates LoRA params)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def reset(self) -> None:
        """Reset LoRA parameters and buffer (call at level start)."""
        for lora in self.lora_layers:
            lora.reset_lora()

        # Reset optimizer state
        self.optimizer.state.clear()

        # Clear buffer
        self.frame_buffer.clear()
        self.action_buffer.clear()
        self.next_frame_buffer.clear()
        self.step_count = 0

    def trainable_param_count(self) -> int:
        """Count trainable LoRA parameters."""
        return sum(
            p.numel() for lora in self.lora_layers
            for p in [lora.lora_A, lora.lora_B]
        )
