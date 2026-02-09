"""
Policy heads for action selection, separate from the world model.

Two heads that read from the backbone's hidden states:
1. ActionTypeHead: "what kind of action?" (8-way classification)
2. ActionLocationHead: "where?" (65-way spatial attention: 64 VQ cells + NULL)

The policy operates on MASKED context where all action tokens are replaced
with a learned MASK token. This prevents mode collapse — the policy can only
learn from visual consequences of actions, not from copying action history.

Total parameters: ~500K (trains in minutes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PolicyConfig


class ActionTypeHead(nn.Module):
    """Predict action type from frame representation.

    Input: hidden state at last VQ token position (960-dim).
    Output: logits over 8 action types.
    """

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        config = config or PolicyConfig()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.type_head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.type_head_hidden, config.num_action_types),
        )

    def forward(self, h_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_frame: [B, hidden_size] hidden state at frame boundary

        Returns:
            logits: [B, num_action_types]
        """
        return self.net(h_frame)


class ActionLocationHead(nn.Module):
    """Predict action location via spatial attention.

    Query: learned projection of frame summary hidden state.
    Keys: projection of each VQ-cell hidden state + 1 learnable NULL key.
    Output: softmax over 65 positions (64 VQ cells + NULL).
    """

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        config = config or PolicyConfig()
        self.dim = config.loc_head_dim

        # Query from frame summary
        self.query_proj = nn.Linear(config.hidden_size, config.loc_head_dim)

        # Keys from VQ cell hidden states
        self.key_proj = nn.Linear(config.hidden_size, config.loc_head_dim)

        # Learnable NULL key (for non-spatial actions)
        self.null_key = nn.Parameter(torch.randn(config.loc_head_dim) * 0.02)

    def forward(
        self, h_frame: torch.Tensor, h_vq_cells: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_frame: [B, hidden_size] frame summary hidden state
            h_vq_cells: [B, 64, hidden_size] hidden states at VQ cell positions

        Returns:
            logits: [B, 65] attention scores over 64 cells + NULL
        """
        B = h_frame.shape[0]

        # Query: [B, dim]
        query = self.query_proj(h_frame)

        # Keys: [B, 64, dim]
        cell_keys = self.key_proj(h_vq_cells)

        # Append NULL key: [B, 65, dim]
        null_key = self.null_key.unsqueeze(0).expand(B, -1).unsqueeze(1)  # [B, 1, dim]
        keys = torch.cat([cell_keys, null_key], dim=1)  # [B, 65, dim]

        # Scaled dot-product attention scores
        scale = self.dim ** 0.5
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / scale  # [B, 65]

        return logits


class PolicyHeads(nn.Module):
    """Combined policy heads with MASK token embedding.

    Wraps ActionTypeHead + ActionLocationHead and provides
    the mask_actions utility for creating policy input.
    """

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        config = config or PolicyConfig()
        self.config = config

        self.type_head = ActionTypeHead(config)
        self.location_head = ActionLocationHead(config)

        # Learned MASK token embedding (same dimension as backbone embeddings)
        self.mask_embedding = nn.Parameter(
            torch.randn(config.hidden_size) * 0.02
        )

    def forward(
        self,
        h_frame: torch.Tensor,
        h_vq_cells: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_frame: [B, hidden_size] hidden state at frame boundary
            h_vq_cells: [B, 64, hidden_size] hidden states at VQ positions

        Returns:
            type_logits: [B, num_action_types]
            loc_logits: [B, num_action_locs]
        """
        type_logits = self.type_head(h_frame)
        loc_logits = self.location_head(h_frame, h_vq_cells)
        return type_logits, loc_logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
