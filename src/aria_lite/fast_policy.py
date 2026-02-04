"""
ARIA-Lite Fast Policy

Fast policy network for habitual/reflexive actions. This is the "System 1"
component that provides quick responses based on pattern matching.

Architecture:
    Input: state [256]
    → Linear(256, 256) + GELU
    → Linear(256, 256) + GELU (x2)
    → Action Head: Linear(256, 8) → softmax
    → Confidence Head: Linear(256, 1) → sigmoid
    → Coordinate Heads: Factorized (x, y) prediction

Target: ~0.4M parameters
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARIALiteConfig, FastPolicyConfig


@dataclass
class FastPolicyOutput:
    """Output from fast policy."""

    action_logits: torch.Tensor  # [B, num_actions] raw logits
    action_probs: torch.Tensor  # [B, num_actions] softmax probabilities
    confidence: torch.Tensor  # [B] confidence score [0, 1]
    x_logits: torch.Tensor  # [B, grid_size] x-coordinate logits
    y_logits: torch.Tensor  # [B, grid_size] y-coordinate logits


class FastPolicy(nn.Module):
    """
    Fast policy network for habitual actions.

    A lightweight MLP that produces quick action decisions based on
    the current state. Includes confidence estimation to signal when
    the slow policy should take over.
    """

    def __init__(self, config: Optional[FastPolicyConfig] = None):
        super().__init__()

        if config is None:
            config = FastPolicyConfig()

        self.config = config
        self.num_actions = config.num_actions
        self.grid_size = config.grid_size

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(config.num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                )
            )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_actions),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Factorized coordinate heads
        coord_hidden = config.hidden_dim // 2

        # X coordinate head
        self.x_head = nn.Sequential(
            nn.Linear(config.hidden_dim, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, config.grid_size),
        )

        # Y coordinate head (conditioned on x)
        self.y_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.grid_size, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, config.grid_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        temperature: float = 1.0,
    ) -> FastPolicyOutput:
        """
        Compute action probabilities and confidence.

        Args:
            state: [B, state_dim] encoded state
            temperature: temperature for softmax

        Returns:
            FastPolicyOutput with action probabilities and confidence
        """
        # Encode state
        x = self.state_encoder(state)

        # Hidden layers with residual
        for layer in self.hidden_layers:
            x = layer(x) + x

        # Action head
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits / temperature, dim=-1)

        # Confidence head
        confidence = torch.sigmoid(self.confidence_head(x)).squeeze(-1)

        # Coordinate heads (factorized)
        x_logits = self.x_head(x)

        # Y conditioned on x (use soft attention over x)
        x_soft = F.softmax(x_logits, dim=-1)
        y_input = torch.cat([x, x_soft], dim=-1)
        y_logits = self.y_head(y_input)

        return FastPolicyOutput(
            action_logits=action_logits,
            action_probs=action_probs,
            confidence=confidence,
            x_logits=x_logits,
            y_logits=y_logits,
        )

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, FastPolicyOutput]:
        """
        Sample action from policy.

        Args:
            state: [B, state_dim] encoded state
            deterministic: if True, return argmax action
            temperature: temperature for sampling

        Returns:
            action: [B] action indices
            output: FastPolicyOutput with full outputs
        """
        output = self.forward(state, temperature=temperature)

        if deterministic:
            action = output.action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(output.action_probs, num_samples=1).squeeze(-1)

        return action, output

    def get_coordinates(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get (x, y) coordinates for grid operations.

        Args:
            state: [B, state_dim] encoded state
            deterministic: if True, return argmax coordinates
            temperature: temperature for sampling

        Returns:
            x: [B] x-coordinates
            y: [B] y-coordinates
        """
        output = self.forward(state, temperature=temperature)

        x_probs = F.softmax(output.x_logits / temperature, dim=-1)
        y_probs = F.softmax(output.y_logits / temperature, dim=-1)

        if deterministic:
            x = x_probs.argmax(dim=-1)
            y = y_probs.argmax(dim=-1)
        else:
            x = torch.multinomial(x_probs, num_samples=1).squeeze(-1)
            y = torch.multinomial(y_probs, num_samples=1).squeeze(-1)

        return x, y

    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of action.

        Args:
            state: [B, state_dim]
            action: [B] action indices

        Returns:
            log_prob: [B] log probabilities
        """
        output = self.forward(state)
        log_probs = F.log_softmax(output.action_logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution.

        Args:
            state: [B, state_dim]

        Returns:
            entropy: [B] entropy values
        """
        output = self.forward(state)
        log_probs = F.log_softmax(output.action_logits, dim=-1)
        return -(output.action_probs * log_probs).sum(dim=-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_fast_policy(config: Optional[ARIALiteConfig] = None) -> FastPolicy:
    """Factory function to create fast policy from full config."""
    if config is None:
        config = ARIALiteConfig()
    return FastPolicy(config.fast_policy)
