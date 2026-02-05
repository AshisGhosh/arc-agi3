"""
ARIA-Lite Slow Policy

Transformer-based deliberative planner for complex reasoning. This is the
"System 2" component that engages when the fast policy is uncertain.

Architecture:
    Input: state [256] + belief [256] + goal [64] = 576
    → Linear(576, 384)
    → TransformerEncoder(6 layers, 6 heads, 384 dim, 1024 ff)
    → PolicyHead: Linear(384, 8) → softmax
    → ValueHead: Linear(384, 1)
    → UncertaintyHead: Linear(384, 1)

Target: ~8.5M parameters
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARIALiteConfig, SlowPolicyConfig


@dataclass
class SlowPolicyOutput:
    """Output from slow policy."""

    action_logits: torch.Tensor  # [B, num_actions]
    action_probs: torch.Tensor  # [B, num_actions]
    value: torch.Tensor  # [B] estimated value
    uncertainty: torch.Tensor  # [B] uncertainty estimate
    hidden_states: Optional[torch.Tensor] = None  # [B, seq_len, hidden_dim]


class SlowPolicy(nn.Module):
    """
    Slow policy network for deliberative planning.

    A transformer-based policy that takes more time to reason about
    complex situations. Used when the fast policy is uncertain.
    """

    def __init__(self, config: Optional[SlowPolicyConfig] = None):
        super().__init__()

        if config is None:
            config = SlowPolicyConfig()

        self.config = config
        self.num_actions = config.num_actions

        # Input projection
        input_dim = config.state_dim + config.belief_dim + config.goal_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        # Positional encoding for transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output heads
        self.policy_head = nn.Linear(config.hidden_dim, config.num_actions)
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

        # Goal encoder (simple projection)
        self.goal_encoder = nn.Linear(64, config.goal_dim)

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
        belief: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_hidden: bool = False,
    ) -> SlowPolicyOutput:
        """
        Compute action probabilities, value, and uncertainty.

        Args:
            state: [B, state_dim] encoded state
            belief: [B, belief_dim] belief state
            goal: [B, 64] goal embedding (optional)
            temperature: temperature for softmax
            return_hidden: whether to return hidden states

        Returns:
            SlowPolicyOutput with action probabilities, value, and uncertainty
        """
        B = state.shape[0]

        # Handle missing goal
        if goal is None:
            goal = torch.zeros(B, 64, device=state.device)

        # Ensure goal is correct size
        if goal.shape[-1] != self.config.goal_dim:
            goal = self.goal_encoder(goal)

        # Concatenate inputs
        x = torch.cat([state, belief, goal], dim=-1)

        # Project to hidden dim
        x = self.input_proj(x)

        # Add positional embedding (single token for now)
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        x = x + self.pos_embedding

        # Transformer encoding
        hidden = self.transformer(x)

        # Pool to single vector (use first/only token)
        pooled = hidden[:, 0]

        # Output heads
        action_logits = self.policy_head(pooled)
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        value = self.value_head(pooled).squeeze(-1)
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled)).squeeze(-1)

        return SlowPolicyOutput(
            action_logits=action_logits,
            action_probs=action_probs,
            value=value,
            uncertainty=uncertainty,
            hidden_states=hidden if return_hidden else None,
        )

    def plan(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        planning_steps: int = 1,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, SlowPolicyOutput]:
        """
        Plan action with optional multi-step lookahead.

        For now, this is equivalent to a single forward pass.
        Can be extended for tree search or MCTS.

        Args:
            state: [B, state_dim]
            belief: [B, belief_dim]
            goal: [B, 64] goal embedding
            planning_steps: number of planning steps (reserved for future)
            temperature: sampling temperature

        Returns:
            action: [B] selected actions
            output: SlowPolicyOutput with full outputs
        """
        output = self.forward(state, belief, goal, temperature=temperature)
        action = output.action_probs.argmax(dim=-1)
        return action, output

    def get_action(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, SlowPolicyOutput]:
        """
        Sample action from policy.

        Args:
            state: [B, state_dim]
            belief: [B, belief_dim]
            goal: [B, 64] goal embedding
            deterministic: if True, return argmax action
            temperature: temperature for sampling

        Returns:
            action: [B] action indices
            output: SlowPolicyOutput with full outputs
        """
        output = self.forward(state, belief, goal, temperature=temperature)

        if deterministic:
            action = output.action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(output.action_probs, num_samples=1).squeeze(-1)

        return action, output

    def evaluate_action(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO-style training.

        Args:
            state: [B, state_dim]
            belief: [B, belief_dim]
            action: [B] action indices
            goal: [B, 64] goal embedding

        Returns:
            log_prob: [B] log probabilities
            value: [B] value estimates
            entropy: [B] entropy values
        """
        output = self.forward(state, belief, goal)

        log_probs = F.log_softmax(output.action_logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        entropy = -(output.action_probs * log_probs).sum(dim=-1)

        return log_prob, output.value, entropy

    def log_prob(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of action.

        Args:
            state: [B, state_dim]
            belief: [B, belief_dim]
            action: [B] action indices
            goal: [B, 64] goal embedding

        Returns:
            log_prob: [B] log probabilities
        """
        output = self.forward(state, belief, goal)
        log_probs = F.log_softmax(output.action_logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

    def entropy(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy of action distribution.

        Args:
            state: [B, state_dim]
            belief: [B, belief_dim]
            goal: [B, 64] goal embedding

        Returns:
            entropy: [B] entropy values
        """
        output = self.forward(state, belief, goal)
        log_probs = F.log_softmax(output.action_logits, dim=-1)
        return -(output.action_probs * log_probs).sum(dim=-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_slow_policy(config: Optional[ARIALiteConfig] = None) -> SlowPolicy:
    """Factory function to create slow policy from full config."""
    if config is None:
        config = ARIALiteConfig()
    return SlowPolicy(config.slow_policy)
