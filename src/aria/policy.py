"""
ARIA Fast Policy Network

Small, highly optimized neural policy for habitual actions.
Target: 100k+ FPS on GPU.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import ARIAConfig
from .types import FastPolicyOutput


class FastPolicy(nn.Module):
    """
    Habit policy: pattern-matched actions.

    Architecture:
    - Spatial compression (AdaptiveAvgPool)
    - MLP for action type
    - Factorized coordinate prediction: P(x,y|a) = P(x|a) * P(y|a,x)
    - Confidence estimation for arbiter

    ~2M parameters, ~4MB memory
    Inference: <0.1ms on GPU
    """

    def __init__(self, config: ARIAConfig):
        super().__init__()

        self.config = config
        self.grid_size = config.grid_size
        self.num_actions = config.num_actions
        hidden_dim = config.policy_hidden_dim

        # Feature compression (from perception tower)
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Downsample to 8x8
            nn.Flatten(),
            nn.Linear(config.hidden_dim * 64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Action type prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.num_actions),
        )

        # Coordinate prediction (factorized)
        self.coord_embed = nn.Embedding(config.num_actions, hidden_dim // 4)

        # P(x | state, action)
        self.x_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.grid_size),
        )

        # P(y | state, action, x)
        self.y_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + config.grid_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.grid_size),
        )

        # Confidence estimation (epistemic uncertainty)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Context conditioning (for test-time adaptation)
        self.context_proj = nn.Linear(config.hidden_dim, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_log_prob: bool = False,
    ) -> FastPolicyOutput:
        """
        Forward pass.

        Args:
            features: [B, hidden_dim, H, W] from perception tower
            context: [B, context_dim] optional context for adaptation
            deterministic: If True, use argmax instead of sampling
            return_log_prob: If True, compute and return log probabilities

        Returns:
            FastPolicyOutput with action, coordinates, and confidence
        """
        # Compress spatial features
        h = self.compress(features)  # [B, hidden]

        # Add context if provided (FiLM-like modulation)
        if context is not None:
            context_h = self.context_proj(context)
            h = h + context_h  # Simple additive; could use FiLM here

        # Predict action distribution
        action_logits = self.action_head(h)  # [B, num_actions]

        # Sample or argmax action
        action_dist = Categorical(logits=action_logits)
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = action_dist.sample()

        # Compute log probability if needed
        action_log_prob = None
        if return_log_prob:
            action_log_prob = action_dist.log_prob(action)

        # Predict coordinates conditioned on action
        action_embed = self.coord_embed(action)  # [B, hidden/4]
        h_coord = torch.cat([h, action_embed], dim=-1)

        # X coordinate
        x_logits = self.x_head(h_coord)  # [B, grid_size]
        x_dist = Categorical(logits=x_logits)
        if deterministic:
            x = x_logits.argmax(dim=-1)
        else:
            x = x_dist.sample()

        # Y coordinate (conditioned on x)
        x_onehot = F.one_hot(x, self.grid_size).float()  # [B, grid_size]
        h_y = torch.cat([h_coord, x_onehot], dim=-1)
        y_logits = self.y_head(h_y)  # [B, grid_size]
        y_dist = Categorical(logits=y_logits)
        if deterministic:
            y = y_logits.argmax(dim=-1)
        else:
            y = y_dist.sample()

        # Update log prob to include coordinates
        if return_log_prob:
            action_log_prob = action_log_prob + x_dist.log_prob(x) + y_dist.log_prob(y)

        # Confidence score (sigmoid to [0, 1])
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)

        return FastPolicyOutput(
            action=action,
            x=x,
            y=y,
            action_logits=action_logits,
            confidence=confidence,
            action_log_prob=action_log_prob,
        )

    def log_prob(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of given action and coordinates.
        Used for policy gradient training.
        """
        h = self.compress(features)

        if context is not None:
            context_h = self.context_proj(context)
            h = h + context_h

        # Action log prob
        action_logits = self.action_head(h)
        action_log_prob = F.log_softmax(action_logits, dim=-1)
        action_log_prob = action_log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        # X log prob
        action_embed = self.coord_embed(action)
        h_coord = torch.cat([h, action_embed], dim=-1)
        x_logits = self.x_head(h_coord)
        x_log_prob = F.log_softmax(x_logits, dim=-1)
        x_log_prob = x_log_prob.gather(-1, x.unsqueeze(-1)).squeeze(-1)

        # Y log prob
        x_onehot = F.one_hot(x, self.grid_size).float()
        h_y = torch.cat([h_coord, x_onehot], dim=-1)
        y_logits = self.y_head(h_y)
        y_log_prob = F.log_softmax(y_logits, dim=-1)
        y_log_prob = y_log_prob.gather(-1, y.unsqueeze(-1)).squeeze(-1)

        return action_log_prob + x_log_prob + y_log_prob

    def entropy(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute entropy of action distribution.
        Used for entropy bonus in PPO.
        """
        h = self.compress(features)

        if context is not None:
            context_h = self.context_proj(context)
            h = h + context_h

        action_logits = self.action_head(h)
        action_dist = Categorical(logits=action_logits)

        # Only compute action entropy (coordinate entropy is expensive)
        return action_dist.entropy()

    def get_value_features(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get compressed features for value network.
        Shares computation with policy.
        """
        h = self.compress(features)

        if context is not None:
            context_h = self.context_proj(context)
            h = h + context_h

        return h


class ValueNetwork(nn.Module):
    """
    Value network (critic) for actor-critic training.
    Shares feature compression with policy.
    """

    def __init__(self, config: ARIAConfig):
        super().__init__()

        hidden_dim = config.policy_hidden_dim

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, hidden_dim] compressed features from policy

        Returns:
            values: [B] predicted state values
        """
        return self.value_head(features).squeeze(-1)


class EnsembleFastPolicy(nn.Module):
    """
    Ensemble of fast policies for uncertainty estimation.
    Used for exploration via disagreement bonus.
    """

    def __init__(self, config: ARIAConfig, num_heads: int = 5):
        super().__init__()

        self.num_heads = num_heads
        self.policies = nn.ModuleList([FastPolicy(config) for _ in range(num_heads)])

    def forward(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> FastPolicyOutput:
        """
        Forward through ensemble, return mean prediction.
        """
        outputs = [p(features, context, deterministic=True) for p in self.policies]

        # Average action logits
        mean_logits = torch.stack([o.action_logits for o in outputs]).mean(dim=0)

        # Use first policy's prediction with averaged logits
        output = outputs[0]
        output.action_logits = mean_logits

        # Compute disagreement as uncertainty measure
        action_preds = torch.stack([o.action for o in outputs])  # [num_heads, B]
        disagreement = (action_preds != action_preds[0]).float().mean(dim=0)

        # Override confidence with agreement-based measure
        output.confidence = 1.0 - disagreement

        return output

    def disagreement(
        self,
        features: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ensemble disagreement for intrinsic motivation.
        """
        outputs = [p(features, context, deterministic=True) for p in self.policies]
        action_logits = torch.stack([o.action_logits for o in outputs])  # [num_heads, B, A]

        # Compute variance across ensemble
        variance = action_logits.var(dim=0).sum(dim=-1)  # [B]

        return variance
