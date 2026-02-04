"""
ARIA-Lite World Model

3-head ensemble world model for next-state prediction with uncertainty estimation.
Each head predicts (next_state, reward, done) given (state, action).

Architecture per head:
    Input: state [256] + action_onehot [8] = [264]
    → Linear(264, 1024) + LayerNorm + GELU
    → Linear(1024, 1024) + LayerNorm + GELU (x2)
    → State Predictor: Linear(1024, 256) + residual
    → Reward Predictor: Linear(1024, 1)
    → Done Predictor: Linear(1024, 1) + Sigmoid

Target: ~8M parameters per head, ~24M total (3 heads)
         (Reduced from 15M due to config tuning)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARIALiteConfig, WorldModelConfig


@dataclass
class WorldModelOutput:
    """Output from world model prediction."""

    next_state: torch.Tensor  # [B, state_dim]
    reward: torch.Tensor  # [B, 1]
    done: torch.Tensor  # [B, 1] probability
    uncertainty: torch.Tensor  # [B] epistemic uncertainty


class WorldModelHead(nn.Module):
    """Single world model head for next-state prediction."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        input_dim = config.state_dim + config.action_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
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

        # Output heads
        self.state_predictor = nn.Linear(config.hidden_dim, config.state_dim)
        self.reward_predictor = nn.Linear(config.hidden_dim, 1)
        self.done_predictor = nn.Linear(config.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale down state predictor for residual connection
        nn.init.xavier_uniform_(self.state_predictor.weight, gain=0.1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state, reward, and done.

        Args:
            state: [B, state_dim] current state
            action: [B, action_dim] action (one-hot or continuous)

        Returns:
            next_state: [B, state_dim]
            reward: [B, 1]
            done: [B, 1] probability
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Forward through network
        x = self.input_proj(x)
        for layer in self.hidden_layers:
            x = layer(x) + x  # Residual connection

        # Predict outputs
        state_delta = self.state_predictor(x)
        next_state = state + state_delta  # Residual prediction

        reward = self.reward_predictor(x)
        done = torch.sigmoid(self.done_predictor(x))

        return next_state, reward, done

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleWorldModel(nn.Module):
    """
    Ensemble world model with uncertainty estimation.

    Uses multiple heads to estimate epistemic uncertainty through
    ensemble disagreement.
    """

    def __init__(self, config: Optional[WorldModelConfig] = None):
        super().__init__()

        if config is None:
            config = WorldModelConfig()

        self.config = config
        self.num_ensemble = config.num_ensemble

        # Create ensemble heads
        self.heads = nn.ModuleList(
            [WorldModelHead(config) for _ in range(config.num_ensemble)]
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_all_predictions: bool = False,
    ) -> WorldModelOutput:
        """
        Predict next state with uncertainty estimation.

        Args:
            state: [B, state_dim] current state
            action: [B, action_dim] action (one-hot)
            return_all_predictions: if True, return predictions from all heads

        Returns:
            WorldModelOutput with mean predictions and uncertainty
        """
        # Get predictions from all heads
        all_states = []
        all_rewards = []
        all_dones = []

        for head in self.heads:
            next_state, reward, done = head(state, action)
            all_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(done)

        # Stack predictions: [num_ensemble, B, dim]
        stacked_states = torch.stack(all_states, dim=0)
        stacked_rewards = torch.stack(all_rewards, dim=0)
        stacked_dones = torch.stack(all_dones, dim=0)

        # Compute mean predictions
        mean_state = stacked_states.mean(dim=0)
        mean_reward = stacked_rewards.mean(dim=0)
        mean_done = stacked_dones.mean(dim=0)

        # Compute epistemic uncertainty (variance across ensemble)
        state_variance = stacked_states.var(dim=0)  # [B, state_dim]
        uncertainty = state_variance.mean(dim=-1)  # [B]

        return WorldModelOutput(
            next_state=mean_state,
            reward=mean_reward,
            done=mean_done,
            uncertainty=uncertainty,
        )

    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict a trajectory of states given a sequence of actions.

        Args:
            initial_state: [B, state_dim] starting state
            actions: [B, T, action_dim] sequence of actions

        Returns:
            states: [B, T+1, state_dim] predicted states
            rewards: [B, T, 1] predicted rewards
            dones: [B, T, 1] predicted done probabilities
            uncertainties: [B, T] accumulated uncertainty
        """
        _, T, _ = actions.shape

        states = [initial_state]
        rewards = []
        dones = []
        uncertainties = []

        current_state = initial_state

        for t in range(T):
            action = actions[:, t]
            output = self.forward(current_state, action)

            states.append(output.next_state)
            rewards.append(output.reward)
            dones.append(output.done)
            uncertainties.append(output.uncertainty)

            current_state = output.next_state

        # Stack outputs
        states = torch.stack(states, dim=1)  # [B, T+1, state_dim]
        rewards = torch.stack(rewards, dim=1)  # [B, T, 1]
        dones = torch.stack(dones, dim=1)  # [B, T, 1]
        uncertainties = torch.stack(uncertainties, dim=1)  # [B, T]

        return states, rewards, dones, uncertainties

    def get_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get epistemic uncertainty for state-action pair.

        Args:
            state: [B, state_dim]
            action: [B, action_dim]

        Returns:
            uncertainty: [B] epistemic uncertainty
        """
        output = self.forward(state, action)
        return output.uncertainty

    def sample_head(self) -> WorldModelHead:
        """Sample a random head for Thompson sampling."""
        idx = torch.randint(0, self.num_ensemble, (1,)).item()
        return self.heads[idx]

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_world_model(config: Optional[ARIALiteConfig] = None) -> EnsembleWorldModel:
    """Factory function to create world model from full config."""
    if config is None:
        config = ARIALiteConfig()
    return EnsembleWorldModel(config.world_model)


def action_to_onehot(action: torch.Tensor, num_actions: int = 8) -> torch.Tensor:
    """
    Convert action indices to one-hot encoding.

    Args:
        action: [B] or [B, T] action indices
        num_actions: number of possible actions

    Returns:
        [B, num_actions] or [B, T, num_actions] one-hot encoding
    """
    return F.one_hot(action.long(), num_actions).float()
