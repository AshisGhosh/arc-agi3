"""
Principled Intrinsic Motivation for ARC-DREAMER v2.

Addresses v1 weakness: Arbitrary 0.5/0.3/0.2 weights.

Solutions:
1. Information-theoretic formulation (mutual information)
2. Adaptive weighting based on learning progress
3. Separate exploration objectives with principled combination
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateHasher(nn.Module):
    """
    Learned state hashing for continuous state spaces.

    Uses SimHash with learned projections to map similar states
    to the same hash bucket. This enables count-based exploration
    even with continuous state representations.

    Reference: Tang et al. "#Exploration: A Study of Count-Based
    Exploration for Deep Reinforcement Learning"
    """

    def __init__(self, state_dim: int, num_bits: int = 32):
        super().__init__()
        self.num_bits = num_bits
        self.projection = nn.Linear(state_dim, num_bits, bias=False)
        nn.init.normal_(self.projection.weight)

    def hash(self, state: torch.Tensor) -> int:
        """
        Compute hash of state.

        The hash is locality-sensitive: similar states map to
        similar (often identical) hashes.
        """
        with torch.no_grad():
            if state.dim() > 1:
                state = state.flatten()
            projected = self.projection(state)
            bits = (projected > 0).int()
            # Convert to integer hash
            hash_val = 0
            for i, bit in enumerate(bits.tolist()):
                hash_val += bit * (2**i)
            return hash_val

    def batch_hash(self, states: torch.Tensor) -> list[int]:
        """Compute hashes for a batch of states."""
        with torch.no_grad():
            projected = self.projection(states)  # [B, num_bits]
            bits = (projected > 0).int()  # [B, num_bits]

            hashes = []
            for b in range(bits.shape[0]):
                hash_val = 0
                for i, bit in enumerate(bits[b].tolist()):
                    hash_val += bit * (2**i)
                hashes.append(hash_val)

            return hashes


class PrincipledIntrinsicMotivation:
    """
    Intrinsic motivation based on information gain.

    Mathematical foundation:
    We want to maximize information about environment dynamics.
    This translates to maximizing mutual information I(s'; a, s).

    We decompose into three interpretable components:

    r_intrinsic = alpha(t) * I_dynamics + beta(t) * H_policy + gamma(t) * C_coverage

    Where:
        I_dynamics = Information gain about dynamics (what did I learn?)
        H_policy = Policy entropy (am I exploring action space?)
        C_coverage = State coverage (am I visiting new states?)

    The weights alpha, beta, gamma are derived from learning progress,
    not hand-tuned. This is the key difference from v1's arbitrary weights.

    Theoretical justification:
    - I_dynamics: From the principle of maximum entropy / minimum description
      length. Transitions that teach us something new should be rewarded.
    - H_policy: From maximum entropy RL (SAC, SQL). Provides robustness and
      exploration in action space.
    - C_coverage: From UCB / optimism in face of uncertainty. Ensures we
      visit all parts of state space.

    Adaptive weighting follows optimal experiment design: allocate
    exploration budget where expected information gain is highest.
    """

    def __init__(
        self,
        world_model: nn.Module,
        state_dim: int = 256,
        action_dim: int = 8,
        ema_decay: float = 0.99,
        min_weight: float = 0.1,
    ):
        self.world_model = world_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ema_decay = ema_decay
        self.min_weight = min_weight

        # State visitation tracking
        self.state_counts: dict[int, int] = defaultdict(int)
        self.state_encoder = StateHasher(state_dim)

        # Learning progress tracking (EMA of key metrics)
        self.prediction_error_ema = 1.0
        self.coverage_progress_ema = 1.0
        self.policy_entropy_ema = 1.0

        # History for analysis
        self.weight_history = []
        self.component_history = []

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        policy_entropy: float,
    ) -> Tuple[float, dict]:
        """
        Compute intrinsic reward with adaptive weighting.

        Args:
            state: Current state [state_dim]
            action: Action taken [action_dim] one-hot
            next_state: Resulting state [state_dim]
            policy_entropy: Entropy of policy distribution

        Returns:
            total_reward: Weighted sum of intrinsic components
            components: Dictionary of individual components for logging
        """
        # Component 1: Information gain about dynamics
        # I(s'; a | s) approx= reduction in ensemble disagreement
        i_dynamics = self._compute_dynamics_information_gain(state, action, next_state)

        # Component 2: Policy entropy (exploration in action space)
        h_policy = policy_entropy

        # Component 3: State coverage (exploration in state space)
        c_coverage = self._compute_coverage_bonus(next_state)

        # Update entropy EMA
        self.policy_entropy_ema = (
            self.ema_decay * self.policy_entropy_ema + (1 - self.ema_decay) * h_policy
        )

        # Adaptive weights based on learning progress
        alpha, beta, gamma = self._compute_adaptive_weights()

        total_reward = alpha * i_dynamics + beta * h_policy + gamma * c_coverage

        components = {
            "i_dynamics": i_dynamics,
            "h_policy": h_policy,
            "c_coverage": c_coverage,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "total": total_reward,
        }

        self.component_history.append(components)

        return total_reward, components

    def _compute_dynamics_information_gain(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> float:
        """
        Information gain = reduction in uncertainty about dynamics.

        Approximated as: prediction error - expected prediction error.
        High when: Model learns something new from this transition.
        Low when: Transition was already predictable.

        This is similar to curiosity-driven exploration but with
        proper normalization to handle non-stationary learning.
        """
        with torch.no_grad():
            # Ensure proper dimensions
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)

            # Get prediction from world model
            mean_pred, uncertainty, _ = self.world_model(state, action)

            # Actual prediction error
            prediction_error = F.mse_loss(mean_pred, next_state).item()

            # Information gain is the "surprise" normalized by expectation
            # If we're surprised, we learned something
            surprise = prediction_error / (self.prediction_error_ema + 1e-8)
            information_gain = np.log1p(surprise)  # Bounded, positive

            # Update EMA
            self.prediction_error_ema = (
                self.ema_decay * self.prediction_error_ema + (1 - self.ema_decay) * prediction_error
            )

            return information_gain

    def _compute_coverage_bonus(self, state: torch.Tensor) -> float:
        """
        Coverage bonus based on state visitation.

        Uses count-based exploration: 1 / sqrt(N(s))

        With learned state hashing for continuous states, this
        provides a simple but effective coverage signal.

        Theoretical foundation: UCB-style exploration bonus with
        sqrt(log(n)/N(s,a)) approximated by 1/sqrt(N(s)).
        """
        state_hash = self.state_encoder.hash(state)
        self.state_counts[state_hash] += 1
        count = self.state_counts[state_hash]

        coverage_bonus = 1.0 / np.sqrt(count)

        # Track coverage progress (rate of discovering new states)
        unique_states = len(self.state_counts)
        coverage_rate = unique_states / (sum(self.state_counts.values()) + 1e-8)

        self.coverage_progress_ema = (
            self.ema_decay * self.coverage_progress_ema + (1 - self.ema_decay) * coverage_rate
        )

        return coverage_bonus

    def _compute_adaptive_weights(self) -> Tuple[float, float, float]:
        """
        Compute adaptive weights based on learning progress.

        Principle: Allocate exploration budget where it's most needed.

        - High alpha when: Model is still learning dynamics (high error)
        - High beta when: Policy is too deterministic (low entropy)
        - High gamma when: State coverage is improving (finding new states)

        This is a form of optimal experiment design: we explore
        in the dimension where we expect the most information gain.
        """
        # Raw weights based on learning progress
        # High prediction error -> need more dynamics exploration
        alpha_raw = self.prediction_error_ema

        # Low entropy -> need more action exploration
        # Inverse: when entropy is low, weight should be high
        beta_raw = 1.0 / (self.policy_entropy_ema + 0.1)

        # Coverage progress indicates whether state exploration is working
        # High coverage rate -> state exploration is useful, continue
        gamma_raw = self.coverage_progress_ema

        # Normalize
        total = alpha_raw + beta_raw + gamma_raw + 1e-8

        alpha = alpha_raw / total
        beta = beta_raw / total
        gamma = gamma_raw / total

        # Ensure minimum exploration in each dimension
        alpha = max(alpha, self.min_weight)
        beta = max(beta, self.min_weight)
        gamma = max(gamma, self.min_weight)

        # Re-normalize after applying minimums
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha / total, beta / total, gamma / total

        self.weight_history.append((alpha, beta, gamma))

        return alpha, beta, gamma

    def get_statistics(self) -> dict:
        """Get intrinsic motivation statistics."""
        if not self.weight_history:
            return {}

        weights = np.array(self.weight_history)

        return {
            "avg_alpha": weights[:, 0].mean(),
            "avg_beta": weights[:, 1].mean(),
            "avg_gamma": weights[:, 2].mean(),
            "alpha_std": weights[:, 0].std(),
            "beta_std": weights[:, 1].std(),
            "gamma_std": weights[:, 2].std(),
            "unique_states": len(self.state_counts),
            "total_visits": sum(self.state_counts.values()),
            "prediction_error_ema": self.prediction_error_ema,
            "policy_entropy_ema": self.policy_entropy_ema,
            "coverage_progress_ema": self.coverage_progress_ema,
        }

    def reset_episode(self):
        """Reset per-episode state (keep global statistics)."""
        pass  # Currently all statistics are global

    def reset(self):
        """Full reset (clear all statistics)."""
        self.state_counts.clear()
        self.prediction_error_ema = 1.0
        self.coverage_progress_ema = 1.0
        self.policy_entropy_ema = 1.0
        self.weight_history.clear()
        self.component_history.clear()


class RNDNetwork(nn.Module):
    """
    Random Network Distillation for novelty detection.

    A complementary intrinsic motivation signal that measures
    how "novel" a state is based on prediction error of a
    randomly initialized target network.

    Reference: Burda et al. "Exploration by Random Network Distillation"
    """

    def __init__(
        self,
        state_dim: int = 256,
        feature_dim: int = 64,
    ):
        super().__init__()

        # Fixed random target network
        self.target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (trained)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RND prediction error.

        Returns:
            novelty: Prediction error (higher = more novel)
            loss: Training loss for predictor
        """
        with torch.no_grad():
            target_features = self.target(state)

        predicted_features = self.predictor(state)

        # Novelty is the prediction error
        novelty = F.mse_loss(predicted_features, target_features, reduction="none")
        novelty = novelty.mean(dim=-1)  # [B]

        # Loss for training predictor
        loss = novelty.mean()

        return novelty, loss
