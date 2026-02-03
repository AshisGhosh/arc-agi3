"""
Hidden State Inference (POMDP) for ARC-DREAMER v2.

Addresses v1 weakness: Can't detect latent variables.

Solutions:
1. Belief state tracking over unobserved variables
2. POMDP formulation with learned belief updates
3. Anomaly detection when observations don't match predictions
"""

from __future__ import annotations

from collections import deque
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenTransitionModel(nn.Module):
    """
    Models transitions in hidden state space.

    P(z_t | z_{t-1}, a_{t-1})
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 8,
        stochastic: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stochastic = stochastic

        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        if stochastic:
            self.mean = nn.Linear(128, hidden_dim)
            self.logvar = nn.Linear(128, hidden_dim)
        else:
            self.output = nn.Linear(128, hidden_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next hidden state.

        Args:
            hidden_state: [N, hidden_dim] current hidden states (particles)
            action: [N, action_dim] actions (broadcast if [1, action_dim])

        Returns:
            next_hidden: [N, hidden_dim] predicted next hidden states
        """
        if action.shape[0] == 1 and hidden_state.shape[0] > 1:
            action = action.expand(hidden_state.shape[0], -1)

        combined = torch.cat([hidden_state, action], dim=-1)
        features = self.transition(combined)

        if self.stochastic:
            mean = self.mean(features)
            logvar = self.logvar(features).clamp(-10, 2)
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(mean)
            return mean + std * noise
        else:
            return self.output(features)


class ObservationModel(nn.Module):
    """
    Models observation likelihood given hidden state.

    P(o_t | z_t)
    """

    def __init__(
        self,
        hidden_dim: int,
        observation_dim: int,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, observation_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict observation from hidden state."""
        return self.decoder(hidden_state)

    def log_prob(
        self,
        hidden_states: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of observation given hidden states.

        Args:
            hidden_states: [N, hidden_dim] hidden states (particles)
            observation: [obs_dim] single observation

        Returns:
            log_probs: [N] log probability for each particle
        """
        predicted_obs = self.decoder(hidden_states)  # [N, obs_dim]

        if observation.dim() == 1:
            observation = observation.unsqueeze(0).expand(predicted_obs.shape[0], -1)

        # Gaussian likelihood
        log_probs = -0.5 * ((predicted_obs - observation) ** 2).sum(dim=-1)
        return log_probs


class HiddenEncoder(nn.Module):
    """Encodes observations to initial hidden state estimates."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to hidden state."""
        return self.encoder(observation)


class BeliefStateTracker:
    """
    Maintains belief distribution over hidden states.

    Uses a particle filter approach with learned transition model.

    b_t = P(z_t | o_1:t, a_1:t-1)

    Where:
    - z_t is the hidden state
    - o_t is the observation (grid frame / latent state)
    - a_t is the action

    This addresses v1's assumption of full observability by
    explicitly tracking uncertainty over hidden state.

    Applications in ARC-AGI-3:
    - Keys unlock doors (key possession is hidden)
    - Counters track interactions (count is hidden)
    - Order matters (history is hidden)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_particles: int = 100,
        observation_dim: int = 256,
        action_dim: int = 8,
    ):
        self.hidden_dim = hidden_dim
        self.num_particles = num_particles
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Particle representation of belief
        self.particles = torch.randn(num_particles, hidden_dim)
        self.weights = torch.ones(num_particles) / num_particles

        # Learned models
        self.transition_model = HiddenTransitionModel(hidden_dim, action_dim)
        self.observation_model = ObservationModel(hidden_dim, observation_dim)
        self.hidden_encoder = HiddenEncoder(observation_dim, hidden_dim)

        # History for analysis
        self.uncertainty_history = []

    def update(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update belief after taking action and receiving observation.

        Particle Filter Algorithm:
        1. Predict: propagate particles through transition model
        2. Update: reweight based on observation likelihood
        3. Resample: if effective sample size too low

        Args:
            action: [action_dim] action taken (one-hot)
            observation: [obs_dim] observation received

        Returns:
            belief_state: [hidden_dim] weighted mean of particles (belief summary)
        """
        # Predict step: z_t ~ P(z_t | z_{t-1}, a_{t-1})
        self.particles = self.transition_model(
            self.particles,
            action.unsqueeze(0),  # Broadcast to all particles
        )

        # Update step: reweight by P(o_t | z_t)
        log_likelihoods = self.observation_model.log_prob(self.particles, observation)
        self.weights = F.softmax(torch.log(self.weights + 1e-8) + log_likelihoods, dim=0)

        # Resample if effective sample size is low
        ess = 1.0 / (self.weights**2).sum()
        if ess < self.num_particles / 2:
            self._resample()

        # Return belief summary (weighted mean)
        belief_state = (self.particles * self.weights.unsqueeze(-1)).sum(dim=0)

        # Track uncertainty
        uncertainty = self.get_uncertainty()
        self.uncertainty_history.append(uncertainty)

        return belief_state

    def _resample(self):
        """
        Resample particles according to weights.

        This prevents particle degeneracy where only a few
        particles have non-negligible weight.
        """
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def reset(self):
        """Reset belief to prior (uniform uncertainty)."""
        self.particles = torch.randn(self.num_particles, self.hidden_dim)
        self.weights = torch.ones(self.num_particles) / self.num_particles
        self.uncertainty_history.clear()

    def initialize_from_observation(self, observation: torch.Tensor):
        """Initialize belief from first observation."""
        # Encode observation to get initial hidden state estimate
        initial_hidden = self.hidden_encoder(observation)

        # Initialize particles around this estimate
        noise = torch.randn(self.num_particles, self.hidden_dim) * 0.1
        self.particles = initial_hidden.unsqueeze(0) + noise
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def get_uncertainty(self) -> float:
        """
        Return uncertainty in current belief.

        High uncertainty = hidden state is ambiguous.
        Low uncertainty = confident about hidden state.

        Uses weighted variance of particles as uncertainty measure.
        """
        mean = (self.particles * self.weights.unsqueeze(-1)).sum(dim=0)
        variance = (self.weights.unsqueeze(-1) * (self.particles - mean) ** 2).sum(dim=0).mean()
        return variance.item()

    def get_belief_summary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get belief mean and variance.

        Returns:
            mean: [hidden_dim] weighted mean
            variance: [hidden_dim] weighted variance
        """
        mean = (self.particles * self.weights.unsqueeze(-1)).sum(dim=0)
        variance = (self.weights.unsqueeze(-1) * (self.particles - mean) ** 2).sum(dim=0)
        return mean, variance

    def sample_hidden_states(self, num_samples: int = 10) -> torch.Tensor:
        """
        Sample hidden states from current belief.

        Useful for planning under uncertainty.
        """
        indices = torch.multinomial(self.weights, num_samples, replacement=True)
        return self.particles[indices]

    def train_models(
        self,
        trajectories: list,
        num_epochs: int = 10,
        lr: float = 1e-3,
    ):
        """
        Train transition and observation models from data.

        Uses reconstruction loss and prediction loss.
        """
        optimizer = torch.optim.Adam(
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.hidden_encoder.parameters()),
            lr=lr,
        )

        for epoch in range(num_epochs):
            total_loss = 0.0

            for traj in trajectories:
                for i in range(len(traj.states) - 1):
                    obs = traj.states[i]
                    action = F.one_hot(torch.tensor([traj.actions[i]]), self.action_dim).float()
                    next_obs = traj.states[i + 1]

                    # Encode current observation
                    hidden = self.hidden_encoder(obs)

                    # Predict next hidden state
                    next_hidden = self.transition_model(hidden.unsqueeze(0), action).squeeze(0)

                    # Reconstruct next observation
                    pred_next_obs = self.observation_model(next_hidden)

                    # Loss
                    loss = F.mse_loss(pred_next_obs, next_obs)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()


class AnomalyDetector:
    """
    Detects when observations don't match predictions.

    Anomaly indicates:
    1. World model is wrong (need more learning)
    2. Hidden state changed unexpectedly
    3. New mechanic discovered

    This triggers appropriate responses like increased
    exploration or belief reset.
    """

    def __init__(
        self,
        threshold_percentile: float = 95,
        window_size: int = 100,
    ):
        self.threshold_percentile = threshold_percentile
        self.window_size = window_size
        self.error_history: deque[float] = deque(maxlen=window_size)

        # Statistics
        self.anomalies_detected = 0
        self.anomaly_types: dict[str, int] = {
            "observation": 0,
            "belief": 0,
        }

    def check_anomaly(
        self,
        predicted_obs: torch.Tensor,
        actual_obs: torch.Tensor,
        predicted_belief: torch.Tensor | None = None,
        actual_belief: torch.Tensor | None = None,
    ) -> Tuple[bool, float, str | None]:
        """
        Check if current observation is anomalous.

        Args:
            predicted_obs: World model's prediction
            actual_obs: Actual observation
            predicted_belief: Expected belief state (optional)
            actual_belief: Actual belief state (optional)

        Returns:
            is_anomaly: Whether anomaly detected
            anomaly_score: How anomalous (higher = more)
            anomaly_type: 'observation' or 'belief' or None
        """
        # Observation prediction error
        obs_error = F.mse_loss(predicted_obs, actual_obs).item()
        self.error_history.append(obs_error)

        # Dynamic threshold based on history
        if len(self.error_history) >= 10:
            threshold = np.percentile(list(self.error_history), self.threshold_percentile)
        else:
            threshold = obs_error * 2  # Generous initially

        # Check observation anomaly
        if obs_error > threshold:
            self.anomalies_detected += 1
            self.anomaly_types["observation"] += 1
            return True, obs_error / (threshold + 1e-8), "observation"

        # Check belief divergence (if provided)
        if predicted_belief is not None and actual_belief is not None:
            belief_divergence = F.kl_div(
                F.log_softmax(predicted_belief, dim=-1),
                F.softmax(actual_belief, dim=-1),
                reduction="batchmean",
            ).item()

            if belief_divergence > 1.0:  # Significant belief shift
                self.anomalies_detected += 1
                self.anomaly_types["belief"] += 1
                return True, belief_divergence, "belief"

        return False, obs_error / (threshold + 1e-8), None

    def on_anomaly_detected(
        self,
        anomaly_type: str,
        policy: Any,
        belief_tracker: BeliefStateTracker | None = None,
    ):
        """
        React to detected anomaly.

        Actions:
        1. Trigger exploration to understand new mechanic
        2. Reset world model uncertainty estimates
        3. Store anomaly for later analysis
        """
        if anomaly_type == "observation":
            # World model needs updating - increase exploration
            if hasattr(policy, "increase_exploration_bonus"):
                policy.increase_exploration_bonus(factor=2.0, duration=10)

        elif anomaly_type == "belief":
            # Hidden state changed - reset belief and explore
            if belief_tracker is not None:
                belief_tracker.reset()
            if hasattr(policy, "increase_exploration_bonus"):
                policy.increase_exploration_bonus(factor=1.5, duration=5)

    def get_statistics(self) -> dict:
        """Get anomaly detection statistics."""
        return {
            "total_anomalies": self.anomalies_detected,
            "by_type": dict(self.anomaly_types),
            "avg_error": (
                sum(self.error_history) / len(self.error_history) if self.error_history else 0
            ),
            "current_threshold": (
                np.percentile(list(self.error_history), self.threshold_percentile)
                if len(self.error_history) >= 10
                else None
            ),
        }

    def reset(self):
        """Reset detector state."""
        self.error_history.clear()
        self.anomalies_detected = 0
        self.anomaly_types = {"observation": 0, "belief": 0}
