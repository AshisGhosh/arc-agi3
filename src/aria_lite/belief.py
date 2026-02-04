"""
ARIA-Lite Belief State Tracker

RSSM-style belief state tracker with particle filtering for maintaining
belief over hidden states given observations and actions.

Architecture:
    - Transition Model: Predicts next belief from (belief, action)
    - Observation Model: Computes likelihood of observation given belief
    - Particle Filter: Maintains distribution over beliefs

Target: ~0.7M parameters (per config estimate)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARIALiteConfig, BeliefConfig


@dataclass
class BeliefOutput:
    """Output from belief update."""

    belief: torch.Tensor  # [B, hidden_dim] mean belief
    particles: torch.Tensor  # [B, num_particles, hidden_dim]
    weights: torch.Tensor  # [B, num_particles]
    effective_sample_size: torch.Tensor  # [B]


class TransitionModel(nn.Module):
    """Predicts next belief state from current belief and action."""

    def __init__(self, config: BeliefConfig):
        super().__init__()
        self.config = config

        input_dim = config.hidden_dim + config.action_dim

        # GRU-style gated update
        self.reset_gate = nn.Linear(input_dim, config.hidden_dim)
        self.update_gate = nn.Linear(input_dim, config.hidden_dim)
        self.candidate = nn.Linear(input_dim, config.hidden_dim)

        # Stochastic layer for sampling
        self.stochastic_mean = nn.Linear(config.hidden_dim, config.stochastic_dim)
        self.stochastic_logvar = nn.Linear(config.hidden_dim, config.stochastic_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        belief: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next belief.

        Args:
            belief: [B, hidden_dim] or [B, N, hidden_dim] for particles
            action: [B, action_dim] or [B, N, action_dim]
            sample: whether to sample from stochastic layer

        Returns:
            next_belief: [B, hidden_dim] or [B, N, hidden_dim]
            mean: [B, stochastic_dim]
            logvar: [B, stochastic_dim]
        """
        # Handle both single beliefs and particles
        if belief.dim() == 3:
            # Particles: [B, N, hidden_dim]
            B, N, _ = belief.shape
            belief = belief.view(B * N, -1)
            if action.dim() == 2:
                action = action.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            else:
                action = action.view(B * N, -1)
            reshape_back = True
        else:
            reshape_back = False

        # Concatenate inputs
        x = torch.cat([belief, action], dim=-1)

        # GRU-style update
        reset = torch.sigmoid(self.reset_gate(x))
        update = torch.sigmoid(self.update_gate(x))

        x_reset = torch.cat([belief * reset, action], dim=-1)
        candidate = torch.tanh(self.candidate(x_reset))

        next_deterministic = (1 - update) * belief + update * candidate

        # Stochastic component
        mean = self.stochastic_mean(next_deterministic)
        logvar = self.stochastic_logvar(next_deterministic)

        # Sample from stochastic layer (for VAE-style training)
        # Currently using deterministic path; stochastic reserved for future use
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            _ = mean + std * eps  # Stochastic sample (unused for now)

        # For simplicity, just use deterministic for hidden state
        next_belief = next_deterministic

        if reshape_back:
            next_belief = next_belief.view(B, N, -1)
            mean = mean.view(B, N, -1)
            logvar = logvar.view(B, N, -1)

        return next_belief, mean, logvar


class ObservationModel(nn.Module):
    """Computes likelihood of observation given belief."""

    def __init__(self, config: BeliefConfig):
        super().__init__()
        self.config = config

        input_dim = config.hidden_dim + config.observation_dim

        layers = [nn.Linear(input_dim, config.hidden_dim), nn.GELU()]
        for _ in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(config.hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        belief: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-likelihood of observation given belief.

        Args:
            belief: [B, hidden_dim] or [B, N, hidden_dim] for particles
            observation: [B, observation_dim]

        Returns:
            log_likelihood: [B] or [B, N]
        """
        if belief.dim() == 3:
            # Particles: expand observation
            B, N, _ = belief.shape
            observation = observation.unsqueeze(1).expand(-1, N, -1)

        x = torch.cat([belief, observation], dim=-1)
        log_likelihood = self.network(x).squeeze(-1)

        return log_likelihood


class BeliefStateTracker(nn.Module):
    """
    Belief state tracker using particle filtering.

    Maintains a distribution over hidden states using a set of particles
    that are updated based on transitions and observations.
    """

    def __init__(self, config: Optional[BeliefConfig] = None):
        super().__init__()

        if config is None:
            config = BeliefConfig()

        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_particles = config.num_particles

        # Models
        self.transition_model = TransitionModel(config)
        self.observation_model = ObservationModel(config)

        # Posterior network (for when we have observations)
        posterior_input = config.hidden_dim + config.observation_dim
        self.posterior = nn.Sequential(
            nn.Linear(posterior_input, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Decoder (belief -> observation prediction)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim + config.stochastic_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.observation_dim),
        )

    def init_belief(self, batch_size: int, device: torch.device) -> BeliefOutput:
        """
        Initialize belief state.

        Args:
            batch_size: number of environments
            device: torch device

        Returns:
            Initial belief output with zero-initialized particles
        """
        particles = torch.zeros(
            batch_size, self.num_particles, self.hidden_dim, device=device
        )
        weights = torch.ones(batch_size, self.num_particles, device=device)
        weights = weights / self.num_particles

        mean_belief = particles.mean(dim=1)
        ess = self._effective_sample_size(weights)

        return BeliefOutput(
            belief=mean_belief,
            particles=particles,
            weights=weights,
            effective_sample_size=ess,
        )

    def update(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        prev_output: BeliefOutput,
        resample_threshold: float = 0.5,
    ) -> BeliefOutput:
        """
        Update belief state given action and observation.

        Args:
            action: [B, action_dim] action taken
            observation: [B, observation_dim] new observation
            prev_output: previous belief output
            resample_threshold: ESS threshold for resampling (as fraction of particles)

        Returns:
            Updated belief output
        """
        particles = prev_output.particles
        weights = prev_output.weights

        # 1. Predict: propagate particles through transition model
        next_particles, _, _ = self.transition_model(particles, action, sample=True)

        # 2. Update: weight particles by observation likelihood
        log_likelihoods = self.observation_model(next_particles, observation)

        # Normalize weights in log space for stability
        log_weights = torch.log(weights + 1e-10) + log_likelihoods
        log_weights = log_weights - log_weights.max(dim=1, keepdim=True).values
        new_weights = F.softmax(log_weights, dim=1)

        # 3. Check effective sample size
        ess = self._effective_sample_size(new_weights)
        threshold = resample_threshold * self.num_particles

        # 4. Resample if needed
        should_resample = ess < threshold
        if should_resample.any():
            resampled_particles, resampled_weights = self._resample(
                next_particles, new_weights, should_resample
            )
            # Use resampled for those that need it
            next_particles = torch.where(
                should_resample.unsqueeze(-1).unsqueeze(-1),
                resampled_particles,
                next_particles,
            )
            new_weights = torch.where(
                should_resample.unsqueeze(-1),
                resampled_weights,
                new_weights,
            )

        # 5. Compute mean belief
        mean_belief = (next_particles * new_weights.unsqueeze(-1)).sum(dim=1)

        return BeliefOutput(
            belief=mean_belief,
            particles=next_particles,
            weights=new_weights,
            effective_sample_size=ess,
        )

    def _effective_sample_size(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute effective sample size from weights."""
        # ESS = 1 / sum(w^2) for normalized weights
        return 1.0 / (weights.pow(2).sum(dim=1) + 1e-10)

    def _resample(
        self,
        particles: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Systematic resampling of particles.

        Args:
            particles: [B, N, hidden_dim]
            weights: [B, N]
            mask: [B] which batches to resample

        Returns:
            Resampled particles and uniform weights
        """
        B, N, D = particles.shape
        device = particles.device

        # Systematic resampling
        cumsum = torch.cumsum(weights, dim=1)

        # Generate systematic samples
        u = torch.rand(B, 1, device=device) / N
        positions = u + torch.arange(N, device=device).float() / N

        # Find indices
        indices = torch.searchsorted(cumsum, positions.clamp(max=0.9999))
        indices = indices.clamp(0, N - 1)

        # Gather resampled particles
        resampled = torch.gather(
            particles, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )

        # Uniform weights after resampling
        uniform_weights = torch.ones_like(weights) / N

        return resampled, uniform_weights

    def get_belief_embedding(self, belief_output: BeliefOutput) -> torch.Tensor:
        """Get belief embedding for policy input."""
        return belief_output.belief

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_belief_tracker(config: Optional[ARIALiteConfig] = None) -> BeliefStateTracker:
    """Factory function to create belief tracker from full config."""
    if config is None:
        config = ARIALiteConfig()
    return BeliefStateTracker(config.belief)
