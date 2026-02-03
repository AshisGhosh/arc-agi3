"""
ARC-DREAMER v2 Agent: Complete Integration.

This module integrates all ARC-DREAMER v2 components into a
complete agent that can play ARC-AGI-3 environments.

Target Score: 9/10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .belief_tracking import AnomalyDetector, BeliefStateTracker
from .goal_discovery import GoalDiscoveryModule
from .hierarchy import (
    GoalConditionedPrimitivePolicy,
    HierarchicalPolicy,
    OptionDiscovery,
    PrimitivePolicy,
)
from .intrinsic_motivation import PrincipledIntrinsicMotivation
from .planning import AdaptivePlanner, MCTSPlanner, ValueNetwork
from .symbolic_grounding import SymbolicGrounding
from .world_model import (
    BackwardModel,
    BidirectionalConsistency,
    EnsembleWorldModel,
    GroundingController,
)


@dataclass
class AgentConfig:
    """Configuration for ARC-DREAMER v2 agent."""

    # Dimensions
    state_dim: int = 256
    action_dim: int = 8
    goal_dim: int = 64
    hidden_dim: int = 512
    slot_dim: int = 64

    # World model
    num_ensemble: int = 5
    grounding_interval: int = 5

    # Hierarchy
    num_options: int = 16
    strategy_interval: int = 20
    subgoal_interval: int = 5

    # Planning
    mcts_simulations: int = 100
    max_planning_depth: int = 50

    # Belief tracking
    num_particles: int = 100
    hidden_belief_dim: int = 32

    # Symbolic
    num_slots: int = 16

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99


class StateEncoder(nn.Module):
    """Encodes grid observations to latent state."""

    def __init__(
        self,
        grid_channels: int = 10,
        state_dim: int = 256,
        grid_size: int = 30,
    ):
        super().__init__()
        self.state_dim = state_dim

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # MLP head
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode grid to latent state.

        Args:
            grid: [B, H, W] grid of color values 0-9

        Returns:
            state: [B, state_dim] latent state
        """
        # One-hot encode
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)

        grid_onehot = F.one_hot(grid.long(), 10).permute(0, 3, 1, 2).float()
        features = self.cnn(grid_onehot)
        state = self.mlp(features)
        return state


@dataclass
class Trajectory:
    """Collected trajectory data."""

    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[torch.Tensor] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        info: dict,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)

    def __len__(self):
        return len(self.states)


class ARCDreamerV2Agent:
    """
    Complete ARC-DREAMER v2 Agent.

    Integrates all components:
    1. Error-correcting world model (ensemble + consistency + grounding)
    2. Principled intrinsic motivation (information-theoretic)
    3. Defined hierarchical policy (options + subgoals)
    4. Goal discovery (contrastive learning)
    5. Belief state tracking (POMDP)
    6. Symbolic grounding (slot attention)
    7. Extended planning (MCTS with 50+ steps)

    Target Score: 9/10 on ARC-AGI-3
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_components()
        self._setup_training()

        # Episode state
        self.current_trajectory = Trajectory()
        self.episode_count = 0
        self.total_steps = 0

    def _build_components(self):
        """Build all agent components."""
        cfg = self.config

        # State encoder
        self.state_encoder = StateEncoder(
            grid_channels=10,
            state_dim=cfg.state_dim,
        ).to(self.device)

        # 1. Error-correcting world model
        self.world_model = EnsembleWorldModel(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            num_ensemble=cfg.num_ensemble,
        ).to(self.device)

        self.backward_model = BackwardModel(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        self.consistency_checker = BidirectionalConsistency(
            forward_model=self.world_model,
            backward_model=self.backward_model,
        )

        self.grounding_controller = GroundingController(
            base_grounding_interval=cfg.grounding_interval,
        )

        # 2. Principled intrinsic motivation
        self.intrinsic_motivation = PrincipledIntrinsicMotivation(
            world_model=self.world_model,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
        )

        # 3. Hierarchical policy
        self.option_discovery = OptionDiscovery(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            max_options=cfg.num_options,
        )

        self.primitive_policy = PrimitivePolicy(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        self.goal_conditioned_policy = GoalConditionedPrimitivePolicy(
            state_dim=cfg.state_dim,
            goal_dim=cfg.goal_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        self.hierarchical_policy = HierarchicalPolicy(
            option_discovery=self.option_discovery,
            primitive_policy=self.primitive_policy,
            goal_conditioned_policy=self.goal_conditioned_policy,
        )

        # 4. Goal discovery
        self.goal_discovery = GoalDiscoveryModule(
            state_dim=cfg.state_dim,
            goal_dim=cfg.goal_dim,
        )

        # 5. Belief state tracking
        self.belief_tracker = BeliefStateTracker(
            hidden_dim=cfg.hidden_belief_dim,
            num_particles=cfg.num_particles,
            observation_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
        )

        self.anomaly_detector = AnomalyDetector()

        # 6. Symbolic grounding
        self.symbolic_grounding = SymbolicGrounding(
            grid_channels=10,
            slot_dim=cfg.slot_dim,
            num_slots=cfg.num_slots,
        ).to(self.device)

        # 7. Extended planning
        self.value_network = ValueNetwork(
            state_dim=cfg.state_dim,
        ).to(self.device)

        self.mcts_planner = MCTSPlanner(
            world_model=self.world_model,
            policy=self.hierarchical_policy,
            value_function=self.value_network,
            grounding_controller=self.grounding_controller,
            num_simulations=cfg.mcts_simulations,
            max_depth=cfg.max_planning_depth,
            action_dim=cfg.action_dim,
        )

        self.adaptive_planner = AdaptivePlanner(
            mcts_planner=self.mcts_planner,
            policy=self.hierarchical_policy,
        )

    def _setup_training(self):
        """Setup optimizers for training."""
        cfg = self.config

        # Collect all parameters
        params = []
        params.extend(self.state_encoder.parameters())
        params.extend(self.world_model.parameters())
        params.extend(self.backward_model.parameters())
        params.extend(self.primitive_policy.parameters())
        params.extend(self.goal_conditioned_policy.parameters())
        params.extend(self.symbolic_grounding.parameters())
        params.extend(self.value_network.parameters())

        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    def act(
        self,
        observation: torch.Tensor,
        use_planning: bool = True,
    ) -> Tuple[int, dict]:
        """
        Select action given observation.

        Args:
            observation: [H, W] grid observation
            use_planning: Whether to use MCTS planning

        Returns:
            action: Action index
            info: Action selection info
        """
        # Ensure tensor is on device
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, device=self.device)
        else:
            observation = observation.to(self.device)

        # Encode state
        with torch.no_grad():
            state = self.state_encoder(observation)

            # Extract symbolic state
            slots, symbolic_state = self.symbolic_grounding(observation.unsqueeze(0))

        # Update belief tracker
        if self.total_steps > 0:
            prev_action = (
                self.current_trajectory.actions[-1] if self.current_trajectory.actions else 0
            )
            action_onehot = (
                F.one_hot(torch.tensor([prev_action], device=self.device), self.config.action_dim)
                .float()
                .squeeze(0)
            )
            self.belief_tracker.update(action_onehot, state.squeeze(0))
        else:
            self.belief_tracker.initialize_from_observation(state.squeeze(0))

        # Check for anomaly
        if self.total_steps > 0 and len(self.current_trajectory.states) > 0:
            prev_state = self.current_trajectory.states[-1]
            predicted_state, _, _ = self.world_model(
                prev_state.unsqueeze(0),
                F.one_hot(
                    torch.tensor([self.current_trajectory.actions[-1]]), self.config.action_dim
                ).float(),
            )

            is_anomaly, anomaly_score, anomaly_type = self.anomaly_detector.check_anomaly(
                predicted_state.squeeze(0),
                state.squeeze(0),
            )

            if is_anomaly:
                self.anomaly_detector.on_anomaly_detected(
                    anomaly_type,
                    self.hierarchical_policy,
                    self.belief_tracker,
                )

        # Select action
        if use_planning and self.total_steps > 10:
            # Use MCTS planning
            action, info = self.adaptive_planner.get_action(
                state.squeeze(0),
                symbolic_state,
            )
        else:
            # Use hierarchical policy directly
            action, info = self.hierarchical_policy.act(
                state.squeeze(0),
                symbolic_state,
                strategy_interval=self.config.strategy_interval,
                subgoal_interval=self.config.subgoal_interval,
            )

        # Store for intrinsic motivation
        self._last_state = state.squeeze(0)
        self._last_action = action

        return action, info

    def observe(
        self,
        observation: torch.Tensor,
        action: int,
        reward: float,
        next_observation: torch.Tensor,
        done: bool,
        info: dict | None = None,
    ):
        """
        Observe transition and update agent.

        Args:
            observation: Previous observation
            action: Action taken
            reward: Environment reward
            next_observation: Resulting observation
            done: Whether episode ended
            info: Additional info
        """
        info = info or {}

        # Encode states
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, device=self.device)
            if not isinstance(next_observation, torch.Tensor):
                next_observation = torch.tensor(next_observation, device=self.device)

            state = self.state_encoder(
                observation.unsqueeze(0) if observation.dim() == 2 else observation
            )
            next_state = self.state_encoder(
                next_observation.unsqueeze(0) if next_observation.dim() == 2 else next_observation
            )

        # Compute intrinsic reward
        action_onehot = F.one_hot(
            torch.tensor([action], device=self.device), self.config.action_dim
        ).float()

        # Get policy entropy for intrinsic motivation
        with torch.no_grad():
            policy_dist = self.primitive_policy(state)
            policy_entropy = policy_dist.entropy().mean().item()

        intrinsic_reward, intrinsic_info = self.intrinsic_motivation.compute_intrinsic_reward(
            state.squeeze(0),
            action_onehot.squeeze(0),
            next_state.squeeze(0),
            policy_entropy,
        )

        # Total reward
        total_reward = reward + intrinsic_reward

        # Store transition
        self.current_trajectory.add(
            state=state.squeeze(0).detach(),
            action=action,
            reward=total_reward,
            next_state=next_state.squeeze(0).detach(),
            done=done,
            info={**info, **intrinsic_info},
        )

        self.total_steps += 1

        if done:
            self._on_episode_end()

    def _on_episode_end(self):
        """Handle end of episode."""
        self.episode_count += 1

        # Learn from trajectory if long enough
        if len(self.current_trajectory) > 10:
            self._update_from_trajectory(self.current_trajectory)

        # Reset episode state
        self.current_trajectory = Trajectory()
        self.grounding_controller.reset()
        self.belief_tracker.reset()
        self.anomaly_detector.reset()
        self.adaptive_planner.reset()
        self.intrinsic_motivation.reset_episode()

    def _update_from_trajectory(self, trajectory: Trajectory):
        """Update agent from collected trajectory."""
        # Prepare batch data
        states = torch.stack(trajectory.states)
        actions = torch.tensor(trajectory.actions, device=self.device)
        rewards = torch.tensor(trajectory.rewards, device=self.device)
        next_states = torch.stack(trajectory.next_states)
        dones = torch.tensor(trajectory.dones, device=self.device, dtype=torch.float)

        # World model loss
        action_onehot = F.one_hot(actions, self.config.action_dim).float()
        pred_next, uncertainty, _ = self.world_model(states, action_onehot)
        world_model_loss = F.mse_loss(pred_next, next_states)

        # Consistency loss
        consistency_loss = self.consistency_checker.training_loss(
            states, action_onehot, next_states
        )

        # Policy loss (simple policy gradient)
        policy_dist = self.primitive_policy(states)
        log_probs = policy_dist.log_prob(actions)

        # Compute returns
        returns = self._compute_returns(rewards, dones)
        advantages = returns - self.value_network(states).detach()

        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_pred = self.value_network(states)
        value_loss = F.mse_loss(value_pred, returns)

        # Entropy bonus
        entropy_loss = -policy_dist.entropy().mean()

        # Symbolic grounding auxiliary losses
        _, symbolic = self.symbolic_grounding(
            torch.stack(
                [
                    s.view(-1)[:900].view(30, 30) if s.numel() >= 900 else torch.zeros(30, 30)
                    for s in trajectory.states[:10]
                ]
            )
        )
        aux_losses = self.symbolic_grounding.auxiliary_losses(
            self.symbolic_grounding(
                torch.stack(
                    [
                        s.view(-1)[:900].view(30, 30) if s.numel() >= 900 else torch.zeros(30, 30)
                        for s in trajectory.states[:10]
                    ]
                )
            )[0],
            symbolic,
        )
        aux_loss = sum(aux_losses.values()) if aux_losses else torch.tensor(0.0)

        # Total loss
        total_loss = (
            world_model_loss
            + 0.1 * consistency_loss
            + policy_loss
            + 0.5 * value_loss
            + 0.01 * entropy_loss
            + 0.1 * aux_loss
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], max_norm=1.0)
        self.optimizer.step()

    def _compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        return returns

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "world_model": self.world_model.state_dict(),
                "backward_model": self.backward_model.state_dict(),
                "primitive_policy": self.primitive_policy.state_dict(),
                "goal_conditioned_policy": self.goal_conditioned_policy.state_dict(),
                "symbolic_grounding": self.symbolic_grounding.state_dict(),
                "value_network": self.value_network.state_dict(),
                "config": self.config,
                "episode_count": self.episode_count,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.backward_model.load_state_dict(checkpoint["backward_model"])
        self.primitive_policy.load_state_dict(checkpoint["primitive_policy"])
        self.goal_conditioned_policy.load_state_dict(checkpoint["goal_conditioned_policy"])
        self.symbolic_grounding.load_state_dict(checkpoint["symbolic_grounding"])
        self.value_network.load_state_dict(checkpoint["value_network"])

        self.episode_count = checkpoint["episode_count"]
        self.total_steps = checkpoint["total_steps"]

    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "intrinsic_motivation": self.intrinsic_motivation.get_statistics(),
            "grounding": self.grounding_controller.get_statistics(),
            "anomaly_detection": self.anomaly_detector.get_statistics(),
            "planning": self.mcts_planner.get_statistics(),
        }
