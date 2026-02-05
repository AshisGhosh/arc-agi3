"""
ARIA-Lite Agent

Main orchestration component that ties together all ARIA-Lite components
into a unified agent for ARC-AGI-3 tasks.

Flow:
    1. Encode observation → state
    2. Update belief tracker
    3. Get fast policy output (action + confidence)
    4. Arbiter decides: use fast or slow?
    5. If slow: get slow policy output with goal from LLM
    6. Execute selected action
    7. Update world model with transition

Target: Unified interface for training and evaluation
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .arbiter import ArbiterDecision, create_arbiter
from .belief import BeliefOutput, create_belief_tracker
from .config import ARIALiteConfig
from .encoder import create_encoder
from .fast_policy import FastPolicyOutput, create_fast_policy
from .llm import create_llm_interface
from .slow_policy import SlowPolicyOutput, create_slow_policy
from .world_model import WorldModelOutput, create_world_model


@dataclass
class AgentState:
    """Internal state of the agent."""

    belief: BeliefOutput
    last_action: Optional[torch.Tensor] = None
    last_state: Optional[torch.Tensor] = None
    step_count: int = 0
    episode_reward: float = 0.0


@dataclass
class AgentOutput:
    """Output from agent step."""

    action: torch.Tensor  # [B] or [B, action_dim] depending on action space
    action_probs: torch.Tensor  # [B, num_actions]
    system_used: str  # "fast" or "slow"
    fast_output: FastPolicyOutput
    slow_output: Optional[SlowPolicyOutput]
    arbiter_decision: ArbiterDecision
    state: torch.Tensor  # [B, state_dim] encoded state
    world_model_uncertainty: torch.Tensor  # [B]
    metadata: dict = field(default_factory=dict)


class ARIALiteAgent(nn.Module):
    """
    ARIA-Lite Agent for ARC-AGI-3.

    Orchestrates the dual-system architecture:
    - Fast policy for quick, habitual responses
    - Slow policy for deliberate planning
    - Arbiter for metacognitive switching
    """

    def __init__(self, config: Optional[ARIALiteConfig] = None):
        super().__init__()

        if config is None:
            config = ARIALiteConfig()

        self.config = config

        # Core components (nn.Module for parameter tracking)
        self.encoder = create_encoder(config)
        self.world_model = create_world_model(config)
        self.belief_tracker = create_belief_tracker(config)
        self.fast_policy = create_fast_policy(config)
        self.slow_policy = create_slow_policy(config)
        self.arbiter = create_arbiter(config)

        # LLM interface (not nn.Module)
        self.llm = create_llm_interface(config)

        # State tracking
        self._state: Optional[AgentState] = None

    def reset(self, batch_size: int = 1) -> AgentState:
        """
        Reset agent state for new episode(s).

        Args:
            batch_size: number of parallel environments

        Returns:
            Initial agent state
        """
        device = next(self.parameters()).device

        belief = self.belief_tracker.init_belief(batch_size, device)

        self._state = AgentState(
            belief=belief,
            last_action=None,
            last_state=None,
            step_count=0,
            episode_reward=0.0,
        )

        return self._state

    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        force_slow: bool = False,
    ) -> AgentOutput:
        """
        Select action given observation.

        Args:
            observation: [B, H, W] grid observation
            deterministic: if True, use argmax actions
            force_slow: if True, always use slow policy

        Returns:
            AgentOutput with action and metadata
        """
        if self._state is None:
            self.reset(observation.shape[0])

        device = observation.device
        B = observation.shape[0]

        # 1. Encode observation
        state = self.encoder(observation)

        # 2. Update belief
        if self._state.last_action is not None:
            action_onehot = torch.zeros(B, self.config.fast_policy.num_actions, device=device)
            action_onehot.scatter_(1, self._state.last_action.unsqueeze(-1), 1)
            self._state.belief = self.belief_tracker.update(
                action_onehot, state, self._state.belief
            )

        # 3. Get world model uncertainty (for arbiter)
        if self._state.last_state is not None and self._state.last_action is not None:
            action_onehot = torch.zeros(B, self.config.fast_policy.num_actions, device=device)
            action_onehot.scatter_(1, self._state.last_action.unsqueeze(-1), 1)
            wm_uncertainty = self.world_model.get_uncertainty(
                self._state.last_state, action_onehot
            )
        else:
            wm_uncertainty = torch.zeros(B, device=device)

        # 4. Get fast policy output
        fast_action, fast_output = self.fast_policy.get_action(
            state, deterministic=deterministic
        )

        # 5. Arbiter decision
        arbiter_decision = self.arbiter(
            fast_output, wm_uncertainty, force_slow=force_slow
        )

        # 6. Get slow policy output if needed
        slow_output = None
        if arbiter_decision.use_slow.any() or force_slow:
            # Get goal from LLM
            goal = self._get_goal_embedding(observation, device)

            # Get slow policy action
            belief_embedding = self.belief_tracker.get_belief_embedding(self._state.belief)
            slow_action, slow_output = self.slow_policy.get_action(
                state, belief_embedding, goal, deterministic=deterministic
            )

        # 7. Select final action
        if slow_output is not None:
            action_probs, action = self.arbiter.select_action(
                fast_output, slow_output, arbiter_decision.use_slow
            )
            system_used = "mixed" if arbiter_decision.use_slow.any() and not arbiter_decision.use_slow.all() else (
                "slow" if arbiter_decision.use_slow.all() else "fast"
            )
        else:
            action = fast_action
            action_probs = fast_output.action_probs
            system_used = "fast"

        # 8. Update state
        self._state.last_action = action
        self._state.last_state = state
        self._state.step_count += 1

        return AgentOutput(
            action=action,
            action_probs=action_probs,
            system_used=system_used,
            fast_output=fast_output,
            slow_output=slow_output,
            arbiter_decision=arbiter_decision,
            state=state,
            world_model_uncertainty=wm_uncertainty,
            metadata={
                "step": self._state.step_count,
                "confidence": fast_output.confidence.mean().item(),
                "uncertainty": wm_uncertainty.mean().item(),
            },
        )

    def _get_goal_embedding(
        self,
        observation: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Get goal embedding from LLM."""
        B = observation.shape[0]

        # Use first observation in batch for goal (assuming same task)
        response = self.llm.generate_goal_hypothesis(observation[0])

        if response.goal_embedding is not None:
            goal = response.goal_embedding.to(device)
            goal = goal.unsqueeze(0).expand(B, -1)
        else:
            goal = torch.zeros(B, 64, device=device)

        return goal

    def update_world_model(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> WorldModelOutput:
        """
        Update world model with transition.

        Args:
            state: [B, state_dim]
            action: [B] action indices
            next_state: [B, state_dim]
            reward: [B]
            done: [B]

        Returns:
            WorldModelOutput with predictions
        """
        B = state.shape[0]
        device = state.device

        # Convert action to one-hot
        action_onehot = torch.zeros(B, self.config.fast_policy.num_actions, device=device)
        action_onehot.scatter_(1, action.unsqueeze(-1), 1)

        # Get world model prediction
        output = self.world_model(state, action_onehot)

        return output

    def imagine_trajectory(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine trajectory using world model.

        Args:
            state: [B, state_dim] starting state
            actions: [B, T] action indices

        Returns:
            states: [B, T+1, state_dim]
            rewards: [B, T]
            dones: [B, T]
            uncertainties: [B, T]
        """
        B, T = actions.shape
        device = state.device

        # Convert actions to one-hot: [B, T, num_actions]
        actions_onehot = torch.zeros(B, T, self.config.fast_policy.num_actions, device=device)
        actions_onehot.scatter_(2, actions.unsqueeze(-1), 1)

        # Predict trajectory
        states, rewards, dones, uncertainties = self.world_model.predict_trajectory(
            state, actions_onehot
        )

        return states, rewards.squeeze(-1), dones.squeeze(-1), uncertainties

    def get_value(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimate for observation.

        Args:
            observation: [B, H, W] grid observation

        Returns:
            value: [B] value estimates
        """
        state = self.encoder(observation)

        if self._state is None:
            belief = self.belief_tracker.init_belief(
                observation.shape[0], observation.device
            )
        else:
            belief = self._state.belief

        belief_embedding = self.belief_tracker.get_belief_embedding(belief)

        # Get value from slow policy
        output = self.slow_policy(state, belief_embedding)
        return output.value

    def get_stats(self) -> dict:
        """Get agent statistics."""
        arbiter_stats = self.arbiter.get_stats()
        llm_stats = self.llm.get_cache_stats()

        return {
            "arbiter": arbiter_stats,
            "llm_cache": llm_stats,
            "step_count": self._state.step_count if self._state else 0,
        }

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        return {
            "encoder": self.encoder.count_parameters(),
            "world_model": self.world_model.count_parameters(),
            "belief_tracker": self.belief_tracker.count_parameters(),
            "fast_policy": self.fast_policy.count_parameters(),
            "slow_policy": self.slow_policy.count_parameters(),
            "arbiter": self.arbiter.count_parameters(),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def create_agent(config: Optional[ARIALiteConfig] = None) -> ARIALiteAgent:
    """Factory function to create ARIA-Lite agent."""
    return ARIALiteAgent(config)
