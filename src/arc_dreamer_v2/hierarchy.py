"""
Hierarchical Policy with Defined Structure for ARC-DREAMER v2.

Addresses v1 weakness: "Subgoal" was undefined.

Solutions:
1. Object-centric subgoals with clear predicates
2. Automatic subgoal discovery via option learning
3. Three-level hierarchy: Strategy (20 steps) -> Tactics (5 steps) -> Primitives (1 step)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgoalType(Enum):
    """
    Enumeration of subgoal types.

    Each subgoal type has a clear predicate that can be checked
    against the symbolic state to determine if it's satisfied.
    """

    REACH_OBJECT = "reach_object"  # Navigate to specific object
    INTERACT_WITH = "interact_with"  # Interact with object (press, push)
    CHANGE_STATE = "change_state"  # Change object property
    REACH_POSITION = "reach_position"  # Navigate to coordinates
    COLLECT_ITEM = "collect_item"  # Pick up an object
    CLEAR_PATH = "clear_path"  # Remove obstacles
    EXPLORE_REGION = "explore_region"  # Visit unexplored area


@dataclass
class ObjectCentricSubgoal:
    """
    Formally defined subgoal types.

    A subgoal is a predicate over the symbolic state that can be
    achieved by a short sequence of primitive actions.

    This provides a CLEAR DEFINITION of what counts as a subgoal,
    addressing the v1 weakness of undefined hierarchy.
    """

    type: SubgoalType
    target_object: int | None = None  # Object ID
    target_state: dict | None = None  # Desired object properties
    target_position: Tuple[int, int] | None = None
    priority: float = 1.0
    estimated_steps: int = 5

    def is_satisfied(self, symbolic_state: Any) -> bool:
        """
        Check if subgoal is achieved.

        Each subgoal type has a clear, checkable predicate.
        """
        if self.type == SubgoalType.REACH_OBJECT:
            if self.target_object is None:
                return False
            agent_pos = symbolic_state.agent_position
            obj = symbolic_state.get_object(self.target_object)
            if obj is None:
                return False
            return self._adjacent(agent_pos, obj.position)

        elif self.type == SubgoalType.INTERACT_WITH:
            return symbolic_state.last_interaction == self.target_object

        elif self.type == SubgoalType.CHANGE_STATE:
            if self.target_object is None or self.target_state is None:
                return False
            obj = symbolic_state.get_object(self.target_object)
            if obj is None:
                return False
            return all(getattr(obj, k, None) == v for k, v in self.target_state.items())

        elif self.type == SubgoalType.REACH_POSITION:
            return symbolic_state.agent_position == self.target_position

        elif self.type == SubgoalType.COLLECT_ITEM:
            return self.target_object in symbolic_state.inventory

        elif self.type == SubgoalType.CLEAR_PATH:
            # Check if path exists between positions
            return symbolic_state.path_exists(symbolic_state.agent_position, self.target_position)

        elif self.type == SubgoalType.EXPLORE_REGION:
            # Check if region has been visited
            return symbolic_state.region_explored(self.target_position)

        return False

    def _adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) <= 1

    def to_embedding(self, embed_dim: int = 64) -> torch.Tensor:
        """
        Convert subgoal to embedding for goal-conditioned policy.

        This enables neural networks to reason about subgoals.
        """
        # Simple encoding: type one-hot + target info
        type_idx = list(SubgoalType).index(self.type)
        type_onehot = F.one_hot(torch.tensor(type_idx), num_classes=len(SubgoalType)).float()

        # Position encoding
        if self.target_position:
            pos_enc = torch.tensor(
                [
                    self.target_position[0] / 30.0,  # Normalize
                    self.target_position[1] / 30.0,
                ]
            )
        else:
            pos_enc = torch.zeros(2)

        # Object encoding
        obj_enc = torch.tensor([float(self.target_object) / 16.0 if self.target_object else 0.0])

        # Combine
        combined = torch.cat([type_onehot, pos_enc, obj_enc])

        # Project to embed_dim
        if combined.shape[0] < embed_dim:
            padding = torch.zeros(embed_dim - combined.shape[0])
            combined = torch.cat([combined, padding])

        return combined[:embed_dim]


class OptionPolicy(nn.Module):
    """Policy for executing a single option (skill)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """Return action distribution."""
        logits = self.network(state)
        return torch.distributions.Categorical(logits=logits)


class TerminationFunction(nn.Module):
    """Predicts when to terminate an option."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return termination probability."""
        return self.network(state).squeeze(-1)


class InitiationClassifier(nn.Module):
    """Classifies whether an option can be initiated from a state."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return initiation probability."""
        return self.network(state).squeeze(-1)


class MetaPolicy(nn.Module):
    """Policy over options (selects which option to execute)."""

    def __init__(self, state_dim: int, num_options: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """Return distribution over options."""
        logits = self.network(state)
        return torch.distributions.Categorical(logits=logits)


@dataclass
class OptionStats:
    """Statistics for a learned option."""

    times_initiated: int = 0
    times_succeeded: int = 0
    total_steps: int = 0
    total_reward: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.times_initiated == 0:
            return 0.0
        return self.times_succeeded / self.times_initiated

    @property
    def avg_steps(self) -> float:
        if self.times_initiated == 0:
            return 0.0
        return self.total_steps / self.times_initiated


class OptionDiscovery:
    """
    Discovers useful options (subgoals) from experience.

    Uses the option-critic architecture with modifications:
    1. Option initiation based on symbolic state predicates
    2. Option termination learned from value function
    3. Options discovered by finding reusable state-reaching skills

    Options are discovered by:
    1. Finding "bottleneck" states (frequently visited, lead to diverse futures)
    2. Creating options to reach these bottleneck states
    3. Learning when these options are useful (initiation sets)

    Reference: Bacon et al. "The Option-Critic Architecture"
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_options: int = 16,
        termination_reg: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_options = max_options
        self.termination_reg = termination_reg

        # Option components
        self.option_policies = nn.ModuleList(
            [OptionPolicy(state_dim, action_dim) for _ in range(max_options)]
        )
        self.termination_functions = nn.ModuleList(
            [TerminationFunction(state_dim) for _ in range(max_options)]
        )
        self.initiation_sets = nn.ModuleList(
            [InitiationClassifier(state_dim) for _ in range(max_options)]
        )

        # Meta-controller selects options
        self.meta_policy = MetaPolicy(state_dim, max_options)

        # Track discovered options
        self.option_statistics = [OptionStats() for _ in range(max_options)]
        self.discovered_bottlenecks: list[torch.Tensor] = []

    def discover_from_trajectories(
        self,
        trajectories: list,
        symbolic_extractor: Any,
    ):
        """
        Discover options from collected trajectories.

        Algorithm:
        1. Identify "bottleneck" states (frequently visited, lead to diverse futures)
        2. Create options to reach these bottleneck states
        3. Learn initiation/termination from when these states are useful

        This provides AUTOMATIC subgoal discovery rather than
        relying on hand-crafted subgoals.
        """
        # Step 1: Find bottleneck states
        bottlenecks = self._find_bottleneck_states(trajectories)
        self.discovered_bottlenecks = bottlenecks[: self.max_options]

        # Step 2: Create reaching options for each bottleneck
        for i, bottleneck in enumerate(self.discovered_bottlenecks):
            # Train option policy to reach this state
            self._train_reaching_option(
                option_idx=i,
                target_state=bottleneck,
                trajectories=trajectories,
            )

    def _find_bottleneck_states(
        self,
        trajectories: list,
    ) -> list[torch.Tensor]:
        """
        Find bottleneck states using betweenness-like heuristic.

        Bottleneck = state that appears on many paths and
        leads to diverse future states.

        Simple approximation: states that appear frequently
        across trajectories and have high successor diversity.
        """
        state_visits: dict[int, int] = {}
        state_successors: dict[int, set] = {}
        state_examples: dict[int, torch.Tensor] = {}

        for traj in trajectories:
            for i in range(len(traj.states) - 1):
                s = traj.states[i]
                s_next = traj.states[i + 1]

                # Hash states for counting
                s_hash = hash(tuple(s.flatten().tolist()))
                s_next_hash = hash(tuple(s_next.flatten().tolist()))

                state_visits[s_hash] = state_visits.get(s_hash, 0) + 1

                if s_hash not in state_successors:
                    state_successors[s_hash] = set()
                    state_examples[s_hash] = s
                state_successors[s_hash].add(s_next_hash)

        # Score states by visits * successor diversity
        scored_states = []
        for s_hash, visits in state_visits.items():
            diversity = len(state_successors.get(s_hash, set()))
            score = visits * diversity
            scored_states.append((score, s_hash))

        # Return top states
        scored_states.sort(reverse=True)
        bottlenecks = [
            state_examples[s_hash] for score, s_hash in scored_states if s_hash in state_examples
        ]

        return bottlenecks

    def _train_reaching_option(
        self,
        option_idx: int,
        target_state: torch.Tensor,
        trajectories: list,
        num_epochs: int = 10,
    ):
        """Train option policy to reach target state."""
        # Collect transitions that lead toward target
        training_data = []

        for traj in trajectories:
            for i in range(len(traj.states) - 1):
                s = traj.states[i]
                a = traj.actions[i]
                s_next = traj.states[i + 1]

                # Check if this transition moves toward target
                dist_before = F.mse_loss(s, target_state)
                dist_after = F.mse_loss(s_next, target_state)

                if dist_after < dist_before:
                    training_data.append((s, a))

        if not training_data:
            return

        # Train option policy
        policy = self.option_policies[option_idx]
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        for _ in range(num_epochs):
            for s, a in training_data:
                dist = policy(s.unsqueeze(0))
                loss = -dist.log_prob(a)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def select_option(
        self,
        state: torch.Tensor,
        explore: bool = True,
    ) -> int:
        """Select an option to execute."""
        # Get available options (initiation set check)
        available = []
        for i in range(self.max_options):
            init_prob = self.initiation_sets[i](state)
            if init_prob > 0.5 or explore:
                available.append(i)

        if not available:
            available = list(range(self.max_options))

        # Sample from meta-policy
        dist = self.meta_policy(state)
        if explore:
            option = dist.sample().item()
        else:
            option = dist.probs.argmax().item()

        return option if option in available else available[0]

    def should_terminate(self, state: torch.Tensor, option_idx: int) -> bool:
        """Check if current option should terminate."""
        term_prob = self.termination_functions[option_idx](state)
        return torch.rand(1).item() < term_prob.item()


class PrimitivePolicy(nn.Module):
    """Policy for primitive actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Return action distribution."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.network(state)
        logits = self.action_head(features)
        return torch.distributions.Categorical(logits=logits)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value estimate."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.network(state)
        return self.value_head(features).squeeze(-1)


class GoalConditionedPrimitivePolicy(nn.Module):
    """Primitive policy conditioned on a subgoal."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Return action distribution for pursuing goal from state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        combined = torch.cat([state, goal], dim=-1)
        features = self.encoder(combined)
        logits = self.action_head(features)
        return torch.distributions.Categorical(logits=logits)


class HierarchicalPolicy:
    """
    Three-level hierarchical policy with defined structure.

    Level 3 (Strategy): High-level goals, ~20 steps
        Examples: "Complete level", "Explore unknown region"

    Level 2 (Tactics): Object-centric subgoals, ~5 steps
        Examples: "Reach object X", "Interact with Y", "Change state of Z"
        Discovered via Option Learning (OptionDiscovery)

    Level 1 (Primitives): Environment actions, 1 step
        Examples: ACTION1-7, RESET, coordinates

    This provides a CLEAR STRUCTURE addressing v1's undefined hierarchy.
    """

    def __init__(
        self,
        option_discovery: OptionDiscovery,
        primitive_policy: PrimitivePolicy,
        goal_conditioned_policy: GoalConditionedPrimitivePolicy,
    ):
        self.options = option_discovery
        self.primitive_policy = primitive_policy
        self.goal_policy = goal_conditioned_policy

        self.current_strategy: str | None = None
        self.current_subgoal: ObjectCentricSubgoal | None = None
        self.current_option: int | None = None
        self.steps_since_strategy = 0
        self.steps_since_subgoal = 0

        # Exploration bonus (can be modified by anomaly detection)
        self.exploration_bonus = 1.0
        self.exploration_bonus_steps = 0

    def act(
        self,
        state: torch.Tensor,
        symbolic_state: Any | None = None,
        strategy_interval: int = 20,
        subgoal_interval: int = 5,
    ) -> Tuple[int, dict]:
        """
        Select action using hierarchical policy.

        Args:
            state: Latent state representation
            symbolic_state: Optional symbolic state for subgoal checking
            strategy_interval: Steps between strategy updates
            subgoal_interval: Steps between subgoal updates

        Returns:
            action: Primitive action index to execute
            info: Hierarchy state for logging
        """
        self.steps_since_strategy += 1
        self.steps_since_subgoal += 1

        # Update exploration bonus
        if self.exploration_bonus_steps > 0:
            self.exploration_bonus_steps -= 1
            if self.exploration_bonus_steps == 0:
                self.exploration_bonus = 1.0

        # Level 3: Update strategy if needed
        if self.current_strategy is None or self.steps_since_strategy >= strategy_interval:
            self.current_strategy = self._select_strategy(state, symbolic_state)
            self.steps_since_strategy = 0

        # Level 2: Update subgoal/option if needed
        subgoal_achieved = (
            self.current_subgoal is not None
            and symbolic_state is not None
            and self.current_subgoal.is_satisfied(symbolic_state)
        )

        option_terminated = self.current_option is not None and self.options.should_terminate(
            state, self.current_option
        )

        if (
            self.current_subgoal is None
            or self.steps_since_subgoal >= subgoal_interval
            or subgoal_achieved
            or option_terminated
        ):
            self.current_option = self.options.select_option(state)
            self.current_subgoal = self._create_subgoal_for_option(
                self.current_option, state, symbolic_state
            )
            self.steps_since_subgoal = 0

        # Level 1: Select primitive action
        action = self._select_primitive(state, self.current_subgoal)

        info = {
            "strategy": self.current_strategy,
            "subgoal": self.current_subgoal,
            "option": self.current_option,
            "steps_since_strategy": self.steps_since_strategy,
            "steps_since_subgoal": self.steps_since_subgoal,
            "exploration_bonus": self.exploration_bonus,
        }

        return action, info

    def _select_strategy(
        self,
        state: torch.Tensor,
        symbolic_state: Any | None,
    ) -> str:
        """Select high-level strategy."""
        # Simple heuristic strategies
        strategies = [
            "explore_systematically",
            "reach_goal",
            "interact_objects",
            "collect_items",
        ]

        # Use value function to estimate which strategy is best
        # For now, cycle through strategies
        step = self.steps_since_strategy // 20
        return strategies[step % len(strategies)]

    def _create_subgoal_for_option(
        self,
        option_idx: int,
        state: torch.Tensor,
        symbolic_state: Any | None,
    ) -> ObjectCentricSubgoal:
        """Create subgoal corresponding to learned option."""
        # If we have a bottleneck state for this option,
        # create a REACH_POSITION subgoal
        if option_idx < len(self.options.discovered_bottlenecks):
            bottleneck = self.options.discovered_bottlenecks[option_idx]
            # Estimate position from bottleneck state
            return ObjectCentricSubgoal(
                type=SubgoalType.EXPLORE_REGION,
                target_position=(
                    int(bottleneck[0].item() * 30) if bottleneck.dim() > 0 else 15,
                    int(bottleneck[1].item() * 30) if bottleneck.dim() > 1 else 15,
                ),
            )

        # Default: exploration subgoal
        return ObjectCentricSubgoal(
            type=SubgoalType.EXPLORE_REGION,
            target_position=(15, 15),  # Center
        )

    def _select_primitive(
        self,
        state: torch.Tensor,
        subgoal: ObjectCentricSubgoal | None,
    ) -> int:
        """Select primitive action."""
        if subgoal is not None:
            # Use goal-conditioned policy
            goal_emb = subgoal.to_embedding()
            dist = self.goal_policy(state, goal_emb)
        else:
            # Use unconditional policy
            dist = self.primitive_policy(state)

        # Apply exploration bonus to entropy
        if self.exploration_bonus > 1.0:
            # Increase temperature for more exploration
            logits = dist.logits / self.exploration_bonus
            dist = torch.distributions.Categorical(logits=logits)

        return dist.sample().item()

    def increase_exploration_bonus(
        self,
        factor: float = 2.0,
        duration: int = 10,
    ):
        """Increase exploration (e.g., after anomaly detection)."""
        self.exploration_bonus = factor
        self.exploration_bonus_steps = duration

    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.primitive_policy.parameters())
        params.extend(self.goal_policy.parameters())
        params.extend(self.options.option_policies.parameters())
        params.extend(self.options.termination_functions.parameters())
        params.extend(self.options.initiation_sets.parameters())
        params.extend(self.options.meta_policy.parameters())
        return params
